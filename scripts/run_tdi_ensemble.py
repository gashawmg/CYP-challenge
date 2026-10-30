#!/usr/bin/env python
"""TDI track, Track A treatment — a diverse classifier pool plus the unused isoforms.

    python scripts/run_tdi_ensemble.py --tag tdiM --save-oof       cross-validate only
    python scripts/run_tdi_ensemble.py --tag tdiM --predict --save-oof   + blind file

(--fast was removed; the line above used to name it and no longer matches argparse.)

Two changes from run_tdi.py, both structural rather than cosmetic.

DIVERSITY. The old script was one LightGBM classifier plus one structured rule score,
both on the same 267-column vector. Here the pool is:

    lgb_clf   LightGBM            rdkit + cyp + ecfp + maccs
    xgb_clf   XGBoost             rdkit + cyp
    cat_clf   CatBoost            rdkit + cyp + maccs
    et_clf    ExtraTrees          rdkit + cyp + maccs
    knn_tan   Tanimoto kNN vote   ecfp
    svc_tan   SVC, Tanimoto kernel ecfp
    struct    the recovered rule applied softly to two regressed arms
    pooled    ONE model over all four isoforms at once (see below)

Scores are rank-averaged before thresholding — the models are on different probability
scales and MCC only cares about the ordering.

THE UNUSED LABELS. `is_TDI` is scored on CYP3A4 and CYP2D6 only, but TRAIN_Emax carries
it for CYP1A2 and CYP2C9 too: 1,413 + 1,285 rows that no model in this project has ever
seen. Time-dependent inhibition is mechanism-based — it comes from the same reactive
metabolites and the same structural alerts regardless of which isoform bioactivates the
compound. The `pooled` learner trains on all four isoforms at once with a one-hot
isoform indicator, roughly doubling the label count that the CYP2D6 head (the
data-starved one, 1,497 rows) can borrow from.

WHERE THE TWO ENDPOINTS NOW STAND (from submission_tdi_blendcp, MA-MCC 0.3760;
rank 9 in Sept, re-scored identically at rank 24 on 14 Oct as the field moved).
Inverting every competitor's accuracy / precision / recall recovers their cut rate,
their d', and the blind prevalence — which came out to 0.2915 (CYP3A4) and 0.0706
(CYP2D6) on EVERY competitor, to four decimals.

    CYP3A4   us 1.290 | wbot 1.289 | tabicl 1.312 | jeremy 1.319     <- we are AT the top
    CYP2D6   us 0.980 | AL 0.664   | jeremy 1.392                    <- we are far behind

So the endpoints need opposite work. CYP3A4 is a solved model with a mis-placed cut:
wbot has our d' to three decimals, cut at 0.314 against our 0.229, and scored 0.4554
against our 0.4470. CYP2D6's cut is nearly right and the entire 0.12 MCC gap to the
leader is discriminability — blind AUC 0.756 against his 0.838.

AUC is therefore still what decides CYP2D6, and MCC is what AUC buys you. But the
threshold is no longer treated as housekeeping: see the rate table below.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit import cyp_descriptors as CD  # noqa: E402
from cypkit import ensemble as E  # noqa: E402
from cypkit import features, io, splits  # noqa: E402
from cypkit import tdi as T  # noqa: E402
from scripts.run_ensemble import build_bank  # noqa: E402

AUX_FEATURES = "rdkit+cyp+tdialert+ecfp+maccs"   # struct and pooled learners
TEST_PREVALENCE = 0.173  # reconstructed from the leaderboard accuracy; see phase 2

# Cut rates, MEASURED rather than guessed. `submission_tdi_best` gave us our own d' on the blind set
# (CYP3A4 1.290, CYP2D6 0.980) and confirmed the prevalence to four decimals on every
# competitor. The rates below are the argmax of MCC under a binormal ROC at those two
# numbers; that model reproduced both of our actual scores to within 0.0008.
#
# Correcting an earlier claim in this file: "MCC is flat to within 0.002 for any rate
# between 0.12 and 0.29" was fitted over too narrow a band. Across the range competitors
# actually use, cutting at the optimum is worth +0.0087 on CYP3A4 and +0.0046 on CYP2D6.
# Small, but free -- and `wbot` proves the CYP3A4 half empirically: identical d' (1.289
# vs our 1.290), cut at 0.314 instead of 0.229, scored 0.4554 against our 0.4470.
#
# These are OUR rates. A better model moves the optimum -- rerun scripts/tdi_splice.py.
TEST_PREVALENCE_BY_ISO = {"CYP3A4": 0.2915, "CYP2D6": 0.0706}
OPTIMAL_RATE = {"CYP3A4": 0.306, "CYP2D6": 0.142}

# EFFECTIVE vs SUBMITTED RATE -- these are not the same number, and conflating them
# cost a submission's worth of threshold gain once already.
#
# OPTIMAL_RATE above is the rate the LEADERBOARD sees. The leaderboard scores a SUBSET
# of the 750 blind compounds (only those with TDI ground truth for that isoform), and
# our predicted positives are enriched in that subset. Measured on submission_tdi_best:
#
#     isoform   we submitted   leaderboard saw   inflation
#     CYP3A4       0.2107          0.2290          1.087
#     CYP2D6       0.1707          0.2136          1.252
#
# So cutting the file at OPTIMAL_RATE overshoots: 0.306 submitted lands at ~0.333
# effective, past the peak. Divide by the inflation factor to hit the target.
# Refit these whenever a submission's accuracy/precision/recall are published --
# scripts/tdi_splice.py does the inversion.
RATE_INFLATION = {"CYP3A4": 0.2290 / 0.2107, "CYP2D6": 0.2136 / 0.1707}


def submitted_rate(cyp, effective):
    """Convert a target EFFECTIVE positive rate into the rate to cut the file at."""
    return effective / RATE_INFLATION.get(cyp, 1.0)

POOL = {
    "lgb_clf": dict(kind="lgb", features="rdkit+cyp+tdialert+ecfp+maccs"),
    "xgb_clf": dict(kind="xgb", features="rdkit+cyp+tdialert"),
    "cat_clf": dict(kind="cat", features="rdkit+cyp+tdialert+maccs"),
    "et_clf": dict(kind="et", features="rdkit+cyp+tdialert+maccs"),
    "knn_tan": dict(kind="knn", features="ecfp"),
    "svc_tan": dict(kind="svc", features="ecfp"),
    "lgb_alert": dict(kind="lgb", features="cyp+tdialert"),
    # ---- primary-screen transfer ---------------------------------------
    # The single-concentration screen covers 4,376 compounds against 1,497 CYP2D6 TDI
    # labels -- three times the labelled set, and untouched until now. It cannot be used
    # as a raw column (no blind-test compound appears in it), so `screen` is the
    # out-of-fold prediction of log2fc and P(hit) per isoform. See cypkit/screen_prior.py.
    "lgb_screen": dict(kind="lgb", features="rdkit+cyp+tdialert+screen"),
    "cat_screen": dict(kind="cat", features="cyp+tdialert+screen+maccs"),
    # ---- external transfer ---------------------------------------------
    # cypkit/external_prior.py turns the Veith/NCGC qHTS panel (PubChem AID 1851,
    # 16,145 compounds in artifacts/external_cyp.csv) into ten columns: a predicted
    # pIC50 and P(active) for each of CYP1A2, 2C9, 2C19, 2D6 and 3A4, from models that
    # never see a challenge label. It was built in September, measured on the pIC50
    # track (mean Spearman 0.495 -> 0.507, and 0.542 -> 0.565 on fitted-curve rows),
    # wired into build_bank as the `prior` block -- and no TDI member has ever used it.
    #
    # Direct overlap is too small to use as rows: 198 of 6,145 TDI compounds appear in
    # the qHTS panel by InChIKey skeleton, 2.3% of the CYP3A4 labelled set and 6.2% of
    # CYP2D6. On that handful the external CYP3A4 pIC50 scores AUC 0.788 against
    # CYP3A4 is_TDI (81 compounds, 8 positives -- suggestive, not evidence), while every
    # external column sits at or below chance against CYP2D6 is_TDI. So this is expected
    # to be a CYP3A4 member, which is the endpoint whose column is frozen; its value
    # there is diversity for the blend, not a new best single model.
    "lgb_prior": dict(kind="lgb", features="rdkit+cyp+tdialert+prior"),
    "cat_prior": dict(kind="cat", features="cyp+tdialert+prior+maccs"),
    # ---- similarity-kernel variants ------------------------------------
    # The fitted CYP2D6 weights put 62% on knn_tan + svc_tan, and that blend gained
    # +0.036 MCC on the blind set (0.1825 -> 0.2180) where every tree-based idea has
    # failed. Two separate weight fits, on different machines, agreed. But we had only
    # ever tried ONE configuration of each: ECFP4, 2048 bits, k=15, C=3.
    #
    # These vary the two axes that matter for a Tanimoto kernel -- which fingerprint
    # defines "similar", and how many neighbours vote. Each costs seconds to fit.
    # They are deliberately NOT a hyperparameter search: every one is a different
    # notion of chemical similarity, so they are new members for the blend rather than
    # tunings of an old one.
    "knn_fcfp": dict(kind="knn", features="fcfp"),          # pharmacophoric, not topological
    "svc_fcfp": dict(kind="svc", features="fcfp"),
    "knn_ap": dict(kind="knn", features="ap"),              # atom pairs: through-bond distance
    "svc_ap": dict(kind="svc", features="ap"),
    "knn_maccs": dict(kind="knn", features="maccs"),        # 167 curated keys
    "knn_ecfpc": dict(kind="knn", features="ecfpc"),        # counts, not presence/absence
    "knn_k5": dict(kind="knn", features="ecfp", opts=dict(k=5)),    # sharper, local
    "knn_k40": dict(kind="knn", features="ecfp", opts=dict(k=40)),  # smoother, global
    "svc_c10": dict(kind="svc", features="ecfp", opts=dict(C=10.0)),
    # Avalon was built, documented and benchmarked in cypkit/features.py --
    # "a path/feature hybrid from a DIFFERENT enumeration than Morgan, so it
    # tends to disagree with ecfp rather than duplicate it" -- and then never
    # reached this pool. train_tdi_chemprop.py measured it as CYP2D6's best
    # single fingerprint (AUC 0.6043 against ECFP's) and made it that
    # endpoint's ONLY block. Disagreement with ecfp is precisely what a rank
    # blend is paid for, so it belongs here as two members, not none.
    "knn_avalon": dict(kind="knn", features="avalon"),
    "svc_avalon": dict(kind="svc", features="avalon"),
    # ---- disagreement, not accuracy ------------------------------------
    # Section 54's addendum measured what Avalon actually bought: svc_avalon scores
    # BELOW knn_fcfp on CYP2D6 (AUC 0.6116 against 0.6283) and still gains +0.050 MCC,
    # because its rank correlation with knn_fcfp is 0.369 where the other similarity
    # members sit at 0.503. A rank blend converts disagreement into score and cannot
    # convert anything else. So these five are chosen as different ENUMERATIONS of
    # substructure rather than as better models, and the number to read on each is its
    # correlation with knn_fcfp, not its AUC.
    #   rdkpath   linear paths up to 7 bonds -- no circular environments at all
    #   tt        topological torsions: 4-atom paths with atom types
    #   pattern   substructure-screening keys, built for subgraph matching
    #   layered   the same paths hashed under several atom/bond abstractions at once
    #   avalon1k  Avalon at the 1024 bits train_tdi_chemprop.py benchmarked at 0.6043;
    #             cypkit/features.py builds "avalon" at 2048, so this is a real variant
    "knn_rdkpath": dict(kind="knn", features="rdkpath"),
    "svc_rdkpath": dict(kind="svc", features="rdkpath"),
    "knn_tt": dict(kind="knn", features="tt"),
    "svc_tt": dict(kind="svc", features="tt"),
    "knn_pattern": dict(kind="knn", features="pattern"),
    "svc_pattern": dict(kind="svc", features="pattern"),
    "knn_layered": dict(kind="knn", features="layered"),
    "svc_layered": dict(kind="svc", features="layered"),
    "knn_avalon1k": dict(kind="knn", features="avalon1k"),
    "svc_avalon1k": dict(kind="svc", features="avalon1k"),
}

LGB_C = dict(objective="binary", verbose=-1, learning_rate=0.05, num_leaves=31,
             min_data_in_leaf=20, feature_fraction=0.6, bagging_fraction=0.8,
             bagging_freq=1, seed=42)
LGB_R = dict(objective="regression_l1", verbose=-1, learning_rate=0.05, num_leaves=63,
             min_data_in_leaf=10, feature_fraction=0.6, bagging_fraction=0.8,
             bagging_freq=1, seed=42)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def clf_predict(kind, Xtr, y, Xte, n_rounds, w=None, opts=None):
    opts = opts or {}
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return np.full(Xte.shape[0], float(y.mean()))
    if kind == "lgb":
        import lightgbm as lgb  # noqa: PLC0415
        return lgb.train(LGB_C, lgb.Dataset(Xtr, label=y, weight=w),
                         num_boost_round=n_rounds).predict(Xte)
    if kind == "xgb":
        import xgboost as xgb  # noqa: PLC0415
        p = dict(objective="binary:logistic", eta=0.05, max_depth=6, subsample=0.8,
                 colsample_bytree=0.5, min_child_weight=5, tree_method="hist",
                 eval_metric="logloss")
        m = xgb.train(p, xgb.DMatrix(Xtr, label=y, weight=w), num_boost_round=n_rounds)
        return m.predict(xgb.DMatrix(Xte))
    if kind == "cat":
        from catboost import CatBoostClassifier  # noqa: PLC0415
        m = CatBoostClassifier(iterations=n_rounds * 2, learning_rate=0.06, depth=7,
                               l2_leaf_reg=3.0, random_seed=13, verbose=0,
                               allow_writing_files=False)
        m.fit(Xtr, y, sample_weight=w)
        return m.predict_proba(Xte)[:, 1]
    if kind == "et":
        from sklearn.ensemble import ExtraTreesClassifier  # noqa: PLC0415
        m = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                                 max_features=0.3, n_jobs=-1, random_state=21)
        m.fit(Xtr, y, sample_weight=w)
        return m.predict_proba(Xte)[:, 1]
    if kind == "svc":
        from sklearn.svm import SVC  # noqa: PLC0415
        K = E.tanimoto(Xtr, Xtr).astype(np.float64)
        m = SVC(kernel="precomputed", C=float(opts.get("C", 3.0)), probability=False)
        m.fit(K, y, sample_weight=w)
        d = m.decision_function(E.tanimoto(Xte, Xtr).astype(np.float64))
        return 1.0 / (1.0 + np.exp(-d))
    if kind == "knn":
        out = np.empty(Xte.shape[0])
        for s in range(0, Xte.shape[0], 512):
            S = E.tanimoto(Xte[s:s + 512], Xtr)
            k = min(int(opts.get("k", 15)), S.shape[1])
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            sims = np.take_along_axis(S, idx, axis=1) ** float(opts.get("power", 3.0))
            tot = sims.sum(1)
            out[s:s + 512] = np.where(tot > 0, (sims * y[idx]).sum(1) / np.maximum(tot, 1e-12),
                                      y.mean())
        return out
    raise ValueError(kind)


def structured(Xtr, sub_tr, Xte, n_rounds):
    """Regress both arms and measurability, then apply the recovered rule softly."""
    import lightgbm as lgb  # noqa: PLC0415

    p_meas = clf_predict("lgb", Xtr, sub_tr["measurable"].to_numpy().astype(int),
                         Xte, n_rounds)
    d = sub_tr["direct"].to_numpy(dtype=float)
    t = sub_tr["tdi_arm"].to_numpy(dtype=float)
    ok_d, ok_t = np.isfinite(d), np.isfinite(t)
    if ok_d.sum() < 20 or ok_t.sum() < 20:
        return None
    pd_ = lgb.train(LGB_R, lgb.Dataset(Xtr[ok_d], label=d[ok_d]),
                    num_boost_round=n_rounds).predict(Xte)
    pt_ = lgb.train(LGB_R, lgb.Dataset(Xtr[ok_t], label=t[ok_t]),
                    num_boost_round=n_rounds).predict(Xte)
    return T.structured_score(p_meas, pd_, pt_)


# ---------------------------------------------------------------------------


def all_isoform_table(tdi_raw, emax):
    """is_TDI for all four isoforms, preferring the Emax file which carries them all."""
    rows = []
    for cyp in C.ISOFORMS:
        col = f"{cyp}_is_TDI"
        src = emax if (emax is not None and col in emax.columns) else tdi_raw
        if col not in src.columns:
            continue
        sub = src[src[col].notna()]
        rows.append(pd.DataFrame({
            C.ID_COL: sub[C.ID_COL].to_numpy(),
            "isoform": cyp,
            "y": sub[col].astype(bool).to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def rank(x):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return out
    r = pd.Series(x[ok]).rank(pct=True).to_numpy()
    out[ok] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learners", nargs="*", default=list(POOL))
    ap.add_argument("--n-rounds", type=int, default=300)
    ap.add_argument("--no-pooled", action="store_true")
    ap.add_argument("--no-struct", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--tag", default="tdiE")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--drop-unmeasurable", action="store_true",
                    help="exclude compounds whose direct-inhibition pIC50 is pinned at "
                         "the assay floor from TRAINING; their labels were assigned "
                         "against a baseline that was never measured and score at "
                         "chance. They are still scored. PREFER --floor-weight: it "
                         "buys the same AUC without discarding rows the blind set "
                         "almost certainly contains too.")
    ap.add_argument("--floor-weight", type=float, default=None,
                    help="train on the floor-pinned compounds at this weight instead "
                         "of dropping them (0.3 is the measured optimum). Keeps them "
                         "in the model's support while stopping their noisy labels "
                         "from steering the boundary.")
    ap.add_argument("--floor-tol", type=float, default=0.2,
                    help="how close to the observed minimum counts as pinned")
    ap.add_argument("--extra-oof", default=None,
                    help="CSV of out-of-fold scores from a model trained OUTSIDE this "
                         "pool (Molecule_Name + <ISO>_is_TDI_prob), added to the "
                         "rank blend as another member")
    ap.add_argument("--extra-test", default=None,
                    help="the same model's blind-set scores, same columns")
    ap.add_argument("--extra-name", default="chemprop")
    ap.add_argument("--extra-weight", type=float, default=1.0,
                    help="how many members' worth of weight the extra model carries "
                         "in the rank blend; 1.0 = the same as any pool member")
    ap.add_argument("--save-oof", action="store_true",
                    help="persist every member's OOF scores so a blend can be refitted "
                         "later without re-running the pool")
    args = ap.parse_args()

    from sklearn.metrics import matthews_corrcoef  # noqa: PLC0415

    tdi_raw = io.load("train_tdi", args.data_dir)
    try:
        emax = io.load("train_emax", args.data_dir)
    except Exception:  # noqa: BLE001
        emax = None
    test = io.load("test", args.data_dir)

    print(T.verify_rule(tdi_raw).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    table = T.assemble(tdi_raw)
    aux = all_isoform_table(tdi_raw, emax)
    print("\nis_TDI labels available per isoform (scored isoforms marked *):")
    for cyp in C.ISOFORMS:
        n = int((aux["isoform"] == cyp).sum())
        print(f"  {cyp}{'*' if cyp in C.TDI_ISOFORMS else ' '} {n:5d}")

    folds = splits.load_folds(C.artifact("folds.csv"))
    n_folds = int(folds["fold"].max()) + 1
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))

    smiles_by_id = dict(zip(table[C.ID_COL], table[C.SMILES_COL]))
    smiles_by_id.update(dict(zip(tdi_raw[C.ID_COL], tdi_raw[C.SMILES_COL])))
    smiles_by_id.update(dict(zip(test[C.ID_COL], test[C.SMILES_COL])))
    smiles_by_id = {k: v for k, v in smiles_by_id.items() if isinstance(v, str)}
    extra = sorted(set(smiles_by_id) - set(fold_of) - set(test[C.ID_COL]))
    if extra:
        fold_of.update(dict(zip(extra, splits.assign_extra_to_folds(
            [smiles_by_id[i] for i in extra], folds))))

    want = {b for n in args.learners for b in POOL[n]["features"].split("+")}
    if not (args.no_struct and args.no_pooled):
        want |= set(AUX_FEATURES.split("+"))
    order = ["rdkit", "cyp", "tdialert", "screen", "prior", "ecfp", "ecfpc", "fcfp",
             "maccs", "ap", "avalon", "avalon1k", "rdkpath", "tt", "pattern", "layered",
             "3d"]
    blocks = [b for b in order if b in want]
    print(f"\nfeaturising {len(smiles_by_id)} compounds, blocks = {'+'.join(blocks)}")
    bank = build_bank(smiles_by_id, blocks)

    aux = aux[aux[C.ID_COL].isin(bank.index_of)]
    iso_onehot = {c: np.eye(len(C.ISOFORMS))[j] for j, c in enumerate(C.ISOFORMS)}

    # ---- out-of-fold ------------------------------------------------------
    scores: dict[str, dict[str, np.ndarray]] = {}
    per_iso = {}
    # Compounds whose DIRECT-inhibition pIC50 is pinned at the very bottom of the
    # assay range. The organisers' rule says a compound below the reliable detection
    # threshold (pIC50 <= 4.0) is called TDI only if the pre-incubation arm jumps to
    # >= 4.301, so for these the label was assigned against a baseline that was never
    # measured. Scored on their own they come out at AUC 0.4939 -- chance -- while
    # every other slice of CYP2D6 sits at 0.62-0.64. Dropping the 45 of them from
    # TRAINING (they are still scored) moved CYP2D6 OOF AUC 0.6008 -> 0.6106.
    drop_ids = set()
    if args.drop_unmeasurable or args.floor_weight is not None:
        raw = io.load("train_tdi", args.data_dir)
        raw.columns = [c.strip() for c in raw.columns]
        for cyp in C.TDI_ISOFORMS:
            col = f"{cyp}_pIC50_direct_inhibition"
            if col not in raw.columns:
                continue
            v = raw[col]
            floor = float(v.min())
            hit = raw.loc[(v <= floor + args.floor_tol), C.ID_COL]
            drop_ids |= set(hit)
        if args.drop_unmeasurable:
            print(f"  dropping {len(drop_ids)} compounds pinned at the direct-inhibition "
                  f"floor from TRAINING (still scored)")
        else:
            print(f"  down-weighting {len(drop_ids)} floor-pinned compounds to "
                  f"{args.floor_weight} in TRAINING (kept, and still scored)")

    for cyp in C.TDI_ISOFORMS:
        sub = table[(table["isoform"] == cyp) & table[C.ID_COL].isin(bank.index_of)].copy()
        sub["fold"] = sub[C.ID_COL].map(fold_of)
        pinned = sub[C.ID_COL].isin(drop_ids)
        # Two knobs, one set of rows. `--drop-unmeasurable` removes them; `--floor-weight`
        # keeps them at reduced influence, which is the better default: the blind set was
        # drawn from the same library and will contain compounds whose direct arm is also
        # below detection, so a model that has never seen one has no support there.
        sub["trainable"] = ~pinned if args.drop_unmeasurable else True
        sub["w"] = np.where(pinned & (args.floor_weight is not None),
                            args.floor_weight if args.floor_weight is not None else 1.0,
                            1.0)
        per_iso[cyp] = sub.reset_index(drop=True)

    for name in args.learners:
        kind, feat = POOL[name]["kind"], POOL[name]["features"]
        X = bank.view(feat)
        t0 = time.time()
        scores[name] = {}
        for cyp in C.TDI_ISOFORMS:
            sub = per_iso[cyp]
            idx = sub[C.ID_COL].map(bank.index_of).to_numpy()
            s = np.full(len(sub), np.nan)
            trainable = sub["trainable"].to_numpy()
            wcol = sub["w"].to_numpy(float)
            for f in range(n_folds):
                te = (sub["fold"] == f).to_numpy()
                tr = ~te & trainable          # held-out rows are still SCORED
                if te.sum() == 0 or tr.sum() < 20:
                    continue
                s[te] = clf_predict(kind, X[idx[tr]], sub["y"].to_numpy()[tr],
                                    X[idx[te]], args.n_rounds,
                                    w=wcol[tr] if (wcol[tr] != 1).any() else None,
                                    opts=POOL[name].get("opts"))
            scores[name][cyp] = s
        print(f"  {name:8s} [{feat}] ({time.time()-t0:.0f}s)")

    if not args.no_struct:
        X = bank.view(AUX_FEATURES)
        t0 = time.time()
        scores["struct"] = {}
        for cyp in C.TDI_ISOFORMS:
            sub = per_iso[cyp]
            idx = sub[C.ID_COL].map(bank.index_of).to_numpy()
            s = np.full(len(sub), np.nan)
            for f in range(n_folds):
                te = (sub["fold"] == f).to_numpy()
                tr = ~te
                if te.sum() == 0 or tr.sum() < 20:
                    continue
                out = structured(X[idx[tr]], sub[tr], X[idx[te]], args.n_rounds)
                if out is not None:
                    s[te] = out
            scores["struct"][cyp] = s
        print(f"  struct   [{AUX_FEATURES}] ({time.time()-t0:.0f}s)")

    if not args.no_pooled:
        # One model, all four isoforms, isoform identity as a one-hot input block.
        Xb = bank.view(AUX_FEATURES)
        t0 = time.time()
        aux_idx = aux[C.ID_COL].map(bank.index_of).to_numpy()
        aux_fold = aux[C.ID_COL].map(fold_of).fillna(-1).to_numpy()
        A = np.hstack([Xb[aux_idx],
                       np.vstack([iso_onehot[c] for c in aux["isoform"]])]).astype(np.float32)
        ay = aux["y"].to_numpy().astype(int)
        scores["pooled"] = {}
        for cyp in C.TDI_ISOFORMS:
            sub = per_iso[cyp]
            idx = sub[C.ID_COL].map(bank.index_of).to_numpy()
            Q = np.hstack([Xb[idx],
                           np.tile(iso_onehot[cyp], (len(idx), 1))]).astype(np.float32)
            s = np.full(len(sub), np.nan)
            for f in range(n_folds):
                te = (sub["fold"] == f).to_numpy()
                tr_aux = aux_fold != f
                if te.sum() == 0 or tr_aux.sum() < 50:
                    continue
                s[te] = clf_predict("lgb", A[tr_aux], ay[tr_aux], Q[te], args.n_rounds)
            scores["pooled"][cyp] = s
        print(f"  pooled   [all 4 isoforms, {len(aux)} rows] ({time.time()-t0:.0f}s)")

    # ---- an outside member ------------------------------------------------
    # A ChemProp model trained separately can join the blend without this script
    # knowing anything about it: all the rank blend needs is a score per compound per
    # isoform, aligned to the same rows. Missing compounds stay NaN and `rank()`
    # handles them, so a member that covers fewer compounds is not a problem.
    if args.extra_oof:
        ex = pd.read_csv(args.extra_oof)
        scores[args.extra_name] = {}
        for cyp in C.TDI_ISOFORMS:
            col = next((c for c in ex.columns if c.startswith(cyp) and "prob" in c), None)
            m = dict(zip(ex[C.ID_COL], ex[col])) if col else {}
            v = per_iso[cyp][C.ID_COL].map(m).to_numpy(float)
            scores[args.extra_name][cyp] = v
            print(f"  {args.extra_name:8s} {cyp}: {int(np.isfinite(v).sum())}/{len(v)} "
                  f"rows matched from {args.extra_oof}")

    # ---- evaluate ---------------------------------------------------------
    names = list(scores)
    weight = {n: (args.extra_weight if n == args.extra_name else 1.0) for n in names}

    def blend_of(score_map, cyp):
        """Weighted mean of per-member RANKS. Ranks, not probabilities, because the
        members are not calibrated against each other and MCC only cares about order."""
        R = np.vstack([rank(score_map[n][cyp]) for n in names])
        w = np.array([weight[n] for n in names])[:, None]
        return np.nansum(R * w, axis=0) / w.sum()

    if args.save_oof:
        for cyp in C.TDI_ISOFORMS:
            d = per_iso[cyp][[C.ID_COL, "y"]].copy()
            for n in names:
                d[n] = scores[n][cyp]
            d.to_csv(C.artifact(f"tdi_member_oof_{cyp}_{args.tag}.csv"), index=False)
        print(f"  member OOF written to artifacts/tdi_member_oof_<iso>_{args.tag}.csv")
    rows = []
    for name in names + ["RANK-BLEND"]:
        mccs, aucs = [], []
        for cyp in C.TDI_ISOFORMS:
            sub = per_iso[cyp]
            y = sub["y"].to_numpy().astype(bool)
            if name == "RANK-BLEND":
                s = blend_of(scores, cyp)
            else:
                s = scores[name][cyp]
            ok = np.isfinite(s)
            _, m = T.best_mcc_threshold(y[ok], s[ok])
            from sklearn.metrics import roc_auc_score  # noqa: PLC0415
            aucs.append(roc_auc_score(y[ok], s[ok]))
            mccs.append(m)
        rows.append(dict(model=name, **{f"MCC_{c}": v for c, v in zip(C.TDI_ISOFORMS, mccs)},
                         MCC_macro=float(np.mean(mccs)),
                         AUC_macro=float(np.mean(aucs))))
    tab = pd.DataFrame(rows).sort_values("AUC_macro", ascending=False)
    print("\n=== out-of-fold, sorted by AUC — the metric that decides ===")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    tab.to_csv(C.artifact(f"tdi_ensemble_summary_{args.tag}.csv"), index=False)
    print("\nprevious leaderboard: OOF MCC 0.2524 -> actual 0.285 (rank 22)")

    blend_oof = {cyp: blend_of(scores, cyp) for cyp in C.TDI_ISOFORMS}
    thr = {}
    for cyp in C.TDI_ISOFORMS:
        sub = per_iso[cyp]
        y = sub["y"].to_numpy().astype(bool)
        s = blend_oof[cyp]
        ok = np.isfinite(s)
        t_mcc, m = T.best_mcc_threshold(y[ok], s[ok])
        thr[cyp] = dict(mcc_optimal=t_mcc, mcc=m,
                        train_rate=float(np.quantile(s[ok], 1 - y.mean())),
                        test_rate=float(np.quantile(s[ok], 1 - TEST_PREVALENCE)))
        print(f"  {cyp}: MCC-optimal thr {t_mcc:.3f} (MCC {m:.4f}), "
              f"train-rate thr {thr[cyp]['train_rate']:.3f}, "
              f"test-rate({TEST_PREVALENCE:.3f}) thr {thr[cyp]['test_rate']:.3f}")

    # ---- blind set --------------------------------------------------------
    if args.predict:
        print("\n=== refitting on all data and predicting the blind set ===")
        test_idx = test[C.ID_COL].map(bank.index_of).to_numpy()
        test_scores = {}
        for name in args.learners:
            kind, feat = POOL[name]["kind"], POOL[name]["features"]
            X = bank.view(feat)
            test_scores[name] = {}
            for cyp in C.TDI_ISOFORMS:
                sub = per_iso[cyp]
                idx = sub[C.ID_COL].map(bank.index_of).to_numpy()
                tr = sub["trainable"].to_numpy()
                wc = sub["w"].to_numpy(float)[tr]
                test_scores[name][cyp] = clf_predict(
                    kind, X[idx[tr]], sub["y"].to_numpy()[tr], X[test_idx],
                    args.n_rounds, w=wc if (wc != 1).any() else None,
                    opts=POOL[name].get("opts"))
            print(f"  {name} done")
        if not args.no_struct:
            X = bank.view(AUX_FEATURES)
            test_scores["struct"] = {}
            for cyp in C.TDI_ISOFORMS:
                sub = per_iso[cyp]
                idx = sub[C.ID_COL].map(bank.index_of).to_numpy()
                test_scores["struct"][cyp] = structured(X[idx], sub, X[test_idx], args.n_rounds)
            print("  struct done")
        if not args.no_pooled:
            Xb = bank.view(AUX_FEATURES)
            aux_idx = aux[C.ID_COL].map(bank.index_of).to_numpy()
            A = np.hstack([Xb[aux_idx],
                           np.vstack([iso_onehot[c] for c in aux["isoform"]])]).astype(np.float32)
            ay = aux["y"].to_numpy().astype(int)
            test_scores["pooled"] = {}
            for cyp in C.TDI_ISOFORMS:
                Q = np.hstack([Xb[test_idx],
                               np.tile(iso_onehot[cyp], (len(test_idx), 1))]).astype(np.float32)
                test_scores["pooled"][cyp] = clf_predict("lgb", A, ay, Q, args.n_rounds)
            print("  pooled done")

        if args.extra_test:
            ex = pd.read_csv(args.extra_test)
            test_scores[args.extra_name] = {}
            for cyp in C.TDI_ISOFORMS:
                col = next((c for c in ex.columns
                            if c.startswith(cyp) and "prob" in c), None)
                m = dict(zip(ex[C.ID_COL], ex[col])) if col else {}
                test_scores[args.extra_name][cyp] = test[C.ID_COL].map(m).to_numpy(float)
            print(f"  {args.extra_name} test scores loaded from {args.extra_test}")
        elif args.extra_oof:
            raise SystemExit(
                "--extra-oof was given without --extra-test: the blind-set scores for "
                "that member are missing, so the submission would silently be built "
                "from a different blend than the one that was validated.")

        out = test[[C.SMILES_COL, C.ID_COL]].copy()
        scores_out = test[[C.ID_COL]].copy()
        for cyp in C.TDI_ISOFORMS:
            s = blend_of(test_scores, cyp)
            scores_out[f"{cyp}_score"] = s
            # Rank space is not comparable across train and test, so match the rate
            # rather than transplanting the threshold.
            want_eff = OPTIMAL_RATE.get(cyp, TEST_PREVALENCE)
            rate = submitted_rate(cyp, want_eff)
            cut = float(np.quantile(s, 1 - rate))
            out[f"{cyp}_is_TDI"] = s >= cut
            got = float((s >= cut).mean())
            print(f"  {cyp}: cut at {rate:.3f} of the file -> {got:.3f} submitted, "
                  f"~{got * RATE_INFLATION.get(cyp, 1.0):.3f} effective "
                  f"(target {want_eff:.3f})")
        scores_out.to_csv(C.artifact(f"testscores_{args.tag}.csv"), index=False)
        if args.save_oof:
            # Persist EVERY member's blind-set scores, not just the blend's. Without
            # this a weight refit is impossible without re-running the whole pool --
            # which is exactly what blocked us last time: the out-of-fold scores were
            # saved, the test scores were not, so tuned weights could be measured but
            # never submitted.
            for name in names:
                d = test[[C.ID_COL]].copy()
                for cyp in C.TDI_ISOFORMS:
                    d[f"{cyp}_score"] = test_scores[name][cyp]
                d.to_csv(C.artifact(f"tdi_member_test_{name}_{args.tag}.csv"), index=False)
            print(f"  per-member blind-set scores -> "
                  f"artifacts/tdi_member_test_<member>_{args.tag}.csv")
        path = C.artifact(f"submission_{args.tag}.csv")
        out.to_csv(path, index=False)
        print(f"\nwritten {path}  (+ testscores_{args.tag}.csv for re-thresholding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
