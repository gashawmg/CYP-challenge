r"""
tdi_external_2d6.py  --  can external data make the CYP2D6 TDI ranking readable?

THE DIAGNOSIS THIS ANSWERS. scripts\tdi_yardstick_v2.py reweights our out-of-fold rows to
the blind set's similarity distribution and then predicts a board MCC. On CYP3A4 it
predicted 0.4596 against a board that returned 0.4601. On CYP2D6 it predicted 0.0780
against a board that returned 0.2918, z +3.28, and every one of the twenty members in the
pool failed the same way -- the best, svc_tan, reaches weighted AUC 0.690 and still
predicts only 0.148. The shift model does not rescue it either (0.650). So CYP2D6 is not
a model-selection problem. The populations differ: our labelled rows are 21.6 per cent
TDI-positive and the blind set is 7.1 per cent. We train and validate on an enriched,
hard population and are scored on a dilute, easy one.

WHAT THE EXTERNAL DATA IS FOR, AND IT IS NOT MORE POSITIVES. artifacts/external_cyp_v2.csv
holds 16,233 PubChem/Veith qHTS compounds, of which 13,676 are CYP2D6-INACTIVE. A compound
that does not bind CYP2D6 cannot time-dependently inactivate it, so those are presumed TDI
negatives -- and they are precisely the kind of easy negative our labelled set lacks and the
blind set is full of. Adding about 3,100 of them brings training prevalence from 0.2164 to
the blind set's 0.0706.

    ARM control    our 1,497 labelled rows, as we ship today
    ARM prior      plus the external affinity prior as FEATURES (cypkit.external_prior)
    ARM negatives  plus external CYP2D6-inactive rows as presumed negatives, to blind prevalence
    ARM both       prior features and presumed negatives together

THREE RULES THIS SCRIPT DOES NOT BEND.

  THE LEAKAGE GUARD STAYS ON. Any external compound sharing an InChIKey connectivity block
  with a blind compound is dropped before anything else happens, and the count is printed.
  If the printed count is zero, something is wrong with the guard, not with the data.

  PRESUMED NEGATIVES NEVER ENTER THE EVALUATION. They are an assumption, so scoring against
  them would be scoring against ourselves. Every arm is evaluated on the SAME 1,497 real
  labelled rows with real labels; the external rows only ever appear in a training half.
  This makes the test conservative: it asks whether easy negatives help on the HARD
  population, which is a stricter question than the board asks.

  EXTERNAL ROWS INHERIT A FOLD. Each external compound is assigned the fold of its nearest
  labelled neighbour, so an analogue of a held-out compound is held out with it instead of
  leaking into that fold's training half.

THE VERDICT IS THE YARDSTICK, NOT THE RAW AUC. Each arm is reported as blind-weighted AUC,
the implied d', and the board MCC that implies. The number that matters is whether the
predicted MCC moves from 0.0780 toward the 0.2918 the board actually returned. A rise in
raw out-of-fold AUC that does not move the prediction is not progress.

Run (Windows, admet_gpu -- needs lightgbm, which the project env has):
  python scripts\tdi_external_2d6.py
  python scripts\tdi_external_2d6.py --seeds 0 1 2 --target-prevalence 0.0706
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tdi_yardstick_v2 import (  # noqa: E402
    BOARD, auc_to_dprime, fingerprints, max_tanimoto, predict_board, weighted_auc,
)

ISO = "CYP2D6"
TDI_CSV = ROOT / "data" / "cyp-challenge-TRAIN_TDI.csv"
BLIND_CSV = ROOT / "data" / "cyp-challenge-TEST-BLINDED.csv"
FOLDS_CSV = ROOT / "artifacts" / "folds_tdi.csv"
EXT_CSV = ROOT / "artifacts" / "external_cyp_v2.csv"


def inchikey_blocks(smiles):
    """first block of the InChIKey -- the connectivity layer, ignoring stereo and salts"""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        out.append(Chem.MolToInchiKey(m).split("-")[0] if m is not None else None)
    return out


def ecfp(smiles, n_bits=2048):
    return fingerprints(smiles)  # (array, kept index)


def fit_predict(Xtr, ytr, Xva, seed):
    import lightgbm as lgb

    d = lgb.Dataset(Xtr, label=ytr.astype(int))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=20, feature_fraction=0.5, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1, seed=seed, num_threads=0)
    booster = lgb.train(params, d, num_boost_round=400)
    return booster.predict(Xva)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--target-prevalence", type=float, default=0.0706,
                    help="blind CYP2D6 TDI prevalence; external negatives are added to reach it")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--no-prior", action="store_true", help="skip the external-prior arms")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    b = BOARD[ISO]

    # ---------------------------------------------------------------- labelled rows
    tdi = pd.read_csv(TDI_CSV)
    folds = pd.read_csv(FOLDS_CSV)
    lab = tdi[tdi[f"{ISO}_is_TDI"].notna()].merge(folds, on="Molecule_Name", how="left")
    lab = lab[lab["fold"].notna()].reset_index(drop=True)
    fp_lab, keep = fingerprints(lab["SMILES"].tolist())
    lab = lab.iloc[keep].reset_index(drop=True)
    y = lab[f"{ISO}_is_TDI"].astype(bool).to_numpy()
    fold = lab["fold"].to_numpy().astype(int)
    print(f"labelled rows {len(lab)}  positives {int(y.sum())}  prevalence {y.mean():.4f}")

    # ---------------------------------------------------------------- blind + the guard
    blind = pd.read_csv(BLIND_CSV)
    fp_blind, bkeep = fingerprints(blind["SMILES"].tolist())
    blind_blocks = set(x for x in inchikey_blocks(blind["SMILES"].tolist()) if x)

    ext = pd.read_csv(EXT_CSV)
    ext = ext[ext[f"{ISO}_active"].notna()].reset_index(drop=True)
    neg = ext[ext[f"{ISO}_active"].astype(float) == 0.0].reset_index(drop=True)
    print(f"external rows {len(ext)}   CYP2D6-inactive (presumed TDI-negative) {len(neg)}")

    if "block" in neg.columns:
        before = len(neg)
        neg = neg[~neg["block"].isin(blind_blocks)].reset_index(drop=True)
        print(f"LEAKAGE GUARD: dropped {before-len(neg)} external rows sharing an "
              f"InChIKey connectivity block with a blind compound ({len(neg)} remain)")
    else:
        blocks = inchikey_blocks(neg["smiles_std"].tolist())
        neg["block"] = blocks
        before = len(neg)
        neg = neg[~neg["block"].isin(blind_blocks)].reset_index(drop=True)
        print(f"LEAKAGE GUARD (recomputed): dropped {before-len(neg)} external rows "
              f"({len(neg)} remain)")
    if len(neg) == before:
        print("  WARNING: the guard dropped NOTHING. Verify it before trusting any result.")

    # how many presumed negatives bring us to the blind prevalence
    n_pos = int(y.sum())
    n_need = int(round(n_pos / a.target_prevalence)) - len(lab)
    n_add = max(0, min(n_need, len(neg)))
    print(f"to reach prevalence {a.target_prevalence:.4f} we add {n_add} presumed negatives "
          f"(of {len(neg)} available)")

    fp_neg, nkeep = fingerprints(neg["smiles_std"].tolist())
    neg = neg.iloc[nkeep].reset_index(drop=True)

    # each external row inherits the fold of its nearest labelled neighbour
    print("assigning external rows to folds by nearest labelled neighbour ...")
    nn_fold = np.empty(len(neg), dtype=int)
    refT = fp_lab.T.astype(np.float32)
    rsum = fp_lab.sum(axis=1).astype(np.float32)
    for i in range(0, len(neg), 256):
        q = fp_neg[i : i + 256].astype(np.float32)
        inter = q @ refT
        union = q.sum(axis=1, keepdims=True) + rsum[None, :] - inter
        nn_fold[i : i + 256] = fold[np.argmax(inter / np.maximum(union, 1e-9), axis=1)]

    sel = rng.permutation(len(neg))[:n_add]
    neg_fp, neg_fold = fp_neg[sel], nn_fold[sel]

    # ---------------------------------------------------------------- optional prior features
    prior_lab = prior_neg = None
    if not a.no_prior:
        try:
            from cypkit import external_prior as EP
            pl, names = EP.calculate(lab["SMILES"].tolist())
            pn, _ = EP.calculate(neg.iloc[sel]["smiles_std"].tolist())
            prior_lab, prior_neg = np.asarray(pl, float), np.asarray(pn, float)
            print(f"external prior features: {prior_lab.shape[1]} columns ({', '.join(names[:4])}, ...)")
        except Exception as e:  # noqa: BLE001
            print(f"external prior unavailable ({e.__class__.__name__}: {e}); "
                  f"running the two arms that do not need it")

    # ---------------------------------------------------------------- yardstick weights
    sim = np.zeros(len(lab), dtype=np.float32)
    for k in sorted(set(fold)):
        va = fold == k
        if va.sum() and (~va).sum():
            sim[va] = max_tanimoto(fp_lab[va], fp_lab[~va])
    blind_sim = max_tanimoto(fp_blind, fp_lab)
    edges = np.unique(np.quantile(blind_sim, np.linspace(0, 1, a.bins + 1)))
    edges[0], edges[-1] = -1.0, 2.0
    pb, _ = np.histogram(blind_sim, bins=edges)
    po, _ = np.histogram(sim, bins=edges)
    ratio = np.clip(np.where(po > 0, (pb / pb.sum()) / np.maximum(po / po.sum(), 1e-12), 0.0),
                    0.0, a.clip)
    w = ratio[np.clip(np.digitize(sim, edges) - 1, 0, len(ratio) - 1)]
    w = w / w.mean()

    # ---------------------------------------------------------------- the four arms
    arms = {"control": (False, False), "prior": (True, False),
            "negatives": (False, True), "both": (True, True)}
    if prior_lab is None:
        arms = {k: v for k, v in arms.items() if not v[0]}

    print(f"\n{'arm':<12}{'raw AUC':>9}{'wtd AUC':>9}{'d-prime':>9}{'pred MCC':>10}"
          f"{'z vs board':>12}{'verdict':>10}")
    out = {}
    for name, (use_prior, use_neg) in arms.items():
        oof = np.zeros(len(lab))
        for seed in a.seeds:
            for k in sorted(set(fold)):
                va, tr = fold == k, fold != k
                Xtr = fp_lab[tr].astype(np.float32)
                ytr = y[tr]
                if use_prior:
                    Xtr = np.hstack([Xtr, prior_lab[tr]])
                if use_neg:
                    nsel = neg_fold != k
                    Xn = neg_fp[nsel].astype(np.float32)
                    if use_prior:
                        Xn = np.hstack([Xn, prior_neg[nsel]])
                    Xtr = np.vstack([Xtr, Xn])
                    ytr = np.concatenate([ytr, np.zeros(nsel.sum(), dtype=bool)])
                Xva = fp_lab[va].astype(np.float32)
                if use_prior:
                    Xva = np.hstack([Xva, prior_lab[va]])
                oof[va] += fit_predict(Xtr, ytr, Xva, seed)
        oof /= len(a.seeds)
        ar = weighted_auc(y, oof, np.ones_like(w))
        aw = weighted_auc(y, oof, w)
        dp = auc_to_dprime(aw)
        pred = predict_board(dp, b["prevalence"], b["call_rate"], b["n_half"], rng, 3000)
        z = (b["mcc"] - pred.mean()) / pred.std()
        ok = abs(z) < 2.0
        out[name] = (ar, aw, dp, pred.mean(), z, oof)
        print(f"{name:<12}{ar:>9.4f}{aw:>9.4f}{dp:>9.3f}{pred.mean():>10.4f}{z:>+12.2f}"
              f"{'PASS' if ok else 'fail':>10}")

    print(f"\nboard CYP2D6 MCC {b['mcc']:.4f} implies d' 1.227 / weighted AUC 0.807")
    print("A PASS means the ruler finally brackets the board on CYP2D6 and the endpoint can")
    print("be unfrozen. A fail that still moves pred MCC toward 0.2918 is progress worth")
    print("another arm; a fail that does not move it says external data is not the answer.")

    best = max(out.items(), key=lambda kv: kv[1][3])
    np.save(ROOT / "artifacts" / f"oof_tdi_2d6_external_{best[0]}.npy", best[1][5])
    print(f"\nbest arm '{best[0]}' out-of-fold scores saved to "
          f"artifacts/oof_tdi_2d6_external_{best[0]}.npy")
    print("Nothing was submitted and no submission file was written.")


if __name__ == "__main__":
    main()
