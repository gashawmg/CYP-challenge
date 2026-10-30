#!/usr/bin/env python
r"""How much accuracy does an INTERPRETABLE CYP model actually cost?

    python train_interpretable.py --iso all
    python train_interpretable.py --iso CYP2D6 --show-model

THE QUESTION. Everything this project ships is a boosted ensemble over thousands of
columns. Nobody can read it, and it cannot be defended in a paper beyond "the ensemble said
so". Section 36 cut the descriptor block to ~40 mechanism-justified columns with no
accuracy lost and two endpoints improved. This asks the next question: how far UP the
interpretability ladder can we go before accuracy actually breaks?

FIVE RUNGS, most readable first, identical folds, identical features:

  linear-10     ridge on 10 descriptors chosen INSIDE each training fold. Ten coefficients.
                You can write the model on one line and a chemist can argue with it.
  GAM           additive splines, one shape function per descriptor. No interactions, so
                "what does this model think logD does to CYP2D6 potency" has one answer
                and it can be plotted.
  lgbm-mono     gradient boosting with MONOTONE CONSTRAINTS taken from known CYP
                pharmacology (below). Interactions are allowed; directions are not free.
  lgbm-lean     the same booster, unconstrained -- the reference.
  lgbm-full     unconstrained on the full pre-Section-36 block.

THE CONSTRAINTS ARE PRIOR PHARMACOLOGY, WRITTEN DOWN BEFORE THE FIT:

    MolLogP                +   lipophilicity drives CYP binding; the oldest QSAR in the field
    TPSA                   -   polar surface resists partitioning into a hydrophobic site
    NumHDonors             -   desolvation penalty on binding
    core_heme_aromatic_n   +   type-II coordination of the heme iron
    core_imidazole         +   the canonical type-II motif (ketoconazole, miconazole)
    core_triazole          +   likewise (fluconazole, voriconazole)
    CYP2D6_is_base         +   basic N salt-bridges Asp301/Glu216
    phys_net_charge_74     +   on CYP2D6 only: the site recognises CATIONS
                           -   on CYP2C9 only: Arg108 recognises ANIONS
    CYP2C9_is_acid         +   same site, from the ligand side
    CYP1A2_planarity_index +   a narrow planar slot admits flat aromatics
    CYP3A4_bulk_index      +   a large promiscuous pocket rewards bulk

IF THE CONSTRAINED MODEL MATCHES THE FREE ONE, that is a result in itself: the data agree
with fifty years of CYP pharmacology, and the model can be defended mechanistically rather
than only statistically. IF IT LOSES BADLY, either our priors are wrong on this assay or
the free model is exploiting something that is not binding physics -- both worth knowing.

SCORED on the similarity-reweighted yardstick (Section 30), paired bootstrap on every gap.
No submission follows from this; it decides what we can claim to understand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cyp_target_descriptors as TD  # noqa: E402
from cyp_target_descriptors_lean import lean_index, lean_names  # noqa: E402

EDGES = np.array([0.0, .30, .35, .40, .45, .50, .55, .60, .65, .70, 1.01])
PRIOR = {                      # feature -> required sign, shared across isoforms
    "MolLogP": +1, "TPSA": -1, "NumHDonors": -1,
    "core_heme_aromatic_n": +1, "core_imidazole": +1, "core_triazole": +1,
}
PRIOR_ISO = {                  # feature -> sign, only for that isoform
    "CYP1A2": {"CYP1A2_planarity_index": +1, "CYP1A2_fused_aromatic_rings": +1},
    "CYP2C9": {"CYP2C9_is_acid": +1, "phys_net_charge_74": -1},
    "CYP2D6": {"CYP2D6_is_base": +1, "phys_net_charge_74": +1},
    "CYP3A4": {"CYP3A4_bulk_index": +1},
}


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", default="all")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--k", type=int, default=10, help="features in the linear rung")
    ap.add_argument("--show-model", action="store_true")
    a = ap.parse_args()

    from rdkit import Chem, DataStructs, RDLogger                       # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator                       # noqa: PLC0415
    from sklearn.linear_model import RidgeCV                            # noqa: PLC0415
    from sklearn.pipeline import make_pipeline                          # noqa: PLC0415
    from sklearn.preprocessing import SplineTransformer, StandardScaler  # noqa: PLC0415
    import lightgbm as lgb                                              # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")

    tr = pd.read_csv("data/cyp-challenge-TRAIN_inhibition.csv")
    te = pd.read_csv("data/cyp-challenge-TEST-BLINDED.csv")
    for d in (tr, te):
        d.columns = [c.strip() for c in d.columns]
    sc = next(c for c in tr.columns if c.lower() == "smiles")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_tr = [gen.GetFingerprint(m) for m in
             (Chem.MolFromSmiles(str(s)) for s in tr[sc]) if m is not None]
    nn_blind = np.array([max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), fp_tr))
                         for m in (Chem.MolFromSmiles(str(s)) for s in te[sc]) if m is not None])

    lab = pd.read_csv("artifacts/labels.csv")
    folds = pd.read_csv("artifacts/folds.csv"); folds.columns = [c.strip() for c in folds.columns]
    fold_of = dict(zip(folds["Molecule_Name"], folds["fold"]))
    sim_of = dict(zip(folds["Molecule_Name"], folds["max_sim_to_other_folds"]))

    def weights(nn):
        pb = np.clip(np.digitize(nn, EDGES) - 1, 0, len(EDGES) - 2)
        bb = np.clip(np.digitize(nn_blind, EDGES) - 1, 0, len(EDGES) - 2)
        p = np.bincount(pb, minlength=len(EDGES) - 1).astype(float)
        q = np.bincount(bb, minlength=len(EDGES) - 1).astype(float)
        p /= max(p.sum(), 1); q /= max(q.sum(), 1)
        w = np.where(p > 0, q / np.maximum(p, 1e-12), 0.0)[pb]
        return w / w.sum() if w.sum() > 0 else np.full(len(nn), 1 / len(nn))

    BOOST = dict(n_estimators=700, learning_rate=0.035, num_leaves=63,
                 min_child_samples=20, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.6, reg_lambda=1.0, verbose=-1, n_jobs=-1)
    isos = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4") if a.iso == "all" else (a.iso,)
    rng = np.random.default_rng(0)
    table = []

    for iso in isos:
        t = lab[(lab["isoform"] == iso) & (lab["kind"] != "left")].dropna(subset=["y"])
        t = t[t["Molecule_Name"].isin(fold_of) & t["Molecule_Name"].isin(sim_of)].copy()
        t["fold"] = t["Molecule_Name"].map(fold_of); t = t.reset_index(drop=True)
        y = t["y"].to_numpy(float)
        Xf, _ = TD.calculate(t["SMILES"].tolist(), iso)
        Xf = np.nan_to_num(np.asarray(Xf, float))
        cols = lean_index(iso); names = lean_names(iso)
        X = Xf[:, cols]
        w = weights(t["Molecule_Name"].map(sim_of).to_numpy(float))

        signs = dict(PRIOR); signs.update(PRIOR_ISO.get(iso, {}))
        mono = [signs.get(n, 0) for n in names]
        n_con = sum(1 for m in mono if m)

        preds, chosen = {}, []
        for f in sorted(t["fold"].unique()):
            trm = (t["fold"] != f).to_numpy(); tem = (t["fold"] == f).to_numpy()
            # ---- linear-10: selection happens INSIDE the fold, or it leaks -----------
            r = np.array([abs(spearman(X[trm, j], y[trm])) for j in range(X.shape[1])])
            sel = list(np.argsort(r)[::-1][:a.k]); chosen.append(sel)
            lin = make_pipeline(StandardScaler(),
                                RidgeCV(alphas=np.logspace(-2, 3, 20)))
            lin.fit(X[np.ix_(trm, sel)], y[trm])
            preds.setdefault("linear-10", np.zeros(len(t)))[tem] = lin.predict(X[np.ix_(tem, sel)])
            # ---- GAM: additive splines, one shape per descriptor --------------------
            gam = make_pipeline(StandardScaler(),
                                SplineTransformer(n_knots=6, degree=3),
                                RidgeCV(alphas=np.logspace(-1, 4, 20)))
            gam.fit(X[trm], y[trm])
            preds.setdefault("GAM", np.zeros(len(t)))[tem] = gam.predict(X[tem])
            # ---- boosters -----------------------------------------------------------
            mm = lgb.LGBMRegressor(monotone_constraints=mono, **BOOST)
            mm.fit(X[trm], y[trm])
            preds.setdefault("lgbm-mono", np.zeros(len(t)))[tem] = mm.predict(X[tem])
            fm = lgb.LGBMRegressor(**BOOST); fm.fit(X[trm], y[trm])
            preds.setdefault("lgbm-lean", np.zeros(len(t)))[tem] = fm.predict(X[tem])
            am = lgb.LGBMRegressor(**BOOST); am.fit(Xf[trm], y[trm])
            preds.setdefault("lgbm-full", np.zeros(len(t)))[tem] = am.predict(Xf[tem])

        ref = preds["lgbm-lean"]
        print(f"\n=== {iso}   {len(t)} rows, {len(names)} lean descriptors, "
              f"{n_con} of them sign-constrained")
        print(f"  {'model':12s} {'rho':>7s} {'vs lgbm-lean':>13s} {'95% CI':>20s}")
        for k, v in preds.items():
            b = []
            for _ in range(a.boot):
                i = rng.choice(len(y), len(y), replace=True, p=w)
                if len(np.unique(y[i])) > 2:
                    b.append(spearman(v[i], y[i]) - spearman(ref[i], y[i]))
            b = np.asarray(b)
            print(f"  {k:12s} {spearman(v, y):7.4f} {b.mean():+13.4f} "
                  f"[{np.quantile(b, .025):+.4f},{np.quantile(b, .975):+.4f}]")
            table.append(dict(isoform=iso, model=k, rho=spearman(v, y), gap=float(b.mean()),
                              lo=float(np.quantile(b, .025)), hi=float(np.quantile(b, .975))))

        if a.show_model:
            counts = pd.Series([names[j] for s in chosen for j in s]).value_counts()
            print(f"\n  descriptors the linear rung picked in all 5 folds:")
            for n, c in counts.items():
                if c == 5:
                    print(f"    {n}")
            lin = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
            keep = [names.index(n) for n, c in counts.items() if c == 5]
            lin.fit(X[:, keep], y)
            co = lin[-1].coef_
            print(f"\n  the whole model, in standardised pIC50 units per SD of descriptor:")
            for n, c in sorted(zip([names[j] for j in keep], co), key=lambda z: -abs(z[1])):
                flag = ""
                if n in signs:
                    flag = "  AGREES with prior" if np.sign(c) == signs[n] else \
                           "  *** CONTRADICTS prior ***"
                print(f"    {c:+7.3f}  {n}{flag}")

    pd.DataFrame(table).to_csv("artifacts/interpretable_ladder.csv", index=False)
    if len(isos) > 1:
        print("\n" + "=" * 70)
        p = pd.DataFrame(table).pivot_table(index="model", columns="isoform", values="rho")
        print("\nrho by rung (reweighted to blind difficulty)\n")
        print(p.round(4).to_string())
        print("\n  Read the LADDER, not any single row: the cost of each step toward")
        print("  readability is the row-to-row difference, and a step that costs less than")
        print("  its own interval is free interpretability.")
    print("\n  wrote artifacts/interpretable_ladder.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
