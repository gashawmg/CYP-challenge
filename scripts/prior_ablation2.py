#!/usr/bin/env python
"""Sharper ablation of the external prior.

Two things were wrong with the first pass:

1. FEATURE DILUTION. Five prior columns sitting among 2,048 ECFP bits, with
   colsample_bytree=0.3, barely get looked at. The ensemble's descriptor learners use
   ~250 columns, where five priors are 2% of the feature space rather than 0.24%.

2. THE WRONG QUESTION. Mean absolute error over all rows is dominated by the
   screen-only inactives sitting at the floor, which we already predict well. The
   leaderboard gap is RANKING among compounds that actually have a fitted curve
   (top of the board: Spearman 0.76; us: 0.65). So Spearman is reported twice --
   over everything, and over the fitted rows alone.
"""
from __future__ import annotations

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from scipy import stats
import lightgbm as lgb
from cypkit import features, config as C
from cypkit import cyp_descriptors as CD

AUX = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]


def main():
    t0 = time.time()
    folds = pd.read_csv("artifacts/folds.csv")
    lab = pd.read_csv("artifacts/labels.csv")
    P = np.load("artifacts/prior_train.npy")
    smi = folds["SMILES"].tolist()

    Xd, _ = features.rdkit_2d(smi)
    Xc, _ = CD.calculate(smi)
    D = np.nan_to_num(np.hstack([Xd, Xc]).astype(np.float32), nan=0.0,
                      posinf=0.0, neginf=0.0)
    print(f"descriptor block {D.shape} ({time.time()-t0:.0f}s)", flush=True)

    idx = {m: i for i, m in enumerate(folds["Molecule_Name"])}
    fold_of = dict(zip(folds["Molecule_Name"], folds["fold"]))
    hdr = (f"{'iso':8s} {'n':>5s} {'nfit':>5s} | {'rho all':>8s} {'+prior':>7s} | "
           f"{'rho fitted':>10s} {'+prior':>7s} | {'MAE':>6s} {'+prior':>7s}")
    print("\n" + hdr, flush=True)
    out = []
    for iso in C.ISOFORMS:
        sub = lab[(lab["isoform"] == iso) & lab["Molecule_Name"].isin(idx)]
        y = sub["y"].to_numpy(float); hi = sub["y_high"].to_numpy(float)
        fitted = np.isfinite(y)
        y = np.where(fitted, y, np.where(np.isfinite(hi), hi, C.PIC50_FLOOR))
        rid = sub["Molecule_Name"].map(idx).to_numpy()
        f = sub["Molecule_Name"].map(fold_of).to_numpy()
        res = {}
        for tag, X in (("base", D), ("prior", np.hstack([D, P]))):
            oof = np.full(len(sub), np.nan)
            for k in range(5):
                tr, te = f != k, f == k
                if te.sum() == 0:
                    continue
                m = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.04,
                                      num_leaves=31, colsample_bytree=0.7,
                                      subsample=0.8, subsample_freq=1,
                                      min_child_samples=15, verbose=-1, n_jobs=2,
                                      random_state=0)
                m.fit(X[rid[tr]], y[tr]); oof[te] = m.predict(X[rid[te]])
            res[tag] = (stats.spearmanr(oof, y)[0],
                        stats.spearmanr(oof[fitted], y[fitted])[0],
                        float(np.abs(oof - y).mean()))
        print(f"{iso:8s} {len(sub):5d} {int(fitted.sum()):5d} | "
              f"{res['base'][0]:8.3f} {res['prior'][0]:7.3f} | "
              f"{res['base'][1]:10.3f} {res['prior'][1]:7.3f} | "
              f"{res['base'][2]:6.3f} {res['prior'][2]:7.3f}  ({time.time()-t0:.0f}s)",
              flush=True)
        out.append({"isoform": iso, "n": len(sub), "n_fitted": int(fitted.sum()),
                    "rho_base": res["base"][0], "rho_prior": res["prior"][0],
                    "rhofit_base": res["base"][1], "rhofit_prior": res["prior"][1],
                    "mae_base": res["base"][2], "mae_prior": res["prior"][2]})
    d = pd.DataFrame(out)
    d.to_csv("artifacts/prior_ablation2.csv", index=False)
    print(f"\nmean rho all    {d.rho_base.mean():.3f} -> {d.rho_prior.mean():.3f}")
    print(f"mean rho fitted {d.rhofit_base.mean():.3f} -> {d.rhofit_prior.mean():.3f}")
    print(f"mean MAE        {d.mae_base.mean():.3f} -> {d.mae_prior.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
