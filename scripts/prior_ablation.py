#!/usr/bin/env python
"""Does an external-trained prior add signal ON TOP of what we already model?

    python scripts/prior_ablation.py

The prior is one number per isoform per molecule: what a model trained ONLY on the
PubChem/Veith panel predicts for it. It never sees a challenge label, so it is a
legitimate feature for both train and test, and it carries knowledge from 16k
molecules into a 4.9k-molecule problem.

The 175 compounds present in BOTH sets are dropped from the external model's training
data first. Otherwise the prior would be near-perfect on exactly those rows and the
ablation would flatter itself.
"""
from __future__ import annotations

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from scipy import stats
import lightgbm as lgb
from cypkit import features, config as C
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser, Uncharger

RDLogger.DisableLog("rdApp.*")
LF, UC = LargestFragmentChooser(), Uncharger()
AUX = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]


def key(s):
    m = Chem.MolFromSmiles(str(s))
    return None if m is None else Chem.MolToInchiKey(UC.uncharge(LF.choose(m))).split("-")[0]


def main():
    t0 = time.time()
    ex = pd.read_csv("artifacts/external_cyp.csv")
    folds = pd.read_csv("artifacts/folds.csv")
    lab = pd.read_csv("artifacts/labels.csv")
    ch = folds[["Molecule_Name", "SMILES", "fold"]].copy()
    ch["inchikey"] = [key(s) for s in ch["SMILES"]]
    n0 = len(ex)
    ex = ex[~ex["inchikey"].isin(set(ch["inchikey"].dropna()))].reset_index(drop=True)
    print(f"external {n0} -> {len(ex)} after dropping challenge-train overlap", flush=True)

    Xe, _, _ = features.fingerprint_block(ex["smiles_std"].tolist(), "ecfp")
    Xc, _, _ = features.fingerprint_block(ch["SMILES"].tolist(), "ecfp")
    print(f"ECFP built ({time.time()-t0:.0f}s)", flush=True)

    P = np.zeros((len(ch), len(AUX)), np.float32)
    for j, iso in enumerate(AUX):
        y = ex[f"{iso}_pIC50"]; ok = y.notna().to_numpy()
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                              verbose=-1, n_jobs=2, random_state=0)
        m.fit(Xe[ok], y[ok]); P[:, j] = m.predict(Xc)
        print(f"  prior {iso}: trained on {int(ok.sum())} ({time.time()-t0:.0f}s)", flush=True)
    np.save("artifacts/prior_train.npy", P)

    idx = {m: i for i, m in enumerate(ch["Molecule_Name"])}
    fold_of = dict(zip(ch["Molecule_Name"], ch["fold"]))
    print(f"\n{'iso':8s} {'n':>5s} {'rho base':>9s} {'rho+prior':>10s} "
          f"{'MAE base':>9s} {'MAE+prior':>10s}", flush=True)
    out = []
    for iso in C.ISOFORMS:
        sub = lab[(lab["isoform"] == iso) & lab["Molecule_Name"].isin(idx)]
        y = sub["y"].to_numpy(float); hi = sub["y_high"].to_numpy(float)
        y = np.where(np.isfinite(y), y, np.where(np.isfinite(hi), hi, C.PIC50_FLOOR))
        rid = sub["Molecule_Name"].map(idx).to_numpy()
        f = sub["Molecule_Name"].map(fold_of).to_numpy()
        res = {}
        for tag, X in (("base", Xc), ("prior", np.hstack([Xc, P]))):
            oof = np.full(len(sub), np.nan)
            for k in range(5):
                tr, te = f != k, f == k
                if te.sum() == 0:
                    continue
                m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=31,
                                      colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                                      min_child_samples=15, verbose=-1, n_jobs=2,
                                      random_state=0)
                m.fit(X[rid[tr]], y[tr]); oof[te] = m.predict(X[rid[te]])
            res[tag] = (stats.spearmanr(oof, y)[0], float(np.abs(oof - y).mean()))
            np.save(f"artifacts/prior_ab_{iso}_{tag}.npy", oof)
        print(f"{iso:8s} {len(sub):5d} {res['base'][0]:9.3f} {res['prior'][0]:10.3f} "
              f"{res['base'][1]:9.3f} {res['prior'][1]:10.3f}  ({time.time()-t0:.0f}s)",
              flush=True)
        out.append({"isoform": iso, "n": len(sub),
                    "rho_base": res["base"][0], "rho_prior": res["prior"][0],
                    "mae_base": res["base"][1], "mae_prior": res["prior"][1]})
    pd.DataFrame(out).to_csv("artifacts/prior_ablation.csv", index=False)
    d = pd.DataFrame(out)
    print(f"\nmean rho {d.rho_base.mean():.3f} -> {d.rho_prior.mean():.3f}   "
          f"mean MAE {d.mae_base.mean():.3f} -> {d.mae_prior.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
