#!/usr/bin/env python
"""Weight training by how much the metric CHARGES for each compound.

    python scripts/interval_weighting.py

WHY. ST-RAE is zero inside a compound's fitted credible interval, so a wide interval
is a wide dead zone and a narrow one is nearly a point target. Decomposing our
submitted blend's out-of-fold cost by interval width:

    quartile        share of rows   land inside   share of TOTAL ST-RAE cost
    narrowest              25%         5-11%              38-60%
    widest                 25%        35-62%              11-18%

The narrowest quartile is a quarter of the rows and up to SIXTY PERCENT of the cost.
Every model in this project trains on those rows with the same weight as a compound
whose interval is three times wider and therefore nearly free. We have been optimising
mean squared error against a metric that does not charge like mean squared error.

That is also the shape of the leaderboard around us. Ranks 44 and 46 have identical
MAE (0.6922 / 0.6906), R2 (0.4818 / 0.4856) and Spearman (0.7148 / 0.7147) -- and
ST-RAE 0.6443 vs 0.6691. A 0.025 gap that none of the ordinary regression metrics
sees. Where the errors fall relative to the intervals is what separates them.

WHAT IS TESTED. Sample weights proportional to 1/width, at several strengths, plus a
cap so a handful of very tight intervals cannot dominate the fit. Scored by OOF ST-RAE
on the frozen folds -- the metric itself, not a proxy.

The weights use only TRAINING intervals. The blind set's intervals are unknown and
never needed: this changes what the model pays attention to while fitting, not what it
outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

from cypkit import features, splits, config as C
from cypkit import cyp_descriptors as CD

POWERS = [0.0, 0.5, 1.0, 1.5]      # weight = (median_width / width) ** power
CAP = 4.0                          # no compound may outweigh the median by more than this


def strae(p, y, lo, hi):
    err = np.maximum(0.0, np.maximum(lo - p, p - hi))
    den = np.abs(y - y.mean()).mean()
    return float(err.mean() / den)


def main():
    tr = pd.read_csv("data/cyp-challenge-TRAIN_inhibition.csv")
    tr.columns = [c.strip() for c in tr.columns]
    folds = splits.load_folds(C.artifact("folds.csv"))
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))

    smiles = folds[C.SMILES_COL].tolist()
    Xd, _ = features.rdkit_2d(smiles)
    Xc, _ = CD.calculate(smiles)
    Xf, _, _ = features.fingerprint_block(smiles, "ecfp")
    X = np.nan_to_num(np.hstack([Xd, Xc, Xf]).astype(np.float32), nan=0.0,
                      posinf=0.0, neginf=0.0)
    idx = {m: i for i, m in enumerate(folds[C.ID_COL])}
    print(f"features {X.shape}\n")

    print(f"{'isoform':8s} {'power':>6s} {'ST-RAE':>8s} {'MAE':>7s} {'rho':>7s} "
          f"{'inside':>7s} {'narrowQ1 cost':>14s}")
    summary = {}
    for iso in C.ISOFORMS:
        t = f"{iso}_pIC50_direct_inhibition"
        m = tr[["Molecule_Name", t, t + "_conf_low", t + "_conf_high"]].dropna(subset=[t])
        m = m[m["Molecule_Name"].isin(idx)]
        y = m[t].to_numpy(float)
        lo = m[t + "_conf_low"].to_numpy(float)
        hi = m[t + "_conf_high"].to_numpy(float)
        # a missing bound means no dead zone -- treat the point value as the target
        lo = np.where(np.isfinite(lo), lo, y)
        hi = np.where(np.isfinite(hi), hi, y)
        width = np.maximum(hi - lo, 1e-3)
        rid = m["Molecule_Name"].map(idx).to_numpy()
        f = m["Molecule_Name"].map(fold_of).to_numpy()
        narrow = width <= np.quantile(width, 0.25)
        den = np.abs(y - y.mean()).mean()

        for power in POWERS:
            w = np.clip((np.median(width) / width) ** power, 1.0 / CAP, CAP)
            oof = np.full(len(m), np.nan)
            for k in range(5):
                trm, te = f != k, f == k
                if te.sum() == 0:
                    continue
                g = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.04,
                                      num_leaves=31, colsample_bytree=0.3,
                                      subsample=0.8, subsample_freq=1,
                                      min_child_samples=15, verbose=-1, n_jobs=2,
                                      random_state=0)
                g.fit(X[rid[trm]], y[trm], sample_weight=w[trm])
                oof[te] = g.predict(X[rid[te]])
            s = strae(oof, y, lo, hi)
            cost = np.maximum(0, np.maximum(lo - oof, oof - hi))
            print(f"{iso:8s} {power:6.1f} {s:8.4f} {np.abs(oof-y).mean():7.4f} "
                  f"{stats.spearmanr(oof, y)[0]:7.4f} {100*(cost==0).mean():7.1f} "
                  f"{cost[narrow].mean()/den:14.4f}")
            summary.setdefault(power, []).append(s)
        print()

    print(f"{'power':>6s} {'macro ST-RAE':>13s}   (0.0 = current behaviour)")
    base = np.mean(summary[0.0])
    for p in POWERS:
        v = np.mean(summary[p])
        print(f"{p:6.1f} {v:13.4f}   {'baseline' if p == 0 else f'{v-base:+.4f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
