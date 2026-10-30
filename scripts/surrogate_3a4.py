#!/usr/bin/env python
r"""Do the 1,244 CYP3A4 compounds that have a TDI-condition pIC50 and no direct measurement
anywhere help a direct-inhibition model?

They are not censored rows. They were never in the direct panel: of the 1,249 such
compounds, 11 appear in cyp-challenge-TRAIN_inhibition.csv and none of those has a CYP3A4
direct value. Their TDI-condition pIC50 has median 5.40, so they are potent under
preincubation, and on the 2,334 compounds where both numbers exist
    D = 0.915 * T + 0.039,  r = 0.947,  residual sd 0.351.

This fits HGB on the real curve rows with and without those surrogate rows, on identical
scaffold folds, and scores both on held-out REAL rows only.

    python scripts/surrogate_3a4.py --folds 5
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd

CACHE = "artifacts/tdi_feat_cache.npz"


def scaffold_folds(scaf, k, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, s in enumerate(scaf):
        groups.setdefault(s if s else f"__s{i}", []).append(i)
    fold = np.zeros(len(scaf), dtype=int); load = np.zeros(k, dtype=int)
    for g in sorted(groups.values(), key=lambda g: (-len(g), rng.random())):
        j = int(load.argmin())
        for i in g:
            fold[i] = j
        load[j] += len(g)
    return fold


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/mnt/user-data/uploads/CYP450/cyp/data/"
                                       "cyp-challenge-TRAIN_TDI.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])
    a = ap.parse_args(argv)
    from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

    df = pd.read_csv(a.train)
    z = np.load(CACHE, allow_pickle=True)
    X, scaf = z["X"], z["scaf"]
    T = df["CYP3A4_pIC50_TDI_condition"].values.astype(float)
    D = df["CYP3A4_pIC50_direct_inhibition"].values.astype(float)
    real = np.where(~np.isnan(T) & ~np.isnan(D))[0]
    surr = np.where(~np.isnan(T) & np.isnan(D))[0]

    slope, icpt = np.polyfit(T[real], D[real], 1)
    resid = (D[real] - (slope * T[real] + icpt)).std()
    Dsurr = slope * T[surr] + icpt
    print(f"real curve rows {len(real)}   surrogate rows {len(surr)}")
    print(f"surrogate target D = {slope:.4f} * T + {icpt:.4f}   residual sd {resid:.4f}")
    print(f"real D: mean {D[real].mean():.3f} sd {D[real].std():.3f} | "
          f"surrogate D: mean {Dsurr.mean():.3f} sd {Dsurr.std():.3f}")

    fold = scaffold_folds(scaf[real], a.folds, a.seed)
    res = {}
    for w in a.weights:
        oof = np.zeros(len(real))
        for f in range(a.folds):
            va = fold == f
            tr = real[~va]
            if w > 0:
                Xtr = np.vstack([X[tr], X[surr]])
                ytr = np.concatenate([D[tr], Dsurr])
                sw = np.concatenate([np.ones(len(tr)), np.full(len(surr), w)])
            else:
                Xtr, ytr, sw = X[tr], D[tr], None
            m = HGBR(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                     l2_regularization=1.0, random_state=a.seed)
            m.fit(Xtr, ytr, sample_weight=sw)
            oof[va] = m.predict(X[real[va]])
        r = np.corrcoef(oof, D[real])[0, 1]
        rmse = float(np.sqrt(np.mean((oof - D[real]) ** 2)))
        from scipy.stats import spearmanr
        rho = spearmanr(oof, D[real]).statistic
        res[w] = (r, rho, rmse, oof.copy())
        tag = "surrogate OFF (control)" if w == 0 else f"surrogate weight {w:.2f}"
        print(f"  {tag:26s} pearson {r:.4f}  spearman {rho:.4f}  rmse {rmse:.4f}")

    base = res[0.0]
    rng = np.random.default_rng(0)
    print("\npaired bootstrap over the held-out real rows, vs the control")
    for w in a.weights:
        if w == 0:
            continue
        d = []
        for _ in range(2000):
            i = rng.integers(0, len(real), len(real))
            a1 = np.sqrt(np.mean((res[w][3][i] - D[real][i]) ** 2))
            a0 = np.sqrt(np.mean((base[3][i] - D[real][i]) ** 2))
            d.append(a1 - a0)
        d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  weight {w:.2f}: RMSE change {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"   P(better) {(d < 0).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
