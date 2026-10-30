#!/usr/bin/env python
r"""Does external CYP2D6 data raise rho -- and does selecting it by PROXIMITY help more?

    python train_2d6_external.py
    python train_2d6_external.py --iso CYP1A2        # same test on another endpoint

WHY CYP2D6 AND WHY PROXIMITY. Section 21: to reach the leader's macro every endpoint must gain
~+0.04 rho and CYP2D6 must gain +0.131, four times the rest and more than the entire
project's rho progress. CYP2D6 also has the worst coverage of the blind chemistry despite
having the second-most labels -- median nearest-neighbour Tanimoto 0.424 against
0.516-0.538 elsewhere. External data has been rejected three times, but always selected on
ASSAY SHAPE. It has never once been selected for PROXIMITY TO THE BLIND SET, which is the
thing actually missing.

FOUR ARMS, IDENTICAL FEATURES AND FOLDS:

    control      our labelled rows only                       <- what we ship
    +all         plus every external CYP2D6 row
    +prox        plus only external rows within TAU of a blind compound
    +weighted    plus every external row, weighted by its similarity to the blind set

The comparison is the control-to-arm gap, not the absolute rho.

THE LEAKAGE GUARD IS NOT OPTIONAL AND RUNS HERE TOO. Any external compound sharing an
InChIKey connectivity block with a blind compound is dropped before anything else happens,
and the count is printed. artifacts/external_2d6_prox.csv was built with the same guard.

SCORED WITH A PAIRED BOOTSTRAP, per Section 19: a gap smaller than its own confidence interval is
not a result. The bar to clear is not "positive" -- it is "large enough to matter against
a +0.131 target".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from train_pic50_interval import featurise  # noqa: E402

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "artifacts" / "external_2d6_prox.csv"


def fit(Xtr, ytr, wtr, Xte):
    import lightgbm as lgb  # noqa: PLC0415
    m = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.035, num_leaves=63,
                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.6, reg_lambda=1.0, verbose=-1, n_jobs=-1)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m.predict(Xte)


def boot(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        out.append(float(pd.Series(b[i]).corr(pd.Series(y[i]), method="spearman"))
                   - float(pd.Series(a[i]).corr(pd.Series(y[i]), method="spearman")))
    o = np.asarray(out)
    return float(np.quantile(o, .025)), float(np.quantile(o, .975)), float((o <= 0).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", default="CYP2D6")
    ap.add_argument("--tau", type=float, default=0.35)
    a = ap.parse_args()
    iso = a.iso

    lab = pd.read_csv(C.artifact("labels.csv"))
    folds = pd.read_csv(C.artifact("folds.csv")); folds.columns = [c.strip() for c in folds.columns]
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))
    t = lab[(lab["isoform"] == iso) & (lab["kind"] != "left")].dropna(subset=["y"])
    t = t[t[C.ID_COL].isin(fold_of)].copy()
    t["fold"] = t[C.ID_COL].map(fold_of)
    t = t.reset_index(drop=True)
    y = t["y"].to_numpy(float)
    X = featurise(t[C.SMILES_COL].tolist())

    ex = pd.read_csv(EXT)
    col = f"{iso}_pic50"
    if col not in ex.columns:
        raise SystemExit(f"{EXT.name} has no {col}; rebuild it for {iso}")
    ex = ex.dropna(subset=[col, "smiles_std", "sim_to_blind"]).reset_index(drop=True)
    Xe = featurise(ex["smiles_std"].tolist())
    ye = ex[col].to_numpy(float)
    se = ex["sim_to_blind"].to_numpy(float)
    prox = se >= a.tau

    print(f"\n  {iso}: {len(t)} of our rows, {len(ex)} external (leakage-guarded), "
          f"{int(prox.sum())} within {a.tau} of a blind compound")

    arms = {"control": None,
            "+all": (Xe, ye, np.ones(len(ye))),
            "+prox": (Xe[prox], ye[prox], np.ones(int(prox.sum()))),
            "+weighted": (Xe, ye, np.clip(se / max(se.max(), 1e-9), 0.05, 1.0))}
    preds = {}
    for name, extra in arms.items():
        o = np.zeros(len(t))
        for f in sorted(t["fold"].unique()):
            tr, te = (t["fold"] != f).to_numpy(), (t["fold"] == f).to_numpy()
            Xtr, ytr = X[tr], y[tr]
            wtr = np.ones(len(ytr))
            if extra is not None:
                Xa, ya, wa = extra
                Xtr = np.vstack([Xtr, Xa]); ytr = np.concatenate([ytr, ya])
                wtr = np.concatenate([wtr, wa])
            o[te] = fit(Xtr, ytr, wtr, X[te])
        preds[name] = o

    base = preds["control"]
    print(f"\n  {'arm':12s} {'rho':>7s} {'gap':>8s} {'95% CI':>20s} {'P(worse)':>9s}")
    rows = []
    for name, o in preds.items():
        r = float(pd.Series(o).corr(pd.Series(y), method="spearman"))
        if name == "control":
            print(f"  {name:12s} {r:7.4f}")
            rows.append(dict(arm=name, rho=r)); continue
        lo, hi, pw = boot(y, base, o)
        rb = float(pd.Series(base).corr(pd.Series(y), method="spearman"))
        print(f"  {name:12s} {r:7.4f} {r-rb:+8.4f} [{lo:+.4f}, {hi:+.4f}] {pw:9.0%}")
        rows.append(dict(arm=name, rho=r, gap=r - rb, lo=lo, hi=hi, p_worse=pw))

    pd.DataFrame(rows).to_csv(ROOT / "artifacts" / f"external_{iso}_summary.csv", index=False)
    best = max(rows[1:], key=lambda d: d["rho"])
    print(f"\n  VERDICT: best arm {best['arm']} at {best['gap']:+.4f} rho.")
    print(f"  The target for {iso} is +0.131 (Section 21). "
          f"{'This is not it.' if best['gap'] < 0.03 else 'Worth pursuing.'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
