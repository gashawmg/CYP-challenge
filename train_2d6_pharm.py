#!/usr/bin/env python
r"""Does encoding the CYP2D6 pharmacophore explicitly raise rho?

    python train_2d6_pharm.py                 # CYP2D6, with the CYP3A4 specificity control
    python train_2d6_pharm.py --iso CYP2D6 --no-control

THE LAST UNTESTED CYP2D6 LEVER. Section 22 ruled out data coverage, data quantity and label
quality: proximity-selected external data made rho significantly WORSE, the proximate slice
improved blind coverage while the model degraded, and CYP2D6 has the narrowest intervals
and second-best signal-to-noise of the four endpoints. What remains is that our features do
not encode what a CYP2D6 binder is -- a protonated basic nitrogen salt-bridging Glu216
about 5-7 A from an aromatic centroid. See cypkit/cyp2d6_pharm.py.

THREE ARMS, IDENTICAL FOLDS:

    control     Morgan + descriptors                    <- what every model here uses
    +2D         plus topological pharmacophore features (conformation-free, instant)
    +3D         plus the ETKDG distance features as well

THE SPECIFICITY CONTROL IS THE POINT, NOT A FORMALITY. A feature block of a dozen extra
columns can lift rho simply by giving the trees more to split on. So the same block is run
on CYP3A4, whose active site is large, promiscuous and NOT organised around a cation-anion
pair. If the pharmacophore is doing mechanistic work it helps CYP2D6 and does little for
CYP3A4; if it helps both about equally, it is capacity and not chemistry, and the result
means nothing.

Paired bootstrap on every gap, per Section 19. The CYP2D6 target is +0.131 (Section 21).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C, cyp2d6_pharm as PH  # noqa: E402
from train_pic50_interval import featurise  # noqa: E402

ROOT = Path(__file__).resolve().parent


def fit(Xtr, ytr, Xte):
    import lightgbm as lgb  # noqa: PLC0415
    m = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.035, num_leaves=63,
                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.6, reg_lambda=1.0, verbose=-1, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def boot(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    o = np.asarray([float(pd.Series(b[i]).corr(pd.Series(y[i]), method="spearman"))
                    - float(pd.Series(a[i]).corr(pd.Series(y[i]), method="spearman"))
                    for i in (rng.integers(0, len(y), len(y)) for _ in range(n))])
    return float(np.quantile(o, .025)), float(np.quantile(o, .975)), float((o <= 0).mean())


def run(iso: str) -> dict:
    lab = pd.read_csv(C.artifact("labels.csv"))
    folds = pd.read_csv(C.artifact("folds.csv")); folds.columns = [c.strip() for c in folds.columns]
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))
    t = lab[(lab["isoform"] == iso) & (lab["kind"] != "left")].dropna(subset=["y"])
    t = t[t[C.ID_COL].isin(fold_of)].copy()
    t["fold"] = t[C.ID_COL].map(fold_of); t = t.reset_index(drop=True)
    y = t["y"].to_numpy(float); smi = t[C.SMILES_COL].tolist()

    print(f"\n=== {iso}  {len(t)} rows")
    X0 = featurise(smi)
    P2, _ = PH.build(smi, use_3d=False, verbose=True)
    P3, _ = PH.build(smi, use_3d=True, verbose=True)
    arms = {"control": X0, "+2D": np.hstack([X0, P2]), "+3D": np.hstack([X0, P3])}

    preds = {}
    for name, X in arms.items():
        o = np.zeros(len(t))
        for f in sorted(t["fold"].unique()):
            tr, te = (t["fold"] != f).to_numpy(), (t["fold"] == f).to_numpy()
            o[te] = fit(X[tr], y[tr], X[te])
        preds[name] = o

    base = preds["control"]
    rb = float(pd.Series(base).corr(pd.Series(y), method="spearman"))
    print(f"\n  {'arm':10s} {'rho':>7s} {'gap':>8s} {'95% CI':>20s} {'P(worse)':>9s}")
    out = {}
    for name, o in preds.items():
        r = float(pd.Series(o).corr(pd.Series(y), method="spearman"))
        if name == "control":
            print(f"  {name:10s} {r:7.4f}"); out[name] = 0.0; continue
        lo, hi, pw = boot(y, base, o)
        print(f"  {name:10s} {r:7.4f} {r-rb:+8.4f} [{lo:+.4f}, {hi:+.4f}] {pw:9.0%}")
        out[name] = r - rb
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", default="CYP2D6")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the CYP3A4 specificity control (do not use for a verdict)")
    a = ap.parse_args()

    tgt = run(a.iso)
    if a.no_control:
        return 0
    ctl = run("CYP3A4")
    print("\nVERDICT")
    print(f"  {'arm':10s} {a.iso + ' gap':>14s} {'CYP3A4 gap':>12s} {'ratio':>8s}")
    for k in ("+2D", "+3D"):
        ratio = tgt[k] / ctl[k] if abs(ctl[k]) > 1e-6 else float("inf")
        print(f"  {k:10s} {tgt[k]:+14.4f} {ctl[k]:+12.4f} {ratio:8.2f}")
    best = max(tgt.values())
    if best < 0.02:
        print("  -> the pharmacophore does not move CYP2D6. The representation hypothesis "
              "fails\n     on these features; a learned 3D representation is the only thing left.")
    elif tgt["+2D"] > 2 * max(ctl["+2D"], 1e-6) or tgt["+3D"] > 2 * max(ctl["+3D"], 1e-6):
        print("  -> helps CYP2D6 specifically, not CYP3A4: this is chemistry, not capacity.")
    else:
        print("  -> helps both about equally: extra split capacity, not the pharmacophore. "
              "Do not ship it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
