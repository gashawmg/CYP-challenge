#!/usr/bin/env python
r"""Does Uni-Mol's 3D embedding, used as tree features, raise rho?

    python scripts\unimol_embed.py --extract --weights <path>    # once, on the GPU box
    python train_unimol_feat.py                                  # then this, anywhere

FIVE ARMS PER ENDPOINT, identical folds, identical booster:

    control        Morgan + descriptors                      <- what every model here uses
    +unimol        plus the Uni-Mol CLS embedding
    unimol only    the embedding INSTEAD of Morgan+descriptors
    +pharm         plus the CYP2D6 pharmacophore block (Section 23)
    +unimol+pharm  both

"unimol only" is in there because it is the question the +unimol arm cannot answer. If
adding the embedding helps, that could mean it carries geometry the 2D features lack, or
merely that more columns help. If the embedding ALONE is competitive, it is carrying real
structure; if it collapses alone and only helps when bolted on, it is decoration.

RUN ON ALL FOUR ENDPOINTS. Section 23's pharmacophore result only became interpretable next to a
CYP3A4 control, and the same applies here: a 3D embedding should help the two small,
shape-constrained sites (CYP2D6, CYP2C9) more than the large promiscuous one (CYP3A4). If
it helps everything equally it is capacity, not geometry.

THE BAR, per Section 19: a gap smaller than its own confidence interval is not a result, and the
CYP2D6 target is +0.131 (Section 21). Nothing here justifies a submission on its own.
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
REPR = ROOT / "artifacts" / "unimol_repr.npz"


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", default="all")
    a = ap.parse_args()

    if not REPR.exists():
        raise SystemExit(f"{REPR.name} not found -- run scripts/unimol_embed.py --extract "
                         f"on the GPU box first")
    z = np.load(REPR, allow_pickle=True)
    E = {m: v for m, v in zip(z["ids"], z["X"])}
    print(f"  Uni-Mol embedding: {z['X'].shape[1]} dims for {len(E)} molecules")

    lab = pd.read_csv(C.artifact("labels.csv"))
    folds = pd.read_csv(C.artifact("folds.csv")); folds.columns = [c.strip() for c in folds.columns]
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))
    isos = C.ISOFORMS if a.iso == "all" else (a.iso,)
    summary = []

    for iso in isos:
        t = lab[(lab["isoform"] == iso) & (lab["kind"] != "left")].dropna(subset=["y"])
        t = t[t[C.ID_COL].isin(fold_of) & t[C.ID_COL].isin(E)].copy()
        t["fold"] = t[C.ID_COL].map(fold_of); t = t.reset_index(drop=True)
        y = t["y"].to_numpy(float); smi = t[C.SMILES_COL].tolist()
        X0 = featurise(smi)
        U = np.vstack([E[m] for m in t[C.ID_COL]])
        P, _ = PH.build(smi, use_3d=False)
        arms = {"control": X0, "+unimol": np.hstack([X0, U]), "unimol only": U,
                "+pharm": np.hstack([X0, P]), "+unimol+pharm": np.hstack([X0, U, P])}

        print(f"\n=== {iso}  {len(t)} rows")
        preds = {}
        for name, X in arms.items():
            o = np.zeros(len(t))
            for f in sorted(t["fold"].unique()):
                tr, te = (t["fold"] != f).to_numpy(), (t["fold"] == f).to_numpy()
                o[te] = fit(X[tr], y[tr], X[te])
            preds[name] = o
        base = preds["control"]
        rb = float(pd.Series(base).corr(pd.Series(y), method="spearman"))
        print(f"  {'arm':15s} {'rho':>7s} {'gap':>8s} {'95% CI':>20s} {'P(worse)':>9s}")
        for name, o in preds.items():
            r = float(pd.Series(o).corr(pd.Series(y), method="spearman"))
            if name == "control":
                print(f"  {name:15s} {r:7.4f}")
                summary.append(dict(isoform=iso, arm=name, rho=r, gap=0.0)); continue
            lo, hi, pw = boot(y, base, o)
            print(f"  {name:15s} {r:7.4f} {r-rb:+8.4f} [{lo:+.4f}, {hi:+.4f}] {pw:9.0%}")
            summary.append(dict(isoform=iso, arm=name, rho=r, gap=r - rb,
                                lo=lo, hi=hi, p_worse=pw))

    s = pd.DataFrame(summary)
    s.to_csv(ROOT / "artifacts" / "unimol_feat_summary.csv", index=False)
    if len(isos) > 1:
        print("\nGEOMETRY SHOULD HELP THE SMALL SITES MORE THAN THE BIG ONE")
        p = s[s.arm != "control"].pivot_table(index="arm", columns="isoform", values="gap")
        print(p.round(4).to_string())
        if "CYP2D6" in p.columns and "CYP3A4" in p.columns:
            best = p["CYP2D6"].idxmax()
            print(f"\n  best CYP2D6 arm: {best} at {p.loc[best,'CYP2D6']:+.4f}, "
                  f"against {p.loc[best,'CYP3A4']:+.4f} on CYP3A4")
            print("  target for CYP2D6 is +0.131 (Section 21)")
    print("\n  wrote artifacts/unimol_feat_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
