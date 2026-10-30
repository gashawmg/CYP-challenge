#!/usr/bin/env python
"""Fit the one post-hoc transform ST-RAE actually rewards, then apply it to a submission.

    python scripts/tune_affine.py --oof artifacts/oof_blend_ensA.csv \
                                  --submission artifacts/submission_ensA.csv

ST-RAE is a *relative* error: the denominator is a constant predictor sitting at the
test set's own mean. So the metric does not ask "is this prediction right", it asks
"is this prediction better than the mean, in soft-L1". Two consequences the model's
training loss never sees:

    scale   With Spearman ~0.60 the ranking is only moderately informative. Expanding
            an uninformative prediction away from the mean adds error faster than it
            adds signal; shrinking it toward the mean is the safe direction. The
            optimum is almost never 1.0, and no amount of extra model capacity finds
            it, because the training loss is absolute and this is relative.

    offset  We know the offset is wrong: on 8,424 known-inactive rows the l1 model
            predicts mean 4.48 with only 15.2% below the floor, while being unbiased
            on the actives it was fitted to. Two independent diagnostics put the
            correction at -0.8 to -1.0.

Both are searched per endpoint on the dead-zone loss over the FULL label matrix,
censored rows included — the distribution the blind set was drawn from. Scale is
applied around the OOF mean, which is also what the transform uses at prediction time,
so nothing about the test set leaks in.

This is worth running on any candidate before submitting it. It costs nothing and it
is the part of the score our models were never optimising.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit import io, labels as L, losses, scoring  # noqa: E402

DENSE_A, DENSE_B = 0.776, 0.212
CURRENT_BEST_LB = 0.9171


def dense_strae(p, lo, hi, y, floor=C.PIC50_FLOOR):
    err = (np.where(np.isfinite(hi), np.clip(p - hi, 0, None), 0.0)
           + np.where(np.isfinite(lo), np.clip(lo - p, 0, None), 0.0))
    m = float(np.mean(np.where(np.isfinite(y), y, floor)))
    base = (np.where(np.isfinite(hi), np.clip(m - hi, 0, None), 0.0)
            + np.where(np.isfinite(lo), np.clip(lo - m, 0, None), 0.0))
    d = float(np.sum(base))
    return float(np.sum(err) / d) if d > 0 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", required=True, help="out-of-fold predictions to fit on")
    ap.add_argument("--submission", default=None, help="blind-set file to transform")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scales", type=float, nargs=3, default=(0.5, 1.6, 0.05),
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--offsets", type=float, nargs=3, default=(-1.2, 0.61, 0.05),
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--censored-weight", type=float, default=0.5)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    train = io.load("train_inhibition", args.data_dir)
    screen = io.load("single_conc", args.data_dir)
    lm = L.build_label_matrix(train, screen, include_censored=True,
                              censored_weight=args.censored_weight)
    oof = pd.read_csv(args.oof)

    scales = np.arange(*args.scales)
    offsets = np.arange(*args.offsets)
    fitted, before, after = {}, [], []

    print(f"fitting on {Path(args.oof).name}\n")
    print(f"{'endpoint':34s} {'scale':>6s} {'offset':>7s} "
          f"{'dense before':>13s} {'dense after':>12s}")
    for ep in C.REGRESSION_ENDPOINTS:
        rows = lm[lm["endpoint"] == ep]
        p0 = rows[C.ID_COL].map(dict(zip(oof[C.ID_COL], oof[ep]))).to_numpy(dtype=float)
        keep = np.isfinite(p0)
        rows, p0 = rows[keep], p0[keep]
        lo = rows["y_low"].to_numpy(dtype=float)
        hi = rows["y_high"].to_numpy(dtype=float)
        y = rows["y"].to_numpy(dtype=float)
        mu = float(np.mean(p0))

        base = dense_strae(p0, lo, hi, y)
        best = (base, 1.0, 0.0)
        for s in scales:
            q = mu + s * (p0 - mu)
            for d in offsets:
                v = dense_strae(q + d, lo, hi, y)
                if v < best[0]:
                    best = (v, float(s), float(d))
        v, s, d = best
        fitted[ep] = dict(scale=s, offset=d, centre=mu)
        before.append(base)
        after.append(v)
        print(f"{ep:34s} {s:6.2f} {d:+7.2f} {base:13.4f} {v:12.4f}")

    b, a = float(np.mean(before)), float(np.mean(after))
    print(f"\n  dense MA   {b:.4f}  ->  {a:.4f}")
    print(f"  recalibrated leaderboard estimate  {DENSE_A + DENSE_B*b:.4f}  ->  "
          f"{DENSE_A + DENSE_B*a:.4f}   (current best {CURRENT_BEST_LB:.4f})")

    # Leaderboard-identical scorer on the transformed OOF, for the second opinion.
    t = oof.copy()
    for ep in C.REGRESSION_ENDPOINTS:
        f = fitted[ep]
        t[ep] = f["centre"] + f["scale"] * (oof[ep] - f["centre"]) + f["offset"]
    rep = scoring.oof_report(t, train, n_bootstrap=100)
    ma = rep[rep["Endpoint"] == "MA"].iloc[0]
    ma0 = scoring.oof_report(oof, train, n_bootstrap=100)
    ma0 = ma0[ma0["Endpoint"] == "MA"].iloc[0]
    print(f"  leaderboard-identical scorer (hit-selected rows only) "
          f"{ma0['ST-RAE']:.4f} -> {ma['ST-RAE']:.4f}")
    print("\n  These two disagree by construction: the hit-selected scorer cannot see the"
          "\n  inactives, so it penalises the downward offset the dense proxy rewards."
          "\n  The leaderboard test set was assayed densely. Trust the dense number.")

    with open(C.artifact("affine_calibration.json"), "w") as fh:
        json.dump(fitted, fh, indent=2)

    if args.submission:
        sub = pd.read_csv(args.submission)
        out = sub.copy()
        for ep in C.REGRESSION_ENDPOINTS:
            f = fitted[ep]
            out[ep] = f["centre"] + f["scale"] * (sub[ep] - f["centre"]) + f["offset"]
        path = Path(args.out or C.artifact(
            Path(args.submission).stem + "_affine.csv"))
        out.to_csv(path, index=False)
        print(f"\nwritten {path}")
        cmp = pd.DataFrame({
            "before_mean": [sub[ep].mean() for ep in C.REGRESSION_ENDPOINTS],
            "after_mean": [out[ep].mean() for ep in C.REGRESSION_ENDPOINTS],
            "before_sd": [sub[ep].std() for ep in C.REGRESSION_ENDPOINTS],
            "after_sd": [out[ep].std() for ep in C.REGRESSION_ENDPOINTS],
            "frac_below_floor": [(out[ep] < C.PIC50_FLOOR).mean()
                                 for ep in C.REGRESSION_ENDPOINTS],
        }, index=[e.split("_")[0] for e in C.REGRESSION_ENDPOINTS])
        print(cmp.to_string(float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
