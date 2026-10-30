#!/usr/bin/env python
"""Ask whether two OOF files DISAGREE usefully, before spending a submission on either.

    python scripts/blend_probe.py --iso CYP2D6 \
        oof_st_v5_2d6.csv artifacts/oof_pseudo_blend45.csv

WHY. Two models that rank at 0.464 and 0.480 are, individually, the same model. But if
their rankings agree at only 0.85, their errors are partly independent, and the average
of two 0.47 rankings can land above either -- that is the one place left where the model
track can still gain without new chemistry. If they agree at 0.97 they are one model
written twice and the blend is a rounding error. This costs nothing to check and settles
which of those two worlds we are in.

WHAT IT DOES. Standardises each column (blending raw pIC50 columns with different centres
and spreads is just a weighted shift), sweeps the mixing weight, and scores every mix in
the blind-matched validator so the numbers are comparable to everything else in the
project. `rho` is the only column to read: ST-RAE at this stage is dominated by dispersion,
and dispersion is what the affine map supplies for free.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from cypkit import config as C
from blind_validator import eval_pool, score, load_pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--iso", default="CYP2D6")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--weights", nargs="*", type=float,
                    default=[0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0],
                    help="weight on the FIRST file")
    ap.add_argument("--out", default=None, help="write the best blend as an OOF csv")
    ap.add_argument("--equal-all", action="store_true",
                    help="also score the equal-weight blend of every file given")
    args = ap.parse_args()

    ep = f"{args.iso}_pIC50_direct_inhibition"
    cols = {}
    for f in args.files:
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        if ep not in d.columns:
            print(f"  {f}: no {ep}, skipped"); continue
        cols[Path(f).name] = pd.Series(d[ep].to_numpy(float), index=d[C.ID_COL])
    if len(cols) < 2:
        raise SystemExit("need at least two files carrying that endpoint")

    names = list(cols)
    M = pd.DataFrame(cols).dropna()
    print(f"{len(M)} compounds shared across {len(names)} files\n")

    print("pairwise agreement (Spearman) -- 0.97+ means one model written twice:")
    r = M.rank().corr()
    print(r.round(4).to_string(), "\n")

    Z = (M - M.mean()) / M.std()
    pool = eval_pool()
    a, b = names[0], names[1]
    print(f"blend sweep: w * {a} + (1-w) * {b}\n")
    print(f"  {'w':>5s} {'rho':>17s} {'ST-RAE':>17s}")
    best, rows = None, []
    for w in args.weights:
        z = w * Z[a] + (1 - w) * Z[b]
        res = score({ep: dict(zip(z.index, z.to_numpy()))}, pool, args.draws)
        if res.empty:
            continue
        x = res.iloc[0]
        rows.append((w, float(x.rho), float(x.strae)))
        print(f"  {w:5.2f} {x.rho:8.4f}+-{x.rho_sd:.4f} {x.strae:8.4f}+-{x.strae_sd:.4f}")
        if best is None or x.rho > best[1]:
            best = (w, float(x.rho), z)

    w, rho, z = best
    print(f"\nbest w = {w:.2f}  rho {rho:.4f}")
    ends = [x[1] for x in rows if x[0] in (0.0, 1.0)]
    if ends:
        print(f"  best single file: {max(ends):.4f}   blend edge {rho - max(ends):+.4f}")
        print("  (an edge smaller than the +- above is not an edge)")

    if len(names) >= 3 and args.equal_all:
        za = Z[names].mean(axis=1)
        res = score({ep: dict(zip(za.index, za.to_numpy()))}, pool, args.draws)
        x = res.iloc[0]
        print(f"\nequal-weight blend of all {len(names)} files: "
              f"rho {x.rho:.4f}+-{x.rho_sd:.4f}")
        if x.rho > rho:
            best, z = (None, float(x.rho), za), za
            rho = float(x.rho)

    if args.out:
        # write back on the ORIGINAL scale of the better single file, so the result
        # drops into the existing pipeline without a second calibration step
        ref = M[a] if w >= 0.5 else M[b]
        q = ref.mean() + z * ref.std()
        pd.DataFrame({C.ID_COL: q.index, ep: q.to_numpy()}).to_csv(args.out, index=False)
        print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
