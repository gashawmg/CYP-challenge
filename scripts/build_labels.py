#!/usr/bin/env python
"""Build the censored label matrix and report what it recovers.

    python scripts/build_labels.py --synthetic
    python scripts/build_labels.py --censored-weight 0.5

Writes artifacts/labels.csv (long format) and prints the before/after sparsity picture:
how many (compound, isoform) pairs the dose-response file gives you, and how many the
primary screen adds back as left-censored observations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit import io, labels as L  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-seed", type=int, default=0,
                    help="seed for the synthetic stand-in; shared across scripts via artifacts/synthetic/")
    ap.add_argument("--floor", type=float, default=C.PIC50_FLOOR)
    ap.add_argument("--censored-weight", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--keep-subfloor", nargs="*", default=["CYP3A4"],
                    help="isoforms whose sub-floor fitted curves are KEPT as intervals "
                         "rather than censored. Default CYP3A4 only: it has 944 such "
                         "rows (40.4%) and ChemProp ranks them at rho 0.54, while "
                         "CYP2D6 has 129 and ranks them at -0.04. Pass nothing to "
                         "censor them everywhere (the pre-2026-09-07 behaviour).")
    args = ap.parse_args()

    if args.synthetic:
        d = io.synthetic_dataset(seed=args.synthetic_seed, cache_dir=C.ARTIFACT_DIR / "synthetic")
        train, screen = d["train_inhibition"], d["single_conc"]
        print(f"[synthetic] seed={args.synthetic_seed} "
              f"(cached in {C.ARTIFACT_DIR / 'synthetic'})")
    else:
        train = io.load("train_inhibition", args.data_dir)
        screen = io.load("single_conc", args.data_dir)

    n_compounds = screen[C.ID_COL].nunique()
    print(f"compounds in primary screen : {n_compounds}")
    print(f"compounds with a curve      : {train[C.ID_COL].nunique()}")

    drc_only = L.build_label_matrix(train, screen, floor=args.floor, include_censored=False)
    full = L.build_label_matrix(
        train, screen, floor=args.floor, include_censored=True,
        censored_weight=args.censored_weight,
        keep_subfloor=tuple(args.keep_subfloor or ()),
    )

    print("\nDRC only")
    print(L.summarise(drc_only).to_string(index=False))
    print("\nWith primary-screen censoring")
    print(L.summarise(full).to_string(index=False))

    dense = n_compounds * len(C.ISOFORMS)
    print(
        f"\n(compound, isoform) pairs"
        f"\n  dose-response       : {len(drc_only):>7}   {100*len(drc_only)/dense:5.1f}% of the full matrix"
        f"\n  + censored non-hits : {len(full):>7}   {100*len(full)/dense:5.1f}% of the full matrix"
        f"\n  recovered           : {len(full)-len(drc_only):>7} rows the tutorial discards"
    )

    hit_rate = (
        L.hits_from_screen(screen).groupby("enzyme")["is_hit"].mean().mul(100).round(1)
    )
    print(f"\nprimary-screen hit rate (%) : {hit_rate.to_dict()}")

    # ---- guard: a fitted curve must never be thrown away as censored ----------------
    # For a fifth of the training set (CYP3A4: 40.4%) we used to widen a real fitted
    # value with real bounds to "-inf below 4.0", while `scoring.build_ground_truth`
    # charged ST-RAE against those same bounds. Censoring belongs only to rows with no
    # curve at all. If this ever fires, that mismatch is back.
    keep = set(args.keep_subfloor or ())
    bad = full[(full["kind"] == "left") & (full["source"] == "drc")
               & (full["isoform"].isin(keep))]
    if len(bad):
        raise SystemExit(
            f"REGRESSION: {len(bad)} dose-response rows were left-censored. A fitted "
            f"curve keeps its bounds wherever it sits; only screen non-hits are "
            f"censored. See cypkit/labels.py:clip_interval_at_floor."
        )
    iv = full[full["kind"] == "interval"]
    sub = iv[iv["y"] <= C.PIC50_FLOOR]
    print(f"\ncurves retained below the {C.PIC50_FLOOR} floor : {len(sub):>7} "
          f"(lowest {iv['y'].min():.2f}, mean band width "
          f"{(sub['y_high'] - sub['y_low']).mean():.2f})")
    print(f"  per isoform: "
          f"{sub.groupby('isoform').size().to_dict()}")

    print(f"  sub-floor curves KEPT for: {sorted(keep)}  "
          f"(censored elsewhere -- see cypkit/labels.py:keep_subfloor)")

    out = args.out or C.artifact("labels.csv")
    full.to_csv(out, index=False)
    print(f"\nwritten to {out}")

    with pd.option_context("display.width", 120):
        print("\nsample rows")
        print(full.groupby("kind").head(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
