#!/usr/bin/env python
"""Calibrate a Butina cutoff against the real train-to-test gap, then freeze the folds.

    python scripts/make_folds.py --synthetic        # smoke test, no data needed
    python scripts/make_folds.py                    # against the real CSVs
    python scripts/make_folds.py --cutoff 0.5       # skip calibration, force a cutoff

Writes artifacts/folds.csv. Run once. Regenerating folds mid-project makes every
earlier number incomparable, which is why the loader refuses to create them lazily.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit import io, splits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use generated stand-in data")
    ap.add_argument("--synthetic-seed", type=int, default=0,
                    help="seed for the synthetic stand-in; shared across scripts via artifacts/synthetic/")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--cutoff", type=float, default=None, help="skip calibration")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.synthetic:
        d = io.synthetic_dataset(seed=args.synthetic_seed, cache_dir=C.ARTIFACT_DIR / "synthetic")
        train, test = d["train_inhibition"], d["test"]
        print(f"[synthetic] seed={args.synthetic_seed} train={len(train)} test={len(test)} "
              f"(cached in {C.ARTIFACT_DIR / 'synthetic'})")
    else:
        train = io.load("train_inhibition", args.data_dir)
        test = io.load("test", args.data_dir)
        print(f"train={len(train)} test={len(test)}")

    print("\nTraining-set coverage per isoform")
    print(io.describe_coverage(train).to_string(index=False))

    cutoff = args.cutoff
    if cutoff is None:
        print("\nCalibrating Butina cutoff against the real train-to-test similarity gap...")
        table = splits.calibrate_cutoff(
            train[C.SMILES_COL], test[C.SMILES_COL], n_folds=args.n_folds, seed=args.seed
        )
        target = table.attrs["target_median_similarity"]
        print(f"  median max-Tanimoto, test -> train : {target:.3f}   (the gap to reproduce)")
        print(table.to_string(index=False))
        cutoff = float(table.iloc[0]["cutoff"])
        print(f"\n  chosen cutoff = {cutoff}")
    else:
        print(f"\nUsing forced cutoff {cutoff}")

    folds = splits.make_folds(train, cutoff=cutoff, n_folds=args.n_folds, seed=args.seed)
    out = args.out or C.artifact("folds.csv")
    splits.save_folds(folds, out)

    sizes = folds["fold"].value_counts().sort_index()
    print(f"\nfolds written to {out}")
    print(f"  clusters       : {folds['cluster'].nunique()}")
    print(f"  fold sizes     : {sizes.to_dict()}")
    print(f"  held-out median max-Tanimoto : {np.nanmedian(folds['max_sim_to_other_folds']):.3f}")
    per_fold = folds.groupby("fold")["max_sim_to_other_folds"].median().round(3)
    print(f"  per fold       : {per_fold.to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
