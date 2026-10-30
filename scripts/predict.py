#!/usr/bin/env python
"""Train on all the data and predict the 750 blind test compounds.

    python scripts/predict.py --variants l1 censored_imputed
    python scripts/predict.py --variants l1 --features both --n-rounds 300

Writes artifacts/submission_<variant>.csv in the challenge's required format and runs
the repo's own validator against it. Prints a distribution comparison against the
training set, because the most likely failure here is silent: a model that predicts a
plausible-looking but systematically shifted range.

No cross-validation — these models see every labelled row. Use `run_baselines.py` for
anything you intend to compare.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit import io, labels as L, splits  # noqa: E402
from scripts.run_baselines import build_features, fit_one  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=["l1", "censored_imputed"])
    ap.add_argument("--features", choices=("rdkit", "cyp", "both"), default="both")
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--censored-weight", type=float, default=0.5)
    ap.add_argument("--n-rounds", type=int, default=300)
    ap.add_argument("--dz-learning-rate", type=float, default=0.20)
    ap.add_argument("--dz-round-multiplier", type=float, default=2.0)
    ap.add_argument("--include-3d", action="store_true")
    ap.add_argument("--no-clip", action="store_true",
                    help="skip clipping predictions to the observed training range")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    train = io.load("train_inhibition", args.data_dir)
    screen = io.load("single_conc", args.data_dir)
    test = io.load("test", args.data_dir)
    print(f"train={len(train)}  test={len(test)}")

    label_matrix = L.build_label_matrix(
        train, screen, include_censored=True, censored_weight=args.censored_weight
    )

    smiles_by_id = dict(zip(label_matrix[C.ID_COL], label_matrix[C.SMILES_COL]))
    smiles_by_id.update(dict(zip(train[C.ID_COL], train[C.SMILES_COL])))
    smiles_by_id.update(dict(zip(test[C.ID_COL], test[C.SMILES_COL])))
    smiles_by_id = {k: v for k, v in smiles_by_id.items() if isinstance(v, str)}

    print(f"featurising {len(smiles_by_id)} compounds (train + screen + test)...")
    t0 = time.time()
    ids, X, _ = build_features(smiles_by_id, args.include_3d, args.features)
    index_of = {m: i for i, m in enumerate(ids)}
    print(f"  X = {X.shape}  [{args.features}]  ({time.time()-t0:.1f}s)")

    missing = [m for m in test[C.ID_COL] if m not in index_of]
    if missing:
        raise RuntimeError(f"{len(missing)} test compound(s) failed featurisation: {missing[:5]}")
    test_idx = test[C.ID_COL].map(index_of).to_numpy()

    base = dict(
        verbose=-1, learning_rate=0.05, num_leaves=63, min_data_in_leaf=10,
        feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
        n_rounds=args.n_rounds, seed=42,
    )

    for variant in args.variants:
        print(f"\n=== {variant} ===")
        p = dict(base)
        if variant in ("deadzone", "censored"):
            p["learning_rate"] = args.dz_learning_rate
            p["n_rounds"] = int(args.n_rounds * args.dz_round_multiplier)

        out = test[[C.SMILES_COL, C.ID_COL]].copy()
        rows = []
        for cyp in C.ISOFORMS:
            ep = f"{cyp}_pIC50_direct_inhibition"
            sub = label_matrix[label_matrix["isoform"] == cyp]
            if not variant.startswith("censored"):
                sub = sub[sub["source"] == "drc"]
            sub = sub[sub[C.ID_COL].isin(index_of)]
            tr_idx = sub[C.ID_COL].map(index_of).to_numpy()

            pred = fit_one(variant, X[tr_idx], sub, X[test_idx], p, args.delta)
            if pred is None:
                raise RuntimeError(f"no model trained for {ep}")

            measured = sub.loc[sub["y"].notna(), "y"]
            lo_obs, hi_obs = float(measured.min()), float(measured.max())
            n_clipped = int(((pred < lo_obs) | (pred > hi_obs)).sum())
            if not args.no_clip:
                pred = np.clip(pred, lo_obs, hi_obs)

            out[ep] = pred
            rows.append(
                dict(
                    endpoint=ep,
                    n_train=len(sub),
                    train_mean=float(measured.mean()),
                    train_sd=float(measured.std()),
                    pred_mean=float(pred.mean()),
                    pred_sd=float(pred.std()),
                    pred_min=float(pred.min()),
                    pred_max=float(pred.max()),
                    frac_below_floor=float((pred < C.PIC50_FLOOR).mean()),
                    n_clipped=n_clipped,
                )
            )

        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        path = C.artifact(f"submission_{variant}.csv")
        out.to_csv(path, index=False)
        print(f"\nwritten to {path}  ({len(out)} rows)")

        ok, errors = _validate(path, set(test[C.ID_COL]))
        if ok:
            print("  validator: PASS")
        else:
            print("  validator: FAIL")
            for e in errors:
                print(f"    - {e}")

    print(
        "\nSanity notes:"
        "\n  - pred_sd well below train_sd means the model is hedging toward the mean;"
        "\n    under ST-RAE that is cheap on weak compounds and expensive on potent ones."
        "\n  - frac_below_floor should be substantial for censored variants and near zero"
        "\n    for the curve-only ones. If they look alike, the censoring did not take."
    )
    return 0


def _validate(path: Path, expected_ids: set[str]):
    sys.path.insert(0, str(C.CHALLENGE_REPO))
    try:
        from validation.activity_validation import validate_activity_submission  # noqa: PLC0415
    except ImportError as exc:
        return False, [f"could not import the challenge validator: {exc}"]
    return validate_activity_submission(path, expected_ids=expected_ids)


if __name__ == "__main__":
    raise SystemExit(main())
