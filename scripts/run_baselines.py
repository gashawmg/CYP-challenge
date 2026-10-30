#!/usr/bin/env python
"""Baseline ladder, all scored with the challenge's own ST-RAE on frozen folds.

    python scripts/run_baselines.py --synthetic --fast
    python scripts/run_baselines.py

Four variants, each a single change from the one above it:

    tutorial   LightGBM defaults (L2) on the point estimate, curve rows only
    l1         same, MAE objective                     -- is L1 the right shape?
    deadzone   dead-zone objective on the credible band -- does the metric's own
                                                          shape help?
    censored_imputed  non-hits imputed at the floor, L1 -- the obvious way to use them
    censored   dead-zone + non-hits as *left-censored* rows
                                                       -- the main hypothesis

Every variant predicts every compound for every isoform; the scorer drops compounds with
no measurement per endpoint, exactly as the leaderboard does.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit import cyp_descriptors as CD  # noqa: E402
from cypkit import features, io, labels as L, losses, scoring, splits  # noqa: E402

VARIANTS = ("tutorial", "l1", "deadzone", "censored_imputed", "censored")


BLOCKS = ("rdkit", "cyp", "ecfp", "ecfpc", "fcfp", "maccs", "ap", "3d")


def build_features(
    smiles_by_id: dict[str, str], include_3d: bool = False, kind: str = "rdkit+cyp"
) -> tuple[list[str], np.ndarray, list[str]]:
    """Assemble the feature matrix from a '+'-separated list of blocks.

    e.g. --features rdkit+cyp+ecfp . Each block's contribution is measurable rather than
    assumed, which is the only reason to keep them separable.
    """
    ids = sorted(smiles_by_id)
    smi = [smiles_by_id[i] for i in ids]
    wanted = [b for b in kind.split("+") if b]
    if kind == "both":  # backwards compatibility with the earlier flag
        wanted = ["rdkit", "cyp"]
    bad = [b for b in wanted if b not in BLOCKS]
    if bad:
        raise ValueError(f"unknown feature block(s) {bad}; choose from {BLOCKS}")
    if include_3d and "3d" not in wanted:
        wanted.append("3d")

    blocks, names = [], []
    valid = np.ones(len(ids), dtype=bool)
    for b in wanted:
        if b == "rdkit":
            X, v = features.rdkit_2d(smi)
            names += features.descriptor_names()
        elif b == "cyp":
            X, v = CD.calculate(smi)
            names += CD.feature_names()
        elif b == "3d":
            X, v = features.rdkit_3d(smi)
            X = np.nan_to_num(X, nan=0.0)
            names += list(features._3D_NAMES)
        else:
            X, v, nm = features.fingerprint_block(smi, b)
            names += nm
        blocks.append(X)
        valid = valid & v

    X = np.hstack(blocks)
    if not valid.all():
        print(f"  warning: {int((~valid).sum())} compound(s) failed featurisation")
    n_varying = int((X.std(axis=0) > 0).sum())
    if n_varying == 0:
        raise RuntimeError(
            "The feature matrix has no varying columns — every model would just predict "
            "the training mean. Run `python scripts/check_env.py` to find out why."
        )
    return ids, X, names


def fit_one(variant, Xtr, rows, Xte, params, delta):
    """Train one isoform-fold model and return predictions for Xte."""
    if variant in ("tutorial", "l1", "censored_imputed"):
        y = rows["y"].to_numpy(dtype=float)
        if variant == "censored_imputed":
            # The obvious thing to do with a non-hit: pretend it was measured at the
            # floor. Included so the dead-zone objective has to earn its place against
            # it rather than against no censoring at all.
            y = np.where(np.isfinite(y), y, rows["y_high"].to_numpy(dtype=float))
        keep = np.isfinite(y)
        if keep.sum() < 10:
            return None
        p = dict(params)
        p["objective"] = "regression" if variant == "tutorial" else "regression_l1"
        model = lgb.train(p, lgb.Dataset(Xtr[keep], label=y[keep]), num_boost_round=p.pop("n_rounds"))
        return model.predict(Xte)

    lo = rows["y_low"].to_numpy(dtype=float)
    hi = rows["y_high"].to_numpy(dtype=float)
    w = rows["weight"].to_numpy(dtype=float)
    if len(lo) < 10:
        return None
    # Start from the median measured potency. Censored rows have no point estimate, so
    # letting them into the initial value would drag it down to the floor before the
    # first tree is grown.
    measured = rows["y"].to_numpy(dtype=float)
    measured = measured[np.isfinite(measured)]
    init = float(np.median(measured)) if len(measured) else float(np.nanmedian(hi))
    p = dict(params)
    n_rounds = p.pop("n_rounds")
    p["objective"] = None
    ds = lgb.Dataset(Xtr, label=np.zeros(len(lo)), init_score=np.full(len(lo), init))
    model = lgb.train(
        p, ds, num_boost_round=n_rounds,
        fobj=losses.make_lgb_objective(lo, hi, delta=delta, weight=w),
    ) if _lgb_takes_fobj() else lgb.train(
        {**p, "objective": losses.make_lgb_objective(lo, hi, delta=delta, weight=w)},
        ds, num_boost_round=n_rounds,
    )
    return model.predict(Xte) + init


def _lgb_takes_fobj() -> bool:
    import inspect  # noqa: PLC0415
    return "fobj" in inspect.signature(lgb.train).parameters


def run_variant(variant, label_matrix, fold_of, ids, X, index_of, params, delta, n_folds):
    """Out-of-fold predictions for one variant: rows = compounds, cols = endpoints."""
    oof = pd.DataFrame({C.ID_COL: ids})
    use_censored = variant.startswith("censored")

    for cyp in C.ISOFORMS:
        ep = f"{cyp}_pIC50_direct_inhibition"
        sub = label_matrix[label_matrix["isoform"] == cyp]
        if not use_censored:
            sub = sub[sub["source"] == "drc"]
        sub = sub[sub[C.ID_COL].isin(index_of)]
        preds = np.full(len(ids), np.nan)

        for f in range(n_folds):
            tr = sub[sub[C.ID_COL].map(fold_of) != f]
            if len(tr) < 10:
                continue
            tr_idx = tr[C.ID_COL].map(index_of).to_numpy()
            te_mask = np.array([fold_of.get(i, -1) == f for i in ids])
            if not te_mask.any():
                continue
            out = fit_one(variant, X[tr_idx], tr, X[te_mask], params, delta)
            if out is not None:
                preds[te_mask] = out

        # A compound in no fold, or an isoform with too little data, falls back to the
        # training median rather than NaN — the scorer rejects missing predictions.
        if np.isnan(preds).any():
            fill = float(np.nanmedian(sub["y"])) if sub["y"].notna().any() else C.PIC50_FLOOR
            preds = np.where(np.isnan(preds), fill, preds)
        oof[ep] = preds
    return oof


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-seed", type=int, default=0,
                    help="seed for the synthetic stand-in; shared across scripts via artifacts/synthetic/")
    ap.add_argument("--fast", action="store_true", help="100 bootstrap samples instead of 1000")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--delta", type=float, default=0.25, help="dead-zone Huber width")
    ap.add_argument("--censored-weight", type=float, default=0.5)
    ap.add_argument("--n-rounds", type=int, default=300)
    ap.add_argument("--dz-learning-rate", type=float, default=0.20)
    ap.add_argument("--dz-round-multiplier", type=float, default=2.0)
    ap.add_argument("--features", default="rdkit+cyp",
                    help="'+'-joined blocks: rdkit, cyp, ecfp, ecfpc, fcfp, maccs, ap, 3d")
    ap.add_argument("--include-3d", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for artifact filenames")
    ap.add_argument("--folds", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.synthetic:
        d = io.synthetic_dataset(seed=args.synthetic_seed, cache_dir=C.ARTIFACT_DIR / "synthetic")
        train, screen = d["train_inhibition"], d["single_conc"]
        print(f"[synthetic] seed={args.synthetic_seed} "
              f"(cached in {C.ARTIFACT_DIR / 'synthetic'})")
    else:
        train = io.load("train_inhibition", args.data_dir)
        screen = io.load("single_conc", args.data_dir)

    folds = splits.load_folds(args.folds or C.artifact("folds.csv"))
    n_folds = int(folds["fold"].max()) + 1
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))

    label_matrix = L.build_label_matrix(
        train, screen, include_censored=True, censored_weight=args.censored_weight
    )

    # Screen-only compounds need a fold too, or their analogues leak across the split.
    extra = sorted(set(label_matrix[C.ID_COL]) - set(fold_of))
    if extra:
        smi = dict(zip(label_matrix[C.ID_COL], label_matrix[C.SMILES_COL]))
        print(f"assigning {len(extra)} screen-only compounds to folds by nearest neighbour...")
        assigned = splits.assign_extra_to_folds([smi[i] for i in extra], folds)
        fold_of.update(dict(zip(extra, assigned)))

    smiles_by_id = dict(zip(label_matrix[C.ID_COL], label_matrix[C.SMILES_COL]))
    smiles_by_id.update(dict(zip(train[C.ID_COL], train[C.SMILES_COL])))
    smiles_by_id = {k: v for k, v in smiles_by_id.items() if isinstance(v, str)}

    print(f"featurising {len(smiles_by_id)} compounds...")
    t0 = time.time()
    ids, X, feat_names = build_features(smiles_by_id, args.include_3d, args.features)
    index_of = {m: i for i, m in enumerate(ids)}
    print(f"  X = {X.shape}  [{args.features}]  ({time.time()-t0:.1f}s)")

    print("measuring test-to-train similarity (this is what the folds must match)...")
    test = io.synthetic_dataset(seed=args.synthetic_seed,
                                cache_dir=C.ARTIFACT_DIR / "synthetic")["test"] \
        if args.synthetic else io.load("test", args.data_dir)
    test_sim = splits.max_similarity(
        splits.fingerprints(test[C.SMILES_COL]), splits.fingerprints(train[C.SMILES_COL])
    )
    print(f"  median test -> train max-Tanimoto = {np.nanmedian(test_sim):.3f}; "
          f"held-out median = {folds['max_sim_to_other_folds'].median():.3f}")

    params = dict(
        verbose=-1, learning_rate=0.05, num_leaves=63, min_data_in_leaf=10,
        feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
        n_rounds=args.n_rounds, seed=42,
    )
    n_boot = 100 if args.fast else None

    results, dense, sim = {}, {}, {}
    for variant in args.variants:
        print(f"\n=== {variant} ===")
        t0 = time.time()
        p = dict(params)
        if variant in ("deadzone", "censored"):
            # The dead-zone gradient is bounded by 1, so leaf values are far smaller
            # than an L2 objective's. Same nominal rounds would badly underfit.
            p["learning_rate"] = args.dz_learning_rate
            p["n_rounds"] = int(args.n_rounds * args.dz_round_multiplier)
        oof = run_variant(
            variant, label_matrix, fold_of, ids, X, index_of, p, args.delta, n_folds
        )
        oof.to_csv(C.artifact(f"oof_{variant}{args.tag}.csv"), index=False)

        summary = scoring.oof_report(oof, train, n_bootstrap=n_boot)
        summary.insert(0, "variant", variant)
        results[variant] = summary
        print(scoring.format_report(summary))

        if variant == args.variants[0]:
            diff = scoring.verify_strae(oof, train, C.REGRESSION_ENDPOINTS[0])
            print(f"  [local ST-RAE matches the challenge's to {diff:.2e}]")
        d = scoring.dense_proxy_report(oof, label_matrix)
        d.insert(0, "variant", variant)
        dense[variant] = d
        print("\n  dense proxy (censored rows included)")
        print("  " + d.to_string(index=False, float_format=lambda x: f"{x:.4f}").replace("\n", "\n  "))
        by_bin, matched = scoring.similarity_report(oof, train, folds, test_sim)
        sim[variant] = matched
        if variant == args.variants[0]:
            print("\n  ST-RAE by held-out similarity band (test_share = how much of the")
            print("  real test set sits in that band)")
            piv = by_bin.pivot_table(index="bin", columns="Endpoint", values="ST_RAE")
            share = by_bin.groupby("bin")["test_share"].first().round(3)
            piv.insert(0, "test_share", share)
            print("  " + piv.to_string(float_format=lambda x: f"{x:.3f}").replace("\n", "\n  "))
        m = matched[matched["Endpoint"] == "MA"].iloc[0]
        print(f"\n  similarity-matched MA-ST-RAE = {m['matched_STRAE']:.4f} "
              f"[{m['lo']:.4f}, {m['hi']:.4f}]")
        print(f"({time.time()-t0:.1f}s)")

    if len(results) > 1:
        comp = pd.concat(results.values())
        dcomp = pd.concat(dense.values())
        ma = comp[comp["Endpoint"] == "MA"][
            ["variant", "ST-RAE", "ST-RAE_lo", "ST-RAE_hi", "MAE", "Spearman_R"]
        ]
        dma = dcomp[dcomp["Endpoint"] == "MA"][["variant", "dense_STRAE", "in_band"]]
        scomp = pd.concat({k: v for k, v in sim.items()}, names=["variant"]).reset_index(level=0)
        sma = scomp[scomp["Endpoint"] == "MA"][["variant", "matched_STRAE"]]
        merged = ma.merge(sma, on="variant").merge(dma, on="variant")

        print("\n=== summary (lower ST-RAE is better) ===")
        print(merged.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        b1 = merged.sort_values("ST-RAE").iloc[0]["variant"]
        b2 = merged.sort_values("dense_STRAE").iloc[0]["variant"]
        print(f"\nbest on the leaderboard metric (hit-selected compounds only) : {b1}")
        print(f"best on the dense proxy (what the real test set looks like)   : {b2}")
        if b1 != b2:
            print(
                "\nThese disagree, which is the point: the leaderboard-identical scorer can\n"
                "only score compounds that had a curve, and a compound only had a curve\n"
                "because it hit the screen. It cannot see the inactives the test set is full\n"
                "of. Track both; trust the dense proxy for anything about censoring."
            )
        comp.to_csv(C.artifact(f"baseline_summary{args.tag}.csv"), index=False)
        dcomp.to_csv(C.artifact(f"baseline_dense_summary{args.tag}.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
