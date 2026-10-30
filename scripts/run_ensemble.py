#!/usr/bin/env python
"""Track A — the diverse classical-ML ensemble, out-of-fold and then on the blind set.

    python scripts/run_ensemble.py --fast --learners lgb_dz xgb_dz knn_tan
    python scripts/run_ensemble.py --predict

What it does, in order:

  1. One featurisation pass over train + screen + test, sliced per learner.
  2. Out-of-fold predictions for every learner on the frozen folds, scored three ways:
     the challenge's own ST-RAE (hit-selected rows only), the dense proxy (censored
     rows included), and the 2-point-recalibrated leaderboard predictor.
  3. A per-endpoint non-negative blend, with weights fitted on the dense label matrix.
  4. With --predict, refits every learner on all the data and writes a submission.

The recalibration in step 2 is the part that matters. Two leaderboard results told us
how far each local scorer is from the truth:

    actual = 0.597 * simulator + 0.430      (2 points, R^2 = 1 by construction)
    actual = 0.212 * dense     + 0.776

so a dense-proxy improvement of 0.10 is worth about 0.02 on the leaderboard. Any
candidate here has to clear 0.9171 after that shrinkage, not before it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit import cyp_descriptors as CD  # noqa: E402
from cypkit import ensemble as E  # noqa: E402
from cypkit import features, io, labels as L, losses, scoring, splits  # noqa: E402

# Fitted in scripts/calibrate_validation.py from the two leaderboard results we have.
DENSE_A, DENSE_B = 0.776, 0.212
CURRENT_BEST_LB = 0.9171


def build_bank(smiles_by_id: dict[str, str], blocks: list[str]) -> E.FeatureBank:
    ids = sorted(smiles_by_id)
    smi = [smiles_by_id[i] for i in ids]
    mats, spans, at = [], {}, 0
    for b in blocks:
        t0 = time.time()
        if b == "rdkit":
            X, _ = features.rdkit_2d(smi)
        elif b == "cyp":
            X, _ = CD.calculate(smi)
        elif b == "screen":
            from cypkit import screen_prior as SP  # noqa: PLC0415
            X, _ = SP.calculate(smi)
        elif b == "prior":
            from cypkit import external_prior as EP  # noqa: PLC0415
            X, _ = EP.calculate(smi)
        elif b == "tdialert":
            from cypkit import tdi_alerts as TA  # noqa: PLC0415
            X, _ = TA.calculate(smi)
        elif b == "3d":
            X, _ = features.rdkit_3d(smi)
            X = np.nan_to_num(X, nan=0.0)
        else:
            X, _, _ = features.fingerprint_block(smi, b)
        X = np.ascontiguousarray(np.nan_to_num(X.astype(np.float32), nan=0.0,
                                               posinf=0.0, neginf=0.0))
        spans[b] = (at, at + X.shape[1])
        at += X.shape[1]
        mats.append(X)
        print(f"    {b:6s} {X.shape[1]:5d} cols  ({time.time()-t0:.1f}s)")
    return E.FeatureBank(ids, np.hstack(mats), spans)


def oof_for(name, bank, label_matrix, fold_of, n_folds, delta, verbose=True):
    """Out-of-fold predictions for one learner: DataFrame of ids x endpoints."""
    X = bank.view(E.LEARNERS[name]["features"])
    ids, index_of = bank.ids, bank.index_of
    oof = pd.DataFrame({C.ID_COL: ids})
    for cyp in C.ISOFORMS:
        ep = f"{cyp}_pIC50_direct_inhibition"
        sub = label_matrix[(label_matrix["isoform"] == cyp)
                           & label_matrix[C.ID_COL].isin(index_of)]
        preds = np.full(len(ids), np.nan)
        for f in range(n_folds):
            tr = sub[sub[C.ID_COL].map(fold_of) != f]
            te_mask = np.array([fold_of.get(i, -1) == f for i in ids])
            if len(tr) < 20 or not te_mask.any():
                continue
            tr_idx = tr[C.ID_COL].map(index_of).to_numpy()
            out = E.fit_predict(name, X[tr_idx], tr, X[te_mask], delta=delta)
            if out is not None:
                preds[te_mask] = out
        if np.isnan(preds).any():
            fill = float(np.nanmedian(sub["y"])) if sub["y"].notna().any() else C.PIC50_FLOOR
            preds = np.where(np.isnan(preds), fill, preds)
        oof[ep] = preds
        if verbose:
            print(f"    {ep:34s} mean {preds.mean():5.2f}  sd {preds.std():4.2f}")
    return oof


def dense_rows(label_matrix, oof_or_pred: dict[str, np.ndarray] | None = None):
    """Row index / band arrays per endpoint over the FULL label matrix."""
    out = {}
    for ep in C.REGRESSION_ENDPOINTS:
        r = label_matrix[label_matrix["endpoint"] == ep]
        out[ep] = dict(
            ids=r[C.ID_COL].to_numpy(),
            lo=r["y_low"].to_numpy(dtype=float),
            hi=r["y_high"].to_numpy(dtype=float),
            y=r["y"].to_numpy(dtype=float),
        )
    return out


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
    ap.add_argument("--learners", nargs="*", default=list(E.LEARNERS))
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--censored-weight", type=float, default=0.5)
    ap.add_argument("--pseudo", action="store_true",
                    help="add screen-derived pseudo-pIC50 labels (~2,900 compounds per "
                         "isoform) via the monotone log2fc map; see "
                         "cypkit.labels.pseudo_from_screen")
    ap.add_argument("--pseudo-band", type=float, default=1.0)
    ap.add_argument("--pseudo-weight", type=float, default=0.5)
    ap.add_argument("--pseudo-nonhit-interval", action="store_true",
                    help="give screen non-hits a two-sided interval instead of an "
                         "upper bound (the aggressive variant)")
    ap.add_argument("--fast", action="store_true", help="100 bootstrap samples, not 1000")
    ap.add_argument("--predict", action="store_true", help="also write a submission")
    ap.add_argument("--tag", default="ensA")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    bad = [n for n in args.learners if n not in E.LEARNERS]
    if bad:
        raise SystemExit(f"unknown learner(s) {bad}; choose from {list(E.LEARNERS)}")

    train = io.load("train_inhibition", args.data_dir)
    screen = io.load("single_conc", args.data_dir)
    test = io.load("test", args.data_dir)
    folds = splits.load_folds(C.artifact("folds.csv"))
    n_folds = int(folds["fold"].max()) + 1
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))

    label_matrix = L.build_label_matrix(
        train, screen, include_censored=True, censored_weight=args.censored_weight,
        pseudo=args.pseudo, pseudo_band=args.pseudo_band,
        pseudo_weight=args.pseudo_weight,
        pseudo_nonhit_interval=args.pseudo_nonhit_interval)
    extra = sorted(set(label_matrix[C.ID_COL]) - set(fold_of))
    if extra:
        smi = dict(zip(label_matrix[C.ID_COL], label_matrix[C.SMILES_COL]))
        print(f"assigning {len(extra)} screen-only compounds to folds...")
        fold_of.update(dict(zip(extra, splits.assign_extra_to_folds(
            [smi[i] for i in extra], folds))))

    smiles_by_id = dict(zip(label_matrix[C.ID_COL], label_matrix[C.SMILES_COL]))
    smiles_by_id.update(dict(zip(train[C.ID_COL], train[C.SMILES_COL])))
    smiles_by_id.update(dict(zip(test[C.ID_COL], test[C.SMILES_COL])))
    smiles_by_id = {k: v for k, v in smiles_by_id.items() if isinstance(v, str)}

    blocks = E.blocks_needed(args.learners)
    print(f"featurising {len(smiles_by_id)} compounds, blocks = {'+'.join(blocks)}")
    bank = build_bank(smiles_by_id, blocks)
    print(f"  bank = {bank.X.shape}")

    n_boot = 100 if args.fast else None
    dense = dense_rows(label_matrix)

    oofs, rows = {}, []
    for name in args.learners:
        print(f"\n=== {name}  [{E.LEARNERS[name]['features']}] ===")
        t0 = time.time()
        oof = oof_for(name, bank, label_matrix, fold_of, n_folds, args.delta)
        oof.to_csv(C.artifact(f"oof_{name}_{args.tag}.csv"), index=False)
        oofs[name] = oof

        lb = scoring.oof_report(oof, train, n_bootstrap=n_boot)
        ma = lb[lb["Endpoint"] == "MA"].iloc[0]
        pred_by = {ep: dict(zip(oof[C.ID_COL], oof[ep])) for ep in C.REGRESSION_ENDPOINTS}
        ds = []
        for ep in C.REGRESSION_ENDPOINTS:
            d = dense[ep]
            p = pd.Series(d["ids"]).map(pred_by[ep]).to_numpy(dtype=float)
            ds.append(dense_strae(p, d["lo"], d["hi"], d["y"]))
        dma = float(np.mean(ds))
        rows.append(dict(model=name, lb_strae=float(ma["ST-RAE"]), mae=float(ma["MAE"]),
                         spearman=float(ma["Spearman_R"]), dense=dma,
                         recal=DENSE_A + DENSE_B * dma, secs=round(time.time() - t0, 1)))
        print(f"  ST-RAE(oof) {ma['ST-RAE']:.4f}   dense {dma:.4f}   "
              f"-> recalibrated leaderboard {DENSE_A + DENSE_B*dma:.4f}  "
              f"({time.time()-t0:.0f}s)")

    tab = pd.DataFrame(rows).sort_values("recal")
    print("\n=== single learners ===")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- blend -----------------------------------------------------------
    print("\n=== blend (weights fitted per endpoint on the dense label matrix) ===")
    names = list(oofs)
    weights, blended = {}, pd.DataFrame({C.ID_COL: bank.ids})
    for ep in C.REGRESSION_ENDPOINTS:
        d = dense[ep]
        P = np.column_stack([
            pd.Series(d["ids"]).map(dict(zip(oofs[n][C.ID_COL], oofs[n][ep]))).to_numpy(float)
            for n in names])
        keep = np.isfinite(P).all(1)
        w = E.blend_weights(P[keep], d["lo"][keep], d["hi"][keep])
        weights[ep] = dict(zip(names, w.round(4).tolist()))
        blended[ep] = np.column_stack([oofs[n][ep].to_numpy() for n in names]) @ w
        top = ", ".join(f"{n} {v:.2f}" for n, v in
                        sorted(weights[ep].items(), key=lambda kv: -kv[1]) if v > 0.02)
        print(f"  {ep:34s} {top}")

    lb = scoring.oof_report(blended, train, n_bootstrap=n_boot)
    ma = lb[lb["Endpoint"] == "MA"].iloc[0]
    pred_by = {ep: dict(zip(blended[C.ID_COL], blended[ep])) for ep in C.REGRESSION_ENDPOINTS}
    ds = [dense_strae(pd.Series(dense[ep]["ids"]).map(pred_by[ep]).to_numpy(float),
                      dense[ep]["lo"], dense[ep]["hi"], dense[ep]["y"])
          for ep in C.REGRESSION_ENDPOINTS]
    dma = float(np.mean(ds))
    recal = DENSE_A + DENSE_B * dma
    print(f"\n  BLEND  ST-RAE(oof) {ma['ST-RAE']:.4f}   dense {dma:.4f}   "
          f"-> recalibrated leaderboard {recal:.4f}")
    print(f"  current leaderboard best = {CURRENT_BEST_LB:.4f}  "
          f"({'BEATS IT' if recal < CURRENT_BEST_LB else 'does NOT beat it'})")

    blended.to_csv(C.artifact(f"oof_blend_{args.tag}.csv"), index=False)
    tab.to_csv(C.artifact(f"ensemble_summary_{args.tag}.csv"), index=False)
    with open(C.artifact(f"ensemble_weights_{args.tag}.json"), "w") as fh:
        json.dump(weights, fh, indent=2)

    # ---- blind set -------------------------------------------------------
    if args.predict:
        print("\n=== refitting on all data and predicting the blind set ===")
        test_idx = test[C.ID_COL].map(bank.index_of).to_numpy()
        if np.isnan(test_idx.astype(float)).any():
            raise RuntimeError("some test compounds failed featurisation")
        per = {}
        for name in names:
            t0 = time.time()
            X = bank.view(E.LEARNERS[name]["features"])
            cols = {}
            for cyp in C.ISOFORMS:
                ep = f"{cyp}_pIC50_direct_inhibition"
                sub = label_matrix[(label_matrix["isoform"] == cyp)
                                   & label_matrix[C.ID_COL].isin(bank.index_of)]
                tr_idx = sub[C.ID_COL].map(bank.index_of).to_numpy()
                cols[ep] = E.fit_predict(name, X[tr_idx], sub, X[test_idx], delta=args.delta)
            per[name] = cols
            m = test[[C.SMILES_COL, C.ID_COL]].copy()
            for ep, v in cols.items():
                m[ep] = v
            m.to_csv(C.artifact(f"testpred_{name}.csv"), index=False)
            print(f"  {name:8s} done ({time.time()-t0:.0f}s)  -> testpred_{name}.csv")

        out = test[[C.SMILES_COL, C.ID_COL]].copy()
        for ep in C.REGRESSION_ENDPOINTS:
            w = np.array([weights[ep][n] for n in names])
            out[ep] = np.column_stack([per[n][ep] for n in names]) @ w
        path = C.artifact(f"submission_{args.tag}.csv")
        out.to_csv(path, index=False)
        print(f"\nwritten to {path}")
        print(out[list(C.REGRESSION_ENDPOINTS)].describe().T.to_string(
            float_format=lambda x: f"{x:.3f}"))
        frac = {ep: float((out[ep] < C.PIC50_FLOOR).mean()) for ep in C.REGRESSION_ENDPOINTS}
        print("  fraction below the pIC50 floor: "
              + ", ".join(f"{k.split('_')[0]} {v:.2f}" for k, v in frac.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
