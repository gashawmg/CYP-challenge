#!/usr/bin/env python
"""TDI track: out-of-fold comparison, then a validated submission.

    python scripts/run_tdi.py
    python scripts/run_tdi.py --predict          # also write the test submission

Variants:

    clf_050      LGBMClassifier thresholded at 0.5 — the tutorial baseline
    clf_tuned    same model, threshold chosen to maximise out-of-fold MCC
    structured   regress both pIC50 arms + P(direct curve fitted), apply the exact
                 published rule softly, tune the threshold
    blend        mean of the two scores, threshold tuned

Scored with the challenge's own MCC on the frozen folds from make_folds.py.
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
from cypkit import io, scoring, splits  # noqa: E402
from cypkit import tdi as T  # noqa: E402
from scripts.run_baselines import build_features  # noqa: E402

CLF = dict(objective="binary", verbose=-1, learning_rate=0.05, num_leaves=31,
           min_data_in_leaf=20, feature_fraction=0.6, bagging_fraction=0.8,
           bagging_freq=1, seed=42)
REG = dict(objective="regression_l1", verbose=-1, learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=10, feature_fraction=0.6, bagging_fraction=0.8,
           bagging_freq=1, seed=42)


def _fit_predict(params, Xtr, y, Xte, n_rounds, weight=None):
    ds = lgb.Dataset(Xtr, label=y, weight=weight)
    return lgb.train(params, ds, num_boost_round=n_rounds).predict(Xte)


def oof_scores(table, X, index_of, fold_of, n_folds, n_rounds):
    """Out-of-fold scores for every variant, per isoform."""
    out = {}
    for cyp in C.TDI_ISOFORMS:
        sub = table[(table["isoform"] == cyp) & table[C.ID_COL].isin(index_of)].copy()
        sub["fold"] = sub[C.ID_COL].map(fold_of)
        idx = sub[C.ID_COL].map(index_of).to_numpy()
        n = len(sub)
        s_clf = np.full(n, np.nan)
        s_str = np.full(n, np.nan)

        for f in range(n_folds):
            te = (sub["fold"] == f).to_numpy()
            tr = ~te
            if te.sum() == 0 or tr.sum() < 20:
                continue
            Xtr, Xte = X[idx[tr]], X[idx[te]]

            s_clf[te] = _fit_predict(CLF, Xtr, sub["y"].to_numpy()[tr].astype(int), Xte, n_rounds)

            # Structured: three heads, then the rule.
            p_meas = _fit_predict(CLF, Xtr, sub["measurable"].to_numpy()[tr].astype(int), Xte, n_rounds)
            d = sub["direct"].to_numpy()[tr]
            t = sub["tdi_arm"].to_numpy()[tr]
            ok_d, ok_t = np.isfinite(d), np.isfinite(t)
            if ok_d.sum() < 20 or ok_t.sum() < 20:
                continue
            pred_d = _fit_predict(REG, Xtr[ok_d], d[ok_d], Xte, n_rounds)
            pred_t = _fit_predict(REG, Xtr[ok_t], t[ok_t], Xte, n_rounds)
            s_str[te] = T.structured_score(p_meas, pred_d, pred_t)

        sub["score_clf"] = s_clf
        sub["score_str"] = s_str
        sub["score_blend"] = np.nanmean(np.vstack([s_clf, s_str]), axis=0)
        out[cyp] = sub
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=("rdkit", "cyp", "both"), default="both")
    ap.add_argument("--n-rounds", type=int, default=300)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--predict", action="store_true", help="also write a test submission")
    ap.add_argument("--submit-variant", default="best")
    ap.add_argument("--folds", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    tdi_raw = io.load("train_tdi", args.data_dir)
    test = io.load("test", args.data_dir)

    print("Verifying the recovered label rule against the released labels")
    print(T.verify_rule(tdi_raw).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("  rule: direct-arm fitted AND shift > log10(2) AND TDI-arm pIC50 > 4.301")

    table = T.assemble(tdi_raw)
    folds = splits.load_folds(args.folds or C.artifact("folds.csv"))
    n_folds = int(folds["fold"].max()) + 1
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))

    smiles_by_id = dict(zip(table[C.ID_COL], table[C.SMILES_COL]))
    smiles_by_id.update(dict(zip(test[C.ID_COL], test[C.SMILES_COL])))
    extra = sorted(set(smiles_by_id) - set(fold_of) - set(test[C.ID_COL]))
    if extra:
        print(f"assigning {len(extra)} TDI-only compounds to folds by nearest neighbour...")
        fold_of.update(dict(zip(extra, splits.assign_extra_to_folds(
            [smiles_by_id[i] for i in extra], folds))))

    print(f"featurising {len(smiles_by_id)} compounds...")
    t0 = time.time()
    ids, X, _ = build_features(smiles_by_id, False, args.features)
    index_of = {m: i for i, m in enumerate(ids)}
    print(f"  X = {X.shape}  ({time.time()-t0:.1f}s)")

    per_iso = oof_scores(table, X, index_of, fold_of, n_folds, args.n_rounds)

    from sklearn.metrics import matthews_corrcoef

    thresholds, rows = {}, []
    for name, col in [("clf_050", "score_clf"), ("clf_tuned", "score_clf"),
                      ("structured", "score_str"), ("blend", "score_blend")]:
        preds = {}
        for cyp, sub in per_iso.items():
            s = sub[col].to_numpy()
            y = sub["y"].to_numpy().astype(bool)
            ok = np.isfinite(s)
            if name == "clf_050":
                t = 0.5
            else:
                t, _ = T.best_mcc_threshold(y[ok], s[ok])
            thresholds[(name, cyp)] = t
            rows.append(dict(variant=name, isoform=cyp, threshold=t, n=int(ok.sum()),
                             MCC=matthews_corrcoef(y[ok], s[ok] >= t),
                             pred_pos_rate=float(np.mean(s[ok] >= t)),
                             true_pos_rate=float(y[ok].mean())))
            preds[cyp] = (sub[C.ID_COL].to_numpy()[ok], s[ok] >= t)

    res = pd.DataFrame(rows)
    macro = res.groupby("variant")["MCC"].mean().rename("MA_MCC")
    print("\n=== out-of-fold MCC ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== macro-averaged (the leaderboard sorts on MCC) ===")
    print(macro.sort_values(ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    res.to_csv(C.artifact("tdi_oof_summary.csv"), index=False)

    best = macro.idxmax()
    print(f"\nbest variant: {best}  (MA-MCC {macro.max():.4f})")

    if not args.predict:
        print("\n(pass --predict to train on everything and write the submission)")
        return 0

    variant = best if args.submit_variant == "best" else args.submit_variant
    col = {"clf_050": "score_clf", "clf_tuned": "score_clf",
           "structured": "score_str", "blend": "score_blend"}[variant]
    print(f"\nTraining final {variant} models on all labelled data...")

    test_idx = test[C.ID_COL].map(index_of).to_numpy()
    sub_out = test[[C.SMILES_COL, C.ID_COL]].copy()
    for cyp in C.TDI_ISOFORMS:
        s = per_iso[cyp]
        tr_idx = s[C.ID_COL].map(index_of).to_numpy()
        y = s["y"].to_numpy().astype(int)
        score_clf = _fit_predict(CLF, X[tr_idx], y, X[test_idx], args.n_rounds)

        p_meas = _fit_predict(CLF, X[tr_idx], s["measurable"].to_numpy().astype(int),
                              X[test_idx], args.n_rounds)
        d, t = s["direct"].to_numpy(), s["tdi_arm"].to_numpy()
        ok_d, ok_t = np.isfinite(d), np.isfinite(t)
        pred_d = _fit_predict(REG, X[tr_idx][ok_d], d[ok_d], X[test_idx], args.n_rounds)
        pred_t = _fit_predict(REG, X[tr_idx][ok_t], t[ok_t], X[test_idx], args.n_rounds)
        score_str = T.structured_score(p_meas, pred_d, pred_t)

        score = {"score_clf": score_clf, "score_str": score_str,
                 "score_blend": (score_clf + score_str) / 2}[col]
        thr = thresholds[(variant, cyp)]
        sub_out[f"{cyp}_is_TDI"] = score >= thr
        print(f"  {cyp}: threshold={thr:.4f}  predicted positive rate="
              f"{float(np.mean(score >= thr)):.3f}  (training {s['y'].mean():.3f})")

    path = C.artifact(f"submission_tdi_{variant}.csv")
    sub_out.to_csv(path, index=False)
    print(f"\nwritten to {path}  ({len(sub_out)} rows)")

    sys.path.insert(0, str(C.CHALLENGE_REPO))
    from validation.tdi_validation import validate_tdi_submission  # noqa: PLC0415

    ok, errors = validate_tdi_submission(path, expected_ids=set(test[C.ID_COL]))
    print("  validator: " + ("PASS" if ok else "FAIL"))
    for e in errors:
        print(f"    - {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
