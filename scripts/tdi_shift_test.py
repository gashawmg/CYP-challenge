#!/usr/bin/env python
"""Is the TDI label better reached through the pIC50 SHIFT than through the label?

    python scripts/tdi_shift_test.py

WHAT THIS IS TESTING. `is_TDI` is very nearly a deterministic function of two numbers
we are already given:

    shift = pIC50_TDI_condition - pIC50_direct_inhibition

    is_TDI ~= (shift > 0.30) AND (TDI_conf_low - direct_conf_high > s)

which recovers the published label at MCC 0.969 (CYP2D6) and 0.944 (CYP3A4). 0.30 is
log10(2) -- the rule is the standard "at least a two-fold IC50 shift, and the shift is
statistically significant". So the binary label is a THRESHOLDED MEASUREMENT, and we
have been modelling the thresholded version while the underlying continuous quantity
sits right there in the training file.

Two ways to predict it, compared here head to head on the frozen folds:

    A. classify   -> LightGBM on is_TDI, threshold the probability
    B. regress    -> LightGBM on the shift, threshold the predicted shift

B should win: the shift carries magnitude, it is defined for compounds whose binary
label is missing, and 21-33% positive rates make binary training data thin. If it
wins by a wide margin the ChemProp TDI model should have the shift as its main head.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from cypkit import features, splits, config as C
from cypkit import cyp_descriptors as CD

SCORED = ["CYP3A4", "CYP2D6"]
ALL = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def best_mcc(y, score):
    """Best MCC over all thresholds, and the predicted-positive rate there."""
    order = np.unique(score)
    if len(order) > 400:
        order = np.quantile(score, np.linspace(0.01, 0.99, 400))
    best = (-1.0, 0.0, 0.0)
    for t in order:
        p = (score > t).astype(int)
        if p.sum() == 0 or p.sum() == len(p):
            continue
        m = matthews_corrcoef(y, p)
        if m > best[0]:
            best = (m, float(t), float(p.mean()))
    return best


def main():
    d = pd.read_csv("data/cyp-challenge-TRAIN_TDI.csv")
    d.columns = [c.strip() for c in d.columns]
    folds = splits.load_folds(C.artifact("folds.csv"))
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))
    extra = sorted(set(d[C.ID_COL]) - set(fold_of))
    if extra:
        smi = dict(zip(d[C.ID_COL], d[C.SMILES_COL]))
        print(f"assigning {len(extra)} TDI-only compounds to folds by nearest neighbour")
        fold_of.update(dict(zip(extra, splits.assign_extra_to_folds(
            [smi[i] for i in extra], folds))))
    f = d[C.ID_COL].map(fold_of).to_numpy()

    smiles = d[C.SMILES_COL].tolist()
    Xr, _ = features.rdkit_2d(smiles)
    Xc, _ = CD.calculate(smiles)
    Xf, _, _ = features.fingerprint_block(smiles, "ecfp")
    X = np.nan_to_num(np.hstack([Xr, Xc, Xf]).astype(np.float32), nan=0.0,
                      posinf=0.0, neginf=0.0)
    print(f"features {X.shape}")

    print(f"\n{'iso':8s} {'route':10s} {'n_train':>8s} {'n_eval':>7s} {'AUC':>7s} "
          f"{'bestMCC':>8s} {'rate':>6s}")
    rows = []
    for iso in SCORED:
        lab = d[f"{iso}_is_TDI"]
        shift = (d[f"{iso}_pIC50_TDI_condition"] - d[f"{iso}_pIC50_direct_inhibition"])
        ev = lab.notna().to_numpy()
        y = lab[ev].astype(int).to_numpy()

        # ---- A: classify the label -------------------------------------
        oof = np.full(len(d), np.nan)
        for k in range(5):
            tr = ev & (f != k); te = (f == k)
            if tr.sum() < 50 or te.sum() == 0:
                continue
            m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.04, num_leaves=31,
                                   colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                                   min_child_samples=15, verbose=-1, n_jobs=2,
                                   random_state=0)
            m.fit(X[tr], lab[tr].astype(int))
            oof[te] = m.predict_proba(X[te])[:, 1]
        s = oof[ev]
        ok = np.isfinite(s)
        mc, t, r = best_mcc(y[ok], s[ok])
        print(f"{iso:8s} {'classify':10s} {int(ev.sum()):8d} {int(ok.sum()):7d} "
              f"{roc_auc_score(y[ok], s[ok]):7.4f} {mc:8.4f} {r:6.3f}")
        rows.append({"isoform": iso, "route": "classify", "auc": roc_auc_score(y[ok], s[ok]),
                     "mcc": mc, "rate": r})
        np.save(f"artifacts/tdi_oof_clf_{iso}.npy", oof)

        # ---- B: regress the shift --------------------------------------
        sv = shift.notna().to_numpy()
        oof2 = np.full(len(d), np.nan)
        for k in range(5):
            tr = sv & (f != k); te = (f == k)
            if tr.sum() < 50 or te.sum() == 0:
                continue
            m = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.04, num_leaves=31,
                                  colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                                  min_child_samples=15, verbose=-1, n_jobs=2,
                                  random_state=0)
            m.fit(X[tr], shift[tr])
            oof2[te] = m.predict(X[te])
        s2 = oof2[ev]
        ok2 = np.isfinite(s2)
        mc2, t2, r2 = best_mcc(y[ok2], s2[ok2])
        print(f"{iso:8s} {'regress':10s} {int(sv.sum()):8d} {int(ok2.sum()):7d} "
              f"{roc_auc_score(y[ok2], s2[ok2]):7.4f} {mc2:8.4f} {r2:6.3f}   "
              f"(threshold on predicted shift = {t2:+.3f})")
        rows.append({"isoform": iso, "route": "regress", "auc": roc_auc_score(y[ok2], s2[ok2]),
                     "mcc": mc2, "rate": r2, "threshold": t2})
        np.save(f"artifacts/tdi_oof_shift_{iso}.npy", oof2)

        # ---- C: both, averaged after rank-normalising ------------------
        both = ev & np.isfinite(oof) & np.isfinite(oof2)
        rk = (pd.Series(oof[both]).rank(pct=True).to_numpy()
              + pd.Series(oof2[both]).rank(pct=True).to_numpy()) / 2
        mc3, _, r3 = best_mcc(lab[both].astype(int).to_numpy(), rk)
        print(f"{iso:8s} {'both':10s} {'':8s} {int(both.sum()):7d} "
              f"{roc_auc_score(lab[both].astype(int), rk):7.4f} {mc3:8.4f} {r3:6.3f}")
        rows.append({"isoform": iso, "route": "both", "mcc": mc3, "rate": r3})

    r = pd.DataFrame(rows)
    r.to_csv("artifacts/tdi_shift_test.csv", index=False)
    for route in ("classify", "regress", "both"):
        v = r[r.route == route]["mcc"]
        print(f"\nMA-MCC ({route:8s}) = {v.mean():.4f}")
    print("current leaderboard MA-MCC 0.2981 (rank 23); leaders 0.40-0.55")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
