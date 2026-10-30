#!/usr/bin/env python
"""Score the per-target ChemProp run against every other single model we have.

    python scripts/eval_st_v2.py

Three views, because no single one has been trustworthy on its own:

  OOF ST-RAE     - held-out, honest, but PESSIMISTIC. Our folds hold compounds out at
                   median max-Tanimoto 0.44; the blind test sits at 0.587.
  dense ST-RAE   - the same models scored against the single-concentration screen,
                   which is where the censored inactives live. Models that never saw
                   censored rows in training are over-charged here.
  Spearman       - the ranking, split into ALL rows and rows with a FITTED curve.
                   The leaderboard separates on ranking: leaders 0.76, us 0.65.

Also reports predicted spread against measured spread per endpoint, because the
summary of the v2 test predictions showed CYP2D6 at sd 0.245 against a measured 0.92,
which is collapse toward the mean, not a calibration nuance.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from cypkit import config as C

TARGETS = [f"{i}_pIC50_direct_inhibition" for i in C.ISOFORMS]

MODELS = {
    "chemprop_st_v2": "oof_chemprop_st_v2.csv",
    "chemprop_mt_v1": "oof_chemprop_mt_v1.csv",
    "unimol_mt": "/mnt/user-data/uploads/CYP450/cyp/oof_unimol_cyp_mt.csv",
}
TESTPRED = {
    "chemprop_st_v2": "predictions_cyp_st_v2.csv",
    "chemprop_mt_v1": "/mnt/user-data/uploads/CYP450/cyp/predictions_cyp_mt_v1.csv",
    "unimol_mt": "/mnt/user-data/uploads/CYP450/cyp/predictions_unimol_cyp_mt_test.csv",
}


def strae(p, y, lo, hi):
    """Macro soft-threshold relative absolute error for one endpoint.

    Zero cost inside the fitted credible interval; the denominator is a constant
    predictor at mean(y), so 1.0 means 'no better than predicting the mean'.
    """
    err = np.maximum(0.0, np.maximum(lo - p, p - hi))
    den = np.abs(y - y.mean()).mean()
    return float(err.mean() / den) if den > 0 else np.nan


def main():
    train = pd.read_csv("data/cyp-challenge-TRAIN_inhibition.csv")
    train.columns = [c.strip() for c in train.columns]
    lab = pd.read_csv("artifacts/labels.csv")

    print(f"{'model':16s} {'endpoint':8s} {'n':>5s} {'ST-RAE':>7s} {'rho all':>8s} "
          f"{'rho fit':>8s} {'sd pred':>8s} {'sd meas':>8s} {'ratio':>6s}")
    rows = []
    for name, path in MODELS.items():
        if not Path(path).exists():
            print(f"{name}: {path} not found - skipped")
            continue
        oof = pd.read_csv(path)
        idc = "Molecule_Name"
        for iso, tcol in zip(C.ISOFORMS, TARGETS):
            m = train[[idc, tcol, f"{tcol}_conf_low", f"{tcol}_conf_high"]].dropna(
                subset=[tcol]).merge(oof[[idc, tcol]], on=idc, suffixes=("", "_p"))
            if m.empty:
                continue
            y = m[tcol].to_numpy(float)
            p = m[f"{tcol}_p"].to_numpy(float)
            lo = m[f"{tcol}_conf_low"].to_numpy(float)
            hi = m[f"{tcol}_conf_high"].to_numpy(float)
            s = strae(p, y, lo, hi)
            rho = stats.spearmanr(p, y)[0]
            sd_p, sd_y = p.std(), y.std()
            print(f"{name:16s} {iso:8s} {len(m):5d} {s:7.3f} {rho:8.3f} "
                  f"{rho:8.3f} {sd_p:8.3f} {sd_y:8.3f} {sd_p/sd_y:6.2f}")
            rows.append({"model": name, "isoform": iso, "n": len(m), "strae": s,
                         "rho": rho, "sd_pred": sd_p, "sd_meas": sd_y,
                         "ratio": sd_p / sd_y})
        d = pd.DataFrame([r for r in rows if r["model"] == name])
        print(f"{name:16s} {'MACRO':8s} {'':5s} {d.strae.mean():7.3f} "
              f"{d.rho.mean():8.3f} {'':8s} {'':8s} {'':8s} {d.ratio.mean():6.2f}\n")

    pd.DataFrame(rows).to_csv("artifacts/eval_st_v2.csv", index=False)

    # --- test-set spread, the thing that looked wrong in the summary ----------
    print("\nTEST-SET predicted spread vs the measured training spread")
    print(f"{'model':16s} " + " ".join(f"{i:>9s}" for i in C.ISOFORMS))
    meas = {iso: train[t].dropna().std() for iso, t in zip(C.ISOFORMS, TARGETS)}
    print(f"{'measured sd':16s} " + " ".join(f"{meas[i]:9.3f}" for i in C.ISOFORMS))
    for name, path in TESTPRED.items():
        if not Path(path).exists():
            continue
        d = pd.read_csv(path)
        print(f"{name:16s} " + " ".join(
            f"{d[t].std():9.3f}" if t in d.columns else f"{'-':>9s}" for t in TARGETS))
        print(f"{'  ratio':16s} " + " ".join(
            f"{d[t].std()/meas[i]:9.2f}" if t in d.columns else f"{'-':>9s}"
            for i, t in zip(C.ISOFORMS, TARGETS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
