#!/usr/bin/env python
"""Fit TDI blend weights on AUC, and measure the gain HONESTLY.

    python scripts/tdi_weights.py --tag tdiI                    # measure only
    python scripts/tdi_weights.py --tag tdiI --submission artifacts/submission_tdi_w.csv

WHY AUC AND NOT MCC. Inverting the per-isoform leaderboard gives every competitor's
discriminability d' and their cut rate, and the ratio MCC/d' is astonishingly stable:

    CYP3A4   MCC/d' = 0.343 - 0.353 across cut rates 0.216 - 0.314
    CYP2D6   MCC/d' = 0.209 - 0.244 across cut rates 0.082 - 0.214

CORRECTED. An earlier version of this docstring read "spread 0.010 ... the threshold is
not a lever". On CYP3A4 that holds. On CYP2D6 it does not: MCC/d' runs 0.209 to 0.244, a
17% spread, because at 7% prevalence the optimum sits near 0.14 and we were cutting at
0.21. Discriminability is still most of the game -- but the threshold is worth ~0.005
per endpoint and costs nothing, so take it. So weights are fitted on AUC: smooth, threshold-free, and far less
noisy than MCC-at-a-rate, which jumps around because it depends on a handful of
compounds near the cut.

WHY NESTED VALIDATION. Fitting weights on the same rows you score them on is how our
earlier sweep produced CYP2D6 MCC 0.1423 -> 0.1949 with 4,000 random simplex draws over
ten members. That number was optimism, not signal, and acting on it would have wasted a
submission. Here weights are fitted on four folds and scored on the fifth, rotating --
so the reported gain is what a NEW test set should see. Final weights are then refit on
all folds, which is standard and mildly optimistic in the same way any model is.

WHAT NESTED VALIDATION DOES NOT PROTECT AGAINST -- MEASURED, NOT THEORISED. The first
submission built from these weights split the two endpoints hard:

    CYP3A4   d' 1.290 -> 1.064   MCC 0.4470 -> 0.3798   (nested OOF said +0.0057)
    CYP2D6   d' 0.861 -> 0.980   MCC 0.1825 -> 0.2180   (nested OOF said +0.0069)

Nested CV strips out fitting noise WITHIN the cross-validation distribution. It cannot
see distribution SHIFT between that distribution and the blind set, and our folds hold
compounds out at median max-Tanimoto 0.44 while the blind set sits at 0.587. So weight
fitting selects the members that survive distant held-out chemistry, which is not the
same as the members that win on close analogues.

That predicts exactly the split we got. CYP2D6's fitted weights loaded on the
similarity kernels (knn_tan 0.32 + svc_tan 0.30) -- methods that get BETTER as the test
set moves closer to training -- and it gained. CYP3A4's loaded on `struct` (0.29), a
rule-derived member, and it lost a fifth of its discriminability.

RULE OF THUMB FROM THIS: fit weights per isoform, submit them per isoform, and only
keep a fitted set for an endpoint where it has beaten equal weights ON THE LEADERBOARD.
Two anchors per endpoint is cheap; guessing from OOF is not.

RIDGE TOWARD EQUAL WEIGHTS. Ten members and ~1,500 CYP2D6 rows is a regime where free
weights overfit. `--ridge` pulls the solution toward the equal-weight blend; the
default is chosen by the nested score, not by taste.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from cypkit import config as C, splits

# MCC = K[iso] * d', measured from the leaderboard (see the docstring)
K = {"CYP3A4": 0.349, "CYP2D6": 0.214}
RATE = {"CYP3A4": 0.306, "CYP2D6": 0.142}   # see scripts/tdi_splice.py


def rank(v):
    return pd.Series(v).rank(pct=True).to_numpy()


def dprime(auc):
    return float(np.sqrt(2) * norm.ppf(min(max(auc, 1e-6), 1 - 1e-6)))


def fit_weights(R, y, ridge, seed=0):
    """Maximise AUC of a weighted rank-average. Simplex-constrained, ridged to equal."""
    m = R.shape[0]
    eq = np.ones(m) / m

    def neg(w):
        w = np.clip(w, 0, None)
        if w.sum() <= 0:
            return 1e9
        w = w / w.sum()
        s = w @ R
        return -roc_auc_score(y, s) + ridge * float(((w - eq) ** 2).sum())

    best = None
    rng = np.random.default_rng(seed)
    starts = [eq] + [rng.dirichlet(np.ones(m)) for _ in range(8)]
    for w0 in starts:
        r = minimize(neg, w0, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-7))
        if best is None or r.fun < best.fun:
            best = r
    w = np.clip(best.x, 0, None)
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="tdiI")
    ap.add_argument("--ridge", type=float, nargs="*", default=[0.0, 0.05, 0.2, 0.5])
    ap.add_argument("--submission", default=None,
                    help="write a blind-set submission with the refit weights "
                         "(needs artifacts/tdi_member_test_<member>_<tag>.csv)")
    args = ap.parse_args()

    folds = splits.load_folds(C.artifact("folds.csv"))
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))

    chosen, per_iso = {}, {}
    for iso in C.TDI_ISOFORMS:
        path = C.artifact(f"tdi_member_oof_{iso}_{args.tag}.csv")
        if not Path(path).exists():
            raise SystemExit(f"{path} not found -- run run_tdi_ensemble.py --save-oof")
        d = pd.read_csv(path)
        members = [c for c in d.columns if c not in (C.ID_COL, "y")]
        y = d["y"].to_numpy().astype(int)
        R = np.vstack([rank(d[m].to_numpy()) for m in members])
        f = d[C.ID_COL].map(fold_of).fillna(-1).to_numpy().copy()
        # TDI-only compounds have no frozen fold; spread them so every fold is used
        miss = f < 0
        if miss.any():
            f[miss] = np.arange(miss.sum()) % 5

        eq = roc_auc_score(y, np.ones(len(members)) / len(members) @ R)
        print(f"\n{iso}: {len(y)} rows, {len(members)} members, "
              f"equal-weight AUC {eq:.4f}  (d' {dprime(eq):.3f}, "
              f"MCC ~ {K[iso]*dprime(eq):.4f})")
        print(f"  {'ridge':>6s} {'nested AUC':>11s} {'vs equal':>9s} "
              f"{'refit AUC':>10s}  <- nested is the honest one")
        results = {}
        for ridge in args.ridge:
            held = np.full(len(y), np.nan)
            for k in range(5):
                tr, te = f != k, f == k
                if te.sum() < 20 or len(np.unique(y[tr])) < 2:
                    continue
                w = fit_weights(R[:, tr], y[tr], ridge)
                held[te] = w @ R[:, te]
            ok = np.isfinite(held)
            nested = roc_auc_score(y[ok], held[ok])
            w_full = fit_weights(R, y, ridge)
            refit = roc_auc_score(y, w_full @ R)
            results[ridge] = (nested, w_full)
            print(f"  {ridge:6.2f} {nested:11.4f} {nested-eq:+9.4f} {refit:10.4f}")
        best_ridge = max(results, key=lambda r: results[r][0])
        nested, w_full = results[best_ridge]
        chosen[iso] = (members, w_full, best_ridge, nested, eq)
        top = ", ".join(f"{m} {v:.2f}" for m, v in
                        sorted(zip(members, w_full), key=lambda kv: -kv[1]) if v > 0.03)
        print(f"  chosen ridge {best_ridge}: {top}")
        if nested <= eq + 1e-4:
            print("  NOTE: nested AUC does not beat equal weights. Use equal weights;")
            print("        the apparent gain from free weights is fitting noise.")
        per_iso[iso] = (eq, nested)

    print(f"\n{'isoform':8s} {'equal-weight':>13s} {'weighted (nested)':>18s} "
          f"{'MCC equal':>10s} {'MCC weighted':>13s}")
    m_eq, m_w = [], []
    for iso, (eq, nested) in per_iso.items():
        a, b = K[iso]*dprime(eq), K[iso]*dprime(max(nested, eq))
        m_eq.append(a); m_w.append(b)
        print(f"{iso:8s} {eq:13.4f} {nested:18.4f} {a:10.4f} {b:13.4f}")
    print(f"{'MA-MCC':8s} {'':13s} {'':18s} {np.mean(m_eq):10.4f} {np.mean(m_w):13.4f}")
    print(f"actual last submission: 0.3147 (CYP3A4 0.4470, CYP2D6 0.1825)")

    if args.submission:
        test = pd.read_csv("data/cyp-challenge-TEST-BLINDED.csv")
        test.columns = [c.strip() for c in test.columns]
        sub = test[[C.SMILES_COL, C.ID_COL]].copy()
        for iso in C.TDI_ISOFORMS:
            members, w, ridge, nested, eq = chosen[iso]
            if nested <= eq + 1e-4:
                w = np.ones(len(members)) / len(members)
                print(f"{iso}: using EQUAL weights (nested gain not real)")
            cols = []
            for m in members:
                p = C.artifact(f"tdi_member_test_{m}_{args.tag}.csv")
                if not Path(p).exists():
                    raise SystemExit(
                        f"{p} not found -- re-run run_tdi_ensemble.py --predict "
                        "--save-oof with the updated script that saves per-member "
                        "blind-set scores.")
                t = pd.read_csv(p)
                cols.append(rank(sub[C.ID_COL].map(
                    dict(zip(t[C.ID_COL], t[f"{iso}_score"]))).to_numpy(float)))
            s = w @ np.vstack(cols)
            cut = float(np.quantile(s, 1 - RATE[iso]))
            sub[f"{iso}_is_TDI"] = s > cut
            print(f"  {iso}: predicted positive rate {sub[f'{iso}_is_TDI'].mean():.3f}")
        sub.to_csv(args.submission, index=False)
        print(f"written {args.submission}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
