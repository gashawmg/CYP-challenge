#!/usr/bin/env python
"""Predict what a TDI score column would SCORE, before spending a submission on it.

    python scripts/tdi_validator.py --tag tdiK
    python scripts/tdi_validator.py --tag tdiK --members lgb_clf pooled RANK-BLEND
    python scripts/tdi_validator.py --oof artifacts/oof_tdi_chemprop.csv

THE GAP THIS FILLS. The TDI track could already predict one thing very well and one thing
not at all:

    quantity                              predicted   actual
    binormal MCC, CYP3A4 at scored cut      0.4473    0.4470
    binormal MCC, CYP2D6 at scored cut      0.2189    0.2180
    tdi_recut, the THRESHOLD half          +0.0087   +0.0087
    tdi_recut, the SCORE-COLUMN half    not predicted  -0.0358

The threshold model is excellent. But swapping in a different score column was un-modelled,
and it cost 0.036 -- the whole reason `submission_tdi_recut` went 0.3325 -> 0.3187. That is
the hole: nothing converted a LOCAL model quality into an expected leaderboard MCC.

WHY OUT-OF-FOLD AUC CANNOT BE USED DIRECTLY. Converting AUC to discriminability
(d' = sqrt(2) * Phi^-1(AUC)) and comparing with the d' recovered from our scored
submission:

    endpoint   OOF d'   blind d'   ratio
    CYP3A4      1.289     1.290     1.00
    CYP2D6      0.436     0.980     2.25

CYP3A4 transfers exactly. CYP2D6 out-of-fold UNDERSTATES the model by more than double,
because our folds carry 21.6% positives against a blind 7.06% and the positives themselves
differ. So a raw OOF comparison on CYP2D6 is being made on the wrong population, and the
correction has to be explicit rather than hoped for.

WHAT THIS DOES.

  1. Bootstrap the out-of-fold AUC of each candidate column, so a difference between two
     candidates has a NOISE FLOOR rather than a bare decimal.
  2. Convert to d', then apply the per-endpoint shrinkage above.
  3. Feed the corrected d' and the recovered blind prevalence into the binormal MCC model
     -- the one that reproduces both scored MCCs to 0.0008 -- at the optimal cut.
  4. Report predicted MA-MCC with a band, and say plainly whether the candidate clears the
     standing best by more than that band.

HOW MUCH TO TRUST IT. The shrinkage is fitted on ONE scored submission per endpoint. On
CYP3A4 it is ~1.0 and barely doing anything, so that half is safe. On CYP2D6 it is a
single 2.25 with no error bar of its own, so treat the CYP2D6 number as a direction with
a magnitude, not a measurement -- and prefer candidates whose advantage survives even at
shrinkage 1.0 (reported as the pessimistic column). That is the honest reading, and it is
still far better than the nothing that was there when tdi_recut was submitted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from cypkit import config as C

# Recovered by inverting every competitor's published accuracy / precision / recall;
# identical to four decimals across all of them.
PREVALENCE = {"CYP3A4": 0.2915, "CYP2D6": 0.0706}
# Our own blind discriminability, from submission_tdi_best (MA-MCC 0.3325).
BLIND_DPRIME = {"CYP3A4": 1.290, "CYP2D6": 0.980}
BEST_MCC = {"CYP3A4": 0.4470, "CYP2D6": 0.2180}
BEST_MA = float(np.mean(list(BEST_MCC.values())))
# OOF d' of the pool that produced those scores. Ratio = blind / OOF.
REF_OOF_DPRIME = {"CYP3A4": 1.289, "CYP2D6": 0.436}
SHRINK = {k: BLIND_DPRIME[k] / REF_OOF_DPRIME[k] for k in BLIND_DPRIME}


def d_prime(auc):
    return np.sqrt(2) * norm.ppf(np.clip(auc, 1e-6, 1 - 1e-6))


def mcc_binormal(d, prev, t):
    tpr, fpr = 1 - norm.cdf(t - d), 1 - norm.cdf(t)
    TP, FN = prev * tpr, prev * (1 - tpr)
    FP, TN = (1 - prev) * fpr, (1 - prev) * (1 - fpr)
    den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return ((TP * TN - FP * FN) / den if den > 0 else 0.0), TP + FP


def best_mcc(d, prev):
    ts = np.linspace(-3, 5, 4000)
    v = [mcc_binormal(d, prev, t) for t in ts]
    i = int(np.argmax([x[0] for x in v]))
    return v[i][0], v[i][1]


def rank01(x):
    r = pd.Series(x).rank().to_numpy()
    return (r - r.min()) / max(r.max() - r.min(), 1e-9)


def bootstrap_auc(y, s, n=400, seed=0):
    rng = np.random.default_rng(seed)
    out, k = [], len(y)
    for _ in range(n):
        i = rng.integers(0, k, k)
        if y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], s[i]))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None,
                    help="read artifacts/tdi_member_oof_<iso>_<tag>.csv")
    ap.add_argument("--oof", nargs="*", default=None,
                    help="score files carrying Molecule_Name, a label column, and one "
                         "or more score columns")
    ap.add_argument("--members", nargs="*", default=None,
                    help="restrict to these columns; RANK-BLEND is always added")
    ap.add_argument("--draws", type=int, default=400)
    args = ap.parse_args()

    tables = {}
    if args.tag:
        for iso in C.TDI_ISOFORMS:
            p = C.artifact(f"tdi_member_oof_{iso}_{args.tag}.csv")
            if Path(p).exists():
                tables[iso] = pd.read_csv(p)
    for f in (args.oof or []):
        d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]
        for iso in C.TDI_ISOFORMS:
            sc = [c for c in d.columns if c.startswith(iso) and "score" in c.lower()]
            yc = [c for c in d.columns if c in ("y", f"{iso}_y")]
            if sc and yc:
                t = d[[C.ID_COL, yc[0]] + sc].rename(columns={yc[0]: "y"})
                tables[iso] = (t if iso not in tables else
                               tables[iso].merge(t.drop(columns=["y"]), on=C.ID_COL))
    if not tables:
        raise SystemExit("nothing to score -- give --tag or --oof")

    print(f"shrinkage applied (blind d' / OOF d'): "
          f"{ {k: round(v, 2) for k, v in SHRINK.items()} }")
    print(f"standing best MA-MCC {BEST_MA:.4f}   {BEST_MCC}\n")

    results = {}
    for iso, t in tables.items():
        # A merged multi-isoform file carries rows that only one isoform measured, so y
        # is NaN there. astype(int) on NaN does not raise -- it produces a vast garbage
        # integer, which then printed a positive rate of -2.4e20 rather than failing.
        t = t[np.isfinite(pd.to_numeric(t["y"], errors="coerce"))].reset_index(drop=True)
        y = t["y"].to_numpy(float).astype(int)
        cols = [c for c in t.columns if c not in (C.ID_COL, "y")]
        if args.members:
            cols = [c for c in cols if c in args.members] or cols
        scores = {c: t[c].to_numpy(float) for c in cols}
        if len(cols) > 1:
            scores["RANK-BLEND"] = np.mean([rank01(scores[c]) for c in cols], axis=0)

        print("=" * 78)
        print(f"{iso}   n={len(y)}  positives {y.mean()*100:.1f}%  "
              f"(blind {PREVALENCE[iso]*100:.1f}%)\n")
        print(f"  {'candidate':16s} {'OOF AUC':>16s} {'OOF d':>7s} {'corrected d':>12s} "
              f"{'pred MCC':>9s} {'pessimistic':>12s}")
        rows = []
        for nm, s in scores.items():
            ok = np.isfinite(s)
            a = roc_auc_score(y[ok], s[ok])
            bs = bootstrap_auc(y[ok], s[ok], args.draws)
            d0 = d_prime(a)
            m_cor = best_mcc(d0 * SHRINK[iso], PREVALENCE[iso])[0]
            m_pes = best_mcc(d0, PREVALENCE[iso])[0]          # shrinkage = 1.0
            lo, hi = np.percentile(bs, [2.5, 97.5])
            m_lo = best_mcc(d_prime(lo) * SHRINK[iso], PREVALENCE[iso])[0]
            m_hi = best_mcc(d_prime(hi) * SHRINK[iso], PREVALENCE[iso])[0]
            rows.append(dict(name=nm, auc=a, d=d0, mcc=m_cor, pes=m_pes,
                             lo=m_lo, hi=m_hi))
            print(f"  {nm:16s} {a:8.4f}[{lo:.3f},{hi:.3f}] {d0:7.3f} "
                  f"{d0*SHRINK[iso]:12.3f} {m_cor:9.4f} {m_pes:12.4f}")
        results[iso] = sorted(rows, key=lambda r: -r["mcc"])
        b = results[iso][0]
        print(f"\n  best: {b['name']}  predicted MCC {b['mcc']:.4f} "
              f"[{b['lo']:.4f}, {b['hi']:.4f}]   scored last time {BEST_MCC[iso]:.4f}")

    if len(results) == len(C.TDI_ISOFORMS):
        ma = np.mean([results[i][0]["mcc"] for i in C.TDI_ISOFORMS])
        lo = np.mean([results[i][0]["lo"] for i in C.TDI_ISOFORMS])
        hi = np.mean([results[i][0]["hi"] for i in C.TDI_ISOFORMS])
        pes = np.mean([results[i][0]["pes"] for i in C.TDI_ISOFORMS])
        print("\n" + "=" * 78)
        print("VERDICT\n")
        print(f"  predicted MA-MCC at the optimal cut : {ma:.4f}  [{lo:.4f}, {hi:.4f}]")
        print(f"  same, with NO shrinkage correction  : {pes:.4f}")
        print(f"  standing best                       : {BEST_MA:.4f}")
        gain = ma - BEST_MA
        print(f"  predicted gain {gain:+.4f}")
        if lo > BEST_MA and pes > BEST_MA:
            print("\n  CLEARS the standing best on both the corrected AND the "
                  "uncorrected reading,\n  and the bootstrap lower bound is above it. "
                  "This is a submission.")
        elif ma > BEST_MA:
            print("\n  Ahead on the corrected reading only. The CYP2D6 shrinkage is "
                  "fitted on ONE\n  submission, so an advantage that vanishes in the "
                  "'no shrinkage' column is a bet\n  on that single number, not a "
                  "measurement. Prefer a candidate that clears both.")
        else:
            print("\n  Does NOT clear the standing best. Do not spend a slot.")
    print("\nAlways diff the resulting labels against the standing best before "
          "submitting:\n  the threshold half of a change is predictable to ~0.001; "
          "the score-column half\n  is what this script estimates, and it is the half "
          "that cost 0.036 once already.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
