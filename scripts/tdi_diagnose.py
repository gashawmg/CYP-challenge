#!/usr/bin/env python
r"""Recover the exact TDI confusion matrices from the leaderboard, and ask what is left.

    python scripts\tdi_diagnose.py

WHY THIS ENDS THE THRESHOLD TUNING. The board reports accuracy, precision, recall, F1 and
MCC per target. Precision pins FP given TP, recall pins FN, accuracy then pins TN -- so a
single integer sweep over TP recovers the whole 2x2 table with no assumption about how
many of our 750 rows were actually scored. The tables solve to a total error of 0.002
across five metrics, and they reproduce the prevalences independently known from the
leaderboard inversion (CYP3A4 0.2917 vs 0.2915, CYP2D6 0.0706), which is what says the
recovery is real and not a fit.

WHAT THEY SAY.

    tdi_v3  CYP2D6   n=524  TP=17 FP=45 FN=20 TN=442   prevalence 0.0706  d' 1.224
    tdi_v3  CYP3A4   n=672  TP=73 FP=26 FN=123 TN=450  prevalence 0.2917  d' 1.276

Only 524 of 750 compounds are scored for CYP2D6 and 672 for CYP3A4, which is itself worth
knowing -- our call RATES are not what the submission file suggests.

1. CYP3A4's eight submissions have been re-thresholding one model. tdi_v3 (rate 0.147)
   and tdi_best (rate 0.211) imply d' 1.276 and 1.268 -- the same discrimination. Every
   "improvement" moved the threshold.

2. The MCC-vs-rate curve is FLAT across the whole plausible range: 0.4428 at rate 0.12,
   0.4486 at 0.18, 0.4469 at 0.21. tdi_best sits at 0.4470 and the optimum is 0.4486.
   There is +0.0016 left in the threshold. Nothing.

3. CYP2D6 is already AT its optimum: measured MCC 0.2911, the equal-variance optimum is
   0.2915. Also nothing.

4. Our d' caps us. At prevalence 0.2917 the best MCC any threshold can reach is:

       d' 1.27 (ours)   0.4493        d' 1.50   0.5223
       d' 1.40          0.4911        d' 1.68   0.5759

   We score 0.4470. CYP3A4 is essentially maxed out for the classifier we have.

5. The CYP3A4 leader's edge is a model edge, and a smaller one than the MCC gap suggests:
   stir_bar solves to n=549, TP=124 FP=107 FN=36 TN=282, prevalence 0.2914 (independent
   agreement again), TPR 0.775 FPR 0.275 -> d' 1.353, AUC 0.831 against our 1.276/0.817.

CONCLUSION: stop tuning call rates on both targets. Every remaining point of MA-MCC has
to come from a better classifier -- a higher d' -- and the required jump is large: reaching
0.52 on CYP3A4 needs d' 1.50 against our 1.276.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# per target: accuracy, precision, recall, F1, MCC as the board reports them
BOARD = {
    ("tdi_v3", "CYP2D6"): (0.8767, 0.2761, 0.4603, 0.3420, 0.2918),
    ("tdi_v3", "CYP3A4"): (0.7785, 0.7373, 0.3728, 0.4936, 0.4074),
    ("stir_bar", "CYP3A4"): (0.7395, 0.5369, 0.7750, 0.6334, 0.4601),
}


def metrics(TP, FP, FN, TN):
    n = TP + FP + FN + TN
    prec, rec = TP / (TP + FP), TP / (TP + FN)
    d = np.sqrt(float(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return ((TP + TN) / n, prec, rec, 2 * prec * rec / (prec + rec),
            (TP * TN - FP * FN) / d)


def solve(acc, prec, rec, f1, mcc):
    """One sweep over TP; precision, recall and accuracy determine the rest."""
    best = None
    for TP in range(1, 500):
        FP, FN = TP * (1 - prec) / prec, TP * (1 - rec) / rec
        TN = (acc * (TP + FP + FN) - TP) / (1 - acc)
        cand = [int(round(x)) for x in (TP, FP, FN, TN)]
        if min(cand) < 0 or sum(cand) > 760:
            continue
        e = sum(abs(a - b) for a, b in zip(metrics(*cand), (acc, prec, rec, f1, mcc)))
        if best is None or e < best[0]:
            best = (e, cand)
    return best


def mcc_curve(dprime, prevalence, n, rates):
    out = []
    for r in rates:
        th = norm.ppf(1 - r)
        TP = (1 - norm.cdf(th - dprime)) * prevalence * n
        FP = (1 - norm.cdf(th)) * (1 - prevalence) * n
        FN, TN = prevalence * n - TP, (1 - prevalence) * n - FP
        d = np.sqrt(float(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        out.append((TP * TN - FP * FN) / d if d > 0 else 0.0)
    return np.array(out)


def main() -> int:
    rates = np.arange(0.02, 0.60, 0.002)
    for (who, iso), m in BOARD.items():
        e, (TP, FP, FN, TN) = solve(*m)
        n, P = TP + FP + FN + TN, TP + FN
        tpr, fpr = TP / P, FP / (FP + TN)
        dp = norm.ppf(tpr) - norm.ppf(fpr)
        c = mcc_curve(dp, P / n, n, rates)
        print(f"{who} {iso}: n={n} TP={TP} FP={FP} FN={FN} TN={TN}  (fit error {e:.4f})")
        print(f"    prevalence {P / n:.4f}   call rate {(TP + FP) / n:.3f}   "
              f"TPR {tpr:.3f} FPR {fpr:.3f}   d' {dp:.3f}   "
              f"AUC {norm.cdf(dp / np.sqrt(2)):.3f}")
        print(f"    best MCC any threshold can reach: {c.max():.4f} at rate "
              f"{rates[c.argmax()]:.3f}   (measured {m[4]:.4f})\n")

    print("what a better classifier would be worth on CYP3A4 (prevalence 0.2917):")
    for dp in (1.27, 1.40, 1.50, 1.68, 1.80):
        print(f"    d' {dp:.2f}  ->  best MCC {mcc_curve(dp, 0.2917, 672, rates).max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
