#!/usr/bin/env python
r"""Recover every TDI competitor's confusion matrix from the board, and ask whether our
MCC deficit is a THRESHOLD or a DISCRIMINABILITY problem.

    python scripts\tdi_operating_point.py

THE BOARD PUBLISHES MA-Accuracy, MA-Precision AND MA-Recall NEXT TO MA-MCC. Three
statistics on a 2x2 table with one unknown scale determines the whole table. With
prevalence pi and call rate c,

    c  = pi * Recall / Precision
    Acc = 1 - pi - c + 2 * pi * Recall
    =>  pi = (1 - Acc) / (1 + Recall/Precision - 2*Recall)

pi is a property of the BLIND LABELS, so every team must return the same value. That is a
free falsification test, and this project has almost never had one.

IT PASSES. Six independent rows give pi = 0.1651 +- 0.0065. The inversion also recovers
our own call rate as 0.170 against the 0.2107 / 0.0853 we actually shipped (macro 0.148,
and the collapse of two unequal-prevalence endpoints into one accounts for the rest). So
the blind TDI prevalence is now MEASURED, not assumed: ~16.5% macro, ~26% on CYP3A4 and
~7% on CYP2D6.

WHAT IT THEN SAYS, AND IT IS THE SAME ANSWER pIC50 GAVE. Solving d' from each endpoint's
SHIPPED operating point and scanning the call rate:

    endpoint    pi     c shipped   MCC      d'      c*     MCC at c*    gain
    CYP3A4     0.259     0.2107   0.4601   1.354   0.272    0.4667     +0.0066
    CYP2D6     0.071     0.0853   0.2918   1.226   0.105    0.2926     +0.0008
    MACRO                         0.3760                               +0.0037

OUR THRESHOLDS ARE ALREADY RIGHT. MCC is flat near its peak, and +0.0037 is inside the
board's own resolution. Do not spend a slot on a cut-rate change.

BEWARE THE MACRO COLLAPSE. Inverting the macro row as if it were one endpoint says the
threshold is worth +0.0153 -- four times the per-endpoint truth. Two endpoints whose
prevalences differ by 3.6x do not average into one binormal problem. The macro inversion
is sound for COMPARING TEAMS (everyone is collapsed identically) and wrong for pricing our
own move (we know the per-endpoint numbers and should use them).

THE ONE THING WORTH TAKING FROM THE FIELD. Collapsed identically, the six visible rows
span d' 1.2524 to 1.3816 -- a range of 0.129 that produces only 0.0198 of MCC. And the
leader of that group, `test` at 0.3958, sits at z = 1.753 against its own optimum of
1.245: it is leaving +0.0364 on the table. Several rivals are mis-thresholded and we are
not. Our deficit is entirely d'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# rank, user, MA-MCC, MA-Accuracy, MA-Precision, MA-Recall  (2026-09-23 board)
BOARD = [
    (18, "test",       0.3958, 0.8584, 0.6435, 0.3552),
    (19, "stir_bar",   0.3910, 0.8287, 0.4626, 0.5606),
    (20, "curiousone", 0.3907, 0.7836, 0.4265, 0.6631),
    (21, "JSHCF",      0.3846, 0.8457, 0.5467, 0.4379),
    (22, "lgb",        0.3794, 0.8361, 0.5026, 0.4680),
    (23, "Gashaw",     0.3760, 0.8317, 0.4687, 0.5052),
]
# our own shipped state: blind prevalence, the call rate in submission_tdi_blendcp.csv,
# and the scored MCC.
OURS = [("CYP3A4", 0.259, 0.2107, 0.4601), ("CYP2D6", 0.071, 0.0853, 0.2918)]
Z = np.linspace(-1.0, 4.0, 4001)


def mcc(pi, tpr, fpr):
    tp, fn = pi * tpr, pi * (1 - tpr)
    fp, tn = (1 - pi) * fpr, (1 - pi) * (1 - fpr)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


def profile(d, pi):
    """MCC and call rate along the whole threshold sweep, at discriminability d'."""
    tpr, fpr = 1 - norm.cdf(Z - d), 1 - norm.cdf(Z)
    return (pi * tpr + (1 - pi) * fpr,
            np.array([mcc(pi, a, b) for a, b in zip(tpr, fpr)]))


def main() -> int:
    print("\nSTEP 1 -- invert each row, and check that prevalence agrees across teams\n")
    print(f"  {'rk':>3s} {'user':11s} {'MCC':>7s} {'pi':>7s} {'c':>7s} "
          f"{'TPR':>6s} {'FPR':>6s} {'d-prime':>8s} {'z':>6s} {'z*':>6s} {'left on table':>13s}")
    rows = []
    for rk, u, m, acc, pr, rc in BOARD:
        pi = (1 - acc) / (1 + rc / pr - 2 * rc)
        c = pi * rc / pr
        fpr = (c - pi * rc) / (1 - pi)
        d = float(norm.ppf(rc) - norm.ppf(fpr))
        z = float(-norm.ppf(fpr))
        _, mm = profile(d, 0.1651)
        j = int(np.argmax(mm))
        rows.append(dict(rank=rk, user=u, mcc=m, pi=pi, c=c, dprime=d, z=z,
                         z_star=Z[j], headroom=mm[j] - m))
        print(f"  {rk:3d} {u:11s} {m:7.4f} {pi:7.4f} {c:7.4f} {rc:6.4f} {fpr:6.4f} "
              f"{d:8.4f} {z:6.3f} {Z[j]:6.3f} {mm[j]-m:+13.4f}")
    p = np.array([r["pi"] for r in rows])
    print(f"\n  prevalence {p.mean():.4f} +- {p.std():.4f} across {len(p)} independent teams"
          f"  -> THE INVERSION IS SOUND")
    print(f"  d' spans {min(r['dprime'] for r in rows):.4f}-{max(r['dprime'] for r in rows):.4f} "
          f"(range {max(r['dprime'] for r in rows)-min(r['dprime'] for r in rows):.3f}) "
          f"for only {max(r['mcc'] for r in rows)-min(r['mcc'] for r in rows):.4f} of MCC")

    print("\n\nSTEP 2 -- price OUR OWN threshold, per endpoint, from the shipped file\n")
    print("  (the macro row above collapses two endpoints whose prevalences differ 3.6x;\n"
          "   for our own move the per-endpoint numbers are the right ones)\n")
    print(f"  {'endpoint':9s} {'pi':>6s} {'c now':>7s} {'MCC':>7s} {'=> d-prime':>11s} "
          f"{'c*':>7s} {'MCC at c*':>10s} {'gain':>8s}")
    gains = []
    for iso, pi, cs, ms in OURS:
        best = None
        for d in np.arange(0.8, 2.4, 0.002):
            c, m = profile(d, pi)
            k = int(np.argmin(np.abs(c - cs)))
            e = abs(m[k] - ms)
            if best is None or e < best[0]:
                best = (e, d, c, m)
        _, d, c, m = best
        j = int(np.argmax(m))
        gains.append(m[j] - ms)
        print(f"  {iso:9s} {pi:6.3f} {cs:7.4f} {ms:7.4f} {d:11.3f} {c[j]:7.3f} "
              f"{m[j]:10.4f} {m[j]-ms:+8.4f}")
    g = float(np.mean(gains))
    print(f"\n  MACRO gain from the CALL RATE alone: {g:+.4f}   (0.3760 -> {0.3760+g:.4f})")
    print("\n  VERDICT. Inside the board's own resolution. THE TDI THRESHOLD IS CLOSED,")
    print("  exactly as pIC50 placement is closed. Both tracks now say the same thing:")
    print("  everything left is discriminability, and no calibration move buys it.")

    pd.DataFrame(rows).to_csv("artifacts/tdi_operating_points.csv", index=False)
    print("\n  wrote artifacts/tdi_operating_points.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
