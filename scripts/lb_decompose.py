#!/usr/bin/env python
r"""Split every leaderboard row into MODEL error and PLACEMENT skill.

    python scripts\lb_decompose.py                      # the rows we already have
    python scripts\lb_decompose.py --add "hfger,0.4374,0.7504,0.6xx"

WHY THIS EXISTS. Section 16 and Section 13 both priced our deficit with ABSORPTION,

    absorption = ST-RAE / sqrt(1 - rho_spearman^2)

and concluded we place badly. That statistic conflates two different failures and it
mislabelled ours. Spearman constrains only the ORDER; ST-RAE is charged on the VALUES. A
model can hold our exact Spearman and still be wrong by twice as much in log units, and
absorption books that as a placement fault.

THE STATISTIC THAT SEPARATES THEM. MA-MAE is published next to MA-ST-RAE. Both are means
over the same compounds and ST-RAE's denominator (the cost of predicting mean(y)) is a
constant fixed by the BLIND LABELS, identical for every team. So

    e = ST-RAE / MAE

is comparable across teams up to that shared constant, and it measures exactly one thing:
WHAT FRACTION OF YOUR ABSOLUTE ERROR LANDS OUTSIDE THE CONFIDENCE INTERVAL. Low e means
your error is parked in the dead zone. That is placement skill, isolated.

    MAE  = how big the errors are          -> the model
    e    = where the errors land           -> the placement

AND IT REVERSES Section 16'S VERDICT ON THE ONE SNAPSHOT WHERE ALL THREE COLUMNS EXIST:

    entry        ST-RAE   rho      MAE      absorption    e
    mike         0.4890   0.7325   0.6908     0.718      0.7079
    leebrubeck   0.4930   0.7326   0.6860     0.724      0.7187
    us           0.4962   0.7509   0.7100     0.751      0.6989   <- BEST e of the three

By absorption we were the worst of the three and "placed badly". By e we placed BETTER
than both and simply had errors 3% larger. Section 16 read the wrong instrument, and Section 18's
"placement is closed" -- reached independently, by a cost model calibrated to reproduce
board scores -- was right all along.

WHAT THIS DECIDES. hfger holds rho 0.7504 against our 0.7509 and scores 0.4374 against
0.4924: matched ranking, 0.0550 better. Under absorption that gap is unexplainable except
as placement. Under this decomposition it is a single number away from settled:

    hfger MA-MAE ~ 0.63  -> our placement equals theirs; their MODEL is better. No
                            calibration recovers it and none should be attempted.
    hfger MA-MAE ~ 0.71  -> matched errors, better parked; placement re-opens, and Section 18's
                            cost model needs a non-affine family.

READ THAT ONE COLUMN BEFORE ANY pIC50 WORK IS PLANNED. Per Section 19 nothing here is a
submission; it decides which track the next submission comes from.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# (entry, MA-ST-RAE, MA-rho, MA-MAE or None)
ROWS = [
    ("preheat-to-450", 0.3700, 0.7875, None),
    ("furtivepeony",   0.4301, 0.7626, None),
    ("leebrubeck-now", 0.4315, 0.7604, None),
    ("hfger",          0.4374, 0.7504, None),
    ("mike@23",        0.4890, 0.7325, 0.6908),
    ("leebrubeck@24",  0.4930, 0.7326, 0.6860),
    ("US blend4z",     0.4962, 0.7509, 0.7100),
    ("US splice4",     0.4924, 0.7509, None),
]
US_E = 0.4962 / 0.7100          # our measured dead-zone efficiency
US_MAE = 0.7100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add", nargs="*", default=[],
                    help='rows as "name,strae,rho[,mae]"')
    a = ap.parse_args()
    rows = list(ROWS)
    for s in a.add:
        p = [x.strip() for x in s.split(",")]
        rows.append((p[0], float(p[1]), float(p[2]),
                     float(p[3]) if len(p) > 3 and p[3] else None))

    d = pd.DataFrame(rows, columns=["entry", "strae", "rho", "mae"])
    d["absorption"] = d.strae / np.sqrt(1 - d.rho ** 2)
    d["e"] = d.strae / d.mae
    d["mae_if_e_like_us"] = d.strae / US_E
    print("\n  absorption = ST-RAE / sqrt(1-rho^2)   -- CONFLATED, kept only for continuity")
    print("  e          = ST-RAE / MAE             -- placement skill, isolated")
    print("  mae_hat    = what their MAE must be IF their placement matches ours\n")
    print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="   --"))

    print("\n\nDISCRIMINATOR -- read MA-MAE off the board for each unknown row\n")
    print(f"  our placement skill  e = {US_E:.4f}   our MA-MAE = {US_MAE:.4f}\n")
    print(f"  {'entry':16s} {'gap vs us':>10s} {'MAE => all model':>17s} "
          f"{'MAE => all placement':>21s}")
    for _, r in d[d.mae.isna()].iterrows():
        gap = 0.4924 - r.strae
        if gap <= 0:
            continue
        print(f"  {r.entry:16s} {gap:10.4f} {r.strae/US_E:17.4f} {US_MAE:21.4f}")
    print("\n  Their MA-MAE lands somewhere between those two columns.\n")

    # gap must be large enough that the share is not a ratio of two small numbers.
    # mike@23 is 0.0034 better than us and returns a "186% model share" -- noise over
    # noise. 0.02 of macro is roughly the board's own resolution.
    known = d[d.mae.notna() & (d.strae < 0.4924 - 0.02)]
    if len(known):
        print("VERDICT, for every row whose MA-MAE is known\n")
        print(f"  {'entry':16s} {'MAE':>7s} {'e':>7s} {'model share':>12s}  reading")
        for _, r in known.iterrows():
            allm = r.strae / US_E
            share = (US_MAE - r.mae) / (US_MAE - allm)
            if share > 0.60:
                verdict = "MODEL gap -- placement closed, spend on rho"
            elif share < 0.40:
                verdict = "PLACEMENT gap -- affine was the wrong family"
            else:
                verdict = "INDETERMINATE -- both, in similar measure"
            print(f"  {r.entry:16s} {r.mae:7.4f} {r.e:7.4f} {share:11.1%}  {verdict}")
        print()

    print("  SENSITIVITY -- how sharp is the reading? (hfger, gap 0.0550)\n")
    print(f"  {'their MA-MAE':>13s} {'model share':>12s}   verdict")
    allm = 0.4374 / US_E
    for m in (0.62, 0.64, 0.65, 0.66, 0.67, 0.68, 0.70, 0.71):
        share = (US_MAE - m) / (US_MAE - allm)
        tag = ("MODEL" if share > 0.60 else
               "PLACEMENT" if share < 0.40 else "indeterminate")
        print(f"  {m:13.2f} {share:11.1%}   {tag}")
    print("\n  The boundary sits at MA-MAE 0.6596 (60%) and 0.7064 (40%). The band is")
    print("  narrow, so READ THE COLUMN EXACTLY -- 0.65 and 0.67 give opposite answers.")

    print("\n  THE RULE THAT FOLLOWS (Section 19). Model share above ~60%: placement is closed for")
    print("  a third time and every remaining slot belongs to rho. Under ~40%: Section 18's")
    print("  affine-only scan was the wrong family and a piecewise map -- shrink where")
    print("  intervals are wide, hold where they are narrow -- is the candidate,")
    print("  back-tested against blend3_placed -> blend3_narrow before it ships.")
    print("  In between: read a SECOND row (furtivepeony, preheat-to-450) and break the")
    print("  tie, rather than acting on one ambiguous number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
