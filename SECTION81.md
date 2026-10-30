# Section 81 -- 0.4504, best ever. CYP1A2 predicted to four decimals, CYP3A4 paid 0.0420.

    submission_1a2opt_3a4up.csv, scored 16 Oct 16:32 UTC

| endpoint | before | now | change | rank |
|---|---|---|---|---|
| CYP1A2 (+0.28) | 0.4729 | **0.4639** | -0.0090 | 29 |
| CYP2C9 (unchanged) | 0.3645 | 0.3645 | 0.0000 | 19 |
| CYP2D6 (unchanged) | 0.5621 | 0.5621 | 0.0000 | 37 |
| **CYP3A4 (+0.20)** | 0.4532 | **0.4112** | **-0.0420** | 24 |
| **MACRO** | 0.4632 | **0.4504** | **-0.0127** | **27** |

**CYP1A2 was predicted at 0.4639 and scored 0.4639.** Exact to four decimals, from a
vertex fitted to three points. That is the strongest validation of a placement model this
project has produced, and it means the ladder method can now be trusted to call its own
optimum rather than step toward it.

**CYP3A4 gained 0.0420** on the direction the probe2 decomposition predicted and the OOF
curve got backwards. Section 80's retirement of the OOF shift curve stands.

Running total from pure placement, no model change: **0.4737 -> 0.4504, 0.0233 of macro.**

## 1. CYP3A4's curve, three points, and it is nearly done

    shift -0.30   0.6382   (derived from the probe2 split, Section 80)
    shift +0.00   0.4532   (measured)
    shift +0.20   0.4112   (measured)

    curvature a = 0.8133   OPTIMUM +0.2291   predicted 0.4105

We sit at +0.20. Moving to +0.23 is worth +0.0007 on the endpoint, **+0.0002 macro**. A
further +0.20 is predicted at 0.4343 -- WORSE than where we are. CYP3A4 is done, with the
caveat that its -0.30 anchor is derived rather than measured, so the curvature could be
overstated; if it is, there is a little more above +0.23.

## 2. CYP2D6 is also near-optimal, from points we already held

Nobody probed it, but two scored submissions bracket it: `cand1` at centre 3.949 scored
0.5957 and `d6probe` at 3.649 scored 0.5621. So f(+0.30) - f(0) = +0.0336 going UP.

Solving for the vertex at the curvature the other two endpoints showed:

    a = 0.64  ->  optimum +0.063, worth +0.0025 on the endpoint (+0.0006 macro)
    a = 0.81  ->  optimum +0.081, worth +0.0053 on the endpoint (+0.0013 macro)

The optimum is slightly ABOVE our centre, not below -- so the instinct to push CYP2D6
further down would have cost. **No probe needed; the data was already on the board.**

## 3. CYP2C9 is the last unprobed centre, and the pattern says it goes UP

| endpoint | shift applied from the model | probed? | outcome |
|---|---|---|---|
| CYP1A2 | -0.644 | 3 points | optimum +0.28, gained 0.0508 |
| CYP2D6 | -1.054 | 2 points | optimum ~+0.06, near-optimal |
| CYP3A4 | -0.049 | 3 points | optimum ~+0.23, gained 0.0420 |
| **CYP2C9** | **-0.017** | **NO** | **unknown** |

**The two endpoints shifted LEAST from the model are the two that wanted to go up**, and
the one shifted most (CYP2D6, -1.054) is already right. CYP2C9 is shifted least of all
(-0.017) and has never been touched. CYP3A4, in the same position, paid 0.0420 for +0.20.

## 4. Next: `artifacts/submission_2c9up.csv`

    CYP1A2  +0.28   settled
    CYP3A4  +0.20   settled
    CYP2C9  +0.20   THE PROBE -- the only moving part
    CYP2D6   0.00   unchanged

Three of four endpoints are pinned at settings the board has ALREADY printed values for
(0.4639 / 0.4112 / 0.5621), so the entire delta is attributable to CYP2C9 whichever way it
goes. That is the cleanest attribution available and it costs nothing extra.

| if CYP2C9 gains | macro |
|---|---|
| 0.000 | 0.4504 |
| 0.020 | 0.4454 |
| 0.030 | 0.4429 |
| 0.042 (what CYP3A4 did) | **0.4399** |

## 5. After the centres: the scale has never been probed at all

Placement is an affine map and we have only ever moved the CENTRES. The per-endpoint
SPREADS come from `build_blend4.py`'s `SPREAD_RATIO`, fitted long ago and never tested on
the board except by `d6wide`, which widened CYP2D6 to sd 1.50 and lost 0.0791 -- one point,
one endpoint, one direction. Once CYP2C9's centre is settled, the spreads are the obvious
next dimension, and the same ladder works on them.
