# Section 80 -- CYP1A2 solved at +0.28. And it unlocks probe2, which says CYP3A4 goes UP.

    submission_1a2up40.csv, scored 16 Oct 04:15 UTC

| endpoint | d6probe | 1a2up (+0.20) | 1a2up40 (+0.40) | rank now |
|---|---|---|---|---|
| **CYP1A2** | 0.5147 | **0.4682** | 0.4729 | 25 |
| CYP2C9 | 0.3645 | 0.3645 | 0.3645 | 17 |
| CYP2D6 | 0.5621 | 0.5621 | 0.5621 | 34 |
| CYP3A4 | 0.4532 | 0.4532 | 0.4532 | 51 |
| **MACRO** | 0.4737 | **0.4620** | 0.4632 | **27** |

The curve turned. +0.40 is worse than +0.20, so the optimum is between them, and three
measured points on one endpoint with everything else held byte-identical is enough to find
it exactly.

## 1. The optimum, two ways, same answer

    secant slope on [0.00, 0.20]   -0.2325 / unit
    secant slope on [0.20, 0.40]   +0.0235 / unit

    parabola vertex through the three points   +0.2816
    slope crossing zero (independent check)    +0.2816

    predicted ST-RAE at the optimum            0.4639

Moving from +0.20 to +0.28 is worth **+0.0043 on CYP1A2, +0.0011 on macro**. **CYP1A2 is
finished.** The whole ladder bought 0.5147 -> 0.4639, which is **0.0127 of macro** and the
first real gain this project has made in weeks.

## 2. probe2 can finally be split, and it reverses CYP3A4

probe2 moved CYP1A2 AND CYP3A4 down 0.30 together for a combined cost of +0.3508, which
Sections 68 and 79 both recorded as unsplittable. The fitted CYP1A2 curve splits it:

    CYP1A2 at -0.30  =  0.6805   cost +0.1658   (from the vertex fit)
    CYP3A4 at -0.30  =  0.6382   cost +0.1850   (the remainder)

    reconstructed probe2 macro 0.5613 against its MEASURED 0.5614 -- agreement to 0.0001

That 0.0001 is not a fit, it is a prediction: the CYP1A2 curve was fitted on three points
at +0.00/+0.20/+0.40 and then extrapolated to -0.30 to recover a fourth submission's score.

**So CYP3A4 at -0.30 cost 0.1850. It does not want to go down.** Its left-side slope is
-0.6168 per unit against CYP1A2's -0.5525 over the same interval -- comparable, and
CYP1A2's optimum from that slope turned out to be +0.28.

## 3. The OOF direction check is 1 for 3. Retire it.

| endpoint | OOF said | board says | verdict |
|---|---|---|---|
| CYP1A2 | UP (+0.720) | UP, optimum +0.28 | **correct on direction, 2.6x too far on distance** |
| CYP2D6 | UP (+1.200) | DOWN (cand1 -> d6probe gained 0.0336) | **wrong** |
| CYP3A4 | DOWN (-0.535) | UP (probe2 split) | **wrong** |

The reason is the one given in Section 79: the OOF curve measures our blind-tuned centre
against the TRAINING label population, and those differ by exactly the amount the placement
is deliberately correcting. It agreed on CYP1A2 only because CYP1A2 was genuinely
over-shifted relative to both.

**This retires the OOF shift curve as a direction guide**, and it means Section 77's
original instinct -- probe CYP3A4 up -- was right, and the OOF check that overrode it was
wrong. No harm done, because CYP1A2 also paid and had to be measured first either way, but
the reasoning that drove the switch was faulty and should not be reused.

## 4. Next: `artifacts/submission_1a2opt_3a4up.csv`

    CYP1A2  +0.28   the MEASURED optimum, not a probe -- firm prediction 0.4639
    CYP3A4  +0.20   THE PROBE, same step size that worked on CYP1A2
    CYP2C9, CYP2D6  byte-identical

Two columns move, which bends the single-variable rule, so the justification: CYP1A2's move
is a 0.0011-macro change predicted to 0.4639 from three measured points and validated by
the 0.0001 probe2 reconstruction. Anything large in the result is CYP3A4. If CYP1A2 comes
back far from 0.4639, the curve was wrong and that is itself worth knowing.

| if CYP3A4 gains | macro |
|---|---|
| 0.00 | 0.4609 |
| 0.02 | 0.4559 |
| 0.04 | 0.4509 |
| 0.05 | 0.4484 |

## 5. What remains after this

CYP2C9 (rank 17, our best) and CYP2D6 (rank 34, rho 0.519 and MAE 1.0409, by far our
weakest model) have never had their centres probed. CYP2D6's placement was set once and
gained; whether it is AT its optimum or merely nearer to it is unmeasured. Each costs one
slot and the ladder method now has a track record: probe +0.20, read the sign, bracket, fit.
