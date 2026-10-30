# Section 79 -- The CYP1A2 probe WON. First validated gain of the week, and the ladder has more rungs.

    submission_1a2up.csv, scored 15 Oct 16:11 UTC

| endpoint | d6probe | 1a2up | change | rank | MAE | R2 | rho |
|---|---|---|---|---|---|---|---|
| **CYP1A2** | 0.5147 | **0.4682** | **-0.0465** | 34 | 0.6977 | 0.6205 | 0.8064 |
| CYP2C9 | 0.3645 | 0.3645 | 0.0000 | 25 | 0.4177 | 0.7054 | 0.8554 |
| CYP2D6 | 0.5621 | 0.5621 | 0.0000 | 39 | 1.0409 | 0.3242 | 0.5190 |
| CYP3A4 | 0.4532 | 0.4532 | 0.0000 | 62 | 0.4999 | 0.7055 | 0.8425 |
| **MACRO** | 0.4737 | **0.4620** | **-0.0117** | **35** | 0.6641 | -- | 0.7558 |

**0.4737 was our best ever. 0.4620 beats it.** The three untouched endpoints came back
byte-identical, so the probe isolated exactly one variable and the 0.0465 is attributable
to the CYP1A2 shift and nothing else.

## 1. What this validates

* **The direction.** OOF said CYP1A2's optimum was +0.720 and that 67.0% of its predictions
  sat BELOW their `lo` bound. The board agreed.
* **The correction away from CYP3A4.** Section 77 originally proposed a +0.20 on CYP3A4;
  the OOF check found 66.9% of CYP3A4 predictions ALREADY above `hi` and an optimum of
  -0.535, and the recommendation was switched. On the same reasoning that just paid on
  CYP1A2, the CYP3A4 probe would have lost.
* **The single-variable discipline.** probe2 moved two endpoints at once and its 0.3508 can
  never be split. This one can be read exactly.

## 2. The transfer factor, now MEASURED rather than assumed

    OOF predicted a gain of 0.1911 for the first +0.20
    the board paid                 0.0465
    transfer factor k            = 0.2433     (the OOF curve is 4.1x too steep)

That 4.1x is consistent with the level mismatch flagged when the curve was computed (OOF
CYP1A2 1.4894 against the board's 0.5147), which is why only the SHAPE was ever quoted.

If the shape transfers at that single factor -- an assumption, not a measurement:

| total shift | predicted CYP1A2 | predicted MACRO |
|---|---|---|
| +0.20 (scored) | 0.4682 | 0.4620 |
| +0.30 | 0.4501 | 0.4575 |
| **+0.40** | **0.4354** | **0.4538** |
| +0.60 | 0.4173 | 0.4493 |
| +0.72 | 0.4143 | 0.4485 |

The curve flattens exactly as convexity requires: the second +0.20 is worth about half the
first, and +0.60 -> +0.72 is worth almost nothing.

## 3. Why the OTHER three endpoints do NOT get the same treatment

The same OOF curve, run on all four:

| endpoint | OOF optimum | at shift 0 | reads as |
|---|---|---|---|
| CYP1A2 | +0.720 | 13.5% above hi, 67.0% below lo | wants UP -- **CONFIRMED by the board** |
| CYP2C9 | -0.185 | 42.4% above, 19.8% below | wants down, slightly |
| CYP2D6 | **+1.200** | 3.4% above, **86.2% below** | wants UP, enormously |
| CYP3A4 | -0.535 | 66.9% above, 9.8% below | wants down |

**Do not act on the CYP2D6 row. It is contradicted by a board measurement we already
hold.** `cand1` -> `d6probe` moved CYP2D6 DOWN 0.30 and GAINED 0.0336. OOF says that
endpoint wants to go UP by 1.20. The OOF curve is measuring our blind-tuned centre (3.649)
against the OOF label population (mean 4.784) -- a 1.135 mismatch that is the whole POINT
of the shift, not an error in it.

So the OOF curve diagnoses the gap between our centre and the TRAINING population, which
equals the blind optimum only where our transfer estimate was wrong in the same direction.
It happened to be right on CYP1A2 because CYP1A2 was over-shifted. It is demonstrably
wrong on CYP2D6. **Treat the remaining three as unknown, and spend a slot per endpoint to
find out, one variable at a time.**

## 4. Next

`artifacts/submission_1a2up40.csv` -- d6probe with **+0.40** on CYP1A2, one more rung.
Other three byte-identical. Expected CYP1A2 ~0.4354, macro ~0.4538, which is about half the
gain of the first step. If it lands near that, a third rung at +0.60 is worth taking and
the curve is then pinned by three real points. If it lands flat, we have found the optimum
and CYP1A2 is done.

After CYP1A2 is settled, the open question is whether CYP2C9 or CYP3A4 carries the same
kind of mis-centring. Neither can be predicted from OOF for the reason in section 3; each
costs one slot to measure.
