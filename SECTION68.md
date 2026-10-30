# Section 68 -- The CYP3A4 re-threshold is real, free, and far smaller than Section 67 claimed

## What was claimed, and the correction

Section 67 ended by pointing at `tdi_splice.py`'s own table:

    CYP3A4  d' 1.290  prev 0.2915  opt rate 0.306  MCC at opt 0.4557
            last rate 0.229  last MCC 0.4470  gain +0.0087

and inferred macro 0.3804, i.e. rank 21. **That inference was wrong, and the error is
a baseline error of exactly the kind this project has now made three times.**

The splice table is anchored on `last MCC 0.4470` at `d' 1.290` -- the CYP3A4 column
*before* blendcp. The column we actually hold scored **0.4601 at d' 1.3206**. blendcp
moved d' up AND moved the call rate toward the optimum at the same time, so most of
the +0.0087 was already collected by blendcp itself. Recomputing the curve against the
column we hold:

| rate | CYP3A4 MCC | delta |
|---|---|---|
| 0.2426 (ours) | 0.4601 | -- |
| 0.2751 (whippoorwill) | 0.4645 | +0.0044 |
| 0.2800 | 0.4649 | +0.0048 |
| **0.3041 (optimum)** | **0.4657** | **+0.0056** |
| 0.3200 | 0.4653 | +0.0052 |
| 0.3600 | 0.4620 | +0.0019 |

The entire re-threshold headroom on CYP3A4 is **+0.0056 MCC**, not +0.0087. On CYP2D6
it is **+0.0004** (optimum 0.104 against our 0.1177 -- we are already there, which
confirms `tdi_blend_2d6.py`'s finding independently). Together: **macro 0.3760 -> 0.3790**,
which is rank 23, one place, and 0.0011 short of rank 21.

## The model behind the curve is confirmed, not assumed

Inverting whippoorwill's CYP3A4 confusion statistics gives d' 1.3209 at call rate
0.2751 and MCC 0.4644. The binormal curve fitted to OUR numbers predicts 0.4645 at
that rate. Four-decimal agreement on a competitor's row we did not use to fit anything.

The consequence is worth stating plainly: **whippoorwill has no model advantage over us
on CYP3A4.** Their d' and ours are identical to three decimals. The whole 0.0043 gap
between their 0.4644 and our 0.4601 is operating point. That is free MCC -- it is just
small, because the MCC-vs-rate curve is flat near its peak (a 4.5-point rate swing from
0.275 to 0.320 costs 0.0003).

## Why this does not get a submission slot

Three reasons, any one sufficient.

1. **Size.** +0.0030 macro is precisely the "little modification" the standing rule
   forbids spending a slot on.
2. **Realisation spread.** CYP3A4 has 651 scored rows (derived: 158 calls / 0.2426,
   cross-checked against prevalence 0.2915, 190 positives, TP 104.5, which reproduces
   prec 0.6613, rec 0.5501 and acc 0.7866). The single-set standard error on an MCC of
   0.46 at n=651 is about 0.03. An expected +0.0056 is 0.2 SE. It is positive in
   expectation and can still land negative -- which collides with "we do not submit
   worse than we had."
3. **Opportunity cost.** The slot is shared. See below.

## Rank 21 requires d', and only a little of it

Rank 21 is 0.3801. Our 0.3760 needs **+0.0083 on one endpoint**. In d' terms that is
almost nothing:

| endpoint | d' now | d' for rank 21 | ratio |
|---|---|---|---|
| CYP3A4 | 1.3206 | 1.3291 (at optimal rate) | x1.006 |
| CYP2D6 | 1.2266 | 1.2533 (at optimal rate) | x1.022 |

A 0.6% discriminability gain on CYP3A4 reaches rank 21. That is well inside what one
decorrelated member has delivered before: the chemprop shift model was worth d'
0.904 -> 1.008 locally at rank correlation 0.644. The pool itself has saturated
(Section 65: four structural changes, net +0.000 after Avalon), so the next member has
to come from a different model class, not another fingerprint.

## CYP2D6 is a d' gap, confirmed twice over

Inverted competitor d' on CYP2D6: c_cypher 1.2689, coconut-research 1.2577, dcvk 1.2412,
Rguava 1.2311, **us 1.2311**. We tie Rguava exactly and the three above us are genuinely
better discriminators. Our own rate is already at its optimum. There is no threshold
move on this endpoint -- only d'. Two board-measured attempts to change the CYP2D6
column have both lost (blend2d6 -> 0.202, structfree -> 0.2727), so the bar for
touching it again is a local signal we trust, and our OOF d' there is 0.283 against a
board-implied 1.224. We do not have that signal yet.

## Unlogged candidate: bestmix is a trap

`artifacts/submission_tdi_bestmix.csv` was built 12 Oct, never submitted, never scored,
never logged. It raises CYP2D6 calls from 64 to 85 -- effective rate 0.1177 -> 0.156.
The board-derived curve says that costs **-0.009 MCC on CYP2D6 at unchanged d'**. Do
not submit it. It is now recorded here so the file does not get picked up later as
"the newest thing on disk."

## The slot

The pIC50 board is sitting at probe2, **0.5614, rank 66**. Our own best is
`artifacts/submission_d6probe.csv`, **0.4737, rank 36**, already scored, byte-identical
on disk. Restoring it recovers **0.0877 macro ST-RAE at zero risk** -- the board is
deterministic, so this is not a bet, it is arithmetic.

0.0877 on the regression track against 0.0030 on the classification track, out of one
shared daily slot. Today's slot goes to `submission_d6probe.csv`.
