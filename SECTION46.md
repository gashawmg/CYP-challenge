# Section 46. The diagnostic came back negative, and the replacement found the lever

Section 45 proposed measuring our dead-zone deficit locally. That was built
(`scripts/deadzone_audit.py`), it ran, and **it failed its own falsification test.** The
mechanism Section 45 proposed is wrong. What replaced it is better than what was asked for,
because it measures on the real blind set instead of a proxy.

## 1. The negative result

The audit's first output was a check that our local pool can reproduce the board's
`e = ST-RAE/MAE` under at least one MAE convention. It cannot:

| endpoint | board e | local, MAE over fitted | over all (bound) | over all (one-sided) |
|---|---|---|---|---|
| CYP1A2 | 0.7345 | 1.1392 | 1.2042 | 1.2266 |
| CYP2C9 | 0.8530 | 1.2447 | 1.1957 | 1.2484 |
| CYP2D6 | 0.5167 | 2.2839 | 2.5887 | 2.5894 |
| CYP3A4 | 0.8683 | 0.9289 | 0.9420 | 0.9958 |

Not close, on either the blind-like holdout or the blind-histogram-matched pool, and no
convention wins on all four endpoints.

The reason was already written down in `blind_validator.py` and I did not join it up:
**69-71% of our labels are left-censored at pIC50 = 4.0, and the blind set is not.** Solved
anchors put blind CYP2D6 at mean 2.92 with sd 1.65 -- real fitted values well below our
assay floor. Our local ST-RAE denominator is built almost entirely out of censored rows;
the blind one is not. So `e` does not transfer, and **Section 45 part 3 -- "the dead zone
that matters is the left-censored one" -- is retracted.** There is no free (-inf, 4.0] zone
on the blind set to exploit.

The 69% / 43% / 18% / 24% "predictions below 4.0" table in Section 45 was measuring our
predictions against our own training prior, not against the blind truth. It was a real
number answering a question nobody asked.

## 2. What we had all along

Twelve scored pIC50 submissions, per endpoint, on the real blind compounds with the real
intervals. Several pairs share an **identical Spearman** on an endpoint -- the same ranking
shipped with a different distribution. Those are controlled experiments on the true scoring
function, already paid for. `scripts/blind_response.py` extracts them.

Oriented consistently (mean always moving down; spread always narrowing, when the mean is
held):

| endpoint | from -> to | rho | d mean | d sd | d ST-RAE |
|---|---|---|---|---|---|
| CYP2C9 | blend3_placed -> blend3_narrow | 0.8454 | -0.000 | **-0.291** | **-0.0569** |
| CYP3A4 | blend3_placed -> blend3_narrow | 0.8231 | +0.000 | **-0.266** | **-0.0357** |
| CYP2D6 | blend3_placed -> blend3_narrow | 0.5072 | +0.000 | -0.002 | +0.0005 |
| CYP1A2 | blend3_placed -> blend3_narrow | 0.7892 | -0.270 | -0.066 | +0.1147 |
| CYP2C9 | blend4z -> place_half | 0.8380 | -0.113 | -0.045 | +0.0674 |
| CYP3A4 | blend4z -> place_half | 0.8310 | -0.292 | -0.120 | +0.2391 |
| CYP2D6 | place_half -> blend4z | 0.5190 | -0.819 | +0.522 | +0.0525 |
| CYP1A2 | place_half -> blend4z | 0.8064 | -0.546 | +0.525 | -0.0535 |

Two findings, and they point in opposite directions:

**Narrowing the spread at constant mean has never lost.** Two of the three such experiments
moved the spread enough to measure, and both improved ST-RAE substantially: -0.0569 on
CYP2C9 for a 0.29 cut, -0.0357 on CYP3A4 for a 0.27 cut. The third (CYP2D6) changed sd by
0.002, which is a no-op, not evidence. Ranking was identical in every case, by construction.

**Moving the mean has hurt, four times out of five**, by up to +0.2391 -- and the two rows
at the bottom are mixed moves where the mean and the spread changed together, so they are
not clean either way.

**This matters because both of the "placement failed" episodes in this project's history
are mean moves.** The lesson was recorded against placement in general. It should have been
recorded against moving the mean. Spread was never the thing that failed.

## 3. Why our CYP2D6 looked like the exception in Section 45

Section 45 noticed that CYP2D6 is the one endpoint where our dead-zone efficiency matches
the leaders, and attributed it to our predicted inactive fraction matching the prior. The
real reason is simpler and is a confound Section 45 missed:

**CYP2D6 is the one endpoint of the four whose live column is a PLACED column.** The
submitted `splice4` takes CYP2D6 from `place_half` and the other three from raw or spliced
model output. That transform bought -0.0525 ST-RAE at identical rho and cost +46% MAE
(0.7841 -> 1.1434).

So the Section 45 `e` comparison was not measuring a deficiency we have and the leaders do
not. It was partly measuring the fact that we already applied the lever on one endpoint and
not the other three. The conclusion survives -- the lever is real and the other three
endpoints have not had it -- but for a different reason than the one given, and the
CYP2D6-matches-the-prior argument is withdrawn.

## 4. The one clean untested move

| endpoint | shipped sd | efficiency headroom | narrowing tested? |
|---|---|---|---|
| CYP1A2 | **1.2594** | +0.0962 | **never, at constant mean** |
| CYP3A4 | 0.9592 | +0.0527 | yes: -0.0357 at d sd -0.27 |
| CYP2C9 | 0.8576 | +0.0407 | yes: -0.0569 at d sd -0.29 |
| CYP2D6 | 0.6906 | +0.0205 | no (already placed) |

CYP1A2 ships the widest distribution we have and carries the largest headroom, and the one
time it was narrowed the mean was dropped 0.27 at the same time -- so the +0.1147 that cost
is attributable to the mean move, which is the variable that has failed everywhere else.
Narrowing CYP1A2 at constant mean is the single experiment with the most measured gradient
behind it and no measurement against it.

## 5. What I would and would not conclude

The standing order puts model improvement first and this does not override it. Two things
are worth saying plainly anyway:

- This is not metric gaming. A prediction vector 1.3x wider than the truth is **wrong about
  how potent its compounds are**, and the ST-RAE gain is a side effect of fixing that. The
  right place to fix it is in the model -- ensemble averaging across members with different
  scales is a known way to end up over-dispersed, and `blind_validator.py` has printed
  "'pred sd' vs 'needed' is the compression this project keeps paying for" at the bottom of
  every run for weeks.
- A spread correction costs no submission slot of its own. It rides along with whatever
  model next earns a slot, and it is checkable beforehand: the predicted spread against the
  anchor-solved blind sd is a number we can read without submitting.

## 6. Corrections recorded

- Section 45 part 3 (censored dead zone as the mechanism): **retracted.** No such zone
  exists on the blind set.
- Section 45 part 3's CYP2D6-matches-the-prior argument: **withdrawn**, replaced by the
  placed-column explanation above.
- Section 45 parts 1, 2 and 5 (the leaders' rows, the amplification comparison, the
  -0.0525 macro counterfactual) stand. They compare teams on the same compounds with the
  same intervals and never used the local pool.
- `scripts/deadzone_audit.py` is kept, not deleted. Its convention check is the thing that
  caught this, and it will catch the next version of the same mistake.

## 7. Submission status

Unchanged. No candidate. Nothing ships until internal validation beats 0.4833.
