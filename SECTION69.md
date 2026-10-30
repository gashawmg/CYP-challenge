# Section 69 -- The placement lever is dead. The cost is in the tails, and 10% of compounds carry 40% of it

The leaderboard now publishes MA-MAE, MA-R2 and MA-Spearman alongside MA-ST-RAE. That
turns the six rows above us into a controlled experiment, because their rankings are
close to ours and we can finally separate ranking quality from conversion.

## 1. Our ranking is the best in the neighbourhood. Our conversion is the worst.

Absorption is ST-RAE divided by the sqrt(1 - rho^2) the ranking forces -- the fraction
of the irreducible error the metric actually charges us for.

| rank | user | ST-RAE | MA-MAE | MA-R2 | rho | absorption |
|---|---|---|---|---|---|---|
| 29 | Ligand Pro | 0.4625 | 0.6811 | 0.5622 | 0.7323 | **0.6792** |
| 30 | iuo | 0.4637 | 0.6492 | 0.5953 | 0.7326 | 0.6813 |
| 31 | cypextzp-r4g | 0.4644 | 0.6614 | 0.5831 | 0.7361 | 0.6861 |
| 32 | nova | 0.4663 | 0.6361 | 0.6013 | 0.7446 | 0.6986 |
| 33 | admetchallenger | 0.4700 | 0.6458 | 0.5912 | 0.7362 | 0.6945 |
| 34 | q7kp2 | 0.4729 | 0.6473 | 0.5867 | 0.7326 | 0.6948 |
| **35** | **Gashaw (d6probe)** | **0.4737** | 0.6664 | 0.5872 | **0.7558** | **0.7234** |

**Every competitor ahead of us has a LOWER Spearman than we do.** Ligand Pro ranks 0.023
worse and scores 0.0112 better. At the field's best absorption our existing ranking would
score **0.4447** -- about twelve ranks, with the model untouched. That is the size of the
prize. The rest of this section is about where it is not, and where it is.

## 2. It is not in a cleverer monotone map. Measured and closed.

`fit_placement.py` reported a free-form monotone transform worth 3% of absorption. Re-run
nested on the frozen folds against `oof_mhens3` (macro OOF ST-RAE 0.6477, our best local
column), the 3% reproduces exactly -- **against RAW predictions**:

| | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | MACRO |
|---|---|---|---|---|---|
| raw | 0.7619 | 0.5734 | 0.8340 | 0.4213 | 0.6477 |
| + best shift | 0.7477 | 0.5608 | 0.8191 | 0.4126 | 0.6350 |
| + free monotone | 0.7421 | 0.5541 | 0.8208 | 0.3983 | 0.6288 |

+0.0189 macro, 95% CI [+0.0107, +0.0273], P(worse) 0.0%. It looks like a submission.

**It is not.** Our pipeline does not submit raw predictions -- it submits an affinely
placed column. Fitting the affine properly, nested, and comparing like with like:

| | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | MACRO |
|---|---|---|---|---|---|
| affine | 0.7449 | 0.5475 | 0.8201 | 0.3969 | **0.6273** |
| free monotone | 0.7421 | 0.5541 | 0.8208 | 0.3983 | 0.6288 |
| monotone minus affine | +0.0028 | **-0.0066** | -0.0007 | -0.0014 | **-0.0015** |

Monotone over affine: **-0.0015 macro, 95% CI [-0.0070, +0.0040], P(worse) 70.8%.** The
entire 2.91% was affine's gain, which we already collect. A free-form monotone map adds
nothing and probably costs.

**This retires the whole family.** Any rule that computes a new prediction from the old
one alone -- a shift, a scale, a piecewise curve, a conditional tail expansion gated on
the prediction -- is a monotone map of p, and is covered by this result. It also explains
the board record: `blend3_narrow`, `place_half`, `v3_affine80` and `d6wide` are four
scale moves and four losses.

## 3. Where the cost actually is: 10% of compounds, 40% of the bill

| endpoint | rows | top 1% of cost | top 5% | top 10% | top 25% | rows costing ZERO |
|---|---|---|---|---|---|---|
| CYP1A2 | 1412 | 5.3% | 21.0% | 36.0% | 66.0% | 21.5% |
| CYP2C9 | 1285 | 7.2% | 27.4% | 43.5% | 75.9% | 40.9% |
| CYP2D6 | 1490 | 4.9% | 21.3% | 37.2% | 66.3% | 20.7% |
| CYP3A4 | 2334 | 8.3% | 28.1% | 45.4% | 77.9% | 42.0% |

A quarter of the compounds carry three quarters of the metric. Up to 42% of rows are
already free and no amount of further work on them returns anything.

## 4. The expensive compounds are the tails, and the model does not move for them

Splitting each endpoint at the 90th percentile of cost:

| endpoint | group | y mean | **y sd** | **pred mean** | median width |
|---|---|---|---|---|---|
| CYP1A2 | top 10% cost | 4.157 | **2.119** | **5.159** | 2.10 |
| | the rest | 5.045 | 0.774 | 5.213 | 0.32 |
| CYP2C9 | top 10% cost | 4.594 | **1.480** | **4.806** | 0.40 |
| | the rest | 4.579 | 0.660 | 4.769 | 0.53 |
| CYP2D6 | top 10% cost | 4.468 | **2.115** | **4.989** | 0.31 |
| | the rest | 4.817 | 0.647 | 4.931 | 0.27 |
| CYP3A4 | top 10% cost | 4.856 | 1.001 | **4.127** | 0.22 |
| | the rest | 4.011 | 1.070 | 4.115 | 0.42 |

The expensive compounds have two to three times the label spread of everyone else, and
**our predicted mean on them is within 0.06 of our predicted mean on the rest.** We
answer the mean for the extremes. A compound at 1.9 or 7.9 predicted at 5.2 costs three
log units, and ten percent of those eat forty percent of the budget.

Prediction sd against label sd: **0.368 (CYP1A2), 0.483 (CYP2C9), 0.410 (CYP2D6), 0.808
(CYP3A4)**. We are compressed to roughly 40% of the label spread on three endpoints.

This is also why every global widening lost. Expanding helps the 10% and hurts the 90%,
and the 90% is where the narrow intervals are. The expansion has to be **conditional**.

## 5. A correction to `interval_weighting.py`'s premise

Its header says the narrowest quartile is up to 60% of cost and the widest 11-18%, so
weight by 1/width. Measured on the current column, that holds **only on CYP3A4**:

| endpoint | narrowest quartile share of cost | widest quartile | widest: fraction landing inside |
|---|---|---|---|
| CYP1A2 | 29.2% | **46.4%** | 8.5% |
| CYP2C9 | 39.1% | **37.9%** | 22.7% |
| CYP2D6 | 28.5% | **33.8%** | 18.5% |
| CYP3A4 | 39.6% | 7.0% | 84.9% |

Three of four endpoints are U-shaped, because the widest intervals belong to the extreme
compounds of Section 4 -- badly determined curves on compounds we miss by more than the
interval is wide. **1/width weighting would down-weight the rows carrying 34-46% of the
cost on three endpoints.** Do not run it as written.

## 6. The gate does not come from ensemble disagreement

A conditional expansion needs a signal that is not a function of p (Section 2 rules those
out). Seed spread across the four `mhens3` seeds was the cheapest candidate. It is
worthless:

| endpoint | rho(spread, cost) | AUC for top-decile cost | AUC for top-decile \|y - mean\| |
|---|---|---|---|
| CYP1A2 | 0.042 | 0.500 | 0.543 |
| CYP2C9 | 0.013 | 0.571 | 0.548 |
| CYP2D6 | 0.090 | 0.512 | 0.507 |
| CYP3A4 | 0.017 | 0.538 | 0.517 |

The ensemble does not know which compounds it is about to lose money on.

## 7. What this leaves, and what it is worth

A **dedicated extremeness model on features** -- a classifier for "this compound is in
the tail of the label distribution" -- trained separately, used only to license expansion
on the compounds it is confident about. Encouragingly, our own point prediction already
carries some of that signal (AUC for top-decile |y - mean|: 0.726 CYP1A2, 0.796 CYP2C9,
0.837 CYP3A4, 0.570 CYP2D6), but it cannot be used on its own, because exploiting it
through p is a monotone map. It has to be an independent head whose output gates the
expansion.

### RETRACTED, same day, before anything was built

An earlier version of this section put the prize at blind 0.3797, from the arithmetic
"what if the top-cost decile cost half as much". **That is an accounting identity, not an
achievable score, and it should never have been written as a target.** It uses the labels
to choose the decile and then assumes the cost on it can simply be reduced.

Simulating the actual mechanism -- a gate that flags 10% of compounds as extreme, with an
expansion factor fitted out of fold on flagged rows only -- gives:

| gate AUC | OOF macro gain | relative | implied blind |
|---|---|---|---|
| 0.70 | +0.0004 | +0.06% | 0.4734 |
| 0.80 (what our own prediction already achieves) | +0.0028 | +0.43% | 0.4717 |
| 0.90 | +0.0098 | +1.51% | 0.4666 |
| 0.95 | +0.0184 | +2.84% | 0.4602 |
| **1.00, a perfect oracle** | **+0.0366** | **+5.66%** | **0.4469** |

**A gate that knows the answer with certainty is worth 0.4469, not 0.3797.** Knowing a
compound is extreme does not say WHERE in the tail it sits; all a gate licenses is scaling
up a prediction that already sits near the mean, and scaling cannot manufacture the
information that would place a compound at 1.9 or 7.9. The cost-halving arithmetic assumed
it could.

Two consequences, both of which outrank the retracted number.

The oracle ceiling of 0.4469 lands on the 0.4447 that the field's best absorption implies
(Section 1). That is a genuine confirmation: the absorption gap IS the tails. But closing
it requires a model that PREDICTS tail values. No post-hoc transform reaches it -- now
established for the gated family as well as the monotone family.

And at the gate quality actually available (AUC 0.80), the whole programme is worth
+0.0028. That is noise. The extremeness head is not the next move. Section 7 as originally
written was one measurement short of knowing that.

## 8. The slot

Nothing is validated end to end, so nothing is submitted, and after the retraction above
there is no candidate to build either. `submission_d6probe.csv` stays on the board at
0.4737.

Four levers were measured this session and all four came back negative:

| lever | result | status |
|---|---|---|
| free-form monotone re-placement | -0.0015 vs affine, P(worse) 70.8% | closed |
| 1/width interval weighting | premise false on 3 of 4 endpoints | closed as written |
| seed-spread gating | AUC 0.50-0.57 for the expensive decile | closed |
| extremeness gating | +0.0028 at a realistic gate; 0.4469 at a perfect one | closed |

What survives is the Section 1 finding itself: our rho is the best in the neighbourhood
and our conversion is the worst, and the entire difference is tail prediction. That is a
training problem on the tail compounds, with no transform in front of it. Any further
pIC50 work should go there, and should be measured as tail rho -- Spearman restricted to
the top and bottom label deciles -- not as macro rho, which the bulk dominates and which
we are already winning.
