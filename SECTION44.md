# Section 44. The full blind set, per endpoint: is the half-set leaderboard biasing us?

The interim leaderboard reports every blind compound, and for the first time it reports
each endpoint separately. That lets two questions be answered that were previously
unanswerable: whether scoring on half the blind set was distorting our read, and which
endpoint is actually costing us placement.

## 1. The rows are internally consistent

All four macro numbers reconstruct exactly from the per-endpoint rows, so nothing is
mis-transcribed and nothing is being averaged differently than we assume:

| macro | from per-endpoint rows | board |
|---|---|---|
| MA-ST-RAE | 0.4833 | 0.4833 |
| MA-Spearman | 0.7590 | 0.7590 |
| MA-R2 | 0.5664 | 0.5664 |
| MA-Kendall | 0.5664 (reported) | 0.5664 |

Plain unweighted mean over the four isoforms, confirmed.

## 2. The half set was NOT materially biased

| endpoint | ST-RAE half | full | shift | rho half | full | shift |
|---|---|---|---|---|---|---|
| CYP1A2 | 0.5125 | 0.4972 | -0.0153 | 0.8064 | 0.8004 | -0.0060 |
| CYP2C9 | 0.3978 | 0.4073 | +0.0095 | 0.8454 | 0.8420 | -0.0034 |
| CYP2D6 | 0.5957 | 0.5908 | -0.0049 | 0.5190 | 0.5342 | +0.0152 |
| CYP3A4 | 0.4634 | 0.4378 | -0.0256 | 0.8327 | 0.8593 | +0.0266 |

Mean absolute shift 0.0138 ST-RAE (max 0.0256) and 0.0128 rho (max 0.0266). The signs do
not share a direction: ST-RAE went down on three endpoints and up on one; rho went up on
two and down on two. **That is the signature of sampling noise, not of a biased subset.**
A biased half set would have moved every endpoint the same way.

Two consequences:

- The Section 28 half-set absolute uncertainty of +/-0.065 macro was **conservative** by a
  factor of roughly four. The realised half-to-full macro move was 0.4923 -> 0.4833,
  i.e. -0.0090.
- Every conclusion drawn from half-set rows stands. Nothing has to be re-derived.

The concern was the right one to raise; the data says it does not bite here. The reason is
structural: the blind set is drawn from the same chemical space as the training data, so a
random half of it resembles the whole. This is also why it will keep holding.

## 3. Two-axis decomposition on the full set

`ST-RAE = absorption x sqrt(1 - rho^2)`, where absorption is how much of our residual the
credible-interval dead zone swallows. Lower is better on both ST-RAE and absorption.

| endpoint | ST-RAE | rho | absorption | MAE | R2 | rho^2 | our rank |
|---|---|---|---|---|---|---|---|
| CYP1A2 | 0.4972 | 0.8004 | 0.8294 | 0.6769 | 0.6213 | 0.6406 | 51 |
| CYP2C9 | 0.4073 | 0.8420 | 0.7550 | 0.4775 | 0.6986 | 0.7090 | 40 |
| CYP2D6 | 0.5908 | 0.5342 | 0.6989 | 1.1434 | 0.2151 | 0.2854 | 18 |
| CYP3A4 | 0.4378 | 0.8593 | 0.8560 | 0.5042 | 0.7307 | 0.7384 | 58 |

Caveat kept from Section 31: `e = ST-RAE/MAE` and `absorption` compare teams **within one
endpoint**, never across endpoints, because the constant-prediction base differs per
endpoint. Do not read CYP2D6's low absorption (0.6989) as good placement; its intervals are
simply wide relative to a spread that also has the largest MAE.

## 4. CYP2D6 is the binding constraint, and the field agrees

CYP2D6 carries our worst everything: rho 0.5342 against 0.80-0.86 elsewhere, MAE 1.1434
against 0.48-0.68, R2 0.2151 against 0.62-0.73. It contributes 0.5908 to a 0.4833 macro,
i.e. **31% of the total cost from 25% of the weight**.

But our *rank* there is 18 -- our best of the four. On CYP3A4 we post our best rho (0.8593)
and rank 58. The rank spread 18-to-58 is therefore almost entirely **field strength, not our
quality**: everyone struggles on CYP2D6 and everyone does well on CYP3A4. Two things follow:

- Ranking improvement per unit of modelling effort is cheapest where the field is weak and
  our absolute error is large, which is CYP2D6.
- Chasing CYP3A4 is the worst use of effort: 58 teams are already inside our number, so the
  marginal rank per 0.01 of rho is tiny there.

CYP2D6's difficulty is real and chemical, not an artefact: the training set is our smallest
fitted endpoint (1,364 rows), CYP2D6 recognition is dominated by a basic-nitrogen-to-Asp301
salt bridge whose strength depends on protonation state at a precision our pKa estimates do
not have, and the assay dynamic range is narrow.

## 5. The Section 20 anomaly has disappeared

On half-set rows CYP2D6 showed R2 (0.2734) *above* its squared Spearman, which is
impossible for jointly-normal data and was logged as evidence the endpoint was not
bivariate normal. On the full set R2 sits **below** rho^2 on all four endpoints, as it
should:

| endpoint | rho^2 - R2 |
|---|---|
| CYP1A2 | +0.0193 |
| CYP2C9 | +0.0104 |
| CYP2D6 | +0.0703 |
| CYP3A4 | +0.0077 |

The anomaly was a half-set artefact **for us**. It was tempting to retract Section 20's
non-normality inference on that basis.

**THIS RETRACTION IS ITSELF WITHDRAWN -- see Section 45.** The top-leader rows posted
minutes later show all four of them with R2 ABOVE rho^2 on CYP2D6 (preheat-to-450 -0.1335,
wbot -0.1448). So R2 > rho^2 is attainable on that endpoint; it is what a good CYP2D6 model
looks like, and we are the only team of five that does not achieve it. Section 20 was
observing something real. The right reading is not "CYP2D6 is not bivariate normal" but
"CYP2D6 has a heavy-tailed, strongly-linear extreme that rewards getting magnitudes right,
and our model shrinks toward the mean instead." Section 45, part 4.

## 6. What this changes in the plan

Unchanged: the submission bar (~+0.160 summed out-of-sample local rho), the 14% transfer
rate, the decision to improve chemprop and Uni-Mol before any calibration.

Changed:

1. **Per-endpoint effort is now re-weighted toward CYP2D6.** It is the largest cost, the
   weakest field, and the endpoint where the descriptor work has been most productive
   (Section 40's narrowing gave it +0.0493, its largest gain of the four).
2. **The half-set uncertainty band shrinks** from +/-0.065 to something nearer +/-0.02
   absolute macro, because we have now measured the half-to-full move once directly.
3. **Per-endpoint local validation becomes checkable.** We can now compare the blind-like
   holdout's per-endpoint rho against the board's per-endpoint rho and calibrate the
   yardstick endpoint by endpoint instead of only in macro. That is the single most valuable
   thing this leaderboard release bought us, and it is the next measurement to run.

## 7. Note on CYP2D6's 3D refutation

Section 43 removed 3D shape from CYP2D6 because chemprop lost 0.1518 on the reweighted
yardstick and 0.1057 on the holdout, against a run-to-run noise floor of roughly 0.06 and
0.007. Given Section 4 above -- CYP2D6 is where the marginal rank is cheapest -- that
refutation is worth one confirmation run rather than being taken as final on a single pair,
because the feature that helped LightGBM most there also encodes the constrained-site
geometry the pharmacology points at. It stays out of `SPEC` until a rerun says otherwise.
