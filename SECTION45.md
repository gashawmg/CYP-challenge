# Section 45. The top leaders, per endpoint. Placement is NOT closed, and here is the mechanism

Five leaderboard tabs now give the top four teams' per-endpoint numbers alongside ours.
This overturns a conclusion this project has been operating under since Section 31.

## 1. The rows

| endpoint | team | ST-RAE | MAE | e = ST-RAE/MAE | rho | R2 |
|---|---|---|---|---|---|---|
| CYP1A2 | wbot | 0.3752 | 0.6192 | 0.6059 | 0.8507 | 0.6886 |
| | cypadeedoodah | 0.3795 | 0.6565 | 0.5781 | 0.8444 | 0.6658 |
| | preheat-to-450 | 0.3856 | 0.6474 | 0.5956 | 0.8464 | 0.6700 |
| | EleanorRigby | 0.3911 | 0.6630 | 0.5899 | 0.8306 | 0.6473 |
| | **US (51)** | **0.4972** | **0.6769** | **0.7345** | **0.8004** | **0.6213** |
| CYP2C9 | preheat-to-450 | 0.3130 | 0.4155 | 0.7533 | 0.8862 | 0.7576 |
| | cypadeedoodah | 0.3308 | 0.4314 | 0.7668 | 0.8703 | 0.7276 |
| | Rubyy | 0.3459 | 0.4431 | 0.7806 | 0.8735 | 0.7334 |
| | wbot | 0.3574 | 0.4638 | 0.7706 | 0.8583 | 0.6988 |
| | **US (40)** | **0.4073** | **0.4775** | **0.8530** | **0.8420** | **0.6986** |
| CYP2D6 | preheat-to-450 | 0.4568 | 0.9113 | 0.5013 | 0.5982 | 0.4913 |
| | Rubyy | 0.4838 | 0.9845 | 0.4914 | 0.5561 | 0.3744 |
| | wbot | 0.4941 | 0.9520 | 0.5190 | 0.5526 | 0.4502 |
| | cypadeedoodah | 0.5052 | 1.0451 | 0.4834 | 0.5818 | 0.3665 |
| | **US (18)** | **0.5908** | **1.1434** | **0.5167** | **0.5342** | **0.2151** |
| CYP3A4 | preheat-to-450 | 0.3331 | 0.4276 | 0.7790 | 0.8882 | 0.7890 |
| | wbot | 0.3382 | 0.4430 | 0.7634 | 0.8885 | 0.7716 |
| | first-chair | 0.3412 | 0.4468 | 0.7637 | 0.8842 | 0.7698 |
| | cypadeedoodah | 0.3530 | 0.4712 | 0.7492 | 0.8825 | 0.7536 |
| | **US (58)** | **0.4378** | **0.5042** | **0.8683** | **0.8593** | **0.7307** |

`preheat-to-450` is first on three endpoints and third on the fourth. That is one strong
model, not four lucky draws.

## 2. The headline: our MAE is close, our ST-RAE is not

On CYP1A2 our MAE (0.6769) is **2.1% worse** than EleanorRigby's (0.6630) and our ST-RAE is
**27.1% worse**. Compare that to how ST-RAE responds to MAE *among the leaders themselves*:

| endpoint | leader-vs-leader amplification | us vs nearest leader |
|---|---|---|
| CYP1A2 | 0.50x | **12.9x** (MAE +2.1%, ST-RAE +27.1%) |
| CYP2C9 | 1.29x | 4.7x (MAE +3.0%, ST-RAE +14.0%) |
| CYP2D6 | 0.60x | 1.8x (MAE +9.4%, ST-RAE +16.9%) |
| CYP3A4 | 0.57x | 3.4x (MAE +7.0%, ST-RAE +24.0%) |

Among the leaders, ST-RAE moves *sub-linearly* with MAE (mean amplification 0.74x). Our
excess ST-RAE is therefore **not a mechanical consequence of slightly worse MAE**. It is a
separate deficit.

Counterfactual, holding our own MAE fixed and giving us the leaders' mean dead-zone
efficiency:

| endpoint | our e | leaders' e | ST-RAE now | ST-RAE at their e |
|---|---|---|---|---|
| CYP1A2 | 0.7345 | 0.5924 | 0.4972 | 0.4010 (-0.0962) |
| CYP2C9 | 0.8530 | 0.7678 | 0.4073 | 0.3666 (-0.0407) |
| CYP2D6 | 0.5167 | 0.4988 | 0.5908 | 0.5703 (-0.0205) |
| CYP3A4 | 0.8683 | 0.7638 | 0.4378 | 0.3851 (-0.0527) |
| **macro** | | | **0.4833** | **0.4308 (-0.0525)** |

**Placement alone is worth -0.0525 macro, which is 47% of our entire 0.1112 gap to rank 1,
with no change to the models.** For scale: every descriptor and architecture result in
Sections 36-43 combined is worth a fraction of that on the board after the 14% transfer
rate.

Section 31's "placement is closed, confirmed three ways" is **retracted**. It was derived
from macro rows where the tiny-gap guard discarded almost every comparator; per-endpoint
rows against known-strong teams say the opposite.

## 3. The mechanism, and why this is a modelling defect rather than metric gaming

> **RETRACTED -- see Section 46.** The censoring mechanism proposed in this part is wrong.
> The blind set is not left-censored the way our training data is: solved anchors put blind
> CYP2D6 at mean 2.92, sd 1.65, i.e. real fitted values well below our 4.0 floor, so there
> is no free (-inf, 4.0] zone to exploit. The local diagnostic built to test this failed its
> own convention check (local e 0.93-2.28 against a board 0.52-0.87) and that failure is
> what exposed the error. The CYP2D6 correspondence below has a different and simpler
> cause: CYP2D6 is the one endpoint whose live column is a PLACED column. Parts 1, 2, 4 and
> 5 of this section are unaffected -- they compare teams on the same compounds and never
> used the local pool. The text below is kept as written so the reasoning can be audited.


The credible intervals on *measured* values are narrow -- median half-width 0.145 (CYP1A2),
0.216 (CYP2C9), 0.131 (CYP2D6), 0.190 (CYP3A4) log units. Against an MAE of 0.5-1.1 those
dead zones absorb almost nothing, and they cannot possibly produce a 24% spread in `e`
between teams.

The dead zone that matters is the **left-censored** one. Our own training labels are
overwhelmingly censored at pIC50 = 4.0:

| endpoint | interval rows | left-censored rows | censored fraction |
|---|---|---|---|
| CYP1A2 | 1,180 | 2,932 | 71% |
| CYP2C9 | 1,026 | 2,393 | 70% |
| CYP2D6 | 1,364 | 3,012 | 69% |
| CYP3A4 | 2,335 | 707 | 23% |

For a censored compound the interval is (-inf, 4.0]: **any prediction at or below 4.0 costs
exactly zero ST-RAE, however far below it sits**, while still contributing to MAE. That is
the one place in this metric where a large amount of cost can be removed without improving
the ranking at all.

Now the fraction of our submitted (`submission_splice4.csv`) predictions that fall below
4.0, against the fraction of the training data that is censored:

| endpoint | training censored | our preds below 4.0 | our preds within +/-0.3 of 4.0 | e deficit |
|---|---|---|---|---|
| CYP1A2 | 71% | 43.2% | 15.6% | **+0.142** |
| CYP2C9 | 70% | 18.1% | 11.6% | +0.085 |
| CYP2D6 | 69% | 66.8% | 27.6% | **+0.018** |
| CYP3A4 | 23% | 24.1% | 17.3% | +0.105 |

**CYP2D6 is the endpoint where our predicted inactive fraction matches the prior, and it is
the only endpoint with essentially no placement deficit** (our e 0.5167 sits inside the
leaders' range [0.4834, 0.5190]). CYP1A2 and CYP2C9 place far too few compounds below the
censoring line and carry the two largest deficits per unit of available lever.

This reframes the whole issue. We are not failing to *place* predictions cleverly; **our
regressors do not predict inactivity.** They are fitted on the interval subset -- 1,026 to
2,335 rows -- and hedge the censored majority up toward the active range. A compound the
assay could not measure at all is being predicted at 4.3 instead of 3.2. That is a wrong
prediction by any standard, not a metric artefact, and fixing it is squarely inside the
brief of "develop accurate predicting models".

CYP3A4 is the exception that needs its own explanation: only 23% of it is censored, our
predicted fraction matches, and we still carry a 0.105 e deficit. That one is not the
censoring lever and remains unexplained.

## 4. CYP2D6: we are the only team whose R2 is below its squared Spearman

| team | rho^2 | R2 | rho^2 - R2 |
|---|---|---|---|
| preheat-to-450 | 0.3578 | 0.4913 | **-0.1335** |
| wbot | 0.3054 | 0.4502 | **-0.1448** |
| Rubyy | 0.3092 | 0.3744 | -0.0652 |
| cypadeedoodah | 0.3385 | 0.3665 | -0.0280 |
| **US** | 0.2854 | 0.2151 | **+0.0703** |

All four leaders beat their squared Spearman on R2; we are the only team that does not. R2
above rho^2 means a set of extreme compounds is being predicted numerically well -- their
magnitudes, not just their order. Our CYP2D6 predictions have roughly Rubyy's ranking
(0.5342 vs 0.5561) and **57% of Rubyy's R2**. We are shrinking toward the mean where they
are not.

This **reinstates Section 20's substance and cancels the retraction written in Section 44.**
R2 > rho^2 on CYP2D6 is not impossible and not a half-set artefact -- it is what four other
teams achieve. The correct statement is: CYP2D6 has a heavy-tailed, strongly-linear extreme
that everyone else exploits and our model does not.

## 5. The two gaps, sized

Gap to rank 1 (preheat-to-450, macro 0.3721) is 0.1112:

- **placement / censored-inactive prediction: -0.0525 (47%)**
- **model error: -0.0587 (53%)**, i.e. macro MAE 0.7005 -> 0.6005 and macro rho 0.7590 ->
  0.8048. Per-endpoint rho shortfall to the best on each: CYP1A2 -0.0503, CYP2C9 -0.0442,
  CYP2D6 -0.0640, CYP3A4 -0.0292, summing to **0.1877** -- close to the 0.160 submission
  bar, so the bar was set at about the right scale.

## 6. What to do, in order

1. **Diagnose the censored arm locally, on the blind-like holdout.** Compute our `e` on
   `artifacts/holdout_blindlike.csv` with the real intervals, per endpoint. If local `e`
   reproduces 0.73 / 0.85 / 0.52 / 0.87, the holdout becomes a direct instrument for this
   deficit and everything below is measurable before submitting. This is the single most
   valuable next run and it needs no training.
2. **Fix how the censored majority enters the fit**, not the output. Candidates, in
   increasing order of intervention: raise the weight of censored rows in the dead-zone
   loss; train the CYP1A2 and CYP2C9 regressors on the full label matrix rather than the
   interval subset; add the existing TDI/active classifier as a gating prior so a compound
   confidently classed inactive is pulled below 4.0. All three are model changes, testable
   on the holdout, and none is a post-hoc transform of a finished prediction vector.
3. **CYP2D6 variance restoration.** Test whether the shrinkage in Section 4 above is in the
   model or in the blend averaging. If blending four members is what flattens the extremes,
   a single strong member may beat the blend on this endpoint.
4. **Do not spend effort on CYP3A4 ranking.** 58 teams are inside our number there and our
   rho is already 0.8593 against a best of 0.8885; the marginal rank per unit of rho is the
   worst of the four. Its e deficit is worth chasing, its rho is not.

## 7. Submission status

Still no candidate. Nothing here has been measured locally yet -- item 1 above must run
first, and the standing rule holds: we do not submit until internal validation says the
candidate beats 0.4833.
