# Section 72 -- Uncertainty has no channel into this metric, and an oracle sigma proves it

Prompted by a request to apply a ChemRxiv uncertainty-quantification paper
(10.26434/chemrxiv.15010324). THE PAPER COULD NOT BE RETRIEVED -- the PDF endpoint
returned a client error. Nothing below depends on its contents; it is all measured here.
The last part says what the paper would have to propose for any of this to change.

## 1. The structural argument comes first

The submission format accepts **one number per compound per endpoint**. There is no field
for a variance, an interval or a confidence. So an uncertainty estimate can only reach the
score through one of two channels:

  (a) it changes the POINT ESTIMATE we submit, or
  (b) it changes TRAINING, and therefore the point estimate indirectly.

Channel (a) is testable to its ceiling without building anything, because we can hand the
procedure a perfect sigma and see what it does. That is Section 2. Channel (b) is still
open and is the only thing worth discussing afterwards.

## 2. The ceiling on channel (a): a PERFECT sigma, nested out of fold

Per-compound scale `b` fitted per sigma-quintile on four folds, applied to the fifth, with
`sigma = |y - p|` -- the true error magnitude, which no method can ever estimate:

| endpoint | affine | oracle-sigma gated | gain | b by sigma quintile, low to high |
|---|---|---|---|---|
| CYP1A2 | 0.7427 | 0.6140 | **+0.1287** | 1.00 0.80 0.50 0.70 **2.50** |
| CYP2C9 | 0.5438 | 0.5362 | +0.0076 | 1.00 0.90 1.00 1.30 1.40 |
| CYP2D6 | 0.8163 | 0.8287 | **-0.0124** | 1.00 1.00 1.00 0.80 0.95 |
| CYP3A4 | 0.3967 | 0.4192 | **-0.0225** | 1.00 0.90 0.80 1.00 0.95 |
| MACRO | 0.6249 | 0.5995 | +0.0254 (+4.1%) | |

Read the failure, not the headline. **Two of four endpoints get WORSE with perfect
information**, which means the per-bin fitting is already at its own noise limit. And the
whole macro gain is one endpoint whose fitted pattern is `... 0.70 2.50` -- the most
uncertain quintile EXPANDED to the cap. That is not hedging, it is the oracle leaking:
`sigma = |y - p|` encodes y, so the top-sigma bin is the set of compounds whose truth is
far from our prediction, and expanding them moves them toward it. A real sigma cannot do
that. The honest ceiling on channel (a) is therefore well below +4%, and plausibly zero.

## 3. Three uncertainty proxies, three failures

Measured on `oof_mhens3`, out of fold, against both |error| and the metric's own cost:

| signal | rho vs cost | AUC for the top-decile cost | verdict |
|---|---|---|---|
| seed spread over 4 mhens3 seeds | 0.01 - 0.09 | 0.50 - 0.57 | chance (Section 69) |
| 1 - max Tanimoto to other folds | -0.11 - +0.03 | **0.41** - 0.55 | chance, and INVERTED on CYP1A2 |
| 1 - mean top-5 Tanimoto | -0.04 - +0.07 | 0.46 - 0.59 | chance |
| interval width (known at train time) | +0.16, +0.01, +0.05, **-0.50** | 0.65, 0.47, 0.60, 0.31 | predicts ERROR, anti-predicts COST |

The last row is the one to understand. Interval width correlates with |error| at rho
0.18-0.42 -- a wide credible interval means a poorly determined curve means a noisy label.
But its correlation with COST is near zero and on CYP3A4 is **-0.50**: a wide interval is a
wide dead zone, so the compounds hardest to predict are the cheapest to get wrong. Error
and cost are different quantities here, and only cost is scored. Any uncertainty method
validated against error -- which is what UQ papers validate against -- is being validated
against the wrong target for this competition.

Classic applicability-domain distance being INVERTED on CYP1A2 (AUC 0.413) is worth
keeping separately: on that endpoint, compounds further from the training set have SMALLER
errors. Do not reach for similarity-to-training as a confidence signal here.

## 4. What remains open: channel (b)

Nothing above touches training. One specific idea is untested and is not ruled out by any
of it: the challenge gives a **per-compound credible interval on every training label**,
which is a direct measurement of that label's noise. A heteroscedastic or dead-zone
likelihood that treats the interval as the label's noise model is a TRAINING change that
improves the point estimate, and it is categorically different from the post-hoc gating
that Section 2 just killed. `cypkit/losses.py` already implements the dead-zone objective
with gradient and Hessian; `train_cyp_mh.py` does not use it.

That is also the only form in which the requested paper could matter. If it proposes
post-hoc uncertainty for ranking, filtering or re-placement, Sections 2 and 3 answer it
and the answer is no. If it proposes an uncertainty-aware TRAINING objective, it is worth
reading closely, and the thing to extract is the loss, not the uncertainty estimate.

## 5. A standing methodological note

This section was written the way Section 69's retraction said to: **compute the oracle
ceiling before building the estimator.** Doing it cost four minutes and closed a direction
that would otherwise have cost a week. Apply it to the next proposal too.
