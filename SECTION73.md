# Section 73 -- Li, Abreu, Kraft and Green (CQR): read, tested, null for this metric

Paper: "Molecular Property Prediction with Well-Calibrated Uncertainties via Graph Neural
Networks and Conformal Quantile Regression", Shih-Cheng Li, Charlles R. A. Abreu, Markus
Kraft, William H. Green. ChemRxiv 10.26434/chemrxiv.15010324/v1, 12 October 2026.

## 1. What it actually does, which is better news than the title suggests

Section 72 said post-hoc uncertainty was dead here and that only an uncertainty-aware
TRAINING objective could matter. This paper is the training kind:

  * Chemprop (D-MPNN) is trained with the **pinball / quantile loss** (their eq 2),
    multitask across quantile levels, with two heads for `Q_LB` at `tau = alpha/2` and
    `Q_UB` at `1 - alpha/2`. Optionally a third head at `tau = 1/2` for the median.
  * The point prediction is the **midpoint** of the two bounds (Chemprop v1.7.1's default).
    Their SI S5 reports median and midpoint give similar accuracy.
  * Conformalization adds a single GLOBAL constant `Delta`, the `gamma(n_c+1)/n_c`
    empirical quantile of the calibration conformity scores, to both bounds.
  * They settle on `alpha = 0.3` globally and benchmark against ensembling, MC dropout,
    MVE and evidential regression on 16 datasets.

Two things make it worth taking seriously for us rather than dismissing. ST-RAE is itself
a two-sided pinball loss -- `clip(p - hi, 0) + clip(lo - p, 0)` is one-sided L1 above `hi`
plus one-sided L1 below `lo` -- so the paper's objective family is the same family as our
metric. And `cypkit/losses.py` already implements exactly that as `dead_zone_loss`, with
gradient and Hessian, unused by `train_cyp_mh.py`.

## 2. The cheap test first: is y off-centre in its own interval?

If the credible interval were skewed about the fitted label, the ST-RAE-optimal training
target would be the midpoint rather than `y`, and that is a one-line change. It is not:

| endpoint | relative position of y in [lo, hi] | mean skew (mid - y) |
|---|---|---|
| CYP1A2 | mean 0.5105, median 0.5081 | -0.0145 |
| CYP2C9 | mean 0.4996, median 0.4934 | -0.0147 |
| CYP2D6 | mean 0.5012, median 0.4911 | -0.0194 |
| CYP3A4 | mean 0.4971, median 0.4897 | -0.0348 |

Dead centre, and `y` is inside its own interval for 100% of rows. No free target shift.

## 3. The real test: four objectives, one rig, frozen folds

LightGBM on 1024-bit ECFP4, identical folds and hyperparameters, the only difference being
the loss. Both arms placed optimally before comparison, because Section 69 established
that comparing raw ST-RAE mostly compares two accidental placements.

| objective | macro rho | raw ST-RAE | **placed ST-RAE** |
|---|---|---|---|
| L2 (what we train on) | 0.4305 | 0.9190 | **0.8255** |
| L1 | 0.4309 | 0.8445 | **0.8251** (-0.0004) |
| dead-zone, delta 0.25 | 0.4291 | 0.8595 | **0.8249** (-0.0006) |
| dead-zone, delta 0.10 | 0.4247 | 0.8704 | **0.8305** (+0.0050) |

**Null.** Four objectives, macro rho identical to three decimals (0.4247-0.4309), placed
ST-RAE within 0.006.

Read the raw column, because it is the trap. L1 improves raw ST-RAE from 0.9190 to 0.8445
-- a 8.1% gain that would look like a result. All of it is placement, and we already
collect placement with a fitted affine. Once both arms are placed, it is worth -0.0004.

FIRST ATTEMPT WAS A BUG, RECORDED SO NOBODY REPEATS IT. The dead-zone arm initially scored
rho -0.05 on all four endpoints, which reads as a crushing refutation. It was wiring: a
LightGBM custom objective starts from raw score 0, so every prediction began far below
`lo`, the gradient was a constant -1, and the model never moved. It needs
`Dataset(..., init_score=mean(y_train))` and `predict() + init_score` on the way out. The
corrected arm trains normally. A refutation that strong on every endpoint at once is a
bug until proven otherwise.

## 4. Why the null is principled, not an accident of this rig

The loss shape determines WHERE the optimum sits -- L2 gives a conditional mean, L1 a
conditional median, dead-zone the ST-RAE-optimal point. It does not determine how well the
model can ORDER compounds, which comes from the representation and the data. Our placement
is already solved to its optimum (Section 69: free monotone map -0.0015 against affine,
P(worse) 70.8%), so a loss that improves placement improves something we already have.

That is also the whole account of channels (a) and (b) from Section 72:

| channel | mechanism | result |
|---|---|---|
| (a) post-hoc uncertainty -> point estimate | sigma-gated rescaling | oracle ceiling +4%, two endpoints WORSE with perfect sigma; real ceiling ~0 (Section 72) |
| (b) uncertainty-aware training objective | L1 / dead-zone / pinball | **null, within 0.006 of L2 (this section)** |

Both channels are now measured. There is no third one: the submission accepts one number
per compound per endpoint.

## 5. What the paper IS good for here

Not the score. But the leaderboard has a "Model Report Link" column and every competitor
shows "Not submitted". CQR would give calibrated, heteroskedastic prediction intervals on
the 750 blind compounds at a stated coverage, which is exactly the content a model report
wants and which we currently cannot produce at all. If a report is worth anything in the
final judging, this is the method to build it with, and it costs one quantile-loss training
run plus a conformal constant.

Keep the distinction straight: **CQR is the right tool for reporting what we do not know,
and the wrong tool for improving what we predict.**

## 6. One caveat on scope

This was measured on a LightGBM/ECFP rig at macro rho 0.4305, well below the production
multihead's 0.6453. A loss shape could in principle matter more for a stronger model. The
mechanism in Section 4 argues it is general, but the honest statement is that the null is
established where it was measured.
