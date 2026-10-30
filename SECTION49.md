# Section 49. A private scorer, a feature cache, and the first number that matters

Run date 2026-10-11. Three scripts were built and run end to end for the first time:
`scripts/eval_service.py`, `scripts/build_features.py`, `scripts/fit_tabular.py`, in a new
conda env `cyp_tab` kept separate from `admet_gpu` so the working chemprop install was
never at risk.

The headline is not that TabM won. It did not. It is that **three unrelated model families
now agree to within 0.005 macro ST-RAE, and the one model that disagrees is the one trained
on different data.**

---

## 1. Why this was built

From Smbatyan et al., bioRxiv 2026.10.01.755314 (Deep Origin), which replayed ExpansionRx
and PXR -- our two challenges -- with LLM agents and reached 4th/104 and 1st/104. Two
things in that paper were worth copying:

1. The enabling infrastructure was a **private evaluation service that withheld test
   labels and enforced the submission protocol**. Not the agent.
2. Their winning solutions did **not** fine-tune graph networks. They put frozen
   representations into small tabular models -- TabM, TabPFN, kernels.

Point 1 addressed a standing problem here: local validation did not track the leaderboard,
and early stopping in `train_cyp_clean.py` was selecting checkpoints on the very fold being
reported as held out. Point 2 was a model class never tried on this challenge.

## 2. The operational result

| | wall clock, 4 endpoints |
|---|---|
| `train_cyp_clean.py` (chemprop, 10 folds) | ~58 min |
| `fit_tabular.py --model tabm` (5 folds) | **15.8 s** |

Feature cache: 54.7 s for 217 RDKit descriptors, 4.0 s for 44 ionisation features, 9.8 s
for 10,240 fingerprint counts -- once, for all 5,655 compounds, then free.

That ratio is the point. It converts "one configuration an evening" into "twenty", which
is the precondition for any search, agentic or manual.

## 3. The scientific result, on one identical holdout

1,000 compounds, scaffold-disjoint, activity-balanced, carved by `eval_service init`
(seed 20261010). Labels private; IDs public so training excludes them. 281 / 260 / 313 /
478 scored per endpoint. Every row below is the SAME 1,000 compounds.

| | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | MACRO |
|---|---|---|---|---|---|
| **mhens3** (production blend) | 0.7412 | 0.5391 | 0.8524 | 0.4225 | **0.6388** |
| tabm_core | 0.8368 | 0.6901 | 0.9265 | 0.5683 | 0.7554 |
| clean_v1 (chemprop) | 0.8911 | 0.6864 | 0.9798 | 0.4649 | 0.7556 |
| hgb_core | 0.8339 | 0.6841 | 0.9593 | 0.5617 | 0.7598 |

ST-RAE, lower is better.

**TabM lost to the production blend on all four endpoints, by +0.1166 macro.** That is the
direct answer to "does the paper's approach transfer", and on this evidence it is no.

## 4. But look at rows 2-4

A D-MPNN message-passing network, histogram gradient boosting on descriptors, and a TabM
weight-shared MLP are about as architecturally unrelated as three regressors can be. They
land at **0.7554, 0.7556, 0.7598** -- a spread of 0.0044, which is noise.

The one thing they share is the training data: **dose-response compounds only**. The one
model that escapes, `mhens3` at 0.6388, is the one trained on the censored rows, the
external qHTS data and all four endpoints jointly.

So the 0.12 macro that separates them is **not** architecture. Architecture has been varied
across three families and moved nothing. It is the ~2,900 censored compounds per endpoint
that the curve-only premise throws away, and whatever the multi-task and external-data
arms add on top.

This supersedes the reading in Section 48 that we were at an information limit. We are at
the limit *of the data we chose to keep*.

## 5. Calibration says something separate and odd

| | CYP1A2 slope | CYP2C9 slope |
|---|---|---|
| mhens3 | 1.439 | 1.450 |
| tabm_core | 0.991 | 1.043 |

Slope is the factor truth-on-prediction wants; 1.0 is optimal. **TabM is well calibrated
and loses. mhens3 is badly mis-scaled -- its predictions are roughly 1.45x too narrow on
these compounds -- and wins anyway.** Its ranking advantage (rho higher by 0.10-0.13 on
every endpoint) is large enough to absorb a scaling error that should cost it real
ST-RAE.

Two consequences. First, mhens3 has unclaimed headroom: widening its CYP1A2 and CYP2C9
columns by ~1.45 about their mean should gain at unchanged ranking. Second, calibration is
not what separates these models, so the spread work is not where the remaining 0.12 lives.

## 6. The validation is now trustworthy, which it was not before

`tabm_core`'s own 5-fold estimate against the private holdout it never saw:

| | OOF | holdout | diff |
|---|---|---|---|
| CYP1A2 | 0.8181 | 0.8368 | +0.0187 |
| CYP2C9 | 0.6685 | 0.6901 | +0.0216 |
| CYP2D6 | 0.9086 | 0.9265 | +0.0179 |
| CYP3A4 | 0.5939 | 0.5683 | -0.0256 |
| **macro** | **0.7473** | **0.7554** | **+0.0081** |

Eight thousandths on the macro, and every endpoint inside 0.026. The cross-validation now
predicts held-out performance. Given how much of this project has been spent on local
numbers that did not transfer, that is worth more than any single model result here.

What is still NOT established is the local-to-board mapping. The service's `calibrate`
needs three submissions scored here AND uploaded; we have none. Until then local ST-RAE is
an internally consistent yardstick, not a board prediction.

## 7. Two diagnostics worth acting on

**CYP3A4 overfits badly under TabM.** Pooled gap +51%, fold 3 at +81% with train R2 0.9802
against valid 0.5325. 1,857 rows, 245 features, a 512-wide 3-layer net at dropout 0.15 is
too much capacity. Try `--dropout 0.3 --width 256`.

**CYP2D6 is the worst endpoint under every approach ever tried here.** ST-RAE 0.9265 means
7% better than one constant. Slope 1.233 says the predictions are additionally too narrow.
Nothing in this section changes Section 48's reading of it.

## 8. What follows

The ranked next experiment is no longer architectural. In order:

1. **Train on the censored rows with a one-sided loss.** Sections 44-48 and this one now
   point at the same thing from three directions, and Section 49's three-family
   convergence is the strongest statement of it. Expected effect: the bulk of the 0.12.
2. **Multi-task across the four endpoints** was tried (Section 48 addendum, `mt1`) and lost
   16/16 metrics -- but it was tried with curve-only data. It deserves one re-test once
   the censored rows are in, because the negative transfer measured there may have been an
   artefact of four weak, data-starved tasks.
3. **Widen mhens3's CYP1A2 and CYP2C9 columns**, which costs nothing and is supported by
   the slope measurements in Section 5 above.

Recorded because the infrastructure took an evening and produced one clean negative result
plus one finding that reframes the whole remaining effort. The negative result -- TabM does
not beat the blend -- cost 16 seconds to obtain, which is the argument for having built the
infrastructure at all.

---

# Addendum, same day. The censored-row prediction was wrong, and the harness cannot settle it

Section 8 above predicted that training on the censored rows would recover "the bulk of
the 0.12". It was implemented (`fit_tabular.py --use_censored`, a one-sided hinge that
charges nothing for predicting a censored compound at or below pIC50 4.0 -- deliberately
the same shape as ST-RAE's dead zone) and tested on CYP2C9, 3 folds, everything else held
fixed. The censored set is derived from the raw screen files, not from labels.csv:
screened, not a primary-screen hit, no curve. It reproduces labels.csv's `left` set
exactly (2,700 / 2,134 / 2,883 / 707).

## The result

| cens_weight | scored on curve rows | scored on curve + censored | rho on curve rows |
|---|---|---|---|
| 0.00 (off) | **0.5280** | 0.6342 | 0.7059 |
| 0.05 | 0.5791 | 0.5398 | 0.7022 |
| 0.20 | 0.6635 | 0.4431 | 0.6906 |
| 1.00 | 1.4098 | **0.3919** | 0.6244 |

ST-RAE, lower better. **The ordering is exactly inverted between the two columns.** Best
without censored rows, best with them, depending only on which compounds you score.

Meanwhile Spearman on the curve compounds moves 0.706 -> 0.624. The censored rows add
essentially **no ranking information among actives**. What they do is move the predicted
distribution down toward the library's, which is wrong if you score actives and right if
you score the library.

## What this means

1. **Section 8's prediction is not supported.** Censored training does not buy ranking. It
   trades error on actives against error on inactives.
2. **The harness as built is blind to the question.** Its holdout contains only
   curve-fitted compounds, by the argument in Section 3 of this document -- which remains
   right for matching the blind set's *rho*, and is wrong for measuring anything about
   inactive compounds. The blind set is the unfiltered library; a model that has never
   seen an inactive will place inactives too high, and nothing in our holdout can see it.
3. **It is an implementation problem, not a dead end.** `mhens3` trains on these same
   censored rows and scores 0.5391 on the curve-only holdout -- better than any
   curve-only-trained model here. So a model CAN use them without wrecking its placement
   of actives. A raw hinge at uniform weight cannot; a proper censored likelihood,
   combined with the external-assay and multi-task arms, evidently can.

## What would actually resolve it

Nothing local, on the present evidence. Two compositions give opposite answers and we
cannot choose between them from training data alone -- the blind set's own active/inactive
proportion is unknown and three attempts in Sections 3 and 5 to infer it from board
feedback have not converged. The board is the only instrument, at one slot a day.

The cheap next move is therefore NOT another modelling run. It is to spend one slot
anchoring the local-to-board mapping: upload a file that has been scored through
`eval_service` with `--note` set, so `calibrate` gets its first anchor. Three anchors and
the harness starts predicting board movement, which is what every question above is
actually waiting on.

Recorded because a prediction made in the morning was tested and refuted by the evening,
which is the behaviour the harness was built to enable.
