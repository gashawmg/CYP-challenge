# Section 53. The TDI validator on tdiM, and a shrinkage factor that still has no mechanism

Run date 2026-10-11, late. `scripts/run_tdi_ensemble.py --tag tdiM --save-oof` reproduced
the pool, and `scripts/tdi_validator.py --tag tdiM` scored every member. The verdict is
**do not submit**, and the reason is worth more than the table.

---

## 1. The verdict, and why it is a no

| | predicted MA-MCC |
|---|---|
| at the optimal cut, shrinkage applied | 0.3484  [0.2990, 0.4065] |
| the same, with NO shrinkage correction | **0.2772** |
| standing best, actually scored | **0.3325** |

The candidate clears the standing best by +0.0159 in the corrected column and loses to it
by 0.055 in the uncorrected one. The correction is `blind d' / OOF d'`, which is 1.00 for
CYP3A4 and **2.25 for CYP2D6**, and that 2.25 is fitted on a single scored submission.

The standing rule is that we do not submit worse than we had. A candidate that wins only
in the column fitted on one data point does not clear that rule, and the slot is spoken
for by `probe2` regardless. The validator's own closing note says the same thing: prefer
a candidate that clears both columns.

## 2. CYP3A4: the 21-member blend is not earning its keep

| candidate | OOF AUC | d' | predicted MCC |
|---|---|---|---|
| **lgb_clf** | **0.8200** [0.804, 0.837] | 1.295 | **0.4576** |
| RANK-BLEND | 0.8186 [0.804, 0.834] | 1.287 | 0.4551 |
| lgb_screen | 0.8121 | 1.253 | 0.4438 |
| pooled | 0.8117 | 1.251 | 0.4431 |
| struct | 0.8071 | 1.226 | 0.4351 |

A single LightGBM beats the rank blend of twenty-one members. The margin, +0.0025, is
inside the interval and inside the noise -- but that is the point: twenty extra models buy
nothing measurable here. The CYP3A4 header in `run_tdi_ensemble.py` already said this
endpoint is a solved model with a mis-placed cut (wbot matched our d' to three decimals,
cut at 0.314 against our 0.229, and scored 0.4554 against our 0.4470). The validator agrees:
0.4576 predicted at the OPTIMAL cut against 0.4470 actually scored is about +0.011, and
almost all of it is threshold, not model.

## 3. CYP2D6: similarity beats boosting, and the rule score is near last

| family | best member | AUC |
|---|---|---|
| fingerprint similarity | knn_fcfp | **0.6284** |
| | svc_c10 0.6203, svc_tan 0.6201, knn_k40 0.6167, knn_tan 0.6166 | |
| rank blend | RANK-BLEND | 0.6191 |
| boosted trees | lgb_clf | 0.5927 |
| | cat_clf 0.5882, et_clf 0.5789, xgb_clf 0.5656 | |
| the recovered rule | **struct** | **0.5361** |

Two readings. First, CYP2D6 time-dependent inhibition behaves like local structural
analogy, not like a global descriptor trend: every neighbour and kernel method beats every
boosted tree, by about 0.04 AUC. Second, `struct` -- the Section 52 rule applied softly to
two regressed arms -- sits nineteenth of twenty-one at 0.5361. That is an independent
confirmation, from this project's own tuned pipeline rather than my prototype, of Section
52's conclusion that the label rule does not help CYP2D6.

A caution on knn_fcfp's 0.2392: it is the maximum of twenty-one correlated estimates whose
intervals overlap almost completely (the whole similarity family sits in 0.616-0.628 with
intervals roughly +-0.034). Selecting the top of that list is selecting on noise, and the
true expected value of "whichever member happened to win" is below the number printed.

## 4. A mechanism for the 2.25, proposed and refuted the same evening

The field inversion recovers a blind prevalence of 0.0706 for CYP2D6 against 0.2164 in
training, and 0.2915 for CYP3A4 against 0.2132. Using the exact rule from Section 52, the
shift cut that reproduces each blind rate on the training rows is:

| | label's own cut | cut matching the blind rate | fold-shift |
|---|---|---|---|
| CYP2D6 | 0.301 | **0.676** | 4.74x rather than 2.00x |
| CYP3A4 | 0.301 | **0.163** | 1.45x rather than 2.00x |

That suggested a mechanism: the blind CYP2D6 positives would be the strong-shift third of
ours, which should be easier to separate, giving a d' ratio near the 2.25 the validator
assumes -- while CYP3A4, whose cut moves the other way, would show roughly 1.00.

Tested on a Tanimoto-kernel SVC and a Tanimoto kNN -- the two families that actually win
on CYP2D6 -- over identical scaffold folds, relabelling and changing nothing else
(`scripts/tdi_strict_label.py`):

| | d' as labelled | d' blind-matched | ratio | validator assumes |
|---|---|---|---|---|
| CYP2D6 svc_tan | 0.4158 | 0.3038 | **0.731** [0.229, 1.306] | 2.25 |
| CYP2D6 knn_tan | 0.3296 | 0.1570 | 0.476 [-0.238, 1.154] | 2.25 |
| CYP3A4 svc_tan | 0.7961 | 1.0285 | **1.292** [1.195, 1.406] | 1.00 |
| CYP3A4 knn_tan | 0.7815 | 0.9076 | 1.161 [1.106, 1.303] | 1.00 |

Wrong on both endpoints, and in opposite directions. Strong-shift CYP2D6 compounds are
HARDER to pick out, not easier.

And on reflection the premise was wrong as well. The organisers apply the same rule to the
blind set; its prevalence differs because its COMPOUNDS differ, not because its cut does.
Relabelling our training rows at a different shift emulates a blind set that does not
exist. The measurement still rules out one explanation -- a prevalence difference alone
does not produce the 2.25 -- but the remedy it suggested was never valid.

So the CYP2D6 shrinkage remains a one-point fit with no mechanism behind it, which is
exactly the kind of number this project has twice been burned by (Section 50's two-point
solve, Section 48's Gaussian inversion). Treat the uncorrected column as the one that
decides.

## 5. What this leaves

1. **CYP3A4 TDI**: the model is done. The remaining ~0.011 is the cut, and
   `run_tdi_ensemble.py` already matches a rate rather than transplanting a threshold.
   Dropping to `lgb_clf` alone would simplify the pipeline at no measured cost, but it is
   not a scoring gain.
2. **CYP2D6 TDI**: the gap is discriminability (blind AUC 0.756 against the leader's
   0.838) and the only family with traction is fingerprint similarity. More boosted trees
   will not close it; neither will the label rule.
3. **A second scored TDI submission is what the shrinkage needs.** Until there are two
   points, the corrected column is not evidence. That is an argument for eventually
   spending a slot on TDI, but not for spending tomorrow's on a candidate that loses
   without the correction.

The pIC50 track keeps priority: `probe2` tomorrow, then the CYP3A4 surrogate pair from
Section 52 if it clears its gate.

Recorded because a hypothesis formed at midnight was tested by one o'clock and failed, and
because the thing it was meant to explain is still unexplained.
