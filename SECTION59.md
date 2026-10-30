# Section 59. The CYP2D6 prediction was right, the CYP3A4 one was wrong, and I broke my own rule

Run date 2026-10-12. `submission_tdi_structfree.csv` scored. It is **not worse** on the
metric -- MA-MCC 0.3393 against a previous best of 0.3325, the highest this project has
recorded. The rank fell from 15 to 32 because the field moved over five weeks, not because
the score did. But one half of it went badly wrong, and the failure was predictable from a
condition I wrote down in Section 58 and then violated.

---

## 1. The split

| | predicted | scored | previous best |
|---|---|---|---|
| CYP2D6 | 0.258 - 0.268 | **0.2727** (rank 35) | 0.2180 |
| CYP3A4 | ~0.450 | **0.4059** (rank 49) | 0.4470 |
| macro | ~0.355 | **0.3393** (rank 32) | 0.3325 |

CYP2D6 beat its prediction. CYP3A4 missed by 0.044 and gave up 0.041 against a column we
had already scored.

## 2. The ratio method worked exactly where it was licensed, and only there

Section 58 derived the prediction from OOF d' ratios and noted the condition: *"A ratio
argument is only safe when the two columns are from the same family."*

**CYP2D6 -- same family, and the ratio was near exact:**

| | OOF d' | blind d' |
|---|---|---|
| previously scored, AUC-fitted blend (knn_tan 0.32 + svc_tan 0.30) | 0.4356 | 1.019 |
| new, six similarity members + pooled | 0.5352 | **1.274** |
| ratio | **1.229** | **1.250** |

Predicted a 1.229x lift in discriminability, got 1.250x. The shrinkage factor cancelled
exactly as argued, and the candidate beat its own forecast.

**CYP3A4 -- family changed, and the ratio was meaningless:**

| column | OOF d' | blind d' | OOF -> blind |
|---|---|---|---|
| equal-weight blend of ~20 members | 1.287 | **1.281** | x0.995 |
| `lgb_clf` alone | 1.295 | **1.163** | **x0.898** |
| fitted-weight blend, `struct` at 0.29 | -- | 1.088 | -- |

Out of fold, `lgb_clf` alone (1.295) edges the broad blend (1.287). On the blind set the
broad blend holds almost all of its discriminability and the single model loses a tenth of
its. I moved from a 20-member blend to one model and called it "same family" without
checking. Breadth IS the family on this endpoint.

## 3. Three CYP3A4 results now say one thing

| CYP3A4 column | blind MCC |
|---|---|
| **equal-weight blend, ~20 members** | **0.4470** |
| `lgb_clf` alone | 0.4059 |
| fitted-weight blend concentrated on `struct` | 0.3798 |

Every concentration loses, whether the concentrating is done by fitting weights or by
picking the single best member, and the loss scales with how concentrated it is. Sections
56 and 57 read the out-of-fold tables and concluded "ship the best single model"; the board
has now said no to that in the same way it said no to fitted weights. **CYP3A4 wants
breadth, and out-of-fold cannot see it** -- the folds hold compounds out at median
max-Tanimoto 0.44, the blind set sits at 0.587, and a broad average is what survives that
gap.

So Section 56's central recommendation is withdrawn. What replaces it is narrower and
board-backed: on CYP3A4, do not change the column at all without a leaderboard reason.

CYP2D6 is the mirror image and also board-backed: concentrating on the similarity family
has now gained twice, 0.1825 -> 0.2180 -> 0.2727.

## 4. The next submission is already measured on both halves

`artifacts/submission_tdi_bestmix.csv`, built and verified:

- **CYP3A4** labels taken verbatim from `submission_tdi_best.csv` -- the exact column that
  scored **0.4470**. Identical on 750/750, rate 0.2107.
- **CYP2D6** labels taken verbatim from `submission_tdi_structfree.csv` -- the exact column
  that scored **0.2727**. Identical on 750/750, rate 0.1133.

Predicted MA-MCC = (0.4470 + 0.2727) / 2 = **0.3599**, and neither half is an estimate.
No re-thresholding: moving the cut on a scored column is a change to a scored column, and
the +0.009 it might buy is not worth re-opening the one thing that has been reliable.

This is exactly the situation `tdi_splice.py` was written for, and its own record on that
situation is a prediction of 0.3325 against an outcome of 0.3325.

## 5. What I got wrong, stated plainly

I wrote the safety condition for the ratio argument in Section 58 and then applied the
argument across a family boundary on CYP3A4 in the same breath. The CYP2D6 half of that
section was good work and is now confirmed to three decimals. The CYP3A4 half cost 0.041
of MCC, and it cost it in the one direction this project has already been burned in twice.

The standing rule that survives all of this: **on the TDI track, a column changes only
when the leaderboard, not the out-of-fold table, says the new one is better.** That is
`tdi_weights.py`'s rule of thumb, written months ago, and this is the third time ignoring
it has cost a submission.

Recorded because the same evening produced the best TDI score this project has scored and
a 0.041 self-inflicted loss inside it.
