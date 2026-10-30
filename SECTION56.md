# Section 56. Where the equal-weight blend came from, and why my own recommendation was unsafe

Run date 2026-10-12. Two questions: how the blend ended up weighting every member equally,
and whether to blend or to ship the best model. The answer to the first is in
`scripts/tdi_weights.py`, which I had not read before recommending per-endpoint weights in
Section 54. It changes the answer to the second, and it partly retracts Section 54.

---

## 1. Equal weights is a retreat, not a default

`scripts/tdi_weights.py` fits blend weights properly: simplex-constrained maximisation of
AUC over rank-averaged members, ridged toward equal weights because ten members on ~1,500
CYP2D6 rows overfits, and **nested** -- weights fitted on four folds, scored on the fifth,
rotating -- precisely so the reported gain is not the optimism of fitting and scoring on
the same rows. Its own docstring records that an earlier non-nested sweep produced CYP2D6
MCC 0.1423 to 0.1949 over 4,000 simplex draws and that this "was optimism, not signal".

It was run, the weights were submitted, and the two endpoints split:

| | nested OOF predicted | board delivered |
|---|---|---|
| CYP3A4 | +0.0057 | d' 1.290 -> 1.064, **MCC 0.4470 -> 0.3798** |
| CYP2D6 | +0.0069 | d' 0.861 -> 0.980, **MCC 0.1825 -> 0.2180** |

CYP3A4 lost 0.067. CYP2D6 gained 0.036. Nested validation predicted a small gain on both.

**The diagnosis in that file is distribution shift, with a number attached.** Our folds
hold compounds out at median max-Tanimoto 0.44; the blind set sits at **0.587**. Nested CV
removes fitting noise within the cross-validation distribution, and is blind to the fact
that the blind set is much CLOSER to training than any fold is. Weight fitting therefore
selects the members that survive distant held-out chemistry, which is not the same set as
the members that win on close analogues.

That predicts the split exactly. CYP2D6's fitted weights loaded on the similarity kernels
(knn_tan 0.32, svc_tan 0.30) -- methods that get better as the test set moves closer to
training -- and gained. CYP3A4's loaded on **`struct` at 0.29**, a rule-derived member, and
it lost a fifth of its discriminability.

So `run_tdi_ensemble.py` reverting to `weight = 1.0` for every member is the scar from
that submission, not an oversight. The rule of thumb written there is already right: *fit
weights per isoform, submit them per isoform, and keep a fitted set only for an endpoint
where it has beaten equal weights ON THE LEADERBOARD.*

## 2. This lands on my own Section 54 recommendation

Section 54 reported per-endpoint greedy selection gaining +0.0442 MCC on CYP3A4, and the
member it loaded most heavily was **`struct`, chosen 10 times across 5 folds**. That is
the same method, selecting the same member, on the same endpoint, that already cost 0.067
on the board. The method is nested; so was the fit that failed. Nesting was never the
protection that was missing.

Measured directly, on tdiB, same folds, same blind-rate cuts:

| CYP3A4 | AUC | MCC |
|---|---|---|
| per-endpoint greedy, all 32 members | 0.8420 | **0.4379** |
| per-endpoint greedy, `struct` excluded | 0.8308 | **0.4166** |
| **`lgb_clf` alone** | 0.8200 | **0.4166** |
| equal-weight, all 32 | 0.8154 | 0.3841 |

**Without `struct`, the per-endpoint blend equals the single best model to four decimals.**
Every point of CYP3A4's blending advantage is `struct`, and `struct`-weighted CYP3A4 has
one board result, which is a 0.067 loss. Section 54's CYP3A4 number should be read as
0.4166, not 0.4379, and at 0.4166 blending buys nothing at all.

CYP2D6 is the opposite case and survives the same test:

| CYP2D6 | AUC | MCC |
|---|---|---|
| per-endpoint greedy (knn_fcfp, svc_avalon1k, svc_c10, knn_tt, svc_tan) | 0.6435 | **0.1881** |
| the same with `struct` excluded -- identical, it is never picked | 0.6435 | 0.1881 |
| `knn_fcfp` alone | 0.6283 | 0.1501 |
| equal-weight, all 32 | 0.6274 | 0.1481 |

**+0.0380 over the best single**, from genuine multi-member blending, and every member it
picks is from the similarity family -- the family that gained +0.036 on the board the one
time it was weighted up.

## 3. The answer: best model on one endpoint, blend on the other

Not a uniform policy. The evidence points opposite ways and has a mechanism for why.

**CYP3A4 -- ship the single best model.** `lgb_clf` predicts 0.4576 at the optimal cut
against 0.4470 actually scored. Thirty-one other members do not beat it; the equal-weight
blend of all of them scores 0.3841 against its 0.4166 at a matched rate; and the only
thing that ever beat it in a fitted blend is the member that collapsed it on the board.
This endpoint has a solved model and a threshold problem, and blending is not the lever.

**CYP2D6 -- blend, within the similarity family only.** No single model is close: the best
is 0.6283 AUC and a selected blend of five reaches 0.6435, worth +0.038 MCC at the blind
rate. The members are kNN and SVC on different fingerprint enumerations, which is the one
construction with a board-confirmed gain on this endpoint. Restricting the pool to that
family also removes the mechanism that burned CYP3A4, because similarity kernels are the
members that improve as the test set moves closer, and the blind set is closer.

That is implementable today with the flag that already exists:

```
python scripts\run_tdi_ensemble.py --tag tdi3a4 --learners lgb_clf --predict
python scripts\run_tdi_ensemble.py --tag tdi2d6 --learners knn_fcfp svc_avalon1k svc_c10 knn_tt svc_tan svc_avalon --predict
```

and splicing the CYP3A4 column from the first file and the CYP2D6 column from the second
(`scripts/tdi_splice.py` exists for exactly this).

## 4. What it does not fix

Macro MCC still comes to roughly (0.4166 + 0.1881) / 2 = **0.3024** locally at blind rates,
against a standing best of **0.3325** actually scored. The honest reading of Sections 54
and 55 is now: per-endpoint structure bought CYP2D6 +0.038 and CYP3A4 nothing, and the
+0.0442 I reported on CYP3A4 was an artefact of a member we have already been burned by.

The gate is unchanged and still not met. The one thing that would move it is a second
scored TDI submission -- which is also the only way to settle whether the validator's 2.25
CYP2D6 shrinkage is real, and the only way to satisfy `tdi_weights.py`'s own rule that a
fitted set is kept only once it has beaten equal weights on the leaderboard.

Recorded because the file that explained all of this was in `scripts/` the whole time, and
reading it two days earlier would have saved a recommendation.
