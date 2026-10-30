# CYP Challenge -- current state, 15 October 2026

Forty-three SECTION files is too many to re-read. This is the one page that says where we
are. It indexes them; it does not replace them.

## 1. The competition

Two tracks, ONE shared submission slot per day.

| track | metric | our live board row | our best ever | field best |
|---|---|---|---|---|
| pIC50 regression | MA-ST-RAE, lower better | **0.4751, rank 47** (blend4tdi) | **0.4737** (d6probe) | 0.3706 |
| TDI classification | MA-MCC, higher better | **0.3760, rank 24** (blendcp) | **0.3760** -- we are at it | ~0.46 |

TDI is sitting at our best. pIC50 is 0.0014 below ours after the blend4tdi loss. The board
shows the LATEST submission, not the best, so a bad submission costs real position until
the next slot.

## 2. What the metric is, which took a long time to understand properly

ST-RAE charges `clip(p - hi, 0) + clip(lo - p, 0)`: the distance OUTSIDE each compound's
credible interval, zero inside it. Consequences that drive everything else:

* It is a two-sided pinball loss, not a squared error.
* A compound with a NARROW interval is expensive; a wide one is nearly free. **Spearman
  counts every compound equally and the metric does not.**
* The cost is brutally concentrated: the top 10% of compounds carry 36-45% of the bill,
  the top 25% carry 66-78%, and 21-42% of rows cost exactly zero.
* Placement -- the per-endpoint mean and spread of the submitted column -- matters as much
  as ranking, and is a separate problem from the model.

## 3. The diagnosis, from the leaderboard itself

Our ranking is competitive. Our conversion of it is the worst in the visible field.

| | rho | MAE | ST-RAE |
|---|---|---|---|
| furtivepeony (rank 11) | 0.7554 | 0.6323 | 0.4348 |
| hfger (rank 8) | 0.7597 | 0.6268 | 0.4242 |
| hf-far (rank 17) | 0.7525 | **0.6638** | 0.4419 |
| **us** | **0.7558** | **0.6664** | **0.4737** |

furtivepeony ranks WORSE than us and scores 0.0389 better: 101% of that gap is conversion.
hf-far has our ranking AND our average error to within 0.4%, and scores 6.7% better.
Against rank 1, conversion is 80% of the gap. Our absorption (ST-RAE / sqrt(1-rho^2)) is
0.7234 against a field of 0.5975-0.6972.

**Neither rho nor MAE can see what separates us from them.** It is WHERE the error falls
relative to each compound's interval.

## 4. Closed, with evidence. Do not reopen without new information.

| direction | result | section |
|---|---|---|
| free-form monotone re-placement | -0.0015 vs a fitted affine, P(worse) 70.8% | 69 |
| post-hoc uncertainty gating | oracle ceiling ~0; two endpoints WORSE with a perfect sigma | 72 |
| uncertainty-aware training loss (L1, dead-zone, CQR/pinball) | null, four objectives within 0.006 | 73 |
| seed-spread as a confidence signal | AUC 0.50-0.57 for the expensive decile | 69 |
| applicability domain / Tanimoto distance | AUC 0.41-0.59, INVERTED on CYP1A2 | 72 |
| extremeness gating | +0.0028 at a realistic gate; 0.4469 at a PERFECT one | 72 |
| 1/width training weights, as originally specified | premise false on 3 of 4 endpoints | 69 |
| re-choosing the blend composition | +0.0086 apparent, but 26 compositions on the rows that pick the winner | 74 |
| proprietary data explaining the top | no top team is flagged as using it | 77 |

## 5. Established and real

* **Seed ensembling**: +5.3% at member level (0.6600 -> 0.6249 placed macro). Section 75.
* **CYP3A4 TDI head**: +0.0070 rho, replicated THREE times across two model families, every
  interval excluding zero. Improves the other three endpoints through the shared trunk even
  though only CYP3A4 has the head. Sections 70, 71, 74.
* **Cost-weighted training**: +0.71% single-seed, ~+0.58% at three seeds, 8/8 endpoint-power
  cells positive, sign test p = 0.004. Section 78.
* **The blend keeps 12% of a member-level gain.** This reprices every member result above.
  Section 75.
* **60-70% of the members' errors are SHARED** (mean pairwise error correlation 0.571-0.698).
  No amount of averaging touches that. Section 75.
* **Decorrelation is the currency of a blend**, not member quality. Section 76 is the
  expensive proof.

## 6. The open question

Every mechanism we have proposed for the conversion gap is closed (section 4), and the gap
is still there. The rank-1 team's own write-up says the ST-RAE optimum sits above the
population centre, that "the direction is argued, but the distance is not determined", and
lists an interval-width model as an OPEN PROBLEM. **Nobody on that board has solved this.**
We are not missing a known trick.

## 7. Immediate queue

1. **`artifacts/submission_1a2up.csv`** -- d6probe with +0.20 on CYP1A2 only. Placement
   changes are NOT diluted by the blend, which is why this outranks more training. It also
   restores the three endpoints lost to blend4tdi, so only CYP1A2 is at risk. Evidence:
   67.0% of CYP1A2 predictions sit below their `lo` bound at our centre, OOF-optimal shift
   +0.720, and probe2 moved CYP1A2 DOWN and the pair lost 0.3508.
   **Capture the PER-ENDPOINT scores** -- probe2's were never recorded, which is why its
   0.3508 had to be split between two endpoints by assumption.
2. `artifacts/submission_3a4up.csv` exists but is CONTRA-INDICATED: 66.9% of CYP3A4
   predictions are already ABOVE their `hi` bound and the OOF-optimal shift is -0.535.
3. Regenerate `mh_OLD`'s out-of-fold predictions from its 7 September checkpoints. Its OOF
   is a 263-row smoke stub, so **every pIC50 submission we have made contains a member that
   has never been locally validatable**, and the five-member blend cannot be scored offline
   until that is fixed.

## 8. Rules paid for in lost submissions

* **Before swapping a blend member, measure the member being REMOVED, not just the one
  going in.** A proxy must come from the INCUMBENT's family. (blend4tdi, -0.0014)
* **A bootstrap bounds sampling noise, never specification error.** P(worse) 0.0% on a
  mis-specified contrast is worth nothing. (blend4tdi)
* **A small predicted gain leaves no margin for a specification error**, so high confidence
  is worth least exactly when the effect is smallest. The size rule and the confidence rule
  are not independent.
* **Read the ledger, then check its last row against the newest file on disk.** Four
  stale-baseline incidents: Sections 51, 67, 68, 74.
* **Compute the oracle ceiling before building the estimator.** Section 72 closed a week of
  work in four minutes. Section 69's retracted 0.3797 is what happens otherwise.
* **A refutation that is crushing on every endpoint at once is a bug until proven
  otherwise.** (the dead-zone objective's missing `init_score`, Section 73)
