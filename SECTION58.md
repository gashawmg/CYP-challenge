# Section 58. The struct-free splice, and the first TDI candidate that does not need the 2.25

Run date 2026-10-12. `tdi3a4b` (lgb_clf alone, no struct, no pooled) and `tdi2d6b` (six
similarity members plus pooled, no struct) both ran, and `scripts/tdi_splice.py` combined
them into `artifacts/submission_tdi_structfree.csv`.

I had been comparing the wrong things in Sections 55 to 57. Comparing a local macro MCC
against a scored macro MCC is comparing two different instruments. The right comparison is
**column against scored column**, and when you do it that way the contested shrinkage
factor cancels.

---

## 1. The file, verified

| | |
|---|---|
| rows | 750 |
| CYP3A4 positive rate | 0.2813 submitted, ~0.306 effective (target 0.305) |
| CYP2D6 positive rate | 0.1133 submitted, ~0.142 effective (target 0.142) |
| CYP3A4 column | identical to `submission_tdi3a4b.csv` on 750/750 |
| CYP2D6 column | identical to `submission_tdi2d6b.csv` on 750/750 |
| versus the standing best | CYP3A4 differs on 73 compounds, CYP2D6 on 49 |

No `struct` anywhere in it.

## 2. The shrinkage factor cancels

`tdi_validator` corrects CYP2D6's out-of-fold d' by blind/OOF = 2.25, fitted on one
submission, and Section 53 failed to find a mechanism for it. But that number only has to
be known if you are predicting an ABSOLUTE score. Comparing a new column against a column
that has already been scored, through the same pipeline, needs only that the ratio holds:

```
new blind d'  =  f * new OOF d'
old blind d'  =  f * old OOF d'   (known: 0.980, from the scored submission)
                                  ->  new blind d' = 0.980 * (new OOF d' / old OOF d')
```

and `f` drops out. The same for CYP3A4 with its 1.290.

| | new column | new OOF d' | scored column's OOF d' | change |
|---|---|---|---|---|
| CYP3A4 | `lgb_clf` | 1.2947 | 1.2900 | **+0.4%** |
| CYP2D6 | 6 similarity + `pooled` | 0.5352 | 0.4356 | **+22.9%** |

Carrying those ratios through at the splice's optimal rates:

| | scored last time | predicted |
|---|---|---|
| CYP3A4 | 0.4470 | ~0.449 to 0.452 |
| CYP2D6 | 0.2180 | ~0.258 to 0.268 |
| **macro** | **0.3325** | **~0.355** |

**Roughly +0.022, and none of it rests on the value of the shrinkage.**

### Why the ratio argument is defensible here

`tdi_weights.py`'s lesson is that different member FAMILIES shift differently between
fold chemistry (median max-Tanimoto 0.44) and blind chemistry (0.587) -- similarity
kernels improve as the test set moves closer, rule-derived members do not. A ratio
argument is only safe when the two columns are from the same family, and here they are:
the scored CYP2D6 column was the AUC-fitted blend that loaded on knn_tan 0.32 and svc_tan
0.30, and the new column is six similarity members plus `pooled`. Same family, more of it.
That is precisely the comparison the 2.25 was measured on.

It would NOT be safe on a struct-loaded CYP3A4 column, which is the other reason to have
dropped it.

## 3. Correcting Sections 55 to 57

Those sections read "macro 0.2981 local against 0.3325 scored, so it does not clear the
gate". That comparison was never valid -- the 0.2981 is an out-of-fold number at blind
rates and the 0.3325 is a board number, and the gap between those two instruments is
exactly what the shrinkage describes. Read column-against-column, the same candidate is
about 0.022 ahead.

What survives from those sections unchanged: `struct` is the only thing that ever improves
CYP3A4 locally and the only thing that has ever damaged it live, so it stays out; Avalon
earned its place on CYP2D6 by disagreement; the five later fingerprint families bought
nothing; and the equal-weight blend over all 32 members is actively worse than its own best
member on CYP3A4.

## 4. Recommendation: pIC50 first, this second

Both tracks share one slot a day. The pIC50 board is currently showing **0.4934, rank 47**
because `d6wide` lost and the board displays the latest submission -- so every day that is
not corrected, we sit 0.0197 below our own best. That is an active loss. The TDI board is
sitting at its best.

So: **`probe2` today** -- it restores CYP2D6 to the 0.5621 column and buys the CYP1A2 and
CYP3A4 placement gradients in the same upload. **`submission_tdi_structfree.csv`
tomorrow.**

When the TDI file does go up it also pays twice, because a second scored TDI submission is
the only way to test whether the 2.25 is real: this candidate's prediction was derived
without it, so comparing the outcome against ~0.355 measures the column change directly
and gives the shrinkage its second anchor at the same time.

Recorded because the candidate had been sitting behind a comparison I was making wrongly,
and the fix was to stop comparing a local number with a board number.
