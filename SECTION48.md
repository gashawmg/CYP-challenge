# Section 48. Why the predictions stopped improving: the similarity ceiling

The question was "R2 is low and MAE is >= 0.5 -- how do we improve that?" The answer is
that on three of four endpoints there is very little left to improve, and this section is
the measurement that says so.

## 1. R2 was the wrong lens; use the trivial baseline instead

R2 = 1 - MSE/Var depends on the spread of the rows scored (Section 44 note). The
denominator-free version is our MAE against simply predicting the median:

| endpoint | our MAE | MAE of the median predictor | ratio |
|---|---|---|---|
| CYP1A2 | 0.649 | 0.731 | **0.89** |
| CYP2C9 | 0.467 | 0.592 | **0.79** |
| CYP2D6 | 0.563 | 0.629 | **0.89** |
| CYP3A4 | 0.494 | 0.885 | **0.56** |

We beat "one number for every compound" by 11-21% on three endpoints. Only CYP3A4 shows
real skill.

## 2. The whole field is in the same place

Best team's MAE against ours on the board: 0.619 / 0.677, 0.415 / 0.477, 0.911 / 1.143,
0.428 / 0.504. **The leaders are 6-20% better than us.** With 81 teams, nobody has broken
out. The credible intervals imply a repeat-measurement floor near 0.10-0.15 MAE, but that
is curve-fit uncertainty inside one experiment, not assay-to-assay reproducibility -- treat
it as an upper bound on headroom, not a target.

## 3. Similarity does not carry activity here

Mean |pIC50 difference| between two training compounds, binned by Tanimoto:

| endpoint | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 |
|---|---|---|---|---|---|---|
| CYP1A2 | 0.71 | 0.67 | 0.72 | - | - | - |
| CYP2C9 | 0.58 | 0.66 | 0.62 | - | - | - |
| CYP2D6 | 0.71 | 0.68 | 0.63 | - | - | - |
| CYP3A4 | 1.14 | 1.16 | 0.97 | 0.74 | **0.48** | 0.51 |

**Flat on three endpoints.** Two compounds at Tanimoto 0.55 differ as much as two at 0.35.
CYP3A4 behaves as chemistry says it should, falling 1.14 -> 0.48 -- and CYP3A4 is the one
endpoint we predict well. The endpoint where similarity tracks activity is the endpoint
that is modellable; the three where it does not, are not.

The sharpest form of this: for a normal with sd s, two random draws differ by 1.128 s. The
**five nearest labelled neighbours** of a blind compound differ by:

| endpoint | 5-NN disagreement | random-pair expectation | ratio | blind at or worse than random |
|---|---|---|---|---|
| CYP2D6 | 0.55 | 0.73 | **0.75** | 23% |
| CYP3A4 | 1.03 | 1.23 | 0.83 | 29% |
| CYP1A2 | 0.59 | 0.71 | 0.84 | 32% |
| CYP2C9 | 0.58 | 0.62 | **0.94** | **44%** |

**Choosing the five most similar labelled compounds buys 6-25% over choosing five at
random.** That is the entire structure-activity signal in these fingerprints at this data
density.

## 4. What the t-SNE figures show (`figures/tsne_*.png`)

- `tsne_train_vs_blind.png` -- blind compounds tile the same regions on every endpoint.
  There is no separated blind island anywhere. **The blind set is the same 750 compounds
  in all four panels**; what differs is which compounds carry a curve for that isoform.
- `tsne_activity_field.png` -- CYP1A2, CYP2C9 and CYP2D6 are speckled at every scale.
  CYP3A4 has visible structure.
- `tsne_neighbour_disagreement.png` -- each blind compound coloured by its own
  disagreement ratio, diverging about a true zero at 1.0. CYP2C9 carries a large red
  region across its lower right: a whole area where similar compounds have unrelated
  CYP2C9 activity.

Coverage of the LABELLED subset (not the library) also differs, more than expected:

| endpoint | labelled | median NN blind -> labelled | blind below 0.40 |
|---|---|---|---|
| CYP3A4 | 2335 | 0.538 | 21% |
| CYP2C9 | 1026 | 0.518 | 36% |
| CYP1A2 | 1180 | 0.516 | 31% |
| CYP2D6 | 1364 | **0.424** | **46%** |

These sit below the 0.587 in `artifacts/nnsim.csv`, which measured against the whole
4,906-compound library. The blind set is inside the library but further from the compounds
that actually have a curve.

## 5. Two things this does NOT support

**Applicability domain.** Structurally the blind set is inside our domain -- closer to
training than training is to itself. The term was used loosely in Section 45 and should not
have been.

**A CYP2D6 label-shift.** Tested directly: predict each endpoint's board MAE from our
submitted distribution, the anchor-solved blind distribution and the board's Spearman.
CYP1A2 0.99, CYP2C9 0.99, CYP3A4 1.00 -- and **CYP2D6 1.23**, the one endpoint the argument
depended on. The observed MAE is consistent with a blind CYP2D6 mean near 3.7-3.8 rather
than 2.92. The CYP2D6 anchor is the least reliable of the four and should be re-solved
once another submission scores.

Note the tension left over: CYP2C9 has the LEAST informative neighbourhoods (0.94x random)
yet is our second-best endpoint, and CYP2D6 has the MOST informative (0.75x) and is our
worst. Local similarity structure alone does not determine predictability -- the models are
extracting something beyond nearest-neighbour behaviour, consistent with CYP2C9's MAE of
0.477 sitting well below its 1-NN difference of 0.625.

## 6. What follows

The 6-20% that separates us from first place on MAE is roughly the whole available prize on
prediction error, and it is being contested at the information limit. Expect single-digit
percentages from further architecture work; today's best model change (Section 47,
descriptors on the multi-head trunk) was +0.0199 macro rho standalone and +0.0016 macro
after the blend absorbed it.

The levers that do not require beating the information limit are proportionally larger than
they were being treated as:

1. **Ranking**, where a blend still extracts a little.
2. **Spread**, which converts the same predictions into better ST-RAE and is measured, not
   argued (Section 46: -0.0569 CYP2C9, -0.0357 CYP3A4 at constant mean, never lost).
3. **The TDI track**, untouched today and at rank 32. The ceiling argued here is about
   pIC50 REGRESSION at the information limit; a classification task scored by MCC is a
   different problem and may not be at its own ceiling.

Recorded because the instruction to put modelling ahead of placement was reasonable and
produced a real result, and because this measurement is the strongest evidence yet that it
was the smaller half.
