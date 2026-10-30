# Section 52. The is_TDI label is a formula, and the TDI file hides 1,244 CYP3A4 compounds

Run date 2026-10-11, overnight, while the pIC50 slot was on cooldown. Three findings, in
descending order of how much they are worth.

---

## 1. The exact label rule

`is_TDI` is not an opaque annotation. It is reproduced **exactly** by

```
is_TDI = (pIC50_TDI_condition - pIC50_direct_inhibition >= log10(2))
         AND (pIC50_TDI_condition >= 4.301)
```

with a missing direct curve making it False. Checked against all 6,145 rows of
`cyp-challenge-TRAIN_TDI.csv`:

| | labelled rows | rule agrees | disagreements |
|---|---|---|---|
| CYP2D6 | 1,497 | 1,497 | **0** |
| CYP3A4 | 3,584 | 3,584 | **0** |

The rule also calls every unlabelled row False, which is what they are. A grid search over
the second threshold shows the rule is exact for any cut in [4.298, 4.310], so the constant
is 4.301 and not a fitted artefact. Both numbers are the textbook TDI criteria: log10(2) is
a two-fold IC50 shift, 4.301 is pIC50 for 50 uM.

### What that structure says

Positives live entirely inside a gate -- a direct curve must exist AND the TDI-condition
pIC50 must clear 4.301:

| | gate size | positives inside | P(shift >= log10(2)) inside the gate | MCC of the gate alone |
|---|---|---|---|---|
| CYP3A4 | 1,406 / 3,584 | 764 / 764 | 0.543 | **0.648** |
| CYP2D6 | 1,407 / 1,497 | 324 / 324 | 0.230 | 0.133 |

For CYP3A4 the gate alone, known perfectly, would score 0.648 against the 0.39 a classifier
on structure gets. For CYP2D6 the gate is useless because 94% of labelled compounds are
already in it.

## 2. Imposing the rule does not beat a classifier, and the reason is specific

Three components were fitted on identical scaffold folds (HGB on 217 RDKit descriptors plus
512 Morgan counts): T, D, and whether a direct curve exists. Then the rule was applied.

| CYP3A4 | pearson |
|---|---|
| T, the TDI-condition pIC50 | 0.740 |
| D, the direct pIC50 | 0.750 |
| shift, regressed directly | 0.244 |
| **shift, by differencing T-hat and D-hat** | **0.150** |

The shift is a small difference of two large correlated numbers -- its own standard
deviation is 0.30 while each component is predicted with RMSE around 0.7. Differencing two
predictions is strictly worse than regressing the difference, and both are far too weak to
threshold at log10(2). CYP2D6 is worse still: shift pearson 0.057.

Scored with the decision cut chosen **out of fold**, so the threshold never sees the rows
it is scored on:

| scorer | CYP3A4 MCC | CYP2D6 MCC |
|---|---|---|
| binary classifier (control) | 0.3866 | 0.0075 |
| binary x gate | 0.3897 (+0.003, CI spans 0) | 0.0059 |
| binary x shift | 0.3737 | 0.0863 (+0.078, CI [+0.023, +0.134]) |
| binary x gate x shift | 0.3920 (+0.005, CI spans 0) | 0.0235 |

At **matched predicted-positive rates**, which removes the operating-point confound, the
picture is honest and small: on CYP3A4 `binary x gate` beats `binary` at every rate at or
above 0.17 (0.3844 vs 0.3516 at rate 0.22; 0.3952 vs 0.3700 at 0.30) and lifts AUC 0.7996
to 0.8095. On CYP2D6 the `binary x shift` advantage is entirely at the top of the list --
AUC is unchanged at 0.569 -- so it is a reshaping of the score, not new ranking information.

**Verdict: not submission-worthy from this prototype.** But the components here are far
weaker than the production models -- this CYP3A4 D model has pearson 0.750 where the
production one scores Spearman 0.8425 on the board. The gate's ceiling is 0.648 and the
gap to 0.39 is real, so the gate is worth one honest re-test with production-quality T and
D before being written off. The thing to build is a chemprop regression on
`{iso}_pIC50_TDI_condition` -- a target with 3,583 CYP3A4 and 1,497 CYP2D6 labelled values
that this project has never modelled -- and then multiply the existing TDI blend score by
`sigmoid((T_hat - 4.301)/0.3) * P(direct curve)`.

CYP2D6 TDI should be treated as near-unlearnable in this feature space: AUC 0.569, shift
pearson 0.057, and the gate holds 94% of compounds. Effort spent there has poor odds.

## 3. The finding worth the most: 1,244 unused CYP3A4 compounds

`cyp-challenge-TRAIN_TDI.csv` contains **1,249 CYP3A4 compounds with a TDI-condition pIC50
and no direct-inhibition pIC50**. They are not censored rows. They were never in the direct
panel at all: of the 1,249, eleven appear in `cyp-challenge-TRAIN_inhibition.csv` and none
of those eleven has a CYP3A4 direct value. Where both files do have a value, the two agree
to 0.00e+00, so this is a coverage gap, not a discrepancy.

Their TDI-condition pIC50 has median 5.40 -- these are potent compounds, not screen
failures. And on the 2,334 compounds where both numbers exist,

```
D = 0.9148 * T + 0.0391     r = 0.947     residual sd 0.351
```

against a CYP3A4 target whose own standard deviation is 1.093. So the surrogate carries
about 90% of the variance.

1,244 of them have no CYP3A4 row in `labels.csv` at all. Adding them takes CYP3A4 from
3,042 training rows to 4,286, **+41%**, on the endpoint ranked worst on the board (64th).
None of them is in the blind set.

### Tested, not asserted

HGB on the same features and the same scaffold folds, scored on held-out **real** curve
rows only:

| | pearson | spearman | RMSE |
|---|---|---|---|
| surrogate off (control) | 0.7466 | 0.7423 | 0.7276 |
| surrogate at weight 0.25 | 0.7539 | 0.7502 | 0.7183 |
| **surrogate at weight 0.50** | **0.7549** | **0.7519** | **0.7170** |
| surrogate at weight 1.00 | 0.7489 | 0.7469 | 0.7245 |

Paired bootstrap on the held-out rows: weight 0.50 changes RMSE by -0.0106 with a 95%
interval of [-0.0208, -0.0005] and P(better) 0.984. Weight 1.00 is not significant --
the surrogate's noise starts to dominate once it outweighs the real labels.

A +0.010 Spearman on a weak HGB baseline is not a big jump by itself. It is worth running
because the data is free, the gain is measured rather than modelled, and the production
model has 53% more curve rows to work with rather than 0.750 pearson of them.

### Files

- `artifacts/labels_3a4_surrogate.csv` -- the 1,244 rows in `labels.csv` format, with
  `kind=interval`, bounds at y +- 0.3508, `weight=0.5` and `source=tdi_surrogate` so they
  can be filtered out in one line.
- `artifacts/labels_plus_3a4.csv` -- `labels.csv` with those rows appended, 16,193 rows.
- `scripts/tdi_rule.py`, `scripts/tdi_rule2.py` -- the rule decomposition and its scoring.
- `scripts/surrogate_3a4.py` -- the surrogate experiment above.

## 4. What to do with the GPU tonight

1. **Retrain CYP3A4 direct inhibition on `labels_plus_3a4.csv` at surrogate weight 0.5**
   and score it on the existing blind-like holdout. This is the one change with a measured
   effect and free data behind it. Gate: it must beat the current CYP3A4 OOF, or it does
   not ship.
2. If that passes, CYP3A4 becomes the second probe in a later pIC50 submission, after
   `probe2` has returned its CYP1A2 and CYP3A4 placement gradients.
3. The TDI gate re-test is third. It needs a new chemprop target (`pIC50_TDI_condition`)
   and has a prototype result of +0.003 MCC with a confidence interval spanning zero, so
   it does not deserve GPU time before item 1.

Recorded because the label definition was sitting in the training files the whole time, and
reading it out turned a classification problem into two regressions -- which did not help
the classification, and did hand over 1,244 compounds for the track that was never the
point of the file.
