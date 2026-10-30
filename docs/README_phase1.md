# cyp — Phase 1

Foundation for the OpenADMET CYP Inhibition Blind Challenge. Three things, in dependency
order: a scorer you can trust, splits that match the task, and the label matrix the
tutorial throws away.

Everything runs on synthetic data with `--synthetic`, so the pipeline was exercised
end-to-end before the real CSVs arrived. The synthetic set is generated once and cached
in `artifacts/synthetic/`, shared by all three scripts — they must see the *same*
molecules or the folds built by one refer to different compounds than the models trained
by another, which fails silently because the compound names still line up. Delete that
directory to regenerate; `--synthetic-seed` picks a different draw.

## Get the data

Hugging Face is not reachable from every environment, so fetch these five files by hand
into `cyp/data/`:

```
https://huggingface.co/datasets/openadmet/cyp-challenge-train-test

  cyp-challenge-TRAIN_inhibition.csv
  cyp-challenge-TRAIN_TDI.csv
  cyp-challenge-TRAIN_Emax.csv
  cyp-challenge-single-concentration-TRAIN.csv
  cyp-challenge-TEST-BLINDED.csv
```

Or point `CYP_DATA_DIR` somewhere else. `CYP_CHALLENGE_REPO` must point at the directory
containing the tutorial's `evaluation/` package; it is auto-detected if the tutorial repo
sits next to this one.

```bash
pip install -r requirements.txt
python scripts/check_env.py     # confirms rdkit/lightgbm/loguru and the descriptor path
```

**Pin your RDKit for the duration.** The descriptor set is version-dependent — 2022.09
exposes 208 descriptors, 2026.03 exposes 217 — so the same code on two RDKit versions
gives different models and non-comparable scores. `features.py` picks the right
descriptor path automatically (`Descriptors.CalcMolDescriptors` only exists from
2023.09), so either works; just don't change versions mid-project. `check_env.py` prints
which path it took.

## Run

```bash
python scripts/make_folds.py                 # calibrate cutoff, freeze folds
python scripts/build_labels.py               # censored label matrix + sparsity report
python scripts/run_baselines.py              # the ladder, scored on both metrics
```

Add `--synthetic` to any of them to run without data. Add `--fast` to `run_baselines.py`
for 100 bootstrap samples instead of 1000.

## What each piece does

**`cypkit/scoring.py` — the scorer.** Imports the challenge's own `evaluation/` package
rather than reimplementing ST-RAE, so there is exactly one definition of the metric in
this project and it is theirs. `oof_report()` scores out-of-fold predictions against the
*training set's* `_conf_low`/`_conf_high` columns, giving leaderboard-identical numbers
offline, bootstrap CIs included. It refuses to run without the bound columns, because
without them ST-RAE silently degrades to plain RAE and the number looks fine.

**`cypkit/splits.py` — the folds.** `calibrate_cutoff()` measures the actual median
max-Tanimoto from test to train, then reports which Butina cutoff produces held-out folds
at the same distance. The cutoff is a measurement, not a convention. Folds are written
once to `artifacts/folds.csv` and the loader refuses to regenerate them, because
regenerating mid-project makes every earlier number incomparable. Screen-only compounds
are placed in the fold of their nearest labelled neighbour so their analogues don't
straddle the split.

**`cypkit/labels.py` — the censored matrix.** One long table, one row per
(compound, isoform):

| kind | y_low | y_high | meaning |
|---|---|---|---|
| `interval` | fitted lower bound | fitted upper bound | a measured dose-response curve |
| `left` | −inf | floor (pIC50 4) | screened, didn't hit — the true value is below the floor |

A measured value that sits at or below the floor becomes `left` too: it is a bound, not a
measurement. A band that merely dips below the floor keeps its fitted lower edge.

**`cypkit/losses.py` — the dead-zone objective.** One residual definition covers both
rows: zero inside `[y_low, y_high]`, distance to the nearest edge outside it. A measured
compound gets its credible interval; a non-hit gets `(−inf, floor]`, so it is free below
the floor and charged above it. Huberised near the edge so the Hessian stays positive
(constant 1.0 — the trick LightGBM's own L1 objective uses; near-zero Hessians make leaf
values explode, which is exactly what happened on the first attempt).

**`cypkit/features.py` — descriptors.** RDKit 2D, plus an optional 3D block that embeds a
real ETKDGv3 conformer first and returns a validity mask instead of zero-filling
failures. The PXR project lost NPR1/NPR2/Asphericity to a silent zero-fill behind a bare
`except` across its entire v4 series; this is the fix, ported from the one place that got
it right (the UniMol notebook).

## The two scorers, and why there are two

`run_baselines.py` reports both:

- **ST-RAE** — leaderboard-identical, via the challenge's own code.
- **dense proxy** — the same dead-zone error over the *full* (compound, isoform) matrix,
  censored rows included.

They exist separately because of a trap worth stating plainly: **the honest local scorer
inherits the training set's selection bias.** It can only score compounds that have a
measured pIC50, and a compound only has one because it hit the primary screen. So it
cannot see the inactives, and it marks a model down for learning that most compounds are
inactive — which is the single most useful thing a model can learn here, because the real
test set was assayed densely and is full of them.

Track both. Trust the dense proxy for anything about censoring; trust ST-RAE for anything
about potency ranking among actives.

## Real-data results (2026-09-01, 5 folds, 100 bootstrap samples)

`verify_strae()` confirms the local ST-RAE matches the challenge's own implementation to
**0.00e+00** — the numbers below are the leaderboard metric, not an approximation of it.

### The label matrix

| | rows | share of the full (compound, isoform) matrix |
|---|---|---|
| dose-response only | 6,525 | 37.3% |
| + primary-screen non-hits | 14,949 | 85.4% |
| **recovered** | **8,424** | rows the tutorial discards |

Primary-screen hit rates are much higher than the "single digit" impression the notebook
gives: CYP1A2 29.9%, CYP2C9 33.2%, CYP2D6 27.4%, **CYP3A4 66.5%**. Note also that 24% of
*measured* pIC50 values sit at or below the floor, so they enter as bounds, not points.

### The dead zone is enormous at the weak end

Median credible-interval width, pooled across isoforms:

| true pIC50 | n | median CI width |
|---|---|---|
| < 4.0 | 1,564 | **2.141** |
| 4.0 – 4.5 | 1,161 | 0.428 |
| 4.5 – 5.0 | 1,586 | 0.308 |
| 5.0 – 5.5 | 1,254 | 0.246 |
| ≥ 5.5 | 960 | **0.195** |

An 11× ratio — larger than the 5.3× measured on the PXR unblinded set. Weak compounds are
nearly free; potent compounds decide the score.

### The split advice inverts on this dataset

| | median max-Tanimoto | ≥ 0.50 | ≥ 0.60 |
|---|---|---|---|
| test → train | **0.587** | 95.7% | 44.0% |
| held-out → other folds (near-random split) | **0.438** | 28.2% | 9.5% |

The test compounds are purchased analogues of screening hits, so they sit *closer* to the
training set than training compounds sit to each other. **No Butina cutoff reproduces the
gap** — the calibration table bottoms out at 0.438 with cutoff 0.2, which is effectively a
random split. Every structure-based split therefore makes cross-validation *harder* than
the real task, not easier. This is the opposite of the usual advice, and the opposite of
what this README said before the data arrived.

It matters less than it might, because ST-RAE turns out to be roughly flat across
similarity bands (CYP3A4 improves with similarity, CYP2C9 worsens, the rest are flat), and
the similarity-matched score tracks the plain one closely (0.7826 vs 0.7843 for
`tutorial`). Both are reported; treat the matched one as the headline.

### The ladder

```
         variant  ST-RAE  ST-RAE_lo  ST-RAE_hi     MAE  Spearman  matched  dense_STRAE  in_band
        tutorial  0.7843     0.7601     0.8077  0.5882    0.5329   0.7826       1.0519   0.2867
              l1  0.7518     0.7335     0.7691  0.5772    0.5345   0.7500       1.1157   0.2492
        deadzone  0.8576     0.8408     0.8804  0.6575    0.4484   0.8293       1.3896   0.1344
censored_imputed  1.0172     0.9923     1.0505  0.6870    0.5335   0.9424       0.6738   0.2867
        censored  1.0608     1.0357     1.0853  0.7253    0.4554   0.9987       0.7582   0.3677
```

1. **L1 beats L2, and the confidence intervals do not overlap** (0.7518 [0.7335, 0.7691]
   vs 0.7843 [0.7601, 0.8077]). This reverses both the synthetic result and the PXR
   caution that an L1 objective flattered OOF and hurt the blind set. Use `regression_l1`.
2. **The dead-zone objective is worse on measured compounds** (0.8576), on real data as on
   synthetic. Its value is that it makes censoring expressible, not that it is a better
   loss for curves.
3. **Censoring still splits the two scorers.** It costs ~0.28 on the leaderboard metric and
   gains ~0.38 on the dense proxy — the same divergence as synthetic, now on real data.
   The open question is which one the blind test set behaves like; it is dense, so the
   dense proxy is the better guide, but that is an argument, not a measurement.
4. **Imputation still beats censoring** (0.6738 vs 0.7582 dense), and on real data the
   synthetic caveat no longer applies. Censoring does put far more predictions inside the
   band (0.368 vs 0.287); it just leaves the stragglers further out.
5. **CYP2D6 is the endpoint to fix.** Per-isoform ST-RAE for `l1`: CYP3A4 0.570, CYP2C9
   0.691, CYP1A2 0.834, **CYP2D6 0.913**. Macro-averaging weights them equally, and 2D6
   has the tightest credible intervals (median 0.272), so its dead zone forgives least.

### Do the CYP descriptors help?

`--features {rdkit,cyp,both}` measures it rather than assuming. MA-ST-RAE:

| variant | rdkit (217) | cyp (50) | both (267) | both − rdkit |
|---|---|---|---|---|
| tutorial | 0.7958 | 0.9046 | 0.7843 | **−0.0115** |
| l1 | 0.7564 | 0.8101 | 0.7518 | **−0.0045** |
| deadzone | 0.8731 | 0.9208 | 0.8576 | −0.0155 |
| censored_imputed | 1.0354 | 1.1022 | 1.0172 | −0.0183 |
| censored | 1.0759 | 1.1729 | 1.0608 | −0.0151 |

Adding the block helps **every variant on both metrics — 10 out of 10** — but by only
0.5–2% relative, well inside any single comparison's bootstrap CI. The consistent sign is
the evidence, not the size.

Feature-importance says where it comes from, and it is not where the pharmacology
suggested: the CYP block takes 5–8% of total gain while being 18.7% of the features, and
almost all of that goes to four aggregate descriptors — `cyp3a4_bulk` (logP × size),
`planarity_index`, `aromatic_ch`, `frac_aromatic_atoms`. The mechanistic features
(heme-ligand counts, isoform pharmacophores, MBI alerts) contribute very little to
**direct inhibition**. That is a fair negative result for this track. The MBI alerts were
always a TDI hypothesis, and the TDI track is where they should be tested.

## Layout

```
cypkit/config.py     endpoints, floor, hit thresholds, paths
cypkit/io.py         loaders, column normalisation, synthetic stand-in
cypkit/scoring.py    ST-RAE via the challenge's own code + the dense proxy
cypkit/splits.py     Butina clustering, cutoff calibration, frozen folds
cypkit/labels.py     the censored label matrix
cypkit/losses.py     dead-zone objective, gradients, standalone ST-RAE
cypkit/features.py   RDKit 2D + honest 3D
scripts/             three CLI entry points
artifacts/           folds.csv, labels.csv, oof_*.csv, summaries
```
