# Track B — ChemProp (your box) and UniMol (Kaggle), multitask baselines

Two files, one architecture decision each, same rung of the ladder: **learned
features only, no target-relevant descriptors**. Run these first, record the
numbers, then add descriptors as rung 2.

- `train_cyp_mt_v1.py` — port of your `train_pec50_pxr_v4_4.py`
- `unimol_kaggle_cyp_multitask_5fold.ipynb` — port of `unimol_kaggle_aug2_8fold.ipynb`

---

## Two bugs found in the source files

**1. `--no_sample_weights` in v4_4 had `default=True`.** The flag was permanently
on, so sample weights were dead in every v4_4 run whether or not you passed it.
The A/B test that flag was written for could never have shown a difference. In
the CYP port the switch is inverted to `--sample_weights` (opt-in, default off),
which both fixes the logic and gives the baseline a clean architecture
measurement. If you ever back-port to PXR, that changes v4_4's behaviour.

**2. The UniMol OOF folds did not match the models that produced them.** Your
notebook built OOF folds with a greedy largest-scaffold-first bin-pack, but
`unimol_tools` splits with `sklearn.GroupKFold(n_splits=k)` grouped by Murcko
scaffold (`unimol_tools/data/split.py`, line 45) and saves `model_i.pth` in that
enumeration order. The two assignments are unrelated, so for most compounds the
"OOF" prediction came from a model that had trained on that compound. That
inflates the OOF and, more importantly, any blend weight fitted on it — which
would have quietly over-weighted UniMol in your PXR ensemble. Cell 7 of the new
notebook replicates `GroupKFold` exactly.

---

## 1. ChemProp — `train_cyp_mt_v1.py`

```
conda activate admet_gpu
python train_cyp_mt_v1.py --mode BOTH          # activity_scaffold, as in v4_4
```

**Multitask with native masking.** Four targets in one model; missing labels stay
NaN. Chemprop 2.x does `mask = targets.isfinite()` inside `MPNN.training_step`, so
a compound contributes to exactly the isoforms it was measured on — no imputation,
no dropped rows. Verified against the installed 2.3.1.

**Adaptive config unchanged.** N = 4,905 lands in `very_large`: hidden 768,
depth 6, 4 FFN layers, dropout 0.20, augmentation 1x. Your tiers carried over
verbatim.

**Per-task y scaling.** `RobustScaler` cannot take NaN, so each target column is
fitted on its own non-missing rows and the NaNs are restored afterwards. One
scaler per isoform, saved to `scalers_y.pkl`.

**Split: `activity_scaffold` is the default, matching v4_4.** Generalised to
multitask by binning each row on the mean of its z-scored *available* targets, so
a compound measured on one isoform is binned on that isoform rather than dragged
to the middle by three NaNs.

Measured on this dataset, so the choice is a number rather than a label:

| split | scaffold leakage across folds | median held-out max-Tanimoto |
|---|---|---|
| `scaffold` | 0.0% | 0.433 |
| `activity_scaffold` | **11.2%** | 0.438 |
| *blind test -> train* | — | **0.587** |

`activity_scaffold` deals multi-member scaffold groups round-robin across folds,
so analogues do span the split — it is activity-stratified but only partly
scaffold-held-out. On this dataset that costs almost nothing in difficulty (0.438
vs 0.433) and buys real activity balance, which matters because three of four
endpoints are weak and a fold could otherwise end up nearly free of actives. Every
run now prints the leakage figure for whichever strategy you pick.

The caveat that actually bites is the third row: **both splits are harder than the
real task.** Blind-test compounds were purchased as analogues of screening hits and
sit at 0.587 to the training set. Any structure-based CV here is pessimistic
relative to the leaderboard — which is why the composite validator is calibrated
against real leaderboard results, not against a CV number.

**Use `--split_strategy frozen --folds_csv artifacts/folds.csv` if you intend to
blend.** Blend weights fitted across two different fold assignments are fitted on
compounds that were in-fold for one member and out-of-fold for the other, which
inflates whichever member saw them. Run both if you have time; they answer
different questions.

Rung 2, once the baseline is recorded:

```
python train_cyp_mt_v1.py --mode BOTH --use_descriptors \
    --model_dir models_cyp_mt_v2 --output predictions_cyp_mt_v2.csv \
    --oof_output oof_chemprop_mt_v2.csv
```

**A smoke test already ran here** — 2 folds, patience 1, so essentially one epoch
of training on CPU. Even that undertrained gave:

```
endpoint   ST_RAE     classical ensemble OOF
CYP1A2     0.9288  <- 1.050
CYP2C9     1.0035     0.951
CYP2D6     1.0711     1.047
CYP3A4     0.8688     0.495
MA         0.9681     0.886
```

CYP1A2 already beats the classical ensemble after one epoch. That is the shared
trunk doing exactly what it was brought in for. CYP3A4 is far behind only because
it is the endpoint with enough data to need real training. Do not read anything
else into these numbers — they are a plumbing check, not a result.

## 2. UniMol — `unimol_kaggle_cyp_multitask_5fold.ipynb`

Upload to Kaggle with a dataset containing `cyp-challenge-TRAIN_inhibition.csv`
and `cyp-challenge-TEST-BLINDED.csv`, enable the GPU accelerator, run top to
bottom.

**Multitask with native masking, confirmed in the source.** `task=
'multilabel_regression'` uses the `MAEwithNan` loss (`models/loss.py`: `mask =
~torch.isnan(y_true)`) and `TargetScaler` calls `np.ma.masked_invalid`. So NaN is
the correct missing-label marker here too — do not impute.

Batch size raised 4 -> 8 (four targets per row, still comfortable in 16 GB);
8-fold -> 5-fold; the holdout-30 path dropped, no equivalent here.

**Split note.** `unimol_tools` offers only random / scaffold / group / stratified —
there is no activity-stratified-scaffold option, so the notebook uses `scaffold`
(GroupKFold on Murcko). That is *stricter* than the chemprop default: 0% scaffold
leakage against 11.2%. Expect slightly worse OOF numbers from UniMol than from
chemprop at equal model quality, and do not read the gap as a model difference.

Cell 8 prints ST-RAE per endpoint next to the classical ensemble's numbers, so
you can see immediately whether the 3D representation is buying anything on the
three starved isoforms.

## 3. Folding the results back in

Both write OOF in the same shape (`Molecule_Name` + four endpoint columns):

```
python scripts/blend_composite.py --oof artifacts/oof_*_ensA.csv artifacts/oof_chemprop_mt_v1.csv
```

**Judge them on the composite, not on OOF ST-RAE alone.** Two leaderboard results
pinned the conversion:

```
predicted actual = 0.5 * OOF_ST-RAE + 0.5 * dense_ST-RAE + 0.080
```

which fits both anchors to 0.001 and 0.002. The composite blend currently sits at
a predicted 0.8514, and 0.8278 after per-endpoint expansion calibration. A new
member has to move that number, not just look good on one scorer.

**Where the headroom is.** Three of four endpoints are at or above ST-RAE 1.0,
which on this metric means no better than a constant predictor:

| endpoint | curves | OOF ST-RAE | Spearman |
|---|---|---|---|
| CYP3A4 | 2,335 | 0.495 | 0.773 |
| CYP2C9 | 1,285 | 0.951 | 0.579 |
| CYP2D6 | 1,493 | 1.047 | 0.454 |
| CYP1A2 | 1,412 | 1.050 | 0.526 |

CYP3A4 is solved and diluted by three that aren't. Multitask borrowing is the
single most plausible fix, which is why both baselines here are multitask rather
than four single-task models — and why the smoke test's CYP1A2 number is the one
worth watching when the real run finishes.
