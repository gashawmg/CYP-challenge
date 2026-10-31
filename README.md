# cyp — OpenADMET CYP Inhibition Blind Challenge

Two tracks, one shared daily submission slot:

| track | metric | best | Rank| 
|-------|--------|------|-----|
| Direct inhibition (pIC50) | MA-ST-RAE, lower better | **0.4407**| 42 | 
| TDI (classification) | MA-MCC, higher better | **0.3788**| 25 |

Predict pIC50 for 750 blind compounds across CYP1A2, CYP2C9, CYP2D6 and CYP3A4, scored by
a soft-thresholded relative absolute error that charges nothing inside a compound's
credible interval and the distance outside it.

## What is actually in here

Four model families — LightGBM on RDKit descriptors, a ChemProp D-MPNN, a Uni-Mol 3D
model, and a multi-head network with auxiliary heads — blended per endpoint, then mapped
onto the blind set with a fitted affine placement.

The placement turned out to matter more than the models. Spearman has been flat at ~0.756
throughout, while macro ST-RAE went 0.4737 → 0.4412 purely by moving each endpoint's
centre and spread. Most of this repository is the record of finding that out, and of the
nine modelling directions that were measured and closed.

`artifacts/submissions.csv` is the primary record: every scored submission, per endpoint,
with the reasoning and the outcome. Start there.

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
conda activate admet_gpu
pip install -r requirements.txt
python scripts/check_env.py     # confirms rdkit/lightgbm/loguru and the descriptor path
```

**Pin your RDKit for the duration.** The descriptor set is version-dependent — 2022.09
exposes 208 descriptors, 2026.03 exposes 217 — so the same code on two RDKit versions
gives different models and non-comparable scores.

## Reproducing a submission

Trained weights are not committed (15 GB, all regenerable). To rebuild the current
submission from a clean checkout you need to retrain the members first; see
`HOW_WE_PREDICT.md` for the pipeline and `RUNBOOK.md` for the commands.

To verify a submission file that already exists:

```
python scripts\build_reblend.py --dry-run
```

## Contributing

See `CONTRIBUTING.md` — in particular the four rules that each cost a submission slot to
learn. `PUBLISHING.md` is the checklist for making this repository public.
