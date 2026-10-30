# Contributing

Working rules for this repository. Short on purpose; the reasoning behind each rule lives
in `artifacts/submissions.csv` and the `SECTION*.md` notes.

## Environment

Windows, conda. Everything runs in the `admet_gpu` environment:

```
conda activate admet_gpu
python scripts\<name>.py --help
```

Scripts take single-line commands. Use `^` for continuation, never `\`.

PyTDC is not installed and must not be added -- it has broken this environment before.

## Layout

```
cypkit/       importable package: config, features, submission writer, external prior
scripts/      one file per experiment or build step, each runnable standalone
artifacts/    submissions and the ledger (everything else is gitignored)
data/         challenge CSVs, fetched by hand -- see README
```

Model weights, out-of-fold matrices and prediction files are **not** committed. They are
15 GB and regenerable. `.gitignore` enforces this; if you need to share one, say where it
came from rather than committing it.

## Every script is a document

A script that encodes a decision carries a module docstring explaining **why**, not just
how. State what was measured, what the alternative was, and what would falsify the result.
A reader six weeks later needs the reasoning, and that reader is usually you.

## Four rules that are not negotiable

These exist because each one has already cost a submission slot.

1. **Compare against what actually shipped.** Not a proxy, not a re-run, not the same
   model family. If the incumbent is a blend, reconstruct the blend. Give the incumbent
   every advantage the candidate gets -- including refitting its scale after any
   averaging.

2. **Gate before you submit.** A candidate needs a paired bootstrap over compounds with
   `P(worse) < 0.05`, and the gain must exceed the board's own sampling noise. A gap
   smaller than its own confidence interval is not a result.

3. **Validate the ruler, not just the model.** Out-of-fold scores are measured on a harder
   population than the blind set (median max-Tanimoto 0.438 against 0.587). Before a new
   yardstick picks anything, make it reproduce board results it has not seen.

4. **Pre-flight every submission file.** Run the checks in `cypkit/submission.py`: SMILES
   first, both id columns, all four value columns, 750 rows, no duplicate names, no NaN or
   inf. Build by copying a file the platform has already accepted and replacing only the
   columns that change.

## Working with submission files

Thirteen SMILES are quoted CXSMILES containing commas (`|&1:17,19|`). Splitting a line on
commas left-to-right corrupts those rows silently. Use `pandas`, the `csv` module, or
`rsplit(",", 4)` -- the four value columns are last and never contain a comma.

A monotone transform that does not return Spearman `1.000000` against its source is a
parsing bug, not a rounding artifact.

## The ledger

`artifacts/submissions.csv` is the project's memory: every scored submission, per endpoint,
with the reasoning and the result. Append to it in the same change that produces a result,
including negative ones. Refutations are the most valuable rows in the file.

Columns: `track, order, name, endpoint, metric, value, rank, note`.

## Commits

- One logical change per commit; present tense, say why in the body.
- Never commit weights, raw data, credentials, or anything under `models_*/`.
- `git status` before every commit -- `.gitignore` is wide and a new output directory can
  slip past it.
