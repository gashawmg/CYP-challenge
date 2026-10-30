# RUNBOOK: from here to the next submission

Windows, `admet_gpu`, one line per command. Run in order. Two hard gates; do not skip them.

```
conda activate admet_gpu
d:
cd D:\CYP450\cyp
```

---

## Step 0 — free, 2 minutes, no GPU

On the leaderboard, set the **Proprietary Data** filter to **No** and read off the best
open-data macro ST-RAE.

Why first: if `preheat-to-450` (0.3721) and `wbot` (0.3912) are training on proprietary
data, then 0.3721 is not our target and the effort split below should shift. Everything
after this step is hours of GPU; this is the one input that could change the plan.

---

## Step 1 — seed ensemble (the unlock)

Five seeds of the current best recipe. This is the largest change available that cannot
fail for a reason of substance: two identical runs correlate only r = 0.927, so a single
run carries ~7% pure seed noise.

```
python scripts\seed_ensemble.py --seeds 42 1337 2024 7 99 --tag ens5 --resume -- --descriptor_spec per_endpoint
```

- `--resume` is safe to re-pass after an interrupt; a finished seed is skipped.
- A seed that crashes is dropped loudly and never averaged in silently.
- Expect roughly 5x one training run. Leave it.

Outputs `oof_chemprop_st_v3_ens5.csv` and `predictions_cyp_st_v3_ens5.csv`, then prints the
per-seed vs ensemble table on the blind-like holdout.

**Read that table before going on.** If `ens` does not beat `mean seed` on at least three
of four endpoints, stop and say so — averaging failing means something is wrong with the
members, and nothing below is worth running.

The same table prints the mean pairwise r between seeds. That number is the noise floor
every future A/B has to clear, and it is the second reason this step exists.

---

## Step 2 — register the new member

Edit `scripts\pic50_blend_search.py`, in `MEMBERS` (around line 45), add:

```python
    "cp_ens5": ("oof_chemprop_st_v3_ens5.csv", "predictions_cyp_st_v3_ens5.csv"),
```

---

## Step 3 — blend search, no writing yet

```
python scripts\pic50_blend_search.py
```

---

## GATE A — the submission bar

Sum the **out-of-sample** gain across the four endpoints, each on its own honest yardstick
(Section 40): CYP1A2 `hist x sim`, CYP2C9 `histogram`, CYP2D6 `plain`, CYP3A4 `hist x sim`.

- Bar: **+0.160** summed out-of-sample rho against blend4z.
- Where it stood before this work: **+0.0518** (32% of the bar).

**If the sum clears +0.160 → go to Step 6.**
**If it does not → Step 4. Do not submit a partial improvement; the slot is worth more than
that.**

---

## Step 4 — CYP2D6 only (if Gate A fails)

CYP2D6 is 31% of our macro cost from 25% of the weight, is our worst endpoint on every
metric, is the only endpoint blending does not help (+0.0006 out-of-sample), and is our
best *rank* (18) because the field is weakest there. It is where a point of rho buys the
most.

```
python train_2d6_external.py
python scripts\seed_ensemble.py --seeds 42 1337 2024 --tag 2d6ext --resume -- --isoforms CYP2D6 --descriptor_spec per_endpoint
```

Then re-test, now that the floor is low enough for the answer to mean something, whether 3D
belongs on CYP2D6 after all — Section 43 removed it on a single pair of runs whose
difference sat near the noise floor:

```
python scripts\seed_ensemble.py --seeds 42 1337 2024 --tag 2d6_3d --resume -- --isoforms CYP2D6 --descriptor_spec lean3d
```

Register both, rerun Step 3, return to Gate A.

Do **not** spend effort on CYP3A4 ranking: 58 teams are already inside our number there and
our rho is 0.8593 against a best of 0.8885.

---

## Step 5 — TDI, only if it has a validated candidate

One slot per day, shared across both tracks. Spend it on whichever track has a candidate
that passed its gate, not on whichever finished most recently.

---

## Step 6 — write the blended predictions

```
python scripts\pic50_blend_search.py --write
```

Writes `pred_st_blendsearch.csv`.

---

## Step 7 — the spread decision

```
python scripts\blind_response.py --propose
python scripts\make_submission.py --pred pred_st_blendsearch.csv --tag cand1 --dry-run
```

The dry run prints each endpoint's mean and sd against the anchor-solved blind
distribution. Measured on the real blind set, from controlled pairs that shipped the same
ranking twice:

| move | result |
|---|---|
| narrow the spread at constant mean | CYP2C9 **-0.0569**, CYP3A4 **-0.0357** — never lost |
| lower the mean | up to **+0.2391 worse** — lost 4 of 5 |

CYP1A2 ships sd/needed 0.70, the least shrunk of the four, and has never been narrowed at
constant mean. It carries the most headroom (+0.0962) with no measurement against it.

To apply it:

```
python scripts\make_submission.py --pred pred_st_blendsearch.csv --tag cand1 --shrink CYP1A2=0.80 --dry-run
```

There is deliberately **no `--shift`**. The mean guard refuses any write where a mean moved
more than 0.01, and rankings are asserted bit-identical.

Treat the shrink as optional and separable: if Gate A passed on modelling alone, the
submission is already worth making without it.

---

## Step 8 — write and upload

```
python scripts\make_submission.py --pred pred_st_blendsearch.csv --tag cand1 --shrink CYP1A2=0.80
```

Writes `artifacts\submission_cand1.csv` as a copy of an accepted reference with only the
value columns replaced, so the layout cannot drift.

Upload it by hand through the web form, as usual.

---

## GATE B — after it scores

Add the per-endpoint row to `artifacts\submissions.csv`: ST-RAE, MAE, rho, R2, rank, per
endpoint and overall.

This is not bookkeeping. That file is the instrument `scripts\blind_response.py` runs on,
and it is the only place we can measure the true scoring function. Every row skipped is a
controlled experiment thrown away — the CYP3A4 MAE that went unread on day 15 is why the
`raw3a4` probe is half-usable now.

---

## The two numbers to keep in view

- **0.4833** — our live macro. Nothing ships unless local validation says it beats this.
- **+0.160** — summed out-of-sample local rho that has historically been needed to move the
  board, at the measured 14% transfer rate.
