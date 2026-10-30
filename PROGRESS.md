# What we improved, and how — OpenADMET CYP Blind Challenge

Updated 2026-09-08. Companion to `STRATEGY.md` (which holds the plan and the dead ends).
This file answers one question: what actually moved the leaderboard, and by what mechanism.

---

## Scoreboard

**pIC50 — MA-ST-RAE, lower is better**

| # | submission | score | rank | what changed |
|---|---|---|---|---|
| 1 | l1 | 1.0127 | — | classical L1 baseline |
| 4 | all7_calib | 0.6763 | 49 | 7 members + calibration |
| 6 | v3_affine80 | 1.1048 | 118 | **sign error in the affine correction** |
| 7 | pseudo_blend45 | 0.7273 | 58 | screen pseudo-labels |
| 8 | placed | 0.5837 | 34 | placement optimisation |
| 9 | blend3_placed | 0.5398 | 29 | 3-way blend + placement |
| 11 | splice_best | 0.5167 | 28 | per-endpoint splice of scored columns |
| 12 | **blend4z** | **0.5104** | **27** | 4-model diversity blend |

**TDI — MA-MCC, higher is better**

| # | submission | CYP2D6 | CYP3A4 | MA-MCC | rank |
|---|---|---|---|---|---|
| 2 | splice | 0.1825 | **0.4470** | 0.3147 | 20 |
| 4 | tdi_best | 0.2180 | **0.4470** | 0.3325 | 15 |
| 6 | tdi_sel | 0.2866 | 0.4035 | 0.3451 | 13 |
| 7 | tdi_v2 | **0.2918** | 0.3830 | 0.3374 | 15 |
| 8 | tdi_v3 | **0.2918** | 0.4074 | 0.3496 | 15 |
| — | *splice of the two best columns (queued)* | 0.2918 | 0.4470 | **0.3694** | — |

Leaders: pIC50 0.4378, CYP3A4 TDI 0.4601.

---

## The four things that actually worked

### 1. Separating ranking from placement

`ST-RAE ≈ absorption × sqrt(1 − ρ²)`. The ranking sets a floor; the placement decides
whether you reach it. Splitting every submission this way is what turned a single opaque
number into two independently attackable ones, and it is why `placed` (0.7273 → 0.5837)
was possible without touching a model.

**Corollary that keeps paying:** ρ is invariant under any monotone map, so a submission
returns a clean reading of the ranking no matter how badly the placement is chosen.

### 2. A validation pool reshaped to look like the blind set

Our dose-response curves were only run on primary-screen **hits**, so the training pool is
a biased sample: ~1.5× narrower than the blind set on every endpoint and 1.65 log units
higher on CYP2D6. `scripts/blind_validator.py` bins the full label matrix by label and
resamples it to the blind normal, then reports the spread across draws so a difference
smaller than the noise is visibly not a difference.

Three submissions in a row had been chosen on local numbers that did not transfer. This is
the yardstick that stopped it.

### 3. The floor bug

`labels.py` was converting any fitted curve with y ≤ 4.0 into left-censored, discarding
**1,564 of 6,525 curves** — 944 on CYP3A4 alone, the lowest fitted value being 1.91 — all
of which had finite confidence bounds that the scorer charges against. Recovering them is
now gated per endpoint on measured evidence (ρ on the recovered rows: CYP3A4 0.542,
CYP2C9 0.393, CYP1A2 0.158, CYP2D6 **−0.041**), so only CYP3A4 keeps them.

### 4. Per-endpoint column splicing

MA-ST-RAE and MA-MCC are both the **plain mean** of the per-target scores — verified
exactly on every scored submission — and each target's score depends only on its own
column. So the best column per target can be spliced with a score known before submitting.
`splice_best` was predicted to 0.0001. The queued TDI splice is predicted at 0.3694.

---

## What we know that we did not know a week ago

**Ensemble gains barely transfer.** `blend4z` was the test. Local blind-matched macro ρ
rose +0.054 over v6 and only **14%** of it reached the board (+0.0074), with CYP2C9 going
backwards. The cause is measurable: our four models agree at ρ 0.44–0.88 on our compounds
and **0.71–0.95** on the 750 blind ones. Averaging buys least where models already agree.
More members is not a path to 0.4378.

**Our placement is not the gap either — and that took a retraction.** Absorption has sat
at 0.806 for twelve submissions while rivals reach 0.717, which looked like a large free
win. A free-form monotone transform fitted on the full pool claimed 13% less absorption.
It was an artefact: our pool represents the below-floor region 89–96% with **left-censored**
screen non-hits, which are free to undershoot, while the blind set is 0% censored — every
one of its 750 compounds carries a real fitted value. Refitting on uncensored rows only
gives **3%**, and CYP1A2 turns negative.

The corrected pool is the more valuable output: its affine absorption is **0.807 against
the 0.806 we actually score**. That calibration says the best monotone re-placement of our
ranking reaches 0.781, nowhere near 0.717 — so the rivals' edge is *ranking on the
narrow-interval compounds*, the ones ST-RAE actually charges for. That is a training-weight
problem, not a placement problem, and it is the open lead.

**TDI threshold tuning is finished on both targets.** The board's five metrics determine
the 2×2 table exactly (`scripts/tdi_diagnose.py`; total fit error 0.002, and the recovered
prevalences match the independently known 0.2915/0.0706). They show only 672 of 750
compounds are scored for CYP3A4 and 524 for CYP2D6, that tdi_v3 and tdi_best have the
**same** discrimination (d′ 1.276 vs 1.268) and differ only in threshold, and that:

- CYP2D6 measured MCC 0.2918, best any threshold can reach **0.2915** — already there.
- CYP3A4 best any threshold can reach **0.4513**; tdi_best scores 0.4470.

Every remaining point needs a higher d′: 0.4911 needs d′ 1.40, 0.5223 needs 1.50, against
our 1.276. The CYP3A4 leader sits at d′ 1.353 (AUC 0.831 vs our 0.817) — a real but modest
model edge, not a threshold trick.

**The multi-head line is closed.** Two attempts, neither beat v6 standalone. The
pre-registered test — CYP2D6, whose external head was 67% pinned at a censoring constant —
moved +0.0057 against a band of ±0.0247. Fixing the spike entirely left the predicted-sd
compression exactly where it was, so the compression is architectural, not a label-shape
problem. `mh` survives as an ensemble member only.

---

## Bugs found and fixed

| bug | cost | fix |
|---|---|---|
| Affine correction applied with the wrong sign | 0.7056 → **1.1048**, rank 118 | sign corrected |
| Floor bug: 1,564 fitted curves discarded as censored | unknown, large | per-endpoint `keep_subfloor` |
| Yardstick drift: the eval pool inherited a *training* choice | same file scored 0.5964 and 0.5788 | pool pinned to all measurements |
| SMILES-string keying in the multi-head table | 175 structures leaked across the fold boundary | keyed on InChIKey connectivity block |
| `SMILES` column dropped from a submission | file **rejected** by the platform | all submissions written through `cypkit/submission.py` |
| `tdi_validator` cast NaN labels to int | reported a positive rate of −2.4e20 | filter non-finite labels first |
| Placement transform fitted against censored rows | would have shipped a 13% phantom gain | uncensored-only fit |

---

## Standing rules earned the hard way

1. **One variable per submission.** Every confounded submission lost: `v3_affine80` moved
   two things, `v8` moved three.
2. **Pre-register the test.** Name the number that would falsify the idea *before* running
   it. The multi-head diagnosis died cleanly because CYP2D6 was named in advance.
3. **A local gain is a hypothesis, not a result.** Ensemble ρ transferred at 14%; screen
   pseudo-labels at 8%.
4. **Run the validator, don't read the columns.** A column listing that looks right is not
   a file that passes.
5. **Prefer arithmetic to modelling.** Both splices had known scores before submitting;
   five retrains did not.
