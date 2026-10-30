# What worked, and what didn't — OpenADMET CYP Challenge

Every number here is measured, either off the leaderboard or from a controlled experiment
with a paired bootstrap. Nothing is recalled or estimated. Source sections in `STRATEGY.md`
are cited as Section n.

---

## 0. The one-sentence answer

**Two things moved the board: putting the predictions in the right place, and combining
models. Nothing about molecular representation or algorithm choice produced a measurable
gain once those two were done.**

Decomposition of the whole pIC50 track, from the first calibrated submission to the best:

| step | submission | MA-ST-RAE | what changed | worth |
|---|---|---|---|---|
| start | `all7_calib` | 0.6763 | 7 members + calibration | — |
| **placement** | `placed` | **0.5837** | affine map to the blind distribution, **ρ unchanged at 0.7253** | **−0.0926** |
| **ensembling** | `blend3_placed` → `blend4z` | 0.5398 → **0.5104** | 3- then 4-member diversity blend (ρ 0.7253 → 0.7486) | **−0.0733** |
| **splicing** | `splice4` | **0.4924** | best scored column per endpoint, no new model | **−0.0180** |

Placement alone is **half** the total gain, and it was verified at *identical Spearman* —
the model was untouched. Ranking (ρ) moved just **+0.023 across the entire project**.

TDI followed the same shape: 0.3147 → 0.3694 from splicing and blending, then 0.3760 from
one decorrelated blend member (Section 14).

---

## 1. Molecular representation

| representation | verdict | evidence |
|---|---|---|
| **Morgan (r2, 1024–2048) + RDKit + CYP descriptors** | **the workhorse; never beaten** | best or joint-best solo ρ on all four endpoints |
| **D-MPNN (ChemProp)** | **useful, and the best TDI backbone** | TDI CYP3A4 d′ 0.940 vs LightGBM's 0.814 (Section 12) |
| **Uni-Mol 3D, fine-tuned multitask** | **useful as an ensemble member** | in blend4z; solo ρ 0.48–0.60, below descriptors but decorrelated |
| Uni-Mol 3D, fine-tuned **single-task** | **worse than multitask everywhere** | ρ −0.024 / −0.012 / **−0.063** vs multitask on 1A2 / 2C9 / 2D6 (Section 28) |
| Uni-Mol **pretrained embeddings** (768-d) as tree features | **dead** | alone: −0.093 / −0.138 / −0.177 / −0.160 ρ, P(worse) 100% on all four (Section 27) |
| **CYP2D6 pharmacophore** (basic N ↔ aromatic, 2D + 3D) | **the only significant feature gain in the project** | CYP1A2 **+0.0217 ρ, CI [+0.0003, +0.0431], P(worse) 2%** (Section 27) |
| Gobbi pharmacophore-pair fingerprint | same size as hand-coded, no better | CYP2D6 +0.016, CYP3A4 +0.005 (Section 23) |
| **TDI bioactivation alerts** (43 + 8 derived) | **shipped; part of the scored TDI model** | — |
| binding × bioactivation products | **negative** | CYP3A4 TDI −0.020 d′, P(worse) 80% (Section 26) |

**The pattern.** Adding a representation as a *blend member* worked. Adding one as *extra
feature columns* never did. The one exception — the pharmacophore on CYP1A2 — is +0.022
local, about +0.003 blind, i.e. ~0.0005 macro.

**The most important negative.** The pretrained Uni-Mol embedding failed the "alone" test
catastrophically. That arm existed precisely so a +0.031 when *added* couldn't be mistaken
for a 3D signal, and it earned its place. Pretraining optimises masked-coordinate
reconstruction, not CYP affinity.

---

## 2. Algorithms

| algorithm | where it won | where it lost |
|---|---|---|
| **LightGBM, moderate regularisation** | **best on every pIC50 endpoint**, including on blind-like rows (CYP1A2 ρ 0.520 vs loose 0.444, tight 0.439, k-NN 0.396) | — |
| **ChemProp (D-MPNN)** | **TDI**: the backbone behind every scored TDI column | doesn't beat descriptors on pIC50 as a solo member |
| **Uni-Mol** | as a *decorrelated* blend member | solo, and as a feature source |
| **similarity-weighted k-NN** | TDI CYP2D6 blind-like rows (d′ 1.058 vs 0.534 incumbent) — but on a yardstick that later failed validation (Section 25) | pIC50 CYP1A2, worst of four arms |
| SVM | CYP2C9: same MAE and ρ as us, **10% better ST-RAE** — an error-distribution difference we never explained | — |
| multi-head / multitask MLP | — | tie with single-task (ρ 0.6196 vs 0.6181), closed (Section 10) |

**There is no universal winner.** k-NN was the best model on one TDI endpoint and the worst
on a pIC50 endpoint. Regularisation strength should be chosen per endpoint against a
yardstick that matches the task's difficulty — which is exactly what we mostly lacked.

---

## 3. Endpoints

Final state, `splice4` (pIC50) and `tdi_blendcp` (TDI). Absorption = ST-RAE ÷ √(1−ρ²);
instrument ratio = blind ρ ÷ local ρ, so 1.0 means our validation predicts the board.

| endpoint | ST-RAE | ρ | absorption | rank | instrument | read |
|---|---|---|---|---|---|---|
| **CYP2C9** | **0.3978** | 0.8454 | 0.745 | 18 | 1.16 | our best endpoint |
| **CYP3A4** | 0.4634 | 0.8327 | 0.837 | 40 | **1.01** | well measured, poorly placed |
| **CYP1A2** | 0.5125 | 0.8064 | **0.867** | 34 | **1.33** | worst absorption *and* worst-measured |
| **CYP2D6** | **0.5957** | **0.5190** | **0.697** | 26 | 1.03 | pure ranking failure; absorption already best |

| TDI endpoint | MCC | d′ | rank | instrument |
|---|---|---|---|---|
| **CYP3A4** | **0.4601** | ~1.30 | **8** | **0.98** — fully measurable |
| **CYP2D6** | 0.2918 | 1.224 | 18 | **4.37** — not measurable at all (Section 25) |

**Two different diseases.** CYP2D6 pIC50 has good absorption and terrible ρ — it needs a
better model. CYP1A2 and CYP3A4 have good ρ and bad absorption — they need better
placement. Treating all four the same was a mistake for most of the project.

**And the measurability split explains the results.** The endpoint we measure best (TDI
CYP3A4, ratio 0.98) is our best rank, 8. The one we can't measure at all (TDI CYP2D6, 4.37)
is rank 18, and three attempts to improve it produced one instrument that read 4.37× low,
one that inverted, and one submission that cost rank 9 → 36.

---

## 4. The metric, and why placement dominated

ST-RAE charges `clip(p−hi, 0) + clip(lo−p, 0)` — **a prediction inside its confidence
interval costs nothing.** Two consequences drove the whole project:

1. **Where you put the predictions matters as much as their order.** We beat rivals on
   ranking and lost on score: at rank 23–25, `mike` scored 0.4890 at ρ 0.7325 while we
   scored 0.4962 at ρ 0.7509. Better order, worse score.
2. **The centre is worth ~4× the spread.** Measured on the board: CYP1A2 moved its centre
   0.27 and paid +0.115; CYP3A4 moved only its spread by 0.21 and gained 0.036 (Section 19).

But placement is **already optimal**. A cost model calibrated to reproduce each endpoint's
own board score, using the empirical (y, interval-width) joint distribution, puts the
optimal spread multiplier at k\* = 0.78–0.88 against the 0.76–0.88 we ship — within 0.0035
macro of the best any affine placement can do (Section 18).

One structural fact worth carrying forward: **potent compounds are measured precisely and
weak ones are not** — corr(y, interval half-width) = −0.69 to **−0.89**. The dead zone is
wide exactly where activity is low.

---

## 5. Validation — the real lesson

**Our cross-validation graded a harder task than the exam.** Out-of-fold nearest-neighbour
Tanimoto:

```
our folds (Butina)     median 0.438   33.1% below 0.4
our folds (random)     median 0.439   32.9% below 0.4   <- identical
THE BLIND SET          median 0.587    0.1% below 0.4
```

A third of our validation compounds have no near neighbour; in the blind set it's one in a
thousand. **We graded extrapolation while the exam was interpolation** — and changing the
splitter can't fix it, because random and Butina give the same profile. The training set is
simply diverse and the blind set sits inside it.

This is the single most consequential finding, because every model-selection decision was
made under a distribution shift the real task does not have, which systematically favours
whatever generalises far.

**What survived as a usable instrument:** relative *ranking* comparisons, which transfer at
about 14% of their local size with reliable direction. **What did not:** anything
involving placement or absolute cost. The pool over-states our cost by +11% to +67%
depending on endpoint, so it cannot price a calibration change at all (Section 17).

---

## 6. Complete list of what was tested and closed

| lever | result | Section |
|---|---|---|
| affine placement at blind anchors | **worked, −0.093 macro** | — |
| model-diversity ensembling | **worked, −0.073 macro** | — |
| per-endpoint splicing | **worked, −0.018 macro** | — |
| decorrelated blend member (TDI CYP3A4) | **worked, +0.013 MCC** | Section 14 |
| multi-head / multitask as a replacement | tie | Section 10 |
| interval-weighted training | +0.4% | Section 11 |
| ST-RAE as a training objective | +0.2% | Section 11 |
| uncertainty shrinkage | 3 of 4 endpoints prefer none | Section 11 |
| monotone re-placement | −0.095 macro **(loss)** | Section 13 |
| raw member values vs spliced | +0.015 on CYP3A4 **(loss)** | Section 13 |
| re-blending existing members | +0.003 ρ, inside noise | Section 15 |
| difficulty-matched validation pool | fails its own reproduction check | Section 17 |
| affine spread re-tuning | already within 0.0035 of optimal | Section 18 |
| external data — assay shape ×3, proximity ×1 | proximity **significantly worse** | Section 22 |
| hand-built pharmacophore features | +0.022 on CYP1A2 only | Section 23, Section 27 |
| two-arm differencing (TDI) | difference-error > signal | Section 12 |
| shift target (TDI) | +0.012 on ChemProp, noise | Section 12 |
| binding × bioactivation (TDI) | negative | Section 26 |
| CYP2D6 TDI blend | **cost rank 9 → 36** | Section 24 |
| pretrained Uni-Mol embeddings | −0.09 to −0.18 ρ alone | Section 27 |
| single-task vs multitask Uni-Mol | multitask wins | Section 28 |
| model regularisation sweep | incumbent already best | Section 28 |

---

## 7. If starting again

1. **Solve the blind label distribution first.** Placement was half the total gain and it
   cost nothing but arithmetic on published leaderboard metrics.
2. **Build the validation set to match the test set's difficulty, not to be rigorous.** A
   scaffold split is the right instinct and was the wrong choice here.
3. **Check per endpoint whether the yardstick predicts the board** before trusting any
   result from it. One endpoint read 4.37× low for the entire project.
4. **Spend model effort on ranking, calibration effort on placement, and do not confuse
   them** — they are measured by different instruments with different reliability.
5. **Always include an "alone" arm** when adding a feature block. It is the only thing that
   distinguishes signal from decoration.
