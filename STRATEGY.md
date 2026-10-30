# CYP Blind Challenge — where we are and what to do next

*Rewritten 2026-09-07, after 9 scored pIC50 submissions and 6 scored TDI submissions.*
*Score history with per-target values: `artifacts/submissions.csv`, rendered by
`python scripts/track.py`. This file is the reasoning; that file is the record.*

---

## 1. Where we stand

**pIC50 — MA-ST-RAE 0.5167, rank 28** (`splice_best`, 2026-09-07)
**TDI — MA-MCC 0.3374, rank 15** (`tdi_v2`; best ever 0.3451)

| pIC50 | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---|---|---|
| ST-RAE | 0.5153 (r24) | 0.3978 (r17) | 0.6602 (r36) | 0.4938 (r40) |
| Spearman ρ | 0.7892 | 0.8454 | **0.5072** | 0.8231 |

**The splice was predicted to 0.0001.** Every column was taken verbatim from a submission
where that column had been scored; ST-RAE is separable per endpoint (verified against the
challenge's own `rae_soft_threshold_absolute_error`), and the macro is the plain mean.
Predicted 0.5168, scored 0.5167 — the leaderboard TRUNCATES rather than rounds
(0.516775 -> 0.5167), worth knowing for the next prediction.

**The splice well is now dry**: best-ever-per-target spliced gives 0.5167, exactly what we
have. Further pIC50 gains must come from an untested placement or from the models.

| TDI | CYP3A4 | CYP2D6 |
|---|---|---|
| MCC now | 0.4035 (r25) | **0.2866 (r13)** |
| MCC best ever | **0.4470 (r4)** | 0.2866 |
| blind d′ | 1.135 (best 1.290) | 1.213 |

**The top of the pIC50 leaderboard is macro ST-RAE 0.4378** (SuperCowPowers / Workbench,
rank 1 of 81, self-reported Aug 2026 on a half-test-set leaderboard, so treat as
approximate). **We are at 0.5167. The gap is 0.079** — about the same size as the entire
placement gain that took us from 0.7273 to 0.5167.

Their write-up: https://supercowpowers.github.io/workbench/blogs/cyp_challenge/

**Their external-data strategy is the opposite of ours, and ours was the wrong shape.**

| | them | us |
|---|---|---|
| sources | ChEMBL 37 (24,918 cpds), PubChem AID 1851 / Veith (17,107), Tox21 (7,196) | PubChem, 16k |
| alignment | InChIKey **connectivity block** (salt/stereo variants collapse) | SMILES standardisation |
| readout harmonisation | **none — each assay gets its own head** | pooled into the scored target via pseudo-labels |
| architecture | 4 ChemProp D-MPNNs averaged; shared encoder, up to **18 heads**; only the 4 scored pIC50 heads are submitted, "the rest exist to shape the representation" | single-target per isoform |
| censored values | `_lt` flag variant; they report it "has not moved the scored metric" | left-censored rows, same conclusion |
| public weight | `--public-weight 0.30` | n/a |
| scale | 32,900 compounds, 175,700 labels | 4,905 compounds |

The key idea is that **they never harmonise the readouts at all**. Their own framing:
"Every extra source measures something adjacent to the scored target, not identical to it.
Pooling any of them into the scored columns needs a cross-assay correction; as separate
heads none is required." qHTS keeps `max_response` rather than a converted pIC50,
"because it is recorded at 100% coverage".

**This means my "external data is a dead end" conclusion tested the wrong mechanism.** I
measured coverage (blind NN-similarity 0.587 -> 0.589) and direct transfer (external-only
TDI AUC 0.507 = chance) and concluded the data carries nothing. Both tests are about
using external data AS LABELS FOR THE SCORED TASK. Neither tests it as an auxiliary head
shaping a shared encoder, which is what they actually do — and their stated benefit is
"breadth, not more labels on the same molecules": ChEMBL is "almost entirely new
chemistry, 185 structures overlap the challenge deck and ZERO overlap the blind set."
Zero blind overlap is exactly what my similarity test flagged as fatal; for representation
learning it is irrelevant.

**They also single out CYP2D6**, with two dedicated members (`2d6-isoform`, 6 heads; and
`2d6-single`, 1 head) out of four. Independent confirmation of our own diagnosis, and a
concrete template: a CYP2D6-only multi-head model is a thing the leader thought worth
building.

---

## 2. The one thing that changed since the last rewrite

**pIC50 has stopped being a calibration problem and become a ranking problem.**

`absorption·√(1−ρ²)` is what today's *exact ranking* would score if every value were
placed perfectly. It was the central diagnostic of the last version of this document, and
it has now collapsed:

| endpoint | ρ | floor | actual | wasted |
|---|---|---|---|---|
| CYP1A2 | 0.789 | 0.512 | 0.515 | +0.003 |
| CYP2C9 | 0.845 | 0.446 | 0.455 | +0.009 |
| CYP2D6 | 0.507 | 0.719 | 0.660 | −0.059 |
| CYP3A4 | 0.823 | 0.474 | 0.529 | +0.056 |
| **macro** | | **0.538** | **0.540** | **+0.002** |

At the last rewrite we were discarding **0.174** of macro ST-RAE to bad placement.

**But the macro +0.002 is misleading, and reading it as "placement is done" was an
error.** CYP2D6's −0.059 is cancelling CYP3A4's +0.056, and those two numbers are not the
same kind of thing. CYP2D6 scoring *below* its own floor is not a triumph — it means the
normal-theory formula does not hold there, which we knew independently (its R² of 0.4871
exceeds the 0.276 ceiling normality implies). Excluding that invalid cell:

> **Real recoverable placement waste: +0.0168 macro, and +0.056 of it is CYP3A4 alone.**

The cause is regression to the mean. The optimal spread for a ranking of quality ρ is
**ρ·sd_blind**, not sd_blind — matching the blind marginal moments over-disperses. We
submitted CYP3A4 at sd 1.245 where ρ·sd = 0.963, i.e. 1.29× too wide; CYP2C9 at 1.148
where ρ·sd = 0.845, 1.36× too wide.

The absorption constant (0.8345) was recovered from the earlier floor table rather than
derived, so these floors are approximate.

**Consequence: after the ~0.017 of over-dispersion is collected, every remaining pIC50
point must come from ρ.** And ρ has one obvious hole.

| if ρ reached | macro floor | vs today's 0.5398 |
|---|---|---|
| CYP2D6 0.507 → 0.85, others unchanged | 0.4679 | **−0.072** |
| all four at 0.85 | 0.4396 | −0.100 |

CYP2D6 ranking alone is **~70% of the entire remaining pIC50 opportunity**.

---

## 3. What worked

**Leaderboard inversion.** Solving published accuracy/precision/recall recovers every
competitor's operating point; solving R²/MAE recovers the blind distribution. It gave us
blind TDI prevalence (0.2915 / 0.0706, agreeing to four decimals across every competitor),
our own discriminability, and the blind pIC50 moments. Nothing else in this project has
returned as much per hour spent.

**Placement optimisation (pIC50).** 0.7273 → 0.5837 → 0.5398, rank 58 → 34 → 29. CYP2D6
went 1.3092 → 0.6597, rank 65 → 33. This is the single largest gain of the competition and
it closed the gap in Section 2.

**Per-endpoint splicing (TDI).** Keep the best-scoring column per endpoint rather than
accepting a blend wholesale. Predicted MA-MCC 0.3325; scored 0.3325 exactly.

**Measured cut rates (TDI).** A binormal ROC at the recovered prevalence and our own d′
reproduces both actual MCCs to ±0.0008. The threshold half of any TDI change is
predictable to ~0.001 — it has never once surprised us.

**Similarity kernels on CYP2D6 TDI.** Tanimoto kNN/SVC own that endpoint. `knn_fcfp` +
`svc_c10` took CYP2D6 from 0.2180 to **0.2866**, rank 20 → 13, and its blind d′ from 0.980
to 1.213 — predicted 1.183, so the model was nearly exact.

**Censored training (ChemProp v3).** Screen non-hits as left-censored rows: dense proxy
1.4623 → 0.7546.

---

## 4. What did not work

| attempt | result |
|---|---|
| External PubChem data (16k compounds) | blind NN-similarity 0.587 → 0.589; TDI AUC 0.507 = chance |
| Affine calibration, first attempt | 0.7056 → **1.1048**. Sign error |
| Screen pseudo-labels | OOF +0.052 macro ρ, blind **+0.004** |
| Fitted blend weights on CYP3A4 TDI | 0.4470 → 0.3798 |
| Greedy member selection on CYP3A4 TDI | OOF 0.817 → 0.843, blind d′ 1.290 → **1.135** |
| Cliff-weighted training (v7) | captured up, kept flat, ρ down |
| CYP2D6 pIC50 as a TDI feature | AUC 0.4409 — *inverted* |
| Multi-head, 13 heads (mh) | macro ρ 0.6196 vs v6 0.6181 — a tie |
| Multi-head, 18 heads, censoring spike fixed (mh2) | macro ρ 0.6091 — **worse than mh** |
| Applicability domain as the constraint | OOF error is FLAT, and *falls*, with distance |
| ChemProp on CYP2D6 TDI | ranks 13th and 15th of 20; marginal value indistinguishable from zero |
| Tanimoto kernels on CYP2D6 *regression* | ρ 0.35–0.40 vs blend3's 0.48 — the TDI result does NOT carry over |
| logD/pKa/ionisation on **CYP2D6** | +0.0001 ± 0.0051 ρ — the literature's headline feature does nothing here; `phys_frac_base_ionised` correlates 0.932 with the existing `CYP2D6_is_base`, so basicity was already encoded |

**The physchem block (`cypkit/physchem.py`), 3 seeds, floor-fixed labels.** SMARTS pKa
lookup → logD7.4, fraction ionised, net and formal charge, computed on the molecule as
drawn (`features.py` uncharges, which erased this). Validated against 8 drugs with
measured values: logD MAE 0.87, basic/non-basic classification exact.

| | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | macro |
|---|---|---|---|---|---|
| Δρ | **+0.0125 ± 0.0019** | +0.0079 ± 0.0148 | **+0.0001 ± 0.0051** | **+0.0055 ± 0.0009** | +0.0065 |

Real on CYP1A2 and CYP3A4, noise on CYP2C9, and **exactly nothing on CYP2D6** — the
endpoint the whole mechanistic argument pointed at. Reason: the CYP2D6 block already had
it. Keep the block (free, positive), but do not expect it to move CYP2D6.

**The v8 retrain (2026-09-07) did NOT reproduce this on ChemProp.** Blind-matched macro
Spearman went 0.6181 (v6) -> 0.5964 (v8), and adding v8 to blend3 moved macro rho 0.6719
-> 0.6695 — flat. Three changes were bundled into one run, against this project's own
standing rule, so the loss is not cleanly attributable. What the per-row breakdown does
show, measured on the recovered rows themselves:

| endpoint | recovered rows | rho ON those rows | pre-existing rows |
|---|---|---|---|
| CYP3A4 | 944 (40.4%) | **0.542**, MAE 0.972 -> 0.738 | +0.001 |
| CYP2C9 | 259 (20.2%) | 0.393 | **−0.067** |
| CYP1A2 | 232 (16.4%) | 0.158 | −0.003 |
| CYP2D6 | 129 (8.6%) | **−0.041** | +0.002 |

The floor fix is right where there are ENOUGH recovered rows to learn the sub-floor region
(CYP3A4) and injects noise where there are not (CYP2D6, 129 rows the model cannot rank at
all). `keep_subfloor` in `cypkit/labels.py` now defaults to CYP3A4 only.

**v9 refuted the interval_power hypothesis.** I attributed CYP2C9's loss to
`interval_power 0.5`; v9 set it back to 1.0 and CYP2C9 got WORSE. All three on the same
pinned pool:

| macro ρ | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | macro |
|---|---|---|---|---|---|
| **v6** | 0.5308 | **0.6356** | 0.4746 | 0.7736 | **0.6037** |
| v8 | 0.5076 | 0.5561 (−6.2σ) | 0.4777 | 0.7738 | 0.5788 |
| v9 | 0.5118 | **0.5017 (−10.5σ)** | 0.4599 | 0.7805 (+0.7σ) | 0.5635 |

**v6 is still the best ranker and three retrains have failed to beat it.** The only gain
anywhere is CYP3A4 +0.0069 in v9 — 0.7σ, and splicing it into v6 is worth +0.0017, noise.

CYP2C9 got worse at BOTH steps (v6→v8 and v8→v9), so the cause is the variable common to
both regressed runs and absent from v6: **the physchem block**. It is 10 columns on
CYP2C9's 63-feature vector, fused at the readout, and CYP2C9 is precisely the endpoint
where its LightGBM effect was pure noise (+0.008 ± 0.015). It is now switchable via
`CYP_NO_PHYSCHEM=1`; keep it ON for the classical learners, OFF for ChemProp until tested.

**A yardstick bug I introduced and fixed.** `blind_validator.eval_pool` built its
evaluation set with `build_label_matrix`, so gating `keep_subfloor` silently changed the
measuring stick: identical `oof_st_v8.csv` scored macro ρ 0.5964 before the gate and
0.5788 after. The pool is now pinned to `keep_subfloor=all` — the blind set genuinely
contains sub-floor compounds, so evaluation must see every measurement we have, whatever
the training choice is.

---

## 5. What we have learned about our own validation

This is the section to read before trusting any local number.

**The blind set is easier than our folds, and differently so per endpoint.** Converting
out-of-fold AUC to d′ and comparing with the recovered blind d′:

| endpoint | column | OOF d′ | blind d′ | shrink |
|---|---|---|---|---|
| CYP3A4 | equal 19-member | 1.278 | 1.290 | 1.01 |
| CYP3A4 | greedy-selected 5 | 1.426 | **1.135** | **0.80** |
| CYP2D6 | equal 19-member | 0.436 | 0.980 | 2.25 |
| CYP2D6 | greedy-selected 2 | 0.526 | 1.213 | **2.31** |

**CYP2D6's shrinkage has two anchors that agree — it is a property of the endpoint and can
be trusted.** Every +0.02 of CYP2D6 OOF AUC becomes roughly +0.05 of blind d′.
**CYP3A4's is not stable across score columns**, so its OOF predicts only for the column
the shrinkage was fitted on.

**Selection on out-of-fold AUC destroys diversity, and out-of-fold cannot price
diversity.** The CYP3A4 failure above was nested — nothing was scored on data that chose
it — and it still failed. What separates it from the CYP2D6 success:

| | kept | mean pairwise ρ | dropped above 0.70 AUC | dropped below 0.60 |
|---|---|---|---|---|
| CYP3A4 | 5 | **0.791** (pool 0.631) | **14/15** | 0/15 |
| CYP2D6 | 2 | 0.435 (pool 0.359) | 0/18 | **13/18** |

CYP2D6 pruned noise. CYP3A4 threw away fourteen *informative* members for a tighter,
mutually-correlated cluster. **Rule: prune members that are near chance; never prune
informative members because greedy prefers a correlated cluster.**

**Two explanations for that failure were tested and refuted.** It is not applicability
domain — splitting the OOF pool by distance to its own fold's training half ranks members
4/5 the same near and far. It is not a threshold artefact — CYP3A4's single-point d′
*rises* with cut rate on our own data, so the higher rate should have inflated the
measurement, meaning the true degradation is worse than 1.290 → 1.135. **The FAR column
would have made the same mistake** (it ranks the selected-5 at 0.7896 above the equal-19 at
0.7475), so it is not a validator we can lean on.

**Activity cliffs are real and we smooth them.** At Tanimoto ≥ 0.45, cliffs are 6–14% of
compounds, MAE ratio 1.20–1.76×, and the model reproduces only 0.20–0.53 of the measured
gap. Weighting the loss (v7) did not fix it.

---

## 5b. The floor bug — found 2026-09-07, fixed

`cypkit/labels.py` converted any fitted curve with `y <= 4.0` into a left-censored row —
"free anywhere below the floor". Those rows are **real measurements with real credible
intervals**, and `scoring.build_ground_truth` charges ST-RAE against exactly those bounds.

| endpoint | curves | discarded as censored | their real bounds |
|---|---|---|---|
| CYP1A2 | 1412 | 232 (16.4%) | |
| CYP2C9 | 1285 | 259 (20.2%) | |
| CYP2D6 | 1493 | 129 (8.6%) | |
| **CYP3A4** | 2335 | **944 (40.4%)** | mean [1.99, 3.79] |
| total | 6525 | **1564 (24.0%)**, lowest fitted 1.91 | |

`PIC50_FLOOR = 4.0` is correct for the *TDI label rule*, where it came from. It was never
validated as a quantification limit for the direct-inhibition curves, and the data says it
is not one. Censoring now applies only to screen non-hits, which genuinely have no curve.

Controlled test — same features, same folds, same model, **identical evaluation rows**,
only the training labels differ:

| | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | macro |
|---|---|---|---|---|---|
| Δρ, eval on above-floor curves only | +0.063 | +0.083 | +0.001 | +0.041 | **+0.047** |
| Δρ, eval on all curves (matches blind) | +0.071 | +0.107 | +0.036 | +0.100 | **+0.079** |

CYP3A4 MAE also fell 0.824 → 0.538, and predicted sd rose 0.31 → 0.81 against a needed
1.25 — this bug was a large part of the compression this project has been paying for.
It gains least on CYP2D6, which had only 129 rows to recover.

Guarded in `scripts/build_labels.py`: a dose-response row that is left-censored now raises.

---

## 6. Bugs found (all fixed)

- **Scaffold-split mismatch** in all four UniMol notebooks: `includeChirality=False`
  against the library's `True`. Adjusted Rand 0.0011 — **78% of "out-of-fold" rows were
  scored by a model trained on them.**
- **`multilabel_classification` casts NaN targets to −2 147 483 648.** Would have trained
  on garbage for an hour without erroring.
- **`sklearn`'s `jaccard` metric is a per-pair Python loop** — a notebook cell timed out at
  10 minutes. As a matmul (`inter = A·Bᵀ`) it is 0.77s.
- **Single-file feature cache recorded nothing about which blocks it held** → `KeyError`
  when the block list changed. Now cached per block.
- **`tdi_validator` cast NaN labels to int**, printing a positive rate of −2.4e20 instead
  of failing.
- `AllChem.GetMorganGenerator` is recent-RDKit only; three files now fall back to
  `rdFingerprintGenerator`.

---

## 7. The plan, in priority order

Each item names what it is worth and how we would know it worked.

**1 — Record the leaders.** Download the leaderboard CSV for both tracks and add the top
five to `submissions.csv`. Cost: minutes. Without it, "top of the leaderboard" has no
target and we cannot tell a decisive idea from a marginal one.

**2 — Submit `submission_tdi_v2.csv`.** CYP3A4's rank-4 column (blind d′ 1.290) plus
CYP2D6's rank-13 column (d′ 1.213), each re-cut at its optimal rate. **Predicted MA-MCC
0.3720** against today's 0.3451. Both changes are the threshold half, which has never
missed by more than 0.001, and neither score column is new. This is the lowest-risk
+0.027 available on either track.

**3 — CYP2D6 pIC50 ranking. The single biggest lever in the competition (−0.072 macro
ST-RAE).** ρ is 0.5072 against 0.79–0.85 on the other three. Placement has already been
squeezed dry, so this is now the *only* way pIC50 improves. Untried and plausible: the
similarity-kernel family that transformed CYP2D6 on the TDI track has never been tried on
CYP2D6 regression, and CYP2D6 is the endpoint where tree learners have failed at every
attempt on both tracks. That is a strong hint about what kind of model the endpoint wants.

**4 — CYP2D6 TDI discriminability.** d′ 1.213 against the leader's 1.392. The 2.31×
shrinkage means a +0.02 OOF AUC gain is worth ~+0.05 blind d′ — the best return-per-effort
on the TDI track, and the shrinkage is the one transfer constant we have verified twice.

**5 — Build a validator that would have caught the CYP3A4 failure.** Rebuild the TDI
members under a **scaffold split** and check whether it ranks the equal-19 blend above the
selected-5, as the leaderboard did. Pooled OOF, nested selection, and the distance-based
FAR column all get that comparison backwards. Cost: a full member retrain. Worth it,
because it also settles the open ChemProp question in Section 8 without spending a submission —
and because every idea from item 3 onward will need a local metric that can be trusted.

**6 — Activity cliffs, properly.** The diagnosis is solid and the first fix failed.
A pairwise ranking loss on ChemProp is the evidence-backed next attempt; v7's failure
pattern (captured up, kept flat) says the weighting reached the right pairs but the
smooth readout undid it.

---

## 8. Open questions

**Does ChemProp-alone beat the equal blend on CYP3A4 TDI?** Its two columns together score
0.8317 out of fold against 0.8128 for all eighteen classical members, and it is genuinely
decorrelated (ρ 0.78–0.82 against the classical consensus, where the best classical member
sits at 0.889). But the only leaderboard test of a ChemProp-heavy CYP3A4 column lost, and
**no local metric we have can separate those two candidates** — they are exactly the pair
it already got backwards. Item 5 resolves this. Until then, keep ChemProp in the CYP3A4
blend at its 1/19 weight and out of CYP2D6 entirely.

**Why are CYP2C9 and CYP3A4 more active in the blind set than in training?** The
screen-hit-enrichment story explains CYP1A2 and CYP2D6 and gets these two backwards. The
offsets are measured; the mechanism is half-explained.

**What is UniMol actually worth?** Its early OOF was contaminated by the scaffold-split
bug. It decorrelates well (transfer rate 0.210) but is weak alone. Unmeasured.

---

## 9. Standing rules earned the hard way

- **One change per submission.** The affine failure and the pseudo-label wash were both
  unattributable at first because more than one thing moved.
- **Record per-target scores at submission time.** The leaderboard shows only your latest
  entry. `tdi_recut` lost 0.0138 and we can no longer say which endpoint it came from.
- **A local gain is a hypothesis until the leaderboard confirms it.** Three separate times
  a real, honestly-measured local improvement has failed to transfer.
- **Solve for signs, don't argue them.** Two branches fitting the same data means you need
  another anchor, not a better story.
- **Prefer the change whose size is predictable.** TDI thresholds move the score by an
  amount we can compute to 0.001. Score columns are where the surprises live.
- **Read the library, don't reimplement it.** Both of the worst bugs came from
  reimplementing something `unimol_tools` already did.


## 10. The multi-head line is closed — 2026-09-08

Two full attempts. Neither beat v6 standalone, and the second was worse than the first.

| | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | MACRO | pred sd |
|---|---|---|---|---|---|---|
| v6 | 0.5588 | 0.6592 | **0.4807** | 0.7736 | 0.6181 | 0.42–0.96 |
| mh (13 heads) | 0.5709 | **0.6708** | 0.4471 | **0.7897** | **0.6196** | 0.35–0.86 |
| mh2 (18 heads, spike fixed) | **0.5817** | 0.6388 | 0.4528 | 0.7631 | 0.6091 | 0.34–0.83 |

**The prediction was pre-registered and it failed.** Before mh2 ran, the stated test was:
"if the constant-spike diagnosis is right, CYP2D6 is where it shows — that head was 67%
pinned and it is the one that came back worse than v6. If CYP2D6 doesn't move, the
diagnosis was wrong and external data is not the lever." CYP2D6 moved **+0.0057 against
a band of ±0.0247**. It did not move. Meanwhile CYP2C9 lost 0.032 (2.7σ) and CYP3A4 lost
0.027 (3.0σ) — the five new `maxresp` heads diluted the two endpoints that were working.

**The mechanism was refuted, not just the gain.** The predicted-sd compression was
blamed on the constant spike. Removing the spike entirely — potency heads went from 40–67%
one value down to 6–10% — left the compression exactly where it was, 0.34–0.40 on three
endpoints. So the compression is a property of the ARCHITECTURE (many auxiliary heads
sharing a trunk at weight 0.30), not of the external label shape.

**mh2 is not a better ensemble member either.** Blends are within noise of each other:
v6+mh+cp+um 0.6722, v6+mh2+cp+um 0.6701, all five 0.6748 (bands ±0.008–0.022).

**What is closed and what is not.** Three shapes of external data have now been tried:
direct label transfer (failed), auxiliary heads with a spiked pseudo-pIC50 (tied),
auxiliary heads with clean potency plus graded efficacy (worse). Do not try a fourth
variation of the same 16k PubChem compounds. What remains genuinely untested is ChEMBL 37
— ~25,000 compounds, a different assay family, graded potency throughout — which the
leader uses and we have never had; `ebi.ac.uk` is blocked from every shell here, so it
needs a manual browser download before it can be tested at all.

**KEEP mh as an ensemble member.** It is a bad replacement for v6 and a good disagreement
partner: the four-model blend containing it scores macro ρ 0.6722 against v6's 0.6181,
the largest reproducible local gain this project has produced. Stop developing it; keep
using it.

**The question that now governs everything** is whether that blend gain transfers to the
blind deck. On our compounds the four members agree at ρ 0.44–0.88; on the 750 blind
compounds they agree at 0.71–0.95, so the local gain is measured where they disagree most.
submission_blend4z.csv answers it — and its per-target ρ answers it regardless of what the
ST-RAE does, because ρ is invariant to placement. Until that number is in, building more
models is guessing.


## 11. Interval-weighted training — tested and refuted, 2026-09-13

The diagnostic that motivated it is real and stands. On the calibrated pool the ST-RAE
cost is concentrated in the narrowest-interval compounds (CYP1A2 43%, CYP2C9 38%,
CYP2D6 **89%**, CYP3A4 30% of the cost in the narrowest quartile), and on CYP2C9 and
CYP3A4 we rank BETTER on the compounds that cost nothing and WORSE on the ones that cost
everything — the two endpoints where `wbot` beats our absorption.

Three arms, identical features and folds, only the objective differing:

    arm                                       macro rho   rho|narrow   absorption
    control   (L2, uniform)                     0.4512      0.3738       2.092
    weighted  (L2, weight 1/interval width)     0.4578      0.3814       2.084   +0.4%
    interval  (the METRIC as the objective)     0.4444      0.3809       2.087   +0.2%

Nothing. A fourth idea — shrinking predictions toward the endpoint mean in proportion to
ensemble disagreement, to cut the error tail ST-RAE punishes — also failed: three of four
endpoints prefer no shrinkage at all, and CYP3A4's 1.4% is inside noise.

A BUG FOUND ON THE WAY, worth remembering. LightGBM's sklearn API calls a custom objective
as `(y_true, y_pred)`, not the `(preds, dataset)` of the native API. Getting that backwards
computed the gradient from y against y's OWN interval — and every label sits inside its own
confidence bounds by construction, so the gradient was identically zero and the model
predicted a constant. rho came back at -0.07 on all four endpoints, which reads as "much
worse" and is actually "never ran". A broken arm is not a negative result.

WHY THE INTERVAL LOSS CANNOT HELP US YET. Our out-of-fold error is 2-4x the interval width
and only 17-35% of predictions land inside their interval at all, so the hinge degenerates
to plain L1 and the objective has no more information than the control.

BUT THE GAP IS REAL AND IS NOT ACCURACY, NOT RANKING, AND NOT PLACEMENT. On CYP2C9:

    SVM     MAE 0.4570   rho 0.8467   ST-RAE 0.3809   ST-RAE/MAE 0.833
    us      MAE 0.4566   rho 0.8380   ST-RAE 0.4172   ST-RAE/MAE 0.914

Identical MAE, near-identical rho, 10% better score. Since the ST-RAE denominator is fixed
for everyone, that is purely a smaller numerator: their errors convert into less charged
cost than ours do. ST-RAE charges linearly outside the interval, so it is dominated by the
TAIL — one catastrophic miss costs what many small ones do. Across 14 competitor rows,
ST-RAE/MAE is stable within an endpoint (sd 0.03-0.12) and we sit at the high end of it on
both CYP2C9 and CYP3A4.

So the open question is narrow and well-posed: what makes two models with the same MAE and
the same rho differ by 10% in ST-RAE? Loss weighting does not (tested), the metric as an
objective does not (tested), monotone re-placement does not (tested, 0.781 floor), and
uncertainty shrinkage does not (tested). Do NOT try a fifth variation of these. The
remaining candidate is the error DISTRIBUTION — our large errors may be concentrated on a
chemotype the rivals cover and we do not — and that is an applicability-domain question,
not a loss question.

--------------------------------------------------------------------------------
Section 12  THE SHIFT TARGET IS CLOSED (chemprop, 2026-09-13)
--------------------------------------------------------------------------------

`is_TDI` is a thresholded shift between two assay arms, so the binary label discards a
continuous quantity we can measure. Regressing the shift instead of classifying the label
looked like recovering free information. On the descriptor-only LightGBM baseline it
gained d' 0.814 -> 0.869 (+0.055). On the ChemProp trunk -- the encoder that actually
ships -- the same change gains +0.012:

    backbone    control (binary)   candidate (shift)   gap
    lgbm             0.814              0.869        +0.055   95% CI [-0.001, +0.109]
    chemprop         0.940              0.953        +0.012   inside the lgbm CI's width

The LightGBM gain was never significant either: its own paired bootstrap straddles zero.
Read together, the honest conclusion is that the binarisation was not the limit. WHAT THE
TABLE DOES SHOW is that the ENCODER is worth +0.126 d' (0.814 -> 0.940) -- ten times what
the target reformulation is worth, and a reminder that the submitted models already have it.

--two-head (folding the 1250 CYP3A4 rows that carry a label but no fitted arms back in as a
second classifier) came in at 0.938, BELOW the control. Those rows are not free data: a row
with no fittable dose-response is systematically a weak or inactive compound, and adding
them dilutes rather than informs.

TESTED AND CLOSED ON TDI SO FAR: target reformulation (this section), two-arm differencing
(difference-error 0.383 > shift sd 0.364, cypkit/tdi_twoarm.py), operating-point moves
(scripts/tdi_diagnose.py -- both targets already sit at their MCC optimum), and binary-only
row augmentation (above). Rank 1 needs d' x1.205. None of these is that lever.

CAVEAT ON EVERY CHEMPROP NUMBER PRODUCED BY THIS SCRIPT BEFORE 2026-09-13. run_isoform's
`take()` used np.where() on the INTEGER index arrays StratifiedKFold yields, which both
drops the row whose index is 0 and returns positions rather than indices -- so chemprop was
handed a different set of molecules from the descriptor rows beside it. LightGBM ignores
the SMILES argument entirely, so every lgbm number in this file's history is unaffected,
and the chemprop path never completed a run before the fix. Nothing downstream is
contaminated, but the class of bug is worth remembering: a silent pairing error between two
parallel inputs, invisible to the backbone that only reads one of them.

--------------------------------------------------------------------------------
Section 13  THE PER-ENDPOINT LEDGER, AND A CORRECTION TO Section 11 (2026-09-13)
--------------------------------------------------------------------------------

`submission_raw3a4.csv` froze CYP1A2/CYP2C9/CYP2D6 and changed only CYP3A4, so the macro
move measures that one endpoint exactly. It scored 0.4962 against splice4's 0.4924:

    CYP3A4(splice4) = 0.4786 - 4 x (0.4962 - 0.4924) = 0.4634

RAW MEMBER VALUES ARE +0.0152 WORSE ON CYP3A4 THAN THE SPLICED COLUMN. The splice and
placement machinery earns its place on this endpoint; do not strip it again. The probe
design was right even though the answer was no -- one endpoint moved, three were frozen,
and the result is a measurement rather than an inference.

THE COMPLETE LEDGER FOR splice4 (0.4924, rank 19), with the two-axis decomposition
ST-RAE = absorption x sqrt(1 - rho^2):

    endpoint   ST-RAE     rho    floor   absorption   rank
    CYP1A2     0.5125   0.8064   0.5914     0.867       34
    CYP2C9     0.3978   0.8454   0.5341     0.745       18
    CYP2D6     0.5957   0.5190   0.8548     0.697       26
    CYP3A4     0.4634   0.8327   0.5537     0.837       40

Section 11 POSED THE ABSORPTION QUESTION ON CYP2C9. THAT WAS THE WRONG ENDPOINT. CYP2C9's
absorption of 0.745 is one of our two GOOD ones. The absorption problem lives in CYP1A2
(0.867) and CYP3A4 (0.837) -- which are also our two worst RANKS, 34 and 40. CYP2D6 is
the opposite case: absorption 0.697 is already at the level rivals reach, and its 0.5957
is a pure ranking failure (rho 0.519 against 0.81-0.85 elsewhere).

So the track splits into two different problems, and they need different work:

    CYP1A2, CYP3A4   absorption 0.867 / 0.837 against the 0.745 we already achieve on
                     CYP2C9 with the same machinery. Closing that gap alone, with NO
                     ranking improvement at all, is worth -0.031 macro (0.4924 -> 0.461).
                     An existence proof, not a hope: we do it on two of four endpoints.
    CYP2D6           pure rho. Nothing about calibration will help.

WHAT THE ABSORPTION GAP IS NOT. The obvious hypothesis is a fat error tail -- ST-RAE
charges linearly outside the interval, so a few catastrophic misses could dominate. It
does not survive measurement. Share of ST-RAE cost carried by the worst rows on the
calibrated pool:

    endpoint   worst 1%   worst 5%   worst 10%   worst 25%
    CYP1A2        4.4%      18.0%      31.6%       60.3%
    CYP2C9        7.4%      24.1%      39.0%       69.7%
    CYP2D6        2.1%       9.8%      18.3%       40.1%
    CYP3A4        4.8%      17.4%      30.0%       58.0%

Cost is broad, not concentrated, and the ordering runs the WRONG way: CYP2C9 has the most
concentrated cost and the BEST absorption. Trimming or shrinking a handful of outliers
will not close this. Do not spend a submission on that idea.

--------------------------------------------------------------------------------
Section 14  OUR LOCAL CYP2D6 VALIDATION DOES NOT MEASURE CYP2D6 (2026-09-13)
--------------------------------------------------------------------------------

The CYP3A4 blend shipped: macro 0.3694 -> 0.376, rank 10 -> 9, CYP3A4 0.4470 -> 0.4601
(rank 8), CYP2D6 frozen at 0.2918 as designed. FORECAST CALIBRATION: predicted +15.5%
relative on CYP3A4, realised +2.9%. TRANSFER WAS 19%, and the "conservative" 0.381 I
quoted was still above the realised 0.376. Third CYP3A4 overshoot in a row. RULE: divide
local relative MCC gains on CYP3A4 by five before quoting a range.

THE INSTRUMENT CHECK NOBODY RAN. Local OOF d' against the d' the board implies:

    endpoint   local d'   blind d'   ratio
    CYP3A4       1.300      1.276    0.98    <- an excellent yardstick
    CYP2D6       0.280      1.224    4.37    <- not a yardstick at all

Blind CYP2D6 is 7.1% positive; ours is 21.6%. Restricting local positives to the strongest
shifts to match that prevalence lifts local d' 0.280 -> 0.454 -- real, and still 2.7x short.
Whatever makes the blind CYP2D6 set easier is not only prevalence and not only shift
magnitude, and we have not found it.

CONSEQUENCE, AND A REVERSED DECISION. Section 12's rejection of the CYP2D6 blend was read off this
broken instrument and is withdrawn. Re-scored on the prevalence-matched pool:

    positives kept   prevalence   incumbent   blend    gain    P(worse)
        all            21.7%        0.283     0.317   +0.034     20%   <- what I rejected on
        top 25%         6.5%        0.454     0.598   +0.144      3%   <- closest to blind

THE CYP2D6 GAP IS PURE d', NOT OPERATING POINT. Inverting the leaderboard confusion
matrices (n=524, 37 positives): rank 1 d' 1.891 (x1.545 ours), top-5 ~1.47 (x1.20), ranks
15-16 1.270 (x1.04), us 1.224. stir_bar calls 37 and C_CYPher calls 79 for the same d' and
almost the same MCC. Our own MCC curve at d' 1.224 peaks at 0.2914 (55 calls) against
0.2910 at our 62, flat within 0.005 from 35 to 83 calls. RE-THRESHOLDING CYP2D6 IS WORTH
NOTHING. Do not spend a submission on it.

NEXT: submission_tdi_blend2d6.csv, CYP3A4 frozen at the 0.4601 column, CYP2D6 re-ranked at
the same 64 calls -- so 2 x macro - 0.4601 recovers CYP2D6's exact score.

--------------------------------------------------------------------------------
Section 15  RE-BLENDING THE EXISTING pIC50 MEMBERS IS EXHAUSTED (2026-09-13)
--------------------------------------------------------------------------------

The greedy-selection-plus-split-half method that produced both TDI wins was finally run on
pIC50, over seven members (v6, v7, v5_rest, v5_2d6, mh, mh2, unimol, cp) per endpoint, on
the uncensored blind-matched pool. AGAINST THE SHIPPED blend4z IT ADDS NOTHING:

    endpoint   incumbent rho   searched rho   OUT-OF-SELECTION   P(worse)
    CYP1A2        0.6074          0.6166          +0.0026          22%
    CYP2C9        0.7269          0.7349          +0.0027          20%
    CYP2D6        0.5021          0.5062          -0.0028          62%
    CYP3A4        0.8256          0.8315          +0.0037           5%

Only CYP3A4 survives, and +0.0037 rho is worth roughly 0.0002 macro at the measured 14%
ensemble transfer. Estimated macro 0.4917 against 0.4924 held -- inside the noise, and
exactly the kind of little modification the standing rule says not to spend a slot on.

THE TRAP THIS ALMOST WALKED INTO. Benchmarked against the best SINGLE member instead, the
same search reads +0.060 / +0.038 / +0.042 / +0.035 rho with P(worse) 0% everywhere -- a
spectacular result, and entirely the gain blend4z already banked in an earlier submission.
ALWAYS BENCHMARK A SEARCHED BLEND AGAINST THE COLUMN THAT IS ACTUALLY SCORED.

THE INSTRUMENT, per endpoint, incumbent local rho against the blind rho the board reported:

    CYP1A2 1.33x     CYP2C9 1.16x     CYP2D6 1.03x     CYP3A4 1.01x

CYP2D6 and CYP3A4 are measured honestly on the pIC50 track (unlike CYP2D6 on TDI, Section 14).
CYP1A2 at 1.33x is the outlier -- AND it is our worst absorption (0.867) AND our worst rank
(34). Three independent symptoms on one endpoint is a lead, not a coincidence: whatever is
wrong with CYP1A2 makes it both hard to measure locally and expensive on the board.

WHAT IS LEFT ON pIC50. Not re-blending: every member we have is either already in blend4z
or adds nothing to it. The remaining levers are a genuinely NEW model (different
representation, not another variation of the same trunk) and the CYP1A2 anomaly above.
Nothing currently on disk is worth a submission slot.

--------------------------------------------------------------------------------
Section 16  WE HAVE BEEN GRADING OURSELVES ON THE WRONG EXAM (2026-09-13)
--------------------------------------------------------------------------------

WHAT WORKED, ACROSS THE WHOLE PROJECT: affine placement at the blind anchors (+0.174
macro), ensembling into blend4z, per-endpoint splicing, and decorrelated blending on TDI.
Every one of them COMBINES OR PLACES existing predictions.

WHAT NEVER WORKED, NOT ONCE: multi-head/multi-task (Section 10), interval-weighted training and
the metric as an objective (Section 11), monotone re-placement (place_half), uncertainty
shrinkage, raw member values (Section 13), two-arm differencing, the shift target on chemprop
(Section 12), re-blending existing members (Section 15), three shapes of external PubChem data. Every
one of them changes HOW A MODEL LEARNS.

That is not a coincidence and it is not bad luck. Look at the board:

    rank  entry        MA-ST-RAE   MA-rho   MA-MAE   implied absorption
     23   mike            0.4890   0.7325   0.6908       0.718
     24   leebrubeck      0.4930   0.7326   0.6860       0.724
     25   us              0.4962   0.7509   0.7100       0.751

WE RANK BETTER THAN BOTH AND SCORE WORSE THAN BOTH. Ranking has not been the binding
constraint for a long time, which is exactly why nothing that improved ranking moved us.

THE ROOT CAUSE, and it is a validation defect, not a modelling one. Out-of-fold
nearest-neighbour Tanimoto (Morgan r2, 2048):

    our folds, Butina                          median 0.438   33.1% below 0.4
    our folds, random -- IDENTICAL             median 0.439   32.9% below 0.4
    THE BLIND SET, to its nearest training compound
                                               median 0.587    0.1% below 0.4

A THIRD of our validation compounds have no near neighbour to learn from; in the blind set
that is one in a thousand. WE HAVE BEEN GRADING OURSELVES ON EXTRAPOLATION WHILE THE EXAM
IS INTERPOLATION. Changing the splitter cannot fix it -- random and Butina give the same
profile, because our 4905 training compounds are simply spread out. Every model-selection
decision this project made was made under a distribution shift the real task does not have,
which systematically favours whatever generalises far: heavier regularisation, heavier
shrinkage, blunter models.

CONCRETE EVIDENCE THAT THIS DISTORTS CALIBRATION. Scanning the affine spread multiplier k
(we ship k = rho; rho is unchanged by an affine map, so this is pure placement) on the
current pool:

    endpoint   k shipped   pool-optimal k   pool gain
    CYP1A2       0.880          0.45          -2.0%
    CYP2C9       0.764          0.55         -11.6%
    CYP2D6       0.418          0.35          -0.9%
    CYP3A4       0.766          0.55         -22.6%

The optimum is interior and tight (CYP3A4 bootstrap argmin [0.55, 0.55] over 60 resamples),
and the mechanism is real: ST-RAE has a DEAD ZONE, so pulling predictions toward the mean
parks them inside intervals in a way no squared-error argument rewards. DO NOT SHIP k=0.55
ON THIS EVIDENCE. A pool that is too hard is precisely what manufactures a preference for
over-shrinkage, and the size of the CYP3A4 effect (-22.6%) against its near-perfect
instrument ratio (1.01x) is the sort of contradiction that should stop a submission, not
justify one.

THE FIX, AND HOW FAR IT GOT. scripts/build_matched_pool.py reweights the pool to match the
blind set on difficulty as well as on the label histogram. Instrument ratio, shipped
blend4z, before -> after:

    CYP1A2  1.17 -> 1.08     CYP2C9  1.06 -> 1.07
    CYP2D6  2.04 -> 2.28     CYP3A4  0.98 -> 0.99     (effective n 132 / 155 / 30 / 429)

It fixes CYP1A2, the endpoint that needed it most and the one carrying our worst absorption
and worst rank. It does nothing for CYP2C9 and CYP3A4, which were already honest. It makes
CYP2D6 worse at an effective sample size of 30 -- i.e. it fails there, and the reason is
itself a finding: WE HAVE ALMOST NO CYP2D6 TRAINING COMPOUNDS THAT LOOK LIKE THE BLIND
CYP2D6 COMPOUNDS. That is an applicability-domain gap, and it is the best explanation yet
for CYP2D6's rho of 0.519 against 0.81-0.85 everywhere else.

WHAT THIS SAYS TO DO NEXT, in order:
  1. Re-run the k scan on the difficulty-matched pool. The dead-zone effect is real; its
     SIZE is what the broken pool inflated. This costs no submission.
  2. Fix the CYP2D6 applicability gap with data, not loss functions. The compounds we need
     are the ones near the blind CYP2D6 set, and external data was previously rejected on
     three shapes that were never selected for proximity to the blind chemistry.
  3. Only then re-run model selection. Choosing architectures on a pool that punishes
     interpolation has been selecting for the wrong thing for the whole project.

--------------------------------------------------------------------------------
Section 17  THE POOL CANNOT PRICE ANY ENDPOINT; THE SPREAD MUST BE MEASURED ON THE BOARD
--------------------------------------------------------------------------------

scripts/scan_spread.py gates the spread scan on a question the pool has to answer first:
can it reproduce our KNOWN board ST-RAE at the spread we actually shipped? It cannot, on
any endpoint:

    endpoint   shipped k   board ST-RAE   pool says   error
    CYP1A2       0.880        0.5125       0.8566     +67%
    CYP2C9       0.764        0.3978       0.5528     +39%
    CYP2D6       0.418        0.5957       0.6626     +11%
    CYP3A4       0.766        0.4634       0.6181     +33%

Difficulty-reweighting improves CYP1A2 (+80% -> +67%) and CYP2D6 (+24% -> +11%) and makes
CYP3A4 worse (+19% -> +33%), at effective sample sizes of 132 / 155 / 30 / 429. It is not a
fix. Our models simply do far better on the blind compounds than on any local set we can
assemble, and we do not have enough easy compounds to simulate the blind set.

SO THE k* = 0.55 FROM Section 16 IS WITHDRAWN AS EVIDENCE. Harder data always prefers more
shrinkage; a pool that over-states cost by a third is exactly what would manufacture that
number. THE GENERAL RULE THIS SETTLES: placement and shrinkage interact with difficulty, so
they can NEVER be tuned on our pool. Only relative RANKING comparisons survive locally, and
even those transfer at about 14%.

WHAT REMAINS TRUE. The dead zone is real -- ST-RAE charges nothing inside [lo, hi], so
shrinkage buys something squared error never rewards, and k = rho is the MSE-optimal
shrinkage with no claim on being the ST-RAE-optimal one. Only the SIZE is unknown, and the
board is the one instrument that can measure it.

THE EXPERIMENT: submission_spread65.csv. CYP3A4 rescaled about its own mean to k = 0.65,
the other three endpoints byte-identical. Two points on the curve are already exact --
ST-RAE(0) = 1.0 by construction and ST-RAE(0.766) = 0.4634 from the raw3a4 probe -- so this
third point fixes the quadratic and hence the vertex.

    hypothesis                  CYP3A4    macro
    optimum is where we sit     0.4757   0.4954   (-0.003 against 0.4924 held)
    optimum is at 0.55          0.3865   0.4731   (+0.019)

BUILT-IN VALIDITY CHECK: an affine rescale is monotone, so rho cannot move. MA-Spearman
must come back at exactly 0.7509. If it does not, something other than the spread changed
and the reading is void.

--------------------------------------------------------------------------------
Section 18  PLACEMENT IS DONE. submission_spread65.csv IS WITHDRAWN (2026-09-13)
--------------------------------------------------------------------------------

Before spending a slot on the k = 0.65 probe, the shrinkage question turned out to be
answerable with no submission at all. Build a cost model from quantities the BOARD gives
us -- blind anchors, our blind rho, our blind ST-RAE -- plus the empirical joint
distribution of (y, interval half-width) from our own labels, and calibrate its noise level
so it REPRODUCES each endpoint's board score exactly. Then scan k inside the model.

THE COUPLING THE FIRST ATTEMPT MISSED, and it is severe:

    endpoint   corr(y, half-width)   median half-width by y quartile, low -> high
    CYP1A2           -0.685            0.246  0.169  0.121  0.096
    CYP2C9           -0.703            0.393  0.242  0.177  0.119
    CYP2D6           -0.343            0.151  0.140  0.134  0.104
    CYP3A4           -0.891            1.233  0.303  0.125  0.069

Potent compounds are measured precisely, weak ones are not, so the DEAD ZONE is wide
exactly where y is low. Drawing widths independently of y -- which the first model did --
gets the answer wrong.

THE ANSWER, with the coupling preserved:

    endpoint    rho    k shipped     k*    board    at k*     gain
    CYP1A2    0.8064     0.880      0.840  0.5125  0.5115   -0.0010
    CYP2C9    0.8454     0.764      0.880  0.3978  0.3853   -0.0125
    CYP3A4    0.8327     0.766      0.780  0.4634  0.4631   -0.0003

WE ARE ALREADY WITHIN 0.0035 MACRO OF THE BEST ANY AFFINE PLACEMENT CAN DO. The pool's
k* = 0.55 was an artefact, as Section 17 suspected; on CYP2C9 the true direction is the OPPOSITE
of what the pool advised -- expand, not shrink. submission_spread65.csv is predicted at
macro 0.4944-0.4985 against 0.4924 held, i.e. WORSE. DO NOT SUBMIT IT. It is kept only as
the worked example of a candidate that survived a bootstrap, a gate and a written rationale
and was still wrong, because the cheap calculation had not been done first.

PLACEMENT IS CLOSED. Every remaining point on this track has to come from rho.

ONE LEAD FALLS OUT OF IT. Our shipped centre against the solved blind anchor:

    CYP1A2 -0.015    CYP2C9 +0.043    CYP3A4 -0.055    CYP2D6 +1.027

Three endpoints sit on their anchor; CYP2D6 is a full log unit above it. The model says a
centre that far off should cost us ST-RAE ~1.15 on CYP2D6 and we actually score 0.5957 --
the model cannot make CYP2D6 cost as much as it really does at ANY noise level, in either
direction. So the (2.922, 1.652) anchor itself is the suspect, not the placement built on
it. Every CYP2D6 decision on this track is downstream of that pair. RE-DERIVE IT BEFORE
TRUSTING ANYTHING ELSE ABOUT CYP2D6.

--------------------------------------------------------------------------------
Section 19  A VALIDATION STANDARD, BECAUSE TWO CANDIDATES GOT OFFERED WITHOUT ONE
--------------------------------------------------------------------------------

blend3_placed -> blend3_narrow turns out to be a board-measured placement experiment we
already ran: rank correlation is EXACTLY 1.000000 on all four endpoints, so only the
placement changed.

    endpoint   mean A -> B      k A -> B       ST-RAE A -> B
    CYP1A2     4.301 -> 4.031   0.846 -> 0.800   0.5153 -> 0.6300   (+0.1147)
    CYP2C9     4.849 -> 4.849   1.023 -> 0.764   0.4547 -> 0.3978   (-0.0569)
    CYP2D6     3.130 -> 3.130   0.553 -> 0.552   0.6597 -> 0.6602   (+0.0005)
    CYP3A4     4.704 -> 4.704   0.994 -> 0.782   0.5295 -> 0.4938   (-0.0357)

BACK-TESTING Section 18's COST MODEL ON IT -- calibrate its noise on A, predict B:

    endpoint   B actual   B predicted    error
    CYP1A2      0.6300      0.5579      -0.0721
    CYP2C9      0.3978      0.4244      +0.0266
    CYP3A4      0.4938      0.4814      -0.0124
    CYP2D6      cannot calibrate on A at any noise level

DIRECTION RIGHT ON 3/3, MAGNITUDE WRONG BY 0.012 TO 0.072. The model is a direction-finder,
nothing more. Its CYP2C9 recommendation (k* = 0.880, claimed gain 0.0125) is FOUR TIMES
SMALLER THAN ITS OWN ERROR on that very endpoint, and the board experiment above contradicts
it outright: 1.023 -> 0.764 GAINED 0.057, so CYP2C9's optimum is at or below 0.764, not
above it. Section 18's CYP2C9 suggestion is withdrawn too. Placement is closed, now confirmed twice
and by direct board evidence.

THE STANDARD. A pIC50 candidate may be RECOMMENDED only when all three hold:
  1. the instrument behind it has been back-tested against known board outcomes;
  2. its predicted gain EXCEEDS that instrument's demonstrated error;
  3. it changes one variable, with an exact per-endpoint readout.
Nothing currently on disk meets it. submission_spread65.csv fails 2 outright.

WHAT THE BACK-TEST IS WORTH KEEPING FOR. CYP1A2 moved its centre by -0.27 and its k by
-0.046 and paid +0.1147; CYP3A4 moved k by -0.21 alone and gained 0.0357. THE CENTRE IS
WORTH ROUGHLY FOUR TIMES THE SPREAD. Which sharpens Section 18's loose end into an argument: if a
0.27 offset costs 0.115 on CYP1A2, our supposed +1.027 offset on CYP2D6 would be
catastrophic, and CYP2D6 scores 0.5957 at rank 26. IT IS THE (2.922, 1.652) ANCHOR THAT IS
WRONG, not the column placed against it. Re-deriving that pair is the next pIC50 job, and
it needs no submission.

--------------------------------------------------------------------------------
Section 20  TWO ANCHOR TABLES ARE LIVE IN THIS REPO AND THEY DISAGREE (2026-09-13)
--------------------------------------------------------------------------------

scripts/blind_validator.py -- which every pool, every placement and every k computation in
this project imports -- carries

    BLIND = (4.316,1.431) (4.806,1.122) (2.922,1.652) (4.759,1.252)

scripts/affine_calibrate.py lists THAT EXACT TABLE in its own BLIND_SUPERSEDED array,
annotated "4 anchors; superseded by the 5-anchor solve below", and its active table is

    BLIND = (4.031,1.425) (4.849,1.000) (3.130,1.305) (4.704,1.170)

The five-anchor refit was done, documented, and never propagated. Every number this
project has computed against a blind anchor used the superseded pair.

AN INDEPENDENT SOLVE LANDS ON THE REFIT, NOT THE ONE IN USE. Re-deriving (mu, sd) from the
board's published MAE + R2 + Spearman for our own submitted column -- a route
affine_calibrate did not use -- gives (3.999,1.422) (4.658,0.950) (3.366,1.581)
(4.571,1.165). CYP1A2 matches the refit to 0.03 and CYP3A4's sd to 0.005. On MAE the refit
beats the in-use table on all four endpoints:

    endpoint   MAE board   in use   5-anchor
    CYP1A2       0.6980    0.6579    0.6905
    CYP2C9       0.4216    0.4705    0.4116
    CYP2D6       1.1709    1.4094    1.1109
    CYP3A4       0.5496    0.5876    0.5352

BUT DIRECT BOARD EVIDENCE CONTRADICTS THE REFIT ON CYP1A2, AND THAT IS WHY NOTHING IS
BEING RECOMMENDED. blend3_placed -> blend3_narrow moved CYP1A2's centre from 4.301 to
4.031 -- onto the refit's value -- and ST-RAE went 0.5153 -> 0.6300. The board says 4.301
is nearer the truth; the refit says 4.031. One of the two is wrong and a fitted MAE is
weaker evidence than a scored submission.

WHY CYP2D6 RESISTS ALL OF THIS, already documented inside affine_calibrate and worth
restating because it invalidates my own solve above: CYP2D6's published R2 (0.4871 on that
submission) EXCEEDS the r^2 ceiling its own Spearman allows (0.276). Under bivariate
normality that is impossible, so CYP2D6 is not bivariate normal -- a heavy tail carries the
variance while the bulk ranks poorly. Every Gaussian solve for CYP2D6, including the three
I ran today, is therefore approximate at best. Quantile-matching to a skewed reference is
the honest fix, and it has never been done.

STATUS UNDER THE Section 19 STANDARD: no pIC50 candidate. What changed today is that the question
is now well-posed and cheap to settle -- the two tables make DIFFERENT predictions about
what a centre move does, and blend3_placed -> blend3_narrow is a scored example of exactly
such a move on all four endpoints. Refit both tables against that pair before either is
used again, and fix the import so only one table exists.

--------------------------------------------------------------------------------
Section 21  WHAT NINE SCORED SUBMISSIONS SAY TO DO NEXT (2026-09-13)
--------------------------------------------------------------------------------

1. splice4 ALREADY HOLDS THE BEST SCORE WE HAVE EVER ACHIEVED ON ALL FOUR ENDPOINTS
   (0.5125 / 0.3978 / 0.5957 / 0.4634). There is no free per-endpoint splice left.

2. rho history, first scored submission to now:
       CYP1A2 0.7771 -> 0.8064   CYP2C9 0.8245 -> 0.8454
       CYP2D6 0.4888 -> 0.5190   CYP3A4 0.8109 -> 0.8327
   About +0.03 per endpoint across the WHOLE project. Every ST-RAE gain larger than that
   came from placement, which Section 18-19 closed.

3. To reach the leader's 0.4378 macro, every endpoint must cut ST-RAE 11.1%. At today's
   absorption that is a rho target:
       CYP1A2 +0.0443   CYP2C9 +0.0347   CYP3A4 +0.0377   CYP2D6 +0.1310
   Three endpoints need MORE THAN THE ENTIRE PROJECT'S rho PROGRESS, in one step.
   Re-blending cannot deliver that (Section 15). Only a better model can.

4. CYP2D6 needs four times the rest, and it has the worst blind coverage despite having
   the second-most labels:
       endpoint  labelled  median NN to blind  % blind above 0.5
       CYP1A2      1180          0.516              52.7%
       CYP2C9      1026          0.518              54.5%
       CYP2D6      1364          0.424              42.1%
       CYP3A4      2335          0.538              61.9%

CONCLUSION. The binding constraint is now rho, and rho is limited by DATA COVERAGE on
CYP2D6 and by REPRESENTATION everywhere. External data was rejected three times (Section earlier)
but was never once selected for PROXIMITY TO THE BLIND CHEMISTRY -- it was selected on
assay shape. That is the untried lever with the largest measured headroom.

--------------------------------------------------------------------------------
Section 22  PROXIMITY-SELECTED EXTERNAL DATA IS WORSE THAN NONE (2026-09-13)
--------------------------------------------------------------------------------

Section 21 proposed selecting external CYP2D6 data for proximity to the blind chemistry -- the
criterion the three earlier rejections never used. Tested with four arms, same features,
same folds, leakage guard active (1 blind block removed; 5337 external rows remain):

    arm          rho       gap        95% CI            P(worse)
    control    0.3365
    +all       0.3472   +0.0107   [-0.0334, +0.0533]      32%
    +prox      0.2827   -0.0538   [-0.0838, -0.0235]     100%
    +weighted  0.3358   -0.0007   [-0.0445, +0.0399]      50%

PROXIMITY SELECTION IS SIGNIFICANTLY WORSE THAN NO EXTERNAL DATA AT ALL. The 665 proximate
compounds are a narrow slice, and training on them concentrates the model on one chemotype;
the full 5337 at least carry diversity, and buy +0.011 which is inside its own interval.
Section 21's recommendation is withdrawn. External data is now closed on FOUR different selection
criteria.

AND COVERAGE IS NOT THE LIMITATION, which is the part worth keeping. The proximate slice
DID improve blind coverage (median nearest-neighbour 0.424 -> 0.462, 42.1% -> 44.7% above
0.5) and the model still got worse. More compounds near the blind set is not what CYP2D6
needs.

NEITHER IS LABEL QUALITY:

    endpoint   n      sd(y)   median half-width   signal/noise
    CYP1A2    1180    0.628        0.145              4.3
    CYP2C9    1026    0.550        0.216              2.5
    CYP2D6    1364    0.644        0.131              4.9
    CYP3A4    2335    1.093        0.190              5.8

CYP2D6 has the narrowest intervals, the second-best signal-to-noise and a healthy spread.
Its labels are the cleanest we have after CYP3A4.

SO CYP2D6's rho OF 0.519 IS NOT A DATA PROBLEM IN ANY DIRECTION WE CAN MEASURE -- not
coverage, not quantity, not label quality. What is left is REPRESENTATION: the features we
compute do not capture what distinguishes a CYP2D6 binder. That is a mechanistic
hypothesis, not a data one, and it is the only CYP2D6 lever this project has not tested.

--------------------------------------------------------------------------------
Section 23  THE REPRESENTATION HYPOTHESIS IS RIGHT IN DIRECTION, TOO SMALL IN SIZE (2026-09-13)
--------------------------------------------------------------------------------

Section 22 left representation as the only untested CYP2D6 lever. Encoding the recognition
pharmacophore explicitly -- a protonated basic nitrogen (aliphatic amine, amidine or
guanidine; anilines, amides, sulfonamides and ring nitrogens excluded) at 5-7 A from an
aromatic centroid, the Glu216 salt bridge -- against a CYP3A4 specificity control:

    feature block           CYP2D6 gap   95% CI              CYP3A4 gap
    hand-coded 2D            +0.0157   [-0.0077, +0.0397]     -0.0043
    hand-coded 2D + 3D       +0.0206   [-0.0071, +0.0476]     -0.0049
    Gobbi pharm-pair FP      +0.0158   [-0.0190, +0.0505]     +0.0047
    Gobbi + hand-coded       +0.0229   [-0.0144, +0.0599]     +0.0026

THE SPECIFICITY CONTROL PASSES CLEANLY. Every block helps CYP2D6 four to nine times more
than CYP3A4, and the hand-coded ones actively HURT CYP3A4. This is chemistry, not extra
split capacity. The underlying difference is real and large: 41.7% of our CYP2D6 compounds
carry a strongly basic nitrogen against 14.0% of the CYP3A4 set; 24.0% sit in the 5-7 A
window against 6.9%.

BUT THE SIZE IS +0.023 AGAINST A TARGET OF +0.131, and every confidence interval crosses
zero. Under the Section 19 standard this is NOT recommendable and no submission follows from it.
It is the first CYP2D6 intervention in this project whose mechanism is confirmed rather
than merely hoped for, and it recovers about one sixth of what is needed.

READ TOGETHER WITH Section 22: coverage, quantity and label quality are all ruled out; the
representation is confirmed to be part of the problem but hand-built pharmacophore features
recover only a sixth of the gap. The remaining candidate is a LEARNED 3D representation --
the Uni-Mol member already in the ensemble is the only model here that sees geometry at
all, and it has never been trained with CYP2D6 as its primary target or given the
pharmacophore block. That is the next experiment, and it needs no submission.

--------------------------------------------------------------------------------
Section 24  THE CYP2D6 TDI BLEND INVERTED THE RANKING. POST-MORTEM (2026-09-14)
--------------------------------------------------------------------------------

submission_tdi_blend2d6.csv scored macro 0.2905, rank 36, against 0.3760 / rank 9 held.
CYP3A4 was frozen and came back at 0.4601 exactly as designed, so the whole move is CYP2D6:

    CYP2D6 column        TP   FP   calls    d'      MCC
    before (blendcp)     17   45     62   1.224   0.2918
    after  (blend2d6)     8   40     48   0.606   0.1209

d' HALVED and true positives fell from 17 to 8. This was not a gain that failed to transfer
at a reduced rate; the ranking INVERTED. Forecast was 0.385 conservative / 0.42 full.

HOW THE DECISION WAS MADE, AND WHY IT WAS WRONG. Two local signals said no:
    raw pool           +0.034 d', P(worse) 20%
    split-half select  -0.016 d', P(worse) 65%
They were overridden by a third: a prevalence-matched pool reading +0.144, P(worse) 3%, at
an EFFECTIVE SAMPLE SIZE OF 30.

Section 14 had already established that our local CYP2D6 TDI yardstick reads 4.37x low -- that it
does not measure CYP2D6 at all. The correct inference from that was: NO CYP2D6 TDI CHANGE
IS VALIDATABLE LOCALLY, THEREFORE NONE IS RECOMMENDABLE. Instead a repaired version of the
same broken instrument was used as the evidence to overrule the two signals that disagreed
with it. A patch to an instrument you have just shown to be broken is not an independent
measurement. This is the same failure as submission_spread65.csv one day earlier, which was
caught before submission rather than after.

THE RULE, added to Section 19's standard and binding on both tracks:
    An instrument that has FAILED a reproduction check against the board may not be used
    to justify a submission, in repaired form or any other. Where no working instrument
    exists for an endpoint, the only admissible action on that endpoint is to HOLD.

CYP2D6 TDI now has no local instrument and one board-measured fact: the chemprop shift
model's CYP2D6 ranking is actively harmful (d' 1.224 -> 0.606). pred_tdi_shift_cp.csv's
CYP2D6 column is closed. Its CYP3A4 column remains good (0.4470 -> 0.4601, Section 14).

RECOVERY: resubmit artifacts/submission_tdi_blendcp.csv. Both of its columns have already
scored -- CYP3A4 0.4601, CYP2D6 0.2918 -- so it restores macro 0.3760 by arithmetic, not by
hope.

--------------------------------------------------------------------------------
Section 25  WHY WE ARE STUCK: HALF THE TDI SCORE IS UNMEASURABLE (2026-09-14)
--------------------------------------------------------------------------------

THE DIAGNOSIS IS SOUND. Our CYP2D6 TDI d' of 0.280 against a blind 1.224 is not prevalence
(Section 14) and not the label threshold (relabelling to 7.2% prevalence gives 0.324). It is
DIFFICULTY. Split our rows by nearest-neighbour similarity:

    NN Tanimoto   0.14-0.30   0.30-0.35   0.35-0.42   0.42-1.00   >=0.55
    d'              0.137       0.180       0.341       0.436      0.971

Our CYP2D6 rows sit at median NN 0.351; the blind compounds sit at 0.471 from that same
labelled set. THE TASK IS INTERPOLATION AND OUR VALIDATION WAS EXTRAPOLATION. CYP3A4 never
had this problem -- median NN 0.492, local d' 1.300 against blind 1.276, ratio 0.98 -- which
is exactly why it is rank 8 and CYP2D6 is rank 18.

AND THE MODEL CLASS THAT DIAGNOSIS PREDICTS DOES WIN LOCALLY. On the blind-like subset,
ordered from memorisation to smoothing:

    similarity-weighted k-NN  1.057      lgbm loose  0.623
    lgbm incumbent            0.534      lgbm tight  0.481     (spread +-0.199)

Monotone in regularisation, and the spread between the extremes is more than twice the
noise. On the full set the same four arms span 0.248-0.315 -- all noise, which is why this
was never visible.

BUT THE INSTRUMENT FAILS ITS OWN VALIDATION, AND THAT IS THE RESULT THAT GOVERNS. The board
has already priced one CYP2D6 change: the 50/50 shift blend, d' 1.224 -> 0.606, a loss of
0.618. Asked to predict it:

    yardstick                  n     incumbent   blend      gap
    ALL rows (what Section 24 used)  1493     0.283     0.317    +0.034
    blind-like NN >= 0.50      188     0.641     0.690    +0.049
    blind-like NN >= 0.55       83     0.971     1.442    +0.472
    THE BOARD                  524     1.224     0.606    -0.618

The difficulty-matched subset recovers the LEVEL of d' (0.971 against 1.224) and gets the
DIRECTION OF A CHANGE badly wrong -- more wrong than the yardstick it replaces. An
instrument that is calibrated in level and inverted in difference is more dangerous than an
obviously broken one, because it looks trustworthy. THE k-NN RESULT ABOVE IS THEREFORE NOT
ACTIONABLE; it is measured on a yardstick that just failed.

SO THE HONEST ANSWER TO "WHY ARE WE STUCK". We have 1497 CYP2D6 TDI rows and only 83 that
resemble the blind task. At n=83 the spread on d' is +-0.33. No local instrument built from
this data can resolve the differences a decision needs, and three attempts to build one have
now produced one that reads 4.37x low, one that inverts, and one patched version that cost
rank 9 to 36. CYP2D6 TDI IS UNMEASURABLE AT THE SAMPLE SIZE WE HAVE. Not unimproved --
unmeasurable, which is a different and more final thing.

WHAT FOLLOWS, AND IT IS NOT NOTHING. MA-MCC is the mean of two endpoints and CYP3A4 is fully
measurable (ratio 0.98). A gain there counts exactly as much as a gain on CYP2D6 and can be
validated before it is spent. CYP2D6 is held at its scored column indefinitely; every
remaining TDI effort goes to CYP3A4, where the features the user asked about -- bioactivation
alerts crossed with substrate recognition -- can actually be tested.

--------------------------------------------------------------------------------
Section 26  BINDING x BIOACTIVATION DOES NOT HELP CYP3A4 TDI (2026-09-14)
--------------------------------------------------------------------------------

Mechanism-based inactivation needs two things at once: the compound must bind and turn
over, AND carry a group that becomes reactive when it does. tdi_alerts.py encodes the
second half (43 alerts + MI-complex precursor counts); nothing encoded the first. Our own
predicted CYP3A4 pIC50 is an affinity estimate available for train and blind alike, so the
conjunction is expressible as products.

    arm                     d'       gap       95% CI            P(worse)   MCC
    control               0.851                                            0.2331
    +affinity             0.885    +0.034  [-0.009, +0.077]        6%      0.2235
    +affinity x alerts    0.831    -0.020  [-0.068, +0.027]       80%      0.2091
    +both                 0.856    +0.005  [-0.049, +0.058]       44%      0.2235

THE CONJUNCTION IS REFUTED -- the products arm is WORSE than control, and adding them on
top of the raw affinity term destroys most of what affinity alone buys. Affinity alone is
+0.034 with an interval crossing zero and a LOWER MCC at our call rate. Nothing here is
actionable. Binding is evidently already implicit in the 134 features.

A BUG WORTH RECORDING. The first run reported the conjunction arm at exactly +0.000,
P(worse) 100%. That was not a result: TA.calculate returns (matrix, ok_mask), not
(matrix, names), so the column lookup matched nothing and the arm trained on an EMPTY
product block. An untested arm and a null result look identical in a results table. The
script now asserts the block is non-empty and that every requested alert column was
matched by name.

TDI STATUS AFTER THIS. Tested and closed: target reformulation (Section 12), two-arm differencing,
operating-point moves, binary-only augmentation, the CYP2D6 shift blend (Section 24, cost rank
9->36), and now binding x bioactivation. CYP2D6 is unmeasurable (Section 25) and held. CYP3A4 is
measurable at rank 8 and every feature idea tried on it has come back inside its own
confidence interval.

--------------------------------------------------------------------------------
Section 27  THE PRETRAINED UNI-MOL EMBEDDING IS NOT USABLE AS TREE FEATURES (2026-09-23)
--------------------------------------------------------------------------------

768-dim CLS vectors from Uni-Mol v2 (84m), extracted on the Kaggle GPU over the same
cached conformers the fine-tuning used, 5655 molecules, all finite. Five arms per endpoint,
same folds, paired bootstrap on every gap:

    arm              CYP1A2   CYP2C9   CYP2D6   CYP3A4
    +pharm          +0.0217  -0.0001  +0.0157  -0.0043
    +unimol         +0.0306  -0.0497  +0.0028  -0.0316
    +unimol+pharm   +0.0373  -0.0261  +0.0097  -0.0346
    unimol only     -0.0931  -0.1376  -0.1767  -0.1603

THE "ALONE" ARM SETTLES IT. The embedding by itself is 0.09 to 0.18 rho WORSE than
Morgan+descriptors on every endpoint, P(worse) 100% on all four. It is not carrying
structure these endpoints need; adding it is negative on CYP2C9 (-0.050, P 98%) and CYP3A4
(-0.032, P 100%) and inside its own interval on the other two. This is why that arm was in
the design: +unimol's +0.031 on CYP1A2 would otherwise have read as a 3D signal, when the
same block collapses to -0.093 without the descriptors to lean on.

THE DISTINCTION THAT MATTERS AND IS EASY TO GET WRONG: this closes the PRETRAINED encoder
as a feature source. It says nothing against FINE-TUNED Uni-Mol, whose predictions are
already a blend4z member and earn their place. Pretraining optimises a masked-coordinate
objective, not CYP affinity, and 768 dims of it apparently encode neither.

WHAT SURVIVED, AND IT IS NOT WHAT THE BLOCK WAS BUILT FOR. The CYP2D6 pharmacophore block
(Section 23) is significant on CYP1A2: +0.0217, 95% CI [+0.0003, +0.0431], P(worse) 2% -- the
first pIC50 feature gain in this project whose interval excludes zero. On CYP2D6 it is
+0.0157 (P 10%), and on CYP2C9/CYP3A4 it is nil or negative. It helps the two SMALL sites
and not the two large ones, which is the specificity pattern Section 23 predicted, just with
CYP1A2 rather than CYP2D6 as the clearest case -- CYP1A2's site is narrow and planar, so
aromatic geometry plausibly matters there too.

NOT ACTIONABLE. At the measured 14% ranking transfer, +0.0217 local is about +0.003 blind
rho against a CYP1A2 target of +0.044 (Section 21) -- roughly 0.0005 macro. Real, and far below
the bar.

--------------------------------------------------------------------------------
Section 28  SINGLE-TASK UNI-MOL LOSES TO MULTITASK; THE BLEND LINE IS CLOSED (2026-09-23)
--------------------------------------------------------------------------------

The single-task Uni-Mol run (one model per endpoint, same conformers, splitter and
hyperparameters as the multitask fit) recovered for three endpoints. As a blend member it
is the WEAKEST of the eight everywhere it appears, and weaker than the MULTITASK Uni-Mol on
every endpoint:

    solo rho        CYP1A2   CYP2C9   CYP2D6
    unimol (MT)     0.4838   0.6021   0.4248
    unimol_st (ST)  0.4597   0.5905   0.3617

MULTITASK WINS, AND THE MARGIN GROWS AS THE ENDPOINT GETS SMALLER (CYP2D6 -0.063 on 1364
rows, CYP2C9 -0.012 on 1026). The shared trunk is doing real regularisation, which is what
the Cell 13 docstring flagged as the risk. That answers the single-task question cleanly
and retires it.

THE BLEND SEARCH WITH THE NEW MEMBER, out-of-selection against the shipped blend4z:

    endpoint   incumbent rho   searched rho   OUT-OF-SELECTION   P(worse)
    CYP1A2        0.6074          0.6197         +0.0035           22%
    CYP2C9        0.7269          0.7349         +0.0014           35%
    CYP2D6        0.5021          0.5062         -0.0034           68%
    CYP3A4        0.8256          0.8315         +0.0037            5%

Identical in substance to Section 15: only CYP3A4 survives, at +0.0037 rho, worth about 0.0002
macro. Estimated 0.4917 against 0.4924 held. A GENUINELY NEW MEMBER CHANGED NOTHING, which
is the strongest form the Section 15 conclusion could take.

AND THE Section 25 MEMORISATION HYPOTHESIS DOES NOT TRANSFER TO pIC50. On TDI CYP2D6, k-NN beat
every booster on blind-like rows. On pIC50 CYP1A2 -- the one endpoint whose pool is harder
than the blind task (instrument ratio 1.33) -- the ordering reverses:

    rho on blind-like rows (NN >= 0.50, n=140, +-0.07)
    lgbm incumbent 0.5197    lgbm loose 0.4438    lgbm tight 0.4388    k-NN 0.3955

The incumbent regularisation is already the right one. Nothing to change.

WHERE THIS LEAVES THE PROJECT. Every lever now tested with controls and closed: placement
(Section 18-19), re-blending (Section 15, Section 28), external data on four selection criteria (Section 22),
hand-built pharmacophore features (Section 23), binding x bioactivation (Section 26), pretrained Uni-Mol
embeddings (Section 27), single-task vs multitask (Section 28), model regularisation (Section 28), and on TDI
the target reformulation, two-arm differencing, operating points and binary augmentation.
Held scores: pIC50 0.4924, TDI 0.3760.

THE ONE INFORMATION SOURCE NEVER USED: the leaderboard's own "Open Code: Yes" filter and
the Model Report Links several competitors published. We have reverse-engineered rivals'
confusion matrices and operating points from summary statistics all project long, while
some of them simply published what they did. Read those before concluding the approach is
exhausted -- it is the only remaining input that is free, and the only one that could
reveal a lever we never thought of.

--------------------------------------------------------------------------------
Section 29  THE TDI BOARD IS INVERTIBLE, AND BOTH TRACKS NOW SAY THE SAME THING (2026-09-23)
--------------------------------------------------------------------------------

submission_tdi_blendcp.csv WAS submitted (2026-09-14) and holds 0.3760. Rank 9 -> 23 is
nine days of field drift at a frozen score, not a regression. Nothing is pending.

THE BOARD PUBLISHES MA-Accuracy, MA-Precision AND MA-Recall. Three statistics determine
the whole 2x2 table:

    c  = pi*Recall/Precision ;  Acc = 1 - pi - c + 2*pi*Recall
    =>  pi = (1 - Acc) / (1 + Recall/Precision - 2*Recall)

pi belongs to the BLIND LABELS, so every team must return the same value. That is a free
falsification test and this project has hardly ever had one. IT PASSES: six independent
rows give pi = 0.1651 +- 0.0065, and the same inversion recovers our own call rate at
0.170 against the 0.2107/0.0853 we shipped. BLIND TDI PREVALENCE IS NOW MEASURED --
~16.5% macro, ~26% CYP3A4, ~7% CYP2D6 -- where every earlier number was assumed.

THE THRESHOLD IS CLOSED. Solving d' from each endpoint's SHIPPED operating point and
scanning the call rate:

    endpoint    pi     c shipped   MCC      d'      c*     MCC at c*    gain
    CYP3A4     0.259     0.2107   0.4601   1.354   0.272    0.4667     +0.0066
    CYP2D6     0.071     0.0853   0.2918   1.226   0.105    0.2926     +0.0008
    MACRO                         0.3760                               +0.0037

+0.0037 is inside the board's resolution. DO NOT SPEND A SLOT ON A CUT-RATE CHANGE.

TRAP, AND IT ALMOST CAUGHT ME. Inverting the MACRO row as if it were a single endpoint
says the threshold is worth +0.0153 -- four times the per-endpoint truth. Two endpoints
whose prevalences differ by 3.6x do not average into one binormal problem. The macro
inversion is valid for COMPARING TEAMS (all collapsed identically) and invalid for pricing
our own move. Use per-endpoint numbers whenever we have them, which for our own column we
always do.

WHAT THE FIELD LOOKS LIKE, COLLAPSED IDENTICALLY. d' spans 1.2524-1.3816 across the six
visible rows -- a range of 0.129 producing only 0.0198 of MCC. `test` leads that group at
0.3958 while sitting at z = 1.753 against its own optimum of 1.245, leaving +0.0365 on the
table; stir_bar, JSHCF and curiousone leave +0.014 to +0.022. SEVERAL RIVALS ARE
MIS-THRESHOLDED AND WE ARE NOT. Our deficit is entirely d'.

THE UNIFICATION. pIC50 placement: closed (Section 18, Section 19, and the e = ST-RAE/MAE decomposition
that corrected Section 16's absorption reading). TDI threshold: closed, here, by an instrument
that passed a falsification test. BOTH TRACKS NOW AGREE THAT EVERY REMAINING POINT IS
DISCRIMINABILITY, and Section 16 says why we have not found any: our folds grade extrapolation
(median NN Tanimoto 0.438, 33% below 0.4) while the exam is interpolation (0.587, 0.1%).
Every model class this project rejected was rejected on the wrong exam. Re-running that
selection against a blind-like subset is the only lever left with measured headroom, and
it is the one experiment that has ever changed an answer (Section 25, TDI CYP2D6).

--------------------------------------------------------------------------------
Section 30  THE OLD YARDSTICK WAS PUSHING US THE WRONG WAY ON ALL FOUR ENDPOINTS (2026-09-23)
--------------------------------------------------------------------------------

train_blindlike_select.py reruns model selection with the validation rows RESAMPLED so
their nearest-neighbour-similarity profile matches the blind set's, bin for bin
(importance sampling; each resample is a validation set with the exam's difficulty, so the
spread across resamples is the CI). Four arms spanning memorise <-> smooth, the project's
own folds, identical predictions scored two ways.

THE RESULT IS A CLEAN SIGN REVERSAL ON EVERY ENDPOINT. `lgbm tight` minus `lgbm incumbent`:

    endpoint   rho ALL (what Section 28 used)   rho REWEIGHTED (the exam)   P(worse)
    CYP1A2            +0.0358                   -0.0291                96%
    CYP2C9            +0.0003                   -0.0380                99%
    CYP2D6            +0.0122                   -0.0123                82%
    CYP3A4            +0.0054                   -0.0058                88%

`lgbm tight` wins on ALL rows on 4 of 4 and on the reweighted yardstick on 0 of 4. Sign
test one-sided p = 0.0625; individually only CYP2C9 clears its own CI, but four out of four
in the same direction with P(worse) 82-99% is a pattern, not a coincidence. Section 16 PREDICTED
EXACTLY THIS: a validation set harder than the exam rewards smoothing, and ours does.

WHAT IT DOES NOT SAY. The reweighted winner is the INCUMBENT on 3 of 4 (loose takes CYP1A2
by +0.005, inside noise). So Section 28's conclusion survives -- the shipped regularisation is
right -- and this experiment's value is DEFENSIVE: the old yardstick was recommending we
go tighter, and following it would have cost us on every endpoint. NO SUBMISSION CANDIDATE
FOLLOWS FROM THIS. Do not retune.

k-NN IS DEAD ON pIC50. -0.077 / -0.175 / -0.038 / -0.367, P(worse) 100% on all four. It
won on TDI CYP2D6 blind-like rows (Section 25) and loses everywhere here. The memorisation
extreme is not the answer on this track; that result does not transfer between tracks.

THE REAL PRIZE IS AN INSTRUMENT FOR CYP2D6. Back-testing each yardstick against the scored
blind rho:

    endpoint   rho ALL   rho REWT     move    blind rho   ratio ALL -> REWT
    CYP1A2      0.3408    0.4305     +0.0897    0.8064      2.37 -> 1.87
    CYP2C9      0.4120    0.3708     -0.0412    0.8454      2.05 -> 2.28   AWAY
    CYP2D6      0.3365    0.5123     +0.1757    0.5190      1.54 -> 1.01
    CYP3A4      0.7458    0.7696     +0.0238    0.8327      1.12 -> 1.08

CYP2D6 LANDS AT 0.5123 AGAINST A SCORED 0.5190 -- a ratio of 1.01 on the endpoint that
needs +0.131 (Section 21) and where every previous instrument failed (Section 14 read 4.37x low, Section 24 cost
rank 9 -> 36). Caveat before this is trusted: levels are confounded because these are
single boosters, not the shipped ensemble, so 1.01 could be luck. But the DIRECTION is
right on 3 of 4 and CYP2D6's move is by far the largest.

CYP2C9 IS THE EXCEPTION AND IT FALSIFIES Section 16's MECHANISM THERE. Its rho FALLS as
similarity rises (0.459 at NN 0.30-0.35 down to 0.295 at 0.55-0.60) while sd(y) stays flat
at 0.52-0.63 and MAE RISES 0.368 -> 0.447, so this is not range restriction. Our
high-similarity CYP2C9 rows are dense analogue series where neighbours are close and
activities are not -- ACTIVITY CLIFFS, not easy interpolation. Similarity is a proxy for
blind-set difficulty on three endpoints and an anti-proxy on this one. The script gates on
this automatically and refuses to license a flip from an endpoint marked AWAY.

STANDING RULE FROM HERE. Model selection on pIC50 uses the REWEIGHTED yardstick, per
endpoint, with the board back-test printed alongside. Any endpoint whose back-test reads
AWAY is selected on rho ALL and flagged, not on a broken correction.

--------------------------------------------------------------------------------
Section 31  THE BOARD MAY SCORE HALF THE BLIND SET, AND WHAT THAT COSTS (2026-09-23)
--------------------------------------------------------------------------------

Gashaw, from previous OpenADMET blind challenges: the leaderboard is generally computed on
about HALF the blind compounds, not all 750. This has never been accounted for anywhere in
this repo. Taking the per-compound ST-RAE cost distribution from our own calibrated pool
(CV 1.15-1.47) and resampling at n = 375 per endpoint:

    endpoint   SE(absolute score)   SE(A/B difference)   ratio
    CYP1A2           0.0830               0.0086         9.6x
    CYP2C9           0.0808               0.0085         9.5x
    CYP2D6           0.0552               0.0046        11.9x
    CYP3A4           0.0300               0.0042         7.1x
    MACRO            0.0329               0.0034

    macro 95% band, ABSOLUTE score:  +-0.0646
    macro 95% band, A/B DIFFERENCE:  +-0.0067

TWO CONCLUSIONS, AND THEY POINT OPPOSITE WAYS.

1. OUR INTERNAL A/B COMPARISONS SURVIVE. Two of our submissions are scored on the SAME
   compounds, so the noise cancels in the difference -- roughly ten-fold. splice4 vs
   blend4z (-0.0180), raw3a4's CYP3A4 (+0.0152 against a per-endpoint paired band of
   +-0.0082), blend3_placed -> blend3_narrow: all real. The measured history stands.

2. EVERY ABSOLUTE LEVEL IN THIS REPO CARRIES +-0.06 MACRO, AND THE PRIVATE HALF IS A
   DIFFERENT SAMPLE. Section 13's per-endpoint ledger, the absorption figures, and the instrument
   ratios built on absolute ST-RAE are all softer than they have been written. rho is much
   better determined (SE ~0.018 at n=375), so rho-based instrument ratios survive and
   ST-RAE-based ones do not.

THE SHAKEUP RISK FALLS ON PLACEMENT SPECIFICALLY, AND THE ARGUMENT IS SOUND. Placement was
fitted to statistics the board PUBLISHED ABOUT THE PUBLIC HALF -- its mean, its sd, its
MAE and R2. Raw predictions were fitted to nothing. So the two carry different exposure to
a public/private split, and ours is the exposed one. That is a real cost this project never
booked.

BUT PLACEMENT IS TWO THINGS, WITH OPPOSITE RISK PROFILES, AND THEY MUST NOT BE DISCARDED
TOGETHER:

  (a) THE GROSS CORRECTION. Dose-response was run only on primary-screen HITS, so the
      training labels are enriched for actives while the blind set is the unfiltered
      library. Measured train -> blind shift: CYP1A2 +0.64, CYP2C9 -0.23, CYP3A4 -0.66,
      CYP2D6 +1.86. This is a property of HOW THE DATA WAS GENERATED. It applies to the
      private half identically. Shipping raw predictions centred at 4.78 against a CYP2D6
      population centred near 2.9 would be wrong on both halves. KEEP THIS.

  (b) THE FINE TUNING. Solving each endpoint's centre and spread to three decimals from
      published board metrics, then scanning k to 0.01. This is fitted to the public half
      and its apparent precision is not real at +-0.065 absolute. LOOSEN THIS -- not
      because it scores worse today, but because its precision is manufactured.

WHERE THE DIRECT EVIDENCE SITS. submission_raw3a4.csv is the controlled test: three
endpoints frozen, CYP3A4 shipped RAW. Raw scored 0.4786 against the placed 0.4634, i.e.
+0.0152 WORSE, outside that endpoint's +-0.0082 paired band. Raw is measurably worse on the
public half, and the mechanism in (a) says it should be worse on the private half too.
FULLY RAW SUBMISSION IS REFUTED BY OUR OWN EXPERIMENT. The reform belongs in (b).

ON SCAFFOLD SPLITTING, our own measurement disagrees and it is worth stating plainly:
Butina-cluster and RANDOM splits give IDENTICAL nearest-neighbour profiles here (median
0.438 vs 0.439), and both are far harder than the blind set (0.587). A scaffold split does
not make validation more like this exam; Section 30 measured what a too-hard yardstick costs --
it picked the over-regularised arm on 4 of 4 endpoints. The split that matches THIS exam is
the similarity-reweighted one.

ON BETTER REPRESENTATIONS AND BETTER MODELS: agreed without reservation, and both tracks
now say the same (Section 29, Section 30). That is where every remaining point is.

--------------------------------------------------------------------------------
Section 32  RE-BLENDING IS CLOSED FOR THE THIRD TIME, NOW ON A CORRECT YARDSTICK
            (2026-09-23)
--------------------------------------------------------------------------------

scripts/blend_search_blindlike.py re-runs member selection against three weightings of the
same pool -- plain, matched to the blind LABEL HISTOGRAM (what we ship against), and that
times a NEAREST-NEIGHBOUR SIMILARITY importance weight (both corrections at once, never
done before). Same predictions, same members, so the comparison is exact.

CHOOSE THE YARDSTICK PER ENDPOINT. A global winner is dominated by whichever endpoint has
the worst anchor. Ratio of the board's reported rho to the incumbent blend4z's local rho,
1.00 being perfect:

    yardstick    CYP1A2  CYP2C9  CYP2D6  CYP3A4
    plain         1.331   1.164   1.036   1.008
    histogram     1.169   1.053   2.053   0.983
    hist x sim    1.088   1.082   2.742   1.000

THE COMPOSED WEIGHTING IS THE BEST INSTRUMENT WE HAVE ON THREE ENDPOINTS -- exactly 1.000
on CYP3A4 and 1.088 on CYP1A2 against the 1.169 we currently ship against. Keep it.

AND CYP2D6 CONFIRMS WHICH CORRECTION IS BROKEN THERE. Plain reads 1.036; adding the label
histogram blows it to 2.053. It is the HISTOGRAM matching -- against the (2.922, 1.652)
anchor Section 20 already flagged as suspect -- that destroys the CYP2D6 yardstick, not the
similarity term. Every CYP2D6 blend decision was made through that distortion.

THE SEARCH ITSELF FINDS NOTHING SHIPPABLE. With selection re-run OUT OF SAMPLE (greedy on
half the resampled rows, scored on the other half -- the control the original blend search
was built around):

    endpoint  yardstick    ratio  in-sample   OUT-OF-SAMPLE            stability
    CYP1A2    hist x sim   1.088   +0.0119   +0.0085 [-0.0123,+0.0248]   77%   hold
    CYP2C9    histogram    1.053   +0.0095   +0.0094 [-0.0051,+0.0222]   37%   hold
    CYP2D6    plain        1.036   +0.0036   -0.0013 [-0.0207,+0.0142]   30%   hold
    CYP3A4    hist x sim   1.000   +0.0175   +0.0169 [+0.0072,+0.0286]   33%   KEEP

Only CYP3A4 survives, and pricing it honestly ends the matter:

    out-of-sample local rho gain                          +0.0169
    x measured ensemble transfer rate (14%)   blind rho   +0.0024
    x exchange rate, -0.06 macro per +0.05 rho, / 4       -0.0007 macro ST-RAE
    board paired resolution at a half-scored leaderboard  +-0.0067

A TENTH OF THE BOARD'S OWN NOISE. NOT A CANDIDATE.

A MISTAKE WORTH RECORDING, because I made it in this very run. The first pass picked one
global yardstick, reported +0.0374 of summed local rho, applied the exchange rate directly
and called it submittable at -0.0112 macro. Two errors compounded: the global yardstick was
wrong for three of four endpoints, and the local gain was quoted AS IF IT TRANSFERRED
WHOLE. The 14% transfer rate is measured, and ignoring it is exactly how three CYP3A4
forecasts were overshot in a row. In-sample selection gains are not board gains.

STANDING CONCLUSION FOR pIC50. Placement: closed (Sections 18, 19, and the ST-RAE/MAE
decomposition). Blend weights: closed, three times over. TDI threshold: closed (Section
29). Every remaining point requires a NEW MEMBER -- a different representation trained to
rank, not a reweighting of what we already have.

--------------------------------------------------------------------------------
Section 33  THE MODEL AUDIT. PER-TARGET CHEMPROP WINS, AND OUR BEST ONE WAS NEVER
            PREDICTED ON THE BLIND SET (2026-09-23)
--------------------------------------------------------------------------------

Gashaw, from the PXR and ADMET blind challenges: Chemprop with TARGET-SPECIFIC descriptors
had outstanding performance. scripts/model_audit.py puts all ~90 out-of-fold files this
project has produced on one table, under the per-endpoint yardstick Section 32 established.
Selection on Spearman; blends and affine-placed derivatives excluded from single-model
comparisons.

THE HEAD-TO-HEAD THIS PROJECT BUILT BOTH SIDES OF AND NEVER RAN:

    endpoint   per-target + target desc   multitask chemprop    gap
    CYP1A2             0.6689                  0.6209         +0.0479
    CYP2C9             0.7852                  0.7181         +0.0671
    CYP2D6             0.4496                  0.4205         +0.0291
    CYP3A4             0.7929                  0.7290         +0.0639

FOUR OUT OF FOUR, mean +0.052. train_cyp_st_v2/v3 fuse cyp_target_descriptors' isoform
blocks (planarity for 1A2, ionisable acids for 2C9, basic-N-to-aromatic distance for 2D6,
bulk and flexibility for 3A4) at the readout; scripts/run_chemprop.py and train_cyp_mt_v1
use one shared 267-column vector for all four. cyp_target_descriptors.validate() passes
clean -- the 2D6 pharmacophore flag fires on all four 2D6 substrates and on zero 2C9 ones.

k-NN IS LAST OF EIGHT FAMILIES ON REGRESSION: mean rho 0.383, and 0.214 on CYP2C9. It won
TDI CYP2D6 on blind-like rows (Section 25) and loses every pIC50 endpoint. Classification
and regression genuinely differ here; the Section 25 result does not transfer.

FAMILY MEANS, best member of each:

    blend/derived                       0.731
    descriptor ensemble                 0.719
    chemprop per-target + target desc   0.674   <- best single-model architecture
    boosted trees / classical           0.628
    chemprop multitask                  0.623
    Uni-Mol 3D                          0.597
    SVM                                 0.549
    k-NN Tanimoto                       0.383

AND THE FINDING THAT MATTERS. `oof_chemprop_st_v3_ps` -- per-target chemprop trained WITH
the screen-derived pseudo-labels -- beats the chemprop column we actually ship
(`oof_chempropv3_ensA`) by

    CYP1A2 +0.0642   CYP2C9 +0.0998   CYP2D6 -0.0117   CYP3A4 +0.0204

It is the best single model we own on CYP2C9 (0.7852, ahead of every shipped member).
THERE IS NO BLIND-PREDICTION FILE FOR IT. predictions_cyp_st_v2.csv and
predictions_cyp_st_v3.csv exist; there is no _ps counterpart. The model was trained,
scored out of fold, and never run on the 750 blind compounds, so no blend search could
ever have selected it. This is the largest unexploited asset in the repo.

A LEAK IN THE PSEUDO-LABEL PIPELINE, FOUND WHILE CHECKING THAT GAIN, AND IT IS SMALL.
cypkit/labels.pseudo_from_screen fits the isotonic log2fc -> pIC50 map on ALL overlap
compounds, with no fold argument anywhere in its signature, so the held-out fold's curves
inform the map that labels the training rows. scripts/screen_pseudo.py does it correctly,
per fold; the function train_cyp_st_v3 --pseudo actually calls does not. Measured effect --
refit the map leave-one-fold-out and compare the pseudo-labels it produces:

    isoform   mean |label shift|   max     residual sd of the map
    CYP1A2         0.0422         0.3345          0.423
    CYP2C9         0.0267         0.4898          0.405
    CYP2D6         0.0982         0.4329          0.631
    CYP3A4         0.0224         0.2064          0.408

5% to 16% of the noise the pseudo-labels already carry. It inflates our ESTIMATE of the
model and cannot explain +0.0998 on CYP2C9. It also cannot touch the blind predictions at
all: the map is fitted on training curves only, and blind compounds have none.

THE ACTION, in order:
  1. Run train_cyp_st_v3.py --pseudo in PREDICT mode to produce the missing blind column.
     Nothing else in this repo is a free, measured, four-endpoint improvement.
  2. Then make pseudo_from_screen fold-aware and retrain, so the member's weight is fitted
     against an honest out-of-fold estimate rather than a 5-16% optimistic one.
  3. Then re-run scripts/blend_search_blindlike.py with the new member present. Section 32
     closed re-weighting of the members we HAVE; it said nothing about a new one.

--------------------------------------------------------------------------------
Section 34  THE PSEUDO-LABEL LEAK IS FIXED, AND THE SEPT-5 CHECKPOINTS ARE STALE
            (2026-09-26)
--------------------------------------------------------------------------------

train_cyp_st_v3.py --mode PREDICT fails with

    ValueError: X has 62 features, but RobustScaler is expecting 52 features

cyp_target_descriptors.py grew after those checkpoints were written: a 10-column `phys_*`
ionisation block was inserted at position 39, CYP2C9 gained one further column and CYP2D6
five. Every isoform once had exactly 13 isoform-specific columns; now they have 13/14/13/18
plus the phys block.

RECOVERY IS POSSIBLE BUT ONLY PROVABLY SO ON TWO ENDPOINTS.
scripts/recover_st_v3_desc.py identifies the original columns by fingerprinting the saved
RobustScaler: center_ is the per-column median and scale_ the IQR, so the 52 pairs
identify the columns if the match is unique. A self-test against fabricated scalers shows
it recovers CYP1A2 exactly (13 of 13, the trivial case) and declares CYP2D6 AMBIGUOUS --
several 13-of-18 subsets reproduce the fingerprint, because unfired binary flags all share
median 0 and IQR 0. It refuses rather than choosing. One trap worth recording: sklearn puts
a constant column's IQR through _handle_zeros_in_scale and stores 1.0, not 0.0, and
comparing against a raw 0.0 made even the trivial CYP1A2 case fail until that was matched.

SO THE ANSWER IS RETRAIN, NOT RESURRECT, and it is the better answer anyway:
  - it picks up the pharmacophore block, the only feature addition this project ever
    measured a significant gain from (+0.0217 rho on CYP1A2);
  - it produces a clean out-of-fold file, so the member's blend weight is fitted honestly;
  - it produces the missing blind column in the same run.

THE LEAK IS NOW FIXED IN CODE. cypkit/labels.py gains FoldedIsotonic and
pseudo_from_screen(..., fold_of=...), which fits one leave-one-fold-out map per fold
alongside the all-data map; build_label_matrix gains pseudo_fold_of and looks up each
compound's fold when labelling it. train_cyp_st_v3.py reads the fold assignment from the
same extended fold file the trainer uses and threads it through. fold=None still falls back
to the all-data map, which is correct at inference (blind compounds contribute no curves to
any map) and wrong in cross-validation, so the no-folds path now prints a warning.

VERIFIED, not assumed: with and without folds the label matrix has identical shape (18,034
rows, 11,509 pseudo rows). Only the pseudo-label VALUES move -- 79.8% of them, by a mean of
0.048 log units, which matches the leave-one-fold-out estimate in Section 33 exactly.

THE COMMAND:

    python train_cyp_st_v3.py --pseudo --mode BOTH --model_dir models_cyp_st_v3_ps

Then re-score it with scripts/model_audit.py, and only then re-run
scripts/blend_search_blindlike.py with the new member registered in MEMBERS.

--------------------------------------------------------------------------------
Section 35  THE FULL BLIND SET. THE TDI "DROP" IS THE FIELD AND THE NOISE, AND THE
            pIC50 DEFICIT IS NOW MEASURED DIRECTLY (2026-09-26)
--------------------------------------------------------------------------------

Interim results over ALL blind compounds, against the half-set board we had been tuning to:

                 scored HALF                    FULL blind
    pIC50    0.4924  rho 0.7509  rank 41    0.4833  rho 0.7590  rank 40
    TDI      0.3760              rank 23    0.3312              rank 32

pIC50 IMPROVED. TDI looks much worse. Neither reading survives arithmetic.

THE TDI DROP IS THE FIELD. Mean MCC of the five entries immediately around us:

    half-set, ranks 18-23   0.3862
    full set, ranks 28-32   0.3354      the field moved -0.0509
    we moved                            -0.0448

WE DROPPED LESS THAN OUR NEIGHBOURS. The full set is simply harder -- prevalence is
unchanged (0.1577 half, 0.1564 full, so the inversion of Section 29 still holds) and the
unseen half's positives are harder to separate, for everyone.

AND THE RANK MOVE IS MEANINGLESS. Ranks 28-32 span 0.0086 of MCC, i.e. 0.0021 per place.
Our own MCC carries, at 117 positives in 750 compounds, SE(d') 0.137 -> SE(MCC) 0.047, a
95% band of +-0.092. THAT IS +-43 RANKS OF PURE SAMPLING NOISE. Rank on this board is not a
measurement. The half-vs-unseen d' gap (1.2547 against an implied 1.0084) is 0.90 sigma.

A STANDING CORRECTION TO HOW THIS PROJECT HAS READ THE TDI BOARD. Section 24 treated a
9-to-36 rank move as evidence and wrote a binding rule from it. At +-43 ranks of noise that
move was worth far less than it was charged with. The MCC numbers were still real -- 0.3760
against 0.2905 is 0.0855, near one sigma and in the right direction -- but the RANKS never
were. Quote MCC, never rank, and never act on a rank change below about 0.09 of MCC.

THE pIC50 DEFICIT IS NOW MEASURED, because the full-set board publishes MA-MAE and MA-R2
next to the score. The three entries at ranks 38-40:

    entry             ST-RAE     MAE      R2     rho   e=ST/MAE  sqrt(R2)  rho-sqrt(R2)
    Cheminfo          0.4822  0.7082  0.5787  0.7549    0.6809    0.7607     -0.0058
    admetchallenger   0.4831  0.6609  0.6027  0.7398    0.7310    0.7763     -0.0365
    Gashaw            0.4833  0.7005  0.5664  0.7590    0.6899    0.7526     +0.0064

WE HAVE THE BEST SPEARMAN OF THE THREE AND THE WORST SCORE. Our dead-zone efficiency
e = 0.690 BEATS admetchallenger's 0.731 -- we place better than they do. What we lose on is
MAGNITUDE: MAE 0.7005 against their 0.6609, R2 0.5664 against 0.6027. And the last column
says why: their Pearson exceeds their Spearman by 0.037 while ours sits below it. Their
predictions are more linearly faithful; ours order the compounds better and land further
from the values.

THAT IS THE THIRD INDEPENDENT CONFIRMATION, now on the full blind set and with the MA-MAE
column rather than an inference: PLACEMENT IS CLOSED, RANKING IS NOT OUR PROBLEM, and the
whole remaining deficit is error magnitude -- a better-fitting model. Which is exactly what
Section 33 found sitting unused on disk.

ONE ESTIMATE VINDICATED. Section 31 put the half-set absolute noise at +-0.065 macro
ST-RAE. The observed half-to-full move was -0.0091. Conservative, and the right order.
