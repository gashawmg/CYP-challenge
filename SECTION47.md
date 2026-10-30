# Section 47. Target descriptors on the multi-head trunk. The best member we have

For six weeks the descriptor programme and the winning architecture were separate
projects, and nobody noticed.

## 1. What was missed

`train_cyp_mh.py` — the multi-head external-data model, 18 heads over ~21,000 compounds,
the architecture the rank-1 team publicly describes — **had no descriptors of any kind.**
Pure D-MPNN on SMILES, `hidden=400 depth=4 layers=2`, untouched since 8 September.

Meanwhile Sections 36 through 43 built and validated a whole descriptor programme (the
lean block, conformer shape, the per-endpoint spec) **exclusively on the single-task
chemprop path** — which every head-to-head comparison says is the weaker architecture:

| | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---|---|---|
| `mh` R2 | +0.163 | +0.328 | +0.187 | +0.625 |
| best chemprop R2 | −0.159 | +0.291 | +0.145 | +0.545 |
| `mh` bias | −0.022 | −0.013 | −0.020 | −0.037 |
| best chemprop bias | −0.257 | +0.009 | −0.138 | +0.222 |

And the thing Gashaw knew from PXR — that target-specific descriptors carried 75% of that
blend — had never been tried on the model that was actually winning here.

## 2. The shared block

One encoder over ~21,000 rows including external chemistry means there is no isoform to
key a per-isoform block on. `cyp_descriptor_spec.build_shared()` takes the **intersection
of the four lean blocks**: every column already validated in Sections 36/40, minus the
isoform-specific ones, computed rather than hand-listed so it cannot drift when
`cyp_target_descriptors_lean.py` changes.

Isoform specificity is not lost; it moves to where this architecture puts it anyway. The
trunk gets shared chemistry, the per-assay heads learn the rest.

## 3. The result

A same-day control (`--descriptor_spec none`) against the arm (`shared`), identical folds
and seeds, so descriptors are the only difference. Pooled over all folds:

| | control | + descriptors | Δ |
|---|---|---|---|
| CYP1A2 rho | 0.5249 | 0.5310 | +0.0061 |
| CYP2C9 rho | 0.6309 | 0.6705 | **+0.0396** |
| CYP2D6 rho | 0.4459 | 0.4594 | +0.0135 |
| CYP3A4 rho | 0.7878 | 0.8085 | **+0.0207** |
| **macro rho** | 0.5974 | **0.6173** | **+0.0199** |
| macro ST-RAE | 0.706 | **0.675** | −0.031 |
| macro R2 | 0.322 | 0.345 | +0.023 |
| macro MAE | 0.558 | 0.543 | −0.015 |

**Every endpoint improved on every metric.** The two folds read before pooling agreed to
within 0.0006 on macro rho (+0.0174, +0.0168), and the pooled figure came in slightly
above both.

Against the other members on the same pooled metric:

| | macro rho | macro ST-RAE |
|---|---|---|
| `oof_st_v6.csv` | 0.6128 | 0.763 |
| `oof_st_mh2.csv` | 0.6079 | 0.696 |
| **`oof_st_mh_desc.csv`** | **0.6173** | **0.675** |

Best single member on both axes.

## 4. It is a ranker, not a candidate

| | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---|---|---|
| `mh_desc` mean / sd | 5.16 / 0.443 | 5.06 / 0.612 | 4.76 / 0.427 | 4.75 / 0.983 |
| blind anchors | 4.32 / 1.431 | 4.81 / 1.122 | **2.92 / 1.652** | 4.76 / 1.252 |

CYP2D6 sits **+1.84 above** the anchor-solved blind mean with a quarter of the needed
spread. Uploaded raw it would score far worse than 0.4833 despite being the better model.
Its rho belongs in the blend; the level comes from elsewhere. `splice4`, which is what is
actually on the board, is well aligned on three of four means — that alignment came from
the placement machinery, not from any member.

## 5. Two defects found along the way

**A smoke test destroyed a production artifact.** `train_cyp_mh.py --smoke` took the
default `--oof_output` and wrote its 263-row fold-0 stub over `oof_st_mh.csv` — a
registered blend member, and one of the four in `BLEND4Z`, the incumbent every search is
measured against. `pred_st_mh.csv` survived only because `--smoke` skips PREDICT.
`--smoke` now redirects any output path left at its default.

**The blend search checked existence, not size.** A truncated member passed `exists()` and
would have contributed 263 rows to the baseline silently. `pic50_blend_search.py` now
flags any member under half the widest member's row count and treats it as missing —
fatal if it is in `BLEND4Z`. Per-endpoint coverage is deliberately not the test, since
`unimol_st` legitimately carries ~1,400 rows per endpoint and `v5_2d6` only one endpoint.

**Consequence to carry forward:** the original `mh` OOF is unrecoverable, so `mh` now
points at the control re-run. The board-scored blend4z used the original. On the pooled
metric the intact Sept-8 sibling `mh2` scores 0.6079 and the re-run 0.5974, so the
incumbent is now an approximation about 0.01 of rho weaker on one of four members, and
gains measured against it are very slightly overstated.

## 6. Next

1. Blend search with `mh` repointed and `mh_desc` registered — that yardstick, not this
   one, is where the +0.160 submission bar lives.
2. `--descriptor_spec shared3d` has now earned its ~20 minutes of conformer embedding.
3. Seed-ensembling comes back into scope once the architecture is settled, not before.

Still no submission candidate. The bar is unchanged: nothing ships until internal
validation beats 0.4833.
