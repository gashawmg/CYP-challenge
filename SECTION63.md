# Section 63. External data for TDI: the useful kind is already on disk and unused

Run date 2026-10-12. Short answer: **external TDI labels will not help, and external CYP
INHIBITION data probably will -- and it was fetched in September, turned into a feature
block, wired into the feature bank, and never once requested by a TDI model.**

---

## 1. External TDI labels: the data does not exist where we need it

| source | size | threshold | vs ours |
|---|---|---|---|
| OpenADMET CYP3A4 TDI (ours) | **3,584** labelled | IC50 shift >= 2.0 | -- |
| OpenADMET CYP2D6 TDI (ours) | **1,497** labelled | IC50 shift >= 2.0 | -- |
| FDA public CYP3A4 TDI (Faramarzi 2025) | ~620 | IC50 shift >= **1.5** | 17% more rows |
| FDA public CYP2D6 TDI | -- | -- | **none usable** |

Two things kill it.

**The threshold differs.** The FDA uses a 1.5-fold IC50 shift; the OpenADMET rule recovered
in Section 52 uses 2.0-fold. Pooling labels defined at different cutoffs imports the
disagreement as label noise, on a binary target where we already have only 764 and 324
positives.

**The endpoint that needs help has no public data.** The FDA paper states plainly that TDI
data was harvested for CYP1A2, 2B6, 2C8, 2C9 and 2C19 as well as 2D6, and that "the
databases were small and did not yield viable models (data not shown)". A team with FDA
curation resource could not build a CYP2D6 TDI model from public data. We are not going to
either. Our 1,497 rows are probably one of the larger CYP2D6 TDI sets in existence.

For CYP3A4, 620 external rows at a different threshold, added to 3,584 of our own, on the
endpoint whose submission column is frozen at the 0.4470 blend per Section 60 -- that is
not where an evening goes.

## 2. External inhibition data: already fetched, already a feature, never used

`cypkit/external_prior.py` exists since 2 September. It turns the Veith / NCGC qHTS panel
(PubChem AID 1851, **16,145 compounds** in `artifacts/external_cyp.csv`) into ten columns
-- a predicted pIC50 and a predicted P(active) for each of CYP1A2, 2C9, 2C19, 2D6 and 3A4
-- from models that never see a challenge label. Its own docstring records what it was
worth on the pIC50 track: mean Spearman **0.495 -> 0.507** over all rows, and
**0.542 -> 0.565** over fitted-curve rows.

`scripts/run_ensemble.py` builds it as the `prior` block. And every TDI pool member's
feature string is one of rdkit, cyp, tdialert, screen, ecfp, ecfpc, fcfp, maccs, ap,
avalon, avalon1k, rdkpath, tt, pattern, layered. **`prior` appears in none of them.**

Thirty learners, seven fingerprint families, five new ones added last night -- and the one
block built from 16,145 external molecules was never asked for.

## 3. What the direct overlap says, with its own caveat

Only 198 of 6,145 TDI compounds match the qHTS panel on InChIKey skeleton: 2.3% of the
CYP3A4 labelled set, 6.2% of CYP2D6. So external data cannot enter as extra training rows
-- which is exactly the conclusion `external_prior.py` reached for the pIC50 track, and for
a second reason it documents: on the 178 compounds measured in both, Veith and challenge
pIC50 correlate at Spearman 0.52-0.81 but sit on different scales, bias -0.3 to -1.1 log
units, fitted slopes 0.60 to 1.86.

On the handful that do overlap, external activity against `is_TDI`:

| external column | CYP3A4 is_TDI | CYP2D6 is_TDI |
|---|---|---|
| CYP3A4 pIC50 | **AUC 0.788** (81 cpds, 8 pos) | 0.365 (87, 23) |
| CYP2C19 pIC50 | 0.634 (84, 10) | 0.403 (92, 23) |
| CYP2C9 pIC50 | 0.555 | 0.480 |
| CYP2D6 pIC50 | 0.548 | 0.407 (93, 23) |
| CYP1A2 pIC50 | 0.489 | 0.483 |

Eight positives is not evidence and I am not going to pretend it is. But the direction is
mechanistically right for CYP3A4 -- a time-dependent inactivator has to bind first, so
being a CYP3A4 inhibitor in a completely independent assay should predict being a CYP3A4
TDI -- and it is flat-to-inverted on CYP2D6, consistent with everything else about that
endpoint.

## 4. Done: two members added

```
"lgb_prior": dict(kind="lgb", features="rdkit+cyp+tdialert+prior"),
"cat_prior": dict(kind="cat", features="cyp+tdialert+prior+maccs"),
```

mirroring `lgb_screen` and `cat_screen`, with `prior` added to the block order. The file
parses and `build_bank`'s `prior` branch already calls `EP.calculate`.

```
python scripts\run_tdi_ensemble.py --tag tdiP --save-oof
```

```
python scripts\tdi_per_endpoint_blend.py --tag tdiP
```

## 5. What I expect, stated before the run

A CYP3A4 member, on an endpoint whose submission column is frozen. Its value is diversity
inside the blend, not a new best single model -- and Section 59 established that CYP3A4's
blind set rewards **breadth**, with the broad equal-weight blend holding d' 1.287 -> 1.281
while `lgb_clf` alone fell 1.295 -> 1.163. A member built on genuinely external chemistry
is the most decorrelated thing available, and decorrelation is the only currency a rank
blend has.

On CYP2D6 I expect nothing, and the overlap table plus the FDA's own failure to model that
endpoint are two independent reasons why.

Nothing ships on this. The gate is unchanged: `artifacts/submission_tdi_splice_best.csv`
restores 0.3694, and per Section 60 a column changes only when the leaderboard says so.

Recorded because the answer to "would external data help" turned out to be that the
external data has been sitting in `artifacts/` for five weeks with a feature block already
written for it.
