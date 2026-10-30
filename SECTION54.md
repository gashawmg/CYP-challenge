# Section 54. One blend for two endpoints that agree about nothing, and the missing Avalon

Run date 2026-10-11, late. Analysis of the `tdiM` pool run. Two findings, one of them an
omission.

---

## 1. The two endpoints rank the members with zero relationship

Member AUC on CYP3A4 against member AUC on CYP2D6, across all twenty pool members:

**Spearman +0.003.**

Not weakly related. Unrelated.

| member | CYP3A4 AUC | CYP2D6 AUC |
|---|---|---|
| lgb_clf | **0.8200** | 0.5927 |
| lgb_screen | 0.8121 | 0.5638 |
| pooled | 0.8117 | 0.5872 |
| cat_clf | 0.8095 | 0.5882 |
| knn_fcfp | 0.7287 | **0.6283** |
| svc_c10 | 0.7404 | 0.6203 |
| svc_tan | 0.7406 | 0.6201 |
| struct | 0.8071 | 0.5361 |

The boosted trees own CYP3A4 and are mediocre on CYP2D6; the similarity kernels own CYP2D6
and are near the bottom on CYP3A4. `struct` is third on one endpoint and last on the other.

`blend_of()` applies one weight vector to both. That is a single compromise between two
orderings with no common direction, and the table says it is costing real score: the
RANK-BLEND is first on macro AUC (0.7198) while being **first on neither endpoint**, behind
lgb_clf on CYP3A4 (0.4087 against 0.4233) and behind four similarity members on CYP2D6
(0.1731 against knn_k5's 0.1922).

## 2. How much the compromise costs, measured honestly

Greedy forward selection with replacement on rank-averaged members, by AUC, **with the
selection run inside the cross-validation** so the reported number is not the maximum of a
search over the rows it is scored on. Thresholds taken at the blind rate, also out of fold.

| | equal-weight blend of all 20 | per-endpoint greedy | gain |
|---|---|---|---|
| CYP3A4 AUC | 0.8205 | **0.8404** | +0.0199 |
| CYP3A4 MCC at rate 0.2915 | 0.3993 | **0.4435** | **+0.0442** |
| CYP2D6 AUC | 0.6191 | **0.6410** | +0.0218 |
| CYP2D6 MCC at rate 0.0706 | 0.1332 | **0.1361** | +0.0029 |
| **macro MCC** | **0.2663** | **0.2898** | **+0.0235** |

Both columns are scored identically, so the +0.0235 is the compromise, not a metric change.
Note how differently it lands: CYP3A4 gains 0.044 of MCC, CYP2D6 gains 0.003 despite a
larger AUC gain, because at a 7% cut rate AUC barely moves MCC.

What gets chosen, across the five selection folds:

- **CYP3A4**: struct x11, lgb_clf x6, svc_fcfp x5, cat_screen x5, pooled x4, et_clf x2
- **CYP2D6**: knn_fcfp x7, svc_c10 x4, svc_tan x2, cat_clf x1

`struct` is the most heavily weighted CYP3A4 member despite ranking fourteenth by macro
AUC. That is not a contradiction -- it is decorrelation. Section 52 concluded the recovered
label rule was not worth submission time as a standalone score, and that stands; as a blend
member on the endpoint where the gate is informative, it is the single most useful thing in
the pool. On CYP2D6 it is the member LEAST correlated with the best single (r = 0.141) and
also the worst (AUC 0.5361), which is decorrelation without signal, and the selector
correctly never picks it.

This also agrees with the board rather than contradicting it. `run_tdi_ensemble.py`'s own
header records that fitted CYP2D6 weights putting 62% on knn_tan + svc_tan gained **+0.036
MCC on the blind set** (0.1825 -> 0.2180). Per-endpoint weighting has already been scored
and it worked.

## 3. Avalon is built, documented, benchmarked -- and not in the pool

`cypkit/features.py` supports an `avalon` block at 2048 bits, and its own docstring makes
the case for it:

> a path/feature hybrid from a DIFFERENT enumeration than Morgan, so it tends to disagree
> with ecfp rather than duplicate it, which is what a blend actually wants

`train_tdi_chemprop.py` measured it and acted on the result: Avalon is recorded there as
CYP2D6's best single fingerprint, and that script's default block list for CYP2D6 is
`["avalon"]` -- Avalon alone, nothing else.

`run_tdi_ensemble.py` never built it. The `order` list ran
`rdkit, cyp, tdialert, screen, ecfp, ecfpc, fcfp, maccs, ap, 3d`, with no `avalon`, and the
POOL had no Avalon member. The similarity-kernel block that was added specifically to vary
"which fingerprint defines similar" covered ECFP, FCFP, atom pairs, MACCS and ECFP counts,
and skipped the one fingerprint this project had already measured as best on the endpoint
that needed it.

Fixed: `avalon` added to `order`, and two members added to POOL --

```
"knn_avalon": dict(kind="knn", features="avalon"),
"svc_avalon": dict(kind="svc", features="avalon"),
```

`build_bank`'s fallback branch passes any unrecognised block name straight to
`features.fingerprint_block`, which already knows `avalon`, so no other change was needed.
Verified: the file parses, and `pyAvalonTools.GetAvalonFP` runs in `admet_gpu`.

One discrepancy to be aware of: `cypkit/features.py` sets Avalon to **2048** bits, while
`train_tdi_chemprop.py`'s benchmark table reports it at **1024** and `_superseded/
features_root_copy.py` used 1024. The 0.6043 figure quoted for CYP2D6 was measured at 1024
bits, so the new members are not reproducing that exact configuration.

## 4. What to run

```
python scripts\run_tdi_ensemble.py --tag tdiA --save-oof
```

Two things to read off it. First, whether `knn_avalon` and `svc_avalon` land in the CYP2D6
similarity group (AUC 0.61-0.63) or above it. Second, whether they are decorrelated from
knn_fcfp -- a new member only pays if it errs on different compounds, and the fingerprint
docstring's whole argument is that Avalon disagrees with Morgan.

Then re-run the per-endpoint selection on the new member OOF:

```
python scripts\tdi_per_endpoint_blend.py --tag tdiA
```

The gate is unchanged: per-endpoint selection plus Avalon has to clear the standing best
in `tdi_validator`'s **uncorrected** column, not only the shrinkage-corrected one. On the
numbers above, per-endpoint selection alone takes macro MCC from 0.2663 to 0.2898 at blind
rates, against a standing best of 0.3325 scored -- so it is not yet enough on its own, and
the previous local-to-board anchor (OOF 0.2524 scoring 0.285, a +0.033 offset) is a single
point and should not be leaned on.

Recorded because the pool had been built to vary exactly the axis Avalon sits on, and had
left out the one value of it the project had already measured as best.

---

# Addendum, same night. Avalon ran, and it pays on exactly one endpoint

`--tag tdiA` with `knn_avalon` and `svc_avalon` added, 22 members. Same per-endpoint
selection, same folds, same blind-rate cuts, paired bootstrap of the difference.

| | without Avalon (tdiM) | with Avalon (tdiA) | effect on MCC |
|---|---|---|---|
| CYP3A4 AUC / MCC | 0.8404 / 0.4435 | 0.8424 / 0.4402 | -0.0035, CI [-0.0222, +0.0153] |
| CYP2D6 AUC / MCC | 0.6410 / 0.1361 | 0.6371 / **0.1857** | **+0.0499, CI [+0.0054, +0.0948]**, P(better) 0.987 |
| macro MCC | 0.2898 | **0.3130** | **+0.0232** |

Nothing on CYP3A4, and a large, significant gain on CYP2D6 -- the endpoint where
`train_tdi_chemprop.py` had already said Avalon was best, and the endpoint that has
resisted everything else.

**The mechanism is the one `features.py` predicted.** On CYP2D6, `svc_avalon` scores AUC
0.6116, slightly BELOW `knn_fcfp`'s 0.6283 -- and its rank correlation with knn_fcfp is
**0.369**, against a median of 0.503 for the other similarity members. It is not a better
model, it is a differently wrong one, which is the only thing a rank blend can convert
into score. The docstring's claim that Avalon "tends to disagree with ecfp rather than
duplicate it" is now a measured number rather than an argument.

Two details worth carrying forward. The SVC form earns its place and the kNN form does
not (`svc_avalon` 0.6116 against `knn_avalon` 0.5864; the selector picks svc_avalon x4
across five folds and never picks knn_avalon). And CYP2D6's greedy AUC actually fell
slightly, 0.6410 to 0.6371, while MCC rose 0.05: at a 7% cut rate only the very top of the
ranking is scored, so AUC is the wrong summary for that endpoint and the rate-matched MCC
is the one to read.

## Where this leaves the gate

| | macro MCC |
|---|---|
| equal-weight blend of 20, blind-rate cuts | 0.2663 |
| per-endpoint selection, 20 members | 0.2898 |
| per-endpoint selection, 22 with Avalon | **0.3130** |
| standing best, actually scored | **0.3325** |

Still 0.0195 short locally, and the only local-to-board anchor we have (OOF 0.2524 scoring
0.285, a +0.033 offset) is a single point of the kind this project has been wrong about
three times. So this does not clear the gate yet -- but two structural changes have now
taken 0.047 of macro MCC out of the same member pool without a single new model family,
and the one that worked did so by being decorrelated rather than by being good.

That names the next experiment precisely: **more fingerprint families on CYP2D6, chosen
for disagreement rather than accuracy.** RDKit path, topological torsion, pattern, layered,
and Avalon at 1024 bits -- which is the configuration `train_tdi_chemprop.py` actually
benchmarked at 0.6043, while `cypkit/features.py` builds 2048. Each costs seconds. The
test is not their AUC; it is their correlation with knn_fcfp.
