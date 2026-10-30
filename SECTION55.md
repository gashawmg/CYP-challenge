# Section 55. Five more fingerprint families, and a line that ends here

Run date 2026-10-12. `--tag tdiB` added ten members to the pool -- kNN and SVC on RDKit
path, topological torsion, pattern, layered, and Avalon at 1024 bits -- taking it from 22
to 32. The stated gate was that CYP2D6's rate-matched MCC had to clear 0.1857. It reached
**0.1881**. That is inside the noise, and this line is finished.

---

## 1. The whole arc, on one scale

Macro MCC, blind-rate cuts, cut always taken out of fold:

| pool | CYP3A4 equal / per-endpoint | CYP2D6 equal / per-endpoint | macro equal / per-endpoint |
|---|---|---|---|
| tdiM, 20 members | 0.3993 / 0.4435 | 0.1332 / 0.1361 | 0.2663 / **0.2898** |
| tdiA, 22 (+Avalon) | 0.4051 / 0.4402 | 0.1419 / **0.1857** | 0.2735 / **0.3130** |
| tdiB, 32 (+5 families) | 0.3841 / 0.4379 | 0.1481 / 0.1881 | 0.2661 / **0.3130** |

Three changes, in order: per-endpoint selection bought **+0.0232**, Avalon bought
**+0.0232**, and five further fingerprint families bought **+0.0000**.

Paired bootstrap, tdiB against tdiA on identical folds:

| | difference | 95% CI | P(better) |
|---|---|---|---|
| CYP2D6 | +0.0023 | [-0.0461, +0.0498] | 0.534 |
| CYP3A4 | -0.0023 | [-0.0095, +0.0046] | 0.264 |

A coin flip on both endpoints.

## 2. The prediction held at the member level and still bought nothing

Section 54's addendum predicted that one or two new members would land where `svc_avalon`
did, that the rest would be noise, and that the SVC forms would beat the kNN forms. All
three held:

| family | kNN AUC | SVC AUC |
|---|---|---|
| avalon1k | 0.5702 | **0.6132** |
| pattern | 0.5717 | 0.6035 |
| layered | 0.5580 | 0.6001 |
| rdkpath | 0.5684 | 0.5827 |
| tt | **0.6114** | 0.6005 |

SVC wins four of five; topological torsion is the exception. And `svc_avalon1k` at 0.6132
and `knn_tt` at 0.6114 both sit exactly where `svc_avalon` did at 0.6116 -- two members
landing on target, as predicted.

The decorrelation was there too. On CYP2D6 the median rank correlation with `knn_fcfp`
across all 32 members is 0.372, and `svc_pattern` reaches 0.316 at AUC 0.6035 -- the exact
profile that made `svc_avalon` pay. The selector picks them: `svc_avalon1k` x6, `knn_tt`
x3, `svc_pattern` x2 across five folds.

**And the blend did not move.** They substitute rather than add -- `svc_avalon`, picked
five times in tdiA, is picked zero times in tdiB, and `svc_tan` drops from three to two.
The information these fingerprints share is the same information, re-enumerated. CYP2D6
TDI tops out around AUC 0.64 from structural similarity, and more notions of similarity do
not raise it.

So the Section 54 reading was right that decorrelation is what a rank blend converts into
score, and wrong to assume more of it was available. Avalon was not the first of a series.
It was the one that was missing.

## 3. An operational warning: the shipped blend is now worse than before

`blend_of()` applies equal weights across every pool member. On CYP3A4 that blend went
**0.4051 (22 members) to 0.3841 (32 members)** -- adding ten members that do not help that
endpoint actively diluted it, costing 0.021 of MCC. The per-endpoint selection is immune
(0.4402 to 0.4379) because it simply never picks them.

**So `run_tdi_ensemble.py --predict` on the 32-member pool would ship a WORSE CYP3A4
column than the 22-member pool did, and a worse one than the 20-member pool.** Any
submission built from this pool has to either restrict with `--learners` or go through
per-endpoint weights. This is the cost of a single global weight vector showing up as an
operational hazard rather than a missed opportunity.

## 4. Where the TDI track stands, and why it stops here

| | macro MCC |
|---|---|
| best local, per-endpoint selection | 0.3130 |
| standing best, actually scored | **0.3325** |

Still 0.0195 short locally after three structural changes, two of which worked. The only
local-to-board anchor is a single point (OOF 0.2524 scoring 0.285, a +0.033 offset) and
this project has been wrong about one-point extrapolations three times -- Sections 48, 50
and 53. The validator's shrinkage correction for CYP2D6 is a fourth such number and
Section 53 failed to find a mechanism for it.

What the evidence now says about each endpoint:

- **CYP3A4** is a solved model with a threshold problem. `lgb_clf` alone predicts 0.4576
  at the optimal cut against 0.4470 actually scored; thirty-one other members do not beat
  it. The remaining ~0.011 is rate placement, which `run_tdi_ensemble.py` already handles.
- **CYP2D6** is a discriminability problem that structural similarity cannot solve.
  AUC 0.6284 for the best single, 0.6435 for a selected blend of six, against the leader's
  blind 0.838. Twelve similarity members across seven fingerprint enumerations now agree
  on that ceiling.

Neither of those is closed by another evening of pool members. The pIC50 track has
`probe2` waiting on the slot and a measured CYP3A4 surrogate experiment behind it, so the
TDI track goes back in the queue until a second scored submission exists to anchor the
local-to-board mapping.

Recorded because a prediction that was right in every particular produced no gain, which
is a more useful thing to have written down than another small win.
