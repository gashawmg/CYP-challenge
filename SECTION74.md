# Section 74 -- mhtdi5 confirmed. And our best model has never been in a submission.

## 1. The five-seed pair is decisive, and the single-seed pair was not

`oof_mhens3` against `oof_mhtdi5`, same trainer, same seeds, same `--descriptor_spec
shared`, differing only in the CYP3A4 TDI head. Paired bootstrap, 800 resamples, both arms
placed optimally first:

| endpoint | d rho | 95% CI | P(worse) | d placed | 95% CI | P(worse) | arm corr |
|---|---|---|---|---|---|---|---|
| CYP1A2 | +0.0009 | [-0.0118, +0.0127] | 44.2% | -0.0020 | [-0.0137, +0.0109] | 62.1% | 0.9573 |
| CYP2C9 | +0.0092 | [-0.0028, +0.0193] | 6.9% | +0.0105 | [-0.0050, +0.0251] | 8.4% | 0.9673 |
| CYP2D6 | +0.0076 | [-0.0062, +0.0211] | 13.5% | +0.0083 | [-0.0021, +0.0176] | 4.2% | 0.9481 |
| CYP3A4 | **+0.0070** | **[+0.0026, +0.0112]** | **0.1%** | **+0.0131** | **[+0.0058, +0.0206]** | **0.0%** | 0.9844 |
| **MACRO** | | | | **+0.0075** | **[+0.0022, +0.0127]** | **0.4%** | |

**rho improved on all four endpoints, and three of them have no TDI head.** Only CYP3A4
got one. The other three gained through the shared trunk -- 1,249 extra CYP3A4 compounds
reshaping the encoder that all four heads read from. That is precisely what the multihead
architecture is for, and it is the first time this project has observed the effect
directly.

CYP3A4 is now confirmed a THIRD time, by a third route: surrogate rows (+0.0135), a single
-seed head (+0.0129), a five-seed head (+0.0070 rho / +0.0131 placed), every interval
excluding zero.

THE ENSEMBLING DID WHAT seed_ensemble.py SAID IT WOULD. Arm-to-arm correlations are now
0.948-0.984, ABOVE the r = 0.927 single-run noise floor; in the single-seed pair
(Section 71) three of four sat BELOW it. Averaging five seeds dropped the floor far enough
for the A/B to resolve, exactly as that script's docstring promised. Two endpoints that
read as losses at one seed (CYP1A2 -0.0255, CYP2D6 -0.0138) were noise, and both are now
neutral-to-positive.

## 2. The finding that matters more: what we actually ship

`scripts/build_blend4.py` defines the shipped column:

    MEMBERS = {"v6": "pred_st_v6.csv", "mh": "pred_st_mh.csv", "cp": ..., "unimol": ...}

`mh` is `pred_st_mh.csv`, dated **7 September**. Its out-of-fold partner, `oof_st_mh.csv`,
is **263 rows** -- the fold-0 smoke stub. So the multihead member inside every pIC50
submission we have made is a single-seed model from before descriptors, before seed
ensembling, before the TDI head, and **its out-of-fold performance has never been measured
on the full pool at all.** `mhens3`, our best local column for six weeks, is registered in
`pic50_blend_search.py` as a member but is not in `build_blend4.py`.

## 3. How much the swap is worth, and why it is so little

Equal z-average of (v6, cp, unimol, X), placed, varying only X:

| 4th member | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | MACRO |
|---|---|---|---|---|---|
| single-seed multihead | 0.7079 | 0.5040 | 0.8011 | 0.3974 | 0.6026 |
| mhens3 (5 seeds) | 0.7096 | 0.4979 | 0.8024 | 0.3943 | 0.6010 |
| mhtdi5 (5 seeds + TDI) | 0.7099 | 0.4966 | 0.8013 | 0.3921 | **0.6000** |

Single-seed to mhtdi5 inside the blend: **+0.43%**, blind 0.4737 -> **0.4716**. The member
-level gain was +1.19%; the blend returns about a third of it. An equal four-member average
dilutes any single member's improvement roughly fourfold, because the other three are
unchanged and outvote it.

That ratio is the thing to remember: **in this blend, a member-level gain of X is worth
about X/4 on the board.** It reprices every member-level result this project has produced.

(The proxy caveat: `mh1seed` above is THIS session's single-seed multihead, not the Sept-7
file, whose OOF cannot be measured. It is the closest available stand-in, not the article.)

## 4. Re-choosing the composition looks better than it is

Scanning all 26 compositions of {v6, cp, unimol, mhens3, mhtdi5}: the best is
`v6+unimol+mhtdi5+mhens3` at 0.5917 against 0.6003 for the shipped set with the new member,
an apparent +0.0086. Do not act on it. Twenty-six compositions were scored on the rows that
then chose the winner, and every top composition includes BOTH multihead variants, which
correlate at 0.95+ -- that is not a better composition, it is the multihead family
silently up-weighted to half the blend. Fitted blend weights have already lost on this
board once.

## 5. Status

Not a submission. +0.43% macro is 0.4737 -> 0.4716, about one rank, well inside the
"no small modifications" rule. `submission_d6probe.csv` stays.

What is banked: a confirmed model improvement, a measured blend-dilution ratio, and the
knowledge that the shipped blend carries a six-week-old member whose OOF was never checked.
The next lever is not a fifth member -- it is that three of the four current members
(v6 Sept 5, unimol Sept 2, mh Sept 7) predate everything learned since, and improving one
member returns a quarter of its gain while improving all four returns all of it.

## 6. Stale baseline, fourth instance, now fixed

`pic50_blend_search.py` carried `BLIND_ST` with CYP2D6 = 0.5957 and `HELD_NAME = "cand1"`.
The incumbent has been `d6probe` (0.4737) since order 17, whose CYP2D6 is **0.5621**. Every
candidate this script graded was being compared against a CYP2D6 column 0.0336 worse than
the one we actually hold. Corrected, with a comment naming the three earlier instances
(Sections 51, 67, 68) and an instruction to check `artifacts/submissions.csv` first.
