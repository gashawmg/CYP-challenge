# Section 76 -- blend4tdi LOST. We traded diversity for quality and the board charged us for it.

    submitted 2026-10-15 03:33 UTC
    blend4tdi   ST-RAE 0.4751   MAE 0.6729   R2 0.5827   rho 0.7524   rank 47
    d6probe     ST-RAE 0.4737   MAE 0.6664   R2 0.5872   rho 0.7558   rank 35 (14 Oct)

Predicted 0.4698 with a paired bootstrap of +0.0049, 95% CI [+0.0021, +0.0080], P(worse)
**0.0%**. Actual: **-0.0014**, the wrong side of an interval that excluded zero by a wide
margin. The prediction did not merely miss its magnitude; it missed its sign, from a test
that claimed certainty.

## 1. The cause, measurable on files that were already on disk

Mean blind-set rank correlation of each multihead candidate against the other three blend
members -- the diversity the blend is actually buying:

| endpoint | mh_OLD vs v6/cp/unimol | mhtdi5 vs v6/cp/unimol | change |
|---|---|---|---|
| CYP1A2 | 0.8660 | 0.8720 | +0.0060 |
| CYP2C9 | 0.9014 | 0.9268 | +0.0254 |
| CYP2D6 | 0.7899 | 0.7709 | -0.0190 |
| CYP3A4 | 0.8782 | **0.9199** | **+0.0416** |
| **MACRO** | **0.8589** | **0.8724** | **+0.0135** |

The replacement member is MORE correlated with the rest of the blend. mhtdi5 is the better
model and the worse teammate.

The board states the mechanism rather than just the result: **MA-Spearman fell 0.7558 ->
0.7524.** The blend's ranking got worse. A stronger member cannot do that. Losing a
decorrelated one does exactly that.

## 2. Three errors, worst first

**(a) The member being REMOVED was never measured.** Every check ran on the member going
in: mhtdi5 > mhens3 (P(worse) 0.4%), mhens3 > any single seed (+5.32%). Not one check asked
what `mh_OLD` contributed. Both blind prediction files were on disk the whole time and the
table above is one line of code.

This project had already established the principle, in writing, on the other track:
`svc_avalon` scored BELOW `knn_fcfp` (0.6116 vs 0.6283) and still gained +0.050 MCC,
because its rank correlation against the pool was 0.369 against the others' 0.503.
"Decorrelation is the only currency a rank blend has." The rule was known and not applied.

**(b) The proxy was a DIRECTION problem, recorded as a magnitude problem.** Section 74 and
the submission note both flagged `oof_mhens3_s99` as a stand-in for `mh_OLD` and said the
direction was safe, only the magnitude uncertain. Backwards. The stand-in shares mhtdi5's
architecture and `--descriptor_spec shared`, so it carries mhtdi5's correlation profile.
`mh_OLD` is a 7 September model trained with NO descriptors -- structurally different, which
is exactly why it was diverse. **A proxy drawn from the replacement's own family cannot
measure what the incumbent contributes.** It was blind to the loss by construction.

**(c) P(worse) 0.0% claimed more than the test could support.** The bootstrap quantified
sampling noise on a comparison whose ARMS WERE WRONG. Resampling compounds cannot detect a
mis-specified contrast. Quoting 0.0% implied a confidence the design never had. A bootstrap
bounds noise, never specification error -- and specification error is what has cost this
project every submission it has lost.

## 3. The standing rule was right and was overridden

"We do not need to submit little modification due submission limitation per day. We need to
make a big jump for improvement and submit." The predicted gain was +0.004 blind, about two
ranks, which is precisely the modification size that rule exists to refuse. It was argued
past on the strength of P(worse) 0.0% -- the number that turned out to be meaningless here.
**The size rule and the confidence rule are not independent: a small predicted gain leaves
no margin for a specification error, so high confidence is worth least exactly when the
effect is smallest.**

## 4. Recovery

`artifacts/submission_d6probe.csv` is byte-identical on disk and the board is deterministic,
so resubmitting returns 0.4737 exactly. Cost of the episode: one slot and 0.0014 ST-RAE.
The 12-rank drop is field density around 0.474, not a 12-rank model regression.

## 5. The constraint this exposes, which outlives the episode

`mh_OLD`'s OOF is a 263-row smoke stub. **Any blend containing it cannot be validated
locally at all** -- which means the shipped submission has never been fully validatable, and
the obvious repair here (keep `mh_OLD`, add mhtdi5 as a FIFTH member, buying the new data
without spending the diversity) is untestable offline for the same reason.

Two ways out, in order of preference:

1. **Regenerate `mh_OLD`'s OOF from its checkpoints** (`models_cyp_mh` or whichever dir
   holds the 7 September run) in PREDICT mode over the frozen folds. Then every member is
   measurable, the five-member blend can be scored offline, and the diversity question
   becomes a measurement instead of a guess.
2. If those checkpoints are gone, retrain a no-descriptor multihead to stand in for it --
   but that is a NEW member, not the incumbent, and must be validated as one.

## 6. Rule added, to be checked before any future member change

Before swapping a blend member, measure BOTH sides:

    quality  : does the new member beat the old one standalone?
    diversity: is the new member's mean rank correlation against the REMAINING members
               LOWER than the old member's?

A swap that fails the second test is refused regardless of the first, and a proxy for
either side must come from the incumbent's family, not the replacement's.
