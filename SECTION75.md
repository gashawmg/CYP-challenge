# Section 75 -- Yes, the models are better. The blend keeps 12% of it, and here is why.

Written in answer to a fair question: after all this training, do we have better
predictions? The models are measurably better. The submission is barely. Both halves of
that are now quantified, and the reason is structural rather than a failure of any run.

## 1. The members really did improve

| change | member level (placed macro) | relative |
|---|---|---|
| single seed -> 5-seed average | 0.6600 -> 0.6249 | **+5.32%** |
| 5-seed -> 5-seed + CYP3A4 TDI head | 0.6249 -> 0.6172 | **+1.19%**, P(worse) 0.4% |
| both together | 0.6600 -> 0.6172 | **+6.48%** |

None of this is in doubt. The TDI head carries a paired-bootstrap interval excluding zero
and has now replicated three times across two model families.

## 2. The blend keeps 12% of it

The identical change, measured inside the four-member equal z-blend that actually produces
our submission:

| 4th member of (v6, cp, unimol, X) | blend placed macro |
|---|---|
| X = single-seed multihead | 0.6049 |
| X = 5-seed | 0.6010 |
| X = 5-seed + TDI | **0.6000** |

    member level, 1 seed -> 5 seeds :  +5.32%
    same change inside the blend    :  +0.64%
    fraction surviving              :  12%

Everything this session produced, carried fully into the blend, is 0.6049 -> 0.6000,
**+0.81%**: blind 0.4737 -> about 0.4699. Two ranks.

## 3. Why -- and this is the part worth keeping

**Seed ensembling removes variance. The blend already removes variance.** Averaging four
decorrelated models is itself a variance reduction, so it had already cancelled most of the
independent noise that averaging five seeds of one member cancels again. The second pass
over the same noise returns almost nothing. The TDI head is the same story in a subtler
form: its gain came through a better-conditioned shared trunk, which is largely a variance
effect too.

What the blend CANNOT remove is error the members share. Measuring the members' errors
directly rather than their predictions:

| endpoint | mean pairwise ERROR correlation across v6 / cp / unimol / mhtdi5 |
|---|---|
| CYP1A2 | 0.618 |
| CYP2C9 | 0.571 |
| CYP2D6 | **0.698** |
| CYP3A4 | 0.605 |

**Roughly 60-70% of each member's error is shared with the others.** That fraction is
untouchable by averaging, by more seeds, by better heads, and by any further work on these
four models. It is the floor, and it is where the gap to the leaderboard lives.

The pairwise detail names the one direction that is working: `unimol` is the most
decorrelated member on every endpoint (0.503-0.598 against the others), while v6, cp and
mhtdi5 correlate 0.63-0.78 with each other. Three of our four members are 2D graph or
descriptor models that make the same mistakes. Unimol is the 3D one -- and it is the oldest
file in the blend, dated 2 September.

## 4. This also explains the whole session in one line

Every lever that worked -- seed ensembling, the TDI head, the multihead trunk -- was a
variance reduction. Every lever that failed -- monotone re-placement, interval weighting,
seed-spread gating, extremeness gating, uncertainty-shaped losses, CQR -- was an attempt to
re-spend information the pipeline already had. Neither kind attacks shared bias, and shared
bias is 60-70% of the error.

Section 1's absorption finding said the same thing from the leaderboard side: our rho leads
the neighbourhood and our conversion trails it, and the difference is ranking the expensive
compounds. The members agree about those compounds, and they are wrong about them together.

## 5. What to do

**Carry what we have, do not submit it.** `build_blend4.py` should point its `mh` slot at
`pred_mhtdi5.csv` instead of the 7 September `pred_st_mh.csv`. That is the correct state of
the pipeline and it is free. At +0.81% it is not a submission on its own, and the standing
rule holds.

**Then stop improving these four models.** Further variance reduction returns 12%. The next
member has to be wrong in a DIFFERENT WAY, and the error-correlation table says the
decorrelated direction is the 3D one, where our only member is six weeks old and never
ensembled. A refreshed and seed-ensembled 3D member, or a second one from a different 3D
representation, attacks the 0.57-0.70 floor rather than the 30-40% the blend has already
spent.

That is a real lever with a measured target, which is more than this session had when it
started. It is also honest that the five training runs bought two ranks and a map.
