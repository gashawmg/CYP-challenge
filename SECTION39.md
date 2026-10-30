--------------------------------------------------------------------------------
Section 39  A HELD-OUT SET THAT IS AS EASY AS THE BLIND SET, AND IT PREDICTS THE
            BOARD BETTER THAN CROSS-VALIDATION (2026-09-26)
--------------------------------------------------------------------------------

Gashaw: hold out compounds structurally similar to the blind set and their score tells us
the blind performance directly. Right in spirit, and WRONG AS LITERALLY STATED in a way
that makes the instrument read backwards.

THE TRAP. A blind compound B is easy because its nearest training neighbour T sits at
Tanimoto 0.70 and the model learned from T. Put T in the holdout and T is no longer in
training -- so when T is scored, its own nearest AVAILABLE neighbour might be 0.50. T has
been graded as a harder compound than B will ever be, by removing exactly the neighbour
that made it easy. Selecting on "similar to the blind set" STRIPS the blind set's
advantage. It is the Section 16 defect with the sign flipped: a validation set that is too
hard, again.

THE QUANTITY TO MATCH is not similarity to the blind set. It is

    NN(holdout member -> REMAINING training set) ~ NN(blind compound -> FULL training set)

as distributions, which is measurable on both sides.

THE COUPLING AND ITS FIX. Removing compounds changes the remaining neighbours of the
others: hold out two close analogues together and each destroys the other's nearest
neighbour. scripts/build_blindlike_holdout.py therefore requires the holdout to be
INTERNALLY DISPERSED -- no two members above --min-sep -- and then VERIFIES by recomputing
every member's true NN against the final training set.

THE RESULT, 394 compounds at --min-sep 0.55, stratified on label:

    NN bin        blind %   holdout %    diff
    0.45-0.50        3.9%        5.1%   +1.2%
    0.50-0.55       23.5%       20.6%   -2.9%
    0.55-0.60       28.3%       27.7%   -0.6%
    0.60-0.65       24.3%       23.1%   -1.2%
    0.65-0.70        9.5%        9.6%   +0.2%
    0.70-0.80        8.8%        8.9%   +0.1%

    median NN   blind 0.587   holdout 0.583
    total absolute histogram difference   4.7%
    NN lost to co-removal   mean +0.0048   (the dispersion rule holds)
    label coverage   train mean 3.380 sd 0.733   holdout mean 3.339 sd 0.720

AND IT PREDICTS THE BOARD. Same LightGBM, same features, two yardsticks -- ordinary 5-fold
out-of-fold against train-excluding-holdout scored on the holdout:

    endpoint   5-fold OOF   HOLDOUT    BOARD   CV ratio   HO ratio   n_holdout
    CYP1A2         0.3608    0.4954   0.8064      2.24       1.63        61
    CYP2C9         0.4068    0.4036   0.8454      2.08       2.09        84
    CYP2D6         0.2924    0.5389   0.5190      1.77       0.96        73
    CYP3A4         0.7240    0.7451   0.8327      1.15       1.12       248

    mean |ratio - 1|:   5-fold CV 0.810   ->   blind-like holdout 0.469

A 42% REDUCTION IN INSTRUMENT ERROR, and CYP2D6 -- the endpoint that defeated every
previous instrument, reading 4.37x low in Section 14 and costing a rank-9-to-36 submission
in Section 24 -- lands at 0.96. That is the closest any local yardstick has come on it.

WHAT IT DOES NOT FIX. CYP2C9 is unchanged (2.08 -> 2.09), exactly as Section 32 predicted:
similarity is an ANTI-proxy for difficulty there, because its high-similarity rows are
dense analogue series full of activity cliffs. Two independent constructions now agree on
that, so it is a property of the CYP2C9 data and not of either instrument.

THE HARD CEILING ON SIZE, measured rather than assumed. Per-endpoint the holdout is only
61 / 84 / 73 / 248 rows, and CYP1A2 at n=61 carries a Spearman SE near 0.10. Enlarging it
breaks the match, because the training set simply does not contain many dispersed compounds
at high NN:

    --n 400 --min-sep 0.55   394 kept   median 0.583   histogram diff   4.7%
    --n 600 --min-sep 0.55   512 kept   median 0.571   histogram diff  12.7%
    --n 900 --min-sep 0.65   850 kept   median 0.522   histogram diff  35.8%
    --n 900 --min-sep 0.75   871 kept   median 0.510   histogram diff  43.4%

Relaxing dispersion to buy size walks straight back into the trap: co-removal loss climbs
from +0.005 to +0.087 and the holdout becomes HARDER than the blind set. 394 is close to
the ceiling this training set supports. Use it as a coarse but honest instrument -- it can
separate a large effect from nothing, and cannot rank two similar models.

STANDING USE. Keep artifacts/holdout_blindlike.csv FIXED across experiments; re-selecting
it per model turns it back into a tuning surface. Exclude those Molecule_Names from
training, report the holdout score beside the out-of-fold score, and prefer the holdout
wherever the two disagree -- except on CYP2C9, where neither is trustworthy and the
reweighted yardstick of Section 32 remains the fallback.
