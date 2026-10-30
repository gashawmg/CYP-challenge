--------------------------------------------------------------------------------
Section 42  THE NEW MEMBER IS SELECTED EVERYWHERE AND STILL NOT ENOUGH (2026-09-27)
--------------------------------------------------------------------------------

cp_lean -- the per-target chemprop retrained with the fold-aware pseudo-label map
(Section 34) and the lean descriptor block (Section 36) -- registered in MEMBERS and the
blend search re-run. Section 32 closed RE-WEIGHTING the members we had; a genuinely new
member is the case it did not cover.

IT IS SELECTED ON EVERY ENDPOINT AND ON ESSENTIALLY EVERY YARDSTICK:

    CYP1A2  hist x sim   cp_lean+mh+unimol+v6+v7            (87% stable)
    CYP2C9  histogram    cp_lean+mh+unimol+v5_rest+v6       (53%)
    CYP2D6  plain        cp+cp_lean+mh2+unimol+v6           (33%)
    CYP3A4  hist x sim   cp_lean+mh+v6+v7                   (47%)

Four of four. That is the clearest evidence yet that the lean descriptors produce a
genuinely useful member rather than a reshuffle of the same information.

PRICED HONESTLY IT IS STILL NOT A SUBMISSION:

    endpoint  yardstick   ratio   in-sample        OUT-OF-SAMPLE          verdict
    CYP1A2    hist x sim  1.088   +0.0191   +0.0153 [-0.0086,+0.0321]     hold
    CYP2C9    histogram   1.053   +0.0164   +0.0145 [+0.0006,+0.0259]     KEEP
    CYP2D6    plain       1.036   +0.0067   +0.0006 [-0.0178,+0.0166]     hold
    CYP3A4    hist x sim  1.000   +0.0217   +0.0214 [+0.0117,+0.0319]     KEEP

    summed out-of-sample local rho   +0.0359
    x 14% measured ensemble transfer    +0.0050 blind rho
    x exchange rate / 4                 -0.0015 macro ST-RAE
    board paired resolution             +-0.0067

A fifth of the board's own noise. HOLD.

THE BAR, STATED AS A NUMBER SO FUTURE WORK CAN BE AIMED AT IT. Inverting the same chain,
a blend change needs roughly +0.160 of SUMMED out-of-sample local rho to be worth a
submission slot:

    Section 32, old members only     +0.0169    11% of the bar
    today, with cp_lean              +0.0359    23% of the bar

Progress doubled, and it is still a quarter of the way. This is the honest scale of the
problem: re-weighting and one improved member cannot get there. Section 41 measured
+0.0717 on CYP1A2 and +0.0514 on CYP2D6 from 3D descriptors at MEMBER level. If even half
of that survives into chemprop and then into the blend, it is the right order of
magnitude. Nothing else on the table is.

TWO HOUSEKEEPING FACTS. artifacts/nnsim.csv did not exist on disk -- every script that
needed it had been recomputing the blind similarity profile inline. It is now generated
once (train: NN to the other training compounds, median 0.450; blind: NN to the full
training set, median 0.587). And `unimol_st` remains absent everywhere, so the optional
member is skipped rather than silently dropped from the baseline.

STANDING POSITION. pIC50 0.4833 rank 40, TDI 0.3312 rank 32, last submitted 14-15 Sept.
No candidate on either track. The next submission waits on per-endpoint descriptor specs
with 3D in the chemprop path (Section 41), not on any further arrangement of what exists.
