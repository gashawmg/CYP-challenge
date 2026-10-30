--------------------------------------------------------------------------------
Section 43  THE 3D SPEC IN CHEMPROP: ONE BIG WIN, TWO REFUTATIONS, AND A NOISE FLOOR
            THIS PROJECT HAD NEVER MEASURED (2026-09-27)
--------------------------------------------------------------------------------

Section 41 set a per-endpoint descriptor table from LightGBM arms. Run through the actual
model, only one line of it survives.

    endpoint   spec run   LEAN   SPEC-LEAN   holdout SPEC-LEAN   verdict
    CYP1A2      0.5228   0.4475    +0.0753        +0.1224        3D KEPT
    CYP2C9      0.5137   0.5735    -0.0598        -0.0067        CONTROL (see below)
    CYP2D6      0.3687   0.5204    -0.1518        -0.1057        3D REMOVED
    CYP3A4      0.7866   0.8033    -0.0167        -0.0298        wide+3D REMOVED

CYP2C9 IS A CONTROL AND I DID NOT PLAN IT THAT WAY. Its spec is lean-only, which is exactly
what the previous run used -- 38 identical columns, same data, same code. The only
difference between those two CYP2C9 columns is the training run itself. They correlate at
PEARSON r = 0.927, not 1.0, and the reweighted yardstick reports a -0.0598 gap between
them.

THAT IS A NOISE FLOOR, AND IT CHANGES HOW EVERY CHEMPROP COMPARISON HERE SHOULD BE READ.
A single pair of chemprop runs cannot resolve a feature effect smaller than roughly 0.06 on
the reweighted yardstick. Which forces an honest correction to Section 40: its LEAN vs FULL
gaps were +0.0315 / +0.0234 / +0.0493 / -0.0028, ALL AT OR BELOW THIS FLOOR. The direction
was right on four of four and it matched a prediction made in advance from LightGBM, so the
conclusion stands -- but I stated the per-endpoint magnitudes with more confidence than one
pair of runs can support. They should have carried this caveat and did not.

THE TWO YARDSTICKS DISAGREE ABOUT THE FLOOR ITSELF, and the control tells us which to
believe. On CYP2C9, where the true model difference is near zero by construction, the
holdout reports -0.0067 and the reweighted reports -0.0598. The reweighted estimator is
adding about 0.06 of its own noise -- unsurprising at an effective sample size of 118-132
(Section 32). THE HOLDOUT IS THE BETTER INSTRUMENT FOR CHEMPROP COMPARISONS. Read against
its own control of 0.007:

    CYP1A2  +0.1224   18x the control   REAL, and large
    CYP2D6  -0.1057   16x the control   REAL, and bad
    CYP3A4  -0.0298    4x the control   real, small, negative

CYP1A2 IS THE RESULT. Section 41 predicted +0.0717 from LightGBM; chemprop delivered
+0.0753 reweighted and +0.1224 on the holdout. Second time a prediction recorded in advance
has held. The endpoint with the worst absorption, the worst instrument ratio and an
additive SAR (Section 37) was missing the one property its narrow planar slot selects on.

CYP2D6 IS THE REFUTATION, AND IT IS THE INTERESTING ONE. 3D helped there in LightGBM
(+0.0514) and hurts badly in chemprop (-0.106 holdout). Section 41 flagged exactly this
endpoint as "promising, the two yardsticks disagree" and hedged -- correctly. The likely
reason: a D-MPNN already encodes the molecular topology that fixes the basic-N-to-aromatic
geometry, so eight conformer columns add variance without information. A feature that helps
a booster is not a feature that helps a graph network, and this project has now measured
that twice in opposite directions.

REVISED SPEC, now measured in the model that will ship it:

    CYP1A2   lean + 3D
    CYP2C9   lean
    CYP2D6   lean
    CYP3A4   lean

WHAT TO DO ABOUT THE NOISE FLOOR. Every future chemprop feature claim needs either multiple
seeds per arm or a gap that clears the holdout control by a wide margin. The cheap version
is to keep using CYP2C9 -- or whichever endpoint the spec leaves unchanged -- as an
in-run control, which costs nothing and calibrates the comparison for free.
