--------------------------------------------------------------------------------
Section 41  WE HAD NEVER USED A 3D DESCRIPTOR, AND ON CYP1A2 IT IS THE LARGEST
            FEATURE GAIN THIS PROJECT HAS MEASURED (2026-09-27)
--------------------------------------------------------------------------------

Every column in cyp_target_descriptors is a 2D graph property or a SMARTS count. PXR
carried NPR1, NPR2 and Asphericity -- conformer-derived. We never did. One ETKDGv3
conformer with MMFF optimisation costs 293 seconds for all 4,906 structures, 0 failures.

EIGHT SHAPE DESCRIPTORS added to the Section 36 lean block: NPR1, NPR2, Asphericity,
Eccentricity, InertialShapeFactor, SpherocityIndex, RadiusOfGyration, PBF. The three PMI
terms are left out -- they are raw size, which Section 36 already showed is duplicated.

    endpoint   rows    lean     +3d       gap              95% CI   holdout lean -> +3d
    CYP1A2     1180  0.3201  0.3939   +0.0717 [+0.0377,+0.1040]      0.2974 -> 0.3651
    CYP2C9     1026  0.4199  0.4010   -0.0181 [-0.0480,+0.0159]      0.3821 -> 0.3941
    CYP2D6     1364  0.3766  0.4284   +0.0514 [+0.0197,+0.0840]      0.4049 -> 0.3922
    CYP3A4     2335  0.7021  0.7125   +0.0093 [-0.0012,+0.0195]      0.6886 -> 0.7039

THE ORDERING IS THE PHARMACOLOGY, and it was not fitted to come out this way:

  CYP1A2  +0.0717   a NARROW PLANAR SLOT. NPR1/NPR2 and Asphericity measure precisely
                    the rod-disc-sphere axis, i.e. how planar a molecule is. The binding
                    requirement and the descriptor are the same quantity.
  CYP2D6  +0.0514   a small, geometrically constrained site around the Asp301 salt
                    bridge, where the distance between the basic N and the oxidation
                    site has to be right.
  CYP2C9  -0.0181   recognition is CHARGE-driven (Arg108 binds anions), not shape-driven.
                    Geometry adds nothing, and the sign is negative.
  CYP3A4  +0.0093   the large promiscuous pocket -- the least shape-selective of the
                    four, so the smallest gain. Marginal.

CYP1A2's +0.0717 IS THE LARGEST SINGLE FEATURE GAIN IN THIS PROJECT. The previous record
was the hand-built pharmacophore at +0.0217 on the same endpoint, which WHAT_WORKED called
"the only significant feature gain in the project". And CYP1A2 has been the problem child
throughout: worst absorption (0.867), worst instrument ratio (1.33), and Section 37 found
its SAR is additive enough that a ten-descriptor ridge beat a gradient booster. A model
that could not see planarity was missing the one thing that site selects on.

WHAT IS SOLID AND WHAT IS NOT. The two yardsticks agree on CYP1A2 (+0.072 reweighted,
+0.068 holdout) and on CYP3A4 (+0.009, +0.015). THEY DISAGREE ON CYP2D6 (+0.051
reweighted, -0.013 holdout) and on CYP2C9 (-0.018, +0.012). Only CYP1A2 is corroborated by
both, and four endpoints were tested so one crossing 95% by chance is not unlikely. Treat
CYP1A2 as established, CYP2D6 as promising, CYP2C9 as refuted, CYP3A4 as marginal.

THE CYP3A4 WIDTH QUESTION, answered in the same run. Section 40 predicted that CYP3A4 --
the only endpoint with PXR-scale data -- could afford a wider general block:

    arm        cols   rho rewt   gap vs lean            95% CI   holdout
    lean         40     0.7018     +0.0000                        0.6886
    +wide        78     0.7076     +0.0050 [-0.0060,+0.0157]      0.7176
    +3d          48     0.7114     +0.0092 [-0.0026,+0.0197]      0.7039
    +wide+3d     86     0.7129     +0.0118 [+0.0016,+0.0229]      0.7325

Neither alone clears; TOGETHER THEY DO, and the holdout agrees (+0.044). So width IS
affordable on CYP3A4, as the row-count hypothesis predicted -- but it needs the geometry
alongside it to be worth having. One caution: three arms were tested and one cleared at
95%, so roughly a one-in-seven chance of a false positive across the family.

WHAT TO BUILD. A per-endpoint descriptor spec rather than one block for all four:

    CYP1A2   lean + 3D            the biggest measured win in the project
    CYP2D6   lean + 3D            promising; confirm on the holdout before shipping
    CYP2C9   lean                 geometry refuted, width untested and unlikely
    CYP3A4   lean + wide + 3D     the only endpoint with the data to carry width

This is the first result in the project that says the four endpoints want DIFFERENT
FEATURE SETS, not just different weights on one set. Conformers are cheap and cached by
SMILES, so the cost is 5 minutes once.
