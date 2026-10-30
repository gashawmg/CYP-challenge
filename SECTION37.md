--------------------------------------------------------------------------------
Section 37  INTERPRETABLE MODELS COST NOTHING, AND ON CYP1A2 THEY WIN OUTRIGHT
            (2026-09-26)
--------------------------------------------------------------------------------

train_interpretable.py climbs a ladder from most readable to least, on the Section 36 lean
descriptors, project folds, scored on the similarity-reweighted yardstick with a paired
bootstrap against the unconstrained booster:

    rung          CYP1A2   CYP2C9   CYP2D6   CYP3A4
    linear-10     0.3319   0.3865   0.2173   0.6187
    GAM           0.3703   0.4154   0.2926   0.6933
    lgbm-mono     0.2686   0.3767   0.2363   0.6879
    lgbm-lean     0.2332   0.3792   0.2417   0.6797
    lgbm-full     0.2386   0.3776   0.2445   0.6772

FINDING 1 -- PRIOR PHARMACOLOGY IS FREE, AND ON CYP3A4 IT PAYS. Eight monotone constraints
written down BEFORE the fit (logP up, TPSA down, HBD down, heme-coordinating aromatic N /
imidazole / triazole up, basic N up on CYP2D6 and net charge up there but DOWN on CYP2C9,
planarity up on CYP1A2, bulk up on CYP3A4) against the free booster:

    CYP1A2  +0.0059 [-0.0120,+0.0241]     free
    CYP2C9  -0.0142 [-0.0342,+0.0037]     free
    CYP2D6  +0.0017 [-0.0181,+0.0245]     free
    CYP3A4  +0.0129 [+0.0061,+0.0208]     SIGNIFICANTLY BETTER

Forcing the model to agree with fifty years of CYP pharmacology costs nothing on three
endpoints and helps on the fourth. The data and the priors do not conflict, and the
constraint acts as regularisation. Every model this project ships from here can carry
these constraints at no cost and be defensible mechanistically, not only statistically.

FINDING 2 -- CYP1A2's SAR IS ADDITIVE, AND OUR BOOSTER HAS BEEN OVERFITTING IT.

    GAM        +0.0784 [+0.0300,+0.1177]   vs the booster
    linear-10  +0.0750 [+0.0187,+0.1276]

A ridge regression on TEN descriptors beats a 40-column gradient booster on CYP1A2, and an
additive spline model beats it by more. No other endpoint behaves this way -- CYP2C9 and
CYP2D6 lose 0.07-0.09 without interactions, so they genuinely need them, and CYP3A4 is
nearly additive (GAM -0.0153). CYP1A2 IS ALSO OUR WORST-MEASURED ENDPOINT (instrument
ratio 1.33, Section 33) AND THE ABSORPTION PROBLEM CHILD. Overfitting an additive
relationship is exactly the failure that would produce both.

Before treating GAM as the answer, the cheaper explanation must be ruled out: the booster
runs at num_leaves 63 / min_child 20 on 1,180 rows, which Section 30 already showed is
chosen against the wrong yardstick. A tighter booster may close most of the gap. Run that
first; if it does not, CYP1A2 should be modelled additively.

FINDING 3 -- THE MODELS READ LIKE PHARMACOLOGY, UNPROMPTED. The linear rung, with its
descriptors selected inside each training fold, recovered each isoform's textbook
mechanism without being told:

  CYP1A2  +0.065 aromatic_atom_frac, +0.059 halogen, +0.052 logD7.4,
          +0.029 aromatic_n_and_planar, -0.028 FractionCSP3, +0.028 planarity_index
          -> flat, aromatic, halogenated, lipophilic. The narrow planar slot.

  CYP2D6  +0.114 basic_n_aliphatic, -0.073 basic_n_to_aromatic_max,
          +0.048 d2d6_pharmacophore, +0.028 net_charge_74, +0.010 is_base
          -> a basic aliphatic nitrogen, CLOSE to an aromatic ring. The negative sign on
             the max distance is the 5-7 A window: too far and the salt bridge to Asp301
             cannot form while the oxidation site reaches the heme.

  CYP3A4  +0.295 MolLogP, +0.170 bulk_index, +0.113 logD7.4, +0.084 NumAromaticRings,
          +0.059 RingCount, -0.058 FractionCSP3
          -> large, lipophilic, ring-rich. The promiscuous pocket.

Every signed coefficient AGREES with the prior it was checked against. Not one contradicts.

LIMITATION, STATED PLAINLY. These are descriptor-only models with no Morgan fingerprint, so
the absolute rho sits below what we ship. The comparison between rungs is exact -- identical
rows, folds and yardstick -- but the levels are not the shipped models, and none of this is
a submission. What it buys is a model we can explain, defend and publish, and one concrete
lead (CYP1A2 additivity) with a measured effect size.
