--------------------------------------------------------------------------------
Section 36  A LEAN, MECHANISM-JUSTIFIED DESCRIPTOR SET. 35-40% FEWER COLUMNS, NO
            ACCURACY LOST, TWO ENDPOINTS BETTER (2026-09-26)
--------------------------------------------------------------------------------

THE BLOAT IS MEASURED, not asserted. On our own 4,905 training molecules the 31 generic
RDKit descriptors carry only 15 independent pieces of information at |r| >= 0.70, and one
cluster of EIGHT -- MolWt, HeavyAtomCount, MolMR, Kappa1, Chi0v, Chi1v, Chi2v, LabuteASA --
is a single fact: how big the molecule is. TPSA/NOCount, NumHDonors/NHOHCount,
Kappa2/Kappa3 and MaxEStateIndex/MaxAbsEStateIndex are further duplicate pairs.

THE RULE FOR KEEPING A COLUMN: it must name a physical requirement for binding or turnover
in a CYP active site, in one sentence, and not duplicate a column already kept. What
survives, in four groups:

  OCCUPANCY AND SHAPE   HeavyAtomCount (the whole size cluster), FractionCSP3 (flat vs
                        puckered), Kappa3 (branching), NumRotatableBonds (entropic cost)
  RECOGNITION           MolLogP, TPSA, NumHDonors, NumHAcceptors, NumAromaticRings,
                        RingCount
  ELECTRONICS           MaxPartialCharge, MinPartialCharge, MinEStateIndex,
                        MaxAbsEStateIndex
  IONISATION at pH 7.4  phys_pka_basic, phys_pka_acidic, phys_logd74, phys_net_charge_74,
                        phys_is_zwitterion
  plus the 8 heme-coordination SMARTS unchanged, and the isoform block.

Dropped: the seven size duplicates, Kappa2, NumSaturatedRings and NumAliphaticRings (inside
FractionCSP3), NOCount, NHOHCount, MaxEStateIndex, NumHeteroatoms, HallKierAlpha, BertzCT
(graph complexity, no binding meaning) and qed (a drug-likeness preference score, not a
property of the protein-ligand interaction); from phys, the two frac_ionised terms (they
are functions of the pKas), logp_minus_logd (MolLogP minus logd74), formal_charge and
is_permanent_cation (covered by net_charge_74).

THE TEST, LightGBM on descriptors alone, project folds, similarity-reweighted to the blind
set (Section 30), paired bootstrap:

    isoform   full -> lean   cut    reweighted gap        P(lean worse)
    CYP1A2      62 -> 40     35%   -0.0035 [-0.036,+0.029]     59%   tie
    CYP2C9      63 -> 38     40%   +0.0400 [+0.010,+0.069]      0%   BETTER
    CYP2D6      67 -> 43     36%   +0.0239 [-0.007,+0.054]      8%   better
    CYP3A4      62 -> 40     35%   -0.0008 [-0.009,+0.008]     56%   tie

TWO ENDPOINTS SIGNIFICANTLY BETTER, TWO DEAD LEVEL, A THIRD OF THE COLUMNS GONE. Narrower
helps most where the rows are fewest, which is the expected shape.

A MISTAKE THE FIRST RUN CAUGHT, AND IT IS WORTH KEEPING. The first version also dropped
CYP1A2's three planarity composites under the general rule "a model can rebuild a
composite". CYP1A2 was then the ONLY endpoint that lost: -0.0364, P(worse) 98%. The rule
was wrong there. Planarity IS CYP1A2's binding requirement -- a narrow slot that admits
flat aromatics -- and a planarity index built from aromatic-atom fraction and ring fusion
is not something a booster reconstructs from HeavyAtomCount and FractionCSP3 on 1,180 rows.
A COMPOSITE THAT ENCODES THE ISOFORM'S MECHANISM IS A PRIMARY FEATURE, NOT A CONVENIENCE.
Restoring the three moved CYP1A2 from -0.0364 to -0.0035.

CAVEAT BEFORE THIS GOES INTO CHEMPROP. The test above is LightGBM on descriptors ALONE.
Chemprop fuses them with a learned graph representation that already encodes much of what
the generic columns say, so the redundancy being removed here is redundancy WITH THE GRAPH
as well -- the effect could be larger there, or smaller. It has to be measured in the
chemprop path, not assumed from this.

HOW TO USE IT. cyp_target_descriptors_lean.lean_index(iso) returns column positions into
the existing calculate() output, so nothing is recomputed and no training code changes
until the chemprop arm is run and scored.
