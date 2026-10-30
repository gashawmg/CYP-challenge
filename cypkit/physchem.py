#!/usr/bin/env python
"""Ionisation state at pH 7.4 — pKa, logD, fraction charged. RDKit only, no new deps.

WHY THIS EXISTS. Lipophilicity is the strongest physicochemical driver of CYP inhibition
in every survey of the endpoint, but the descriptor that carries it — cLogP — is the WRONG
one for CYP2D6, and CYP2D6 is where we are weakest (blind Spearman 0.507 against 0.79-0.85
elsewhere).

CYP2D6's pocket is anionic (Glu216 / Asp301) and its inhibitors are overwhelmingly basic
amines that are PROTONATED at pH 7.4. Protonation is exactly what separates logP from
logD: a compound with cLogP 4 and a basic nitrogen at pKa 10 has logD7.4 near 1.4, because
99.6% of it is charged. Published surveys put CYP2D6 inhibitors at mean logD7.4 ~0.57
while their cLogP averages ~4 — the two descriptors disagree by three and a half log units
on precisely this class. We were feeding the model only the one that disagrees.

Worse, `cypkit/features.py:_standardise` runs `rdMolStandardize.Uncharger()` before
featurisation, so a drawn quaternary or protonated nitrogen is neutralised before any
descriptor sees it. Everything here therefore works from the molecule AS DRAWN and never
uncharges.

WHAT THIS IS NOT. It is not a pKa predictor. Real pKa prediction needs a trained model
(ACD, Epik, MoKa) or a package we are not willing to install into a working conda env.
This is a curated SMARTS lookup: match the strongest ionisable group present and take a
literature-typical pKa for that class. It gets CLASS right (strong base / weak base /
acid / neutral / permanent cation) and gets individual compounds wrong by up to ~1.5 pKa
units. For a tree learner choosing splits that is enough, and `validate()` below states
the accuracy honestly against measured drugs rather than asserting it.

THE FEATURES.

    pka_basic          strongest basic pKa present (0 if none)
    pka_acidic         strongest acidic pKa present (14 if none)
    frac_base_ionised  Henderson-Hasselbalch, fraction protonated at 7.4
    frac_acid_ionised  fraction deprotonated at 7.4
    logd74             logP corrected for both equilibria
    logp_minus_logd    how much ionisation costs — the CYP2D6 signal in one number
    net_charge_74      expected net charge at pH 7.4
    formal_charge      charge AS DRAWN, which Uncharger would have erased
    is_zwitterion      basic and acidic centre both ionised
    is_permanent_cation quaternary N — charged at every pH, no equilibrium
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen

RDLogger.DisableLog("rdApp.*")

PH = 7.4
# Empirical ceiling on the logP->logD correction (ion-pair partitioning floor).
MAX_IONISATION_PENALTY = 3.0

# (name, SMARTS, typical pKa of the conjugate acid). Ordered strongest first; the first
# match wins, so a guanidine beats an aliphatic amine on the same molecule.
BASIC = [
    ("quaternary_n",    "[N+;H0;!$([N+][O-]);X4]",                     None),  # permanent
    ("guanidine",       "[NX3][CX3](=[NX2,NX3+])[NX3]",                12.5),
    ("amidine",         "[NX3][CX3]=[NX2,NX3+]",                       11.0),
    ("primary_alkyl_amine",   "[NX3;H2;!$(NC=[O,S,N]);!$(Na)]",        10.6),
    ("secondary_alkyl_amine", "[NX3;H1;!$(NC=[O,S,N]);!$(Na);!R]",     10.8),
    ("tertiary_alkyl_amine",  "[NX3;H0;!$(NC=[O,S,N]);!$(Na);!R;!$(N=*)]", 9.8),
    ("piperazine",      "[NX3;H1,H0;R]1[CX4][CX4][NX3;R][CX4][CX4]1",   9.0),
    ("cyclic_amine",    "[NX3;R;!$(NC=[O,S,N]);!$(Na);!$(N=*)]",        10.0),
    ("pyridine",        "[nX2;$(n1ccccc1)]",                             5.2),
    # A xanthine imidazole (caffeine, theophylline) is NOT basic -- measured pKa ~0.6.
    # Requiring both ring carbons to be free of exocyclic C=O excludes the purinones
    # while keeping genuine imidazoles (ketoconazole, cimetidine).
    ("imidazole",       "[nX2;$(n1c([!$(*=O)])nc([!$(*=O)])c1),$(n1c([!$(*=O)])c([!$(*=O)])nc1)]", 6.9),
    ("aniline",         "[NX3;!$(NC=[O,S,N]);$(Na)]",                    4.6),
]

ACIDIC = [
    ("sulfonic_acid",   "[SX4](=O)(=O)[OX2H1,OX1-]",                    -1.0),
    ("carboxylic_acid", "[CX3](=O)[OX2H1,OX1-]",                         4.2),
    ("tetrazole",       "[$(c1nnn[nH]1),$(c1nn[nH]n1),$(c1[nH]nnn1)]",   4.9),
    ("acyl_sulfonamide","[CX3](=O)[NX3H1][SX4](=O)(=O)",                 4.5),
    ("phosphonic",      "[PX4](=O)([OX2H1,OX1-])",                       2.0),
    # 4-hydroxycoumarin (warfarin) perceives as AROMATIC, so an all-[CX3] pattern
    # never fired and warfarin came out neutral with logD 3.6 against a measured 0.9.
    ("vinylogous_acid", "[OX2H1,OX1-][c,CX3]=,:[c,CX3][c,CX3]=[OX1]",    5.0),
    ("phenol",          "[OX2H1][c]",                                    9.9),
    ("sulfonamide_nh",  "[SX4](=O)(=O)[NX3H1,NX3H2]",                   10.3),
    ("thiol",           "[SX2H1]",                                      10.0),
]

NAMES = ["pka_basic", "pka_acidic", "frac_base_ionised", "frac_acid_ionised",
         "logd74", "logp_minus_logd", "net_charge_74", "formal_charge",
         "is_zwitterion", "is_permanent_cation"]

_C: dict = {}


def _pats():
    if not _C:
        for tag, table in (("b", BASIC), ("a", ACIDIC)):
            out = []
            for nm, sm, pk in table:
                p = Chem.MolFromSmarts(sm)
                if p is None:
                    raise ValueError(f"invalid SMARTS for {nm}: {sm}")
                out.append((nm, p, pk))
            _C[tag] = out
    return _C["b"], _C["a"]


def n_features() -> int:
    return len(NAMES)


def calculate(smiles) -> tuple[np.ndarray, np.ndarray]:
    """(X, valid). Works on the molecule AS DRAWN — never uncharges."""
    bas, aci = _pats()
    X = np.zeros((len(smiles), len(NAMES)))
    valid = np.zeros(len(smiles), bool)

    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s)) if isinstance(s, str) else None
        if mol is None:
            continue
        valid[i] = True

        perm = False
        pka_b = 0.0
        for nm, pat, pk in bas:
            if mol.HasSubstructMatch(pat):
                if pk is None:            # quaternary N: no equilibrium to solve
                    perm = True
                    continue
                pka_b = max(pka_b, pk)
        pka_a = 14.0
        for nm, pat, pk in aci:
            if mol.HasSubstructMatch(pat):
                pka_a = min(pka_a, pk)

        # Henderson-Hasselbalch. Clipped exponents so a pKa far from 7.4 cannot overflow.
        fb = 1.0 / (1.0 + 10 ** np.clip(PH - pka_b, -12, 12)) if pka_b > 0 else 0.0
        fa = 1.0 / (1.0 + 10 ** np.clip(pka_a - PH, -12, 12)) if pka_a < 14 else 0.0

        logp = float(Crippen.MolLogP(mol))
        # Only the neutral fraction partitions into octanol; each equilibrium subtracts
        # its own log term. This is the standard single-species approximation.
        d = logp
        if pka_b > 0:
            d -= np.log10(1.0 + 10 ** np.clip(pka_b - PH, -12, 12))
        if pka_a < 14:
            d -= np.log10(1.0 + 10 ** np.clip(PH - pka_a, -12, 12))
        # Ion-pair partitioning puts a FLOOR under logD: a fully ionised base still
        # partitions as an ion pair, so the correction saturates. Without this cap
        # debrisoquine (guanidine, pKa 12.5) came out at logD -4.2 against a measured
        # -0.5, because Henderson-Hasselbalch alone subtracts 5.1 log units.
        d = max(d, logp - MAX_IONISATION_PENALTY)
        if perm:
            d = min(d, logp - 3.0)        # a permanent cation never partitions freely

        fc = float(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
        net = fb - fa + (1.0 if perm else 0.0)
        X[i] = [pka_b, pka_a, fb, fa, d, logp - d, net, fc,
                float(fb > 0.5 and fa > 0.5), float(perm)]

    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), valid


def validate(verbose: bool = True) -> bool:
    """Against drugs with measured pKa / logD7.4. States the error, does not hide it."""
    # (name, SMILES, measured strongest basic pKa, measured logD7.4)
    ref = [
        ("quinidine",   "COc1ccc2nccc([C@@H](O)[C@H]3C[C@@H]4CCN3C[C@@H]4C=C)c2c1", 8.6, 2.0),
        ("propranolol", "CC(C)NCC(O)COc1cccc2ccccc12",                              9.5, 1.2),
        ("fluoxetine",  "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",                       9.8, 1.9),
        ("debrisoquine","N=C(N)N1CC2=CC=CC=C2C1",                                  11.9, -0.5),
        ("diclofenac",  "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",                           0.0, 1.1),
        ("warfarin",    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",                    0.0, 0.9),
        ("caffeine",    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                               0.0, -0.1),
        ("midazolam",   "Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2",                   6.0, 3.9),
    ]
    smi = [r[1] for r in ref]
    X, valid = calculate(smi)
    if not valid.all():
        if verbose:
            print("FAIL: unparsed SMILES")
        return False
    ib, il = NAMES.index("pka_basic"), NAMES.index("logd74")
    ok = True
    if verbose:
        print(f"  {'drug':13s} {'pKa pred':>8s} {'pKa obs':>8s} {'logD pred':>9s} "
              f"{'logD obs':>8s} {'logP':>6s}")
    for (nm, s, pk, ld), row in zip(ref, X):
        lp = float(Crippen.MolLogP(Chem.MolFromSmiles(s)))
        if verbose:
            print(f"  {nm:13s} {row[ib]:8.1f} {pk:8.1f} {row[il]:9.2f} {ld:8.2f} {lp:6.2f}")
        # the CLASS call is what must be right: base vs not-a-base
        if (row[ib] > 7.0) != (pk > 7.0):
            ok = False
            if verbose:
                print(f"    ^ MISCLASSIFIED as {'base' if row[ib] > 7 else 'non-base'}")
    err = np.abs(X[:, il] - np.array([r[3] for r in ref]))
    if verbose:
        print(f"\n  logD7.4 mean absolute error {err.mean():.2f} "
              f"(max {err.max():.2f}) over {len(ref)} drugs")
        print(f"  basic/non-basic classification: {'PASS' if ok else 'FAIL'}")
        print("  This is a SMARTS lookup, not a pKa model: trust the class, not the "
              "decimal.")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
