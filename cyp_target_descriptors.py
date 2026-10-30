"""Isoform-specific descriptor blocks for the per-target Chemprop models.

Your PXR pipeline concatenated ONE hand-built 61-descriptor vector to the learned
embedding. The four CYP isoforms do not share a pharmacophore, so a single vector
would be three-quarters noise for each model. Each isoform gets a shared core plus a
block built from what is actually known to drive its recognition:

    CYP1A2   A narrow, planar active site. Substrates are small, flat and aromatic —
             caffeine, phenacetin, melatonin, aryl amines. The block measures
             flatness and aromatic extent: fused-aromatic-ring count, the size of the
             largest fused aromatic system, aromatic atom fraction, sp3 fraction
             (inverted flatness), and small-and-planar flags.

    CYP2C9   An active site with Arg108 driving recognition of ANIONS. Substrates are
             acidic — diclofenac, warfarin, tolbutamide, losartan. The block counts
             ionisable acids (carboxylic acid, tetrazole, acylsulfonamide, acidic
             sulfonamide, phenol) and pairs them with lipophilicity.

    CYP2D6   The most specific pharmacophore of the four: a BASIC NITROGEN, positively
             charged at pH 7.4, forming a salt bridge with Asp301/Glu216, sited
             roughly 5-7 A from the oxidation position — dextromethorphan, quinidine,
             debrisoquine, paroxetine. Generic descriptors count nitrogens; they
             cannot express "basic nitrogen at the right distance from an aromatic
             ring". `basic_n_to_aromatic_min/max/mean` walks the topological shortest
             path from each basic nitrogen to the nearest aromatic atom, and
             `d2d6_pharmacophore` flags the 2-5 bond window that corresponds to it.

    CYP3A4   A large, flexible, promiscuous pocket. Size, lipophilicity and
             conformational freedom dominate — midazolam, ketoconazole, testosterone,
             erythromycin. The block measures bulk (MW, heavy atoms, Labute ASA),
             flexibility (rotatable bonds, rotatable-per-ring), macrocycles, and the
             large-and-lipophilic combination.

Every block is SMARTS counts and simple graph walks — nothing fitted, so nothing can
leak. `validate()` checks each block fires on that isoform's canonical substrates and
stays quiet on the others; run it before trusting a change.
"""

from __future__ import annotations

import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors as rdMD

from cypkit import physchem as PC

# The physchem block is measurably positive for the CLASSICAL learners (+0.0125 CYP1A2,
# +0.0055 CYP3A4 over 3 seeds) but is the one variable present in BOTH ChemProp runs that
# regressed and absent from v6, and CYP2C9 -- the endpoint where its LightGBM effect was
# pure noise (+0.008 +- 0.015) -- collapsed by 10.5 sd on ChemProp. Ten extra columns is
# 16% of CYP2C9's 63-feature descriptor vector, fused at the readout. Set
# CYP_NO_PHYSCHEM=1 to drop the block; unset it and the block is included.
INCLUDE_PHYSCHEM = os.environ.get("CYP_NO_PHYSCHEM", "") not in ("1", "true", "True")

ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")

# ---------------------------------------------------------------------------
# Shared core — physicochemistry every isoform cares about, plus heme ligation
# ---------------------------------------------------------------------------

CORE_RDKIT = [
    "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "NumSaturatedRings", "NumAliphaticRings",
    "FractionCSP3", "HeavyAtomCount", "MolMR", "BertzCT", "Kappa1", "Kappa2",
    "Kappa3", "Chi0v", "Chi1v", "Chi2v", "HallKierAlpha", "LabuteASA",
    "MaxPartialCharge", "MinPartialCharge", "MaxEStateIndex", "MinEStateIndex",
    "MaxAbsEStateIndex", "NumHeteroatoms", "NHOHCount", "NOCount", "qed",
]

CORE_SMARTS = {
    "heme_aromatic_n": "[n;X2;!$(n[#6]=O)]",       # pyridine-type N, coordinates Fe
    "imidazole": "c1cnc[nH]1",
    "triazole": "c1ncnn1",
    "nitrile": "[NX1]#[CX2]",
    "amide": "[NX3][CX3]=[OX1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "ether": "[OX2;!$([OX2H]);!$(OC=O)]",
    "halogen": "[F,Cl,Br,I]",
}

# ---------------------------------------------------------------------------
# Isoform-specific SMARTS
# ---------------------------------------------------------------------------

ISO_SMARTS = {
    "CYP1A2": {
        "aryl_amine": "[NX3;H2,H1;!$(NC=[O,S,N])]c",
        "aryl_amide": "[a][NX3H][CX3]=O",
        "aryl_methyl": "[CH3]c",
        "aryl_methoxy": "[CH3]Oc",
        "xanthine_like": "O=c1[nX3][cX3][nX3]c(=O)[nX3]1",
        "biaryl": "c-c",
        "furan_fused": "c1ccc2occc2c1",
    },
    "CYP2C9": {
        "carboxylic_acid": "[CX3](=O)[OX2H1]",
        "tetrazole": "c1nnn[nH]1",
        "acyl_sulfonamide": "[SX4](=O)(=O)[NX3H][CX3]=O",
        "sulfonamide_nh": "[SX4](=O)(=O)[NX3;H1,H2]",
        "phenol": "[OX2H]c",
        "carboxylate_anion": "[CX3](=O)[OX1-]",
        "aryl_acetic_acid": "c[CH2][CX3](=O)[OX2H1]",
        "coumarin": "O=c1ccc2ccccc2o1",
        # Warfarin is acidic (pKa ~5) through the 4-hydroxycoumarin enol, not a
        # carboxyl. Without this the block calls the archetypal CYP2C9 substrate
        # neutral -- caught by validate().
        "vinylogous_acid": "[OX2H][#6]=,:[#6][#6]=[OX1]",
    },
    "CYP2D6": {
        "basic_n_aliphatic": "[NX3;H0,H1,H2;!$(N[a]);!$(NC=[O,S,N]);!$(N=*);!$(N#*)]",
        "basic_n_tertiary": "[NX3;H0;!$(N[a]);!$(NC=[O,S,N]);!$(N=*)]",
        "basic_n_secondary": "[NX3;H1;!$(N[a]);!$(NC=[O,S,N]);!$(N=*)]",
        "basic_n_primary": "[NX3;H2;!$(N[a]);!$(NC=[O,S,N])]",
        "n_methyl": "[NX3;!$(N[a]);!$(NC=[O,S,N])][CH3]",
        "piperidine": "[NX3;R;!$(NC=[O,S,N])]1[CX4][CX4][CX4][CX4][CX4]1",
        "pyrrolidine": "[NX3;R;!$(NC=[O,S,N])]1[CX4][CX4][CX4][CX4]1",
        "benzylamine": "[NX3;!$(NC=[O,S,N])][CH2]c",
        "guanidine_amidine": "[NX3][CX3]=[NX2]",
        # An amidine N is sp2 and was excluded by the !$(N=*) filter on
        # basic_n_aliphatic -- so debrisoquine, the classic CYP2D6 probe drug,
        # scored as having no basic nitrogen. Amidines are pKa ~11.
        "amidine_sp2_n": "[NX2]=[CX3][NX3]",
    },
    "CYP3A4": {
        "basic_n_any": "[NX3;!$(N[a]);!$(NC=[O,S,N]);!$(N=*)]",
        "amide_bond": "[NX3][CX3]=[OX1]",
        "ester_bond": "[CX3](=O)[OX2][CX4]",
        "steroid_like": "C1CC2CCC3C(CCC4CCCCC34)C2C1",
        "azole": "c1cncn1",
        "piperazine": "[NX3;R][CX4;R][CX4;R][NX3;R]",
        "sugar_like": "[OX2][CX4][CX4][OX2]",
    },
}

ISO_DERIVED = {
    "CYP1A2": ["fused_aromatic_rings", "largest_aromatic_system", "aromatic_atom_frac",
               "planarity_index", "small_and_planar", "aromatic_n_and_planar"],
    "CYP2C9": ["n_acidic_groups", "is_acid", "acid_and_lipophilic", "acid_per_heavy",
               "anionic_logp_product"],
    "CYP2D6": ["n_basic_n", "is_base", "basic_n_to_aromatic_min",
               "basic_n_to_aromatic_mean", "basic_n_to_aromatic_max",
               "d2d6_pharmacophore", "base_and_aromatic", "basic_n_per_heavy"],
    "CYP3A4": ["largest_ring_size", "is_macrocycle", "rot_per_ring",
               "large_and_lipophilic", "bulk_index", "n_stereocentres"],
}

_CACHE: dict = {}


def _compile():
    if not _CACHE:
        _CACHE["core"] = {k: Chem.MolFromSmarts(v) for k, v in CORE_SMARTS.items()}
        for iso, d in ISO_SMARTS.items():
            _CACHE[iso] = {k: Chem.MolFromSmarts(v) for k, v in d.items()}
        bad = [(g, k) for g, d in _CACHE.items() for k, v in d.items() if v is None]
        if bad:
            raise ValueError(f"invalid SMARTS: {bad}")
    return _CACHE


def feature_names(isoform: str) -> list[str]:
    # The physchem block is SHARED: ionisation state drives lipophilicity for every
    # isoform. It matters most for CYP2D6, whose inhibitors are protonated bases -- the
    # one class where cLogP and logD7.4 disagree by several log units -- but CYP2C9's
    # anionic recognition needs the acidic half for the same reason.
    return (CORE_RDKIT + [f"core_{k}" for k in CORE_SMARTS]
            + ([f"phys_{k}" for k in PC.NAMES] if INCLUDE_PHYSCHEM else [])
            + [f"{isoform}_{k}" for k in ISO_SMARTS[isoform]]
            + [f"{isoform}_{k}" for k in ISO_DERIVED[isoform]])


def n_features(isoform: str) -> int:
    return len(feature_names(isoform))


# ---------------------------------------------------------------------------
# The CYP2D6 pharmacophore geometry
# ---------------------------------------------------------------------------


def _basic_nitrogens(mol, pats) -> list[int]:
    """Indices of nitrogens protonated at pH 7.4.

    Aliphatic amines plus amidines and guanidines. The sp2 nitrogen of an amidine is
    excluded by the usual `!$(N=*)` amine filter, yet it is the most basic nitrogen a
    drug is likely to carry (pKa ~11) -- which is why debrisoquine, a canonical
    CYP2D6 probe substrate, is invisible to a naive basic-amine pattern.
    """
    idx = {m[0] for m in mol.GetSubstructMatches(pats["basic_n_aliphatic"])}
    for key in ("amidine_sp2_n", "guanidine_amidine"):
        for m in mol.GetSubstructMatches(pats[key]):
            idx.update(a for a in m if mol.GetAtomWithIdx(a).GetSymbol() == "N")
    return sorted(idx)


def _basic_n_to_aromatic(mol, pats) -> tuple[float, float, float, float]:
    """Topological bond distance from each basic N to the nearest aromatic atom.

    The CYP2D6 pharmacophore is a protonated nitrogen roughly 5-7 A from the site of
    oxidation, which in practice means a basic amine a few bonds from an aromatic
    ring. Bond count is a cheap, conformation-free stand-in for that distance, and
    unlike a 3D descriptor it cannot be corrupted by a bad embedding.

    Returns (min, mean, max, pharmacophore_flag). Zeros when the molecule has no
    basic nitrogen or no aromatic atom.
    """
    basic = _basic_nitrogens(mol, pats)
    arom = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    if not basic or not arom:
        return 0.0, 0.0, 0.0, 0.0
    D = Chem.GetDistanceMatrix(mol)
    per_n = [float(np.min([D[b, a] for a in arom])) for b in basic]
    per_n = [d for d in per_n if np.isfinite(d) and d < 1e6]
    if not per_n:
        return 0.0, 0.0, 0.0, 0.0
    lo = float(np.min(per_n))
    # 2-5 bonds is the window that corresponds to the 5-7 A separation
    flag = float(any(2.0 <= d <= 5.0 for d in per_n))
    return lo, float(np.mean(per_n)), float(np.max(per_n)), flag


def _largest_aromatic_system(mol) -> int:
    """Atom count of the largest set of fused aromatic rings.

    CYP1A2's slot is narrow and planar; a naphthalene or larger fused system is the
    shape that fits it. Ring COUNT does not distinguish a fused system from two
    separate rings, so this walks the fusion graph.
    """
    ri = mol.GetRingInfo()
    arom = [set(r) for r in ri.AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    if not arom:
        return 0
    merged, changed = arom[:], True
    while changed:
        changed = False
        out = []
        while merged:
            cur = merged.pop()
            hit = [g for g in out if g & cur]
            if hit:
                for g in hit:
                    out.remove(g)
                    cur = cur | g
                changed = True
            out.append(cur)
        merged = out
    return max(len(g) for g in merged)


# ---------------------------------------------------------------------------


def calculate(smiles, isoform: str) -> tuple[np.ndarray, np.ndarray]:
    """(X, valid) for one isoform's descriptor block."""
    if isoform not in ISO_SMARTS:
        raise ValueError(f"unknown isoform {isoform!r}")
    pats = _compile()
    core_p, iso_p = pats["core"], pats[isoform]
    core_keys, iso_keys = list(CORE_SMARTS), list(ISO_SMARTS[isoform])
    n = n_features(isoform)
    X = np.zeros((len(smiles), n), dtype=np.float64)
    valid = np.zeros(len(smiles), dtype=bool)

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        valid[i] = True
        row = []
        for d in CORE_RDKIT:
            f = getattr(Descriptors, d, None) or getattr(rdMD, d, None) or getattr(Crippen, d, None)
            try:
                v = float(f(mol))
            except Exception:
                v = 0.0
            row.append(v if np.isfinite(v) else 0.0)
        core = {k: len(mol.GetSubstructMatches(core_p[k])) for k in core_keys}
        row += [float(core[k]) for k in core_keys]
        # physchem is computed from the SMILES as drawn -- cypkit.features uncharges
        # before featurising, which would erase exactly the signal this block carries
        if INCLUDE_PHYSCHEM:
            row += [float(v) for v in PC.calculate([smi])[0][0]]
        iso = {k: len(mol.GetSubstructMatches(iso_p[k])) for k in iso_keys}
        row += [float(iso[k]) for k in iso_keys]

        heavy = max(mol.GetNumHeavyAtoms(), 1)
        logp = Crippen.MolLogP(mol)
        rings = mol.GetRingInfo().AtomRings()
        n_arom_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())

        if isoform == "CYP1A2":
            fused = sum(1 for r in rings if sum(1 for r2 in rings
                        if r2 != r and set(r) & set(r2)) > 0
                        and all(mol.GetAtomWithIdx(x).GetIsAromatic() for x in r))
            largest = _largest_aromatic_system(mol)
            af = n_arom_atoms / heavy
            planarity = 1.0 - Descriptors.FractionCSP3(mol)
            row += [float(fused), float(largest), af, planarity,
                    float(Descriptors.MolWt(mol) < 300 and planarity > 0.6),
                    float(core["heme_aromatic_n"] > 0 and planarity > 0.6)]

        elif isoform == "CYP2C9":
            acid = (iso["carboxylic_acid"] + iso["tetrazole"] + iso["acyl_sulfonamide"]
                    + iso["carboxylate_anion"] + iso["vinylogous_acid"])
            row += [float(acid), float(acid > 0), float(acid > 0) * logp,
                    acid / heavy, float(acid > 0) * max(logp, 0.0)]

        elif isoform == "CYP2D6":
            nb = len(_basic_nitrogens(mol, iso_p))
            lo, mean, hi, flag = _basic_n_to_aromatic(mol, iso_p)
            row += [float(nb), float(nb > 0), lo, mean, hi, flag,
                    float(nb > 0 and rdMD.CalcNumAromaticRings(mol) > 0), nb / heavy]

        else:  # CYP3A4
            largest_ring = max((len(r) for r in rings), default=0)
            rot = Descriptors.NumRotatableBonds(mol)
            mw = Descriptors.MolWt(mol)
            row += [float(largest_ring), float(largest_ring >= 12),
                    rot / max(len(rings), 1),
                    float(mw > 400 and logp > 3), mw * max(logp, 0.0) / 1000.0,
                    float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)))]

        X[i] = np.asarray(row, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), valid


# ---------------------------------------------------------------------------


def validate(verbose: bool = True) -> bool:
    """Each block should fire on its own isoform's canonical substrates.

    Not a unit test of numeric values — a check that the chemistry each block claims
    to encode is actually being detected. Run it after editing any SMARTS.
    """
    probes = {
        "CYP1A2": {"caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
                   "phenacetin": "CCOc1ccc(NC(C)=O)cc1",
                   "melatonin": "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",
                   "naphthoflavone": "O=c1cc(-c2ccccc2)oc2c1ccc1ccccc12"},
        "CYP2C9": {"diclofenac": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
                   "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
                   "tolbutamide": "CCCCNC(=O)NS(=O)(=O)c1ccc(C)cc1",
                   "losartan": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"},
        "CYP2D6": {"dextromethorphan": "COc1ccc2c(c1)[C@]13CCCC[C@H]1[C@@H](C2)N(C)CC3",
                   "debrisoquine": "N=C1NCc2ccccc2C1",
                   "quinidine": "C=C[C@H]1CN2CC[C@H]1C[C@@H]2[C@H](O)c1ccnc2ccc(OC)cc12",
                   "paroxetine": "Fc1ccc([C@@H]2CCNC[C@H]2COc2ccc3OCOc3c2)cc1"},
        "CYP3A4": {"midazolam": "Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2",
                   "ketoconazole": "CC(=O)N1CCN(c2ccc(OC[C@H]3CO[C@](Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1",
                   "testosterone": "C[C@]12CC[C@H]3[C@@H](CC[C@H]4CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O"},
    }
    key = {"CYP1A2": "CYP1A2_planarity_index", "CYP2C9": "CYP2C9_is_acid",
           "CYP2D6": "CYP2D6_is_base", "CYP3A4": "CYP3A4_bulk_index"}
    ok = True
    for iso, mols in probes.items():
        names = feature_names(iso)
        X, v = calculate(list(mols.values()), iso)
        j = names.index(key[iso])
        vals = X[:, j]
        if verbose:
            print(f"{iso}  ({len(names)} features)  {key[iso]}:")
            for nm, val in zip(mols, vals):
                print(f"    {nm:18s} {val:8.3f}")
        if not v.all():
            print(f"  FAIL: unparseable SMILES in {iso} probes"); ok = False
        if iso in ("CYP2C9", "CYP2D6") and not (vals > 0).all():
            print(f"  FAIL: {key[iso]} did not fire on every {iso} probe"); ok = False
    # CYP2D6 pharmacophore flag must fire on all four 2D6 drugs
    names = feature_names("CYP2D6")
    X, _ = calculate(list(probes["CYP2D6"].values()), "CYP2D6")
    ph = X[:, names.index("CYP2D6_d2d6_pharmacophore")]
    if verbose:
        print(f"\nCYP2D6 pharmacophore flag on 2D6 substrates: {ph.tolist()}")
    if not (ph > 0).all():
        print("  FAIL: 2D6 pharmacophore missed a canonical substrate"); ok = False
    # and should be quiet on the acidic 2C9 drugs
    X, _ = calculate(list(probes["CYP2C9"].values()), "CYP2D6")
    ph2 = X[:, names.index("CYP2D6_d2d6_pharmacophore")]
    if verbose:
        print(f"CYP2D6 pharmacophore flag on 2C9 substrates: {ph2.tolist()}  "
              f"(should be mostly 0)")
    return ok


if __name__ == "__main__":
    print("feature counts:", {i: n_features(i) for i in ISOFORMS}, "\n")
    print("VALIDATION PASSED" if validate() else "VALIDATION FAILED")
