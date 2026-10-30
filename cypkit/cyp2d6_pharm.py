#!/usr/bin/env python
r"""CYP2D6's recognition pharmacophore, as explicit features.

WHY THIS ENDPOINT GETS ITS OWN FEATURE SET. CYP2D6 has the narrowest confidence intervals,
the second-best signal-to-noise and the second-most labels of the four endpoints, and the
worst rho by a factor of 1.6 (0.519 against 0.81-0.85). Section 22 ruled out coverage, quantity
and label quality as causes -- proximity-selected external data made it significantly
WORSE, not better. What is left is that our features do not encode what a CYP2D6 binder is.

THE MECHANISM WE ARE ENCODING. CYP2D6's active site is small and anionic. Substrate
recognition is dominated by a salt bridge between a PROTONATED BASIC NITROGEN on the ligand
and Glu216 (with Asp301 orienting the backbone), together with a flat aromatic region that
stacks against Phe120/Phe483. The classical pharmacophore places the basic nitrogen
roughly 5-7 A from the aromatic centroid. None of that survives into a Morgan bitvector,
which sees atom environments and not the DISTANCE BETWEEN TWO PHARMACOPHORIC POINTS, and
bulk descriptors average it away.

WHAT IS COMPUTED. Basic nitrogens are the strongly basic ones only -- aliphatic amines,
amidines and guanidines -- with amides, sulfonamides, nitriles, nitro groups, anilines and
aromatic-ring nitrogens excluded, because a pyridine N at pKa ~5 is not protonated at
physiological pH and does not make this salt bridge. Weakly basic nitrogens are counted
separately rather than thrown away.

Distances come in two flavours because they answer different questions and cost different
amounts. The TOPOLOGICAL distance (bonds along the shortest path) is exact, instant, and
conformation-free. The 3D distance is the one the pharmacophore is actually stated in, but
depends on a single ETKDG conformer, which for a flexible ligand is one sample from a
population. Both are provided; `build()` takes a flag, so the two can be tested separately
and the 3D cost only paid if it earns its place.

NOT A MODEL, A FEATURE BLOCK. These columns are meant to be concatenated onto whatever
representation is already in use and tested against a control that lacks them.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# Strongly basic: protonated at pH 7.4 and able to make the Glu216 salt bridge.
BASIC = Chem.MolFromSmarts(
    "[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(N[a]);!$(N=*);!$([N+](=O)[O-]);"
    "!$(NS(=O)=O);!$(NC#N);!$(N[CX3]=[CX3])]")
AMIDINE = Chem.MolFromSmarts("[NX3][CX3]=[NX2,NX3+]")
# Weakly basic: counted, but not treated as a salt-bridge donor.
WEAK = Chem.MolFromSmarts("[$([nX2]),$(N[a]),$(N[CX3]=[CX3])]")

NAMES = ["n_basic_N", "has_basic_N", "n_weak_N", "n_arom_rings", "frac_arom_atoms",
         "topo_min", "topo_mean", "topo_max", "topo_in_window", "n_basic_arom_pairs",
         "tpsa_per_heavy", "logp_x_basic"]
NAMES_3D = ["d3_min", "d3_mean", "d3_max", "d3_in_window", "radius_gyration"]


def _basic_atoms(mol) -> list[int]:
    idx = {a[0] for a in mol.GetSubstructMatches(BASIC)}
    idx |= {a[0] for a in mol.GetSubstructMatches(AMIDINE)}
    return sorted(idx)


def _aromatic_rings(mol) -> list[list[int]]:
    return [list(r) for r in mol.GetRingInfo().AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]


def _features(smi: str, use_3d: bool) -> np.ndarray:
    n = len(NAMES) + (len(NAMES_3D) if use_3d else 0)
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return np.zeros(n, np.float32)
    basic = _basic_atoms(mol)
    rings = _aromatic_rings(mol)
    heavy = mol.GetNumHeavyAtoms() or 1
    n_weak = len(mol.GetSubstructMatches(WEAK))
    arom_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())

    # topological: bonds from each basic N to each aromatic ring, ring taken at its
    # closest atom. 3-6 bonds is the span that puts the two groups the right distance
    # apart in an extended conformation.
    topo = []
    if basic and rings:
        D = Chem.GetDistanceMatrix(mol)
        for b in basic:
            for r in rings:
                topo.append(min(D[b][i] for i in r))
    topo = np.asarray(topo, float) if topo else np.array([0.0])
    in_window = float(np.mean((topo >= 3) & (topo <= 6))) if topo.size and basic else 0.0

    logp = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
    out = [len(basic), float(bool(basic)), n_weak, len(rings), arom_atoms / heavy,
           topo.min(), topo.mean(), topo.max(), in_window, len(basic) * len(rings),
           rdMolDescriptors.CalcTPSA(mol) / heavy, logp * float(bool(basic))]

    if use_3d:
        d3 = [0.0]; rg = 0.0
        try:
            mh = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mh, randomSeed=0xf00d, maxAttempts=5) == 0:
                AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
                conf = mh.GetConformer()
                pos = np.array([list(conf.GetAtomPosition(i))
                                for i in range(mol.GetNumAtoms())])
                rg = float(np.sqrt(((pos - pos.mean(0)) ** 2).sum(1).mean()))
                if basic and rings:
                    d3 = [float(np.linalg.norm(pos[b] - pos[r].mean(0)))
                          for b in basic for r in rings]
        except Exception:
            pass
        d3 = np.asarray(d3, float)
        # the classical CYP2D6 window: basic N 5-7 A from the aromatic centroid
        out += [d3.min(), d3.mean(), d3.max(),
                float(np.mean((d3 >= 5.0) & (d3 <= 7.0))) if basic and rings else 0.0, rg]
    return np.nan_to_num(np.asarray(out, np.float32), posinf=0.0, neginf=0.0)


def build(smiles, use_3d: bool = False, verbose: bool = False):
    """(n, d) feature matrix plus the column names. 3D costs ~0.1 s per molecule."""
    X = np.vstack([_features(s, use_3d) for s in smiles])
    names = NAMES + (NAMES_3D if use_3d else [])
    if verbose:
        print(f"  pharmacophore block: {X.shape[0]} molecules x {X.shape[1]} features"
              f"{' (with 3D)' if use_3d else ''}")
        print(f"    carry a strongly basic nitrogen: {np.mean(X[:, 1]):.1%}")
        if use_3d:
            j = names.index("d3_in_window")
            print(f"    have a basic N 5-7 A from an aromatic centroid: "
                  f"{np.mean(X[:, j] > 0):.1%}")
    return X, names
