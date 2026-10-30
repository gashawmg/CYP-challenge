"""CYP-specific structural descriptors.

Generic RDKit descriptors know nothing about heme iron, and the four isoforms have
genuinely different substrate preferences. This module encodes the standard medicinal-
chemistry priors as SMARTS counts, grouped by the pharmacology they come from:

  heme coordination   type II binders ligate the ferric iron through a lone pair — the
                      azole antifungals are the canonical potent CYP inhibitors, and an
                      unhindered aromatic nitrogen is the single strongest structural
                      prior for CYP inhibition there is
  CYP2D6              active site carries Glu216/Asp301; substrates are typically basic
                      amines, protonated at physiological pH, with an aromatic ring
  CYP2C9              Arg108 in the active site; substrates skew acidic or weakly acidic
                      (warfarin, the NSAIDs)
  CYP1A2              narrow planar pocket; substrates are flat fused aromatics
                      (caffeine, phenacetin)
  CYP3A4              large flexible hydrophobic pocket; big lipophilic molecules
  soft spots          sites the enzyme actually attacks — N- and O-dealkylation,
                      benzylic and aromatic hydroxylation
  MBI alerts          groups that form reactive metabolites or MI complexes; these drive
                      time-dependent inhibition and matter for both tracks

Every feature is a count, so a tree can learn its own threshold. Nothing here is
asserted to help — `run_baselines.py --features` measures whether it does.

Sources: standard CYP structure-activity literature and the mechanism-based
inactivation alert lists in common use. Treat the groupings as motivation, not proof;
the isoform labels say why a feature is here, not that it only matters for that isoform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# name -> (SMARTS, rationale)
SMARTS_FEATURES: dict[str, tuple[str, str]] = {
    # ---- heme coordination (type II binding) --------------------------------
    "heme_pyridine_n": ("n1ccccc1", "pyridine N ligates ferric iron"),
    "heme_imidazole": ("c1cnc[nX3]1", "imidazole — the azole inhibitor motif"),
    "heme_imidazole_nh": ("c1cnc[nH]1", "NH-imidazole"),
    "heme_triazole": ("c1ncnn1", "1,2,4-triazole — azole antifungal motif"),
    "heme_thiazole_n": ("c1cscn1", "thiazole N"),
    "heme_oxazole_n": ("c1cocn1", "oxazole N"),
    "heme_nitrile": ("[NX1]#[CX2]", "nitrile N can coordinate iron"),
    "heme_thioether": ("[#16X2H0]([#6])[#6]", "sulfur lone pair coordination"),
    "arom_n_unsubst": ("[nX2]", "any 2-connected aromatic N — pooled coordination count"),
    "arom_n_total": ("[n]", "all aromatic nitrogens"),

    # ---- CYP2D6 pharmacophore: basic amine + aromatic ------------------------
    "basic_amine": (
        "[NX3;H0,H1,H2;!$(NC=[O,S,N]);!$(N[a]);!$(N=*);!$(N[N,O]);!$([N+])]",
        "aliphatic amine, protonated at pH 7.4 — the 2D6 anchor",
    ),
    "basic_amine_tertiary": (
        "[NX3;H0;!$(NC=[O,S,N]);!$(N[a]);!$(N=*);!$(N[N,O])]",
        "tertiary aliphatic amine",
    ),
    "amidine_guanidine": ("[NX3][CX3]=[NX2]", "strongly basic amidine/guanidine"),
    "aniline_n": ("[NX3;!$(NC=[O,S,N])][a]", "aniline nitrogen — weakly basic, and an MBI alert"),

    # ---- CYP2C9 pharmacophore: acidic ---------------------------------------
    "carboxylic_acid": ("[CX3](=O)[OX2H1]", "anionic at pH 7.4 — the 2C9 anchor"),
    "phenol_oh": ("[OX2H][c]", "weakly acidic phenol"),
    "acyl_sulfonamide": ("[CX3](=O)[NX3][SX4](=O)(=O)", "acid bioisostere"),
    "sulfonamide_nh": ("[SX4](=O)(=O)[NX3;H1,H2]", "acidic sulfonamide NH"),
    "tetrazole": ("c1nnn[nH]1", "carboxylate bioisostere"),

    # ---- metabolic soft spots ----------------------------------------------
    "n_methyl": ("[NX3;!$(NC=O)][CH3]", "N-demethylation site"),
    "n_alkyl_ch2": ("[NX3;!$(NC=O)][CH2][CH3]", "N-dealkylation site"),
    "o_aryl_methyl": ("[OX2]([CH3])[a]", "O-demethylation site (aryl methyl ether)"),
    "o_alkyl_ether": ("[OX2]([CH2])[a]", "aryl alkyl ether — O-dealkylation"),
    "benzylic_ch": ("[CX4;H1,H2,H3][a]", "benzylic position — hydroxylation site"),
    "aromatic_ch": ("[cH]", "free aromatic CH — aromatic hydroxylation site"),

    # ---- mechanism-based inactivation alerts --------------------------------
    "mbi_methylenedioxyphenyl": ("c1ccc2c(c1)OCO2", "benzodioxole — carbene MI complex"),
    "mbi_terminal_alkyne": ("[CX2H]#[CX2]", "terminal acetylene — classic MBI"),
    "mbi_furan": ("c1ccoc1", "furan — epoxide/enedial reactive metabolite"),
    "mbi_thiophene": ("c1ccsc1", "thiophene — S-oxide reactive metabolite"),
    "mbi_thiourea": ("[NX3][CX3]=[SX1]", "thiourea/thioamide oxidation"),
    "mbi_hydrazine": ("[NX3][NX3]", "hydrazine"),
    "mbi_cyclopropylamine": ("[NX3]C1CC1", "cyclopropylamine radical clock"),
    "mbi_nitroaromatic": ("[a][$([NX3](=O)=O),$([NX3+](=O)[O-])]", "nitroaromatic"),
    "mbi_michael_acceptor": ("[CX3]=[CX3][CX3]=[OX1]", "Michael acceptor"),
    "mbi_epoxide": ("[OX2r3]1[#6r3][#6r3]1", "epoxide"),
    "mbi_quinone": ("O=[CX3]1[CX3]=[CX3][CX3](=O)[CX3]=[CX3]1", "quinone — redox cycling"),
    "mbi_anilide": ("[a][NX3H][CX3]=O", "anilide — quinone-imine precursor"),
}

# Derived features, computed from the molecule rather than a single SMARTS.
DERIVED_FEATURES = (
    "n_aromatic_rings",
    "n_fused_aromatic_pairs",
    "largest_aromatic_system",
    "frac_aromatic_atoms",
    "planarity_index",
    "n_heteroaromatic_rings",
    "cyp2d6_pharmacophore",
    "cyp2c9_pharmacophore",
    "cyp1a2_pharmacophore",
    "cyp3a4_bulk",
    "n_heme_ligands",
    "n_mbi_alerts",
    "n_soft_spots",
)


def feature_names() -> list[str]:
    return list(SMARTS_FEATURES) + list(DERIVED_FEATURES)


_COMPILED: dict[str, object] | None = None


def _compile():
    """Compile every SMARTS once, and refuse to run if any of them is invalid.

    A bad SMARTS returns None from MolFromSmarts and would otherwise become a silently
    all-zero column — the same failure mode that cost the PXR project three descriptors.
    """
    global _COMPILED
    if _COMPILED is not None:
        return _COMPILED
    from rdkit import Chem  # noqa: PLC0415

    compiled, bad = {}, []
    for name, (smarts, _) in SMARTS_FEATURES.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            bad.append((name, smarts))
        else:
            compiled[name] = patt
    if bad:
        raise ValueError(
            "invalid SMARTS in cyp_descriptors: "
            + ", ".join(f"{n} ({s})" for n, s in bad)
        )
    _COMPILED = compiled
    return compiled


def _aromatic_systems(mol) -> tuple[int, int, int]:
    """(n_aromatic_rings, n_fused_aromatic_pairs, atoms in largest aromatic system)."""
    ri = mol.GetRingInfo()
    arom_rings = [
        set(r) for r in ri.AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
    ]
    n_rings = len(arom_rings)
    if n_rings == 0:
        return 0, 0, 0

    # Union-find over rings that share two or more atoms (i.e. are fused).
    parent = list(range(n_rings))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    n_fused = 0
    for i in range(n_rings):
        for j in range(i + 1, n_rings):
            if len(arom_rings[i] & arom_rings[j]) >= 2:
                n_fused += 1
                ri_, rj = find(i), find(j)
                if ri_ != rj:
                    parent[ri_] = rj

    groups: dict[int, set] = {}
    for i in range(n_rings):
        groups.setdefault(find(i), set()).update(arom_rings[i])
    largest = max(len(g) for g in groups.values())
    return n_rings, n_fused, largest


def calculate(smiles, standardise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """CYP descriptor matrix.

    Returns (X, valid_mask), same contract as `features.rdkit_2d`: failures are flagged,
    not silently zeroed.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import Crippen, Descriptors  # noqa: PLC0415

    from .features import _standardise  # noqa: PLC0415

    patterns = _compile()
    names = feature_names()
    smarts_names = list(SMARTS_FEATURES)

    X = np.full((len(smiles), len(names)), np.nan, dtype=np.float64)
    valid = np.zeros(len(smiles), dtype=bool)

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        if standardise:
            mol, _ = _standardise(mol)

        try:
            counts = {
                n: len(mol.GetSubstructMatches(patterns[n], uniquify=True))
                for n in smarts_names
            }

            n_rings, n_fused, largest = _aromatic_systems(mol)
            heavy = mol.GetNumHeavyAtoms() or 1
            n_arom_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
            n_hetero_arom = sum(
                1
                for r in mol.GetRingInfo().AtomRings()
                if all(mol.GetAtomWithIdx(x).GetIsAromatic() for x in r)
                and any(mol.GetAtomWithIdx(x).GetSymbol() != "C" for x in r)
            )
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            csp3 = Descriptors.FractionCSP3(mol)

            heme = sum(
                counts[k]
                for k in (
                    "heme_pyridine_n", "heme_imidazole", "heme_imidazole_nh",
                    "heme_triazole", "heme_thiazole_n", "heme_oxazole_n",
                    "heme_nitrile", "heme_thioether",
                )
            )
            mbi = sum(counts[k] for k in counts if k.startswith("mbi_"))
            soft = sum(
                counts[k]
                for k in ("n_methyl", "n_alkyl_ch2", "o_aryl_methyl", "o_alkyl_ether", "benzylic_ch")
            )

            derived = {
                "n_aromatic_rings": n_rings,
                "n_fused_aromatic_pairs": n_fused,
                "largest_aromatic_system": largest,
                "frac_aromatic_atoms": n_arom_atoms / heavy,
                # Flatness: aromatic fraction penalised by sp3 content.
                "planarity_index": (n_arom_atoms / heavy) * (1.0 - csp3),
                "n_heteroaromatic_rings": n_hetero_arom,
                # Joint pharmacophores — a basic amine only reads as a 2D6 motif when
                # there is an aromatic ring to sit against, and so on.
                "cyp2d6_pharmacophore": float(counts["basic_amine"] > 0 and n_rings > 0),
                "cyp2c9_pharmacophore": float(
                    (counts["carboxylic_acid"] + counts["acyl_sulfonamide"] + counts["tetrazole"]) > 0
                    and logp > 1.0
                ),
                # A fused planar aromatic core — the 1A2 signature. Deliberately not
                # gated on whole-molecule sp3 fraction: caffeine carries three methyls
                # (CSP3 = 0.375) and is the canonical 1A2 substrate.
                "cyp1a2_pharmacophore": float(largest >= 9 and n_fused >= 1),
                "cyp3a4_bulk": logp * (heavy / 10.0),
                "n_heme_ligands": heme,
                "n_mbi_alerts": mbi,
                "n_soft_spots": soft,
            }
            row = [counts[n] for n in smarts_names] + [derived[n] for n in DERIVED_FEATURES]
            X[i] = row
            valid[i] = True
        except Exception:
            continue

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, valid


def describe() -> pd.DataFrame:
    """The feature list with its rationale — for reading, and for the model report."""
    rows = [
        dict(feature=n, kind="smarts", pattern=s, rationale=r)
        for n, (s, r) in SMARTS_FEATURES.items()
    ]
    rows += [dict(feature=n, kind="derived", pattern="", rationale="") for n in DERIVED_FEATURES]
    return pd.DataFrame(rows)
