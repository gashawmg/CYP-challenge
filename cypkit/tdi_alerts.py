"""Bioactivation alerts — a feature block built for the TDI track specifically.

Backing out every nearby competitor's operating point from the leaderboard's accuracy /
precision / recall columns showed something the MCC column hides. Under a binormal ROC,
ranks 16-23 all sit at d' = 1.09-1.13 (AUC ~0.78) and differ almost entirely in where
they put their threshold. We were the exception at d' = 1.00 (AUC 0.76). Threshold
placement is nearly flat between q = 0.12 and 0.29, so it is worth ~0.002 MCC; closing
the d' gap to the pack is worth ~0.04, and reaching d' = 1.3 is worth ~0.11.

So the TDI track is a RANKING problem, and the ranking has to come from chemistry the
generic descriptors do not encode. Time-dependent inhibition is not potency — it is
bioactivation. The compound must be turned into something reactive by the enzyme it
then disables: a nitroso species that coordinates the heme iron, an epoxide or
quinone-imine that alkylates the apoprotein, a carbene from a methylenedioxy bridge.
Those are structural questions with known answers, and 2,048 ECFP bits only reach them
by accident.

Two things here that the general CYP block does not have:

  MI-COMPLEX PRECURSORS. The largest single class of CYP3A4 time-dependent inhibitors
  is alkylamines that N-dealkylate to a nitroso species and coordinate the heme iron —
  erythromycin, diltiazem, verapamil, fluoxetine. That needs the amine's substitution
  pattern (N-methyl, N-ethyl, N-benzyl, secondary vs tertiary, ring vs chain), not the
  presence of a nitrogen.

  BIOACTIVATION IN CONTEXT. A furan is only a liability if it can be oxidised, and an
  aniline only if the ring is not already blocked. Several features here pair a
  reactive motif with the substitution that decides whether it survives first-pass
  metabolism intact.

Everything is a SMARTS count plus a handful of derived combinations. Nothing is fitted,
so the block adds no leakage risk and can be dropped in beside any other.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# SMARTS
# ---------------------------------------------------------------------------

ALERTS: dict[str, tuple[str, str]] = {
    # --- MI-complex precursors: amines that N-dealkylate to nitroso -----------
    "amine_n_methyl_tert": ("[NX3;H0;!$(N[a]);!$(NC=[O,S,N])](C)(C)C",
                            "N,N-dimethyl tertiary amine — the classic MI-complex donor"),
    "amine_n_methyl_sec": ("[NX3;H1;!$(N[a]);!$(NC=[O,S,N])]C",
                           "N-methyl secondary amine"),
    "amine_tert_alkyl": ("[NX3;H0;!$(N[a]);!$(NC=[O,S,N]);!$(N=*)]",
                         "tertiary aliphatic amine"),
    "amine_sec_alkyl": ("[NX3;H1;!$(N[a]);!$(NC=[O,S,N]);!$(N=*)]",
                        "secondary aliphatic amine"),
    "amine_benzylic_n": ("[NX3;!$(NC=[O,S,N])][CH2][a]",
                         "benzylamine — facile N-dealkylation"),
    "piperidine_n_alkyl": ("[NX3;H0;R]([CX4])[CX4]", "N-alkyl saturated azacycle"),
    "piperazine_aryl": ("[NX3;R]([a])[CX4;R]", "N-aryl piperazine"),
    "morpholine": ("C1COCCN1", "morpholine"),
    "amine_propylamine_chain": ("[NX3][CH2][CH2][CH2][a]",
                                "3-aryl propylamine — fluoxetine/paroxetine motif"),

    # --- carbene / methylenedioxy -------------------------------------------
    "methylenedioxy_aryl": ("[a]1[a][a]2OCO[a]2[a][a]1",
                            "fused benzodioxole — carbene MI complex"),
    "methylenedioxy_any": ("OCO", "methylenedioxy bridge, any context"),

    # --- alkyne / alkene ----------------------------------------------------
    "alkyne_terminal": ("[CX2H]#[CX2]", "terminal acetylene — oxirene/ketene"),
    "alkyne_internal": ("[CX2;!$([CX2H])]#[CX2;!$([CX2H])]", "internal alkyne"),
    "alkene_terminal": ("[CX3H2]=[CX3H1]", "terminal olefin — epoxide precursor"),
    "alkene_conjugated_aryl": ("[a][CX3]=[CX3]", "styrenyl alkene"),

    # --- reactive heteroaromatics -------------------------------------------
    "furan_ring": ("c1ccoc1", "furan — enedial / epoxide"),
    "furan_substituted": ("c1cc([!#1])oc1", "substituted furan"),
    "thiophene_ring": ("c1ccsc1", "thiophene — S-oxide"),
    "thiophene_2sub": ("c1ccc([!#1])s1", "2-substituted thiophene — ticlopidine motif"),
    "pyrrole_ring": ("c1cc[nH]c1", "pyrrole"),
    "indole_3_methyl": ("[CH3]c1c[nH]c2ccccc12", "3-methylindole — iminium precursor"),
    "thiazole_2amino": ("[NX3]c1nccs1", "2-aminothiazole"),
    "isothiazole": ("c1ccsn1", "isothiazole"),
    "imidazole_nh": ("c1c[nH]cn1", "NH-imidazole — heme ligation + bioactivation"),
    "triazole_ring": ("c1ncnn1", "triazole"),

    # --- quinone / quinone-imine precursors ---------------------------------
    "phenol_para_open": ("[OX2H]c1ccc([!#1])cc1", "para-substituted phenol — quinone precursor"),
    "aniline_unblocked": ("[NX3;H2,H1;!$(NC=[O,S,N])]c1cc[cH]cc1",
                          "aniline with a free para position"),
    "anilide_open": ("[a][NX3H][CX3]=O", "anilide — quinone-imine precursor"),
    "catechol": ("[OX2H]c1[c;$(c[OX2H])]cccc1", "catechol — o-quinone"),
    "methoxy_aryl": ("[CH3]O[a]", "aryl methyl ether — O-dealkylation to phenol"),

    # --- direct alkylators / other ------------------------------------------
    "epoxide": ("[OX2r3]1[#6r3][#6r3]1", "epoxide"),
    "michael_acceptor": ("[CX3]=[CX3][CX3]=[OX1]", "Michael acceptor"),
    "thiourea_thioamide": ("[NX3][CX3]=[SX1]", "thiourea / thioamide"),
    "hydrazine_any": ("[NX3][NX3]", "hydrazine"),
    "cyclopropylamine": ("[NX3]C1CC1", "cyclopropylamine radical clock"),
    "nitroaromatic": ("[a][$([NX3](=O)=O),$([NX3+](=O)[O-])]", "nitroaromatic"),
    "alkyl_halide": ("[CX4][F,Cl,Br,I]", "alkyl halide"),
    "aryl_halide_activated": ("[a][Cl,Br,I]", "aryl halide"),
    "sulfide_thioether": ("[#6][SX2][#6]", "thioether — S-oxidation"),
    "carbamate": ("[NX3][CX3](=O)[OX2]", "carbamate"),
    "acyl_glucuronide_acid": ("[CX3](=O)[OX2H1]", "carboxylic acid"),
    "tetrazole": ("c1nnn[nH]1", "tetrazole"),
    "macrocycle_ester": ("[CX3](=O)[OX2][CX4]", "ester — macrolide-like when in a large ring"),
    # --- from Faramarzi et al., Front Pharmacol 2025 (FDA) --------------------
    # Table 4 lists the MBI alerts that survived statistical qualification on their
    # 620-compound CYP3A4 TDI set; Table 3 the top CYP3A4 TDI features by Z-score.
    # Every candidate was measured on THIS library before being added here (Section 64):
    # six of their sixteen have fewer than 20 hits in 6,145 compounds and were dropped,
    # including the whole quinone-METHIDE phenol family, which has essentially no
    # coverage on the OpenADMET set (para-hydroquinone 0 compounds, alkylphenols 10-16).
    # The ten below are the ones with enough coverage to carry information, kept with
    # their measured lift so the next reader does not have to re-derive it.
    #
    # lift = P(TDI | alert) - P(TDI | no alert), on the labelled rows
    #                                    CYP3A4 (n)      CYP2D6 (n)
    "fda_arene_phenethylamine": (        # +0.144 (79)     -0.100 (90)   <- sign flips
        "c1ccccc1[CX4][CX4][NX3;!$(N[a]);!$(NC=[O,S,N])]",
        "arene with an aliphatic amine two carbons out; arene-oxide route"),
    "fda_alkoxy_alkylbenzene_ortho": (   # +0.078 (125)    +0.021 (72)
        "[CX4][OX2]c1ccccc1[CX4;H1,H2,H3]",
        "ortho-alkoxy alkylbenzene; O-dealkylation then quinone-methide"),
    "fda_alkylhalobenzene_ortho": (      # +0.052 (110)    -0.098 (65)
        "[F,Cl,Br,I]c1ccccc1[CX4;H1,H2,H3]",
        "ortho alkyl halobenzene; FDA's highest-Z CYP3A4 TDI feature family"),
    "fda_alkoxy_alkylbenzene_para": (    # +0.032 (160)    -0.006 (57)
        "[CX4][OX2]c1ccc([CX4;H1,H2,H3])cc1",
        "para-alkoxy alkylbenzene; FDA Table 3, Z 6"),
    "fda_piperazine_tert": (             # -0.032 (304)    +0.056 (108)
        "[NX3;H0;R]1[CX4][CX4][NX3;R][CX4][CX4]1",
        "piperazine with a tertiary N; carbinolamine / iminium route"),
    "fda_amine_sec_cyclic": (            # -0.004 (62)     -0.080 (78)
        "[NX3;H1;R;!$(N[a]);!$(NC=[O,S,N])]([CX4])[CX4]",
        "secondary cyclic non-aromatic amine, as a class"),
    "fda_alkylhalobenzene_para": (       # -0.056 (126)    -0.049 (94)
        "[F,Cl,Br,I]c1ccc([CX4;H1,H2,H3])cc1",
        "para alkyl halobenzene"),
    "fda_benzopyran": (                  # -0.085 (31)     too few
        "[OX2;R]1[#6;R][#6;R][#6;R]c2ccccc21",
        "benzopyran / chromane; arene-oxide route"),
    "fda_arene_alkene": (                # -0.094 (25)     too few
        "[CX3]=[CX3]c1ccccc1",
        "arene conjugated to an alkene"),
    "fda_diphenoxyethane": (             # -0.114 (40)     too few
        "c[OX2][CH2][CH2][OX2]c",
        "1,2-diphenoxyethane; FDA Table 3, Z 6"),
}

DERIVED = [
    "n_alerts_total",
    "n_mi_complex_precursors",
    "n_bioactivation_alerts",
    "amine_and_aryl",
    "reactive_ring_and_basic_n",
    "alert_density",
    "largest_ring_size",
    "n_rotatable_over_rings",
]

_MI_KEYS = [k for k in ALERTS if k.startswith(("amine_", "piperidine", "piperazine",
                                               "morpholine", "methylenedioxy"))]
_BIO_KEYS = [k for k in ALERTS if k not in _MI_KEYS]


def feature_names() -> list[str]:
    return [f"tdi_{k}" for k in ALERTS] + [f"tdi_{k}" for k in DERIVED]


_COMPILED = None


def _compile():
    global _COMPILED  # noqa: PLW0603
    if _COMPILED is None:
        from rdkit import Chem  # noqa: PLC0415
        _COMPILED = {k: Chem.MolFromSmarts(s) for k, (s, _) in ALERTS.items()}
        bad = [k for k, v in _COMPILED.items() if v is None]
        if bad:
            raise ValueError(f"invalid SMARTS for {bad}")
    return _COMPILED


def calculate(smiles) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, valid) — one row per SMILES, columns as `feature_names()`."""
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import Descriptors, rdMolDescriptors  # noqa: PLC0415

    pats = _compile()
    keys = list(ALERTS)
    n = len(keys) + len(DERIVED)
    X = np.zeros((len(smiles), n), dtype=np.float64)
    valid = np.zeros(len(smiles), dtype=bool)

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        valid[i] = True
        counts = {}
        for j, k in enumerate(keys):
            c = len(mol.GetSubstructMatches(pats[k]))
            counts[k] = c
            X[i, j] = c

        heavy = max(mol.GetNumHeavyAtoms(), 1)
        rings = mol.GetRingInfo().AtomRings()
        largest = max((len(r) for r in rings), default=0)
        n_mi = sum(counts[k] for k in _MI_KEYS)
        n_bio = sum(counts[k] for k in _BIO_KEYS)
        basic_n = counts["amine_tert_alkyl"] + counts["amine_sec_alkyl"]
        aryl = rdMolDescriptors.CalcNumAromaticRings(mol)
        reactive_ring = (counts["furan_ring"] + counts["thiophene_ring"]
                         + counts["pyrrole_ring"] + counts["thiazole_2amino"])

        derived = [
            n_mi + n_bio,
            n_mi,
            n_bio,
            float(basic_n > 0 and aryl > 0),
            float(reactive_ring > 0 and basic_n > 0),
            (n_mi + n_bio) / heavy,
            largest,
            Descriptors.NumRotatableBonds(mol) / max(len(rings), 1),
        ]
        X[i, len(keys):] = derived

    return X, valid
