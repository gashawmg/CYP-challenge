"""Where the warhead sits relative to the basic nitrogen -- the missing CYP2D6 feature.

    from cypkit import tdi_geometry as TG
    X, names = TG.calculate(smiles)

WHAT WAS MISSING. The TDI model's descriptor set carries exactly four CYP2D6-relevant
columns: `basic_amine`, `basic_amine_tertiary`, `mbi_cyclopropylamine`, and
`cyp2d6_pharmacophore` -- which is defined as "has a basic amine AND has an aromatic
ring", a binary that fires on most of the training set and therefore separates almost
nothing. The richer 18-column CYP2D6 block (basic-nitrogen typing, N-to-aromatic
topological distances, the pharmacophore window) exists in `cyp_target_descriptors.py`
but is wired only into the pIC50 models.

WHAT NEITHER OF THEM HAS. CYP2D6 recognition is a protonated nitrogen ion-pairing with
Glu216/Asp301, with the site of oxidation roughly 5-7 A away. Mechanism-based
inactivation needs BOTH: the base to dock the molecule, and a bioactivatable group
sitting where the oxidation happens. Having a methylenedioxyphenyl somewhere in the
molecule is not the same as having one positioned at the oxidation site, and a model
given the two facts separately has to learn the conjunction from data -- of which there
is almost none (21 methylenedioxy compounds in 1,497 CYP2D6 labels).

So this block computes the conjunction directly: the topological bond distance from the
nearest basic nitrogen to each warhead, and a flag for the 3-8 bond window that
corresponds to 5-7 A in an extended chain. Bond count is a conformation-free stand-in
for the spatial distance -- crude, but it is the same approximation the CYP2D6
pharmacophore block already makes, and it costs nothing.

Nothing here is fitted, so nothing can leak. `validate()` checks the block fires on the
canonical CYP2D6 inactivators (paroxetine, MDMA) and stays quiet on molecules that
carry a warhead but no base, or a base but no warhead.
"""
from __future__ import annotations

import numpy as np

# Warheads that CYP oxidises to something that does not leave. Kept deliberately short:
# every one of these has documented CYP2D6 or CYP3A4 mechanism-based inactivation
# behind it, rather than being a general reactivity filter.
WARHEADS = {
    "methylenedioxy": "[cX3]1[cX3][cX3][cX3]2OC(O[cX3]2[cX3]1)",   # paroxetine, MDMA
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "terminal_alkyne": "[CX2H1]#[CX2]",
    "cyclopropylamine": "[NX3][CH1]1[CH2][CH2]1",
    "thioether": "[#6][SX2][#6]",
    "aniline": "[NX3;H2,H1;!$(NC=[O,S,N])][c]",
    "n_methyl": "[NX3;!$(N[a]);!$(NC=[O,S,N])][CH3]",             # N-dealkylation -> MI complex
    "nitroaromatic": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][c]",
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",
    "hydrazine": "[NX3][NX3]",
}

# A basic nitrogen: trivalent, not attached to an aromatic ring, not an amide/imine.
# The exclusions matter -- an anilinic or amide nitrogen is not protonated at
# physiological pH and does not make the CYP2D6 ion pair.
BASIC_N = "[NX3;H0,H1,H2;!$(N[a]);!$(NC=[O,S,N]);!$(N=*);!$(N#*);!$([N+](=O)[O-])]"

# 5-7 A between the basic nitrogen and the site of oxidation is the published CYP2D6
# geometry; in an extended chain that is roughly 3-8 bonds. Wide on purpose -- a
# topological count cannot resolve better than that, and a window too narrow would
# simply never fire.
WINDOW = (3, 8)

_CACHE: dict = {}


def _compile():
    if not _CACHE:
        from rdkit import Chem  # noqa: PLC0415
        _CACHE["basic"] = Chem.MolFromSmarts(BASIC_N)
        _CACHE["war"] = {k: Chem.MolFromSmarts(v) for k, v in WARHEADS.items()}
        bad = [k for k, v in _CACHE["war"].items() if v is None]
        if bad or _CACHE["basic"] is None:
            raise ValueError(f"invalid SMARTS: {bad or 'BASIC_N'}")
    return _CACHE


def feature_names() -> list[str]:
    out = ["n_basic_n", "has_basic_n", "basic_n_tertiary", "lipophilic_base"]
    for w in WARHEADS:
        out += [f"d_baseN_{w}", f"win_baseN_{w}"]
    out += ["d_baseN_warhead_min", "n_warheads_in_window", "base_and_warhead",
            "warhead_no_base", "base_no_warhead"]
    return out


def _min_path(mol, a_idx, b_idx, dmat):
    return min((dmat[a][b] for a in a_idx for b in b_idx), default=np.nan)


def calculate(smiles, standardise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(n, len(feature_names())) matrix and a validity mask.

    Same contract as the other descriptor blocks: a molecule that fails to parse is
    flagged invalid rather than silently zeroed.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import Crippen, rdmolops  # noqa: PLC0415

    pats = _compile()
    names = feature_names()
    X = np.full((len(smiles), len(names)), np.nan, float)
    valid = np.zeros(len(smiles), bool)
    # a distance of "no path" has to be a number the model can use; the largest
    # plausible molecular diameter is a more honest filler than 0, which would read as
    # "the warhead is ON the nitrogen"
    FAR = 30.0

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        try:
            if standardise:
                from .features import _standardise  # noqa: PLC0415
                mol, _ = _standardise(mol)
            base = [m[0] for m in mol.GetSubstructMatches(pats["basic"], uniquify=True)]
            dmat = rdmolops.GetDistanceMatrix(mol)
            row = {}
            row["n_basic_n"] = float(len(base))
            row["has_basic_n"] = float(len(base) > 0)
            row["basic_n_tertiary"] = float(sum(
                1 for b in base if mol.GetAtomWithIdx(b).GetTotalNumHs() == 0))
            # the classic CYP2D6 substrate signature: a lipophilic base
            row["lipophilic_base"] = float(len(base) > 0 and Crippen.MolLogP(mol) > 2.0)

            dists, in_win = [], 0
            any_war = False
            for w, pat in pats["war"].items():
                hits = [a for m in mol.GetSubstructMatches(pat, uniquify=True) for a in m]
                if hits:
                    any_war = True
                if not hits or not base:
                    row[f"d_baseN_{w}"] = FAR
                    row[f"win_baseN_{w}"] = 0.0
                    continue
                d = _min_path(mol, base, hits, dmat)
                d = FAR if not np.isfinite(d) else float(d)
                row[f"d_baseN_{w}"] = d
                hit_win = float(WINDOW[0] <= d <= WINDOW[1])
                row[f"win_baseN_{w}"] = hit_win
                in_win += int(hit_win)
                dists.append(d)
            row["d_baseN_warhead_min"] = float(min(dists)) if dists else FAR
            row["n_warheads_in_window"] = float(in_win)
            row["base_and_warhead"] = float(bool(base) and any_war)
            row["warhead_no_base"] = float(any_war and not base)
            row["base_no_warhead"] = float(bool(base) and not any_war)
            X[i] = [row[n] for n in names]
            valid[i] = True
        except Exception:  # noqa: BLE001
            continue
    return X, valid


def validate(verbose: bool = True) -> bool:
    """Fire on the canonical CYP2D6 inactivators, stay quiet without the conjunction."""
    cases = {
        # paroxetine: methylenedioxyphenyl + piperidine NH, the textbook 2D6 MBI
        "paroxetine": ("FC1=CC=C(C2CCNCC2COC2=CC3=C(OCO3)C=C2)C=C1",
                       "win_baseN_methylenedioxy", 1.0),
        # MDMA: methylenedioxy + secondary amine
        "MDMA": ("CC(NC)CC1=CC2=C(OCO2)C=C1", "win_baseN_methylenedioxy", 1.0),
        # safrole-like: the warhead, but no basic nitrogen to dock it
        "no base": ("C=CCC1=CC2=C(OCO2)C=C1", "warhead_no_base", 1.0),
        # a plain lipophilic base with nothing to bioactivate
        "no warhead": ("CCCCN(CC)CCCC", "base_no_warhead", 1.0),
    }
    names = feature_names()
    ok = True
    for label, (smi, feat, want) in cases.items():
        X, v = calculate([smi])
        got = X[0, names.index(feat)] if v[0] else np.nan
        good = bool(v[0]) and abs(got - want) < 1e-6
        ok &= good
        if verbose:
            print(f"  {'ok ' if good else 'FAIL'} {label:11s} {feat:26s} "
                  f"= {got} (want {want})")
    return bool(ok)


if __name__ == "__main__":
    print("tdi_geometry self-check")
    raise SystemExit(0 if validate() else 1)
