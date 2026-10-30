#!/usr/bin/env python
r"""The per-endpoint descriptor specification, and the one place that assembles it.

WHY PER-ENDPOINT. Until Section 41 every CYP model here used one descriptor recipe for all
four isoforms and varied only the weights. Two measurements broke that:

  Section 40 -- the benefit of NARROWING the general block scales with scarcity, in
  perfect rank order: +0.0493 CYP2D6 (1,364 fitted rows), +0.0315 CYP1A2 (1,444),
  +0.0234 CYP2C9 (1,983), -0.0028 CYP3A4 (4,199). CYP3A4 is the only endpoint with
  PXR-scale data and the only one that did not want the cut.

  Section 41 -- 3D shape helps in the order the pharmacology predicts: +0.0717 CYP1A2
  (narrow planar slot, and NPR1/NPR2/Asphericity ARE planarity), +0.0514 CYP2D6 (small
  constrained site), -0.0181 CYP2C9 (recognition is charge-driven via Arg108, not shape),
  +0.0093 CYP3A4 (large promiscuous pocket, least shape-selective).

So the four endpoints want different FEATURE SETS, not different weights on one set:

    CYP1A2   lean + 3D            established: both yardsticks agree (+0.072, +0.068)
    CYP2D6   lean + 3D            promising: reweighted +0.051, holdout -0.013, disagree
    CYP2C9   lean                 3D refuted; width untested and unlikely at 1,983 rows
    CYP3A4   lean + wide + 3D     +0.0118 [+0.0016,+0.0229] for the pair; neither alone

HONEST STATUS OF EACH LINE. CYP1A2 is corroborated by two independent yardsticks. CYP3A4
rests on one arm of a three-arm family, so roughly a one-in-seven chance of being a false
positive. CYP2D6's two yardsticks disagree and it is included because the reweighted one
is the better-calibrated of the two on that endpoint (Section 32 ratio 1.036 for plain,
and Section 39 put the holdout at 0.96) -- reasonable, not settled. CYP2C9 changes nothing.

THE WIDE BLOCK restores the 17 general columns Section 36 removed plus a PXR-style
VSA/substructure set, approaching PXR's 57 general descriptors. It is applied ONLY where
the data supports it.
"""
from __future__ import annotations

import numpy as np

import cyp_target_descriptors as TD
from cyp_target_descriptors_lean import lean_index, lean_names
from cypkit import shape3d as S3

# endpoint -> (use_wide, use_3d)
#
# REVISED 2026-09-27 AFTER MEASURING IT IN CHEMPROP (Section 43). The table below was
# first set from LightGBM arms (Section 41). Run it through the real model and only ONE
# line survives:
#
#     endpoint   chemprop SPEC - LEAN      holdout      verdict
#     CYP1A2       +0.0753               +0.1224        3D KEPT -- large, both agree
#     CYP2D6       -0.1518               -0.1057        3D REMOVED -- helped LightGBM,
#                                                       hurts the D-MPNN badly
#     CYP3A4       -0.0167               -0.0298        wide+3D REMOVED -- not worth it
#     CYP2C9       -0.0598               -0.0067        control: IDENTICAL features both
#                                                       runs, so this is run-to-run noise
#
# The CYP2C9 row is the important one: same 38 columns, same data, same code, and the two
# runs still correlate only r = 0.927. A chemprop feature comparison from a single pair of
# runs cannot resolve anything smaller than that. CYP1A2's gain clears it on the holdout by
# a factor of 18; CYP2D6's loss clears it too, in the wrong direction.
SPEC = {
    "CYP1A2": (False, True),     # measured +0.1224 on the holdout
    "CYP2C9": (False, False),
    "CYP2D6": (False, False),    # was (False, True); 3D refuted in chemprop
    "CYP3A4": (False, False),    # was (True, True); wide+3D refuted in chemprop
}
WIDE_RDKIT = ["MolWt", "MolMR", "BertzCT", "Kappa1", "Kappa2", "Chi0v", "Chi1v", "Chi2v",
              "HallKierAlpha", "LabuteASA", "MaxEStateIndex", "NumHeteroatoms",
              "NHOHCount", "NOCount", "NumSaturatedRings", "NumAliphaticRings", "qed",
              "SMR_VSA5", "SMR_VSA7", "SlogP_VSA2", "SlogP_VSA5", "PEOE_VSA6",
              "PEOE_VSA7", "NumAliphaticCarbocycles", "NumAromaticCarbocycles",
              "NumAromaticHeterocycles", "NumSaturatedHeterocycles", "HeavyAtomMolWt"]
WIDE_SMARTS = {
    "num_Cl": "[Cl]", "num_F": "[F]", "num_Br": "[Br]", "num_carbonyl": "[CX3]=[OX1]",
    "num_ketone": "[#6][CX3](=O)[#6]", "num_ether_O": "[OD2]([#6])[#6]",
    "num_hydroxyl": "[OX2H]", "num_sulfoxide": "[$([#16X3]=[OX1])]",
    "num_NH": "[NX3;H1,H2]", "num_michael": "[CX3]=[CX3][CX3]=[OX1]",
}
_WIDE_CACHE: dict = {}


def wide_block(smiles):
    from rdkit import Chem                                          # noqa: PLC0415
    from rdkit.Chem import Descriptors                              # noqa: PLC0415
    if "pats" not in _WIDE_CACHE:
        _WIDE_CACHE["pats"] = {k: Chem.MolFromSmarts(v) for k, v in WIDE_SMARTS.items()}
        _WIDE_CACHE["fns"] = [(n, getattr(Descriptors, n)) for n in WIDE_RDKIT
                              if hasattr(Descriptors, n)]
    pats, fns = _WIDE_CACHE["pats"], _WIDE_CACHE["fns"]
    X = np.zeros((len(smiles), len(fns) + len(pats)), float)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        for j, (_, f) in enumerate(fns):
            try:
                X[i, j] = f(m)
            except Exception:                                       # noqa: BLE001
                pass
        for j, p in enumerate(pats.values()):
            X[i, len(fns) + j] = len(m.GetSubstructMatches(p))
    names = [f"wide_{n}" for n, _ in fns] + [f"wide_{k}" for k in pats]
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def shared_names() -> list[str]:
    """The lean columns present in EVERY isoform's block, in CYP1A2's column order.

    WHY AN INTERSECTION RATHER THAN A HAND-WRITTEN LIST. The multi-head model in
    train_cyp_mh.py has ONE shared encoder over ~20,000 compounds drawn from all four
    assays plus external chemistry, so it cannot take a per-isoform descriptor block --
    there is no isoform to key on for an external row. Taking the intersection of the four
    lean blocks keeps every column that is already validated (Section 36/40) and drops
    exactly the isoform-specific ones, with no new hand-maintained list to drift out of
    sync when cyp_target_descriptors_lean.py changes.

    Isoform specificity does not disappear; it moves to where this architecture puts it
    anyway. The trunk gets shared chemistry, the per-assay heads learn the rest.
    """
    isos = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
    common = set.intersection(*(set(lean_names(i)) for i in isos))
    return [n for n in lean_names("CYP1A2") if n in common]


def build_shared(smiles, use_3d: bool = False, verbose: bool = True):
    """(X, names) for the isoform-agnostic block. Same selection path as build()."""
    want = shared_names()
    all_names = TD.feature_names("CYP1A2")
    idx = [j for j, n in enumerate(all_names) if n in set(want)]
    # both `want` and `idx` are built in all_names order, so they line up; assert it
    # rather than trust it, because a silent column/name mismatch here would train the
    # model on one thing and describe it as another.
    assert [all_names[j] for j in idx] == want, "shared block column order mismatch"

    Xa, _ = TD.calculate(smiles, "CYP1A2")
    X = np.nan_to_num(np.asarray(Xa, float))[:, idx]
    names = list(want)
    if use_3d:
        X = np.hstack([X, S3.calculate_filled(smiles, verbose=verbose)])
        names += S3.feature_names()
    assert X.shape[1] == len(names), f"{X.shape[1]} columns vs {len(names)} names"
    if verbose:
        print(f"  shared descriptor block: {X.shape[1]} columns = {len(want)} lean-common"
              + (f" + {S3.n_features()} shape3d" if use_3d else ""))
    return X, names


def spec_for(isoform: str, mode: str = "per_endpoint") -> tuple[bool, bool]:
    """(use_wide, use_3d). `mode` lets a run override the table for an A/B."""
    if mode == "lean":
        return (False, False)
    if mode == "lean3d":
        return (False, True)
    if mode == "wide3d":
        return (True, True)
    return SPEC.get(isoform, (False, False))


def build(smiles, isoform: str, mode: str = "per_endpoint", verbose: bool = True):
    """The full descriptor matrix for this endpoint under this spec, plus column names.

    The lean block is SELECTED BY NAME from the same calculate() output the full block
    uses, so lean and full remain directly comparable and nothing is recomputed.
    """
    use_wide, use_3d = spec_for(isoform, mode)
    Xa, _ = TD.calculate(smiles, isoform)
    blocks = [np.nan_to_num(np.asarray(Xa, float))[:, lean_index(isoform)]]
    names = list(lean_names(isoform))
    if use_wide:
        W, wn = wide_block(smiles)
        blocks.append(W); names += wn
    if use_3d:
        blocks.append(S3.calculate_filled(smiles, verbose=verbose))
        names += S3.feature_names()
    X = np.hstack(blocks)
    assert X.shape[1] == len(names), f"{X.shape[1]} columns vs {len(names)} names"
    if verbose:
        print(f"  descriptor spec [{isoform}] mode={mode}: {X.shape[1]} columns "
              f"= {len(lean_names(isoform))} lean"
              + (f" + {len(wn)} wide" if use_wide else "")
              + (f" + {S3.n_features()} shape3d" if use_3d else ""))
    return X, names


if __name__ == "__main__":
    for iso in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"):
        w, d = spec_for(iso)
        print(f"  {iso}: lean{' + wide' if w else ''}{' + 3D' if d else ''}")
