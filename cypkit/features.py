"""Molecular features.

Two rules learned the hard way on the PXR project:

1. Never compute a 3D descriptor on a molecule with no conformer. RDKit raises
   "molecule has no conformers", a bare `except` swallows it, and the feature is a
   silent zero for every row. `add_3d` embeds first and reports failures out loud.
2. Never let a featuriser fail quietly. Unparseable SMILES are returned as a mask, not
   as a row of zeros indistinguishable from a real measurement.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_DESC_CACHE: list[str] | None = None


def _rdkit():
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import Descriptors  # noqa: PLC0415

    return Chem, Descriptors


def descriptor_names() -> list[str]:
    global _DESC_CACHE
    if _DESC_CACHE is None:
        _, Descriptors = _rdkit()
        _DESC_CACHE = [n for n, _ in Descriptors._descList]
    return list(_DESC_CACHE)


def _make_calculator(names: list[str]):
    """Return a per-molecule descriptor function, whichever RDKit this is.

    `Descriptors.CalcMolDescriptors` only exists from RDKit 2023.09. Older builds have
    `Descriptors._descList`, which is what `descriptor_names()` already reads, so fall
    back to calling those functions directly.
    """
    _, Descriptors = _rdkit()
    fns = dict(Descriptors._descList)

    def manual(mol):
        return [fns[n](mol) for n in names]

    if not hasattr(Descriptors, "CalcMolDescriptors"):
        return manual, "Descriptors._descList (RDKit < 2023.09)"

    def modern(mol):
        vals = Descriptors.CalcMolDescriptors(mol)
        return [vals.get(n, np.nan) for n in names]

    return modern, "Descriptors.CalcMolDescriptors"


def rdkit_2d(
    smiles,
    standardise: bool = True,
    strict: bool = True,
    min_valid_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """All RDKit 2D descriptors.

    Returns (X, valid_mask). Rows where the molecule could not be featurised are zero
    and flagged False.

    `strict` (default) raises when fewer than `min_valid_fraction` of molecules
    featurised, and the message names the failure mode and quotes the first real
    exception. Without it a broken RDKit install produces an all-zero matrix, every
    model predicts the training mean, and the scores look merely bad rather than wrong.
    """
    Chem, _ = _rdkit()
    names = descriptor_names()
    calc, calc_kind = _make_calculator(names)

    X = np.full((len(smiles), len(names)), np.nan, dtype=np.float64)
    valid = np.zeros(len(smiles), dtype=bool)
    failures: dict[str, int] = {"parse": 0, "standardise": 0, "descriptor": 0}
    first_error: dict[str, str] = {}

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            failures["parse"] += 1
            continue

        if standardise:
            cleaned, err = _standardise(mol)
            if err is not None:
                failures["standardise"] += 1
                first_error.setdefault("standardise", err)
            # Standardisation is a nicety. If it fails we featurise the original
            # molecule rather than dropping the row.
            mol = cleaned

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                row = calc(mol)
            except Exception as exc:
                failures["descriptor"] += 1
                first_error.setdefault("descriptor", f"{type(exc).__name__}: {exc}")
                continue

        X[i] = row
        valid[i] = True

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    frac = float(valid.mean()) if len(valid) else 0.0
    if failures["standardise"]:
        warnings.warn(
            f"standardisation failed for {failures['standardise']} molecule(s); "
            f"featurised the originals instead. First: {first_error['standardise']}",
            stacklevel=2,
        )
    if strict and frac < min_valid_fraction:
        detail = ", ".join(f"{k}={v}" for k, v in failures.items() if v)
        raise RuntimeError(
            f"Only {frac:.1%} of {len(smiles)} molecules featurised "
            f"(needed {min_valid_fraction:.0%}).\n"
            f"  failure counts : {detail or 'none recorded'}\n"
            f"  descriptor path: {calc_kind} ({len(names)} descriptors)\n"
            f"  first error    : {first_error.get('descriptor', 'n/a')}\n"
            "Continuing would produce an all-zero feature matrix, every model would "
            "predict the training mean, and ST-RAE would sit at 1.0 while looking like "
            "a real result. Fix the descriptor path, or pass strict=False if you "
            "genuinely want the zero rows."
        )
    if frac < 1.0:
        warnings.warn(
            f"{int((~valid).sum())}/{len(smiles)} molecules failed featurisation "
            f"({', '.join(f'{k}={v}' for k, v in failures.items() if v)})",
            stacklevel=2,
        )
    return X, valid


def _standardise(mol):
    """Largest fragment, neutralised — the PXR MoleculeStandardizer, ported.

    Returns (mol, error). On failure the *original* molecule comes back alongside an
    error string, so the caller can report the problem without losing the row.
    Standardisation is a nicety; failing it is not a reason to drop a compound.
    """
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: PLC0415
    except ImportError as exc:
        return mol, f"MolStandardize unavailable: {exc}"
    try:
        out = rdMolStandardize.LargestFragmentChooser().choose(mol)
        out = rdMolStandardize.Uncharger().uncharge(out)
        if out is None:
            return mol, "standardiser returned None"
        return out, None
    except Exception as exc:
        return mol, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# 3D block — only with real conformers
# ---------------------------------------------------------------------------

_3D_NAMES = (
    "NPR1", "NPR2", "Asphericity", "Eccentricity", "InertialShapeFactor",
    "RadiusOfGyration", "SpherocityIndex", "PBF",
)


def embed_conformer(mol, seed: int = 42, max_attempts: int = 5):
    """ETKDGv3 with seeded retries, plain-ETKDG fallback, MMFF94 minimisation.

    Ported from the PXR UniMol notebook, which is the one place in that project where
    conformer generation was done correctly.
    """
    from rdkit.Chem import AllChem  # noqa: PLC0415

    Chem, _ = _rdkit()
    mol_h = Chem.AddHs(mol)
    ok = -1
    for attempt in range(max_attempts):
        p = AllChem.ETKDGv3()
        p.randomSeed = seed + attempt
        p.numThreads = 0
        p.enforceChirality = True
        p.useSmallRingTorsions = True
        ok = AllChem.EmbedMolecule(mol_h, p)
        if ok == 0:
            break
    if ok != 0:
        ok = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
    if ok != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        pass
    return mol_h


def rdkit_3d(smiles, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Shape descriptors computed on a real embedded conformer.

    Returns (X, valid_mask). If a conformer cannot be generated the row is flagged
    False rather than zero-filled — a zero here is indistinguishable from a genuinely
    spherical molecule, which is how the PXR pipeline lost three descriptors silently.
    """
    from rdkit.Chem import rdMolDescriptors as rdMD  # noqa: PLC0415

    Chem, _ = _rdkit()
    X = np.full((len(smiles), len(_3D_NAMES)), np.nan, dtype=np.float64)
    valid = np.zeros(len(smiles), dtype=bool)
    fns = {
        "NPR1": rdMD.CalcNPR1, "NPR2": rdMD.CalcNPR2,
        "Asphericity": rdMD.CalcAsphericity, "Eccentricity": rdMD.CalcEccentricity,
        "InertialShapeFactor": rdMD.CalcInertialShapeFactor,
        "RadiusOfGyration": rdMD.CalcRadiusOfGyration,
        "SpherocityIndex": rdMD.CalcSpherocityIndex, "PBF": rdMD.CalcPBF,
    }

    n_failed = 0
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            n_failed += 1
            continue
        mol_h = embed_conformer(mol, seed=seed)
        if mol_h is None:
            n_failed += 1
            continue
        try:
            X[i] = [fns[n](mol_h) for n in _3D_NAMES]
            valid[i] = True
        except Exception:
            n_failed += 1

    if n_failed:
        warnings.warn(
            f"3D features unavailable for {n_failed}/{len(smiles)} molecules "
            "(conformer generation failed). Those rows are flagged invalid, not "
            "zero-filled.",
            stacklevel=2,
        )
    return X, valid


def feature_frame(smiles, include_3d: bool = False, seed: int = 42) -> pd.DataFrame:
    """Convenience wrapper returning a named DataFrame."""
    X2, valid2 = rdkit_2d(smiles)
    df = pd.DataFrame(X2, columns=descriptor_names())
    df["_valid_2d"] = valid2
    if include_3d:
        X3, valid3 = rdkit_3d(smiles, seed=seed)
        for j, n in enumerate(_3D_NAMES):
            df[n] = X3[:, j]
        df["_valid_3d"] = valid3
    return df


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------
#
# Phase 1 shipped without any of these, which in hindsight is the most obvious gap in
# it: gradient boosting on substructure bits is the standard strong baseline for this
# kind of task, and the PXR project's own classical-ML stack used ECFP4 throughout.

FP_SIZES = {"ecfp": 2048, "ecfpc": 1024, "fcfp": 2048, "maccs": 167, "ap": 2048,
            "avalon": 2048,
            # Added for the CYP2D6 TDI blend. Section 54 measured that what pays there is
            # DISAGREEMENT with the Morgan family, not accuracy: svc_avalon scores below
            # knn_fcfp (0.6116 against 0.6283) and still gains +0.050 MCC, because its
            # rank correlation with knn_fcfp is 0.369 against the others' 0.503. These
            # four are each a different enumeration of substructure, chosen on that
            # basis rather than on expected single-model performance.
            "rdkpath": 2048, "tt": 2048, "pattern": 2048, "layered": 2048,
            # Avalon at the size train_tdi_chemprop.py actually benchmarked (0.6043 on
            # CYP2D6). features.py builds 2048 for "avalon", so this is NOT a duplicate.
            "avalon1k": 1024}


def fingerprint_block(smiles, kind: str = "ecfp", radius: int = 2):
    """One fingerprint block. Returns (X, valid_mask, names).

    kind:
        ecfp   Morgan radius 2, 2048 binary bits — the workhorse
        ecfpc  Morgan radius 2 folded *counts*, 1024 — keeps multiplicity
        fcfp   Morgan radius 2 with feature invariants (pharmacophoric)
        maccs  166 structural keys + index 0
        ap     atom pairs, 2048 binary — encodes topological distance
        avalon Avalon 2048 binary — a path/feature hybrid from a DIFFERENT enumeration
               than Morgan, so it tends to disagree with ecfp rather than duplicate it,
               which is what a blend actually wants. Note it takes a BIT COUNT, not a
               radius: there is no radius parameter to set.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator as rdfp  # noqa: PLC0415
    from rdkit.Chem import MACCSkeys  # noqa: PLC0415

    n = FP_SIZES[kind]
    gen = None
    if kind in ("ecfp", "ecfpc"):
        gen = rdfp.GetMorganGenerator(radius=radius, fpSize=n)
    elif kind == "fcfp":
        inv = rdfp.GetMorganFeatureAtomInvGen()
        gen = rdfp.GetMorganGenerator(radius=radius, fpSize=n, atomInvariantsGenerator=inv)
    elif kind == "ap":
        gen = rdfp.GetAtomPairGenerator(fpSize=n)
    elif kind == "rdkpath":
        gen = rdfp.GetRDKitFPGenerator(fpSize=n)        # linear paths, not circular
    elif kind == "tt":
        gen = rdfp.GetTopologicalTorsionGenerator(fpSize=n)   # 4-atom torsion paths
    elif kind in ("avalon", "avalon1k", "pattern", "layered"):
        gen = None          # each of these has its own function, not a generator object

    X = np.zeros((len(smiles), n), dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        mol, _ = _standardise(mol)
        try:
            if kind in ("avalon", "avalon1k"):
                from rdkit.Avalon import pyAvalonTools as _pa  # noqa: PLC0415
                fp = _pa.GetAvalonFP(mol, nBits=n)
                X[i] = np.frombuffer(bytes(fp.ToBitString(), "ascii"), "u1") - ord("0")
            elif kind == "pattern":
                fp = Chem.PatternFingerprint(mol, fpSize=n)
                X[i] = np.frombuffer(bytes(fp.ToBitString(), "ascii"), "u1") - ord("0")
            elif kind == "layered":
                fp = Chem.LayeredFingerprint(mol, fpSize=n)
                X[i] = np.frombuffer(bytes(fp.ToBitString(), "ascii"), "u1") - ord("0")
            elif kind == "maccs":
                fp = MACCSkeys.GenMACCSKeys(mol)
                X[i] = np.frombuffer(bytes(fp.ToBitString(), "ascii"), "u1") - ord("0")
            elif kind == "ecfpc":
                X[i] = gen.GetCountFingerprintAsNumPy(mol)
            else:
                X[i] = gen.GetFingerprintAsNumPy(mol)
            valid[i] = True
        except Exception:
            continue
    return X, valid, [f"{kind}_{j}" for j in range(n)]
