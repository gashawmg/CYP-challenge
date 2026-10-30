#!/usr/bin/env python
r"""Conformer shape descriptors, computed once and cached on disk by SMILES.

Section 41 measured these as the largest single feature gain this project has found:
+0.0717 rho on CYP1A2, +0.0514 on CYP2D6, both intervals excluding zero. Every other
column in the pipeline is a 2D graph property or a SMARTS count -- we had never used
geometry at all, while the PXR pipeline carried NPR1, NPR2 and Asphericity.

WHY THESE EIGHT. NPR1/NPR2 and Asphericity place a molecule on the rod-disc-sphere axis,
which for CYP1A2's narrow planar slot IS the binding requirement expressed as a number.
Eccentricity, InertialShapeFactor, SpherocityIndex, RadiusOfGyration and PBF (plane of
best fit) add the rest of the shape picture. THE THREE PMI TERMS ARE DELIBERATELY ABSENT:
they are raw moments of inertia, dominated by size, and Section 36 measured what duplicate
size columns cost on 1,000-2,300 rows.

THE CACHE MATTERS MORE THAN IT LOOKS. Embedding 4,906 structures takes ~290 s, which is
nothing once but real if every fold, every endpoint and every PREDICT run repeats it.
Cached on disk by CANONICAL SMILES, so train and blind share entries and a rerun is free.

DETERMINISM IS NOT OPTIONAL HERE. ETKDGv3 with a fixed seed gives the same conformer for
the same input every time. Without that, a model trained on Monday's geometry and asked to
predict with Tuesday's would be scored on features it never saw -- the same class of
silent mismatch as the descriptor-width bug of 2026-09-26, but invisible, because the
shapes would still be plausible.

A FAILED EMBED BECOMES NaN, NEVER ZERO. A zero row reads to a booster as a legitimate
flat, compact, spherical molecule. Callers fill NaN with the column mean and the failure
count is reported.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

DESC_3D = ["NPR1", "NPR2", "Asphericity", "Eccentricity", "InertialShapeFactor",
           "SpherocityIndex", "RadiusOfGyration", "PBF"]
CACHE_PATH = os.path.join("artifacts", "shape3d_cache.pkl")
SEED = 0xC0FFEE
_MEM: dict[str, np.ndarray] | None = None


def n_features() -> int:
    return len(DESC_3D)


def feature_names() -> list[str]:
    return [f"shape3d_{n}" for n in DESC_3D]


def _load_cache() -> dict:
    global _MEM                                                     # noqa: PLW0603
    if _MEM is None:
        _MEM = {}
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "rb") as fh:
                    _MEM = pickle.load(fh)
            except Exception as e:                                  # noqa: BLE001
                print(f"  shape3d: cache unreadable ({e}); starting empty")
                _MEM = {}
    return _MEM


def _save_cache() -> None:
    if _MEM is None:
        return
    os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(_MEM, fh, protocol=4)
    os.replace(tmp, CACHE_PATH)      # atomic: a killed run cannot leave a torn cache


def _one(smiles: str) -> np.ndarray:
    from rdkit import Chem                                          # noqa: PLC0415
    from rdkit.Chem import AllChem, Descriptors3D                   # noqa: PLC0415
    v = np.full(len(DESC_3D), np.nan)
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return v
    m = Chem.AddHs(m)
    p = AllChem.ETKDGv3()
    p.randomSeed = SEED
    p.maxIterations = 200
    if AllChem.EmbedMolecule(m, p) != 0:
        return v
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    except Exception:                                               # noqa: BLE001
        pass
    for j, n in enumerate(DESC_3D):
        try:
            v[j] = getattr(Descriptors3D, n)(m)
        except Exception:                                           # noqa: BLE001
            pass
    return v


def calculate(smiles, verbose: bool = True) -> tuple[np.ndarray, int]:
    """(n, 8) shape matrix with NaN for failures, plus the failure count.

    Canonicalises the key so the same molecule written two ways hits one cache entry.
    """
    from rdkit import Chem                                          # noqa: PLC0415
    cache = _load_cache()
    keys, new = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        k = Chem.MolToSmiles(m) if m is not None else f"__bad__{s}"
        keys.append(k)
        if k not in cache:
            new.append(k)
    new = list(dict.fromkeys(new))
    if new:
        t0 = time.time()
        if verbose:
            print(f"  shape3d: embedding {len(new)} new structures "
                  f"({len(cache)} already cached)...", flush=True)
        for i, k in enumerate(new, 1):
            cache[k] = _one(k)
            if verbose and i % 1000 == 0:
                print(f"    {i}/{len(new)}  ({time.time()-t0:.0f}s)", flush=True)
        _save_cache()
        if verbose:
            print(f"  shape3d: done in {time.time()-t0:.0f}s, cache now {len(cache)}")
    X = np.vstack([cache[k] for k in keys])
    bad = int(np.isnan(X).all(axis=1).sum())
    if bad and verbose:
        print(f"  shape3d: {bad} of {len(X)} failed to embed (left as NaN)")
    return X, bad


def calculate_filled(smiles, verbose: bool = True) -> np.ndarray:
    """As `calculate`, with NaN replaced by that column's mean over THESE rows."""
    X, _ = calculate(smiles, verbose)
    col = np.nanmean(X, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    idx = np.where(np.isnan(X))
    X = X.copy()
    X[idx] = np.take(col, idx[1])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
