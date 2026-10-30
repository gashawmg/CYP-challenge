"""Analog-series cross-validation folds, frozen to disk.

The test set is purchased analogues of confirmed screening hits, sitting at roughly
0.50-0.55 max ECFP4 Tanimoto to the training set. A random split puts near-neighbours on
both sides and reports a number that has nothing to do with the task.

So rather than guessing a Butina cutoff, `calibrate_cutoff` measures the real
train-to-test similarity gap and picks the cutoff whose held-out folds reproduce it.
The folds are then written to disk and never regenerated, so every experiment in the
project is compared on identical splits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def _rdkit():
    from rdkit import Chem, DataStructs  # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator  # noqa: PLC0415

    return Chem, DataStructs, rdFingerprintGenerator


def fingerprints(smiles, radius: int = 2, n_bits: int = 2048):
    """ECFP4 bit vectors; None for SMILES RDKit cannot parse."""
    Chem, _, rdfp = _rdkit()
    gen = rdfp.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        out.append(gen.GetFingerprint(mol) if mol is not None else None)
    return out


def max_similarity(query_fps, reference_fps) -> np.ndarray:
    """For each query fingerprint, its maximum Tanimoto to the reference set."""
    _, DataStructs, _ = _rdkit()
    ref = [f for f in reference_fps if f is not None]
    if not ref:
        return np.full(len(query_fps), np.nan)
    out = np.full(len(query_fps), np.nan)
    for i, fp in enumerate(query_fps):
        if fp is None:
            continue
        out[i] = max(DataStructs.BulkTanimotoSimilarity(fp, ref))
    return out


# ---------------------------------------------------------------------------
# Butina clustering
# ---------------------------------------------------------------------------


def butina_clusters(fps, cutoff: float) -> np.ndarray:
    """Cluster labels from RDKit's Butina algorithm at a Tanimoto *distance* cutoff.

    Unparseable molecules each become their own singleton cluster so indices stay
    aligned with the input.
    """
    from rdkit.ML.Cluster import Butina  # noqa: PLC0415

    _, DataStructs, _ = _rdkit()
    valid_idx = [i for i, f in enumerate(fps) if f is not None]
    valid = [fps[i] for i in valid_idx]

    dists: list[float] = []
    for i in range(1, len(valid)):
        sims = DataStructs.BulkTanimotoSimilarity(valid[i], valid[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(dists, len(valid), cutoff, isDistData=True)

    labels = np.full(len(fps), -1, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            labels[valid_idx[m]] = cid
    next_id = len(clusters)
    for i in np.where(labels < 0)[0]:
        labels[i] = next_id
        next_id += 1
    return labels


def assign_folds(cluster_labels: np.ndarray, n_folds: int, seed: int = 42) -> np.ndarray:
    """Greedy balancing: largest clusters first, each into the currently smallest fold."""
    rng = np.random.default_rng(seed)
    sizes = pd.Series(cluster_labels).value_counts()
    order = sizes.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).sort_values(
        ascending=False, kind="stable"
    )
    fold_of_cluster: dict[int, int] = {}
    load = np.zeros(n_folds, dtype=int)
    for cid, size in order.items():
        f = int(np.argmin(load))
        fold_of_cluster[cid] = f
        load[f] += size
    return np.array([fold_of_cluster[c] for c in cluster_labels], dtype=int)


# ---------------------------------------------------------------------------
# Cutoff calibration
# ---------------------------------------------------------------------------


def _held_out_similarity(fps, folds: np.ndarray) -> np.ndarray:
    """Max Tanimoto from each compound to the compounds in the *other* folds."""
    out = np.full(len(fps), np.nan)
    for f in np.unique(folds):
        held = np.where(folds == f)[0]
        rest = np.where(folds != f)[0]
        out[held] = max_similarity([fps[i] for i in held], [fps[i] for i in rest])
    return out


def calibrate_cutoff(
    train_smiles,
    test_smiles,
    n_folds: int = 5,
    candidates=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    seed: int = 42,
) -> pd.DataFrame:
    """Compare candidate Butina cutoffs against the real train-to-test similarity gap.

    Returns a table with the target median similarity and what each cutoff achieves,
    so the choice is a measurement rather than a convention.
    """
    train_fps = fingerprints(train_smiles)
    test_fps = fingerprints(test_smiles)
    target = float(np.nanmedian(max_similarity(test_fps, train_fps)))

    rows = []
    for cutoff in candidates:
        labels = butina_clusters(train_fps, cutoff)
        folds = assign_folds(labels, n_folds, seed=seed)
        sim = _held_out_similarity(train_fps, folds)
        med = float(np.nanmedian(sim))
        rows.append(
            dict(
                cutoff=cutoff,
                n_clusters=int(len(np.unique(labels))),
                median_heldout_similarity=med,
                gap_to_test=abs(med - target),
                largest_fold=int(np.bincount(folds).max()),
                smallest_fold=int(np.bincount(folds).min()),
            )
        )
    out = pd.DataFrame(rows).sort_values("gap_to_test").reset_index(drop=True)
    out.attrs["target_median_similarity"] = target
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def make_folds(
    train: pd.DataFrame,
    cutoff: float,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    fps = fingerprints(train[C.SMILES_COL])
    labels = butina_clusters(fps, cutoff)
    folds = assign_folds(labels, n_folds, seed=seed)
    sim = _held_out_similarity(fps, folds)
    return pd.DataFrame(
        {
            C.ID_COL: train[C.ID_COL].to_numpy(),
            C.SMILES_COL: train[C.SMILES_COL].to_numpy(),
            "cluster": labels,
            "fold": folds,
            "max_sim_to_other_folds": sim,
        }
    )


def assign_extra_to_folds(
    extra_smiles,
    folds: pd.DataFrame,
    batch: int = 256,
) -> np.ndarray:
    """Place compounds that have no measured label into the fold of their nearest
    labelled neighbour.

    Primary-screen non-hits become censored training rows but are not scored, so they
    never appear in the fold table. They still must not sit in the training half of a
    fold whose held-out compounds are their close analogues, or the split leaks.
    """
    _, DataStructs, _ = _rdkit()
    ref_fps = fingerprints(folds[C.SMILES_COL])
    ref_fold = folds["fold"].to_numpy()
    keep = [i for i, f in enumerate(ref_fps) if f is not None]
    ref_fps = [ref_fps[i] for i in keep]
    ref_fold = ref_fold[keep]

    out = np.full(len(extra_smiles), -1, dtype=int)
    query = fingerprints(extra_smiles)
    for i, fp in enumerate(query):
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        out[i] = int(ref_fold[int(np.argmax(sims))])
    # Anything unassignable is spread round-robin rather than silently dropped.
    unassigned = np.where(out < 0)[0]
    if len(unassigned):
        n_folds = int(folds["fold"].max()) + 1
        out[unassigned] = np.arange(len(unassigned)) % n_folds
    return out


def save_folds(folds: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)
    return path


def load_folds(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/make_folds.py first. Folds are frozen on "
            "purpose: regenerating them mid-project makes every earlier number "
            "incomparable."
        )
    return pd.read_csv(path)
