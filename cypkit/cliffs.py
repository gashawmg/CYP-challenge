"""Cliff-aware training weights: make the model pay for smoothing.

    from cypkit import cliffs
    w = cliffs.weights(smiles, y, lt, isoform="CYP2D6", alpha=1.5)

THE DEFECT THIS TARGETS, MEASURED. `scripts/activity_cliffs.py` scores, for every pair of
compounds with ECFP4 Tanimoto >= 0.45 whose measured pIC50 differ by >= 1.0, how much of
that measured gap the model reproduces:

    endpoint   measured |dy|   pseudo_blend45 keeps   v6 keeps
    CYP1A2         1.449            0.27 (17% of pairs)   0.43 (39%)
    CYP2C9         1.402            0.41 (24%)            0.53 (41%)
    CYP2D6         1.454            0.20 ( 5%)            0.32 (19%)
    CYP3A4         1.370            0.38 (29%)            0.46 (40%)

On CYP2D6 the blend reproduces a fifth of the gap and gets one pair in twenty. Both
members are being parked near their neighbourhood's average. Out-of-fold MAE on cliff
compounds runs 1.20-1.76x the non-cliff rate, worst on CYP2D6 -- the endpoint we are
furthest behind on.

The reason this survived every metric we have: putting both members of a pair equally
wrong in the SAME direction barely moves MAE and does not move Spearman at all. It is
invisible unless you look at pairs.

WHY WEIGHTS AND NOT A PAIRWISE LOSS TERM. A true within-batch pair penalty --
`(pred_i - pred_j - (y_i - y_j))^2` over similar pairs -- is the direct expression of the
defect, and it is what I would write if the plumbing were free. It is not: chemprop's loss
receives `(preds, targets, mask, weights, lt_mask, gt_mask)` and no datapoint identity, so
pairing requires either a custom sampler that co-locates partners in a batch or smuggling
indices through `x_d` (which then leaks the partner's LABEL into the features unless the
model is also patched to strip them). This project has already paid twice for clever
reimplementation -- a scaffold split that contaminated 78% of its own out-of-fold rows,
and an affine sign error that cost a submission slot -- so the cheap, native, verifiable
intervention goes first.

WHAT THE WEIGHT IS. For compound i, its LOCAL CONTRAST is how far its label sits from the
similarity-weighted mean label of its neighbours:

    contrast_i = | y_i - sum_j(s_ij * y_j) / sum_j(s_ij) |      over j with s_ij >= sim

A compound in a smooth neighbourhood has contrast ~0 and keeps weight 1. A compound whose
close analogues behave completely differently has high contrast and is upweighted:

    w_i = 1 + alpha * min(contrast_i / delta, 1)

Capped at 1 + alpha, so a single mislabelled outlier cannot dominate a fold. This raises
the cost of splitting the difference exactly where splitting the difference is what the
model currently does, and it does it through `MoleculeDatapoint(weight=...)`, which
chemprop already honours.

CENSORED ROWS ARE EXCLUDED from both the neighbour average and the upweighting. Their
label is a bound, not a value, so a "gap" against one is not a cliff -- it is the assay
floor, and treating it as a cliff would upweight the very rows whose numbers we do not
know.

HONEST LIMITS. Weighting makes cliff compounds count more; it does not directly force the
model to separate a specific pair, so expect it to move `kept` part of the way, not to
1.0. Judge it with `scripts/activity_cliffs.py` on the resulting OOF file -- `kept` and
`captured` are the metrics this exists to move, and blind-matched rho is the metric that
decides whether the trade was worth it.
"""
from __future__ import annotations

import numpy as np

SIM_DEFAULT = 0.45      # this library's median held-out neighbour sits at 0.44
DELTA_DEFAULT = 1.0     # log units that count as a cliff


def _fps(smiles):
    """ECFP4 bit vectors, tolerant of the RDKit version.

    `AllChem.GetMorganGenerator` only exists on recent builds. The repo's own working
    code (cypkit/features.py, cypkit/splits.py) reaches the generator through
    `rdkit.Chem.rdFingerprintGenerator`, which is where it has lived for far longer, so
    use that first and keep the pre-generator call as a last resort. Getting this wrong
    is a crash on someone else's environment, not a silent error -- but it is still a
    crash after the run has been queued.
    """
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")
    gen = None
    try:
        from rdkit.Chem import rdFingerprintGenerator as rdfp  # noqa: PLC0415
        gen = rdfp.GetMorganGenerator(radius=2, fpSize=2048)
    except (ImportError, AttributeError):
        from rdkit.Chem import AllChem  # noqa: PLC0415
        if hasattr(AllChem, "GetMorganGenerator"):
            gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)

    def one(m):
        if gen is not None:
            return gen.GetFingerprint(m)
        from rdkit.Chem import AllChem  # noqa: PLC0415
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        out.append(one(m) if m is not None else None)
    return out


def local_contrast(smiles, y, lt=None, sim: float = SIM_DEFAULT):
    """|label - similarity-weighted mean label of neighbours| per compound.

    Censored rows contribute nothing and receive 0. A compound with no neighbour above
    `sim` also receives 0 -- it is isolated, not a cliff.
    """
    from rdkit import DataStructs  # noqa: PLC0415

    y = np.asarray(y, float)
    lt = np.zeros(len(y), bool) if lt is None else np.asarray(lt, bool)
    fps = _fps(smiles)
    usable = np.array([f is not None for f in fps]) & np.isfinite(y) & ~lt

    idx = np.where(usable)[0]
    ref = [fps[i] for i in idx]
    out = np.zeros(len(y))
    if len(idx) < 10:
        return out, 0

    n_pairs = 0
    for pos, i in enumerate(idx):
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], ref))
        s[pos] = 0.0                                   # never its own neighbour
        near = s >= sim
        if not near.any():
            continue
        n_pairs += int(near.sum())
        w = s[near]
        out[i] = abs(y[i] - float(np.sum(w * y[idx[near]]) / np.sum(w)))
    return out, n_pairs // 2


def weights(smiles, y, lt=None, alpha: float = 1.0, sim: float = SIM_DEFAULT,
            delta: float = DELTA_DEFAULT, verbose: bool = True):
    """Multiplicative per-row weight in [1, 1 + alpha]. Censored rows stay at 1."""
    if alpha <= 0:
        return np.ones(len(y))
    c, n_pairs = local_contrast(smiles, y, lt, sim)
    w = 1.0 + alpha * np.clip(c / max(delta, 1e-6), 0.0, 1.0)
    if verbose:
        hot = c >= delta
        print(f"    cliff weighting: sim>={sim}, delta={delta}, alpha={alpha} -> "
              f"{int(hot.sum())} of {len(w)} rows at the cap ({hot.mean()*100:.1f}%), "
              f"{n_pairs} similar pairs, mean weight {w.mean():.3f}")
    return w
