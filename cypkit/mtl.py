"""Multi-task network with censored losses.

Phase 1 trained four independent single-task gradient-boosted trees on the four pIC50
endpoints and threw away every other column in the dataset. This shares one trunk across
all of them and adds the supervision that was sitting unused:

    pIC50 direct   4 heads, interval/left-censored dead-zone loss  (the scored targets)
    log2fc         4 heads, L1 — the primary screen, the ONLY dense label matrix here
    TDI-arm pIC50  4 heads, L1 — reported for all four isoforms
    Emax           4 heads, L1
    is_TDI         4 heads, BCE — labels exist for all four isoforms in TRAIN_Emax,
                   not just the two that are scored

Every loss is masked, so a compound contributes to exactly the heads it has labels for.
That is the point: CYP2D6 is data-starved on both tracks, and a shared trunk lets its
head borrow structure from the isoforms that aren't.

The dead-zone loss is the same one as `losses.py`, in torch: zero inside
[y_low, y_high], distance to the nearest edge outside. A left-censored row carries
y_low = -LARGE, so it is free below the floor and charged above it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

LARGE = 1.0e4  # stand-in for -inf on the censored side


def _torch():
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    return torch, nn


# ---------------------------------------------------------------------------
# Target assembly
# ---------------------------------------------------------------------------


def build_targets(
    ids: list[str],
    label_matrix: pd.DataFrame,
    screen: pd.DataFrame | None = None,
    tdi_frame: pd.DataFrame | None = None,
    emax: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """Wide target arrays aligned to `ids`, with a mask per head.

    Returns arrays of shape (n_compounds, 4) in C.ISOFORMS order:
        pic50_lo, pic50_hi, pic50_mask   the censored band
        log2fc, log2fc_mask
        tdiarm, tdiarm_mask
        emax, emax_mask
        istdi, istdi_mask
    """
    n, k = len(ids), len(C.ISOFORMS)
    pos = {m: i for i, m in enumerate(ids)}
    out = {
        "pic50_lo": np.full((n, k), -LARGE, np.float32),
        "pic50_hi": np.full((n, k), LARGE, np.float32),
        "pic50_mask": np.zeros((n, k), np.float32),
        "log2fc": np.zeros((n, k), np.float32),
        "log2fc_mask": np.zeros((n, k), np.float32),
        "tdiarm": np.zeros((n, k), np.float32),
        "tdiarm_mask": np.zeros((n, k), np.float32),
        "emax": np.zeros((n, k), np.float32),
        "emax_mask": np.zeros((n, k), np.float32),
        "istdi": np.zeros((n, k), np.float32),
        "istdi_mask": np.zeros((n, k), np.float32),
    }

    iso_idx = {c: j for j, c in enumerate(C.ISOFORMS)}

    for r in label_matrix.itertuples(index=False):
        i = pos.get(getattr(r, C.ID_COL))
        if i is None:
            continue
        j = iso_idx[r.isoform]
        lo, hi = float(r.y_low), float(r.y_high)
        out["pic50_lo"][i, j] = -LARGE if not np.isfinite(lo) else lo
        out["pic50_hi"][i, j] = LARGE if not np.isfinite(hi) else hi
        out["pic50_mask"][i, j] = 1.0

    if screen is not None:
        for r in screen.itertuples(index=False):
            i = pos.get(getattr(r, C.ID_COL))
            if i is None:
                continue
            j = iso_idx.get(r.enzyme)
            if j is None or not np.isfinite(r.log2fc_estimate):
                continue
            out["log2fc"][i, j] = float(r.log2fc_estimate)
            out["log2fc_mask"][i, j] = 1.0

    def _fill(frame, col_tmpl, key):
        if frame is None:
            return
        for cyp in C.ISOFORMS:
            col = col_tmpl.format(cyp=cyp)
            if col not in frame.columns:
                continue
            j = iso_idx[cyp]
            sub = frame[[C.ID_COL, col]].dropna()
            for mid, val in zip(sub[C.ID_COL], sub[col]):
                i = pos.get(mid)
                if i is None:
                    continue
                out[key][i, j] = float(val)
                out[key + "_mask"][i, j] = 1.0

    _fill(tdi_frame, "{cyp}_pIC50_TDI_condition", "tdiarm")
    _fill(emax, "{cyp}_EmaxVsPosCtrl_direct_inhibition", "emax")

    # is_TDI: prefer the Emax file, which carries all four isoforms.
    for frame in (emax, tdi_frame):
        if frame is None:
            continue
        for cyp in C.ISOFORMS:
            col = f"{cyp}_is_TDI"
            if col not in frame.columns:
                continue
            j = iso_idx[cyp]
            sub = frame[[C.ID_COL, col]].dropna()
            for mid, val in zip(sub[C.ID_COL], sub[col]):
                i = pos.get(mid)
                if i is None or out["istdi_mask"][i, j] == 1.0:
                    continue
                out["istdi"][i, j] = float(bool(val))
                out["istdi_mask"][i, j] = 1.0

    return out


def target_summary(t: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for head in ("pic50", "log2fc", "tdiarm", "emax", "istdi"):
        m = t[head + "_mask"]
        rows.append(
            dict(
                head=head,
                labelled_rows=int(m.sum()),
                density=float(m.mean()),
                **{c: int(m[:, j].sum()) for j, c in enumerate(C.ISOFORMS)},
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def make_model(n_features: int, hidden=(512, 256), dropout: float = 0.25):
    torch, nn = _torch()
    k = len(C.ISOFORMS)

    class MTL(nn.Module):
        def __init__(self):
            super().__init__()
            layers, d = [], n_features
            for h in hidden:
                layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            self.trunk = nn.Sequential(*layers)
            self.pic50 = nn.Linear(d, k)
            self.log2fc = nn.Linear(d, k)
            self.tdiarm = nn.Linear(d, k)
            self.emax = nn.Linear(d, k)
            self.istdi = nn.Linear(d, k)

        def forward(self, x):
            z = self.trunk(x)
            return {
                "pic50": self.pic50(z),
                "log2fc": self.log2fc(z),
                "tdiarm": self.tdiarm(z),
                "emax": self.emax(z),
                "istdi": self.istdi(z),
            }

    return MTL()


def dead_zone_loss_torch(pred, lo, hi, mask, delta: float = 0.25):
    """Masked Huberised dead-zone loss — the torch twin of losses.dead_zone_loss."""
    torch, _ = _torch()
    above = torch.clamp(pred - hi, min=0.0)
    below = torch.clamp(lo - pred, min=0.0)
    r = above + below
    quad = 0.5 * r.pow(2) / delta
    lin = r - 0.5 * delta
    per = torch.where(r <= delta, quad, lin)
    denom = mask.sum().clamp(min=1.0)
    return (per * mask).sum() / denom


def masked_l1(pred, target, mask):
    torch, _ = _torch()
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target).abs() * mask).sum() / denom


def masked_bce(logit, target, mask, pos_weight: float = 1.0):
    torch, nn = _torch()
    per = nn.functional.binary_cross_entropy_with_logits(
        logit, target, reduction="none",
        pos_weight=torch.tensor(pos_weight, dtype=logit.dtype),
    )
    denom = mask.sum().clamp(min=1.0)
    return (per * mask).sum() / denom
