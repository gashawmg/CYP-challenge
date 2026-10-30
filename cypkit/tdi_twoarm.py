#!/usr/bin/env python
"""TDI as the DIFFERENCE of two dose-response arms, both of which we can model.

`is_TDI` thresholds the shift between two arms of one assay, and the challenge gives us
both for all four isoforms — not just the two that are scored:

    <iso>_pIC50_TDI_condition        potency after preincubation
    <iso>_pIC50_direct_inhibition    potency without

WHY THIS NEEDS A SHARED TRUNK, AND WHY IT IS OTHERWISE HOPELESS. The two arms are nearly
the same measurement:

    isoform   n both arms   sd(TDI arm)   sd(direct)   r(arms)   sd(shift)   shift/arm
    CYP1A2       1412          1.054        1.030       0.986      0.174       0.17
    CYP2C9       1285          0.801        0.782       0.962      0.218       0.28
    CYP2D6       1493          0.776        0.916       0.896      0.409       0.45
    CYP3A4       2334          1.132        1.093       0.947      0.364       0.33

Predict each arm independently with error sigma and the difference carries sigma*sqrt(2),
against a shift sd of only 0.17-0.41 — hopeless. But with a shared encoder the two heads'
errors are correlated and cancel:

    CYP3A4, to reach shift rho 0.6, the per-arm error may be
        0.206   if the arms' errors are uncorrelated
        0.460   at error correlation 0.80
        0.651   at error correlation 0.90     <- our pIC50 models sit at 0.50-0.70 MAE

So the whole idea stands or falls on one measurable quantity: how correlated the two
heads' errors are. `evaluate()` reports it directly rather than assuming it.

THE HEADROOM. The true shift separates the classes at AUC 0.9865 on CYP3A4 (d' 3.13). Our
current d' of 1.276 corresponds to capturing only rho ~0.41 of the shift. Reaching 0.49
implies d' 1.538, which is what rank 1 scores. A 20% relative gain on a quantity we have
never modelled with the good machinery.

WHY REOPENING THE MULTI-HEAD LINE IS NOT BACKSLIDING. `STRATEGY.md` Section 10 closed multi-head
as a REPLACEMENT for v6 on pIC50, where the heads were loosely related assays and sharing
bought nothing (rho 0.6196 vs 0.6181, a tie). Here the two heads are the two arms of ONE
assay on the same molecule, and the shared-trunk error cancellation is not a side effect —
it is the entire mechanism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C, io

ARMS = ("TDI_condition", "direct_inhibition")

# THE BETTER PARAMETERISATION, found by measuring the first one. Predicting both arms and
# subtracting carries a difference-error of sigma*sqrt(2(1-ec)); with our measured sigma
# 0.773 and ec 0.877 that is 0.383 against a shift sd of 0.364 -- larger than the signal,
# so shift rho collapsed to zero. Making the SHIFT ITSELF a head removes the differencing
# entirely: the shift head is trained on the shift, and the direct arms stay as auxiliary
# heads that shape the shared trunk. Same multi-task benefit, none of the amplification.
MODE_SHIFT_HEAD = "shift_head"
SCORED = ("CYP3A4", "CYP2D6")          # the only two with an is_TDI label


def head_names() -> list[str]:
    return [f"{i}_{a}" for i in C.ISOFORMS for a in ARMS]


def build(data_dir=None, shift_head: bool = False, verbose: bool = True) -> dict:
    """One row per compound; Y is (n, 8) with NaN where an arm was not measured.

    shift_head=True replaces each `<iso>_TDI_condition` column with `<iso>_shift`
    (TDI arm minus direct arm), so the quantity we actually need is a trained head
    rather than a difference of two.
    """
    d = io.load("train_tdi", data_dir)
    d.columns = [c.strip() for c in d.columns]
    names = head_names()
    Y = np.full((len(d), len(names)), np.nan, np.float32)
    for j, n in enumerate(names):
        iso, arm = n.rsplit("_", 2)[0], n[len(n.rsplit("_", 2)[0]) + 1:]
        col = f"{iso}_pIC50_{arm}"
        Y[:, j] = pd.to_numeric(d[col], errors="coerce").to_numpy(np.float32)

    if shift_head:
        for iso in C.ISOFORMS:
            t = names.index(f"{iso}_TDI_condition")
            dd = names.index(f"{iso}_direct_inhibition")
            Y[:, t] = Y[:, t] - Y[:, dd]
        names = [n.replace("_TDI_condition", "_shift") for n in names]

    lab = {i: d[f"{i}_is_TDI"] for i in SCORED}
    keep = np.isfinite(Y).any(axis=1)
    out = dict(smiles=d.loc[keep, C.SMILES_COL].tolist(),
               ids=d.loc[keep, C.ID_COL].tolist(),
               Y=Y[keep], names=names,
               labels={i: lab[i][keep].to_numpy() for i in SCORED})

    if verbose:
        print(f"two-arm table: {int(keep.sum())} compounds x {len(names)} heads")
        print(f"  {'head':28s} {'labels':>8s} {'mean':>8s} {'sd':>7s}")
        for j, n in enumerate(names):
            col = out["Y"][:, j]; ok = np.isfinite(col)
            tag = "  <- scored isoform" if n.split("_")[0] in SCORED else ""
            print(f"  {n:28s} {int(ok.sum()):8d} {col[ok].mean():8.3f} "
                  f"{col[ok].std():7.3f}{tag}")
        for i in SCORED:
            v = out["labels"][i]
            print(f"  is_TDI {i}: {int(pd.notna(v).sum())} labelled, "
                  f"{int((v == True).sum())} positive")            # noqa: E712
    return out


def shift_index(names: list[str], iso: str) -> tuple[int, int]:
    """(TDI-or-shift column, direct column). With shift_head the first IS the prediction."""
    first = (f"{iso}_shift" if f"{iso}_shift" in names else f"{iso}_TDI_condition")
    return names.index(first), names.index(f"{iso}_direct_inhibition")
