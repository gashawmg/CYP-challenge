#!/usr/bin/env python
"""The TDI target we should have been training on: the IC50 SHIFT, not the binary.

`is_TDI` is not a measurement. It is a threshold applied to one — the difference between
the two arms of the assay, both of which the challenge gives us:

    <iso>_pIC50_TDI_condition        potency after preincubation
    <iso>_pIC50_direct_inhibition    potency without
    ..._std for each                 the measurement noise on each arm

and the rule the organisers used is recoverable from the training data:

    CYP3A4    shift - 1.96 * pooled_sd    AUC 0.9865   <- a SIGNIFICANCE test
              raw shift                   AUC 0.8546
    CYP2D6    raw shift                   AUC 0.9615   <- a MAGNITUDE threshold
              shift - 1.96 * pooled_sd    AUC 0.8401

WHY THIS MATTERS. `train_tdi_chemprop.py` trains on `is_TDI.astype(int)`. A compound with
a shift of +2.0 and one at +0.31 become the same "1"; one at +0.29 becomes a "0"; and
every compound near the threshold is close to a coin flip that we fit as a hard label.
Our classifier sits at d' 1.276 while the underlying measurement separates the classes at
AUC 0.96-0.99. The binarisation is a plausible cause of that gap, and it is the one thing
in the TDI pipeline we have never varied.

THE COST, STATED UP FRONT. CYP3A4 has 3,584 rows carrying `is_TDI` but only 2,334 where
both arms exist, so regressing the shift trades about 35% of the labelled rows for a far
more informative target. That is a real trade and this module reports it rather than
hiding it; `build()` returns the binary-only rows too, so a two-head model can keep them.

WEIGHTS. Each row is weighted by 1 / pooled_sd, clipped, so a shift measured to +-0.05
counts more than one measured to +-0.6. This is the same lesson the pIC50 side just
learned the hard way: the metric is dominated by the well-measured compounds, so the
training should be too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C, io

TDI_ISOFORMS = ("CYP3A4", "CYP2D6")

# How the organisers appear to define the label, per isoform. Recovered empirically --
# see the AUC table above and `recover_rule()`, which re-derives it from the data so a
# change in the challenge files cannot silently invalidate this.
TARGET_RULE = {"CYP3A4": "significance", "CYP2D6": "magnitude"}

SD_FLOOR = 0.05          # no row counts more than 20x the least precise
SD_CEIL = 2.0


def _num(d: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(d[col], errors="coerce")


def arms(d: pd.DataFrame, iso: str) -> dict[str, pd.Series]:
    return dict(tdi=_num(d, f"{iso}_pIC50_TDI_condition"),
                direct=_num(d, f"{iso}_pIC50_direct_inhibition"),
                sd_tdi=_num(d, f"{iso}_pIC50_TDI_condition_std"),
                sd_direct=_num(d, f"{iso}_pIC50_direct_inhibition_std"))


def target(d: pd.DataFrame, iso: str, rule: str | None = None) -> pd.Series:
    """The continuous quantity `is_TDI` thresholds."""
    a = arms(d, iso)
    shift = a["tdi"] - a["direct"]
    rule = rule or TARGET_RULE[iso]
    if rule == "magnitude":
        return shift
    pooled = np.sqrt(a["sd_tdi"] ** 2 + a["sd_direct"] ** 2)
    return shift - 1.96 * pooled


def recover_rule(d: pd.DataFrame, iso: str) -> pd.DataFrame:
    """Re-derive which formulation matches `is_TDI` best. Run this, do not trust the
    constant above -- if the challenge files change, this is what catches it."""
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    a = arms(d, iso)
    shift = a["tdi"] - a["direct"]
    pooled = np.sqrt(a["sd_tdi"] ** 2 + a["sd_direct"] ** 2)
    y = d[f"{iso}_is_TDI"]
    m = shift.notna() & pooled.notna() & y.notna()
    yy = (y[m] == True).astype(int)                                      # noqa: E712
    cands = {"magnitude (raw shift)": shift[m],
             "significance (shift - 1.96*pooled)": (shift - 1.96 * pooled)[m],
             "z (shift / pooled)": (shift / pooled)[m],
             "shift - pooled": (shift - pooled)[m]}
    rows = [dict(formulation=k, auc=roc_auc_score(yy, v)) for k, v in cands.items()]
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def build(iso: str, data_dir=None, rule: str | None = None,
          verbose: bool = True) -> dict:
    """Rows for the shift regression, plus the binary-only rows a second head could use.

    Returns dict with:
        reg    DataFrame [Molecule_Name, SMILES, y_shift, w, y_bin]  -- both arms present
        clf    DataFrame [Molecule_Name, SMILES, y_bin]              -- label but no shift
    """
    d = io.load("train_tdi", data_dir)
    d.columns = [c.strip() for c in d.columns]
    a = arms(d, iso)
    tgt = target(d, iso, rule)
    pooled = np.sqrt(a["sd_tdi"] ** 2 + a["sd_direct"] ** 2)
    ybin = d[f"{iso}_is_TDI"]

    has_shift = tgt.notna() & pooled.notna()
    has_bin = ybin.notna()

    w = 1.0 / np.clip(pooled, SD_FLOOR, SD_CEIL)
    w = w / w[has_shift].mean()

    reg = pd.DataFrame({
        C.ID_COL: d.loc[has_shift, C.ID_COL],
        C.SMILES_COL: d.loc[has_shift, C.SMILES_COL],
        "y_shift": tgt[has_shift],
        "w": w[has_shift],
        # the direct arm is the baseline the shift is measured against; carrying it lets
        # a model condition on affinity instead of having to infer it
        "direct": a["direct"][has_shift],
        "y_bin": ybin[has_shift],
    }).reset_index(drop=True)

    only_bin = has_bin & ~has_shift
    clf = pd.DataFrame({
        C.ID_COL: d.loc[only_bin, C.ID_COL],
        C.SMILES_COL: d.loc[only_bin, C.SMILES_COL],
        "y_bin": ybin[only_bin],
    }).reset_index(drop=True)

    if verbose:
        print(f"{iso}: rule = {rule or TARGET_RULE[iso]}")
        print(f"  shift regression rows   {len(reg):5d}  "
              f"({int((reg.y_bin == True).sum())} labelled positive)")      # noqa: E712
        print(f"  binary-only rows        {len(clf):5d}   <- LOST if we use the shift "
              f"alone ({100 * len(clf) / max(len(reg) + len(clf), 1):.0f}% of labels)")
        print(f"  target  mean {reg.y_shift.mean():+.3f}  sd {reg.y_shift.std():.3f}")
        print(f"  weight  min {reg.w.min():.2f}  max {reg.w.max():.2f}  "
              f"(1/pooled sd, clipped to [{SD_FLOOR}, {SD_CEIL}])")
    return dict(reg=reg, clf=clf)
