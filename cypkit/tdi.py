"""The TDI track.

`is_TDI` is not a noisy biological label — it is a deterministic function of two
measured quantities. Recovered from the released data at **100% agreement** on both
scored isoforms (3,584 CYP3A4 rows, 1,497 CYP2D6 rows, zero exceptions):

    is_TDI  =  direct-arm curve was fitted
           AND (pIC50_TDI - pIC50_direct)  >  log10(2)
           AND  pIC50_TDI                  >  4.301

Note what this is *not*. The challenge blog describes "inferred positives" — compounds
whose direct arm falls below the floor but whose TDI arm exceeds 4.301. In the released
labels those compounds are **all False**: 1,249 CYP3A4 compounds have a TDI-arm pIC50
(median 5.40, 84.5% above 4.301) and no direct-arm fit, and every one is labelled
False. Applying the blog's rule as written agrees with the released labels on only 58.8%
of CYP3A4 rows.

So the track reduces to two regressions plus a rule. That matters because the
regressions have far more to learn from: the continuous shift carries magnitude
information the boolean discards, and both arms are reported for all four isoforms even
though only two are scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

LOG2 = float(np.log10(2))
TDI_ARM_FLOOR = 4.301


def apply_rule(direct, tdi_arm, measurable=None) -> np.ndarray:
    """The exact published rule.

    Args:
        direct: predicted or measured direct-arm pIC50. NaN means no fitted curve.
        tdi_arm: predicted or measured pre-incubation pIC50.
        measurable: optional explicit "direct arm was fitted" flag. When omitted it is
            taken from `direct` being non-NaN, which is right for measured data and
            wrong for predictions — a model always emits a number.
    """
    direct = np.asarray(direct, dtype=float)
    tdi_arm = np.asarray(tdi_arm, dtype=float)
    fitted = ~np.isnan(direct) if measurable is None else np.asarray(measurable, dtype=bool)
    shift = tdi_arm - np.nan_to_num(direct, nan=0.0)
    return fitted & (shift > LOG2) & (tdi_arm > TDI_ARM_FLOOR)


def verify_rule(tdi_frame: pd.DataFrame, isoforms=C.TDI_ISOFORMS) -> pd.DataFrame:
    """Check the rule still reproduces the released labels. Run this before trusting it."""
    rows = []
    for cyp in isoforms:
        lab = tdi_frame[f"{cyp}_is_TDI"]
        has = lab.notna()
        pred = apply_rule(
            tdi_frame.loc[has, f"{cyp}_pIC50_direct_inhibition"],
            tdi_frame.loc[has, f"{cyp}_pIC50_TDI_condition"],
        )
        y = lab[has].astype(bool).to_numpy()
        rows.append(
            dict(isoform=cyp, n=int(has.sum()), agreement=float((pred == y).mean()),
                 positive_rate=float(y.mean()))
        )
    return pd.DataFrame(rows)


def assemble(tdi_frame: pd.DataFrame) -> pd.DataFrame:
    """Long-format TDI training table, one row per (compound, scored isoform)."""
    rows = []
    for cyp in C.TDI_ISOFORMS:
        lab = tdi_frame[f"{cyp}_is_TDI"]
        sub = tdi_frame[lab.notna()]
        rows.append(
            pd.DataFrame(
                {
                    C.ID_COL: sub[C.ID_COL].to_numpy(),
                    C.SMILES_COL: sub[C.SMILES_COL].to_numpy(),
                    "isoform": cyp,
                    "y": sub[f"{cyp}_is_TDI"].astype(bool).to_numpy(),
                    "direct": sub[f"{cyp}_pIC50_direct_inhibition"].to_numpy(),
                    "tdi_arm": sub[f"{cyp}_pIC50_TDI_condition"].to_numpy(),
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    out["measurable"] = out["direct"].notna()
    out["shift"] = out["tdi_arm"] - out["direct"]
    return out


# ---------------------------------------------------------------------------
# Turning three model outputs into a probability
# ---------------------------------------------------------------------------


def structured_score(
    p_measurable, pred_direct, pred_tdi, scale: float = 0.20
) -> np.ndarray:
    """Soft version of the rule, so a threshold can be tuned for MCC.

    Each hard condition becomes a logistic in how far the prediction clears it. `scale`
    is the softness in log units — small enough to stay close to the rule, large enough
    that a compound sitting on the boundary does not flip on regression noise.
    """
    p_measurable = np.asarray(p_measurable, dtype=float)
    shift = np.asarray(pred_tdi, dtype=float) - np.asarray(pred_direct, dtype=float)
    tdi = np.asarray(pred_tdi, dtype=float)
    return (
        p_measurable
        * _sigmoid((shift - LOG2) / scale)
        * _sigmoid((tdi - TDI_ARM_FLOOR) / scale)
    )


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def best_mcc_threshold(y_true, score, grid=None) -> tuple[float, float]:
    """Threshold maximising MCC, and the MCC it achieves.

    The tutorial thresholds at 0.5. MCC is not threshold-invariant and the classes are
    ~21% positive, so 0.5 is close to arbitrary — this is usually worth more than any
    architecture change in the track.
    """
    from sklearn.metrics import matthews_corrcoef  # noqa: PLC0415

    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=float)
    if grid is None:
        grid = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 199)))
    best_t, best_m = 0.5, -1.0
    for t in grid:
        m = matthews_corrcoef(y_true, score >= t)
        if m > best_m:
            best_t, best_m = float(t), float(m)
    return best_t, best_m
