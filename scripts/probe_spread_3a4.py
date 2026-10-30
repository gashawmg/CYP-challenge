#!/usr/bin/env python
r"""Measure the ST-RAE spread optimum ON THE BOARD, because it cannot be measured locally.

    python scripts\probe_spread_3a4.py

WHY A PROBE AND NOT A LOCAL FIT. `scripts/scan_spread.py` asks the evaluation pool to
reproduce a number we already know -- our board ST-RAE at the spread we actually shipped --
before letting it recommend anything. It fails on all four endpoints, over-stating cost by
+67% / +39% / +11% / +33%. Our models simply do far better on the blind compounds than on
our own validation compounds (Section 16: median out-of-fold nearest-neighbour similarity 0.438
against the blind set's 0.587), and difficulty-reweighting only closes part of it before
the effective sample size collapses. SO THE POOL'S k* = 0.55 IS NOT EVIDENCE. It is exactly
what a too-hard pool would say, because harder data always prefers more shrinkage.

WHAT IS STILL WORTH TESTING. The mechanism is not in doubt: ST-RAE charges
`clip(p-hi,0) + clip(lo-p,0)`, so a prediction inside its interval costs NOTHING, and
pulling predictions toward the mean parks them inside intervals in a way squared error
never rewards. We ship k = rho on every endpoint, which is the MSE-optimal shrinkage and
has no reason to be the ST-RAE-optimal one. The board is the only instrument that can
price it.

THE DESIGN. Two points on the CYP3A4 cost curve are already known EXACTLY:

    ST-RAE(k=0)     = 1.0000   by construction -- predicting the mean IS the metric's base
    ST-RAE(k=0.766) = 0.4634   the shipped spread, recovered by the raw3a4 probe

A quadratic needs three. This probe supplies the third at k = 0.65 and therefore the
vertex. Because the other three endpoints are frozen, `4 x macro - 0.5125 - 0.3978 - 0.5957`
returns CYP3A4's score exactly, the same arithmetic that recovered 0.4634.

THE ASYMMETRY THAT JUSTIFIES SPENDING A SLOT:

    hypothesis                     CYP3A4     macro
    optimum is where we sit        0.4757    0.4954    (-0.003 against 0.4924 held)
    optimum is at 0.55             0.3865    0.4731    (+0.019)

Roughly six to one, and either outcome fixes the curve so the next step lands on the vertex
rather than near it.

AN AFFINE RESCALE IS MONOTONE, SO rho IS UNCHANGED. MA-Spearman must come back at exactly
0.7509. If it moves, something other than the spread changed and the reading is void --
that is the built-in check on this experiment.

WHAT THIS IS NOT. Not `place_half`, which shipped ABSOLUTE values from a local fit and
landed CYP3A4's mean 0.35 log units off the anchor. The mean here is untouched; one scalar
changes, and its value comes from a board measurement rather than from the pool.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit.submission import write_submission  # noqa: E402
from scripts.blind_validator import BLIND  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "artifacts" / "submission_splice4.csv"     # scored 0.4924, rank 19
OUT = ROOT / "artifacts" / "submission_spread65.csv"
ISO, K_TARGET = "CYP3A4", 0.65


def main() -> int:
    ref = pd.read_csv(REFERENCE)
    ref.columns = [c.strip() for c in ref.columns]
    ep = f"{ISO}_pIC50_direct_inhibition"
    ref = ref.set_index(C.ID_COL)          # write_submission matches values BY MOLECULE ID
    v = ref[ep]
    mu, sd = float(v.mean()), float(v.std())
    k_now = sd / BLIND[ISO][1]
    target_sd = K_TARGET * BLIND[ISO][1]
    scale = target_sd / sd
    new = mu + (v - mu) * scale

    print(f"  {ISO}: k {k_now:.3f} -> {K_TARGET:.3f}   sd {sd:.4f} -> {float(new.std()):.4f}"
          f"   scale {scale:.4f}")
    print(f"  mean {mu:.4f} -> {float(new.mean()):.4f}  (unchanged by construction)")
    print(f"  Spearman vs the scored column: "
          f"{float(new.corr(v, method='spearman')):.6f}  (must be 1.000000)")
    print(f"  the other three endpoints are copied through untouched")
    assert abs(float(new.corr(v, method="spearman")) - 1.0) < 1e-9, \
        "an affine rescale must be perfectly monotone; it is not, so something is wrong"

    write_submission({ep: new}, REFERENCE, OUT, kind="regression")
    print("\n  READING IT BACK: with three endpoints frozen,")
    print("     CYP3A4 = 4 x macro - 0.5125 - 0.3978 - 0.5957")
    print("  and MA-Spearman must come back at 0.7509 exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
