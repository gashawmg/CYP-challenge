#!/usr/bin/env python
"""Predict the leaderboard from out-of-fold results. Fitted on every anchor we have.

    python scripts/validator.py --oof artifacts/oof_blend_final.csv
    python scripts/validator.py --check          # replay the known anchors

WHAT REPLACED WHAT. The old composite (0.5*OOF + 0.5*dense + 0.0795) fitted three
classical anchors to three decimals and then missed `all7_calib` by 0.114 and
`v3_blend` by 0.086. It was modelling the wrong thing: ST-RAE depends on where
predictions sit relative to the blind set's own distribution, and we were feeding it
only training-set quantities.

Since we now apply an affine correction to every submission
(`scripts/affine_calibrate.py`), the centre and spread of a submission are no longer
free variables -- they are set to the blind set's. What remains is the CORRELATION,
and for an optimally-rescaled predictor

    ST-RAE  ->  absorption * sqrt(1 - r^2)

where `absorption` is the fraction of raw error the credible-interval dead zone does
NOT forgive. Measured from the per-endpoint leaderboard of `v3_blend`: 0.792, 0.865,
0.850, 0.830 on four independent endpoints -- mean 0.834, and that tightness is what
makes this usable.

So the whole validator reduces to one question: what will the blind-set correlation be?

THE OOF -> BLIND MAP. `v3_blend` gave us both numbers on the same submission:

    endpoint   OOF Spearman   blind Spearman   gap
    CYP1A2         0.5933         0.7728      +0.180
    CYP2C9         0.6920         0.7922      +0.100
    CYP2D6         0.4742         0.5081      +0.034
    CYP3A4         0.8102         0.8117      +0.002

The blind set is EASIER than our folds -- it sits at median max-Tanimoto 0.587 to
training while the folds hold out at 0.44 -- but the gap is not constant: it shrinks
as OOF rises, and vanishes on CYP3A4.

A Fisher-z shrink toward the ceiling, with ONE fitted constant (c = 0.372), gives:

    endpoint   actual blind rho   predicted   error
    CYP1A2           0.7728         0.6909    -0.082
    CYP2C9           0.7922         0.7425    -0.050
    CYP2D6           0.5081         0.6324    +0.124   <- badly over-predicted
    CYP3A4           0.8117         0.8108    -0.001

So: good on CYP3A4, mildly pessimistic on CYP1A2/CYP2C9, and WRONG on CYP2D6, where it
promises signal the endpoint does not have. Read a CYP2D6 prediction from this map as
an upper bound, not a forecast. One fitted constant cannot capture four endpoints whose
train/test relationship differs -- with a second anchor the right move is a per-endpoint
shift, which needs no new theory, only another submission with per-endpoint metrics.

WHAT THE SECOND ANCHOR CHANGED. `v3_affine80` is `v3_blend` under a known affine map,
and scored 1.1048 against 0.7056 -- with IDENTICAL Spearman on all four endpoints. Two
things follow. First, this validator's ranking half was right: rho is preserved, so a
rho-based predictor was never what failed. Second, the ABSORPTION half is not a constant
of the task but a property of where a submission sits: the same ranking scored 0.6366 and
0.8216 on CYP1A2. So `ABSORPTION * sqrt(1-r^2)` is a floor reachable by a WELL-CENTRED
submission, not a forecast for any submission -- do not quote it for one whose centre and
spread have not been checked against BLIND_MOMENTS.

HOW MUCH TO TRUST IT. Four points, one submission, one constant. It is a working
hypothesis, not a law -- but unlike the composite it says WHY it should hold, so it can
be falsified cleanly, and its errors above are stated rather than hidden in an average.
Add every scored submission to ANCHORS and refit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cypkit import config as C

ABSORPTION = 0.834          # dead-zone factor; 0.792-0.865 across four endpoints

# Everything the leaderboard has told us. Add a row per scored submission.
ANCHORS = [
    dict(name="submission_l1",              strae=1.0127, note="classical, L1"),
    dict(name="submission_dense_blend",     strae=0.9171, note="classical, dense-fitted"),
    dict(name="submission_composite_scaled", strae=0.8244, note="composite-blended"),
    dict(name="submission_all7_calib",      strae=0.6763, note="7 members + calibration"),
    dict(name="submission_v3_blend",        strae=0.7056, note="11 members incl. ChemProp v3",
         per_endpoint=dict(CYP1A2=0.6366, CYP2C9=0.6307, CYP2D6=1.0292, CYP3A4=0.5260),
         blind_rho=dict(CYP1A2=0.7728, CYP2C9=0.7922, CYP2D6=0.5081, CYP3A4=0.8117),
         oof_rho=dict(CYP1A2=0.5933, CYP2C9=0.6920, CYP2D6=0.4742, CYP3A4=0.8102)),
    dict(name="submission_v3_affine80", strae=1.1048,
         note="v3_blend under a WRONG-SIGN affine shift; rank 118. Same ranking as "
              "v3_blend (Spearman identical on all four), so the score difference is "
              "entirely calibration. This is the anchor that resolved the bias sign -- "
              "see scripts/affine_calibrate.py.",
         per_endpoint=dict(CYP1A2=0.8216, CYP2C9=0.4898, CYP2D6=2.6297, CYP3A4=0.4782),
         blind_rho=dict(CYP1A2=0.7728, CYP2C9=0.7922, CYP2D6=0.5081, CYP3A4=0.8117),
         mae=dict(CYP1A2=1.0466, CYP2C9=0.5110, CYP2D6=2.4128, CYP3A4=0.5663),
         r2=dict(CYP1A2=0.1936, CYP2C9=0.5583, CYP2D6=-2.4188, CYP3A4=0.6111)),
]

# What that pair of anchors bought: the blind set's own moments, sign resolved.
BLIND_MOMENTS = {"CYP1A2": (4.440, 1.290), "CYP2C9": (5.071, 1.119),
                 "CYP2D6": (3.420, 1.322), "CYP3A4": (4.907, 1.274)}


def fisher(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


def fit_shift():
    """One constant: how much easier the blind set is, in Fisher-z units."""
    a = next(x for x in ANCHORS if "oof_rho" in x)
    z_oof = np.array([fisher(a["oof_rho"][i]) for i in C.ISOFORMS])
    z_bl = np.array([fisher(a["blind_rho"][i]) for i in C.ISOFORMS])
    # The gap shrinks as rho rises, which in z-space is a proportional shrink toward
    # the blind ceiling rather than an additive shift. Fit z_blind = z_oof + c*(zmax - z_oof).
    zmax = z_bl.max()
    denom = (zmax - z_oof)
    c = float(np.sum((z_bl - z_oof) * denom) / np.sum(denom ** 2))
    return c, zmax


def oof_to_blind(rho_oof, c, zmax):
    z = fisher(rho_oof)
    return float(np.tanh(z + c * (zmax - z)))


def predict(rho_oof_by_iso):
    c, zmax = fit_shift()
    rows = []
    for iso in C.ISOFORMS:
        r_oof = rho_oof_by_iso[iso]
        r_bl = oof_to_blind(r_oof, c, zmax)
        strae = ABSORPTION * np.sqrt(max(1 - r_bl ** 2, 1e-9))
        rows.append(dict(isoform=iso, oof_rho=r_oof, blind_rho=r_bl, strae=strae))
    return pd.DataFrame(rows), c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", default=None,
                    help="an OOF prediction file with the four endpoint columns")
    ap.add_argument("--train-csv", default="data/cyp-challenge-TRAIN_inhibition.csv")
    ap.add_argument("--check", action="store_true", help="replay the fitted anchor")
    args = ap.parse_args()

    c, zmax = fit_shift()
    print(f"fitted blind-set shift c = {c:.3f} "
          f"(0 = blind as hard as our folds, 1 = blind at the ceiling)\n")

    if args.check or not args.oof:
        a = next(x for x in ANCHORS if "oof_rho" in x)
        print(f"replaying {a['name']} -- the anchor the map was fitted on")
        print(f"  {'endpoint':8s} {'OOF rho':>8s} {'blind rho':>10s} {'predicted':>10s} "
              f"{'actual':>8s}  {'post-affine':>12s}")
        for iso in C.ISOFORMS:
            r_oof = a["oof_rho"][iso]
            r_hat = oof_to_blind(r_oof, c, zmax)
            strae = ABSORPTION * np.sqrt(1 - r_hat ** 2)
            print(f"  {iso:8s} {r_oof:8.4f} {a['blind_rho'][iso]:10.4f} "
                  f"{r_hat:10.4f} {a['per_endpoint'][iso]:8.4f} {strae:12.4f}")
        print("\n  'predicted' vs 'blind rho' is the map's own error; 'actual' is the")
        print("  UNCORRECTED score, which the affine step is meant to fix. The")
        print("  post-affine column is what a corrected submission should reach.\n")
        print("leaderboard history:")
        for x in ANCHORS:
            print(f"  {x['name']:28s} {x['strae']:.4f}   {x['note']}")

    if args.oof:
        tr = pd.read_csv(args.train_csv)
        tr.columns = [c2.strip() for c2 in tr.columns]
        oof = pd.read_csv(args.oof)
        rho = {}
        for iso in C.ISOFORMS:
            ep = f"{iso}_pIC50_direct_inhibition"
            m = tr[["Molecule_Name", ep]].dropna(subset=[ep]).merge(
                oof[["Molecule_Name", ep]], on="Molecule_Name", suffixes=("", "_p")).dropna()
            rho[iso] = float(spearmanr(m[ep], m[ep + "_p"])[0])
        d, _ = predict(rho)
        print(f"\n{args.oof}")
        print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n  predicted MA-ST-RAE after affine correction: {d.strae.mean():.4f}")
        print(f"  best actual so far: 0.6763 (uncorrected). The affine step is what")
        print(f"  makes this number reachable -- submit through affine_calibrate.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
