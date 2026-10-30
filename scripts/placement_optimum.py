#!/usr/bin/env python
"""How much is LEFT in placement, per endpoint, given the ranking we already have.

    python scripts/placement_optimum.py
    python scripts/placement_optimum.py --rho CYP2C9=0.83 CYP2D6=0.49

WHY. `submission_placed` moved MA-ST-RAE 0.6763 -> 0.5837 with the model untouched, so
placement is the lever that has actually worked. But it was placed from the THIRD
calibration table, and the fourth refit says two endpoints were over-expanded and one
under-expanded. This asks what the same ranking scores at the CORRECT placement, so the
next submission is chosen against a number rather than a hope.

THE MODEL. Take truth Y ~ N(m, s) on the blind set and a prediction P = a + b*Z, with Z
the standardised model score and corr(Z, Y) = rho. Then Y - P is normal with mean (m - a)
and sd sqrt(s^2 + b^2 - 2*b*rho*s), which gives E|Y - P| in closed form. ST-RAE is
soft-thresholded rather than plain absolute error, but it is monotone in this quantity, so
one fitted constant per endpoint turns predicted MAE into predicted ST-RAE.

The constant is fitted on `submission_placed`, whose per-endpoint scores are published.
If the model reproduces those four numbers, its optimum is quotable -- the same standard
the binormal TDI threshold model had to meet before its cut rates were acted on. If it
does not, the table below is a story and should be ignored.

IT DID NOT MEET THAT STANDARD. READ THIS BEFORE ACTING ON STEP 2.

`v3_blend` and `v3_affine80` are the same ranking at two different placements, and both
are scored per endpoint. Fitting rho to `v3_blend` with k from `submission_placed`, then
PREDICTING `v3_affine80`, is an honest out-of-sample test with one degree of freedom per
endpoint. The result:

    endpoint   fitted rho   affine80 predicted   actual   error
    CYP1A2        0.887           0.7636        0.8216   -0.0580
    CYP2C9        0.709           0.6015        0.4898   +0.1117
    CYP2D6        no root -- cannot reproduce 1.0292 at ANY rho
    CYP3A4        0.884           0.5229        0.4782   +0.0447

Mean absolute error 0.0715. STEP 2 below claims optimal re-placement is worth 0.0234 of
macro ST-RAE. The gain is THREE TIMES SMALLER THAN THE MODEL'S OWN ERROR, so acting on it
is not a calibrated bet, it is the affine80 mistake with better arithmetic. CYP2D6 having
no root at all says the BLIND table's (2.922, 1.652) and this soft-threshold constant are
not jointly consistent with an earlier scored submission on that endpoint.

WHAT SURVIVES. STEP 3 is a RATIO within one endpoint -- k and the blind moments largely
cancel -- so the exchange rate between rho and ST-RAE is far better determined than the
absolute levels. That is the part worth quoting: +0.05 of rho is worth about -0.06 of
macro ST-RAE, and +0.10 is worth about -0.14. Both are well outside the 0.07 error above,
which is precisely why a +0.011 blend edge is not a submission and a new representation is.

WHAT IT SAYS (subject to the above). The optimum is a = m and b = rho * s -- centre on the
blind mean, and expand to rho times the blind sd, NOT to the blind sd. Over-expanding past
rho*s costs, which is what happened to CYP2C9 and CYP3A4.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import norm

from cypkit import config as C

# Fourth refit; see scripts/affine_calibrate.py.
BLIND = {"CYP1A2": (4.316, 1.431), "CYP2C9": (4.806, 1.122),
         "CYP2D6": (2.922, 1.652), "CYP3A4": (4.759, 1.252)}
RHO = {"CYP1A2": 0.770, "CYP2C9": 0.817, "CYP2D6": 0.724, "CYP3A4": 0.811}
# What submission_placed actually put on the leaderboard, and what it scored.
PLACED = {"CYP1A2": (4.30, 1.21, 0.5495), "CYP2C9": (4.85, 1.15, 0.5055),
          "CYP2D6": (3.13, 0.91, 0.7034), "CYP3A4": (4.70, 1.24, 0.5763)}


def expected_abs(mu, sigma):
    """E|X| for X ~ N(mu, sigma)."""
    if sigma <= 0:
        return abs(mu)
    return (sigma * np.sqrt(2 / np.pi) * np.exp(-mu * mu / (2 * sigma * sigma))
            + mu * (1 - 2 * norm.cdf(-mu / sigma)))


def mae(a, b, m, s, rho):
    """E|Y - P| for Y ~ N(m, s), P = a + b*Z, corr(Z, Y) = rho, Z standard normal."""
    sd = np.sqrt(max(s * s + b * b - 2 * b * rho * s, 1e-12))
    return expected_abs(m - a, sd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho", nargs="*", default=[],
                    help="override, e.g. CYP2C9=0.83 (use a refit correlation)")
    args = ap.parse_args()

    rho = dict(RHO)
    for kv in args.rho:
        k, v = kv.split("="); rho[k] = float(v)

    print("STEP 1 -- does the model reproduce submission_placed?\n")
    print(f"  {'endpoint':8s} {'pred MAE':>9s} {'actual ST-RAE':>14s} {'k = ST/MAE':>11s}")
    k_fit, total = {}, []
    for iso in C.ISOFORMS:
        m, s = BLIND[iso]
        a, b, actual = PLACED[iso]
        pm = mae(a, b, m, s, rho[iso])
        k_fit[iso] = actual / pm
        print(f"  {iso:8s} {pm:9.4f} {actual:14.4f} {k_fit[iso]:11.4f}")
    ks = np.array(list(k_fit.values()))
    print(f"\n  k spread {ks.min():.4f}-{ks.max():.4f} (cv {ks.std()/ks.mean():.3f}). "
          f"A tight spread means one\n  soft-threshold constant explains all four "
          f"endpoints, and the optimum below is real.")

    print("\n\nSTEP 2 -- the optimum at the SAME ranking (a = m, b = rho*s)\n")
    print(f"  {'endpoint':8s} {'placed a/b':>13s} {'optimal a/b':>14s} "
          f"{'ST-RAE now':>11s} {'at optimum':>11s} {'gain':>8s}")
    now, opt = [], []
    for iso in C.ISOFORMS:
        m, s = BLIND[iso]
        a, b, actual = PLACED[iso]
        ao, bo = m, rho[iso] * s
        so = k_fit[iso] * mae(ao, bo, m, s, rho[iso])
        now.append(actual); opt.append(so)
        print(f"  {iso:8s} {a:6.2f}/{b:5.2f} {ao:7.2f}/{bo:5.2f} "
              f"{actual:11.4f} {so:11.4f} {so-actual:+8.4f}")
    print(f"\n  MACRO   {np.mean(now):32.4f} {np.mean(opt):11.4f} "
          f"{np.mean(opt)-np.mean(now):+8.4f}")

    print("\n\nSTEP 3 -- what a BETTER RANKING is worth, at optimal placement\n")
    print("  ST-RAE at the optimum, as rho moves. This is the exchange rate between the\n"
          "  model track and the placement track, and it is why +0.01 of rho is not a\n"
          "  submission.\n")
    print(f"  {'endpoint':8s} " + " ".join(f"{d:>+8.2f}" for d in
                                           (-0.05, 0.0, 0.05, 0.10, 0.20)))
    for iso in C.ISOFORMS:
        m, s = BLIND[iso]
        row = []
        for d in (-0.05, 0.0, 0.05, 0.10, 0.20):
            r = min(max(rho[iso] + d, 0.01), 0.99)
            row.append(k_fit[iso] * mae(m, r * s, m, s, r))
        print(f"  {iso:8s} " + " ".join(f"{v:8.4f}" for v in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
