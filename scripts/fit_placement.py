#!/usr/bin/env python
r"""Fit the placement transform that ST-RAE actually rewards, and build submissions.

    python scripts\fit_placement.py

THE QUESTION. Our absorption -- ST-RAE divided by the sqrt(1-rho^2) that the ranking
forces -- has sat at 0.806 across twelve submissions while rivals reach 0.717 with
comparable rankings:

    CYP2C9   blend4z  rho 0.8380  ST-RAE 0.4172   absorption 0.765
             wbot     rho 0.8517         0.3655              0.697
    CYP3A4   blend4z  rho 0.8310         0.4636              0.833
             wbot     rho 0.8561         0.3816              0.738

Since ST-RAE charges `clip(p - hi, 0) + clip(lo - p, 0)`, a prediction anywhere inside
its interval costs zero, so the optimal map is not obviously affine -- which is what we
have used every time. This script asks how much a free-form monotone map can win.

THE ANSWER IS 3%, NOT 13%, AND THE DIFFERENCE IS AN ARTEFACT WE ALMOST SHIPPED.
Fitted against the full evaluation pool the transform claimed 13% less absorption and
parked 83% of CYP2D6 predictions at the assay floor. That gain was not real. Our pool
represents the below-floor region two ways -- real sub-floor curve fits with finite
intervals, and screen non-hits that are LEFT-CENSORED and therefore free to undershoot --
and once resampled to the blind histogram the below-floor mass is 89-96% censored on
CYP1A2/2C9/2D6. The blind set has no censored rows at all: every one of its 750 compounds
carries a real fitted value. The transform was buying a discount that does not exist on
the board.

Refit on uncensored rows only -- an interval structure that matches the blind set:

    endpoint   affine abs   fitted abs   gain
    CYP1A2       0.723        0.742      -2%    <- WORSE, keeps the affine map
    CYP2C9       0.725        0.709       2%
    CYP2D6       0.951        0.861       9%    <- on 129 sub-floor rows, thin
    CYP3A4       0.831        0.811       2%
    MACRO        0.807        0.781       3%

THE CORRECTED POOL IS WORTH MORE THAN THE TRANSFORM. Its affine absorption is 0.807
against the 0.806 we actually score on the blind set. That is now a calibrated yardstick
for placement, and it says something the raw pool hid: the best monotone re-placement of
our ranking reaches 0.781, nowhere near the 0.717 rivals achieve. Their advantage is
therefore NOT a cleverer placement of a similar ranking. ST-RAE weights each compound by
how far outside its interval you land, so a model can share our overall rho and still win
by ranking better among the NARROW-interval compounds -- the expensive ones. No monotone
map can fix that; it is a training-weight problem, not a placement problem.

RHO IS UNTOUCHED. The transform is monotone in the ranking, so Spearman is identical by
construction: a wrong transform costs placement and cannot cost us the ranking we have.

TWO CANDIDATES, and per-endpoint selection -- CYP1A2 keeps its affine map because the
fitted one lost on held-out compounds:

    submission_place_half.csv   halfway between affine and fitted, in value space
    submission_place_full.csv   the fitted transform

Expected macro, applying the fitted/affine ratio per endpoint to our measured blind
absorption: about 0.490 against blend4z's 0.5104. Real, free, and small -- not the 0.459
the artefact promised.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit.submission import write_submission  # noqa: E402
from scripts.blind_validator import BLIND, eval_pool, match_weights, strae  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OOF = ROOT / "oof_bz_all4_plain.csv"          # the blend's out-of-fold ranking
PRED = {"v6": "pred_st_v6.csv", "mh": "pred_st_mh.csv",
        "cp": "predictions_cyp_st_v3.csv", "um": "predictions_unimol_cyp_mt_test.csv"}
REFERENCE = ROOT / "artifacts" / "submission_splice_best.csv"
KNOTS = 6


def monotone(theta: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Piecewise-linear and increasing in the rank u, by construction."""
    knots = np.concatenate([[theta[0]], theta[0] + np.cumsum(np.exp(theta[1:]))])
    return np.interp(u, np.linspace(0, 1, len(knots)), knots)


def ranks(x: np.ndarray) -> np.ndarray:
    return (pd.Series(x).rank().to_numpy() - 0.5) / len(x)


def fit_endpoint(iso: str, pool: pd.DataFrame, oof: pd.DataFrame,
                 rng: np.random.Generator, draws: int = 30) -> dict:
    # UNCENSORED ROWS ONLY, and this is the whole correctness of this script.
    # Our eval pool represents the below-floor region two ways: real sub-floor curve fits
    # (finite intervals, what the floor fix recovered) and screen non-hits, which are
    # LEFT-CENSORED, so any prediction below the bound costs zero. Resampled to the blind
    # histogram, the below-floor mass is 89-96% censored on CYP1A2/2C9/2D6. The blind set
    # is 0% censored -- all 750 compounds carry a real fitted value. Fitting against the
    # censored rows therefore buys a discount that does not exist on the board: the first
    # version of this fit parked 83% of CYP2D6 at the floor and claimed 13% less
    # absorption; on the corrected pool that gain is 3%, and CYP1A2 turns NEGATIVE.
    # The corrected pool is also the one that matches reality -- its affine absorption is
    # 0.807 against the 0.806 we actually score on the blind set.
    s = pool[(pool["isoform"] == iso) & (~pool["censored"])]
    s = s[s[C.ID_COL].isin(oof.index)]
    ids = s[C.ID_COL].to_numpy()
    y = s["y_eval"].to_numpy(float)
    cen = np.zeros(len(y), bool)
    lo = s["y_low"].to_numpy(float)
    hi = s["y_high"].to_numpy(float)
    w, _ = match_weights(y, cen, *BLIND[iso])
    u = ranks(oof.loc[ids, f"{iso}_pIC50_direct_inhibition"].to_numpy(float))
    rho = float(pd.Series(u).corr(pd.Series(y), method="spearman"))

    # Split COMPOUNDS, and rebuild the matching weights inside each half so the target
    # histogram still holds -- otherwise the held-out score is measured on a pool that no
    # longer looks like the blind set.
    half = rng.random(len(ids)) < 0.5

    def draw_set(mask):
        idx = np.where(mask)[0]
        ww, _ = match_weights(y[idx], cen[idx], *BLIND[iso])
        return [idx[rng.choice(len(idx), size=len(idx), replace=True, p=ww)]
                for _ in range(draws)]

    tr, te = draw_set(half), draw_set(~half)
    obj = lambda th, D: float(np.mean(  # noqa: E731
        [strae(monotone(th, u)[k], lo[k], hi[k], y[k]) for k in D]))

    aff = minimize(lambda t: obj(np.array([t[0], np.log(max(t[1], 1e-3))]), tr),
                   [BLIND[iso][0], BLIND[iso][1]], method="Nelder-Mead",
                   options=dict(maxiter=400, xatol=1e-3, fatol=1e-4))
    th_aff = np.array([aff.x[0], np.log(max(aff.x[1], 1e-3))])
    th0 = np.concatenate([[aff.x[0]],
                          np.log(np.full(KNOTS, max(aff.x[1], 1e-3) / KNOTS))])
    fit = minimize(obj, th0, args=(tr,), method="Nelder-Mead",
                   options=dict(maxiter=4000, xatol=1e-3, fatol=1e-5))

    grid = np.linspace(0, 1, 201)
    absorb = lambda v: v / np.sqrt(1 - rho ** 2)  # noqa: E731
    return dict(iso=iso, rho=rho,
                affine_out=obj(th_aff, te), fitted_out=obj(fit.x, te),
                abs_affine=absorb(obj(th_aff, te)), abs_fitted=absorb(obj(fit.x, te)),
                grid=grid, curve_aff=monotone(th_aff, grid),
                curve_fit=monotone(fit.x, grid))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=30)
    a = ap.parse_args()

    oof = pd.read_csv(OOF)
    oof.columns = [c.strip() for c in oof.columns]
    oof = oof.set_index(C.ID_COL)
    pool = eval_pool()
    rng = np.random.default_rng(0)

    fits = {}
    print(f"{'endpoint':9s} {'rho':>7s} | {'affine (held out)':>18s} {'abs':>6s} | "
          f"{'fitted (held out)':>18s} {'abs':>6s}")
    for iso in C.ISOFORMS:
        f = fit_endpoint(iso, pool, oof, rng, a.draws)
        fits[iso] = f
        print(f"{iso:9s} {f['rho']:7.4f} | {f['affine_out']:18.4f} "
              f"{f['abs_affine']:6.3f} | {f['fitted_out']:18.4f} {f['abs_fitted']:6.3f}")
    ma = np.mean([f["abs_affine"] for f in fits.values()])
    mf = np.mean([f["abs_fitted"] for f in fits.values()])
    print(f"{'MACRO':9s} {'':7s} | {'':18s} {ma:6.3f} | {'':18s} {mf:6.3f}"
          f"      -> {100 * (1 - mf / ma):.0f}% less absorption")

    # ---- apply the two transforms to the BLIND ranking ------------------------------
    F = {}
    for k, p in PRED.items():
        d = pd.read_csv(ROOT / p)
        d.columns = [c.strip() for c in d.columns]
        F[k] = d.set_index(C.ID_COL)
    ids = pd.read_csv(REFERENCE)
    ids.columns = [c.strip() for c in ids.columns]
    ids = ids[C.ID_COL]

    half_vals, full_vals = {}, {}
    print(f"\n{'endpoint':9s} {'mean':>7s} {'sd':>7s}   (half)      "
          f"{'mean':>7s} {'sd':>7s}   (full)")
    for iso in C.ISOFORMS:
        ep = f"{iso}_pIC50_direct_inhibition"
        z = [(F[k].loc[ids, ep] - F[k].loc[ids, ep].mean()) / F[k].loc[ids, ep].std()
             for k in F]
        u = ranks((sum(z) / len(z)).to_numpy())
        f = fits[iso]
        # PER-ENDPOINT SELECTION. The fitted transform is used only where it actually
        # beat affine on held-out compounds. On CYP1A2 it does not (absorption 0.740 vs
        # 0.723), so that endpoint keeps the affine map rather than being dragged along
        # by a macro average.
        better = f["abs_fitted"] < f["abs_affine"]
        aff = np.interp(u, f["grid"], f["curve_aff"])
        full = np.interp(u, f["grid"], f["curve_fit"]) if better else aff
        half = (full + aff) / 2
        full_vals[ep] = pd.Series(full, index=ids)
        half_vals[ep] = pd.Series(half, index=ids)
        print(f"{iso:9s} {half.mean():7.3f} {half.std():7.3f}   "
              f"          {full.mean():7.3f} {full.std():7.3f}"
              f"   {'fitted' if fits[iso]['abs_fitted'] < fits[iso]['abs_affine'] else 'AFFINE KEPT'}")

    write_submission(half_vals, REFERENCE,
                     ROOT / "artifacts" / "submission_place_half.csv")
    write_submission(full_vals, REFERENCE,
                     ROOT / "artifacts" / "submission_place_full.csv")
    (ROOT / "artifacts" / "placement_fit.json").write_text(json.dumps(
        {i: dict(rho=f["rho"], abs_affine=f["abs_affine"], abs_fitted=f["abs_fitted"],
                 grid=f["grid"].tolist(), affine=f["curve_aff"].tolist(),
                 fitted=f["curve_fit"].tolist()) for i, f in fits.items()},
        indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
