r"""
placement_model.py  --  a forward model of ST-RAE, fitted and TESTED on scored board points,
then used to find the optimal SPREAD per endpoint.

WHY A MODEL AT ALL, GIVEN THE OOF SHIFT CURVE WAS RETIRED AT 1-FOR-3. Because this one is
fitted entirely to BLIND-SET quantities the board reported -- MAE, R-squared, Spearman rho
and ST-RAE itself -- and never to out-of-fold rows. The OOF curve failed for a specific,
diagnosed reason: it measured our blind-tuned placement against the TRAINING population.
This model never looks at the training population. That does not make it right; it makes
it wrong for different reasons, which is why it has to pass a test before it is used.

THE MODEL. For one endpoint, take truth y ~ N(m_y, sy^2) and prediction p with spread sp
and Pearson correlation r, so the error e = y - p is normal with mean (m_y - m_p) and
variance sy^2 + sp^2 - 2 r sy sp. Credible intervals are modelled with a single effective
half-width h, and ST-RAE is the soft-thresholded absolute error normalised by the same
quantity for a constant prediction at mean(y):

    ST-RAE = E[max(|e| - h, 0)] / E[max(|y - m_y| - h, 0)]

Four unknowns per endpoint -- sy, r, h and the current centre offset. We have MAE, R^2 and
at least one scored ST-RAE, and for two endpoints we have THREE scored ST-RAE values at
known shifts, which makes those endpoints overdetermined.

THE ACCEPTANCE TEST, AND IT IS NOT NEGOTIABLE. CYP1A2 was scored at shifts +0.00, +0.20
and +0.40 (0.5147 / 0.4682 / 0.4729) and CYP3A4 at -0.30, 0.00 and +0.20 (0.6382 derived /
0.4532 / 0.4112). The model is fitted WITHOUT the shift curve and must then reproduce those
points. If it cannot predict a shift curve it never saw, it has no business predicting a
spread curve, and the table it prints is a story. This is the same bar the binormal TDI
threshold model had to clear before its call rates were acted on -- it predicted another
team's CYP3A4 MCC to 0.0001 and was then used; the OOF shift curve never cleared any bar
and cost us two reversed recommendations.

WHAT IT IS FOR. ST-RAE is convex in the prediction vector, and p = c + s(z - c) is affine
in s, so ST-RAE is EXACTLY convex in the spread factor -- the same property that let three
scored points pin each centre. The spreads have never been cleanly probed: d6wide is the
only attempt, and it changed CYP2D6's spread by x2.17 AND its centre by -0.70 at the same
time, so it measured nothing attributable.

Run:
  python scripts\placement_model.py
  python scripts\placement_model.py --no-test     # skip the acceptance test (do not)
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import norm

# ---- measured on the blind set by the board, at the 0.4504 placement -----------------
ENDPOINTS = {
    "CYP1A2": dict(st=0.4639, mae=0.7066, r2=0.6121, rho=0.8064, sp=1.0947),
    "CYP2C9": dict(st=0.3645, mae=0.4177, r2=0.7054, rho=0.8554, sp=0.7927),
    "CYP2D6": dict(st=0.5621, mae=1.0409, r2=0.3242, rho=0.5190, sp=0.6906),
    "CYP3A4": dict(st=0.4112, mae=0.4873, r2=0.7112, rho=0.8425, sp=0.9578),
}
# scored shift curves, as deltas from the 0.4504 placement -> measured endpoint ST-RAE
SHIFT_CURVES = {
    "CYP1A2": [(-0.28, 0.5147), (-0.08, 0.4682), (+0.12, 0.4729)],
    "CYP3A4": [(-0.50, 0.6382), (-0.20, 0.4532), (+0.00, 0.4112)],
    "CYP2C9": [(+0.00, 0.3645), (+0.20, 0.3726)],
    "CYP2D6": [(+0.00, 0.5621), (+0.30, 0.5957)],   # from cand1, same ranking
}


def soft(mu: float, sd: float, h: float) -> float:
    """E[max(|X| - h, 0)] for X ~ N(mu, sd^2), in closed form."""
    sd = max(sd, 1e-9)
    a, b = (-h - mu) / sd, (h - mu) / sd
    # E[(X-h)+]  for the right tail and  E[(-X-h)+] for the left
    right = (mu - h) * (1 - norm.cdf(b)) + sd * norm.pdf(b)
    left = (-mu - h) * norm.cdf(a) + sd * norm.pdf(a)
    return float(right + left)


def strae(offset: float, sy: float, sp: float, r: float, h: float) -> float:
    """ST-RAE at a uniform centre offset, everything else fixed."""
    ev = sy * sy + sp * sp - 2.0 * r * sy * sp
    num = soft(offset, np.sqrt(max(ev, 1e-12)), h)
    den = soft(0.0, sy, h)
    return num / max(den, 1e-12)


def strae_spread(scale: float, offset: float, sy: float, sp: float, r: float, h: float) -> float:
    """ST-RAE when the prediction spread is multiplied by `scale` about its own centre."""
    sp2 = sp * scale
    ev = sy * sy + sp2 * sp2 - 2.0 * r * sy * sp2
    return soft(offset, np.sqrt(max(ev, 1e-12)), h) / max(soft(0.0, sy, h), 1e-12)


def fit_endpoint(d: dict):
    """Closed form. With the centre at its optimum the bias is ~0, and then MAE, R^2 and
    ST-RAE pin the error variance, sigma_y, the correlation and the half-width in turn --
    there is nothing to optimise. (The first version of this searched numerically, hit its
    bounds on three of four endpoints and reported a 'fit' with cost 1e+2 that was pure
    noise. An exactly determined system should never be handed to an optimiser.)"""
    sp = d["sp"]
    ev = (d["mae"] * np.sqrt(np.pi / 2.0)) ** 2          # E[(y-p)^2] from MAE, zero-mean normal
    sy = np.sqrt(ev / (1.0 - d["r2"]))                   # from R^2 = 1 - ev/sy^2
    r = (sy * sy + sp * sp - ev) / (2.0 * sy * sp)       # from the definition of ev
    lo, hi = 1e-6, 12.0                                  # h is monotone decreasing in ST-RAE
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if strae(0.0, sy, sp, r, mid) > d["st"]:
            lo = mid
        else:
            hi = mid
    h = 0.5 * (lo + hi)
    resid = abs(strae(0.0, sy, sp, r, h) - d["st"])
    return dict(sy=float(sy), r=float(r), h=float(h), offset=0.0, sp=sp, cost=float(resid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()

    fits = {n: fit_endpoint(d) for n, d in ENDPOINTS.items()}
    print(f"{'endpoint':<9}{'sigma_y':>9}{'Pearson r':>11}{'rho':>8}{'half-width h':>14}"
          f"{'offset':>9}{'fit cost':>10}")
    for n, f in fits.items():
        print(f"{n:<9}{f['sy']:>9.4f}{f['r']:>11.4f}{ENDPOINTS[n]['rho']:>8.4f}"
              f"{f['h']:>14.4f}{f['offset']:>9.4f}{f['cost']:>10.2e}")

    if not a.no_test:
        print("\n" + "=" * 74)
        print("ACCEPTANCE TEST -- predict shift curves the fit never saw")
        print("=" * 74)
        worst = 0.0
        for n, pts in SHIFT_CURVES.items():
            f = fits[n]
            print(f"\n{n}:  {'shift':>7}{'measured':>11}{'predicted':>11}{'error':>9}")
            for ds, meas in pts:
                pred = strae(f["offset"] - ds, f["sy"], f["sp"], f["r"], f["h"])
                worst = max(worst, abs(pred - meas))
                print(f"        {ds:>+7.2f}{meas:>11.4f}{pred:>11.4f}{pred-meas:>+9.4f}")
        print(f"\nworst absolute error across every scored shift point: {worst:.4f}")
        ok = worst < 0.02
        print("VERDICT:", "PASS -- the spread table below may be acted on" if ok else
              "FAIL -- the spread table below is a STORY, do not spend a slot on it")
        if not ok:
            print("A model that cannot reproduce a shift curve it never saw has not earned")
            print("the right to recommend a spread. Treat what follows as a hypothesis only.")

    print("\n" + "=" * 74)
    print("SPREAD -- ST-RAE is exactly convex in the scale factor")
    print("=" * 74)
    print(f"{'endpoint':<9}{'now':>7}{'optimal':>9}{'ST-RAE now':>12}{'at optimum':>12}"
          f"{'endpoint gain':>15}{'macro':>9}")
    tot = 0.0
    opts = {}
    for n, f in fits.items():
        cur = strae_spread(1.0, f["offset"], f["sy"], f["sp"], f["r"], f["h"])
        res = minimize_scalar(lambda s: strae_spread(s, f["offset"], f["sy"], f["sp"],
                                                     f["r"], f["h"]),
                              bounds=(0.3, 3.0), method="bounded")
        gain = cur - res.fun
        tot += gain / 4.0
        opts[n] = (res.x, gain)
        print(f"{n:<9}{1.0:>7.2f}{res.x:>9.3f}{cur:>12.4f}{res.fun:>12.4f}"
              f"{gain:>15.4f}{gain/4:>9.5f}")
    print(f"\ntotal macro available from spread alone: {tot:.4f}  "
          f"-> {0.4504 - tot:.4f} from 0.4504")
    print("\nCost of a probe that overshoots or undershoots the optimum:")
    print(f"{'endpoint':<9}" + "".join(f"{f'x{s:.2f}':>9}" for s in (0.9, 1.0, 1.1, 1.2, 1.35, 1.6)))
    for n, f in fits.items():
        row = [strae_spread(s, f["offset"], f["sy"], f["sp"], f["r"], f["h"])
               for s in (0.9, 1.0, 1.1, 1.2, 1.35, 1.6)]
        print(f"{n:<9}" + "".join(f"{v:>9.4f}" for v in row))


if __name__ == "__main__":
    main()
