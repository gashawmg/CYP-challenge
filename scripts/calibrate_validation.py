#!/usr/bin/env python
"""Calibrate local validation against the actual leaderboard.

    python scripts/calibrate_validation.py --score artifacts/submission_*.csv

Phase 1 left us with two local scorers that disagreed, and the leaderboard showed both
were wrong in different directions:

    OOF ST-RAE   0.7518  ->  leaderboard 1.0127   (26% optimistic)
    dense proxy  1.1157  ->  leaderboard 1.0127   (10% pessimistic)

Neither is a calibrated predictor. This builds one, by inverting the problem: we know our
submitted predictions, and the leaderboard told us four metrics they achieved. Fit a
latent test-truth model that reproduces those metrics, then score any new candidate
against it.

    y_test = (mean_p + a) + b * (p - mean_p) + N(0, s)

Fitted on MAE, R^2 and Spearman only. ST-RAE is deliberately held out as a check — if the
fitted model also reproduces the reported ST-RAE, it has captured something real rather
than just interpolating three numbers.

Credible-interval widths come from the empirical band-vs-potency relationship measured on
the training curves, so the simulated ST-RAE uses realistic dead zones.

LIMITATION, and it matters: the simulated truth is a linear function of ONE submission's
predictions. It is well suited to judging corrections to those predictions (a shift, a
rescale, a censoring change that moves the whole distribution) and less suited to judging
a genuinely different model, which it will tend to under-credit wherever that model
departs from the reference in ways the linear fit cannot express. Where it disagrees with
the dense proxy, treat the disagreement as the finding.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit import io  # noqa: E402

# What the leaderboard reported for the reference submission.
REFERENCE = dict(
    file="artifacts/submission_l1.csv",
    strae=1.0127, mae=1.0798, r2=-0.0949, spearman=0.5729,
)


def band_model(train: pd.DataFrame):
    """Median credible-interval width as a function of true potency."""
    rows = []
    for ep in C.REGRESSION_ENDPOINTS:
        d = train[[ep, ep + "_conf_low", ep + "_conf_high"]].dropna()
        d.columns = ["y", "lo", "hi"]
        rows.append(d)
    B = pd.concat(rows)
    B["w"] = B.hi - B.lo
    xs = [3.0, 3.75, 4.25, 4.75, 5.25, 6.0]
    cuts = [(None, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0), (5.0, 5.5), (5.5, None)]
    ws = []
    for lo, hi in cuts:
        m = np.ones(len(B), bool)
        if lo is not None:
            m &= B.y > lo
        if hi is not None:
            m &= B.y <= hi
        ws.append(float(B.loc[m, "w"].median()))
    return xs, ws


def make_scorer(xs, ws):
    def bands(y):
        w = np.interp(y, xs, ws)
        return y - w / 2, y + w / 2

    def strae(Y, P):
        lo, hi = bands(Y)
        err = np.clip(P - hi, 0, None) + np.clip(lo - P, 0, None)
        mu = Y.mean(0)
        base = np.clip(mu - hi, 0, None) + np.clip(lo - mu, 0, None)
        return float(np.mean([err[:, j].sum() / base[:, j].sum() for j in range(Y.shape[1])]))

    return strae


def fit(P0, strae, target, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for b in np.arange(0.6, 2.61, 0.1):
        for s in np.arange(0.4, 1.81, 0.05):
            for a in np.arange(-1.4, 0.81, 0.1):
                Y = (P0.mean(0) + a) + b * (P0 - P0.mean(0)) + rng.normal(0, s, P0.shape)
                d = Y - P0
                mae = np.abs(d).mean()
                r2 = 1 - (d**2).sum() / ((Y - Y.mean(0)) ** 2).sum()
                sp = np.mean([spearmanr(Y[:, j], P0[:, j]).correlation for j in range(Y.shape[1])])
                e = (abs(mae - target["mae"]) / target["mae"]
                     + abs(r2 - target["r2"]) / 0.5
                     + abs(sp - target["spearman"]) / target["spearman"])
                if best is None or e < best[0]:
                    best = (e, a, b, s, dict(mae=mae, r2=r2, spearman=sp, strae=strae(Y, P0)))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", nargs="*", default=["artifacts/submission_*.csv"])
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    train = io.load("train_inhibition")
    xs, ws = band_model(train)
    strae = make_scorer(xs, ws)

    ref = pd.read_csv(REFERENCE["file"])
    P0 = np.column_stack([ref[ep].to_numpy() for ep in C.REGRESSION_ENDPOINTS])

    err, a, b, s, got = fit(P0, strae, REFERENCE)
    print(f"latent test truth:  y = (mean_p {a:+.2f}) + {b:.2f}*(p - mean_p) + N(0, {s:.2f})")
    print(f"  implied test mean {P0.mean()+a:.2f} sd ~{np.hypot(b*P0.std(), s):.2f}"
          f"   (our reference predictions: mean {P0.mean():.2f} sd {P0.std():.2f})")
    print(f"\n{'metric':10s} {'reported':>10s} {'simulated':>10s}")
    for k in ("mae", "r2", "spearman", "strae"):
        flag = "   <- HELD OUT of the fit" if k == "strae" else ""
        print(f"{k:10s} {REFERENCE[k]:10.4f} {got[k]:10.4f}{flag}")

    rng = np.random.default_rng(args.seed)
    truths = [(P0.mean(0) + a) + b * (P0 - P0.mean(0)) + rng.normal(0, s, P0.shape)
              for _ in range(args.draws)]

    files = sorted({f for pat in args.score for f in glob.glob(pat)})
    print(f"\n{'candidate':42s} {'pred LB ST-RAE':>16s}")
    out = []
    for f in files:
        d = pd.read_csv(f)
        if not all(ep in d.columns for ep in C.REGRESSION_ENDPOINTS):
            continue
        P = np.column_stack([d[ep].to_numpy() for ep in C.REGRESSION_ENDPOINTS])
        v = [strae(T, P) for T in truths]
        note = "  <- reference, actual %.4f" % REFERENCE["strae"] if f == REFERENCE["file"] else ""
        print(f"{Path(f).name:42s} {np.mean(v):10.4f} +/-{np.std(v):.4f}{note}")
        out.append(dict(file=Path(f).name, pred_lb_strae=float(np.mean(v)), sd=float(np.std(v))))
    pd.DataFrame(out).to_csv(C.artifact("predicted_leaderboard.csv"), index=False)
    print("\n  -> artifacts/predicted_leaderboard.csv")
    print("\n  Re-fit this after every new leaderboard result; two reference points would")
    print("  let the fit drop its linearity assumption, which is its main weakness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
