#!/usr/bin/env python
"""Solve per-endpoint training weights so each endpoint's label mass matches the blind set.

    python scripts/fit_endpoint_weights.py
    python scripts/fit_endpoint_weights.py --write cypkit/endpoint_weights.json

WHY PER ENDPOINT. One global setting has been wrong for at least one endpoint the whole
time, and in OPPOSITE directions:

    endpoint   blind mass below pIC50 4.0   what v3 defaults give it
    CYP1A2               42%                        52%   too much
    CYP2C9               26%                        45%   far too much
    CYP2D6               73%                        50%   FAR TOO LITTLE
    CYP3A4               31%                        35%   about right

CYP2D6 is the endpoint we cannot predict, it is the one that needs the most
low-activity signal, and it is the one getting the least. The cause is two settings
compounding:

  * only 8.6% of CYP2D6 curves sit below 4.0 (against 40% for CYP3A4), and
  * `interval_power=1.0` multiplies those rows by ~0.1, because a weak compound's
    credible interval is wide (median half-width 1.44 in the 2.5-4.0 band) and the
    interval weighting was designed to charge narrow intervals more.

That weighting is correct WITHIN the training distribution -- ST-RAE really does forgive
errors inside a wide interval. Under a shift toward the low end it is exactly backwards:
it suppresses the rows that carry the part of the distribution the blind set is made of.

WHAT THIS SOLVES. For each endpoint, the (interval_power, censored_weight) pair whose
resulting weight-below-4.0 is closest to what the blind distribution needs. Grid search,
not gradient -- there are two knobs and the objective is a step function of a histogram.

Ties are broken toward interval_power=1.0, because within-distribution the interval
weighting is doing something real and should only be given up where it has to be.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import norm

from cypkit import config as C, io, labels as L
from cypkit.ensemble import interval_weights

BLIND = {"CYP1A2": (4.301, 1.528), "CYP2C9": (4.849, 1.324),
         "CYP2D6": (3.130, 1.456), "CYP3A4": (4.703, 1.442)}
SPLIT = C.PIC50_FLOOR


def share_below(lm, iso, power, split=SPLIT):
    s = lm[lm["isoform"] == iso].copy()
    w = s["weight"].to_numpy(float) * interval_weights(s, power=power)
    y = np.where(s["kind"].to_numpy() == "left",
                 np.where(np.isfinite(s["y_high"]), s["y_high"], split),
                 s["y"].to_numpy(float))
    ok = np.isfinite(y) & np.isfinite(w)
    tot = w[ok].sum()
    return float(w[ok][y[ok] <= split].sum() / tot) if tot > 0 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--powers", type=float, nargs="*",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--weights", type=float, nargs="*",
                    default=[0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0])
    ap.add_argument("--pseudo", action="store_true",
                    help="solve for the --pseudo --pseudo_nonhit_interval label matrix")
    ap.add_argument("--write", default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    tr = io.load("train_inhibition", args.data_dir)
    sc = io.load("single_conc", args.data_dir)

    cache = {}
    for cw in args.weights:
        cache[cw] = L.build_label_matrix(
            tr, sc, include_censored=True, censored_weight=cw,
            pseudo=args.pseudo, pseudo_nonhit_interval=args.pseudo,
            pseudo_weight=cw)

    print(f"{'endpoint':8s} {'blind needs':>12s} {'v3 default':>11s} | "
          f"{'power':>6s} {'cens_w':>7s} {'achieved':>9s} {'error':>7s}")
    out = {}
    for iso in C.ISOFORMS:
        need = norm.cdf(SPLIT, *BLIND[iso])
        base = share_below(cache[0.5], iso, 1.0)
        best = None
        for p in args.powers:
            for cw in args.weights:
                got = share_below(cache[cw], iso, p)
                if not np.isfinite(got):
                    continue
                # tie-break toward power=1.0, which is the within-distribution optimum
                key = (abs(got - need), abs(p - 1.0))
                if best is None or key < best[0]:
                    best = (key, p, cw, got)
        _, p, cw, got = best
        out[iso] = dict(interval_power=p, censored_weight=cw,
                        target_share=round(need, 4), achieved=round(got, 4))
        print(f"{iso:8s} {need:11.1%} {base:10.1%} | {p:6.2f} {cw:7.2f} "
              f"{got:8.1%} {got - need:+7.1%}")

    print("\nrun each endpoint with its own settings (same --model_dir is fine):")
    for iso, v in out.items():
        print(f"  --isoforms {iso} --interval_power {v['interval_power']} "
              f"--censored_weight {v['censored_weight']}")

    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text(json.dumps(out, indent=2))
        print(f"\nwritten {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
