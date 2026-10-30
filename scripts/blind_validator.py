#!/usr/bin/env python
"""Score an OOF file on a validation set RESHAPED to look like the blind test set.

    python scripts/blind_validator.py --oof oof_chemprop_st_v3_ps.csv
    python scripts/blind_validator.py --compare oof_a.csv oof_b.csv ...

WHY THIS EXISTS. Between `v3_blend` and `pseudo_blend45` we improved out-of-fold macro
Spearman by +0.052 and the blind set moved +0.004. On CYP2D6 it went DOWN. Three
submissions in a row have been chosen on local numbers that did not transfer, so the
local number is the thing to fix before any more GPU time is spent.

THE DIAGNOSIS. Our folds and the blind set are different populations.

    endpoint   our eval pool (curves)   blind (solved from 3 anchors)
    CYP1A2         4.96 / 1.03               4.32 / 1.43
    CYP2C9         4.58 / 0.78               4.81 / 1.12
    CYP2D6         4.78 / 0.92               2.92 / 1.65
    CYP3A4         4.10 / 1.09               4.76 / 1.25

The blind set is ~1.5x wider on every endpoint, and on CYP2D6 it sits 1.65 log units
lower. Dose-response curves were only run on primary-screen HITS, so the curve pool is
a biased sample of the library the blind set was drawn from. Ranking improvements
measured on that pool are measured on the wrong chemistry.

WHAT THIS DOES. Histogram matching, then resampling.

  1. The evaluation pool is the FULL label matrix -- fitted curves plus screen non-hits,
     which are the only rows we have that live below the assay floor and therefore the
     only way to represent the part of the blind distribution our curves never cover.
  2. Rows are binned by their label; censored rows form the "below floor" bin.
  3. Each bin is given the mass the blind normal N(mean, sd) puts there.
  4. Rows are RESAMPLED with replacement to those masses, B times, and the metrics are
     computed on each draw.

Resampling rather than weighting because Spearman and ST-RAE have no clean weighted
form, and because the spread across draws is itself the useful output: it says whether
a difference between two models is larger than the noise of the evaluation.

WHAT IT DOES NOT DO. It cannot conjure chemistry we never measured. If the blind set's
low-activity region contains scaffolds absent from our screen, matching the LABEL
distribution still leaves that gap. Treat this as necessary, not sufficient -- but the
current metric is not even necessary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from cypkit import config as C, io, labels as L

# Solved from three scored submissions; see scripts/affine_calibrate.py.
BLIND = {"CYP1A2": (4.316, 1.431), "CYP2C9": (4.806, 1.122),
         "CYP2D6": (2.922, 1.652), "CYP3A4": (4.759, 1.252)}
FLOOR = C.PIC50_FLOOR


def eval_pool(data_dir=None):
    """One row per (compound, isoform): label, interval, and whether it is censored."""
    tr = io.load("train_inhibition", data_dir)
    sc = io.load("single_conc", data_dir)
    # PIN THE YARDSTICK. `build_label_matrix` defaults to keep_subfloor=("CYP3A4",) --
    # a TRAINING choice, made because 129 sub-floor rows are too few to teach CYP2D6 the
    # region. The EVALUATION pool must not inherit it: the blind set genuinely contains
    # sub-floor compounds with real fitted values, so censoring them here makes the
    # yardstick easier and unrepresentative, and worse, it makes scores incomparable
    # between runs whenever the training choice changes. Observed directly: the same
    # oof_st_v8.csv scored macro rho 0.5964 against the ungated pool and 0.5788 against
    # the gated one. Always evaluate against every measurement we have.
    lm = L.build_label_matrix(tr, sc, include_censored=True,
                              keep_subfloor=tuple(C.ISOFORMS))
    lm = lm.copy()
    lm["censored"] = lm["kind"].to_numpy() == "left"
    # a censored row's label is unknown; it is represented by its bound
    lm["y_eval"] = np.where(lm["censored"],
                            np.where(np.isfinite(lm["y_high"]), lm["y_high"], FLOOR),
                            lm["y"])
    return lm.dropna(subset=["y_eval"])


def match_weights(y, censored, m, s, n_bins=12):
    """Per-row resampling weights that make the pool's label histogram match N(m, s)."""
    from scipy.stats import norm  # noqa: PLC0415

    edges = np.linspace(m - 3 * s, m + 3 * s, n_bins + 1)
    # bin 0 is "below the floor": the censored rows, plus the blind mass under FLOOR
    target, pool = [], []
    idx_sets = []

    below = censored | (y <= FLOOR + 1e-9)
    idx_sets.append(np.where(below)[0])
    target.append(norm.cdf(FLOOR, m, s))

    hi_edges = [e for e in edges if e > FLOOR] + [np.inf]
    lo = FLOOR
    for hi in hi_edges:
        sel = (~below) & (y > lo) & (y <= hi)
        idx_sets.append(np.where(sel)[0])
        target.append(norm.cdf(hi, m, s) - norm.cdf(lo, m, s))
        lo = hi

    target = np.array(target, float)
    w = np.zeros(len(y))
    for t, idx in zip(target, idx_sets):
        if len(idx) and t > 0:
            w[idx] = t / len(idx)
        pool.append(len(idx))
    if w.sum() <= 0:
        return None, None
    return w / w.sum(), (target, np.array(pool))


def strae(p, lo, hi, y):
    err = (np.where(np.isfinite(hi), np.clip(p - hi, 0, None), 0.0)
           + np.where(np.isfinite(lo), np.clip(lo - p, 0, None), 0.0))
    m = float(np.mean(y))
    base = (np.where(np.isfinite(hi), np.clip(m - hi, 0, None), 0.0)
            + np.where(np.isfinite(lo), np.clip(lo - m, 0, None), 0.0))
    d = float(np.sum(base))
    return float(np.sum(err) / d) if d > 0 else np.nan


def score(pred_by_iso, pool, n_draws=200, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    rows = []
    for iso in C.ISOFORMS:
        ep = f"{iso}_pIC50_direct_inhibition"
        s = pool[pool["isoform"] == iso]
        p = s[C.ID_COL].map(pred_by_iso.get(ep, {})).to_numpy(float)
        ok = np.isfinite(p)
        s, p = s[ok], p[ok]
        if len(s) < 100:
            continue
        y = s["y_eval"].to_numpy(float)
        cen = s["censored"].to_numpy(bool)
        lo = s["y_low"].to_numpy(float)
        hi = s["y_high"].to_numpy(float)
        m, sd = BLIND[iso]
        w, diag = match_weights(y, cen, m, sd)
        if w is None:
            continue
        n = len(y)
        rh, mae, st = [], [], []
        for _ in range(n_draws):
            d = rng.choice(n, size=n, replace=True, p=w)
            yd, pd_ = y[d], p[d]
            fit = ~cen[d]
            if fit.sum() > 30:
                rh.append(pd.Series(yd[fit]).rank().corr(pd.Series(pd_[fit]).rank()))
                mae.append(float(np.mean(np.abs(yd[fit] - pd_[fit]))))
            st.append(strae(pd_, lo[d], hi[d], yd))
        rows.append(dict(isoform=iso, n=n,
                         rho=np.mean(rh), rho_sd=np.std(rh),
                         mae=np.mean(mae), strae=np.mean(st), strae_sd=np.std(st),
                         pred_sd=float(np.std(p)), need_sd=sd))
    return pd.DataFrame(rows)


def load_pred(path):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    return {ep: dict(zip(d[C.ID_COL], d[ep]))
            for ep in C.REGRESSION_ENDPOINTS if ep in d.columns}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", default=None)
    ap.add_argument("--compare", nargs="*", default=None,
                    help="score several OOF files side by side")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    pool = eval_pool(args.data_dir)
    print(f"evaluation pool: {len(pool)} (compound, isoform) rows, "
          f"{int(pool['censored'].sum())} censored\n")

    files = args.compare or ([args.oof] if args.oof else [])
    if not files:
        raise SystemExit("give --oof FILE or --compare FILE [FILE ...]")

    allr = {}
    for f in files:
        if not Path(f).exists():
            print(f"  {f}: not found, skipped"); continue
        allr[f] = score(load_pred(f), pool, args.draws)

    print("BLIND-MATCHED validation  (+- is the spread across resampling draws:")
    print("a difference smaller than that is not a difference)\n")
    for f, r in allr.items():
        print(Path(f).name)
        print(f"  {'endpoint':8s} {'rho':>16s} {'MAE':>7s} {'ST-RAE':>16s} "
              f"{'pred sd':>8s} {'needed':>7s}")
        for _, x in r.iterrows():
            print(f"  {x.isoform:8s} {x.rho:8.4f}+-{x.rho_sd:.4f} {x.mae:7.4f} "
                  f"{x.strae:8.4f}+-{x.strae_sd:.4f} {x.pred_sd:8.2f} {x.need_sd:7.2f}")
        print(f"  {'MACRO':8s} {r.rho.mean():8.4f}{'':7s} {r.mae.mean():7.4f} "
              f"{r.strae.mean():8.4f}\n")

    if len(allr) > 1:
        print("ranked by blind-matched ST-RAE (lower is better):")
        for f, r in sorted(allr.items(), key=lambda kv: kv[1].strae.mean()):
            print(f"  {r.strae.mean():.4f}   {Path(f).name}")
    print("\n'pred sd' vs 'needed' is the compression this project keeps paying for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
