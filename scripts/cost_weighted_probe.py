#!/usr/bin/env python
r"""Does weighting training by what ST-RAE CHARGES beat uniform weighting?

    python scripts\cost_weighted_probe.py
    python scripts\cost_weighted_probe.py --powers 0 0.5 1 1.5 2 --nest 600
    python scripts\cost_weighted_probe.py --iso CYP3A4 --seeds 0 1 2

A CHEAP DECISIVE TEST BEFORE ANY GPU HOURS. LightGBM on ECFP4, the frozen folds, one
variable: the per-row training weight. If cost weighting cannot move this rig it will not
be worth porting into train_cyp_mh.py, and if it can, the power sweep says how hard to
push before a chemprop run is spent on it.

WHY THIS IS THE EXPERIMENT. The leaderboard says our gap is allocation, not accuracy:

    rank 17 hf-far   rho 0.7525   MAE 0.6638   ST-RAE 0.4419
    us (rank 47)     rho 0.7558   MAE 0.6664   ST-RAE 0.4737

Better ranking than them, average error within 0.4% of theirs, and 6.7% worse on the
metric. Neither rho nor MAE can see that difference, so it is WHERE the error falls
relative to each compound's interval. Across ranks 11-21 the ST-RAE-per-MAE ratio runs
0.666-0.705; ours is 0.711, worse than all eleven.

THE WEIGHT, AND WHY IT IS NOT 1/width. Section 69 rejected 1/width because the cost is
U-shaped: the WIDEST quartile carries 34-46% of the bill on three endpoints, because we
miss the extremes so badly that even a wide interval does not save us. That observation was
right and the conclusion (do not weight) was wrong. Two different quantities:

    total cost       where the bill currently falls
    marginal charge  what ONE unit of error costs -- 1 outside the interval, 0 inside

Training should follow the marginal charge:

    w_i = P(|e| > h_i)  =  2 * (1 - Phi(h_i / s))        h = half-width, s = error scale

high for narrow intervals AND for wide-interval compounds we miss badly -- both arms of the
U, which 1/width only half captures. `--weight invwidth` reproduces the old proxy for
comparison; `--weight poutside` (default) is the corrected one.

s IS FITTED ONCE on the UNWEIGHTED arm and reused for every power, so the arms differ only
in the weight exponent. Letting each arm refit s would change the weight AND its scale
together and the sweep would mean nothing.

SCORED THE WAY THE BOARD SCORES. Every arm is re-placed with its own best affine before
comparison (Section 69: comparing raw ST-RAE mostly compares two accidental placements),
and reported on BOTH rulers -- plain rho and cost-weighted rho (Section 77).

READ `delta placed` AS THE RESULT. A power that improves plain rho but not placed ST-RAE
has not helped; a power that improves placed ST-RAE while plain rho falls is exactly the
trade this experiment is looking for, and is the signature we would expect if the
leaderboard diagnosis is right.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, norm
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
ID = "Molecule_Name"


def cost(p, lo, hi):
    return np.clip(p - hi, 0, None) + np.clip(lo - p, 0, None)


def opt_shift_cost(q, lo, hi):
    A = np.sort(hi - q)
    B = np.sort(lo - q)
    cand = np.sort(np.concatenate([A, B]))
    d = np.searchsorted(A, cand, side="left") - (len(B) - np.searchsorted(B, cand, side="right"))
    s = cand[min(int(np.searchsorted(d, 0, side="left")), len(cand) - 1)]
    return np.clip(q + s - hi, 0, None).sum() + np.clip(lo - q - s, 0, None).sum()


def placed(p, lo, hi, y):
    mu = p.mean()
    best = min(opt_shift_cost(b * (p - mu), lo, hi) for b in np.linspace(0.2, 2.2, 41))
    return best / cost(np.full_like(y, y.mean()), lo, hi).sum()


def weighted_spearman(y, p, w):
    ry = pd.Series(y).rank().to_numpy()
    rp = pd.Series(p).rank().to_numpy()
    my = np.average(ry, weights=w)
    mp = np.average(rp, weights=w)
    cov = np.average((ry - my) * (rp - mp), weights=w)
    vy = np.average((ry - my) ** 2, weights=w)
    vp = np.average((rp - mp) ** 2, weights=w)
    return float(cov / np.sqrt(vy * vp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", nargs="*", default=ISO)
    ap.add_argument("--powers", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--weight", choices=["poutside", "invwidth"], default="poutside")
    ap.add_argument("--cap", type=float, default=6.0,
                    help="no row may outweigh the median by more than this, either way")
    ap.add_argument("--nest", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="more than one averages the arms, which halves the noise floor "
                         "this rig has to beat before a power can be called real")
    ap.add_argument("--fp", type=int, default=1024)
    a = ap.parse_args()

    tr = pd.read_csv(ROOT / "data" / "cyp-challenge-TRAIN_inhibition.csv")
    tr.columns = [c.strip() for c in tr.columns]
    tr = tr.set_index(ID)
    fo = pd.read_csv(ROOT / "artifacts" / "folds.csv")
    fo.columns = [c.strip() for c in fo.columns]
    fo = fo.set_index(ID)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=a.fp)
    ids = list(fo.index)
    X = np.zeros((len(ids), a.fp), dtype=np.uint8)
    bad = 0
    for i, k in enumerate(ids):
        m = Chem.MolFromSmiles(str(fo.loc[k, "SMILES"]))
        if m is None:
            bad += 1
            continue
        X[i] = gen.GetFingerprintAsNumPy(m)[:a.fp]
    Xd = pd.DataFrame(X, index=ids)
    print(f"features {X.shape}" + (f"   ({bad} SMILES failed to parse)" if bad else ""))
    print(f"weight = {a.weight}   cap = {a.cap}   powers = {a.powers}   seeds = {a.seeds}\n")

    P = {"num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 20, "verbose": -1,
         "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1}
    res = {}
    for iso in a.iso:
        c = f"{iso}_pIC50_direct_inhibition"
        idx = [k for k in ids if k in tr.index and pd.notna(tr.loc[k, c])]
        y = tr.loc[idx, c].to_numpy()
        lo = tr.loc[idx, c + "_conf_low"].to_numpy()
        hi = tr.loc[idx, c + "_conf_high"].to_numpy()
        Xi = Xd.loc[idx].to_numpy(np.float32)
        fi = fo.loc[idx, "fold"].to_numpy()

        # the unweighted arm first -- it fixes s for every weighted arm that follows
        def run(w):
            oof = np.zeros(len(idx))
            for sd in a.seeds:
                o = np.zeros(len(idx))
                for k in sorted(set(fi)):
                    te = fi == k
                    t = ~te
                    ds = lgb.Dataset(Xi[t], label=y[t], weight=None if w is None else w[t])
                    m = lgb.train({**P, "objective": "l2", "seed": sd}, ds, num_boost_round=a.nest)
                    o[te] = m.predict(Xi[te])
                oof += o / len(a.seeds)
            return oof

        base_oof = run(None)
        s = float(np.std(y - base_oof))
        h = np.maximum((hi - lo) / 2.0, 1e-6)
        raw = (1.0 / np.maximum(hi - lo, 0.05)) if a.weight == "invwidth" \
            else 2.0 * (1.0 - norm.cdf(h / s))
        raw = np.maximum(raw, 1e-6)

        print(f"{iso}   n={len(idx)}   error scale s={s:.3f}   "
              f"weight range before cap {raw.min():.3f}-{raw.max():.3f}")
        print(f"  {'power':>6}{'rho':>9}{'cost-wtd rho':>14}{'raw ST-RAE':>12}"
              f"{'PLACED':>9}{'delta placed':>14}")
        wq = 2.0 * (1.0 - norm.cdf(h / s))
        base_placed = None
        for pw in a.powers:
            w = np.clip((raw / np.median(raw)) ** pw, 1.0 / a.cap, a.cap)
            oof = base_oof if pw == 0 else run(w)
            pl = placed(oof, lo, hi, y)
            if base_placed is None:
                base_placed = pl
            nb = cost(np.full_like(y, y.mean()), lo, hi).sum()
            print(f"  {pw:>6.1f}{spearmanr(y, oof).statistic:>9.4f}"
                  f"{weighted_spearman(y, oof, wq):>14.4f}"
                  f"{cost(oof, lo, hi).sum() / nb:>12.4f}{pl:>9.4f}"
                  f"{base_placed - pl:>+14.4f}")
            res.setdefault(pw, []).append(pl)
        print()

    # THE REFERENCE ROW IS powers[0], WHICH IS ONLY "UNIFORM" WHEN IT IS 0.0.
    # The first version of this block labelled powers[0] "baseline (uniform weights)"
    # unconditionally and then declared "Uniform weighting wins" whenever it came first.
    # Run with `--powers 2 3 4 6` that printed exactly backwards: power 2.0 is HEAVY cost
    # weighting, it beat 3/4/6, and the script announced that cost weighting had failed.
    # A verdict line that can contradict its own table is worse than no verdict line.
    ref = a.powers[0]
    uniform_present = any(abs(p_) < 1e-9 for p_ in a.powers)
    b = np.mean(res[ref])
    print("=== macro placed ST-RAE by weight power ===")
    for pw in a.powers:
        v = np.mean(res[pw])
        tag = ("   <- reference" + (" (uniform weights)" if abs(pw) < 1e-9 else
               f" (power {pw:g}, NOT uniform)")) if pw == ref else \
              f"   {b - v:+.4f}  ({(b - v) / b:+.2%})"
        print(f"  power {pw:>4.1f}   {v:.4f}{tag}")
    best = min(a.powers, key=lambda p_: np.mean(res[p_]))
    print(f"\n  best power: {best:g}")

    if not uniform_present:
        print(f"  NOTE: power 0 was not in this sweep, so NOTHING here is a comparison")
        print(f"  against uniform weighting. This run only orders {a.powers} among themselves.")
        if best == ref:
            print(f"  Within that set the smallest power tested wins, so the optimum is at or")
            print(f"  below {ref:g} -- re-run including 0 to say anything about whether weighting helps.")
    elif abs(best) < 1e-9:
        print("  Uniform weighting wins. Cost weighting does not transfer to this rig, and")
        print("  porting it into train_cyp_mh.py is not justified on this evidence.")
    else:
        print(f"  Cost weighting wins by {(b - np.mean(res[best])) / b:+.2%} of macro placed ST-RAE.")
        if len(a.seeds) == 1:
            print("  BEFORE spending GPU hours: re-run with --seeds 0 1 2 to confirm the gap")
            print("  survives the seed noise floor. A single-seed gap on this rig decides nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
