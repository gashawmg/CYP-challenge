r"""
ranking_headroom.py  --  what is a new model actually worth, before one is built?

PLACEMENT IS FINISHED. The ledger, with no model change anywhere in it, is d6probe 0.4737
-> CYP1A2 centre 0.4620 -> CYP3A4 centre 0.4504 -> CYP2D6 spread 0.4440, and every
remaining centre and spread lever adds up to about 0.003 macro, inside our own estimate
error. Spearman has sat at 0.7558 through all of it. Any further gain has to come from the
ranking, which means from the models.

THE TRAP THIS SCRIPT EXISTS TO AVOID. Two facts already measured say that most model work
is wasted here. The blend keeps roughly 12 per cent of a member-level gain (Section 75), and
60-86 per cent of every member's error is shared with the other members. Target descriptors
gained +0.0183 macro standalone and +0.0016 after blending. Seed averaging gained +0.13
summed standalone and roughly zero after blending. So "train a better model" has been tried,
repeatedly, and the blend ate it every time.

What has never been priced is the other axis. A blend's error variance is
(1/n^2) * sum_ij cov(e_i, e_j), so what a new member contributes depends on its CORRELATION
to the pool at least as much as on its own accuracy. This script measures the pool's real
error covariance and then answers, for a member that does not exist yet:

    given quality q (its error sd relative to the pool's) and mean error correlation rho
    to the existing members, what does adding it move the MACRO by?

That map is the point. It says whether a decorrelated fifth member is worth weeks, and how
good it would have to be -- BEFORE anyone trains it. If the map says a perfectly
decorrelated member at equal quality buys 0.004, then the ranking is not where the
remaining gap lives either, and we should say so rather than grind.

EVERY COMPARISON IS PLACED. Each member gets its OWN optimal affine before being scored,
because comparing two accidental placements is not comparing two models. Rows are the
intersection where the member and the truth both exist, per endpoint, on the frozen folds.

Run (Windows, admet_gpu):
  python scripts\ranking_headroom.py
  python scripts\ranking_headroom.py --boot 3000
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
TRUTH = ROOT / "data" / "cyp-challenge-TRAIN_inhibition.csv"
FOLDS = ROOT / "artifacts" / "folds.csv"
ISO = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")

MEMBERS = {
    "v6": "oof_st_v6.csv",
    "v7": "oof_st_v7.csv",
    "v5_rest": "oof_st_v5_rest.csv",
    "mh_ctl": "oof_st_mh_ctl.csv",
    "mh2": "oof_st_mh2.csv",
    "mhens3": "oof_mhens3.csv",
    "mhtdi5": "oof_mhtdi5.csv",
    "unimol": "oof_unimol_cyp_mt.csv",
    "cp_ensA": "artifacts/oof_chempropv3_ensA.csv",
    "cp_v3": "oof_chemprop_st_v3.csv",
}
SHIPPED = ["mh_ctl", "v6", "cp_ensA", "unimol"]   # the four-member family that ships


def strae(p, lo, hi):
    """soft-thresholded absolute error, the raw numerator of the board metric"""
    return np.clip(p - hi, 0, None) + np.clip(lo - p, 0, None)


def opt_shift(p, lo, hi):
    """exact minimiser of mean strae over a uniform shift: the derivative
    #(p+s > hi) - #(p+s < lo) crosses zero at a breakpoint"""
    cand = np.concatenate([hi - p, lo - p])
    cand.sort()
    best, bs = np.inf, 0.0
    for s in cand:
        v = strae(p + s, lo, hi).mean()
        if v < best:
            best, bs = v, s
    return bs, best


def place(p, y, lo, hi, scales=None):
    """fit this member its OWN optimal affine, then score. Anything less compares
    two accidental placements rather than two models."""
    if scales is None:
        scales = np.linspace(0.5, 1.8, 27)
    m = p.mean()
    best = (np.inf, 1.0, 0.0)
    for s in scales:
        q = m + s * (p - m)
        sh, v = opt_shift(q, lo, hi)
        if v < best[0]:
            best = (v, s, sh)
    v, s, sh = best
    return v, m + s * (p - m) + sh


def load_members(verbose=True):
    truth = pd.read_csv(TRUTH)
    folds = pd.read_csv(FOLDS)[["Molecule_Name", "fold"]]
    truth = truth.merge(folds, on="Molecule_Name", how="inner")
    out = {}
    for name, rel in MEMBERS.items():
        p = ROOT / rel
        if not p.exists():
            if verbose:
                print(f"  {name:<10} MISSING {rel}")
            continue
        out[name] = pd.read_csv(p)
    return truth, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    truth, mem = load_members()
    print(f"members loaded: {', '.join(mem)}\n")

    macro_single, macro_err, summary = {}, {}, {}
    for iso in ISO:
        col = f"{iso}_pIC50_direct_inhibition"
        lo_c, hi_c = f"{col}_conf_low", f"{col}_conf_high"
        t = truth[["Molecule_Name", col, lo_c, hi_c]].dropna()
        names, preds = [], []
        for n, d in mem.items():
            if col not in d.columns:
                continue
            s = d[["Molecule_Name", col]].dropna().rename(columns={col: "p"})
            j = t.merge(s, on="Molecule_Name", how="inner")
            if len(j) < 500:
                continue
            names.append(n)
            preds.append(j)
        # restrict to rows EVERY member covers, so the covariance is on one row set
        common = set(preds[0]["Molecule_Name"])
        for j in preds[1:]:
            common &= set(j["Molecule_Name"])
        common = sorted(common)
        base = t[t["Molecule_Name"].isin(common)].set_index("Molecule_Name").loc[common]
        y = base[col].to_numpy(float)
        lo = base[lo_c].to_numpy(float)
        hi = base[hi_c].to_numpy(float)
        P, E, S = {}, {}, {}
        for n, j in zip(names, preds):
            p = j.set_index("Molecule_Name").loc[common, "p"].to_numpy(float)
            v, placed = place(p, y, lo, hi)
            P[n] = placed
            E[n] = placed - y
            S[n] = (v, spearmanr(p, y).statistic)
        print(f"=== {iso}   {len(common)} rows shared by all {len(names)} members")
        print(f"{'member':<10}{'placed raw ST-RAE':>19}{'rho':>8}{'err sd':>9}")
        for n in names:
            print(f"{n:<10}{S[n][0]:>19.4f}{S[n][1]:>8.4f}{E[n].std():>9.4f}")
        M = np.corrcoef(np.vstack([E[n] for n in names]))
        iu = np.triu_indices(len(names), 1)
        print(f"  mean pairwise ERROR correlation {M[iu].mean():.4f}   "
              f"range {M[iu].min():.4f} to {M[iu].max():.4f}")
        shipped = [n for n in SHIPPED if n in names]
        bl = np.mean([P[n] for n in shipped], axis=0)
        sh, vbl = opt_shift(bl, lo, hi)
        best_single = min(S[n][0] for n in names)
        print(f"  shipped blend ({'+'.join(shipped)}): {vbl:.4f}   "
              f"best single member: {best_single:.4f}   "
              f"blend advantage {best_single - vbl:+.4f}")
        summary[iso] = dict(names=names, E=E, P=P, y=y, lo=lo, hi=hi,
                            shipped=shipped, vbl=vbl, corr=M[iu].mean())
        print()

    # ---------------- the headroom map ----------------
    print("=" * 78)
    print("HEADROOM: what a member that does not exist yet would be worth")
    print("=" * 78)
    print("A synthetic member is built with a target error correlation to the pool and a")
    print("target error sd, by mixing the pool's own error direction with fresh noise, then")
    print("added to the shipped blend and re-placed. This prices DECORRELATION against")
    print("QUALITY on the only ruler that matters, the macro.\n")
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8]
    qs = [0.8, 1.0, 1.2, 1.5]
    grid = np.zeros((len(qs), len(rhos)))
    base_macro = np.mean([summary[i]["vbl"] for i in ISO])
    for qi, q in enumerate(qs):
        for ri, rho in enumerate(rhos):
            vals = []
            for iso in ISO:
                s = summary[iso]
                Epool = np.mean([s["E"][n] for n in s["shipped"]], axis=0)
                sd_pool = Epool.std()
                reps = []
                for _ in range(12):
                    z = rng.standard_normal(len(Epool))
                    z = (z - z.mean()) / z.std()
                    base_dir = (Epool - Epool.mean()) / sd_pool
                    e_new = rho * base_dir + np.sqrt(max(1 - rho * rho, 0)) * z
                    e_new = e_new / e_new.std() * (sd_pool * q)
                    p_new = s["y"] + e_new
                    bl = np.mean([s["P"][n] for n in s["shipped"]] + [p_new], axis=0)
                    reps.append(opt_shift(bl, s["lo"], s["hi"])[1])
                vals.append(np.mean(reps))
            grid[qi, ri] = np.mean(vals)
    print(f"{'q = err sd vs pool':<20}" + "".join(f"{f'rho={r:.1f}':>11}" for r in rhos))
    for qi, q in enumerate(qs):
        print(f"{q:<20.2f}" + "".join(f"{grid[qi,ri]-base_macro:>+11.4f}" for ri in range(len(rhos))))
    print(f"\n(macro CHANGE vs the shipped blend at {base_macro:.4f}; negative is better)")
    print("Read the row for the quality you could plausibly build and the column for the")
    print("correlation you could plausibly achieve. If the best cell you can honestly reach")
    print("is smaller than the 0.003 left in placement, the ranking is not the answer either.")


if __name__ == "__main__":
    main()
