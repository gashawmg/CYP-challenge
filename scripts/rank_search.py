#!/usr/bin/env python
"""Search for a better RANKING, which is the only pIC50 lever left.

    python scripts/rank_search.py
    python scripts/rank_search.py --write artifacts/oof_blend4.csv

WHY RANKING AND NOTHING ELSE. The floor `absorption*sqrt(1-rho^2)` says what today's exact
ranking would score if every value were placed perfectly. On `blend3_placed` the gap
between that floor and the actual score is +0.002 macro ST-RAE, down from +0.174 before
the placement work. Placement is finished. Every remaining point has to come from rho,
and rho is concentrated in one endpoint: CYP2D6 sits at blind 0.5072 against 0.79-0.85 on
the other three, and lifting it alone to 0.85 is worth -0.072 macro ST-RAE -- about 70% of
everything left.

WHY THE SEARCH IS PER ENDPOINT. The four endpoints do not agree on which member is best,
and the macro average does not care whether a gain comes from one endpoint or spreads
across four. `blend3` already uses per-endpoint weights for this reason.

HOW THE OVERFITTING IS CONTROLLED. `blind_validator` resamples the evaluation pool to the
solved blind label distribution B times. Picking the argmax over many weight vectors on
the mean of those draws would fit the resampling noise, so:

  * the search is over a COARSE simplex (0.1 steps), which is far fewer candidates than
    the number of effective observations;
  * every reported winner carries the across-draw sd, and a candidate that beats the
    incumbent by less than that sd is reported as a TIE, not a win;
  * the incumbent (`blend3`) is scored in the same run on the same draws, so the
    comparison is paired rather than against a remembered number.

WHAT IT WILL NOT DO. It cannot tell you the blind rho. The validator reshapes the LABEL
distribution to match the blind set; it cannot conjure chemistry we never measured. Its
job is to stop us choosing a ranking that is worse -- which has happened three times --
not to promise the improvement transfers.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cypkit import config as C

# absorption constant recovered from the scored floor table; used only to translate a
# rho gain into an ST-RAE expectation, never to choose anything
ABSORPTION = 0.8345
BLIND_RHO = {"CYP1A2": 0.7892, "CYP2C9": 0.8454, "CYP2D6": 0.5072, "CYP3A4": 0.8231}

MEMBERS = {
    "pb45":   "artifacts/oof_pseudo_blend45.csv",
    "v6":     "oof_st_v6.csv",
    "v7":     "oof_st_v7.csv",
    "uni":    "artifacts/oof_unimol_cyp_mt.csv",
    "cp3ps":  "oof_chemprop_st_v3_ps.csv",
}
# blend3 = pb45 / v6 / uni at these per-endpoint weights (artifacts/blend3_weights.json)
BLEND3 = {"CYP1A2": (.5, .2, .3), "CYP2C9": (.5, .3, .2),
          "CYP2D6": (.3, .5, .2), "CYP3A4": (.5, .3, .2)}


def load_members():
    out = {}
    for nm, p in MEMBERS.items():
        if not Path(p).exists():
            print(f"  {nm}: {p} missing, skipped")
            continue
        d = pd.read_csv(p); d.columns = [c.strip() for c in d.columns]
        out[nm] = d.set_index(C.ID_COL)
    return out


def truth():
    lab = pd.read_csv(C.artifact("labels.csv"))
    lab = lab[lab["kind"] != "left"]
    return lab.pivot_table(index=C.ID_COL, columns="isoform", values="y", aggfunc="mean")


def rank01(x):
    r = pd.Series(x).rank().to_numpy()
    return (r - np.nanmin(r)) / max(np.nanmax(r) - np.nanmin(r), 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--write", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mem = load_members()
    piv = truth()
    names = list(mem)
    print(f"members: {names}\n")

    rng = np.random.default_rng(args.seed)
    grid = [w for w in itertools.product(
        np.round(np.arange(0, 1 + 1e-9, args.step), 3), repeat=len(names))
        if abs(sum(w) - 1) < 1e-9]
    print(f"{len(grid)} weight vectors on a {args.step} simplex, "
          f"{args.draws} bootstrap draws each\n")

    chosen, blended = {}, {}
    for iso in C.ISOFORMS:
        ep = f"{iso}_pIC50_direct_inhibition"
        ids = piv.index[np.isfinite(piv[iso])]
        ids = [i for i in ids if all(i in mem[n].index for n in names)]
        y = piv.loc[ids, iso].to_numpy(float)
        R = {n: rank01(mem[n].loc[ids, ep].to_numpy(float)) for n in names}

        # paired bootstrap: the SAME resample indices score every candidate
        draws = [rng.integers(0, len(ids), len(ids)) for _ in range(args.draws)]

        # Two stages, because 1001 candidates x B bootstraps of a fresh rank sort does not
        # finish. Stage 1 ranks every candidate ONCE on the full pool -- cheap, and a
        # point estimate is enough to discard the obviously worse. Stage 2 spends the
        # paired bootstrap only on the shortlist, which is where the noise question
        # actually lives.
        ry = pd.Series(y).rank().to_numpy()

        def point(w):
            s = sum(wi * R[n] for wi, n in zip(w, names))
            return np.corrcoef(ry, pd.Series(s).rank().to_numpy())[0, 1]

        def score(w):
            s = sum(wi * R[n] for wi, n in zip(w, names))
            return np.array([spearmanr(y[d], s[d]).statistic for d in draws])

        # blend3's weights are indexed by ITS member order, not by the search's
        b3 = dict(zip(("pb45", "v6", "uni"), BLEND3[iso]))
        inc_w = tuple(b3.get(n, 0.0) for n in names)
        short = sorted(grid, key=point, reverse=True)[:8]
        inc = score(inc_w)
        best, best_s = None, None
        for w in short:
            v = score(w)
            if best_s is None or v.mean() > best_s.mean():
                best, best_s = w, v

        d = best_s - inc                      # paired difference, so its sd is the test
        tie = abs(d.mean()) < d.std()
        chosen[iso] = (best, best_s.mean(), inc.mean(), d.mean(), d.std(), tie)
        w_txt = " ".join(f"{n}{wi:.1f}" for n, wi in zip(names, best) if wi > 0)
        print(f"{iso}")
        print(f"  incumbent blend3   rho {inc.mean():.4f}")
        print(f"  best on the grid   rho {best_s.mean():.4f}   [{w_txt}]")
        print(f"  paired difference  {d.mean():+.4f} +- {d.std():.4f}"
              f"   {'TIE -- keep the incumbent' if tie else 'REAL'}\n")

        use = inc_w if tie else best
        blended[iso] = (use, {i: v for i, v in zip(
            ids, sum(wi * R[n] for wi, n in zip(use, names)))})

    print("=" * 74)
    print("WHAT THIS WOULD BE WORTH, if the local gain transferred at 1:1\n")
    print(f"  {'endpoint':9s} {'blind rho now':>14s} {'local delta':>12s} "
          f"{'implied rho':>12s} {'floor now':>10s} {'floor then':>11s}")
    f0, f1 = [], []
    for iso in C.ISOFORMS:
        _, b, i, dm, ds, tie = chosen[iso]
        r0 = BLIND_RHO[iso]
        r1 = r0 if tie else min(r0 + dm, 0.99)
        a = ABSORPTION * np.sqrt(1 - r0 ** 2)
        c = ABSORPTION * np.sqrt(1 - r1 ** 2)
        f0.append(a); f1.append(c)
        print(f"  {iso:9s} {r0:14.4f} {(0 if tie else dm):+12.4f} {r1:12.4f} "
              f"{a:10.4f} {c:11.4f}")
    print(f"\n  macro floor {np.mean(f0):.4f} -> {np.mean(f1):.4f}  "
          f"({np.mean(f1)-np.mean(f0):+.4f})")
    print("\n  A 1:1 transfer is the OPTIMISTIC bound, not a forecast: local rho gains\n"
          "  have transferred at roughly 0.08 of face value on this track "
          "(+0.052 local -> +0.004 blind).")

    if args.write:
        rows = []
        for iso, (w, m) in blended.items():
            rows.append(pd.Series(m, name=f"{iso}_pIC50_direct_inhibition"))
        out = pd.concat(rows, axis=1)
        out.index.name = C.ID_COL
        out.reset_index().to_csv(args.write, index=False)
        print(f"\nwrote {args.write} (RANKS, not pIC50 -- for comparison only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
