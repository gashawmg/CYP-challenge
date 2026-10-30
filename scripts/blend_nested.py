r"""
blend_nested.py  --  is the shipped blend beaten by something already on disk?

WHAT PROVOKED THIS. scripts\ranking_headroom.py measured, on the frozen folds with every
member given its own optimal affine, that the shipped four-member blend is BEATEN by the
best single member on three endpoints out of four:

    CYP2C9   blend 0.1675   best single 0.1672   (mhtdi5)
    CYP2D6   blend 0.3418   best single 0.3373   (mhtdi5)
    CYP3A4   blend 0.2129   best single 0.2009   (mhtdi5)

and mhtdi5 -- the five-seed multi-head with the CYP3A4 TDI head -- is not in the shipped
blend at all. It also measured the pool's placed-error correlations at 0.85 (CYP2C9), 0.96
(CYP2D6) and 0.85 (CYP3A4), far above the 0.571-0.698 recorded in STATUS.md. At those
correlations averaging buys almost nothing, and averaging a strong member with two weak
ones (unimol and cp_ensA are the two worst on every endpoint) actively costs.

WHY THAT IS NOT YET A REASON TO CHANGE ANYTHING. Section 74 already measured a blend
re-composition at "+0.0086 apparent" and then found the figure was manufactured by picking
the winner among 26 compositions on the same rows used to score them. A greedy search that
chooses and scores on one set of rows will always flatter itself. The blend4tdi submission
lost real board points to exactly this family of error.

SO THE SELECTION IS NESTED HERE. For each held-out fold the composition is chosen greedily
on the OTHER four folds only, then scored on the held-out fold it never saw. Every arm --
shipped, best-single, greedy -- is selected the same way and scored on the same rows, so
the comparison is between procedures rather than between one tuned thing and one untuned
thing. A paired bootstrap over compounds follows, and the gate is the one this project
already uses: P(worse) must exclude zero.

WHAT A WIN HERE WOULD AND WOULD NOT MEAN. Out-of-fold ST-RAE levels are not board levels --
our folds are harder than the blind set (median max-Tanimoto 0.438 against 0.587). Only
DIFFERENCES transfer, and even those transfer imperfectly. A win says the composition is
worth a slot; it does not predict the board number.

Run (Windows, admet_gpu):
  python scripts\blend_nested.py
  python scripts\blend_nested.py --boot 4000 --max-size 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from ranking_headroom import ISO, MEMBERS, SHIPPED, opt_shift, place, strae  # noqa: E402

TRUTH = ROOT / "data" / "cyp-challenge-TRAIN_inhibition.csv"
FOLDS = ROOT / "artifacts" / "folds.csv"


def build(iso, truth, mem):
    col = f"{iso}_pIC50_direct_inhibition"
    lo_c, hi_c = f"{col}_conf_low", f"{col}_conf_high"
    t = truth[["Molecule_Name", col, lo_c, hi_c, "fold"]].dropna()
    names, frames = [], []
    for n, d in mem.items():
        if col not in d.columns:
            continue
        s = d[["Molecule_Name", col]].dropna().rename(columns={col: n})
        if len(s) < 500:
            continue
        names.append(n)
        frames.append(s)
    j = t
    for f in frames:
        j = j.merge(f, on="Molecule_Name", how="inner")
    return col, lo_c, hi_c, names, j.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--max-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    truth = pd.read_csv(TRUTH).merge(pd.read_csv(FOLDS)[["Molecule_Name", "fold"]],
                                     on="Molecule_Name", how="inner")
    mem = {n: pd.read_csv(ROOT / r) for n, r in MEMBERS.items() if (ROOT / r).exists()}

    totals = {"shipped": [], "best_single": [], "greedy": []}
    for iso in ISO:
        col, lo_c, hi_c, names, J = build(iso, truth, mem)
        fold = J["fold"].to_numpy().astype(int)
        y = J[col].to_numpy(float)
        lo = J[lo_c].to_numpy(float)
        hi = J[hi_c].to_numpy(float)
        # place every member ONCE on the full row set, then only the CHOICE is nested
        P = {}
        for n in names:
            _, P[n] = place(J[n].to_numpy(float), y, lo, hi)
        shipped = [n for n in SHIPPED if n in names]

        def score(sel, mask):
            b = np.mean([P[n][mask] for n in sel], axis=0)
            return opt_shift(b, lo[mask], hi[mask])[1]

        oof = {k: np.zeros(len(J)) for k in totals}
        picks = []
        for k in sorted(set(fold)):
            te, tr = fold == k, fold != k
            # greedy forward selection on the TRAINING folds only
            cur, best_v = [], np.inf
            while len(cur) < a.max_size:
                cand = [(score(cur + [n], tr), n) for n in names if n not in cur]
                v, n = min(cand)
                if v >= best_v - 1e-6:
                    break
                cur, best_v = cur + [n], v
            single = min(((score([n], tr), n) for n in names))[1]
            picks.append((k, tuple(cur), single))
            for key, sel in (("shipped", shipped), ("best_single", [single]),
                             ("greedy", cur)):
                b = np.mean([P[n] for n in sel], axis=0)
                sh, _ = opt_shift(b[tr], lo[tr], hi[tr])   # shift fitted on train folds too
                oof[key][te] = b[te] + sh
        print(f"\n=== {iso}   {len(J)} rows, {len(names)} members")
        for k, cur, single in picks:
            print(f"   fold {k}: greedy picked {'+'.join(cur)}   | best single {single}")
        res = {}
        for key in totals:
            v = strae(oof[key], lo, hi).mean()
            res[key] = v
            totals[key].append(v)
        print(f"   NESTED raw ST-RAE   shipped {res['shipped']:.4f}   "
              f"best_single {res['best_single']:.4f}   greedy {res['greedy']:.4f}")
        for key in ("best_single", "greedy"):
            d = np.empty(a.boot)
            n = len(y)
            for b in range(a.boot):
                ix = rng.integers(0, n, n)
                d[b] = (strae(oof[key][ix], lo[ix], hi[ix]).mean()
                        - strae(oof["shipped"][ix], lo[ix], hi[ix]).mean())
            print(f"     {key:<12} vs shipped: {res[key]-res['shipped']:+.4f}   "
                  f"P(worse) {float((d >= 0).mean()):.3f}")

    print("\n" + "=" * 70)
    print("MACRO of the nested raw ST-RAE (not a board number -- differences only)")
    for key in totals:
        print(f"   {key:<12}{np.mean(totals[key]):.4f}")
    print("\nIf greedy beats shipped here it beat a composition it was never shown, which")
    print("is the thing Section 74's +0.0086 could not claim. Transfer to the board is still")
    print("an assumption: our folds are harder than the blind set and only deltas carry.")


if __name__ == "__main__":
    main()
