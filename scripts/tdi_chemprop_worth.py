#!/usr/bin/env python
"""Does ChemProp earn its place in the TDI blend, or is classical ML enough?

    python scripts/tdi_chemprop_worth.py
    python scripts/tdi_chemprop_worth.py --tag tdiK --far-only

WHY THE OBVIOUS TEST IS THE WRONG ONE. "ChemProp's out-of-fold AUC is 0.821, the best
single member on CYP3A4" is true and nearly useless. A blend does not want the best
member, it wants the member that is wrong somewhere the others are right. And we have
just been taught what happens when you select on pooled out-of-fold AUC: the five-member
CYP3A4 blend that included BOTH ChemProp columns scored 0.843 out of fold and then
realised blind d' 1.135 against the 19-member equal blend's 1.290. So standalone AUC is
exactly the number that misled us.

THE FOUR QUESTIONS THIS ASKS INSTEAD.

  1. MARGINAL VALUE. Take the classical pool as it stands and ADD ChemProp. That is the
     decision actually on the table -- not "ChemProp instead of classical ML", which is
     what a standalone comparison implicitly asks.

  2. DOES IT SURVIVE OFF-MANIFOLD? Every compound's distance to its own fold's training
     half splits the pool at the median. The blind set sits at the FAR end of our own
     distribution, so a contribution that exists only in NEAR is a contribution that will
     not be there on the leaderboard.

  3. IS IT SAYING SOMETHING NEW? Rank correlation between ChemProp and the classical
     consensus. A member at rho 0.95 cannot help however good it is; one at 0.7 can help
     even if it is worse.

  4. IS THE ANSWER THE SAME ON BOTH ENDPOINTS? It is not, and that is the finding. Nothing
     here generalises across CYP3A4 and CYP2D6 -- they have different amounts of signal,
     different prevalence, and a measured d' shrinkage of 1.0 against 2.3.

Nested where it matters: any comparison involving a CHOICE is made on training folds and
scored on a held-out fold. Adding a fixed member to a fixed pool involves no choice, so it
is scored directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cypkit import config as C
from scripts.tdi_select import rank01, blend
from scripts.tdi_transfer import fingerprints, dist_to_own_train

CHEMPROP = ("chemprop", "chemprop_tdi")


def d_prime(a):
    from scipy.stats import norm
    return np.sqrt(2) * norm.ppf(np.clip(a, 1e-6, 1 - 1e-6))


def boot_delta(y, a, b, n=600, seed=0):
    """Bootstrap the DIFFERENCE in AUC, paired -- the only way to know if it is real."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="tdiK")
    ap.add_argument("--extra", default="artifacts/oof_tdi_chemprop.csv")
    ap.add_argument("--folds", default=None)
    args = ap.parse_args()

    folds = pd.read_csv(args.folds or C.artifact("folds_tdi.csv"))
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))
    src = pd.read_csv(C.DATA_DIR / C.DATA_FILES["train_tdi"])
    src.columns = [c.strip() for c in src.columns]
    smi_of = dict(zip(src[C.ID_COL], src[C.SMILES_COL]))
    cp = pd.read_csv(args.extra); cp.columns = [c.strip() for c in cp.columns]

    for iso in C.TDI_ISOFORMS:
        t = pd.read_csv(C.artifact(f"tdi_member_oof_{iso}_{args.tag}.csv"))
        if f"{iso}_score" in cp.columns:
            t = t.merge(cp[[C.ID_COL, f"{iso}_score"]].rename(
                columns={f"{iso}_score": "chemprop_tdi"}), on=C.ID_COL, how="left")
        members = [c for c in t.columns
                   if c not in (C.ID_COL, "y") and not t[c].isna().any()]
        classical = [m for m in members if m not in CHEMPROP]
        cps = [m for m in members if m in CHEMPROP]
        y = t["y"].to_numpy().astype(int)
        R = {m: rank01(t[m].to_numpy(float)) for m in members}

        fold = t[C.ID_COL].map(fold_of).to_numpy()
        d = dist_to_own_train(fingerprints([smi_of.get(i) for i in t[C.ID_COL]]), fold)
        cut = np.nanmedian(d)
        NEAR, FAR = d >= cut, d < cut

        print("=" * 78)
        print(f"{iso}   {len(classical)} classical members, {len(cps)} ChemProp\n")

        # ---- 1 & 2 -------------------------------------------------- marginal value
        base = blend(R, classical)
        combos = [("classical only", classical),
                  ("+ chemprop", classical + ["chemprop"]),
                  ("+ chemprop_tdi", classical + ["chemprop_tdi"]),
                  ("+ both", classical + cps),
                  ("ChemProp ALONE", cps)]
        print(f"  {'blend':18s} {'all':>8s} {'NEAR':>8s} {'FAR':>8s} "
              f"{'d vs classical (all)':>24s}")
        for nm, ms in combos:
            s = blend(R, ms)
            a = roc_auc_score(y, s)
            an = roc_auc_score(y[NEAR], s[NEAR])
            af = roc_auc_score(y[FAR], s[FAR])
            if nm == "classical only":
                extra = ""
            else:
                bd = boot_delta(y, s, base)
                lo, hi = np.percentile(bd, [2.5, 97.5])
                sig = "  real" if lo > 0 else ("  HURTS" if hi < 0 else "  noise")
                extra = f"{a-roc_auc_score(y, base):+8.4f} [{lo:+.4f},{hi:+.4f}]{sig}"
            print(f"  {nm:18s} {a:8.4f} {an:8.4f} {af:8.4f} {extra:>24s}")

        # ---- 3 ------------------------------------------------------- is it new info
        print(f"\n  rank correlation with the classical consensus:")
        for m in cps:
            rho = np.corrcoef(R[m], base)[0, 1]
            best_cl = max(classical, key=lambda x: roc_auc_score(y, R[x]))
            print(f"    {m:14s} rho {rho:.3f}   "
                  f"(best classical member {best_cl} sits at "
                  f"{np.corrcoef(R[best_cl], base)[0,1]:.3f})")

        # ---- standalone, for reference only
        print(f"\n  standalone AUC -- the number that misled us, kept for reference:")
        rank = sorted(members, key=lambda m: -roc_auc_score(y, R[m]))
        for i, m in enumerate(rank[:5], 1):
            tagm = "  <- ChemProp" if m in CHEMPROP else ""
            print(f"    {i}. {m:14s} {roc_auc_score(y, R[m]):.4f}{tagm}")
        for m in cps:
            if m not in rank[:5]:
                print(f"    {rank.index(m)+1}. {m:14s} "
                      f"{roc_auc_score(y, R[m]):.4f}  <- ChemProp")
        print()

    print("HOW TO DECIDE FROM THIS. Keep ChemProp where '+ both' beats 'classical only'\n"
          "by more than its bootstrap interval AND the gain is present in the FAR column.\n"
          "A gain that lives only in NEAR is a gain that stays out of fold: the blind set\n"
          "is farther from our training set than our own held-out compounds are.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
