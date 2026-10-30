#!/usr/bin/env python
"""Turn a chosen member list into a TDI submission, with the cut rate re-derived.

    python scripts/tdi_submit_selected.py --out artifacts/submission_tdi_sel.csv

WHY THIS IS NOT JUST A RANK-AND-CUT. Two things change together when the score column
improves, and only one of them is obvious.

  1. The RANKING changes -- that is the point, and it is what `tdi_select.py` chose.
  2. The OPTIMAL CUT RATE changes too. The rate that maximises MCC is a function of
     discriminability: a sharper model should call MORE positives on CYP3A4 and fewer on
     CYP2D6 than a blunt one, because the binormal optimum moves as d' moves. The
     standing OPTIMAL_RATE = {CYP3A4 0.306, CYP2D6 0.142} was solved at the OLD d'
     (1.290 / 0.980). Reusing it with a better score column throws away part of the gain.

So this re-solves the optimum at the new d' and only then cuts, applying the measured
rate inflation (the leaderboard scores a subset, and our submitted rate arrives ~8.7% /
25% higher there than we sent it).

THE CHECK THAT MATTERS is the last block: how many labels actually move against the
standing best. A change that flips 3% of the file cannot possibly be worth a slot; one
that flips 25% is a genuinely different submission and the predicted gain is the thing
to trust or not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from cypkit import config as C
from scripts.tdi_validator import (PREVALENCE, SHRINK, d_prime, best_mcc)
from scripts.tdi_select import rank01, blend

RATE_INFLATION = {"CYP3A4": 0.2290 / 0.2107, "CYP2D6": 0.2136 / 0.1707}

# Chosen by scripts/tdi_select.py under nested selection; see that script's output.
PICK = {"CYP3A4": ["lgb_clf", "struct", "pooled", "chemprop", "chemprop_tdi"],
        "CYP2D6": ["knn_fcfp", "svc_c10"]}


def member_test(name, iso, tag, chemprop_tdi):
    if name == "chemprop_tdi":
        d = pd.read_csv(chemprop_tdi)
    else:
        d = pd.read_csv(C.artifact(f"tdi_member_test_{name}_{tag}.csv"))
    d.columns = [c.strip() for c in d.columns]
    return d[[C.ID_COL, f"{iso}_score"]].rename(columns={f"{iso}_score": name})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="tdiK")
    ap.add_argument("--oof", default="artifacts/oof_tdi_sel.csv")
    ap.add_argument("--chemprop-test", default="artifacts/pred_tdi_chemprop.csv")
    ap.add_argument("--reference", default="artifacts/submission_tdi_best.csv")
    ap.add_argument("--out", default="artifacts/submission_tdi_sel.csv")
    args = ap.parse_args()

    oof = pd.read_csv(args.oof)
    test = pd.read_csv(C.DATA_DIR / C.DATA_FILES["test"])
    test.columns = [c.strip() for c in test.columns]
    ref = pd.read_csv(args.reference)

    sub = test[[C.SMILES_COL, C.ID_COL]].copy()
    print(f"{'iso':8s} {'OOF AUC':>8s} {'corr d':>7s} {'opt rate':>9s} {'submit':>8s} "
          f"{'pred MCC':>9s}")
    for iso, ms in PICK.items():
        y = pd.to_numeric(oof[f"{iso}_y"], errors="coerce")
        s = oof[f"{iso}_score"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(s)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y[ok].astype(int), s[ok])
        d = d_prime(auc) * SHRINK[iso]
        mcc, rate = best_mcc(d, PREVALENCE[iso])
        send = rate / RATE_INFLATION[iso]

        cols = [member_test(m, iso, args.tag, args.chemprop_test) for m in ms]
        T = cols[0]
        for c in cols[1:]:
            T = T.merge(c, on=C.ID_COL, how="inner")
        if len(T) != len(test):
            raise SystemExit(f"{iso}: {len(T)} test rows from members, expected {len(test)}")
        r = {m: rank01(T[m].to_numpy(float)) for m in ms}
        score = blend(r, ms)
        T["_s"] = score
        m = sub[C.ID_COL].map(dict(zip(T[C.ID_COL], T["_s"]))).to_numpy(float)
        cut = float(np.quantile(m, 1 - send))
        sub[f"{iso}_is_TDI"] = m >= cut
        print(f"{iso:8s} {auc:8.4f} {d:7.3f} {rate:9.3f} {send:8.3f} {mcc:9.4f}")

    sub.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    print("\nAGAINST THE STANDING BEST")
    j = sub.merge(ref, on=C.ID_COL, suffixes=("", "_ref"))
    for iso in PICK:
        a = j[f"{iso}_is_TDI"].astype(bool)
        b = j[f"{iso}_is_TDI_ref"].astype(bool)
        print(f"  {iso}: called {a.mean():.3f} (was {b.mean():.3f})   "
              f"labels moved {(a != b).mean()*100:.1f}%  "
              f"[+{int((a & ~b).sum())} / -{int((~a & b).sum())}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
