#!/usr/bin/env python
"""Do activity cliffs explain where this model loses, and does it predict them at all?

    python scripts/activity_cliffs.py
    python scripts/activity_cliffs.py --oof oof_st_v6.csv --sim 0.8 --delta 1.5

WHY NOW. `scripts/applicability_domain.py` found that out-of-fold error is FLAT in
distance-to-training, and on CYP1A2 and CYP3A4 it actually FALLS as compounds get more
isolated (MAE 0.618 -> 0.483). That is backwards for an extrapolation problem and is the
signature of the opposite one: the dense, well-covered neighbourhoods are where near-
identical structures have very different activity, and no amount of extra coverage helps
there. That was a hypothesis in the applicability-domain write-up. This script tests it.

WHAT A CLIFF IS HERE. A pair of compounds with ECFP4 Tanimoto >= `--sim` whose measured
pIC50 differ by >= `--delta`. SALI = |dy| / (1 - Tanimoto) ranks them: two molecules that
are nearly the same and behave completely differently score highest.

THE THREE QUESTIONS, IN ORDER OF WHAT THEY SETTLE.

  1. HOW MUCH CLIFF IS THERE? Fraction of compounds in at least one cliff pair, per
     endpoint. If it is a few percent, cliffs are a curiosity; if it is a third of the
     set, they are the problem.

  2. DO WE LOSE ON THEM? Out-of-fold MAE on cliff compounds against non-cliff compounds.
     This is the part that decides whether any of it is worth acting on -- a population
     can be large and still not be where the error lives.

  3. DO WE PREDICT THE CLIFF, OR SMOOTH IT? For every cliff pair, compare the PREDICTED
     difference with the measured one. A model that has learned the cliff gets the sign
     and roughly the size; a model that has smoothed it predicts near zero for both
     members. This is the sharpest diagnostic, because a graph network with a smooth
     readout is architecturally biased toward smoothing, and the usual metrics cannot
     see it: getting both members of a cliff pair equally wrong in the same direction
     barely moves MAE, and moves Spearman not at all if the rest of the ranking holds.

WHAT TO DO ABOUT IT DEPENDS ON THE ANSWER, so this script deliberately stops at the
diagnosis. The options -- pair-weighted loss, SALI-weighted sampling, cliff-aware splits,
or simply flagging cliff neighbourhoods as low-confidence at prediction time -- are not
equally sensible, and which one applies is decided by whether question 3 says we smooth
cliffs or merely find them hard.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cypkit import config as C

RDLogger.DisableLog("rdApp.*")
def _morgan_gen():
    """AllChem.GetMorganGenerator is recent-RDKit only; rdFingerprintGenerator
    is where the repo's own code finds it and has existed far longer."""
    try:
        from rdkit.Chem import rdFingerprintGenerator as rdfp  # noqa: PLC0415
        return rdfp.GetMorganGenerator(radius=2, fpSize=2048)
    except (ImportError, AttributeError):
        return AllChem.GetMorganGenerator(radius=2, fpSize=2048)

GEN = _morgan_gen()


def labels_wide(data_dir=None):
    """One column per isoform of measured pIC50, fitted curves only."""
    lab = pd.read_csv(C.artifact("labels.csv"))
    lab = lab[lab["kind"] != "left"]
    return lab.pivot_table(index=C.ID_COL, columns="isoform", values="y", aggfunc="mean")


def cliff_pairs(smiles, ids, y, sim_cut, delta_cut, chunk=1500):
    """All (i, j) with Tanimoto >= sim_cut. Returns a frame with sim, dy, SALI."""
    fps, keep = [], []
    for k, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is not None and np.isfinite(y[k]):
            fps.append(GEN.GetFingerprint(m)); keep.append(k)
    keep = np.array(keep)
    rows = []
    for a in range(len(fps)):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[a], fps[a + 1:]))
        if not len(sims):
            continue
        hit = np.where(sims >= sim_cut)[0]
        for h in hit:
            b = a + 1 + h
            dy = abs(y[keep[a]] - y[keep[b]])
            rows.append((keep[a], keep[b], float(sims[h]), float(dy)))
    d = pd.DataFrame(rows, columns=["i", "j", "sim", "dy"])
    if d.empty:
        return d
    d["SALI"] = d["dy"] / np.clip(1 - d["sim"], 1e-3, None)
    d["cliff"] = d["dy"] >= delta_cut
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", nargs="*",
                    default=["artifacts/oof_pseudo_blend45.csv", "oof_st_v6.csv"])
    ap.add_argument("--sim", type=float, default=0.75)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--folds", default="artifacts/folds.csv")
    args = ap.parse_args()

    folds = pd.read_csv(args.folds)
    piv = labels_wide()
    preds = {}
    for p in args.oof:
        if not Path(p).exists():
            print(f"  {p}: not found, skipped"); continue
        d = pd.read_csv(p); d.columns = [c.strip() for c in d.columns]
        preds[Path(p).name] = d.set_index(C.ID_COL)

    print(f"cliff = ECFP4 Tanimoto >= {args.sim} and |dpIC50| >= {args.delta}\n")
    print("=" * 78)
    print("1. HOW MUCH CLIFF IS THERE?\n")
    print(f"  {'endpoint':8s} {'n':>6s} {'sim pairs':>10s} {'cliff pairs':>12s} "
          f"{'% of pairs':>11s} {'compounds in a cliff':>21s}")
    store = {}
    for iso in C.ISOFORMS:
        if iso not in piv.columns:
            continue
        sub = folds[folds[C.ID_COL].isin(piv.index)].copy()
        sub["y"] = sub[C.ID_COL].map(piv[iso])
        sub = sub[np.isfinite(sub["y"])].reset_index(drop=True)
        d = cliff_pairs(sub[C.SMILES_COL].tolist(), sub[C.ID_COL].tolist(),
                        sub["y"].to_numpy(float), args.sim, args.delta)
        store[iso] = (sub, d)
        if d.empty:
            print(f"  {iso:8s} {len(sub):6d} {0:10d}"); continue
        involved = set(d.loc[d.cliff, "i"]) | set(d.loc[d.cliff, "j"])
        print(f"  {iso:8s} {len(sub):6d} {len(d):10d} {int(d.cliff.sum()):12d} "
              f"{d.cliff.mean()*100:10.1f}% {len(involved):9d} "
              f"({len(involved)/len(sub)*100:.1f}%)")

    print("\n" + "=" * 78)
    print("2. DO WE LOSE ON THEM?  out-of-fold MAE, cliff vs non-cliff\n")
    for nm, P in preds.items():
        print(f"  {nm}")
        print(f"    {'endpoint':8s} {'non-cliff':>10s} {'cliff':>8s} {'ratio':>7s} "
              f"{'n cliff':>8s}")
        for iso in C.ISOFORMS:
            if iso not in store:
                continue
            sub, d = store[iso]
            ep = f"{iso}_pIC50_direct_inhibition"
            if d.empty or ep not in P.columns:
                continue
            p = sub[C.ID_COL].map(P[ep]).to_numpy(float)
            err = np.abs(p - sub["y"].to_numpy(float))
            inv = np.zeros(len(sub), bool)
            inv[list(set(d.loc[d.cliff, "i"]) | set(d.loc[d.cliff, "j"]))] = True
            ok = np.isfinite(err)
            a, b = err[ok & ~inv].mean(), err[ok & inv].mean()
            print(f"    {iso:8s} {a:10.4f} {b:8.4f} {b/a:7.2f}x {int((ok&inv).sum()):8d}")
        print()

    print("=" * 78)
    print("3. DO WE PREDICT THE CLIFF, OR SMOOTH IT?\n")
    print("   For each cliff pair: measured |dy| against predicted |dy|.")
    print("   'captured' = predicted gap at least half the measured one.\n")
    for nm, P in preds.items():
        print(f"  {nm}")
        print(f"    {'endpoint':8s} {'pairs':>6s} {'measured |dy|':>14s} "
              f"{'predicted |dy|':>15s} {'kept':>6s} {'r(dy)':>7s} {'captured':>9s}")
        for iso in C.ISOFORMS:
            if iso not in store:
                continue
            sub, d = store[iso]
            ep = f"{iso}_pIC50_direct_inhibition"
            if d.empty or ep not in P.columns:
                continue
            cl = d[d.cliff]
            if len(cl) < 10:
                continue
            p = sub[C.ID_COL].map(P[ep]).to_numpy(float)
            yv = sub["y"].to_numpy(float)
            dy_true = np.abs(yv[cl["i"].to_numpy()] - yv[cl["j"].to_numpy()])
            dy_pred = np.abs(p[cl["i"].to_numpy()] - p[cl["j"].to_numpy()])
            ok = np.isfinite(dy_pred)
            r = np.corrcoef(dy_true[ok], dy_pred[ok])[0, 1] if ok.sum() > 5 else np.nan
            print(f"    {iso:8s} {int(ok.sum()):6d} {dy_true[ok].mean():14.3f} "
                  f"{dy_pred[ok].mean():15.3f} "
                  f"{dy_pred[ok].mean()/dy_true[ok].mean():6.2f} {r:7.3f} "
                  f"{(dy_pred[ok] >= 0.5*dy_true[ok]).mean()*100:8.1f}%")
        print()

    print("HOW TO READ SECTION 3. 'kept' is the fraction of the measured gap the model\n"
          "reproduces. Near 1.0 with a positive r means cliffs are being learned and the\n"
          "remaining error is ordinary noise. Far below 1.0 means the model is SMOOTHING\n"
          "-- it puts both members of the pair near their local average -- and that is a\n"
          "loss-function and architecture problem, not a data problem, so more compounds\n"
          "will not fix it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
