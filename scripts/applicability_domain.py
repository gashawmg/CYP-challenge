#!/usr/bin/env python
"""Is the external panel the same chemistry as ours, and does distance predict error?

    python scripts/applicability_domain.py
    python scripts/applicability_domain.py --oof artifacts/oof_pseudo_blend45.csv

WHY. `cypkit/external_prior.py` asserts in one line that the blind set sits at median
max-Tanimoto 0.587 to the challenge training set but only 0.368 to the external panel,
and that adding 16k external molecules moves the blind set's nearest-neighbour similarity
from 0.587 to 0.589. That single number was used to justify treating external data as
knowledge transfer rather than coverage. It is the right instinct but the wrong test,
because an average can hide the case that matters: external data could be useless for the
750 blind compounds we ALREADY cover well and still be the only thing standing near the
ones we cover badly. Averages do not answer that; conditioning on our own distance does.

WHAT THIS ASKS, IN ORDER.

  1. Are they the same kind of molecule at all? Property space (MW, cLogP, TPSA, rings,
     rotatable bonds) and Murcko scaffold overlap. If the two libraries do not even
     overlap in property space, similarity numbers are beside the point.

  2. How far is the blind set from each source? Full distribution of max-Tanimoto, not
     just the median, plus the fraction of blind compounds with a genuine near neighbour
     (>= 0.4, 0.5, 0.7) in each.

  3. THE QUESTION THAT DECIDES IT. Bucket the blind compounds by how far they sit from
     OUR training set. Inside each bucket, how many get a closer neighbour from the
     external panel than they had from us? A source that rescues the orphan bucket is
     worth having even if it changes nothing on average.

  4. Does distance predict error at all? Out-of-fold, each compound's distance to its own
     fold's training set is known and so is its error. If error is flat in distance, the
     applicability domain is not the binding constraint and no amount of coverage helps.
     That is the control this analysis needs -- without it, step 3 measures something that
     may not matter.

Fingerprints are ECFP4 (Morgan radius 2, 2048 bits), the standard for this comparison.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold

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


def fps(smiles):
    out, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        out.append(GEN.GetFingerprint(m))
        keep.append(i)
    return out, np.array(keep)


def max_sim(query_fps, ref_fps, chunk=2000):
    """Max Tanimoto from each query to any reference."""
    best = np.zeros(len(query_fps))
    for i, q in enumerate(query_fps):
        b = 0.0
        for j in range(0, len(ref_fps), chunk):
            s = DataStructs.BulkTanimotoSimilarity(q, ref_fps[j:j + chunk])
            m = max(s) if s else 0.0
            if m > b:
                b = m
        best[i] = b
    return best


def props(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        rows.append((Descriptors.MolWt(m), Crippen.MolLogP(m),
                     Descriptors.TPSA(m), Descriptors.RingCount(m),
                     Descriptors.NumRotatableBonds(m)))
    return pd.DataFrame(rows, columns=["MW", "cLogP", "TPSA", "rings", "rotB"])


def scaffolds(smiles):
    out = set()
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            out.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--external", default="artifacts/external_cyp.csv")
    ap.add_argument("--test-csv", default="data/cyp-challenge-TEST-BLINDED.csv")
    ap.add_argument("--folds", default="artifacts/folds.csv")
    ap.add_argument("--labels", default="artifacts/labels.csv")
    ap.add_argument("--oof", default="artifacts/oof_pseudo_blend45.csv")
    args = ap.parse_args()

    folds = pd.read_csv(args.folds)
    test = pd.read_csv(args.test_csv); test.columns = [c.strip() for c in test.columns]
    ext = pd.read_csv(args.external)
    ext_smi = ext["smiles_std"].fillna(ext["SMILES"]).tolist()

    print("building fingerprints ...")
    f_ch, k_ch = fps(folds["SMILES"].tolist())
    f_bl, k_bl = fps(test[C.SMILES_COL].tolist())
    f_ex, _ = fps(ext_smi)
    print(f"  challenge {len(f_ch)}   blind {len(f_bl)}   external {len(f_ex)}\n")

    # ---------------------------------------------------------------- 1. property space
    print("=" * 74)
    print("1. ARE THEY THE SAME KIND OF MOLECULE?\n")
    pc, pe, pb = (props(folds["SMILES"]), props(ext_smi), props(test[C.SMILES_COL]))
    print(f"  {'':10s} " + " ".join(f"{c:>14s}" for c in pc.columns))
    for nm, d in (("challenge", pc), ("external", pe), ("blind", pb)):
        print(f"  {nm:10s} " + " ".join(
            f"{d[c].mean():7.1f}+-{d[c].std():5.1f}" for c in pc.columns))
    sc_ch, sc_ex, sc_bl = (scaffolds(folds["SMILES"]), scaffolds(ext_smi),
                           scaffolds(test[C.SMILES_COL]))
    print(f"\n  Murcko scaffolds: challenge {len(sc_ch)}, external {len(sc_ex)}, "
          f"blind {len(sc_bl)}")
    print(f"  blind scaffolds also in challenge: {len(sc_bl & sc_ch):4d} "
          f"({len(sc_bl & sc_ch)/len(sc_bl)*100:.1f}%)")
    print(f"  blind scaffolds also in external : {len(sc_bl & sc_ex):4d} "
          f"({len(sc_bl & sc_ex)/len(sc_bl)*100:.1f}%)")
    print(f"  blind scaffolds in NEITHER       : "
          f"{len(sc_bl - sc_ch - sc_ex):4d} "
          f"({len(sc_bl - sc_ch - sc_ex)/len(sc_bl)*100:.1f}%)")
    print(f"  challenge/external scaffold overlap: {len(sc_ch & sc_ex)}")

    # ---------------------------------------------------------------- 2. distance
    print("\n" + "=" * 74)
    print("2. HOW FAR IS THE BLIND SET FROM EACH SOURCE?\n")
    d_ch = max_sim(f_bl, f_ch)
    d_ex = max_sim(f_bl, f_ex)
    print(f"  {'source':12s} {'p10':>6s} {'p25':>6s} {'median':>7s} {'p75':>6s} "
          f"{'p90':>6s}   {'>=0.4':>6s} {'>=0.5':>6s} {'>=0.7':>6s}")
    for nm, d in (("challenge", d_ch), ("external", d_ex),
                  ("union", np.maximum(d_ch, d_ex))):
        q = np.percentile(d, [10, 25, 50, 75, 90])
        print(f"  {nm:12s} " + " ".join(f"{v:6.3f}" for v in q) +
              f"   {(d>=0.4).mean()*100:5.1f}% {(d>=0.5).mean()*100:5.1f}% "
              f"{(d>=0.7).mean()*100:5.1f}%")

    # ------------------------------------------------------ 3. does it rescue the tail?
    print("\n" + "=" * 74)
    print("3. DOES EXTERNAL DATA HELP WHERE WE ARE WEAKEST?\n")
    print("   Blind compounds bucketed by distance to OUR training set.\n")
    edges = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]
    print(f"  {'our max-sim':>14s} {'n':>5s} {'ext max-sim':>12s} "
          f"{'ext closer':>11s} {'gain if closer':>15s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d_ch >= lo) & (d_ch < hi)
        if sel.sum() == 0:
            continue
        closer = (d_ex[sel] > d_ch[sel])
        gain = (d_ex[sel][closer] - d_ch[sel][closer]).mean() if closer.any() else 0.0
        print(f"  [{lo:.2f}, {hi:.2f}) {sel.sum():7d} {d_ex[sel].mean():12.3f} "
              f"{closer.mean()*100:10.1f}% {gain:15.3f}")
    print("\n  'ext closer' is the fraction of that bucket whose nearest external molecule\n"
          "  beats its nearest challenge molecule. A source that extends coverage shows a\n"
          "  HIGH percentage in the LOW buckets -- that is what rescuing the tail means.")

    # ------------------------------------------- 4. does distance predict error at all?
    print("\n" + "=" * 74)
    print("4. CONTROL -- DOES DISTANCE PREDICT OUR ERROR?\n")
    if not (Path(args.oof).exists() and Path(args.labels).exists()):
        print("  (need --oof and --labels; skipped)")
        return 0
    lab = pd.read_csv(args.labels)
    lab = lab[lab["kind"] != "left"]
    piv = lab.pivot_table(index=C.ID_COL, columns="isoform", values="y", aggfunc="mean")
    oof = pd.read_csv(args.oof); oof.columns = [c.strip() for c in oof.columns]

    sub = folds.iloc[k_ch].reset_index(drop=True)
    fold_of = sub["fold"].to_numpy()
    d_self = np.zeros(len(f_ch))
    for k in sorted(set(fold_of)):
        q = [f_ch[i] for i in np.where(fold_of == k)[0]]
        r = [f_ch[i] for i in np.where(fold_of != k)[0]]
        d_self[fold_of == k] = max_sim(q, r)
    sub["d_train"] = d_self

    print(f"  {'d(train)':>12s} {'n':>6s} " +
          " ".join(f"{i:>8s}" for i in C.ISOFORMS))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sub["d_train"] >= lo) & (sub["d_train"] < hi)
        if sel.sum() < 20:
            continue
        ids = set(sub.loc[sel, C.ID_COL])
        cells = []
        for iso in C.ISOFORMS:
            ep = f"{iso}_pIC50_direct_inhibition"
            if iso not in piv.columns or ep not in oof.columns:
                cells.append("     -  "); continue
            y = oof[C.ID_COL].map(piv[iso]).to_numpy(float)
            p = oof[ep].to_numpy(float)
            m = oof[C.ID_COL].isin(ids).to_numpy() & np.isfinite(y) & np.isfinite(p)
            cells.append(f"{np.mean(np.abs(y[m]-p[m])):8.3f}" if m.sum() >= 15
                         else f"{'n<15':>8s}")
        print(f"  [{lo:.2f}, {hi:.2f}) {int(sel.sum()):6d} " + " ".join(cells))
    print("\n  Out-of-fold MAE by distance to that fold's own training set. If these\n"
          "  columns are FLAT, our error is not driven by coverage and no external\n"
          "  library can fix it. If they rise steeply, coverage is the constraint and\n"
          "  the right source is one that fills the low buckets in section 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
