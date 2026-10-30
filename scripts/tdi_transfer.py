#!/usr/bin/env python
"""Which members keep working on chemistry they have not seen? The OOF column cannot say.

    python scripts/tdi_transfer.py --tag tdiK
    python scripts/tdi_transfer.py --tag tdiK --extra artifacts/oof_tdi_chemprop.csv

WHY THIS EXISTS -- WRITTEN AFTER GETTING IT WRONG. `tdi_select.py` chose members by
out-of-fold AUC under honest nested selection, and the result split cleanly:

    endpoint   OOF AUC          predicted blind d'   REALISED blind d'
    CYP3A4     0.817 -> 0.843   1.290 -> 1.427       1.290 -> 1.135   WORSE
    CYP2D6     0.621 -> 0.645   0.980 -> 1.183       0.980 -> 1.213   BETTER

CYP2D6 landed almost exactly on prediction. CYP3A4 moved the opposite way to a rise in
out-of-fold AUC. Nested selection was not the flaw -- nothing was scored on data that
chose it. The flaw is that OOF AUC and blind AUC are measured on DIFFERENT CHEMISTRY, and
selection maximised the first.

Every out-of-fold compound has near neighbours in its own training half; the blind set
sits at median max-Tanimoto 0.587 to the training set, which is the FAR end of our own
distribution. A high-capacity learner interpolates beautifully between near neighbours
and degrades off-manifold. A similarity learner is doing the same thing near and far, only
worse near. Rank the members on the near population -- which is what an OOF AUC mostly
is -- and you systematically prefer the ones that will fall off hardest.

WHAT THIS MEASURES. Each compound's max Tanimoto to its OWN fold's training half. Split
the OOF pool at the median into NEAR and FAR, and score every member separately in each.
Then report:

    transfer = AUC(far) - AUC(near)

A member with a big positive OOF AUC and a big negative transfer is exactly the trap.
The FAR column is the closer analogue of the blind task, so selection should be run on
it -- which is what `--rank-by far` prints at the end.

WHAT IT CANNOT DO. FAR is still our own chemistry, just the thinner part of it; it is a
better proxy for the blind set than the pooled OOF, not a substitute for the leaderboard.
Treat a large negative transfer as a reason to distrust a member, and a large positive
OOF AUC that survives into FAR as the only kind worth selecting on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from sklearn.metrics import roc_auc_score

from cypkit import config as C

RDLogger.DisableLog("rdApp.*")


def _gen():
    try:
        from rdkit.Chem import rdFingerprintGenerator as rdfp  # noqa: PLC0415
        return rdfp.GetMorganGenerator(radius=2, fpSize=2048)
    except (ImportError, AttributeError):
        from rdkit.Chem import AllChem  # noqa: PLC0415
        return AllChem.GetMorganGenerator(radius=2, fpSize=2048)


GEN = _gen()


def fingerprints(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        out.append(GEN.GetFingerprint(m) if m is not None else None)
    return out


def dist_to_own_train(fps, fold):
    """Max Tanimoto from each compound to the training half of ITS fold."""
    d = np.full(len(fps), np.nan)
    for k in sorted(set(fold)):
        te = np.where((fold == k) & np.array([f is not None for f in fps]))[0]
        tr = [fps[i] for i in np.where((fold != k))[0] if fps[i] is not None]
        if not tr:
            continue
        for i in te:
            d[i] = max(DataStructs.BulkTanimotoSimilarity(fps[i], tr))
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="tdiK")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--extra-name", nargs="*", default=[])
    ap.add_argument("--folds", default=None)
    ap.add_argument("--smiles", default=None, help="default: the TDI training file")
    args = ap.parse_args()

    names_x = list(args.extra_name)
    while len(names_x) < len(args.extra):
        names_x.append(Path(args.extra[len(names_x)]).stem.replace("oof_", ""))

    folds = pd.read_csv(args.folds or C.artifact("folds_tdi.csv"))
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))

    src = pd.read_csv(args.smiles or (C.DATA_DIR / C.DATA_FILES["train_tdi"]))
    src.columns = [c.strip() for c in src.columns]
    smi_of = dict(zip(src[C.ID_COL], src[C.SMILES_COL]))

    for iso in C.TDI_ISOFORMS:
        p = C.artifact(f"tdi_member_oof_{iso}_{args.tag}.csv")
        if not Path(p).exists():
            continue
        t = pd.read_csv(p)
        for f, nm in zip(args.extra, names_x):
            d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]
            if f"{iso}_score" in d.columns:
                t = t.merge(d[[C.ID_COL, f"{iso}_score"]].rename(
                    columns={f"{iso}_score": nm}), on=C.ID_COL, how="left")

        members = [c for c in t.columns
                   if c not in (C.ID_COL, "y") and not t[c].isna().any()]
        y = t["y"].to_numpy().astype(int)
        fold = t[C.ID_COL].map(fold_of).to_numpy()
        fps = fingerprints([smi_of.get(i) for i in t[C.ID_COL]])
        d = dist_to_own_train(fps, fold)
        cut = np.nanmedian(d)
        near, far = (d >= cut), (d < cut)

        print("=" * 78)
        print(f"{iso}   n={len(y)}   median max-Tanimoto to own training half {cut:.3f}")
        print(f"  NEAR (sim >= {cut:.3f}) n={near.sum()}, {y[near].mean()*100:.1f}% pos   "
              f"FAR (sim < {cut:.3f}) n={far.sum()}, {y[far].mean()*100:.1f}% pos")
        print(f"  the blind set sits at median 0.587 to the FULL training set -- "
              f"nearer the FAR column\n")
        print(f"  {'member':16s} {'pooled OOF':>11s} {'NEAR':>8s} {'FAR':>8s} "
              f"{'transfer':>9s}")
        rows = []
        for m in members:
            s = t[m].to_numpy(float)
            a_all = roc_auc_score(y, s)
            a_n = roc_auc_score(y[near], s[near]) if len(set(y[near])) > 1 else np.nan
            a_f = roc_auc_score(y[far], s[far]) if len(set(y[far])) > 1 else np.nan
            rows.append((m, a_all, a_n, a_f, a_f - a_n))
        for m, a, n, f, tr in sorted(rows, key=lambda r: -r[1]):
            flag = "   <-- trap" if (a > np.median([r[1] for r in rows]) and tr < -0.03) \
                   else ""
            print(f"  {m:16s} {a:11.4f} {n:8.4f} {f:8.4f} {tr:+9.4f}{flag}")

        best_pool = sorted(rows, key=lambda r: -r[1])[:5]
        best_far = sorted(rows, key=lambda r: -r[3])[:5]
        print(f"\n  top-5 by pooled OOF : {[m for m, *_ in best_pool]}")
        print(f"  top-5 by FAR AUC    : {[m for m, *_ in best_far]}")
        ov = len({m for m, *_ in best_pool} & {m for m, *_ in best_far})
        print(f"  overlap {ov}/5 -- the two rankings {'agree' if ov >= 4 else 'DISAGREE'}"
              f", and only the FAR one resembles the blind task.\n")
    print("READ IT LIKE THIS. A member flagged 'trap' scores well on the pooled OOF and "
          "loses\nmore than 0.03 AUC once its near neighbours are taken away. Selecting on "
          "the pooled\ncolumn concentrates those; selecting on FAR does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
