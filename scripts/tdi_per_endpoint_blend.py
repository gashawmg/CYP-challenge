#!/usr/bin/env python
r"""The rank blend is fitted once and used on both endpoints. The member table says that
is the wrong shape: CYP3A4 and CYP2D6 rank the members almost in opposite orders, so a
single global blend has to compromise between them.

This measures the size of that compromise. For each endpoint separately it runs greedy
forward selection with replacement on rank-averaged members, choosing by AUC, with the
selection itself done INSIDE a cross-validation loop so the reported number is not the
maximum of a search over the rows it is scored on.

    python scripts/tdi_per_endpoint_blend.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np, pandas as pd

# artifacts/ beside the repo root, which is this file's parent's parent. Resolved from
# __file__ rather than the working directory so the script runs from anywhere, and
# overridable with --artifacts.
ART = Path(__file__).resolve().parent.parent / "artifacts"


def auc(y, s):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return 0.5
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(len(s))
    return (r[y].mean() - (y.sum() - 1) / 2) / (~y).sum()


def mcc(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, bool)
    tp = int((y & p).sum()); tn = int((~y & ~p).sum())
    fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    d = float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn)
    return 0.0 if d == 0 else (tp*tn - fp*fn)/np.sqrt(d)


def pct(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def greedy(R, y, names, rows, n_steps=8):
    """Forward selection with replacement, by AUC, on the given rows only."""
    chosen, cur, best_auc = [], np.zeros(len(y)), 0.0
    for _ in range(n_steps):
        cand = [(auc(y[rows], ((cur + R[:, j]) / (len(chosen) + 1))[rows]), j)
                for j in range(R.shape[1])]
        a, j = max(cand)
        if a <= best_auc + 1e-5:
            break
        best_auc = a; chosen.append(j); cur = cur + R[:, j]
    return chosen, best_auc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tdiM")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--artifacts", type=Path, default=ART,
                    help=f"directory holding tdi_member_oof_<iso>_<tag>.csv "
                         f"(default {ART})")
    a = ap.parse_args(argv)

    missing = [f"tdi_member_oof_{i}_{a.tag}.csv" for i in ("CYP3A4", "CYP2D6")
               if not (a.artifacts / f"tdi_member_oof_{i}_{a.tag}.csv").exists()]
    if missing:
        have = sorted(p.name for p in a.artifacts.glob("tdi_member_oof_*.csv"))
        raise SystemExit(
            f"not found in {a.artifacts}: {', '.join(missing)}\n"
            f"  member OOF files present: {', '.join(have) or 'none'}\n"
            f"  these are written by run_tdi_ensemble.py --save-oof; run that first "
            f"with the same --tag.")

    for iso, rate in [("CYP3A4", 0.2915), ("CYP2D6", 0.0706)]:
        d = pd.read_csv(a.artifacts / f"tdi_member_oof_{iso}_{a.tag}.csv")
        y = d["y"].to_numpy().astype(bool)
        names = [c for c in d.columns if c not in ("Molecule_Name", "y")]
        R = np.column_stack([pct(d[c].to_numpy(float)) for c in names])
        blend_all = R.mean(1)
        print(f"\n=== {iso}   n {len(d)}   positives {y.sum()} ({y.mean():.4f})"
              f"   blind rate {rate:.4f}")
        print(f"  equal-weight blend of all {len(names)}: AUC {auc(y, blend_all):.4f}")
        singles = sorted(((auc(y, R[:, j]), names[j]) for j in range(len(names))),
                         reverse=True)
        print("  best singles: " + ", ".join(f"{n} {v:.4f}" for v, n in singles[:4]))

        # honest: select the blend on 4 folds, score it on the 5th
        rng = np.random.default_rng(a.seed)
        fold = rng.permutation(np.arange(len(d)) % a.folds)
        oof = np.zeros(len(d)); picks = []
        for f in range(a.folds):
            tr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
            ch, _ = greedy(R, y, names, tr)
            picks.append([names[j] for j in ch])
            oof[va] = R[np.ix_(va, ch)].mean(1)
        a_sel = auc(y, oof)
        print(f"  per-endpoint greedy blend, selection held out: AUC {a_sel:.4f}"
              f"   ({a_sel - auc(y, blend_all):+.4f} vs the equal-weight blend)")
        from collections import Counter
        c = Counter(n for p in picks for n in p)
        print("  members chosen (of 5 folds): "
              + ", ".join(f"{n} x{k}" for n, k in c.most_common(6)))

        # MCC at the rate the submission actually uses, cut taken out of fold
        for nm, s in [("equal-weight blend", blend_all), ("per-endpoint greedy", oof)]:
            pred = np.zeros(len(d), bool)
            for f in range(a.folds):
                tr = fold != f; va = fold == f
                cut = np.quantile(s[tr], 1 - rate)
                pred[va] = s[va] >= cut
            print(f"  MCC at blind rate {rate:.3f}, cut out of fold: "
                  f"{nm:20s} {mcc(y, pred):.4f}")

        # DISAGREEMENT, which is the only thing a rank blend can turn into score.
        # A member with a lower AUC than the leader still pays if it is wrong about
        # different compounds; a member that merely agrees cannot pay at all.
        best_j = int(np.argmax([auc(y, R[:, j]) for j in range(len(names))]))
        C = np.corrcoef(R.T)
        rows = sorted((C[best_j, j], auc(y, R[:, j]), names[j])
                      for j in range(len(names)) if j != best_j)
        print(f"  least correlated with the best single ({names[best_j]}, "
              f"AUC {auc(y, R[:, best_j]):.4f}):")
        print(f"    {'member':14s}{'corr':>8}{'AUC':>9}")
        for c_, a_, n_ in rows[:6]:
            print(f"    {n_:14s}{c_:>8.3f}{a_:>9.4f}")
        print(f"    median corr across all members: "
              f"{np.median([C[best_j, j] for j in range(len(names)) if j != best_j]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
