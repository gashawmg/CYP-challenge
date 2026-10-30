#!/usr/bin/env python
"""Choose WHICH members go in the TDI blend, and prove the choice is worth making.

    python scripts/tdi_select.py --tag tdiK
    python scripts/tdi_select.py --tag tdiK --extra artifacts/oof_tdi_chemprop.csv
    python scripts/tdi_select.py --tag tdiK --extra artifacts/oof_tdi_chemprop.csv \
        --write-oof artifacts/oof_tdi_greedy.csv

WHY THIS EXISTS. Every TDI submission so far has used an EQUAL-weight rank blend of every
member in the pool. The out-of-fold AUC chart says that is a mistake, and says it in the
bluntest way available: on CYP2D6 the 19-member blend scores 0.620 while two of its own
members score higher on their own -- knn_fcfp 0.623 and svc_tan 0.622. A blend that loses
to a member it contains is not aggregating, it is diluting. The many tree learners on that
endpoint sit at 0.562-0.594 and each one drags the mean down.

The obvious fix -- keep the members that scored best -- is the classic way to fool
yourself, because "best" was measured on the same out-of-fold pool you would then score.
So the comparison here is NESTED: the member subset is chosen on four folds and scored on
the fifth, rotated, so no strategy is ever graded on data that shaped it. That costs some
absolute AUC (each held-out fifth is small and harder than the full pool), which is why
the numbers below must be read as a RANKING OF STRATEGIES and not as a forecast. The
forecast comes from `scripts/tdi_validator.py`, which is fed the winning strategy's
full-data blend.

THE STRATEGIES COMPARED.

  equal            every member, equal weight -- what we have always submitted
  top-k            the k members with the best AUC on the training folds
  greedy forward   start empty; repeatedly add the member that most improves the
                   blend's AUC on the training folds; stop when nothing improves it

Greedy forward is the interesting one because it selects for CONTRIBUTION rather than for
individual quality: a mediocre member that is wrong in a different place than the rest
earns its seat, and the third near-copy of an already-included member does not.

WHAT COMES OUT. The per-endpoint winning strategy, the members it picked on the full pool,
and how often each member survived selection across the nested folds -- which is the only
honest read on whether a pick is real or an artefact of one split. With --write-oof the
selected blend is written in the shape `tdi_validator.py` reads, so the two scripts chain.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cypkit import config as C


def rank01(x):
    r = pd.Series(x).rank().to_numpy()
    return (r - r.min()) / max(r.max() - r.min(), 1e-9)


def blend(ranks, members):
    return np.mean([ranks[m] for m in members], axis=0)


def safe_auc(y, s):
    return roc_auc_score(y, s) if y.min() != y.max() else np.nan


# --------------------------------------------------------------------------- strategies
# Each takes (ranks, y, train_mask, names) and returns the member list to use.

def sel_equal(ranks, y, tr, names):
    return list(names)


def sel_topk(k):
    def f(ranks, y, tr, names):
        sc = {m: safe_auc(y[tr], ranks[m][tr]) for m in names}
        return sorted(names, key=lambda m: -sc[m])[:k]
    f.__name__ = f"top-{k}"
    return f


def sel_greedy(min_gain=0.0):
    def f(ranks, y, tr, names):
        chosen, best = [], -np.inf
        pool = list(names)
        while pool:
            cand = [(safe_auc(y[tr], blend(ranks, chosen + [m])[tr]), m) for m in pool]
            cand = [(a, m) for a, m in cand if np.isfinite(a)]
            if not cand:
                break
            a, m = max(cand)
            if a <= best + min_gain:
                break
            chosen.append(m); pool.remove(m); best = a
        return chosen or [max(names, key=lambda m: safe_auc(y[tr], ranks[m][tr]))]
    f.__name__ = "greedy"
    return f


def sel_consensus(min_gain=0.0, share=0.5):
    """Greedy is run once per INNER fold of the training half; a member is kept only if
    it survives in at least `share` of them.

    This exists because greedy's own stability table can contradict its full-pool pick --
    on CYP2D6 the full-pool run chose svc_tan, which had survived only 1 of 5 nested
    folds, over svc_c10, which had survived 4. A single greedy path is a chain of
    argmaxes over noisy AUC differences, so it is free to take a different route on a
    slightly different sample. Consensus asks each route the same question and keeps only
    what they agree on, which is a strictly weaker claim and therefore a safer one."""
    g = sel_greedy(min_gain)

    def f(ranks, y, tr, names):
        idx = np.where(tr)[0]
        parts = np.array_split(idx, 5)
        votes = Counter()
        for p in parts:
            inner = np.zeros(len(y), bool)
            inner[idx] = True
            inner[p] = False                      # hold this inner slice out
            votes.update(g(ranks, y, inner, names))
        keep = [m for m in names if votes[m] >= share * len(parts)]
        return keep or g(ranks, y, tr, names)
    f.__name__ = "consensus"
    return f


def load_pool(tag, extras, extra_names):
    """{isoform: (frame with Molecule_Name/y/<members>)}"""
    tables = {}
    for iso in C.TDI_ISOFORMS:
        p = C.artifact(f"tdi_member_oof_{iso}_{tag}.csv")
        if Path(p).exists():
            tables[iso] = pd.read_csv(p)
    for f, nm in zip(extras, extra_names):
        d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]
        for iso in C.TDI_ISOFORMS:
            col = f"{iso}_score"
            if iso in tables and col in d.columns:
                add = d[[C.ID_COL, col]].rename(columns={col: nm}).dropna()
                tables[iso] = tables[iso].merge(add, on=C.ID_COL, how="left")
    return tables


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="tdiK")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra OOF files with <ISO>_score columns")
    ap.add_argument("--extra-name", nargs="*", default=[])
    ap.add_argument("--folds", default=None, help="default artifacts/folds_tdi.csv")
    ap.add_argument("--write-oof", default=None)
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="greedy stops when the best addition gains less than this")
    args = ap.parse_args()

    extra_names = list(args.extra_name)
    while len(extra_names) < len(args.extra):
        extra_names.append(Path(args.extra[len(extra_names)]).stem.replace("oof_", ""))

    tables = load_pool(args.tag, args.extra, extra_names)
    if not tables:
        raise SystemExit(f"no member files for tag {args.tag}")
    folds = pd.read_csv(args.folds or C.artifact("folds_tdi.csv"))
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))

    STRATS = [sel_equal, sel_topk(3), sel_topk(5), sel_greedy(args.min_gain),
              sel_consensus(args.min_gain)]

    print(f"pool: tag {args.tag}" + (f" + {', '.join(extra_names)}" if extra_names else ""))
    print("nested: subset chosen on 4 folds, AUC measured on the 5th, rotated.\n")

    full_choice, full_auc = {}, {}
    for iso, t in tables.items():
        names = [c for c in t.columns if c not in (C.ID_COL, "y")]
        # a member missing on some rows cannot be rank-blended row-wise; drop it rather
        # than silently imputing a rank that would flatter it
        drop = [m for m in names if t[m].isna().any()]
        if drop:
            print(f"  {iso}: dropping members with missing rows: {drop}")
            names = [m for m in names if m not in drop]
        y = t["y"].to_numpy().astype(int)
        ranks = {m: rank01(t[m].to_numpy(float)) for m in names}
        fold = t[C.ID_COL].map(fold_of).to_numpy()
        ks = sorted({f for f in fold if np.isfinite(f)})

        print("=" * 78)
        print(f"{iso}   n={len(y)}  positives {y.mean()*100:.1f}%  "
              f"members {len(names)}\n")

        picks = {s.__name__: Counter() for s in STRATS}
        held = {s.__name__: [] for s in STRATS}
        sizes = {s.__name__: [] for s in STRATS}
        for k in ks:
            te = fold == k
            tr = ~te
            for s in STRATS:
                ms = s(ranks, y, tr, names)
                a = safe_auc(y[te], blend(ranks, ms)[te])
                if np.isfinite(a):
                    held[s.__name__].append(a)
                    sizes[s.__name__].append(len(ms))
                    picks[s.__name__].update(ms)

        print(f"  {'strategy':18s} {'held-out AUC':>13s} {'sd':>7s} {'members':>9s}")
        for s in STRATS:
            n = s.__name__
            print(f"  {n:18s} {np.mean(held[n]):13.4f} {np.std(held[n]):7.4f} "
                  f"{np.mean(sizes[n]):9.1f}")
        winner = max(STRATS, key=lambda s: np.mean(held[s.__name__]))
        gain = np.mean(held[winner.__name__]) - np.mean(held["sel_equal"
                                                            if False else
                                                            sel_equal.__name__])
        print(f"\n  winner: {winner.__name__}  "
              f"{np.mean(held[winner.__name__]):.4f}  ({gain:+.4f} vs equal)")

        st = picks[winner.__name__]
        if winner.__name__ != sel_equal.__name__:
            print(f"\n  selection stability -- folds (of {len(ks)}) each member survived:")
            for m, c in st.most_common():
                bar = "#" * c
                print(f"    {m:16s} {c}/{len(ks)}  {bar}")

        # ---- refit the winning STRATEGY on the whole pool: this is what we would submit
        ms = winner(ranks, y, np.ones(len(y), bool), names)
        full_choice[iso] = ms
        full_auc[iso] = safe_auc(y, blend(ranks, ms))
        eq = safe_auc(y, blend(ranks, names))
        print(f"\n  on the FULL pool: {winner.__name__} picks {len(ms)} -> AUC "
              f"{full_auc[iso]:.4f}   (equal blend {eq:.4f})")
        print(f"    {ms}")
        print("    Full-pool AUC is optimistic for the SELECTION; the nested block above "
              "is the\n    honest comparison. Use these members, trust that gain.\n")

    if args.write_oof:
        parts = []
        for iso, t in tables.items():
            names = full_choice[iso]
            ranks = {m: rank01(t[m].to_numpy(float)) for m in names}
            parts.append(pd.DataFrame({
                C.ID_COL: t[C.ID_COL],
                f"{iso}_y": t["y"].to_numpy().astype(int),
                f"{iso}_score": blend(ranks, names)}))
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=C.ID_COL, how="outer")
        out.to_csv(args.write_oof, index=False)
        print(f"wrote {args.write_oof}  ({len(out)} rows)")
        print("  next:  python scripts/tdi_validator.py --oof " + args.write_oof)

    print("\nMEMBER LISTS TO CARRY FORWARD")
    for iso, ms in full_choice.items():
        print(f"  {iso}: {' '.join(ms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
