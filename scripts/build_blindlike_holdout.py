#!/usr/bin/env python
r"""Build a held-out validation set that is as EASY as the blind set, so its score reads
as a direct forecast of the board.

    python scripts\build_blindlike_holdout.py --n 400
    python scripts\build_blindlike_holdout.py --n 400 --min-sep 0.55 --seed 0

THE IDEA, AND THE TRAP IN ITS OBVIOUS FORM. Hold out the training compounds most similar
to the blind set, and their score should predict blind performance. That is right in
spirit and WRONG AS STATED, in a way that makes the instrument read backwards.

A blind compound B is easy because its nearest training neighbour T sits at, say, Tanimoto
0.70 and the model has learned from T. If we put T in the holdout, T is no longer in
training -- so when we score T, its own nearest available neighbour might be 0.50. We have
graded T as a HARDER compound than B will ever be, by removing exactly the neighbour that
made it easy. Selecting on "similar to the blind set" strips the blind set's advantage.
This is the Section 16 defect with the sign flipped: a holdout that is too hard, again.

WHAT ACTUALLY DETERMINES DIFFICULTY is not similarity to the blind set. It is each
compound's nearest-neighbour similarity TO THE DATA THE MODEL WAS TRAINED ON. So the
holdout must satisfy:

    NN(holdout member -> remaining training set)  ~  NN(blind compound -> full training set)

as DISTRIBUTIONS. That is the quantity to match, and it is measurable both sides.

THE COUPLING, AND THE DISPERSION RULE THAT DEFUSES IT. Removing compounds changes the
remaining neighbours of the others, so the constraint is self-referential: hold out two
close analogues together and each destroys the other's nearest neighbour. The fix is to
require the holdout to be INTERNALLY DISPERSED -- no two members above `--min-sep`. Then
removing them barely perturbs each other's NN, and selection becomes a simple matching
problem again. The script VERIFIES this after the fact rather than assuming it: it
recomputes every member's true NN against the final training set and reports the match.

WHY THIS BEATS THE IMPORTANCE REWEIGHTING IT REPLACES (Section 30, Section 32). The
reweighted bootstrap matches the same distribution, but at CYP2C9 its effective sample
size collapsed to 121 of 1,026 rows -- a large variance cost, and it can only rescore
existing out-of-fold predictions. A real holdout gives genuine held-out predictions, can
score ANY model including chemprop, and is a fixed set every experiment can share. It is
what the PXR pipeline did with set1_holdout_30_stratified.csv.

STRATIFIED on the label so the holdout spans the activity range rather than piling up
wherever the similarity happens to be convenient.

OUTPUT artifacts/holdout_blindlike.csv -- Molecule_Name, SMILES, nn_to_remaining_train --
plus a printed verification table. Nothing is trained or scored here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
EDGES = np.array([0.0, .35, .40, .45, .50, .55, .60, .65, .70, .80, 1.01])


def fps(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            out.append(gen.GetFingerprint(m)); keep.append(i)
    return out, np.asarray(keep, int)


def hist(x):
    h = np.bincount(np.clip(np.digitize(x, EDGES) - 1, 0, len(EDGES) - 2),
                    minlength=len(EDGES) - 1).astype(float)
    return h / max(h.sum(), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="holdout size")
    ap.add_argument("--min-sep", type=float, default=0.55,
                    help="no two holdout members may exceed this Tanimoto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tr = pd.read_csv(ROOT / "data" / "cyp-challenge-TRAIN_inhibition.csv")
    te = pd.read_csv(ROOT / "data" / "cyp-challenge-TEST-BLINDED.csv")
    for d in (tr, te):
        d.columns = [c.strip() for c in d.columns]
    sc = next(c for c in tr.columns if c.lower() == "smiles")
    idc = next(c for c in tr.columns if c.lower() in ("molecule_name", "name", "id"))

    f_tr, k_tr = fps(tr[sc].tolist()); tr = tr.iloc[k_tr].reset_index(drop=True)
    f_te, _ = fps(te[sc].tolist())
    n = len(f_tr)
    print(f"  {n} training compounds, {len(f_te)} blind\n")

    # TARGET: what a blind compound sees, against the FULL training set
    nn_blind = np.array([max(DataStructs.BulkTanimotoSimilarity(q, f_tr)) for q in f_te])
    tgt = hist(nn_blind)
    print(f"  blind NN to full train: median {np.median(nn_blind):.3f}")

    # each training compound's NN to all OTHER training compounds -- the value it would
    # have if it alone were held out. The dispersion rule keeps this close to the truth.
    sim = np.zeros((n, n), dtype=np.float32)
    for i, q in enumerate(f_tr):
        sim[i] = DataStructs.BulkTanimotoSimilarity(q, f_tr)
    np.fill_diagonal(sim, -1.0)
    nn_solo = sim.max(axis=1)
    print(f"  train NN to other train: median {np.median(nn_solo):.3f}")

    # label stratum, so the holdout spans activity rather than clustering by similarity
    ycols = [c for c in tr.columns if "pIC50" in c]
    ybar = tr[ycols].mean(axis=1).to_numpy(float) if ycols else np.zeros(n)
    fin = np.isfinite(ybar)
    strat = np.full(n, -1)
    if fin.any():
        q = np.nanquantile(ybar[fin], [.25, .5, .75])
        strat[fin] = np.digitize(ybar[fin], q)

    # greedy: fill the target NN histogram, stratum by stratum, honouring dispersion
    rng = np.random.default_rng(a.seed)
    want = np.round(tgt * a.n).astype(int)
    bins = np.clip(np.digitize(nn_solo, EDGES) - 1, 0, len(EDGES) - 2)
    chosen: list[int] = []
    for b in np.argsort(-want):                      # hardest-to-fill bins first
        need = want[b]
        pool = [i for i in np.where(bins == b)[0] if i not in set(chosen)]
        rng.shuffle(pool)
        # round-robin over label strata so no bin is all-actives or all-inactives
        pool.sort(key=lambda i: (list(chosen).count(strat[i]), rng.random()))
        for i in pool:
            if need <= 0:
                break
            if chosen and sim[i, chosen].max() > a.min_sep:
                continue                              # too close to someone already in
            chosen.append(i); need -= 1
    chosen = np.asarray(sorted(chosen), int)
    keep = np.setdiff1d(np.arange(n), chosen)
    print(f"\n  selected {len(chosen)} of a requested {a.n} "
          f"(dispersion floor --min-sep {a.min_sep} limits the pool)")

    # VERIFY against the real remaining training set, not the solo approximation
    nn_real = np.array([sim[i, keep].max() for i in chosen])
    print(f"\n  VERIFICATION -- does the holdout see what the blind set sees?\n")
    print(f"  {'NN bin':>12s} {'blind %':>9s} {'holdout %':>11s} {'diff':>7s}")
    hb, hh = hist(nn_blind), hist(nn_real)
    for lo, hi, x, y in zip(EDGES[:-1], EDGES[1:], hb, hh):
        print(f"  {f'{lo:.2f}-{hi:.2f}':>12s} {x:9.1%} {y:11.1%} {y-x:+7.1%}")
    print(f"\n  median NN   blind {np.median(nn_blind):.3f}   holdout {np.median(nn_real):.3f}")
    print(f"  total absolute histogram difference: {np.abs(hb-hh).sum()/2:.1%}")
    drop = nn_solo[chosen] - nn_real
    print(f"  NN lost to co-removal: mean {drop.mean():+.4f}, max {drop.max():+.4f} "
          f"(the dispersion rule is working if this is near zero)")

    if ycols:
        print(f"\n  label coverage (mean pIC50 across endpoints)")
        print(f"    full train  mean {np.nanmean(ybar):.3f}  sd {np.nanstd(ybar):.3f}")
        print(f"    holdout     mean {np.nanmean(ybar[chosen]):.3f}  "
              f"sd {np.nanstd(ybar[chosen]):.3f}")

    out = tr.iloc[chosen][[idc, sc]].copy()
    out.columns = ["Molecule_Name", "SMILES"]
    out["nn_to_remaining_train"] = nn_real
    out.to_csv(ROOT / "artifacts" / "holdout_blindlike.csv", index=False)
    print(f"\n  wrote artifacts/holdout_blindlike.csv ({len(out)} compounds)")
    print("\n  HOW TO USE IT. Exclude these Molecule_Names from training entirely, then")
    print("  score on them. Their score is a forecast of the board in a way no")
    print("  cross-validation fold has been, because the difficulty matches. Keep the")
    print("  set FIXED across experiments -- re-selecting it per model turns it back")
    print("  into a tuning surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
