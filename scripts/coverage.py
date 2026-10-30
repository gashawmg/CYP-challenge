#!/usr/bin/env python
"""How much closer does the external set put the blind test to our training data?

The single best predictor of how well any QSAR model does on a given test compound is
how close its nearest training neighbour is. We measured this before: the blind test
sits at median max-Tanimoto 0.587 to the challenge training set, while our CV folds
hold compounds out at 0.44 -- which is why cross-validation reads pessimistic.

If 16k external molecules move that median materially, that is worth more than any
architecture change, because it is the one thing that lifts RANKING, and ranking is
where the leaderboard gap is (top of the board Spearman 0.76, us 0.65).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from cypkit import features


def maxsim(A, B, chunk=256):
    """Max Tanimoto from each row of A to any row of B."""
    A = (A > 0).astype(np.float32); B = (B > 0).astype(np.float32)
    b = B.sum(1)[None, :]
    out = np.zeros(len(A), np.float32)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        inter = a @ B.T
        den = a.sum(1)[:, None] + b - inter
        out[i:i + chunk] = np.divide(inter, den, out=np.zeros_like(inter),
                                     where=den > 0).max(1)
    return out


def main():
    test = pd.read_csv("data/cyp-challenge-TEST-BLINDED.csv")
    test.columns = [c.strip() for c in test.columns]
    tcol = next(c for c in test.columns if c.lower() == "smiles")
    train = pd.read_csv("artifacts/folds.csv")
    ex = pd.read_csv("artifacts/external_cyp.csv")

    Xt, _, _ = features.fingerprint_block(test[tcol].tolist(), "ecfp")
    Xr, _, _ = features.fingerprint_block(train["SMILES"].tolist(), "ecfp")
    Xe, _, _ = features.fingerprint_block(ex["smiles_std"].tolist(), "ecfp")
    print(f"test {Xt.shape[0]}  challenge-train {Xr.shape[0]}  external {Xe.shape[0]}")

    s_now = maxsim(Xt, Xr)
    s_ext = maxsim(Xt, Xe)
    s_both = np.maximum(s_now, s_ext)

    q = [10, 25, 50, 75, 90]
    print(f"\nmax Tanimoto from each BLIND TEST compound to its nearest neighbour")
    print(f"{'percentile':>10s} " + " ".join(f"{p:>7d}" for p in q) + f"{'  mean':>8s}")
    for name, s in (("challenge train", s_now), ("external only", s_ext),
                    ("both", s_both)):
        print(f"{name:>16s} " + " ".join(f"{v:7.3f}" for v in np.percentile(s, q))
              + f" {s.mean():7.3f}")

    for thr in (0.4, 0.5, 0.6, 0.7):
        a = float((s_now >= thr).mean()); b = float((s_both >= thr).mean())
        print(f"  test compounds with a neighbour at Tanimoto >= {thr}: "
              f"{100*a:5.1f}%  ->  {100*b:5.1f}%")

    # where does the external set help most -- the compounds we currently cover worst?
    worst = s_now < np.percentile(s_now, 25)
    print(f"\nfor the quartile of the blind test we cover WORST "
          f"(max sim < {np.percentile(s_now,25):.3f}):")
    print(f"  median nearest neighbour {np.median(s_now[worst]):.3f} -> "
          f"{np.median(s_both[worst]):.3f}")
    pd.DataFrame({"max_sim_train": s_now, "max_sim_external": s_ext,
                  "max_sim_both": s_both}).to_csv("artifacts/coverage.csv", index=False)
    print("\nwritten artifacts/coverage.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
