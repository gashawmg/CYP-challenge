#!/usr/bin/env python
r"""Why is the CYP2D6 shrinkage 2.25, and can it be checked without a submission?

tdi_validator corrects CYP2D6's out-of-fold d' by blind/OOF = 2.25, a factor fitted on a
single scored submission. The exact label rule from Section 52 supplies a mechanism, and
a mechanism can be tested locally.

The blind CYP2D6 prevalence recovered from the field is 0.0706 against 0.2164 in training.
Under `is_TDI = (shift >= log10(2)) AND (T >= 4.301)`, reproducing 0.0706 on the training
rows needs the shift cut moved from 0.301 to 0.676 -- a 4.74-fold IC50 shift rather than
2.00-fold. So the blind positives are not a random 7% of our positives: they are the
strong-shift third of them, and the marginal two thirds are absent.

CYP3A4 goes the other way: blind 0.2915 against 0.2132, cut moving DOWN to 0.163, which
is why its shrinkage is 1.0.

If that is the mechanism, relabelling the training rows at the strict cut should raise
d' by roughly the same factor, with nothing else changed. This measures it on a Tanimoto
kernel SVC and a Tanimoto kNN -- the two families that actually win on CYP2D6 in the
validator table -- over identical scaffold folds.

    python scripts/tdi_strict_label.py
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd

LOG2 = np.log10(2.0)
T_BOUND = 4.301


def morgan(smiles, bits=2048, radius=2):
    from rdkit import Chem, RDLogger, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    X = np.zeros((len(smiles), bits), dtype=np.uint8)
    scaf = []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            scaf.append(f"__bad{i}"); continue
        fp = gen.GetFingerprint(m)
        arr = np.zeros(bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
        try:
            scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m) or f"__e{i}")
        except Exception:                                            # noqa: BLE001
            scaf.append(f"__e{i}")
    return X, np.array(scaf)


def tanimoto(A, B):
    A = A.astype(np.float32); B = B.astype(np.float32)
    inter = A @ B.T
    a = A.sum(1)[:, None]; b = B.sum(1)[None, :]
    return inter / np.maximum(a + b - inter, 1e-9)


def scaffold_folds(scaf, k, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, s in enumerate(scaf):
        groups.setdefault(s, []).append(i)
    fold = np.zeros(len(scaf), int); load = np.zeros(k, int)
    for g in sorted(groups.values(), key=lambda g: (-len(g), rng.random())):
        j = int(load.argmin())
        for i in g:
            fold[i] = j
        load[j] += len(g)
    return fold


def auc_of(y, s):
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(len(s))
    y = np.asarray(y, bool)
    return (r[y].mean() - (y.sum() - 1) / 2) / (~y).sum()


def dprime(auc):
    from scipy.stats import norm
    return float(np.sqrt(2) * norm.ppf(min(max(auc, 1e-6), 1 - 1e-6)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/mnt/user-data/uploads/CYP450/cyp/data/"
                                       "cyp-challenge-TRAIN_TDI.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    from sklearn.svm import SVC

    df = pd.read_csv(a.train)
    BLIND = {"CYP2D6": 0.0706, "CYP3A4": 0.2915}
    for iso, blind in BLIND.items():
        g = df[df[f"{iso}_is_TDI"].notna()].reset_index(drop=True)
        T = g[f"{iso}_pIC50_TDI_condition"].values.astype(float)
        D = g[f"{iso}_pIC50_direct_inhibition"].values.astype(float)
        sh = T - D
        gate = (~np.isnan(sh)) & (T >= T_BOUND)
        y_real = (g[f"{iso}_is_TDI"] == True).values                 # noqa: E712
        want = int(round(blind * len(g)))
        cut = np.sort(np.where(gate, sh, -np.inf))[::-1][want - 1]
        y_strict = gate & (sh >= cut)
        X, scaf = morgan(g.SMILES.tolist())
        K = tanimoto(X, X)
        fold = scaffold_folds(scaf, a.folds, a.seed)
        print(f"\n=== {iso}  n {len(g)}   label cut {LOG2:.3f} -> strict cut {cut:.3f}"
              f"   ({10**cut:.2f}-fold vs 2.00-fold)")
        print(f"    prevalence {y_real.mean():.4f} -> {y_strict.mean():.4f}"
              f"   (blind {blind:.4f})")
        out = {}
        for nm, y in [("label as given", y_real), ("strict, blind-matched", y_strict)]:
            s_svc = np.zeros(len(g)); s_knn = np.zeros(len(g))
            for f in range(a.folds):
                va = fold == f; tr = ~va
                m = SVC(C=1.0, kernel="precomputed")
                m.fit(K[np.ix_(tr, tr)], y[tr].astype(int))
                s_svc[va] = m.decision_function(K[np.ix_(va, tr)])
                sim = K[np.ix_(va, tr)].copy()
                nn = np.argsort(-sim, axis=1)[:, :10]
                s_knn[va] = np.array([y[tr][nn[i]].mean() for i in range(va.sum())])
            for mn, s in [("svc_tan", s_svc), ("knn_tan", s_knn)]:
                A = auc_of(y, s); out[(nm, mn)] = (A, dprime(A), y, s)
                print(f"    {nm:24s} {mn:8s} AUC {A:.4f}   d' {dprime(A):.4f}")
        for mn in ("svc_tan", "knn_tan"):
            a0 = out[("label as given", mn)]; a1 = out[("strict, blind-matched", mn)]
            ratio = a1[1] / a0[1] if a0[1] else float("nan")
            rng = np.random.default_rng(0); rs = []
            for _ in range(a.boot):
                i = rng.integers(0, len(g), len(g))
                try:
                    d0 = dprime(auc_of(a0[2][i], a0[3][i]))
                    d1 = dprime(auc_of(a1[2][i], a1[3][i]))
                    if d0 > 0.05:
                        rs.append(d1 / d0)
                except Exception:                                    # noqa: BLE001
                    pass
            lo, hi = np.percentile(rs, [2.5, 97.5])
            print(f"    {mn}: d' ratio strict/given {ratio:.3f}   95% CI "
                  f"[{lo:.3f}, {hi:.3f}]"
                  + ("   <- validator assumes 2.25" if iso == "CYP2D6" else
                     "   <- validator assumes 1.00"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
