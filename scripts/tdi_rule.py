#!/usr/bin/env python
r"""Build the TDI prediction from the LABEL RULE instead of classifying the label.

The organisers' is_TDI flag is not an opaque annotation. On every one of the 1,497 CYP2D6
and 3,584 CYP3A4 labelled rows it is reproduced exactly by

    is_TDI = (pIC50_TDI_condition - pIC50_direct_inhibition >= log10(2))
             AND (pIC50_TDI_condition >= 4.301)

with a missing direct curve making it False. log10(2) is a two-fold IC50 shift and 4.301
is pIC50 for 50 uM -- the textbook TDI criteria. Zero disagreements, both endpoints.

So the task decomposes into three things we can model as regressions and a classifier:
    T       the TDI-condition pIC50
    D       the direct-inhibition pIC50
    curve   whether a direct curve exists at all (no curve -> always False)

This script fits all three on identical scaffold folds, combines them through the rule,
and scores the result against a direct binary classifier trained on the same features and
the same folds. Nothing is recommended unless the rule-structured version wins that
head-to-head.

    python scripts/tdi_rule.py --iso CYP3A4 CYP2D6 --folds 5
"""
from __future__ import annotations

import argparse, sys, time
import numpy as np, pandas as pd

LOG2 = np.log10(2.0)
T_BOUND = 4.301


def featurise(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, countSimulation=False)
    names = [n for n, _ in Descriptors.descList]
    X, scaf, ok = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            X.append(None); scaf.append(""); ok.append(False); continue
        try:
            d = list(Descriptors.CalcMolDescriptors(m).values())
        except Exception:                                            # noqa: BLE001
            X.append(None); scaf.append(""); ok.append(False); continue
        fp = np.log1p(gen.GetCountFingerprintAsNumPy(m).astype(np.float32))
        X.append(np.concatenate([np.asarray(d, dtype=np.float64), fp]))
        try:
            scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:                                            # noqa: BLE001
            scaf.append("")
        ok.append(True)
    width = len(names) + 2048
    Xa = np.vstack([x if x is not None else np.full(width, np.nan) for x in X])
    Xa = np.nan_to_num(Xa, nan=0.0, posinf=0.0, neginf=0.0)
    return Xa.astype(np.float32), np.array(scaf), np.array(ok)


def scaffold_folds(scaf, k, seed):
    """Group by Bemis-Murcko scaffold, then greedily fill folds largest-group-first so no
    scaffold is split across a fold boundary."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, s in enumerate(scaf):
        groups.setdefault(s if s else f"__singleton_{i}", []).append(i)
    order = sorted(groups.values(), key=lambda g: (-len(g), rng.random()))
    fold = np.zeros(len(scaf), dtype=int)
    load = np.zeros(k, dtype=int)
    for g in order:
        j = int(load.argmin())
        for i in g:
            fold[i] = j
        load[j] += len(g)
    return fold


def mcc(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, bool)
    tp = int((y & p).sum()); tn = int((~y & ~p).sum())
    fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    den = float(tp+fp) * float(tp+fn) * float(tn+fp) * float(tn+fn)
    return 0.0 if den == 0 else (tp*tn - fp*fn) / np.sqrt(den)


def best_threshold(y, score):
    """The MCC-maximising cut, searched over the score's own quantiles."""
    qs = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 197)))
    best = (-2.0, 0.5)
    for q in qs:
        m = mcc(y, score >= q)
        if m > best[0]:
            best = (m, float(q))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/mnt/user-data/uploads/CYP450/cyp/data/"
                                       "cyp-challenge-TRAIN_TDI.csv")
    ap.add_argument("--iso", nargs="+", default=["CYP3A4", "CYP2D6"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out", default="artifacts/tdi_rule_oof.csv")
    a = ap.parse_args(argv)

    from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
    from sklearn.ensemble import HistGradientBoostingClassifier as HGBC

    df = pd.read_csv(a.train)
    t0 = time.time()
    X, scaf, ok = featurise(df.SMILES.tolist())
    print(f"features {X.shape} in {time.time()-t0:.1f} s   unparsable {int((~ok).sum())}")

    rows = []
    for iso in a.iso:
        T = df[f"{iso}_pIC50_TDI_condition"].values.astype(float)
        D = df[f"{iso}_pIC50_direct_inhibition"].values.astype(float)
        lab = df[f"{iso}_is_TDI"]
        m = lab.notna().values & ok
        y = (lab.values[m] == True)                                  # noqa: E712
        idx = np.where(m)[0]
        fold = scaffold_folds(scaf[idx], a.folds, a.seed)
        print(f"\n=== {iso}   scored rows {m.sum()}   positives {y.sum()} "
              f"({y.mean():.4f})   folds {a.folds}")

        # the three components, each fitted on whichever TRAINING rows have that target
        oT = np.full(m.sum(), np.nan)
        oD = np.full(m.sum(), np.nan)
        oC = np.full(m.sum(), np.nan)
        oB = np.full(m.sum(), np.nan)                                # the control
        for f in range(a.folds):
            va = fold == f
            va_glob = idx[va]
            held = set(va_glob.tolist())
            tr_all = np.array([i for i in range(len(df)) if ok[i] and i not in held])

            def fit_reg(target):
                g = tr_all[~np.isnan(target[tr_all])]
                mdl = HGBR(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                           l2_regularization=1.0, random_state=a.seed)
                mdl.fit(X[g], target[g])
                return mdl.predict(X[va_glob])

            oT[va] = fit_reg(T)
            oD[va] = fit_reg(D)

            hasD = (~np.isnan(D)).astype(int)
            c = HGBC(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                     l2_regularization=1.0, random_state=a.seed)
            c.fit(X[tr_all], hasD[tr_all])
            oC[va] = c.predict_proba(X[va_glob])[:, 1]

            tr_lab = np.array([i for i in idx if i not in held])
            b = HGBC(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                     l2_regularization=1.0, random_state=a.seed)
            b.fit(X[tr_lab], (lab.values[tr_lab] == True).astype(int))   # noqa: E712
            oB[va] = b.predict_proba(X[va_glob])[:, 1]

        sh = oT - oD
        print(f"  component quality on held-out rows")
        for nm, pr, tg in [("T  (TDI-condition pIC50)", oT, T[idx]),
                           ("D  (direct pIC50)       ", oD, D[idx])]:
            k = ~np.isnan(tg)
            r = np.corrcoef(pr[k], tg[k])[0, 1]
            print(f"    {nm}  n {k.sum():5d}  pearson {r:.4f}  "
                  f"rmse {np.sqrt(np.mean((pr[k]-tg[k])**2)):.4f}")
        kk = ~np.isnan(T[idx]) & ~np.isnan(D[idx])
        print(f"    shift T-D              n {kk.sum():5d}  "
              f"pearson {np.corrcoef(sh[kk], (T[idx]-D[idx])[kk])[0,1]:.4f}")
        print(f"    curve-exists classifier       AUC-ish pos rate "
              f"{oC.mean():.3f}")

        # scores
        s_hard = ((sh >= LOG2) & (oT >= T_BOUND)).astype(float)
        def sig(z, s): return 1.0 / (1.0 + np.exp(-z / s))
        s_soft = sig(oT - T_BOUND, 0.35) * sig(sh - LOG2, 0.18) * oC
        s_comb = 0.5 * s_soft + 0.5 * oB

        print(f"  MCC, each scored at its own MCC-optimal cut")
        for nm, s in [("incumbent-style binary clf", oB),
                      ("rule, hard thresholds    ", s_hard),
                      ("rule, soft               ", s_soft),
                      ("rule + binary, 50/50     ", s_comb)]:
            if nm.startswith("rule, hard"):
                best, cut = mcc(y, s > 0.5), 0.5
            else:
                best, cut = best_threshold(y, s)
            rows.append(dict(isoform=iso, model=nm.strip(), MCC=best, cut=cut,
                             pos_rate=float((s >= cut).mean())))
            print(f"    {nm}  MCC {best:.4f}   cut {cut:.4f}   "
                  f"pred pos rate {(s >= cut).mean():.4f}")

    out = pd.DataFrame(rows)
    import os; os.makedirs("artifacts", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
