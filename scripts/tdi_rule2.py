#!/usr/bin/env python
r"""Second pass on the TDI label rule. The first pass showed that imposing the rule on two
separately predicted pIC50 values loses: the shift T-D is a small difference of two large
correlated numbers, so differencing two predictions with RMSE ~0.7 gives pearson 0.157 on
a signal whose own standard deviation is 0.30.

But the rule also says the positives sit inside a GATE -- a direct curve must exist and the
TDI-condition pIC50 must clear 4.301 -- and the gate alone, known perfectly, scores
MCC 0.648 on CYP3A4 against the binary classifier's 0.406. So the question this pass asks
is not "can the rule replace the classifier" but "does telling the classifier about the
gate help it".

Four scorers on identical scaffold folds:
    binary        a classifier on the structural features alone (the control)
    gate          predicted gate membership only, nothing about the shift
    gate x shift  the gate times a directly-regressed shift, not a difference of two
    stacked       the binary classifier given T-hat, D-hat, shift-hat and P(curve) as
                  four extra features, with NESTED inner folds so the extra features are
                  always out-of-sample for the rows the stacker trains on

    python scripts/tdi_rule2.py --iso CYP3A4 CYP2D6
"""
from __future__ import annotations

import argparse, os, time
import numpy as np, pandas as pd

LOG2 = np.log10(2.0)
T_BOUND = 4.301
CACHE = "artifacts/tdi_feat_cache.npz"


def featurise(smiles, n_bits):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    nd = len(Descriptors.descList)
    X, scaf, ok = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            X.append(np.zeros(nd + n_bits)); scaf.append(""); ok.append(False); continue
        try:
            d = np.asarray(list(Descriptors.CalcMolDescriptors(m).values()), dtype=float)
        except Exception:                                            # noqa: BLE001
            X.append(np.zeros(nd + n_bits)); scaf.append(""); ok.append(False); continue
        fp = np.log1p(gen.GetCountFingerprintAsNumPy(m).astype(np.float64))
        X.append(np.concatenate([d, fp])); ok.append(True)
        try:
            scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:                                            # noqa: BLE001
            scaf.append("")
    X = np.nan_to_num(np.vstack(X), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X, np.array(scaf), np.array(ok)


def scaffold_folds(scaf, k, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, s in enumerate(scaf):
        groups.setdefault(s if s else f"__s{i}", []).append(i)
    fold = np.zeros(len(scaf), dtype=int); load = np.zeros(k, dtype=int)
    for g in sorted(groups.values(), key=lambda g: (-len(g), rng.random())):
        j = int(load.argmin())
        for i in g:
            fold[i] = j
        load[j] += len(g)
    return fold


def mcc(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, bool)
    tp = int((y & p).sum()); tn = int((~y & ~p).sum())
    fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    den = float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn)
    return 0.0 if den == 0 else (tp*tn-fp*fn)/np.sqrt(den)


def best_cut(y, s):
    qs = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 145)))
    b = (-2.0, 0.5)
    for q in qs:
        m = mcc(y, s >= q)
        if m > b[0]:
            b = (m, float(q))
    return b


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/mnt/user-data/uploads/CYP450/cyp/data/"
                                       "cyp-challenge-TRAIN_TDI.csv")
    ap.add_argument("--iso", nargs="+", default=["CYP3A4", "CYP2D6"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bits", type=int, default=512)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--no_stack", action="store_true",
                    help="skip the nested stacker; it triples the fit count")
    a = ap.parse_args(argv)

    from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
    from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv(a.train)
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        X, scaf, ok = z["X"], z["scaf"], z["ok"]
        print(f"features {X.shape} from cache")
    else:
        t0 = time.time()
        X, scaf, ok = featurise(df.SMILES.tolist(), a.bits)
        np.savez_compressed(CACHE, X=X, scaf=scaf, ok=ok)
        print(f"features {X.shape} in {time.time()-t0:.1f} s, cached")

    def reg(tgt, tr, te):
        g = tr[~np.isnan(tgt[tr])]
        m = HGBR(max_iter=a.iters, learning_rate=0.08, max_leaf_nodes=31,
                 l2_regularization=1.0, random_state=a.seed)
        m.fit(X[g], tgt[g]); return m.predict(X[te])

    def clf(lbl, tr, te):
        m = HGBC(max_iter=a.iters, learning_rate=0.08, max_leaf_nodes=31,
                 l2_regularization=1.0, random_state=a.seed)
        m.fit(X[tr], lbl[tr]); return m.predict_proba(X[te])[:, 1]

    out = []
    for iso in a.iso:
        T = df[f"{iso}_pIC50_TDI_condition"].values.astype(float)
        D = df[f"{iso}_pIC50_direct_inhibition"].values.astype(float)
        SH = T - D
        hasD = (~np.isnan(D)).astype(int)
        lab = df[f"{iso}_is_TDI"]
        m = lab.notna().values & ok
        idx = np.where(m)[0]
        y = (lab.values[idx] == True)                                # noqa: E712
        fold = scaffold_folds(scaf[idx], a.folds, a.seed)
        allok = np.where(ok)[0]
        print(f"\n=== {iso}  rows {len(idx)}  positives {y.sum()} ({y.mean():.4f})")

        n = len(idx)
        oT = np.zeros(n); oD = np.zeros(n); oS = np.zeros(n)
        oC = np.zeros(n); oB = np.zeros(n); oK = np.zeros(n)
        t0 = time.time()
        for f in range(a.folds):
            va = fold == f
            te = idx[va]
            held = set(te.tolist())
            tr = np.array([i for i in allok if i not in held])
            trl = np.array([i for i in idx if i not in held])
            oT[va] = reg(T, tr, te)
            oD[va] = reg(D, tr, te)
            oS[va] = reg(SH, tr, te)
            oC[va] = clf(hasD, tr, te)
            oB[va] = clf((lab.values == True).astype(int), trl, te)  # noqa: E712

            if a.no_stack:
                print(f"  fold {f+1}/{a.folds} done  {time.time()-t0:.0f} s")
                continue
            # nested: build the same four features for the stacker's own training rows,
            # out-of-sample, using inner folds inside the outer training set
            inner = scaffold_folds(scaf[trl], 3, a.seed + 1)
            fT = np.zeros(len(trl)); fD = np.zeros(len(trl))
            fS = np.zeros(len(trl)); fC = np.zeros(len(trl))
            for g in range(3):
                iva = inner == g
                ite = trl[iva]
                ih = set(ite.tolist())
                itr = np.array([i for i in tr if i not in ih])
                fT[iva] = reg(T, itr, ite); fD[iva] = reg(D, itr, ite)
                fS[iva] = reg(SH, itr, ite); fC[iva] = clf(hasD, itr, ite)
            Xtr = np.column_stack([X[trl], fT, fD, fT - fD, fS, fC])
            Xte = np.column_stack([X[te], oT[va], oD[va], oT[va]-oD[va], oS[va], oC[va]])
            s = HGBC(max_iter=a.iters, learning_rate=0.08, max_leaf_nodes=31,
                     l2_regularization=1.0, random_state=a.seed)
            s.fit(Xtr, (lab.values[trl] == True).astype(int))        # noqa: E712
            oK[va] = s.predict_proba(Xte)[:, 1]
            print(f"  fold {f+1}/{a.folds} done  {time.time()-t0:.0f} s")

        for nm, pr, tg in [("T", oT, T[idx]), ("D", oD, D[idx]), ("shift direct", oS, SH[idx])]:
            k = ~np.isnan(tg)
            print(f"  {nm:14s} pearson {np.corrcoef(pr[k], tg[k])[0,1]:.4f}  n {k.sum()}")
        print(f"  shift by differencing  pearson "
              f"{np.corrcoef((oT-oD)[~np.isnan(SH[idx])], SH[idx][~np.isnan(SH[idx])])[0,1]:.4f}")

        def sig(z, s): return 1.0/(1.0+np.exp(-z/s))
        gate = sig(oT - T_BOUND, 0.30) * oC
        scorers = [("binary (control)", oB), ("gate only", gate),
                   ("gate x shift", gate * sig(oS - LOG2, 0.15)),
                   ("binary x gate", oB * gate),
                   ("binary x shift", oB * sig(oS - LOG2, 0.15))]
        if not a.no_stack:
            scorers += [("stacked", oK), ("stacked + binary", 0.5*oK + 0.5*oB)]
        print(f"  {'scorer':20s}{'MCC':>8}{'cut':>9}{'pos rate':>10}")
        for nm, s in scorers:
            v, c = best_cut(y, s)
            out.append(dict(isoform=iso, scorer=nm, MCC=v, cut=c,
                            pos_rate=float((s >= c).mean())))
            print(f"  {nm:20s}{v:>8.4f}{c:>9.4f}{(s>=c).mean():>10.4f}")
        np.savez(f"artifacts/tdi_oof_{iso}.npz", idx=idx, y=y, fold=fold,
                 oT=oT, oD=oD, oS=oS, oC=oC, oB=oB, oK=oK)

    pd.DataFrame(out).to_csv("artifacts/tdi_rule2_summary.csv", index=False)
    print("\nwrote artifacts/tdi_rule2_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
