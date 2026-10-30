#!/usr/bin/env python
r"""CYP2D6 TDI: a validation set that actually resembles the blind task, and the model
class that diagnosis implies.

    python train_tdi_2d6_local.py
    python train_tdi_2d6_local.py --cutoff 0.55

WHY EVERY CYP2D6 TDI DECISION SO FAR WAS MADE AGAINST NOISE. Our out-of-fold d' on CYP2D6
is 0.280 against the 1.224 the board implies -- a factor of 4.37 that Section 14 blamed on
prevalence and Section 24 paid for with a rank-9-to-36 submission. Neither diagnosis was right.
Split our own rows by how close each one's nearest labelled neighbour is:

    nearest-neighbour Tanimoto    n      d'
    0.14 - 0.30                  374   0.137
    0.30 - 0.35                  373   0.180
    0.35 - 0.42                  375   0.341
    0.42 - 1.00                  375   0.436
    >= 0.55  (blind-like)         83   0.971      <- against a blind d' of 1.224

THE MODEL WAS NEVER BROKEN; IT WAS BEING GRADED ON COMPOUNDS FAR HARDER THAN THE EXAM.
Our CYP2D6 TDI rows have a median nearest-neighbour similarity of 0.351. The blind
compounds sit at 0.471 from that same labelled set, and 83 of our rows reach 0.55. The
task is INTERPOLATION; our validation was EXTRAPOLATION. CYP3A4 never had this problem --
median 0.492, and its local d' of 1.300 matches its blind 1.276 almost exactly, which is
why that endpoint is rank 8 and this one is rank 18.

WHAT THAT PREDICTS ABOUT MODELS, AND WHAT THIS SCRIPT TESTS. Selecting models under
extrapolation rewards smoothness and heavy regularisation -- and penalises exactly the
behaviour that wins at interpolation: exploiting near neighbours. So the models this
project rejected may have been rejected for succeeding at the wrong task. The arms below
span that axis deliberately, from a pure similarity-weighted k-NN (all memorisation, no
smoothing) to a heavily regularised booster (all smoothing), scored on the blind-like
subset rather than on everything.

READ THE UNCERTAINTY BEFORE THE RANKING. At cutoff 0.55 the subset is 83 rows and d'
carries about +-0.33; at 0.50 it is 188 rows and +-0.21. This instrument is honest but
coarse -- it can separate a large effect from nothing and cannot rank two similar models.
Per Section 19 and Section 24: it may support a submission only if the gap it shows exceeds that spread,
and where it does not, the admissible action is to HOLD.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from cypkit import config as C, io  # noqa: E402
from train_tdi_shift import features  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
BLIND_D = 1.224


def fps(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            out.append(gen.GetFingerprint(m)); keep.append(i)
    return out, np.asarray(keep, int)


def dprime(y, s):
    return float(np.sqrt(2) * norm.ppf(np.clip(roc_auc_score(y, s), 1e-6, 1 - 1e-6)))


def knn_predict(fp_tr, y_tr, fp_te, k=15, power=3.0):
    """Similarity-weighted k-NN -- pure memorisation, the interpolation extreme."""
    out = np.zeros(len(fp_te))
    for i, q in enumerate(fp_te):
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(q, fp_tr))
        idx = np.argsort(s)[::-1][:k]
        w = s[idx] ** power
        out[i] = float(np.dot(w, y_tr[idx]) / w.sum()) if w.sum() > 0 else y_tr.mean()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", type=float, default=0.50,
                    help="nearest-neighbour Tanimoto floor for the evaluation subset")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    tr = io.load("train_tdi"); tr.columns = [c.strip() for c in tr.columns]
    sub = tr.dropna(subset=["CYP2D6_is_TDI"]).reset_index(drop=True)
    fp, keep = fps(sub[C.SMILES_COL].tolist())
    sub = sub.iloc[keep].reset_index(drop=True)
    y = (sub["CYP2D6_is_TDI"] == True).astype(int).to_numpy()      # noqa: E712
    X, _ = features(sub[C.SMILES_COL].tolist())
    nn = np.array([max(np.delete(DataStructs.BulkTanimotoSimilarity(q, fp), i))
                   for i, q in enumerate(fp)])
    ev = nn >= a.cutoff
    print(f"\n  {len(y)} labelled rows; {int(ev.sum())} at nearest-neighbour >= {a.cutoff}")
    print(f"  evaluation subset: {y[ev].mean():.1%} positive "
          f"(blind is 7.1%); blind d' is {BLIND_D:.3f}\n")

    import lightgbm as lgb  # noqa: PLC0415
    base = dict(n_estimators=600, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=-1)
    arms = {
        "knn (memorise)":  None,
        "lgbm loose":      dict(learning_rate=0.05, num_leaves=127, min_child_samples=5,
                                colsample_bytree=0.8, reg_lambda=0.0, **base),
        "lgbm incumbent":  dict(learning_rate=0.03, num_leaves=31, min_child_samples=20,
                                colsample_bytree=0.7, reg_lambda=1.0, **base),
        "lgbm tight":      dict(learning_rate=0.02, num_leaves=15, min_child_samples=60,
                                colsample_bytree=0.4, reg_lambda=10.0, **base),
    }
    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=0)
    rng = np.random.default_rng(0)
    print(f"  {'arm':18s} {'d’ ALL rows':>12s} {'d’ blind-like':>14s} {'+-':>7s}")
    rows = []
    for name, params in arms.items():
        o = np.zeros(len(y))
        for tr_i, te_i in skf.split(X, y):
            if params is None:
                o[te_i] = knn_predict([fp[i] for i in tr_i], y[tr_i],
                                      [fp[i] for i in te_i])
            else:
                m = lgb.LGBMClassifier(**params)
                m.fit(X[tr_i], y[tr_i])
                o[te_i] = m.predict_proba(X[te_i])[:, 1]
        d_all = dprime(y, o)
        d_ev = dprime(y[ev], o[ev])
        b = [dprime(y[ev][i], o[ev][i])
             for i in (rng.integers(0, int(ev.sum()), int(ev.sum())) for _ in range(400))
             if len(set(y[ev][i])) > 1]
        print(f"  {name:18s} {d_all:12.3f} {d_ev:14.3f} {np.std(b):7.3f}")
        rows.append(dict(arm=name, d_all=d_all, d_blindlike=d_ev, sd=float(np.std(b))))

    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "artifacts" / "tdi_2d6_local_summary.csv", index=False)
    best_all = r.loc[r.d_all.idxmax(), "arm"]
    best_ev = r.loc[r.d_blindlike.idxmax(), "arm"]
    print(f"\n  best on ALL rows (the old, wrong yardstick): {best_all}")
    print(f"  best on BLIND-LIKE rows (the task):           {best_ev}")
    if best_all != best_ev:
        print("  -> THE TWO YARDSTICKS DISAGREE. Model selection on the full set has been")
        print("     choosing for extrapolation, which is not the task being scored.")
    else:
        print("  -> the yardsticks agree here; the difficulty split does not change the "
              "model choice.")
    sd = r.sd.max()
    print(f"\n  spread on the blind-like subset is +-{sd:.3f}; a gap smaller than that is "
          f"not a result\n  (Section 19, Section 24) and the admissible action is to HOLD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
