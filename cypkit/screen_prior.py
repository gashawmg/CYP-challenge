"""Transfer features from the single-concentration primary screen.

    from cypkit import screen_prior as SP
    X, names = SP.calculate(smiles_list)      # (n, 8)

Eight columns per molecule: a predicted `log2fc_estimate` and a predicted P(hit) for
each of the four isoforms, from models fitted on
`cyp-challenge-single-concentration-TRAIN.csv` -- 4,376 compounds x 4 enzymes.

WHY THIS IS A DIFFERENT KIND OF PRIOR FROM `external_prior`. The PubChem panel is
foreign chemistry (blind-set nearest-neighbour similarity 0.368) and buys knowledge
transfer only. The screen is the SAME chemical series, run by the same people, on the
same plates -- and it covers 4,376 compounds against 1,497 CYP2D6 TDI labels and 1,493
CYP2D6 curves. It is the largest untapped label source in the challenge.

WHY PREDICTED AND NOT MEASURED. Zero of the 750 blind-test compounds appear in the
screen. A measured log2fc column would therefore be present in training and absent at
test -- the single most reliable way to build a model that validates beautifully and
scores nothing. So the screen is consumed through a model: a compound's own reading
trains the model that scores everyone else.

LEAKAGE. A compound that has its own screen reading gets an OUT-OF-FOLD prediction,
from the four folds that exclude it; everything else (the blind set, and any compound
the screen never covered) gets the full-fit model. Without this the feature is
near-exact on training rows and merely predicted at test, which inflates
cross-validation and buys nothing.

Cached under `artifacts/screen_prior.csv` keyed by canonical SMILES; delete to refit.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from cypkit import config as C, features, splits

ISO = list(C.ISOFORMS)
CACHE = "artifacts/screen_prior.csv"


def feature_names() -> list[str]:
    return [f"screen_{i}_log2fc" for i in ISO] + [f"screen_{i}_phit" for i in ISO]


def _canon(smiles):
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        out.append(Chem.MolToSmiles(m) if m is not None else str(s))
    return out


def _screen(data_dir=None) -> pd.DataFrame:
    from cypkit import io  # noqa: PLC0415
    d = io.load("single_conc", data_dir)
    d.columns = [c.strip() for c in d.columns]
    g = (d.groupby([C.ID_COL, "enzyme"])
           .agg(lfc=("log2fc_estimate", "mean"), fdr=("log2fc_fdr", "min"))
           .reset_index())
    g["hit"] = ((g["lfc"] <= C.HIT_LOG2FC_MAX) & (g["fdr"] <= C.HIT_FDR_MAX)).astype(int)
    smi = d.drop_duplicates(C.ID_COL).set_index(C.ID_COL)[C.SMILES_COL]
    g[C.SMILES_COL] = g[C.ID_COL].map(smi)
    return g


def build(cache: str = CACHE, verbose=True) -> pd.DataFrame:
    """Fit the screen models and write the per-SMILES feature table.

    Two passes per isoform: a per-fold fit that scores the held-out screen compounds,
    and a full fit that scores everything else.
    """
    import lightgbm as lgb  # noqa: PLC0415

    g = _screen()
    ids = sorted(g[C.ID_COL].unique())
    smi_of = g.drop_duplicates(C.ID_COL).set_index(C.ID_COL)[C.SMILES_COL].to_dict()
    smiles = [smi_of[i] for i in ids]
    if verbose:
        print(f"  screen prior: fitting on {len(ids)} screened compounds")
    X, _, _ = features.fingerprint_block(smiles, "ecfp")
    row_of = {i: k for k, i in enumerate(ids)}

    folds = splits.load_folds(C.artifact("folds.csv"))
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))
    n_folds = int(folds["fold"].max()) + 1
    extra = [i for i in ids if i not in fold_of]
    if extra:
        fold_of.update(dict(zip(extra, splits.assign_extra_to_folds(
            [smi_of[i] for i in extra], folds))))
    f_all = np.array([fold_of[i] for i in ids])

    out = pd.DataFrame({"smiles_canon": _canon(smiles)})
    models = {}
    for iso in ISO:
        s = g[g["enzyme"] == iso]
        r = np.array([row_of[i] for i in s[C.ID_COL]])
        y = s["lfc"].to_numpy(float)
        h = s["hit"].to_numpy(int)
        f = f_all[r]

        oof_l = np.full(len(ids), np.nan)
        oof_h = np.full(len(ids), np.nan)
        for k in range(n_folds):
            tr, te = f != k, f == k
            if te.sum() == 0 or tr.sum() < 100:
                continue
            m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                  colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                                  verbose=-1, n_jobs=-1, random_state=0)
            m.fit(X[r[tr]], y[tr])
            oof_l[r[te]] = m.predict(X[r[te]])
            if 0 < h[tr].mean() < 1:
                c = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                       num_leaves=63, colsample_bytree=0.3,
                                       subsample=0.8, subsample_freq=1, verbose=-1,
                                       n_jobs=-1, random_state=0)
                c.fit(X[r[tr]], h[tr])
                oof_h[r[te]] = c.predict_proba(X[r[te]])[:, 1]

        full_r = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                   colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                                   verbose=-1, n_jobs=-1, random_state=0)
        full_r.fit(X[r], y)
        full_c = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                    colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                                    verbose=-1, n_jobs=-1, random_state=0)
        full_c.fit(X[r], h)
        models[iso] = (full_r, full_c)
        out[f"screen_{iso}_log2fc"] = oof_l
        out[f"screen_{iso}_phit"] = oof_h
        if verbose:
            from scipy.stats import spearmanr  # noqa: PLC0415
            ok = np.isfinite(oof_l)
            print(f"    {iso}: OOF rho(log2fc) {spearmanr(y, oof_l[r])[0]:+.3f}, "
                  f"hit rate {h.mean():.3f}")

    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    out.to_csv(cache, index=False)
    import pickle  # noqa: PLC0415
    with open(cache.replace(".csv", "_models.pkl"), "wb") as fh:
        pickle.dump(models, fh)
    return out


def calculate(smiles, verbose=True) -> tuple[np.ndarray, list[str]]:
    """(n, 8) screen-prior features, in `feature_names()` order. NaN-free."""
    import pickle  # noqa: PLC0415
    if not os.path.exists(CACHE):
        build(verbose=verbose)
    tab = pd.read_csv(CACHE).drop_duplicates("smiles_canon").set_index("smiles_canon")
    with open(CACHE.replace(".csv", "_models.pkl"), "rb") as fh:
        models = pickle.load(fh)

    canon = _canon(list(smiles))
    known = tab.reindex(canon)
    need = known[feature_names()[0]].isna().to_numpy()
    cols = {n: known[n].to_numpy(float).copy() for n in feature_names()}
    if need.any():
        X, _, _ = features.fingerprint_block([s for s, m in zip(smiles, need) if m], "ecfp")
        for iso in ISO:
            r, c = models[iso]
            cols[f"screen_{iso}_log2fc"][need] = r.predict(X)
            cols[f"screen_{iso}_phit"][need] = c.predict_proba(X)[:, 1]
    if verbose:
        print(f"    screen prior: {int((~need).sum())} out-of-fold, "
              f"{int(need.sum())} model-predicted")
    out = np.vstack([cols[n] for n in feature_names()]).T.astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), feature_names()
