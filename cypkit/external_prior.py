"""Transfer features: what a model trained only on public CYP data thinks.

    from cypkit import external_prior as EP
    X, names = EP.calculate(smiles_list)      # (n, 10)

Ten columns per molecule -- a predicted pIC50 and a predicted P(active) for each of
CYP1A2, 2C9, 2C19, 2D6, 3A4 -- produced by models fitted on the PubChem/Veith qHTS
panel (`artifacts/external_cyp.csv`, ~16k compounds). Those models never see a
challenge label, so the columns are a legitimate feature for training and test rows
alike, and they carry SAR learned from 3x more molecules than we have.

WHY A FEATURE AND NOT EXTRA TRAINING ROWS. On the 178 compounds measured in both
sets, Veith pIC50 and challenge pIC50 correlate well (Spearman 0.52-0.81) but sit on
DIFFERENT SCALES: bias -0.3 to -1.1 log units, fitted slopes 0.60 to 1.86. Pooling
them as one regression target would import that distortion. As a feature, the
challenge model learns the mapping itself, per isoform, and is free to ignore it.

WHAT IT IS WORTH, MEASURED. On the frozen folds, added to the rdkit+cyp descriptor
block: mean Spearman over all rows 0.495 -> 0.507, over rows with a FITTED curve
(the ranking that the leaderboard actually separates people on) 0.542 -> 0.565, mean
absolute error 0.422 -> 0.414. Real, consistent across all four isoforms, and modest.

WHAT IT IS NOT WORTH. It does not extend coverage. The blind test sits at median
max-Tanimoto 0.587 to the challenge training set but only 0.368 to the external set;
adding all 16k external molecules moves blind-test nearest-neighbour similarity from
0.587 to 0.589. The Veith library is a diverse HTS collection, the challenge set is a
focused series -- different chemical worlds. So this is a knowledge transfer, not a
coverage fix, and it should be expected to buy tenths, not the leaderboard.

The fitted models are cached under `artifacts/external_prior_models.pkl`; delete that
file to refit.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from cypkit import features

AUX = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
EXTERNAL_CSV = "artifacts/external_cyp.csv"
MODEL_CACHE = "artifacts/external_prior_models.pkl"
TRAIN_CSV = "data/cyp-challenge-TRAIN_inhibition.csv"


def feature_names() -> list[str]:
    return ([f"prior_{i}_pIC50" for i in AUX] + [f"prior_{i}_pact" for i in AUX])


def _challenge_keys(train_csv: str = TRAIN_CSV) -> set[str]:
    """InChIKey first blocks of the challenge training set."""
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem.MolStandardize.rdMolStandardize import (  # noqa: PLC0415
        LargestFragmentChooser, Uncharger)
    RDLogger.DisableLog("rdApp.*")
    lf, uc = LargestFragmentChooser(), Uncharger()
    if not os.path.exists(train_csv):
        return set()
    tr = pd.read_csv(train_csv)
    tr.columns = [c.strip() for c in tr.columns]
    col = next(c for c in tr.columns if c.lower() == "smiles")
    out = set()
    for s in tr[col]:
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            out.add(Chem.MolToInchiKey(uc.uncharge(lf.choose(m))).split("-")[0])
    return out


def fit(external_csv: str = EXTERNAL_CSV, cache: str = MODEL_CACHE, verbose=True):
    """Fit one regressor and one classifier per isoform on the external panel.

    Compounds that also appear in the challenge TRAINING set are removed first.
    Leaving them in would make the prior near-perfect on exactly those rows, which
    inflates cross-validation without helping the blind test at all -- the classic way
    a transfer feature lies to you.
    """
    import lightgbm as lgb  # noqa: PLC0415

    ex = pd.read_csv(external_csv)
    n0 = len(ex)
    if "inchikey" in ex.columns:
        ex = ex[~ex["inchikey"].isin(_challenge_keys())].reset_index(drop=True)
    smi_col = "smiles_std" if "smiles_std" in ex.columns else "SMILES"
    if verbose:
        print(f"  external prior: {n0} -> {len(ex)} compounds "
              f"(challenge-train overlap removed)")
    X, _, _ = features.fingerprint_block(ex[smi_col].tolist(), "ecfp")

    models = {}
    for iso in AUX:
        pc, ac = f"{iso}_pIC50", f"{iso}_active"
        if pc in ex.columns:
            y = ex[pc]; ok = y.notna().to_numpy()
            r = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                  colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                                  verbose=-1, n_jobs=-1, random_state=0)
            r.fit(X[ok], y[ok]); models[pc] = r
        if ac in ex.columns:
            y = ex[ac]; ok = y.notna().to_numpy()
            if ok.sum() > 100 and 0 < y[ok].mean() < 1:
                c = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                       num_leaves=63, colsample_bytree=0.3,
                                       subsample=0.8, subsample_freq=1, verbose=-1,
                                       n_jobs=-1, random_state=0)
                c.fit(X[ok], y[ok].astype(int)); models[ac] = c
        if verbose:
            print(f"    {iso}: regressor + classifier fitted")
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(models, fh)
    return models


def _models(cache: str = MODEL_CACHE, verbose=True):
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    if not os.path.exists(EXTERNAL_CSV):
        raise FileNotFoundError(
            f"{EXTERNAL_CSV} not found. Run scripts/fetch_pubchem_cyp.py first, "
            "or drop the 'prior' feature block.")
    return fit(verbose=verbose)


def calculate(smiles, verbose=True) -> tuple[np.ndarray, list[str]]:
    """(n, 10) prior features, in `feature_names()` order. NaN-free."""
    models = _models(verbose=verbose)
    X, _, _ = features.fingerprint_block(list(smiles), "ecfp")
    cols = []
    for iso in AUX:
        m = models.get(f"{iso}_pIC50")
        cols.append(m.predict(X) if m is not None else np.full(len(X), np.nan))
    for iso in AUX:
        m = models.get(f"{iso}_active")
        cols.append(m.predict_proba(X)[:, 1] if m is not None
                    else np.full(len(X), np.nan))
    out = np.vstack(cols).T.astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), feature_names()
