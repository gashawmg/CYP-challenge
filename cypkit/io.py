"""Loading and normalising the five challenge CSVs.

The real files are not downloadable from every environment, so `synthetic_dataset()`
generates a schema-identical stand-in. Every other module in this package is written
against the normalised frames these functions return, which means the whole pipeline
can be exercised before the real data lands.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


class DataError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

_ID_ALIASES = ("molecule_name", "molecule name", "moleculename", "compound_id", "name")
_SMILES_ALIASES = ("smiles", "canonical_smiles")


def _rename_to(df: pd.DataFrame, target: str, aliases: tuple[str, ...]) -> pd.DataFrame:
    if target in df.columns:
        return df
    lookup = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for alias in aliases:
        key = alias.replace(" ", "_")
        if key in lookup:
            return df.rename(columns={lookup[key]: target})
    return df


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and standardise the id/SMILES columns."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = _rename_to(df, C.ID_COL, _ID_ALIASES)
    df = _rename_to(df, C.SMILES_COL, _SMILES_ALIASES)
    return df


def require_columns(df: pd.DataFrame, cols, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(
            f"{what} is missing expected column(s): {missing}\n"
            f"  present: {sorted(df.columns)[:25]}{' ...' if len(df.columns) > 25 else ''}"
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load one of the five challenge files by its short key."""
    if name not in C.DATA_FILES:
        raise KeyError(f"unknown dataset '{name}'; expected one of {sorted(C.DATA_FILES)}")
    data_dir = Path(data_dir or C.DATA_DIR)
    path = data_dir / C.DATA_FILES[name]
    if not path.exists():
        raise DataError(
            f"{path} not found.\n"
            f"Download the five CSVs from "
            f"https://huggingface.co/datasets/openadmet/cyp-challenge-train-test "
            f"into {data_dir} (see README), or run with --synthetic to smoke-test "
            f"the pipeline first."
        )
    return normalise(pd.read_csv(path))


def load_all(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    return {k: load(k, data_dir) for k in C.DATA_FILES}


def conf_columns(endpoint: str) -> tuple[str, str]:
    return endpoint + C.CONF_LOW_SUFFIX, endpoint + C.CONF_HIGH_SUFFIX


def describe_coverage(train: pd.DataFrame) -> pd.DataFrame:
    """Per-isoform count of measured pIC50 values, and how many carry usable bounds."""
    rows = []
    for ep in C.REGRESSION_ENDPOINTS:
        lo, hi = conf_columns(ep)
        n = int(train[ep].notna().sum()) if ep in train.columns else 0
        has_bounds = (
            int((train[ep].notna() & train[lo].notna() & train[hi].notna()).sum())
            if {ep, lo, hi} <= set(train.columns)
            else 0
        )
        width = (
            (train[hi] - train[lo]).where(train[ep].notna()).median()
            if {lo, hi} <= set(train.columns)
            else np.nan
        )
        rows.append(
            dict(endpoint=ep, n_measured=n, n_with_bounds=has_bounds, median_ci_width=width)
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic stand-in
# ---------------------------------------------------------------------------

# Mono-cyclic only, each using ring-closure digit 1, so the second fragment can be
# renumbered to 2 and the concatenation stays a valid SMILES.
_FRAGMENTS = [
    "c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCNCC1", "c1ccsc1",
    "C1COCCN1", "C1CC1", "C1CCOC1", "c1ccoc1", "C1CCCC1",
]
_LINKERS = ["C(=O)N", "CC", "OC", "S(=O)(=O)N", "NC", "C#C", "COC", "CNC"]
_CAPS = ["C", "CC", "CCC", "F", "Cl", "N", "O", "OC", "C(F)(F)F", "C(C)C"]


def synthetic_dataset(
    n_train: int = 1800,
    n_test: int = 750,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> dict:
    """Generate (or reuse) the synthetic stand-in.

    Cached on disk because the smoke test spans three scripts: folds are built by one
    and consumed by another, so if each regenerated its own data the fold assignment
    would refer to different molecules than the models see. Compound names would still
    line up, and nothing would error — the split would just silently stop being a
    split. Delete artifacts/synthetic/ to regenerate.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        marker = cache_dir / "SEED"
        if marker.exists() and marker.read_text().strip() == f"{seed}:{n_train}:{n_test}":
            return {
                k: normalise(pd.read_csv(cache_dir / f"{k}.csv")) for k in C.DATA_FILES
            }
    out = _generate_synthetic(n_train=n_train, n_test=n_test, seed=seed)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            v.to_csv(cache_dir / f"{k}.csv", index=False)
        (cache_dir / "SEED").write_text(f"{seed}:{n_train}:{n_test}")
    return out


def _generate_synthetic(n_train: int, n_test: int, seed: int) -> dict:
    """Schema-identical fake data, for exercising the pipeline without the real files.

    Deliberately reproduces the two structural features that matter: the training set
    is sparse because only primary-screen hits got a curve, and credible intervals are
    much wider for weakly-active compounds.
    """
    rng = np.random.default_rng(seed)

    n_all = n_train + n_test
    n_iso = len(C.ISOFORMS)

    # Per-isoform additive contributions of each building block, so potency is a real
    # (if simple) function of structure and a model can actually learn it. Without this
    # the pipeline runs but every variant scores like noise, which tells you nothing.
    w_frag = rng.normal(0, 0.55, size=(len(_FRAGMENTS), n_iso))
    w_link = rng.normal(0, 0.40, size=(len(_LINKERS), n_iso))
    w_cap = rng.normal(0, 0.25, size=(len(_CAPS), n_iso))

    smiles, names = [], []
    latent = np.zeros((n_all, n_iso))
    for i in range(n_all):
        ia = int(rng.integers(len(_FRAGMENTS)))
        ib = int(rng.integers(len(_LINKERS)))
        ic = int(rng.integers(len(_FRAGMENTS)))
        idx = int(rng.integers(len(_CAPS)))
        smiles.append(
            f"{_FRAGMENTS[ia]}{_LINKERS[ib]}{_FRAGMENTS[ic].replace('1', '2')}{_CAPS[idx]}"
        )
        names.append(f"OCNT-{i:07d}")
        latent[i] = (
            4.0 + w_frag[ia] + w_frag[ic] + w_link[ib] + w_cap[idx] + rng.normal(0, 0.35, n_iso)
        )
    latent[:, 3] += 0.45  # CYP3A4 is the promiscuous one

    # Primary screen: log2 fold change anti-correlates with potency.
    sc_rows = []
    for j, cyp in enumerate(C.ISOFORMS):
        fc = -(latent[:, j] - 3.0) * 0.9 + rng.normal(0, 0.6, n_all)
        fdr = rng.beta(0.6, 8.0, n_all)
        sc_rows.append(
            pd.DataFrame(
                {
                    C.ID_COL: names,
                    C.SMILES_COL: smiles,
                    "enzyme": cyp,
                    "log2fc_estimate": fc,
                    "log2fc_fdr": fdr,
                }
            )
        )
    single_conc = pd.concat(sc_rows, ignore_index=True)

    is_hit = (single_conc["log2fc_estimate"] < C.HIT_LOG2FC_MAX) & (
        single_conc["log2fc_fdr"] < C.HIT_FDR_MAX
    )
    hit_map = {
        (r[C.ID_COL], r["enzyme"]): h
        for r, h in zip(single_conc.to_dict("records"), is_hit)
    }

    train_names = names[:n_train]
    test_names = names[n_train:]

    train = {C.ID_COL: train_names, C.SMILES_COL: smiles[:n_train]}
    for j, cyp in enumerate(C.ISOFORMS):
        ep = f"{cyp}_pIC50_direct_inhibition"
        vals, los, his, stds = [], [], [], []
        for i, nm in enumerate(train_names):
            if not hit_map[(nm, cyp)]:
                vals.append(np.nan); los.append(np.nan); his.append(np.nan); stds.append(np.nan)
                continue
            y = float(np.clip(latent[i, j] + rng.normal(0, 0.15), 2.5, 9.0))
            # Weak compounds get much wider bands, as in the real curve fits.
            # Calibrated against the OpenADMET PXR unblinded set, where the median
            # credible-interval width is ~1.47 below pIC50 3.5 and ~0.28 above 5.5.
            half = float(np.clip(0.73 - 0.196 * (y - 3.5), 0.09, 0.90)) * rng.uniform(0.8, 1.25)
            vals.append(y); los.append(y - half); his.append(y + half); stds.append(half / 1.96)
        train[ep] = vals
        train[ep + C.CONF_LOW_SUFFIX] = los
        train[ep + C.CONF_HIGH_SUFFIX] = his
        train[ep + "_std"] = stds
    train_inhibition = pd.DataFrame(train)

    tdi = {C.ID_COL: train_names, C.SMILES_COL: smiles[:n_train]}
    for j, cyp in enumerate(C.ISOFORMS):
        direct = train_inhibition[f"{cyp}_pIC50_direct_inhibition"].to_numpy()
        shift = rng.gamma(1.2, 0.28, n_train) * (rng.random(n_train) < 0.35)
        tdi[f"{cyp}_pIC50_direct_inhibition"] = direct
        tdi[f"{cyp}_pIC50_TDI_condition"] = direct + shift
    for cyp in C.TDI_ISOFORMS:
        d = np.asarray(tdi[f"{cyp}_pIC50_direct_inhibition"], dtype=float)
        t = np.asarray(tdi[f"{cyp}_pIC50_TDI_condition"], dtype=float)
        label = np.where(
            np.isnan(d) | np.isnan(t),
            np.nan,
            np.where(
                (d < C.PIC50_FLOOR) & (t < C.PIC50_FLOOR),
                0.0,
                ((t - d) > np.log10(2)).astype(float),
            ),
        )
        tdi[f"{cyp}_is_TDI"] = label
    train_tdi = pd.DataFrame(tdi)

    emax = {C.ID_COL: train_names, C.SMILES_COL: smiles[:n_train]}
    for cyp in C.ISOFORMS:
        emax[f"{cyp}_EmaxVsPosCtrl_direct_inhibition"] = -rng.normal(1.0, 0.06, n_train)
    train_emax = pd.DataFrame(emax)

    test = pd.DataFrame({C.ID_COL: test_names, C.SMILES_COL: smiles[n_train:]})

    return {
        "train_inhibition": train_inhibition,
        "train_tdi": train_tdi,
        "train_emax": train_emax,
        "single_conc": single_conc,
        "test": test,
    }
