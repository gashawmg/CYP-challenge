#!/usr/bin/env python
"""One place that writes submission files, so a file can only leave here well-formed.

WHY THIS EXISTS. submission_blend4z.csv was rejected by the platform for missing the
SMILES column. The builder had done `set_index("Molecule_Name")`, assembled predictions
into a fresh DataFrame, and reset the index -- which silently dropped SMILES, because
nothing in that path ever asserted the output looked like the file it was derived from.
The TDI splice script had the identical bug and it was caught only by eyeballing a column
listing. Eyeballing is not a check, so the check lives here now.

WHAT THE OFFICIAL VALIDATOR REQUIRES (validation/activity_validation.py and
tdi_validation.py in the challenge tutorial repo):

    required_id_columns    = ("SMILES", "Molecule_Name")     <- BOTH, and SMILES first
    required_value_columns = the four pIC50 columns, or the two _is_TDI columns
    750 rows, no duplicate Molecule_Name, no NaN or inf in any value column

THE RULE HERE: a submission is always written as a COPY OF A REFERENCE submission that
the platform has already accepted, with only the value columns replaced. Layout, id
columns, dtypes and row order therefore cannot drift, and anything the reference carried
that we did not think about is carried too.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
REGRESSION_COLUMNS = tuple(f"{i}_pIC50_direct_inhibition" for i in ISOFORMS)
TDI_COLUMNS = ("CYP3A4_is_TDI", "CYP2D6_is_TDI")
ID_COLUMNS = ("SMILES", "Molecule_Name")
N_ROWS = 750


def write_submission(values: dict[str, pd.Series], reference: str | Path,
                     out: str | Path, kind: str = "regression",
                     verbose: bool = True) -> pd.DataFrame:
    """Write `values` into a copy of `reference` and check it before it hits disk.

    values     column name -> Series indexed by Molecule_Name
    reference  a submission the platform ACCEPTED; supplies the layout
    kind       "regression" or "tdi"
    """
    required = REGRESSION_COLUMNS if kind == "regression" else TDI_COLUMNS
    ref = pd.read_csv(reference)
    ref.columns = [c.strip() for c in ref.columns]

    missing_ref = [c for c in (*ID_COLUMNS, *required) if c not in ref.columns]
    if missing_ref:
        raise ValueError(f"reference {Path(reference).name} is not a valid {kind} "
                         f"submission -- missing {missing_ref}")

    out_df = ref.copy()
    idx = out_df["Molecule_Name"]
    for col, series in values.items():
        if col not in required:
            raise ValueError(f"{col} is not a {kind} value column {required}")
        s = pd.Series(series)
        missing = set(idx) - set(s.index)
        if missing:
            raise ValueError(f"{col}: no value for {len(missing)} molecules, "
                             f"e.g. {sorted(missing)[:5]}")
        out_df[col] = s.reindex(idx).to_numpy()

    # --- the checks the platform will run, run here first --------------------------
    errs: list[str] = []
    if list(out_df.columns) != list(ref.columns):
        errs.append(f"column layout drifted: {list(out_df.columns)} "
                    f"vs reference {list(ref.columns)}")
    if len(out_df) != N_ROWS:
        errs.append(f"{len(out_df)} rows, expected {N_ROWS}")
    for c in ID_COLUMNS:
        if out_df[c].isna().any():
            errs.append(f"{c} has {int(out_df[c].isna().sum())} missing values")
    if out_df["Molecule_Name"].duplicated().any():
        errs.append(f"{int(out_df['Molecule_Name'].duplicated().sum())} duplicate ids")
    for c in required:
        if kind == "regression":
            v = pd.to_numeric(out_df[c], errors="coerce").to_numpy(float)
            if not np.isfinite(v).all():
                errs.append(f"{c}: {int((~np.isfinite(v)).sum())} NaN/inf values")
        elif out_df[c].dtype != bool:
            errs.append(f"{c}: dtype {out_df[c].dtype}, expected bool")
    if errs:
        raise ValueError("submission would be rejected:\n  - " + "\n  - ".join(errs))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    if verbose:
        print(f"wrote {Path(out).name}: {len(out_df)} rows, layout matches "
              f"{Path(reference).name}, all checks passed")
    return out_df
