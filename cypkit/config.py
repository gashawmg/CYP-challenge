"""Shared constants and path resolution for the CYP challenge project.

Nothing here is imported from the challenge repo; `scoring.py` does that at call time
so this module stays importable even if the tutorial repo has moved.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
TDI_ISOFORMS = ("CYP3A4", "CYP2D6")

REGRESSION_ENDPOINTS = tuple(f"{cyp}_pIC50_direct_inhibition" for cyp in ISOFORMS)
CLASSIFICATION_ENDPOINTS = tuple(f"{cyp}_is_TDI" for cyp in TDI_ISOFORMS)

CONF_LOW_SUFFIX = "_conf_low"
CONF_HIGH_SUFFIX = "_conf_high"

ID_COL = "Molecule_Name"
SMILES_COL = "SMILES"

# ---------------------------------------------------------------------------
# Assay constants
# ---------------------------------------------------------------------------

# Below this pIC50 the dose-response fit is outside the reliable range. The challenge
# blog uses 4.0 as the floor when defining TDI labels; we reuse it as the left-censoring
# bound for compounds that were screened but never promoted to a curve.
PIC50_FLOOR = 4.0

# Primary-screen hit definition, straight from the tutorial notebook.
HIT_LOG2FC_MAX = -1.0
HIT_FDR_MAX = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CYP_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACT_DIR = Path(os.environ.get("CYP_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))

DATA_FILES = {
    "train_inhibition": "cyp-challenge-TRAIN_inhibition.csv",
    "train_tdi": "cyp-challenge-TRAIN_TDI.csv",
    "train_emax": "cyp-challenge-TRAIN_Emax.csv",
    "single_conc": "cyp-challenge-single-concentration-TRAIN.csv",
    "test": "cyp-challenge-TEST-BLINDED.csv",
}


def _default_challenge_repo() -> Path:
    """Best guess at where the tutorial repo (with `evaluation/`) lives."""
    env = os.environ.get("CYP_CHALLENGE_REPO")
    if env:
        return Path(env)
    # The GitHub zip unpacks to a doubled directory name; check both shapes, and
    # walk up a couple of levels so this works from anywhere under the project.
    outer = PROJECT_ROOT.parent / "CYP-Challenge-Tutorial-main"
    for candidate in (outer / "CYP-Challenge-Tutorial-main", outer):
        if (candidate / "evaluation" / "config.py").exists():
            return candidate
    return outer / "CYP-Challenge-Tutorial-main"


CHALLENGE_REPO = _default_challenge_repo()


def artifact(*parts: str) -> Path:
    """Path under the artifacts dir, creating parent directories as needed."""
    p = ARTIFACT_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
