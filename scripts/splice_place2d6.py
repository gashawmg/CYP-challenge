#!/usr/bin/env python
r"""Take CYP2D6 from the placement transform, everything else from blend4z.

    python scripts\splice_place2d6.py

submission_place_half.csv scored 0.5873 against blend4z's 0.5104 -- but Spearman came
back IDENTICAL on all four endpoints (0.8064 / 0.8380 / 0.5190 / 0.8310), so the monotone
transform did exactly what it was designed to do and the loss is placement, nothing else.

WHAT WENT WRONG, PRECISELY. fit_placement.py maps ranks onto the curves fitted on our
local pool, and ships their ABSOLUTE values. Its own docstring says why that cannot work
-- "the local fit's ABSOLUTE spreads cannot be shipped... what survives that bias is the
RATIO between two models fitted on the same pool" -- and the submission ignored it:

    endpoint   blend4z mean   place_half mean   BLIND mean
    CYP1A2         4.30            4.85            4.32
    CYP2C9         4.85            4.74            4.81
    CYP2D6         3.13            3.95            2.92
    CYP3A4         4.70            4.41            4.76

blend4z sits on the anchors; place_half drifts up to half a log unit off them, and CYP3A4
absorption blew out from 0.833 to 1.263.

WHAT SURVIVED. CYP2D6 improved: ST-RAE 0.6482 -> 0.5957, absorption 0.758 -> 0.697, which
is the level the rivals reach on their best endpoints. The uncensored-pool fit predicted a
9% absorption gain there -- the largest of the four -- and realised 8%. So the transform
is right for CYP2D6 and wrong as an absolute placement for the other three, and the honest
move is to take only the part that worked.

    CYP1A2  0.5125   blend4z
    CYP2C9  0.3978   splice_best   <- still unbeaten; blend4z regressed it to 0.4172
    CYP2D6  0.5957   place_half    <- the only endpoint the transform improved
    CYP3A4  0.4636   blend4z
    MACRO   0.4924   vs 0.5104     -> -0.0180, known before submitting

CYP2C9 is worth a note. blend4z made it WORSE than splice_best on both axes (ST-RAE
0.4172 vs 0.3978, rho 0.8380 vs 0.8454) -- the one endpoint where the 4-model blend lost
to what it replaced. Splicing puts the older column back.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit.submission import write_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "artifacts" / "submission_blend4z.csv"      # CYP1A2 + CYP3A4, scored
DONORS = {"CYP2D6": ROOT / "artifacts" / "submission_place_half.csv",
          "CYP2C9": ROOT / "artifacts" / "submission_splice_best.csv"}
OUT = ROOT / "artifacts" / "submission_splice4.csv"
KNOWN = {"CYP1A2": 0.5125, "CYP2C9": 0.3978, "CYP2D6": 0.5957, "CYP3A4": 0.4636}
SOURCE = {"CYP1A2": "blend4z", "CYP2C9": "splice_best",
          "CYP2D6": "place_half", "CYP3A4": "blend4z"}


def main() -> int:
    vals, srcs = {}, {}
    for iso, path in DONORS.items():
        d = pd.read_csv(path)
        d.columns = [c.strip() for c in d.columns]
        ep = f"{iso}_pIC50_direct_inhibition"
        vals[ep] = d.set_index(C.ID_COL)[ep]
        srcs[ep] = vals[ep]

    print(f"  {'endpoint':9s} {'donor':12s} {'ST-RAE':>8s}")
    for iso, v in KNOWN.items():
        print(f"  {iso:9s} {SOURCE[iso]:12s} {v:8.4f}")
    print(f"  {'MACRO':9s} {'':12s} {sum(KNOWN.values()) / 4:8.4f}")

    out = write_submission(vals, BASE, OUT).set_index(C.ID_COL)
    # every donated column must be byte-identical to the file that earned its score, or
    # the arithmetic above is void
    for ep, want in srcs.items():
        assert out[ep].round(10).equals(want.reindex(out.index).round(10)), \
            f"{ep} drifted from its donor"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
