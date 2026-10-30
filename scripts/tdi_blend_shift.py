#!/usr/bin/env python
r"""Rank-blend the shift model into the incumbent TDI ranking, at the SAME call rates.

    python scripts\tdi_blend_shift.py

WHY, WHEN THE SHIFT MODEL LOSES STANDALONE. Out of fold, on the same compounds and the
same labels, it adds to the incumbent on BOTH isoforms even though it is a weaker model
on one of them:

    CYP3A4   incumbent d' 0.904   shift d' 0.878   BLEND d' 0.995   MCC 0.2772 -> 0.3086
    CYP2D6   incumbent d' 0.283   shift d' 0.282   BLEND d' 0.357   MCC 0.0430 -> 0.0834

CYP2D6 is the interesting one: the shift target bought nothing standalone there (+0.000)
and still adds +0.074 to the blend. Different target, different backbone, different
errors -- which is the only mechanism that has ever produced a real gain in this project.

ONE VARIABLE. The call rates are pinned to the ones that produced our scored columns --
CYP3A4 158/750 as in submission_tdi_best.csv (MCC 0.4470), CYP2D6 64/750 as in
submission_tdi_v3.csv (MCC 0.2918). So the only thing that changes is WHICH compounds are
called, not how many. scripts/tdi_diagnose.py already showed the MCC-vs-rate curve is flat
near the optimum and both targets sit on it, so moving the rate as well would risk more
than it could win.

WHAT THIS IS NOT. It is not a known score. The two spliced columns it replaces are worth
0.3694 by arithmetic; this is a genuine bet on a ranking change, and the pIC50 side has
already shown that local ensemble gains can transfer at as little as 14%. Expected value
is positive but the honest range is wide -- roughly 0.37 to 0.40, against 0.3694 held.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit.submission import write_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIFT = ROOT / "pred_tdi_shift.csv"
INCUMBENT = ROOT / "predictions_tdi_chemprop_best.csv"
REFERENCE = ROOT / "artifacts" / "submission_tdi_splice_best.csv"   # scored 0.3694
OUT = ROOT / "artifacts" / "submission_tdi_blend.csv"
# positives held at the counts of the columns that earned our best per-target MCCs
N_POSITIVE = {"CYP3A4": 158, "CYP2D6": 64}


def load(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p)
    d.columns = [c.strip() for c in d.columns]
    return d.set_index(C.ID_COL)


def main() -> int:
    sh, inc, ref = load(SHIFT), load(INCUMBENT), load(REFERENCE)
    ids = ref.index
    assert set(ids) == set(sh.index) == set(inc.index), "id sets differ"

    values = {}
    print(f"  {'target':8s} {'positives':>10s} {'rate':>7s} "
          f"{'agreement with the scored column':>34s}")
    for iso, n_pos in N_POSITIVE.items():
        a = inc.loc[ids, f"{iso}_is_TDI_prob"].rank(pct=True)
        b = sh.loc[ids, f"{iso}_shift_pred"].rank(pct=True)
        blend = (a + b) / 2
        thr = blend.sort_values(ascending=False).iloc[n_pos - 1]
        call = (blend >= thr)
        # keep exactly n_pos in case of ties at the threshold
        if call.sum() != n_pos:
            call = blend.rank(ascending=False, method="first") <= n_pos
        col = f"{iso}_is_TDI"
        values[col] = call.astype(bool)
        same = float((call.to_numpy() == ref[col].to_numpy().astype(bool)).mean())
        print(f"  {iso:8s} {int(call.sum()):10d} {call.mean():7.3f} {same:33.1%}")

    write_submission(values, REFERENCE, OUT, kind="tdi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
