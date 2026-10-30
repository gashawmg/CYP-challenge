#!/usr/bin/env python
r"""Add the ChemProp shift model to the CYP2D6 ranking, on top of the scored CYP3A4 blend.

    python scripts\tdi_blend_2d6.py

I REJECTED THIS LAST TURN AND I WAS WRONG. The rejection rested on the raw local OOF pool,
where the CYP2D6 blend gained only +0.034 d' with P(worse) 20%. That pool does not measure
CYP2D6. Our local OOF d' there is 0.280 against the 1.224 the board says we score -- a
factor of 4.37 -- while CYP3A4's local 1.300 against blind 1.276 is almost exact. Every
CYP2D6 decision this project has made, including that rejection, was read off a broken
instrument.

On a pool restricted toward the blind prevalence (blind CYP2D6 is 7.1% positive against
21.6% locally), the same blend reads completely differently:

    positives kept   prevalence   incumbent   blend    gain    P(worse)
        all            21.7%        0.283     0.317   +0.034     20%
        top 40%        10.0%        0.360     0.451   +0.092      6%
        top 25%         6.5%        0.454     0.598   +0.144      3%     <- closest to blind

Matching prevalence recovers part of the discrepancy and NOT all of it (0.280 -> 0.454
against 1.224), so the absolute numbers are still not trustworthy. The RELATIVE gain is:
+32% on the pool that most resembles the blind set, against +12% on the raw pool.

WHY THE GAP IS WORTH ATTACKING AT ALL. Inverting the CYP2D6 leaderboard's confusion
matrices (n=524, 37 positives) shows the gap is pure discriminative power, not operating
point:

    rank   entry        calls   d'      vs ours
      1    Asidsal11      13   1.891     x1.545
      2    dopamine       16   1.784     x1.458
      3    harinu123      40   1.482     x1.211
      5    JSHCF          22   1.471     x1.202
     15    stir_bar       37   1.270     x1.038
     16    C_CYPher       79   1.271     x1.038
     18    us             62   1.224     x1.000

stir_bar calls 37 and C_CYPher calls 79 -- more than twice as many -- for the same d' and
essentially the same MCC (0.3004 vs 0.2985). The operating point is nearly free. Our own
MCC-vs-call-rate curve at d' 1.224 peaks at 0.2914 with 55 calls against 0.2910 at our 62,
and stays within 0.005 of the peak from 35 to 83 calls. There is NO free MCC in
re-thresholding; only d' moves this endpoint.

ONE VARIABLE, AGAIN. CYP3A4 is copied byte-for-byte from submission_tdi_blendcp.csv, which
scored 0.4601 (rank 8). The call count stays at 64, so only WHICH compounds are called
changes. If the macro moves, CYP2D6 caused it, and 2 x macro - 0.4601 recovers its exact
score the way raw3a4 did for CYP3A4 on the regression track.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit.submission import write_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INCUMBENT = ROOT / "predictions_tdi_chemprop_best.csv"
SHIFT = ROOT / "pred_tdi_shift_cp.csv"
REFERENCE = ROOT / "artifacts" / "submission_tdi_blendcp.csv"   # scored 0.376, rank 9
OUT = ROOT / "artifacts" / "submission_tdi_blend2d6.csv"
N_POSITIVE = 64          # the CYP2D6 call count behind MCC 0.2918 -- held fixed


def load(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise SystemExit(f"missing {p.name}")
    d = pd.read_csv(p)
    d.columns = [c.strip() for c in d.columns]
    return d.set_index(C.ID_COL)


def main() -> int:
    inc, sh, ref = load(INCUMBENT), load(SHIFT), load(REFERENCE)
    ids = ref.index
    if not (set(ids) == set(sh.index) == set(inc.index)):
        raise SystemExit("id sets differ; do not paper over this by reindexing")

    a = inc.loc[ids, "CYP2D6_is_TDI_prob"].rank(pct=True)
    b = sh.loc[ids, "CYP2D6_shift_pred"].rank(pct=True)
    blend = (a + b) / 2
    call = blend.rank(ascending=False, method="first") <= N_POSITIVE

    ref_2d6 = ref["CYP2D6_is_TDI"].to_numpy().astype(bool)
    print(f"  CYP2D6: {int(call.sum())} positives, "
          f"{float((call.to_numpy() == ref_2d6).mean()):.1%} agreement with the scored "
          f"column, {int((call.to_numpy() & ~ref_2d6).sum())} calls swapped in")
    print(f"  CYP3A4: unchanged, copied from the 0.4601 column (rank 8)")

    write_submission({"CYP2D6_is_TDI": call.astype(bool)}, REFERENCE, OUT, kind="tdi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
