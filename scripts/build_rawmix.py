#!/usr/bin/env python
r"""Keep the members' ACTUAL pIC50 estimates where that wins, placed columns where it does not.

    python scripts\build_rawmix.py

THE LEADERBOARD NEIGHBOURHOOD SAID IT FIRST. At 0.4924 (rank 19) we sit among six models
scoring 0.4828-0.4902, and the comparison is lopsided:

    who                ST-RAE     MAE      R2     rho
    admetchallenger    0.4828  0.6543  0.5827  0.7285
    blue               0.4873  0.6146  0.6176  0.7405
    EleanorRigby       0.4902  0.6532  0.5873  0.7668
    Gashaw             0.4924  0.6996  0.5381  0.7504   <- WORST MAE, WORST R2, 2nd-best rho

Right order, wrong values. And that is something our pipeline does deliberately: we
rank-standardise the members and map the ranks onto fitted anchors, discarding the pIC50
estimates the models were trained to produce. On the calibrated pool, per endpoint:

    endpoint   RAW member mean            z-avg -> anchors (what we ship)
               MAE     ST-RAE   abs       MAE     ST-RAE   abs
    CYP1A2    0.8570   0.6570  0.825     0.8099   0.7569  0.953   <- RAW wins
    CYP2C9    0.5647   0.5909  0.861     0.5209   0.5236  0.763   <- placed wins
    CYP2D6    1.5375   1.6210  1.874     1.0199   1.2091  1.398   <- placed wins
    CYP3A4    0.5741   0.4655  0.829     0.7214   0.6518  1.155   <- RAW wins, by 29%

rho is unchanged to three decimals in every case, so this is purely about the values.
CYP3A4 is the prize: keeping the raw estimates cuts its absorption from 1.155 to 0.829.

WHY THIS TRANSFER IS SAFER THAN THE LAST ONE. submission_place_half.csv failed because it
shipped ABSOLUTE values fitted on the local pool. This ships a per-endpoint CHOICE between
two constructions of the same underlying models, decided by an A/B on identical rows with
identical weights. The quantity carried across is a within-endpoint ratio, not a fitted
level.

    CYP1A2  raw mean of v6/mh/cp/unimol         0.5125 x (0.6570/0.7569) ~ 0.445
    CYP2C9  splice_best column, unchanged       0.3978
    CYP2D6  place_half column, unchanged        0.5957
    CYP3A4  raw mean of v6/mh/cp/unimol         0.4636 x (0.4655/0.6518) ~ 0.331
    MACRO                                       ~0.44 against 0.4924 held

That projection is a ratio transfer, NOT arithmetic -- the two unchanged columns are known,
the two new ones are a bet.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402
from cypkit import config as C  # noqa: E402
from cypkit.submission import write_submission  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "artifacts" / "submission_splice4.csv"      # scored 0.4924
MEMBERS = ["pred_st_v6.csv", "pred_st_mh.csv", "predictions_cyp_st_v3.csv",
           "predictions_unimol_cyp_mt_test.csv"]
# ONE ENDPOINT, NOT TWO. CYP3A4 carries the bigger local effect (-29% vs CYP1A2's -13%)
# and isolating it leaves THREE of four columns at known values, so the readout is clean:
# the raw construction is severely compressed (sd 0.75 against the 1.25 the blind set
# wants) and the one narrow CYP3A4 we ever shipped scored 0.7027. This bet tests the
# compression question on one endpoint instead of betting the macro on it.
RAW_ENDPOINTS = ("CYP3A4",)


def load(p):
    d = pd.read_csv(p); d.columns = [c.strip() for c in d.columns]
    return d.set_index(C.ID_COL)


def main() -> int:
    base = load(BASE)
    F = [load(ROOT / m) for m in MEMBERS]
    ids = base.index
    vals = {}
    print(f"  {'endpoint':9s} {'source':34s} {'mean':>7s} {'sd':>7s}")
    for iso in C.ISOFORMS:
        ep = f"{iso}_pIC50_direct_inhibition"
        if iso in RAW_ENDPOINTS:
            v = sum(f.loc[ids, ep] for f in F) / len(F)
            vals[ep] = v
            src = "RAW mean of 4 members"
        else:
            v, src = base[ep], "unchanged from splice4 (scored)"
        print(f"  {iso:9s} {src:34s} {v.mean():7.3f} {v.std():7.3f}")
    write_submission(vals, BASE, ROOT / "artifacts" / "submission_raw3a4.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
