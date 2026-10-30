#!/usr/bin/env python
r"""Add the ChemProp shift model to the CYP3A4 TDI ranking. CYP2D6 is left untouched.

    python scripts\tdi_blend_cp.py

WHY THIS ONE IS DIFFERENT FROM EVERY OTHER TDI LEVER WE HAVE TRIED. Target
reformulation, two-arm differencing, operating-point moves and binary-only augmentation
all produced gaps inside their own confidence intervals (STRATEGY.md Section 12). This is the
first TDI change whose paired bootstrap excludes zero:

    CYP3A4   d' 0.904 -> 1.008  (+0.103)  95% CI [+0.055, +0.154]  P(worse)  0%
    CYP2D6   d' 0.283 -> 0.317  (+0.034)  95% CI [-0.047, +0.114]  P(worse) 20%

and the CYP3A4 result is not an artefact of picking the best of several blends. A greedy
member search re-run inside 40 split-halves -- selecting members on one half and scoring
on the other -- keeps +0.096 [+0.073, +0.120] with P(worse) 0%. Almost none of the
in-sample gain was selection bias.

WHY THE SHIFT MODEL EARNS A PLACE DESPITE LOSING STANDALONE. It scores d' 0.889 against
the incumbent's 0.904, so it is the weaker model. What it contributes is DISAGREEMENT:
the two rankings correlate at only 0.644, because one is trained on the binary label and
the other regresses the continuous shift between the two assay arms. Errors that do not
coincide cancel when the ranks are averaged. Every real gain this project has produced,
on either track, has come from that mechanism and from nothing else.

WHY CYP2D6 IS NOT TOUCHED. Two independent reasons, either of which is sufficient. Its
bootstrap straddles zero, and the split-half selection test actively reversed (-0.016,
P(worse) 65%). More fundamentally, local CYP2D6 evidence is untrustworthy in both
directions: our OOF d' there is 0.283, barely above chance, while the blind MCC of 0.2918
implies d' 1.224. A local yardstick that disagrees with the board by that margin cannot
license replacing 88% of the calls in a column that is already scoring.

ONE VARIABLE. The incumbent member is `chemprop_best` -- the exact model behind the
CYP3A4 column that scored MCC 0.4470 -- not the marginally better `chemprop_avg` the
greedy search preferred (d' 1.009 vs 1.008, a tie). Swapping the base model AND adding a
member would be two changes, and we would not know which one moved the board. The call
rate stays pinned at 158/750, so the only thing that changes is WHICH compounds are
called, never how many.

MEASURE MCC AT THE RATE YOU ACTUALLY SUBMIT AT. MCC depends on the call rate, and
train_tdi_shift.py's CALL_RATE for CYP3A4 is 0.174 -- the local label prevalence, not the
158/750 = 0.2107 our scored column calls at. Scoring the blend at the rate we submit at
roughly doubles the measured gain:

    call rate   incumbent   blend    delta      P(MCC worse)
      0.1740      0.2772   0.2965   + 7.0%          11%
      0.2107      0.2752   0.3178   +15.5%           1%     <- the rate we submit at

WHAT TO EXPECT. Our blind CYP3A4 MCC is 0.4470 and CYP2D6 holds at 0.2918. Full transfer
of +15.5% gives CYP3A4 0.516 and macro 0.404; the historical CYP3A4 transfer shortfall
(we have overshot on this endpoint twice -- predicted d' 1.427 realised 1.135, predicted
MA-MCC 0.3935 scored 0.3496) puts the conservative end near macro 0.381, against 0.3694
held. This is a bet, not arithmetic, and the range is wide -- but it is the first TDI bet
whose local evidence survives both a paired bootstrap and a split-half selection test.
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
SHIFT = ROOT / "pred_tdi_shift_cp.csv"              # chemprop backbone, --predict
REFERENCE = ROOT / "artifacts" / "submission_tdi_splice_best.csv"    # scored 0.3694
OUT = ROOT / "artifacts" / "submission_tdi_blendcp.csv"
N_POSITIVE = 158          # the CYP3A4 call count behind MCC 0.4470 -- held fixed


def load(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise SystemExit(f"missing {p.name}; run train_tdi_shift.py --backbone chemprop "
                         f"--iso both --predict pred_tdi_shift_cp.csv first")
    d = pd.read_csv(p)
    d.columns = [c.strip() for c in d.columns]
    return d.set_index(C.ID_COL)


def main() -> int:
    inc, sh, ref = load(INCUMBENT), load(SHIFT), load(REFERENCE)
    ids = ref.index
    if not (set(ids) == set(sh.index) == set(inc.index)):
        raise SystemExit("id sets differ between the reference, the incumbent and the "
                         "shift predictions -- do not paper over this by reindexing")

    a = inc.loc[ids, "CYP3A4_is_TDI_prob"].rank(pct=True)
    b = sh.loc[ids, "CYP3A4_shift_pred"].rank(pct=True)
    blend = (a + b) / 2
    # rank(method="first") breaks ties deterministically, so the count is exactly n
    call = blend.rank(ascending=False, method="first") <= N_POSITIVE

    ref_3a4 = ref["CYP3A4_is_TDI"].to_numpy().astype(bool)
    same = float((call.to_numpy() == ref_3a4).mean())
    swapped = int((call.to_numpy() & ~ref_3a4).sum())
    print(f"  CYP3A4: {int(call.sum())} positives ({call.mean():.3f}), "
          f"{same:.1%} agreement with the scored column, {swapped} calls swapped in")
    print(f"  CYP2D6: unchanged, {int(ref['CYP2D6_is_TDI'].astype(bool).sum())} positives "
          f"copied straight from the 0.3694 submission")

    # CYP2D6 is deliberately absent from `values`: write_submission copies the reference
    # column through untouched, which is stronger than recomputing something identical.
    write_submission({"CYP3A4_is_TDI": call.astype(bool)}, REFERENCE, OUT, kind="tdi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
