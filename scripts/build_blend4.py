#!/usr/bin/env python
"""Build the 4-model diversity blend, placed by RATIO TRANSFER rather than by copying.

THE RANKING. Equal-weight z-average of four models that disagree with each other:

    v6      ChemProp D-MPNN, single-task, censored-aware      the incumbent
    mh      13-head multi-head D-MPNN + external + screen     different supervision
    st_v3   ChemProp v3                                       different training era
    unimol  Uni-Mol 3D conformer transformer                  different REPRESENTATION

Blind-matched macro rho: v6 alone 0.6181, v6+v7 0.6341, v6+v8 0.6430, v6+mh 0.6544,
all four 0.6722. The ordering follows member DISSIMILARITY exactly, which is the
signature of variance reduction. Nothing is fitted -- equal weights, no search.

THE PLACEMENT, AND WHY NOT JUST COPY splice_best's.
The first version of this script mapped each endpoint onto splice_best's exact mean and
sd, reasoning that inheriting a placement measured at 0.5167 changes only one variable.
Measured on the blind-matched pool, that construction is WORSE than v6:

    at splice_best's placement     macro rho 0.6722   macro ST-RAE 0.8158
    v6, same placement             macro rho 0.6181   macro ST-RAE 0.7546

A four-model average is regressed further toward the mean than any member, so its
optimal spread is narrower; stretching it to a single model's spread throws away more
than the better ranking wins. With placement refit per model the gain is real and large
(macro ST-RAE: v6 0.5976, mh 0.5959, this blend 0.5256).

But the local fit's ABSOLUTE spreads cannot be shipped: it wants 0.71-0.85 for v6, while
splice_best actually submitted 0.86-1.21 and scored well. The local pool is matched to
the blind LABEL histogram, not to the blind chemistry, and it systematically prefers
narrower spreads for every model. What survives that bias is the RATIO between two
models fitted on the same pool. So:

    spread = splice_best's submitted spread  x  (blend's fitted spread / v6's fitted)
    mean   = splice_best's submitted mean     (unchanged; the fits agree on centre)

    endpoint   splice_best sd   blend/v6 ratio   shipped sd
    CYP1A2         1.211            1.04            1.26
    CYP2C9         0.858            0.80            0.69
    CYP2D6         0.911            1.33            1.21
    CYP3A4         0.979            0.98            0.96

WHAT THIS SUBMISSION BUYS EVEN IF THE SCORE DOES NOT MOVE. rho is invariant to any
monotone placement, so the leaderboard's per-target rho reads the blend's TRUE blind
ranking regardless of whether the spreads above are right. That single number settles
the question no local experiment can: whether ensemble gains transfer to the blind deck
at all. On our own compounds these four models agree at rho 0.44-0.88; on the 750 blind
compounds they agree at 0.71-0.95, so the local gain is measured where they disagree
most and cannot be trusted at full size.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from cypkit import config as C
from cypkit.submission import write_submission

ROOT = Path(__file__).resolve().parent.parent

MEMBERS = {"v6": "pred_st_v6.csv", "mh": "pred_st_mh.csv",
           "cp": "predictions_cyp_st_v3.csv", "um": "predictions_unimol_cyp_mt_test.csv"}
REFERENCE = "artifacts/submission_splice_best.csv"     # placement donor, scored 0.5167
# blend-vs-v6 spread ratio, fitted per endpoint on the blind-matched pool
SPREAD_RATIO = {"CYP1A2": 1.04, "CYP2C9": 0.80, "CYP2D6": 1.33, "CYP3A4": 0.98}
OUT = "artifacts/submission_blend4z.csv"


def load(p):
    d = pd.read_csv(p); d.columns = [x.strip() for x in d.columns]
    return d.set_index(C.ID_COL)


def main() -> int:
    F = {k: load(v) for k, v in MEMBERS.items()}
    ref = load(REFERENCE)
    ids = ref.index
    for k, v in F.items():
        assert set(ids) == set(v.index), f"{k}: id set differs from the reference"
        F[k] = v.loc[ids]

    values = {}
    print(f"{'endpoint':9s} {'mean':>7s} {'sd':>7s} {'ref sd':>7s} {'ratio':>6s}"
          f"   rho vs splice_best")
    for iso in C.ISOFORMS:
        ep = f"{iso}_pIC50_direct_inhibition"
        z = [(F[k][ep] - F[k][ep].mean()) / F[k][ep].std() for k in F]
        s = sum(z) / len(z)
        s = (s - s.mean()) / s.std()
        sd = ref[ep].std() * SPREAD_RATIO[iso]
        values[ep] = s * sd + ref[ep].mean()
        r = values[ep].corr(ref[ep], method="spearman")
        print(f"{iso:9s} {values[ep].mean():7.3f} {values[ep].std():7.3f} "
              f"{ref[ep].std():7.3f} {SPREAD_RATIO[iso]:6.2f}   {r:.3f}")

    # write_submission copies the ACCEPTED reference file and replaces only the value
    # columns, so SMILES -- which an earlier version of this script silently dropped,
    # and which the official validator requires -- cannot go missing again.
    write_submission(values, ROOT / REFERENCE, ROOT / OUT, kind="regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
