#!/usr/bin/env python
r"""Score OOF files side by side on the fitted rows, and show the two defects that matter.

    python scripts/compare_oof.py
    python scripts/compare_oof.py --oof oof_clf_s42.csv oof_chemprop_st_v3_ens5_s42.csv
    python scripts/compare_oof.py --iso CYP1A2 CYP2D6

WHY THESE COLUMNS AND NOT rho ALONE. The training script prints OOF rho and R2 on the
fitted rows and those numbers look alarming -- rho 0.44 on CYP1A2 against a board 0.80.
Almost none of that gap is the model. The fitted pool is dose-response curves, which were
only run on primary-screen HITS, so it is a narrow already-active band: CYP1A2 fitted
labels have sd 0.628 against a blind sd of 1.431. Ranking within a narrow band is hard and
ranking across the blind range is easy, so LOCAL OOF rho UNDERSTATES BOARD rho by a lot,
and by a different amount per endpoint (+0.36 CYP1A2, +0.35 CYP2C9, +0.11 CYP2D6, +0.08
CYP3A4). Read it as a between-run comparison, never as a forecast.

R2 IS THE MOST POOL-SENSITIVE NUMBER THIS PROJECT PRINTS, AND IT KEEPS CAUSING ALARM.
R2 = 1 - MSE/Var(y), so it depends on the SPREAD OF THE ROWS YOU SCORE ON as much as on the
model. Our fitted pool is narrow; the blind set is wide. The 2026-09-27 run, unchanged, put
through both denominators:

    endpoint   MSE    var(fitted)  R2 local | var(blind)  R2 at the SAME MSE | board R2
    CYP1A2    0.532      0.395      -0.350  |   2.048            +0.740      |  0.6213
    CYP2C9    0.294      0.302      +0.028  |   1.259            +0.767      |  0.6986
    CYP3A4    0.544      1.195      +0.545  |   1.568            +0.653      |  0.7307
    CYP2D6    0.445      0.415      -0.072  |   2.729            +0.837      |  0.2151

Three of four land within ~0.12 of the board once the denominator is corrected. A negative
local R2 and a +0.62 board R2 are THE SAME MODEL MAKING THE SAME ERRORS. Do not steer on
local R2, and do not read it as "worse than predicting the mean" -- it means "worse than
predicting the mean OF A BAND 0.63 LOG UNITS WIDE", which is a much easier thing to beat.

CYP2D6 IS THE EXCEPTION AND IT IS A REAL FINDING. The rescaling predicts 0.837 and the
board gives 0.2151 -- a 0.62 miss where the others miss by 0.12. Our blind error on CYP2D6
is far worse than our local error, which is what you would expect when the blind
distribution sits ~2 log units below the training pool (blind mean 2.92 against a fitted
mean of 4.98). The local pool flatters CYP2D6 specifically, and only CYP2D6.

Two columns here ARE real defects and are not artifacts of the pool:

  bias      mean(pred) - mean(y) over the FITTED rows. Every chemprop variant in this
            project predicts CYP1A2 about 0.35 low, and that single constant is where the
            negative R2 comes from: removing it takes CYP1A2 from -0.350 to -0.041 and
            CYP2D6 from -0.072 to +0.116. A constant offset is not a ranking problem, which
            is why rho never flagged it.

  cens>bnd  the fraction of LEFT-CENSORED rows -- compounds the assay measured as having no
            quantifiable activity -- predicted ABOVE their own bound. It ran 83 / 73 / 94 /
            53% on the 2026-09-27 run. The loss already gives a censored row zero cost for
            any prediction below its bound, so this is not a loss bug: the model cannot
            separate actives from inactives and settles between them, which drags the
            actives down (the bias above) and leaves the inactives up. One defect, two
            symptoms.

  sd ratio  sd(pred)/sd(y) against the optimum, which is roughly the correlation, not 1.
            Predicting with a spread wider than your accuracy justifies costs R2 outright.

THIS IS THE FILE TO RUN AFTER ANY ARCHITECTURE CHANGE aimed at the separation problem
(--multitask_clf above all). If cens>bnd does not fall, the change did not do the thing it
was built to do, whatever happened to rho.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

from cypkit import config as C                                        # noqa: E402

# Board rho on the full blind set for the live submission, for scale only.
BOARD_RHO = {"CYP1A2": 0.8004, "CYP2C9": 0.8420, "CYP2D6": 0.5342, "CYP3A4": 0.8593}
DEFAULT = ["oof_chemprop_st_v3_ens5_s42.csv", "oof_chemprop_st_v3.csv",
           "oof_chemprop_st_v3_lean.csv", "oof_chemprop_st_v3_spec.csv",
           "oof_st_mh.csv", "oof_unimol_cyp_mt.csv"]


def score(path: Path, lab: pd.DataFrame, isos):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    if C.ID_COL not in d.columns:
        return None
    rows = []
    for iso in isos:
        col = f"{iso}_pIC50_direct_inhibition"
        if col not in d.columns:
            continue
        fit = lab[(lab["isoform"] == iso) & (lab["kind"] != "left")].dropna(subset=["y"])
        j = fit.merge(d[[C.ID_COL, col]], on=C.ID_COL).dropna(subset=[col])
        if len(j) < 50:
            continue
        y, p = j["y"].to_numpy(float), j[col].to_numpy(float)
        bias = float(p.mean() - y.mean())
        var = float(np.var(y))
        r2 = 1 - float(np.mean((y - p) ** 2)) / var
        r2d = 1 - float(np.mean((y - (p - bias)) ** 2)) / var
        rho = float(pd.Series(p).corr(pd.Series(y), method="spearman"))

        cen = lab[(lab["isoform"] == iso) & (lab["kind"] == "left")]
        cj = cen.merge(d[[C.ID_COL, col]], on=C.ID_COL).dropna(subset=[col])
        over = (float((cj[col] > cj["y_high"]).mean()) if len(cj) else np.nan)

        rows.append(dict(isoform=iso, n=len(j), rho=rho, r2=r2, r2_debiased=r2d,
                         bias=bias, sd_ratio=float(p.std() / y.std()),
                         cens_over=over, n_cen=len(cj)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", nargs="*", default=None)
    ap.add_argument("--iso", nargs="*", default=list(C.ISOFORMS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lab = pd.read_csv(C.ARTIFACT_DIR / "labels.csv")
    lab.columns = [c.strip() for c in lab.columns]

    files = [Path(f) for f in (a.oof or DEFAULT) if Path(f).exists()]
    if not files:
        raise SystemExit("no OOF files found; pass --oof FILE [FILE ...]")

    print("Fitted rows only. `rho` here is NOT a board forecast -- see the header.\n")
    frames = []
    for f in files:
        r = score(f, lab, a.iso)
        if r is None or r.empty:
            print(f"{f.name}: unreadable or no matching rows\n")
            continue
        r.insert(0, "oof", f.name)
        frames.append(r)
        print(f.name)
        print(f"  {'endpoint':9s} {'n':>5s} {'rho':>7s} {'R2':>7s} {'R2 -bias':>9s} "
              f"{'bias':>7s} {'sd p/y':>7s} {'cens>bnd':>9s}   {'board rho':>9s}")
        for _, x in r.iterrows():
            co = f"{x.cens_over:9.0%}" if np.isfinite(x.cens_over) else "        -"
            print(f"  {x.isoform:9s} {x.n:5d} {x.rho:7.4f} {x.r2:7.3f} "
                  f"{x.r2_debiased:9.3f} {x.bias:+7.3f} {x.sd_ratio:7.2f} {co}   "
                  f"{BOARD_RHO.get(x.isoform, float('nan')):9.4f}")
        print()

    if len(frames) > 1:
        allr = pd.concat(frames, ignore_index=True)
        print("=" * 78)
        print("BEST PER ENDPOINT")
        print("=" * 78)
        for iso in a.iso:
            g = allr[allr["isoform"] == iso]
            if g.empty:
                continue
            br = g.loc[g["rho"].idxmax()]
            bc = g.loc[g["cens_over"].idxmin()] if g["cens_over"].notna().any() else None
            line = f"  {iso:9s} rho {br.rho:.4f} ({br.oof})"
            if bc is not None:
                line += f"   |  lowest cens>bnd {bc.cens_over:.0%} ({bc.oof})"
            print(line)
        print("\n  A member that wins on rho and a member that wins on separation are not")
        print("  the same model. That is a blend question, and it is also a hint about")
        print("  which training change to keep.")
        if a.out:
            allr.to_csv(a.out, index=False)
            print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
