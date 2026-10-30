#!/usr/bin/env python
r"""What the REAL blind set does to ST-RAE when we change the prediction distribution.

    python scripts/blind_response.py
    python scripts/blind_response.py --propose        (also print untested moves)

WHY THIS REPLACED THE LOCAL DIAGNOSTIC. Section 45 proposed measuring our dead-zone
deficit on a local pool. `scripts/deadzone_audit.py` was built for that and it FAILED its
own falsification test: no MAE convention reproduces the board's `e` on our data (local e
0.93-2.28 against a board 0.52-0.87), on either the blind-like holdout or the
histogram-matched pool. The reason is structural and was already documented in
`blind_validator.py` -- 69-71% of our labels are left-censored at pIC50 = 4.0, while the
blind set carries real fitted values well below the floor (solved anchors put CYP2D6 at
mean 2.92, sd 1.65). Our pool's ST-RAE base is built almost entirely out of censored rows
and the blind set's is not, so `e` does not transfer and the censoring mechanism proposed
in Section 45 part 3 is WRONG. That section is corrected.

THE INSTRUMENT WE ACTUALLY HAVE is better than any local pool: twelve scored pIC50
submissions, per-endpoint, on the real blind compounds with the real intervals. Several
pairs share an IDENTICAL Spearman on an endpoint -- the same ranking shipped with a
different distribution. Those pairs are controlled experiments on the true scoring
function, and they are free.

WHAT THE CONTROLLED PAIRS ALREADY SAY (the output below re-derives all of this):

    endpoint  pair                          d mean   d sd    d ST-RAE
    CYP2C9    blend3_placed -> _narrow       +0.00  -0.29    -0.0569
    CYP3A4    blend3_placed -> _narrow       +0.00  -0.27    -0.0357
    CYP2D6    blend4z       -> place_half    +0.82  -0.52    -0.0525
    CYP1A2    blend3_placed -> _narrow       -0.27  -0.07    +0.1147
    CYP3A4    blend4z       -> place_half    -0.29  -0.12    +0.2391

NARROWING THE SPREAD AT CONSTANT MEAN HELPS, EVERY TIME IT HAS BEEN TRIED. Moving the mean
DOWN is the thing that has hurt, twice, badly -- and both of the "placement failed"
episodes in this project's history are mean moves, not spread moves. They were recorded
together and the lesson was drawn against all of it.

WHY OUR CYP2D6 IS THE EXCEPTION ON THE BOARD, AND WHAT THAT PROVES. Section 45 observed
that CYP2D6 is the one endpoint where our dead-zone efficiency matches the leaders. The
reason is now visible: CYP2D6 is the one endpoint of the four where the live submission
ships a PLACED column (place_half) rather than a raw one. The transform bought -0.0525
ST-RAE and cost +46% MAE at identical rho. So the lever is real, it is already proven on
our own data on the real blind set, and it is simply not applied to the other three.

THE ONE CLEAN UNTESTED MOVE. CYP1A2 ships the widest distribution we have (sd 1.2594) and
has never been narrowed at constant mean -- blend3_narrow narrowed it by only 0.07 while
dropping the mean 0.27, so the +0.1147 it cost is attributable to the mean move, not the
spread. Every endpoint where spread was cut at constant mean improved.

HONEST STATUS. This is a distribution correction, which the standing instruction puts
behind model improvement, and that ordering is not overridden here. It is written up so the
size of the prize is on record and measured rather than argued: the claim is that our
models are mis-scaled, not that the metric should be gamed. A prediction whose spread is
1.3x the truth's is wrong about how potent its compounds are, and fixing that is a
modelling defect repair that happens to pay on this metric.

NOTHING HERE IS A SUBMISSION. The script reports measured gradients and names the untested
moves; it does not write a candidate, and the standing rule stands -- nothing ships until
internal validation says it beats 0.4833.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

from cypkit import config as C                                        # noqa: E402

# submission name in artifacts/submissions.csv -> the file that was actually uploaded
FILES = {
    "placed": "submission_placed.csv",
    "blend3_placed": "submission_blend3_placed.csv",
    "blend3_narrow": "submission_blend3_narrow.csv",
    "splice_best": "submission_splice_best.csv",
    "blend4z": "submission_blend4z.csv",
    "place_half": "submission_place_half.csv",
    "splice4": "submission_splice4.csv",
    "raw3a4": "submission_raw3a4.csv",
    "pseudo_blend45": "submission_pseudo_blend45.csv",
}
# The live submission's per-endpoint FULL-blind-set row, and the top four teams' mean
# dead-zone efficiency on each endpoint (Section 45).
BOARD_FULL = {
    "CYP1A2": dict(strae=0.4972, mae=0.6769, rho=0.8004, e_leaders=0.5924),
    "CYP2C9": dict(strae=0.4073, mae=0.4775, rho=0.8420, e_leaders=0.7678),
    "CYP2D6": dict(strae=0.5908, mae=1.1434, rho=0.5342, e_leaders=0.4988),
    "CYP3A4": dict(strae=0.4378, mae=0.5042, rho=0.8593, e_leaders=0.7638),
}
# Which column of the live submission came from a placement transform rather than raw
# model output. This is the confound Section 45 missed.
PLACED_COLUMN = {"CYP1A2": False, "CYP2C9": False, "CYP2D6": True, "CYP3A4": False}
RHO_TIE = 1e-4      # two submissions "share a ranking" if rho matches to the printed digit


def build_table(art: Path) -> pd.DataFrame:
    """Board results joined to the distribution of the file that produced them."""
    d = pd.read_csv(art / "submissions.csv")
    d.columns = [c.strip() for c in d.columns]
    d = d[d["track"] == "pIC50"]

    dist = {}
    for sub, fn in FILES.items():
        p = art / fn
        if not p.exists():
            continue
        s = pd.read_csv(p)
        s.columns = [c.strip() for c in s.columns]
        for iso in C.ISOFORMS:
            col = f"{iso}_pIC50_direct_inhibition"
            if col in s.columns:
                v = s[col].astype(float)
                dist[(sub, iso)] = (float(v.mean()), float(v.std()),
                                    float((v < C.PIC50_FLOOR).mean()))

    rows = []
    for (sub, iso), g in d.groupby(["submission", "scope"]):
        if iso not in C.ISOFORMS:
            continue
        v = {m: float(x) for m, x in zip(g["metric"], g["value"])}
        if "ST-RAE" not in v:
            continue
        order = g["order"].iloc[0]
        pm, ps, pb = dist.get((sub, iso), (np.nan, np.nan, np.nan))
        rows.append(dict(order=order, sub=sub, iso=iso, strae=v.get("ST-RAE"),
                         mae=v.get("MAE"), rho=v.get("rho"), r2=v.get("R2"),
                         pmean=pm, psd=ps, pbelow=pb))
    t = pd.DataFrame(rows)
    t["e"] = t["strae"] / t["mae"]
    return t.sort_values(["iso", "strae"]).reset_index(drop=True)


MEAN_MOVE = 0.05        # a mean change smaller than this is a spread-only experiment


def controlled_pairs(t: pd.DataFrame) -> pd.DataFrame:
    """Every pair on one endpoint that shipped the SAME ranking with a different shape.

    Same rho means the model's ordering is identical, so any ST-RAE difference is caused
    entirely by where the numbers were placed. This is the only clean experiment available
    on the true scoring function, and it costs nothing because it already happened.

    ORIENTATION IS FIXED BY THE DISTRIBUTION CHANGE, NEVER BY THE SCORE. A first version of
    this function enumerated pairs off a frame sorted by ST-RAE, which made `to` the worse
    row every time and printed "0 of 7 spread cuts improved" directly above a paragraph
    saying narrowing always helps. The sign of the answer must not depend on how the rows
    happened to be sorted, so: a mean move is always reported in the DOWNWARD direction and
    a spread-only move in the NARROWING direction, and d_strae is then the honest cost or
    benefit of doing that thing.
    """
    seen, out = set(), []
    for iso, g in t.groupby("iso"):
        g = g.dropna(subset=["rho", "psd", "pmean"])
        rec = g.to_dict("records")
        for i in range(len(rec)):
            for j in range(i + 1, len(rec)):
                a, b = rec[i], rec[j]
                if abs(a["rho"] - b["rho"]) > RHO_TIE:
                    continue
                dm, ds = b["pmean"] - a["pmean"], b["psd"] - a["psd"]
                if abs(dm) < 1e-3 and abs(ds) < 1e-3:
                    continue            # the same vector submitted twice
                # orient: mean moves point DOWN, spread-only moves point NARROWER
                flip = (dm > 0) if abs(dm) >= MEAN_MOVE else (ds > 0)
                if flip:
                    a, b, dm, ds = b, a, -dm, -ds
                key = (iso, round(dm, 4), round(ds, 4),
                       round(b["strae"] - a["strae"], 4))
                if key in seen:         # splice4 / splice_best ship identical columns
                    continue
                seen.add(key)
                out.append(dict(iso=iso, frm=a["sub"], to=b["sub"], rho=a["rho"],
                                d_mean=dm, d_sd=ds,
                                d_strae=b["strae"] - a["strae"],
                                d_mae=(b["mae"] - a["mae"]) if np.isfinite(
                                    a["mae"]) and np.isfinite(b["mae"]) else np.nan,
                                kind=("narrow spread" if abs(dm) < MEAN_MOVE
                                      else "lower mean")))
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=C.ARTIFACT_DIR)
    ap.add_argument("--propose", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t = build_table(a.artifacts)
    if t.empty:
        raise SystemExit("no scored submissions found in artifacts/submissions.csv")

    print("SCORED SUBMISSION HISTORY, per endpoint, on the real blind compounds.")
    print("`e` = ST-RAE/MAE is the dead-zone efficiency; lower is better and it is")
    print("comparable between rows here because the endpoint and intervals are fixed.\n")
    for iso in C.ISOFORMS:
        g = t[t["iso"] == iso]
        if g.empty:
            continue
        print(f"  {iso}")
        print(f"    {'submission':16s} {'ST-RAE':>8s} {'MAE':>7s} {'e':>7s} {'rho':>7s} "
              f"{'pred mean':>10s} {'pred sd':>8s} {'<4.0':>6s}")
        for _, x in g.iterrows():
            f = lambda v, w=8, p=4: (f"{v:{w}.{p}f}" if np.isfinite(v) else " " * (w - 1) + "-")  # noqa: E731
            print(f"    {x['sub']:16s} {f(x.strae)} {f(x.mae,7)} {f(x.e,7)} {f(x.rho,7)} "
                  f"{f(x.pmean,10,3)} {f(x.psd,8,4)} {f(x.pbelow,6,2)}")
        b = BOARD_FULL[iso]
        print(f"    {'LIVE (full set)':16s} {b['strae']:8.4f} {b['mae']:7.4f} "
              f"{b['strae']/b['mae']:7.4f} {b['rho']:7.4f}"
              f"     leaders' e {b['e_leaders']:.4f}"
              + ("   [PLACED column]" if PLACED_COLUMN[iso] else "   [raw column]"))
        print()

    cp = controlled_pairs(t)
    print("=" * 84)
    print("CONTROLLED PAIRS: identical Spearman, different distribution.")
    print("Same ordering shipped twice, so d ST-RAE is caused by placement and nothing")
    print("else. These are measurements on the true scoring function, not estimates.")
    print("=" * 84)
    if cp.empty:
        print("  none found")
    else:
        cp = cp.sort_values(["kind", "d_strae"])
        print("\n  Every row is oriented the same way: the mean goes DOWN, or (when the")
        print("  mean is held) the spread goes NARROWER. A negative d ST-RAE means doing")
        print("  that thing HELPED on the real blind set.\n")
        print(f"  {'endpoint':9s} {'from -> to':34s} {'rho':>7s} {'d mean':>7s} "
              f"{'d sd':>7s} {'d MAE':>7s} {'d ST-RAE':>9s}")
        for _, x in cp.iterrows():
            dm = f"{x.d_mae:7.4f}" if np.isfinite(x.d_mae) else "      -"
            print(f"  {x.iso:9s} {x.frm + ' -> ' + x.to:34s} {x.rho:7.4f} "
                  f"{x.d_mean:+7.3f} {x.d_sd:+7.3f} {dm} {x.d_strae:+9.4f}")
        print()
        for k, g in cp.groupby("kind"):
            print(f"  {k:14s} n={len(g):2d}   mean d ST-RAE {g.d_strae.mean():+.4f}   "
                  f"best {g.d_strae.min():+.4f}   worst {g.d_strae.max():+.4f}")
        so = cp[cp["kind"] == "narrow spread"]
        if len(so):
            win = int((so.d_strae < -1e-4).sum())
            neutral = int((so.d_strae.abs() <= 1e-4).sum())
            print(f"\n  NARROWING at constant mean: {win} of {len(so)} improved ST-RAE, "
                  f"{neutral} were a no-op\n  (a spread change too small to measure), "
                  f"{len(so) - win - neutral} lost.")
        md = cp[cp["kind"] == "lower mean"]
        if len(md):
            print(f"  LOWERING THE MEAN: {int((md.d_strae > 0).sum())} of {len(md)} made "
                  f"it worse, by up to {md.d_strae.max():+.4f}.\n  Both of this project's "
                  f"'placement failed' episodes are in this row, and neither\n  is about "
                  f"spread -- the lesson was drawn against the wrong variable.")

    if a.propose:
        print("\n" + "=" * 84)
        print("UNTESTED MOVES, ranked by how much of the gradient is already measured")
        print("=" * 84)
        live = {iso: t[(t["iso"] == iso) & (t["sub"] == "splice4")] for iso in C.ISOFORMS}
        for iso in C.ISOFORMS:
            g = live[iso]
            if g.empty:
                continue
            x = g.iloc[0]
            b = BOARD_FULL[iso]
            head = b["strae"] - b["e_leaders"] * b["mae"]
            tried = cp[(cp["iso"] == iso) & (cp["kind"] == "narrow spread")
                       & (cp["d_sd"].abs() > 0.05)]
            note = ("spread never cut at constant mean here"
                    if tried.empty else
                    "narrowing measured at " + ", ".join(
                        f"{s:+.4f} (d sd {v:+.2f})"
                        for s, v in zip(tried.d_strae, tried.d_sd)))
            print(f"  {iso}  shipped sd {x.psd:.4f}, mean {x.pmean:.3f}"
                  + ("  [already placed -- least headroom]" if PLACED_COLUMN[iso] else "")
                  + f"\n      efficiency headroom to the leaders {head:+.4f} ST-RAE;"
                    f" {note}")
        print("\n  Read this as a ranking of EXPERIMENTS, not a plan. The standing order")
        print("  puts model improvement first, and a distribution change should ride")
        print("  along with the next model that earns a submission slot, not consume one.")

    if a.out:
        t.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
