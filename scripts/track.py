#!/usr/bin/env python
"""The submission ledger: what we sent, what it scored, per target, and what moved.

    python scripts/track.py
    python scripts/track.py --track pIC50
    python scripts/track.py --add pIC50 blend3_narrow overall 0.51 27 "narrow placement"
    python scripts/track.py --md > SUBMISSIONS.md

WHY A LEDGER AND NOT A TABLE IN A README. The leaderboard shows only your LATEST
submission, so score history exists nowhere but here -- once a submission is superseded,
its per-target numbers are gone unless they were written down at the time. That has
already cost us: `tdi_recut` has no per-target row because nobody recorded it before the
next submission replaced it, so the one question it could have answered -- which endpoint
the -0.0138 came from -- is now unanswerable.

Two things this makes visible that a flat list does not:

  1. BEST-EVER PER TARGET vs BEST OVERALL. They are usually different files, and the gap
     between "the best overall submission" and "the best column for each target, spliced"
     is free score that no modelling has to earn. On TDI that gap is currently 0.0217 --
     CYP3A4's best (0.4470, from tdi_best) and CYP2D6's best (0.2866, from tdi_sel) have
     never been in the same file.

  2. WHICH TARGET MOVED. An overall score that improves while a target regresses is the
     signature of a change that helped one endpoint and hurt another -- worth knowing
     before repeating it. The delta column is per target, not just overall.

SCORE DIRECTION is per track: pIC50 is MA-ST-RAE where LOWER is better; TDI is MA-MCC
where HIGHER is better. Everything here -- best, deltas, arrows -- respects that.

BLANK CELLS ARE HONEST. A missing per-target value means it was not recorded, not that it
was zero. Fill them by hand from the leaderboard's per-target tabs, or with --add.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from cypkit import config as C

LEDGER = C.artifact("submissions.csv")

# lower_is_better, the targets that make up the macro average, and the metric's name
TRACKS = {
    "pIC50": (True, ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"], "MA-ST-RAE"),
    "TDI":   (False, ["CYP3A4", "CYP2D6"], "MA-MCC"),
}


def load(metric=None):
    d = pd.read_csv(LEDGER)
    if "metric" not in d.columns:                 # ledgers written before metrics existed
        d["metric"] = d["track"].map({"pIC50": "ST-RAE", "TDI": "MCC"})
    if metric:
        # the overall row of a secondary metric is stored macro-averaged ("MA-rho")
        # while the per-target rows are bare ("rho"); accept both spellings
        want = {metric, f"MA-{metric}", metric.replace("MA-", "")}
        d = d[d["metric"].isin(want)]
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d["rank"] = pd.to_numeric(d["rank"], errors="coerce")
    return d


def cell(v, r):
    if not np.isfinite(v):
        return "—"
    return f"{v:.4f}" + (f" (r{int(r)})" if np.isfinite(r) else "")


def render(d, track, md=False, metric_name=None):
    lower, targets, metric = TRACKS[track]
    if metric_name:                # a secondary metric has its own direction
        metric = metric_name
        lower = metric_name.lower() not in ("rho", "r2", "mcc")
    t = d[d.track == track]
    if t.empty:
        return
    # the note lives on ONE row per submission; taking the first row after a sort picks
    # an empty one, so collapse by longest non-null note instead
    t = t.copy(); t["note"] = t["note"].fillna("")
    order = (t.sort_values("note", key=lambda s: s.str.len(), ascending=False)
             .drop_duplicates("submission")[["order", "submission", "note"]]
             .sort_values("order"))
    piv = t.pivot_table(index="submission", columns="scope", values="value",
                        aggfunc="last")
    rnk = t.pivot_table(index="submission", columns="scope", values="rank",
                        aggfunc="last")
    cols = ["overall"] + targets
    if "overall" not in piv:
        print(f"\n## {track}: no 'overall' row recorded for {metric}"); return
    best_i = (piv["overall"].idxmin() if lower else piv["overall"].idxmax())

    head = ["submission", metric, *targets, "what changed"]
    rows = []
    prev = None
    for _, r in order.iterrows():
        s = r["submission"]
        line = [f"**{s}**" if s == best_i else s]
        for c in cols:
            v = piv[c].get(s, np.nan) if c in piv else np.nan
            k = rnk[c].get(s, np.nan) if c in rnk else np.nan
            txt = cell(v, k)
            if c == "overall" and prev is not None and np.isfinite(v) and np.isfinite(prev):
                dv = v - prev
                good = (dv < 0) if lower else (dv > 0)
                txt += f"  {'↑' if good else '↓'}{abs(dv):.4f}"
            line.append(f"**{txt}**" if s == best_i and c == "overall" else txt)
            if c == "overall" and np.isfinite(v):
                prev = v
        line.append(str(r["note"]) if isinstance(r["note"], str) else "")
        rows.append(line)

    print(f"\n## {track}  ({metric}, {'lower' if lower else 'higher'} is better)\n")
    if md:
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join(["---"] * len(head)) + "|")
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
        print("  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
        print("  " + "  ".join("-" * w[i] for i in range(len(head))))
        for r in rows:
            print("  " + "  ".join(r[i].ljust(w[i]) for i in range(len(head))))

    # ---- the gap that costs nothing to close -------------------------------------
    print(f"\n  best overall: {best_i} at {piv['overall'][best_i]:.4f}")
    have = [c for c in targets if c in piv and piv[c].notna().any()]
    if not have:
        print("  (no per-target rows recorded yet -- fill them and this section fills in)")
        return
    parts, vals = [], []
    for c in have:
        i = piv[c].idxmin() if lower else piv[c].idxmax()
        parts.append(f"{c} {piv[c][i]:.4f} ({i})")
        vals.append(piv[c][i])
    print("  best ever per target: " + " · ".join(parts))
    if len(have) < len(targets):
        missing = [c for c in targets if c not in have]
        print(f"  splice not computable -- no per-target rows for {', '.join(missing)}")
        return
    owners = {(piv[c].idxmin() if lower else piv[c].idxmax()) for c in targets}
    spliced = float(np.mean(vals))
    if len(owners) == 1:
        print(f"  all four best-per-target values come from ONE submission "
              f"({owners.pop()}), so the splice is just that submission "
              f"({spliced:.4f}) -- record per-target rows for the others before "
              f"reading anything into this")
        return
    # gain is signed by the track: lower-is-better means the splice must come in BELOW
    gain = (piv["overall"][best_i] - spliced) if lower else \
           (spliced - piv["overall"][best_i])
    if gain > 0:
        print(f"  splicing those gives {spliced:.4f}  ->  {gain:+.4f} better than the "
              f"best single submission, with no modelling")
    else:
        print(f"  splicing those gives {spliced:.4f}, which is {abs(gain):.4f} WORSE "
              f"than {best_i} -- the best submission already beats the per-target bests, "
              f"so its own per-target rows are missing from the ledger")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=list(TRACKS), default=None)
    ap.add_argument("--md", action="store_true", help="markdown tables")
    ap.add_argument("--metric", default=None,
                    help="secondary metric to table instead of the primary "
                         "(pIC50: MAE, R2, rho). Direction is inferred: rho and R2 are "
                         "higher-is-better, errors are lower-is-better.")
    ap.add_argument("--add", nargs="+", metavar="V",
                    help="track submission scope value rank [note]")
    args = ap.parse_args()

    if args.add:
        a = args.add + [""] * (6 - len(args.add))
        tr, sub, scope, val, rk, note = a[:6]
        d = load()
        prev = d[(d.track == tr) & (d.submission == sub)]["order"]
        o = (prev.iloc[0] if len(prev)
             else (d[d.track == tr]["order"].max() + 1 if (d.track == tr).any() else 1))
        row = pd.DataFrame([dict(track=tr, order=o, submission=sub, scope=scope,
                                 value=val, rank=rk or np.nan, note=note)])
        pd.concat([d, row], ignore_index=True).to_csv(LEDGER, index=False)
        print(f"added: {tr} {sub} {scope} {val} r{rk}")

    prim = {"pIC50": "ST-RAE", "TDI": "MCC"}
    for tr in ([args.track] if args.track else list(TRACKS)):
        m = args.metric or prim[tr]
        render(load(m), tr, md=args.md, metric_name=args.metric)
    print(f"\nledger: {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
