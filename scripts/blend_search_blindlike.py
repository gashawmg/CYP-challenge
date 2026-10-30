#!/usr/bin/env python
r"""Re-fit the pIC50 blend against a yardstick that matches the blind set on DIFFICULTY,
not only on the label histogram.

    python scripts\blend_search_blindlike.py
    python scripts\blend_search_blindlike.py --boot 600

WHY. scripts/pic50_blend_search.py selects members against `eval_pool()` resampled to the
blind LABEL HISTOGRAM. Section 30 showed that is only half the correction. Our validation
rows are also far HARDER than the blind ones -- nearest-neighbour Tanimoto to the set a
model could learn from, median 0.438 for us against 0.587 for the blind set, 28% of our
rows above 0.50 against 96% of theirs -- and a too-hard yardstick picked the
over-regularised model on 4 of 4 endpoints. Every blend weight this project ships was
chosen under that distortion.

THREE YARDSTICKS, ONE SET OF PREDICTIONS, SO THE COMPARISON IS EXACT.

    plain        the uncensored pool as it is
    histogram    resampled to the blind label histogram        <- what we ship against
    hist x sim   histogram weights TIMES similarity importance <- both corrections, new

The similarity weights are P_blind(bin) / P_pool(bin) over nearest-neighbour Tanimoto
bins; multiplying them by the histogram weights and renormalising composes the two. No
instrument in this project has ever matched the blind set on both axes at once.

THE INSTRUMENT CHECK DECIDES WHICH YARDSTICK TO BELIEVE, AND IT RUNS BEFORE THE SEARCH.
For each weighting we print the incumbent blend4z's local rho against the rho the board
actually reported for that endpoint of submission_splice4. A ratio of 1.00 means the
yardstick predicts the board. Section 30's gate applies unchanged: if the composed
weighting moves an endpoint's ratio AWAY from 1.00, its selection is not licensed and that
endpoint keeps the incumbent.

WHAT MAY FOLLOW FROM THIS (Section 19, Section 31). A blend change is a candidate only if
its gain exceeds the instrument's demonstrated error AND exceeds the board's own paired
resolution, which at a half-scored leaderboard is about +-0.0067 macro ST-RAE. Selection
happens on rho, which is invariant under the affine placement and is the only quantity a
blend can move. Nothing here writes a submission.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from scripts.blind_validator import BLIND, eval_pool, match_weights  # noqa: E402
from scripts.pic50_blend_search import (BLEND4Z, BLIND_RHO, MEMBERS,  # noqa: E402
                                        col_for, greedy, load)

ROOT = Path(__file__).resolve().parent.parent
EDGES = np.array([0.0, .30, .35, .40, .45, .50, .55, .60, .65, .70, 1.01])


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def sim_weights(nn_pool, nn_blind):
    pb = np.clip(np.digitize(nn_pool, EDGES) - 1, 0, len(EDGES) - 2)
    bb = np.clip(np.digitize(nn_blind, EDGES) - 1, 0, len(EDGES) - 2)
    pp = np.bincount(pb, minlength=len(EDGES) - 1).astype(float)
    qq = np.bincount(bb, minlength=len(EDGES) - 1).astype(float)
    pp /= max(pp.sum(), 1); qq /= max(qq.sum(), 1)
    return np.where(pp > 0, qq / np.maximum(pp, 1e-12), 0.0)[pb]


def norm1(w):
    w = np.clip(np.nan_to_num(w, nan=0.0), 0, None)
    return w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--iso", default="all")
    a = ap.parse_args()

    missing = [n for n, (o, _) in MEMBERS.items() if not (ROOT / o).exists()]
    fatal = [n for n in missing if n in BLEND4Z]
    if fatal:
        raise SystemExit(f"\n  {fatal} are part of blend4z, the incumbent this search is "
                         f"measured against. Restore those files before running.")
    members = {n: v for n, v in MEMBERS.items() if n not in missing}
    for n in missing:
        print(f"  skipping optional member '{n}' ({MEMBERS[n][0]} absent)")
    oof = {n: load(o) for n, (o, _) in members.items()}

    pool = eval_pool()
    folds = pd.read_csv(C.artifact("folds.csv"))
    folds.columns = [c.strip() for c in folds.columns]
    sim_of = dict(zip(folds[C.ID_COL], folds["max_sim_to_other_folds"]))
    nnf = pd.read_csv(C.artifact("nnsim.csv"))
    nnf.columns = [c.strip() for c in nnf.columns]
    nn_blind = nnf.loc[nnf["split"] != "train", "nnsim"].to_numpy(float)

    rng = np.random.default_rng(0)
    isos = C.ISOFORMS if a.iso == "all" else (a.iso,)
    rows = []

    for iso in isos:
        names = []
        for n in members:
            c = col_for(oof[n], iso)
            if c is None:
                continue
            v = pd.to_numeric(oof[n][c], errors="coerce")
            if v.notna().sum() < 100 or float(v.std(skipna=True) or 0.0) < 1e-9:
                continue
            names.append(n)

        s = pool[(pool["isoform"] == iso) & (~pool["censored"])]
        ids = s[C.ID_COL]
        for n in names:
            ids = ids[ids.isin(oof[n].index)]
        s = s[s[C.ID_COL].isin(ids) & s[C.ID_COL].isin(sim_of)]
        ids = s[C.ID_COL].to_numpy()
        y = s["y_eval"].to_numpy(float)
        nn = np.array([sim_of[i] for i in ids], float)
        Z = np.nan_to_num(np.column_stack([
            (lambda v: (v - np.nanmean(v)) / (np.nanstd(v) or 1.0))(
                oof[n].loc[ids, col_for(oof[n], iso)].to_numpy(float)) for n in names]))

        m, sd = BLIND[iso]
        # match_weights returns (weights, diagnostics) -- unpacking it as one array is
        # how this crashed the first time. It also returns (None, None) when no bin has
        # mass, so fall back to uniform rather than propagating a None.
        wh, _ = match_weights(y, s["censored"].to_numpy(bool), m, sd)
        w_hist = norm1(np.ones(len(y)) if wh is None else np.asarray(wh, float))
        w_sim = sim_weights(nn, nn_blind)
        W = {"plain": norm1(np.ones(len(y))),
             "histogram": w_hist,
             "hist x sim": norm1(w_hist * w_sim)}

        b4 = [n for n in BLEND4Z if n in names]
        base = Z[:, [names.index(n) for n in b4]].mean(axis=1)

        print(f"\n=== {iso}   {len(y)} uncensored pool rows, {len(names)} members, "
              f"median NN {np.median(nn):.3f} (blind {np.median(nn_blind):.3f})")
        print(f"  {'yardstick':12s} {'ESS':>6s} {'blend4z rho':>12s} {'blind rho':>10s} "
              f"{'ratio':>7s}   selection")

        for tag, w in W.items():
            ess = 1.0 / float(np.sum(w ** 2))
            # selection is run INSIDE the weighting: resample, then pick greedily
            picks, rhos = [], []
            for _ in range(max(20, a.boot // 10)):
                i = rng.choice(len(y), len(y), replace=True, p=w)
                if len(np.unique(y[i])) < 3:
                    continue
                ch, _, _ = greedy(y[i], Z[i], names)
                picks.append(tuple(sorted(set(ch))))
            top = pd.Series(picks).value_counts()
            sel = list(top.index[0]) if len(top) else b4
            freq = float(top.iloc[0] / top.sum()) if len(top) else 0.0
            cand = Z[:, [names.index(n) for n in sel]].mean(axis=1)
            # IN-SAMPLE gain: paired, incumbent and candidate on the SAME resamples.
            # This number is OPTIMISTIC and must never be quoted alone -- `sel` was chosen
            # on these very rows, so part of the gap is the search fitting noise.
            g = []
            for _ in range(a.boot):
                i = rng.choice(len(y), len(y), replace=True, p=w)
                if len(np.unique(y[i])) < 3:
                    continue
                g.append(spearman(cand[i], y[i]) - spearman(base[i], y[i]))
                rhos.append(spearman(base[i], y[i]))
            g = np.asarray(g)
            # OUT-OF-SAMPLE gain: re-run the SELECTION on half the rows and score it on
            # the other half. This is the control scripts/pic50_blend_search.py was built
            # around ("nothing has ever re-run the SELECTION out of sample") and it is the
            # only honest price of a searched blend. Section 15 measured +0.003 for
            # re-blending and called it noise; anything here has to beat that.
            oos = []
            for _ in range(a.boot // 2):
                i = rng.choice(len(y), len(y), replace=True, p=w)
                if len(np.unique(y[i])) < 6:
                    continue
                half = rng.random(len(i)) < 0.5
                A, B = i[half], i[~half]
                if len(A) < 30 or len(B) < 30 or len(np.unique(y[B])) < 3:
                    continue
                ch, _, _ = greedy(y[A], Z[A], names)
                v = Z[:, [names.index(n) for n in ch]].mean(axis=1)
                oos.append(spearman(v[B], y[B]) - spearman(base[B], y[B]))
            oos = np.asarray(oos) if len(oos) else np.zeros(1)
            r_base = float(np.mean(rhos))
            ratio = BLIND_RHO[iso] / r_base if r_base > 0 else np.nan
            print(f"  {tag:12s} {ess:6.0f} {r_base:12.4f} {BLIND_RHO[iso]:10.4f} "
                  f"{ratio:7.2f}   {'+'.join(sel)} ({freq:.0%})")
            print(f"  {'':12s} {'':6s} in-sample  {g.mean():+.4f} "
                  f"[{np.quantile(g, .025):+.4f}, {np.quantile(g, .975):+.4f}]   "
                  f"OUT-OF-SAMPLE {oos.mean():+.4f} "
                  f"[{np.quantile(oos, .025):+.4f}, {np.quantile(oos, .975):+.4f}]")
            rows.append(dict(isoform=iso, yardstick=tag, ess=ess, rho_blend4z=r_base,
                             blind_rho=BLIND_RHO[iso], ratio=ratio,
                             selection="+".join(sel), stability=freq,
                             gain=float(g.mean()), lo=float(np.quantile(g, .025)),
                             hi=float(np.quantile(g, .975)),
                             oos=float(oos.mean()),
                             oos_lo=float(np.quantile(oos, .025)),
                             oos_hi=float(np.quantile(oos, .975)),
                             p_worse=float((g <= 0).mean())))

    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "artifacts" / "blend_search_blindlike.csv", index=False)

    print("\n" + "=" * 74)
    print("WHICH YARDSTICK PREDICTS THE BOARD?  PER ENDPOINT, not globally.\n")
    print("  A global winner is dominated by whichever endpoint has the worst anchor --")
    print("  here CYP2D6, whose (2.922, 1.652) pair Section 20 already flagged as the")
    print("  suspect one. Choose the yardstick endpoint by endpoint.\n")
    piv = d.pivot_table(index="yardstick", columns="isoform", values="ratio")
    print(piv.round(3).to_string())

    print("\n\nTHE CANDIDATE, PRICED HONESTLY\n")
    print(f"  {'endpoint':9s} {'yardstick':12s} {'ratio':>6s} {'in-sample':>10s} "
          f"{'OUT-OF-SAMPLE':>22s} {'stab':>5s}")
    live = 0.0
    for iso, gg in d.groupby("isoform"):
        b = gg.loc[(gg.ratio - 1).abs().idxmin()]
        keep = b.oos_lo > 0
        print(f"  {iso:9s} {b.yardstick:12s} {b.ratio:6.3f} {b.gain:+10.4f} "
              f"{b.oos:+10.4f} [{b.oos_lo:+.4f},{b.oos_hi:+.4f}] {b.stability:5.0%}"
              + ("  KEEP" if keep else "  hold incumbent"))
        if keep:
            live += b.oos
    # Section 16: ensemble RANKING gains transfer to the board at about 14% of local size.
    # Quoting a local rho gain as if it transferred whole is how three CYP3A4 forecasts
    # were overshot in a row.
    TRANSFER = 0.14
    blind_rho = live * TRANSFER
    macro_st = blind_rho * (0.06 / 0.05) / 4
    print(f"\n  out-of-sample local rho gain, summed over kept endpoints: {live:+.4f}")
    print(f"  x measured ensemble transfer rate {TRANSFER:.0%}          -> blind rho "
          f"{blind_rho:+.4f}")
    print(f"  x exchange rate (-0.06 macro ST-RAE per +0.05 rho) / 4    -> macro "
          f"{-macro_st:+.4f}")
    print(f"  board paired resolution at a half-scored leaderboard:        +-0.0067")
    if macro_st > 0.0067:
        print("\n  -> CLEARS the board's resolution. Price it per endpoint and take it to")
        print("     the Section 19 checklist before any slot is spent.")
    else:
        print("\n  -> BELOW the board's own resolution. NOT A CANDIDATE. Re-weighting the")
        print("     members we already have cannot move this track, which is what Section")
        print("     15 measured and what this rerun now confirms on a better yardstick.")
        print("     The next point must come from a NEW MEMBER, not new weights.")
    print("\n  wrote artifacts/blend_search_blindlike.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
