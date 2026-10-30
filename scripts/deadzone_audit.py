#!/usr/bin/env python
r"""Where our dead-zone cost actually comes from, and how much of it is free to remove.

    python scripts/deadzone_audit.py
    python scripts/deadzone_audit.py --oof oof_chemprop_st_v3_lean.csv oof_st_splice_a.csv
    python scripts/deadzone_audit.py --pool holdout          (blind-like 394 only)
    python scripts/deadzone_audit.py --pool matched          (blind histogram-matched)

WHY THIS EXISTS (Section 45). The per-endpoint leaderboard showed that our MAE is within
2-9% of the top four teams while our ST-RAE is 14-27% worse. Among the leaders themselves
ST-RAE moves SUB-linearly with MAE (0.74x), so our excess is a separate deficit, not an
amplification of slightly worse error. Matching their dead-zone efficiency at our own MAE
would take macro ST-RAE from 0.4833 to 0.4308 -- 47% of the whole gap to rank 1, with no
model change.

    endpoint  our e = ST-RAE/MAE   leaders' mean e   deficit
    CYP1A2         0.7345              0.5924         +0.142
    CYP2C9         0.8530              0.7678         +0.085
    CYP2D6         0.5167              0.4988         +0.018
    CYP3A4         0.8683              0.7638         +0.105

THE HYPOTHESIS THIS SCRIPT TESTS. The intervals on MEASURED values are tiny -- median
half-width 0.13-0.22 log units against an MAE of 0.5-1.1 -- so they absorb almost nothing
and cannot possibly spread `e` by 24% between teams. The dead zone that can is the
LEFT-CENSORED one. 69-71% of our CYP1A2/2C9/2D6 labels are censored at pIC50 = 4.0, and for
those the interval is (-inf, 4.0]: any prediction at or below 4.0 costs EXACTLY ZERO,
however far below, while still contributing to MAE. Our submitted predictions put only 18%
of CYP2C9 and 43% of CYP1A2 below 4.0 against a 70% prior -- and CYP2D6, the one endpoint
where our predicted inactive fraction matches (67% vs 69%), is the one endpoint with no
placement deficit at all.

So the claim is not that we place predictions badly. It is that OUR REGRESSORS DO NOT
PREDICT INACTIVITY: fitted on the interval subset, they hedge the censored majority upward,
and a compound the assay could not measure gets predicted at 4.3 instead of 3.2. That is a
wrong prediction on its own terms.

WHAT THIS SCRIPT PRODUCES, in four parts.

  1. CONVENTION CALIBRATION. The board reports one MAE and one ST-RAE per endpoint but does
     not say whether MAE covers censored compounds. That choice changes `e` a lot, so we
     compute it three ways and report which one reproduces the board's numbers. Until one
     does, every `e` comparison in Section 45 is provisional -- this is the falsification
     test for the whole argument, and it is the first thing to read in the output.

  2. COST DECOMPOSITION. Total dead-zone cost split into the censored arm and the measured
     arm. If the censored arm is a small share, the hypothesis is dead and we go back to
     plain model error.

  3. THE FREE LEVER, measured. `free = sum over censored rows of clip(p - bound, 0)`. Every
     unit of that is removable by predicting lower on a compound that has no measurable
     activity, and removing it cannot change the ranking among MEASURED compounds at all.
     Reported as the ST-RAE we would post if it were zero. This is a CEILING, computed with
     knowledge of which rows are censored, so it is not achievable -- it sizes the prize.

  4. THE HONEST LEVER. The same thing decided by the model instead of by the answer key:
     push predictions below `cut` down by `shift`, sweep both, and report the ST-RAE/rho
     frontier. A compound wrongly pushed is a compound whose real activity we then miss, so
     this arm pays for its mistakes and the frontier shows whether the trade is worth it.
     THIS IS A DIAGNOSTIC, NOT A SUBMISSION PATH: it measures how much of the ceiling a
     decision rule can reach, which tells us how hard to push on the training-side fix
     (weighting censored rows in the loss, or gating on an inactive classifier).

WHAT IT DELIBERATELY DOES NOT DO. It does not write a submission, and it does not fit the
cut/shift on the same rows it reports. The sweep is scored out-of-fold using the fold
assignment in artifacts/folds.csv, because a cut chosen and scored on the same compounds
would report a lever that does not exist.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

from cypkit import config as C                                        # noqa: E402
from scripts.blind_validator import BLIND, eval_pool, match_weights   # noqa: E402

FLOOR = C.PIC50_FLOOR

# The board's full-blind-set per-endpoint rows for our live submission (splice4), and the
# mean of the top four teams on each endpoint. Section 45 has the full table.
BOARD = {
    "CYP1A2": dict(strae=0.4972, mae=0.6769, rho=0.8004, r2=0.6213, e_leaders=0.5924),
    "CYP2C9": dict(strae=0.4073, mae=0.4775, rho=0.8420, r2=0.6986, e_leaders=0.7678),
    "CYP2D6": dict(strae=0.5908, mae=1.1434, rho=0.5342, r2=0.2151, e_leaders=0.4988),
    "CYP3A4": dict(strae=0.4378, mae=0.5042, rho=0.8593, r2=0.7307, e_leaders=0.7638),
}
DEFAULT_OOF = ["oof_st_splice_a.csv", "oof_st_splice_b.csv", "oof_chemprop_st_v3_lean.csv",
               "artifacts/oof_blend_ensPS.csv", "oof_chemprop_st_v3_spec.csv"]


# --------------------------------------------------------------------------------------
# the metric, split so each arm can be read on its own
# --------------------------------------------------------------------------------------
def cost_terms(p, lo, hi):
    """Per-row dead-zone cost: how far outside its own interval the prediction sits."""
    over = np.where(np.isfinite(hi), np.clip(p - hi, 0, None), 0.0)
    under = np.where(np.isfinite(lo), np.clip(lo - p, 0, None), 0.0)
    return over, under


def base_cost(y, lo, hi):
    """The denominator: the same cost for a constant prediction of mean(y)."""
    m = float(np.mean(y))
    o, u = cost_terms(np.full(len(y), m), lo, hi)
    return float(np.sum(o + u))


def decompose(p, y, lo, hi, cen):
    """ST-RAE plus the censored/measured split of its numerator."""
    over, under = cost_terms(p, lo, hi)
    tot = over + under
    d = base_cost(y, lo, hi)
    if d <= 0:
        return None
    return dict(
        strae=float(tot.sum() / d),
        cost_censored=float(tot[cen].sum() / d),
        cost_measured=float(tot[~cen].sum() / d),
        # the censored arm is one-sided by construction: a censored row can only be
        # punished for being too HIGH, so all of its cost is removable by predicting lower
        free=float(over[cen].sum() / d),
        base=d,
        n_cen=int(cen.sum()), n_mea=int((~cen).sum()),
        frac_below=float(np.mean(p[cen] <= hi[cen])) if cen.any() else np.nan,
    )


def mae_conventions(p, y, hi, cen):
    """The board prints one MAE; it does not say which of these it is."""
    fit = ~cen
    out = {}
    out["fitted"] = float(np.mean(np.abs(y[fit] - p[fit]))) if fit.any() else np.nan
    # censored row treated as if its true value were exactly the bound
    y_all = np.where(cen, hi, y)
    out["all_bound"] = float(np.mean(np.abs(y_all - p)))
    # censored row penalised only for being above the bound (one-sided, like the dead zone)
    err = np.where(cen, np.clip(p - hi, 0, None), np.abs(y - p))
    out["all_clip"] = float(np.mean(err))
    return out


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def r2(y, p):
    v = float(np.var(y))
    return float(1 - np.mean((y - p) ** 2) / v) if v > 0 else np.nan


# --------------------------------------------------------------------------------------
# pools
# --------------------------------------------------------------------------------------
def load_pool(kind: str, data_dir=None):
    pool = eval_pool(data_dir)
    if kind == "holdout":
        h = pd.read_csv(C.ARTIFACT_DIR / "holdout_blindlike.csv")
        h.columns = [c.strip() for c in h.columns]
        keep = set(h[C.ID_COL])
        pool = pool[pool[C.ID_COL].isin(keep)]
    return pool


def draw_index(y, cen, iso, rng, n_draws):
    """Resampling indices that make the label histogram look like the blind set."""
    m, sd = BLIND[iso]
    w, _ = match_weights(y, cen, m, sd)
    if w is None:
        return None
    n = len(y)
    return [rng.choice(n, size=n, replace=True, p=w) for _ in range(n_draws)]


def load_pred(path):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    return {ep: dict(zip(d[C.ID_COL], d[ep]))
            for ep in C.REGRESSION_ENDPOINTS if ep in d.columns}


# --------------------------------------------------------------------------------------
# parts 1-3
# --------------------------------------------------------------------------------------
def audit(pred_by_iso, pool, matched: bool, n_draws: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for iso in C.ISOFORMS:
        ep = f"{iso}_pIC50_direct_inhibition"
        s = pool[pool["isoform"] == iso]
        p = s[C.ID_COL].map(pred_by_iso.get(ep, {})).to_numpy(float)
        ok = np.isfinite(p)
        s, p = s[ok], p[ok]
        if len(s) < 80:
            continue
        y = s["y_eval"].to_numpy(float)
        lo = s["y_low"].to_numpy(float)
        hi = s["y_high"].to_numpy(float)
        cen = s["censored"].to_numpy(bool)

        idx = (draw_index(y, cen, iso, rng, n_draws) if matched
               else [np.arange(len(y))])
        acc = []
        for d in idx:
            dd = decompose(p[d], y[d], lo[d], hi[d], cen[d])
            if dd is None:
                continue
            mm = mae_conventions(p[d], y[d], hi[d], cen[d])
            fit = ~cen[d]
            dd.update({f"mae_{k}": v for k, v in mm.items()})
            dd["rho"] = spearman(y[d][fit], p[d][fit]) if fit.sum() > 30 else np.nan
            dd["r2"] = r2(y[d][fit], p[d][fit]) if fit.sum() > 30 else np.nan
            acc.append(dd)
        if not acc:
            continue
        a = pd.DataFrame(acc)
        r = {k: float(np.nanmean(a[k])) for k in a.columns}
        r["strae_sd"] = float(np.nanstd(a["strae"]))
        r["isoform"] = iso
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# part 4: the honest lever
# --------------------------------------------------------------------------------------
def lever_sweep(pred_by_iso, pool, folds, cuts, shifts, matched, n_draws, seed):
    """Out-of-fold: choose (cut, shift) on the other folds, score on this one."""
    rng = np.random.default_rng(seed)
    out = []
    for iso in C.ISOFORMS:
        ep = f"{iso}_pIC50_direct_inhibition"
        s = pool[pool["isoform"] == iso].copy()
        s["pred"] = s[C.ID_COL].map(pred_by_iso.get(ep, {}))
        s["fold"] = s[C.ID_COL].map(folds)
        s = s[np.isfinite(s["pred"]) & s["fold"].notna()]
        if len(s) < 200:
            continue
        y = s["y_eval"].to_numpy(float); p0 = s["pred"].to_numpy(float)
        lo = s["y_low"].to_numpy(float); hi = s["y_high"].to_numpy(float)
        cen = s["censored"].to_numpy(bool); fo = s["fold"].to_numpy()

        def apply(p, cut, shift):
            return np.where(p < cut, p - shift, p)

        def score(mask, cut, shift):
            p = apply(p0[mask], cut, shift)
            dd = decompose(p, y[mask], lo[mask], hi[mask], cen[mask])
            fit = ~cen[mask]
            rh = spearman(y[mask][fit], p[fit]) if fit.sum() > 30 else np.nan
            return dd["strae"], rh

        # pick per fold on the complement, score on the fold
        chosen, st_oof, rho_oof = [], [], []
        for f in sorted(set(fo)):
            te = fo == f
            tr = ~te
            best, bst = None, np.inf
            for c in cuts:
                for sh in shifts:
                    v, _ = score(tr, c, sh)
                    if v < bst:
                        bst, best = v, (c, sh)
            chosen.append(best)
            v, rh = score(te, *best)
            st_oof.append(v); rho_oof.append(rh)
        base_st, base_rho = score(np.ones(len(y), bool), -np.inf, 0.0)
        # the out-of-fold ST-RAE is a mean over folds with different bases; recompute
        # properly by applying each fold's own choice to the whole vector
        p_oof = p0.copy()
        for f, (c, sh) in zip(sorted(set(fo)), chosen):
            m = fo == f
            p_oof[m] = apply(p0[m], c, sh)
        dd = decompose(p_oof, y, lo, hi, cen)
        fit = ~cen
        out.append(dict(isoform=iso,
                        strae_before=base_st, strae_after=dd["strae"],
                        rho_before=base_rho, rho_after=spearman(y[fit], p_oof[fit]),
                        cuts=";".join(f"{c:.2f}/{sh:.2f}" for c, sh in chosen),
                        agree=len({c for c, _ in chosen}) == 1))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", nargs="*", default=None)
    ap.add_argument("--pool", choices=["full", "holdout", "matched"], default="matched",
                    help="full = every labelled row as-is; holdout = the blind-like 394; "
                         "matched = full, resampled to the blind label histogram")
    ap.add_argument("--draws", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-lever", action="store_true")
    ap.add_argument("--out", default="artifacts/deadzone_audit.csv")
    a = ap.parse_args()

    files = [f for f in (a.oof or DEFAULT_OOF) if Path(f).exists()]
    if not files:
        raise SystemExit("no OOF files found; pass --oof FILE [FILE ...]")

    pool = load_pool("holdout" if a.pool == "holdout" else "full")
    matched = a.pool == "matched"
    print(f"pool = {a.pool}: {len(pool)} (compound, isoform) rows, "
          f"{int(pool['censored'].sum())} censored "
          f"({pool['censored'].mean():.0%})")
    if matched:
        print(f"      resampled to the blind label histogram, {a.draws} draws")
    print()

    # ---- interval widths: the premise of the whole argument --------------------------
    print("PREMISE. Half-width of the interval on a MEASURED value, by endpoint.")
    print("If these are small next to MAE, the measured dead zone cannot explain anything.")
    print(f"  {'endpoint':9s} {'median hw':>10s} {'90th hw':>9s} {'board MAE':>10s} "
          f"{'hw/MAE':>8s}")
    for iso in C.ISOFORMS:
        s = pool[(pool["isoform"] == iso) & (~pool["censored"])]
        hw = ((s["y_high"] - s["y_low"]) / 2).dropna()
        if not len(hw):
            continue
        print(f"  {iso:9s} {hw.median():10.3f} {hw.quantile(.9):9.3f} "
              f"{BOARD[iso]['mae']:10.4f} {hw.median()/BOARD[iso]['mae']:8.1%}")
    print()

    allr = {}
    for f in files:
        allr[f] = audit(load_pred(f), pool, matched, a.draws, a.seed)

    # ---- part 1: which MAE convention reproduces the board? --------------------------
    print("=" * 86)
    print("PART 1. WHICH MAE CONVENTION IS THE BOARD USING?")
    print("The board's e must be reproducible by SOME convention on our own data, or the")
    print("Section 45 comparison is not measuring what it claims to measure.")
    print("=" * 86)
    print(f"\n  {'endpoint':9s} {'board e':>8s} | "
          + " ".join(f"{k:>10s}" for k in ("fitted", "all_bound", "all_clip"))
          + "   (our local e, best OOF)")
    best = min(allr.items(), key=lambda kv: kv[1]["strae"].mean())
    print(f"  using {Path(best[0]).name}\n")
    verdict = {}
    for _, x in best[1].iterrows():
        iso = x["isoform"]
        be = BOARD[iso]["strae"] / BOARD[iso]["mae"]
        es = {k: x["strae"] / x[f"mae_{k}"] for k in ("fitted", "all_bound", "all_clip")}
        pick = min(es, key=lambda k: abs(es[k] - be))
        verdict[iso] = pick
        print(f"  {iso:9s} {be:8.4f} | "
              + " ".join(f"{es[k]:10.4f}" + ("*" if k == pick else " ")
                         for k in ("fitted", "all_bound", "all_clip")))
    print("\n  * = closest to the board. If one column wins on all four endpoints that is")
    print("  the convention; if they disagree, our OOF population differs from the blind")
    print("  set enough that `e` is not transferable and Section 45 must be downgraded.")
    agree = len(set(verdict.values())) == 1
    print(f"\n  VERDICT: {'CONSISTENT -- ' + next(iter(verdict.values())) if agree else 'INCONSISTENT: ' + str(verdict)}")

    # ---- part 2/3: decomposition and the free lever -----------------------------------
    print("\n" + "=" * 86)
    print("PART 2/3. WHERE THE COST IS, AND HOW MUCH OF IT IS FREE")
    print("'free' is cost from a CENSORED compound predicted above its bound. Removing it")
    print("cannot change the ranking of any measured compound. It is a ceiling, not a plan.")
    print("=" * 86)
    for f, r in allr.items():
        if r.empty:
            continue
        print(f"\n{Path(f).name}")
        print(f"  {'endpoint':9s} {'ST-RAE':>8s} {'cen arm':>8s} {'mea arm':>8s} "
              f"{'free':>7s} {'ST-RAE-free':>12s} {'below bnd':>10s} {'rho':>7s}")
        for _, x in r.iterrows():
            print(f"  {x.isoform:9s} {x.strae:8.4f} {x.cost_censored:8.4f} "
                  f"{x.cost_measured:8.4f} {x.free:7.4f} "
                  f"{x.strae - x.free:12.4f} {x.frac_below:10.1%} {x.rho:7.4f}")
        print(f"  {'MACRO':9s} {r.strae.mean():8.4f} {r.cost_censored.mean():8.4f} "
              f"{r.cost_measured.mean():8.4f} {r.free.mean():7.4f} "
              f"{(r.strae - r.free).mean():12.4f}")
        print(f"  censored arm is {r.cost_censored.sum()/r.strae.sum():.0%} of total cost; "
              f"the free part is {r.free.sum()/r.strae.sum():.0%}")

    # ---- part 4 ----------------------------------------------------------------------
    if not a.no_lever:
        fp = C.ARTIFACT_DIR / "folds.csv"
        if fp.exists():
            fd = pd.read_csv(fp); fd.columns = [c.strip() for c in fd.columns]
            folds = dict(zip(fd[C.ID_COL], fd["fold"]))
            print("\n" + "=" * 86)
            print("PART 4. HOW MUCH OF THE CEILING A DECISION RULE ACTUALLY REACHES")
            print("(cut, shift) chosen on the other folds and scored on the held-out one,")
            print("so a rule that only works in hindsight shows up as no gain here.")
            print("=" * 86)
            cuts = np.arange(3.8, 5.61, 0.2)
            shifts = np.arange(0.0, 2.01, 0.25)
            lv = lever_sweep(load_pred(best[0]), pool, folds, cuts, shifts,
                             matched, a.draws, a.seed)
            if not lv.empty:
                print(f"\n{Path(best[0]).name}")
                print(f"  {'endpoint':9s} {'ST-RAE':>8s} -> {'after':>8s} {'gain':>8s}  "
                      f"{'rho':>7s} -> {'after':>7s}   per-fold cut/shift")
                for _, x in lv.iterrows():
                    print(f"  {x.isoform:9s} {x.strae_before:8.4f} -> {x.strae_after:8.4f} "
                          f"{x.strae_after-x.strae_before:+8.4f}  {x.rho_before:7.4f} -> "
                          f"{x.rho_after:7.4f}   {x.cuts}")
                print(f"  {'MACRO':9s} {lv.strae_before.mean():8.4f} -> "
                      f"{lv.strae_after.mean():8.4f} "
                      f"{(lv.strae_after-lv.strae_before).mean():+8.4f}")
                print("\n  A gain here that survives out-of-fold is the size of the prize a")
                print("  TRAINING-SIDE fix should aim at (weighting censored rows in the")
                print("  loss, or gating on an inactive classifier). It is not a submission.")

    frames = []
    for f, r in allr.items():
        if r.empty:
            continue
        r = r.copy(); r.insert(0, "oof", Path(f).name); frames.append(r)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}  ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
