r"""
build_reblend.py  --  ship the re-composed blend, with placement held fixed.

WHAT IS BEING CHANGED AND WHAT IS NOT. scripts\blend_nested.py showed, with the
composition chosen on four folds and scored on the fifth it never saw, that re-composing
the member pool beats the shipped blend on all four endpoints (macro raw 0.2714 -> 0.2625,
P(worse) 0.031 / 0.000 / 0.028 / 0.000). The cause is mundane: mhtdi5 and mhens3 were built
after the blend was frozen and never added, while two of the four shipped members, unimol
and cp_ensA, are the WORST on every endpoint.

This changes the RANKING only. The placement we banked -- CYP1A2 +0.28, CYP3A4 +0.20,
CYP2D6 spread x1.25, worth 0.0297 macro -- was tuned against the OLD ranking, so every
endpoint here is rescaled to the EXACT mean and standard deviation of the 0.4440 submission.
Placement is therefore held fixed by construction and the ranking is the only variable, which
is what makes the board's per-endpoint answer attributable.

(The one thing this does leave on the table: the optimal spread is r*sigma_y, so a better
ranking has a higher r and wants a slightly WIDER spread. Matching the old sd deliberately
forgoes that, because taking it would change two things at once. It is the next probe, not
this one.)

THE COMBINATION RULE IS THE ONE THAT WAS VALIDATED. blend_nested averaged each member AFTER
fitting it its own optimal affine to out-of-fold truth. On the blind set there is no truth to
fit, so members are averaged as z-scores instead. Those two rules are close but not identical,
and shipping a rule that differs from the validated one is how a measured gain quietly turns
into an unmeasured one. So this script re-scores the nested protocol under z-score averaging
and refuses to write a file unless that rule also wins.

Run (Windows, admet_gpu):
  python scripts\build_reblend.py
  python scripts\build_reblend.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ranking_headroom import ISO, MEMBERS, SHIPPED, strae  # noqa: E402


def opt_shift(p, lo, hi):
    """Exact minimiser of mean soft-thresholded error over a uniform shift, O(n log n).

    cost(s) = sum_i max(s - a_i, 0) + sum_i max(b_i - s, 0),  a_i = hi_i - p_i,
    b_i = lo_i - p_i. Both arms are piecewise linear with breakpoints at a and b, so the
    minimum sits on one of them and every candidate is evaluated at once with prefix sums.
    (The O(n^2) loop this replaces was correct but made the scale search time out.)"""
    p = np.asarray(p, float)
    a = np.sort(np.asarray(hi, float) - p)
    b = np.sort(np.asarray(lo, float) - p)
    n = len(p)
    ca = np.concatenate([[0.0], np.cumsum(a)])
    cb = np.concatenate([[0.0], np.cumsum(b)])
    s = np.unique(np.concatenate([a, b]))
    ka = np.searchsorted(a, s, side="left")          # count of a_i < s
    up = ka * s - ca[ka]
    kb = np.searchsorted(b, s, side="right")         # count of b_i <= s
    dn = (cb[n] - cb[kb]) - (n - kb) * s
    cost = (up + dn) / n
    i = int(np.argmin(cost))
    return float(s[i]), float(cost[i])

TRUTH = ROOT / "data" / "cyp-challenge-TRAIN_inhibition.csv"
FOLDS = ROOT / "artifacts" / "folds.csv"
REF = ROOT / "artifacts" / "submission_d6spread_only.csv"      # the 0.4440 file
OUT = ROOT / "artifacts" / "submission_reblend.csv"
PRED = {
    "mhtdi5": "pred_mhtdi5.csv", "mhens3": "pred_mhens3.csv",
    "cp_v3": "predictions_cyp_st_v3.csv", "v5_rest": "pred_st_v5_rest.csv",
    "v6": "pred_st_v6.csv", "v7": "pred_st_v7.csv",
    "unimol": "predictions_unimol_cyp_mt_test.csv", "mh_ctl": "pred_st_mh_ctl.csv",
    "mh2": "pred_st_mh2.csv", "cp_ensA": "predictions_cyp_st_v3.csv",
}


def z(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-12 else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-size", type=int, default=4)
    ap.add_argument("--boot", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    truth = pd.read_csv(TRUTH).merge(pd.read_csv(FOLDS)[["Molecule_Name", "fold"]],
                                     on="Molecule_Name", how="inner")
    mem = {n: pd.read_csv(ROOT / r) for n, r in MEMBERS.items() if (ROOT / r).exists()}

    chosen, nested, BOOT = {}, {}, {}
    for iso in ISO:
        col = f"{iso}_pIC50_direct_inhibition"
        lo_c, hi_c = f"{col}_conf_low", f"{col}_conf_high"
        t = truth[["Molecule_Name", col, lo_c, hi_c, "fold"]].dropna()
        names, J = [], t
        for n, d in mem.items():
            if col not in d.columns:
                continue
            s = d[["Molecule_Name", col]].dropna().rename(columns={col: n})
            if len(s) < 500:
                continue
            names.append(n)
            J = J.merge(s, on="Molecule_Name", how="inner")
        J = J.reset_index(drop=True)
        y = J[col].to_numpy(float)
        lo, hi = J[lo_c].to_numpy(float), J[hi_c].to_numpy(float)
        fold = J["fold"].to_numpy().astype(int)
        # every member as a z-score -- the SAME rule used on the blind set below
        Z = {n: z(J[n].to_numpy(float)) for n in names}
        tgt_m, tgt_s = y.mean(), y.std()

        SCALES = np.linspace(0.35, 1.25, 19)   # in units of sigma_y; optimum is r*sigma_y

        def fit_affine(b, mask):
            """best (scale, shift) for this z-blend on these rows. The first version of
            this pinned the scale at sigma_y, which over-spreads EVERY arm -- the optimum
            is r*sigma_y -- and distorted the selection as well as the levels."""
            best = (np.inf, 1.0, 0.0)
            for s in SCALES:
                p = tgt_m + tgt_s * s * b[mask]
                sh, v = opt_shift(p, lo[mask], hi[mask])
                if v < best[0]:
                    best = (v, s, sh)
            return best

        def sc(sel, mask):
            b = np.mean([Z[n] for n in sel], axis=0)
            return fit_affine(b, mask)[0]

        # nested check under the SHIPPING rule
        oof_g, oof_s = np.zeros(len(J)), np.zeros(len(J))
        shipped = [n for n in SHIPPED if n in names]
        for k in sorted(set(fold)):
            te, tr = fold == k, fold != k
            cur, bv = [], np.inf
            while len(cur) < a.max_size:
                v, n = min((sc(cur + [n], tr), n) for n in names if n not in cur)
                if v >= bv - 1e-6:
                    break
                cur, bv = cur + [n], v
            for store, sel in ((oof_g, cur), (oof_s, shipped)):
                b = np.mean([Z[n] for n in sel], axis=0)
                _, s_, sh = fit_affine(b, tr)          # affine fitted on TRAIN folds only
                store[te] = tgt_m + tgt_s * s_ * b[te] + sh
        vg, vs = strae(oof_g, lo, hi).mean(), strae(oof_s, lo, hi).mean()
        d = np.empty(a.boot)
        for b_ in range(a.boot):
            ix = rng.integers(0, len(y), len(y))
            d[b_] = (strae(oof_g[ix], lo[ix], hi[ix]).mean()
                     - strae(oof_s[ix], lo[ix], hi[ix]).mean())
        pw = float((d >= 0).mean())
        nested[iso] = (vs, vg, pw)
        BOOT[iso] = d          # keep the per-endpoint draws for the MACRO test below

        # production composition: greedy on ALL folds
        cur, bv = [], np.inf
        full = np.ones(len(J), bool)
        while len(cur) < a.max_size:
            v, n = min((sc(cur + [n], full), n) for n in names if n not in cur)
            if v >= bv - 1e-6:
                break
            cur, bv = cur + [n], v
        chosen[iso] = cur
        print(f"{iso:<8} nested shipped {vs:.4f} -> greedy {vg:.4f} "
              f"({vg-vs:+.4f}, P(worse) {pw:.3f})   production pick: {'+'.join(cur)}")

    ms, mg = (np.mean([v[0] for v in nested.values()]),
              np.mean([v[1] for v in nested.values()]))
    # The GATE is on the MACRO, because the macro is what the board scores. Requiring all
    # four endpoints to pass individually is a stricter test than the competition applies
    # and would discard a real macro gain built from four same-signed parts.
    dm = np.mean([BOOT[i] for i in ISO], axis=0)
    pm = float((dm >= 0).mean())
    passed = [i for i in ISO if nested[i][2] < 0.05]
    cons = np.mean([nested[i][1] if i in passed else nested[i][0] for i in ISO])
    print(f"\nz-score rule, nested macro: shipped {ms:.4f} -> greedy {mg:.4f} "
          f"({mg-ms:+.4f})   MACRO P(worse) {pm:.3f}")
    print(f"endpoints passing their OWN gate: {', '.join(passed) if passed else 'none'}")
    print(f"conservative variant, changing only those: macro {cons:.4f} ({cons-ms:+.4f})")
    if not (mg < ms and pm < 0.05):
        print("MACRO GATE FAILED under the shipping rule. No file written.")
        return
    print("MACRO GATE PASSED under the rule that will actually ship.")
    if a.dry_run:
        print("--dry-run: stopping before writing.")
        return

    # ---------------- build on the blind set ----------------
    ref = pd.read_csv(REF)
    out = ref.copy()
    preds = {n: pd.read_csv(ROOT / f) for n, f in PRED.items() if (ROOT / f).exists()}
    print(f"\n{'endpoint':<9}{'members':<34}{'spearman vs 0.4440 file':>24}")
    for iso in ISO:
        col = f"{iso}_pIC50_direct_inhibition"
        zs = []
        for n in chosen[iso]:
            d = preds[n]
            s = d.set_index("Molecule_Name")[col].reindex(ref["Molecule_Name"])
            if s.isna().any():
                raise SystemExit(f"{n} has {int(s.isna().sum())} missing blind rows for {iso}")
            zs.append(z(s.to_numpy(float)))
        b = np.mean(zs, axis=0)
        # placement held EXACTLY fixed: match the reference column's mean and sd
        tgt = ref[col].to_numpy(float)
        out[col] = tgt.mean() + tgt.std() * z(b)
        print(f"{iso:<9}{'+'.join(chosen[iso]):<34}"
              f"{spearmanr(ref[col], out[col]).statistic:>24.6f}")
    out.to_csv(OUT, index=False)

    chk = pd.read_csv(OUT)
    print(f"\n{'column':<34}{'mean d':>10}{'sd ratio':>10}")
    for c in ref.columns:
        if ref[c].dtype.kind != "f":
            continue
        print(f"{c:<34}{chk[c].mean()-ref[c].mean():>+10.2e}{chk[c].std()/ref[c].std():>10.6f}")
    print(f"\nrows {len(chk)} | NaN {int(chk.isna().sum().sum())} "
          f"| inf {int(np.isinf(chk.select_dtypes('number')).sum().sum())} "
          f"| dup {int(chk.Molecule_Name.duplicated().sum())} "
          f"| SMILES preserved {(chk.SMILES==ref.SMILES).all()}")
    print(f"md5 {hashlib.md5(open(OUT,'rb').read()).hexdigest()}")
    print(f"\nwritten: {OUT}")
    print("Placement is byte-fixed by construction; the ranking is the only change.")


if __name__ == "__main__":
    main()
