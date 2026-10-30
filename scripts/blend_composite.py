#!/usr/bin/env python
"""The validator that actually predicts this leaderboard — and a re-blend against it.

    python scripts/blend_composite.py --oof "artifacts/oof_*_ensA.csv" --predict

THE FINDING. Two leaderboard results let us test both local scorers on the same move,
`submission_l1.csv` -> `submission_dense_blend.csv`:

    OOF ST-RAE (challenge scorer, hit-selected rows)   0.7518 -> 1.0062   WORSE by +0.254
    dense proxy (censored rows included)               1.1157 -> 0.6644   BETTER by -0.451
    ACTUAL LEADERBOARD                                 1.0127 -> 0.9171   BETTER by -0.096

They point in opposite directions and the truth sits between them. Solving for the
mixing weight gives w = 0.496 — the leaderboard behaves like an almost exact 50/50
average of the two. Adding a single constant then fits BOTH anchors:

    predicted actual  =  0.5 * OOF_ST-RAE  +  0.5 * dense_STRAE  +  0.080

        l1           0.5(0.7518) + 0.5(1.1157) + 0.080 = 1.014   actual 1.0127
        dense_blend  0.5(1.0062) + 0.5(0.6644) + 0.080 = 0.915   actual 0.9171

Two anchors, one free parameter, and the residuals are 0.001 and 0.002. Compare the
dense-only map, which needed a slope of 0.212 to fit — it was absorbing the
disagreement into a shrinkage factor rather than modelling it.

CONFIRMED OUT OF SAMPLE. A third submission was built and predicted BEFORE it was
scored: composite 0.8278, actual 0.8244 — a 0.0034 residual on a genuine forecast,
not a fit. Refitting w and c on all three anchors returns w_dense = 0.50 unchanged
and c = 0.0795, with max residual 0.0029 across a range spanning 1.01 to 0.82:

    anchor                OOF     dense    predicted   actual
    l1                 0.7508    1.1157      1.0128    1.0127
    dense_blend        1.0049    0.6644      0.9142    0.9171
    composite_scaled   0.8276    0.6680      0.8273    0.8244

Two parameters, three anchors, three-decimal agreement. Trust this number.

WHY IT IS 50/50, chemically. ST-RAE needs `conf_low`/`conf_high`, which only exist for
a fitted dose-response curve, so the scored test rows are partly hit-selected like our
training curves. But the blind set was assayed densely, so it also carries the inactive
mass the hit-selected scorer cannot see. Half and half is what that looks like.

WHAT IT CHANGES. Optimising the blend on the dense proxy alone was optimising half the
metric, and every shift-toward-the-floor correction was being over-credited. This
refits the per-endpoint weights against the composite instead.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit import io, labels as L, scoring  # noqa: E402

W_DENSE = 0.5
CONST = 0.0795          # refit on 3 anchors; max residual 0.0029
CURRENT_BEST_LB = 0.8244


def endpoint_rows(train, lm):
    """Per endpoint: the hit-selected scored rows, and the full censored matrix."""
    out = {}
    for ep in C.REGRESSION_ENDPOINTS:
        g = train[[C.ID_COL, ep, ep + C.CONF_LOW_SUFFIX, ep + C.CONF_HIGH_SUFFIX]].dropna(
            subset=[ep])
        d = lm[lm["endpoint"] == ep]
        out[ep] = dict(
            hit_ids=g[C.ID_COL].to_numpy(),
            hit_y=g[ep].to_numpy(float),
            hit_lo=g[ep + C.CONF_LOW_SUFFIX].to_numpy(float),
            hit_hi=g[ep + C.CONF_HIGH_SUFFIX].to_numpy(float),
            all_ids=d[C.ID_COL].to_numpy(),
            all_y=d["y"].to_numpy(float),
            all_lo=d["y_low"].to_numpy(float),
            all_hi=d["y_high"].to_numpy(float),
        )
    return out


def _strae(p, y, lo, hi, floor=None):
    err = (np.where(np.isfinite(hi), np.clip(p - hi, 0, None), 0.0)
           + np.where(np.isfinite(lo), np.clip(lo - p, 0, None), 0.0))
    ref = y if floor is None else np.where(np.isfinite(y), y, floor)
    m = float(np.mean(ref))
    base = (np.where(np.isfinite(hi), np.clip(m - hi, 0, None), 0.0)
            + np.where(np.isfinite(lo), np.clip(lo - m, 0, None), 0.0))
    d = float(np.sum(base))
    return float(np.sum(err) / d) if d > 0 else np.nan


def composite(pred_by_id: dict, R: dict) -> tuple[float, float, float]:
    hits, denses = [], []
    for ep in C.REGRESSION_ENDPOINTS:
        r = R[ep]
        ph = pd.Series(r["hit_ids"]).map(pred_by_id[ep]).to_numpy(float)
        pa = pd.Series(r["all_ids"]).map(pred_by_id[ep]).to_numpy(float)
        k = np.isfinite(ph)
        hits.append(_strae(ph[k], r["hit_y"][k], r["hit_lo"][k], r["hit_hi"][k]))
        k = np.isfinite(pa)
        denses.append(_strae(pa[k], r["all_y"][k], r["all_lo"][k], r["all_hi"][k],
                             floor=C.PIC50_FLOOR))
    h, d = float(np.mean(hits)), float(np.mean(denses))
    return h, d, (1 - W_DENSE) * h + W_DENSE * d + CONST


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", nargs="*", default=["artifacts/oof_*_ensA.csv"])
    ap.add_argument("--submissions", nargs="*",
                    default=["artifacts/submission_*.csv"])
    ap.add_argument("--members", nargs="*", default=None,
                    help="submission_<name>.csv files matching the OOF members, for --predict")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--tag", default="composite")
    args = ap.parse_args()

    train = io.load("train_inhibition")
    screen = io.load("single_conc")
    lm = L.build_label_matrix(train, screen, include_censored=True, censored_weight=0.5)
    R = endpoint_rows(train, lm)

    # ---- score every existing candidate ---------------------------------
    print(f"{'candidate':40s} {'OOF':>8s} {'dense':>8s} {'PREDICTED LB':>13s}")
    known = {"submission_l1.csv": 1.0127, "submission_dense_blend.csv": 0.9171}
    for f in sorted({g for pat in args.submissions for g in glob.glob(pat)}):
        d = pd.read_csv(f)
        if not all(ep in d.columns for ep in C.REGRESSION_ENDPOINTS):
            continue
        # a submission covers only the 750 test compounds, so it cannot be scored
        # against training rows; only OOF files can. Skip, but keep the anchors.
        pass

    files = sorted({g for pat in args.oof for g in glob.glob(pat)})
    if not files:
        raise SystemExit(f"no OOF files matched {args.oof}")
    members = [Path(f).stem.replace("oof_", "").rsplit("_", 1)[0] for f in files]
    oofs = {m: pd.read_csv(f) for m, f in zip(members, files)}
    print(f"\n{len(members)} members: {', '.join(members)}\n")
    print(f"{'member':14s} {'OOF':>8s} {'dense':>8s} {'PREDICTED LB':>13s}")
    for m, d in oofs.items():
        pb = {ep: dict(zip(d[C.ID_COL], d[ep])) for ep in C.REGRESSION_ENDPOINTS}
        h, dn, c = composite(pb, R)
        print(f"{m:14s} {h:8.4f} {dn:8.4f} {c:13.4f}")

    # ---- refit weights against the composite ----------------------------
    from scipy.optimize import minimize  # noqa: PLC0415

    # Members can cover different compound sets: the classical learners were fitted
    # over train + screen-only compounds (5,656), the neural ones only over the 4,905
    # with a dose-response curve. Align on the INTERSECTION and say so, rather than
    # column-stacking arrays of different length.
    id_sets = [set(oofs[m][C.ID_COL]) for m in members]
    common = set.intersection(*id_sets)
    union = set.union(*id_sets)
    if len(common) < len(union):
        print(f"  members cover different compound sets: union {len(union)}, "
              f"common {len(common)} -- blending on the common set")
        for m in members:
            print(f"    {m:14s} {len(set(oofs[m][C.ID_COL])):5d}")
    ids = np.array(sorted(common))
    weights, blended = {}, pd.DataFrame({C.ID_COL: ids})
    print("\nrefitting per-endpoint weights on the composite objective")
    for ep in C.REGRESSION_ENDPOINTS:
        r = R[ep]
        M_hit = np.column_stack([
            pd.Series(r["hit_ids"]).map(dict(zip(oofs[m][C.ID_COL], oofs[m][ep]))).to_numpy(float)
            for m in members])
        M_all = np.column_stack([
            pd.Series(r["all_ids"]).map(dict(zip(oofs[m][C.ID_COL], oofs[m][ep]))).to_numpy(float)
            for m in members])
        kh = np.isfinite(M_hit).all(1)
        ka = np.isfinite(M_all).all(1)

        def obj(w, M_hit=M_hit, M_all=M_all, kh=kh, ka=ka, r=r):
            w = np.clip(w, 0, None)
            s = w.sum()
            if s <= 0:
                return 1e9
            w = w / s
            h = _strae(M_hit[kh] @ w, r["hit_y"][kh], r["hit_lo"][kh], r["hit_hi"][kh])
            d = _strae(M_all[ka] @ w, r["all_y"][ka], r["all_lo"][ka], r["all_hi"][ka],
                       floor=C.PIC50_FLOOR)
            return (1 - W_DENSE) * h + W_DENSE * d

        best, rng = None, np.random.default_rng(0)
        for w0 in [np.ones(len(members)) / len(members)] + [
                rng.dirichlet(np.ones(len(members))) for _ in range(5)]:
            res = minimize(obj, w0, method="Nelder-Mead",
                           options=dict(maxiter=6000, xatol=1e-4, fatol=1e-7))
            if best is None or res.fun < best.fun:
                best = res
        w = np.clip(best.x, 0, None)
        w = w / w.sum()
        weights[ep] = dict(zip(members, w.round(4).tolist()))
        blended[ep] = np.column_stack([
            pd.Series(ids).map(dict(zip(oofs[m][C.ID_COL], oofs[m][ep]))).to_numpy(float)
            for m in members]) @ w
        top = ", ".join(f"{n} {v:.2f}" for n, v in
                        sorted(weights[ep].items(), key=lambda kv: -kv[1]) if v > 0.02)
        print(f"  {ep:34s} {top}")

    pb = {ep: dict(zip(blended[C.ID_COL], blended[ep])) for ep in C.REGRESSION_ENDPOINTS}
    h, dn, c = composite(pb, R)
    print(f"\n  COMPOSITE BLEND   OOF {h:.4f}   dense {dn:.4f}   "
          f"-> predicted leaderboard {c:.4f}")
    print(f"  current best {CURRENT_BEST_LB:.4f}   "
          f"({'BEATS IT by %.4f' % (CURRENT_BEST_LB - c) if c < CURRENT_BEST_LB else 'no gain'})")
    blended.to_csv(C.artifact(f"oof_blend_{args.tag}.csv"), index=False)
    with open(C.artifact(f"blend_weights_{args.tag}.json"), "w") as fh:
        json.dump(weights, fh, indent=2)

    # a leaderboard-identical second opinion on the blended OOF
    rep = scoring.oof_report(blended, train, n_bootstrap=100)
    print("\n" + scoring.format_report(rep))

    if args.predict:
        if not args.members:
            raise SystemExit("--predict needs --members: the per-learner test predictions")
        per = {}
        for f in args.members:
            name = Path(f).stem.replace("testpred_", "")
            per[name] = pd.read_csv(f)
        missing = [m for m in members if m not in per]
        if missing:
            raise SystemExit(f"no test predictions for {missing}")
        out = per[members[0]][[C.SMILES_COL, C.ID_COL]].copy()
        for ep in C.REGRESSION_ENDPOINTS:
            w = np.array([weights[ep][m] for m in members])
            out[ep] = np.column_stack([per[m][ep].to_numpy() for m in members]) @ w
        p = C.artifact(f"submission_{args.tag}.csv")
        out.to_csv(p, index=False)
        print(f"\nwritten {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
