#!/usr/bin/env python
r"""Train pIC50 on what ST-RAE actually charges for, instead of on every compound equally.

    python train_pic50_interval.py                 # all four endpoints
    python train_pic50_interval.py --iso CYP3A4

THE FINDING THIS TESTS. ST-RAE charges `clip(p - hi, 0) + clip(lo - p, 0)`: a prediction
inside its confidence interval costs nothing, and a narrow interval starts charging almost
immediately. So the score is concentrated in the well-measured compounds -- and on the
blind-matched pool it is concentrated brutally:

    endpoint   share of ST-RAE cost in the narrowest quartile   our rho there   rho in the widest
    CYP1A2                43.3%                                     0.389            0.236
    CYP2C9                38.2%                                     0.392            0.539   <- inverted
    CYP2D6                89.2%                                     0.513           -0.068
    CYP3A4                29.5%                                     0.471            0.575   <- inverted

On CYP2C9 and CYP3A4 we rank BETTER on the compounds that cost nothing and WORSE on the
ones that cost everything. Those are exactly the two endpoints where `wbot` beats our
absorption (0.697 vs 0.765, 0.738 vs 0.833). Every model this project has trained used a
uniform loss, so nothing has ever told it which compounds matter.

THREE ARMS, IDENTICAL FEATURES AND FOLDS, one thing different:

    control     L2 on y, every compound weighted 1          <- what we have always done
    weighted    L2 on y, weight proportional to 1/interval width
    interval    the METRIC ITSELF as the training objective: zero gradient inside
                [lo, hi], +-1 outside. Piecewise linear, so a constant hessian.

The third is the honest version of the idea -- not a proxy for the metric but the metric,
optimised directly.

WHY LIGHTGBM AND NOT CHEMPROP. chemprop does not import in this container (RDKit ABI), and
a descriptor baseline is the right place to test a LOSS anyway: the control runs beside the
candidate, so the comparison is about the objective and nothing else. The absolute numbers
will read below v6; the control-to-candidate gap is the result. If it holds, the same three
lines port into train_cyp_st_v3.py, where they would run on the model that ships.

SCORED ON THE CALIBRATED POOL. Not our raw folds -- the uncensored, blind-histogram-matched
pool whose affine absorption is 0.807 against the 0.806 we actually score on the board. It
is the only local yardstick this project has that predicts the blind result, and it exists
because fitting against censored rows once produced a 13% phantom gain.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

from cypkit import config as C, cyp_descriptors as CD  # noqa: E402
from scripts.blind_validator import BLIND, eval_pool, match_weights, strae  # noqa: E402

RDLogger.DisableLog("rdApp.*")
WIDTH_FLOOR, WIDTH_CEIL = 0.10, 2.50      # the observed range; clipping caps the leverage


def featurise(smiles: list[str]) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = np.zeros((len(smiles), 1024), np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            fp[i] = np.asarray(gen.GetFingerprintAsNumPy(m), np.float32)
    desc, _ = CD.calculate(smiles)
    desc = np.nan_to_num(np.asarray(desc, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.hstack([fp, desc])


def interval_objective(lo: np.ndarray, hi: np.ndarray):
    """ST-RAE's numerator as a LightGBM objective.

    cost(p) = clip(p - hi, 0) + clip(lo - p, 0)
    grad    = +1 above hi, -1 below lo, 0 inside -- the metric has no opinion inside
    hess    = 1 everywhere (the loss is piecewise linear, so the true hessian is 0 and
              LightGBM needs a positive constant to take a step)
    """
    # SIGNATURE. LightGBM's sklearn API calls this as (y_true, y_pred) -- NOT the
    # (preds, dataset) of the native lgb.train API. Getting that backwards computed the
    # gradient from y against y's OWN interval, and since every label sits inside its own
    # confidence bounds by construction the gradient was identically zero at every step:
    # the model never moved off its init and predicted a constant. A silent, total no-op.
    def obj(y_true: np.ndarray, pred: np.ndarray):
        g = np.zeros_like(pred)
        g[pred > hi] = 1.0
        g[pred < lo] = -1.0
        return g, np.ones_like(pred)
    return obj


def fit(arm: str, X, y, lo, hi, w, Xte):
    import lightgbm as lgb  # noqa: PLC0415
    p = dict(n_estimators=700, learning_rate=0.04, num_leaves=63, min_child_samples=20,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
             verbose=-1, n_jobs=-1)
    if arm == "interval":
        # init_score matters: with a hinge loss and a zero start, every prediction begins
        # far below every interval and the first trees are spent walking up.
        m = lgb.LGBMRegressor(objective=interval_objective(lo, hi), **p)
        m.fit(X, y, init_score=np.full(len(y), float(np.mean(y))))
        return m.predict(Xte) + float(np.mean(y))
    m = lgb.LGBMRegressor(**p)
    m.fit(X, y, sample_weight=w)
    return m.predict(Xte)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", default="all")
    ap.add_argument("--out", default="oof_st_ivw.csv")
    a = ap.parse_args()

    lab = pd.read_csv(C.artifact("labels.csv"))
    folds = pd.read_csv(C.artifact("folds.csv"))
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))
    pool = eval_pool()
    rng = np.random.default_rng(0)
    isos = C.ISOFORMS if a.iso == "all" else (a.iso,)
    oof_frames, summary = [], []

    for iso in isos:
        # train on real fitted curves only -- a censored row has no y to regress and its
        # interval is unbounded below, which the hinge loss would read as "free".
        t = lab[(lab["isoform"] == iso) & (lab["kind"] != "left")].copy()
        t = t[t[C.ID_COL].isin(fold_of)].dropna(subset=["y", "y_low", "y_high"])
        t["fold"] = t[C.ID_COL].map(fold_of)
        t = t.reset_index(drop=True)
        X = featurise(t[C.SMILES_COL].tolist())
        y = t["y"].to_numpy(float)
        lo, hi = t["y_low"].to_numpy(float), t["y_high"].to_numpy(float)
        width = np.clip(hi - lo, WIDTH_FLOOR, WIDTH_CEIL)
        w = 1.0 / width
        w = w / w.mean()

        print(f"\n=== {iso}  {len(t)} fitted curves, interval width "
              f"{np.median(hi - lo):.2f} median, weight {w.min():.2f}-{w.max():.2f}")

        preds = {}
        for arm in ("control", "weighted", "interval"):
            o = np.zeros(len(t))
            for f in sorted(t["fold"].unique()):
                tr, te = (t["fold"] != f).to_numpy(), (t["fold"] == f).to_numpy()
                o[te] = fit(arm, X[tr], y[tr], lo[tr], hi[tr],
                            None if arm == "control" else w[tr], X[te])
            preds[arm] = o

        # ---- score on the calibrated pool ------------------------------------------
        s = pool[(pool["isoform"] == iso) & (~pool["censored"])]
        s = s[s[C.ID_COL].isin(t[C.ID_COL])]
        ids = s[C.ID_COL].to_numpy()
        ys = s["y_eval"].to_numpy(float)
        los, his = s["y_low"].to_numpy(float), s["y_high"].to_numpy(float)
        ww, _ = match_weights(ys, np.zeros(len(ys), bool), *BLIND[iso])
        draws = [rng.choice(len(ys), size=len(ys), replace=True, p=ww) for _ in range(40)]
        narrow = (his - los) <= np.quantile(his - los, 0.25)
        by_id = {m: i for i, m in enumerate(t[C.ID_COL])}
        idx = np.array([by_id[m] for m in ids])

        print(f"  {'arm':10s} {'rho':>7s} {'rho|narrow 25%':>15s} {'ST-RAE':>8s} "
              f"{'absorption':>11s}")
        for arm, o in preds.items():
            p = o[idx]
            rho = float(pd.Series(p).corr(pd.Series(ys), method="spearman"))
            rn = float(pd.Series(p[narrow]).corr(pd.Series(ys[narrow]), method="spearman"))
            # place affinely at the blind anchors so ST-RAE is on the board's scale
            z = (p - p.mean()) / p.std()
            pp = z * BLIND[iso][1] * rho + BLIND[iso][0]
            st = float(np.mean([strae(pp[d], los[d], his[d], ys[d]) for d in draws]))
            ab = st / np.sqrt(1 - rho ** 2)
            print(f"  {arm:10s} {rho:7.4f} {rn:15.4f} {st:8.4f} {ab:11.3f}")
            summary.append(dict(isoform=iso, arm=arm, rho=rho, rho_narrow=rn,
                                strae=st, absorption=ab))
        oof_frames.append(pd.DataFrame({
            C.ID_COL: t[C.ID_COL],
            **{f"{iso}_pIC50_direct_inhibition_{k}": v for k, v in preds.items()}}))

    r = pd.DataFrame(summary)
    print("\nMACRO")
    print(f"  {'arm':10s} {'rho':>7s} {'rho|narrow':>11s} {'absorption':>11s}")
    for arm in ("control", "weighted", "interval"):
        g = r[r.arm == arm]
        print(f"  {arm:10s} {g.rho.mean():7.4f} {g.rho_narrow.mean():11.4f} "
              f"{g.absorption.mean():11.3f}")
    base = r[r.arm == "control"].absorption.mean()
    for arm in ("weighted", "interval"):
        v = r[r.arm == arm].absorption.mean()
        print(f"  {arm} vs control: absorption {100 * (1 - v / base):+.1f}%")
    out = oof_frames[0]
    for f in oof_frames[1:]:
        out = out.merge(f, on=C.ID_COL, how="outer")
    out.to_csv(a.out, index=False)
    r.to_csv("artifacts/interval_weight_summary.csv", index=False)
    print(f"\nwrote {a.out} and artifacts/interval_weight_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
