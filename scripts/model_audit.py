#!/usr/bin/env python
r"""Score EVERY model this project has ever produced, on the yardstick that predicts the
board, per endpoint.

    python scripts\model_audit.py
    python scripts\model_audit.py --boot 600

WHY. Roughly forty out-of-fold prediction files sit in this repo. They were produced over
weeks, compared in twos and threes, against a pool we now know was wrong on three of four
endpoints (Section 30, Section 32). No one has ever put them all on one table under one
ruler. Model choices were made pairwise, under a distortion, and never revisited.

THE RULER, chosen per endpoint by which weighting reproduces the board (Section 32):

    CYP1A2   hist x sim   ratio 1.088
    CYP2C9   histogram    ratio 1.053
    CYP2D6   plain        ratio 1.036   <- the label-histogram match BREAKS this endpoint
    CYP3A4   hist x sim   ratio 1.000      (the (2.922,1.652) anchor, Section 20)

WHAT THE AUDIT IS LOOKING FOR, stated before it runs so the answer cannot be fitted to it:

1.  PER-TARGET CHEMPROP WITH TARGET-SPECIFIC DESCRIPTORS (train_cyp_st_v2/v3, fusing
    cyp_target_descriptors' isoform blocks at the readout) against MULTITASK CHEMPROP
    (train_cyp_mt_v1, scripts/run_chemprop.py, one shared 267-descriptor vector). These
    are the two architectures the project built and never benchmarked head to head. The
    blend member `cp` is the single-task one; nothing recorded why.

2.  k-NN. It won TDI CYP2D6 on blind-like rows (Section 25) and lost badly on pIC50 under
    the OLD yardstick (Section 30). Classification and regression may genuinely differ,
    but the pIC50 verdict was reached on the ruler we have since replaced.

3.  SVM. Section 2 of WHAT_WORKED records a rival matching our MAE and rho on CYP2C9 while
    scoring 10% better ST-RAE, never explained. We have oof_svr_tan_ensA.csv in hand.

4.  Whether the four members we actually ship are the four best we have.

Selection is on Spearman: it is invariant under the affine placement, it is the only thing
a model can move now that placement and blending are closed, and it is the quantity whose
transfer to the board has been measured (about 14%).

NOTHING HERE IS A SUBMISSION. It says which model family deserves the next training run.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from scripts.blind_validator import BLIND, eval_pool, match_weights  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EDGES = np.array([0.0, .30, .35, .40, .45, .50, .55, .60, .65, .70, 1.01])
# Section 32: which weighting reproduces the board, endpoint by endpoint.
YARDSTICK = {"CYP1A2": "hist x sim", "CYP2C9": "histogram",
             "CYP2D6": "plain", "CYP3A4": "hist x sim"}
BLIND_RHO = {"CYP1A2": 0.8064, "CYP2C9": 0.8454, "CYP2D6": 0.5190, "CYP3A4": 0.8327}
SHIPPED = {"oof_st_v6", "oof_st_mh", "oof_chempropv3_ensA", "oof_unimol_cyp_mt"}


def family(name: str) -> str:
    n = name.lower()
    if "chemprop" in n and ("st_v" in n or "chempropst" in n or "chempropv" in n):
        return "chemprop per-target + target desc"
    if "chemprop" in n:
        return "chemprop multitask"
    if "knn" in n:
        return "k-NN Tanimoto"
    if "svr" in n or "svm" in n:
        return "SVM"
    if "unimol" in n:
        return "Uni-Mol 3D"
    if n.startswith("oof_bl") or n.startswith("oof_bz") or "blend" in n or "avg" in n:
        return "blend/derived"
    if any(k in n for k in ("lgb", "xgb", "cat", "_et_", "deadzone", "l1_", "tutorial")):
        return "boosted trees / classical"
    return "descriptor ensemble"


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
    w = np.clip(np.nan_to_num(np.asarray(w, float), nan=0.0), 0, None)
    return w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))


def col_for(d: pd.DataFrame, iso: str):
    hits = [c for c in d.columns if c.startswith(iso) and "pIC50" in c]
    return hits[0] if hits else next((c for c in d.columns if c.startswith(iso)), None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--min-rows", type=int, default=400)
    a = ap.parse_args()

    files = sorted(set(list(ROOT.glob("oof_*.csv")) + list((ROOT / "artifacts").glob("oof_*.csv"))))
    print(f"  {len(files)} out-of-fold files found\n")

    pool = eval_pool()
    folds = pd.read_csv(C.artifact("folds.csv")); folds.columns = [c.strip() for c in folds.columns]
    sim_of = dict(zip(folds[C.ID_COL], folds["max_sim_to_other_folds"]))
    nnf = pd.read_csv(C.artifact("nnsim.csv")); nnf.columns = [c.strip() for c in nnf.columns]
    nn_blind = nnf.loc[nnf["split"] != "train", "nnsim"].to_numpy(float)

    loaded = {}
    for f in files:
        try:
            d = pd.read_csv(f)
            d.columns = [c.strip() for c in d.columns]
            if C.ID_COL in d.columns:
                loaded[f.stem] = d.set_index(C.ID_COL)
        except Exception as e:                                          # noqa: BLE001
            print(f"  skipping {f.name}: {e}")

    rng = np.random.default_rng(0)
    rows = []
    for iso in C.ISOFORMS:
        s = pool[(pool["isoform"] == iso) & (~pool["censored"])]
        s = s[s[C.ID_COL].isin(sim_of)]
        ids_all = s[C.ID_COL].to_numpy()
        y_all = s["y_eval"].to_numpy(float)
        nn_all = np.array([sim_of[i] for i in ids_all], float)
        cen_all = s["censored"].to_numpy(bool)
        m, sd = BLIND[iso]

        print(f"\n=== {iso}   pool {len(ids_all)} uncensored rows   "
              f"yardstick: {YARDSTICK[iso]}")
        scored = []
        for name, d in loaded.items():
            c = col_for(d, iso)
            if c is None:
                continue
            v = pd.to_numeric(d[c], errors="coerce")
            have = np.array([i in v.index for i in ids_all])
            vals = np.full(len(ids_all), np.nan)
            vals[have] = v.reindex(ids_all[have]).to_numpy(float)
            ok = np.isfinite(vals)
            if ok.sum() < a.min_rows or float(np.nanstd(vals[ok]) or 0) < 1e-9:
                continue
            yy, pp_, nn = y_all[ok], vals[ok], nn_all[ok]
            wh, _ = match_weights(yy, cen_all[ok], m, sd)
            w_hist = norm1(np.ones(len(yy)) if wh is None else wh)
            W = {"plain": norm1(np.ones(len(yy))), "histogram": w_hist,
                 "hist x sim": norm1(w_hist * sim_weights(nn, nn_blind))}
            w = W[YARDSTICK[iso]]
            b = []
            for _ in range(a.boot):
                i = rng.choice(len(yy), len(yy), replace=True, p=w)
                if len(np.unique(yy[i])) > 2:
                    b.append(spearman(pp_[i], yy[i]))
            b = np.asarray(b)
            scored.append(dict(isoform=iso, model=name, family=family(name),
                               n=int(ok.sum()), rho_plain=spearman(pp_, yy),
                               rho=float(b.mean()), lo=float(np.quantile(b, .025)),
                               hi=float(np.quantile(b, .975)),
                               shipped=name in SHIPPED))
        sc = pd.DataFrame(scored).sort_values("rho", ascending=False)
        rows.append(sc)
        print(f"  {'rank':>4s} {'model':34s} {'family':34s} {'n':>5s} {'rho':>7s} "
              f"{'95% CI':>18s}")
        for k, (_, r) in enumerate(sc.head(12).iterrows(), 1):
            mark = " <== SHIPPED" if r.shipped else ""
            print(f"  {k:4d} {r.model:34s} {r.family:34s} {r.n:5d} {r.rho:7.4f} "
                  f"[{r.lo:+.4f},{r.hi:+.4f}]{mark}")
        ship = sc[sc.shipped]
        if len(ship):
            worst = ship.rho.min()
            better = sc[(sc.rho > sc[sc.shipped].rho.max()) & (~sc.shipped)]
            print(f"  shipped members rank "
                  f"{', '.join(str(sc.index.get_loc(i) + 1) for i in ship.index)} "
                  f"of {len(sc)}; {len(better)} non-shipped model(s) beat the best of them")

    d = pd.concat(rows, ignore_index=True)
    d.to_csv(ROOT / "artifacts" / "model_audit.csv", index=False)

    print("\n" + "=" * 78)
    print("BY FAMILY -- mean rho over the four endpoints, best member of each family\n")
    best = d.loc[d.groupby(["isoform", "family"]).rho.idxmax()]
    piv = best.pivot_table(index="family", columns="isoform", values="rho")
    piv["MEAN"] = piv.mean(axis=1)
    print(piv.sort_values("MEAN", ascending=False).round(4).to_string())

    print("\n\nTHE HEAD-TO-HEAD THIS PROJECT NEVER RAN\n")
    for iso in C.ISOFORMS:
        g = d[d.isoform == iso]
        st = g[g.family == "chemprop per-target + target desc"]
        mt = g[g.family == "chemprop multitask"]
        kn = g[g.family == "k-NN Tanimoto"]
        f = lambda x: f"{x.rho.max():.4f}" if len(x) else "  --  "   # noqa: E731
        print(f"  {iso:8s} per-target+desc {f(st)}   multitask {f(mt)}   "
              f"k-NN {f(kn)}   board {BLIND_RHO[iso]:.4f}")
    print("\n  wrote artifacts/model_audit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
