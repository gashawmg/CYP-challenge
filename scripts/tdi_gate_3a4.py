r"""
tdi_gate_3a4.py  --  gate a CYP3A4 TDI candidate against the SUBMITTED column, one ruler.

This is the first TDI script in the project entitled to predict a board number, because
the ruler it uses was scored against the board and passed:

    scripts\tdi_yardstick_v2.py   CYP3A4   predicted 0.4596, board returned 0.4601, z +0.01
                                  CYP2D6   predicted 0.0780, board returned 0.2918, z +3.28

So CYP3A4 may be gated offline and CYP2D6 may not. Only CYP3A4 is handled here.

THE INCUMBENT IS THE FILE THAT SCORED, NOT A CONVENIENT PROXY. The CYP3A4 column behind
MA-MCC 0.3760 is the rank average of chemprop_best and the ChemProp shift model
(scripts\tdi_blend_cp.py). Four earlier results in this project were wrong because a
candidate was compared against something that was not the incumbent -- the pIC50 blend
member swap being the most expensive. The incumbent is reconstructed here from its own
two out-of-fold files and every arm is scored on the intersection of rows all arms share.

WHAT A PASS REQUIRES, all three:
  1 the blind-weighted AUC delta against the incumbent excludes zero in a paired
    bootstrap over compounds, P(worse) < 0.05
  2 the predicted board MCC exceeds the incumbent's by more than the board's own
    sampling noise can manufacture -- the public board is scored on about half the
    blind compounds, so that noise is +-0.054 MCC on CYP3A4 and is the real bar
  3 the gain survives at the call rate we actually submit at, not at a rate chosen
    to flatter it

Run:
  python scripts\tdi_gate_3a4.py
  python scripts\tdi_gate_3a4.py --boot 4000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdi_yardstick_v2 import (  # noqa: E402
    BLIND, BOARD, FOLDS, ROOT, TDI, auc_to_dprime, fingerprints,
    max_tanimoto, predict_board, weighted_auc,
)

TREE = ["lgb_clf", "xgb_clf", "cat_clf", "et_clf", "lgb_alert", "lgb_screen",
        "cat_screen", "pooled", "struct"]


def rank01(x):
    return rankdata(x) / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="tdiNB")
    ap.add_argument("--boot", type=int, default=2500)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    iso, b = "CYP3A4", BOARD["CYP3A4"]

    df = pd.read_csv(TDI)
    folds = pd.read_csv(FOLDS)
    blind = pd.read_csv(BLIND)
    bfp, _ = fingerprints(blind["SMILES"].tolist())

    lab = df[df[f"{iso}_is_TDI"].notna()].merge(folds, on="Molecule_Name", how="left")
    lab = lab[lab["fold"].notna()].reset_index(drop=True)
    fp, keep = fingerprints(lab["SMILES"].tolist())
    lab = lab.iloc[keep].reset_index(drop=True)
    fold = lab["fold"].to_numpy().astype(int)
    sim = np.zeros(len(lab), dtype=np.float32)
    for k in sorted(set(fold)):
        va = fold == k
        if va.sum() and (~va).sum():
            sim[va] = max_tanimoto(fp[va], fp[~va])
    blind_sim = max_tanimoto(bfp, fp)

    members = pd.read_csv(ROOT / "artifacts" / f"tdi_member_oof_{iso}_{a.pool}.csv")
    cp = pd.read_csv(ROOT / "oof_tdi_chemprop_best.csv")[["Molecule_Name", f"{iso}_is_TDI_prob"]]
    sh = pd.read_csv(ROOT / "oof_tdi_shift.csv")[["Molecule_Name", f"{iso}_shift_pred"]]

    m = lab[["Molecule_Name"]].copy()
    m["sim"] = sim
    for part in (members, cp, sh):
        m = m.merge(part, on="Molecule_Name", how="inner")
    # The ChemProp shift model does not cover every labelled row and carries a few NaNs.
    # A NaN inside rankdata poisons the whole arm silently (it cost one run of this
    # script, which reported the incumbent at AUC 0.4510 -- below chance -- and then
    # cheerfully recommended SUBMIT). Every arm is scored on rows where EVERY arm is
    # defined, and the surviving subset is printed because it is not the full fold set.
    need = [c for c in m.columns if c not in ("Molecule_Name",)]
    before = len(m)
    m = m.dropna(subset=need).reset_index(drop=True)
    y = m["y"].astype(bool).to_numpy()
    sm = m["sim"].to_numpy(float)
    print(f"rows all arms share: {len(m)} of {before} merged, "
          f"{len(members)} labelled  (dropped {before-len(m)} with a missing member)")
    print(f"positives {int(y.sum())} ({y.mean():.4f} prevalence) -- NOTE this subset is "
          f"not the full fold set, so levels here are not the yardstick's calibrated levels")

    edges = np.unique(np.quantile(blind_sim, np.linspace(0, 1, a.bins + 1)))
    edges[0], edges[-1] = -1.0, 2.0
    pb, _ = np.histogram(blind_sim, bins=edges)
    po, _ = np.histogram(sm, bins=edges)
    ratio = np.clip(np.where(po > 0, (pb / pb.sum()) / np.maximum(po / po.sum(), 1e-12), 0.0),
                    0.0, a.clip)
    w = ratio[np.clip(np.digitize(sm, edges) - 1, 0, len(ratio) - 1)]
    w = w / w.mean()
    print(f"effective n after blind reweighting: {w.sum()**2/np.sum(w**2):.0f}")

    arms = {
        "INCUMBENT (blendcp)": 0.5 * rank01(m[f"{iso}_is_TDI_prob"].to_numpy(float))
                               + 0.5 * rank01(m[f"{iso}_shift_pred"].to_numpy(float)),
        "chemprop_best alone": rank01(m[f"{iso}_is_TDI_prob"].to_numpy(float)),
        "ALL rank blend": np.mean([rank01(m[c].to_numpy(float))
                                   for c in members.columns
                                   if c not in ("Molecule_Name", "y")], axis=0),
        "TREE rank blend": np.mean([rank01(m[c].to_numpy(float))
                                    for c in TREE if c in members.columns], axis=0),
    }
    arms["TREE + chemprop + shift"] = (
        0.5 * arms["TREE rank blend"]
        + 0.25 * rank01(m[f"{iso}_is_TDI_prob"].to_numpy(float))
        + 0.25 * rank01(m[f"{iso}_shift_pred"].to_numpy(float))
    )

    base = arms["INCUMBENT (blendcp)"]
    base_auc = weighted_auc(y, base, w)
    base_pred = predict_board(auc_to_dprime(base_auc), b["prevalence"], b["call_rate"],
                              b["n_half"], rng, 4000)
    print(f"\n{'arm':<26}{'wtd AUC':>9}{'d-prime':>9}{'pred MCC':>10}"
          f"{'d MCC':>9}{'P(worse)':>10}")
    print(f"{'INCUMBENT (blendcp)':<26}{base_auc:>9.4f}{auc_to_dprime(base_auc):>9.3f}"
          f"{base_pred.mean():>10.4f}{'--':>9}{'--':>10}")

    results = []
    for name, s in arms.items():
        if name == "INCUMBENT (blendcp)":
            continue
        aw = weighted_auc(y, s, w)
        pred = predict_board(auc_to_dprime(aw), b["prevalence"], b["call_rate"],
                             b["n_half"], rng, 4000)
        d = np.empty(a.boot)
        n = len(y)
        for i in range(a.boot):
            idx = rng.integers(0, n, n)
            yy = y[idx]
            if yy.sum() == 0 or yy.sum() == n:
                d[i] = 0.0
                continue
            d[i] = weighted_auc(yy, s[idx], w[idx]) - weighted_auc(yy, base[idx], w[idx])
        pw = float((d <= 0).mean())
        results.append((name, aw, pred.mean(), pred.mean() - base_pred.mean(), pw))
        print(f"{name:<26}{aw:>9.4f}{auc_to_dprime(aw):>9.3f}{pred.mean():>10.4f}"
              f"{pred.mean()-base_pred.mean():>+9.4f}{pw:>10.3f}")

    print(f"\nboard sampling noise on the public half board: +-{base_pred.std():.4f} MCC")
    print(f"so a CYP3A4 gain under {2*base_pred.std():.4f} MCC "
          f"({base_pred.std():.4f} MA-MCC) cannot be confirmed by one submission.")
    best = max(results, key=lambda t: t[3]) if results else None
    if best:
        name, aw, pm, dm, pw = best
        ma = dm / 2.0
        print(f"\nBEST ARM: {name}")
        print(f"  predicted CYP3A4 MCC {pm:.4f} against the incumbent's {base_pred.mean():.4f}")
        print(f"  predicted MA-MCC {0.3760 + ma:.4f} against 0.3760 held "
              f"(CYP2D6 unchanged at 0.2918)")
        ok = pw < 0.05 and dm > 0
        print(f"  gate: P(worse) {pw:.3f} {'<' if pw < 0.05 else '>='} 0.05  ->  "
              f"{'SUBMIT' if ok else 'DO NOT SUBMIT'}")
        if ok and dm < 2 * base_pred.std():
            print("  CAVEAT: the predicted gain is inside the board's own half-set noise,")
            print("  so a single submission can neither confirm nor refute it.")


if __name__ == "__main__":
    main()
