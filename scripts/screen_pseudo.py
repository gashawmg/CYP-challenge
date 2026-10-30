#!/usr/bin/env python
"""Turn the single-concentration screen into ~3,000 extra pIC50 labels per isoform.

    python scripts/screen_pseudo.py                      # measure the gain
    python scripts/screen_pseudo.py --write              # + write artifacts/pseudo_pic50.csv

THE OBSERVATION. `cyp-challenge-single-concentration-TRAIN.csv` holds 4,376 compounds
measured against all four isoforms at 50 uM. We had been using it only as a binary
"screened but never promoted to a curve" signal (the left-censored floor rows in
train_cyp_st_v3.py). But its `log2fc_estimate` is a QUANTITATIVE readout, and on the
compounds that have both a screen reading and a fitted curve it tracks pIC50 at:

    CYP1A2  rho = -0.862   (n = 1412)
    CYP2C9  rho = -0.896   (n = 1285)
    CYP2D6  rho = -0.828   (n = 1493)
    CYP3A4  rho = -0.936   (n = 1805)

Read that against what our models actually achieve on the blind set -- 0.773 / 0.792 /
0.508 / 0.812. The screen is a BETTER predictor of pIC50 than our CYP2D6 model is, and
it exists for 2,883 CYP2D6 compounds that have no curve at all. Nearly tripling the
CYP2D6 label count with a signal that strong is not a small modification.

WHY A MONOTONE MAP AND NOT A REGRESSION. log2fc at a single concentration is a
saturating function of potency: once a compound kills the signal it cannot kill it
more, so the top of the pIC50 range compresses. Isotonic regression fits exactly that
shape without assuming its form, and is fitted PER FOLD on training rows only, so a
compound's pseudo-label never sees its own curve.

WHAT THE PSEUDO-LABELS ARE WORTH. They are noisier than a fitted curve, so they enter
with a weight below 1 (`--pseudo-weight`) and with the residual spread of the map
recorded per isoform, which is what a censored/interval-aware learner should use as
their uncertainty rather than treating them as exact.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from cypkit import config as C, io, splits  # noqa: E402
from scripts.run_ensemble import build_bank  # noqa: E402

LGB_R = dict(objective="regression_l1", verbose=-1, learning_rate=0.05, num_leaves=63,
             min_data_in_leaf=10, feature_fraction=0.6, bagging_fraction=0.8,
             bagging_freq=1, seed=42)


def screen_table(data_dir=None):
    """One row per compound per isoform: the screen's quantitative readout."""
    d = io.load("single_conc", data_dir)
    d.columns = [c.strip() for c in d.columns]
    g = (d.groupby([C.ID_COL, "enzyme"])
           .agg(lfc=("log2fc_estimate", "mean"),
                lfc_med=("log2fc_median", "mean"),
                se=("log2fc_std_error", "mean"),
                fdr=("log2fc_fdr", "min"))
           .reset_index())
    smi = d.drop_duplicates(C.ID_COL).set_index(C.ID_COL)[C.SMILES_COL].to_dict()
    return g, smi


def fit_map(lfc_tr, y_tr):
    """Monotone decreasing log2fc -> pIC50, plus the residual spread it leaves."""
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(lfc_tr, y_tr)
    resid = y_tr - iso.predict(lfc_tr)
    return iso, float(np.std(resid))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pseudo-weight", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0],
                    help="training weight of a pseudo-labelled row relative to a real "
                         "curve; 0.0 is the baseline (no pseudo-labels at all)")
    ap.add_argument("--n-rounds", type=int, default=400)
    ap.add_argument("--features", default="rdkit+cyp+ecfp")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    import lightgbm as lgb  # noqa: PLC0415

    inh = io.load("train_inhibition", args.data_dir)
    inh.columns = [c.strip() for c in inh.columns]
    test = io.load("test", args.data_dir)
    scr, scr_smi = screen_table(args.data_dir)

    folds = splits.load_folds(C.artifact("folds.csv"))
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))
    n_folds = int(folds["fold"].max()) + 1

    smiles_by_id = dict(zip(inh[C.ID_COL], inh[C.SMILES_COL]))
    smiles_by_id.update(scr_smi)
    smiles_by_id.update(dict(zip(test[C.ID_COL], test[C.SMILES_COL])))
    smiles_by_id = {k: v for k, v in smiles_by_id.items() if isinstance(v, str)}

    # screen-only compounds have no frozen fold; give them one so a pseudo-label
    # never lands in the same fold as a curve it could inform
    missing = sorted(set(smiles_by_id) - set(fold_of) - set(test[C.ID_COL]))
    if missing:
        fold_of.update(dict(zip(missing, splits.assign_extra_to_folds(
            [smiles_by_id[i] for i in missing], folds))))

    blocks = [b for b in ["rdkit", "cyp", "ecfp", "maccs"] if b in args.features.split("+")]
    print(f"featurising {len(smiles_by_id)} compounds, blocks = {'+'.join(blocks)}")
    bank = build_bank(smiles_by_id, blocks)
    X = bank.view("+".join(blocks))

    summary, pseudo_rows = [], []
    for iso in C.ISOFORMS:
        ep = f"{iso}_pIC50_direct_inhibition"
        real = inh.loc[inh[ep].notna(), [C.ID_COL, ep]].rename(columns={ep: "y"})
        real = real[real[C.ID_COL].isin(bank.index_of)].copy()
        real["fold"] = real[C.ID_COL].map(fold_of)

        s = scr[scr["enzyme"] == iso][[C.ID_COL, "lfc", "fdr"]]
        s = s[s[C.ID_COL].isin(bank.index_of)].copy()
        s["fold"] = s[C.ID_COL].map(fold_of)
        lfc_of = dict(zip(s[C.ID_COL], s["lfc"]))
        only = s[~s[C.ID_COL].isin(set(real[C.ID_COL]))].copy()

        both = real.merge(s[[C.ID_COL, "lfc"]], on=C.ID_COL)
        rho = spearmanr(both["lfc"], both["y"])[0]
        print(f"\n{iso}: {len(real)} curves, {len(only)} screen-only "
              f"(+{len(only)/max(len(real),1)*100:.0f}%), "
              f"screen rho on the {len(both)} overlap = {rho:+.3f}")

        idx_real = real[C.ID_COL].map(bank.index_of).to_numpy()
        idx_only = only[C.ID_COL].map(bank.index_of).to_numpy()
        y_real = real["y"].to_numpy(float)
        f_real = real["fold"].to_numpy()
        f_only = only["fold"].to_numpy()

        for w_pseudo in args.pseudo_weight:
            t0 = time.time()
            oof = np.full(len(real), np.nan)
            spreads = []
            for f in range(n_folds):
                te = f_real == f
                tr = ~te
                if te.sum() == 0 or tr.sum() < 50:
                    continue
                Xtr, ytr = X[idx_real[tr]], y_real[tr]
                wtr = np.ones(tr.sum())
                if w_pseudo > 0:
                    # map fitted on TRAINING-fold overlap only
                    ov = np.array([lfc_of.get(i, np.nan) for i in real[C.ID_COL][tr]])
                    ok = np.isfinite(ov)
                    if ok.sum() >= 50:
                        m, sd = fit_map(ov[ok], ytr[ok])
                        spreads.append(sd)
                        add = f_only != f          # keep the pseudo rows out of the fold
                        if add.sum():
                            Xtr = np.vstack([Xtr, X[idx_only[add]]])
                            ytr = np.concatenate([ytr, m.predict(only["lfc"].to_numpy()[add])])
                            wtr = np.concatenate([wtr, np.full(add.sum(), w_pseudo)])
                mdl = lgb.train(LGB_R, lgb.Dataset(Xtr, label=ytr, weight=wtr),
                                num_boost_round=args.n_rounds)
                oof[te] = mdl.predict(X[idx_real[te]])
            ok = np.isfinite(oof)
            r = float(spearmanr(y_real[ok], oof[ok])[0])
            strae = 0.834 * np.sqrt(max(1 - r ** 2, 1e-9))
            tag = "baseline" if w_pseudo == 0 else f"pseudo w={w_pseudo}"
            print(f"    {tag:16s} OOF rho {r:.4f}   proxy ST-RAE {strae:.4f}"
                  + (f"   map residual sd {np.mean(spreads):.3f}" if spreads else "")
                  + f"   ({time.time()-t0:.0f}s)")
            summary.append(dict(isoform=iso, pseudo_weight=w_pseudo, oof_rho=r,
                                proxy_strae=strae))

        if args.write:
            m, sd = fit_map(both["lfc"].to_numpy(), both["y"].to_numpy())
            pseudo_rows.append(pd.DataFrame({
                C.ID_COL: only[C.ID_COL].to_numpy(),
                "isoform": iso,
                "pseudo_pIC50": m.predict(only["lfc"].to_numpy()),
                "resid_sd": sd,
                "log2fc": only["lfc"].to_numpy(),
                "fdr": only["fdr"].to_numpy(),
            }))

    tab = pd.DataFrame(summary)
    piv = tab.pivot(index="isoform", columns="pseudo_weight", values="oof_rho")
    print("\n=== OOF Spearman by pseudo-label weight (0.0 = baseline) ===")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    best = piv.idxmax(axis=1)
    print("\nbest weight per isoform:")
    for iso in piv.index:
        b = best[iso]
        print(f"  {iso}: w={b}  rho {piv.loc[iso, 0.0]:.4f} -> {piv.loc[iso, b]:.4f} "
              f"({piv.loc[iso, b]-piv.loc[iso, 0.0]:+.4f})")
    macro0 = 0.834 * np.sqrt(1 - piv[0.0].to_numpy() ** 2)
    macrob = 0.834 * np.sqrt(1 - np.array([piv.loc[i, best[i]] for i in piv.index]) ** 2)
    print(f"\nproxy MA-ST-RAE  {macro0.mean():.4f} -> {macrob.mean():.4f}")
    tab.to_csv(C.artifact("screen_pseudo_summary.csv"), index=False)

    if args.write and pseudo_rows:
        out = pd.concat(pseudo_rows, ignore_index=True)
        out.to_csv(C.artifact("pseudo_pic50.csv"), index=False)
        print(f"written artifacts/pseudo_pic50.csv  ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
