#!/usr/bin/env python
"""Recover HONEST UniMol out-of-fold predictions from a finished Kaggle run. No GPU.

    python scripts/unimol_oof_from_cvdata.py --model-dir <dir with cv.data> \
        --out artifacts/oof_unimol_honest.csv [--censored]

WHY THIS EXISTS. `unimol_tools` writes the true out-of-fold matrix during training:
models/nnmodel.py does `y_pred[te_idx] = _y_pred` for every fold and dumps the assembled
array to `cv.data`. That is the OOF matrix in the library's OWN split, by construction.

The Kaggle notebooks used to rebuild the split themselves to produce OOF, and got it
wrong: they computed Murcko scaffolds with includeChirality=False while the library uses
includeChirality=True. That changed 350 of 4,905 scaffold strings, and because GroupKFold
assigns whole groups and ~4,520 of the groups are singletons, the resulting partition was
unrelated to the real one -- adjusted Rand index 0.0011, best fold-label permutation
overlapping on 22% of rows against 20% for chance. Roughly 78% of "out-of-fold" rows were
scored by a model that had trained on them.

So every UniMol OOF number in this project is optimistic, and any blend weight fitted on
one is inflated. This script replaces them without retraining anything.

WHAT YOU NEED FROM KAGGLE. Two small files from the run's output directory:

    cv.data            the scaled OOF matrix
    target_scaler.ss   the per-column scalers (a joblib list of sklearn scalers)

Both are a few hundred KB. You do NOT need the model_*.pth files, a GPU, or
unimol_tools installed -- the scalers are plain sklearn objects.

ROW ORDER. cv.data has one row per TRAINING row, in the order the notebook built it,
after conformer failures were filtered out. This script rebuilds that frame the same way
and refuses to guess if the counts disagree, because a silent off-by-N would produce a
plausible-looking file with every label attached to the wrong molecule.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from cypkit import config as C

ISOFORMS = list(C.ISOFORMS)
TARGET_COLS = [f"{c}_pIC50_direct_inhibition" for c in ISOFORMS]
HIT_LOG2FC_MAX, HIT_FDR_MAX, PIC50_FLOOR = -1.0, 0.05, 4.0


def curve_frame(train_csv):
    df = pd.read_csv(train_csv)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=[C.SMILES_COL]).reset_index(drop=True)
    Y = df[TARGET_COLS].to_numpy(float)
    return df[np.isfinite(Y).any(axis=1)].reset_index(drop=True)


def censored_frame(curve_df, screen_csv, fraction=0.5, seed=42):
    """Reproduce Cell 4 of the censored notebook exactly."""
    s = pd.read_csv(screen_csv)
    s.columns = [c.strip() for c in s.columns]
    s["is_hit"] = (s["log2fc_estimate"] < HIT_LOG2FC_MAX) & (s["log2fc_fdr"] < HIT_FDR_MAX)
    non = s[~s["is_hit"]]
    have = {ISOFORMS[j]: set(curve_df.loc[np.isfinite(curve_df[ep].to_numpy(float)),
                                          C.ID_COL])
            for j, ep in enumerate(TARGET_COLS)}
    rows = {}
    for r in non.itertuples(index=False):
        iso = getattr(r, "enzyme", None)
        if iso not in ISOFORMS:
            continue
        name = getattr(r, C.ID_COL)
        if name in have[iso]:
            continue
        rec = rows.setdefault(name, {C.ID_COL: name, C.SMILES_COL: getattr(r, C.SMILES_COL),
                                     **{ep: np.nan for ep in TARGET_COLS}})
        rec[f"{iso}_pIC50_direct_inhibition"] = PIC50_FLOOR
    out = pd.DataFrame(list(rows.values()))
    if 0 < fraction < 1 and len(out):
        out = out.sample(frac=fraction, random_state=seed).reset_index(drop=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True,
                    help="directory holding cv.data and target_scaler.ss")
    ap.add_argument("--out", default="artifacts/oof_unimol_honest.csv")
    ap.add_argument("--train-csv", default="data/cyp-challenge-TRAIN_inhibition.csv")
    ap.add_argument("--screen-csv", default="data/cyp-challenge-single-concentration-TRAIN.csv")
    ap.add_argument("--censored", action="store_true",
                    help="the run included screen-derived floor rows "
                         "(unimol_kaggle_cyp_censored_5fold)")
    ap.add_argument("--censored-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="write anyway on a row-count mismatch. Do not use unless you "
                         "know exactly which conformers failed.")
    args = ap.parse_args()

    import joblib  # noqa: PLC0415

    d = Path(args.model_dir)
    cv = np.asarray(joblib.load(d / "cv.data"), dtype=float)
    if cv.ndim == 1:
        cv = cv.reshape(-1, len(TARGET_COLS))
    print(f"cv.data           : {cv.shape}")

    curve = curve_frame(args.train_csv)
    frame = curve
    if args.censored:
        cens = censored_frame(curve, args.screen_csv, args.censored_fraction, args.seed)
        dup = set(curve[C.ID_COL]) & set(cens[C.ID_COL])
        if dup:
            merged = cens[cens[C.ID_COL].isin(dup)].set_index(C.ID_COL)
            curve = curve.set_index(C.ID_COL)
            for ep in TARGET_COLS:
                curve[ep] = curve[ep].where(curve[ep].notna(),
                                            merged[ep].reindex(curve.index))
            curve = curve.reset_index()
            cens = cens[~cens[C.ID_COL].isin(dup)].reset_index(drop=True)
        frame = pd.concat([curve, cens], ignore_index=True)
        print(f"rebuilt frame     : {len(curve)} curve + {len(cens)} floor "
              f"= {len(frame)} rows")
    else:
        print(f"rebuilt frame     : {len(frame)} curve rows")

    if len(frame) != cv.shape[0]:
        msg = (f"\nROW COUNT MISMATCH: cv.data has {cv.shape[0]} rows, the rebuilt "
               f"frame has {len(frame)}.\nThe difference is almost certainly conformer "
               f"failures, which the notebook filtered out before training.\nThe run's "
               f"Cell 5 printed how many failed -- if it printed 0, one of the config "
               f"values\nhere is wrong (--censored / --censored-fraction / --seed). "
               f"Mapping rows without\nknowing WHICH ones dropped would attach every "
               f"prediction to the wrong molecule.")
        if not args.force:
            raise SystemExit(msg + "\n\nRefusing to guess. Re-run with --force only if "
                                   "you know the failed indices.")
        print(msg + "\n(--force given; truncating to the shorter of the two)")
        n = min(len(frame), cv.shape[0])
        frame, cv = frame.iloc[:n], cv[:n]

    # cv.data is stored SCALED. target_scaler.ss is a plain joblib list of sklearn
    # scalers, one per target column, so no unimol_tools import is needed.
    sc_path = d / "target_scaler.ss"
    if sc_path.exists():
        scalers = joblib.load(sc_path)
        if isinstance(scalers, list) and len(scalers) == cv.shape[1]:
            cv = np.column_stack([scalers[j].inverse_transform(cv[:, j:j + 1]).ravel()
                                  for j in range(cv.shape[1])])
            print("inverse-transformed with target_scaler.ss")
        else:
            print(f"WARNING: target_scaler.ss is not a per-column list "
                  f"({type(scalers).__name__}); leaving values as stored.")
    else:
        print("WARNING: no target_scaler.ss -- values stay on the scaled axis. "
              "Rankings are still valid; absolute pIC50 values are not.")

    out = pd.DataFrame({C.ID_COL: frame[C.ID_COL].to_numpy()})
    for j, ep in enumerate(TARGET_COLS):
        out[ep] = cv[:, j]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwritten {args.out}  ({len(out)} rows)")

    # Score it against the curves, which is the only ground truth we have.
    tr = pd.read_csv(args.train_csv)
    tr.columns = [c.strip() for c in tr.columns]
    print(f"\n{'endpoint':8s} {'n':>6s} {'Spearman':>10s} {'MAE':>8s}"
          f"    (contaminated run reported)")
    prev = dict(CYP1A2=0.4838, CYP2C9=0.6021, CYP2D6=0.4261, CYP3A4=0.7525)
    rr = []
    for ep, iso in zip(TARGET_COLS, ISOFORMS):
        m = tr[[C.ID_COL, ep]].dropna().merge(out[[C.ID_COL, ep]], on=C.ID_COL,
                                              suffixes=("", "_p")).dropna()
        if len(m) < 20:
            continue
        r = float(m[ep].rank().corr(m[ep + "_p"].rank()))
        mae = float(np.mean(np.abs(m[ep] - m[ep + "_p"])))
        rr.append(r)
        print(f"{iso:8s} {len(m):6d} {r:10.4f} {mae:8.4f}    {prev[iso]:.4f}"
              f"  ({r - prev[iso]:+.4f})")
    print(f"{'MACRO':8s} {'':6s} {np.mean(rr):10.4f}")
    print("\nA large NEGATIVE delta is the expected, healthy result: it is the "
          "memorisation\ncoming out. Refit any blend weights on THIS file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
