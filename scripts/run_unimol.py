#!/usr/bin/env python
"""Track B-2 — Uni-Mol, the 3D pretrained transformer, on our frozen folds.

    # once, on a machine with normal internet:
    python scripts/run_unimol.py --fetch-weights

    python scripts/run_unimol.py --smoke
    python scripts/run_unimol.py --epochs 40 --predict

Uni-Mol is the one member of the pool that sees the molecule in three dimensions. Every
other model here — boosters, kernels, even the D-MPNN — reasons about a 2D graph or a
vector derived from one. CYP inhibition is a shape-and-orientation problem: whether a
nitrogen reaches the heme iron depends on geometry, and a planar fingerprint cannot say.
That is the specific thing this model is in the ensemble to contribute.

WHAT IT DOES NOT DO: Uni-Mol's trainer exposes MAE/MSE only, so there is no dead-zone
objective here. Censored rows are folded in the crude way instead — a primary-screen
non-hit enters as a measurement AT the pIC50 floor. That is the `censored_imputed`
variant from the phase-1 ladder, which was competitive; the dead zone is the boosters'
job, and the blend decides how much geometry is worth.

WEIGHTS. `unimol_tools` pulls its checkpoint from Hugging Face, which is blocked from
this sandbox (403 at the proxy) — both from the container and from the desktop VM. Run
`--fetch-weights` on your GPU box, which has ordinary internet, or download
    https://huggingface.co/dptech/Uni-Mol-Models  ->  mol_pre_no_h_220816.pt
by hand and point `--weights` at it. Nothing else in the plan depends on this file, so
if it stays unavailable the ensemble simply runs without a 3D member.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit import io, labels as L, splits  # noqa: E402

TARGET_COLS = [f"{c}_pIC50_direct_inhibition" for c in C.ISOFORMS]


def fetch_weights(dest: Path) -> int:
    """Download the pretrained checkpoint. Needs ordinary internet."""
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    for endpoint in (None, "https://hf-mirror.com"):
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
        try:
            p = snapshot_download(
                "dptech/Uni-Mol-Models", local_dir=str(dest),
                allow_patterns=["mol_pre_no_h_220816.pt", "mol_pre_all_h_220816.pt",
                                "*.txt", "*.json"],
            )
            print(f"weights in {p}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"  {endpoint or 'huggingface.co'}: {type(exc).__name__}: {exc}")
    print("\nCould not reach Hugging Face from here. Download mol_pre_no_h_220816.pt from"
          "\n  https://huggingface.co/dptech/Uni-Mol-Models"
          "\nand pass --weights /path/to/mol_pre_no_h_220816.pt")
    return 1


def build_table(args) -> tuple[pd.DataFrame, dict, int, pd.DataFrame, pd.DataFrame]:
    """One row per compound, four target columns, censored rows imputed at the floor."""
    train = io.load("train_inhibition", args.data_dir)
    screen = io.load("single_conc", args.data_dir)
    lm = L.build_label_matrix(train, screen, include_censored=True,
                              censored_weight=args.censored_weight)

    folds = splits.load_folds(C.artifact("folds.csv"))
    n_folds = int(folds["fold"].max()) + 1
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))
    extra = sorted(set(lm[C.ID_COL]) - set(fold_of))
    if extra:
        smi = dict(zip(lm[C.ID_COL], lm[C.SMILES_COL]))
        fold_of.update(dict(zip(extra, splits.assign_extra_to_folds(
            [smi[i] for i in extra], folds))))

    smiles_by_id = dict(zip(lm[C.ID_COL], lm[C.SMILES_COL]))
    smiles_by_id.update(dict(zip(train[C.ID_COL], train[C.SMILES_COL])))
    smiles_by_id = {k: v for k, v in smiles_by_id.items() if isinstance(v, str)}
    ids = sorted(smiles_by_id)
    pos = {m: i for i, m in enumerate(ids)}

    Y = np.full((len(ids), len(C.ISOFORMS)), np.nan)
    for r in lm.itertuples(index=False):
        i = pos.get(getattr(r, C.ID_COL))
        if i is None:
            continue
        j = C.ISOFORMS.index(r.isoform)
        y = float(r.y) if np.isfinite(r.y) else float(r.y_high)
        Y[i, j] = y if np.isfinite(y) else C.PIC50_FLOOR

    tab = pd.DataFrame({C.ID_COL: ids, "SMILES": [smiles_by_id[i] for i in ids]})
    for j, col in enumerate(TARGET_COLS):
        tab[col] = Y[:, j]
    tab["fold"] = [fold_of.get(i, -1) for i in ids]
    return tab, fold_of, n_folds, train, lm


def _trainer(args, save_path, epochs):
    from unimol_tools import MolTrain  # noqa: PLC0415

    kw = dict(
        task="multilabel_regression", data_type="molecule", epochs=epochs,
        learning_rate=args.lr, batch_size=args.batch_size, early_stopping=args.patience,
        metrics="mae", split="random", kfold=1, save_path=str(save_path),
        smiles_col="SMILES", target_cols=TARGET_COLS, target_normalize="auto",
        use_cuda=not args.cpu, use_amp=not args.cpu, model_name=args.model_name,
        model_size=args.model_size, remove_hs=True,
    )
    if args.weights:
        kw["pretrained_model_path"] = str(args.weights)
    return MolTrain(**kw)


def _fit_predict(args, train_df, test_df, tag, epochs):
    from unimol_tools import MolPredict  # noqa: PLC0415

    save_path = Path(args.workdir) / tag
    clf = _trainer(args, save_path, epochs)
    clf.fit(train_df[["SMILES", *TARGET_COLS]].copy())
    return np.asarray(MolPredict(load_model=str(save_path)).predict(
        test_df[["SMILES"]].copy()), dtype=float).reshape(len(test_df), -1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch-weights", action="store_true")
    ap.add_argument("--weights", type=Path, default=None,
                    help="path to mol_pre_no_h_220816.pt if not in the default cache")
    ap.add_argument("--weights-dir", type=Path, default=Path("weights/unimol"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--model-name", default="unimolv1")
    ap.add_argument("--model-size", default="84m")
    ap.add_argument("--censored-weight", type=float, default=0.5)
    ap.add_argument("--workdir", default="artifacts/unimol")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, fold 0, 400 rows")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--tag", default="um")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.fetch_weights:
        return fetch_weights(args.weights_dir)

    tab, fold_of, n_folds, train, lm = build_table(args)
    print(f"{len(tab)} compounds; labelled per isoform: "
          + ", ".join(f"{c} {int(tab[col].notna().sum())}"
                      for c, col in zip(C.ISOFORMS, TARGET_COLS)))

    epochs = 2 if args.smoke else args.epochs
    folds_to_run = [0] if args.smoke else range(n_folds)
    P = np.full((len(tab), len(C.ISOFORMS)), np.nan)

    for f in folds_to_run:
        tr = tab[(tab.fold != f) & (tab.fold >= 0)]
        te = tab[tab.fold == f]
        tr = tr[tr[TARGET_COLS].notna().any(axis=1)]
        if args.smoke:
            tr, te = tr.head(400), te.head(100)
        if len(tr) < 50 or len(te) == 0:
            continue
        t0 = time.time()
        P[te.index.to_numpy()] = _fit_predict(args, tr, te, f"fold{f}", epochs)
        print(f"  fold {f}: train {len(tr)} / predict {len(te)}  ({time.time()-t0:.0f}s)")

    oof = pd.DataFrame({C.ID_COL: tab[C.ID_COL]})
    for j, col in enumerate(TARGET_COLS):
        oof[col] = np.where(np.isfinite(P[:, j]), P[:, j], C.PIC50_FLOOR)
    path = C.artifact(f"oof_unimol_{args.tag}.csv")
    oof.to_csv(path, index=False)
    print(f"\nwritten {path}")

    if not args.smoke:
        from cypkit import scoring  # noqa: PLC0415
        print(scoring.format_report(scoring.oof_report(oof, train, n_bootstrap=100)))
        dp = scoring.dense_proxy_report(oof, lm)
        print(dp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        dma = float(dp[dp["Endpoint"] == "MA"]["dense_STRAE"].iloc[0])
        print(f"\n  recalibrated leaderboard estimate = {0.776 + 0.212*dma:.4f}  "
              f"(current best 0.9171)")

    if args.predict:
        test = io.load("test", args.data_dir)
        tr = tab[tab[TARGET_COLS].notna().any(axis=1)]
        pred = _fit_predict(args, tr, test.rename(columns={C.SMILES_COL: "SMILES"}),
                            "full", epochs)
        out = test[[C.SMILES_COL, C.ID_COL]].copy()
        for j, col in enumerate(TARGET_COLS):
            out[col] = pred[:, j]
        p = C.artifact(f"submission_unimol_{args.tag}.csv")
        out.to_csv(p, index=False)
        print(f"\nwritten {p}")
        print(out[TARGET_COLS].describe().T.to_string(float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
