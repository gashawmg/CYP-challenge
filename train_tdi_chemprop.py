#!/usr/bin/env python
"""ChemProp for the TDI track: learned representation + target-specific descriptors.

    python train_tdi_chemprop.py --mode BOTH --isoforms CYP3A4 CYP2D6
    python train_tdi_chemprop.py --mode BOTH --balanced --extra_blocks avalon

WHY A SEPARATE SCRIPT. The TDI pool is eighteen tree/kernel learners over the same two
feature families, and the correlation matrix in `tdi_pipeline.ipynb` Section 9c shows how little
they disagree. A message-passing network is the one member that builds its own
representation instead of consuming a fixed one, which is exactly the kind of
disagreement a rank-blend is short of.

WHAT IT FEEDS THE READOUT, AND WHY THAT CHOICE IS MEASURED RATHER THAN ASSUMED.
Per-block out-of-fold AUC on the frozen folds, one LightGBM per block:

    block       cols   CYP3A4   CYP2D6    gap
    rdkit        217   0.8045   0.5550   0.249   <- CYP3A4's best single block
    cyp           50   0.7428   0.5651   0.178
    tdialert      51   0.6788   0.5395   0.139
    ecfp        2048   0.7529   0.5967   0.156
    maccs        167   0.7545   0.5460   0.208
    fcfp        2048   0.7451   0.5944   0.151
    avalon      1024   0.7685   0.6043   0.164   <- CYP2D6's best, and best FP on both
    ALL         5605   0.8224   0.6043

Two things follow. CYP3A4 lives on physicochemical descriptors -- `rdkit` alone reaches
0.8045 against 0.8224 for everything combined. CYP2D6 does not: **Avalon alone equals all
5,605 columns**, so every other block is noise there. That is why `--extra_blocks` defaults
per isoform rather than being one global list, and why the default for CYP2D6 is Avalon
only. Feeding CYP2D6 the CYP3A4 stack is not neutral; it is 4,581 columns of distraction
for an endpoint with 1,497 rows.

THE TWO ENDPOINTS BARELY OVERLAP. Only 259 compounds carry both labels, and among those
phi is +0.160 (lift 1.39x over independence, P(2D6+|3A4+) 0.274 against 0.144). So they
share a little mechanism and are mostly separate problems -- which is why this trains one
model per isoform rather than a two-task head.

CLASS BALANCE. Training prevalence is 21.3% (CYP3A4) and 21.6% (CYP2D6) against a blind
0.2915 and 0.0706. CYP2D6 is trained three times too rich in positives, and the OOF-to-
blind d' ratio of 2.34 says local numbers understate it badly there. `--balanced` weights
positives by neg/pos so the loss sees the blind proportion instead.

OUTPUT. `oof_tdi_chemprop.csv` (Molecule_Name, y, score per isoform) and
`pred_tdi_chemprop.csv`, both in the shape `run_tdi_ensemble.py --extra-oof/--extra-test`
already accepts, so this drops into the existing pool as one more member.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation
from chemprop.nn.predictors import BinaryClassificationFFN

from rdkit import Chem, RDLogger

from cypkit import config as C, features as F, cyp_descriptors as CD, tdi_alerts as TA
from cypkit import splits as SP

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chosen from the per-block table in the docstring, NOT by symmetry.
DEFAULT_BLOCKS = {
    "CYP3A4": ["rdkit", "cyp", "tdialert", "avalon"],
    "CYP2D6": ["avalon"],
}


def build_block(name, smiles):
    if name == "rdkit":
        X, _ = F.rdkit_2d(smiles)
    elif name == "cyp":
        X, _ = CD.calculate(smiles)
    elif name == "tdialert":
        X, _ = TA.calculate(smiles)
    else:
        X, _, _ = F.fingerprint_block(smiles, name)
    return np.nan_to_num(np.asarray(X, np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def descriptors(smiles, blocks):
    """Target-specific block, robust-scaled and clipped -- same treatment as the
    regression track's TargetDescriptorCalculator, so the two are comparable."""
    X = np.hstack([build_block(b, smiles) for b in blocks])
    return X


def labels(isoform, data_dir=None):
    d = pd.read_csv(Path(data_dir or "data") / "cyp-challenge-TRAIN_TDI.csv")
    d.columns = [c.strip() for c in d.columns]
    m = d[f"{isoform}_is_TDI"].notna()
    out = pd.DataFrame({
        C.ID_COL: d.loc[m, C.ID_COL].to_numpy(),
        C.SMILES_COL: d.loc[m, C.SMILES_COL].to_numpy(),
        "y": d.loc[m, f"{isoform}_is_TDI"].astype(bool).astype(int).to_numpy()})
    return out[[Chem.MolFromSmiles(str(s)) is not None
                for s in out[C.SMILES_COL]]].reset_index(drop=True)


def attach_folds(df, folds_csv):
    frozen = pd.read_csv(folds_csv)
    f = df[C.ID_COL].map(dict(zip(frozen[C.ID_COL], frozen["fold"])))
    if f.isna().any():
        extra = df.loc[f.isna(), C.SMILES_COL].tolist()
        f.loc[f.isna()] = SP.assign_extra_to_folds(extra, frozen)
        print(f"    {len(extra)} compounds placed by nearest labelled neighbour")
    return f.to_numpy(int)


def make_model(n_desc, cfg, pos_weight=None):
    mp = BondMessagePassing(d_h=cfg["hidden"], depth=cfg["depth"],
                            dropout=cfg["dropout"], activation="relu")
    # input_dim = learned embedding ++ the target-specific descriptor block; chemprop
    # concatenates x_d onto the aggregated representation before the FFN, which is the
    # "learned representation concatenated with target-specific features" arrangement.
    task_weights = None
    ffn = BinaryClassificationFFN(input_dim=cfg["hidden"] + n_desc,
                                  hidden_dim=cfg["hidden"], n_layers=cfg["layers"],
                                  dropout=cfg["dropout"], task_weights=task_weights)
    return MPNN(message_passing=mp, agg=MeanAggregation(), predictor=ffn,
                batch_norm=True, X_d_transform=None)


def dps(smiles, y, desc, w=None):
    out = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        kw = {"x_d": desc[i].astype(np.float32)}
        if w is not None:
            kw["weight"] = float(w[i])
        out.append(MoleculeDatapoint(mol=mol, y=np.array([float(y[i])], np.float32), **kw))
    return out


def run_isoform(iso, args):
    print(f"\n{'='*70}\n{iso}")
    df = labels(iso, args.data_dir)
    df["fold"] = attach_folds(df, args.folds_csv)
    blocks = args.extra_blocks or DEFAULT_BLOCKS[iso]
    print(f"  {len(df)} labelled rows, {df.y.mean()*100:.1f}% positive")
    print(f"  descriptor blocks: {blocks}")

    D_raw = descriptors(df[C.SMILES_COL].tolist(), blocks)
    test = pd.read_csv(Path(args.data_dir or "data") / "cyp-challenge-TEST-BLINDED.csv")
    test.columns = [c.strip() for c in test.columns]
    T_raw = descriptors(test[C.SMILES_COL].tolist(), blocks)
    print(f"  descriptor block: {D_raw.shape[1]} cols")

    cfg = dict(hidden=args.hidden, depth=args.depth, layers=args.layers,
               dropout=args.dropout)
    y = df.y.to_numpy(); f = df.fold.to_numpy()
    oof = np.full(len(df), np.nan)
    test_preds = []

    for k in sorted(set(f)):
        tr, va = f != k, f == k
        # scaler fitted on the TRAINING half only -- fitting on all rows leaks the
        # held-out fold's distribution into its own features
        sc = RobustScaler().fit(D_raw[tr])
        D = np.clip(sc.transform(D_raw), -10, 10)
        T = np.clip(sc.transform(T_raw), -10, 10)
        w = None
        if args.balanced:
            pos = y[tr].mean()
            w = np.where(y == 1, (1 - pos) / max(pos, 1e-6), 1.0)
            if k == 0:
                print(f"  --balanced: positives weighted {(1-pos)/pos:.2f}x")
        tl = build_dataloader(MoleculeDataset(dps(df.loc[tr, C.SMILES_COL].tolist(),
                                                  y[tr], D[tr], None if w is None else w[tr])),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
        vl = build_dataloader(MoleculeDataset(dps(df.loc[va, C.SMILES_COL].tolist(),
                                                  y[va], D[va])),
                              batch_size=args.batch_size, shuffle=False, num_workers=0)
        model = make_model(D.shape[1], cfg)
        trainer = pl.Trainer(accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
                             max_epochs=args.epochs, logger=False, enable_progress_bar=False,
                             enable_checkpointing=False, gradient_clip_val=1.0,
                             callbacks=[EarlyStopping(monitor="val_loss",
                                                      patience=args.patience, mode="min")])
        trainer.fit(model, tl, vl)
        with torch.inference_mode():
            p = trainer.predict(model, vl)
        oof[va] = np.concatenate([x.cpu().numpy() for x in p]).ravel()[:int(va.sum())]
        tdl = build_dataloader(MoleculeDataset(dps(test[C.SMILES_COL].tolist(),
                                                   np.zeros(len(test)), T)),
                               batch_size=args.batch_size, shuffle=False, num_workers=0)
        with torch.inference_mode():
            q = trainer.predict(model, tdl)
        test_preds.append(np.concatenate([x.cpu().numpy() for x in q]).ravel()[:len(test)])
        print(f"  fold {k}: {int(va.sum())} held out, AUC {roc_auc_score(y[va], oof[va]):.4f}",
              flush=True)

    print(f"  {iso} out-of-fold AUC {roc_auc_score(y, oof):.4f}")
    return (pd.DataFrame({C.ID_COL: df[C.ID_COL], "y": y, f"{iso}_score": oof}),
            pd.DataFrame({C.ID_COL: test[C.ID_COL],
                          f"{iso}_score": np.mean(test_preds, axis=0)}))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="BOTH", choices=["OOF", "PREDICT", "BOTH"])
    p.add_argument("--isoforms", nargs="*", default=list(C.TDI_ISOFORMS))
    p.add_argument("--extra_blocks", nargs="*", default=None,
                   help="override the per-isoform default. The defaults come from the "
                        "measured per-block AUC table in the docstring: CYP3A4 gets the "
                        "descriptor stack, CYP2D6 gets Avalon ONLY because nothing else "
                        "adds to it there.")
    p.add_argument("--balanced", action="store_true",
                   help="weight positives by neg/pos. Training prevalence is ~21% on "
                        "both endpoints against a blind 0.2915 and 0.0706 -- CYP2D6 is "
                        "trained three times too rich, and this is the first lever in "
                        "the pool that addresses it.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=300)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--folds_csv", default="artifacts/folds.csv")
    p.add_argument("--data-dir", dest="data_dir", default=None)
    p.add_argument("--oof_output", default="artifacts/oof_tdi_chemprop.csv")
    p.add_argument("--output", default="artifacts/pred_tdi_chemprop.csv")
    args = p.parse_args()

    print(f"device: {DEVICE}")
    oofs, preds = [], []
    for iso in args.isoforms:
        o, t = run_isoform(iso, args)
        oofs.append(o.rename(columns={"y": f"{iso}_y"})); preds.append(t)

    if args.mode in ("OOF", "BOTH"):
        out = oofs[0]
        for d in oofs[1:]:
            out = out.merge(d, on=C.ID_COL, how="outer")
        out.to_csv(args.oof_output, index=False)
        print(f"\nOOF written: {args.oof_output}  ({len(out)} rows)")
    if args.mode in ("PREDICT", "BOTH"):
        out = preds[0]
        for d in preds[1:]:
            out = out.merge(d, on=C.ID_COL, how="outer")
        out.to_csv(args.output, index=False)
        print(f"Predictions written: {args.output}  ({len(out)} rows)")
        print("\nFeed it to the pool with:")
        print(f"  python scripts/run_tdi_ensemble.py --save-oof --predict --tag tdiCP \\")
        print(f"    --extra-oof {args.oof_output} --extra-test {args.output} "
              f"--extra-name chemprop_tdi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
