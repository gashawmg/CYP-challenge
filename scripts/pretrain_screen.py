#!/usr/bin/env python
"""Pretrain the message-passing trunk on the primary screen's 17,504 activity readings.

    python scripts/pretrain_screen.py --out models_screen_trunk
    python scripts/pretrain_screen.py --out models_screen_trunk --epochs 60

WHY. "Is this compound active at all" is the one quantity the four endpoints genuinely
share. Measured on the training curves, predicting activity on target X from the screen's
own log2fc on X reaches AUC 0.90-0.995, while the best SISTER ENDPOINT reaches only
0.43-0.87 -- and on CYP2D6 the sister is 0.432, worse than a coin flip. So a multitask
trunk trained on curve data is learning the shared part from the weakest available
source. The screen measures it directly, on 4,376 compounds x 4 isoforms = 17,504 labels,
against ~6,000 curve rows.

WHAT IT DOES. One MPNN, four regression heads, target = `log2fc_estimate` per isoform,
standardised per column. The heads are thrown away; only the trunk is kept. Downstream,
`train_cyp_st_v3.py --init_from` loads it into each per-target model before fine-tuning.

THE LEAKAGE TRAP, AND WHY THIS TRAINS FIVE TRUNKS. log2fc and pIC50 correlate at 0.83 to
0.94. A trunk that has seen compound X's screen reading, fine-tuned to predict X's pIC50,
and then evaluated on X because X is in the held-out fold, is leaking -- not through the
label itself but through a 0.9-correlated proxy of it. This is the same shape of mistake
as the UniMol scaffold split, where 78% of "out-of-fold" rows had been trained on.

So the trunk is pretrained ONCE PER FOLD, each time excluding that fold's compounds, and
fold k's downstream model is initialised only from trunk k. Five pretrains instead of
one, and out-of-fold numbers that mean what they say.

WHAT TO EXPECT. This does not conjure test-time information: no blind compound appears in
the screen, so at inference the trunk is still structure -> activity like everything else.
Its advantage is 3x the labels for the shared representation, which should show up as a
better-behaved low-activity range rather than as a better ranking of potent compounds.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from cypkit import config as C, io, splits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_cyp_mt_v1 import ADAPTIVE_CONFIG, set_global_seed  # noqa: E402

import lightning.pytorch as pl  # noqa: E402
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader  # noqa: E402
from chemprop.models import MPNN  # noqa: E402
from chemprop.nn import BondMessagePassing, MeanAggregation  # noqa: E402
from chemprop.nn.predictors import RegressionFFN  # noqa: E402
from lightning.pytorch.callbacks import EarlyStopping  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRUNK_TIER = "very_large"      # every endpoint lands here, so the dims always match


def screen_matrix(data_dir=None):
    """One row per compound, four columns of log2fc (all four are always measured)."""
    d = io.load("single_conc", data_dir)
    d.columns = [c.strip() for c in d.columns]
    g = (d.groupby([C.ID_COL, "enzyme"])["log2fc_estimate"].mean()
         .unstack("enzyme").reindex(columns=list(C.ISOFORMS)))
    smi = d.drop_duplicates(C.ID_COL).set_index(C.ID_COL)[C.SMILES_COL]
    g = g.join(smi).dropna(subset=[C.SMILES_COL])
    return g.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models_screen_trunk")
    ap.add_argument("--folds_csv", default="artifacts/folds.csv")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = ADAPTIVE_CONFIG[TRUNK_TIER]
    os.makedirs(args.out, exist_ok=True)
    g = screen_matrix(args.data_dir)
    Y = g[list(C.ISOFORMS)].to_numpy(float)
    mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0) + 1e-9
    Ys = (Y - mu) / sd
    print(f"screen matrix: {Y.shape[0]} compounds x {Y.shape[1]} isoforms "
          f"({int(np.isfinite(Y).sum())} labels)")

    folds = splits.load_folds(C.artifact("folds.csv") if args.folds_csv is None
                             else args.folds_csv)
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))
    n_folds = int(folds["fold"].max()) + 1
    extra = [i for i in g[C.ID_COL] if i not in fold_of]
    if extra:
        smi = dict(zip(g[C.ID_COL], g[C.SMILES_COL]))
        fold_of.update(dict(zip(extra, splits.assign_extra_to_folds(
            [smi[i] for i in extra], folds))))
    f = np.array([fold_of[i] for i in g[C.ID_COL]])

    mols, keep = [], []
    for k, s in enumerate(g[C.SMILES_COL]):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            mols.append(m); keep.append(k)
    keep = np.array(keep)
    print(f"  {len(keep)} parsed; pretraining {n_folds} trunks, one per held-out fold\n")

    for fi in range(n_folds):
        path = os.path.join(args.out, f"screen_trunk_fold{fi}.pt")
        if os.path.exists(path):
            print(f"fold {fi}: exists, skipping"); continue
        set_global_seed(args.seed + fi)
        tr = keep[f[keep] != fi]
        va = keep[f[keep] == fi]
        # the held-out fold is used ONLY for early stopping, never for gradient
        def dps(idx):
            return [MoleculeDatapoint(mol=mols[list(keep).index(i)],
                                      y=Ys[i].astype(np.float32)) for i in idx]
        tl = build_dataloader(MoleculeDataset(dps(tr)), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
        vl = build_dataloader(MoleculeDataset(dps(va)), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
        mp = BondMessagePassing(d_h=cfg["hidden_dim"], depth=cfg["depth"],
                                dropout=cfg["dropout"], activation="relu")
        ffn = RegressionFFN(input_dim=cfg["hidden_dim"], hidden_dim=cfg["hidden_dim"],
                            n_layers=cfg["n_layers"], dropout=cfg["dropout"],
                            n_tasks=len(C.ISOFORMS))
        model = MPNN(message_passing=mp, agg=MeanAggregation(), predictor=ffn,
                     batch_norm=True)
        trainer = pl.Trainer(accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
                             max_epochs=args.epochs, logger=False,
                             enable_progress_bar=True, gradient_clip_val=1.0,
                             callbacks=[EarlyStopping(monitor="val_loss",
                                                      patience=args.patience, mode="min")])
        print(f"fold {fi}: train {len(tr)}  val {len(va)}")
        trainer.fit(model, tl, vl)
        torch.save({"message_passing": model.message_passing.state_dict(),
                    "hidden_dim": cfg["hidden_dim"], "depth": cfg["depth"],
                    "isoforms": list(C.ISOFORMS)}, path)
        print(f"  saved {path}")

    print(f"\nNow fine-tune with:  --init_from {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
