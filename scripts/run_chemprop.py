#!/usr/bin/env python
"""Track B-1 — ChemProp D-MPNN, multi-task, censored, with descriptor fusion.

    python scripts/run_chemprop.py --smoke                     # 2 epochs, fold 0 only
    python scripts/run_chemprop.py --epochs 60 --accelerator gpu
    python scripts/run_chemprop.py --epochs 60 --accelerator gpu --predict

Why this is a different model and not another flavour of the same one: every model in
the project so far consumes a fixed 2,482-column vector that we chose. A D-MPNN learns
its own representation from the molecular graph. When it disagrees with the boosters it
disagrees for structural reasons, which is the only kind of disagreement a blend can
profit from.

Three things make this a real port of our loss rather than a stock ChemProp run:

1.  THE DEAD ZONE, EXACTLY. ChemProp's task vector is widened to 8 columns, two per
    isoform: (y_low, y_high). The custom `DeadZone` criterion reads them as a band and
    charges |pred - nearest edge| outside it, zero inside — the same objective the
    LightGBM models use, and the shape of the leaderboard metric. Odd columns carry no
    prediction; they exist so the band survives ChemProp's target plumbing intact.

2.  THE CENSORED ROWS. A primary-screen non-hit enters as (-1e4, floor]: free below the
    floor, charged above it. That is the correction that fixed our 0.9-log-unit bias on
    inactives, and it is the reason this is trained on ~15k rows rather than ~6.5k.

3.  DESCRIPTOR FUSION. The 217 RDKit + 50 CYP-specific descriptors are standardised and
    concatenated to the learned graph embedding before the readout FFN, so the network
    keeps the CYP pharmacophore signal (heme coordination, 2D6 basic amine, 2C9 acid)
    it would otherwise have to rediscover from scratch on 5k molecules.

Targets are deliberately NOT scaled — the band edges are in pIC50 units and must stay
there for the dead zone to mean anything.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit import cyp_descriptors as CD  # noqa: E402
from cypkit import features, io, labels as L, splits  # noqa: E402

LARGE = 1.0e4


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------


def make_deadzone_criterion(delta: float = 0.25):
    import torch  # noqa: PLC0415
    from chemprop.nn.metrics import ChempropMetric  # noqa: PLC0415

    class DeadZone(ChempropMetric):
        """Huberised dead-zone L1 over (lo, hi) target pairs.

        preds/targets arrive as (B, 2K). Column 2j of `preds` is isoform j's
        prediction; columns 2j and 2j+1 of `targets` are its band edges. The returned
        tensor is (B, 2K) so ChempropMetric's own masking and weighting still apply —
        the loss is written into the even column and the odd column is zero.
        """

        def _calc_unreduced_loss(self, preds, targets, *args):
            p = preds[:, 0::2]
            lo = targets[:, 0::2]
            hi = targets[:, 1::2]
            r = torch.clamp(p - hi, min=0.0) + torch.clamp(lo - p, min=0.0)
            per = torch.where(r <= delta, 0.5 * r.pow(2) / delta, r - 0.5 * delta)
            out = torch.zeros_like(preds)
            out[:, 0::2] = per
            return out

    return DeadZone()


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def build_frames(args):
    train = io.load("train_inhibition", args.data_dir)
    screen = io.load("single_conc", args.data_dir)
    test = io.load("test", args.data_dir)
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

    k = len(C.ISOFORMS)
    Y = np.full((len(ids), 2 * k), np.nan, dtype=np.float32)
    W = np.ones(len(ids), dtype=np.float32)
    pos = {m: i for i, m in enumerate(ids)}
    iso_idx = {c: j for j, c in enumerate(C.ISOFORMS)}
    for r in lm.itertuples(index=False):
        i = pos.get(getattr(r, C.ID_COL))
        if i is None:
            continue
        j = iso_idx[r.isoform]
        lo, hi = float(r.y_low), float(r.y_high)
        Y[i, 2 * j] = -LARGE if not np.isfinite(lo) else lo
        Y[i, 2 * j + 1] = LARGE if not np.isfinite(hi) else hi

    return dict(train=train, test=test, lm=lm, ids=ids,
                smiles=[smiles_by_id[i] for i in ids], Y=Y, W=W,
                fold_of=fold_of, n_folds=n_folds)


def descriptors(smiles: list[str]) -> np.ndarray:
    Xr, _ = features.rdkit_2d(smiles)
    Xc, _ = CD.calculate(smiles)
    X = np.hstack([Xr, Xc]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return np.clip((X - mu) / sd, -8, 8).astype(np.float32)


def datapoints(smiles, Y, Xd, W, idx):
    from chemprop.data import MoleculeDatapoint  # noqa: PLC0415
    from rdkit import Chem  # noqa: PLC0415

    out = []
    for i in idx:
        mol = Chem.MolFromSmiles(smiles[i])
        if mol is None:
            continue
        out.append(MoleculeDatapoint(mol=mol, y=Y[i], weight=float(W[i]), x_d=Xd[i]))
    return out


def make_model(n_desc: int, args):
    from chemprop import models, nn as cnn  # noqa: PLC0415

    mp = cnn.BondMessagePassing(d_h=args.hidden, depth=args.depth, dropout=args.dropout)
    agg = cnn.MeanAggregation()
    ffn = cnn.RegressionFFN(
        n_tasks=2 * len(C.ISOFORMS),
        input_dim=mp.output_dim + n_desc,
        hidden_dim=args.ffn_hidden,
        n_layers=args.ffn_layers,
        dropout=args.dropout,
        criterion=make_deadzone_criterion(args.delta),
        output_transform=None,   # bands are in pIC50 units; never scale them
    )
    return models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[make_deadzone_criterion(args.delta)],
                       max_lr=args.lr)


def train_one(tr_idx, te_idx, D, Xd, args):
    import lightning.pytorch as pl  # noqa: PLC0415
    from chemprop.data import MoleculeDataset, build_dataloader  # noqa: PLC0415

    tr = MoleculeDataset(datapoints(D["smiles"], D["Y"], Xd, D["W"], tr_idx))
    te = MoleculeDataset(datapoints(D["smiles"], D["Y"], Xd, D["W"], te_idx))
    dl_tr = build_dataloader(tr, batch_size=args.batch_size, num_workers=args.workers,
                             shuffle=True)
    dl_te = build_dataloader(te, batch_size=args.batch_size, num_workers=args.workers,
                             shuffle=False)

    model = make_model(Xd.shape[1], args)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator=args.accelerator,
                         devices=1, logger=False, enable_checkpointing=False,
                         enable_progress_bar=args.progress,
                         enable_model_summary=False)
    trainer.fit(model, dl_tr)
    P = np.concatenate([b.numpy() for b in trainer.predict(model, dl_te)], axis=0)
    return P.reshape(len(te_idx), -1)[:, 0::2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=400)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--ffn-hidden", type=int, default=512)
    ap.add_argument("--ffn-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--censored-weight", type=float, default=0.5)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1, help="model seeds per fold, averaged")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, fold 0 only")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--tag", default="cp")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 2

    import torch  # noqa: PLC0415
    torch.set_float32_matmul_precision("medium")

    D = build_frames(args)
    print(f"{len(D['ids'])} compounds, {int(np.isfinite(D['Y'][:, 0::2]).sum())} "
          f"labelled (compound, isoform) pairs")

    t0 = time.time()
    Xd = descriptors(D["smiles"])
    print(f"descriptors {Xd.shape}  ({time.time()-t0:.1f}s)")

    fold_arr = np.array([D["fold_of"].get(i, -1) for i in D["ids"]])
    oof = pd.DataFrame({C.ID_COL: D["ids"]})
    P = np.full((len(D["ids"]), len(C.ISOFORMS)), np.nan, dtype=np.float32)

    folds_to_run = [0] if args.smoke else range(D["n_folds"])
    for f in folds_to_run:
        te_idx = np.where(fold_arr == f)[0]
        tr_idx = np.where((fold_arr != f) & (fold_arr >= 0))[0]
        # only train on rows that carry at least one band
        tr_idx = tr_idx[np.isfinite(D["Y"][tr_idx, 0::2]).any(1)]
        if len(te_idx) == 0 or len(tr_idx) < 50:
            continue
        t0 = time.time()
        acc = np.zeros((len(te_idx), len(C.ISOFORMS)), dtype=np.float64)
        for s in range(args.seeds):
            import lightning.pytorch as pl  # noqa: PLC0415
            pl.seed_everything(1000 + s, workers=True)
            acc += train_one(tr_idx, te_idx, D, Xd, args)
        P[te_idx] = (acc / args.seeds).astype(np.float32)
        print(f"  fold {f}: train {len(tr_idx)} / predict {len(te_idx)}  "
              f"({time.time()-t0:.0f}s)")

    for j, cyp in enumerate(C.ISOFORMS):
        col = P[:, j]
        oof[f"{cyp}_pIC50_direct_inhibition"] = np.where(
            np.isfinite(col), col, C.PIC50_FLOOR)
    path = C.artifact(f"oof_chemprop_{args.tag}.csv")
    oof.to_csv(path, index=False)
    print(f"\nwritten {path}")

    if not args.smoke:
        _score(oof, D)

    if args.predict:
        _predict(D, Xd, args)
    return 0


def _score(oof, D):
    from cypkit import scoring  # noqa: PLC0415

    lb = scoring.oof_report(oof, D["train"], n_bootstrap=100)
    print(scoring.format_report(lb))
    dp = scoring.dense_proxy_report(oof, D["lm"])
    print(dp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    dma = float(dp[dp["Endpoint"] == "MA"]["dense_STRAE"].iloc[0])
    print(f"\n  recalibrated leaderboard estimate = {0.776 + 0.212*dma:.4f}  "
          f"(current best 0.9171)")


def _predict(D, Xd, args):
    """Refit on everything and score the blind set."""
    import lightning.pytorch as pl  # noqa: PLC0415
    from chemprop.data import MoleculeDataset, build_dataloader  # noqa: PLC0415
    from rdkit import Chem  # noqa: PLC0415
    from chemprop.data import MoleculeDatapoint  # noqa: PLC0415

    test = D["test"]
    test_smiles = test[C.SMILES_COL].tolist()
    Xd_test = descriptors(D["smiles"] + test_smiles)[len(D["smiles"]):]

    tr_idx = np.where(np.isfinite(D["Y"][:, 0::2]).any(1))[0]
    tr = MoleculeDataset(datapoints(D["smiles"], D["Y"], Xd, D["W"], tr_idx))
    dl_tr = build_dataloader(tr, batch_size=args.batch_size, num_workers=args.workers,
                             shuffle=True)
    te_dps = [MoleculeDatapoint(mol=Chem.MolFromSmiles(s), x_d=Xd_test[i])
              for i, s in enumerate(test_smiles)]
    dl_te = build_dataloader(MoleculeDataset(te_dps), batch_size=args.batch_size,
                             num_workers=args.workers, shuffle=False)

    acc = np.zeros((len(test), len(C.ISOFORMS)))
    for s in range(max(args.seeds, 1)):
        pl.seed_everything(2000 + s, workers=True)
        model = make_model(Xd.shape[1], args)
        trainer = pl.Trainer(max_epochs=args.epochs, accelerator=args.accelerator,
                             devices=1, logger=False, enable_checkpointing=False,
                             enable_progress_bar=args.progress, enable_model_summary=False)
        trainer.fit(model, dl_tr)
        pred = np.concatenate([b.numpy() for b in trainer.predict(model, dl_te)], axis=0)
        acc += pred.reshape(len(test), -1)[:, 0::2]
    acc /= max(args.seeds, 1)

    out = test[[C.SMILES_COL, C.ID_COL]].copy()
    for j, cyp in enumerate(C.ISOFORMS):
        out[f"{cyp}_pIC50_direct_inhibition"] = acc[:, j]
    path = C.artifact(f"submission_chemprop_{args.tag}.csv")
    out.to_csv(path, index=False)
    print(f"\nwritten {path}")
    print(out[list(C.REGRESSION_ENDPOINTS)].describe().T.to_string(
        float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    raise SystemExit(main())
