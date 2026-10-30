"""
CYP pIC50 — Multitask Chemprop with EXTERNAL auxiliary heads (v3)
==================================================================
The leaderboard analysis said the gap is data, not architecture: the top explains 62%
of blind-set variance (Spearman 0.76) while we manage 15% (0.65), and calibration is
not the lever — a global expansion sweep on the composite bottoms out at scale 1.2 and
degrades from there. Ranking is what separates us, and ranking needs more molecules.

This trains ONE network on both label sets at once:

    4 regression heads   CYP1A2/2C9/2D6/3A4 pIC50        the scored targets, ~4.9k rows
    5 auxiliary heads    CYP1A2/2C9/2C19/2D6/3A4 binary  Veith/NCGC, ~12-17k compounds

The auxiliary labels are inhibitor / non-inhibitor calls, so they cannot be regression
targets — the loss is MIXED: squared error on the first four columns, binary
cross-entropy with logits on the last five, with the auxiliary block downweighted by
`--aux-weight`. Both are masked on NaN, so every compound contributes to exactly the
heads it has labels for and nothing is imputed.

WHY THIS SHOULD ALSO FIX THE INACTIVE BIAS. The neural baselines were badly calibrated
on inactives — UniMol predicted mean 4.36 on 8,424 known-inactive rows with only 24%
below the floor — because they only ever saw compounds that had a dose-response curve,
i.e. screening hits. The Veith panel is a full qHTS screen and is mostly NON-inhibitors.
For the first time the network sees a large, unbiased sample of what inactive looks
like.

FOLD SAFETY, which is the part that is easy to get wrong. External compounds are
training-only — they are never validated on, because they are not the task. But an
external compound that is an analogue of a validation compound would leak. So folds are
assigned over the UNION of challenge and external compounds by Murcko scaffold: an
external molecule sharing a scaffold with a held-out challenge molecule lands in that
same fold and is excluded from training for it. Validation sets contain challenge
compounds only, so the reported ST-RAE stays comparable with every earlier run.

Prerequisite
------------
  python scripts/fetch_external_cyp.py        # writes artifacts/external_cyp.csv
                                              # (removes blind-test compounds by InChIKey)

Usage
-----
  python train_cyp_mt_v3.py --mode BOTH
  python train_cyp_mt_v3.py --mode BOTH --aux-weight 0.3      # weaker auxiliary pull
  python train_cyp_mt_v3.py --mode BOTH --external ""         # ablation: no external
"""

import argparse, glob, os, time, warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.agg import MeanAggregation
from chemprop.nn.predictors import RegressionFFN
from chemprop.models import MPNN

from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import RobustScaler

from train_cyp_mt_v1 import (
    ISOFORMS, TARGET_COLS, SMILES_COL, NAME_COL, CONF_LOW, CONF_HIGH,
    MoleculeStandardizer, SMILESAugmenter, get_adaptive_config, split_activity_scaffold,
    load_training, set_global_seed, masked_report, save_pickle, load_pickle,
)

rdBase.DisableLog("rdApp.error"); rdBase.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore")

TRAIN_CSV = "data/cyp-challenge-TRAIN_inhibition.csv"
TEST_CSV = "data/cyp-challenge-TEST-BLINDED.csv"
EXTERNAL_CSV = "artifacts/external_cyp.csv"
AUX_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
N_REG = len(TARGET_COLS)
GLOBAL_SEED = 42
N_FOLDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_mixed_criterion(n_reg: int, aux_weight: float):
    """MSE on the regression block, BCE-with-logits on the auxiliary block."""
    import torch.nn.functional as F  # noqa: PLC0415
    from chemprop.nn.metrics import ChempropMetric  # noqa: PLC0415

    class MixedRegClf(ChempropMetric):
        # chemprop's logger reads .alias off every metric; a bare subclass
        # raises AttributeError inside the validation loop without it.
        alias = "mixedregclf"
        def _calc_unreduced_loss(self, preds, targets, mask, weights, lt_mask, gt_mask):
            out = torch.zeros_like(preds)
            out[:, :n_reg] = (preds[:, :n_reg] - targets[:, :n_reg]) ** 2
            if preds.shape[1] > n_reg:
                # targets are 0/1; nan_to_num upstream already replaced NaN with 0,
                # and `mask` zeroes those rows out, so the value is irrelevant there.
                out[:, n_reg:] = aux_weight * F.binary_cross_entropy_with_logits(
                    preds[:, n_reg:], targets[:, n_reg:].clamp(0, 1), reduction="none")
            return out

    return MixedRegClf()


def build_model(cfg, n_task, aux_weight):
    """Construct the architecture. Used for training AND for reloading weights.

    `MPNN.load_from_checkpoint` rebuilds the model from pickled hparams, but Lightning
    refuses to pickle a custom criterion ("Attribute 'metrics' removed from hparams")
    and the reload then dies with KeyError: 'metrics'. Building the architecture here
    and loading only the state_dict sidesteps that entirely, and is explicit about
    what is being restored.
    """
    mp = BondMessagePassing(d_h=cfg["hidden_dim"], depth=cfg["depth"],
                            dropout=cfg["dropout"], activation="relu")
    ffn = RegressionFFN(n_tasks=n_task, input_dim=cfg["hidden_dim"],
                        hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"],
                        dropout=cfg["dropout"],
                        criterion=make_mixed_criterion(N_REG, aux_weight),
                        output_transform=None)
    return MPNN(mp, MeanAggregation(), ffn, batch_norm=True,
                metrics=[make_mixed_criterion(N_REG, aux_weight)])


def load_weights(path, cfg, n_task, aux_weight):
    model = build_model(cfg, n_task, aux_weight)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    return model


def murcko(smi):
    mol = Chem.MolFromSmiles(smi)
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
    except Exception:
        return smi


def build_table(args, std):
    """Challenge rows + external rows, one frame, NaN where a label is absent."""
    ch = load_training(args.train_csv, std)
    n_task = N_REG + len(AUX_ISOFORMS)
    rows_smi = ch["SMILES_clean"].tolist()
    rows_name = ch[NAME_COL].tolist()
    Y = np.full((len(ch), n_task), np.nan, dtype=np.float32)
    Y[:, :N_REG] = ch[TARGET_COLS].to_numpy(float)
    is_ext = np.zeros(len(ch), bool)
    ch_key = {}

    if args.external and os.path.exists(args.external):
        ex = pd.read_csv(args.external)
        smi_col = "smiles_std" if "smiles_std" in ex.columns else "SMILES"
        have = [i for i in AUX_ISOFORMS if i in ex.columns]
        print(f"external: {len(ex)} compounds, auxiliary labels for {have}")

        # attach auxiliary labels to challenge compounds where they overlap
        from scripts.fetch_external_cyp import standardise  # noqa: PLC0415
        ch_key = {}
        for i, s in enumerate(rows_smi):
            _, k = standardise(s)
            if k:
                ch_key.setdefault(k, i)
        ex_keys = ex["inchikey"] if "inchikey" in ex.columns else [
            standardise(s)[1] for s in ex[smi_col]]
        matched, new_smi, new_Y = 0, [], []
        for (_, r), k in zip(ex.iterrows(), ex_keys):
            vals = [float(r[i]) if i in have and pd.notna(r[i]) else np.nan
                    for i in AUX_ISOFORMS]
            if k in ch_key:
                Y[ch_key[k], N_REG:] = vals
                matched += 1
            else:
                new_smi.append(r[smi_col])
                new_Y.append([np.nan] * N_REG + vals)
        print(f"  {matched} matched a challenge compound (auxiliary labels attached)")
        print(f"  {len(new_smi)} are NEW molecules -> {len(rows_smi)+len(new_smi)} total")
        if new_smi:
            rows_smi += list(new_smi)
            rows_name += [f"EXT-{i:06d}" for i in range(len(new_smi))]
            Y = np.vstack([Y, np.asarray(new_Y, np.float32)])
            is_ext = np.concatenate([is_ext, np.ones(len(new_smi), bool)])
    else:
        if args.external:
            print(f"NOTE: {args.external} not found - training WITHOUT external data.")

    return rows_smi, rows_name, Y, is_ext, ch


def union_folds(smiles, is_ext, Y, n_folds, seed):
    """Folds over challenge+external by scaffold; validation is challenge-only.

    An external analogue of a held-out challenge compound must not be trainable for
    that fold, so both are placed in the same fold by shared Murcko scaffold.
    """
    folds = split_activity_scaffold(smiles, Y[:, :N_REG], n_folds=n_folds, seed=seed)
    fold_of = np.empty(len(smiles), int)
    for i, f in enumerate(folds):
        for j in f:
            fold_of[j] = i
    scaf = np.array([murcko(s) for s in smiles])
    leaked = 0
    for sc in pd.unique(scaf):
        idx = np.where(scaf == sc)[0]
        if len(set(fold_of[idx])) > 1:
            leaked += len(idx)
    print(f"  fold sizes: {[len(f) for f in folds]}")
    print(f"  scaffold leakage across folds: {leaked/len(smiles)*100:.1f}%")
    print(f"  external compounds are training-only "
          f"({int(is_ext.sum())} of {len(smiles)} rows never validated)")
    return fold_of


def main():
    p = argparse.ArgumentParser(description="CYP multitask + external auxiliary heads")
    p.add_argument("--mode", choices=["TRAIN", "PREDICT", "BOTH"], default="BOTH")
    p.add_argument("--train_csv", default=TRAIN_CSV)
    p.add_argument("--test_csv", default=TEST_CSV)
    p.add_argument("--external", default=EXTERNAL_CSV)
    p.add_argument("--aux-weight", type=float, default=0.5,
                   help="weight on the auxiliary BCE block relative to the regression MSE")
    p.add_argument("--model_dir", default="models_cyp_mt_v3")
    p.add_argument("--output", default="predictions_cyp_mt_v3.csv")
    p.add_argument("--oof_output", default="oof_chemprop_mt_v3.csv")
    p.add_argument("--n_folds", type=int, default=N_FOLDS)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = p.parse_args()

    set_global_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)
    print(f"Device: {DEVICE}   aux weight: {args.aux_weight}")

    t0 = time.time()
    std = MoleculeStandardizer()
    smiles, names, Y, is_ext, ch = build_table(args, std)
    n_task = Y.shape[1]
    print(f"\nlabelled cells per head:")
    for j in range(n_task):
        nm = TARGET_COLS[j] if j < N_REG else f"AUX_{AUX_ISOFORMS[j-N_REG]}"
        print(f"  {nm:38s} {int(np.isfinite(Y[:, j]).sum()):6d}")

    # scale only the regression block; auxiliary stays 0/1
    Ys = Y.copy()
    scalers = []
    for j in range(N_REG):
        col = Y[:, j]; ok = np.isfinite(col)
        sc = RobustScaler(quantile_range=(1.0, 99.0))
        sc.fit(col[ok].reshape(-1, 1))
        Ys[ok, j] = sc.transform(col[ok].reshape(-1, 1)).ravel().astype(np.float32)
        scalers.append(sc)

    cfg = get_adaptive_config(len(smiles))
    if args.patience is not None:
        cfg["patience"] = args.patience
    print(f"\ntier -> hidden={cfg['hidden_dim']} depth={cfg['depth']} "
          f"layers={cfg['n_layers']} aug={cfg['augmentation_factor']}x "
          f"patience={cfg['patience']}")

    fold_of = union_folds(smiles, is_ext, Y, args.n_folds, args.seed)
    aug = SMILESAugmenter(random_seed=args.seed)

    def dps(smi, Yv):
        out = []
        for i, s in enumerate(smi):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                out.append(MoleculeDatapoint(mol=mol, y=Yv[i].astype(np.float32)))
        return out

    oof = np.full((len(smiles), N_REG), np.nan)
    best_epochs = []
    if args.mode in ("TRAIN", "BOTH"):
        for f in range(args.n_folds):
            val = np.where((fold_of == f) & ~is_ext)[0]
            tr = np.where(fold_of != f)[0]
            tr = tr[np.isfinite(Ys[tr]).any(1)]
            if len(val) == 0 or len(tr) < 50:
                continue
            set_global_seed(args.seed + f)
            print(f"\n--- fold {f+1}/{args.n_folds}: train {len(tr)} "
                  f"({int(is_ext[tr].sum())} external) / validate {len(val)} ---")

            tr_smi = [smiles[i] for i in tr]
            if cfg["use_augmentation"] and cfg["augmentation_factor"] > 1:
                a_smi, a_Y, _ = aug.augment_dataset(tr_smi, Ys[tr], cfg["augmentation_factor"])
            else:
                a_smi, a_Y = tr_smi, Ys[tr]
            tl = build_dataloader(MoleculeDataset(dps(a_smi, a_Y)),
                                  batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
            vl = build_dataloader(MoleculeDataset(dps([smiles[i] for i in val], Ys[val])),
                                  batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

            ck = os.path.join(args.model_dir, f"fold_{f}.ckpt")
            if os.path.exists(ck):
                print("  checkpoint found, skipping training")
                model = load_weights(ck, cfg, n_task, args.aux_weight)
                best_epochs.append(-1)
            else:
                model = build_model(cfg, n_task, args.aux_weight)
                cb = ModelCheckpoint(dirpath=args.model_dir, filename=f"fold_{f}",
                                     monitor="val_loss", mode="min", save_top_k=1)
                tr_ = pl.Trainer(accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
                                 max_epochs=cfg["max_epochs"],
                                 callbacks=[EarlyStopping(monitor="val_loss",
                                                          patience=cfg["patience"], mode="min"), cb],
                                 enable_progress_bar=True, logger=False, gradient_clip_val=1.0)
                tr_.fit(model, tl, vl)
                best_epochs.append(max(0, tr_.current_epoch - cfg["patience"]))
                if cb.best_model_path and os.path.exists(cb.best_model_path):
                    model = load_weights(cb.best_model_path, cfg, n_task, args.aux_weight)
            model.eval(); model.to(torch.device(DEVICE))
            pt = pl.Trainer(accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
                            logger=False, enable_progress_bar=False)
            with torch.inference_mode():
                raw = pt.predict(model, vl)
            P = np.concatenate([b.cpu().numpy() for b in raw], axis=0).reshape(-1, n_task)
            keep = [i for i, s in enumerate([smiles[i] for i in val]) if Chem.MolFromSmiles(s)]
            for k, li in enumerate(keep):
                oof[val[li]] = P[k, :N_REG]

        for j in range(N_REG):
            ok = np.isfinite(oof[:, j])
            oof[ok, j] = scalers[j].inverse_transform(oof[ok, j].reshape(-1, 1)).ravel()

        ch_rows = ~is_ext
        bands = []
        for ep in TARGET_COLS:
            lo, hi = ep + CONF_LOW, ep + CONF_HIGH
            bands.append((ch[lo].to_numpy(float), ch[hi].to_numpy(float))
                         if lo in ch.columns else (None, None))
        rep = masked_report(Y[ch_rows][:, :N_REG], oof[ch_rows],
                            bands=bands if bands[0][0] is not None else None)
        print("\n" + rep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nreference OOF ST-RAE   multitask v1 (no external): "
              "1A2 0.848  2C9 0.663  2D6 0.946  3A4 0.558  MA 0.754")
        print("                       classical ensemble:          "
              "1A2 1.047  2C9 0.949  2D6 1.048  3A4 0.494  MA 0.885")

        out = pd.DataFrame({NAME_COL: [n for n, e in zip(names, is_ext) if not e]})
        for j, ep in enumerate(TARGET_COLS):
            out[ep] = oof[ch_rows][:, j]
        out.to_csv(args.oof_output, index=False)
        save_pickle(scalers, os.path.join(args.model_dir, "scalers_y.pkl"))
        save_pickle(best_epochs, os.path.join(args.model_dir, "best_epochs.pkl"))
        print(f"\nOOF written: {args.oof_output}   ({(time.time()-t0)/60:.1f} min)")

    if args.mode in ("PREDICT", "BOTH"):
        te = pd.read_csv(args.test_csv)
        te.columns = [c.strip() for c in te.columns]
        sc_ = next(c for c in te.columns if c.lower() == "smiles")
        te = te.rename(columns={sc_: SMILES_COL})
        nm = next((c for c in te.columns
                   if c.lower().replace(" ", "_") in ("molecule_name", "name", "id")), NAME_COL)
        te = te.rename(columns={nm: NAME_COL})
        te["s"] = te[SMILES_COL].apply(
            lambda x: (lambda m: Chem.MolToSmiles(m) if m else x)(std.sanitize(x)))
        scalers = load_pickle(os.path.join(args.model_dir, "scalers_y.pkl"))
        cks = sorted(glob.glob(os.path.join(args.model_dir, "fold_*.ckpt")))
        dummy = np.zeros((len(te), n_task), np.float32)
        dl = build_dataloader(MoleculeDataset(dps(te["s"].tolist(), dummy)),
                              batch_size=64, shuffle=False, num_workers=0)
        acc, n = np.zeros((len(te), n_task)), 0
        for c in cks:
            m = load_weights(c, cfg, n_task, args.aux_weight)
            m.eval(); m.to(torch.device(DEVICE))
            pt = pl.Trainer(accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
                            logger=False, enable_progress_bar=False)
            with torch.inference_mode():
                raw = pt.predict(m, dl)
            acc += np.concatenate([b.cpu().numpy() for b in raw], axis=0).reshape(-1, n_task)
            n += 1
        acc /= max(n, 1)
        res = te[[SMILES_COL, NAME_COL]].copy()
        for j, ep in enumerate(TARGET_COLS):
            res[ep] = scalers[j].inverse_transform(acc[:, j].reshape(-1, 1)).ravel()
        res.to_csv(args.output, index=False)
        print(f"\nPredictions written: {args.output}  (ensemble of {n} folds)")
        print(res[TARGET_COLS].describe().T.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
