"""
CYP pIC50 — Per-target Chemprop with fused target-relevant descriptors (v2)
===========================================================================
The PXR v4_4 architecture applied the way you built it: ONE model per endpoint,
learned message-passing features CONCATENATED with a hand-built descriptor vector
before the readout FFN.

Two differences from `train_cyp_mt_v1.py`, both deliberate:

1.  SINGLE-TASK, four models. The multitask baseline lumped all 4,905 compounds into
    one `very_large` tier, so your adaptive config never actually adapted. Per target
    the counts are 1,412 / 1,285 / 1,493 / 2,335, which puts CYP1A2, CYP2C9 and
    CYP2D6 in `large` (hidden 768, depth 5, 3 FFN layers, 2x augmentation) and
    CYP3A4 in `very_large`. The starved endpoints now get more augmentation and a
    shallower network, which is what the tiers were designed to do.

2.  ISOFORM-SPECIFIC DESCRIPTORS, not one shared vector. See
    `cyp_target_descriptors.py`. A CYP2D6 model gets the basic-nitrogen-to-aromatic
    topological distance that encodes its salt-bridge pharmacophore; a CYP2C9 model
    gets ionisable-acid counts for the Arg108 anion site; CYP1A2 gets fused-aromatic
    extent and planarity for its narrow slot; CYP3A4 gets bulk and flexibility. One
    shared vector would be three-quarters irrelevant for each model.

WHAT THIS TRADES AWAY, and why it is still worth running. Single-task loses the
cross-isoform borrowing that made the multitask baseline strong exactly where the
data is thin — its CYP1A2 out-of-fold ST-RAE was 0.848 against the classical
ensemble's 1.047. This bets that per-target capacity plus domain features beats that
borrowing. It may not. The two are complementary either way: run this, then blend
both into `blend_composite.py` and let the composite decide.

Shared with the multitask script (imported, not duplicated, so the splits are
provably identical): the standardiser, augmenter, split strategies, adaptive config,
and ST-RAE reporting.

Usage
-----
  python train_cyp_st_v2.py --mode BOTH                     # all four isoforms
  python train_cyp_st_v2.py --mode BOTH --isoforms CYP2D6   # one, for a quick look
  python train_cyp_st_v2.py --mode BOTH --no_descriptors    # ablation: is the
                                                            # descriptor block earning
                                                            # its place?
  python train_cyp_st_v2.py --mode BOTH --split_strategy frozen \
      --folds_csv artifacts/folds.csv                       # blendable OOF
"""

import argparse, glob, os, pickle, time, warnings
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
from sklearn.preprocessing import RobustScaler

import cyp_target_descriptors as TD
from train_cyp_mt_v1 import (
    ISOFORMS, TARGET_COLS, SMILES_COL, NAME_COL, CONF_LOW, CONF_HIGH,
    MoleculeStandardizer, SMILESAugmenter, get_adaptive_config, get_folds,
    load_training, set_global_seed, strae, save_pickle, load_pickle,
)

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore")

TRAIN_CSV = "data/cyp-challenge-TRAIN_inhibition.csv"
TEST_CSV  = "data/cyp-challenge-TEST-BLINDED.csv"
GLOBAL_SEED = 42
N_FOLDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TargetDescriptorCalculator:
    """Isoform-specific block, robust-scaled and clipped.

    `lean=True` selects the Section 36 subset: the mechanism-justified columns only,
    35-40% narrower. Measured against the full block on the similarity-reweighted
    yardstick, it is +0.0400 on CYP2C9, +0.0239 on CYP2D6 and level on CYP1A2 and CYP3A4.
    Columns are SELECTED BY NAME from the same calculate() output, so nothing is
    recomputed and the two settings are directly comparable.

    THE WIDTH IS CHECKED ON EVERY transform(). A model trained with one setting and
    predicted with the other is exactly the failure that killed a PREDICT run on
    2026-09-26 -- "X has 62 features, but RobustScaler is expecting 52" -- after the
    descriptor module grew underneath a set of checkpoints. The scaler alone cannot say
    which columns it was fitted on, so the error below names the flag rather than leaving
    the next person to work it out from a shape mismatch.
    """

    def __init__(self, isoform, lean=False, spec=None):
        """`spec` (Section 41) supersedes `lean` when given: 'per_endpoint' uses the
        measured per-isoform recipe -- lean+3D on CYP1A2 and CYP2D6, lean on CYP2C9,
        lean+wide+3D on CYP3A4 -- and 'lean' / 'lean3d' / 'wide3d' force one recipe on
        every endpoint for an A/B. `lean=True` alone is the Section 36 behaviour."""
        self.isoform = isoform
        self.lean = bool(lean)
        self.spec = spec
        self.scaler = RobustScaler()
        self.all_names = list(TD.feature_names(isoform))
        if self.spec:
            # widths depend on which blocks the spec turns on, and the 3D block is only
            # meaningful once there are molecules, so names are filled on first use.
            self.cols = None
            self.names = None
        elif self.lean:
            from cyp_target_descriptors_lean import lean_index, lean_names  # noqa: PLC0415
            self.cols = list(lean_index(isoform))
            self.names = list(lean_names(isoform))
        else:
            self.cols = list(range(len(self.all_names)))
            self.names = list(self.all_names)

    def n_features(self):
        return len(self.names) if self.names else 0

    def _select(self, smiles):
        if self.spec:
            import cyp_descriptor_spec as SPEC                      # noqa: PLC0415
            X, names = SPEC.build(smiles, self.isoform, self.spec,
                                  verbose=self.names is None)
            self.names = names
            return np.asarray(X, dtype=float)
        X, _ = TD.calculate(smiles, self.isoform)
        return np.asarray(X, dtype=float)[:, self.cols]

    def fit_transform(self, smiles):
        return np.clip(self.scaler.fit_transform(self._select(smiles)), -10, 10)

    def transform(self, smiles):
        X = self._select(smiles)
        want = getattr(self.scaler, "n_features_in_", None)
        if want is not None and X.shape[1] != want:
            raise ValueError(
                f"{self.isoform}: this calculator produces {X.shape[1]} columns "
                f"(lean={self.lean}, spec={self.spec}) but the saved scaler was fitted "
                f"on {want}. "
                f"The checkpoints were trained with a different descriptor setting -- "
                f"re-run PREDICT with the matching --lean_descriptors setting, or "
                f"retrain. Do not pad or truncate: the column ORDER would be wrong even "
                f"if the width matched.")
        return np.clip(self.scaler.transform(X), -10, 10)


class SingleTargetMPNN:
    """One Chemprop model for one isoform, descriptors fused at the readout."""

    def __init__(self, isoform, model_dir, use_descriptors=True):
        self.isoform = isoform
        self.model_dir = model_dir
        self.use_descriptors = use_descriptors
        os.makedirs(model_dir, exist_ok=True)
        self.scaler_y = RobustScaler(quantile_range=(1.0, 99.0))
        self.desc_calc: Optional[TargetDescriptorCalculator] = None
        self.augmenter = SMILESAugmenter(random_seed=GLOBAL_SEED)
        self.best_epochs: List[int] = []
        self.models: list = []

    def _build(self, cfg, n_desc):
        hd = cfg["hidden_dim"]
        mp = BondMessagePassing(d_h=hd, depth=cfg["depth"], dropout=cfg["dropout"],
                                activation="relu")
        # THE FUSION: readout input is the learned embedding PLUS the descriptor block.
        ffn = RegressionFFN(input_dim=hd + n_desc, hidden_dim=hd,
                            n_layers=cfg["n_layers"], dropout=cfg["dropout"], n_tasks=1)
        return MPNN(message_passing=mp, agg=MeanAggregation(), predictor=ffn,
                    batch_norm=True)

    def _safe_load(self, path):
        for kw in ({"weights_only": False}, {}):
            try:
                return MPNN.load_from_checkpoint(path, map_location=DEVICE, **kw)
            except Exception:
                continue
        return None

    def _dps(self, smiles, y, desc):
        out = []
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            kw = {"x_d": desc[i].astype(np.float32)} if desc is not None else {}
            out.append(MoleculeDatapoint(mol=mol, y=np.array([float(y[i])], np.float32), **kw))
        return out

    def _predict(self, model, loader, n):
        tr = pl.Trainer(accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
                        logger=False, enable_progress_bar=False)
        with torch.inference_mode():
            raw = tr.predict(model, loader)
        return np.concatenate([p.cpu().numpy() for p in raw]).ravel()[:n]

    def train(self, smiles, y, names, split_strategy, folds_csv, patience=None):
        cfg = get_adaptive_config(len(smiles))
        if patience is not None:
            cfg["patience"] = patience
        print(f"\n{'='*66}")
        print(f"{self.isoform}  N={len(smiles)}  tier -> hidden={cfg['hidden_dim']} "
              f"depth={cfg['depth']} layers={cfg['n_layers']} aug={cfg['augmentation_factor']}x")
        print(f"Descriptors: {'ON' if self.use_descriptors else 'OFF (ablation)'}")
        print(f"{'='*66}")

        y_sc = self.scaler_y.fit_transform(np.asarray(y, float).reshape(-1, 1)).ravel()
        X_desc, n_desc = None, 0
        if self.use_descriptors:
            self.desc_calc = TargetDescriptorCalculator(self.isoform)
            X_desc = self.desc_calc.fit_transform(smiles)
            n_desc = X_desc.shape[1]
            print(f"Descriptor block: {X_desc.shape}  ({self.isoform}-specific)")

        Y2 = np.asarray(y, float).reshape(-1, 1)
        folds = get_folds(split_strategy, smiles, Y=Y2, desc_matrix=X_desc,
                          names=names, folds_csv=folds_csv, n_folds=N_FOLDS,
                          seed=GLOBAL_SEED)

        oof = np.full(len(smiles), np.nan)
        for fi, val_idx in enumerate(folds):
            set_global_seed(GLOBAL_SEED + fi)
            tr_mask = np.ones(len(smiles), bool)
            tr_mask[val_idx] = False
            tr_i = np.where(tr_mask)[0]
            tr_smi = [smiles[i] for i in tr_i]
            vl_smi = [smiles[i] for i in val_idx]

            if cfg["use_augmentation"] and cfg["augmentation_factor"] > 1:
                aug_smi, aug_y, _ = self.augmenter.augment_dataset(
                    tr_smi, y_sc[tr_i].reshape(-1, 1), cfg["augmentation_factor"])
                aug_y = aug_y.ravel()
            else:
                aug_smi, aug_y = tr_smi, y_sc[tr_i]

            aug_d = self.desc_calc.transform(aug_smi) if self.use_descriptors else None
            vl_d = X_desc[val_idx] if self.use_descriptors else None

            tl = build_dataloader(MoleculeDataset(self._dps(aug_smi, aug_y, aug_d)),
                                  batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
            vl = build_dataloader(MoleculeDataset(self._dps(vl_smi, y_sc[val_idx], vl_d)),
                                  batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

            ckpt_fn = f"{self.isoform}_fold_{fi}"
            ckpt_path = os.path.join(self.model_dir, f"{ckpt_fn}.ckpt")
            if os.path.exists(ckpt_path):
                print(f"  fold {fi+1}: checkpoint found, skipping training")
                best = self._safe_load(ckpt_path)
                self.best_epochs.append(-1)
            else:
                model = self._build(cfg, n_desc)
                cb = ModelCheckpoint(dirpath=self.model_dir, filename=ckpt_fn,
                                     monitor="val_loss", mode="min", save_top_k=1)
                trainer = pl.Trainer(
                    accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
                    max_epochs=cfg["max_epochs"],
                    callbacks=[EarlyStopping(monitor="val_loss", patience=cfg["patience"],
                                             mode="min"), cb],
                    enable_progress_bar=True, logger=False, gradient_clip_val=1.0)
                trainer.fit(model, tl, vl)
                self.best_epochs.append(max(0, trainer.current_epoch - cfg["patience"]))
                best = self._safe_load(cb.best_model_path) if cb.best_model_path else None
            if best is None:
                print(f"  fold {fi+1}: WARNING no usable checkpoint")
                continue
            best.eval(); best.to(torch.device(DEVICE))
            keep = [i for i, s in enumerate(vl_smi) if Chem.MolFromSmiles(s)]
            p = self._predict(best, vl, len(keep))
            for k, li in enumerate(keep):
                oof[val_idx[li]] = p[k]
            got = self.scaler_y.inverse_transform(oof[val_idx].reshape(-1, 1)).ravel()
            m = np.isfinite(got)
            print(f"  fold {fi+1}/{len(folds)}: MAE={np.mean(np.abs(np.asarray(y)[val_idx][m]-got[m])):.4f}")

        pred = self.scaler_y.inverse_transform(oof.reshape(-1, 1)).ravel()
        save_pickle(self.scaler_y, os.path.join(self.model_dir, f"{self.isoform}_scaler_y.pkl"))
        save_pickle(self.best_epochs, os.path.join(self.model_dir, f"{self.isoform}_best_epochs.pkl"))
        if self.use_descriptors:
            save_pickle(self.desc_calc.scaler,
                        os.path.join(self.model_dir, f"{self.isoform}_desc_scaler.pkl"))
        return pred

    def predict(self, smiles):
        ck = sorted(glob.glob(os.path.join(self.model_dir, f"{self.isoform}_fold_*.ckpt")))
        if not ck:
            raise RuntimeError(f"no checkpoints for {self.isoform} in {self.model_dir}")
        self.scaler_y = load_pickle(os.path.join(self.model_dir, f"{self.isoform}_scaler_y.pkl"))
        if self.use_descriptors:
            self.desc_calc = TargetDescriptorCalculator(self.isoform)
            self.desc_calc.scaler = load_pickle(
                os.path.join(self.model_dir, f"{self.isoform}_desc_scaler.pkl"))
        keep = [i for i, s in enumerate(smiles) if Chem.MolFromSmiles(s)]
        smi = [smiles[i] for i in keep]
        d = self.desc_calc.transform(smi) if self.use_descriptors else None
        loader = build_dataloader(MoleculeDataset(self._dps(smi, np.zeros(len(smi)), d)),
                                  batch_size=64, shuffle=False, num_workers=0)
        acc = np.zeros(len(smi))
        n = 0
        for c in ck:
            m = self._safe_load(c)
            if m is None:
                continue
            m.eval(); m.to(torch.device(DEVICE))
            acc += self._predict(m, loader, len(smi)); n += 1
        print(f"  {self.isoform}: averaged {n} fold model(s)")
        out = np.full(len(smiles), np.nan)
        out[keep] = self.scaler_y.inverse_transform((acc / max(n, 1)).reshape(-1, 1)).ravel()
        return np.where(np.isfinite(out), out, np.nanmedian(out))


def main():
    p = argparse.ArgumentParser(description="CYP per-target Chemprop with fused descriptors")
    p.add_argument("--mode", choices=["TRAIN", "PREDICT", "BOTH"], default="BOTH")
    p.add_argument("--isoforms", nargs="*", default=list(ISOFORMS))
    p.add_argument("--train_csv", default=TRAIN_CSV)
    p.add_argument("--test_csv", default=TEST_CSV)
    p.add_argument("--model_dir", default="models_cyp_st_v2")
    p.add_argument("--output", default="predictions_cyp_st_v2.csv")
    p.add_argument("--oof_output", default="oof_chemprop_st_v2.csv")
    p.add_argument("--split_strategy",
                   choices=["scaffold", "butina", "descriptor", "activity_scaffold", "frozen"],
                   default="activity_scaffold")
    p.add_argument("--folds_csv", default="artifacts/folds.csv")
    p.add_argument("--no_descriptors", action="store_true",
                   help="ablation: learned features only, for comparison")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = p.parse_args()

    set_global_seed(args.seed)
    print(f"Device: {DEVICE}   descriptors: {'OFF' if args.no_descriptors else 'ON'}")
    if not args.no_descriptors:
        print("Validating descriptor blocks against canonical substrates...")
        if not TD.validate(verbose=False):
            raise SystemExit("descriptor validation FAILED - do not trust the features")
        print("  passed\n")

    t0 = time.time()
    std = MoleculeStandardizer()
    df = load_training(args.train_csv, std)
    smiles_all = df["SMILES_clean"].tolist()
    names_all = df[NAME_COL].tolist()

    oof = pd.DataFrame({NAME_COL: names_all})
    report = []

    if args.mode in ("TRAIN", "BOTH"):
        for iso in args.isoforms:
            ep = f"{iso}_pIC50_direct_inhibition"
            y_all = df[ep].to_numpy(float)
            m = np.isfinite(y_all)
            smiles = [s for s, k in zip(smiles_all, m) if k]
            names = [n for n, k in zip(names_all, m) if k]
            y = y_all[m]

            mdl = SingleTargetMPNN(iso, args.model_dir, not args.no_descriptors)
            pred = mdl.train(smiles, y, names, args.split_strategy, args.folds_csv,
                             args.patience)

            col = np.full(len(df), np.nan)
            col[np.where(m)[0]] = pred
            oof[ep] = np.where(np.isfinite(col), col, np.nanmedian(pred))

            ok = np.isfinite(pred) & np.isfinite(y)
            row = dict(endpoint=iso, n=int(ok.sum()),
                       mae=float(np.mean(np.abs(y[ok] - pred[ok]))),
                       r2=float(1 - np.sum((y[ok]-pred[ok])**2)/(np.sum((y[ok]-y[ok].mean())**2)+1e-12)))
            lo, hi = ep + CONF_LOW, ep + CONF_HIGH
            if lo in df.columns and hi in df.columns:
                row["ST_RAE"] = strae(y[ok], pred[ok],
                                      df.loc[m, lo].to_numpy(float)[ok],
                                      df.loc[m, hi].to_numpy(float)[ok])
            report.append(row)
            print(f"  -> {iso}: " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in row.items() if k != "endpoint"))

        rep = pd.DataFrame(report)
        if len(rep):
            rep.loc[len(rep)] = dict(endpoint="MA", n=int(rep["n"].sum()),
                                     **{c: rep[c].mean() for c in rep.columns
                                        if c not in ("endpoint", "n")})
        print("\n" + rep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nreference - multitask v1 OOF ST-RAE: "
              "1A2 0.848  2C9 0.663  2D6 0.946  3A4 0.558  MA 0.754")
        print("reference - classical ensemble:       "
              "1A2 1.047  2C9 0.949  2D6 1.048  3A4 0.494  MA 0.885")
        oof.to_csv(args.oof_output, index=False)
        rep.to_csv(os.path.join(args.model_dir, "oof_report.csv"), index=False)
        print(f"\nOOF written: {args.oof_output}")

    if args.mode in ("PREDICT", "BOTH"):
        te = pd.read_csv(args.test_csv)
        te.columns = [c.strip() for c in te.columns]
        smi_col = next((c for c in te.columns if c.lower() == "smiles"), SMILES_COL)
        te = te.rename(columns={smi_col: SMILES_COL})
        nm = next((c for c in te.columns
                   if c.lower().replace(" ", "_") in ("molecule_name", "name", "id")), NAME_COL)
        te = te.rename(columns={nm: NAME_COL})
        te["mol"] = te[SMILES_COL].apply(std.sanitize)
        te["SMILES_clean"] = te["mol"].apply(
            lambda m: Chem.MolToSmiles(m, canonical=True) if m else None)
        te["SMILES_clean"] = te["SMILES_clean"].fillna(te[SMILES_COL])

        out = te[[SMILES_COL, NAME_COL]].copy()
        for iso in args.isoforms:
            mdl = SingleTargetMPNN(iso, args.model_dir, not args.no_descriptors)
            out[f"{iso}_pIC50_direct_inhibition"] = mdl.predict(te["SMILES_clean"].tolist())
        out.to_csv(args.output, index=False)
        print(f"\nPredictions written: {args.output}")
        cols = [f"{i}_pIC50_direct_inhibition" for i in args.isoforms]
        print(out[cols].describe().T.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
