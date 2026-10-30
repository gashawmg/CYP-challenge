#!/usr/bin/env python
"""Multi-head ChemProp: one shared encoder, 13 heads, only 4 of them submitted.

    python train_cyp_mh.py --smoke                      # 1 fold, 3 epochs -- run FIRST
    python train_cyp_mh.py --mode BOTH --model_dir models_mh1
    python train_cyp_mh.py --mode BOTH --no-external    # ablation: does breadth help?

WHY. Every external-data attempt so far converted an outside measurement into a
pseudo-pIC50 and poured it into the scored column, which needs a cross-assay correction we
never had (+0.052 out-of-fold macro rho, +0.004 blind). The current leader
(https://supercowpowers.github.io/workbench/blogs/cyp_challenge/, macro ST-RAE 0.4378,
rank 1 of 81) never harmonises: each assay gets its own head on a shared trunk, and "only
the four scored pIC50 heads are ever submitted; the rest exist to shape the
representation".

That is what this trains. `cypkit/multihead.py` builds the table; nothing is converted.

WHAT IS DIFFERENT FROM train_cyp_st_v3.py

  * ONE model for all four isoforms instead of four single-target models, because the
    point is a shared encoder.
  * 13 output tasks. NaN marks "this compound has no measurement in this assay";
    ChemProp v2 masks those out of the loss natively (`batch.Y.isfinite()`).
  * Folds are assigned on CHALLENGE compounds only. External compounds carry no scored
    label, so they live permanently in the training half of every fold and can never
    appear in a validation set. That is also what stops them contaminating the OOF file.
  * Per-head target scaling, fitted on the TRAINING half of each fold. Heads are in
    different units (pIC50 vs log2 fold-change) so a single scaler would be meaningless.

WEIGHTING, AND ONE HONEST APPROXIMATION. ChemProp takes ONE weight per datapoint, not one
per task, so the per-head weight matrix from `multihead.build_table` is collapsed to a
per-compound weight: 1.0 if the compound carries any challenge label, `--public-weight`
(default 0.30, the leader's value) otherwise. The two differ only for compounds that
carry BOTH challenge and external labels, and there are 175 of those out of 21,047 --
0.8% of rows. The script prints the exact count so the approximation stays visible.

WHAT TO CHECK WHEN IT FINISHES. `oof_st_mh.csv` is written in the same shape as every
other OOF file, so:

    python scripts/blind_validator.py --compare oof_st_mh.csv oof_st_v6.csv

rho is the number that matters -- v6's macro is 0.6037 and three retrains have failed to
beat it. If this one does not either, the multi-head idea is dead for us and the honest
move is to say so rather than tune it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation
from chemprop.nn.predictors import RegressionFFN
from rdkit import Chem, RDLogger

from cypkit import config as C, io, multihead as MH, scoring as S
import cyp_descriptor_spec as DS

RDLogger.DisableLog("rdApp.*")
GLOBAL_SEED, N_FOLDS = 42, 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def safe_load(path):
    """Load a ChemProp checkpoint across PyTorch's weights_only default change.

    PyTorch >= 2.6 defaults `torch.load(weights_only=True)`, which refuses a Lightning
    checkpoint because it carries pickled hyperparameter objects. `train_cyp_st_v3.py`
    has carried this exact ladder since it was first written; I mirrored that script's
    structure for this one and dropped the guard, which is what broke the smoke test.
    Try the explicit override first, then the bare call for older versions.
    """
    last = None
    for kw in ({"weights_only": False}, {}):
        try:
            return MPNN.load_from_checkpoint(path, map_location=DEVICE, **kw)
        except Exception as e:                       # noqa: BLE001
            last = e
    raise RuntimeError(
        f"could not load {path}. Last error: {type(last).__name__}: {last}\n"
        f"If this is a torch.load weights_only refusal on a newer PyTorch, the "
        f"override above is already being tried -- report the message.")


def datapoints(smiles, Y, W, Xd=None):
    """`Xd` is the ALREADY-SCALED descriptor matrix, one row per entry of `smiles`.

    It is indexed by position in `smiles`, not by the returned `keep` -- a molecule RDKit
    cannot parse is skipped here, so scaling and slicing must both happen against the
    original ordering or the descriptors silently belong to the wrong compounds.
    """
    out, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        kw = {}
        if Xd is not None:
            kw["x_d"] = Xd[i].astype(np.float32)
        out.append(MoleculeDatapoint(mol=m, y=Y[i].astype(np.float32),
                                     weight=float(W[i]), **kw))
        keep.append(i)
    return out, np.array(keep)


def scale_desc(X, rows):
    """Mean/sd fitted on `rows` only. Returns (scaled X, mu, sd)."""
    mu = np.nanmean(X[rows], axis=0)
    sd = np.nanstd(X[rows], axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    return np.nan_to_num((X - mu) / sd), mu, sd


def main() -> int:
    # Declared before argparse reads GLOBAL_SEED as the --seed default. Python
    # resolves `global` for the WHOLE function body, so placing it next to the
    # assignment lower down is a SyntaxError that ast.parse does not catch --
    # only compile() does. Same trap as train_cyp_st_v3.py.
    global GLOBAL_SEED                                        # noqa: PLW0603
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["TRAIN", "PREDICT", "BOTH"], default="BOTH")
    p.add_argument("--model_dir", default="models_cyp_mh")
    p.add_argument("--oof_output", default="oof_st_mh.csv")
    p.add_argument("--output", default="pred_st_mh.csv")
    p.add_argument("--external", default="artifacts/external_cyp_v2.csv")
    p.add_argument("--no-external", dest="no_external", action="store_true",
                   help="ablation: same architecture, challenge heads only")
    p.add_argument("--no-screen", dest="no_screen", action="store_true")
    p.add_argument("--tdi-isoforms", nargs="*", default=None,
                   help="which endpoints get a TDI-condition head. Default CYP3A4 "
                        "only (Section 71): it is the only one where the TDI arm adds "
                        "compounds. Pass e.g. CYP3A4 CYP2C9 to widen.")
    p.add_argument("--no-tdi", dest="no_tdi", action="store_true",
                   help="drop the four TDI-condition heads (Section 70 ablation). "
                        "They add 1,249 CYP3A4 compounds the scored file does not "
                        "have, plus a paired second reading on the rest.")
    p.add_argument("--no-maxresp", dest="no_maxresp", action="store_true",
                   help="drop the 5 ex_<iso>_maxresp heads. They are the "
                        "whole point of external_cyp_v2: they carry the "
                        "42,460 screen-inactive rows that the old file "
                        "pinned at a constant pseudo-pIC50. Use this only "
                        "to ablate them.")
    p.add_argument("--public-weight", type=float, default=0.30)
    p.add_argument("--descriptor_spec", choices=["none", "shared", "shared3d"],
                   default="none",
                   help="Target-specific descriptors on the shared trunk. 'none' (the "
                        "default) reproduces the member currently in the blend, bit for "
                        "bit. 'shared' adds the isoform-agnostic intersection of the "
                        "Section 36 lean blocks; 'shared3d' adds the 8 conformer shape "
                        "columns on top. This model had NO descriptors at all until now, "
                        "while every descriptor result in Sections 36-43 was measured on "
                        "the single-task path -- which is the weaker architecture.")
    p.add_argument("--hidden", type=int, default=400)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--max-external", type=int, default=0,
                   help="keep only the N external compounds most similar to the "
                        "challenge deck. 0 = keep all 15,970. Training time is roughly "
                        "linear in the compound count, so 5000 costs about a third of "
                        "the wall clock. Use it if the full run is too slow to iterate "
                        "on; the ablation it supports (does MORE breadth help?) is worth "
                        "knowing anyway.")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED,
                   help="Replaces the module-level GLOBAL_SEED. Exposed so "
                        "scripts/seed_ensemble.py can average several runs of this "
                        "model; fold seeds remain seed + fold_index.")
    p.add_argument("--smoke", action="store_true",
                   help="1 fold, 3 epochs, 2000 compounds -- catches wiring bugs in "
                        "minutes instead of hours. ALWAYS run this first.")
    a = p.parse_args()
    GLOBAL_SEED = a.seed

    # --smoke MUST NOT WRITE TO THE LIVE ARTIFACT NAMES. On 2026-09-28 a smoke run took
    # the defaults and overwrote oof_st_mh.csv -- a REGISTERED BLEND MEMBER -- with its
    # 263-row fold-0 stub, and replaced models_cyp_mh/scale_0.npy with the smoke label
    # scaler. Nothing warned, because the OOF writer is not guarded by --smoke and the
    # blend search checks only that a member file EXISTS, not that it has the right rows.
    # A test that can destroy the thing it is testing is a defect in the test, so --smoke
    # now redirects any output path left at its default.
    if a.smoke:
        d = p.parse_args([])          # the defaults, to detect what the user did not set
        moved = []
        if a.model_dir == d.model_dir:
            a.model_dir = d.model_dir + "_smoke"; moved.append(f"model_dir={a.model_dir}")
        if a.oof_output == d.oof_output:
            a.oof_output = "smoke_" + d.oof_output; moved.append(a.oof_output)
        if a.output == d.output:
            a.output = "smoke_" + d.output; moved.append(a.output)
        if moved:
            print(f"  --smoke: redirected default outputs -> {', '.join(moved)}")

    tab = MH.build_table(external_csv=None if a.no_external else a.external,
                         with_screen=not a.no_screen,
                         with_maxresp=not a.no_maxresp,
                         with_tdi=False if a.no_tdi else (a.tdi_isoforms or True),
                         public_weight=a.public_weight)
    smiles, Y, names = tab["smiles"], tab["Y"], tab["names"]
    is_ch = tab["is_challenge"]
    scored = MH.scored_head_index(names)

    # per-compound weight (see the module docstring for why this is an approximation)
    ext_only = ~is_ch
    Wc = np.where(ext_only, a.public_weight, 1.0).astype(np.float32)
    both = int((is_ch & np.isfinite(Y[:, [j for j, n in enumerate(names)
                                          if n.startswith("ex_")]]).any(1)).sum()) \
        if any(n.startswith("ex_") for n in names) else 0
    print(f"per-compound weights: {int(is_ch.sum())} challenge at 1.0, "
          f"{int(ext_only.sum())} external at {a.public_weight}")
    print(f"  compounds carrying BOTH kinds of label (where the per-compound weight is "
          f"an approximation): {both}  ({both/len(smiles)*100:.1f}%)")

    if a.max_external and int(ext_only.sum()) > a.max_external:
        rng = np.random.default_rng(GLOBAL_SEED)
        drop = rng.permutation(np.where(ext_only)[0])[a.max_external:]
        mask = np.ones(len(smiles), bool); mask[drop] = False
        smiles = [s_ for s_, m_ in zip(smiles, mask) if m_]
        Y, Wc, is_ch = Y[mask], Wc[mask], is_ch[mask]
        ext_only = ~is_ch
        print(f"  --max-external {a.max_external}: kept {int(ext_only.sum())} external "
              f"compounds, {len(smiles)} total")

    if a.smoke:
        keep = np.r_[np.where(is_ch)[0][:1200], np.where(ext_only)[0][:800]]
        smiles = [smiles[i] for i in keep]; Y = Y[keep]; Wc = Wc[keep]; is_ch = is_ch[keep]
        a.epochs, a.patience = 3, 3
        print(f"\nSMOKE: {len(smiles)} compounds, {a.epochs} epochs, 1 fold\n")

    # ---- descriptors, computed ONCE on the final row set -----------------------------
    # After --max-external and --smoke have subset `smiles`, so row i of X is row i of
    # `smiles` for the rest of the run.
    # Guarded on TRAIN: in PREDICT-only mode the ~20,000 training rows are never used,
    # and with shared3d that would be a 15,000-conformer embed for nothing.
    X_desc, desc_names = None, []
    if a.descriptor_spec != "none" and a.mode in ("TRAIN", "BOTH"):
        X_desc, desc_names = DS.build_shared(
            smiles, use_3d=(a.descriptor_spec == "shared3d"))
        print(f"descriptors: {a.descriptor_spec}, {X_desc.shape[1]} columns over "
              f"{len(smiles)} compounds")
    n_desc = 0 if X_desc is None else X_desc.shape[1]

    # folds on challenge compounds only; external is always training
    folds = pd.read_csv(C.artifact("folds.csv"))
    fold_of = dict(zip(folds[C.SMILES_COL], folds["fold"]))
    fold = np.array([fold_of.get(s, -1) for s in smiles])
    print(f"fold sizes (−1 = external, always training): "
          f"{ {int(k): int((fold==k).sum()) for k in sorted(set(fold))} }")

    # ---- ground-truth intervals, for the leaderboard metrics per fold ---------------
    gt = S.build_ground_truth(io.load("train_inhibition")).set_index(C.ID_COL)
    folds_df = pd.read_csv(C.artifact("folds.csv"))
    id_of = {v: k for k, v in zip(folds_df[C.ID_COL], folds_df[C.SMILES_COL])}

    def report(pred, rowmask, title):
        """Every leaderboard metric on the rows in `rowmask`, per scored endpoint."""
        rows = {}
        ids = [id_of.get(smiles[i]) for i in np.where(rowmask)[0]]
        sel = [i for i, m in zip(np.where(rowmask)[0], ids) if m in gt.index]
        mids = [m for m in ids if m in gt.index]
        if not mids:
            return "  (no scored labels in this fold)"
        g = gt.loc[mids]
        for iso, j in zip(C.ISOFORMS, scored):
            ep = f"{iso}_pIC50_direct_inhibition"
            rows[iso] = S.regression_report(
                g[ep].to_numpy(), pred[sel, j],
                g[ep + C.CONF_LOW_SUFFIX].to_numpy(),
                g[ep + C.CONF_HIGH_SUFFIX].to_numpy())
        return S.format_report(rows, title)

    os.makedirs(a.model_dir, exist_ok=True)
    n_tasks = Y.shape[1]
    oof = np.full((len(smiles), n_tasks), np.nan, np.float32)
    ks = [0] if a.smoke else sorted(k for k in set(fold) if k >= 0)
    fold_times: list[float] = []

    if a.mode in ("TRAIN", "BOTH"):
        for k in ks:
            set_seed(GLOBAL_SEED + k)
            va = fold == k
            tr = ~va                      # external (fold −1) is always in training
            # per-head scaler on the TRAINING half only
            mu = np.nanmean(np.where(tr[:, None], Y, np.nan), axis=0)
            sd = np.nanstd(np.where(tr[:, None], Y, np.nan), axis=0)
            sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            Ys = (Y - mu) / sd

            # descriptors scaled on the TRAINING half only, exactly like the label heads
            if X_desc is None:
                Xtr = Xva = None
            else:
                Xs, dmu, dsd = scale_desc(X_desc, tr)
                Xtr, Xva = Xs[tr], Xs[va]
                np.save(os.path.join(a.model_dir, f"dscale_{k}.npy"),
                        np.vstack([dmu, dsd]))

            dtr, _ = datapoints([smiles[i] for i in np.where(tr)[0]], Ys[tr], Wc[tr], Xtr)
            dva, kva = datapoints([smiles[i] for i in np.where(va)[0]], Ys[va], Wc[va],
                                  Xva)
            ltr = build_dataloader(MoleculeDataset(dtr), batch_size=a.batch, shuffle=True,
                                   num_workers=0)
            lva = build_dataloader(MoleculeDataset(dva), batch_size=a.batch, shuffle=False,
                                   num_workers=0)

            mp = BondMessagePassing(d_h=a.hidden, depth=a.depth, dropout=a.dropout)
            ffn = RegressionFFN(input_dim=a.hidden + n_desc, hidden_dim=a.hidden,
                                n_layers=a.layers, dropout=a.dropout, n_tasks=n_tasks)
            model = MPNN(message_passing=mp, agg=MeanAggregation(), predictor=ffn,
                         batch_norm=True)
            ck = ModelCheckpoint(dirpath=a.model_dir, filename=f"fold_{k}",
                                 monitor="val_loss", mode="min", save_top_k=1)
            tr_ = pl.Trainer(accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
                             max_epochs=a.epochs, logger=False, enable_progress_bar=True,
                             gradient_clip_val=1.0,
                             callbacks=[EarlyStopping(monitor="val_loss",
                                                      patience=a.patience, mode="min"),
                                        ck])
            print(f"\n=== fold {k}: train {int(tr.sum())} / val {int(va.sum())} ===")
            t0 = time.time()
            tr_.fit(model, ltr, lva)
            fold_times.append(time.time() - t0)

            best = safe_load(ck.best_model_path) if ck.best_model_path else model
            best.eval().to(torch.device(DEVICE))
            with torch.inference_mode():
                pr = pl.Trainer(accelerator="gpu" if DEVICE == "cuda" else "cpu",
                                devices=1, logger=False,
                                enable_progress_bar=False).predict(best, lva)
            pr = np.concatenate([x.cpu().numpy() for x in pr]).reshape(-1, n_tasks)
            idx = np.where(va)[0][kva]
            oof[idx] = pr[:len(kva)] * sd + mu        # back to the original units
            np.save(os.path.join(a.model_dir, f"scale_{k}.npy"),
                    np.vstack([mu, sd]))
            print()
            print(report(oof, va & is_ch, f"fold {k} — held-out compounds"))
            if fold_times:
                el = fold_times[-1]
                done, left = len(fold_times), len(ks) - len(fold_times)
                avg = sum(fold_times) / done
                print(f"\n  fold {k} trained in {el/60:.1f} min "
                      f"({el/max(tr_.current_epoch, 1):.1f}s per epoch, "
                      f"{tr_.current_epoch} epochs)")
                if left:
                    print(f"  PROJECTED: {left} folds left x {avg/60:.1f} min "
                          f"= {left*avg/60:.0f} min remaining "
                          f"(total ~{(done+left)*avg/60:.0f} min)")

        # ---- OOF file, scored heads only, in the shape every other script expects -----
        out = pd.DataFrame({C.SMILES_COL: smiles})
        keep = is_ch & np.isfinite(oof[:, scored]).any(1)
        out = out[keep].copy()
        inv = {v: k for k, v in zip(folds[C.ID_COL], folds[C.SMILES_COL])}
        out[C.ID_COL] = [inv.get(s) for s in out[C.SMILES_COL]]
        for iso, j in zip(C.ISOFORMS, scored):
            out[f"{iso}_pIC50_direct_inhibition"] = oof[keep, j]
        out = out[[C.ID_COL] + [f"{i}_pIC50_direct_inhibition" for i in C.ISOFORMS]]
        out.to_csv(a.oof_output, index=False)
        print(f"\nOOF written: {a.oof_output} ({len(out)} compounds, "
              f"scored heads only -- the other {n_tasks - 4} exist to shape the trunk)")
        print()
        print(report(oof, keep, "ALL FOLDS POOLED — this is the run's real score"))
        print("\n  For reference on the same metric, oof_st_v6.csv pools to")
        print("  rho 0.6128  tau 0.4415  R2 0.300  MAE 0.589  ST-RAE 0.763  placed 0.6707")

    if a.mode in ("PREDICT", "BOTH") and not a.smoke:
        test = pd.read_csv(C.DATA_DIR / C.DATA_FILES["test"])
        test.columns = [c.strip() for c in test.columns]
        Xtest_raw = None
        if a.descriptor_spec != "none":
            Xtest_raw, _ = DS.build_shared(
                test[C.SMILES_COL].tolist(), use_3d=(a.descriptor_spec == "shared3d"))
        preds = []
        for k in ks:
            f = os.path.join(a.model_dir, f"fold_{k}.ckpt")
            if not os.path.exists(f):
                continue
            mu, sd = np.load(os.path.join(a.model_dir, f"scale_{k}.npy"))
            m = safe_load(f).eval().to(DEVICE)
            Xte = None
            dsp = os.path.join(a.model_dir, f"dscale_{k}.npy")
            if os.path.exists(dsp):
                dmu, dsd = np.load(dsp)
                if Xtest_raw is None:
                    raise SystemExit(
                        f"{a.model_dir} was TRAINED WITH DESCRIPTORS (dscale_{k}.npy is "
                        f"present) but this run passed --descriptor_spec none. Predicting "
                        f"without them would feed the checkpoint a different input width. "
                        f"Re-run PREDICT with the same --descriptor_spec used to train.")
                if Xtest_raw.shape[1] != len(dmu):
                    raise SystemExit(
                        f"descriptor width mismatch: this run builds "
                        f"{Xtest_raw.shape[1]} columns, fold {k} was trained on "
                        f"{len(dmu)}. Do not pad or truncate -- the column ORDER would be "
                        f"wrong even if the width matched.")
                Xte = np.nan_to_num((Xtest_raw - dmu) / dsd)
            elif Xtest_raw is not None:
                raise SystemExit(
                    f"--descriptor_spec {a.descriptor_spec} was given but {a.model_dir} "
                    f"has no dscale_{k}.npy, so fold {k} was trained WITHOUT descriptors.")
            d, kk = datapoints(test[C.SMILES_COL].tolist(),
                               np.zeros((len(test), n_tasks), np.float32),
                               np.ones(len(test), np.float32), Xte)
            dl = build_dataloader(MoleculeDataset(d), batch_size=a.batch, shuffle=False,
                                  num_workers=0)
            with torch.inference_mode():
                pr = pl.Trainer(accelerator="gpu" if DEVICE == "cuda" else "cpu",
                                devices=1, logger=False,
                                enable_progress_bar=False).predict(m, dl)
            pr = np.concatenate([x.cpu().numpy() for x in pr]).reshape(-1, n_tasks)
            full = np.full((len(test), n_tasks), np.nan, np.float32)
            full[kk] = pr[:len(kk)] * sd + mu
            preds.append(full)
        P = np.nanmean(np.stack(preds), axis=0)
        sub = test[[C.SMILES_COL, C.ID_COL]].copy()
        for iso, j in zip(C.ISOFORMS, scored):
            sub[f"{iso}_pIC50_direct_inhibition"] = P[:, j]
        sub.to_csv(a.output, index=False)
        print(f"Predictions written: {a.output} ({len(sub)} rows, "
              f"averaged over {len(preds)} folds)")

    print("\nNEXT:  python scripts/blind_validator.py --compare "
          f"{a.oof_output} oof_st_v6.csv")
    print("  Read rho, not ST-RAE. v6 macro rho is 0.6037 and is the bar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
