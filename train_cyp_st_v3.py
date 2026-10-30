#!/usr/bin/env python
"""Per-target ChemProp, descriptors fused at the readout, TRAINED ON THE INACTIVES TOO.

    python train_cyp_st_v3.py --mode BOTH
    python train_cyp_st_v3.py --mode BOTH --no_censored     # the v2 ablation
    python train_cyp_st_v3.py --mode BOTH --no_descriptors  # descriptor ablation

WHAT THIS CHANGES, AND WHY IT IS THE WHOLE POINT
------------------------------------------------
v2 is the best single model we have on held-out CURVES -- OOF ST-RAE 0.7345 against
0.7456 for UniMol and 0.7537 for the multitask ChemProp. On the dense proxy it is the
WORST thing we have ever scored: 1.4623, against 0.68-0.73 for the classical learners.

That is not a modelling failure, it is a data-coverage failure, and it is entirely our
doing. v2 trains on the 4,905 compounds with a fitted dose-response curve. The other
751 compounds were screened at a single concentration, did not hit, and were dropped.
So the model has never once been shown a molecule that does nothing, and it prices
every molecule as if it were drawn from the hit population. Measured directly:
ChemProp predicts a mean of 4.24 on known inactives and lands 30.4% of them BELOW the
assay floor.

The blend then punishes it for that. In the 11-member composite refit, `chempropst`
draws weight on CYP3A4 (0.23) and zero on the other three endpoints -- the dense term
prices it out. We are paying for a model we then decline to use.

THE FIX. Bring the censored rows in, and give the loss a way to say "below the floor,
value unknown". Chemprop supports exactly this: a datapoint carries `lt_mask`, and
`BoundedMSE` clamps a prediction that already satisfies the bound before taking the
squared error. So a compound known only to be inactive costs nothing while the model
predicts anything under the floor, and is penalised as soon as it predicts activity.
By itself that invents no numbers. `--pseudo` (added later) goes one step further and
turns the screen's log2fc into an interval-valued estimate -- see the flag's help and
`cypkit.labels.pseudo_from_screen`; it is off by default so the two remain separable.

This is the same left-censored treatment the classical learners have had all along
(`--censored_weight`, `deadzone` loss). It is the reason they score 0.68 on dense
while the neural models score 1.06-1.46. The neural models were never given the chance.

WHAT TO EXPECT. OOF ST-RAE should move little -- the censored rows add no curves. The
dense term should fall a long way. If it lands near the classical 0.70, this single
model's composite is ~0.79 on its own, against 0.8097 for the entire current 11-member
blend, and the blend refit will finally give it real weight on all four endpoints.

CYP2D6 IS THE ONE TO WATCH. It is everyone's worst endpoint (best ST-RAE 0.610, best
Spearman 0.459) and v2 collapsed on it outright: predicted spread 0.42 of measured
out-of-fold, 0.27 on the blind test -- close to a constant. 2D6 also has the lowest
active fraction of the panel, which is precisely the regime where throwing away the
inactives hurts most, so this is where the censored rows should pay.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.nn import BondMessagePassing, MeanAggregation
from chemprop.nn.predictors import RegressionFFN
from chemprop.nn.metrics import BoundedMSE


class DeadZoneSpreadMSE(BoundedMSE):
    """Bounded MSE with a dead zone, plus a penalty for predicting too narrowly.

    TWO CHANGES, BOTH AIMED AT THE SAME MEASURED DEFECT.

    Dead zone. ST-RAE charges nothing for a prediction inside the compound's credible
    interval. Squared error charges for every log unit, so it spends capacity chasing
    precision the metric does not reward and buys that precision by shrinking toward the
    mean. `dead_zone` (in SCALED units, i.e. multiples of the label scale) is subtracted
    from |residual| before squaring, so errors smaller than the assay's own resolution
    cost nothing.

    Spread. Our predictions come out ~4x narrower than the blind labels: CYP2D6 sd 0.33
    against an optimal 0.91. Some shrinkage is correct for a conditional mean, but ST-RAE
    is an absolute-error metric with a dead zone and it rewards reaching the tails. This
    adds lambda * max(0, sd(target) - sd(pred))^2 per batch -- ONE-SIDED, so it can push
    a collapsed predictor out but never inflate one that is already wide enough, and it
    is measured against the batch's own targets rather than a constant, so it cannot ask
    the model to hallucinate spread the labels do not have.

    Censored rows keep the bounded behaviour: free below the bound, charged above it.
    """

    def __init__(self, dead_zone: float = 0.0, spread_lambda: float = 0.0, **kw):
        super().__init__(**kw)
        self.dead_zone = float(dead_zone)
        self.spread_lambda = float(spread_lambda)

    def _calc_unreduced_loss(self, preds, targets, mask, weights, lt_mask, gt_mask):
        p = preds if preds.ndim == targets.ndim else preds.squeeze(-1)
        r = p - targets
        # a left-censored row is satisfied by any prediction below its bound
        r = torch.where(lt_mask & (r < 0), torch.zeros_like(r), r)
        r = torch.where(gt_mask & (r > 0), torch.zeros_like(r), r)
        r = torch.clamp(r.abs() - self.dead_zone, min=0.0)
        return r * r

    def update(self, preds, targets, mask=None, weights=None, lt_mask=None, gt_mask=None):
        super().update(preds, targets, mask, weights, lt_mask, gt_mask)
        if self.spread_lambda <= 0:
            return
        p = preds.reshape(preds.shape[0], -1)[:, 0]
        t = targets.reshape(targets.shape[0], -1)[:, 0]
        keep = torch.ones_like(t, dtype=torch.bool) if lt_mask is None \
            else ~lt_mask.reshape(lt_mask.shape[0], -1)[:, 0]
        if int(keep.sum()) < 8:
            return
        gap = torch.clamp(t[keep].std() - p.std(), min=0.0)
        n = mask.sum() if mask is not None else torch.tensor(
            float(p.numel()), device=p.device)
        self.total_loss = self.total_loss + self.spread_lambda * gap * gap * n
from chemprop.models import MPNN

from rdkit import Chem, rdBase
from sklearn.preprocessing import RobustScaler

# the isoform-specific descriptor block and its scaler live in v2 -- reused verbatim
# so the two runs differ ONLY in whether the censored rows are present
from train_cyp_st_v2 import TargetDescriptorCalculator
from cypkit import config as C, io, labels as L, splits
from train_cyp_mt_v1 import (
    ISOFORMS, SMILES_COL, NAME_COL, MoleculeStandardizer, SMILESAugmenter,
    get_adaptive_config, get_folds, load_training, set_global_seed,
    save_pickle, load_pickle,
)

rdBase.DisableLog("rdApp.error"); rdBase.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# QUIET. Lightning prints a hardware banner and a litmodels advertisement for every
# Trainer it builds -- and this script builds one per fold per isoform, so the real
# output drowns. Both are INFO-level on the lightning loggers, so raising the level to
# ERROR removes them and leaves genuine failures visible.
#
# ONE BANNER IS WORTH KEEPING and the script prints it itself: whether CUDA was found.
# Silencing "GPU available: False" without saying so would hide a CPU run that takes
# hours instead of minutes.
# ---------------------------------------------------------------------------
for _lg in ("lightning.pytorch", "lightning.pytorch.utilities.rank_zero",
            "lightning.pytorch.accelerators.cuda", "lightning.fabric",
            "lightning.fabric.utilities.rank_zero", "pytorch_lightning",
            "pytorch_lightning.utilities.rank_zero",
            "pytorch_lightning.accelerators.cuda"):
    logging.getLogger(_lg).setLevel(logging.ERROR)
os.environ.setdefault("POSSIBLE_USER_WARNINGS", "off")
os.environ.setdefault("LIGHTNING_MODEL_REGISTRY_DISABLE", "1")

TRAIN_CSV = "data/cyp-challenge-TRAIN_inhibition.csv"
TEST_CSV = "data/cyp-challenge-TEST-BLINDED.csv"
GLOBAL_SEED = 42
N_FOLDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _clean(std, smiles):
    """Canonical SMILES via the project's standardizer (largest fragment, uncharged)."""
    mol = std.sanitize(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def extended_folds(folds_csv: str, out_csv: str, data_dir=None,
                   pseudo: bool = False) -> str:
    """Frozen folds, plus a fold for every screen-only compound.

    `artifacts/folds.csv` covers the 4,905 compounds with a curve. The censored rows
    bring in ~750 more, and `split_frozen` would scatter those round-robin -- which
    puts close analogues of a held-out compound into the training half and quietly
    inflates every fold score. `assign_extra_to_folds` instead places each one in the
    fold of its nearest labelled neighbour by Tanimoto, the same rule the classical
    ensemble already uses, so the two remain comparable.
    """
    if os.path.exists(out_csv):
        return out_csv
    folds = splits.load_folds(folds_csv)
    train = io.load("train_inhibition", data_dir)
    screen = io.load("single_conc", data_dir)
    lm = L.build_label_matrix(train, screen, include_censored=True, pseudo=pseudo)
    smi = dict(zip(lm[C.ID_COL], lm[C.SMILES_COL]))
    extra = sorted(set(lm[C.ID_COL]) - set(folds[C.ID_COL]))
    if not extra:
        return folds_csv
    print(f"  assigning {len(extra)} screen-only compounds to folds "
          f"by nearest labelled neighbour")
    assigned = splits.assign_extra_to_folds([smi[i] for i in extra], folds)
    add = pd.DataFrame({C.ID_COL: extra, C.SMILES_COL: [smi[i] for i in extra],
                        "fold": assigned})
    out = pd.concat([folds[[C.ID_COL, C.SMILES_COL, "fold"]], add], ignore_index=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv} ({len(out)} compounds)")
    return out_csv


def build_endpoint_table(isoform: str, censored_weight: float, std, data_dir=None,
                         interval_power: float = 1.0, pseudo: bool = False,
                         pseudo_band: float = 1.0, pseudo_weight: float = 0.5,
                         pseudo_nonhit_interval: bool = False,
                         cliff_weight: float = 0.0, cliff_sim: float = 0.45,
                         cliff_delta: float = 1.0, pseudo_fold_of: dict | None = None):
    """Rows for one isoform: fitted curves as points, screen non-hits as `< floor`.

    Returns (smiles, y, lt, weight, names). `lt[i]` True means y[i] is an UPPER bound:
    the true value is somewhere below it and the loss must not push the prediction up
    to meet it.
    """
    train = io.load("train_inhibition", data_dir)
    screen = io.load("single_conc", data_dir)
    # pseudo_fold_of makes the log2fc -> pIC50 map leave-one-fold-out, so a compound's
    # own curve never shapes the map that labels it. Without it every out-of-fold score
    # built on pseudo-labels is optimistic by 5-16% of the map's own noise (Section 33).
    lm = L.build_label_matrix(train, screen, include_censored=True,
                              censored_weight=censored_weight, pseudo=pseudo,
                              pseudo_band=pseudo_band, pseudo_weight=pseudo_weight,
                              pseudo_nonhit_interval=pseudo_nonhit_interval,
                              pseudo_fold_of=pseudo_fold_of)
    sub = lm[lm["isoform"] == isoform].copy()

    # `kind` is the authority on censoring, not the presence of a y value.
    #
    # CORRECTED 2026-09-07. This comment used to say that a curve fitting at or below
    # the floor must NOT be treated as a point measurement. That was wrong, and it was
    # the justification for a real bug: 1,564 of 6,525 fitted curves (CYP3A4: 944,
    # 40.4%) were being discarded as left-censored even though every one carries finite
    # _conf_low/_conf_high, and `scoring.build_ground_truth` charges ST-RAE against
    # exactly those bounds. Training said "free below 4.0"; the metric charged.
    # `cypkit/labels.py` now keeps a fitted curve wherever it sits, so `kind == "left"`
    # here means genuinely no curve — a screen non-hit. Measured effect of the fix,
    # same features/folds/model on identical eval rows: +0.079 macro Spearman.
    lt = (sub["kind"].to_numpy() == "left")
    y = sub["y"].to_numpy(float)
    hi = sub["y_high"].to_numpy(float)
    # for a censored row the informative number is its UPPER BOUND (~the 4.0 floor);
    # for a fitted row it is the point estimate
    y = np.where(lt, np.where(np.isfinite(hi), hi, C.PIC50_FLOOR),
                 np.where(np.isfinite(y), y, np.where(np.isfinite(hi), hi,
                                                      C.PIC50_FLOOR)))
    w = sub["weight"].to_numpy(float) if "weight" in sub else np.ones(len(sub))
    # Charge the model where the METRIC charges. ST-RAE is zero inside a compound's
    # credible interval, so a narrow interval is nearly a point target and a wide one
    # is almost free. On our submitted blend the narrowest quartile of intervals
    # carried 38-60% of the total ST-RAE cost while being 25% of the rows. Weighting
    # by 1/width moved macro OOF ST-RAE 0.4576 -> 0.4489 on the classical learners.
    # Censored rows have no interval and keep their weight unchanged.
    from cypkit.ensemble import interval_weights  # noqa: PLC0415
    w = w * interval_weights(sub, power=interval_power)
    print(f"    interval weighting power {interval_power}: multiplier range "
          f"[{interval_weights(sub, power=interval_power).min():.2f}, "
          f"{interval_weights(sub, power=interval_power).max():.2f}]")

    smi = [_clean(std, s) for s in sub[C.SMILES_COL]]
    keep = [i for i, s in enumerate(smi) if isinstance(s, str) and Chem.MolFromSmiles(s)]
    idx = np.array(keep, int)
    src = sub["source"].to_numpy()[idx] if "source" in sub else np.array(["drc"]*len(idx))
    print(f"  {isoform}: {len(idx)} rows = {int((~lt[idx]).sum())} fitted "
          f"+ {int(lt[idx].sum())} censored (weight {censored_weight})"
          + (f", of which {int((src=='pseudo').sum())} screen-derived pseudo-labels"
             if (src == 'pseudo').any() else ""))
    smi_keep, y_keep, lt_keep, w_keep = ([smi[i] for i in idx], y[idx], lt[idx], w[idx])

    # Cliff weighting goes LAST, on the rows that survive cleaning, so the neighbour
    # graph is built over exactly the molecules that will be trained on. Measured
    # defect it targets: this model reproduces only 20-46% of the activity gap across
    # similar pairs (scripts/activity_cliffs.py), and out-of-fold MAE on cliff
    # compounds runs 1.20-1.76x the non-cliff rate. See cypkit/cliffs.py.
    if cliff_weight > 0:
        from cypkit import cliffs as CL  # noqa: PLC0415
        w_keep = w_keep * CL.weights(smi_keep, y_keep, lt_keep, alpha=cliff_weight,
                                     sim=cliff_sim, delta=cliff_delta)

    return (smi_keep, y_keep, lt_keep, w_keep, sub[C.ID_COL].to_numpy()[idx])


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# RUN RECORD. Every fold, every endpoint, every run -- appended to one CSV so two
# experiments can be compared months apart without re-reading a terminal scrollback.
# The whole point of Sections 30-39 is that a number is worthless without the yardstick
# it was measured on, so each row carries the config that produced it: lean descriptors
# or full, pseudo-labels on or off, multitask head on or off, folds, patience.
# ---------------------------------------------------------------------------
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
RUN_ROWS: List[dict] = []
RUN_CONFIG: dict = {}


def record(scope, isoform, m, **extra):
    RUN_ROWS.append(dict(run_id=RUN_ID, isoform=isoform, scope=scope,
                         n=m["n"], st_rae=m["st_rae"], mae=m["mae"], r2=m["r2"],
                         rho=m["rho"], tau=m["tau"], **extra))


def flush_run_summary(model_dir):
    """One JSON for this run, plus an append to the cumulative CSV.

    WRAPPED IN try/except ON PURPOSE. This runs AFTER the OOF and prediction files are
    written, so a bug here destroys nothing -- but on 2026-09-27 it still ended a full
    GPU run with a traceback instead of a summary, which reads like a failed run and
    invites re-running it. A bookkeeping convenience must never be able to do that.
    """
    try:
        _flush_run_summary(model_dir)
    except Exception as e:                                          # noqa: BLE001
        print(f"\n  run summary could not be written ({type(e).__name__}: {e})")
        print("  THE MODEL OUTPUTS ARE UNAFFECTED -- they were written before this.")


def _flush_run_summary(model_dir):
    if not RUN_ROWS:
        return
    df = pd.DataFrame(RUN_ROWS)
    for k, v in RUN_CONFIG.items():
        # argparse hands back LISTS for nargs arguments -- args.isoforms is 4 items --
        # and pandas tries to broadcast a 4-element list across 28 rows and raises.
        # Anything that is not a scalar becomes its string form; the JSON keeps the
        # real structure.
        df[k] = v if (v is None or np.isscalar(v)) else str(v)
    os.makedirs("artifacts", exist_ok=True)
    cum = os.path.join("artifacts", "training_runs.csv")
    # NEVER plain-append. The config columns come from argparse, so the moment a new flag
    # is added -- --descriptor_spec was added the same day this was written -- the next
    # run writes a row with one extra field and the WHOLE cumulative file stops parsing:
    # "Expected 16 fields in line 30, saw 17". Read, concat on the union of columns, and
    # rewrite. At tens of rows per run the cost is nothing and the history stays readable.
    # `all_runs` is what gets WRITTEN; `df` stays THIS run's rows. Conflating the two
    # made the printed forecast average over every previous run -- "macro over 12
    # endpoints" on a four-endpoint run, with a number to match.
    all_runs = df
    if os.path.exists(cum):
        try:
            all_runs = pd.concat([pd.read_csv(cum), df], ignore_index=True, sort=False)
        except Exception as e:                                      # noqa: BLE001
            bak = f"{cum}.{RUN_ID}.bak"
            os.replace(cum, bak)
            print(f"  existing {cum} was unreadable ({type(e).__name__}); "
                  f"kept as {bak} and starting a new one")
    all_runs.to_csv(cum, index=False)
    os.makedirs(model_dir, exist_ok=True)
    jp = os.path.join(model_dir, "run_summary.json")
    with open(jp, "w") as fh:
        json.dump({"run_id": RUN_ID, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "config": RUN_CONFIG, "rows": RUN_ROWS}, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"RUN SUMMARY  {RUN_ID}")
    cfg = "  ".join(f"{k}={v}" for k, v in RUN_CONFIG.items()
                    if k in ("pseudo", "lean_descriptors", "multitask_clf",
                             "clf_weight", "patience", "folds_csv"))
    print(f"  {cfg}\n")
    piv = df[df.scope.isin(("oof", "blindlike"))].pivot_table(
        index="isoform", columns="scope", values=["st_rae", "rho"])
    print(piv.round(4).to_string())
    bl = df[df.scope == "blindlike"]
    assert len(bl) <= 4, (f"blind-forecast rows = {len(bl)}; this run has at most four "
                          f"endpoints, so the frame is not this run's")
    if len(bl):
        print(f"\n  BLIND FORECAST (macro over {len(bl)} endpoints): "
              f"ST-RAE {bl.st_rae.mean():.4f}   rho {bl.rho.mean():.4f}")
        print("  Compare against the board: pIC50 0.4833 / rho 0.7590 on the full blind "
              "set.\n  These are not the same compounds, so read the DIRECTION of a "
              "change between runs,\n  not the absolute level (Section 39).")
    print(f"\n  appended to {cum}")
    print(f"  wrote {jp}")
    print("=" * 78)


def board_metrics(y, pred, lo=None, hi=None, lt=None):
    """The five columns the leaderboard actually publishes, on whatever rows we have.

    MA-ST-RAE is the scored metric and it needs the credible INTERVALS, not just the
    point label: cost = clip(p - hi, 0) + clip(lo - p, 0), divided by the same cost for
    a constant prediction of mean(y). A prediction inside its interval is free, which is
    why MAE alone has never told us what the board would say -- Section 35 measured a
    rival matching our MAE and beating our ST-RAE, and the reverse.

    Left-censored rows (screen non-hits) are excluded from every statistic here: their
    label is a bound, so an "error" against it is not an error. `lt` marks them.
    """
    from scipy.stats import kendalltau, spearmanr                     # noqa: PLC0415
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    if lt is not None:
        ok &= ~np.asarray(lt, bool)
    y, pred = y[ok], pred[ok]
    out = {"n": int(ok.sum()), "st_rae": np.nan, "mae": np.nan, "r2": np.nan,
           "rho": np.nan, "tau": np.nan}
    if len(y) < 5:
        return out
    out["mae"] = float(np.mean(np.abs(y - pred)))
    ss = float(np.sum((y - y.mean()) ** 2))
    out["r2"] = float(1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else np.nan
    out["rho"] = float(spearmanr(y, pred).statistic)
    out["tau"] = float(kendalltau(y, pred).statistic)
    if lo is not None and hi is not None:
        lo = np.asarray(lo, float)[ok]; hi = np.asarray(hi, float)[ok]
        cost = lambda q: (np.where(np.isfinite(hi), np.clip(q - hi, 0, None), 0.0)
                          + np.where(np.isfinite(lo), np.clip(lo - q, 0, None), 0.0))
        base = float(np.sum(cost(np.full(len(y), y.mean()))))
        if base > 0:
            out["st_rae"] = float(np.sum(cost(pred)) / base)
    return out


def fmt_metrics(tag, m, extra=""):
    f = lambda v: "  --  " if not np.isfinite(v) else f"{v:6.4f}"
    return (f"  {tag:22s} n={m['n']:5d}  ST-RAE {f(m['st_rae'])}  MAE {f(m['mae'])}  "
            f"R2 {f(m['r2'])}  rho {f(m['rho'])}  tau {f(m['tau'])}{extra}")


class CYPMultitaskMPNN(pl.LightningModule):
    """The PXR v4_13d head, ported onto our dead-zone objective.

    WHAT WE ALREADY HAD, AND WHAT THIS ADDS. SingleTargetMPNN is already the pp50
    architecture from the PXR pipeline -- a per-target D-MPNN with target-specific
    descriptors concatenated to the aggregated embedding before the readout FFN. That
    model carried 75% of the final PXR blend. The one thing v4_13d adds on top is a
    SECOND HEAD with an ASYMMETRIC classification loss, and it is the piece worth porting:

        total = (1 - clf_weight) * dead_zone_reg_loss
              +      clf_weight  * BCE(clf_logit, 0) OVER INACTIVE ROWS ONLY

    Standard BCE punishes false negatives too, which drags every active embedding down
    and compresses the potency ceiling. Computing the loss only on inactives leaves
    genuine actives with zero classification gradient, so the trunk learns exactly one
    thing: DO NOT ENCODE AN INACTIVE THE WAY YOU ENCODE AN ACTIVE.

    WHY IT SHOULD MATTER MORE HERE THAN IT DID FOR PXR. Our dose-response curves were run
    only on primary-screen HITS, so training is enriched for actives while the blind set
    is the unfiltered library. The measured train -> blind mean shift is +0.64 / -0.23 /
    +1.86 / -0.66 on CYP1A2 / CYP2C9 / CYP2D6 / CYP3A4. Inactive over-prediction is not a
    side effect here, it is the central distribution problem, and the affine placement
    machinery exists to paper over it after the fact. This attacks it in the model.

    AND OUR INACTIVE LABEL IS BETTER THAN PXR'S. PXR had to threshold pEC50 < 4.0. We have
    thousands of rows that are MEASURED inactives -- the left-censored screen non-hits,
    already carried through the pipeline as lt_mask. A row counts as inactive here if it
    is left-censored OR its label sits at or below act_thresh, so the supervision is a
    real experimental result rather than a cut through the label distribution.

    At inference only the regression head is used, exactly as in v4_13d.
    """

    def __init__(self, mp, agg, hidden_dim, n_layers, dropout, input_extra,
                 dead_zone=0.0, clf_weight=0.2, act_thresh_scaled=0.0, lr=1e-4):
        super().__init__()
        self.mp, self.agg = mp, agg
        self.dead_zone = float(dead_zone)
        self.clf_weight = float(clf_weight)
        self.act_thresh = float(act_thresh_scaled)
        self._lr = float(lr)
        in_dim = mp.output_dim + int(input_extra)
        layers, prev = [], in_dim
        for _ in range(max(n_layers - 1, 0)):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden_dim
        self.trunk = nn.Sequential(*layers)
        out_dim = hidden_dim if n_layers > 1 else in_dim
        self.reg_head = nn.Linear(out_dim, 1)
        self.clf_head = nn.Linear(out_dim, 1)

    def _encode(self, bmg, V_d, X_d):
        H = self.agg(self.mp(bmg, V_d), bmg.batch)
        return H if X_d is None else torch.cat([H, X_d], dim=-1)

    def forward(self, bmg, V_d=None, X_d=None):
        return self.reg_head(self.trunk(self._encode(bmg, V_d, X_d)))

    @staticmethod
    def _unpack(batch):
        """chemprop's TrainingBatch, defensively. lt_mask is position 5 in 2.x; if a
        future version reorders it, a missing mask degrades to 'threshold only' rather
        than crashing or silently masking the wrong rows."""
        bmg, V_d, X_d, targets, weights = (batch[0], batch[1], batch[2],
                                           batch[3], batch[4])
        lt = batch[5] if len(batch) > 5 else None
        return bmg, V_d, X_d, targets, weights, lt

    def _reg_loss(self, pred, y, w, lt):
        """The SAME dead-zone objective DeadZoneSpreadMSE applies, so switching heads
        does not quietly switch loss functions as well."""
        r = pred - y
        if lt is not None:
            r = torch.where(lt & (r < 0), torch.zeros_like(r), r)
        r = torch.clamp(r.abs() - self.dead_zone, min=0.0)
        return (w * r * r).mean()

    def training_step(self, batch, batch_idx):
        bmg, V_d, X_d, targets, weights, lt = self._unpack(batch)
        shared = self.trunk(self._encode(bmg, V_d, X_d))
        y = targets.squeeze(-1)
        w = weights.squeeze(-1) if weights is not None else torch.ones_like(y)
        ltm = lt.squeeze(-1).bool() if lt is not None else None
        reg = self._reg_loss(self.reg_head(shared).squeeze(-1), y, w, ltm)

        # asymmetric: only inactive rows contribute, and their target is always 0
        inactive = (y <= self.act_thresh)
        if ltm is not None:
            inactive = inactive | ltm
        if inactive.any():
            lg = self.clf_head(shared).squeeze(-1)[inactive]
            clf = F.binary_cross_entropy_with_logits(
                lg, torch.zeros_like(lg), weight=w[inactive])
        else:
            clf = torch.zeros((), device=y.device)
        loss = (1.0 - self.clf_weight) * reg + self.clf_weight * clf
        self.log_dict({"train_loss": loss, "reg_loss": reg, "clf_loss": clf},
                      prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # REGRESSION ONLY. Early stopping must watch the quantity we are scored on, not
        # a joint objective the auxiliary head can improve on its own.
        bmg, V_d, X_d, targets, weights, lt = self._unpack(batch)
        y = targets.squeeze(-1)
        w = weights.squeeze(-1) if weights is not None else torch.ones_like(y)
        ltm = lt.squeeze(-1).bool() if lt is not None else None
        pred = self.reg_head(self.trunk(self._encode(bmg, V_d, X_d))).squeeze(-1)
        vl = self._reg_loss(pred, y, w, ltm)
        self.log("val_loss", vl, prog_bar=True, on_step=False, on_epoch=True)
        return vl

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        bmg, V_d, X_d = batch[0], batch[1], batch[2]
        return self.forward(bmg, V_d, X_d)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)


class SingleTargetMPNN:
    """One ChemProp model for one isoform, descriptors fused at the readout."""

    def __init__(self, isoform, model_dir, use_descriptors=True, use_censored=True,
                 dead_zone=0.0, spread_lambda=0.0, init_from=None,
                 lean_descriptors=False, multitask_clf=False, clf_weight=0.2,
                 act_thresh=None, descriptor_spec=None):
        self.isoform = isoform
        self.model_dir = model_dir
        self.use_descriptors = use_descriptors
        self.lean_descriptors = bool(lean_descriptors)
        self.descriptor_spec = descriptor_spec
        self.multitask_clf = bool(multitask_clf)
        self.clf_weight = float(clf_weight)
        # act_thresh defaults to the assay floor: below it the screen could not quantify,
        # which is the same line the left-censored rows sit on.
        self.act_thresh = (C.PIC50_FLOOR if act_thresh is None else float(act_thresh))
        self.use_censored = use_censored
        self.dead_zone = float(dead_zone)
        self.spread_lambda = float(spread_lambda)
        self.init_from = init_from
        os.makedirs(model_dir, exist_ok=True)
        self.scaler_y = RobustScaler(quantile_range=(1.0, 99.0))
        self.desc_calc: Optional[TargetDescriptorCalculator] = None
        self.augmenter = SMILESAugmenter(random_seed=GLOBAL_SEED)
        self.best_epochs: List[int] = []
        self._bnd = None

    def _build(self, cfg, n_desc, fold=None):
        hd = cfg["hidden_dim"]
        mp = BondMessagePassing(d_h=hd, depth=cfg["depth"], dropout=cfg["dropout"],
                                activation="relu")
        # Optional warm start from the screen-pretrained trunk. Fold-matched on purpose:
        # trunk k never saw fold k's compounds, so initialising fold k's model from any
        # other trunk would leak a 0.9-correlated proxy of the label into the held-out
        # rows. See scripts/pretrain_screen.py.
        if self.init_from and fold is not None:
            tp = os.path.join(self.init_from, f"screen_trunk_fold{fold}.pt")
            if not os.path.exists(tp):
                raise SystemExit(f"--init_from given but {tp} is missing; run "
                                 "scripts/pretrain_screen.py first")
            ck = torch.load(tp, map_location="cpu", weights_only=False)
            if ck["hidden_dim"] != hd or ck["depth"] != cfg["depth"]:
                raise SystemExit(
                    f"trunk was pretrained at hidden={ck['hidden_dim']} depth="
                    f"{ck['depth']} but this endpoint wants hidden={hd} depth="
                    f"{cfg['depth']}; re-run pretrain_screen.py or fix the tier")
            mp.load_state_dict(ck["message_passing"])
            print(f"    warm start: trunk fold {fold} loaded from {tp}")
        # BoundedMSE is what makes a censored row cost nothing while the prediction
        # stays on the correct side of the bound. Passing it as the criterion AND as
        # the metric keeps early stopping on the same objective as training.
        ffn = RegressionFFN(input_dim=hd + n_desc, hidden_dim=hd,
                            n_layers=cfg["n_layers"], dropout=cfg["dropout"],
                            n_tasks=1,
                            criterion=DeadZoneSpreadMSE(self.dead_zone,
                                                        self.spread_lambda))
        # early stopping watches the SAME objective, dead zone included, but with the
        # spread term off -- a validation score should measure fit, not the regulariser.
        if self.multitask_clf:
            # act_thresh has to live in SCALED space: scaler_y is already fitted by the
            # time _build runs, and comparing a raw pIC50 against scaled targets would
            # mark either every row or no row as inactive.
            ats = float(self.scaler_y.transform(
                np.array([[self.act_thresh]], dtype=float))[0, 0])
            print(f"    multitask head ON: clf_weight={self.clf_weight}, "
                  f"act_thresh={self.act_thresh} (scaled {ats:+.3f})")
            self._model_cfg = dict(hidden_dim=hd, depth=cfg["depth"],
                                   n_layers=cfg["n_layers"], dropout=cfg["dropout"],
                                   input_extra=int(n_desc), dead_zone=self.dead_zone,
                                   clf_weight=self.clf_weight, act_thresh_scaled=ats)
            with open(os.path.join(self.model_dir,
                                   f"{self.isoform}_model_config.json"), "w") as fh:
                json.dump(self._model_cfg, fh, indent=2)
            return CYPMultitaskMPNN(mp=mp, agg=MeanAggregation(), **{
                k: v for k, v in self._model_cfg.items() if k != "depth"})
        return MPNN(message_passing=mp, agg=MeanAggregation(), predictor=ffn,
                    batch_norm=True,
                    metrics=[DeadZoneSpreadMSE(self.dead_zone, 0.0)])

    def _safe_load(self, path):
        # A CYPMultitaskMPNN cannot be rebuilt by MPNN.load_from_checkpoint -- it is not
        # an MPNN. Its architecture was written beside the checkpoints at train time for
        # exactly this reason; the same lesson as the descriptor meta file.
        cfgp = os.path.join(self.model_dir, f"{self.isoform}_model_config.json")
        if os.path.exists(cfgp):
            with open(cfgp) as fh:
                mc = json.load(fh)
            mdl = CYPMultitaskMPNN(
                mp=BondMessagePassing(d_h=mc["hidden_dim"], depth=mc["depth"],
                                      dropout=mc["dropout"], activation="relu"),
                agg=MeanAggregation(),
                **{k: v for k, v in mc.items() if k != "depth"})
            for kw in ({"weights_only": False}, {}):
                try:
                    ck = torch.load(path, map_location=DEVICE, **kw)
                    mdl.load_state_dict(ck["state_dict"])
                    return mdl
                except Exception:
                    continue
            return None
        for kw in ({"weights_only": False}, {}):
            try:
                return MPNN.load_from_checkpoint(path, map_location=DEVICE, **kw)
            except Exception:
                continue
        return None

    def _bounds(self, names):
        """y_low / y_high for these compounds, cached. ST-RAE is undefined without them."""
        if getattr(self, "_bnd", None) is None:
            lm = L.build_label_matrix(io.load("train_inhibition"), io.load("single_conc"),
                                      include_censored=True)
            sub = lm[lm["isoform"] == self.isoform]
            self._bnd = (dict(zip(sub[C.ID_COL], sub["y_low"])),
                         dict(zip(sub[C.ID_COL], sub["y_high"])))
        lod, hid = self._bnd
        return (np.array([lod.get(n, np.nan) for n in names], float),
                np.array([hid.get(n, np.nan) for n in names], float))

    def _dps(self, smiles, y, desc, lt=None, w=None):
        out = []
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            kw = {}
            if desc is not None:
                kw["x_d"] = desc[i].astype(np.float32)
            if lt is not None:
                kw["lt_mask"] = np.array([bool(lt[i])])
            if w is not None:
                kw["weight"] = float(w[i])
            out.append(MoleculeDatapoint(mol=mol,
                                         y=np.array([float(y[i])], np.float32), **kw))
        return out

    def _predict(self, model, loader, n):
        tr = pl.Trainer(accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
                        logger=False, enable_progress_bar=False)
        with torch.inference_mode():
            raw = tr.predict(model, loader)
        return np.concatenate([p.cpu().numpy() for p in raw]).ravel()[:n]

    def train(self, smiles, y, lt, w, names, folds_csv, patience=None):
        cfg = get_adaptive_config(len(smiles))
        if patience is not None:
            cfg["patience"] = patience
        print(f"\n{'='*70}")
        print(f"{self.isoform}  N={len(smiles)} "
              f"({int((~lt).sum())} fitted, {int(lt.sum())} censored)")
        print(f"  tier -> hidden={cfg['hidden_dim']} depth={cfg['depth']} "
              f"layers={cfg['n_layers']} aug={cfg['augmentation_factor']}x")
        print(f"  descriptors: {'ON' if self.use_descriptors else 'OFF (ablation)'}   "
              f"censored rows: {'ON' if self.use_censored else 'OFF (v2 behaviour)'}")
        print(f"{'='*70}")

        # The scaler must be fitted on the FITTED values only. Including a pile of
        # identical floor values would drag the median down and squash the scale --
        # which is exactly the collapse v2 showed on CYP2D6.
        self.scaler_y.fit(np.asarray(y, float)[~lt].reshape(-1, 1))
        y_sc = self.scaler_y.transform(np.asarray(y, float).reshape(-1, 1)).ravel()

        X_desc, n_desc = None, 0
        if self.use_descriptors:
            self.desc_calc = TargetDescriptorCalculator(self.isoform,
                                                        lean=self.lean_descriptors,
                                                        spec=self.descriptor_spec)
            print(f"  descriptor block: {self.desc_calc.n_features()} columns "
                  f"({'LEAN, Section 36' if self.lean_descriptors else 'full'})")
            X_desc = self.desc_calc.fit_transform(smiles)
            n_desc = X_desc.shape[1]
            print(f"  descriptor block: {X_desc.shape}")

        Y2 = np.asarray(y, float).reshape(-1, 1)
        folds = get_folds("frozen" if folds_csv else "activity_scaffold", smiles,
                          Y=Y2, desc_matrix=X_desc, names=names, folds_csv=folds_csv,
                          n_folds=N_FOLDS, seed=GLOBAL_SEED)

        oof = np.full(len(smiles), np.nan)
        for fi, val_idx in enumerate(folds):
            set_global_seed(GLOBAL_SEED + fi)
            tr_mask = np.ones(len(smiles), bool); tr_mask[val_idx] = False
            tr_i = np.where(tr_mask)[0]
            tr_smi = [smiles[i] for i in tr_i]
            vl_smi = [smiles[i] for i in val_idx]

            # augmentation has to carry lt and weight along with the label, or the
            # censored rows silently become point measurements at the floor.
            # augment_dataset has no index channel, but it DOES pass `weights`
            # through unchanged -- so send the original row index down it and read
            # lt/weight back off that. Exact, and no assumption about how many
            # variants each molecule produced.
            if cfg["use_augmentation"] and cfg["augmentation_factor"] > 1:
                aug_smi, aug_y, back = self.augmenter.augment_dataset(
                    tr_smi, y_sc[tr_i].reshape(-1, 1), cfg["augmentation_factor"],
                    weights=np.arange(len(tr_i)))
                aug_y = np.asarray(aug_y).ravel()
                back = np.asarray(back, int)
                aug_lt = lt[tr_i][back]
                aug_w = w[tr_i][back]
            else:
                aug_smi, aug_y = tr_smi, y_sc[tr_i]
                aug_lt, aug_w = lt[tr_i], w[tr_i]

            aug_d = self.desc_calc.transform(aug_smi) if self.use_descriptors else None
            vl_d = X_desc[val_idx] if self.use_descriptors else None

            tl = build_dataloader(MoleculeDataset(
                self._dps(aug_smi, aug_y, aug_d, aug_lt, aug_w)),
                batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
            vl = build_dataloader(MoleculeDataset(
                self._dps(vl_smi, y_sc[val_idx], vl_d, lt[val_idx], w[val_idx])),
                batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

            ckpt_fn = f"{self.isoform}_fold_{fi}"
            ckpt_path = os.path.join(self.model_dir, f"{ckpt_fn}.ckpt")
            if os.path.exists(ckpt_path):
                print(f"  fold {fi+1}: checkpoint found, skipping training")
                best = self._safe_load(ckpt_path); self.best_epochs.append(-1)
            else:
                model = self._build(cfg, n_desc, fold=fi)
                cb = ModelCheckpoint(dirpath=self.model_dir, filename=ckpt_fn,
                                     monitor="val_loss", mode="min", save_top_k=1)
                trainer = pl.Trainer(
                    accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
                    max_epochs=cfg["max_epochs"],
                    callbacks=[EarlyStopping(monitor="val_loss",
                                             patience=cfg["patience"], mode="min"), cb],
                    enable_progress_bar=True, logger=False, gradient_clip_val=1.0)
                trainer.fit(model, tl, vl)
                self.best_epochs.append(max(0, trainer.current_epoch - cfg["patience"]))
                best = self._safe_load(cb.best_model_path) if cb.best_model_path else None
            if best is None:
                print(f"  fold {fi+1}: WARNING no usable checkpoint"); continue
            best.eval(); best.to(torch.device(DEVICE))
            keep = [i for i, s in enumerate(vl_smi) if Chem.MolFromSmiles(s)]
            p = self._predict(best, vl, len(keep))
            for k2, li in enumerate(keep):
                oof[val_idx[li]] = p[k2]

            got = self.scaler_y.inverse_transform(oof[val_idx].reshape(-1, 1)).ravel()
            m = np.isfinite(got)
            yv = np.asarray(y)[val_idx]; lv = np.asarray(lt)[val_idx]
            over = np.mean(got[m & lv] > yv[m & lv]) if (m & lv).any() else np.nan
            bl = self._bounds(np.asarray(names)[val_idx])
            fm = board_metrics(yv, got, bl[0], bl[1], lv)
            print(fmt_metrics(f"fold {fi+1}/{len(folds)}", fm,
                              f"  censored over bound {100*over:.0f}%"))
            record(f"fold{fi+1}", self.isoform, fm, censored_over=float(over))

        pred = self.scaler_y.inverse_transform(oof.reshape(-1, 1)).ravel()
        nm = np.asarray(names); ltv = np.asarray(lt)
        lo_a, hi_a = self._bounds(nm)
        print()
        _oof = board_metrics(y, pred, lo_a, hi_a, ltv)
        print(fmt_metrics("OOF all folds", _oof)); record("oof", self.isoform, _oof)
        # Section 39: a fixed subset whose nearest-neighbour profile matches the blind
        # set's, so its score is a FORECAST of the board rather than another fold score.
        hp = os.path.join("artifacts", "holdout_blindlike.csv")
        if os.path.exists(hp):
            hold = set(pd.read_csv(hp)["Molecule_Name"])
            hm = np.array([n in hold for n in nm])
            if hm.sum() >= 20:
                _bl = board_metrics(np.asarray(y)[hm], pred[hm],
                                    lo_a[hm], hi_a[hm], ltv[hm])
                print(fmt_metrics("  blind-like subset", _bl, "   <- board forecast"))
                record("blindlike", self.isoform, _bl)
                print("  (these rows were in TRAINING for 4 of 5 folds, so this is an "
                      "out-of-fold\n   score on blind-like chemistry, not a true "
                      "holdout -- see Section 39)")
        print()
        save_pickle(self.scaler_y, os.path.join(self.model_dir, f"{self.isoform}_scaler_y.pkl"))
        save_pickle(self.best_epochs, os.path.join(self.model_dir, f"{self.isoform}_best_epochs.pkl"))
        if self.use_descriptors:
            save_pickle(self.desc_calc.scaler,
                        os.path.join(self.model_dir, f"{self.isoform}_desc_scaler.pkl"))
            # A RobustScaler fitted on a numpy array keeps no column names, so nothing in
            # the checkpoint recorded WHICH descriptors it saw. That is why a PREDICT run
            # died on 2026-09-26 when the module had grown underneath it. Write the names
            # and the lean flag beside it; predict() reads them back.
            save_pickle({"lean": self.lean_descriptors,
                         "spec": self.descriptor_spec,
                         "names": list(self.desc_calc.names)},
                        os.path.join(self.model_dir, f"{self.isoform}_desc_meta.pkl"))
        return pred

    def predict(self, smiles):
        self.scaler_y = load_pickle(os.path.join(self.model_dir, f"{self.isoform}_scaler_y.pkl"))
        n_desc, X = 0, None
        if self.use_descriptors:
            # prefer what TRAIN recorded over what the CLI happens to say this time
            meta_p = os.path.join(self.model_dir, f"{self.isoform}_desc_meta.pkl")
            lean = self.lean_descriptors
            if os.path.exists(meta_p):
                meta = load_pickle(meta_p)
                lean = bool(meta.get("lean", lean))
                spec = meta.get("spec", self.descriptor_spec)
                if spec != self.descriptor_spec:
                    print(f"  [{self.isoform}] checkpoints were trained with "
                          f"spec={spec}; using that, not --descriptor_spec")
                self.descriptor_spec = spec
                if lean != self.lean_descriptors:
                    print(f"  [{self.isoform}] checkpoints were trained with "
                          f"lean={lean}; using that, not the --lean_descriptors flag")
            else:
                print(f"  [{self.isoform}] no _desc_meta.pkl (checkpoints predate it); "
                      f"trusting --lean_descriptors={lean}")
            self.desc_calc = TargetDescriptorCalculator(self.isoform, lean=lean,
                                                        spec=self.descriptor_spec)
            self.desc_calc.scaler = load_pickle(
                os.path.join(self.model_dir, f"{self.isoform}_desc_scaler.pkl"))
            X = self.desc_calc.transform(smiles)
            n_desc = X.shape[1]
        keep = [i for i, s in enumerate(smiles) if Chem.MolFromSmiles(str(s))]
        ks = [smiles[i] for i in keep]
        kd = X[keep] if X is not None else None
        dl = build_dataloader(MoleculeDataset(
            self._dps(ks, np.zeros(len(ks)), kd)), batch_size=64, shuffle=False,
            num_workers=0)
        acc, n = np.zeros(len(ks)), 0
        for fi in range(N_FOLDS):
            path = os.path.join(self.model_dir, f"{self.isoform}_fold_{fi}.ckpt")
            if not os.path.exists(path):
                continue
            m = self._safe_load(path)
            if m is None:
                continue
            m.eval(); m.to(torch.device(DEVICE))
            acc += self._predict(m, dl, len(ks)); n += 1
        if n == 0:
            raise SystemExit(f"no checkpoints for {self.isoform} in {self.model_dir}")
        out = np.full(len(smiles), np.nan)
        out[keep] = self.scaler_y.inverse_transform((acc / n).reshape(-1, 1)).ravel()
        print(f"  {self.isoform}: averaged {n} folds")
        return out


def main():
    # Declared here, before argparse reads GLOBAL_SEED as the --seed default.
    # Python resolves `global` for the WHOLE function body, so putting it lower
    # (next to the assignment, where it reads better) is a SyntaxError: "name
    # 'GLOBAL_SEED' is used prior to global declaration". ast.parse does not
    # catch that -- only compile() does.
    global GLOBAL_SEED                                        # noqa: PLW0603
    p = argparse.ArgumentParser(
        description="per-target ChemProp with fused descriptors and censored rows")
    p.add_argument("--mode", choices=["TRAIN", "PREDICT", "BOTH"], default="BOTH")
    p.add_argument("--train_csv", default=TRAIN_CSV)
    p.add_argument("--test_csv", default=TEST_CSV)
    p.add_argument("--folds_csv", default="artifacts/folds.csv")
    p.add_argument("--model_dir", default="models_cyp_st_v3")
    p.add_argument("--output", default="predictions_cyp_st_v3.csv")
    p.add_argument("--oof_output", default="oof_chemprop_st_v3.csv")
    p.add_argument("--censored_weight", type=float, default=0.5,
                   help="relative weight of screen-derived censored rows; there are "
                        "many more of them than curves and 1.0 lets them dominate")
    p.add_argument("--interval_power", type=float, default=1.0,
                   help="training weight ~ (median_width/width)**power. ST-RAE is "
                        "zero inside the credible interval, so narrow intervals cost "
                        "far more; 0 restores equal weighting.")
    p.add_argument("--pseudo", action="store_true",
                   help="turn the primary screen's log2fc into pseudo-pIC50 labels via "
                        "a monotone map (see cypkit.labels.pseudo_from_screen). Adds "
                        "~2,900 CYP2D6 / ~3,100 CYP2C9 compounds that have a screen "
                        "reading but no curve, as INTERVALS whose width is the map's "
                        "residual sd. Measured on a LightGBM baseline: OOF Spearman "
                        "+0.029 / +0.068 / +0.022 / +0.022 on 1A2/2C9/2D6/3A4.")
    p.add_argument("--pseudo_band", type=float, default=1.0,
                   help="half-width of a pseudo-label's interval, in units of the map's "
                        "residual sd. Larger = the model is charged less for missing it.")
    p.add_argument("--pseudo_weight", type=float, default=0.5,
                   help="weight of a pseudo-labelled row relative to a fitted curve")
    p.add_argument("--pseudo_nonhit_interval", action="store_true",
                   help="also give screen NON-hits a two-sided interval instead of an "
                        "upper bound. This is what the LightGBM attribution test did; "
                        "it is more aggressive and invents sub-floor precision.")
    p.add_argument("--no_censored", action="store_true",
                   help="reproduce v2: fitted curves only")
    p.add_argument("--dead_zone", type=float, default=0.0,
                   help="ST-RAE charges nothing inside a compound's credible interval; "
                        "this subtracts that much from |residual| before squaring, in "
                        "SCALED units. 0.25 is roughly the median half-width after "
                        "RobustScaler. 0 = plain bounded MSE (v3 behaviour).")
    p.add_argument("--spread_lambda", type=float, default=0.0,
                   help="penalty on predicting NARROWER than the batch's own targets. "
                        "One-sided, so it can never inflate an already-wide predictor. "
                        "0.5 is a reasonable first try; 0 = off (v3 behaviour).")
    p.add_argument("--cliff_weight", type=float, default=0.0,
                   help="upweight compounds whose close analogues behave differently. "
                        "w = 1 + alpha*min(local_contrast/delta, 1), capped at 1+alpha. "
                        "This model reproduces only a fifth of the measured activity "
                        "gap on CYP2D6 cliff pairs and one pair in twenty; that is "
                        "invisible to MAE and Spearman because both members are wrong "
                        "in the SAME direction. Start at 1.0 and judge it with "
                        "scripts/activity_cliffs.py, not only with rho.")
    p.add_argument("--cliff_sim", type=float, default=0.45,
                   help="Tanimoto floor for 'a close analogue'. 0.45 is calibrated to "
                        "THIS library, whose median held-out neighbour sits at 0.44 -- "
                        "the textbook 0.75 finds five pairs in the whole training set.")
    p.add_argument("--cliff_delta", type=float, default=1.0,
                   help="log units of difference that count as a full cliff (the cap).")
    p.add_argument("--init_from", default=None,
                   help="directory of screen-pretrained trunks from "
                        "scripts/pretrain_screen.py. Fold-matched warm start.")
    p.add_argument("--endpoint_weights", default=None,
                   help="JSON from scripts/fit_endpoint_weights.py. Overrides "
                        "--interval_power and --censored_weight PER ENDPOINT, which is "
                        "the point: the blind set needs 73%% of CYP2D6's weight below "
                        "pIC50 4.0 and only 26%% of CYP2C9's. One global value cannot "
                        "be right for both.")
    p.add_argument("--no_descriptors", action="store_true")
    p.add_argument("--lean_descriptors", action="store_true",
                   help="use the Section 36 mechanism-justified subset (35-40%% fewer "
                        "columns; +0.0400 rho on CYP2C9, +0.0239 on CYP2D6, level "
                        "elsewhere against the full block). The setting is recorded in "
                        "the checkpoint directory and reused automatically at PREDICT.")
    p.add_argument("--multitask_clf", action="store_true",
                   help="add the PXR v4_13d asymmetric classification head: a second "
                        "output trained with BCE on INACTIVE rows only, so the trunk "
                        "learns not to encode inactives like actives without dragging "
                        "the potency ceiling down. Inference uses the regression head.")
    p.add_argument("--clf_weight", type=float, default=0.2,
                   help="weight of the classification term; total = (1-w)*dead-zone MSE "
                        "+ w*BCE. PXR used 0.2.")
    p.add_argument("--act_thresh", type=float, default=None,
                   help="pIC50 at or below which a row counts as inactive for the "
                        "classification head (default: the assay floor). Left-censored "
                        "screen non-hits always count, whatever this is set to.")
    p.add_argument("--descriptor_spec", default=None,
                   choices=["per_endpoint", "lean", "lean3d", "wide3d"],
                   help="Section 41 per-endpoint recipe. 'per_endpoint' gives CYP1A2 and "
                        "CYP2D6 lean+3D, CYP2C9 lean, CYP3A4 lean+wide+3D. The other "
                        "values force one recipe everywhere for an A/B. Overrides "
                        "--lean_descriptors; recorded in the checkpoint directory and "
                        "reused automatically at PREDICT.")
    p.add_argument("--isoforms", nargs="*", default=list(ISOFORMS))
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED,
                   help="Replaces the module-level GLOBAL_SEED for this run. Exposed so "
                        "scripts/seed_ensemble.py can average several runs: two runs of "
                        "this script with IDENTICAL settings correlate only r = 0.927, so "
                        "a single run carries roughly 7% pure seed noise and no feature "
                        "A/B smaller than that is decidable from one pair. Recorded in "
                        "the run summary; fold seeds remain seed + fold_index.")
    args = p.parse_args()
    GLOBAL_SEED = args.seed

    RUN_CONFIG.update({k: v for k, v in vars(args).items()})
    RUN_CONFIG["device"] = DEVICE
    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("  " + "!" * 68)
        print("  WARNING: no GPU visible. Check torch.cuda.is_available() in this env.")
    std = MoleculeStandardizer()

    folds_csv = args.folds_csv
    if not args.no_censored and args.mode in ("TRAIN", "BOTH"):
        # `--pseudo` brings in screen HITS that have no curve, which the censored-only
        # matrix never included, so it needs its own fold file rather than reusing one
        # built without them.
        folds_csv = extended_folds(
            args.folds_csv,
            "artifacts/folds_pseudo.csv" if args.pseudo
            else "artifacts/folds_censored.csv",
            pseudo=args.pseudo)

    # the same fold assignment the trainer uses, handed to the pseudo-label map so it can
    # be fitted leave-one-fold-out. It must cover the screen-only compounds too, which is
    # exactly what extended_folds just produced.
    pseudo_fold_of = None
    if args.pseudo:
        _f = pd.read_csv(folds_csv); _f.columns = [c.strip() for c in _f.columns]
        pseudo_fold_of = dict(zip(_f[NAME_COL], _f["fold"]))
        print(f"  pseudo-label map will be fitted leave-one-fold-out over "
              f"{len(set(pseudo_fold_of.values()))} folds, {len(pseudo_fold_of)} compounds")

    oof_frames, test_frames = {}, {}
    test = pd.read_csv(args.test_csv)
    test.columns = [c.strip() for c in test.columns]
    tsmi = [_clean(std, s) for s in test[SMILES_COL]]

    ep_cfg = {}
    if args.endpoint_weights:
        import json as _json  # noqa: PLC0415
        ep_cfg = _json.loads(Path(args.endpoint_weights).read_text())
        print(f"per-endpoint weights from {args.endpoint_weights}:")
        for k, v in ep_cfg.items():
            print(f"  {k}: interval_power={v['interval_power']} "
                  f"censored_weight={v['censored_weight']} "
                  f"(targets {v['target_share']:.0%} of weight below pIC50 4.0)")

    for iso in args.isoforms:
        cw = ep_cfg.get(iso, {}).get("censored_weight", args.censored_weight)
        ip = ep_cfg.get(iso, {}).get("interval_power", args.interval_power)
        mdl = SingleTargetMPNN(iso, args.model_dir,
                               use_descriptors=not args.no_descriptors,
                               use_censored=not args.no_censored,
                               dead_zone=args.dead_zone,
                               spread_lambda=args.spread_lambda,
                               init_from=args.init_from,
                               lean_descriptors=args.lean_descriptors,
                               multitask_clf=args.multitask_clf,
                               clf_weight=args.clf_weight,
                               act_thresh=args.act_thresh,
                               descriptor_spec=args.descriptor_spec)
        if args.mode in ("TRAIN", "BOTH"):
            if args.no_censored:
                df = load_training(args.train_csv, std)
                col = f"{iso}_pIC50_direct_inhibition"
                d = df.dropna(subset=[col])
                smi = d["SMILES_clean"].tolist(); yy = d[col].to_numpy(float)
                lt = np.zeros(len(yy), bool); ww = np.ones(len(yy))
                nm = d[NAME_COL].to_numpy()
            else:
                smi, yy, lt, ww, nm = build_endpoint_table(
                    iso, cw, std,
                    interval_power=ip, pseudo=args.pseudo,
                    pseudo_band=args.pseudo_band, pseudo_weight=args.pseudo_weight,
                    pseudo_nonhit_interval=args.pseudo_nonhit_interval,
                    cliff_weight=args.cliff_weight, cliff_sim=args.cliff_sim,
                    cliff_delta=args.cliff_delta,
                    pseudo_fold_of=pseudo_fold_of)
            pred = mdl.train(smi, yy, lt, ww, nm, folds_csv, args.patience)
            oof_frames[iso] = pd.DataFrame({
                NAME_COL: nm, f"{iso}_pIC50_direct_inhibition": pred})
        if args.mode in ("PREDICT", "BOTH"):
            test_frames[iso] = mdl.predict(tsmi)

    if oof_frames:
        out = None
        for iso, d in oof_frames.items():
            out = d if out is None else out.merge(d, on=NAME_COL, how="outer")
        out.to_csv(args.oof_output, index=False)
        print(f"\nOOF written: {args.oof_output}  ({len(out)} compounds)")
    if test_frames:
        d = pd.DataFrame({"SMILES": test[SMILES_COL], NAME_COL: test[NAME_COL]})
        for iso, v in test_frames.items():
            d[f"{iso}_pIC50_direct_inhibition"] = v
        d.to_csv(args.output, index=False)
        print(f"Predictions written: {args.output}")
        cols = [c for c in d.columns if c.endswith("_direct_inhibition")]
        print(d[cols].describe().T.to_string(float_format=lambda v: f"{v:.3f}"))
    flush_run_summary(args.model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
