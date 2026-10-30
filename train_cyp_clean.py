#!/usr/bin/env python
r"""A single, self-contained ChemProp model for the CYP challenge -- curve compounds only.

    python train_cyp_clean.py                          # all four endpoints, 10 folds
    python train_cyp_clean.py --isoforms CYP2D6        # one endpoint
    python train_cyp_clean.py --smoke                  # 2 folds, 3 epochs -- run this first

WHAT THIS IS, AND WHY IT EXISTS
-------------------------------
Everything needed is in this file: data loading, descriptors, the split, the model, the
loss, training, out-of-fold scoring, blind prediction and the leaderboard metrics. No
imports from the project's own packages. You can read it top to bottom and know exactly
what produced a number.

It exists because the production pipeline stopped being explainable. That path trains
eleven model variants, blends them per endpoint with a greedy search, maps the blend back
to pIC50 through an affine transform anchored on the blind set's own solved-back mean and
spread, splices columns from different submissions, and multiplies two endpoints' spreads
by constants fitted to leaderboard feedback. Each step was measured and each paid at the
time. Together they cannot be written up or defended as chemistry.

THREE DELIBERATE CHOICES, AND WHAT THEY COST
--------------------------------------------
1. ONLY DOSE-RESPONSE COMPOUNDS. 1,180 / 1,026 / 1,364 / 2,335. Two kinds of row are
   dropped: the ~2,900 per endpoint recorded as "no measurable activity" (no value at
   all), and the 232 / 259 / 129 / 0 whose fitted pIC50 falls BELOW the assay floor and
   is therefore an extrapolation past the lowest tested concentration rather than a
   fit landed below pIC50 4.0 and whose sub-4.0 ORDERING the model cannot learn. That
   second cut is a modelling decision, not an assay fact -- see the SUBFLOOR_BOUND block
   and --keep_subfloor; it is what made the first run report CYP1A2 R2 = 0.262 where the
   above-floor curves alone give -0.245.

   This is a real sacrifice, not a simplification. Those censored rows are 69-71% of the
   data on three endpoints, they are genuine measurements, and the production pipeline uses
   them with a one-sided loss. Dropping them means this model never learns what an inactive
   compound looks like, and it will over-predict them. It also means the training set is
   the primary-screen HITS -- an activity-enriched sample -- while the blind set is the
   unfiltered library. Expect the model to be optimistic and compressed at the bottom.
   That is the price of a model you can explain in one page.

2. ACTIVITY-SCAFFOLD SPLIT, 10 FOLDS. Compounds sharing a Bemis-Murcko scaffold always go
   to the same fold, so the model is never validated on a close analogue of something it
   trained on. Scaffold groups are then assigned to balance the ACTIVITY DISTRIBUTION as
   well as fold size -- a pure scaffold split can hand one fold most of the potent
   compounds, and a fold with no actives scores meaninglessly. Random k-fold on this data
   is optimistic for exactly the reason scaffold splitting exists.

3. INTERPRETABLE DESCRIPTORS ALONGSIDE THE LEARNED EMBEDDING. 27 columns, each of which
   names a physical requirement for binding or turnover in one sentence (see DESCRIPTORS
   below). They are concatenated to the D-MPNN's pooled vector before the readout.

   The ionisation block here is SMARTS-based rather than the fitted-pKa block the
   production code uses. That is a genuine simplification: a predicted pKa is more
   accurate than counting basic centres. It is done so this file stands alone, and it is
   the one place where this script is knowingly weaker than the pipeline it replaces.

WHAT IT REPORTS
---------------
One table per endpoint, one row per fold:

           ------------ held out -----------  -- trained on --
 fold     n   ST-RAE     MAE      R2     rho       MAE      R2    gap    epoch
------------------------------------------------------------------------------
    1   141   0.7928  0.6174  0.2963  0.5071    0.5450  0.4837  +13%   14/60 *
 ...
 pool  1412   0.7940  0.6280  0.2905  0.5156    0.5196  0.5345  +17%

Left block = the held-out compounds, the only honest numbers. Right block = the SAME
model re-scored on the rows it trained on; it is not a performance estimate, it is the
reference the gap is measured against. `epoch` is the kept checkpoint and the cap, and
`*` means early stopping fired.

    gap small, train R2 low      underfitting, or the task is at its information limit
    gap large                    overfitting -- memorising rows
    gap small, train R2 high     the fit is doing what it should

Then a VERDICT block with four lines -- fit, spread, ranking, and how it compares to
predicting one constant -- and a SUMMARY table across all endpoints at the end.

TWO CAVEATS THAT ARE EASY TO MISS.
  - Early stopping keeps the best-VALIDATION checkpoint, so the train score is read at a
    point already chosen to look good on validation. The gap is a FLOOR on how much the
    model could memorise, not the full amount -- which is why `epoch` is printed, and why
    a cut-short run is called underfit rather than balanced.
  - R2 below rho^2 shows the scaling is wrong but NOT which way; too wide and too narrow
    both lower R2. Only `slope` in the verdict block carries the direction.

CONSOLE NOISE. Lightning's banners, warnings and parameter table are suppressed, and so
is chemprop's progress bar -- four Trainers are built per fold, so those blocks otherwise
repeat ~160 times in a full run and bury the table. `--progress` restores the bar,
`--verbose` restores everything.

WHAT IT WRITES
--------------
    oof_<tag>.csv    out-of-fold predictions on the training compounds -- for validating
                     this model against others before anything is uploaded
    pred_<tag>.csv   the 750 blind compounds, in the blind file's own row order, with the
                     four endpoint columns the upload form expects

The second file is SUBMISSION-SHAPED: run all four endpoints and it can be renamed to
submission_<name>.csv and uploaded as is. The script checks that itself at the end -- rows,
row order, all four columns, no empty cells -- and says so.

WHAT IT IS NOT
--------------
Shape is not quality. There is no blending, no placement, no splice and no spread
multiplier: the numbers are the model's own. Two consequences, both deliberate:

  - ST-RAE pays the distance from a credible interval, so the column's mean and spread
    matter as much as its ranking, and nothing here maps them onto the blind set's.
  - dropping the censored rows means the model has never seen an inactive compound, so it
    will place inactive blind compounds too high and its low tail is compressed.

Validate against the standing best on the out-of-fold pool before uploading anything.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Keep the console readable
# --------------------------------------------------------------------------------------
_QUIETED = False


def quiet_lightning():
    """Silence Lightning's per-Trainer banners so the fold metrics are readable.

    A Trainer is built four times per fold here -- fit, then predict on validation, train
    and blind rows -- and each one reprints the same block: 'GPU available', 'TPU
    available', 'LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES', the litmodels and litlogger tips,
    and 'Loading `train_dataloader` to estimate number of stepping batches'. Over 10 folds
    and 4 endpoints that is ~160 repetitions of text that says nothing new, and it buries
    the numbers you are actually reading.

    All of them go through loggers under the `lightning.*` namespace, so raising that
    namespace to ERROR removes the lot. TIMING IS THE WHOLE TRICK: `lightning.pytorch`
    sets its own logger to INFO when it is IMPORTED, so a level set before that import is
    silently overwritten. This function therefore imports lightning itself, and must be
    called after it -- never earlier in the file.

    Errors and real exceptions are untouched: only INFO and WARNING are dropped.
    """
    global _QUIETED
    if _QUIETED:
        return
    import lightning.pytorch                                          # noqa: F401
    try:
        # Silences "You are using a CUDA device that has Tensor Cores ..." AND is the
        # faster setting on any Ampere card or newer (the RTX 3060 here included): TF32
        # matmuls, ~1.5-2x on the dense layers, at a precision loss far below the noise
        # in a pIC50 measurement.
        import torch
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
    except Exception:                                                 # noqa: BLE001
        pass
    for n in list(logging.root.manager.loggerDict) + ["lightning", "pytorch_lightning",
                                                      "chemprop"]:
        if n.split(".")[0] in ("lightning", "pytorch_lightning", "chemprop"):
            logging.getLogger(n).setLevel(logging.ERROR)
    for c in (UserWarning, DeprecationWarning, FutureWarning):
        warnings.filterwarnings("ignore", category=c)
    try:
        from lightning.pytorch.utilities.warnings import PossibleUserWarning
        warnings.filterwarnings("ignore", category=PossibleUserWarning)
    except Exception:                                                 # noqa: BLE001
        pass
    _QUIETED = True


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
TRAIN_CSV = "data/cyp-challenge-TRAIN_inhibition.csv"
TEST_CSV = "data/cyp-challenge-TEST-BLINDED.csv"
ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
ID_COL, SMILES_COL = "Molecule_Name", "SMILES"

# THE pIC50 4.0 BOUNDARY -- the single most consequential line in this file, and the one
# most easily misread. It is NOT an assay detection limit.
#
# WHERE 4.0 COMES FROM. The challenge blog uses pIC50 4.0 to define the TDI labels, and
# this project reused that constant as a left-censoring bound (cypkit/config.py:
# PIC50_FLOOR). cypkit/labels.py is explicit that it "was never validated as a
# quantification limit for the direct-inhibition curves, and the data says it is not
# one". Checked directly here: the pIC50 distribution runs smoothly through 4.0 on every
# endpoint (CYP1A2 has 19 values in (3.9, 4.0] and 20 in (4.0, 4.1]), no value repeats at
# 4.000, and all four endpoints bottom out near 1.91. There is no instrumental boundary
# at 4.0 anywhere in this data.
#
# WHAT THE SUB-4.0 ROWS ACTUALLY ARE. Genuine dose-response fits that landed low. All of
# them carry finite _conf_low/_conf_high, and ST-RAE charges against exactly those
# bounds. They are measurements, not extrapolations -- an earlier version of this comment
# claimed otherwise and was wrong.
#
# WHY DROP THEM AT ALL, THEN. Because the model cannot always RANK them. Measured on
# ChemProp v8 (cypkit/labels.py), Spearman on the recovered rows themselves:
#
#   endpoint   sub-4.0 rows          rho on those rows
#   CYP3A4        944 (40.4%)        0.542    clearly useful -- MAE 0.972 -> 0.738
#   CYP2C9        259 (20.2%)        0.393    real signal
#   CYP1A2        232 (16.4%)        0.158    weak
#   CYP2D6        129 ( 8.6%)       -0.041    pure noise
#
# So this is a MODELLING decision per endpoint, not a fact about the assay, and it
# mirrors cypkit/labels.py's own `keep_subfloor=("CYP3A4",)`. Dropping them on CYP1A2
# and CYP2D6 is well supported. On CYP2C9 it is arguable: rho 0.393 on those rows is
# close to this model's whole-endpoint OOF rho of 0.45. Measured here, dropping CYP2C9's
# 259 rows improved ST-RAE 1.0659 -> 0.8355 on identical scored rows -- but that is this
# setup, which trains on curves only, whereas the 0.393 was measured in a pipeline that
# also trains on the censored rows with a one-sided loss. Different regimes; use
# --keep_subfloor to re-test rather than trusting either number blind.
#
# The effect on R2 is real regardless of interpretation: the sub-4.0 rows widen the
# variance R2 divides by. The first run reported CYP1A2 R2 = 0.262 on 1412 rows and
# -0.245 on the 1180 above-floor rows.
SUBFLOOR_BOUND = 4.0
# endpoints whose sub-4.0 curves are KEPT, mirroring cypkit/labels.py
KEEP_SUBFLOOR = ("CYP3A4",)


def ep(iso):
    return f"{iso}_pIC50_direct_inhibition"


def subfloor_bound(iso, a):
    """The pIC50 below which this endpoint's fitted curves are DROPPED, or None to keep
    them all. None is not "this assay has no floor" -- it is "for this endpoint we judge
    the sub-floor ordering learnable". See the SUBFLOOR_BOUND block."""
    if a.keep_below_floor or iso in (a.keep_subfloor or ()):
        return None
    return SUBFLOOR_BOUND if a.floor is None else a.floor


# --------------------------------------------------------------------------------------
# DESCRIPTORS -- 27 columns, each with a one-sentence justification
# --------------------------------------------------------------------------------------
# Occupancy and shape: does it fit the pocket?
#   HeavyAtomCount     how much volume it asks for
#   FractionCSP3       flat versus puckered; CYP1A2's slot is narrow and planar
#   Kappa3             branching, independent of size
#   NumRotatableBonds  entropic cost of freezing into the site
# Recognition: will it partition in and hydrogen-bond?
#   MolLogP, TPSA, NumHDonors, NumHAcceptors, NumAromaticRings, RingCount
# Electronics: how easily is it oxidised, and where?
#   MaxPartialCharge, MinPartialCharge   the electron-poor and electron-rich extremes
#   MinEStateIndex, MaxAbsEStateIndex    the most reactive atom environments
RDKIT_DESC = [
    "HeavyAtomCount", "FractionCSP3", "Kappa3", "NumRotatableBonds",
    "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumAromaticRings", "RingCount",
    "MaxPartialCharge", "MinPartialCharge", "MinEStateIndex", "MaxAbsEStateIndex",
]

# Heme coordination: the classic type-II binders that ligate the iron directly, plus the
# common metabolically-relevant groups.
HEME_SMARTS = {
    "heme_aromatic_n": "[n;X2;!$(n[#6]=O)]",     # pyridine-type N, coordinates Fe
    "imidazole": "c1cnc[nH]1",
    "triazole": "c1ncnn1",
    "nitrile": "[NX1]#[CX2]",
    "amide": "[NX3][CX3]=[OX1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "ether": "[OX2;!$([OX2H]);!$(OC=O)]",
    "halogen": "[F,Cl,Br,I]",
}

# Ionisation at pH 7.4 -- the single biggest determinant of CYP2C9 (prefers anions, via
# Arg108) versus CYP2D6 (prefers cations, via Asp301) selectivity. Counted from SMARTS
# rather than predicted pKa: less accurate, but self-contained and readable.
IONISABLE_SMARTS = {
    "basic_centre": "[NX3;H2,H1,H0;!$(N[a]);!$(NC=[O,S,N]);!$(N[SX4](=O)=O);!$([N+])]",
    "strong_base": "[$([NX3][CX3]=[NX2]),$(NC(=N)N)]",      # amidine / guanidine
    "acid_centre": "[CX3](=O)[OX2H1]",
    "weak_acid": "[$(c1nnn[nH]1),$([SX4](=O)(=O)[NX3H])]",  # tetrazole / sulfonamide NH
}
DESC_NAMES = (RDKIT_DESC + [f"heme_{k}" for k in HEME_SMARTS]
              + list(IONISABLE_SMARTS) + ["net_charge_74"])
N_DESC = len(DESC_NAMES)


def compute_descriptors(smiles_list):
    """(n, 27) matrix. Unparseable molecules get NaN and are filtered by the caller."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")

    fns = [(n, getattr(Descriptors, n)) for n in RDKIT_DESC]
    heme = {k: Chem.MolFromSmarts(v) for k, v in HEME_SMARTS.items()}
    ion = {k: Chem.MolFromSmarts(v) for k, v in IONISABLE_SMARTS.items()}

    X = np.full((len(smiles_list), N_DESC), np.nan)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        row = []
        for _, f in fns:
            try:
                v = float(f(m))
            except Exception:                                        # noqa: BLE001
                v = np.nan
            row.append(v)
        row += [float(len(m.GetSubstructMatches(p))) for p in heme.values()]
        counts = {k: len(m.GetSubstructMatches(p)) for k, p in ion.items()}
        row += [float(counts[k]) for k in IONISABLE_SMARTS]
        # crude net charge at pH 7.4: basic centres protonate, acids deprotonate
        row.append(float(counts["basic_centre"] + counts["strong_base"]
                         - counts["acid_centre"] - counts["weak_acid"]))
        X[i] = row
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------------------
# ACTIVITY-SCAFFOLD SPLIT
# --------------------------------------------------------------------------------------
def bemis_murcko(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            out.append(f"__bad__{s}"); continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:                                            # noqa: BLE001
            sc = ""
        # acyclic molecules have an empty scaffold; give each its own group rather than
        # letting them all collapse into one enormous fold-dominating cluster
        out.append(sc if sc else f"__acyclic__{Chem.MolToSmiles(m)}")
    return out


def activity_scaffold_folds(smiles, y, n_folds=10, n_bins=4, seed=0):
    """Assign fold ids so that (a) a scaffold never spans folds and (b) each fold gets a
    similar ACTIVITY distribution, not merely a similar count.

    Why both. A pure scaffold split keeps analogues together, which is the point -- but it
    is free to put most of the potent compounds in one fold, and a fold whose held-out
    compounds are all inactive produces a Spearman that means nothing. Balancing only the
    count does not fix that. So groups are placed largest-first into whichever fold is
    furthest BELOW its target share of each activity bin.
    """
    y = np.asarray(y, float)
    scaff = bemis_murcko(smiles)
    bins = pd.qcut(y, n_bins, labels=False, duplicates="drop")
    n_bins = int(np.nanmax(bins)) + 1

    groups = {}
    for i, s in enumerate(scaff):
        groups.setdefault(s, []).append(i)

    rng = np.random.default_rng(seed)
    order = sorted(groups.values(), key=lambda g: (-len(g), rng.random()))

    target = np.zeros((n_folds, n_bins))
    for b in range(n_bins):
        target[:, b] = (bins == b).sum() / n_folds
    have = np.zeros((n_folds, n_bins))
    fold = np.full(len(y), -1, int)

    for g in order:
        add = np.zeros(n_bins)
        for i in g:
            add[bins[i]] += 1
        # place where it most reduces the squared shortfall against target
        cost = ((have + add - target) ** 2).sum(1) - ((have - target) ** 2).sum(1)
        k = int(np.argmin(cost))
        have[k] += add
        for i in g:
            fold[i] = k
    assert (fold >= 0).all()
    return fold, scaff


def multitask_matrix(tr, a):
    """The union table: one row per compound, four label columns, NaN where absent.

    Rows are kept if ANY endpoint has a usable curve. The assay floor is applied per
    COLUMN, so a compound can be a genuine CYP3A4 curve at pIC50 3.1 and simultaneously
    a NaN for CYP1A2 because its CYP1A2 fit was an extrapolation below that assay's
    floor. Those are different facts about the same molecule and must not be conflated.
    """
    keep = pd.Series(False, index=tr.index)
    Y = np.full((len(tr), len(ISOFORMS)), np.nan)
    LO = np.full_like(Y, np.nan); HI = np.full_like(Y, np.nan)
    for j, iso in enumerate(ISOFORMS):
        col = ep(iso)
        floor = subfloor_bound(iso, a)
        ok = tr[col].notna()
        if floor is not None and not a.keep_below_floor:
            ok &= tr[col] > floor
        Y[ok.to_numpy(), j] = tr.loc[ok, col].to_numpy(float)
        for arr, suf in ((LO, "_conf_low"), (HI, "_conf_high")):
            if col + suf in tr.columns:
                arr[ok.to_numpy(), j] = tr.loc[ok, col + suf].to_numpy(float)
        keep |= ok
    d = tr[keep].reset_index(drop=True)
    k = keep.to_numpy()
    return d, Y[k], LO[k], HI[k]


def multitask_folds(smiles, Y, n_folds, n_bins, seed):
    """Scaffold-disjoint folds balanced on a COMPOSITE activity score.

    A single compound carries up to four labels on four different scales, so the
    balancing score is the mean of its available per-endpoint z-scores. A compound with
    only a CYP3A4 value is binned on that alone; one with all four on their average.
    Without this the folds balance on whichever endpoint happens to be present and the
    sparse endpoints end up lopsided.
    """
    Z = (Y - np.nanmean(Y, axis=0)) / np.where(np.nanstd(Y, axis=0) > 0,
                                               np.nanstd(Y, axis=0), 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = np.nanmean(Z, axis=1)
    score = np.where(np.isfinite(score), score, 0.0)
    return activity_scaffold_folds(smiles, score, n_folds, n_bins, seed)


# --------------------------------------------------------------------------------------
# METRICS -- the challenge's own, plus the usual regression ones
# --------------------------------------------------------------------------------------
def st_rae(pred, y, lo, hi):
    """Scaled-threshold relative absolute error: a prediction INSIDE the credible interval
    costs nothing; outside, you pay the distance to the nearer edge. Normalised by the
    cost a constant prediction of mean(y) would incur, so 1.0 = no better than a constant.
    """
    def cost(p):
        over = np.where(np.isfinite(hi), np.clip(p - hi, 0, None), 0.0)
        under = np.where(np.isfinite(lo), np.clip(lo - p, 0, None), 0.0)
        return float(np.sum(over + under))
    base = cost(np.full(len(y), float(np.mean(y))))
    return cost(pred) / base if base > 0 else np.nan


def report(pred, y, lo, hi):
    """Metrics, plus the one number that says whether the SPREAD is right.

    `slope` is the least-squares slope of truth regressed on prediction, cov(p,y)/var(p).
    It is the factor the predictions would have to be multiplied by (about their own mean)
    to be optimally scaled:

        slope > 1   predictions are too NARROW -- squeezed toward the mean, widening gains
        slope ~ 1   the spread is right; only the ranking limits R2
        slope < 1   predictions are too WIDE -- shrinking gains

    This matters here because ST-RAE pays the distance from a credible interval, so spread
    converts directly into score. R2 against rho^2 can show that the scaling is wrong but
    NOT which way -- being too wide and too narrow both lower R2 by the same mechanism.
    Only the slope carries the sign.
    """
    s = pd.Series(pred); t = pd.Series(y)
    vp = float(np.var(pred))
    return dict(n=len(y),
                st_rae=st_rae(pred, y, lo, hi),
                mae=float(np.mean(np.abs(pred - y))),
                r2=float(1 - np.mean((y - pred) ** 2) / np.var(y)),
                rho=float(s.corr(t, method="spearman")),
                tau=float(s.corr(t, method="kendall")),
                slope=float(np.cov(pred, y, bias=True)[0, 1] / vp) if vp > 1e-12
                else float("nan"),
                sd_ratio=float(np.std(pred) / (np.std(y) or np.nan)))


# --------------------------------------------------------------------------------------
# CONSOLE LAYOUT
# --------------------------------------------------------------------------------------
# One table per endpoint, one row per fold, column headers printed once. The held-out
# block is on the left because it is the number that counts; the training block is
# narrower and on the right because it exists only as the reference the gap is measured
# against. Everything is fixed-width so the columns line up when the terminal is copied
# into a report.
W = 78
COLS = (("fold", 5), ("n", 6), ("ST-RAE", 9), ("MAE", 8), ("R2", 8), ("rho", 8),
        ("", 2), ("MAE", 8), ("R2", 8), ("gap", 7), ("epoch", 9))


def rule(ch="-", w=W):
    return ch * w


def banner(text, ch="="):
    return f"\n{ch * W}\n  {text}\n{ch * W}"


def field(label, value):
    return f"  {label:<13s} {value}"


def status(msg, on=True):
    """A one-line 'what is running right now', overwritten in place.

    A fold takes minutes and prints nothing until it finishes, so without this the
    terminal looks hung and -- with four endpoints in one run -- gives no clue WHICH
    endpoint is training. Carriage return with no newline, so the finished table row
    lands on the same line and no scrollback is wasted.
    """
    if not on:
        return
    sys.stdout.write("\r" + (f"  {msg}"[:W]).ljust(W) + ("\r" if not msg else ""))
    sys.stdout.flush()


def table_head():
    """Two header lines: the group spans, then the column names."""
    left = sum(w for _, w in COLS[2:6])        # ST-RAE MAE R2 rho
    right = sum(w for _, w in COLS[7:9])       # train MAE R2
    span = (" " * (5 + 6) + " held out ".center(left, "-")
            + "  " + " trained on ".center(right, "-"))
    names = "".join(f"{n:>{w}s}" for n, w in COLS)
    return f"\n{span}\n{names}\n{rule()}"


def row(tag, mv, mt=None, epoch="", star=""):
    """One fold (or the pooled line). mt=None prints the held-out half only."""
    def num(v, w, p=4):
        return f"{v:>{w}.{p}f}" if v is not None and np.isfinite(v) else f"{'-':>{w}s}"
    gap = (f"{(mv['mae'] - mt['mae']) / max(mv['mae'], 1e-9):>+6.0%} " if mt else
           f"{'':>7s}")
    right = (num(mt["mae"], 8) + num(mt["r2"], 8) if mt else f"{'':>16s}")
    return (f"{tag:>5s}{mv['n']:>6d}" + num(mv["st_rae"], 9) + num(mv["mae"], 8)
            + num(mv["r2"], 8) + num(mv["rho"], 8) + "  " + right + gap
            + f"{epoch:>7s}{star:>2s}").rstrip()


def _fit_verdict(mt, mv, fits=()):
    """(label, what to do about it). The single source of truth for the fit call, so the
    per-endpoint verdict and the final summary table can never disagree.

    ORDER MATTERS. The gap is checked before the epoch count, because a model whose
    validation error is far above its training error is by definition NOT underfitting
    however early it stopped -- and calling that "cut short" points at the wrong fix.
    """
    rel = (mv["mae"] - mt["mae"]) / max(mv["mae"], 1e-9)
    early = sum(f["early"] for f in fits)
    best = float(np.mean([f["best_ep"] for f in fits])) if fits else float("nan")
    cap = fits[0]["cap"] if fits else 0
    cut_short = early > len(fits) / 2 and np.isfinite(best) and best < 0.6 * cap

    if rel > 0.55:
        return ("OVERFIT", "raise --dropout, shrink --hidden/--depth, or stop earlier")
    if rel > 0.30:
        return ("SOME OVERFIT", "normal for a scaffold split; held-out scaffolds "
                                "are genuinely harder")
    # The train-R2 bar was 0.75 and misfired: the first full run labelled CYP2C9
    # UNDERFIT on train R2 0.67 with a +28% gap, which is a model fitting its training
    # data perfectly respectably. A model that reaches R2 0.6 on its own rows is not
    # failing to fit; only one well below that is.
    if mt["r2"] < 0.35:
        return ("UNDERFIT", "does not fit even its own training rows -- more capacity, "
                            "more epochs, or a lower --dropout")
    if mt["r2"] < 0.60 and cut_short:
        return ("UNDERFIT", "training was cut short -- raise --patience and --epochs")
    return ("BALANCED", "not cut short; remaining error is data, not fitting")


def diagnose_label(mt, mv, fits=()):
    return _fit_verdict(mt, mv, fits)[0]


def diagnose(mt, mv, fits=(), y=None):
    """The verdict block: four labelled lines, each a claim plus the number behind it.

    Prose was the wrong shape for this. Every run asks the same four questions -- is the
    fit right, is the spread right, how good is the ranking, is it beating a constant --
    so they get four fixed rows in a fixed order, and a run can be compared to the one
    before it at a glance.

    THE TRAP THIS EXISTS TO AVOID. A small train-validation gap is not automatically good
    news. Early stopping keeps the best-VALIDATION checkpoint, so if it fires at epoch 14
    of 60 the train score comes from a model that never finished fitting: a mediocre train
    score and a small gap then mean "training was cut short", not "the architecture is
    right-sized". Dropout is off at evaluation, so a train R2 well under 0.8 on a 400-wide
    D-MPNN is a real signal and not a dropout artefact.

    ORDER MATTERS in the fit test below. The gap is checked before the epoch count,
    because a model whose validation error is far above its training error is by
    definition NOT underfitting however early it stopped -- and calling that "cut short"
    points at exactly the wrong fix.

    Thresholds are conventions for THIS task at this data size, not laws.
    """
    rel = (mv["mae"] - mt["mae"]) / max(mv["mae"], 1e-9)
    early = [f for f in fits if f["early"]]
    best = float(np.mean([f["best_ep"] for f in fits])) if fits else float("nan")
    cap = fits[0]["cap"] if fits else 0
    fit = _fit_verdict(mt, mv, fits)

    sl = mv.get("slope", float("nan"))
    if not np.isfinite(sl):
        spread = ("UNDEFINED", "constant predictions")
    elif sl > 1.15:
        spread = ("TOO NARROW", f"multiply about the mean by ~{sl:.2f} to widen")
    elif sl < 0.87:
        spread = ("TOO WIDE", f"multiply about the mean by ~{sl:.2f} to shrink")
    else:
        spread = ("OPTIMAL", "ranking, not scaling, is what limits R2")

    rho2 = mv["rho"] ** 2
    lines = [
        rule(),
        "  VERDICT",
        f"    {'fit':<9s} {fit[0]:<13s} "
        f"train MAE {mt['mae']:.3f} -> valid {mv['mae']:.3f} ({rel:+.0%})",
        f"    {'':<9s} {'':<13s} "
        + (f"{len(early)}/{len(fits)} folds stopped early, best epoch {best:.0f} of {cap}"
           if fits else "epoch data unavailable"),
        f"    {'':<9s} {'':<13s} {fit[1]}",
        f"    {'spread':<9s} {spread[0]:<13s} "
        f"slope {sl:.3f}, sd(pred)/sd(y) {mv.get('sd_ratio', float('nan')):.3f}",
        f"    {'':<9s} {'':<13s} {spread[1]}",
        f"    {'ranking':<9s} {'rho ' + format(mv['rho'], '.3f'):<13s} "
        f"rho^2 {rho2:.3f} vs R2 {mv['r2']:.3f} ({mv['r2'] - rho2:+.3f})",
    ]
    if y is not None and len(y):
        med = float(np.median(y))
        base = float(np.mean(np.abs(y - med)))
        lines.append(f"    {'baseline':<9s} {'vs median':<13s} "
                     f"MAE {mv['mae']:.3f} against {base:.3f} for one constant "
                     f"({mv['mae'] / base - 1:+.0%})")
    return "\n".join(x.rstrip() for x in lines)


def summary_table(rows):
    """One line per endpoint at the end, so a run is comparable to the run before it."""
    out = [banner("SUMMARY   pooled out-of-fold, held-out compounds only"),
           f"{'endpoint':<11s}{'n':>6s}{'ST-RAE':>9s}{'MAE':>8s}{'R2':>8s}{'rho':>8s}"
           f"{'slope':>8s}  fit", rule()]
    for r in rows:
        out.append(f"{r['iso']:<11s}{r['n']:>6d}{r['st_rae']:>9.4f}{r['mae']:>8.4f}"
                   f"{r['r2']:>8.4f}{r['rho']:>8.4f}{r['slope']:>8.3f}  {r['fit']}")
    if len(rows) > 1:
        out += [rule(),
                f"{'macro':<11s}{'':>6s}"
                + "".join(f"{float(np.mean([r[k] for r in rows])):>{w}.4f}"
                          for k, w in (("st_rae", 9), ("mae", 8), ("r2", 8), ("rho", 8)))]
    return "\n".join(x.rstrip() for x in out)


def tidy_callback(keep_progress=False):
    """Remove the parameter-count table and the progress bar that CHEMPROP attaches.

    `enable_model_summary=False` on the Trainer is not enough. That flag only stops the
    Trainer from adding a summary of its own; chemprop's model supplies its own callbacks
    when fit() runs, which is after the flag has had its say -- and `RichProgressBar`
    additionally pulls in `RichModelSummary` automatically. The result is the parameter
    table and a progress bar reappearing once per fold no matter what the Trainer was
    told.

    Removing them in `setup()` works because setup runs before `on_fit_start`, which is
    where a summary prints. The list is REPLACED rather than mutated in place, so the
    hook loop currently walking the old list is not disturbed.

    Checkpointing and early stopping are untouched; only display callbacks are dropped.
    """
    from lightning.pytorch.callbacks import Callback, ModelSummary, ProgressBar

    class _Tidy(Callback):
        def setup(self, trainer, pl_module, stage=None):
            trainer.callbacks = [
                c for c in trainer.callbacks
                if not isinstance(c, ModelSummary)
                and (keep_progress or not isinstance(c, ProgressBar))
            ]
    return _Tidy()


# --------------------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------------------
def build_model(hidden, depth, layers, dropout, n_desc, n_tasks=1):
    """One encoder; n_tasks outputs from the readout.

    n_tasks=1 is the per-endpoint model. n_tasks=4 is the shared trunk: the SAME pooled
    graph vector feeds four outputs, so every compound with a curve on any endpoint
    contributes gradient to the encoder that all four heads read from. That is the whole
    point -- each endpoint alone has 1,026-2,335 curve compounds, the union has 4,563.
    """
    from chemprop.models import MPNN
    from chemprop.nn import BondMessagePassing, MeanAggregation
    from chemprop.nn.predictors import RegressionFFN
    mp = BondMessagePassing(d_h=hidden, depth=depth, dropout=dropout)
    # the descriptor block is concatenated to the pooled graph vector, so the readout has
    # to be told the wider input
    ffn = RegressionFFN(input_dim=hidden + n_desc, hidden_dim=hidden,
                        n_layers=layers, dropout=dropout, n_tasks=n_tasks)
    return MPNN(message_passing=mp, agg=MeanAggregation(), predictor=ffn, batch_norm=True)


def datapoints(smiles, y, Xd, w=None):
    """y may be 1-D (single task) or (n, n_tasks) with NaN where an endpoint is missing.

    NaN is load-bearing in the multi-task case and is NOT filled in. chemprop masks it:
    `mask = targets.isfinite()` in MPNN.training_step, so a compound with a CYP3A4 curve
    and no CYP2D6 curve trains the encoder and the 3A4 head, and contributes nothing at
    all to the 2D6 head. Filling the gap with a zero or a column mean would instead teach
    every head a fabricated label.
    """
    from chemprop.data import MoleculeDatapoint
    from rdkit import Chem
    y = np.asarray(y, np.float32)
    if y.ndim == 1:
        y = y[:, None]
    pts, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        pts.append(MoleculeDatapoint(mol=m, y=y[i].astype(np.float32),
                                     x_d=Xd[i].astype(np.float32),
                                     weight=1.0 if w is None else float(w[i])))
        keep.append(i)
    return pts, np.asarray(keep)


def fit_scaler(X, rows):
    """Median and IQR from the TRAINING rows only. A constant column keeps scale 1, so it
    passes through as zero rather than exploding. Returned as a pair so the SAME numbers
    scale the validation rows and the blind set -- refitting on those would leak."""
    q1, q2, q3 = np.nanpercentile(X[rows], [25, 50, 75], axis=0)
    iqr = q3 - q1
    return q2, np.where(np.abs(iqr) < 1e-9, 1.0, iqr)


def apply_scaler(X, scaler):
    centre, scale = scaler
    return (X - centre) / scale


def inner_split(tr_idx, scaff, frac, seed):
    """Carve a scaffold-disjoint INNER validation set out of the training rows.

    WHY THIS EXISTS -- it fixes a leak. Early stopping and checkpoint selection need a
    validation signal. Using the outer fold for that means the checkpoint is chosen by
    minimising loss on the very compounds whose score is then reported as held out: the
    reported number is the best of ~60 epochs measured on those rows, not an honest
    estimate. The bias is small per fold but systematic, and it is exactly the kind of
    thing that makes local validation disagree with a leaderboard.

    The inner set is carved by SCAFFOLD, like the outer split, so an analogue of an
    inner-validation compound cannot sit in inner-training either.
    """
    groups = {}
    for i in tr_idx:
        groups.setdefault(scaff[i], []).append(int(i))
    keys = list(groups)
    np.random.default_rng(seed).shuffle(keys)
    target, pick = int(round(frac * len(tr_idx))), []
    for k in keys:
        if len(pick) >= target:
            break
        pick.extend(groups[k])
    hold = set(pick)
    inner_tr = np.array([i for i in tr_idx if i not in hold], int)
    inner_va = np.array(sorted(hold), int)
    return inner_tr, inner_va


def train_fold(tr_idx, va_idx, smiles, y, Xd, a, seed, scaff=None, w=None, real=None):
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from chemprop.data import MoleculeDataset, build_dataloader
    if not a.verbose:
        quiet_lightning()

    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    scaler = fit_scaler(Xd, tr_idx)
    Xs = apply_scaler(Xd, scaler)
    mu, sd = float(np.mean(y[tr_idx])), float(np.std(y[tr_idx]) or 1.0)
    ys = (y - mu) / sd

    # The rows the optimiser is allowed to look at, and the rows that stop it. With
    # --inner_frac 0 the stopping signal is the outer fold itself, which leaks (see
    # inner_split) and is only kept so the leaked behaviour can be reproduced.
    #
    # SURROGATE ROWS (--extra_csv) never form the stopping signal. They carry an imputed
    # label, so letting them pick the checkpoint would optimise the model against the
    # imputation rather than against measured chemistry. They are carved out of the
    # candidate pool first and added back to the fitting side afterwards.
    if real is None:
        pool, extra = tr_idx, np.array([], int)
    else:
        pool = tr_idx[real[tr_idx]]
        extra = tr_idx[~real[tr_idx]]
    if scaff is not None and a.inner_frac > 0:
        fit_idx, stop_idx = inner_split(pool, scaff, a.inner_frac, seed)
    else:
        fit_idx, stop_idx = pool, va_idx
    if len(extra):
        fit_idx = np.concatenate([fit_idx, extra])

    dtr, _ = datapoints([smiles[i] for i in fit_idx], ys[fit_idx], Xs[fit_idx],
                        None if w is None else w[fit_idx])
    dva, kva = datapoints([smiles[i] for i in stop_idx], ys[stop_idx], Xs[stop_idx])
    if len(dtr) == 0 or len(dva) == 0:
        raise RuntimeError("empty fitting or stopping split; check --inner_frac "
                           "and --extra_csv")
    ltr = build_dataloader(MoleculeDataset(dtr), batch_size=a.batch, shuffle=True,
                           num_workers=0)
    lva = build_dataloader(MoleculeDataset(dva), batch_size=a.batch, shuffle=False,
                           num_workers=0)

    model = build_model(a.hidden, a.depth, a.layers, a.dropout, N_DESC)
    ck = ModelCheckpoint(dirpath=a.model_dir, filename=f"f{seed}", monitor="val_loss",
                         mode="min", save_top_k=1)
    # min_epochs holds early stopping off while the learning-rate schedule is still in
    # its high phase. chemprop sizes that schedule against max_epochs, so a run killed
    # at epoch 10 of 60 never reaches the low-LR part where the fit sharpens -- which is
    # the most likely reason every fold stopped between epochs 7 and 14 with a train R2
    # near 0.29 on CYP1A2 while CYP3A4, which ran to epoch 23, reached 0.86.
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, max_epochs=a.epochs, min_epochs=a.min_epochs,
                         logger=False,
                         enable_progress_bar=a.progress, gradient_clip_val=1.0,
                         enable_model_summary=False,
                         callbacks=([] if a.no_early_stop else
                                    [EarlyStopping(monitor="val_loss",
                                                   patience=a.patience, mode="min")])
                                   + [ck, tidy_callback(a.progress)])
    trainer.fit(model, ltr, lva)

    # Which epoch the kept checkpoint came from, and whether training was cut short.
    # This is what tells an underfit apart from a hard task: a model that stopped at
    # epoch 14 of a possible 60 never got the chance to fit, and its mediocre TRAIN score
    # says nothing about the capacity of the architecture.
    ran = int(trainer.current_epoch)
    stopped_early = ran < a.epochs and not a.no_early_stop
    best_ep = ran
    if a.no_early_stop:
        return (model, (mu, sd), Xs, scaler, kva,
                dict(ran=ran, best_ep=ran, early=False, cap=a.epochs))
    try:
        import re
        mo = re.search(r"epoch=(\d+)", ck.best_model_path or "")
        if mo:
            best_ep = int(mo.group(1))
        elif getattr(ck, "best_model_score", None) is not None:
            best_ep = max(0, ran - a.patience)      # the patience window implies it
    except Exception:                                            # noqa: BLE001
        pass

    best = model
    if not a.no_early_stop and ck.best_model_path and os.path.exists(ck.best_model_path):
        from chemprop.models import MPNN
        for kw in ({"weights_only": False}, {}):
            try:
                best = MPNN.load_from_checkpoint(ck.best_model_path, **kw); break
            except Exception:                                        # noqa: BLE001
                continue
    fit_info = dict(ran=ran, best_ep=best_ep, early=stopped_early, cap=a.epochs)
    return best, (mu, sd), Xs, scaler, kva, fit_info


def predict(model, smiles, Xs, batch, verbose=False):
    import torch
    import lightning.pytorch as pl
    from chemprop.data import MoleculeDataset, build_dataloader
    if not verbose:
        quiet_lightning()
    d, keep = datapoints(smiles, np.zeros(len(smiles), np.float32), Xs)
    dl = build_dataloader(MoleculeDataset(d), batch_size=batch, shuffle=False,
                          num_workers=0)
    with torch.inference_mode():
        out = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, logger=False, enable_progress_bar=False,
                         enable_model_summary=False,
                         callbacks=[tidy_callback(False)]).predict(model, dl)
    return np.concatenate([x.cpu().numpy() for x in out]).ravel(), keep


# --------------------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--isoforms", nargs="*", default=list(ISOFORMS))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--bins", type=int, default=4, help="activity bins the split balances")
    p.add_argument("--hidden", type=int, default=400)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_dir", default="models_cyp_clean")
    p.add_argument("--tag", default="clean")
    p.add_argument("--floor", type=float, default=None,
                   help=f"the sub-floor bound (default {SUBFLOOR_BOUND}); it is the TDI "
                        "label threshold, NOT a detection limit -- see SUBFLOOR_BOUND")
    p.add_argument("--keep_subfloor", nargs="*", default=list(KEEP_SUBFLOOR),
                   metavar="ISO",
                   help=f"endpoints whose sub-{SUBFLOOR_BOUND} curves are KEPT "
                        f"(default: {' '.join(KEEP_SUBFLOOR)}). Pass all four to keep "
                        "every fitted curve, or none to drop them everywhere. CYP2C9 is "
                        "the one worth re-testing: rho 0.393 on its sub-floor rows")
    p.add_argument("--keep_below_floor", action="store_true",
                   help="alias for --keep_subfloor with all four endpoints")
    p.add_argument("--train_csv", default=TRAIN_CSV,
                   help="the real training file; the default is the challenge's own")
    p.add_argument("--extra_csv", default=None,
                   help="extra TRAINING rows in the same schema as the inhibition CSV "
                        "(Molecule_Name, SMILES, <endpoint>, _conf_low, _conf_high). "
                        "They join the fitting side only: never a validation fold, "
                        "never the early-stopping set, never a reported score. Rows "
                        "whose Molecule_Name is already in the real training file are "
                        "dropped. Default None, which leaves the run byte-identical to "
                        "one without this flag.")
    p.add_argument("--extra_weight", type=float, default=0.5,
                   help="per-row loss weight for --extra_csv rows (real rows are 1.0). "
                        "0.5 is what the CYP3A4 surrogate measured best at; 1.0 was not "
                        "significant. Ignored without --extra_csv.")
    p.add_argument("--inner_frac", type=float, default=0.12,
                   help="fraction of each fold's TRAINING rows held out, by scaffold, to "
                        "drive early stopping and checkpoint choice. 0 reverts to using "
                        "the outer fold, which leaks -- see inner_split")
    p.add_argument("--multitask", action="store_true",
                   help="ONE shared encoder with four heads, trained once on the union "
                        "of the four endpoints, instead of four separate models. Each "
                        "endpoint has only 1,026-2,335 curve compounds; the union has "
                        "4,563. Missing endpoints are NaN and are masked out of the loss")
    p.add_argument("--no_early_stop", action="store_true",
                   help="train the full --epochs budget and keep the LAST model, not the "
                        "best-validation checkpoint. This is the capacity test: it asks "
                        "whether the architecture CAN fit these data, with no selection "
                        "rule in the way. Not for producing a submission")
    p.add_argument("--min_epochs", type=int, default=15,
                   help="hold early stopping off for this many epochs, so the run is not "
                        "killed while the learning rate is still high")
    p.add_argument("--retries", type=int, default=2,
                   help="re-run a fold whose optimiser never got going (default 2)")
    p.add_argument("--dead_epoch", type=int, default=5,
                   help="a fold is suspect if its kept checkpoint is at or before this")
    p.add_argument("--dead_r2", type=float, default=0.25,
                   help="...and its TRAIN R2 is below this. Both must hold to retry")
    p.add_argument("--progress", action="store_true",
                   help="show chemprop's per-epoch progress bar (off by default: it "
                        "interleaves with the fold table and makes it unreadable)")
    p.add_argument("--verbose", action="store_true",
                   help="restore Lightning's own banners, warnings and parameter table")
    p.add_argument("--smoke", action="store_true",
                   help="2 folds, 3 epochs, outputs renamed -- ALWAYS run this first")
    a = p.parse_args()
    a.progress = a.progress or a.verbose

    if a.smoke:
        a.folds, a.epochs, a.patience = 2, 3, 3
        a.tag = "smoke_" + a.tag
        a.model_dir = a.model_dir + "_smoke"
    os.makedirs(a.model_dir, exist_ok=True)

    tr = pd.read_csv(a.train_csv); tr.columns = [c.strip() for c in tr.columns]
    te = pd.read_csv(TEST_CSV); te.columns = [c.strip() for c in te.columns]
    ex = None
    if a.extra_csv:
        ex = pd.read_csv(a.extra_csv); ex.columns = [c.strip() for c in ex.columns]
        before = len(ex)
        ex = ex[~ex[ID_COL].isin(set(tr[ID_COL]))].reset_index(drop=True)
        blind_ids = set(te[ID_COL])
        leak = ex[ex[ID_COL].isin(blind_ids)]
        if len(leak):
            raise SystemExit(f"--extra_csv contains {len(leak)} blind-set compounds; "
                             f"refusing to train on them")
        extra_note = (f"{len(ex)} from {a.extra_csv} at weight {a.extra_weight:.2f}"
                      + (f"   ({before - len(ex)} already in the real training file, "
                         f"dropped)" if before != len(ex) else "")
                      + "   never scored, never a stopping row")

    print(banner("CYP inhibition -- ChemProp, curve compounds only"))
    print(field("run", f"tag '{a.tag}'   seed {a.seed}   "
                       + ("SMOKE TEST (2 folds, 3 epochs)" if a.smoke else
                          f"{a.folds} folds")))
    print(field("model", f"D-MPNN hidden {a.hidden}, depth {a.depth}, "
                         f"FFN {a.layers} x {a.hidden}, dropout {a.dropout:.2f}"))
    print(field("training", f"up to {a.epochs} epochs, early stop patience {a.patience}, "
                            f"batch {a.batch}"))
    print(field("data", f"{len(tr)} train rows, {len(te)} blind rows"))
    if ex is not None:
        print(field("extra rows", extra_note))
    print(field("endpoints", ", ".join(a.isoforms)))

    oof_store, pred_store, summary_rows = {}, {}, []

    if a.multitask:
        run_multitask(tr, te, a, oof_store, pred_store, summary_rows)
        write_outputs(te, a, oof_store, pred_store, summary_rows)
        return 0

    for iso in a.isoforms:
        col = ep(iso)
        lo_col, hi_col = col + "_conf_low", col + "_conf_high"
        have = tr[tr[col].notna()]
        floor = subfloor_bound(iso, a)
        if a.keep_below_floor or floor is None:
            d, dropped = have.reset_index(drop=True), 0
        else:
            d = have[have[col] > floor].reset_index(drop=True)
            dropped = len(have) - len(d)
        smiles = d[SMILES_COL].tolist()
        y = d[col].to_numpy(float)
        lo = d[lo_col].to_numpy(float) if lo_col in d else np.full(len(d), np.nan)
        hi = d[hi_col].to_numpy(float) if hi_col in d else np.full(len(d), np.nan)
        n_real = len(d)
        # Surrogate rows are appended AFTER the real ones and after the folds are cut,
        # so every index below n_real is measured and every index at or above it is
        # imputed. `real` carries that distinction to the one place it matters.
        xd_rows = None
        if ex is not None and col in ex:
            xd_rows = ex[ex[col].notna()].reset_index(drop=True)
            if len(xd_rows):
                smiles = smiles + xd_rows[SMILES_COL].tolist()
                y = np.concatenate([y, xd_rows[col].to_numpy(float)])
                lo = np.concatenate([lo, (xd_rows[lo_col].to_numpy(float)
                                          if lo_col in xd_rows
                                          else np.full(len(xd_rows), np.nan))])
                hi = np.concatenate([hi, (xd_rows[hi_col].to_numpy(float)
                                          if hi_col in xd_rows
                                          else np.full(len(xd_rows), np.nan))])
        n_extra = len(smiles) - n_real
        real = np.arange(len(smiles)) < n_real
        w = np.where(real, 1.0, a.extra_weight)

        print(banner(f"{iso}      endpoint {a.isoforms.index(iso) + 1} of "
                     f"{len(a.isoforms)}"))
        print(field("compounds", f"{len(d)} with a dose-response curve"))
        print(field("", f"{len(tr) - len(have)} with no measurable activity (no value), "
                        f"dropped"
                        + (f"; {dropped} extrapolated below the pIC50 {floor:.1f} assay "
                           f"floor, dropped" if dropped else "")
                        + ("; floor override: KEEPING sub-floor rows"
                           if a.keep_below_floor and floor is not None else "")))
        print(field("pIC50", f"mean {y[:n_real].mean():.2f}   "
                             f"sd {y[:n_real].std():.2f}   "
                             f"range {y[:n_real].min():.2f} - {y[:n_real].max():.2f}"))

        # Folds are cut on the MEASURED rows alone, so adding surrogates cannot move a
        # real compound into a different fold and the two runs stay comparable.
        fold, scaff = activity_scaffold_folds(smiles[:n_real], y[:n_real],
                                              a.folds, a.bins, a.seed)
        if n_extra:
            # A surrogate inherits the fold of its scaffold when that scaffold is also in
            # the real data. That is what keeps it out of the training side of the fold
            # holding its own analogues -- otherwise an imputed label would leak into the
            # held-out score through a shared Bemis-Murcko core. A scaffold seen nowhere
            # in the real rows gets -1 and therefore trains in every fold.
            sx = bemis_murcko(smiles[n_real:])
            home = {}
            for i, s in enumerate(scaff):
                home.setdefault(s, fold[i])
            fold = np.concatenate([fold, np.array([home.get(s, -1) for s in sx], int)])
            scaff = np.concatenate([np.asarray(scaff), np.asarray(sx)])
            shared = int(sum(1 for s in sx if s in home))
            print(field("surrogates", f"{n_extra} rows: {shared} share a scaffold with "
                                      f"the real data and inherit its fold, "
                                      f"{n_extra - shared} train in every fold"))
        sizes = np.bincount(fold[:n_real], minlength=a.folds)
        print(field("split", f"{len(set(scaff[:n_real]))} Bemis-Murcko scaffolds -> "
                             f"{a.folds} activity-balanced folds"))
        print(field("", f"fold sizes {sizes.min()}-{sizes.max()}, held-out mean pIC50 "
                        f"{min(y[:n_real][fold[:n_real] == k].mean() for k in range(a.folds)):.2f}-"
                        f"{max(y[:n_real][fold[:n_real] == k].mean() for k in range(a.folds)):.2f}"))

        Xd = compute_descriptors(smiles)
        Xte = compute_descriptors(te[SMILES_COL].tolist())
        print(field("descriptors", f"{N_DESC} interpretable columns"))
        print(table_head())

        oof = np.full(len(d), np.nan)
        te_pred, t0 = [], time.time()
        tr_ms, va_ms, fits = [], [], []
        for k in range(a.folds):
            # A validation fold is measured rows only. Surrogates that inherited this
            # fold are simply absent from both sides for this k, which is the point.
            va = np.where((fold == k) & real)[0]
            trn = np.where(fold != k)[0]
            # DEAD-FOLD RETRY. In the first full run, 3 folds out of 40 stopped at epoch
            # 2-5 with a train R2 of 0.08-0.17: the optimiser never got going from that
            # initialisation, and early stopping then locked in an untrained model. Those
            # folds are not information about the data, they are failed runs, and they go
            # straight into the pooled score and the blind average. A fold is judged dead
            # if it kept a checkpoint from the first few epochs AND cannot fit its own
            # training rows; it is retried with a different seed, at most --retries times.
            attempt, model, fi = 0, None, None
            while True:
                status(f"{iso}  fold {k + 1}/{a.folds}  training on {len(trn)} compounds"
                       + (f"  (retry {attempt})" if attempt else "") + " ...",
                       not a.progress)
                model, (mu, sd), Xs, scaler, _, fi = train_fold(
                    trn, va, smiles, y, Xd, a, a.seed + k + 1000 * attempt, scaff,
                    w=w, real=real)
                status(f"{iso}  fold {k + 1}/{a.folds}  scoring ...", not a.progress)
                # The train-side reference is read on MEASURED training rows only, so
                # the fit verdict is never flattered by rows whose label was imputed.
                trn_real = trn[real[trn]]
                pt_, kt_ = predict(model, [smiles[i] for i in trn_real], Xs[trn_real],
                                   a.batch, a.verbose)
                ti_ = trn_real[kt_]
                train_r2 = float(1 - np.mean((y[ti_] - (pt_ * sd + mu)) ** 2)
                                 / np.var(y[ti_]))
                dead = fi["best_ep"] <= a.dead_epoch and train_r2 < a.dead_r2
                if not dead or attempt >= a.retries:
                    if dead:
                        status("", not a.progress)
                        print(f"  NOTE fold {k + 1}: still not fitting after "
                              f"{a.retries} retries (train R2 {train_r2:.3f}); kept")
                    break
                attempt += 1
            fits.append(fi)

            # HELD-OUT rows: the honest number
            pv, keep = predict(model, [smiles[i] for i in va], Xs[va], a.batch,
                               a.verbose)
            oof[va[keep]] = pv * sd + mu
            mv = report(oof[va[keep]], y[va[keep]], lo[va[keep]], hi[va[keep]])

            # THE SAME model re-scored on its OWN TRAINING rows. This is not a performance
            # estimate -- it is the reference the held-out number is read against. A model
            # that fits its training data no better than its validation data has not learnt
            # what it could (underfitting); a model that fits it far better has learnt the
            # rows rather than the chemistry (overfitting).
            pt, keept = pt_, kt_                      # already computed by the retry loop
            ti = trn_real[keept]
            mt = report(pt * sd + mu, y[ti], lo[ti], hi[ti])
            tr_ms.append(mt); va_ms.append(mv)

            # blind set: scaled by THIS fold's training median and IQR, never its own
            tp, tk = predict(model, te[SMILES_COL].tolist(),
                             apply_scaler(Xte, scaler), a.batch, a.verbose)
            full = np.full(len(te), np.nan); full[tk] = tp * sd + mu
            te_pred.append(full)

            status("", not a.progress)          # clear the live line, then the row
            print(row(str(k + 1), mv, mt, f"{fi['best_ep']}/{fi['cap']}",
                      "*" if fi["early"] else ""))

        ok = np.isfinite(oof)
        mean_tr = {k_: float(np.nanmean([m[k_] for m in tr_ms])) for k_ in tr_ms[0]}
        mean_va = {k_: float(np.nanmean([m[k_] for m in va_ms])) for k_ in va_ms[0]}
        mean_tr["n"] = int(mean_tr["n"]); mean_va["n"] = int(mean_va["n"])
        # oof covers the measured rows only, so the truth arrays are sliced to match
        # before the boolean mask is applied. Every reported number is measured.
        y_r, lo_r, hi_r = y[:n_real], lo[:n_real], hi[:n_real]
        pooled = report(oof[ok], y_r[ok], lo_r[ok], hi_r[ok])
        print(rule())
        print(row("pool", pooled, mean_tr))
        n_early = sum(f["early"] for f in fits)
        print(f"\n  {a.folds} folds in {(time.time() - t0) / 60:.1f} min"
              + (f"    * = stopped early ({n_early} of {a.folds})" if n_early else ""))
        # The verdict reads SPREAD off the POOLED row, not the mean of the per-fold
        # slopes. They are different numbers -- the first run printed slope 1.068 in the
        # CYP2D6 verdict (mean of folds) and 0.867 in the summary (pooled) -- and the
        # pooled one is the one that matters, because it is the actual scaling of the
        # column that gets submitted. Averaging per-fold slopes averages ten noisy
        # ratios and is not the ratio of the average.
        spread_src = dict(mean_va, slope=pooled["slope"], sd_ratio=pooled["sd_ratio"],
                          r2=pooled["r2"], rho=pooled["rho"])
        print(diagnose(mean_tr, spread_src, fits, y=y_r[ok]))

        fit_label = diagnose_label(mean_tr, mean_va, fits)
        summary_rows.append(dict(iso=iso, fit=fit_label, **{
            k_: pooled[k_] for k_ in ("n", "st_rae", "mae", "r2", "rho", "slope")}))
        oof_store[iso] = pd.Series(oof, index=d[ID_COL])
        pred_store[iso] = np.nanmean(np.vstack(te_pred), axis=0)

    write_outputs(te, a, oof_store, pred_store, summary_rows)
    return 0


def write_outputs(te, a, oof_store, pred_store, summary_rows):
    """The two CSVs, in the shape everything downstream expects. Shared by both paths."""
    print(summary_table(summary_rows))
    ids = sorted(set().union(*[set(v.index) for v in oof_store.values()]))
    oof_out = pd.DataFrame({ID_COL: ids})
    for iso, v in oof_store.items():
        oof_out[ep(iso)] = oof_out[ID_COL].map(v)
    oof_out.to_csv(f"oof_{a.tag}.csv", index=False)

    # blind predictions, in the blind file's own row order, one column per endpoint run
    sub = te[[SMILES_COL, ID_COL]].copy()
    for iso in ISOFORMS:                      # fixed column order, not --isoforms order
        if iso in pred_store:
            sub[ep(iso)] = pred_store[iso]
    sub.to_csv(f"pred_{a.tag}.csv", index=False)
    print(f"wrote oof_{a.tag}.csv ({len(oof_out)} rows) and "
          f"pred_{a.tag}.csv ({len(sub)} rows)")
    submission_check(sub, te, a)


def train_fold_mt(tr_idx, va_idx, smiles, Ys, Xd, a, seed, scaff):
    """One shared-trunk model for one fold. Ys is already per-task standardised."""
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from chemprop.data import MoleculeDataset, build_dataloader
    if not a.verbose:
        quiet_lightning()

    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if a.inner_frac > 0:
        fit_idx, stop_idx = inner_split(tr_idx, scaff, a.inner_frac, seed)
    else:
        fit_idx, stop_idx = tr_idx, va_idx

    dtr, _ = datapoints([smiles[i] for i in fit_idx], Ys[fit_idx], Xd[fit_idx])
    dva, _ = datapoints([smiles[i] for i in stop_idx], Ys[stop_idx], Xd[stop_idx])
    ltr = build_dataloader(MoleculeDataset(dtr), batch_size=a.batch, shuffle=True,
                           num_workers=0)
    lva = build_dataloader(MoleculeDataset(dva), batch_size=a.batch, shuffle=False,
                           num_workers=0)

    model = build_model(a.hidden, a.depth, a.layers, a.dropout, N_DESC, len(ISOFORMS))
    ck = ModelCheckpoint(dirpath=a.model_dir, filename=f"mt{seed}", monitor="val_loss",
                         mode="min", save_top_k=1)
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, max_epochs=a.epochs, min_epochs=a.min_epochs,
                         logger=False, enable_progress_bar=a.progress,
                         gradient_clip_val=1.0, enable_model_summary=False,
                         callbacks=([] if a.no_early_stop else
                                    [EarlyStopping(monitor="val_loss",
                                                   patience=a.patience, mode="min")])
                                   + [ck, tidy_callback(a.progress)])
    trainer.fit(model, ltr, lva)

    ran = int(trainer.current_epoch)
    fi = dict(ran=ran, best_ep=ran, early=False, cap=a.epochs)
    best = model
    if not a.no_early_stop:
        fi["early"] = ran < a.epochs
        try:
            import re
            mo = re.search(r"epoch=(\d+)", ck.best_model_path or "")
            fi["best_ep"] = int(mo.group(1)) if mo else max(0, ran - a.patience)
        except Exception:                                            # noqa: BLE001
            pass
        if ck.best_model_path and os.path.exists(ck.best_model_path):
            from chemprop.models import MPNN
            for kw in ({"weights_only": False}, {}):
                try:
                    best = MPNN.load_from_checkpoint(ck.best_model_path, **kw); break
                except Exception:                                    # noqa: BLE001
                    continue
    return best, fi


def predict_mt(model, smiles, Xs, batch, verbose=False):
    """(n, n_tasks) predictions, plus the indices of the molecules RDKit could parse."""
    import torch
    import lightning.pytorch as pl
    from chemprop.data import MoleculeDataset, build_dataloader
    if not verbose:
        quiet_lightning()
    d, keep = datapoints(smiles, np.zeros((len(smiles), len(ISOFORMS)), np.float32), Xs)
    dl = build_dataloader(MoleculeDataset(d), batch_size=batch, shuffle=False,
                          num_workers=0)
    with torch.inference_mode():
        out = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, logger=False, enable_progress_bar=False,
                         enable_model_summary=False,
                         callbacks=[tidy_callback(False)]).predict(model, dl)
    P = np.concatenate([x.cpu().numpy() for x in out])
    return P.reshape(len(keep), -1), keep


def run_multitask(tr, te, a, oof_store, pred_store, summary_rows):
    """One encoder, four heads, trained once on the union of the four endpoints.

    WHY. Each endpoint has 1,026-2,335 curve compounds; the union is 4,563 rows over
    4,563 compounds, and the four CYPs share most of what determines binding -- size,
    lipophilicity, ionisation, a heme-coordinating nitrogen. A compound measured only on
    CYP3A4 still teaches the encoder what a CYP ligand looks like, and the CYP2D6 head
    reads from that same encoder. The single-task run is the control: CYP3A4, which kept
    77% of its data, reached train R2 0.865 while the three endpoints reduced to ~30% of
    theirs reached 0.29-0.38.

    WHAT IS SHARED AND WHAT IS NOT. The message-passing encoder and the first FFN layer
    are shared; the four outputs are separate. Each endpoint keeps its own mean and sd,
    standardised on training rows only, so no cross-assay harmonisation is assumed.
    """
    d, Y, LO, HI = multitask_matrix(tr, a)
    smiles = d[SMILES_COL].tolist()
    n_t = len(ISOFORMS)

    print(banner("SHARED TRUNK   one encoder, four heads"))
    print(field("compounds", f"{len(d)} with a curve on at least one endpoint"))
    per = "  ".join(f"{iso} {int(np.isfinite(Y[:, j]).sum())}"
                    for j, iso in enumerate(ISOFORMS))
    print(field("per endpoint", per))
    both = np.isfinite(Y).sum(1)
    print(field("labels each", "  ".join(f"{k} endpoint{'s' if k > 1 else ' '}: "
                                         f"{int((both == k).sum())}"
                                         for k in range(1, n_t + 1))))

    fold, scaff = multitask_folds(smiles, Y, a.folds, a.bins, a.seed)
    print(field("split", f"{len(set(scaff))} Bemis-Murcko scaffolds -> {a.folds} folds, "
                         f"balanced on the mean available z-score"))
    Xd = compute_descriptors(smiles)
    Xte = compute_descriptors(te[SMILES_COL].tolist())
    print(field("descriptors", f"{N_DESC} interpretable columns"))

    oof = np.full_like(Y, np.nan)
    te_pred, fits, t0 = [], [], time.time()
    tr_ms = {iso: [] for iso in ISOFORMS}      # per-fold TRAIN metrics, per head
    for k in range(a.folds):
        va = np.where(fold == k)[0]; trn = np.where(fold != k)[0]
        status(f"shared trunk  fold {k + 1}/{a.folds}  training on {len(trn)} "
               f"compounds ...", not a.progress)
        scaler = fit_scaler(Xd, trn)
        Xs = apply_scaler(Xd, scaler)
        # per-task standardisation, training rows only; NaN stays NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu = np.nanmean(Y[trn], axis=0)
            sd = np.nanstd(Y[trn], axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        Ys = (Y - mu) / sd

        model, fi = train_fold_mt(trn, va, smiles, Ys, Xs, a, a.seed + k, scaff)
        fits.append(fi)

        status(f"shared trunk  fold {k + 1}/{a.folds}  predicting ...", not a.progress)
        pv, kv = predict_mt(model, [smiles[i] for i in va], Xs[va], a.batch, a.verbose)
        oof[va[kv]] = pv * sd + mu

        # the same model on its OWN training rows, per head -- the overfit reference
        pt, kt_ = predict_mt(model, [smiles[i] for i in trn], Xs[trn], a.batch, a.verbose)
        ti = trn[kt_]; Pt = pt * sd + mu
        for j, iso in enumerate(ISOFORMS):
            mj = np.isfinite(Y[ti, j])
            if mj.sum() > 10:
                tr_ms[iso].append(report(Pt[mj, j], Y[ti][mj, j],
                                         LO[ti][mj, j], HI[ti][mj, j]))
        tp, kt = predict_mt(model, te[SMILES_COL].tolist(), apply_scaler(Xte, scaler),
                            a.batch, a.verbose)
        full = np.full((len(te), n_t), np.nan); full[kt] = tp * sd + mu
        te_pred.append(full)
        status("", not a.progress)
        print(f"  fold {k + 1:2d}/{a.folds}  n_va {len(va):4d}  "
              f"kept epoch {fi['best_ep']}/{fi['cap']}{'  *' if fi['early'] else ''}")

    print(f"\n  {a.folds} folds in {(time.time() - t0) / 60:.1f} min "
          f"(one model, not {len(ISOFORMS)})")

    blind = np.nanmean(np.stack(te_pred), axis=0)
    for j, iso in enumerate(ISOFORMS):
        m = np.isfinite(Y[:, j]) & np.isfinite(oof[:, j])
        if not m.sum():
            continue
        pooled = report(oof[m, j], Y[m, j], LO[m, j], HI[m, j])
        ms = tr_ms[iso]
        mean_tr = ({k_: float(np.nanmean([x[k_] for x in ms])) for k_ in ms[0]}
                   if ms else None)
        if mean_tr:
            mean_tr["n"] = int(mean_tr["n"])
        print(banner(f"{iso}      shared-trunk head"))
        print(table_head())
        print(row("pool", pooled, mean_tr))
        if mean_tr:
            spread_src = dict(pooled)
            print(diagnose(mean_tr, spread_src, fits, y=Y[m, j]))
        fit_label = diagnose_label(mean_tr, pooled, fits) if mean_tr else "shared trunk"
        summary_rows.append(dict(iso=iso, fit=fit_label, **{
            k_: pooled[k_] for k_ in ("n", "st_rae", "mae", "r2", "rho", "slope")}))
        oof_store[iso] = pd.Series(oof[:, j], index=d[ID_COL])
        pred_store[iso] = blind[:, j]


def submission_check(sub, te, a):
    """Say plainly whether pred_<tag>.csv could be uploaded, and what is still missing.

    Shape and content are different questions. This checks shape -- the columns, the rows,
    the order, and that no cell is empty -- because a silently broken file costs a day's
    submission slot. It does NOT and cannot tell you the file will score well.
    """
    print("\n" + "=" * 78)
    print("SUBMISSION READINESS")
    missing_iso = [i for i in ISOFORMS if ep(i) not in sub.columns]
    cols = [ep(i) for i in ISOFORMS if ep(i) in sub.columns]

    ok_rows = len(sub) == len(te)
    ok_order = bool((sub[ID_COL].to_numpy() == te[ID_COL].to_numpy()).all())
    nan = {c: int(sub[c].isna().sum()) for c in cols}
    n_nan = sum(nan.values())

    print(f"  rows          {len(sub)} (blind file has {len(te)})   "
          f"{'OK' if ok_rows else 'MISMATCH'}")
    print(f"  row order     {'identical to the blind file' if ok_order else 'REORDERED'}")
    print(f"  columns       {len(cols)}/4 endpoints present")
    if missing_iso:
        print(f"                MISSING: {', '.join(missing_iso)}")
    print(f"  empty cells   {n_nan}"
          + (f"  ({', '.join(f'{c.split(chr(95))[0]}={v}' for c, v in nan.items() if v)})"
             if n_nan else ""))

    print("\n  distribution of the predictions written to the file:")
    print(f"  {'endpoint':10s} {'blind mean':>11s} {'blind sd':>9s} {'blind range':>14s}")
    for i in ISOFORMS:
        if ep(i) not in sub.columns:
            continue
        v = sub[ep(i)].to_numpy(float); v = v[np.isfinite(v)]
        print(f"  {i:10s} {v.mean():11.3f} {v.std():9.3f} "
              f"{f'{v.min():.2f}-{v.max():.2f}':>14s}")

    ready = ok_rows and ok_order and not missing_iso and n_nan == 0 and not a.smoke
    if a.smoke:
        print("\n  SMOKE RUN -- 2 folds and 3 epochs. Not a candidate for anything.")
    elif ready:
        print(f"\n  SHAPE: pred_{a.tag}.csv has the columns, rows and order the upload form"
              f"\n  expects. Rename it to submission_<name>.csv and it would be accepted.")
    else:
        print("\n  SHAPE: NOT uploadable as it stands. Fix the lines marked above first."
              "\n  A subset run (--isoforms) is expected to land here: the form needs all"
              "\n  four endpoints in one file.")

    print("""
  ACCEPTED IS NOT THE SAME AS GOOD. Two reasons this file is unlikely to beat the
  current best submission, both of them consequences of choices made on purpose:

  1. NO PLACEMENT. ST-RAE pays the distance from a credible interval, so the mean and
     spread of the column matter as much as the ranking. This file's mean and spread
     are whatever the model produced. Nothing was mapped onto an estimate of the blind
     set's own distribution.
  2. CENSORED ROWS WERE DROPPED. Training used the primary-screen hits only, so the
     model has never seen an inactive compound. It will place genuinely inactive blind
     compounds too high, and the bottom of its range is compressed. Compare the means
     above against the previous submission's before assuming otherwise.

  So: validate first, submit second. Compare this file against the standing best
  per endpoint on the out-of-fold pool, and only upload where it wins -- one slot a
  day is shared with the TDI track.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
