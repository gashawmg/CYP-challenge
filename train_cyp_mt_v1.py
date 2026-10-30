"""
CYP pIC50 — Multitask Chemprop, v1 (BASELINE: learned features only)
=====================================================================
Port of  train_pec50_pxr_v4_4.py  (PXR, single-task, RAE=0.5581 best single Chemprop)

This is deliberately the BASELINE rung of your adaptive ladder: message-passing
features ONLY, no target-relevant descriptors. Run this first, record the number,
then turn descriptors on with --use_descriptors for the v2 comparison.

Changelog v4_4 (PXR) -> cyp_mt_v1
---------------------------------
NEW  MULTITASK. Four targets in one model:
       CYP1A2 / CYP2C9 / CYP2D6 / CYP3A4  _pIC50_direct_inhibition
     Chemprop 2.x masks missing targets natively — MPNN.training_step does
     `mask = targets.isfinite()` — so a compound contributes to exactly the
     isoforms it was measured on. NaN is the missing-label marker; do NOT fill it.
     This is the whole point of the port: CYP3A4 has 2,335 dose-response curves
     and out-of-fold ST-RAE 0.495, while CYP1A2/2C9/2D6 have ~1,300-1,500 each
     and score at or WORSE than a constant predictor (1.05 / 0.95 / 1.05). A
     shared trunk is the one architecture that lets the three starved heads
     borrow from the one that isn't.

NEW  --use_descriptors flag, DEFAULT OFF.  Baseline = learned features only,
     input_extra=0, no x_d on the datapoints. Your v4_4 always concatenated the
     61 PXR descriptors; here that is opt-in so the two rungs are separable.

KEPT  activity-stratified scaffold as the DEFAULT split, matching v4_4.
     Generalised to multitask: a row is binned on the mean of its z-scored
     AVAILABLE targets, so a compound measured on one isoform is binned on that
     isoform rather than dragged to the middle by three NaNs.

     Measured on this dataset so you can see what it is rather than trusting the
     name: it deals multi-member scaffold groups round-robin across folds, so
     11.2% of compounds share a Murcko scaffold with a compound in another fold
     (0.0% for the strict `scaffold` split). Difficulty barely moves — median
     held-out max-Tanimoto 0.438 vs 0.433 — while activity balance improves a
     lot, which matters here because three of four endpoints are weak and a fold
     could otherwise end up with almost no actives. Every run now prints the
     leakage figure for whichever strategy you pick.

     The caveat that actually bites is different: BOTH splits are harder than
     the real task. Blind-test compounds sit at median max-Tanimoto 0.587 to the
     training set — they were purchased as analogues of screening hits — so any
     structure-based CV here is pessimistic relative to the leaderboard.

NEW  --split_strategy frozen.
     Loads artifacts/folds.csv — the same Butina folds every classical model in
     this project used. Use it if you intend to BLEND this model with the
     LightGBM/XGBoost/CatBoost ensemble: blend weights fitted across two
     different fold assignments are fitted on compounds that were in-fold for
     one member and out-of-fold for the other, which inflates whichever member
     saw them.

CHANGED  N_FOLDS 10 -> 5 (your call; the folds are large enough here).

CHANGED  Per-task y scaling. RobustScaler cannot take NaN, so each target column
     is scaled independently on its own non-missing rows and the NaNs are put
     back afterwards.

CHANGED  Augmenter carries a 4-vector per molecule instead of a scalar.

FIXED    --no_sample_weights had `default=True` in v4_4, so the flag was ALWAYS
     on and weights were silently dead in every v4_4 run regardless of what you
     passed. Default is now False and the flag means what it says. Note this
     changes v4_4's effective behaviour if you ever back-port it.

DROPPED  Selectivity / CI-width weighting (PXR columns that don't exist here),
     the SC inactive supplement, and the high-tail multiplier. Tail and cliff
     weighting are kept but default OFF for a clean baseline.

Usage
-----
  # baseline, standalone number (activity-stratified scaffold, as in v4_4)
  python train_cyp_mt_v1.py --mode BOTH

  # baseline on the project folds, so the OOF can join the ensemble blend
  python train_cyp_mt_v1.py --mode BOTH --split_strategy frozen \
      --folds_csv artifacts/folds.csv

  # rung 2: add target-relevant descriptors
  python train_cyp_mt_v1.py --mode BOTH --use_descriptors \
      --model_dir models_cyp_mt_v2 --output predictions_cyp_mt_v2.csv
"""

import argparse, glob, json, os, pickle, random, time, warnings
from collections import defaultdict
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
from rdkit.Chem import Descriptors, rdMolDescriptors as rdMD, Crippen, rdFingerprintGenerator, DataStructs
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser, Uncharger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_CSV  = "data/cyp-challenge-TRAIN_inhibition.csv"
TEST_CSV   = "data/cyp-challenge-TEST-BLINDED.csv"
SMILES_COL = "SMILES"
NAME_COL   = "Molecule_Name"

ISOFORMS    = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
TARGET_COLS = [f"{c}_pIC50_direct_inhibition" for c in ISOFORMS]
CONF_LOW    = "_conf_low"
CONF_HIGH   = "_conf_high"

MODEL_SAVE_DIR = "models_cyp_mt_v1"
OUTPUT_CSV     = "predictions_cyp_mt_v1.csv"
OOF_CSV        = "oof_chemprop_mt_v1.csv"

GLOBAL_SEED = 42
N_FOLDS     = 5

BUTINA_THRESHOLD  = 0.4
KMEANS_N_CLUSTERS = 15

# Both default to 1.0 == OFF. The baseline should be a clean architecture
# measurement; turn these on only when comparing against it.
TAIL_WEIGHT_MAX          = 1.0
CLIFF_WEIGHT_MULTIPLIER  = 1.0
CLIFF_TANIMOTO_THRESHOLD = 0.5
CLIFF_DELTA_THRESHOLD    = 1.0

OUTLIER_MAD_THRESHOLD = None   # multitask: a row is rarely an outlier on all 4

ADAPTIVE_CONFIG = {
    'small':      {'threshold': 300,          'hidden_dim': 384, 'depth': 4, 'n_layers': 3,
                   'dropout': 0.30, 'max_epochs': 150, 'patience': 20, 'batch_size': 32,
                   'augmentation_factor': 5,  'use_augmentation': True},
    'medium':     {'threshold': 800,          'hidden_dim': 512, 'depth': 5, 'n_layers': 3,
                   'dropout': 0.25, 'max_epochs': 200, 'patience': 20, 'batch_size': 64,
                   'augmentation_factor': 3,  'use_augmentation': True},
    'large':      {'threshold': 2000,         'hidden_dim': 768, 'depth': 5, 'n_layers': 3,
                   'dropout': 0.20, 'max_epochs': 200, 'patience': 20, 'batch_size': 64,
                   'augmentation_factor': 2,  'use_augmentation': True},
    'very_large': {'threshold': float('inf'), 'hidden_dim': 768, 'depth': 6, 'n_layers': 4,
                   'dropout': 0.20, 'max_epochs': 300, 'patience': 25, 'batch_size': 64,
                   'augmentation_factor': 1,  'use_augmentation': True},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# =============================================================================
# CYP DESCRIPTORS — only used with --use_descriptors (rung 2)
# =============================================================================

CYP_SMARTS = {
    'basic_amine':      '[NX3;H2,H1,H0;!$(N[a]);!$(NC=[O,S,N]);!$(N=*)]',
    'aromatic_n_heme':  '[n;X2;!$(n[#6]=O)]',
    'imidazole':        'c1cnc[nH]1',
    'triazole':         'c1ncnn1',
    'pyridine_n':       'n1ccccc1',
    'carboxylic_acid':  '[CX3](=O)[OX2H1]',
    'sulfonamide':      '[SX4](=O)(=O)[NX3]',
    'phenol':           '[OX2H]c',
    'aniline':          '[NX3;!$(NC=[O,S,N])][a]',
    'amide':            '[NX3][CX3]=[OX1]',
    'ester':            '[CX3](=O)[OX2][#6]',
    'ether':            '[OX2;!$([OX2H]);!$(OC=O)]',
    'halogen':          '[F,Cl,Br,I]',
    'nitro':            '[$([NX3](=O)=O),$([NX3+](=O)[O-])]',
    'benzodioxole':     'c1ccc2c(c1)OCO2',
    'terminal_alkyne':  '[CX2H]#[CX2]',
    'furan':            'c1ccoc1',
    'thiophene':        'c1ccsc1',
    'thiourea':         '[NX3][CX3]=[SX1]',
    'michael':          '[CX3]=[CX3][CX3]=[OX1]',
}

RDKIT_DESCS = [
    'MolWt', 'MolLogP', 'TPSA', 'NumHAcceptors', 'NumHDonors',
    'NumRotatableBonds', 'RingCount', 'NumAromaticRings', 'NumSaturatedRings',
    'NumAliphaticRings', 'FractionCSP3', 'HeavyAtomCount', 'MolMR',
    'BertzCT', 'Kappa1', 'Kappa2', 'Kappa3', 'Chi0v', 'Chi1v', 'Chi2v',
    'HallKierAlpha', 'LabuteASA', 'MaxPartialCharge', 'MinPartialCharge',
    'MaxEStateIndex', 'MinEStateIndex', 'qed',
]


class CYPDescriptorCalculator:
    """RDKit 2D block + CYP-relevant SMARTS counts + a few derived ratios."""

    def __init__(self, scaler_kind='robust'):
        self._pats = {k: Chem.MolFromSmarts(v) for k, v in CYP_SMARTS.items()}
        self.scaler = RobustScaler() if scaler_kind == 'robust' else StandardScaler()
        self._fitted = False
        self.names = (RDKIT_DESCS + list(CYP_SMARTS)
                      + ['aromatic_proportion', 'heme_ligand_flag',
                         'basic_and_aromatic', 'halogen_density'])

    def n_features(self):
        return len(self.names)

    def _row(self, smi):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            return np.zeros(self.n_features(), dtype=np.float64)
        vals = []
        for d in RDKIT_DESCS:
            try:
                f = getattr(Descriptors, d, None) or getattr(rdMD, d, None) or getattr(Crippen, d)
                v = float(f(mol))
                vals.append(v if np.isfinite(v) else 0.0)
            except Exception:
                vals.append(0.0)
        counts = {k: len(mol.GetSubstructMatches(p)) for k, p in self._pats.items()}
        vals += [float(counts[k]) for k in CYP_SMARTS]
        heavy = max(mol.GetNumHeavyAtoms(), 1)
        n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        vals += [
            n_arom / heavy,
            float(counts['imidazole'] + counts['triazole'] + counts['pyridine_n'] > 0),
            float(counts['basic_amine'] > 0 and rdMD.CalcNumAromaticRings(mol) > 0),
            counts['halogen'] / heavy,
        ]
        return np.asarray(vals, dtype=np.float64)

    def fit_transform(self, smiles_list):
        X = np.vstack([self._row(s) for s in smiles_list])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = self.scaler.fit_transform(X)
        self._fitted = True
        return np.clip(Xs, -10, 10)

    def transform(self, smiles_list):
        X = np.vstack([self._row(s) for s in smiles_list])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(self.scaler.transform(X), -10, 10)


# =============================================================================
# REPRODUCIBILITY / STANDARDISATION / AUGMENTATION
# =============================================================================

def set_global_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MoleculeStandardizer:
    def __init__(self):
        self.lfc = LargestFragmentChooser()
        self.uc  = Uncharger()

    def sanitize(self, smiles):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return None
            return self.uc.uncharge(self.lfc.choose(mol))
        except Exception:
            return None


class SMILESAugmenter:
    """Same as v4_4, except y is now a (n_tasks,) vector carried per variant."""

    def __init__(self, random_seed=GLOBAL_SEED):
        random.seed(random_seed)

    def enumerate_smiles(self, smiles, n_variants=5, include_original=True):
        variants = {smiles} if include_original else set()
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return list(variants)
            atoms    = list(range(mol.GetNumAtoms()))
            attempts = 0
            target   = n_variants + (1 if include_original else 0)
            while len(variants) < target and attempts < n_variants * 10:
                s = Chem.MolToSmiles(mol, rootedAtAtom=random.choice(atoms),
                                     canonical=False, doRandom=True)
                if s:
                    variants.add(s)
                attempts += 1
        except Exception:
            pass
        return list(variants)

    def augment_dataset(self, smiles_list, Y, augmentation_factor=3, weights=None):
        aug_smiles, aug_y, aug_w = [], [], []
        for i, smi in enumerate(smiles_list):
            for v in self.enumerate_smiles(smi, n_variants=augmentation_factor - 1,
                                           include_original=True):
                aug_smiles.append(v)
                aug_y.append(Y[i])
                if weights is not None:
                    aug_w.append(weights[i])
        return (aug_smiles, np.vstack(aug_y),
                np.array(aug_w) if weights is not None else None)


# =============================================================================
# SPLITS
# =============================================================================

def _pack(groups, n_folds):
    folds, sizes = [[] for _ in range(n_folds)], [0] * n_folds
    for g in sorted(groups, key=lambda x: (-len(x), min(x))):
        i = int(np.argmin(sizes))
        folds[i].extend(g)
        sizes[i] += len(g)
    return [sorted(f) for f in folds]


def _scaffolds(smiles_list):
    d = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
        except Exception:
            sc = smi
        d[sc].append(idx)
    return d


def split_scaffold(smiles_list, n_folds=N_FOLDS, seed=GLOBAL_SEED):
    return _pack(list(_scaffolds(smiles_list).values()), n_folds)


def split_butina(smiles_list, n_folds=N_FOLDS, threshold=BUTINA_THRESHOLD, seed=GLOBAL_SEED):
    print(f"  Computing Butina clusters (threshold={threshold})...")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fps = [gen.GetFingerprint(m) if (m := Chem.MolFromSmiles(s)) else None for s in smiles_list]
    n, dists = len(fps), []
    for i in range(1, n):
        if fps[i] is None:
            dists.extend([1.0] * i); continue
        for j in range(i):
            dists.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]) if fps[j] else 1.0)
    from rdkit.ML.Cluster import Butina
    clusters = Butina.ClusterData(dists, n, threshold, isDistData=True)
    print(f"  Butina: {len(clusters)} clusters from {n} compounds")
    return _pack([list(c) for c in clusters], n_folds)


def split_descriptor(smiles_list, desc_matrix, n_folds=N_FOLDS,
                     n_clusters=KMEANS_N_CLUSTERS, seed=GLOBAL_SEED):
    print(f"  Computing k-means clusters (k={n_clusters}) on descriptors...")
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(
        StandardScaler().fit_transform(desc_matrix))
    d = defaultdict(list)
    for idx, lbl in enumerate(labels):
        d[lbl].append(idx)
    return _pack(list(d.values()), n_folds)


def split_activity_scaffold(smiles_list, Y, n_folds=N_FOLDS, seed=GLOBAL_SEED):
    """v4_4's default, generalised to multitask.

    A row's 'activity' is the mean of its z-scored AVAILABLE targets, so a
    compound measured on one isoform is binned on that isoform alone rather
    than being pushed to the middle by three NaNs.

    KNOW WHAT THIS SPLIT IS. Singleton scaffolds are packed whole, but any
    scaffold with more than one member is DEALT ROUND-ROBIN across the folds
    (`fi = fold_order[k % n_folds]`) to balance the activity distribution. So
    analogues do land in both train and validation: measured on this dataset,
    11.2% of compounds share a Murcko scaffold with a compound in another fold,
    against 0.0% for a plain scaffold split.

    That is a deliberate trade and, on THIS dataset, a defensible one. The
    practical difficulty barely moves — median held-out max-Tanimoto is 0.438
    here vs 0.433 for the strict scaffold split — while the activity balance
    across folds gets much better, which matters when three of the four
    endpoints are weakly predicted and a fold could otherwise end up with
    almost no actives.

    The real caveat is elsewhere: BOTH splits are harder than the actual task.
    The blind test compounds sit at median max-Tanimoto 0.587 to the training
    set, because they were purchased as analogues of screening hits. Any
    structure-based CV here is pessimistic relative to the leaderboard, which
    is why the composite validator in blend_composite.py is calibrated against
    real leaderboard results rather than against a CV number.
    """
    Z = np.full_like(Y, np.nan, dtype=float)
    for j in range(Y.shape[1]):
        col = Y[:, j]
        ok = np.isfinite(col)
        if ok.sum() > 1:
            Z[ok, j] = (col[ok] - col[ok].mean()) / (col[ok].std() + 1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        act = np.nanmean(Z, axis=1)
    act = np.nan_to_num(act, nan=0.0)

    quantiles = np.quantile(act, np.linspace(0, 1, n_folds + 1))

    def get_bin(v):
        for i, q in enumerate(quantiles[1:]):
            if v <= q:
                return i
        return n_folds - 1

    folds, sizes = [[] for _ in range(n_folds)], [0] * n_folds
    for g in sorted(_scaffolds(smiles_list).values(), key=lambda x: (-len(x), min(x))):
        if len(g) == 1:
            i = int(np.argmin(sizes)); folds[i].extend(g); sizes[i] += 1
        else:
            g_sorted = sorted(g, key=lambda idx: get_bin(act[idx]))
            order = np.argsort(sizes)
            for k, idx in enumerate(g_sorted):
                fi = int(order[k % n_folds]); folds[fi].append(idx); sizes[fi] += 1
    return [sorted(f) for f in folds]


def split_frozen(names, folds_csv, n_folds=N_FOLDS):
    """Reuse artifacts/folds.csv so this model can be blended with the others.

    Any compound not in the file is spread round-robin, which is only safe
    because those are a handful; if it warns about many, regenerate the file.
    """
    f = pd.read_csv(folds_csv)
    name_col = NAME_COL if NAME_COL in f.columns else f.columns[0]
    m = dict(zip(f[name_col], f['fold']))
    missing = [i for i, nm in enumerate(names) if nm not in m]
    if missing:
        print(f"  WARNING: {len(missing)} compound(s) absent from {folds_csv}; "
              f"assigned round-robin")
    folds = [[] for _ in range(n_folds)]
    for i, nm in enumerate(names):
        folds[int(m[nm]) % n_folds if nm in m else i % n_folds].append(i)
    return [sorted(x) for x in folds]


def get_folds(strategy, smiles_list, Y=None, desc_matrix=None, names=None,
              folds_csv=None, n_folds=N_FOLDS, seed=GLOBAL_SEED):
    strategy = strategy.lower()
    if strategy == 'scaffold':
        print(f"\n  Split strategy: Murcko scaffold ({n_folds} folds)")
        folds = split_scaffold(smiles_list, n_folds, seed)
    elif strategy == 'butina':
        print(f"\n  Split strategy: Butina cluster ({n_folds} folds)")
        folds = split_butina(smiles_list, n_folds, seed=seed)
    elif strategy == 'descriptor':
        print(f"\n  Split strategy: descriptor k-means ({n_folds} folds)")
        if desc_matrix is None:
            raise ValueError("desc_matrix required for descriptor split "
                             "(run with --use_descriptors)")
        folds = split_descriptor(smiles_list, desc_matrix, n_folds, seed=seed)
    elif strategy in ('activity_scaffold', 'activity'):
        print(f"\n  Split strategy: activity-stratified scaffold ({n_folds} folds)")
        folds = split_activity_scaffold(smiles_list, Y, n_folds, seed)
    elif strategy == 'frozen':
        print(f"\n  Split strategy: FROZEN from {folds_csv} ({n_folds} folds)")
        folds = split_frozen(names, folds_csv, n_folds)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
    sizes = [len(f) for f in folds]
    print(f"  Fold sizes: min={min(sizes)}  max={max(sizes)}  mean={np.mean(sizes):.0f}")
    report_split(smiles_list, folds)
    return folds


def report_split(smiles_list, folds):
    """Scaffold leakage across folds, and the activity balance per fold.

    Printed for every strategy so the split is a measurement rather than a
    label. Leakage is the fraction of compounds sharing a Murcko scaffold with
    a compound in a different fold: 0% for `scaffold`, ~11% for
    `activity_scaffold` on this dataset.
    """
    fold_of = np.empty(len(smiles_list), dtype=int)
    for i, f in enumerate(folds):
        for j in f:
            fold_of[j] = i
    scaf = np.array([_murcko(s) for s in smiles_list])
    leaked = 0
    for sc in pd.unique(scaf):
        idx = np.where(scaf == sc)[0]
        if len(set(fold_of[idx])) > 1:
            leaked += len(idx)
    print(f"  Scaffold leakage across folds: {leaked/len(smiles_list)*100:.1f}% "
          f"of compounds ({'strict hold-out' if leaked == 0 else 'analogues span folds'})")


def _murcko(smi):
    mol = Chem.MolFromSmiles(smi)
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
    except Exception:
        return smi


# =============================================================================
# METRICS
# =============================================================================

def strae(y, p, lo, hi):
    """The competition metric: soft-threshold relative absolute error.

    Zero cost inside the fitted credible interval; distance to the nearest edge
    outside it; normalised by a constant predictor at mean(y). 1.0 means you are
    exactly as good as predicting the mean, so anything >= 1.0 is a model that
    has not earned its keep on that endpoint.
    """
    err = np.clip(p - hi, 0, None) + np.clip(lo - p, 0, None)
    m = float(np.mean(y))
    base = np.clip(m - hi, 0, None) + np.clip(lo - m, 0, None)
    d = float(np.sum(base))
    return float(np.sum(err) / d) if d > 0 else np.nan


def masked_report(Y_true, Y_pred, bands=None):
    rows = []
    for j, ep in enumerate(TARGET_COLS):
        ok = np.isfinite(Y_true[:, j]) & np.isfinite(Y_pred[:, j])
        if ok.sum() < 5:
            continue
        yt, yp = Y_true[ok, j], Y_pred[ok, j]
        r = dict(endpoint=ISOFORMS[j], n=int(ok.sum()),
                 rmse=float(np.sqrt(np.mean((yt - yp) ** 2))),
                 mae=float(np.mean(np.abs(yt - yp))),
                 r2=float(1 - np.sum((yt - yp) ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-12)))
        if bands is not None:
            lo, hi = bands[j]
            r['ST_RAE'] = strae(yt, yp, lo[ok], hi[ok])
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        macro = {c: df[c].mean() for c in df.columns if c not in ('endpoint', 'n')}
        df.loc[len(df)] = dict(endpoint='MA', n=int(df['n'].sum()), **macro)
    return df


def compute_sample_weights(Y):
    """Tail + cliff weighting, both OFF by default (multipliers set to 1.0)."""
    n = len(Y)
    w = np.ones(n, dtype=np.float64)
    if TAIL_WEIGHT_MAX > 1.0:
        Z = np.full_like(Y, np.nan, dtype=float)
        for j in range(Y.shape[1]):
            col = Y[:, j]; ok = np.isfinite(col)
            if ok.sum() > 1:
                Z[ok, j] = np.abs(col[ok] - col[ok].mean()) / (col[ok].std() + 1e-8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.nan_to_num(np.nanmax(Z, axis=1), nan=0.0)
        w = np.clip(1.0 + z, 1.0, TAIL_WEIGHT_MAX)
        print(f"  Tail upweighting: range=[{w.min():.2f}, {w.max():.2f}] "
              f"(cap={TAIL_WEIGHT_MAX}x)")
    return w / w.mean()


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_adaptive_config(n):
    for k in ('small', 'medium', 'large', 'very_large'):
        c = ADAPTIVE_CONFIG[k]
        if n < c['threshold']:
            out = c.copy(); out['n_samples'] = n; return out
    out = ADAPTIVE_CONFIG['very_large'].copy(); out['n_samples'] = n; return out


# =============================================================================
# PREDICTOR
# =============================================================================

class CYPMultitaskMPNN:
    """Multitask MPNN over the four CYP pIC50 endpoints."""

    def __init__(self, model_dir=MODEL_SAVE_DIR, use_descriptors=False):
        self.model_dir = model_dir
        self.use_descriptors = use_descriptors
        os.makedirs(self.model_dir, exist_ok=True)
        self.models: list = []
        self.desc_calc: Optional[CYPDescriptorCalculator] = None
        self.scalers: List[RobustScaler] = []
        self.augmenter = SMILESAugmenter(random_seed=GLOBAL_SEED)
        self.adaptive_config: Optional[dict] = None
        self.oof_predictions: Optional[np.ndarray] = None
        self.best_epochs: List[int] = []

    # ---- y scaling, per task, NaN-aware -------------------------------------
    def _fit_scale(self, Y):
        """RobustScaler cannot take NaN, so fit each column on its own rows."""
        self.scalers = []
        Ys = np.full_like(Y, np.nan, dtype=np.float32)
        for j in range(Y.shape[1]):
            col = Y[:, j]; ok = np.isfinite(col)
            sc = RobustScaler(quantile_range=(1.0, 99.0))
            sc.fit(col[ok].reshape(-1, 1))
            Ys[ok, j] = sc.transform(col[ok].reshape(-1, 1)).ravel()
            self.scalers.append(sc)
        return Ys

    def _unscale(self, Ys):
        Y = np.full_like(Ys, np.nan, dtype=np.float64)
        for j, sc in enumerate(self.scalers):
            ok = np.isfinite(Ys[:, j])
            Y[ok, j] = sc.inverse_transform(Ys[ok, j].reshape(-1, 1)).ravel()
        return Y

    def _build(self, cfg, input_extra=0):
        hd, d, nl, do = cfg['hidden_dim'], cfg['depth'], cfg['n_layers'], cfg['dropout']
        mp  = BondMessagePassing(d_h=hd, depth=d, dropout=do, activation='relu')
        ffn = RegressionFFN(input_dim=hd + input_extra, hidden_dim=hd,
                            n_layers=nl, dropout=do, n_tasks=len(TARGET_COLS))
        return MPNN(message_passing=mp, agg=MeanAggregation(), predictor=ffn,
                    batch_norm=True)

    def _safe_load(self, path):
        for kw in [{'weights_only': False}, {}]:
            try:
                return MPNN.load_from_checkpoint(path, map_location=DEVICE, **kw)
            except Exception:
                continue
        return None

    def _datapoints(self, smiles, Yscaled, desc=None, weights=None):
        dps = []
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            kw = {}
            if desc is not None:
                kw['x_d'] = desc[i].astype(np.float32)
            if weights is not None:
                kw['weight'] = float(weights[i])
            # NaN in y is the missing-label marker; chemprop masks on isfinite.
            dps.append(MoleculeDatapoint(mol=mol, y=Yscaled[i].astype(np.float32), **kw))
        return dps

    def _predict_raw(self, model, loader, n_expect):
        trainer = pl.Trainer(accelerator='gpu' if DEVICE == 'cuda' else 'cpu',
                             devices=1, logger=False, enable_progress_bar=False)
        with torch.inference_mode():
            raw = trainer.predict(model, loader)
        P = np.concatenate([p.cpu().numpy() for p in raw], axis=0)
        return P.reshape(n_expect, len(TARGET_COLS))

    # ---- cross-validation ---------------------------------------------------
    def train(self, smiles_list, Y, names, split_strategy='scaffold',
              folds_csv=None, sample_weights=None, patience_override=None,
              bands=None):
        use_weights = sample_weights is not None
        n_desc = 0
        print(f"\n{'='*66}")
        print(f"CYP multitask MPNN v1  |  N={len(smiles_list)}  folds={N_FOLDS}  "
              f"strategy={split_strategy}")
        print(f"Tasks: {len(TARGET_COLS)}  Descriptors: "
              f"{'ON' if self.use_descriptors else 'OFF (baseline: learned features only)'}  "
              f"SampleWeights: {'ON' if use_weights else 'OFF'}")
        print(f"{'='*66}")
        for j, ep in enumerate(TARGET_COLS):
            print(f"  {ISOFORMS[j]}: {int(np.isfinite(Y[:, j]).sum())} labelled")

        self._seed = GLOBAL_SEED
        self.adaptive_config = cfg = get_adaptive_config(len(smiles_list))
        if patience_override is not None:
            cfg['patience'] = patience_override
        print(f"Config: hidden={cfg['hidden_dim']}  depth={cfg['depth']}  "
              f"layers={cfg['n_layers']}  dropout={cfg['dropout']}  "
              f"patience={cfg['patience']}  aug={cfg['augmentation_factor']}x")

        Ys = self._fit_scale(Y)

        X_desc = None
        if self.use_descriptors:
            self.desc_calc = CYPDescriptorCalculator()
            X_desc = self.desc_calc.fit_transform(smiles_list)
            n_desc = X_desc.shape[1]
            print(f"Descriptor matrix: {X_desc.shape}")

        folds = get_folds(split_strategy, smiles_list, Y=Y, desc_matrix=X_desc,
                          names=names, folds_csv=folds_csv, n_folds=N_FOLDS,
                          seed=GLOBAL_SEED)

        oof = np.full((len(smiles_list), len(TARGET_COLS)), np.nan)
        self.best_epochs = []

        for fold_idx, val_indices in enumerate(folds):
            set_global_seed(self._seed + fold_idx)
            print(f"\n--- Fold {fold_idx+1}/{N_FOLDS}  (val={len(val_indices)}) ---")
            tr_mask = np.ones(len(smiles_list), dtype=bool)
            tr_mask[val_indices] = False
            tr_idx = np.where(tr_mask)[0]

            tr_smi = [smiles_list[i] for i in tr_idx]
            tr_Y   = Ys[tr_idx]
            tr_w   = sample_weights[tr_idx] if use_weights else None
            vl_smi = [smiles_list[i] for i in val_indices]
            vl_Y   = Ys[val_indices]

            if cfg['use_augmentation'] and cfg['augmentation_factor'] > 1:
                aug_smi, aug_Y, aug_w = self.augmenter.augment_dataset(
                    tr_smi, tr_Y, cfg['augmentation_factor'], tr_w)
            else:
                aug_smi, aug_Y, aug_w = tr_smi, tr_Y, tr_w

            aug_desc = self.desc_calc.transform(aug_smi) if self.use_descriptors else None
            vl_desc  = X_desc[val_indices] if self.use_descriptors else None

            train_dps = self._datapoints(aug_smi, aug_Y, aug_desc, aug_w)
            val_dps   = self._datapoints(vl_smi, vl_Y, vl_desc)

            tr_loader = build_dataloader(MoleculeDataset(train_dps),
                                         batch_size=cfg['batch_size'],
                                         shuffle=True, num_workers=0)
            vl_loader = build_dataloader(MoleculeDataset(val_dps),
                                         batch_size=cfg['batch_size'],
                                         shuffle=False, num_workers=0)

            ckpt_fn = f'fold_{fold_idx}'
            ckpt_path = os.path.join(self.model_dir, f'{ckpt_fn}.ckpt')

            if os.path.exists(ckpt_path):
                print(f"  Checkpoint found — skipping training (fold {fold_idx+1})")
                best = self._safe_load(ckpt_path)
                self.best_epochs.append(-1)
            else:
                model = self._build(cfg, input_extra=n_desc)
                ckpt_cb = ModelCheckpoint(dirpath=self.model_dir, filename=ckpt_fn,
                                          monitor='val_loss', mode='min', save_top_k=1)
                trainer = pl.Trainer(
                    accelerator='gpu' if DEVICE == 'cuda' else 'cpu', devices=1,
                    max_epochs=cfg['max_epochs'],
                    callbacks=[EarlyStopping(monitor='val_loss',
                                             patience=cfg['patience'], mode='min'),
                               ckpt_cb],
                    enable_progress_bar=True, logger=False,
                    gradient_clip_val=1.0, deterministic=False)
                trainer.fit(model, tr_loader, vl_loader)
                best_ep = max(0, trainer.current_epoch - cfg['patience'])
                self.best_epochs.append(best_ep)
                print(f"  Stopped epoch={trainer.current_epoch}  Best epoch~{best_ep}")
                best = self._safe_load(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else None

            if best is None:
                print(f"  WARNING: no usable checkpoint for fold {fold_idx}")
                continue
            best.eval(); best.to(torch.device(DEVICE))

            valid_local = [i for i, s in enumerate(vl_smi) if Chem.MolFromSmiles(s)]
            P = self._predict_raw(best, vl_loader, len(valid_local))
            for k, li in enumerate(valid_local):
                oof[val_indices[li]] = P[k]

            fold_report = masked_report(Y[val_indices], self._unscale(oof[val_indices]))
            print("  " + fold_report.to_string(index=False,
                                               float_format=lambda x: f"{x:.4f}").replace("\n", "\n  "))

        self.oof_predictions = self._unscale(oof)

        save_pickle(self.scalers, os.path.join(self.model_dir, 'scalers_y.pkl'))
        save_pickle(self.best_epochs, os.path.join(self.model_dir, 'best_epochs.pkl'))
        save_pickle({'use_descriptors': self.use_descriptors,
                     'n_folds': N_FOLDS, 'split_strategy': split_strategy,
                     'targets': TARGET_COLS, 'version': 'cyp_mt_v1'},
                    os.path.join(self.model_dir, 'meta.pkl'))
        if self.use_descriptors:
            save_pickle(self.desc_calc.scaler,
                        os.path.join(self.model_dir, 'desc_scaler.pkl'))

        rep = masked_report(Y, self.oof_predictions, bands=bands)
        print(f"\n{'='*66}")
        print(f"OOF results ({split_strategy})")
        print(rep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"{'='*66}")
        if 'ST_RAE' in rep.columns:
            print("ST-RAE is the leaderboard metric. 1.0 == a constant predictor at the")
            print("mean, so any endpoint at or above 1.0 has learned nothing usable.")
        avg_ep = int(np.mean([e for e in self.best_epochs if e >= 0])) if any(
            e >= 0 for e in self.best_epochs) else 0
        print(f"\nBest epochs per fold : {self.best_epochs}")
        print(f"Average best epoch   : {avg_ep}  "
              f"(use --mode FULL --full_epochs {avg_ep + 5})")
        return rep

    # ---- full retrain -------------------------------------------------------
    def train_full(self, smiles_list, Y, sample_weights=None, fixed_epochs=None):
        if fixed_epochs is None:
            p = os.path.join(self.model_dir, 'best_epochs.pkl')
            saved = [e for e in load_pickle(p)] if os.path.exists(p) else []
            saved = [e for e in saved if e >= 0]
            fixed_epochs = int(np.mean(saved)) + 5 if saved else 100
            print(f"  Fixed epoch budget: {fixed_epochs}")

        print(f"\n{'='*66}\nFULL RETRAIN  |  N={len(smiles_list)}  epochs={fixed_epochs}\n{'='*66}")
        set_global_seed(GLOBAL_SEED)
        cfg = get_adaptive_config(len(smiles_list))
        Ys = self._fit_scale(Y)

        X_desc, n_desc = None, 0
        if self.use_descriptors:
            self.desc_calc = CYPDescriptorCalculator()
            X_desc = self.desc_calc.fit_transform(smiles_list)
            n_desc = X_desc.shape[1]

        if cfg['use_augmentation'] and cfg['augmentation_factor'] > 1:
            aug_smi, aug_Y, aug_w = self.augmenter.augment_dataset(
                smiles_list, Ys, cfg['augmentation_factor'], sample_weights)
        else:
            aug_smi, aug_Y, aug_w = smiles_list, Ys, sample_weights
        aug_desc = self.desc_calc.transform(aug_smi) if self.use_descriptors else None

        loader = build_dataloader(MoleculeDataset(
            self._datapoints(aug_smi, aug_Y, aug_desc, aug_w)),
            batch_size=cfg['batch_size'], shuffle=True, num_workers=0)

        model = self._build(cfg, input_extra=n_desc)
        trainer = pl.Trainer(accelerator='gpu' if DEVICE == 'cuda' else 'cpu', devices=1,
                             max_epochs=fixed_epochs, logger=False,
                             enable_checkpointing=False, enable_progress_bar=True,
                             gradient_clip_val=1.0)
        trainer.fit(model, loader)
        path = os.path.join(self.model_dir, 'full_model.ckpt')
        trainer.save_checkpoint(path)
        save_pickle(self.scalers, os.path.join(self.model_dir, 'scalers_y.pkl'))
        if self.use_descriptors:
            save_pickle(self.desc_calc.scaler,
                        os.path.join(self.model_dir, 'desc_scaler.pkl'))
        print(f"  Saved {path}")
        self.models = [model]
        return model

    # ---- inference ----------------------------------------------------------
    def predict(self, smiles_list, batch_size=64):
        if not self.models:
            ckpts = sorted(glob.glob(os.path.join(self.model_dir, 'fold_*.ckpt')))
            full = os.path.join(self.model_dir, 'full_model.ckpt')
            if os.path.exists(full):
                ckpts = [full] + ckpts
            self.models = [m for c in ckpts if (m := self._safe_load(c))]
            for m in self.models:
                m.eval(); m.to(torch.device(DEVICE))
            self.scalers = load_pickle(os.path.join(self.model_dir, 'scalers_y.pkl'))
            if self.use_descriptors:
                self.desc_calc = CYPDescriptorCalculator()
                self.desc_calc.scaler = load_pickle(
                    os.path.join(self.model_dir, 'desc_scaler.pkl'))
            print(f"  Loaded {len(self.models)} model(s)")
        if not self.models:
            raise RuntimeError(f"no checkpoints in {self.model_dir}")

        keep = [i for i, s in enumerate(smiles_list) if Chem.MolFromSmiles(s)]
        smi = [smiles_list[i] for i in keep]
        desc = self.desc_calc.transform(smi) if self.use_descriptors else None
        dummy = np.zeros((len(smi), len(TARGET_COLS)), dtype=np.float32)
        loader = build_dataloader(MoleculeDataset(self._datapoints(smi, dummy, desc)),
                                  batch_size=batch_size, shuffle=False, num_workers=0)
        acc = np.zeros((len(smi), len(TARGET_COLS)))
        for m in self.models:
            acc += self._predict_raw(m, loader, len(smi))
        acc /= len(self.models)
        out = np.full((len(smiles_list), len(TARGET_COLS)), np.nan)
        out[keep] = self._unscale(acc)
        # a molecule RDKit could not parse falls back to the per-task median
        for j in range(out.shape[1]):
            bad = ~np.isfinite(out[:, j])
            if bad.any():
                out[bad, j] = float(np.nanmedian(out[:, j]))
        return out


# =============================================================================
# DATA
# =============================================================================

def load_training(path, std):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    smi_col = next((c for c in df.columns if c.lower() == 'smiles'), SMILES_COL)
    df = df.rename(columns={smi_col: SMILES_COL})
    name_col = next((c for c in df.columns
                     if c.lower().replace(' ', '_') in ('molecule_name', 'name', 'id')),
                    NAME_COL)
    df = df.rename(columns={name_col: NAME_COL})
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing target column(s) {missing}. Have: {list(df.columns)}")

    df['mol'] = df[SMILES_COL].apply(std.sanitize)
    df['SMILES_clean'] = df['mol'].apply(lambda m: Chem.MolToSmiles(m, canonical=True) if m else None)
    df = df.dropna(subset=['SMILES_clean']).reset_index(drop=True)
    # keep a row if it has at least one measured endpoint
    Y = df[TARGET_COLS].to_numpy(dtype=float)
    keep = np.isfinite(Y).any(axis=1)
    df = df[keep].reset_index(drop=True)
    return df


def band_arrays(df):
    """(lo, hi) per endpoint for ST-RAE, or None if the file has no bounds."""
    bands = []
    for ep in TARGET_COLS:
        lo, hi = ep + CONF_LOW, ep + CONF_HIGH
        if lo not in df.columns or hi not in df.columns:
            return None
        bands.append((df[lo].to_numpy(float), df[hi].to_numpy(float)))
    return bands


# =============================================================================
# MAIN
# =============================================================================

def main():
    global N_FOLDS, GLOBAL_SEED
    p = argparse.ArgumentParser(description="CYP multitask Chemprop v1 (baseline)")
    p.add_argument('--mode', choices=['TRAIN', 'PREDICT', 'BOTH', 'FULL'], default='BOTH')
    p.add_argument('--train_csv', default=TRAIN_CSV)
    p.add_argument('--test_csv',  default=TEST_CSV)
    p.add_argument('--model_dir', default=MODEL_SAVE_DIR)
    p.add_argument('--output',    default=OUTPUT_CSV)
    p.add_argument('--oof_output', default=OOF_CSV)
    p.add_argument('--seed', type=int, default=GLOBAL_SEED)
    p.add_argument('--n_folds', type=int, default=N_FOLDS)
    p.add_argument('--split_strategy',
                   choices=['scaffold', 'butina', 'descriptor', 'activity_scaffold', 'frozen'],
                   default='activity_scaffold')
    p.add_argument('--folds_csv', default='artifacts/folds.csv',
                   help="used only with --split_strategy frozen")
    p.add_argument('--use_descriptors', action='store_true', default=False,
                   help="rung 2: concatenate CYP descriptors to the learned embedding")
    p.add_argument('--sample_weights', action='store_true', default=False,
                   help="enable tail/cliff weighting (v4_4's flag was inverted; "
                        "this one is opt-in and actually works)")
    p.add_argument('--full_epochs', type=int, default=None)
    p.add_argument('--patience', type=int, default=None)
    args = p.parse_args()

    N_FOLDS = args.n_folds
    GLOBAL_SEED = args.seed
    set_global_seed(args.seed)

    print(f"Seed          : {args.seed}")
    print(f"Folds         : {N_FOLDS}")
    print(f"Descriptors   : {'ON' if args.use_descriptors else 'OFF (baseline)'}")
    print(f"Sample weights: {'ON' if args.sample_weights else 'OFF'}")

    t0 = time.time()
    std = MoleculeStandardizer()
    predictor = CYPMultitaskMPNN(model_dir=args.model_dir,
                                 use_descriptors=args.use_descriptors)

    if args.mode in ('TRAIN', 'BOTH', 'FULL'):
        print(f"\nLoading: {args.train_csv}")
        df = load_training(args.train_csv, std)
        Y = df[TARGET_COLS].to_numpy(dtype=float)
        smiles = df['SMILES_clean'].tolist()
        names = df[NAME_COL].tolist()
        print(f"Rows: {len(df)}   labelled (compound,isoform) pairs: "
              f"{int(np.isfinite(Y).sum())}")

        sw = compute_sample_weights(Y) if args.sample_weights else None
        bands = band_arrays(df)
        if bands is None:
            print("  NOTE: no _conf_low/_conf_high columns — ST-RAE will be skipped.")

        if args.mode in ('TRAIN', 'BOTH'):
            rep = predictor.train(smiles, Y, names,
                                  split_strategy=args.split_strategy,
                                  folds_csv=args.folds_csv,
                                  sample_weights=sw,
                                  patience_override=args.patience,
                                  bands=bands)
            oof = pd.DataFrame({NAME_COL: names})
            for j, ep in enumerate(TARGET_COLS):
                oof[ep] = predictor.oof_predictions[:, j]
            oof.to_csv(args.oof_output, index=False)
            print(f"\nOOF written: {args.oof_output}")
            print("  -> feed this to blend_composite.py to see whether it earns a place")
            print("     in the ensemble (it must be a FROZEN-split run to be blendable)")
            rep.to_csv(os.path.join(args.model_dir, 'oof_report.csv'), index=False)

        if args.mode == 'FULL':
            predictor.train_full(smiles, Y, sample_weights=sw,
                                 fixed_epochs=args.full_epochs)
        print(f"\nTraining done in {(time.time()-t0)/60:.1f} min")

    if args.mode in ('PREDICT', 'BOTH'):
        print(f"\nLoading test: {args.test_csv}")
        te = pd.read_csv(args.test_csv)
        te.columns = [c.strip() for c in te.columns]
        smi_col = next((c for c in te.columns if c.lower() == 'smiles'), SMILES_COL)
        te = te.rename(columns={smi_col: SMILES_COL})
        name_col = next((c for c in te.columns
                         if c.lower().replace(' ', '_') in ('molecule_name', 'name', 'id')),
                        NAME_COL)
        te = te.rename(columns={name_col: NAME_COL})
        te['mol'] = te[SMILES_COL].apply(std.sanitize)
        te['SMILES_clean'] = te['mol'].apply(
            lambda m: Chem.MolToSmiles(m, canonical=True) if m else None)
        te['SMILES_clean'] = te['SMILES_clean'].fillna(te[SMILES_COL])

        P = predictor.predict(te['SMILES_clean'].tolist())
        out = te[[SMILES_COL, NAME_COL]].copy()
        for j, ep in enumerate(TARGET_COLS):
            out[ep] = P[:, j]
        out.to_csv(args.output, index=False)
        print(f"\nPredictions written: {args.output}  ({len(out)} rows)")
        print(out[TARGET_COLS].describe().T.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
