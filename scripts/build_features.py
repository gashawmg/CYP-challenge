#!/usr/bin/env python
r"""Compute every molecular representation ONCE and cache it to disk.

    python scripts/build_features.py --probe          # what can this env compute?
    python scripts/build_features.py                  # build everything available
    python scripts/build_features.py --blocks rdkit2d morgan ionisation
    python scripts/build_features.py --force          # recompute, ignoring the cache

WHY THIS EXISTS
---------------
Smbatyan et al. (bioRxiv 2026.10.01.755314) placed 4th/104 on ExpansionRx and 1st/104 on
PXR. Their solutions did NOT fine-tune graph networks as the primary model. They fed
FROZEN pretrained encoders and large descriptor banks into small tabular learners -- TabM,
TabPFN, TabICL, kernel regressors -- and ensembled those.

That matters here for a practical reason as much as a scientific one. Fine-tuning chemprop
costs 10-25 minutes per endpoint per fold on this project's GPU, because the encoder is
retrained every time. Frozen embeddings are computed once, cached, and then a tabular head
trains in seconds. The same hardware goes from testing one configuration an evening to
testing dozens -- which is the difference between an agent loop being theoretical and
being practical.

WHAT IT WILL NOT DO
-------------------
It will not install anything. Blocks whose packages are missing are SKIPPED, named, and
the install command is printed for you to run deliberately or not at all. Nothing here
touches the environment. Run --probe first to see what you already have; the core blocks
need only RDKit and cover a large part of what the paper's solutions used.

OUTPUT
------
    features/<block>.npy        float32, one row per compound, in index order
    features/<block>.cols.json  column names
    features/index.csv          Molecule_Name, SMILES, split  -- the row order for ALL blocks
    features/manifest.json      what was built, when, with what shape

Every block shares features/index.csv row-for-row, so blocks concatenate without a merge.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_CSV = "data/cyp-challenge-TRAIN_inhibition.csv"
TEST_CSV = "data/cyp-challenge-TEST-BLINDED.csv"
OUT = Path("features")
ID_COL, SMILES_COL = "Molecule_Name", "SMILES"


# --------------------------------------------------------------------------------------
# Blocks. Each returns (matrix, column_names) or raises ImportError with the pip command.
# --------------------------------------------------------------------------------------
def block_rdkit2d(smiles, a):
    """All ~217 RDKit 2D descriptors. The paper's single most-used tabular input."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    names = [n for n, _ in Descriptors.descList]
    fns = [f for _, f in Descriptors.descList]
    X = np.full((len(smiles), len(fns)), np.nan, np.float64)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        for j, f in enumerate(fns):
            try:
                X[i, j] = f(m)
            except Exception:                                        # noqa: BLE001
                pass
    return X, names


def block_morgan(smiles, a):
    """Morgan, feature-Morgan, atom-pair and topological-torsion COUNTS.

    Counts rather than bits, and log1p-compressed, because the paper's kernel and tabular
    components use counts clipped and transformed this way -- a bit vector throws away how
    many times a substructure occurs, which is exactly what distinguishes a mono- from a
    tri-substituted ring.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator as fg
    RDLogger.DisableLog("rdApp.*")
    gens = [("mg2", fg.GetMorganGenerator(radius=2, fpSize=a.fp_bits)),
            ("mg3", fg.GetMorganGenerator(radius=3, fpSize=a.fp_bits)),
            ("fmg2", fg.GetMorganGenerator(
                radius=2, fpSize=a.fp_bits,
                atomInvariantsGenerator=fg.GetMorganFeatureAtomInvGen())),
            ("ap", fg.GetAtomPairGenerator(fpSize=a.fp_bits)),
            ("tt", fg.GetTopologicalTorsionGenerator(fpSize=a.fp_bits))]
    cols, mats = [], []
    for tag, g in gens:
        M = np.zeros((len(smiles), a.fp_bits), np.float32)
        for i, s in enumerate(smiles):
            m = Chem.MolFromSmiles(str(s))
            if m is None:
                continue
            for k, v in g.GetCountFingerprint(m).GetNonzeroElements().items():
                M[i, k] = v
        mats.append(np.log1p(np.clip(M, 0, 255)))
        cols += [f"{tag}_{k}" for k in range(a.fp_bits)]
    return np.hstack(mats), cols


def block_ionisation(smiles, a):
    """Ionisation state across pH, the paper's '153 rule-derived ionization features'.

    This is the single biggest determinant of CYP2C9 (anion-preferring, Arg108) versus
    CYP2D6 (cation-preferring, Asp301) selectivity, and the project's existing descriptor
    set carries only four SMARTS counts for it. Without a trained pKa model the honest
    substitute is a richer SMARTS census plus Crippen/TPSA context, evaluated so that a
    downstream model can learn the pH dependence rather than having it hard-coded.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors
    RDLogger.DisableLog("rdApp.*")
    PATTERNS = {
        "amine_prim": "[NX3;H2;!$(N[a]);!$(NC=[O,S,N]);!$(N[SX4](=O)=O)]",
        "amine_sec": "[NX3;H1;!$(N[a]);!$(NC=[O,S,N]);!$(N[SX4](=O)=O)]",
        "amine_tert": "[NX3;H0;!$(N[a]);!$(NC=[O,S,N]);!$(N[SX4](=O)=O);!$([N+])]",
        "aniline": "[NX3;$(N[a])]",
        "amidine": "[NX3][CX3]=[NX2]",
        "guanidine": "NC(=N)N",
        "pyridine_n": "[n;X2;!$(n[#6]=O)]",
        "imidazole": "c1cnc[nH]1",
        "tetrazole": "c1nnn[nH]1",
        "triazole": "c1ncnn1",
        "carboxylic_acid": "[CX3](=O)[OX2H1]",
        "sulfonamide_nh": "[SX4](=O)(=O)[NX3H]",
        "sulfonic_acid": "[SX4](=O)(=O)[OX2H1]",
        "phosphate": "[PX4](=O)([OX2H1])",
        "phenol": "[OX2H][c]",
        "alcohol": "[OX2H][CX4]",
        "thiol": "[SX2H]",
        "imide": "[NX3H1](C=O)C=O",
        "quaternary_n": "[NX4+]",
        "nitro": "[N+](=O)[O-]",
    }
    pats = {k: Chem.MolFromSmarts(v) for k, v in PATTERNS.items()}
    # representative pKa values; crude, and the model is free to reweight them
    PKA = {"amine_prim": 10.5, "amine_sec": 10.8, "amine_tert": 9.8, "aniline": 4.6,
           "amidine": 11.5, "guanidine": 12.5, "pyridine_n": 5.2, "imidazole": 7.0,
           "quaternary_n": 99.0}
    ACID = {"carboxylic_acid": 4.2, "tetrazole": 4.9, "sulfonamide_nh": 10.0,
            "sulfonic_acid": -1.0, "phosphate": 2.0, "phenol": 10.0, "thiol": 8.5,
            "imide": 9.5}
    pHs = [5.5, 6.5, 7.4, 9.0]

    cols = list(PATTERNS) + ["crippen_logp", "tpsa", "mw", "n_heavy"]
    for pH in pHs:
        cols += [f"frac_pos_pH{pH}", f"frac_neg_pH{pH}", f"net_charge_pH{pH}",
                 f"logd_est_pH{pH}", f"zwitterion_pH{pH}"]
    X = np.zeros((len(smiles), len(cols)), np.float64)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            X[i] = np.nan; continue
        counts = {k: len(m.GetSubstructMatches(p)) for k, p in pats.items()}
        row = [float(counts[k]) for k in PATTERNS]
        try:
            logp = float(Crippen.MolLogP(m)); tpsa = float(Descriptors.TPSA(m))
            mw = float(Descriptors.MolWt(m)); nh = float(m.GetNumHeavyAtoms())
        except Exception:                                            # noqa: BLE001
            logp = tpsa = mw = nh = np.nan
        row += [logp, tpsa, mw, nh]
        for pH in pHs:
            # Henderson-Hasselbalch per site, summed
            pos = sum(counts[k] / (1 + 10 ** (pH - v)) for k, v in PKA.items())
            neg = sum(counts[k] / (1 + 10 ** (v - pH)) for k, v in ACID.items())
            n_b = sum(counts[k] for k in PKA) or 1
            n_a = sum(counts[k] for k in ACID) or 1
            net = pos - neg
            # logD falls from logP by roughly the ionised fraction
            logd = logp - np.log10(1 + 10 ** abs(net)) if np.isfinite(logp) else np.nan
            row += [pos / n_b, neg / n_a, net, logd,
                    float(pos > 0.5 and neg > 0.5)]
        X[i] = row
    return X, cols


def block_mordred(smiles, a):
    """~1,600 Mordred 2D descriptors. Slow (minutes) but it is a one-time cost."""
    try:
        from mordred import Calculator, descriptors
    except ImportError as e:
        raise ImportError("pip install mordredcommunity   (NOT `mordred` -- the\n                           original is abandoned and breaks on numpy>=1.24; the\n                           community fork keeps the import name `mordred`)") from e
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(str(s)) for s in smiles]
    ok = [m if m is not None else Chem.MolFromSmiles("C") for m in mols]
    df = calc.pandas(ok, quiet=True, nproc=a.nproc)
    X = df.apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    for i, m in enumerate(mols):
        if m is None:
            X[i] = np.nan
    return X, [str(c) for c in df.columns]


def block_chemberta(smiles, a):
    """Frozen ChemBERTa embedding (mean-pooled). Transfer representation, not fine-tuned."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError("pip install transformers") from e
    name = a.chemberta
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(dev)
    outs = []
    with torch.inference_mode():
        for i in range(0, len(smiles), a.batch):
            b = [str(s) for s in smiles[i:i + a.batch]]
            enc = tok(b, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(dev)
            h = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            outs.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
    X = np.vstack(outs)
    return X, [f"chemberta_{i}" for i in range(X.shape[1])]


def block_unimol(smiles, a):
    """Frozen Uni-Mol / Uni-Mol2 molecular embedding."""
    try:
        from unimol_tools import UniMolRepr
    except ImportError as e:
        raise ImportError("pip install unimol_tools  (this project already uses Uni-Mol; "
                          "check the admet_gpu env before installing anything)") from e
    rep = UniMolRepr(data_type="molecule", remove_hs=False)
    out = rep.get_repr([str(s) for s in smiles], return_atomic_reprs=False)
    X = np.asarray(out["cls_repr"], np.float64)
    return X, [f"unimol_{i}" for i in range(X.shape[1])]


BLOCKS = {
    "rdkit2d": (block_rdkit2d, "~217 RDKit 2D descriptors", "core"),
    "ionisation": (block_ionisation, "ionisation census + Henderson-Hasselbalch at "
                                     "pH 5.5/6.5/7.4/9.0", "core"),
    "morgan": (block_morgan, "Morgan r2/r3, feature-Morgan, atom-pair, torsion COUNTS",
               "core"),
    "mordred": (block_mordred, "~1,613 Mordred 2D descriptors "
                               "(package: mordredcommunity)", "mordred"),
    "chemberta": (block_chemberta, "frozen ChemBERTa mean-pooled embedding",
                  "transformers"),
    "unimol": (block_unimol, "frozen Uni-Mol molecular embedding", "unimol_tools"),
}


# --------------------------------------------------------------------------------------
def build_index(a):
    tr = pd.read_csv(TRAIN_CSV); tr.columns = [c.strip() for c in tr.columns]
    te = pd.read_csv(TEST_CSV); te.columns = [c.strip() for c in te.columns]
    idx = pd.concat([tr[[ID_COL, SMILES_COL]].assign(split="train"),
                     te[[ID_COL, SMILES_COL]].assign(split="blind")], ignore_index=True)
    # a compound appearing in both files would silently duplicate rows downstream
    dup = idx[ID_COL].duplicated().sum()
    if dup:
        print(f"  NOTE {dup} duplicate ids across train/blind; keeping both rows, "
              f"distinguished by the split column")
    return idx


def probe():
    print(f"{'block':14s}{'needs':16s}{'status':10s}  what it is")
    print("-" * 92)
    for name, (_, desc, req) in BLOCKS.items():
        if req == "core":
            ok, note = True, "ready"
        else:
            try:
                __import__(req); ok, note = True, "ready"
            except ImportError:
                ok, note = False, "MISSING"
        print(f"{name:14s}{req:16s}{note:10s}  {desc}")
    print("\nNothing is installed by this script. Core blocks need only RDKit and already")
    print("cover descriptors, ionisation and count fingerprints. For a MISSING block,")
    print("install it yourself only if you want it -- and check the admet_gpu env first.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blocks", nargs="*", default=None,
                   help=f"subset of {list(BLOCKS)}; default is every available block")
    p.add_argument("--probe", action="store_true", help="report availability and exit")
    p.add_argument("--force", action="store_true", help="recompute cached blocks")
    p.add_argument("--fp_bits", type=int, default=2048)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--chemberta", default="DeepChem/ChemBERTa-77M-MLM")
    a = p.parse_args()

    if a.probe:
        return probe()

    OUT.mkdir(exist_ok=True)
    idx = build_index(a)
    idx.to_csv(OUT / "index.csv", index=False)
    smiles = idx[SMILES_COL].tolist()
    print(f"index: {len(idx)} compounds "
          f"({(idx.split == 'train').sum()} train + {(idx.split == 'blind').sum()} blind)"
          f"  -> {OUT / 'index.csv'}\n")

    want = a.blocks or list(BLOCKS)
    manifest = {}
    mf = OUT / "manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text())

    for name in want:
        if name not in BLOCKS:
            print(f"{name:14s} UNKNOWN block, skipped"); continue
        fn, desc, req = BLOCKS[name]
        npy = OUT / f"{name}.npy"
        if npy.exists() and not a.force:
            X = np.load(npy, mmap_mode="r")
            if X.shape[0] == len(idx):
                print(f"{name:14s} cached   {X.shape}   (--force to rebuild)")
                continue
            print(f"{name:14s} cached file has {X.shape[0]} rows but the index has "
                  f"{len(idx)}; rebuilding")
        t0 = time.time()
        try:
            X, cols = fn(smiles, a)
        except ImportError as e:
            print(f"{name:14s} SKIPPED  needs `{e}`"); continue
        except Exception as e:                                       # noqa: BLE001
            print(f"{name:14s} FAILED   {type(e).__name__}: {e}"); continue

        X = np.asarray(X, np.float32)
        bad = int(np.isnan(X).all(axis=1).sum())
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        np.save(npy, X)
        (OUT / f"{name}.cols.json").write_text(json.dumps(cols))
        manifest[name] = dict(shape=list(X.shape), built=time.strftime("%Y-%m-%d %H:%M"),
                              seconds=round(time.time() - t0, 1), description=desc,
                              unparseable_rows=bad)
        print(f"{name:14s} built    {X.shape}  in {time.time() - t0:.1f}s"
              + (f"   ({bad} unparseable molecules zero-filled)" if bad else ""))

    mf.write_text(json.dumps(manifest, indent=2))
    total = sum(m["shape"][1] for m in manifest.values())
    print(f"\n{len(manifest)} blocks, {total} columns total -> {OUT}/")
    print("Rows are aligned to features/index.csv in every block, so they concatenate "
          "with np.hstack\nand no merge. Load what a model needs and leave the rest on "
          "disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
