#!/usr/bin/env python
r"""CYP3A4 only: does a WIDER general block, or RDKit 3D shape, buy anything?

    python train_3a4_width.py --boot 300

WHY CYP3A4 AND WHY NOW. Section 40 found the lean descriptor cut helps in perfect rank
order with scarcity -- +0.0493 on CYP2D6 (1,364 fitted rows), +0.0315 CYP1A2 (1,444),
+0.0234 CYP2C9 (1,983), and -0.0028 on CYP3A4 (4,199). CYP3A4 is the only endpoint with
PXR-scale data (PXR trained on 4,262) and the only one where narrowing did not help. If
width is affordable anywhere here, it is affordable here.

THE PXR COMPARISON, CORRECTED. PXR's 61 descriptors are not 61 target-specific features:
its own section headers label THREE as PXR-specific (num_ring_hydroxyl, num_carboxyl,
bile_acid_score) plus steroid_scaffold_score. The other 57 are general physicochemistry.
Our lean vector is the other way round -- 14 general and 26 target-specific on CYP3A4. So
the honest question is not "add more target-specific descriptors", we already have six
times PXR's count. It is: DID WE CUT THE GENERAL BLOCK TOO HARD ON THE ONE ENDPOINT THAT
COULD AFFORD IT?

AND THE 3D GAP, WHICH IS REAL. PXR carried NPR1, NPR2 and Asphericity -- conformer-derived
shape descriptors. WE HAVE NEVER USED ANY 3D DESCRIPTOR. Every column in
cyp_target_descriptors is a 2D graph property or a SMARTS count. CYP3A4's pocket is the
large, flexible, shape-selective one, so if geometry matters anywhere among the four, it
matters here. One ETKDGv3 conformer with MMFF optimisation costs about 4.6 minutes across
the whole training set, which is nothing.

FOUR ARMS, identical folds, identical booster:

    lean            the Section 36 block, 40 columns          <- the incumbent
    +wide           plus the 17 general columns Section 36 dropped and a PXR-style
                    VSA / substructure-count set, approaching PXR's 57 general
    +3d             plus 8 conformer shape descriptors (the 3 PMI terms are left out:
                    they are raw size, already covered, and Section 36 measured what
                    duplicate size columns cost)
    +wide+3d        both

SCORED two ways -- similarity-reweighted rho (Section 30) and the fixed blind-like
holdout (Section 39, 248 CYP3A4 rows, the largest of the four) -- with a paired bootstrap.
A gap inside its own interval is not a result.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ISO = "CYP3A4"
EDGES = np.array([0.0, .30, .35, .40, .45, .50, .55, .60, .65, .70, 1.01])
# the general columns Section 36 removed, restored here to test whether CYP3A4 wants them
WIDE_RDKIT = ["MolWt", "MolMR", "BertzCT", "Kappa1", "Kappa2", "Chi0v", "Chi1v", "Chi2v",
              "HallKierAlpha", "LabuteASA", "MaxEStateIndex", "NumHeteroatoms",
              "NHOHCount", "NOCount", "NumSaturatedRings", "NumAliphaticRings", "qed",
              # PXR-style additions we never had
              "SMR_VSA5", "SMR_VSA7", "SlogP_VSA2", "SlogP_VSA5", "PEOE_VSA6",
              "PEOE_VSA7", "NumAliphaticCarbocycles", "NumAromaticCarbocycles",
              "NumAromaticHeterocycles", "NumSaturatedHeterocycles", "HeavyAtomMolWt"]
WIDE_SMARTS = {  # PXR counted these explicitly; they are cheap and interpretable
    "num_Cl": "[Cl]", "num_F": "[F]", "num_Br": "[Br]", "num_carbonyl": "[CX3]=[OX1]",
    "num_ketone": "[#6][CX3](=O)[#6]", "num_ether_O": "[OD2]([#6])[#6]",
    "num_hydroxyl": "[OX2H]", "num_sulfoxide": "[$([#16X3]=[OX1])]",
    "num_NH": "[NX3;H1,H2]", "num_michael": "[CX3]=[CX3][CX3]=[OX1]",
}
DESC_3D = ["NPR1", "NPR2", "Asphericity", "Eccentricity", "InertialShapeFactor",
           "SpherocityIndex", "RadiusOfGyration", "PBF"]


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def wide_block(smiles):
    from rdkit import Chem                                          # noqa: PLC0415
    from rdkit.Chem import Descriptors                              # noqa: PLC0415
    pats = {k: Chem.MolFromSmarts(v) for k, v in WIDE_SMARTS.items()}
    fns = [(n, getattr(Descriptors, n)) for n in WIDE_RDKIT if hasattr(Descriptors, n)]
    missing = [n for n in WIDE_RDKIT if not hasattr(Descriptors, n)]
    if missing:
        print(f"  note: not in this RDKit, skipped: {', '.join(missing)}")
    X = np.zeros((len(smiles), len(fns) + len(pats)), float)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        for j, (_, f) in enumerate(fns):
            try:
                X[i, j] = f(m)
            except Exception:                                       # noqa: BLE001
                pass
        for j, (_, p) in enumerate(pats.items()):
            X[i, len(fns) + j] = len(m.GetSubstructMatches(p))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), \
        [n for n, _ in fns] + list(pats)


def three_d_block(smiles, seed=0xC0FFEE):
    """One ETKDGv3 conformer, MMFF-optimised, then the shape descriptors.

    A FAILED EMBED IS RECORDED, NOT SILENTLY ZEROED: a zero row looks like a legitimate
    sphere-like molecule to the booster. Failures get NaN -> column mean instead, and the
    count is printed so a high failure rate cannot pass unnoticed.
    """
    from rdkit import Chem                                          # noqa: PLC0415
    from rdkit.Chem import AllChem, Descriptors3D                   # noqa: PLC0415
    X = np.full((len(smiles), len(DESC_3D)), np.nan)
    bad = 0
    t0 = time.time()
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            bad += 1; continue
        m = Chem.AddHs(m)
        p = AllChem.ETKDGv3(); p.randomSeed = seed; p.maxIterations = 200
        if AllChem.EmbedMolecule(m, p) != 0:
            bad += 1; continue
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:                                           # noqa: BLE001
            pass
        for j, n in enumerate(DESC_3D):
            try:
                X[i, j] = getattr(Descriptors3D, n)(m)
            except Exception:                                       # noqa: BLE001
                pass
        if i and i % 1000 == 0:
            print(f"    conformers {i}/{len(smiles)}  ({time.time()-t0:.0f}s)")
    col = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col, idx[1])
    print(f"  3D block: {len(smiles)-bad}/{len(smiles)} conformers in "
          f"{time.time()-t0:.0f}s, {bad} failed (filled with the column mean)")
    return np.nan_to_num(X), list(DESC_3D)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()

    import lightgbm as lgb                                          # noqa: PLC0415
    from rdkit import Chem, DataStructs, RDLogger                   # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator                   # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")
    import cyp_target_descriptors as TD                             # noqa: PLC0415
    from cyp_target_descriptors_lean import lean_index, lean_names  # noqa: PLC0415

    lab = pd.read_csv("artifacts/labels.csv")
    folds = pd.read_csv("artifacts/folds.csv"); folds.columns = [c.strip() for c in folds.columns]
    fold_of = dict(zip(folds["Molecule_Name"], folds["fold"]))
    sim_of = dict(zip(folds["Molecule_Name"], folds["max_sim_to_other_folds"]))
    hold = set(pd.read_csv("artifacts/holdout_blindlike.csv")["Molecule_Name"])

    t = lab[(lab.isoform == ISO) & (lab.kind != "left")].dropna(subset=["y"])
    t = t[t.Molecule_Name.isin(fold_of) & t.Molecule_Name.isin(sim_of)].reset_index(drop=True)
    t["fold"] = t.Molecule_Name.map(fold_of)
    y = t.y.to_numpy(float); smi = t.SMILES.tolist()
    print(f"\n  {ISO}: {len(t)} fitted rows\n")

    Xa, _ = TD.calculate(smi, ISO)
    L = np.nan_to_num(np.asarray(Xa, float))[:, lean_index(ISO)]
    W, wn = wide_block(smi)
    D, dn = three_d_block(smi)
    print(f"  lean {L.shape[1]}   +wide {W.shape[1]}   +3d {D.shape[1]}")

    arms = {"lean": L, "+wide": np.hstack([L, W]), "+3d": np.hstack([L, D]),
            "+wide+3d": np.hstack([L, W, D])}

    # the two yardsticks
    te = pd.read_csv("data/cyp-challenge-TEST-BLINDED.csv"); te.columns = [c.strip() for c in te.columns]
    tr = pd.read_csv("data/cyp-challenge-TRAIN_inhibition.csv"); tr.columns = [c.strip() for c in tr.columns]
    sc = next(c for c in tr.columns if c.lower() == "smiles")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = [gen.GetFingerprint(m) for m in (Chem.MolFromSmiles(str(s)) for s in tr[sc]) if m]
    nnb = np.array([max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), fp))
                    for m in (Chem.MolFromSmiles(str(s)) for s in te[sc]) if m])
    nn = t.Molecule_Name.map(sim_of).to_numpy(float)
    pb = np.clip(np.digitize(nn, EDGES) - 1, 0, len(EDGES) - 2)
    bb = np.clip(np.digitize(nnb, EDGES) - 1, 0, len(EDGES) - 2)
    pp = np.bincount(pb, minlength=len(EDGES) - 1).astype(float)
    qq = np.bincount(bb, minlength=len(EDGES) - 1).astype(float)
    pp /= max(pp.sum(), 1); qq /= max(qq.sum(), 1)
    w = np.where(pp > 0, qq / np.maximum(pp, 1e-12), 0.0)[pb]; w /= w.sum()
    hm = t.Molecule_Name.isin(hold).to_numpy()

    P = dict(n_estimators=700, learning_rate=0.035, num_leaves=63, min_child_samples=20,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
             verbose=-1, n_jobs=-1)
    preds = {}
    for name, X in arms.items():
        o = np.zeros(len(t))
        for f in sorted(t.fold.unique()):
            m = (t.fold != f).to_numpy()
            g = lgb.LGBMRegressor(**P); g.fit(X[m], y[m]); o[~m] = g.predict(X[~m])
        preds[name] = o

    rng = np.random.default_rng(0)
    base = preds["lean"]
    print(f"\n  {'arm':10s} {'cols':>5s} {'rho rewt':>9s} {'gap vs lean':>12s} "
          f"{'95% CI':>20s} {'rho holdout':>12s}")
    rows = []
    for name, X in arms.items():
        o = preds[name]
        g = []
        for _ in range(a.boot):
            i = rng.choice(len(y), len(y), replace=True, p=w)
            if len(np.unique(y[i])) > 2:
                g.append(spearman(o[i], y[i]) - spearman(base[i], y[i]))
        g = np.asarray(g)
        rw = np.mean([spearman(o[i], y[i]) for i in
                      (rng.choice(len(y), len(y), replace=True, p=w) for _ in range(150))])
        ho = spearman(o[hm], y[hm])
        print(f"  {name:10s} {X.shape[1]:5d} {rw:9.4f} {g.mean():+12.4f} "
              f"[{np.quantile(g,.025):+.4f},{np.quantile(g,.975):+.4f}] {ho:12.4f}")
        rows.append(dict(arm=name, cols=X.shape[1], rho_rewt=rw, gap=float(g.mean()),
                         lo=float(np.quantile(g, .025)), hi=float(np.quantile(g, .975)),
                         rho_holdout=ho, n_holdout=int(hm.sum())))
    pd.DataFrame(rows).to_csv("artifacts/3a4_width_summary.csv", index=False)
    print(f"\n  holdout n = {int(hm.sum())};  board rho for CYP3A4 is 0.8327")
    print("\n  A gap whose interval contains zero is not a result. If +3d clears it, we")
    print("  have never used geometry and should; if only +wide clears it, width scales")
    print("  with row count and CYP3A4 alone should carry the wide block.")
    print("\n  wrote artifacts/3a4_width_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
