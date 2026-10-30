#!/usr/bin/env python
"""A mechanistically-justified minimal descriptor set for the CYP models, and the test of
whether cutting to it costs anything.

    python cyp_target_descriptors_lean.py --iso all

WHY CUT. The current blocks are 62 / 63 / 67 / 62 columns wide on 1,000-2,300 labelled
rows per endpoint. Measured on our own 4,905 training molecules, the 31 generic RDKit
descriptors carry only 15 independent pieces of information at |r| >= 0.70, and one
cluster of EIGHT -- MolWt, HeavyAtomCount, MolMR, Kappa1, Chi0v, Chi1v, Chi2v, LabuteASA --
is a single fact: how big the molecule is. Width like that costs interpretability outright
and costs accuracy whenever the row count is small.

THE RULE FOR KEEPING A COLUMN: it must name a physical requirement for binding or turnover
in a CYP active site, in one sentence, and not be a near-duplicate of a column already
kept. Composite columns that a model can rebuild by itself (products, ratios, per-heavy-atom
normalisations) are dropped -- a booster splits on them for free and an MPNN's FFN can
multiply.

WHAT SURVIVES, AND WHY, IN FOUR GROUPS:

  OCCUPANCY AND SHAPE -- does it fit the pocket?
    HeavyAtomCount      how much volume it asks for (the whole 8-way size cluster)
    FractionCSP3        flat vs puckered; CYP1A2's slot is narrow and planar, CYP3A4's
                        is not
    Kappa3              shape/branching, independent of size
    NumRotatableBonds   entropic cost of freezing into the pocket; CYP3A4 tolerates it

  RECOGNITION -- will it partition in and hydrogen-bond?
    MolLogP             neutral lipophilicity
    TPSA                polar surface (NOCount dropped, |r| > 0.8 with it)
    NumHDonors          (NHOHCount dropped, duplicate)
    NumHAcceptors
    NumAromaticRings    pi-stacking above the heme
    RingCount

  ELECTRONICS -- how easily is it oxidised, and where?
    MaxPartialCharge    electron-poor extreme
    MinPartialCharge    electron-rich extreme, where abstraction happens
    MinEStateIndex      the most electron-poor atom environment
    MaxAbsEStateIndex   (MaxEStateIndex dropped, duplicate)

  IONISATION at pH 7.4 -- the single biggest determinant of CYP2C9 vs CYP2D6 selectivity
    phys_pka_basic, phys_pka_acidic, phys_logd74, phys_net_charge_74, phys_is_zwitterion
    (frac_base/acid_ionised are functions of the pKas; logp_minus_logd is MolLogP minus
     logd74; formal_charge and is_permanent_cation are covered by net_charge_74)

  Plus the 8 heme-coordination SMARTS unchanged -- aromatic N, imidazole, triazole,
  nitrile and friends are the classic type-II binders and there is nothing redundant in
  them -- and the isoform block with its self-derived composites removed.

DROPPED FROM THE GENERIC CORE, with reasons: MolWt / MolMR / Kappa1 / Chi0v / Chi1v /
Chi2v / LabuteASA (all the size cluster), Kappa2 (with Kappa3), NumSaturatedRings and
NumAliphaticRings (inside FractionCSP3), NOCount (with TPSA), NHOHCount (with NumHDonors),
MaxEStateIndex (with MaxAbsEStateIndex), NumHeteroatoms (largely TPSA), HallKierAlpha
(size and heteroatom count again), BertzCT (graph complexity -- no binding meaning), qed
(a drug-likeness preference score, not a property of the protein-ligand interaction).

THIS FILE ONLY DEFINES AND TESTS THE SET. It does not change the training path; it selects
columns by name from the existing calculate(), so nothing has to be recomputed and nothing
downstream is touched until the test says it should be.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cyp_target_descriptors as TD  # noqa: E402

KEEP_CORE_RDKIT = [
    "HeavyAtomCount", "FractionCSP3", "Kappa3", "NumRotatableBonds",
    "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumAromaticRings", "RingCount",
    "MaxPartialCharge", "MinPartialCharge", "MinEStateIndex", "MaxAbsEStateIndex",
]
KEEP_PHYS = ["phys_pka_basic", "phys_pka_acidic", "phys_logd74",
             "phys_net_charge_74", "phys_is_zwitterion"]
# isoform composites a model can rebuild from columns already present
# CORRECTED after the first run. Dropping CYP1A2's three planarity composites cost
# -0.0364 rho with P(worse) 98% -- the only endpoint that lost. The general rule ("a model
# can rebuild a composite") does not apply to them: planarity IS CYP1A2's binding
# requirement, its narrow slot admits flat aromatics, and a planarity index built from
# aromatic-atom fraction and ring fusion is not something a booster reconstructs from
# HeavyAtomCount and FractionCSP3 on 1,180 rows. A composite that encodes the isoform's
# mechanism is a primary feature, not a convenience.
DROP_ISO_COMPOSITE = {
    "CYP1A2": [],
    "CYP2C9": ["CYP2C9_acid_and_lipophilic", "CYP2C9_acid_per_heavy",
               "CYP2C9_anionic_logp_product"],
    "CYP2D6": ["CYP2D6_base_and_aromatic", "CYP2D6_basic_n_per_heavy"],
    "CYP3A4": [],
}


def lean_names(iso: str) -> list[str]:
    """The kept columns, in the order calculate() emits them."""
    keep = set(KEEP_CORE_RDKIT) | set(KEEP_PHYS) | {
        n for n in TD.feature_names(iso) if n.startswith("core_")}
    drop = set(DROP_ISO_COMPOSITE.get(iso, []))
    out = []
    for n in TD.feature_names(iso):
        if n in drop:
            continue
        if n in keep or (n.startswith(iso) and n not in drop):
            out.append(n)
    return out


def lean_index(iso: str) -> list[int]:
    want = set(lean_names(iso))
    return [j for j, n in enumerate(TD.feature_names(iso)) if n in want]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", default="all")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()

    from rdkit import Chem, DataStructs, RDLogger                      # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator                      # noqa: PLC0415
    import lightgbm as lgb                                             # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")

    ISOS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
    print("  WIDTH\n")
    print(f"  {'isoform':9s} {'full':>6s} {'lean':>6s} {'cut':>7s}")
    for iso in ISOS:
        f, l = TD.n_features(iso), len(lean_names(iso))
        print(f"  {iso:9s} {f:6d} {l:6d} {1-l/f:7.0%}")

    # --- the blind-similarity yardstick (Section 30) --------------------------------
    tr = pd.read_csv("data/cyp-challenge-TRAIN_inhibition.csv")
    te = pd.read_csv("data/cyp-challenge-TEST-BLINDED.csv")
    for d in (tr, te):
        d.columns = [c.strip() for c in d.columns]
    sc = next(c for c in tr.columns if c.lower() == "smiles")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_tr = [gen.GetFingerprint(m) for m in
             (Chem.MolFromSmiles(str(s)) for s in tr[sc]) if m is not None]
    nn_blind = np.array([max(DataStructs.BulkTanimotoSimilarity(
        gen.GetFingerprint(m), fp_tr))
        for m in (Chem.MolFromSmiles(str(s)) for s in te[sc]) if m is not None])
    print(f"\n  blind set: {len(nn_blind)} compounds, median NN to train "
          f"{np.median(nn_blind):.3f}")

    lab = pd.read_csv("artifacts/labels.csv")
    folds = pd.read_csv("artifacts/folds.csv"); folds.columns = [c.strip() for c in folds.columns]
    fold_of = dict(zip(folds["Molecule_Name"], folds["fold"]))
    sim_of = dict(zip(folds["Molecule_Name"], folds["max_sim_to_other_folds"]))
    EDGES = np.array([0.0, .30, .35, .40, .45, .50, .55, .60, .65, .70, 1.01])

    def weights(nn):
        pb = np.clip(np.digitize(nn, EDGES) - 1, 0, len(EDGES) - 2)
        bb = np.clip(np.digitize(nn_blind, EDGES) - 1, 0, len(EDGES) - 2)
        p = np.bincount(pb, minlength=len(EDGES) - 1).astype(float)
        q = np.bincount(bb, minlength=len(EDGES) - 1).astype(float)
        p /= max(p.sum(), 1); q /= max(q.sum(), 1)
        w = np.where(p > 0, q / np.maximum(p, 1e-12), 0.0)[pb]
        return w / w.sum() if w.sum() > 0 else np.full(len(nn), 1 / len(nn))

    rng = np.random.default_rng(0)
    isos = ISOS if a.iso == "all" else (a.iso,)
    print()
    for iso in isos:
        t = lab[(lab["isoform"] == iso) & (lab["kind"] != "left")].dropna(subset=["y"])
        t = t[t["Molecule_Name"].isin(fold_of) & t["Molecule_Name"].isin(sim_of)].copy()
        t["fold"] = t["Molecule_Name"].map(fold_of)
        t = t.reset_index(drop=True)
        y = t["y"].to_numpy(float)
        X, _ = TD.calculate(t["SMILES"].tolist(), iso)
        X = np.nan_to_num(np.asarray(X, float))
        nn = t["Molecule_Name"].map(sim_of).to_numpy(float)
        w = weights(nn)
        arms = {"full": list(range(X.shape[1])), "lean": lean_index(iso)}
        preds = {}
        for name, cols in arms.items():
            o = np.zeros(len(t))
            for f in sorted(t["fold"].unique()):
                trm = (t["fold"] != f).to_numpy(); tem = (t["fold"] == f).to_numpy()
                m = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.035, num_leaves=63,
                                      min_child_samples=20, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.6, reg_lambda=1.0, verbose=-1,
                                      n_jobs=-1)
                m.fit(X[np.ix_(trm, cols)], y[trm])
                o[tem] = m.predict(X[np.ix_(tem, cols)])
            preds[name] = o
        g = []
        for _ in range(a.boot):
            i = rng.choice(len(y), len(y), replace=True, p=w)
            if len(np.unique(y[i])) < 3:
                continue
            s = pd.Series
            g.append(float(s(preds["lean"][i]).corr(s(y[i]), method="spearman"))
                     - float(s(preds["full"][i]).corr(s(y[i]), method="spearman")))
        g = np.asarray(g)
        rho = {k: float(pd.Series(v).corr(pd.Series(y), method="spearman"))
               for k, v in preds.items()}
        print(f"  {iso}  {len(t)} rows   full {len(arms['full'])} cols rho {rho['full']:.4f}"
              f"   lean {len(arms['lean'])} cols rho {rho['lean']:.4f}")
        print(f"      reweighted gap  {g.mean():+.4f} "
              f"[{np.quantile(g, .025):+.4f}, {np.quantile(g, .975):+.4f}]  "
              f"P(lean worse) {float((g <= 0).mean()):.0%}")
    print("\n  A lean set is worth shipping if the gap's interval CONTAINS ZERO: same")
    print("  accuracy, a third fewer columns, every one explainable to a chemist. A")
    print("  significant negative gap means a column carrying real signal was cut.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
