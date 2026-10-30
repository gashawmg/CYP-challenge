#!/usr/bin/env python
r"""Which compounds does EVERY model get wrong, and is that predictable without the label?

    python scripts/shared_failure.py
    python scripts/shared_failure.py --iso CYP1A2 --quantile 0.8

WHY. 74-86% of a typical member's squared error survives averaging all eleven members
(measured 2026-09-29): they make the same mistakes on the same compounds. Ensembling
cannot touch that, and neither can a marginally better member -- which is why descriptors,
3D shape and seed-averaging each gained standalone and vanished in the blend. The score
that is left lives in the shared component, so the useful question is not "can the model
be better" but "what are these compounds, and is there anything about them we could act
on".

THE ONE DISTINCTION THAT MATTERS. A shared failure is only actionable if it can be spotted
WITHOUT the answer. So every candidate explanation here is scored by how well it separates
failures from the rest using information available at prediction time:

    nn_sim        Tanimoto to the nearest labelled training compound
    knn_spread    how much that compound's 5 nearest labelled neighbours disagree
    knn_mean      what those neighbours would predict
    ens_sd        how much the members disagree with EACH OTHER
    MolWt, MolLogP, TPSA, charge ...

`ens_sd` is the interesting one. It is the standard uncertainty estimate, and the whole
point of a SHARED failure is that the members agree while being wrong -- so if ens_sd does
not separate them, the models have no idea they are failing, and any triage built on
member disagreement is blind to exactly the errors that matter.

`y` itself is reported too, but as diagnosis, not as a lever: knowing that failures sit in
the low tail cannot flag a blind compound, though it can tell us which region to train
harder on.

READ THE AUC COLUMN, NOT THE MEANS. A difference in means between failures and the rest is
easy to produce and easy to over-read. AUC asks the operational question directly: given
one failure and one non-failure, how often does this feature rank them correctly? 0.50 is
useless; 0.60 is a weak but real signal; below 0.55 there is nothing to build on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

ISOS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
MEMBERS = {
    "v6": "oof_st_v6.csv", "v7": "oof_st_v7.csv", "v5_rest": "oof_st_v5_rest.csv",
    "mh": "oof_st_mh_ctl.csv", "mh2": "oof_st_mh2.csv",
    "unimol": "oof_unimol_cyp_mt.csv", "cp": "artifacts/oof_chempropv3_ensA.csv",
    "cp_lean": "oof_chemprop_st_v3_lean.csv", "mh_desc": "oof_st_mh_desc.csv",
    "mh_d3d": "oof_st_mh_desc3d.csv", "mhens3": "oof_mhens3_k3.csv",
}
DESCS = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
         "NumRotatableBonds", "NumAromaticRings", "FractionCSP3"]


def auc(flag, x):
    """P(feature ranks a failure above a non-failure). Ties count half."""
    ok = np.isfinite(x)
    f, v = flag[ok], x[ok]
    if f.sum() < 10 or (~f).sum() < 10:
        return np.nan
    r = pd.Series(v).rank().to_numpy()
    n1, n0 = f.sum(), (~f).sum()
    return (r[f].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fingerprints(smiles):
    from rdkit import Chem, RDLogger                                  # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator                     # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        X.append(np.frombuffer(gen.GetFingerprint(m).ToBitString().encode(), "u1")
                 - ord("0"))
        keep.append(i)
    return np.asarray(X, bool), np.asarray(keep)


def descriptors(smiles):
    from rdkit import Chem                                            # noqa: PLC0415
    from rdkit.Chem import Descriptors                                # noqa: PLC0415
    fns = [(n, getattr(Descriptors, n)) for n in DESCS if hasattr(Descriptors, n)]
    X = np.full((len(smiles), len(fns)), np.nan)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        for j, (_, f) in enumerate(fns):
            try:
                X[i, j] = f(m)
            except Exception:                                         # noqa: BLE001
                pass
    return X, [n for n, _ in fns]


def neighbour_stats(X, y, k=5, block=256):
    """For each row: similarity to its nearest OTHER row, and the spread of its k nearest."""
    n = len(X)
    b = X.astype(np.float32)
    nb = b.sum(1)
    nn = np.empty(n, np.float32); sp = np.empty(n, np.float32); km = np.empty(n, np.float32)
    for i in range(0, n, block):
        a = b[i:i + block]
        inter = a @ b.T
        sim = inter / np.maximum(a.sum(1)[:, None] + nb[None, :] - inter, 1e-9)
        for j in range(len(a)):
            s = sim[j].copy(); s[i + j] = -1          # exclude self
            nn[i + j] = s.max()
            idx = np.argpartition(-s, k)[:k]
            v = y[idx]
            sp[i + j] = float(np.mean(np.abs(v[:, None] - v[None, :])))
            km[i + j] = float(v.mean())
    return nn, sp, km


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", nargs="*", default=list(ISOS))
    ap.add_argument("--quantile", type=float, default=0.80,
                    help="a compound is a SHARED FAILURE if the consensus |residual| is "
                         "above this quantile (default: the worst 20%)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="artifacts/shared_failure.csv")
    a = ap.parse_args()

    lab = pd.read_csv("artifacts/labels.csv")
    lab.columns = [c.strip() for c in lab.columns]
    rows = []

    for iso in a.iso:
        col = f"{iso}_pIC50_direct_inhibition"
        t = lab[(lab["isoform"] == iso) & (lab["kind"] != "left")].dropna(subset=["y"])
        t = t[["Molecule_Name", "SMILES", "y"]].drop_duplicates("Molecule_Name")

        P = {}
        for n, f in MEMBERS.items():
            if not Path(f).exists():
                continue
            d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]
            if col not in d.columns:
                continue
            j = t.merge(d[["Molecule_Name", col]], on="Molecule_Name").dropna(subset=[col])
            if len(j) > 500:
                P[n] = j.set_index("Molecule_Name")[col]
        if len(P) < 4:
            print(f"{iso}: only {len(P)} members, skipped"); continue

        ids = sorted(set.intersection(*[set(v.index) for v in P.values()]))
        tt = t.set_index("Molecule_Name").loc[ids]
        y = tt["y"].to_numpy(float)
        M = np.vstack([P[n].loc[ids].to_numpy(float) for n in P])
        cons = M.mean(0)
        res = cons - y
        ens_sd = M.std(0)

        fp, keep = fingerprints(tt["SMILES"].tolist())
        if len(keep) != len(ids):
            tt = tt.iloc[keep]; y = y[keep]; res = res[keep]
            cons = cons[keep]; ens_sd = ens_sd[keep]
            ids = [ids[i] for i in keep]
        nn, sp, km = neighbour_stats(fp, y, k=a.k)
        D, dn = descriptors(tt["SMILES"].tolist())

        thr = np.quantile(np.abs(res), a.quantile)
        fail = np.abs(res) > thr

        feats = {"nn_sim": nn, "knn_spread": sp, "ens_sd": ens_sd,
                 "|knn_mean - y|": np.abs(km - y), "y": y}
        feats.update({dn[i]: D[:, i] for i in range(len(dn))})

        print(f"\n{'='*78}\n{iso}: {len(ids)} compounds, {len(P)} members, "
              f"{int(fail.sum())} shared failures (worst {1-a.quantile:.0%})")
        print(f"  consensus |residual|: failures {np.abs(res)[fail].mean():.3f} "
              f"vs rest {np.abs(res)[~fail].mean():.3f}   "
              f"(over-predicted {np.mean(res[fail] > 0):.0%} of failures)")
        print(f"\n  {'feature':16s} {'failures':>9s} {'rest':>9s} {'AUC':>7s}  "
              f"{'usable at predict time?':<24s}")
        order = sorted(feats, key=lambda k_: -abs(auc(fail, feats[k_]) - 0.5)
                       if np.isfinite(auc(fail, feats[k_])) else 0)
        for k_ in order:
            v = feats[k_]
            A = auc(fail, v)
            tag = "no - needs the answer" if k_ == "y" or "knn_mean" in k_ else "yes"
            print(f"  {k_:16s} {np.nanmean(v[fail]):9.3f} {np.nanmean(v[~fail]):9.3f} "
                  f"{A:7.3f}  {tag:<24s}")
            rows.append(dict(isoform=iso, feature=k_, auc=A,
                             mean_fail=float(np.nanmean(v[fail])),
                             mean_rest=float(np.nanmean(v[~fail])), usable=(tag == "yes")))

    if rows:
        out = pd.DataFrame(rows)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")
        u = out[out.usable]
        best = u.loc[u.groupby("isoform")["auc"].apply(
            lambda s: (s - 0.5).abs().idxmax())]
        print("\nBEST LABEL-FREE SEPARATOR PER ENDPOINT")
        for _, x in best.iterrows():
            print(f"  {x.isoform:9s} {x.feature:16s} AUC {x.auc:.3f}")
        print("\n  AUC near 0.50 on every usable feature means the shared failures are")
        print("  INVISIBLE from the input side: we cannot tell which predictions are")
        print("  about to be wrong, so triage, reweighting and abstention are all off")
        print("  the table, and only a better representation can help.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
