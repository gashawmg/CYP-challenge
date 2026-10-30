#!/usr/bin/env python
r"""Recover the exact descriptor columns the Sept-5 per-target Chemprop was trained on,
PROVE the recovery against the saved scaler, and write the mapping.

    python scripts\recover_st_v3_desc.py --model_dir models_cyp_st_v3
    python scripts\recover_st_v3_desc.py --model_dir models_cyp_st_v2

THE PROBLEM. train_cyp_st_v3.py --mode PREDICT dies with

    ValueError: X has 62 features, but RobustScaler is expecting 52 features

because cyp_target_descriptors.py grew after those checkpoints were written. Only the
SCALER is pickled, not its column names, and RobustScaler fitted on a numpy array keeps no
feature_names_in_. So the 52 columns have to be identified some other way.

THE ARITHMETIC NARROWS IT ALMOST TO A POINT. Current widths are 62 / 63 / 67 / 62 against
a required 52. The name lists are core_rdkit (31, continuous) ++ core_smarts (8) ++ a
10-column `phys_*` ionisation block ++ the isoform block. Dropping `phys_*` leaves
52 / 53 / 57 / 52, so all four isoforms once had exactly 13 isoform-specific columns:
CYP1A2 and CYP3A4 need only the phys block removed, CYP2C9 one further column, CYP2D6 five.

THE PROOF, because that last step would otherwise be a guess. RobustScaler stores
center_ (the per-column median) and scale_ (the IQR) of the matrix it was fitted on. Those
52 pairs are a fingerprint. This script recomputes every current column on the training
molecules and:

  1. checks center_[0:31] against the 31 continuous core_rdkit medians. Those columns
     cannot have moved, so a mismatch here means the TRAINING ROW SET is different and
     nothing downstream can be trusted -- the script stops rather than matching noise.
  2. searches, over order-preserving choices of 13 columns from the isoform block, for the
     one whose (median, IQR) pairs reproduce center_[39:52] and scale_[39:52].
  3. requires the winner to be UNIQUE within tolerance. Binary flags tie easily, so
     ambiguity is reported as failure, not resolved by preference.

If it proves out it writes artifacts/st_v3_desc_columns.json, which
scripts/predict_st_v3_ps.py consumes to run the checkpoints unchanged. If it does not,
the honest answer is to retrain, and the script says so instead of shipping a feature
vector that is silently in the wrong order.

WHY NOT JUST RETRAIN. The out-of-fold file we measured -- oof_chemprop_st_v3_ps, the best
single model we own on CYP2C9 (Section 33) -- came from these 52-column checkpoints.
Predicting with exactly those columns reproduces the member that was measured. Retraining
on 62 columns produces a DIFFERENT model whose out-of-fold score we do not have.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cyp_target_descriptors as TD  # noqa: E402
from cypkit import config as C, io  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_CORE_RDKIT = len(TD.CORE_RDKIT)
N_CORE_SMARTS = len(TD.CORE_SMARTS)


def load_pickle(p):
    import pickle  # noqa: PLC0415
    with open(p, "rb") as f:
        return pickle.load(f)


def med_iqr(X):
    """Median and IQR the way RobustScaler stores them.

    sklearn runs the IQR through _handle_zeros_in_scale, so a CONSTANT column -- and the
    isoform blocks are full of binary flags that never fire -- is stored as scale_ = 1.0,
    not 0.0. Comparing against a raw 0.0 makes every candidate subset mismatch, which is
    exactly how the self-test failed on CYP1A2's trivial 13-of-13 case.
    """
    q1, q2, q3 = np.nanpercentile(X, [25, 50, 75], axis=0)
    iqr = q3 - q1
    return q2, np.where(np.abs(iqr) < 10 * np.finfo(float).eps, 1.0, iqr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", default="models_cyp_st_v3")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="relative tolerance when matching a median/IQR pair")
    a = ap.parse_args()
    md = ROOT / a.model_dir
    if not md.exists():
        raise SystemExit(f"  {md} not found -- nothing to recover from.")

    tr = io.load("train_inhibition"); tr.columns = [c.strip() for c in tr.columns]
    smiles = tr[C.SMILES_COL].astype(str).tolist()
    print(f"  {len(smiles)} training molecules from train_inhibition\n")

    out, failed = {}, []
    for iso in C.ISOFORMS:
        pkl = md / f"{iso}_desc_scaler.pkl"
        if not pkl.exists():
            print(f"  {iso}: no {pkl.name}; skipping")
            continue
        sc = load_pickle(pkl)
        want_n = int(sc.n_features_in_)
        names = TD.feature_names(iso)
        X, _ = TD.calculate(smiles, iso)
        X = np.asarray(X, float)
        cur_med, cur_iqr = med_iqr(X)
        print(f"=== {iso}: scaler wants {want_n}, module now emits {len(names)}")
        if want_n == len(names):
            out[iso] = names
            print("  widths already agree; no recovery needed")
            continue

        # ---- step 1: the continuous core must line up, or the row set is wrong --------
        c_ref = np.asarray(sc.center_, float)[:N_CORE_RDKIT]
        c_now = cur_med[:N_CORE_RDKIT]
        rel = np.abs(c_ref - c_now) / np.maximum(np.abs(c_ref), 1e-9)
        bad = int((rel > 1e-3).sum())
        print(f"  core_rdkit medians reproduce: {N_CORE_RDKIT - bad}/{N_CORE_RDKIT} "
              f"(max rel dev {rel.max():.2e})")
        if bad:
            print("  -> THE TRAINING ROW SET DIFFERS from the one the scaler was fitted on.")
            print("     Column identification below would be fitting noise. Retrain instead.")
            failed.append(iso); continue

        # ---- step 2: which isoform columns survive? ------------------------------------
        head = N_CORE_RDKIT + N_CORE_SMARTS
        iso_idx = [j for j, n in enumerate(names) if n.startswith(iso)]
        k = want_n - head
        ref_c = np.asarray(sc.center_, float)[head:]
        ref_s = np.asarray(sc.scale_, float)[head:]
        print(f"  isoform block has {len(iso_idx)} columns; the scaler has room for {k}")
        if k < 1 or k > len(iso_idx):
            print("  -> arithmetic does not close; retrain.")
            failed.append(iso); continue

        hits = []
        for combo in itertools.combinations(iso_idx, k):          # order-preserving
            cm, ci = cur_med[list(combo)], cur_iqr[list(combo)]
            if (np.allclose(cm, ref_c, rtol=a.tol, atol=a.tol)
                    and np.allclose(ci, ref_s, rtol=a.tol, atol=a.tol)):
                hits.append(combo)
                if len(hits) > 1:
                    break
        if len(hits) == 1:
            keep = [j for j in range(len(names)) if j < head] + list(hits[0])
            out[iso] = [names[j] for j in keep]
            print(f"  UNIQUE MATCH. dropped: "
                  f"{[names[j] for j in range(len(names)) if j not in keep]}")
        elif not hits:
            print("  -> NO subset of the isoform block reproduces the scaler. The change "
                  "was not\n     a pure insertion; retrain.")
            failed.append(iso)
        else:
            print("  -> AMBIGUOUS: more than one subset matches (binary flags tie). "
                  "Retrain rather\n     than pick one.")
            failed.append(iso)

    if out:
        p = ROOT / "artifacts" / "st_v3_desc_columns.json"
        p.write_text(json.dumps({"model_dir": a.model_dir, "columns": out}, indent=2))
        print(f"\n  wrote {p.relative_to(ROOT)} for "
              f"{', '.join(out)}")
    if failed:
        print(f"\n  COULD NOT RECOVER: {', '.join(failed)}. For those isoforms the only")
        print("  honest route is to retrain, which also lets the pseudo-label map be made")
        print("  fold-aware first (Section 33).")
    return 0 if out and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
