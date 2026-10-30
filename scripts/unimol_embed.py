#!/usr/bin/env python
r"""Extract Uni-Mol's 3D molecular embedding once, cache it, and reuse it as tree features.

    python scripts\unimol_embed.py --extract
    python scripts\unimol_embed.py --extract --weights D:\unimol_finetuning\mol_pre_no_h_220816.pt

RUN THIS ON THE GPU BOX. unimol_tools pulls its checkpoint from Hugging Face, which is 403
from both sandboxes; --weights points at a local copy instead. Extraction is a forward pass
only -- no training, no labels, a few minutes for all 5655 molecules -- and the result is
cached to artifacts\unimol_repr.npz so nothing after this needs Uni-Mol installed at all.

WHY THE PRETRAINED EMBEDDING AND NOT A FINETUNED ONE. This is the leakage question and it
decides the whole design. Uni-Mol's pretrained encoder has never seen our labels, so its
embedding of a molecule is a fixed function of structure: computing it for a training row
and for a blind row are the same operation, and putting it in a fold-based CV is exactly as
safe as putting a Morgan fingerprint there. A FINETUNED encoder is not: finetune on all our
rows, extract, then train a booster on those vectors, and every fold's "held-out" molecules
were already seen by the encoder. The out-of-fold rho that comes back is then fiction. Doing
it honestly means finetuning once per fold and extracting each fold's held-out rows from the
model that never saw them -- five times the cost, and worth it only if the free version
shows something first.

WHAT THIS BUYS THAT run_unimol.py DOES NOT. That script trains Uni-Mol end to end, and its
docstring records the limitation: the trainer exposes MAE/MSE only, so there is no way to
express ST-RAE's dead zone inside it. Extracting the embedding moves the geometry into the
booster, where the objective, the interval weighting and the censored-row handling are all
already under our control -- and where it can sit beside the descriptors and the CYP2D6
pharmacophore block rather than instead of them.

ALSO WHY NOT "CONCATENATE DESCRIPTORS INTO UNI-MOL". You can feed Uni-Mol extra columns
only by editing its head, and you would still inherit its loss. The booster route carries
the same information with none of that, and it is testable with the discipline we already
use. If the embedding earns its place here, THEN a custom Uni-Mol head is worth the work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C, io  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "unimol_repr.npz"


def all_smiles(data_dir=None) -> pd.DataFrame:
    """Every molecule we will ever need a vector for: labelled rows plus the blind 750."""
    folds = pd.read_csv(ROOT / "artifacts" / "folds.csv")
    folds.columns = [c.strip() for c in folds.columns]
    te = io.load("test", data_dir); te.columns = [c.strip() for c in te.columns]
    a = folds[[C.ID_COL, C.SMILES_COL]].assign(split="train")
    b = te[[C.ID_COL, C.SMILES_COL]].assign(split="blind")
    d = pd.concat([a, b], ignore_index=True).dropna(subset=[C.SMILES_COL])
    return d.drop_duplicates(subset=[C.ID_COL]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--weights", default=None, help="local mol_pre_no_h_220816.pt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--data-dir", type=Path, default=None)
    a = ap.parse_args()

    d = all_smiles(a.data_dir)
    print(f"  {len(d)} molecules ({int((d.split=='train').sum())} labelled, "
          f"{int((d.split=='blind').sum())} blind)")
    if not a.extract:
        print("  pass --extract to compute; nothing done")
        return 0

    try:
        from unimol_tools import UniMolRepr  # noqa: PLC0415
    except Exception as e:                                    # noqa: BLE001
        raise SystemExit(
            f"unimol_tools will not import here ({e}).\n"
            f"Run this on the GPU box where run_unimol.py already works.") from e

    kw = {}
    if a.weights:
        kw["pretrained_model_path"] = a.weights
    clf = UniMolRepr(data_type="molecule", batch_size=a.batch, remove_hs=False, **kw)

    smi = d[C.SMILES_COL].astype(str).tolist()
    res = clf.get_repr(smi, return_atomic_reprs=False)
    # unimol_tools returns {'cls_repr': [...]} or a bare list depending on version
    cls = res["cls_repr"] if isinstance(res, dict) else res
    X = np.asarray(cls, np.float32)
    if X.ndim != 2 or len(X) != len(d):
        raise SystemExit(f"expected ({len(d)}, d) embeddings, got {X.shape} -- do not "
                         f"proceed, the rows would not line up with the ids")
    np.savez_compressed(OUT, X=X, ids=d[C.ID_COL].to_numpy(),
                        split=d["split"].to_numpy())
    print(f"  wrote {OUT.name}: {X.shape[0]} x {X.shape[1]}, "
          f"{np.isfinite(X).all(axis=1).mean():.1%} finite rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
