#!/usr/bin/env python
"""Fit each ensemble member on ALL the data and write its blind-set predictions.

    python scripts/predict_members.py --learners lgb_dz lgb_l1 xgb_dz xgb_l1 svr_tan

Split out of run_ensemble.py because the out-of-fold pass and the final refit are
independent, and re-running the OOF loop just to reach the refit costs an hour for
nothing. The per-member files are what `blend_composite.py --predict` combines, so the
blend weights can be re-fitted against a new validator without retraining anything.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from cypkit import ensemble as E  # noqa: E402
from cypkit import io, labels as L, splits  # noqa: E402
from scripts.run_ensemble import build_bank  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learners", nargs="*", default=list(E.LEARNERS))
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--censored-weight", type=float, default=0.5)
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    train = io.load("train_inhibition", args.data_dir)
    screen = io.load("single_conc", args.data_dir)
    test = io.load("test", args.data_dir)
    lm = L.build_label_matrix(train, screen, include_censored=True,
                              censored_weight=args.censored_weight)

    smiles_by_id = dict(zip(lm[C.ID_COL], lm[C.SMILES_COL]))
    smiles_by_id.update(dict(zip(train[C.ID_COL], train[C.SMILES_COL])))
    smiles_by_id.update(dict(zip(test[C.ID_COL], test[C.SMILES_COL])))
    smiles_by_id = {k: v for k, v in smiles_by_id.items() if isinstance(v, str)}

    blocks = E.blocks_needed(args.learners)
    print(f"featurising {len(smiles_by_id)} compounds, blocks = {'+'.join(blocks)}")
    bank = build_bank(smiles_by_id, blocks)
    test_idx = test[C.ID_COL].map(bank.index_of).to_numpy()

    for name in args.learners:
        out = C.artifact(f"testpred_{name}.csv")
        if out.exists():
            print(f"  {name:8s} already written, skipping")
            continue
        t0 = time.time()
        X = bank.view(E.LEARNERS[name]["features"])
        m = test[[C.SMILES_COL, C.ID_COL]].copy()
        for cyp in C.ISOFORMS:
            ep = f"{cyp}_pIC50_direct_inhibition"
            sub = lm[(lm["isoform"] == cyp) & lm[C.ID_COL].isin(bank.index_of)]
            tr = sub[C.ID_COL].map(bank.index_of).to_numpy()
            m[ep] = E.fit_predict(name, X[tr], sub, X[test_idx], delta=args.delta)
        m.to_csv(out, index=False)
        print(f"  {name:8s} -> {out.name}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
