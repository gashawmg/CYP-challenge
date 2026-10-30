#!/usr/bin/env python
r"""Match the evaluation pool to the blind set on DIFFICULTY, not just on the label histogram.

    python scripts\build_matched_pool.py            # diagnose, write artifacts\nnsim.csv
    python scripts\build_matched_pool.py --check    # also re-score blend4z, before vs after

THE PROBLEM THIS FIXES, AND WHY IT HAS DISTORTED EVERY DECISION IN THE PROJECT.
`blind_validator.eval_pool()` resamples our out-of-fold rows to the blind LABEL histogram.
It says nothing about how hard the compounds are, and on that axis the two sets are not
remotely alike:

    out-of-fold nearest-neighbour Tanimoto      median   25th   10th    <0.4
    our folds (Butina)                           0.438  0.375  0.320   33.1%
    our folds (random -- IDENTICAL)              0.439  0.377  0.320   32.9%
    THE BLIND SET, to its nearest training compound
                                                 0.587  0.543  0.516    0.1%

A THIRD of our validation compounds have no near neighbour to learn from. In the blind set
that number is one in a thousand. We have been grading ourselves on extrapolation while
the exam is interpolation.

THE SPLIT SCHEME IS NOT THE CULPRIT -- random folds give the same 0.438, because our 4905
training compounds are simply spread out. Changing the splitter cannot fix this. What fixes
it is evaluating on the part of our data that looks like the exam: weight the pool toward
compounds that HAVE a close neighbour in the rest of the training set, until the pool's
similarity profile matches the blind set's.

WHAT IT EXPLAINS. The per-endpoint instrument ratios (local rho against the blind rho the
board reported) are 1.33x on CYP1A2, 1.16x CYP2C9, 1.03x CYP2D6, 1.01x CYP3A4. A pool that
is uniformly too hard reads uniformly too low, and the endpoints whose chemistry is most
spread out read worst.

WHAT IT INVALIDATES. Anything chosen because it "generalises well" was chosen under a
distribution shift the real task does not have. That includes heavy shrinkage: the
k-multiplier scan on the current pool wants k = 0.55 against the k = rho = 0.83 we ship,
and a pool that is too hard is exactly what would manufacture that preference. Re-run that
scan here before believing it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

from cypkit import config as C, io  # noqa: E402
from scripts.blind_validator import BLIND, eval_pool, match_weights  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
NNSIM = ROOT / "artifacts" / "nnsim.csv"


def fingerprints(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            fp.append(gen.GetFingerprint(m)); keep.append(i)
    return fp, np.asarray(keep, int)


def nn_similarity(data_dir=None) -> pd.DataFrame:
    """Per compound: the similarity to its nearest neighbour among the OTHER training rows,
    and, for the blind compounds, to their nearest training row. One row per molecule."""
    folds = pd.read_csv(ROOT / "artifacts" / "folds.csv")
    folds.columns = [c.strip() for c in folds.columns]
    te = io.load("test", data_dir); te.columns = [c.strip() for c in te.columns]

    tr_fp, tr_keep = fingerprints(folds[C.SMILES_COL].tolist())
    te_fp, te_keep = fingerprints(te[C.SMILES_COL].tolist())
    tr = folds.iloc[tr_keep].reset_index(drop=True)

    # leave-one-out: the self-match is always 1.0, so take the second largest
    own = []
    for i, q in enumerate(tr_fp):
        s = DataStructs.BulkTanimotoSimilarity(q, tr_fp)
        s[i] = -1.0
        own.append(max(s))
    blind = [max(DataStructs.BulkTanimotoSimilarity(q, tr_fp)) for q in te_fp]
    return (pd.DataFrame({C.ID_COL: tr[C.ID_COL], "nnsim": own, "split": "train"}),
            pd.DataFrame({C.ID_COL: te.iloc[te_keep][C.ID_COL], "nnsim": blind,
                          "split": "blind"}))


def match_weights_2d(y, nnsim, mean, sd, blind_nnsim, n_bins=8):
    """Weights that match the pool to the blind set on BOTH label and difficulty.

    The label axis reuses blind_validator.match_weights so the existing behaviour is
    preserved exactly; the difficulty axis is a second histogram ratio on nearest-neighbour
    similarity. Multiplying two marginal ratios does not reproduce a joint distribution,
    but the two axes are close to independent here (|r| < 0.1 on every endpoint) and a
    marginal correction removes most of a gap this large.
    """
    w_y, _ = match_weights(y, np.zeros(len(y), bool), mean, sd)
    edges = np.quantile(blind_nnsim, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    want = np.histogram(blind_nnsim, bins=edges)[0].astype(float)
    have = np.histogram(nnsim, bins=edges)[0].astype(float)
    want /= want.sum()
    have = np.where(have > 0, have, np.nan)
    ratio = want / (have / np.nansum(have))
    idx = np.clip(np.digitize(nnsim, edges) - 1, 0, n_bins - 1)
    w_s = np.nan_to_num(ratio[idx], nan=0.0)
    w = w_y * w_s
    if w.sum() <= 0:
        return w_y, 0.0
    w = w / w.sum()
    ess = 1.0 / np.sum(w ** 2)          # effective sample size: how much we paid for this
    return w, ess


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="re-score the shipped blend4z on the matched pool")
    ap.add_argument("--data-dir", type=Path, default=None)
    a = ap.parse_args()

    if NNSIM.exists():
        both = pd.read_csv(NNSIM)
        tr_s = both[both.split == "train"]; te_s = both[both.split == "blind"]
        print(f"  reusing {NNSIM.name}")
    else:
        tr_s, te_s = nn_similarity(a.data_dir)
        pd.concat([tr_s, te_s]).to_csv(NNSIM, index=False)
        print(f"  wrote {NNSIM.name}")

    print(f"\n  {'set':28s} {'median':>8s} {'25th':>8s} {'10th':>8s} {'<0.4':>7s}")
    for n, v in (("our folds", tr_s.nnsim.to_numpy()), ("the blind set", te_s.nnsim.to_numpy())):
        print(f"  {n:28s} {np.median(v):8.3f} {np.quantile(v,.25):8.3f} "
              f"{np.quantile(v,.10):8.3f} {(v<0.4).mean():6.1%}")

    if not a.check:
        return 0

    b4 = {}
    for n, f in (("v6", "oof_st_v6.csv"), ("mh", "oof_st_mh.csv"),
                 ("cp", "artifacts/oof_chempropv3_ensA.csv"), ("um", "oof_unimol_cyp_mt.csv")):
        d = pd.read_csv(ROOT / f); d.columns = [c.strip() for c in d.columns]
        b4[n] = d.set_index(C.ID_COL)
    sim = dict(zip(tr_s[C.ID_COL], tr_s.nnsim))
    blind_nn = te_s.nnsim.to_numpy()
    pool = eval_pool()
    BLIND_RHO = {"CYP1A2": 0.8064, "CYP2C9": 0.8454, "CYP2D6": 0.5190, "CYP3A4": 0.8327}

    print(f"\n  local rho for the SHIPPED blend4z, before and after difficulty matching\n")
    print(f"  {'endpoint':9s} {'blind rho':>10s} {'y-matched':>10s} {'ratio':>6s} "
          f"{'+difficulty':>12s} {'ratio':>6s} {'ESS':>7s}")
    for iso in C.ISOFORMS:
        ep = f"{iso}_pIC50_direct_inhibition"
        s = pool[(pool.isoform == iso) & (~pool.censored)]
        ids = s[C.ID_COL]
        for d in b4.values():
            ids = ids[ids.isin(d.index)]
        ids = ids[ids.isin(sim)]
        s = s[s[C.ID_COL].isin(ids)]; ids = s[C.ID_COL].to_numpy()
        y = s["y_eval"].to_numpy(float)
        z = np.nan_to_num(np.mean([((lambda v: (v - np.nanmean(v)) / (np.nanstd(v) or 1))
                                    (d.loc[ids, ep].to_numpy(float))) for d in b4.values()],
                                  axis=0))
        nn = np.array([sim[m] for m in ids])
        rng = np.random.default_rng(0)
        w1, _ = match_weights(y, np.zeros(len(y), bool), *BLIND[iso])
        w2, ess = match_weights_2d(y, nn, *BLIND[iso], blind_nn)

        def rho_under(w):
            r = [float(pd.Series(z[d]).corr(pd.Series(y[d]), method="spearman"))
                 for d in (rng.choice(len(y), size=len(y), replace=True, p=w)
                           for _ in range(40))]
            return float(np.mean(r))

        r1, r2 = rho_under(w1), rho_under(w2)
        print(f"  {iso:9s} {BLIND_RHO[iso]:10.4f} {r1:10.4f} {BLIND_RHO[iso]/r1:6.2f} "
              f"{r2:12.4f} {BLIND_RHO[iso]/r2:6.2f} {ess:7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
