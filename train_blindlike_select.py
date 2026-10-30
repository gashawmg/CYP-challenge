#!/usr/bin/env python
r"""Re-run pIC50 model selection on the exam we are actually sitting.

    python train_blindlike_select.py
    python train_blindlike_select.py --iso CYP2D6
    python train_blindlike_select.py --cutoff 0.55 --boot 600

WHY. Section 16 measured the defect and never acted on it. Nearest-neighbour Tanimoto from each
compound to the set a model could actually learn from:

    our folds, to OTHER folds   median 0.438   28.2% at >= 0.50   15.7% at >= 0.55
    THE BLIND SET, to train     median 0.587   95.7% at >= 0.50   72.3% at >= 0.55

Almost every blind compound has a close relative in training. Barely a quarter of our
validation rows do. EVERY MODEL CLASS THIS PROJECT REJECTED WAS REJECTED ON AN EXAM FOUR
TIMES HARDER THAN THE REAL ONE, and a hard exam systematically rewards smoothness and
heavy regularisation -- the opposite of what wins at interpolation. Section 28's regularisation
sweep concluded "the incumbent is already best". That conclusion is only as good as the
yardstick it was measured on.

This is the one experiment that has ever reversed an answer in this project: Section 25 ran it on
TDI CYP2D6 and the model ranking flipped outright.

THREE YARDSTICKS, SAME PREDICTIONS, SO THE COMPARISON IS EXACT.

    rho ALL          every out-of-fold row            <- what Section 28 and every sweep used
    rho >= cutoff    the blind-like rows only         <- honest, but small and coarse
    rho REWEIGHTED   all rows, resampled so the NN-similarity profile MATCHES the blind
                     set bin for bin                  <- uses every row AND the right
                                                         difficulty; this is the verdict

The reweighting is importance sampling on nearest-neighbour similarity: bin it, form
w = P_blind(bin) / P_train(bin), and bootstrap rows with probability proportional to w.
Each resample is a validation set with the blind set's difficulty, so the spread of rho
across resamples IS the confidence interval. No weighted-Spearman approximation is needed.

FOUR ARMS SPANNING MEMORISE <-> SMOOTH, deliberately, because that is the axis the defect
distorts:

    knn            similarity-weighted, k=15, w = sim^3   -- pure memorisation
    lgbm loose     127 leaves, min_child 5,  reg_lambda 0
    lgbm incumbent  63 leaves, min_child 20, reg_lambda 1  -- what we ship
    lgbm tight      15 leaves, min_child 60, reg_lambda 10 -- maximum smoothing

WHAT COUNTS AS A RESULT (Section 19). A gap must exceed the reweighted bootstrap spread, and the
two yardsticks must be reported side by side. If ALL and REWEIGHTED pick the same arm,
the defect does not change the model choice and Section 28 stands. If they disagree, every
regularisation decision in this project is suspect and the rerun is the candidate.

NOTHING HERE IS A SUBMISSION. It selects what to build next.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

from cypkit import config as C  # noqa: E402
from train_pic50_interval import featurise  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
BASE = dict(n_estimators=700, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=-1)
ARMS = {
    "knn (memorise)": None,
    "lgbm loose":     dict(learning_rate=0.050, num_leaves=127, min_child_samples=5,
                           colsample_bytree=0.8, reg_lambda=0.0, **BASE),
    "lgbm incumbent": dict(learning_rate=0.035, num_leaves=63, min_child_samples=20,
                           colsample_bytree=0.6, reg_lambda=1.0, **BASE),
    "lgbm tight":     dict(learning_rate=0.020, num_leaves=15, min_child_samples=60,
                           colsample_bytree=0.4, reg_lambda=10.0, **BASE),
}
EDGES = np.array([0.0, .30, .35, .40, .45, .50, .55, .60, .65, .70, 1.01])
# What the BOARD says each endpoint's rho really is. Reweighting is supposed to make the
# local yardstick read closer to these. If it reads FURTHER away, the reweighting is not
# a fix and nothing selected under it may be acted on (Section 19, condition 1).
BLIND_RHO = {"CYP1A2": 0.8064, "CYP2C9": 0.8454, "CYP2D6": 0.5190, "CYP3A4": 0.8327}


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def fingerprints(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        out.append(gen.GetFingerprint(m) if m is not None else None)
    return out


def knn_predict(fp_tr, y_tr, fp_te, k=15, power=3.0):
    ok = [i for i, f in enumerate(fp_tr) if f is not None]
    ref = [fp_tr[i] for i in ok]
    yr = y_tr[ok]
    out = np.full(len(fp_te), float(yr.mean()))
    for i, q in enumerate(fp_te):
        if q is None:
            continue
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(q, ref))
        idx = np.argsort(s)[::-1][:k]
        w = s[idx] ** power
        if w.sum() > 0:
            out[i] = float(np.dot(w, yr[idx]) / w.sum())
    return out


def blind_weights(nn_train, nn_blind):
    """Importance weights making the training NN profile match the blind one."""
    tb = np.clip(np.digitize(nn_train, EDGES) - 1, 0, len(EDGES) - 2)
    bb = np.clip(np.digitize(nn_blind, EDGES) - 1, 0, len(EDGES) - 2)
    pt = np.bincount(tb, minlength=len(EDGES) - 1).astype(float)
    pb = np.bincount(bb, minlength=len(EDGES) - 1).astype(float)
    pt /= max(pt.sum(), 1); pb /= max(pb.sum(), 1)
    w = np.where(pt > 0, pb / np.maximum(pt, 1e-12), 0.0)[tb]
    return w / w.sum() if w.sum() > 0 else np.full(len(nn_train), 1 / len(nn_train))


def reweighted(y, preds, w, n_boot, seed=0):
    """rho for every arm on the SAME resamples -- paired, so gaps are comparable."""
    rng = np.random.default_rng(seed)
    names = list(preds)
    acc = {k: [] for k in names}
    n = len(y)
    for _ in range(n_boot):
        i = rng.choice(n, n, replace=True, p=w)
        if len(np.unique(y[i])) < 3:
            continue
        for k in names:
            acc[k].append(spearman(preds[k][i], y[i]))
    return {k: np.asarray(v) for k, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", default="all")
    ap.add_argument("--cutoff", type=float, default=0.50)
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()

    lab = pd.read_csv(C.artifact("labels.csv"))
    folds = pd.read_csv(C.artifact("folds.csv")); folds.columns = [c.strip() for c in folds.columns]
    nn = pd.read_csv(C.artifact("nnsim.csv")); nn.columns = [c.strip() for c in nn.columns]
    nn_blind = nn.loc[nn["split"] != "train", "nnsim"].to_numpy(float)
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))
    sim_of = dict(zip(folds[C.ID_COL], folds["max_sim_to_other_folds"]))

    print(f"\n  blind set: median NN {np.median(nn_blind):.3f}, "
          f"{(nn_blind >= a.cutoff).mean():.1%} at >= {a.cutoff}")
    print(f"  our folds: median NN {folds['max_sim_to_other_folds'].median():.3f}, "
          f"{(folds['max_sim_to_other_folds'] >= a.cutoff).mean():.1%} at >= {a.cutoff}")

    import lightgbm as lgb  # noqa: PLC0415
    isos = C.ISOFORMS if a.iso == "all" else (a.iso,)
    summary, flips = [], []

    for iso in isos:
        t = lab[(lab["isoform"] == iso) & (lab["kind"] != "left")].dropna(subset=["y"])
        t = t[t[C.ID_COL].isin(fold_of)].copy()
        t["fold"] = t[C.ID_COL].map(fold_of)
        t["nn"] = t[C.ID_COL].map(sim_of)
        t = t.dropna(subset=["nn"]).reset_index(drop=True)
        y = t["y"].to_numpy(float)
        smi = t[C.SMILES_COL].tolist()
        X = featurise(smi)
        fp = fingerprints(smi)
        nnv = t["nn"].to_numpy(float)
        ev = nnv >= a.cutoff
        w = blind_weights(nnv, nn_blind)

        print(f"\n=== {iso}   {len(t)} rows, {int(ev.sum())} at NN >= {a.cutoff} "
              f"({ev.mean():.1%}); effective n after reweighting "
              f"{1.0/np.sum(w**2):.0f}")

        preds = {}
        for name, params in ARMS.items():
            o = np.zeros(len(t))
            for f in sorted(t["fold"].unique()):
                tr = (t["fold"] != f).to_numpy(); te = (t["fold"] == f).to_numpy()
                if params is None:
                    o[te] = knn_predict([fp[i] for i in np.where(tr)[0]], y[tr],
                                        [fp[i] for i in np.where(te)[0]])
                else:
                    m = lgb.LGBMRegressor(**params)
                    m.fit(X[tr], y[tr])
                    o[te] = m.predict(X[te])
            preds[name] = o

        bt = reweighted(y, preds, w, a.boot)
        print(f"  {'arm':16s} {'rho ALL':>9s} {'rho >=cut':>10s} {'rho REWEIGHTED':>15s} "
              f"{'95% CI':>18s}")
        for name in ARMS:
            o = preds[name]
            r_all = spearman(o, y)
            r_ev = spearman(o[ev], y[ev]) if ev.sum() > 30 else np.nan
            b = bt[name]
            print(f"  {name:16s} {r_all:9.4f} {r_ev:10.4f} {b.mean():15.4f} "
                  f"  [{np.quantile(b, .025):+.4f},{np.quantile(b, .975):+.4f}]")
            summary.append(dict(isoform=iso, arm=name, rho_all=r_all, rho_cut=r_ev,
                                rho_rw=float(b.mean()),
                                lo=float(np.quantile(b, .025)),
                                hi=float(np.quantile(b, .975))))

        inc = bt["lgbm incumbent"]
        print(f"\n  gap vs the incumbent, on the SAME resamples (paired):")
        for name in ARMS:
            if name == "lgbm incumbent":
                continue
            d = bt[name] - inc
            print(f"    {name:16s} {d.mean():+7.4f}  "
                  f"[{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}]  "
                  f"P(worse) {float((d <= 0).mean()):.0%}")

        # A FLIP IS ONLY A FLIP IF IT SURVIVES THE PAIRED BOOTSTRAP. On CYP2C9 the top
        # two arms on rho ALL differ by 0.0003 -- naming a "winner" from point estimates
        # would manufacture a disagreement out of the fourth decimal, which is precisely
        # how blend2d6 cost rank 9 -> 36 (Section 24). The two candidates are compared on the
        # SAME resamples and the flip is declared only when that gap excludes zero.
        w_all = max(ARMS, key=lambda k: spearman(preds[k], y))
        w_rw = max(ARMS, key=lambda k: bt[k].mean())
        real = False
        if w_all != w_rw:
            d = bt[w_rw] - bt[w_all]
            real = float(np.quantile(d, .025)) > 0
            tag = ("   <-- REAL DISAGREEMENT" if real else
                   "   (point estimates only -- gap spans 0, NOT a flip)")
            extra = (f"    {w_rw} - {w_all} = {d.mean():+.4f} "
                     f"[{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}]")
        else:
            tag, extra = "   (agree)", ""
        flips.append((iso, w_all, w_rw, real))
        print(f"\n  best on ALL rows (the yardstick Section 28 used): {w_all}")
        print(f"  best REWEIGHTED to blind difficulty:        {w_rw}{tag}")
        if extra:
            print(extra)

        # is the defect even real on this endpoint?
        inc_o = preds["lgbm incumbent"]
        # READ sd(y) BEFORE READING rho HERE. Spearman inside a narrow similarity band is
        # range-restricted: if the band also happens to be narrow in y, rho falls for a
        # reason that has nothing to do with difficulty. MAE is not range-restricted, so
        # the two columns together say which effect is which. On CYP2C9 rho DROPS as
        # similarity rises, which looks like the opposite of Section 16 -- check sd(y) first.
        print("\n  incumbent by NN-similarity bin  (sd(y) guards against range restriction)")
        print(f"    {'bin':11s} {'n':>5s} {'sd(y)':>7s} {'rho':>7s} {'MAE':>7s}")
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (nnv >= lo) & (nnv < hi)
            if m.sum() >= 40:
                print(f"    {lo:.2f}-{hi:.2f}  {int(m.sum()):5d} {y[m].std():7.3f} "
                      f"{spearman(inc_o[m], y[m]):7.4f} "
                      f"{np.mean(np.abs(inc_o[m] - y[m])):7.4f}")

    pd.DataFrame(summary).to_csv(ROOT / "artifacts" / "blindlike_select.csv", index=False)

    print("\n" + "=" * 72)
    print("BACK-TEST -- DOES REWEIGHTING MOVE THE YARDSTICK TOWARD THE BOARD? (Section 19)\n")
    print("  The whole premise is that blind-like rows are EASIER, so a difficulty-matched")
    print("  yardstick should read HIGHER and closer to the scored blind rho. Levels are")
    print("  not comparable (one booster vs the shipped ensemble) but the DIRECTION is.\n")
    print(f"  {'endpoint':9s} {'rho ALL':>9s} {'rho REWT':>9s} {'move':>8s} "
          f"{'blind rho':>10s}  reading")
    sdf = pd.DataFrame(summary)
    good = 0
    for iso in isos:
        r = sdf[(sdf.isoform == iso) & (sdf.arm == "lgbm incumbent")]
        if r.empty:
            continue
        ra, rw = float(r.rho_all.iloc[0]), float(r.rho_rw.iloc[0])
        ok = rw > ra
        good += ok
        print(f"  {iso:9s} {ra:9.4f} {rw:9.4f} {rw-ra:+8.4f} {BLIND_RHO[iso]:10.4f}  "
              + ("toward the board" if ok else "AWAY from the board"))
    if good < len(isos):
        print("\n  WARNING. On the endpoints marked AWAY, matching the blind set's")
        print("  NEAREST-NEIGHBOUR profile makes the task look HARDER, while the board says")
        print("  it is EASIER (instrument ratios 1.01-1.33, all >= 1). Similarity is then")
        print("  not what makes the blind set easy, and the likely reason is visible in the")
        print("  per-bin table: our high-similarity rows are dense analogue series, where")
        print("  neighbours are close but activities differ sharply -- activity cliffs, not")
        print("  easy interpolation. Section 16's premise does not hold on those endpoints.")
        print("  DO NOT ACT ON A FLIP FROM AN ENDPOINT MARKED AWAY. It selected under a")
        print("  distribution shift of its own, which is the mistake it was built to fix.")
    print("\n" + "=" * 72)
    print("VERDICT")
    n_flip = sum(1 for *_, real in flips if real)
    for iso, x, z, real in flips:
        print(f"  {iso:8s} ALL -> {x:16s} REWEIGHTED -> {z:16s}"
              + ("  FLIP (significant)" if real else
                 "  (differs, but inside noise)" if x != z else ""))
    if n_flip == 0:
        print("\n  No endpoint changes its winner. The validation defect is real (Section 16) but")
        print("  it does NOT distort the model choice: Section 28 stands and the incumbent holds.")
        print("  Spend the next effort on a genuinely new representation, not on retuning.")
    else:
        print(f"\n  {n_flip} of {len(flips)} endpoints pick a DIFFERENT arm once difficulty")
        print("  matches the exam. Every regularisation decision in this project was made")
        print("  on the wrong yardstick. Rebuild those endpoints with the reweighted")
        print("  winner, re-blend, and price the result before any submission (Section 19).")
    print("\n  wrote artifacts/blindlike_select.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
