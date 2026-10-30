r"""
tdi_yardstick_v2.py  --  STAGE B: a TDI ruler that reads what the board reads.

STAGE A found the mechanism. Our folds are harder than the blind set, and the gap is
large enough to account for most of the disagreement:

    CYP3A4   fold-holdout median max-Tanimoto 0.464   blind 0.540   (+0.076)
    CYP2D6   fold-holdout median max-Tanimoto 0.342   blind 0.471   (+0.129)

and model skill rises steeply with that similarity (CYP3A4 AUC 0.76 -> 0.95 across the
similarity range; CYP2D6 0.55 -> 0.71). So a raw out-of-fold AUC is not an estimate of
board performance -- it is an estimate of performance on a HARDER population than the one
being scored.

WHAT THIS SCRIPT DOES DIFFERENTLY FROM STAGE A. Stage A extrapolated a straight line
through five similarity bins and read it outside its range, which is exactly the kind of
step that has misled this project before. This instead REWEIGHTS: every out-of-fold
compound gets an importance weight so that the weighted holdout-similarity distribution
matches the blind set's, and the AUC is recomputed under those weights. Nothing is
extrapolated; compounds that resemble the blind set simply count for more. Weights are
clipped and the effective sample size is reported, because a reweighting that leans on a
handful of compounds is not a measurement.

THEN IT PREDICTS A BOARD NUMBER, which is the only thing that makes it a yardstick:
weighted AUC -> d' -> the equal-variance binormal at the BLIND prevalence and the call
rate we actually submit at -> an expected MCC with the board's own sampling error around
it. If the prediction brackets the score the board returned, the ruler works on that
endpoint and may be used to gate a change. If it does not, the endpoint stays frozen.

ACCEPTANCE TEST. The board has already told us the answer twice, so the ruler is scored
against it rather than argued for:

    CYP3A4   board MCC 0.4601    CYP2D6   board MCC 0.2918

A pass is the observed value inside the predicted interval. Stage A's crude linear
version already passed CYP3A4 (z = -0.26) and failed CYP2D6 (z = +2.41, p = 0.010).

Run:
  python scripts\tdi_yardstick_v2.py
  python scripts\tdi_yardstick_v2.py --score-col pooled --bins 12 --clip 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
TDI = ROOT / "data" / "cyp-challenge-TRAIN_TDI.csv"
BLIND = ROOT / "data" / "cyp-challenge-TEST-BLINDED.csv"
FOLDS = ROOT / "artifacts" / "folds_tdi.csv"

BOARD = {
    "CYP3A4": dict(mcc=0.4601, prevalence=0.2915, call_rate=0.2426, n_half=326),
    "CYP2D6": dict(mcc=0.2918, prevalence=0.0706, call_rate=0.1177, n_half=375),
}


def mcc_counts(tp, fp, fn, tn) -> float:
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else float((tp * tn - fp * fn) / den)


def weighted_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), both drawn under w."""
    pos, neg = y, ~y
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_s, y_s, w_s = s[order], y[order], w[order]
    # running weight of negatives at or below each point, with ties at half credit
    cum_neg = np.cumsum(np.where(~y_s, w_s, 0.0))
    below = np.concatenate([[0.0], cum_neg[:-1]])
    # tie handling: equal scores share
    num = 0.0
    i = 0
    n = len(s_s)
    while i < n:
        j = i
        while j + 1 < n and s_s[j + 1] == s_s[i]:
            j += 1
        blk = slice(i, j + 1)
        wneg_blk = np.sum(np.where(~y_s[blk], w_s[blk], 0.0))
        wpos_blk = np.where(y_s[blk], w_s[blk], 0.0)
        num += np.sum(wpos_blk * (below[i] + 0.5 * wneg_blk))
        i = j + 1
    return float(num / (np.sum(w[pos]) * np.sum(w[neg])))


def auc_to_dprime(a: float) -> float:
    a = min(max(a, 0.500001), 0.999999)
    return float(np.sqrt(2.0) * norm.ppf(a))


def fingerprints(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out, keep = [], []
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if m is None:
            continue
        out.append(gen.GetFingerprintAsNumPy(m).astype(np.uint8))
        keep.append(i)
    return np.array(out, dtype=np.uint8), np.array(keep, dtype=int)


def max_tanimoto(query, ref, block=256):
    refT = ref.T.astype(np.float32)
    rsum = ref.sum(axis=1).astype(np.float32)
    best = np.zeros(len(query), dtype=np.float32)
    for i in range(0, len(query), block):
        q = query[i : i + block].astype(np.float32)
        inter = q @ refT
        union = q.sum(axis=1, keepdims=True) + rsum[None, :] - inter
        best[i : i + block] = np.max(inter / np.maximum(union, 1e-9), axis=1)
    return best


def predict_board(dp, prevalence, call_rate, n, rng, nboot=8000):
    """expected board MCC and its sampling spread, for a model of this d'"""
    out = np.empty(nboot)
    for b in range(nboot):
        y = rng.random(n) < prevalence
        s = rng.normal(dp * y, 1.0)
        k = max(1, int(round(call_rate * n)))
        thr = np.sort(s)[::-1][k - 1]
        c = s >= thr
        out[b] = mcc_counts(int((y & c).sum()), int((~y & c).sum()),
                            int((y & ~c).sum()), int((~y & ~c).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="tdiNB")
    ap.add_argument("--score-col", default="pooled")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--clip", type=float, default=20.0, help="max importance weight")
    ap.add_argument("--nboot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    df = pd.read_csv(TDI)
    folds = pd.read_csv(FOLDS)
    blind = pd.read_csv(BLIND)
    bfp, _ = fingerprints(blind["SMILES"].tolist())

    print(f"pool={a.pool}  score column={a.score_col}  bins={a.bins}  weight clip={a.clip}")
    verdicts = []
    for iso, b in BOARD.items():
        col = f"{iso}_is_TDI"
        lab = df[df[col].notna()].merge(folds, on="Molecule_Name", how="left")
        lab = lab[lab["fold"].notna()].reset_index(drop=True)
        fp, keep = fingerprints(lab["SMILES"].tolist())
        lab = lab.iloc[keep].reset_index(drop=True)
        fold = lab["fold"].to_numpy().astype(int)

        sim = np.zeros(len(lab), dtype=np.float32)
        for k in sorted(set(fold)):
            va = fold == k
            if va.sum() and (~va).sum():
                sim[va] = max_tanimoto(fp[va], fp[~va])
        blind_sim = max_tanimoto(bfp, fp)

        oof = pd.read_csv(ROOT / "artifacts" / f"tdi_member_oof_{iso}_{a.pool}.csv")
        sc = a.score_col if a.score_col in oof.columns else oof.columns[-1]
        m = lab[["Molecule_Name"]].copy()
        m["sim"] = sim
        m = m.merge(oof[["Molecule_Name", "y", sc]], on="Molecule_Name", how="inner")
        y = m["y"].astype(bool).to_numpy()
        s = m[sc].to_numpy(dtype=float)
        sm = m["sim"].to_numpy(dtype=float)

        # importance weights: match the BLIND similarity histogram
        edges = np.unique(np.quantile(blind_sim, np.linspace(0, 1, a.bins + 1)))
        edges[0], edges[-1] = -1.0, 2.0
        p_blind, _ = np.histogram(blind_sim, bins=edges)
        p_oof, _ = np.histogram(sm, bins=edges)
        p_blind = p_blind / p_blind.sum()
        p_oof = p_oof / max(p_oof.sum(), 1)
        ratio = np.where(p_oof > 0, p_blind / np.maximum(p_oof, 1e-12), 0.0)
        ratio = np.clip(ratio, 0.0, a.clip)
        idx = np.clip(np.digitize(sm, edges) - 1, 0, len(ratio) - 1)
        w = ratio[idx]
        if w.sum() == 0:
            print(f"{iso}: no overlap between fold and blind similarity -- cannot reweight")
            continue
        w = w / w.mean()
        ess = w.sum() ** 2 / np.sum(w ** 2)

        a_raw = weighted_auc(y, s, np.ones_like(w))
        a_w = weighted_auc(y, s, w)
        dp_raw, dp_w = auc_to_dprime(a_raw), auc_to_dprime(a_w)
        dp_board = auc_to_dprime(
            norm.cdf(np.sqrt(2.0) ** -1 * 0) if False else a_w)  # placeholder, replaced below

        print(f"\n=== {iso}   n={len(m)}  weighted positives "
              f"{np.sum(w[y]) / w.sum():.4f}  effective n {ess:.0f} of {len(m)}")
        print(f"    raw  OOF AUC {a_raw:.4f}  (d' {dp_raw:.3f})   median holdout sim {np.median(sm):.3f}")
        print(f"    BLIND-WEIGHTED AUC {a_w:.4f}  (d' {dp_w:.3f})   blind median sim {np.median(blind_sim):.3f}")

        pred = predict_board(dp_w, b["prevalence"], b["call_rate"], b["n_half"], rng, a.nboot)
        lo, hi = np.percentile(pred, [2.5, 97.5])
        z = (b["mcc"] - pred.mean()) / pred.std()
        ok = lo <= b["mcc"] <= hi
        print(f"    predicted board MCC {pred.mean():.4f} +- {pred.std():.4f}   95% [{lo:.4f}, {hi:.4f}]")
        print(f"    board returned      {b['mcc']:.4f}   z {z:+.2f}   "
              f"{'PASS -- the ruler reads this endpoint' if ok else 'FAIL -- endpoint stays frozen'}")
        verdicts.append((iso, a_raw, a_w, pred.mean(), b["mcc"], z, ok))

    print("\n" + "=" * 78)
    print(f"{'endpoint':<10}{'raw AUC':>9}{'wtd AUC':>9}{'pred MCC':>10}{'board':>9}{'z':>7}  verdict")
    for iso, ar, aw, pm, bm, z, ok in verdicts:
        print(f"{iso:<10}{ar:>9.4f}{aw:>9.4f}{pm:>10.4f}{bm:>9.4f}{z:>+7.2f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("\nA PASS means a change to that endpoint can now be gated offline: measure the")
    print("blind-weighted AUC of the candidate and the incumbent on the SAME rows, and")
    print("require the paired bootstrap to exclude zero. A FAIL means the ruler still does")
    print("not see what the board sees, and no change to that column may be submitted.")


if __name__ == "__main__":
    main()
