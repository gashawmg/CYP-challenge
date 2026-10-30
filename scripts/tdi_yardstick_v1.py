r"""
tdi_yardstick_v1.py  --  STAGE A: why does the TDI out-of-fold set disagree with the board?

The problem this exists to solve. Our CYP2D6 TDI out-of-fold AUC is 0.619, which under an
equal-variance binormal model is d' 0.43. The board's MCC of 0.2918 at a call rate of
0.1177 against a prevalence of 0.0706 implies d' 1.23, i.e. AUC 0.807. A yardstick that
reads 0.62 where the board reads 0.81 cannot license ANY change to that column, which is
why CYP2D6 has been frozen since September. This script measures the three things that
could produce that gap, and says how much of it each one explains.

  1 DIFFICULTY.  Scaffold-held-out folds may simply be harder than the blind set. For the
    pIC50 track the median max-Tanimoto of a fold holdout to its training half is 0.438
    while the blind set sits at 0.587 -- our validation is harder than the board. This
    measures the same two numbers for each TDI isoform, on that isoform's own labelled
    rows, and reports an AUC-vs-similarity curve so the gap can be priced rather than
    just noted.

  2 POPULATION.  Out-of-fold prevalence is 0.2164 (CYP2D6) against the board's 0.0706.
    MCC moves with prevalence even at fixed discriminability, so MCC is not comparable
    across the two populations. AUC is. Everything here is reported in AUC and d' first,
    and MCC only after the population is stated.

  3 BOARD NOISE.  The public board is scored on roughly half the blind compounds. At
    CYP2D6's prevalence that is on the order of 26 positives, and an MCC measured on 26
    positives has a wide sampling distribution. This simulates the board's own standard
    error at the stated d', for the half set and the full set, which sets the minimum
    change that is detectable at all.

Nothing here changes a model or writes a submission. It measures the ruler.

Run:
  python scripts\tdi_yardstick_v1.py
  python scripts\tdi_yardstick_v1.py --nboot 4000 --half-n 375
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parent.parent
TDI = ROOT / "data" / "cyp-challenge-TRAIN_TDI.csv"
BLIND = ROOT / "data" / "cyp-challenge-TEST-BLINDED.csv"
FOLDS = ROOT / "artifacts" / "folds_tdi.csv"

# what the board actually returned, from artifacts/submissions.csv (tdi_blendcp_rerun)
BOARD = {
    "CYP3A4": dict(mcc=0.4601, prevalence=0.2915, call_rate=0.2426, n_scored=651),
    "CYP2D6": dict(mcc=0.2918, prevalence=0.0706, call_rate=0.1177, n_scored=750),
}


# ---------------------------------------------------------------- small helpers
def auc_of(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def mcc_counts(tp, fp, fn, tn) -> float:
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else float((tp * tn - fp * fn) / den)


def dprime_from_board(mcc: float, prevalence: float, call_rate: float) -> float:
    """Invert an equal-variance binormal model from one scored operating point."""
    from scipy.optimize import brentq

    def gap(dp: float) -> float:
        # threshold that produces this call rate under this d' and prevalence
        def rate(t):
            return prevalence * (1 - norm.cdf(t - dp)) + (1 - prevalence) * (1 - norm.cdf(t))
        t = brentq(lambda t: rate(t) - call_rate, -8.0, 8.0)
        tpr = 1 - norm.cdf(t - dp)
        fpr = 1 - norm.cdf(t)
        tp, fn = prevalence * tpr, prevalence * (1 - tpr)
        fp, tn = (1 - prevalence) * fpr, (1 - prevalence) * (1 - fpr)
        return mcc_counts(tp, fp, fn, tn) - mcc

    return float(brentq(gap, 0.02, 5.0))


def auc_from_dprime(dp: float) -> float:
    return float(norm.cdf(dp / np.sqrt(2.0)))


def dprime_from_auc(a: float) -> float:
    return float(np.sqrt(2.0) * norm.ppf(a))


# ---------------------------------------------------------------- fingerprints
def fingerprints(smiles: list[str]):
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


def max_tanimoto(query: np.ndarray, ref: np.ndarray, block: int = 256) -> np.ndarray:
    """max Tanimoto of every query row against the whole reference set"""
    refT = ref.T.astype(np.float32)
    rsum = ref.sum(axis=1).astype(np.float32)
    best = np.zeros(len(query), dtype=np.float32)
    for i in range(0, len(query), block):
        q = query[i : i + block].astype(np.float32)
        inter = q @ refT
        qsum = q.sum(axis=1, keepdims=True)
        union = qsum + rsum[None, :] - inter
        best[i : i + block] = np.max(inter / np.maximum(union, 1e-9), axis=1)
    return best


# ---------------------------------------------------------------- the three measurements
def difficulty(df, folds, blind_fp, iso, oof_path: Path | None):
    col = f"{iso}_is_TDI"
    lab = df[df[col].notna()].copy()
    lab = lab.merge(folds, on="Molecule_Name", how="left")
    lab = lab[lab["fold"].notna()]
    fp, keep = fingerprints(lab["SMILES"].tolist())
    lab = lab.iloc[keep].reset_index(drop=True)
    fold = lab["fold"].to_numpy().astype(int)

    sims, aucs_by_bin = [], None
    per_cpd_sim = np.zeros(len(lab), dtype=np.float32)
    for k in sorted(set(fold)):
        va = fold == k
        if va.sum() == 0 or (~va).sum() == 0:
            continue
        s = max_tanimoto(fp[va], fp[~va])
        per_cpd_sim[va] = s
        sims.append(s)
    sims = np.concatenate(sims)

    blind_sim = max_tanimoto(blind_fp, fp)

    print(f"\n--- {iso}: how hard is our holdout compared with the blind set")
    print(f"    labelled rows {len(lab)}   positives {int(lab[col].astype(bool).sum())} "
          f"({lab[col].astype(bool).mean():.4f} prevalence)")
    q = lambda a: np.percentile(a, [25, 50, 75])
    a1, a2 = q(sims), q(blind_sim)
    print(f"    fold holdout -> its training half   q25 {a1[0]:.3f}  median {a1[1]:.3f}  q75 {a1[2]:.3f}")
    print(f"    BLIND        -> all labelled rows   q25 {a2[0]:.3f}  median {a2[1]:.3f}  q75 {a2[2]:.3f}")
    print(f"    gap in medians {a2[1]-a1[1]:+.3f}   "
          f"(blind is {'EASIER' if a2[1] > a1[1] else 'HARDER'} than our validation)")

    # AUC as a function of how close the holdout compound is to training
    if oof_path is not None and oof_path.exists():
        oof = pd.read_csv(oof_path)
        score_col = "pooled" if "pooled" in oof.columns else oof.columns[-1]
        m = lab[["Molecule_Name", col]].merge(oof[["Molecule_Name", "y", score_col]],
                                              on="Molecule_Name", how="inner")
        sim_map = dict(zip(lab["Molecule_Name"], per_cpd_sim))
        m["sim"] = m["Molecule_Name"].map(sim_map)
        m = m[m["sim"].notna()]
        y = m["y"].astype(bool).to_numpy()
        s = m[score_col].to_numpy()
        sim = m["sim"].to_numpy()
        edges = np.percentile(sim, [0, 20, 40, 60, 80, 100])
        print(f"    AUC by holdout-to-train similarity  (score column: {score_col})")
        aucs_by_bin = []
        for i in range(5):
            sel = (sim >= edges[i]) & (sim <= edges[i + 1] if i == 4 else sim < edges[i + 1])
            if sel.sum() < 30 or y[sel].sum() in (0, int(sel.sum())):
                continue
            a = auc_of(y[sel], s[sel])
            aucs_by_bin.append((float(np.median(sim[sel])), a, int(sel.sum())))
            print(f"      sim {edges[i]:.3f}-{edges[i+1]:.3f}   n {sel.sum():5d}   AUC {a:.4f}")
        if len(aucs_by_bin) >= 2:
            xs = np.array([b[0] for b in aucs_by_bin])
            ys = np.array([b[1] for b in aucs_by_bin])
            slope, icept = np.polyfit(xs, ys, 1)
            pred = icept + slope * a2[1]
            print(f"      linear in similarity: AUC = {icept:.4f} + {slope:.4f} * sim")
            print(f"      extrapolated to the BLIND median sim {a2[1]:.3f}  ->  AUC {pred:.4f}"
                  f"   (d' {dprime_from_auc(min(max(pred,0.51),0.999)):.3f})")
    return float(a1[1]), float(a2[1]), aucs_by_bin


def board_noise(iso: str, dp: float, prevalence: float, call_rate: float,
                n_full: int, n_half: int, rng, nboot: int):
    """sampling SD of the board's own MCC at this d', for a half set and a full set"""
    def one(n):
        out = np.empty(nboot)
        for b in range(nboot):
            y = rng.random(n) < prevalence
            s = rng.normal(dp * y, 1.0)
            k = max(1, int(round(call_rate * n)))
            thr = np.sort(s)[::-1][k - 1]
            call = s >= thr
            tp = int(np.sum(y & call)); fp = int(np.sum(~y & call))
            fn = int(np.sum(y & ~call)); tn = int(np.sum(~y & ~call))
            out[b] = mcc_counts(tp, fp, fn, tn)
        return out
    h, f = one(n_half), one(n_full)
    print(f"\n--- {iso}: the board's own sampling noise at d' {dp:.4f}")
    print(f"    half set  n={n_half:4d}  positives ~{n_half*prevalence:.0f}   "
          f"MCC {h.mean():.4f} +- {h.std():.4f}   95% range [{np.percentile(h,2.5):.4f}, {np.percentile(h,97.5):.4f}]")
    print(f"    full set  n={n_full:4d}  positives ~{n_full*prevalence:.0f}   "
          f"MCC {f.mean():.4f} +- {f.std():.4f}   95% range [{np.percentile(f,2.5):.4f}, {np.percentile(f,97.5):.4f}]")
    print(f"    smallest change visible on the half set (2 SD): {2*h.std():.4f} MCC "
          f"= {h.std():.4f} MA-MCC if one isoform moves")
    return float(h.std()), float(f.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="tdiNB", help="member-OOF pool for the AUC-vs-similarity curve")
    ap.add_argument("--nboot", type=int, default=1500)
    ap.add_argument("--half-n", type=int, default=None,
                    help="scored rows on the public half board; default = half of n_scored")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    df = pd.read_csv(TDI)
    folds = pd.read_csv(FOLDS)
    blind = pd.read_csv(BLIND)
    bfp, bkeep = fingerprints(blind["SMILES"].tolist())
    print(f"blind compounds fingerprinted: {len(bkeep)} of {len(blind)}")

    print("\n" + "=" * 78)
    print("1 + 2   DIFFICULTY AND POPULATION")
    print("=" * 78)
    summary = {}
    for iso in ("CYP3A4", "CYP2D6"):
        oof = ROOT / "artifacts" / f"tdi_member_oof_{iso}_{a.pool}.csv"
        summary[iso] = difficulty(df, folds, bfp, iso, oof)

    print("\n" + "=" * 78)
    print("3   WHAT THE BOARD IMPLIES, AND HOW NOISY THE BOARD IS")
    print("=" * 78)
    for iso, b in BOARD.items():
        dp = dprime_from_board(b["mcc"], b["prevalence"], b["call_rate"])
        print(f"\n--- {iso}: board MCC {b['mcc']:.4f} at rate {b['call_rate']:.4f}, "
              f"prevalence {b['prevalence']:.4f}")
        print(f"    implied d' {dp:.4f}   implied AUC {auc_from_dprime(dp):.4f}")
        half = a.half_n if a.half_n else int(round(b["n_scored"] / 2))
        board_noise(iso, dp, b["prevalence"], b["call_rate"], b["n_scored"], half, rng, a.nboot)

    print("\n" + "=" * 78)
    print("READ THIS BEFORE USING ANY NUMBER ABOVE")
    print("=" * 78)
    print("The d' values are inverted from a SINGLE scored operating point under an")
    print("equal-variance binormal model. That model earned some trust on CYP3A4 -- it")
    print("predicts whippoorwill's board MCC at their call rate to 0.0001 -- but it has")
    print("never been checked on CYP2D6. The similarity extrapolation is a straight line")
    print("fitted to five bins and read outside its range; treat it as a direction and a")
    print("rough size, not a predicted board value.")


if __name__ == "__main__":
    main()
