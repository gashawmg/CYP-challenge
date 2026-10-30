#!/usr/bin/env python
r"""The yardstick, weighted by what ST-RAE actually charges.

    python scripts\rho_yardstick.py --oof oof_mhens3.csv oof_mhtdi5.csv
    python scripts\rho_yardstick.py --oof oof_mhtdi5.csv --weight invwidth
    python scripts\rho_yardstick.py --oof oof_mhtdi5.csv --validate

WHY THIS CHANGED (Section 77). Every A/B this project has run was scored on plain macro
rho or plain OOF ST-RAE, in which every compound counts the same. The board does not count
them the same. ST-RAE charges the distance OUTSIDE each compound's credible interval, so a
narrow interval is an expensive compound and a wide one is nearly free. Measured on
oof_mhtdi5, our best column:

    endpoint   plain rho   cost-weighted rho   drop
    CYP1A2       0.5906          0.4986       -0.0920
    CYP2C9       0.7027          0.6706       -0.0320
    CYP2D6       0.4798          0.5081       +0.0283
    CYP3A4       0.8334          0.7397       -0.0938
    MACRO        0.6516          0.6042       -0.0474

0.0474 of rho is larger than every improvement this project measured in October put
together (TDI head +0.0070, seed ensembling ~+0.02). We were selecting models with a ruler
that disagrees with the board by more than the differences we were trying to resolve.

The leaderboard says the same thing from the other side. furtivepeony at rank 11 has a
WORSE ranking than ours (rho 0.7554 against 0.7558) and scores 0.4348 against our 0.4737.
101% of that gap is conversion and -1% is ranking. Against rank 1, conversion is 80%.

THE WEIGHT, AND WHY IT IS NOT 1/width. Section 69 rejected 1/width weighting because the
cost is U-shaped -- the WIDEST quartile carries 34-46% of the bill on three endpoints. That
observation was right and the conclusion drawn from it was wrong. Two different quantities:

    total cost       where the bill currently falls -- U-shaped, because we miss the
                     extremes so badly that even a wide interval does not save us
    marginal charge  what ONE unit of error costs -- 1 outside the interval, 0 inside

Training and model selection should follow the marginal charge. For a compound whose
interval is half-width h and an error scale s, the chance an error is charged at all is

    w = P(|e| > h)  with  e ~ N(0, s)   ->   w = 2 * (1 - Phi(h / s))

which is high for narrow intervals AND high for wide-interval compounds we miss badly --
both arms of the U. 1/width only captures the first arm, which is why it was the wrong
proxy.

THE SCALE s IS DELIBERATELY MODEL-INDEPENDENT. It is fitted ONCE from a reference column
(--ref, default oof_mhtdi5.csv) and then applied unchanged to every model compared in the
run. Letting each model supply its own s would grade each on its own curve: a model with
huge errors would declare every compound "expensive" and score itself generously. The
reference is printed on every run so a comparison can never be quoted without it.

--validate checks the yardstick rather than the models: across the columns given, it
reports how well plain rho and cost-weighted rho each predict the placed ST-RAE those
columns actually achieve. A yardstick that predicts the thing we are scored on worse than
the old one does not deserve to replace it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "cyp-challenge-TRAIN_inhibition.csv"
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
ID = "Molecule_Name"


def cost(p, lo, hi):
    return np.clip(p - hi, 0, None) + np.clip(lo - p, 0, None)


def opt_shift_cost(q, lo, hi):
    """Exact optimal shift: the objective is piecewise linear in s, so the minimiser sits
    at a breakpoint and the derivative #(A<s) - #(B>s) crosses zero there."""
    A = np.sort(hi - q)
    B = np.sort(lo - q)
    cand = np.sort(np.concatenate([A, B]))
    d = np.searchsorted(A, cand, side="left") - (len(B) - np.searchsorted(B, cand, side="right"))
    s = cand[min(int(np.searchsorted(d, 0, side="left")), len(cand) - 1)]
    return np.clip(q + s - hi, 0, None).sum() + np.clip(lo - q - s, 0, None).sum()


def affine_placed(p, lo, hi, y):
    """ST-RAE after the best affine re-placement -- fair to both arms, since placement is
    already solved (Section 69) and comparing raw ST-RAE mostly compares two accidents."""
    mu = p.mean()
    best = min(opt_shift_cost(b * (p - mu), lo, hi) for b in np.linspace(0.2, 2.2, 41))
    return best / cost(np.full_like(y, y.mean()), lo, hi).sum()


def charge_weight(lo, hi, s, kind):
    """Per-compound weight: how likely a unit of error is charged at all."""
    h = np.maximum((hi - lo) / 2.0, 1e-6)
    if kind == "none":
        return np.ones_like(h)
    if kind == "invwidth":
        return 1.0 / np.maximum(hi - lo, 0.05)
    return 2.0 * (1.0 - norm.cdf(h / max(s, 1e-6)))        # "poutside", the default


def weighted_spearman(y, p, w):
    ry = pd.Series(y).rank().to_numpy()
    rp = pd.Series(p).rank().to_numpy()
    my = np.average(ry, weights=w)
    mp = np.average(rp, weights=w)
    cov = np.average((ry - my) * (rp - mp), weights=w)
    vy = np.average((ry - my) ** 2, weights=w)
    vp = np.average((rp - mp) ** 2, weights=w)
    return float(cov / np.sqrt(vy * vp)) if vy > 0 and vp > 0 else float("nan")


def load(path, tr):
    p = Path(path)
    if not p.exists():
        p = ROOT / path
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d.columns = [c.strip() for c in d.columns]
    return d.set_index(ID)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True)
    ap.add_argument("--iso", nargs="*", default=ISO)
    ap.add_argument("--ref", default="oof_mhtdi5.csv",
                    help="column the per-endpoint error scale s is fitted from, ONCE, and "
                         "then applied to every model in the run")
    ap.add_argument("--weight", choices=["poutside", "invwidth", "none"], default="poutside")
    ap.add_argument("--validate", action="store_true",
                    help="score the YARDSTICK: which rho better predicts placed ST-RAE")
    ap.add_argument("--train", default=str(TRAIN))
    a = ap.parse_args()

    tr = pd.read_csv(a.train)
    tr.columns = [c.strip() for c in tr.columns]
    tr = tr.set_index(ID)

    # ---- the error scale, fitted once from the reference column -------------------
    ref = load(a.ref, tr)
    S = {}
    for iso in a.iso:
        c = f"{iso}_pIC50_direct_inhibition"
        if ref is None or c not in ref.columns:
            S[iso] = float(tr[c].std()) * 0.5
            continue
        idx = ref.index.intersection(tr.index)
        idx = idx[tr.loc[idx, c].notna() & ref.loc[idx, c].notna()]
        r = tr.loc[idx, c].to_numpy() - ref.loc[idx, c].to_numpy()
        S[iso] = float(np.sqrt(np.mean((r - r.mean()) ** 2)))
    print(f"weight = {a.weight}" + (f"   error scale s fitted from {a.ref}: "
          + "  ".join(f"{i} {S[i]:.3f}" for i in a.iso) if a.weight == "poutside" else ""))
    print()

    print(f"{'oof':<22}{'iso':<8}{'n':>6}{'rho':>8}{'COST-WTD':>10}{'drop':>8}"
          f"{'AUC hi|lo':>11}{'ST-RAE':>9}{'placed':>9}")
    rows = []
    for f in a.oof:
        d = load(f, tr)
        if d is None:
            print(f"  missing: {f}")
            continue
        pl, cw, pz = [], [], []
        for iso in a.iso:
            c = f"{iso}_pIC50_direct_inhibition"
            if c not in d.columns:
                continue
            idx = d.index.intersection(tr.index)
            idx = idx[tr.loc[idx, c].notna() & d.loc[idx, c].notna()]
            if len(idx) < 100:
                continue
            y = tr.loc[idx, c].to_numpy()
            p = d.loc[idx, c].to_numpy()
            lo = tr.loc[idx, c + "_conf_low"].to_numpy()
            hi = tr.loc[idx, c + "_conf_high"].to_numpy()
            w = charge_weight(lo, hi, S[iso], a.weight)
            r = spearmanr(y, p).statistic
            rw = weighted_spearman(y, p, w)
            q1, q9 = np.quantile(y, 0.10), np.quantile(y, 0.90)
            low, high = y <= q1, y >= q9
            sc = np.concatenate([p[low], p[high]])
            lab = np.concatenate([np.zeros(low.sum()), np.ones(high.sum())])
            o = np.argsort(sc)
            rr = np.empty(len(sc))
            rr[o] = np.arange(len(sc))
            n1 = int(lab.sum())
            auc = (rr[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * (len(lab) - n1))
            base = cost(np.full_like(y, y.mean()), lo, hi).sum()
            raw = cost(p, lo, hi).sum() / base
            placed = affine_placed(p, lo, hi, y)
            pl.append(r); cw.append(rw); pz.append(placed)
            print(f"{Path(f).name:<22}{iso:<8}{len(idx):>6}{r:>8.4f}{rw:>10.4f}{rw-r:>+8.4f}"
                  f"{auc:>11.4f}{raw:>9.4f}{placed:>9.4f}")
        if pl:
            rows.append((Path(f).name, float(np.mean(pl)), float(np.mean(cw)), float(np.mean(pz))))
        print()

    if rows:
        print("=== macro ===")
        print(f"{'oof':<22}{'rho':>9}{'COST-WTD rho':>14}{'placed ST-RAE':>15}")
        for n, p_, c_, z in rows:
            print(f"{n:<22}{p_:>9.4f}{c_:>14.4f}{z:>15.4f}")
        if len(rows) > 1:
            best_plain = max(rows, key=lambda r: r[1])[0]
            best_cw = max(rows, key=lambda r: r[2])[0]
            best_st = min(rows, key=lambda r: r[3])[0]
            print(f"\n  best by plain rho      : {best_plain}")
            print(f"  best by cost-wtd rho   : {best_cw}")
            print(f"  best by placed ST-RAE  : {best_st}   <- the one that matters")
            if best_cw == best_st and best_plain != best_st:
                print("  the cost-weighted ruler picked the right column and the plain one did not.")

    if a.validate and len(rows) > 2:
        pr = np.array([r[1] for r in rows]); cwv = np.array([r[2] for r in rows])
        zv = np.array([r[3] for r in rows])
        print("\n=== validating the YARDSTICK (does it predict what we are scored on?) ===")
        print(f"  columns compared: {len(rows)}")
        print(f"  corr(plain rho,        placed ST-RAE) = {spearmanr(pr, zv).statistic:+.4f}")
        print(f"  corr(cost-weighted rho, placed ST-RAE) = {spearmanr(cwv, zv).statistic:+.4f}")
        print("  (both should be strongly NEGATIVE; the more negative one is the better ruler)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
