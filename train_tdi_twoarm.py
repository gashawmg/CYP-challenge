#!/usr/bin/env python
r"""Predict both arms of the TDI assay on ONE shared trunk, then difference them.

    python train_tdi_twoarm.py                  # shared vs independent, both isoforms
    python train_tdi_twoarm.py --epochs 120

THE CLAIM BEING TESTED, AND THE ONE NUMBER THAT DECIDES IT. `is_TDI` thresholds the shift
between two arms of one assay. The arms correlate at 0.90-0.99 and the shift is only
17-45% of an arm's spread, so independent models would amplify error hopelessly. A shared
encoder makes the two heads' errors correlated, and correlated errors CANCEL in a
difference. Everything rests on how correlated they actually are, so this script measures
that directly instead of assuming it.

TWO ARMS, IDENTICAL FEATURES, FOLDS AND CAPACITY:

    shared        one trunk -> 8 heads (4 isoforms x 2 assay arms)
    independent   a separate trunk per head, same total width   <- the CONTROL

If the shared trunk is doing what the argument says, it shows up as a higher error
correlation between the two arms AND a better shift rho. If the error correlation is high
in both, the mechanism is not the sharing and the idea is wrong.

WHY AN MLP AND NOT CHEMPROP. chemprop does not import in this container, and this is a
question about ARCHITECTURE (shared vs not), not about the encoder's quality. A small MLP
on Morgan + descriptors answers it honestly and cheaply; if the mechanism holds, the same
structure ports to the chemprop trunk, where the absolute numbers would be far better.

SCORED AGAINST THE BINARY, because that is what the board uses: binormal d' of the
predicted shift as a ranking of is_TDI, next to our incumbent classifier on the same rows.
scripts/tdi_diagnose.py showed both targets already sit at their MCC optimum, so d' is the
only thing that can move MA-MCC, and rank 1 needs ours x1.205.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from cypkit import config as C, tdi_twoarm as TA  # noqa: E402
from train_pic50_interval import featurise  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class MultiHead(nn.Module):
    """shared=True: one trunk feeding every head. shared=False: a trunk per head."""

    def __init__(self, d_in: int, n_heads: int, width: int = 256, shared: bool = True):
        super().__init__()
        self.shared = shared
        def trunk(w):
            return nn.Sequential(nn.Linear(d_in, w), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(w, w // 2), nn.ReLU(), nn.Dropout(0.15))
        if shared:
            self.body = trunk(width)
            self.heads = nn.ModuleList([nn.Linear(width // 2, 1) for _ in range(n_heads)])
        else:
            # same total capacity, split across heads, so the comparison is architecture
            # and not parameter count
            w = max(32, width // n_heads * 2)
            self.bodies = nn.ModuleList([trunk(w) for _ in range(n_heads)])
            self.heads = nn.ModuleList([nn.Linear(w // 2, 1) for _ in range(n_heads)])

    def forward(self, x):
        if self.shared:
            h = self.body(x)
            return torch.cat([head(h) for head in self.heads], dim=1)
        return torch.cat([head(b(x)) for b, head in zip(self.bodies, self.heads)], dim=1)


def fit_fold(Xtr, Ytr, Xte, shared: bool, epochs: int, seed: int = 0):
    torch.manual_seed(seed)
    mu = np.nanmean(Ytr, axis=0); sd = np.nanstd(Ytr, axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Z = (Ytr - mu) / sd
    mask = torch.tensor(np.isfinite(Z).astype(np.float32))
    Zt = torch.tensor(np.nan_to_num(Z, nan=0.0).astype(np.float32))
    Xt = torch.tensor(Xtr.astype(np.float32))
    m = MultiHead(Xtr.shape[1], Ytr.shape[1], shared=shared).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, Zt, mask = Xt.to(DEV), Zt.to(DEV), mask.to(DEV)
    n = len(Xt)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=DEV)
        for k in range(0, n, 128):
            b = perm[k:k + 128]
            opt.zero_grad()
            p = m(Xt[b])
            # masked MSE: a head contributes only where that arm was measured
            loss = (((p - Zt[b]) ** 2) * mask[b]).sum() / mask[b].sum().clamp(min=1)
            loss.backward()
            opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        out = m(torch.tensor(Xte.astype(np.float32)).to(DEV)).cpu().numpy()
    return out * sd + mu


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--shift-head", dest="shift_head", action="store_true",
                    help="make the SHIFT a trained head instead of differencing two arms")
    ap.add_argument("--out", default="oof_tdi_twoarm.csv")
    a = ap.parse_args()

    tab = TA.build(shift_head=a.shift_head)
    X = featurise(tab["smiles"])
    Y, names = tab["Y"], tab["names"]
    print(f"\nfeatures {X.shape}, device {DEV}\n")

    kf = KFold(n_splits=a.folds, shuffle=True, random_state=0)
    preds = {}
    for shared in (True, False):
        P = np.zeros_like(Y)
        for f, (tr, te) in enumerate(kf.split(X)):
            P[te] = fit_fold(X[tr], Y[tr], X[te], shared, a.epochs, seed=f)
        preds["shared" if shared else "independent"] = P
        print(f"  {'shared trunk' if shared else 'independent trunks'}: trained")

    inc = pd.read_csv("oof_tdi_chemprop_best.csv")
    inc.columns = [c.strip() for c in inc.columns]
    inc = inc.set_index(C.ID_COL)
    ids = pd.Index(tab["ids"])
    rows = []
    for iso in TA.SCORED:
        ti, di = TA.shift_index(names, iso)
        lab = tab["labels"][iso]
        ok = pd.notna(lab) & np.isfinite(Y[:, ti]) & np.isfinite(Y[:, di])
        y = (lab[ok] == True).astype(int)                          # noqa: E712
        true_shift = Y[ok, ti] if a.shift_head else Y[ok, ti] - Y[ok, di]
        print(f"\n=== {iso}   {int(ok.sum())} rows with both arms and a label, "
              f"{y.mean():.1%} positive")
        print(f"  {'arm':14s} {'err corr':>9s} {'shift rho':>10s} {'AUC':>7s} {'d’':>7s}")
        for name, P in preds.items():
            # In shift-head mode column ti is the SHIFT, so correlating its error with
            # the direct arm's is not the cancellation quantity at all -- report NaN
            # rather than a number that looks like one and is not.
            if a.shift_head:
                ec = float("nan")
            else:
                e_t = P[ok, ti] - Y[ok, ti]
                e_d = P[ok, di] - Y[ok, di]
                ec = float(np.corrcoef(e_t, e_d)[0, 1])
            # with a shift head, column ti IS the prediction; otherwise difference them
            sh = P[ok, ti] if a.shift_head else P[ok, ti] - P[ok, di]
            sr = float(pd.Series(sh).corr(pd.Series(true_shift), method="spearman"))
            auc = roc_auc_score(y, sh)
            dp = float(np.sqrt(2) * norm.ppf(min(max(auc, 1e-6), 1 - 1e-6)))
            ecs = f"{ec:9.3f}" if np.isfinite(ec) else f"{'n/a':>9s}"
            print(f"  {name:14s} {ecs} {sr:10.4f} {auc:7.4f} {dp:7.3f}")
            rows.append(dict(isoform=iso, arm=name, err_corr=ec, shift_rho=sr,
                             auc=auc, dprime=dp))
        # the incumbent, on exactly these rows
        p = inc.reindex(ids[ok])[f"{iso}_is_TDI_prob"]
        m2 = p.notna().to_numpy()
        if m2.sum() > 200:
            auc = roc_auc_score(y[m2], p[m2])
            dp = float(np.sqrt(2) * norm.ppf(min(max(auc, 1e-6), 1 - 1e-6)))
            print(f"  {'INCUMBENT':14s} {'':9s} {'':10s} {auc:7.4f} {dp:7.3f}"
                  f"   <- chemprop classifier, same rows")
            rows.append(dict(isoform=iso, arm="incumbent", err_corr=np.nan,
                             shift_rho=np.nan, auc=auc, dprime=dp))

    r = pd.DataFrame(rows)
    print("\nVERDICT")
    for iso in TA.SCORED:
        g = r[r.isoform == iso].set_index("arm")
        s, i_, inc_d = g.loc["shared"], g.loc["independent"], g.loc["incumbent", "dprime"]
        print(f"  {iso}: error correlation shared {s.err_corr:.3f} vs independent "
              f"{i_.err_corr:.3f}")
        print(f"          d' shared {s.dprime:.3f}  independent {i_.dprime:.3f}  "
              f"incumbent {inc_d:.3f}")
        if not np.isfinite(s.err_corr):
            print("          -> shift-head mode: error correlation does not apply; "
                  "compare d' only")
        elif s.err_corr <= i_.err_corr + 0.02:
            print("          -> sharing did NOT raise the error correlation; the "
                  "mechanism is absent")
        elif s.dprime > i_.dprime + 0.05:
            print("          -> sharing raises BOTH the error correlation and d'; "
                  "the mechanism holds")
        else:
            print("          -> errors correlate more but d' did not follow")
    pd.DataFrame({C.ID_COL: tab["ids"],
                  **{f"{n}_shared": preds["shared"][:, j] for j, n in enumerate(names)}}
                 ).to_csv(a.out, index=False)
    r.to_csv("artifacts/tdi_twoarm_summary.csv", index=False)
    print(f"\nwrote {a.out} and artifacts/tdi_twoarm_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
