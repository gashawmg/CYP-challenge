#!/usr/bin/env python
"""Phase 2: multi-task network, scored against the Phase 1 ladder on the frozen folds.

    python scripts/run_mtl.py --heads pic50
    python scripts/run_mtl.py --heads pic50+log2fc+tdiarm+emax+istdi --predict

`--heads` selects which auxiliary supervision is switched on, so each one has to earn
its place rather than being assumed to help. `pic50` alone is multi-task across the four
isoforms only; adding heads brings in the primary screen, the TDI arms, Emax, and the
TDI labels for all four isoforms.

Reports the same three numbers as Phase 1 — leaderboard ST-RAE, the dense proxy (the one
that actually transferred), and macro MCC — so results are directly comparable.
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
from cypkit import io, labels as L, mtl, scoring, splits, tdi as T  # noqa: E402
from scripts.run_baselines import build_features  # noqa: E402

ALL_HEADS = ("pic50", "log2fc", "tdiarm", "emax", "istdi")


def dense_strae(pred_wide, label_matrix, ids):
    """The dense proxy, per endpoint, from a wide prediction frame."""
    pos = {m: i for i, m in enumerate(ids)}
    out = []
    for j, cyp in enumerate(C.ISOFORMS):
        ep = f"{cyp}_pIC50_direct_inhibition"
        rows = label_matrix[label_matrix["endpoint"] == ep]
        idx = rows[C.ID_COL].map(pos).to_numpy()
        keep = np.array([i is not None and not pd.isna(i) for i in idx])
        rows = rows[keep]
        p = pred_wide[np.asarray(idx[keep], dtype=int), j]
        lo = rows.y_low.to_numpy(float)
        hi = rows.y_high.to_numpy(float)
        y = rows.y.to_numpy(float)
        yref = np.where(np.isfinite(y), y, C.PIC50_FLOOR)
        err = np.where(np.isfinite(hi), np.clip(p - hi, 0, None), 0) + np.where(
            np.isfinite(lo), np.clip(lo - p, 0, None), 0
        )
        m = yref.mean()
        base = np.where(np.isfinite(hi), np.clip(m - hi, 0, None), 0) + np.where(
            np.isfinite(lo), np.clip(lo - m, 0, None), 0
        )
        out.append(err.sum() / base.sum())
    return float(np.mean(out)), out


def train_fold(X_tr, T_tr, X_va, T_va, heads, args, n_features):
    import torch  # noqa: PLC0415

    torch.manual_seed(args.seed)
    model = mtl.make_model(n_features, hidden=tuple(args.hidden), dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def to_t(d, sl=None):
        return {k: torch.from_numpy(v if sl is None else v[sl]) for k, v in d.items()}

    Xt = torch.from_numpy(X_tr)
    Tt = to_t(T_tr)
    Xv = torch.from_numpy(X_va)
    Tv = to_t(T_va)

    n = len(X_tr)
    best, best_state, patience = np.inf, None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, args.batch):
            b = perm[s : s + args.batch]
            if len(b) < 8:
                continue
            out = model(Xt[b])
            loss = 0.0
            if "pic50" in heads:
                loss = loss + mtl.dead_zone_loss_torch(
                    out["pic50"], Tt["pic50_lo"][b], Tt["pic50_hi"][b],
                    Tt["pic50_mask"][b], delta=args.delta,
                )
            for h, w in (("log2fc", args.w_aux), ("tdiarm", args.w_aux), ("emax", args.w_aux)):
                if h in heads:
                    loss = loss + w * mtl.masked_l1(out[h], Tt[h][b], Tt[h + "_mask"][b])
            if "istdi" in heads:
                loss = loss + args.w_tdi * mtl.masked_bce(
                    out["istdi"], Tt["istdi"][b], Tt["istdi_mask"][b], pos_weight=args.pos_weight
                )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            ov = model(Xv)
            vloss = mtl.dead_zone_loss_torch(
                ov["pic50"], Tv["pic50_lo"], Tv["pic50_hi"], Tv["pic50_mask"], delta=args.delta
            ).item()
        if vloss < best - 1e-5:
            best, patience = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        o = model(Xv)
        return (
            o["pic50"].numpy(),
            torch.sigmoid(o["istdi"]).numpy(),
            model,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", default="pic50+log2fc+tdiarm+emax+istdi")
    ap.add_argument("--features", default="rdkit+cyp+ecfp+maccs")
    ap.add_argument("--hidden", type=int, nargs="*", default=[512, 256])
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--w-aux", type=float, default=0.3)
    ap.add_argument("--w-tdi", type=float, default=0.3)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--predict", action="store_true",
                    help="also train on everything and write test predictions")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    heads = [h for h in args.heads.split("+") if h]
    bad = [h for h in heads if h not in ALL_HEADS]
    if bad:
        raise SystemExit(f"unknown head(s) {bad}; choose from {ALL_HEADS}")
    print(f"heads: {heads}")

    train = io.load("train_inhibition", args.data_dir)
    screen = io.load("single_conc", args.data_dir)
    tdi_frame = io.load("train_tdi", args.data_dir)
    emax = io.load("train_emax", args.data_dir)
    test = io.load("test", args.data_dir)

    label_matrix = L.build_label_matrix(train, screen, include_censored=True)
    folds = splits.load_folds(C.artifact("folds.csv"))
    n_folds = int(folds["fold"].max()) + 1
    fold_of = dict(zip(folds[C.ID_COL], folds["fold"]))

    smiles = {}
    for f in (train, tdi_frame, emax, test):
        smiles.update(dict(zip(f[C.ID_COL], f[C.SMILES_COL])))
    smiles.update(dict(zip(label_matrix[C.ID_COL], label_matrix[C.SMILES_COL])))
    smiles.update(dict(zip(screen[C.ID_COL], screen[C.SMILES_COL])))
    smiles = {k: v for k, v in smiles.items() if isinstance(v, str)}

    extra = sorted(set(smiles) - set(fold_of) - set(test[C.ID_COL]))
    if extra:
        print(f"assigning {len(extra)} unlabelled-in-folds compounds by nearest neighbour...")
        fold_of.update(dict(zip(extra, splits.assign_extra_to_folds([smiles[i] for i in extra], folds))))

    print(f"featurising {len(smiles)} compounds...")
    t0 = time.time()
    ids, X, _ = build_features(smiles, False, args.features)
    print(f"  X = {X.shape}  ({time.time()-t0:.1f}s)")

    targets = mtl.build_targets(ids, label_matrix, screen, tdi_frame, emax)
    print("\nsupervision available per head")
    print(mtl.target_summary(targets).to_string(index=False))

    # Scale once per fold on the training half only.
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    test_ids = set(test[C.ID_COL])
    fold_vec = np.array([fold_of.get(m, -1) for m in ids])
    is_test = np.array([m in test_ids for m in ids])
    trainable = (~is_test) & (targets["pic50_mask"].sum(1) > 0)
    print(f"\ntrainable compounds: {int(trainable.sum())}  (test held out: {int(is_test.sum())})")

    oof = np.full((len(ids), len(C.ISOFORMS)), np.nan, np.float32)
    oof_tdi = np.full((len(ids), len(C.ISOFORMS)), np.nan, np.float32)

    for f in range(n_folds):
        va = trainable & (fold_vec == f)
        tr = trainable & (fold_vec != f)
        if va.sum() == 0 or tr.sum() < 100:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr = np.clip(sc.transform(X[tr]), -10, 10).astype(np.float32)
        Xva = np.clip(sc.transform(X[va]), -10, 10).astype(np.float32)
        Ttr = {k: v[tr] for k, v in targets.items()}
        Tva = {k: v[va] for k, v in targets.items()}
        t1 = time.time()
        p, ptdi, _ = train_fold(Xtr, Ttr, Xva, Tva, heads, args, X.shape[1])
        oof[va] = p
        oof_tdi[va] = ptdi
        print(f"  fold {f}: train={int(tr.sum())} val={int(va.sum())} ({time.time()-t1:.0f}s)")

    # ---- score, exactly as Phase 1 did ----
    got = ~np.isnan(oof[:, 0])
    frame = pd.DataFrame({C.ID_COL: [m for m, g in zip(ids, got) if g]})
    for j, cyp in enumerate(C.ISOFORMS):
        frame[f"{cyp}_pIC50_direct_inhibition"] = oof[got, j]
    # the scorer needs a prediction for every ground-truth compound
    gt_ids = set(scoring.build_ground_truth(train)[C.ID_COL])
    missing = gt_ids - set(frame[C.ID_COL])
    if missing:
        med = frame[[f"{c}_pIC50_direct_inhibition" for c in C.ISOFORMS]].median()
        pad = pd.DataFrame({C.ID_COL: sorted(missing)})
        for c in C.ISOFORMS:
            pad[f"{c}_pIC50_direct_inhibition"] = med[f"{c}_pIC50_direct_inhibition"]
        frame = pd.concat([frame, pad], ignore_index=True)
        print(f"  (padded {len(missing)} compounds with the median prediction)")

    summary = scoring.oof_report(frame, train, n_bootstrap=100)
    print("\n=== leaderboard-identical ST-RAE (optimistic — hit-selected only) ===")
    print(scoring.format_report(summary))

    dm, per = dense_strae(oof, label_matrix, ids)
    print("\n=== dense proxy (the scorer that transferred) ===")
    for cyp, v in zip(C.ISOFORMS, per):
        print(f"  {cyp:8s} {v:.4f}")
    print(f"  MACRO    {dm:.4f}")
    print("\n  Phase 1 reference (LightGBM): censored_imputed 0.6738 | l1 1.1157 -> LB 1.0127")

    if "istdi" in heads:
        from sklearn.metrics import matthews_corrcoef  # noqa: PLC0415

        print("\n=== TDI (macro over the two scored isoforms) ===")
        mccs = []
        for j, cyp in enumerate(C.ISOFORMS):
            if cyp not in C.TDI_ISOFORMS:
                continue
            m = (targets["istdi_mask"][:, j] > 0) & ~np.isnan(oof_tdi[:, j]) & trainable
            if m.sum() < 50:
                continue
            y = targets["istdi"][m, j].astype(bool)
            s = oof_tdi[m, j]
            t, mc = T.best_mcc_threshold(y, s)
            mccs.append(mc)
            print(f"  {cyp:8s} n={int(m.sum()):5d}  MCC={mc:.4f}  (threshold {t:.3f})")
        if mccs:
            print(f"  MACRO    {np.mean(mccs):.4f}")
            print("  Phase 1 reference (LightGBM): 0.2524 OOF -> 0.285 leaderboard")

    if args.predict:
        print("\ntraining final model on all trainable compounds...")
        sc = StandardScaler().fit(X[trainable])
        Xtr = np.clip(sc.transform(X[trainable]), -10, 10).astype(np.float32)
        Xte = np.clip(sc.transform(X[is_test]), -10, 10).astype(np.float32)
        Ttr = {k: v[trainable] for k, v in targets.items()}
        # a small held-out slice purely for early stopping
        rs = np.random.default_rng(args.seed).permutation(int(trainable.sum()))
        cut = int(0.9 * len(rs))
        tr_i, va_i = rs[:cut], rs[cut:]
        p_te, ptdi_te, _ = train_fold(
            Xtr[tr_i], {k: v[tr_i] for k, v in Ttr.items()},
            np.vstack([Xtr[va_i], Xte]),
            {k: np.vstack([v[va_i], np.zeros((int(is_test.sum()), v.shape[1]), v.dtype)])
             for k, v in Ttr.items()},
            heads, args, X.shape[1])
        n_va = len(va_i)
        test_pred, test_tdi = p_te[n_va:], ptdi_te[n_va:]
        test_ids_ord = [m for m, t in zip(ids, is_test) if t]
        out = pd.DataFrame({C.ID_COL: test_ids_ord})
        for j, cyp in enumerate(C.ISOFORMS):
            out[f"{cyp}_pIC50_direct_inhibition"] = test_pred[:, j]
            out[f"{cyp}_tdi_score"] = test_tdi[:, j]
        out = test[[C.SMILES_COL, C.ID_COL]].merge(out, on=C.ID_COL, how="left")
        out.to_csv(C.artifact(f"mtl_test{args.tag}.csv"), index=False)
        print(f"  written to artifacts/mtl_test{args.tag}.csv")

    np.save(C.artifact(f"mtl_oof{args.tag}.npy"), oof)
    np.save(C.artifact(f"mtl_oof_tdi{args.tag}.npy"), oof_tdi)
    pd.Series(ids).to_csv(C.artifact("mtl_ids.csv"), index=False, header=["Molecule_Name"])
    print(f"\nOOF saved to artifacts/mtl_oof{args.tag}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
