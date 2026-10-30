#!/usr/bin/env python
r"""Train TDI on the SHIFT instead of the binary, and measure whether it moves d'.

    python train_tdi_shift.py                          # both isoforms, LightGBM
    python train_tdi_shift.py --iso CYP3A4 --folds 5
    python train_tdi_shift.py --backbone chemprop      # if chemprop is importable

WHAT IS BEING TESTED, AND WHY THE CONTROL MATTERS MORE THAN THE CANDIDATE.
`is_TDI` is a threshold applied to the IC50 shift, and the rule is recoverable:
CYP3A4 is a significance test (`shift - 1.96*pooled_sd`, AUC 0.9865 against the label),
CYP2D6 a magnitude threshold (raw shift, AUC 0.9615). We have always trained on the
thresholded version, so a compound with a shift of +2.0 and one at +0.31 were the same
"1" and everything near the boundary was a coin flip fit as a hard label.

This script therefore trains TWO models on IDENTICAL features and IDENTICAL folds, with
exactly one difference:

    control     classifier on  is_TDI            <- what we have always done
    candidate   regressor  on  the shift, weighted by 1/pooled_sd

Both are then scored the same way: AUC and binormal d' of the OOF ranking against the
binary label, which is the quantity the leaderboard MCC is a function of. Reporting the
control from the same run is the whole point -- it separates "the target was wrong" from
"the backbone or features changed", and this project has been burned by exactly that
confusion before.

THE COST IS PRINTED, NOT HIDDEN. CYP3A4 has 3,584 labelled rows but only 2,334 with both
arms, so the regression trades ~35% of the labels for a better target. CYP2D6 loses 4
rows -- for that isoform the change is nearly free. `--two-head` keeps the binary-only
CYP3A4 rows by rank-averaging a classifier trained on them into the regression ranking.

WHAT THIS DOES NOT DO. It does not pick a threshold. scripts/tdi_diagnose.py already
showed both targets sit at their MCC optimum for the CURRENT d', and that every remaining
point needs a better d'. This script only asks whether the target change delivers one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import matthews_corrcoef, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from cypkit import config as C, cyp_descriptors as CD, io  # noqa: E402
from cypkit import tdi_alerts as TA, tdi_geometry as TG, tdi_shift as TS  # noqa: E402

# call rates recovered from the leaderboard by scripts/tdi_diagnose.py -- the MCC optimum
# for our current discrimination. Held fixed here so the comparison is about d' only.
CALL_RATE = {"CYP3A4": 0.174, "CYP2D6": 0.080}


def features(smiles: list[str], verbose: bool = True) -> tuple[np.ndarray, list[str]]:
    """Bioactivation alerts + basic-N-to-warhead geometry + the general CYP block."""
    blocks, names = [], []
    for mod, tag in ((TA, "alert"), (TG, "geom"), (CD, "cyp")):
        X, _ = mod.calculate(smiles)
        blocks.append(np.asarray(X, dtype=np.float64))
        names += [f"{tag}:{n}" for n in mod.feature_names()]
    X = np.hstack(blocks)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if verbose:
        print(f"  features: {X.shape[1]} columns "
              f"({', '.join(f'{t} {b.shape[1]}' for b, t in zip(blocks, ('alerts', 'geometry', 'cyp')))})")
    return X, names


def dprime(y: np.ndarray, score: np.ndarray) -> float:
    """Binormal d' from AUC -- the quantity tdi_diagnose.py works in."""
    auc = roc_auc_score(y, score)
    return float(np.sqrt(2) * norm.ppf(min(max(auc, 1e-6), 1 - 1e-6)))


def dprime_gap_ci(y, s_base, s_cand, n_boot: int = 2000, seed: int = 0):
    """Paired bootstrap CI for d'(candidate) - d'(control), and P(candidate is worse).

    WHY THIS EXISTS. This script's verdict used to be a bare threshold on the gap, which
    cannot tell a small real effect from noise. The two arms are scored on THE SAME
    compounds, so the comparison must be paired: resample compounds once per replicate
    and recompute both d' on that same resample. Their errors are strongly correlated,
    so the paired interval is far tighter than either arm's own -- treating them as
    independent would hide a real gain, and eyeballing the gap would invent one.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        if y[b].min() == y[b].max():          # a replicate with one class has no AUC
            continue
        out.append(dprime(y[b], s_cand[b]) - dprime(y[b], s_base[b]))
    out = np.asarray(out)
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)), float((out <= 0).mean())


def mcc_at_rate(y: np.ndarray, score: np.ndarray, rate: float) -> float:
    k = max(1, int(round(rate * len(score))))
    thr = np.sort(score)[::-1][k - 1]
    return float(matthews_corrcoef(y, (score >= thr).astype(int)))


def _chemprop_fit(smi_tr, Xtr, ytr, wtr, smi_te, Xte, task, cfg):
    """Mirror of train_tdi_chemprop.py's arrangement: learned D-MPNN representation
    concatenated with the target-specific descriptor block, then an FFN.

    Two things differ from that script and both matter here:
      - RegressionFFN for the shift target, BinaryClassificationFFN for the control, so
        the two arms differ ONLY in the target, which is the whole point of the run;
      - the regression target is standardised on the TRAINING half and unscaled after.
        chemprop's regression head has no output transform wired here, and an unscaled
        pIC50-shift target trains badly.
    """
    import warnings  # noqa: PLC0415
    import lightning.pytorch as pl  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from lightning.pytorch.callbacks import EarlyStopping  # noqa: PLC0415
    from rdkit import Chem  # noqa: PLC0415
    from sklearn.preprocessing import RobustScaler  # noqa: PLC0415
    from chemprop.data import (MoleculeDatapoint, MoleculeDataset,  # noqa: PLC0415
                               build_dataloader)
    from chemprop.models import MPNN  # noqa: PLC0415
    from chemprop.nn import BondMessagePassing, MeanAggregation  # noqa: PLC0415
    from chemprop.nn.predictors import (BinaryClassificationFFN,  # noqa: PLC0415
                                        RegressionFFN)
    warnings.filterwarnings("ignore")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def loader(ds, shuffle):
        """Build a dataloader without depending on this chemprop version's signature.

        chemprop 2.2.1's `build_dataloader` does NOT take `drop_last`. It computes its
        own and passes it positionally to DataLoader alongside `**kwargs`, so supplying
        the kwarg collides:  TypeError: got multiple values for keyword argument
        'drop_last'.  Other versions do accept it. Rather than branch on a version
        string, we pass NO such kwarg and defeat the rule it exists for. Verbatim from
        chemprop/data/dataloader.py:

            if len(dataset) % batch_size == 1:
                ... drop_last = True      # to protect batch norm from a batch of one

        So any batch size that cannot leave a remainder of exactly one is safe. We search
        outward from the configured size for the nearest such value -- a loop, not a
        single decrement: n = k*b + 1 can also satisfy n % (b-1) == 1 whenever k is a
        multiple of b-1 (b=64, n=4033 is the first), and a one-step fix would silently
        drop a row there and resurrect the shape mismatch this replaced.
        """
        n, b0 = len(ds), cfg["batch"]
        b = None
        if n > 1:
            for c in [b0] + [x for i in range(1, 16) for x in (b0 - i, b0 + i)]:
                if c >= 2 and n % c != 1:
                    b = c
                    break
        if b is None:                       # unreachable for any realistic n; n % n == 0
            b = max(2, n)
        return build_dataloader(ds, batch_size=b, shuffle=shuffle, num_workers=0)

    sc = RobustScaler().fit(Xtr)
    A = np.clip(sc.transform(Xtr), -10, 10)
    B = np.clip(sc.transform(Xte), -10, 10)
    if task == "reg":
        mu, sd = float(np.mean(ytr)), float(np.std(ytr)) or 1.0
        ytr_s = (np.asarray(ytr, float) - mu) / sd
    else:
        mu, sd, ytr_s = 0.0, 1.0, np.asarray(ytr, float)

    def dps(smiles, y, desc, w=None):
        """Returns (datapoints, kept_indices).

        RDKit fails on a small number of the challenge SMILES, and a datapoint list that
        silently skips them comes back SHORTER than the input. train_tdi_chemprop.py hides
        that with `.ravel()[:n]`, which is worse than crashing: truncating shifts every
        prediction after the dropped molecule onto the wrong compound. Returning the kept
        indices lets the caller scatter predictions back to their real positions.
        """
        out, kept = [], []
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            kw = {"x_d": desc[i].astype(np.float32)}
            if w is not None:
                kw["weight"] = float(w[i])
            out.append(MoleculeDatapoint(mol=mol,
                                         y=np.array([float(y[i])], np.float32), **kw))
            kept.append(i)
        return out, np.asarray(kept, int)

    mp = BondMessagePassing(d_h=cfg["hidden"], depth=cfg["depth"],
                            dropout=cfg["dropout"], activation="relu")
    FFN = RegressionFFN if task == "reg" else BinaryClassificationFFN
    ffn = FFN(input_dim=cfg["hidden"] + A.shape[1], hidden_dim=cfg["hidden"],
              n_layers=cfg["layers"], dropout=cfg["dropout"])
    model = MPNN(message_passing=mp, agg=MeanAggregation(), predictor=ffn,
                 batch_norm=True, X_d_transform=None)

    # a slice of training is held out purely for early stopping; it never reaches the
    # reported out-of-fold predictions, which come from the outer CV fold
    n = len(smi_tr)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    cut = max(1, int(0.9 * n))
    tr_i, va_i = idx[:cut], idx[cut:]
    tr_dp, _ = dps([smi_tr[i] for i in tr_i], ytr_s[tr_i], A[tr_i],
                   None if wtr is None else np.asarray(wtr)[tr_i])
    va_dp, _ = dps([smi_tr[i] for i in va_i], ytr_s[va_i], A[va_i])
    tl = loader(MoleculeDataset(tr_dp), True)
    vl = loader(MoleculeDataset(va_dp), False)
    trainer = pl.Trainer(accelerator="gpu" if dev == "cuda" else "cpu", devices=1,
                         max_epochs=cfg["epochs"], logger=False,
                         enable_progress_bar=False, enable_checkpointing=False,
                         gradient_clip_val=1.0,
                         callbacks=[EarlyStopping(monitor="val_loss",
                                                  patience=cfg["patience"], mode="min")])
    trainer.fit(model, tl, vl)
    te_dp, te_keep = dps(list(smi_te), np.zeros(len(smi_te)), B)
    dl = loader(MoleculeDataset(te_dp), False)
    with torch.inference_mode():
        p = trainer.predict(model, dl)
    got = np.concatenate([x.cpu().numpy() for x in p]).ravel()
    # SCATTER, do not truncate. Unparseable molecules get the training mean, which for a
    # ranking metric is the neutral value; the alternative is silently shifting every
    # later prediction onto the wrong compound.
    out = np.full(len(smi_te), float(np.mean(ytr_s)), float)
    if len(got) != len(te_keep):
        import chemprop as _cp  # noqa: PLC0415
        print(f"      chemprop {getattr(_cp, '__version__', 'unknown')}, "
              f"batch {cfg['batch']}, dataset {len(te_dp)}")
        # We know one way chemprop can shorten a prediction (drop_last) and it is now
        # disabled. If the lengths still disagree, something else is dropping rows and
        # the honest response is to say so, not to pad quietly and report a clean number.
        raise RuntimeError(
            f"chemprop returned {len(got)} predictions for {len(te_keep)} molecules "
            f"({len(smi_te)} requested). Padding here would silently misalign "
            f"predictions with compounds -- fix the cause before trusting any result.")
    out[te_keep] = got
    if len(te_keep) < len(smi_te):
        print(f"      {len(smi_te) - len(te_keep)} unparseable SMILES filled with the "
              f"training mean")
    return out * sd + mu


def fit_predict(backbone: str, Xtr, ytr, wtr, Xte, task: str,
                smi_tr=None, smi_te=None, cfg=None):
    if backbone == "lgbm":
        import lightgbm as lgb  # noqa: PLC0415
        common = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                      min_child_samples=20, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=-1)
        m = (lgb.LGBMRegressor(**common) if task == "reg"
             else lgb.LGBMClassifier(**common))
        m.fit(Xtr, ytr, sample_weight=wtr)
        return (m.predict(Xte) if task == "reg" else m.predict_proba(Xte)[:, 1])
    if backbone == "chemprop":
        if smi_tr is None or smi_te is None:
            raise SystemExit("chemprop needs SMILES; this call site did not pass them")
        # A SMILES list and a descriptor matrix that disagree in length are not paired
        # row-for-row, and chemprop would train on mismatched molecules without ever
        # complaining. lgbm never reads the SMILES, so nothing else in this script can
        # catch it -- assert here, where the two are first put side by side.
        if len(smi_tr) != len(Xtr) or len(smi_te) != len(Xte):
            raise RuntimeError(
                f"SMILES/descriptor length mismatch: train {len(smi_tr)} vs {len(Xtr)}, "
                f"test {len(smi_te)} vs {len(Xte)}. The two are supposed to be the same "
                f"rows in the same order; fix the indexing rather than trimming.")
        return _chemprop_fit(smi_tr, Xtr, ytr, wtr, smi_te, Xte, task, cfg)
    raise SystemExit(f"unknown backbone '{backbone}'")


def run_isoform(iso: str, a) -> dict:
    print(f"\n=== {iso} " + "=" * 56)
    tab = TS.build(iso, a.data_dir)
    reg, clf_only = tab["reg"], tab["clf"]
    reg = reg[reg["y_bin"].notna()].reset_index(drop=True)   # need the label to score
    y = (reg["y_bin"] == True).to_numpy().astype(int)        # noqa: E712
    smi = reg[C.SMILES_COL].tolist()
    X, _ = features(smi)
    cfg = dict(hidden=a.hidden, depth=a.depth, layers=a.layers, dropout=a.dropout,
               batch=a.batch, epochs=a.epochs, patience=a.patience)
    def take(lst, idx):
        """Select rows of a python list by the index array StratifiedKFold yields.

        THIS WAS `np.where(m)[0]`, WHICH IS ONLY CORRECT FOR A BOOLEAN MASK.
        `skf.split` yields INTEGER indices, and np.where on those does two wrong
        things at once: it drops the entry whose value is 0 (hence the off-by-one
        `shape mismatch (777,) vs (778,)` on whichever fold contains row 0), and it
        returns POSITIONS rather than the indices themselves -- so every fold was
        handing chemprop `smi[0], smi[1], smi[2], ...` while the descriptor matrix
        beside it held the fold's actual rows. Wrong molecules, silently.

        LightGBM ignores smi_tr/smi_te entirely, which is why every lgbm result in
        this script is unaffected and why the bug stayed invisible until the chemprop
        backbone first read them.
        """
        idx = np.asarray(idx)
        if idx.dtype == bool:                      # tolerate a mask if one ever arrives
            idx = np.where(idx)[0]
        return [lst[int(i)] for i in idx]

    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=0)
    oof_reg = np.zeros(len(reg))
    oof_clf = np.zeros(len(reg))
    for tr, te in skf.split(X, y):
        # CANDIDATE: regress the shift, weighted by measurement precision
        oof_reg[te] = fit_predict(a.backbone, X[tr], reg["y_shift"].to_numpy()[tr],
                                  reg["w"].to_numpy()[tr], X[te], "reg",
                                  take(smi, tr), take(smi, te), cfg)
        # CONTROL: same rows, same features, same folds, binary target, no weights
        oof_clf[te] = fit_predict(a.backbone, X[tr], y[tr], None, X[te], "clf",
                                  take(smi, tr), take(smi, te), cfg)

    rows, scores = [], {}
    for name, s in (("control  (binary target)", oof_clf),
                    ("candidate (shift target)", oof_reg)):
        scores[name] = s
        rows.append(dict(model=name, auc=roc_auc_score(y, s), dprime=dprime(y, s),
                         mcc=mcc_at_rate(y, s, CALL_RATE[iso])))
    if a.two_head and len(clf_only) > 50:
        print(f"  --two-head: {len(clf_only)} binary-only rows folded back in")
        smi_o = clf_only[C.SMILES_COL].tolist()
        Xo, _ = features(smi_o, verbose=False)
        yo = (clf_only["y_bin"] == True).to_numpy().astype(int)   # noqa: E712
        extra = np.zeros(len(reg))
        for tr, te in skf.split(X, y):
            Xa = np.vstack([X[tr], Xo]); ya = np.concatenate([y[tr], yo])
            extra[te] = fit_predict(a.backbone, Xa, ya, None, X[te], "clf",
                                    take(smi, tr) + smi_o, take(smi, te), cfg)
        blend = (pd.Series(oof_reg).rank(pct=True)
                 + pd.Series(extra).rank(pct=True)).to_numpy() / 2
        scores["candidate + binary-only head"] = blend
        rows.append(dict(model="candidate + binary-only head", auc=roc_auc_score(y, blend),
                         dprime=dprime(y, blend), mcc=mcc_at_rate(y, blend, CALL_RATE[iso])))

    r = pd.DataFrame(rows)
    print(f"\n  {'model':30s} {'AUC':>7s} {'d’':>7s} {'MCC @ rate':>11s}")
    for _, x in r.iterrows():
        print(f"  {x.model:30s} {x.auc:7.4f} {x.dprime:7.3f} {x.mcc:11.4f}")
    base = r.iloc[0]
    best = r.loc[r.dprime.idxmax()]
    lo, hi, p_worse = dprime_gap_ci(y, scores[base.model], scores[best.model])
    print(f"\n  d' {base.dprime:.3f} -> {best.dprime:.3f}  ({best.model.strip()})")
    print(f"  paired bootstrap: gap {best.dprime - base.dprime:+.3f}, "
          f"95% CI [{lo:+.3f}, {hi:+.3f}], P(candidate worse) {p_worse:.0%}")
    if a.backbone == "lgbm":
        print(f"  NOTE these read LOWER than the blind d' of 1.276 (CYP3A4) / 1.224 "
              f"(CYP2D6)\n  that tdi_diagnose.py recovered, because this is a "
              f"descriptor-only LightGBM\n  baseline, not the ChemProp ensemble that "
              f"produced those submissions. The\n  ABSOLUTE numbers are not comparable; "
              f"only the control-to-candidate gap is,\n  and that gap is what the whole "
              f"script exists to measure.")
    else:
        print(f"  NOTE these still read below the blind d' of 1.276 (CYP3A4) / 1.224 "
              f"(CYP2D6):\n  same encoder, but a single model on {a.folds} folds rather "
              f"than the submitted\n  ensemble, and scored on the training compounds "
              f"rather than the blind set.")
    test_pred = None
    if a.predict:
        # Refit on every row that has the shift, then score the 750 blind compounds.
        # Nothing about the blind set enters the fit; this is the same model the OOF
        # numbers above describe, trained once on all of it.
        te_df = io.load("test", a.data_dir)
        te_df.columns = [c.strip() for c in te_df.columns]
        Xte, _ = features(te_df[C.SMILES_COL].tolist(), verbose=False)
        test_pred = pd.DataFrame({
            C.ID_COL: te_df[C.ID_COL],
            f"{iso}_shift_pred": fit_predict(a.backbone, X, reg["y_shift"].to_numpy(),
                                             reg["w"].to_numpy(), Xte, "reg",
                                             smi, te_df[C.SMILES_COL].tolist(), cfg)})
        print(f"  blind predictions: {len(test_pred)} compounds")

    return dict(iso=iso, table=r, test=test_pred, oof=pd.DataFrame({
        C.ID_COL: reg[C.ID_COL], f"{iso}_shift_pred": oof_reg,
        f"{iso}_clf_prob": oof_clf}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", choices=list(TS.TDI_ISOFORMS) + ["both"], default="both")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--backbone", choices=["lgbm", "chemprop"], default="lgbm")
    ap.add_argument("--two-head", action="store_true",
                    help="fold the binary-only rows back in via a second head "
                         "(matters for CYP3A4, which loses 35%% of its labels)")
    ap.add_argument("--data-dir", type=Path, default=None)
    # chemprop hyperparameters, matched to train_tdi_chemprop.py so the backbones are
    # compared at the settings that produced our submitted TDI models
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--out", default="oof_tdi_shift.csv")
    ap.add_argument("--predict", default=None,
                    help="also refit on ALL labelled rows and write blind-set "
                         "predictions here (e.g. pred_tdi_shift.csv)")
    a = ap.parse_args()

    isos = TS.TDI_ISOFORMS if a.iso == "both" else (a.iso,)
    results = [run_isoform(i, a) for i in isos]

    oof = results[0]["oof"]
    for r in results[1:]:
        oof = oof.merge(r["oof"], on=C.ID_COL, how="outer")
    oof.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(oof)} rows)")
    if a.predict:
        t = results[0]["test"]
        for r in results[1:]:
            t = t.merge(r["test"], on=C.ID_COL, how="outer")
        t.to_csv(a.predict, index=False)
        print(f"wrote {a.predict}  ({len(t)} rows)")

    print("\nVERDICT")
    for r in results:
        t = r["table"]
        gain = t.dprime.max() - t.dprime.iloc[0]
        verdict = ("the target change helps" if gain > 0.05 else
                   "no meaningful gain -- the binarisation was not the limit")
        print(f"  {r['iso']}: d' {t.dprime.iloc[0]:.3f} -> {t.dprime.max():.3f} "
              f"({gain:+.3f})  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
