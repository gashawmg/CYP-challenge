#!/usr/bin/env python
r"""CYP3A4 TDI: does BINDING x BIOACTIVATION beat either one alone?

    python train_tdi_3a4_affinity.py

WHY CYP3A4 AND NOT CYP2D6. Section 25: CYP2D6 TDI is unmeasurable at our sample size -- three
instruments in a row read 4.37x low, inverted, or cost rank 9 to 36. CYP3A4's local d' of
1.300 against a blind 1.276 is a ratio of 0.98, so it is the one TDI endpoint where a
feature idea can be tested before it is spent. MA-MCC is the mean of the two, so a gain
here counts exactly as much.

THE MECHANISM, AND WHY THE CURRENT FEATURES CANNOT EXPRESS IT. Mechanism-based
inactivation needs TWO things at once: the compound must bind and turn over in the active
site, AND it must carry a group that becomes reactive when it does. cypkit/tdi_alerts.py
already encodes the second half well -- 43 substructure alerts plus MI-complex precursor
counts. Nothing encodes the first. A methylenedioxyphenyl on a compound that never enters
CYP3A4 is not a time-dependent inhibitor; the same group on a potent binder is the
textbook case.

We already have an affinity estimate for every compound, train and blind: our own
predicted CYP3A4 pIC50. The MEASURED value exists for training rows, but using it would
put a feature in training that the blind set gets only as a prediction, so the predicted
value is used on both sides.

FOUR ARMS, IDENTICAL FOLDS, and the third is the one that matters:

    control            the 134 existing TDI features
    +affinity          plus predicted CYP3A4 pIC50 alone
    +affinity x alerts plus the PRODUCTS only, no raw affinity term
    +both

If +affinity wins, binding predicts TDI and that is worth knowing but is not a
conjunction. If the products arm beats the raw-affinity arm, the CONJUNCTION is doing
work that neither block could express separately -- which is the actual hypothesis.

Paired bootstrap on every gap (Section 19). A gap inside its own interval is not a result, and
per Section 24 no submission follows from this script alone.
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

from cypkit import config as C, io, tdi_alerts as TA  # noqa: E402
from train_tdi_shift import features  # noqa: E402

ROOT = Path(__file__).resolve().parent
CALL_RATE = 0.174


def dprime(y, s):
    return float(np.sqrt(2) * norm.ppf(np.clip(roc_auc_score(y, s), 1e-6, 1 - 1e-6)))


def mcc_at(y, s, rate):
    k = max(1, int(round(rate * len(y))))
    return matthews_corrcoef(y, (s >= np.sort(s)[::-1][k - 1]).astype(int))


def boot(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    o = np.asarray([dprime(y[i], b[i]) - dprime(y[i], a[i])
                    for i in (rng.integers(0, len(y), len(y)) for _ in range(n))
                    if len(set(y[i])) > 1])
    return float(np.quantile(o, .025)), float(np.quantile(o, .975)), float((o <= 0).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    iso = "CYP3A4"

    tr = io.load("train_tdi"); tr.columns = [c.strip() for c in tr.columns]
    tr = tr.dropna(subset=[f"{iso}_is_TDI"]).reset_index(drop=True)

    aff = pd.read_csv(ROOT / "oof_st_v6.csv")
    aff.columns = [c.strip() for c in aff.columns]
    acol = f"{iso}_pIC50_direct_inhibition"
    aff = aff.set_index(C.ID_COL)[acol].dropna()
    tr = tr[tr[C.ID_COL].isin(aff.index)].reset_index(drop=True)
    y = (tr[f"{iso}_is_TDI"] == True).astype(int).to_numpy()        # noqa: E712
    smi = tr[C.SMILES_COL].tolist()
    X0, _ = features(smi)
    A = aff.loc[tr[C.ID_COL]].to_numpy(float).reshape(-1, 1)
    print(f"\n  {len(y)} rows with both a TDI label and a predicted {iso} pIC50, "
          f"{y.mean():.1%} positive")
    print(f"  predicted pIC50: mean {A.mean():.2f}, sd {A.std():.2f}")

    # the alert columns, taken from the block that already exists rather than re-derived
    # TA.calculate returns (matrix, ok_mask) -- the NAMES come from feature_names() and
    # carry a "tdi_" prefix. Getting that wrong silently produced an EMPTY product block
    # and an arm that scored exactly +0.000, which is what an untested arm looks like.
    al = np.nan_to_num(np.asarray(TA.calculate(smi)[0], float))
    names = list(TA.feature_names())
    assert al.shape[1] == len(names), f"{al.shape[1]} columns vs {len(names)} names"
    want = ("tdi_n_alerts_total", "tdi_n_mi_complex_precursors",
            "tdi_n_bioactivation_alerts", "tdi_amine_and_aryl",
            "tdi_reactive_ring_and_basic_n", "tdi_alert_density")
    keep = [j for j, n in enumerate(names) if n in want]
    assert len(keep) == len(want), f"matched {len(keep)} of {len(want)} alert columns"
    Az = (A - A.mean()) / (A.std() or 1.0)
    prod = Az * al[:, keep]                    # binding x bioactivation, standardised
    print(f"  conjunction block: {prod.shape[1]} products "
          f"({', '.join(names[j] for j in keep)})")

    arms = {"control": X0,
            "+affinity": np.hstack([X0, A]),
            "+affinity x alerts": np.hstack([X0, prod]),
            "+both": np.hstack([X0, A, prod])}

    import lightgbm as lgb  # noqa: PLC0415
    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=0)
    preds = {}
    for name, X in arms.items():
        o = np.zeros(len(y))
        for tr_i, te_i in skf.split(X, y):
            m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                   min_child_samples=20, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.7, reg_lambda=1.0, verbose=-1,
                                   n_jobs=-1)
            m.fit(X[tr_i], y[tr_i])
            o[te_i] = m.predict_proba(X[te_i])[:, 1]
        preds[name] = o

    base = preds["control"]
    db = dprime(y, base)
    print(f"\n  {'arm':20s} {'d’':>7s} {'gap':>8s} {'95% CI':>20s} {'P(worse)':>9s} "
          f"{'MCC':>8s}")
    rows = []
    for name, o in preds.items():
        d = dprime(y, o); m = mcc_at(y, o, CALL_RATE)
        if name == "control":
            print(f"  {name:20s} {d:7.3f} {'':8s} {'':20s} {'':9s} {m:8.4f}")
            rows.append(dict(arm=name, dprime=d, mcc=m, gap=0.0)); continue
        lo, hi, pw = boot(y, base, o)
        print(f"  {name:20s} {d:7.3f} {d-db:+8.3f} [{lo:+.3f}, {hi:+.3f}] {pw:9.0%} "
              f"{m:8.4f}")
        rows.append(dict(arm=name, dprime=d, mcc=m, gap=d - db, lo=lo, hi=hi, p_worse=pw))

    r = pd.DataFrame(rows).set_index("arm")
    r.to_csv(ROOT / "artifacts" / "tdi_3a4_affinity_summary.csv")
    g_aff = r.loc["+affinity", "gap"]
    g_prod = r.loc["+affinity x alerts", "gap"]
    print("\nVERDICT")
    print(f"  affinity alone      {g_aff:+.3f}")
    print(f"  the conjunction     {g_prod:+.3f}")
    if g_prod > g_aff + 0.02 and r.loc["+affinity x alerts", "p_worse"] < 0.10:
        print("  -> the PRODUCTS beat the raw affinity term: binding x bioactivation is")
        print("     doing work neither block could express alone.")
    elif max(g_aff, g_prod) < 0.02:
        print("  -> neither helps. Binding is already implicit in the 134 features, or")
        print("     the conjunction is not what limits CYP3A4 TDI.")
    else:
        print("  -> affinity helps but the conjunction adds nothing beyond it; ship the")
        print("     plain affinity term if anything, not the products.")
    print(f"\n  our blind CYP3A4 TDI is MCC 0.4601 at d' ~1.30 (rank 8); rank 1 is 0.4987")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
