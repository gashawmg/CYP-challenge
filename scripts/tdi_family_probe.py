"""
tdi_family_probe.py -- is the TDI MA-MCC limited by the RANKING or by the threshold?

Same-ruler comparison of member-family rank blends on the frozen TDI OOF.
Every arm is scored three ways on the SAME rows:
  AUC                 threshold-free, the pure ranking question
  MCC @ matched rate  all arms forced to the same call rate, so the operating
                      point cannot flatter one arm over another
  MCC @ best rate     each arm's own OOF-optimal rate (optimistic for ALL arms;
                      use for ordering only, never as a predicted board value)

Paired bootstrap over compounds for the delta against the incumbent ALL blend.

Run:  python scripts\tdi_family_probe.py
      python scripts\tdi_family_probe.py --pool tdiL --boot 4000
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import rankdata

SIM = ["knn_tan","svc_tan","knn_fcfp","svc_fcfp","knn_ap","svc_ap",
       "knn_maccs","knn_ecfpc","knn_k5","knn_k40","svc_c10"]
TREE = ["lgb_clf","xgb_clf","cat_clf","et_clf","lgb_alert","lgb_screen",
        "cat_screen","pooled","struct"]
TOP4 = {"CYP2D6": ["knn_k5","knn_ecfpc","svc_tan","svc_c10"],
        "CYP3A4": ["lgb_clf","pooled","lgb_screen","cat_clf"]}


def mcc(y, call):
    tp = np.sum(y & call); fp = np.sum(~y & call)
    fn = np.sum(y & ~call); tn = np.sum(~y & ~call)
    den = np.sqrt(float(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return 0.0 if den == 0 else (tp*tn - fp*fn)/den


def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y)-n1
    if n1 == 0 or n0 == 0: return np.nan
    return (r[y].sum() - n1*(n1+1)/2.0)/(n1*n0)


def mcc_at_rate(y, s, rate):
    k = max(1, int(round(rate*len(s))))
    thr = np.sort(s)[::-1][k-1]
    return mcc(y, s >= thr)


def mcc_best(y, s):
    order = np.argsort(-s); best = -9.0; bestr = np.nan
    ys = y[order]; n = len(ys)
    tp = np.cumsum(ys); fp = np.cumsum(~ys)
    P = ys.sum(); N = n - P
    for k in range(1, n):
        TP, FP = tp[k-1], fp[k-1]; FN, TN = P-TP, N-FP
        den = np.sqrt(float(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if den == 0: continue
        v = (TP*TN - FP*FN)/den
        if v > best: best, bestr = v, k/n
    return best, bestr


def blend(df, cols):
    """mean of per-member rank percentiles -- scale-free, the incumbent recipe"""
    R = np.column_stack([rankdata(df[c].values)/len(df) for c in cols])
    return R.mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="tdiNB")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    macro = {}
    for iso in ("CYP3A4", "CYP2D6"):
        p = os.path.join("artifacts", f"tdi_member_oof_{iso}_{a.pool}.csv")
        df = pd.read_csv(p)
        y = df["y"].astype(bool).values
        members = [c for c in df.columns if c not in ("Molecule_Name", "y")]
        arms = {"ALL (incumbent)": members,
                "SIM only": [c for c in SIM if c in members],
                "TREE only": [c for c in TREE if c in members],
                f"TOP4 {iso}": [c for c in TOP4[iso] if c in members]}
        S = {k: blend(df, v) for k, v in arms.items()}
        prev = y.mean()
        print(f"\n=== {iso}   n={len(df)}  prevalence={prev:.4f}  pool={a.pool}")
        print(f"{'arm':<18}{'AUC':>8}{'MCC@prev':>10}{'MCC@best':>10}{'rate*':>8}"
              f"{'dAUC':>9}{'P(worse)':>10}")
        base = S["ALL (incumbent)"]
        for k, s in S.items():
            A = auc(y, s); Mp = mcc_at_rate(y, s, prev); Mb, rb = mcc_best(y, s)
            if k == "ALL (incumbent)":
                print(f"{k:<18}{A:>8.4f}{Mp:>10.4f}{Mb:>10.4f}{rb:>8.3f}"
                      f"{'--':>9}{'--':>10}")
                macro.setdefault("ALL (incumbent)", []).append((A, Mp, Mb))
                continue
            d = np.empty(a.boot)
            for b in range(a.boot):
                idx = rng.integers(0, len(y), len(y))
                yy = y[idx]
                if yy.sum() == 0 or yy.sum() == len(yy): d[b] = 0.0; continue
                d[b] = auc(yy, s[idx]) - auc(yy, base[idx])
            pw = float((d <= 0).mean())
            print(f"{k:<18}{A:>8.4f}{Mp:>10.4f}{Mb:>10.4f}{rb:>8.3f}"
                  f"{A-auc(y,base):>+9.4f}{pw:>10.3f}")
            macro.setdefault(k, []).append((A, Mp, Mb))

    print("\n=== MACRO (mean of the two isoforms)")
    print(f"{'arm':<18}{'AUC':>8}{'MA-MCC@prev':>13}{'MA-MCC@best':>13}")
    for k, v in macro.items():
        v = np.array(v)
        print(f"{k:<18}{v[:,0].mean():>8.4f}{v[:,1].mean():>13.4f}{v[:,2].mean():>13.4f}")
    print("\nNOTE: MCC@best is fitted on the same rows it is scored on, so it is")
    print("      optimistic for EVERY arm. Board transfer is not implied by any")
    print("      column here -- OOF is scaffold-held-out and harder than the blind set.")


if __name__ == "__main__":
    main()
