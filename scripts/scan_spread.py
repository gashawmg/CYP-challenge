#!/usr/bin/env python
r"""Is the affine spread we ship optimal for ST-RAE? Answered on a pool that must first
prove it can reproduce the board.

    python scripts\scan_spread.py

THE MECHANISM BEING TESTED. ST-RAE charges `clip(p-hi,0) + clip(lo-p,0)`: a prediction
inside its interval costs NOTHING. That dead zone means pulling predictions toward the mean
parks them inside intervals, a reward squared error never offers. We ship k = rho, the
MSE-optimal shrinkage, on every endpoint. If the dead zone matters, k* < rho.

WHY THIS SCRIPT EXISTS RATHER THAN A SUBMISSION. On the label-matched pool the scan says
k* = 0.55 against the 0.766 we ship on CYP3A4, worth -22.6%. That pool is also, per Section 16,
far harder than the blind set: median out-of-fold nearest-neighbour similarity 0.438
against the blind set's 0.587, with a third of its compounds below 0.4 against one in a
thousand. A pool that is too hard is EXACTLY what manufactures a preference for
over-shrinkage, so the scan's own headline number is the thing most likely to be an
artefact of it.

THE GATE. Before any k* from this pool is allowed to influence a submission, the pool has
to reproduce a number we already know: our ST-RAE on the board at the spread we actually
shipped, per endpoint. Measured off artifacts/submission_splice4.csv:

    endpoint   shipped mean   shipped k   board ST-RAE
    CYP1A2        4.301         0.880        0.5125
    CYP2C9        4.849         0.764        0.3978
    CYP2D6        3.949         0.418        0.5957
    CYP3A4        4.704         0.766        0.4634    (recovered exactly by the raw3a4 probe)

An endpoint whose pool cannot reproduce its own board score to within roughly 10% has no
business recommending a placement change, however tight its bootstrap looks.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from scripts.blind_validator import BLIND, eval_pool, match_weights, strae  # noqa: E402
from scripts.build_matched_pool import match_weights_2d  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# read off the submitted file, not assumed
SHIPPED = {"CYP1A2": (4.301, 0.880), "CYP2C9": (4.849, 0.764),
           "CYP2D6": (3.949, 0.418), "CYP3A4": (4.704, 0.766)}
BOARD = {"CYP1A2": 0.5125, "CYP2C9": 0.3978, "CYP2D6": 0.5957, "CYP3A4": 0.4634}
MEMBERS = (("v6", "oof_st_v6.csv"), ("mh", "oof_st_mh.csv"),
           ("cp", "artifacts/oof_chempropv3_ensA.csv"), ("um", "oof_unimol_cyp_mt.csv"))
KS = np.arange(0.20, 1.45, 0.05)


def main() -> int:
    b4 = {}
    for n, f in MEMBERS:
        d = pd.read_csv(ROOT / f); d.columns = [c.strip() for c in d.columns]
        b4[n] = d.set_index(C.ID_COL)
    nn = pd.read_csv(ROOT / "artifacts" / "nnsim.csv")
    sim = dict(zip(nn[nn.split == "train"][C.ID_COL], nn[nn.split == "train"].nnsim))
    blind_nn = nn[nn.split == "blind"].nnsim.to_numpy()
    pool = eval_pool()
    rng = np.random.default_rng(0)
    rows = []

    for iso in C.ISOFORMS:
        ep = f"{iso}_pIC50_direct_inhibition"
        s = pool[(pool.isoform == iso) & (~pool.censored)]
        ids = s[C.ID_COL]
        for d in b4.values():
            ids = ids[ids.isin(d.index)]
        ids = ids[ids.isin(sim)]
        s = s[s[C.ID_COL].isin(ids)]; ids = s[C.ID_COL].to_numpy()
        y = s["y_eval"].to_numpy(float)
        lo, hi = s["y_low"].to_numpy(float), s["y_high"].to_numpy(float)
        z = np.nan_to_num(np.mean([((lambda v: (v - np.nanmean(v)) / (np.nanstd(v) or 1))
                                    (d.loc[ids, ep].to_numpy(float))) for d in b4.values()],
                                  axis=0))
        z = (z - z.mean()) / z.std()
        nnv = np.array([sim[m] for m in ids])
        mu_ship, k_ship = SHIPPED[iso]

        w1, _ = match_weights(y, np.zeros(len(y), bool), *BLIND[iso])
        w2, ess = match_weights_2d(y, nnv, *BLIND[iso], blind_nn)

        def curve(w, n_draws=60):
            draws = [rng.choice(len(y), size=len(y), replace=True, p=w) for _ in range(n_draws)]
            # place exactly as the submission does: its own mean, k x the blind sd
            out = [float(np.mean([strae((z * BLIND[iso][1] * k + mu_ship)[d],
                                        lo[d], hi[d], y[d]) for d in draws])) for k in KS]
            return np.asarray(out), draws

        c1, _ = curve(w1)
        c2, draws2 = curve(w2)
        at_ship1 = c1[int(np.argmin(np.abs(KS - k_ship)))]
        at_ship2 = c2[int(np.argmin(np.abs(KS - k_ship)))]
        k1, k2 = KS[int(np.argmin(c1))], KS[int(np.argmin(c2))]
        # bootstrap the argmin under the matched weights
        opts = []
        for _ in range(60):
            d = rng.choice(len(y), size=len(y), replace=True, p=w2)
            cc = [strae((z * BLIND[iso][1] * k + mu_ship)[d], lo[d], hi[d], y[d]) for k in KS]
            opts.append(KS[int(np.argmin(cc))])
        o = np.asarray(opts)
        err1 = at_ship1 / BOARD[iso] - 1
        err2 = at_ship2 / BOARD[iso] - 1
        gate = abs(err2) <= 0.10
        rows.append(dict(isoform=iso, k_shipped=k_ship, board=BOARD[iso],
                         pool_at_ship=at_ship2, repro_err=err2, ess=ess,
                         k_label_matched=k1, k_difficulty_matched=k2,
                         k_lo=np.quantile(o, .1), k_hi=np.quantile(o, .9),
                         gain=c2[int(np.argmin(c2))] / at_ship2 - 1, gate=gate))
        print(f"\n=== {iso}   shipped k {k_ship:.3f}, board ST-RAE {BOARD[iso]:.4f}")
        print(f"  REPRODUCTION at the shipped k: label-matched {at_ship1:.4f} "
              f"({100*err1:+.0f}%)   difficulty-matched {at_ship2:.4f} ({100*err2:+.0f}%)"
              f"   effective n {ess:.0f}")
        print(f"  k*: label-matched {k1:.2f}   difficulty-matched {k2:.2f} "
              f"[{np.quantile(o,.1):.2f}, {np.quantile(o,.9):.2f}]"
              f"   gain at k* {100*(c2[int(np.argmin(c2))]/at_ship2-1):+.1f}%")
        print(f"  GATE: {'PASS -- this pool can price this endpoint' if gate else 'FAIL -- pool cannot reproduce the board; k* here means nothing'}")

    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "artifacts" / "spread_scan.csv", index=False)
    ok = r[r.gate]
    print("\n" + "=" * 74)
    print(f"  endpoints whose pool passes the reproduction gate: "
          f"{list(ok.isoform) or 'NONE'}")
    if len(ok):
        est = sum(BOARD[i] for i in C.ISOFORMS)
        for _, x in ok.iterrows():
            est += BOARD[x.isoform] * x.gain
        print(f"  macro if every passing endpoint moves to its k* and it transfers fully: "
              f"{est/4:.4f} against 0.4924 held")
    print("  wrote artifacts/spread_scan.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
