#!/usr/bin/env python
r"""Per-endpoint blend search for pIC50, with the selection-bias control the TDI work added.

    python scripts\pic50_blend_search.py
    python scripts\pic50_blend_search.py --write

WHY NOW. The same method -- greedy member selection, validated by re-running the selection
inside split-halves -- produced the only two significant gains this project has had on the
TDI track. `scripts/rank_search.py` did a weight search on pIC50 months ago, but nothing has
ever re-run the SELECTION out of sample, so we have never known how much of a searched
blend's local gain was real. That distinction is the whole reason the TDI blend transferred.

WHAT IT SCORES ON. `blind_validator.eval_pool()` restricted to uncensored rows and
resampled to the blind histogram -- the only local yardstick that has ever predicted the
board (affine absorption 0.807 local against the 0.806 we score). Our pool's below-floor
mass is 89-96% left-censored screen non-hits and the blind set is 0% censored; fitting
against those rows once bought a 13% gain that did not exist.

THE INSTRUMENT CHECK RUNS FIRST, AND IT IS NOT CEREMONY. Section 14 found our local CYP2D6 TDI
yardstick reads 4.37x low while CYP3A4's is exact, which silently reversed a decision. The
macro absorption match says nothing about whether any INDIVIDUAL endpoint is measured
honestly, so this prints local rho against the blind rho the board reported for each
endpoint of submission_splice4 before it searches anything.

SELECTION IS ON RHO, NOT ST-RAE. Placement is affine at the blind anchors, so
ST-RAE = absorption x sqrt(1 - rho^2) with absorption near-fixed per endpoint; rho is the
part a blend can move and it is invariant under any monotone map, which makes it stable
under the resampling. ST-RAE is reported for the chosen blend so the number stays
comparable to the board.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cypkit import config as C  # noqa: E402
from scripts.blind_validator import BLIND, eval_pool, match_weights, strae  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MEMBERS = {
    "v6":      ("oof_st_v6.csv",      "pred_st_v6.csv"),
    "v7":      ("oof_st_v7.csv",      "pred_st_v7.csv"),
    "v5_rest": ("oof_st_v5_rest.csv", "pred_st_v5_rest.csv"),
    "v5_2d6":  ("oof_st_v5_2d6.csv",  "pred_st_v5_2d6.csv"),
    # REPOINTED 2026-09-28. oof_st_mh.csv was DESTROYED by a smoke run that took the
    # default --oof_output and wrote its 263-row fold-0 stub over the real file. The
    # original is unrecoverable, so `mh` now points at oof_st_mh_ctl.csv: a same-day
    # re-run of train_cyp_mh.py with --descriptor_spec none, i.e. the same configuration,
    # same folds, same seeds.
    #
    # READ THIS BEFORE TRUSTING THE BASELINE. `mh` is a member of BLEND4Z, the incumbent
    # this whole search is measured against, and blend4z as SCORED ON THE BOARD used the
    # original file. The re-run is not bit-identical to it: on the pooled mh metric the
    # intact Sept-8 sibling mh2 scores rho 0.6079 and this re-run scores 0.5974, so the
    # incumbent is now an APPROXIMATION of what shipped, very slightly weaker. A gain
    # measured against it is therefore very slightly overstated -- about 0.01 of rho on
    # one of four members. Do not quietly forget this when a candidate lands near the bar.
    "mh":      ("oof_st_mh_ctl.csv",  "pred_st_mh_ctl.csv"),
    "mh2":     ("oof_st_mh2.csv",     "pred_st_mh2.csv"),
    "unimol":  ("oof_unimol_cyp_mt.csv", "predictions_unimol_cyp_mt_test.csv"),
    "cp":      ("artifacts/oof_chempropv3_ensA.csv", "predictions_cyp_st_v3.csv"),
    # NEW, 2026-09-23: Uni-Mol SINGLE-TASK, one model per endpoint. The first genuinely
    # new member since blend4z, which is what Section 15 said this track needs. CYP3A4 is absent
    # from it -- that Kaggle run died in fold 0 -- so its column is all-NaN and the guard
    # below drops it for that endpoint only.
    "unimol_st": ("artifacts/oof_unimol_cyp_st.csv",
                  "artifacts/predictions_unimol_cyp_st_test.csv"),
    # NEW, 2026-09-27: per-target chemprop retrained with the fold-aware pseudo-label map
    # (Section 34) and the Section 36 lean descriptor block. Section 40 scored it against
    # the shipped `cp` column at +0.0332 CYP2C9 and +0.0281 CYP3A4, -0.0210 CYP1A2 and
    # -0.0309 CYP2D6 -- better on two, worse on two, which is a blend question and not a
    # splice to pick by eye. Section 32 closed RE-WEIGHTING the members we had; it said
    # nothing about a new one, which is why this search is worth re-running.
    "cp_lean": ("oof_chemprop_st_v3_lean.csv", "predictions_cyp_st_v3_lean.csv"),
    # NEW, 2026-09-28: the multi-head external-data model WITH target descriptors on the
    # shared trunk (Section 47). train_cyp_mh.py had no descriptors at all until now,
    # while every descriptor result in Sections 36-43 was measured on the single-task
    # path -- the weaker architecture. Against a same-day no-descriptor control on the
    # same folds and seeds, this improved EVERY endpoint on EVERY metric: macro rho
    # 0.5974 -> 0.6173, macro ST-RAE 0.706 -> 0.675. That makes it the best single member
    # we have on that pool, ahead of v6 (0.6128) and mh2 (0.6079).
    #
    # ITS RAW PREDICTIONS ARE NOT SUBMITTABLE and it is not a candidate on its own: it
    # sits +1.84 above the anchor-solved blind mean on CYP2D6 with a quarter of the
    # needed spread. It is a RANKER. The blend is where its rho is used and the level
    # comes from elsewhere.
    "mh_desc": ("oof_st_mh_desc.csv", "pred_st_mh_desc.csv"),
    # NEW, 2026-09-28: the same multi-head trunk with the 8 conformer shape columns added
    # to the shared block (35 columns = 27 lean-common + 8 shape3d).
    #
    # IT IS REGISTERED BECAUSE OF DECORRELATION, NOT BECAUSE IT IS BETTER. Against
    # mh_desc, pooled: macro rho 0.6173 -> 0.6209 (+0.0036), but macro ST-RAE 0.675 ->
    # 0.678 and macro R2 0.345 -> 0.333, both slightly WORSE. The cost is concentrated
    # entirely on CYP1A2 -- R2 0.216 -> 0.167, ST-RAE 0.792 -> 0.818 -- which is the
    # endpoint 3D was adopted FOR on the single-task path in Section 43. The gain is on
    # CYP2C9 (+0.009 rho) and CYP2D6 (+0.004).
    #
    # So 3D on the shared trunk is roughly neutral on its own and does NOT replace
    # mh_desc. It is here only so the search can decide whether a differently-wrong
    # member adds anything a correlated better one cannot. If it takes no weight, drop it
    # and stop paying the conformer-embedding cost.
    "mh_d3d":  ("oof_st_mh_desc3d.csv", "pred_st_mh_desc3d.csv"),
    # NEW, 2026-09-29: three seeds of the mh_desc recipe, averaged. Pure variance
    # reduction -- identical features, identical folds, only the seed differs.
    #
    # THE REASON IT WORKS IS MEASURED, NOT ASSUMED. Mean pairwise correlation between two
    # runs of this model: CYP1A2 0.797, CYP2D6 0.796, CYP2C9 0.904, CYP3A4 0.932. The
    # ensemble's gain on the blind-like holdout tracks (1 - r) across the four endpoints
    # with a correlation of 0.986: +0.0376, +0.0414, +0.0264, +0.0218. That is variance
    # reduction behaving exactly as the arithmetic says it should.
    #
    # IT ALSO REVISES THE NOISE FLOOR DOWNWARD FOR EVERY EARLIER A/B. Section 43 measured
    # r = 0.927 on the SINGLE-TASK model and that figure was reused for the multi-head
    # one. On CYP1A2 and CYP2D6 the true figure is ~0.80 -- roughly 20% of prediction
    # variance is seed noise, nearly three times what was assumed. Any single-pair
    # comparison on this architecture smaller than that floor is undecided, which
    # includes `shared` vs `shared3d` (+0.0036 macro rho).
    # 2026-09-29, second pass: the file is now FIVE seeds (42/1337/2024/7/99), so the key
    # is renamed to match -- a member called "ens3" pointing at a 5-seed average is the
    # kind of quiet lie that costs an afternoon later. The 3-seed version is preserved as
    # oof_mhens3_k3.csv with a .seeds.json sidecar if it is ever needed.
    #
    # On ALL fitted rows the 5-seed average beats the 3-seed one on every endpoint:
    # CYP1A2 0.4851 vs 0.4750, CYP2C9 0.5858 vs 0.5731, CYP2D6 0.4785 vs 0.4744,
    # CYP3A4 0.8263 vs 0.8234. On the 61-248 compound blind-like holdout it looked 0.045
    # WORSE on CYP2D6, which was that panel's n=72 talking, not the model.
    "mhens5":  ("oof_mhens3.csv", "pred_mhens3.csv"),
}
# THE BASELINE MUST BE WHAT WE ACTUALLY SHIP, NOT THE BEST SINGLE MEMBER. blend4z is an
# equal z-average of these four and supplies the CYP1A2 and CYP3A4 columns of splice4;
# comparing a searched blend against a solo member would credit it with the gain blend4z
# already earned. scripts/build_blend4.py is the definition.
BLEND4Z = ("v6", "mh", "cp", "unimol")
# what submission_splice4 (0.4924, rank 19) actually scored, read off the board
BLIND_RHO = {"CYP1A2": 0.8064, "CYP2C9": 0.8454, "CYP2D6": 0.5190, "CYP3A4": 0.8327}
# UPDATED 2026-09-29: the incumbent is now `cand1` (order 16), not `splice4`. Leaving the
# old row here would measure every future candidate against a submission we have already
# beaten, and quietly re-promise gains that are banked. splice4 for reference:
#     CYP1A2 0.5125  CYP2C9 0.3978  CYP2D6 0.5957  CYP3A4 0.4634   macro 0.4924
# All half-set values -- the LIVE board is the half set, proven by cand1 carrying CYP2D6
# over unchanged and scoring 0.5957 exactly (its full-set value is 0.5908).
# UPDATED 2026-10-14 (Section 74): the incumbent is `d6probe` (order 17, macro 0.4737),
# not cand1 (0.4820). Only CYP2D6 differs -- d6probe is cand1 with a -0.30 shift on that
# one endpoint -- so the stale row understated our own CYP2D6 by 0.0336 and every
# candidate measured against it was being graded on a curve. FOURTH stale-baseline
# instance in this project (Sections 51, 67, 68 were the others). Check this row against
# artifacts/submissions.csv before trusting any number this script prints.
BLIND_ST = {"CYP1A2": 0.5147, "CYP2C9": 0.3645, "CYP2D6": 0.5621, "CYP3A4": 0.4532}
HELD_NAME = "d6probe"
# Endpoints whose BLENDED column is already what the incumbent ships. The search compares
# against blend4z's pool performance, so for these it re-derives a gain we have already
# taken. Without this note the summary reads as new headroom.
SHIPPED_BLEND = {"CYP2C9", "CYP3A4"}

# HOW MUCH OF A POOL-LEVEL ST-RAE CHANGE ACTUALLY REACHES THE BOARD.
#
# This was 0.15 from 2026-09-13 to 2026-09-29, taken from an ensemble-ranking transfer
# measured on a different question, and it was badly wrong. `cand1` (order 16) is the
# first submission built FROM this search, so it is the first direct measurement:
#
#     endpoint   pool change   predicted at 100%   observed    transfer
#     CYP2C9      -6.13%          -0.0244          -0.0333      136%
#     CYP3A4      -4.34%          -0.0201          -0.0102       51%
#
# CYP3A4 is the clean one: its final spread (0.9571) was within 0.002 of the column it
# replaced (0.9586), so only the RANKING changed. CYP2C9 changed ranking and spread
# together, so its 136% is not attributable to ranking alone.
#
# 0.50 is therefore the conservative reading -- the clean single-variable case -- and it is
# still 3.3x the old value. At 0.15 this search forecast cand1's macro move at -0.0016 and
# said "do not submit"; the actual was -0.0104 and it moved us 40 -> 39. A gate that tells
# you not to ship a submission that works is worse than no gate.
#
# REVISIT THIS after the next scored submission. It rests on two endpoints from one upload.
TRANSFER = 0.50


def load(p: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / p)
    d.columns = [c.strip() for c in d.columns]
    return d.set_index(C.ID_COL)


def col_for(d: pd.DataFrame, iso: str) -> str | None:
    hits = [c for c in d.columns if c.startswith(iso) and "pIC50" in c]
    return hits[0] if hits else next((c for c in d.columns if c.startswith(iso)), None)


def greedy(y, Z, names, max_steps=6):
    """Forward selection WITH replacement; a repeat is how a member gets extra weight."""
    chosen, cur, best_rho = [], np.zeros(len(y)), -np.inf
    for _ in range(max_steps):
        cand = [((cur * len(chosen) + Z[:, j]) / (len(chosen) + 1), j) for j in range(len(names))]
        scored = [(float(pd.Series(c).corr(pd.Series(y), method="spearman")), c, j)
                  for c, j in cand]
        r, c, j = max(scored, key=lambda t: t[0])
        if chosen and r <= best_rho + 1e-4:
            break
        chosen.append(names[j]); cur, best_rho = c, r
    return chosen, cur, best_rho


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the blended blind predictions")
    ap.add_argument("--splits", type=int, default=40)
    a = ap.parse_args()

    # A MISSING MEMBER FILE MUST NOT BE A TRACEBACK, AND MUST NOT BE SILENTLY SKIPPED
    # EITHER IF IT IS PART OF THE BASELINE. blend4z is v6+mh+cp+unimol; dropping one of
    # those would quietly compare the search against a THREE-member incumbent and credit
    # the blend with the missing member's contribution -- the Section 15 trap in a new costume.
    # A TRUNCATED MEMBER IS WORSE THAN A MISSING ONE, because `exists()` is happy with it.
    # That is exactly how a 263-row smoke stub sat in `mh` -- a BLEND4Z member -- looking
    # like a real file. Every legitimate member here covers 4,376-4,906 compounds, so
    # anything under half the widest member is a stub, not a design choice. Per-endpoint
    # coverage is deliberately NOT the test: unimol_st legitimately carries only ~1,400
    # fitted rows per endpoint and v5_2d6 only one endpoint at all.
    rows = {}
    for n, (o, _) in MEMBERS.items():
        if not (ROOT / o).exists():
            continue
        try:
            with open(ROOT / o, encoding="utf-8") as fh:
                rows[n] = sum(1 for _ in fh) - 1
        except Exception:                                            # noqa: BLE001
            rows[n] = 0
    nmax = max(rows.values()) if rows else 0
    stub = [n for n, r in rows.items() if nmax and r < 0.5 * nmax]
    for n in stub:
        print(f"  TRUNCATED member '{n}': {MEMBERS[n][0]} has {rows[n]} rows against a "
              f"widest member of {nmax}. Treating it as missing.")

    missing = [n for n, (o, _) in MEMBERS.items()
               if not (ROOT / o).exists()] + stub
    if missing:
        print("  missing member files:")
        for n in missing:
            print(f"    {n:10s} {MEMBERS[n][0]}")
    fatal = [n for n in missing if n in BLEND4Z]
    if fatal:
        raise SystemExit(
            f"\n  {fatal} are part of blend4z, the incumbent this search is measured "
            f"against.\n  Without them the baseline is not blend4z and every gap would be "
            f"overstated.\n  Restore those files before running.\n  (A member listed "
            f"above as TRUNCATED is present on disk but too short to be the real "
            f"thing -- regenerate it rather than deleting the check.)")
    for n in missing:
        print(f"  skipping optional member '{n}'")
        MEMBERS.pop(n)
    oof = {n: load(o) for n, (o, _) in MEMBERS.items()}
    pool = eval_pool()
    rng = np.random.default_rng(0)
    out_pred, summary = {}, []

    for iso in C.ISOFORMS:
        # A member whose column for this endpoint is entirely (or almost entirely) missing
        # standardises to a constant after nan_to_num -- a column of zeros that the greedy
        # search can still "select" while contributing nothing. Drop it by name instead,
        # so the summary says which members actually competed.
        names = []
        for n in MEMBERS:
            c = col_for(oof[n], iso)
            if c is None:
                continue
            v = pd.to_numeric(oof[n][c], errors="coerce")
            if v.notna().sum() < 100 or float(v.std(skipna=True) or 0.0) < 1e-9:
                print(f"  [{iso}] dropping member '{n}': "
                      f"{int(v.notna().sum())} finite values, sd {v.std(skipna=True):.3g}")
                continue
            names.append(n)
        s = pool[(pool["isoform"] == iso) & (~pool["censored"])]
        ids = s[C.ID_COL]
        for n in names:
            ids = ids[ids.isin(oof[n].index)]
        s = s[s[C.ID_COL].isin(ids)]
        y = s["y_eval"].to_numpy(float)
        lo, hi = s["y_low"].to_numpy(float), s["y_high"].to_numpy(float)
        ids = s[C.ID_COL].to_numpy()
        # standardise: blending raw pIC50 columns with different centres is a weighted shift
        Z = np.column_stack([
            (lambda v: (v - np.nanmean(v)) / (np.nanstd(v) or 1.0))(
                oof[n].loc[ids, col_for(oof[n], iso)].to_numpy(float)) for n in names])
        Z = np.nan_to_num(Z)

        print(f"\n=== {iso}  {len(y)} uncensored pool rows, {len(names)} members")
        solo = {n: float(pd.Series(Z[:, j]).corr(pd.Series(y), method="spearman"))
                for j, n in enumerate(names)}
        print("  solo rho: " + ", ".join(f"{n} {v:.4f}" for n, v in solo.items()))
        # the incumbent: the blend4z recipe, rebuilt on exactly these rows
        b4 = [n for n in BLEND4Z if n in names]
        base_vec = Z[:, [names.index(n) for n in b4]].mean(axis=1)
        rho_base = float(pd.Series(base_vec).corr(pd.Series(y), method="spearman"))
        best_solo = max(solo, key=solo.get)
        print(f"  incumbent blend4z ({'+'.join(b4)}): local rho {rho_base:.4f}   "
              f"best solo {best_solo} {solo[best_solo]:.4f}")
        print(f"  INSTRUMENT: incumbent local rho {rho_base:.4f} vs the "
              f"blind {BLIND_RHO[iso]:.4f} the board reported -> "
              f"ratio {BLIND_RHO[iso] / rho_base:.2f}x")

        chosen, blend, rho_in = greedy(y, Z, names)
        gains = []
        for _ in range(a.splits):
            m = rng.random(len(y)) < 0.5
            c2, _, _ = greedy(y[m], Z[m], names)
            idx = [names.index(x) for x in c2]
            held = Z[~m][:, idx].mean(axis=1)
            base = Z[~m][:, [names.index(n) for n in b4]].mean(axis=1)   # vs what we ship
            gains.append(float(pd.Series(held).corr(pd.Series(y[~m]), method="spearman"))
                         - float(pd.Series(base).corr(pd.Series(y[~m]), method="spearman")))
        g = np.asarray(gains)

        ww, _ = match_weights(y, np.zeros(len(y), bool), *BLIND[iso])
        draws = [rng.choice(len(y), size=len(y), replace=True, p=ww) for _ in range(40)]
        def st_of(v):
            r = float(pd.Series(v).corr(pd.Series(y), method="spearman"))
            z = (v - v.mean()) / (v.std() or 1.0)
            p = z * BLIND[iso][1] * r + BLIND[iso][0]
            return float(np.mean([strae(p[d], lo[d], hi[d], y[d]) for d in draws])), r

        st_b, _ = st_of(blend)
        st_s, _ = st_of(base_vec)
        print(f"  greedy pick: {chosen}")
        print(f"  rho {rho_base:.4f} -> {rho_in:.4f} in sample; "
              f"OUT-OF-SELECTION {g.mean():+.4f} "
              f"[{np.quantile(g, .1):+.4f}, {np.quantile(g, .9):+.4f}]  "
              f"P(worse) {(g <= 0).mean():.0%}")
        print(f"  pool ST-RAE {st_s:.4f} -> {st_b:.4f}")
        summary.append(dict(isoform=iso, chosen="+".join(chosen),
                            rho_base=rho_base, rho_blend=rho_in,
                            oos_gain=g.mean(), p_worse=(g <= 0).mean(),
                            st_base=st_s, st_blend=st_b,
                            instrument_ratio=BLIND_RHO[iso] / rho_base))

        if a.write:
            # rebuild the SAME blend on the blind set, standardising each member there too
            preds = {n: load(MEMBERS[n][1]) for n in set(chosen)}
            base_ids = preds[chosen[0]].index
            cols = []
            for n in chosen:
                v = preds[n].loc[base_ids, col_for(preds[n], iso)].to_numpy(float)
                cols.append((v - np.nanmean(v)) / (np.nanstd(v) or 1.0))
            z = np.nan_to_num(np.mean(cols, axis=0))
            # place affinely at the blind anchors, shrunk by the achieved rho -- the same
            # placement the scored submissions use
            out_pred[iso] = pd.Series(z * BLIND[iso][1] * rho_in + BLIND[iso][0],
                                      index=base_ids)

    r = pd.DataFrame(summary)
    print("\nSUMMARY")
    print(r.to_string(index=False))
    keep = r[(r.p_worse <= 0.10) & (r.oos_gain > 0)]
    print(f"\n  endpoints where the blend survives selection: "
          f"{list(keep.isoform) or 'none'}")
    est = sum(BLIND_ST[i] for i in C.ISOFORMS) / 4
    print(f"  (macro of the scored per-endpoint ledger: {est:.4f})")
    for _, x in keep.iterrows():
        # scale the blind ST-RAE by the pool's ST-RAE ratio, then discount: the measured
        # ensemble transfer on this track is 14%, and CYP3A4 MCC transfer was 19%
        est += (BLIND_ST[x.isoform] * (x.st_blend / x.st_base - 1) * TRANSFER) / 4
    print(f"  macro at {TRANSFER:.0%} transfer of the pool ST-RAE change: {est:.4f} "
          f"against {sum(BLIND_ST.values())/4:.4f} held ({HELD_NAME})")
    if SHIPPED_BLEND:
        print(f"\n  ALREADY ON THE BOARD: the blend for {sorted(SHIPPED_BLEND)} is what "
              f"{HELD_NAME} ships.\n  Gain projected for those endpoints is BANKED, not "
              f"available again -- read it as a\n  consistency check, and look to the "
              f"other endpoints for anything new.")
    r.to_csv(ROOT / "artifacts" / "pic50_blend_search.csv", index=False)
    if a.write and out_pred:
        w = pd.DataFrame({f"{i}_pIC50_direct_inhibition": v for i, v in out_pred.items()})
        w.index.name = C.ID_COL
        w.reset_index().to_csv(ROOT / "pred_st_blendsearch.csv", index=False)
        print(f"  wrote pred_st_blendsearch.csv ({len(w)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
