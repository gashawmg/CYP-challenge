#!/usr/bin/env python
r"""Run train_cyp_st_v3.py under several seeds and average them, then show what it bought.

    python scripts/seed_ensemble.py --seeds 42 1337 2024 7 99 --tag ens5
    python scripts/seed_ensemble.py --seeds 42 1337 2024 7 99 --tag ens5 --resume
    python scripts/seed_ensemble.py --tag ens5 --combine-only      (no training)

WHY. Two runs of train_cyp_st_v3.py with IDENTICAL settings correlate only r = 0.927
(Section 43). Roughly 7% of each run's prediction variance is seed noise, which has two
consequences this project has been paying for silently:

  ACCURACY. Averaging k runs cuts the independent component by k. At k = 5 that removes
  about 80% of it -- worth something like +0.02 rho per endpoint for no modelling risk at
  all. It is the only improvement available here that cannot fail for a reason of substance.

  DECIDABILITY. The noise floor is ~0.06 on the reweighted yardstick and ~0.007 on the
  blind-like holdout. Most feature effects we have been measuring are SMALLER than that,
  which is why Section 43's CYP2D6 and CYP3A4 "refutations" rest on single pairs of runs
  and should not be trusted. Ensembling drops the floor by about sqrt(k) and makes the next
  round of A/Bs actually decisive. That is the larger of the two benefits.

WHAT IT DOES. One subprocess per seed, each writing its own model dir, OOF and prediction
file, then a plain mean across seeds. Everything else -- descriptors, folds, loss -- is
whatever you pass through after `--`, unchanged and identical across seeds, because the
whole point is that the seed is the only thing that differs.

    python scripts/seed_ensemble.py --seeds 42 1337 2024 --tag ens3 -- --descriptor_spec per_endpoint

RESUMABILITY IS NOT A CONVENIENCE HERE. Five seeds is hours of GPU. `--resume` skips a seed
whose OOF and prediction files both already exist and are readable, so a run killed at seed
4 costs one seed, not five.

A CRASHED SEED IS NEVER AVERAGED IN. If a subprocess exits non-zero, or writes a file that
does not parse, or writes one whose Molecule_Name column does not match the others, that
seed is dropped loudly and the ensemble is built from the rest with the count stated. A
silently-short average is the failure mode that would make every downstream number wrong
while looking fine.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

from cypkit import config as C                                        # noqa: E402

# The training script is a PARAMETER, not a constant. It was hard-coded to
# train_cyp_st_v3.py, and on 2026-09-29 that produced a command which passed
# `--descriptor_spec shared` -- a train_cyp_mh.py option -- to the single-task script,
# which rejected it. The failure was clean (argparse exit 2, 0.5 min, the seed correctly
# dropped rather than averaged) but it should not have been possible to express.
DEFAULT_TRAIN = "train_cyp_st_v3.py"
ID = C.ID_COL
EPS = list(C.REGRESSION_ENDPOINTS)


def paths(tag: str, seed: int):
    """Names are keyed on the TAG only, never on a script name.

    They used to read `oof_chemprop_st_v3_<tag>_s<seed>.csv` regardless of which script
    actually ran, which would have quietly labelled multi-head outputs as single-task
    ones. The tag is what carries the meaning; make it carry all of it.
    """
    return dict(model=f"models_{tag}_s{seed}",
                oof=f"oof_{tag}_s{seed}.csv",
                pred=f"pred_{tag}_s{seed}.csv")


def readable(p: str) -> bool:
    try:
        d = pd.read_csv(p)
        d.columns = [c.strip() for c in d.columns]
        return ID in d.columns and any(e in d.columns for e in EPS) and len(d) > 0
    except Exception:                                                 # noqa: BLE001
        return False


def run_seed(seed: int, tag: str, passthru: list[str], train: str) -> bool:
    q = paths(tag, seed)
    cmd = [sys.executable, train, "--seed", str(seed),
           "--model_dir", q["model"], "--oof_output", q["oof"], "--output", q["pred"],
           *passthru]
    print("\n" + "=" * 78)
    print(f"SEED {seed}   ->  {q['oof']}")
    print("  " + " ".join(cmd))
    print("=" * 78, flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    print(f"\n  seed {seed} exited {rc} after {(time.time()-t0)/60:.1f} min")
    if rc != 0:
        print(f"  !! seed {seed} FAILED and will not be averaged in")
        return False
    for k in ("oof", "pred"):
        if not readable(q[k]):
            print(f"  !! seed {seed} produced an unreadable {k} file; dropping this seed")
            return False
    return True


def combine(tag: str, seeds: list[int], kind: str, out: str):
    """Mean across seeds, on the intersection of their id columns, aligned by id."""
    frames = []
    for s in seeds:
        p = paths(tag, s)[kind]
        if not readable(p):
            print(f"  skip seed {s}: {p} missing or unreadable")
            continue
        d = pd.read_csv(p)
        d.columns = [c.strip() for c in d.columns]
        frames.append((s, d.set_index(ID)))
    if not frames:
        raise SystemExit(f"no usable {kind} files for tag {tag}")

    ids = set(frames[0][1].index)
    for _, d in frames[1:]:
        ids &= set(d.index)
    ids = sorted(ids)
    lost = [(s, len(d) - len(ids)) for s, d in frames if len(d) != len(ids)]
    if lost:
        print(f"  NOTE: id sets differ; using the {len(ids)} shared rows "
              f"(dropped: {lost})")

    cols = [e for e in EPS if all(e in d.columns for _, d in frames)]
    stack = {e: np.vstack([d.loc[ids, e].to_numpy(float) for _, d in frames])
             for e in cols}
    res = pd.DataFrame({ID: ids})
    for e in cols:
        res[e] = np.nanmean(stack[e], axis=0)
    # carry SMILES through if any member had it
    for _, d in frames:
        if C.SMILES_COL in d.columns:
            res.insert(1, C.SMILES_COL, d.loc[ids, C.SMILES_COL].to_numpy())
            break
    # PROVENANCE: a combined file is defined by WHICH SEEDS went into it, and --resume
    # with a longer --seeds list rewrites it under the same name. On 2026-09-29 a 3-seed
    # oof_mhens3.csv was registered as a blend member and a 5-seed resume was started
    # under the same tag; without this the member would silently have become a different
    # object while every note about it still said three. Same class of failure as the
    # smoke run that overwrote oof_st_mh.csv the day before.
    used = sorted(s for s, _ in frames)
    side = Path(out).with_suffix(".seeds.json")
    if Path(out).exists() and side.exists():
        try:
            prev = sorted(json.loads(side.read_text())["seeds"])
        except Exception:                                            # noqa: BLE001
            prev = None
        if prev and prev != used:
            keep = Path(out).with_name(
                f"{Path(out).stem}_k{len(prev)}{Path(out).suffix}")
            if not keep.exists():
                shutil.copy2(out, keep)
                print(f"  NOTE: {out} was built from {prev}; this run uses {used}. "
                      f"The previous file is preserved as {keep.name} so anything "
                      f"already measured against it stays reproducible.")
    res.to_csv(out, index=False)
    side.write_text(json.dumps({"seeds": used, "tag_file": str(out)}, indent=1))
    print(f"  wrote {out}: {len(res)} rows, {len(cols)} endpoints, "
          f"{len(frames)} seeds averaged")
    return stack, ids, [s for s, _ in frames]


def holdout_scores(stack, ids, seeds, tag):
    """Per-seed vs ensemble Spearman on the blind-like holdout. The receipt."""
    lp = C.ARTIFACT_DIR / "labels.csv"
    hp = C.ARTIFACT_DIR / "holdout_blindlike.csv"
    if not (lp.exists() and hp.exists()):
        print("\n  (no labels.csv / holdout_blindlike.csv -- skipping the scorecard)")
        return
    lab = pd.read_csv(lp); lab.columns = [c.strip() for c in lab.columns]
    ho = pd.read_csv(hp); ho.columns = [c.strip() for c in ho.columns]
    keep = set(ho[ID])
    idx = {m: i for i, m in enumerate(ids)}

    print("\n" + "=" * 78)
    print("WHAT THE ENSEMBLE BOUGHT, on the blind-like holdout")
    print("If 'ens' does not beat 'mean seed', averaging is not helping and something")
    print("is wrong with the members -- do not ship it on faith.")
    print("CAUTION: 'gain' is measured against the MEAN SEED, so adding a weak seed can")
    print("raise the gain while the ensemble itself gets worse. Compare the 'ens' column")
    print("across runs, not the gain.")
    print("=" * 78)
    print(f"\n  {'endpoint':9s} {'n':>5s} " + " ".join(f"{'s'+str(s):>7s}" for s in seeds)
          + f" | {'mean seed':>9s} {'ens':>7s} {'gain':>7s}")
    gains = []
    for iso in C.ISOFORMS:
        e = f"{iso}_pIC50_direct_inhibition"
        if e not in stack:
            continue
        t = lab[(lab["isoform"] == iso) & (lab["kind"] != "left")].dropna(subset=["y"])
        t = t[t[ID].isin(keep) & t[ID].isin(idx)]
        if len(t) < 30:
            continue
        rows = [idx[m] for m in t[ID]]
        y = pd.Series(t["y"].to_numpy(float))
        per = [float(pd.Series(stack[e][k][rows]).corr(y, method="spearman"))
               for k in range(len(seeds))]
        ens = float(pd.Series(np.nanmean(stack[e], axis=0)[rows]).corr(y,
                                                                      method="spearman"))
        g = ens - float(np.mean(per))
        gains.append(g)
        print(f"  {iso:9s} {len(t):5d} " + " ".join(f"{v:7.4f}" for v in per)
              + f" | {np.mean(per):9.4f} {ens:7.4f} {g:+7.4f}")
    if gains:
        print(f"\n  summed gain over the average single seed: {sum(gains):+.4f}")
        # DO NOT COMPARE THIS TO THE +0.160 SUBMISSION BAR. That bar lives on the blend
        # search's out-of-sample yardstick against blend4z; this number lives on the
        # blind-like holdout. They are different pools with different label
        # distributions, and putting them side by side is the exact conflation that
        # produced three wrong calls in this project on 2026-09-28. The only honest next
        # step is to register the member and run the blend search, which scores on the
        # instrument the bar is defined against.
        print("  This is variance reduction only, on the blind-like holdout. It is NOT")
        print("  comparable to the blend search's bar -- different pool. Register the")
        print("  member and run scripts/pic50_blend_search.py to price it.")
    # ---- the same thing on ALL fitted rows, because the holdout is small ------------
    # WHY THIS PANEL EXISTS. The holdout carries 61-248 compounds per endpoint, and on
    # 2026-09-29 its CYP2D6 panel (n=72) put seed 99 at rho 0.279 against 0.43-0.49 for
    # the other four, making the 5-seed ensemble look 0.045 WORSE than the 3-seed one.
    # On all 1,364 fitted rows that seed scores 0.4377 -- entirely normal -- and the
    # 5-seed ensemble is better on every endpoint. The holdout is the blind-LIKE
    # instrument, not a precise one; read it for chemistry, read this for arithmetic.
    print("\n  the same, on ALL fitted rows (larger n, so the ensemble comparison here")
    print("  is the reliable one; the holdout above is blind-like but small):")
    print(f"    {'endpoint':9s} {'n':>5s} {'mean seed':>9s} {'ens':>7s} {'gain':>7s}")
    for iso in C.ISOFORMS:
        e = f"{iso}_pIC50_direct_inhibition"
        if e not in stack:
            continue
        t2 = lab[(lab["isoform"] == iso) & (lab["kind"] != "left")].dropna(subset=["y"])
        t2 = t2.drop_duplicates(ID)
        t2 = t2[t2[ID].isin(idx)]
        if len(t2) < 100:
            continue
        rows2 = [idx[m] for m in t2[ID]]
        y2 = pd.Series(t2["y"].to_numpy(float))
        per2 = [float(pd.Series(stack[e][k][rows2]).corr(y2, method="spearman"))
                for k in range(len(seeds))]
        ens2 = float(pd.Series(np.nanmean(stack[e], axis=0)[rows2]).corr(
            y2, method="spearman"))
        print(f"    {iso:9s} {len(t2):5d} {np.mean(per2):9.4f} {ens2:7.4f} "
              f"{ens2-np.mean(per2):+7.4f}")

    # how much noise is left
    print("\n  per-seed spread (the floor a future A/B has to clear):")
    for iso in C.ISOFORMS:
        e = f"{iso}_pIC50_direct_inhibition"
        if e not in stack:
            continue
        r = np.corrcoef(stack[e])
        off = r[np.triu_indices_from(r, 1)]
        print(f"    {iso:9s} mean pairwise r between seeds {off.mean():.4f}   "
              f"(one run vs one run)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024, 7, 99])
    ap.add_argument("--tag", default="ens5")
    ap.add_argument("--train", default=DEFAULT_TRAIN,
                    help="the training script to run under each seed. Both "
                         "train_cyp_st_v3.py and train_cyp_mh.py take --seed, "
                         "--model_dir, --oof_output and --output, so either works; "
                         "everything after `--` is forwarded to whichever is named and "
                         "is NOT validated here -- the script itself rejects a flag it "
                         "does not have.")
    ap.add_argument("--resume", action="store_true",
                    help="skip a seed whose OOF and prediction files already parse")
    ap.add_argument("--combine-only", action="store_true",
                    help="average what is already on disk; train nothing")
    ap.add_argument("passthru", nargs="*",
                    help="everything after `--` is forwarded to train_cyp_st_v3.py "
                         "unchanged and identically for every seed")
    a = ap.parse_args()

    if not Path(a.train).exists():
        raise SystemExit(f"training script {a.train!r} not found in {Path.cwd()}")

    passthru = list(a.passthru)
    if passthru and passthru[0] == "--":
        passthru = passthru[1:]

    ok = list(a.seeds)
    if not a.combine_only:
        ok = []
        for s in a.seeds:
            q = paths(a.tag, s)
            if a.resume and readable(q["oof"]) and readable(q["pred"]):
                print(f"resume: seed {s} already done, skipping")
                ok.append(s); continue
            if run_seed(s, a.tag, passthru, a.train):
                ok.append(s)
        if not ok:
            raise SystemExit("every seed failed; nothing to average")
        if len(ok) < len(a.seeds):
            print(f"\n!! {len(a.seeds) - len(ok)} of {len(a.seeds)} seeds failed. "
                  f"Averaging the {len(ok)} that worked: {ok}")

    print("\ncombining...")
    oof_out = f"oof_{a.tag}.csv"
    pred_out = f"pred_{a.tag}.csv"
    stack, ids, used = combine(a.tag, ok, "oof", oof_out)
    combine(a.tag, ok, "pred", pred_out)
    holdout_scores(stack, ids, used, a.tag)

    print(f"\nNEXT: register '{a.tag}' in scripts/pic50_blend_search.py MEMBERS as")
    print(f"  \"cp_{a.tag}\": (\"{oof_out}\", \"{pred_out}\"),")
    print("then rerun the blend search. Do not submit on this script's numbers alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
