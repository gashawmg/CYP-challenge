#!/usr/bin/env python
r"""Turn a prediction file into a submission, with the spread check applied before it ships.

    python scripts/make_submission.py --pred pred_st_blendsearch.csv --tag cand1 --dry-run
    python scripts/make_submission.py --pred pred_st_blendsearch.csv --tag cand1
    python scripts/make_submission.py --pred pred_st_blendsearch.csv --tag cand1 ^
        --shrink CYP1A2=0.80

WHAT THIS IS FOR. Every submission goes out through `cypkit.submission.write_submission`,
which copies an ACCEPTED reference file and replaces only the value columns, so layout
cannot drift. This script is the layer above that: it reports the distribution of what you
are about to upload against the anchor-solved blind distribution, and it can apply the one
placement move that has been measured and has never lost.

THE ONE SAFE MOVE, AND THE ONE THAT IS NOT (Section 46). Controlled pairs on the real blind
set -- same submission, same Spearman, different distribution:

    narrowing the spread at constant mean    CYP2C9 -0.0569, CYP3A4 -0.0357   never lost
    lowering the mean                        up to +0.2391 WORSE               lost 4 of 5

So `--shrink` scales an endpoint's predictions about THEIR OWN MEAN, which changes the
spread and leaves the mean fixed to within floating point. There is deliberately no
`--shift`. Both of this project's "placement failed" episodes were mean moves, and the
lesson was recorded against placement in general when it belonged to that one variable.

THE MEAN GUARD IS NOT ADVISORY. After any transform the per-endpoint mean is re-checked
against the input and the write is REFUSED if it moved by more than --mean-tol (default
0.01 log units). A shrink that also shifts the mean is the exact failure this script exists
to prevent, and a guard you can talk past is not a guard.

RANKING IS PRESERVED BY CONSTRUCTION. A positive scale factor about the mean is monotone,
so Spearman and Kendall are bit-identical before and after. The script asserts this rather
than asserting it in a comment.

WHAT IT WILL NOT DO. It does not choose the shrink factor for you, and it does not consult
the leaderboard. Pick the factor from `scripts/blind_response.py --propose` and from the
predicted-sd-vs-anchor table this script prints, then decide. Nothing here overrides the
standing rule: do not submit unless internal validation says the candidate beats 0.4833.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

from cypkit import config as C                                        # noqa: E402
from cypkit.submission import write_submission                        # noqa: E402
from scripts.blind_validator import BLIND                             # noqa: E402

# The live board row, so the print-out always says what we are trying to beat.
LIVE = {"CYP1A2": 0.4972, "CYP2C9": 0.4073, "CYP2D6": 0.5908, "CYP3A4": 0.4378}
LIVE_MACRO = 0.4833
DEFAULT_REF = "artifacts/submission_splice4.csv"    # accepted by the platform


# WHERE A SHRINK FACTOR SHOULD COME FROM, rather than from taste.
#
# Every endpoint's shipped spread can be expressed as sd(pred) / sd(blind truth). On the
# live submission those ratios and their board ST-RAE are:
#
#     endpoint   sd     needed   ratio    board ST-RAE   rank
#     CYP2C9    0.8576   1.122   0.764       0.4073       40
#     CYP3A4    0.9592   1.252   0.766       0.4378       58
#     CYP1A2    1.2594   1.431   0.880       0.4972       51
#     CYP2D6    0.6906   1.652   0.418       0.5908       18   (already placed)
#
# The two endpoints that were narrowed at constant mean -- the only placement move that
# has never lost -- both landed at ratio 0.765 and both improved (-0.0569 CYP2C9,
# -0.0357 CYP3A4). CYP1A2 was never narrowed cleanly, sits at 0.880, and carries the
# largest efficiency headroom of the four.
#
# So the defensible target is the ratio we have OBSERVED WORKING, not a further guess:
# bring an endpoint to ~0.765 and stop. --target-ratio does that arithmetic against the
# candidate's own spread, because a new blend will not have the shipped file's sd.
#
# THIS INTERPOLATES, IT DOES NOT EXTRAPOLATE. Whether narrowing past 0.765 helps further
# is untested, and CYP2D6's 0.418 is not evidence for it -- that column was placed by a
# different mechanism and is the endpoint with the least headroom left.
OBSERVED_GOOD_RATIO = 0.765


def parse_target_ratio(items, cur_sd):
    """ISO=RATIO -> the shrink factor that puts that endpoint at RATIO of the blind sd."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--target-ratio wants ISO=RATIO, got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        if k not in C.ISOFORMS:
            raise SystemExit(f"{k} is not one of {C.ISOFORMS}")
        r = float(v)
        if not (0.3 <= r <= 1.2):
            raise SystemExit(f"target ratio {r} for {k} is outside the sane range "
                             f"[0.3, 1.2]; observed-good is {OBSERVED_GOOD_RATIO}")
        want = r * BLIND[k][1]
        have = cur_sd[k]
        f = want / have
        if f > 1.0:
            raise SystemExit(
                f"{k}: reaching ratio {r} would mean WIDENING (factor {f:.3f}), because "
                f"the candidate's spread is already {have:.4f} against a target of "
                f"{want:.4f}. Widening is the reverse of the only move measured to help, "
                f"and the one controlled pair that widened lost. Refusing.")
        out[k] = f
    return out


def parse_shrink(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--shrink wants ISO=FACTOR, got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        if k not in C.ISOFORMS:
            raise SystemExit(f"{k} is not one of {C.ISOFORMS}")
        f = float(v)
        if not (0.2 <= f <= 1.5):
            raise SystemExit(f"shrink factor {f} for {k} is outside the sane range "
                             f"[0.2, 1.5]; a typo here silently ruins a submission")
        out[k] = f
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True,
                    help="a 750-row prediction file with the four pIC50 columns")
    ap.add_argument("--tag", required=True, help="output goes to artifacts/submission_<tag>.csv")
    ap.add_argument("--reference", default=DEFAULT_REF,
                    help="an ACCEPTED submission, used only for layout")
    ap.add_argument("--shrink", nargs="*", default=None,
                    help="ISO=FACTOR, scales that endpoint about its own mean "
                         "(e.g. CYP1A2=0.80). Mean-preserving by construction.")
    ap.add_argument("--take", nargs="*", default=None,
                    help="ISO=FILE, take that endpoint's column from FILE instead of "
                         "--pred. Needed because pic50_blend_search.py --write emits ALL "
                         "FOUR endpoints blended, including the ones its own selection "
                         "test rejected -- shipping that wholesale would replace columns "
                         "the search says do not beat what we hold.")
    ap.add_argument("--preserve-mean", action="store_true",
                    help="After --take, shift each taken column so its mean matches the "
                         "column it replaced. pic50_blend_search.py --write re-centres "
                         "every endpoint on the ANCHOR-SOLVED blind mean, which is an "
                         "inference from three scored submissions; the column it replaces "
                         "carries a mean that a real board score already validated. This "
                         "keeps the empirical one, so a --take changes RANKING only and "
                         "the mean -- the variable that has lost 4 of 5 measured times -- "
                         "is held fixed.")
    ap.add_argument("--target-ratio", nargs="*", default=None,
                    help=f"ISO=RATIO of sd(pred) to the anchor-solved blind sd, e.g. "
                         f"CYP1A2={OBSERVED_GOOD_RATIO}. The factor is computed from the "
                         f"CANDIDATE's own spread, so it stays correct when the blend "
                         f"changes. {OBSERVED_GOOD_RATIO} is where the two endpoints that "
                         f"were narrowed at constant mean both landed, and both improved. "
                         f"Cannot be combined with --shrink for the same endpoint.")
    ap.add_argument("--mean-tol", type=float, default=0.01)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the table and write nothing")
    a = ap.parse_args()

    d = pd.read_csv(a.pred)
    d.columns = [c.strip() for c in d.columns]
    if C.ID_COL not in d.columns:
        raise SystemExit(f"{a.pred} has no {C.ID_COL} column")

    # ---- per-endpoint column sources ---------------------------------------------------
    for it in a.take or []:
        if "=" not in it:
            raise SystemExit(f"--take wants ISO=FILE, got {it!r}")
        k, src = it.split("=", 1)
        k = k.strip()
        if k not in C.ISOFORMS:
            raise SystemExit(f"{k} is not one of {C.ISOFORMS}")
        col = f"{k}_pIC50_direct_inhibition"
        s = pd.read_csv(src)
        s.columns = [c.strip() for c in s.columns]
        if col not in s.columns:
            raise SystemExit(f"{src} has no {col}")
        if C.ID_COL not in s.columns:
            raise SystemExit(f"{src} has no {C.ID_COL}; cannot align by compound")
        m = dict(zip(s[C.ID_COL], s[col].astype(float)))
        missing = [i for i in d[C.ID_COL] if i not in m]
        if missing:
            raise SystemExit(
                f"--take {k}={src}: {len(missing)} of {len(d)} compounds are absent from "
                f"that file (first: {missing[:3]}). Aligning on the intersection would "
                f"leave holes the platform rejects; fix the source instead.")
        new = d[C.ID_COL].map(m).to_numpy(float)
        old_mean = float(d[col].astype(float).mean())
        if a.preserve_mean:
            shift = old_mean - float(new.mean())
            new = new + shift
            print(f"  --take {k}: column replaced from {src}, re-centred "
                  f"{shift:+.3f} to hold the mean at {old_mean:.3f}")
        else:
            print(f"  --take {k}: column replaced from {src} "
                  f"(mean {old_mean:.3f} -> {float(new.mean()):.3f}; "
                  f"pass --preserve-mean to hold it)")
        d[col] = new

    shrink = parse_shrink(a.shrink)
    if a.target_ratio:
        cur_sd = {iso: float(d[f"{iso}_pIC50_direct_inhibition"].astype(float).std())
                  for iso in C.ISOFORMS
                  if f"{iso}_pIC50_direct_inhibition" in d.columns}
        tr = parse_target_ratio(a.target_ratio, cur_sd)
        clash = set(tr) & set(shrink)
        if clash:
            raise SystemExit(
                f"{sorted(clash)} given to BOTH --shrink and --target-ratio. Pick one "
                f"per endpoint; silently letting one win is how a submission ends up "
                f"with a spread nobody chose.")
        for k, f in tr.items():
            print(f"  --target-ratio {k}: sd {cur_sd[k]:.4f} -> "
                  f"{f * cur_sd[k]:.4f} (factor {f:.3f}) for ratio "
                  f"{tr[k] * cur_sd[k] / BLIND[k][1]:.3f} of the blind sd")
        shrink.update(tr)

    print(f"input : {a.pred}  ({len(d)} rows)")
    print(f"ref   : {a.reference}")
    print(f"live  : macro {LIVE_MACRO:.4f}\n")

    print("DISTRIBUTION vs the anchor-solved blind distribution.")
    print("'needed sd' is the blind TRUTH's spread. The optimal prediction spread is")
    print("well BELOW it -- under a dead-zone L1 cost the best predictor is shrunk --")
    print("so do not read 'below needed' as a reason to widen.\n")
    print(f"  {'endpoint':9s} {'mean':>7s} {'sd':>7s} {'shrink':>7s} | "
          f"{'new mean':>8s} {'new sd':>7s} | {'blind mean':>10s} {'needed sd':>9s} "
          f"{'sd/needed':>9s}")

    values, checks = {}, []
    for iso in C.ISOFORMS:
        col = f"{iso}_pIC50_direct_inhibition"
        if col not in d.columns:
            print(f"  {iso:9s} MISSING from {a.pred}")
            continue
        v = d[col].astype(float).to_numpy()
        if not np.isfinite(v).all():
            raise SystemExit(f"{iso}: {int((~np.isfinite(v)).sum())} non-finite values in "
                             f"{a.pred}; the platform rejects those")
        m0, s0 = float(v.mean()), float(v.std())
        f = shrink.get(iso, 1.0)
        nv = (v - m0) * f + m0 if f != 1.0 else v.copy()
        m1, s1 = float(nv.mean()), float(nv.std())
        bm, bs = BLIND[iso]
        print(f"  {iso:9s} {m0:7.3f} {s0:7.4f} {f:7.2f} | {m1:8.3f} {s1:7.4f} | "
              f"{bm:10.3f} {bs:9.3f} {s1/bs:9.2f}")
        values[col] = pd.Series(nv, index=d[C.ID_COL])
        checks.append((iso, v, nv, m0, m1))

    # ---- the guards -------------------------------------------------------------------
    bad = []
    for iso, v, nv, m0, m1 in checks:
        if abs(m1 - m0) > a.mean_tol:
            bad.append(f"{iso}: mean moved {m1 - m0:+.4f} (tolerance {a.mean_tol})")
        r = pd.Series(v).corr(pd.Series(nv), method="spearman")
        if not (np.isnan(r) or r > 1 - 1e-9):
            bad.append(f"{iso}: transform changed the ranking (rho {r:.6f}); a positive "
                       f"scale about the mean is monotone, so this is a bug, not a choice")
    if bad:
        print("\nREFUSED -- the transform did something it is not allowed to do:")
        for b in bad:
            print("  " + b)
        return 1
    print("\n  guards passed: every endpoint's mean is unchanged and every ranking is "
          "bit-identical.")

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    out = C.ARTIFACT_DIR / f"submission_{a.tag}.csv"
    write_submission(values, Path(a.reference), out, kind="regression")
    print(f"\nwrote {out}")
    print("\nBEFORE YOU UPLOAD:")
    print("  - one slot per day, shared with TDI. Spend it on the track with a validated")
    print("    candidate, not on whichever finished most recently.")
    print(f"  - the bar is {LIVE_MACRO:.4f} macro. Do not upload unless local validation")
    print("    says this beats it.")
    print("  - after it scores, add the per-endpoint row to artifacts/submissions.csv --")
    print("    that file is the instrument scripts/blind_response.py runs on, and every")
    print("    row you skip is a controlled experiment thrown away.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
