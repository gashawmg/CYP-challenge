#!/usr/bin/env python
r"""A private scoring service for the CYP pIC50 track -- the thing to build FIRST.

    python scripts/eval_service.py init
    python scripts/eval_service.py score pred_clean2.csv --note "chemprop, curves only"
    python scripts/eval_service.py status
    python scripts/eval_service.py calibrate

WHY THIS EXISTS
---------------
Smbatyan et al. (Deep Origin, bioRxiv 2026.10.01.755314) replayed the ExpansionRx and PXR
challenges with LLM agents and placed 4th/104 and 1st/104. The enabling piece was not the
agent. It was this:

    "Retrospective challenge replays used PRIVATE EVALUATION SERVICES that withheld test
     labels and enforced the specified submission and feedback protocols."

An agent -- or a person -- that iterates against a validation set it can inspect will fit
that set within a few loops. This project has already hit the mild version of that: early
stopping was selecting checkpoints on the very fold being reported as held out. A loop
that runs twenty times unsupervised will find every such crack. So the scorer owns the
labels, reports aggregates only, and counts submissions.

WHAT IT GUARANTEES, AND WHAT IT CANNOT
--------------------------------------
Guaranteed: the holdout IDs are public (you must be able to exclude them from training),
the holdout LABELS are not, only aggregate per-endpoint numbers come back, and every
scored submission is logged with a content hash so a result can always be traced to the
file that produced it.

Not guaranteed: this is a file on the same disk as everything else. It stops accident and
self-deception, not determined cheating. That is the right threat model for a
single-researcher project.

THE ONE DESIGN DECISION THAT MATTERS: WHAT GOES IN THE HOLDOUT
--------------------------------------------------------------
Measured against this project's own leaderboard history, scoring composition dominates
everything else. Three candidate compositions, same predictions (oof_mhens3_k3), against
the board's per-endpoint figures:

    scored on                    CYP1A2 rho   CYP2C9 rho   CYP2D6 rho   CYP3A4 rho
    above-floor curves only        0.4750       0.5731       0.4744       0.8234
    + screen non-hits at 4.0       0.5002       0.4654       0.4608       0.7622
    ALL curve-fitted rows          0.5753       0.6811       0.4677       0.8234
    BOARD                          0.7892       0.8454       0.4888       0.8109

ALL CURVE-FITTED ROWS is the closest, on three of four endpoints and on ST-RAE too. The
reason is visible in the data: CYP3A4's training labels are continuous and full-range
(2,335 fits spanning 1.91-7.19, only 23% censored) and CYP3A4 is the ONLY endpoint whose
local numbers already tracked the board. The blind set is evidently fully curve-fitted in
the same way. The screen non-hits are the opposite -- 2,700 rows pinned at exactly 4.0,
one enormous tie block that exists in our training data and not in the blind labels, which
is why adding them makes Spearman worse rather than better.

So: a row enters the holdout if that endpoint has a fitted pIC50, wherever it sits. The
sub-4.0 fits are included. That is a SCORING decision and is independent of the TRAINING
decision in train_cyp_clean.py, where dropping the sub-4.0 rows on CYP1A2/CYP2D6 measurably
helped. Train on what teaches; score on what resembles the target.

A RESIDUAL GAP REMAINS and is not explained away here: local ST-RAE still reads high
against the board (CYP1A2 0.78 vs 0.52). Part is placement -- the board figures come from
blended, placed submissions and these are raw out-of-fold predictions -- and part is that
out-of-fold training compounds are not blind compounds. `calibrate` fits the local-to-board
mapping from the submission history instead of pretending the scales match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
ID_COL, SMILES_COL = "Molecule_Name", "SMILES"
TRAIN_CSV = "data/cyp-challenge-TRAIN_inhibition.csv"
PRIVATE = Path(".eval_private")          # labels live here; never read by model code
PUBLIC = Path("artifacts/eval")          # holdout IDs + ledger live here
DAILY_BUDGET = 1                         # the real challenge: one slot a day, shared with TDI


def ep(iso):
    return f"{iso}_pIC50_direct_inhibition"


# ------------------------------------------------------------------ the metric
def st_rae(pred, lo, hi, const):
    """Soft-threshold relative absolute error.

    A prediction inside [lo, hi] costs nothing; outside, it pays the distance to the
    nearer edge. Normalised by the cost a single constant prediction would incur, so
    1.0 means "no better than one number for every compound" and lower is better.
    """
    def cost(q):
        over = np.where(np.isfinite(hi), np.clip(q - hi, 0, None), 0.0)
        under = np.where(np.isfinite(lo), np.clip(lo - q, 0, None), 0.0)
        return float(np.sum(over + under))
    base = cost(np.full(len(pred), float(const)))
    return cost(pred) / base if base > 0 else float("nan")


def metrics(pred, y, lo, hi, const):
    s, t = pd.Series(pred), pd.Series(y)
    vp = float(np.var(pred))
    return dict(
        n=int(len(y)),
        st_rae=st_rae(pred, lo, hi, const),
        mae=float(np.mean(np.abs(pred - y))),
        r2=float(1 - np.mean((y - pred) ** 2) / np.var(y)) if np.var(y) > 0 else np.nan,
        rho=float(s.corr(t, method="spearman")),
        slope=float(np.cov(pred, y, bias=True)[0, 1] / vp) if vp > 1e-12 else np.nan,
    )


# ------------------------------------------------------------------ holdout
def scaffolds(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            out.append(f"__bad__{s}"); continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:                                            # noqa: BLE001
            sc = ""
        out.append(sc or f"__acyclic__{Chem.MolToSmiles(m)}")
    return out


def carve_holdout(tr, frac, seed):
    """A scaffold-disjoint holdout, balanced per endpoint on activity.

    Scaffold-disjoint because an analogue of a holdout compound sitting in training is
    the cheapest way to make a validation number look good and mean nothing. Balanced on
    activity because a holdout that happens to collect the potent compounds scores
    differently for reasons that have nothing to do with the model.
    """
    sc = scaffolds(tr[SMILES_COL].tolist())
    groups = {}
    for i, s in enumerate(sc):
        groups.setdefault(s, []).append(i)

    # a compound's balancing score: mean of its available per-endpoint z-scores
    Y = np.full((len(tr), len(ISOFORMS)), np.nan)
    for j, iso in enumerate(ISOFORMS):
        v = tr[ep(iso)].to_numpy(float)
        Y[:, j] = v
    mu, sd = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)
    Z = (Y - mu) / np.where(sd > 0, sd, 1.0)
    with np.errstate(invalid="ignore"):
        score = np.nanmean(Z, axis=1)
    score = np.where(np.isfinite(score), score, 0.0)
    bins = pd.qcut(pd.Series(score), 4, labels=False, duplicates="drop").to_numpy()
    n_bins = int(np.nanmax(bins)) + 1

    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    target = frac * np.array([(bins == b).sum() for b in range(n_bins)], float)
    have = np.zeros(n_bins)
    pick = []
    for k in keys:
        g = groups[k]
        add = np.zeros(n_bins)
        for i in g:
            add[bins[i]] += 1
        # take the group only while it moves every bin toward its target
        if np.all(have + add <= target + max(1.0, 0.02 * target.max())):
            have += add
            pick.extend(g)
    return sorted(tr.iloc[pick][ID_COL].tolist()), sc


def cmd_init(a):
    tr = pd.read_csv(TRAIN_CSV); tr.columns = [c.strip() for c in tr.columns]
    PRIVATE.mkdir(exist_ok=True); PUBLIC.mkdir(parents=True, exist_ok=True)
    if (PRIVATE / "holdout_labels.csv").exists() and not a.force:
        print("holdout already exists; pass --force to rebuild (this invalidates the "
              "ledger, because past scores were measured on a different holdout)")
        return 1

    ids, _ = carve_holdout(tr, a.frac, a.seed)
    held = tr[tr[ID_COL].isin(ids)]

    # private: the labels and the credible bounds
    cols = [ID_COL]
    for iso in ISOFORMS:
        c = ep(iso)
        cols += [c, c + "_conf_low", c + "_conf_high"]
    held[[c for c in cols if c in held.columns]].to_csv(
        PRIVATE / "holdout_labels.csv", index=False)

    # public: who is in it, and nothing else. Training code MUST read this and exclude.
    pub = held[[ID_COL, SMILES_COL]].copy()
    pub.to_csv(PUBLIC / "holdout_ids.csv", index=False)

    counts = {iso: int(held[ep(iso)].notna().sum()) for iso in ISOFORMS}
    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), seed=a.seed,
                    frac=a.frac, n_compounds=len(held), scored_per_endpoint=counts,
                    composition="all curve-fitted rows, including sub-4.0 fits",
                    train_rows=len(tr))
    (PUBLIC / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"holdout: {len(held)} compounds of {len(tr)}")
    for iso in ISOFORMS:
        print(f"  {iso:8s} {counts[iso]:5d} scored "
              f"({counts[iso] / max(tr[ep(iso)].notna().sum(), 1):.0%} of its curves)")
    print(f"\n  labels   -> {PRIVATE / 'holdout_labels.csv'}   (do not read this from "
          f"model code)\n  ids      -> {PUBLIC / 'holdout_ids.csv'}   (EXCLUDE these from "
          f"training)\n  manifest -> {PUBLIC / 'manifest.json'}")
    return 0


# ------------------------------------------------------------------ scoring
def load_ledger():
    f = PUBLIC / "ledger.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def cmd_score(a):
    lab_f = PRIVATE / "holdout_labels.csv"
    if not lab_f.exists():
        print("no holdout; run `init` first"); return 1
    lab = pd.read_csv(lab_f)
    sub = pd.read_csv(a.file); sub.columns = [c.strip() for c in sub.columns]
    if ID_COL not in sub.columns:
        print(f"{a.file}: no {ID_COL} column"); return 1

    led = load_ledger()
    today = time.strftime("%Y-%m-%d")
    used = int((led["date"] == today).sum()) if len(led) and "date" in led else 0
    if used >= a.budget and not a.dry_run:
        print(f"BUDGET: {used}/{a.budget} scored today. The real challenge gives one slot "
              f"a day shared with TDI, so the harness mirrors it.\n"
              f"Use --dry-run to score without spending a slot (it is NOT logged, and a "
              f"dry run you then act on is exactly the self-deception this prevents).")
        return 2

    j = lab.merge(sub, on=ID_COL, how="inner", suffixes=("", "_pred"))
    rows, missing = [], []
    for iso in ISOFORMS:
        c = ep(iso)
        pc = c + "_pred" if c + "_pred" in j.columns else c
        if pc not in sub.columns and pc not in j.columns:
            missing.append(iso); continue
        m = j[c].notna() & j[pc].notna()
        if m.sum() < 20:
            missing.append(iso); continue
        y = j.loc[m, c].to_numpy(float)
        p = j.loc[m, pc].to_numpy(float)
        lo = j.loc[m, c + "_conf_low"].to_numpy(float)
        hi = j.loc[m, c + "_conf_high"].to_numpy(float)
        rows.append(dict(iso=iso, **metrics(p, y, lo, hi, const=float(np.mean(y)))))

    if not rows:
        print("nothing scorable -- check the column names"); return 1

    print(f"\n{'=' * 74}\n  {Path(a.file).name}"
          + ("   [DRY RUN -- not logged]" if a.dry_run else "")
          + f"\n{'=' * 74}")
    print(f"  {'endpoint':10s}{'n':>6s}{'ST-RAE':>9s}{'MAE':>8s}{'R2':>8s}{'rho':>8s}"
          f"{'slope':>8s}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['iso']:10s}{r['n']:>6d}{r['st_rae']:>9.4f}{r['mae']:>8.4f}"
              f"{r['r2']:>8.4f}{r['rho']:>8.4f}{r['slope']:>8.3f}")
    macro = {k: float(np.mean([r[k] for r in rows]))
             for k in ("st_rae", "mae", "r2", "rho")}
    print("  " + "-" * 70)
    print(f"  {'MACRO':10s}{'':>6s}{macro['st_rae']:>9.4f}{macro['mae']:>8.4f}"
          f"{macro['r2']:>8.4f}{macro['rho']:>8.4f}")
    if missing:
        print(f"\n  NOT SCORED: {', '.join(missing)} -- absent or too few overlapping rows")

    # where this sits against everything scored before
    if len(led) and "macro_st_rae" in led:
        best = led["macro_st_rae"].min()
        d = macro["st_rae"] - best
        print(f"\n  best logged macro ST-RAE so far: {best:.4f}   this: "
              f"{macro['st_rae']:.4f} ({d:+.4f})"
              + ("   <- NEW BEST" if d < 0 else "   <- not an improvement"))

    board = predict_board(macro["st_rae"])
    if board:
        print(f"\n  {board}")

    if not a.dry_run:
        h = hashlib.sha256(Path(a.file).read_bytes()).hexdigest()[:12]
        rec = dict(date=today, time=time.strftime("%H:%M:%S"), file=str(a.file),
                   sha=h, note=a.note or "", macro_st_rae=macro["st_rae"],
                   macro_mae=macro["mae"], macro_r2=macro["r2"], macro_rho=macro["rho"],
                   **{f"{r['iso']}_st_rae": r["st_rae"] for r in rows})
        led = pd.concat([led, pd.DataFrame([rec])], ignore_index=True)
        PUBLIC.mkdir(parents=True, exist_ok=True)
        led.to_csv(PUBLIC / "ledger.csv", index=False)
        print(f"\n  logged as {h}   ({used + 1}/{a.budget} used today)")
    return 0


# ------------------------------------------------------------------ calibration
def board_pairs():
    """(local, board) pairs for submissions that were both scored here and uploaded."""
    f = Path("artifacts/submissions.csv")
    led = load_ledger()
    if not f.exists() or not len(led):
        return pd.DataFrame()
    s = pd.read_csv(f)
    s = s[(s["track"] == "pIC50") & (s["scope"] == "overall")
          & (s["metric"] == "ST-RAE")][["submission", "value"]]
    if "note" not in led:
        return pd.DataFrame()
    j = led.merge(s, left_on="note", right_on="submission", how="inner")
    return j[["note", "macro_st_rae", "value"]].rename(columns={"value": "board"})


def predict_board(local):
    p = board_pairs()
    if len(p) < 3:
        return ("board prediction unavailable: fewer than 3 submissions have been both "
                "scored here and uploaded. Score a submission with --note set to its "
                "name in artifacts/submissions.csv to build the mapping.")
    k, b = np.polyfit(p["macro_st_rae"], p["board"], 1)
    resid = p["board"] - (k * p["macro_st_rae"] + b)
    return (f"predicted board macro ST-RAE {k * local + b:.4f} "
            f"+/- {resid.std(ddof=1):.4f}  (from {len(p)} anchored submissions, "
            f"transfer slope {k:.2f})")


def backfill(a):
    """Score every PAST submission that already has a board number. Costs no slot.

    The daily budget exists to stop repeated local scoring being used to tune against the
    holdout. That reasoning does not apply here: these files were uploaded weeks ago and
    their board scores are already spent and recorded. Measuring them locally adds no new
    information to tune against -- it only pairs a local number with a board number that
    history has already fixed. So the budget is bypassed, deliberately and only here.
    """
    f = Path("artifacts/submissions.csv")
    if not f.exists():
        print("artifacts/submissions.csv not found"); return []
    s_ = pd.read_csv(f)
    s_ = s_[(s_["track"] == "pIC50") & (s_["scope"] == "overall")
            & (s_["metric"] == "ST-RAE")]
    lab = pd.read_csv(PRIVATE / "holdout_labels.csv")
    rows, missing = [], []
    for _, r in s_.iterrows():
        name = str(r["submission"])
        cand = [Path(f"artifacts/submission_{name}.csv"), Path(f"submission_{name}.csv"),
                Path(f"_superseded/submission_{name}.csv")]
        src = next((c for c in cand if c.exists()), None)
        if src is None:
            missing.append(name); continue
        sub = pd.read_csv(src); sub.columns = [c.strip() for c in sub.columns]
        j = lab.merge(sub, on=ID_COL, how="inner", suffixes=("", "_p"))
        if not len(j):
            missing.append(f"{name} (no overlap)"); continue
        per = []
        for iso in ISOFORMS:
            c = ep(iso)
            pc = c + "_p" if c + "_p" in j.columns else c
            if pc not in j.columns:
                continue
            m = j[c].notna() & j[pc].notna()
            if m.sum() < 20:
                continue
            y = j.loc[m, c].to_numpy(float)
            per.append(st_rae(j.loc[m, pc].to_numpy(float),
                              j.loc[m, c + "_conf_low"].to_numpy(float),
                              j.loc[m, c + "_conf_high"].to_numpy(float),
                              float(np.mean(y))))
        if len(per) == len(ISOFORMS):
            rows.append(dict(note=name, macro_st_rae=float(np.mean(per)),
                             board=float(r["value"]),
                             rank=r.get("rank", float("nan")), file=str(src)))
    if missing:
        print(f"  unusable: {', '.join(missing)}\n")
    if not rows:
        print("NO ANCHORS CAN BE BUILT THIS WAY, and the reason is structural.\n")
        print("  A submission file predicts the 750 BLIND compounds. This holdout is\n"
              "  1,000 TRAINING compounds. They do not intersect -- by construction,\n"
              "  since the blind labels are what the challenge withholds.\n")
        print("  To anchor local against board you need ONE MODEL scored both ways:\n"
              "    - its out-of-fold predictions on the holdout  (an oof_*.csv), and\n"
              "    - its blind predictions uploaded and scored   (a board number).\n")
        print("  Past submissions were blends and splices whose out-of-fold equivalents\n"
              "  were not kept, so history cannot supply this retroactively. Each anchor\n"
              "  costs one real submission slot: upload a file whose matching oof_*.csv\n"
              "  has been scored here, then record the board number in\n"
              "  artifacts/submissions.csv under the same name as the --note you used.")
    return rows


def cmd_calibrate(a):
    if a.backfill:
        rows = backfill(a)
        if rows:
            df = pd.DataFrame(rows)
            PUBLIC.mkdir(parents=True, exist_ok=True)
            df.to_csv(PUBLIC / "anchors.csv", index=False)
            print(f"{'submission':22s}{'local':>9s}{'board':>9s}{'rank':>7s}")
            print(rule() if False else "-" * 47)
            for _, r in df.sort_values("board").iterrows():
                rk = "" if not np.isfinite(r["rank"]) else f"{int(r['rank'])}"
                print(f"{r['note'][:21]:22s}{r.macro_st_rae:>9.4f}{r.board:>9.4f}{rk:>7s}")
            k, b = np.polyfit(df.macro_st_rae, df.board, 1)
            pred = k * df.macro_st_rae + b
            resid = df.board - pred
            rho = df[["macro_st_rae", "board"]].corr(method="spearman").iloc[0, 1]
            print("-" * 47)
            print(f"\n  {len(df)} anchors, no submission slots spent")
            print(f"  board ~ {k:.3f} x local + {b:.3f}")
            print(f"  residual sd {resid.std(ddof=1):.4f}   Spearman(local, board) "
                  f"{rho:.3f}")
            print(f"  a local gain of 0.01 is worth {k * 0.01:+.4f} on the board")
            print(f"\n  wrote {PUBLIC / 'anchors.csv'}")
            if rho < 0.5:
                print("\n  WARNING: local and board barely agree on ORDERING. The mapping "
                      "above\n  should not be used to choose between candidates until "
                      "that is understood.")
        return 0
    p = board_pairs()
    if not len(p):
        print("no anchored submissions yet. Score each uploaded file with\n"
              "  --note '<its name in artifacts/submissions.csv>'\n"
              "and the mapping builds itself.")
        return 0
    print(f"{'submission':28s}{'local':>9s}{'board':>9s}{'resid':>9s}")
    if len(p) >= 3:
        k, b = np.polyfit(p["macro_st_rae"], p["board"], 1)
    else:
        k, b = float("nan"), float("nan")
    for _, r in p.iterrows():
        pred = k * r.macro_st_rae + b
        print(f"{str(r['note'])[:27]:28s}{r.macro_st_rae:>9.4f}{r.board:>9.4f}"
              f"{r.board - pred:>9.4f}" if np.isfinite(k) else
              f"{str(r['note'])[:27]:28s}{r.macro_st_rae:>9.4f}{r.board:>9.4f}{'-':>9s}")
    if np.isfinite(k):
        print(f"\n  board ~ {k:.3f} x local + {b:.3f}")
        print(f"  a local gain of 0.01 is worth about {k * 0.01:.4f} on the board")
    print("\n  This mapping is the honest substitute for pretending local and board are\n"
          "  the same number. They are not: local scores out-of-fold TRAINING compounds,\n"
          "  the board scores blind ones, and board entries are placed and blended.")
    return 0


def cmd_status(a):
    led = load_ledger()
    man = PUBLIC / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        print(f"holdout: {m['n_compounds']} compounds, seed {m['seed']}, "
              f"created {m['created']}")
        print(f"  scored per endpoint: "
              + "  ".join(f"{k} {v}" for k, v in m["scored_per_endpoint"].items()))
    if not len(led):
        print("\nno submissions scored yet"); return 0
    today = time.strftime("%Y-%m-%d")
    print(f"\n{len(led)} scored, {int((led['date'] == today).sum())} today "
          f"(budget {a.budget}/day)\n")
    show = led.sort_values("macro_st_rae").head(a.top)
    print(f"  {'macro':>8s}  {'date':10s} {'sha':12s} note")
    for _, r in show.iterrows():
        print(f"  {r.macro_st_rae:>8.4f}  {r.date:10s} {r.sha:12s} {str(r.note)[:40]}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="carve the private holdout (run once)")
    i.add_argument("--frac", type=float, default=0.20)
    i.add_argument("--seed", type=int, default=20261010)
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_init)

    s = sub.add_parser("score", help="score a prediction CSV against the holdout")
    s.add_argument("file")
    s.add_argument("--note", default="", help="name it as in artifacts/submissions.csv "
                                              "if you also upload it -- that is what "
                                              "builds the board calibration")
    s.add_argument("--budget", type=int, default=DAILY_BUDGET)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_score)

    c = sub.add_parser("calibrate", help="local -> board mapping from the history")
    c.add_argument("--backfill", action="store_true",
                   help="score every past submission that already has a board number. "
                        "Costs NO submission slot -- those scores are already spent")
    c.set_defaults(fn=cmd_calibrate)

    t = sub.add_parser("status", help="budget and the best scores so far")
    t.add_argument("--top", type=int, default=10)
    t.add_argument("--budget", type=int, default=DAILY_BUDGET)
    t.set_defaults(fn=cmd_status)

    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
