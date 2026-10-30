#!/usr/bin/env python
r"""Read the challenge leaderboard and write our scores into the ledger.

    python scripts\fetch_leaderboard.py --probe     # discover the API surface, write nothing
    python scripts\fetch_leaderboard.py             # fetch, parse, append to the ledger

WHY --probe EXISTS. huggingface.co answers 403 from both sandboxes available to me, so I
could not inspect the Space's API while writing this and will not pretend to know its
schema. Run --probe once on your machine; it prints every endpoint `gradio_client`
advertises and every file in any leaderboard-looking dataset repo, and writes the result
to artifacts\leaderboard_probe.json. Paste that back and the column mapping below gets
replaced with the real names instead of the heuristics.

WHAT IT WRITES. Rows appended to artifacts\submissions.csv in the long format track.py
already reads:

    track,order,submission,scope,metric,value,rank,note

one row per (submission, endpoint, metric). This is the file whose per-target columns
have been filled in by hand from screenshots; the point of this script is to stop doing
that, because per-target history is what every placement and splice decision is made
from, and transcription is where those decisions silently go wrong.

IDEMPOTENT. A (track, submission, scope, metric) that is already present with the same
value is not appended again, so running this hourly is safe. A row whose value CHANGED is
reported loudly rather than quietly overwritten -- on this leaderboard that means either
a rescoring or a parse error, and both are worth a human look.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "artifacts" / "submissions.csv"
CONFIG = ROOT / "artifacts" / "submit_config.json"
PROBE_OUT = ROOT / "artifacts" / "leaderboard_probe.json"
SPACE = "openadmet/cyp-challenge"
ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")

# Heuristic column matching, replaced once --probe reports the real names.
METRIC_PATTERNS = {
    "ST-RAE": r"st[_\- ]?rae|rae",
    "MCC": r"mcc",
    "rho": r"rho|spearman",
    "R2": r"r2|r\^2",
    "MAE": r"\bmae\b",
}


def probe() -> int:
    from gradio_client import Client  # noqa: PLC0415
    from huggingface_hub import get_token  # noqa: PLC0415

    out: dict = {}
    client = Client(SPACE, token=get_token())
    try:
        out["api"] = client.view_api(return_format="dict", print_info=False)
    except Exception as e:                                        # noqa: BLE001
        out["api_error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(out.get("api", out.get("api_error")), indent=2)[:6000])

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
        api = HfApi()
        ds = [d.id for d in api.list_datasets(search="cyp-challenge", limit=30)]
        out["datasets"] = ds
        print("\ncandidate dataset repos:", ds)
        for repo in ds:
            try:
                out.setdefault("files", {})[repo] = api.list_repo_files(
                    repo, repo_type="dataset")[:40]
            except Exception:                                     # noqa: BLE001, S110
                pass
        print(json.dumps(out.get("files", {}), indent=2)[:3000])
    except Exception as e:                                        # noqa: BLE001
        out["hub_error"] = f"{type(e).__name__}: {e}"

    PROBE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {PROBE_OUT} -- send me this file and I will replace the heuristics")
    return 0


def fetch_tables() -> list[pd.DataFrame]:
    """Every table the Space will hand us; the caller decides which are leaderboards."""
    from gradio_client import Client  # noqa: PLC0415
    from huggingface_hub import get_token  # noqa: PLC0415

    client = Client(SPACE, token=get_token())
    tables: list[pd.DataFrame] = []
    try:
        api = client.view_api(return_format="dict", print_info=False)
        names = [n for n in (api.get("named_endpoints") or {})
                 if re.search(r"board|rank|result|score", n, re.I)]
    except Exception:                                             # noqa: BLE001
        names = []
    for n in names or ["/leaderboard"]:
        try:
            r = client.predict(api_name=n)
        except Exception:                                         # noqa: BLE001
            continue
        for obj in (r if isinstance(r, (list, tuple)) else [r]):
            if isinstance(obj, dict) and "data" in obj:
                tables.append(pd.DataFrame(obj["data"],
                                           columns=obj.get("headers")))
            elif isinstance(obj, str) and obj.endswith((".csv", ".json")):
                tables.append(pd.read_csv(obj) if obj.endswith(".csv")
                              else pd.read_json(obj))
    return tables


def classify(df: pd.DataFrame) -> str | None:
    cols = " ".join(str(c).lower() for c in df.columns)
    if re.search(r"mcc", cols):
        return "TDI"
    if re.search(r"rae|mae|r2|rho", cols):
        return "pIC50"
    return None


def our_rows(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    who = {str(cfg.get(k, "")).lower() for k in
           ("username", "user_alias", "participant_name")} - {""}
    idc = [c for c in df.columns
           if re.search(r"user|alias|name|participant|team", str(c), re.I)]
    if not idc or not who:
        return df.iloc[0:0]
    m = pd.Series(False, index=df.index)
    for c in idc:
        m |= df[c].astype(str).str.lower().isin(who)
    return df[m]


def to_ledger(df: pd.DataFrame, track: str) -> list[dict]:
    rows: list[dict] = []
    rank = None
    for c in df.columns:
        if re.fullmatch(r"\s*rank\s*", str(c), re.I):
            rank = c
    for _, r in df.iterrows():
        sub = next((str(r[c]) for c in df.columns
                    if re.search(r"model[_ ]?tag|submission|file", str(c), re.I)
                    and str(r[c]).strip()), "leaderboard")
        for c in df.columns:
            name = str(c)
            scope = next((i for i in ISOFORMS if i.lower() in name.lower()), "overall")
            metric = next((m for m, pat in METRIC_PATTERNS.items()
                           if re.search(pat, name, re.I)), None)
            if metric is None:
                continue
            try:
                val = float(r[c])
            except (TypeError, ValueError):
                continue
            rows.append(dict(track=track, order=pd.NA, submission=sub, scope=scope,
                             metric=metric, value=val,
                             rank=(r[rank] if rank else pd.NA), note="auto"))
    return rows


def merge(rows: list[dict]) -> int:
    if not rows:
        print("no rows parsed -- run --probe and send me the output")
        return 1
    old = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=list(rows[0]))
    key = ["track", "submission", "scope", "metric"]
    idx = {tuple(str(r[k]) for k in key): r for _, r in old.iterrows()}
    new, changed = [], []
    for r in rows:
        k = tuple(str(r[k_]) for k_ in key)
        if k not in idx:
            new.append(r)
        elif abs(float(idx[k]["value"]) - r["value"]) > 1e-9:
            changed.append((k, float(idx[k]["value"]), r["value"]))
    for k, a, b in changed:
        print(f"  CHANGED {k}: ledger {a} vs board {b} -- NOT overwritten, check why")
    if new:
        pd.concat([old, pd.DataFrame(new)], ignore_index=True).to_csv(LEDGER, index=False)
    print(f"{len(new)} new rows appended, {len(changed)} conflicting, "
          f"{len(rows) - len(new) - len(changed)} already present")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        return probe()
    if not CONFIG.exists():
        print(f"missing {CONFIG}")
        return 2
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for df in fetch_tables():
        track = classify(df)
        if track is None:
            continue
        mine = our_rows(df, cfg)
        print(f"  {track} table: {len(df)} rows, {len(mine)} ours")
        rows += to_ledger(mine, track)
    return merge(rows)


if __name__ == "__main__":
    raise SystemExit(main())
