#!/usr/bin/env python
r"""Ship a queued submission at the next legal window. Windows Task Scheduler job.

    python scripts\auto_submit.py --dry-run      # validate + report, submit nothing
    python scripts\auto_submit.py                # submit if a window is open

WHY THIS RUNS ON WINDOWS AND NOT IN A SANDBOX. huggingface.co answers 403 from both the
cloud container and the device sandbox -- egress is blocked in each. The submitting step
therefore has to run natively in the conda env where the HF token already lives. Schedule
it HOURLY; the 12-hour rule is enforced here, not by the schedule, so a missed hour or a
sleeping machine costs nothing and a machine that wakes up late still ships promptly.

WHAT IT DOES NOT DO: choose. The queue file is the gate, and a human writes it.

    artifacts\queue_pic50.txt     one line: a path, optionally "path | note"
    artifacts\queue_tdi.txt

Empty queue means no submission, and that is the correct outcome, not a failure -- an
unused window costs nothing while a wasted one costs a day. The reason this is not
automated is the project's own record: v7, v8, v9 and mh each looked like the best local
candidate and none beat v6; submission_v3_affine80 went out on an arithmetic argument and
came back 118th; and submission_blend4z was simultaneously a +0.028 win by one arithmetic
path and a -0.06 loss when measured directly. Every rule simple enough for a scheduler
would have shipped at least one of those.

THE 12-HOUR GUARD reads artifacts\submission_log.csv -- the file THIS script writes, not
the leaderboard -- so it is correct even when the board is slow to update. The two tracks
have independent windows and are checked independently.

FAILURE POLICY: a validation failure leaves the queue entry in place and exits non-zero,
so the next hourly run retries after you fix the file. A successful submit clears the
entry, so a crash after the API call cannot double-submit -- the log is written BEFORE
the queue is cleared and the guard reads the log.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "artifacts" / "submission_log.csv"
CONFIG = ROOT / "artifacts" / "submit_config.json"
MIN_GAP = timedelta(hours=12)
SPACE = "openadmet/cyp-challenge"

TRACKS = {
    "pic50": dict(queue="queue_pic50.txt", track_select="Regression Prediction",
                  validator="activity"),
    "tdi": dict(queue="queue_tdi.txt", track_select="Classification Prediction",
                validator="tdi"),
}


def tutorial_root() -> Path | None:
    """The tutorial repo holds the official validators; find it next to the project."""
    for p in (ROOT.parent / "CYP-Challenge-Tutorial-main" / "CYP-Challenge-Tutorial-main",
              ROOT.parent / "CYP-Challenge-Tutorial-main",
              ROOT / "CYP-Challenge-Tutorial-main"):
        if (p / "validation" / "activity_validation.py").exists():
            return p
    return None


def expected_ids() -> set[str]:
    from cypkit import config as C, io  # noqa: PLC0415
    t = io.load("test")
    t.columns = [c.strip() for c in t.columns]
    return {str(v) for v in t[C.ID_COL]}


def validate(path: Path, which: str) -> tuple[bool, list[str]]:
    tr = tutorial_root()
    if tr is None:
        return False, ["cannot find the tutorial repo, so cannot run the official "
                       "validator -- refusing to submit unvalidated"]
    sys.path.insert(0, str(tr))
    ids = expected_ids()
    if which == "activity":
        from validation.activity_validation import validate_activity_submission as v
    else:
        from validation.tdi_validation import validate_tdi_submission as v
    return v(path, expected_ids=ids)


def read_log() -> pd.DataFrame:
    if LOG.exists():
        d = pd.read_csv(LOG)
        d["when"] = pd.to_datetime(d["when"], utc=True, errors="coerce")
        return d
    return pd.DataFrame(columns=["when", "track", "file", "note", "status", "response"])


def last_submit(log: pd.DataFrame, track: str):
    s = log[(log["track"] == track) & (log["status"] == "ok")]
    return s["when"].max() if len(s) else None


def append_log(row: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    d = read_log()
    d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    d.to_csv(LOG, index=False)


def read_queue(name: str):
    q = ROOT / "artifacts" / name
    if not q.exists():
        return None, None
    txt = q.read_text(encoding="utf-8").strip()
    if not txt:
        return None, None
    line = [ln for ln in txt.splitlines() if ln.strip()][0]
    path, _, note = line.partition("|")
    p = Path(path.strip())
    return (p if p.is_absolute() else ROOT / p), note.strip()


def submit(path: Path, track_select: str, cfg: dict):
    from gradio_client import Client, handle_file  # noqa: PLC0415
    from huggingface_hub import get_token  # noqa: PLC0415

    token = get_token()
    if not token:
        raise RuntimeError("no HF token on this machine -- run `huggingface-cli login`")
    client = Client(SPACE, token=token)
    return client.predict(
        username=cfg["username"], user_alias=cfg["user_alias"],
        anon_checkbox=cfg.get("anon", True), participant_name=cfg["participant_name"],
        discord_username=cfg.get("discord_username", ""), email=cfg["email"],
        affiliation=cfg.get("affiliation", ""), model_tag=cfg.get("model_tag", ""),
        paper_checkbox=cfg.get("paper", False),
        # FALSE IS CORRECT AND IS NOT AN OVERSIGHT. The box reads "I used proprietary
        # data (not publicly available)". Our external data is PubChem AID 1851, which
        # is public, so this stays false -- as it is in the web form. It would have to
        # flip only if we ever trained on something genuinely non-public.
        proprietary_data_checkbox=cfg.get("proprietary_data", False),
        open_code_checkbox=cfg.get("open_code", False),
        track_select=track_select, file_input=handle_file(str(path)),
        api_name="/submit_predictions")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and report, contact nothing")
    ap.add_argument("--track", choices=list(TRACKS) + ["both"], default="both")
    a = ap.parse_args()

    if not CONFIG.exists():
        print(f"missing {CONFIG} -- copy submit_config.example.json and fill it in")
        return 2
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    log = read_log()
    now = datetime.now(timezone.utc)
    rc, acted = 0, False

    for name, spec in TRACKS.items():
        if a.track not in (name, "both"):
            continue
        path, note = read_queue(spec["queue"])
        if path is None:
            print(f"[{name}] queue empty -- nothing to ship")
            continue
        if not path.exists():
            print(f"[{name}] QUEUED FILE MISSING: {path}")
            rc = 1
            continue

        last = last_submit(log, name)
        if last is not None and now - last < MIN_GAP:
            nxt = last + MIN_GAP
            print(f"[{name}] window closed -- last ok submit {last:%Y-%m-%d %H:%M}Z, "
                  f"next {nxt:%Y-%m-%d %H:%M}Z ({(nxt - now).total_seconds()/3600:.1f}h)")
            continue

        ok, errs = validate(path, spec["validator"])
        if not ok:
            print(f"[{name}] VALIDATION FAILED for {path.name}; queue left in place")
            for e in errs:
                print(f"    - {e}")
            append_log(dict(when=now.isoformat(), track=name, file=path.name,
                            note=note or "", status="invalid",
                            response=" | ".join(errs)[:500]))
            rc = 1
            continue

        print(f"[{name}] {path.name} valid, window OPEN" + (f"  ({note})" if note else ""))
        if a.dry_run:
            print(f"[{name}] --dry-run: not submitting")
            continue

        try:
            resp = submit(path, spec["track_select"], cfg)
            # LOG BEFORE CLEARING THE QUEUE. If the process dies between these two, the
            # next run sees a fresh log entry, finds the window closed, and does not
            # resubmit. The reverse order could double-submit.
            append_log(dict(when=now.isoformat(), track=name, file=path.name,
                            note=note or "", status="ok", response=str(resp)[:500]))
            (ROOT / "artifacts" / spec["queue"]).write_text("", encoding="utf-8")
            print(f"[{name}] SUBMITTED {path.name}\n    {str(resp)[:300]}")
            acted = True
        except Exception as e:                                    # noqa: BLE001
            append_log(dict(when=now.isoformat(), track=name, file=path.name,
                            note=note or "", status="error",
                            response=f"{type(e).__name__}: {e}"[:500]))
            print(f"[{name}] SUBMIT FAILED ({type(e).__name__}: {e}); queue left in "
                  f"place, will retry next hour")
            rc = 1

    if not acted and rc == 0:
        print("nothing submitted this run")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
