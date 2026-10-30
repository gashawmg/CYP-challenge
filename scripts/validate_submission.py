#!/usr/bin/env python
r"""Pre-flight a submission file against the challenge's OWN validator, before uploading.

    python scripts\validate_submission.py artifacts\submission_blend4z.csv
    check_submission artifacts\submission_blend4z.csv          (same thing, shorter)

Run this every time, even when the file "obviously" looks right. submission_blend4z.csv
was rejected by the platform for a missing SMILES column after its column listing had
been checked by eye -- the validator requires BOTH "SMILES" and "Molecule_Name" as
identifier columns, and a listing that looks reasonable is not the same as a file that
passes.

The track is detected from the columns, so there is nothing to remember:

    *_pIC50_direct_inhibition  ->  validation.activity_validation
    *_is_TDI                   ->  validation.tdi_validation

It reports exactly what the platform would report and nothing else -- it does not judge
whether the numbers are any good, only whether the file will be accepted.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def tutorial_root() -> Path | None:
    for p in (ROOT.parent / "CYP-Challenge-Tutorial-main" / "CYP-Challenge-Tutorial-main",
              ROOT.parent / "CYP-Challenge-Tutorial-main",
              ROOT / "CYP-Challenge-Tutorial-main"):
        if (p / "validation" / "activity_validation.py").exists():
            return p
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print(r"usage: python scripts\validate_submission.py <submission csv>")
        return 2
    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"FAIL  no such file: {path}")
        return 1

    tr = tutorial_root()
    if tr is None:
        print("FAIL  cannot find the challenge tutorial repo, so the official validator "
              "is unavailable -- do not upload an unvalidated file")
        return 1
    sys.path.insert(0, str(tr))

    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    if any(c.endswith("_is_TDI") for c in d.columns):
        from validation.tdi_validation import validate_tdi_submission as check
        track, label = "TDI", "Classification Prediction"
    elif any(c.endswith("_pIC50_direct_inhibition") for c in d.columns):
        from validation.activity_validation import validate_activity_submission as check
        track, label = "pIC50", "Regression Prediction"
    else:
        print(f"FAIL  {path.name}: columns match neither track -- {list(d.columns)}")
        return 1

    from cypkit import config as C, io  # noqa: PLC0415
    t = io.load("test")
    t.columns = [c.strip() for c in t.columns]
    ids = {str(v) for v in t[C.ID_COL]}

    ok, errs = check(path, expected_ids=ids)
    print(f"{path.name}\n  track    {track}  (upload under \"{label}\")\n"
          f"  rows     {len(d)}\n  columns  {list(d.columns)}")
    if ok:
        print("\n  PASS -- this file will be accepted")
        return 0
    print("\n  FAIL -- the platform would reject this:")
    for e in errs:
        print(f"    - {e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
