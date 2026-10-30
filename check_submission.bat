@echo off
REM Pre-flight a submission file against the challenge's own validator.
REM
REM   check_submission artifacts\submission_blend4z.csv
REM
REM Worth running every time, whatever else changes: submission_blend4z.csv was
REM rejected for a missing SMILES column that looked fine in a column listing.
if "%~1"=="" (
  echo usage: check_submission ^<path to submission csv^>
  exit /b 2
)
python scripts\validate_submission.py "%~1"
