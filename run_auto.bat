@echo off
REM ---------------------------------------------------------------------------
REM Hourly job: capture the leaderboard, then ship whatever is queued.
REM
REM Register it (run this once, in an ADMIN Command Prompt):
REM
REM   schtasks /create /tn "CYP challenge auto" /tr "D:\CYP450\cyp\run_auto.bat" ^
REM            /sc hourly /mo 1 /st 00:05 /rl highest /f
REM
REM Check it:      schtasks /query /tn "CYP challenge auto" /v /fo list
REM Run it now:    schtasks /run   /tn "CYP challenge auto"
REM Remove it:     schtasks /delete /tn "CYP challenge auto" /f
REM
REM HOURLY, not twice daily, is deliberate. The 12-hour rule lives in
REM auto_submit.py and is enforced against artifacts\submission_log.csv, so a
REM sleeping or offline machine costs nothing: the next hour it is awake, the
REM window is still open and the queued file ships. A twice-daily schedule
REM would silently skip a whole cycle every time the PC was asleep at the wrong
REM minute.
REM ---------------------------------------------------------------------------
setlocal
set PROJ=D:\CYP450\cyp
set CONDA_ENV=base
set LOGDIR=%PROJ%\artifacts\autologs

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set DSTAMP=%%c-%%a-%%b
set OUT=%LOGDIR%\auto_%DSTAMP%.log

call "%USERPROFILE%\anaconda3\Scripts\activate.bat" %CONDA_ENV%
if errorlevel 1 (
  echo [%date% %time%] could not activate conda env %CONDA_ENV% >> "%OUT%"
  exit /b 1
)

cd /d "%PROJ%"
echo. >> "%OUT%"
echo ===== %date% %time% ===== >> "%OUT%"

REM Leaderboard first: if a result landed since the last run it is in the ledger
REM BEFORE anything new goes out, so the record is never behind the submissions.
python scripts\fetch_leaderboard.py >> "%OUT%" 2>&1
python scripts\auto_submit.py       >> "%OUT%" 2>&1

endlocal
