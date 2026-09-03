@echo off
REM One-click benchmark for drift-sense -- Windows cmd double-click
REM Assumes internet for pip install. Delegates to benchmark.ps1 for real work.
REM   double-click benchmark.bat              -> quick (40)
REM   benchmark.bat --full                    -> full (120)
REM   benchmark.bat --quick --seed 42 --num 20
setlocal
cd /d "%~dp0"
echo === drift-sense benchmark (cmd) ===
echo Repo: %CD%

REM Prefer py launcher, fallback to python
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3.11 -V >nul 2>&1
  if %ERRORLEVEL%==0 (
    echo Found py launcher (Python 3.11)
    powershell -NoProfile -ExecutionPolicy Bypass -File benchmark.ps1 %*
    goto :end
  )
)
where python >nul 2>&1
if %ERRORLEVEL%==0 (
  echo Found python
  powershell -NoProfile -ExecutionPolicy Bypass -File benchmark.ps1 %*
  goto :end
)
echo ERROR: no Python found. Install Python 3.11 from https://www.python.org ^(check "py launcher" box^).
pause
exit /b 1
:end
if "%~1"=="" pause
