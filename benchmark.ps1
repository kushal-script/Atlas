# One-click benchmark for drift-sense -- Windows PowerShell
# Assumes internet for pip install. Modes: -Quick (40, default) vs -Full (120)
#   powershell -ExecutionPolicy Bypass -File benchmark.ps1
#   powershell -ExecutionPolicy Bypass -File benchmark.ps1 -Full
#   powershell -ExecutionPolicy Bypass -File benchmark.ps1 -Quick -Seed 42 -Num 20
param(
  [switch]$Quick,
  [switch]$Full,
  [int]$Seed = 999,
  [int]$Num = 0,
  [string]$Out = "",
  [switch]$NoPause
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$modeArgs = @()
if ($Full) { $modeArgs += "--full" }
elseif ($Quick) { $modeArgs += "--quick" }
elseif ($Num -gt 0) { $modeArgs += @("--num", "$Num") }
else { $modeArgs += "--quick" }  # default
if ($Seed -ne 999) { $modeArgs += @("--seed", "$Seed") }
if ($Num -gt 0 -and ($Quick -or $Full)) { $modeArgs += @("--num", "$Num") }
if ($Out -ne "") { $modeArgs += @("--out", "$Out") }

Write-Host "=== drift-sense benchmark (PowerShell) ===" -ForegroundColor Cyan
Write-Host "Repo: $PSScriptRoot"
Write-Host "Args: $modeArgs"

# Find Python: py -3.11 > python3.11 > python
$PY = $null
foreach ($c in @($env:PYTHON, $env:PYTHON311, "py", "python", "python3.11", "python3")) {
  if (-not $c) { continue }
  try {
    if ($c -eq "py") {
      $p = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
      if ($LASTEXITCODE -eq 0 -and $p) { $PY = "py"; $PY_ARGS = @("-3.11"); break }
    } else {
      & $c -c "import sys; exit(0 if sys.version_info[:2]>=(3,8) else 1)" 2>$null
      if ($LASTEXITCODE -eq 0) { $PY = $c; $PY_ARGS = @(); break }
    }
  } catch {}
}
if (-not $PY) {
  Write-Host "ERROR: no Python found. Install Python 3.11 from https://www.python.org" -ForegroundColor Red
  Write-Host "  Ensure 'py launcher' is checked during install, or set `$env:PYTHON to python.exe path." -ForegroundColor Red
  exit 1
}
if ($PY -eq "py") {
  $ver = & py -3.11 -V 2>&1
} else {
  $ver = & $PY -V 2>&1
}
Write-Host "Python: $PY $ver"
try {
  if ($PY -eq "py") { & py -3.11 -c "import sys; exit(0 if sys.version_info[:2]==(3,11) else 1)" }
  else { & $PY -c "import sys; exit(0 if sys.version_info[:2]==(3,11) else 1)" }
  if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: reference machine is Python 3.11; you have $ver -- continuing (runtime 12% off, see results/runtime_protocol.json)" -ForegroundColor Yellow
  }
} catch {}

$VENV = ".venv.benchmark"
Write-Host "Creating $VENV ..."
if (Test-Path $VENV) { Remove-Item -Recurse -Force $VENV }
if ($PY -eq "py") {
  & py -3.11 -m venv $VENV
} else {
  & $PY -m venv $VENV
}
$VENV_PY = Join-Path $VENV "Scripts\python.exe"
if (-not (Test-Path $VENV_PY)) { $VENV_PY = Join-Path $VENV "bin\python" }
& $VENV_PY -m pip install --quiet --upgrade pip
Write-Host "Installing requirements_phase2.txt ..."
& $VENV_PY -m pip install --quiet -r requirements_phase2.txt
& $VENV_PY -c "import cv2, numpy, scipy, PIL; print(f'  deps: numpy {numpy.__version__} scipy {scipy.__version__} cv2 {cv2.__version__} pillow {PIL.__version__}')"

Write-Host "Running benchmark driver: scripts/run_benchmark.py $modeArgs" -ForegroundColor Cyan
$argsList = @("scripts/run_benchmark.py") + $modeArgs
& $VENV_PY @argsList
$code = $LASTEXITCODE
if ($code -ne 0) { Write-Host "benchmark exited with code $code" -ForegroundColor Yellow }
if (-not $NoPause) { Read-Host "Done -- press Enter to close" | Out-Null }
exit $code
