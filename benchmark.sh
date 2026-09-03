#!/usr/bin/env bash
# One-click benchmark for drift-sense -- macOS / Linux / Git-Bash-on-Windows
# Assumes internet for pip install; no wheelhouse bundled.
# Modes: --quick (40 pairs, ~70s, default) vs --full (120 pairs, ~200s, headline)
#   ./benchmark.sh              # quick
#   ./benchmark.sh --full       # full headline
#   ./benchmark.sh --quick --seed 42 --num 20
set -euo pipefail
cd "$(dirname "$0")"

MODE_ARGS=()
for arg in "$@"; do
  MODE_ARGS+=("$arg")
done
if [ ${#MODE_ARGS[@]} -eq 0 ]; then
  MODE_ARGS=(--quick)
fi

echo "=== drift-sense benchmark ==="
echo "Repo: $(pwd)"
echo "Args: ${MODE_ARGS[*]}"

# --- find Python ---
PY=""
for c in "${PYTHON:-}" "${PYTHON311:-}" python3.11 python3 python; do
  [ -z "$c" ] && continue
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; exit(0 if sys.version_info[:2]>=(3,8) else 1)' 2>/dev/null; then
    PY=$(command -v "$c")
    break
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: no python found. Install Python 3.11 (https://www.python.org) and retry." >&2
  echo "  On Windows, use 'py -3.11' or run benchmark.ps1 / benchmark.bat instead." >&2
  exit 1
fi
echo "Python: $PY ($("$PY" -V 2>&1))"
if ! "$PY" -c 'import sys; exit(0 if sys.version_info[:2]==(3,11) else 1)'; then
  echo "WARNING: reference machine is Python 3.11; you have $("$PY" -V 2>&1) -- continuing, results comparable but runtime 12% off (see results/runtime_protocol.json)" >&2
fi

VENV=".venv.benchmark"
echo "Creating $VENV from $PY ..."
rm -rf "$VENV"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
if [ -f "$VENV/bin/activate" ]; then
  # Unix
  VENV_PY="$VENV/bin/python"
  VENV_PIP="$VENV/bin/pip"
else
  # Windows Git Bash fallback
  VENV_PY="$VENV/Scripts/python.exe"
  VENV_PIP="$VENV/Scripts/pip.exe"
fi
"$VENV_PY" -m pip install --quiet --upgrade pip
echo "Installing requirements_phase2.txt ..."
"$VENV_PY" -m pip install --quiet -r requirements_phase2.txt

# sanity
"$VENV_PY" -c "import cv2, numpy, scipy, PIL; print(f'  deps: numpy {numpy.__version__} scipy {scipy.__version__} cv2 {cv2.__version__} pillow {PIL.__version__}')"

echo "Running benchmark driver: scripts/run_benchmark.py ${MODE_ARGS[*]}"
exec "$VENV_PY" scripts/run_benchmark.py "${MODE_ARGS[@]}"
