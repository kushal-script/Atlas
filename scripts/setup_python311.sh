#!/usr/bin/env bash
# Build the project interpreter. Python 3.11 is the reference machine version
# and the only version this project is developed, measured or scored on.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}

PY=${PYTHON311:-}
if [ -z "$PY" ]; then
  for c in /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 python3.11; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys;exit(sys.version_info[:2]!=(3,11))'; then
      PY=$(command -v "$c"); break
    fi
  done
fi
[ -n "$PY" ] || { echo "no Python 3.11 found; set PYTHON311 to one" >&2; exit 1; }

echo "building $VENV from $PY ($("$PY" -V 2>&1))"
rm -rf "$VENV"
"$PY" -m venv "$VENV"
"$VENV"/bin/python -m pip install --quiet --upgrade pip
"$VENV"/bin/python -m pip install --quiet -r requirements_phase2.txt
"$VENV"/bin/python -m pip install --quiet matplotlib pytest

"$VENV"/bin/python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 11), sys.version
sys.path.insert(0, "src")
import drift_sense.localize, drift_sense.presence
import cv2, numpy, scipy, PIL
print(f"  python {sys.version.split()[0]}  numpy {numpy.__version__}  "
      f"cv2 {cv2.__version__}  scipy {scipy.__version__}  pillow {PIL.__version__}")
print("  shipped modules import cleanly")
PY
echo "done. use $VENV/bin/python for everything in this repository."
