"""Phase 2 registration entry point.

Reads a manifest of reference/search pairs and writes one prediction row per
pair with the exact column contract required by the evaluation harness:

    pair_id,x,y,theta,scale,found,score

This is the official Phase 2 interface. It is deliberately a thin orchestration
layer over the existing localizer; Day 1 does not change the localization
algorithm or the residual stage, it only widens the pose grid (phase2_config)
and fixes the output conventions.

Conventions (verified by tests/test_pose_conventions.py, Day 1):
  * theta is CCW-positive about the match centre. The localizer internally
    returns the negated rotation, so we output theta = -diag["theta_deg"].
  * scale is the magnification (8, 10, 12); the localizer works in `scale`
    relative to the 10x nominal, so output scale = 10.0 * diag["scale"].
   * found is decided by the trained presence model (Day 4, T1+T3) via
     decide_found(diag, REJECTION_THRESHOLD); Set C no-instance pairs are
     rejected (found = 0, pose columns zeroed, score kept).

Constraints honoured:
  * CPU only. Any --device request other than cpu is ignored; we never import
    the deep-learning framework at module import time (only numpy / opencv /
    scipy are imported).
  * Missing reference or search files raise loudly instead of being skipped, so
    dev failures are visible rather than silently scored as zero.
"""

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from drift_sense.localize import load_gray, locate, optical_config, phase2_config
from drift_sense.api import decide_found

OUTPUT_COLUMNS = ["pair_id", "x", "y", "theta", "scale", "found", "score"]

# Rejection decision threshold on the trained presence probability (Phase 2, Day 4).
# The logistic model (api.py) reaches Rejection F1 = 0.9058 on the 200-pair mixed
# set at threshold 0.5. Lower it to trade precision for recall if the operating
# point changes.
REJECTION_THRESHOLD = 0.5


def _read_manifest(path):
    """Tolerant manifest reader.

    Accepts a pair_id column plus one of the known reference columns and one of
    the known search columns. Relative paths are resolved against the manifest
    directory. Mirrors the column resolution in localize.py:read_manifest.
    """
    pairs = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SystemExit(f"manifest {path} is empty")
        keys = {k.lower().strip(): k for k in reader.fieldnames}
        rk = next((keys[k] for k in ("reference_path", "reference", "ref_path", "ref")
                   if k in keys), None)
        sk = next((keys[k] for k in ("search_path", "search", "wide_path", "wide")
                   if k in keys), None)
        if not rk or not sk:
            raise SystemExit(
                f"manifest needs reference + search columns; found {reader.fieldnames}")
        pid_key = keys.get("pair_id")
        for i, row in enumerate(reader):
            pid = (row.get(pid_key) if pid_key else None) or f"row_{i:04d}"
            ref = Path(row[rk])
            search = Path(row[sk])
            if not ref.is_absolute():
                ref = (path.parent / ref).resolve()
            if not search.is_absolute():
                search = (path.parent / search).resolve()
            pairs.append((pid, ref, search))
    if not pairs:
        raise SystemExit(f"manifest {path} contained no rows")
    return pairs


def _process(ref_path, search_path, pid):
    ref, ref_rgb = load_gray(ref_path)
    search, search_rgb = load_gray(search_path)
    cfg = optical_config() if (ref_rgb or search_rgb) else phase2_config()
    cfg.device = "cpu"
    x, y, diag, _ = locate(ref, search, cfg)
    # Verified conventions (Day 1, tests/test_pose_conventions.py):
    theta = -diag["theta_deg"]          # CCW-positive about match centre
    scale = 10.0 * diag["scale"]        # magnification units (8, 10, 12)
    score = float(diag["score"])
    # Rejection decision (Day 4, T1+T3): the trained presence model decides whether
    # a true instance underlies the peak. Set C (no-instance) pairs are rejected.
    found = decide_found(diag, REJECTION_THRESHOLD)
    # found=0 contract (Day 3): when a pair is rejected the rubric requires the
    # pose columns to be zeroed, but `score` is KEPT as diag["score"] (NOT zeroed)
    # so the calibration AUC has a continuous, monotonic ranking across present and
    # absent pairs.
    if found == 0:
        x = y = theta = scale = 0.0
    return pid, x, y, theta, scale, found, score


def main():
    ap = argparse.ArgumentParser(
        description="Phase 2 registration: produce pair_id,x,y,theta,scale,found,score")
    ap.add_argument("--input", required=True, type=Path,
                    help="manifest CSV with pair_id + reference/search columns")
    ap.add_argument("--output", required=True, type=Path,
                    help="predictions CSV to write")
    ap.add_argument("--device", default="cpu",
                    help="ignored: CPU only is the submitted path (no GPU)")
    args = ap.parse_args()

    # CPU only: never honour an accelerator request, never pull in a GPU
    # deep-learning framework.
    if args.device not in ("cpu", None):
        print(f"[register] ignoring device '{args.device}', forcing cpu",
              file=sys.stderr)

    pairs = _read_manifest(args.input)

    rows = []
    for pid, ref_path, search_path in pairs:
        if not ref_path.exists():
            raise SystemExit(f"missing reference file: {ref_path}")
        if not search_path.exists():
            raise SystemExit(f"missing search file: {search_path}")
        _, x, y, theta, scale, found, score = _process(ref_path, search_path, pid)
        rows.append({
            "pair_id": pid,
            "x": f"{x:.3f}",
            "y": f"{y:.3f}",
            "theta": f"{theta:.4f}",
            "scale": f"{scale:.4f}",
            "found": int(found),
            "score": f"{score:.4f}",
        })
        print(f"{pid} x={x:.1f} y={y:.1f} theta={theta:+.2f} scale={scale:.1f} "
              f"found={found} score={score:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
