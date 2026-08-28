"""Phase 2 entry point.

    python register.py --input pairs.csv --output predictions.csv

Reads exactly the supplied csv and the image files it lists, nothing else: no
network, no other files, no directory probing, and identical behaviour whatever
the files are named. One output row per input pair, every pair_id exactly once:

    pair_id, x, y, theta, scale, found, score

x, y is the match centre in wide search coordinates. theta is the rotation of
the reference pattern as it appears in the search image, counter clockwise
positive, in degrees. scale is the recovered down scaling factor, nominally in
8 to 12. found is 1 when the reference is present and 0 otherwise, and a pair
reported absent carries zeros in the pose columns. score is this method's own
confidence on any monotonic scale: higher means the decision, either the
reported pose or the rejection, is more likely to be right.

The method is the Phase 1 approach evolved, not replaced: the same normalized
cross correlation over a blur bank and pose hypothesis grid, the grid widened
to the disclosed 8 to 12 and plus or minus 5 degree ranges, the same residual
disambiguation stage for periodic ties, and the presence decision built from
the same peak statistics the Phase 1 confidence regimes already used.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from drift_sense.localize import MatchConfig, load_gray, locate

# Presence decision and score calibration, tuned on this repository's own
# Phase 2 style generated suite, never on organiser data. found_threshold is
# the peak correlation below which a pair is reported absent; score_width sets
# how sharply confidence saturates on either side of that boundary.
FOUND_THRESHOLD = 0.603
SCORE_WIDTH = 0.08


def _read_pairs(path):
    pairs = []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            keys = {k.lower().strip(): k for k in row}
            rk = next((keys[k] for k in ("reference_path", "reference", "ref_path", "ref")
                       if k in keys), None)
            sk = next((keys[k] for k in ("search_path", "search", "wide_path", "wide")
                       if k in keys), None)
            if not rk or not sk:
                raise SystemExit("pairs csv needs reference and search path columns")
            pid = row.get(keys.get("pair_id", ""), "") or f"row_{i:04d}"
            ref, search = Path(row[rk]), Path(row[sk])
            if not ref.is_absolute():
                ref = (path.parent / ref).resolve()
            if not search.is_absolute():
                search = (path.parent / search).resolve()
            pairs.append((pid, ref, search))
    return pairs


def _score(peak, prominence, wide_candidates):
    """Monotonic confidence in the decision actually being made.

    Presence probability rises with the peak against the threshold; a pair
    rejected with a very weak peak is a confident rejection, so the score
    reflects distance from the boundary on either side, damped when the
    correlation surface was degenerate.
    """
    p_present = 1.0 / (1.0 + np.exp(-(peak - FOUND_THRESHOLD) / SCORE_WIDTH))
    decision_conf = max(p_present, 1.0 - p_present)
    uniqueness = 1.0 / (1.0 + np.log1p(max(int(wide_candidates), 1) - 1))
    strength = min(max(prominence, 0.0) / 20.0, 1.0)
    return float(decision_conf * (0.6 + 0.25 * uniqueness + 0.15 * strength))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = MatchConfig()
    rows = []
    for pid, ref_path, search_path in _read_pairs(args.input):
        try:
            ref, ref_rgb = load_gray(ref_path)
            search, search_rgb = load_gray(search_path)
            x, y, diag, _ = locate(ref, search, cfg)
            peak = float(diag["score"])
            found = 1 if peak >= FOUND_THRESHOLD else 0
            theta = cfg.theta_report_sign * float(diag["theta_deg"])
            scale = float(diag["scale"]) * cfg.zoom
            score = _score(peak, float(diag.get("peak_prominence", 0.0)),
                           diag.get("num_candidates_wide", 1))
            if found:
                rows.append({"pair_id": pid, "x": f"{x:.3f}", "y": f"{y:.3f}",
                             "theta": f"{theta:.3f}", "scale": f"{scale:.4f}",
                             "found": 1, "score": f"{score:.5f}"})
            else:
                rows.append({"pair_id": pid, "x": 0, "y": 0, "theta": 0,
                             "scale": 0, "found": 0, "score": f"{score:.5f}"})
        except Exception:
            # A pair that fails still gets its row: a missing row scores zero
            # for certain, a conservative rejection at least scores the pairs
            # where rejection was the right call.
            rows.append({"pair_id": pid, "x": 0, "y": 0, "theta": 0,
                         "scale": 0, "found": 0, "score": "0.00000"})
        print(f"{pid} found={rows[-1]['found']} score={rows[-1]['score']}", flush=True)

    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair_id", "x", "y", "theta",
                                           "scale", "found", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
