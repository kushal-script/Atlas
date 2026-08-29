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
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from drift_sense.localize import MatchConfig, load_gray, locate, optical_config
from drift_sense.presence import features_from_diag, presence_probability
from scipy.ndimage import grey_dilation, grey_erosion

# Polygon scaling is one of the disclosed degradations: the search capture's
# feature widths can differ from the reference by up to twenty percent, which
# decorrelates edge dominated correlation. When the first pass shows the
# mislock signature, a weak peak on a degenerate surface that is still called
# present, the reference is re matched under small morphological width
# changes, the imaging analogue of a CD offset, and the strongest answer
# wins. Probing severe pairs rescued three of eleven mislocks this way. The
# rescue is skipped once the pair's wall clock budget is nearly spent, since
# an overrun scores zero for certain.
RESCUE_PEAK_BELOW = 0.62
RESCUE_MARGIN = 0.02
RESCUE_DEADLINE_S = 10.0

# The presence decision. A single peak threshold cannot separate a degraded
# present pair from a clean absent one, because an impostor reference from the
# same architecture lets the scale search rescale its lattice onto the search
# lattice and correlate respectably. The decision is therefore a small logistic
# model over the diagnostics the localizer already produces, fitted on this
# repository's own generated suite, never on organiser data, and shipped with
# the submission as models/presence_model.json. The fallback threshold below is
# used only if that file is somehow missing.
MODEL_PATH = Path(__file__).resolve().parent / "models" / "presence_model.json"
FOUND_THRESHOLD = 0.55
SCORE_WIDTH = 0.08


def _load_model():
    try:
        return json.load(open(MODEL_PATH))
    except Exception:
        return None


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
    cfg_optical = optical_config()
    model = _load_model()
    rows = []
    for pid, ref_path, search_path in _read_pairs(args.input):
        try:
            ref, ref_rgb = load_gray(ref_path)
            search, search_rgb = load_gray(search_path)
            active_cfg = cfg_optical if (ref_rgb or search_rgb) else cfg
            t_pair = time.perf_counter()
            x, y, diag, _ = locate(ref, search, active_cfg)
            if (float(diag["score"]) < RESCUE_PEAK_BELOW
                    and int(diag.get("num_candidates_wide", 1)) > 1
                    and not (ref_rgb or search_rgb)):
                # strongest variants first, so a slow machine that hits the
                # deadline after one variant still ran the most valuable one
                for op, k in ((grey_erosion, 3), (grey_dilation, 3),
                              (grey_erosion, 2), (grey_dilation, 2)):
                    if time.perf_counter() - t_pair > RESCUE_DEADLINE_S:
                        break
                    ref_cd = op(ref, size=(k, k)).astype(ref.dtype)
                    x2, y2, d2, _ = locate(ref_cd, search, active_cfg)
                    if float(d2["score"]) > float(diag["score"]) + RESCUE_MARGIN:
                        x, y, diag = x2, y2, d2
            peak = float(diag["score"])
            if model is not None:
                p_present = presence_probability(model, features_from_diag(diag))
                found = 1 if p_present >= model["prob_threshold"] else 0
                score = float(max(p_present, 1.0 - p_present))
                if found:
                    # A found row's confidence also reflects whether the four
                    # template quadrants agreed on the site, which tracks
                    # localization correctness; measured on the training suite
                    # this damping lifts the calibration auc.
                    agree = max(int(diag.get("quad_agree", -1)), 0)
                    score *= 0.5 + 0.5 * min(agree / 4.0, 1.0)
            else:
                found = 1 if peak >= FOUND_THRESHOLD else 0
                score = _score(peak, float(diag.get("peak_prominence", 0.0)),
                               diag.get("num_candidates_wide", 1))
            theta = active_cfg.theta_report_sign * float(diag["theta_deg"])
            scale = float(diag["scale"]) * active_cfg.zoom
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
