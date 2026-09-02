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
import math
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from drift_sense.localize import MatchConfig, load_gray, locate, optical_config
from drift_sense.presence import features_for_model, presence_probability
from scipy.ndimage import grey_dilation, grey_erosion

PAIR_HARD_TIMEOUT_S = 18.0
_HAS_ALARM = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


class _PairTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _PairTimeout()


def _finite(v, fallback=0.0):
    """Never write nan or inf into the predictions file.

    A degenerate pair, a constant image or an all zero variance window can put
    a nan through normalized correlation, and the row formatting would render
    it as the literal text nan, which is not a number the scorer can read and
    may cost more than the pair itself.
    """
    v = float(v)
    return v if math.isfinite(v) else float(fallback)


RESCUE_PEAK_BELOW = 0.62
RESCUE_MARGIN = 0.02
RESCUE_START_BEFORE = 0.5

MODEL_PATH = Path(__file__).resolve().parent / "models" / "presence_model.json"
FALLBACK_FOUND_THRESHOLD = 0.55
FALLBACK_SCORE_WIDTH = 0.08


def _load_model():
    """The fitted presence decision, or None to fall back to a peak threshold.

    The fallback is much weaker, reject class F1 about 0.46 against 0.655, so a
    silent fall back would quietly cost points with nothing in the output to
    say why. Any failure to load is therefore reported on stderr, which does
    not disturb the predictions file.
    """
    try:
        model = json.load(open(MODEL_PATH))
        if len(model.get("weights", [])) != len(model.get("features", [])):
            raise ValueError("weights and features disagree in length")
        return model
    except Exception as exc:
        print(f"WARNING: presence model at {MODEL_PATH} unusable ({exc}); "
              f"falling back to the peak threshold rule, which scores worse",
              file=sys.stderr, flush=True)
        return None


def _read_pairs(path):
    pairs = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            keys = {k.lower().strip(): k for k in row if isinstance(k, str)}
            rk = next((keys[k] for k in ("reference_path", "reference", "ref_path", "ref")
                       if k in keys), None)
            sk = next((keys[k] for k in ("search_path", "search", "wide_path", "wide")
                       if k in keys), None)
            if not rk or not sk:
                if i == 0:
                    print("WARNING: pairs csv has no recognised reference and "
                          "search path columns; every row will be reported "
                          f"absent. Columns seen: {sorted(str(k) for k in row)}",
                          file=sys.stderr, flush=True)
                pid = row.get(keys.get("pair_id", ""), "") or f"row_{i:04d}"
                pairs.append((pid, Path(""), Path("")))
                continue
            pid = row.get(keys.get("pair_id", ""), "") or f"row_{i:04d}"
            try:
                ref, search = Path(row[rk] or ""), Path(row[sk] or "")
                if str(ref) and not ref.is_absolute():
                    ref = (path.parent / ref).resolve()
                if str(search) and not search.is_absolute():
                    search = (path.parent / search).resolve()
            except (TypeError, ValueError, OSError):
                ref = search = Path("")
            pairs.append((pid, ref, search))
    return pairs


def _score(peak, prominence, wide_candidates):
    """Monotonic confidence in the decision actually being made.

    Presence probability rises with the peak against the threshold; a pair
    rejected with a very weak peak is a confident rejection, so the score
    reflects distance from the boundary on either side, damped when the
    correlation surface was degenerate.
    """
    p_present = 1.0 / (1.0 + np.exp(-(peak - FALLBACK_FOUND_THRESHOLD) / FALLBACK_SCORE_WIDTH))
    decision_conf = max(p_present, 1.0 - p_present)
    uniqueness = 1.0 / (1.0 + np.log1p(max(int(wide_candidates), 1) - 1))
    strength = min(max(prominence, 0.0) / 20.0, 1.0)
    return float(decision_conf * (0.6 + 0.25 * uniqueness + 0.15 * strength))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if sys.version_info[:2] != (3, 11):
        print(f"warning: running on Python {sys.version_info.major}."
              f"{sys.version_info.minor}, the reference machine is 3.11",
              file=sys.stderr)

    cfg = MatchConfig()
    cfg_optical = optical_config()
    model = _load_model()
    rows = []
    pairs = list(_read_pairs(args.input))
    out_fh = open(args.output, "w", newline="")
    writer = csv.DictWriter(out_fh, fieldnames=["pair_id", "x", "y", "theta",
                                                "scale", "found", "score"])
    writer.writeheader()
    out_fh.flush()

    def emit(row):
        rows.append(row)
        writer.writerow(row)
        out_fh.flush()

    if _HAS_ALARM:
        signal.signal(signal.SIGALRM, _on_alarm)
    for pid, ref_path, search_path in pairs:
        try:
            t_pair = time.perf_counter()
            if _HAS_ALARM:
                signal.setitimer(signal.ITIMER_REAL, PAIR_HARD_TIMEOUT_S)
            ref, ref_rgb = load_gray(ref_path)
            search, search_rgb = load_gray(search_path)
            if float(np.std(ref)) < 1.0 or float(np.std(search)) < 1.0:
                emit({"pair_id": pid, "x": 0, "y": 0, "theta": 0,
                             "scale": 0, "found": 0, "score": "0.50000"})
                continue
            active_cfg = cfg_optical if (ref_rgb or search_rgb) else cfg
            x, y, diag, _ = locate(ref, search, active_cfg, t_start=t_pair)
            if (float(diag["score"]) < RESCUE_PEAK_BELOW
                    and int(diag.get("num_candidates_wide", 1)) > 1
                    and not (ref_rgb or search_rgb)):
                for op in (grey_erosion, grey_dilation):
                    if time.perf_counter() - t_pair > RESCUE_START_BEFORE * active_cfg.time_budget_s:
                        break
                    ref_cd = op(ref, size=(3, 3)).astype(ref.dtype)
                    x2, y2, d2, _ = locate(ref_cd, search, active_cfg, t_start=t_pair)
                    if float(d2["score"]) > float(diag["score"]) + RESCUE_MARGIN:
                        x, y, diag = x2, y2, d2
            peak = float(diag["score"])
            if model is not None:
                p_present = presence_probability(model, features_for_model(model, diag))
                found = 1 if p_present >= model["prob_threshold"] else 0
                score = float(max(p_present, 1.0 - p_present))
                if ref_rgb or search_rgb:
                    found = 1
                    score = float(p_present)
                if found and not (ref_rgb or search_rgb):
                    agree = max(int(diag.get("quad_agree", -1)), 0)
                    score *= 0.5 + 0.5 * min(agree / 4.0, 1.0)
            else:
                found = 1 if peak >= FALLBACK_FOUND_THRESHOLD else 0
                score = _score(peak, float(diag.get("peak_prominence", 0.0)),
                               diag.get("num_candidates_wide", 1))
            theta = active_cfg.theta_report_sign * float(diag["theta_deg"])
            scale = float(diag["scale"]) * active_cfg.zoom
            x, y = _finite(x), _finite(y)
            theta, scale = _finite(theta), _finite(scale, active_cfg.zoom)
            score = _finite(score)
            if found:
                emit({"pair_id": pid, "x": f"{x:.3f}", "y": f"{y:.3f}",
                             "theta": f"{theta:.3f}", "scale": f"{scale:.4f}",
                             "found": 1, "score": f"{score:.5f}"})
            else:
                emit({"pair_id": pid, "x": 0, "y": 0, "theta": 0,
                             "scale": 0, "found": 0, "score": f"{score:.5f}"})
        except _PairTimeout:
            print(f"WARNING: {pid} exceeded {PAIR_HARD_TIMEOUT_S:.0f}s and was "
                  f"reported absent", file=sys.stderr, flush=True)
            emit({"pair_id": pid, "x": 0, "y": 0, "theta": 0,
                         "scale": 0, "found": 0, "score": "0.00000"})
        except Exception:
            emit({"pair_id": pid, "x": 0, "y": 0, "theta": 0,
                         "scale": 0, "found": 0, "score": "0.00000"})
        finally:
            if _HAS_ALARM:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
        print(f"{pid} found={rows[-1]['found']} score={rows[-1]['score']}", flush=True)

    written = {r["pair_id"] for r in rows}
    for pid, _, _ in pairs:
        if pid not in written:
            emit({"pair_id": pid, "x": 0, "y": 0, "theta": 0,
                  "scale": 0, "found": 0, "score": "0.00000"})
            written.add(pid)
    out_fh.close()
    print(f"wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
