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
from drift_sense.presence import features_from_diag, presence_probability
from scipy.ndimage import grey_dilation, grey_erosion

# Width rescue. Polygon scaling changes feature widths between the two
# captures, so when the first pass shows the mislock signature, a weak peak on
# a degenerate surface, the reference is re matched under a small morphological
# width change and the higher scoring answer is kept. Two variants rather than
# four: measured on a held out suite the pass is worth about one point of
# localization, and every extra variant is another full pass over the pose
# grid, so the cheapest form that keeps the gain is the one that ships.
# The scored run gives every pair a hard twenty second timeout and a pair that
# overruns scores zero. The internal budget gates optional stages but cannot
# abort work already running, so a wall clock alarm is armed per pair as the
# last line of defence: it fires below the limit, unwinds into the handler that
# already writes a conservative row, and leaves the remaining pairs to run. On
# a platform without SIGALRM the alarm is simply absent and the internal budget
# is all there is, which is the behaviour this file had before.
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
# Fraction of the pair's budget past which no further rescue pass is started.
RESCUE_START_BEFORE = 0.5

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
    # utf-8-sig, not the platform default: a pairs csv saved from a spreadsheet
    # carries a byte order mark that stays glued to the first header name, and
    # since it is not whitespace it survives strip(). The pair_id column then
    # goes unrecognised and every output row is named row_0000 onward, which
    # joins to nothing and scores zero on a run that otherwise looks perfect.
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            # A data row carrying more fields than the header puts the surplus
            # under DictReader's restkey, which is None, and a row carrying
            # fewer leaves a None value. Both arrive from a manifest a person
            # edited by hand or joined without quoting, and neither is a reason
            # to abandon the run: the None key is dropped here so the row is
            # read on its recognised columns alone, because raising instead
            # would escape this loop before the per row handler below and cost
            # every pair rather than the one that is malformed.
            keys = {k.lower().strip(): k for k in row if isinstance(k, str)}
            rk = next((keys[k] for k in ("reference_path", "reference", "ref_path", "ref")
                       if k in keys), None)
            sk = next((keys[k] for k in ("search_path", "search", "wide_path", "wide")
                       if k in keys), None)
            if not rk or not sk:
                # A manifest whose path columns cannot be recognised used to
                # end the process before a single row was written, which turns
                # one unreadable header into a zero on every pair. The run
                # continues instead: each row still gets an id and a pair of
                # paths that will fail to load, which the per pair handler
                # turns into a conservative rejection, so the absent pairs
                # still score and the failure is reported rather than fatal.
                if i == 0:
                    print("WARNING: pairs csv has no recognised reference and "
                          "search path columns; every row will be reported "
                          f"absent. Columns seen: {sorted(str(k) for k in row)}",
                          file=sys.stderr, flush=True)
                pid = row.get(keys.get("pair_id", ""), "") or f"row_{i:04d}"
                pairs.append((pid, Path(""), Path("")))
                continue
            pid = row.get(keys.get("pair_id", ""), "") or f"row_{i:04d}"
            # One unreadable row must not cost the other hundred and ninety
            # nine. A row whose paths are missing or malformed still gets a
            # pair id and a pair of paths that will fail to load, which the
            # per pair handler turns into a conservative rejection; raising
            # here instead would abandon the whole run and every row with it.
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

    # The reference machine runs 3.11. A different interpreter is reported and
    # then tolerated rather than refused, because aborting the scored run would
    # forfeit every pair to defend against a difference that may not matter.
    if sys.version_info[:2] != (3, 11):
        print(f"warning: running on Python {sys.version_info.major}."
              f"{sys.version_info.minor}, the reference machine is 3.11",
              file=sys.stderr)

    cfg = MatchConfig()
    cfg_optical = optical_config()
    model = _load_model()
    rows = []
    if _HAS_ALARM:
        signal.signal(signal.SIGALRM, _on_alarm)
    for pid, ref_path, search_path in _read_pairs(args.input):
        try:
            t_pair = time.perf_counter()
            if _HAS_ALARM:
                signal.setitimer(signal.ITIMER_REAL, PAIR_HARD_TIMEOUT_S)
            ref, ref_rgb = load_gray(ref_path)
            search, search_rgb = load_gray(search_path)
            active_cfg = cfg_optical if (ref_rgb or search_rgb) else cfg
            x, y, diag, _ = locate(ref, search, active_cfg, t_start=t_pair)
            if (float(diag["score"]) < RESCUE_PEAK_BELOW
                    and int(diag.get("num_candidates_wide", 1)) > 1
                    and not (ref_rgb or search_rgb)):
                for op in (grey_erosion, grey_dilation):
                    # A rescue pass the remaining budget cannot cover is not
                    # started at all. The rescue fires on a weak peak, and a
                    # weak peak is what a heavily degraded pair produces, so
                    # without this the passes pile onto exactly the pairs that
                    # are already closest to the scored timeout.
                    if time.perf_counter() - t_pair > RESCUE_START_BEFORE * active_cfg.time_budget_s:
                        break
                    ref_cd = op(ref, size=(3, 3)).astype(ref.dtype)
                    x2, y2, d2, _ = locate(ref_cd, search, active_cfg, t_start=t_pair)
                    # Refusing an override from a pass the clock cut short was
                    # tried here and measured worse, costing the degraded set
                    # 0.515 against 0.508 while changing the holdout not at
                    # all: a starved pass searches fewer poses but is often
                    # right anyway, and declining it keeps a worse answer. The
                    # diagnostic that guard read is still recorded.
                    if float(d2["score"]) > float(diag["score"]) + RESCUE_MARGIN:
                        x, y, diag = x2, y2, d2
            peak = float(diag["score"])
            if model is not None:
                p_present = presence_probability(model, features_from_diag(diag))
                found = 1 if p_present >= model["prob_threshold"] else 0
                score = float(max(p_present, 1.0 - p_present))
                if ref_rgb or search_rgb:
                    # The bonus set is disclosed as reference present, and its
                    # rejection is never scored: the F1 runs over the grayscale
                    # pairs only, so a rejected optical pair can only forfeit
                    # bonus credit, never earn anything. The disclosed fact is
                    # used the way the disclosed pose bounds are, and the score
                    # becomes the model's probability itself, which is the
                    # confidence that the forced answer is right.
                    found = 1
                    score = float(p_present)
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
            x, y = _finite(x), _finite(y)
            theta, scale = _finite(theta), _finite(scale, active_cfg.zoom)
            score = _finite(score)
            if found:
                rows.append({"pair_id": pid, "x": f"{x:.3f}", "y": f"{y:.3f}",
                             "theta": f"{theta:.3f}", "scale": f"{scale:.4f}",
                             "found": 1, "score": f"{score:.5f}"})
            else:
                rows.append({"pair_id": pid, "x": 0, "y": 0, "theta": 0,
                             "scale": 0, "found": 0, "score": f"{score:.5f}"})
        except _PairTimeout:
            # The pair reached the wall clock limit. Its row is written and the
            # loop moves on, so one slow pair costs one pair rather than every
            # pair that would have followed it.
            print(f"WARNING: {pid} exceeded {PAIR_HARD_TIMEOUT_S:.0f}s and was "
                  f"reported absent", file=sys.stderr, flush=True)
            rows.append({"pair_id": pid, "x": 0, "y": 0, "theta": 0,
                         "scale": 0, "found": 0, "score": "0.00000"})
        except Exception:
            # A pair that fails still gets its row: a missing row scores zero
            # for certain, a conservative rejection at least scores the pairs
            # where rejection was the right call.
            rows.append({"pair_id": pid, "x": 0, "y": 0, "theta": 0,
                         "scale": 0, "found": 0, "score": "0.00000"})
        finally:
            if _HAS_ALARM:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
        print(f"{pid} found={rows[-1]['found']} score={rows[-1]['score']}", flush=True)

    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair_id", "x", "y", "theta",
                                           "scale", "found", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
