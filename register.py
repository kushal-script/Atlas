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
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from drift_sense.api import load_presence_model, register_pair
from drift_sense.localize import MatchConfig, load_gray, optical_config

PAIR_HARD_TIMEOUT_S = 18.0
_HAS_ALARM = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


class _PairTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _PairTimeout()


MODEL_PATH = Path(__file__).resolve().parent / "models" / "presence_model.json"


def _load_model():
    """The fitted presence decision, or None to fall back to a peak threshold.

    The fallback is much weaker, reject class F1 about 0.46 against 0.655, so a
    silent fall back would quietly cost points with nothing in the output to
    say why. Any failure to load is therefore reported on stderr, which does
    not disturb the predictions file.
    """
    try:
        return load_presence_model(MODEL_PATH)
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
            pid = (row.get(keys.get("pair_id", ""), "") or f"row_{i:04d}").strip()
            try:
                ref = Path((row[rk] or "").strip())
                search = Path((row[sk] or "").strip())
                if str(ref) and not ref.is_absolute():
                    ref = (path.parent / ref).resolve()
                if str(search) and not search.is_absolute():
                    search = (path.parent / search).resolve()
            except (TypeError, ValueError, OSError):
                ref = search = Path("")
            pairs.append((pid, ref, search))
    return pairs


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
    try:
        pairs = list(_read_pairs(args.input))
    except OSError as exc:
        print(f"cannot read {args.input}: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
    if args.output.parent and not args.output.parent.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
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
            result = register_pair(ref, search, reference_rgb=ref_rgb,
                                   search_rgb=search_rgb, model=model, config=cfg,
                                   optical=cfg_optical, t_start=t_pair)
            if result.diagnostics.get("budget_gated"):
                print(f"WARNING: {pid} hit the wall clock budget and skipped "
                      f"optional stages", file=sys.stderr, flush=True)
            emit(result.as_row(pid))
        except _PairTimeout:
            print(f"WARNING: {pid} exceeded {PAIR_HARD_TIMEOUT_S:.0f}s and was "
                  f"reported absent", file=sys.stderr, flush=True)
            emit({"pair_id": pid, "x": 0, "y": 0, "theta": 0,
                         "scale": 0, "found": 0, "score": "0.00000"})
        except Exception as exc:
            print(f"WARNING: {pid} failed ({type(exc).__name__}: {exc}) and was "
                  f"reported absent", file=sys.stderr, flush=True)
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
