"""Evaluate the shipped pipeline on the degraded set, broken down by severity.

Set B carries 0.55 of the localization weight, gates the optical bonus through
the sets A to C clause, and is the first tie breaker in the addendum, so it is
worth measuring on its own rather than inside an aggregate. This harness runs
the same steps register.py runs, the width rescue and the presence model
included, so a configuration measured here is the configuration that ships.

Every run is serial on purpose. The localizer carries a wall clock guard that
skips its optional stages when a pair runs long, so concurrent evaluations
change each other's answers and rank configurations wrongly; that failure is
recorded in docs/phase2_failure_analysis.md. Never run two of these at once.

    .venv/bin/python scripts/eval_degraded.py --dataset data/p2holdout --label base
"""

import argparse
import collections
import csv
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion

from drift_sense.localize import MatchConfig, load_gray, locate
from drift_sense.presence import features_from_diag, presence_probability

RESCUE_PEAK_BELOW = 0.62
RESCUE_MARGIN = 0.02
RESCUE_START_BEFORE = 0.5
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "presence_model.json"


def credit(err):
    return 1.0 if err <= 1 else 0.8 if err <= 2 else 0.6 if err <= 3 else 0.4 if err <= 5 else 0.0


def build_config(args):
    cfg = MatchConfig()
    if args.psf_bank:
        cfg.psf_sigma_bank_nm = tuple(float(v) for v in args.psf_bank.split(","))
    if args.wide_bank:
        cfg.wide_sigma_bank_nm = tuple(float(v) for v in args.wide_bank.split(","))
    if args.top_k:
        cfg.prescreen_top_k = args.top_k
    if args.tone_norm:
        cfg.tone_norm = args.tone_norm
    if args.bandpass is not None:
        cfg.bandpass_sigma_px = args.bandpass
    if args.denoise_max is not None:
        cfg.denoise_sigma_max = args.denoise_max
    if args.prescreen_downsample:
        cfg.prescreen_downsample = args.prescreen_downsample
    if args.budget is not None:
        cfg.time_budget_s = args.budget
    return cfg


def run_pair(ref_path, search_path, cfg, model, rescue_below):
    t_pair = time.perf_counter()
    ref, _ = load_gray(ref_path)
    search, _ = load_gray(search_path)
    x, y, diag, _ = locate(ref, search, cfg, t_start=t_pair)
    if (float(diag["score"]) < rescue_below
            and int(diag.get("num_candidates_wide", 1)) > 1):
        for op in (grey_erosion, grey_dilation):
            if time.perf_counter() - t_pair > RESCUE_START_BEFORE * cfg.time_budget_s:
                break
            ref_cd = op(ref, size=(3, 3)).astype(ref.dtype)
            x2, y2, d2, _ = locate(ref_cd, search, cfg, t_start=t_pair)
            if float(d2["score"]) > float(diag["score"]) + RESCUE_MARGIN:
                x, y, diag = x2, y2, d2
    if model is not None:
        p = presence_probability(model, features_from_diag(diag))
        found = 1 if p >= model["prob_threshold"] else 0
    else:
        found = 1
    return x, y, found, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--psf_bank", default="")
    ap.add_argument("--wide_bank", default="")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--tone_norm", default="")
    ap.add_argument("--bandpass", type=float, default=None)
    ap.add_argument("--denoise_max", type=float, default=None)
    ap.add_argument("--prescreen_downsample", type=int, default=0)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--rescue_below", type=float, default=RESCUE_PEAK_BELOW)
    ap.add_argument("--sets", default="A_nominal,B_degraded")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", type=Path, default=None)
    # The scored machine has four cores; a development machine with more of
    # them reports a runtime the scored run will not reproduce, and the
    # efficiency component is ranked on median wall clock. Capping the thread
    # count is the closest approximation available here.
    ap.add_argument("--threads", type=int, default=0)
    # The scored hard timeout. Lower it to model a slower reference machine:
    # 13.3 stands for one half again as slow as this one.
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    if args.threads:
        cv2.setNumThreads(args.threads)

    keep = set(args.sets.split(","))
    rows = [r for r in csv.DictReader(open(args.dataset / "ground_truth.csv"))
            if r["modality"] == "sem" and r["set"] in keep]
    if args.limit:
        rows = rows[:args.limit]

    cfg = build_config(args)
    model = json.loads(MODEL_PATH.read_text()) if MODEL_PATH.exists() else None

    # A pair that overruns the scored timeout is worth nothing at all, so a
    # harness that does not charge for it will prefer a configuration that
    # buys accuracy with time it does not have. Measured on this machine the
    # difference is not academic: the per call budget scored 0.521 on the
    # degraded set ignoring the clock, 0.515 charging the twenty second limit,
    # and 0.501 charging a limit scaled for a reference machine half again as
    # slow, all from the same run.
    def scored(err, found, seconds):
        if seconds > args.timeout or not found:
            return 0.0
        return credit(err)

    per = collections.defaultdict(list)
    times, dump = [], []
    for r in rows:
        t0 = time.perf_counter()
        try:
            x, y, found, diag = run_pair(args.dataset / r["reference_path"],
                                         args.dataset / r["search_path"],
                                         cfg, model, args.rescue_below)
            err = float(np.hypot(x - float(r["gt_x"]), y - float(r["gt_y"])))
        except Exception:
            err, found, diag = float("inf"), 0, {}
        dt = time.perf_counter() - t0
        times.append(dt)
        # An absent pair carries no localization credit, so it is measured for
        # runtime only; it is also the slowest case, because a weak peak is
        # exactly what triggers the width rescue's extra passes.
        key = (r["set"], r["severity"])
        if r["found"] == "1":
            per[key].append((err, found, dt))
        else:
            per[key].append((float("nan"), -1, dt))
        dump.append({"pair_id": r["pair_id"], "set": r["set"],
                     "severity": r["severity"], "err": f"{err:.3f}",
                     "found": found, "seconds": f"{dt:.2f}",
                     "peak": f"{float(diag.get('score', 0)):.4f}"})

    print(f"\n  {args.label}")
    print(f"  {'set / severity':<18}{'n':>4}{'credit':>9}{'median err':>13}"
          f"{'rejected':>10}{'median s':>10}{'max s':>8}")
    for k in sorted(per):
        v = per[k]
        pres = [(e, f, t) for e, f, t in v if f >= 0]
        c = (sum(scored(e, f, t) for e, f, t in pres) / len(pres)) if pres else float("nan")
        acc = [e for e, f, _ in pres if f]
        print(f"  {k[0] + ' / ' + k[1]:<18}{len(v):>4}{c:>9.3f}"
              f"{(st.median(acc) if acc else float('nan')):>13.2f}"
              f"{sum(1 for e, f, _ in pres if not f):>10}"
              f"{st.median(t for _, _, t in v):>10.2f}{max(t for _, _, t in v):>8.2f}")
    agg = {}
    for s in ("A_nominal", "B_degraded"):
        v = [(e, f, t) for k, val in per.items() if k[0] == s for e, f, t in val if f >= 0]
        agg[s] = sum(scored(e, f, t) for e, f, t in v) / len(v) if v else 0.0
    weighted = 0.45 * agg["A_nominal"] + 0.55 * agg["B_degraded"]
    print(f"  {'SET A credit':<18}{agg['A_nominal']:>13.3f}")
    print(f"  {'SET B credit':<18}{agg['B_degraded']:>13.3f}")
    print(f"  {'weighted':<18}{weighted:>13.3f}   localization {weighted * 40:.2f} / 40")
    ts = sorted(times)
    print(f"  runtime  median {st.median(ts):.2f}s   p90 {ts[int(0.9 * (len(ts) - 1))]:.2f}s"
          f"   max {ts[-1]:.2f}s   over 5s {sum(1 for t in ts if t > 5) / len(ts) * 100:.0f}%",
          flush=True)
    if args.dump:
        with open(args.dump, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dump[0].keys()))
            w.writeheader()
            w.writerows(dump)


if __name__ == "__main__":
    main()
