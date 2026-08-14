"""Compare localizer configurations across every available dataset.

Ablations are only trustworthy when they are measured on all the domains at
once, because a change that helps one generator often costs another. This
script runs a named set of configurations over a named set of datasets and
writes one table with pass rates at the thresholds the organiser specification
asks for, plus runtime, into a timestamped experiment folder.

Usage:
    python scripts/compare_configs.py --datasets data/train40_v2 data/stress30 \
        data/spec40 data/amat40 --name config_ablation
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from drift_sense.localize import MatchConfig, load_gray, locate

TOLERANCES = (1.0, 2.0, 4.0, 5.0, 10.0)

CONFIGS = {
    # Round one, recorded for the log: anti aliasing and the nominal first early
    # exit were both net regressions, so point sampling is restored below.
    "baseline_loose_tol": dict(antialias=False, nominal_accept_score=9.9,
                               nominal_preference=-9.9, peak_tolerance=0.015),
    "aa_nominal_loose_tol": dict(antialias=True, peak_tolerance=0.015),
    "aa_nominal_mid_tol": dict(antialias=True, peak_tolerance=0.006),
    "aa_nominal_tight_tol": dict(antialias=True, peak_tolerance=0.002),

    # Round two, isolating the two surviving ideas with point sampling restored
    # and no early exit, so the pose grid is always searched: does requiring an
    # off nominal pose to earn its acceptance help, and does restricting the
    # equal match set to numerical ties help.
    "ps_wide_loose": dict(antialias=False, nominal_accept_score=9.9,
                          nominal_preference=-9.9, peak_tolerance=0.015),
    "ps_wide_tight": dict(antialias=False, nominal_accept_score=9.9,
                          nominal_preference=-9.9, peak_tolerance=0.003),
    "ps_prefer_loose": dict(antialias=False, nominal_accept_score=9.9,
                            nominal_preference=0.02, peak_tolerance=0.015),
    "ps_prefer_tight": dict(antialias=False, nominal_accept_score=9.9,
                            nominal_preference=0.02, peak_tolerance=0.003),

    # Round three: the current defaults, which add the scale adaptive template,
    # adaptive denoise and the extended blur banks on top of ps_prefer_tight,
    # and the same with each of the two new mechanisms disabled to attribute
    # any change.
    "defaults_v3": dict(),
    "v3_no_adaptive_template": dict(scale_adaptive_template=False),
    "v3_no_adaptive_denoise": dict(adaptive_denoise=False),
    # Tests the causal claim for the stress regression precisely: only the
    # 36 nm wide bank level is removed, keeping the 28 nm nominal addition and
    # both adaptive mechanisms, because the regression appeared in every v3
    # variant and the one change common to all of them was the wide bank
    # growing the hypothesis grid against a fixed prescreen budget.
    "v3_no_wide36": dict(wide_sigma_bank_nm=(4.0, 9.0, 16.0, 25.0)),
    # Removing only the 36 nm wide level recovered just part of the stress
    # regression, leaving one candidate common to every measured variant: the
    # 28 nm nominal addition. This reverts both banks to their original sizes
    # while keeping the two adaptive mechanisms, completing the attribution.
    "v3_banks_reverted": dict(
        psf_sigma_bank_nm=(2.0, 4.0, 6.5, 9.0, 14.0, 20.0),
        wide_sigma_bank_nm=(4.0, 9.0, 16.0, 25.0)),

    # Round four, adaptive prescreen budget. The attribution round showed the
    # wide grid competing 176 hypotheses for six full resolution slots and
    # implicated candidate survival in the stress regression; this asks
    # whether six was undersized all along. Prediction registered before the
    # data: stress improves through candidate survival on the wide path,
    # physics and amat are untouched because they resolve on the nominal
    # path, runtime rises by under 0.2 s, and the 9.0 to 1 pose boundary may
    # recover as a side effect because endpoint hypotheses rank low at half
    # resolution for the same reason.
    "k12": dict(prescreen_top_k=12),
    "k24": dict(prescreen_top_k=24),
}


def run(dataset, cfg_kwargs):
    pairs = sorted(d for d in dataset.iterdir()
                   if d.is_dir() and d.name.startswith("pair_"))
    errs, times, poses = [], [], []
    for pd in pairs:
        meta = json.loads((pd / "meta.json").read_text())
        ref, _ = load_gray(pd / "reference.png")
        search, _ = load_gray(pd / "search.png")
        x, y, diag, _ = locate(ref, search, MatchConfig(**cfg_kwargs))
        g = meta["ground_truth"]
        errs.append(float(np.hypot(x - g["x"], y - g["y"])))
        times.append(diag["runtime_s"])
        poses.append(diag["pose_source"])
    e, t = np.array(errs), np.array(times)
    row = {"n": len(e), "median_px": round(float(np.median(e)), 3),
           "mean_px": round(float(e.mean()), 2),
           "worst_px": round(float(e.max()), 1),
           "mean_runtime_s": round(float(t.mean()), 2),
           "wide_grid_used": poses.count("wide_grid")}
    for tol in TOLERANCES:
        row[f"within_{tol}px_pct"] = round(float(100.0 * (e <= tol).mean()), 1)
    return row, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--name", default="config_ablation")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "experiments" / f"{stamp}_{args.name}"
    out_dir.mkdir(parents=True)
    table, raw = [], {}
    t0 = time.time()
    for cfg_name in args.configs:
        for dataset in args.datasets:
            if not dataset.exists():
                print(f"skip missing {dataset}")
                continue
            row, errs = run(dataset, CONFIGS[cfg_name])
            row = {"config": cfg_name, "dataset": dataset.name, **row}
            table.append(row)
            raw[f"{cfg_name}|{dataset.name}"] = errs
            print(f"{cfg_name:22s} {dataset.name:14s} "
                  f"w1={row['within_1.0px_pct']:5.1f}% w5={row['within_5.0px_pct']:5.1f}% "
                  f"med={row['median_px']:8.3f} t={row['mean_runtime_s']:.2f}s "
                  f"wide={row['wide_grid_used']}", flush=True)

    with open(out_dir / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    with open(out_dir / "raw_errors.json", "w") as fh:
        json.dump(raw, fh)
    print(f"\n{time.time() - t0:.0f}s total, wrote {out_dir}")


if __name__ == "__main__":
    main()
