"""Phase 2 extreme-range accuracy diagnosis (Day 2, T2) -- measurement only.

Goal: isolate whether the low pass rate at the 8..12x / +/-5 deg extremes is a
POSE-SEARCH failure (the grid/refine misses the right hypothesis) or an
APPEARANCE/TEMPLATE failure (even at the true pose the template cannot match,
i.e. the magnification cliff / blur mismatch).

For each generated pair we run two localizations:
  * free search  : phase2_config()  -> error e_free
  * oracle pose  : a MatchConfig pinned to the TRUE scale and rotation, with
                    nominal_accept_score=99 so the wide grid is never skipped and
                    prescreen_top_k=1. This removes the pose-search step entirely;
                    any remaining error is appearance/template, not search.

Output: a CSV (experiments/.../diagnose_p2_extremes/results.csv) and a printed
table of median e_free vs median e_oracle and the count where e_oracle <= 5 px,
per magnification and per rotation. No config default or locate() is changed.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from drift_sense.localize import phase2_config, MatchConfig, locate
from generate_amat_proxy import BASE_PARAMS, TIERS, generate_pair

MAGNIFICATIONS = (8.0, 9.0, 10.0, 11.0, 12.0)
ROTATIONS = (0.0, -5.0, 5.0)
TOL = 5.0


def _oracle_config(true_scale, true_rot):
    """Pin the search to the true pose; isolate appearance from search/refine.

    The wide grid holds a SINGLE hypothesis (the true scale/rotation); cfg.zoom
    stays at its default (10x) so the template is built at the correct effective
    zoom cfg.zoom*scale. refine and residual re-disambiguation are disabled so
    nothing can move the peak; the resulting error is a pure appearance/template
    measurement. With nominal_accept_score=99 the wide grid is never skipped.
    """
    return MatchConfig(
        coarse_scales=(round(true_scale, 4),),
        coarse_rotations_deg=(round(true_rot, 4),),
        wide_sigma_bank_nm=(6.5,),
        prescreen_downsample=4,
        prescreen_top_k=1,
        nominal_accept_score=99.0,   # NCC is < 1, so wide grid always runs
        refine_levels=0,
        residual_disambiguation=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--tier", default="medium", choices=list(TIERS))
    args = ap.parse_args()

    out_dir = (REPO / "experiments" /
               f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_diagnose_p2_extremes")
    out_dir.mkdir(parents=True)
    params = dict(BASE_PARAMS)
    params.update(TIERS[args.tier])

    rows = []
    t0 = time.time()
    for mag in MAGNIFICATIONS:
        for rot in ROTATIONS:
            for k in range(args.repeats):
                seed = 100000 + int(mag * 1000) + int((rot + 10) * 100) + k * 7
                ref, search, meta = generate_pair(
                    seed=seed, kind="dram", params=params,
                    rotation_deg=rot, scale=mag / 10.0, boundary_bias=1.0)
                g = meta["ground_truth"]

                xf, yf, df, _ = locate(ref, search, phase2_config())
                e_free = float(np.hypot(xf - g["x"], yf - g["y"]))

                xo, yo, dfo, _ = locate(ref, search,
                                        _oracle_config(mag / 10.0, rot))
                e_oracle = float(np.hypot(xo - g["x"], yo - g["y"]))

                rows.append({
                    "magnification": mag, "rotation_deg": rot, "repeat": k,
                    "gt_x": round(g["x"], 3), "gt_y": round(g["y"], 3),
                    "e_free_px": round(e_free, 4),
                    "e_oracle_px": round(e_oracle, 4),
                    "oracle_ok": int(e_oracle <= TOL),
                })
                print(f"mag={mag:4.1f} rot={rot:+5.1f} k={k} "
                      f"e_free={e_free:8.2f} e_oracle={e_oracle:8.2f}", flush=True)

    by_mag = {}
    for mag in MAGNIFICATIONS:
        sub = [r for r in rows if r["magnification"] == mag]
        ef = np.array([r["e_free_px"] for r in sub])
        eo = np.array([r["e_oracle_px"] for r in sub])
        ok = int(np.sum([r["oracle_ok"] for r in sub]))
        by_mag[mag] = {
            "median_e_free": round(float(np.median(ef)), 3),
            "median_e_oracle": round(float(np.median(eo)), 3),
            "oracle_le5_count": ok, "n": len(sub),
        }
    by_rot = {}
    for rot in ROTATIONS:
        sub = [r for r in rows if r["rotation_deg"] == rot]
        ef = np.array([r["e_free_px"] for r in sub])
        eo = np.array([r["e_oracle_px"] for r in sub])
        ok = int(np.sum([r["oracle_ok"] for r in sub]))
        by_rot[rot] = {
            "median_e_free": round(float(np.median(ef)), 3),
            "median_e_oracle": round(float(np.median(eo)), 3),
            "oracle_le5_count": ok, "n": len(sub),
        }

    summary = {
        "tolerance_px": TOL,
        "overall": {
            "median_e_free": round(float(np.median([r["e_free_px"] for r in rows])), 3),
            "median_e_oracle": round(float(np.median([r["e_oracle_px"] for r in rows])), 3),
            "oracle_le5_count": int(np.sum([r["oracle_ok"] for r in rows])),
            "n": len(rows),
        },
        "by_magnification": by_mag,
        "by_rotation": by_rot,
    }
    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json_dumps_safe(summary, indent=2))
    print(f"{time.time() - t0:.0f}s, wrote {out_dir}")


def json_dumps_safe(obj, indent=2):
    import json
    return json.dumps(obj, indent=indent)


if __name__ == "__main__":
    main()
