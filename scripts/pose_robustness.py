"""Explicit pose robustness grid over magnification and rotation.

The specification states that robustness testing may span roughly 9 to 1
through 11 to 1 magnification and about 1 to 2 degrees of rotation. Randomised
datasets only sample that space thinly, so this generates a deliberate grid at
exact magnification and rotation values and reports the pass rate at each, which
answers the requirement directly rather than by inference.

Experimental control that matters: every reference site is placed straddling a
mat boundary, so the content is identifiable by construction. Without that, a
failure could be caused either by the pose or by the site being in a periodic
region where localization is ill posed, and the two would be inseparable. This
grid therefore isolates pose sensitivity.

Usage:
    python scripts/pose_robustness.py --repeats 2 --name pose_robustness
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from drift_sense.api import match_pair
from drift_sense.localize import phase2_config
from generate_amat_proxy import BASE_PARAMS, TIERS, generate_pair

# Extended to the Phase 2 ranges (Day 1, T4): zoom 8..12x and rotation +/-5 deg.
MAGNIFICATIONS = (8.0, 9.0, 10.0, 11.0, 12.0)
ROTATIONS = (-5.0, -2.5, 0.0, 2.5, 5.0)
TOL = 5.0
ACCENT = "#2a78d6"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#ececea", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#5a5a55", labelsize=9)


def _marginal(rows, key, values):
    out = []
    for v in values:
        sub = [r for r in rows if abs(r[key] - v) < 1e-6]
        if not sub:
            continue
        e = np.array([r["err_px"] for r in sub])
        out.append({"value": v, "n": len(sub),
                    "pass_rate_pct": round(float(100 * (e <= TOL).mean()), 1),
                    "median_px": round(float(np.median(e)), 3),
                    "mean_px": round(float(e.mean()), 2),
                    "worst_px": round(float(e.max()), 1)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2,
                    help="pairs per magnification and rotation combination")
    ap.add_argument("--tier", default="medium", choices=list(TIERS))
    ap.add_argument("--name", default="pose_robustness")
    ap.add_argument("--top_k", type=int, default=None,
                    help="override the prescreen budget for survival experiments")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "experiments" / f"{stamp}_{args.name}"
    (out_dir / "plots").mkdir(parents=True)

    params = dict(BASE_PARAMS)
    params.update(TIERS[args.tier])
    rows = []
    t0 = time.time()
    seed = 0
    for mag in MAGNIFICATIONS:
        for rot in ROTATIONS:
            for k in range(args.repeats):
                seed += 1
                kind = "dram" if (seed % 2 == 0) else "finfet"
                ref, search, meta = generate_pair(
                    seed=90000 + seed * 31, kind=kind, params=params,
                    rotation_deg=rot, scale=mag / 10.0, boundary_bias=1.0)
                cfg_kwargs = ({"prescreen_top_k": args.top_k}
                               if args.top_k else {})
                res = match_pair(ref, search,
                                  cfg=phase2_config(**cfg_kwargs) if cfg_kwargs else phase2_config())
                g = meta["ground_truth"]
                err = float(np.hypot(res["x"] - g["x"], res["y"] - g["y"]))
                rows.append({"magnification": mag, "rotation_deg": rot,
                             "repeat": k, "style": kind,
                             "gt_x": round(g["x"], 3), "gt_y": round(g["y"], 3),
                             "pred_x": round(res["x"], 3),
                             "pred_y": round(res["y"], 3),
                             "err_px": round(err, 4),
                             "est_scale": round(res["scale"], 4),
                             "est_rotation_deg": round(res["rotation_deg"], 3),
                             "confidence": round(res["score"], 4),
                             "confidence_regime": res["confidence_regime"],
                             "runtime_s": round(res["runtime_s"], 3)})
                print(f"mag={mag:4.1f}:1 rot={rot:+.1f}deg {kind:6s} "
                      f"err={err:8.2f}px est_scale={res['scale']:.3f} "
                      f"est_rot={res['rotation_deg']:+.2f}", flush=True)

    by_mag = _marginal(rows, "magnification", MAGNIFICATIONS)
    by_rot = _marginal(rows, "rotation_deg", ROTATIONS)
    e = np.array([r["err_px"] for r in rows])
    summary = {
        "tolerance_px": TOL, "tier": args.tier, "n_pairs": len(rows),
        "boundary_bias": 1.0,
        "note": "every site straddles a boundary so content is identifiable, "
                "isolating pose sensitivity from periodic ambiguity",
        "overall": {"pass_rate_pct": round(float(100 * (e <= TOL).mean()), 1),
                    "median_px": round(float(np.median(e)), 3),
                    "mean_px": round(float(e.mean()), 2),
                    "worst_px": round(float(e.max()), 1)},
        "by_magnification": by_mag, "by_rotation": by_rot,
    }

    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    for block, key, label in ((by_mag, "value", "magnification, ratio to 1"),
                              (by_rot, "value", "rotation, degrees")):
        fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=150)
        ax.plot([b[key] for b in block], [b["pass_rate_pct"] for b in block],
                marker="o", linewidth=2, color=ACCENT)
        ax.set_xlabel(label)
        ax.set_ylabel(f"pass rate within {TOL:.0f} px, %")
        ax.set_ylim(-5, 105)
        ax.set_title(f"Pose robustness versus {label.split(',')[0]}",
                     fontsize=10, color="#1f1f1d")
        _style(ax)
        fig.tight_layout()
        tag = "magnification" if "magnification" in label else "rotation"
        fig.savefig(out_dir / "plots" / f"pass_rate_vs_{tag}.png")
        plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"{time.time() - t0:.0f}s, wrote {out_dir}")


if __name__ == "__main__":
    main()
