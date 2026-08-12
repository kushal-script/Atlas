"""Factor sweep analysis over an evaluated dataset.

The organiser specification asks for results across multiple noise levels,
target positions, scales and rotations. Rather than generating one dataset per
factor, this reads an evaluated run together with the per pair metadata the
generators record and reports accuracy against each factor by binning, which
uses every pair for every factor and keeps the factors jointly randomised.

Usage:
    python scripts/sweep_analysis.py --run experiments/<stamp>_<name> \
        --dataset data/amat40 --name factor_sweep
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

TOL = 5.0
ACCENT = "#2a78d6"


def _factor(meta):
    p = meta.get("params", {})
    settings = meta.get("search_capture", {}).get("settings", {})
    gt = meta["ground_truth"]
    centre = np.hypot(gt["x"] - 499.5, gt["y"] - 499.5)
    return {
        "search_dose": float(settings.get("dose_e", p.get("dose_search", np.nan))),
        "abs_rotation_deg": abs(float(meta.get("relative_rotation_deg", 0.0))),
        "abs_scale_error_pct": abs(float(meta.get("search_scale_error", 0.0))) * 100.0,
        "distance_from_centre_px": float(centre),
        "read_noise_sigma": float(p.get("detector_noise_sigma_search", np.nan)),
        "shear_px": float(p.get("shear_amplitude_px", meta.get("shear_px", np.nan))),
        "jitter_px": float(p.get("drift_jitter_px", np.nan)),
        "beam_spot_nm": float(p.get("beam_spot_size_nm", np.nan)),
    }


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#ececea", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#5a5a55", labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="experiment folder holding results.csv")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--name", default="factor_sweep")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.run / "results.csv")))
    records = []
    for r in rows:
        meta_path = args.dataset / r["pair_id"] / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        records.append({"err_px": float(r["err_px"]), **_factor(meta)})
    if not records:
        raise SystemExit("no pairs matched between the run and the dataset")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "experiments" / f"{stamp}_{args.name}"
    (out_dir / "plots").mkdir(parents=True)

    factors = [k for k in records[0] if k != "err_px"]
    summary = {}
    for factor in factors:
        vals = np.array([r[factor] for r in records], dtype=float)
        errs = np.array([r["err_px"] for r in records], dtype=float)
        ok = np.isfinite(vals)
        if ok.sum() < 8 or np.nanstd(vals[ok]) == 0:
            continue
        vals, errs = vals[ok], errs[ok]
        n_bins = min(4, max(2, ok.sum() // 8))
        edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        bins = []
        for i in range(n_bins):
            m = (vals >= edges[i]) & (vals < edges[i + 1])
            if m.sum() == 0:
                continue
            bins.append({
                "range": [round(float(edges[i]), 4), round(float(edges[i + 1]), 4)],
                "n": int(m.sum()),
                "pass_rate_pct": round(float(100 * (errs[m] <= TOL).mean()), 1),
                "median_px": round(float(np.median(errs[m])), 3),
            })
        summary[factor] = bins

        fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=150)
        centres = [np.mean(b["range"]) for b in bins]
        ax.plot(centres, [b["pass_rate_pct"] for b in bins], marker="o",
                linewidth=2, color=ACCENT)
        for c, b in zip(centres, bins):
            ax.annotate(f"n={b['n']}", (c, b["pass_rate_pct"]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color="#5a5a55")
        ax.set_xlabel(factor.replace("_", " "))
        ax.set_ylabel(f"pass rate within {TOL:.0f} px, %")
        ax.set_ylim(-5, 105)
        ax.set_title(f"Accuracy versus {factor.replace('_', ' ')}", fontsize=10,
                     color="#1f1f1d")
        _style(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"pass_rate_vs_{factor}.png")
        plt.close(fig)

    with open(out_dir / "factor_summary.json", "w") as fh:
        json.dump({"tolerance_px": TOL, "n_pairs": len(records),
                   "source_run": str(args.run), "dataset": str(args.dataset),
                   "factors": summary}, fh, indent=2)
    for factor, bins in summary.items():
        spread = max(b["pass_rate_pct"] for b in bins) - min(b["pass_rate_pct"] for b in bins)
        print(f"{factor:26s} spread={spread:5.1f} points  " +
              "  ".join(f"[{b['range'][0]:g},{b['range'][1]:g}]={b['pass_rate_pct']:.0f}%"
                        for b in bins))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
