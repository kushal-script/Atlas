"""Tiered robustness evaluation with precision recall analysis.

The organiser specification asks for results across multiple noise levels and
for a repeatable confidence score. This harness evaluates the localizer over
the four noise tiers used by the reference project (low, medium, high, severe),
reporting the pass rate at the 5, 4, 2 and 1 pixel thresholds per tier plus
average precision obtained by ranking predictions on the returned confidence.

Average precision here follows the same convention as the reference harness:
every pair has exactly one true match, so recall is true positives over the
number of pairs and average precision is bounded above by accuracy. The
interesting question it answers is whether the confidence score separates
correct predictions from incorrect ones, which is what makes an abstention
policy usable on a real tool.

Usage:
    python scripts/evaluate_tiers.py --dataset data/amat_tiers --name tier_report
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from drift_sense.api import match_pair
from drift_sense.localize import load_gray

TOLERANCES = (1.0, 2.0, 4.0, 5.0)
TIER_COLORS = {"low": "#2a78d6", "medium": "#1baf7a",
               "high": "#eda100", "severe": "#eb6834"}


def precision_recall(scores, corrects):
    order = np.argsort(-np.asarray(scores, dtype=float))
    c = np.asarray(corrects, dtype=bool)[order]
    tp = np.cumsum(c)
    fp = np.cumsum(~c)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(len(c), 1)
    return (np.concatenate([[1.0], precision]),
            np.concatenate([[0.0], recall]))


def average_precision(precision, recall):
    order = np.argsort(recall)
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(np.asarray(precision)[order], np.asarray(recall)[order]))


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#ececea", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#5a5a55", labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--name", default="tier_report")
    ap.add_argument("--tolerance_px", type=float, default=5.0)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "experiments" / f"{stamp}_{args.name}"
    (out_dir / "plots").mkdir(parents=True)

    pairs = sorted(d for d in args.dataset.iterdir()
                   if d.is_dir() and d.name.startswith("pair_"))
    rows = []
    t0 = time.time()
    for pd in pairs:
        meta = json.loads((pd / "meta.json").read_text())
        ref, _ = load_gray(pd / "reference.png")
        search, _ = load_gray(pd / "search.png")
        res = match_pair(ref, search)
        g = meta["ground_truth"]
        err = float(np.hypot(res["x"] - g["x"], res["y"] - g["y"]))
        rows.append({
            "pair_id": pd.name,
            "reference_path": str((pd / "reference.png").relative_to(args.dataset)),
            "search_path": str((pd / "search.png").relative_to(args.dataset)),
            "tier": meta.get("tier", meta.get("placement", "all")),
            "style": meta.get("style", ""),
            "gt_x": round(g["x"], 3), "gt_y": round(g["y"], 3),
            "pred_x": round(res["x"], 3), "pred_y": round(res["y"], 3),
            "err_px": round(err, 4),
            "confidence": round(res["score"], 4),
            "peak_score": round(res["peak_score"], 4),
            "confidence_regime": res["confidence_regime"],
            "runtime_s": round(res["runtime_s"], 3),
        })
        print(f"{pd.name} {rows[-1]['tier']:8s} err={err:8.2f} "
              f"conf={res['score']:.3f} {res['confidence_regime']}", flush=True)

    by_tier = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)

    summary = {}
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    for tier in sorted(by_tier):
        sub = by_tier[tier]
        e = np.array([r["err_px"] for r in sub])
        corrects = e <= args.tolerance_px
        p, rc = precision_recall([r["confidence"] for r in sub], corrects)
        apv = average_precision(p, rc)
        block = {"n": len(sub), "ap": round(apv, 4),
                 "accuracy": round(float(corrects.mean()), 4),
                 "median_px": round(float(np.median(e)), 3),
                 "mean_px": round(float(e.mean()), 2),
                 "worst_px": round(float(e.max()), 1),
                 "mean_runtime_s": round(float(np.mean([r["runtime_s"] for r in sub])), 3)}
        for tol in TOLERANCES:
            block[f"within_{tol}px_pct"] = round(float(100 * (e <= tol).mean()), 1)
        summary[tier] = block
        ax.plot(rc, p, marker="o", markersize=3, linewidth=1.8,
                color=TIER_COLORS.get(tier, "#7a7a76"),
                label=f"{tier} (AP={apv:.2f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Confidence ranked precision recall, tolerance {args.tolerance_px} px",
                 fontsize=10, color="#1f1f1d")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "precision_recall_by_tier.png")
    plt.close(fig)

    tiers = sorted(summary)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    xs = np.arange(len(tiers))
    ax.plot(xs, [summary[t]["ap"] for t in tiers], marker="o", linewidth=2,
            color="#2a78d6", label="average precision")
    ax.plot(xs, [summary[t]["accuracy"] for t in tiers], marker="s", linewidth=2,
            color="#eb6834", label=f"accuracy within {args.tolerance_px} px")
    ax.set_xticks(xs)
    ax.set_xticklabels(tiers)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.set_title("Localizer quality versus acquisition severity", fontsize=10,
                 color="#1f1f1d")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "quality_vs_severity.png")
    plt.close(fig)

    overall = np.array([r["err_px"] for r in rows])
    summary["overall"] = {
        "n": len(rows),
        "median_px": round(float(np.median(overall)), 3),
        "mean_px": round(float(overall.mean()), 2),
        "worst_px": round(float(overall.max()), 1),
        "mean_runtime_s": round(float(np.mean([r["runtime_s"] for r in rows])), 3),
        **{f"within_{tol}px_pct": round(float(100 * (overall <= tol).mean()), 1)
           for tol in TOLERANCES},
    }
    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"{time.time() - t0:.0f}s, wrote {out_dir}")


if __name__ == "__main__":
    main()
