"""Evaluation harness: runs the localizer over a generated dataset, computes
accuracy and timing metrics, and writes plots plus a summary into a timestamped
experiment folder."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .localize import MatchConfig, load_gray, locate, optical_config

STYLE_COLORS = {"dram": "#2a78d6"}
TOLERANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0)


def _metrics_block(errors, runtimes):
    e = np.asarray(errors)
    r = np.asarray(runtimes)
    if e.size == 0:
        return {}
    block = {
        "n": int(e.size),
        "mean_err_px": float(e.mean()),
        "median_err_px": float(np.median(e)),
        "p95_err_px": float(np.percentile(e, 95)),
        "max_err_px": float(e.max()),
        "catastrophic_over_20px": int((e > 20).sum()),
        "mean_runtime_s": float(r.mean()),
        "median_runtime_s": float(np.median(r)),
    }
    for tol in TOLERANCES:
        block[f"within_{tol}px_pct"] = float(100.0 * (e <= tol).mean())
    return block


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c8c8c4")
    ax.tick_params(colors="#5a5a55", labelsize=9)
    ax.grid(True, color="#ececea", linewidth=0.8)
    ax.set_axisbelow(True)


def _plot_cdf(rows, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    groups = {"all": [r["err_px"] for r in rows]}
    for style in sorted({r["style"] for r in rows}):
        groups[style] = [r["err_px"] for r in rows if r["style"] == style]
    colors = {"all": "#3a3a37", **STYLE_COLORS}
    for name, errs in groups.items():
        e = np.sort(np.asarray(errs))
        frac = np.arange(1, e.size + 1) / e.size * 100
        ax.plot(e, frac, color=colors.get(name, "#888"), linewidth=2, label=f"{name} (n={e.size})")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("localization error, search image pixels")
    ax.set_ylabel("pairs within error, %")
    ax.set_title("Error distribution", fontsize=11, color="#1f1f1d")
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_hist(rows, path):
    e = np.asarray([r["err_px"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    bins = np.logspace(-2, np.log10(max(e.max() * 1.2, 1.0)), 30)
    ax.hist(e, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("localization error, search image pixels")
    ax.set_ylabel("pairs")
    ax.set_title("Error histogram", fontsize=11, color="#1f1f1d")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_scatter(rows, key, xlabel, path, xlog=False):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for style in sorted({r["style"] for r in rows}):
        xs = [r[key] for r in rows if r["style"] == style]
        ys = [max(r["err_px"], 1e-3) for r in rows if r["style"] == style]
        ax.scatter(xs, ys, s=36, color=STYLE_COLORS.get(style, "#7a7a76"), label=style, alpha=0.85,
                   edgecolors="white", linewidths=0.5)
    ax.set_yscale("log")
    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("error, px (log)")
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _montage(pair_dir, row, resp, path, title):
    ref, _ = load_gray(pair_dir / "reference.png")
    search, _ = load_gray(pair_dir / "search.png")
    meta = json.loads((pair_dir / "meta.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=150)
    axes[0].imshow(ref, cmap="gray")
    axes[0].set_title("reference, 1 nm per px", fontsize=9)
    axes[1].imshow(search, cmap="gray")
    if "gt_corners_xy" in meta:
        corners = np.array(meta["gt_corners_xy"] + [meta["gt_corners_xy"][0]])
        axes[1].plot(corners[:, 0], corners[:, 1], color="#1baf7a", linewidth=1.5)
    axes[1].plot(row["gt_x"], row["gt_y"], "+", color="#1baf7a", markersize=12,
                 markeredgewidth=2, label="truth")
    axes[1].plot(row["pred_x"], row["pred_y"], "x", color="#eb6834", markersize=10,
                 markeredgewidth=2, label="predicted")
    axes[1].legend(frameon=True, fontsize=8, loc="upper right")
    axes[1].set_title(f"search, err {row['err_px']:.2f} px", fontsize=9)
    im = axes[2].imshow(resp, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_title("correlation response", fontsize=9)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_evaluation(dataset_dir, out_dir, cfg=None, limit=None):
    cfg = cfg or MatchConfig()
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    pair_dirs = sorted(d for d in dataset_dir.iterdir()
                       if d.is_dir() and d.name.startswith("pair_"))
    if limit:
        pair_dirs = pair_dirs[:limit]

    rows = []
    responses = {}
    for pd in pair_dirs:
        meta = json.loads((pd / "meta.json").read_text())
        ref, ref_rgb = load_gray(pd / "reference.png")
        search, _ = load_gray(pd / "search.png")
        x, y, diag, resp = locate(ref, search, optical_config() if ref_rgb else cfg)
        gt = meta["ground_truth"]
        err = float(np.hypot(x - gt["x"], y - gt["y"]))
        row = {
            "pair_id": pd.name,
            "reference_path": str((pd / "reference.png").relative_to(dataset_dir)),
            "search_path": str((pd / "search.png").relative_to(dataset_dir)),
            "style": meta["style"],
            "placement": meta["placement"],
            "gt_x": gt["x"], "gt_y": gt["y"],
            "pred_x": x, "pred_y": y,
            "err_px": err,
            "err_nm": err * meta["search_capture"]["pose"]["pixel_nm"],
            "score": diag["score"],
            "theta_est_deg": diag["theta_deg"],
            "scale_est": diag["scale"],
            "gt_rotation_deg": meta["relative_rotation_deg"],
            "gt_scale_error": meta["search_scale_error"],
            "search_dose_e": meta["search_capture"]["settings"].get(
                "dose_e", meta["search_capture"]["settings"].get("photon_dose", 0.0)),
            "num_candidates": diag["num_candidates"],
            "stage2_used": diag["stage2"]["used"],
            "stage2_margin": diag["stage2"]["margin"],
            "runtime_s": diag["runtime_s"],
        }
        rows.append(row)
        responses[pd.name] = resp
        print(f"{pd.name} {meta['style']:7s} err={err:7.3f}px "
              f"runtime={diag['runtime_s']:.2f}s score={diag['score']:.3f}")

    metrics = {"overall": _metrics_block([r["err_px"] for r in rows],
                                         [r["runtime_s"] for r in rows])}
    for group_key in ("style", "placement"):
        for val in sorted({r[group_key] for r in rows}):
            sub = [r for r in rows if r[group_key] == val]
            metrics[f"{group_key}_{val}"] = _metrics_block(
                [r["err_px"] for r in sub], [r["runtime_s"] for r in sub])

    with open(out_dir / "results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(out_dir / "config.json", "w") as fh:
        json.dump({"dataset": str(dataset_dir), "match_config": vars(cfg),
                   "n_pairs": len(rows)}, fh, indent=2, default=str)

    _plot_cdf(rows, plots / "error_cdf.png")
    _plot_hist(rows, plots / "error_histogram.png")
    _plot_scatter(rows, "gt_rotation_deg", "true relative rotation, deg",
                  plots / "error_vs_rotation.png")
    _plot_scatter(rows, "search_dose_e", "search image dose, electrons per px",
                  plots / "error_vs_dose.png", xlog=True)

    best = min(rows, key=lambda r: r["err_px"])
    worst = max(rows, key=lambda r: r["err_px"])
    _montage(dataset_dir / best["pair_id"], best, responses[best["pair_id"]],
             plots / "montage_success.png", f"success case {best['pair_id']}")
    _montage(dataset_dir / worst["pair_id"], worst, responses[worst["pair_id"]],
             plots / "montage_failure.png", f"hardest case {worst['pair_id']}")

    return rows, metrics
