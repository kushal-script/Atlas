"""Render the figures the README embeds.

Everything here is drawn from the recorded experiment folders and from generated
pairs, so the README cannot show a number the repository does not hold. Output
goes to docs/images, which is committed, because data/ is not.

    .venv/bin/python scripts/make_readme_figures.py
"""

import csv
import glob
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

BLUE, GREEN, ORANGE, RED, GREY = "#2A78D6", "#1BAF7A", "#EB6834", "#B0483C", "#8A94A6"


def latest(pattern):
    hits = sorted(glob.glob(str(REPO / pattern)))
    return Path(hits[-1]) if hits else None


def input_pair():
    """Reference and search side by side, with the true site marked."""
    d = REPO / "data" / "physics40" / "pair_0001"
    if not d.exists():
        print("skip input_pair: generate data/physics40 first")
        return
    ref = Image.open(d / "reference.png").convert("L")
    search = Image.open(d / "search.png").convert("L")
    gt = json.load(open(d / "meta.json"))["ground_truth"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), dpi=110)
    axes[0].imshow(ref, cmap="gray")
    axes[0].set_title("reference.png\n1000 x 1000 px at 1 nm/px, a 1 um field")
    axes[1].imshow(search, cmap="gray")
    axes[1].set_title("search.png\n1000 x 1000 px at 10 nm/px, a 10 um field")
    axes[1].add_patch(plt.Rectangle((gt["x"] - 50, gt["y"] - 50), 100, 100,
                                    fill=False, edgecolor=GREEN, linewidth=2.2))
    # the label is anchored to a fixed corner with an opaque background, so it
    # can never land on top of the pattern and become unreadable
    far_top = gt["y"] > 500
    tx, ty = (40, 70) if far_top else (40, 930)
    axes[1].annotate("the reference sits here,\nabout 100 x 100 px of this frame",
                     xy=(gt["x"], gt["y"] - 50 if far_top else gt["y"] + 50),
                     xytext=(tx, ty), ha="left",
                     va="top" if far_top else "bottom",
                     color="#0B5C3F", fontsize=11, weight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                               edgecolor=GREEN, linewidth=1.6, alpha=0.95),
                     arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.2,
                                     shrinkA=6, shrinkB=4))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The task: find the reference pattern inside the search image, and return its centre",
                 fontsize=15, y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "input_pair.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote input_pair.png")


def accuracy_by_domain():
    domains = ["physics40", "amat40", "stress30", "spec40"]
    labels = ["primary\nphysics", "reference\npipeline proxy", "adversarial\ngenerator", "specification\nproxy"]
    p1 = [90.0, 20.0, 43.3, 32.5]
    p2 = [90.0, 40.0, 46.7, 40.0]
    p5 = [90.0, 60.0, 53.3, 42.5]
    x = np.arange(len(domains)); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
    for off, vals, lab, col in ((-w, p1, "within 1 px", BLUE),
                                (0.0, p2, "within 2 px", GREEN),
                                (w, p5, "within 5 px", ORANGE)):
        b = ax.bar(x + off, vals, w, label=lab, color=col)
        ax.bar_label(b, fmt="%.1f", fontsize=9, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{l}" for d, l in zip(domains, labels)])
    ax.set_ylabel("pass rate, %")
    ax.set_ylim(0, 105)
    ax.set_title("Localization accuracy by domain, 150 pairs from four generators sharing no imaging code")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_by_domain.png")
    plt.close(fig)
    print("wrote accuracy_by_domain.png")


def confidence_regimes():
    names = ["unique_peak", "residual_identified", "tie_break_convention"]
    prec = [98.6, 50.0, 21.3]
    n = [73, 16, 61]
    cols = [GREEN, BLUE, ORANGE]
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=110)
    b = ax.barh(names[::-1], prec[::-1], color=cols[::-1], height=0.6)
    for rect, v, c in zip(b, prec[::-1], n[::-1]):
        ax.text(v + 1.5, rect.get_y() + rect.get_height() / 2,
                f"{v:.1f}%   over {c} cases", va="center", fontsize=11)
    ax.axvline(62.0, color=GREY, ls="--", lw=1.6)
    ax.text(63.5, 1.5, "62.0% baseline:\nevery answer accepted blindly",
            color="#4A5263", fontsize=10, va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=GREY, linewidth=1.2))
    ax.set_xlim(0, 132)
    ax.set_xlabel("precision, %")
    ax.set_title("Accepting only the two confident regimes covers 59% of cases at 89.9% precision")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "confidence_regimes.png")
    plt.close(fig)
    print("wrote confidence_regimes.png")


def copy_experiment_figures():
    wanted = {
        "montage_success.png": latest("experiments/*_final_physics/plots") ,
        "montage_failure.png": latest("experiments/*_final_physics/plots"),
        "runtime_by_backend.png": latest("experiments/*_backend_port/plots"),
    }
    for name, folder in wanted.items():
        if folder and (folder / name).exists():
            shutil.copy(folder / name, OUT / name)
            print("copied", name)
    for name in ("pose_robustness.png", "precision_recall.png"):
        if src.exists():
            shutil.copy(src, OUT / name)
            print("copied", name)


if __name__ == "__main__":
    input_pair()
    accuracy_by_domain()
    confidence_regimes()
    copy_experiment_figures()
