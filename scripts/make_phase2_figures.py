"""Render the Phase 2 figures the README embeds.

Every number is transcribed from a committed experiment log, named beside the
value it feeds, so the README cannot show a figure the repository does not
hold. Output goes to docs/images.

    .venv/bin/python scripts/make_phase2_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

BLUE, GREEN, ORANGE, RED, GREY = "#2A78D6", "#1BAF7A", "#EB6834", "#B0483C", "#8A94A6"


def core_by_suite():
    """experiments/20260901_stress_and_decoys report and final battery."""
    battery = [
        ("released recipe, hardened 40", 81.54, GREEN),
        ("released recipe, sample 40", 79.22, GREEN),
        ("released recipe, holdout 60", 76.64, GREEN),
        ("own generator, holdout A 120", 68.35, GREY),
        ("own generator, holdout B 120", 59.17, GREY),
    ]
    seeds = [("909090", 81.68), ("171819", 79.61), ("141516", 79.20),
             ("202122", 76.82), ("111213", 76.59)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    names = [b[0] for b in battery][::-1]
    vals = [b[1] for b in battery][::-1]
    cols = [b[2] for b in battery][::-1]
    ax1.barh(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax1.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=11)
    ax1.set_xlim(0, 92)
    ax1.axvline(85, color=RED, lw=1, ls="--")
    ax1.text(85.4, 0.1, "85 core points", color=RED, fontsize=10, rotation=90, va="bottom")
    ax1.set_xlabel("estimated core score of 85")
    ax1.set_title("Held out suites, both generators")
    ax1.spines[["top", "right"]].set_visible(False)

    sn = [s[0] for s in seeds][::-1]
    sv = [s[1] for s in seeds][::-1]
    ax2.barh(sn, sv, color=BLUE)
    for i, v in enumerate(sv):
        ax2.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=11)
    mean = float(np.mean(sv))
    ax2.axvline(mean, color=ORANGE, lw=2)
    ax2.text(mean, 4.55, f"mean {mean:.1f}", color=ORANGE, fontsize=11, ha="center")
    ax2.set_xlim(70, 86)
    ax2.set_xlabel("estimated core score of 85")
    ax2.set_ylabel("post freeze seed")
    ax2.set_title("Five surprise seeds, chosen after freezing")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "p2_core_by_suite.png", dpi=150)
    plt.close(fig)


def severity_credit():
    """Own ladder: readme degraded attribution; released ladder and severity
    five: experiments/20260901_stress_and_decoys."""
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot([1, 2, 3, 4], [0.788, 0.637, 0.433, 0.206], "o-", color=GREY,
            lw=2, label="own generator, severities 1 to 4")
    ax.plot([2, 3, 4, 5], [0.971, 0.933, 0.850, 0.680], "s-", color=GREEN,
            lw=2, label="released recipe, severities 2 to 5")
    ax.axvspan(4.5, 5.5, color=ORANGE, alpha=0.12)
    ax.text(5.0, 0.06, "past the\ndisclosed ladder", ha="center", fontsize=10, color=ORANGE)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("degradation severity")
    ax.set_ylabel("localization credit")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Credit versus severity: the self grading physics is the harsher one")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "p2_severity_credit.png", dpi=150)
    plt.close(fig)


def alien_mutations():
    """experiments/20260901_alien_distribution results log."""
    muts = ["dark streak bands", "vertical streak bands", "gamma 0.55",
            "tone crush to 70 grays", "speckle 0.5", "inverted contrast"]
    before = [1.000, 1.000, 1.000, 1.000, 1.000, 0.714]
    after = [1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
    y = np.arange(len(muts))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.barh(y + 0.18, before, height=0.34, color=GREY, label="before the polarity fix")
    ax.barh(y - 0.18, after, height=0.34, color=GREEN, label="after")
    ax.set_yticks(y, muts)
    ax.axvline(1.0, color=RED, lw=1, ls="--")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("present pair localization credit")
    ax.set_title("Six appearance families the training never saw")
    ax.legend(frameon=False, loc="lower left")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "p2_alien_mutations.png", dpi=150)
    plt.close(fig)


def threshold_tradeoff():
    """experiments/20260901_stress_and_decoys threshold_selection.log."""
    thr = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
           0.60, 0.65, 0.70, 0.75, 0.80]
    found = [71.02, 71.01, 70.91, 70.81, 70.77, 70.60, 70.26, 70.14, 69.97,
             69.76, 69.67, 69.43, 69.00, 68.58, 68.19]
    reject = [64.31, 65.41, 65.66, 65.98, 66.41, 66.45, 66.15, 66.25, 66.38,
              66.45, 66.55, 66.58, 66.19, 65.82, 65.41]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(thr, found, "o-", color=BLUE, lw=2, label="found class grading")
    ax.plot(thr, reject, "s-", color=ORANGE, lw=2, label="reject class grading")
    ax.axvline(0.35, color=GREEN, lw=2)
    ax.text(0.357, 64.6, "shipped 0.35", color=GREEN, fontsize=11)
    ax.set_xlabel("presence threshold")
    ax.set_ylabel("pooled core, 492 pairs, five suites")
    ax.set_title("Operating point under both readings of the graded F1")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "p2_threshold_tradeoff.png", dpi=150)
    plt.close(fig)


def runtime():
    """Backend microbenchmark and the 60 pair gate in
    experiments/20260901_stress_and_decoys."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax1.bar(["matchTemplate\nper call", "cached transforms\nper call"],
            [52.5, 25.5], color=[GREY, GREEN], width=0.55)
    for i, v in enumerate([52.5, 25.5]):
        ax1.text(i, v + 1.2, f"{v:.1f} ms", ha="center", fontsize=11)
    ax1.set_ylabel("milliseconds")
    ax1.set_title("One correlation, 95 px template in 1000 px")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.bar(["direct\ncorrelator", "calibrated\ncorrelator"], [3.85, 2.85],
            color=[GREY, GREEN], width=0.55)
    for i, v in enumerate([3.85, 2.85]):
        ax2.text(i, v + 0.08, f"{v:.2f} s", ha="center", fontsize=11)
    ax2.axhline(5.0, color=RED, lw=1.5, ls="--")
    ax2.text(1.32, 5.05, "5 s median requirement", color=RED, fontsize=10, ha="right")
    ax2.set_ylabel("median seconds per pair")
    ax2.set_ylim(0, 5.6)
    ax2.set_title("60 pair identity gate, byte identical output")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "p2_runtime.png", dpi=150)
    plt.close(fig)


def organiser_sample():
    """experiments/20260901_organiser_sample_validation and the final battery."""
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    labels = ["organisers' own\nZNCC baseline", "this pipeline,\nSeptember 1 morning",
              "this pipeline,\ncurrent build"]
    vals = [0.800, 0.925, 0.988]
    bars = ax.bar(labels, vals, color=[GREY, BLUE, GREEN], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=12)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("present pair credit")
    ax.set_title("The organisers' shared 20 pairs, never fitted on")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "p2_organiser_sample.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    core_by_suite()
    severity_credit()
    alien_mutations()
    threshold_tradeoff()
    runtime()
    organiser_sample()
    print("wrote 6 figures to", OUT)
