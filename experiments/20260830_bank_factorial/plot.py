"""Plots for the blur bank factorial and the per pair budget fix.

    .venv/bin/python experiments/20260830_bank_factorial/plot.py
"""

import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "plots"

# Measured serially on data/p2degraded, 200 pairs, master seed 7001.
CONFIGS = [
    ("baseline\n4,9,16,25  k6", 0.701, 0.521, 41.82),
    ("+36\nk6", 0.796, 0.496, 16.02),
    ("4,9,16,25\nk8", 0.700, 0.509, 14.84),
    ("+36\nk8", 0.797, 0.497, 17.04),
    ("+32,42\nk9", 0.795, 0.491, 13.78),
    ("shipped\nper pair budget", 0.700, 0.503, 6.45),
]


def credit(e):
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0


def bar_factorial():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(CONFIGS))
    w = 0.38
    a = [c[1] for c in CONFIGS]
    b = [c[2] for c in CONFIGS]
    ax.bar(x - w / 2, a, w, label="Set A nominal", color="#4472a8")
    ax.bar(x + w / 2, b, w, label="Set B degraded", color="#c8663a")
    ax.axhline(CONFIGS[0][2], ls="--", lw=1, color="#c8663a", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in CONFIGS], fontsize=8)
    ax.set_ylabel("localization credit")
    ax.set_title("Blur bank factorial: every extension lifts Set A and costs Set B,\n"
                 "and the prescreen budget changes neither", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "factorial_credit.png", dpi=150)


def runtime_tail():
    rows = {n: list(csv.DictReader(open(HERE / f)))
            for n, f in (("per call budget", "results_baseline.csv"),
                         ("per pair budget", "results_shipped.csv"))}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, r in rows.items():
        t = np.sort([float(x["seconds"]) for x in r])
        ax1.plot(t, np.arange(1, len(t) + 1) / len(t) * 100, label=name, lw=1.8)
    ax1.axvline(20, color="crimson", ls="--", lw=1.2)
    ax1.text(20.4, 40, "scored hard timeout", color="crimson", fontsize=8, rotation=90)
    ax1.set_xlabel("seconds per pair")
    ax1.set_ylabel("percent of pairs at or below")
    ax1.set_xscale("log")
    ax1.set_title("Runtime distribution, 200 pairs", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    for name, r in rows.items():
        sev = {}
        for x in r:
            sev.setdefault(x["severity"], []).append(float(x["seconds"]))
        ks = sorted(sev)
        ax2.plot(ks, [np.median(sev[k]) for k in ks], marker="o", label=f"{name} median")
        ax2.plot(ks, [max(sev[k]) for k in ks], marker="^", ls="--", alpha=0.7,
                 label=f"{name} max")
    ax2.axhline(20, color="crimson", ls="--", lw=1.2)
    ax2.set_xlabel("degradation severity (0 is nominal)")
    ax2.set_ylabel("seconds per pair")
    ax2.set_title("The budget was per call, so the width rescue tripled\n"
                  "exactly the pairs already nearest the timeout", fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "runtime_tail.png", dpi=150)


def honest_scoring():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    limits = np.linspace(5, 25, 60)
    for name, f, colour in (("per call budget", "results_baseline.csv", "#c8663a"),
                            ("per pair budget", "results_shipped.csv", "#4472a8")):
        r = [x for x in csv.DictReader(open(HERE / f)) if x["set"] == "B_degraded"]
        y = [sum(credit(float(x["err"])) if (x["found"] == "1"
                                             and float(x["seconds"]) <= lim) else 0.0
                 for x in r) / len(r) for lim in limits]
        ax.plot(limits, y, label=name, lw=2, color=colour)
    ax.axvline(20, color="crimson", ls="--", lw=1.2)
    ax.text(20.3, 0.46, "scored timeout", color="crimson", fontsize=8, rotation=90)
    ax.set_xlabel("hard timeout charged, seconds")
    ax.set_ylabel("Set B credit")
    ax.set_title("Charging for the timeout reverses the comparison:\n"
                 "the per call budget only wins on time it does not have", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "honest_scoring.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    bar_factorial()
    runtime_tail()
    honest_scoring()
    print(f"wrote {len(list(OUT.glob('*.png')))} plots to {OUT}")
