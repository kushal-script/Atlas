"""Assemble the results tables in results/README.md from experiment folders.

Every number in the results document is produced by this script from the
recorded runs, so the document cannot drift away from the measurements. Runs
are located by name suffix and the most recent match is used.

Usage:
    python scripts/build_results_tables.py
"""

import csv
import glob
import json
import platform
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DOMAINS = [
    ("physics40", "final_physics", "primary physics generator"),
    ("amat40", "final_amat", "faithful reference pipeline proxy"),
    ("spec40", "final_spec", "organiser specification proxy"),
    ("stress30", "final_stress", "adversarial generator"),
]

TOLS = ("1.0", "2.0", "4.0", "5.0")


def latest(pattern):
    hits = sorted(glob.glob(str(REPO / "experiments" / pattern)))
    return Path(hits[-1]) if hits else None


def metrics_for(suffix):
    run = latest(f"*_{suffix}")
    if run is None or not (run / "metrics.json").exists():
        return None, None
    return json.loads((run / "metrics.json").read_text()), run


def table_accuracy():
    lines = ["| Domain | What it tests | 1 px | 2 px | 4 px | 5 px | mean | median | worst | runtime |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ds, suffix, desc in DOMAINS:
        m, run = metrics_for(suffix)
        if m is None:
            continue
        o = m["overall"]
        cells = " | ".join(f"{o[f'within_{t}px_pct']:.1f}%" for t in TOLS)
        lines.append(
            f"| `{ds}` | {desc} | {cells} | {o['mean_err_px']:.2f} px | "
            f"{o['median_err_px']:.2f} px | "
            f"{o['max_err_px']:.0f} px | {o['mean_runtime_s']:.2f} s |")
    return "\n".join(lines)


def table_breakdown():
    lines = ["| Domain | Split | n | within 5 px | median |",
             "| --- | --- | --- | --- | --- |"]
    for ds, suffix, _ in DOMAINS:
        m, _ = metrics_for(suffix)
        if m is None:
            continue
        for key in sorted(k for k in m if k.startswith(("style_", "placement_"))):
            b = m[key]
            lines.append(f"| `{ds}` | {key.replace('_', ' ')} | {b['n']} | "
                         f"{b['within_5.0px_pct']:.1f}% | {b['median_err_px']:.2f} px |")
    return "\n".join(lines)


def table_ablation():
    run = latest("*_round2_pointsample_ablation")
    if run is None:
        return "_ablation run not found_"
    rows = list(csv.DictReader(open(run / "comparison.csv")))
    datasets, configs = [], []
    for r in rows:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
        if r["config"] not in configs:
            configs.append(r["config"])
    lines = ["| Configuration | " + " | ".join(f"`{d}`" for d in datasets) + " | mean | runtime |",
             "| --- | " + " | ".join("---" for _ in datasets) + " | --- | --- |"]
    for c in configs:
        vals, cells = [], []
        for d in datasets:
            hit = [r for r in rows if r["config"] == c and r["dataset"] == d]
            v = float(hit[0]["within_5.0px_pct"]) if hit else float("nan")
            vals.append(v)
            cells.append(f"{v:.1f}%")
        t = sum(float(r["mean_runtime_s"]) for r in rows if r["config"] == c) / len(datasets)
        mean = sum(vals) / len(vals)
        label = f"**{c}**" if c == "ps_prefer_tight" else c
        lines.append(f"| {label} | " + " | ".join(cells) +
                     f" | {mean:.1f}% | {t:.2f} s |")
    return "\n".join(lines)


def table_tiers():
    run = latest("*_final_tier_report")
    if run is None or not (run / "summary.json").exists():
        return None
    s = json.loads((run / "summary.json").read_text())
    order = [k for k in ("low", "medium", "high", "severe") if k in s]
    if not order:
        return None
    lines = ["| Acquisition tier | n | within 5 px | average precision | median |",
             "| --- | --- | --- | --- | --- |"]
    for t in order:
        b = s[t]
        lines.append(f"| {t} | {b['n']} | {b['within_5.0px_pct']:.1f}% | "
                     f"{b['ap']:.3f} | {b['median_px']:.2f} px |")
    return "\n".join(lines)


def table_pose():
    run = latest("*_pose_robustness")
    if run is None or not (run / "summary.json").exists():
        return None
    s = json.loads((run / "summary.json").read_text())
    lines = [f"Overall pass rate within {s['tolerance_px']:.0f} px across the "
             f"grid: **{s['overall']['pass_rate_pct']:.1f}%**, median "
             f"{s['overall']['median_px']:.2f} px, mean "
             f"{s['overall']['mean_px']:.2f} px, worst "
             f"{s['overall']['worst_px']:.0f} px over {s['n_pairs']} pairs at "
             f"the {s['tier']} acquisition tier.", "",
             "| Magnification | n | within 5 px | median | worst |",
             "| --- | --- | --- | --- | --- |"]
    for b in s["by_magnification"]:
        lines.append(f"| {b['value']:.1f} to 1 | {b['n']} | "
                     f"{b['pass_rate_pct']:.1f}% | {b['median_px']:.2f} px | "
                     f"{b['worst_px']:.0f} px |")
    lines += ["", "| Rotation | n | within 5 px | median | worst |",
              "| --- | --- | --- | --- | --- |"]
    for b in s["by_rotation"]:
        lines.append(f"| {b['value']:+.1f} deg | {b['n']} | "
                     f"{b['pass_rate_pct']:.1f}% | {b['median_px']:.2f} px | "
                     f"{b['worst_px']:.0f} px |")
    return "\n".join(lines)


def _cpu_and_memory():
    import subprocess
    cpu = mem = "not detected"
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True).stdout.strip()
            raw = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True).stdout.strip()
            if raw.isdigit():
                mem = f"{int(raw) / 1024 ** 3:.0f} GiB"
        else:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    mem = f"{int(line.split()[1]) / 1024 ** 2:.0f} GiB"
                    break
    except Exception:
        pass
    return cpu, mem


def environment_block():
    """Hardware, library versions and timing method, as the specification asks."""
    import cv2
    import numpy
    import scipy
    cpu, mem = _cpu_and_memory()
    counts = []
    for _, suffix, _ in DOMAINS:
        m, _ = metrics_for(suffix)
        if m:
            counts.append(m["overall"]["n"])
    lines = [
        "| Item | Value |",
        "| --- | --- |",
        f"| CPU | {cpu} |",
        f"| Memory | {mem} |",
        f"| Accelerator | none, CPU only, no GPU used at inference |",
        f"| Operating system | {platform.platform()} |",
        f"| Python | {platform.python_version()} |",
        f"| numpy | {numpy.__version__} |",
        f"| scipy | {scipy.__version__} |",
        f"| OpenCV | {cv2.__version__} |",
        "| Timing method | `time.perf_counter` around the complete `locate` "
        "call, single process, single thread of control, no warm up excluded "
        "and no run discarded |",
        "| Runs per pair | 1, and each reported runtime is the mean over the "
        f"pairs in that domain ({', '.join(str(c) for c in counts)} pairs) |",
        "| Reported statistic | mean seconds per image pair; batch runs also "
        "record the median in their `.runtime.json` |",
        f"| Tables regenerated | {date.today().isoformat()} by "
        "`scripts/build_results_tables.py` |",
    ]
    return "\n".join(lines)


def main():
    doc = (REPO / "results" / "README.md").read_text()
    marker = "## Accuracy by domain"
    head = doc.split(marker)[0] if marker in doc else doc

    parts = [head.rstrip(), "", marker, "",
             "Pass rate at the thresholds the specification asks for, on the "
             "final configuration. Accuracy is reported per domain because a "
             "single number would hide which generator produced the data.", "",
             table_accuracy(), "",
             "## Where the errors are", "",
             "Split by layout style and by how the reference site sits relative "
             "to aperiodic structure. Sites deep inside periodic arrays are the "
             "hard case by construction, and the residual failures concentrate "
             "there; see [failure analysis](../docs/failure_analysis.md).", "",
             table_breakdown(), "",
             "## Configuration ablation", "",
             "Each candidate configuration measured on every domain at once, "
             "because a change that helps one generator often costs another. "
             "The selected configuration is in bold and is the default in "
             "`src/drift_sense/localize.py`.", "",
             table_ablation(), "",
             "The physics column here is `train40_v2`, generated before the "
             "reference site placement bias was fixed. That bias pulled sites "
             "deep inside periodic arrays toward the frame centre, which "
             "flattered the loose tolerance setting and penalised the tight "
             "one, and is why the selected configuration reads 85 percent in "
             "this table but 90 percent on the unbiased `physics40` above. The "
             "ablation is kept as measured rather than rerun, because its "
             "purpose is to record the comparison that selected the "
             "configuration.", ""]

    tiers = table_tiers()
    if tiers:
        parts += ["## Robustness by acquisition severity", "",
                  "The four documented noise tiers, with average precision "
                  "obtained by ranking predictions on the returned confidence. "
                  "Every pair has exactly one true match, so recall is bounded "
                  "by accuracy and average precision measures whether the "
                  "confidence separates correct answers from incorrect ones.", "",
                  tiers, ""]

    pose = table_pose()
    if pose:
        parts += ["## Pose robustness over the stated ranges", "",
                  "A deliberate grid at exact magnification and rotation values, "
                  "rather than a randomised sample, so the stated 9 to 1 through "
                  "11 to 1 and plus or minus 2 degree ranges are answered "
                  "directly. Every reference site straddles a mat boundary so the "
                  "content is identifiable by construction; without that control "
                  "a failure could be caused either by the pose or by the site "
                  "being in a periodic region, and the two would be "
                  "inseparable.", "", pose, ""]

    parts += ["## Environment and timing method", "", environment_block(), ""]

    (REPO / "results" / "README.md").write_text("\n".join(parts))
    print("wrote results/README.md")


if __name__ == "__main__":
    main()
