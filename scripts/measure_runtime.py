"""Controlled runtime measurement protocol.

Runtime figures accumulated during long experiment sessions are not
comparable, because sustained compute thermally throttles this machine by
roughly 1.7 times. This protocol produces the reportable number: run from a
cool state, perform warm up localizations that are discarded, then time N
localizations and report the median and mean together with the hardware,
library versions and timing method. The result is written to
results/runtime_protocol.json and is the source for the runtime quoted in the
presentation.

Usage, after the machine has been idle:
    python scripts/measure_runtime.py --dataset data/physics40 --n 10
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from drift_sense.localize import MatchConfig, load_gray, locate, phase2_config


def _cpu_mem():
    cpu = mem = "not detected"
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True).stdout.strip()
            raw = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True).stdout.strip()
            if raw.isdigit():
                mem = f"{int(raw) / 1024 ** 3:.0f} GiB"
    except Exception:
        pass
    return cpu, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=REPO / "data" / "physics40")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    pairs = sorted(d for d in args.dataset.iterdir()
                   if d.is_dir() and d.name.startswith("pair_"))[:args.n + args.warmup]
    if len(pairs) < args.n + args.warmup:
        raise SystemExit("not enough pairs in the dataset")

    cfg = phase2_config()
    times = []
    for i, pd in enumerate(pairs):
        ref, _ = load_gray(pd / "reference.png")
        search, _ = load_gray(pd / "search.png")
        t0 = time.perf_counter()
        locate(ref, search, cfg)
        dt = time.perf_counter() - t0
        role = "warmup" if i < args.warmup else "timed"
        if role == "timed":
            times.append(dt)
        print(f"{pd.name} {role:6s} {dt:.3f}s", flush=True)

    t = np.array(times)
    import cv2
    import numpy
    import scipy
    cpu, mem = _cpu_mem()
    record = {
        "protocol": "cool start, discard warmup runs, time N localizations "
                    "with time.perf_counter around the complete locate call, "
                    "single process, report median and mean",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "warmup_runs_discarded": args.warmup,
        "timed_runs": len(times),
        "median_s_per_pair": round(float(np.median(t)), 3),
        "mean_s_per_pair": round(float(t.mean()), 3),
        "min_s": round(float(t.min()), 3),
        "max_s": round(float(t.max()), 3),
        "cpu": cpu, "memory": mem,
        "accelerator": "none, CPU only",
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "opencv": cv2.__version__,
    }
    out = REPO / "results" / "runtime_protocol.json"
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
