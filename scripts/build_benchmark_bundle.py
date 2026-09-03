"""Build a portable benchmark bundle for cross-device testing.

Creates drift-sense-benchmark-<date>.zip containing only what is needed to
run the one-click benchmark on another machine (which has internet for pip).

    .venv/bin/python scripts/build_benchmark_bundle.py
    .venv/bin/python scripts/build_benchmark_bundle.py --out dist/benchmark.zip --smoke-num 8

Verification (default): extracts to a temp dir and runs a smoke benchmark
(--num 8) with sockets allowed, asserts benchmark_report.json exists with
required keys and that predictions have correct columns/order. This mirrors
scripts/package_submission.py self-test but for the benchmark path.

Assumes target has internet; no wheelhouse bundled.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Minimal runnable set: reuse submission INCLUDE plus benchmark additions
# We import INCLUDE from package_submission to stay in sync, then add ours.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from package_submission import INCLUDE as SUBMISSION_INCLUDE
except Exception:
    # fallback if import fails
    SUBMISSION_INCLUDE = []

BENCHMARK_EXTRA = [
    "benchmark.sh",
    "benchmark.ps1",
    "benchmark.bat",
    "scripts/run_benchmark.py",
    "scripts/build_benchmark_bundle.py",
]

# Files we intentionally exclude even if listed in INCLUDE
EXCLUDE = {
    # large regenerable data stays out of bundle
    "data",
}

def _collect_files():
    files = []
    seen = set()
    for f in SUBMISSION_INCLUDE + BENCHMARK_EXTRA:
        # skip empty/invalid
        if not f or f in seen:
            continue
        seen.add(f)
        src = REPO / f
        if not src.exists():
            # BENCHMARK_EXTRA files must exist (we just created them)
            # SUBMISSION_INCLUDE files may have models etc; warn not fail for optional
            print(f"  warning: {f} listed but not found on disk, skipping")
            continue
        # skip if under excluded prefix
        if any(f == e or f.startswith(e + "/") for e in EXCLUDE):
            continue
        files.append(f)
    # Also ensure .gitattributes if we create it
    if (REPO / ".gitattributes").exists() and ".gitattributes" not in files:
        files.append(".gitattributes")
    return files


def main():
    ap = argparse.ArgumentParser(description="Build benchmark bundle")
    ap.add_argument("--out", type=Path, default=REPO / "dist" / "drift-sense-benchmark.zip",
                    help="output zip path")
    ap.add_argument("--smoke-num", type=int, default=4, help="pairs for self-test (0 to skip)")
    ap.add_argument("--no-smoke", action="store_true", help="skip self-test")
    ap.add_argument("--python", type=Path, default=Path(sys.executable),
                    help="python to run smoke test with")
    args = ap.parse_args()

    files = _collect_files()
    missing = [f for f in files if not (REPO / f).exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            src = REPO / f
            # preserve executable bit for .sh
            info = zipfile.ZipInfo.from_file(src, f)
            # ensure benchmark.sh is executable
            if f == "benchmark.sh":
                info.external_attr = (0o755 << 16) | info.external_attr
            with open(src, "rb") as fh:
                z.writestr(info, fh.read())
        # also store a pip freeze at build time for provenance
        try:
            freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                    capture_output=True, text=True, timeout=10).stdout
            z.writestr("build_pip_freeze.txt", freeze)
        except Exception:
            pass

    sz = args.out.stat().st_size
    print(f"packed {args.out} ({sz/1024:.0f} KB, {len(files)} files)")
    print("  includes:")
    for f in sorted(files):
        print(f"    {f}")

    # Verify zip has required entry points
    with zipfile.ZipFile(args.out) as z:
        names = set(z.namelist())
    for req in ["benchmark.sh", "benchmark.ps1", "benchmark.bat",
                "scripts/run_benchmark.py", "register.py", "requirements_phase2.txt"]:
        if req not in names:
            raise SystemExit(f"bundle missing required {req}")

    if args.no_smoke or args.smoke_num <= 0:
        print("smoke test skipped (--no-smoke)")
        return

    # Self-test: extract to temp, run quick smoke benchmark
    smoke_num = args.smoke_num
    print(f"\n--- self-test: extract bundle, run benchmark --num {smoke_num} ---")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(args.out) as z:
            z.extractall(td / "run")
        run_dir = td / "run"
        # Ensure benchmark.sh is executable on unix
        try:
            (run_dir / "benchmark.sh").chmod(0o755)
        except Exception:
            pass
        # Run the Python driver directly (faster than via shell, same logic)
        cmd = [str(args.python), "scripts/run_benchmark.py", "--num", str(smoke_num), "--seed", "1",
               "--out", str(td / "out")]
        print("  " + " ".join(cmd) + f"  (cwd={run_dir})")
        res = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True, timeout=600)
        print(res.stdout[-3000:] if len(res.stdout) > 3000 else res.stdout)
        if res.stderr:
            print("--- stderr ---")
            print(res.stderr[-2000:])
        if res.returncode != 0:
            raise SystemExit(f"smoke benchmark failed (exit {res.returncode})")
        report = td / "out" / "benchmark_report.json"
        pred = td / "out" / "predictions.csv"
        if not report.exists():
            raise SystemExit("smoke benchmark produced no benchmark_report.json")
        if not pred.exists():
            raise SystemExit("smoke benchmark produced no predictions.csv")
        data = json.loads(report.read_text())
        for k in ["runtime", "estimated_core", "system", "mode"]:
            if k not in data:
                raise SystemExit(f"report missing key {k}")
        runtime = data["runtime"]
        for k in ["median_s_per_pair", "mean_s_per_pair", "max_s"]:
            if k not in runtime:
                raise SystemExit(f"runtime missing {k}")
        # Check predictions contract like package_submission self-test
        import csv as _csv
        rows = list(_csv.DictReader(open(pred, newline="", encoding="utf-8")))
        gt_rows = list(_csv.DictReader(open(td / "out" / "ground_truth.csv", newline="", encoding="utf-8")))
        expected = len(gt_rows) if gt_rows else smoke_num
        if len(rows) != expected:
            raise SystemExit(f"predictions row count {len(rows)} != ground truth {expected} (requested {smoke_num}, generator rounds composition)")
        if list(rows[0].keys()) != ["pair_id", "x", "y", "theta", "scale", "found", "score"]:
            raise SystemExit(f"predictions wrong columns: {list(rows[0].keys())}")
        print(f"SELF-TEST PASSED: {len(rows)} rows (requested {smoke_num}), median {runtime['median_s_per_pair']:.2f}s, "
              f"est.core {data['estimated_core']:.1f}, report at {report}")

if __name__ == "__main__":
    main()
