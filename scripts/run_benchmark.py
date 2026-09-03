"""Cross-platform benchmark driver for drift-sense.

Generates a Phase-2 suite, times register_pair per pair, scores via the
Phase-2 scorer, and writes a portable report for cross-device comparison.

Usage (called via benchmark.sh / benchmark.ps1, or directly):
    .venv/bin/python scripts/run_benchmark.py --quick
    .venv/bin/python scripts/run_benchmark.py --full --seed 999
    .venv/bin/python scripts/run_benchmark.py --num 40 --seed 1 --out results/benchmark_manual

Modes:
    --quick  40 pairs (~70s) for ad-hoc device check
    --full  120 pairs (~200s) for headline numbers (README 76.6-81.7 band)
    --num N  overrides quick/full
Requires internet only for pip install (done by wrapper); this driver is offline.
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from drift_sense.api import load_presence_model, register_pair
from drift_sense.localize import MatchConfig, load_gray, optical_config


def _cpu_mem():
    cpu = mem = "not detected"
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5).stdout.strip() or cpu
            raw = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            if raw.isdigit():
                mem = f"{int(raw) / 1024 ** 3:.0f} GiB"
        elif sys.platform.startswith("linux"):
            try:
                txt = Path("/proc/cpuinfo").read_text()
                for line in txt.splitlines():
                    if "model name" in line:
                        cpu = line.split(":", 1)[1].strip()
                        break
                txt2 = Path("/proc/meminfo").read_text()
                for line in txt2.splitlines():
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        mem = f"{kb / 1024 / 1024:.0f} GiB"
                        break
            except Exception:
                try:
                    cpu = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
                except Exception:
                    pass
        elif sys.platform == "win32":
            # Try powershell Get-CimInstance, fallback to wmic
            try:
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                    capture_output=True, text=True, timeout=5)
                if out.stdout.strip():
                    cpu = out.stdout.strip().splitlines()[0].strip()
            except Exception:
                pass
            if cpu == "not detected":
                try:
                    out = subprocess.run(["wmic", "cpu", "get", "name"],
                                         capture_output=True, text=True, timeout=5)
                    lines = [l.strip() for l in out.stdout.splitlines() if l.strip() and l.strip().lower() != "name"]
                    if lines:
                        cpu = lines[0]
                except Exception:
                    pass
            try:
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"],
                    capture_output=True, text=True, timeout=5)
                if out.stdout.strip().isdigit():
                    mem = f"{out.stdout.strip()} GiB"
                else:
                    # wmic fallback
                    out2 = subprocess.run(["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                                          capture_output=True, text=True, timeout=5)
                    for line in out2.stdout.splitlines():
                        line=line.strip()
                        if line.isdigit():
                            mem = f"{int(line)/1024**3:.0f} GiB"
                            break
            except Exception:
                pass
            if cpu == "not detected":
                cpu = platform.processor() or cpu
    except Exception:
        pass
    return cpu, mem


def _versions():
    import importlib
    out = {}
    for pkg in ["numpy", "scipy", "PIL", "cv2"]:
        try:
            mod = importlib.import_module("cv2" if pkg == "cv2" else pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except Exception as e:
            out[pkg] = f"missing: {e}"
    out["python"] = platform.python_version()
    out["platform"] = platform.platform()
    return out


def _pct(arr):
    if len(arr) == 0:
        return {}
    a = np.asarray(arr, float)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max()),
            "min": float(a.min()), "mean": float(a.mean())}


def _score(truth_path, preds):
    # Mirrors scripts/score_predictions.py logic exactly for consistency
    truth = {r["pair_id"]: r for r in csv.DictReader(open(truth_path, newline="", encoding="utf-8-sig"))
             if r["modality"] == "sem"}
    pred_map = {r["pair_id"]: r for r in preds}
    missing = set(truth) - set(pred_map)
    if missing:
        print(f"WARNING: missing rows for {sorted(missing)[:5]}", file=sys.stderr)

    loc = {"A_nominal": [], "B_degraded": []}
    pose_s, pose_r = [], []
    scores, labels = [], []
    by_sev = {}
    raw_scale, raw_rot, raw_err = [], [], []
    for pid, t in truth.items():
        p = pred_map.get(pid)
        t_found = int(t["found"]) == 1
        p_found = p is not None and str(p["found"]) == "1"
        ok = False
        if t_found:
            err = (np.hypot(float(p["x"]) - float(t["gt_x"]),
                            float(p["y"]) - float(t["gt_y"])) if p_found else 1e9)
            c = 1.0 if err <= 1 else 0.8 if err <= 2 else 0.6 if err <= 3 else 0.4 if err <= 5 else 0.0
            loc[t["set"]].append(c)
            if c > 0:
                zerr = abs(float(p["scale"]) - float(t["gt_zoom"])) / float(t["gt_zoom"]) * 100
                rerr = abs(float(p["theta"]) - float(t["gt_rotation_deg"]))
                pose_s.append(1.0 if zerr <= 1 else 0.6 if zerr <= 2 else 0.3 if zerr <= 5 else 0.0)
                pose_r.append(1.0 if rerr <= 0.25 else 0.6 if rerr <= 0.5 else 0.3 if rerr <= 1.0 else 0.0)
                raw_scale.append(zerr); raw_rot.append(rerr)
            if p_found and err < 1e8:
                raw_err.append(err)
            key = f"{t['set']}/sev{t['severity']}"
            by_sev.setdefault(key, []).append(c)
            ok = p_found and err <= 5
        else:
            ok = not p_found
        scores.append(float(p["score"]) if p and p["score"] != "" else 0.0)
        labels.append(1 if ok else 0)

    tp = sum(1 for pid, t in truth.items()
             if int(t["found"]) == 0 and pred_map.get(pid, {}).get("found") in ("0", 0))
    fp = sum(1 for pid, t in truth.items()
             if int(t["found"]) == 1 and str(pred_map.get(pid, {}).get("found")) == "0")
    fn = sum(1 for pid, t in truth.items()
             if int(t["found"]) == 0 and str(pred_map.get(pid, {}).get("found")) == "1")
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    credit_A = float(np.mean(loc["A_nominal"])) if loc["A_nominal"] else 0.0
    credit_B = float(np.mean(loc["B_degraded"])) if loc["B_degraded"] else 0.0
    loc_pts = 40 * (0.45 * credit_A + 0.55 * credit_B)
    pose_pts = (10 * float(np.mean(pose_s)) + 10 * float(np.mean(pose_r))) if pose_s else 0.0

    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = [i for i, l in enumerate(labels) if l]; neg = [i for i, l in enumerate(labels) if not l]
    auc = ((sum(ranks[i] for i in pos) - len(pos) * (len(pos) + 1) / 2)
           / max(len(pos) * len(neg), 1)) if pos and neg else 1.0

    sc = np.asarray(scores, float); lb = np.asarray(labels, float)
    brier = float(np.mean((sc - lb) ** 2)) if len(sc) else 0.0
    base = float(lb.mean()) if len(lb) else 0.0
    brier_ref = float(np.mean((base - lb) ** 2)) if len(lb) else 0.0
    rel = res = 0.0
    if len(sc):
        edges = np.linspace(0.0, 1.0 + 1e-9, 11)
        idx = np.clip(np.digitize(sc, edges) - 1, 0, 9)
        for b in range(10):
            m = idx == b
            if not m.any():
                continue
            w = m.sum() / len(sc)
            rel += w * (sc[m].mean() - lb[m].mean()) ** 2
            res += w * (lb[m].mean() - base) ** 2
    cdf = {f"within_{q}px": float(np.mean(np.asarray(raw_err) <= q)) if raw_err else 0.0
           for q in (0.5, 1.0, 2.0, 5.0)}

    rep = {"pairs": len(truth),
           "localization": {"credit_A": credit_A, "credit_B": credit_B, "points": loc_pts},
           "pose": {"scale_credit": float(np.mean(pose_s)) if pose_s else 0,
                    "rotation_credit": float(np.mean(pose_r)) if pose_r else 0,
                    "points": pose_pts},
           "rejection": {"f1": f1, "precision": prec, "recall": rec,
                         "tp": tp, "fp": fp, "fn": fn, "points": 15 * f1},
           "calibration": {"auc": float(auc), "points": 10 * float(auc),
                           "brier": brier, "brier_vs_base_rate": brier_ref,
                           "brier_reliability": rel, "brier_resolution": res},
           "per_severity_credit": {k: {"n": len(v), "credit": float(np.mean(v))}
                                   for k, v in sorted(by_sev.items())},
           "error_distribution": {
               "localization_px": _pct(raw_err),
               "scale_error_pct": _pct(raw_scale),
               "rotation_error_deg": _pct(raw_rot),
               "localization_cdf": cdf},
           "estimated_core": loc_pts + pose_pts + 15 * f1 + 10 * float(auc)}
    return rep


def main():
    ap = argparse.ArgumentParser(description="Benchmark drift-sense on this device")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="40 pairs quick check (~70s)")
    g.add_argument("--full", action="store_true", help="120 pairs full headline (~200s)")
    ap.add_argument("--num", type=int, default=None, help="override pair count")
    ap.add_argument("--seed", type=int, default=999, help="suite master seed")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default results/benchmark_<ts>)")
    ap.add_argument("--suite", type=Path, default=None, help="reuse existing suite dir (skip generation)")
    ap.add_argument("--keep-suite", action="store_true", help="keep generated suite (default: keep)")
    args = ap.parse_args()

    if args.num is not None:
        num = args.num
        mode = f"custom_{num}"
    elif args.full:
        num = 120
        mode = "full"
    elif args.quick:
        num = 40
        mode = "quick"
    else:
        # default quick
        num = 40
        mode = "quick"

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out.resolve() if args.out else (REPO / "results" / f"benchmark_{ts}_{mode}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_dir = args.suite.resolve() if args.suite else (out_dir / "suite")
    # Generate suite if not provided
    if args.suite is None:
        print(f"[benchmark] generating {num}-pair suite (seed={args.seed}) at {suite_dir} ...", flush=True)
        cmd = [sys.executable, str(REPO / "scripts" / "generate_phase2_suite.py"),
               "--out", str(suite_dir), "--num", str(num), "--seed", str(args.seed)]
        print("  " + " ".join(cmd), flush=True)
        res = subprocess.run(cmd, cwd=str(REPO))
        if res.returncode != 0:
            raise SystemExit(f"suite generation failed ({res.returncode})")
    else:
        if not (suite_dir / "ground_truth.csv").exists():
            raise SystemExit(f"--suite {suite_dir} has no ground_truth.csv")

    gt_csv = suite_dir / "ground_truth.csv"
    truth_rows = list(csv.DictReader(open(gt_csv, newline="", encoding="utf-8-sig")))
    print(f"[benchmark] suite: {len(truth_rows)} pairs, composition {json.dumps({k: sum(1 for r in truth_rows if r['set']==k) for k in sorted(set(r['set'] for r in truth_rows))})}")

    # System info
    cpu, mem = _cpu_mem()
    vers = _versions()
    print(f"[benchmark] system: {vers['platform']} | {cpu} | {mem} | Python {vers['python']} numpy {vers.get('numpy')} scipy {vers.get('scipy')} cv2 {vers.get('cv2')}", flush=True)

    # Load model + configs
    model = None
    try:
        model = load_presence_model()
        print(f"[benchmark] presence model loaded: threshold {model.get('prob_threshold')}")
    except Exception as e:
        print(f"[benchmark] WARNING: presence model not loaded ({e}), using fallback", file=sys.stderr, flush=True)

    cfg = MatchConfig()
    cfg_optical = optical_config()

    # Timed registration loop (mirrors register.py per-pair budget)
    pred_path = out_dir / "predictions.csv"
    timings = []
    preds = []
    budget_gated = 0
    # ensure incremental flush like register.py
    with open(pred_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pair_id", "x", "y", "theta", "scale", "found", "score"])
        writer.writeheader()
        fh.flush()
        for r in truth_rows:
            pid = r["pair_id"]
            ref_p = suite_dir / r["reference_path"]
            search_p = suite_dir / r["search_path"]
            t_pair = time.perf_counter()
            try:
                ref, ref_rgb = load_gray(ref_p)
                search, search_rgb = load_gray(search_p)
                result = register_pair(ref, search, reference_rgb=ref_rgb, search_rgb=search_rgb,
                                       model=model, config=cfg, optical=cfg_optical, t_start=t_pair)
                if result.diagnostics.get("budget_gated"):
                    budget_gated += 1
                row = result.as_row(pid)
            except Exception as exc:
                print(f"WARNING: {pid} failed ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
                row = {"pair_id": pid, "x": 0, "y": 0, "theta": 0, "scale": 0, "found": 0, "score": "0.50000"}
            dt = time.perf_counter() - t_pair
            timings.append(dt)
            preds.append(row)
            writer.writerow(row)
            fh.flush()
            print(f"  {pid} found={row['found']} score={row['score']} {dt:.2f}s", flush=True)

    # Write timings json
    t_arr = np.array(timings)
    runtime = {
        "median_s_per_pair": float(np.median(t_arr)),
        "mean_s_per_pair": float(np.mean(t_arr)),
        "p90_s": float(np.percentile(t_arr, 90)),
        "max_s": float(np.max(t_arr)),
        "min_s": float(np.min(t_arr)),
        "over_5s_pct": float(np.mean(t_arr > 5.0) * 100),
        "requirement_median_s": 5.0,
        "hard_timeout_s": 20.0,
        "budget_gated_pairs": budget_gated,
        "num_pairs": len(timings),
        "per_pair_s": [float(x) for x in timings],
        "protocol": "register_pair per-pair wall clock via time.perf_counter, sole occupant, includes generation-time not in this span; suite generation timed separately",
    }
    print(f"[benchmark] runtime median {runtime['median_s_per_pair']:.2f}s mean {runtime['mean_s_per_pair']:.2f}s p90 {runtime['p90_s']:.2f}s max {runtime['max_s']:.2f}s over5s {runtime['over_5s_pct']:.1f}%", flush=True)
    if runtime["median_s_per_pair"] <= 5.0:
        print(f"[benchmark] median PASS (<=5.0s)")
    else:
        print(f"[benchmark] median FAIL (>5.0s)", file=sys.stderr)
    if runtime["max_s"] <= 20.0:
        print(f"[benchmark] hard timeout PASS (max <=20s)")
    else:
        print(f"[benchmark] hard timeout FAIL (max >20s)", file=sys.stderr)

    # Score (SEM pairs only, like organiser)
    score = _score(gt_csv, preds)
    print(json.dumps(score, indent=2))

    # Build final report
    pip_freeze = ""
    try:
        pip_freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                    capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        pass

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "num_pairs": num,
        "seed": args.seed,
        "suite_dir": str(suite_dir),
        "composition": {k: sum(1 for r in truth_rows if r["set"] == k) for k in sorted(set(r["set"] for r in truth_rows))},
        "system": {"cpu": cpu, "memory": mem, **vers},
        "runtime": runtime,
        **score,
        "pip_freeze": pip_freeze.splitlines()[:80],
    }
    # enrich runtime with reference comparison
    try:
        ref_protocol = json.loads((REPO / "results" / "runtime_protocol.json").read_text())
        report["reference_protocol"] = ref_protocol.get("phase2", ref_protocol)
    except Exception:
        pass

    (out_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "score.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    (out_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    # also copy ground_truth for reproducibility
    try:
        import shutil as _sh
        _sh.copy(gt_csv, out_dir / "ground_truth.csv")
    except Exception:
        pass

    # Markdown summary
    md = []
    md.append(f"# Drift-Sense Benchmark — {mode} ({num} pairs, seed {args.seed})")
    md.append("")
    md.append(f"Generated at {report['generated_at']} on `{vers['platform']}`")
    md.append(f"CPU: {cpu} | Memory: {mem} | Python {vers['python']} | numpy {vers.get('numpy')} scipy {vers.get('scipy')} cv2 {vers.get('cv2')}")
    md.append("")
    md.append(f"Suite: `{suite_dir}` composition `{report['composition']}`")
    md.append("")
    md.append("## Runtime")
    md.append("")
    md.append(f"| median | mean | p90 | max | over 5s | budget_gated | requirement |")
    md.append(f"|---|---|---|---|---|---|---|")
    md.append(f"| {runtime['median_s_per_pair']:.2f}s | {runtime['mean_s_per_pair']:.2f}s | {runtime['p90_s']:.2f}s | {runtime['max_s']:.2f}s | {runtime['over_5s_pct']:.1f}% | {budget_gated} | median 5.0s, hard 20s |")
    md.append("")
    median_pass = "PASS" if runtime["median_s_per_pair"] <= 5.0 else "FAIL"
    max_pass = "PASS" if runtime["max_s"] <= 20.0 else "FAIL"
    md.append(f"Median: **{median_pass}**, Hard timeout: **{max_pass}**")
    md.append("")
    md.append("## Score (SEM pairs, organiser scheme)")
    md.append("")
    md.append(f"| Loc A (nominal) | Loc B (degraded) | Loc pts /40 | Pose /20 | Reject F1 | AUC | Est. core /85 |")
    md.append(f"|---|---|---|---|---|---|---|")
    loc = score["localization"]; pose = score["pose"]; rej = score["rejection"]; cal = score["calibration"]
    md.append(f"| {loc['credit_A']:.3f} | {loc['credit_B']:.3f} | {loc['points']:.2f} | {pose['points']:.2f} | {rej['f1']:.3f} | {cal['auc']:.3f} | {score['estimated_core']:.2f} |")
    md.append("")
    md.append("### Per-severity credit")
    md.append("")
    md.append("| tier | n | credit |")
    md.append("|---|---|---|")
    for k, v in sorted(score["per_severity_credit"].items()):
        md.append(f"| {k} | {v['n']} | {v['credit']:.3f} |")
    md.append("")
    md.append(f"Predictions: `{pred_path}` | Full report: `benchmark_report.json`")
    md.append("")
    md.append("## How to compare across devices")
    md.append("")
    md.append("Commit `benchmark_report.json` from each device and diff `runtime.median_s_per_pair`, `runtime.p90_s`, `runtime.max_s`, `estimated_core`, `rejection.f1`, `calibration.auc`. Headline full-suite numbers should sit in the 76.6–81.7 band for `full` (120) and scale similarly for `quick` (40).")
    md.append("")
    (out_dir / "benchmark_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[benchmark] wrote {out_dir / 'benchmark_report.json'}", flush=True)
    print(f"[benchmark] wrote {out_dir / 'benchmark_report.md'}", flush=True)
    print(f"[benchmark] wrote {pred_path} ({len(preds)} rows)", flush=True)
    # exit code reflects runtime gate only, not accuracy
    if runtime["max_s"] > 20.0:
        print("[benchmark] WARNING: at least one pair exceeded 20s hard timeout — would score 0 on that pair on the scorer", file=sys.stderr)
    # do not fail on median exceed; just report
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
