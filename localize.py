"""Drift Sense localization entry point.

Finds the high magnification reference pattern inside the wide search image and
prints the centre coordinates in search image pixels. The origin is the top
left pixel, x increases to the right and y increases downward, matching the
coordinate convention in the problem statement.

Single pair:
    python localize.py reference.png search.png
        prints one line, "x y"

    python localize.py reference.png search.png --json
        prints full diagnostics including confidence and runtime

Batch, a directory of pair folders (each holding reference.png and search.png):
    python localize.py --batch path/to/dataset --out predictions.csv

Batch, an explicit manifest CSV with reference_path and search_path columns:
    python localize.py --manifest pairs.csv --out predictions.csv

Batch, a flat directory of images paired by a shared prefix:
    python localize.py --batch path/to/images --pattern flat --out predictions.csv

No source code changes are needed for any of these forms.
"""

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from drift_sense.api import _regime
from drift_sense.localize import MatchConfig, load_gray, locate, optical_config

REPO = Path(__file__).resolve().parent
REF_NAMES = ("reference.png", "reference.tif", "reference.tiff", "ref.png")
SEARCH_NAMES = ("search.png", "search.tif", "search.tiff", "wide.png")


def _find(folder, names):
    for n in names:
        p = folder / n
        if p.exists():
            return p
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".bmp"):
            stem = p.stem.lower()
            if any(n.split(".")[0] in stem for n in names):
                return p
    return None


def discover_nested(root):
    pairs = []
    for folder in sorted(d for d in root.iterdir() if d.is_dir()):
        ref = _find(folder, REF_NAMES)
        search = _find(folder, SEARCH_NAMES)
        if ref and search:
            pairs.append((folder.name, ref, search))
    return pairs


def discover_flat(root):
    """Pair images in one directory by the token before the role keyword."""
    refs, searches = {}, {}
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in (".png", ".tif", ".tiff", ".jpg", ".bmp"):
            continue
        stem = p.stem.lower()
        for key, bucket in (("ref", refs), ("search", searches), ("wide", searches)):
            if key in stem:
                token = stem.replace(key, "").strip("_-. ") or p.stem
                bucket[token] = p
                break
    pairs = []
    for token in sorted(set(refs) & set(searches)):
        pairs.append((token, refs[token], searches[token]))
    return pairs


def read_manifest(path):
    pairs = []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            keys = {k.lower().strip(): k for k in row}
            rk = next((keys[k] for k in ("reference_path", "reference", "ref_path", "ref")
                       if k in keys), None)
            sk = next((keys[k] for k in ("search_path", "search", "wide_path", "wide")
                       if k in keys), None)
            if not rk or not sk:
                raise SystemExit("manifest needs reference_path and search_path columns")
            pid = row.get(keys.get("pair_id", ""), f"row_{i:04d}") or f"row_{i:04d}"
            ref, search = Path(row[rk]), Path(row[sk])
            if not ref.is_absolute():
                ref = (path.parent / ref).resolve()
            if not search.is_absolute():
                search = (path.parent / search).resolve()
            pairs.append((pid, ref, search))
    return pairs


def run_one(ref_path, search_path, use_reranker):
    ref, ref_rgb = load_gray(ref_path)
    search, search_rgb = load_gray(search_path)
    cfg = optical_config() if (ref_rgb or search_rgb) else MatchConfig()
    if use_reranker:
        weights = REPO / "models" / "reranker.npz"
        if not weights.exists():
            raise SystemExit(f"re-ranker weights not found at {weights}")
        cfg.reranker_path = str(weights)
    x, y, diag, _ = locate(ref, search, cfg)
    return x, y, diag


def confidence_regime(diag):
    """Single source of truth for the regime rule, shared with the API."""
    return _regime(diag)


def main():
    ap = argparse.ArgumentParser(
        description="Locate the reference pattern inside the search image")
    ap.add_argument("reference", type=Path, nargs="?")
    ap.add_argument("search", type=Path, nargs="?")
    ap.add_argument("--batch", type=Path, help="directory of pairs to process")
    ap.add_argument("--manifest", type=Path, help="CSV with reference_path and search_path")
    ap.add_argument("--pattern", choices=["nested", "flat"], default="nested",
                    help="batch layout: pair subfolders (default) or one flat directory")
    ap.add_argument("--out", type=Path, help="write batch predictions to this CSV")
    ap.add_argument("--json", action="store_true", help="full diagnostics for a single pair")
    ap.add_argument("--reranker", action="store_true",
                    help="use the learned re-ranker for the stage two decision")
    args = ap.parse_args()

    if args.batch or args.manifest:
        if args.manifest:
            pairs = read_manifest(args.manifest)
        elif args.pattern == "flat":
            pairs = discover_flat(args.batch)
        else:
            pairs = discover_nested(args.batch)
            if not pairs:
                pairs = discover_flat(args.batch)
        if not pairs:
            raise SystemExit("no reference and search image pairs found")

        rows, t0 = [], time.perf_counter()
        for pid, ref_path, search_path in pairs:
            x, y, diag = run_one(ref_path, search_path, args.reranker)
            rows.append({
                "pair_id": pid,
                "reference_path": str(ref_path),
                "search_path": str(search_path),
                "pred_x": f"{x:.3f}",
                "pred_y": f"{y:.3f}",
                "score": f"{diag['score']:.4f}",
                "confidence_regime": confidence_regime(diag),
                "num_candidates": diag["num_candidates"],
                "est_rotation_deg": f"{diag['theta_deg']:.3f}",
                "est_scale": f"{diag['scale']:.4f}",
                "runtime_s": f"{diag['runtime_s']:.3f}",
            })
            print(f"{pid} {x:.2f} {y:.2f} "
                  f"[{rows[-1]['confidence_regime']}] {diag['runtime_s']:.2f}s")
        total = time.perf_counter() - t0
        times = np.array([float(r["runtime_s"]) for r in rows])

        out = args.out or Path("predictions.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        env = {
            "pairs": len(rows),
            "total_wall_clock_s": round(total, 2),
            "mean_runtime_s_per_pair": round(float(times.mean()), 3),
            "median_runtime_s_per_pair": round(float(np.median(times)), 3),
            "timing_method": "time.perf_counter around the full locate call, "
                             "single process, no warm up excluded",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        }
        with open(out.with_suffix(".runtime.json"), "w") as fh:
            json.dump(env, fh, indent=2)
        print(f"\nwrote {out} and {out.with_suffix('.runtime.json')}")
        print(f"{len(rows)} pairs, mean {env['mean_runtime_s_per_pair']}s per pair "
              f"on {env['processor']}, Python {env['python']}")
        return

    if not args.reference or not args.search:
        ap.error("give reference and search paths, or use --batch or --manifest")
    x, y, diag = run_one(args.reference, args.search, args.reranker)
    if args.json:
        print(json.dumps({"x": x, "y": y,
                          "confidence_regime": confidence_regime(diag), **diag},
                         indent=2, default=str))
    else:
        print(f"{x:.2f} {y:.2f}")


if __name__ == "__main__":
    main()
