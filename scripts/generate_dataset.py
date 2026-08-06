"""Dataset generation CLI.

Usage:
    python scripts/generate_dataset.py --style mixed --num 40 --out data/train --seed 7 --previews
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drift_sense import __version__
from drift_sense.generator import generate_pair, save_pair


def main():
    ap = argparse.ArgumentParser(description="Generate SEM style reference and search image pairs")
    ap.add_argument("--style", choices=["dram", "finfet", "mixed"], default="mixed")
    ap.add_argument("--modality", choices=["sem", "optical"], default="sem")
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--previews", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for i in range(args.num):
        style = args.style if args.style != "mixed" else ("dram" if i % 2 == 0 else "finfet")
        t = time.time()
        result = generate_pair(seed=args.seed * 1_000_003 + i, style=style,
                               modality=args.modality)
        save_pair(args.out, i, result, preview=args.previews)
        gt = result["meta"]["ground_truth"]
        rows.append({
            "pair_id": f"pair_{i:04d}",
            "style": style,
            "gt_x": f"{gt['x']:.3f}",
            "gt_y": f"{gt['y']:.3f}",
            "relative_rotation_deg": f"{result['meta']['relative_rotation_deg']:.4f}",
            "search_scale_error": f"{result['meta']['search_scale_error']:.5f}",
            "placement": result["meta"]["placement"],
            "search_dose_e": f"{result['meta']['search_capture']['settings'].get('dose_e', result['meta']['search_capture']['settings'].get('photon_dose', 0.0)):.1f}",
        })
        print(f"pair_{i:04d} {style:7s} gt=({gt['x']:.1f}, {gt['y']:.1f}) "
              f"{result['meta']['placement']:13s} {time.time() - t:.1f}s")

    with open(args.out / "ground_truth.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out / "dataset_meta.json", "w") as fh:
        json.dump({"generator_version": __version__, "style": args.style,
                   "num_pairs": args.num, "seed": args.seed}, fh, indent=2)
    print(f"done, {args.num} pairs in {time.time() - t0:.1f}s, output {args.out}")


if __name__ == "__main__":
    main()
