"""Phase 2 mixed dataset generator (Day 3, T2).

Builds a mixed set of pairs for tuning and validating the Phase 2 localizer and
its rejection decision:

  * present pairs  -- generated with phase2=True, so the unknown zoom spans
    8..12x (scale error +/-0.20) and rotation spans +/-5 deg. The reference
    structure genuinely appears in the search image.
  * absent pairs (Set C) -- generated with absent=True: the reference structure
    is placed outside the search frame, so there is NO true instance. The search
    still shows unrelated layout, so a matcher may emit a spurious peak that the
    rejection rule must suppress.

A manifest CSV is written alongside the pair directories carrying, per row:
pair_id, reference, search, present (0/1), gt_x, gt_y. register.py consumes the
reference/search/pair_id columns; the scorer (T5) consumes present/gt_x/gt_y.

Usage:
    python scripts/generate_phase2_dataset.py --num 200 --out data/phase2_mixed --seed 7
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from drift_sense.generator import generate_pair, save_pair   # noqa: E402
from drift_sense.params import GeneratorConfig               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--present-frac", type=float, default=0.8,
                    help="fraction of pairs that contain a true instance (Set C is the rest)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--style", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cfg = GeneratorConfig()

    rows = []
    present_n = absent_n = 0
    for i in range(args.num):
        is_present = rng.random() < args.present_frac
        res = generate_pair(i, args.style, cfg, modality="sem",
                            absent=not is_present, phase2=True)
        save_pair(args.out, i, res, preview=args.preview)
        meta = res["meta"]
        gt = meta["ground_truth"]
        rows.append({
            "pair_id": f"pair_{i:04d}",
            "reference": f"pair_{i:04d}/reference.png",
            "search": f"pair_{i:04d}/search.png",
            "present": 1 if is_present else 0,
            "gt_x": ("" if gt["x"] is None else f"{gt['x']:.3f}"),
            "gt_y": ("" if gt["y"] is None else f"{gt['y']:.3f}"),
        })
        if is_present:
            present_n += 1
        else:
            absent_n += 1

    manifest = args.out / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair_id", "reference", "search",
                                           "present", "gt_x", "gt_y"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {args.num} pairs to {args.out}")
    print(f"  present (Set A/B): {present_n}")
    print(f"  absent  (Set C):   {absent_n}")
    print(f"  manifest: {manifest}")


if __name__ == "__main__":
    main()
