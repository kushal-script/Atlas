"""Generate a Phase 2 training or validation suite mirroring the blind set.

The scored set is 200 organiser generated pairs: 70 nominal, 70 degraded over
four undisclosed severity levels, 40 with no true instance, and 20 optical
bonus pairs. This script emits the same composition at any size so thresholds
can be tuned and validated on data shaped like the data that will score them.

The organiser sample pairs are never read here or anywhere else in generation
or tuning: train and validation come from this generator alone, under disjoint
master seeds, and the sample pairs stay what they are for, a check of the io
contract.

    .venv/bin/python scripts/generate_phase2_suite.py --out data/p2train --num 240 --seed 3001
    .venv/bin/python scripts/generate_phase2_suite.py --out data/p2holdout --num 120 --seed 9001
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from drift_sense import __version__
from drift_sense.generator import generate_pair, save_pair
from drift_sense.params import GeneratorConfig

# The blind composition, as fractions of the whole.
MIX = (("A_nominal", 0.35), ("B_degraded", 0.35), ("C_absent", 0.20), ("D_optical", 0.10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num", type=int, default=240)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sets = []
    for name, frac in MIX:
        sets += [name] * int(round(args.num * frac))
    sets = sets[:args.num]
    rng.shuffle(sets)

    cfg = GeneratorConfig()
    cfg.phase2 = True
    rows, t0 = [], time.time()
    for i, set_name in enumerate(sets):
        style = "dram" if rng.random() < 0.5 else "finfet"
        severity = int(rng.integers(1, 5)) if set_name == "B_degraded" else 0
        absent = set_name == "C_absent"
        # Half the absent pairs are degraded across the same severity ladder.
        # With every absent pair clean, measured noise separates presence from
        # absence on this suite alone, and a decision fitted to that shortcut
        # would collapse on any blind set whose absent pairs are degraded too.
        if absent and rng.random() < 0.5:
            severity = int(rng.integers(1, 5))
        modality = "optical" if set_name == "D_optical" else "sem"
        t = time.time()
        result = generate_pair(seed=args.seed * 1_000_003 + i, style=style,
                               cfg=cfg, modality=modality,
                               absent=absent, degrade=severity)
        pair_dir = save_pair(out, i, result)
        meta = result["meta"]
        gt = meta["ground_truth"]
        rows.append({
            "pair_id": f"pair_{i:04d}",
            "set": set_name,
            "severity": severity,
            "reference_path": str((pair_dir / "reference.png").relative_to(out)),
            "search_path": str((pair_dir / "search.png").relative_to(out)),
            "style": style,
            "modality": modality,
            "found": meta["found"],
            "gt_x": f"{gt['x']:.3f}",
            "gt_y": f"{gt['y']:.3f}",
            "gt_zoom": f"{meta['zoom']:.5f}",
            "gt_rotation_deg": f"{meta['relative_rotation_deg']:.4f}",
            "seed": meta["seed"],
        })
        tag = "ABSENT" if absent else f"({gt['x']:.0f},{gt['y']:.0f})"
        print(f"pair_{i:04d} {set_name:10s} sev{severity} {style:6s} "
              f"zoom {meta['zoom']:6.3f} gt={tag:14s} {time.time()-t:.1f}s", flush=True)

    with open(out / "ground_truth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    comp = dict(Counter(r["set"] for r in rows))
    json.dump({"generator_version": __version__, "phase": 2, "seed": args.seed,
               "num_pairs": len(rows), "composition": comp,
               "note": "organiser sample pairs are not used here in any way"},
              open(out / "dataset_meta.json", "w"), indent=2)
    print(f"done, {len(rows)} pairs, {comp}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
