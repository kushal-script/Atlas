"""Drift Sense dataset generation entry point, DRAM only.

This variant builds one architecture: periodic horizontal word lines and
vertical bit lines crossing at right angles with a contact dot at every
intersection. There is no style option, because there is no second style to
choose between.

Three independent generators are available, deliberately sharing no image
formation code, so the localizer can be measured under domain shift rather
than only against the assumptions it was tuned on.

    physics   the primary generator: a specimen material and height map is
              built once per pair and imaged twice through a secondary
              electron model (tilt dependent yield, beam optics, dose driven
              shot noise, charging, scan artifacts). Supports SEM grayscale
              and RGB optical brightfield.
    spec      an independent reimplementation of the organiser specification
              and starter Space parameters: coarse structure presets, 9:1 to
              11:1 magnification, and the full published degradation list.
    stress    an adversarial generator with painted edges, plain Gaussian
              noise and area averaged downsampling.

Examples:
    python generate_dataset.py --generator physics --num 40 \
        --out data/train40 --seed 1 --previews
    python generate_dataset.py --generator physics --modality optical --num 30 \
        --out data/optical30 --seed 33
    python generate_dataset.py --generator spec --num 40 --out data/spec40 --seed 11
    python generate_dataset.py --generator stress --num 30 --out data/stress30 --seed 5

Every pair folder contains reference.png, search.png and meta.json holding the
random seed, structure parameters, every transformation and noise setting, and
the exact ground truth centre. A manifest CSV summarises the dataset.
"""

import argparse
import csv
import json
import subprocess
import sys
import time

import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from drift_sense_dram import __version__
from drift_sense_dram.generator import generate_pair, save_pair
from drift_sense_dram.params import GeneratorConfig


def _physics(args):
    rows = []
    t0 = time.time()
    cfg = GeneratorConfig()
    cfg.phase2 = bool(args.phase2)
    # Which pairs carry no true instance is decided up front from the master
    # seed, so the absent set is reproducible and evenly spread rather than
    # clustered at the end of the run.
    absent_flags = [False] * args.num
    if args.phase2 and args.absent_fraction > 0:
        rng = np.random.default_rng(args.seed * 7_919 + 11)
        n_absent = int(round(args.num * args.absent_fraction))
        for idx in rng.choice(args.num, size=n_absent, replace=False):
            absent_flags[int(idx)] = True
    for i in range(args.num):
        t = time.time()
        result = generate_pair(seed=args.seed * 1_000_003 + i,
                               modality=args.modality, cfg=cfg,
                               absent=absent_flags[i])
        pair_dir = save_pair(args.out, i, result, preview=args.previews)
        meta = result["meta"]
        gt = meta["ground_truth"]
        settings = meta["search_capture"]["settings"]
        rows.append({
            "pair_id": f"pair_{i:04d}",
            "reference_path": str((pair_dir / "reference.png").relative_to(args.out)),
            "search_path": str((pair_dir / "search.png").relative_to(args.out)),
            "style": meta["style"],
            "modality": args.modality,
            "found": meta.get("found", 1),
            "gt_x": f"{gt['x']:.3f}",
            "gt_y": f"{gt['y']:.3f}",
            "gt_zoom": f"{meta.get('zoom', 10.0):.5f}",
            "relative_rotation_deg": f"{meta['relative_rotation_deg']:.4f}",
            "search_scale_error": f"{meta['search_scale_error']:.5f}",
            "placement": meta["placement"],
            "search_dose_e": f"{settings.get('dose_e', settings.get('photon_dose', 0.0)):.1f}",
            "seed": meta["seed"],
        })
        tag = "ABSENT" if meta.get("found", 1) == 0 else f"({gt['x']:.0f}, {gt['y']:.0f})"
        print(f"pair_{i:04d} {meta['style']:7s} zoom {meta.get('zoom', 10.0):6.3f} "
              f"rot {meta['relative_rotation_deg']:+6.2f} gt={tag:16s} {time.time() - t:.1f}s")
    return rows, time.time() - t0


def _delegate(script, args, extra):
    cmd = [sys.executable, str(REPO / "scripts" / script),
           "--num", str(args.num), "--out", str(args.out), "--seed", str(args.seed)] + extra
    subprocess.run(cmd, check=True, cwd=REPO)
    manifest = args.out / "ground_truth.csv"
    if not manifest.exists():
        raise SystemExit(f"{script} wrote no ground_truth.csv under {args.out}")
    rows = list(csv.DictReader(open(manifest)))
    for r in rows:
        r.setdefault("reference_path", f"{r.get('pair_id', '')}/reference.png")
        r.setdefault("search_path", f"{r.get('pair_id', '')}/search.png")
    return rows, 0.0


def main():
    ap = argparse.ArgumentParser(
        description="Generate reference and search image pairs with exact ground truth")
    ap.add_argument("--generator",
                    choices=["physics", "spec", "amat_proxy", "stress"],
                    default="physics")
    ap.add_argument("--modality", choices=["sem", "optical"], default="sem",
                    help="physics generator only: grayscale SEM or RGB optical brightfield")
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase2", action="store_true",
                    help="draw the zoom ratio and rotation from the disclosed "
                         "Phase 2 ranges instead of jittering around ten to one")
    ap.add_argument("--absent_fraction", type=float, default=0.0,
                    help="fraction of pairs whose reference has no instance in "
                         "the search image, taken from a second specimen of the "
                         "same architecture; Phase 2 only")
    ap.add_argument("--previews", action="store_true",
                    help="also write search images with the ground truth region drawn on")
    ap.add_argument("--barrel_max", type=float, default=0.02,
                    help="spec generator only: radial distortion amplitude")
    ap.add_argument("--tier", default="cycle",
                    help="amat_proxy only: low, medium, high, severe, cycle or variants")
    ap.add_argument("--rotation_deg", type=float, default=0.0,
                    help="amat_proxy only: robustness rotation amplitude")
    ap.add_argument("--scale_jitter", type=float, default=0.0,
                    help="amat_proxy only: robustness magnification error amplitude")
    args = ap.parse_args()
    # Delegated generators run with the repository as their working directory,
    # so a relative output path would resolve differently for parent and child
    # and the pairs would land somewhere the caller never asked for.
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.generator == "physics":
        rows, elapsed = _physics(args)
    elif args.generator == "spec":
        rows, elapsed = _delegate("generate_starter_spec_dataset.py", args,
                                  ["--barrel_max", str(args.barrel_max)])
    elif args.generator == "amat_proxy":
        rows, elapsed = _delegate("generate_amat_proxy.py", args,
                                  ["--tier", args.tier,
                                   "--rotation_deg", str(args.rotation_deg),
                                   "--scale_jitter", str(args.scale_jitter)])
    else:
        rows, elapsed = _delegate("generate_stress_dataset.py", args, [])

    # Only the physics generator builds the DRAM layout; the others carry their
    # own fixed structure, so recording dram for them would put a claim in the
    # provenance record that the pixels do not support.
    styles = sorted({r["style"] for r in rows if r.get("style")})
    if args.generator == "physics":
        recorded_style = "dram"
    elif styles:
        recorded_style = styles[0] if len(styles) == 1 else "mixed"
    else:
        recorded_style = "generator defined"

    if rows:
        with open(args.out / "ground_truth.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(args.out / "dataset_meta.json", "w") as fh:
        json.dump({"generator_version": __version__, "generator": args.generator,
                   "style": recorded_style, "style_requested": "dram",
                   "style_honoured": args.generator == "physics",
                   "modality": args.modality,
                   "num_pairs": args.num, "seed": args.seed,
                   "coordinate_convention": "origin at centre of top left pixel, "
                                            "x increases right, y increases down"},
                  fh, indent=2)
    print(f"done, {len(rows)} pairs, output {args.out}"
          + (f", {elapsed:.1f}s" if elapsed else ""))


if __name__ == "__main__":
    main()
