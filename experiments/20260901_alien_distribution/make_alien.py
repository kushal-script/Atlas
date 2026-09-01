"""A distribution family neither generator resembles, with exact labels.

The audit's honest residue was that seed disjoint is not distribution
disjoint. This suite attacks that directly: pairs are generated from the
released pipeline so the geometry and ground truth are exact, and the search
capture is then mutated by one of six appearance transformations chosen to
break the assumptions the Phase 2 adaptations lean on, not to flatter them:

  invert            contrast polarity flipped, the case their generator never renders
  gamma_down        tone curve 0.55, the opposite direction of their gamma 1.25
  dark_streaks      negative full width bands; the shipped corrector only lifts bright rows
  vertical_streaks  full height column bands; the corrector is row only and must not misfire
  tone_crush        dynamic range compressed into 70 gray levels with an offset
  heavy_speckle     multiplicative speckle at 0.5, past their ladder's 0.30

Every mutation is photometric, so gt x, y, theta and scale remain exact by
construction. Failure here is informative either way: graceful degradation
(lost credit, honest scores) is the designed behaviour, and any active
misfire (a corrector eating structure, an override firing wrongly) is a
defect this suite exists to catch before the blind set can.

    .venv/bin/python experiments/20260901_alien_distribution/make_alien.py \
        --seed 313233 --num 48 --out data/alien48
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "20260901_organiser_sample_validation"))

MUTATIONS = ["invert", "gamma_down", "dark_streaks", "vertical_streaks",
             "tone_crush", "heavy_speckle"]


def mutate(img, name, rng):
    x = img.astype(np.float32)
    if name == "invert":
        x = 255.0 - x
    elif name == "gamma_down":
        x = 255.0 * np.power(np.clip(x / 255.0, 0, 1), 0.55)
    elif name == "dark_streaks":
        n = rng.poisson(25)
        for _ in range(n):
            r0 = int(rng.integers(0, x.shape[0] - 5))
            h = int(rng.integers(1, 5))
            x[r0:r0 + h] -= float(rng.uniform(0.5, 1.0)) * 60.0
    elif name == "vertical_streaks":
        n = rng.poisson(25)
        for _ in range(n):
            c0 = int(rng.integers(0, x.shape[1] - 5))
            w = int(rng.integers(1, 5))
            x[:, c0:c0 + w] += float(rng.uniform(0.5, 1.0)) * 60.0
    elif name == "tone_crush":
        x = 90.0 + (x - x.min()) / max(x.max() - x.min(), 1.0) * 70.0
    elif name == "heavy_speckle":
        x = x * rng.normal(1.0, 0.5, x.shape).clip(0.05, 3.0)
    return np.clip(x, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num", type=int, default=48)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import cv2
    from generate_from_organiser import ARCHS, SEVERITY
    sys.path.insert(0, str(REPO / "OneDrive_1_1-9-2026" / "AMP_Phase 2 material" / "generator"))
    from src.phase2_pipeline import Phase2Params, generate_phase2_sample

    args.out.mkdir(parents=True, exist_ok=True)
    master = np.random.default_rng(args.seed)
    n_present = round(args.num * 0.8)
    rows = []
    for i in range(args.num):
        present = i < n_present
        sev = [0, 1, 2][i % 3]
        zoom = float(master.uniform(8.0, 12.0))
        theta = float(master.uniform(-5.0, 5.0))
        arch = ARCHS[int(master.integers(0, len(ARCHS)))]
        mut = MUTATIONS[i % len(MUTATIONS)]
        pid = f"a{i:03d}"
        t0 = time.time()
        rng = np.random.default_rng(args.seed + i * 7919)
        params = Phase2Params(zoom=zoom, theta_deg=theta, present=True,
                              boundary_bias=0.70, **SEVERITY[sev])
        s = generate_phase2_sample(arch, params, rng)
        ref, srch, gt = s["reference_img"], s["search_img"], s["gt"]
        if not present:
            rng2 = np.random.default_rng(args.seed + i * 7919 + 3571)
            s2 = generate_phase2_sample(arch, params, rng2)
            ref = s2["reference_img"]
            gt = {"present": 0, "x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0}
        srch = mutate(srch, mut, np.random.default_rng(args.seed + i * 104729))
        (args.out / pid).mkdir(exist_ok=True)
        cv2.imwrite(str(args.out / pid / "reference.png"), ref)
        cv2.imwrite(str(args.out / pid / "search.png"), srch)
        rows.append({"pair_id": pid, "set": "A_nominal" if present else "C_absent",
                     "severity": sev, "mutation": mut,
                     "reference_path": f"{pid}/reference.png",
                     "search_path": f"{pid}/search.png",
                     "style": "dram" if arch.startswith("dram") else "finfet",
                     "modality": "sem", "found": int(gt["present"]),
                     "gt_x": gt["x"], "gt_y": gt["y"],
                     "gt_zoom": zoom, "gt_rotation_deg": theta,
                     "seed": args.seed + i * 7919})
        print(f"{pid} {mut:16s} sev{sev} present {int(gt['present'])} "
              f"{time.time() - t0:4.1f}s", flush=True)
    with open(args.out / "ground_truth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
