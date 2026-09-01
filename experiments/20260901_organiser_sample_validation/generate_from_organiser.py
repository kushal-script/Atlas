"""Generate fresh suites from the organisers' own generator under new seeds.

The shared bundle carries the full Phase 2 generation pipeline and a
deterministic driver. This wrapper reuses that pipeline under new seeds in
two recipes: one faithful to the shared 20 pair sample, and one matching the
jury recommendation recorded in the bundle's readme for the real set, Set A
raised to severity 1 and Set B shifted toward severities 3 and 4.

Standing of the data this produces: it is this team's own generated data,
which the addendum explicitly allows for augmentation, retraining and
threshold tuning, and the addendum's own FAQ says regenerating a dataset is
needed to tune the found threshold. The line that must never be crossed is
different and stands: the organisers' SHARED pairs themselves are never
fitted on, only validated against. Suites produced here for fitting always
use seeds disjoint from suites used for judging.

Runs only when the shared bundle is present beside the repository; nothing
from the bundle is copied into the repository.

    .venv/bin/python experiments/20260901_organiser_sample_validation/generate_from_organiser.py \
        --recipe sample --num 40 --seed 424242 --out local/amatgen_sample40
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
GEN = REPO / "OneDrive_1_1-9-2026" / "AMP_Phase 2 material" / "generator"
sys.path.insert(0, str(GEN))

from src.phase2_pipeline import (Phase2Params, generate_phase2_sample,
                                 to_optical_rgb)

SEVERITY = {
    0: dict(dose_search=300.0, shear_amplitude_px=1.0, drift_jitter_px=0.30,
            detector_noise_sigma_search=4.0),
    1: dict(dose_search=150.0, shear_amplitude_px=1.5, drift_jitter_px=0.45,
            detector_noise_sigma_search=6.0, speckle_sigma=0.06),
    2: dict(dose_search=90.0, shear_amplitude_px=2.0, drift_jitter_px=0.65,
            detector_noise_sigma_search=8.0, charging_streak_prob=1.5,
            charging_streak_intensity=1.2, speckle_sigma=0.11,
            vignette_strength=0.10),
    3: dict(dose_search=55.0, shear_amplitude_px=2.5, drift_jitter_px=0.85,
            detector_noise_sigma_search=10.0, charging_streak_prob=3.0,
            charging_streak_intensity=2.0, speckle_sigma=0.19,
            salt_pepper_prob=0.005, astigmatism_ratio=1.35,
            vignette_strength=0.18, linewidth_bias_nm=-4.0),
    4: dict(dose_search=32.0, shear_amplitude_px=3.0, drift_jitter_px=1.05,
            detector_noise_sigma_search=14.0, charging_streak_prob=4.5,
            charging_streak_intensity=2.8, speckle_sigma=0.30,
            salt_pepper_prob=0.012, astigmatism_ratio=1.60,
            vignette_strength=0.30, gamma=1.25, barrel_distortion_k=0.005,
            linewidth_bias_nm=6.0),
}

ARCHS = ["dram_1x", "dram_dense", "dram_wide", "dram_loose", "dram_compact",
         "finfet_7nm", "finfet_10nm", "finfet_14nm", "finfet_22nm"]

B_SEV = {"sample": [1, 2, 3, 4, 2, 3],
         "hard": [3, 4, 3, 4, 2, 3]}
A_SEV = {"sample": 0, "hard": 1}


def build_plan(recipe, num, rng):
    n_a = round(num * 0.35)
    n_b = round(num * 0.35)
    n_c = round(num * 0.20)
    n_d = num - n_a - n_b - n_c
    plan = []
    for i in range(num):
        if i < n_a:
            subset, present, sev = "A", True, A_SEV[recipe]
        elif i < n_a + n_b:
            subset, present, sev = "B", True, B_SEV[recipe][(i - n_a) % len(B_SEV[recipe])]
        elif i < n_a + n_b + n_c:
            subset, present, sev = "C", False, [0, 1, 2, 0][(i - n_a - n_b) % 4]
        else:
            subset, present, sev = "D", True, [0, 1][(i - n_a - n_b - n_c) % 2]
        zoom = float(rng.uniform(8.0, 12.0))
        theta = float(rng.uniform(-5.0, 5.0))
        if i == 0:
            zoom, theta = 8.0, -5.0
        if i == 1:
            zoom, theta = 12.0, 5.0
        arch = ARCHS[int(rng.integers(0, len(ARCHS)))]
        plan.append((f"v{i:03d}", subset, arch, zoom, theta, present, sev))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=("sample", "hard"), required=True)
    ap.add_argument("--num", type=int, default=40)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import cv2

    args.out.mkdir(parents=True, exist_ok=True)
    master = np.random.default_rng(args.seed)
    plan = build_plan(args.recipe, args.num, master)
    gt_rows, our_rows = [], []
    for idx, (pid, subset, arch, zoom, theta, present, sev) in enumerate(plan):
        t0 = time.time()
        rng = np.random.default_rng(args.seed + idx * 7919)
        params = Phase2Params(zoom=zoom, theta_deg=theta, present=present,
                              boundary_bias=0.70, **SEVERITY[sev])
        s = generate_phase2_sample(arch, params, rng)
        ref, srch, gt = s["reference_img"], s["search_img"], s["gt"]
        if subset == "D":
            ref = to_optical_rgb(ref, rng)
            srch = to_optical_rgb(srch, rng)
        (args.out / pid).mkdir(exist_ok=True)
        cv2.imwrite(str(args.out / pid / "reference.png"), ref)
        cv2.imwrite(str(args.out / pid / "search.png"), srch)
        gt_rows.append({"pair_id": pid, "set": subset, "severity": sev,
                        "architecture": arch, "present": int(gt["present"]),
                        "x": gt["x"], "y": gt["y"], "theta": gt["theta"],
                        "scale": gt["scale"]})
        our_rows.append({"pair_id": pid,
                         "set": {"A": "A_nominal", "B": "B_degraded",
                                 "C": "C_absent", "D": "D_optical"}[subset],
                         "severity": sev,
                         "reference_path": f"{pid}/reference.png",
                         "search_path": f"{pid}/search.png",
                         "style": "dram" if arch.startswith("dram") else "finfet",
                         "modality": "optical" if subset == "D" else "sem",
                         "found": int(gt["present"]),
                         "gt_x": gt["x"], "gt_y": gt["y"],
                         "gt_zoom": zoom, "gt_rotation_deg": theta,
                         "seed": args.seed + idx * 7919})
        print(f"{pid} {subset} sev{sev} {arch:14s} z {zoom:5.2f} th {theta:+5.2f} "
              f"{time.time() - t0:5.1f}s", flush=True)
    with open(args.out / "ground_truth_org.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gt_rows[0].keys()))
        w.writeheader()
        w.writerows(gt_rows)
    with open(args.out / "ground_truth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(our_rows[0].keys()))
        w.writeheader()
        w.writerows(our_rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
