"""Stress suites past the disclosed ladder, and the hardest decoy class.

Two recipes on top of the organisers' pipeline. "extreme" extends their
severity ladder one rung past its top, every knob continued along its own
disclosed progression, to measure whether the pipeline degrades gracefully,
meaning failures land as rejections and low scores rather than confident
wrong answers. "harddecoy" keeps the jury hardened severity mix but builds
every absent pair the way the organisers' own readme recommends for the real
set and their sample deliberately avoids: a decoy reference from a canvas
with IDENTICAL zone geometry and merely different random structure, produced
by generating a second independent present style sample and taking only its
reference, so no zone width signature separates absent from present.

    .venv/bin/python experiments/20260901_stress_and_decoys/generate_stress.py \
        --recipe harddecoy --num 60 --seed 707001 --out data/stress_harddecoy
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
sys.path.insert(0, str(REPO / "experiments" / "20260901_organiser_sample_validation"))

from generate_from_organiser import ARCHS, SEVERITY
from src.phase2_pipeline import Phase2Params, generate_phase2_sample

SEVERITY = dict(SEVERITY)
SEVERITY[5] = dict(dose_search=20.0, shear_amplitude_px=4.0, drift_jitter_px=1.40,
                   detector_noise_sigma_search=18.0, charging_streak_prob=6.0,
                   charging_streak_intensity=3.5, speckle_sigma=0.42,
                   salt_pepper_prob=0.02, astigmatism_ratio=1.80,
                   vignette_strength=0.40, gamma=1.40, barrel_distortion_k=0.008,
                   linewidth_bias_nm=-8.0)

B_SEV = {"extreme": [4, 5, 4, 5, 3, 5], "harddecoy": [3, 4, 3, 4, 2, 3]}
A_SEV = {"extreme": 2, "harddecoy": 1}
C_SEV = {"extreme": [1, 2, 3, 0], "harddecoy": [0, 1, 2, 0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=("extreme", "harddecoy"), required=True)
    ap.add_argument("--num", type=int, default=60)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import cv2

    args.out.mkdir(parents=True, exist_ok=True)
    master = np.random.default_rng(args.seed)
    n_a = round(args.num * 0.35)
    n_b = round(args.num * 0.35)
    n_c = args.num - n_a - n_b
    rows = []
    for i in range(args.num):
        if i < n_a:
            subset, present, sev = "A", True, A_SEV[args.recipe]
        elif i < n_a + n_b:
            subset, present, sev = "B", True, B_SEV[args.recipe][(i - n_a) % 6]
        else:
            subset, present, sev = "C", False, C_SEV[args.recipe][(i - n_a - n_b) % 4]
        zoom = float(master.uniform(8.0, 12.0))
        theta = float(master.uniform(-5.0, 5.0))
        if i == 0:
            zoom, theta = 12.0, 5.0
        if i == 1:
            zoom, theta = 8.0, -5.0
        arch = ARCHS[int(master.integers(0, len(ARCHS)))]
        pid = f"s{i:03d}"
        t0 = time.time()
        # At severity 5 the organisers' own verifiability gate sometimes finds
        # no crop whose label clears its margin floor, which is itself a
        # finding: the rung sits at the edge of what their gate calls
        # generatable. The stress suite retries fresh draws at a relaxed
        # margin floor, keeping the 3 px label verifiability requirement, and
        # steps the severity down one rung only as a last resort.
        s = None
        for attempt in range(4):
            rng = np.random.default_rng(args.seed + i * 7919 + attempt * 104729)
            use_sev = sev if attempt < 3 else max(sev - 1, 0)
            params = Phase2Params(zoom=zoom, theta_deg=theta, present=True,
                                  boundary_bias=0.70, **SEVERITY[use_sev])
            try:
                s = generate_phase2_sample(arch, params, rng, min_margin=0.005)
                sev = use_sev
                break
            except RuntimeError:
                continue
        if s is None:
            print(f"{pid} skipped, ungeneratable at sev {sev}", flush=True)
            continue
        ref, srch, gt = s["reference_img"], s["search_img"], s["gt"]
        if not present:
            # identical zone geometry decoy: an independent second sample of
            # the same architecture supplies the reference; the search keeps
            # the first sample's canvas, so the pair is absent with no zone
            # width signature at all
            rng2 = np.random.default_rng(args.seed + i * 7919 + 3571)
            s2 = generate_phase2_sample(arch, params, rng2)
            ref = s2["reference_img"]
            gt = {"present": 0, "x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0}
        (args.out / pid).mkdir(exist_ok=True)
        cv2.imwrite(str(args.out / pid / "reference.png"), ref)
        cv2.imwrite(str(args.out / pid / "search.png"), srch)
        rows.append({"pair_id": pid,
                     "set": {"A": "A_nominal", "B": "B_degraded", "C": "C_absent"}[subset],
                     "severity": sev,
                     "reference_path": f"{pid}/reference.png",
                     "search_path": f"{pid}/search.png",
                     "style": "dram" if arch.startswith("dram") else "finfet",
                     "modality": "sem", "found": int(gt["present"]),
                     "gt_x": gt["x"], "gt_y": gt["y"],
                     "gt_zoom": zoom, "gt_rotation_deg": theta,
                     "seed": args.seed + i * 7919})
        print(f"{pid} {subset} sev{sev} {arch:14s} z {zoom:5.2f} th {theta:+5.2f} "
              f"{time.time() - t0:5.1f}s", flush=True)
    with open(args.out / "ground_truth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
