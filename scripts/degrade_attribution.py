"""Which degradation factor actually drives the severe set's loss.

The severity ladder pushes six factors at once, so the collapse at severities
3 and 4 has no attribution: a de jitter or de drift preprocessing pass would
be worth building only if the structured factors are what break the matcher,
and worth nothing if the loss belongs to dose and beam spot, which no
preprocessing can undo. This generates paired suites from identical seeds, a
control under the full severity 4 ladder and one suite per factor with that
factor alone restored to nominal, and localizes each without the presence
decision, so the difference per factor is measured on the same specimens.

    .venv/bin/python scripts/degrade_attribution.py --num 20 --seed 8501
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import drift_sense.generator as G
from drift_sense.localize import MatchConfig, locate
from drift_sense.params import GeneratorConfig

FULL = dict(G.DEGRADE_LADDER[4])
KNOCKOUTS = {
    "control": {},
    "no_dose": {"dose": 1.0},
    "no_psf": {"psf": 1.0},
    "no_charge": {"charge": 1.0},
    "no_drift": {"drift_px": 1.2},
    "no_jitter": {"jitter": 1.0},
    "no_poly": {"poly": 0.0},
}


def credit(e):
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--seed", type=int, default=8501)
    ap.add_argument("--out", type=Path, default=Path("local/degrade_attribution.csv"))
    args = ap.parse_args()

    cfg = GeneratorConfig(); cfg.phase2 = True
    mcfg = MatchConfig()
    rng = np.random.default_rng(args.seed)
    styles = ["dram" if rng.random() < 0.5 else "finfet" for _ in range(args.num)]

    rows = []
    for name, patch in KNOCKOUTS.items():
        G.DEGRADE_LADDER[4] = {**FULL, **patch}
        errs = []
        for i in range(args.num):
            r = G.generate_pair(seed=args.seed * 1_000_003 + i, style=styles[i],
                                cfg=cfg, modality="sem", absent=False, degrade=4)
            gt = r["meta"]["ground_truth"]
            t0 = time.perf_counter()
            x, y, _, _ = locate(r["reference"], r["search"], mcfg, t_start=t0)
            e = float(np.hypot(x - gt["x"], y - gt["y"]))
            errs.append(e)
            rows.append({"suite": name, "pair": i, "err": f"{e:.3f}"})
        errs = np.array(errs)
        print(f"  {name:<11} credit {np.mean([credit(e) for e in errs]):.3f}  "
              f"within5px {(errs <= 5).mean() * 100:3.0f}%  median {np.median(errs):8.2f}px",
              flush=True)
    G.DEGRADE_LADDER[4] = FULL
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["suite", "pair", "err"])
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
