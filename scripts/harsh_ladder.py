"""How fast does the degraded set fall if the organiser's ladder is harsher.

The blind severity parameters are undisclosed, so the honest question is not
whether our ladder matches theirs but how steeply credit falls beyond our own
hardest rung. A fifth severity is extrapolated one step past the ladder in
every factor, paired suites are generated from identical seeds at severities
four and five, and both are localized, giving a measured slope per rung.

    .venv/bin/python scripts/harsh_ladder.py --num 24 --seed 8701
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import drift_sense.generator as G
from drift_sense.localize import MatchConfig, locate
from drift_sense.params import GeneratorConfig


def credit(e):
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=24)
    ap.add_argument("--seed", type=int, default=8701)
    args = ap.parse_args()
    G.DEGRADE_LADDER[5] = {"dose": 0.10, "psf": 3.5, "charge": 5.0,
                           "drift_px": 12.0, "jitter": 7.0, "poly": 0.25}
    cfg = GeneratorConfig(); cfg.phase2 = True
    mcfg = MatchConfig()
    rng = np.random.default_rng(args.seed)
    styles = ["dram" if rng.random() < 0.5 else "finfet" for _ in range(args.num)]
    for sev in (3, 4, 5):
        errs = []
        for i in range(args.num):
            r = G.generate_pair(seed=args.seed * 1_000_003 + i, style=styles[i],
                                cfg=cfg, modality="sem", absent=False, degrade=sev)
            gt = r["meta"]["ground_truth"]
            t0 = time.perf_counter()
            x, y, _, _ = locate(r["reference"], r["search"], mcfg, t_start=t0)
            errs.append(float(np.hypot(x - gt["x"], y - gt["y"])))
        e = np.array(errs)
        print(f"  severity {sev}: credit {np.mean([credit(v) for v in e]):.3f}  "
              f"within5px {(e <= 5).mean() * 100:3.0f}%  median {np.median(e):8.2f}px", flush=True)
    del G.DEGRADE_LADDER[5]


if __name__ == "__main__":
    main()
