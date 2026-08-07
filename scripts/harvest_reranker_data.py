"""Harvest re-ranker training data from generated datasets.

Runs the classical pipeline over every pair, keeps the pairs whose correlation
surface is degenerate (a stage two pool exists), and saves one compressed npz
per pair holding the four channel candidate stacks, the two scalar features,
and the label: the pool position of the candidate nearest ground truth within
3 px, or minus one when no pool candidate is correct (the null class).

Usage:
    python scripts/harvest_reranker_data.py --datasets data/reranker_train240 --out data/reranker_harvest
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drift_sense.localize import MatchConfig, load_gray, locate
from drift_sense.reranker import build_pool, build_stacks


def harvest_pair(pair_dir, cfg):
    meta = json.loads((pair_dir / "meta.json").read_text())
    ref, _ = load_gray(pair_dir / "reference.png")
    search, _ = load_gray(pair_dir / "search.png")
    _, _, diag, _ = locate(ref, search, cfg, return_artifacts=True)
    art = diag["artifacts"]
    if art["r2"] is None:
        return None
    wide = art["wide"]
    t = cfg.template_px
    half = (t - 1) / 2.0
    g = meta["ground_truth"]
    d = np.hypot(wide[:, 1] + half - g["x"], wide[:, 0] + half - g["y"])
    truth_idx = int(np.argmin(d)) if d.min() <= 3.0 else None

    pool_idx = build_pool(art["resp"], art["r2"], wide, cfg.reranker_pool)
    label = -1
    if truth_idx is not None:
        if truth_idx not in pool_idx:
            pool_idx = np.append(pool_idx, truth_idx)
        label = int(np.where(pool_idx == truth_idx)[0][0])

    x, s = build_stacks(art["search"], art["template"], art["med"], art["rt0"],
                        art["resp"], art["r2"], wide, pool_idx)
    return {"x": x.astype(np.float16), "s": s, "label": label,
            "style": meta["style"], "placement": meta["placement"],
            "gt_x": g["x"], "gt_y": g["y"],
            "positions": wide[pool_idx] + half}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = MatchConfig()
    kept = skipped = 0
    for ds in args.datasets:
        for pd in sorted(d for d in ds.iterdir()
                         if d.is_dir() and d.name.startswith("pair_")):
            rec = harvest_pair(pd, cfg)
            if rec is None:
                skipped += 1
                continue
            out = args.out / f"{ds.name}_{pd.name}.npz"
            np.savez_compressed(out, **rec)
            kept += 1
            print(f"{out.name} label={rec['label']} pool={len(rec['s'])}")
    print(f"kept {kept} degenerate pairs, skipped {skipped} unambiguous pairs")


if __name__ == "__main__":
    main()
