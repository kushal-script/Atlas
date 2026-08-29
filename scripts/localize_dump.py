"""One-pass localize dump for Phase 2 Day 5 analysis.

Localizes every pair in a manifest exactly as register.py does, then saves the
pose, the raw correlation peak, the trained presence probability, and the full
presence feature vector. This lets the threshold sweep (T2) and the calibration
analysis (T1/T4) re-classify instantly without re-running the ~17 min localize
pass. Run it twice (cool) for a variance check.

Usage:
    uv run python scripts/localize_dump.py --input data/phase2_mixed/manifest.csv \
        --out data/phase2_mixed/dump1.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from drift_sense.localize import load_gray, locate, optical_config, phase2_config
from drift_sense.api import presence_features, presence_probability


def _read_manifest(path):
    pairs = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        keys = {k.lower().strip(): k for k in reader.fieldnames}
        rk = next((keys[k] for k in ("reference_path", "reference", "ref_path", "ref") if k in keys), None)
        sk = next((keys[k] for k in ("search_path", "search", "wide_path", "wide") if k in keys), None)
        pid_key = keys.get("pair_id")
        pres_key = keys.get("present")
        gx = keys.get("gt_x")
        gy = keys.get("gt_y")
        for i, row in enumerate(reader):
            pid = (row.get(pid_key) if pid_key else None) or f"row_{i:04d}"
            ref = Path(row[rk]); search = Path(row[sk])
            if not ref.is_absolute():
                ref = (path.parent / ref).resolve()
            if not search.is_absolute():
                search = (path.parent / search).resolve()
            pairs.append({
                "pid": pid,
                "ref": str(ref), "search": str(search),
                "present": int(row[pres_key]) if (pres_key and row.get(pres_key) not in (None, "")) else 0,
                "gt_x": float(row[gx]) if gx and row.get(gx) not in (None, "") else None,
                "gt_y": float(row[gy]) if gy and row.get(gy) not in (None, "") else None,
            })
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    pairs = _read_manifest(args.input)
    rows = []
    for p in pairs:
        ref, ref_rgb = load_gray(p["ref"])
        search, search_rgb = load_gray(p["search"])
        cfg = optical_config() if (ref_rgb or search_rgb) else phase2_config()
        cfg.device = "cpu"
        x, y, diag, _ = locate(ref, search, cfg)
        feats = presence_features(diag)
        rows.append({
            "pid": p["pid"],
            "present": p["present"],
            "gt_x": p["gt_x"], "gt_y": p["gt_y"],
            "x": float(x), "y": float(y),
            "theta": -float(diag["theta_deg"]),
            "scale": 10.0 * float(diag["scale"]),
            "diag_score": float(diag["score"]),
            "prob": presence_probability(diag),
            "features": feats,
        })
        print(f"{p['pid']} present={p['present']} prob={rows[-1]['prob']:.3f} "
              f"x={x:.1f} y={y:.1f} scale={rows[-1]['scale']:.1f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
