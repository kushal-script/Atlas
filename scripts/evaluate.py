"""Evaluation CLI.

Usage:
    python scripts/evaluate.py --dataset data/train --name baseline_ncc
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drift_sense.evaluate import run_evaluation
from drift_sense.localize import MatchConfig


def main():
    ap = argparse.ArgumentParser(description="Evaluate the localizer on a generated dataset")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--name", type=str, default="run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_stage2", action="store_true",
                    help="disable the residual disambiguation stage")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parents[1] / "experiments" / f"{stamp}_{args.name}"
    cfg = MatchConfig(residual_disambiguation=not args.no_stage2)
    t0 = time.time()
    rows, metrics = run_evaluation(args.dataset, out_dir, cfg, args.limit)
    overall = metrics["overall"]
    print(json.dumps(overall, indent=2))
    print(f"evaluated {len(rows)} pairs in {time.time() - t0:.1f}s, results in {out_dir}")


if __name__ == "__main__":
    main()
