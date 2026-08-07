"""Localization CLI.

Usage:
    python scripts/localize.py <reference.png> <search.png>

Prints the center x y pixel coordinates of the matched region in the search
image, x is the column from the left and y is the row from the top, origin at
the center of the top left pixel.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drift_sense.localize import MatchConfig, load_gray, locate, optical_config


def main():
    ap = argparse.ArgumentParser(description="Locate the reference pattern in the search image")
    ap.add_argument("reference", type=Path)
    ap.add_argument("search", type=Path)
    ap.add_argument("--json", action="store_true", help="print full diagnostics as JSON")
    ap.add_argument("--reranker", action="store_true",
                    help="use the learned re-ranker for the stage two decision")
    args = ap.parse_args()

    ref, ref_rgb = load_gray(args.reference)
    search, search_rgb = load_gray(args.search)
    cfg = optical_config() if (ref_rgb or search_rgb) else MatchConfig()
    if args.reranker:
        cfg.reranker_path = str(Path(__file__).resolve().parents[1] / "models" / "reranker.npz")
    x, y, diag, _ = locate(ref, search, cfg)
    if args.json:
        print(json.dumps({"x": x, "y": y, **{k: v for k, v in diag.items()}}, indent=2))
    else:
        print(f"{x:.2f} {y:.2f}")


if __name__ == "__main__":
    main()
