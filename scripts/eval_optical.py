"""Evaluate optical matching strategies on a generated optical suite.

Set D is bonus only and gated: it pays nothing unless the grayscale sets clear
their own bar, so the goal here is the credit threshold rather than the last
decimal. Strategies are compared on the same pairs and the same pose search;
only how colour reaches the correlation changes.

    .venv/bin/python scripts/eval_optical.py --dataset data/opt_train --strategy all
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from drift_sense.localize import (MatchConfig, load_colour, load_gray, locate,
                                  optical_config)


def credit(err):
    return 1.0 if err <= 1 else 0.8 if err <= 2 else 0.6 if err <= 3 else 0.4 if err <= 5 else 0.0


def run_planes(ref_planes, search_planes, cfg):
    """Locate using the first plane, then refine the answer by summing the
    correlation surfaces of every plane at the winning pose."""
    x, y, diag, resp = locate(ref_planes[0], search_planes[0], cfg,
                              return_artifacts=True)
    if len(ref_planes) == 1:
        return x, y, diag
    art = diag["artifacts"]
    tmpl = art["template"]
    total = resp.astype(np.float32).copy()
    half = (tmpl.shape[0] - 1) / 2.0
    theta, scale, sig = diag["theta_grid_deg"], diag["scale"], diag["psf_sigma_nm"]
    from drift_sense.localize import _make_template, _preprocess, _effective_sigma
    from drift_sense import backend
    for rp, sp in list(zip(ref_planes, search_planes))[1:]:
        refp = rp.astype(np.float32)
        ref_low = backend.gaussian(refp, cfg.bandpass_sigma_px * cfg.zoom, cfg.device)
        band = backend.gaussian(refp, _effective_sigma(sig, cfg), cfg.device) - ref_low
        t = _make_template(band, theta, scale, cfg)
        sproc, _ = _preprocess(sp, cfg, denoise=True)
        r = cv2.matchTemplate(np.ascontiguousarray(sproc, dtype=np.float32),
                              np.ascontiguousarray(t, dtype=np.float32),
                              cv2.TM_CCOEFF_NORMED)
        if r.shape == total.shape:
            total += r
    py, px = np.unravel_index(int(np.argmax(total)), total.shape)
    return float(px + half), float(py + half), diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--strategy", default="luminance")
    ap.add_argument("--psf_bank", default="", help="comma separated nm")
    ap.add_argument("--wide_bank", default="", help="comma separated nm")
    ap.add_argument("--bandpass", type=float, default=None)
    ap.add_argument("--denoise", type=float, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", type=Path, default=None)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.dataset / "ground_truth.csv"))
            if r.get("found", "1") == "1"]
    if args.limit:
        rows = rows[:args.limit]
    strategies = (["luminance", "colour_sum"] if args.strategy == "all"
                  else [args.strategy])
    for strat in strategies:
        cfg = optical_config()
        if args.psf_bank:
            cfg.psf_sigma_bank_nm = tuple(float(v) for v in args.psf_bank.split(","))
        if args.wide_bank:
            cfg.wide_sigma_bank_nm = tuple(float(v) for v in args.wide_bank.split(","))
        if args.bandpass is not None:
            cfg.bandpass_sigma_px = args.bandpass
        if args.denoise is not None:
            cfg.denoise_sigma_px = args.denoise
        creds, errs, ts = [], [], []
        for r in rows:
            rp = args.dataset / r["reference_path"]
            sp = args.dataset / r["search_path"]
            t0 = time.perf_counter()
            if strat == "luminance":
                ref, _ = load_gray(rp); search, _ = load_gray(sp)
                x, y, _, = run_planes([ref], [search], cfg)
            else:
                refs, _ = load_colour(rp); searches, _ = load_colour(sp)
                x, y, _ = run_planes(refs, searches, cfg)
            ts.append(time.perf_counter() - t0)
            e = float(np.hypot(x - float(r["gt_x"]), y - float(r["gt_y"])))
            errs.append(e); creds.append(credit(e))
        e = np.array(errs)
        if args.dump:
            with open(args.dump, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["pair_id", "err", "credit"])
                for r_, e_, c_ in zip(rows, errs, creds):
                    w.writerow([r_["pair_id"], f"{e_:.3f}", c_])
        tag = args.label or strat
        print(f"  {tag:28s} credit {np.mean(creds):.3f}  within5px {(e <= 5).mean()*100:3.0f}%  "
              f"median {np.median(e):7.2f}px  {np.median(ts):.2f}s/pair", flush=True)


if __name__ == "__main__":
    main()
