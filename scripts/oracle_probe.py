"""Is the degraded set a search failure or an information limit?

The distinction decides what is worth building. If the true site still wins the
correlation once the true pose is handed to the matcher, the pose search is at
fault and a better search recovers those pairs. If an impostor site outscores
the true one even under the true pose, no search improvement can help, because
the evidence itself prefers the wrong answer, and only better discrimination
between equally correlating sites could.

The pose is taken from the generator's own record, which no scored run has.
That is the point: this measures the ceiling a perfect pose search would hit.

    .venv/bin/python scripts/oracle_probe.py --dataset data/p2degraded --severity 4
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from drift_sense.localize import (MatchConfig, _effective_sigma, _lowpass,
                                  _make_template, _preprocess, load_gray,
                                  optical_config)
from drift_sense import backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--severity", default="4")
    ap.add_argument("--set", dest="setname", default="B_degraded")
    ap.add_argument("--preset", default="sem", choices=["sem", "optical"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = optical_config() if args.preset == "optical" else MatchConfig()
    rows = [r for r in csv.DictReader(open(args.dataset / "ground_truth.csv"))
            if r["set"] == args.setname and r["found"] == "1"
            and (args.setname != "B_degraded" or r["severity"] == args.severity)]
    if args.limit:
        rows = rows[:args.limit]

    out = []
    for r in rows:
        ref, _ = load_gray(args.dataset / r["reference_path"])
        search, _ = load_gray(args.dataset / r["search_path"])
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        scale = float(r["gt_zoom"]) / cfg.zoom
        theta = -float(r["gt_rotation_deg"]) / cfg.theta_report_sign * -1.0
        reff = ref.astype(np.float32)
        low = _lowpass(reff, cfg.bandpass_sigma_px * cfg.zoom, cfg)
        proc, _ = _preprocess(search, cfg, denoise=True)
        best = None
        for sig in cfg.psf_sigma_bank_nm:
            band = backend.gaussian(reff, _effective_sigma(sig, cfg), cfg.device) - low
            tmpl = _make_template(band, theta, scale, cfg)
            resp = cv2.matchTemplate(np.ascontiguousarray(proc, dtype=np.float32),
                                     np.ascontiguousarray(tmpl, dtype=np.float32),
                                     cv2.TM_CCOEFF_NORMED)
            half = (tmpl.shape[0] - 1) / 2.0
            ty, tx = int(round(gy - half)), int(round(gx - half))
            if not (0 <= ty < resp.shape[0] and 0 <= tx < resp.shape[1]):
                continue
            w = 3
            y0, y1 = max(0, ty - w), min(resp.shape[0], ty + w + 1)
            x0, x1 = max(0, tx - w), min(resp.shape[1], tx + w + 1)
            true_peak = float(resp[y0:y1, x0:x1].max())
            gmax = float(resp.max())
            gy_i, gx_i = np.unravel_index(int(np.argmax(resp)), resp.shape)
            dist = float(np.hypot(gx_i + half - gx, gy_i + half - gy))
            rank = int((resp > true_peak).sum())
            if best is None or true_peak > best["true_peak"]:
                best = {"pair_id": r["pair_id"], "severity": r["severity"],
                        "true_peak": true_peak, "global_max": gmax,
                        "argmax_dist_px": dist, "pixels_above_true": rank,
                        "true_site_is_argmax": int(dist <= 5.0)}
        if best:
            out.append(best)
            print(f"  {best['pair_id']} true_peak {best['true_peak']:.3f} "
                  f"global_max {best['global_max']:.3f} argmax {best['argmax_dist_px']:7.1f}px "
                  f"{'TRUE SITE WINS' if best['true_site_is_argmax'] else 'impostor wins'}", flush=True)

    wins = sum(o["true_site_is_argmax"] for o in out)
    print(f"\n  severity {args.severity}: {len(out)} present pairs probed with the TRUE pose supplied")
    print(f"    the true site is the global correlation maximum in {wins} of {len(out)} "
          f"({wins/max(len(out),1)*100:.0f}%)")
    gap = [o["global_max"] - o["true_peak"] for o in out if not o["true_site_is_argmax"]]
    if gap:
        print(f"    where an impostor wins, it beats the true site by a median of "
              f"{np.median(gap):.3f} correlation")
    print(f"    interpretation: pairs where the true site loses under a true pose are "
          f"information limited and no pose search can recover them")
    if args.out and out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)


if __name__ == "__main__":
    main()
