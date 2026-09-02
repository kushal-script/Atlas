"""Retest the severity four information limit under evidence that postdates it.

The recorded bound says the true site loses the correlation on 66 percent of
severity four pairs even with the true pose supplied, so no better search can
help. That measurement predates two evidence functions and never tried a
third, so each is probed at the true pose on the same pairs:

  shipped     the bandpassed bank template, reproducing the recorded bound
  raw         the full reference formation statistic the released gate uses
  fullband    the bandpassed template widened to cover the whole reference
  wiener      SNR weighted correlation, the matched filter under the measured
              noise floor in place of the fixed bandpass

A win is the global argmax landing within 5 px of truth. If any arm clears
the shipped one materially, the bound was a property of one evidence function
and a shippable change follows; if none does, the limit is triple confirmed
as information in the pixels.

    .venv/bin/python experiments/20260902_sev4_oracle_revisited/probe.py \
        --dataset data/p2degraded --severity 4
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import cv2
import numpy as np

from drift_sense.localize import (MatchConfig, _effective_sigma, _lowpass,
                                  _make_template, _preprocess, _raw_confirm,
                                  load_gray)
from drift_sense import backend


def wiener_filter_pair(tmpl, search, noise_sigma):
    """Correlation inputs reweighted by the template's own spectral SNR."""
    h, w = search.shape
    ft = np.fft.rfft2(tmpl - tmpl.mean(), s=(h, w))
    pt = np.abs(ft) ** 2
    n_power = (noise_sigma ** 2) * tmpl.size
    gain = pt / (pt + n_power + 1e-9)
    fs = np.fft.rfft2(search - search.mean())
    s_f = np.fft.irfft2(fs * gain, s=(h, w)).astype(np.float32)
    ftt = np.fft.rfft2(tmpl - tmpl.mean(), s=tmpl.shape)
    gt = np.abs(ftt) ** 2
    gaint = gt / (gt + (noise_sigma ** 2) * tmpl.size + 1e-9)
    t_f = np.fft.irfft2(ftt * gaint, s=tmpl.shape).astype(np.float32)
    return t_f, s_f


def argmax_dist(resp, half, gx, gy):
    gy_i, gx_i = np.unravel_index(int(np.argmax(resp)), resp.shape)
    return float(np.hypot(gx_i + half - gx, gy_i + half - gy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--severity", default="4")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = MatchConfig()
    rows = [r for r in csv.DictReader(open(args.dataset / "ground_truth.csv"))
            if r["set"] == "B_degraded" and r["found"] == "1"
            and r["severity"] == args.severity]
    if args.limit:
        rows = rows[:args.limit]

    wins = {"shipped": 0, "raw": 0, "fullband": 0, "wiener": 0}
    both_lose = 0
    for r in rows:
        ref, _ = load_gray(args.dataset / r["reference_path"])
        search, _ = load_gray(args.dataset / r["search_path"])
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        z = float(r["gt_zoom"])
        scale = z / cfg.zoom
        theta = float(r["gt_rotation_deg"]) * cfg.theta_report_sign
        reff = ref.astype(np.float32)
        low = _lowpass(reff, cfg.bandpass_sigma_px * cfg.zoom, cfg)
        proc, noise = _preprocess(search, cfg, denoise=True)
        proc = np.ascontiguousarray(proc, dtype=np.float32)

        res = {}
        for arm in ("shipped", "fullband", "wiener"):
            acfg = MatchConfig()
            if arm == "fullband":
                acfg.template_px = max(int(round(1000.0 / (cfg.zoom * scale))) - 2, 90)
                acfg.scale_adaptive_template = False
            best = None
            for sig in cfg.psf_sigma_bank_nm:
                band = backend.gaussian(reff, _effective_sigma(sig, cfg), cfg.device) - low
                tmpl = _make_template(band, theta, scale, acfg)
                if tmpl.shape[0] >= min(proc.shape):
                    continue
                if arm == "wiener":
                    t_in, s_in = wiener_filter_pair(tmpl, proc, max(noise, 1.0))
                else:
                    t_in, s_in = tmpl, proc
                resp = cv2.matchTemplate(s_in, np.ascontiguousarray(t_in), cv2.TM_CCOEFF_NORMED)
                half = (tmpl.shape[0] - 1) / 2.0
                d = argmax_dist(resp, half, gx, gy)
                pk = float(resp.max())
                if best is None or pk > best[0]:
                    best = (pk, d)
            res[arm] = best[1] if best else 1e9

        rc = _raw_confirm(ref, search, z, float(r["gt_rotation_deg"]), gx, gy)
        res["raw"] = float(np.hypot(rc["x"] - gx, rc["y"] - gy)) if rc else 1e9

        line = f"  {r['pair_id']}"
        for arm in ("shipped", "raw", "fullband", "wiener"):
            ok = res[arm] <= 5.0
            wins[arm] += ok
            line += f"  {arm} {'WIN ' if ok else 'lose'} ({min(res[arm], 9999):6.1f}px)"
        if res["shipped"] > 5.0 and min(res.values()) > 5.0:
            both_lose += 1
        print(line, flush=True)

    n = len(rows)
    print(f"\n  severity {args.severity}, {n} pairs, TRUE pose supplied to every arm")
    for arm in ("shipped", "raw", "fullband", "wiener"):
        print(f"    {arm:9s} true site wins {wins[arm]:2d} of {n}  ({wins[arm]/max(n,1)*100:.0f}%)")
    any_wins = n - both_lose
    print(f"    any arm   true site wins {any_wins:2d} of {n}  ({any_wins/max(n,1)*100:.0f}%)")


if __name__ == "__main__":
    main()
