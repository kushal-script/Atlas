"""Assert every compute backend returns the same answer as the reference path.

A faster backend that changes the result is not an optimization, so this is run
before any backend is used for a reported number. The reference is the original
path: scipy for the blur, OpenCV for the correlation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from drift_sense import backend


def report(name, a, b, tol):
    d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    ok = d <= tol
    print(f"  {'pass' if ok else 'FAIL'}  {name:<44} max abs diff {d:.3e}  tol {tol:.0e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/physics40"))
    ap.add_argument("--num", type=int, default=4)
    args = ap.parse_args()

    devices = [d for d in backend.available_devices() if d != "cpu"]
    print(f"reference: scipy blur, OpenCV correlation")
    print(f"testing backends: cpu, {', '.join(devices) if devices else '(no accelerator present)'}\n")

    folders = sorted(d for d in args.pairs.iterdir() if d.is_dir())[:args.num]
    if not folders:
        raise SystemExit(f"no pair folders under {args.pairs}")

    ok = True
    for folder in folders:
        ref = cv2.imread(str(folder / "reference.png"), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(str(folder / "search.png"), cv2.IMREAD_GRAYSCALE)
        if ref is None or search is None:
            continue
        print(folder.name)
        for sigma in (2.0, 6.5, 20.0):
            want = gaussian_filter(ref.astype(np.float32), sigma)
            ok &= report(f"blur sigma {sigma} cpu vs scipy",
                         backend.gaussian(ref, sigma, "cpu"), want, 1e-2)
            for dev in devices:
                ok &= report(f"blur sigma {sigma} {dev} vs scipy",
                             backend.gaussian(ref, sigma, dev), want, 1e-2)
        tmpl = cv2.resize(ref, (90, 90), interpolation=cv2.INTER_AREA)
        want = backend.match_ccoeff_normed(search, tmpl, "cpu")
        for dev in devices:
            got = backend.match_ccoeff_normed(search, tmpl, dev)
            ok &= report(f"correlation {dev} vs OpenCV", got, want, 2e-5)
            corr = backend.make_correlator(search, dev)
            ok &= report(f"fft correlator {dev} vs OpenCV", corr.full(tmpl), want, 2e-4)
            small = [cv2.resize(ref, (45, 45), interpolation=cv2.INTER_AREA),
                     cv2.resize(ref, (52, 52), interpolation=cv2.INTER_AREA)]
            cpu_pk = backend.make_correlator(search, "cpu").peaks(small)
            gpu_pk = corr.peaks(small)
            ok &= report(f"batched peaks {dev} vs OpenCV",
                         np.array(gpu_pk), np.array(cpu_pk), 2e-4)
            dy, dx = np.unravel_index(np.argmax(want), want.shape)
            gy, gx = np.unravel_index(np.argmax(got), got.shape)
            same = (dy, dx) == (gy, gx)
            print(f"  {'pass' if same else 'FAIL'}  peak location {dev} "
                  f"{(gx, gy)} vs OpenCV {(dx, dy)}")
            ok &= same
        print()

    print("ALL BACKENDS AGREE" if ok else "BACKEND MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
