"""Independent stress dataset generator, used as a domain shift proxy for the
hidden test set. It deliberately shares no image formation code or conventions
with the main generator: starter prompt style lattice (word lines, bit lines,
a contact dot at every intersection), painted gradient edge brightening
instead of height map physics, plain Gaussian noise instead of Poisson dose
noise, area averaged downsampling instead of point sampling, gamma and
contrast jitter, rotations to plus minus 5 deg and scale errors to plus minus
3 percent, both beyond the main generator's ranges.

Usage:
    python scripts/generate_stress_dataset.py --num 30 --out data/stress30 --seed 5
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

CANVAS_NM = 11200.0
CANVAS_PX_NM = 2.0
OUT_PX = 1000
REF_PIX_NM = 1.0
SEARCH_PIX_NM = 10.0


def build_canvas(rng):
    n = int(CANVAS_NM / CANVAS_PX_NM)
    img = np.full((n, n), 0.12, np.float32)
    coords_nm = (np.arange(n, dtype=np.float32) + 0.5) * CANVAS_PX_NM

    wl_pitch = rng.uniform(36.0, 56.0)
    bl_pitch = rng.uniform(54.0, 84.0)
    wl_w = wl_pitch * rng.uniform(0.35, 0.50)
    bl_w = bl_pitch * rng.uniform(0.35, 0.50)
    phase_wl = rng.uniform(0, wl_pitch)
    phase_bl = rng.uniform(0, bl_pitch)

    img[((coords_nm - phase_wl) % wl_pitch) < wl_w, :] = 0.40
    img[:, ((coords_nm - phase_bl) % bl_pitch) < bl_w] = 0.62

    wls = np.arange(phase_wl + wl_w / 2, CANVAS_NM, wl_pitch)
    bls = np.arange(phase_bl + bl_w / 2, CANVAS_NM, bl_pitch)
    cu, cv2_ = np.meshgrid(bls, wls)
    cu, cv_ = cu.ravel(), cv2_.ravel()
    missing = rng.random(cu.size) < rng.uniform(0.004, 0.020)
    r_nm = 0.42 * min(wl_pitch, bl_pitch)
    r_px = r_nm / CANVAS_PX_NM
    for u, v, miss in zip(cu / CANVAS_PX_NM, cv_ / CANVAS_PX_NM, missing):
        if miss:
            continue
        cv2.circle(img, (int(round(u)), int(round(v))), int(round(r_px)), 0.95, -1)

    for _ in range(rng.integers(4, 10)):
        w = rng.uniform(150, 1600) / CANVAS_PX_NM
        h = rng.uniform(150, 1600) / CANVAS_PX_NM
        x0 = rng.uniform(0, n - w)
        y0 = rng.uniform(0, n - h)
        val = rng.uniform(0.2, 0.8)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), float(val), -1)
    for _ in range(rng.integers(1, 4)):
        p1 = (int(rng.uniform(0, n)), int(rng.uniform(0, n)))
        p2 = (p1[0] + int(rng.uniform(-400, 400)), p1[1] + int(rng.uniform(-400, 400)))
        cv2.line(img, p1, p2, float(rng.uniform(0.05, 0.9)), int(rng.uniform(2, 6)))

    meta = {"wl_pitch_nm": wl_pitch, "bl_pitch_nm": bl_pitch,
            "wl_width_nm": wl_w, "bl_width_nm": bl_w, "contact_radius_nm": r_nm}
    return img, meta


def painted_edges(img, rng):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag /= max(mag.max(), 1e-6)
    return img + rng.uniform(0.4, 1.0) * mag


def tone(img, rng, noise_sigma):
    img = img + rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    img = img ** rng.uniform(0.85, 1.15)
    return (img * 255).astype(np.uint8)


def sample_window(canvas, center_nm, theta_rad, pixel_nm, out_px):
    half = (out_px - 1) / 2.0
    idx = np.arange(out_px, dtype=np.float64)
    du = (idx - half) * pixel_nm
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    u = center_nm[0] + du[None, :] * cos_t - du[:, None] * sin_t
    v = center_nm[1] + du[None, :] * sin_t + du[:, None] * cos_t
    return map_coordinates(canvas, [v / CANVAS_PX_NM - 0.5, u / CANVAS_PX_NM - 0.5],
                           order=1, mode="nearest").astype(np.float32)


def generate_pair(seed):
    rng = np.random.default_rng(seed)
    canvas, layout = build_canvas(rng)

    theta_s = np.deg2rad(rng.uniform(-5.0, 5.0))
    scale_err = rng.uniform(-0.03, 0.03)
    search_pix = SEARCH_PIX_NM * (1 + scale_err)
    s_center = (CANVAS_NM / 2 + rng.uniform(-150, 150),
                CANVAS_NM / 2 + rng.uniform(-150, 150))

    margin_px = 80
    off = (rng.uniform(margin_px, OUT_PX - 1 - margin_px) - (OUT_PX - 1) / 2,
           rng.uniform(margin_px, OUT_PX - 1 - margin_px) - (OUT_PX - 1) / 2)
    cos_t, sin_t = np.cos(theta_s), np.sin(theta_s)
    r_center = (s_center[0] + (off[0] * cos_t - off[1] * sin_t) * search_pix,
                s_center[1] + (off[0] * sin_t + off[1] * cos_t) * search_pix)

    ref = sample_window(canvas, r_center, 0.0, REF_PIX_NM, OUT_PX)
    ref = painted_edges(ref, rng)
    ref = gaussian_filter(ref, rng.uniform(1.5, 3.0))
    ref8 = tone(ref, rng, rng.uniform(0.010, 0.030))

    inter_px = OUT_PX * 5
    inter = sample_window(canvas, s_center, theta_s, search_pix / 5.0, inter_px)
    inter = painted_edges(inter, rng)
    inter = gaussian_filter(inter, rng.uniform(10.0, 30.0) / (search_pix / 5.0))
    search = cv2.resize(inter, (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)
    search8 = tone(search, rng, rng.uniform(0.030, 0.090))

    du = r_center[0] - s_center[0]
    dv = r_center[1] - s_center[1]
    gx = (du * cos_t + dv * sin_t) / search_pix + (OUT_PX - 1) / 2.0
    gy = (-du * sin_t + dv * cos_t) / search_pix + (OUT_PX - 1) / 2.0

    meta = {"seed": int(seed), "style": "stress_basic",
            "ground_truth": {"x": float(gx), "y": float(gy)},
            "relative_rotation_deg": float(np.rad2deg(theta_s)),
            "search_scale_error": float(scale_err),
            "placement": "uniform",
            "layout": layout,
            "search_capture": {"pose": {"pixel_nm": float(search_pix)},
                               "settings": {"dose_e": 0.0}}}
    return ref8, search8, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.num):
        ref8, search8, meta = generate_pair(args.seed * 999_983 + i)
        pd = args.out / f"pair_{i:04d}"
        pd.mkdir(exist_ok=True)
        cv2.imwrite(str(pd / "reference.png"), ref8)
        cv2.imwrite(str(pd / "search.png"), search8)
        (pd / "meta.json").write_text(json.dumps(meta, indent=2))
        g = meta["ground_truth"]
        rows.append({"pair_id": pd.name, "gt_x": f"{g['x']:.3f}", "gt_y": f"{g['y']:.3f}",
                     "rotation_deg": f"{meta['relative_rotation_deg']:.3f}",
                     "scale_error": f"{meta['search_scale_error']:.5f}"})
        print(f"{pd.name} gt=({g['x']:.1f}, {g['y']:.1f}) rot={meta['relative_rotation_deg']:.2f}")
    with open(args.out / "ground_truth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
