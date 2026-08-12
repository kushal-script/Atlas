"""Third independent generator built to the published organiser specification.

The participant help document and the official starter Space fix a number of
things our physics generator does not cover, and this script reproduces that
specification exactly so the localizer can be measured against it before the
hidden test set arrives. It shares no image formation code with either the
main generator or the adversarial stress generator.

Specification points reproduced here:
structure presets on the coarse end (DRAM feature size 24 to 80 nm, word line
pitch 48 to 240 nm, FinFET fin pitch 40 to 140 nm, gate pitch 76 to 260 nm),
which are 2 to 4 times coarser than the modern node dimensions our physics
generator uses; magnification from 9 to 1 up to 11 to 1; rotation of 1 to 2
degrees; and the full degradation list named in the sessions, namely beam spot
blur with astigmatism, dose driven shot noise with a much lower search dose,
raster shear and row jitter, line width bias, corner rounding, barrel and
pincushion distortion, vignetting, gamma change, horizontal charging streaks,
multiplicative speckle and salt and pepper impulse noise.

Usage:
    python scripts/generate_starter_spec_dataset.py --num 40 --out data/spec40 --seed 11
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

OUT_PX = 1000
REF_PIX_NM = 1.0
SEARCH_PIX_NM_NOMINAL = 10.0
CANVAS_PX_NM = 2.0
CANVAS_NM = 12000.0

DRAM_PRESETS = [
    {"f": 32, "wl_pitch": 64, "wl_w": 32, "bl_pitch": 96, "bl_w": 32, "contact": 32},
    {"f": 24, "wl_pitch": 48, "wl_w": 24, "bl_pitch": 72, "bl_w": 24, "contact": 24},
    {"f": 48, "wl_pitch": 96, "wl_w": 48, "bl_pitch": 144, "bl_w": 48, "contact": 48},
    {"f": 60, "wl_pitch": 120, "wl_w": 56, "bl_pitch": 180, "bl_w": 60, "contact": 58},
    {"f": 36, "wl_pitch": 72, "wl_w": 30, "bl_pitch": 108, "bl_w": 34, "contact": 30},
    {"f": 80, "wl_pitch": 160, "wl_w": 78, "bl_pitch": 240, "bl_w": 80, "contact": 78},
]

FINFET_PRESETS = [
    {"fin_pitch": 48, "fin_w": 16, "gate_pitch": 90, "gate_l": 28, "contact": 28},
    {"fin_pitch": 40, "fin_w": 14, "gate_pitch": 76, "gate_l": 24, "contact": 24},
    {"fin_pitch": 60, "fin_w": 20, "gate_pitch": 110, "gate_l": 34, "contact": 34},
    {"fin_pitch": 80, "fin_w": 26, "gate_pitch": 150, "gate_l": 46, "contact": 44},
    {"fin_pitch": 96, "fin_w": 32, "gate_pitch": 180, "gate_l": 56, "contact": 52},
    {"fin_pitch": 140, "fin_w": 46, "gate_pitch": 260, "gate_l": 80, "contact": 76},
]

VARIANTS = ["clean", "low_dose", "heavy_drift", "speckle_salt_pepper", "charging"]

BARREL_MAX = 0.02


def _nm(v):
    return v / CANVAS_PX_NM


def _dram_block(canvas, u0, v0, u1, v1, p, rng, bias):
    wl_w = max(_nm(p["wl_w"] + bias), 1.0)
    bl_w = max(_nm(p["bl_w"] + bias), 1.0)
    wl_pitch, bl_pitch = _nm(p["wl_pitch"]), _nm(p["bl_pitch"])
    r = max(_nm((p["contact"] + bias) / 2.0), 1.0)
    canvas[int(v0):int(v1), int(u0):int(u1)] = 0.18
    for v in np.arange(v0 + rng.uniform(0, wl_pitch), v1, wl_pitch):
        canvas[int(v):int(v + wl_w), int(u0):int(u1)] = 0.45
    for u in np.arange(u0 + rng.uniform(0, bl_pitch), u1, bl_pitch):
        canvas[int(v0):int(v1), int(u):int(u + bl_w)] = 0.66
    for v in np.arange(v0 + wl_pitch, v1 - wl_pitch, wl_pitch):
        for u in np.arange(u0 + bl_pitch, u1 - bl_pitch, bl_pitch):
            if rng.random() < 0.012:
                continue
            cv2.circle(canvas, (int(u + bl_w / 2), int(v + wl_w / 2)), int(r), 0.95, -1)


def _finfet_block(canvas, u0, v0, u1, v1, p, rng, bias):
    fin_w = max(_nm(p["fin_w"] + bias), 1.0)
    gate_l = max(_nm(p["gate_l"] + bias), 1.0)
    fin_pitch, gate_pitch = _nm(p["fin_pitch"]), _nm(p["gate_pitch"])
    canvas[int(v0):int(v1), int(u0):int(u1)] = 0.15
    for v in np.arange(v0 + rng.uniform(0, fin_pitch), v1, fin_pitch):
        canvas[int(v):int(v + fin_w), int(u0):int(u1)] = 0.52
    for u in np.arange(u0 + rng.uniform(0, gate_pitch), u1, gate_pitch):
        canvas[int(v0):int(v1), int(u):int(u + gate_l)] = 0.72
    cw = max(_nm(p["contact"] * 0.5), 1.0)
    for u in np.arange(u0 + gate_pitch / 2, u1 - gate_pitch, gate_pitch):
        if rng.random() < 0.45:
            continue
        canvas[int(v0 + fin_pitch):int(v1 - fin_pitch), int(u):int(u + cw)] = 0.90


def build_canvas(rng, kind, preset, mat_nm, strip_nm, bias):
    n = int(CANVAS_NM / CANVAS_PX_NM)
    canvas = np.full((n, n), 0.30, np.float32)
    mat, strip = _nm(mat_nm), _nm(strip_nm)
    block = _dram_block if kind == "dram" else _finfet_block
    v = rng.uniform(-mat, 0)
    boundaries = []
    while v < n:
        u = rng.uniform(-mat, 0)
        while u < n:
            u0, v0 = max(u, 0), max(v, 0)
            u1, v1 = min(u + mat, n), min(v + mat, n)
            if u1 - u0 > 20 and v1 - v0 > 20:
                block(canvas, u0, v0, u1, v1, preset, rng, bias)
                boundaries.append((u0 * CANVAS_PX_NM, v0 * CANVAS_PX_NM,
                                   u1 * CANVAS_PX_NM, v1 * CANVAS_PX_NM))
            u += mat + strip
        v += mat + strip
    for _ in range(rng.integers(6, 16)):
        w, h = rng.uniform(30, 260), rng.uniform(20, 120)
        if rng.random() < 0.5:
            w, h = h, w
        u0, v0 = rng.uniform(0, n - w), rng.uniform(0, n - h)
        canvas[int(v0):int(v0 + h), int(u0):int(u0 + w)] = rng.uniform(0.1, 0.95)
    return canvas, boundaries


def edge_brighten(img, rng):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag /= max(mag.max(), 1e-6)
    return img + rng.uniform(0.35, 0.9) * mag


def forward_map(cu, cv, theta, pixel_nm, out_px, barrel_k, shear_px, jitter):
    """Specimen position in nm imaged by every output pixel.

    This single function defines the geometry, and ground truth is obtained by
    numerically inverting it, so the recorded truth stays exact under barrel
    distortion, raster shear and row jitter rather than assuming the geometry
    is affine.
    """
    half = (out_px - 1) / 2.0
    idx = np.arange(out_px, dtype=np.float64)
    xx, yy = np.meshgrid(idx - half, idx - half)
    if barrel_k:
        r2 = (xx * xx + yy * yy) / (half * half)
        f = 1.0 + barrel_k * r2
        xx, yy = xx * f, yy * f
    if shear_px:
        xx = xx + shear_px * (yy / (2 * half))
    if jitter is not None:
        xx = xx + jitter[:, None]
    du, dv = xx * pixel_nm, yy * pixel_nm
    ct, st = np.cos(theta), np.sin(theta)
    return cu + du * ct - dv * st, cv + du * st + dv * ct


def sample_canvas(canvas, u, v):
    return map_coordinates(canvas, [v / CANVAS_PX_NM - 0.5, u / CANVAS_PX_NM - 0.5],
                           order=1, mode="nearest").astype(np.float32)


def solve_ground_truth(u_map, v_map, ru, rv):
    """Output pixel imaging the reference centre, to sub pixel accuracy."""
    d2 = (u_map - ru) ** 2 + (v_map - rv) ** 2
    r0, c0 = np.unravel_index(int(np.argmin(d2)), d2.shape)
    r0 = int(np.clip(r0, 1, u_map.shape[0] - 2))
    c0 = int(np.clip(c0, 1, u_map.shape[1] - 2))
    j = np.array([[(u_map[r0, c0 + 1] - u_map[r0, c0 - 1]) / 2.0,
                   (u_map[r0 + 1, c0] - u_map[r0 - 1, c0]) / 2.0],
                  [(v_map[r0, c0 + 1] - v_map[r0, c0 - 1]) / 2.0,
                   (v_map[r0 + 1, c0] - v_map[r0 - 1, c0]) / 2.0]])
    res = np.array([ru - u_map[r0, c0], rv - v_map[r0, c0]])
    try:
        dc, dr = np.linalg.solve(j, res)
    except np.linalg.LinAlgError:
        dc = dr = 0.0
    return float(c0 + np.clip(dc, -1, 1)), float(r0 + np.clip(dr, -1, 1))


def apply_imaging(img, rng, dose, spot_nm, pixel_nm, astig, variant, gamma,
                  vignette, streak_prob, streak_int, speckle, sp_prob):
    if spot_nm > 0:
        sx = max(spot_nm / pixel_nm, 0.3)
        img = gaussian_filter(img, (sx * astig, sx))
    n = img.shape[0]
    if vignette:
        idx = np.arange(n) - (n - 1) / 2.0
        xx, yy = np.meshgrid(idx, idx)
        r2 = (xx * xx + yy * yy) / (2.0 * ((n - 1) / 2.0) ** 2)
        img = img * (1.0 - vignette * r2)
    if streak_prob > 0 and streak_int > 0:
        for _ in range(int(rng.poisson(streak_prob * n / 100.0))):
            r0 = int(rng.integers(0, n))
            h = int(rng.integers(1, 6))
            img[r0:r0 + h] *= 1.0 + streak_int * rng.uniform(0.3, 1.0)
    img = np.clip(img, 0, None)
    img = rng.poisson(img * dose).astype(np.float32) / dose
    if speckle > 0:
        img = img * (1.0 + rng.normal(0, speckle, img.shape).astype(np.float32))
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma
    img8 = (img * 255).astype(np.uint8)
    if sp_prob > 0:
        m = rng.random(img8.shape)
        img8[m < sp_prob / 2] = 0
        img8[m > 1 - sp_prob / 2] = 255
    return img8


def generate_pair(seed, kind, variant):
    rng = np.random.default_rng(seed)
    preset = (DRAM_PRESETS if kind == "dram" else FINFET_PRESETS)[
        int(rng.integers(0, 6))]
    fscale = float(rng.uniform(0.7, 1.4))
    preset = {k: v * fscale for k, v in preset.items()}
    bias = float(rng.uniform(-6, 6))
    mat_nm = float(rng.uniform(900, 4800))
    strip_nm = float(rng.uniform(100, 760))
    canvas, boundaries = build_canvas(rng, kind, preset, mat_nm, strip_nm, bias)

    rounding = rng.uniform(0, 5)
    if rounding > 0.5:
        k = int(rounding) * 2 + 1
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    canvas = edge_brighten(canvas, rng)

    mag = float(rng.uniform(9.0, 11.0))
    search_pix = REF_PIX_NM * mag
    theta = float(np.deg2rad(rng.uniform(-2.0, 2.0)))
    barrel = float(rng.uniform(-BARREL_MAX, BARREL_MAX))
    shear = float(rng.uniform(0, 5.0)) if variant == "heavy_drift" else float(rng.uniform(0, 1.5))
    jitter_amp = 2.5 if variant == "heavy_drift" else 0.5
    jitter = rng.normal(0, jitter_amp, OUT_PX).astype(np.float64).cumsum() * 0.15

    cu = CANVAS_NM / 2 + rng.uniform(-200, 200)
    cv_ = CANVAS_NM / 2 + rng.uniform(-200, 200)
    margin = 90
    ct, st = np.cos(theta), np.sin(theta)

    def place(want_boundary):
        """Boundary straddling crops are placed deliberately, per the starter
        Space's boundary_bias parameter; the remainder is sampled uniformly at
        random exactly as that parameter's description states."""
        for _ in range(400):
            ox = rng.uniform(margin, OUT_PX - 1 - margin) - (OUT_PX - 1) / 2
            oy = rng.uniform(margin, OUT_PX - 1 - margin) - (OUT_PX - 1) / 2
            u = cu + (ox * ct - oy * st) * search_pix
            v = cv_ + (ox * st + oy * ct) * search_pix
            if not want_boundary:
                return u, v
            d = min(min(abs(u - b[0]), abs(u - b[2]), abs(v - b[1]), abs(v - b[3]))
                    for b in boundaries) if boundaries else 1e9
            if d < 600.0:
                return u, v
        return u, v

    straddles = bool(rng.random() < 0.35)
    ru, rv = place(straddles)

    dose_ref = float(rng.uniform(1200, 5000))
    dose_search = {"clean": rng.uniform(600, 2000), "low_dose": rng.uniform(20, 120)}.get(
        variant, rng.uniform(120, 600))
    spot = float(rng.uniform(1.0, 20.0))
    astig = float(rng.uniform(0.6, 1.9))
    gamma_r = float(rng.uniform(0.6, 1.6))
    gamma_s = float(rng.uniform(0.4, 2.5))
    sp = 0.03 if variant == "speckle_salt_pepper" else 0.0
    spk = 0.5 if variant == "speckle_salt_pepper" else 0.0
    streak_p = 4.0 if variant == "charging" else 0.0
    streak_i = 1.5 if variant == "charging" else 0.0

    ru_map, rv_map = forward_map(ru, rv, 0.0, REF_PIX_NM, OUT_PX, 0.0, 0.0, None)
    ref8 = apply_imaging(sample_canvas(canvas, ru_map, rv_map), rng, dose_ref,
                         max(spot * 0.4, 1.0), REF_PIX_NM, 1.0, "clean", gamma_r,
                         0.0, 0.0, 0.0, 0.0, 0.0)

    su_map, sv_map = forward_map(cu, cv_, theta, search_pix, OUT_PX,
                                 barrel, shear, jitter)
    canvas = gaussian_filter(canvas, (spot * astig / CANVAS_PX_NM,
                                      spot / CANVAS_PX_NM))
    vign = float(rng.uniform(0, 0.6))
    search8 = apply_imaging(sample_canvas(canvas, su_map, sv_map), rng, dose_search,
                            0.0, search_pix, 1.0, variant, gamma_s,
                            vign, streak_p, streak_i, spk, sp)

    gx, gy = solve_ground_truth(su_map, sv_map, ru, rv)
    meta = {
        "seed": int(seed), "style": kind, "variant": variant,
        "ground_truth": {"x": float(gx), "y": float(gy)},
        "magnification": mag, "relative_rotation_deg": float(np.rad2deg(theta)),
        "search_scale_error": float(mag / SEARCH_PIX_NM_NOMINAL - 1.0),
        "placement": "near_boundary" if straddles else "uniform",
        "variant_label": variant, "barrel_k": barrel, "shear_px": shear,
        "gamma_search": gamma_s, "salt_pepper": sp, "speckle": spk,
        "vignette": vign,
        "layout": {"mat_nm": mat_nm, "strip_nm": strip_nm, "feature_scale": fscale},
        "search_capture": {"pose": {"pixel_nm": search_pix},
                           "settings": {"dose_e": dose_search}},
    }
    return ref8, search8, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--barrel_max", type=float, default=0.02,
                    help="radial distortion amplitude; the session degradation list "
                         "does not include it, so the default is small and 0.15 "
                         "matches the widest setting exposed by the starter Space")
    args = ap.parse_args()
    global BARREL_MAX
    BARREL_MAX = args.barrel_max
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.num):
        kind = "dram" if i % 2 == 0 else "finfet"
        variant = VARIANTS[i % len(VARIANTS)]
        ref8, search8, meta = generate_pair(args.seed * 7919 + i, kind, variant)
        pd = args.out / f"pair_{i:04d}"
        pd.mkdir(exist_ok=True)
        cv2.imwrite(str(pd / "reference.png"), ref8)
        cv2.imwrite(str(pd / "search.png"), search8)
        (pd / "meta.json").write_text(json.dumps(meta, indent=2))
        g = meta["ground_truth"]
        rows.append({"pair_id": pd.name, "style": kind, "variant": variant,
                     "gt_x": f"{g['x']:.3f}", "gt_y": f"{g['y']:.3f}",
                     "magnification": f"{meta['magnification']:.3f}",
                     "rotation_deg": f"{meta['relative_rotation_deg']:.3f}"})
        print(f"{pd.name} {kind:6s} {variant:20s} mag={meta['magnification']:.2f} "
              f"gt=({g['x']:.1f}, {g['y']:.1f})")
    with open(args.out / "ground_truth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
