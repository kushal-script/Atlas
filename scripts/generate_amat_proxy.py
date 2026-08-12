"""Faithful proxy of the organiser reference pipeline, used to predict hidden
test set performance.

This is an independent reimplementation of the published Applied Materials
starter Space (structure presets, zone composition, and the documented SEM
imaging chain with its exact order of operations and formulas) together with
the four noise tiers and five acquisition variants used by that project's own
evaluation harness. It exists so the localizer can be measured against the
distribution the official test set is most likely drawn from, rather than only
against our own physics generator.

Differences from our primary generator that matter, all reproduced here:
the fine canvas is built at 1 nm per pixel over the full 10 um field and the
search image is produced by blurring that canvas and decimating it with area
averaging, so the search image is anti aliased rather than point sampled; the
lattice is placed by a random walk in pitch, so it drifts and is not exactly
periodic; structure is composed into mats separated by routing strips, with a
different preset per mat; the magnification is exactly 10 to 1 with no
rotation unless robustness flags are given; and gamma, speckle, impulse noise
and astigmatism are applied identically to both captures while shear, jitter,
vignetting and radial distortion are applied asymmetrically.

Usage:
    python scripts/generate_amat_proxy.py --num 40 --out data/amat40 --seed 5
    python scripts/generate_amat_proxy.py --num 40 --out data/amat_hard --seed 5 \
        --tier severe --rotation_deg 2 --scale_jitter 0.1
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

FINE_PX = 10000
OUT_PX = 1000
FACTOR = 10

BACKGROUND, LINE_A, LINE_B, CONTACT = 40, 150, 170, 225
STRIP_BASE, STRIP_LINE = 95, 128
STRIP_LINE_PITCH, STRIP_LINE_WIDTH = 220, 9
DRAM_POS_JITTER, FINFET_POS_JITTER = 1.5, 1.0
WIDTH_JITTER_FRACTION = 0.10
COLLAPSE_PROB = 0.7

DRAM_PRESETS = [
    {"word_line_pitch_nm": 64, "word_line_width_nm": 32, "bit_line_pitch_nm": 96,
     "bit_line_width_nm": 32, "contact_diameter_nm": 32},
    {"word_line_pitch_nm": 48, "word_line_width_nm": 24, "bit_line_pitch_nm": 72,
     "bit_line_width_nm": 24, "contact_diameter_nm": 24},
    {"word_line_pitch_nm": 96, "word_line_width_nm": 48, "bit_line_pitch_nm": 144,
     "bit_line_width_nm": 48, "contact_diameter_nm": 48},
    {"word_line_pitch_nm": 120, "word_line_width_nm": 56, "bit_line_pitch_nm": 180,
     "bit_line_width_nm": 60, "contact_diameter_nm": 58},
    {"word_line_pitch_nm": 72, "word_line_width_nm": 30, "bit_line_pitch_nm": 108,
     "bit_line_width_nm": 34, "contact_diameter_nm": 30},
    {"word_line_pitch_nm": 160, "word_line_width_nm": 78, "bit_line_pitch_nm": 240,
     "bit_line_width_nm": 80, "contact_diameter_nm": 78},
]

FINFET_PRESETS = [
    {"fin_pitch_nm": 48, "fin_width_nm": 16, "gate_pitch_nm": 90,
     "gate_length_nm": 28, "contact_size_nm": 28},
    {"fin_pitch_nm": 40, "fin_width_nm": 14, "gate_pitch_nm": 76,
     "gate_length_nm": 24, "contact_size_nm": 24},
    {"fin_pitch_nm": 60, "fin_width_nm": 20, "gate_pitch_nm": 110,
     "gate_length_nm": 34, "contact_size_nm": 34},
    {"fin_pitch_nm": 80, "fin_width_nm": 26, "gate_pitch_nm": 150,
     "gate_length_nm": 46, "contact_size_nm": 44},
    {"fin_pitch_nm": 96, "fin_width_nm": 32, "gate_pitch_nm": 180,
     "gate_length_nm": 56, "contact_size_nm": 52},
    {"fin_pitch_nm": 140, "fin_width_nm": 46, "gate_pitch_nm": 260,
     "gate_length_nm": 80, "contact_size_nm": 76},
]

TIERS = {
    "low": {"dose_search": 800.0, "detector_noise_sigma_search": 2.0,
            "shear_amplitude_px": 0.5, "drift_jitter_px": 0.2},
    "medium": {"dose_search": 200.0, "detector_noise_sigma_search": 5.0,
               "shear_amplitude_px": 1.5, "drift_jitter_px": 0.5},
    "high": {"dose_search": 60.0, "detector_noise_sigma_search": 8.0,
             "shear_amplitude_px": 2.5, "drift_jitter_px": 1.0,
             "speckle_sigma": 0.15},
    "severe": {"dose_search": 25.0, "detector_noise_sigma_search": 12.0,
               "shear_amplitude_px": 4.0, "drift_jitter_px": 1.8,
               "speckle_sigma": 0.3, "salt_pepper_prob": 0.01},
}

VARIANTS = {
    "clean": {"dose_search": 900.0, "shear_amplitude_px": 0.3, "drift_jitter_px": 0.15},
    "low_dose": {"dose_search": 55.0, "shear_amplitude_px": 1.0, "drift_jitter_px": 0.4},
    "heavy_drift": {"shear_amplitude_px": 4.5, "drift_jitter_px": 2.0},
    "speckle_salt_pepper": {"speckle_sigma": 0.3, "salt_pepper_prob": 0.012},
    "charging": {"charging_streak_prob": 3.5, "charging_streak_intensity": 2.2},
}

BASE_PARAMS = {
    "beam_spot_size_nm": 5.0, "collapse_threshold_nm": 10.0,
    "dose_reference": 2000.0, "dose_search": 200.0,
    "shear_amplitude_px": 1.5, "drift_jitter_px": 0.5,
    "detector_noise_sigma_ref": 2.0, "detector_noise_sigma_search": 5.0,
    "astigmatism_ratio": 1.0, "vignette_strength": 0.0, "gamma": 1.0,
    "barrel_distortion_k": 0.0, "charging_streak_prob": 0.0,
    "charging_streak_intensity": 0.0, "speckle_sigma": 0.0,
    "salt_pepper_prob": 0.0, "linewidth_bias_nm": 0.0, "corner_rounding_px": 0.0,
    "mat_size_nm": 2600.0, "strip_width_nm": 320.0,
}


def _line_positions(size_px, pitch_nm, jitter_nm, rng):
    pos = rng.uniform(0, pitch_nm)
    out = []
    while pos < size_px:
        out.append(pos)
        pos += pitch_nm + rng.normal(0, jitter_nm)
    return np.array(out)


def _line_mask(size_px, positions, width_nm, collapse_nm, rng, bias_nm):
    mask = np.zeros(size_px, bool)
    base = max(width_nm + bias_nm, 1.0)
    widths = base * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION, size=len(positions)))
    widths = np.clip(widths, base * 0.5, base * 1.5)
    for i, c in enumerate(positions):
        hw = widths[i] / 2.0
        mask[max(int(round(c - hw)), 0):min(int(round(c + hw)), size_px)] = True
        if i + 1 < len(positions):
            nhw = widths[i + 1] / 2.0
            gap = (positions[i + 1] - nhw) - (c + hw)
            if gap < collapse_nm and rng.random() < COLLAPSE_PROB:
                mask[max(int(round(c + hw)), 0):
                     min(int(round(positions[i + 1] - nhw)), size_px)] = True
    return mask


def _dram_tile(h, w, preset, collapse_nm, rng, bias, rounding):
    tile = np.full((h, w), BACKGROUND, np.uint8)
    wl = _line_positions(h, preset["word_line_pitch_nm"], DRAM_POS_JITTER, rng)
    bl = _line_positions(w, preset["bit_line_pitch_nm"], DRAM_POS_JITTER, rng)
    tile[_line_mask(h, wl, preset["word_line_width_nm"], collapse_nm, rng, bias), :] = LINE_A
    tile[:, _line_mask(w, bl, preset["bit_line_width_nm"], collapse_nm, rng, bias)] = LINE_B
    r = max(int(round((preset["contact_diameter_nm"] + bias) / 2.0)), 1)
    for cy in wl:
        for cx in bl:
            y, x = int(round(cy)), int(round(cx))
            if 0 <= y < h and 0 <= x < w and rng.random() > 0.01:
                cv2.circle(tile, (x, y), r, CONTACT, -1)
    if rounding > 0.5:
        k = int(rounding) * 2 + 1
        tile = cv2.morphologyEx(tile, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return tile


def _finfet_tile(h, w, preset, collapse_nm, rng, bias, rounding):
    tile = np.full((h, w), BACKGROUND, np.uint8)
    fins = _line_positions(h, preset["fin_pitch_nm"], FINFET_POS_JITTER, rng)
    gates = _line_positions(w, preset["gate_pitch_nm"], FINFET_POS_JITTER, rng)
    tile[_line_mask(h, fins, preset["fin_width_nm"], collapse_nm, rng, bias), :] = LINE_A
    tile[:, _line_mask(w, gates, preset["gate_length_nm"], collapse_nm, rng, bias)] = LINE_B
    cw = max(int(round((preset["contact_size_nm"] + bias) * 0.5)), 1)
    for i in range(len(gates) - 1):
        if rng.random() < 0.45:
            continue
        mid = int(round((gates[i] + gates[i + 1]) / 2.0))
        if 0 <= mid < w - cw:
            tile[:, mid:mid + cw] = CONTACT
    if rounding > 0.5:
        k = int(rounding) * 2 + 1
        tile = cv2.morphologyEx(tile, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return tile


def _zone_spans(size_px, mat_nm, strip_nm):
    spans, pos, is_mat = [], 0.0, True
    while pos < size_px:
        end = min(pos + (mat_nm if is_mat else strip_nm), size_px)
        spans.append((is_mat, int(round(pos)), int(round(end))))
        pos = end
        is_mat = not is_mat
    return spans


def _strip_texture(size_px, rng):
    canvas = np.full((size_px, size_px), STRIP_BASE, np.uint8)
    half = STRIP_LINE_WIDTH / 2.0
    for axis in (0, 1):
        phase = rng.uniform(0, STRIP_LINE_PITCH)
        for c in np.arange(phase, size_px, STRIP_LINE_PITCH):
            lo, hi = max(int(round(c - half)), 0), min(int(round(c + half)), size_px)
            if axis == 0:
                canvas[lo:hi, :] = STRIP_LINE
            else:
                canvas[:, lo:hi] = STRIP_LINE
    return canvas


def build_zone_canvas(size_px, kind, rng, params):
    canvas = _strip_texture(size_px, rng)
    presets = DRAM_PRESETS if kind == "dram" else FINFET_PRESETS
    tile_fn = _dram_tile if kind == "dram" else _finfet_tile
    spans = _zone_spans(size_px, params["mat_size_nm"], params["strip_width_nm"])
    mat_rects, strip_rects = [], []
    for is_row_mat, y0, y1 in spans:
        for is_col_mat, x0, x1 in spans:
            if is_row_mat and is_col_mat:
                preset = presets[int(rng.integers(0, len(presets)))]
                canvas[y0:y1, x0:x1] = tile_fn(
                    y1 - y0, x1 - x0, preset, params["collapse_threshold_nm"], rng,
                    params["linewidth_bias_nm"], params["corner_rounding_px"])
                mat_rects.append((x0, y0, x1 - x0, y1 - y0))
            else:
                strip_rects.append((x0, y0, x1 - x0, y1 - y0))
    return canvas, mat_rects, strip_rects


def gaussian_psf_blur(img, spot_nm, pixel_nm, astig):
    sx = max(spot_nm / pixel_nm, 1e-6)
    sy = max(sx * astig, 1e-6)
    k = max(int(2 * round(3 * max(sx, sy)) + 1), 3)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sx, sigmaY=sy)


def apply_raster_drift(img, shear_px, jitter_px, rng):
    if shear_px == 0 and jitter_px == 0:
        return img
    h, w = img.shape
    shear = shear_px * (np.arange(h) / max(h - 1, 1))
    jitter = rng.normal(0, jitter_px, size=h) if jitter_px > 0 else np.zeros(h)
    shift = (shear + jitter).astype(np.float32)
    map_x = np.arange(w, dtype=np.float32)[None, :] + shift[:, None]
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def apply_barrel(img, k):
    if k == 0.0:
        return img
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - cx) / cx, (yy - cy) / cy
    f = 1.0 + k * (nx ** 2 + ny ** 2)
    return cv2.remap(img, (nx * f) * cx + cx, (ny * f) * cy + cy,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def add_shot_noise(img, dose, rng):
    counts = np.clip(img.astype(np.float64) / 255.0 * dose, 0, None)
    return np.clip(rng.poisson(counts) / dose * 255.0, 0, 255).astype(np.uint8)


def add_detector_noise(img, sigma, rng):
    if sigma <= 0:
        return img
    return np.clip(img.astype(np.float64) + rng.normal(0, sigma, img.shape),
                   0, 255).astype(np.uint8)


def add_speckle(img, sigma, rng):
    if sigma <= 0:
        return img
    return np.clip(img.astype(np.float64) * (1.0 + rng.normal(0, sigma, img.shape)),
                   0, 255).astype(np.uint8)


def add_salt_pepper(img, prob, rng):
    if prob <= 0:
        return img
    out = img.copy()
    hit = rng.random(img.shape) < prob
    salt = rng.random(img.shape) < 0.5
    out[hit & salt] = 255
    out[hit & ~salt] = 0
    return out


def apply_vignette(img, strength):
    if strength <= 0:
        return img
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.clip(np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2) / np.sqrt(2), 0, 1)
    return np.clip(img.astype(np.float64) * (1.0 - strength * r ** 2), 0, 255).astype(np.uint8)


def apply_gamma(img, gamma):
    if gamma == 1.0:
        return img
    return np.clip(np.power(np.clip(img.astype(np.float64) / 255.0, 0, 1), gamma) * 255.0,
                   0, 255).astype(np.uint8)


def add_charging_streaks(img, prob, intensity, rng):
    if prob <= 0 or intensity <= 0:
        return img
    h, w = img.shape
    out = img.astype(np.float64)
    for _ in range(rng.poisson(max(prob * (h / 100.0), 0))):
        row = int(rng.integers(0, h))
        band = max(1, int(rng.normal(2, 1)))
        out[max(row - band, 0):min(row + band, h), :] += (
            intensity * rng.uniform(0.5, 1.0) * 255.0 / 10.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def image_reference(crop, p, rng):
    img = gaussian_psf_blur(crop, p["beam_spot_size_nm"], 1.0, p["astigmatism_ratio"])
    img = apply_raster_drift(img, 0.0, p["drift_jitter_px"] * 0.2, rng)
    img = apply_barrel(img, p["barrel_distortion_k"] * 0.3)
    img = add_shot_noise(img, p["dose_reference"], rng)
    img = add_detector_noise(img, p["detector_noise_sigma_ref"], rng)
    img = add_speckle(img, p["speckle_sigma"], rng)
    img = add_salt_pepper(img, p["salt_pepper_prob"], rng)
    img = apply_vignette(img, p["vignette_strength"] * 0.5)
    img = apply_gamma(img, p["gamma"])
    return add_charging_streaks(img, p["charging_streak_prob"],
                                p["charging_streak_intensity"], rng)


def image_search(canvas, p, rng, rotation_deg=0.0, scale=1.0):
    blurred = gaussian_psf_blur(canvas, p["beam_spot_size_nm"], 1.0,
                                p["astigmatism_ratio"])
    img = cv2.resize(blurred, (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)
    if rotation_deg != 0.0 or scale != 1.0:
        m = cv2.getRotationMatrix2D(((OUT_PX - 1) / 2.0, (OUT_PX - 1) / 2.0),
                                    rotation_deg, 1.0 / scale)
        img = cv2.warpAffine(img, m, (OUT_PX, OUT_PX), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    img = apply_raster_drift(img, p["shear_amplitude_px"], p["drift_jitter_px"], rng)
    img = apply_barrel(img, p["barrel_distortion_k"])
    img = add_shot_noise(img, p["dose_search"], rng)
    img = add_detector_noise(img, p["detector_noise_sigma_search"], rng)
    img = add_speckle(img, p["speckle_sigma"], rng)
    img = add_salt_pepper(img, p["salt_pepper_prob"], rng)
    img = apply_vignette(img, p["vignette_strength"])
    img = apply_gamma(img, p["gamma"])
    return add_charging_streaks(img, p["charging_streak_prob"],
                                p["charging_streak_intensity"], rng)


def generate_pair(seed, kind, params, rotation_deg=0.0, scale=1.0, boundary_bias=0.35):
    rng = np.random.default_rng(seed)
    canvas, mat_rects, strip_rects = build_zone_canvas(FINE_PX, kind, rng, params)

    max_off = FINE_PX - OUT_PX
    if strip_rects and rng.random() < boundary_bias:
        sx, sy, sw, sh = strip_rects[int(rng.integers(0, len(strip_rects)))]
        x0 = int(np.clip(sx + sw // 2 - OUT_PX // 2 + rng.integers(-400, 401), 0, max_off))
        y0 = int(np.clip(sy + sh // 2 - OUT_PX // 2 + rng.integers(-400, 401), 0, max_off))
        straddles = True
    else:
        x0 = int(rng.integers(0, max_off + 1))
        y0 = int(rng.integers(0, max_off + 1))
        straddles = False

    crop = canvas[y0:y0 + OUT_PX, x0:x0 + OUT_PX]
    ref = image_reference(crop, params, rng)
    search = image_search(canvas, params, rng, rotation_deg, scale)

    gt_x = x0 / FACTOR + (OUT_PX // FACTOR) / 2.0
    gt_y = y0 / FACTOR + (OUT_PX // FACTOR) / 2.0
    if rotation_deg != 0.0 or scale != 1.0:
        c = (OUT_PX - 1) / 2.0
        m = cv2.getRotationMatrix2D((c, c), rotation_deg, 1.0 / scale)
        inv = cv2.invertAffineTransform(m)
        src = np.array([gt_x, gt_y, 1.0])
        fwd = np.linalg.inv(np.vstack([inv, [0, 0, 1]]))
        p = fwd @ src
        gt_x, gt_y = float(p[0]), float(p[1])

    meta = {
        "seed": int(seed), "style": kind,
        "ground_truth": {"x": float(gt_x), "y": float(gt_y)},
        "gt_box_xywh": [x0 / FACTOR, y0 / FACTOR, OUT_PX // FACTOR, OUT_PX // FACTOR],
        "crop_origin_fine_px": [x0, y0],
        "placement": "near_boundary" if straddles else "uniform",
        "relative_rotation_deg": float(rotation_deg),
        "search_scale_error": float(scale - 1.0),
        "params": {k: float(v) if isinstance(v, (int, float)) else v
                   for k, v in params.items()},
        "search_capture": {"pose": {"pixel_nm": 10.0 * scale},
                           "settings": {"dose_e": params["dose_search"]}},
    }
    return ref, search, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--tier", choices=list(TIERS) + ["cycle", "variants"], default="cycle")
    ap.add_argument("--rotation_deg", type=float, default=0.0,
                    help="robustness only, the reference pipeline has no rotation")
    ap.add_argument("--scale_jitter", type=float, default=0.0,
                    help="robustness only, magnification error amplitude")
    ap.add_argument("--boundary_bias", type=float, default=0.35)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tier_names = list(TIERS) if args.tier == "cycle" else (
        list(VARIANTS) if args.tier == "variants" else [args.tier])
    rows = []
    for i in range(args.num):
        kind = "dram" if i % 2 == 0 else "finfet"
        name = tier_names[i % len(tier_names)]
        params = dict(BASE_PARAMS)
        params.update(TIERS.get(name, {}))
        params.update(VARIANTS.get(name, {}))
        rng0 = np.random.default_rng(args.seed * 104729 + i)
        rot = float(rng0.uniform(-args.rotation_deg, args.rotation_deg)) if args.rotation_deg else 0.0
        scale = float(1.0 + rng0.uniform(-args.scale_jitter, args.scale_jitter)) if args.scale_jitter else 1.0
        ref, search, meta = generate_pair(args.seed * 7919 + i, kind, params, rot, scale,
                                          args.boundary_bias)
        meta["tier"] = name
        pd = args.out / f"pair_{i:04d}"
        pd.mkdir(exist_ok=True)
        cv2.imwrite(str(pd / "reference.png"), ref)
        cv2.imwrite(str(pd / "search.png"), search)
        (pd / "meta.json").write_text(json.dumps(meta, indent=2))
        g = meta["ground_truth"]
        rows.append({"pair_id": pd.name,
                     "reference_path": f"{pd.name}/reference.png",
                     "search_path": f"{pd.name}/search.png",
                     "style": kind, "tier": name,
                     "gt_x": f"{g['x']:.3f}", "gt_y": f"{g['y']:.3f}",
                     "relative_rotation_deg": f"{rot:.3f}",
                     "search_scale_error": f"{scale - 1.0:.5f}",
                     "placement": meta["placement"]})
        print(f"{pd.name} {kind:6s} {name:20s} gt=({g['x']:.1f}, {g['y']:.1f}) "
              f"{meta['placement']}")
    with open(args.out / "ground_truth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
