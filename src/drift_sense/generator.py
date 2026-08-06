"""Image pair generation: builds a layout canvas, runs the SEM image formation
model twice with independent noise generators, and records exact ground truth
by mapping the reference capture center through both capture transforms."""

import json
import numpy as np
from PIL import Image, ImageDraw

from .geometry import LAYOUT_BUILDERS
from .imaging.sem import (build_se_canvas, capture_to_specimen, render_capture,
                          specimen_to_capture)
from .params import GeneratorConfig


def _rect_distance(u, v, rect):
    u0, v0, u1, v1 = rect
    du = max(u0 - u, 0.0, u - u1)
    dv = max(v0 - v, 0.0, v - v1)
    return float(np.hypot(du, dv))


def _inside_rect(u, v, rect, inset):
    u0, v0, u1, v1 = rect
    return (u0 + inset <= u <= u1 - inset) and (v0 + inset <= v <= v1 - inset)


def _pick_placement(rng, strategy, zones, search_pose, margin_px, out_px):
    n = out_px
    for _ in range(300):
        c = rng.uniform(margin_px, n - 1 - margin_px)
        r = rng.uniform(margin_px, n - 1 - margin_px)
        p = search_pose["pixel_nm"]
        du = (c - (n - 1) / 2.0) * p
        dv = (r - (n - 1) / 2.0) * p
        cos_t, sin_t = np.cos(search_pose["theta_rad"]), np.sin(search_pose["theta_rad"])
        u = search_pose["center_u_nm"] + du * cos_t - dv * sin_t
        v = search_pose["center_v_nm"] + du * sin_t + dv * cos_t
        if strategy == "uniform":
            return u, v
        if strategy == "deep_array":
            if any(_inside_rect(u, v, z, 400.0) for z in zones["periodic_zones"]):
                if not zones["anchor_zones"] or all(
                        _rect_distance(u, v, z) > 600.0 for z in zones["anchor_zones"]):
                    return u, v
        if strategy == "near_boundary":
            if zones["anchor_zones"]:
                if any(_rect_distance(u, v, z) < 450.0 for z in zones["anchor_zones"]):
                    return u, v
            else:
                if all(not _inside_rect(u, v, z, -300.0) for z in zones["periodic_zones"]):
                    return u, v
    c = rng.uniform(margin_px, n - 1 - margin_px)
    r = rng.uniform(margin_px, n - 1 - margin_px)
    p = search_pose["pixel_nm"]
    du = (c - (n - 1) / 2.0) * p
    dv = (r - (n - 1) / 2.0) * p
    u = search_pose["center_u_nm"] + du
    v = search_pose["center_v_nm"] + dv
    return u, v


def generate_pair(seed, style, cfg=None):
    cfg = cfg or GeneratorConfig()
    seq = np.random.SeedSequence(seed)
    rng_layout, rng_canvas, rng_pose, rng_ref, rng_search = [
        np.random.default_rng(s) for s in seq.spawn(5)]

    size = cfg.canvas.size_px
    mat = np.zeros((size, size), dtype=np.uint8)
    hgt = np.zeros((size, size), dtype=np.float32)
    style_params = cfg.dram if style == "dram" else cfg.finfet
    zones = LAYOUT_BUILDERS[style](mat, hgt, cfg.canvas.pixel_nm, rng_layout, style_params)
    se, se_info = build_se_canvas(mat, hgt, cfg.canvas.pixel_nm, rng_canvas, cfg.canvas)

    extent = cfg.canvas.extent_nm
    pp = cfg.pose
    theta_s = float(np.clip(rng_pose.normal(0.0, np.deg2rad(pp.rotation_deg_sigma)),
                            -np.deg2rad(pp.rotation_deg_max), np.deg2rad(pp.rotation_deg_max)))
    theta_r = float(rng_pose.normal(0.0, np.deg2rad(0.15)))
    scale_err = float(np.clip(rng_pose.normal(0.0, pp.scale_err_sigma),
                              -pp.scale_err_max, pp.scale_err_max))
    search_pose = {
        "center_u_nm": extent / 2.0 + float(rng_pose.uniform(-pp.search_center_jitter_nm,
                                                             pp.search_center_jitter_nm)),
        "center_v_nm": extent / 2.0 + float(rng_pose.uniform(-pp.search_center_jitter_nm,
                                                             pp.search_center_jitter_nm)),
        "theta_rad": theta_s,
        "pixel_nm": cfg.search.pixel_nm * (1.0 + scale_err),
    }

    strategies, weights = zip(*cfg.placement_mix)
    strategy = str(rng_pose.choice(strategies, p=np.array(weights) / sum(weights)))
    ru, rv = _pick_placement(rng_pose, strategy, zones, search_pose,
                             pp.ref_margin_px, cfg.search.out_px)
    ref_pose = {"center_u_nm": ru, "center_v_nm": rv,
                "theta_rad": theta_r, "pixel_nm": cfg.reference.pixel_nm}

    ref_img, dx_r, dy_r, ref_meta = render_capture(
        se, mat, cfg.canvas.pixel_nm, ref_pose, cfg.reference, rng_ref)
    search_img, dx_s, dy_s, search_meta = render_capture(
        se, mat, cfg.canvas.pixel_nm, search_pose, cfg.search, rng_search)

    nr = cfg.reference.out_px
    center = (nr - 1) / 2.0
    pu, pv = capture_to_specimen(center, center, ref_pose, dx_r, dy_r, nr)
    gt_c, gt_r = specimen_to_capture(pu, pv, search_pose, dx_s, dy_s, cfg.search.out_px)

    corners = []
    for (rr, cc) in [(0.0, 0.0), (0.0, nr - 1.0), (nr - 1.0, nr - 1.0), (nr - 1.0, 0.0)]:
        cu, cv2 = capture_to_specimen(rr, cc, ref_pose, dx_r, dy_r, nr)
        sc, sr = specimen_to_capture(cu, cv2, search_pose, dx_s, dy_s, cfg.search.out_px)
        corners.append([float(sc), float(sr)])

    meta = {
        "seed": int(seed),
        "style": style,
        "placement": strategy,
        "ground_truth": {"x": float(gt_c), "y": float(gt_r)},
        "gt_corners_xy": corners,
        "relative_rotation_deg": float(np.rad2deg(theta_s - theta_r)),
        "search_scale_error": scale_err,
        "layout": zones["info"],
        "se_model": se_info,
        "reference_capture": ref_meta,
        "search_capture": search_meta,
        "pixel_convention": "origin at center of top left pixel, x rightward column, y downward row",
    }
    return {"reference": ref_img, "search": search_img, "meta": meta}


def save_pair(out_dir, index, result, preview=False):
    pair_dir = out_dir / f"pair_{index:04d}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result["reference"]).save(pair_dir / "reference.png")
    Image.fromarray(result["search"]).save(pair_dir / "search.png")
    with open(pair_dir / "meta.json", "w") as fh:
        json.dump(result["meta"], fh, indent=2)
    if preview:
        prev_dir = out_dir / "previews"
        prev_dir.mkdir(exist_ok=True)
        img = Image.fromarray(result["search"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        pts = [tuple(p) for p in result["meta"]["gt_corners_xy"]]
        draw.polygon(pts, outline=(255, 64, 64))
        gx = result["meta"]["ground_truth"]["x"]
        gy = result["meta"]["ground_truth"]["y"]
        draw.line([(gx - 8, gy), (gx + 8, gy)], fill=(255, 64, 64))
        draw.line([(gx, gy - 8), (gx, gy + 8)], fill=(255, 64, 64))
        img.save(prev_dir / f"pair_{index:04d}.png")
    return pair_dir
