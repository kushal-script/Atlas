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


def _uniform_site(rng, search_pose, margin_px, out_px):
    """Specimen coordinate of a site drawn uniformly inside the search frame."""
    n = out_px
    c = rng.uniform(margin_px, n - 1 - margin_px)
    r = rng.uniform(margin_px, n - 1 - margin_px)
    p = search_pose["pixel_nm"]
    du = (c - (n - 1) / 2.0) * p
    dv = (r - (n - 1) / 2.0) * p
    cos_t, sin_t = np.cos(search_pose["theta_rad"]), np.sin(search_pose["theta_rad"])
    u = search_pose["center_u_nm"] + du * cos_t - dv * sin_t
    v = search_pose["center_v_nm"] + du * sin_t + dv * cos_t
    return u, v


def _far_site(rng, search_pose, extent_nm, frame_radius_nm):
    """Specimen coordinate of a site guaranteed to lie outside the search frame.

    Used for Set C (no true instance): the reference structure is placed here so it
    cannot appear in the search image, while the search still shows unrelated layout.
    """
    cu = search_pose["center_u_nm"]
    cv = search_pose["center_v_nm"]
    lo, hi = extent_nm * 0.12, extent_nm * 0.88
    for _ in range(64):
        u = rng.uniform(lo, hi)
        v = rng.uniform(lo, hi)
        if np.hypot(u - cu, v - cv) > 0.35 * extent_nm:
            return u, v
    return extent_nm - cu, extent_nm - cv


def generate_pair(seed, style, cfg=None, modality="sem", absent=False, phase2=False):
    cfg = cfg or GeneratorConfig()
    if phase2:
        # Phase 2 widens the unknown zoom to 8..12x (scale error +/-0.20) and the
        # unknown rotation to +/-5 deg (CCW positive). The localizer's grid is
        # built to match these ranges (see phase2_config()).
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.pose.rotation_deg_sigma = 2.5
        cfg.pose.rotation_deg_max = 5.0
        cfg.pose.scale_err_sigma = 0.1
        cfg.pose.scale_err_max = 0.2
    seq = np.random.SeedSequence(seed)
    rng_layout, rng_canvas, rng_pose, rng_ref, rng_search = [
        np.random.default_rng(s) for s in seq.spawn(5)]

    size = cfg.canvas.size_px
    mat = np.zeros((size, size), dtype=np.uint8)
    hgt = np.zeros((size, size), dtype=np.float32)
    style_params = cfg.dram if style == "dram" else cfg.finfet
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

    # The reference site is drawn uniformly in the search frame first, and the
    # layout is then generated so the requested local structure lands on it.
    # Searching a finished layout for a structure instead would bias where the
    # site falls in the frame, because large structures sit preferentially near
    # the frame centre, and that bias would flatter any decision rule that
    # favours the frame centre.
    strategies, weights = zip(*cfg.placement_mix)
    strategy = str(rng_pose.choice(strategies, p=np.array(weights) / sum(weights)))
    if absent:
        # Set C: no true instance. Draw the reference site well outside the search
        # frame so the reference structure never appears in the search image; the
        # search still carries unrelated layout, so a matcher may still emit a
        # spurious peak that rejection must catch.
        ru, rv = _far_site(rng_pose, search_pose, extent,
                           cfg.search.out_px * cfg.search.pixel_nm * 1.5)
        want = None
    else:
        ru, rv = _uniform_site(rng_pose, search_pose, pp.ref_margin_px, cfg.search.out_px)
        want = {"deep_array": "deep", "near_boundary": "boundary"}.get(strategy)
    zones = LAYOUT_BUILDERS[style](mat, hgt, cfg.canvas.pixel_nm, rng_layout,
                                   style_params, target=(ru, rv), want=want)
    se_info = {}
    se_sub = None
    sub_mat = None
    if modality == "sem":
        se, se_info = build_se_canvas(mat, hgt, cfg.canvas.pixel_nm, rng_canvas,
                                      cfg.canvas)
        se_sub = None
        if absent:
            # Set C must be a *genuine* negative: render the search over a
            # substrate-only canvas so no CDU-like structure can produce a
            # spurious match. A correlation matcher cannot distinguish a true
            # CDU instance from a *different* CDU of the same type, so negatives
            # are defined as regions with no reference-like structure.
            sub_mat = np.zeros_like(mat)
            sub_hgt = np.zeros_like(hgt)
            se_sub, _ = build_se_canvas(sub_mat, sub_hgt, cfg.canvas.pixel_nm,
                                        rng_canvas, cfg.canvas)

    ref_pose = {"center_u_nm": ru, "center_v_nm": rv,
                "theta_rad": theta_r, "pixel_nm": cfg.reference.pixel_nm}

    if modality == "optical":
        from .imaging.optical import render_optical_capture, sample_optical_settings
        opt_ref = sample_optical_settings(rng_ref, "reference")
        opt_search = sample_optical_settings(rng_search, "search")
        ref_img, ref_meta = render_optical_capture(
            mat, hgt, cfg.canvas.pixel_nm, ref_pose, opt_ref, rng_ref)
        search_img, search_meta = render_optical_capture(
            mat, hgt, cfg.canvas.pixel_nm, search_pose, opt_search, rng_search)
        zero = np.zeros(cfg.search.out_px, dtype=np.float32)
        dx_r = dy_r = dx_s = dy_s = zero
    else:
        ref_img, dx_r, dy_r, ref_meta = render_capture(
            se, mat, cfg.canvas.pixel_nm, ref_pose, cfg.reference, rng_ref)
        search_src = se_sub if (absent and se_sub is not None) else se
        search_mat = sub_mat if (absent and se_sub is not None) else mat
        search_img, dx_s, dy_s, search_meta = render_capture(
            search_src, search_mat, cfg.canvas.pixel_nm, search_pose, cfg.search,
            rng_search)

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
        "modality": modality,
        "placement": strategy,
        "present": (not absent),
        "ground_truth": ({"x": None, "y": None}
                        if absent else {"x": float(gt_c), "y": float(gt_r)}),
        "gt_corners_xy": (None if absent else corners),
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
