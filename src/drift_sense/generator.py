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


# Severity ladder for the Phase 2 degraded set. The addendum names the
# degradations, charging, scan distortion, defocus, elevated shot noise and
# polygon scaling to twenty percent, and discloses that there are four severity
# levels, but not the parameters. These values are therefore this repository's
# own reading, spanning from mild to well past where the Phase 1 noise sits.
# Every factor scales the search capture only: in the organiser pipeline the
# reference is a clean crop and the degradations corrupt the wide image, so a
# degradation shared by both captures would train against the wrong problem.
DEGRADE_LADDER = {
    1: {"dose": 0.70, "psf": 1.3, "charge": 1.5, "drift_px": 2.0, "jitter": 2.0, "poly": 0.05},
    2: {"dose": 0.45, "psf": 1.7, "charge": 2.2, "drift_px": 4.0, "jitter": 3.0, "poly": 0.10},
    3: {"dose": 0.28, "psf": 2.2, "charge": 3.0, "drift_px": 7.0, "jitter": 4.0, "poly": 0.15},
    4: {"dose": 0.15, "psf": 3.0, "charge": 4.0, "drift_px": 10.0, "jitter": 6.0, "poly": 0.20},
}


def _scale_widths(style_params, factor):
    """Style params with every drawn feature width scaled, pitches untouched.

    Widths are sampled as pitch times a fraction drawn from a range, so scaling
    the range endpoints scales the drawn width by exactly the factor while the
    rng call sequence, and with it the lattice pitch and phase, is unchanged.
    """
    from dataclasses import replace
    fields = {}
    for name in ("wl_duty", "bl_duty", "contact_radius_f",
                 "fin_width_frac", "gate_width_frac"):
        if hasattr(style_params, name):
            lo, hi = getattr(style_params, name)
            fields[name] = (min(lo * factor, 0.95), min(hi * factor, 0.95))
    return replace(style_params, **fields)


def generate_pair(seed, style, cfg=None, modality="sem", absent=False, degrade=0):
    cfg = cfg or GeneratorConfig()
    seq = np.random.SeedSequence(seed)
    children = seq.spawn(8)
    (rng_layout, rng_canvas, rng_pose, rng_ref, rng_search,
     rng_layout2, rng_canvas2, rng_degrade) = [np.random.default_rng(s) for s in children]

    size = cfg.canvas.size_px
    mat = np.zeros((size, size), dtype=np.uint8)
    hgt = np.zeros((size, size), dtype=np.float32)
    style_params = cfg.dram if style == "dram" else cfg.finfet
    extent = cfg.canvas.extent_nm
    pp = cfg.pose
    theta_r = float(rng_pose.normal(0.0, np.deg2rad(0.15)))
    if cfg.phase2:
        # The relative rotation is drawn directly rather than as the difference
        # of two capture angles, so it cannot land outside the disclosed range.
        rel = float(rng_pose.uniform(-pp.rel_rotation_deg_max, pp.rel_rotation_deg_max))
        theta_s = theta_r + np.deg2rad(rel)
        zoom = float(rng_pose.uniform(pp.zoom_min, pp.zoom_max))
        search_pixel_nm = cfg.reference.pixel_nm * zoom
        scale_err = search_pixel_nm / cfg.search.pixel_nm - 1.0
    else:
        theta_s = float(np.clip(rng_pose.normal(0.0, np.deg2rad(pp.rotation_deg_sigma)),
                                -np.deg2rad(pp.rotation_deg_max), np.deg2rad(pp.rotation_deg_max)))
        scale_err = float(np.clip(rng_pose.normal(0.0, pp.scale_err_sigma),
                                  -pp.scale_err_max, pp.scale_err_max))
        search_pixel_nm = cfg.search.pixel_nm * (1.0 + scale_err)
        zoom = search_pixel_nm / cfg.reference.pixel_nm
    search_pose = {
        "center_u_nm": extent / 2.0 + float(rng_pose.uniform(-pp.search_center_jitter_nm,
                                                             pp.search_center_jitter_nm)),
        "center_v_nm": extent / 2.0 + float(rng_pose.uniform(-pp.search_center_jitter_nm,
                                                             pp.search_center_jitter_nm)),
        "theta_rad": theta_s,
        "pixel_nm": search_pixel_nm,
    }

    # The reference site is drawn uniformly in the search frame first, and the
    # layout is then generated so the requested local structure lands on it.
    # Searching a finished layout for a structure instead would bias where the
    # site falls in the frame, because large structures sit preferentially near
    # the frame centre, and that bias would flatter any decision rule that
    # favours the frame centre.
    strategies, weights = zip(*cfg.placement_mix)
    strategy = str(rng_pose.choice(strategies, p=np.array(weights) / sum(weights)))
    ru, rv = _uniform_site(rng_pose, search_pose, pp.ref_margin_px, cfg.search.out_px)
    want = {"deep_array": "deep", "near_boundary": "boundary"}.get(strategy)
    zones = LAYOUT_BUILDERS[style](mat, hgt, cfg.canvas.pixel_nm, rng_layout,
                                   style_params, target=(ru, rv), want=want)
    se_info = {}
    if modality == "sem":
        se, se_info = build_se_canvas(mat, hgt, cfg.canvas.pixel_nm, rng_canvas,
                                      cfg.canvas)

    # A degraded pair renders the search capture from a second specimen whose
    # feature widths are scaled by the drawn polygon factor, built from the same
    # seed stream so the lattice is identical, and through a capture whose dose,
    # beam spot, charging and scan stability are pushed by the severity level.
    degrade_info = None
    search_src_mat, search_src_hgt, search_src_se = mat, hgt, None
    search_capture_params = cfg.search
    if degrade:
        lad = DEGRADE_LADDER[int(degrade)]
        poly = float(1.0 + rng_degrade.uniform(-lad["poly"], lad["poly"]))
        from dataclasses import replace as _rep
        sp2 = _scale_widths(style_params, poly)
        search_src_mat = np.zeros((size, size), dtype=np.uint8)
        search_src_hgt = np.zeros((size, size), dtype=np.float32)
        LAYOUT_BUILDERS[style](search_src_mat, search_src_hgt, cfg.canvas.pixel_nm,
                               np.random.default_rng(children[0]), sp2,
                               target=(ru, rv), want=want)
        cap = cfg.search
        search_capture_params = _rep(
            cap,
            dose_e=(cap.dose_e[0] * lad["dose"], cap.dose_e[1] * lad["dose"]),
            psf_sigma_nm=(cap.psf_sigma_nm[0] * lad["psf"], cap.psf_sigma_nm[1] * lad["psf"]),
            charging_amp=(min(cap.charging_amp[0] * lad["charge"], 0.35),
                          min(cap.charging_amp[1] * lad["charge"], 0.35)),
            drift_total_px=(lad["drift_px"] * 0.5, lad["drift_px"]),
            jitter_sigma_px=(cap.jitter_sigma_px[0] * lad["jitter"],
                             cap.jitter_sigma_px[1] * lad["jitter"]),
        )
        if modality == "sem":
            search_src_se, _ = build_se_canvas(search_src_mat, search_src_hgt,
                                               cfg.canvas.pixel_nm,
                                               np.random.default_rng(children[1]),
                                               cfg.canvas)
        degrade_info = {"severity": int(degrade), "poly_scale": poly,
                        "dose_factor": lad["dose"], "psf_factor": lad["psf"],
                        "charge_factor": lad["charge"], "drift_px": lad["drift_px"]}

    # An absent pair takes its reference from a second, independently drawn
    # specimen of the same architecture. The layout statistics and the imaging
    # are identical, so the reference is plausible and periodically similar to
    # the search image while genuinely having no instance inside it. Cropping a
    # far corner of the same canvas would not do: the periodic lattice is
    # continuous, so the same cell content really does appear in the frame.
    ref_mat, ref_hgt, ref_se = mat, hgt, (se if modality == "sem" else None)
    if absent:
        ref_mat = np.zeros((size, size), dtype=np.uint8)
        ref_hgt = np.zeros((size, size), dtype=np.float32)
        ru = float(rng_layout2.uniform(extent * 0.3, extent * 0.7))
        rv = float(rng_layout2.uniform(extent * 0.3, extent * 0.7))
        zones = LAYOUT_BUILDERS[style](ref_mat, ref_hgt, cfg.canvas.pixel_nm,
                                       rng_layout2, style_params,
                                       target=(ru, rv), want=want)
        if modality == "sem":
            ref_se, _ = build_se_canvas(ref_mat, ref_hgt, cfg.canvas.pixel_nm,
                                        rng_canvas2, cfg.canvas)

    ref_pose = {"center_u_nm": ru, "center_v_nm": rv,
                "theta_rad": theta_r, "pixel_nm": cfg.reference.pixel_nm}

    if modality == "optical":
        from .imaging.optical import render_optical_capture, sample_optical_settings
        opt_ref = sample_optical_settings(rng_ref, "reference")
        opt_search = sample_optical_settings(rng_search, "search")
        ref_img, ref_meta = render_optical_capture(
            ref_mat, ref_hgt, cfg.canvas.pixel_nm, ref_pose, opt_ref, rng_ref)
        search_img, search_meta = render_optical_capture(
            search_src_mat, search_src_hgt, cfg.canvas.pixel_nm, search_pose,
            opt_search, rng_search)
        zero = np.zeros(cfg.search.out_px, dtype=np.float32)
        dx_r = dy_r = dx_s = dy_s = zero
    else:
        ref_img, dx_r, dy_r, ref_meta = render_capture(
            ref_se, ref_mat, cfg.canvas.pixel_nm, ref_pose, cfg.reference, rng_ref)
        search_img, dx_s, dy_s, search_meta = render_capture(
            search_src_se if search_src_se is not None else se,
            search_src_mat, cfg.canvas.pixel_nm, search_pose,
            search_capture_params, rng_search)

    nr = cfg.reference.out_px
    center = (nr - 1) / 2.0
    corners = []
    if absent:
        # There is no true instance, so there is no centre and no footprint.
        gt_c = gt_r = 0.0
    else:
        pu, pv = capture_to_specimen(center, center, ref_pose, dx_r, dy_r, nr)
        gt_c, gt_r = specimen_to_capture(pu, pv, search_pose, dx_s, dy_s, cfg.search.out_px)
        for (rr, cc) in [(0.0, 0.0), (0.0, nr - 1.0), (nr - 1.0, nr - 1.0), (nr - 1.0, 0.0)]:
            cu, cv2 = capture_to_specimen(rr, cc, ref_pose, dx_r, dy_r, nr)
            sc, sr = specimen_to_capture(cu, cv2, search_pose, dx_s, dy_s, cfg.search.out_px)
            corners.append([float(sc), float(sr)])

    meta = {
        "seed": int(seed),
        "style": style,
        "modality": modality,
        "placement": strategy,
        "found": 0 if absent else 1,
        "degrade": degrade_info,
        "ground_truth": {"x": float(gt_c), "y": float(gt_r)},
        "zoom": float(zoom),
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
