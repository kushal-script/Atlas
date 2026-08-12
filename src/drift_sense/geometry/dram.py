"""DRAM style layout: periodic word line / bit line arrays with storage node
contacts, organised into mats separated by sense amplifier and sub word line
driver stripes, following the 6F2 buried word line architecture described in
docs/citations.md."""

import numpy as np

from ..params import (MATERIAL_GATE, MATERIAL_NITRIDE, MATERIAL_SILICON,
                      MATERIAL_STI, MATERIAL_TUNGSTEN)
from .primitives import corr_noise_1d, paint_dots, paint_hstripe, paint_rect, paint_vstripe


def _sample(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _stripe_texture(mat, hgt, px, rect, rng):
    u0, v0, u1, v1 = rect
    area_um2 = (u1 - u0) * (v1 - v0) / 1e6
    n_blocks = max(3, int(area_um2 / 0.12))
    for _ in range(n_blocks):
        w = rng.uniform(60.0, 380.0)
        h = rng.uniform(40.0, 130.0)
        if rng.random() < 0.5:
            w, h = h, w
        cu = rng.uniform(u0, u1)
        cv = rng.uniform(v0, v1)
        material = int(rng.choice([MATERIAL_GATE, MATERIAL_TUNGSTEN, MATERIAL_NITRIDE]))
        height = rng.uniform(35.0, 75.0)
        paint_rect(mat, hgt, px, max(cu - w / 2, u0), max(cv - h / 2, v0),
                   min(cu + w / 2, u1), min(cv + h / 2, v1), material, height)


def _phase_start(target, span, gap, want, rng):
    """Grid origin placing a chosen specimen coordinate at a chosen phase.

    Selecting the reference site by searching the canvas for a structure biases
    where that site lands in the search frame, because large structures sit
    preferentially near the frame centre. Instead the frame position is drawn
    uniformly first and the grid phase is solved here so the requested local
    structure lands on it, which decouples structure type from frame position.
    """
    if target is None or want is None:
        return float(rng.uniform(-span, 0.0))
    period = span + gap
    offset = span / 2.0 if want == "deep" else span + gap / 2.0
    start = target - offset
    return float(start - period * np.ceil(start / period))


def build_dram_layout(mat, hgt, pixel_nm, rng, p, target=None, want=None):
    extent = mat.shape[0] * pixel_nm
    f = _sample(rng, p.feature_nm)
    wl_pitch = 2.0 * f
    bl_pitch = 3.0 * f
    wl_width = wl_pitch * _sample(rng, p.wl_duty)
    bl_width = bl_pitch * _sample(rng, p.bl_duty)
    contact_r = f * _sample(rng, p.contact_radius_f)
    missing_prob = _sample(rng, p.contact_missing_prob)
    ler_sigma = _sample(rng, p.ler_sigma_nm)
    ler_corr = _sample(rng, p.ler_corr_nm)

    mat[:] = MATERIAL_SILICON
    hgt[:] = 0.0

    mat_w = _sample(rng, p.mat_width_nm)
    mat_h = _sample(rng, p.mat_height_nm)
    sa_h = _sample(rng, p.sa_stripe_nm)
    swd_w = _sample(rng, p.swd_stripe_nm)

    tu = target[0] if target is not None else None
    tv = target[1] if target is not None else None
    u_edges = []
    u = _phase_start(tu, mat_w, swd_w, want, rng)
    while u < extent:
        u_edges.append((u, u + mat_w))
        u += mat_w + swd_w
    v_edges = []
    v = _phase_start(tv, mat_h, sa_h, want, rng)
    while v < extent:
        v_edges.append((v, v + mat_h))
        v += mat_h + sa_h

    phase_wl = rng.uniform(0, wl_pitch)
    phase_bl = rng.uniform(0, bl_pitch)
    n = mat.shape[0]

    mat_rects, stripe_rects = [], []
    for (v0, v1) in v_edges:
        for (u0, u1) in u_edges:
            mat_rects.append((max(u0, 0.0), max(v0, 0.0), min(u1, extent), min(v1, extent)))

    for (v0, v1) in v_edges:
        wl_start = int(np.ceil((max(v0, 0.0) - phase_wl) / wl_pitch))
        wl_end = int(np.floor((min(v1, extent) - phase_wl) / wl_pitch))
        for k in range(wl_start, wl_end + 1):
            vc = phase_wl + k * wl_pitch
            et = corr_noise_1d(rng, n, ler_sigma, ler_corr, pixel_nm)
            eb = corr_noise_1d(rng, n, ler_sigma, ler_corr, pixel_nm)
            for (u0, u1) in u_edges:
                paint_hstripe(mat, hgt, pixel_nm, vc, wl_width, max(u0, 0.0), min(u1, extent),
                              MATERIAL_GATE, p.wl_height_nm, edge_top=et, edge_bot=eb)

    for (u0, u1) in u_edges:
        bl_start = int(np.ceil((max(u0, 0.0) - phase_bl) / bl_pitch))
        bl_end = int(np.floor((min(u1, extent) - phase_bl) / bl_pitch))
        for k in range(bl_start, bl_end + 1):
            uc = phase_bl + k * bl_pitch
            el = corr_noise_1d(rng, n, ler_sigma, ler_corr, pixel_nm)
            er = corr_noise_1d(rng, n, ler_sigma, ler_corr, pixel_nm)
            for (v0, v1) in v_edges:
                paint_vstripe(mat, hgt, pixel_nm, uc, bl_width, max(v0, 0.0), min(v1, extent),
                              MATERIAL_TUNGSTEN, p.bl_height_nm, edge_left=el, edge_right=er)

    for (v0, v1) in v_edges:
        for (u0, u1) in u_edges:
            uu0, vv0 = max(u0, 0.0), max(v0, 0.0)
            uu1, vv1 = min(u1, extent), min(v1, extent)
            bls = np.arange(np.ceil((uu0 - phase_bl) / bl_pitch),
                            np.floor((uu1 - phase_bl) / bl_pitch) + 1) * bl_pitch + phase_bl
            wls = np.arange(np.ceil((vv0 - phase_wl) / wl_pitch),
                            np.floor((vv1 - phase_wl) / wl_pitch) + 1) * wl_pitch + phase_wl
            if len(bls) == 0 or len(wls) == 0:
                continue
            cu, cv = np.meshgrid(bls + bl_pitch / 2.0, wls + wl_pitch / 2.0)
            cu = cu.ravel() + rng.normal(0, 0.8, cu.size)
            cv = cv.ravel() + rng.normal(0, 0.8, cv.size)
            keep = rng.random(cu.size) >= missing_prob
            inside = (cu < uu1) & (cv < vv1)
            keep &= inside
            radii = contact_r * (1.0 + rng.normal(0, p.contact_cd_sigma, cu.size))
            paint_dots(mat, hgt, pixel_nm, cu[keep], cv[keep], radii[keep],
                       MATERIAL_TUNGSTEN, p.contact_height_nm)

    prev = 0.0
    for (v0, v1) in v_edges:
        if v0 > prev:
            rect = (0.0, max(prev, 0.0), extent, min(v0, extent))
            stripe_rects.append(rect)
            _stripe_texture(mat, hgt, pixel_nm, rect, rng)
        prev = v1
    if prev < extent:
        rect = (0.0, max(prev, 0.0), extent, extent)
        stripe_rects.append(rect)
        _stripe_texture(mat, hgt, pixel_nm, rect, rng)
    prev = 0.0
    for (u0, u1) in u_edges:
        if u0 > prev:
            rect = (max(prev, 0.0), 0.0, min(u0, extent), extent)
            stripe_rects.append(rect)
            _stripe_texture(mat, hgt, pixel_nm, rect, rng)
        prev = u1
    if prev < extent:
        rect = (max(prev, 0.0), 0.0, extent, extent)
        stripe_rects.append(rect)
        _stripe_texture(mat, hgt, pixel_nm, rect, rng)

    layout_info = {
        "style": "dram",
        "feature_nm": f,
        "wl_pitch_nm": wl_pitch,
        "bl_pitch_nm": bl_pitch,
        "wl_width_nm": wl_width,
        "bl_width_nm": bl_width,
        "contact_radius_nm": contact_r,
        "ler_sigma_nm": ler_sigma,
        "ler_corr_nm": ler_corr,
        "mat_width_nm": mat_w,
        "mat_height_nm": mat_h,
        "sa_stripe_nm": sa_h,
        "swd_stripe_nm": swd_w,
        "contact_missing_prob": missing_prob,
    }
    return {"periodic_zones": mat_rects, "anchor_zones": stripe_rects, "info": layout_info}
