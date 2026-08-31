"""FinFET style layout: horizontal fin grid crossed by vertical gates on a
fixed contacted poly pitch, organised into standard cell rows with random cell
widths, diffusion breaks, trench contacts and vias, plus one perfectly regular
SRAM block acting as the highly periodic hard region. Parameter provenance is
documented in docs/citations.md."""

import numpy as np

from ..params import (MATERIAL_GATE, MATERIAL_SILICON, MATERIAL_STI,
                      MATERIAL_TUNGSTEN)
from .primitives import corr_noise_1d, paint_dots, paint_hstripe, paint_rect, paint_vstripe


def _sample(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def build_finfet_layout(mat, hgt, pixel_nm, rng, p, target=None, want=None):
    extent = mat.shape[0] * pixel_nm
    n = mat.shape[0]
    fp = _sample(rng, p.fin_pitch_nm)
    wf = fp * _sample(rng, p.fin_width_frac)
    cpp = _sample(rng, p.gate_pitch_nm)
    wg = cpp * _sample(rng, p.gate_width_frac)
    fins_per_row = int(rng.integers(p.fins_per_row[0], p.fins_per_row[1] + 1))
    ler_sigma = _sample(rng, p.ler_sigma_nm)
    ler_corr = _sample(rng, p.ler_corr_nm)

    mat[:] = MATERIAL_STI
    hgt[:] = 0.0

    row_pitch = (fins_per_row + p.row_gap_fins) * fp
    phase_fin = rng.uniform(0, fp)
    phase_gate = rng.uniform(0, cpp)
    phase_row = rng.uniform(0, row_pitch)

    sram_w = _sample(rng, p.sram_width_nm)
    sram_h = _sample(rng, p.sram_height_nm)
    # An SRAM macro is a micron scale block, so a canvas narrower than the block
    # sits inside one region of the die and carries no macro at all. Without this
    # the placement below draws from an inverted interval and raises, which is
    # what any field scale under about 0.4 used to do.
    sram_fits = (sram_w < 0.8 * extent) and (sram_h < 0.8 * extent)
    if target is not None and want == "deep":
        # Centre the perfectly regular block on the requested site, so that the
        # hard case is created by structure rather than by searching the canvas
        # for it, which would bias where the site lands in the search frame.
        sram_u0 = float(np.clip(target[0] - sram_w / 2.0, 0.0, extent - sram_w))
        sram_v0 = float(np.clip(target[1] - sram_h / 2.0, 0.0, extent - sram_h))
    elif target is not None and want == "boundary":
        sram_u0 = float(np.clip(target[0] + (sram_w if target[0] < extent / 2
                                             else -2.0 * sram_w),
                                0.0, extent - sram_w))
        sram_v0 = float(np.clip(target[1] + (sram_h if target[1] < extent / 2
                                             else -2.0 * sram_h),
                                0.0, extent - sram_h))
    else:
        sram_u0 = rng.uniform(0.1 * extent, max(0.1 * extent, 0.9 * extent - sram_w))
        sram_v0 = rng.uniform(0.1 * extent, max(0.1 * extent, 0.9 * extent - sram_h))
    sram = ((sram_u0, sram_v0, sram_u0 + sram_w, sram_v0 + sram_h)
            if sram_fits else None)

    row_starts = np.arange(phase_row - row_pitch, extent + row_pitch, row_pitch)
    fin_zone_h = fins_per_row * fp

    for v_row in row_starts:
        for k in range(fins_per_row):
            vc = v_row + phase_fin + k * fp
            if vc < -fp or vc > extent + fp:
                continue
            et = corr_noise_1d(rng, n, ler_sigma, ler_corr, pixel_nm)
            eb = corr_noise_1d(rng, n, ler_sigma, ler_corr, pixel_nm)
            paint_hstripe(mat, hgt, pixel_nm, vc, wf, 0.0, extent,
                          MATERIAL_SILICON, p.fin_height_nm, edge_top=et, edge_bot=eb)

    break_bands = []
    for v_row in row_starts:
        v0 = v_row
        v1 = v_row + fin_zone_h
        u = phase_gate - cpp * int(np.ceil((phase_gate) / cpp))
        cell_edges = []
        while u < extent + cpp:
            width_cells = int(rng.integers(p.cell_width_cpp[0], p.cell_width_cpp[1] + 1))
            u += width_cells * cpp
            cell_edges.append(u)
        for ue in cell_edges:
            band_w = 0.8 * cpp
            paint_rect(mat, hgt, pixel_nm, ue - band_w / 2, v0 - 0.5 * fp,
                       ue + band_w / 2, v1 + 0.5 * fp, MATERIAL_STI, 0.0)
            break_bands.append((ue, v0, v1))

    mat_before = mat.copy()
    gate_us = np.arange(phase_gate - cpp, extent + cpp, cpp)
    for uc in gate_us:
        el = corr_noise_1d(rng, n, ler_sigma, ler_corr, pixel_nm)
        er = corr_noise_1d(rng, n, ler_sigma, ler_corr, pixel_nm)
        paint_vstripe(mat, hgt, pixel_nm, uc, wg, 0.0, extent,
                      MATERIAL_GATE, p.gate_field_height_nm, edge_left=el, edge_right=er)
    over_fin = (mat == MATERIAL_GATE) & (mat_before == MATERIAL_SILICON)
    hgt[over_fin] = p.gate_fin_height_nm

    contact_w = 0.40 * cpp
    for v_row in row_starts:
        v0 = v_row + 0.2 * fp
        v1 = v_row + fin_zone_h - 0.2 * fp
        if v1 < 0 or v0 > extent:
            continue
        for uc in gate_us[:-1]:
            um = uc + cpp / 2.0
            in_break = any(abs(um - ue) < 1.2 * cpp and v_row == bv0
                           for (ue, bv0, _) in break_bands)
            if in_break or rng.random() >= p.contact_prob:
                continue
            paint_vstripe(mat, hgt, pixel_nm, um, contact_w, v0, v1,
                          MATERIAL_TUNGSTEN, p.contact_height_nm)
            if rng.random() < p.via_prob:
                cv = rng.uniform(v0 + 0.2 * (v1 - v0), v1 - 0.2 * (v1 - v0))
                paint_dots(mat, hgt, pixel_nm, np.array([um]), np.array([cv]),
                           np.array([0.35 * contact_w + 4.0]), MATERIAL_TUNGSTEN,
                           p.via_height_nm)

    if sram is not None:
        su0, sv0, su1, sv1 = sram
        paint_rect(mat, hgt, pixel_nm, su0, sv0, su1, sv1, MATERIAL_STI, 0.0)
        sram_fp = fp
        for vc in np.arange(sv0 + sram_fp, sv1, sram_fp):
            paint_hstripe(mat, hgt, pixel_nm, vc, wf, su0, su1,
                          MATERIAL_SILICON, p.fin_height_nm)
        before_sram = mat.copy()
        for uc in np.arange(su0 + cpp / 2, su1, cpp):
            paint_vstripe(mat, hgt, pixel_nm, uc, wg, sv0, sv1,
                          MATERIAL_GATE, p.gate_field_height_nm)
        sram_over_fin = (mat == MATERIAL_GATE) & (before_sram == MATERIAL_SILICON)
        hgt[sram_over_fin] = p.gate_fin_height_nm
        for i, uc in enumerate(np.arange(su0 + cpp, su1 - cpp / 2, cpp)):
            for j, vc in enumerate(np.arange(sv0 + 2 * sram_fp, sv1 - sram_fp, 2 * sram_fp)):
                paint_vstripe(mat, hgt, pixel_nm, uc, contact_w,
                              vc - 0.8 * sram_fp, vc + 0.8 * sram_fp,
                              MATERIAL_TUNGSTEN, p.contact_height_nm)
                if (i + j) % 2 == 0:
                    paint_dots(mat, hgt, pixel_nm, np.array([uc]), np.array([vc]),
                               np.array([0.35 * contact_w + 4.0]), MATERIAL_TUNGSTEN,
                               p.via_height_nm)
    layout_info = {
        "style": "finfet",
        "fin_pitch_nm": fp,
        "fin_width_nm": wf,
        "gate_pitch_nm": cpp,
        "gate_width_nm": wg,
        "fins_per_row": fins_per_row,
        "row_pitch_nm": row_pitch,
        "ler_sigma_nm": ler_sigma,
        "ler_corr_nm": ler_corr,
        "sram_rect_nm": list(sram) if sram is not None else None,
    }
    return {"periodic_zones": [z for z in (sram,) if z is not None],
            "anchor_zones": [], "info": layout_info}
