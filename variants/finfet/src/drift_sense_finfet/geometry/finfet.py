"""FinFET style layout: dense parallel vertical fin lines on a fixed fin pitch,
crossed by horizontal gate bars on a contacted poly pitch, organised into
standard cell bands with random cell lengths, diffusion breaks, trench contacts
and vias, plus one perfectly regular SRAM block acting as the highly periodic
hard region.

The published architecture reference for this variant reads: dense parallel
vertical fin lines, crossed by one or two horizontal gate bars at the
intersection region. The builder this file was derived from painted the
transpose of that, fins horizontal and gates vertical, so the whole arrangement
is turned a quarter turn here. The topology is unchanged. What moves is the
axis each family of features lives on: fins run along v and are spaced along u,
gates run along u and are spaced along v, a standard cell band is a column of
fins side by side rather than a row of them stacked, and cells are chained down
the image on the gate pitch rather than across it. Every coordinate, width,
phase and extent moves with its axis, and the SRAM block moves with them so its
interior stays coherent with the field around it.

Parameter provenance is documented in docs/citations.md."""

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
    fins_per_band = int(rng.integers(p.fins_per_row[0], p.fins_per_row[1] + 1))
    ler_sigma = _sample(rng, p.ler_sigma_nm)
    ler_corr = _sample(rng, p.ler_corr_nm)

    mat[:] = MATERIAL_STI
    hgt[:] = 0.0

    # A cell band holds its fins side by side along u and is separated from the
    # next band by a gap of whole fin pitches, so the band pitch is measured
    # along u now rather than along v.
    band_pitch = (fins_per_band + p.row_gap_fins) * fp
    phase_fin = rng.uniform(0, fp)
    phase_gate = rng.uniform(0, cpp)
    phase_band = rng.uniform(0, band_pitch)

    sram_w = _sample(rng, p.sram_width_nm)
    sram_h = _sample(rng, p.sram_height_nm)
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
        sram_u0 = rng.uniform(0.1 * extent, 0.9 * extent - sram_w)
        sram_v0 = rng.uniform(0.1 * extent, 0.9 * extent - sram_h)
    sram = (sram_u0, sram_v0, sram_u0 + sram_w, sram_v0 + sram_h)

    band_starts = np.arange(phase_band - band_pitch, extent + band_pitch, band_pitch)
    fin_zone_w = fins_per_band * fp

    for u_band in band_starts:
        for k in range(fins_per_band):
            uc = u_band + phase_fin + k * fp
            if uc < -fp or uc > extent + fp:
                continue
            # Line edge roughness now deviates the left and right flanks of a
            # vertical fin, indexed down the rows, where before it deviated the
            # top and bottom flanks of a horizontal one.
            el = corr_noise_1d(rng, n, ler_sigma, ler_corr, pixel_nm)
            er = corr_noise_1d(rng, n, ler_sigma, ler_corr, pixel_nm)
            paint_vstripe(mat, hgt, pixel_nm, uc, wf, 0.0, extent,
                          MATERIAL_SILICON, p.fin_height_nm, edge_left=el, edge_right=er)

    # Diffusion breaks cut the fins of one band at cell boundaries. The cells
    # are chained along v on the gate pitch, so a break is a horizontal band of
    # shallow trench isolation spanning that cell band's fins in u.
    break_bands = []
    for u_band in band_starts:
        u0 = u_band
        u1 = u_band + fin_zone_w
        v = phase_gate - cpp * int(np.ceil((phase_gate) / cpp))
        cell_edges = []
        while v < extent + cpp:
            length_cells = int(rng.integers(p.cell_width_cpp[0], p.cell_width_cpp[1] + 1))
            v += length_cells * cpp
            cell_edges.append(v)
        for ve in cell_edges:
            band_w = 0.8 * cpp
            paint_rect(mat, hgt, pixel_nm, u0 - 0.5 * fp, ve - band_w / 2,
                       u1 + 0.5 * fp, ve + band_w / 2, MATERIAL_STI, 0.0)
            break_bands.append((ve, u0, u1))

    mat_before = mat.copy()
    gate_vs = np.arange(phase_gate - cpp, extent + cpp, cpp)
    for vc in gate_vs:
        et = corr_noise_1d(rng, n, ler_sigma, ler_corr, pixel_nm)
        eb = corr_noise_1d(rng, n, ler_sigma, ler_corr, pixel_nm)
        paint_hstripe(mat, hgt, pixel_nm, vc, wg, 0.0, extent,
                      MATERIAL_GATE, p.gate_field_height_nm, edge_top=et, edge_bot=eb)
    over_fin = (mat == MATERIAL_GATE) & (mat_before == MATERIAL_SILICON)
    hgt[over_fin] = p.gate_fin_height_nm

    # A trench contact lands between two gates and runs across the fins of its
    # cell band, so it is a short horizontal bar at the midpoint of a gate
    # interval, spanning that band's fins in u.
    contact_w = 0.40 * cpp
    for u_band in band_starts:
        u0 = u_band + 0.2 * fp
        u1 = u_band + fin_zone_w - 0.2 * fp
        if u1 < 0 or u0 > extent:
            continue
        for vc in gate_vs[:-1]:
            vm = vc + cpp / 2.0
            in_break = any(abs(vm - ve) < 1.2 * cpp and u_band == bu0
                           for (ve, bu0, _) in break_bands)
            if in_break or rng.random() >= p.contact_prob:
                continue
            paint_hstripe(mat, hgt, pixel_nm, vm, contact_w, u0, u1,
                          MATERIAL_TUNGSTEN, p.contact_height_nm)
            if rng.random() < p.via_prob:
                cu = rng.uniform(u0 + 0.2 * (u1 - u0), u1 - 0.2 * (u1 - u0))
                paint_dots(mat, hgt, pixel_nm, np.array([cu]), np.array([vm]),
                           np.array([0.35 * contact_w + 4.0]), MATERIAL_TUNGSTEN,
                           p.via_height_nm)

    su0, sv0, su1, sv1 = sram
    paint_rect(mat, hgt, pixel_nm, su0, sv0, su1, sv1, MATERIAL_STI, 0.0)
    sram_fp = fp
    for uc in np.arange(su0 + sram_fp, su1, sram_fp):
        paint_vstripe(mat, hgt, pixel_nm, uc, wf, sv0, sv1,
                      MATERIAL_SILICON, p.fin_height_nm)
    before_sram = mat.copy()
    for vc in np.arange(sv0 + cpp / 2, sv1, cpp):
        paint_hstripe(mat, hgt, pixel_nm, vc, wg, su0, su1,
                      MATERIAL_GATE, p.gate_field_height_nm)
    sram_over_fin = (mat == MATERIAL_GATE) & (before_sram == MATERIAL_SILICON)
    hgt[sram_over_fin] = p.gate_fin_height_nm
    # The SRAM contact grid indexes gates down the block and fin pairs across
    # it, the transpose of the field arrangement, so its checkerboard of vias
    # keeps the same alternation.
    for i, vc in enumerate(np.arange(sv0 + cpp, sv1 - cpp / 2, cpp)):
        for j, uc in enumerate(np.arange(su0 + 2 * sram_fp, su1 - sram_fp, 2 * sram_fp)):
            paint_hstripe(mat, hgt, pixel_nm, vc, contact_w,
                          uc - 0.8 * sram_fp, uc + 0.8 * sram_fp,
                          MATERIAL_TUNGSTEN, p.contact_height_nm)
            if (i + j) % 2 == 0:
                paint_dots(mat, hgt, pixel_nm, np.array([uc]), np.array([vc]),
                           np.array([0.35 * contact_w + 4.0]), MATERIAL_TUNGSTEN,
                           p.via_height_nm)

    layout_info = {
        "style": "finfet",
        # Recorded per pair so the orientation is auditable from any meta.json
        # rather than inferred from the pixels.
        "fin_axis": "vertical",
        "gate_axis": "horizontal",
        "fin_pitch_nm": fp,
        "fin_width_nm": wf,
        "gate_pitch_nm": cpp,
        "gate_width_nm": wg,
        "fins_per_band": fins_per_band,
        "band_pitch_nm": band_pitch,
        "ler_sigma_nm": ler_sigma,
        "ler_corr_nm": ler_corr,
        "sram_rect_nm": list(sram),
    }
    return {"periodic_zones": [sram], "anchor_zones": [], "info": layout_info}
