"""Localization of the reference pattern inside the search image.

The reference is blurred to a bank of plausible search optics resolutions,
resampled onto the search pixel grid through a single affine transform per
rotation and scale hypothesis, and matched with normalized cross correlation.
A wide hypothesis grid (rotation to plus minus 6 deg, scale to plus minus
4 percent, three blur levels) is screened at half resolution, the top
hypotheses are rescored at full resolution and the best is refined on a
shrinking grid. Contrast polarity is detected from the signed correlation
scores so inverted tone conventions still match. Candidate peaks within a
tolerance of the best score are collected; a residual disambiguation stage
separates lattice degenerate candidates by their deviation fields, and when
indecisive the problem statement tie break returns the candidate closest to
the search image center. Sub pixel output comes from a parabolic fit on the
correlation peak.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, maximum_filter


@dataclass
class MatchConfig:
    zoom: float = 10.0
    template_px: int = 90
    psf_sigma_bank_nm: tuple = (4.0, 9.0, 16.0, 25.0)
    bandpass_sigma_px: float = 25.0
    denoise_sigma_px: float = 0.6
    coarse_rotations_deg: tuple = (-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0)
    coarse_scales: tuple = (0.96, 0.98, 1.0, 1.02, 1.04)
    prescreen_downsample: int = 2
    prescreen_top_k: int = 6
    refine_rot_step_deg: float = 0.375
    refine_scale_step: float = 0.0075
    refine_levels: int = 2
    peak_tolerance: float = 0.015
    stage2_tolerance_frac: float = 0.6
    stage2_tolerance_cap: float = 0.06
    peak_min_separation_px: int = 4
    residual_disambiguation: bool = True
    residual_pad_px: int = 3
    residual_min_candidates: int = 3
    residual_margin: float = 0.035
    residual_z_thresh: float = 5.0
    residual_z_pool_min: int = 9
    residual_median_k: int = 31


def _preprocess(img, cfg, denoise):
    x = img.astype(np.float32)
    if denoise and cfg.denoise_sigma_px > 0:
        x = gaussian_filter(x, cfg.denoise_sigma_px)
    if cfg.bandpass_sigma_px > 0:
        x = x - gaussian_filter(x, cfg.bandpass_sigma_px)
    return x


def _make_template(ref_band, theta_deg, scale, cfg):
    t = cfg.template_px
    zoom = cfg.zoom * scale
    ang = np.deg2rad(theta_deg)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    matrix = rot * zoom
    half = (t - 1) / 2.0
    center_ref = (ref_band.shape[0] - 1) / 2.0
    offset = center_ref - matrix @ np.array([half, half])
    return affine_transform(ref_band, matrix, offset=offset,
                            output_shape=(t, t), order=1, mode="constant",
                            cval=float(ref_band.mean()))


def _parabolic(v_m, v_0, v_p):
    denom = v_m - 2 * v_0 + v_p
    if abs(denom) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (v_m - v_p) / denom, -0.5, 0.5))


def _residual_score_map(search, tmpl, aligned_positions, cfg):
    """Cell to cell reference subtraction, computed densely.

    The pixelwise median over the best aligned candidate windows estimates the
    shared periodic content. The normalized correlation between the template's
    deviation field and every window's deviation field then decomposes into
    two raw correlations plus integral image sums, giving an exact dense score
    map with no candidate selection bias."""
    t = tmpl.shape[0]
    n_pix = float(t * t)
    k = min(cfg.residual_median_k, len(aligned_positions))
    if k % 2 == 0:
        k -= 1
    stack = np.stack([search[py:py + t, px:px + t]
                      for py, px in aligned_positions[:k]])
    med = np.median(stack, axis=0).astype(np.float32)
    gy_m, gx_m = np.gradient(med)
    basis = np.stack([np.ones_like(med).ravel(), med.ravel(),
                      gx_m.ravel(), gy_m.ravel()], axis=1)
    coef, _, _, _ = np.linalg.lstsq(basis, tmpl.ravel(), rcond=None)
    rt0 = np.ascontiguousarray((tmpl.ravel() - basis @ coef).reshape(med.shape))
    e_rt = float(np.sum(rt0 * rt0))
    if e_rt < 1e-4 * float(np.sum(med * med)):
        return None, med

    corr_rt = cv2.matchTemplate(search, rt0, cv2.TM_CCORR).astype(np.float64)
    corr_med = cv2.matchTemplate(search, np.ascontiguousarray(med),
                                 cv2.TM_CCORR).astype(np.float64)
    c_med_rt = float(np.sum(med.astype(np.float64) * rt0))
    num = corr_rt - c_med_rt

    s64 = search.astype(np.float64)
    i1 = cv2.integral(s64)
    i2 = cv2.integral(s64 * s64)
    s1 = i1[t:, t:] - i1[:-t, t:] - i1[t:, :-t] + i1[:-t, :-t]
    s2 = i2[t:, t:] - i2[:-t, t:] - i2[t:, :-t] + i2[:-t, :-t]
    sum_med = float(np.sum(med, dtype=np.float64))
    sum_med2 = float(np.sum(med.astype(np.float64) ** 2))
    var = s2 - 2.0 * corr_med + sum_med2 - (s1 - sum_med) ** 2 / n_pix
    np.clip(var, 1e-6, None, out=var)
    return (num / np.sqrt(var * e_rt)).astype(np.float32), med


def locate(ref_img, search_img, cfg=None):
    cfg = cfg or MatchConfig()
    t0 = time.perf_counter()
    ref = ref_img.astype(np.float32)
    ref_low = gaussian_filter(ref, cfg.bandpass_sigma_px * cfg.zoom)
    ref_bank = {s: gaussian_filter(ref, s) - ref_low for s in cfg.psf_sigma_bank_nm}
    search = _preprocess(search_img, cfg, denoise=True)

    ds = cfg.prescreen_downsample
    small = cv2.resize(search, (search.shape[1] // ds, search.shape[0] // ds),
                       interpolation=cv2.INTER_AREA)
    ts = cfg.template_px // ds
    prescreen = []
    for sig in cfg.psf_sigma_bank_nm:
        for th in cfg.coarse_rotations_deg:
            for sc in cfg.coarse_scales:
                tmpl = _make_template(ref_bank[sig], th, sc, cfg)
                tmpl_s = cv2.resize(tmpl, (ts, ts), interpolation=cv2.INTER_AREA)
                resp = cv2.matchTemplate(small, tmpl_s, cv2.TM_CCOEFF_NORMED)
                mn, mx, _, _ = cv2.minMaxLoc(resp)
                prescreen.append((max(mx, -mn), mx >= -mn, sig, th, sc))
    prescreen.sort(key=lambda p: -p[0])
    inverted = not prescreen[0][1]
    if inverted:
        search = -search

    tried = {}

    def evaluate(sig, theta, scale):
        key = (round(theta, 4), round(scale, 5), sig)
        if key in tried:
            return tried[key]
        tmpl = _make_template(ref_bank[sig], theta, scale, cfg)
        score = float(cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED).max())
        tried[key] = score
        return score

    seen = set()
    for _, _, sig, th, sc in prescreen[:cfg.prescreen_top_k * 2]:
        if (sig, th, sc) in seen or len(seen) >= cfg.prescreen_top_k:
            continue
        seen.add((sig, th, sc))
        evaluate(sig, th, sc)

    best = max(tried, key=tried.get)
    rot_step = cfg.refine_rot_step_deg * 2
    scale_step = cfg.refine_scale_step * 2
    for _ in range(cfg.refine_levels):
        th0, sc0, sig0 = best
        for th in (th0 - rot_step, th0, th0 + rot_step):
            for sc in (sc0 - scale_step, sc0, sc0 + scale_step):
                evaluate(sig0, th, sc)
        best = max(tried, key=tried.get)
        rot_step /= 2
        scale_step /= 2

    theta_best, scale_best, sigma_best = best
    tmpl_best = _make_template(ref_bank[sigma_best], theta_best, scale_best, cfg)
    resp = cv2.matchTemplate(search, tmpl_best, cv2.TM_CCOEFF_NORMED)
    score_best = float(resp.max())

    tol_wide = min(cfg.stage2_tolerance_cap,
                   max(cfg.peak_tolerance,
                       cfg.stage2_tolerance_frac * (1.0 - score_best)))
    local_max = maximum_filter(resp, size=cfg.peak_min_separation_px) == resp
    wide = np.argwhere(local_max & (resp >= score_best - tol_wide))
    strict = np.argwhere(local_max & (resp >= score_best - cfg.peak_tolerance))
    t = cfg.template_px
    half = (t - 1) / 2.0
    center = (search.shape[1] - 1) / 2.0, (search.shape[0] - 1) / 2.0
    dists = [(np.hypot(c[1] + half - center[0], c[0] + half - center[1]), c)
             for c in strict]
    dists.sort(key=lambda item: item[0])

    stage2 = {"used": False, "evaluated": 0, "margin": None,
              "tol_wide": float(tol_wide)}
    x = y = None
    if cfg.residual_disambiguation and len(wide) >= cfg.residual_min_candidates:
        ordered = sorted(([int(c[0]), int(c[1])] for c in wide),
                         key=lambda c: -resp[c[0], c[1]])
        r2, _ = _residual_score_map(search, tmpl_best, ordered, cfg)
        decisive = False
        if r2 is not None:
            slack = 2 * cfg.residual_pad_px + 1
            vals = r2[wide[:, 0], wide[:, 1]]
            wi = int(np.argmax(vals))
            d2 = (wide[:, 0] - wide[wi, 0]) ** 2 + (wide[:, 1] - wide[wi, 1]) ** 2
            others = vals[d2 > slack * slack]
            margin = float(vals[wi] - others.max()) if others.size else 1.0
            med_v = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med_v)))
            z = (float(vals[wi]) - med_v) / max(1.4826 * mad, 1e-6)
            if len(vals) >= cfg.residual_z_pool_min:
                decisive = z >= cfg.residual_z_thresh and margin >= 0.02
            else:
                decisive = margin >= cfg.residual_margin
            order = np.argsort(-vals)
            stage2.update(evaluated=int(len(vals)), margin=margin, z=float(z),
                          best_residual_score=float(vals[wi]),
                          top_scores=[float(v) for v in vals[order[:8]]])
        if decisive:
            py, px = int(wide[wi, 0]), int(wide[wi, 1])
            dx = dy = 0.0
            if 0 < px < resp.shape[1] - 1:
                dx = _parabolic(resp[py, px - 1], resp[py, px], resp[py, px + 1])
            if 0 < py < resp.shape[0] - 1:
                dy = _parabolic(resp[py - 1, px], resp[py, px], resp[py + 1, px])
            x = px + dx + half
            y = py + dy + half
            stage2["used"] = True

    if x is None:
        peak = dists[0][1]
        py, px = int(peak[0]), int(peak[1])
        dx = dy = 0.0
        if 0 < px < resp.shape[1] - 1:
            dx = _parabolic(resp[py, px - 1], resp[py, px], resp[py, px + 1])
        if 0 < py < resp.shape[0] - 1:
            dy = _parabolic(resp[py - 1, px], resp[py, px], resp[py + 1, px])
        x = px + dx + half
        y = py + dy + half

    diag = {
        "score": score_best,
        "theta_deg": float(theta_best),
        "scale": float(scale_best),
        "psf_sigma_nm": float(sigma_best),
        "inverted_contrast": bool(inverted),
        "num_candidates": int(len(strict)),
        "num_candidates_wide": int(len(wide)),
        "candidate_peaks_xy": [[float(c[1] + half), float(c[0] + half)]
                               for _, c in dists[:12]],
        "peak_value": float(resp[py, px]),
        "stage2": stage2,
        "runtime_s": time.perf_counter() - t0,
        "response_shape": list(resp.shape),
    }
    return float(x), float(y), diag, resp


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return img
