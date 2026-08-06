"""Localization of the reference pattern inside the search image.

The reference is blurred to the search optics resolution, resampled onto the
search pixel grid through a single affine transform per rotation and scale
hypothesis, and matched with normalized cross correlation. A coarse grid over
rotation and scale is refined around the best hypothesis, candidate peaks
within a tolerance of the best score are collected, and the peak closest to
the search image center is returned with sub pixel parabolic refinement, as
required by the problem statement tie break rule.
"""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, maximum_filter


@dataclass
class MatchConfig:
    zoom: float = 10.0
    template_px: int = 90
    psf_sigma_nm: float = 6.0
    bandpass_sigma_px: float = 25.0
    denoise_sigma_px: float = 0.6
    coarse_rotations_deg: tuple = (-3.0, -1.5, 0.0, 1.5, 3.0)
    coarse_scales: tuple = (0.97, 1.0, 1.03)
    refine_rot_step_deg: float = 0.375
    refine_scale_step: float = 0.0075
    refine_levels: int = 2
    peak_tolerance: float = 0.015
    peak_min_separation_px: int = 4


def _preprocess(img, cfg, denoise):
    x = img.astype(np.float32)
    if denoise and cfg.denoise_sigma_px > 0:
        x = gaussian_filter(x, cfg.denoise_sigma_px)
    if cfg.bandpass_sigma_px > 0:
        x = x - gaussian_filter(x, cfg.bandpass_sigma_px)
    return x


def _make_template(ref_blurred, theta_deg, scale, cfg):
    t = cfg.template_px
    zoom = cfg.zoom * scale
    ang = np.deg2rad(theta_deg)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    matrix = rot * zoom
    half = (t - 1) / 2.0
    center_ref = (ref_blurred.shape[0] - 1) / 2.0
    offset = center_ref - matrix @ np.array([half, half])
    tmpl = affine_transform(ref_blurred, matrix, offset=offset,
                            output_shape=(t, t), order=1, mode="constant",
                            cval=float(ref_blurred.mean()))
    return tmpl


def _match(search, tmpl):
    return cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)


def _best_score(resp):
    return float(resp.max())


def locate(ref_img, search_img, cfg=None):
    cfg = cfg or MatchConfig()
    t0 = time.perf_counter()
    ref_blurred = gaussian_filter(ref_img.astype(np.float32), cfg.psf_sigma_nm)
    ref_blurred = ref_blurred - gaussian_filter(ref_blurred, cfg.bandpass_sigma_px * cfg.zoom)
    search = _preprocess(search_img, cfg, denoise=True)

    tried = {}

    def evaluate(theta, scale):
        key = (round(theta, 4), round(scale, 5))
        if key in tried:
            return tried[key]
        tmpl = _make_template(ref_blurred, theta, scale, cfg)
        score = _best_score(_match(search, tmpl))
        tried[key] = score
        return score

    for th in cfg.coarse_rotations_deg:
        for sc in cfg.coarse_scales:
            evaluate(th, sc)

    best = max(tried, key=tried.get)
    rot_step = cfg.refine_rot_step_deg * 2
    scale_step = cfg.refine_scale_step * 2
    for _ in range(cfg.refine_levels):
        th0, sc0 = best
        for th in (th0 - rot_step, th0, th0 + rot_step):
            for sc in (sc0 - scale_step, sc0, sc0 + scale_step):
                evaluate(th, sc)
        best = max(tried, key=tried.get)
        rot_step /= 2
        scale_step /= 2

    theta_best, scale_best = best
    resp = _match(search, _make_template(ref_blurred, theta_best, scale_best, cfg))
    score_best = float(resp.max())

    local_max = maximum_filter(resp, size=cfg.peak_min_separation_px) == resp
    candidates = np.argwhere(local_max & (resp >= score_best - cfg.peak_tolerance))
    t = cfg.template_px
    half = (t - 1) / 2.0
    center = (search.shape[1] - 1) / 2.0, (search.shape[0] - 1) / 2.0
    dists = [(np.hypot(c[1] + half - center[0], c[0] + half - center[1]), c)
             for c in candidates]
    dists.sort(key=lambda item: item[0])
    peak = dists[0][1]

    py, px = int(peak[0]), int(peak[1])
    dx = dy = 0.0
    if 0 < px < resp.shape[1] - 1:
        denom = resp[py, px - 1] - 2 * resp[py, px] + resp[py, px + 1]
        if abs(denom) > 1e-9:
            dx = float(np.clip(0.5 * (resp[py, px - 1] - resp[py, px + 1]) / denom, -0.5, 0.5))
    if 0 < py < resp.shape[0] - 1:
        denom = resp[py - 1, px] - 2 * resp[py, px] + resp[py + 1, px]
        if abs(denom) > 1e-9:
            dy = float(np.clip(0.5 * (resp[py - 1, px] - resp[py + 1, px]) / denom, -0.5, 0.5))

    x = px + dx + half
    y = py + dy + half
    diag = {
        "score": score_best,
        "theta_deg": float(theta_best),
        "scale": float(scale_best),
        "num_candidates": int(len(candidates)),
        "candidate_peaks_xy": [[float(c[1] + half), float(c[0] + half)]
                               for _, c in dists[:12]],
        "peak_value": float(resp[py, px]),
        "runtime_s": time.perf_counter() - t0,
        "response_shape": list(resp.shape),
    }
    return float(x), float(y), diag, resp


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img
