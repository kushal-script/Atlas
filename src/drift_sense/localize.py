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
from scipy.ndimage import affine_transform, maximum_filter

from . import backend


@dataclass
class MatchConfig:
    zoom: float = 10.0
    # Compute backend for the blur bank and the correlation, the two
    # operations that dominate runtime. Defaults to cpu, which needs no
    # framework and is the submitted inference path; an accelerator is only
    # ever used when asked for explicitly. scripts/verify_backends.py
    # asserts that every backend returns the same answer.
    device: str = "cpu"
    template_px: int = 90
    # At an under unity scale hypothesis a fixed pixel size template covers
    # less of the reference (810 nm at 9 to 1 against 900 at 10 to 1), which
    # measurably cost accuracy at the low end of the stated magnification
    # range (60 percent at 9.0 and 9.5 to 1 against 90 at and above 10 to 1,
    # experiments/*_pose_robustness). When enabled, the template grows as
    # 1 over scale below unity so its physical reference coverage stays
    # constant across the magnification range.
    scale_adaptive_template: bool = True
    # Extending these banks to 28 and 36 nm to represent the specification
    # proxy's heaviest blur was tried and reverted on attribution evidence
    # (experiments/*_v3_banks_reverted): the additions caused the whole stress
    # domain regression through the enlarged hypothesis grid competing for a
    # fixed prescreen budget, while the specification domain's gain turned out
    # to come from the scale adaptive template, not the bank. Representing
    # blur beyond 25 nm therefore remains an open trade off; scaling the
    # prescreen budget with the bank size is the named future experiment.
    psf_sigma_bank_nm: tuple = (2.0, 4.0, 6.5, 9.0, 14.0, 20.0)
    wide_sigma_bank_nm: tuple = (4.0, 9.0, 16.0, 25.0)
    # Anti aliasing the template models one particular search pipeline, the one
    # that decimates by area averaging. Measured over four independent
    # generators it cost more on the point sampling ones than it gained on the
    # area averaging one, so it is off by default and kept as an option. See
    # experiments/*_tolerance_and_template_ablation.
    antialias: bool = False
    nominal_preference: float = 0.02
    # Above 1.0 the early exit never fires and the pose grid is always searched,
    # which measured better than exiting early on data carrying real rotation
    # and magnification error. Lower it to trade accuracy for runtime.
    nominal_accept_score: float = 9.9
    bandpass_sigma_px: float = 25.0
    denoise_sigma_px: float = 0.6
    # At the heaviest acquisition tiers (dose 25 to 60 electrons per pixel)
    # a fixed 0.6 px denoise leaves the correlation dominated by noise, so
    # the denoise strength scales with the measured noise level, floored at
    # the fixed value so clean images are untouched.
    adaptive_denoise: bool = True
    denoise_noise_gain: float = 0.04
    denoise_sigma_max: float = 2.0
    impulse_median_ksize: int = 3
    impulse_detect_frac: float = 0.005
    impulse_detect_delta: int = 55
    tone_norm: str = "none"
    coarse_rotations_deg: tuple = (-3.0, -1.5, 0.0, 1.5, 3.0)
    coarse_scales: tuple = (0.90, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.10)
    prescreen_downsample: int = 2
    prescreen_top_k: int = 6
    refine_rot_step_deg: float = 0.375
    refine_scale_step: float = 0.0075
    refine_levels: int = 2
    # The equal match set the centre tie break applies to. Kept tight on
    # purpose: the reference crop origin is sampled uniformly, so proximity to
    # the frame centre carries no information about correctness, and applying
    # the tie break to candidates that score measurably worse only loses
    # accuracy. Measured over four generators, 0.003 beat 0.015 on three of
    # them and improved median error on all four.
    peak_tolerance: float = 0.003
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
    reranker_path: str = ""
    reranker_prob: float = 0.5
    reranker_pool: int = 48


def _despeckle(img, cfg):
    """Remove impulse noise when present.

    Salt and pepper pixels are unbounded outliers, so they dominate the sums
    inside normalized cross correlation and are not attenuated by Gaussian
    denoising. A 3x3 median removes them exactly, but it also softens fine
    structure, so it is applied only when the impulse fraction is detectable.
    """
    if cfg.impulse_median_ksize <= 1 or img.dtype != np.uint8:
        return img, 0.0
    med = cv2.medianBlur(img, cfg.impulse_median_ksize)
    frac = float(np.mean(np.abs(img.astype(np.int16) - med.astype(np.int16))
                         > cfg.impulse_detect_delta))
    return (med if frac >= cfg.impulse_detect_frac else img), frac


def _tone_normalize(img, cfg):
    """Remove monotonic intensity differences between the two captures.

    Correlation is invariant to affine intensity changes but not to the gamma
    and contrast curve differences the organiser specification allows, so
    mapping both captures to a uniform histogram makes any monotonic tone
    difference irrelevant.
    """
    if cfg.tone_norm == "none" or img.dtype != np.uint8:
        return img
    if cfg.tone_norm == "equalize":
        return cv2.equalizeHist(img)
    if cfg.tone_norm == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    raise ValueError(f"unknown tone_norm {cfg.tone_norm}")


_NOISE_KERNEL = np.array([[1.0, -2.0, 1.0],
                          [-2.0, 4.0, -2.0],
                          [1.0, -2.0, 1.0]], dtype=np.float32)


def _noise_sigma(img):
    """Immerkaer style noise estimate, median based so edges do not inflate it."""
    lap = cv2.filter2D(img.astype(np.float32), -1, _NOISE_KERNEL)
    return float(np.sqrt(np.pi / 2.0) * np.median(np.abs(lap)) / 6.0)


def _preprocess(img, cfg, denoise):
    x = img.astype(np.float32)
    noise = 0.0
    if denoise and cfg.denoise_sigma_px > 0:
        sigma = cfg.denoise_sigma_px
        if cfg.adaptive_denoise:
            noise = _noise_sigma(img)
            sigma = float(np.clip(cfg.denoise_noise_gain * noise,
                                  cfg.denoise_sigma_px, cfg.denoise_sigma_max))
        x = backend.gaussian(x, sigma, cfg.device)
    if cfg.bandpass_sigma_px > 0:
        x = x - backend.gaussian(x, cfg.bandpass_sigma_px, cfg.device)
    return x, noise


def _effective_sigma(spot_nm, cfg):
    """Template blur matching the search image's formation.

    A search pixel integrates a zoom by zoom block of specimen area, which is a
    box filter of variance zoom squared over twelve. Point sampling the
    reference without that pre filter leaves aliasing the search image does not
    have, so the box is folded into the template blur in quadrature with the
    beam spot.
    """
    if not cfg.antialias:
        return spot_nm
    box = cfg.zoom / np.sqrt(12.0)
    return float(np.hypot(spot_nm, box))


def _make_template(ref_band, theta_deg, scale, cfg):
    t = cfg.template_px
    if cfg.scale_adaptive_template and scale < 1.0:
        t = int(round(cfg.template_px / scale))
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
        return None, med, None

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
    return (num / np.sqrt(var * e_rt)).astype(np.float32), med, rt0


def locate(ref_img, search_img, cfg=None, return_artifacts=False):
    cfg = cfg or MatchConfig()
    t0 = time.perf_counter()
    ref_img, ref_impulse = _despeckle(ref_img, cfg)
    search_img, search_impulse = _despeckle(search_img, cfg)
    ref_img = _tone_normalize(ref_img, cfg)
    search_img = _tone_normalize(search_img, cfg)
    ref = ref_img.astype(np.float32)
    ref_low = backend.gaussian(ref, cfg.bandpass_sigma_px * cfg.zoom, cfg.device)
    all_sigmas = sorted(set(cfg.psf_sigma_bank_nm) | set(cfg.wide_sigma_bank_nm))
    ref_bank = {s: backend.gaussian(ref, _effective_sigma(s, cfg), cfg.device) - ref_low
                for s in all_sigmas}
    search, search_noise = _preprocess(search_img, cfg, denoise=True)
    corr = backend.make_correlator(search, cfg.device)

    tried = {}

    def evaluate(sig, theta, scale):
        key = (round(theta, 4), round(scale, 5), sig)
        if key in tried:
            return tried[key]
        tmpl = _make_template(ref_bank[sig], theta, scale, cfg)
        tried[key] = float(corr.peaks([tmpl])[0])
        return tried[key]

    def refine(seed_key):
        """Local descent on rotation and scale, keeping the seed's blur level."""
        best_local = seed_key
        rot_step = cfg.refine_rot_step_deg * 2
        scale_step = cfg.refine_scale_step * 2
        for _ in range(cfg.refine_levels):
            th0, sc0, sig0 = best_local
            neighbourhood = [best_local]
            for th, sc in ((th0 - rot_step, sc0), (th0 + rot_step, sc0),
                           (th0, sc0 - scale_step), (th0, sc0 + scale_step)):
                evaluate(sig0, th, sc)
                neighbourhood.append((round(th, 4), round(sc, 5), sig0))
            best_local = max(neighbourhood, key=tried.get)
            rot_step /= 2
            scale_step /= 2
        return best_local

    # Contrast polarity, decided once from the nominal pose at half resolution.
    ds = cfg.prescreen_downsample
    small = cv2.resize(search, (search.shape[1] // ds, search.shape[0] // ds),
                       interpolation=cv2.INTER_AREA)
    ts = cfg.template_px // ds
    pol_mx = pol_mn = 0.0
    pol_tmpls = [cv2.resize(_make_template(ref_bank[sig], 0.0, 1.0, cfg), (ts, ts),
                            interpolation=cv2.INTER_AREA)
                 for sig in cfg.wide_sigma_bank_nm]
    corr_small = backend.make_correlator(small, cfg.device)
    for mn, mx in corr_small.peaks(pol_tmpls, want_min=True):
        pol_mx, pol_mn = max(pol_mx, mx), min(pol_mn, mn)
    inverted = -pol_mn > pol_mx
    if inverted:
        search = -search
        corr = backend.make_correlator(search, cfg.device)
        tried.clear()

    # Stage one, the nominal pose. The reference pipeline is an exact 10 to 1
    # decimation with no rotation, so the nominal pose is by far the most likely
    # and is evaluated at full resolution over the whole blur bank.
    for sig in cfg.psf_sigma_bank_nm:
        evaluate(sig, 0.0, 1.0)
    nominal_key = max(tried, key=tried.get)
    nominal_key = refine(nominal_key)
    nominal_score = tried[nominal_key]

    # Stage two, the wide pose grid, screened at half resolution. It is skipped
    # when the nominal pose already correlates strongly, which is the common
    # case for an exact decimation and keeps the runtime budget down. An off
    # nominal hypothesis must beat the nominal one by a margin before it is
    # accepted, because a wide grid gives many chances for a wrong pose to win
    # on noise alone.
    wide_score = nominal_score
    wide_key = nominal_key
    used_wide = False
    if nominal_score < cfg.nominal_accept_score:
        poses, grid_tmpls = [], []
        for sig in cfg.wide_sigma_bank_nm:
            for th in cfg.coarse_rotations_deg:
                for sc in cfg.coarse_scales:
                    if th == 0.0 and sc == 1.0:
                        continue
                    tmpl_h = _make_template(ref_bank[sig], th, sc, cfg)
                    th_px = max(tmpl_h.shape[0] // ds, 8)
                    grid_tmpls.append(cv2.resize(tmpl_h, (th_px, th_px),
                                                 interpolation=cv2.INTER_AREA))
                    poses.append((sig, th, sc))
        peaks = corr_small.peaks(grid_tmpls)
        prescreen = [(peak, sig, th, sc)
                     for peak, (sig, th, sc) in zip(peaks, poses)]
        prescreen.sort(key=lambda p: -p[0])
        wide_keys = []
        for _, sig, th, sc in prescreen[:cfg.prescreen_top_k]:
            evaluate(sig, th, sc)
            wide_keys.append((round(th, 4), round(sc, 5), sig))
        cand = max(wide_keys, key=tried.get, default=nominal_key)
        cand = refine(cand)
        if tried[cand] > nominal_score + cfg.nominal_preference:
            wide_key, wide_score, used_wide = cand, tried[cand], True

    best = wide_key if used_wide else nominal_key
    theta_best, scale_best, sigma_best = best
    tmpl_best = _make_template(ref_bank[sigma_best], theta_best, scale_best, cfg)
    resp = corr.full(tmpl_best)
    score_best = float(resp.max())

    tol_wide = min(cfg.stage2_tolerance_cap,
                   max(cfg.peak_tolerance,
                       cfg.stage2_tolerance_frac * (1.0 - score_best)))
    local_max = maximum_filter(resp, size=cfg.peak_min_separation_px) == resp
    wide = np.argwhere(local_max & (resp >= score_best - tol_wide))
    strict = np.argwhere(local_max & (resp >= score_best - cfg.peak_tolerance))
    t = tmpl_best.shape[0]
    half = (t - 1) / 2.0
    center = (search.shape[1] - 1) / 2.0, (search.shape[0] - 1) / 2.0
    dists = [(np.hypot(c[1] + half - center[0], c[0] + half - center[1]), c)
             for c in strict]
    dists.sort(key=lambda item: item[0])

    stage2 = {"used": False, "evaluated": 0, "margin": None,
              "tol_wide": float(tol_wide)}
    x = y = None
    r2 = med = rt0 = None
    if cfg.residual_disambiguation and len(wide) >= cfg.residual_min_candidates:
        ordered = sorted(([int(c[0]), int(c[1])] for c in wide),
                         key=lambda c: -resp[c[0], c[1]])
        r2, med, rt0 = _residual_score_map(search, tmpl_best, ordered, cfg)
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
            order = np.argsort(-vals)
            stage2.update(evaluated=int(len(vals)), margin=margin, z=float(z),
                          best_residual_score=float(vals[wi]),
                          top_scores=[float(v) for v in vals[order[:8]]])
            # The classical rule is the decision. An enabled re-ranker may
            # override which candidate is chosen, but when it abstains the
            # classical rule still decides, because letting an abstention fall
            # through to the centre tie break discards a sub pixel answer the
            # deviation field had already identified.
            if len(vals) >= cfg.residual_z_pool_min:
                decisive = z >= cfg.residual_z_thresh and margin >= 0.02
            else:
                decisive = margin >= cfg.residual_margin
            if cfg.reranker_path:
                from .reranker import rerank
                choice, prob = rerank(search, tmpl_best, med, rt0, resp, r2,
                                      wide, cfg)
                stage2.update(reranker_prob=float(prob),
                              reranker_abstained=choice is None)
                if choice is not None:
                    wi = int(choice)
                    decisive = True
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
        "pose_source": "wide_grid" if used_wide else "nominal",
        "nominal_score": float(nominal_score),
        "wide_score": float(wide_score),
        "inverted_contrast": bool(inverted),
        "impulse_fraction": [float(ref_impulse), float(search_impulse)],
        "search_noise_sigma": float(search_noise),
        "template_px_used": int(t),
        "num_candidates": int(len(strict)),
        "num_candidates_wide": int(len(wide)),
        "candidate_peaks_xy": [[float(c[1] + half), float(c[0] + half)]
                               for _, c in dists[:12]],
        "peak_value": float(resp[py, px]),
        "stage2": stage2,
        "runtime_s": time.perf_counter() - t0,
        "response_shape": list(resp.shape),
    }
    if return_artifacts:
        diag["artifacts"] = {"search": search, "template": tmpl_best,
                             "resp": resp, "r2": r2, "med": med, "rt0": rt0,
                             "wide": wide}
    return float(x), float(y), diag, resp


def optical_config():
    return MatchConfig(psf_sigma_bank_nm=(2.0, 8.0),
                      bandpass_sigma_px=120.0,
                      denoise_sigma_px=1.5)


def load_gray(path):
    """Returns (image, is_rgb); RGB inputs are converted to luminance."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY), True
    return img, False
