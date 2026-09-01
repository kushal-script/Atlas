"""Localization of the reference pattern inside the search image.

The reference is blurred to a bank of plausible search optics resolutions,
resampled onto the search pixel grid through a single affine transform per
rotation and scale hypothesis, and matched with normalized cross correlation.
A hypothesis grid is screened at quarter resolution, the top six hypotheses
are rescored at full resolution and the best is refined on a shrinking grid of
five levels. The grid covers the Phase 2 constraints directly: rotation over
plus minus 5 deg in nine steps, and scale over 0.8 to 1.2 of the nominal ten
times zoom in seventeen steps, which spans the disclosed eight to twelve range
rather than assuming a fixed magnification. The reference is blurred to six
resolutions, and a low confidence pair is retried against four wider ones.
Contrast polarity is detected from the signed correlation scores so inverted
tone conventions still match. Candidate peaks within a
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
    # Deciding contrast polarity from its own heavier blur bank was tried and
    # reverted. On the severity balanced screening suite it lifted the nominal
    # set from 0.701 to 0.800, five pairs flipping outright, which is the
    # binary signature a polarity call produces; on the held out proportional
    # suite it cost the nominal set 0.786 against 0.762 and about nine tenths
    # of a point end to end, and it eroded the optical bonus margin from 0.033
    # to 0.006. A gain that appears only on the suite a change was developed
    # against is not a gain (experiments/20260830_bank_factorial).
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
    coarse_rotations_deg: tuple = (-5.0, -3.75, -2.5, -1.25, 0.0, 1.25, 2.5, 3.75, 5.0)
    # Phase 2 draws the zoom ratio uniformly over 8 to 12 rather than jittering
    # around ten to one, so the grid has to span that range or the pose is not
    # reachable at all on roughly half the pairs. Widening it alone doubled the
    # runtime, which the prescreen dominates; screening at quarter resolution
    # instead of half recovers all of it and then some, measured at the same
    # accuracy and a lower runtime than the narrow Phase 1 grid.
    coarse_scales: tuple = (0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2)
    prescreen_downsample: int = 4
    prescreen_top_k: int = 6
    refine_rot_step_deg: float = 0.375
    refine_scale_step: float = 0.0075
    # Five levels take the final rotation step to 0.023 degrees and the final
    # scale step to 0.047 percent, both an order of magnitude inside the Phase 2
    # full credit bands of 0.25 degrees and 1 percent, at eight extra
    # correlations per pair. The step halves per level from twice the configured
    # value, so the count and the two step sizes have to move together.
    refine_levels: int = 5
    # The equal match set the centre tie break applies to. Kept tight on
    # purpose: the reference crop origin is sampled uniformly, so proximity to
    # the frame centre carries no information about correctness, and applying
    # the tie break to candidates that score measurably worse only loses
    # accuracy. Measured over four generators, 0.003 beat 0.015 on three of
    # them and improved median error on all four.
    peak_tolerance: float = 0.003
    # Wall clock ceiling. The scored run gives every pair a hard timeout, and a
    # pair that overruns scores nothing at all, so the optional stages are
    # skipped once the budget is nearly spent rather than risking the whole
    # pair. Degrading to the answer already in hand always beats returning none.
    time_budget_s: float = 9.0
    # Sign of the reported rotation. The pose grid angle is the rotation applied
    # to the reference to bring it onto the search image; the reported theta is
    # asked for as the rotation of the reference pattern as it appears in the
    # search image, counter clockwise positive, which is the opposite sense.
    # Measured against this repository's own generator the two differ by exactly
    # a sign: median error 2.62 degrees as the grid reports it against 0.25
    # degrees negated. Which sign the organiser's ground truth uses is settled
    # by the three sample pairs shipped with the addendum, not by assumption, so
    # it is one constant here rather than a convention buried in the geometry.
    theta_report_sign: float = -1.0
    # Half width of the local window, in search pixels, over which each
    # template quadrant is independently re matched for the presence check.
    quadrant_window_px: int = 12
    stage2_tolerance_frac: float = 0.6
    stage2_tolerance_cap: float = 0.06
    peak_min_separation_px: int = 4
    residual_disambiguation: bool = True
    residual_pad_px: int = 3
    residual_min_candidates: int = 2
    residual_margin: float = 0.035
    # The floor applied when a robust z is available and carries the decision;
    # the configured margin above is the bar when there is no usable z.
    residual_margin_floor: float = 0.02
    residual_z_thresh: float = 5.0
    residual_z_pool_min: int = 9
    residual_median_k: int = 31
    # Absolute residual re rank. NCC divides each window's mismatch by that
    # window's own contrast, so a high contrast impostor is forgiven exactly
    # the mismatch that convicts it; the RMS of the residual after a least
    # squares gain and offset fit forgives nothing. Ranked among the top peaks
    # it prefers the true site far more often than the correlation does on
    # degraded pairs, at severity 4 by 50 percent against 8 on the fitting
    # suite and 43 against 14 held out, and it overrides the classical choice
    # only when its relative advantage clears a margin fitted off suite.
    residual_rms_rerank: bool = True
    # Fitted on p2train alone and fixed before any held out number was read.
    # The optimum is a plateau from 0.030 to 0.060, all of it worth 26.78 of 40
    # against 25.84 with the re rank off, so the midpoint ships rather than an
    # edge of the plateau. On the fitting suite it fires on 16 of 168 pairs,
    # rescues 4 and damages none, and set A is unchanged to three decimals.
    residual_rms_margin: float = 0.045
    # Learned combiner over seven statistics of the raw pixels at each top
    # peak. As a localization override it was fitted on p2train and measured
    # a held out delta of exactly zero (experiments/20260901_rerank_combiner),
    # so the margin ships inert and the absolute residual override below stays
    # the shipped one; the diagnostics still run because the presence model
    # reads the combiner's score, margin and agreement as ambiguity evidence.
    # The model file names its own features so a stale file is refused rather
    # than misread.
    rerank_combiner: bool = True
    rerank_combiner_margin: float = 9.0
    # Full reference confirmation override. The organisers' released pipeline
    # regenerates every present pair until the raw full reference correlation's
    # global argmax lands on the label with a margin of at least 0.02, so when
    # that statistic disagrees with the pipeline's answer and clears the same
    # 0.02, the answer moves to it. Swept on the three fitting suites (plus
    # 0.81 of 40, 7 rescued 2 damaged) and judged held out at plus 1.29, plus
    # 0.86 and exactly zero on three suites with four rescues and no damage;
    # experiments/20260901_raw_confirm_and_found_f1.
    raw_override: bool = True
    raw_override_margin: float = 0.02
    # Neither the override nor the pose arbiter acts unless the raw peak
    # itself is meaningful. The released gate floors present pair raw peaks
    # near 0.34, so 0.25 never gates an on recipe fire, while a raw statistic
    # computed under a broken appearance convention peaks near noise and must
    # not move anything; the alien suite's inverted pairs measured that
    # misfire before this floor existed.
    raw_override_min_peak: float = 0.25
    # Pose arbitration by the same raw statistic. When the wide grid ran, the
    # top distinct pose candidates are each scored by the raw full reference
    # correlation, and the pose switches when another candidate's raw peak
    # beats the chosen one's by the margin. The failure this addresses is
    # scale aliasing at the range corners: at z twelve the true template is
    # smallest while a wrong scale lattice lock can win the bandpassed
    # correlation, and on the diagnosed pair the raw statistic separated the
    # true pose from the impostor at 0.837 against 0.695 with the true argmax
    # 0.3 px from truth.
    pose_arbiter: bool = True
    pose_arbiter_top_k: int = 4
    pose_arbiter_margin: float = 0.05
    # Additive full width charging streak rows, the organisers' charging
    # mechanism, corrected per row on the search capture before matching. A
    # row is corrected only when its median exceeds a running median baseline
    # by both a robust threshold and an absolute floor, and nothing happens
    # when more than a quarter of rows flag, which is structure rather than
    # streaks. Measured at plus 2.09 localization of 40 on the hardened
    # organiser recipe fitting suite and exactly zero with no false trigger
    # on the sample recipe or on 216 pairs of this repository's own
    # generator; experiments/20260901_stress_and_decoys.
    streak_suppress: bool = True
    streak_k_mad: float = 5.0
    streak_min_gray: float = 10.0
    residual_rms_top_k: int = 10
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


def _suppress_streak_rows(img, cfg):
    """Correct additive full width bright streak rows on the search capture.

    The organisers' charging model adds full width horizontal bands one to
    five rows tall and up to about seventy gray. The hazard is a horizontal
    lattice whose own bright rows a naive running median baseline would eat,
    so the baseline is phase aware: the row profile is detrended, its
    dominant vertical period measured, and each row compared against the
    median of the rows sharing its phase. A lattice row reconciles with its
    cohort and reads zero; a sporadic streak stands against thirty odd
    cohort members and cannot hide. Without a strong period the running
    median stands in, and if more than thirty percent of rows flag the frame
    is structure rather than streaks and nothing is done."""
    from scipy.ndimage import median_filter
    x = img.astype(np.float32)
    row_med = np.median(x, axis=1)
    smooth = median_filter(row_med, size=61, mode="nearest")
    detr = row_med - smooth
    f = np.abs(np.fft.rfft(detr * np.hanning(len(detr))))
    f[:2] = 0
    kbin = int(np.argmax(f))
    period = int(round(len(detr) / kbin)) if kbin else 0
    if kbin and f[kbin] > 4 * np.median(f) and 2 <= period <= 64:
        base = np.empty_like(detr)
        for ph in range(period):
            base[ph::period] = np.median(detr[ph::period])
    else:
        base = median_filter(detr, size=31, mode="nearest")
    resid = detr - base
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
    hot = resid > max(cfg.streak_min_gray, cfg.streak_k_mad * 1.4826 * mad)
    # the running median's boundary windows are asymmetric and flag edge rows
    # of any structured frame, so the outer band is never corrected
    hot[:32] = False
    hot[-32:] = False
    n = int(hot.sum())
    if n == 0 or n > int(x.shape[0] * 0.30):
        return img, 0
    x[hot] -= resid[hot, None]
    return np.clip(x, 0, 255).astype(img.dtype), n


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
        x = x - _lowpass(x, cfg.bandpass_sigma_px, cfg)
    return x, noise


def _lowpass(x, sigma, cfg):
    """The bandpass's low frequency estimate.

    A Gaussian whose sigma exceeds the frame converges to the frame's mean,
    so computing it convolutionally spends seconds to produce a constant. The
    optical preset's 120 px bandpass against a zoom of ten asks for a 1200 px
    sigma on a 1000 px image, measured at 2.5 s per call for an output whose
    whole range is one hundredth of a percent of the image's own deviation.
    Beyond the frame the closed form answer is returned instead. The secondary
    electron preset's 250 px sigma is left alone: its output still carries
    nearly a tenth of the image deviation and is real structure.
    """
    if sigma >= min(x.shape[:2]):
        return np.full_like(x, float(np.mean(x)))
    return backend.gaussian(x, sigma, cfg.device)


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
    if cfg.scale_adaptive_template:
        # And never sample past the reference edge. A template of t search
        # pixels reads t times the zoom reference pixels, so above a zoom of
        # about 11.1 the fixed 90 px template asks for more than the 1000 px
        # the reference has and the surplus is filled with a constant. A
        # constant ring carries no covariance but still counts in the
        # normalization, so it depresses the correlation more the larger the
        # scale hypothesis, and the pose search answers by preferring a
        # smaller one. Measured on held out pairs the reported scale sat
        # 1.54 percent low above that threshold against 0.13 percent below
        # it, which is the difference between the one percent full credit
        # band and the two percent band.
        t = min(t, int(ref_band.shape[0] / zoom))
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


def _residual_rms_at(search, tmpl, py, px):
    """RMS of the residual after a gain and offset least squares fit."""
    t = tmpl.shape[0]
    win = search[py:py + t, px:px + t].astype(np.float64)
    if win.shape != tmpl.shape:
        return None
    tv = tmpl.astype(np.float64)
    tz = tv - tv.mean()
    a = float((win * tz).sum() / max((tz * tz).sum(), 1e-9))
    r = win - a * tv - (win.mean() - a * tv.mean())
    return float(r.std())


_COMBINER = {"model": None, "path": None}


def _load_combiner():
    """The re rank model, loaded once per process from beside the package."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent.parent / "models" / "rerank_combiner.json"
    if _COMBINER["path"] != str(path):
        try:
            _COMBINER["model"] = json.load(open(path))
        except (OSError, ValueError):
            _COMBINER["model"] = None
        _COMBINER["path"] = str(path)
    return _COMBINER["model"]


def _lattice_lag_of(tmpl):
    prof = tmpl.mean(axis=0) - tmpl.mean()
    f = np.abs(np.fft.rfft(prof * np.hanning(len(prof))))
    f[:3] = 0
    return max(int(round(len(prof) / max(int(np.argmax(f)), 1))), 2)


def _lattice_lags_of(tmpl):
    """Dominant period of each axis of the template, in pixels."""
    lags = []
    for prof in (tmpl.mean(axis=0), tmpl.mean(axis=1)):
        p = prof - prof.mean()
        f = np.abs(np.fft.rfft(p * np.hanning(len(p))))
        f[:3] = 0
        lags.append(max(int(round(len(p) / max(int(np.argmax(f)), 1))), 2))
    return lags


def _raw_confirm(ref_raw, search_raw, z_hat, theta_report, x_hat, y_hat):
    """Full reference correlation at the estimated pose, on the raw pixels.

    The organisers' released pipeline regenerates every present pair until the
    global argmax of exactly this statistic, the full reference box filtered by
    the integer zoom and warped to the search scale, correlated against the
    unprocessed search image, lands within 3 px of the label. The blind set is
    therefore guaranteed solvable by this statistic on present pairs, and its
    peak is the very number the organisers' own absent separability
    calibration reads, computed on content our bandpass deliberately removes.
    One correlation of a roughly hundred pixel template, about 15 ms."""
    try:
        k = max(2, int(round(z_hat)))
        r = cv2.blur(ref_raw, (k, k))
        h, w = ref_raw.shape[:2]
        out = int(round(h / z_hat))
        if out < 8 or out >= min(search_raw.shape[:2]):
            return None
        M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0),
                                    float(theta_report), 1.0 / z_hat)
        M[0, 2] += (out - 1) / 2.0 - (w - 1) / 2.0
        M[1, 2] += (out - 1) / 2.0 - (h - 1) / 2.0
        tpl = cv2.warpAffine(r, M, (out, out), flags=cv2.INTER_LINEAR)
        resp = cv2.matchTemplate(search_raw, tpl, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(resp)
        rx = loc[0] + (out - 1) / 2.0
        ry = loc[1] + (out - 1) / 2.0
        rad = max(int(0.6 * out), 4)
        masked = resp.copy()
        masked[max(0, loc[1] - rad):loc[1] + rad + 1,
               max(0, loc[0] - rad):loc[0] + rad + 1] = -2.0
        second = float(masked.max())
        dist = float(np.hypot(rx - x_hat, ry - y_hat))
        return {"peak": float(peak), "margin": float(peak - second),
                "agree": bool(dist <= 3.0), "dist": dist,
                "x": float(rx), "y": float(ry)}
    except cv2.error:
        return None


def _lattice_balance_of(img):
    """Spectral balance between the two principal axes of the reference.

    A DRAM array is a two dimensional lattice and carries comparable energy on
    both axes; a FinFET field concentrates its energy on one. Reimplements the
    measurement recorded in experiments/20260831_architecture_dispatch, whose
    per pair values this reproduces to machine precision; DRAM sits near 0.55
    and FinFET near 0.24 with the genuine overlap between 0.17 and 0.38."""
    a = img.astype(np.float64)
    a = a - a.mean()
    a *= np.hanning(a.shape[0])[:, None]
    a *= np.hanning(a.shape[1])[None, :]
    F = np.abs(np.fft.fftshift(np.fft.fft2(a)))
    cy, cx = F.shape[0] // 2, F.shape[1] // 2
    F[cy - 1:cy + 2, cx - 1:cx + 2] = 0
    h, v = F[cy, :].max(), F[:, cx].max()
    return float(min(h, v) / max(max(h, v), 1e-9))


def _template_context(tmpl):
    """Everything in the statistic battery that depends on the template alone.

    Computed once per surface rather than once per candidate, which halves the
    blur work of the battery; every value is identical to what the per
    candidate computation produced, only hoisted."""
    tv = tmpl.astype(np.float64)
    tz = tv - tv.mean()
    gt = np.hypot(*np.gradient(cv2.GaussianBlur(tmpl, (0, 0), 1.0)))
    bt = cv2.GaussianBlur(tmpl, (0, 0), 8.0)
    return {"tv": tv, "tz": tz, "tzz": max((tz * tz).sum(), 1e-9),
            "tmean": tv.mean(), "gt": gt, "gtz": gt - gt.mean(), "gts": gt.std(),
            "bt": bt, "btz": bt - bt.mean(), "bts": bt.std()}


def _candidate_stats(search, tmpl, ctx, edge_n, lag, py, px, ncc):
    """Seven statistics of the raw pixels at one candidate site.

    The correlation score is one of them; the other six each read evidence the
    correlation discards, and the fitted combination beats any one of them at
    ranking the true site among the peaks. Order must match the model file."""
    t = tmpl.shape[0]
    win = search[py:py + t, px:px + t].astype(np.float64)
    if win.shape != tmpl.shape:
        return None
    a = float((win * ctx["tz"]).sum() / ctx["tzz"])
    r = win - a * ctx["tv"] - (win.mean() - a * ctx["tmean"])
    rz = r - r.mean()
    rs = rz.std() + 1e-9
    wf = win.astype(np.float32)
    gw = np.hypot(*np.gradient(cv2.GaussianBlur(wf, (0, 0), 1.0)))
    bw = cv2.GaussianBlur(wf, (0, 0), 8.0)
    return [
        float(ncc),
        -float(np.mean(np.abs(rz / rs) * edge_n)),
        -float(abs(np.mean(rz[:, :-lag] * rz[:, lag:])) / rs ** 2),
        -float(abs(np.mean(rz[:, :-2] * rz[:, 2:])) / rs ** 2),
        float(((gw - gw.mean()) * ctx["gtz"]).mean() / (gw.std() * ctx["gts"] + 1e-9)),
        float(((bw - bw.mean()) * ctx["btz"]).mean() / (bw.std() * ctx["bts"] + 1e-9)),
        -float(rs),
    ]


def _combiner_rerank(search, tmpl, resp, cfg, extra_site=None):
    """Top peaks ranked by the fitted combination of the seven statistics.

    Returns (candidates, probabilities) sorted best first, or (None, None)
    when the model is absent or fewer than two peaks are usable. extra_site,
    when given, is appended so the classical choice is always scored even if
    the peak suppression removed it."""
    model = _load_combiner()
    if model is None:
        return None, None, None
    t = tmpl.shape[0]
    r = resp.copy()
    rad = max(t // 3, 8)
    sites = []
    for _ in range(int(cfg.residual_rms_top_k)):
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx <= -1:
            break
        px, py = int(loc[0]), int(loc[1])
        r[max(0, py - rad):py + rad + 1, max(0, px - rad):px + rad + 1] = -2
        sites.append((py, px, float(mx)))
    if extra_site is not None and all(
            np.hypot(extra_site[0] - py, extra_site[1] - px) > 2 for py, px, _ in sites):
        ey, ex = extra_site
        if 0 <= ey < resp.shape[0] and 0 <= ex < resp.shape[1]:
            sites.append((ey, ex, float(resp[ey, ex])))
    # a degenerate template yields a meaningless huge lag whose shifted slices
    # are empty, the mean of which is nan; clamping keeps every slice populated
    lag = min(_lattice_lag_of(tmpl), max(tmpl.shape[1] // 2, 2))
    edge = np.hypot(*np.gradient(tmpl))
    edge_n = edge / (edge.std() + 1e-9)
    ctx = _template_context(tmpl)
    mu = np.asarray(model["mu"], float)
    sd = np.asarray(model["sd"], float)
    wv = np.asarray(model["weights"], float)
    cands, probs, stats = [], [], []
    for py, px, ncc in sites:
        f = _candidate_stats(search, tmpl, ctx, edge_n, lag, py, px, ncc)
        if f is None or not np.all(np.isfinite(f)):
            continue
        z = (np.asarray(f) - mu) / sd
        m = float(z @ wv + model["bias"])
        cands.append((py, px))
        probs.append(1.0 / (1.0 + np.exp(-m)))
        stats.append([float(v) for v in f])
    if len(cands) < 2:
        return None, None, None
    order = np.argsort(probs)[::-1]
    return ([cands[i] for i in order], [probs[i] for i in order],
            [stats[i] for i in order])


def _rms_rerank(search, tmpl, resp, cfg):
    """Top peaks of the correlation surface ranked by absolute residual.

    Returns (candidates, rms values) with candidates as (py, px) rows, best
    residual first, or (None, None) when fewer than two peaks are usable."""
    t = tmpl.shape[0]
    r = resp.copy()
    rad = max(t // 3, 8)
    cands, rmss = [], []
    for _ in range(int(cfg.residual_rms_top_k)):
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx <= -1:
            break
        px, py = int(loc[0]), int(loc[1])
        r[max(0, py - rad):py + rad + 1, max(0, px - rad):px + rad + 1] = -2
        v = _residual_rms_at(search, tmpl, py, px)
        if v is None:
            continue
        cands.append((py, px))
        rmss.append(v)
    if len(cands) < 2:
        return None, None
    order = np.argsort(rmss)
    return [cands[i] for i in order], [rmss[i] for i in order]


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


def locate(ref_img, search_img, cfg=None, return_artifacts=False, t_start=None):
    cfg = cfg or MatchConfig()
    # The budget belongs to the pair, not to this call. A caller that matches
    # the same pair more than once, as the width rescue does, passes the time
    # it started the pair so every pass draws down one shared allowance;
    # without it each pass gets a full budget and the pair's real ceiling is
    # the budget times the number of passes, which is how pairs came to exceed
    # the scored timeout while every individual call stayed inside it.
    t0 = time.perf_counter() if t_start is None else t_start
    lattice_balance = _lattice_balance_of(ref_img)
    ref_raw, search_raw = ref_img, search_img
    ref_img, ref_impulse = _despeckle(ref_img, cfg)
    search_img, search_impulse = _despeckle(search_img, cfg)
    ref_img = _tone_normalize(ref_img, cfg)
    search_img = _tone_normalize(search_img, cfg)
    streak_rows = 0
    if cfg.streak_suppress:
        search_img, streak_rows = _suppress_streak_rows(search_img, cfg)
    ref = ref_img.astype(np.float32)
    ref_low = _lowpass(ref, cfg.bandpass_sigma_px * cfg.zoom, cfg)
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
        # the raw statistic correlates the unprocessed captures, so the
        # polarity decision has to reach it too; without this the raw peak on
        # an inverted search is meaningless noise, and an override firing on
        # meaningless evidence moved correct answers on the alien suite's
        # inverted pairs, the exact misfire that suite was built to catch
        search_raw = (255 - search_raw).astype(search_raw.dtype)
        corr = backend.make_correlator(search, cfg.device)
        # The prescreen correlator has to be negated with the full resolution
        # one. Rebuilding only the latter left the wide pose grid screened
        # against the opposite polarity image, where normalized correlation
        # returns the negated coefficient, so sorting the hypotheses by
        # descending peak put the worst of them at the top of the list the
        # top k is drawn from. Measured on this repository's own secondary
        # electron pairs the inversion fires on about two pairs in five, so
        # this was not a dormant path. Area resampling is linear, so negating
        # the downsampled image is exactly the downsampled negation.
        small = -small
        corr_small = backend.make_correlator(small, cfg.device)
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
    pose_arbiter_diag = None
    budget_left = lambda frac: (time.perf_counter() - t0) < cfg.time_budget_s * frac
    # Whether the budget, rather than the evidence, decided which stages ran.
    # A caller comparing two passes over the same pair needs this: a pass that
    # was cut short is not weaker evidence, it is a different measurement, and
    # letting it outscore a complete pass replaces a good answer with a worse
    # one purely because the clock ran down.
    budget_gated = nominal_score < cfg.nominal_accept_score and not budget_left(0.45)
    if nominal_score < cfg.nominal_accept_score and budget_left(0.45):
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

        if cfg.pose_arbiter and budget_left(0.75):
            best_key = wide_key if used_wide else nominal_key
            best_th, best_sc, best_sig = best_key
            # Candidate poses from three sources, deduplicated: the evaluated
            # poses, the half resolution prescreen's own ranking (the true
            # pose can lose the prescreen entirely, which is the failure this
            # exists to catch), and a scale ladder at the chosen rotation,
            # because the diagnosed failure mode is a wrong scale lattice
            # lock. The raw statistic is cheap enough to score them all; only
            # a decisive winner is ever evaluated at full resolution.
            seen, cands = set(), []

            def _add(th_k, sc_k, sig_k):
                nm = (round(float(th_k), 2), round(float(sc_k), 3))
                if nm not in seen:
                    seen.add(nm)
                    cands.append((float(th_k), float(sc_k), sig_k))

            _add(*best_key)
            for k in sorted(tried, key=tried.get, reverse=True)[:cfg.pose_arbiter_top_k]:
                _add(*k)
            for _, sig_p, th_p, sc_p in prescreen[:6]:
                _add(th_p, sc_p, sig_p)
            for sc_l in (0.8, 0.9, 1.1, 1.2):
                _add(best_th, sc_l, best_sig)
            scored = []
            for th_k, sc_k, sig_k in cands:
                rc = _raw_confirm(ref_raw, search_raw, sc_k * cfg.zoom,
                                  cfg.theta_report_sign * th_k, 0.0, 0.0)
                scored.append(((th_k, sc_k, sig_k),
                               rc["peak"] if rc else -1.0,
                               rc["margin"] if rc else 0.0))
            cur = scored[0]
            top = max(scored, key=lambda s: s[1])
            pose_arbiter_diag = {
                "cands": [[k[0], k[1], float(pk), float(mg)]
                          for k, pk, mg in scored],
                "cur_raw": float(cur[1]), "top_raw": float(top[1]),
                "fired": False}
            if (top[0][:2] != (best_th, best_sc)
                    and top[1] >= cfg.raw_override_min_peak
                    and top[1] >= cur[1] + cfg.pose_arbiter_margin):
                th_w, sc_w, sig_w = top[0]
                evaluate(sig_w, th_w, sc_w)
                switched = refine((round(th_w, 4), round(sc_w, 5), sig_w))
                wide_key, wide_score, used_wide = switched, tried[switched], True
                pose_arbiter_diag["fired"] = True

    best = wide_key if used_wide else nominal_key
    theta_best, scale_best, sigma_best = best
    tmpl_best = _make_template(ref_bank[sigma_best], theta_best, scale_best, cfg)
    resp = corr.full(tmpl_best)
    score_best = float(resp.max())
    # Pose stability: where does the argmax go when the rotation is perturbed
    # by one refine step. A true match is pinned by aperiodic content and its
    # argmax stays put; a lattice alias lock is free to jump to another cell.
    # Diagnostic only, measured for the presence decision; it does not touch
    # the answer.
    _py0, _px0 = np.unravel_index(int(np.argmax(resp)), resp.shape)
    _stab = []
    for _dth in (-cfg.refine_rot_step_deg, cfg.refine_rot_step_deg):
        _tp = _make_template(ref_bank[sigma_best], theta_best + _dth, scale_best, cfg)
        _rp = corr.full(_tp)
        if _rp.shape == resp.shape:
            _pyy, _pxx = np.unravel_index(int(np.argmax(_rp)), _rp.shape)
            _stab.append(float(np.hypot(_pxx - _px0, _pyy - _py0)))
    pose_stability_px = float(np.median(_stab)) if _stab else -1.0

    # How far the best peak stands clear of the rest of the surface. The raw
    # correlation value falls with noise, so an absolute threshold on it cannot
    # separate a present reference in a degraded capture from an absent one in a
    # clean capture. Prominence is scale free: it asks whether this peak is
    # unlike the surface it sits on, which is the question the found flag is
    # actually asking.
    _rf = resp.ravel()
    _med = float(np.median(_rf))
    _mad = float(np.median(np.abs(_rf - _med)))
    _p99 = float(np.quantile(_rf, 0.99))
    peak_prominence = (score_best - _med) / max(1.4826 * _mad, 1e-6)
    peak_over_p99 = score_best - _p99
    # Peak to correlation energy, the peak's share of the whole surface's
    # energy, was added here and measured to contribute exactly nothing: an
    # ablation over the same records put the cross validated reject F1 at
    # 0.7040 with it and 0.7040 without. It is redundant against prominence,
    # which already asks whether the peak is unlike the surface it sits on,
    # and its class medians barely separate at 27.7 present against 25.2
    # absent. Recorded here rather than left for someone to re derive
    # (experiments/20260830_pce_ablation).

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
    if (cfg.residual_disambiguation and len(wide) >= cfg.residual_min_candidates
            and not budget_left(0.75)):
        budget_gated = True
    if (cfg.residual_disambiguation and len(wide) >= cfg.residual_min_candidates
            and budget_left(0.75)):
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
                          mad=float(mad), residual_median=med_v,
                          best_residual_score=float(vals[wi]),
                          top_scores=[float(v) for v in vals[order[:8]]])
            # The classical rule is the decision. An enabled re-ranker may
            # override which candidate is chosen, but when it abstains the
            # classical rule still decides, because letting an abstention fall
            # through to the centre tie break discards a sub pixel answer the
            # deviation field had already identified.
            # Two regimes, two margins, and they are deliberately different.
            # With a pool large enough for a robust z, the z carries the
            # decision and the margin is only a floor against a numerically
            # trivial separation, so it is the looser of the two. With too few
            # candidates for a z to mean anything, the margin is the whole
            # decision and has to clear the higher configured bar alone.
            if len(vals) >= cfg.residual_z_pool_min:
                decisive = z >= cfg.residual_z_thresh and margin >= cfg.residual_margin_floor
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

    # Absolute residual re rank over the top peaks. Runs after the classical
    # decision so the deviation field rule keeps deciding when this abstains,
    # and before the rotation polish and the quadrant check so both follow
    # whichever site is finally chosen. Both candidate answers and the
    # relative margin are recorded whether or not the override fires, so the
    # margin can be swept offline from one harvest.
    rms_diag = None
    rr_diag = None
    if cfg.rerank_combiner and resp is not None:
        cy, cx = int(round(y - half)), int(round(x - half))
        r_cands, r_probs, r_stats = _combiner_rerank(search, tmpl_best, resp, cfg,
                                                      extra_site=(cy, cx))
        if r_cands is not None:
            # the classical choice's own probability, for the margin and for
            # the presence model's disagreement feature
            p_cls = 0.0
            for (py2, px2), pr in zip(r_cands, r_probs):
                if np.hypot(py2 - cy, px2 - cx) <= 2.0:
                    p_cls = pr
                    break
            b_py, b_px = r_cands[0]
            agree = bool(np.hypot(b_py - cy, b_px - cx) <= 2.0)
            margin2 = float(r_probs[0] - (r_probs[1] if len(r_probs) > 1 else 0.0))
            rr_diag = {"score": float(r_probs[0]), "margin": margin2,
                       "agree": agree, "fired": False,
                       "classical_score": float(p_cls),
                       "top_rc": [int(b_py), int(b_px)],
                       "classical_rc": [int(cy), int(cx)]}
            if getattr(cfg, "rerank_record_stats", False):
                rr_diag["candidates"] = [
                    {"rc": [int(a), int(b)], "prob": float(pr), "stats": st}
                    for (a, b), pr, st in zip(r_cands, r_probs, r_stats)]
            if (not agree
                    and float(r_probs[0]) - p_cls >= cfg.rerank_combiner_margin):
                dx2 = dy2 = 0.0
                if 0 < b_px < resp.shape[1] - 1:
                    dx2 = _parabolic(resp[b_py, b_px - 1], resp[b_py, b_px],
                                     resp[b_py, b_px + 1])
                if 0 < b_py < resp.shape[0] - 1:
                    dy2 = _parabolic(resp[b_py - 1, b_px], resp[b_py, b_px],
                                     resp[b_py + 1, b_px])
                x = b_px + dx2 + half
                y = b_py + dy2 + half
                py, px = b_py, b_px
                rr_diag["fired"] = True
    if (cfg.residual_rms_rerank and resp is not None
            and not (rr_diag and rr_diag["fired"])):
        r_cands, r_vals = _rms_rerank(search, tmpl_best, resp, cfg)
        if r_cands is not None:
            cy, cx = int(round(y - half)), int(round(x - half))
            cur = _residual_rms_at(search, tmpl_best, cy, cx)
            b_py, b_px = r_cands[0]
            far = bool(np.hypot(b_py - cy, b_px - cx) > 2.0)
            rms_diag = {"rel_margin": 0.0, "fired": False,
                        "classical_xy": [float(x), float(y)], "rms_xy": None}
            if cur is not None and far:
                rel = (cur - r_vals[0]) / max(cur, 1e-9)
                dx2 = dy2 = 0.0
                if 0 < b_px < resp.shape[1] - 1:
                    dx2 = _parabolic(resp[b_py, b_px - 1], resp[b_py, b_px],
                                     resp[b_py, b_px + 1])
                if 0 < b_py < resp.shape[0] - 1:
                    dy2 = _parabolic(resp[b_py - 1, b_px], resp[b_py, b_px],
                                     resp[b_py + 1, b_px])
                rms_diag.update(rel_margin=float(rel),
                                rms_xy=[float(b_px + dx2 + half),
                                        float(b_py + dy2 + half)])
                if rel >= cfg.residual_rms_margin:
                    x = b_px + dx2 + half
                    y = b_py + dy2 + half
                    py, px = b_py, b_px
                    rms_diag["fired"] = True

    # Final rotation polish. The grid quantizes rotation and the parabolic
    # refinement inherits that lattice; a euclidean ECC alignment between the
    # matched template and the window under it recovers the residual rotation
    # continuously. The motion model is deliberately euclidean rather than
    # affine: on a periodic lattice an affine fit can slide scale by pitch
    # fractions and made the scale estimate worse, while rotation has no such
    # aliasing channel at this window size. Applied only when the fit converges
    # to a small residual, otherwise the grid answer stands.
    theta_polished = theta_best
    py0, px0 = int(round(y - half)), int(round(x - half))
    t_pol = tmpl_best.shape[0]
    if (0 <= py0 and 0 <= px0 and py0 + t_pol <= search.shape[0]
            and px0 + t_pol <= search.shape[1]):
        win_pol = np.ascontiguousarray(search[py0:py0 + t_pol,
                                              px0:px0 + t_pol], dtype=np.float32)
        try:
            warp = np.eye(2, 3, dtype=np.float32)
            _, warp = cv2.findTransformECC(
                np.ascontiguousarray(tmpl_best, dtype=np.float32), win_pol,
                warp, cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5))
            r_res = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
            if abs(r_res) <= 1.5:
                theta_polished = theta_best + r_res
        except cv2.error:
            pass

    # Quadrant consistency, an absolute presence check. Each quadrant of the
    # matched template is correlated independently in a small window around the
    # winning site. A true match pins every quadrant to the same offset,
    # because the aperiodic content agrees quadrant by quadrant; an impostor
    # matches only the shared lattice, so any lattice aligned offset serves any
    # quadrant equally and their best offsets scatter across cells. The
    # dispersion of the four offsets is scale free and survives degradation.
    quad_disp, quad_agree = -1.0, -1
    th, tw = tmpl_best.shape
    if min(th, tw) >= 32:
        qy, qx = int(py), int(px)
        m = cfg.quadrant_window_px
        offs = []
        for qi in range(2):
            for qj in range(2):
                q = tmpl_best[qi * th // 2:(qi + 1) * th // 2,
                              qj * tw // 2:(qj + 1) * tw // 2]
                oy, ox = qi * th // 2, qj * tw // 2
                sub = search[max(qy + oy - m, 0):qy + oy + q.shape[0] + m,
                             max(qx + ox - m, 0):qx + ox + q.shape[1] + m]
                if sub.shape[0] <= q.shape[0] or sub.shape[1] <= q.shape[1]:
                    continue
                r = cv2.matchTemplate(np.ascontiguousarray(sub),
                                      np.ascontiguousarray(q),
                                      cv2.TM_CCOEFF_NORMED)
                ry, rx = np.unravel_index(int(np.argmax(r)), r.shape)
                base_y = max(qy + oy - m, 0)
                base_x = max(qx + ox - m, 0)
                offs.append(((base_y + ry) - (qy + oy), (base_x + rx) - (qx + ox)))
        if len(offs) == 4:
            d = [float(np.hypot(dy_, dx_)) for dy_, dx_ in offs]
            quad_disp = float(np.median(d))
            quad_agree = int(sum(1 for v in d if v <= 2.0))

    # Three ambiguity diagnostics for the presence model. The second peak is
    # searched outside one full lattice period in each axis because inside it
    # the runner up is the same site, and at the naive radius the runner up of
    # a periodic layout is a lattice replica of the true site whether or not
    # the site is right, which is exactly the case the ratio test was not
    # built for. Near one the surface is comb only; the lower the ratio the
    # more the chosen site carries that the rest of the lattice does not.
    lag_h, lag_v = _lattice_lags_of(tmpl_best)
    _rad = max(lag_h, lag_v, 8)
    _r1 = float(resp[py, px])
    _masked = resp.copy()
    _masked[max(0, py - _rad):py + _rad + 1, max(0, px - _rad):px + _rad + 1] = -2.0
    _r2 = float(_masked.max())
    period_ratio = float(max(_r2, 0.0) / _r1) if _r1 > 1e-6 else 1.0
    peak_curv = 0.0
    if 0 < py < resp.shape[0] - 1 and 0 < px < resp.shape[1] - 1:
        peak_curv = float(((2 * resp[py, px] - resp[py, px - 1] - resp[py, px + 1])
                           + (2 * resp[py, px] - resp[py - 1, px] - resp[py + 1, px]))
                          / max(abs(_r1), 1e-6))

    raw_confirm = _raw_confirm(ref_raw, search_raw,
                               float(scale_best) * cfg.zoom,
                               cfg.theta_report_sign * float(theta_polished),
                               float(x), float(y))
    if raw_confirm is not None:
        raw_confirm["fired"] = bool(cfg.raw_override
                                    and not raw_confirm["agree"]
                                    and raw_confirm["peak"] >= cfg.raw_override_min_peak
                                    and raw_confirm["margin"] >= cfg.raw_override_margin)
        if raw_confirm["fired"]:
            x, y = raw_confirm["x"], raw_confirm["y"]

    diag = {
        "score": score_best,
        "raw_confirm": raw_confirm,
        "pose_arbiter": pose_arbiter_diag,
        "streak_rows": int(streak_rows),
        "lattice_balance": lattice_balance,
        "period_ratio": period_ratio,
        "peak_curv": peak_curv,
        "theta_deg": float(theta_polished),
        "theta_grid_deg": float(theta_best),
        "scale": float(scale_best),
        "psf_sigma_nm": float(sigma_best),
        "pose_source": "wide_grid" if used_wide else "nominal",
        "budget_gated": bool(budget_gated),
        "residual_rms": rms_diag,
        "rerank": rr_diag,
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
        "resp_median": _med,
        "resp_p99": _p99,
        "peak_prominence": float(peak_prominence),
        "pose_stability_px": float(pose_stability_px),
        "quad_disp": float(quad_disp),
        "quad_agree": int(quad_agree),
        "peak_over_p99": float(peak_over_p99),
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
    """Returns (image, is_rgb); RGB inputs are converted to luminance.

    The flag reports whether the capture carries colour, not whether the file
    happens to have three planes. A grayscale capture exported as an RGB or
    RGBA png is an ordinary thing for a tool chain to produce, and deciding
    the modality from the array's rank would route every such pair through the
    optical preset, which is a different blur bank and a different bandpass
    and measurably halves the localization credit on grayscale pairs. The
    planes are compared instead, with a tolerance rather than an equality test
    because a lossy round trip can perturb them by a count or two.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        planes = img[:, :, :3].astype(np.int16)
        spread = max(int(np.abs(planes[:, :, 0] - planes[:, :, 1]).max()),
                     int(np.abs(planes[:, :, 1] - planes[:, :, 2]).max()))
        grey = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        return grey, spread > 2
    return img, False


def load_colour(path):
    """Returns (planes, is_rgb) where planes is a list of matching planes.

    In an optical capture the contrast between materials comes from thin film
    interference, so two materials can share a luminance and differ only in
    colour. Measured on generated optical pairs the standard deviation of the
    channel differences is comparable to the standard deviation of the
    luminance itself, which is half the available signal thrown away by
    collapsing to grey before matching. Correlating each plane and summing the
    surfaces keeps it, at the cost of one correlation per plane.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim != 3:
        return [img], False
    bgr = img[:, :, :3].astype(np.float32)
    lum = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    planes = [lum, bgr[:, :, 0] - bgr[:, :, 1], bgr[:, :, 1] - bgr[:, :, 2]]
    out = []
    for pl in planes:
        lo, hi = float(pl.min()), float(pl.max())
        out.append(np.ascontiguousarray(
            ((pl - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)))
    return out, True
