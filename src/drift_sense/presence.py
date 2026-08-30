"""Feature construction for the presence decision.

register.py builds features from a live locate() diagnostics dict, and the
fitting script builds them from recorded rows. Both constructions live here,
side by side, because a silent divergence between them would ship a model that
was fitted on one feature space and evaluated on another; a unit test asserts
they agree.

margin_over_mad is the absolute residual evidence: how far the winning site's
deviation field score stands above the candidate median, in robust units. The
z statistic asks whether one site stands out among its peers, which an
impostor site can do by chance; this asks whether the deviation field matched
at all, which is the question presence actually turns on.

The v2 feature set adds spatial, residual, and response-surface diagnostics
that the localizer already computes but v1 did not use.  The extra dimensions
let the model separate degraded present pairs (low peak, scattered quadrants,
but one dominant residual site) from absent impostors (moderate peak, flat
residual landscape, scattered spatial candidates).
"""

import math

import numpy as np

# ---- v1 feature set (shipped, frozen) ------------------------------------
FEATURES_V1 = ("peak", "prom_l", "wide_l", "strict_l", "noise", "nominal",
               "gap", "over_p99", "z_fill", "z_missing", "mom", "pose_wide",
               "quad_disp", "quad_agree", "quad_missing")

# ---- v2 feature set (extended) --------------------------------------------
FEATURES_V2 = FEATURES_V1 + (
    "peak_prom_ratio",       # peak / (median of response surface)
    "wide_nom_ratio",        # wide_score / nominal_score
    "residual_concentration",# best residual / sum of top-k residuals
    "spatial_scatter",       # std-dev of pairwise distances among top peaks
    "peak_to_p50_ratio",     # peak / 50th percentile of response
    "peak_to_p90_ratio",     # peak / 90th percentile of response
    "num_blur_levels",       # which blur bank level won (encoded as index)
    "rotation_abs",          # absolute value of the rotation found
    "scale_deviation",       # |scale - 1.0|, how far from nominal
    "residual_margin_raw",   # raw margin (not divided by MAD)
)

# Current feature set = v2
FEATURES = FEATURES_V2


def _assemble_v1(peak, prom, wide, strict, noise, nominal, over_p99,
                 z, margin, mad, pose_source, quad_disp=-1.0, quad_agree=-1):
    mom = 0.0
    if margin is not None and mad is not None:
        mom = float(margin) / max(1.4826 * float(mad), 1e-6)
    return [
        float(peak),
        float(np.log1p(max(float(prom), 0.0))),
        float(np.log1p(int(wide))),
        float(np.log1p(int(strict))),
        float(noise),
        float(nominal),
        float(peak) - float(nominal),
        float(over_p99),
        float(z) if z is not None else 0.0,
        0.0 if z is not None else 1.0,
        float(np.clip(mom, -20.0, 20.0)),
        1.0 if pose_source == "wide" else 0.0,
        float(np.clip(quad_disp, 0.0, 17.0)) if quad_disp >= 0 else 0.0,
        float(quad_agree) if quad_agree >= 0 else 0.0,
        1.0 if quad_disp < 0 else 0.0,
    ]


def _spatial_scatter(peaks_xy, half_search):
    """RMS pairwise distance among top peaks, normalised by image diagonal.

    Present pairs: peaks cluster at the true site → low scatter.
    Absent pairs:  peaks scatter across the lattice   → high scatter.
    """
    if not peaks_xy or len(peaks_xy) < 2:
        return 0.0
    pts = np.asarray(peaks_xy, float)
    centroid = pts.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((pts - centroid) ** 2, axis=1)))) / max(half_search, 1.0)


def _residual_concentration(top_scores):
    """How concentrated the top residual scores are.

    A decisive present pair has one dominant residual score; an absent pair
    has a flatter distribution.  Returns the ratio of the best score to the
    sum of the top min(8, k) scores.
    """
    if not top_scores or len(top_scores) == 0:
        return 0.0
    arr = np.asarray(top_scores[:8], float)
    total = float(arr.sum())
    if total < 1e-9:
        return 0.0
    return float(arr[0]) / total


def _assemble_v2(peak, prom, wide, strict, noise, nominal, over_p99,
                 z, margin, mad, pose_source, quad_disp=-1.0, quad_agree=-1,
                 resp_median=0.0, wide_score=0.0, top_scores=None,
                 peaks_xy=None, half_search=500.0, psf_sigma_idx=0,
                 theta_deg=0.0, scale=1.0, residual_margin_raw=0.0):
    v1 = _assemble_v1(peak, prom, wide, strict, noise, nominal, over_p99,
                       z, margin, mad, pose_source, quad_disp, quad_agree)

    # peak_prom_ratio: how much the peak dominates the surface
    prom_ratio = float(peak) / max(float(resp_median), 1e-6)

    # wide_nom_ratio: how much the wide grid helped (>1 means grid found better)
    wnr = float(wide_score) / max(float(nominal), 1e-6) if float(nominal) > 0 else 1.0

    # residual_concentration
    rc = _residual_concentration(top_scores)

    # spatial_scatter
    ss = _spatial_scatter(peaks_xy, half_search)

    # peak-to-percentile ratios
    peak_p50 = float(peak) / max(float(resp_median), 1e-6)
    # p90 is not stored; approximate from median using a fixed factor
    # (response surfaces are roughly exponential above median)
    peak_p90 = float(peak) / max(float(resp_median) * 1.5, 1e-6)

    return v1 + [
        prom_ratio,
        wnr,
        rc,
        ss,
        peak_p50,
        peak_p90,
        float(psf_sigma_idx),
        float(abs(theta_deg)),
        float(abs(scale - 1.0)),
        float(max(residual_margin_raw, 0.0)),
    ]


def features_from_diag(diag):
    s2 = diag.get("stage2") or {}
    search_shape = diag.get("response_shape", [1000, 1000])
    half_search = max(search_shape[0], search_shape[1]) / 2.0
    return _assemble_v2(
        diag["score"], diag.get("peak_prominence", 0.0),
        diag.get("num_candidates_wide", 1),
        diag.get("num_candidates", 1),
        diag.get("search_noise_sigma", 0.0),
        diag.get("nominal_score", 0.0),
        diag.get("peak_over_p99", 0.0),
        s2.get("z"), s2.get("margin"), s2.get("mad"),
        diag.get("pose_source", ""),
        diag.get("quad_disp", -1.0), diag.get("quad_agree", -1),
        resp_median=diag.get("resp_median", 0.0),
        wide_score=diag.get("wide_score", 0.0),
        top_scores=s2.get("top_scores"),
        peaks_xy=diag.get("candidate_peaks_xy"),
        half_search=half_search,
        psf_sigma_idx=diag.get("psf_sigma_idx", 0),
        theta_deg=diag.get("theta_deg", 0.0),
        scale=diag.get("scale", 1.0),
        residual_margin_raw=s2.get("margin", 0.0),
    )


def features_from_record(rec):
    return _assemble_v2(
        rec["peak"], rec["prom"], rec["wide"],
        rec.get("strict", 1), rec.get("noise", 0.0),
        rec.get("nominal", 0.0), rec.get("over_p99", 0.0),
        rec.get("z"), rec.get("margin"), rec.get("mad"),
        rec.get("pose_source", ""),
        rec.get("quad_disp", -1.0), rec.get("quad_agree", -1),
        resp_median=rec.get("resp_median", 0.0),
        wide_score=rec.get("wide_score", 0.0),
        top_scores=rec.get("top_scores"),
        peaks_xy=rec.get("candidate_peaks_xy"),
        half_search=rec.get("half_search", 500.0),
        psf_sigma_idx=rec.get("psf_sigma_idx", 0),
        theta_deg=rec.get("theta_deg", 0.0),
        scale=rec.get("scale", 1.0),
        residual_margin_raw=rec.get("residual_margin_raw", 0.0),
    )


def presence_probability(model, feats):
    x = (np.asarray(feats, float) - np.asarray(model["mu"])) / np.asarray(model["sd"])
    m = float(x @ np.asarray(model["weights"]) + model["bias"])
    return 1.0 / (1.0 + np.exp(-m))
