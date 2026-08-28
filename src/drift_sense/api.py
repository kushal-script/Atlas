"""Drop in matcher interface.

Evaluation harnesses in this problem space call a matcher as

    match = matcher(reference_img, search_img)
    x, y, score = match["x"], match["y"], match["score"]

with both images as 1000x1000 arrays, the coordinates in search image pixels
with the origin at the top left, and a single scalar confidence used to rank
predictions. This module exposes exactly that signature so the localizer can be
substituted into such a harness without any wrapper code, and so the confidence
is a calibrated continuous value rather than the raw correlation score.

    from drift_sense.api import zncc_match, match_pair
    result = zncc_match(reference_img, search_img)
"""

import math

import numpy as np

from .localize import MatchConfig, locate, optical_config


UNIQUE_PEAK_MAX_WIDE = 2


def _regime(diag):
    """Which evidence regime produced the answer.

    The strict equal match count alone is not a safe indicator: a thoroughly
    degenerate surface can still isolate one peak inside the strict tolerance
    by noise, which previously earned the most confident label. Measured over
    150 pairs from four generators, the wide candidate pool separates correct
    from incorrect answers far better (median 1 when correct against 29 when
    wrong), so the confident label additionally requires that essentially no
    competing candidate existed. That lifts unique_peak precision from 77.7 to
    98.6 percent.
    """
    strict = int(diag["num_candidates"])
    wide = int(diag.get("num_candidates_wide", strict))
    if strict <= 1 and wide <= UNIQUE_PEAK_MAX_WIDE:
        return "unique_peak"
    if diag.get("stage2", {}).get("used"):
        return "residual_identified"
    return "tie_break_convention"


def _confidence(diag):
    """Continuous confidence in [0, 1], higher meaning more trustworthy.

    Ranking based metrics need a score that separates correct from incorrect
    predictions, so peak strength is combined with the two things that actually
    predict correctness here: how degenerate the correlation surface was, taken
    from the wide candidate pool rather than the strict one, and how decisively
    the deviation field singled out one candidate.
    """
    peak = float(np.clip(diag["score"], 0.0, 1.0))
    n = max(int(diag.get("num_candidates_wide", diag["num_candidates"])), 1)
    uniqueness = 1.0 / (1.0 + np.log1p(n - 1))
    stage2 = diag.get("stage2") or {}
    z = stage2.get("z")
    if stage2.get("used") and z is not None:
        identified = float(np.clip(z / 20.0, 0.0, 1.0))
    elif n <= UNIQUE_PEAK_MAX_WIDE:
        identified = 1.0
    else:
        identified = 0.0
    conf = 0.55 * peak + 0.30 * uniqueness + 0.15 * identified
    return float(np.clip(conf, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Presence / rejection (Phase 2, Day 3, T3)
# ---------------------------------------------------------------------------
# ~20% of Phase 2 pairs (Set C) have no true instance. The localizer still emits
# a (spurious) peak on those, so a presence decision is required. The features
# below are deliberately interpretable and cheap so a single threshold can do the
# job; if a sweep shows one threshold cannot reach F1 >= 0.90, escalate to a tiny
# numpy logistic regression on these same features (no torch).

PRESENCE_WEIGHTS = {
    "peak_score": 0.45,
    "uniqueness": 0.25,
    "stage2_identifiability": 0.20,
    "margin_strength": 0.10,
}


def presence_features(diag):
    """Interpretable signals of whether a true instance underlies the peak.

    peak_score              -- raw correlation peak (high when well matched)
    num_candidates_wide     -- competing peaks in the wide pool (more => ambiguous)
    stage2_used / z / margin-- residual disambiguation decisiveness
    inverted_contrast       -- polarity flip (informative, not penalized)
    search_noise_sigma      -- acquisition noise level
    regime                  -- which evidence regime produced the answer
    """
    peak = float(np.clip(diag["score"], 0.0, 1.0))
    geo = float(np.clip(diag.get("geo_consistency", 0.0), 0.0, 1.0))
    strict = int(diag["num_candidates"])
    wide = int(diag.get("num_candidates_wide", strict))
    stage2 = diag.get("stage2") or {}
    z = stage2.get("z")
    margin = stage2.get("margin")
    s2_used = bool(stage2.get("used"))
    uniqueness = 1.0 / (1.0 + math.log1p(max(wide - 1, 0)))
    if s2_used and z is not None:
        ident = float(np.clip(z / 20.0, 0.0, 1.0))
    elif wide <= 2:
        ident = 1.0
    else:
        ident = 0.0
    margin_strength = 0.0
    if margin is not None:
        margin_strength = float(np.clip(margin / 0.1, 0.0, 1.0))
    return {
        "peak_score": peak,
        "num_candidates_wide": wide,
        "num_candidates_strict": strict,
        "uniqueness": uniqueness,
        "stage2_used": s2_used,
        "stage2_z": (float(z) if z is not None else 0.0),
        "stage2_margin": (float(margin) if margin is not None else 0.0),
        "stage2_identifiability": ident,
        "margin_strength": margin_strength,
        "peak_contrast": float(diag.get("peak_contrast", 0.0)),
        "peak_contrast_ratio": float(diag.get("peak_contrast_ratio", 1.0)),
        "geo_consistency": geo,
        "inverted_contrast": bool(diag.get("inverted_contrast")),
        "search_noise_sigma": float(diag.get("search_noise_sigma", 0.0)),
        "regime": _regime(diag),
    }


def presence_score(diag):
    """Transparent weighted combination of presence_features in [0, 1].

    geo_consistency dominates (the only signal that is positively correlated with
    true presence at 8..12x); peak_contrast and the prior correlation features are
    secondary tie-breakers. The logistic-regression tune can reweight freely.
    """
    f = presence_features(diag)
    s = (0.55 * f["geo_consistency"]
         + 0.15 * f["peak_contrast"]
         + 0.10 * f["peak_score"]
         + 0.10 * f["uniqueness"]
         + 0.10 * f["stage2_identifiability"])
    return float(np.clip(s, 0.0, 1.0))


def decide_found(diag, threshold):
    """Rejection decision: 1 = present, 0 = no instance (Set C)."""
    return 1 if presence_score(diag) >= threshold else 0


def match_pair(reference_img, search_img, cfg=None, reranker_path=None, device=None):
    """Locate the reference pattern and return position, confidence, diagnostics."""
    ref = np.asarray(reference_img)
    search = np.asarray(search_img)
    if cfg is None:
        rgb = ref.ndim == 3 or search.ndim == 3
        cfg = optical_config() if rgb else MatchConfig()
    if reranker_path:
        cfg.reranker_path = str(reranker_path)
    if device:
        cfg.device = device
    if ref.ndim == 3:
        ref = ref[..., :3].mean(axis=2).astype(np.uint8)
    if search.ndim == 3:
        search = search[..., :3].mean(axis=2).astype(np.uint8)
    x, y, diag, _ = locate(ref, search, cfg)
    regime = _regime(diag)
    return {
        "x": float(x),
        "y": float(y),
        "score": _confidence(diag),
        "peak_score": float(diag["score"]),
        "scale": float(diag["scale"]),
        "rotation_deg": float(diag["theta_deg"]),
        "confidence_regime": regime,
        "runtime_s": float(diag["runtime_s"]),
        "diagnostics": diag,
    }


def zncc_match(reference_img, search_img):
    """Positional signature matching the reference baseline interface."""
    return match_pair(reference_img, search_img)


def shift_report(pred_x, pred_y, true_x, true_y):
    dx, dy = float(pred_x - true_x), float(pred_y - true_y)
    return {"dx": dx, "dy": dy, "distance_px": float(np.hypot(dx, dy))}
