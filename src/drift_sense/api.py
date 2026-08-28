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

    geo_consistency is the DOMINANT, positively-correlated signal: a true instance
    aligns at full resolution (high), a distractor/substrate does not (low). The
    prior correlation features are anti-correlated with true presence at 8..12x
    (a spurious match out-scores the degraded true instance), so they are used
    only as a weak tie-breaker. uniqueness is low for spurious many-candidate
    matches and therefore nudges absent pairs further down; peak_score/contrast are
    intentionally NOT weighted positively. The logistic-regression tune may reweight
    freely, but the single-threshold path rests on geo_consistency.
    """
    f = presence_features(diag)
    s = 0.90 * f["geo_consistency"] + 0.10 * f["uniqueness"]
    return float(np.clip(s, 0.0, 1.0))


# Trained presence classifier (Phase 2, Day 4, T1+T3). Fitted on the 200-pair
# mixed set (experiments/tune_rejection_v7) with a tiny numpy logistic regression;
# it reaches Rejection F1 = 0.9058 (>= 0.90 target). The single-threshold
# presence_score path falls short (0.877) because the prior correlation features
# are anti-correlated with true presence, so the production decision uses this
# standardized logistic model, applied at inference with no dataset available.
_PRESENCE_FEATURE_ORDER = ["peak_score", "num_candidates_wide", "uniqueness",
                           "stage2_identifiability", "margin_strength",
                           "peak_contrast", "peak_contrast_ratio", "geo_consistency",
                           "search_noise_sigma", "inverted_contrast"]
_REJECTION_W = np.array(
    [0.7411985285604407, 1.9120198895745746, 1.403379135890412,
     7.424607874267439, 3.0684358378437486, -1.3275959003435012,
     2.76989673409477, 1.3788508934271153, -0.16269377223982667,
     5.607167932903188], dtype=np.float64)
_REJECTION_BIAS = 1.9065467536870204
_REJECTION_MU = np.array(
    [0.787544724792242, 93.895, 0.8642922854031078,
     0.8287665258697782, 0.054052753448486326, 0.2199938226491213,
     1.4850390671267806, 0.4264083573548123, 4.20382450224574, 0.07],
    dtype=np.float64)
_REJECTION_SD = np.array(
    [0.21243651642724487, 608.3786445888421, 0.2931199494578426,
     0.37318386329021813, 0.1864483111646982, 0.15821311064866214,
     0.47013993892788475, 0.2278774865070163, 0.5426768152935143,
     0.2551480164434611], dtype=np.float64)


def presence_probability(diag):
    """P(present) under the trained logistic presence model."""
    f = presence_features(diag)
    x = np.array([float(f[k]) for k in _PRESENCE_FEATURE_ORDER], dtype=np.float64)
    z = (x - _REJECTION_MU) / _REJECTION_SD
    logit = _REJECTION_BIAS + float(_REJECTION_W @ z)
    return float(1.0 / (1.0 + np.exp(-logit)))


def decide_found(diag, threshold=0.5):
    """Rejection decision: 1 = present, 0 = no instance (Set C).

    Uses the trained logistic presence model (geo_consistency + correlation
    features); threshold is on the predicted probability (default 0.5).
    """
    return 1 if presence_probability(diag) >= threshold else 0


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
