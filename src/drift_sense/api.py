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

import numpy as np

from .localize import MatchConfig, locate, optical_config


def _confidence(diag):
    """Continuous confidence in [0, 1], higher meaning more trustworthy.

    Ranking based metrics need a score that separates correct from incorrect
    predictions, so peak strength is combined with the two things that actually
    predict correctness here: whether the correlation surface was degenerate,
    and how decisively the deviation field singled out one candidate.
    """
    peak = float(np.clip(diag["score"], 0.0, 1.0))
    n = max(int(diag["num_candidates"]), 1)
    uniqueness = 1.0 / (1.0 + np.log1p(n - 1))
    stage2 = diag.get("stage2") or {}
    z = stage2.get("z")
    if stage2.get("used") and z is not None:
        identified = float(np.clip(z / 20.0, 0.0, 1.0))
    elif n == 1:
        identified = 1.0
    else:
        identified = 0.0
    conf = 0.55 * peak + 0.30 * uniqueness + 0.15 * identified
    return float(np.clip(conf, 0.0, 1.0))


def match_pair(reference_img, search_img, cfg=None, reranker_path=None):
    """Locate the reference pattern and return position, confidence, diagnostics."""
    ref = np.asarray(reference_img)
    search = np.asarray(search_img)
    if cfg is None:
        rgb = ref.ndim == 3 or search.ndim == 3
        cfg = optical_config() if rgb else MatchConfig()
    if reranker_path:
        cfg.reranker_path = str(reranker_path)
    if ref.ndim == 3:
        ref = ref[..., :3].mean(axis=2).astype(np.uint8)
    if search.ndim == 3:
        search = search[..., :3].mean(axis=2).astype(np.uint8)
    x, y, diag, _ = locate(ref, search, cfg)
    regime = ("unique_peak" if diag["num_candidates"] <= 1
              else "residual_identified" if diag["stage2"]["used"]
              else "tie_break_convention")
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
