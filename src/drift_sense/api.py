"""Library interface to the localizer and the Phase 2 registration decision.

Two entry points. `match_pair` and its positional alias `zncc_match` are the
Phase 1 drop in matcher: position, a calibrated confidence and the evidence
regime, with both images as arrays. `register_pair` is the full Phase 2
decision the shipped entry point makes for every pair, identical in every
branch to what `register.py` writes: the degenerate input guard, the width
rescue passes, the learned presence decision, the optical disclosure rule,
the quadrant damped score and the finite guards, returned as a
`Registration` whose `as_row` renders exactly the predictions CSV row.

    from drift_sense.api import register_pair, load_presence_model
    model = load_presence_model()
    result = register_pair(reference_img, search_img, model=model)
    result.found, result.x, result.y, result.theta_deg, result.scale, result.score
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion

from .localize import MatchConfig, locate, optical_config
from .presence import features_for_model, presence_probability

UNIQUE_PEAK_MAX_WIDE = 2

RESCUE_PEAK_BELOW = 0.62
RESCUE_MARGIN = 0.02
RESCUE_START_BEFORE = 0.5
FALLBACK_FOUND_THRESHOLD = 0.55
FALLBACK_SCORE_WIDTH = 0.08
DEGENERATE_STD = 1.0

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "presence_model.json"


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


def match_pair(reference_img, search_img, cfg=None, reranker_path=None, device=None):
    """Locate the reference pattern and return position, confidence, diagnostics.

    This is the Phase 1 always present matcher: it carries no presence
    decision, its score is a heuristic confidence rather than the presence
    probability, and its rotation_deg and scale are the localizer's internal
    grid values, the rotation applied to the template, whose sign is opposite
    to the reported convention, and the scale relative to the nominal zoom.
    A caller wanting the contract units and the found flag uses
    `register_pair`."""
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


def load_presence_model(path=None):
    """The fitted presence decision from its json file.

    Raises on a missing or inconsistent file; the batch entry point catches
    that and falls back to the peak threshold rule with a warning, a library
    caller gets the exception."""
    path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    model = json.load(open(path))
    if len(model.get("weights", [])) != len(model.get("features", [])):
        raise ValueError("weights and features disagree in length")
    return model


def _finite(v, fallback=0.0):
    """Never let nan or inf into a reported quantity.

    A degenerate pair can put a nan through normalized correlation, and a nan
    rendered into a predictions file is not a number a scorer can read."""
    v = float(v)
    return v if math.isfinite(v) else float(fallback)


def _fallback_score(peak, prominence, wide_candidates):
    """Monotonic confidence in the decision when no presence model is loaded."""
    p_present = 1.0 / (1.0 + np.exp(-(peak - FALLBACK_FOUND_THRESHOLD) / FALLBACK_SCORE_WIDTH))
    decision_conf = max(p_present, 1.0 - p_present)
    uniqueness = 1.0 / (1.0 + np.log1p(max(int(wide_candidates), 1) - 1))
    strength = min(max(prominence, 0.0) / 20.0, 1.0)
    return float(decision_conf * (0.6 + 0.25 * uniqueness + 0.15 * strength))


@dataclass
class Registration:
    """One pair's registration decision.

    x, y is the match centre in search image pixels, origin at the centre of
    the top left pixel, x rightward and y downward. theta_deg is the rotation
    of the reference pattern as it appears in the search image, counter
    clockwise positive. scale is the recovered magnification ratio, nominally
    8 to 12. found is the presence decision; a rejected pair carries zeros in
    its pose when rendered as a row. score is confidence in the decision made,
    higher meaning the reported pose or the rejection is more likely right.
    reason names the branch: matched, rejected, optical_disclosed for a colour
    pair where presence is disclosed, degenerate_input for a blank frame."""
    x: float
    y: float
    theta_deg: float
    scale: float
    found: bool
    score: float
    regime: str
    reason: str
    runtime_s: float
    diagnostics: dict = field(default_factory=dict, repr=False)

    def as_row(self, pair_id):
        if self.found:
            return {"pair_id": pair_id, "x": f"{self.x:.3f}", "y": f"{self.y:.3f}",
                    "theta": f"{self.theta_deg:.3f}", "scale": f"{self.scale:.4f}",
                    "found": 1, "score": f"{self.score:.5f}"}
        return {"pair_id": pair_id, "x": 0, "y": 0, "theta": 0, "scale": 0,
                "found": 0, "score": f"{self.score:.5f}"}


def register_pair(reference, search, *, reference_rgb=False, search_rgb=False,
                  model=None, config=None, optical=None, t_start=None):
    """The full Phase 2 decision for one pair, as the batch entry point makes it.

    reference and search are grayscale uint8 arrays as `load_gray` returns
    them, with the rgb flags saying whether either capture carried colour.
    model is the loaded presence model, or None for the peak threshold
    fallback. config and optical are the SEM and optical match configurations,
    defaulting to the shipped ones. t_start lets a caller charge the pair's
    time budget from an earlier instant, which the batch loop uses so the
    width rescues share the pair's budget."""
    t_start = time.perf_counter() if t_start is None else t_start
    ref = np.asarray(reference)
    search = np.asarray(search)
    if float(np.std(ref)) < DEGENERATE_STD or float(np.std(search)) < DEGENERATE_STD:
        return Registration(0.0, 0.0, 0.0, 0.0, False, 0.5, "degenerate",
                            "degenerate_input", time.perf_counter() - t_start)
    rgb = bool(reference_rgb or search_rgb)
    cfg = (optical if optical is not None else optical_config()) if rgb \
        else (config if config is not None else MatchConfig())
    x, y, diag, _ = locate(ref, search, cfg, t_start=t_start)
    if (float(diag["score"]) < RESCUE_PEAK_BELOW
            and int(diag.get("num_candidates_wide", 1)) > 1 and not rgb):
        for op in (grey_erosion, grey_dilation):
            if time.perf_counter() - t_start > RESCUE_START_BEFORE * cfg.time_budget_s:
                break
            ref_cd = op(ref, size=(3, 3)).astype(ref.dtype)
            x2, y2, d2, _ = locate(ref_cd, search, cfg, t_start=t_start)
            if float(d2["score"]) > float(diag["score"]) + RESCUE_MARGIN:
                x, y, diag = x2, y2, d2
    peak = float(diag["score"])
    if model is not None:
        p_present = presence_probability(model, features_for_model(model, diag))
        found = 1 if p_present >= model["prob_threshold"] else 0
        score = float(max(p_present, 1.0 - p_present))
        if rgb:
            found = 1
            score = float(p_present)
        if found and not rgb:
            agree = max(int(diag.get("quad_agree", -1)), 0)
            score *= 0.5 + 0.5 * min(agree / 4.0, 1.0)
    else:
        found = 1 if peak >= FALLBACK_FOUND_THRESHOLD else 0
        score = _fallback_score(peak, float(diag.get("peak_prominence", 0.0)),
                                diag.get("num_candidates_wide", 1))
    theta = cfg.theta_report_sign * float(diag["theta_deg"])
    scale = float(diag["scale"]) * cfg.zoom
    reason = "optical_disclosed" if rgb else ("matched" if found else "rejected")
    return Registration(_finite(x), _finite(y), _finite(theta), _finite(scale, cfg.zoom),
                        bool(found), _finite(score), _regime(diag), reason,
                        float(diag.get("runtime_s", time.perf_counter() - t_start)), diag)
