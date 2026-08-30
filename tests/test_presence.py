"""The two feature constructions must agree, or the shipped model is poisoned.

register.py featurizes a live diagnostics dict while the fitting script
featurizes recorded rows; these are the same twelve numbers built from two
shapes of input, and any silent divergence would fit a model on one feature
space and score it on another.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_sense.presence import (FEATURES, features_from_diag,
                                  features_from_record, presence_probability)


DIAG = {
    "score": 0.71, "peak_prominence": 6.2, "num_candidates_wide": 14,
    "num_candidates": 2, "search_noise_sigma": 5.5, "nominal_score": 0.64,
    "peak_over_p99": 0.21, "pose_source": "wide_grid",
    "quad_disp": 1.5, "quad_agree": 3,
    "stage2": {"z": 3.1, "margin": 120.0, "mad": 40.0},
}
RECORD = {
    "peak": 0.71, "prom": 6.2, "wide": 14, "strict": 2, "noise": 5.5,
    "nominal": 0.64, "over_p99": 0.21, "pose_source": "wide_grid",
    "z": 3.1, "margin": 120.0, "mad": 40.0,
    "quad_disp": 1.5, "quad_agree": 3,
}


def test_pose_wide_matches_what_the_localizer_actually_emits():
    """The fixtures must use the localizer's own string, not a plausible one.

    An earlier revision compared pose_source against "wide" while the localizer
    emits "wide_grid", so the feature was constant zero; the fixtures used
    "wide" too, which made this file agree with a bug instead of catching it.
    """
    from drift_sense.presence import FEATURES, _assemble
    i = FEATURES.index("pose_wide")
    assert _assemble(0.7, 6.2, 14, 2, 5.5, 0.64, 0.21, 1.0, 0.0, 0.0,
                     "wide_grid")[i] == 1.0
    assert _assemble(0.7, 6.2, 14, 2, 5.5, 0.64, 0.21, 1.0, 0.0, 0.0,
                     "nominal")[i] == 0.0


def test_diag_and_record_features_agree():
    a = features_from_diag(DIAG)
    b = features_from_record(RECORD)
    assert len(a) == len(FEATURES)
    assert a == b


def test_missing_stage2_fills_indicator():
    d = dict(DIAG, stage2={})
    f = features_from_diag(d)
    names = dict(zip(FEATURES, f))
    assert names["z_fill"] == 0.0
    assert names["z_missing"] == 1.0
    assert names["mom"] == 0.0


def test_probability_is_monotonic_in_peak():
    model = {"mu": [0.0] * len(FEATURES), "sd": [1.0] * len(FEATURES),
             "weights": [1.0] + [0.0] * (len(FEATURES) - 1), "bias": 0.0}
    lo = presence_probability(model, features_from_diag(dict(DIAG, score=0.2)))
    hi = presence_probability(model, features_from_diag(dict(DIAG, score=0.9)))
    assert hi > lo
