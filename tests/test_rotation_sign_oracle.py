"""The reported rotation convention pinned independently of this generator.

The organisers form each search capture by warping the reference through
cv2.getRotationMatrix2D at the relative rotation and one over the zoom, the
exact statistic _raw_confirm reproduces. Building a pair that way from plain
noise, with no code from this repository's generator, and asserting the
reported theta matches the injected angle in sign and magnitude locks the
whole four convention chain to the organisers' construction.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_sense.localize import MatchConfig, locate


def _organiser_style_pair(theta_deg, zoom, rng):
    spec = cv2.GaussianBlur((rng.random((1000, 1000)) * 255).astype(np.float32),
                            (0, 0), 3.0)
    spec = cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ref = spec
    out = int(round(1000 / zoom))
    m = cv2.getRotationMatrix2D((499.5, 499.5), float(theta_deg), 1.0 / zoom)
    m[0, 2] += (out - 1) / 2.0 - 499.5
    m[1, 2] += (out - 1) / 2.0 - 499.5
    patch = cv2.warpAffine(cv2.blur(ref, (int(zoom), int(zoom))), m, (out, out),
                           flags=cv2.INTER_LINEAR)
    search = (rng.random((1000, 1000)) * 30 + 100).astype(np.uint8)
    cx, cy = 430, 570
    h = out // 2
    search[cy - h:cy - h + out, cx - h:cx - h + out] = patch
    return ref, search, float(cx), float(cy)


@pytest.mark.parametrize("theta", [4.0, -4.0])
def test_reported_theta_matches_the_organisers_construction(theta):
    rng = np.random.default_rng(11 if theta > 0 else 12)
    ref, search, cx, cy = _organiser_style_pair(theta, 10.0, rng)
    x, y, diag, _ = locate(ref, search, MatchConfig())
    cfg = MatchConfig()
    reported = cfg.theta_report_sign * float(diag["theta_deg"])
    assert np.hypot(x - cx, y - cy) <= 3.0
    assert abs(reported - theta) <= 0.35
    assert reported * theta > 0
