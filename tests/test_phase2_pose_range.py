"""The search grid must cover the Phase 2 pose ranges the addendum discloses.

Nothing else in the suite exercises the extremes. A grid that silently narrowed,
or a refinement whose level count changed without its step sizes, would cost pose
credit on exactly the pairs at the edges of the disclosed range and no existing
test would notice. These assertions are cheap and pin the contract; the end to end
case at the corner of the range is slower and pins that the pipeline actually
resolves a pose there.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from drift_sense.localize import MatchConfig, locate

ZOOM_MIN, ZOOM_MAX = 8.0, 12.0
ROT_LIMIT = 5.0


def test_scale_grid_spans_the_disclosed_zoom_range():
    cfg = MatchConfig()
    lo = min(cfg.coarse_scales) * cfg.zoom
    hi = max(cfg.coarse_scales) * cfg.zoom
    assert lo <= ZOOM_MIN + 1e-9, f"grid starts at {lo}, misses zoom {ZOOM_MIN}"
    assert hi >= ZOOM_MAX - 1e-9, f"grid ends at {hi}, misses zoom {ZOOM_MAX}"


def test_rotation_grid_spans_plus_minus_five_degrees():
    cfg = MatchConfig()
    assert min(cfg.coarse_rotations_deg) <= -ROT_LIMIT + 1e-9
    assert max(cfg.coarse_rotations_deg) >= ROT_LIMIT - 1e-9


def test_grid_step_is_no_coarser_than_the_refinement_can_close():
    """A coarse step wider than twice the refinement start cannot be walked in."""
    cfg = MatchConfig()
    rot = sorted(cfg.coarse_rotations_deg)
    sca = sorted(cfg.coarse_scales)
    rot_step = max(b - a for a, b in zip(rot, rot[1:]))
    sca_step = max(b - a for a, b in zip(sca, sca[1:]))
    assert rot_step <= 2 * cfg.refine_rot_step_deg * 2 + 1e-9
    assert sca_step <= 2 * cfg.refine_scale_step * 2 + 1e-9
    assert min(sca) * cfg.zoom == 8.0
    assert max(sca) * cfg.zoom == 12.0
    assert min(rot) == -5.0
    assert max(rot) == 5.0


def test_refinement_resolves_inside_the_full_credit_bands():
    """Full pose credit needs 0.25 degrees and 1 percent; the final step must beat both."""
    cfg = MatchConfig()
    rot = cfg.refine_rot_step_deg * 2
    sca = cfg.refine_scale_step * 2
    for _ in range(cfg.refine_levels):
        rot /= 2.0
        sca /= 2.0
    assert rot < 0.25, f"final rotation step {rot} does not resolve the 0.25 degree band"
    assert sca * 100.0 < 1.0, f"final scale step {100*sca} percent does not resolve the 1 percent band"


@pytest.mark.parametrize("zoom,rot", [(ZOOM_MIN, ROT_LIMIT), (ZOOM_MAX, -ROT_LIMIT)])
def test_localizes_at_the_corners_of_the_disclosed_range(zoom, rot):
    """A synthetic pair at the extreme corners, with truth known by construction."""
    rng = np.random.default_rng(20260901)
    big = 900
    field = rng.normal(128, 34, (big, big)).astype(np.float32)
    field = np.clip(field, 0, 255)
    field[400:460, 300:380] += 70.0
    field[520:545, 250:470] -= 55.0
    ref_n = 1000
    cx, cy = 380.0, 430.0
    half = ref_n / (2.0 * zoom)
    import cv2
    M = cv2.getRotationMatrix2D((cx, cy), -rot, zoom)
    M[0, 2] += ref_n / 2.0 - cx
    M[1, 2] += ref_n / 2.0 - cy
    ref = cv2.warpAffine(field, M, (ref_n, ref_n), flags=cv2.INTER_LINEAR)
    search = np.clip(field + rng.normal(0, 4.0, field.shape), 0, 255).astype(np.uint8)
    ref = np.clip(ref + rng.normal(0, 2.0, ref.shape), 0, 255).astype(np.uint8)

    x, y, diag, _ = locate(ref, search, MatchConfig())
    err = float(np.hypot(x - cx, y - cy))
    assert err <= 5.0, f"zoom {zoom} rot {rot}: error {err:.2f} px, diag scale {diag['scale']}"
    assert abs(diag["scale"] * MatchConfig().zoom - zoom) / zoom < 0.05
    reported = MatchConfig().theta_report_sign * float(diag["theta_deg"])
    assert abs(reported - rot) <= 0.5, f"theta {reported:.2f} vs injected {rot}"
    assert reported * rot > 0
