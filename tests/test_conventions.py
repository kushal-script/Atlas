"""Contract tests for the coordinate convention and the localizer's accuracy
on cases whose answer is known exactly by construction.

These are the tests that would catch a silent sign flip, an off by one in the
template offset, or a half pixel bias, all of which would be invisible in the
aggregate accuracy numbers but would systematically shift every prediction.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from drift_sense.api import zncc_match
from drift_sense.localize import MatchConfig, locate

ZOOM = 10
SIZE = 1000


def _lattice_canvas(rng, size, block=1700):
    """Blocks of differing lattice pitch separated by routing strips.

    Every window of reference footprint size lands in a block with a distinctive
    pitch or straddles a boundary, so the correct answer is identifiable. A
    single global lattice would be genuinely degenerate and could not be used to
    test localization accuracy at all.
    """
    canvas = np.full((size, size), 95, np.uint8)
    pitches = [(37, 53), (43, 61), (29, 47), (53, 71), (31, 67), (47, 59)]
    for by in range(0, size, block):
        for bx in range(0, size, block):
            py, px = pitches[int(rng.integers(0, len(pitches)))]
            y1, x1 = min(by + block - 220, size), min(bx + block - 220, size)
            if y1 - by < 200 or x1 - bx < 200:
                continue
            tile = np.full((y1 - by, x1 - bx), 40, np.uint8)
            tile[::py, :] = 150
            tile[:, ::px] = 170
            tile[py // 2::py, px // 2::px] = 235
            canvas[by:y1, bx:x1] = tile
    return canvas


def _make_exact_pair(rng, gt_x, gt_y):
    """Build a search image and the reference crop that produced it.

    The search image is an exact area averaged decimation of a fine canvas, and
    the reference is the fine crop centred on the requested search pixel, so the
    correct answer is (gt_x, gt_y) by construction.
    """
    import cv2
    fine = _lattice_canvas(rng, SIZE * ZOOM)
    search = cv2.resize(fine, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    x0 = int(round((gt_x - (SIZE // ZOOM) / 2.0) * ZOOM))
    y0 = int(round((gt_y - (SIZE // ZOOM) / 2.0) * ZOOM))
    reference = fine[y0:y0 + SIZE, x0:x0 + SIZE]
    return reference, search


@pytest.mark.parametrize("gt_x,gt_y", [(250.0, 300.0), (700.0, 180.0), (480.0, 620.0)])
def test_noise_free_localization_is_sub_pixel(gt_x, gt_y):
    rng = np.random.default_rng(7)
    reference, search = _make_exact_pair(rng, gt_x, gt_y)
    x, y, diag, _ = locate(reference, search, MatchConfig())
    assert np.hypot(x - gt_x, y - gt_y) < 1.0, (
        f"predicted ({x:.2f}, {y:.2f}) for truth ({gt_x}, {gt_y}), "
        f"regime {diag['num_candidates']} candidates")


def test_x_is_column_and_y_is_row():
    """A pattern placed far left and near the top must return small x and small y.

    This fails loudly if x and y are ever transposed, which a symmetric test
    case would not detect.
    """
    rng = np.random.default_rng(11)
    reference, search = _make_exact_pair(rng, gt_x=120.0, gt_y=850.0)
    x, y, _, _ = locate(reference, search, MatchConfig())
    assert x < 300.0, f"x should be small for a left placed pattern, got {x:.1f}"
    assert y > 700.0, f"y should be large for a low placed pattern, got {y:.1f}"


def test_localizer_is_deterministic():
    rng = np.random.default_rng(3)
    reference, search = _make_exact_pair(rng, 400.0, 500.0)
    first = locate(reference, search, MatchConfig())[:2]
    second = locate(reference, search, MatchConfig())[:2]
    assert first == second


def test_api_contract():
    rng = np.random.default_rng(5)
    reference, search = _make_exact_pair(rng, 350.0, 450.0)
    result = zncc_match(reference, search)
    for key in ("x", "y", "score"):
        assert key in result, f"missing required key {key}"
        assert isinstance(result[key], float)
    assert 0.0 <= result["score"] <= 1.0
    assert result["confidence_regime"] in (
        "unique_peak", "residual_identified", "tie_break_convention")


def test_confidence_is_higher_when_correct_than_when_degenerate():
    """A unique pattern should outrank a featureless periodic one.

    Ranking metrics depend on this ordering, so it is worth asserting rather
    than assuming.
    """
    import cv2
    rng = np.random.default_rng(13)
    reference, search = _make_exact_pair(rng, 500.0, 500.0)
    good = zncc_match(reference, search)["score"]

    flat = np.full((SIZE * ZOOM, SIZE * ZOOM), 40, np.uint8)
    flat[::37, :] = 150
    flat[:, ::53] = 170
    flat_search = cv2.resize(flat, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    flat_reference = flat[2000:2000 + SIZE, 3000:3000 + SIZE]
    degenerate = zncc_match(flat_reference, flat_search)["score"]
    assert good > degenerate, (
        f"unique pattern confidence {good:.3f} should exceed "
        f"perfectly periodic confidence {degenerate:.3f}")
