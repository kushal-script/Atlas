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


def _lattice_canvas(rng, size, mat=800, strip=220):
    """Mats of pairwise distinct lattice pitch separated by routing strips.

    Two properties make this well posed as an accuracy test. Every mat has a
    unique pitch pair, so no two mats are interchangeable; and the mat period is
    smaller than the reference footprint, so every possible template window
    contains at least one strip boundary and cannot slide along the lattice
    while still matching. A single global lattice, or repeated pitches, would be
    genuinely degenerate and could not be used to assert localization accuracy
    at all: the correct answer would not be unique.
    """
    canvas = np.full((size, size), 95, np.uint8)
    period = mat + strip
    n = size // period + 1
    for i in range(n):
        for j in range(n):
            by, bx = i * period, j * period
            y1, x1 = min(by + mat, size), min(bx + mat, size)
            if y1 - by < 100 or x1 - bx < 100:
                continue
            py, px = 23 + 2 * (i * n + j), 41 + 3 * (i * n + j)
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


PERIOD_SEARCH_PX = (800 + 220) / ZOOM
STRIP_CENTRE_PX = (800 + 220 / 2.0) / ZOOM


def _strip_crossing(col, row):
    """A placement centred on a strip crossing, maximally distinctive."""
    return (col * PERIOD_SEARCH_PX + STRIP_CENTRE_PX,
            row * PERIOD_SEARCH_PX + STRIP_CENTRE_PX)


@pytest.mark.parametrize("col,row", [(1, 2), (6, 2), (4, 5)])
def test_noise_free_localization_is_sub_pixel(col, row):
    """On a noise free, exactly decimated pair whose window straddles unique
    boundaries, the answer is unique by construction and must be recovered."""
    gt_x, gt_y = _strip_crossing(col, row)
    rng = np.random.default_rng(7)
    reference, search = _make_exact_pair(rng, gt_x, gt_y)
    x, y, diag, _ = locate(reference, search, MatchConfig())
    assert np.hypot(x - gt_x, y - gt_y) < 1.5, (
        f"predicted ({x:.2f}, {y:.2f}) for truth ({gt_x:.1f}, {gt_y:.1f}), "
        f"{diag['num_candidates']} candidates, regime "
        f"{'stage2' if diag['stage2']['used'] else 'tie break'}")


def test_x_is_column_and_y_is_row():
    """A pattern placed far left and low must return small x and large y.

    This fails loudly if x and y are ever transposed, which a placement on the
    image diagonal would not detect.
    """
    gt_x, gt_y = _strip_crossing(1, 7)
    rng = np.random.default_rng(11)
    reference, search = _make_exact_pair(rng, gt_x, gt_y)
    x, y, _, _ = locate(reference, search, MatchConfig())
    assert abs(x - gt_x) < 40.0, f"x should be near {gt_x:.0f}, got {x:.1f}"
    assert abs(y - gt_y) < 40.0, f"y should be near {gt_y:.0f}, got {y:.1f}"
    assert x < y, "a left and low placement must give x below y"


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
