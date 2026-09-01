"""The streak corrector must remove streaks and only streaks, and the raw
confirmation must read high on a true instance and low on an absent one.

Both operate on raw pixels ahead of everything the models were fitted on, so
a regression here silently poisons every downstream decision.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_sense.localize import MatchConfig, _raw_confirm, _suppress_streak_rows


def _lattice(h, w, period=12):
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    img = 90 + 50 * ((y % period) < 3) + 40 * ((x % period) < 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_streak_rows_removed_and_lattice_untouched():
    cfg = MatchConfig()
    img = _lattice(400, 400)
    hot = [50, 51, 137, 220, 221, 222, 305]
    streaked = img.astype(np.float32)
    streaked[hot] += 60
    streaked = np.clip(streaked, 0, 255).astype(np.uint8)
    out, n = _suppress_streak_rows(streaked, cfg)
    assert n >= len(hot) - 1
    resid = out.astype(np.float32) - img.astype(np.float32)
    assert abs(resid[hot].mean()) < 12
    cold = [r for r in range(400) if r not in hot]
    assert np.abs(resid[cold]).max() < 1.0


def test_clean_image_is_untouched():
    cfg = MatchConfig()
    img = _lattice(400, 400)
    out, n = _suppress_streak_rows(img, cfg)
    assert n == 0
    assert np.array_equal(out, img)


def test_wide_structure_refuses_correction():
    """Half the rows bright is structure, not streaks; nothing may change."""
    cfg = MatchConfig()
    img = _lattice(400, 400, period=4)
    out, n = _suppress_streak_rows(img, cfg)
    assert n == 0
    assert np.array_equal(out, img)


def test_raw_confirm_separates_present_from_absent():
    rng = np.random.default_rng(7)
    canvas = (rng.random((4000, 4000)) * 60 + 60).astype(np.uint8)
    yy, xx = np.mgrid[0:4000, 0:4000]
    canvas = np.clip(canvas + 80 * (((yy // 40) + (xx // 40)) % 3 == 0), 0, 255).astype(np.uint8)
    ref = canvas[1200:2200, 800:1800].copy()
    import cv2
    search = cv2.resize(canvas, (1000, 1000), interpolation=cv2.INTER_AREA)
    rc = _raw_confirm(ref, search, 4.0, 0.0, (800 + 500) / 4.0, (1200 + 500) / 4.0)
    assert rc is not None and rc["peak"] > 0.5 and rc["agree"]
    absent_ref = (rng.random((1000, 1000)) * 255).astype(np.uint8)
    rc2 = _raw_confirm(absent_ref, search, 4.0, 0.0, 0.0, 0.0)
    assert rc2 is not None and rc2["peak"] < rc["peak"] - 0.2
