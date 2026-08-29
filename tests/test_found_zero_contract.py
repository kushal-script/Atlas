"""found=0 output-contract test (Phase 2, Day 7).

The rubric requires that when a pair is rejected (found=0) the pose columns
(x, y, theta, scale) are zeroed but the `score` column is KEPT (not zeroed),
so the calibration AUC retains a continuous, monotonic ranking across present
and absent pairs. This test isolates register._process's contract branch with
the localizer and decision functions mocked, so it is fast and dataset-free.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import register  # noqa: E402


def _patch_io(monkeypatch, found, score):
    monkeypatch.setattr(register, "load_gray",
                        lambda *a, **k: (np.zeros((64, 64), dtype=np.float32), False))
    monkeypatch.setattr(register, "locate",
                        lambda *a, **k: (123.4, 234.5, {"theta_deg": 2.0,
                                                         "scale": 1.1}, None))
    monkeypatch.setattr(register, "decide_found", lambda diag, thr: found)
    monkeypatch.setattr(register, "prediction_confidence", lambda diag: score)


def test_found_zero_zeroes_pose_but_keeps_score(monkeypatch):
    _patch_io(monkeypatch, found=0, score=0.31415)
    pid, x, y, theta, scale, found, score = register._process("r.png", "s.png", "p1")
    assert found == 0
    assert (x, y, theta, scale) == (0.0, 0.0, 0.0, 0.0)
    assert score == 0.31415  # KEPT, not zeroed


def test_found_one_keeps_pose_and_score(monkeypatch):
    _patch_io(monkeypatch, found=1, score=0.99)
    pid, x, y, theta, scale, found, score = register._process("r.png", "s.png", "p2")
    assert found == 1
    assert (x, y) == (123.4, 234.5)
    # register applies the locked conventions: theta = -diag["theta_deg"],
    # scale = 10.0 * diag["scale"].
    assert theta == -2.0
    assert scale == 11.0
    assert score == 0.99
