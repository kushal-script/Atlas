"""Deeper containers must load to the same eight bit frame a plain export
gives, whatever range the data actually occupies.

An evaluator's tool may write an eight bit capture into a sixteen bit PNG
without rescaling, or a twelve bit detector's range into the same container;
a fixed shift would turn the first into a blank frame the pipeline rejects
and crush the second to sixteen levels.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_sense.localize import load_gray


def _frame():
    rng = np.random.default_rng(3)
    return (rng.random((120, 160)) * 200 + 20).astype(np.uint8)


def test_eight_bit_data_in_a_sixteen_bit_container_is_identity(tmp_path):
    frame = _frame()
    cv2.imwrite(str(tmp_path / "plain.png"), frame)
    cv2.imwrite(str(tmp_path / "wide.png"), frame.astype(np.uint16))
    plain, _ = load_gray(tmp_path / "plain.png")
    wide, _ = load_gray(tmp_path / "wide.png")
    assert wide.dtype == np.uint8
    assert np.array_equal(plain, wide)


def test_full_range_sixteen_bit_keeps_the_shift_convention(tmp_path):
    frame = _frame()
    cv2.imwrite(str(tmp_path / "plain.png"), frame)
    cv2.imwrite(str(tmp_path / "full.png"), frame.astype(np.uint16) << 8)
    plain, _ = load_gray(tmp_path / "plain.png")
    full, _ = load_gray(tmp_path / "full.png")
    assert np.array_equal(plain, full)


def test_twelve_bit_range_is_rescaled_not_crushed(tmp_path):
    frame = _frame()
    twelve = np.round(frame.astype(np.float32) * (4095.0 / 255.0)).astype(np.uint16)
    cv2.imwrite(str(tmp_path / "twelve.png"), twelve)
    back, _ = load_gray(tmp_path / "twelve.png")
    assert back.dtype == np.uint8
    assert int(back.max()) > 100
    assert np.abs(back.astype(int) - frame.astype(int)).max() <= 1
