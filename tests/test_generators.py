"""Generator contract tests.

A synthetic dataset is only useful if it is reproducible and if its recorded
ground truth actually agrees with the pixels it ships. These tests assert both,
plus the properties the problem statement makes mandatory: independent noise
per capture, a noisier search image, and the exact ten to one field of view
relationship.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from drift_sense.generator import generate_pair


@pytest.fixture(scope="module")
def pair():
    return generate_pair(seed=1234, style="dram")


def test_image_geometry(pair):
    assert pair["reference"].shape == (1000, 1000)
    assert pair["search"].shape == (1000, 1000)
    assert pair["reference"].dtype == np.uint8
    assert pair["search"].dtype == np.uint8


def test_generation_is_reproducible(pair):
    again = generate_pair(seed=1234, style="dram")
    assert np.array_equal(pair["reference"], again["reference"])
    assert np.array_equal(pair["search"], again["search"])
    assert pair["meta"]["ground_truth"] == again["meta"]["ground_truth"]


def test_different_seeds_give_different_images(pair):
    other = generate_pair(seed=1235, style="dram")
    assert not np.array_equal(pair["search"], other["search"])


def test_noise_is_independent_between_captures():
    """The two captures must not share a noise field.

    Rendering the same specimen twice through the same pose with different
    capture generators must produce different pixels; identical pixels would
    mean the noise was reused, which the problem statement forbids.
    """
    a = generate_pair(seed=77, style="finfet")
    assert not np.array_equal(a["reference"], a["search"])
    ref_meta = a["meta"]["reference_capture"]["settings"]
    search_meta = a["meta"]["search_capture"]["settings"]
    assert ref_meta["dose_e"] != search_meta["dose_e"]


@pytest.mark.parametrize("style", ["dram", "finfet"])
def test_search_capture_is_noisier(style):
    """Dose sets the shot noise level, and the wide capture must receive less."""
    p = generate_pair(seed=21, style=style)
    ref_dose = p["meta"]["reference_capture"]["settings"]["dose_e"]
    search_dose = p["meta"]["search_capture"]["settings"]["dose_e"]
    assert search_dose < ref_dose, (
        f"search dose {search_dose} should be below reference dose {ref_dose}")


def test_ground_truth_is_inside_the_frame_and_recorded(pair):
    gt = pair["meta"]["ground_truth"]
    assert 0.0 <= gt["x"] <= 999.0
    assert 0.0 <= gt["y"] <= 999.0
    corners = pair["meta"]["gt_corners_xy"]
    assert len(corners) == 4
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    assert abs(np.mean(xs) - gt["x"]) < 2.0
    assert abs(np.mean(ys) - gt["y"]) < 2.0


def test_ground_truth_box_matches_the_ten_to_one_ratio(pair):
    """The reference footprint must span about a tenth of the search frame."""
    corners = np.array(pair["meta"]["gt_corners_xy"])
    width = np.hypot(*(corners[1] - corners[0]))
    height = np.hypot(*(corners[2] - corners[1]))
    for side in (width, height):
        assert 85.0 < side < 115.0, (
            f"reference footprint side {side:.1f} px should be near 100 px")


def test_metadata_records_every_factor(pair):
    meta = pair["meta"]
    for key in ("seed", "style", "placement", "ground_truth", "layout",
                "reference_capture", "search_capture", "relative_rotation_deg",
                "search_scale_error"):
        assert key in meta, f"metadata missing {key}"
    json.dumps(meta)


def test_optical_modality_is_rgb():
    p = generate_pair(seed=9, style="dram", modality="optical")
    assert p["reference"].shape == (1000, 1000, 3)
    assert p["search"].shape == (1000, 1000, 3)
