"""register.py end to end over real generated images.

Every other test exercises a slice; this one runs the shipped entry point as
a subprocess on a manifest of freshly generated pairs, one clean present, one
absent, one RGB optical, and asserts the full contract on the rows that come
back: columns, decisions, pose accuracy on the present pair, zeroed pose on
the rejection, the forced found flag on the optical pair, and score ranges.
"""

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_sense.generator import generate_pair

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    present = generate_pair(seed=4242, style="dram")
    absent = generate_pair(seed=555, style="finfet", absent=True)
    cv2.imwrite(str(d / "p_ref.png"), present["reference"])
    cv2.imwrite(str(d / "p_search.png"), present["search"])
    cv2.imwrite(str(d / "a_ref.png"), absent["reference"])
    cv2.imwrite(str(d / "a_search.png"), absent["search"])
    rgb_ref = np.stack([present["reference"]] * 3, axis=-1)
    rgb_ref[..., 2] = np.clip(rgb_ref[..., 2].astype(int) + 6, 0, 255).astype(np.uint8)
    rgb_search = np.stack([present["search"]] * 3, axis=-1)
    rgb_search[..., 2] = np.clip(rgb_search[..., 2].astype(int) + 6, 0, 255).astype(np.uint8)
    cv2.imwrite(str(d / "o_ref.png"), rgb_ref)
    cv2.imwrite(str(d / "o_search.png"), rgb_search)
    with open(d / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "reference_path", "search_path"])
        w.writerow(["e2e_present", "p_ref.png", "p_search.png"])
        w.writerow(["e2e_absent", "a_ref.png", "a_search.png"])
        w.writerow(["e2e_optical", "o_ref.png", "o_search.png"])
    out = d / "pred.csv"
    proc = subprocess.run([sys.executable, str(REPO / "register.py"),
                           "--input", str(d / "pairs.csv"), "--output", str(out)],
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-2000:]
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    return present["meta"], {r["pair_id"]: r for r in rows}, rows


def test_contract_columns_and_row_count(run):
    _, by_id, rows = run
    assert len(rows) == 3
    assert list(rows[0].keys()) == ["pair_id", "x", "y", "theta", "scale", "found", "score"]
    assert set(by_id) == {"e2e_present", "e2e_absent", "e2e_optical"}


def test_present_pair_is_found_and_accurate(run):
    meta, by_id, _ = run
    r = by_id["e2e_present"]
    assert r["found"] == "1"
    gt = meta["ground_truth"]
    assert np.hypot(float(r["x"]) - gt["x"], float(r["y"]) - gt["y"]) <= 5.0
    assert abs(float(r["theta"]) - meta["relative_rotation_deg"]) <= 1.0
    assert abs(float(r["scale"]) - meta["zoom"]) / meta["zoom"] <= 0.05
    assert 0.25 <= float(r["score"]) <= 1.0


def test_absent_pair_is_rejected_with_zero_pose(run):
    _, by_id, _ = run
    r = by_id["e2e_absent"]
    assert r["found"] == "0"
    assert (r["x"], r["y"], r["theta"], r["scale"]) == ("0", "0", "0", "0")
    assert 0.0 <= float(r["score"]) <= 1.0


def test_optical_pair_is_always_found(run):
    _, by_id, _ = run
    r = by_id["e2e_optical"]
    assert r["found"] == "1"
    assert 0.0 <= float(r["score"]) <= 1.0
