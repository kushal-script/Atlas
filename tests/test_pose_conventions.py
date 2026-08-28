"""Pose convention contract tests (Phase 2, Day 1).

These tests pin down the two highest-risk correctness conventions for Pose
Recovery before any tuning is attempted:

  * theta sign  -- the scoring rubric requires theta CCW-positive about the
                   match centre. We verify what the localizer actually emits and
                   therefore what register.py must output.
  * scale unit  -- the scorer expects the magnification (8, 10, 12), while the
                   localizer works in `scale` relative to the 10x nominal. We
                   verify the mapping from internal scale to output magnification.

Ground truth is taken from the project's own generator (generate_amat_proxy),
because that is exactly what the evaluation harness uses: it warps the search
image with cv2.getRotationMatrix2D(centre, rotation_deg, 1.0 / scale) and records
`relative_rotation_deg` / `search_scale_error` as truth. Building pairs through
the generator is therefore the faithful "rotate the search by a known +theta and
scale by a known factor" construction the task asks for, and it respects the
localizer's assumption that the reference is a 10x (higher resolution) crop.

VERIFIED CONVENTION (with evidence, see test bodies / Day 1 report):
  * The localizer returns theta_deg == -(true CCW rotation). To report a
    CCW-positive theta that matches the scorer's ground truth we therefore
    output  theta = -diag["theta_deg"].
  * The localizer returns scale == true magnification / 10. The output
    magnification is therefore  scale_out = 10.0 * diag["scale"].
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from drift_sense.localize import phase2_config, locate
from generate_amat_proxy import BASE_PARAMS, TIERS, generate_pair

POS_TOL_PX = 2.0
THETA_TOL_DEG = 0.5
SCALE_TOL = 0.1

TIER = "medium"
# These seeds are verified to localize cleanly under phase2_config (see Day 1
# report). The widened grid is seed sensitive; a contract test pins the
# convention on a known-good construction rather than probing robustness, which
# is a separate Day 2 measurement.
ROT_SEED = 12345
SCALE_SEED = 777


def _pair(rotation_deg, scale, seed=20260827):
    """Build a (ref, search, gt) pair from the project generator.

    boundary_bias=1.0 forces the reference site onto a mat/strip boundary so the
    content is uniquely identifiable; this isolates pose recovery from periodic
    ambiguity, exactly as scripts/pose_robustness.py does.
    """
    params = dict(BASE_PARAMS)
    params.update(TIERS[TIER])
    ref, search, meta = generate_pair(
        seed=seed, kind="dram", params=params,
        rotation_deg=rotation_deg, scale=scale, boundary_bias=1.0)
    return ref, search, meta["ground_truth"]


def _locate(ref, search):
    cfg = phase2_config()
    x, y, diag, _ = locate(ref, search, cfg)
    return {
        "x": x, "y": y,
        "theta_out": -diag["theta_deg"],   # CCW-positive, matches scorer GT
        "scale_out": 10.0 * diag["scale"],  # magnification units
        "diag": diag,
    }


@pytest.mark.parametrize("rot_deg", [3.0, -4.0, 5.0, -5.0])
def test_theta_is_ccw_positive_matches_ground_truth(rot_deg):
    """The localizer emits theta == -applied; output must be the negation.

    Evidence: generator applies cv2.getRotationMatrix2D(..., +rot_deg, ...) (CCW
    positive) and records rot_deg as ground truth. The localizer returns
    approximately -rot_deg, so the committed output theta = -diag["theta_deg"]
    recovers the CCW-positive ground truth within THETA_TOL_DEG.
    """
    ref, search, gt = _pair(rot_deg, 1.0, seed=ROT_SEED)
    out = _locate(ref, search)
    assert np.hypot(out["x"] - gt["x"], out["y"] - gt["y"]) < POS_TOL_PX, (
        f"position off: pred ({out['x']:.2f},{out['y']:.2f}) "
        f"gt ({gt['x']:.2f},{gt['y']:.2f})")
    assert abs(out["theta_out"] - rot_deg) < THETA_TOL_DEG, (
        f"theta sign/value wrong: output {out['theta_out']:+.3f} "
        f"vs ground truth {rot_deg:+.1f} (raw diag theta "
        f"{out['diag']['theta_deg']:+.3f})")
    assert np.sign(out["theta_out"]) == np.sign(rot_deg), (
        f"theta sign flipped: output {out['theta_out']:+.3f} for "
        f"ground truth {rot_deg:+.1f}")


def test_theta_sign_symmetry_positive_and_negative():
    """+theta and -theta must be recovered with opposite, correct signs."""
    ref_p, search_p, _ = _pair(3.0, 1.0, seed=ROT_SEED)
    ref_n, search_n, _ = _pair(-4.0, 1.0, seed=ROT_SEED)
    pos = _locate(ref_p, search_p)
    neg = _locate(ref_n, search_n)
    assert pos["theta_out"] > 0 and neg["theta_out"] < 0, (
        f"sign symmetry broken: +3 -> {pos['theta_out']:+.3f}, "
        f"-4 -> {neg['theta_out']:+.3f}")
    assert abs(pos["theta_out"] - 3.0) < THETA_TOL_DEG
    assert abs(neg["theta_out"] - (-4.0)) < THETA_TOL_DEG


@pytest.mark.parametrize("scale,seed", [
    (1.0, 777), (1.1, 777), (1.2, 12345),
])
def test_scale_unit_is_magnification(scale, seed):
    """Output scale column == 10 * internal scale == true magnification.

    Evidence: generator scale=1.2 yields output ~12.0; scale=1.0 yields ~10.0.
    The localizer's internal scale is magnification / 10, so the output
    magnification is 10.0 * diag["scale"].
    """
    ref, search, gt = _pair(0.0, scale, seed=seed)
    out = _locate(ref, search)
    assert np.hypot(out["x"] - gt["x"], out["y"] - gt["y"]) < POS_TOL_PX, (
        f"position off at scale {scale}: pred ({out['x']:.2f},{out['y']:.2f}) "
        f"gt ({gt['x']:.2f},{gt['y']:.2f})")
    expected = 10.0 * scale
    assert abs(out["scale_out"] - expected) < SCALE_TOL, (
        f"scale unit wrong: output {out['scale_out']:.3f} "
        f"vs expected {expected:.1f} (raw diag scale {out['diag']['scale']:.4f})")


def test_scale_8x_localizes():
    """8x (scale=0.8) is in range and must map to output ~8.0.

    Evidence: generator scale=0.8 -> rec_scale=0.8000, out_scale=8.000,
    position error ~1.9 px (measured). NOTE: localization at the 0.8 grid edge
    is seed sensitive (some pairs collapse to a wrong position); this fixed seed
    is a known-good case that exercises the convention. Robustness across the 8x
    tier is a Day 2 concern, not a unit/sign bug.
    """
    ref, search, gt = _pair(0.0, 0.8, seed=SCALE_SEED)
    out = _locate(ref, search)
    assert np.hypot(out["x"] - gt["x"], out["y"] - gt["y"]) < POS_TOL_PX, (
        f"8x position off: pred ({out['x']:.2f},{out['y']:.2f}) "
        f"gt ({gt['x']:.2f},{gt['y']:.2f})")
    assert abs(out["scale_out"] - 8.0) < SCALE_TOL, (
        f"8x scale unit wrong: output {out['scale_out']:.3f} "
        f"(raw diag scale {out['diag']['scale']:.4f})")
