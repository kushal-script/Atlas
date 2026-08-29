"""Magnification-cliff template test (Phase 2, Day 3, T1).

Two gates, both mandatory:

1. Oracle metric (appearance, not search): with the search pinned to the EXACT
   true scale + rotation, and refine/residual disabled, the localization error
   must be <= 5 px at 8, 9, 10, 11, 12x. If the true pose cannot localize, the
   template PSF does not match the search image's PSF off-nominal -- the
   magnification cliff. This codifies the oracle experiment as a regression test.

2. Nominal-10x equivalence: the full phase2_config() answer at scale=1.0 must be
   unchanged within 1e-3 px versus the pre-change baseline, proving the
   effective-zoom-aware blur (base / scale) is a no-op at scale=1.0.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from drift_sense.localize import phase2_config, locate, MatchConfig
from generate_amat_proxy import BASE_PARAMS, TIERS, generate_pair

# Pre-change nominal-10x answer (captured on commit before the T1 blur fix;
# antialias remains off in phase2_config, so the per-scale blur is a no-op at
# scale=1.0 and this baseline is unchanged within 1e-3 px).
BASELINE_X = 598.9141913354397
BASELINE_Y = 556.8861803412437
NOMINAL_EQ_TOL = 1e-3

ORACLE_TOL_PX = 5.0
TIER = "medium"


def _pair(scale, seed):
    params = dict(BASE_PARAMS)
    params.update(TIERS[TIER])
    ref, search, meta = generate_pair(
        seed=seed, kind="dram", params=params,
        rotation_deg=0.0, scale=scale, boundary_bias=1.0)
    return ref, search, meta["ground_truth"]


def _oracle_config(true_scale):
    """Pin to the true pose; isolate appearance from search/refine."""
    cfg = phase2_config()
    cfg.coarse_scales = (round(true_scale, 4),)
    cfg.coarse_rotations_deg = (0.0,)
    cfg.refine_levels = 0
    cfg.residual_disambiguation = False
    cfg.nominal_accept_score = 99.0   # ensure the wide (true-scale) grid runs
    return cfg


@pytest.mark.parametrize("scale,seed", [
    (0.8, 1001), (0.9, 1002), (1.0, 1003), (1.1, 1004), (1.2, 1005),
])
def test_oracle_true_pose_localizes(scale, seed):
    """At the exact true pose, error must be <= 5 px across 8..12x."""
    ref, search, gt = _pair(scale, seed)
    cfg = _oracle_config(scale)
    x, y, diag, _ = locate(ref, search, cfg)
    err = float(np.hypot(x - gt["x"], y - gt["y"]))
    assert err <= ORACLE_TOL_PX, (
        f"oracle (true pose) failed at scale {scale}: err={err:.2f} px "
        f"(pred {x:.2f},{y:.2f} gt {gt['x']:.2f},{gt['y']:.2f}); "
        f"template PSF still mismatched off-nominal")


def test_nominal_10x_equivalence_unchanged():
    """Full phase2_config() at scale=1.0 must match pre-change baseline."""
    ref, search, gt = _pair(1.0, 4242)
    cfg = phase2_config()
    x, y, diag, _ = locate(ref, search, cfg)
    assert abs(x - BASELINE_X) < NOMINAL_EQ_TOL, (
        f"nominal x changed: {x} vs baseline {BASELINE_X}")
    assert abs(y - BASELINE_Y) < NOMINAL_EQ_TOL, (
        f"nominal y changed: {y} vs baseline {BASELINE_Y}")
    # and it still localizes correctly
    assert np.hypot(x - gt["x"], y - gt["y"]) < 3.0
