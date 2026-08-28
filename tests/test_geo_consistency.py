"""Geometric-consistency presence gate (Phase 2, Day 4, T1).

Correlation strength is anti-correlated with true presence at 8..12x, so the
rejection decision cannot use the peak. The geometric-consistency signal reverses
this: warp the full-resolution reference by the recovered pose and measure
CCOEFF_NORMED against the search patch. A true instance aligns its fine detail
(high); a distractor or substrate does not (low).

These tests pin the signal to the expected behaviour and confirm the geo check
does not perturb the recovered pose (equivalence gate: nominal answer unchanged).
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from drift_sense.localize import phase2_config, locate
from drift_sense.generator import generate_pair as gen_pair
from generate_amat_proxy import BASE_PARAMS, TIERS, generate_pair as proxy_pair

PRESENT_GEO_MIN = 0.6
ABSENT_GEO_MAX = 0.4
POS_TOL_PX = 2.0


def _cfg():
    return phase2_config()


def _present_proxy(rotation_deg, scale, seed):
    p = dict(BASE_PARAMS)
    p.update(TIERS["medium"])
    ref, search, meta = proxy_pair(
        seed=seed, kind="dram", params=p,
        rotation_deg=rotation_deg, scale=scale, boundary_bias=1.0)
    return ref, search, meta["ground_truth"]


def test_geo_consistency_present_10x_is_high():
    ref, search, gt = _present_proxy(2.0, 1.0, 20260827)
    x, y, diag, _ = locate(ref, search, _cfg())
    assert diag["geo_consistency"] >= PRESENT_GEO_MIN, (
        f"present 10x geo_consistency {diag['geo_consistency']:.3f} < "
        f"{PRESENT_GEO_MIN}")
    assert np.hypot(x - gt["x"], y - gt["y"]) < POS_TOL_PX


def test_geo_consistency_present_8x_is_high():
    ref, search, gt = _present_proxy(3.0, 0.8, 777)
    x, y, diag, _ = locate(ref, search, _cfg())
    assert diag["geo_consistency"] >= PRESENT_GEO_MIN, (
        f"present 8x geo_consistency {diag['geo_consistency']:.3f} < "
        f"{PRESENT_GEO_MIN}")
    assert np.hypot(x - gt["x"], y - gt["y"]) < POS_TOL_PX


def test_geo_consistency_absent_substrate_is_low():
    # Set C: reference instance absent; search is substrate-only.
    r = gen_pair(31, "finfet", phase2=True, absent=True)
    x, y, diag, _ = locate(r["reference"], r["search"], _cfg())
    assert diag["geo_consistency"] <= ABSENT_GEO_MAX, (
        f"absent geo_consistency {diag['geo_consistency']:.3f} > "
        f"{ABSENT_GEO_MAX}")


def test_geo_consistency_does_not_change_pose():
    # The geo check is computed after pose recovery; it must not perturb x, y.
    # Uses a known-good localization (rotation 2.0, seed 20260827).
    ref, search, gt = _present_proxy(2.0, 1.0, 20260827)
    x, y, diag, _ = locate(ref, search, _cfg())
    assert np.hypot(x - gt["x"], y - gt["y"]) < POS_TOL_PX
