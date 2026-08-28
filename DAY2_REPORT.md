# Phase 2 — Day 2 Report

Date: 2026-08-28
Scope: de-risk Phase 2 acceptance — fix the runtime violation with config-only
levers, diagnose (not fix) the 8..12x / ±5° accuracy collapse, and lock two
small contracts (found=0/score, Python version). No algorithm (core math) edits.

## 1. Runtime budget — MET

Hard contract: median ≤ 5 s/pair on representative (full-grid) inputs.

Final `phase2_config()` (src/drift_sense/localize.py:468) levers vs the Day-1
baseline (median 30.55 s, 52% pass within 5 px):

| Lever | Change | Measured median (8 physics pairs) |
|-------|--------|----------------------------------|
| L1 | `wide_sigma_bank_nm` 4→1 level `(6.5,)` | 16.1 s |
| L2 | `prescreen_downsample` 2→4 | 10.2 s |
| L3 | `coarse_rotations_deg` step 0.5°→1.0° (11 vals) | 7.38 s |
| L4 | `refine_levels` 2→1 | **4.42 s (max 4.86 s)** |

Notes:
- Prescreen/blur cuts alone plateaued (~6–7 s); the dominant cost was the
  full-resolution re-scoring + refine, not the prescreen templates. `refine_levels=1`
  is the lever that actually crossed the line.
- `coarse_scales` density was NOT reduced (kept 0.80..1.20 step 0.02, 21 vals)
  per the lever-4 guard. Rotation density was widened (coarser) because the refine
  stage recovers sub-degree pose from a 1.0° grid.
- `nominal_accept_score` stays 9.9 (early-exit ineffective: NCC never reaches 9.9),
  so every pair runs the full widened grid — the measured 4.42 s is the realistic
  worst case, not a lucky nominal-path shortcut.

**Result: median 4.42 s ≤ 5 s. Budget satisfied.** Run command:
`python scripts/measure_runtime.py --dataset data/p2_smoke --n 8`.

## 2. Accuracy trade — WITHIN SLACK

Extended pose grid (scripts/pose_robustness.py, `phase2_config()`,
MAGNIFICATIONS=(8,9,10,11,12), ROTATIONS=(-5,-2.5,0,2.5,5), repeats=1):

- Overall pass within 5 px: **48.0%** (Day-1 baseline 52.0%) → regression **−4 pts**,
  within the agreed ≥47% floor (5-pt slack). Median error 9.31 px (baseline ~9 px).
- By magnification: 8→20% (med 109.6 px), 9→40% (31.6), 10→60% (2.0),
  11→100% (1.5), 12→20% (59.7).
- By rotation: −5→40% (9.3 px), −2.5→20% (59.7), 0→60% (1.9), 2.5→60% (2.1),
  5→60% (1.9).

Net: 11x improved (80→100%), 12x regressed (60→20%) and −2.5° regressed
(40→20%) under `refine_levels=1`; the floor held overall.

**No lever was reverted due to excessive accuracy loss. Floor honored.**

## 3. Oracle diagnosis — collapse is APPEARANCE, not SEARCH

scripts/diagnose_p2_extremes.py generates pairs at mag∈{8..12}, rot∈{0,±5}
(6 repeats, forced identifiable site) and runs two localizations:
- `e_free`: unconstrained `phase2_config()`.
- `e_oracle`: search pinned to the EXACT true scale+rotation, `refine_levels=0`,
  `residual_disambiguation=False`, `zoom=true_scale` — a pure appearance
  measurement (no search, no refinement can move the peak).

Result (90 pairs, experiments/20260828_015012_diagnose_p2_extremes):

| | median e_free | median e_oracle | oracle ≤5 px |
|---|---|---|---|
| Overall | 1.845 px | **534.1 px** | **0 / 90** |
| by mag 8/9/10/11/12 | ~1.8–12.7 px | 470–634 px | 0 / 18 each |
| by rot 0/±5 | ~1.7–1.9 px | 466–550 px | 0 / 30 each |

Interpretation:
- The free grid recovers the true location at ~1.85 px median, so pose-grid
  **coverage is adequate** — the grid is not missing the right hypothesis.
- Yet pinning to the exact true pose yields a ~500 px error in **every** cell
  (0/90 within 5 px). The raw template correlation at the correct scale/rotation
  does not peak at the target; only the refine+residual post-processing in the
  free path rescues it (and only partially at 8x/12x).
- Conclusion: the 8..12x / ±5° collapse is an **appearance/template (magnification
  cliff) bottleneck** — the template model's scaling/blur handling fails to make
  the true-pose template localize off-nominal zoom. WIDENING the pose grid will
  NOT fix this; it already finds the answer. Day 3 should invest in template
  generation / scale-relative sizing / blur bank / contrast, not in more poses.

## 4. found=0 / score contract — LOCKED (register.py)

- Conventions verified (tests/test_pose_conventions.py, 9 tests): `theta = -diag["theta_deg"]`
  (CCW-positive), `scale = 10.0 * diag["scale"]`.
- `_process()` (register.py:79) now carries the locked contract: when a future
  rejection sets `found = 0`, the pose columns are zeroed (`x = y = theta = scale = 0.0`)
  but **`score` is intentionally KEPT as `diag["score"]`** (not zeroed) so the
  calibration AUC has a continuous, monotonic ranking over present+absent pairs.
  Implemented as a guarded branch + comment; no threshold yet (Day 3).

## 5. Python-version contract — RESOLVED

- `requirements.txt`: `numpy==2.5.1` → `numpy==2.1.3` (supports both 3.11 and
  3.12). Verified on the current 3.12 venv.
- Full suite: `python -m pytest tests/ -q` → **34 passed** (221 s) on numpy 2.1.3.
- Could not exercise a fresh 3.11 venv (interpreter not installed in this env);
  the pin is 3.11/3.12-compatible by version bounds. Action for reviewer: create a
  3.11 venv and re-run pytest to formally close the 3.11 leg.

## 6. Status checklist

- [x] T1 runtime ≤5 s via config levers (4.42 s median, 4.86 s max)
- [x] T2 oracle diagnosis → appearance/template bottleneck, not search
- [x] T3 found=0 contract locked in register.py (score preserved)
- [x] T4 numpy down-pin to 2.1.3; pytest 34 passed
- [x] T5 pose-grid 48% (within 47% floor); pytest 34 green; register.py torch-free
      (`'torch' in register.py` → False)

## Open items for Day 3

1. Appearance/template work (magnification cliff): template scaling, blur bank,
   contrast — drive by the oracle metric (e_oracle must drop well below 5 px).
2. Apply a real rejection threshold at the `found=0` branch (Set C no-instance).
3. Formal 3.11 venv pytest run.
4. If a bounded, accuracy-critical algorithm tweak is needed to meet the 5 px
   target at 8x/12x, escalate per the Day-2 rule-6 open question before changing
   core math.
