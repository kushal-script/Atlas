# Development log

Major changes in chronological order. Each entry corresponds to one commit.

## 2026 08 07, project start and dataset generator

Toolchain decision: pure Python with numpy, scipy, Pillow, OpenCV and matplotlib inside a project virtual environment. Simulation suites (process emulators, finite element tools) were considered and rejected: the deliverable must run from a plain clone with pip installed requirements, and the scored part of the generator is the SEM image formation model, which is fully expressible with array code.

Built the generator as specimen first, capture second (see architecture.md). Two layout styles implemented, DRAM and FinFET, both with per pair structural randomisation and deliberate aperiodic anchors (mat stripes for DRAM, logic cell randomness for FinFET) plus deliberately hard periodic regions (mat interiors, an SRAM block). First visual review led to one geometry correction: diffusion breaks initially erased about 43 percent of the fin field because every cell boundary carried a 1.5 gate pitch wide break; reduced to a 0.8 gate pitch break with wider cells, matching single diffusion break practice, after which the fin fabric reads like real top down CD SEM frames.

## 2026 08 07, localizer

Implemented the matched formation approach: instead of generic multi scale template matching, the reference is pushed through the same optical chain the search image went through (blur to search resolution, affine resample onto the search grid with point sampling so aliasing is reproduced), then normalized cross correlation with a coarse to fine rotation and scale search. Tie break per the problem statement: among near equal peaks, closest to the search image center. Smoke test on three pairs landed under 0.2 px error.

## 2026 08 07, evaluation harness

Added the evaluation module: per pair localization against ground truth, metrics overall and split by style and placement, tolerance curve, plots and success and failure montages with the correlation response, written to a timestamped folder under experiments per run.
