# Development log

Major changes in chronological order. Each entry corresponds to one commit.

## 2026 08 07, project start and dataset generator

Toolchain decision: pure Python with numpy, scipy, Pillow, OpenCV and matplotlib inside a project virtual environment. Simulation suites (process emulators, finite element tools) were considered and rejected: the deliverable must run from a plain clone with pip installed requirements, and the scored part of the generator is the SEM image formation model, which is fully expressible with array code.

Built the generator as specimen first, capture second (see architecture.md). Two layout styles implemented, DRAM and FinFET, both with per pair structural randomisation and deliberate aperiodic anchors (mat stripes for DRAM, logic cell randomness for FinFET) plus deliberately hard periodic regions (mat interiors, an SRAM block). First visual review led to one geometry correction: diffusion breaks initially erased about 43 percent of the fin field because every cell boundary carried a 1.5 gate pitch wide break; reduced to a 0.8 gate pitch break with wider cells, matching single diffusion break practice, after which the fin fabric reads like real top down CD SEM frames.

## 2026 08 07, localizer

Implemented the matched formation approach: instead of generic multi scale template matching, the reference is pushed through the same optical chain the search image went through (blur to search resolution, affine resample onto the search grid with point sampling so aliasing is reproduced), then normalized cross correlation with a coarse to fine rotation and scale search. Tie break per the problem statement: among near equal peaks, closest to the search image center. Smoke test on three pairs landed under 0.2 px error.

## 2026 08 07, evaluation harness

Added the evaluation module: per pair localization against ground truth, metrics overall and split by style and placement, tolerance curve, plots and success and failure montages with the correlation response, written to a timestamped folder under experiments per run.

## 2026 08 07, baseline results and two corrections

Baseline run on 40 mixed pairs: median error 0.153 px, every one of the 30 pairs whose window contains aperiodic content localized sub pixel with a single correlation candidate, and all 10 failures were lattice mislocks confined to deep periodic placements with 5 to 18 near equal candidates, the degeneracy the problem statement predicts. Two changes followed. First, a literature verification pass (all 23 references checked against publishers) surfaced that real DRAM local lines run 256 to 512 cells, so mats were enlarged from about 2 um to 4.5 to 9 um to match published organisation. Second, a residual disambiguation stage was added to the localizer: among near equal candidates, cell to cell reference subtraction recovers each window's deviation field and matches it against the template's, which identifies the true site whenever the window carries any defect signature, and otherwise falls back to the mandated center tie break. On the worst baseline failure (711 px mislock in a DRAM mat) the stage recovered the site to 0.07 px; on a defect free SRAM interior it correctly reported indecision, which is the physically honest outcome.
