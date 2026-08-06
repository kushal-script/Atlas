# Drift Sense

Navigation error recovery for wafer inspection tools. The repository contains two deliverables built for the Applied Materials problem statement:

1. A synthetic dataset generator that produces physically grounded SEM style image pairs of DRAM or FinFET die layouts, with exact ground truth.
2. A localization algorithm that finds the reference pattern inside the wide search image and reports the center coordinates.

## Problem

A reference image (1000x1000 px, 1 nm per pixel, 1 um field of view) shows a die site at high magnification. A search image (1000x1000 px, 10 nm per pixel, 10 um field of view) shows the surrounding region at 10x lower magnification, with more noise, possible rotation and slight scale error. The task is to find where the reference pattern sits inside the search image and print its center x y in search image pixels. When several regions match equally well, the one closest to the search image center is returned.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python 3.11 or newer. All commands below assume the repository root as working directory.

## Generate a dataset

```
.venv/bin/python scripts/generate_dataset.py --style mixed --num 40 --out data/train40 --seed 1 --previews
```

Arguments: `--style` is `dram`, `finfet` or `mixed`, `--num` is the number of pairs, `--out` the output folder, `--seed` the master seed, `--previews` also writes search images with the ground truth region drawn on top. Each pair folder contains `reference.png`, `search.png` and `meta.json` with the full parameter record and ground truth. A `ground_truth.csv` manifest summarises the dataset.

## Localize

```
.venv/bin/python scripts/localize.py path/to/reference.png path/to/search.png
```

Prints one line, `x y`, the center of the matched region in search image pixels with sub pixel precision. `x` is the column measured from the left, `y` is the row measured from the top, origin at the center of the top left pixel. Add `--json` for full diagnostics including the matched rotation, scale, score and runtime.

## Evaluate

```
.venv/bin/python scripts/evaluate.py --dataset data/train40 --name baseline
```

Runs the localizer over every pair, compares against ground truth and writes metrics, a results table and plots into a timestamped folder under `experiments/`.

## Repository layout

```
src/drift_sense/
    params.py            parameter definitions with literature provenance
    geometry/            procedural DRAM and FinFET layout builders
    imaging/sem.py       SEM image formation model
    generator.py         pair generation and ground truth bookkeeping
    localize.py          matching pipeline
    evaluate.py          metrics and plots
scripts/                 command line entry points
docs/                    method documentation and citations
experiments/             evaluation runs, one timestamped folder each
```

## Method in one paragraph each

Generator. A large continuous die layout is built once per pair as a material map and a height map on a 2 nm grid, using published pitches and dimensions for either a DRAM cell array (word lines, bit lines, storage node contacts, array mats separated by sense amplifier and driver stripes) or FinFET logic (fin grid, gate grid, standard cell rows, diffusion breaks, contacts, vias and one perfectly regular SRAM block). Secondary electron emission is modelled as material yield times the secant of the local surface tilt, which produces the bright feature edges characteristic of SEM, plus a detector side asymmetry. Each capture then samples this common specimen through its own pose (center, rotation, pixel size), applies beam blur, per line scan drift, jitter and vibration, dielectric charging, Poisson shot noise set by the electron dose and Gaussian read noise, and quantizes to 8 bit. The two captures use independent random generators, and the search capture always receives a far lower dose than the reference, so it is always noisier. Ground truth is computed by mapping the reference center through both capture transforms, including the scan offsets.

Localizer. The reference is blurred to the search optics resolution and resampled onto the search pixel grid by a single affine transform per rotation and scale hypothesis, reproducing the 10x zoom relationship including its aliasing. Normalized cross correlation against the preprocessed search image scores each hypothesis on a coarse rotation and scale grid, the best hypothesis is refined on a shrinking grid, and candidate peaks within a small tolerance of the best score are collected. A single candidate is returned directly with sub pixel parabolic refinement. When the correlation surface is degenerate (several near equal peaks inside a periodic array), a residual disambiguation stage applies cell to cell reference subtraction across the candidate windows and matches each window's deviation field against the template's, which identifies the true site whenever the window carries defect signatures; if no candidate wins decisively the site is genuinely ambiguous and the mandated tie break returns the candidate closest to the search image center. The diagnostics expose which of the three regimes produced each answer.

## Citations

Every noise model, augmentation and structural parameter is justified against public literature in [docs/citations.md](docs/citations.md).
