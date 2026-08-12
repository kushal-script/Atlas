# Drift Sense

Navigation error recovery for wafer inspection tools. Given a high magnification reference image of a die site and a wider, noisier search image of the surrounding region, find where the reference pattern sits inside the search image and return its centre coordinates.

The repository contains two deliverables plus the evidence behind them: a synthetic dataset generator grounded in published semiconductor structure and SEM imaging physics, and a localization algorithm with a documented failure analysis.

## Problem and conventions

| Item | Value |
| --- | --- |
| Reference image | 1000 x 1000 grayscale, 1 nm per pixel, 1 um field of view |
| Search image | 1000 x 1000 grayscale, nominally 10 nm per pixel, 10 um field of view |
| Scale relationship | nominally 10 to 1, handled explicitly over 9 to 1 through 11 to 1 |
| Rotation | handled to plus or minus 6 degrees, refined to 0.09 degrees |
| Output | centre coordinates x y in search image pixels, sub pixel |
| Coordinate origin | centre of the top left pixel, x increases right, y increases down |
| Multiple matches | the match closest to the search image centre is returned |

The reference pattern occupies roughly 100 x 100 pixels inside the search image. The two images are independent physical captures, so their noise is independent and the search image is the noisier of the two.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python 3.11 or newer. Inference needs only numpy, scipy, OpenCV, Pillow and matplotlib; no deep learning framework is required. All commands below run from the repository root.

## Generate a dataset

```
.venv/bin/python generate_dataset.py --generator physics --style mixed --num 40 --out data/train40 --seed 1 --previews
```

Three generators are available, sharing no image formation code, so the localizer can be measured under domain shift instead of only against its own assumptions:

| `--generator` | What it produces |
| --- | --- |
| `physics` | primary generator, specimen material and height maps imaged twice through a secondary electron model; supports `--modality sem` and `--modality optical` |
| `spec` | independent reimplementation of the published organiser specification, coarse structure presets, 9 to 1 through 11 to 1 magnification, the full published degradation list |
| `amat_proxy` | faithful reproduction of the reference starter pipeline, 10 um fine canvas at 1 nm per pixel decimated by area averaging, four noise tiers and five acquisition variants |
| `stress` | adversarial generator with painted edges, plain Gaussian noise and area averaged downsampling |

Other flags: `--style` is `dram`, `finfet` or `mixed`; `--num` the pair count; `--seed` the master seed; `--previews` also writes search images with the truth region drawn on.

Each pair folder holds `reference.png`, `search.png` and `meta.json` recording the seed, every structure parameter, every transformation and noise setting, and the exact ground truth centre. `ground_truth.csv` is the dataset manifest with reference path, search path and truth coordinates per pair.

## Localize

Single pair, prints one line `x y`:

```
.venv/bin/python localize.py path/to/reference.png path/to/search.png
```

Full diagnostics including the confidence regime, matched pose and runtime:

```
.venv/bin/python localize.py path/to/reference.png path/to/search.png --json
```

Batch over a directory of pair folders, writing predictions and a runtime record:

```
.venv/bin/python localize.py --batch path/to/dataset --out results/predictions.csv
```

Batch over an explicit manifest with `reference_path` and `search_path` columns:

```
.venv/bin/python localize.py --manifest pairs.csv --out results/predictions.csv
```

Batch over a flat directory whose files pair by shared prefix:

```
.venv/bin/python localize.py --batch path/to/images --pattern flat --out results/predictions.csv
```

No source changes are needed for any of these forms. The predictions CSV carries pair id, both input paths, predicted x and y, peak score, confidence regime, candidate count, matched rotation and scale, and per pair runtime; a companion `.runtime.json` records hardware, Python version and the timing method.

RGB input is detected automatically and routed through an optical preset.

To use the localizer inside another harness, the drop in interface returns position and a calibrated confidence:

```python
from drift_sense.api import zncc_match
result = zncc_match(reference_img, search_img)
result["x"], result["y"], result["score"]
```

## Evaluate

Accuracy, timing, plots and success and failure montages against a generated dataset:

```
.venv/bin/python scripts/evaluate.py --dataset data/train40 --name baseline
```

Pass rates per noise tier with confidence ranked precision recall curves:

```
.venv/bin/python scripts/evaluate_tiers.py --dataset data/amat_tiers --name tier_report
```

Configuration ablation across every dataset at once:

```
.venv/bin/python scripts/compare_configs.py --datasets data/train40 data/stress30 --name ablation
```

Every run writes into a timestamped folder under `experiments/`, holding the configuration, a per pair results table, aggregate metrics and plots, so any number quoted anywhere in this repository is traceable to the run that produced it.

## Repository layout

```
generate_dataset.py        dataset generation entry point
localize.py                localization entry point, single pair and batch
configs/                   generator and matcher configurations
src/drift_sense/
    params.py              structure and imaging parameters with literature provenance
    geometry/              procedural DRAM and FinFET layout builders
    imaging/sem.py         secondary electron image formation
    imaging/optical.py     RGB brightfield image formation
    generator.py           pair generation and ground truth bookkeeping
    localize.py            matching pipeline
    reranker.py            optional learned decision, numpy inference
    api.py                 drop in matcher interface
    evaluate.py            metrics and plots
scripts/                   auxiliary generators, evaluation and training tools
models/                    exported re-ranker weights
notebooks/                 re-ranker training notebook
docs/                      method, citations, dataset format, development log
references/                bibliography
results/                   headline result tables and figures
experiments/               one timestamped folder per run
```

## Method

**Generator.** A specimen is built once per pair as a material map and a height map on a nanometre grid, using published pitches and dimensions for either a DRAM cell array (word lines, bit lines, storage node contacts, mats separated by sense amplifier and driver stripes) or FinFET logic (fin and gate grids, standard cell rows, diffusion breaks, contacts, vias, and one perfectly regular SRAM block). Secondary electron emission is modelled as material yield times the secant of the local surface tilt, so the bright feature edges characteristic of SEM emerge from surface physics rather than from an edge filter. Each capture then samples that one specimen through its own pose and applies beam blur with astigmatism, per line scan drift, jitter and vibration, dielectric charging, Poisson shot noise set by electron dose, Gaussian read noise, and an 8 bit tone map. The two captures draw from independent random generators and the search capture always receives a far lower dose. Ground truth is computed by mapping the reference centre through both capture transforms including the scan offsets, so the recorded truth cannot disagree with the rendered pixels.

**Localizer.** The reference is blurred to a bank of candidate search resolutions, each blur including the box filter equivalent of the magnification ratio so the template carries the same anti aliasing the search image does. The nominal pose is evaluated first at full resolution, because an exact decimation with no rotation is by far the most likely case; a wide grid over rotation and scale is screened at half resolution only when the nominal pose correlates weakly, and an off nominal pose must beat the nominal one by a margin before it is accepted, since a wide grid gives many chances for a wrong pose to win on noise. Impulse noise is removed by an adaptive median that fires only when outliers are detected, and contrast polarity is decided from the signed correlation extrema.

Among the resulting correlation peaks, a single dominant peak is returned directly with parabolic sub pixel refinement. When the surface is degenerate, a residual disambiguation stage runs: the shared periodic content is estimated by a pixelwise median over aligned candidate windows and projected out of the template together with its sub pixel shift terms, the remaining deviation field is scored densely in closed form at every position, and a robust z score decides whether one candidate is identifiably the true site through its defect signature. If none is, the site is genuinely ambiguous at the available noise level and the mandated tie break returns the equal match closest to the search image centre. Every answer therefore carries its regime: unique peak, identified by residual, or tie break convention.

## Results

See [results/README.md](results/README.md) for the headline tables and the failure analysis, and `experiments/` for the full record of every run.

## Documentation

* [docs/architecture.md](docs/architecture.md) coordinate frames, generator and localizer pipelines, failure modes by construction
* [docs/citations.md](docs/citations.md) every noise model, augmentation and structural parameter mapped to public literature
* [docs/dataset_format.md](docs/dataset_format.md) on disk format and metadata fields
* [docs/development_log.md](docs/development_log.md) chronological record of every major change and the measurement that motivated it
* [references/references.bib](references/references.bib) bibliography

## Assumptions and limitations

* The magnification is assumed to lie between 9 to 1 and 11 to 1 and the relative rotation within plus or minus 6 degrees. Outside those ranges the pose search will not find the correct hypothesis.
* Radial distortion is not part of the pose model. Over a 100 pixel footprint local radial distortion is close to affine and is largely absorbed by the scale search, but strong distortion is not corrected.
* Row jitter is modelled as an appearance change rather than inverted, so per row displacements much larger than a pixel degrade correlation.
* Localization inside a defect free periodic region is not solvable in principle: two positions whose neighbourhoods are identical to within the noise floor cannot be distinguished by any algorithm. The localizer reports this case rather than hiding it, and the mandated tie break decides the answer.
* No proprietary fab data is used anywhere. All structure parameters come from public literature, listed in the citations document.
