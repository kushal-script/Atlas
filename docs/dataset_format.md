# Dataset format

A generated dataset folder contains:

```
pair_0000/
    reference.png    1000x1000, 8 bit grayscale, 1 nm per pixel
    search.png       1000x1000, 8 bit grayscale, nominal 10 nm per pixel
    meta.json        full parameter record and ground truth
pair_0001/
...
previews/            search images with the truth region outlined (only with --previews)
ground_truth.csv     one row per pair with truth coordinates and key parameters
dataset_meta.json    generator version, style, count, master seed
```

## meta.json fields

* `ground_truth.x`, `ground_truth.y`: center of the reference pattern in search image pixels. x is the column from the left, y the row from the top, origin at the center of the top left pixel, sub pixel float values.
* `gt_corners_xy`: the four reference image corners mapped into search image pixels, useful for drawing the matched region.
* `relative_rotation_deg`: rotation of the search capture relative to the reference capture.
* `search_scale_error`: relative pixel size error of the search capture against the nominal 10 nm.
* `placement`: sampling strategy for the reference site, `uniform`, `deep_array` or `near_boundary`.
* `layout`: every sampled structural parameter of the die layout.
* `se_model`: sampled secondary electron yields and detector asymmetry.
* `reference_capture`, `search_capture`: pose, beam, dose, noise, scan and charging parameters of each capture.

## ground_truth.csv columns for a Phase 2 suite

The scoring, recording and fitting tools read one row per pair with these
columns, which `scripts/generate_phase2_suite.py` writes and which a
laboratory suite must supply to use the refit workflow in
`docs/lab_guide.md`:

| column | meaning |
| --- | --- |
| `pair_id` | the identifier the predictions row must carry |
| `set` | `A_nominal`, `B_degraded`, `C_absent` or `D_optical`; present pairs must be A or B, since localization credit is weighted by set |
| `severity` | integer degradation rung, 0 for clean |
| `reference_path`, `search_path` | paths relative to the csv |
| `style` | `dram` or `finfet` |
| `modality` | `sem` or `optical`; only `sem` rows enter the grayscale scoring |
| `found` | 1 when the reference is present in the search, 0 when absent |
| `gt_x`, `gt_y` | match centre in search pixels, blank for absent pairs |
| `gt_zoom` | magnification ratio, nominally 8 to 12 |
| `gt_rotation_deg` | relative rotation in the reported convention |
| `seed` | the pair's generator seed |

## Conventions

The evaluation treats an algorithm output (x, y) as matching pixel convention above. Errors are reported in search image pixels; one search pixel corresponds to 10 nm nominal.
