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

## Conventions

The evaluation treats an algorithm output (x, y) as matching pixel convention above. Errors are reported in search image pixels; one search pixel corresponds to 10 nm nominal.
