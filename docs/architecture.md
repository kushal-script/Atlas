# Architecture

## Design principle

The generator does not paint SEM artifacts onto a drawing. It builds a specimen (a material map and a height map in nanometer units) and then simulates two independent captures of that same specimen through a physical signal chain. Edge brightening, contrast between regions, aliasing in the zoomed out image, and the noise difference between the two images all emerge from the model instead of being applied as filters. This matters for the localization task: the search image is a genuinely different observation of the same object, exactly as on a real tool.

## Coordinate frames

Specimen frame: nanometers, u rightward, v downward. The canvas covers 11.2 um at 2 nm per pixel so that a rotated 10 um search field plus margins always fits.

Capture frame: image pixels, x is the column, y is the row, origin at the center of the top left pixel. A capture pose is a center (u, v), a rotation theta and a pixel size. Pixel (r, c) samples the specimen at the affine image of (r + dy(r), c + dx(r)), where dx and dy are the per line scan offsets from drift, jitter and vibration.

The same two mapping functions, `capture_to_specimen` and `specimen_to_capture`, are used both for rendering and for ground truth, so the recorded truth cannot disagree with the rendered images. Ground truth is the reference capture center mapped to specimen coordinates and back through the search capture transform, scan offsets included, solved with a short fixed point iteration.

## Generation pipeline per pair

1. Independent random generators are spawned from the pair seed for layout, canvas physics, poses, reference capture and search capture. The two captures therefore have fully independent noise, as the problem statement requires.
2. The layout builder rasterises the chosen style onto the canvas. Structural randomisation per pair: pitches, duty cycles, phases, block sizes, line edge roughness amplitude and correlation length, contact size variation, missing contact defects, cell width sequences, contact and via presence.
3. The secondary electron canvas is computed once: material yield lookup, height map smoothing for finite sidewall slope, secant of surface tilt for edge brightening with a tilt clamp, and a directional detector asymmetry term.
4. Each capture renders through: supersampled affine sampling of the SE canvas (5x for the search capture so the point sampling to the 10 nm grid keeps its physical aliasing), anisotropic Gaussian beam blur, decimation with per line scan offsets, dielectric charging as a smooth multiplicative field over oxide regions, Poisson shot noise scaled by electron dose, Gaussian read noise, percentile tone mapping with random headroom and 8 bit quantization.
5. The search capture pose carries a random rotation (sigma 0.8 deg, clamped to 2 deg), a pixel size error up to 1.8 percent and a center offset. The reference capture carries a small residual rotation. Reference placement follows a strategy mix: uniform, deep inside a periodic array (the deliberately hard case) or near an aperiodic boundary.

## Localizer pipeline

1. The reference is blurred with a Gaussian matched to typical search optics resolution, then band passed. The search image is lightly denoised and band passed with the same physical cutoff.
2. For each rotation and scale hypothesis, one affine transform resamples the blurred reference onto the search pixel grid (10x zoom times hypothesis scale, rotated), point sampling included so template aliasing matches search aliasing.
3. Normalized cross correlation (OpenCV, FFT backed) scores each hypothesis by its response maximum. A coarse grid over rotation (plus minus 3 deg) and scale (plus minus 3 percent) is refined twice around the best hypothesis with halved steps.
4. On the best response map, all local maxima within a small tolerance of the global maximum are candidates. A single candidate is returned directly with parabolic sub pixel refinement.
5. When several candidates exist, the correlation surface is degenerate and a residual disambiguation stage runs. It is cell to cell reference subtraction, the standard array inspection technique, applied to localization: the median across all candidate windows estimates the shared periodic content, each window minus the median is its deviation field (missing contacts, size outliers, roughness lumps), and the template's deviation field is correlated against each candidate's. Candidates are first clustered by spatial proximity (half a template) and only cluster representatives enter the comparison, because windows a few pixels apart overlap almost completely and would violate the independence the median needs; the stack is kept at odd size so the pixelwise median is robust to the one outlier that carries the defect. If one representative wins by a decisive margin it is returned, because the deviation field is the only physically identifying information a periodic array offers. If no candidate is decisive the site is genuinely ambiguous at the available noise level, and the problem statement tie break applies: the candidate closest to the search image center.

The stage 2 margin is exposed in the diagnostics, so every answer carries its own confidence: one candidate means unambiguous, a decisive residual margin means identified by defects, an indecisive margin means the output is the tie break convention rather than an evidence based location.

## Failure modes by construction

The generator can place the reference deep inside a uniform array (placement `deep_array`). There the correlation surface has a lattice of near equal peaks and the tie break rule decides, which is exactly the failure regime the problem statement asks to be demonstrated and explained. The evaluation harness separates metrics by placement so this regime is measured rather than hidden.
