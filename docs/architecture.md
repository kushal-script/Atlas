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

1. The reference is blurred to a bank of plausible search optics resolutions (4, 9, 16 and 25 nm sigma), each band passed at the same physical cutoff as the denoised, band passed search image. The bank exists because the search blur on unseen test data is unknown and correlation quality depends strongly on matching it.
2. For each blur, rotation and scale hypothesis, one affine transform resamples the blurred reference onto the search pixel grid (10x zoom times hypothesis scale, rotated), point sampling included so template aliasing matches search aliasing.
3. The full hypothesis grid (rotation to plus minus 6 deg, scale to plus minus 4 percent, four blurs, 180 combinations) is screened at half resolution first, which costs a few milliseconds per hypothesis. The top hypotheses are rescored at full resolution and the best is refined twice on a shrinking rotation and scale grid. Contrast polarity is detected from the signed correlation extrema during the prescreen, so inverted tone conventions still match.
4. On the best response map, all local maxima within a strict tolerance (0.015) of the global maximum form the equal match set, and a wider noise scaled tolerance defines the rescue pool for stage two. A single equal match is returned directly with parabolic sub pixel refinement.
5. When the surface is degenerate, the residual disambiguation stage runs: cell to cell reference subtraction, the standard array inspection technique, applied to localization. The pixelwise median over the best aligned candidate windows estimates the shared periodic content. The template's deviation field is what remains after projecting out the median and its sub pixel shift terms (a least squares fit of the median and its two gradient images), which removes the systematic lattice leakage that would otherwise correlate with every aligned window. The normalized correlation between this deviation field and every window's deviation field is then computed densely in closed form (two raw correlations plus integral image sums), so every candidate position is scored with no selection bias regardless of how many near ties exist. The decision is a robust z score of the best candidate against the candidate population, which self calibrates to pool size and noise; small pools use a fixed margin instead. A decisive winner is returned because the deviation field is the only physically identifying information a periodic array offers; otherwise the problem statement tie break applies among the strict equal match set: the candidate closest to the search image center.

The diagnostics expose which regime produced each answer (single candidate, decisive residual, or tie break convention) plus the residual z score, so every output carries its own confidence.

## Failure modes by construction

The generator can place the reference deep inside a uniform array (placement `deep_array`). There the correlation surface has a lattice of near equal peaks and the tie break rule decides, which is exactly the failure regime the problem statement asks to be demonstrated and explained. The evaluation harness separates metrics by placement so this regime is measured rather than hidden.
