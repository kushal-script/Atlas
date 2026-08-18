# Compute backend port

Two operations dominate localization runtime, so only those two were abstracted behind a backend: building the reference blur bank and normalized cross correlation. Profiling one 1000 by 1000 pair put 52 percent of the time in the blur and 45 percent in the correlation, and everything else stays plain numpy.

## Result

| Backend | physics40 | amat40 | spec40 | stress30 |
| --- | --- | --- | --- | --- |
| baseline, scipy blur and per call correlation | 3.65 s | 3.57 s | 3.53 s | 3.63 s |
| cpu backend | 2.03 s | 2.06 s | 2.10 s | 2.35 s |
| accelerator backend | 0.78 s | 0.83 s | 0.82 s | 0.81 s |

Accuracy is unchanged on all four domains: 90.0, 20.0, 32.5 and 43.3 percent within 1 px and medians of 0.13, 2.85, 18.83 and 4.13 px, matching the frozen record exactly. The worst coordinate disagreement between the two backends over all 150 pairs is 0.0014 px, against a 1 px pass threshold, and no pair changes its error, its confidence regime or its pass or fail status.

The accelerator measured here is Apple MPS, because that is the accelerator present on the development machine. The same code path selects CUDA with `--device cuda`; no CUDA hardware was available, so no CUDA number is claimed.

## The part worth recording: the naive port was eight times slower

Offloading each operation as it was reached made the localizer far slower than the CPU it was meant to beat.

| Stage | Median per pair | Why |
| --- | --- | --- |
| per call offload | 17.48 s | roughly 200 correlations per pair, each copying a fresh 4 MB search image to the device and the full response map back |
| batched prescreen | 12.13 s | the 179 pose hypotheses grouped by template size into one batched call each, which removed most launches but not the full resolution cost |
| FFT correlator, cached spectra | 0.52 s | the correlation evaluated through the Fourier transform, with the search spectrum and the per window sums computed once and held on the device |

The decisive finding was in the third row. A direct convolution with a 90 by 90 kernel over a 1000 by 1000 image measured 16.5 s for 28 calls on the accelerator against 0.67 s on the CPU, a factor of twenty five the wrong way. OpenCV transforms large templates rather than convolving them directly, and the accelerator path had to do the same before the comparison was fair at all. This is the ordinary shape of the mistake: the workload was never compute bound, it was bound by transfers and by an unsuitable algorithm, and reaching for faster hardware first would have hidden that.

The CPU gain in the second row of the result table is independent of any accelerator. Replacing the scipy Gaussian with the OpenCV one, at a kernel radius pinned to scipy's four standard deviation truncation so the amount of blur cannot change, is worth 1.7 times on its own and needs no new dependency.

## Correctness

`scripts/verify_backends.py` asserts that every backend reproduces the reference path: the blur against scipy to 1e-2 grey levels, and the correlation against `cv2.TM_CCOEFF_NORMED` to 2e-4, including the location of the peak. Two defects were caught by it during this work. Torch's reflect padding drops the edge sample, which is `cv2.BORDER_REFLECT_101` rather than the `BORDER_REFLECT` that scipy and OpenCV use, and it changed the blur near the border by up to 11 grey levels until the reflection was built explicitly. The FFT correlator's valid region indexing was also verified against OpenCV rather than reasoned about.

## Reproducing

```
.venv/bin/python scripts/verify_backends.py
.venv/bin/python localize.py --batch data/physics40 --out results/predictions.csv
.venv/bin/python localize.py --batch data/physics40 --device auto --out results/predictions_gpu.csv
```
