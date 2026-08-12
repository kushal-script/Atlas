# Failure analysis

Every claim here is traceable to a run under `experiments/`. Numbers are quoted with the dataset they were measured on, because accuracy on this task is meaningless without saying which generator produced the data.

## The four measurement domains

The localizer is measured on four datasets built by generators that share no image formation code. This matters: a localizer tuned on one generator will silently learn that generator's physics, and the hidden test set comes from a different one.

| Dataset | Generator | What it tests |
| --- | --- | --- |
| `train40_v2` | primary physics generator | modern node dimensions, secant law edge brightening, dose driven noise, scan artifacts |
| `amat40` | faithful reproduction of the reference starter pipeline | coarse structure presets, area averaged decimation, four documented noise tiers |
| `spec40` | independent reading of the organiser specification | 9 to 1 through 11 to 1 magnification, rotation, the full published degradation list |
| `stress30` | adversarial generator | painted edges, plain Gaussian noise, harsher pose ranges |

## Root cause one, the centre tie break overriding a decisive maximum

This was the largest single defect found, and it was found by a test rather than by an accuracy number.

The problem statement requires that when several valid matches exist, the one closest to the search image centre is returned. The implementation defined "several matches" as every local maximum within 0.015 of the best score. On noise free, exactly decimated image pairs whose reference window straddles unique boundaries, so that the correct answer is unique by construction, that window admitted **68 to 81 candidates**, and the tie break then discarded a correlation maximum that was exactly at the true location. Measured errors of 368 px and 102 px on cases whose truth was the global maximum with a score gap of 0.0000.

The reason this is a defect and not a faithful reading of the rule: the reference pipeline samples the reference crop origin uniformly across the fine canvas, so the truth is uniformly distributed in the search frame and carries no bias toward the centre. Proximity to the centre therefore provides no information about correctness, and applying it to candidates that are measurably worse can only lose accuracy. The rule is meaningful only among matches that are genuinely indistinguishable.

The fix restricts the equal match set to numerical ties, so the tie break fires only when the correlation genuinely cannot separate candidates, while the wider noise scaled pool continues to feed the residual disambiguation stage. `experiments/*_tolerance_and_template_ablation` records the sweep that selected the threshold.

## Root cause two, template formation not matching search formation

The reference pipeline forms the search image by blurring the fine canvas and decimating it with area averaging, so each search pixel integrates a 10 by 10 block of specimen. The template was formed by bilinear point sampling at 10 times decimation, which reads one pixel in a hundred and retains aliasing the search image does not contain. The fix folds the decimation box filter into the template blur in quadrature with the beam spot, so the template carries matched anti aliasing. This is the same class of error as matching a sharp template against a soft image: the correlation peak survives but loses margin, and lost margin is exactly what turns an identifiable site into a near tie.

## Root cause three, an over wide pose search

Widening the magnification search to the stated 9 to 1 through 11 to 1 range was necessary, because the previous plus or minus 4 percent would have failed outright on any pair beyond 10.4 to 1. But a wider grid also gives many more chances for a wrong pose to score highly at a wrong location. On a proxy pair whose true magnification was exactly 10 to 1, the search selected a scale of 0.885 and locked onto a site 273 px away.

The fix evaluates the nominal pose first at full resolution and requires an off nominal pose to beat it by a margin before it is accepted. Coverage is retained for genuinely rotated or mis scaled inputs, while the common exact decimation case is both faster and less likely to be captured by a spurious hypothesis. The pose actually used is reported per pair as `pose_source`.

## Root cause four, impulse noise

Salt and pepper noise up to a few percent of pixels appears in the published degradation list. Impulse pixels are unbounded outliers, so they dominate the sums inside normalized cross correlation, and Gaussian denoising spreads them rather than removing them. An adaptive 3 by 3 median that fires only when the impulse fraction is detectable recovered one affected pair from 765 px to 0.09 px.

## The irreducible failure mode

The remaining failures are concentrated in defect free periodic interiors, and they are not a deficiency of the algorithm.

If two candidate windows inside a periodic array are identical to within the noise floor, then no function of the two images can prefer one over the other, because the two inputs are the same. Localization inside such a region is not a hard problem, it is an ill posed one. The information that disambiguates has to come from aperiodic content: array boundaries, termination structures, missing or oversized contacts, line edge roughness that happens to be locally distinctive.

An oracle experiment quantified this. Supplying the true scale and rotation from the generator metadata, so that the pose search cannot be blamed, and then measuring the correlation at the true location against the global maximum, the failures on the specification proxy split into three classes:

| Class | Share | Diagnosis |
| --- | --- | --- |
| Search failure | 2 of 10 | the oracle pose lands within 2 px, so the pose search missed a recoverable answer |
| Appearance corruption | 3 of 10 | correlation at the truth is 0.21 to 0.54 even with the oracle pose, all in charging streak or extreme contrast variants |
| Genuine near tie | 5 of 10 | correlation at the truth is within 0.02 to 0.05 of a wrong lattice peak |

Only the first class is straightforwardly fixable. The third class is the ill posed regime, and the honest engineering response is not to guess better but to report the situation, which the localizer does: every answer carries a regime label of `unique_peak`, `residual_identified` or `tie_break_convention`. On a real tool that distinction is the difference between trusting a measurement and re acquiring at lower magnification.

## What the residual disambiguation stage buys, and where it stops

Where the window does contain distinguishing content, cell to cell reference subtraction recovers it. The shared periodic content is estimated by a pixelwise median over aligned candidate windows and projected out of the template along with its sub pixel shift terms; the remaining deviation field is scored densely at every position in closed form; a robust z score decides whether one candidate stands out.

On the primary dataset this converts lattice mislocks into sub pixel answers and raised the pass rate within 1 px from 82.5 to 87.5 percent. On the deliberately anchor free interiors it correctly abstains. The stage cannot manufacture information, so its ceiling is set by identifiability, not by tuning.

## A recorded negative result

Because the specification permits a gamma range of 0.4 to 2.5 and normalized cross correlation is invariant to affine intensity changes but not to gamma, tone normalization was an obvious candidate fix. Both global histogram equalization and adaptive local equalization were implemented and measured on all three datasets available at the time. Both were neutral to harmful: on the stress dataset the pass rate within 5 px fell from 60.0 percent with no tone normalization to 56.7 percent with equalization and 46.7 percent with adaptive equalization, and neither changed the primary dataset.

The explanation came from reading the reference pipeline: gamma, speckle and impulse noise are applied identically to both captures, while only shear, jitter, vignetting and radial distortion are applied asymmetrically. A monotonic tone change shared by both images is largely absorbed by a locally normalized correlation, so there was nothing for tone normalization to fix, and its cost in fine structure was real. The option remains in the configuration, switched off, with this measurement as the reason.

## A recorded negative result on learning

A small convolutional re-ranker was trained to replace the statistical decision in the residual stage, using labels that are free because ground truth is known. It matched the classical rule in distribution and lost to it out of distribution, falling from 43.3 to 30.0 percent within 1 px on the held out adversarial generator. A network trained on one generator's physics did not transfer to an independent generator, and the hidden test set is by definition an independent generator. The physics grounded decision therefore remains the default and the re-ranker ships behind a flag, with the full training and comparison record under `experiments/`.

The claim this evidence supports is narrow and worth stating precisely: this re-ranker, trained on 88 degenerate pairs from one generator, did not transfer. It is not evidence that learned re-ranking cannot work here. Establishing that would require substantially more training data and domain randomisation over the generator parameters, so that the network is given a fair opportunity to fail.
