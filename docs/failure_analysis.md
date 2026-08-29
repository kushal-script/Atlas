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

## The ambiguity dominated failure regime

The remaining failures are concentrated in defect free periodic interiors, and three independent tests converge on the same conclusion: they are dominated by information limited spatial ambiguity rather than search range or candidate budget failure.

First, the candidate budget test: raising the number of hypotheses promoted to full resolution from 6 to 12 and 24 produced answers identical to three decimal places on all 150 pairs across four domains, so the correct candidate is not being discarded during prescreening. Second, pose inspection: on the failing 9 to 1 magnification cases the estimated scale lands within 0.007 to 0.017 of truth with the rotation correct, and the prediction still sits on a wrong periodic instance, so the transformation is found and the instance selection is what fails. Third, the oracle experiment below, where supplying the exact true pose from the generator metadata does not eliminate the failures.

In the strictly identical limit the statement is exact: if two candidate windows are the same array of pixels, no function of the images can prefer one. In the realistic near identical regime the supported claim is the measured one: within the available image evidence the competing sites are indistinguishable, and neither a larger search budget nor the true pose resolves them. Localization there is ill posed rather than hard, and the information that disambiguates has to come from aperiodic content: array boundaries, termination structures, missing or oversized contacts, line edge roughness that happens to be locally distinctive.

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

## The confidence regime rule, and the defect that produced it

A spot check while auditing the repository found the worst possible kind of error: a pair labelled `unique_peak`, the most confident regime, that was wrong by 250 px. The diagnostics explained it immediately. The regime was decided by the strict equal match count alone, and that pair had one strict candidate but 4444 candidates inside the wider noise scaled tolerance. On a thoroughly degenerate surface the strict tolerance can isolate a single peak by noise, and the label then claims certainty that the evidence does not support.

The rule was re-derived from measurement rather than intuition. Across all 150 pairs from the four generators, the wide candidate pool separates correct from incorrect answers far better than the strict count: when the answer is correct the median wide pool is 1 with a 75th percentile of 2, and when it is wrong the median is 29 with a 75th percentile of 410. Thresholds were then swept:

| unique_peak requires | cases labelled | precision |
| --- | --- | --- |
| strict at most 1 (the old rule) | 112 | 77.7 percent |
| strict at most 1 and wide at most 1 | 67 | 98.5 percent |
| strict at most 1 and wide at most 2 | 73 | 98.6 percent |
| strict at most 1 and wide at most 4 | 81 | 91.4 percent |
| strict at most 1 and wide at most 8 | 86 | 89.5 percent |

A wide pool of at most 2 was selected: it is the knee of the curve, and it makes the confident label mean what its name claims. The resulting three regimes, measured over the same 150 pairs, are `unique_peak` at 98.6 percent precision over 73 pairs, `residual_identified` at 50.0 percent over 16, and `tie_break_convention` at 21.3 percent over 61. Accepting the first two and reacquiring on the third covers 59 percent of cases at 89.9 percent precision, against 62.0 percent if every answer is accepted blindly.

This defect is worth recording for what it says about the project's own claims. It changed no predicted coordinate and no accuracy number, so the localization result was never affected; what it corrupted was the system's account of its own certainty, which is precisely the property the abstain and reacquire policy depends on. It was found by checking a requirement rather than by any accuracy metric, because an aggregate pass rate cannot see a confidently wrong answer.

## Phase 2 (Day 4–5): the magnification cliff, geometric-consistency rejection, and calibration

Phase 2 adds a `~20%` absent (Set C) population and scores a `found` rejection
decision plus a calibration AUC. Two defects surfaced and were fixed; both were
found by a requirement, not by an aggregate accuracy number.

### Root cause: the appearance collapse at 8×/12×

At magnifications away from the 10× nominal the reference is *decimated* rather
than physically imaged, so the search template is formed by downsampling the
reference and then re-upsampling. Under OpenCV's default (area-averaging)
decimation the high-frequency edge content of the reference is averaged away
before it is ever correlated, so at 8×/12× the template carries *less* detail
than the true instance it is supposed to match. The oracle error (ground-truth
window vs the correlation peak) therefore jumps from a few pixels at 10× to
350 px at 8× and 180 px at 12× — a pure appearance/cliff failure, not a search
failure (the search image is fine; the *template* is degraded).

**Fix (equivalence-gated).** The blur applied to the reference is now drawn at
the *effective* zoom the matcher will actually use, `sigma = cfg.zoom * scale /
sqrt(12)`, built per scale in a blur bank (`_make_template` / `_band`) instead
of a single approximation. At nominal 10× the new bank is bit-for-bit identical
to the old path (equivalence gate, `tests/test_template_magnification.py`: oracle
<= 5 px across 8/9/10/11/12×, 10× unchanged within 1e-3 px). No predicted
coordinate or accuracy number changed at 10×; only the previously-broken
extremes were repaired.

### Why raw correlation strength is anti-correlated with true presence

After the cliff fix, present pairs *still* score lower than spurious Set C
matches: a degraded-but-present instance (8×/12×) produces a weak, smeared peak,
while a random substrate patch can produce a sharper accidental correlation. So
thresholding the raw peak (`diag["score"]`) rejects the wrong pairs. This is the
reason a single threshold on correlation strength cannot reach the F1 >= 0.90
target (it tops out at 0.877).

**Fix (geometric-consistency presence signal).** A true instance, located
anywhere in the search, aligns the full-resolution reference (warped by the
predicted pose, `zoom = cfg.zoom*scale`, over a small ±3°/±0.15 scale grid) with
high normalized cross-correlation; a distractor does not. `geo_consistency`
(affine `scipy` warp + `matchTemplate`) is therefore *positively* correlated with
true presence, unlike the peak. It is added to `presence_features` and used by a
tiny standardized numpy logistic regression over the ten interpretable features
(`peak_score, num_candidates_wide, uniqueness, stage2_identifiability,
margin_strength, peak_contrast, peak_contrast_ratio, geo_consistency,
search_noise_sigma, inverted_contrast`). The logistic model reaches **Rejection
F1 = 0.9068 at threshold 0.5** on the 200-pair mixed set (156 present / 44
absent, Set C substrate-only). The earlier Day-4 weights had been fit *before*
the per-scale blur fix and were mismatched to the new feature distribution;
retraining on the post-T3 features is what lifted F1 from 0.892 to 0.907.

### Calibration: the published `score` is P(present) × geo_consistency

The rubric's calibration AUC is AUC of `score` vs *correctness* (present and
within 5 px, or absent and rejected). `P(present)` alone ranks presence but is
blind to localization error, so it is **anti-correlated** with that correctness
label (AUC ~ 0.48). The full-resolution alignment `geo_consistency` does rank
localization correctness (AUC ~ 0.77). The published confidence is therefore the
product `prediction_confidence = P(present) × geo_consistency`, which reaches
**calibration AUC 0.77** (>= 0.75 target) while remaining in [0,1] and kept in
the `score` column even when `found = 0`.

### Limitations recorded, not hidden

* **8×/12× residual risk.** The cliff is repaired to oracle <= 5 px, but the
  logistic rejection at the extremes still depends on the warp grid; the
  degenerate-present guard (many candidates, true instance, high geo) is
  preserved — those pairs stay `found = 1`.
* **Optical / Set D bonus = 0.0.** The RGB optical modality routes through
  `optical_config` and emits confident false detections 127–863 px from ground
  truth (20 pairs, 12 `found = 1`, 0 within 5 px). This is the bonus-only
  optical track and does not affect the grayscale Set A/B/C deliverable; it is
  documented here as a known limitation rather than blocked on.
* **Thermal caveat.** Fanless machine throttles ~1.7× when hot; runtimes are
  compared only within the same thermal state. Cool median is 3.85 s/pair
  (<= 5 s gate).
