# What a degraded reference would cost, measured rather than assumed

Every degradation in this generator scales the search capture alone. That follows the
addendum, which describes the reference as a clean crop and puts the corruption on the wide
image, and it matches the stated fact that the search image carries more noise than the
reference on the real test data. It is nonetheless a premise about the organisers' pipeline
rather than something we can check, and until now nothing had measured what a blind set that
corrupted both captures would cost.

`generate_pair` gained a `degrade_reference` fraction for this. At 0.0, which is the
default, the reference capture is `cfg.reference` itself and the code path, the parameters
and the random stream are untouched; at 1.0 the reference carries the same severity ladder
factors the search does, applied to its own dose and beam spot ranges. Polygon scaling is
deliberately not applied to the reference, because it models process drift between two
visits rather than a property of a single capture.

## The default is provably unchanged

Generation is byte identical with the knob at its default, verified over eight
configurations, both architectures crossed with nominal, severity 2, severity 4 and absent.
The hashes of the raw reference and search arrays and the ground truth coordinates match the
pre change generator exactly.

A first attempt at this check was wrong and is recorded so the mistake is not repeated. Six
pairs were regenerated at the committed seed and ten of twelve images differed, which looked
alarming. The cause was the test: `generate_phase2_suite.py` distributes set membership by
shuffling a list built from the requested count, so pair_0000 of a six pair run is a
different set, style and severity from pair_0000 of a hundred and twenty pair run. The two
were never comparable. The valid check calls `generate_pair` with identical arguments
against the previous file.

## The knob does what it claims

At severity 4, with the fraction swept, the noise floor of the reference rises while the
search is untouched:

| fraction | reference noise | search noise | reference dose factor |
| --- | --- | --- | --- |
| 0.00 | 3.07 | 40.96 | 1.000, untouched |
| 0.25 | 4.25 | 40.96 | 0.787 |
| 0.50 | 6.03 | 40.96 | 0.575 |
| 1.00 | 13.10 | 40.96 | 0.150 |

The search noise is constant across the sweep, which is the check that the knob reaches only
the reference. Even fully degraded the reference stays the cleaner of the two captures,
which is right: it starts from between six and nine times the electron dose.

## The result

Two suites of 120 pairs from the same master seed, so composition, styles, severities and
poses are identical and the only difference is whether the reference was corrupted.

| reference | set A | set B | weighted | localization | median runtime |
| --- | --- | --- | --- | --- | --- |
| clean, as shipped | 0.690 | 0.457 | 0.562 | 22.49 of 40 | 3.88 s |
| fully degraded | 0.690 | 0.452 | 0.560 | 22.38 of 40 | 4.30 s |

The cost of corrupting the reference as hard as the search is 0.11 points of 40. Set A is
identical by construction, since a nominal pair has no severity for the fraction to scale.
Set B moves by 0.005.

This is a small risk correctly sized rather than a risk removed. The pipeline matches on
normalized cross correlation, which is invariant to the gain and offset that most of the
degradation acts through, and the reference remains far the cleaner capture even at the
extreme, so the evidence that decides a match is largely intact. The premise that the
organisers keep the reference clean is now one the submission does not depend on.
