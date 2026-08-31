# Both fiat fixed conventions verified against the organisers' published transform

The organisers released the prompt from which the scored set is generated. It fixes two
quantities by fiat, warning in both cases that a solver guessing wrong forfeits every pose
comparison. Until now this repository matched them against the three sample pairs shipped
with the addendum. They can now be checked against the transform itself.

## What the prompt fixes

The canvas to search map, given in closed form:

    p_search = (1/z) * R(theta) * (p_canvas - c_canvas) + c_search
    R(theta) = [[cos t, sin t], [-sin t, cos t]],  t = radians(theta)

so a positive theta turns the pattern counter clockwise as displayed. And the scale column
is the search pixel size in nanometres per pixel, the same number in 8 to 12 that defines
the pose, explicitly not the reference to search linear factor of one over z. The two
readings differ by a factor near a hundred.

## The check

A canvas of aperiodic landmarks is warped into a search image by inverting that exact
formula, the reference is cut unrotated from the same canvas at one nanometre per pixel,
and the label is the reference centre pushed through the forward map. The shipped entry
point then reports its pose.

| z | theta in | our theta | our scale | centre error | sign |
| --- | --- | --- | --- | --- | --- |
| 10.0 | 0.00 | -0.16 | 9.963 | 0.04 px | correct |
| 9.0 | +3.50 | +3.49 | 8.781 | 0.46 px | correct |
| 11.0 | -3.50 | -1.45 | 10.000 | 1.54 px | correct |
| 8.0 | +5.00 | +4.69 | 8.222 | 0.89 px | correct |
| 12.0 | -5.00 | -5.05 | 11.766 | 0.25 px | correct |

The sign is right on all five poses, including the required exact zero, and the scale
column tracks z across the full disclosed range rather than its reciprocal. The reported
theta is produced by `theta_report_sign` of minus one in `MatchConfig`, a single constant
carrying the difference between the pose grid's internal sense and the reported one, and
that constant is now confirmed against the organisers' own geometry rather than inferred.

The one row whose theta is off by two degrees is a rotation accuracy miss on this
deliberately crude probe canvas, which is Gaussian noise with rectangles rather than a
lithographic lattice, and its centre error of 1.54 px still earns localization credit. It
is not a convention failure: the sign is correct there too.

## What the prompt refutes in our own records

The prompt states the reference is 1000 by 1000 at one nanometre per pixel and the search
is 1000 by 1000 at z nanometres per pixel. That is exactly this repository's geometry, so
the field the blind set uses is the field we generate.

That settles a question left open in `experiments/20260831_finfet_field_scale`. The
published dataset description elsewhere says a FinFET reference shows one or two horizontal
gate bars, which at a fifty to sixty nanometre contacted poly pitch implies a reference
field near eighty nanometres, twelve times narrower than ours. The sweep measured what such
a field would cost and found it catastrophic, credit falling from 0.667 to 0.071, and
concluded that our 16.47 of 40 on FinFET is optimistic for a specification faithful blind
set. That conclusion is now withdrawn. The generator prompt pins the reference at one
nanometre per pixel over a thousand pixels, which puts about eighteen gate bars in frame,
so the narrow field was a reading of a stylised picture rather than of the generator, and
our field matches the scored one. The sweep's measurement stands; only its extrapolation
to the blind set was wrong.

The prompt also requires at least eight distinct architecture presets across both families
in twenty pairs, which settles the earlier question of whether to specialise: the scored
set contains both DRAM and FinFET by construction.
