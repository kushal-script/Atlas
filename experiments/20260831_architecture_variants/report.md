# Separate DRAM and FinFET pipelines, and what measuring them showed

Two self contained variants were built under `variants/`, each a full clone of the
pipeline with its package renamed, its generator restricted to one architecture, and a
matcher specialised for that architecture. The purpose was to find out whether the large
gap between the two architectures is something a specialised matcher can close.

It is not. Both matcher specialisations were measured and both are declined. The variants
are kept because the generators are now architecture faithful and because the measurement
that killed the specialisations is the most useful result here.

## The gap, measured through the scored entry point

| style | set A | set B | weighted | of 40 pts | present pairs |
| --- | --- | --- | --- | --- | --- |
| DRAM | 0.792 | 0.568 | 0.669 | 26.75 | 49 |
| FinFET | 0.556 | 0.294 | 0.412 | 16.47 | 35 |
| combined, as shipped | 0.690 | 0.457 | 0.562 | 22.49 | 84 |

Ten points of a hundred separate the two architectures, and the blind set's composition is
not disclosed, so this is the largest single uncertainty left in the submission.

## The gap is an information limit, not a search limit

The oracle probe supplies the true zoom and rotation from the generator's own record, which
removes the pose search from the problem entirely and asks only whether the correlation
prefers the true site.

| style | regime | n | true site wins | median impostor lead |
| --- | --- | --- | --- | --- |
| DRAM | nominal | 24 | 83 percent | 0.0000 |
| DRAM | degraded | 25 | 48 percent | 0.0104 |
| FinFET | nominal | 18 | 61 percent | 0.0046 |
| FinFET | degraded | 17 | 24 percent | 0.0328 |

FinFET loses to an impostor more often than DRAM even when the pose is handed to it, and by
a wider margin when it loses. No amount of searching recovers a site the evidence itself
ranks second, so the architecture gap cannot be closed in the matcher. The cause is visible
in the generator: `geometry/dram.py` drops a small random fraction of contacts and varies
the size of the rest, and those defects are the only aperiodic content in an otherwise
identical field, so they are the only evidence that can say which cell is the right one.
`geometry/finfet.py` has no equivalent. Every FinFET cell is exactly like its neighbours.

## Specialisation one, contact emphasis for DRAM, declined

A Laplacian of Gaussian matched to the contact radius, applied to both operands at full
resolution, on the reasoning that weighting the correlation towards the dot lattice makes
the missing contacts dominant. The sigma is derived from the generator's own contact radius
rather than tuned. Measured over all 49 present DRAM pairs, localization only:

| emphasis | set A | set B | weighted | of 40 | median runtime |
| --- | --- | --- | --- | --- | --- |
| off | 0.833 | 0.648 | 0.731 | 29.26 | 3.75 s |
| on | 0.850 | 0.560 | 0.691 | 27.62 | 3.79 s |

It does what it was designed to do on clean pairs, lifting nominal credit by 0.017, and
loses far more than that on degraded pairs, 0.088, for a weighted cost of 1.64 points of
40. The reason is that a Laplacian of Gaussian is a high pass filter and the degraded
captures are shot noise dominated at exactly the frequencies it emphasises, so it amplifies
the noise faster than the fiducials. The flag stays in the code, defaulting off, so the
result can be reproduced.

## Specialisation two, gate emphasis for FinFET, declined on a false premise

This one was designed to resolve a correlation ridge: a dense fin array is nearly invariant
under translation along the fins, so the argument ran that only the sparse gate bars pin
the position along that axis, and a channel that removes the fin periodic component would
supply the missing evidence. The premise was tested rather than assumed, and it is false.
The 90 percent contour of the correlation peak at the true pose measures one pixel by two,
isotropic, on both nominal and severity 4 pairs. There is no ridge. Consistent with that,
the two channels select sites 480 px apart on one severe pair and 96 px apart on another,
so blending them averages two surfaces that disagree rather than crossing two ridges. The
flag stays in the code, defaulting off.

The lesson is the one this project keeps relearning: a mechanism that explains a number is
not evidence for that mechanism, and the cheap measurement that would falsify it is worth
running before the fix is built rather than after.

## What the variants are worth keeping for

The FinFET builder was transposed to the published specification. The inherited builder
painted fins horizontally and gates vertically; the specification says dense parallel
vertical fin lines crossed by horizontal gate bars. For an isotropic matcher the difference
is invisible, which is why it survived this long, and it would have inverted any
anisotropic work built on it.

One discrepancy remains open and is a judgement for the team rather than a defect. The
specification describes one or two gate bars in the reference; measured by transform on a
generated reference, ours carries roughly eighteen to twenty across the 1000 nm field,
because the contacted poly pitch of 50 to 60 nm is physically correct for FinFET and the
specification's picture is about ten times coarser. The consequence was measured rather
than left as reasoning, and it runs opposite to the guess recorded here first. Rendered at
the field the specification describes, with one and a half gate bars in the reference, this
pipeline scores 0.071 against the 0.667 it scores at the shipped field, because a narrow
reference is a generic tile of a periodic array while a wide one carries enough structure to
rule impostors out. Our field is the most favourable of four measured, so 16.47 is optimistic
for a specification faithful blind set rather than pessimistic; see
`experiments/20260831_finfet_field_scale`.

With both flags off each variant reproduces the shipped entry point byte for byte on a six
pair mix spanning nominal, degraded and absent, so the specialisation is the only
difference between them and the submission.
