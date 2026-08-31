# Narrowing the FinFET field to the published picture makes it nine times worse

The published dataset description says a FinFET reference shows dense parallel fins
crossed by one or two horizontal gate bars. At the shipped reference field of 1000 nm a
contacted poly pitch of 50 to 60 nm puts about eighteen bars in the frame, so this
generator renders a field roughly twelve times wider than the description, and a search
image carrying 58651 candidate cells against the 394 the description implies. The
hypothesis under test was that this makes our FinFET measurements pessimistic, because
every extra candidate is another chance for an impostor to win.

The hypothesis is false, and the measurement inverts it.

## The sweep

The same device is imaged over a narrower area at each step. Only the sampling geometry
scales, the canvas extent and pixel and the pixel of each capture; every physical quantity
stays in nanometres, so the pitches, the beam spot, the sidewall roughness and the charging
length are identical across the four suites and only the imaged area changes. The zoom ratio
is preserved so the Phase 2 range of 8 to 12 still applies. Sixty pairs per suite from one
master seed, 48 of them carrying a true instance, nominal dose.

| field scale | reference field | gate bars | candidate cells | macro | oracle wins | credit | median error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | 1000 nm | 18.2 | 58651 | present | 54 percent | 0.667 | 0.20 px |
| 0.500 | 500 nm | 9.1 | 14663 | present | 62 percent | 0.625 | 0.24 px |
| 0.250 | 250 nm | 4.5 | 3666 | omitted | 52 percent | 0.500 | 35.68 px |
| 0.082 | 82 nm | 1.5 | 394 | omitted | 17 percent | 0.071 | 476.08 px |

Removing 147 impostor sites out of every 148 costs nine tenths of the credit. At the field
the specification describes, the pipeline localizes almost nothing.

## Why, and it is not the missing macro

An SRAM macro is a micron scale block, so a field of 820 nm sits inside one region of the
die rather than spanning several and the generator omits it. That removes aperiodic content
and is a genuine confound between the two upper rows and the two lower ones. It does not
explain the result. Both the 0.250 and the 0.082 suites omit the macro, and credit still
falls from 0.500 to 0.071 between them, a drop three times larger than the entire step at
which the macro disappeared. Scale carries the collapse on its own.

The oracle column says what the mechanism is. It supplies the true zoom and rotation, so
the pose search is removed and the only question left is whether the correlation prefers
the true site. At the widest field it does on 54 percent of pairs; at the published field
on 17. The evidence gets worse as the field narrows, which cannot be a search failure and
cannot be fixed by any matcher.

The reason is that candidate count is the wrong quantity to reason about. What decides a
match is how much information the reference carries that is not repeated elsewhere, and a
narrow field carries less of everything: at 82 nm the reference holds about one and a half
gate bars and under three fins, which is a generic tile of a periodic array and describes
almost any site in the search image equally well. A wide field is ambiguous in more places
but each of its candidates is far better distinguished, because the reference contains
enough structure, including whatever aperiodic content the field happens to span, to rule
almost all of them out. Information in the reference beats scarcity of impostors.

## What this changes

Two things, and one of them is a retraction.

The shipped generator is not making FinFET unrealistically hard. Of the four fields
measured it is the most favourable, so the 16.47 of 40 that FinFET scores through the entry
point is not an artefact of a field chosen too wide, and there is nothing to recover by
narrowing it.

WITHDRAWN, see `experiments/20260901_organiser_convention`. The organisers have since released the generator prompt, which pins the reference at 1000 by 1000 pixels at one nanometre per pixel and the search at 1000 by 1000 at z nanometres per pixel. That is this repository's own geometry, so the scored set uses the field we generate and the narrow field measured below describes no blind set that will exist. The sweep's measurement stands and its mechanism stands; the extrapolation in this section does not, and the paragraph that follows is kept only so the retraction has its subject.

The risk runs the other way from the direction previously recorded. The earlier note in
`experiments/20260831_architecture_variants` reasoned that a specification faithful FinFET
would carry aperiodic landmarks ours lacks and that our figure was therefore pessimistic.
That was reasoning, not measurement, and the measurement contradicts it: rendered at the
field the specification describes, this pipeline scores 0.071 rather than 0.667. If the
organisers generate a reference showing one or two gate bars, FinFET is far harder than our
number suggests, not easier. The correct posture is that our figure is optimistic for that
case, and that the exposure to an undisclosed architecture is larger than previously
recorded rather than smaller.

## A cost worth recording

A specification faithful field is expensive to generate. Rendering 1000 pixels across 82 nm
requires the specimen canvas to carry sub angstrom detail, 0.164 nm per canvas pixel, so a
15 nm fin spans about 91 pixels instead of about 7.5 and painting becomes area bound. The
measured cost is 49 seconds per pair against 12 at the shipped field, a factor of four
after the smaller feature count is netted off.
