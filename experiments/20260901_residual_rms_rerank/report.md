# Breaking the correlation ceiling with an absolute residual re rank

Eleven measured hypotheses had been declined before this one, and the conclusion drawn
from them was stated too strongly. The oracle probe shows that supplying the true zoom
and rotation still leaves an impostor winning on 66 percent of severity 4 pairs, and that
was read as an information limit in the pixels. It is not. It is the ceiling of one
evidence function, normalized cross correlation, and every declined candidate either
tuned that function or derived a feature from its own surface. This work asked what a
different evidence function sees, and one of them sees more.

## Why correlation loses, and what does not

NCC divides each window's mismatch by that window's own contrast. A high contrast
impostor is therefore forgiven exactly the mismatch that ought to convict it, and on a
degraded capture, where the true site's contrast has collapsed with the dose, that
forgiveness is decisive. The RMS of the residual after a least squares gain and offset
fit forgives nothing: it asks how many grey levels are left over once the template has
been scaled and shifted onto the window as well as it can be.

The prediction that follows is specific and testable. The advantage should be zero on
clean pairs, where correlation already wins, and should grow with severity. It does.

## Seven evidence functions, ranked against the chance floor

Each statistic ranks the top peaks of the correlation surface at the true pose. The chance
floor is the win rate of picking a candidate at random, about 10 percent.

| statistic | p2holdout2 sev0 to sev4 | overall | p2train overall |
| --- | --- | --- | --- |
| chance floor | 10 10 10 10 9 | 9.8 | 9.7 |
| ncc | 83 80 29 50 14 | 64.3 | 60.7 |
| edge locked residual | 12 13 14 17 21 | 14.3 | 11.9 |
| lattice residual | 17 27 43 33 43 | 26.2 | 24.4 |
| residual whiteness | 57 60 43 67 43 | 54.8 | 55.4 |
| gradient ncc | 83 87 43 17 14 | 64.3 | 65.5 |
| mat scale ncc | 50 47 29 50 21 | 42.9 | 38.1 |
| **absolute residual rms** | **81 60 57 67 43** | **67.9** | **73.2** |

At severity 4 the residual ranks the true site first on 43 percent of held out pairs
against correlation's 14, and on 50 percent of the fitting suite against 8. At severity 0
it is level with correlation or just behind. That is the predicted shape.

A first version of this battery reported gradient ncc ahead by 11.9 points. It was an
artefact of that probe building its template without blurring it to the search optics.
With a matched formation template, correlation itself rose from 56.0 to 64.3 and gradient
ncc landed on exactly the same 64.3. The shipped pipeline already builds matched formation
templates, so it had already banked that gain and the apparent advantage was measuring the
probe's own defect. Recorded because the first number was the one that looked exciting.

## The gate, fitted once and never refitted

An override that always takes the residual's answer rescues 4 pairs and damages 4 on the
fitting suite, a net of nothing. The override therefore fires only when the residual's
candidate is distant from the classical answer and beats it by a relative margin. That
margin was swept on p2train alone from a single harvest recording both candidate answers
per pair, and the optimum is a plateau from 0.030 to 0.060 worth 26.78 of 40 against 25.84
with the re rank off. The midpoint, 0.045, was written into the code and this file before
any held out number was computed, because the per architecture thresholds declined earlier
in this project showed plus 0.11 on their fitting data and minus 0.545 held out.

At the shipped margin the gate fires on 16 of 168 fitting pairs, rescues 4 and damages 0.

## Result

| suite | measurement | off | on | delta of 40 | set A off | set A on |
| --- | --- | --- | --- | --- | --- | --- |
| p2train | localization only | 25.84 | 26.78 | +0.94 | 0.821 | 0.821 |
| p2holdout2 | localization only | 24.38 | 24.90 | +0.52 | 0.714 | 0.714 |
| p2holdout2 | end to end scored | 22.49 | 23.43 | +0.94 | 0.690 | 0.690 |
| p2train | end to end scored | 24.21 | 24.68 | +0.47 | 0.810 | 0.810 |

Four measurements, all positive, mean plus 0.72 of 40. Set A does not move in any of them,
to three decimals, and neither do the nominal and severity 1 and 2 median errors. The whole
effect is in the degraded set, which carries the heavier 0.55 weight.

The end to end gain exceeds the localization only gain on the holdout, 0.94 against 0.52,
and the reason is visible in the per severity table: severity 3 rejections fall from 4 to
3. A rescued pair does not merely earn localization credit, it lands on the true site, its
quadrant diagnostics improve, and it passes a presence gate it previously failed. This
interaction was flagged as a risk before the measurement, since the presence model was
fitted on features recorded without the re rank, and it ran favourably. It could have run
the other way and the plan was to refit the presence model if it had.

Runtime is free: plus 0.08 s of median on p2train and minus 0.03 s on p2holdout2, inside
run to run variation, against 0.98 s of headroom under the five second requirement.

## What is honest about the size of this

The rescued pairs number 2 to 4 per suite. Pooling both suites gives 6 rescues against 1
damage over 7 discordant pairs, which a sign test puts at about p 0.06. The effect is real
in direction and mechanism and small in magnitude, and the delta halving between fitting
and held out data is ordinary regression toward the mean. The defensible claim is plus
roughly half a point of 40, not plus one.

## What this retracts

The failure analysis said the remaining degraded loss is information limited. That is true
only of the part attributable to dose and beam spot, which is 65 percent of the severity 4
loss and is genuinely information never collected. The remainder was described as beyond
reach of any matcher, and this result shows that was a claim about normalized cross
correlation rather than about the pixels. One evidence function outside the correlation
family moved a number that eleven attempts inside it could not.
