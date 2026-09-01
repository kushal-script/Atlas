# Hardening cycle: pose arbitration, streak correction, a faster correlator, severity five, and the hardest decoys

Four changes measured under the four arm attribution protocol on three fitting
suites (p2train, and the sample and hardened organiser recipe suites), two
adopted and one declined, plus a correlator rewrite gated on identity, and two
stress suites probing past the disclosed problem.

## Pose arbitration by the raw statistic, adopted

The diagnosed failure: at zoom twelve with rotation five, the true template is
the smallest the range produces and a wrong scale lattice lock at nine to one
won the wide grid at 0.868, mislocalizing by 731 px; the raw full reference
statistic separated the poses cleanly, 0.837 with margin 0.200 at the true
pose against 0.695 at the impostor, with its argmax 0.3 px from truth. The
arbiter scores the top distinct pose candidates from the evaluated grid, the
half resolution prescreen's own ranking (the true pose had lost the prescreen
entirely, so arbitrating only over evaluated poses inherits its blindness),
and a scale ladder at the chosen rotation, each by the raw statistic at about
15 ms; a decisive winner is evaluated and refined normally. A/B on the
fitting suites: plus 1.93 localization of 40 on the hardened recipe, exactly
zero difference on p2train and the sample recipe. The diagnosed pair went
from 730.91 px to 0.44 px with recovered scale 11.92 against a truth of 12.

## Charging streak row correction, adopted after its own test caught a hazard

The organisers' charging renders additive full width streak bands, up to some
forty five per frame at high severity. A naive running median baseline eats a
horizontal DRAM word line family at zero rotation, and the unit test written
to that hazard failed the first implementation, so the shipped baseline is
phase aware: detrend the row profile, measure its dominant vertical period,
and compare each row against the median of rows sharing its phase, so lattice
rows reconcile with their cohort and sporadic streaks stand against thirty
odd members. Correction requires a robust threshold plus an absolute floor,
never touches the outer 32 rows where the boundary windows misbehave, and
refuses frames where more than thirty percent of rows flag. A/B: plus 2.09
localization on the hardened recipe (Set B 0.876 to 0.971), exactly zero and
zero false triggers on 216 p2train pairs and the sample recipe. The
destriping literature's warning that stripe parallel structure is the failure
mode, and masked correlation as the exact alternative, are recorded in
docs/citations.md; subtraction is correct here because the disclosed streak
model is exactly a per row constant.

## Impulse trigger change, declined

Lowering the median filter's trigger to engage on the severity three salt and
pepper rate cost 0.86 localization on the sample recipe Set A and gained
nothing anywhere, so it is declined; the existing threshold stands.

## The shared transform correlator, adopted on identity

cv2.matchTemplate recomputes the search side transform and window tables on
every call, tens of times per pair against one fixed search image. The
correlator now computes the search spectrum and integral images once per pair
and evaluates each hypothesis with one template transform, one spectrum
product and one inverse transform, the fast normalized cross correlation
factorisation with the search side hoisted. Gate: surface agreement with
matchTemplate to 2.4e-7 with identical argmax on a probe, byte identical
predictions on the smoke suite, and every credit figure, median error and
rejection count identical over the 60 pair organiser recipe holdout, where
the median fell from 3.85 s to 2.85 s per pair, matching the 2.06x per call
microbenchmark through the correlation share of runtime. The direct path
stays selectable as device cpu_direct.

## Severity five: what lies past the disclosed ladder

A rung extending every disclosed knob along its own progression (dose 20,
detector 18, streaks 6.0 at 3.5, speckle 0.42, salt and pepper 0.02,
astigmatism 1.8, vignette 0.40, gamma 1.40, barrel 0.008, shear 4, jitter
1.4), each continuation bounded by published physics: low dose CD metrology
operates at single digit detected electrons per pixel, per row white jitter
and a linear shear ramp are the published scan noise and constant drift
forms, and the tone and shading levels sit inside operator practice. Their
own verifiability gate refused some severity five draws outright, which
bounds the rung as near the edge of the generatable. Measured end to end:
credit 0.850 at severity four and 0.680 at severity five, no false rejects,
and the score column still separates the three confidently wrong grabs
(median score 0.514) from correct ones (1.000), so past the disclosed ladder
the pipeline degrades by losing credit, not by lying.

## The hardest decoys, measured and priced into the threshold

Absent pairs built the way the organisers' readme recommends for the real set
and their sample avoids, identical zone geometry and merely different random
structure, are the hardest rejection case: at threshold 0.35 eleven of
eighteen such decoys are grabbed, reject class F1 0.560 on that suite, while
present pair localization is untouched and no present pair is sacrificed.
The response is not a point fix but a pricing: the suite joins the threshold
selection pool, and over the five pooled suites (492 pairs, both generators,
easy and hard decoys) the found graded objective still peaks flat from 0.10
to 0.30 while the reject graded curve is flat within 0.17 points from 0.30
to 0.65; the shipped 0.35 sits within 0.42 of the found optimum and 0.13 of
the reject optimum, effectively the best worst case under the grading
ambiguity, so the operating point stands with the hard decoy cost now inside
the measurement rather than outside it. The identical geometry decoy remains
the recorded residual weakness: if the blind set leans heavily on that class,
rejection recall on it is near 0.39 at this operating point, the cost of not
sacrificing found class F1 and localization everywhere else.

## Postscript: the surprise seed

After every decision of the campaign was frozen and committed, one further 40
pair hardened recipe suite was generated from seed 909090, chosen at that
moment, so nothing in the pipeline could have seen it even indirectly. It
scored an estimated 81.68 of 85 against the battery's 81.54 on the same
recipe: localization 39.74 of 40 with every severity 3 and 4 pair at full
credit, pose 19.38, no false rejects, decision AUC 0.971, and two grabbed
decoys in the pattern the located weakness predicts. The headline number
reproduces on data that postdates every choice it could flatter.

## Postscript two: five post freeze seeds, not one

One surprise seed is a sanity check; five make a distribution. Four further
40 pair suites were generated after the postscript above, seeds 111213 and
141516 on the hardened recipe and 171819 and 202122 on the sample recipe,
all chosen after every decision was frozen:

    seed 909090 hardened   81.68
    seed 141516 hardened   79.20
    seed 171819 sample     79.61
    seed 202122 sample     76.82
    seed 111213 hardened   76.59

Mean 78.78 of 85 over 200 post freeze pairs, spread 76.59 to 81.68. The
battery estimates sit inside the band. The decomposition matters more than
the mean: localization is tight across every seed, 36.2 to 39.7 of 40, and
pose tighter still, 19.0 to 19.6 of 20, while nearly all the spread comes
from the rejection block, F1 0.615 to 0.875, because each suite carries only
eight absent pairs and one flipped decoy moves the F1 by roughly a tenth.
The honest reading of the blind set follows: the localization and pose
claims carry seed level error bars of about a point, and the rejection score
on any single 40 pair draw carries error bars of two to three, which is the
sampling reality of a forty absent pair blind set as much as of these suites.

## Postscript three: the correlator choice is now measured, not assumed

An external re audit on a Windows x86 machine measured the cached transform
correlator 1.3 to 3.2 times SLOWER than cv2.matchTemplate there, the inverse
of this machine's factor two gain, credibly because matchTemplate enjoys IPP
acceleration on x86 that the plain dft path does not, and the reference
machine is x86. The CPU correlator is therefore chosen by a one time timed
calibration of both paths against the first search image, about a fifth of a
second per process amortised over every pair, with either path still
selectable explicitly; the two agree to float rounding and were verified
prediction identical over full suites, so the calibration affects speed
alone. The same audit found the nine second internal budget consuming the
blur bank before the wide grid ran at extreme poses on that slower machine,
so the budget rises to fifteen, inside the eighteen second alarm and twenty
second forfeit: on this machine 57 tests pass, the smoke suite is byte
identical, and the hardened forty changed exactly one row, an absent decoy
grabbed either way whose unscored coordinates moved when a previously gated
stage got room to run, every scored quantity identical at 81.54.
