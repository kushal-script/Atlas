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
