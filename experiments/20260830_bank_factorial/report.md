# Blur bank factorial, and the per pair time budget

Two questions, one measurement campaign. Whether extending the wide blur bank
with a larger prescreen budget lifts the degraded set, which the revert note in
`localize.py` named as the outstanding experiment; and whether the per pair
runtime is what the shipped figure claimed.

Every run in this folder is serial. The localizer carries a wall clock guard
that skips optional stages when a pair runs long, so concurrent evaluations
change each other's answers and rank configurations wrongly.

## Why the degraded set was worth the campaign

The Phase 2 addendum weights it at 0.55 against the nominal set's 0.45, gates
the optical bonus behind a sets A to C clause, and names it the first tie
breaker before rejection F1, median error and median runtime. Three reasons
pointing at one set.

## Suites

`data/p2degraded`, 200 pairs under master seed 7001, mixed 25 percent nominal
and 75 percent degraded across the four severities. A proportional suite spends
most of its generation on pairs the question does not concern; this one puts 32
pairs at the severity where the collapse is, against 9 in the proportional
holdout, which is the difference between a 0.05 credit shift meaning two pairs
and meaning seven. The nominal quarter is the regression guard, because a change
that lifts the degraded set by breaking the nominal one loses on the weighting.
Screening only. Every winner is confirmed on `data/p2holdout`, seed 9001.

## Where the degraded credit actually goes

Baseline, 150 degraded pairs.

| severity | n | credit | rejected | of those, recoverable | localizer ceiling |
| --- | --- | --- | --- | --- | --- |
| 1 | 33 | 0.788 | 5 | 0 | 0.788 |
| 2 | 43 | 0.637 | 11 | 1 | 0.660 |
| 3 | 42 | 0.433 | 16 | 1 | 0.457 |
| 4 | 32 | 0.206 | 20 | 0 | 0.206 |

Of 52 rejected present pairs only 2 would have earned any credit; the rest miss
by a median of 458 px. With nothing rejected at all the set scores 0.535 against
the 0.521 it scores now, so the presence decision is within 0.014 of optimal as
self error detection and there is nothing to win by loosening it. The degraded
set is capped by localization capability, not by the presence threshold, and the
tempting next moves, tuning that threshold or adding tone normalization against
charging, were dropped on this evidence rather than tried.

A nine pair sample from the proportional holdout had suggested the opposite,
that the presence model was rejecting good pairs. Thirty two pairs settle it the
other way. That reversal is the screening suite paying for itself.

## The factorial

| # | wide bank, nm | top k | Set A | Set B | weighted | median s | max s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4, 9, 16, 25 | 6 | 0.701 | **0.521** | 0.602 | 3.65 | 41.82 |
| 2 | 4, 9, 16, 25, 36 | 6 | 0.796 | 0.496 | 0.631 | 3.53 | 16.02 |
| 3 | 4, 9, 16, 25 | 8 | 0.700 | 0.509 | 0.595 | 3.11 | 14.84 |
| 4 | 4, 9, 16, 25, 36 | 8 | 0.797 | 0.497 | 0.632 | 3.51 | 17.04 |
| 5 | 4, 9, 16, 25, 32, 42 | 9 | 0.795 | 0.491 | 0.628 | 3.90 | 13.78 |

The named experiment does not work. Cell 4 is the bank extension with the
enlarged prescreen budget the revert note proposed, and it is indistinguishable
from cell 2 without it: widening the budget recovers none of what the bank costs
the degraded set. Cell 3 confirms it from the other side, where the budget alone
moves nothing. The budget was never the binding constraint, and the explanation
recorded in the source for the earlier revert was wrong.

Nor does the bank ceiling explain the degraded collapse. The motivation was
sound on its face, since severity 3 and 4 search captures blur to 26 and 36 nm
against a bank topping out at 25, but every extension scores the degraded set
between 0.491 and 0.497 against the shipped bank's 0.521, and wider is worse.
The severity table already said why: those pairs miss by hundreds of pixels,
which a missing blur rung does not explain.

## What the factorial did find

Set A lands at 0.795 to 0.797 for all three extensions regardless of which
sigmas are added or how large the budget is. A gain that identical across three
different banks is not a pose grid responding to better blur coverage, it is a
binary decision flipping on a few pairs. `wide_sigma_bank_nm` was serving two
unrelated jobs: enumerating pose hypotheses, and building the templates that
decide contrast polarity once per pair. Polarity is a single binary call that a
heavily blurred template makes more reliably, while the pose grid pays for every
extra level in hypotheses that crowd the prescreen.

Splitting it into `polarity_sigma_bank_nm` predicted Set A near 0.797 with Set B
held at 0.521. Measured: 0.800 and 0.515, and the pair level diff shows exactly
five nominal pairs moving 0.0 to 1.0, which is the binary signature the
mechanism predicts rather than a distribution shifting.

The gain does not reproduce. On the proportional holdout Set A moves 0.786 to
0.762 and Set B 0.643 to 0.667, about one pair each way, for a net of plus 0.09
localization points against plus 1.65 on the screening suite. Five polarity
flips in fifty nominal pairs is a ten percent rate, high enough that seed 7001
likely over represents polarity ambiguous pairs. It was kept briefly on the
argument that a mechanistically motivated change which harms neither set costs
nothing to carry, and then removed when the end to end score showed it costing
nine tenths of a point through pose and calibration and eroding the optical
bonus margin from 0.033 to 0.006. Measuring localization alone had hidden both.

## The defect the runtime question found

`time_budget_s` was enforced inside `locate`, which sets its own start time,
while the entry point calls `locate` up to three times per pair for the width
rescue. The budget therefore bounded each call and not the pair, and the pair's
real ceiling was three times the intended one. The two interact perversely: the
rescue fires on a weak peak, and a weak peak is what a heavily degraded capture
produces, so the mechanism tripled the runtime of exactly the pairs already
closest to the timeout. Four pairs in 200 exceeded the twenty second hard
timeout, all of them weak peaked, three at severity 4, and a pair that overruns
scores nothing at all.

The median hid it completely, at 3.65 s against a 5 s target, which is why the
figure carried in the readme was never wrong and never sufficient.

The fix threads one shared pair start time through every pass, declines to start
a rescue pass the remaining budget cannot cover, and refuses to let a pass the
clock cut short overrule a complete one, since a starved pass searched fewer
poses and its score is not measured on the same terms.

| | max s | pairs over 20 s | Set B, clock ignored | charging 20 s | charging 13.3 s |
| --- | --- | --- | --- | --- | --- |
| per call budget | 41.82 | 4 | 0.521 | 0.515 | 0.501 |
| per pair budget | 6.68 | 0 | 0.515 | 0.515 | 0.515 |

Six degraded pairs lose their credit to the fix, which read as damage until the
timeout was charged. The harness had been scoring every pair regardless of how
long it took, so the per call budget was being credited for pairs it wins with
time the scored run does not give it. Charged honestly the two are identical,
and against a reference machine half again as slower the fix is ahead. The
harness now charges for the timeout by default so that a configuration cannot
buy accuracy with time it does not have.

## Runtime on the scored machine

Median 3.0 s and a 6.0 s maximum on the proportional holdout with the absent
pairs included, which are the slowest in the mix because a weak peak is what
triggers the rescue. Capping OpenCV to four threads to match the scored
machine's core count changes the median by 0.15 s and the credit not at all, so
the pipeline is effectively single threaded and the four core limit costs
nothing. What the scored machine will cost is per core speed, which cannot be
measured from here; the 13.3 s column above is the guard against it.

## Method note

Two runs of the same configuration on the same pairs, hours apart under
different machine load, returned identical credit to three decimals, and
`eval_degraded.py` reproduced `register.py` exactly on the holdout. The wall
clock guard changes runtime without changing answers on this data, so the small
differences in the factorial are real rather than drift. That check was run
because the guard made it a live doubt, not as a formality.

## What shipped, and the claim it does not support

The shipped configuration is the per pair budget and the rescue start gate,
without the polarity split. Measured end to end through the entry point:

| | localization | core | max s | pairs over 20 s |
| --- | --- | --- | --- | --- |
| committed, on the proportional holdout | 28.29 | 64.64 | 11.37 | 0 |
| shipped, on the proportional holdout | 28.81 | 64.88 | 6.11 | 0 |
| committed, on the screening suite | 24.07 | | 41.82 | 4 |
| shipped, on the screening suite | 23.66 | | 6.45 | 0 |

The accuracy effect reverses between the two suites: plus 0.52 localization
points on the proportional holdout, minus 0.41 on the severity balanced suite.
Pooled by pair count it is slightly negative and indistinguishable from zero.
That is the same pattern the polarity split was rejected for, and the same
standard applies to it: **the shipped configuration is accuracy neutral, and no
point gain is claimed for it.** The proportional holdout matches the blind set's
composition and is the better predictor of the scored result, which is the only
reason the holdout number is quoted first rather than the pooled one; it is not
a reason to treat plus 0.24 as real.

What does reproduce, on both suites and in both directions, is the runtime tail.
The worst case falls from 41.82 s to 6.45 s, the four pairs that breached the
twenty second hard timeout become none, and charging a timeout scaled for a
reference machine half again as slow changes the credit not at all. That is the
whole case for the change, and it is enough on its own: a pair that overruns
scores zero outright, the exposure was invisible in a median of 3.65 s, and the
scored machine is slower than the one all of this was measured on.

The optical suites are unaffected in the way that matters: 0.433 on the training
suite unchanged, 0.592 against 0.633 on the holdout, both clear of the 0.40 gate.

## Three claims retracted during this campaign

Recorded because each was stated before it was checked, and the check is the
only reason it did not ship.

The bank extension would recover wrongly rejected pairs as well as mislocalized
ones. It would not: of 52 rejected present pairs only 2 were recoverable, so
there was never a second gain to collect.

The polarity split was worth 1.65 localization points. On the independent
holdout it was worth 0.09, and end to end it cost nine tenths of a point.

The per pair budget cost accuracy, and then, two measurements later, that it
earned accuracy. Neither: it is credit neutral, and the effect being attributed
to it belonged to the rescue start gate, which turned out not to reproduce
either.
