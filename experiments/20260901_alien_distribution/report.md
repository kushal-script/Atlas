# A distribution family neither generator resembles, and the defect it caught

Seed disjoint is not distribution disjoint, so this suite attacks the gap
directly: 48 pairs from the released pipeline, geometry and labels exact,
with every search capture mutated by one of six appearance transformations
chosen to break the assumptions the Phase 2 adaptations lean on. Inverted
contrast, gamma 0.55 against their 1.25, dark streak bands the corrector
cannot lift, vertical streak bands the row corrector must not misfire on,
tone crushed into seventy gray levels, and multiplicative speckle at 0.5,
past their ladder's maximum. All photometric, so ground truth survives by
construction.

## Result before the fix

    mutation           present credit   wrong grabs   false rejects
    dark_streaks            1.000            0              0
    gamma_down              1.000            0              0
    heavy_speckle           1.000            0              0
    tone_crush              1.000            0              0
    vertical_streaks        1.000            0              0
    invert                  0.714            2              0

Five of six families localize perfectly: the pipeline's invariances, gain
and offset invariant correlation, the bandpass, the phase aware gating of
the streak corrector, transfer to appearance families it never saw, and the
row corrector does not misfire on column streaks. The sixth family found a
real defect: two inverted pairs mislocalized at score 0.952, confidently
wrong, the exact misfire class this suite was built to catch.

## The defect and the fix

The main path detects contrast inversion and negates the search, but the raw
full reference confirmation and the pose arbiter correlate the UNPROCESSED
captures, which the polarity decision never reached. On an inverted search
the raw statistic is meaningless noise, and an override that fires on
meaningless evidence moved two correct answers. Two changes, both inert on
convention: the polarity decision now also inverts the raw search copy, and
neither the override nor the arbiter acts unless the raw peak clears 0.25, a
floor the released gate's own present pair minimum near 0.34 never trips.
After the fix the inverted family reads 1.000 with no wrong grabs, the smoke
suite is byte identical, and the hardened recipe forty, the suite where the
override and arbiter earn the most, is byte identical, so the floor blocks
nothing legitimate.

## Audit items closed in the same pass

The external audit's nan finding is fixed at both levels rather than left
accidentally correct: degenerate candidate statistics are finite filtered
with the lattice lag clamped, every presence feature block falls back to its
neutral default on a non finite value, and a near constant capture is now
rejected by an explicit decision at the entry point, since with finite
features the model would otherwise accept a blank frame it has no business
judging. The stale readme sentence describing a narrower Phase 1 grid is
corrected, and the contract battery gains symlinked image paths and the
degenerate blank pair, 57 tests.

## What this does and does not prove

It does not prove robustness to every unseen family; nothing can before the
blind run. It proves the specific thing that was provable: across six
deliberate appearance breaks the pipeline either kept full credit or, once,
failed in the one way the design forbids, and that one way is now measured,
fixed, floored against recurrence, and covered by the suite that found it.
