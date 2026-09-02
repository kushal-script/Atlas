# The raw override's one destructive mode, closed at margin 0.05

A fifth external audit reproduced the raw confirmation override converting a
correct answer into a wrong lattice grab: a pair from this repository's own
generator whose classical answer sat 0.13 px from truth had its raw statistic
at the estimated pose put a wrong site 42 px away at margin 0.0202, a hair
over the 0.02 floor, and the override shipped the wrong site at score
0.89994 with nothing in the row marking it. The mechanism is real, was
verifiable from this repository's own records, and the fix went through two
rounds because the first round exposed a stale measurement problem worth
recording on its own.

## Round one: the recorded margins, and what they seemed to say

The six September 1 record sets hold every disagreeing raw confirmation with
its margin, peak and both candidates' errors. Under the exact shipped
trigger, disagree and peak at least 0.25 and margin at least the floor, they
split at first sight cleanly: every recorded damage at margin 0.0200 or
lower, worst p2train pair_0128 whose 0.32 px answer the override moved to
3.61 px at margin exactly 0.0200, and every recorded rescue at 0.0369 or
higher. The original 0.02 step sweep never looked between 0.02 and 0.04 and
omitted the peak floor, so the audited damage at 0.0202 sat exactly in the
unswept gap. `sweep.log` holds the round one resweep on those records:
fitting plateau 0.021 to 0.036, midpoint 0.0285.

## What end to end validation exposed

Full serial reruns at 0.0285 disagreed with the offline prediction: the four
p2train rescues did not fire, because their margins are not what the
September 1 records say they are. Remeasured on the current build, margins
recorded at 0.0369 to 0.0932 on own generator mislocked pairs collapsed
below 0.01, and the h2 rescues at 0.0724 and 0.0881 collapsed the same way.
Those firings were pose estimate luck of an earlier build: on a pair whose
classical answer is a hundreds of pixels mislock, the estimated pose feeding
the raw statistic is itself wrong, and the correlator calibration and budget
commits that came after September 1 moved it enough to dissolve the margins.
The stable numbers are the ones with physics behind them: the damage at
0.0200 to 0.0202 across two builds and the external audit's machine, and the
released recipe rescues, where the generator manufactures the statistic's
uniqueness, at 0.2049 on amatgen_holdout v007 in both builds.

## Round two: fresh records, and the floor that follows

All six suites were rerun with the override disabled on the current build,
660 pairs serial, recording the classical answer and the raw candidate for
every pair (`--no_raw_override`, added to `scripts/tune_phase2.py` for this).
The fresh population of eligible firings, disagree and peak at least 0.25,
is five pairs:

    fit   p2train pair_0128   margin 0.0201  0.32 px overridden to 3.61   DAMAGE
    fit   amat_h  v007        margin 0.1089  23.3 px to 7.6, both zero    neutral
    judge h2      pair_0028   margin 0.0193  54 px to 53, both zero       neutral
    judge h2      pair_0107   margin 0.0322  64 px to 65, both zero       neutral
    judge amath   v007        margin 0.2049  19.2 px to 0.63 px           RESCUE

`sweep_fresh.log`: the fitting optimum is every floor from 0.021 up, a
plateau bounded below by the one damage and unbounded above inside the
sweep range, so plateau midpoint selection does not apply and the floor is
set by the observed bands instead: every damage at 0.0202 or below, every
genuine raw resolution at 0.1089 or above. The shipped floor is 0.05, the
external audit's own recommendation, clearing the damage band by 2.5x with
the rescue band 2x above it; all three held out suites move exactly zero
against any other in band floor, and the one scored effect anywhere is the
kept amath rescue. Validation beyond the pool: the organisers' sample
severity 4 pair p012, the flagship released recipe rescue, remeasured at
margin 0.1453 with the pipeline now landing on the raw site before the
override is even needed, byte identical to the final battery row; the smoke
suite is byte identical between the old and new code on this machine; the
gate logic is extracted into `_override_fires` with a regression test
locking the damage band out and the rescue band in; 60 tests pass.

What this does not change: the override's held out value on the released
recipe, the +0.86 amath delta and the sample pair rescue, both reproduce on
the current build. What it does change: the own generator rescues in the
original ledger row were artifacts of a superseded build, so the honest
current claim is narrower, the override is a released recipe instrument
whose firings elsewhere are now confined to a band where nothing scored has
ever been lost.

## Disposition of the audit's remaining findings

The score column on wrong site grabs, proposed damping by raw versus
classical disagreement: declined. Above the 0.05 floor the firings are the
rescues, where the shipped answer is correct and a disagreement damp would
rank right answers low; the demonstrated wrong grab path no longer fires.
The quadrant damp already carries the correctness proxy and the score's AUC
against per pair correctness is 0.914 held out, 0.971 on the surprise seed.

Runtime on slower hardware: the audit measured 10.6 s median on a Windows
box about 2.5x slower than the reference stack with nothing budget gated;
the README runtime table now carries that row and the point that what
transfers to unknown hardware is the 20 s forfeit distance, not the 5 s
headroom.

Identical geometry decoys: already documented and priced in the failure
analysis; no change.

Stale mermaid diagnostic count: fixed three commits before the audited
revision; both diagrams read eighteen at head.

Variants at threshold 0.45: frozen historical forks by design;
`variants/README.md` now says so explicitly.
