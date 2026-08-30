# The reference interpreter is 12 percent slower and scores the same

The project was developed and measured on Python 3.14 while the reference
machine the addendum names runs 3.11. This experiment establishes what that
difference costs, because a runtime figure measured on the wrong stack is a
runtime figure that does not describe the scored run.

## What was wrong before the measurement

No Python 3.11 on the development machine could import the entry point at all:
`register.py` imports cv2 unconditionally through `src/drift_sense/backend.py`
and `src/drift_sense/localize.py`, and no 3.11 environment had OpenCV installed.
`scripts/package_submission.py` accepted the interpreter as a path argument and
never checked what it pointed at, so a build could report a self test passing
under an interpreter that could not have run it.

Both are now closed: `scripts/setup_python311.sh` builds the single project venv
from a verified 3.11, and the packager probes the interpreter for its version
and for cv2, numpy, scipy and Pillow before it packs anything.

## Accuracy is unaffected

Predictions are byte for byte identical across the two interpreters on a twelve
pair sample spanning nominal, degraded and absent, and the full 120 pair
proportional holdout returns the same credit on both stacks:

| | set A | set B | weighted | localization |
| --- | --- | --- | --- | --- |
| reference 3.11 | 0.690 | 0.457 | 0.562 | 22.49 of 40 |
| development 3.14 | 0.690 | 0.457 | 0.562 | 22.49 of 40 |

Every accuracy measurement taken during development therefore transfers without
being re-derived, which is the result that matters most.

## Runtime is not

| stack | median | p90 | max | over 5 s |
| --- | --- | --- | --- | --- |
| reference, 3.11.14, numpy 2.4.6, scipy 1.17.1 | 4.02 s | 5.60 s | 6.35 s | 25 percent |
| development, 3.14.0, numpy 2.5.1, scipy 1.18.0 | 3.59 s | 5.29 s | 5.74 s | 20 percent |

The reference stack is 12.0 percent slower at the median, 5.9 at the ninetieth
percentile and 10.6 at the worst pair, on the same dataset with the same code
and the interpreter and its numpy and scipy as the only variables. The
consequence is a margin, not a failure: the scored requirement is a median at or
under five seconds and the reference stack meets it at 4.02, but the headroom is
0.98 s rather than the 1.41 s the development figure implied, which is 30
percent less room than the project believed it had. Against the twenty second
hard timeout that forfeits a pair, the worst pair still finishes with 13.65 s to
spare, so the timeout is not the exposure. The exposure is the median, and it is
now measured where it will be scored.

The organisers run all fifteen submissions back to back on one machine, so any
contention on the day pushes in the same direction as this difference rather
than against it. That is the reason to hold the remaining margin rather than
spend it on accuracy.
