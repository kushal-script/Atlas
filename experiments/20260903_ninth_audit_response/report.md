# Ninth review round: the catastrophic failure claim reproduced, and it is the presence gate doing its job

Two further external reviews arrived on submission day, a ten pair head to
head that now concedes the earlier rotation sign and accuracy claims and
scores this build ahead on the rubric, and a thirty agent audit run on a
Windows laptop against synthetic pairs from this repository's own
generator, its own postscript confirming no held out suite and no organiser
material was available to it. Every concrete claim was reproduced here
rather than argued with.

The catastrophic failure. Both reviews flag seed 42 at about 450 px. It
reproduces exactly: generate_pair(seed=42, style dram), the bare localizer
answers 450.2 px off with 6411 near equal wide candidates, the recorded
periodic ambiguity class where the surface genuinely does not decide. Both
harnesses called the internal localizer directly. The shipped entry point
on the same pair rejects it, found 0 at probability 0.651, and the finfet
pair from the same seed localizes to 0.12 px at probability 0.993. The
wrong centre never reaches the output; the failure mode the reviews warn
about is the exact behaviour the presence gate exists to produce, a low
scored refusal instead of a confident miss.

The intensity scaling violation. The metamorphic claim, that scaled
intensities flip contrast polarity and silently mislocalize by hundreds of
pixels, was retested at the product level: six seeds, both halved and
doubled with clipping, twelve pairs through register.py. Nine localize
correctly, three saturated pairs are rejected, and zero land on a wrong
site. Doubling with clipping is saturation, not an affine intensity change,
so the invariance being tested does not apply to it, and the gate again
converts the hard cases into refusals rather than silent errors.

The NaN findings. The probability path is guarded where it matters: a non
finite score cannot reach the csv because the entry point passes every
reported quantity through the finite guard, a NaN probability compares
false against the threshold and yields the conservative rejection, and the
per pair exception handler converts any crash into a found 0 row, so the
contract survives every input the reviews constructed. Two hardenings ship
anyway: the float image branch of the loader now maps NaN and infinity to
the frame's floor and ceiling before normalising, with a test loading such
a frame finitely, and the per pair exception handler now names the
exception on stderr instead of swallowing it silently, so a failed pair is
diagnosable during a run.

The timeout wording. The review read the readme's twenty second hard
timeout as a misstatement of the eighteen second alarm; the sentence
describes the organisers' own forfeit rule, which is twenty seconds, and
now says whose timeout it is at both places it appears.

Runtime on Windows. The 6.94 and 11.3 second medians were measured on
checkouts predating the pooled surface evaluation; the response to the
reference machine budget is that pool, committed with byte identical math,
and the number that decides is a fresh measurement on reference class
hardware with the current build and the pinned stack.

The rest restates settled ground: the SIGALRM trade with the shared wall
clock budget as the bound, the always entered wide grid, the F1 figure the
review compared across two different protocols, cross validation in the
model file against held out end to end in the readme, and the correlator
pair whose byte identity on full suites is gated on every packaging run.

## Addendum: the same reviewer's re run on the organisers' sample

The reviewer whose synthetic harness raised the claims above re ran the
current build on the organisers' twenty shared pairs on a Windows machine
and independently reproduced the committed numbers: sixteen of sixteen
present pairs within 5 px, mean error 0.37 px against the 0.38 px median
recorded here, maximum 1.15 px, and the two identical geometry decoys
grabbed with the other two absents rejected, exactly the priced behaviour
the threshold selection recorded. A second operating system, a second
architecture and a second pair of hands landing on the same digits is the
strongest reproduction the sample allows.
