# Sixth external audit: no critical findings, two fixes shipped, the rest verified or declined with arithmetic

The sixth audit verified the contract, the coordinate system, the pose grid,
the presence decision, the leakage discipline, the packaging and the tie
break convention finding by finding and reported no critical issue. Its
remaining findings, dispositioned after independent verification at head:

Windows timeout, reported HIGH with a 45 second worst case: the arithmetic
is wrong by the audit's own evidence. Every locate call, the original and
both width rescues, receives the same pair start timestamp, so all three
share one 15 second wall clock budget rather than stacking three, and the
rescue loop refuses to start past half of it. The bound on a platform
without SIGALRM is 15 seconds plus a single stage's granularity, and the
profile puts every stage at sub second C calls. The recommended fix, a wall
clock check inside the pair loop, is what the internal budget already is.
Declined; the README runtime table now states the no SIGALRM bound
explicitly.

Fallback constant naming, HIGH: legitimate. FOUND_THRESHOLD and SCORE_WIDTH
are dead code whenever the model file loads and only name the no model
fallback; renamed to FALLBACK_FOUND_THRESHOLD and FALLBACK_SCORE_WIDTH,
behavior identical.

quad_missing standardization at 1e-09, MEDIUM: real latent fragility,
currently inert because the feature's weight is exactly zero. The fitting
script now floors every standardization constant at 1e-4 so a future refit
cannot inherit the hazard; the shipped model file is untouched because
rewriting it would change bytes for zero behavioral gain.

Oversized images, MEDIUM: declined with arithmetic. The specification fixes
both captures at 1000 by 1000; an off specification giant would slow the
bank ahead of the first budget gate, on Unix the 18 second alarm converts
that to one conservative row, and on Windows the cost is one slow pair
whose worst outcome, the 20 second forfeit, is already priced per pair by
the row flushed writer. A resize guard would touch coordinate semantics on
the shipped path to defend against an input the contract rules out.

Score 0.5 for degenerate images against 0.0 for timeout rows, MEDIUM: both
deliberate, a coin flip confidence for a pair the pipeline chose to reject
against no confidence for a pair it never finished; no timeout row has ever
been emitted in any recorded suite, so the distinction has never touched a
scored quantity.

Pillow as an inference dependency, LOW: correct as listed, the generator in
the same archive needs it.

Correlator choice timing dependence, LOW: by design, the calibration is the
shipped answer to the platform inversion an earlier audit measured, and the
two correlators agree to float rounding.

dataset_format.md missing from the packaging list, LOW: stale; the file is
in the INCLUDE list and in the archive at head.

The audit's sharpest observation was about the test suite rather than the
code: nothing ran the entry point end to end on real images. That gap is
closed with four new tests that generate a present, an absent and an RGB
optical pair, run register.py as a subprocess on a manifest of relative
paths, and assert the exact columns, the found decisions, sub 5 px accuracy
with pose inside the credit bands on the present pair, the zeroed rejection
row, the forced optical found flag and the score ranges on the CSV it
actually writes. 64 tests pass; the smoke suite is byte identical through
the rename.
