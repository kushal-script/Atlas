# Optical bonus margin, on a suite large enough to answer the question

The bonus pays six points if Set D localization credit clears 0.40 while the
grayscale sets clear 0.50. The two optical suites this repository had were 36
and 24 pairs, which is too few to separate a real margin from a lucky draw:
they returned 0.444 and 0.633, a spread of 0.19 on the same method.

A third suite of 72 pairs was generated under master seed 8101, disjoint from
5001 and 9501, through a generator mix that draws the bonus set alone rather
than a tenth of a proportional suite at a time. It returns 0.619, and the three
pool to 0.574 over 132 pairs, each with committed per pair credit here.

## The number that is actually scored

`eval_optical.py` measures localization only. The scored entry point also
applies the presence gate, which rejects 2 of the 72 as absent and costs 0.011,
so the figure to plan against is **0.608**, not 0.619. This was a real error in
what the readme reported until it was measured through `register.py` itself.

## What the margin is worth against a 20 pair blind set

Per pair credit is close to bimodal, 26 zeros and 39 full marks with 7 in
between, so a 20 pair mean carries real sampling noise rather than averaging
smoothly. Bootstrapping 200000 draws of 20 pairs from the measured
distribution puts the fifth percentile at 0.44 and the probability of falling
below the 0.40 gate at about 2 percent.

That is the optimistic reading and is recorded as such. The three suite means
spread more than pair level sampling alone explains, which implies seed level
variation on top of it, and the whole estimate assumes the organiser's optical
analogue resembles this generator's, which nothing measurable from here can
confirm. The honest summary is that the gate is comfortably clear on every
suite measured and the dominant remaining risk is one this repository cannot
sample.

## Files

`per_pair_credit_opt_large.csv` per pair error and tiered credit, 72 pairs.
`predictions_opt_large_register.csv` the entry point's own output on the same
suite, the rows the scored figure is computed from.
`ground_truth_opt_large.csv` the suite manifest.
`metrics.json` every number quoted above, machine readable.
