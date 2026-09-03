# Tenth review: the critical is the recorded decline read as a bug, six small hardenings ship, the convention gains an independent oracle

The tenth review, a static read with nothing executed, is the most precise
of the ten. Its critical finding, that the combiner override margin of 9.0
can never be reached by a probability difference, describes the shipped
behaviour exactly and mistakes it for an accident: the combiner override
was fitted, measured at exactly zero held out, and declined
(`experiments/20260901_rerank_combiner`), with its diagnostics deliberately
kept flowing to the presence model; the unreachable margin is how the
decline ships. The readme attributes the rejection F1 gain to the feature
block, which is correct, and its ledger row now states outright that the
margin is the decline's mechanism, so the next reader cannot trip where
two audits have.

Its strongest documentation catch was real: the method section said the
wide grid is screened only when the nominal pose correlates weakly, and at
half resolution; the shipped gate cannot block, the grid runs whenever the
budget allows, and the prescreen is quarter resolution. Both sentences and
the complexity table row now say what the code does.

Shipped hardenings, all validated byte identical on the smoke suite:
whitespace is stripped from csv path values and pair ids, so a padded path
resolves instead of silently scoring absent; the entry point creates the
output directory and reports an unreadable input csv on stderr with a
clean exit instead of a traceback before any row; a budget gated pair now
says so on stderr; the degenerate input guard treats a non finite standard
deviation as degenerate instead of passing it through; and the template
centre uses each axis's own centre, identical on the square contract
images and correct on any other.

The test suite gains what the review asked for: the pose range test loses
a tenfold slack factor and pins the grid endpoints to exactly 8.0, 12.0
and plus and minus 5 degrees, and a new oracle test constructs pairs
through the organisers' own warp, plain noise and cv2 only, no code from
this repository's generator, and asserts the reported rotation matches the
injected angle in sign and magnitude within 0.35 degrees both ways. It
passed first try, agreeing with the 0.975 rotation credit measured on the
organisers' labels.

The packaging self test now runs the sample through csv relative paths
with a deliberately padded entry, includes absent pairs, and asserts
finiteness, row order and all four pose zeros on any rejection.

Declined, with the record: the always entered wide grid and the C call
granularity of the budget are settled rounds three through nine; the
correlator calibration is the shipped answer to a measured platform
inversion and the byte identity claims are per machine statements about
recorded gates; duplicate ids, the exception score of zero, the optical
disclosure rule and every frozen constant stay exactly as measured.
