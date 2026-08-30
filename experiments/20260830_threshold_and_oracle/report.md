# The presence threshold, and the oracle that bounds the degraded set

Two questions with one harvest. Whether the operating threshold of the presence
decision sits where the scored objective wants it, and whether the degraded
set's collapse is a search failure a better matcher could recover or an
information limit nothing can.

## The harvest

`scripts/tune_threshold.py` runs the shipped pipeline once per pair over the
proportional holdout, 108 grayscale pairs of which 24 are absent, recording the
presence probability beside the localization error and both pose errors. The
threshold is then swept over the recorded values, scoring localization, pose,
rejection F1 and calibration AUC together, so the sweep costs one localizer
pass rather than one per threshold. `harvest_per_pair.csv` is that record and
`threshold_sweep.csv` the sweep over it.

## What the sweep found

The fitted threshold of 0.48 rejects 36 of 108 pairs, a third of a batch whose
disclosed composition is 22 percent absent, so the decision was over rejecting
relative to the one fact the addendum states about the blind set. The sweep's
plateau runs from 0.30 to 0.60 with its top between 0.40 and 0.45, worth about
0.4 core points over the fitted value. The operating point ships at 0.45, the
plateau edge nearest the independently fitted 0.48, because the difference
between 0.40 and 0.45 is a fraction of one pair and choosing the far edge would
be fitting the holdout.

## A rule proposed, measured and rejected

Thresholding a probability calibrated on our own generator assumes the
calibration transfers to the organiser's. A batch adaptive rule was designed to
weaken that assumption to ranking alone: reject the lowest 22 percent of the
batch by presence probability, clipped to the sweep's plateau so degenerate
batch compositions degrade to the fixed behaviour. Simulated on the harvest
under logistic shifts of the probability distribution, the adaptive rule
recovered at most 0.14 core points at a shift of minus two logits while costing
0.35 on unshifted data. The fixed threshold barely degrades under shift because
the model's probabilities are strongly bimodal, so a uniform shift moves few
pairs across any fixed boundary, and the impostors the model misses sit high in
the ranking where quantile selection cannot reach them either. The rule is
recorded here and not shipped.

## The oracle, and what the degraded set actually is

`scripts/oracle_probe.py` hands the matcher the true zoom and rotation from the
generator's own record, which no scored run has, and asks whether the true site
then wins the correlation. Severity 1: it wins on 88 percent of pairs, and
where an impostor wins it wins by 0.005, a whisker. Severity 3: 64 percent.
Severity 4: 34 percent, and the winning impostor leads by a median of 0.054,
eighteen times the rescue margin.

A perfect pose search would therefore localize about one severity 4 pair in
three. The shipped pipeline scores on about one in five, so most of the gap to
a perfect search is already closed, and the majority of the remaining degraded
loss is information limited: the evidence itself prefers the wrong site. No
pose search, blur bank, prescreen budget or threshold recovers those pairs,
which is why every configuration experiment aimed at them measured negative,
and why rejecting them, as the presence decision does, is the correct response
on a tool where a confident wrong grab silently corrupts a measurement and a
rejection costs one cheap rescan.

## The optical decision

The addendum discloses the bonus set as reference present, and its rejection is
never scored: the F1 runs over the grayscale pairs alone. A rejected optical
pair can therefore only forfeit bonus credit, and the entry point measured 2 of
72 such rejections costing 0.011 credit on this repository's own optical suite.
The disclosed fact is now used the way the disclosed pose bounds are: an RGB
pair is always reported found, with the model's probability itself as the
score, so calibration still ranks a doubtful forced answer below a confident
one.

## End to end confirmation

The entry point over the full proportional holdout reproduces the sweep's 0.45
row to every digit: localization 29.24, pose 17.73, rejection 9.83, calibration
9.25, core 66.05, F1 0.655 at tp 19, fp 15, fn 5. The 72 pair optical suite
returns 0.619 with nothing rejected, recovering the 2 pairs the gate forfeited.
`predictions_holdout_final.csv` is the entry point's own output for that run,
and the session's core progression on this holdout, one scorer throughout, is
64.64 at the start, 64.88 after the audit fixes, 65.71 after the template cap,
66.05 here.
