# The learned combiner override is declined; its diagnostics move to the presence model

## What was proposed

Replace the absolute residual re rank override with a fitted combination of all
seven candidate statistics (correlation, edge locked residual, lattice residual
autocorrelation at one and two pixels, gradient correlation, coarse scale
correlation, absolute residual), fitted as a logistic over 2163 candidates from
the 216 SEM pairs of p2train at the estimated pose, with the override margin
fitted on the training suite before any held out number was read, exactly the
protocol that shipped the absolute residual override.

## What the measurement said

    fitting suite, override off: 25.84 of 40
    optimum 25.84 on margins 0.14 to 0.50; midpoint 0.32 chosen BEFORE reading the holdout
    HELD OUT: off 22.90 -> at 0.32 22.90   delta +0.00 of 40

On the fitting suite the best the fitted override can do at any margin above
0.14 is nothing, and below 0.14 it damages; held out on p2holdout2 the delta is
exactly zero. The shipped absolute residual override earned plus 0.52 held out
over the same baseline, so the single statistic with a relative margin remains
the shipped localizer and the fitted combination is declined as an override.

## Why the diagnostics survive the decline

The override and the diagnostics are separate hypotheses. The combiner still
produces, per pair, its probability for the chosen site, the margin over the
runner up, and whether it agrees with the correlation's choice. Disagreement
between two independent evidence functions over which site is true is close to
a direct measurement of ambiguity, and ambiguity is what a rejection is. A
first harvest level comparison (experiments/20260901_presence_rerank_features)
measured the three features at plus 0.030 reject F1 and plus 0.114 decision
AUC held out under a fitting protocol held identical between arms, so the
diagnostics feed the presence model while the override stays off. The config
ships rerank_combiner true with rerank_combiner_margin 9.0, which computes the
diagnostics and can never move the answer; predictions were verified byte
identical to HEAD on the smoke suite with the fifteen feature model in place.
