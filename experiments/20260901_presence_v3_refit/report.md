# The eighteen feature presence model ships at threshold 0.55; the ambiguity block is measured and declined

## Protocol

One records harvest of the 216 SEM pairs of p2train through the current
localizer (`tune_phase2.py`, all twenty one diagnostics recorded), three model
tiers fitted from the same records by the same class weighted logistic
(`fit_presence.py --features v1|v2|v3`): the fifteen shipped diagnostics, plus
the rerank combiner's score, margin and agreement (eighteen), plus the period
aware second peak ratio, peak curvature and lattice balance (twenty one). The
threshold sweep then reruns the register loop over three suites and stores the
full feature vector per pair, so every candidate model is rescored over the
same 378 pairs without another localizer run, and the operating point is the
midpoint of the plateau within a quarter core point of the pooled top, the
rule that shipped 0.45 for the fifteen feature model.

## Fitting suite cross validation (p2train, threshold chosen there)

    v1 fifteen features:   est core 54.1  reject F1 0.702
    v2 plus rerank block:  est core 55.5  reject F1 0.731
    v3 plus ambiguity:     est core 56.0  reject F1 0.763

## Pooled sweep, 378 pairs over p2blind180, p2holdout2, p2holdout

    shipped fifteen feature model  core 62.40  F1 0.579  auc 0.900
    v1 refit                       core 62.39  F1 0.590  auc 0.867
    v2 (eighteen)                  core 63.78  F1 0.646  auc 0.885   plateau 0.425 to 0.675, midpoint 0.55
    v3 (twenty one)                core 63.66  F1 0.657  auc 0.866   plateau 0.600 to 0.675

The v1 refit reproducing the shipped level to a hundredth of a point is the
protocol sanity check: same features, new records, same answer.

## Per suite at the chosen operating points

    suite   shipped   v1      v2      v3
    b180    62.39     62.90   63.73   63.43
    h2      58.82     57.08   59.96   60.68
    h1      66.10     66.66   67.62   66.92

v2 beats shipped on three of three suites, by 1.34, 1.14 and 1.52. v3 beats v2
on one suite of three, which under this repository's rule is a draw read from
the favourable side, so the ambiguity block is declined and the eighteen
feature model ships. The cross validation preferring v3 while the pooled sweep
declines it is the familiar signature of three extra coefficients on 216 pairs.

## End to end confirmation through the entry point

register.py itself over the two holdout suites, scored by
`score_predictions.py`:

    p2holdout   core 67.62  loc 29.24 (A 0.810, B 0.667)  pose 17.78  rej 12.00 (F1 0.800, tp 22 fp 9 fn 2)  cal 8.60
    baseline    core 66.05  loc 29.24 (A 0.810, B 0.667)  pose 17.73  rej  9.83 (F1 0.655, tp 19 fp 15 fn 5)  cal 9.25
    p2holdout2  core 59.96  loc 24.48                     pose 18.79  rej  8.24 (F1 0.549)                    cal 8.46

Plus 1.57 core on the suite every prior baseline was recorded on, with
localization identical to two decimals, which is the strongest available check
that the change touches only the decision and its confidence: six fewer
present pairs are falsely rejected and three fewer absent pairs falsely
grabbed, while the localizer's answers did not move. Both end to end cores
reproduce the harvest sweep's predictions exactly (67.62 and 59.96).

The first end to end run of this experiment scored 11.19 of 105 and is kept in
the log as a lesson: the evaluation harness built its input csv by column
position from ground truth files whose column order differs between suites, so
every image path pointed at a set label and the entry point conservatively
zeroed every pair exactly as designed. The pipeline was never wrong; the
harness was.

## Runtime, priced three ways until the attribution was clean

A first probe measured the battery at 0.55 s per locate call, which turned out
to be the per candidate recomputation of its template side blurs; hoisting the
template context (`_template_context`) leaves the battery at 5 ms per call
measured directly on real artifacts, beside 22 ms for the lattice balance and
under a millisecond for the period statistics, with live probabilities
matching the pre hoist harvest to 1.3e-6, the harvest csv's own rounding.

Suite level wall clock then contradicted itself: the same code family
measured medians of 3.46, 3.80, 4.38, 4.50 and 4.55 s across five runs on two
suites over one afternoon, including a back to back pair on one suite that
differed by 1.09 s in whichever direction the run order suggested. An
interleaved A B in a single process, alternating the combiner on and off pair
by pair over 16 pairs, resolves it: median 3.71 s on against 3.73 s off, per
pair deltas scattered half a second in both directions. The diagnostics cost
their directly priced 30 to 80 ms per pair and the rest of the suite level
spread is the machine, the same hazard the repository already records for
parallel sweeps, now observed between serial runs minutes apart. The recorded
4.02 s baseline median therefore stands as the anchor within a roughly one
second run to run band, and the worst pair observed across every run today
finished in 9.01 s, eleven seconds inside the timeout that forfeits a pair.

## What the score column now is

Unchanged in construction: confidence in the decision actually made, the
larger of the model probability and its complement, damped for found rows by
quadrant agreement. Only the probability under it comes from eighteen
features instead of fifteen.
