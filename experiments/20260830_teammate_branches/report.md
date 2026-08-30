# Three teammate branches, evaluated rather than merged

Six branches existed beside main. Two carried ideas worth measuring, one carried
a number that does not mean what it appears to, and the rest were supersets of
those. Everything below was measured in an isolated git worktree with main's
checkout untouched, and each candidate was rebased onto current main first so
it was compared against the same baseline rather than an older one.

## The claim that does not survive reading

`phase2/day7-archive` reports "retrain presence LR on post-T3 features
(F1 0.9068)", which is above the addendum's 0.90 bonus gate. The number is not
comparable to ours, for three reasons visible in `scripts/retrain_day5.py`:
the model is fitted and then scored on the same rows, the threshold is swept on
those same rows and the best one reported, and `y = present` with the F1 taken
over the present class. The addendum scores F1 on the reject decision, which is
the only reading under which its own sentence, that never rejecting scores zero,
is true. On a suite that is 78 percent present, answering present every time
scores about 0.88 on the present class and exactly zero on the graded one.

## geo_consistency, from the same branch

Re warps the whole reference at the winning site over a three by three pose grid
and takes the best normalized correlation, asking whether the reference refits
here rather than whether the template's quadrants agree with each other. That is
a genuinely different question and worth measuring. Cross validated on the
reject class with matched folds it contributes **+0.0029**, against +0.0114 for
a pose stability feature written the same day. Not merged.

## test1, whose method is sound

`test1` adds ten features and evaluates them with stratified five fold cross
validation on the reject class, and its commit claims 0.90 as a target rather
than a result. Measured over identical records and folds:

| feature added to the shipped fifteen | cv reject F1 |
| --- | --- |
| baseline | 0.7258 |
| wide_nom_ratio | 0.7288 |
| residual_concentration, spatial_scatter, num_blur_levels, rotation_abs, scale_deviation, residual_margin_raw | 0.7258, all six exactly unchanged |
| peak_prom_ratio, peak_to_p50_ratio, peak_to_p90_ratio | 0.7227, all three worse |

All ten together are worth **+0.0075**. The three peak ratio features actively
hurt, which is the prediction the feature separation ranking makes: measured one
at a time, the quadrant features split present from absent at about 77 percent
while the peak splits at 61 and its prominence at 54, so features derived from
the peak carry noise rather than signal and a linear model pays for them.

`spatial_scatter` is worth singling out because it was the one expected to work,
being geometric rather than peak derived. It contributed exactly nothing, which
is most easily read as the quadrant features already capturing the spatial
scatter of the candidate set.

## What this says about the rejection ceiling

Every presence feature measured across the whole campaign contributes at most a
hundredth of cross validated reject F1: peak to correlation energy 0.0000,
geo_consistency +0.0029, the ten v2 features +0.0075 together, pose stability
+0.0114. Reaching the 0.90 bonus gate from a blind measured 0.568 would need
roughly thirty independent features of that size. The gap is missing evidence
rather than missing tuning, and the oracle says why: at severity 4 an impostor
outscores the true site on two pairs in three even when the true pose is handed
to the matcher.
