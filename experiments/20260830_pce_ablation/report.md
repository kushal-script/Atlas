# Peak to correlation energy, tested as a presence feature and rejected

An outside review proposed replacing the presence decision's peak statistics
with peak to sidelobe ratio and adding peak to correlation energy, on the
argument that a raw correlation value is a poor confidence metric. The first
half was already true of this pipeline: `peak_prominence` is
(peak minus median) over 1.4826 times the median absolute deviation of the
whole correlation plane, which is peak to sidelobe ratio with median and
robust deviation in place of mean and standard deviation. Ours is computed
over the whole plane rather than a twenty pixel window, which matters here
because the failure mode is a periodic mislock a median of 455 px away, far
outside any local window a sidelobe estimate would see.

Peak to correlation energy was genuinely absent, so it was added:
peak squared over the mean squared response, entered as log1p into the
logistic model, and the model refitted on freshly harvested training records.

## Ablation

The refit's cross validated reject F1 rose from the shipped model's number,
but the same commit had revived `pose_wide`, a feature that had been constant
zero because it compared the pose source against "wide" while the localizer
emits "wide_grid". Attribution therefore required an ablation over identical
records and identical folds.

| features | cv reject F1 |
| --- | --- |
| 16, pce and a live pose_wide | 0.7040 |
| 15, live pose_wide, no pce | 0.7040 |
| 15, pce, no pose_wide | 0.6931 |
| 14, neither | 0.6931 |

Peak to correlation energy contributes 0.0000. The entire gain, 0.0109, belongs
to the revived feature. The metric is redundant against prominence, which
already asks whether the peak stands apart from the surface it sits on, and its
class medians barely separate, 27.7 for present pairs against 25.2 for absent.

It was removed rather than carried at zero value, and the finding is recorded
in `localize.py` where the computation would otherwise be re derived.

## The proposal's other half was not adopted either

The same review recommended writing the raw ratio into the score column. That
would regress the calibration component: the addendum scores the area under the
score column against per pair correctness, and a correct rejection is correct,
so it has to rank high. A raw peak ratio ranks confident rejections low, which
inverts the ordering for the forty absent pairs. The shipped score is confidence
in the decision actually made, which measures an area under the curve of 0.925.
