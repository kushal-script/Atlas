# Variance stabilisation and an extreme value null, both declined

Two proposals from the literature survey were implemented and measured against
the shipped pipeline. Neither earns a place in it. One is declined because its
premise is not satisfied by the data, the other because what it measures is
already measured. Both are recorded here with the evidence, because a reader who
meets these ideas in the same papers deserves the reason they are absent rather
than the impression they were never considered.

## The generalised Anscombe transform

Electron counting is a Poisson process, so the noise in a raw detector frame has
a variance equal to its mean. Normalised cross correlation assumes constant
variance, so the standard remedy is to apply `2 sqrt(z + 3/8)`, which maps
Poisson noise to unit variance and restores the assumption.

The premise was tested before the transform was trusted. Local variance was
measured over 56250 flat eight by eight tiles drawn from twelve search captures,
binned by local mean, keeping only the flattest thirty percent of tiles so the
quantity measured is noise rather than structure.

| mean bin | local variance | variance / mean |
| --- | --- | --- |
| 19 to 68 | 721.64 | 16.337 |
| 68 to 122 | 1194.16 | 11.429 |
| 122 to 135 | 1213.58 | 9.413 |
| 135 to 144 | 1163.74 | 8.328 |
| 144 to 153 | 1140.34 | 7.695 |
| 153 to 181 | 1124.98 | 7.089 |

Under a Poisson law the final column is constant. It is not: it falls
monotonically by a factor of 2.3 while the raw variance stays nearly flat, 1043
in the darker half of the range against 1143 in the brighter, a ratio of 1.10
across an eightfold change in brightness. The eight bit tone map applied during
capture has already equalised the variance the transform exists to equalise, so
the data reaching the matcher is close to homoscedastic before any correction
and the transform can only bend it the other way.

The end to end A/B that was run alongside this is reported for completeness but
is **not** evidence about the transform. With the transform enabled every pair in
the suite was rejected, 50 of 50 nominal and 33, 43, 42 and 32 of the four
degraded severities, for a credit of exactly zero. That is a presence failure,
not a matching failure: the presence model is fitted on untransformed features,
so rescaling every input moved all fifteen features off the range the model was
fitted on and the decision collapsed to a constant. A fair test would require
refitting the presence model on transformed features, which is not worth the cost
given the table above shows the premise itself does not hold.

Reported honestly: the transform is not shown to harm matching. It is shown to be
answering a question this data does not ask.

## A Gumbel null for the correlation peak

The maximum of many correlation samples follows an extreme value law, so a peak
can be scored by how far it sits into the tail of a Gumbel fitted to the rest of
the surface, rather than by its raw height. Fitted by method of moments with the
peak neighbourhood excluded, this separates the classes well:

| statistic | present median | absent median | separation |
| --- | --- | --- | --- |
| gumbel z | 7.402 | 4.281 | 1.02 sd |
| gumbel log p | -3.215 | -1.862 | 1.02 sd |
| peak prominence | 7.173 | 4.938 | 0.52 sd |

Twice the separation of the prominence feature already shipped, which is why it
was worth carrying to an ablation. The ablation prints the correlation against
the existing features before the cross validated result, because separation is
not the same as new information:

| existing feature | correlation with gumbel z |
| --- | --- |
| prom_l | +0.856 |
| over_p99 | +0.709 |
| peak | +0.239 |
| wide_l | -0.199 |
| nominal | +0.198 |

At r = 0.856 against the log prominence feature, the new statistic is very
nearly a monotone rescaling of one already present. Over identical records and
folds the cross validated reject F1 is 0.7040 on the shipped fifteen features and
0.7037 with the sixteenth added, so it contributes -0.0003 and is removed.

The separation was real. The information was not new. This is the third feature
to fail in exactly this way, after peak to correlation energy and spatial
scatter, and the pattern is now firm enough to state as a rule: any candidate
derived from the correlation surface peak is a rescaling of the peak features
already present, and the features that carry independent signal are the ones
computed from a different construction, such as the quadrant agreement statistic
which separates at 3.0 sd and splits the classes at 77 percent.
