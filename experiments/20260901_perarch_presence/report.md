# Fully separate per architecture presence models, measured and declined; the architecture enters as a covariate instead

## What was asked

The August 31 dispatch experiment declined per architecture presence
thresholds at minus 0.545 held out, but that split shared one fitted model and
moved only the boundary. The stronger design fits an entire presence model per
detected architecture, so every weight can specialise, and routes at runtime
by the lattice balance detector. This measures that design at harvest level,
routed by the DETECTED architecture on both suites, never the generator label,
because the blind set carries no label.

## Result (fit on p2train harvest, judged on p2holdout2 harvest)

    arm 1 one shared eighteen feature model:      total 34.31  F1 0.293  auc 0.829
    arm 2 shared model, per architecture thresholds: total 37.16  F1 0.511  auc 0.745  (+2.85)
    arm 3 fully separate models per architecture:    total 35.19  F1 0.409  auc 0.743  (+0.88)

## Why this does not overturn the August 31 decline

Arm 2's plus 2.85 is over a deliberately weak single suite baseline whose
threshold (0.28) lands far from the shipped pooled operating point; the
shipped protocol reaches F1 0.655 on this same suite. What the split mostly
does here is repair a deficiency of the quick protocol's threshold choice, not
beat the shipped one; under the shipped protocol the same split measured minus
0.545 held out. Three mechanisms, each literature backed and each consistent
with this repository's measurements, predict the full model split loses at
this data size: specialisation pays only when the classes demand materially
different predictors, and the two fitted threshold optima were 0.310 and
0.390; hard routing halves 216 pairs to 127 and 89 with about two dozen
absents behind 18 coefficients each; and the detector's misses concentrate in
the genuine overlap (balances 0.17 to 0.38), which is exactly where the
presence decision is hardest. Both arms also lose 0.086 of AUC, since a
threshold split contributes nothing to ranking and the halved fits rank worse.

## What ships instead

The literature's soft alternative: the continuous lattice balance value joins
the single pooled presence model as a covariate, beside a period aware second
peak ratio and the peak curvature, so the model can move its own boundary
continuously with the architecture evidence instead of a hard router moving it
for two half sized models. Measured through the shipped protocol in the
presence refit that follows this experiment.
