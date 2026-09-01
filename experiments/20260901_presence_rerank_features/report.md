# The rerank combiner's three diagnostics earn a place in the presence model

## Protocol

Both arms fit the same class weighted logistic with the same L2, the same
five fold stratified cross validation for threshold choice, and the same total
score proxy (localization 40 plus rejection F1 15 plus decision AUC 10, pose
omitted because it is a mean over already credited pairs and moves only
through composition), on the 216 SEM pair p2train harvest, judged on the 108
SEM pair p2holdout2 harvest. The only difference between the arms is the three
combiner features, so the delta is the value of the features and nothing else.
Harvest provenance was verified by triangulating each record's ground truth
from its candidate error radii: 84 of 84 present holdout records match
p2holdout2 and zero match p2holdout.

## Result

    v1 fifteen features:  holdout total 33.58  loc 22.48  F1 0.263  auc 0.716
    v2 plus rerank block: holdout total 34.31  loc 21.63  F1 0.293  auc 0.829
    held out delta v2 minus v1: +0.73 proxy points (F1 +0.030, auc +0.114)

The AUC gain of 0.114 is threshold free and is the strongest signal: the
combiner's score, margin and agreement rank decision correctness better than
the fifteen surface diagnostics alone. Fitted weights on the three features
are all positive and material (rr_score +0.675, rr_margin +1.409, rr_agree
+0.603).

## Caveat that decides the next step

Both arms' absolute F1 (0.263 and 0.293 at thresholds 0.21 and 0.28) sit far
below the shipped model's 0.655 on the same suite, because this quick refit
protocol chooses its threshold on one suite where the shipped protocol pools
three. The feature value is proven here; the ship decision is made only by
refitting through the shipped protocol (tune_phase2 records, fit_presence,
tune_threshold pooled sweep) and scoring end to end through register.py.
