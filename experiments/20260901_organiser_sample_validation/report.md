# The entry point validated on the organisers' own material: conventions proven, 0.925 against their baseline's 0.800, and an estimated 74.7 to 77.2 of 85 on their recipe

## What was shared and how it is used

On September 1 the organisers shared their Phase 2 bundle: 20 pairs with
ground truth, the jury parameter manifest, their naive ZNCC baseline's
calibration run, and the full generator source including the Phase 2 pipeline
and the exact severity parameter ladder their readme says was deliberately
kept out of participant docs. The bundle is marked confidential, lives outside
version control (`OneDrive_1_1-9-2026/` is ignored), and everything below
treats it under the mentor's rule: validation only. No model, threshold or
parameter of the submission was fitted, adjusted or selected on any of it.

## Conventions proven end to end on organiser data

Their `canvas_to_search_affine` is exactly the closed form this repository's
conventions were verified against on August 31, and the empirical check now
closes the loop: over the 16 present pairs the entry point's theta matches
their ground truth in sign and magnitude on every credited pair (worst
rotation error 0.27 degrees, on the one pair whose hand specified pose their
own generator rounded), and the scale column tracks z at both endpoints
(8.01/8.00 and 11.95/12.00). The convention risk that would forfeit all 24
pose points is retired on the organisers' own labels.

## The shared 20 pairs, against their own baseline

Their brute force ZNCC baseline (0.5x, 1 degree grid) is the reference row in
`baseline_calibration.txt`. Ours beside it:

    set                     their baseline     this entry point
    A nominal (8)           1.000              1.000  (all under 1 px)
    B degraded (6)          0.467              0.800
    D optical (2)           1.000              1.000
    present overall         0.800              0.925
    pose, worst credited    3.0 percent, 1.10 deg   under 1 percent, 0.27 deg
    reject class F1         0.727              0.857

Both severity 3 pairs their baseline scores zero on (p011, p012's neighbour
p014) localize at full credit here, at 0.56 and 0.42 px. The two misses are
informative rather than surprising: p012, the single severity 4 pair,
mislocalizes by 426 px, which is the information limited regime the oracle
probe already bounded on our own data; and p015, one clean absent pair of
four, is grabbed at score 0.750. No false rejects.

## Fresh suites from their generator, their recipe at scale

Their driver is deterministic and seedable, so two 40 pair suites this
repository had never seen were generated from their pipeline under new seeds:
one faithful to the shared sample's composition, and one following the jury
recommendation recorded in their own readme for the real 200 pair set, Set A
raised to severity 1 and Set B shifted toward severities 3 and 4. Both mirror
the blind composition of 35/35/20/10 and cycle all nine architecture presets.

    recipe            loc/40   pose/20   reject F1   auc     estimated core/85
    sample faithful   35.54    19.86     0.800       0.981   77.21
    jury hardened     35.00    18.75     0.750       0.974   74.74

Per severity on their physics: severity 3 earns 0.83 to 0.95 credit and
severity 4 earns 0.50 to 0.80, against 0.16 to 0.21 for severity 4 on this
repository's own generator. The self graded suites were built on harsher
physics than the organisers', which is the survivable direction of the
mismatch the mentor briefing warned about: a matcher tuned on forgiving noise
looks good until the blind set, while one tuned on punishing noise meets a
blind set that is easier than its training diet. The estimated core on their
recipe sits 7 to 10 points above the 67.62 measured on our own holdout.

## What this changes in the submission

Nothing, which is the point. Both residual weaknesses their data shows,
severity 4 localization and the occasional confident grab of an absent pair,
were already known from our own suites and already shaped the operating
point; their data confirms the standing rather than motivating a change, and
the rule that nothing is fitted on organiser material stands. The one
actionable note for any remaining time is that their imaging chain carries
degradations ours does not render, vignetting, gamma, astigmatism and barrel
distortion, and the measured credit above says the pipeline absorbs them as
built.

Files here: `pred_amat20.csv` is the entry point's output on the shared 20;
`generate_from_organiser.py` and `score_against_gt.py` reproduce the fresh
suites and their scores when the bundle is present; `gt_gen_*.csv` and
`pred_gen_*.csv` are the generated suites' truths and our outputs.
