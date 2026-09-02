# The decoy weakness taken to its decision boundary: the signal found, five deployments measured, all declined

The recorded residual weakness is the identical geometry decoy class, absent
pairs whose search contains a zone with the right macro geometry and wrong
random structure, grabbed at about six in ten with rejection F1 0.560 on the
decoy suite at the shipped operating point. This campaign set out to solve
it rather than price it, found the discriminating signal, and measured every
deployment of it this repository's protocol permits. None ships. The
difference from before is that the boundary is now measured rather than
assumed, on 876 fresh pairs across eight suites.

## The signal exists, and where it lives

`_residual_score_map` already computes the absolute deviation correlation of
the winning site, the normalized match between the reference's site unique
content and the window's, which is exactly the question a decoy fails: it
has the geometry but not the structure. The value was computed and discarded;
the harness now records it (`dev_best`, `dev_median`) and `presence.py`
exposes it as a named feature block. Where the residual stage runs, present
against absent separation of `dev_best`:

    stress harddecoy       d' 1.68   present 0.368, decoy absent 0.211
    amatgen train h        d' 1.84   organiser recipe
    amatgen holdout        d' 1.79   organiser recipe
    amatgen train s        d' 1.53   organiser recipe
    p2train                d' 0.15   own generator
    p2holdout              d' 0.31   own generator
    p2holdout2             d' 0.23   own generator
    p2train, low noise slice   d' 0.02

The signal is real and regime bound: strong on the organiser recipe and on
the decoy class, absent on this repository's own harsher generator, whose
impostor references match deviation content about as well as true sites
even at low noise. That last row matters most: it is not noise burying the
signal, the signal is not present in the fit domain at all.

## Five deployments, judged per suite under the authoritative scoring

`eval_models.py`, shipped as `scripts/eval_presence_models.py`, rebuilds every decision offline exactly as register.py
makes it, override at 0.05 applied to effective errors, localization gated
on found, pose gated on localization, both F1 readings, calibration from
the shipped score construction; its baseline row reproduces the committed
battery to the second decimal on every suite. Candidates fitted on fresh
p2train records (the transfer direction the September campaign measured as
the only one that works) and on the pooled fitting records, judged on seven
suites, reject reading core against the shipped model:

    fit on p2train, dev block (v5):        decoy F1 0.560 to 0.538, worse
    fit on p2train, raw block (v4):        amath minus 0.73, amats minus 0.63
    fit on p2train, raw plus dev (v6):     amath minus 0.61, three suites down
    pooled fit, 18 features:               decoy F1 0.667 but b180 minus 1.88
    pooled fit, dev variants:              no better than pooled 18

The p2train fit assigns the dev features backwards weights because the fit
domain carries no signal; the pooled fit deploys them correctly and lifts
the decoy suite by a full point of F1, but pays for it on the blind sized
own generator suite, the same transfer asymmetry that declined pooled
refits in September, now reconfirmed on fresh records with the new
evidence included. The raw confirmation block as presence features is also
re declined on fresh records, its September decline having rested on stale
margins. Under the rule that a change ships only when it wins without
damaging anything, nothing here ships.

## The score construction, measured instead of argued

The fifth audit proposed damping found row scores by raw or re ranker
disagreement. Three constructions were measured against the shipped
quadrant damp over the seven suites, per suite calibration AUC:

    quad (shipped)     0.933  0.869  0.905  0.914  0.941  1.000  0.989
    quad and raw       0.906  0.894  0.919  0.959  0.941  1.000  0.984
    quad and rerank    0.947  0.881  0.913  0.947  0.915  1.000  0.982
    quad, raw, rerank  0.914  0.896  0.920  0.959  0.915  1.000  0.979

Every alternative wins some suites and damages others, b180 or the decoy
suite among them each time; the shipped damp is the hedge, now measured
rather than reasoned.

## Runtime, profiled to its floor

One 2.2 s pair under cProfile: 0.86 s in the calibrated FFT correlations,
0.53 s in the Gaussian blur bank, 0.26 s in raw confirmations, 0.23 s in
template warps. Every remaining reduction, approximate large sigma blurs,
merged bank sigmas, fewer pose surfaces, changes predictions, which trades
scored accuracy for speed. The identical math floor is reached; what
protects slow machines is the 15 s budget, the 18 s alarm and the row
flushed writer against the 20 s forfeit, and the README says so.

## What ships from this campaign

The instrumentation and the boundary. The harness records the deviation
evidence, `presence.py` names it, the fitted candidates and the judging
logs are committed beside this report, and the shipped model, threshold
and score construction are unchanged, byte identical smoke and 60 passing
tests. The decoy weakness remains priced at the operating point, and its
solution is now a measured impossibility under linear deployment on this
protocol rather than an open question: the discriminating signal is real,
lives exactly where the blind set's generator lives, and any future
deployment needs either decoy class examples in the fitting domain or a
regime conditional model, both of which this repository's own transfer
measurements currently veto.
