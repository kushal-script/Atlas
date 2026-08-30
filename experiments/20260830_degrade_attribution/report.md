# What actually breaks the severe set, measured factor by factor

The degraded set's collapse at severity 4 had been described as information
limited, which is a claim about physics rather than about code, and it had
never been measured. The severity ladder pushes six factors at once, so no
attribution existed: dose, beam spot, charging, drift, jitter and polygon
scaling all move together and the credit only reports their sum.

## Method

`scripts/degrade_attribution.py` generates paired suites from identical seeds,
so every suite renders the same twenty specimens with the same lattices and the
same pose, and differs only in the degradation applied. One suite is the control
under the full severity 4 ladder; each of the others restores exactly one factor
to its nominal value. Localization runs without the presence decision, so the
number reported is the matcher's own behaviour rather than a threshold's.

## Result

| factor restored | credit | delta | share of the attributable loss | invertible |
| --- | --- | --- | --- | --- |
| electron dose | 0.550 | +0.350 | 43.8 percent | no, quantum |
| beam spot | 0.370 | +0.170 | 21.2 percent | no, optics |
| scan jitter | 0.310 | +0.110 | 13.7 percent | yes, structured |
| polygon scaling | 0.270 | +0.070 | 8.8 percent | partly, the width rescue |
| drift | 0.170 | -0.030 | none | moot |
| charging | 0.150 | -0.050 | none | moot |

Control credit is 0.200 with a median error of 500 px. Restoring dose alone
drops that median to 1.22 px, which is the single most informative number here:
with the beam spot still tripled, charging quadrupled, drift at ten pixels,
jitter at six and polygon widths off by a fifth, the matcher localizes to about
one pixel as soon as the electrons are there to see with.

## What follows from it

Sixty five percent of the whole loss is shot noise and optical blur, seventy four percent of the part any single factor explains; the twelve percent gap is interaction this design cannot split. Neither is
corruption applied to information that exists; both are information that was
never collected, so no filter, deconvolution or learned prior recovers it. The
remaining twenty six percent is structured and inverts in principle.

That quarter was priced before deciding. Perfect jitter inversion lifts severity
4 credit from 0.200 to at most 0.310, severity 4 is about a fifth of the
degraded set, so the ceiling is roughly half a localization point, and a row
shift estimator would have to measure displacements from the same shot noise
that dominates the loss. It was not built.

Drift is worth recording separately because it is the factor an outside reading
of the problem argues hardest for. It measures nothing at all, and the physics
says why: the ladder's drift is ten pixels across a thousand row capture, which
is about one pixel inside the hundred pixel template footprint the matcher
actually reads.

## Files

`per_pair_errors.csv` one row per suite and pair with the localization error.
`report.md` this file. The generator patch lives in
`scripts/degrade_attribution.py` and restores the ladder on exit.
