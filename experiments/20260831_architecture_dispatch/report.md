# Detecting the architecture works; there is nothing profitable to dispatch to

Two proposals were built on separate branches and measured. Detect the die architecture
from the reference and specialise by it, and fuse both architecture specialisations into
one localizer. The detector is real and cheap. Neither use of it earns a place.

## The detector

The reference of a DRAM array is a two dimensional lattice, so its spectrum carries
comparable energy on both principal axes; a FinFET field is a dense family of parallel
lines and concentrates its energy on one. A single number captures that: take the 2D
transform of the mean removed, Hann windowed reference, zero a small box at DC, and
compare the strongest response along the horizontal centre line against the strongest
along the vertical, as min over max. DRAM sits near 0.55 and FinFET near 0.24.

It measures the image rather than the file, so it is a legitimate operation on a blind
set, and it costs one transform of the reference.

Its accuracy is lower and less stable than the first measurement suggested, which is
worth recording because the first measurement is the one that would have been believed:

| suite | pairs | detection accuracy |
| --- | --- | --- |
| p2holdout2, references alone | 108 | 99.1 percent |
| p2train, through the pipeline | 216 | 88.0 percent |
| p2holdout2, through the pipeline | 108 | 96.3 percent |

The misses sit at balances between 0.17 and 0.38, in the genuine overlap between the two
classes rather than at the extremes. A detector wrong on one pair in eight to one in
twenty seven also caps whatever dispatch can deliver, since those pairs receive the other
architecture's treatment.

## Per architecture presence thresholds, declined

The presence threshold is a single global value for both architectures, and the two have
visibly different score distributions: on p2train the median presence probability of a
present DRAM pair is 0.988 against 0.713 for FinFET, while the absent medians are close at
0.263 and 0.243. Rejection is also the weakest scored component, so if a split were to pay
anywhere it would pay here.

Thresholds were fitted on p2train and scored on p2holdout2, never on the same rows.

| configuration | scored on | localization | reject F1 | core |
| --- | --- | --- | --- | --- |
| shipped global 0.45 | p2train | 24.21 | 0.5496 | 32.45 |
| best global 0.350 | p2train | 24.90 | 0.5593 | 33.29 |
| per architecture 0.310 and 0.390 | p2train | 24.90 | 0.5667 | 33.40 |
| shipped global 0.45 | p2holdout2 | 22.49 | 0.4918 | 29.86 |
| best global 0.350 | p2holdout2 | 23.43 | 0.4528 | 30.22 |
| per architecture 0.310 and 0.390 | p2holdout2 | 23.01 | 0.4444 | 29.68 |

On the data it was fitted to, the split is worth 0.11 core points over the best single
threshold. On held out data it is worth minus 0.545, and it also falls below the shipped
setting. The gain was fitting noise and the split is declined.

Three reasons, in the order the evidence supports them. The two fitted optima are 0.310 and
0.390, so the architectures wanted nearly the same boundary and there was little for a split
to exploit. The detector is imperfect, so roughly one pair in nine to one in twenty seven
receives the wrong threshold. And splitting halves the pairs behind each threshold, on a
quantity whose optimum already moves between suites.

The best single threshold on p2train, 0.350, carries plus 0.84 core points there and only
plus 0.358 on the holdout. That shrinkage is the signature of a draw rather than a signal,
and the shipped 0.45 was chosen by a sweep pooled over several suites for exactly this
reason, so it is left alone.

Recorded separately because the mistake is instructive: the first sweep computed F1 over the
present class and reported 0.804 at the shipped threshold. The addendum grades the reject
decision, and on a suite that is about four fifths present, answering present for every pair
scores 0.80 on the present class while scoring zero on the decision being graded. The correct
figure at that threshold is 0.5496. This is the same defect found earlier in a teammate branch
that reported a rejection F1 of 0.9068, and it was reproduced here by the person who found it.

## Fusing both specialisations, declined

The second branch applied both architecture specialisations to every pair regardless of
architecture, the contact emphasis prefilter and the gate emphasis channel together, behind
a single master flag. Measured over the 84 present grayscale pairs of the proportional
holdout, localization only:

| configuration | set A | set B | weighted | of 40 | median runtime |
| --- | --- | --- | --- | --- | --- |
| all flags off, shipped | 0.714 | 0.524 | 0.610 | 24.38 | 3.58 s |
| both channels on | 0.714 | 0.476 | 0.583 | 23.33 | 3.83 s |

It costs 1.05 points of 40 and a quarter of a second per pair. The engineer who built it
predicted between 2 and 4 points of loss, so the outcome is better than forecast and still
firmly negative.

The shape of the loss identifies the mechanism exactly. Set A is unchanged to three
decimals, 0.714 against 0.714, and the entire loss falls on set B, 0.524 to 0.476. That is
the signature already established for the contact emphasis filter measured alone: a
Laplacian of Gaussian is a high pass, degraded captures are shot noise dominated at the
frequencies it emphasises, and clean captures are not. Combining two components that are
individually negative did not repair either of them, and applying a contact matched filter
to an architecture with no contact lattice adds the noise amplification without the
offsetting gain that the dropped contacts give it on DRAM.

## What is worth keeping

The detector itself, as a diagnostic. It is cheap, it is measured, and knowing which
architecture a pair belongs to is useful for reading a scored run even when nothing branches
on it. It stays behind a flag that defaults to off, and the shipped decision path does not
read it.
