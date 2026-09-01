# The organisers' scripts, read line by line, yield two shipped changes and four declines

The shared bundle's generator and scorer were deep read in full. Two findings
ship, one as a localization override and one as a threshold correction; the
training scale refit the plan called for was measured and declined because the
existing model beat every refit; and three smaller ideas were declined or
deferred. The organisers' shared 20 pairs stayed a pure test throughout: no
model, threshold or margin was fitted on them, and every fitted quantity came
from this repository's own suites or from suites this repository generated
itself from the released pipeline under fresh seeds, which the addendum
explicitly allows and its FAQ explicitly recommends for threshold tuning.

## Finding one: the graded F1 is computed on the found class

The addendum's rejection page defines the false positive as a found absent
pair and the false negative as a rejected present pair, which makes found the
positive class, and the organisers' reference scorer implements exactly that
arithmetic: its shipped calibration line, TP 13, FP 0, FN 3 on 16 present and
4 absent, is only consistent with the found class. This repository had been
optimising the reject class reading, whose optimum sits materially higher in
threshold. Swept over four pooled harvests spanning both generators, 438
pairs:

    thr    found graded   reject graded
    0.15      69.69           63.74
    0.35      69.43           65.16
    0.55      68.70           65.32

The found graded plateau midpoint is 0.15, but the threshold ships at 0.35:
that point keeps three quarters of the found graded gain, is positive on all
four suites under the found reading (plus 0.71, 0.88, 1.10, 0.01), and costs
0.16 pooled under the reject reading, inside run noise. The residual
ambiguity, one loose slide sentence that reads reject class against two
pieces of primary evidence for found class, is priced by that hedge rather
than argued away.

## Finding two: the raw full reference confirmation

Their pipeline regenerates every present pair until the global argmax of one
statistic, the full 1000 px reference box filtered by the integer zoom and
warped to search scale, correlated raw against the unprocessed search image,
lands within 3 px of the label with margin at least 0.02. The blind set is
therefore guaranteed solvable by that statistic on present pairs, and this
pipeline never computed it; the bandpass strips the zone scale content it
reads. `locate` now computes it once at the estimated pose, about 15 ms, and
an override moves the answer to its argmax when it disagrees and clears the
same 0.02 their gate floors at. Swept on the three fitting suites at plus
0.81 of 40 (7 rescued, 2 damaged), then judged held out: plus 1.29 on
p2holdout2, plus 0.86 on the fresh organiser recipe holdout, exactly zero on
p2holdout, four rescues and no damage. On the untouched organiser 20 the
override changed exactly one row, p012, the severity 4 pair, from a 426 px
miss to 0.31 px.

## The training scale refit, measured and declined

Three record pools were fitted and compared over the four harvest sweep: the
shipped model (fitted on p2train alone), pooled refits on 324 pairs spanning
both generators, and domain matched refits on the organiser recipe records
alone. The shipped model won the pooled found graded comparison at 69.69
against 69.37 and 69.56 for the pooled refits and 66.30 to 67.01 for the
domain matched ones, which win their own suite at 79.00 and collapse
elsewhere. The raw confirmation block as presence features was declined the
same way, none of three fits beating the shipped model. What transfers is
the model fitted on the harder generator; what does not is any fit that
leans on the easier one.

## Final battery, everything through register.py

    suite                       core     loc      pose    rejF1   foundF1  auc
    p2holdout                   68.46    30.19    17.62   0.792            0.878
    p2holdout2                  58.34    25.33    18.71   0.359            0.891
    amatgen holdout             75.55    35.83    18.85   0.762            0.944
    organiser recipe sample40   77.96    36.83    19.60   0.800            0.953
    organiser recipe hard40     75.77    36.57    18.41   0.750            0.953
    organiser 20 (pure test)    credit 0.988 vs baseline 0.800   0.857   0.970

Printed cores use the pessimistic reject class reading; under the found class
grading every suite sits higher. Localization moved in one direction
everywhere: p2holdout 29.24 to 30.19, sample40 35.54 to 36.83, and the
hardened recipe's five severity 4 pairs at full credit where the same suite
scored 0.80 before the override. p2holdout2's reject graded dip is the
threshold trade already priced above, minus 1.9 under the reading the
evidence disfavours, plus 0.9 under the one it favours.

## Declined and deferred from the code review

Fitting on their shared pairs was never on the table. The decoy strip width
signature their own readme flags as temporary was declined as a feature. The
impulse median trigger sitting exactly at their severity 3 salt and pepper
rate, the box formation template bank entry, the streak row suppression and
the optical bank extension are recorded as candidates with their evidence but
not shipped inside this cycle's measurement budget; the harvest and sweep
infrastructure here reruns them in one pass each if taken up.
