# Final adversarial audit: five dimensions, ten findings, six confirmed, every one acted on or closed by measurement

Before submission the repository was audited once more, this time by five
independent auditors each owning one dimension, overfitting and leakage,
hackathon rules and hardcoding, physics authenticity, laboratory readiness
and input robustness, with a skeptic assigned to refute every finding
before it could be acted on. Ten findings came back; the skeptics refuted
four and confirmed six. The refutations are recorded because they matter
as much as the confirmations: the threshold and floor choices were checked
against the addendum's own allowed list and stand, the organisers' shared
sample was confirmed never fitted on, the headline rows were traced to
judging suites, and the colour flag was confirmed to match the organisers'
documented modality signal.

## Confirmed and shipped

Sixteen bit containers holding eight or twelve bit data. `load_gray` shifted
every uint16 image down eight bits, so an eight bit capture saved into a
sixteen bit PNG loaded as an all zero frame and was rejected as blank, and
a twelve bit detector range collapsed to sixteen levels. The coercion is
now range aware, identity below 256, rescaled below 4096, shifted above,
and three tests lock each case. No shipped prediction changes: the uint8
path is untouched and the full range sixteen bit test still takes the
shift.

Dielectric charging was a no op on DRAM. The charging mask keyed on the
isolation oxide material, which the DRAM builder never paints, so at any
amplitude the multiply was the identity on every DRAM specimen; a render
at seed 8501 confirmed byte identical DRAM captures across the whole
charging range. The mask now covers nitride as well, the same render moves
the DRAM search image by 39 gray levels mean at the ladder cap, the
generator tests pass, and the attribution claim that charging measures
nothing is corrected everywhere it appeared: it measured nothing on a
design in which twelve of twenty pairs could not respond.

The archive omitted the training scripts. The presence model's fitting
script and its records harness were not packed, and three generator modes
the packed README advertises delegated by subprocess to scripts the archive
did not carry, so `generate_dataset.py --generator stress` crashed from a
clean extraction. All of them ship now, with the model evaluator and the
laboratory guide.

The anti aliasing claim. The README, the architecture note and the failure
analysis said every template blur folds in the magnification ratio's box
filter; the option exists, was measured in the August 12 template ablation,
declined, and ships off, and only the development log recorded that. Every
claim is corrected, the ledger carries the decline, and the raw
confirmation stage, which does apply the organisers' box filter to the full
reference, is named as where the matched formation actually lives. The
slide deck's two sentences making the same claim are corrected.

The laboratory refit guide pointed at an undocumented schema and an
unpacked script. `docs/dataset_format.md` now documents every
`ground_truth.csv` column the harness and fitter read, and the evaluator
ships as `scripts/eval_presence_models.py`.

## Confirmed, measured, and declined

The presence model's fit domain. The shipped model was refit on the first
p2train suite, whose 48 absent pairs are all clean captures, and carries
its largest weight on measured noise; the failure analysis claimed the
generator fix had removed that shortcut, which is false at head, since the
refit predates the fix. The remedy the finding implies was measured: the
same 18 features fitted on p2train2, 240 pairs whose absents span
severities 0 to 4, judged on the eight other fresh suites under the
authoritative scoring at both the shipped threshold and its own pooled
sweep midpoint:

    suite     shipped   refit on degraded absents
    p2train    65.69     61.30
    b180       63.69     61.12
    h1         68.35     65.39
    h2         58.23     60.67
    amath      76.64     75.42
    amats      79.86     78.22
    amath_t    79.08     76.45
    decoy      73.79     70.48

The refit wins one suite of eight and loses the rest by 1.2 to 3.3 core
points, the organiser recipe suites included, whose absents are degraded
exactly as the blind set's are. The noise feature is not a shortcut that
inverts on degraded absents; it is presence evidence at this operating
point, and the fit domain the shipped model stands on is now backed by
this measurement rather than by a claim. The prose that said the shortcut
was removed is corrected to say what is true: the weight is large, the
refit that would remove it was measured and declined, and the organiser
recipe suites, absents degraded, are where the shipped model reads its
highest rejection F1.

## Refuted, with the reasoning kept

Threshold tuning on fresh organiser recipe suites: the addendum's allowed
list names regenerating one's own dataset to tune the found threshold, the
shared sample stayed a pure test, and the reports say so.

Battery rows drawn from the threshold pool: the released recipe 40 pair
rows are judging suites whose coordinates match none of the fitting
suites, the five post freeze seeds and the organiser sample sit outside
every pool, and the README names the pool.

`match_pair` reporting grid units: it is the documented Phase 1 matcher,
the README and the guide send Phase 2 callers to `register_pair`, and its
docstring now states the units and the sign.

The colour flag firing on a single coloured pixel: the organisers' format
note fixes the channel count as the modality signal, every shared
grayscale frame is single channel, and a fraction rule would risk the
disclosed optical bonus on a faintly coloured pair to defend against an
input the contract excludes; declined as an asymmetric trade.

## State at the end

The batch entry point now delegates every pair to `register_pair` in
`drift_sense.api`, so a library caller and the predictions CSV are one
function; the 60 pair organiser recipe holdout is byte identical before
and after the extraction at core 76.64, the smoke suite is byte identical
through every change in this round, the package installs with `pip
install -e .`, and the test suite covers the entry point end to end on
generated present, absent and optical pairs, the library and batch
agreement, the override floor bands and the bit depth cases.
