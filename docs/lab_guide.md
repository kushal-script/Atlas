# Using the localizer in a laboratory

This guide is for a scientist or tool engineer who wants to run the
registration on their own microscope's captures, read its answers correctly,
and adapt its learned decision to their instrument. Everything here is the
shipped behaviour; nothing needs source changes.

## Install

Python 3.11 is the reference interpreter. From the repository root:

    python3.11 -m venv .venv
    .venv/bin/pip install -r requirements_phase2.txt
    .venv/bin/pip install -e .

The editable install makes `import drift_sense` work from any directory; the
entry points `register.py` and `localize.py` add `src` to the path themselves
and run without it.

## One pair from Python

    from drift_sense.api import register_pair, load_presence_model
    from drift_sense.localize import load_gray

    model = load_presence_model()
    reference, ref_rgb = load_gray("reference.png")
    search, search_rgb = load_gray("search.png")
    result = register_pair(reference, search, reference_rgb=ref_rgb,
                           search_rgb=search_rgb, model=model)

    result.found        # True when the reference is present in the search
    result.x, result.y  # match centre, search pixels, origin at the centre of the top left pixel
    result.theta_deg    # rotation of the reference as it appears in the search, counter clockwise positive
    result.scale        # magnification ratio, nominally 8 to 12
    result.score        # confidence in the decision made, higher is more trustworthy
    result.regime       # unique_peak, residual_identified or tie_break_convention
    result.reason       # matched, rejected, optical_disclosed or degenerate_input
    result.diagnostics  # every statistic the decision read, for logging or audit

`register_pair` is the same function the batch entry point calls for every
row, so a library caller and the batch CSV can never disagree; the unit test
`test_end_to_end.py` asserts that on generated pairs. `load_gray` accepts 8
bit, 16 bit and float images and reports whether the capture carried colour;
pass the flags through so an optical pair takes the optical preset.

## A batch from the shell

    .venv/bin/python register.py --input pairs.csv --output predictions.csv

`pairs.csv` needs a `pair_id` column and two path columns named
`reference_path` and `search_path` (several aliases are accepted); relative
paths resolve against the CSV's own directory. The output is one row per
pair, `pair_id,x,y,theta,scale,found,score`, flushed to disk as each pair
finishes so an interrupted batch keeps every finished row. A pair reported
absent carries zeros in its pose columns.

## Reading the answer

The found flag is the decision. A present verdict with `regime` equal to
`unique_peak` means one correlation peak dominated; `residual_identified`
means the surface was periodic and the site unique deviation field singled
out one candidate; `tie_break_convention` means nothing decided and the
candidate nearest the search centre is reported, which is the specified
convention for a genuine tie and the regime to treat with the least trust.
The score ranks decisions: a found row scores between 0.25 and 1.0, damped
by how many of the four template quadrants independently agreed on the
site; a rejected row scores between 0.5 and 1.0, higher when the rejection
was confident. A `degenerate_input` reason means a blank or constant frame
was refused before any matching ran.

## Runtime and limits

A pair takes about two seconds at the median with the pose surfaces
pooled over the machine's four cores and no accelerator, with a tail
observed to about six and a half seconds, a 15 second internal budget that
trims the search on a slow machine rather than overrunning, and an 18
second alarm on Unix. The
search covers magnification 8 to 12 and rotation within 5 degrees of
nominal, the ranges the problem states; a capture outside them is reported
at the nearest edge of the searched range. The information limit is real:
under heavy dose reduction a periodic array's true site can lose the
correlation to a neighbour on the physics alone, measured with the true pose
supplied, and the correct response is the found flag rather than a
confident wrong site. Decoy sites with identical macro geometry and
different random structure are the recorded weakness of the presence
decision; `docs/phase2_failure_analysis.md` prices it.

## Adapting to a new instrument

The localizer's geometry is fixed by the physics of the capture and needs
no tuning. The presence decision is a small logistic model fitted on this
repository's generated pairs, and an instrument whose noise or contrast
differs materially may warrant a refit. The workflow, all on your own
labelled pairs:

1. Arrange pairs as a suite with a `ground_truth.csv` in the format
   `docs/dataset_format.md` describes, with `found`, `gt_x`, `gt_y`,
   `gt_zoom` and `gt_rotation_deg` per pair; a generated suite from
   `scripts/generate_phase2_suite.py` shows the layout.
2. Record the localizer's diagnostics over the suite:
   `scripts/tune_phase2.py --dataset <suite> --name mytool --no_raw_override`,
   which writes `experiments/<stamp>_mytool/records.json`.
3. Fit: `scripts/fit_presence.py --records <records.json> --features v2 --out models/presence_model.json`.
   Keep the fitting suite separate from any suite you will judge on.
4. Judge before shipping: `scripts/eval_presence_models.py` scores any
   model over any recorded suites under the exact scoring the batch entry
   point earns, and sweeps the threshold on a pool you name.
   The discipline this repository followed, and recommends, is to ship a
   refit only when it wins on every held out suite separately.

The raw confirmation override in `MatchConfig` assumes the reference and
search share a formation the full reference correlation can read at the
estimated pose; on an instrument where that is not true, set
`raw_override=False` and the pipeline falls back to its classical answer
with nothing else changed.

## Configuration surface

`MatchConfig` in `src/drift_sense/localize.py` holds every constant with a
one line meaning beside it. The ones a laboratory might touch: `zoom`, the
nominal magnification ratio; `coarse_scales` and `coarse_rotations_deg`, the
searched pose ranges; `time_budget_s`, the per pair budget;
`raw_override` and its `raw_override_margin`, the confirmation override and
its floor; `device`, `cpu` by default with `cuda` or `mps` available when
torch is installed. Any change to these is a change to what is searched and
should be re measured with the evaluator above before it is trusted.
