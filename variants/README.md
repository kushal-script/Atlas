# Architecture specific variants

Two self contained pipelines, one per die architecture, built so that a matcher tuned for
one architecture could be measured against the shipped pipeline that handles both. Neither
tuning survived measurement; see `experiments/20260831_architecture_variants`.

```
variants/dram/     register.py, generate_dataset.py, models/, src/drift_sense_dram/
variants/finfet/   register.py, generate_dataset.py, models/, src/drift_sense_finfet/
```

Each is a complete clone with its package renamed. Intra package imports are relative, so
nothing needed rewiring; only the two entry points named the package absolutely. Neither
variant imports the other, and neither imports the shipped `drift_sense`. Nothing here is
part of the submission and nothing here modifies it.

Run either exactly as the shipped entry point is run, from its own directory:

```
cd variants/dram
../../.venv/bin/python generate_dataset.py --num 40 --out data/dram40 --seed 1
../../.venv/bin/python register.py --input pairs.csv --output predictions.csv
```

## What differs from the shipped pipeline

**Both.** The generator produces one architecture only. The other architecture's builder is
deleted rather than merely unselected, so the variant cannot silently fall back to it, and
`--style` is gone from the command line.

**FinFET, and a correction.** This variant briefly carried a transposed layout
builder, on the reading that painting fins with `paint_hstripe` meant the rendered fins
ran horizontally while the specification calls for vertical ones. That reading was wrong
and the transpose has been reverted. The capture transposes the canvas, so a stripe
painted along the canvas u axis renders as a vertical line in the image: measured on the
2D spectrum of the reference, the shipped generator puts its dominant line family at 90
degrees on five of five pairs at the fin pitch, which is the vertical fin array the
specification describes, and the transposed builder put it at 0 degrees on five of five.
The shipped generator was already conformant and the transpose moved away from the
specification rather than toward it.

The lesson is worth more than the fix. The canvas convention and the image convention are
opposites here, so `geometry/finfet.py` and its docstring describe a horizontal fin grid
while the images contain a vertical one, and both statements are true of their own frame.
Anything that reasons about orientation must be settled in the pixels.

**Matcher specialisations, both disabled.** `MatchConfig.contact_emphasis` on the DRAM
variant and `MatchConfig.gate_emphasis` on the FinFET variant are present, documented and
default to off, because each was measured and each costs credit. The code is kept so the
measurements can be reproduced by setting the flag true. With both off, each variant
reproduces the shipped entry point byte for byte.

## Why the architectures differ, which is the useful finding

Handed the true pose so that search is removed from the problem, the correlation still
prefers an impostor on 39 percent of nominal FinFET pairs and 76 percent of degraded ones,
against 17 and 52 percent for DRAM. The gap is in the evidence, not in the search, so it is
not reachable from the matcher. The mechanism is in the generator: the DRAM builder drops a
small random fraction of contacts and varies the size of the rest, and those defects are
the only aperiodic content in an otherwise perfectly regular field. The FinFET builder has
no equivalent, so every cell is identical to its neighbours.
