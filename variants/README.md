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

**FinFET.** The layout builder was transposed to the published specification: fins are now
painted on the vertical axis and gate bars on the horizontal, where the inherited builder
had them the other way around. Every coordinate, width, phase and extent moved with the
axis, the standard cell rows became cell bands running along the new fin direction, and the
SRAM block transposed with the field. `layout_info` records `fin_axis` and `gate_axis` in
every `meta.json` so the orientation is auditable without opening the pixels.

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
