# The runtime budget read against the reference machine, and the pooled surface evaluation that answers it

The addendum fixes the runtime terms precisely: the reference machine is a
four core x86 CPU with 8 GB of RAM and Python 3.11, the budget is median at
most 5 seconds per pair, and the 20 second timeout zeroes the pair. The
development numbers, 2.85 to 4.5 seconds median, were measured on a laptop
whose cores are faster than the reference class; a four core x86 box
running the pinned software stack measured 6.6 seconds median on the
serial build, which is inside every forfeit bound but above the median
budget on the machine class that matters.

The response uses the reference machine's four cores without changing a
single computed number. The pipeline's cost is hundreds of independent
correlation surfaces per pair, evaluated serially although OpenCV and
scipy release the interpreter lock; a four worker pool now spreads the
quarter resolution prescreen grid, its template construction, the blur
bank, the full resolution rescore and the refinement neighbourhoods, with
results collected in submission order so every insertion order tie break
is untouched. Each surface is the same function on the same input, so the
math is identical element for element.

Validation: the smoke suite is byte identical to the serial build; the 60
pair organiser recipe holdout matches on 59 of 60 rows with every scored
quantity identical to fourteen decimals, estimated core 76.64415584415585,
rejection F1 and calibration auc unchanged; the one differing row is an
absent pair whose rejection stands in both builds and whose score moved by
0.0015 because the faster build cleared a wall clock budget gate and
computed evidence the serial build skipped, the documented budget
adaptivity, in the direction of more evidence rather than less. 69 tests
pass. On the development machine the pool cuts the 60 pair holdout from
about 4.4 to 3.0 seconds per pair at 198 percent CPU; the reference class
number must come from a reference class box and the 6.6 second serial
measurement is the baseline it is expected to beat.
