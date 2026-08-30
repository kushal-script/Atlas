#!/bin/bash
# Serial on purpose. The localizer's wall clock guard skips optional stages
# under load, so concurrent runs rank configurations wrongly.
cd /Users/kushalsathyanarayan/Desktop/semicon_india_hackathon/PS2
P=.venv/bin/python
D=data/p2degraded
S=/private/tmp/claude-501/-Users-kushalsathyanarayan-Desktop-semicon-india-hackathon-PS2/cf7fa46e-912b-4c4d-8a06-5ae3ccbee8ba/scratchpad

until [ -f $D/ground_truth.csv ]; do sleep 20; done
echo "GEN_READY"

$P scripts/eval_degraded.py --dataset $D --dump $S/d_base.csv \
   --label "1 baseline          bank 4,9,16,25       k6"
$P scripts/eval_degraded.py --dataset $D --wide_bank 4,9,16,25,36 \
   --label "2 bank only         bank 4,9,16,25,36    k6"
$P scripts/eval_degraded.py --dataset $D --top_k 8 \
   --label "3 budget only       bank 4,9,16,25       k8"
$P scripts/eval_degraded.py --dataset $D --wide_bank 4,9,16,25,36 --top_k 8 --dump $S/d_bank.csv \
   --label "4 bank and budget   bank 4,9,16,25,36    k8"
$P scripts/eval_degraded.py --dataset $D --wide_bank 4,9,16,25,32,42 --top_k 9 \
   --label "5 wider bank        bank 4,9,16,25,32,42 k9"
echo "SWEEP_DONE"

# The runtime question, on the proportional holdout and with the absent pairs
# included, since a weak peak triggers the width rescue and those are the
# slowest pairs in the scored mix. Four threads approximates the scored
# machine's core count on a development machine that has eight.
$P scripts/eval_degraded.py --dataset data/p2holdout --sets A_nominal,B_degraded,C_absent \
   --label "runtime all 8 threads"
$P scripts/eval_degraded.py --dataset data/p2holdout --sets A_nominal,B_degraded,C_absent --threads 4 \
   --label "runtime capped 4 threads"
echo "ALL_DONE"
