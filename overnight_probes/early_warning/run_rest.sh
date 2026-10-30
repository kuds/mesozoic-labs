#!/bin/bash
# Restart after the ECS phase was stopped at 01:06 UTC (its first rows were lost; see NOTES).  One simulation process
# at a time, nice 15, single-threaded: the three r10 finals, then the ECS s80 and s42 lineages.
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
H=$S/com_revisit/notes/early_warning
cd $H
export PYTHONPATH=$S/r10round/ecs/code:$H OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
date -u +%FT%TZ > $H/run_rest.started
nice -n 15 $S/venv/bin/python -B ew_measure.py ew_r10.jsonl s42,s60,s80 >> run_r10.log 2>> run_r10.err
nice -n 15 $S/venv/bin/python -B ew_measure.py ew_ecs.jsonl s80_A_1301,s80_A_1311,s80_A_1321,s80_A_1331,s80_A_1341,s80_A_1351,s80_B_1401,s80_B_1411,s80_B_1421,s80_B_1431,s80_B_1441,s80_B_1451,s42_A_1101,s42_A_1111,s42_A_1121,s42_A_1131,s42_A_1141,s42_A_1151,s42_B_1201,s42_B_1211,s42_B_1221,s42_B_1231,s42_B_1241,s42_B_1251 > run_ecs.log 2> run_ecs.err
date -u +%FT%TZ > $H/run_rest.finished
