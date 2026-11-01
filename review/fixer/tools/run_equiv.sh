#!/bin/bash
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad; PY=$S2/venv/bin/python
F=/home/user/work/pc_study/review/fixer; B=/home/user/work/pc_study/build/tools; E=$F/equiv
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TMPDIR=$F/tests_out MPLBACKEND=Agg
cd $F/scratch
rm -rf $E/base $E/head_off $E/head_on
nice -n 10 $PY -I -B $B/equiv_run.py $F/trees/base $E/base > $E/base.log 2>&1; echo "base exit $?" >> $E/times.txt
nice -n 10 $PY -I -B $B/equiv_run.py $F/trees/head $E/head_off > $E/head_off.log 2>&1; echo "head_off exit $?" >> $E/times.txt
nice -n 10 $PY -I -B $B/equiv_run.py $F/trees/head $E/head_on --critic-phase > $E/head_on.log 2>&1; echo "head_on exit $?" >> $E/times.txt
$PY -I -B $B/equiv_compare.py $E/base $E/head_off > $E/compare_base_vs_head_off.txt 2>&1; echo "compare off exit $?" >> $E/times.txt
$PY -I -B $B/equiv_compare.py $E/head_off $E/head_on > $E/compare_head_off_vs_head_on.txt 2>&1; echo "compare on exit $?" >> $E/times.txt
date -u >> $E/times.txt
