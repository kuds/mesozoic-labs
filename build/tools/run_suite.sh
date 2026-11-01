#!/bin/sh
# usage: run_suite.sh TREE OUTNAME PATHS...   (nice 10, one pytest process, outputs under ST/build/suites/OUTNAME)
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
B=/home/user/work/pc_study/build
T=$1; O=$B/suites/$2; shift 2
mkdir -p $O; cd $T
echo "start $(date -u) tree $T head $(git -C $T rev-parse HEAD 2>&1 | head -1) paths $*" > $O/run.txt
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg nice -n 10 $PY -I -B -m pytest -p no:cacheprovider "$@" -vv -rfEs --durations=30 --tb=short --basetemp=$O/tmp --junitxml=$O/junit.xml > $O/pytest.log 2>&1
echo "exit $? end $(date -u)" >> $O/run.txt
$PY -I -B $B/tools/junit_failed.py $O/junit.xml > $O/failed.txt 2>&1
rm -rf $O/tmp
