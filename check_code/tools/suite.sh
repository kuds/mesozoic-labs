#!/bin/bash
# usage: suite.sh TREE OUTNAME PATHS...  (nice 10; outputs and tmp under check_code/suites/OUTNAME)
set -u
TREE=$1; NAME=$2; shift 2
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
O=/home/user/work/pc_study/check_code/suites/$NAME
mkdir -p $O/tmp $O/tmpdir
echo "start $(date -u +%FT%TZ) tree $TREE paths $*" > $O/run.txt
cd $TREE
PYTHONDONTWRITEBYTECODE=1 TMPDIR=$O/tmpdir MPLBACKEND=Agg OMP_NUM_THREADS=1 nice -n 10 $PY -I -B -m pytest -p no:cacheprovider "$@" -vv -rfEs --tb=short --basetemp=$O/tmp --junitxml=$O/junit.xml > $O/pytest.log 2>&1
echo "exit $? end $(date -u +%FT%TZ)" >> $O/run.txt
rm -rf $O/tmp $O/tmpdir
