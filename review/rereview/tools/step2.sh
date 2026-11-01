#!/bin/bash
# rereview step 2: remove trees/base; 2-update switch-ON smoke (ST/tools/smoke_pc.py train) on trees/head (e4d222bb);
# test_manip_pc.py on it; the stand-in dry run (ST/tools/standin_s6.py) into rereview/standin_s6. nice 10.
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad; PY=$S2/venv/bin/python
R=/home/user/work/pc_study/review/rereview; T=/home/user/work/pc_study/tools
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TMPDIR=$R/tmp MPLBACKEND=Agg GIT_OPTIONAL_LOCKS=0
log() { echo "$(date -u +%FT%TZ) $*" >> $R/cmdlog.txt; }
rm -rf $R/trees/base; log "removed trees/base (8737cd3d archive) after the equivalence run"
cd $R/tools
nice -n 10 $PY -I -B $T/smoke_pc.py train --copy $R/trees/head --out $R/smoke_train > $R/smoke_train.out 2> $R/smoke_train.err; log "smoke_pc.py train (switch ON, 2 PPO updates) on trees/head exit $?"
mkdir -p $R/scorer_tests
PC_TEST_COPY=$R/trees/head PC_TEST_SMOKE_MODELS=$R/smoke_train/run/models nice -n 10 $PY -I -B -m pytest -p no:cacheprovider $T/tests/test_manip_pc.py -v -rfEs --tb=short --basetemp=$R/tmp/manip > $R/scorer_tests/pytest_manip_pc.log 2>&1; log "pytest test_manip_pc.py (copy trees/head, smoke rereview/smoke_train) exit $?"
nice -n 10 $PY -I -B $T/standin_s6.py --out $R/standin_s6 > $R/standin_s6.stdout 2> $R/standin_s6.stderr; log "standin_s6.py --out rereview/standin_s6 exit $?"
log "step2 done"
