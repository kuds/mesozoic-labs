#!/bin/bash
# rereview step 1: git archive of WT e4d222bb into rereview/trees/head; the 36 critic-phase unit tests there;
# equiv_run switch OFF and ON on it; git archive 8737cd3d into trees/base; equiv_run OFF there; compare. nice 10.
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad; PY=$S2/venv/bin/python
R=/home/user/work/pc_study/review/rereview; B=/home/user/work/pc_study/build/tools; WT=/home/user/work/pc_branch
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TMPDIR=$R/tmp MPLBACKEND=Agg GIT_OPTIONAL_LOCKS=0
mkdir -p $R/tmp $R/trees/head $R/trees/base $R/cp_tests $R/equiv
log() { echo "$(date -u +%FT%TZ) $*" >> $R/cmdlog.txt; }
HEAD=$(git -C $WT --no-optional-locks rev-parse HEAD); log "step1 start WT HEAD $HEAD"
git -C $WT --no-optional-locks archive e4d222bbde28800d7ee0851380123ff4ac9642a8 | tar -x -C $R/trees/head; log "archive e4d222bb -> trees/head exit $?"
cd $R/trees/head
PYTHONPATH=$R/trees/head nice -n 10 $PY -I -B -m pytest -p no:cacheprovider environments/shared/tests/test_critic_phase.py environments/shared/tests/test_critic_phase_ppo.py -q -rfEs --tb=short --basetemp=$R/tmp/cp > $R/cp_tests/pytest.log 2>&1
log "critic-phase unit tests on trees/head exit $?"
cd $R/tools
nice -n 10 $PY -I -B $B/equiv_run.py $R/trees/head $R/equiv/head_off > $R/equiv/head_off.log 2>&1; log "equiv_run head OFF exit $?"
nice -n 10 $PY -I -B $B/equiv_run.py $R/trees/head $R/equiv/head_on --critic-phase > $R/equiv/head_on.log 2>&1; log "equiv_run head ON exit $?"
git -C $WT --no-optional-locks archive 8737cd3d86431de1bdd0fcdb49a0d956a45e084d | tar -x -C $R/trees/base; log "archive 8737cd3d -> trees/base exit $?"
nice -n 10 $PY -I -B $B/equiv_run.py $R/trees/base $R/equiv/base > $R/equiv/base.log 2>&1; log "equiv_run base OFF exit $?"
$PY -I -B $B/equiv_compare.py $R/equiv/base $R/equiv/head_off > $R/equiv/compare_base_vs_head_off.txt 2>&1; log "compare base vs head OFF exit $?"
$PY -I -B $B/equiv_compare.py $R/equiv/head_off $R/equiv/head_on > $R/equiv/compare_head_off_vs_head_on.txt 2>&1; log "compare head OFF vs ON exit $?"
log "step1 done"
