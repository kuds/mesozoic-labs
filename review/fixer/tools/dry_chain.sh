#!/bin/bash
# fixer: rebuild the measuring copy of WT HEAD and re-run the adapter's dry runs on it (ST/tools/dry), as the adapter's
# cmdlog lines 4-31 record them; nice 10; outputs into ST/tools/dry (the adapter's files kept in review/fixer/orig/tools/dry)
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad; PY=$S2/venv/bin/python
T=/home/user/work/pc_study/tools; D=$T/dry; F=/home/user/work/pc_study/review/fixer
HEAD=$(git -C /home/user/work/pc_branch --no-optional-locks rev-parse HEAD)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TMPDIR=$F/tests_out MPLBACKEND=Agg GIT_OPTIONAL_LOCKS=0
log() { echo "$(date -u +%FT%TZ) $*" >> $F/dry_chain.log; }
log "start HEAD $HEAD"
rm -rf $D/copy_head $D/pins_sc $D/smoke_train $D/smoke_bundle
nice -n 10 $PY -I -B $T/pc/measure_r10round.py build-copy --dest $D/copy_head --commit $HEAD > $D/build_copy.out 2> $F/build_copy.err; log "build-copy exit $?"
nice -n 10 $PY -I -B $T/smoke_pc.py pins --copy $D/copy_head --out $D/pins_sc > $D/pins_sc.out 2> $D/pins_sc.err; log "pins exit $?"
nice -n 10 $PY -I -B $T/smoke_pc.py train --copy $D/copy_head --out $D/smoke_train > $D/smoke_train.out 2> $D/smoke_train.err; log "train exit $?"
SRC=$(find $D/smoke_train/run -maxdepth 1 -mindepth 1 -type d -name "01_*" | head -1)
log "train stage dir: $SRC"
GITHUB_SHA=$HEAD nice -n 10 $PY -I -B $T/smoke_pc.py bundle --copy $D/copy_head --source $SRC --out $D/smoke_bundle/stance7_r10LTSBpc_s42 --run-id stance7_r10LTSBpc_s42 --seed 42 --label "r10 LTSB phase-critic s42" > $D/smoke_bundle.out 2> $D/smoke_bundle.err; log "bundle exit $?"
log "done"
