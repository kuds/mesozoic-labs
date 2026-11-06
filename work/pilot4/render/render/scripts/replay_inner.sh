#!/bin/bash
# Inside withx (one private Xvfb): the notebook's replay writer (stage_artifacts._record_stage_replays, through
# scripts/replay_render.py) twice, one after the other, on the COPY of the smoke's launch run
# (replay/run_copy/pilot4_walk_s42/03_locomotion), from this lane's clone at 0e6a57b through the pinned iso.py.
# Each run writes only its own out folder. No training.
R=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/render/render
T=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/tools
PY=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/branch/venv_full/bin/python
TREE=$R/tree
cd "$TREE" || exit 9
echo "DISPLAY=$DISPLAY MUJOCO_GL=$MUJOCO_GL MESA_SHADER_CACHE_DISABLE=$MESA_SHADER_CACHE_DISABLE TMPDIR=$TMPDIR"
for k in run1 run2; do
  t0=$(date -u +%FT%TZ)
  ISO_WT=$TREE timeout 3600 $PY -I -B $T/iso.py $R/scripts/replay_render.py \
    $R/replay/run_copy/pilot4_walk_s42/03_locomotion $R/replay/$k > $R/logs/replay_$k.log 2>&1
  rc=$?
  echo "$k replay_rc=$rc start $t0 end $(date -u +%FT%TZ)" | tee $R/logs/replay_$k.rc
done
