#!/bin/bash
# Inside withx (one private Xvfb): the pinned pilot4_video.py twice, as two independent processes started together,
# on the COPY of the smoke's judged cell OUT/s42/65536 with --from-score on the COPY of the smoke's validation score
# JSON (the CLI the scorer prints), from this lane's own clone of tools/tree at 0e6a57b. Each run writes its own
# out folder; nothing else. No training.
R=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/render/render
T=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/tools
PY=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/branch/venv_full/bin/python
TREE=$R/tree
cd "$TREE" || exit 9
echo "DISPLAY=$DISPLAY MUJOCO_GL=$MUJOCO_GL MESA_SHADER_CACHE_DISABLE=$MESA_SHADER_CACHE_DISABLE TMPDIR=$TMPDIR"
for k in run1 run2; do
  ( t0=$(date -u +%FT%TZ)
    ISO_WT=$TREE timeout 9000 $PY -I -B $T/iso.py $T/pilot4_video.py $R/cells/OUT/s42/65536 \
      --from-score $R/cells/score/score_validation_65536.json --out $R/videos/$k/s42 > $R/logs/video_$k.log 2>&1
    rc=$?
    echo "$k video_rc=$rc start $t0 end $(date -u +%FT%TZ)" | tee $R/logs/video_$k.rc ) &
done
wait
