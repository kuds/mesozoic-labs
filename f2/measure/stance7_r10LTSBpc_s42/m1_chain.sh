#!/bin/bash
# Runs M1 at the six labels one after another (one simulation process at a time), via run_one.sh.
M=/home/user/work/pc_study/f2/measure/stance7_r10LTSBpc_s42
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
R=stance7_r10LTSBpc_s42
F=/home/user/work/pc_study/round/fetch/F2/$R/copies/01_stance
cd /tmp || exit 99
for L in 4194304 5242880 6291456 8388608 10485760 final; do
  if [ "$L" = final ]; then
    Z=$F/models/stage1_final.zip.copy1; P=$F/models/stage1_final_vecnorm.pkl.copy1
  else
    Z=$F/models/stage1_${L}_steps.zip.copy1; P=$F/models/stage1_vecnormalize_${L}_steps.pkl.copy1
  fi
  "$M/run_one.sh" "m1_$L" nice -n 10 env OMP_NUM_THREADS=1 "$PY" -I -B /home/user/work/pc_study/tools/pc/measure_r10round.py m1 --copy /home/user/work/pc_study/f2/copy_1dfe2327 --zip "$Z" --pkl "$P" --stage-config "$F/stage_config.json.copy1" --run-id "$R" --label "$L"
done
echo done > "$M/m1_chain.done"
