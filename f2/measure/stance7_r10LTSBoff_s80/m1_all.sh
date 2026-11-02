#!/bin/bash
# M1 at the six C7 labels, one process at a time, each through runrec.sh (cmdlog line per label)
D=/home/user/work/pc_study/f2/measure/stance7_r10LTSBoff_s80
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
S=/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBoff_s80/copies/01_stance; M=$S/models
cd /tmp || exit 99
for L in 4194304 5242880 6291456 8388608 10485760 final; do
  if [ $L = final ]; then Z=$M/stage1_final.zip.copy1; P=$M/stage1_final_vecnorm.pkl.copy1; else Z=$M/stage1_${L}_steps.zip.copy1; P=$M/stage1_vecnormalize_${L}_steps.pkl.copy1; fi
  $D/runrec.sh m1_$L $D/m1_$L.out nice -n 10 env OMP_NUM_THREADS=1 $PY -I -B /home/user/work/pc_study/tools/pc/measure_r10round.py m1 --copy /home/user/work/pc_study/f2/copy_1dfe2327 --zip $Z --pkl $P --stage-config $S/stage_config.json.copy1 --run-id stance7_r10LTSBoff_s80 --label $L
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) m1 $L rc=$?" >> $D/m1_all.log
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) m1_all done" >> $D/m1_all.log
