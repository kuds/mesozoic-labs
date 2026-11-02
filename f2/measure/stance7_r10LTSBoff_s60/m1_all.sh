#!/bin/bash
# M1 at the six labels C7 names, one at a time (sequential), each through runrec.sh (cmdlog.txt)
D=/home/user/work/pc_study/f2/measure/stance7_r10LTSBoff_s60
P=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
M=/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBoff_s60/copies/01_stance/models
SC=/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBoff_s60/copies/01_stance/stage_config.json.copy1
echo $$ > $D/m1_all.pid
cd /tmp || exit 99
for L in 4194304 5242880 6291456 8388608 10485760 final; do
  if [ $L = final ]; then Z=$M/stage1_final.zip.copy1; K=$M/stage1_final_vecnorm.pkl.copy1; else Z=$M/stage1_${L}_steps.zip.copy1; K=$M/stage1_vecnormalize_${L}_steps.pkl.copy1; fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start m1 $L" >> $D/m1_all.log
  $D/runrec.sh m1_$L nice -n 10 env OMP_NUM_THREADS=1 $P -I -B /home/user/work/pc_study/tools/pc/measure_r10round.py m1 --copy /home/user/work/pc_study/f2/copy_1dfe2327 --zip $Z --pkl $K --stage-config $SC --run-id stance7_r10LTSBoff_s60 --label $L
  rc=$?
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end m1 $L rc=$rc" >> $D/m1_all.log
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ALL DONE" >> $D/m1_all.log
echo 0 > $D/m1_all.exit
