#!/bin/bash
# M3 at final, through runrec.sh (stdout m3_final.out, stderr m3_final.err, exit m3_final.exit, cmdlog.txt)
D=/home/user/work/pc_study/f2/measure/stance7_r10LTSBoff_s60
P=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
M=/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBoff_s60/copies/01_stance/models
SC=/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBoff_s60/copies/01_stance/stage_config.json.copy1
echo $$ > $D/m3_final.pid
cd /tmp || exit 99
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start m3 final" >> $D/m3_final.log
$D/runrec.sh m3_final nice -n 10 env OMP_NUM_THREADS=1 $P -I -B /home/user/work/pc_study/tools/pc/measure_r10round.py m3 --copy /home/user/work/pc_study/f2/copy_1dfe2327 --zip $M/stage1_final.zip.copy1 --pkl $M/stage1_final_vecnorm.pkl.copy1 --stage-config $SC --run-id stance7_r10LTSBoff_s60 --label final
rc=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end m3 final rc=$rc" >> $D/m3_final.log
