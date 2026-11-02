#!/bin/bash
# M3 (certificate) at final, through runrec.sh (cmdlog line); one simulation process
D=/home/user/work/pc_study/f2/measure/stance7_r10LTSBoff_s80
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
S=/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBoff_s80/copies/01_stance; M=$S/models
cd /tmp || exit 99
$D/runrec.sh m3_final $D/m3_final.out nice -n 10 env OMP_NUM_THREADS=1 $PY -I -B /home/user/work/pc_study/tools/pc/measure_r10round.py m3 --copy /home/user/work/pc_study/f2/copy_1dfe2327 --zip $M/stage1_final.zip.copy1 --pkl $M/stage1_final_vecnorm.pkl.copy1 --stage-config $S/stage_config.json.copy1 --run-id stance7_r10LTSBoff_s80 --label final
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) m3 final rc=$?" >> $D/m3_final.log
