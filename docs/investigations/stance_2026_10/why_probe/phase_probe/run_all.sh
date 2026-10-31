#!/bin/bash
# PHASE PROBE: the pre-registered rollouts (PREREG.txt), two queues at nice 10 (at most 2 simulation processes).
# Writes only under this folder: rollouts/<label>.npz and .json, logs/<label>.log, run_all.started / run_all.finished.
set -u
P=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/phase_probe
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
S1=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
F42=$S1/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance
F80=$S1/foot/stance6/fetch/F2/stance6_r10LTSB_s80/copies/01_stance
R=$S2/r10round/bias_screen2/runs
mkdir -p $P/rollouts $P/logs
date -u +"%Y-%m-%d %H:%M:%S UTC" > $P/run_all.started

one() {  # label zip pkl stage_config extra...
  local label=$1 zip=$2 pkl=$3 sc=$4; shift 4
  nice -n 10 $PY -I -B $P/phase_rollouts.py --label $label --zip $zip --pkl $pkl --stage-config $sc \
    --out $P/rollouts/$label.npz "$@" > $P/logs/$label.log 2>&1
  echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC") $label exit $?" >> $P/logs/queue.log
}

q1() {
  one P1   $F42/models/stage1_1048576_steps.zip.copy1 $F42/models/stage1_vecnormalize_1048576_steps.pkl.copy1 $F42/stage_config.json.copy1
  one C11a $R/control_2511/01_stance/models/stage1_3145728_steps.zip $R/control_2511/01_stance/models/stage1_vecnormalize_3145728_steps.pkl $R/control_2511/01_stance/stage_config.json
  one C31a $R/control_2531/01_stance/models/stage1_3145728_steps.zip $R/control_2531/01_stance/models/stage1_vecnormalize_3145728_steps.pkl $R/control_2531/01_stance/stage_config.json
  one B11E $R/critic_blind_tail3_2511/01_stance/models/stage1_final.zip $R/critic_blind_tail3_2511/01_stance/models/stage1_final_vecnorm.pkl $R/critic_blind_tail3_2511/01_stance/stage_config.json --blind
}
q2() {
  one P2   $F42/models/stage1_2097152_steps.zip.copy1 $F42/models/stage1_vecnormalize_2097152_steps.pkl.copy1 $F42/stage_config.json.copy1
  one C11E $R/control_2511/01_stance/models/stage1_final.zip $R/control_2511/01_stance/models/stage1_final_vecnorm.pkl $R/control_2511/01_stance/stage_config.json
  one C31E $R/control_2531/01_stance/models/stage1_final.zip $R/control_2531/01_stance/models/stage1_final_vecnorm.pkl $R/control_2531/01_stance/stage_config.json
  one N80  $F80/models/stage1_4194304_steps.zip.copy1 $F80/models/stage1_vecnormalize_4194304_steps.pkl.copy1 $F80/stage_config.json.copy1 --joint neck_pitch --episodes-per-env 8
}
q1 &
q2 &
wait
date -u +"%Y-%m-%d %H:%M:%S UTC" > $P/run_all.finished
