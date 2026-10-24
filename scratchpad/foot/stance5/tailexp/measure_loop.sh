#!/bin/bash
# Measure each arm's checkpoints as they appear (report only): measure.py --k3 per checkpoint, under the band clone.
# usage: measure_loop.sh ARM TRAIN_PID
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
T=$S/foot/stance5/tailexp
arm=$1; pid=$2; D=$T/$arm
for st in 4325376 4456448 4587520 4718592 4849664 4980736 5111808 5242880 final; do
  if [ "$st" = final ]; then
    while kill -0 $pid 2>/dev/null; do sleep 15; done
    [ -f $D/stage1_final.zip ] && [ -f $D/stage1_final_vecnorm.pkl ] || { echo "no final for $arm"; exit 1; }
  else
    while ! [ -f $D/stage1_vecnormalize_${st}_steps.pkl ]; do
      if ! kill -0 $pid 2>/dev/null && ! [ -f $D/stage1_vecnormalize_${st}_steps.pkl ]; then echo "training $pid ended without $st"; exit 1; fi
      sleep 15
    done
    sleep 10
  fi
  echo "== $(date -u +%H:%M:%S) measure $arm $st"
  env ISO_WT=$S/fix/scratch_copband nice -n 19 python3 $S/foot/stance3/tools/iso.py $T/measure.py \
      --models-dir $D --label $arm --steps $st --k3 2>&1 | grep -v '^\[iso\]'
done
echo "== $(date -u +%H:%M:%S) done $arm"
