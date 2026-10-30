#!/bin/bash
# after wave 2: move old-format outputs aside, redo wave 2's missing rows, then wave 3 (extra seeds at key checkpoints)
H=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias
export H S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
while kill -0 22797 2>/dev/null; do sleep 15; done
python3 $H/eb_move_old.py
$H/eb_jobs.sh $H/eb_jobs_wave2.txt
$H/eb_jobs.sh $H/eb_jobs_wave3.txt
echo CHAIN-DONE
