#!/bin/sh
# D2(a)/A6: ROUND V0 (h)'s rollout-buffer digests with the switch OFF on a git archive of HEAD (rollbuf.py of
# S2/r10round/runbranch/build/tools, read only; no train(), no optimizer step).  usage: run_rb.sh first|r33
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
P=$S2/venv/bin/python; RB=$S2/r10round/runbranch/build/tools/rollbuf.py
F=/home/user/work/pc_study/review/fixer; T=$F/trees/head; O=$F/rb
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TMPDIR=$O/tmp MPLBACKEND=Agg
mkdir -p $O/tmp
if [ "$1" = first ]; then
  for seed in 60 42 80; do
    echo "seed $seed start $(date -u +%FT%TZ)" >> $O/first_times.txt
    nice -n 10 $P -I -B $RB $T $O/scr_$seed $O/head_$seed.pkl $seed 1 > $O/head_$seed.log 2>&1
    echo "seed $seed exit $? end $(date -u +%FT%TZ)" >> $O/first_times.txt
  done
else
  echo "s42 x33 start $(date -u +%FT%TZ)" >> $O/r33_times.txt
  nice -n 10 $P -I -B $RB $T $O/scr_42_33 $O/head_42_33.pkl 42 33 > $O/head_42_33.log 2>&1
  echo "s42 x33 exit $? end $(date -u +%FT%TZ)" >> $O/r33_times.txt
fi
