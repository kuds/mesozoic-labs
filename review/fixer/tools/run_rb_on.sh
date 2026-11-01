#!/bin/sh
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
P=$S2/venv/bin/python; F=/home/user/work/pc_study/review/fixer; T=$F/trees/head; O=$F/rb
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TMPDIR=$O/tmp MPLBACKEND=Agg
for seed in 60 42 80; do
  echo "on seed $seed start $(date -u +%FT%TZ)" >> $O/on_times.txt
  nice -n 10 $P -I -B $F/tools/rb_on.py $T $O/scr_on_$seed $O/on_$seed.pkl $seed 1 > $O/on_$seed.log 2>&1
  echo "on seed $seed exit $? end $(date -u +%FT%TZ)" >> $O/on_times.txt
done
