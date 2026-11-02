#!/bin/bash
# fetch tool wrapper (F2, pc_s60 session): run from /tmp, nice 10, TMPDIR inside the round's fetch folder
cd /tmp || exit 9
export TMPDIR=/home/user/work/pc_study/round/fetch/tmp
export OMP_NUM_THREADS=1
exec nice -n 10 /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I -B /home/user/work/pc_study/tools/pc/fetch_r10round.py --round /home/user/work/pc_study/round "$@"
