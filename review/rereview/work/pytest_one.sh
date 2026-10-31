#!/bin/bash
# usage: pytest_one.sh <name> <pytest args...>; basetemp and TMPDIR inside review/rereview/tmp, no cache, nice 10
R=/home/user/work/phase_screen/review/rereview
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
name="$1"; shift
mkdir -p "$R/tmp/pt/$name" "$R/tmp/tmpdir_$name"
cd "$R/tmp" || exit 9
export TMPDIR="$R/tmp/tmpdir_$name" MPLCONFIGDIR="$R/tmp/tmpdir_$name/mpl" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec nice -n 10 "$PY" -I -B -m pytest -q -p no:cacheprovider --basetemp="$R/tmp/pt/$name" "$@"
