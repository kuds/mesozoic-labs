#!/bin/bash
# usage: pytest_one.sh <basetemp-name> <pytest args...>; pytest with basetemp and TMPDIR inside review/fixer/tmp, no cache
F=/home/user/work/phase_screen/review/fixer
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
name="$1"; shift
mkdir -p "$F/tmp/pt/$name" "$F/tmp/tmpdir_$name"
cd "$F/tmp" || exit 9
export TMPDIR="$F/tmp/tmpdir_$name" MPLCONFIGDIR="$F/tmp/tmpdir_$name/mpl" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec nice -n 10 "$PY" -I -B -m pytest -q -p no:cacheprovider --basetemp="$F/tmp/pt/$name" "$@"
