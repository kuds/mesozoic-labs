#!/bin/bash
# Run a command at the lowest CPU priority, so that it never slows a timed or long-running job beside it.
#
#   bash tools/lowprio.sh python -m designlab.stage1 eval ...
#
# The kernel has sched_autogroup enabled, so a process's own nice value only orders it within its session's group, and
# jobs started from different shells compete as groups. This sets the calling session's autogroup to nice 19 as well,
# pins BLAS to one thread (numpy otherwise spreads a small matrix product over every core) and stops Python writing
# bytecode (useful when the code lives in a checkout another job is using).
echo 19 > /proc/self/autogroup 2>/dev/null || true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec nice -n 19 "$@"
