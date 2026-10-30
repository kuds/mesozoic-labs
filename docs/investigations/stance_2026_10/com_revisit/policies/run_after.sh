#!/bin/bash
# The runs after the M1 panel, one simulation process at a time: the equivalence check in both trees, the held-out
# panel (10 seeds, the 26 policies), the statue null panels (100 seeds).
set -u
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
N=$S/com_revisit/notes/policies
W=$S/com_revisit/wt_policies
PY="nice -n 10 $S/venv/bin/python -B"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $N
date -u +%FT%TZ
PYTHONPATH=$N/code_copy $PY equiv.py $N/equiv_copy.npz > $N/equiv_copy.log 2>&1; echo "equiv copy exit $?"
PYTHONPATH=$W $PY equiv.py $N/equiv_main.npz > $N/equiv_main.log 2>&1; echo "equiv main exit $?"
date -u +%FT%TZ
ONLY=$(PYTHONPATH=$N/code_copy $S/venv/bin/python -B -c "import run_policies as r; print(','.join(w for w, *_ in r.pairs() if w != 'statue'))")
PYTHONPATH=$N/code_copy $PY run_policies.py $N/run_held --seeds 3062-3071 --only $ONLY > $N/run_held.log 2>&1; echo "held exit $?"
date -u +%FT%TZ
PYTHONPATH=$N/code_copy $PY run_policies.py $N/run_null --seeds 3062-3161 --only statue > $N/run_null.log 2>&1; echo "null exit $?"
date -u +%FT%TZ
