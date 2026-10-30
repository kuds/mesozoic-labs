#!/bin/bash
# The D-9 reproduction, after the replay queue (waits for replay.finished): one process.
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
cd $S/com_revisit/notes/clamp
until [ -e replay.finished ]; do sleep 20; done
export PYTHONPATH=$S/com_revisit/notes/policies/code_copy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 $S/venv/bin/python -B d9_repro.py > d9_repro.txt 2> d9_repro.err
echo "exit $?" >> d9_repro.err
