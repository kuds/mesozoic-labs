#!/bin/bash
# Fix lane: phases B (readouts), C (scorer, video) and D (the fix stage's refusals) in order, then both summaries.
F=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/tools/out/fix
T=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/tools
PY=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/branch/venv_full/bin/python
bash $F/scripts/phase_b_readouts.sh > $F/logs/phase_b.stdout.txt 2> $F/logs/phase_b.stderr.txt; echo "phase_b rc=$?"
bash $F/scripts/phase_c_score.sh > $F/logs/phase_c.stdout.txt 2> $F/logs/phase_c.stderr.txt; echo "phase_c rc=$?"
bash $F/scripts/phase_d_fix.sh > $F/logs/phase_d.stdout.txt 2> $F/logs/phase_d.stderr.txt; echo "phase_d rc=$?"
bash $F/scripts/job.sh e2e_summary "$PY -I -B $F/scripts/e2e_summary.py $F > $F/logs/e2e_summary.stdout.txt 2> $F/logs/e2e_summary.stderr.txt"; echo "e2e_summary rc=$?"
bash $F/scripts/job.sh e2e_fix_checks "$PY -I -B $F/scripts/e2e_fix_checks.py $F $T > $F/logs/e2e_fix_checks.stdout.txt 2> $F/logs/e2e_fix_checks.stderr.txt"; echo "e2e_fix_checks rc=$?"
