#!/bin/bash
# The final run of every test and validation on the frozen tool files (walking pilot 3, TOOLS stage).
T=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/tools
F=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/fetch
cd $T
mkdir -p out/final
export TMPDIR=$T/tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MESA_SHADER_CACHE_DISABLE=true MPLCONFIGDIR=$T/tmp/mpl XDG_CACHE_HOME=$T/tmp/xdg_cache GIT_OPTIONAL_LOCKS=0
FILES="iso.py pilot3_judge.py pilot3_score.py test_pilot3_score.py pilot3_habit.py pilot3_video.py nb_pilot_runner3.py gait_probe_pilot.py make_gait_cases.py fixtures/gait_cases.json tests/_tree_facts.py tests/test_pilot3_judge.py tests/test_pilot3_habit.py tests/test_nb_pilot_runner3.py tests/test_pilot3_video.py tests/validate_pc_ledgers.py tests/e2e_parent_cell.py"
sha256sum $FILES > out/final/sha256_before.txt
date -u +%FT%TZ > out/final/started.txt
rm -rf tmp/pt_final
nice -n 10 venv/bin/python -I -B -m pytest -p no:cacheprovider --basetemp=tmp/pt_final -v -rfE --junitxml=out/final/pytest.xml tests/test_pilot3_judge.py tests/test_pilot3_habit.py tests/test_nb_pilot_runner3.py tests/test_pilot3_video.py > out/final/pytest.log.txt 2>&1; echo "pytest rc=$?" >> out/final/rcs.txt
nice -n 10 venv/bin/python -I -B test_pilot3_score.py tree > out/final/test_pilot3_score.log.txt 2>&1; echo "test_pilot3_score rc=$?" >> out/final/rcs.txt
nice -n 10 venv/bin/python -I -B tests/validate_pc_ledgers.py $F > out/final/habit_pc_validation.txt 2>&1; echo "validate_pc_ledgers rc=$?" >> out/final/rcs.txt
(cd tree && ISO_WT=$T/tree MUJOCO_GL=disable nice -n 10 $T/venv/bin/python -I -B $T/iso.py $T/pilot3_habit.py trace --checkpoint $T/tree/pilot_checkpoints/stance7_r10LTSBpc_s42_final/stage1_final.zip --vecnorm $T/tree/pilot_checkpoints/stance7_r10LTSBpc_s42_final/stage1_final_vecnorm.pkl --json $T/out/final/habit_trace_parent.json > $T/out/final/habit_trace_parent.txt 2>&1); echo "habit_trace_parent rc=$?" >> out/final/rcs.txt
venv/bin/python -I -B tests/e2e_parent_cell.py $T > out/final/e2e.log.txt 2>&1; echo "e2e rc=$?" >> out/final/rcs.txt
sha256sum $FILES > out/final/sha256_after.txt
cmp out/final/sha256_before.txt out/final/sha256_after.txt && echo "tool files unchanged during the run" >> out/final/rcs.txt
date -u +%FT%TZ > out/final/ended.txt
