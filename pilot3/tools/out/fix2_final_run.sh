#!/bin/bash
# FIX2: the final run of every test and validation on the fixed and new tool files (walking pilot 3). Mirrors the
# tools stage's out/final_run.sh, plus the new tools' tests, two V - G runs on the smoke's 65,536 pair (equal JSON
# expected) and the noise readout on the smoke's two runs. One simulation at a time; nothing trains.
T=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/tools
F=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/fetch
X=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/fix2
S=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/smoke/clone/logs/tyrannosaurus_rex/anatomical/ppo_quick_test
O=$T/out/fix2_final
cd $T
mkdir -p $O
export TMPDIR=$X/tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MESA_SHADER_CACHE_DISABLE=true MPLCONFIGDIR=$X/tmp/mpl XDG_CACHE_HOME=$X/tmp/xdg_cache GIT_OPTIONAL_LOCKS=0
FILES="iso.py pilot3_judge.py pilot3_score.py test_pilot3_score.py pilot3_habit.py pilot3_video.py pilot3_vg.py pilot3_noise.py nb_pilot_runner3.py gait_probe_pilot.py make_gait_cases.py fixtures/gait_cases.json tests/_tree_facts.py tests/test_pilot3_judge.py tests/test_pilot3_habit.py tests/test_nb_pilot_runner3.py tests/test_pilot3_video.py tests/test_pilot3_vg.py tests/test_pilot3_noise.py tests/validate_pc_ledgers.py tests/e2e_parent_cell.py"
sha256sum $FILES > $O/sha256_before.txt
date -u +%FT%TZ > $O/started.txt
rm -rf $X/tmp/pt_fix2_final
nice -n 10 venv/bin/python -I -B -m pytest -p no:cacheprovider --basetemp=$X/tmp/pt_fix2_final -v -rfE --junitxml=$O/pytest.xml tests/test_pilot3_judge.py tests/test_pilot3_habit.py tests/test_nb_pilot_runner3.py tests/test_pilot3_video.py tests/test_pilot3_vg.py tests/test_pilot3_noise.py > $O/pytest.log.txt 2>&1; echo "pytest rc=$?" >> $O/rcs.txt
nice -n 10 venv/bin/python -I -B test_pilot3_score.py tree > $O/test_pilot3_score.log.txt 2>&1; echo "test_pilot3_score rc=$?" >> $O/rcs.txt
nice -n 10 venv/bin/python -I -B tests/validate_pc_ledgers.py $F > $O/habit_pc_validation.txt 2>&1; echo "validate_pc_ledgers rc=$?" >> $O/rcs.txt
(cd tree && ISO_WT=$T/tree MUJOCO_GL=disable nice -n 10 $T/venv/bin/python -I -B $T/iso.py $T/pilot3_habit.py trace --checkpoint $T/tree/pilot_checkpoints/stance7_r10LTSBpc_s42_final/stage1_final.zip --vecnorm $T/tree/pilot_checkpoints/stance7_r10LTSBpc_s42_final/stage1_final_vecnorm.pkl --json $O/habit_trace_parent.json > $O/habit_trace_parent.txt 2>&1); echo "habit_trace_parent rc=$?" >> $O/rcs.txt
venv/bin/python -I -B tests/e2e_parent_cell.py $T > $O/e2e.log.txt 2>&1; echo "e2e rc=$?" >> $O/rcs.txt
for k in a b; do (cd tree && ISO_WT=$T/tree MUJOCO_GL=disable nice -n 10 $T/venv/bin/python -I -B $T/iso.py $T/pilot3_vg.py --checkpoint $S/pilot3_walk_s42/03_locomotion/models/stage2_65536_steps.zip --vecnorm $S/pilot3_walk_s42/03_locomotion/models/stage2_vecnormalize_65536_steps.pkl --json $O/vg_smoke_65536_$k.json > $O/vg_smoke_65536_$k.txt 2>&1); echo "vg_smoke_65536_$k rc=$?" >> $O/rcs.txt; done
cmp $O/vg_smoke_65536_a.json $O/vg_smoke_65536_b.json && echo "vg: two runs on the smoke's 65,536 pair wrote byte-identical JSON" >> $O/rcs.txt
for r in pilot3_walk_s42 pilot3_walk_s42_rtest; do nice -n 10 venv/bin/python -I -B pilot3_noise.py $S/$r --json $O/noise_smoke_$r.json > $O/noise_smoke_$r.txt 2>&1; echo "noise_smoke_$r rc=$?" >> $O/rcs.txt; done
echo "tree status lines after: $(git -C tree status --porcelain --ignored | wc -l)" >> $O/rcs.txt
sha256sum $FILES > $O/sha256_after.txt
cmp $O/sha256_before.txt $O/sha256_after.txt && echo "tool files unchanged during the run" >> $O/rcs.txt
date -u +%FT%TZ > $O/ended.txt
