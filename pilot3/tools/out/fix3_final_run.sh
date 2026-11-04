#!/bin/bash
# FIX3: the final run of every test and validation on the re-pinned tool files (walking pilot 3). Mirrors FIX2's
# out/fix2_final_run.sh, except: the e2e is not re-run in full (its video step needs Xvfb, which writes under the
# system temp folder); in its place the habit tool's seed mode runs on FIX2's e2e cell (one 20-episode trace) and its
# refusals run on stand-in cells (no simulation), and the scorer runs on that cell with the new record. One V - G run
# on the smoke's 65,536 pair (compared with FIX2's), the noise readout on the smoke's two runs (compared with FIX2's)
# and on a cut gate_progress.npz. One simulation at a time; nothing trains.
T=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/tools
F=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/fetch
X=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/fix3
R=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/review3
S=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/smoke/clone/logs/tyrannosaurus_rex/anatomical/ppo_quick_test
O=$T/out/fix3_final
cd $T
mkdir -p $O
export TMPDIR=$X/tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MESA_SHADER_CACHE_DISABLE=true MPLCONFIGDIR=$X/tmp/mpl XDG_CACHE_HOME=$X/tmp/xdg_cache GIT_OPTIONAL_LOCKS=0
export MUJOCO_GL=disable
PY=$T/venv/bin/python
FILES="iso.py pilot3_judge.py pilot3_score.py test_pilot3_score.py pilot3_habit.py pilot3_video.py pilot3_vg.py pilot3_noise.py nb_pilot_runner3.py gait_probe_pilot.py make_gait_cases.py fixtures/gait_cases.json tests/_tree_facts.py tests/test_pilot3_judge.py tests/test_pilot3_habit.py tests/test_nb_pilot_runner3.py tests/test_pilot3_video.py tests/test_pilot3_vg.py tests/test_pilot3_noise.py tests/validate_pc_ledgers.py tests/e2e_parent_cell.py"
RCS=$O/rcs.txt
: > $RCS
sha256sum $FILES > $O/sha256_before.txt
date -u +%FT%TZ > $O/started.txt
rm -rf $X/tmp/pt_fix3_final
nice -n 10 $PY -I -B -m pytest -p no:cacheprovider --basetemp=$X/tmp/pt_fix3_final -v -rfE --junitxml=$O/pytest.xml tests/test_pilot3_judge.py tests/test_pilot3_habit.py tests/test_nb_pilot_runner3.py tests/test_pilot3_video.py tests/test_pilot3_vg.py tests/test_pilot3_noise.py > $O/pytest.log.txt 2>&1; echo "pytest rc=$?" >> $RCS
nice -n 10 $PY -I -B test_pilot3_score.py tree > $O/test_pilot3_score.log.txt 2>&1; echo "test_pilot3_score rc=$?" >> $RCS
nice -n 10 $PY -I -B tests/validate_pc_ledgers.py $F > $O/habit_pc_validation.txt 2>&1; echo "validate_pc_ledgers rc=$?" >> $RCS
cmp $O/habit_pc_validation.txt out/fix2_final/habit_pc_validation.txt && echo "validate_pc_ledgers: byte-identical to FIX2's" >> $RCS
nice -n 10 $PY -I -B $X/../review2_code/bin/fn_compare.py src/pilot/pilot_score.py pilot3_score.py > $O/fn_compare_score.txt 2>&1; echo "fn_compare rc=$?" >> $RCS
# the habit seed mode on FIX2's e2e cell (one 20-episode trace), then its refusals (no simulation)
CELL=$T/e2e/OUT/s42/8000000
LED=$T/e2e/runs/pilot3_walk_s42/03_locomotion/eval_episodes.csv
(cd tree && ISO_WT=$T/tree nice -n 10 $PY -I -B $T/iso.py $T/pilot3_habit.py seed --seed 42 --ledger $LED --cell $CELL --json $O/habit_seed_e2e_cell.json > $O/habit_seed_e2e_cell.txt 2>&1); echo "habit seed on the e2e cell rc=$?" >> $RCS
(cd tree && ISO_WT=$T/tree nice -n 10 $PY -I -B $T/iso.py $T/pilot3_habit.py seed --seed 42 --ledger $LED --cell $CELL --json $O/habit_seed_e2e_cell.json > $O/habit_refuse_existing_json.txt 2>&1); echo "habit seed, existing --json (expect 2) rc=$?" >> $RCS
nice -n 10 $PY -I -B $X/bin/make_stale_cell.py $CELL $LED $X/tmp/stale_cell 3800000 > $O/make_stale_cell.txt 2>&1; echo "make_stale_cell rc=$?" >> $RCS
(cd tree && ISO_WT=$T/tree nice -n 10 $PY -I -B $T/iso.py $T/pilot3_habit.py seed --seed 42 --ledger $X/tmp/stale_cell/runs/pilot3_walk_s42/03_locomotion/eval_episodes.csv --cell $X/tmp/stale_cell/OUT/s42/8000000 --json $X/tmp/stale_cell/H_stale.json > $O/habit_refuse_stale_window.txt 2>&1); echo "habit seed, stale copy at the judged path (expect 2) rc=$? json_written=$([ -e $X/tmp/stale_cell/H_stale.json ] && echo yes || echo no)" >> $RCS
(cd tree && ISO_WT=$T/tree nice -n 10 $PY -I -B $T/iso.py $T/pilot3_habit.py seed --seed 42 --ledger $X/tmp/stale_cell/runs/pilot3_walk_s42/03_locomotion/eval_episodes.csv --cell $CELL --json $X/tmp/stale_cell/H_elsewhere.json > $O/habit_refuse_same_named_copy.txt 2>&1); echo "habit seed, same-named copy elsewhere (expect 2) rc=$? json_written=$([ -e $X/tmp/stale_cell/H_elsewhere.json ] && echo yes || echo no)" >> $RCS
nice -n 10 $PY -I -B pilot3_score.py e2e/OUT --seeds 42 --habit $O/habit_seed_e2e_cell.json --json $O/score_e2e_cell.json > $O/score_e2e_cell.txt 2>&1; echo "score on the e2e cell rc=$?" >> $RCS
nice -n 10 $PY -I -B $X/bin/stale_ledger_fixed.py $T $F/stance7_r10LTSBoff_s42/01_stance/eval_episodes.csv 8000000 3800000 $X/tmp/stale_work $X/../fix2/out/judge/OUT/retired/pilot3_walk_s42/s42/65536/binding.json > $O/stale_ledger_fixed.txt 2>&1; echo "stale_ledger_fixed rc=$?" >> $RCS
# V - G once on the smoke's 65,536 pair, and the refusal of an existing --json
(cd tree && ISO_WT=$T/tree nice -n 10 $PY -I -B $T/iso.py $T/pilot3_vg.py --checkpoint $S/pilot3_walk_s42/03_locomotion/models/stage2_65536_steps.zip --vecnorm $S/pilot3_walk_s42/03_locomotion/models/stage2_vecnormalize_65536_steps.pkl --json $O/vg_smoke_65536.json > $O/vg_smoke_65536.txt 2>&1); echo "vg_smoke_65536 rc=$?" >> $RCS
(cd tree && ISO_WT=$T/tree nice -n 10 $PY -I -B $T/iso.py $T/pilot3_vg.py --checkpoint $S/pilot3_walk_s42/03_locomotion/models/stage2_65536_steps.zip --vecnorm $S/pilot3_walk_s42/03_locomotion/models/stage2_vecnormalize_65536_steps.pkl --json $O/vg_smoke_65536.json > $O/vg_refuse_existing_json.txt 2>&1); echo "vg, existing --json (expect 2) rc=$?" >> $RCS
# the noise readout on the smoke's two runs, and on a cut gate_progress.npz (the third review's probe)
for r in pilot3_walk_s42 pilot3_walk_s42_rtest; do nice -n 10 $PY -I -B pilot3_noise.py $S/$r --json $O/noise_smoke_$r.json > $O/noise_smoke_$r.txt 2>&1; echo "noise_smoke_$r rc=$?" >> $RCS; done
nice -n 10 $PY -I -B $R/noise_cut_npz.py $T $S/pilot3_walk_s42 $X/tmp > $O/noise_cut_npz.txt 2>&1; echo "noise_cut_npz rc=$?" >> $RCS
nice -n 10 $PY -I -B $X/bin/compare_fix3.py $T $O > $O/compare_with_fix2.txt 2>&1; echo "compare_fix3 rc=$?" >> $RCS
echo "tree status lines after: $(git -C tree status --porcelain --ignored | wc -l)" >> $RCS
sha256sum $FILES > $O/sha256_after.txt
cmp $O/sha256_before.txt $O/sha256_after.txt && echo "tool files unchanged during the run" >> $RCS
date -u +%FT%TZ > $O/ended.txt
