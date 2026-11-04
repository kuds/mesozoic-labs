#!/bin/bash
# AMEND stage (2026-11-04): the final run on the re-pinned tool files after the -1.5 -> -2.0 edit (decision (8)):
# pilot3_judge.py and pilot3_noise.py, their tests and tests/e2e_parent_cell.py changed; tools/tree fast-forwarded
# from 3ae1c959 to the branch head a3f1b378. Every pytest file (the tree moved under all of them), the adapted score
# test, the PC ledger validation (compared with FIX3's), the noise readout on the VALIDATE stage's re-smoke runs
# (compared with that stage's records), and the judge's ten smoke validation steps J1-J10 with the re-pinned judge on
# tools/tree (compared with the VALIDATE stage's diagnostic copy, whose code differs only in docstring line 39).
# The e2e is not re-run (it deletes and rewrites tools/e2e, and its video step starts Xvfb, which writes outside the
# stage folders). One simulation at a time (J1, J7, J10 roll the probe); nothing trains.
T=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/tools
F=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/fetch
X=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/frozen_candidate2
V=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/r6_validate
S=$V/clone/logs/tyrannosaurus_rex/anatomical/ppo_quick_test
O=$T/out/amend_final
J=$X/judge
cd $T
mkdir -p $O $J/logs
export TMPDIR=$X/tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MESA_SHADER_CACHE_DISABLE=true MPLCONFIGDIR=$X/tmp/mpl XDG_CACHE_HOME=$X/tmp/xdg_cache GIT_OPTIONAL_LOCKS=0
export MUJOCO_GL=disable
PY=$T/venv/bin/python
FILES="iso.py pilot3_judge.py pilot3_score.py test_pilot3_score.py pilot3_habit.py pilot3_video.py pilot3_vg.py pilot3_noise.py nb_pilot_runner3.py gait_probe_pilot.py make_gait_cases.py fixtures/gait_cases.json tests/_tree_facts.py tests/test_pilot3_judge.py tests/test_pilot3_habit.py tests/test_nb_pilot_runner3.py tests/test_pilot3_video.py tests/test_pilot3_vg.py tests/test_pilot3_noise.py tests/validate_pc_ledgers.py tests/e2e_parent_cell.py"
RCS=$O/rcs.txt
: > $RCS
sha256sum $FILES > $O/sha256_before.txt
date -u +%FT%TZ > $O/started.txt
echo "tree HEAD $(git -C tree rev-parse HEAD) branch $(git -C tree branch --show-current) status entries $(git -C tree status --porcelain --ignored | wc -l)" >> $RCS
rm -rf $X/tmp/pt_amend_final
nice -n 10 $PY -I -B -m pytest -p no:cacheprovider --basetemp=$X/tmp/pt_amend_final -v -rfE --junitxml=$O/pytest.xml tests/test_pilot3_judge.py tests/test_pilot3_habit.py tests/test_nb_pilot_runner3.py tests/test_pilot3_video.py tests/test_pilot3_vg.py tests/test_pilot3_noise.py > $O/pytest.log.txt 2>&1; echo "pytest rc=$?" >> $RCS
nice -n 10 $PY -I -B test_pilot3_score.py tree > $O/test_pilot3_score.log.txt 2>&1; echo "test_pilot3_score rc=$?" >> $RCS
nice -n 10 $PY -I -B tests/validate_pc_ledgers.py $F > $O/habit_pc_validation.txt 2>&1; echo "validate_pc_ledgers rc=$?" >> $RCS
cmp $O/habit_pc_validation.txt out/fix3_final/habit_pc_validation.txt && echo "validate_pc_ledgers: byte-identical to FIX3's" >> $RCS
for r in pilot3_walk_s42 pilot3_walk_s42_rtest; do nice -n 10 $PY -I -B pilot3_noise.py $S/$r --json $O/noise_resmoke_$r.json > $O/noise_resmoke_$r.txt 2>&1; echo "noise_resmoke_$r rc=$?" >> $RCS; done
# the judge's ten smoke validation steps (VALIDATE's judge/run_judge.sh with the pinned judge and tools/tree)
HEAD=a3f1b378aede66b0ee8630197f36a7c374ec77c0
P3=sha256:1c9d87c3d0f817b048de429bb80cdc89fcd2c83d5fbafd7f444af0b5474b9ff6
P1=sha256:70f3a4447794a01036c27bca5e0377a17db96d0e3d7bfb2d179e9ce94e6f353e
RUN=$S/pilot3_walk_s42
RTEST=$S/pilot3_walk_s42_rtest
C=$V/judge/copies
JR=$J/logs/rcs.txt
: > $JR
judge() {  # judge NAME OUTDIR RUNDIR ARGS...
  local name=$1 out=$2 rundir=$3; shift 3
  local t0; t0=$(date -u +%FT%TZ)
  (cd $T/tree && ISO_WT=$T/tree nice -n 10 $PY -I -B $T/iso.py $T/pilot3_judge.py $rundir --seed 42 \
      --expect-head $HEAD --out $out "$@") > $J/logs/$name.log 2>&1
  local rc=$?
  local st; st=$(git -C $T/tree status --porcelain --ignored | wc -l)
  printf '%s %s rc=%s out_exists=%s tree_status_entries=%s end=%s\n' "$t0" "$name" "$rc" \
      "$([ -e $out ] && echo yes || echo no)" "$st" "$(date -u +%FT%TZ)" >> $JR
}
judge J1_accept_65536 $J/OUT $RUN --steps 65536 --parent-sha256 $P3 --expect-budget 50000
judge J2_refuse_pilot1_parent $J/OUT_J2 $RUN --steps 65536 --parent-sha256 $P1 --expect-budget 50000
judge J3_refuse_no_drop $J/OUT_J3 $C/no_drop --steps 65536 --parent-sha256 $P3 --expect-budget 50000
judge J4_refuse_no_reset $J/OUT_J4 $C/no_reset --steps 65536 --parent-sha256 $P3 --expect-budget 50000
judge J5_refuse_n_envs_2 $J/OUT_J5 $C/n_envs_2 --steps 65536 --parent-sha256 $P3 --expect-budget 50000
judge J6_refuse_other_head $J/OUT_J6 $C/other_head --steps 65536 --parent-sha256 $P3 --expect-budget 50000
judge J7_fallback_accept $J/OUT $RUN --steps 50000 --parent-sha256 $P3 --expect-budget 50000 --final-fallback
judge J8_fallback_refuse_49152 $J/OUT_J8 $C/fallback_wrong --steps 50000 --parent-sha256 $P3 --expect-budget 50000 --final-fallback
judge J9_resume_refuse_off_schedule $J/OUT_J9 $RTEST --steps 65536 --parent-sha256 $P3 --expect-budget 50000
judge J10_resume_report_off_schedule $J/OUT $RTEST --steps 65536 --parent-sha256 $P3 --expect-budget 50000 --report-off-schedule
find $J/OUT -maxdepth 5 -type d | sort > $J/logs/OUT_dirs.txt
cp -p $JR $O/judge_rcs.txt
nice -n 10 $PY -I -B $X/bin/compare_amend.py $T $O $J $V > $O/compare_with_validate.txt 2>&1; echo "compare_amend rc=$?" >> $RCS
echo "tree status lines after: $(git -C tree status --porcelain --ignored | wc -l)" >> $RCS
sha256sum $FILES > $O/sha256_after.txt
cmp $O/sha256_before.txt $O/sha256_after.txt && echo "tool files unchanged during the run" >> $RCS
date -u +%FT%TZ > $O/ended.txt
