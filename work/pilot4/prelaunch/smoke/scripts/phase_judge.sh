#!/bin/bash
# Smoke lane: pilot4_judge.py's validation steps on COPIES of the smoke's run folders (nothing trains; every command
# through job.sh, so CMDLOG.txt and judge/status.txt record it). TREE is this lane's clean clone at the head.
#   A: the launch copy: --check-only, the cells the judge can judge in validation mode (--expect-budget 50000: the
#      32,768 and 65,536 pairs, the final pair through --final-fallback --steps 50000), a second judgement, the pilot
#      budget, and the fallback refusal;
#   R: refusals, each with its own OUT that must not exist afterwards;
#   D: a dirty tree (one tracked file edited in TREE, restored afterwards);
#   B: the RESUME copy: --check-only, its 65,536 pair (validation), the fallback refusal, a non-periodic resume path.
W=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work)
K=$W/pilot4/prelaunch/smoke
T=$W/pilot4/tools
PY=$W/pilot3/branch/venv_full/bin/python
TREE=$K/judge_tree
HEAD=0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8
RL=$K/runs/launch/pilot4_walk_s42
RR=$K/runs/resume/pilot4_walk_s42
CP=$K/judge/copies
OUT=$K/judge/OUT
OUTR=$K/judge/OUT_resume
LG=$K/judge/logs
J="bash $K/scripts/job.sh"
P42=sha256:41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4
P52=sha256:2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00
mkdir -p "$LG"
JUDGE="cd $TREE && ISO_WT=$TREE $PY -I $T/iso.py $T/pilot4_judge.py"
V="--seed 42 --expect-head $HEAD --expect-budget 50000"
$J "tree_clean_start" "git -C $TREE status --porcelain --ignored --untracked-files=all > $LG/tree_status_start.txt && test ! -s $LG/tree_status_start.txt"
# A: the launch copy
$J "A1_check_only_32768_expect0" "$JUDGE $RL $V --steps 32768 --parent-sha256 $P42 --out $OUT --check-only > $LG/A1_check_only_32768.log 2>&1"
$J "A2_check_only_65536_expect0" "$JUDGE $RL $V --steps 65536 --parent-sha256 $P42 --out $OUT --check-only > $LG/A2_check_only_65536.log 2>&1"
$J "A2_out_absent_after_check_only" "test ! -e $OUT"
$J "A3_judge_65536_expect0" "$JUDGE $RL $V --steps 65536 --parent-sha256 $P42 --out $OUT > $LG/A3_judge_65536.log 2>&1" &
$J "A4_judge_32768_expect0" "$JUDGE $RL $V --steps 32768 --parent-sha256 $P42 --out $OUT > $LG/A4_judge_32768.log 2>&1" &
$J "B2_judge_resumed_65536_expect0" "$JUDGE $RR $V --steps 65536 --parent-sha256 $P42 --out $OUTR > $LG/B2_judge_resumed_65536.log 2>&1" &
wait
$J "A5_judge_again_65536_expect2" "$JUDGE $RL $V --steps 65536 --parent-sha256 $P42 --out $OUT > $LG/A5_judge_again_65536.log 2>&1"
$J "A6_pilot_budget_check_only_expect2" "$JUDGE $RL --seed 42 --expect-head $HEAD --steps 65536 --parent-sha256 $P42 --out $K/judge/OUT_A6 --check-only > $LG/A6_pilot_budget.log 2>&1"
$J "A7_final_fallback_50000_expect0" "$JUDGE $RL $V --steps 50000 --final-fallback --parent-sha256 $P42 --out $OUT > $LG/A7_final_fallback_50000.log 2>&1"
$J "A8_fallback_wrong_num_timesteps_expect2" "$JUDGE $CP/fallback_wrong/pilot4_walk_s42 $V --steps 50000 --final-fallback --parent-sha256 $P42 --out $K/judge/OUT_A8 > $LG/A8_fallback_wrong.log 2>&1"
# R: refusals (each its own OUT)
$J "R1_parent_cli_s52_expect2" "$JUDGE $RL $V --steps 65536 --parent-sha256 $P52 --out $K/judge/OUT_R1 > $LG/R1_parent_cli_s52.log 2>&1"
for c in parent_sha_record command_table env_kwarg_drift other_head n_envs_2 warm_start_log_std critic_dropped parent_task no_lineage; do
  $J "R_${c}_expect2" "$JUDGE $CP/$c/pilot4_walk_s42 $V --steps 65536 --parent-sha256 $P42 --out $K/judge/OUT_R_$c > $LG/R_$c.log 2>&1"
done
$J "R_wrong_run_id_expect2" "$JUDGE $CP/wrong_run_id/pilot3_walk_s42 $V --steps 65536 --parent-sha256 $P42 --out $K/judge/OUT_R_wrong_run_id > $LG/R_wrong_run_id.log 2>&1"
# D: a dirty tree
$J "D1_dirty_make" "printf '\n(smoke lane: a local edit that makes the tree dirty)\n' >> $TREE/README.md && git -C $TREE status --porcelain --untracked-files=no > $LG/D1_dirty_status.txt"
$J "D2_dirty_judge_expect2" "$JUDGE $RL $V --steps 65536 --parent-sha256 $P42 --out $K/judge/OUT_D2 > $LG/D2_dirty_judge.log 2>&1"
$J "D3_dirty_restore" "git -C $TREE checkout -- README.md && git -C $TREE status --porcelain --ignored --untracked-files=all > $LG/D3_tree_status_after_restore.txt && test ! -s $LG/D3_tree_status_after_restore.txt"
# B: the RESUME copy
$J "B1_check_only_resumed_65536_expect0" "$JUDGE $RR $V --steps 65536 --parent-sha256 $P42 --out $K/judge/OUT_B1 --check-only > $LG/B1_check_only_resumed_65536.log 2>&1"
$J "B3_resumed_final_fallback_expect2" "$JUDGE $RR $V --steps 50000 --final-fallback --parent-sha256 $P42 --out $K/judge/OUT_B3 > $LG/B3_resumed_final_fallback.log 2>&1"
$J "B4_resume_nonperiodic_expect2" "$JUDGE $CP/resume_nonperiodic/pilot4_walk_s42 $V --steps 65536 --parent-sha256 $P42 --out $K/judge/OUT_B4 > $LG/B4_resume_nonperiodic.log 2>&1"
$J "refusal_outs_absent" "find $K/judge -maxdepth 1 -name 'OUT_*' > $LG/refusal_outs_present.txt && test ! -s $LG/refusal_outs_present.txt"
$J "tree_clean_end" "git -C $TREE status --porcelain --ignored --untracked-files=all > $LG/tree_status_end.txt && test ! -s $LG/tree_status_end.txt"
