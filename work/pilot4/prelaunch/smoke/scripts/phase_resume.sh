#!/bin/bash
# Smoke lane: the RESUME test from the smoke's own 32,768 pair (every command through job.sh).
#   1. the finished launch run is copied out of the clone (runs/launch/pilot4_walk_s42) and its file list recorded;
#   2. the run directory in the clone is replaced by the copy runner_snapshot.py took right after the 32,768 pair was
#      saved (snap/pilot4_walk_s42: the state a runtime loss there leaves), its 32,768 pair checked equal to the
#      launch's;
#   3. the notebook's RESUME cell through the pinned runner (--phase resume, the same in-memory cadence override), the
#      same RUN_ID in the same clone path; the only training here: 32,768 -> 65,536;
#   4. the resumed run is copied out (runs/resume/pilot4_walk_s42) with its stage_config.json before and after.
W=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work)
K=$W/pilot4/prelaunch/smoke
T=$W/pilot4/tools
PY=$W/pilot3/branch/venv_full/bin/python
CL=$K/clone
RD=$CL/logs/tyrannosaurus_rex/anatomical/ppo_quick_test/pilot4_walk_s42
NB=$K/nb
export JOB_STATUS=$NB/status.txt
J="bash $K/scripts/job.sh"
mkdir -p $K/runs/launch $K/runs/resume $K/data
$J "launch_find_clone_logs" "find $CL/logs -type f | sed 's|^$CL/||' | sort > $K/data/clone_logs_files_after_launch.txt"
$J "launch_copy_out" "cp -a $RD $K/runs/launch/"
$J "launch_files" "cd $K/runs/launch/pilot4_walk_s42 && find . -type f -printf '%P %s\n' | sort > $K/data/launch_run_files.txt && find . -type f -print0 | sort -z | xargs -0 sha256sum > $K/data/launch_run_sha256.txt"
$J "launch_copy_equal" "diff -r $RD $K/runs/launch/pilot4_walk_s42 > $K/data/launch_copy_diff.txt"
$J "snap_pair_equal_launch_pair" "cmp $K/snap/pilot4_walk_s42/03_locomotion/models/stage2_32768_steps.zip $K/runs/launch/pilot4_walk_s42/03_locomotion/models/stage2_32768_steps.zip && cmp $K/snap/pilot4_walk_s42/03_locomotion/models/stage2_vecnormalize_32768_steps.pkl $K/runs/launch/pilot4_walk_s42/03_locomotion/models/stage2_vecnormalize_32768_steps.pkl"
$J "snap_files" "cd $K/snap/pilot4_walk_s42 && find . -type f -printf '%P %s\n' | sort > $K/data/snap_run_files.txt"
$J "clone_run_replace_by_snapshot" "rm -rf $RD && cp -a $K/snap/pilot4_walk_s42 $RD && diff -r $K/snap/pilot4_walk_s42 $RD > $K/data/snap_vs_clone_run_diff.txt"
$J "stage_config_before_resume" "cp $RD/03_locomotion/stage_config.json $NB/resume_stage_config_before.json"
$J "clone_tracked_clean_before_resume" "git -C $CL status --porcelain --untracked-files=no > $K/data/clone_tracked_status_before_resume.txt && test ! -s $K/data/clone_tracked_status_before_resume.txt"
$J "resume_start_time" "date -u +%FT%T.%NZ > $NB/resume.start"
$J "RESUME_phase" "cd $CL && nice -n 10 $PY -I $T/nb_pilot_runner4.py --params $NB/params_s42.json --out $NB/resume --clone $CL --phase resume --cadence-override 32768 > $NB/resume.console.txt 2>&1"
$J "resume_end_time" "date -u +%FT%T.%NZ > $NB/resume.end"
$J "stage_config_after_resume" "cp $RD/03_locomotion/stage_config.json $NB/resume_stage_config_after.json"
$J "resume_copy_out" "cp -a $RD $K/runs/resume/"
$J "resume_files" "cd $K/runs/resume/pilot4_walk_s42 && find . -type f -printf '%P %s\n' | sort > $K/data/resume_run_files.txt && find . -type f -print0 | sort -z | xargs -0 sha256sum > $K/data/resume_run_sha256.txt"
$J "clone_tracked_clean_after_resume" "git -C $CL status --porcelain --untracked-files=no > $K/data/clone_tracked_status_after_resume.txt && test ! -s $K/data/clone_tracked_status_after_resume.txt"
