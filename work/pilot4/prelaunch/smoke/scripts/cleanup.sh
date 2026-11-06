#!/bin/bash
# Smoke lane: delete this lane's own clones and temp folder (every command through job.sh), then record what is left.
# Kept: runs/ (the launch and RESUME run copies, the zero-action baseline records), snap/ (the run at its 32,768 pair),
# judge/ (cells, logs, tampered copies), readouts/, nb/, out/, data/, pin/, pairs/ (the read-only parent copy), scripts.
W=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work)
K=$W/pilot4/prelaunch/smoke
export JOB_STATUS=$K/pin/cleanup_status.txt
J="bash $K/scripts/job.sh"
$J "size_before_cleanup" "du -sh $K/clone $K/judge_tree $K/tmp > $K/pin/size_before_cleanup.txt 2>&1"
$J "clone_status_before_delete" "git -C $K/clone status --porcelain --untracked-files=no > $K/pin/clone_status_before_delete.txt && git -C $K/judge_tree status --porcelain --ignored --untracked-files=all > $K/pin/judge_tree_status_before_delete.txt"
$J "delete_clones" "rm -rf $K/clone $K/judge_tree"
# job.sh itself makes TMPDIR ($K/tmp) again, so the temp folder is deleted by the caller after this script.
