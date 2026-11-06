#!/bin/bash
# Smoke lane: the end-of-lane checks (every command through job.sh): the pinned tools' SHA256SUMS.txt again (from the
# tools folder, as pinned), the list's own sha256 unchanged, no file under the tools folder (the tree aside) or under
# pilot 3's tools and run copies newer than the lane's start marker, the read-only tree clean at the head, the parent
# run copy's sha256 unchanged, and no bytecode anywhere in this lane's clones.
W=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work)
K=$W/pilot4/prelaunch/smoke
T=$W/pilot4/tools
P3=$W/pilot3
export JOB_STATUS=$K/pin/status.txt
J="bash $K/scripts/job.sh"
$J "sha256sums_check_end" "cd $T && sha256sum -c SHA256SUMS.txt > $K/pin/sha256sums_check_end.txt 2> $K/pin/sha256sums_check_end.stderr.txt"
$J "sha256sums_ok_count_end" "grep -c ': OK\$' $K/pin/sha256sums_check_end.txt > $K/pin/sha256sums_ok_count_end.txt"
$J "sha256sums_list_sha_end" "cd $T && sha256sum SHA256SUMS.txt > $K/pin/sha256sums_list_sha_end.txt && cmp $K/pin/sha256sums_list_sha_start.txt $K/pin/sha256sums_list_sha_end.txt"
$J "tools_newer_than_start" "find $T -newer $K/tmp/marker_launch_s42 -not -path '$T/tree/*' -not -path '$T/tree' > $K/pin/tools_newer_than_start.txt; test ! -s $K/pin/tools_newer_than_start.txt"
$J "pilot3_newer_than_start" "find $P3/tools $P3/runs/final -newer $K/tmp/marker_launch_s42 > $K/pin/pilot3_newer_than_start.txt; test ! -s $K/pin/pilot3_newer_than_start.txt"
$J "tree_head_end" "git -C $T/tree rev-parse HEAD > $K/pin/tree_head_end.txt && cmp $K/logs/tree_head_start.txt $K/pin/tree_head_end.txt"
$J "tree_status_end" "git -C $T/tree status --porcelain --ignored --untracked-files=all > $K/pin/tree_status_end.txt && test ! -s $K/pin/tree_status_end.txt"
$J "tree_index_mtime_end" "stat -c '%y' $T/tree/.git/index > $K/pin/tree_index_mtime_end.txt"
$J "parent_run_copy_sha_end" "cd $P3/runs/final && sha256sum pilot3_walk_s42/03_locomotion/models/stage2_8000000_steps.zip pilot3_walk_s42/03_locomotion/models/stage2_vecnormalize_8000000_steps.pkl > $K/data/parent_pair_sha256_run_copy_end.txt && cmp $K/data/parent_pair_sha256_run_copy_start.txt $K/data/parent_pair_sha256_run_copy_end.txt"
$J "parent_my_copy_sha_end" "cd $K/pairs && sha256sum pilot3_walk_s42/03_locomotion/models/stage2_8000000_steps.zip pilot3_walk_s42/03_locomotion/models/stage2_vecnormalize_8000000_steps.pkl > $K/data/parent_pair_sha256_my_copy_end.txt && cmp $K/data/parent_pair_sha256_my_copy_start.txt $K/data/parent_pair_sha256_my_copy_end.txt"
$J "no_bytecode_in_lane" "find $K $T -name '*.pyc' -o -name __pycache__ > $K/pin/bytecode_found.txt; test ! -s $K/pin/bytecode_found.txt"
