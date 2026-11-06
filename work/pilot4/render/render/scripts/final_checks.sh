#!/bin/bash
# Read-only end checks: pinned tools list, tools/tree and this lane's clone clean at 0e6a57b, the smoke's files and
# pilot 3's wrapper unchanged, nothing new under the pinned tools, the smoke or pilot 3's folders since this lane
# started, and what lies outside this folder that Xvfb could have touched.
R=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/render/render
W=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work
S=$W/pilot4/prelaunch/smoke
export GIT_OPTIONAL_LOCKS=0
echo "== pinned tools: sha256sum -c"
(cd $W/pilot4/tools && sha256sum -c SHA256SUMS.txt > $R/logs/sha_after.txt 2>&1; echo "rc=$? OK=$(grep -c ': OK$' $R/logs/sha_after.txt) not_OK=$(grep -vc ': OK$' $R/logs/sha_after.txt)")
sha256sum $W/pilot4/tools/SHA256SUMS.txt
echo "== tools/tree and the lane clone"
for t in $W/pilot4/tools/tree $R/tree; do echo "$t HEAD $(git -C $t rev-parse HEAD) status_entries $(git -C $t status --porcelain --ignored | wc -l)"; done
stat -c '%Y %n' $W/pilot4/tools/tree/.git/index
git -C $R/tree remote -v
echo "== the smoke's judged pair and models unchanged"
sha256sum $S/runs/launch/pilot4_walk_s42/03_locomotion/models/stage2_65536_steps.zip $S/runs/launch/pilot4_walk_s42/03_locomotion/models/stage2_vecnormalize_65536_steps.pkl | diff - $R/logs/judged_pair_sha_start.txt && echo judged_pair_unchanged
stat -c '%Y %s %n' $S/runs/launch/pilot4_walk_s42/03_locomotion/models/* | diff - $R/logs/smoke_models_stat_start.txt && echo smoke_models_stat_unchanged
(cd $S/runs/launch/pilot4_walk_s42 && find . -type f -print0 | sort -z | xargs -0 sha256sum) | diff - $R/logs/run_orig_sha.txt && echo smoke_run_unchanged
(cd $S/judge/OUT/s42/65536 && sha256sum *) | diff - $R/logs/cell_orig_sha.txt && echo smoke_cell_unchanged
echo "== files newer than this lane's start (14:06 UTC) under the pinned tools, the smoke, pilot 3, the frozen candidate folder"
find $W/pilot4/tools $S $W/pilot3 -newermt '2026-11-06 14:06:00' 2>&1 | head -20
echo "count: $(find $W/pilot4/tools $S $W/pilot3 -newermt '2026-11-06 14:06:00' 2>&1 | wc -l)"
echo "== outside: /tmp/.X11-unix, /var/lib/xkb, home caches"
ls -la --time-style=full-iso -d /tmp/.X11-unix /var/lib/xkb; ls -A /var/lib/xkb
find /root/.cache /root/.config /var/lib/xkb /tmp/.X11-unix -newermt '2026-11-06 14:06:00' 2>&1 | head
echo "== bytecode in the lane clone"
find $R/tree -name '__pycache__' -o -name '*.pyc' | wc -l
