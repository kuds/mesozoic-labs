#!/bin/sh
# PC design: are the proposed RUN_IDs / RUNLOG KIND new? Read-only greps; output to stdout.
S1=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
RUNLOG=$S1/foot/stance4/live/run_log.txt
echo "RUNLOG lines: $(wc -l < $RUNLOG)"
for pat in stance7 STANCE7 r10LTSBpc phasecritic; do
  echo "RUNLOG lines matching '$pat': $(grep -c "$pat" $RUNLOG)"
done
echo "the round's last complete ppo listing (folders.json _ppo_listing):"
grep -A 30 '"_ppo_listing"' $S1/foot/stance6/fetch/folders.json | grep -c '"stance7' | sed 's/^/  folders named stance7*: /'
grep -m1 -A2 '"_ppo_listing"' $S1/foot/stance6/fetch/folders.json | tail -1
