#!/bin/sh
# Watchdog template: start every chain that is neither done nor running. Generalised from the Stage 1b speed round's
# watchdog, which kept three body chains of stage1/runs/2026-11-10-speed/run_speed_round.sh going through container
# restarts (LESSONS_LEARNED.md §4.2). Copy it, set CHAINS, CHAIN_CMD and DONE_LOG, and pass it to tools/keepalive.sh:
#
#   sh tools/keepalive.sh LOG sh tools/my_watchdog.sh       (run_in_background, timeout 7200000)
#
# Each chain must be idempotent: resume from its own checkpoints and skip the records it already has, so that starting
# it twice, or after a restart, loses nothing. The chain marks itself done by writing " <chain> done" as the last word
# of a line in DONE_LOG (run_speed_round.sh does). Run from hardware/trex_form_study.
CHAINS="60-N-T-10 60-N-H-10 60-N-T-10-K"                                   # one chain per body
CHAIN_CMD="bash stage1/runs/2026-11-10-speed/run_speed_round.sh all"       # the chain, given a body
DONE_LOG=stage1/runs/2026-11-10-speed/logs/run_speed_round.log             # where each chain writes "<chain> done"
export PY=${PY:-.venv/bin/python}                                          # the chains read PY; the system python lacks MuJoCo
echo "$(date -u +%H:%M:%S) watchdog: up $(cut -d' ' -f1 /proc/uptime | cut -d. -f1)s, $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB free"
for c in $CHAINS; do
  if grep -q " $c done$" "$DONE_LOG" 2>/dev/null; then echo "  $c: done"; continue; fi
  if pgrep -f "$CHAIN_CMD $c\$" >/dev/null; then echo "  $c: running"; continue; fi
  setsid nohup $CHAIN_CMD "$c" > /dev/null 2>&1 < /dev/null &
  echo "  $c: started"
done
