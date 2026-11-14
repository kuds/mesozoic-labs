#!/bin/sh
# Heartbeat for long jobs in a cloud session (LESSONS_LEARNED.md §4.2, "Restarts, keep-alive and resuming").
#
# Run it with the Bash tool's run_in_background AND timeout 7200000 (the default timeout, 1800000, stops it after 30
# min), so that the session tracks it: while it runs the container is less likely to be reclaimed as idle,
# and when it ends the session is re-invoked, which is the cue to re-arm it. It writes one line every STEP seconds
# (uptime, load, free memory) to LOG, runs WATCHDOG (if given) every EVERY lines, and stops after MAX seconds or as soon
# as STOP_FILE exists. A restart of the container kills it like every other process: re-arm it, and the watchdog,
# after any restart (an uptime that went down in LOG shows one).
#
#   sh tools/keepalive.sh LOG [WATCHDOG...]      LOG outside the checkout, or under a git-ignored logs/ directory
#   Better, run a copy (sh reads a script as it goes, so editing this file mid-run breaks the running loop):
#   cp tools/keepalive.sh /tmp/keepalive_run.sh && sh /tmp/keepalive_run.sh LOG
#   env: STEP=120  MAX=6900 (just under two hours)  EVERY=5  STOP_FILE=<LOG>.stop
#
# WATCHDOG is a command that restarts any job that is neither done nor running, and is safe to run at any time; each
# job must therefore resume from its own checkpoint (see stage1/runs/2026-11-10-speed/run_speed_round.sh for a chain
# that does). Its output is appended to LOG.
LOG=${1:?usage: keepalive.sh LOG [WATCHDOG...]}
shift
STEP=${STEP:-120}
MAX=${MAX:-6900}
EVERY=${EVERY:-5}
STOP_FILE=${STOP_FILE:-$LOG.stop}
start=$(date +%s)
i=0
while [ $(( $(date +%s) - start )) -lt "$MAX" ]; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) up $(cut -d' ' -f1 /proc/uptime | cut -d. -f1)s load $(cut -d' ' -f1 /proc/loadavg) mem_avail $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB" >> "$LOG"
  if [ $# -gt 0 ] && [ $((i % EVERY)) -eq $((EVERY - 1)) ]; then
    "$@" >> "$LOG" 2>&1
  fi
  [ -e "$STOP_FILE" ] && { echo "$(date -u +%H:%M:%S) stop file found" >> "$LOG"; break; }
  i=$((i + 1))
  sleep "$STEP"
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) keepalive ended after $(( $(date +%s) - start )) s" >> "$LOG"
