#!/bin/sh
# Memory guard for a shared 4-core, 15 GB machine (used in Stage 1, when training jobs, evaluations and workflow agents
# ran side by side). Every 15 s: Python processes whose command line does not contain PROTECT get oom_score_adj 800, so
# that the kernel's OOM killer takes them before the protected jobs; when MemAvailable falls below MIN_MB, the largest
# of them gets SIGTERM (a job that writes its records as it goes loses only its unfinished episode and resumes later).
# Logs a line a minute and every action.
#
#   sh tools/memguard.sh LOG          start it detached:  setsid nohup sh tools/memguard.sh LOG >/dev/null 2>&1 &
#   env: PROTECT="designlab.stage1 train"  MIN_MB=1500
LOG=${1:?usage: memguard.sh LOG}
PROTECT=${PROTECT:-designlab.stage1 train}
MIN_MB=${MIN_MB:-1500}
i=0
while true; do
  big=""
  bigrss=0
  for p in $(pgrep -f python); do
    c=$(tr '\0' ' ' < /proc/"$p"/cmdline 2>/dev/null)
    case "$c" in
      *"$PROTECT"*) ;;
      *python*)
        a=$(cat /proc/"$p"/oom_score_adj 2>/dev/null)
        [ -n "$a" ] && [ "$a" -lt 800 ] && echo 800 > /proc/"$p"/oom_score_adj 2>/dev/null
        r=$(awk '/VmRSS/{print $2}' /proc/"$p"/status 2>/dev/null)
        [ -n "$r" ] && [ "$r" -gt "$bigrss" ] && { bigrss=$r; big=$p; } ;;
    esac
  done
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt "$MIN_MB" ] && [ -n "$big" ]; then
    echo "$(date -u +%H:%M:%S) LOW $avail MB: SIGTERM $big ($((bigrss / 1024)) MB): $(tr '\0' ' ' < /proc/"$big"/cmdline | cut -c1-200)" >> "$LOG"
    kill "$big"
  fi
  i=$((i + 1))
  [ $((i % 4)) -eq 0 ] && echo "$(date -u +%H:%M:%S) $avail MB available, load $(cut -d' ' -f1 /proc/loadavg)" >> "$LOG"
  sleep 15
done
