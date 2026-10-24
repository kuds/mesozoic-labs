#!/bin/bash
# queue5r.sh judge|panels [--dry-run | --once | --watch | --stop] : the stance5 replication judging and panels queues.
# HOW TO RUN IT IN THIS CONTAINER (the stance5 lesson, run_log 2026-10-21 23:48 UTC): a detached setsid worker is not
# tracked by the harness, and the idle container is reclaimed about 10 minutes after a turn ends, killing it.  Run each
# worker as a harness-tracked background command, by its ABSOLUTE path, under a time limit, and relaunch it when it exits:
#     timeout 7000 /tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/replication/tools/queue5r.sh judge --workerwatch
#     timeout 7000 /tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/replication/tools/queue5r.sh panels --workerwatch
# (Bash run_in_background, one command each).  A job cut by the limit leaves no output and is rerun by the next worker.
# --watch below (setsid nohup) is stance5's detached form, kept unchanged; do not rely on it here.  Never launch it as
# "bash queue5r.sh": the worker relaunches "$0" under nohup and needs a path (stance5, 2026-10-21 20:32 UTC).
# Everything below is SP/foot/stance5/tools/queue5.sh with the replication runs (s70, s80), judge_one5r.sh, panels5r_one.sh
# and the replication root (S5 below names SP/foot/stance5/replication), plus one status: "refused", a full-mode slot
# whose judge refused on the run's own records (judge_one5r.sh wrote refused_<steps>.json: an invalid J file, criteria
# RUN HANDLING).  It is not rerun, its panels and final extras are skipped, and it counts as settled.  And "review": a
# refusal on the checkpoint's own attributes only (refused_<steps>.review.json), held until the coordinator confirms it
# (renamed to refused_<steps>.json) or moves it aside; it is not rerun and not settled.
#   queue5r.sh judge --watch    detach a worker (setsid nohup) that judges every placed, unjudged checkpoint of the two
#                              replication runs, one job at a time, then re-scans every 5 minutes for newly placed ones.
#   queue5r.sh judge --once     the same, one pass, then exit.
#   queue5r.sh judge --dry-run  list what a pass would run now.  Writes nothing.
#   queue5r.sh judge --stop     ask the worker to stop after its current job (creates the STOP file).
#   queue5r.sh panels ...       the same for the report-only diagnostic panels at the J checkpoints (after that slot's judge).
# Re-run the launcher after a container restart: finished jobs are skipped, an incomplete output is moved aside and
# rerun (judge_one5r.sh / panels5r_one.sh do that), and a job that failed MAX_FAILS times is skipped until its
# state/<job>.fails file is deleted.  One worker per kind (flock).  The worker exits when every job is done, on the
# STOP file, after 30 hours (stance4 had 20; the band runs take longer: about 13-16 h plus the final judging), or below 1 GB free disk.
# Judge order per pass: full mode at J (4194304 5242880 6291456 8388608 10485760 final), then the report grid
# (1048576 2097152 3145728 7340032 9437184 11534336), then best and robust_best, then quick mode at the other placed
# 524,288 checkpoints, then at final: fresh, tail and common-noise.  Panels: J only, four variants, panels5r_one.sh --wait.
# Budget: one judge process and one panels process at a time (nice -n 19 for panels; the panels worker also waits for
# any other panels4.py or panels5.py).
# Logs: R/judged/logs/queue5_judge.log, R/panels/out/logs/queue5_panels.log; per job: see judge_one5r.sh, panels5r_one.sh.
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
S5=$S/foot/stance5/replication; T=$S5/tools   # R
KIND=$1; MODE=$2
case "$KIND" in
  judge) LOGD=$S5/judged/logs;;
  panels) LOGD=$S5/panels/out/logs;;
  -h|--help|"") awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/,""); print}' "$0"; exit 0;;
  *) echo "usage: queue5r.sh judge|panels [--dry-run|--once|--watch|--stop]"; exit 2;;
esac
LOG=$LOGD/queue5_$KIND.log; LOCK=$LOGD/queue5_$KIND.lock; STATE=$LOGD/queue5_${KIND}_state; STOP=$LOGD/queue5_$KIND.STOP
MAX_FAILS=2; MIN_FREE_KB=1048576; MAX_HOURS=30; SLEEP=300
RUNS="stance5_LTSB_s70:70 stance5_LTSB_s80:80"
J="4194304 5242880 6291456 8388608 10485760 final"
REPORT="1048576 2097152 3145728 7340032 9437184 11534336"
NAMED="best robust_best"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
say() { echo "$(date -Is) $*" >> $LOG; }

placed_periodic() {  # placed_periodic RUN -> the placed periodic step counts, ascending
  ls $S5/runs/$1/01_stance/models/ 2>/dev/null | sed -n 's/^stage1_\([0-9]*\)_steps\.zip$/\1/p' | sort -n
}

joblist() {  # one line per job: RUN SEED STEPS MODE (judge) or RUN SEED STEPS - (panels)
  local r run seed s
  if [ "$KIND" = panels ]; then
    for s in $J; do for r in $RUNS; do echo "${r%%:*} ${r##*:} $s -"; done; done; return
  fi
  for s in $J $REPORT $NAMED; do for r in $RUNS; do echo "${r%%:*} ${r##*:} $s full"; done; done
  for r in $RUNS; do run=${r%%:*}; seed=${r##*:}
    for s in $(placed_periodic $run); do
      case " $J $REPORT " in *" $s "*) ;; *) echo "$run $seed $s quick";; esac
    done
  done
  for m in fresh tail common-noise; do for r in $RUNS; do echo "${r%%:*} ${r##*:} final $m"; done; done
}

status_of() {  # status_of RUN SEED STEPS MODE -> done | ready | missing | waiting-judge | refused | review | failed
  local run=$1 seed=$2 steps=$3 mode=$4 key=$1_$3_$4 rc
  if [ "$KIND" = judge ]; then
    $T/judge_one5r.sh --check $steps $run $seed $mode; rc=$?
    if [ $mode != full ] && [ $mode != quick ] && [ $rc -eq 1 ]; then  # final extras wait for the full final judge
      $T/judge_one5r.sh --check final $run $seed full; case $? in 0) ;; 5) echo refused; return;; 6) echo review; return;; *) echo waiting-judge; return;; esac
    fi
  else
    $T/panels5r_one.sh --check $steps $run; rc=$?
    if [ $rc -eq 1 ]; then $T/judge_one5r.sh --check $steps $run $seed full; case $? in 0) ;; 5) echo refused; return;; 6) echo review; return;; *) echo waiting-judge; return;; esac; fi
  fi
  case $rc in 0) echo done;; 3) echo missing;; 5) echo refused;; 6) echo review;;
    *) if [ -f $STATE/$key.fails ] && [ "$(wc -l < $STATE/$key.fails)" -ge $MAX_FAILS ]; then echo failed; else echo ready; fi;; esac
}

run_job() {  # run_job RUN SEED STEPS MODE
  local run=$1 seed=$2 steps=$3 mode=$4 key=$1_$3_$4 rc t0
  say "run $run $steps $mode"; t0=$(date +%s)
  if [ "$KIND" = judge ]; then $T/judge_one5r.sh $steps $run $seed $mode; rc=$?
  else $T/panels5r_one.sh --wait $steps $run; rc=$?; fi
  if [ $rc -eq 0 ]; then say "ok $run $steps $mode $(( $(date +%s) - t0 ))s"
  elif [ $rc -eq 5 ]; then say "refused $run $steps $mode: the judge refused on the run's own records (refused_$steps.json, an invalid J file; criteria RUN HANDLING)"
  elif [ $rc -eq 6 ]; then say "review $run $steps $mode: the judge refused on the checkpoint's own attributes only (refused_$steps.review.json, held for review; tools/README.md rule 2)"
  else echo "$(date -Is) rc=$rc" >> $STATE/$key.fails; say "FAIL $run $steps $mode rc=$rc $(( $(date +%s) - t0 ))s"; fi
}

one_pass() {  # runs every ready job once; returns 0 if something ran, 1 if nothing was ready
  local ran=1 line st free
  while read -r line; do
    set -- $line
    [ -e $STOP ] && { say "STOP file present; stopping"; return 2; }
    st=$(status_of $1 $2 $3 $4)
    [ "$st" = ready ] || continue
    free=$(df -Pk $S | awk 'NR==2{print $4}')
    [ "$free" -lt $MIN_FREE_KB ] && { say "STOP: only $free KB free"; return 2; }
    run_job $1 $2 $3 $4; ran=0
  done < <(joblist)
  return $ran
}

all_done() { local line; while read -r line; do set -- $line; case "$(status_of $1 $2 $3 $4)" in done|refused) ;; *) return 1;; esac; done < <(joblist); return 0; }

case "$MODE" in
--dry-run)
  if [ -e $LOCK ] && ! flock -n $LOCK true 2>/dev/null; then echo "worker: running (log $LOG)"; else echo "worker: not running"; fi
  [ -e $STOP ] && echo "STOP file present: $STOP"
  plan=$(joblist | while read -r line; do set -- $line; echo "$(status_of $1 $2 $3 $4) $1 $3 $4"; done)
  echo "$plan"; echo "summary: $(echo "$plan" | awk '{print $1}' | sort | uniq -c | tr -s ' \n' ' ')"
  exit 0 ;;
--stop) mkdir -p $LOGD; touch $STOP; echo "created $STOP"; exit 0 ;;
--once|--watch)
  mkdir -p $LOGD
  if [ -e $LOCK ] && ! flock -n $LOCK true 2>/dev/null; then echo "worker already running (log $LOG)"; exit 0; fi
  rm -f $STOP
  setsid nohup "$0" $KIND --worker${MODE#--} >> $LOG 2>&1 < /dev/null &
  echo "launched $KIND worker pid $! (${MODE#--}; log $LOG)"; exit 0 ;;
--workeronce|--workerwatch)
  mkdir -p $LOGD $STATE
  exec 9>>$LOCK
  flock -n 9 || { say "worker already running; exit"; exit 0; }
  say "worker start pid $$ (${MODE#--worker})"
  t_end=$(( $(date +%s) + MAX_HOURS * 3600 ))
  while :; do
    one_pass; r=$?
    [ $r -eq 2 ] && break
    if all_done; then say "all jobs done"; break; fi
    [ "$MODE" = --workeronce ] && break
    [ $(date +%s) -ge $t_end ] && { say "stop: $MAX_HOURS h elapsed"; break; }
    [ $r -eq 1 ] && sleep $SLEEP
  done
  say "worker end; plan now:"
  joblist | while read -r line; do set -- $line; echo "    $(status_of $1 $2 $3 $4) $1 $3 $4"; done >> $LOG
  exit 0 ;;
*) echo "usage: queue5r.sh judge|panels [--dry-run|--once|--watch|--stop]"; exit 2 ;;
esac
