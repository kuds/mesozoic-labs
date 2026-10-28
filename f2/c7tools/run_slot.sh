#!/bin/bash
# One simulation slot: runs its job list sequentially (so two slots = at most 2 simulation processes).
# Usage: run_slot.sh JOBS.tsv SLOTNAME
set -u
JOBS="$1"; SLOT="$2"
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
TOOL=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/lean/build/laneA/tools/measure_r10round.py
COPY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2/copy_8737cd3d
LOGS=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2/logs/measure
SUMMARY="$LOGS/summary_${SLOT}.tsv"
mkdir -p "$LOGS"
cd /tmp || exit 9
while IFS=$'\t' read -r RUN LABEL KIND ZIP PKL SC; do
  [ -z "$RUN" ] && continue
  N=1
  while [ -e "$LOGS/${RUN}_${LABEL}_${KIND}.try${N}.log" ]; do N=$((N+1)); done
  LOG="$LOGS/${RUN}_${LABEL}_${KIND}.try${N}.log"
  START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  {
    echo "start $START slot=$SLOT try=$N"
    echo "cmd: OMP_NUM_THREADS=1 nice -n 10 $PY -I -B $TOOL $KIND --copy $COPY --zip $ZIP --pkl $PKL --stage-config $SC --run-id $RUN --label $LABEL"
  } > "$LOG"
  OMP_NUM_THREADS=1 nice -n 10 "$PY" -I -B "$TOOL" "$KIND" --copy "$COPY" --zip "$ZIP" --pkl "$PKL" \
      --stage-config "$SC" --run-id "$RUN" --label "$LABEL" >> "$LOG" 2>&1 < /dev/null
  RC=$?
  END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "end rc=$RC $END" >> "$LOG"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$RUN" "$LABEL" "$KIND" "$N" "$RC" "$START" "$END" >> "$SUMMARY"
done < "$JOBS"
echo "slot $SLOT finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
