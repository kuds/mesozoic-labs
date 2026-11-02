#!/bin/bash
# Sequential measuring runner for stance7_r10LTSBoff_s42 (one simulation process at a time).
# Usage: run_seq.sh ITEM [ITEM ...]   with ITEM = c8:<label> | m1:<label> | m3:<label>, label = <steps> | final
# Each item: stdout <kind>_<label>.out, stderr <kind>_<label>.err, exit code <kind>_<label>.exit; one line in cmdlog.txt.
cd /tmp || exit 9
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
O=/home/user/work/pc_study/f2/measure/stance7_r10LTSBoff_s42
RUN=stance7_r10LTSBoff_s42
COPY=/home/user/work/pc_study/f2/copy_1dfe2327
M=/home/user/work/pc_study/round/fetch/F2/$RUN/copies/01_stance/models
SC=/home/user/work/pc_study/round/fetch/F2/$RUN/copies/01_stance/stage_config.json.copy1
T=/home/user/work/pc_study/tools/pc
for item in "$@"; do
  kind=${item%%:*}; label=${item#*:}
  if [ "$label" = final ]; then Z=$M/stage1_final.zip.copy1; P=$M/stage1_final_vecnorm.pkl.copy1
  else Z=$M/stage1_${label}_steps.zip.copy1; P=$M/stage1_vecnormalize_${label}_steps.pkl.copy1; fi
  case $kind in
    c8) CMD=("$PY" -I -B "$T/manip_check.py" --copy "$COPY" --zip "$Z" --pkl "$P" --stage-config "$SC" --run-id "$RUN" --label "$label") ;;
    m1|m3) CMD=("$PY" -I -B "$T/measure_r10round.py" "$kind" --copy "$COPY" --zip "$Z" --pkl "$P" --stage-config "$SC" --run-id "$RUN" --label "$label") ;;
    *) echo "unknown kind $kind" >> "$O/run_seq.log"; exit 8 ;;
  esac
  F=$O/${kind}_${label}
  df -B1 / > "$F.df_before"
  S=$(date -u +%s); SU=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "$SU start $kind $label" >> "$O/run_seq.log"
  nice -n 10 env OMP_NUM_THREADS=1 "${CMD[@]}" > "$F.out" 2> "$F.err"
  RC=$?
  E=$(date -u +%s); EU=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "exit=$RC" > "$F.exit"
  echo "$SU .. $EU  rc=$RC wall=$((E-S))s  [${kind}_${label}] nice -n 10 env OMP_NUM_THREADS=1 ${CMD[*]} > ${kind}_${label}.out 2> ${kind}_${label}.err" >> "$O/cmdlog.txt"
  echo "$EU end $kind $label rc=$RC wall=$((E-S))s" >> "$O/run_seq.log"
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ALL DONE" >> "$O/run_seq.log"
