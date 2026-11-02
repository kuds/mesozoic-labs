#!/bin/bash
# run_one.sh TAG KIND LABEL : runs one pinned measuring tool on stance7_r10LTSBpc_s60's F2 copy of LABEL's pair.
# KIND is manip, m1 or m3. Writes TAG.out/.err/.exit/.time in this folder and one line to cmdlog.txt.
set -u
D=/home/user/work/pc_study/f2/measure/stance7_r10LTSBpc_s60
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
TOOLS=/home/user/work/pc_study/tools/pc
COPY=/home/user/work/pc_study/f2/copy_1dfe2327
M=/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBpc_s60/copies/01_stance/models
SC=/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBpc_s60/copies/01_stance/stage_config.json.copy1
RUN=stance7_r10LTSBpc_s60
TAG=$1; KIND=$2; LABEL=$3
if [ "$LABEL" = "final" ]; then
  Z=$M/stage1_final.zip.copy1; P=$M/stage1_final_vecnorm.pkl.copy1
else
  Z=$M/stage1_${LABEL}_steps.zip.copy1; P=$M/stage1_vecnormalize_${LABEL}_steps.pkl.copy1
fi
case "$KIND" in
  manip) TOOL=("$TOOLS/manip_check.py") ;;
  m1)    TOOL=("$TOOLS/measure_r10round.py" m1) ;;
  m3)    TOOL=("$TOOLS/measure_r10round.py" m3) ;;
  *) echo "unknown kind $KIND"; exit 9 ;;
esac
CMD=(nice -n 10 env OMP_NUM_THREADS=1 "$PY" -I -B "${TOOL[@]}" --copy "$COPY" --zip "$Z" --pkl "$P" --stage-config "$SC" --run-id "$RUN" --label "$LABEL")
cd /tmp || exit 9
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ); S=$(date -u +%s)
echo "$T0" > "$D/$TAG.start"
"${CMD[@]}" > "$D/$TAG.out" 2> "$D/$TAG.err"
RC=$?
E=$(date -u +%s); T1=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "$RC" > "$D/$TAG.exit"
echo "start=$T0 end=$T1 wall_s=$((E-S))" > "$D/$TAG.time"
echo "$T0 rc=$RC wall=$((E-S))s end=$T1 [$TAG] CMD: ${CMD[*]} > $D/$TAG.out 2> $D/$TAG.err" >> "$D/cmdlog.txt"
exit $RC
