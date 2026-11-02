#!/bin/bash
# usage: lc.sh RUN_ID TAG -- runs the pinned launch-check (pins as the tool defaults), full output kept unread in
# launch_check/out/; prints only each fact's status and name, the FAULTS line and the tool's exit code
RUN=$1; TAG=$2
OUT=/home/user/work/pc_study/round/launch_check/out/$RUN.$TAG
cd /tmp || exit 9
export TMPDIR=/home/user/work/pc_study/round/launch_check/tmp
[ -e "$OUT.out" ] && { echo "REFUSED: $OUT.out exists"; exit 2; }
nice -n 10 /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I -B /home/user/work/pc_study/tools/pc/complete_r10round.py launch-check --run "$RUN" --round /home/user/work/pc_study/round --json "$OUT.json" > "$OUT.out" 2>&1
RC=$?
echo "tool exit $RC; full output kept unread at $OUT.out (json $OUT.json)"
grep -c "^RUNLOG line" "$OUT.out" | sed 's/^/RUNLOG other-form lines printed (read by no rule): /'
grep -E "^LAUNCH CHECK " "$OUT.out" | sed -E 's/; launch line: .*$/; launch line: (present)/'
grep -E "^  (PASS|FAIL|NOT COMPUTED|NOT ESTABLISHED) " "$OUT.out" | awk '{st=substr($0,3,16); rest=substr($0,20); i=index(rest,": "); name=(i>0?substr(rest,1,i-1):rest); gsub(/ +$/,"",st); note=""; if (st=="NOT COMPUTED") { j=index(rest," [not computed"); if (j>0) note=substr(rest,j) }; print "  " st " | " name note}'
grep -E "^  FAULTS \(computed\)" "$OUT.out"
grep -E "^INPUT ERROR" "$OUT.out"
exit 0
