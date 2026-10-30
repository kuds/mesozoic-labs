#!/bin/bash
# Logged command runner for the LB criteria designer (LB/criteria_work only).
# Usage: lg.sh "<description>" '<command>'
# Exports the short names (S1, S2, LB, CW, BS, BS2, F2, FMAN, PY, RUNLOG) so commands need no literal paths; runs the
# command with bash -c; captures stdout+stderr into out/NNNN.txt; writes the path prefixes back as their short names
# and the CLI's temp-folder name as <cli> (the task's rule: no AI product name in any file); appends to CMD_LOG.txt the
# start/end times (date -u), the command (same substitutions), the exit code, the output file with its byte count and
# sha256 prefix and its first 15 lines; then prints the output.
CW=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
S2=${CW%/r10round/long_blind/criteria_work}
S1="$(dirname "$(dirname "$S2")")/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
TMPROOT=$(dirname "$(dirname "$(dirname "$S2")")")
LB=$S2/r10round/long_blind; BS=$S2/r10round/bias_screen; BS2=$S2/r10round/bias_screen2
F2=$S1/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance; FMAN=$S1/foot/stance6/fetch/manifest.csv
PY=$S2/venv/bin/python; RUNLOG=$S1/foot/stance4/live/run_log.txt
export CW S1 S2 LB BS BS2 F2 FMAN PY RUNLOG
LOG=$CW/CMD_LOG.txt
mkdir -p "$CW/out"
n=$(ls "$CW/out" | wc -l); n=$(printf '%04d' $((n+1))); OUT=$CW/out/$n.txt
desc="$1"; shift; cmd="$*"
san() { sed -e "s#$S1#S1#g" -e "s#$S2#S2#g" -e "s#$TMPROOT#<tmproot>#g" -e 's#[Cc][Ll][Aa][Uu][Dd][Ee]#<cli>#g'; }
t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
bash -c "$cmd" > "$OUT.raw" 2>&1
rc=$?
t1=$(date -u +%Y-%m-%dT%H:%M:%SZ)
san < "$OUT.raw" > "$OUT"; rm -f "$OUT.raw"
sz=$(wc -c < "$OUT"); sh=$(sha256sum "$OUT" | cut -c1-16)
{
  echo "=== [$n] $t0 -> $t1  rc=$rc  ($desc)"
  echo "CMD: $cmd" | san
  echo "OUT: out/$n.txt  bytes=$sz  sha256[:16]=$sh"
  head -n 15 "$OUT" | sed 's/^/  | /'
  echo
} >> "$LOG"
cat "$OUT"
echo "[lg $n rc=$rc t0=$t0 t1=$t1 bytes=$sz]"
