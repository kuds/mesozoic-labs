#!/bin/bash
# Logged command runner for the LB scorer adapter (writes only LB/criteria_work/dryrun/out and ADAPT_NOTES.txt).
# Usage: lg.sh "<description>" '<command>'
# Exports the short names (S1, S2, LB, CW, DR, BS, BS2, PY, RUNLOG, FMAN, F2C) so commands need no literal paths; runs
# the command with bash -c; captures stdout+stderr into out/NNNN.txt; writes the path prefixes back as their short
# names and the CLI's temp-folder name as <cli> (the rule: no AI product name in any file); appends to
# ../ADAPT_NOTES.txt the start/end times (date -u), the command, the exit code, the output file with its byte count and
# sha256 prefix and its first lines; then prints the output.
DR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CW=$(dirname "$DR")
S2=${CW%/r10round/long_blind/criteria_work}
S1="$(dirname "$(dirname "$S2")")/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
TMPROOT=$(dirname "$(dirname "$(dirname "$S2")")")
LB=$S2/r10round/long_blind; BS=$S2/r10round/bias_screen; BS2=$S2/r10round/bias_screen2
F2C=$S1/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies; FMAN=$S1/foot/stance6/fetch/manifest.csv
PY=$S2/venv/bin/python; RUNLOG=$S1/foot/stance4/live/run_log.txt
export DR CW S1 S2 LB BS BS2 F2C FMAN PY RUNLOG
LOG=$CW/ADAPT_NOTES.txt
NL=${LG_LINES:-12}
n=$(ls "$DR/out" | grep -c '^[0-9]*\.txt$'); n=$(printf '%04d' $((n+1))); OUT=$DR/out/$n.txt
desc="$1"; shift; cmd="$*"
san() { sed -e "s#$S1#S1#g" -e "s#$S2#S2#g" -e "s#$TMPROOT#<tmproot>#g" -e 's#[Cc][Ll][Aa][Uu][Dd][Ee]#<cli>#g'; }
t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
bash -c "$cmd" > "$OUT.raw" 2>&1
rc=$?
t1=$(date -u +%Y-%m-%dT%H:%M:%SZ)
san < "$OUT.raw" > "$OUT"; rm -f "$OUT.raw"
sz=$(wc -c < "$OUT"); sh=$(sha256sum "$OUT" | cut -c1-16)
{
  echo "=== [C$n] $t0 -> $t1  rc=$rc  ($desc)"
  echo "CMD: $cmd" | san
  echo "OUT: dryrun/out/$n.txt  bytes=$sz  sha256[:16]=$sh"
  head -n "$NL" "$OUT" | cut -c1-200 | sed 's/^/  | /'
  echo
} >> "$LOG"
cat "$OUT"
echo "[lg C$n rc=$rc t0=$t0 t1=$t1 bytes=$sz]"
