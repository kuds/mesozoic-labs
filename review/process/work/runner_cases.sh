#!/usr/bin/env bash
# Process review of BS2: the real run_waves_bs.sh in its TEST MODE (BS_WAVES_TEST=1, BS_ROOT = the review's copy root,
# never BS2 itself), with RUN_CMD = stub_preflight.py (the copy's run_bs.preflight only; never run()), CHECK / READ =
# `false`.  Writes only under the copy root.  Nothing trains.
set -uo pipefail
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
BS2=$S2/r10round/bias_screen2
W=$BS2/review/process/work
CR=$W/copyroot
PY=$S2/venv/bin/python
RUNNER=$BS2/tools/run_waves_bs.sh
export BS_WAVES_TEST=1 BS_ROOT=$CR BS_RUN_CMD="nice -n 10 $PY -I -B $W/stub_preflight.py" BS_CHECK_CMD=false \
       BS_READ_CMD=false BS_ORPHAN_PATTERN="$W/[s]tub_preflight[.]py --arm" BS_POLL_SECONDS=1 BS_SLOTS=1

approve() {  # a FAKE APPROVED pinning the copy's files (copy root only)
  "$PY" -I -B - "$CR" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
r = Path(sys.argv[1]); h = lambda p: hashlib.sha256((r / p).read_bytes()).hexdigest()
t = ("run_bs.py", "read_bs.py", "run_waves_bs.sh", "bs_common.py", "bs_hooks.py", "score_bs.py")
(r / "APPROVED").write_text(json.dumps({"criteria_sha256": h("criteria.txt"), "screen_sha256": h("SCREEN.json"),
    "code_tree_digest": "331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1",
    "tool_sha256": {x: h("tools/" + x) for x in t}, "approved_by": "FAKE: review copy root only"}) + "\n")
PYEOF
}
clean() { rm -rf "$CR/logs" "$CR/runs" "$CR/APPROVED"; }
show() {
  echo "-- exit $1; copy root now: $(cd "$CR" && ls -A | tr '\n' ' ')"
  [[ -d "$CR/runs" ]] && echo "-- runs/: $(ls -A "$CR/runs" | tr '\n' ' ')"
  for f in run_waves_bs.log decided.tsv HALTED run_waves_bs.exit; do
    [[ -f "$CR/logs/$f" ]] && { echo "-- logs/$f:"; sed 's/^/     /' "$CR/logs/$f"; }
  done
  for f in "$CR"/logs/*.attempt*.log; do [[ -f "$f" ]] && { echo "-- ${f#"$CR"/}:"; sed 's/^/     /' "$f"; }; done
}

cd /tmp || exit 1
clean
echo "== $(date -u +%FT%TZ) CASE A: SCREEN.json and criteria.txt, no APPROVED (run, not --plan)"
bash "$RUNNER"; rc=$?; show $rc
echo "== $(date -u +%FT%TZ) CASE A2: --plan with no APPROVED"
bash "$RUNNER" --plan | head -3; echo "-- exit ${PIPESTATUS[0]}"
clean
echo "== $(date -u +%FT%TZ) CASE B: fake APPROVED pinning the copy, then one byte of criteria.txt changed"
approve
cp -p "$CR/criteria.txt" "$W/criteria.pristine"
"$PY" -I -B -c "import sys; p=sys.argv[1]; b=open(p,'rb').read(); open(p,'wb').write(b.replace(b'CRITERIA v1', b'CRITERIA v2', 1))" "$CR/criteria.txt"
bash "$RUNNER"; rc=$?; show $rc
echo "== $(date -u +%FT%TZ) CASE B2: relaunch while logs/HALTED exists"
bash "$RUNNER"; rc=$?; echo "-- exit $rc"
cp -p "$W/criteria.pristine" "$CR/criteria.txt" && rm -f "$W/criteria.pristine"
clean
echo "== $(date -u +%FT%TZ) CASE C: fake APPROVED, nothing changed: the stub preflight passes (exit 0); CHECK = false"
approve
bash "$RUNNER"; rc=$?; show $rc
clean
echo "== $(date -u +%FT%TZ) done; copy root: $(cd "$CR" && ls -A | tr '\n' ' ')"
(cd "$CR" && sha256sum -c ../copyroot_sha256.txt)
