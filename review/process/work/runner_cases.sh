#!/usr/bin/env bash
# Process review of PS: the REAL PS/tools/run_waves_bs.sh in its TEST MODE (BS_WAVES_TEST=1, BS_ROOT = the review's
# copy root, never PS itself), RUN_CMD = stub_preflight.py (the copy's run_bs.preflight only; never run()), CHECK /
# READ = `false`, BS_SCORER = the copy's score_bs.py.  Writes only under the copy root.  Nothing trains.
set -uo pipefail
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
W=/home/user/work/phase_screen/review/process/work
CR=$W/copyroot
PY=$S2/venv/bin/python
RUNNER=/home/user/work/phase_screen/tools/run_waves_bs.sh
export BS_WAVES_TEST=1 BS_ROOT=$CR BS_RUN_CMD="nice -n 10 $PY -I -B $W/stub_preflight.py" BS_CHECK_CMD=false \
       BS_READ_CMD=false BS_ORPHAN_PATTERN="$W/[s]tub_preflight[.]py --arm" BS_POLL_SECONDS=1 BS_SLOTS=1 \
       BS_SCORER=$CR/tools/score_bs.py TMPDIR=$W/tmp
approve() {  # a FAKE APPROVED pinning the copy's current files (copy root only); $1 = "noscore" drops the scorer
  "$PY" -I -B - "$CR" "${1:-}" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
r = Path(sys.argv[1]); h = lambda p: hashlib.sha256((r / p).read_bytes()).hexdigest()
t = ["run_bs.py", "read_bs.py", "run_waves_bs.sh", "bs_common.py", "bs_hooks.py", "score_bs.py"]
if sys.argv[2] == "noscore":
    t.remove("score_bs.py")
(r / "APPROVED").write_text(json.dumps({"criteria_sha256": h("criteria.txt"), "screen_sha256": h("SCREEN.json"),
    "code_tree_digest": "331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1",
    "tool_sha256": {x: h("tools/" + x) for x in t}, "approved_by": "FAKE: review copy root only"},
    separators=(",", ":")) + "\n")
PYEOF
}
clean() { rm -rf "$CR/logs" "$CR/runs" "$CR/APPROVED"; }
show() {
  echo "-- exit $1; copy root now: $(cd "$CR" && ls -A | tr '\n' ' ')"
  [[ -d "$CR/runs" ]] && echo "-- runs/: $(ls -A "$CR/runs" | tr '\n' ' ')"
  for f in run_waves_bs.log decided.tsv HALTED run_waves_bs.exit; do
    [[ -f "$CR/logs/$f" ]] && { echo "-- logs/$f:"; sed 's/^/     /' "$CR/logs/$f" | cut -c1-400; }
  done
  for f in "$CR"/logs/*.attempt*.log; do [[ -f "$f" ]] && { echo "-- ${f#"$CR"/}:"; sed 's/^/     /' "$f" | cut -c1-600; }; done
}
flipb() { "$PY" -I -B -c "import sys; p=sys.argv[1]; b=open(p,'rb').read(); o=sys.argv[2].encode(); n=sys.argv[3].encode(); assert o in b and len(o)==len(n); open(p,'wb').write(b.replace(o, n, 1))" "$@"; }

cd "$W" || exit 1
clean
echo "== $(date -u +%FT%TZ) CASE A: SCREEN.json and criteria.txt, no APPROVED (run, not --plan)"
bash "$RUNNER"; rc=$?; show $rc
echo "== $(date -u +%FT%TZ) CASE A2: --plan with no APPROVED"
bash "$RUNNER" --plan; echo "-- exit $?"
clean
echo "== $(date -u +%FT%TZ) CASE L: fake APPROVED pinning the copy, nothing changed, TODAY's disk (no launch override)"
approve
bash "$RUNNER"; rc=$?; show $rc
clean
export BS_LAUNCH_DISK_BYTES=0
echo "== $(date -u +%FT%TZ) [BS_LAUNCH_DISK_BYTES=0 from here on, test mode only]"
echo "== $(date -u +%FT%TZ) CASE C: fake APPROVED WITHOUT score_bs.py"
approve noscore
bash "$RUNNER"; rc=$?; show $rc
clean
echo "== $(date -u +%FT%TZ) CASE D: fake APPROVED pinning the copy, then one byte of the copy's score_bs.py changed"
approve
cp -p "$CR/tools/score_bs.py" "$W/score_bs.pristine"
flipb "$CR/tools/score_bs.py" "the criteria's" "the Criteria's"
bash "$RUNNER"; rc=$?; show $rc
cp -p "$W/score_bs.pristine" "$CR/tools/score_bs.py" && rm -f "$W/score_bs.pristine"
clean
echo "== $(date -u +%FT%TZ) CASE B: fake APPROVED pinning the copy, then one byte of criteria.txt changed"
approve
cp -p "$CR/criteria.txt" "$W/criteria.pristine"
flipb "$CR/criteria.txt" "CRITERIA v1 AS INSTALLED" "CRITERIA v2 AS INSTALLED"
bash "$RUNNER"; rc=$?; show $rc
echo "== $(date -u +%FT%TZ) CASE B2: relaunch while logs/HALTED exists"
bash "$RUNNER"; rc=$?; echo "-- exit $rc"
cp -p "$W/criteria.pristine" "$CR/criteria.txt" && rm -f "$W/criteria.pristine"
clean
echo "== $(date -u +%FT%TZ) CASE E: fake APPROVED, nothing changed: the stub preflight passes (exit 0); CHECK = false"
approve
bash "$RUNNER"; rc=$?; show $rc
clean
echo "== $(date -u +%FT%TZ) done; copy root: $(cd "$CR" && ls -A | tr '\n' ' ')"
(cd "$CR" && sha256sum -c ../copyroot_sha256.txt)
