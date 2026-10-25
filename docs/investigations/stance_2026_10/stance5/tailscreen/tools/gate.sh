#!/usr/bin/env bash
# Tail_3 bound screen: the pre-training gate (criteria RUN HANDLING; BUILD_SPEC section 8).  Runs ONLY after
# SCREEN/APPROVED exists; it takes optimizer steps on a small TEST model (T1b, T2, T7b), never on a screen policy.
#
#   gate.sh            the gate
#
# 1. Refuses without SCREEN/APPROVED, if tests_out/gate_pass.json already exists, if sha256(criteria.txt) or
#    sha256(tools/screen_common.py) differ from APPROVED, if coef.json breaks the rule or differs from APPROVED's pins,
#    if the scratch clone is not exactly the approved commit (clean with ignored files counted, branch, linear on
#    cf80ecb1, push URL disabled, the two changed files), if the stack differs, or if a pinned tool's sha256 differs.
# 2. Runs the whole clone test file but test_no_stage_config_enables_symmetry (deselected: it fails on cf80ecb1 itself)
#    with MESOZOIC_TAILSCREEN_GATE=1 (gate tier + build tier + the file's existing tests), then the tools tests again, every run under `python -I -B` with -p no:cacheprovider, writing
#    tests_out/gate_tier.txt and JUnit XML per run.
# 3. Writes tests_out/gate_pass.json ({"head", "passed", "failed": 0, "skipped", "tests": {node id: status},
#    "required", "utc"}) ONLY if both runs exited 0, screen_common.gate_problems() is empty (every GATE_REQUIRED
#    test present and "passed": a skipped, deselected or absent gate-tier test is NOT a pass), and the clone is still
#    clean (status --porcelain --ignored empty) afterwards.
# 4. Prints the sha256 of criteria.txt, every pinned tool, screen_common.py and coef.json for RUNLOG.
# Exit codes: 0 gate passed (gate_pass.json written); 1 tests failed or a required test did not pass; 2 refused.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LC_ALL=C

S2=scratchpad2
SCREEN=$S2/tailround/screen
PY=$S2/venv/bin/python
TOOLS=$SCREEN/tools
OUT=$SCREEN/tests_out
CLONE=$SCREEN/scratch_tail3bound

die() { echo "gate: refused: $*" >&2; exit 2; }

[[ $# -eq 0 ]] || die "gate.sh takes no arguments"
[[ -f "$SCREEN/APPROVED" ]] || die "no SCREEN/APPROVED (the gate runs only after the maintainer's approval)"
[[ ! -e "$OUT/gate_pass.json" ]] || die "$OUT/gate_pass.json exists (written once per gate; move it aside to re-gate)"
mkdir -p "$OUT"

# ---------------------------------------------------------------------------------------- 1. pre-gate checks (no torch)
"$PY" -I -B - "$SCREEN" <<'PYEOF' || die "pre-gate checks failed (see above)"
import importlib.util, json, sys
from pathlib import Path

root = Path(sys.argv[1])
tools = root / "tools"
spec = importlib.util.spec_from_file_location("run_tailscreen", tools / "run_tailscreen.py")
RT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RT)
C = RT.C
problems = []
approved, err = RT._read_json(root / "APPROVED")
if err or not isinstance(approved, dict):
    print(json.dumps({"gate": "refused", "problems": [f"APPROVED: {err or 'not a JSON object'}"]}, indent=1))
    sys.exit(1)
crit = root / "criteria.txt"
if not crit.is_file() or RT.sha256_file(crit) != approved.get("criteria_sha256"):
    problems.append("criteria.txt missing or its sha256 != APPROVED.criteria_sha256")
if RT.sha256_file(tools / "screen_common.py") != approved.get("screen_common_sha256"):
    problems.append("tools/screen_common.py sha256 != APPROVED.screen_common_sha256")
coef_path = root / "calib" / "coef.json"
coef, err = RT._read_json(coef_path)
if err:
    problems.append(f"coef.json: {err}")
else:
    problems += [f"coef.json: {p}" for p in RT.coef_rule_problems(coef)]
    for key in ("COEF_90", "COEF_60"):
        if coef.get(key) != approved.get(key):
            problems.append(f"coef.json {key} {coef.get(key)!r} != APPROVED {approved.get(key)!r}")
    if RT.sha256_file(coef_path) != approved.get("coef_json_sha256"):
        problems.append("coef.json sha256 != APPROVED.coef_json_sha256")
if (root / "runs").exists() and any((root / "runs").iterdir()):
    problems.append("runs/ is not empty: the gate comes before the first training step")
problems += RT.clone_problems(RT.clone_state(C.CLONE), approved)
problems += [f"stack: {p}" for p in C.check_stack()]
problems += RT.tool_pin_problems()
print(json.dumps({"gate": "pre-checks " + ("pass" if not problems else "refused"), "problems": problems}, indent=1))
sys.exit(1 if problems else 0)
PYEOF

HEAD=$(git --no-optional-locks -C "$CLONE" rev-parse HEAD)
LOG=$OUT/gate_tier.txt
{
  echo "# gate tier, $(date -u +%Y-%m-%dT%H:%M:%SZ), clone HEAD $HEAD, MESOZOIC_TAILSCREEN_GATE=1"
  echo "## clone: the whole test_mirror_kl_ppo.py"
} > "$LOG"

# ---------------------------------------------------------------------------------------- 2. the tests
set +e
# One test of the file is deselected (the maintainer's decision of 2026-10-24, criteria APPENDIX G):
# test_no_stage_config_enables_symmetry asserts main's rule that no shipped stage sets [ppo] symmetry*, and it fails on
# cf80ecb1 itself, whose T. rex stance.toml sets symmetry = "mirror_kl" for the band arm.  Every other test runs.
(cd "$SCREEN" && MESOZOIC_TAILSCREEN_GATE=1 "$PY" -I -B -m pytest -p no:cacheprovider --rootdir "$CLONE" \
  "$CLONE/environments/shared/tests/test_mirror_kl_ppo.py" \
  --deselect "environments/shared/tests/test_mirror_kl_ppo.py::test_no_stage_config_enables_symmetry" \
  -rA -q --junitxml "$OUT/gate_junit_clone.xml") >> "$LOG" 2>&1
RC_CLONE=$?
echo "exit $RC_CLONE" >> "$LOG"
echo "## tools tests" >> "$LOG"
(cd "$SCREEN" && MESOZOIC_TAILSCREEN_GATE=1 "$PY" -I -B -m pytest -p no:cacheprovider "$TOOLS/tests" -rA -q \
  --junitxml "$OUT/gate_junit_tools.xml") >> "$LOG" 2>&1
RC_TOOLS=$?
echo "exit $RC_TOOLS" >> "$LOG"
set -e

# ---------------------------------------------------------------------------------------- 3. gate_pass.json
"$PY" -I -B - "$SCREEN" "$HEAD" "$RC_CLONE" "$RC_TOOLS" <<'PYEOF'
import datetime, importlib.util, json, subprocess, sys
import xml.etree.ElementTree as ET
from pathlib import Path

root, head, rc_clone, rc_tools = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
spec = importlib.util.spec_from_file_location("screen_common", root / "tools" / "screen_common.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
tests = {}
for xml in ("gate_junit_clone.xml", "gate_junit_tools.xml"):
    path = root / "tests_out" / xml
    if not path.is_file():
        continue
    for case in ET.parse(path).getroot().iter("testcase"):
        node = f"{case.get('classname')}::{case.get('name')}"
        tags = {child.tag for child in case}
        status = ("error" if "error" in tags else "failed" if "failure" in tags else
                  "skipped" if "skipped" in tags else "passed")
        tests[node] = status
counts = {k: sum(v == k for v in tests.values()) for k in ("passed", "failed", "error", "skipped")}
record = {"head": head, "passed": counts["passed"], "failed": counts["failed"] + counts["error"],
          "skipped": counts["skipped"], "tests": tests, "required": list(C.GATE_REQUIRED),
          "pytest_exit": {"clone": rc_clone, "tools": rc_tools},
          "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
problems = C.gate_problems(record, head)
if rc_clone != 0 or rc_tools != 0:
    problems.append(f"pytest exit codes clone {rc_clone}, tools {rc_tools} (both must be 0)")
status = subprocess.run(["git", "--no-optional-locks", "-C", C.CLONE, "status", "--porcelain", "--ignored"],
                        capture_output=True, text=True).stdout
if status.strip():
    problems.append(f"the scratch clone is not clean after the tests (ignored files counted): {status[:300]!r}")
if problems:
    print(json.dumps({"gate": "FAILED", "problems": problems, "counts": counts}, indent=1))
    sys.exit(1)
with open(root / "tests_out" / "gate_pass.json", "x") as fh:
    json.dump(record, fh, indent=1)
print(json.dumps({"gate": "passed", "head": head, "counts": counts}, indent=1))
PYEOF

# ---------------------------------------------------------------------------------------- 4. sha256 for RUNLOG
echo "sha256 for RUNLOG:"
(cd "$SCREEN" && sha256sum criteria.txt APPROVED calib/coef.json tools/screen_common.py \
  tools/run_tailscreen.py tools/calib_tailscreen.py tools/measure_tailscreen.py tools/score_tailscreen.py \
  tools/copy_durable.sh tools/gate.sh tools/run_waves.sh tests_out/gate_pass.json)
