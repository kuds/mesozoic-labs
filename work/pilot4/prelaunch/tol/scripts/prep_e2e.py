"""Tolerance lane: copy the fix lane's end-to-end and test scripts (pilot4/tools/out/fix/scripts, read only) into this
lane's scripts/ with this lane's paths. Text only; every replacement is counted and must occur as often as listed.

  * the scratchpad prefix spelled out in a shell variable becomes a glob resolved at run time; in a Python helper,
    Path(__file__).resolve().parents[5] (the work folder, from scripts/from_e2e/);
  * LANE=$T/out/fix becomes LANE=$SP/work/pilot4/prelaunch/tol; the validation records become LANE/val_final;
  * the pilot branch's git ref is read from the shared tree at run time ($BR; make_params.py takes it as argv[2]);
  * e2e_summary.py's TOOLS is the pilot-4 tools folder (LANE.parents[1] / "tools").
Writes LANE/scripts/<name> (refusing to overwrite) and LANE/scripts/prep_e2e.json (the counts).

usage: python -I -B prep_e2e.py FIX_SCRIPTS LANE
"""

import json
import re
import sys
from pathlib import Path

SRC, LANE = Path(sys.argv[1]), Path(sys.argv[2])
DST = LANE / "scripts"
PREFIX = re.compile(r"/tmp/[^/]+/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad")
BRANCH = re.compile(r"\b[a-z]+/pilot-walk-r10-p4\b")
GLOB_SP = "$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad)"
BR_LINE = "BR=$(git -C \"$T/tree\" for-each-ref --format='%(refname:short)' 'refs/heads/*/pilot-walk-r10-p4')"


def lit(old: str, new: str, n: int = 1):
    return ("lit", old, new, n)


def rx(pattern, new: str, n: int):
    return ("rx", pattern, new, n)


SH_COMMON = [rx(re.compile("^SP=" + PREFIX.pattern + "$", re.M), "SP=" + GLOB_SP, 1),
             lit("LANE=$T/out/fix\n", "LANE=$SP/work/pilot4/prelaunch/tol\n", 1)]
PLAN = {
    "phase_0_setup.sh": SH_COMMON + [
        lit("# Fix lane, phase 0", "# Tolerance lane (copied from the fix lane), phase 0"),
        lit("TREE=$T/tree\n", "TREE=$T/tree\n" + BR_LINE + "\n"),
        rx(BRANCH, "$BR", 1),
        lit("$S/from_e2e/make_params.py $LANE >", "$S/from_e2e/make_params.py $LANE $BR >")],
    "phase_a_judge.sh": SH_COMMON + [
        lit("# e2e lane, phase A", "# Tolerance lane (copied from the fix lane, from the e2e lane), phase A"),
        lit("TREE=$T/tree\n", "TREE=$T/tree\n" + BR_LINE + "\n"),
        rx(BRANCH, "$BR", 2)],
    "phase_b_readouts.sh": SH_COMMON + [
        lit("# e2e lane, phase B", "# Tolerance lane (copied from the fix lane, from the e2e lane), phase B")],
    "phase_c_score.sh": SH_COMMON + [
        lit("# e2e lane, phase C", "# Tolerance lane (copied from the fix lane, from the e2e lane), phase C")],
    "phase_d_fix.sh": SH_COMMON + [
        lit("# Fix lane, phase D", "# Tolerance lane (copied from the fix lane), phase D")],
    "phase_ff.sh": SH_COMMON + [
        lit("# Fix lane: the final-model", "# Tolerance lane (copied from the fix lane): the final-model")],
    "run_tests.sh": SH_COMMON + [
        lit("# fix lane (copied from the e2e lane)", "# tolerance lane (copied from the fix lane, from the e2e lane)"),
        lit("STRAIGHT_VAL=$T/out/fix/val_final", "STRAIGHT_VAL=$LANE/val_final")],
    "final_state.sh": SH_COMMON + [
        lit("# fix lane (copied from the e2e lane)", "# tolerance lane (copied from the fix lane, from the e2e lane)"),
        lit("files under the other pilot-4 lanes' folders newer than the start of the test run: "
            "$(find \"$T/out\" -path \"$LANE\" -prune -o -newer",
            "files under the pinned tools' evidence (tools/out) newer than the start of the test run: "
            "$(find \"$T/out\" -newer")],
    "e2e_summary.py": [
        lit('TOOLS = LANE.parents[1]\nSTRAIGHT_VAL = TOOLS / "out" / "fix" / "val_final"\n',
            'TOOLS = LANE.parents[1] / "tools"  # tolerance lane: LANE is pilot4/prelaunch/tol\n'
            'STRAIGHT_VAL = LANE / "val_final"\n')],
    "e2e_fix_checks.py": [],
    "make_fix_inputs.py": [],
    "make_tampered_fix.py": [],
    "cmp_old_new_records.py": [],
    "run_pytest.py": [],
    "pytest.ini": [lit("# fix lane: an empty", "# tolerance lane (copied from the fix lane): an empty")],
    "junit_summary.py": [],
    "from_e2e/edge_pairs.py": [rx(re.compile(r'Path\("' + PREFIX.pattern + '/work/'),
                                  'Path(__file__).resolve().parents[5] / ("', 2)],
    "from_e2e/sidecar_facts.py": [rx(re.compile(r'Path\("' + PREFIX.pattern + '/work/'),
                                     'Path(__file__).resolve().parents[5] / ("', 1)],
    "from_e2e/weights_facts.py": [rx(re.compile(r'Path\("' + PREFIX.pattern + '/work/'),
                                     'Path(__file__).resolve().parents[5] / ("', 1)],
    "from_e2e/make_params.py": [
        lit("lane = Path(sys.argv[1])\n", "lane = Path(sys.argv[1])\nrepo_ref = sys.argv[2]  # the pilot branch, read "
            "from the shared tree by the caller (tolerance lane)\n"),
        rx(re.compile(r'"repo_ref": "' + BRANCH.pattern + '"'), '"repo_ref": repo_ref', 1)],
    "from_e2e/make_runs.py": [],
    "from_e2e/make_final.py": [],
    "from_e2e/cmp_records.py": [],
    "from_e2e/dedupe_models.py": [],
}
counts = {}
(DST / "from_e2e").mkdir(parents=True, exist_ok=True)
for name, edits in PLAN.items():
    src = SRC / name
    text = src.read_text()
    c = []
    for kind, old, new, n in edits:
        if kind == "lit":
            k = text.count(old)
            if k != n:
                raise SystemExit(f"{name}: {old[:60]!r} occurs {k} times, not {n}: nothing more written")
            text = text.replace(old, new)
        else:
            text, k = old.subn(new, text)
            if k != n:
                raise SystemExit(f"{name}: /{old.pattern[:60]}/ matched {k} times, not {n}: nothing more written")
        c.append(k)
    left = [m.group(0) for m in PREFIX.finditer(text)]
    dst = DST / name
    if dst.exists():
        raise SystemExit(f"{dst} exists: not overwritten")
    dst.write_text(text)
    if src.stat().st_mode & 0o111:
        dst.chmod(0o755)
    counts[name] = {"replacements": c, "prefix_spelled_out_left": len(left), "branch_spelled_out_left":
                    len(BRANCH.findall(text))}
(DST / "prep_e2e.json").write_text(json.dumps(counts, indent=1) + "\n")
print(json.dumps(counts, indent=1))
