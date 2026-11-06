"""Tolerance lane: re-pin walking pilot 4's SHA256SUMS.txt after maintainer decision (35) (pilot 3's format: '#'
header lines, then '<sha256>  <path>' lines, paths relative to the tools folder, grouped under '#' comments; this
lane's files, outside the tools folder, as ../prelaunch/tol/...). Reads files only; writes TOOLS/SHA256SUMS.txt
through a temporary file in the lane and a rename, refusing unless the list it replaces is byte-equal to the copy kept
in the lane (LANE/SHA256SUMS_before.txt), which this list pins in turn. The FIX stage's evidence under out/ (every
out/ entry of that list) is listed again as it is, and each such file must still match its FIX-stage sha256.

usage: python -I -B make_sums_tol.py TOOLS LANE HEADER.txt [DRY_RUN_OUT]   (DRY_RUN_OUT: write the list there instead)
"""

import hashlib
import os
import sys
from pathlib import Path

TOOLS, LANE = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
HEADER = Path(sys.argv[3]).read_text().rstrip("\n").splitlines()
DEST = TOOLS / "SHA256SUMS.txt"
BEFORE = LANE / "SHA256SUMS_before.txt"
REL = os.path.relpath(LANE, TOOLS)  # ../prelaunch/tol
if DEST.read_bytes() != BEFORE.read_bytes():
    raise SystemExit(f"{DEST} is not the list kept in {BEFORE}: not replaced")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tools_files(pattern: str) -> list[str]:
    return sorted(str(p.relative_to(TOOLS)) for p in TOOLS.glob(pattern) if p.is_file())


def lane_files(pattern: str) -> list[str]:
    return sorted(f"{REL}/{p.relative_to(LANE)}" for p in LANE.glob(pattern) if p.is_file())


before = []
for ln in BEFORE.read_text().splitlines():
    if ln and not ln.startswith("#"):
        h, p = ln.split("  ", 1)
        before.append((h, p))
fix_out = [p for h, p in before if p.startswith("out/")]
moved = [p for h, p in before if p.startswith("out/") and sha(TOOLS / p) != h]
if moved:
    raise SystemExit(f"FIX-stage evidence changed since its pin: {moved[:10]}")

CHANGED = "pilot4_straight.py, pilot4_score.py, test_pilot4_score.py and tests/test_pilot4_straight.py"
GROUPS = [
    ("Judging tools, readouts and the smoke runner (pilot3_vg.py and iso.py are byte copies of pilot 3's pinned "
     "a58d5f4f... and 5da55763...; the tolerance stage changed pilot4_straight.py and pilot4_score.py, and "
     "test_pilot4_score.py's default record folder):",
     ["pilot4_judge.py", "pilot4_straight.py", "pilot4_score.py", "test_pilot4_score.py", "pilot4_habit.py",
      "pilot4_noise.py", "pilot4_video.py", "pilot3_vg.py", "pilot4_mirror.py", "pilot4_mech.py", "nb_pilot_runner4.py",
      "iso.py"]),
    ("Unchanged from pilot 1 (byte copies of pilot 3's pinned files):",
     ["gait_probe_pilot.py", "make_gait_cases.py", "fixtures/gait_cases.json"]),
    ("Tests (the tolerance stage changed tests/test_pilot4_straight.py; tests/_tree_facts.py is a byte copy of pilot "
     "3's pinned helper):",
     tools_files("tests/*.py")),
    ("Sources (byte copies of pilot 3's src/: the frozen gait probe, equal to tools/tree/docs/investigations/gait_2026_09/"
     "gait_probe.py, sha256 61651b16...; pilot 1's and pilot 2's materials; the notebook cells and stance6 criteria):",
     tools_files("src/**/*")),
    ("The tolerance stage's record (../prelaunch/tol): the FIX-stage list it replaces (which pins the FIX stage's "
     "evidence and, through out/fix/SHA256SUMS_before.txt, the TOOLS stage's), the diffs of the four edited files, "
     "their pre-edit bytes (orig/, each equal to its FIX-stage pin), the command logger, the edit and pin scripts:",
     [f"{REL}/SHA256SUMS_before.txt", f"{REL}/lib.sh"] + lane_files("diffs/*") + lane_files("orig/*")
     + lane_files("scripts/*.py") + lane_files("scripts/*.sh") + lane_files("scripts/*.txt")
     + lane_files("scripts/*.ini") + lane_files("scripts/*.json") + lane_files("scripts/*.diff") + lane_files("scripts/from_e2e/*")),
    ("The straightness tool's validation records written by the edited tool (../prelaunch/tol/val_final, read by the "
     "straightness and scorer tests; val: the same runs with their figures, traces, logs and statuses; the baseline "
     "records are v3's pre-launch ENTRY BASELINE):",
     lane_files("val_final/*") + lane_files("val/*")),
    ("Evidence of the final test run (../prelaunch/tol/tests_final; scripts/run_tests.sh ran it; summary.json counts "
     "it):",
     lane_files("tests_final/*")),
    ("Evidence of the end-to-end re-run (the fix lane's scripts, copied with this lane's paths): the set-up path's "
     "records, the stand-in runs and tampered copies, the judged cells, every tool's output and log, the summaries "
     "e2e/e2e_summary.json and e2e/e2e_fix_checks.json, this lane's logs:",
     lane_files("setup/*.json") + lane_files("setup/s*/edge_pairs.json") + lane_files("setup/s*/stage/**/*")
     + lane_files("setup/s*/nb/**/*") + lane_files("runs/**/*") + lane_files("tampered/**/*")
     + lane_files("e2e/**/*") + lane_files("data/**/*") + lane_files("logs/*")),
    ("The FIX stage's evidence, unchanged and listed as the FIX-stage list pinned it (out/fix; its validation records "
     "out/fix/val_final were written by the FIX-stage tool 93d5c558... and are no longer read by the tests):",
     fix_out),
]
# every file of the lane outside tmp/ and pin/ (and CMDLOG.txt, which keeps growing) must be in one group
_listed = {f for _, g in GROUPS for f in g}
_missing = sorted(f for f in lane_files("**/*") if f not in _listed
                  and not f.startswith((f"{REL}/tmp/", f"{REL}/pin/")) and f != f"{REL}/CMDLOG.txt")
if _missing:
    raise SystemExit(f"lane files in no group: {_missing[:20]}")
seen = set()
lines = [("# " + h) if not h.startswith("#") else h for h in HEADER]
n = 0
for title, group in GROUPS:
    body = []
    for f in group:
        if f in seen or f.endswith("SHA256SUMS.txt") or "/tmp/" in f or f.endswith("CMDLOG.txt"):
            continue
        seen.add(f)
        body.append(f"{sha(TOOLS / f)}  {f}")
    if body:
        lines.append("# " + title)
        lines.extend(body)
        n += len(body)
tmp = LANE / "tmp" / "SHA256SUMS.txt.new"
tmp.write_text("\n".join(lines) + "\n")
DRY = Path(sys.argv[4]) if len(sys.argv) > 4 else None
os.replace(tmp, DRY or DEST)
DEST = DRY or DEST
print(f"wrote {DEST}: {n} files ({len(fix_out)} of them the FIX stage's out/ evidence); changed tools: {CHANGED}")
