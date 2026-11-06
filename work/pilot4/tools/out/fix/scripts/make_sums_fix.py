"""Fix lane: re-pin walking pilot 4's SHA256SUMS.txt (pilot 3's format: '#' header lines, then '<sha256>  <path>'
lines, paths relative to the tools folder, grouped under '#' comments). Reads files only; writes TOOLS/SHA256SUMS.txt
through a temporary file in the lane and a rename, refusing unless the list it replaces is byte-equal to the copy kept
in the lane (out/fix/SHA256SUMS_before.txt), which this list pins in turn (so the TOOLS stage's evidence stays
verifiable through it).

usage: python -I -B make_sums_fix.py TOOLS HEADER.txt
"""

import hashlib
import os
import sys
from pathlib import Path

TOOLS = Path(sys.argv[1])
HEADER = Path(sys.argv[2]).read_text().rstrip("\n").splitlines()
DEST = TOOLS / "SHA256SUMS.txt"
FIX = "out/fix"
BEFORE = TOOLS / FIX / "SHA256SUMS_before.txt"
if DEST.read_bytes() != BEFORE.read_bytes():
    raise SystemExit(f"{DEST} is not the list kept in {BEFORE}: not replaced")


def files(pattern: str) -> list[str]:
    return sorted(str(p.relative_to(TOOLS)) for p in TOOLS.glob(pattern) if p.is_file())


GROUPS = [
    ("Judging tools, readouts and the smoke runner (pilot3_vg.py and iso.py are byte copies of pilot 3's pinned "
     "a58d5f4f... and 5da55763...; the fix stage changed pilot4_judge.py, pilot4_straight.py, pilot4_score.py, "
     "test_pilot4_score.py, nb_pilot_runner4.py and pilot4_mech.py):",
     ["pilot4_judge.py", "pilot4_straight.py", "pilot4_score.py", "test_pilot4_score.py", "pilot4_habit.py",
      "pilot4_noise.py", "pilot4_video.py", "pilot3_vg.py", "pilot4_mirror.py", "pilot4_mech.py", "nb_pilot_runner4.py",
      "iso.py"]),
    ("Unchanged from pilot 1 (byte copies of pilot 3's pinned files):",
     ["gait_probe_pilot.py", "make_gait_cases.py", "fixtures/gait_cases.json"]),
    ("Tests (the fix stage changed test_pilot4_judge.py, test_pilot4_straight.py and test_nb_pilot_runner4.py; "
     "tests/_tree_facts.py is a byte copy of pilot 3's pinned helper):",
     files("tests/*.py")),
    ("Sources (byte copies of pilot 3's src/: the frozen gait probe, equal to tools/tree/docs/investigations/gait_2026_09/"
     "gait_probe.py, sha256 61651b16...; pilot 1's and pilot 2's materials; the notebook cells and stance6 criteria):",
     files("src/**/*")),
    ("The fix stage's record: the TOOLS-stage list it replaces (which pins the TOOLS stage's evidence under out/e2e), "
     "the text corrections for v3, the command log's writer:",
     [f"{FIX}/SHA256SUMS_before.txt", f"{FIX}/text_corrections.txt", f"{FIX}/L.sh"] + files(f"{FIX}/diffs/*")),
    ("The straightness tool's validation records, re-run with the pinned tool (out/fix/val_final, read by the "
     "straightness and scorer tests; out/fix/val: the same runs with their figures, traces and logs):",
     files(f"{FIX}/val_final/*") + files(f"{FIX}/val/*")),
    ("Evidence of the final test run (out/fix/tests_final; scripts/run_tests.sh ran it; summary.json counts it):",
     files(f"{FIX}/tests_final/*")),
    ("Evidence of the end-to-end re-run (the e2e lane's scripts, copied, plus the fix stage's own): the set-up "
     "path's records, the stand-in runs and tampered copies, the judged cells, every tool's output and log, the "
     "summaries e2e/e2e_summary.json and e2e/e2e_fix_checks.json:",
     files(f"{FIX}/scripts/*") + files(f"{FIX}/scripts/from_e2e/*")
     + files(f"{FIX}/setup/*.json") + files(f"{FIX}/setup/s*/edge_pairs.json") + files(f"{FIX}/setup/s*/stage/*")
     + files(f"{FIX}/setup/s*/nb/*")
     + files(f"{FIX}/runs/*.json") + files(f"{FIX}/runs/**/*") + files(f"{FIX}/tampered/**/*")
     + files(f"{FIX}/e2e/**/*") + files(f"{FIX}/data/**/*") + files(f"{FIX}/logs/*")),
]
# every file of the lane outside tmp/ and pin/ (and CMDLOG.txt, which keeps growing) must be in one group
_listed = {f for _, g in GROUPS for f in g}
_missing = sorted(f for f in files(f"{FIX}/**/*") if f not in _listed and not f.startswith((f"{FIX}/tmp/", f"{FIX}/pin/"))
                  and f != f"{FIX}/CMDLOG.txt")
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
        body.append(f"{hashlib.sha256((TOOLS / f).read_bytes()).hexdigest()}  {f}")
    if body:
        lines.append("# " + title)
        lines.extend(body)
        n += len(body)
tmp = TOOLS / FIX / "tmp" / "SHA256SUMS.txt.new"
tmp.write_text("\n".join(lines) + "\n")
os.replace(tmp, DEST)
print(f"wrote {DEST}: {n} files")
