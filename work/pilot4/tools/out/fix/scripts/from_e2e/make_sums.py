"""Write walking pilot 4's SHA256SUMS.txt (pilot 3's format: '#' header lines, then '<sha256>  <path>' lines, paths
relative to the tools folder, grouped under '#' comments). Scratch helper of the e2e lane; reads files only, writes
only TOOLS/SHA256SUMS.txt (refused when it exists).

usage: python -I -B make_sums.py TOOLS HEADER.txt
"""

import hashlib
import sys
from pathlib import Path

TOOLS = Path(sys.argv[1])
HEADER = Path(sys.argv[2]).read_text().rstrip("\n").splitlines()
DEST = TOOLS / "SHA256SUMS.txt"
if DEST.exists():
    raise SystemExit(f"{DEST} exists: not overwritten")
E2E = "out/e2e"


def files(pattern: str) -> list[str]:
    return sorted(str(p.relative_to(TOOLS)) for p in TOOLS.glob(pattern) if p.is_file())


GROUPS = [
    ("Judging tools, readouts and the smoke runner (new for pilot 4, or adapted from pilot 3's pinned tools; pilot3_vg.py "
     "and iso.py are byte copies of pilot 3's pinned a58d5f4f... and 5da55763...):",
     ["pilot4_judge.py", "pilot4_straight.py", "pilot4_score.py", "test_pilot4_score.py", "pilot4_habit.py",
      "pilot4_noise.py", "pilot4_video.py", "pilot3_vg.py", "pilot4_mirror.py", "pilot4_mech.py", "nb_pilot_runner4.py",
      "iso.py"]),
    ("Unchanged from pilot 1 (byte copies of pilot 3's pinned files):",
     ["gait_probe_pilot.py", "make_gait_cases.py", "fixtures/gait_cases.json"]),
    ("Tests (tests/_tree_facts.py is a byte copy of pilot 3's pinned helper; no pilot-4 test imports it):",
     files("tests/*.py")),
    ("Sources (byte copies of pilot 3's src/: the frozen gait probe, equal to tools/tree/docs/investigations/gait_2026_09/"
     "gait_probe.py, sha256 61651b16...; pilot 1's and pilot 2's materials; the notebook cells and stance6 criteria):",
     files("src/**/*")),
    ("Evidence of the final test run (out/e2e/tests_final; scripts/run_tests.sh ran it; summary.json counts it):",
     [f"{E2E}/scripts/run_tests.sh", f"{E2E}/scripts/run_pytest.py", f"{E2E}/scripts/pytest.ini",
      f"{E2E}/scripts/junit_summary.py"] + files(f"{E2E}/tests_final/*")),
    ("Evidence of the end-to-end run: the scripts; the set-up path's records (the pairs it saved are pinned as the "
     "stand-in runs' byte-equal copies, setup/models_dedupe.json); the stand-in runs and the tampered copies; the judged "
     "cells; every tool's output and log; the summary e2e/e2e_summary.json:",
     files(f"{E2E}/scripts/*.py") + [f for f in files(f"{E2E}/scripts/*.sh") if not f.endswith("run_tests.sh")]
     + [f"{E2E}/L.sh"]
     + files(f"{E2E}/setup/*.json") + files(f"{E2E}/setup/s*/edge_pairs.json") + files(f"{E2E}/setup/s*/stage/*")
     + files(f"{E2E}/setup/s*/models/*") + files(f"{E2E}/setup/s*/nb/*") + files(f"{E2E}/setup/s*/run_level/*")
     + files(f"{E2E}/setup/zero_action_baselines/*") + files(f"{E2E}/setup/*.json")
     + files(f"{E2E}/runs/*.json") + files(f"{E2E}/runs/**/*")
     + files(f"{E2E}/tampered/**/*")
     + files(f"{E2E}/e2e/*") + files(f"{E2E}/e2e/OUT/**/*") + files(f"{E2E}/e2e/straight/*")
     + files(f"{E2E}/e2e/score/*") + files(f"{E2E}/e2e/habit/*") + files(f"{E2E}/e2e/noise/*")
     + files(f"{E2E}/e2e/readouts/*") + files(f"{E2E}/e2e/inputs/*") + files(f"{E2E}/e2e/logs/*")
     + files(f"{E2E}/e2e/videos/*")
     + files(f"{E2E}/logs/*")),
]
seen = set()
lines = [("# " + h) if not h.startswith("#") else h for h in HEADER]
n = 0
for title, group in GROUPS:
    body = []
    for f in group:
        if f in seen or f.endswith("SHA256SUMS.txt"):
            continue
        seen.add(f)
        body.append(f"{hashlib.sha256((TOOLS / f).read_bytes()).hexdigest()}  {f}")
    if body:
        lines.append("# " + title)
        lines.extend(body)
        n += len(body)
DEST.write_text("\n".join(lines) + "\n")
print(f"wrote {DEST}: {n} files")
