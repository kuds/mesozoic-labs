"""List the RUNLOG lines the scorer prints as 'names a PC RUN_ID and is not a STANCE7 note of a round RUN_ID', by the
pinned rule itself (ST/tools/pc/r10round_common.parse_runlog with C.RUN_IDS, as score_pc.py line 312 calls it).
Reads RUNLOG only (read only); prints the RUNLOG sha256 and line count, the parsed notes by kind, and every other line's
number, sha256, RUN_ID tokens and leading time (not its text).  Usage: PY -I -B runlog_others.py [RUNLOG]"""
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

T = Path("/home/user/work/pc_study/tools/pc")
spec = importlib.util.spec_from_file_location("r10round_common", T / "r10round_common.py")
C = importlib.util.module_from_spec(spec)
sys.modules["r10round_common"] = C
spec.loader.exec_module(C)
runlog = sys.argv[1] if len(sys.argv) > 1 else C.RUNLOG
b = Path(runlog).read_bytes()
shown = str(runlog).replace(str(Path(C.RUNLOG).parents[3]), "S1")
print(f"RUNLOG {shown} sha256 {hashlib.sha256(b).hexdigest()} lines {len(b.decode('utf-8', 'replace').splitlines())}")
print(f"tool r10round_common.py sha256 {hashlib.sha256((T / 'r10round_common.py').read_bytes()).hexdigest()}")
notes, others = C.parse_runlog(runlog, C.RUN_IDS)
print(f"notes of a round RUN_ID: {len(notes)}; by kind {dict(Counter(n['kind'] for n in notes))}")
for n in notes:
    print(f"  NOTE line {n['line']} {n['kind']} {n['run']} {n['time']:%Y-%m-%d %H:%M}")
print(f"other lines naming a PC RUN_ID: {len(others)}: {[i for i, _ in others]}")
for i, line in others:
    toks = sorted(set(C.RUN_TOKEN_RE.findall(line)))
    print(f"OTHER line {i} sha256 {hashlib.sha256(line.encode()).hexdigest()} tokens {toks} starts {line[:23]!r}")
