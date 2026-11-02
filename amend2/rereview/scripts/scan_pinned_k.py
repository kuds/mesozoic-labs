"""Re-review: every out folder under a pytest basetemp: whether its RUNLOG copy holds RUNLOG line 902's bytes, and the
pinned and deviation runs' THE ONE TEST k (None when not computed)."""
import hashlib
import json
import sys
from pathlib import Path

L902 = "0e30b26ed0a346dd0bdacd8bfd18f2ff9d5f6f58e492e1cc6e29d29763fbce12"
for pj in sorted(Path(sys.argv[1]).rglob("pinned.json")):
    out = pj.parent
    rl = out / "runlog_asread.txt"
    has = any(hashlib.sha256(x.encode()).hexdigest() == L902 for x in rl.read_text().splitlines()) if rl.exists() else None
    ks = []
    for n in ("pinned", "deviation"):
        try:
            ot = (json.loads((out / f"{n}.json").read_text()).get("reading") or {}).get("one_test") or {}
            ks.append(ot.get("k"))
        except (OSError, ValueError):
            ks.append("unreadable")
    print(f"{out.relative_to(sys.argv[1])}: line 902 in RUNLOG copy {has}; pinned k {ks[0]}; deviation k {ks[1]}")
