"""Check every correction's OLD (or AFTER) lines against draft v3's lines of those numbers. Read only.
Usage: python -I check_old.py TEXT_CORRECTIONS V3"""
import re
import sys
from pathlib import Path

tc, v3 = (Path(p) for p in sys.argv[1:3])
v3_lines = v3.read_text().splitlines()
cur, mode, bad, n = None, None, [], 0
for ln in tc.read_text().splitlines():
    m = re.match(r"^(C\d+[ab]?)\s+v3 line", ln)
    if m:
        cur, mode = m.group(1), None
        continue
    if ln.strip() in ("OLD:", "AFTER:"):
        mode = "old"
        continue
    if ln.strip() in ("NEW:", "INSERT:") or ln.startswith("===="):
        mode = None
        continue
    if mode == "old":
        mm = re.match(r"^\s*(\d+) \|(.*)$", ln)
        if mm:
            no, text = int(mm.group(1)), mm.group(2)
            text = text[1:] if text.startswith(" ") else text
            n += 1
            if v3_lines[no - 1] != text:
                bad.append((cur, no, text, v3_lines[no - 1]))
print(f"OLD lines checked: {n}; mismatches: {len(bad)}")
for b in bad:
    print(b)
