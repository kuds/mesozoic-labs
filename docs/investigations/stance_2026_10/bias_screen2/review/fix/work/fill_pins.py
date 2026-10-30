"""The fixer's PINS re-fill of <BS2>/criteria.txt: each pinned file's line gets the file's current sha256, and the note
under the list says who filled it and when.  Prints old -> new per file.  Run: PY -I -B fill_pins.py <BS2> <HH:MM>"""
import hashlib
import re
import sys
from pathlib import Path

BS2 = Path(sys.argv[1])
WHEN = sys.argv[2]
FILES = ("SCREEN.json", "tools/score_bs.py", "tools/tests/test_score_bs.py", "tools/bs_common.py", "tools/bs_hooks.py",
         "tools/run_bs.py", "tools/run_waves_bs.sh", "tools/read_bs.py")
crit = BS2 / "criteria.txt"
text = crit.read_text(encoding="utf-8")
head, rest = text.split("\nPINS", 1)
pins, tail = rest.split("\nHOW THE COORDINATOR PINS", 1)
for rel in FILES:
    sha = hashlib.sha256((BS2 / rel).read_bytes()).hexdigest()
    pat = re.compile(rf"^(\s+{re.escape(rel)}\s+)([0-9a-f]{{64}})$", re.M)
    found = pat.findall(pins)
    assert len(found) == 1, rel
    print(f"{rel:<32} {found[0][1][:16]} -> {sha[:16]}{'  (unchanged)' if found[0][1] == sha else ''}")
    pins = pat.sub(lambda m, sha=sha: m.group(1) + sha, pins)
OLD_NOTE = """  (filled by the scorer adapter at 11:32 UTC from the files as installed; the five builder tools as the builder froze
  them, <BS2>/SHA256SUMS.txt and validate/logs/final_tools_sha256_at_launch.txt; bs_hooks.py = BS's)"""
NEW_NOTE = f"""  (filled by the scorer adapter at 11:32 UTC from the files as installed; RE-FILLED by the fixer at {WHEN} UTC from
  the files as fixed: run_waves_bs.sh, score_bs.py and test_score_bs.py changed, the other five unchanged (bs_hooks.py
  = BS's); every file's sha256 also in <BS2>/SHA256SUMS.txt and <BS2>/review/FIXES.txt)"""
if OLD_NOTE in pins:
    pins = pins.replace(OLD_NOTE, NEW_NOTE)
else:
    pins = re.sub(r"  \(filled by the scorer adapter at 11:32 UTC.*?\)\n(?=  The code copy)", NEW_NOTE + "\n", pins,
                  count=1, flags=re.S)
    assert NEW_NOTE in pins
crit.write_text(head + "\nPINS" + pins + "\nHOW THE COORDINATOR PINS" + tail, encoding="utf-8")
print("criteria.txt", hashlib.sha256(crit.read_bytes()).hexdigest())
