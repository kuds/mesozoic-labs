"""VP4 TD part (report only): the TD values the stand-in printed (the scorer on STANCE6's C3-verified ledgers and the
comparators' FMAN-checked copies) against DESIGN/comparators.out (sha256 pinned in the criteria PINS).  Prints."""
import hashlib
import re
import sys
from pathlib import Path

ST = Path("/home/user/work/pc_study")
co = ST / "design/comparators.out"
h = hashlib.sha256(co.read_bytes()).hexdigest()
print(f"DESIGN/comparators.out sha256 {h} = pin 8c924c37...: {h == '8c924c37871fb956c84e7c0e8b78b914e9b68def1283e9c76ee12d014d7d8b46'}")
text = co.read_text()
m = re.search(r'comparators \{"60": ([0-9.]+), "42": ([0-9.]+), "80": ([0-9.]+)\}; the lowest [0-9.]+ \(repr ([0-9.]+)\)', text)
design = {60: float(m.group(1)), 42: float(m.group(2)), 80: float(m.group(3))}
print(f"comparators.out TD (6 d.p.): {design}; lowest repr {m.group(4)}")
ok = True
for arm in ("pc", "off"):
    out = Path(sys.argv[1]) / f"vp4/standin_{arm}.out"
    line = [ln for ln in out.read_text().splitlines() if "TD from the scorer's C3-verified ledgers" in ln][0]
    pairs = re.findall(r"\((\d+), ([0-9.e-]+)\)", line)
    got = {}
    for s, v in pairs:
        got.setdefault(int(s), []).append(float(v))
    for s, vals in got.items():
        e = all(round(v, 6) == design[s] for v in vals) and len(set(vals)) == 1
        ok &= e
        print(f"  {arm} s{s}: stand-in TD {vals} rounded to 6 d.p. equal to comparators.out {design[s]}: {e}")
    low = min(v[0] for v in got.values())
    e = repr(low) == m.group(4)
    ok &= e
    print(f"  {arm} lowest TD repr {low!r} equal to comparators.out repr {m.group(4)}: {e}")
print(f"VP4 TD part: {'PASS' if ok else 'FAIL'}")
