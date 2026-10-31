"""Fixer: re-fill criteria.txt's PINS lines from the files as they stand (every 64-hex PINS line naming a file)."""
import hashlib
import re
from pathlib import Path

PS = Path("/home/user/work/phase_screen")
p = PS / "criteria.txt"
s = p.read_text()
head, rest = s.split("\nPINS", 1)
pins, tail = rest.split("\nHOW THE COORDINATOR PINS", 1)
line = re.compile(r"^([ \t]+)([\w.-]+(?:/[\w.-]+)*)([ \t]+)([0-9a-f]{64})([ \t]*)$", re.M)
changes = []


def sub(m):
    rel = m.group(2)
    now = hashlib.sha256((PS / rel).read_bytes()).hexdigest()
    if now != m.group(4):
        changes.append(f"{rel}: {m.group(4)} -> {now}")
    else:
        changes.append(f"{rel}: unchanged {now}")
    return f"{m.group(1)}{rel}{m.group(3)}{now}{m.group(5)}"


pins2 = line.sub(sub, pins)
p.write_text(head + "\nPINS" + pins2 + "\nHOW THE COORDINATOR PINS" + tail)
print("\n".join(changes))
