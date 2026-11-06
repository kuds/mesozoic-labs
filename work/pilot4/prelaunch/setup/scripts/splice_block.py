"""Replace lines START..END (1-based, inclusive) of FILE with the text of BLOCK. Scratch helper of prelaunch/setup."""
import sys
from pathlib import Path

path, start, end, block = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]).read_text()
lines = path.read_text().splitlines(keepends=True)
first, last = lines[start - 1], lines[end - 1]
print(f"replacing lines {start}-{end}: first {first.rstrip()!r}, last {last.rstrip()!r}")
path.write_text("".join(lines[: start - 1]) + block + "".join(lines[end:]))
