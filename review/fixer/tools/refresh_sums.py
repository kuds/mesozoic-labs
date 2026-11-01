"""Fixer: refresh ST/tools/SHA256SUMS.txt in place: the same paths in the same order (a path that no longer exists is
dropped and printed), every hash recomputed; the fixer's new tool files are not added here (they are listed in
ST/review/fixer/SHA256SUMS.txt)."""
import hashlib
from pathlib import Path
T = Path("/home/user/work/pc_study/tools")
lines = (T / "SHA256SUMS.txt").read_text().splitlines()
out = []
for ln in lines:
    h, p = ln.split(None, 1)
    f = (T / p.strip()).resolve()
    if not f.is_file():
        print(f"dropped (absent): {p}")
        continue
    nh = hashlib.sha256(f.read_bytes()).hexdigest()
    if nh != h:
        print(f"changed: {p}")
    out.append(f"{nh}  {p.strip()}")
(T / "SHA256SUMS.txt").write_text("\n".join(out) + "\n")
