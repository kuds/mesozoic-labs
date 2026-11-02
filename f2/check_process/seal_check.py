"""Independent re-derivation of the measuring copy's seal (read only).

Recomputes the tree digest (sorted 'relpath<TAB>sha256' lines of every file but the seal), counts files,
and lists every path in the copy (files and folders) whose mtime or ctime is later than the seal file's mtime.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

copy = Path(sys.argv[1])
seal_p = copy / ".r10m_seal.json"
seal = json.loads(seal_p.read_text())
seal_m = seal_p.stat().st_mtime
lines = []
later = []
for dirpath, dirnames, filenames in os.walk(copy):
    dirnames.sort()
    for d in dirnames:
        st = os.lstat(Path(dirpath) / d)
        if st.st_mtime > seal_m + 1 or st.st_ctime > seal_m + 1:
            later.append(f"DIR  {Path(dirpath, d).relative_to(copy)} mtime={st.st_mtime:.3f} ctime={st.st_ctime:.3f}")
    for fn in filenames:
        p = Path(dirpath) / fn
        rel = p.relative_to(copy).as_posix()
        st = os.lstat(p)
        if st.st_mtime > seal_m + 1 or st.st_ctime > seal_m + 1:
            later.append(f"FILE {rel} mtime={st.st_mtime:.3f} ctime={st.st_ctime:.3f}")
        if rel == ".r10m_seal.json":
            continue
        if p.is_symlink():
            lines.append(f"{rel}\tsymlink:{os.readlink(p)}")
        else:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            lines.append(f"{rel}\t{h.hexdigest()}")
lines.sort()
digest = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
st = os.lstat(copy)
print(json.dumps({
    "seal_file_sha256": hashlib.sha256(seal_p.read_bytes()).hexdigest(),
    "seal": seal,
    "seal_file_mtime": seal_m,
    "recomputed_tree_digest": digest,
    "recomputed_files": len(lines),
    "digest_equal": digest == seal["tree_digest"],
    "files_equal": len(lines) == seal["files"],
    "copy_root_mtime": st.st_mtime,
    "copy_root_ctime": st.st_ctime,
    "paths_changed_after_seal_plus_1s": later,
    "pycache_or_pyc": [x.split("\t")[0] for x in lines if "__pycache__" in x or x.split("\t")[0].endswith(".pyc")],
}, indent=1))
