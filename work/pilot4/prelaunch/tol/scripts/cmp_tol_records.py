"""Tolerance lane: this lane's validation records (the tool with TWIN_HEADING_TOL_DEG 1e-4) against the fix lane's
(out/fix/val_final, the tool with 1e-5): every value equal but the tool sha256, the files block (the figure's and
traces' sha256s), the constant TWIN_HEADING_TOL_DEG and the turned-twin check's list of flagged episodes. Reads JSON
only (the fix lane's cmp_old_new_records.py, with this stage's expected keys).

usage: python -I -B cmp_tol_records.py OLD_DIR NEW_DIR OUT.json
"""
import json
import sys
from pathlib import Path

old_d, new_d, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        res = []
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if k not in a or k not in b:
                res.append(f"{p} only in {'new' if k in b else 'old'}")
            else:
                res += diff(a[k], b[k], p)
        return res
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} length {len(a)} != {len(b)}"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def expected(x: str) -> bool:
    return (x.startswith(("tool_sha256", "files", "constants.TWIN_HEADING_TOL_DEG:", "checks.turned_twin.bad")))


res = {}
for name in sorted(p.name for p in new_d.glob("*.json")):
    a, b = json.loads((old_d / name).read_text()), json.loads((new_d / name).read_text())
    d = diff(a, b)
    res[name] = {"differences": len(d), "expected": [x for x in d if expected(x)][:12],
                 "unexpected": [x for x in d if not expected(x)],
                 "turned_twin_old": {k: (len(v) if k == "bad" else v) for k, v in a["checks"]["turned_twin"].items()},
                 "turned_twin_new": {k: (len(v) if k == "bad" else v) for k, v in b["checks"]["turned_twin"].items()}}
Path(out).write_text(json.dumps(res, indent=1) + "\n")
for k, v in res.items():
    print(k, v["differences"], "differences,", len(v["unexpected"]), "unexpected", v["unexpected"][:5])
