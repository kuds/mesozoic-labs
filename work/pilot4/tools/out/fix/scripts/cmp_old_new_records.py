"""Fix lane: the re-run validation records against the straightness lane's (the tool before the fix stage): every
value equal but the tool sha256, the files block and the two new report keys of (b). Reads JSON only.

usage: python -I -B cmp_old_new_records.py OLD_DIR NEW_DIR OUT.json
"""
import json
import sys
from pathlib import Path

old_d, new_d, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
NEW_KEYS = {"report_crab_median_deg", "report_end_cross_track_median_m"}


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if k not in a or k not in b:
                out.append(f"{p} only in {'new' if k in b else 'old'}")
            else:
                out += diff(a[k], b[k], p)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} length {len(a)} != {len(b)}"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


res = {}
for name in sorted(p.name for p in new_d.glob("*.json")):
    a, b = json.loads((old_d / name).read_text()), json.loads((new_d / name).read_text())
    d = diff(a, b)
    expected = [x for x in d if x.startswith(("tool_sha256", "files")) or x in
                {f"parts.b.{k} only in new" for k in NEW_KEYS}]
    res[name] = {"differences": len(d), "expected": len(expected), "unexpected": [x for x in d if x not in expected]}
Path(out).write_text(json.dumps(res, indent=1) + "\n")
for k, v in res.items():
    print(k, v["differences"], "differences,", len(v["unexpected"]), "unexpected", v["unexpected"][:5])
