"""Independent re-derivation of the deviation tool's difference list (read only).

argv: dev_folder
Diffs pinned.out vs deviation.out line by line and flattens pinned.json / deviation.json to (path, value) pairs.
Classifies each difference by its path / content; prints counts and any unclassified difference.
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
po = (d / "pinned.out").read_text(encoding="utf-8").splitlines()
do = (d / "deviation.out").read_text(encoding="utf-8").splitlines()
print("output lines:", len(po), len(do))
out_diff = [(i + 1, a, b) for i, (a, b) in enumerate(zip(po, do)) if a != b]
for i, a, b in out_diff:
    kind = ("blindness" if "OUTCOME BLINDNESS" in a or (a.strip().startswith(("s60:", "s42:", "s80:")) and "switch-off" in a)
            else "ONE TEST" if "THE ONE TEST" in a
            else "C1 listing" if '"lines": {' in a and '"902": "SNAPSHOT"' in a and a.replace('"902": "SNAPSHOT", ', "") == b
            else "UNCLASSIFIED")
    print(f"OUT line {i}: {kind}")
    if kind == "UNCLASSIFIED":
        print("   pinned:   ", a[:300])
        print("   deviation:", b[:300])


def flat(x, pre, acc):
    if isinstance(x, dict):
        if not x:
            acc[pre] = "{}"
        for k, v in x.items():
            flat(v, f"{pre}.{k}" if pre else str(k), acc)
    elif isinstance(x, list):
        acc[pre] = "[]" if not x else acc.get(pre, None) or "list"
        for i, v in enumerate(x):
            flat(v, f"{pre}[{i}]", acc)
    else:
        acc[pre] = json.dumps(x)
    return acc


pj = flat(json.loads((d / "pinned.json").read_text(encoding="utf-8")), "", {})
dj = flat(json.loads((d / "deviation.json").read_text(encoding="utf-8")), "", {})
keys = sorted(set(pj) | set(dj))
jd = [(k, pj.get(k, "(absent)"), dj.get(k, "(absent)")) for k in keys if pj.get(k, "(absent)") != dj.get(k, "(absent)")]
counts = {}
for k, a, b in jd:
    if k.startswith("blindness."):
        kind = "blindness"
    elif k.startswith("reading.one_test."):
        kind = "ONE TEST"
    elif ".completeness.C1.value.lines.902" in k or ".p1.value.lines.902" in k:
        kind = "C1 listing"
    else:
        kind = "UNCLASSIFIED"
    counts[kind] = counts.get(kind, 0) + 1
    if kind == "UNCLASSIFIED":
        print("JSON UNCLASSIFIED", k, a[:120], "|", b[:120])
print("JSON differing paths (own flattening; a list's own path counted only when it changes between empty and not):",
      len(jd), counts)
for k, a, b in jd:
    print("  ", k, ":", a[:80], "|", b[:80])
