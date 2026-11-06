"""Leaf diff of two JSON files. Usage: python -I json_leaf_diff.py A.json B.json [LABEL]
Prints every leaf path whose value differs (bitwise for floats), and keys present on one side only.
"""
import json
import math
import sys


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1, a) == math.copysign(1, b))
    return type(a) is type(b) and a == b


def walk(a, b, p, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            q = f"{p}.{k}" if p else str(k)
            if k not in a:
                out.append((q, "<absent>", b[k]))
            elif k not in b:
                out.append((q, a[k], "<absent>"))
            else:
                walk(a[k], b[k], q, out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{p}[{i}]", out)
    elif not same(a, b):
        out.append((p, a, b))


A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
out = []
walk(A, B, "", out)
label = sys.argv[3] if len(sys.argv) > 3 else ""
print(f"{label}: {len(out)} differing leaves")
for q, x, y in out:
    print(f"  {q}: {json.dumps(x, default=str)[:200]} -> {json.dumps(y, default=str)[:200]}")
