"""Deep diff of the setup check records: head (prelaunch/setup/runs) against 7700f28 (postfix/code/runs).

Usage: python -I deep_diff_setup.py HEAD_RUNS_DIR P7700_RUNS_DIR OUT_JSON
For each seed: every leaf path that differs (bitwise; -0.0 vs 0.0 differs, NaN == NaN), split into numeric leaves
(int/float/bool on both sides, or lists of numbers) and others; keys present on one side only are listed. Also
compares every .npy/.npz/.json file both run folders hold by sha256 (common names), and numeric arrays bitwise.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

H, P, OUT = (Path(a) for a in sys.argv[1:4])


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) or isinstance(x, bool)


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return type(a) is type(b) and a == b


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in a:
                out["only_7700"].append(p)
            elif k not in b:
                out["only_head"].append(p)
            else:
                walk(a[k], b[k], p, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["other"].append([p_short(path), f"list len {len(a)} vs {len(b)}"])
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    else:
        out["leaves"] += 1
        if not same(a, b):
            if is_num(a) and is_num(b):
                out["numeric"].append([path, a, b])
            else:
                out["other"].append([path, str(a)[:160], str(b)[:160]])


def p_short(p):
    return p


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


res = {}
for s in (42, 52, 60):
    a = json.load(open(H / f"s{s}" / "setup_check.json"))
    b = json.load(open(P / f"s{s}" / "setup_check.json"))
    out = {"leaves": 0, "numeric": [], "other": [], "only_head": [], "only_7700": []}
    walk(a, b, "", out)
    files = {}
    for f in sorted((P / f"s{s}").glob("*")):
        g = H / f"s{s}" / f.name
        if f.is_file() and g.is_file() and f.suffix in (".npy", ".npz", ".json") and f.name != "setup_check.json":
            files[f.name] = sha(f) == sha(g)
            if f.suffix == ".npy":
                x, y = np.load(f, allow_pickle=False), np.load(g, allow_pickle=False)
                files[f.name + ":array_equal"] = bool(x.shape == y.shape and np.array_equal(x, y, equal_nan=True))
            if f.suffix == ".npz":
                x, y = np.load(f, allow_pickle=False), np.load(g, allow_pickle=False)
                files[f.name + ":array_equal"] = bool(set(x.files) == set(y.files) and all(
                    np.array_equal(x[k], y[k], equal_nan=True) for k in x.files))
    out["files"] = files
    out["files_only_7700"] = sorted(f.name for f in (P / f"s{s}").glob("*") if not (H / f"s{s}" / f.name).exists())
    res[str(s)] = out
json.dump(res, open(OUT, "w"), indent=1, default=str)
for s, o in res.items():
    print(f"s{s}: leaves {o['leaves']}, numeric differing {len(o['numeric'])}, other differing {len(o['other'])}, "
          f"only_head {len(o['only_head'])}, only_7700 {len(o['only_7700'])}")
    for n in o["numeric"][:40]:
        print("   NUM", n)
    print("   files", o["files"], "only_7700", o["files_only_7700"])
