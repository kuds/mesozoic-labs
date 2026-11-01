"""Compare two equiv_capture outputs bitwise.  usage: equiv_compare.py DIR_A DIR_B"""
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

a, b = Path(sys.argv[1]), Path(sys.argv[2])
diffs = []
ca, cb = np.load(a / "caps.npz"), np.load(b / "caps.npz")
ka, kb = set(ca.files), set(cb.files)
for k in sorted(ka ^ kb):
    diffs.append(f"key only in {'A' if k in ka else 'B'}: {k}")
same = 0
for k in sorted(ka & kb):
    x, y = ca[k], cb[k]
    if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
        diffs.append(f"array differs: {k} {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
    else:
        same += 1
print(f"arrays: {same} identical of {len(ka & kb)} shared; {len(ka ^ kb)} unshared")

MASK = re.compile(r"(time|date|duration|elapsed|wall|_at$|started|finished|host|pid|path|dir)", re.I)
ADDR = re.compile(rb"0x[0-9a-f]{6,}")


def norm_json(obj, key=""):
    if isinstance(obj, dict):
        return {k: ("<masked>" if MASK.search(k) else norm_json(v, k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [norm_json(v, key) for v in obj]
    if isinstance(obj, str):
        return obj.replace(str(a), "<RUN>").replace(str(b), "<RUN>")
    return obj


def zip_members(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            out[n] = z.read(n)
    return out


masked_report = []
fa = sorted(p.relative_to(a / "run") for p in (a / "run").rglob("*") if p.is_file())
fb = sorted(p.relative_to(b / "run") for p in (b / "run").rglob("*") if p.is_file())
if fa != fb:
    diffs.append(f"file lists differ: {sorted(set(map(str, fa)) ^ set(map(str, fb)))}")
for rel in sorted(set(fa) & set(fb)):
    pa, pb = a / "run" / rel, b / "run" / rel
    da, db = pa.read_bytes(), pb.read_bytes()
    if da == db:
        print(f"file identical bytes: {rel}")
        continue
    if rel.suffix == ".zip":
        ma, mb = zip_members(pa), zip_members(pb)
        if set(ma) != set(mb):
            diffs.append(f"{rel}: members differ {set(ma) ^ set(mb)}")
        for n in sorted(set(ma) & set(mb)):
            if ma[n] == mb[n]:
                continue
            if n == "data":
                ja, jb = json.loads(ma[n]), json.loads(mb[n])
                ka2 = {k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k)}
                # SB3 data: each entry may hold a pickled blob (:serialized:) whose bytes carry addresses/closures
                for k in sorted(ka2):
                    va, vb = ja.get(k), jb.get(k)
                    if k in ("start_time", "_current_progress_remaining") or MASK.search(k):
                        masked_report.append(f"{rel}:data:{k} masked")
                        continue
                    if isinstance(va, dict) and isinstance(vb, dict) and ":serialized:" in va:
                        sa = {kk: vv for kk, vv in va.items() if kk != ":serialized:"}
                        sb = {kk: vv for kk, vv in vb.items() if kk != ":serialized:"}
                        if sa == sb:
                            masked_report.append(f"{rel}:data:{k} differs only in :serialized: blob")
                            continue
                    diffs.append(f"{rel}:data:{k} differs: {str(va)[:200]} vs {str(vb)[:200]}")
            else:
                diffs.append(f"{rel}: member {n} differs")
        continue
    if rel.suffix == ".json":
        ja, jb = json.loads(da), json.loads(db)
        if norm_json(ja) == norm_json(jb):
            masked_report.append(f"{rel}: equal after masking time/path keys")
        else:
            diffs.append(f"{rel}: json differs after masking")
        continue
    if rel.suffix == ".npz":
        na, nb = np.load(io.BytesIO(da), allow_pickle=False), np.load(io.BytesIO(db), allow_pickle=False)
        bad = [k for k in set(na.files) | set(nb.files)
               if k not in na.files or k not in nb.files or na[k].tobytes() != nb[k].tobytes()]
        if bad:
            masked_report.append(f"{rel}: npz arrays differing: {sorted(bad)}")
            diffs.append(f"{rel}: npz arrays differ {sorted(bad)[:10]}")
        else:
            masked_report.append(f"{rel}: npz arrays identical (container bytes differ)")
        continue
    diffs.append(f"{rel}: bytes differ (sha {hashlib.sha256(da).hexdigest()[:12]} vs {hashlib.sha256(db).hexdigest()[:12]})")
for m in masked_report:
    print("masked:", m)
print(f"DIFFERENCES: {len(diffs)}")
for d in diffs[:400]:
    print("  ", d)
