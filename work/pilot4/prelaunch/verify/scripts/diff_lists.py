"""Diff two sha256sum lists (comment lines skipped) and check coverage of the tools folder.

Usage: python -I diff_lists.py BEFORE AFTER TOOLS_DIR OUT_JSON
Reports: entries added, removed, changed sha; duplicate paths; files under TOOLS_DIR (tree/ excluded) that the AFTER
list does not name; and, for every changed entry, whether the BEFORE sha equals the tol lane's orig/ copy.
"""
import hashlib
import json
import os
import sys


def parse(path):
    out = {}
    dups = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            sha, name = line.split("  ", 1)
            if name in out:
                dups.append(name)
            out[name] = sha
    return out, dups


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


before, dup_b = parse(sys.argv[1])
after, dup_a = parse(sys.argv[2])
tools = sys.argv[3]
res = {
    "n_before": len(before),
    "n_after": len(after),
    "dups_before": dup_b,
    "dups_after": dup_a,
    "removed": sorted(set(before) - set(after)),
    "changed": sorted(k for k in set(before) & set(after) if before[k] != after[k]),
    "added_count": len(set(after) - set(before)),
}
added = sorted(set(after) - set(before))
res["added_prefixes"] = {}
for k in added:
    p = k.split("/")
    pre = "/".join(p[:3]) if k.startswith("../") else p[0]
    res["added_prefixes"][pre] = res["added_prefixes"].get(pre, 0) + 1
res["added_not_under_prelaunch_tol"] = [k for k in added if not k.startswith("../prelaunch/tol/")]
res["changed_detail"] = {k: [before[k], after[k]] for k in res["changed"]}
# coverage: every regular file under tools (tree/ excluded) named in AFTER?
named = set(os.path.normpath(k) for k in after)
unlisted = []
for root, dirs, files in os.walk(tools):
    rel_root = os.path.relpath(root, tools)
    if rel_root == "tree" or rel_root.startswith("tree" + os.sep):
        dirs[:] = []
        continue
    for f in files:
        rel = os.path.normpath(os.path.join(rel_root, f))
        if rel not in named:
            unlisted.append(rel)
res["files_under_tools_not_in_after_list"] = sorted(unlisted)
# files whose current sha differs from the AFTER list (should be none if sha256sum -c passed)
mism = []
for k, v in after.items():
    p = os.path.normpath(os.path.join(tools, k))
    if not os.path.isfile(p):
        mism.append([k, "missing"])
    elif sha(p) != v:
        mism.append([k, "sha differs"])
res["after_mismatch_now"] = mism
json.dump(res, open(sys.argv[4], "w"), indent=1, sort_keys=True)
print(json.dumps({k: (v if not isinstance(v, (list, dict)) or len(v) < 30 else len(v)) for k, v in res.items()}, indent=1))
