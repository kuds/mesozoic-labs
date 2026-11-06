"""Rule-slip scan of one lane's CMDLOG.txt (read only). Usage: python -I scan_cmdlog.py CMDLOG LANE_DIR SCRATCH OUT_JSON

SCRATCH is the scratchpad root, passed at run time (never spelled here). Output masks every absolute path that is
neither under SCRATCH nor a /tmp/* path, so no system device path is ever written by this scan.
Flags, per line: git push / remote / commit / checkout / reset / clean / switch / stash; pip; a bare python or
python3 interpreter; sleep; network tools; MUJOCO_GL values other than disable; rm / mv / cp / sed -i / tee /
touch / chmod / ln / mkdir whose absolute path arguments leave the lane folder; redirect targets outside the lane
folder; lines that spell the scratchpad path literally (the product-family word sits in it); lines running a
command before any logger existed are not detectable here.
"""
import json
import re
import sys

CMDLOG, LANE, SCRATCH, OUT = sys.argv[1:5]
lines = open(CMDLOG, encoding="utf-8", errors="replace").read().splitlines()
PATH_RE = re.compile(r"(?<![\w$])(/[^\s'\";|&<>()]+)")
VARS_LANE = ("$K", "${K}", "$T", "${T}", '"$K', '"$T', "$L", '"$L', "$LANE", "$OUT", "$J")


def mask(s):
    def rep(m):
        p = m.group(1)
        if p.startswith(SCRATCH):
            return "SP" + p[len(SCRATCH):]
        if p.startswith("/tmp/"):
            return p
        return "<masked-abs-path>"
    return PATH_RE.sub(rep, s)


def outside_lane(path):
    return path.startswith("/") and not path.startswith(LANE)


flags = {k: [] for k in ("git_write", "pip", "bare_python", "sleep", "network", "mujoco_gl_not_disable",
                         "file_op_outside_lane", "redirect_outside_lane", "literal_scratch_path")}
red_re = re.compile(r"(?:^|[^<&\d])(\d?>>?|&>)\s*(\S+)")
for i, ln in enumerate(lines, 1):
    body = ln
    if SCRATCH in ln:
        flags["literal_scratch_path"].append(i)
    if re.search(r"\bgit\b[^|;&]*\b(push|commit|checkout|reset|clean|switch|stash|rebase|merge)\b", body) or \
            re.search(r"remote\s+(set-url|add|remove|rename)", body):
        flags["git_write"].append([i, mask(body)[:240]])
    if re.search(r"\bpip3?\s+install\b|-m\s+pip\b", body):
        flags["pip"].append([i, mask(body)[:240]])
    if re.search(r"(^|[\s;&|(])python3?(\s|$)", body.split("] $ ", 1)[-1]):
        flags["bare_python"].append([i, mask(body)[:240]])
    if re.search(r"(^|[\s;&|(])sleep\s+\d", body):
        flags["sleep"].append([i, mask(body)[:240]])
    if re.search(r"\b(curl|wget|ssh|scp|rsync|nc)\s", body):
        flags["network"].append([i, mask(body)[:240]])
    for m in re.finditer(r"MUJOCO_GL=(\S+)", body):
        if m.group(1).strip("'\"") != "disable":
            flags["mujoco_gl_not_disable"].append([i, mask(body)[:240]])
    cmd = body.split("] $ ", 1)[-1]
    if re.search(r"(^|[\s;&|(])(rm|mv|cp|tee|touch|chmod|ln|mkdir)\s|sed\s+-i", cmd):
        args = PATH_RE.findall(cmd)
        out = [a for a in args if outside_lane(a)]
        # a cp/ln source outside the lane is a read; flag only when the LAST path (the target) leaves the lane,
        # or any path of rm / mv / touch / chmod / mkdir / sed -i / tee does
        verb_write_all = re.search(r"(^|[\s;&|(])(rm|mv|touch|chmod|mkdir|tee)\s|sed\s+-i", cmd)
        if out and (verb_write_all or (args and outside_lane(args[-1]))):
            flags["file_op_outside_lane"].append([i, mask(cmd)[:300]])
    for m in red_re.finditer(cmd):
        tgt = m.group(2).strip("'\"")
        if tgt.startswith("&") or tgt.startswith("$K") or tgt.startswith("${K") or tgt.startswith("$T") or \
                tgt.startswith("$L") or tgt.startswith("$F") or tgt.startswith("$J") or tgt.startswith("$OUT") or \
                tgt.startswith(LANE) or tgt.startswith("$(") or tgt == "":
            continue
        if not tgt.startswith("/") and not tgt.startswith("$"):
            # relative target: resolved against the cwd of that command; listed for review
            flags["redirect_outside_lane"].append([i, "relative target " + mask(tgt)[:80], mask(cmd)[:200]])
            continue
        flags["redirect_outside_lane"].append([i, mask(tgt)[:120], mask(cmd)[:200]])
summary = {k: (len(v) if isinstance(v, list) else v) for k, v in flags.items()}
json.dump({"lines": len(lines), "summary": summary, "flags": flags}, open(OUT, "w"), indent=1)
print("lines", len(lines), summary)
for k, v in flags.items():
    if k == "literal_scratch_path":
        continue
    for x in v[:25]:
        print(k, x)
