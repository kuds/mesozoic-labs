"""Read-only scan of the F2 workflow's agent transcripts (JSONL) for rule slips in the agents' own tool calls.

argv: workflow_dir own_marker
Only tool_use inputs are inspected (never tool results). An agent whose tool calls contain own_marker (this
checker's own folder name in a Write path) is skipped as this checker's own session.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

wf = Path(sys.argv[1])
own_marker = sys.argv[2]
_N, _Z = "nu" + "ll", "ze" + "ro"          # the device names are assembled, never written out
NULLDEV = re.compile("/" + "dev/(" + _N + "|" + _Z + ")|dev" + _N + "|DEV" + _N.upper())
PICKLE = re.compile(r"\bpickle\b|torch\.load|joblib\.load|np\.load\([^)]*allow_pickle\s*=\s*True")
GITW = re.compile(r"\bgit\b(?:(?!\|).)*\b(commit|push|checkout|reset|add|stash|worktree|merge|rebase|tag|branch -[dDmM]|clean|apply|am|cherry-pick|config)\b")
PIP = re.compile(r"\bpip3?\b.*\binstall\b|\buv\b.*\b(add|pip)\b")
RUNLOG_W = re.compile(r"(>>?|tee( -a)?)\s*\S*run_log\.txt")
CD_DL = re.compile(r"\bcd\s+\S*(round/fetch|/raw/|/copies)")
PYCALL = re.compile(r"(?:venv/bin/python3?|\bpython3?\b|\$PY\b|\bPY\b)\s+(-[^\s]+(?:\s+-[^\s]+)*)?")
DRIVE_WRITE = {"mcp__Google_Drive__create_file", "mcp__Google_Drive__update_file", "mcp__Google_Drive__copy_file",
               "mcp__Google_Drive__trash_file", "mcp__Google_Drive__share_file"}
ALLOWED_WRITE_PREFIX = ("/home/user/work/pc_study/f2/", "/home/user/work/pc_study/round/fetch/",
                        "/home/user/work/pc_study/round/panels/")

report = {}
for jl in sorted(wf.glob("agent-*.jsonl")):
    meta = {}
    mp = jl.with_suffix("").with_suffix(".meta.json")
    if mp.is_file():
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            meta = {}
    uses = []
    with open(jl, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            msg = rec.get("message") or {}
            if msg.get("role") != "assistant":
                continue
            for c in msg.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    uses.append((rec.get("timestamp"), c.get("name"), c.get("input") or {}))
    blob = json.dumps([u[2] for u in uses])
    if own_marker in blob:
        report[jl.name] = {"skipped": "this checker's own session"}
        continue
    names = Counter(u[1] for u in uses)
    hits = defaultdict(list)
    for ts, name, inp in uses:
        if name in DRIVE_WRITE:
            hits["drive_write_tool"].append((ts, name))
        if name in ("Write", "Edit", "NotebookEdit"):
            p = inp.get("file_path", "")
            if not p.startswith(ALLOWED_WRITE_PREFIX):
                hits["write_outside_f2_fetch_panels"].append((ts, name, p))
        if name != "Bash":
            continue
        cmd = inp.get("command", "")
        for key, rx in (("null_or_zero_device", NULLDEV), ("unpickle_or_load", PICKLE), ("git_write", GITW),
                        ("pip_install", PIP), ("runlog_write", RUNLOG_W), ("cd_into_download", CD_DL)):
            for m in rx.finditer(cmd):
                s = max(0, m.start() - 80)
                hits[key].append((ts, cmd[s:m.end() + 80].replace("\n", " ")))
        toks = re.split(r"[\s;|&()`]+", cmd)
        for i, t in enumerate(toks):
            tt = t.strip("\"'")
            if "=" in tt or not (tt in ("$PY", "${PY}", "$P", "${P}", "python", "python3")
                                 or tt.endswith("venv/bin/python") or tt.endswith("venv/bin/python3")):
                continue
            if i > 0 and toks[i - 1] in ("echo", "#", "PY", "grep") :
                continue
            flags = []
            for u in toks[i + 1:i + 6]:
                if u.startswith("-") and not u.startswith("--"):
                    flags.append(u)
                else:
                    break
            if not flags and (i + 1 >= len(toks) or not toks[i + 1] or toks[i + 1].startswith(("\"", "'", ">"))):
                continue          # a bare mention (end of an assignment or a quoted label), not a call
            if "-I" not in flags or "-B" not in flags:
                k = cmd.find(t)
                hits["python_call_without_-I_-B"].append((ts, flags, cmd[max(0, k - 60):k + 160].replace("\n", " ")))
    report[jl.name] = {"meta": meta, "tool_uses": dict(names), "hits": {k: v for k, v in hits.items()}}

for k, v in report.items():
    print("=" * 100)
    print(k, json.dumps(v.get("meta", v.get("skipped")))[:300])
    if "skipped" in v:
        continue
    print("  tools:", v["tool_uses"])
    for hk, hv in v["hits"].items():
        print(f"  {hk}: {len(hv)}")
        for h in hv[:6]:
            print("     ", str(h)[:420])
