import json, re, sys
from collections import Counter
from pathlib import Path
wf = Path(sys.argv[1]); own = sys.argv[2]
qs = Counter(); dl_keys = Counter(); xcalls = Counter(); first_drive = None; last_drive = None
for jl in sorted(wf.glob("agent-*.jsonl")):
    meta = json.loads(jl.with_suffix("").with_suffix(".meta.json").read_text()).get("description", "")
    allin = []
    rows = []
    for line in open(jl, encoding="utf-8"):
        try: rec = json.loads(line)
        except Exception: continue
        msg = rec.get("message") or {}
        if msg.get("role") != "assistant": continue
        for c in msg.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                allin.append(json.dumps(c.get("input")))
                rows.append((rec.get("timestamp"), c.get("name"), c.get("input") or {}))
    if any(own in a for a in allin): continue
    for ts, n, inp in rows:
        if n.startswith("mcp__Google_Drive__"):
            first_drive = min(first_drive or ts, ts); last_drive = max(last_drive or ts, ts)
            if n.endswith("search_files"):
                q = inp.get("query") or inp.get("q") or json.dumps(inp)
                kind = "parents-by-folder-id" if re.search(r"'[\w-]{20,}'\s+in\s+parents", q) else "OTHER: " + q[:120]
                qs[(meta.split(" #")[0], kind)] += 1
            else:
                dl_keys[(n.split("__")[-1], tuple(sorted(inp)))] += 1
        if n == "Bash":
            cmd = inp.get("command", "")
            for s in ("xfer.sh ", "xsaved.sh ", "xfer_saved.sh ", "list_and_record.sh "):
                k = cmd.count("/" + s) + cmd.count(" " + s)
                if k: xcalls[(meta.split(" #")[0], s.strip())] += k
print("first/last Drive call (UTC):", first_drive, last_drive)
print("search_files query kinds:"); [print("  ", k, v) for k, v in sorted(qs.items())]
print("other Drive tools (name, input keys):"); [print("  ", k, v) for k, v in dl_keys.items()]
print("helper-script mentions in Bash commands (definitions included):"); [print("  ", k, v) for k, v in sorted(xcalls.items())]
