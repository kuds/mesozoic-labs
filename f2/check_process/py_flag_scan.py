import json, re, sys
from pathlib import Path
wf = Path(sys.argv[1]); own = sys.argv[2]
rx = re.compile(r'(venv/bin/python3?|\$\{?PY\}?|\$\{?P\}?|(?<![\w/])python3?)"?\s+-I(?:\s+-[a-zA-Z]+)*(?=\s)')
for jl in sorted(wf.glob("agent-*.jsonl")):
    meta = json.loads(jl.with_suffix("").with_suffix(".meta.json").read_text()).get("description", "")
    rows = []
    allcmds = []
    for line in open(jl, encoding="utf-8"):
        try: rec = json.loads(line)
        except Exception: continue
        msg = rec.get("message") or {}
        if msg.get("role") != "assistant": continue
        for c in msg.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                inp = c.get("input") or {}
                allcmds.append(json.dumps(inp))
                if c.get("name") == "Bash":
                    cmd = inp.get("command", "")
                    for m in rx.finditer(cmd):
                        if "-B" not in m.group(0).split():
                            rows.append((rec.get("timestamp"), cmd[max(0, m.start()-30):m.end()+110].replace("\n", " ")))
    if any(own in x for x in allcmds):
        continue
    for ts, s in rows:
        print(f"{meta} | {ts} | {s}")
