import json, sys, re, os, glob
d = sys.argv[1]; outdir = sys.argv[2]
key = re.compile(r"phase_screen|/home/user/work(?![/\w]*vg_tool)|\$PS\b|\bPS=|tb_buffer", re.I)
summary = []
for f in sorted(glob.glob(os.path.join(d, "agent-*.jsonl"))):
    agent = os.path.basename(f)
    calls, results = [], {}
    for raw in open(f, "rb"):
        try: x = json.loads(raw)
        except Exception: continue
        ts = x.get("timestamp") or ""
        m = x.get("message") or {}
        cont = m.get("content") if isinstance(m, dict) else None
        if not isinstance(cont, list): continue
        for c in cont:
            if not isinstance(c, dict): continue
            if c.get("type") == "tool_use":
                calls.append((ts, c.get("id"), c.get("name"), c.get("input")))
            elif c.get("type") == "tool_result":
                t = c.get("content")
                if isinstance(t, list): t = "\n".join(y.get("text", "") for y in t if isinstance(y, dict))
                results[c.get("tool_use_id")] = (t or "")
    from collections import Counter
    cnt = Counter(c[2] for c in calls)
    hits = []
    for ts, cid, name, inp in calls:
        s = json.dumps(inp)
        if key.search(s):
            hits.append((ts, name, inp, results.get(cid, "")))
    out = os.path.join(outdir, "sub_" + agent.replace(".jsonl", ".txt"))
    with open(out, "w") as fo:
        for ts, name, inp, res in hits:
            body = inp.get("command") if isinstance(inp, dict) and "command" in inp else json.dumps(inp)
            fo.write(f"=== {ts} {name}\n{body}\n--- RESULT (first 1500)\n{res[:1500]}\n")
    summary.append((agent, len(calls), dict(cnt), len(hits), calls[0][0] if calls else None, calls[-1][0] if calls else None))
for s in summary: print(s)
