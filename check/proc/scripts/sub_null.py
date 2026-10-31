import json, sys, glob, os
d = sys.argv[1]
needle = "/dev/" + "null"
needle2 = "/dev/" + "zero"
for f in sorted(glob.glob(os.path.join(d, "agent-*.jsonl"))):
    for raw in open(f, "rb"):
        try: x = json.loads(raw)
        except Exception: continue
        m = x.get("message") or {}
        cont = m.get("content") if isinstance(m, dict) else None
        if not isinstance(cont, list): continue
        for c in cont:
            if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "Bash":
                s = (c.get("input") or {}).get("command", "")
                if needle in s or needle2 in s:
                    i = s.find(needle) if needle in s else s.find(needle2)
                    print(os.path.basename(f)[6:14], x.get("timestamp"), "...", s[max(0, i-160): i+20].replace("\n", " "))
