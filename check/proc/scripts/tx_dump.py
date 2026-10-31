import json, sys
f, out = sys.argv[1:3]
with open(out, "w") as fo:
    for i, l in enumerate(open(f)):
        c = json.loads(l); inp = c["input"]
        s = inp.get("command", "") if c["name"] == "Bash" else json.dumps(inp)[:600]
        fo.write(f"#{i} {c['ts']} {c['name']}\n    " + s[:4000].replace("\n", "\n    ") + "\n")
        r = c.get("result") or {}
        fo.write(f"    RESULT {r.get('ts')} " + (r.get("text") or "")[:1500].replace("\n", "\n    | ") + "\n")
