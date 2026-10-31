import json, sys, re
src = sys.argv[1]
pat = re.compile(r"(null device|/dev/null|/dev/zero|outside (my|your|its) folder)", re.I)
hits = []
for raw in open(src, "rb"):
    try: d = json.loads(raw)
    except Exception: continue
    if d.get("type") != "user": continue
    msg = d.get("message") or {}
    content = msg.get("content")
    texts = []
    if isinstance(content, str): texts.append(content)
    elif isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text": texts.append(c.get("text", ""))
    for t in texts:
        for m in pat.finditer(t):
            hits.append((d.get("timestamp"), t[max(0, m.start()-200): m.end()+200].replace("\n", " ")))
print(len(hits))
for h in hits[-12:]:
    print(h[0], "|", h[1][:420])
