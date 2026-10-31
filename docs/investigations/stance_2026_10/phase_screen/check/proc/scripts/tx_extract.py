import json, sys
src, out, t0, t1 = sys.argv[1:5]
n_lines = 0; calls = []; results = {}
with open(src, "rb") as fh:
    for raw in fh:
        n_lines += 1
        try:
            d = json.loads(raw)
        except Exception:
            continue
        ts = d.get("timestamp") or ""
        msg = d.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "tool_use":
                calls.append({"ts": ts, "id": c.get("id"), "name": c.get("name"), "input": c.get("input"),
                              "sidechain": d.get("isSidechain"), "uuid": d.get("uuid")})
            elif c.get("type") == "tool_result":
                txt = c.get("content")
                if isinstance(txt, list):
                    txt = "\n".join(x.get("text", "") for x in txt if isinstance(x, dict))
                results[c.get("tool_use_id")] = {"ts": ts, "text": (txt or "")[:4000]}
win = [c for c in calls if t0 <= c["ts"] <= t1]
with open(out, "w") as fo:
    for c in win:
        c["result"] = results.get(c["id"])
        fo.write(json.dumps(c) + "\n")
print("lines", n_lines, "tool_use total", len(calls), "in window", len(win))
from collections import Counter
print(Counter(c["name"] for c in win))
print("first/last ts in window:", win[0]["ts"] if win else None, win[-1]["ts"] if win else None)
print("overall first/last call ts:", calls[0]["ts"], calls[-1]["ts"])
