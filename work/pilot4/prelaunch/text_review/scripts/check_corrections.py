"""For each of the 18 corrections, check whether its NEW (or INSERT) text appears in the frozen candidate, comparing
with whitespace collapsed; when it does not, print the word-level difference. Read only.
Usage: python -I check_corrections.py TEXT_CORRECTIONS FINAL_TEXT OUT_JSON"""
import difflib
import json
import re
import sys
from pathlib import Path

tc, final, out = (Path(p) for p in sys.argv[1:4])
lines = tc.read_text().splitlines()
blocks, cur, mode = {}, None, None
for ln in lines:
    m = re.match(r"^(C\d+[ab]?)\s+v3 line", ln)
    if m:
        cur, mode = m.group(1), None
        blocks[cur] = {"new": []}
        continue
    if ln.startswith("====") or ln.startswith("Not changed in the text"):
        cur, mode = (None, None) if ln.startswith("Not changed") else (cur, None)
        continue
    if cur is None:
        continue
    if ln.strip() in ("NEW:", "INSERT:"):
        mode = "new"
        continue
    if ln.strip() in ("OLD:", "AFTER:"):
        mode = "old"
        continue
    if mode == "new":
        blocks[cur]["new"].append(ln)


def norm(s):
    return " ".join(s.split())


ft = norm(final.read_text())
res = {}
for cid, b in blocks.items():
    new = norm("\n".join(b["new"]))
    if new in ft:
        res[cid] = {"verbatim": True}
        continue
    # find the best matching window in the final text by its first 60 characters
    words = new.split()
    fw = ft.split()
    best, best_i = 0.0, None
    n = len(words)
    for i in range(0, max(1, len(fw) - n)):
        if fw[i] != words[0]:
            continue
        r = difflib.SequenceMatcher(None, words, fw[i:i + n + 40]).ratio()
        if r > best:
            best, best_i = r, i
    diff = []
    if best_i is not None:
        sm = difflib.SequenceMatcher(None, words, fw[best_i:best_i + n + 40])
        for op, a1, a2, b1, b2 in sm.get_opcodes():
            if op != "equal":
                diff.append({"op": op, "correction": " ".join(words[a1:a2]), "final": " ".join(fw[best_i + b1:best_i + b2])})
    res[cid] = {"verbatim": False, "ratio": round(best, 4), "diff": diff}
out.write_text(json.dumps(res, indent=1))
for cid, r in res.items():
    print(cid, "verbatim" if r["verbatim"] else f"NOT verbatim (ratio {r['ratio']})")
    for d in r.get("diff", []):
        print("   ", d["op"], "| C:", d["correction"][:300], "| F:", d["final"][:300])
