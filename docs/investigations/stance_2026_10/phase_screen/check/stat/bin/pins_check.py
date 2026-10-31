"""PS statistic check: the pins the STATUS rests on (own code, reads only): APPROVED vs files, every run_meta's
criteria / screen / APPROVED / tool sha256, check.json's and markers.json's reader sha256, the scorer's sha256 in
score.txt, and the criteria's PINS lines vs the files."""
import hashlib
import json
import re
from pathlib import Path

PS = Path("/home/user/work/phase_screen")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap = json.loads((PS / "APPROVED").read_text())
ap_sha = sha(PS / "APPROVED")
bad = []
if sha(PS / "criteria.txt") != ap["criteria_sha256"]:
    bad.append("criteria")
if sha(PS / "SCREEN.json") != ap["screen_sha256"]:
    bad.append("SCREEN")
for t, h in ap["tool_sha256"].items():
    if sha(PS / "tools" / t) != h:
        bad.append(f"tool {t}")
crit = (PS / "criteria.txt").read_text()
pins = re.findall(r"^[ \t]+([\w.-]+(?:/[\w.-]+)*)[ \t]+([0-9a-f]{64})[ \t]*$", crit, re.M)
for rel, h in pins:
    if sha(PS / rel) != h:
        bad.append(f"PINS {rel}")
print(f"APPROVED sha256 {ap_sha}; criteria, SCREEN, 6 tools == APPROVED; {len(pins)} PINS lines == files: {not bad}")
names = [f"{a}_{s}" for s in (2901, 2911, 2921, 2931) for a in ("control", "critic_phase")]
for n in names:
    m = json.loads((PS / "runs" / n / "run_meta.json").read_text())
    c = json.loads((PS / "runs" / n / "check.json").read_text())
    k = json.loads((PS / "reads" / n / "markers.json").read_text())
    ok = (m["criteria_sha256"] == ap["criteria_sha256"] and m["screen_sha256"] == ap["screen_sha256"]
          and m["approved_sha256"] == ap_sha and all(ap["tool_sha256"][t] == h for t, h in m["tool_sha256"].items())
          and c["tool_sha256"] == ap["tool_sha256"]["read_bs.py"] == k["tool_sha256"]
          and m["code"]["tree_digest"] == ap["code_tree_digest"])
    if not ok:
        bad.append(f"run {n}")
    print(f"  {n:<18} run_meta criteria/screen/APPROVED/tools/code digest and check/markers reader sha256 pinned: {ok}")
s = (PS / "score/score.txt").read_text().splitlines()[1]
ok = ap["tool_sha256"]["score_bs.py"] in s
print(f"score.txt names the scorer sha256 pinned by APPROVED: {ok}")
if not ok:
    bad.append("score sha")
print(f"PINS: {'all hold' if not bad else bad}")
