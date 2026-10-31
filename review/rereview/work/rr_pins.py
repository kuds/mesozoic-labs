"""Re-reviewer: the scorer's own PINS check (score_bs.pins_lines / pins_items, fixer P-m2) on the REAL criteria.txt.
Read only, except a copy root under <PS>/review/rereview/pins_root (criteria.txt and SCREEN.json copied there, plus
a fake APPROVED written there only) to exercise provisional_items with an APPROVED present."""
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PS = Path("/home/user/work/phase_screen")
OUT = PS / "review" / "rereview" / "pins_root"
spec = importlib.util.spec_from_file_location("score_bs_rr", str(PS / "tools/score_bs.py"))
SB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SB)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


text = (PS / "criteria.txt").read_text()
lines = SB.pins_lines(text)
print("pins_lines:", json.dumps(lines, indent=1))
tool_now = {t: sha(PS / "tools" / t) for t in SB.PINNED_TOOLS}
print("PINNED_TOOLS:", SB.PINNED_TOOLS)
print("pins_items (real text, real root, no APPROVED):", SB.pins_items(text, PS, sha(PS / "SCREEN.json"), tool_now, None))
print("scorer sha named in text:", sha(PS / "tools/score_bs.py") in text)

assert not OUT.exists()
OUT.mkdir(parents=True)
shutil.copyfile(PS / "criteria.txt", OUT / "criteria.txt")
shutil.copyfile(PS / "SCREEN.json", OUT / "SCREEN.json")
cases = {
    "approved_current": dict(tool_now),
    "approved_stale_run_bs": {**tool_now, "run_bs.py": "8f5f2af7f976fbaa1388cf0f6173ef3421844b919b00220c561a8e46b80b66e5"},
    "approved_stale_score_bs": {**tool_now, "score_bs.py": "fa692f68b36b14b1d73f34df87d73a18661ce78b99d889abc2867486bb538040"},
}
for label, tools in cases.items():
    appr = {"criteria_sha256": sha(OUT / "criteria.txt"), "screen_sha256": sha(OUT / "SCREEN.json"),
            "code_tree_digest": SB.C.TREE_DIGEST, "tool_sha256": tools, "approved_by": "rereview-fake"}
    (OUT / "APPROVED").write_text(json.dumps(appr))
    items = SB.provisional_items(OUT, [], {})
    pins = [x for x in items if "PINS" in x or "APPROVED" in x or "scorer" in x]
    print(f"{label}: {len(items)} items; pin-related: {pins}")
    print(f"   all items: {items}")
(OUT / "APPROVED").unlink()
