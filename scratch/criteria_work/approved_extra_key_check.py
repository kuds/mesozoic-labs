"""Does the UNMODIFIED run_bs.approval_problems accept an APPROVED whose tool_sha256 carries an extra "score_bs.py"
entry?  Runs approval_problems against a temporary root (bs_root patched in memory; nothing under the real <BS> is
written; no run is started).  Also: the same APPROVED with a wrong five-tool pin is refused (the check is live)."""
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

BS = Path(sys.argv[1])
TMP = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("run_bs", BS / "tools" / "run_bs.py")
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)
(TMP / "criteria.txt").write_text("stand-in criteria\n")
shutil.copyfile(BS / "SCREEN.json", TMP / "SCREEN.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


appr = {"criteria_sha256": sha(TMP / "criteria.txt"), "screen_sha256": sha(TMP / "SCREEN.json"),
        "code_tree_digest": rb.C.TREE_DIGEST,
        "tool_sha256": {**{t: sha(BS / "tools" / t) for t in rb.C.TOOL_FILES}, "score_bs.py": sha(BS / "tools" /
                                                                                                  "score_bs.py")},
        "approved_by": "stand-in", "utc": "stand-in"}
(TMP / "APPROVED").write_text(json.dumps(appr, indent=1))
rb.bs_root = lambda: TMP
print("with the extra score_bs.py key:", rb.approval_problems())
appr["tool_sha256"]["run_bs.py"] = "0" * 64
(TMP / "APPROVED").write_text(json.dumps(appr, indent=1))
print("with a wrong run_bs.py pin:", rb.approval_problems())
shutil.rmtree(TMP)
