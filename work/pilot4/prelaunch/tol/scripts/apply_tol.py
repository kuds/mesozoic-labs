"""Tolerance lane: apply maintainer decision (35), C6's alternative (a) in out/fix/text_corrections.txt, to the pinned
walking pilot 4 tools. Exact text replacements only, each old text required exactly once in its file:
  pilot4_straight.TWIN_HEADING_TOL_DEG 1e-5 -> 1e-4 (and its docstring);
  pilot4_score.STRAIGHT_CONSTANTS["TWIN_HEADING_TOL_DEG"] follows, and STRAIGHT_TOOL_SHA256 = the edited tool's sha256;
  tests/test_pilot4_straight.py: the constants test, its docstring, and the record test's expectation (0 flagged on
  every seed);
  test_pilot4_score.py: the default PILOT4_STRAIGHT_VAL becomes this lane's val_final (the records written by the
  edited tool, which that test requires: tool_sha256 == STRAIGHT_TOOL_SHA256).
Before any write, every file's sha256 must equal its entry in the pinned list kept in ORIG_DIR/../SHA256SUMS_before.txt,
and its bytes are kept in ORIG_DIR. Reads and writes text only; imports nothing from the repository; trains nothing.

usage: python -I -B apply_tol.py TOOLS LANE
"""

import hashlib
import json
import sys
from pathlib import Path

TOOLS, LANE = Path(sys.argv[1]), Path(sys.argv[2])
ORIG = LANE / "orig"
BEFORE = LANE / "SHA256SUMS_before.txt"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


pinned = {}
for ln in BEFORE.read_text().splitlines():
    if ln and not ln.startswith("#"):
        h, p = ln.split("  ", 1)
        pinned[p] = h

STRAIGHT = "pilot4_straight.py"
SCORE = "pilot4_score.py"
T_STRAIGHT = "tests/test_pilot4_straight.py"
T_SCORE = "test_pilot4_score.py"

EDITS = {
    STRAIGHT: [
        ("            1e-3), the command-blind turned twin (heading at state 1000 = the twin's + angle within 1e-5 deg), and",
         "            1e-3), the command-blind turned twin (heading at state 1000 = the twin's + angle within 1e-4 deg), and"),
        ("TWIN_HEADING_TOL_DEG = 1e-5   # command-blind walker: heading at state 1000 = the twin's + angle within this\n",
         "TWIN_HEADING_TOL_DEG = 1e-4   # command-blind walker: heading at state 1000 = the twin's + angle within this\n"
         "#                               (maintainer decision (35): 1e-4 deg; v3's 1e-5, from 15 episodes, flagged 3 of\n"
         "#                               s42's 40 nudged entry episodes at 1.1e-5 to 2.7e-5 deg)\n"),
    ],
    T_STRAIGHT: [
        ("  panel heading + angle within 1e-5 deg, the pass check at state 650,",
         "  panel heading + angle within 1e-4 deg, the pass check at state 650,"),
        ('c["TWIN_HEADING_TOL_DEG"] == 1e-5 and', 'c["TWIN_HEADING_TOL_DEG"] == 1e-4 and'),
        ("    # v3's tolerance (1e-5 deg at 10 s) is applied literally: an episode beyond it is flagged (exit 1). Measured on the\n"
         "    # 40-episode panels: s42 flags 3 of 40 (largest 2.7e-5 deg), s52 and s60 none (text issue: the 1e-5 came from\n"
         "    # 15 episodes); every deviation stays far below the 15-deg offset, so the walker keeps the offset.\n"
         "    tw = ch[\"turned_twin\"]\n"
         "    assert tw[\"episodes\"] == n and all(abs(b[\"dev_deg\"]) > ps.TWIN_HEADING_TOL_DEG for b in tw[\"bad\"])\n"
         "    assert tw[\"max_abs_dev_deg\"] < 1e-4 and len(tw[\"bad\"]) == {42: 3, 52: 0, 60: 0}[seed]\n",
         "    # The tolerance is 1e-4 deg at 10 s (maintainer decision (35); v3's 1e-5, from 15 episodes, flagged 3 of s42's 40\n"
         "    # at 1.1e-5 to 2.7e-5 deg): no episode is flagged on any seed (exit 0), and every deviation stays far below the\n"
         "    # 15-deg offset, so the walker keeps the offset.\n"
         "    tw = ch[\"turned_twin\"]\n"
         "    assert ps.TWIN_HEADING_TOL_DEG == 1e-4\n"
         "    assert tw[\"episodes\"] == n and tw[\"bad\"] == [] and tw[\"max_abs_dev_deg\"] <= ps.TWIN_HEADING_TOL_DEG\n"),
    ],
    T_SCORE: [
        ("  PILOT4_STRAIGHT_VAL   the straightness tool's validation records (p3val_s<seed>.json), re-run by the fix stage with\n"
         "                        the pinned tool (default: out/fix/val_final)\n",
         "  PILOT4_STRAIGHT_VAL   the straightness tool's validation records (p3val_s<seed>.json), re-run by the tolerance stage\n"
         "                        with the pinned tool (default: ../prelaunch/tol/val_final from this file's folder)\n"),
        ('STRAIGHT_VAL = Path(os.environ.get("PILOT4_STRAIGHT_VAL") or HERE / "out" / "fix" / "val_final")',
         'STRAIGHT_VAL = Path(os.environ.get("PILOT4_STRAIGHT_VAL") or HERE.parent / "prelaunch" / "tol" / "val_final")'),
    ],
}
SCORE_EDITS_FIXED = [
    ("#: verdict follows from the record's own values under those bars.\n",
     "#: verdict follows from the record's own values under those bars. Tolerance stage: TWIN_HEADING_TOL_DEG 1e-4 deg\n"
     "#: (maintainer decision (35)), so the tool's sha256 below moved with it.\n"),
    ('"TWIN_HEADING_TOL_DEG": 1e-05,', '"TWIN_HEADING_TOL_DEG": 0.0001,'),
]
OLD_TOOL_SHA = 'STRAIGHT_TOOL_SHA256 = "sha256:93d5c558ebffb80036f0f054e918629c17af07d0a18e03ac844532ec2d487848"'

files = [STRAIGHT, SCORE, T_STRAIGHT, T_SCORE]
data = {}
for f in files:
    b = (TOOLS / f).read_bytes()
    if sha(b) != pinned.get(f):
        raise SystemExit(f"{f}: sha256 {sha(b)} is not the pinned {pinned.get(f)}: nothing written")
    data[f] = b.decode()
ORIG.mkdir(parents=True, exist_ok=True)
for f in files:
    dst = ORIG / f.replace("/", "__")
    if dst.exists() and dst.read_text() != data[f]:
        raise SystemExit(f"{dst} exists with other bytes: nothing written")
    dst.write_text(data[f])


def apply(text: str, edits, name: str) -> str:
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{name}: the old text occurs {n} times, not once: {old[:90]!r}: nothing written")
        text = text.replace(old, new)
    return text


new = {STRAIGHT: apply(data[STRAIGHT], EDITS[STRAIGHT], STRAIGHT),
       T_STRAIGHT: apply(data[T_STRAIGHT], EDITS[T_STRAIGHT], T_STRAIGHT),
       T_SCORE: apply(data[T_SCORE], EDITS[T_SCORE], T_SCORE)}
tool_sha = sha(new[STRAIGHT].encode())
new[SCORE] = apply(data[SCORE], SCORE_EDITS_FIXED + [
    (OLD_TOOL_SHA, f'STRAIGHT_TOOL_SHA256 = "sha256:{tool_sha}"')], SCORE)
for f in files:
    (TOOLS / f).write_text(new[f])
rec = {f: {"before": pinned[f], "after": sha(new[f].encode())} for f in files}
rec["straight_tool_sha256"] = "sha256:" + tool_sha
(LANE / "logs" / "apply_tol.json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec, indent=1))
