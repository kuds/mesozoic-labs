"""FIXER of the PC amendment: rewrite ST/amend/after.txt from the files (never by hand).  Section 1: the 17 files the
amendment changed, with their sha256 now (five of them changed again by the fixer: marked).  Section 2: the
amendment editor's own files under ST/amend/ as ST/amend/after.txt listed them at 01:56:04 UTC, less ./after.txt.tmp
(absent: the re-review's I5), each re-hashed and checked equal to the listed value.  Section 3: the fixer's files
(ST/amend/FIXES.txt and ST/amend/fix/, its cmdlog.txt excluded: written last).  --check: compare only.

    PY -I -B write_after.py [--check]
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ST = Path("/home/user/work/pc_study")
AM = ST / "amend"
AFTER = AM / "after.txt"
CHANGED = ["criteria.txt", "COLAB_STEPS.txt", "design/DESIGN_NOTES.txt", "tools/pc/r10round_common.py",
           "tools/pc/complete_r10round.py", "tools/pc/pc_common.py", "tools/pc/score_pc.py", "tools/pc/pins_pc.json",
           "tools/pc/measure_r10round.py", "tools/pc/r10m_common.py", "tools/tests/synth_pc.py",
           "tools/tests/test_score_pc.py", "tools/mutate_pc.py", "tools/standin_s6.py", "tools/vp3_dryrun.py",
           "tools/smoke_pc.py", "tools/f2dry.py"]
FIXED = {"criteria.txt", "COLAB_STEPS.txt", "design/DESIGN_NOTES.txt", "tools/tests/test_score_pc.py",
         "tools/mutate_pc.py"}
REREVIEWED = AM / "fix/rereviewed_sha256.txt"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def editor_entries() -> list[tuple[str, str]]:
    """The editor's own-file entries of the re-reviewed after.txt (kept in ST/amend/fix/after_rereviewed.txt)."""
    text = (AM / "fix/after_rereviewed.txt").read_text()
    part = text.split("files the amendment created in ST/amend/", 1)[1]
    out = []
    for ln in part.splitlines():
        bits = ln.split("  ", 1)
        if len(bits) == 2 and len(bits[0]) == 64 and bits[1].startswith("./"):
            out.append((bits[0], bits[1]))
    return out


def build(stamp: str) -> str:
    lines = [f"sha256 of every file the amendment changed, rewritten {stamp} by the amendment's fixer with "
             "ST/amend/fix/tools/write_after.py (the editor's version of 01:56:04 UTC: ST/amend/fix/after_rereviewed.txt; "
             "before: ST/amend/before.txt; originals: ST/amend/orig/; the editor's diffs: ST/amend/diffs/; the fixer's "
             "diffs from the re-reviewed bytes: ST/amend/fix/diffs/). [F] = changed again by the fixer (its re-reviewed "
             "bytes in ST/amend/fix/rereviewed/, sha256 in ST/amend/fix/rereviewed_sha256.txt)"]
    for rel in CHANGED:
        lines.append(f"{sha(ST / rel)}  {rel}{'   [F]' if rel in FIXED else ''}")
    lines.append("")
    lines.append("files the amendment created in ST/amend/ (the editor's, as listed at 01:56:04 UTC, each re-hashed now "
                 "and equal; ./after.txt.tmp dropped: it never existed on disk, re-review I5; cmdlog.txt and this file "
                 "excluded):")
    for want, rel in editor_entries():
        if rel == "./after.txt.tmp":
            continue
        got = sha(AM / rel)
        assert got == want, f"{rel}: {got} is not the listed {want}"
        lines.append(f"{got}  {rel}")
    lines.append("(the F0-shaped stores under dry/vp3/*/round/ are byte copies of STANCE6 s42's F0 files and of "
                 "ST/tools/dry/pins_sc/*/stage_config.json, with synthetic manifest and RUNLOG files; listed in "
                 "dry/vp3/*.txt)")
    lines.append("")
    lines.append("files the fixer created (ST/amend/FIXES.txt and ST/amend/fix/; fix/cmdlog.txt excluded: written last):")
    fixer = [AM / "FIXES.txt"] + sorted(p for p in (AM / "fix").rglob("*") if p.is_file()
                                        and p.name != "cmdlog.txt" and "__pycache__" not in p.parts
                                        and not ("vp3" in p.parts and "round" in p.parts))
    for p in fixer:
        lines.append(f"{sha(p)}  ./{p.relative_to(AM)}")
    lines.append("(dry/vp3/*/round/ as above: the fixer's re-run of the same cases)")
    return "\n".join(lines) + "\n"


def main() -> int:
    if "--check" in sys.argv[1:]:
        old = AFTER.read_text()
        stamp = old.split("rewritten ", 1)[1].split(" ", 1)[0]
        same = build(stamp) == old
        print(f"after.txt {'equals' if same else 'DIFFERS FROM'} the files")
        return 0 if same else 1
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    AFTER.write_text(build(stamp))
    print(f"wrote {AFTER} sha256 {sha(AFTER)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
