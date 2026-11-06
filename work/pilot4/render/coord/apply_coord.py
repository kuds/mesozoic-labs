"""Coordinator edits to the pilot-4 frozen candidate: the render verify lane's seven fixes (one should-fix, six minor).

Each OLD must occur exactly once; every line written must be <= 118 characters. Usage: python -I apply_coord.py TEXT
"""
import sys

PATH = sys.argv[1]
RT = "87c5d91467ed683e57e80efa8e4910183bd8fb6aef345971348e78077c4855bb"
EDITS = [
    # 1 (should-fix): pin the render CMDLOG and the render/text evidence.
    (
        "  stage's evidence (every file of SP/work/pilot4/render/render/ but its CMDLOG.txt and its clone tree/; its tmp/ is\n"
        "  empty) is pinned by its SHA256SUMS_render.txt, sha256\n",
        "  stage's evidence (every file of SP/work/pilot4/render/render/ but its CMDLOG.txt, the manifest itself and its\n"
        "  clone tree/; its tmp/ is empty) is pinned by its SHA256SUMS_render.txt, sha256\n",
    ),
    (
        "  folder [measured: render/text/logs/render_manifest_check.txt, render_files_all.txt and\n"
        "  render_files_manifest.txt].\n",
        "  folder [measured: render/text/logs/render_manifest_check.txt, render_files_all.txt and\n"
        "  render_files_manifest.txt]. The render lane's CMDLOG.txt, which that manifest leaves out, is sha256\n"
        "  bdd41284d53c29b180d1570214f9e7bf9eaa9e17f5489e9e10ff8f32aacd3e79 [measured]. This text's checks of the render\n"
        "  stage (every file of SP/work/pilot4/render/text/ but its CMDLOG.txt, its empty tmp/ and the two dry-run copies of\n"
        "  this text, out/criteria_pilot4_dryrun*.txt) are pinned by SP/work/pilot4/render/coord/SHA256SUMS_rtext.txt,\n"
        f"  sha256 {RT} (16 entries), sha256sum -c 16 OK from\n"
        "  render/text [measured: render/coord/rtext_check.txt].\n",
    ),
    # 2: which item holds the render record.
    ("      checked (the next item's render record).\n", "      checked (the fourth item's render record).\n"),
    # 3: the GL label.
    (
        "      [measured: the wrapper, render/render/logs/withx_diff_vs_p3_score.txt]; software GL, Mesa llvmpipe [measured as\n"
        "      recorded: the render lane's gl_check.py run, its output not kept in a file].\n",
        "      [measured: the wrapper, render/render/logs/withx_diff_vs_p3_score.txt]; software GL, Mesa llvmpipe [inferred:\n"
        "      the render lane's report of its gl_check.py run; the output is in no file].\n",
    ),
    # 4: the byte total is the 80 mp4s.
    (
        "      file (25 fps, 40.0 s), each h264 640x480 with 1,000 frames, 49,527,598 bytes in all; camera tracking, distance\n",
        "      file (25 fps, 40.0 s), each h264 640x480 with 1,000 frames, the 80 mp4s 49,527,598 bytes in all; camera\n"
        "      tracking, distance\n",
    ),
    # 5: the xkb side effect.
    (
        "    - Outside its folder the render stage's checks found one change, the /var/lib/xkb directory's modification time,\n"
        "      at each Xvfb start, nothing left there but README.compiled [measured: render/render/logs/final_checks.txt;\n"
        "      inferred: Xvfb's keymap compiler]; the pinned tools' list 1152 OK before and after; the smoke's files\n",
        "    - Outside its folder the render stage's checks found one change, the /var/lib/xkb directory's modification time,\n"
        "      last at 15:12:19 UTC, when the last replay ended, nothing left there but README.compiled [measured:\n"
        "      render/render/logs/final_checks.txt; inferred: Xvfb's keymap compiler, which reported 6 times in\n"
        "      render/render/logs/xvfb.stderr.log]; the pinned tools' list 1152 OK before and after; the smoke's files\n",
    ),
    # 6: the durable copy's render/text check.
    (
        "    SHA256SUMS_render.txt run from the copy's render/render folder must give 324 OK.\n",
        "    SHA256SUMS_render.txt run from the copy's render/render folder must give 324 OK, and sha256sum -c\n"
        "    ../coord/SHA256SUMS_rtext.txt run from the copy's render/text folder must give 16 OK.\n",
    ),
    # 7: the Changes bullet names the new checks and these edits.
    (
        "    pending list, Q18); the title, Status, \"Maintainer decisions behind this pilot\" and \"Decisions taken\" name\n"
        "    (37)-(39); Status pins the render stage's evidence and the lanes' folders name its two.\n",
        "    pending list, Q18); the title, Status, \"Maintainer decisions behind this pilot\" and \"Decisions taken\" name\n"
        "    (37)-(39); Status pins the render stage's evidence and the lanes' folders name its two; the durable-copy item\n"
        "    gains two checks, sha256sum -c SHA256SUMS_render.txt 324 OK and SHA256SUMS_rtext.txt 16 OK where the copy\n"
        "    keeps the render evidence. Then the render verify lane's seven fixes, applied by the coordinator with\n"
        "    render/coord/apply_coord.py (the text before them kept as render/coord/criteria_pilot4_before_coord.txt,\n"
        "    sha256 00350cf4...): the render CMDLOG's sha256 and the render/text pin in Status, the fourth-item pointer,\n"
        "    the GL label, the mp4 byte total, the /var/lib/xkb wording and these two checks; no rule or value changed.\n",
    ),
]

text = open(PATH, encoding="utf-8").read()
for i, (old, new) in enumerate(EDITS, 1):
    n = text.count(old)
    if n != 1:
        sys.exit(f"edit {i}: OLD occurs {n} times; nothing written")
    text = text.replace(old, new)
long = [(k, len(line)) for k, line in enumerate(text.split("\n"), 1) if len(line) > 118]
if long:
    sys.exit(f"lines over 118 characters: {long}; nothing written")
open(PATH, "w", encoding="utf-8").write(text)
print(f"applied {len(EDITS)} edits; {text.count(chr(10))} lines")
