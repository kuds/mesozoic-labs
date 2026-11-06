"""Build the text-only durable copy of walking pilot 4 (decision (42)): every pinned file that is text and <= 5 MB,
at its scratchpad-relative position under DEST/work/, plus the frozen text, the approval, the run log and the stage
results; binaries and larger files are listed in LEFT_OUT.txt with their sha256. Usage: python -I build_handoff.py DEST
"""
import hashlib
import json
import os
import shutil
import sys

SP = "/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad"
W = SP + "/work"
DEST = sys.argv[1]
LIMIT = 5_000_000
BINARY_EXT = (".zip", ".npz", ".npy", ".pkl", ".png", ".mp4", ".pt", ".gz", ".tar")
LISTS = [  # (list file, the folder its relative paths are read from)
    (W + "/pilot4/tools/SHA256SUMS.txt", W + "/pilot4/tools"),
    (W + "/pilot4/prelaunch/fix/pin/SHA256SUMS_lanes.txt", W + "/pilot4/prelaunch"),
    (W + "/pilot4/render/render/SHA256SUMS_render.txt", W + "/pilot4/render/render"),
    (W + "/pilot4/render/coord/SHA256SUMS_rtext.txt", W + "/pilot4/render/text"),
]
EXTRA = [
    W + "/run_log.txt",
    W + "/pilot4/frozen_candidate/criteria_pilot4.txt",
    W + "/pilot4/frozen_candidate/APPROVED.json",
    W + "/pilot4/postfix/fix/criteria_pilot4_draft_v3.txt",
    W + "/pilot4/render/render/CMDLOG.txt",
    W + "/pilot4/render/coord/apply_coord.py",
    W + "/pilot4/render/coord/coord.diff",
    W + "/pilot4/render/coord/rtext_check.txt",
    W + "/pilot4/render/coord/criteria_pilot4_before_coord.txt",
    W + "/pilot4/build/ALL_RESULTS.json",
    W + "/pilot4/postfix/ALL_RESULTS.json",
    W + "/pilot4/tools/out/ALL_RESULTS.json",
    W + "/pilot4/prelaunch/ALL_RESULTS.json",
    W + "/pilot4/render/ALL_RESULTS.json",
    W + "/stance_cert/ALL_RESULTS.json",
    W + "/pilot4/coord_verify/build_handoff.py",
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_text(path):
    if path.endswith(BINARY_EXT) or os.path.getsize(path) > LIMIT:
        return False
    try:
        with open(path, "rb") as fh:
            fh.read().decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def place(src):
    rel = os.path.relpath(os.path.realpath(src), SP)
    if rel.startswith(".."):
        raise SystemExit(f"outside the scratchpad: {src}")
    dst = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return rel


if os.path.exists(DEST) and os.listdir(DEST):
    raise SystemExit(f"{DEST} is not empty")
os.makedirs(DEST, exist_ok=True)
copied, left, seen = [], [], set()
for list_file, base in LISTS:
    copied.append(place(list_file))
    for line in open(list_file, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.rstrip("\n").split("  ", 1)
        src = os.path.normpath(os.path.join(base, name))
        if src in seen:
            continue
        seen.add(src)
        if sha(src) != digest:
            raise SystemExit(f"sha256 mismatch against {list_file}: {src}")
        rel = os.path.relpath(src, SP)
        if is_text(src):
            copied.append(place(src))
        else:
            left.append((digest, os.path.getsize(src), rel, os.path.basename(list_file)))
for src in EXTRA:
    if os.path.realpath(src) in seen:
        continue
    if not is_text(src):
        raise SystemExit(f"extra file not text or over the limit: {src}")
    copied.append(place(src))
with open(os.path.join(DEST, "LEFT_OUT.txt"), "w", encoding="utf-8") as fh:
    fh.write("# Pinned files NOT in this text-only copy (binary or > 5 MB), decision (42); they stay in the session\n")
    fh.write("# scratchpad. Columns: sha256, bytes, scratchpad-relative path, the pin list that names it.\n")
    for digest, size, rel, lst in sorted(left, key=lambda r: r[2]):
        fh.write(f"{digest}  {size}  {rel}  {lst}\n")
summary = {"copied_files": len(copied), "left_out_files": len(left),
           "left_out_bytes": sum(r[1] for r in left)}
json.dump(summary, open(os.path.join(DEST, "BUILD_SUMMARY.json"), "w"), indent=1)
print(json.dumps(summary))
