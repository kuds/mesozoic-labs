# Walking pilot 4: durable text-only copy (never merged)

This orphan branch holds a text-only copy of walking pilot 4's frozen pre-registration and evidence, made from the
session scratchpad after the maintainer froze the text (decision (41), 2026-11-06T15:55:52Z) and chose a text-only
copy (decision (42)). Paths keep their scratchpad-relative positions under `work/`.

- Frozen text: `work/pilot4/frozen_candidate/criteria_pilot4.txt`, sha256
  `fa92e483560207194675dbc4cabe79479149ecb28c520249842aefb901b41ce2`; approval record `APPROVED.json` beside it.
- Pilot branch: `claude/pilot-walk-r10-p4` at `0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8` (pushed; never merged).
- Pinned tools and evidence: `work/pilot4/tools/SHA256SUMS.txt` (1152 entries), `work/pilot4/prelaunch/fix/pin/
  SHA256SUMS_lanes.txt` (614), `work/pilot4/render/render/SHA256SUMS_render.txt` (324),
  `work/pilot4/render/coord/SHA256SUMS_rtext.txt` (16).
- Left out: every pinned file that is binary (checkpoint zips, npz/pkl arrays, png figures, mp4 videos) or over 5 MB:
  644 files, listed in `LEFT_OUT.txt` with sha256, size and path. They stay in the session scratchpad.
- The session's run log: `work/run_log.txt`. Stage results: the `ALL_RESULTS.json` files under `work/pilot4/` and
  `work/stance_cert/`.

Check (each from the folder named; `--ignore-missing` skips the left-out files):

    (cd work/pilot4/tools && sha256sum -c --ignore-missing SHA256SUMS.txt)                 # 896 OK
    (cd work/pilot4/prelaunch && sha256sum -c --ignore-missing fix/pin/SHA256SUMS_lanes.txt)  # 490 OK
    (cd work/pilot4/render/render && sha256sum -c --ignore-missing SHA256SUMS_render.txt)   # 60 OK
    (cd work/pilot4/render/text && sha256sum -c ../coord/SHA256SUMS_rtext.txt)                # 16 OK
    sha256sum -c SHA256SUMS_COPY.txt                                                         # every file here

OK plus left out equals each list's length (896 + 256, 490 + 124, 60 + 264, 16 + 0). This departs from the frozen
text's durable-copy check (1152 OK etc.) by the maintainer's decision (42).
