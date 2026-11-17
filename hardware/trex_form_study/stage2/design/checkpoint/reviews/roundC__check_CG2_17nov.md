VERDICT: merge as it stands. Merge tip bd29fe8822c2e164bb743f359b3f2a5266690579 (branch wt/wpc-cg). It merges cleanly into the working branch's current tip 9889291c. One low test gap and two points for the lead are below. None of them blocks the merge.

FINDINGS (most serious first)

1. Low, test gap: `tests/test_stage2_gates2.py`. Nothing pins the printed lines that name a failing reference (`gates2.py` line 1128, `g9_ref_lines`).
   - Evidence: I replaced that loop with `for x in []:` in process (mutation M13). The whole file still passed: "43 passed, 1 skipped in 95.40s (0:01:35)".
   - Ruling 14's "its record names those references" is pinned in the JSON (`parts.a.still_available`, `failing_references`, `followed`). Only the printed lines are not.
   - Fix: in `test_g9_ref_passes_only_when_every_part_does` (line 801), after `run(turn)`, assert that `gates2.g9_ref_lines(doc)` contains "(a) item turn/q90_left fails heading: reference transitions/turn90_pi4_left.npz".

2. Low, the builder's report misstates one number. It gives the "largest jump-to-step ratio" on V0 as 0.0041. Its probe (`.../wave7/cg1415/f_probe.py`) only computed the ratio at each task's largest-jump join.
   - My recomputation gives a largest ratio of 0.03135 on V0: start_stop, hold stand -> finite start, at left_hip_yaw, held to v0.4.
   - On S1 it is 0.006499 (brisk -> brisk_to_nominal, right_ankle_roll), which agrees with the builder.
   - Both are far below 1, so no outcome changes.

3. Low, older than this package, but ruling 14 now acts on it (`gates2.py` line 750). A turn or stop item's reference is taken as the last `transitions/` file in the item's sorted `files` list.
   - This holds for every move name in V0, V6 and V6noF5 (turn90_*, stop1_*, stop2_* all sort after start and nominal_to_*).
   - A move whose name sorted before "start" would get the start named as the reference to mark not available.
   - The refcheck item's `plan` holds the move; reading the reference from there would not depend on names. This is the lead's call; it is not owed by this package.

FOR THE LEAD

A. I confirm the builder's point A on the working tip: ruling 14's procedure cannot be completed with today's tools.
   - `sets.freeze` refuses "a frozen entry is never dropped" (tip `sets.py` line 2648; the builder cited 2146 on the older tree).
   - `manifest_of` (line 2605) leaves out entries marked not available.
   - No `sets` command marks an entry not available.
   - The gate fails closed. Where the refcheck read the latest freeze's files, a failing item's reference is always in that freeze. So G9-ref can only pass after a refreeze without the reference and a fresh refcheck.
   - The builder's options (i), (ii) and (iii) stand. Its choice (fail closed, recommends (i)) is defensible.

B. Reading of "the walk": the limit is computed over the whole window of the walk's file. It is not limited to the frames the track actually played before the join. On S1 and V0 a window is one stride and whole windows play, so the two are the same. On a look set (K = 12 strides, about 2 to 2.5 strides played before a turn, ruling 21) they can differ. Alternative: only the frames played since the walk began.

C. Builder's readings B (a join that leaves no walk) and C (the wrap from last frame to first counts): my probe shows neither alternative, "the walk it enters" or "no wrap", changes any outcome on S1 or V0.

D. Old rule text the lead still owns, confirmed on the tip:
   - `STAGE2_TRACKB_PLAN.md` line 1394 ("joint targets within 1e-6 rad") and lines 1430-1431 ("treats that reference as not available");
   - `ROUND_C_DESIGN.md` line 1189 on the tip (the builder said 1186).

E. Disk is nearly full: the volume holding /tmp shows 99% used and about 460 MB free. My scratch export of the merged tree hit "0 MB free" for a moment. I deleted my copy at once, but a write by another agent in that window could have failed.

CHECKED AND FOUND RIGHT

- Ruling 14 (`gates2.py`):
  - `g9_ref` passes only when `still_available` is empty (line 1100), so it exits 1 otherwise.
  - The record names the references in `parts.a.still_available`, `failing_references` and `followed`. The rule text carries the ruling.
  - "Still available" means the entry is usable (`r5lib.why_not` is None) OR the latest freeze lists it. A reference that cannot be named counts as available. Both choices fail closed.
  - The pinned test walks the four steps: exit 1 → marked but not refrozen, exit 1 → refrozen with the old refcheck, exit 1 → refcheck run again, exit 0.
- Ruling 15:
  - Per joint, the limit is `jump <= the walk's largest frame-to-frame change`; a NaN fails.
  - Both numbers are recorded per join and joint, plus `over`.
  - `JOIN_TOL_RAD` is removed.
  - The label "changed after a result (rule Z2's label: changed after G9's result)" is there, with the numbers 1.1e-4 and 5.3e-5 rad, "before any look set was frozen", and the rule as first written.
- My own recomputation of every join (`f_probe.py`) agrees with `gates2.joins`:
  - 0 mismatches against `gates2.joins`.
  - S1: (f) True, largest jump 1.106e-4 rad; 5 joins fail the old 1e-6 rule.
  - V0: (f) True, largest jump 5.334e-5 rad; 22 joins fail the old rule.
- Each ruling is pinned by a test that fails without it. I ran 17 mutations in process (nothing written to disk); 16 were caught.
  - Ignoring `still_available`, "entry only", "freeze only", the unnamed-reference rule, the freeze-key name and the `followed` text are each caught by the part (a) test or the pass test.
  - No wrap, `<` instead of `<=`, "walk entered", no limit without a walk, NaN passing, the max over joints, and the record's step value are each caught by the table test.
  - Restoring the old 1e-6 rule fails the S1, doctored-S1 and V0 tests ("3 failed, 41 deselected in 21.20s").
  - Only M13 (finding 1) survived.
- No leaked state: monkeypatches are undone; scripts are cached per `Library` instance and `part_f` makes a new one each call; `_marked` deep-copies.
- Only `designlab/stage2/gates2.py` and `tests/test_stage2_gates2.py` differ from the working tip in the merged tree. `run_trackb.sh` and `bar.py` are unchanged.
- No other reader of the renamed keys (`not_available` → `failing_references`, `join_tol_rad` → `join_limit`) exists on the tip.
- Record-key hashes are the registered pair: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`.
- `ruff check` on both files: "All checks passed!".
- After my runs, `git status --porcelain` is empty in the worktree and in the main checkout. No `stage2/tb/gates/` was created.

EXACT OUTPUTS

- `tests/test_stage2_gates2.py` on head bd29fe88: "43 passed, 1 skipped in 92.22s (0:01:32)". The skip is the STAGE2_SLOW r6_start test.
- `g9-ref --body 75-M-H-10 --set V0`, no `--record`, exit 1. First line: "G9-ref V0 75-M-H-10 (plain): (a) True, (b) False, (c) False, (f) True: FAIL; no closed-loop episode runs on this set and body; it goes to the lead". Last line: "G9-ref: FAIL (record sha256 6f82888052c5f6300ead3e61be425a1f64a725af980664efdc1978ba8d43fb0d)". **Part (f): True.** The overall FAIL comes from (b) and (c), as before.

MERGE READINESS

- `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre HEAD` gives a4090dff0749b3920ddca24fd35d148d2eea174b, exit 0, no conflicts.
- Commits not in the working branch: 7466f71f, 85d963b5 (merge), bd29fe88. All three end with exactly the two attribution lines and name no model anywhere else, so none needs listing.
- Bring in: tip bd29fe8822c2e164bb743f359b3f2a5266690579. The working branch moved to 9889291c after the builder's merge (rulings 21-26, the look sets package, ruling 27); the merge is still clean.

NOT CHECKED

- The tests on the merged tree (a4090dff): my scratch export of it filled the disk, so I deleted it and did not retry. The merged tree differs from head only in `sets.py`, `moves_look.py`, the look-set data, docs and two other test files. `gates2` does not import any of them, its tests do not read them, and `run_trackb.sh` does not read `stage2/sets`.
- The STAGE2_SLOW r6_start test.
- The r2lfit tests (`require_g9_ref` is unchanged).
- Anything on a look set (none is frozen).

Scratch files, all under /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave7/check_cg2/:
- test_head.txt
- g9ref_V0.txt
- f_probe.py
- f_probe.txt
- mutplug.py
- MUTATIONS.txt
- MUTATIONS_full.txt
- sets_tip.py
