**Verdict: merge as it stands, by cherry-picking four commits.** I found no defect that gives a wrong result, fails open, or touches a frozen set's records or the record-key hashes. Everything here is simulation only (prescribed motion). I changed no tracked file, committed nothing and froze nothing.

**Findings, most serious first**
1. **Merge route (process).** Merging the tip 38842ac2 would bring 17 older commits into the working branch whose messages do not end with the two required lines:
   - 14 end with "Co-Authored-By: Claude Fable 5.2": b72c61c7, 33652cb7, de96fc06, e0d64e82, f0630220, c9b88198 (a merge), e2696b2a, a16aba49, e62028df, ddf46509, 01cbe305, 1c1c1419, f60fbd3b, 33e9fbdc.
   - 3 merges have no attribution lines: 7fef4d62, 5436c8da, 7f96371b.
   - Their content is already in the working branch as the squash 036ec9d6.
   - **Fix:** cherry-pick in order 447acad1, c8ad130a, f8f56820, 403987c6. I simulated this on 32b5ecc5 with merge-tree: each step is clean and the final tree equals HEAD's (313e56e2).
   - The working branch has since moved to 3da0c6a8. That commit only adds `roundC__build_LS_17nov.md`, which none of the four commits touch.
   - `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre HEAD` is clean (tree 4d7c014c).
   - The six new commits (8555f72f, 447acad1, c8ad130a, f8f56820, 403987c6, 38842ac2) end with the two lines and name no model anywhere else.
2. **LOW (sets.py `verify`, about line 2466).** `verify` reads `blocked` alone. On a record written by older code (a dict), it under-reports:
   - On the committed V6/60-N-T-10, verify prints one stop line (nominal_to_v0.4).
   - `blocks_in_records` on the same records names both nominal_to_v0.4 and start.
   - This fails closed (the set still fails verify) and goes away at the next merge or rebuild.
   - Optional fix: verify also lists any stop that `blocks_in_records` shows and `blocked` leaves out.
3. **LOW (test gap).** Every stand-in test builds with `procs=0`. The path with 2 or more processes, where the start and the change both run and are both named, is untested. By reading the code (as_completed takes the running change; `_blocked_after` keeps the start), it names both.

**What I checked and found right**
- **Ruling 18, built as the paragraph words it and nothing more.**
  - Where Stage 1's start law comes from:
    - `moves.stage1(body, "start", H)` → `transitions.TRANSITIONS["start"]` = ("stand", "nominal") → `refs.walk_specs` → `sizing.NOMINAL`, which is GaitSpec(duty 0.60) at sizing.py:58.
    - `_param_fns("start")` gives a constant duty `g.duty` (transitions.py:256-266).
    - The lift is `moves.TODAY`'s sin²(πu) (moves.py:247-249), which equals `gait.lift_profile(u, 0.5)` (gait.py:430).
  - Measured, not assumed:
    - Stage 1's own plan: every swing has duty 0.6, and the lift law peaks at u = 0.5.
    - The committed Stage 1 `start.npz` on 75-M-H-10 and on 60-N-T-10: every swing's duty lies in the frame bracket (0.6, 0.60625], and the peak is at u = 0.5.
    - The values are the same on every body. The builder's 0.60 and 0.5 are right.
  - Mixing is done by `amp` sampled at the lift-off, on the lift peak and the duty only. The lift height is not mixed.
  - The rule `S2L-L1L4-start` is in `move.rules` and its text is in transitions.json.
  - The turn in place, turns, changes and stops are excluded: `definition_of` adds the rule only through `rule_ids` for a start.
- **On my own scratch arrays** (the five bodies I built, see the table), each start swing has the mix at its lift-off:
  - Left, lift-off 0, amp 0.05792: duty 0.605008, inside the frame bracket [0.6, 0.60625]; the look walk's 0.686 lies outside it. Lift peak fitted on `foot_target` z: 0.49276 (rms about 1e-17 m).
  - Right, lift-off 0.5, amp 0.68256: duty 0.659018, inside [0.65625, 0.6625]. Lift peak 0.41468.
  - Later swings: 0.686465 and 0.375.
  - At plan level the duty error is 0 and the profile error 2.2e-16.
- **No other move changes.** On V6/75-M-H-10 I rebuilt `nominal_to_v0.4` and `stop2_v0.4` with the new code.
  - Their sha256 equal the committed ones (19fd45aa…, e28bba8c…), and the entries are identical key for key.
- **Rulings 23 and 26, and the extra commit, each fail without their code.** I ran the new tests with the old module swapped in on a scratch copy of the worktree:
  - Baseline with the new code: "15 passed in 10.98s".
  - Working branch's `moves_look.py`: "4 failed in 6.15s". Messages include "assert 0.6864653129237532 == 0.6050080709245438", "0.5862816369688507 == 0.5992054324132358" and "'Bind' object has no attribute 'start_law'".
  - `sets.py` at 447acad1 (before 23): "6 failed, 3 passed in 2.20s". The look-set demo fails with "TypeError: 'NoneType' object is not iterable". The 3 that pass are `test_a_built_sets_lead_in_follows_its_walks[V0|F25F9|F25F9cl]`, which only had their dict shape adjusted.
  - `sets.py` at c8ad130a (before 26): "1 failed in 2.49s"; the right turn's reason still carries "also outside join_command".
  - `sets.py` at f8f56820: "1 failed in 1.04s", KeyError 'S2L-L1L4-start'.
- **The fixer's own demo script, on the new code:** after merging nominal_to_v0.4 with no `blocks_set`, `blocked` is [start] and verify still says "the set is stopped: start: …".
- **Dict-shape tests adjusted:** `test_a_failed_start_stops_the_set` (old lines 713 and 716) and `test_a_built_sets_lead_in_follows_its_walks` (old 2242 and 2250).
- **Frozen sets:**
  - Their `blocked` is null and `blocks_in_records` is empty.
  - The verify logs are identical, notes included, between the working branch's code and the new code.
  - No module in a record key changed.
  - `regress.sources` does not include `moves_look` or `sets` (checked by reading).
- **Only the four package files differ from the working branch.** The worktree status is clean, and nothing was written in it during this review.

**Exact outputs**
- Tests:
  - moves_look + moves + sets: "254 passed, 3 skipped in 369.46s (0:06:09)". The skips are test_stage2_moves.py:392 and :1459 and test_stage2_sets.py:2206, all STAGE2_SLOW.
  - lookbar, whole: "44 passed, 6 skipped in 60.51s (0:01:00)". The 6 skips are at :1763, whole closed-loop episodes (STAGE2_SLOW).
  - `test_the_modules_of_every_existing_record_key_are_state_as` and `test_no_round_a_or_stage_1_file_differs_from_state_a`: both PASSED.
- Verify:
  - "V0 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
  - "F25F9 75-M-H-10: verify ok: 12 references, 11 not available, 0 failures; freeze 0"
  - "F25F9cl 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
- Hashes (before and after): `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`

**My scratch read (not a record), new code, start only**
Knee limit is 7.558 rad/s. Limits: join com 1 mm, join zmp 2 mm, zmp_error 4 mm.

| Set/body | Knee ratio | Knee peak | ik failed frames | join_a com/zmp (mm) | join_b com/zmp (mm) | zmp_error last (mm) | rolling_split | Duration |
|---|---|---|---|---|---|---|---|---|
| V6/75-M-H-10 | 0.8041 | left, phase 1.05, u 0.159 (swing off at 1.0, amp 1, L10 flat-foot lift) | 0 | 0.509/0.232 | 0.275/0.530 | 1.234 | 56 rolling, 0 failed | 5.661 s, 800 frames |
| V6noF5/75-M-H-10 | 0.8453 | right, phase 2.55 | 0 | 0.509/0.232 | 0.263/0.511 | 1.224 | none | 5.661 s |
| V6/75-M-H-10-X | 0.7948 (governed by neck peak 0.8052) | left, phase 1.05 | 0 | 0.512/0.233 | 0.274/0.537 | 1.247 | 56, 0 failed | 5.661 s |
| V6/60-N-T-10 | 0.9216 | left, phase 1.05 | 0 | 0.408/0.219 | 0.354/0.523 | 1.266 | 56, 0 failed | 5.093 s |
| V6noF5/60-N-T-10 | 0.9791 | right, phase 2.55 | 0 | 0.408/0.219 | 0.349/0.513 | 1.266 | none | 5.093 s |

- Every gate passes on every body, and nothing is over a Stage 0 limit. **On V6/75 and V6noF5/75 the start now passes every gate.**
- After a passing start, the 75 cm scratch sets read `blocked` null.
- The 60 cm sets stay stopped by nominal_to_v0.4 (from the old record).
- My numbers match the builder's.

**For the lead (readings, not defects)**
- **Duty readers in the first stride.** The duty is mixed in the start's schedule, so L5's bob trough (at touchdowns) and L6's roll phase (per frame) read the mixed duty there. Both are scaled by amp. The alternative is to mix only each swing's touchdown.
- **How the lift law is mixed.** The code mixes the lift_peak parameter. The alternative is to blend the two profiles. Both give Stage 1's law at amp 0 and the look walk's at amp 1.
- **Only two swings are mixed.** The first (amp 0.058) is about 94% Stage 1's, not exactly "the stand's". The knee peak has moved to the first swing at amp 1.
- **`start_v0.4`** is mixed against Stage 1's start into nominal.
- **The turn in place's start is not mixed.** The check measured its knee at 1.107 and 1.166 of the limit, but it is not available on its thigh twist and hip-yaw margin anyway.
- **Turn reasons per direction** (ruling 26) apply to the look sets only.
- **Extra commit.** 403987c6 and `look.start_mix` go beyond the three items; both are harmless.

**What to bring in:** the four cherry-picks above. Nothing else.

**What I did not check**
- The V6/60-N-T-10-X start.
- nominal_to_v0.4 on V6noF5/75 and on -X (the builder's claim only).
- `moves legacy` and `regress check`.
- The STAGE2_SLOW tests and the other test files.
- How large the duty-reader effect is on the bob and roll arrays.
- The RuntimeWarning at moves.py:966 on -X builds (in moves.py, not this package's file).

**Side effect to note:** my cherry-pick simulation used `git commit-tree`. That left unreachable commit objects in the repository's object store; no ref or branch moved.

All files are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/lstart/check/`:
- probe.json, probe_start.py, drive.log, revert.log
- verify_new_*.log, verify_wb_*.log
- tests_three2.log, tests_lookbar.log
- sets_root/ (the scratch builds)
