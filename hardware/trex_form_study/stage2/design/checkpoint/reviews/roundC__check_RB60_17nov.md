Verdict: merge_after_fixes

VERDICT: merge_after_fixes.

The records are right. Every number I re-measured equals the records to every digit. Nothing is frozen, no limit changed, `moves.py` and `bar.py` are untouched, the frozen sets verify as before and the record-key hashes are the registered pair.

Two things stand in the way of merging as it stands:
- A direct merge now conflicts, so the package must come in by cherry-pick (finding 2).
- Bringing in the rebuilt V6/60 records silently turns off the only test of P42's merge path, the path the next 60 cm build will run on these same records (finding 1). Fix that test first, or with the cherry-pick.

The fix is test-only and needs no rebuild.

FINDINGS (most serious first)

1. LOW-MEDIUM, test coverage lost on merge. File: tests/test_stage2_moves_look.py.
   - Affected tests: `test_a_passing_merge_does_not_lift_another_groups_stop_on_a_look_sets_records` (wp-l :1228, skips at :1243) and `test_a_merge_into_a_look_set_built_before_a_rule_records_its_text` (wp-l :1269, skips at :1279; :1293/:1303 in the working branch).
   - Evidence: both read the committed V6/60-N-T-10 records and skip once they are rebuilt. My run of the cherry-picked tree's own test file on its own code (P42) printed:
     "SKIPPED [1] tests/test_stage2_moves_look.py:1267: V6/60-N-T-10's records are no longer the ones the fixer's demo read (rebuilt)"
     "SKIPPED [1] tests/test_stage2_moves_look.py:1303: V6/60-N-T-10's records already state S2L-L1L4-start (rebuilt)"
     "1 passed, 2 skipped, 47 deselected in 1.02s"
   - On the working branch these tests run today. The P42 check's skips were only the three STAGE2_SLOW tests.
   - What is lost: the second test is the only test of a merge into an older look set's records. That merge adds rule texts incl. S2L-P42-warmup, sets `causal_limits` warmup_zmp_mm to 5.0 and gives set.json S2L-join-review, `join_review`, `cut` and `joins_failed`. The 60 cm sets are now the only unfrozen look sets, and their next build will merge into exactly these pre-P42 records.
   - The first test's behaviour is also covered synthetically by tests/test_stage2_sets.py:729.
   - The builder listed this "for the lead" and kept code unchanged so module hashes would not move. But test files are not among the eight `stage2_modules`, so a test change moves no hash.
   - Proposed fix (test-only, one attributed commit in this package's test file): have the second test, and optionally the first, build their "older" records in tmp_path. Either derive them from the current records (drop S2L-L1L4-start, S2L-causal-grid and S2L-P42-warmup from transitions.json `rules`, reset `causal_limits`, drop S2L-join-review and the `look` review fields), or read them from git at a commit on the working branch's first-parent line (e.g. dd431e6c). Then assert as now.

2. LOW, merge route. A direct merge of wt/wp-l into the working branch now conflicts.
   - `git merge-tree --write-tree HEAD claude/mesozoic-trex-stage2-refine-4jffre` exits 1 against tip dd431e6c and also against the current tip 0f5e8ac5.
   - Conflicting paths: moves_look.py, sets.py, test_stage2_moves_look.py, and 12 record files of the three 75 cm look sets.
   - Cause: the working branch holds cherry-picks of the RB75 commits, then P42 (dd431e6c, 17:17), then the 75 cm freezes (8c8430ed) on top.
   - The builder's "merge-tree is clean" was true against c795ce1b (I re-ran it: rc 0). P42 arrived after the builder's last commit (8bc47346, 17:10).
   - Fix: cherry-pick, as listed under WHAT TO BRING IN.

3. LOW, provenance on the working branch's code (for the lead; no action now).
   - The 60 cm entries were built by the pre-P42 code: sets.py 21086c54…, moves_look.py ed66dfd9…. The working branch has sets.py fb7e5c10… and moves_look.py 98deccc7….
   - I ran verify with the cherry-picked tree's code:
     - the counts are unchanged ("V6 60-N-T-10: verify FAILED: 9 references, 11 not available, 12 failures"; V6noF5 7/9/14; -X 8/7/14);
     - every entry gains a note "built by other code (… moves_look.py, sets.py moved since)": 20, 16 and 15 notes.
   - Their transitions.json `causal_limits` still read warmup_zmp_mm 4.0, and `rules` lack S2L-P42-warmup.
   - Their not-available entries are no longer `current`, so the next `sets build` rebuilds them. They are not causal, so no reading should move.
   - The builder's "P42 changes nothing here" holds for the numbers only. When the owner answers on nominal_to_v0.4, build the 60 cm sets in full on the then-current code.

4. LOW, timing-dependent contents of a stopped set (pre-existing `sets.build` behaviour; for the lead, as the builder says).
   - V6/60 holds the tries of nominal_to_v0.35, v0.45 and v0.3 plus three step-2 changes. V6noF5 and -X hold no nominal_to_v0.35 or v0.45.
   - Some not-available reasons read as reachability claims that are not established:
     - V6noF5's nominal_to_v0.3 says "the rung is reached from v0.35 (v0.35_to_v0.3)", but nominal_to_v0.35 was never tried there.
     - V6's nominal_to_v0.35/v0.45 say "reached from v0.4", but nominal_to_v0.4 itself is not available.
   - No verdict is wrong: the set is stopped, and verify fails the items not built.

5. INFO, minor gaps in the builder's report.
   - "Before, it failed at knee 1.2340" is V6 and V6noF5 (1.23402). The old -X start read 1.2232.
   - "V6noF5's nominal swings already read 0.97–0.98": I read 0.969–0.993 (the swing lifting at phase 0.0 reads 0.9926).
   - On V6, ramp 1, the first swing with toe-off back (right, lift 1.5) reads 0.8983; the later ones read 0.876–0.882 ("about 0.88").

6. INFO, my scratch evidence on the moves the records do not hold (not records). On V6/60-N-T-10, causal grid, `moves_look.generate`, the stop and the 0.35 turns pass every gate and Stage 0. On this body nothing but the change into 0.40 m/s blocks them; P42's 5.0 mm is not needed.

   | Move | join_command warm-up ZMP / CoM / joints | join_b CoM / ZMP | Knee | Gates |
   |---|---|---|---|---|
   | stop2_v0.4 | 3.0806 mm / 0.0461 mm / 2.83e-4 rad | 0.1413 / 0.4679 mm | 0.8860 | all pass |
   | turn90_0.35_left, f 1.0 | 2.3760 / 0.0591 / 3.74e-4 | 0.0654 / 0.5014 | 0.9776 | all pass |
   | turn90_0.35_right, f 1.0 | 3.0806 | 0.0592 / 0.5022 | 0.9777 | all pass |

   - The left turn at f 0.8 is over knee speed: 1.1274.
   - Preview grid 4.5683 ms, command frame 160. Whole-plan minimum: 4.5683 ms (stop), 4.5034 ms (f 1.0), 3.7500 ms (f 0.8).

MY RE-MEASUREMENTS BESIDE THE RECORDS

Rebuilt into scratch through `bind_move`, `moves_look.generate`, `moves.summary`, `entry_extra` and `setscreen.drive_flags`, from the set's own committed windows. Nothing was written to the sets.

| Set / move | sha256 vs record | Knee (mine = record) | Peak and swing | join_a CoM / ZMP mm (= record) | join_b CoM / ZMP mm (= record) |
|---|---|---|---|---|---|
| V6/60 start | 32d61184… equal | 0.9216062987 | left_knee, phase 1.05; left swing lifting at 1.0, flat | 0.4083 / 0.2194 | 0.3539 / 0.5226 |
| V6/60 nominal_to_v0.4, 1-stride | f70a60f9… = built_sha256 | 1.0958202331 | right_knee, phase 0.5563; right swing lifting at 0.5, foot pitch 0 | 0.0737 / 0.4752 | 0.2332 / 0.8816 |
| V6/60 nominal_to_v0.4, 2-stride | 2b3f192e… equal | 1.0678268615 | left_knee, phase 1.0562; left swing lifting at 1.0 | 0.0677 / 0.4752 | 0.2260 / 0.8932 |
| -X nominal_to_v0.4, 1-stride | a1f49310… equal | 1.0609307824 | right_knee, 0.5563 | 0.0743 / 0.4703 | 0.2333 / 0.8844 |
| -X nominal_to_v0.4, 2-stride | d203d1ca… equal | 1.0337057269 | left_knee, 1.0562 | 0.0682 / 0.4703 | 0.2261 / 0.8579 |
| V6noF5 nominal_to_v0.4, 1-stride | 5575d797… equal | 1.0963738933 | right_knee, 0.5563 | 0.0729 / 0.4976 | 0.2403 / 0.8864 |

Every gate of these entries passes; Stage 0 lists knee speed alone as over.

Other swings in the changes:
- V6/60, 1-stride: the left swing at 1.0 reads 1.0674. Swings lifting with pitch 0.2991 read 0.7416.
- V6/60, 2-stride: the right swing at 1.5 reads 1.0547.
- -X: the left swing at 1.0 reads 1.0336 (1-stride); the right at 1.5 reads 1.0225 (2-stride). Swings lifting with toe-off 0.0305 read 0.858–0.878.
- V6noF5, 1-stride: the left swing at 1.0 reads 1.0676.
- This reproduces fix_L item 10's diagnosis: the peak is the first flat-foot lift.

From the committed arrays, every file's sha256 equals its record (14/14, 12/12, 14/14). Governing ratios re-read equal the records:
- V6noF5 start: knee 0.9791, right_knee, phase 2.55.
- -X start: knee 0.9025, left_knee, phase 1.05.
- Changes: V6 v0.4_to_v0.35 0.9269, v0.4_to_v0.45 0.9561, v0.35_to_v0.3 0.9265; V6noF5 v0.35_to_v0.3 0.9560; -X nominal_to_v0.3 (2-stride) 0.9913.

Ruling 28 probe (mine, 60 cm): turn90_0.35_left at f 1.0 against f 0.8.
- Warm-up q, CoM, t and phase are bit-equal on all 161 frames.
- zmp_mb is equal on frames 0–158; frames 0–1 are NaN in both. It differs only on frames 159–160, which straddle the command.
- join_command's readings are identical: 2.3760446436 mm, CoM 0.0591354 mm, joints 3.742e-4 rad.

WHAT I CHECKED AND FOUND RIGHT

Code and tree:
- No code change this round: `git diff a3ea12c0 HEAD` is empty on moves_look.py, sets.py, moves.py, bar.py, the two package test files, and the V0/F25F9/F25F9cl and 75 cm records.
- moves.py sha256 ccee9b2c…; moves.py and bar.py have no diff against the working branch tip.
- All six look sets' entries carry one identical set of `stage2_modules`, equal to HEAD's files.
- a61c53d2's tree is 0781f609, equal to `merge-tree a3ea12c0 cab73e8b`.

Provenance:
- Every 60 cm entry has dirty false and a single commit per set (ebf2a85d, 5cff3e9f, f2c26b22). built_by.at on V6/60 is 13:45–14:48, after the second build started (13:40). Nothing came from the killed build.
- tail_phase.json: ebf2a85d changes only the four 60-N-T-10 entries' `built_by`, to dirty false. The values are unchanged (v0.3 −1.7366026, v0.35 −1.7715092, v0.45 and v0.5 −1.7802358). The 75-M-H-10 entries are untouched.

Records against the previous build (a3ea12c0):
- Every walk's sha256 is equal.
- The nominal_to_v0.4 tries' built_sha256 are equal on all three sets.
- The start was not available before (1.2340 / 1.2340 / 1.2232). It is now available.

Records against the rules:
- Each set.json `rules` has S2L-join-review naming the review's path, its sha256 and ruling 29. `look.join_review` holds the same path and sha256; I re-hashed the file: b3ff4576…24212a.
- `look.cut` is null, `cut3_probe` is null, `joins_failed` is {}, `freezes` is [], and `blocked` is [nominal_to_v0.4].
- transitions.json rules equal `ml.RULES` texts. They added only S2L-L1L4-start and S2L-causal-grid; set.json added only S2L-join-review.
- `causal_limits`, g3 limits, method and preview are equal to the previous build and to V6/75's.
- screen.json limits, rules, margins, grips and specs are equal to the previous build. So no limit changed.

Verdicts follow the rules:
- start within gates and Stage 0: available (ruling 18).
- nominal_to_v0.4 over knee speed at both ramps: step (1) only, so "no transition within the gates", and the set stops and goes to the lead (A8, A12). Under ruling 18/26 it goes to the owner.
- v0.5 dropped on V6 (1.0163) and V6noF5 (1.0406); frozen on -X (0.9954).
- -X nominal_to_v0.3 available by step 1 (1-stride try 1.0096, recorded in `fallback.tried`).
- Turn in place, periodic: reference within Stage 0. Thigh twist 0.7988/0.7989, margin 0.0604/0.0603 rad (V6 and V6noF5, identical); -X 0.8127/0.8128, 0.0562. Not available because its own start (L13, unmixed per ruling 36) is over knee speed: 1.1996/1.2566 and -X 1.1944/1.2546 (ruling 22, T2).

Freeze: none of the three sets can be frozen.
- Each is stopped by nominal_to_v0.4, which goes to the owner.
- verify fails on the stop and on the 11 or 13 items not built.
- They have no refcheck.json.
- Ruling 30 needs the lead's reading.

Ignored and tracked files:
- `git check-ignore -v`: the .npz files are ignored by stage2/sets/.gitignore:1, `partial/` by :4 and `logs/` by :6. set.json and SCREEN.md are not ignored; the 15 records are tracked.
- `git status --porcelain` is empty in the worktree and in the main checkout.
- My checks wrote nothing in the worktree: the `--ignored` listing is unchanged and no file under stage2/sets is newer.

Commits: ebf2a85d, 5cff3e9f, f2c26b22, 8bc47346 and a61c53d2 each end with the two attribution lines and name no model elsewhere.

Hash pair, at the start, on the cherry-picked tree and at the end: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1.

Verify with HEAD code, exactly as printed:
- "V0 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
- "F25F9 75-M-H-10: verify ok: 12 references, 11 not available, 0 failures; freeze 0"
- "F25F9cl 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
- "V6 75-M-H-10: verify ok: 15 references, 13 not available, 0 failures"
- "V6noF5 75-M-H-10: verify ok: 17 references, 11 not available, 0 failures"
- "V6 75-M-H-10-X: verify ok: 15 references, 13 not available, 0 failures"
- "V6 60-N-T-10: verify FAILED: 9 references, 11 not available, 12 failures"
- "V6noF5 60-N-T-10: verify FAILED: 7 references, 9 not available, 14 failures"
- "V6 60-N-T-10-X: verify FAILED: 8 references, 7 not available, 14 failures"

The 60 cm failures are the items not built plus "the set is stopped: nominal_to_v0.4 …".

`sets status`: "V6 60-N-T-10: not frozen; missing 11; lead-in start; stopped: nominal_to_v0.4 …". V6noF5 and -X say "missing 13".

On the cherry-picked tree (tip 0f5e8ac5 plus the four commits), run from a scratch study:
- V0, F25F9 and F25F9cl give the same three lines as above.
- The newly frozen look sets: "V6 75-M-H-10: verify ok: 17 references, 11 not available, 0 failures; freeze 0", "V6noF5 75-M-H-10: verify ok: 17 references, 11 not available, 0 failures; freeze 0", "V6 75-M-H-10-X: verify ok: 17 references, 11 not available, 0 failures; freeze 0".
  - Their four left-turn arrays are not in wp-l; I linked them from wp-p42 and verify matched their sha256.
- The 60 cm counts are unchanged.
- tail_phase.json is read by value, and entry `inputs` are walk sha256s, so the provenance-only change cannot touch the frozen sets.

TESTS (my runs, wp-l HEAD 8bc47346)
- tests/test_stage2_moves_look.py tests/test_stage2_moves.py tests/test_stage2_sets.py: "262 passed, 5 skipped in 466.32s (0:07:46)".
  - Skips: moves_look:1243 and :1279 (the rebuilt-records skips of finding 1), moves:392, moves:1459 and sets:2399 (STAGE2_SLOW).
  - The first run's summary was hidden by the repo's addopts `-q`, so I re-ran with `-o addopts=""`.
  - The bit-for-bit test of V0's moves, `test_the_round_sets_moves_build_bit_for_bit_as_before`, is in this run and passed.
- tests/test_stage2_lookbar.py -k state_a: "2 passed, 58 deselected in 1.67s".

WHAT TO BRING IN
- Do not merge wt/wp-l directly (finding 2).
- Cherry-pick onto the working branch, in this order: ebf2a85d, 5cff3e9f, f2c26b22, 8bc47346.
  - Simulated with `merge-tree --merge-base` onto 0f5e8ac5 (still the tip at the end of my check), each applies cleanly.
  - The final tree is c6a994d5. It differs from the tip in exactly the 16 package files: 15 60 cm records and V6/tail_phase.json. Its code and tests equal the tip's.
  - All four end with the two attribution lines.
- Then add the test-only fix of finding 1 as one more attributed commit.
- Do not bring in a61c53d2 (a merge of the working branch into wt/wp-l). The RB75 commits are already on the working branch as cherry-picks.
- Commits not in the working branch that lack the two lines: b72c61c7, 33652cb7, de96fc06, e0d64e82, f0630220, c9b88198, e2696b2a, a16aba49, e62028df, ddf46509, 01cbe305, 7fef4d62, 1c1c1419, f60fbd3b, 33e9fbdc, 5436c8da, 7f96371b.
  - 7fef4d62, 5436c8da and 7f96371b are plain merges with no trailer; the others end with another model's lines.
  - Their content is already in the working branch (RB75 check). Every other commit not in the working branch ends with the two lines.

FOR THE LEAD (choices that are not mine)
- Finding 1's fix: derive older records in tmp_path (my preference), read them from git, or accept the coverage loss.
- Finding 4: leave it, build the rest with `--only`, or make a stopped build drain or refuse deterministically.
- The builder's optional finding-2 patch (record the dt `causal_grid` passed) is still unapplied. Take it in the next look-set code package, or leave it.
- The 60 cm question to the owner (rulings 18/26): nominal_to_v0.4 at 1.0958/1.0678 (V6), 1.0964/1.0679 (V6noF5) and 1.0609/1.0337 (-X). My scratch numbers in finding 6 show the stop and the 0.35 turns would pass on V6/60.

WHAT I DID NOT CHECK
- No rebuild of: V6noF5's or -X's start, the 2-stride ramp on V6noF5, the other changes, or the turn in place; for these I read the committed arrays and records only.
- No rebuild of a stop or turn on V6noF5 or -X, or of the 0.5, pi4 or rate-1 turns.
- No setscreen or tail-phase re-run. No STAGE2_SLOW tests. No refcheck or freeze.
- No full test suite of the cherry-picked tree: only its test_stage2_moves_look.py, three record-reading tests.
- No mutation tests: no package code changed this round.
- I did not check the builder's load-governor logs.
- I created eight unreferenced commit objects with `git commit-tree` while simulating the cherry-picks. No ref points to them and nothing was committed to a branch.

Scratch (scripts and logs; the npz files, scratch studies and pytest temp directories are deleted) is in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rebuild/check_rb60/:
- scripts: probe60.py, arrays60.py, kneeread.py, cmp_warmup.py, table.py, records.py
- probe logs: probe_V6_60_a.log, probe_V6_60_b.log, probe_V6_60X.log, probe_V6noF5_60.log
- verify logs: verify_head_*.log, verify_merged_*.log, verify_picked_*.log
- test logs: tests_main2.log, tests_lookbar.log
