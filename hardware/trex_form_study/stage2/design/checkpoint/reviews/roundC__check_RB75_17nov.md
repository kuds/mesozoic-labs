Verdict: merge

VERDICT: merge, by cherry-picking the five attributed commits listed below. I found no defect in the code or the records. Every number I re-measured matches the records bit for bit. Nothing is frozen, no limit changed, `moves.py` and `bar.py` are untouched, the frozen sets verify as before and the record-key hashes are the registered pair. The findings below are low or informational.

FINDINGS (most serious first)

1. LOW, process (builder's own admission). The V6noF5 and -X builds ran with 2 processes while the 1-minute load went above 2.5, which breaks the load rule. The builds did run one after another: built_by.at ranges are 09:11-09:58, 10:06-10:52 and 11:01-11:54. The records are not affected: my scratch rebuilds below are bit-identical. No fix needed; noted for the lead.

2. LOW, robustness. moves_look.py `preview_grid_record` (about line 1013) and `entry_extra` work out `look.preview_grid` from the plan, not from the grid the controller actually used.
   - Effect: a future caller that builds a look causal move through `moves.generate` directly would still record a 5.2244 ms grid. `verify` cannot tell.
   - Today the only build path is `sets._Job.move`, which calls `moves_look.generate`, and the warm-up test guards it.
   - Proposed (optional): have the `causal_grid` wrapper record the dt it passed, and have `entry_extra` record or assert that value.

3. LOW, tests. Some new tests pass without the change; this is acceptable as stated:
   - `test_a_build_with_two_processes_names_both_stops` passes on the old sets.py. It fills a test gap, which is what ruling 36 asks for. I saw "1 passed" for it.
   - `test_the_round_sets_moves_build_bit_for_bit_as_before` is a guard. Only one V0 move is rebuilt; the separation of the code paths covers the rest.
   - The "refused" half of the mark test is old behaviour kept.
   - The mark tests fail on the old sets.py, but by AttributeError, since `mark` did not exist.

4. INFO, outside the package. On -X, moves.py:966 prints "RuntimeWarning: invalid value encountered in scalar divide" (I saw it in my -X rebuild). The jaw's usable range has a 0 bound, so the reading is 0/0 = nan, and `max()` returns 0.0. The records hold `range_use.jaw` 0.0 and `neck_yaw` 0.0. This is pre-existing; moves.py is not editable here. For the lead.

5. INFO, pre-existing. SCREEN.md's "record re-check equal" reads "no" for exactly the moves with rolling frames:
   - V6/75: start 56 frames, changes 92-122, stop1 28, the right turns 35-42.
   - It reads "yes" for stop2, which has 0 rolling frames.
   - The screen's contact split differs from the builder's rolling-contact split. The column is reported, not gated, and the previous build showed the same pattern for the changes and stop1.

6. INFO, report gap. The builder's report leaves out that the -X rate-1 turns are also over neck_yaw peak (1.1084 left, 1.1073 right). It is recorded in `stage0`; both turns are not available anyway.

MY RE-MEASUREMENTS BESIDE THE RECORDS

I rebuilt the moves into scratch through `bind_move` and `moves_look.generate`, using the set's own committed windows; nothing was written to the set. I also re-read the committed arrays with `setscreen.drive_flags`.

V6/75-M-H-10, all rebuilt with sha256 equal to the record (for the not-available left turn, equal to `numbers.built_sha256`):

| Move | Mine (= record, to every digit) | Gates diff |
|---|---|---|
| start | knee speed 0.8040839513 (left knee, phase 1.05); join_a 0.5093 / 0.2317 mm; join_b 0.2748 / 0.5303 mm | none; pass |
| nominal_to_v0.4 | knee 0.9188351814 (right knee, phase 0.556); join_a 0.0927 / 0.6012; join_b 0.1983 / 1.4052 | none; pass |
| stop2_v0.4 | join_command warm-up ZMP 3.8682044297; join_b 0.1457 / 0.8694; knee 0.7582; grid 5.2244 = whole-plan minimum | none; pass |
| turn90_0.35_left (f 1.0) | join_command 4.621433895847906 mm (CoM 0.0659 mm, joints 3.97e-4 rad; at command CoM 0.0186 mm, joints 5.15e-5 rad); ok False; join_b 0.2079 / 1.6532 | none; not available |
| turn90_0.35_right | join_command 3.8682044297; join_b 0.2196 / 1.1753; knee 0.8545 | none; available |

Preview grid: 5.22440904905741 ms on the causal grid against a whole-plan minimum of 5.13995 ms; command frame 160.

Ruling 28 probe (my own). The left turn at f 0.8 with full correction passes on the causal grid:
- warm-up q and CoM are bit-equal to f 1.0 on all 161 frames;
- zmp_mb is equal on all but the last two frames;
- join_command is 4.621433895847906, identical to f 1.0.

The same turn at f 1.0 on the default grid (`moves.generate`, one solve) reads 5.543442246091401, exactly the review's recorded code value. Its warm-up differs from the causal one by up to 1.87e-4 rad, 0.027 mm CoM and 2.1 mm zmp_mb.

V6noF5/75 turn90_0.35_left, rebuilt: sha equal; join_command 3.206675149 (passes); join_b 0.2097 / 1.7072.

V6/75-M-H-10-X turn90_0.35_left, rebuilt: sha equal to built_sha256; join_command 4.582345204 (fails); join_b 0.2086 / 1.6489.

From the committed arrays (sha256 equal to the records in all 12 checked; governing ratio equal to the record):

| Set | Start | nominal_to_v0.4 | stop2 |
|---|---|---|---|
| V6noF5 | knee 0.8453, right, phase 2.55 | 0.9192 | 0.8281 |
| -X | knee 0.7948, left, phase 1.05; governing neck peak 0.8052 | 0.9044 | 0.7809 |

WHAT I CHECKED AND FOUND RIGHT

Ruling 28 (code):
- `moves_look.causal_grid` replaces `moves.com_path_causal` only for causal moves, inside `moves_look.generate` and `moves_look.kinematic`.
- It restores the function on exit and on error, refuses to nest, and raises if something else replaced the function.
- `_generate_causal` and `_kinematic` look the function up at call time. `k_cmd = pl.out.start`.
- `sets._Job.move` uses it only for look sets; `_produce_inplace` stays non-causal.
- moves.py sha256 is ccee9b2c… and moves.py/bar.py show no diff from f0a064f8 to HEAD.
- With the hook made a no-op by a scratch pytest plugin: "3 failed, 45 deselected in 48.69s", including "AssertionError: (0.8, 'q')".
- In all three sets, every causal entry records grid 5.2244 ms, command frame 160 and the rule `S2L-causal-grid`.
- Stride-rule choices are unchanged (1.0, 1.0, none at 0.8, none at 0.6); readings move only in the 4th decimal.

Ruling 32 (A):
- `sets mark` needs a frozen set, and a G9-ref record of this set and body whose own sha256 is intact, made on the latest freeze.
- The entry must be named in `parts.a.still_available`, with no blocking item and no dependent reference. The record format (`refs/` and `transitions/<name>.npz`, record sha) matches gates2.
- `freeze` drops only an entry marked on the latest freeze and records `dropped` with kind, sha256, the mark and `g9_ref` (path and sha256). Any other drop is refused.
- No round set's records hold a mark or the mark rule.

Ruling 36: `verify` adds the stops that `blocks_in_records` shows and an older `blocked` leaves out. The two-process test exists.

New sets tests against the old sets.py, loaded by a scratch plugin: "18 failed, 1 passed, 210 deselected in 5.97s". The failures include the 2-line verify assertion, the missing `JOIN_REVIEW` / `S2L-join-review`, and the causal-grid rule in the turn definitions.

Records:
- Each `set.json` names the review in `rules` `S2L-join-review` and in `look.join_review` (path and sha256 b3ff4576…).
- I re-hashed the review file: b3ff457673d537f65d8821c1910c27b6bebc7eb27acd43f0d0e0e96d9424212a.
- `look.cut` is null; the cut reason follows ruling 29.
- `cut3_probe.json` holds the review's numbers (4.038 / 4.361; at J1's leave point 4.621 / 3.868 / 3.868).
- `look.joins_failed` matches the moves' gates.
- `freezes` is [] and `blocked` is null in all three sets.
- All 28 entries per set were built with dirty false; their sets.py, moves_look.py and moves.py hashes equal HEAD's.
- Walks are bit-equal to the previous build; stop2, stop1 and the changes keep their arrays.
- `tail_phase.json` (75 cm) differs only in provenance (commit 07347c77, dirty false). The 60 cm entries are still dirty.

Verdicts follow the rules:
- left turns on V6 and -X: join_command fails (4.621 / 4.582 against 4.0), so not available (A11);
- pi4 turns: over leg drives;
- rate-1 turns: over leg drives, plus heading; the left also fails post_window_zmp (5.053 / 5.09 / 5.054 mm);
- nominal_to_v0.5: zmp_error 4.158 / 4.312 / 4.172 mm, so not available (A8 step 3);
- turn in place: thigh twist 0.9114 / 0.9115 (V6 and V6noF5) and 0.9215 / 0.9216 (-X), hip-yaw margin 0.0266 / 0.0235 rad, so not available (ruling 22).

No limit changed: join_command limits are 1.0 mm / 4.0 mm / 1e-3 rad as before. The screen's limits, margins, grips, rules and specs are identical to the previous build. Existing rule texts are unchanged; only S2L-L1L4-start, S2L-causal-grid and S2L-join-review were added.

Could each set be frozen as it stands? No stop, and verify passes (15 / 13 / 0, 17 / 11 / 0, 15 / 13 / 0). Under rulings 29 and 30 the sets wait for:
- the lead's reading of the numbers;
- the owner's answer on the 4.0 mm left turns (V6 and -X);
- a placed refcheck (none exists yet);
- ruling 40's freeze text in sets.py, owed after this merge.
The builder says the same.

Frozen sets and hashes. `sets verify`:
- "V0 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
- "F25F9 75-M-H-10: verify ok: 12 references, 11 not available, 0 failures; freeze 0"
- "F25F9cl 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"

The old sets.py gives the same counts (20/2/0, 12/11/0, 20/2/0). The hash command printed 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1 at the start and at the end.

Look sets:
- "V6 75-M-H-10: verify ok: 15 references, 13 not available, 0 failures"
- "V6noF5 75-M-H-10: verify ok: 17 references, 11 not available, 0 failures"
- "V6 75-M-H-10-X: verify ok: 15 references, 13 not available, 0 failures"
- status: "not frozen; lead-in start" for each.

Arrays, logs and partial/ are git-ignored; the records are not. `git status --porcelain` is empty in the worktree and in the main checkout. Wall times match the builder's logs (TOTAL_WALL_S 3186, 3268, 3672).

TESTS (my runs)
- tests/test_stage2_moves_look.py tests/test_stage2_moves.py tests/test_stage2_sets.py: "264 passed, 3 skipped in 508.28s (0:08:28)". The skips are moves:392, moves:1459 and sets:2399 (STAGE2_SLOW). An earlier run gave exit 0 with the same count, but the repo's addopts suppressed its summary line.
- tests/test_stage2_lookbar.py -k state_a: "2 passed, 48 deselected in 1.51s".

WHAT TO BRING IN
- `git merge-tree --write-tree HEAD claude/mesozoic-trex-stage2-refine-4jffre` (tip 20b9394b) is clean: tree afdf8548.
- Cherry-pick these onto the working branch, in this order: 07347c77, 371f0778, 8876d7dd, 6445fc13, a3ea12c0. Each applies cleanly, and together they give the same tree, afdf8548. All five end with the two attribution lines.
- Do not bring in f0a064f8: it is a merge whose tree equals e4b94077's, already on the working branch. Its message is attributed.
- None of the working branch's newer commits touches this package's files.
- 17 commits not in the working branch lack the two lines. Their content is already in the working branch: b72c61c7, 33652cb7, de96fc06, e0d64e82, f0630220, c9b88198, e2696b2a, a16aba49, e62028df, ddf46509, 01cbe305, 7fef4d62, 1c1c1419, f60fbd3b, 33e9fbdc, 5436c8da, 7f96371b. Of these, 7f96371b has no trailer at all; the others end with another model's lines.

WHAT I DID NOT CHECK
- I did not rebuild the pi4 or rate-1 turns, the turn in place, the changes other than nominal_to_v0.4, stop1, or anything on V6noF5 or -X beyond their left turn and the array reads.
- I did not re-run setscreen or tail-phase.
- I did not re-derive join_command's formula myself; I used `moves.join_command`, which the join-code review checked.
- I did not run the STAGE2_SLOW tests.
- I did not look at the 60 cm sets.
- I did not exercise `sets mark` on a real frozen set (tests only).
- I did not verify the builder's load readings.

My scratch scripts and logs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rebuild/check_rb75/: probe_moves.py, arrays_read.py, probe_V6_75.log, probe2.log, verify.log and tests_main3.log. The pytest temp directories and the probe arrays are deleted.
