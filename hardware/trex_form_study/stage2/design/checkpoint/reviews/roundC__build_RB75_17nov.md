WP L builder report: rulings 28, 32 (A) and 36 in the code, then the three 75 cm look sets rebuilt. Worktree /home/user/mesozoic-wt/wp-l, branch wt/wp-l, head a3ea12c0. Everything was prescribed motion in simulation. Nothing was frozen or pushed, no history was rewritten, and no episode, panel, fit or training ran. moves.py and bar.py are not edited.

**Headline.** All three sets build with no stop (`blocked` null), and `sets verify` passes on each. On V6 and V6/-X every left turn fails `join_command` on the causal grid (4.621 mm and 4.582 mm against 4.0 mm), so those turns are not available. On V6noF5 every turn passes `join_command`.

**1. Merge**
- f0a064f8 merges the working branch at e4b94077. No conflicts. It brought ruling 36, the LS build and check reports, and nothing else of this package.
- The working branch has moved on since then (LT2–LT4, servo S2, rulings 37–40). I did not merge it again.
  - None of those commits touches this package's files.
  - `git merge-tree --write-tree HEAD claude/...` is clean.
  - **Ruling 40's last item is not done:** "the look sets' freeze text in sets.py names the placed refcheck" belongs to the next sets.py change after this rebuild is merged.

**2. Code (commit 07347c77)**

**Ruling 28: a path exists without editing moves.py.**
- `moves_look.causal_grid` is a narrow context that restores what it replaces, in the same pattern as lookbar's.
  - It puts `dt = min(diff(t[:k_cmd+1]))` in front of `moves.com_path_causal` when the caller passes no grid. `_generate_causal` and `_kinematic` look the function up by name, so they pick it up.
  - It restores the function on exit and on an error. It refuses to nest, and it raises rather than restore if something else replaced the function meanwhile.
- `moves_look.generate` and `moves_look.kinematic` open the context for causal moves only.
- `sets._Job.move` uses `moves_look.generate` for look sets only; the round's sets still call `moves.generate`. `pick_stride_factor` and the turn in place go through the moves_look versions.
- New rule `S2L-causal-grid` in the causal moves' `rules`; each causal entry records `look.preview_grid` (grid, whole-plan minimum, command frame).
- Tests:
  - `test_a_look_causal_moves_preview_grid_is_its_frames_up_to_the_command`: the left turn at f 1.0, 0.8 and 0.6 has a grid of 5.2244 ms. The whole-plan minimum is 5.14, 4.295 and 3.221 ms. The warm-up CoM and ZMP are bit-identical on the causal grid and differ on the default grid. It also checks restore, refusal and pass-through.
  - `test_a_look_causal_moves_warmup_reads_the_same_whatever_follows_the_command`: the quick left turn at f 1.0, 0.8 and 0.6, built by `moves_look.generate`, has equal warm-up q, CoM and zmp_mb (all but the last two frames). `join_command` reads 4.621 mm at each.
  - `test_the_round_sets_moves_build_bit_for_bit_as_before`: V0's turn90_pi4_right, whose grid would change on the causal grid, rebuilds to its frozen sha256 in a process that has used the context. This is a guard; it passes with and without the change.
  - The stride-pick test asserts each kinematic pass runs on the causal grid.
- Shown failing without the hook (a scratch pytest plugin making the context a no-op): "3 failed, 45 deselected in 48.19s", including "AssertionError: (0.8, 'q')".

**Ruling 30 / L12 (sets.py).**
- `set.json` rules gain `S2L-join-review` (path plus sha256 b3ff4576…).
- `look` gains `join_review`, `cut` (null), `cut_reason` (ruling 29), `cut3_probe` and `joins_failed`. `merge` writes them; `verify` checks them.
- The review's cut-3 numbers live in a record, `stage2/sets/V6/cut3_probe.json`, because sets.py may name no body (its existing test enforces this).

**Ruling 32 (A).**
- New command `sets mark --set --body --name --reason --g9-ref FILE`. It requires:
  - a frozen set;
  - a G9-ref record of this set and body, with its own sha256, made on the latest freeze;
  - the entry named in that record's `parts.a.still_available`;
  - no blocking item in the record, and no other available entry built from this one.
- The marked entry keeps its definition and provenance. It gains available false, the reason, its numbers and `marked` (when, reason, record path and sha256, freeze, the old file and sha256). It names no file, and set.json lists it under `not_available` and in the stride-rule record.
- `freeze` drops a marked entry and records `dropped` with the mark and the record. Any other drop is still refused.
- `verify`, `build` and `current` accept a marked entry. Earlier freezes, and the round sets' rules and records, are not touched: the mark rule is carried only by the marked entry and by the freeze that drops it.
- Tests:
  - `test_a_freeze_drops_an_entry_marked_on_the_g9_ref_record_that_named_it` (mark, then freeze, records both);
  - `test_a_freeze_that_drops_an_entry_nobody_marked_is_refused`, which also covers 8 refused marks;
  - `test_the_g9_ref_schema_is_the_gates_own`;
  - `test_the_round_sets_records_hold_no_mark_and_no_drop[V0|F25F9|F25F9cl]`.

**Ruling 36.**
- `verify` also names a stop that `blocks_in_records` shows and an older `blocked` leaves out.
- Test changes:
  - the old-dict assertion in `test_blocked_names_every_failing_group_and_a_passing_merge_lifts_only_its_own` now expects 2 lines;
  - the V6/60 demo test is updated the same way;
  - new `test_a_build_with_two_processes_names_both_stops` (spawned workers with the stand-in); it passes on the old code too, because it fills a test gap.
- Shown failing on the old sets.py: "7 failed, 1 passed, 173 deselected in 5.35s". These are the verify test, plus AttributeError on `G9_REF_SCHEMA` and `MARK_RULE` in the mark tests.

**3. Rebuild**

Before the sets, `tail-phase --body 75-M-H-10` ran on a clean tree (commit 371f0778, 7.5 min). All four 75 cm values and check blocks are bit-equal to before; only the provenance changed, to dirty false. The 60-N-T-10 record is left as it was, still dirty.

Each set was rebuilt in an empty directory: the walks, all moves, the turn in place, then the screen. The walks are bit-equal to the previous build, and the changes, stop1 and stop2 have the same arrays. One source state, no dirty entry, legacy check ok.

| | V6/75-M-H-10 | V6noF5/75-M-H-10 | V6/75-M-H-10-X |
|---|---|---|---|
| Start | avail; knee 0.8041 left_knee @ phase 1.05; join_a 0.509/0.232, join_b 0.275/0.530 | avail; knee 0.8453 right_knee @ 2.55; join_a 0.509/0.232, join_b 0.263/0.511 | avail; knee 0.7948 left_knee @ 1.05 (governing: neck peak 0.8052); join_a 0.512/0.233, join_b 0.274/0.537 |
| nominal_to_v0.4 | avail; knee 0.9188; join_a 0.093/0.601; join_b 0.198/1.405 | avail; knee 0.9192; join_a 0.092/0.573; join_b 0.191/1.462 | avail; knee 0.9044; join_a 0.093/0.599; join_b 0.196/1.402 |
| Changes to v0.3/v0.35/v0.45 | avail | avail | avail |
| nominal_to_v0.5 | not avail (zmp_error) | not avail (zmp_error) | not avail (zmp_error) |
| stop2: join_command (warm-up ZMP / CoM; CoM, joint at command) | avail; 3.868 / 0.054 mm; 0.0259 mm, 1.87e-4 rad | avail; 2.785 / 0.0394; 0.0251, 1.29e-4 | avail; 3.836 / 0.0536; 0.0259, 1.81e-4 |
| stop2: join_b; consistency / post-window | 0.146/0.869; 0.325 / 0.869 | 0.146/0.869; 0.325 / 0.869 | 0.147/0.872; 0.325 / 0.872 |
| stop1_v0.4 | avail | avail | avail |

nominal_to_v0.5 numbers (limits: zmp_error 4.0 mm, join_a ZMP 2.0 mm):

| Set | ramp 1 | ramp 2 | v0.45_to_v0.5 |
|---|---|---|---|
| V6 | 4.158 | 4.045 | join_a ZMP 2.28, zmp_error 4.152 |
| V6noF5 | 4.312 | 4.173 | join_a ZMP 2.324, zmp_error 4.312 |
| -X | 4.172 | 4.06 | join_a ZMP 2.273, zmp_error 4.166 |

Turns. The causal grid is 5.2244 ms on every set. In every turn `join_command`'s CoM is within 0.066 mm and its joints within 4.2e-4 rad; at the command, CoM is within 0.026 mm and joints within 1.9e-4 rad.

| Set | Warm-up ZMP, left turns | Warm-up ZMP, right turns |
|---|---|---|
| V6 | 4.621 mm, fails | 3.868 mm, passes |
| V6noF5 | 3.207 mm, passes | 2.785 mm, passes |
| -X | 4.582 mm, fails | 3.836 mm, passes |

Turn verdicts and `join_b` (CoM/ZMP mm, every one within its limits):

| Rate | V6 | V6noF5 | -X |
|---|---|---|---|
| 0.35, f 1.0 | Left not available (join_command), join_b 0.208/1.653; right available, 0.220/1.175 | Both available; L 0.210/1.707, R 0.221/1.261 | Left not available (join_command); right available |
| 0.5, f 1.0 | Left not available (join_command), 0.212/1.467; right available, 0.211/1.742 | Both available; L 0.214/1.532, R 0.211/1.786 | Left not available (join_command); right available |
| pi4, f 0.8 | Both not available | Both not available | Both not available |
| 1, f 0.6 | Both not available | Both not available | Both not available |

- **pi4:** hip_yaw peak is 1.144–1.146 on V6 and V6noF5 and 1.139–1.140 on -X; knee speed is about 1.04 on V6 and V6noF5 and about 1.03 on -X. The left turn on V6 and -X also fails `join_command`.
- **Rate 1:** leg drives are over Stage 0 (hip_yaw about 1.67) and `heading_at_slowest_clock` is about 77.2–77.6°. The left turn also fails `post_window_zmp`: 5.05 mm on V6 and -X, 5.09 mm on V6noF5. On V6 and -X it also fails `join_command`. The post_window_zmp failure is new compared with the last build.

Turn in place (both directions periodic, all not available):

| Set | Thigh-twist use (limit 0.87) | Hip-yaw margin (limit 0.05 rad) | Start's knee speed |
|---|---|---|---|
| V6 and V6noF5 | 0.9114 / 0.9115 | 0.0266 rad | 1.1071 / 1.1664 |
| -X | 0.9215 / 0.9216 | 0.0235 rad | 1.1089 / 1.1705 |

| | V6/75 | V6noF5/75 | V6/75-X |
|---|---|---|---|
| verify | 15 references, 13 not available, 0 failures | 17, 11, 0 | 15, 13, 0 |
| blocked | null | null | null |
| Wall time (build + in place + screen) | 3186 s (2699 + 278 + 208) | 3268 s (2738 + 290 + 238) | 3672 s (3157 + 306 + 208) |

Could each set be frozen as it stands? No stop. Verify passes, the legacy check is ok, one source state, clean tree, and there is no refcheck.json. Under rulings 29 and 30 they wait for the lead, and the left-turn 4.0 mm question for V6 and -X goes to the owner.

**4. Outputs**
- Tests, the three files on the final tree: "264 passed, 3 skipped in 487.26s (0:08:07)". The skips are moves:392 and :1459 and sets:2399, all STAGE2_SLOW.
  - The first full run failed `test_the_rules_say_what_the_plan_says` (a body name in sets.py). I fixed that by moving the numbers into cut3_probe.json, and the rerun gave "264 passed, 3 skipped in 508.07s (0:08:28)".
- `tests/test_stage2_lookbar.py -k state_a`: "2 passed, 48 deselected in 1.66s".
- Frozen sets, before and after the rebuild, with identical logs:
  - "V0 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
  - "F25F9 75-M-H-10: verify ok: 12 references, 11 not available, 0 failures; freeze 0"
  - "F25F9cl 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
- Look sets: verify as in the table above. Status: "V6 75-M-H-10: not frozen; lead-in start", "V6noF5 75-M-H-10: not frozen; lead-in start", "V6 75-M-H-10-X: not frozen; lead-in start".
- Hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair).
- `git check-ignore`: the arrays, logs and partial/ are ignored; the records are not.

**5. Diff and status**
- `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`: "65 files changed, 61158 insertions(+), 29859 deletions(-)". That includes the branch's newer commits, which I have not merged.
- Against the merged point e4b94077: "21 files changed, 60605 insertions(+), 23194 deletions(-)", only this package's files:
  - moves_look.py +114
  - sets.py 321
  - the two test files: 221 and 197
  - V6/cut3_probe.json and V6/tail_phase.json
  - the five records of each of the three sets
- `git status --porcelain` is empty.

**6. Commits, oldest first:** f0a064f8, 07347c77, 371f0778, 8876d7dd, 6445fc13, a3ea12c0. Head: a3ea12c0.

**For the lead**
- **Causal grid in the stride rule:** I applied it to the stride rule's kinematic pass too. The alternative is `generate` only. The chosen factors did not change.
- **Rebuild from an empty directory:** this rebuilt the walks too, so every set is one source state. The alternative was deleting only the move entries.
- **cut3_probe.json:** the probe is V6/75-M-H-10's only. I applied "no cut 3" to the whole body: V6noF5 and -X, neither of which was probed.
- **Mark choices, to confirm:** it requires a frozen set; it refuses a blocking item and a walk that other references were built from; a marked entry's screen block is a verify note.
- **60 cm sets:** verify now gives 18 failures (15 in the fix report). Two of the new lines are this change: the start stop line from ruling 36, and the missing review citation. The third looks like ruling 18's start definition, which I inferred and did not measure at the merge point.
- **Load rule:** I launched V6noF5 and -X with 2 processes at loads 0.87 and 1.21. Other agents' jobs then pushed the 1-minute load above 2.5 (up to 4.06). My attempt to stop the V6noF5 build was denied by the permission check, so both ran on with 2 processes.
- **-X builds:** they print `RuntimeWarning` at moves.py:966. moves.py is not this package's file.
- **Commit-message wording:** the record commits say "code of 07347c77". The entries' `built_by.commit` holds the HEAD at build time (371f0778, 8876d7dd, 6445fc13); the code is the same.
- **60 cm tail phases:** still dirty in tail_phase.json.
- **Not done:** ruling 40's sets.py freeze text, the 60 cm sets, `moves legacy` and `regress check`.

Scratch logs and readouts are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rebuild/: build_*.log, readout_*.json, tests_final.log and the verify logs. The pytest temp directories and the old set directories are deleted.
