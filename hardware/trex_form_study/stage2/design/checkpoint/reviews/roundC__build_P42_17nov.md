P42 BUILDER REPORT. Worktree /home/user/mesozoic-wt/wp-p42, branch wt/wp-p42. Everything was prescribed motion in simulation. Nothing frozen, nothing pushed, no history rewritten; no episode, panel, fit or training ran. moves.py and bar.py are not edited, and neither are the round's sets or the 60 cm sets (`git diff 84b97aed` on those paths is empty).

**POINT 1. Code: P42 applied from moves_look.py (commit b7c534d3)**
- **moves_look.py:**
  - Adds `LOOK_WARMUP_ZMP_LIMIT_MM = 5.0` and the rule `S2L-P42-warmup`. Its text names P42, A7, L12, "changed after a result", 5 in place of P34's 4 mm, and the round keeping 4 mm.
  - `causal_rule_ids()` now names the rule, so the stop's and turns' item definitions, `move.rules` and `look.rules` carry it.
  - New `join_command_look(jc)`: a new block with the same readings. `limits.zmp_mm` becomes 5.0 and `limits.zmp_rule` names the rule. `ok` is re-read with the other limits at moves' values (CoM 1.0 mm over the warm-up and at the command, joints 1e-3 rad). A block whose other limits differ is refused.
  - `entry_extra` applies it to a look set's causal entry, and the gate `join_command` and `pass_` follow it.
  - `join_events` now counts frames against the same 5.0 mm (see "For the lead").
- **sets.py:** a look set's `transitions.json` `causal_limits` reads `warmup_zmp_mm` 5.0 plus `warmup_zmp_rule` (`look_causal_limits`). This applies in `_new_docs` and on every merge into an unfrozen look set. The round's records keep 4.0.
- **New tests:**
  - `test_p42_a_look_causal_move_whose_warmup_reads_between_4_and_5_mm_passes_and_a_round_move_fails`: V6's quick left turn reads 4.621 mm. `moves.summary` (the round path, 4.0) has gate `join_command` False. The look entry (5.0) has it True, with the same readings and every other limit and gate unchanged.
  - `test_p42_join_command_look_moves_the_warmup_zmp_limit_alone`: blocks made by hand at 3.99 / 4.0 / 4.001 / 4.621 / 5.0 / 5.001 mm, the CoM and joint limits held, refusals.
  - Updated: the rule, items, records, merge and join_events tests.
- **Fails without it:** a scratch pytest plugin loaded moves_look.py and sets.py from 84b97aed. Result: "14 failed, 36 deselected in 46.65s", including both new tests (the entry's `limits` read `{'zmp_mm': 4.0} != {'zmp_mm': 5.0}`, and AttributeError for `join_command_look`).

**POINT 3. Freeze text, ruling 40 (commit a0bc4148)**
- `sets.REFCHECK_HOW` holds LT3's suggested line word for word. It is used in the six places:
  - the module docstring;
  - the comment at `REFCHECK`;
  - verify's note when there is no record;
  - `refcheck_lines` when a record is not current;
  - freeze's refusal ("make it again. ...");
  - the `--refcheck` help.
- No "bar refcheck"-only wording is left.
- Tests: new `test_the_freeze_text_names_the_placed_refcheck_for_a_look_set` (checks the constant against the LT3 check file, the docstring, the comment and the help), plus extended assertions in `test_verify_and_freeze_cite_the_bars_kinematic_check`.
- **Fails without it** (sets.py of b7c534d3): "2 failed, 180 deselected in 0.86s" (AttributeError REFCHECK_HOW).

**POINT 2. Records**
- Each set: `sets build --procs 1 --only stop2_v0.4 turn90_0.35 turn90_0.5 turn90_pi4 turn90_1` (merge is inside build), then `setscreen run --procs 1`, then verify and status. Each set was built on a clean tree and committed separately.
- Every rebuilt entry's arrays are bit-equal to before. The newly available left turns hash to their old `numbers.built_sha256`.
- Record diffs are confined to:
  - `transitions.json`: the 9 causal entries, `rules` (+S2L-P42-warmup) and `causal_limits`;
  - `set.json`: `look.joins_failed`, `not_available`, `stride_rules`;
  - `screen.json`/`SCREEN.md`: the 2 new blocks.
- stop1 is not causal and was not rebuilt.

**V6/75-M-H-10** (commit 712a13d2):
- Changed verdicts:
  - turn90_0.35_left and turn90_0.5_left: not available → available. Warm-up ZMP 4.6214 mm (limit 5.0), warm-up CoM 0.0659 mm, joints 3.97e-4 rad; at the command, CoM 0.0186 mm and joints 5.15e-5 rad. join_b 0.2079/1.6532 and 0.2119/1.4667 mm.
  - turn90_pi4_left: still not available; its reason no longer says "also outside join_command".
  - turn90_1_left: still not available; join_command has left its failing gates.
  - `joins_failed` is now {}.
- Quick turns stay not available on the leg drives:
  - pi4 (f 0.8): hip_yaw peak 1.1443 / 1.1456, knee speed 1.0384 / 1.0387.
  - rate 1 (f 0.6): hip_yaw 1.6693 / 1.6712, knee 1.3413 / 1.3414, hip_roll 1.0033 / 1.0049, hip_pitch 1.0063 / 1.0064; heading_at_slowest_clock 77.21° / 77.59° (limit 80); the left also has post_window_zmp 5.0527 mm (limit 4.0).
- Table: start, nominal_to_v0.4, nominal_to_v0.3, nominal_to_v0.35, nominal_to_v0.45, stop2_v0.4, stop1_v0.4 = A; nominal_to_v0.5 = N; turn 0.35 L/R = A/A; 0.5 L/R = A/A; pi4 L/R = N/N; 1 L/R = N/N; turn in place (3 per side) = N.

**V6noF5/75-M-H-10** (commit 633c93a7):
- No verdict changed. The left turns read 3.2067 mm; the right turns and the stop 2.7854 mm.
- All 9 causal entries were refreshed and name the rule. The screen is unchanged.
- Quick turns stay not available:
  - pi4: hip_yaw 1.1443 / 1.1457, knee 1.0384 / 1.0387.
  - rate 1: hip_yaw 1.6695 / 1.6714, knee 1.3413 / 1.3414, heading 77.21° / 77.59°; the left also has post_window_zmp 5.0897 mm.
- Table: the same as V6's.

**V6/75-M-H-10-X** (commit 4ad729c5):
- Changed verdicts:
  - turn90_0.35_left and turn90_0.5_left: not available → available. Warm-up ZMP 4.5823 mm, CoM 0.0651 mm, joints 4.14e-4 rad; at the command, CoM 0.0185 mm and joints 5.14e-5 rad. join_b 0.2086/1.6489 and 0.2129/1.4627 mm.
  - pi4_left and 1_left: join_command no longer among their failures.
  - `joins_failed` is now {}.
- Quick turns stay not available:
  - pi4: hip_yaw 1.1391 / 1.1404, knee 1.0307 / 1.0310.
  - rate 1: hip_yaw 1.6629 / 1.6648, knee 1.3343 / 1.3344, hip_pitch 1.0453 / 1.0454, neck_yaw 1.1084 / 1.1073, heading 77.21° / 77.59°; the left also has post_window_zmp 5.0543 mm.
- Table: the same as V6's.

Wall times: build 2047 s / 2036 s / 2259 s; screen 41 s / 1 s / 42 s.

**OUTPUTS** (final tree, all inputs restored)
- `tests/test_stage2_moves_look.py tests/test_stage2_moves.py tests/test_stage2_sets.py`: "267 passed, 3 skipped in 507.06s (0:08:27)". The skips are moves.py:392 and :1459 and sets.py:2448, all STAGE2_SLOW.
- An earlier run before the inputs were restored gave "257 passed, 13 skipped in 280.24s (0:04:40)". The extra skips needed the Stage 1 arrays.
- `tests/test_stage2_lookbar.py -k state_a`: "2 passed, 58 deselected in 2.13s".
- Verify after the change:
  - "V0 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
  - "F25F9 75-M-H-10: verify ok: 12 references, 11 not available, 0 failures; freeze 0"
  - "F25F9cl 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
  - These three are identical to before.
  - "V6 75-M-H-10: verify ok: 17 references, 11 not available, 0 failures" (was 15/13)
  - "V6noF5 75-M-H-10: verify ok: 17 references, 11 not available, 0 failures"
  - "V6 75-M-H-10-X: verify ok: 17 references, 11 not available, 0 failures" (was 15/13)
- Status: "V6 75-M-H-10: not frozen; lead-in start", "V6noF5 75-M-H-10: not frozen; lead-in start", "V6 75-M-H-10-X: not frozen; lead-in start".
- Hashes, at the start and the end: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1
- `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`: "14 files changed, 8480 insertions(+), 5591 deletions(-)". The working branch is still 84b97aed, so this is only this package:
  - moves_look.py 78, sets.py 49;
  - the two test files 124 and 55;
  - V6/75 and V6/75-X: 4 records each; V6noF5/75: transitions.json and set.json.
- `git merge-tree` with the working branch is clean. `git status --porcelain` is empty.
- Commits, oldest first: b7c534d3, a0bc4148, 712a13d2, 633c93a7, 4ad729c5. Head: 4ad729c5a8635718b4a4baf6b841ecf0266f4706. All end with the two required lines.

**FOR THE LEAD**
1. **Mixed sources.** Only the stop2 and turn groups were rebuilt, so each 75 cm look set now has entries from 2 source states. verify notes it; a freeze needs `--allow-mixed`, or a full rebuild. nominal_to_v0.5 (not available, not causal) now reads as "built by other code", and a plain `sets build` would rebuild it.
2. **join_events.** It now counts frames against 5.0 mm, following ruling 26's principle that join_events uses join_command's limit; the alternative is to keep it at 4.0. On the V6 left turns no frame is over 5.0.
3. **causal_limits.** A look set's `transitions.json` `causal_limits` now says 5.0 and names the rule. Without this the header would still say 4.0.
4. **60 cm look sets.** P42 applies to them too, but they are being rebuilt in wp-l on pre-P42 code. After both branches merge I expect their stop2 and turn definitions not to match. I did not verify this. The fix would be `sets build --only stop2_v0.4 turn90_0.35 turn90_0.5 turn90_pi4 turn90_1` on each.
5. **S2L-moves text.** LOOK_RULES["S2L-moves"] still says the moves are "gated ... as every other set's". I did not edit it: it is a recorded text, and the P42 rule in transitions.json states the exception.
6. **Inputs copied into this worktree** (git-ignored; copies only; every sha256 checked against the records):
   - the three 75 cm look sets' arrays from wp-l (65 files);
   - the round sets' arrays from the main checkout (67);
   - the Stage 1 arrays of 75-M-H-10 and 60-N-T-10 (56);
   - `tools/restore_inputs.py copy-from` the main checkout for the groups picks-models, x and look.
   - The first V6 build was refused for missing models and wrote nothing.
   - The earlier test run had rebuilt the 75 cm model into the ignored stage1/models/: its XML is bit-identical to round two's and its ledger JSON differs. I deleted it.
7. **Provenance.** The screen sessions of V6 and -X record dirty true: the set's own just-built records were not yet committed, as in RB75. The entries' `built_by.commit` is the HEAD at build time (a0bc4148, 712a13d2, 633c93a7); the code is the same in each.
8. **Pre-existing warning.** The -X builds print RuntimeWarning at moves.py:966, as RB75's check reported.

**NOT DONE:** the 60 cm sets, a full rebuild, refcheck, freeze, STAGE2_SLOW tests.

Scratch logs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/p42/ (rebuild_*.log, tests_final.log, verify_*.log, compare_*.json, plugins/old_code.py). The pytest temp directories are deleted.
