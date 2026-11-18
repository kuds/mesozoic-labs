VERDICT: merge as it stands. The tip to merge is 4ad729c5a8635718b4a4baf6b841ecf0266f4706 on wt/wp-p42. It is a fast-forward: the working branch is still at 84b97aed. `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre HEAD` exits 0 with tree f690bc9f043f168d343b0b5a0f9b98758a09d702. I found no defect. The items below are for the lead.

I changed no tracked file, committed nothing, froze nothing and pushed nothing. No file in the worktree was written during my check: the `git status --ignored` list is unchanged and no file is newer than my first script. The 60 cm builder has not written in wp-l's 75 cm folders (last change there 11:58; it is building V6/60-N-T-10-X now).

**P42 APPLIED AS DECIDED**
- `moves_look.LOOK_WARMUP_ZMP_LIMIT_MM = 5.0`, rule `S2L-P42-warmup` (its text names P42, A7, L12 and "changed after a result").
  - `join_command_look` moves only the warm-up ZMP limit. It re-reads `ok` with the CoM limit (1.0 mm) and the joint limit (1e-3 rad), and refuses a block whose other limits differ.
  - `entry_extra` applies it only where `join_command` is not None, which means a causal move on a look set. It then re-sets the gate and `pass_`.
- `git diff 84b97aed..HEAD` is empty on `moves.py`, `bar.py`, the V0, F25F9 and F25F9cl records, the 60 cm sets, `tail_phase.json` and `cut3_probe.json`.
- The round's sets still read 4.0 mm: `join_command` limits [4.0] and `causal_limits` 4.0, with no P42 rule, on all three.
- All 9 causal entries of each 75 cm look set name the rule in `move`, `look` and `item`, and their `join_command.limits.zmp_mm` is 5.0. The rule text in `transitions.json` equals `ml.RULES`. `causal_limits` reads 5.0 and names the rule.

**Tests that fail without it**
- With a pytest plugin of mine that loads `moves_look.py` and `sets.py` from 84b97aed: "16 failed, 3 passed, 213 deselected in 47.36s". The main P42 test fails on `{'zmp_mm': 4.0} != {'zmp_mm': 5.0}`.
- Mutations, each caught:
  - ok not re-read: "2 failed, 48 deselected in 47.39s"
  - CoM at the command dropped from `ok`: "1 failed, 1 passed, 48 deselected in 46.17s"
  - 5.0 put into the round's `causal_limits`: "1 failed, 14 passed, 217 deselected in 4.34s"

**MY RE-MEASUREMENT**
I rebuilt from the pre-P42 records (84b97aed) plus wp-l's arrays (every sha256 checked against those records first). I used this branch's code: `sets build --only stop2_v0.4 turn90_0.35 turn90_0.5 --procs 1 --root <scratch>`. Wall times were 932 s for V6 and 1022 s for -X.

Every rebuilt entry equals the HEAD record in every key except `built_by`, so my numbers are the records' to every digit. Each npz sha256 equals the record; the newly available left turns equal the old `numbers.built_sha256`. No other entry changed.

| | V6/75 | V6/75-X |
|---|---|---|
| Left 0.35/0.5, warm-up ZMP | 4.621433895847906 mm | 4.582345203754297 mm |
| Left, warm-up CoM | 0.06594 mm | 0.06512 mm |
| Left, warm-up joints | 3.969e-4 rad | 4.143e-4 rad |
| Left, at command (CoM / joints) | 0.01861 mm / 5.146e-5 rad | 0.01853 mm / 5.138e-5 rad |
| Left verdict (was not available) | ok True, pass_ True, Stage 0 within limits, available | same |
| join_b, 0.35 left | 0.2079 / 1.6532 mm | 0.2086 / 1.6489 mm |
| join_b, 0.5 left | 0.2119 / 1.4667 mm | 0.2129 / 1.4627 mm |
| Right turns and stop2, warm-up ZMP | 3.8682044297 mm | 3.8360431958 mm |
| Right turns and stop2, verdict | pass (as before) | pass (as before) |

- `joins_failed` is {} on both.
- In set.json, my scratch differs from HEAD only in `pi4`, rate `1` and `joins_failed`, which I did not rebuild.

**From the records only (not rebuilt by me):**
- **V6noF5:** readings unchanged: left turns 3.2067 mm, right turns and stop 2.7854 mm. The diffs are only the limit, rule and `join_events.limit_mm` fields, and all 23 array files match their records.
- **Quick turns** (pi4 at f 0.8, rate 1 at f 0.6): every reading is unchanged against 84b97aed except the join fields. Values are left / right.
  - V6 pi4: hip_yaw 1.1443/1.1456, knee speed 1.0384/1.0387.
  - V6 rate 1:
    - leg drives: hip_roll 1.0033/1.0049, hip_pitch 1.0063/1.0064, hip_yaw 1.6693/1.6712, knee 1.3413/1.3414
    - heading at the slowest clock 77.21/77.59° (limit 80)
    - the left also has post_window_zmp 5.0527 mm (limit 4.0)
  - -X pi4: hip_yaw 1.1391/1.1404, knee 1.0307/1.0310.
  - -X rate 1:
    - leg drives: hip_roll 1.0065/1.0081, hip_pitch 1.0453/1.0454, hip_yaw 1.6629/1.6648, knee 1.3343/1.3344
    - neck_yaw 1.1084/1.1073
    - heading 77.21/77.59°; left post_window_zmp 5.0543 mm
  - V6noF5 pi4: hip_yaw 1.1443/1.1457, knee 1.0384/1.0387.
  - V6noF5 rate 1:
    - leg drives: hip_roll 1.0033/1.0049, hip_pitch 1.0062/1.0064, hip_yaw 1.6695/1.6714, knee 1.3413/1.3414
    - heading 77.21/77.59°; left post_window_zmp 5.0897 mm
  - All quick turns stay not available on the leg drives. On V6 and -X, `join_command` has only left their failing gates.
- **Screen:** diffs are confined to the two new left-turn blocks, the left-turn `not_available` entries, provenance and sessions. Provenance records dirty true, but the only dirty paths are the set's own set.json and transitions.json.

**FREEZE TEXT (ruling 40)**
- `sets.REFCHECK_HOW` is LT3's suggested line exactly (string compare: True).
- It is used in the six places, which sit at 84b97aed lines 63, 155, 2618, 2802, 2922 and 3187 (the `--refcheck` help): the docstring, the `REFCHECK` comment, verify's note, `refcheck_lines`, freeze's refusal and the help.
- No wording that names only `bar refcheck` is left. The V6 verify note prints the new text.

**FOR THE LEAD**
1. **Freezing needs `--allow-mixed` or a full rebuild.** verify on each 75 cm look set notes "the entries were built from 2 differing source states: a freeze refuses them without --allow-mixed". This follows from the partial rebuild the task asked for.
2. **`join_events` now counts frames against 5.0 mm on look sets.** Ruling 26 says "(4.0 mm)" in so many words. It is a reading, not a gate, and the builder's choice follows ruling 26's principle of using `join_command`'s limit. On V6 and -X no frame is over 5.0. The lead should confirm the choice.
3. **60 cm look sets in wp-l.** wp-l's HEAD records (f2c26b22) have no stop2 or turn entries yet, so none of their entries differs from this branch's item definitions; I compared them. If wp-l builds stop2 or turns on pre-P42 code, those entries would fail verify ("not of the definition made now"). To avoid it, either merge P42 into wp-l first, or rebuild afterwards with `--only stop2_v0.4 turn90_0.35 turn90_0.5 turn90_pi4 turn90_1`. Note also that `merge` sets `causal_limits` to 5.0 on any merge into an unfrozen look set.
4. **`LOOK_RULES["S2L-moves"]`** still says the moves are gated "as every other set's". The P42 rule states the exception.
5. **Builder's report, minor gaps.** It leaves out the -X rate-1 hip_roll (1.0065/1.0081), and V6noF5's rate-1 hip_roll and hip_pitch. The records hold them.
6. The known `RuntimeWarning` at `moves.py:966` appears again on the -X build.

**OUTPUTS (mine)**
- Tests:
  - `tests/test_stage2_moves_look.py tests/test_stage2_moves.py tests/test_stage2_sets.py`: "267 passed, 3 skipped in 485.04s (0:08:05)". The skips are `moves.py:392`, `moves.py:1459` and `sets.py:2448`, all STAGE2_SLOW.
  - `tests/test_stage2_lookbar.py -k state_a`: "2 passed, 58 deselected in 2.17s"
- Verify:
  - "V0 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
  - "F25F9 75-M-H-10: verify ok: 12 references, 11 not available, 0 failures; freeze 0"
  - "F25F9cl 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
  - The three round sets give the same counts as RB75's report.
  - "V6 75-M-H-10: verify ok: 17 references, 11 not available, 0 failures" (was 15/13)
  - "V6noF5 75-M-H-10: verify ok: 17 references, 11 not available, 0 failures"
  - "V6 75-M-H-10-X: verify ok: 17 references, 11 not available, 0 failures" (was 15/13)
- Status:
  - "V6 75-M-H-10: not frozen; lead-in start"
  - "V6noF5 75-M-H-10: not frozen; lead-in start"
  - "V6 75-M-H-10-X: not frozen; lead-in start"
- Hashes: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1 (the registered pair).
- Commits b7c534d3, a0bc4148, 712a13d2, 633c93a7 and 4ad729c5 all end with the two attribution lines, and no other line names a model.
- The worktree's arrays were checked against the HEAD records: 23 of 23 files per set match.

**NOT CHECKED**
- I did not rebuild the pi4 and rate-1 groups or any of V6noF5; for those I compared the records only.
- I did not run the STAGE2_SLOW tests, refcheck, freeze or setscreen.
- I did not run verify on the 60 cm sets.
- I did not check the numbers inside the new screen blocks.

My scripts and logs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/p42/check/ (for example build_V6.log, build_V6X.log, tests_full2.log, verify_*.log, status_*.log and plugins/p42old.py). The scratch root and the pytest temp directories are deleted.
