# fix:C (roundA): the agent's final report

WP C after review: head `f697a3a` on `wt/wp-c` (b010c42, ae7477b, f697a3a); not pushed, not merged.

**Findings**
1. Hip roll on turns (major): no change, per C-R1. The gate and constant are untouched. A test pins that 0.0856 rad blocks and 0.1003 does not.
2. Fixed: a file with a swing on its first frame is refused unless `swing_u` there is at most 1e-6; a file with no `swing_u` is refused too. Stage 1's stop cut by 20 frames is refused at 0.3125.
3. Fixed (C-R2): these block, named "not read: ...": no single-support CoP margin, no joint-range margins, no twist use, a double-support split that did not solve, usable ranges that differ from the plant's or were not compared. Applied to walks too.
4. Fixed: `com_forward_lp1hz_peak` unchanged; `..._at_end` and `..._interior` added; SCREEN.md marks it. 13 of Stage 1's 18 peak within 1 s of an end.
5. Fixed (C-R5): `rms` / `cont_ratio` are over time (`np.gradient(t)` weights), with `rms_frames` / `cont_ratio_frames` beside. Frames uniform to 1e-9 get the frame value as both. Walks unchanged: both V0 blocks still equal the committed rungs, and Stage 1's nominal walk priced this way gives the rung's drive numbers exactly. The passive block's sustained values follow the same rule.
6. Fixed: a walk's block has `reference {file, sha256}`, checked on disk; staleness reads both hashes.
7. Fixed: four decimals; a turn's margin column prints `range.tightest_other`, with `tightest_twist` beside its use.
8. Fixed: (a) `verdict.text` is A12's, with the ladder's sentence in `verdict.ladder_text`; (b) a rolling-contact split that did not solve blocks; (c) `A1_GROUPS = ("neck", "tail")`.

**Acceptance at f697a3a**
- `pytest tests/test_stage2_setscreen.py`: exit 0, "38 passed, 4 skipped in 135.77s". Skips: built V0 set, two slow tests, and the moves-builder test.
- screen / look / motion, `-k "not regenerates"`: exit 0, "204 passed, 76 deselected in 371.00s".
- `git diff --stat stage2/look`: exit 0, empty. `git status --porcelain`: exit 0, empty.
- Slow pair (`STAGE2_SLOW=1`): exit 0, "2 passed, 40 deselected in 103.56s".
- The moves-builder test passes against wt/wp-m's `moves.py` (79ef960) in a scratch copy.
- Smoke on Stage 1's 18 transitions, run one commit earlier (before the not-compared item and wording edits): exit 0, 18 blocks, every recheck exact, no split failure, the same 10 blocked by 0.10 rad. Tail cont on `brisk_to_nominal`: 0.4543 over time, 0.4627 by frames. 15–23 s per move.

**Public names changed**
- `drive_flags(src, tau, omega, t)`: `t` is now required.
- `screen_walk(..., reference=None)`; `run` passes the `refs.json` entry.
- `check_swing_window(name, swing, swing_u=None)`.
- `SCHEMA = 2`: a schema-1 `screen.json` is refused.
- New: `frame_weights`, `rms_over_time`, `passive_over_time`, `walk_reference`, `usable_differences`.
- `accel.equals_entry` is replaced by `com_forward_lp1hz_peak_minus_entry`.

**For the lead**
- Key names for S: a walk's `arrays.sha256` compares with `refs.json` `look_sha256`, and `reference.sha256` with `refs.json` `sha256`.
- The screen gates its own 1 Hz reading, which has no history. The builder's reading under amended A7 is a different number, recorded beside it as `com_forward_lp1hz_peak_entry`. Say which one A12 reads; if A7's, `judge_move` should read the entry's value where present.
- My test no longer compares the two acceleration readings, because M's changed under M-R2.
- A move with no single-support frame is now blocked as not read.
- Stage 1's `enter_walk_0.35_left` has 0.0997 rad of hip roll and blocks.

Scratch: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wpc_fix/`; logs in `/home/user/mesozoic-wt/wp-c/hardware/trex_form_study/logs/` (`wpc_fix_acc.summary`).
