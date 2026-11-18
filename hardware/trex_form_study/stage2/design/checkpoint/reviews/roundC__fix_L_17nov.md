WP L fixer report. Worktree /home/user/mesozoic-wt/wp-l, branch wt/wp-l, head 628d60b7. Everything was simulation only. Nothing was frozen, pushed or rewritten, and no episode, fit or training run was made. The only arrays I built were the 60 cm tail-phase check, by prescribed motion, written to scratch.

**Merges.** All three went in without conflict, and the working branch touches no file of this package.
- 984bb841: working branch at 05a428ec.
- 04dd8adf: working branch at 3d261a79, which carries the rules draft's sixth list.
- 628d60b7: working branch at 6a6409f8. This brings **rulings 21–26, which rule on this review**, and they are applied below.

**Findings**
1. **Major; for the lead, now ruled (21).** I confirmed it on the code and the records:
   - r5lib.py:77-80 counts `TURN_LEAD` 2.0/2.5 and `STOP_LEAD` 2.5 "in windows of the walk's file".
   - control.py:979 makes a periodic segment last `strides * ref.period`, where the period is the whole window.
   - On V6noF5/75-M-H-10, `look.window_a`:
     - turn90_0.35_left: a_phase 2.0, a_time_s 1.7153, a_windows 0.1661, command_phase 0.0.
     - turn90_0.35_right: a_phase 2.5, command_phase 0.5.
     - stop2_v0.4: command_phase −0.5.
     - v0.4 has K = 12 and W = 10.327 s. `bar.check_phase` (lead mod 1) lets all three pass.
   - `bind_move` makes the start and the changes leave window a and enter window b at its boundary, so only the causal moves (turns, stop2) are affected.
   - I did not re-measure the joint jumps.
   - Ruling 21 orders a separate package (r5lib plus lookbar). Nothing was changed here.
2. **For the lead, now ruled (22).** The not-available turn in place goes into the ruling-18 rebuild. Nothing was changed here.
3. **Defect confirmed (working-branch code, sets.py:1821-1824 `merge`); ruled (23) "in the rebuild".**
   - Demo on a scratch copy of V6/60-N-T-10: I merged nominal_to_v0.4's group again with no `blocks_set`.
   - `blocked` became None while `start` was still not available, and verify's "the set is stopped" failure disappeared.
   - So a later passing nominal_to_v0.4 alone would unblock a set whose start fails.
   - Not changed here: a list changes set.json's shape, and tests/test_stage2_sets.py:713, 716, 2242 and 2250 (not this package's) assert the dict.
4. **Ruled (24):** accepted as built.
5. **Fixed in 108ae27a.**
   - `join_events` now uses `moves.WARMUP_ZMP_LIMIT_MM` (4.0 mm), and a new helper `_near_events` pads instead of wrapping.
   - New test: `test_join_events_reads_join_commands_limit_and_no_event_across_the_warmups_ends`. It failed without the fix on each half: "assert 2.0 == 4.0" on the old code, and "assert (3 == 2)" with only the limit fixed.
   - The committed records still hold the old readings (limit_mm 2.0); they change when the moves are rebuilt.
6. **Not changed.** The text is working-branch (state A) code: abd8d8b3 comments "the same words for both directions". Ruling 26 orders per-direction reasons in the rebuild. Editing sets.py now would move its hash in every set's provenance, V0, F25F9 and F25F9cl included.
7. **Checked.** I recomputed the 60 cm phases on the clean tree into scratch. All four are bit-equal to the committed values, with dirty False and an equal check block:
   - v0.3: −1.7366026057343578
   - v0.35: −1.7715091907742444
   - v0.45 and v0.5: −1.7802358370342162

   The reviewer already did 75 cm. The committed file's provenance still says dirty; ruling 26 rebuilds it on a clean tree before any freeze. Not done here.
8. **Fixed in 97e54ea0.** The fixture now writes `built.json` to pytest's tmp directory. Check: after three test runs the study's ignored `logs/test_stage2_moves_look_built.json` keeps its time of 05:13:58 UTC, which is the review's run.
9. **Ruled (25):** accepted; no change.
10. **Diagnosed on arrays (a reading, not a record).** I used the reviewer's regenerated files, whose sha256 equal the records' `built_sha256` (f70a60f9…, ad41818a…).
    - V6/60 nominal_to_v0.4 (ramp 1):
      - Swings before the instant lift with foot pitch about 0.30 rad, at knee ratios 0.66–0.74.
      - The swings that lift from L10's flat stances, at phase 0.5 (R) and 1.0 (L), read 1.0958 and 1.0674.
      - From phase 1.5 toe-off returns and the ratio is about 0.88, which equals the v0.4 walk's 0.8845.
    - The start's flat swings at 0.0, 0.5 and 1.0 read 1.2340, 1.0764 and 0.9770.
    - So the change's peak is the flat-foot lift L10 makes, not a code fault. It goes to the owner after the rebuild (ruling 18).

**Not done here.** Ruling 18 (the start mixes the stand's law) and the rebuild of all six sets, which also carries 22, 23 and the rest of 26, belong to the next step. The builder estimated about 45–60 min per set on one core; I did not measure that.

**Outputs**
- Tests, `tests/test_stage2_moves_look.py tests/test_stage2_moves.py tests/test_stage2_sets.py` on 628d60b7: **"244 passed, 3 skipped in 329.63s (0:05:29)"**. Earlier runs gave "244 passed, 3 skipped in 329.29s (0:05:29)" on 97e54ea0 and "... in 331.20s (0:05:31)" on 04dd8adf.
- Record-key hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`.
- Verify, after the code change; the counts are unchanged from the builder's:
  - V6/75: "12 references, 10 not available, 1 failures"
  - V6noF5/75: "16 references, 6 not available, 1 failures"
  - V6/60 and V6noF5/60: "5 references, 3 not available, 15 failures"
  - V6/75-X: "7 references, 1 not available, 15 failures"
  - V6/60-X: "6 references, 2 not available, 15 failures"
  - New notes, not failures: "built by other code … moves_look.py moved since" (22 on V6/75). Not-available entries are therefore no longer current, so the next `sets build` rebuilds them.
- `moves legacy --body 60-N-T-10 --name start --name stop`: "legacy check: 2 of 2 equal".
- `regress check`: "stage2/regress/regress.json: current (full tier, 256 of 256 episode runs matched, 248 distinct committed episodes, the four source hashes are today's)".

**Tracked and ignored files**
- Tracked: moves_look.py, sets.py, tests/test_stage2_moves_look.py, V6/tail_phase.json, and SCREEN.md, refs.json, screen.json, set.json and transitions.json for each of the six set/body pairs (31 files under stage2/sets).
- Ignored by stage2/sets/.gitignore: 85 `.npz` (line 1), 12 log files (line 6), empty `partial/` directories (line 4); no `.lock` is present.
- The study's `logs/` is ignored by .gitignore:34.
- My outputs are outside the repo, in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wpl_fix/ (tail60.log, tail_phase_60.json, blocked_demo.py, blocked_root/, diag60.py, verify_*.log, legacy.log, regress.log, tests1.log, tests2.log, tests3.log).

**`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:** 34 files changed, 123175 insertions(+), 40 deletions(-), all package files:
- designlab/stage2/moves_look.py +1286
- designlab/stage2/sets.py 631
- tests/test_stage2_moves_look.py +1147
- stage2/sets/V6/tail_phase.json +854
- the five records × six set/body pairs

**`git status --porcelain`:** empty.

**Commits this round, oldest first:** 984bb841, 108ae27a, 97e54ea0, 04dd8adf, 628d60b7. Head: 628d60b7.

The earlier builder's merges 7f96371b, 5436c8da and 7fef4d62 have no attribution lines. I left them as they are; the lead can cherry-pick as the review proposes.

**Bearing on training.** Before R5 (V6) or R2L (V6noF5), these still have to happen:
1. The ruling-18 rebuild, 75 cm first.
2. The owner's answer on any start that still fails and on the 60 cm change.
3. The join-code review (ruling 19).
4. The ruling-21 package, after lookbar's look-set rule (ruling 16).
5. G9-ref, then the lead's freeze.