# check:L (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

**Verdict: merge as it stands, by cherry-pick.** Nothing is frozen and nothing should be. No finding in the package's own files gives a wrong number in the records. The minor fixes below can go into the rebuild that ruling 18 already orders. One major finding sits outside the package and must reach the lead before any look-set turn or stop is trained.

Note: the working branch moved while I reviewed. It went from fccebe42 to 6c62f30e at 05:27 UTC (rulings 13–20, the CG merge, the follow-ups). I re-checked everything against 6c62f30e. Rulings 18 (rebuild the start), 19 (join-code review first) and 20 (turn in place stays out) now settle the builder's three "for the lead" questions.

## Findings, most serious first

**1. MAJOR (outside the package: r5lib.py, bar.py; for the lead). The env and the bar place a look set's turns and stop2 after whole windows, not at J1's leave point.**
- Where:
  - r5lib.py:77-80 defines `TURN_LEAD = {"left": 2.0, "right": 2.5}` and `STOP_LEAD = 2.5`, commented "in windows of the walk's file".
  - r5lib.py:155-160 plans `("ref", v0.4, TURN_LEAD[d])`, then the turn.
  - bar.py:813 and :827 do the same.
  - `control.Track.add_periodic` lasts `strides * ref.period`, and `period` is the whole window (`n * step`).
  - bar.check_phase (bar.py:718-728) tests the lead mod 1 as if it counted strides. For an even K it passes a right-foot move placed at a left lift-off.
- What the records say instead: J1 as designed, `look.window_a` with a_phase 2.0/2.5, a_time_s 1.7153/2.1529 s, a_windows 0.1661/0.2085 on V6noF5/75-M-H-10, where v0.4 has K = 12 and W = 10.327 s.
- Measured on the committed arrays (joint jump between the walk where r5lib ends it and the move's first frame):

| Move | r5lib ends the walk at | Built to start at | Largest jump | × walk's own frame-to-frame change |
|---|---|---|---|---|
| turn90_0.35_left | window phase 0 | window phase 2.0 | 0.018 rad (neck_pitch) | 9.4 |
| turn90_0.35_right | window phase 6.007, a left lift-off | window phase 2.5 | 0.421 rad (left_hip_pitch) | 41 (tail_yaw) |
| stop2_v0.4 | window phase 6.007, a left lift-off | window phase 2.5 | 0.421 rad (left_hip_pitch) | 41 (tail_yaw) |

- Effect: G9-ref (f) checks every task's joins (gates2.joins), so this should fail closed, not silently. But no look-set turn or stop can be trained as things stand.
- Fix:
  - Either r5lib (and lookbar's look-set context, ruling 16) play `m + look.window_a.a_windows` windows before a look set's causal move,
  - or, under cut 3, "one window, plus half a stride for a right-foot command", as T1 reads it.
  - The builder's scratch NOTES.md recorded this fact; its report left it out.

**2. MINOR / for the lead (sets.py, `items()` optional group; ruling 20). The turn in place is in no record, and verify passes without it.**
- Ruling 20's premise ("setscreen reads every entry as a walk") holds only for an *available* turn in place. On every built body it is *not available*.
- Test: I copied V6/75-M-H-10 with the builder's not-available turn in place in refs.json into scratch. `setscreen.status` ran and listed it as not_available. `sets verify` printed "V6 75-M-H-10: verify FAILED: 12 references, 16 not available, 1 failures"; the one failure is the stop.
- So L13's numbers could be on record now, per direction (left / right):
  - thigh-twist use 0.9114 / 0.9115 (limit 0.87);
  - start knee 1.1071 / 1.1664;
  - hip-yaw margin 0.0266 rad (limit 0.05).
- Options: one `sets build --only inplace_0.5_left inplace_0.5_right` per 75 cm set, or keep ruling 20 with `absent` giving the reason.

**3. MINOR (sets.py:1821-1822 `merge`, working-branch code). `set.json.blocked` names only the last failing group.**
- On V6/60-N-T-10, V6noF5/60-N-T-10 and V6/60-N-T-10-X, `blocked.by` is nominal_to_v0.4, although `start` fails too (1.2340, 1.2340 and 1.2232).
- `not_available` holds both; verify's "set is stopped" line names one.
- Fix: keep a list.

**4. MINOR / for the lead (scope). The sets.py diff (633 changed lines) goes past the brief's three named parts.**
- It also covers: _Job binding, `_produce_inplace`, verify/status/build of optional items, current/merge/manifest_of, `modules_for`.
- The look sets live in a separate `LOOK_SETS` registry, not in `SETS`, because tests/test_stage2_sets.py:88 pins `SETS` to three.
- sets.py is in no record key, and the other sets are unaffected: V0 "verify ok: 20 references, 2 not available, 0 failures; freeze 0"; F25F9 "12 references, 11 not available, 0 failures"; F25F9cl "20 references, 2 not available, 0 failures".

**5. MINOR (moves_look.py:1245-1272 `join_events`, a reading that gates nothing).**
- It reports `limit_mm` 2.0 (transitions.G3_ZMP_MM) and counts frames over 2 mm. join_command's warm-up limit has been 4.0 mm since A7 was amended.
- `np.roll` wraps, so rolling frames at the end mark the first frames as "near events". The `& (np.arange(...) >= 0)` term is always true.
- Fix: use moves' join limit; pad instead of wrapping.

**6. MINOR. V6 `turn90_pi4_right`'s reason says "also outside join_command"; its own join_command passes (3.6611 mm).** The text comes from the rate group (sets.py `_produce_turns`). The builder flagged it.

**7. LOW. stage2/sets/V6/tail_phase.json was built on a dirty tree** (built_by.dirty true, commit c9b88198, sets.py 98aa1350, not today's d55c8fbd).
- I recomputed the 75 cm phases on today's clean tree into scratch (`make_tail_phases(path=scratch)`): all four are bit-equal (v0.3 and v0.35 −1.7366026057343578; v0.45 and v0.5 −1.7715091907742444).
- 60 cm not recomputed. A clean rebuild before freezing is the lead's call.

**8. LOW. tests/test_stage2_moves_look.py:312 writes `logs/test_stage2_moves_look_built.json` into the study**, not into tmp_path. The path is ignored (.gitignore:34 `logs/`).

**9. For the lead (a reading in the code that the plan does not state).** S2L-L10 also governs F5's toes-up reach (`toe_reach`):
- a swing that lands at or before the instant keeps gait a's reach;
- a later swing has a reach only when both its leaving and its landing stance follow a walk's law.
- L10/L11 say nothing on a swing's reach. The reading is defensible; the alternative is no reach after the instant.

**10. Not diagnosed (hypothesis only).** The 60 cm `nominal_to_v0.4` peaks at the right knee at phase 0.5563, early in the swing that lifts from the flat stance L10 makes over (0, 1]. That is probably the same flat-foot lift as the start. V6noF5's always-flat walks also run higher knee ratios than V6's (60 nominal: 0.9814 against 0.7353). Ruling 18 changes the start only.

## Checked and found right

**Merge readiness.**
- `git merge-tree --write-tree 6c62f30e HEAD` returns rc 0 and one tree line.
- The merged tree differs from 6c62f30e in exactly the 34 package files: moves_look.py, sets.py, tests/test_stage2_moves_look.py, V6/tail_phase.json, and SCREEN.md, refs.json, screen.json, set.json, transitions.json for each of the six set/body pairs.
- moves.py, setio.py, readings.py, bar.py, stage1/ and stage2/look/ are unchanged.
- The record-key hashes are the registered pair on the worktree: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`. The working branch's new commits touch no file of bar_sources (speedfit, bar, readings, setio) and no stage2 module is in panel_sources.

**Tracked and ignored files.**
- Tracked: 31 files under stage2/sets/V6 and V6noF5 (5 records × 6 sets, plus tail_phase.json), and the 3 code files.
- Ignored: 85 .npz by stage2/sets/.gitignore:1, 12 logs by :6, empty partial/ directories by :4. No .lock is present.
- `git status --porcelain` is empty before and after my checks.

**Provenance.** Every entry has built_by.dirty false, accuracy r2, BLAS 1, and today's module hashes (moves_look 202e928e, sets d55c8fbd, moves ccee9b2c, look 306c5efa, setscreen 36110ec2).

**Nothing frozen, no episode.** `freezes` is [] in all six set.json; there is no refcheck.json; the builder's scratch scripts call no episode, eval, env or track.

**Arrays, recomputed by me** (script `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/check_l_17nov/arrays_check.py`, output in `arrays_check.out` beside it).
- Every walk on all six sets:
  - file hashes equal their entries; the two formats have equal q, qd, qdd;
  - 160·K frames, and window_s = K·period;
  - knee speed ratio from qd equals screen.json to 4 decimal places (e.g. V6/75 0.7414–0.8045; V6noF5/60 0.9464–0.9814; 60-N-T-10-X v0.5 0.9954).
- Ladder twins: the V6 look-format nominal and v0.4 files equal the ladder's rungs (75: 139b0b6b, a521b31e; 60: dfc1aeeb, 91b5c126).
- -X bodies: neck_yaw and jaw deviate 0.0 from home in every walk. V6noF5 walks: foot_pitch 0 and rolling false everywhere.
- Every available move on the 75 cm bodies: knee ratio, and final or post-window ZMP error from the arrays, equal the entries. Examples: V6/75 nominal_to_v0.4 0.9188 and 2.5213 mm; stop2_v0.4 0.8694 mm; V6noF5 turn90_0.35_left 1.9845 mm; 75-X nominal_to_v0.4 0.9044 and 2.5298 mm.
- L10 on the arrays: 0 frames with pitch or rolling in the stances that end after the instant and no later than the last ramp's end, in all 17 available moves.

**Stopped sets, rebuilt by prescribed motion in a scratch root.** Each file is bit-identical to the record's built_sha256; the knee speed is read by hand against the limit 8.6917/1.15 = 7.558 rad/s.

| Set / body | Move | sha256 | Knee peak | Ratio |
|---|---|---|---|---|
| V6 / 60-N-T-10 | start | ad41818a… | 9.3268 rad/s, left knee, phase 0.0375 | 1.2340 |
| V6 / 60-N-T-10 | nominal_to_v0.4 (1-stride ramp) | f70a60f9… | 8.2822 rad/s, phase 0.5563 | 1.0958 |
| V6 / 75-M-H-10 | start | f02e0b17… | 8.5113 rad/s, phase 0.0375 | 1.1261 |

- Every entry gate passes (ik, join_a, join_b, zmp_error, rolling_split); only Stage 0 blocks. By A12 then A8, the stops are right:
  - all six sets are stopped by `start`;
  - the 60 cm sets of both heads are also stopped by nominal_to_v0.4 (75-X's nominal_to_v0.4 passes at 0.9044).
- This confirms the builder's diagnosis of the 75 cm peak (left knee, first swing, phase 0.0375).

**"Not available" entries.** Every one holds its reason and numbers, and they match the report: nominal_to_v0.5 zmp 4.158/4.045 mm (V6) and 4.312/4.173 mm (V6noF5); V6/75 turns 0.35/0.5 join_command 5.5434/5.6951 mm; pi4 hip_yaw 1.1444/1.1455 and knee 1.0384/1.0386; turn 1 hip_yaw 1.6693/1.6710; v0.5 dropped at 1.0163 (V6/60) and 1.0406 (V6noF5/60).

**Verify lines** (identical to the builder's):
- V6/75: "12 references, 10 not available, 1 failures"
- V6noF5/75: "16 references, 6 not available, 1 failures"
- V6/60 and V6noF5/60: "5 references, 3 not available, 15 failures" (14 "not built" plus "the set is stopped")
- V6/75-X: "7 references, 1 not available, 15 failures"
- V6/60-X: "6 references, 2 not available, 15 failures"

**Other commands.**
- `moves legacy --body 60-N-T-10 --name start --name stop`: "legacy check: 2 of 2 equal"
- `regress check`: "stage2/regress/regress.json: current (full tier, 256 of 256 episode runs matched, 248 distinct committed episodes, the four source hashes are today's)"

## Tests

`tests/test_stage2_moves_look.py tests/test_stage2_moves.py tests/test_stage2_sets.py`: **"243 passed, 3 skipped in 325.81s (0:05:25)"**.

The skips are the STAGE2_SLOW builds at test_stage2_moves.py:392, test_stage2_moves.py:1459 and test_stage2_sets.py:2085. My first run printed no summary line because the root pyproject's addopts `-q` doubled my `-q`.

## Commits to bring in

- Commits on wt/wp-l not in the working branch that do not end with the two attribution lines: **7f96371b** (Merge wt/wp-m), **5436c8da** (Merge wt/wp-r), **7fef4d62** (Merge wt/wp-c). All three are the earlier builder's merges.
- The earlier builder's other commits end with the older model's two lines; this round's six end with the required lines.
- So: no merge of the tip. Cherry-pick these 17 commits in this order:

  33e9fbdc f60fbd3b 1c1c1419 01cbe305 ddf46509 e62028df a16aba49 e2696b2a f0630220 e0d64e82 de96fc06 33652cb7 b72c61c7 2a5466ab ba439717 a9ea7f58 a024cbe8

- Simulated in a scratch bare clone onto 6c62f30e: all apply without conflict, and the result's tree equals a merge of the tip.
- The moves' records will be rebuilt under ruling 18; the walks can stay.

## Bearing on training

No look set can be frozen now, so a run on V6 (R5) or V6noF5 (R2L) waits for, at least:
1. Ruling 18's rebuild of the moves of all six sets, 75 cm first. The builder's estimate is about 45–60 min per set on one core.
2. The owner's question on any start that still fails, and on the 60 cm change.
3. The join-code review (ruling 19) for the V6/75 turns.
4. Finding 1's fix in r5lib and lookbar (and ruling 16's look-set rule) before G9-ref can pass.
5. G9-ref, then the lead's freeze.

## Not checked

- 60 cm tail phases not recomputed.
- I did not rebuild every stopped move: V6noF5/60 and the -X sets' starts and changes were not regenerated (their records were read, not rebuilt).
- No J1/J2 resampling recomputed independently (join_b, join_command).
- The builder's turn-in-place scratch numbers taken as built, not rebuilt.
- No `bar refcheck`.
- No full test suite beyond the package's three files.
- The builder's note that join_command reads frames that depend on the command frame is not verified (it goes to ruling 19's review).

All my outputs are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/check_l_17nov/` (tests2.log, arrays_check.out, regen.log, tail.log, ip_root/, regen_root/, clone.git/).
