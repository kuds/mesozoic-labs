# fix:M (roundA): the agent's final report

WP M, fix round: every finding is settled under addendum 2 (M-R1 to M-R5), and all acceptance commands pass. Final commit: `aa14c72 Stage 2 moves: the legacy check's record is complete on the reviewed moves.py (36 of 36 Stage 1 transitions equal, both picks)`.

**Mechanics findings**
1. Consistency window (M-R1): fixed. Consistency reads phase < command + 1; `zmp_consistency_all_m` and its history are recorded and ungated; `RULES['S2A-gates']` carries the ruled sentence; no constant moved. The digits reproduce: V0 turn 2.041 / 2.758 mm (all frames 0.002757695402981966), stop 1.622 / 0.530 mm. Both moves pass every gate.
2. Acceleration (M-R2): fixed as ruled. V0 stop reads 0.6226 m/s² low-passed at 0.600 s and 1.9029 raw; the turn reads 0.097 / 1.975.
3. 1.0 rad/s ladder (M-R3): no code change.
4. Touchdowns: counted in the file's frames (0.3711 s, 0.8014 s); scheduled ones are kept as `touchdown_scheduled_s`.
5. Steps: causal moves get `after_command`. Stage 1's construction gets no key (not `None` as proposed), because a new key would break summary equality with Stage 1.

**Bit-identity findings**
1. Slow test restructured: readings first, then one assertion naming failed gates.
2. Record test now checks the panels' source hash and a clean tree. Library versions are not compared, since they belong to the machine.
3. Gate constants pinned; each gate compared with literal limits, on the stop and on a new zero-pass turn.
4. `generate`, `kinematic` and the legacy job put the accuracy level back.
5. Pre-command CoM: option (a), the only one consistent with M-R1's expected digits. Code kept; `RULES['S2A-causal']` states the 0.05 mm bound.
6. The suite writes under pytest's temporary directory (`STAGE2_TEST_LOGS` opts in elsewhere).
7. Each body keeps `checked`, a list of runs ({commit, dirty, dirty_paths, at, names}), not the single dict proposed, so a partial rerun cannot hide a dirty run.

**Acceptance** (no `moves.py` edit after the slow run)
- Fast: exit 0, "35 passed, 2 skipped in 160.28s".
- Slow: exit 0, "3 passed, 34 deselected in 1285.62s".
- Legacy 75-M-H-10: exit 0; 60-N-T-10: exit 0. Last line of each: "legacy check: 18 of 18 equal; record stage2/sets/legacy_check.json".
- `test_stage1.py -k transition`: exit 0, "6 passed, 113 deselected in 20.18s".
- `test_stage2_look.py -k "not regenerates"`: exit 0, "181 passed, 1 skipped, 74 deselected in 351.08s". These two ran separately, because pytest keeps only the last `-k`.
- `git status --porcelain`: empty; only the three owned files differ.

**Public names changed**
- `forward_acceleration(t, com_xy, heading, fs, before=None, after=None, prev=None)`: the third return is now at the frames' own times.
- New: `second_difference`, `walk_forward_acceleration`, `touchdowns`, `ACCEL_HISTORY_S` (3.0 s, my choice).
- Entry: `reference.touchdown_s` is now first stance frames; `zmp_consistency_history_m` is now the first stride.

**Open items for the lead**
1. WP C's `test_the_constants_and_readings_are_the_builders` will fail after merge: it asserts bit equality with the old acceleration method. WP R's `readings._forward_acceleration` uses that old method too. Both should read the entry or call `moves.forward_acceleration` with the history.
2. After merge the record test is red until the legacy check is rerun at state A.
3. WP L's look windows get the move's own stride as filter history (labelled) unless `moves_look` defines `walk_forward_acceleration(ref, phases)`.
4. Design 4.1 still says `array_equal` for the pre-command CoM.

Files:
- /home/user/mesozoic-wt/wp-m/hardware/trex_form_study/designlab/stage2/moves.py
- /home/user/mesozoic-wt/wp-m/hardware/trex_form_study/tests/test_stage2_moves.py
- /home/user/mesozoic-wt/wp-m/hardware/trex_form_study/stage2/sets/legacy_check.json
- Logs: /home/user/mesozoic-wt/wp-m/hardware/trex_form_study/logs (`wpm_fix_*`)
