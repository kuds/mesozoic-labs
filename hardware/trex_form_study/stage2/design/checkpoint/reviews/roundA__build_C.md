# build:C (roundA): the agent's final report

WP C built and committed on wt/wp-c (head c30b457); not pushed, not merged.

Commits: caee27b (working branch merged in for the committed STAGE2_75_PLAN.md; WP R fast-forwarded), 39cf9bc (screen.py line 516), f8bcd83, 785ab30, c30b457 (/home/user/mesozoic-wt/wp-c/hardware/trex_form_study/designlab/stage2/setscreen.py, /home/user/mesozoic-wt/wp-c/hardware/trex_form_study/tests/test_stage2_setscreen.py).

Acceptance:
1. pytest tests/test_stage2_setscreen.py: exit 0, "28 passed, 4 skipped in 143.04s" (skips: built V0 set; two STAGE2_SLOW; one needing moves.py, which passed against a temporary copy of wt/wp-m's).
2. pytest test_stage2_screen.py test_stage2_look.py test_stage2_motion.py -k "not regenerates": exit 0, "204 passed, 76 deselected in 384.80s".
3. git diff --stat stage2/look: empty, exit 0.
4. git status --porcelain: empty, exit 0.
Own: STAGE2_SLOW=1, -k "temporary_directory or ladders_rung_over": exit 0, "2 passed, 30 deselected in 93.84s".

Numbers:
- start: CoP margin 0.0450668650456169 m, friction ratio 0.34179740852222495; enter_walk_0.50_left: 0.04503563863734639 m, 0.3194. Both equal the record exactly (torque, CoP and friction differences 0.0).
- Temporary-directory V0 nominal and v0.4 hash to the ladder's rungs (e2bf5274..., 4545bad6...); their blocks equal the committed rung blocks.
- Controls: lift 0.03 H reads mid-swing 0.015 H and blocks; omega x 1.3 puts knee speed above 1.2 and blocks; a turn at 0.83 twist passes its rule while Stage 0's 0.10 rad gate is false and unapplied; tail peak 1.2 on a turn gives a1_over, no block.
- Smoke through the CLI: 18 of 18 blocks, every recheck exact (scratch numbers). Within limits: start 0.661, stop 0.660 (neck cont); nominal_to_brisk 0.745, brisk_to_nominal 0.746, brisk_to_fr05 0.938, fr05_to_brisk 0.946 (knee speed); exit_walk_0.35 left and right 0.663. Blocked by the 0.10 rad range rule (governing 0.660-0.664, neck cont): both enter_walk_0.35, the four enter/exit_walk_0.50, the four in-place starts and stops.
- Wall time: 16-40 s per move (mean 20.6 s, shared machine); 8-9 s per 160-frame walk.

Deviations:
- A walk block's verdict is a dict (text holds screen_rung's sentence), as sets verify needs; compare with the ladder through setscreen.ladder_differences. A walk's arrays.sha256 is the look-format file's (look_sha256).
- Any foot in swing on the last frame is refused (stricter than the brief).
- A move's 0.10 rad rule is read on clearance.screen's margins, as a walk's; hip_yaw_range_use is computed from q (the builder's formula), the entry's beside it.
- setscreen never writes an entry's a1_over; drive_flags(src, tau, omega) gives the builder both flags. Added: status, procs=0, sessions, refused.
- Compute about 0.6 core-hours (cap 0.5).

For the lead: Stage 1's 0.5 rad/s turn leaves 0.072 rad of hip roll; a quick turn like it would be blocked by "every other joint 0.10 rad".
