# build:CK (roundC, wave 3): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

WORK PACKAGE CK (the look reading: designlab/stage2/lookread.py, tests/test_stage2_lookread.py) - builder's report
Worktree /home/user/mesozoic-wt/wpc-ck, branch wt/wpc-ck, tip 01fccec. Simulation only; everything below is kinematic playback or tests, except the episodes on the view S1 named in section 3 (no record). Nothing pushed, no history rewritten, nothing written in the main checkout.

1. THE MERGE OF THE WORKING BRANCH
- `git merge --no-ff --no-commit claude/mesozoic-trex-stage2-refine-4jffre` (tip cd68f2c): "Automatic merge went well; stopped before committing as requested". No conflict (920 paths added, 23 modified, all the working branch's). Committed with a message file as 8fbd47e. After it, `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat` listed only the two CK files.
- At the merge commit, before any change of mine, the earlier builder's test file gave "49 passed in 4.69s" with nothing skipped (bar.py is there, so the two playback tests ran).

2. WHAT WAS THERE, WHAT I CHANGED
The earlier builder's two commits already held the whole frozen API (BANDS, MIN_STEPS, series, reading, compare, playback, window, sources) plus helpers for gate G9-ref (played, degrade, controls, ladder_numbers, against_ladder, LADDER_LIMITS, RULES) and every test the brief lists. I read it against the brief, design 3.5/3.7, rules U1-U6 and Z2, and the 16 Nov rulings; it stands. My changes:

designlab/stage2/lookread.py (commit 0ccabf2)
- compare: a reference with enough steps whose bob, tail amplitude, tail phase (where the tail is read) or toe-off is missing or not finite now raises LookError. Before, a missing tail amplitude or toe-off passed as "not read". Test: test_a_reference_that_lacks_a_value_is_a_definition_fault.
- compare: new key tail_amp_ratio (recorded; no band reads it). Test: test_the_tails_amplitude_is_recorded_and_no_band_reads_it.
- New window_ref(b) -> (ta_ref, tb_ref) | None: the reference times rule U2's window covers at the nominal clock, from the built condition alone (walk: 1.05 x [t_end - 10, t_end], t_end the last update of round(seconds/dt) steps; turn: [ref_s + 1.05 at_s, ref_s + 1.05 (at_s + 3)]; None for nudge, stop, stock, cut walk). It is what playback takes. Tests: test_the_windows_reference_times_are_the_window_on_the_nominal_clock (stubs), test_the_window_and_the_playback_of_real_bar_items (real BarBuilt).
- against_ladder: also takes another playback's reading as comparand (an -X body, V6noF5) and fails a term whose number is missing on either side instead of raising. Tests: the ladder test's new lines; the -X test's new lines.
- Texts only: the module docstring (which round-A modules are imported; U4's readings and U6 are the wrapper's), RULES["bands"] and RULES["reference"].

tests/test_stage2_lookread.py (commits 62751fc, 01fccec)
- The two `pytest.importorskip("designlab.stage2.bar")` are gone: bar is imported plainly.
- New fast tests: the four named above; test_the_window_and_the_playback_of_real_bar_items builds the bar's own BarBuilt for walk/v0.4, turn/q90_left, nudge/forward, stop/two_step of the frozen set F25F9 (75-M-H-10), plays the tracks (bar.reference_recorder, no episode) and checks window, window_ref, and that the playback of just the window reads what the whole trace reads inside it (skips without the sets' arrays).
- New slow test (STAGE2_SLOW=1): test_a_live_recorders_poses_and_loads_are_what_series_takes: one closed-loop episode on the view S1 (Stage 1's own run directory, walk/brisk, row R0, seed 3042, no record), the recorder lent 50 Hz poses on the instance as lookbar will lend them; series' contact, times, tail and touchdowns equal the recorder's own.

3. ACCEPTANCE
Run from /home/user/mesozoic-wt/wpc-ck/hardware/trex_form_study with the standing environment line; PY = /home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python. All at 01fccec.
- `bash tools/lowprio.sh $PY -m pytest -p no:cacheprovider tests/test_stage2_lookread.py` -> "53 passed, 1 skipped in 6.35s", exit 0 (the skip is the slow test).
- `STAGE2_SLOW=1 bash tools/lowprio.sh $PY -m pytest -p no:cacheprovider tests/test_stage2_lookread.py` -> "54 passed in 9.35s", exit 0.
- `git status --porcelain` -> empty.
- Record-key hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered ones; printed after the merge and again at the tip).
- `ruff check --no-cache` on the two files: "All checks passed!".
- Inputs: the worktree already held picks-models, picks-arrays, x and look. I added the frozen sets' arrays with `tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group sets` ("sets 67 67 0 0 agree"); `git check-ignore -v` gives `stage2/sets/.gitignore:1:*.npz` for them. lookread writes no file; tests write under tmp_path only.

The measures of each look-ladder rung (V6) against stage2/look/metrics.json. Each rung written as a reference file in a temporary directory and played through `lookread.playback` (bar._Played, clock 1.05, a 10 s window); the test's own independent kinematic walk gives the same numbers to the digits shown.
body, rung | ladder: bob mm, head deg, tail cycle, toe-off rad | playback: the same | difference | steps, lift-offs
75-M-H-10 nominal | 5.281, 0.5266, -0.1472, 0.3000 | 5.278, 0.5315, -0.1478, 0.2989 | -0.05 %, +0.0049 deg, -0.0006 cyc, -0.0011 rad (-0.4 %) | 18, 19
75-M-H-10 v040 | 7.498, 0.5391, -0.2819, 0.0558 | 7.494, 0.5419, -0.2812, 0.0542 | -0.06 %, +0.0027, +0.0008, -0.0015 (-2.7 %) | 24, 25
75-M-H-10 fr05 | 9.734, 0.5378, -0.2819, 0.0836 | 9.722, 0.5334, -0.2814, 0.0822 | -0.13 %, -0.0044, +0.0005, -0.0014 (-1.7 %) | 23, 24
60-N-T-10 nominal | 4.258, 0.5313, -0.2708, 0.3000 | 4.255, 0.5423, -0.2700, 0.2984 | -0.07 %, +0.0110, +0.0008, -0.0015 (-0.5 %) | 19, 21
60-N-T-10 v040 | 6.927, 0.5457, -0.2833, 0.0315 | 6.917, 0.5458, -0.2829, 0.0299 | -0.15 %, +0.0001, +0.0004, -0.0015 (-4.9 %) | 27, 28
60-N-T-10 fr05 | 7.849, 0.5473, -0.2833, 0.0197 | 7.837, 0.5551, -0.2823, 0.0187 | -0.16 %, +0.0079, +0.0010, -0.0010 (-5.1 %) | 28, 29
All six pass Z2 (b)'s limits as settled (bob 5 %, head 0.1 deg, tail 0.01 cycle, toe-off 0.02 rad and 10 %). The toe-off reads low on every rung, worst -5.1 % against the 10 % limit. On 60-N-T-10 fr05 both values are under 0.02 rad, so the toe-off term is not read there (U3 names the case). The negative controls (bob x 0.5, 3 degree neck sinusoid, tail 0.2 cycle late, flat feet, a cut to 3 steps) each do what Z2 (c) asks in the tests (75-M-H-10 v040 over a 10 s and a 3 s window; 60-N-T-10 v040; on 60-N-T-10 fr05 flat feet cannot fail because the term is not read).

Scratch checks beyond the brief (stdout only, no record, nothing written):
- Closed-loop on the view S1 only: Stage 1's walk/brisk of 75-M-H-10, seed 3042, once with row R2/42 and once with R0 from a scratch script, and the slow test's same R0 episode twice. With R0: 23 steps against the playback's 23, tail phase difference -0.0029 cycle, amplitude ratio 1.007.
- Kinematic only, read only, on the UNFROZEN look sets in the look package's worktree (wt/wp-l at e2696b2; another package's work in progress, so these are no result): for every walk and turn item I could build of V6 (75-M-H-10: 14 items; 60-N-T-10: 4) and V6noF5 (75-M-H-10: 14), the playback of window_ref holds at least 6 steps and 7 lift-offs, and `controls` is ok (flat feet left out on V6noF5). Reference toe-off in the turn windows: 0.038 to 0.041 rad on 75-M-H-10 (the walk: 0.0545), 0.0221 and 0.0224 rad on 60-N-T-10's quick turns; walk/v0.5 on 75-M-H-10: 0.0207 rad.

4. DIFF AND STATUS AT THE END
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 .../trex_form_study/designlab/stage2/lookread.py   | 592 +++++++++++++++
 .../trex_form_study/tests/test_stage2_lookread.py  | 822 +++++++++++++++++++++
 2 files changed, 1414 insertions(+)
`git status --porcelain`: empty.

5. FOR THE LEAD
Not mine to decide, or needs an owner:
a. The earlier builder's two merge commits cc8eec5 ("Merge branch 'wt/wp-r'") and 3182a6e ("Merge branch 'wt/wp-b'") lack the two attribution lines. I did not rewrite them. As with CB (ruling 2 of ROUND_C_RULINGS_16NOV), the package can come in as its five content commits (6d6d38d, 11a23b4, 0ccabf2, 62751fc, 01fccec): they touch only the two CK files, which the working branch does not hold. I did not try the cherry-pick.
b. U4's `on_command_at_stop` is not built here and is in neither CK's nor CW's brief as written on 15 Nov. It needs an owner (it reads the recorder's CoM rows, not the look measures). Rulings 5 and 8 otherwise concern the companion record and the report's tables, which are lookbar's; in lookread they mean only that nudge and stop items have no window (built so).
c. Z2 (b)'s toe-off limit: the brief and design 3.7 say 0.02 rad; the rule as settled on 15 Nov says "0.02 rad and within 10 % of the ladder's number". The code holds the rule's (LADDER_LIMITS toe_off_rel 0.10).
d. A reference that lacks a value: I made it a LookError (U5: a definition fault that Z2 (b) must have caught). Alternatives: pass as "not read" (as first built), or fail the term. It cannot arise from a playback whose feet alternate; it guards a malformed reference.
e. The tail term reads the phase alone. A tail that barely moves has a fitted phase its motion does not determine, and nothing bands the amplitude. I only record tail_amp_ratio.
f. The bob has no "not read" threshold. On sets without the look's bob (measured: F25F9 walk/v0.4 reference bob under 1e-5 m; Stage 1's brisk walk the same) the bob term always fails, and the reference's own head pitch there is 2.02 deg RMS, over the 1.5 deg band, so a perfect playback fails `head` too. Reading 1 does not count there (U4), but every recorded look_ok on V0, F25F9, F25F9cl will be False.
g. U1 is read literally for contact: any air run longer than the 20 ms debounce is a lift-off and a touchdown. Stance chatter in a closed-loop episode would add short "steps" and low-pitch "lift-offs" (lower bob and toe-off means, an extra stride in the tail's phase). The bar's toe-clearance reading separates such air gaps; U1 does not.
h. Near the 0.02 rad threshold (see the scratch numbers): the 60 cm quick turns' reference toe-off is 2 mrad above it, 75-M-H-10 walk/v0.5 0.7 mrad above. The 50 Hz pose grid alone moved the toe-off by 1.0 % and 2.4 % in two turn windows (a whole-trace grid against the window's own).

Readings the earlier builder chose and I kept (each stated in the module's text):
i. Head pitch: the absolute elevation of the head body's x axis (look.arrays' definition), not "about home" as the brief words it; the reading is an RMS about the mean, so home does not enter.
j. Foot pitch: the elevation of the foot body's x axis below the horizontal (equals the look files' foot_pitch_solved to 1e-9 on the ladder and does not depend on heading, which a turn item needs).
k. Only what lies inside the window is read. The tail is fitted over the poses between the window's first and last left lift-off; with fewer than two left lift-offs the robot's tail value is missing and the term fails.
l. Contact comes from readings.contact_flags (the brief), not stage1.metrics (design 2.2's comment); the rule is the same.
m. U2 against design 3.5's wording: a turn item's window is "the 3 s after t_c + at_s" and lies partly inside the turn move's own file (V6 quick turn: the window covers turn90_pi4_left and then v0.4), not "3 s of walking".

What the next packages need (CW lookbar, CG gates2):
- Beyond the frozen six: series takes t0=0.0; window_ref(b); compare returns tail_amp_ratio, steps, lift_offs, reference {values and side of 0.02 rad}, bands; playback returns the reading plus window_ref and clock_rate; RULES (four rule texts for records); NOT_READ_RAD; played, degrade, controls(kinds=...), ladder_numbers, against_ladder(read, ladder, toe_off=True), LADDER_LIMITS, DEGRADED.
- Use window_ref(b) for playback's times in both the gate and the wrapper, cached per condition; it assumes the track's clock starts at 0 with the episode.
- compare raises LookError for a reference fault (too few steps, a missing value), playback for a track bar._Played refuses ("not played: ..."): catch both.
- window(b, rec) does not read the fall. The bar reads no walk window after a fall, so check out["fell"] first (a fall inside the last two updates would otherwise get a window).
- controls: leave flat_feet out of kinds where the reference's toe-off is not read (V6noF5; 60-N-T-10 fr05), else its ok is False by construction.
- series on a live recorder: series(b.p, rec.poses, rec.loads, rec.dt, tune.LEGS_EVERY * control.FEEDBACK_EVERY) (the slow test pins it).

Not done: no other test file and no full suite run; no -J body (its inputs are not restored here); the -X body only as the plain body's walk copied onto the 16-joint model (the test), no -X set played; no look set is frozen, so the acceptance is on the ladder's rungs through a stub track and the real-item test uses F25F9.

6. COMMITS ON THE BRANCH THAT THE WORKING BRANCH DOES NOT HOLD, OLDEST FIRST
cc8eec59b5453ae28805cdce9cadf8605c7009d2 (earlier builder; merge of wt/wp-r; no attribution lines)
3182a6e8ce4478aef9c7fdbb04775b3dfadf76c8 (earlier builder; merge of wt/wp-b; no attribution lines)
6d6d38d251be96a5eb6afb36869dcb74bf363a1e (earlier builder) lookread.py
11a23b438632ecc7cee184b4bd27b1a78a6cec46 (earlier builder) its tests
8fbd47ee4c985cf9f24b27a56136cbe90e7d31f0 merge of the working branch (cd68f2c)
0ccabf27d969d79a8701f592c88f82caad0cd4a4 lookread: a reference without a value, window_ref, tail_amp_ratio
62751fcce4f5b6763035e02f3ea823d14438f624 tests: no skip on bar, real bar items, window_ref, the new behaviours
01fccec6bfbd2c853c018a07be4db7436c4495af tests: the slow live-recorder test

Files: /home/user/mesozoic-wt/wpc-ck/hardware/trex_form_study/designlab/stage2/lookread.py, /home/user/mesozoic-wt/wpc-ck/hardware/trex_form_study/tests/test_stage2_lookread.py. Scratch scripts (not tracked): /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/ck/table.py, /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/ck/item.py, /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/ck/item_root.py, /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/ck/lookset_scan.py, /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/ck/s1_episode.py.
