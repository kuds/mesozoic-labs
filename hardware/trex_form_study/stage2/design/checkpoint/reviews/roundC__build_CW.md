# build:CW (roundC, wave 4): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule. The package is not merged: its fix round was paused for the usage limit.

WORK PACKAGE CW (rows for the bar runner): REPORT

Worktree /home/user/mesozoic-wt/wpc-cw, branch wt/wpc-cw, tip 5d04b20. Simulation only. No panel, fit or training run was made; no look set was touched; nothing was written in the main checkout; nothing pushed.

1. WHAT I BUILT

/home/user/mesozoic-wt/wpc-cw/hardware/trex_form_study/designlab/stage2/lookbar.py
- LookRow(inner, sidecar=None, name=None): for a bar walk or turn item it lends the built condition metrics.Recorder(legs_every=tune.LEGS_EVERY) as an instance attribute, runs the inner row, takes the attribute back in a finally, returns the inner's (b, out, rec) unchanged. Afterwards it reads the episode's block: lookread.window / series / reading, the reference by lookread.playback over lookread.window_ref (once per built condition), compare; the clock block; on_command_at_stop for a stop item; the window's short air runs against b.air_gap_max_s; the final state's digest; policy_check from inner.provenance(). The block is kept in .last and written as a sidecar before episode returns. A fault in the reading is recorded (fault), never raised. describe/sources/spec/from_spec as the brief says.
  Tests: test_r0l_is_r0_digest_for_digest_and_record_for_record_on_the_s1_view[0|3]; test_on_a_bar_walk_item_the_wrapper_changes_neither_digests_nor_record_nor_trace; test_the_look_block_of_a_walk_item (sidecar read inside the bar's judge); test_the_recorder_is_the_classs_method_again_after_an_episode_that_raised; test_a_fault_in_the_reading_is_recorded_and_never_raised; test_a_spec_round_trip_rebuilds_the_row; test_what_is_relied_on_from_the_bar; three pure tests (clock, on_command_at_stop, air runs).
- R0L registered at import; rows(body, names, sidecar, stage1_run). Tests: test_the_module_registers_r0l_and_names_the_look_sets, test_rows_are_the_bars_names_with_every_inner_row_a_look_row.
- companion_episode (outcome read / fell / unread / error and the four readings), episodes() (the monitor's entry, with open_set reading the set as bar.BarWorker does). Tests: five table tests; test_episodes_on_the_fixture_set; test_episodes_catches_the_bars_unreadable_apart_from_any_other_fault; test_the_monitor_reads_these_episodes (monitor2.baseline on S1, tiny, through the real functions, and baseline_equal on a second run).
- collect / companion (schema s2-look-1, counts by bar.counts, idempotent), run (bar.evaluate with these rows, attribute "none", then collect), CLI. Tests: five on synthetic bar records and sidecars (counts, unread survived and not succeeded in all readings, seven kinds of missing or foreign sidecar refused, non-look set gives counts_look null and the bar's counts, a1_over, not run); test_run_on_the_s1_view_writes_bar_records_sidecars_and_companions; test_run_on_a_fixture_set_and_what_the_bar_refuses_there; test_the_command_line_assigns_mujoco_gl_itself_and_one_thread.

/home/user/mesozoic-wt/wpc-cw/hardware/trex_form_study/designlab/stage2/s2rows.py
- build(): load through s2load.load_checkpoint(expect_run=run.json's); the run's recorded source hashes against today's (refused, or flagged under STAGE2_ALLOW_SOURCE_CHANGE=1); the run's set as it is today by rule K4. S2Policy (prepare before the reset, begin, act through r5core.Core), R5Row(panels.TrackerRow), R6Row(panels.Row) on r5core.R6Tracker, R5G with HeadScript.fixed(episode seed, seconds); configure / $STAGE2_POLICIES; factories return the row inside LookRow; equivalence_check (imports g8a and r5env inside the function).
  Tests: describe keys and no measured float; spec round trip and a changed file refused; other versions give cross_version; foreign run_uuid, perturbed fingerprint, other mode/body/seed refused, also at the command line; changed r5core.py / lookread.py / Stage 1 / panel hashes refused and flagged under the variable; rule K4 (changed file, entry, task, plant inputs; a later freeze accepted, one that changes a used file refused); no r5env or train2 import in a subprocess that builds and runs a row; the row against the env on S1 (cut, k = 5 and 0); a row asked for another set's condition; lookbar.run with a policy row; monitor2.run on the helper's job; a policy with its clock held 10 % slow on a synthetic set gives unread / clock_behind. Slow: whole episodes against the env, the -X body with head neutral and with H1, R6 against R6Env and through the bar's judge, two spawned workers.

tests/test_stage2_lookbar.py, tests/test_stage2_s2rows.py: as above. The helper make_job writes run.json and one checkpoint of a freshly initialised policy; nothing is trained.

2. ACCEPTANCE

Environment as the standing rules give it, python /home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python, every command through bash tools/lowprio.sh, --basetemp under the scratch directory. All at tip 5d04b20 unless said.
- `pytest -p no:cacheprovider tests/test_stage2_lookbar.py tests/test_stage2_s2rows.py`: "44 passed, 11 skipped in 124.54s (0:02:04)", exit 0.
- `STAGE2_SLOW=1 pytest -p no:cacheprovider tests/test_stage2_s2rows.py -k "equivalence or r6"`: "4 passed, 16 deselected in 192.70s (0:03:12)", exit 0. These are the equivalence check at k = 5 on S1 for 75-M-H-10 (six whole stock conditions, a policy with small actions and one whose actions fell the robot), for 75-M-H-10-X (head neutral and R5G), and R6.
- `STAGE2_SLOW=1 pytest -p no:cacheprovider tests/test_stage2_lookbar.py`: "35 passed in 104.31s (0:01:44)", exit 0 (R0L against R0 on S1: six whole stock conditions, digests and records equal).
- `STAGE2_SLOW=1 pytest ... tests/test_stage2_s2rows.py -k spawned`: "1 passed, 19 deselected in 20.49s". The whole file with STAGE2_SLOW=1 at d38d138: "20 passed in 269.09s (0:04:29)".
- An export of d38d138 without the ignored inputs: "20 passed, 35 skipped in 6.04s"; the exported tree's file list was unchanged after the run.
- ruff check on the four files: "All checks passed!".
- Record-key hashes at the tip: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair).

Check on the frozen set V0 of 75-M-H-10 (scratch script /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/cw/v0check.py, set read from the main checkout's stage2/sets, freeze index 0, manifest 34a820e5...; run at d38d138, which differs from the tip only in is_unreadable; no record, traces under scratch only):
- R0L against R0, seed 3042: at 10 ms walk/v0.4 (10539 steps), turn/q90_left (9811), stop/two_step (10037), nudge/forward (7958, fell); at 0 ms walk/v0.4. In all five: digests equal, bar episode records equal, trace sha256 equal.
- Equivalence at k = 5, freshly initialised policy R5/42 (not trained): walk/v0.4 identical over 8683 steps (the row fell at step 8683, the env terminated in env step 869); turn/q90_left identical over 9283 steps (env step 929).
- Seen beside: R0L's stop item on V0 has on_command_at_stop 0.410 m/s against 0.378 needed (met).
- Check episodes on V0: 49 in all. The final run: 24 (R0@10ms and R0L@10ms on the four conditions, R0@0ms and R0L@0ms on walk/v0.4, each run twice, once for digests and once for record and trace: 20; R5/42@10ms and the env on walk/v0.4 and turn/q90_left: 4). An earlier run of the same script: 24. A first attempt that my scratch harness aborted: 1 (R0@10ms walk/v0.4).

Other outputs: pytest temporary directories and scratch jobs under the scratchpad's cw/ directory, outside the repository. The ignored inputs were copied into the worktree with tools/restore_inputs.py copy-from (groups picks-models, picks-arrays, x): "all here with their listed sha256".

3. DIFF AND STATUS

`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 designlab/stage2/lookbar.py 1027 +, designlab/stage2/s2rows.py 646 +, tests/test_stage2_lookbar.py 1043 +, tests/test_stage2_s2rows.py 739 +; "4 files changed, 3455 insertions(+)".
`git status --porcelain`: empty. The working branch moved to fdb31d5 while I built; I merged it (bb28c45) and took in lookread.compare's three new keys.

4. FOR THE LEAD

Disagreements between texts and code:
a. lookbar.run cannot run Track B rows on a look set as bar.py is built. bar._check_rule lets only rows whose controller is "R0" run on any reference set outside V0, F25F9, F25F9cl; "R0L@10ms" is controller "R0L". So `lookbar run --set V6 --rows R0L@10ms` (and R5, R6, R2L) ends with BarError "only R0 rows run on it", with or without --task. Design 2.3 says the rule restricts only the three 75 cm sets. A test pins the as-built behaviour. Not affected: lookbar.episodes (the monitor's path) and S1. What works today: `--rows R0@10ms` through lookbar.rows is the R0 row in the same wrapper, same description as R0L but for the row name. For policy and R2L rows on a look set the choices are a change to bar.py (which moves the bar's source hash) or a runner of lookbar's own; I built neither.
b. A run records no hash of designlab/stage1/policy.py and env.py alone. I compare, for every session of the run, the three Track B files, sources.plant_sources (every designlab/stage1/*.py) and sources.panel_sources. That is stricter than the brief's list.
c. lookbar's sidecar directory look_partial/ is not matched by stage2/tb/.gitignore's `partial/`: ignored under stage2/tb/work/ and logs/, not under stage2/tb/bar/.

Readings I chose (alternatives in brackets):
d. on_command_at_stop: the CoM's displacement from t_c less one commanded stride to t_c along the home heading, over the stride, at least 0.9 x b.speed_cmd (0.378 m/s at the 0.40 m/s walk). [0.9 x the target speed, 0.36; the starting heading.]
e. On a set that is no look set the companion's counts are bar.counts on the bar record's own episodes, so an unread episode is counted there as the bar counts it (an error); "survived and not succeeded" applies on look sets. [U6's count on every set.] succeeded_look is null per episode there.
f. A walk or turn item whose reference cannot be played, or whose reading faults, is an error on a look set; elsewhere the fault is recorded and the bar's reading stands. A fall stays a fall.
g. t_ref_reached is control.run's reference_time_s (the clock at the episode's end), not the last update row; "below 1" is below 1 - 1e-9. R0L's credited survival equals its survival to rounding only (mean factor seen: 0.99999999999997).
h. The sidecar is flat as design 3.5 lists it, with read, reason, fault added and planned_s, state_digest, row_sha256, air_runs, compare, window_ref. The companion adds counts_but_A1_look, the a1_over marking on counts and counts_look, look_failures_by_term, short_air_runs, its own sha256.
i. LookRow keeps a recorder lent from outside (g8a's digest keeper) and makes it hold poses; the block is read from the metrics.Recorder under the wrapper.
j. A plan plays a turn in place when a periodic gait has a yaw rate and no forward speed; the row then names the reference's yaw. equivalence_check refuses such a plan.
k. describe() holds both version sets through s2load.comparable; R6's action block names the span rule, not numbers; a flagged row has one more key, source_hashes.
l. LOOK_SETS is matched by exact name.

For the next packages:
- lookbar.rows raises before any episode: s2load.CheckpointError (not intact, another run's, fingerprint) or s2rows.RowError (no run.json, changed settings, sources, set by K4). These are what the monitor's owed refusals (sixth list, ruling 3) have to catch.
- Companion episodes carry state_digest, start_seed, clock, planned_s; a condition without a reference comes back as not_run.
- bar.evaluate runs a policy row's unread episode again on every run of the same key (the bar holds it as failed to run), and `lookbar run` then exits 4; the companion holds it as unread.
- `python -m designlab.stage2.bar run --row-module designlab.stage2.s2rows` needs MUJOCO_GL=disable; `lookbar run` assigns it itself.
- The K4 check at build reads the set from the study's stage2/sets unless the policies entry names "root". R5G/<seed> falls back on R5/<seed>'s entry.

Not done:
- No other package's test file was run (the brief's rule). monitor2's stand-in is no longer reached, so the trainer's slow tiny-run tests now go through the real rows; I did not run them.
- `lookbar run` through bar.evaluate was run on S1 and on a synthetic set only.
- No 60 cm body, no -J body; R6 and R5G on S1 only; no episode on a look set.

5. COMMITS, OLDEST FIRST

899faec5dfffdd0a3dc1d4c671fb5919c5ab713c lookbar and its tests
3600a655cba2dd9c9d6832b9e472e27742d264c1 s2rows and its tests
1785867e8ff7e2675d452e55bb0d390beaff3c3c the recorder under a gate's wrapper, record hashes, a turn in place by its speed
a3de3654d06689df2934ec88c0b139b07040be43 lookbar's docstring; a test of its command line
bb28c455f6f8575c5977b9555cb369be8ce560c2 merge of the working branch (fdb31d5)
d38d13872241f7447b117c37753da89c706e4ed5 the block keeps the tail's own amplitude
5d04b20d82084959396c78511f9cfc1915507598 is_unreadable reads the last line
Each message ends with the two attribution lines.
