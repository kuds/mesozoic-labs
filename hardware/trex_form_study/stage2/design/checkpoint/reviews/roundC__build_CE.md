# build:CE (roundC): the agent's final report

WP CE report (branch wt/wpc-ce, worktree /home/user/mesozoic-wt/wpc-ce; head 8d8c450; not pushed, not merged into the working branch)

**Commits** (own files only): 49efdcb, 6abf3fd, 507a092, ac9019f, c67f6c5, c0ae77d, 6bb0138, 54dc848, 8d8c450. Merged in: wt/wp-r, wt/wp-b, wt/wp-m (unchanged at the end) and wt/wpc-cp twice (g8a needs `tbenv`).

**Acceptance** (cwd study dir, brief's env)
1. `pytest tests/test_stage2_r5lib.py tests/test_stage2_r5core.py`: exit 0, "42 passed in 18.05s"
2. `pytest tests/test_stage2_r5env.py tests/test_stage2_g8a.py`: exit 0, "29 passed, 1 skipped in 54.82s" (the skip is the slow test; with STAGE2_SLOW=1: "1 passed")
3. `g8a.stock('75-M-H-10','walk/brisk',3042,5,6.0)`: exit 0, prints `True 3000`
4. `r5env.throughput` (exit 0 each, separate processes, load average about 3, so provisional): hard 382 env steps/s, robot_floor_carpet 307; resident memory 262-264 MB per env process; build 2.3-2.5 s
5. `git status --porcelain`: empty; my commits touch only the eight owned files

**Other numbers**
- G8a stock form is also identical on carpet with a fall (row fell at physics step 2232, env ended in env step 224), after a walking push, the in-place turn, the stand and start-stop.
- `bar_form` against `bar.Delayed` R0@0ms and R0@6ms on a synthetic frozen set in tmp: 4 of 4 identical.
- Mid-task start, zero action, 3 frames at k = 0 and 5: 6 of 6 up for 4 s.

**Readings that need the lead's ruling** (rules fixed before the runs)
- W1 (playback scores every penalty 0) conflicts with the literal formulas in three places. I built so that W1 holds:
  - `duty` charges a foot only where the reference's own share over the same stride is inside the band; Stage 1's reading charged its own reference at every start and stop.
  - `progress` is the shortfall against the same mean of the reference's own speed, over rate x v_cmd; W5's literal form charged the playback up to 0.28.
  - `clearance` (a) uses the planned length of the reference's swing in progress, on the reference clock; a per-gait constant charged 0.78 in a speed change.
- W5's `heading` on a real finite turn may still charge a playback, because the reference lags the rectangular pulse. Not testable: no set has a real turn.
- `env.TASKS` has no "stop": the S1 `stop` task is start, 2 strides, stop, hold. The stop task's hold is 4.0 s (T1 gives no length).
- S1 view of -X and -J bodies read from `stage2/x` and `stage2/j`.
- A mid-start's headings are taken about the start less the reference's own turn.
- The gaze train starts at t = 0; training's jaw event is timed like H1's.
- The `range` margin leaves out the jaw's rest end.
- Turn-in-place names are D5's proposals.

**API additions** (optional arguments only): `Library.from_plan(plan, track=None)`, `g8a.bar_form(..., root=)`, `g8a.env_episode(..., action=)`, `r5env.make` accepts a config's `describe()`.

**Not done**
- Nothing run on a real set (rule X1); K-stride windows tested only on a synthetic 2-stride file.
- Throughput on an idle machine.

**Needed**
- Lead: rule on the W1 readings above (W2/W3/W5 wording) before state B.
- Look package: confirm the in-place names.
- CG: `bar_form` on the frozen sets.
