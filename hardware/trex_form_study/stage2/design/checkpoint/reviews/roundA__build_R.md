# build:R (roundA): the agent's final report

WP R is built and committed on `wt/wp-r` (not pushed, not merged): eight files, all three acceptance commands exit 0.

**Commits:** c9d7568, 62a5be1, c971e87, f028eb3, 3faf7c1, 4b5c100, 6bc4dd7.

**Acceptance** (in `/home/user/mesozoic-wt/wp-r/hardware/trex_form_study`):
1. `$PY -m pytest -p no:cacheprovider tests/test_stage2_readings.py tests/test_stage2_s2io.py tests/test_stage2_setio.py`: exit 0, `44 passed in 5.99s`.
2. `$PY -c "import designlab.stage2.readings as r, sys; assert 'mujoco' not in sys.modules"`: exit 0.
3. `git status --porcelain`: exit 0, empty.

**Deviations**
- **4.4 rows 1-2:** the 1e-6 m tolerance is not reachable with linear edge interpolation. Measured spread is 3.58e-6 m; the tests assert the analytic bound 3.77e-6 m. The construction reproduces the reviewer's numbers (0.0423 to 0.0351 H; rows alone 2.37 mm).
- **Step bound:** the 0.42 m/s step reads 0.9316 m/s²; the test asserts the filter's bound 0.9330, not 0.93.
- **`canonical`:** hashes NaN as `"NaN"`, because every start and stop entry holds a stand's NaN period.
- **`Lock`:** also records `boot_id` and `proc_start`, and takes over a reused pid.
- **Unlisted helpers added:** readings `debounce`, `contact_flags`, `air_gap_max`, `toe_clearance_own_swing_ok`, `STEP_OFFSET`, threshold constants; s2io `read_json`, `git_state`, `blas_threads`; setio `planned_swings`.

**3.8 ambiguities, reading chosen**
- **(a)** A reference swing is "in the window" when wholly inside and complete in the trace; it holds a swing that shares an update row. Air gaps count when wholly inside. The table also lists cut runs (`swing_cut`, `air_gap_cut`).
- **(b)** `c(tb)` is interpolated.
- **(c)** Yaw is re-based at row 0; `time_to_within_10deg_s` is the first update from which all later ones are within tolerance.
- **(e)** Forward is the commanded heading. Decision 18 uses trunk yaw at the command, which stops 60 ms earlier on one stock episode. If the body never stops, touchdowns count to the trace's end. Raw deceleration is taken over two strides.
- **`reference_readings`:** `walk` must carry `at_s` or `rate_rad_s` for a turn, since the file does not hold its rate. The raw peak uses the file's own frames: 1.68 m/s² on the prototype stop, against the design's 2.16.
- **`stage1_view`:** hashes all 28 listed files and refuses a missing one.
- **`read_inputs(require_frozen=False)`:** names a freeze only when it covers every file.

**Key numbers** (scratch runs, not records; scripts in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wp_r/`)
- **Prototype arrays:** stop has 2 touchdowns (0.371, 0.801 s), stopped at 1.0327 s = 1.2000 strides, 0.647 m/s²; turn reads 85.17, 80.51, 80.74 deg.
- **Stock `walk/brisk`, R2/42, seed 3042, 0 ms:** own-swing clearance is 0.0433 H where Stage 1's reads 0.0541 H, so the margin to 0.04 H is thin. 23 swings, no gaps, none interrupted.
- **Recorder convention:** feet are within 0.05 mm of the floor at touchdown rows, consistent with `STEP_OFFSET = 0`.
- **`drive_ratios`:** equals `metrics.window` and `control.run`'s `worst_sustained_ratio`.

**Not done:** nothing from the brief. The formal convention check stays with WP B.
