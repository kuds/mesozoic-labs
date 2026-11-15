# fix:R (roundA): the agent's final report

**WP R after review: every finding checked; the code fixes are committed; the toe-clearance rule (C3) needs the lead's decision before any freeze.**

**Findings**
1. C3 own-swing reading (major): right, and not mine to settle. My own run of four stock walks (seed 3042) reproduces it: 0.0436 / 0.0334 H (nominal, R2/42 / R0) and 0.0433 / 0.0359 H (brisk) against Stage 1's 0.0475-0.0541 H; the own touchdown is a median 3-23 ms after the planned one. Code unchanged; the docstring now says this.
2. Tests (major): right. Added the reviewer's 14 and 5 for the new code; `test_built_by_here` now checks level r1. All 85 wrong-rule changes to `readings.py` (the reviewer's 65, 20 on the new code) now fail the tests; the no-change control passes.
3. 1e-6 m (minor): right, no change. Both rows measure 3.58e-6 m; the test's bound is 3.77e-6 m.
4. Five readings chosen in docstrings (minor): right, for the lead.
5. Heading slack: fixed with `EPS_DEG = 1e-9` in `turn_reading` and the reference's 80 deg check; tested.
6. NaN low point: fixed; `ValueError` where a swing is read; tested.
7. `line_reading`: now returns `window_reached`; tested.
8. float32 times: docstring only. On a live episode, times rebuilt as row x 5 x dt give the reading to 1e-12 m (1e-9 m from float32 lows); float32 times move it 4e-7 m.
9. First frame: the reference's readings count from `t[0]`; tested.
10. `canonical`: right, no code change.
11. `stage2/regress/.gitignore`: right, not my file.

Before and after the fixes, readings are bit-identical on two live stock episodes and on Stage 1's 18 transition files.

**Acceptance**
- `$PY -m pytest -p no:cacheprovider tests/test_stage2_readings.py tests/test_stage2_s2io.py tests/test_stage2_setio.py`: exit 0, "63 passed in 6.42s"
- `$PY -c "import designlab.stage2.readings as r, sys; assert 'mujoco' not in sys.modules"`: exit 0
- `git status --porcelain`: exit 0, empty

**Commit:** `70cd798 Stage 2 round A (WP R): review fixes of the readings; one test per rule clause`

**Public names:** none renamed, no signature changed. New: `readings.EPS_DEG`; key `window_reached` in `line_reading`'s result; `toe_clearance_own_swing` can raise `ValueError`.

**For the lead**
- **C3:** keep it with an expectation sentence, or redefine it. I would redefine: the design's 0.081 H lift leaves 0.25 mm, and stock walks lose 2-9 mm. One measured option (same swings, middle half of the planned duration counted from the own lift-off) gives R0 0.0477 H (nominal) and 0.0517 H (brisk), R2/42 0.0435 and 0.0483 H.
- **New, mine:** a turn's window starts at the inside foot's lift-off, so that foot's first swing is cut whenever it lifts early. If its last swing is cut too, `count_ok` fails on a clean walk (synthetic: 11 of 410 window ends). I suggest not counting a reference swing whose own swing the window cuts.
- **Texts:** PREREG C3 and C7 wording from finding 4; design 4.4 (4e-6 m; row 1 tests a pure delay only) and 2.1 (`canonical`).
- **WP B:** rebuild `t100` from dt; use `line_reading`'s `window_reached`; record which edge gave the minimum (`readings` does not return it); hash through `s2io.sha256_json`.
- **regress owner:** add the `.gitignore`.

Scripts and logs: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wp_r_fix`
