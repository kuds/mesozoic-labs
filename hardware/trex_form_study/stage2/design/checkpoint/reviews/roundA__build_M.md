# build:M (roundA): the agent's final report

WP M report (worktree /home/user/mesozoic-wt/wp-m, branch wt/wp-m; not pushed, not merged)

Commits: 99989f1, ad61b58, d781bb0, 14c1971, 03964c7, efa0005. Files: designlab/stage2/moves.py (sha256 559bff2d…), tests/test_stage2_moves.py, stage2/sets/legacy_check.json.

Acceptance (study dir, via tools/lowprio.sh):
1. fast tests: first run exit 1, "1 failed, 22 passed, 2 skipped in 142.25s" (only test_legacy_check_record_is_current: record not yet made). Re-run after step 4: exit 0, "23 passed, 2 skipped in 147.17s".
2. slow -k: exit 1, "1 failed, 2 passed, 22 deselected in 1282.89s". Legacy full-pass equality passed (5 transitions, sha256 = Stage 1's). test_stop_and_turn_from_v0 FAILS one gate: the turn's zmp_consistency.
3. legacy 75-M-H-10: exit 0, 18 of 18 equal.
4. legacy 60-N-T-10: exit 0, 18 of 18 equal. Record: both complete, dirty false.
5. Run as two commands (pytest keeps only the last -k; the literal line selects all 119 Stage 1 tests): test_stage1.py -k transition: exit 0, "6 passed, 113 deselected in 22.11s"; test_stage2_look.py -k "not regenerate": exit 0, "179 passed, 1 skipped, 76 deselected in 187.29s".
6. git status --porcelain: empty; diff from feb84ad = my three files.

Scratch numbers (V0 0.40, 75-M-H-10; logs/test_stage2_moves_scratch.json):
- Pick 0.80 (x1.0 pinned, 16 IK-fail frames; 0.95 0.983; 0.90 0.928; 0.85 0.873; 0.80 0.818; steady turn 0.805).
- stop2_v0.4: pass_. Consistency 1.62 mm (not under 1), post-window 0.53, transient 9.4; touchdowns 0.367 s, 0.798 s; stopped 1.200 strides; 1 Hz 0.646, raw 2.36 m/s^2; join stand CoM 0.151, ZMP 0.501 mm; warm-up CoM 0.009, ZMP 1.35 mm; CoP margin 44.6 mm.
- turn90_pi4_right: ik_ok, twist 0.815, join CoM 0.105 mm, ZMP 0.726 mm, heading 85.45 deg (bar), 80.81 (slowest), kappa-1 6e-14, CoP margin 23.1 mm, post-window 2.76 mm (limit 4.0). **Consistency 2.758 mm > 2.5 mm: pass_ false.** Scratch diagnostic: the maximum sits at lift-offs 1.5 and 2.5 strides after the command, where the fade makes it Stage 1's own error; first stride 2.04 mm. The 2.5 mm constant came from the unfaded prototype. Constant unchanged; needs the lead's ruling.
- At-command ZMP (recorded, ungated): stop 2.40 mm, turn 10.9 mm.

Deviations / notes:
- Added public helpers: step_down (D7's rebuild logic), turn_design, walk_schedule, continuing_walk, com_path_causal, preview_run_switch, causal_errors, walk readers (look solver may override); private _turn(p0=) for the causality test.
- CoM "nothing before the command" is array_equal only on a common preview grid (grid = smallest frame interval, which depends on the commanded plan); own grids agree within 0.05 mm.
- Entries keep NaN for the stand's period/duty (Stage 1's schema): s2io.canonical(allow_nan=False) will refuse MOVE_SUBSET's gaits unless NaN is mapped.
- Python 3.13: Gait("stand") != itself; the stand is compared field by field.
- Legacy record is written after every transition.
- RAMP_S and require_one_thread are local until R merges; that edit changes moves_sha256, so the legacy check must be re-run.

Not done: G2 reference regeneration tests of test_stage2_look.py (two parallel processes; cap). Compute about 1.55 core-h (cap 1.5; legacy check 0.93, about 30 % slower than Stage 1's times).
