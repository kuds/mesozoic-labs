# build:CE-minor (roundC, wave 2a): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

WP CE, second fix round: report (branch wt/wpc-ce, worktree /home/user/mesozoic-wt/wpc-ce, head cdc4762; nothing pushed, main checkout untouched)

All eight findings are fixed and committed. Every acceptance run passed. One slip of mine needs your decision: the closing merge commit cdc4762 lacks the two trailer lines (see 5a).

1. THE MERGE
- At the start the branch was 0 ahead and 9 behind, so `git merge --ff-only` moved it from 16890e8 to bb5b63d. No merge commit, no conflict.
- At the end the working branch had three more commits (tip 82ba1ad: the servo model's note and its review, STATUS.md; four files, none of CE). Merged automatically as cdc4762, no conflict.
- Record-key hashes at cdc4762: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1 (the registered ones).

2. WHAT CHANGED, PER FINDING (g8a.py and its test file are unchanged)
(1) r5env `_check_freeze`: one path for every named freeze, latest or earlier. It compares the used files and the task table with the named freeze's manifest, and checks that the manifest hashes to its name. Test: test_the_task_table_is_compared_also_when_the_named_freeze_is_the_latest.
(2) r5lib `Library._check_crouch`, `CROUCH_RULE`: a set is refused when set.json's crouch_frac differs from that of a walk entry the library reads, or the entry records none. The message names both values. Tests: test_a_set_whose_crouch_differs_from_a_walk_entrys_is_refused (r5lib), and the review's case C4 in test_a_later_freeze_that_changes_what_the_run_uses_is_refused_with_what_differs.
(3) r5core `REWARD_RULE`, `HEAD_RULE`, `READINGS[3]` and the docstrings state 3a to 3f. No formula changed. The sentences are pinned in test_the_reward_rule_states_the_readings_as_coded. Behaviour is pinned by:
- a: test_the_first_env_steps_displacement_runs_from_the_reset_and_spans_18_ms (env)
- b: `0.0 < max(coded) < 1e-4` in test_progress_is_the_shortfall_against_the_references_own_mean_speed
- c: test_dutys_stride_is_kept_between_1_and_256_env_steps
- d: test_clearance_a_is_not_read_without_a_swing_length
- e: test_what_is_read_at_every_physics_step_is_the_plants_own_computed_at_that_steps_start (env)
- f: test_in_r6_the_head_overlay_is_added_to_the_settled_home_pose (both -X bodies)
(4) CE_RULE_WORDING.md entry 10 (T3), Label B, naming commit 225f4b3.
(5) Entry 1b rewritten to your ruling: E4 stays in walking turns, the heading penalty there is not zero, the tracking figure is scoped to the two sets, "every penalty below 1e-4". Label A. I measured the numbers again (see 3).
(6) r5lib `from_plan(plan, track=None, heading="command")`; scripts are kept by plan, track and heading. Tests: test_a_plan_given_as_a_plan_can_name_the_references_yaw (r5lib); the last block of test_a_playback_of_a_turn_in_place_scores_heading_zero (r5core).
(7) r5env and r5lib:
- `r5lib.task_names(body, set, root)` resolves the task names from the JSON records alone. `env_configs(..., root=)` stores them in `tasks`.
- `set_freeze_started` is the first freeze that holds the named manifest.
- A named task the set does not hold raises tune.InputError. A name that is no row of the table stays ValueError.
- describe() gains `task_names`, `task_files`, `crouch_rule`. Its docstring and the constants `RESUME_KEYS`, `LATER_FREEZE_KEYS` state what a trainer compares and what an accepted later freeze changes.
- Tests: test_env_configs_name_their_tasks_so_a_later_freeze_that_adds_a_task_is_accepted, test_set_freeze_started_is_the_index_the_run_started_on, test_describe_holds_the_rules_and_no_path, test_task_weights_and_env_configs.
(8) Three test gaps:
- test_the_envs_own_draw_at_the_end_of_the_last_span_starts_just_before_the_hold drives the env's own generator.
- test_a_task_that_leaves_the_table_under_a_later_freeze_is_refused.
- The fixture `freeze_t0` lists available entries only; it is compared with `sets.manifest_of` in that test.

Wording file beyond items 4 and 5: entry 1a gains the two sentences for 3a and 3e; entry 4a now says "charged below 1e-4 (W1)"; entry 6c is new (E5: the head command in R6, Label A); entry 8 (K4) is rewritten because this round's code made its old text untrue. All 20 "Old" quotes are verbatim in the draft on wt/prereg-b; all 8 commit ids and 27 test names in the file exist.

Each fix reverted alone in the source (file restored afterwards), with the tests that then fail:
| Revert | Failing tests |
|---|---|
| early return when the named freeze is the latest | named_freeze_is_the_latest, set_freeze_started |
| no crouch check | the r5lib crouch test, the env "refused" test |
| from_plan ignores heading | the r5lib plan test, turn-in-place playback [left, right] |
| env_configs names no task | task_weights_and_env_configs, env_configs_name_their_tasks |
| last index instead of first | set_freeze_started |
| ValueError for a task that left | named_freeze_is_the_latest, leaves_the_table |
| no task_files | describe_holds, later_freeze_accepted, env_configs_name_their_tasks |
| inline draw in reset | envs_own_draw |
| duty without the upper bound | dutys_stride |
| clearance (a) without the skip | clearance_a_is_not_read |
| R6 overlay not on home | in_r6_the_head_overlay [both bodies] |
| substep read before the physics step | what_is_read_at_every_physics_step |
With all three sources at their old state, 17 tests fail. The 3a env test pins unchanged behaviour and passes on the old code; the missing sentence is what fails there.

3. ACCEPTANCE (at cdc4762, clean tree, the brief's env, through tools/lowprio.sh; PY is the main checkout's venv python)
- `$PY -m pytest -p no:cacheprovider tests/test_stage2_r5lib.py tests/test_stage2_r5core.py`: "62 passed in 50.94s"
- `... -rs tests/test_stage2_r5env.py tests/test_stage2_g8a.py`: "43 passed, 1 skipped in 92.26s (0:01:32)"; the skip is "tests/test_stage2_r5env.py:675: slow: set STAGE2_SLOW=1". With STAGE2_SLOW=1 -k walks_on: "1 passed, 35 deselected in 6.98s"
- `... tests/test_stage2_tbenv.py`: "11 passed in 0.72s"
- ruff on the eight files: "All checks passed!"
- `g8a.bar_form("75-M-H-10", "V0", ("walk/nominal",), (3044,), (0, 5), root="/home/user/mesozoic-labs/hardware/trex_form_study/stage2/sets")`, exit 0:
  - R0@0ms: "reference_steps": 7030, "env_physics_steps": 7030, "compared_steps": 7030, "first_difference": null, "reference_fell_step": null, "env_terminated_step": null, "ends_alike": true, "identical": true
  - R0@10ms: the same fields with the same values, "identical": true
  - document: "identical": 2, "of": 2, "not_run": [], "passes": true; set_freeze index 0, 34a820e5...; "commit": "cdc476225c4a7ae1edcbf90f6fb13353165778ae", "dirty": false
  - track_b source hashes: r5lib 615e87cb..., r5core a021696a..., r5env 15e2adc0..., g8a b01013a3... (unchanged)
  - Main checkout before and after: `git status --porcelain` 0 lines; the listing of stage2/sets (87 files, sizes and times) identical.
- Numbers measured for the wording file (kinematic playbacks, no physics step, no record; run twice with identical output, the second time at cdc4762):
  - V0 heading penalty up to 0.351, 0.139 and 0.0447 (quick, 0.5 and 0.35 rad/s turns); the reference's yaw 8.97, 5.37 and 3.76 degrees behind; above zero on 98/99, 148/149 and 207/208 of 1,499 steps.
  - Tracking minimum 0.9891 to 0.9938 on V0's walks, 0.9618 in the quick turn, 0.9936 to 0.9938 on Stage 1's tasks, 0.9507 in Stage 1's turn in place.
  - Progress up to 2.11e-05 on V0 and 8.3e-06 on Stage 1's arrays; every other penalty 0.
- Env readings on the S1 view:
  - First env step at a start in motion: 0.3027 m/s read, against 0.3384 m/s over the full 20 ms.
  - Per-physics-step foot loads equal the pass on the state before to 2.8e-9 body weights and differ from the pass after by up to 1.2.
  - Mid-start loads at five starts: at most 0.76 body weights at k = 0 and 2.06 at k = 5, against 2.5 to 4.4 under the home targets.
- V0 builds with its freeze and its 15 tasks named (a build, no episode).
- Logs and scripts: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/ce_fix2/

4. DIFF AND STATUS
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat` (tip 82ba1ad; the branch is 5 ahead, 0 behind):
 designlab/stage2/r5core.py | 66, r5env.py | 148, r5lib.py | 125, stage2/design/CE_RULE_WORDING.md | 238, tests/test_stage2_r5core.py | 159, test_stage2_r5env.py | 442, test_stage2_r5lib.py | 100; "7 files changed, 1132 insertions(+), 146 deletions(-)"
`git status --porcelain`: empty.

5. FOR THE LEAD
a. Merge commit cdc4762 has the default message without the two trailer lines. I did not amend it, because the rules say never rewrite history; it is unpushed, and the bar_form document above names it. Amend it or accept it as you decide.
b. Finding 1: in the latest-freeze case the manifest's own hash is now checked as well, which is one more refusal than the finding asked for. Settings that name no freeze are still not compared (g8a's env names none).
c. Finding 2: "used" is every walk entry whose file the library reads: the walks of every task the set holds, the nominal walk, and plans given as plans. That is stricter than K4's "this env's tasks". The alternative is the env's own tasks only.
d. Finding 7, the index: the settings hold the hash alone, so the index is that of the first freeze holding that manifest. `sets.freeze` never appends a freeze whose manifest an earlier one holds, so on real sets this is the start index. The alternative is a config field for the index; I did not add one.
e. Finding 7, the config: describe()["config"] stays the config as given. The resolved names are in `env_configs()` output and in the new `task_names`. A hand-made config with tasks=() is still refused when a later freeze adds a task.
f. Finding 7, the error: InputError is raised for any named task that is a row of the table and absent from the set, also when no freeze is named.
g. HEAD_RULE carries two body-specific numbers (the jaw settles 0.0015 rad open on 75-M-H-10-X and 0.0007 on 60-N-T-10-X). A test pins them; a rebuilt model or a third -X body needs the string updated.
h. In R6 a scripted jaw target therefore keeps 0.1485 and 0.1493 rad from its limit, not 0.15. `Core` does not refuse this and the range term asks 0.10. Not changed.
i. Two things the new sentences state and I did not change:
   - The reference's flags are read at the clock after a physics step while the contact is of its start: 2 ms apart, inside the 24 ms allowance.
   - A mid-task start's first step reads the speed about 10 % low (tracking term 0.9913 against 0.9970).
j. 3c and 3d are in REWARD_RULE only, not in wording entries 3a and 3b; the finding asked for a and e in the file.
k. Not done:
   - The env's `options["plan"]` takes no heading; only `from_plan` does.
   - No training, no policy episode, nothing on F25F9, F25F9cl or a look set.
   - The suite beyond the five files was not run.
l. Runs beyond the brief's list: the V0 and S1 kinematic playbacks, a few env resets and steps on the S1 view, the V0 build, the revert runs. Pytest's fixtures write under /tmp/pytest-of-root, as before.
m. For CT:
   - Build envs from stored `env_configs()` output.
   - On a resume compare `r5env.RESUME_KEYS` of the description as built, never the whole description; `tasks` holds the weights in force, which `set_task_weights` changes.
   - New names: `r5lib.task_names`, `absent_reason`, `why_not`, `CROUCH_RULE`, `from_plan(..., heading=)`; `r5env.RESUME_KEYS`, `LATER_FREEZE_KEYS`.

6. COMMITS (oldest first)
011f6e2 r5lib: crouch check, from_plan heading, task_names
b4c223e r5core: the rule strings' six details and their tests
4523ed6 r5env: K4 for the latest freeze, env configs name tasks, resume keys, three test gaps
d500d89 CE_RULE_WORDING.md: T3, R6 head command, W1 as ruled, K4
cdc4762 merge of the working branch (82ba1ad); no trailer lines, see 5a
