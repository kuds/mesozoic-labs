# fix:CG (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

## Fix report: work package CG, worktree /home/user/mesozoic-wt/wpc-cg, branch wt/wpc-cg

The working branch has not moved. It is still at fccebe42, which is also the merge base, so no merge was needed. I made two commits on top of the builder's b13bcd4c. No record was written, nothing ran on a look set, nothing was pushed, and the main checkout was not touched. Its untracked review files are not mine.

### Per finding

**1. MAJOR, S2-4 starts without a committed baseline (rule M3): fixed.**
- `stage2/tb/run_trackb.sh` has a new check, `V8-baseline-committed` (function `committed`). It sits after the copy step and before `V8-s2-4`.
- It passes only when all of these hold: `cmp` with the baseline that was made, `git ls-files --error-unmatch`, `git cat-file -e HEAD:./<file>`, `git diff --quiet HEAD -- <file>`, and `git log -1 --format=%H -- <file>` is not empty.
- Otherwise the chain stops with exit 1 and the message "...it goes to the lead, who commits it and runs the chain again". No marker is written.
- `V8-baseline-commit` is renamed `V8-baseline-copy`.
- Test: `test_s2_4_waits_until_rule_m3s_baseline_is_committed`. It runs a copy of the chain in a scratch git repo and covers six cases: absent, untracked, only staged, committed then changed, committed with other bytes, and committed (passes, S2-4 starts).
- Against the chain with the check line removed, the test fails because S2-4 runs.

**2. MAJOR, G9-ref passes while (a) fails for a reference: for the lead. Pass semantics unchanged.**
- Today `followed` says the references "are treated as not available (rule T2, H3)", but no code does that.
- Option A: G9-ref fails (exit 1) while a failing item's reference is still available in the set. `followed` names those references. The lead marks them not available, freezes again and reruns G9-ref. This needs only CG's files.
- Option B: keep the pass and make the env and bar absent those references. That needs `r5lib`/`r5env`/`bar` changes, not CG's files, and `bar` is in a record key.
- Option C: keep the pass, but `require_g9_ref` refuses gates2's closed-loop gates while such a reference is still available. This does not stop `lookbar` or `train2` steps in the chain.
- **Recommendation: A.**
- The related point about the start: I stated the reading in the rule text, as the reviewer allowed. Every item plays the start and none plays it alone, so a failing start shows through `walk/nominal` and blocks there.
- Tests added: `test_g9_ref_part_a_on_a_doctored_reference_check` and `test_g9_ref_passes_only_when_every_part_does`. The second one pins the current reading for a failing item that does not block, and is marked as pending your ruling.

**3. MEDIUM, kill-resume reuses runs made with other settings: fixed.**
- New `k3_asked` and `run_settings_problems`. A run on disk is reused only when its `run.json` settings match on every key:
  - mode, body, seed, set, n_envs, n_steps, tiny, gates;
  - checkpoint and check intervals, budget;
  - config, ppo_overrides;
  - lr start and filter_hz, which when not given are resolved as `train2._validation` resolves them.
- Otherwise the run is moved aside and made again, and the record lists the differences.
- Kill-resume is refused before any run if the trainer would not run 8 envs × 1024 steps.
- `settings_ok` gains `uninterrupted_settings_as_asked`, `killed_settings_as_asked` and `same_settings` (both runs' settings_sha256 equal).
- Tests: `test_a_run_made_with_other_settings_is_not_reused` and `test_what_a_runs_settings_are_compared_with`. The stub settings now give `n_steps=16`.

**4. MINOR, r6-start reuses another set's run: fixed.**
- New `r6_training` and `r6_run_problems` check R6, body, set, seed 42, tiny, gates off and the budget. A mismatch is moved aside and the run is made fresh. It refuses while that run's trainer is alive.
- Test: `test_r6_starts_tiny_run_is_reused_only_when_it_is_this_ones`.

**5. MINOR, G9-ref on an unfrozen set and the files it plays: fixed, with one correction to the review.**
- The correction: an unfrozen set did not pass. With the fix reverted, `r5lib.Library` in part (f) raised `tune.InputError: ... the set is not frozen`, which is exit 2, after (a)–(c) had already played.
- The rest of the finding stands, and is fixed:
  - G9-ref now refuses an unfrozen set (or comparand) up front.
  - `inputs_of(played_files(v))` checks every file (b) and (c) play, and the comparand's walks, with `setio.read_inputs(require_frozen=True)` before anything plays.
  - The record holds `inputs.played`, `inputs.library` and `inputs.comparand`.
- Tests: `test_g9_ref_refuses_a_set_that_is_not_frozen_or_whose_files_are_not_the_freezes`, and new assertions on V0's inputs in `test_g9_ref_on_the_frozen_set_v0`.

**6. MINOR, test gaps: fixed.**
- All eight surviving mutations are now caught: `blocked=[]`, (a) `ok=True`, the pass test without b, without c and f, (d) ignoring `missing`, `np.allclose`, `as_ruled=True`, and G8a ignoring compared_steps.
- New tests: `test_the_comparison_has_tolerance_zero` (one ulp, float32 and float64), `test_a_kill_whose_first_progress_reading_is_past_the_kill_step_is_no_kill_as_ruled`, `test_g8a_counts_a_short_comparison_as_a_difference`, and the missing-only case of (d).

**7. MINOR, G8a passes a reduced record: fixed.**
- A reduced run, or the S1 form, is refused (exit 2, nothing run) when `--record` lies under `stage2/tb/gates/`.
- Test: `test_g8a_writes_no_reduced_record_where_the_gates_record_lies`.

**8. MINOR, G9-run (e) does not check what the reread covers: fixed.**
- New checks `rows_10ms` (every row ends `@10ms`) and `turn_conditions` (every condition starts `turn/`). Tested in `test_g9_run_part_d_and_e`.

**9. LOW**
- The comparand's files are now cited by their own sha256, with `entry_sha256` beside it: fixed and tested.
- `--refcheck` skipping the freeze check: fixed. `files_match_freeze` is checked for every source; if it fails, (a) is not read and blocks. Tested.
- `dev done` failure leaving the entry "listed": fixed with a second marker, `done/<name>.x1`. Tests: `test_a_listed_step_whose_entry_was_not_marked_done_is_marked_on_the_next_run` and the updated marker test.
- The `stub_kr` fixture changing the working directory: fixed. It now uses `MonkeyPatch.context` and restores it.
- G9-ref on V6noF5 inside V1: **for the lead, unchanged.** The options are to keep it, or to move it to just before V10. I recommend moving it: the design's V1 lists V6 only, and X1/Q1 need it only before R2L.

### Readings of mine, for the lead
- K3: when lr or filter is not given, it is compared with what the validation records give at the time of the call. If `lr_probe.json` or `filter.json` changes later, both runs are remade. The alternative is to compare only what was given.
- G9-ref: a played file that is not the freeze's is a refusal (exit 2), not a fail (exit 1).
- The chain's M3 check does not replace the trainer's own pre-rollout refusal. The builder's `--expect-baseline` point is still open in `train2`, which is not CG's file.
- These remain open for the lead, as before: V3 cannot run as built, the 1e-6 rad join rule, and the builder's other readings, chain choices and the trainer builder's four open points.

### Exact outputs
- `nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_gates2.py` (committed head): **41 passed, 1 skipped in 82.78s (0:01:22)**
- `STAGE2_SLOW=1 … -k r6_start` (the slow S1 test plus the new fast r6 test): **2 passed, 40 deselected in 23.21s**
- Reverts and mutations, run in-process with a plugin: 21 runs, 21 caught. Results are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave5/cgfix/RESULTS.txt`.
- `sh -n`: exit 0. `dash -n`: exit 0.
- `TB_DRY=1 sh stage2/tb/run_trackb.sh`: exit 0, 46 steps, last line "dry run (TB_DRY): nothing was run or written".
- Record-key hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.
- `git check-ignore -v`: new live paths are ignored by `stage2/tb/.gitignore:1:work/` (`.x1` markers, `V8-baseline-committed`, the `.discarded.0` directories). `R0L_BASELINE.json` is not ignored (exit 1), as a record should be.

`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 .../trex_form_study/designlab/stage2/gates2.py     | 2075 ++++++++++++++++++++
 hardware/trex_form_study/stage2/tb/run_trackb.sh   |  266 +++
 .../trex_form_study/tests/test_stage2_gates2.py    | 1278 ++++++++++++
 3 files changed, 3619 insertions(+)
```
`git status --porcelain`: empty.

### Commits, oldest first
f84f1ced, 3a0983a0, 8069eda9 (merge), b13bcd4c (the builder's), then mine: 5527741c (chain: M3 check and the X1 marker) and 4329171e (gates fixes and tests). Both of mine end with the two attribution lines. **Head: 4329171e.**

### Not done
- No gate or validation record was made.
- K3's real settings were not run, and the real-env tiny kill-resume on S1 was not rerun after the settings check. The settings tests use the stub.
- Rates were not rerun.
- The chain was only dry-run, plus its tests.
- No other module needs to change for these fixes. Option B of finding 2 would need `r5lib`/`r5env`/`bar`, and M3's pre-rollout refusal would need `train2`.
