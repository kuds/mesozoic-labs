# Walking pilot 3: the pilot-only branch plan (DRAFT, 2026-11-03; revised after review, FIXES.md; finalised for the freeze candidate, CHANGES.md; amended 2026-11-04 for R6 and the -2.0 reset, CHANGES2.md; fixed after the R6 reviews, FIXES4.md)

The plan as drafted, with the maintainer's final choices of 2026-11-03 14:53 UTC applied (budget 8,000,000 and no
futility stop; the 4,194,304 variant is deleted) [measured: SP/work/run_log.txt line 1005]. **As built:** the branch
stage built commits 1, 2 and 3 below on `main` `4d5f4e7` in SP/wt_p3 (local branch `claude/pilot-walk-r10-p3`, not
pushed): `f2399c8ffa7feb9637be8589f5e727550c351dcd`, `76ceae9d1d0985b2585b2266a4d26d9a34df820b`,
`3ae1c95967547ada569ad71aebaa88999afa02cd` [measured: `git log` in SP/wt_p3]; the checks it measured are in the
criteria's "Pre-launch checks". Branch names are written without their agent prefix.

**Amended 2026-11-04 (AMEND stage, CHANGES2.md):** the maintainer's decisions (8)-(11) of 2026-11-04 11:57 UTC
[measured: SP/work/run_log.txt line 1034] added commits 4, 5 and 6 below, built by the implement stage
(SP/work/pilot3/r6_impl/) on `3ae1c959`, not pushed: `b7a146db1c67879f4db9c216d709f9cfb660f7ef` (R6's code, off by
default), `4dfc7cbc38e65a3935096daa1f48c42eac6bd457` (R6 on in `locomotion.toml`), head
`a3f1b378aede66b0ee8630197f36a7c374ec77c0` (`warm_start_log_std` -2.0) [measured: `git log` in SP/wt_p3, read only;
SP/work/pilot3/frozen_candidate2/out/wt_p3_state.txt: clean, `status --porcelain --ignored` empty]. Decision (10)
leaves the gate and decision (11) the inputs and the plant unchanged. The validate stage (SP/work/pilot3/r6_validate/)
measured R6 against SCREEN, the stance path untouched, the parent at -2.0 under R6 and the re-smoke at the new head.

**Fixed 2026-11-04 after the R6 code and text reviews (FIX4 stage, FIXES4.md; SP/work/pilot3/r6_fix/):** commits 7
and 8 below, on `a3f1b378`, not pushed: `99859224b41bf666f0b3587ea4b812ab638ff13d` (a test that a reset starts R6's touchdown event afresh on one
env; the integration test's pattern checks by kind; corrections of commit 4's message) and head
`31d02e42d148511e50f9cba9a598d9a9d36c23af` (two comment lines of `locomotion.toml`) [measured: `git log` in SP/wt_p3;
SP/work/pilot3/r6_fix/out/wt_p3_state.txt]. Neither changes code or a config value; no digest line and no line number
this plan or the criteria cite moves. The smoke was re-run at the new head (SP/work/pilot3/r6_fix/).
Labels: **[measured: source]** = read or computed from the named file; **[inferred: basis]** = reasoning.
"SP" is the session scratchpad folder named in the task. Companion files: `criteria_pilot3.txt` (the rules) and
`NOTES.md` (sources and every inferred point).

## Base, name and handling

- **Base:** `main` at PR #658's merge commit `4d5f4e76232d6e19a4526e2945c28228d561f30d` (merged by the maintainer
  2026-11-03 14:19 UTC [measured: SP/work/run_log.txt line 1003]). It is a merge commit, not the squash this plan
  expected: its parents are `main` `4c9d5e9b` and PR #658's head `cab70394`, and its tree `dd9d6852` equals
  `cab70394`'s tree [measured: `git log`, `git rev-parse` in SP/wt_p3], so the code the branch builds on is exactly
  `cab70394`'s. PR #658's head is `cab70394`, 13 commits on `main` `4c9d5e9b` [measured: `git rev-list --count`].
  The Option B review advised a squash because commits 4-9 fail one identifier test on their own [measured as
  recorded: SP/work/run_log.txt line 997]; with a merge commit those 13 commits are now in `main`'s history. That
  matters for bisecting `main`, not for this branch, whose base tree is the PR's final tree.
- **Name:** `pilot-walk-r10-p3` (with the agent prefix on GitHub). The judge refuses any tree whose branch is not a
  `pilot-walk-*` branch after that prefix [measured: main `docs/investigations/stance_2026_10/pilot/pilot_judge.py`,
  the `branch.startswith(...)` check].
- **Never merged, no pull request.** Pushed once, with the maintainer's OK, before launch (the notebook clones the
  branch from GitHub). **Hold** (pilot 2's rule): after that push nothing more is pushed to it until every seed,
  resumes and `_b` restarts included, is judged.
- **Commit messages** end with the trailer lines the session's rules prescribe; each message says "pilot-only,
  never merged" in its title, as pilot 1's did.
- Every commit is built in a fresh clone or a scratch worktree outside `/home/user/mesozoic-labs`, never in the main
  checkout.

## Commits, in order

### 1. Let a stage config set its evaluation and checkpoint cadence (pilot-only)

Built as `f2399c8` (a `git cherry-pick --no-commit 176af89` with the two conflicts resolved by keeping both sides,
as its message records).

Why: the criteria keep pilot 1's evaluation cadence (10 episodes every 131,072 steps) and need a checkpoint cadence
that lands on 2,000,000 / 4,000,000 / 6,000,000 / 8,000,000. `main` has no stage key for either: the notebook's
`train_stage` passes `eval_freq=50_000, save_freq=100_000` [measured: PR #658 notebook cell 15, extracted with
`D/tools/nbcells.py`], and `main`'s `train_base.py` and `gate_schema.py` hold no `_stage_cadence` or cadence keys
[measured: `git grep` on `origin/main` and `cab70394`].

What: a hand port of the run branch's `176af89` (2026-10-26, "Let a stage config set its evaluation and checkpoint
cadence (r10 screen, never merged)"; itself the port of `412f19ae`, which pilot 1 carried as `85e17e8`).

| File | Change |
|---|---|
| `environments/shared/curriculum/gate_schema.py` | `_CADENCE_KEYS = frozenset({"eval_freq", "eval_episodes", "checkpoint_freq"})` beside `_RETENTION_KEYS`, and its union into `validate_gate_config`'s allowed keys |
| `environments/shared/train_base.py` | `_eval_episodes_for_stage` returns a declared `eval_episodes` exactly; `_positive_cadence`; `_stage_cadence(stage_config, eval_freq, save_freq)`; one call at the top of `_build_core_callbacks` |
| `environments/shared/tests/test_stage_cadence.py` | new, byte-identical to `176af89`'s (68 lines) |

- **It does not apply as a plain patch.** `git apply --check` of `176af89` on a git-archive export of `cab70394`
  fails on `gate_schema.py` (the hunk at line 324, beside PR #658's new `WARM_START_LOG_STD_KEY` block) and on
  `train_base.py` (the hunk at line 182, beside PR #658's new `WARM_START_RUN_KEYS` block); the test file applies
  [measured: `git apply --check -v` of `git format-patch -1 176af89`, CMDLOG 2026-11-03T13:44:53Z; the saved patch file
  was later deleted, NOTES.md]. A three-way cherry-pick or a hand edit is needed; the result is reviewed against
  `176af89`'s code diff (114 lines, kept as `D/src/cadence_176af89.diff`).
- Checks: `test_stage_cadence.py` passes; no committed config declares the keys at this commit, so nothing trains
  differently and no digest moves [inferred: the keys enter no digest, `176af89`'s message and code]; the full shared
  and T. rex suites pass at this commit with nothing deselected, PR #658's off-path tests included (the guard that
  the hand port changed nothing on the default path).

### 2. Configure the T. rex locomotion stage as walking pilot 3 (pilot-only)

| File | Change |
|---|---|
| `configs/tyrannosaurus_rex/anatomical/locomotion.toml` | `[curriculum]` gains, each with a comment naming the pilot, its decision and "never merged": `warm_start_log_std = -1.5` (decision (c), UNTESTED; set to -2.0 by commit 6, decision (8)), `eval_freq = 131072`, `eval_episodes = 10`, `checkpoint_freq = 500000`, `max_checkpoints = 0`. Nothing in `[env]`; no `[ppo] critic_phase` (even `false` moves the recipe digest [measured as recorded: Option B equivalence review]); no `tensorboard_log` |
| `configs/digest_snapshot.generated.txt` | regenerated with `python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --write` |

- **Expected moved digest lines** [inferred: pilot 1's `3e458e6` and pilot 2's `fb5510d` messages; PR #658 adds
  `warm_start_` to `config.SHAPING_KEY_PREFIXES`, so the recipe digests cover the key]: locomotion's
  `hyperparameters_sha256.PPO` (now `fde660ac...`) and `.SAC` (`49ef875a...`) and the two
  `stage_config_view_sha256` lines (`52ddbe9d...`, `db1f017d...`) [measured: the current values, PR #658's snapshot
  lines 398-401]. **Unchanged:** `task_sha256` `5f835d65...` and `gate_sha256` `02602cb0...` (lines 396-397). The
  code review found the same four lines moved in pilot 2's `fb5510d` [measured by the code review, 2026-11-03]. The
  builder records the moved lines exactly; any other moved line stops the build.
- **Known failing tests on the branch** [inferred from reading; the builder measures the exact set at this commit,
  and only the measured set is deselected and written into the criteria]: PR #658's tests
  that pin `main`'s locomotion recipe digest or assume no committed stage declares the reset key. Read so far:
  - `environments/shared/tests/test_critic_phase.py::test_the_tyrannosaurus_rex_recipe_digests_are_main_s[locomotion-PPO]`
    and `[locomotion-SAC]` (they pin `fde660ac...` and `49ef875a...`);
  - `test_critic_phase.py::test_the_recipe_digest_records_a_warm_start_log_std_when_a_stage_declares_one`;
  - in `test_curriculum_warm_start_log_std.py`: `test_no_committed_stage_declares_it` (`declared == {}`),
    `test_the_recipe_digest_covers_it` and `test_a_stage_without_the_key_does_not_set_it` (both read the committed
    locomotion config as the "without" case);
  - likely `test_critic_phase_off_path.py` (its scenario trains from the committed configs,
    `critic_phase_off_scenario.py` line 142, and compares against `main`'s golden, which the reset and the cadence
    keys change) and the `test_critic_phase_drop.py` cases that load the committed T. rex stages (lines 384-441),
    if they assert on the run record or the noise (not established here).
  Each is deselected by id on the branch's test runs, named with its cause in the commit message and in the
  criteria's pre-launch checks, as pilot 1 deselected the one test that pinned 8,000,000. The tests are not edited.
  Any other failure stops the build. Deselecting `test_critic_phase_off_path.py` removes the off-path guard for
  commits 2-3; that is acceptable only because commit 1 runs the full suites with nothing deselected (the guard that
  the cadence port moved nothing on the default path).
- **As measured at commit 2** (`76ceae9`): 14 ids, each passing at commit 1 and failing at commit 2 [measured:
  SP/work/pilot3/branch/out/c2/deselect_ids.txt, c1/shared.txt and c2/shared.txt]: the ten read above (the six named,
  all three `test_critic_phase_off_path.py` tests, and from `test_critic_phase_drop.py` only
  `test_the_trainer_drops_records_and_resumes_the_child_stock`), plus
  `test_critic_phase.py::test_no_stage_config_turns_the_critic_phase_on` and three
  `test_train_base.py::TestTrainCurriculumWalksTheManifest` tests not read above. The criteria list all 14 with their
  causes.
- Checks: `digest_snapshot --block-optional-backends --check` current; the judge's
  `hyperparameters_sha256(config, "ppo")` on the branch equals the snapshot's new PPO line.

No species catalog commit: at 8,000,000 the catalog stays current (`species_catalog --check` current at each of the
three commits [measured: SP/work/pilot3/branch/out/c{1,2,3}/quick.txt]). The plan's catalog commit belonged to the
deleted 4,194,304 variant, so the parent commit, numbered 4 in the draft, is commit 3.

### 3. Add the PC round's stance7_r10LTSBpc_s42 final as walking pilot 3's parent (pilot-only)

| File | Change |
|---|---|
| `pilot_checkpoints/stance7_r10LTSBpc_s42_final/stage1_final.zip` | 4,048,776 bytes, sha256 `1c9d87c3d0f817b048de429bb80cdc89fcd2c83d5fbafd7f444af0b5474b9ff6`, copied byte for byte from `SP/work/fetch/stance7_r10LTSBpc_s42/01_stance/models/`, `git add -f` past the `*.zip` ignore rule |
| `pilot_checkpoints/stance7_r10LTSBpc_s42_final/stage1_final_vecnorm.pkl` | 5,291 bytes, sha256 `5e417fef0ea52b9bbbcb641443854f309598702f25ece32e01e8eb65e6d507b1`, `git add -f` past `*.pkl` |
| `pilot_checkpoints/stance7_r10LTSBpc_s42_final/SOURCE.txt` | the provenance, below |

[measured: sizes and sha256s, CMDLOG 2026-11-03; the `.gitignore` rules `*.zip` and `*.pkl` at lines 42-43 of
`cab70394`.] The sha256s are re-checked after the copy and in the clone after the commit.

`SOURCE.txt`, on pilot 2's model (its `stance3_armLT_s60_final/SOURCE.txt`):
- **What it is:** the final checkpoint (`stage1_final`, 12,001,280 steps) of `stance7_r10LTSBpc_s42`, seed 42,
  physics r10, policy interface r15, trained from scratch on Colab L4 (N_ENVS 4, CRITIC_PHASE True, K3_PANEL True)
  from the never-merged branch `screen-trex-r10-phasecritic` at `1dfe2327` (the run branch `screen-trex-r10-armltsb`
  at `8737cd3d` plus the critic phase); the LTSB stance recipe with mirror-symmetric PPO; launched 2026-11-01 about
  03:27 UTC [measured: run log line 886]. Not certified.
- **Why this one:** the maintainer's decisions of 2026-11-03 (parent, drop, reset, declared exception), with the PC
  round's pinned parent rule's tie (s42, s80) and the maintainer's choice of s42.
- **Where it came from:** the Drive stage directory `MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/
  stance7_r10LTSBpc_s42/01_stance/models/` [inferred: the archive's own TensorBoard path names this run folder];
  re-fetched read-only on 2026-11-02 for Option A, each file stored only when its size and sha256 equalled the PC
  round's F2 manifest [measured as recorded: Option A report, "Re-fetch"].
- **Plant identity:** physics r10 `sha256:6a47a0eb...`, policy interface r15 `sha256:ffbf6631...`, equal to `main`'s
  after PR #658 [measured: the zip's data member; PR #658's digest snapshot line 202]; obs 63, act 15.
- **What the archive carries:** `CriticPhaseActorCriticPolicy` with the phase record (settle 200, gamma 0.98, width
  3), a `mirror_kl` symmetry record, its own run's TensorBoard path; the sidecar is a stock VecNormalize pickle.
  What the warm start does with it (the drop, the reset, the stock-PPO child without the phase or symmetry record),
  with the two expected "Warm start" lines.
- **How it was judged:** Option A's K1-K4, M2, M1 and critic readouts (numbers as in the criteria's "Parent").
- **How the manual cell loads it:** `MANUAL_LOAD_PATH = "/content/mesozoic-labs/pilot_checkpoints/
  stance7_r10LTSBpc_s42_final/stage1_final"`, `MANUAL_VECNORM_PATH = MANUAL_LOAD_PATH + "_vecnorm.pkl"`,
  `MANUAL_LOAD_MODE = "initialize_next_stage"`, CRITIC_PHASE False.
- Text only; no code reads `pilot_checkpoints/`.

### 4. Add a default-off R6 walking reward to the T. rex env (2026-11-04; pilot-only)

Built as `b7a146d` (tree `af6d7744`) [measured: its message; SP/work/pilot3/r6_impl/]. Decision (9): R6's two new
parts as code, each behind a constructor key whose default is off; the other two parts are existing keys
(`airborne_substep_weight`, `forward_vel_max`) and are set in commit 5. The design is
SP/work/pilot3/r6_design/DESIGN.md; R6 as built is in the criteria ("What is trained").

| File | Change |
|---|---|
| `environments/tyrannosaurus_rex/base_env.py` | `walk_support_gate_bw` (default 0.0): alive, posture and height times float(r_foot_contact + l_foot_contact > gate x body weight), the compsognathus rule with the T. rex's weight; `walk_touchdown_weight` (default 0.0): SCREEN's alternating-touchdown event R5p in a pure class `_WalkTouchdownEvents`, paid at k+4; both declared in `OPT_IN_TASK_KWARGS`, so while off they enter no task digest and no recorded stage config; info keys only while on; reset in `_spawn_target`; no hashed method and no shared module touched |
| `environments/tyrannosaurus_rex/tests/test_anatomical_walk_r6.py` | new, 26 cases (off is bitwise 3ae1c959's; units on synthetic sequences against the probe's debounce and SCREEN's ported rule; the env against an offline recompute and SCREEN's R6p) |
| `environments/tyrannosaurus_rex/tests/fixtures/walk_r6/off_golden_3ae1c959.json` | the off-path golden recorded on a git-archive export of 3ae1c959 (generator in the implement stage's folder) |
| `environments/tyrannosaurus_rex/tests/fixtures/walk_r6/screen_cases.npz` | SCREEN's recorded episodes for the R6p test (465,523 bytes, force-added past `*.npz`) |

Checks [measured: its message]: ruff, ruff format --check, mypy with and without SB3 and torch clean; notebook cells
parse; plant_contract, species_catalog and digest_snapshot current (649 lines, no line moves); the T. rex suite 200
passed, 4 xfailed (174 and 4 as at 3ae1c959 plus the 26); the shared suite with the 14 ids deselected 6514 passed, 1
skipped, 0 failed.

### 5. Turn R6 on in the T. rex locomotion stage (2026-11-04; pilot-only)

Built as `4dfc7cb` (tree `12898b06`) [measured: its message].

| File | Change |
|---|---|
| `configs/tyrannosaurus_rex/anatomical/locomotion.toml` | `[env]`: `forward_vel_max = 1.5` (was 2.5), `airborne_substep_weight = 1.0`, `walk_support_gate_bw = 0.04`, `walk_touchdown_weight = 0.1`, each commented with the pilot and decision (9); no `[ppo]`, `[curriculum]` or gate key (decision (10)) |
| `configs/digest_snapshot.generated.txt` | regenerated: exactly 17 moved lines, all T. rex: locomotion's `task_sha256` `5f835d65...` -> `782d95b6...`, `stage_config_view_sha256.PPO`/`.SAC` -> `3604e087...`/`266cc7eb...`, its reward summary, `shape_sha256` and `rounded_values_sha256`, and the `task_sha256` of the 11 T. rex behavior recipes |
| `environments/tyrannosaurus_rex/tests/test_anatomical_walk_r6.py` | its first case asserts this branch's state (only locomotion sets the keys; the recipes inherit them) |
| `pilot_checkpoints/stance7_r10LTSBpc_s42_final/SOURCE.txt` | the provenance caveat no longer calls the locomotion `[env]` production's |

- The 11 behavior recipes inherit locomotion's `[env]` (`train_behaviors.read_recipe`), so they resolve to R6 too;
  the design stage had not expected it; the message discloses it; the pilot trains none of them. Pinning them off
  would need four keys in each recipe's `[env]`; not done.
- Checks [measured: its message]: the same as commit 4; the shared suite with nothing deselected 6514 passed, 1
  skipped, 14 failed, exactly the 14 ids of commit 2 (R6 adds no failure).

### 6. Reset the warm-start log_std to -2.0 (2026-11-04; pilot-only)

Built as `a3f1b378` (tree `a08393c5`), the head until the R6 reviews [measured: its message].

| File | Change |
|---|---|
| `configs/tyrannosaurus_rex/anatomical/locomotion.toml` | `[curriculum] warm_start_log_std = -2.0` (written `-2.0`, not `-2`: the TOML integer gives another recipe digest) and its comment, decision (8) |
| `pilot_checkpoints/stance7_r10LTSBpc_s42_final/SOURCE.txt` | decisions (8) and (9), -2.0 and std 0.1353, the ratios to the parent's std, the expected second Warm start line; the -1.5 lines measured on #658's head kept, labelled |
| `configs/digest_snapshot.generated.txt` | regenerated: exactly four moved lines, all locomotion's: `hyperparameters_sha256.PPO` `8263f82f...` -> `85881fc4...`, `.SAC` `a958a8fb...` -> `05cce74d...`, `stage_config_view_sha256.PPO`/`.SAC` -> `c0f35553...`/`cb47b068...`; task, gate and reward lines unchanged |

Checks [measured: its message]: the same as commit 4; the shared suite with the 14 ids deselected 6514 passed, 1
skipped, 0 failed; the judge's own derivation of the PPO recipe digest equals the new line.

### 7. Test R6's touchdown event across episodes on one env (2026-11-04, after the R6 code review; pilot-only)

Built as `99859224` (tree `5ca3e9b5`) [measured: its message; SP/work/pilot3/r6_fix/out/c7/].

| File | Change |
|---|---|
| `environments/tyrannosaurus_rex/tests/test_anatomical_walk_r6.py` | new `test_a_reset_starts_the_touchdown_event_afresh_on_the_same_env` (three episodes on one env: after the reset the event's state equals a new event's, step 0 is accepted, the episode equals the same episode on a fresh env and an offline recompute, its first touchdown is not paid as an alternation, a foot's first touchdown is timed from the new reset); the integration test's last checks asserted by kind (ALT pays at least one alternating touchdown; HOP pays neither foot of a landing within 2 steps) instead of exact counts |

The message also corrects two wordings of commit 4's: the evaluation ledger has no reward-term column (the term
reaches TensorBoard only, as `reward_terms/eval_walk_touchdown`), and the down flag reads the touch sensors in place
of floor truth. Checks [measured: SP/work/pilot3/r6_fix/out/c7/]: ruff, ruff format --check and mypy (with and
without SB3 and torch) clean; notebook cells parse; plant_contract, species_catalog and digest_snapshot current, no
line moved; the R6 tests 27 passed; the T. rex suite 201 passed, 4 xfailed. The review's mutant T17 (no reset in
`_spawn_target`) and 6 of 7 reset mutants are killed by the new test (SP/work/pilot3/r6_fix/out/mutants_reset.json;
the survivor is equivalent).

### 8. Credit the forward_vel_max cap finding to the config-only screen (2026-11-04, after the R6 text review; pilot-only)

Built as `31d02e42` (tree `17ec0cb9`), the head [measured: its message; SP/work/pilot3/r6_fix/out/c8/].

| File | Change |
|---|---|
| `configs/tyrannosaurus_rex/anatomical/locomotion.toml` | lines 17-18 of the `forward_vel_max` comment: the finding that caps at or below 1.4 m/s stop paying near the gate's cruise is the config-only screen's inference (configonly RESULT.md, "[inferred]"), not a SCREEN measurement; same line count, the parsed TOML equal to `a3f1b378`'s |

Checks [measured: SP/work/pilot3/r6_fix/out/c8/]: the same as commit 7, no digest line moved; the shared suite with the 14 ids deselected 6514 passed, 1 skipped, 0 failed; run alone, the 14 fail, the same ids with the same first-line reasons as at a3f1b378 (SP/work/pilot3/r6_fix/out/c8/shared.txt, deselected14.txt).

### Possible extra commit (not made). A TensorBoard override (pilot-only; not needed, the smoke measured the child's own path)

(Planned as "possible 5" before commits 4-6 existed; the commit numbered 5 on the branch is R6's.)

If the pre-launch smoke shows the child's model carrying the parent's TensorBoard path (criteria, "What is
trained"), a one-line pilot-only change sets the warm-started model's `tensorboard_log` to the child's path before
`learn()`, with a test; the criteria are amended before launch. Not expected [measured by code reading: with
`use_tensorboard=True` (train()'s default; `train_stage` never passes it), `train_base._prepare_alg_kwargs` sets
`tensorboard_log` to the child's path, the `/tmp/tb_buffer` copy on a mount and `gcs_tb_path` elsewhere (cab70394
lines 505-526); `load_sb3_model` passes it to SB3's `load` (policy_loading.py line 562), and SB3 2.9.0's
`BaseAlgorithm.load` applies passed keyword arguments after the archive's data (base_class.py lines 738-739)]. Option
B's "parent path kept" was observed with `use_tensorboard=False`. A local smoke reaches the `gcs_tb_path` branch, not
Colab's buffer branch; both are the same keyword override, and the first hour of each run checks the buffer branch.
Measured since: the smoke's child wrote its events under its own `03_locomotion/tensorboard/PPO_1`, and every child
archive records the child's own `tensorboard_log` [measured: SP/work/pilot3/smoke/out/smoke_evidence_launch_s42.json],
so no such commit is made. (The re-smoke at a3f1b378 measured the same, SP/work/pilot3/r6_validate/RESULT.md
section 4.)

## Not on the branch (the session's scratch, frozen by sha256 before launch)

Pilot 1's tools were never on its branch; byte-identical copies went to `main` later. Pilot 3 does the same.

FIX3 note: `pilot3_habit.py`, `pilot3_score.py`, `pilot3_vg.py`, `pilot3_noise.py` and their tests changed again at
FIX3 (FIXES3.md); the sha256s in the table below are earlier stages'. The current ones are in the criteria ("How it
is judged") and in SP/work/pilot3/tools/SHA256SUMS.txt.

AMEND note (2026-11-04, CHANGES2.md): `pilot3_judge.py` hard-coded `WARM_START_LOG_STD = -1.5` and refused every
run trained at -2.0 (the re-smoke's ten judge steps, SP/work/pilot3/r6_validate/RESULT.md section 4);
`pilot3_noise.py` printed exp(-1.5) as its warm-start reference. Both now read -2.0 (sha256 `df5ea214...` and
`c191c72a...`), with `tests/test_pilot3_judge.py` (87 unit tests, 2 added that refuse -1.5), `tests/test_pilot3_noise.py`
and `tests/e2e_parent_cell.py`; the tests' tree `SP/work/pilot3/tools/tree` moved to `a3f1b378`; every test passed
on these files (tools `out/amend_final/`) and `SHA256SUMS.txt` was re-pinned (89 entries; the FIX3 list kept in
SP/work/pilot3/frozen_candidate2/tools_before/). No tool hard-codes the head (`--expect-head` is an argument) or the
recipe digest (derived from the tree); no other tool changed.

| Tool | From | Change | Status |
|---|---|---|---|
| `pilot3_judge.py` | `pilot_judge.py` `2dc44b6a...` (main copy equal to pilot 2's recorded hash) | `PILOT_BUDGET` 8,000,000; docstring and `--parent-sha256` help name the new parent; new refusals before the probe: `run.critic_phase_dropped` absent or not the parent's record, `run.warm_start_log_std` or `curriculum.warm_start_log_std` not -1.5 (-2.0 since AMEND), `run.n_envs` not 4, a `critic_phase` key in the recorded hyperparameters; `--final-fallback` (judge `stage2_final.zip` + `stage2_final_vecnorm.pkl` only when its num_timesteps equals the uninterrupted end, whole rollouts of 16,384 over the budget); `binding.json` schema bumped and carrying the two records; unit tests on synthetic run copies for the resume branch (budget = recorded timesteps + resume step, the off-schedule rule, the new refusals on a resumed copy) and for `--final-fallback` | built: sha256 `e962c289...`, 69 unit tests passed; FIX2: sha256 `66d238dd...`, 85 unit tests passed (the newest intact pair in the run-end record, git_commit required and 7-40 hex digits, one run per seed in OUT) |
| `pilot3_score.py` | `pilot_score.py` `b4297e13...` | `CHECKPOINTS` (2,000,000, 4,000,000, 6,000,000, 8,000,000), `PRIMARY` 8,000,000, the replay path's RUN_ID `pilot3_walk_s<seed>`; no label, guard, threshold or outcome rule changed (`seed_outcome`, `pilot_outcome`, `reviewed_outcome`, `pilot_outcome_reviewed` byte-identical bodies). Added in new functions: the existence reading, the final reading (`--closed-seeds`), the Q4 reading (`--habit`, pilot3_habit.py's JSON), "VIDEO REVIEW REQUIRED" for every WALKS-by-rule seed; report only: `speed_forward_vel_mps`, and touchdowns/s, alternation and yaw change for STILL episodes | built: sha256 `0529eaef...`; FIX2: sha256 `b343c917...` (a collapse-stopped seed read at its newest intact pair; one run per seed; "0 of 3" without "minority") |
| `test_pilot_score.py` | `f96b50f9...` | the lines that name checkpoints or the budget, plus tests of the added readings (minority WALKS, WALKS seeds disagreeing on habit, NOT RUN, closed seeds with 2, 1 and 0 scored) | built as `test_pilot3_score.py`, sha256 `8a6645de...`: 297/297, pilot 1's 254 among them, with the fixture recomputation; FIX2: sha256 `1cfa8062...`, 307/307 |
| `gait_probe_pilot.py` | `d7711c81...` | none | frozen |
| `gait_probe.py` | `61651b16...` (git blob `c3386473`) | none | frozen, in the tree |
| `fixtures/gait_cases.json` | `24f51b51...` | none | frozen |
| `iso.py` | pilot 1's `addbfc79...` is lost | rebuilt: run a script under `-I` with `ISO_WT` (the tree) first on the import path | built: sha256 `5da55763...` |
| `pilot3_video.py` | pilot 1's `c7079c1d...` is lost | rebuilt to pilot 1's contract: re-roll the panel through the probe's rollout, call a video the probe's episode only if every metric equals `probe.json`'s, real-time and quarter-speed mp4 with time and L/R floor boxes | built: sha256 `43f72bfa...` |
| `pilot3_habit.py` | the PC round's ledger reader (M5), M1 trace, M2 posture | the criteria's "Habit": groups, window, last-written row set after a resume, UNRESOLVED and POSSIBLE ALARM, the 20-episode trace (PARK, DRIVEN), NOT RUN; writes the JSON `pilot3_score.py --habit` reads | built: sha256 `1ade1d9b...`; validated on synthetic ledgers, the PC round's six stance ledgers (`SP/work/fetch/stance7_r10LTSB{off,pc}_s{42,60,80}/01_stance/eval_episodes.csv`) and the parent's trace; **frozen before launch**; FIX2: sha256 `eab5f398...` (the seed mode refuses a dirty tree, an off-schedule or validation-only cell and another run's ledger), 34 unit tests passed |
| other readout tools | the V - G tool (`vg_sigma.py`), the noise readout | locomotion forms: gamma 0.995, stock critic, the buckets and the per-episode paired SE in the criteria | FIX2: built as `pilot3_vg.py` (sha256 `b633fbd1...`, 25 unit tests, a run on the smoke's 65,536 pair) and `pilot3_noise.py` (sha256 `e2a2fd58...`, 7 unit tests; its TensorBoard reader equals TensorBoard's on the smoke's three event files); report only; pinned before launch |
| `nb_pilot_runner3.py` (smoke only, not a judging tool) | pilot 1's `nb_pilot_runner.py` `cc87f44f...` (lost), as pilot 2 extended it (`main` `pilot2/criteria_2026-10-21.txt` lines 395-405) | runs the Colab steps' cells locally on a fresh clone (guard line, Run all, then the manual cell); in-memory override of the locomotion stage's curriculum `eval_freq` and `checkpoint_freq` to 32,768 where `train()` reads the stage config; a second run stopped after its 32,768 pair for the RESUME check; needs a venv with tensorboard | built: sha256 `7f533128...`, frozen before the smoke and recorded by it |

Lost tools [measured: a whole-filesystem name search for `pilot_video.py`, `iso.py` and `nb_pilot_runner.py` on
2026-11-03 found none; `main` holds neither, `git ls-tree` of `docs/investigations/stance_2026_10/pilot/`]. All three
are rebuilt (`iso.py`, `pilot3_video.py`, `nb_pilot_runner3.py`).

Why the smoke needs the runner's override [measured by the code review]: QUICK_TEST changes only the budget (50,000)
and the run tree (PR #658 notebook cells 6, 8, 15); commit 1's `_stage_cadence` replaces `train_stage`'s
50,000/100,000 with the frozen 131,072/500,000 (`176af89`, top of `_build_core_callbacks`), so a plain 50,000-step
smoke writes no `stage2_<steps>_steps.zip` (the only file `pilot_judge.py` reads) and runs no evaluation (no ledger,
no Eval line). With the override the smoke ends at 65,536 with pairs and evaluations at 32,768 and 65,536. The
override enters no digest (`config._hyperparameter_view` digests only the `warmup_`/`ramp_`/`warm_start_` curriculum
keys, config.py lines 638 and 654-663), so the judge still accepts the run; it shows in `stage_config.json`'s
curriculum block. Fallback if the override cannot reach `train()`: `MANUAL_TIMESTEPS = 500,000`, no override.

## Order of work

1. After PR #658 merges: build commits 1, 2 and 3 on the merge commit; run the checks above (the
   full suites at commit 1 with nothing deselected; the known failing set measured at commit 2).
2. Build the scratch tools; re-run `test_pilot_score.py` (adapted) and `test_pilot_parent.py`; run
   `pilot3_judge.py`'s unit tests (resume branch, `--final-fallback`); validate `pilot3_habit.py` (synthetic ledgers,
   the PC round's six stance ledgers) and freeze it.
3. Notebook smoke on a local CPU host (not Colab), a fresh clone of the branch head, a venv with tensorboard, the
   rebuilt runner with its cadence override, the real parent: the two "Warm start" lines (re-measured if the merged
   `train_base.py` or `critic_phase_ppo.py` differs from cab70394); `stage_config.json`'s records (run.n_envs 4, the
   override in the curriculum block); the TensorBoard path; an "Eval num_timesteps=32768" line; the ledger's columns;
   the final replay and its reset seed. Then the smoke RESUME run (stopped after 32,768, resumed by the RESUME cell).
4. Judge the smoke's 65,536 pair (validation only), the refusals (pilot 1's parent, no drop record, no reset record,
   n_envs not 4, another head), the smoke RESUME (off-schedule) and `--final-fallback`; the probe on the parent
   (40/40 STILL expected; anything else holds the launch for the maintainer); the scripted r10 rollout through the
   probe; the habit trace on the parent; the video tool's equality check.
5. Fill every `<TO FILL>` in the criteria; the maintainer's choices (8,000,000, no futility stop, the rest as
   written; decision (e)) are applied and the options not chosen deleted; freeze; the maintainer approves; push
   (maintainer's OK); write the Colab steps (pilot 2's `colab_steps.txt` as the model, with the new paths, the two Warm start lines and
   `CRITIC_PHASE = False`); launch all three seeds.
6. (2026-11-04, decisions (8)-(11), inserted before step 5's approval) Design R6 and the -2.0 reset
   (SP/work/pilot3/r6_design/); build commits 4-6 on `3ae1c959` (SP/work/pilot3/r6_impl/); validate: R6 in the
   head's env against SCREEN, the stance path untouched, the parent at -2.0 under R6, the re-smoke at the new head
   with its judge steps (SP/work/pilot3/r6_validate/); amend the frozen text and re-pin the judge and noise tools
   for -2.0 (SP/work/pilot3/frozen_candidate2/); two reviews (code; text and process) and their fixes.
7. (2026-11-04, after the R6 reviews; FIX4, SP/work/pilot3/r6_fix/) Commits 7 and 8; the smoke re-run at the new
   head with its judge steps; the tools' tree moved there, the tools re-run and re-pinned; the text fixed. Then step
   5's approval of the text, the eight commits and the departures named under the criteria's decision (9), the push
   (the remote branch must read `31d02e42...`), the Colab steps (the second Warm start line "-2 (std 0.135)") and the
   launch.

## Risks for the builder

- The cadence port touches `train_base.py`, which PR #658 changed by +374 lines; a hand port must not move any
  off-path behaviour (PR #658's off-path tests are the guard).
- The known failing set must be measured, not taken from the list above.
- The smoke runner's in-memory cadence override must reach the stage config `train()` reads; `stage_config.json`'s
  curriculum block (32,768) is the proof. If it cannot, the fallback (`MANUAL_TIMESTEPS = 500,000`) is chosen and
  recorded before the smoke runs.
- If `main` moves between the #658 merge and the build, the base stays the merge commit unless the maintainer says
  otherwise; the criteria name it.
- (2026-11-04) R6's code sits in the T. rex env class every T. rex stage uses, stance included: off, it must stay
  bitwise the old path (the golden test of commit 4; the validate stage's stance panel and bit-exact reward captures,
  identical at 3ae1c959 and a3f1b378). The 11 T. rex behavior recipes inherit R6 through locomotion's `[env]`; any
  later use of them on this branch trains on R6.
- (2026-11-04) The judge and noise tools carried -1.5; they were re-pinned for -2.0. A later change of the reset
  value needs the same re-pin before any judging.
