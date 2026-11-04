# Walking pilot 3, DRAFT stage: notes (revised in the FIX stage after the design and code reviews)

FIX stage, 2026-11-03 about 14:17-14:40 UTC: this file, `criteria_pilot3.txt` and `BRANCH_PLAN.md` were copied from
SP/work/pilot3/draft and revised for every review finding; each finding's disposition is in FIXES.md, the diffs
against the draft in diffs/, the FIX stage's commands in this folder's CMDLOG.txt. Sections 1, 2 and 6 below are the
drafter's (section 6's heading now points to FIXES.md); sections 3-5 carry the FIX stage's edits, marked "(FIX)".
The drafter's "D" (SP/work/pilot3/draft) still names the draft folder, where D/src, D/checks and D/tools remain.

FIX2 stage, 2026-11-03 from 20:33 UTC (FIXES2.md beside this file): the report-only tools built (pilot3_vg.py,
pilot3_noise.py), the judge, scorer and habit tool fixed and re-pinned, the criteria's text fixed; nothing on the
branch changed (head 3ae1c959). Items for the later records PR are at the end of this file.

FIX3 stage, 2026-11-03 from 21:40 UTC (FIXES3.md beside this file): after a third code review, the habit tool's seed
mode reads only the judged run copy's own ledger file and only when its window covers the deciding checkpoint; tests
added for the V - G tool's frozen settings and refusals; the noise readout prints an unreadable input "not run"; the
scorer reads a validation-only habit record as NOT RUN; no report tool overwrites an existing --json file. Tools
re-pinned, the criteria's sha256s updated; nothing on the branch changed (head 3ae1c959).

AMEND stage, 2026-11-04 from 16:01 UTC (CHANGES2.md beside this file): the maintainer's decisions (8)-(11) of
2026-11-04 11:57 UTC [measured: SP/work/run_log.txt line 1034] (-2.0 reset; the walking reward R6 as default-off
code, pilot-only; the gate and the inputs and plant unchanged) are recorded in the three files; the branch's commits
4-6 (head a3f1b378) and the measurements of the design, implement and validate stages (SP/work/pilot3/r6_design,
r6_impl, r6_validate) are written in; pilot3_judge.py and pilot3_noise.py and their tests are re-pinned for -2.0.
Points the amendment changes carry "(AMEND: ...)"; new inferred points are I30-I33.

FIX4 stage, 2026-11-04 from 17:27 UTC (FIXES4.md beside this file; SP/work/pilot3/r6_fix): after the R6 code and text
reviews, two pilot-only commits (7: a test that a reset starts R6's touchdown event afresh on one env, the
integration test's checks by kind, and corrections of commit 4's message; 8: two comment lines of locomotion.toml),
the smoke re-run at the new head, the tools' tree moved there and the tools re-run and re-pinned, and the criteria's
HOPS reading, R6 limits, decision (9) departures and two line citations corrected. I33 is rewritten; points FIX4
changes carry "(FIX4: ...)".

FINALISE stage, 2026-11-03 19:42-20:02 UTC (FIX2 correction of "about 19:42-20:20": the last command ran at 20:01:56):
copied from SP/work/pilot3/final into SP/work/pilot3/frozen_candidate
with the maintainer's final choices applied (14:53 UTC [measured: SP/work/run_log.txt line 1005]: budget 8,000,000,
futility stop F0, the habit rule and every remaining default as written) and the 4,194,304 variant and futility stop
F1 deleted. Points the pre-launch checks have since measured carry "(FINALISE: measured since ...)"; section 4 marks
each decision. Every change is in CHANGES.md beside this file; the diffs against final/ are in diffs/.

2026-11-03 (UTC), about 13:33-14:10. Drafts only: no training, no git write, no push, no PR, no Drive access. Small
CPU checks only (pilot 1's own tests on pilot 1's fixtures). Branch names are written without their agent prefix.
"SP" is the session scratchpad folder named in the task; "D" is SP/work/pilot3/draft (this folder); "W" is the
read-only worktree SP/wt_optB, used only through `git -C W` reads of refs.

Labels: **[measured: source]** = read or computed from the named file; **[inferred: basis]** = reasoning.

## 1. Sources read

Git refs (read with `git show`, `git log`, `git ls-tree`, `git grep`, `git rev-parse`, `git merge-base`,
`git archive`, `git cat-file`; nothing written to the repository):
- `origin/main` 4c9d5e9b: `docs/investigations/stance_2026_10/pilot/criteria.txt` (pilot 1's criteria, sha256
  1cdd495c...; copy in D/src/pilot_criteria.txt), `pilot_judge.py`, `gait_probe_pilot.py`, `pilot_score.py`
  (constants and scoring functions), `make_gait_cases.py` and `test_pilot_score.py` (headers); `pilot1_judging/
  run_log.txt`; `pilot2/criteria.txt`, `pilot2/criteria_2026-10-21.txt` (read in full) and `pilot2/colab_steps.txt`;
  `docs/BEHAVIOR_RECIPES_PLAN.md` (the D-D36, D-D45, D-D46 rows and the decision-number list); `docs/
  GAIT_QUALITY_PLAN_2026_09.md` (header notes, the gate table near lines 80-92, GQ-1..GQ-18, section 5.5, section 12's
  GQ-17 note, section 17); `configs/tyrannosaurus_rex/anatomical/locomotion.toml` and `stages.toml`;
  `docs/investigations/stance_2026_10/stance6/criteria.txt` (the saturation and alarm definitions, lines 128-192).
- PR #658 head `cab70394` (`phase-critic-default-off`): `git diff --stat` against main; `environments/shared/
  train_base.py` (the drop record, the log-std reset, the warm-start record, the VecNormalize carry, the callback
  cadence, `_prepare_alg_kwargs`); `critic_phase_ppo.py` (header); `policy_loading.py` (`load_sb3_model`,
  `load_sb3_checkpoint`); `environments/shared/base_env.py` and `environments/tyrannosaurus_rex/base_env.py` (every
  use of the step counter, the reward kwargs' defaults, truncation, the command controller); `config.py` diff;
  `configs/digest_snapshot.generated.txt` (T. rex lines); `notebooks/sb3_training.ipynb` (cells extracted with
  D/tools/nbcells.py); `.gitignore`; `.github/workflows/python-ci.yml` (the generated-file checks); the tests
  `test_critic_phase.py`, `test_curriculum_warm_start_log_std.py`, `test_critic_phase_drop.py` (grep and the pinned
  digests); the anatomical XML's actuator list.
- Pilot branches: `pilot-walk-r9` (eaeef50) and `pilot-walk-r9-b` (8c5478a): every pilot-only commit's message and
  file list; pilot 1's locomotion `[curriculum]` block; pilot 2's parent SOURCE.txt (first 60 lines).
- The run branch `8737cd3d` and the phase branch `1dfe2327`: their commit lists; `176af89` (the cadence commit) in
  full.

Files (read with the file reader or `cat`/`grep`/`sed`):
- SP/handoff/final/NEXT_STEPS.md (lines 1-190 and 380-572: the decision tree Q1-Q4, Option D, determinism).
- SP/work/optA/OPTION_A_REPORT.md (in full).
- SP/handoff/final/stance_solved/synth/ANSWER.md (in full).
- SP/work/run_log.txt (searched; lines 886, 994, 996-1001 read in full).
- SP/work/optB/WORKFLOW_RESULTS.txt (searched; lines 1-12, 46-47, 73, 76-80, 93, 126 read);
  SP/work/optB/impl/parent_evidence/out_warm_start.txt, out_warm_start_try1_tensorboard.txt (tail), warm_start.py
  (grep); SP/work/optB/review_semantics/out/parent_probe.txt (grep).
- SP/work/port/critic/NOTES.md and tool/vg_tool/SPEC.txt (grep: how V - G is defined).
- The SB3 2.9.0 sources in SP/work/optB/venv (on_policy_algorithm.py, dummy_vec_env.py, subproc_vec_env.py,
  callbacks.py: grep).
- The parent's files in SP/work/fetch/stance7_r10LTSBpc_s42/01_stance/models/ (sha256; the zip's JSON data member
  through D/tools/zipdata.py, which uses json and zipfile only and unpickles nothing).
- Not found: the PC round's criteria and score folders (`/home/user/work/pc_study/...`) are not on this machine
  [measured: `ls`, CMDLOG 13:4x]; numbers from them are quoted through the Option A report, NEXT_STEPS and ANSWER.md.

## 2. Checks run (CPU, small)

| Check | Result | Evidence |
|---|---|---|
| Parent files' sha256 | `1c9d87c3...` zip, `5e417fef...` sidecar: equal to the task's values; sizes 4,048,776 and 5,291 bytes | CMDLOG 13:4x; `ls -la` |
| Parent archive's JSON fields | num_timesteps 12,001,280; n_envs 4; plant physics r10 `6a47a0eb...`, interface r15 `ffbf6631...`; `mesozoic_critic_phase` (settle 200, gamma 0.98, width 3); `mesozoic_symmetry` mirror_kl; policy_kwargs with log_std_init -1.5; tensorboard_log `/tmp/tb_buffer/..._stance7_r10LTSBpc_s42_01_stance_tensorboard` | D/checks/parent_zipdata.txt |
| Probe file identity | `gait_probe.py` git blob `c3386473` on eaeef50, origin/main and cab70394; sha256 `61651b16...` | CMDLOG |
| Pilot 1's tool identity on main | `pilot_judge.py` 2dc44b6a, `pilot_score.py` b4297e13, `gait_probe_pilot.py` d7711c81: equal to pilot 2's recorded hashes; main and cab70394 blobs equal | CMDLOG |
| `test_pilot_score.py` on a git-archive export of cab70394 (numpy 2.5.4, Python 3.13.18, OMP_NUM_THREADS=1, TMPDIR inside D) | 254/254 passed; 50 synthetic rows recomputed through the probe's `analyze_episode` equal the fixture; the real walker 30/30 WALK | D/checks/test_pilot_score.log.txt (sha256 137323d7...) |
| `test_pilot_parent.py`, the same export | 40/40 passed (pilot 1's parent tool; not used by pilot 3) | D/checks/test_pilot_parent.log.txt (sha256 0857770b...) |
| `git apply --check -v` of `176af89` on the export | fails on `gate_schema.py` (hunk at 324) and `train_base.py` (hunk at 182); the test file applies | CMDLOG 13:44:53Z |
| Search for pilot 1's scratch tools | `pilot_video.py`, `iso.py`, `nb_pilot_runner.py`: none on this machine (a whole-filesystem name search, read-only, finished in the background) | CMDLOG; the background task's empty output |
| SB3 2.9.0 truncation bootstrap, learn loop, checkpoint condition | `rewards += gamma * V(terminal_obs)` when `TimeLimit.truncated` (lines 238-245); `TimeLimit.truncated = truncated and not terminated` (dummy_vec_env line 66); `while self.num_timesteps < total_timesteps` (line 323); `n_calls % save_freq == 0` (callbacks line 301) | CMDLOG |

Not run (outside this stage's allowance): the probe on any checkpoint (the parent or a child), the notebook smoke,
the judge, the digest snapshot, the species catalog check, the shared test suites. Each is a pre-launch check in the
criteria. (FINALISE: all since run by the branch, tools and smoke stages; the criteria's "Pre-launch checks" give
the measurements.)

## 3. Every inferred point

| # | Point | Basis |
|---|---|---|
| I1 | At 8,000,000 the run ends at 8,011,776 steps (489 rollouts of 4 x 4096) | SB3 2.9.0's learn loop (measured) with N_ENVS 4 and locomotion's n_steps 4096 |
| I2 | A periodic checkpoint exists at exactly 8,000,000 with checkpoint_freq 500,000 | stride 500,000 // 4 = 125,000 calls; call 2,000,000 falls inside rollout 489 (calls 1,998,849-2,002,944); SB3's `n_calls % save_freq` |
| I3 | 61 evaluations; earliest collapse stop 3,801,088 unchanged | eval stride 131,072 // 4; pilot 1's computation, same cadence and collapse keys |
| I4 | Cost: about 6.9-7.0 h per seed at 8M (21 L4-h), up to about 9.4 h (28 L4-h) | pilot 1's measured 3h36-3h41 for 4,194,304 (316-324 steps/s, its run log) and the stance rounds' 238-240 steps/s (NEXT_STEPS); r10 locomotion pace and today's per-evaluation telemetry are unmeasured. NEXT_STEPS assumed the stance pace for locomotion (9.5 h per 8M); pilot 1's measured locomotion pace is faster, and locomotion's batch 256 means half the gradient steps per sample of stance's 128 |
| I5 | (FIX: now labelled measured by code reading, not inferred; the code review confirmed it) On Colab the child writes TensorBoard under its own folder, not the parent's | `train_base._prepare_alg_kwargs` sets `tensorboard_log` whenever `use_tensorboard` (cab70394 lines 505-526; the notebook passes nothing, so train()'s default True), and `load_sb3_model` passes those kwargs to SB3's `load` (policy_loading.py line 562), which applies them after the archive's data (base_class.py lines 738-739). The task text and Option B's note ("a warm-started child keeps the parent's recorded tensorboard_log path") were observed with `use_tensorboard=False`. Even if the parent's path were kept, it names the parent runtime's local buffer `/tmp/tb_buffer/...`, so the child's events would be lost locally, not written into the parent's Drive folder. The local smoke reaches the `gcs_tb_path` branch; the first hour of each run checks the buffer branch; "possible commit 5" is very likely unneeded |
| I6 | The config commit moves exactly locomotion's hyperparameters PPO/SAC and stage_config_view PPO/SAC digest lines; task and gate unchanged | pilot 1's and pilot 2's commit messages; PR #658 adds `warm_start_` to `SHAPING_KEY_PREFIXES` (AMEND: measured since: it holds for commit 2 and for commit 6 (-2.0); commit 5 (R6 on) moved locomotion's task, view and reward lines and the 11 behavior recipes' task digests, gate unchanged; the messages of 4dfc7cb and a3f1b37) |
| I7 | The listed PR #658 tests fail on the branch | reading their asserts (they pin main's locomotion digest, assume no committed stage declares the key, or, for the off-path scenario, train from the committed configs against main's golden: `critic_phase_off_scenario.py` line 142); the full set is for the builder to measure. At commit 1 (cadence only) the suites should pass with no deselection |
| I8 | At 8M the species catalog stays current | pilot 1's catalog commit changed only the budget row (FINALISE: measured since: species_catalog --check current at each of the three commits, SP/work/pilot3/branch/out/c{1,2,3}/quick.txt) |
| I9 | The probe and fixtures apply on r10 and to this warm start | the labels read contact timing, duty, lag and speed; the probe measures L from the model's hip anchors (gait_probe.py line 445) and r10 changed the tail springs only (D-D44); judged episodes run the actor deterministically, so the dropped critic inputs and the reset noise never enter them; the expected L 0.877 m is pilot 1's value |
| I10 | No locomotion reward term is a hidden clock; the truncation is bootstrapped and is not a reward clock | code reading of the T. rex and shared envs (measured lines) and SB3's bootstrap (measured); "hidden clock" as NEXT_STEPS defines it (AMEND: true of main's locomotion reward only: R6's touchdown event reads each foot's time since its previous touchdown and, for its first, the time since the reset, which the observation does not carry; see I32) |
| I11 | The V - G design for locomotion (buckets, 32 stochastic episodes, gamma 0.995, the time-limit bootstrap, the 4 SE flag) | adapted from the phase screen's vg_tool SPEC (stochastic on-policy rollouts; `G_k` with `gamma^(T-k) V(s_T)` at a truncation); buckets chosen for locomotion (gait window from step 100; the last 200 steps are about 1/(1-gamma)) |
| I12 | (FIX: weakened) The warm-up's clip 0.02 limits how fast the actor's mean moves while the critic re-fits after the drop; it does not bound the noise: the warm-up's entropy term (0.02, four times production's 0.005) is not clipped and acts on log_std directly, so the noise can rise during the warm-up | production's warm-up (300,000 steps); Option B's design note "the child's warm-up re-fits the critic"; PPO's clipped surrogate and its separate entropy term (the design review) |
| I13 | The parent's Drive folder is `.../ppo/stance7_r10LTSBpc_s42/01_stance/models/` | the archive's recorded TensorBoard path names it; pilot 2's SOURCE.txt describes the same layout |
| I14 | (FIX: rewritten) The habit readout's adaptation: TD per 2,000,000 steps; NECK alarms on actuators 0-2 by the tail alarm's rule; the window is every evaluation through the deciding checkpoint, the warm-up included; the last-written row set after a resume; UNRESOLVED episodes and a POSSIBLE ALARM settled by the deciding checkpoint's trace; PARK (tail-up) and DRIVEN (either sign) over the full-horizon episodes of 20 deterministic episodes at reset seeds 3042-3061, window steps 100-999 | stance6's "CLASSES PER RUN" (measured in D/src/stance6_criteria.txt, the drafter's copy of main's: PARKED is tail-up only, +0.60, lines 276-277; DRIVEN, either sign, is report only there, line 279; TAIL-UNRESOLVED and "resolved by trace", lines 173-178), adapted: stance used M1's 20 episodes lasting past step 200 at R1 and R2 and resolution checkpoints near the alarm; locomotion has no step-200 settle and one deciding checkpoint. DRIVEN counts as a habit here because a habit-free Q4 YES should not hide a tail driven down |
| I15 | The locomotion evaluation ledger records the saturation, command and head-tip columns | the eval_diagnostics docstring (cab70394 lines 385-420) describes the ledger for every evaluation; only the v2 screen is stance-gated; the smoke checks it |
| I16 | The cadence commit alone moves no digest and changes no training | no committed config declares the keys at that commit; `176af89`'s message ("enter no gate, recipe or task digest") |
| I17 | Reset ratios about x3.9 (mean) and x32 (smallest) | 0.2231 / 0.057 and 0.2231 / 0.007, from the printed three-decimal values (AMEND: at -2.0, about x2.4 (mean), x19 (smallest) and x1.2 (largest): 0.1353 / 0.057, / 0.007, / 0.111) |
| I18 | Storage about 65 MB per seed for 16 checkpoint pairs | about 4.05 MB per pair (the parent's zip 4,048,776 bytes plus a 5 KB sidecar) |
| I19 | The September walkers moved about 0.19 and 0.29 m/s by 4.2M | pilot 1's estimate, quoted |
| I20 | The final replay's reset seed is 2002 + the training seed | pilot 1's record; not re-read in PR #658's code (a pre-launch check) (FINALISE: measured since: 2044 for s42, replay_seed = seed + 2000 + stage, SP/work/pilot3/smoke/replay/out2/replay_check.json) |
| I21 | 8M costs about 10-13 more L4-hours and 3-4.5 more hours of wall clock than pilot 1's 4,194,304 | I4's ranges against pilot 1's measured 3h36-3h41 per seed and the stance pace scaled to 4,194,304 |
| I22 | A resumed run leaves the byte-identical path | NEXT_STEPS section 3 (inferred there too) (FINALISE: measured since on CPU: the smoke's resume at 32,768 did not reproduce the unbroken run's 65,536 checkpoint bitwise, SP/work/pilot3/smoke/out/unbroken_vs_resumed_65536.txt) |
| I23 | `pilot-walk-r10-p3` passes pilot 1's branch check | `pilot_judge.py`'s `startswith` check on the prefixed name |
| I24 | (FIX) With the runner's override (eval_freq and checkpoint_freq 32,768), the QUICK_TEST node ends at 65,536 with pairs and evaluations at 32,768 and 65,536 | rollouts of 16,384 (49,152 < 50,000, so a fourth rollout); stride 32,768 // 4 = 8,192 calls (train_base.py lines 1048 and 1065 divide by n_envs); pilot 2's smoke ended at 65,536 with --checkpoint-freq 32768 (FINALISE: measured since: pairs and evaluations at 32,768 and 65,536, the final model at 65,536, SP/work/pilot3/smoke/out/smoke_evidence_launch_s42.json) |
| I25 | (FIX) Steps 4M-8M more likely refine a gait that has started than start one | pilot 1's flat trajectory 2.1M-4.2M (measured), the September walkers' 39-54 at 2.1M (measured), the entropy floor at 4,000,000 (measured); a reading, not a test |
| I26 | (FINALISE: deleted with futility stop F1; the maintainer chose F0, no early stop) | - |
| I27 | (FIX) The final-model fallback reads nearly the same policy as the deciding checkpoint | stage2_final is the 8,000,000 checkpoint's policy plus the last PPO update (SB3 saves the periodic checkpoint during the last rollout, before its update) |
| I28 | (FIX) A habit alarm during the warm-up is a habit the child carries in from the parent | the warm-up's clip 0.02 keeps the actor's mean near the parent's for 300,000 steps |
| I29 | (FIX) The scripted open-loop rollout can make touchdowns and lift-offs on r10 | not known; the open-loop T. rex may fall at once. The criteria allow "not run" and then say plainly that the WALK and HOP paths are unexercised on r9/r10 (FINALISE: measured since: touchdowns and lift-offs on both feet, 1-2 events per episode, every episode fallen after 85-268 steps, SP/work/pilot3/smoke/scripted/) |
| I30 | (AMEND) Under R6 a policy that turns to a slow hop or to falling after 3,801,088 steps can fall below half its peak evaluation, which the collapse backstop (keys unchanged) reads as a collapse | SCREEN's R6 returns of the hand-built hops at forward weight 2.0 (448-1614 per episode, against 1214-2228 under main's reward) and standing's about 1092 [measured as recorded]; the backstop's drop rule (more than 50% below the rolling-median peak, 10 consecutive evaluations) read from locomotion.toml; whether a learned policy does this is unknown |
| I31 | (AMEND) The scripted open-loop rollout and the video tool's re-rolls at a3f1b378 give 3ae1c959's episodes, so they were not re-run (FIX4: measured since, at 31d02e42 in the R6 env: every scripted variant's result equal to the smoke's at 3ae1c959, and the video tool's 40 re-rolled episodes equal to the judge's probe.json; SP/work/pilot3/r6_fix/out/extras/) | the reward does not enter the dynamics (VALIDATE part 1: 320 re-rolled trajectories under R6 bitwise SCREEN's under main's reward); with R6 off the env is bitwise 3ae1c959's (commit 4's golden test) |
| I32 | (AMEND) The V - G readout may show structure in [0,100) and [100,200) under R6 | the touchdown event's pay depends on each foot's time since its previous touchdown, and for its first on the episode step (capped at 120 steps), which the observation does not carry (code reading of `_WalkTouchdownEvents`) |
| I33 | (AMEND; FIX4: rewritten, the AMEND wording was false for fast hops) A STILL, or a HOPS whose hop is slower than 1.0 m/s, says SCREEN's offline ranking did not carry to what PPO found. A HOPS at 1.0 m/s or faster agrees with that ranking: R6 puts every hand-built hop below its speed-matched walk at both ends of the ramp, but below standing only at forward weight 0.2; at 2.0 (the training reward from 2,000,000 steps on) every hop at 1.0 m/s or faster out-earns standing, so such a HOPS says PPO found no walk it kept, not that the ranking failed. Neither says which part of R6 failed | SCREEN ranked fixed hand-built behaviours only (its "Limits"); in the head's R6 env at weight 2.0, over the best standing behaviour per seed, H24 +135.3 (20 of 20 seeds), HF120 +299.0 (20 of 20), H20 +521.2 (19 of 20), and SCREEN's H25 1491.8 and HF150 1519.9 against about 1092 (measured: r6_validate/out/v2, recomputed in SP/work/pilot3/r6_fix/out/hops_vs_standing.txt; SCREEN RESULT.md); training under R6 is unmeasured |

## 4. Points only the maintainer decides (recommended option first, with its cost)

FINALISE: decided 2026-11-03 14:53 UTC [measured: SP/work/run_log.txt line 1005]: item 1 8,000,000, item 3 F0, and
every other item as written (the first option), except item 9, which stays open for the records PR. The 4,194,304
variant (item 1) and F1 (item 3) are deleted below.

AMEND: decided 2026-11-04 11:57 UTC on the morning report [measured: SP/work/run_log.txt line 1034], each the
recommended option: (8) the reset is -2.0, replacing decision (c)'s -1.5; (9) the walking reward is R6, as new code,
off by default, pilot-only on this branch, with tests; (10) the gate stays as it is for pilot 3 (fixed later with
the gait-quality plan); (11) no change to the inputs or the plant. Item 9 below (the decision number) is still open.

1. **Budget:** DECIDED 8,000,000 (about 21-28 L4-hours, 7-9.4 h per runtime; production's schedule; no
   learning-rate or catalog edit).
2. **Resume rule:** DECIDED as written. The options were: pilot 1's (resume from 2,000,000 or later; restart under `_b` below it; one restart per seed),
   or the stance rounds' "no resume" (every loss restarts from the parent; up to about 7-9 more L4-hours per lost
   seed; keeps every run on one unbroken, reproducible path).
3. **Futility stop:** DECIDED F0, none: every seed trains to the budget (8,000,000).
4. **Evaluation cadence:** DECIDED as written. The options were: pilot 1's 10 episodes every 131,072 steps (needs the cadence port, commit 1; keeps the
   stance alarm definitions' 10-episode basis and pilot 1's pace), or production's 30 every 50,000 (no code on the
   branch; about 160 evaluations of 30 episodes, an unmeasured but larger evaluation cost; the collapse backstop can
   stop a run from about 1.45M; the alarm rule would need restating for 30 episodes; checkpoints at 100,000 with
   `max_checkpoints = 0`).
5. **Checkpoint cadence at 8M:** DECIDED as written. The options were: 500,000 (the deciding checkpoint at the budget) or pilot 1's 524,288 (decide at
   7,864,320 or adapt the judge to the final model).
6. **Seeds:** DECIDED as written. The options were: pilot 1's 42/52/60 (comparability with pilot 1) or new seeds (no known outcomes attached).
7. **The known failing tests:** DECIDED as written (14 ids measured at commit 2). The options were: deselected by id on the branch (pilot 1's precedent) or edited on the branch.
8. **The habit definition** DECIDED as written (the window includes the warm-up; DRIVEN either sign counts;
   a neck-only habit counts). (FIX: now a pre-registered secondary readout that picks the Q4 branch, never the
   outcome; its tool frozen before launch). Choices fixed in the text that the maintainer may change before freezing:
   the window includes the warm-up (or starts after 300,000); DRIVEN (either sign) counts as a habit (or stays report
   only, as in stance6); a POSSIBLE ALARM is settled by the deciding checkpoint's trace (or counted as a habit); a
   neck-only habit takes the "NO, habit back (neck only)" branch (or is report only, given nosedive_weight 0.5).
9. **The decision number** STILL OPEN, for the records PR: the number (main's highest is D-D58 [measured: BEHAVIOR_RECIPES_PLAN.md]; the PC
   round's unpushed records draft may take the next ones).
10. **(FIX) A launch fault caught before any training step spends the seed's one restart** DECIDED as written (the storage cell has
    made the RUN_ID's folder by then). Written so in the criteria; the code review asked the maintainer to confirm it.
    The alternative (such a fault re-uses the same _b slot without spending it) needs a rule for naming a third
    RUN_ID.
11. **(FIX) The final-model fallback** DECIDED as written, allowed (decide at stage2_final when the 8,000,000 pair is missing but the final pair
    is intact at 8,011,776): written as allowed (it reads the 8,000,000 policy plus one update and saves a 7-9 h
    restart); the alternative is "the seed is missing" (restart under _b if unspent). Needs `--final-fallback` in
    the judge either way it is kept.
12. **(FIX) A judge refusal for a binding check found only at judging** DECIDED as written [inferred: the run log's "remaining defaults: Accept all"; its list does not name this item]: it restarts the seed under _b if its restart is
    unspent (written so; the refusal precedes the probe, so no gait of the refused run is read); the alternative is
    pilot 1's "missing". Either way the final reading ends the pilot once every missing seed is closed.
13. **(FIX) The existence reading and Q4 on a minority** DECIDED as written (the minority WALK EXISTS reading): one video-agreed WALKS seed of 3 reads WALK EXISTS and Q4 by
    its habit, printed "minority: 1 of 3" beside the pilot outcome (written so, per the design review); the
    alternative is to read Q4 only from a WALKS pilot outcome.

## 5. Open items for the review stages

- (FIX: settled by code reading, I5) The TensorBoard question; the smoke and the first hour confirm it.
- Whether the probe's rollout of a child checkpoint on today's tree and r10 runs as pilot 1's did (not checkable
  here: no smoke run). (FINALISE: measured since: the smoke's 65,536 child, 40/40 STILL, L 0.8766 m.)
- The exact known failing set (I7). (FINALISE: measured since: 14 ids at commit 2, listed in the criteria.) Deselecting PR #658's off-path test on the branch is acceptable only because
  commit 1 is checked with the full suites first, before any config declares a pilot key.
- Whether to require the in-run statistics of the drop (train/value_loss in the first 300,000 steps) beyond report.
- The draft criteria were 375 lines (this file said 365, a slip the code review caught), against pilot 1's 291; the
  FIX stage's version is longer (its count is in FIXES.md): the parent, tool, readout, habit, final-reading and
  pre-launch sections are new; the guard evidence is cited by full sha256 and line range, not copied.
- NEXT_STEPS's "Before training a walking trunk, decide the actor's inputs (item 5)" is not part of the maintainer's
  decisions for this pilot; the pilot keeps the parent's 63 actor inputs. (AMEND: decided since, decision (11) of
  2026-11-04: no change to the inputs or the plant.)
- (AMEND) For the reviews, from the validate stage's findings [measured: SP/work/pilot3/r6_validate/RESULT.md,
  "Findings for the next stages"]: the 11 T. rex behavior recipes resolve to R6 through locomotion's `[env]` (their
  task digests and reward captures moved; commit 4dfc7cb discloses it; not a stance change; the maintainer may want
  to know); the evaluation ledger has no reward-term columns, so the design's "eval_episodes.csv gains the
  reward_walk_touchdown sum" does not hold (the term goes to TensorBoard only; the criteria say so); the
  TensorBoard tag count is not a fixed number (fall tags appear only when training falls).
- (AMEND) Not run at the new head: the held-out panel 3062-3081 in the head's R6 env, the scripted rollout,
  pilot3_video.py, the replay byte check, pilot3_vg.py and the full e2e (I31; CHANGES2.md). (FIX4: see the
  criteria's "Re-smoke at 31d02e42" for what ran at the new head.)
- (FIX4) For the approval, named explicitly in the criteria's decision (9): R6's down flag reads the touch sensors,
  not floor truth; the event pays at k+4, not SCREEN's k+3; and the 11 T. rex behavior recipes inherit R6 through
  locomotion's [env], although the branch rule says only locomotion.toml turns R6 on (accept, or pin the four keys
  off in the recipes' [env], which would be a branch change and a new head).

## 6. Rule slips (honest list; the drafter's. The FIX stage's own slips are listed in FIXES.md)

- **CMDLOG is not fully literal.** Several entries differ from the executed command in small ways: `echo`
  separators and some `2>&1` or `head` counts are missing, two grep patterns were narrowed in the executed command,
  the first entry abbreviates paths with "...", and two entries describe heredoc file writes in words
  (`D/tools/nbcells.py`, `D/tools/zipdata.py`) instead of quoting them. File reads, writes and edits made with the
  harness's file tools are not in CMDLOG (they are not shell commands); this file's section 1 lists the reads.
- **Model-name words in search patterns.** Two of my shell checks for model names (13:55 and 13:57 UTC) and one
  check of the log (13:59) wrote model-name words into CMDLOG.txt as grep patterns. The 13:55 entry's logged pattern
  also held the device-folder prefix (the executed command did not). I masked the three entries with the file-edit
  tool at 13:59-14:00 UTC and added a dated NOTE in CMDLOG. Later checks used the harness's search tool, which writes
  no file.
- **A null-device reference written to a file.** `git format-patch -1 --stdout 176af89` saved to
  D/src/cadence_176af89.patch included git's new-file header, which names the null device. Found by my own check at
  about 13:59 UTC; the file was deleted at 14:00:06 UTC (its sha256 569a6b1d... is in CMDLOG). No command, pattern or
  comment of mine spelled it.
- **Whole-filesystem search.** The `find /` for pilot 1's lost tools read directory names everywhere on the root
  file system, including under `/home/user/mesozoic-labs` and other sessions' folders; it opened and wrote nothing.
  It ran past the 120 s foreground limit and finished in the background.
- **The repository's object store was read.** Every `git -C W` command reads objects that live in
  `/home/user/mesozoic-labs/.git` (the worktree's shared store); all were read-only (no index refresh, no ref
  change). `git archive` exports were written only inside D.
- Copies of `main`'s documents and code (D/src, D/tree_pr658) sit in this folder as reading aids; they contain the
  repository's own text, including commit trailers that name the product.

## For the later records PR (FIX2; nothing here changes a rule or the branch)

Two wordings on the branch are not strictly true; editing either would move the frozen head 3ae1c959, so they stay
and the records PR says so [measured as recorded: the second code review]:
- Commit 2's comment in `locomotion.toml` says evaluation "never changes training". EvalCollapseEarlyStop reads the
  evaluations and can end the node, and the 131,072 x 10 cadence sets its statistics (the criteria's 3,801,088 floor
  comes from it).
- `pilot_checkpoints/stance7_r10LTSBpc_s42_final/SOURCE.txt` and commit 3's message say no code reads
  `pilot_checkpoints/`. That holds for the repository's code only: `pilot3_judge.py` (`parent_phase_record`) and
  `tests/_tree_facts.py` read the committed parent's JSON data member, and the judge relies on it.
(AMEND) The same holds at the new head a3f1b378: commit 2's comment is unchanged (commits 4-6 did not touch it), and
commit 6's SOURCE.txt still says no code reads `pilot_checkpoints/`. Commit 4dfc7cb's message records that the 11
T. rex behavior recipes inherit R6; if the records PR keeps them, it says so. (FIX4) Commit 8 moves the head again
but edits only the forward_vel_max comment the R6 text review named; the two wordings above are unchanged at
31d02e42 and stay for the records PR. Commit 7's message records two inaccurate wordings of commit 4's message (the
ledger, and "the probe's floor rule on the touch block" without saying that it replaces floor truth), corrected there.
Left as they are, by choice (FIXES2.md gives the reasons): the judge records the curriculum cadence keys without
checking them (the first hour reads them); its tree-clean check, like pilot 1's, ignores untracked files
(`--untracked-files=no`); lint rule B905 (`zip()` without `strict=`) still flags pilot 1's `video_watch`,
`score_panel` and `pilot_outcome_reviewed` (the last must stay byte-identical) and `pilot3_video.py`, whose bytes are
the ones the smoke's video check measured.
