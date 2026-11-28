# Triceratops robot port to the main pipeline (design record, 2026-11-27)

Status: this record alone is on `main` (commit 71ffe746); commits 2-5 continue on the same branch,
`claude/trike60-main-pipeline`, in a follow-up PR, merged with `main` a232da31 (§3). Base: `main` at 2c1039f4; every
`file:line` is at 2c1039f4 unless it names the build or the merge, except in §8, which was revised on 28 Nov 2026 for
a232da31 and cites the merged branch e94efee9 (its preamble). Main has no `hardware/` folder (removed in c42b6a93),
so build files are cited by commit, never by a branch path: build commit 8198001c (the build tip when commits 2-4 were
designed), f6deec9f (the commit that last wrote the source model), fb1e80c8 (the build reports §8 reads) or 743b5c75
(the decision log with D45-D56, §2). Short build paths (`rl/trike_env.py`, `rl/train.py`, `rl/README.md`) are under
`hardware/triceratops_build/` at 8198001c unless another commit is named.

Two kinds of statement appear here. **Design** is what the implementers build. **Measured on the prototype** is
simulation evidence from scratch prototypes of each part, run at 2c1039f4 and since deleted; none of it is on any
branch. §8 adds two more, **Computed** and **Estimate** (its preamble). An owner decision is a row of §2, cited by its
number. A pass is claimed only where a tool's output showed it.

## 1. Summary

- The 60 cm Triceratops robot becomes main-pipeline model `triceratops_horridus/robot`: the eighth model, seventh
  species; 12 leg servos, 59 observations (main's `quadrupedal-heading-free/v1`), 12 actions, 1 ms physics, 50 Hz
  control (D32, D36).
- It trains on main's stance stage, then main's command-driven walk at commands of 0.075-0.15 m/s (D51, D53). The
  stance gate is main's six-heading certificate with D40's 900 reward and 95 % full-horizon rails if a four-foot check
  in commit 5 passes, else D40's reward gate (900 over at least 950 of 1000 steps); the walk gate is main's shared
  `command_tracking/v1` (D52). The gates are provisional (D40); the owner reviews every stance bar, with the statue's
  measured value on it, before the first run (D54).
- Three shared features come first, opt-in and off for every existing model: a servo delay (0-20 ms drawn per
  training episode, 10 ms elsewhere, D33); physics randomisation (friction 0.6-1.0, per-body mass, servo kp x0.9-1.1,
  force limit x0.85-1.0, and whole-robot mass x0.9-1.1 plus a 0-1 kg payload in training only, D38); sensor noise.
- One rule decides who sees what: the trainer marks its rollout envs; every other env is an evaluation env (D37).
- No course and no arrival bonus: main has no target, and nothing ends a walk episode but a fall (D50; D39
  superseded).
- Nothing moves for any existing model: no digest, golden, plant manifest entry, task hash or checkpoint. No T. rex
  file changes. Five commits, each green on its own.

## 2. Owner decisions

Rows of the build's decision log (`hardware/triceratops_build/DECISION_LOG.md`): D32-D44 as recorded at build commit
ea3a6668 (27 Nov 2026); D46 and D49 (27 Nov 2026), D50-D56 (28 Nov 2026) and the status of D32, D39 and D40 as at
build commit 743b5c75, where D32 reads "amended (D53)", D39 "superseded (D50)" and D40 "amended (D52)". D41-D44 were
decided after commits 1-3 were written; commits 1-3 are unchanged by them. D50-D56 were decided after commits 2-4 were
built and the branch took main a232da31; they change only commit 5 (§8). D45, D47 and D48 (the no-spend checks, a
battery cut-off, the room temperature of the heat rules) are outside the port. Where this record gives D32's scope,
it uses D53's text.

| Row | Decision (D32, D39 and D50-D56 in the log's words; the others condensed from the log's question and decision) | How this design meets it |
|---|---|---|
| D32 (amended by D53) | **Also randomisation and noise**: the 60 cm robot becomes a main-pipeline model (Triceratops horridus, robot embodiment) on the main pipeline's own stance and walking stages, with the servo delay (D33), and mass, friction, servo-gain and force-limit randomisation and sensor noise as opt-in shared features, off by default for every existing model. Steering commands and heat costs stay in the build stack for now | D53's text governs (its row, below). §8 adds the model on main's stance and walk; §5-§7 add the features with `None`/`False` defaults; the walk follows main's whole-episode command (§8.6) and the recipes are main's own (§8.7), as D53 ports them; none of the build's steering or heat costs is ported (§11) |
| D33 | Delay opt-in for all models, off by default; on for the 60 cm robot with 0-20 ms drawn per episode in training and a fixed 10 ms in evaluation | §5: `actuation_delay_range_s = [0.0, 0.020]`, `actuation_delay_eval_s = 0.010` in every robot stage and recipe; drawn in trainer rollout envs only (§4.3) |
| D36 | One draft PR into main carrying only the pipeline port, on the new branch `claude/trike60-main-pipeline`: the model, its configs, the shared delay, randomisation and noise, and docs; the build tree stays on its branch, linked from main's hardware notes by commit | §3: five commits on that branch; no build file is copied or imported; the asset is derived by a script that names its source by commit (§8.1) |
| D37 | Evaluate as the build: per-body randomisation and sensor noise on in evaluation, the delay fixed at 10 ms, whole-robot mass and payload in training only; one shared rule (the trainer marks its rollout envs; every other env is an evaluation env) | §4.3: the `training_rollouts` flag and its table; per-body draws in every env built from the stage config; `observation_noise_eval = true` for the robot |
| D38 | Whole-robot mass factor (x0.9-1.1) and payload (0-1 kg): two more opt-in shared keys, off by default; on for the 60 cm robot in training rollouts; evaluation leaves them out | §6: `randomize_total_mass_scale` and `randomize_payload_kg` in the robot's stages and recipes, applied only while `training_rollouts` is true |
| D39 (superseded by D50) | Arrival bonus: on arrival the robot is paid the standing reward its remaining steps would have earned, so arriving is never worse than stalling; a change to the robot model only | Superseded by D50: no course and no arrival bonus (§8.5). The first design's `reward_arrival` (1.8 per remaining step, counting alive, posture, height and heading) is not built: main has no target, and a terminating arrival would be a fall to main's walk gate |
| D40 (amended by D52) | Starting gates: stance reward 900 over at least 950 of 1000 steps; walking reward 2000, 550 steps, 0.15 m/s; course 4 m, 0.3 m arrival radius; provisional, revised with the owner after the first main-pipeline run | As D52 amends it (§8.6): the stance takes D40's 900 reward and 95 % full-horizon episodes as the rails of `stance_quality_heading/v1` if commit 5's four-foot check passes, else D40's `reward_and_length/v1` (900, 950) until revised; the walk takes main's `command_tracking/v1` walk gate, with D40's 2000 and 550 report-only and its 0.15 m/s the fastest command (D51); the course goes (D50). The stage TOML comments, the species README and this record say "provisional (D40)"; a robot test pins the gates. The revision with the owner reaches the stance bars, reviewed before the first run (D54). This record reads D52, which puts the walk on main's shared gate, as leaving the walk bars to main: they move only with main's, and the revision reaches the walk only through D40's report-only 2000 and 550 (§8.6). D52 does not say this in words |
| D41 | The 60 cm design is adopted for building, and main's hardware page names it as the chosen design: option C of D2 (10 X4 and 2 X6 drives on 12S), the capsule sole (D19) and the 128 mm upper foreleg (D25), the model `trike60_C_closed_capsule_fore128.xml` at build commit f6deec9f. Adopting it orders nothing: the 21 Nov 2026 not-ready-to-order finding stands, the one-foreleg bench test comes before any order, and compute (D3) and the rest of the chassis (D5) stay open | §8.11: commit 5 updates `docs/hardware/README.md` to say so (owner point 1) |
| D42 | The robot ships all 11 behaviour recipes: they run only when someone starts them, and no shared test changes | §8.7: all 11 on main's single command contract, none exempted, inert (no committed walk can parent them, D-D62 (17)); D53 ports them as main's own command recipes that run only when started (owner point 6). One shared test change remains: the terrain-scale test hard-codes the control step (`frame_skip` 10 or 5 at the asset's timestep), so it must read the model's own `frame_skip` and timestep, which a 1 ms model needs whatever recipes ship (§8.9). This record reads D42's "no shared test changes" as set against the other option, exempting the two steering recipes; read strictly, the change is the owner's to confirm (§11) |
| D43 | Tanh now, ELU later: the robot's PPO uses SB3's default Tanh, a declared deviation from the build, and the first main-pipeline run is not a like-for-like comparison with the build's runs | §8.6 as written; the activation setting stays a follow-up (§11). D43 covers the activation only. Two further deviations from the build are this design's, the implementer's choices and not owner decisions: half the build's samples per update at main's 4 environments (§8.6), and the 15 inputs of main's schema the robot will not have (§8.4, D46's row) |
| D44 | Randomise in stance too, friction as the build's absolute 0.6-1.0, and a clean reset observation, as planned | §6 and §7 as written (owner points 4 and 5) |
| D46 | Train without the inputs the robot will not have (true body velocity, foot loads, passive ankle angles) first: a step sheet for one 10 M-step run on the robot's own sensors only, which the owner runs on Colab; foot sensors are decided after it | Outside the port: the run is on the build stack. Main's fixed schema keeps those inputs and adds the ankle speeds: 15 of the model's 59, recorded in §8.4 as a sim-only deviation, the implementer's choice (no owner row covers it); an own-sensors schema is a follow-up (§11) |
| D49 | One new draft PR now: main merged into `claude/trike60-main-pipeline` (no force-push, no rewrite), every gate rerun against the new main, then a draft PR carrying commits 2-4, with commit 5 added when built; marked ready only when all five are in and checked; this replaces D36's single PR with two, the record's and this one | §3: main a232da31 is in the branch (e94efee9) and every gate was rerun against it ("Merged with main a232da31"); commit 5 is added to the same draft PR when built, on §9's gates against a232da31; the PR is marked ready only when all five commits are in and checked |
| D50 | Drop both: no course, no arrival bonus; D39 no longer applies, since the truncation it fixed (an arrival that ended the episode unpaid) cannot happen on main | §8.3, §8.5: `SUCCESS_KEYS = ()`, `REWARD_PROBE = (0.2,)`, no course kwargs, no `course_complete` or `reward_arrival`; nothing ends an episode but a fall; the course and arrival tests go (§8.10) |
| D51 | 0.15 m/s: commands 0.075-0.15 m/s, `speed_scale` 0.225 (1.5 x cruise, as both Compsognathus models); D40's 0.15 becomes the fastest command, not a floor (the gate passes walks of about 0.045-0.12 m/s by command); chosen for heat (the 128 mm policies' worst drive 0.70-0.83 of rating at 0.15 against 0.83-0.95 at 0.25); main's pipeline does not train the 0.25 m/s design walk | §8.6: the walk's `[commands]` and `follow_direction.toml` at cruise 0.15 m/s, `speed_range` [0.075, 0.15] (mean 0.1125 m/s), `speed_scale` 0.225; tolerances 0.030 and 0.015 m/s, window 1.10 s; §8.7: the recipes' terrain for that cruise. The choice does not rest on 0.15 as a floor: D51's recorded reason is heat, and this design adds main's practice in support of it (§8.6, "Why 0.15 m/s") |
| D52 | Heading certificate and main's walk gate: stance on `stance_quality_heading/v1` with D40's 900 reward and 95 % full-horizon episodes as its rails, if a four-foot prototype check in commit 5 passes, otherwise D40's `reward_and_length/v1` until revised; the walk on main's shared `command_tracking/v1` walk profile with the quadruped clause; D40's walking numbers (reward 2000, 550 steps) become report-only | §8.6 (the gates) and §8.12 (the check): commit 5 runs the check, declares the kind it passed or the fallback, and says which; the walk declares `PROVISIONAL_WALK_THRESHOLDS` and the quadruped clause; the 2000 and 550 are reported from the statue and certification panels, not gated. Under the fallback, what "until revised" leads to (the D-D74 heading certificate) is this record's reading, and whether the first stance run waits for that revision is open (§8.6, §11) |
| D53 | Amend D32: main's whole-episode walking command, and main's own command recipes that run only when started (D42), are ported; the build's steering (its turning schedule, turning on the spot, its heading estimate) and its heat costs stay in the build stack | §8.6 (the walk's command), §8.7 (the recipes), §8.4 (no heading estimate: the yaw-rate command comes from the simulator's true heading); §11 keeps the build's steering and heat costs as follow-ups. This is the D32 text this record uses throughout |
| D54 | Review, then run: before the first stance run the owner reviews the stance bars as measured on the do-nothing statue in simulation; the run is published as main's chain publishes it; a later revision costs nothing after a failed stance, and a stance retrain (and the walk's) after a passed one | §8.6 ("The first run") and §8.12, item 2: commit 5 sets out for the owner every bar of the stance's kind with the statue's measured value on it (each marked copied, measured or D52's rail, the collapse floor's keys included) and the open question of which optional bars a four-foot stance takes; the first stance run is a certification run, published as the chain publishes it |
| D55 | Measure, then decide: commit 5 measures the robot's speed; if a stance at 4 environments would not fit one session, the robot's README asks for 8 on a runtime with 8 or more CPUs, as the build's runs used | §8.6 ("Training cost") and §8.12: measured against main's 19h31m for the Compsognathus robot's 11M stance at 4 environments; the README gives the measured time and, if needed, asks for 8. At 4 environments an update takes 4 x 1024 samples, half the build's 8 x 1024: a deviation this design declares and does not pin, the implementer's choice, not an owner decision (§8.6) |
| D56 | Separate model page: the robot's generated model page beside the hand-written Triceratops card; two Triceratops cards on the animals list; no website test changes | §8.11: the generated `triceratops_horridus` entry and route are the model's page, linked from the hand-written page; no website test changes; it replaces owner point 6's "one animal page" |

## 3. Landing order

One commit each, in this order; each passes every gate of §9 on its own. Later commits are review fixes only.

| Commit | Contents and files (`environments/shared/` unless a path is given) | What moves |
|---|---|---|
| 1 | This record and its row in `docs/README.md` ("Plans & designs") | nothing |
| 2 | Hook, stream helper, role flag (§4) and actuation delay (§5): new `episode_start.py`, `actuation_delay.py`; `base_env.py`, `train_base.py`, `train_behaviors.py`, `task_fingerprint.py`, `behavior_checkpoint.py`, `symmetry.py`, docstrings of `harnesses/digest_snapshot.py:493` and `scripts/warm_start_noise_scan.py`; `docs/PLANT_CONTRACT.md`, `docs/ROADMAP.md`, `CONTRIBUTING.md`, `CHANGELOG.md`; tests (the SB3 ones in `tests/test_train_base.py` and `tests/test_behavior_checkpoint.py`, §5) | nothing |
| 3 | Domain randomisation (§6): new `episode_physics.py`; `base_env.py`, `behavior_env.py`, `task_fingerprint.py`, `behavior_checkpoint.py`; `configs/species_manifest.toml` and the catalog; docs, with a status note in `docs/hardware/SIM_TO_REAL_PLAN.md`; tests. `plant_contract/validation.py` is not touched | the `domain_randomization` status in `website/src/data/species.generated.json` only |
| 4 | Observation noise (§7): new `observation_noise.py`; `base_env.py`, the `observation_segments()` refactor of `plant_contract/policy_layer.py`, `policy_probe.py`, `task_fingerprint.py`, `behavior_checkpoint.py`; docs, adding noise to the `SIM_TO_REAL_PLAN.md` note; tests | nothing |
| 5 | The model, its configs, tests, docs and website (§8, as revised for a232da31 and decided by D50-D56), built on the merged tree e94efee9: new `environments/triceratops_horridus/**` (species base, robot env, asset, derivation script, provenance, README, tests) and `configs/triceratops_horridus/robot/**` (stages, stance, walk, 11 recipes); `configs/species_manifest.toml`, `configs/plant_versions.toml` (note 25); the shared test tables and changes of §8.9, a second none-stage golden fixture among them; `pyproject.toml` testpaths, `python-ci.yml`'s species suite; docs, `docs/hardware/README.md` (D41) and the website (§8.11). First the measurements of §8.12 (the four-foot check that picks the stance kind, the statue-panel stance bars for the owner's review, the training speed, the live-model readers); then every generated file regenerated on that tree | only the robot's own lines, against a232da31: lines naming `triceratops_horridus` added to the digest snapshot and the `--exact` capture, the robot's entry added to both plant manifests (§9) |

**Must not move, in any commit:** the existing entries of both plant manifests (`configs/plant_manifest.generated.json`,
`environments/shared/data/plant_manifest.generated.json`); every existing line of
`configs/digest_snapshot.generated.txt` (725 at 2c1039f4, 663 at a232da31) and of the `--exact` capture (40,568 at
2c1039f4, measured on the prototype; 33,208 at a232da31); `tests/fixtures/phase_c_reset_golden.json` and its
`rng_state`; from the merge on, `tests/fixtures/none_stage_rollout_golden.json` and
`tests/fixtures/phase_d_pilot_rollout_golden.json` (commit 5 adds a second none-stage fixture beside the first, §8.9);
the T. rex mirror map; every existing stage's `task_sha256` and `stage_config.json` view; every existing checkpoint's
plant identity. No file under `environments/tyrannosaurus_rex/**` or `configs/tyrannosaurus_rex/**` changes.

**Why this order.** The model forwards the features' kwargs to `BaseDinoEnv.__init__`, which accepts none of them at
2c1039f4 (`base_env.py:217-237`). Measured on the prototype: the model's patch alone fails at construction with
`TypeError: ... unexpected keyword argument 'actuation_delay_range_s'` (M1). So the model lands last, and its
generated files come only from the tree that has all three features.

### Merged with main a232da31 (27 Nov 2026)

Commits 2-4 were built and reviewed on 2c1039f4; then `main` at a232da31 (41 commits after 2c1039f4, this record's
commit 71ffe746 among them, with the target-free revision D-D59 and the live commands of D-D60 to D-D64) was merged into
the branch. The `file:line` references elsewhere in this record stay at 2c1039f4, but for §8's, which are on the
merged tree. The merged code differs from the text above in these places (locations on the merged tree):

- **Reset draws (§4.2).** Main's `reset` (`base_env.py:2271-2342`) draws from `np_random` only the joint and
  velocity jitter and the push-schedule seed: the height draw and the target spawn went with D-D59, and a live
  command draws from its own stream (`_command_rng`, `base_env.py:1480`, keyed by
  `command_frame.COMMAND_STREAM_DOMAIN`, D-D60). The features still never draw `np_random`, and the branch still
  leaves `reset`'s source alone; its `super().reset(seed=seed)` is at `:2277`. `episode_start.py`'s docstring names
  the same draws.
- **The step (§5, §7).** Main split `step()` into `step()`, which holds a live zero command's stand weights
  (`_stand_block`, D-D62), and `_step_body()` (`base_env.py:1903-1917`). The delay's FIFO, its info keys and the
  noise are in `_step_body` (`:1959`, `:1993`, `:2063`, `:2094`), the noise after main's command recompute as before,
  so a stand-block step is delayed and noised like any other. Tests pin it: on the Compsognathus robot's live walk,
  whose stage declares stand weights, `set_direction(heading, 0)` makes the command a live zero, and the next step,
  under the stand block, hands its target to the plant k physics steps late with the delay's info keys
  (`test_actuation_delay.py`) and returns its observation with one draw of the noise (`test_observation_noise.py`).
- **`set_direction` (§7, built in commit 4).** Main moved it from the behaviour env into `BaseDinoEnv` (1c224969,
  D-D60), where every live command mode has it. The noise is applied there now (`base_env.py:1528-1546`), so a live
  walk's `set_direction` is noised as a behaviour env's is; `behavior_env.py` no longer defines it. Its test runs on
  a behaviour env and on the Compsognathus robot's live walk, at a cruise speed and at zero, and checks the step
  after it too.
- **The segment table (§7).** Main removed both legacy schemas and the target segments (D-D59,
  `configs/plant_versions.toml` note 24): every model declares `bipedal-heading-free/v1` or
  `quadrupedal-heading-free/v1`. `observation_segments(model, env, observation_schema)`
  (`plant_contract/policy_layer.py:373`) returns main's `_heading_free_observation_segments` for the schema's root
  label and refuses any other schema with `PlantContractError`, which a test checks for both legacy target schemas
  and an unknown one through the table, `observation_layout` and `resolve_observation_noise` (both take an explicit
  schema); the payload keeps main's schema check, its no-mocap check and its width check, and records the table
  `observation_segments()` returns (`:408`). No digest moves.
- **The layout (§7).** `observation_layout` (`observation_noise.py:154`) takes every width from `_segment_width`,
  `foot_contact`'s included: on the heading-free schemas that segment carries its channel count, touch-sensor groups
  or, on the Compsognathus robot, main's floor-contact geom groups (`_foot_observation_floor_geoms`). The
  Brachiosaurus case of eight touch sensors in four channels went with the legacy table.
- **Unit segments (§7).** The rule (`UNIT_SEGMENT_SUFFIXES`) and `OBSERVATION_NOISE_IMPLEMENTATION` are unchanged,
  but on every current schema the root's projected gravity is the only unit segment. The std test's two models are
  both heading-free now; the unit-segment test noises the projected gravity or leaves it out; the probe test feeds
  the Compsognathus robot's floor-contact channels through its reader and perturbs no mocap target. The noise tests
  name only the heading-free schemas' segment kinds, so a segment of a new schema fails their kind lookup until it is
  named there; the probe test checks that the orientation sensor and the root position feed no segment now.
- **Who sees what (§4.3, §6 R15).** On a live-command stage, every walk since D-D62, `StageAwareEvalCallback`
  reseeds the evaluation env before every evaluation (`eval_diagnostics.py:1556`, `curriculum/command_evaluation.py`),
  so there each evaluation round rolls the same per-episode feature draws (plants, delay, noise), fixed by the
  evaluation seed; on a command-free stage each round still sees new ones. Main's new env builders, the
  command-tracking panel (`command_tracking_panel.py:303`) and its report (`reporting/command_tracking_report.py:136,
  148`), build evaluation envs, and the warm-start noise scan, which now sets the cruise ramp's start command, stays
  one; two more, `_env_command_manifest` (`harnesses/digest_snapshot.py:414`) and the push probe of the
  off-distribution recovery panel (`harnesses/recovery_offdist_panel.py:120`, `multiple_for_force`), only construct
  an env and never reset it, so no feature draws there. Main added no trainer path: the marking call sites are those
  of §4.3. Main also changed the fixed-seed panels of §6 "Draws": the hunt's publication-seed panel (after
  `eval_env.seed(PUBLICATION_SEED_START)`) went with the `task_success/v1` gate (3368abe2, D-D59). A walk's gate is
  judged after its training on the `command_tracking/v1` certification panel from seed 3042
  (`PUBLICATION_SEED_START`), and its handoff is chosen by `choose_command_handoff`
  (`reporting/command_tracking_report.py:571`) on a selection panel from seed 2042 (`HANDOFF_SELECTION_SEED_START`),
  not by the EvalCallback env of §4.3's "Checkpoint selection, curriculum gate" row. Both panels are evaluation envs
  reset with one seed per episode (`command_tracking_panel.py:329`), so under the features they are paired per seed
  as §6 says of fixed-seed panels. The stance and recovery panels stay as §6 describes them.
- **Tests (§5, §6).** No plant has a mocap body, so the draw block on the Compsognathus robot is `3 + 19 + 24`, the
  pool-shape refusal names 20 bodies, and the per-body test keeps the world's mass; that robot's committed stage
  lines in the digest snapshot are 21 (stance, recovery and locomotion: main removed its behaviour stage), not 28;
  main removed `SpeciesConfig.behavior_section_label`, so the SB3 marking test's species double drops it.
- **Baselines at a232da31.** The `--exact` capture is 33,208 lines (sha256
  dfc81228eb3459853ca4174ab39d3599961cebadf8a73fa91e96569a5a4c2fc9), and the merged tree's is byte-identical;
  `digest_snapshot --check` is current at 663 lines. The full suite runs 9,187 tests on `main` and 9,661 on the merged
  tree (474 more), with no failure or error on either (48 skipped on both). From the merge on, §9's gates compare
  against the `main` the branch was last merged with, a232da31 now: read that sha for 2c1039f4 in gate 2's `git show`
  and `git diff --exit-code` and in gate 3's base worktree, which gate 4's base run uses too. Against 2c1039f4,
  where main has since moved the generated files and the capture, gate 2's `git diff --exit-code` lists all three
  generated files and gate 3's diff is not empty: a false red. The `--baseline` revision check passes against either
  sha, as main's plant changes carry their revisions, but CI runs it against the PR's base sha
  (`python-ci.yml:147`), a232da31 while `main` stays there. Against a232da31 gate 2's diff is empty, the snapshot is
  663 lines and gate 3's two captures are the 33,208 lines above, with an empty diff. The merge also re-explains its
  `base_env.py` edit in `tests/test_assemble_recovery_profile.py`, which pins that recorded source of the committed
  Compsognathus recovery measurements, and names no PR in this record's status line or in this subsection: a
  landing is recorded only in the consolidation plan's status table and the CHANGELOG (the one-landing-record rule,
  `docs/README.md`; `tests/test_landing_records.py`).
- **Commit 5 (§8), not built.** Its first design predated main's revision: a target mocap body is refused by the
  plant contract (`_require_no_mocap_body`), a target schema has no place beside main's `quadrupedal-heading-free/v1`,
  every walk follows a live command and is judged by `command_tracking/v1` with a course that ends nothing (D-D61,
  D-D62), and `SpeciesConfig.behavior_section_label` is gone. §8 is now revised against a232da31 on the merged tree
  (28 Nov 2026), its owner choices decided by D50-D56 (§2), and commit 5 is built from it. Its gates compare with
  a232da31: only lines naming `triceratops_horridus` may be added to the digest snapshot and the `--exact` capture,
  and only the robot's entry to the two plant manifests; nothing else in them moves (§9).

## 4. Shared pieces

### 4.1 The episode-start hook (commit 2)

New module `environments/shared/episode_start.py`, not fingerprinted:

```python
class EpisodeStartHook(gym.Env):
    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)  # gymnasium seeds np_random
        self._begin_episode(seed)
        return result

    def _begin_episode(self, seed: int | None) -> None:
        """Where each opt-in feature begins its episode. Nothing by default."""
```

`class BaseDinoEnv(gym.Env, ABC)` (`base_env.py:70`) becomes `class BaseDinoEnv(EpisodeStartHook, ABC)`. Its own
`super().reset(seed=seed)` (`base_env.py:1603`) then lands in the hook, before the keyframe is loaded (`:1612`).
`BaseDinoEnv.reset`'s source does not change; the policy layer hashes it as `home_reset` by qualname and tokens
(`policy_layer.py:392-393`, `digests.py:154-166`), and nothing hashes class bases, so no digest moves. No species
overrides `reset`; the behaviour mixin's `reset` (`behavior_env.py:424`) selects the pool model (`:460`) before
`super().reset` (`:476`), so the hook acts on the live pool model. `BaseDinoEnv._begin_episode(seed)` is the one place
each feature begins its episode, in this order, each behind one `is not None` check on a class attribute that
defaults to `None`: `self._episode_physics.begin_episode(self, seed)` (§6), `self._begin_actuation_delay_episode(seed)`
(§5), `self._begin_observation_noise_episode(seed)` (§7). With every feature off it calls nothing, reads no generator
and writes nothing.

### 4.2 One stream policy (commit 2)

Each feature has its own PCG64 stream; none ever draws from `np_random`, so the reset draws (joint, qvel, height,
target, push schedule, command: `base_env.py:1617-1674`) are bit-identical with any feature on or off.

```python
EPISODE_PHYSICS_STREAM_KEY = 0xD0A1        # §6
ACTUATION_DELAY_STREAM_KEY = 0xDE1A        # §5
OBSERVATION_NOISE_STREAM_KEY = 0x0B5E4015  # §7

def feature_stream(env, seed, key) -> np.random.Generator:
    if seed is not None:
        entropy = int(seed)
    else:
        entropy = int(getattr(env, "np_random_seed", -1))   # gymnasium >= 1.0
        if entropy < 0:                                     # np_random assigned directly
            bit_generator = env.np_random.bit_generator
            if not isinstance(bit_generator, np.random.PCG64):
                raise ValueError("... needs a seeded reset or a PCG64 np_random; got <type>")
            state = bit_generator.state["state"]            # read, never drawn
            entropy = (int(state["state"]) << 128) | int(state["inc"])
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy, spawn_key=(key,))))
```

A feature calls it when `seed is not None` or its stream is unset: rebuilt on every seeded reset and on the first,
continued across unseeded resets (SB3's auto-reset), as `np_random` is. A spawn key never coincides with gymnasium's
`SeedSequence(seed)`. The keys are named constants in `episode_start.py`, tested distinct. `pyproject.toml:26` allows
`gymnasium>=0.29.0`, which lacks `np_random_seed`; the fallback covers it (R6, R7).

### 4.3 The role flag (commit 2)

`BaseDinoEnv.training_rollouts = False` in `__init__`: every env is an evaluation env unless the trainer marks it.
`set_training_rollouts(enabled)` sets it; every feature reads it in `_begin_episode`, so a change takes effect at the
next reset, for all three alike. It is set only through `mark_training_rollouts(env, enabled)` in `episode_start.py`,
which looks up `set_training_rollouts` on `env.unwrapped` and does nothing for an env without it, such as the plain
`gym.Env` double of `test_behavior_checkpoint.py:71-93, 1243-1248` (delay P1); a vectorised env (`num_envs` or
`env_method`), whose `unwrapped` lacks the method, raises `TypeError` instead of passing silently. The only call sites
(an AST test pins them):

- `train_base.make_env(..., plant_identity=None, *, training: bool = False)` (`train_base.py:366-393`): in `_init`,
  after `validate_environment_plant`, before `Monitor` and the first `reset` (`:383-390`), since the features draw at
  reset: `if training: mark_training_rollouts(env, True)`. `create_vec_env(..., critic_phase_gamma=None, training:
  bool = False)` (`:396-460`, keyword-only) forwards it in `env_fns` (`:432`).
- `_train_stage_body` passes `training=True` only on `train_env = create_vec_env(...)` (`:1609-1620`). The `eval_env`
  call (`:1624-1634`) is unchanged, keeping the pin `eval_env.args[4] == 'eval_env_seed(seed)'`
  (`test_sb3_notebook_pins.py:2692`).
- `train_behaviors.main`: `mark_training_rollouts(env, True)` right after `create_behavior_env` (`:306`);
  `mark_training_rollouts(env, False)` right before `evaluate_behavior` (`:472`), which reuses the training env.

**Who sees what** (robot values in brackets):

| Env kind | Built at (2c1039f4) | Role | Delay | Per-body randomisation | Whole-robot mass, payload | Noise |
|---|---|---|---|---|---|---|
| Stage training rollouts | `train_base.py:1609-1620` via `create_vec_env(training=True)`, `make_env` | training | k per episode, uniform over whole physics steps in [lo, hi] [0-20] | on | on | on |
| Behaviour training | `train_behaviors.py:306`, then marked | training | drawn | on | on | on |
| Behaviour scoring | the same env, unmarked before `train_behaviors.py:472` | evaluation | `actuation_delay_eval_s` [10 ms] | on | off | if `observation_noise_eval` [on] |
| Checkpoint selection, curriculum gate | EvalCallback env, `train_base.py:1624-1634` | evaluation | eval value | on | off | if eval flag |
| CLI evaluation, stage videos | `evaluation.py:625-636`, `:317`, `:331` | evaluation | eval value | on | off | if eval flag |
| Stage reports | `reporting/stage_artifacts.py:638, 1662`, its dt probes `:1719, 1945` and its stance and trace reports through `env_class`/`env_kwargs` `:2179-2261`; `stance_report.py:251-252, 1260-1270, 1419, 2158-2186`; `episode_trace.py:476-481` | evaluation | eval value | on | off | if eval flag |
| `config.build_env` users | `config.py:431-459`: recovery gate and calibration, zero-action baseline, `widen_checkpoint`, report scripts, `scripts/warm_start_noise_scan.py:686` | evaluation | eval value | on | off | if eval flag |
| Other direct builders from a stage config or recipe | behaviour certification and replay panels (`behavior_certification.py:30-33`); `scripts/calibrate_recovery.py:128`; `scripts/stance_quality_baseline.py:118`; `harnesses/recovery_offdist_panel.py:95, 196` | evaluation | eval value | on | off | if eval flag |
| Digest harness reward captures | `harnesses/digest_snapshot.py:497-499` (stages), `:447-452` (behaviour recipes) | evaluation | eval value | on | off | if eval flag |
| Policy-probe bank | `train_base.py:1771-1773` with `bank_env_kwargs` (`policy_probe.py:163-178`) | evaluation | eval value | on | off | forced off |
| Constructor defaults | plant manifest `manifest.py:83`; harness policy lines `digest_snapshot.py:312`; mirror derivation `symmetry.py:1271`; plant probe on synthetic state (`policy_layer.py:175-210`); `gym.make`; `env_smoke` | none | off | off | off | off |

An eval value of `None` makes evaluation envs draw from the range. The bank forces noise off as it forces pushes off:
`bank_env_kwargs` sets `observation_noise_eval = False` only where the stage declares it (noise P4, option b).
`compsognathus_longipes/scripts/probe_ppo_updates.py:118` builds an unmarked training env; no model there opts in.
`warm_start_noise_scan` is a deliberate evaluation-env diagnostic (fixed 10 ms), as its docstring says (delay P11).
This fails safe: no evaluation builder can be forgotten, a forgotten trainer path trains at 10 ms, and checkpoint
selection and the gate judge one task distribution (each EvalCallback round draws new plants and noise, §6).

## 5. Actuation delay (commit 2)

**Keys** (kwargs appended after `command_config`, `base_env.py:237`): `actuation_delay_range_s: tuple[float, float] |
None = None`, seconds, the training distribution and the switch (`None` is off); `actuation_delay_eval_s: float |
None = None`, seconds, the fixed delay of every env that is not a trainer rollout env (`None`: draw from the range
there too; it need not lie in the range). TOML arrays become tuples (`config.py:395-400`); `--override` works the same
way. A species that does not list the keys raises `TypeError` if a TOML sets them. Robot: `[0.0, 0.020]` s, 21 values
of 0-20 whole 1 ms steps; `0.010` s, 10 steps. The source is D33; the build's steer1 presets carry the same 0-20 ms
(build commit 8198001c, `rl/README.md:461`) and its evaluations a fixed 10 (`rl/train.py:503`); the build's default
0-10 ms (`rl/trike_env.py:95`) is not the source (P9, M6).

**Module `environments/shared/actuation_delay.py`** (not fingerprinted): `ACTUATION_DELAY_ENV_KEYS`;
`ACTUATION_DELAY_IMPLEMENTATION = "ctrl_fifo/v1+per-physics-step+zero-order-hold+reset-ctrl-prime+episode-stream"`,
bumped on any change of meaning like `SCHEDULE_IMPLEMENTATION` (`task_fingerprint.py:77`);
`delay_steps(seconds, timestep, *, name)`, which refuses a bool, a non-number, a non-finite or negative value, and any
`|round(s / dt) x dt - s| > 1e-9` s, naming the timestep and the nearest whole value (0.010 s is 10 steps at 1 ms and
5 at 2 ms; 0.003 s at 2 ms is refused); `validate_actuation_delay(range_s, eval_s, timestep, *, episode_steps=None)`,
which refuses an eval value without a range, a string or non-pair range, `lo > hi` and, given the episode's
`max_episode_steps x frame_skip` physics steps (the env passes them), a range end or eval value of at least that many
steps, named in seconds and steps, since no target would reach the plant (milliseconds typed into a seconds key);
`primed_fifo(hold, steps)`, `steps` float64 copies with `maxlen = steps + 1`.

**State** (after the filter state, `base_env.py:258`): `_actuation_delay` (validated config or `None`),
`_actuation_delay_rng`, `_actuation_delay_k`, `_actuation_delay_fifo`; read-only property `actuation_delay_steps`, valid
from reset on (P3). Off path: two `is not None` checks per step (before the ctrl write and before the info keys) and one
per substep; nothing is copied. **Episode start:** rebuild the stream (§4.2) when seeded or unset; k is the eval value
in an evaluation env when it is not `None`, `lo` when `lo == hi`, else `int(rng.integers(lo, hi + 1))`, so a draw
happens only where k comes from the range; the FIFO is cleared.

**The FIFO.** `step()` keeps filter, scale, write (`base_env.py:1281-1282` stay). At the episode's first step
(`_step_count == 0`) the FIFO is primed from `data.ctrl` as reset left it, before that write. In the substep loop,
immediately before `mujoco.mj_step` (`:1310`): append this step's target, then `data.ctrl[:] = fifo[0]`. A target
set at control step n reaches `data.ctrl` exactly k physics steps later; substeps 0 to k-1 of an episode hold the
prime. k = 0 is bit-identical to off; k may exceed `frame_skip`. The whole nu-wide ctrl is delayed: on the robot, the
12 leg drives, as the build delays them.

**Priming.** The prime is `data.ctrl` as reset left it, which equals `key_ctrl` of `_reset_keyframe_id`
(`mj_resetDataKeyframe`, `base_env.py:1612`) because no shared code writes `data.ctrl` between reset and the first
step. Under every home-keyframe mapping that is `_scale_action(0)`, so zero action is delay-invariant. This is a
precondition: a robot test asserts `_scale_action(np.zeros(12))` equals `model.key_ctrl[env._reset_keyframe_id]` bit
for bit and that the mapping is in `_HOME_KEYFRAME_ACTION_MAPPINGS` (P10).

**Info** (after `info["distance_traveled"]`, `base_env.py:1373`), only while on: `actuation_delay_steps` (int) and
`actuation_delay_s = round(k x timestep, 12)`, since `9 x 0.001` is `0.009000000000000001` in float64 (P4). Tests
compare steps. The reset info stays `{"step": 0}` (`:1677`, in the hashed reset).

**Fingerprint.** After the command section (`task_fingerprint.py:277-278`): when
`payload["env"].get("actuation_delay_range_s") is not None`, add
`payload["actuation_delay"] = {"implementation": ACTUATION_DELAY_IMPLEMENTATION}`. `"actuation_delay"` joins
`_differing_sections` (`:307-309`) and the tuple at `behavior_checkpoint.py:420` (P6); `None == None` for every
existing model. A changed value is refused naming `env`; switching the delay on or off, or a tag bump, also names
`actuation_delay` (P7). Steps are not recorded: seconds and `physics_sha256` (which pins the timestep) fix them.
Existing species do not list the keys, and the fingerprint reads the species signature (`:168-219`).

**Mirroring.** `symmetry._mirror_env_attributes` (`symmetry.py:1178-1235`) copies `_actuation_delay_fifo` through the
ctrl map (`em.model.ctrl.apply`, as at `:1160`) and `_actuation_delay_k`, not `training_rollouts`, which is the env's
role, not episode state (P5). `mirror_env_state` refuses two envs whose delay configurations differ: a delay-off source
would leave a delayed destination with no k, and a delay-off destination would drop the in-flight targets.

**Interactions.** Rewards are still priced on the filtered command (`base_env.py:1359`); torque terms see the delayed
plant. T. rex's `_filter_action` override is unaffected. Pushes write only `xfrc_applied` (`:1264-1279`). Readers of
`data.ctrl` after a step see the applied ctrl. A behaviour pool swap happens in reset and the FIFO re-primes at the
next first step. The FIFO never touches `env.model`, so plant validation holds at any time. The docstring at
`digest_snapshot.py:493` becomes "as `train_base.make_env` builds evaluation envs (`training=False`)" (P8).

**Why a FIFO, not MuJoCo's actuator delay** (3.10.0, measured on the prototype): the same arithmetic (max |dqpos| 0.0
over 150 steps), but `delay=` needs an MJCF edit; its history is zeros after a keyframe reset; a per-episode draw
mutates `actuator_delay`, a physics-digest field (`physics_layer.py:175`); an over-long delay passes silently.

**Files:** §3. **Tests** (new `test_episode_start.py`, `test_actuation_delay.py` in `environments/shared/tests/`):

- Hook and streams: every manifest model's MRO has `EpisodeStartHook` right after `BaseDinoEnv`; `env.reset` resolves
  to `BaseDinoEnv.reset` or `SpeciesBehaviorMixin.reset`; the entropy rules of §4.2 (MT19937 raises, PCG64 is read,
  not drawn); distinct keys; `np_random` after reset equals a features-off env's.
- Role: `mark_training_rollouts` on a plain `gym.Env` is a no-op; an AST scan finds it only in `make_env` and
  `train_behaviors.main`, `training=True` only on `train_env`. Both new files import no SB3 and run in CI's `test`
  matrix. The two checks that need SB3 (`make_env` calls `_ensure_sb3`, `train_base.py:375-377`, which exits without
  it, `policy_loading.py:177-207`) go in files the test-sb3 step already runs (`python-ci.yml:487-524`), so
  `test_ci_tool_pins.py:364-376` holds: `create_vec_env(training=True)` marks each env and the default none, in
  `test_train_base.py` behind `pytest.importorskip("stable_baselines3")` as its other SB3 tests are; and
  `train_behaviors.main` marks its env True before training and False before `evaluate_behavior`, in
  `test_behavior_checkpoint.py` (module-level `importorskip`), on a `PilotEnv` with a recording
  `set_training_rollouts`.
- Off by default: `_actuation_delay is None` exactly when the stage omits the keys (so commit 5 needs no edit); then no
  delay info key, and marking the env leaves a seeded zero-action plus random-action rollout byte-identical. The
  models listing a key are exactly an opted-in test table (empty now, the robot in commit 5), each with both keys at
  `None` and the carve-out; on every other model a TOML setting one raises `TypeError`.
- Exact delay for k = 0, 1, 3, 10, 25 on a test-only Compsognathus-robot subclass (2 ms, `frame_skip` 10; commit 5
  adds the robot): per-substep ctrl is the prime for s < k, then `targets[(s - k) // frame_skip]`; bit-equal to an
  independent deque FIFO on a copied `MjData` over 40 steps; k = 0 bit-identical to off.
- Evaluation envs hold the eval value (20 seeds); training envs draw k in [lo, hi], at least 15 distinct values over 60
  seeds, readable right after reset, independent of the actions; `None` eval draws; a degenerate range never draws,
  and an evaluation env with an eval value (5 steps, or 0) holds it over the range's one value (3) while a training
  env holds 3, which pins the order of the episode-start rule. The FIFO holds the filtered, scaled target. A pushed
  250-step zero-action rollout is bit-identical with the delay.
- No plant digest moves; `validate_environment_plant` passes mid-episode. Nine invalid configs refused at
  construction, two of them as long as an episode.
- Fingerprint: no section on any existing stage; a changed eval value names `env`; switching off names `env` and
  `actuation_delay`; an existing species opting in with the `None` carve-out (monkeypatched registry) keeps every
  committed Compsognathus-robot digest. `test_symmetry.py`: a delayed Velociraptor subclass mirrored mid-episode gives
  max obs error 0.0 and keeps its role. `test_behavior_checkpoint.py`: a changed `actuation_delay` section is refused.

**Measured on the prototype.** Off path bit-identical on all 7 models over 30 rollouts (obs, reward, info, qpos,
qvel, ctrl, generator state); exact k-step delay at 1 ms / 20 and 2 ms / 10; `--exact` 40,568 lines, empty diff;
`digest_snapshot --check` current (725 lines); `plant_contract --check` and `symmetry --check` current. Dropping
in-flight targets in a mirror diverges (max obs error 366 on a T. rex test subclass, 128 on a Velociraptor one).
Overhead within run-to-run noise. The prototype drew k from `np_random` at the first step; commit 2 draws it at reset
from its own stream (§4.2), as designed.

**Built in commit 2, where it differs from the text above** (changed after the build and its review, not after any
training result):

- `BaseDinoEnv._configure_actuation_delay(range_s, eval_s)` validates and installs the delay; `__init__` calls it, and a
  test subclass of a species that does not forward the keys calls it again after `super().__init__()`.
- `set_training_rollouts` refuses a non-bool (`TypeError`); stepping a delayed env before its first reset raises
  `RuntimeError`, since k is chosen at reset; the FIFO is primed again if it is missing mid-episode.
- All three stream keys of §4.2 are defined now, with a `FEATURE_STREAM_KEYS` tuple; in commit 2 only the delay's is
  used (commit 3 uses the physics key).
- The training-draw test uses the range 0-40 ms on the 2 ms Compsognathus-robot subclass (21 values), since 0-20 ms is
  only 11 values at 2 ms.
- The Velociraptor mirror test uses a delay of 6-12 physics steps (`frame_skip` 5): with k below `frame_skip` the FIFO
  holds only the last target, so dropping it would not show. Dropping the FIFO there gives max obs errors of 35.9 and
  73.8 (seeds 3047 and 3059), not the prototype's 128, which came from another configuration.
- Test names avoid the retired species identifiers (`test_retired_species_identifiers.py`).

## 6. Domain randomisation (commit 3)

**Keys** (six kwargs, each `None` or `(low, high)`: finite, `low <= high`, `low > 0` for every key but the payload
(friction included), payload `low >= 0`):

| Key | Unit | What an episode writes | Robot |
|---|---|---|---|
| `randomize_friction` | absolute sliding coefficient | one draw to `geom_friction[:, 0]` of every static floor geom (`_static_floor_geoms`, `base_env.py:1490`) and of the geoms `_randomized_friction_geoms()` returns | [0.6, 1.0] |
| `randomize_body_mass_scale` | factor | per animal body (free-root subtree): mass and inertia by one factor | [0.9, 1.1] |
| `randomize_actuator_kp_scale` | factor | per position servo (FIXED gain, AFFINE bias, `biasprm[1] == -gainprm[0]`): `gainprm[0]` and `biasprm[1]` together; kv kept | [0.9, 1.1] |
| `randomize_actuator_force_limit_scale` | factor | per force-limited actuator: both ends of `forcerange` | [0.85, 1.0] |
| `randomize_total_mass_scale` | factor | one factor on every animal body's mass and inertia, after the per-body factors; training rollouts only | [0.9, 1.1] (D38) |
| `randomize_payload_kg` | kg | added to the payload body (the root if it has mass, else the heaviest welded body, `perturbation.py:201-205`), inertia x (m + p) / m; training rollouts only | [0.0, 1.0] (D38) |

The per-body values are the build's (build commit 8198001c, `rl/trike_env.py:100-103`); the whole-robot values are
D38's (the steer1 presets, `rl/README.md:461-462`). Friction is absolute, not the multiplicative `friction_range` of
`docs/ROADMAP.md:248-250`: it reads as the build's numbers and is model-independent. Refused at construction:
friction with explicit contact pairs (`npair > 0`); kp with no position servo (Allosaurus); force limits with none;
a payload on a zero-mass body (R8). The robot's payload body is the chassis (5.9986 kg), its free root, which §8.1
renames `torso`.

**Friction hook** (R5). `BaseDinoEnv._randomized_friction_geoms()` defaults to every colliding animal geom; the robot
returns its four `<foot>_sole` geoms, as the build writes the floor, the soles and its terrain boxes only
(`rl/trike_env.py:797-799`). Self-contacts keep their authored friction. On a behaviour heightfield pool the
heightfield is a static floor geom and takes the draw.

**Draws.** `begin_episode(env, seed)`: restore every model this env changed; rebuild the stream (§4.2,
`EPISODE_PHYSICS_STREAM_KEY`) when seeded or unset; draw `u = rng.random(3 + nbody + 2 x nu)` whatever is enabled
(49 on the robot): `u[0]` friction, then one per body, one per actuator for kp, one per actuator for force, `u[-2]`
total, `u[-1]` payload, each `low + (high - low) x u`; write the enabled quantities from the nominal copies in the
order friction, per-body, total, payload, kp, force. Total factor and payload apply only while `env.training_rollouts`
is true (elsewhere 1 and 0 kg), so a seed's per-body factors are the same in training and evaluation. Enabling one key
never moves another's draw. Per-seed draws make the fixed-seed panels paired: the recovery evaluation
(`recovery_evaluation.py:195, 268`), the stance report and the publication evaluation after
`eval_env.seed(PUBLICATION_SEED_START)` (`train_base.py:2519`) give every checkpoint and null controller the same
plant in episode k. The EvalCallback env is seeded once in `make_env` (`train_base.py:390`) and then reset unseeded,
so each round sees new plants, as it already sees new poses (R15).

**mj_setConst on scratch data.** If a mass key is on, `mj_setConst(model, record.scratch)` runs on the record's own
`MjData`, never the live one: on live data it moves qpos to `qpos_spring` (measured on all 7 manifest models and the
60 cm model). Then `stat.center`, `stat.extent` and `stat.meansize` go back to nominal; setConst recomputes them from
geometry unlike the compiler, and nothing randomised moves geometry.

**Exact restore.** Each model's record keeps exact copies of `RANDOMIZED_FIELDS` (geom_friction, body_mass,
body_inertia, actuator_gainprm, actuator_biasprm, actuator_forcerange), `SETCONST_FIELDS` (body_subtreemass,
body_invweight0, dof_invweight0, dof_M0, tendon_invweight0, tendon_length0, actuator_acc0, actuator_length0,
cam_pos0, cam_poscom0, cam_mat0, light_pos0, light_poscom0, light_dir0) and `stat` (meaninertia, meanmass, meansize,
extent, center), plus a scratch `MjData`. Restore is `np.copyto` from these, never another setConst (not a no-op on
four unmodified models, measured). Ranges are validated at construction; a model's record is taken the first time an
episode starts on it, which covers the constructor model and behaviour pool models alike and keeps a species edit made
after `super().__init__()` (R16). `nominal_physics()` captures the live values, restores nominal, yields, and writes
them back, bit-exact both ways; it captures on entry and nothing else is kept (R10). `episode_physics_factors` exposes
the episode's draws or `None`. No manifest method is added (as noise P10 drops one).

**Behaviour-env pools.** `SpeciesBehaviorMixin._select_contact_model` (`behavior_env.py:248`) first calls
`self._restore_episode_physics()`, before the swap and `_cache_ids()` (`:261`). Species `_cache_ids` cache weights from
masses (Compsognathus, Allosaurus, T. rex read-only; Velociraptor caches its weight once in `__init__`,
`base_env.py:439`, before any draw); without the restore the Compsognathus contact threshold followed the previous
episode's mass, up to x1.15 (measured).

**Pushes, settle, readers.** Push forces stay in newtons from the nominal plant (`base_env.py:299-329`), as the stage
fingerprint derives them; the settled pose depends on geometry only; no info key is added. The robot caches its
nominal weight in `_cache_ids` and never reads live masses in `_get_obs` or `_get_reward_info`. Readers of the live
model after a reset (`stance_report.py:993`, `balance_telemetry.py:131`, `body_telemetry.py:466`,
`floor_contact/morphology.py:498`) are checked on a robot env; a layout built there moves into `nominal_physics()`
(§8.12, item 4, with their lines on the merged tree).

**Validation.** The physics digest covers every randomised field (`physics_layer.py:95-96, 145, 167-168, 179`), so the
identity is the nominal plant. `plant_contract/validation.py` stays unchanged and strict (R4). Every live site
validates before the first reset (`train_base.py:382-390`, `evaluation.py:316-326, 625-634`,
`stance_report.py:251-258`, `episode_trace.py:482-484`, `warm_start_noise_scan.py:703`, `config.py:457-458`,
`behavior_env.py:143`); after a randomised reset both validators refuse, which is fail-closed. A diagnostic that needs
the nominal plant later uses `with env.nominal_physics():`. `docs/PLANT_CONTRACT.md` states the rules: randomised
values live only between a reset and the next, are never fingerprinted, and no managed field may be written after
construction while randomisation is on.

**Fingerprint.** When any `randomize_*` key in the env section is not `None`:
`payload["episode_physics"] = {"implementation": "mesozoic.episode-physics-randomization/v1"}`; it joins
`_differing_sections` and `behavior_checkpoint.py:420` (R3).

**Catalog and docs.** `configs/species_manifest.toml:44-46`: `domain_randomization` status `"experimental"` (allowed,
`species_catalog.py:103`); `python -m environments.shared.species_catalog` regenerates the site JSON (R11).
`docs/PLANT_CONTRACT.md` gains "Per-episode physics randomisation". `docs/ROADMAP.md:240-266` marks friction, mass,
kp and force-limit randomisation done, notes absolute friction, and notes that the robot randomises in both stages, as
the build does, against the "Stage 2+ configs only" advice (`:264-265`, R12). `docs/hardware/SIM_TO_REAL_PLAN.md`
says the plumbing is absent (`:51-54`) and lists domain randomisation as planned (`:81`); it gains a dated status
note at its top, as its 2026-11-23 one, naming the delay and randomisation as opt-in shared features and the
capability as `experimental` (commit 4 adds noise to it). `docs/RECOMMENDATIONS.md:231-232` ("Not started") stays:
it is a dated snapshot, not maintained (`docs/README.md:48`).

**Files:** §3. **Tests** (new `test_episode_physics.py`):

- Off: `_episode_physics` and `episode_physics_factors` are `None` exactly when the stage omits every key;
  `BaseDinoEnv` has all six with default `None`; the classes listing a key are exactly an opted-in test table (empty
  now, the robot in commit 5), each defaulting it to `None` with an `OPT_IN_TASK_KWARGS` carve-out (R2). An off reset
  never reaches the randomiser (its `begin_episode` patched to raise; the reset record equals the golden's).
- With a randomiser injected by test subclasses on every manifest model: reset qpos, qvel, mocap, push schedule and
  `np_random` state equal the off env's; a managed field differs; on the Compsognathus-robot recovery stage the push
  force and `perturbation_manifest()` are equal. After a restore, every ndarray of `env.model` and the five stats
  equal a randomisation-off env of the same kwargs exactly (R13).
- Each quantity writes only its fields (friction on the floor and the hook's geoms, columns 1-2 kept; inertia ratio
  equals mass ratio; world and mocap masses kept; `gainprm[0] == -biasprm[1]`, `biasprm[2]` kept; one force factor;
  total mass = nominal x factor + payload with inertia ratio (m + p) / m in training, nominal in evaluation for the same
  seed; stats kept). setConst on scratch: with every mass key on and reset noise > 0, reset qpos equals the off env's.
- Draws: repeatable per seed, new on unseeded resets, equal across two envs, friction independent of other keys,
  `3 + nbody + 2 x nu` per episode, `np_random` equal on and off. Both validators refuse after a randomised reset and
  pass inside `nominal_physics()`. Pools restore before `_cache_ids` (Compsognathus robot, `mixed_terrain`, 8 resets).
- Refusals: friction (0.0, 1.0), (1.0, 0.5), (nan, 1), a 3-tuple; payload (-0.1, 0); kp and force keys on
  Allosaurus; a contact `<pair>` model built with MjSpec; a zero-mass payload body. Fingerprint: no section while off,
  the tag while live, off keys carved out.

**Measured on the prototype.** Off path: `plant_contract --check` and `digest_snapshot --check` current (725 lines),
`--exact` byte-identical (40,568 lines); 648 focused tests pass, and the full shared suite (7,305) fails only the 12
sandbox failures base also fails. Restore bit-identical to a fresh compile on all 7 models. On a thin 60 cm test env
over 2000 resets: per-body factors x0.928-1.072 of the 12.7506 kg animal (build: x0.930-1.073); with the whole-robot
factor and payload x0.865-1.211, mean 1.039 (build: x0.864-1.230, mean 1.038; `rl/README.md:400-401`). Reset 233 us
off, 429 us on; step time unchanged. The prototype wrote friction to 33 geoms; this design writes 5 on the robot.

**Built in commit 3, where it differs from the text above** (changed after the build and its review, not after any
training result):

- `BaseDinoEnv._configure_episode_physics(**ranges)` validates and installs the randomisation, restoring any randomised
  model first; `__init__` calls it after `_cache_ids` and the push derivation, and the test subclasses
  (`environments/shared/tests/episode_physics_helpers.py`) of species that do not forward the keys call it again after
  `super().__init__()`, as commit 2's delay subclasses call `_configure_actuation_delay`.
- `episode_physics_factors` holds the applied values by quantity, for the enabled keys only: `friction`,
  `body_mass_scale` (one per animal body), `actuator_kp_scale` (per position servo), `actuator_force_limit_scale` (per
  force-limited actuator), `total_mass_scale` (1.0 in an evaluation env) and `payload_kg` (0.0 there); `None` while
  off, before the first reset and after an episode whose values could not be written (a refusal when its model's
  record is taken, or a raise mid-write). A reset refused before it begins (inside `nominal_physics()`, below) leaves
  them, and everything else, as they were.
- More refusals than the list above, each a `ValueError` naming the key, at construction and again when a model's
  record is taken (its first episode; so a species edit made after `super().__init__()` also decides the payload
  body, the servo and force sets and the friction geoms, R16): friction with no static floor geom; a
  `_randomized_friction_geoms()` that returns no geom, anything but a 1-D array of integer ids (a boolean mask would
  index as geoms 0 and 1), an id outside the model, or a geom that is not a colliding animal geom; a behaviour pool
  model of another `nbody` or `nu` (at its first episode, since the draw layout is the constructor model's);
  `animal_bodies` refuses a model with no free joint. The construction check leaves the env's geom caches as it found
  them, so a collision-mask edit made after `super().__init__()` still reaches the settle.
- `nominal_physics()` blocks nest and write the episode's values back when they raise; a reset inside one raises
  `RuntimeError`, since the block's exit would write the previous episode's values over the new one, and so does
  reconfiguring the randomisation (`_configure_episode_physics`), whose new records would take the episode's values as
  nominal. Both are refused before anything moves: the hook's `reset` gains a first call,
  `self._check_episode_may_begin()`, before `super().reset` (so gymnasium reseeds nothing; nothing by default,
  `BaseDinoEnv`'s one `is not None` check while randomisation is off), and the behaviour mixin's `reset` calls it first
  too, before it moves its episode counters or selects a pool model. No committed digest hashes either source (§4.1:
  `episode_start.py` is not fingerprinted, and the policy layer's `home_reset` is the canonical env's
  `BaseDinoEnv.reset`, unchanged), so no digest moves (gates 2 and 3); a refused reset changes no attribute of the env
  (tested on the Compsognathus robot and its `mixed_terrain` behaviour env, where the refused reset would have swapped
  the plane for the heightfield). A raise while an episode's values are written restores the nominal plant before it
  propagates.
- `symmetry.mirror_env_state` refuses an env that holds a randomised plant (source, destination or in place): it
  mirrors the state, not the plant, and per-body and per-servo draws are not mirror-symmetric (pair-tied draws, §11).
- The scratch `MjData` is defensive: `begin_episode` runs before the keyframe load, which overwrites the live data, so
  setConst on the live data inside `reset` would not move the reset pose (it does anywhere else). The test checks
  which data setConst sees.
- The payload rule spreads the payload like the payload body's own mass, so it assumes a body heavy against the
  payload: a 1 kg payload on a 0.5 kg body triples that body's inertia, and only a massless body is refused. The
  robot's chassis (the torso of §8.1; 6 kg, up to 1 kg, x1.17) satisfies it; the module docstring and
  `docs/PLANT_CONTRACT.md` say so.
- setConst runs only when an episode wrote a mass, not whenever a mass key is on: in an evaluation env with only the
  two training-only keys on, nothing is written and the model stays exactly nominal, so the validators pass there (a
  test pins it). The values are the same either way.
- `TRAINING_ONLY_ENV_KEYS` names the two training-only keys, `GEOMETRIC_STAT_FIELDS` the three statistics put back
  after setConst, and `colliding_animal_geoms` is `_root_subtree_geoms()`'s definition read without its cache.
- Files beyond §3's commit-3 list: `episode_start.py` (the pre-reset check above), `symmetry.py` (the mirror refusal)
  and `tests/test_actuation_delay.py`, whose signature test now places the two delay keys right after
  `command_config` rather than last, since the six randomisation keys follow them.
- Tests: the payload range is 0 to a tenth of each model's animal mass; the pool test runs 16 resets (seed 0, then 15
  unseeded), which return to each pool model while the other's record is the newer; the draw layout is pinned index by
  index on every model, in training and evaluation.
- Handed to commit 5, besides the four readers named under "Pushes, settle, readers": `Morphology.from_env` reads
  `body_subtreemass` for `body_weight_n` (`floor_contact/morphology.py:498`), and `actuator_force_limits`
  (`gait/balance_telemetry.py:128`) the live force ranges, so the layouts built from them after a reset are
  per-episode: `reporting/stance_report.py:294`, `evaluation.py:482`, `eval_diagnostics.py:723, 954` and
  `rollout_ledger._layout_of` (`rollout_ledger.py:311`, built after the first reset and cached). On the robot env
  they are built inside `nominal_physics()` or before the first reset, or documented as per-episode. §8.12 (item 4)
  carries this hand-off into commit 5, on the merged tree's lines and with main's new command-tracking panel.

## 7. Observation noise (commit 4)

**Keys:** `observation_noise_std: Mapping[str, float] | None = None` (std per segment, in its own units; kept as a
plain `dict`, since `SubprocVecEnv` pickles attributes) and `observation_noise_eval: bool = False`. The table is an
inline TOML table; `--override env.observation_noise_std.<segment>=x` works into a declared one (`cli.py:85-110`). A
child stage that `extends` another replaces a nested table whole (`config.py:246-248`), so it must restate or inherit
the whole table (P12). The robot's locomotion is written whole; a robot test asserts its table equals stance's.

**Where.** In `BaseDinoEnv.step` on the final observation, after the command recompute (`base_env.py:1384-1385`),
before render and return (`:1388-1391`), when active. Nothing is added in the hashed `_get_obs` or `reset`
(`policy_layer.py:378-406`). **The reset observation stays noise-free**: one per episode, 0.1 % of a 1000-step
episode; the build noises it (build commit 8198001c, `rl/trike_env.py:903`). Noising it in the hashed reset would move
every home-keyframe model's policy digest; a `Monitor` subclass in `make_env` and `evaluate` could do it later, as
transparent to SB3's `get_attr` as today's `Monitor` (P8).

**Active** when `_observation_noise is not None` and (`training_rollouts` or `observation_noise_eval`), latched in
`_begin_episode`. The spec's three roles, `set_observation_noise_role` and its `evaluation.py` edits are dropped.

**Module `environments/shared/observation_noise.py`** (numpy only at top level):
`OBSERVATION_NOISE_IMPLEMENTATION = "obs_noise/v1+gaussian-per-segment+post-step+episode-stream+unit-segment-renormalize"`;
`observation_layout(env)`, names from `policy_layer.observation_segments(...)`, widths from `_segment_width` except
`foot_contact`, whose width is the number of foot groups (Brachiosaurus: 8 touch sensors, 4 channels);
`ObservationNoise.apply(obs, rng)` adds `std x rng.standard_normal(width)` in float64 over the full width in
observation order (so one segment on or off never changes another's realisation), renormalises each
`*_orientation_quaternion`, `*_projected_gravity`, `*_direction` or `*_direction_body` segment with std above 0
(`v / max(|v|, 1e-9)`), and returns float32; `resolve_observation_noise` returns `None` while off and refuses a
non-bool eval, eval without a table, a non-mapping, widths not summing to the observation width, an unknown segment
(listing valid names), a bool, non-numeric, non-finite or negative std, `command > 0`, and a table with no positive
std. No `observation_noise_manifest()` (P10).

**Stream.** `_begin_observation_noise_episode` rebuilds `_observation_noise_rng` (§4.2,
`OBSERVATION_NOISE_STREAM_KEY`) when seeded or unset, replacing the prototype's `_step_count == 1` detection (P7).

**Segment table** (moves no digest). The inline table and label logic of `_policy_interface_payload`
(`policy_layer.py:344-373`) move into a public `observation_segments(model, env, observation_schema)`, which the
payload calls; the heading-free width check stays in the payload (`:350-359`). Commit 4 keeps today's rule; on main's
two schemas commit 5 needs no change to it (§3, "The segment table"; §8.4).

**Robot table** (written for the first §8's target schema, root `torso`; on main's `quadrupedal-heading-free/v1` the
robot keeps every row but the target rows, §8.4; build values at build commit 8198001c, `rl/trike_env.py:113-119`):

| Segment | Width | Std | Source |
|---|---|---|---|
| `joint_position` | 20 | 0.005 rad | `noise_joint_pos` 0.005 rad, legs and ankles (`:114`); here the neck and tail too |
| `joint_velocity` | 20 | 0.05 rad/s | `noise_joint_vel` 0.05 rad/s (`:115`) |
| `torso_projected_gravity` | 3 | 0.02, renormalised | 0.01 per quaternion component, renormalised (`:117`), tilts about 2 x 0.01 rad per axis: 1.436 deg mean error, against 1.443 deg at 0.02 and 0.719 deg at 0.01 (P2) |
| `torso_angular_velocity` | 3 | 0.02 rad/s | `noise_gyro` 0.02 rad/s (`:116`) |
| `torso_heading_linear_velocity` | 3 | 0.02 m/s | `noise_lin_vel` 0.02 m/s on the privileged velocity (`:118`); isotropic, so the frame does not matter |
| `torso_linear_acceleration` | 3 | 0.2 m/s^2 | not observed by the build; g x 0.02 = 0.196 m/s^2, so it carries no more tilt than the gravity channel (P3) |
| `foot_contact` | 4 | 2.5 N | `noise_foot_load` 0.02 of weight (`:119`): 0.02 x 12.7506 kg x 9.81 m/s^2 = 2.50 N, a literal |
| `command` | 3 | 0 | the command is the task's input; the first design's `target_direction_body` (3) and `target_distance` (1), also 0, went with main's targets (D-D59, §8.4) |

`observation_noise_eval = true` (D37). Heading drift (`rl/trike_env.py:120-123`) is moot: the heading-free
observation carries no heading (P9). TOML comments cite the values by build commit (P13).

**VecNormalize.** Noise is added inside the env, before `VecNormalize`, so `obs_rms` learns the noisy distribution
(measured on the prototype: joint-velocity `obs_rms.var` rose by 0.00259 against std^2 = 0.0025). Never after
normalisation. The command slice is never noised (`load_vecnorm_stats(reseed_command_slice=True)` unaffected);
critic-phase columns are appended after normalisation.

**Fingerprint.** When `observation_noise_std` is not `None`, `payload["observation_noise"] = {"implementation":
OBSERVATION_NOISE_IMPLEMENTATION}`; it joins `_differing_sections` and `behavior_checkpoint.py:420`. **Docs.**
`docs/PLANT_CONTRACT.md`: noise is task-level, in `step()`, outside the hashed methods; probe, validation and the reset
observation are clean. `docs/ROADMAP.md:268-277`: noise done, per-segment keys in place of `obs_noise_scale`. The
`SIM_TO_REAL_PLAN.md` status note (§6) gains sensor noise. `CONTRIBUTING.md:343-344` names the three modules as the
shared robot code's first members (commit 2 starts it; P13).

**Files:** §3. **Tests** (new `test_observation_noise.py`):

- Off: `_observation_noise is None` exactly when the stage omits the table; while inactive, `step()` returns the very
  object `_get_obs` returned, on every model (P5). Marking an env as training is inert while off (byte-identical).
- Layout: widths sum to the observation width; names equal `observation_segments(...)` and the payload's
  `observation_segments`; perturbing each source on the probe state changes only its segments; every segment is fed.
- Std on brachiosaurus (legacy) and allosaurus (heading-free) with a test table: reset until at least 2000 noisy steps,
  tolerance `5 / sqrt(2 n)` (P6); zero elsewhere; unit segments of norm 1 within 1e-6.
- Reproducible per seed; never draws `np_random`; qpos, reward, info equal the clean env's; unseeded episodes differ;
  the reset observation is clean. Active only for training rollouts or with the eval flag; `bank_env_kwargs` sets
  `observation_noise_eval = False` only where declared. Refusals; validation and the plant probe clean; section only
  while on.

**Measured on the prototype.** Off path bit-identical on brachiosaurus, the Compsognathus robot and allosaurus;
`--exact` identical (40,568 lines); `plant_contract --check` current with the refactor. Measured std matched the
setting (brachiosaurus, 1600 steps: 0.00500, 0.04995, 0.02027, 0.01984, 2.526 N); the build's own noise: 0.00499,
0.04994, 0.02032, 0.01999, load 0.02026 of weight. Cost 5.4-6.3 us per call.

**Built in commit 4, where it differs from the text above** (changed after the build and its review, not after any
training result):

- `observation_noise.observation_schema(env)` finds the model's observation schema: the model is the manifest model
  whose `env_entrypoint` class is nearest in `type(env).__mro__`, so a behaviour env and a test subclass read their
  model's schema; any other class is refused with `ValueError`. `observation_layout(env, schema=None)` and
  `resolve_observation_noise(env, std, eval, *, schema=None)` take an optional schema, the model's own by default.
- `BaseDinoEnv._configure_observation_noise(std, eval)` validates and installs the noise; `__init__` calls it once the
  observation space exists, and the test subclasses (`environments/shared/tests/observation_noise_helpers.py`) of
  species that do not forward the keys call it again after `super().__init__()`, as commits 2 and 3 do. The
  `observation_noise_active` property reads the episode's latch: `False` while off and before the first reset.
- A noisy env that steps before its first reset, or after `_configure_observation_noise` until the next reset, raises
  `RuntimeError`, since the latch and the stream begin at reset.
- More refusals: a non-bool `observation_noise_eval` is refused even while the table is off, and so is a std that is
  finite in float64 but not in float32, the observation's dtype (`1e39` would make every noisy observation `inf`).
- `policy_probe.OBSERVATION_NOISE_EVAL_KWARG` names the flag `bank_env_kwargs` holds false, as `PUSH_MULTIPLE_KWARG`
  names the push multiple.
- The behaviour env's `set_direction`, which returns the observation the policy acts on next, noises it in a noised
  episode (one draw of the stream, as `step()`), so no mid-episode observation reaches a policy clean.
- `symmetry.mirror_env_state` refuses an env whose episode is noised (source, destination or in place): its physics
  steps as the mirror image, but each env draws its own noise in observation order, so its step observations are not
  mirror images, and copying the stream would not make them so. An env with the noise configured but its episode not
  noised mirrors as before.
- Files beyond §3's commit-4 list: `behavior_env.py` (`set_direction`), `symmetry.py` (the mirror refusal), the
  bank's descriptions in `train_base.py` and `docs/TRAINING_ARTIFACT_PRESENTATION.md`, and tests outside
  `test_observation_noise.py`: `tests/test_episode_physics.py` (its signature test now places the six keys right
  after the delay's two, which follow `command_config`, since the noise's two follow them),
  `tests/test_episode_start.py` (the feature-off hook test also refuses the noise's episode start and checks its
  attributes stay unset),
  `tests/test_behavior_checkpoint.py` (a changed `observation_noise` section is refused by name) and
  `tests/test_symmetry.py` (the mirror refusal).
- Tests: the stream policy is pinned directly (after an unseeded reset the stream is the same generator, at the state
  the previous episode ended in, not the stream a reset of the env's seed would build), and the unseeded episodes'
  realised noise is compared on the joint velocities with a margin far above float32 rounding. A unit segment at std
  0, as the first design's target direction was, is tested on both schemas: through `apply()` on a vector not of unit
  length, and over 300 noised steps. The plain `dict` is pinned on the stage kwargs, where `SubprocVecEnv` pickles it
  (the env keeps only the frozen `ObservationNoise`): a table read from a stage TOML, set by `--override` into it and
  passed through `bank_env_kwargs` stays a `dict`, and the kwargs pickle to an env with the same noise.

## 8. The triceratops_horridus/robot model (commit 5)

Revised 28 Nov 2026 for `main` a232da31, replacing the first design of this section (written against 2c1039f4), which
main's target-free revision (D-D59) and its command-driven walk (D-D60 to D-D64) overtook (§3, "Commit 5 (§8)"). The
owner's decisions D50-D56 of 28 Nov 2026 (§2) settle the choices the revision raised; each is cited where it applies,
with its reason. In this section every `file:line` is on the merged branch e94efee9 (main a232da31 plus commits 2-4)
unless it names a232da31 or the build; `configs/`, `docs/` and every file commits 2-4 did not touch are byte-identical
to a232da31. Main's decision rows D-D59 to D-D78 are `docs/BEHAVIOR_RECIPES_PLAN.md:1558-1577`. Build files are cited
by commit: f6deec9f for the source model, fb1e80c8 for the build reports read for this revision (`rl/RUNS.md`,
`rl/analysis/REPORT_FORELEG_D22_APPENDIX.md`, `walk/REPORT_WALK60_FORE128.md`, `rl/README.md`).

Kinds of statement, as in the rest of the record: **Design**; **Measured on the prototype** (the first design's
scratch prototypes at 2c1039f4, kept only where main's changes leave them valid); and two more here: **Computed** (for
this revision, from the source asset with MuJoCo 3.10.0, read only; re-measured on the derived asset in commit 5) and
**Estimate** (from the reward weights or a scaling rule, nothing run). An owner decision is cited by its row (§2).
Where a choice is the implementer's or main's maintainer's, the text says so. Nothing in this section has been built or
run on the derived asset; §8.12 lists what commit 5 measures before its design is final.

### 8.0 What stays and what changes

| Part | Kept from the first design | Redesigned for main a232da31 |
|---|---|---|
| Asset (8.1) | Source, sha256 refusal, ElementTree edits, leg renames, spring folds, reach, settled `home`, provenance, no mirror map | No mocap `target` (nmocap 0); the free root `chassis` renamed `torso` |
| Timestep (8.2) | 1 ms, `frame_skip = 20`, the constructor's check | Eleven control-step readers, not eight; recipes inherit both keys from the walk |
| Species and robot env (8.3) | Every hashed method on the species base, weights, filter, `FOOT_SUPPORT`, feature kwargs and carve-outs | No `BEHAVIOR_SECTION_LABEL`; `SUCCESS_KEYS = ()`; `REWARD_PROBE = (0.2,)`; no course kwargs (D50); `_reset_episode_state`; `pelvis_height` info |
| Observation (8.4) | Segment content, action mapping, noise values | Main's `quadrupedal-heading-free/v1`, 59 values, no target segments, no shared edit |
| Rewards, termination (8.5) | The shared terms and the root and body-contact terminations | The command-tracking term and the stand block replace the course; no course termination; no arrival bonus (D50; D39 superseded) |
| Stages and gates (8.6) | Stance then locomotion, the feature block, PPO and SAC, D43 | A live `heading_and_speed` walk at a 0.15 m/s cruise (D51); walk gate `command_tracking/v1` (D52); stance gate `stance_quality_heading/v1` if commit 5's four-foot check passes, else D40's `reward_and_length/v1` (D52); stance 11M steps; the owner reviews the stance bars before the first run (D54); the training speed measured (D55); walk `warmup_ent_coef` 0.005 and `[sac] ent_coef = "auto"` |
| Recipes (8.7) | All 11 ship (D42) | On main's single command contract; `[env]` inherited from the walk; each writes its own `[ppo]` and, on nine, `[terrain_sampler]`; the recipe cruise is the walk's; none can be trained from the committed walk; ported as D53 words it |
| Manifest (8.8) | Entry, revisions 1/1/1, regeneration | Success metric `command_tracking`; plant note 25; new counts |
| Shared tests (8.9), robot tests (8.10), docs (8.11) | Most rows | New per-model tables; two heading-free test changes for the first enabled feature; course and arrival tests go; the model's website page is its generated route beside the hand-written card (D56); website images and robot-page text added |
| Measured in commit 5 (8.12) | The §6 hand-off of live-model readers | The four-foot check that picks the stance kind (D52), the statue-panel stance bars for the owner (D54), the training speed (D55), the values the tests need |

### 8.1 Asset derivation and provenance

- **Source** (never imported), unchanged: build commit f6deec9f9be1566e2c5de8e0932bc4d9d99ea2fd,
  `hardware/triceratops_build/sim/generated/trike60_C_closed_capsule_fore128.xml` (the D25 128 mm foreleg), blob
  04d3078ceb545d72561e1f519e464e94276b1653, sha256 372ac07b14338c26ce2fa8dbd0a34952597080fafcc82149464b10b25a059eb1:
  1 ms step, implicitfast, 16 position actuators (12 legs at kp 598.5; neck and tail at kp 8/6/5/8, kv
  0.3/0.2/0.15/0.2), four spring ankles (2.032 N m/rad), keyframes `neutral` and `home`, no mocap body, 12.750593 kg,
  `cone="elliptic"` (every plant's, `test_friction_cone.py:36`) and a world plane named `floor` (which
  `terrain.build_terrain_model` needs for the recipes, `terrain.py:466-474`).
- **Script** `environments/triceratops_horridus/scripts/derive_robot_model.py`, ElementTree edits (an MjSpec round
  trip lost the zero-distance hind-sole contacts); it refuses another sha256. It finds the asset and
  `data/robot_model_provenance.json` from `environments.shared.paths.REPOSITORY_ROOT` and runs as a module
  (`python -m ...`, below), never by path: `test_repository_root.py:299-308` fails any other module that climbs
  `Path(__file__).parents[k]` to the root (the allowed computations, `:265-273`), and a script run by path with a
  `_repo_root` bootstrap would have to join `SYS_PATH_BOOTSTRAPS` (`:280-297`). The same holds for
  `scripts/view_model.py` (§8.3).
  1. Model name `triceratops_horridus_robot`.
  2. Renames. Leg prefixes become `fl_`, `fr_`, `rl_`, `rr_`: the gait metrics label a quadruped limb by the foot
     site's first `_` token and raise unless the four labels are `fl, fr, rl, rr` (`gait/locomotion_metrics.py:154-155`,
     `:218-227`); 8 unsided part pairs get side names. **New:** the free root body `chassis` (source line 63) becomes
     `torso`, the quadrupedal schema's root label (`heading_observation.py:45-47`), because
     `test_heading_free_observation.py:167` and `:285` require the free joint's body to be `model.body("torso")`.
     Every reader of the name follows (camera tracking, `_cache_ids`, the morphology root of
     `test_gait_morphology.py`). No dynamics change. The implementer's choice: it is the only way to meet both tests
     without a shared test change.
  3. Head and tail servos fold into their joint springs about target 0 (stiffness += kp, springref = k0 r0 / (k0 +
     kp), damping += kv); the 4 actuators and their force sensors go (nu 16 to 12, nsensor 61 to 57): neck_pitch
     8.4 N m/rad, springref -1.471904762 deg, damping 0.34; neck_yaw 6/0/0.24; tail_yaw 5/0/0.19; tail_pitch
     8/0/0.24. Unchanged.
  4. Leg ctrlranges become the build's reach: roll +-0.15 rad, pitch and flexion +-0.40 rad
     (`rl/trike_env.py:84-86`). Unchanged.
  5. **Removed:** the first design's mocap body `target` and its site. Main refuses any mocap body
     (`plant_contract/policy_layer.py:212-221`, `_require_no_mocap_body`; D-D59, fourth change), and every asset
     must have `nmocap == 0` and neither a body `target` nor a site `target_site` (`test_mjcf_assets.py:77`,
     `:86-112`). The source has none, so the derived asset has none: (nq, nv, nu, nkey, nmocap) = (27, 26, 12, 1, 0).
  6. `home` is the only keyframe, settled from qpos0 at ctrl 0 for 3000 x 1 ms, root x = y = 0, yaw removed, 9
     decimals: torso z 0.319100188 m, clearance -0.215 mm. Unchanged. **Computed** for this revision on the source
     asset (16 servos holding at 0): root z 0.319100 m, the same to 6 decimals.
- **Provenance**: the asset's first comment and `environments/triceratops_horridus/data/robot_model_provenance.json`
  (`mesozoic.derived-model-provenance/v1`) name the script, the source commit, path, blob and sha256, MuJoCo 3.10.0 and
  the renames, the root rename included; no branch path. Regenerate:
  `git show f6deec9f9be1566e2c5de8e0932bc4d9d99ea2fd:hardware/triceratops_build/sim/generated/trike60_C_closed_capsule_fore128.xml > /tmp/fore128.xml`,
  then `python -m environments.triceratops_horridus.scripts.derive_robot_model --source /tmp/fore128.xml`; `--check`
  compares byte for byte on MuJoCo 3.10.0.
- **Not mirror-symmetric** (IMU at y = +0.020 m, a 138 g parts reserve at y = -0.055 m, brow-horn meshes differing by
  about 3e-5 m): `derive_model_mirror` refuses it; no mirror map, as for the Compsognathus robot. Main's
  `symmetry.OBSERVATION_LAYOUTS` now maps both heading-free schemas (`symmetry.py:1025-1028`), so there is no
  per-model entry to add or drop; the module docstring and the renames' comment say nothing of a mirror map (M7).
- **Spring fold, measured on the prototype** (valid: the fold and the asset's dynamics are unchanged): the build's
  model-based trot (3 seeds x 20 s, randomisation, 10 ms) peaks neck_pitch at 1.13-1.26 N m against a 1.33 N m limit;
  held servos sit at their limit in 0.014-0.057 % of servo-steps. Filtered random leg targets at 0.2: roots within
  1 mm for 10 s; at 0.4-0.6 they part after 0.7-2.4 s, against a 4-9 s chaos baseline.

### 8.2 Timestep

The 1 ms step stays: the build trains and judges at 1 ms with kp 598.5 implicitfast servos, and D33's values are whole
1 ms steps. Eleven shared sites now take the control step as `stage_config.json`'s `timestep` (default 0.002) x
`frame_skip`: `result_bundle/evidence.py:719, 890, 1073`, `reporting/gates.py:694, 1446`,
`reporting/episode_trace.py:468`, `reporting/stance_report.py:1229, 2140`, `reporting/command_tracking_report.py:284`,
`eval_diagnostics.py:2843, 2876`. The command-tracking report records its control step and the judge refuses a report
not bound to the stage's (`reporting/gates.py:1160-1622`), and `test_tracking_window.py:294` reads `timestep` from the
walk's `[env]` with the same 0.002 default, so an undeclared timestep would misstate the tracking window and every
settle allowance. So the species takes `timestep: float = 0.001`, raises `ValueError` unless it equals
`model.opt.timestep`, and both stage TOMLs declare `frame_skip = 20` and `timestep = 0.001`. **Changed:** the recipes
do not restate them; `train_behaviors.read_recipe` copies the walk's `[env]` under each recipe's
(`train_behaviors.py:155-159`), so every recipe resolves to both (§8.7). The robot test of M8 checks both stage TOMLs
and every recipe as `read_recipe` resolves it. `frame_skip` stays the default 20, which the policy payload records.
Control step 0.02 s; a 1000-step episode is 20 s.

### 8.3 Species base and robot env

- `TriceratopsHorridusEnv(BaseDinoEnv)` (`environments/triceratops_horridus/base_env.py`) holds every hashed method
  (`_cache_ids`, `_get_obs`, `_get_reward_info`, `_is_terminated`, `_reset_episode_state` and the action scaling), so a
  later anatomical model moves no robot qualname. Attributes: `HEIGHT_LABEL = "Torso height"`; `SUCCESS_KEYS = ()`
  (D50: no course, as six of the seven models); SB3 only (`supported_training_backends = ("stable-baselines3",)`);
  `action_mapping = "home-keyframe-residual/v1"`; camera tracking `torso`; `FEET = ("fr", "fl", "rr", "rl")`.
  **Removed:** `BEHAVIOR_SECTION_LABEL`; main removed it and `SpeciesConfig.behavior_section_label` (D-D59, third
  change).
- **Constructor**: explicit, defaulted parameters only, no `**kwargs` pass-through: the task fingerprint and
  `stage_config.json` read defaults from the signature (`task_fingerprint.py:95-107`, `:192-243`;
  `config.py:1312-1353`). Defaults: `frame_skip=20`, `timestep=0.001`, `max_episode_steps=1000` (20 s);
  `forward_vel_weight=0.0`, `backward_vel_penalty_weight=0.0`, `forward_vel_max=0.25` m/s; `alive_bonus=1.0`,
  `energy_penalty_weight=0.01`, `fall_penalty=-50.0`; `posture_weight=1.0`, `height_weight=0.5`,
  `height_tolerance=0.03` m; `drift_penalty_weight`, `heading_weight`, `lateral_penalty_weight`,
  `spin_penalty_weight`, `smoothness_weight`, `gait_symmetry_weight` 0.0; `healthy_z_range=(0.19, 0.6)` m,
  `max_tilt_angle=0.611` rad (35 deg), `reset_noise_scale=0.02`; the five `perturbation_*` kwargs; `command_mode="none"`
  and `command_config=None`, which `load_stage_config` fills from a stage's `[commands]` (`config.py:558-561`); the ten
  feature kwargs, each off, with the carve-outs of off value `None` (R2): `("actuation_delay_range_s", None,
  ACTUATION_DELAY_ENV_KEYS)`, `(name, None, (name,))` per `randomize_*` key, and `("observation_noise_std", None,
  OBSERVATION_NOISE_ENV_KEYS)`. **Removed:** `course_length` and `course_radius` (D50). The progress, heading and
  penalty weights carry the shared names the behaviour env zeroes (`behavior_env.py:134-145`), and every weight the
  stand block may swap is a numeric parameter held as an attribute of the same name and read in `step()`
  (`base_env.py:551-586`).
- `_cache_ids`: `_cache_home_keyframe`; torso and floor ids (no target id); IMU offsets by sensor name (orientation 0,
  gyro 4, accelerometer 7; the base defaults differ); `_foot_sensor_groups`, one touch sensor `<foot>_normal_load` per
  foot; the height term's standing reference 0.3191 m; nominal `body_mass` 12.7506 kg; `_contact_threshold` = 0.02 x
  weight = 2.50 N; `_body_ground_geoms` = the 28 colliding subtree geoms other than the soles. Gait contact threshold
  2.0 N.
- `_reset_episode_state` (main's replacement for a target spawn, `base_env.py:718`): clears the per-episode state,
  draws nothing from `np_random`, and sets neither `_initial_pos_2d` nor `_reference_heading_2d`. `reset()` is not
  overridden: the policy-interface digest hashes `reset` for a home-keyframe mapping (`policy_layer.py:418-460`).
- **Info**: the shared reward keys, Brachiosaurus's names, and `reward_total`; per-foot MIN forces under
  Brachiosaurus's keys (`r_` and `l_` the front feet, `rr_` and `rl_` the hind); `pelvis_height` = the torso height, as
  Brachiosaurus copies it (`brachiosaurus_altithorax/base_env.py:404`), which every model's terrain-relative height
  test reads (`test_behavior_env.py:275`); `derive_stance_info`. **Removed:** `course_complete` and `reward_arrival`
  (D50).
- `TriceratopsHorridusRobotEnv` (`robot/env.py`): `embodiment = "robot"`; `action_filter_cutoff_hz =
  (exp(2 pi x 4 x 0.02) - 1) / (2 pi x 0.02) = 5.19724 Hz`, giving the shared filter (`action_filter.py:56`) the
  build's 4 Hz alpha 0.395077; `FOOT_SUPPORT = {"support": ("{s}_sole",), "sole": None, "foot_site": "{s}_load_cell"}`
  (capsules have no flat pad, `morphology.py:610-621`); `REWARD_PROBE = (0.2,)`: with no success keys the digest
  harness takes the roll amplitude alone, and the first design's `"body"` effector is refused
  (`harnesses/digest_snapshot.py:166`, `:560-569`; `test_plant_contract_digest_snapshot.py:143-151`, `:166-178`).
  `scripts/view_model.py`, run as `python -m environments.triceratops_horridus.scripts.view_model` and importing
  `paths.REPOSITORY_ROOT` for any repository path (§8.1). The shared filter seeds from the first action
  (`base_env.py:1233-1249`); the build from 0.

### 8.4 Observation schema and action mapping

- **Schema**: main's `quadrupedal-heading-free/v1` (`heading_observation.py:41-48`), declared as the plant's
  `observation_schema` in `[plants.triceratops_horridus.robot]`. **No new schema:** the first design's
  `quadrupedal-target-heading-free/v1` existed to carry the target direction and distance, which went with every
  target (D-D59), and the plant contract refuses any other schema string (`policy_layer.py:385-389`, `:402-406`). So
  commit 5 edits neither `heading_observation.py` nor `policy_layer.py`, and the payload keeps main's single
  heading-free branch (schema, no-mocap and width checks).
- **Heading-free for a robot without a magnetometer.** The schema observes no world yaw: tilt is projected gravity
  from the free root's own quaternion `qpos[3:7]`, and the root velocity is expressed in the gravity-aligned heading
  frame, both invariant to a turn about world vertical (`heading_observation.py:1-31`). The heading reaches the
  policy only through the command's yaw-rate component, `clip(yaw_gain x heading error, +-yaw_rate_max)`, which the
  env's command controller computes from the simulator's true heading (`direction_commands.py:586-596`). The policy
  therefore needs no heading estimate; on hardware the yaw-rate command would come from whatever issues commands. One
  sim-only difference is recorded, not changed: the build's steering adapter works from "the true yaw plus the gyro
  drift" (`rl/README.md`, Steering, at fb1e80c8), main's controller from the true yaw; the build's drift model, a
  random walk of 0.01 rad/sqrt(s), reaches about 2.6 deg in 20 s (`rl/README.md`, Observation noise), under the
  walk's 15 deg heading tolerance. A drift in main's controller would be a shared change, and the build's heading
  estimate is part of the steering D53 keeps in the build stack, so it stays out.
- **`_get_obs`, 59 float32** (the first design's 63 less the four target values), in main's fixed segment order
  (`policy_layer.py:327-370`), the command last (`:411-416`):

  | Segment | Width | Content |
  |---|---|---|
  | `joint_position` | 20 | `qpos[7:]`: neck_pitch, neck_yaw, tail_yaw, tail_pitch, then fl, fr, rl, rr x roll, pitch, flexion, ankle |
  | `joint_velocity` | 20 | `qvel[6:]` |
  | `torso_projected_gravity` | 3 | `heading_free_root_segments` from `qpos[3:7]` |
  | `torso_angular_velocity` | 3 | gyro at the IMU site |
  | `torso_heading_linear_velocity` | 3 | `qvel[0:3]` in the heading frame (privileged) |
  | `torso_linear_acceleration` | 3 | accelerometer at the IMU site |
  | `foot_contact` | 4 | one touch-sensor load per foot, N |
  | `command` | 3 | `self._command`: `(v_x, v_y, yaw rate)`, each over its scale; zero under `"none"` |

  `_get_obs` reads `self._command`, since the plant contract's probe writes a non-zero one (`policy_layer.py:174-200`);
  training holds the command slice's normalisation at identity, and the walk's warm start zeroes the stance's command
  columns (`command_frame.py:146-162`; `train_base.py:815-827`). No previous action (§11).
- **Inputs the robot will not have (D46).** The fixed table includes the inputs D46 names as absent from the core
  build, the body velocity (3), the foot loads (4) and the passive ankle angles (4), and also the ankle speeds (4): 15
  of the 59. The build drops the first 11 with its three flags, `obs_ankle_angles=False`,
  `obs_privileged_lin_vel=False` and `obs_foot_loads=False` ("All three off: the robot's own sensors only (D46)",
  `rl/trike_env.py:127-132`). The ankle speeds are 4 inputs that main's `joint_velocity` adds and the build never
  observed: its velocity slot is the 12 leg drives (`_obs_layout`, `rl/trike_env.py:697-698`). Main's schema cannot
  drop any of the 15, and a third schema is a shared policy-interface change. Accepting main's schema follows from
  D32's chosen option (the owner declined "full port of the build env (its observation, ...)"); D46's own run is on
  the build stack. No owner row names these 15 inputs: accepting them in the main-pipeline model, recorded as a
  sim-only deviation from the build, and leaving an own-sensors schema to a follow-up (§11) are the implementer's
  choices, not owner decisions (D43 covers only the activation). The main-pipeline policies cannot run on the core
  robot as they are.
- **Noise table** (§7): the robot table of §7 on this schema, without its target rows: `joint_position` 0.005 rad,
  `joint_velocity` 0.05 rad/s, `torso_projected_gravity` 0.02 renormalised, `torso_angular_velocity` 0.02 rad/s,
  `torso_heading_linear_velocity` 0.02 m/s, `torso_linear_acceleration` 0.2 m/s^2, `foot_contact` 2.5 N, `command` 0;
  widths 20 + 20 + 3 + 3 + 3 + 3 + 4 + 3 = 59. `docs/PLANT_CONTRACT.md` already names the schema; commit 5 adds no
  paragraph for it.
- **Actions** use `home-keyframe-residual/v1` unchanged: home ctrl is 0 and ctrlrange is the reach, so
  `_scale_action(a)` is exactly `a x reach`, the build's `neutral + a x scale`.

### 8.5 Rewards, termination, and what replaces the course

- **Terms**, one left-associated chain with Brachiosaurus's info keys: alive; forward, backward, drift
  (`_progress_terms`, `base_env.py:923-949`); heading, lateral (`_heading_terms`, `:961-980`); posture;
  `reward_target_centered_height`; spin; energy; smoothness; quadruped gait symmetry; and, on a live-command stage
  only, the base class's tracking term, `tracking_weight x exp(-(e_v/sigma_v)^2 - (e_yaw/sigma_yaw)^2)`, windowed at a
  moving command and per step at a stop (`base_env.py:1548-1619`; `direction_commands.py:784-871`). Forward, heading
  and lateral read `_forward_reference_2d()` and drift `_drift_anchor_2d()` (`base_env.py:638-666`): the spawn heading
  and position under `"none"`, the commanded heading and the command event's start under the live walk (D-D62 (1));
  `test_walk_command_mechanism.py:325-355` pins `info["forward_vel"]` to the velocity along the commanded heading. The
  first design's target-relative frame goes. No idle term, so the idle rule at a zero command (`base_env.py:982-999`)
  has nothing to skip.
- **Termination**: non-finite state; `_root_termination` (fallen below 0.19 m, too high above 0.6 m, tilt above
  35 deg); a non-sole floor contact (`body_contact`). Nothing else. **Removed:** the `course_complete` termination.
  There is no target, and under a live command any termination but `terrain_boundary` is a fall to the walk gate
  (`behavior_evaluation.py:280`), while a cruise episode counts as survived only when it reaches the horizon
  (`curriculum/command_tracking_gate.py:1104-1106`): a terminating arrival would fail every arriving episode. `step()`
  adds `fall_penalty` on termination unless `success` (`base_env.py:2047-2050`); the robot reports no success.
- **No course and no arrival bonus (D50; D39 superseded).** The owner dropped both: there is no target to arrive at,
  nothing ends a walk episode early but a fall, and the truncation D39 fixed (an arrival that ended the episode and
  paid nothing) cannot happen on main. The alternative was the Allosaurus's course, a line along the reset's
  commanded heading that ends nothing (`allosaurus_fragilis/base_env.py:30-43`, `:204-243`; D-D62 (4)), kept as a
  report figure; at the 0.15 m/s cruise (D51) a 20 s episode covers at most 3.0 m, short of D40's 4 m, and D40's
  0.3 m radius has no meaning on a line.
- **What replaces the course and the arrival: the walk's command.** Each walk episode draws one whole-episode command
  at reset (§8.6): a cruise episode is paid the tracking term on every step to the horizon, a stand episode the
  stance's weights (the stand block) and the per-step stop tracking. That walking out-earns stalling is an
  **Estimate** from the weights, nothing run, ignoring the energy and smoothness costs of walking and any height lost
  in the gait: on a cruise episode the zero-action statue earns about alive 1.0 + height 0.5 + tracking at most 0.05 =
  1.5 a step (at `speed_scale` 0.225 its velocity error is the commanded speed, 2-4 velocity sigmas, and its yaw-rate
  error up to 2 sigmas), a walker that tracks its command about 1.5 + 2.5 = 4.0; on a stand episode the statue earns
  about 4.0 (the stop is tracked). The statue's walk numbers are measured on the live walk once the model is in the
  manifest (`zero_action_baseline.py --model-id triceratops_horridus robot --stage 2 --episodes 40 --seed 3042`;
  D-D62 (12); §8.12).
- D-D62 (16) (c), that `gait_symmetry` pays a quadruped statue its whole weight, does not arise: the robot's weight
  is 0 in both stages. The first design's follow-up on the arrival bonus under `VecNormalize`'s reward clip and the
  old owner point 3 (a walker passing the target to the side) are moot: with no target there is nothing to pay and no
  side to pass (§11).

### 8.6 Stages and gates (provisional)

`stages.toml` (`mesozoic.stage-manifest/v2`): `stance` (legacy 1, recipe `stance`, deliverable), then `locomotion`
(legacy 2, recipe `walk`, `warm_start_from = "stance"`, deliverable); no recovery stage, which main allows (every chain
ends at walk, `test_stage_manifest.py:794-803`). Both TOMLs written whole: `read_recipe` reads the walk raw and does not
resolve `extends` (D-D62 (13)). Budgets: stance 11M steps, the rule for every stance (D-D70; the first design had 3M);
walk 10M, of which the cruise ramp takes the first 2M (20 %).

| | Stance | Locomotion (the walk) |
|---|---|---|
| `[env]` weights | the defaults; drift 0.5, smoothness 0.02, spin 0.02; heading 0 (a command-free heading-free stance observes no heading error: D-D60 (1), D-D59) | alive 1.0, posture 1.0, height 0.5, energy 0.01, smoothness 0.02, spin 0.02, gait symmetry 0; forward, backward, heading, lateral and drift **0** (the tracking term subsumes them, D-D62 (10)); `forward_vel_max` 0.25 kept for diagnostics and the recipes |
| `[env]` episode | `frame_skip = 20`, `timestep = 0.001`, 1000 steps, reset noise 0.02, the feature block | the same; no `[env]` line cites a stance line, so `pilot_rollout_golden.WALK_STAND_SETTINGS_SINCE_CAPTURE` needs no entry (`test_phase_d_command_plumbing.py:529-549`) |
| `[commands]` | none (a stance may not declare one, `config.py:283-287`) | below |
| Gate | `stance_quality_heading/v1` or, if the four-foot check fails, `reward_and_length/v1` (D52, below) | `command_tracking/v1`, walk profile, quadruped clause (D52, below) |
| Collapse backstop | stance floor below | none |

**The walk's `[commands]`** (Design; the distribution is pinned for every model,
`test_walk_command_distribution.py:113-130`). The walk follows main's whole-episode heading-and-speed command: one
heading within 15 deg of the spawn's, held for the whole episode with no turns, at a speed in [0.5, 1] x cruise, or a
stand. D53 amends D32 for it: "main's whole-episode walking command, and main's own command recipes that run only when
started (D42), are ported; the build's steering (its turning schedule, turning on the spot, its heading estimate) and
its heat costs stay in the build stack". So none of the build's steering is ported, in D53's terms: its turning
schedule (requests every 3-5 s turning up to 60 deg, at 0.1-0.27 m/s and up to 0.4 rad/s), turning on the spot and its
heading estimate (a drifting one); nor are its heat costs.

| Key | Value | Rule and derivation |
|---|---|---|
| `mode` | `"heading_and_speed"` | every walk (`test_phase_d_command_plumbing.py:153-165`) |
| `cruise_speed` | 0.15 m/s (D51) | equal to `behaviors/follow_direction.toml`'s: D-D62 (9) gives the walk "the model's pilot `behaviors/follow_direction.toml`'s" scaling; why 0.15, below the table |
| `speed_scale`, `lateral_speed_scale` | 0.225 (D51) | equal to the recipe's; 1.5 x cruise, as both Compsognathus models, the loosest relative tolerance of main's ratios (1.24-1.5; the quadrupeds 1.33 and 1.39); no test fixes the ratio (`test_walk_command_distribution.py:121` and `test_behavior_recipes.py:155-171` pin only walk = recipe and one contract across the recipes). It caps every request: `set_target` refuses a speed above it (`direction_commands.py:527-528`) |
| `yaw_rate_scale` | 0.6 | equal to the recipe's; 0.6 on every model |
| `speed_range` | [0.075, 0.15] | [0.5, 1] x cruise; uniform, mean 0.1125 m/s |
| `stop_probability` | 0.15 | whole-episode stands |
| `initial_heading_offset_max` | 0.26 rad | the requested heading is the spawn's turned by U(-0.26, 0.26) |
| `turn_increment_max`, `switch_jitter_s` | 0, 0 | no turns |
| `switch_interval_s` | 600 | longer than the 20 s horizon |
| `turn_slowdown`, `yaw_rate_max` | true, 0.3 rad/s | pinned |
| `tracking_weight` | 2.5 | pinned; sigmas and tolerances at their `speed_scale` defaults (`direction_commands.py:328-352`), each 0.25 / 0.2 / 0.1 x `speed_scale` / 1.5: velocity sigma 0.0375 m/s, velocity tolerance 0.030 m/s, stop-speed tolerance 0.015 m/s; yaw-rate sigma and tolerance 0.15 rad/s; heading tolerance pi/12 |
| `tracking_window_s` | 1.10 s, 55 x 0.02 s | D-D64: the stride at 0.5 x cruise. No reference walk runs on this plant, so it takes D-D64 (3)'s no-reference branch, which is written for bipeds ("the largest dimensionless stride of the repository's bipeds"): the T. rex's T sqrt(g/L) = 3.40 at Froude 0.0445, scaled as Fr^-0.2. L = 0.2953 m (**Computed**: the settled root's height above the four `_load_cell` sites, the gait module's definition; re-measured as the mean of seeds 3042-3044 on the derived asset, §8.12), Fr = 0.075^2 / (9.81 x 0.2953) = 1.94e-3, W = 3.40 x (1.94e-3 / 0.0445)^-0.2 x sqrt(0.2953 / 9.81) = 1.104 s. Quadruped cross-check (**Computed**, the same scaling from the two quadruped reference walks): the brachiosaurus (1.40 s at 0.35 m/s, L 1.163 m: 4.07 at Fr 0.0107) gives 0.99 s and the dibothrosuchus (0.80 s at 0.157 m/s, L 0.308 m: 4.52 at Fr 0.0082) 1.04 s; the adult Triceratops reference walk outside the manifest (`examples/triceratops_horridus`: 2.4 s strides at 0.259 m/s, `validation.json`; L 1.798 m, its settled root height above its four sole sites, **Computed**: 5.61 at Fr 0.0038) gives 1.11 s, the bipeds' value to 0.01 s. So the bipeds' value is longer than the two committed quadrupeds' and within 0.01 s of the adult Triceratops's, inside D-D65's 0.70-1.50 s, on the safe side of the boxcar (its residual sway rises steeply for a stride longer than W, D-D64 (1)). The implementer's choice, with the cross-check in the walk's comment; which branch a quadruped with no reference walk takes is main's maintainer's rule (D-D64), asked with commit 5 (§11). The window is a task setting: a change trains the walk again |
| `cruise_ramp_start_fraction`, `cruise_ramp_timesteps` | 0.25, 2,000,000 | pinned (D-D62 (5)) |

**Why 0.15 m/s (D51).** On main the walk's cruise is no floor the walk must clear: it sets what the robot is asked to
do. D40's 0.15 m/s becomes the fastest command; the commands are uniform in [0.075, 0.15] m/s, mean 0.1125 m/s, on the
85 % of episodes that are not stands. The gate's velocity and speed tolerances, 0.030 m/s at this scale, pass a
cruise episode that holds its command to within 0.030 m/s, so a walk as slow as about 0.045 m/s at the slowest
command, or 0.12 m/s at the fastest, passes (the speed ratio's half the command, 0.0375-0.075 m/s, is the weaker
bar). So the choice does not rest on 0.15 as a floor. D51 records one reason, heat; this design adds main's practice
in support of it:

- **Heat (D51's recorded reason).** Main's reward has no heat cost, and D53 keeps the build's heat costs in the build
  stack. The build's four 128 mm policies ran their worst drive at 0.70-0.83 of its rating at 0.15 m/s, against
  0.83-0.95 at 0.25 m/s (D51).
- **Main's practice (this design's argument, not D51's).** D-D62 (9) takes the walk's scaling from the model's pilot
  `follow_direction.toml`, whose cruise was the old walking floor on five of the seven models (the T. rex and the
  Allosaurus kept a recipe cruise above theirs). D40's walking number was 0.15 m/s, and the owner declined a 0.20 m/s
  gate in the same row.

What it gives up, as D51 records: `speed_scale` 0.225 is below the adopted design's 0.25-0.27 m/s walk
(`docs/hardware/SIZES.md:103`; D2), and it caps every request, so main's pipeline does not train the 0.25 m/s design
walk and no caller can request it.

**Stand block** `[commands.stand_weights]` (D-D62 (2), (11)): the walk's `[env]` pays every stance weight but drift,
so the block is one line, `drift_penalty_weight = 0.5  # stance.toml:N` (the line that sets it). Every setting of a
stand term is the stance's (both stages leave them at the constructor defaults), so the per-model tables of
`test_walk_stand_block.py` take empty entries: `WEIGHT_DIFFERENCES[TRI] = {}`, `SETTING_DIFFERENCES[TRI] = {}`
(`:62`, `:77`).

**What the walk's numbers mean at this scale.** A cruise command asks 0.075-0.15 m/s along a heading within 15 deg of
the spawn's; the gate's tracking fraction asks, on 60 % of the steps after the settle (1 s + |e0| / 0.3 rad/s + the
1.10 s window, at most about 3 s), that a step be in tolerance on all four counts: the windowed velocity error under
0.030 m/s, the yaw-rate error under 0.15 rad/s, the heading error under 15 deg and the speed within 0.030 m/s of the
request (`sample_in_tolerance`, `curriculum/command_tracking_gate.py:1263-1297`); its speed ratio asks at least half the
commanded speed along the heading; a stand asks the per-step root speed under 0.015 m/s and the yaw rate under
0.15 rad/s on 60 % of the steps after 1 s. For scale, the build's own policies (trained on the build's reward, not
main's, so reported only): the four 128 mm policies made 0.143-0.147 m/s at a 0.15 m/s command and 0.235-0.244 m/s at
0.25, randomised, 10 ms delay, 5 episodes each, no falls (`REPORT_FORELEG_D22_APPENDIX.md` C; the two older models'
policies 0.098-0.099 m/s at 0.10, `rl/RUNS.md`); the scripted trot 0.139-0.140 m/s at 0.15 (`REPORT_WALK60_FORE128.md`,
R3). Their straightness is the open part, and the build's figures do not measure what main's gate reads. Appendix C,
at a 0.25 m/s command only (5 episodes, the window from 2 s, 18 s): the 128 mm policies' sideways drift, mean of
episodes, was 0.42 / -0.30 / 0.48 / -0.48 m, worst episode 0.64 / -0.46 / 0.65 / -0.63 m, and their worst heading
change 9.2 / -10.0 / 11.2 / -18.1 deg. The build's drift is world travel across the reset heading
(`rl/gait_metrics.py:9`, `:128`), and most of it follows from the heading change: an 11 deg turn at 0.24 m/s over
18 s alone gives about 0.4 m. Main's 0.030 m/s tolerance applies to the windowed velocity error in the robot's own
current heading frame (`direction_commands.py:666-667`), with heading error under its own 15 deg tolerance
(`:670-674`). So the figure of the build's that bears on main's gate is the heading change, up to 18.1 deg at
0.25 m/s, past the 15 deg heading tolerance; but the D22 policies walked the build's straight task with no yaw-rate
command, which main's walk gives, so this predicts little. The body-frame sideways speed of the build gaits was not
measured. At 0.15 m/s the appendix reports drift only in its servo-delay sweep (appendix F, "delay, 0.15 m/s": nominal
plant, noise on, 8 episodes, the mean of the drift's absolute value): 0.24 / 0.24 / 0.30 / 0.35 m for the four 128 mm
policies at the 10 ms delay, again in the reset frame and with no heading change reported. Main's tracking term pays
the velocity error on every step, which the build's reward priced differently, so these figures predict nothing about
a main-trained policy.

**Gates.**

- **Stance (D52).** The kind follows the four-foot prototype check that commit 5 runs (§8.12, item 1), and commit 5
  says which kind it chose, with the check's output, in this section, the stance TOML's gate comment, the README and
  the PR description.
  - **If the check passes: `stance_quality_heading/v1`** (D-D32; the kind D-D74 moves every stance to before its next
    run), each bar measured on the robot's own zero-action panel through the report path (`stance_quality_baseline`,
    40 seeds from 3042, plus out-of-sample blocks as the other certificates did), with D52's rails, D40's numbers where
    the kind has keys: `min_avg_reward = 900.0` (D40's 900, 0.60 x the statue's 1500 measured on the prototype) and
    `min_full_horizon_fraction = 0.95` (D40's "at least 950 of 1000 steps", read as the share of panel episodes that
    reach the horizon; the three biped certificates declare the same 0.95). Of the ten required v2 keys
    (`stance_gate_v2.py:292-305`), five are **copied** from the three biped certificates, which declare them alike:
    `min_eval_episodes = 40`, `min_clean_stance_lcb = 0.80` (admits 37/40), `min_all_feet_support = 0.98` (here all
    four feet down), `max_touchdown_rate = 0.25`, `max_actuator_saturation_fraction = 0.10`. Five are **measured** on
    the statue panel (§8.12, item 2): `settle_steps` and the two settle bars (`max_settle_airborne_substeps`,
    `max_settle_peak_floor_force_bw`) from the statue's settle, `max_window_displacement_m` from the statue's travel,
    and `min_foot_load_share` re-derived for four feet (the bipeds' 0.40 is 0.8 of an even two-foot share; the even
    share is 0.25 here and the build's front share of weight 0.45-0.53, appendix C, so about 0.8 x the statue's
    lightest foot). The three biped certificates also declare optional keys against stance hacks, which this design
    does not yet commit to; every one of them is listed below, after this stance item, as a candidate for the owner,
    or excluded or not taken with its reason ("The certificate's optional keys"). All three also declare the
    statue-relative reward rail `min_avg_reward_statue_ratio = 0.60`
    (`configs/tyrannosaurus_rex/anatomical/stance.toml:1016`,
    `configs/velociraptor_mongoliensis/anatomical/stance.toml:558`,
    `configs/compsognathus_longipes/anatomical/stance.toml:545`), the rail D-D74's target gate carries. It is not
    taken, deliberately: D52 names two rails, D40's 900 and 95 % full-horizon episodes, and this would be a third.
    D40's 900 is already 0.60 x the statue's 1500 measured on the prototype, so the ratio would mostly restate it
    against each report's own statue, and needs that statue panel rolled with every report (the gate fails a declared
    ratio with no statue panel, `stance_gate_v2.py:827-843`); it is listed for the owner's review (§8.12, item 2).
    The certificate refuses a live command, pushes and a heightfield (`stance_report.py:2040-2067`), none of which the
    stance has.
  - **If it fails: `reward_and_length/v1` exactly as D40**, until revised (D52): `min_avg_reward = 900.0`,
    `min_avg_episode_length = 950`, `required_consecutive = 3`; deliverable `stance_reward_rail`, whose text says the
    statue clears it, as both quadrupeds' do. D52 says only "until revised"; this record reads the revision as the
    move to the heading certificate (D-D74), the model's next gate change. D52 does not say when it comes. Read with
    D54, the first stance run trains on the reward rail once the owner has reviewed its bars; it would wait for the
    D-D74 revision if the owner asks for that at the D54 review, or if main's maintainer confirms the reading below
    that a new model on a reward rail gets no stance session. Which applies is open (§11). `docs/NEXT_STEPS.md` §3
    (`:562-564`, `:785`, `:801-803`) and D-D74 deny a stance session to four named models until their D-D74 revisions
    land (the Compsognathus robot and the Allosaurus on `stance_quality/v1`, the
    Brachiosaurus and the Dibothrosuchus on `reward_and_length/v1`); neither names a new model. That a new model
    landing on a reward rail would get no stance session is an inference from D-D74's "each model's stance gate is
    upgraded before that model's next stance run", for main's maintainer to confirm if this branch is taken (§11).
  - Why the certificate first (D52): main is moving every stance to it (D-D74), and a reward gate passes a still
    statue (`docs/KNOWN_ISSUES.md`, HIGH: reward rails certify statue-like, three-legged or stacked-foot stances). Why
    the check first: no quadruped has declared a floor-truth kind; the v2 statistics are generic in the number of feet
    (`stance_gate_v2.py:578-579`) but untested on four, and the capsule soles have no sole criteria.
  - Either way the statue is expected to pass: under the certificate the check asks it to be clean at every heading,
    and under the reward gate it clears 900 (1500 measured on the prototype at 2c1039f4). That is an expectation,
    measured in §8.12 (items 1 and 2), not a pass. A stance the statue passes, kept as the walk's warm start as for
    Brachiosaurus, is the design's recommendation (owner point 6, §11), not an owner row.
- **The certificate's optional keys (D54; under the certificate only).** Beyond the ten required keys and the rails,
  the three biped certificates declare further optional keys (read on e94efee9; T the T. rex, V the Velociraptor, C
  the anatomical Compsognathus). This design commits to none of them yet: each is either a candidate put to the owner
  at the D54 review or excluded or not taken here, with its reason (§8.12, item 2).
  - **Candidates, for the owner.** Each is set out with the statue's value on it, the worst heading in and out of
    sample, or with the note that its statistic is not finite on four feet, which rules it out as it stands. The
    statue panels of the four-foot check give these values with no further roll: the report writes every metric to
    the panel CSV, declared or not (`reporting/stance_report.py:4111-4129`).
    - `max_foot_contact_fraction` (T, V, C: 0.02), the share of window substeps with one leg touching another
      (`stance_gate_v2.py:204` reads it as `foot_on_foot_fraction`): the bar against the stacked-foot stance that
      `docs/KNOWN_ISSUES.md` gives as a reason for the certificate (on main's Compsognathus robot, `:1608-1610`); on
      four feet it would also read a fore foot on a hind one.
    - `min_support_geom_duty` (V, C: 0.50) and `min_support_geom_coverage` (V, C: 0.80), the bars the gate names for
      a padless foot (`stance_gate_v2.py:288-291`). The capsule soles are padless and the robot declares
      `FOOT_SUPPORT["support"] = ("{s}_sole",)` (§8.3), one support geom per foot. A foot's coverage is then its one
      geom's duty (`gait/stance_metrics.py:384-390`, `:417-418`), so on this foot the two keys read one number: the
      share of window steps on which the least-loaded sole capsule itself carries load, where `min_all_feet_support`
      reads the floor force on every geom of the leg (a reading of the code, not measured).
    - `min_foot_load_share_windowed` (T, V, C: 0.35), `min_foot_load_share_statue_ratio` (V, C: 0.80),
      `max_phantom_support_fraction` (T, V, C: 0.05; against a touch sensor reading a load the floor does not carry,
      such as the sole-on-sole force the same issue names), `max_nonfoot_load_fraction` (T 0.02; V, C 0.01),
      `max_settle_stance_width_change_m` (T 0.08, V 0.05, C 0.011 m), `max_settle_touchdowns` (T, V: 2) and
      `max_episode_yaw_change_deg` (T, V: 25.0).
    - The panel rail `max_hop_or_fall_episodes` (T, V: 1; the Compsognathus declares none, "adding one is a separate
      threshold decision", `configs/compsognathus_longipes/anatomical/stance.toml:354-355`), and the window hop pair
      `max_window_airborne_substeps` (C: 40) and `max_window_peak_floor_force_bw` (C: 2.0), which read a hop on a
      plant whose flights are shorter than half a control step (D-D26) and, once declared, count toward that rail
      (`HOP_OR_FALL_KEYS`, `stance_gate_v2.py:255-271`).

    Four of these count feet or depend on the plant, and would be re-derived on the statue as `min_foot_load_share`
    is, not copied: the windowed share (the bipeds' 0.35 is 0.7 of an even two-foot share), the settle touchdowns
    (summed over the feet: the bipeds' 2 admits one landing and one re-plant on two feet), the window airborne
    substeps (substeps with every foot unloaded) and the stance-width change (a length: the largest change over the
    six foot pairs, `gait/stance_metrics.py:340-345`, `:457-462`). The rest are fractions, ratios, degrees, episode
    counts or body weights, and could carry over as numbers.
  - **Excluded.** The sole bars `max_sole_tilt_deg` (T, C: 2.0), `max_sole_corner_lift_m` (T 0.004, C 0.003 m),
    `min_sole_contacts` (T, C: 1.5) and `max_sole_cop_fore_aft` (T 0.80, C 0.70): with `FOOT_SUPPORT["sole"] = None`
    every sole metric is NaN (`gait/stance_metrics.py:508-528`), and an unmeasured criterion fails its episode
    (`stance_gate_v2.py:25-30`), so declared they would refuse the statue on every episode (the Velociraptor's
    certificate says so of `max_sole_cop_fore_aft`, `configs/velociraptor_mongoliensis/anatomical/stance.toml:472-474`).
    The schema's two optional criteria that no certificate declares go too, as they would be unmeasured as well:
    `max_sole_tilt_excess_deg` is a sole bar, and `max_foot_cop_fore_aft` reads a declared ankle and support span
    (`floor_contact/morphology.py:150-153`, `gait/stance_metrics.py:567-573`), which the robot's `FOOT_SUPPORT` does
    not declare.
  - **The hop watch: not taken** (the implementer's choice). The T. rex's and the Velociraptor's
    `hop_watch_max_unsupported_duty = 0.05`, `hop_watch_after_timesteps = 7000000` and `hop_watch_stop = true` are
    no bar: they stop training early and "enter no gate, recipe or task digest" (`curriculum/hop_watch.py:25-28`).
    Both anchor the step to their `ent_coef_decay_timesteps`, where the entropy bonus reaches 0 and a hop can lock in
    (`test_curriculum_hop_watch.py:83-99` pins it); the robot's PPO has `ent_coef` 0 throughout (D43), so it has no
    such step, and any other step would be a number with no measurement under it. A hop the step-level bars can see
    still fails the certificate's required bars at the end, as the T. rex's seed-44 hops did
    (`stance_gate_v2.py:255-264`). A hop whose flights are shorter than half a control step (about 10 ms here: 20
    substeps of 1 ms, `gait/events.py:30-32`) reads every step-level bar clean (D-D26,
    `docs/BEHAVIOR_RECIPES_PLAN.md:1526`), and only the window hop pair, a candidate above, reads it. Whether the
    robot's hops are that short is not measured. So the choice rests on the missing anchor step, not on the required
    bars catching every hop; the owner may ask for the watch, or for the window pair, at the D54 review.

  Which candidates a four-foot stance takes is measured on the prototype, ideally with the scripted hack checks the
  biped certificates were set with, and put to the owner with the other bars (D54; §8.12, item 2). Until then this
  design does not claim the certificate refuses the hacks a reward rail admits on four feet.
- **Stance collapse backstop** (the implementer's choice). Under the certificate: the relative floor of the two
  heading-certificate stances that declare one, three keys: `collapse_peak_floor_reference` = the statue's absolute
  stance reward, measured on the robot's 40-seed statue panel as both measure theirs (`zero_action_baseline.py
  --model-id ... --episodes 40 --seed 3042`, at the stance's reset noise); `collapse_peak_floor_fraction = 0.45` (the
  T. rex's and the velociraptor's, `configs/tyrannosaurus_rex/anatomical/stance.toml:1284`, `:1119`;
  `configs/velociraptor_mongoliensis/anatomical/stance.toml:564`, `:585`); and `collapse_peak_warmup_timesteps =
  1000000` (both declare it), since the untrained policy is the statue and its peak clears 0.45 x the reference at
  once; with `statue_constants_physics_revision = 1` (`test_statue_constant_freshness.py:47-80`). No floor, the
  anatomical Compsognathus's choice (it declares the reference and omits the floor "until learning runs on this
  anatomy establish a peak-return threshold", `configs/compsognathus_longipes/anatomical/stance.toml:546-549`), is
  not taken: the two heading-certificate stances with a backstop are the nearer precedent. Under the reward gate:
  `collapse_peak_floor` = 0.75 x the robot's measured statue return, rounded to the nearest 10 (Brachiosaurus's rule,
  `configs/brachiosaurus_altithorax/anatomical/stance.toml:75`; about 1125 on the prototype's 1500, so 1120 or 1130
  until the robot's own return is measured, §8.12, item 2) with Brachiosaurus's keys and
  `collapse_peak_warmup_timesteps = 1000000`, because under `home-keyframe-residual/v1` the untrained policy is the
  statue and sits above the floor, so without a warm-up the backstop arms on the initialisation peak
  (`configs/brachiosaurus_altithorax/anatomical/stance.toml:75-95`). Either value is measured on the robot's own
  40-seed panel once it is in the manifest (CONTRIBUTING.md, "Adding an embodiment", step 2).
- **Walk collapse backstop: none** (the implementer's choice), as the two Compsognathus walks and the Allosaurus's: a
  matched-basis relative pair (D-D62 (15), D-D65 (1)) needs a reference walk on this plant under a command it matches,
  which does not exist (§11). A collapse after the robot has started to walk is therefore not caught early.
- **Walk (D52).** `gate_kind = "command_tracking/v1"`, `profile = "walk"`, judged once after the stage on the
  certification panel planned from seed 3042, the handoff chosen on a selection panel from 2042, never in training
  (`manager.py:362-365`; `test_species_catalog.py:1135-1136`). Thresholds `PROVISIONAL_WALK_THRESHOLDS`, each line
  naming its source as the other walks do (`command_tracking_gate.py:487-519`): 20 cruise and 5 stand episodes, cruise
  survival LCB 0.80 (only 20/20 clears), success LCB 0.60 (17/20 clears), tracking fraction 0.60, speed ratio 0.50,
  stand hold 0.60, stand success 0.60, no stand falls. Quadruped gait clause
  (`PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS`, `:576-587`): `gait_limbs = 4`, `max_flight_fraction = 0.05`,
  `min_limb_duty = 0.20`, `min_pair_alternation_index = 0.75`, and `min_lead_swaps_per_stride = 1.0`, which
  `test_species_catalog.py:354-390` pins on every quadruped walk. Froude at the cruise 0.15^2 / (9.81 x 0.2953) =
  0.0078 (0.0019 at the slowest command; **Computed**); a quadruped's flight cap is required whatever its Froude number.
  None of these is a per-model number: the walk bars are main's shared constants, to be frozen from main's first walk
  pilot, and `test_species_catalog.py:640-668` pins each model's walk gate to them; D52 takes them as they are, since
  model-specific walk bars would split a shared design main's maintainer owns. How the robot's walk was derived
  instead: the cruise from D51, the tolerances from the cruise, the window from the Froude rule, and the statue as the
  floor: it should survive every episode and fail the cruise family at 0 of 20 on the speed ratio, as every model's
  statue does (D-D62 (18); **Estimate** for this model, measured with the walk statue panel, §8.12). D40's walking
  reward 2000 and 550 steps have no key on this gate and become report-only (D52): the README gives the walk statue
  panel's mean reward and length beside them, and a trained walk's are read from its certification panel's per-step
  records (`command_tracking_panel.py:351-360`), not gated. The bars are provisional and no controller has passed a
  committed walk panel (D-D62 (16)); nothing in the robot's docs calls them attainable.
- **The first run (D54).** D40's "revised with the owner after the first main-pipeline run" meets main's run procedure:
  the SB3 notebook's chain loop writes the summary and the result bundle after every node, before its gate check
  (`save_run_bundle(chain_results(), ...)`, "so every certified deliverable above a failure is published",
  `notebooks/sb3_training.ipynb` cell 20 at a232da31); a bundle is `complete` when the target node is present and
  certified (`save_run_bundle`'s docstring, cell 15), and a complete bundle is immutable: its verdict is never
  re-judged, and after a gate edit a run trunked on it trains the node again (D-B8 as amended 2026-11-24; D-D73 (3)). A
  run whose bundle is not complete, such as a first stance that fails its provisional bars, can be re-judged under
  edited bars without training ("edit a threshold, re-judge, never retrain", D-B8 as amended). So, as D54 decides:
  before the first stance run the owner reviews the stance bars as measured on the statue panel: every bar of the
  stance's kind with the statue's value on it, and the open question of which optional bars a four-foot stance takes
  (§8.12, item 2; no training cost); the first run is a certification run, published as the chain publishes it; a
  revision after it costs nothing if that stance failed (it is re-judged), and if it certified, its verdict stands under
  the bars it was judged by and a revision the walk must meet trains the stance again (an 11M stance, 20-40 h at 4
  environments, below) and the walk again if it trained on that stance. The walk starts only once its stance is
  certified and published, with the warm-start noise scan first (D-D72; `docs/NEXT_STEPS.md` §3). D40's revision reaches
  the stance bars. That the walk bars move only with main's is this record's reading of D52, which puts the walk on
  main's shared gate but does not say that D40's revision stops at the stance; the revision reaches the walk only
  through D40's report-only 2000 and 550.

**Training cost and the number of environments (D55).** Main's notebook trains with `N_ENVS = 4` (its configuration
cell; "notebook defaults unless stated (`N_ENVS = 4`, ...)", `docs/NEXT_STEPS.md:536-538`), a notebook setting no stage
TOML sets. Main's own record: the Compsognathus robot's 11M-step stance took 19h31m on Colab at those defaults
(`docs/NEXT_STEPS.md:898`, `:916`; September 2026, at that month's evaluation cadence), about 157 env steps/s over the
whole stage. This robot steps 20 physics steps of 1 ms per control step against that robot's 10 of 2 ms, about 2.2 x its
physics cost per control step: D55's figure, an **Estimate** here, not pinned. It came from a scratch timing of physics
alone on MuJoCo 3.10.0 (the source asset against the Compsognathus robot's) whose output was not kept, and repeats of
the same script in review on a shared 4-CPU sandbox gave ratios from about 1.5 to 2.75 between runs; commit 5 measures
the real figure (§8.12, item 3). So an 11M stance may take 20-40 h (**Estimate**, D55): the policy update and the
evaluations do not scale with the physics, and the cadence has changed since September. That can be more than one Colab
session of about 24 h, which must also hold the certificate and the bundle (about 0.8 h for the T. rex,
`docs/NEXT_STEPS.md:849-852`). Commit 5 measures it (§8.12, item 3). If an 11M stance at 4 environments would not fit
one session, the robot's README asks for 8 environments on a runtime with 8 or more CPUs, as the build's runs used (8
envs on 8 vCPUs, `rl/RUNS.md` at fb1e80c8), and says why; otherwise it keeps main's 4.

**Samples per update: a declared deviation, not pinned.** The PPO block keeps the build's per-env rollout, `n_steps =
1024`, and its minibatch of 2048. At main's 4 environments an update takes 4 x 1024 = 4096 samples, half the build's 8 x
1024 = 8192: each epoch runs 2 minibatches, not 4, and the policy updates twice as often per env step. Not pinning it is
the implementer's choice, not an owner decision (D43 covers only the activation; D55 decides the environment count the
README asks for, not the samples per update). This design records it as a deviation from the build and does not pin it:
the environment count is the notebook's, not the stage's, and holding 8192 at 4 environments would need `n_steps =
2048`, a per-env rollout of about two whole 1000-step episodes, which would double again at 8. With 8 environments,
D55's case, the update is the build's. The stage TOMLs' `[ppo]` comment and the README say so, and, like D43's, it means
the first main-pipeline run is not a like-for-like comparison with the build's runs.

**Feature block**, in both stages, each line's comment citing its decision, unchanged: `actuation_delay_range_s = [0.0,
0.02]`, `actuation_delay_eval_s = 0.01` (D33); `randomize_friction = [0.6, 1.0]`, `randomize_body_mass_scale = [0.9,
1.1]`, `randomize_actuator_kp_scale = [0.9, 1.1]`, `randomize_actuator_force_limit_scale = [0.85, 1.0]` (D32);
`randomize_total_mass_scale = [0.9, 1.1]`, `randomize_payload_kg = [0.0, 1.0]` (D38, training only);
`observation_noise_eval = true` (D37) and the table of §8.4 (D32). The recipes inherit it (§8.7). On the live walk
every evaluation round rolls the same per-episode draws, fixed by the evaluation seed (§3); the gate panels are
evaluation envs, paired per seed.

**PPO and SAC** (D43; changed only in the walk's warm-up entropy, below): PPO (`rl/train.py:108-121`, `:731-737`) lr
3e-4 to 3e-5 linear, `n_steps` 1024, batch 2048, 5 epochs, gamma 0.99, GAE 0.95, clip 0.2, `ent_coef` 0,
`max_grad_norm` 1.0, [256, 256] for the policy and the value net, `log_std_init` -1.6, SB3's default Tanh (the stage
TOMLs' `[ppo]` comment says so, with the samples per update above); SAC lr 3e-4, batch 256, buffer 300,000, starts
10,000, `train_freq` 4, 4 gradient steps, and `ent_coef = "auto"` written out, as every committed `[sac]` block but the
Allosaurus's writes it (the Allosaurus's two leave SB3's default, also `"auto"`;
`configs/allosaurus_fragilis/anatomical/locomotion.toml:83-85`) (both stages need `[sac]`:
`test_walk_species_training.py:81` trains the walk under PPO and SAC). The walk declares its stage-entry warm-up keys
explicitly, as the Compsognathus robot's does (`warmup_timesteps = 100000`, `warmup_clip_range = 0.02`), with
`warmup_ent_coef = 0.005`, the smallest committed value (both Compsognathus walks; the others declare 0.018-0.02, and
the Allosaurus takes the 0.02 default), not the shared 0.02 default (the implementer's choice). It must be positive:
`train_base.py:1523-1528` builds a `StageWarmupCallback` for every `initialize_next_stage` load, and under SAC it seeds
the auto-tuned coefficient with `log_ent_coef.data.fill_(math.log(warmup_ent_coef))`
(`curriculum/advancement.py:509`), so 0.0 raises `ValueError` at the start of every SAC warm start, the walk smoke of
`test_walk_species_training.py:146-158` included (`gate_schema.py:259-268` never checks the range). Under PPO this
adds an entropy bonus of 0.005 for the first 100k walk steps, then restores `ent_coef` 0; the build's PPO never had
one, a recorded deviation. A warm-up that skips the SAC reseed at 0 would be a shared change to
`StageWarmupCallback`, main's maintainer's to make (§11).

### 8.7 Behaviour recipes (D42, D53)

- **All 11 ship**, still required: `test_behavior_recipes.py:21-33`, `:49-60`; the wheel job
  (`.github/workflows/python-ci.yml:189-212`); `test_model_identity_contract.py:166-176`; `read_recipe`'s SB3 check
  (`train_behaviors.py:76-77`). Each is one of main's own command tasks (`DirectionCommandConfig`), not the build's
  steering mode. D42 ships all 11 because "they run only when someone starts them, and no shared test changes", and
  D53 ports them as "main's own command recipes that run only when started (D42)", so the recipes that train heading
  and speed commands (`follow_direction`, `follow_direction_speed`) are inside the amended D32, not steering it keeps in
  the build stack. Each adds 2 digest lines (22). No recipe is exempted. D42's "no shared test changes" is set against
  its other option, exempting the two steering recipes, which would have changed the shared recipe lists; this record
  reads it so. One shared test change is still needed: the terrain-scale test hard-codes the control step
  (`frame_skip = 10 if compsognathus else 5`, `test_behavior_recipes.py:191-193`), which gives the robot 1250 x 5 x
  0.001 = 6.25 s, not 25 s, so it must read the model's own `frame_skip` and timestep (§8.9). That is an every-model
  generalisation a 1 ms model needs whatever recipes ship, not an exemption; if D42 is read strictly it is the owner's
  to confirm (§11).
- **What they inherit.** `read_recipe` copies the walk's `[env]` raw under the recipe's own, and nothing else: no
  `[commands]`, stand block, `[ppo]`, `[sac]` or `[curriculum]` (`train_behaviors.py:155-159`). So `frame_skip`,
  `timestep` and the whole feature block reach every recipe from the walk, and each recipe's `[env]` holds only
  `max_episode_steps = 1250` (25 s) and `course_distance`, as every model's do.
- **Sections.** `test_behavior_recipes.py:63-68` requires each recipe's tables to be exactly `behavior`, `commands`,
  `terrain`, `env` and `ppo`, plus `terrain_sampler` on the nine recipes outside `follow_direction` and
  `follow_direction_speed`. Since the walk's `[ppo]` does not reach a recipe, each recipe writes its own: `learning_rate
  = 0.00005`, `ent_coef = 0.005`, `target_kl = 0.03`, the table all 77 committed recipes carry. `[terrain_sampler]`
  over `TERRAIN_FAMILIES` in their order (flat, sloped, bumps, depressions, mixed, terrain_contact) is fixed by the test
  (`:103-114`): on `difficult_terrain` and `follow_direction_difficult_terrain` 1 on each family but `terrain_contact`
  0; on the other seven `flat = 1`, the recipe's own family 3 (`terrain_contact` for `terrain_contact`, else its
  template), the rest 0.
- **What the behaviour env does.** It zeroes the progress, heading and penalty weights by name
  (`behavior_env.py:134-145`) and refuses a recipe command with stand weights, a cruise ramp, a tracking window or its
  own `tracking_weight` (`behavior_env.py:97-109`; D-D62 (8), D-D64).
- **One command contract**, equal across the 11 (`test_behavior_recipes.py:155-171`) and, for the four scaling fields,
  equal to the walk's (`test_walk_command_distribution.py:52-53`, `:121`): cruise 0.15 m/s, `speed_scale` and
  `lateral_speed_scale` 0.225 (D51), `yaw_rate_scale` 0.6, `yaw_gain` 1.5, `yaw_rate_max` 0.3, `turn_slowdown` true,
  `minimum_turn_speed_fraction` 0.25 (`yaw_rate_max` written out in every recipe; `yaw_gain`, `turn_slowdown` and
  `minimum_turn_speed_fraction` left at their `DirectionCommandConfig` defaults, as all 77 committed recipes leave
  them). Per recipe, as every committed model writes them and as `test_behavior_recipes.py:80-89`
  requires (`(turn_increment_max > 0) == turning`, `(speed_range is not None) == variable_speed`,
  `(stop_probability > 0) == variable_speed`):
  - the four variable-speed recipes (`follow_direction_speed`, `combined_terrain`, `combined_mixed_terrain`,
    `follow_direction_difficult_terrain`): `speed_range = [0.075, 0.15]` and `stop_probability = 0.15`;
  - the five turning recipes, every name starting `follow_` or `combined_` (`follow_direction`,
    `follow_direction_speed`, `follow_direction_difficult_terrain`, `combined_terrain`, `combined_mixed_terrain`):
    `turn_increment_max = 0.5235987756`, `switch_interval_s = 6.0`, `switch_jitter_s = 1.0`, `straight_probability =
    0.25`;
  - the six non-turning recipes (`sloped_terrain`, `bumps_terrain`, `depressions_terrain`, `mixed_terrain`,
    `difficult_terrain`, `terrain_contact`): `turn_increment_max = 0.0` written out (the field's default is pi/6, which
    the test reads as turning) and `switch_interval_s = 6.0`, no jitter, no `speed_range`, no `stop_probability`.

  These replace the first design's cruise 0.25, range [0.1, 0.25], yaw rate 0.4 and 4 s switches.
- **Terrain** (Design, the implementer's choice: each value by a rule read off the seven committed
  `combined_mixed_terrain.toml` files on e94efee9, nothing run), against `test_behavior_recipes.py:174-198` and
  `TerrainConfig`'s checks (`terrain.py:65-117`), with the root height 0.3191 m and the footprint radius 0.660 m
  (**Computed** on the source asset with the test's own footprint rule). One table, the same on all nine terrain
  recipes but for `mode` and `template`, as every model's recipes share theirs (and the unified pair must equal
  `mixed_terrain`'s and `combined_mixed_terrain`'s); the four `feature_*` lines only on the six recipes whose template
  is not `sloped` (`bumps_terrain`, `depressions_terrain`, `mixed_terrain`, `combined_mixed_terrain`,
  `difficult_terrain`, `follow_direction_difficult_terrain`), as every model writes them; `follow_direction` and
  `follow_direction_speed` hold `enabled = false` alone (`:92`).

  | Key | Value | Rule |
  |---|---|---|
  | `enabled`, `mode`, `template` | `true`; `"gentle"`, `"flat"` on `terrain_contact`; the test's template | fixed by the test (`:95-104`) |
  | `extent`, `nrow`, `ncol` | 5.0 m, 201, 201 | dx 0.05 m; the extent is above cruise x 25 s + 0.66 = 4.41 m |
  | `apron_radius` | 1.0 m | above 0.66 + the 0.071 m cell diagonal = 0.73 m; 1.5 x the footprint, inside the committed models' 1.2-1.9 x |
  | `blend_width` | 0.5 m | 0.5 x the apron, as five of the seven (0.48-0.53: the Compsognathus robot, Brachiosaurus, Dibothrosuchus, T. rex, Velociraptor); apron + diagonal + blend = 1.57 m, inside the 5.0 m extent |
  | `max_height` | 0.4 m | 0.08 x the extent, as five of the seven (the T. rex and the Allosaurus 2.0 at 35 m) |
  | `max_slope_degrees` | 3.0 | every model |
  | `roughness_amplitude` | 0.0045 m | 0.75 x the feature height, as six of the seven (the Allosaurus 0.43 x) |
  | `wavelength_min`, `wavelength_max` | 0.3 m, 0.75 m | the minimum at least 4 dx = 0.20 m; the maximum 2.5 x the minimum, as both quadrupeds and the Velociraptor (the others 2.67-3 x) |
  | `episode_variation` | 0.15 | every model; the test asks it positive (`:115`) |
  | `feature_height` | 0.006 m | 1.9 % of the root height, inside the test's 1-3 % band |
  | `feature_radius_min`, `feature_radius_max` | 0.20 m, 0.35 m | the minimum at least 3 dx = 0.15 m; a ratio of 1.75, inside the committed 1.56-1.78 |
  | `feature_density` | 3.0 per m^2 | 0.12 of the 1 / r_min^2 = 25 per m^2 bound; the committed models' density x r_min^2 is 0.10-0.23 (the T. rex, the Allosaurus and the Velociraptor 0.12, Brachiosaurus 0.13, Dibothrosuchus 0.10) |

  `[env]`: `max_episode_steps = 1250` and `course_distance = 3.0` m, below cruise x 25 s = 3.75 m as the test asks; a
  radial-progress diagnostic, not a gate. Four of the seven models set it at 0.4 x cruise x 25 s, which would be
  1.5 m here; either passes, and the choice is the implementer's. Timesteps and samplers as every model's (the unified
  pair exactly 3,000,000).
- **None can be trained from the committed walk.** `prepare_behavior_checkpoint` refuses a parent whose task
  fingerprint has a `command` section (`behavior_checkpoint.py:249-262`), and every committed walk has one since D-D62:
  main's records say the recipes have no parent a user can produce until a follow node or a command-free walk exists
  (D-D62 (17); `docs/ROADMAP.md:72-74`). CI trains each model's recipes for a few steps from a fabricated command-free
  parent only (`test_behavior_species_training.py:38-75`), at full depth. The README says the robot's recipes are
  inert and why. Behaviour training would still draw the delay and whole-robot factors, since `train_behaviors` marks
  its env (§4.3, M10).

### 8.8 Manifest and generated files

- `configs/species_manifest.toml`: `[species.triceratops_horridus]` appended last (display order 7; `model_ids()` must
  equal the display-name tables in order, `test_species_names.py:97-98`) and its `robot` embodiment (tagline
  "Twelve-Servo 60 cm Quadruped Prototype", gait "Quadrupedal", entry point
  `environments.triceratops_horridus.robot.env:TriceratopsHorridusRobotEnv`, the asset, `training_backends =
  ["stable-baselines3"]`, notebook `sb3_training`). **Success metric**: key `command_tracking`, label "Walk command
  tracking (command_tracking/v1)", definition word for word the walk deliverable's, Brachiosaurus's text
  (`species_manifest.toml:199-217`) with its quadruped clause and lead-swap bar, as `test_species_catalog.py:705-741`
  requires (it must not name `course_complete`, `:736`; D50). **Deliverable metrics**: the walk's, the same text; the
  stance's names the kind commit 5 chose (D52; `species_catalog._build_deliverable_metrics`): `stance_quality` with
  the six-heading certificate's wording (the anatomical Compsognathus's text as the template, for four feet), which
  must name `stance_quality_heading/v1`, keep the phrases "stance_quality/v2 decision" and "-90, -45, 0, 45, 90 and
  180 degrees", and not name `stance_quality/v1` (`test_species_catalog.py:1823-1828`); or `stance_reward_rail` with
  Brachiosaurus's wording, which must name `reward_and_length/v1`, "statue" and "certified stance quality"
  (`:1805-1809`).
- `configs/plant_versions.toml`: **note 25** (note 24 is D-D59's revision of all seven plants,
  `plant_versions.toml:195`): the source commit and sha256, D25, the edits of §8.1 with the root rename, the schema
  `quadrupedal-heading-free/v1`, and which hardware revision physics r1 is; `[plants.triceratops_horridus.robot]` at
  revisions 1/1/1 with `observation_schema = "quadrupedal-heading-free/v1"`.
- Order, on MuJoCo 3.10.0, after the formatting step of §9 (M2): the manifest entry; the measurements of §8.12 and the
  statue measurements every statue-derived number needs (CONTRIBUTING.md step 2; D-D62 (12): `stance_quality_baseline`
  and `zero_action_baseline --stage 1` and `--stage 2`, 40 episodes from 3042), written into the stage TOMLs; then
  `plant_contract --write`, `harnesses.digest_snapshot --block-optional-backends --write`, `species_catalog`
  (CONTRIBUTING step 8's order; each `python -m environments.shared.<module>`); then every gate of §9 against
  a232da31.
- Counts. **Estimate**, read off the Compsognathus robot's lines: the digest snapshot 663 to about 753 lines (17 plant,
  29 policy, 22 behaviour, 14 stage, 8 reward), every added line the robot's. The `--exact` capture (33,208 lines at
  a232da31) grows by the robot's lines only; both measured when built. The first design's 725 to 815 and 40,568 to
  43,984 were measured at 2c1039f4 and no longer apply.

### 8.9 Shared tests to generalise

`TRI = ModelId("triceratops_horridus", "robot")`. Files under `environments/shared/tests/` unless a path is given.

**Tables the first design listed, rechecked on the merged tree:**

| File and table | TRI entry or change |
|---|---|
| `test_mjcf_assets.py` `EXPECTED_MODELS` (`:27`, coverage `:56`) | `(27, 26, 12, 12.750593)`; the asset also meets the every-model `nmocap == 0` and no-target checks (`:77`, `:86-112`) |
| `test_behavior_env.py` `MODEL_IDS` (`:43`), `SPEEDS` (`:52`, coverage `:135-138`) | `SPEEDS[TRI]` = the recipe cruise, 0.15 |
| `test_plant_contract_digest_snapshot.py` `_SUCCESS` (`:73`), `_POSE_REASONS` (`:83`) | `_SUCCESS[TRI] = None` (D50); `_POSE_REASONS[TRI]` from the golden; `_ROLL_STANDS = {TRI}`, a new table, and the roll-must-end assertion (now `:322`) inverted for its members, if the harness's fixed kick still does not topple the robot (measured on the prototype at 0.2, 0.5 and 1.0; re-measured, §8.12); new test logic, a shared test change (§11) |
| `test_plant_contract_layers.py` `SB3_INTERFACE_QUALNAMES` (`:423`, coverage `:463-469`) | `TriceratopsHorridusEnv._get_obs`, `._scale_action`, `._cache_ids` |
| `test_plant_contract_manifest.py` (`:47`, `:57`, `:68`, coverage `:75-87`) | 59 / 12 / (59, 12) |
| `test_gait_morphology.py`, nine tables (`:43`-`:133`, coverage `:172-189`) | weight 125.08 N; feet `("fr", "fl", "rr", "rl")`; root **`"torso"`**; sole `None`; `GENERIC_ONLY`, `REGISTRY_ONLY` `()`; `SUPPORT_SPAN` `None`; `DIGIT_MTP` `{}`; a TRI branch for the leg tops beside the Compsognathus robot's at `:231-234`: every other model's tops are `<label>_thigh`, the robot's are `[f"{label}_roll_yoke" for label in FEET[TRI]]` (the source's leg tops are `<side>_<end>_roll_yoke`, renamed with the leg prefixes, §8.1), a shared test change (§11) |
| `test_gait_pad_telemetry.py` (`WHOLE_FOOT` `:436`, `PAD_FEET` `:446`, `RAIL_FEET` `:611`), `test_gait_recorder.py` (`MODEL_IDS` `:64`, `STATUE_COP_FORE_AFT_CEILING` `:718`) | `(False, 0)`, `{}`, `()`; ceiling `None` |
| `test_species_names.py` (`:29`, `:37`) | "Triceratops horridus", "Triceratops horridus (Robot)", appended last; the notebook's species list equals `species_ids()` (`:129-141`) |
| `test_species_catalog.py` | `MODEL_IDS` (`:65`); dims (`:104-127`) `(59, 12, 27, 26, 12, 12.750593)`; schema (`:198`) `quadrupedal-heading-free/v1`; revisions (`:272`, `:281`, `:288`) 1/1/1; the early-gates test's five stage-1 dicts (`:433`, `:460`, `:468`, `:543`, `:568`) for the stance kind commit 5 chose (D52); TRI in the quadruped set (`:655`), so its stage 2 equals `_command_walk_gate(biped=False, ...)`; the success-semantics test (`:705-741`) passes on the manifest text, unchanged |
| `test_species_integration.py` | `MODEL_ENVS` (`:38`), `SPECIES_DIMS` (`:59`) `{"obs": 59, "act": 12}`, `SPECIES_REWARD_KEYS` (`:68`) the terms of §8.5 and `reward_total` (no `reward_arrival`); `test_gym_make`; not `EXEMPT` (`:35`, M3); `:399` unchanged (no `course_complete`, D50) |
| `test_stage_manifest.py` (`WALK_ONLY_MODELS` is gone) | `test_recipe_labels_per_model`: `expected[TRI] = ("stance", "walk")` beside the Allosaurus (`:713`); `test_v2_manifest_does_not_move_any_dirname`: TRI excluded beside the Allosaurus (`:840`), having joined after the v2 migration |
| `test_curriculum_manager.py:404` | `len(model_ids()) == 8` |
| `test_symmetry.py:112-119` | a `NOT_MIRRORED = {ROBOT: ..., TRI: ...}` table, each with its reason (the robot: an IMU at y = +0.020 m, a parts reserve at y = -0.055 m and brow-horn meshes about 3e-5 m apart, §8.1), in place of the `if model_id == ROBOT:` special case; a shared test change (§11) |
| `test_behavior_recipes.py:191-193` | `frame_skip` from the recipe as `read_recipe` resolves it (the walk's 20), else the class default, not `10 if compsognathus else 5`: an every-model generalisation, not an exemption, and the one shared test change the recipes need (§8.7, D42) |
| `test_model_identity_contract.py` | passes once §8.11's files are tracked (`:166-176`, `:229-262`) |
| Feature tests | `OPTED_IN_ACTUATION_DELAY` (`test_actuation_delay.py:58`), `OPTED_IN_EPISODE_PHYSICS` (`test_episode_physics.py:74`), `OPTED_IN_OBSERVATION_NOISE` (`test_observation_noise.py:81`) gain TRI (their `listing == table` tests fail until then); the exact-delay test gains the robot |

**Per-model tables new since 2c1039f4** (each has a coverage check or is indexed by model):

| File and table | TRI entry |
|---|---|
| `tests/heading_free_helpers.py:97` `HEADING_FREE_ENVS`, `test_heading_free_observation.py:79` `EXPECTED_LAYOUTS` (coverage `:119-125`) | `TriceratopsHorridusRobotEnv`; `(QUADRUPEDAL_HEADING_FREE_SCHEMA, 59)`; `PAIRED_ENDS` (`:101-104`) only if the statue falls inside 250 steps |
| `test_heading_free_observation.py` `WILD_ACTION_SCALES` (`:89`) | `TriceratopsHorridusRobotEnv: s`, the smallest scale at which reset seed 3042 terminates before 1000 steps under the test's `default_rng(1)` stream, measured on the derived asset in the env the test builds (`:243-260`; `assert terminated`, `:258`). **Computed** in review on the source asset without the features (4 Hz filter seeded from the first action, ctrl = a x reach, head and tail held, §8.5's termination rules): at 0.8 seeds 3042-3044 all stand 1000 steps; at 2.0 they end at steps 450 and 332 and 3044 stands; at 5.0 seed 3042 ends at step 127 and 3043 and 3044 stand. So the outcome is seed-specific: the entry pins seed 3042's outcome, and a plant or feature change can make it stand. If no scale ends seed 3042 on the derived asset, commit 5 records that and proposes another route to a terminal state for this model (for example the root tilted past 35 deg before a step), a shared test change for main's maintainer to review |
| `test_heading_free_observation.py` `_stance_env` (`:107-115`), a shared test change | The robot is the first model whose committed stance turns a feature on, and D37's `observation_noise_eval = true` noises every step observation of an evaluation env (`base_env.py:1864`, `:2088-2094`). `test_layout_matches_segments_and_command_is_last` (`:148-209`: joint position and velocity, gyro, accelerometer and foot channels equal to the raw state bit for bit after 5 steps) and `test_local_sensors_are_not_rotated_twice` (`:263-279`, gyro and accelerometer on 10 steps) would fail. Design: `_stance_env` drops `OBSERVATION_NOISE_ENV_KEYS` from the stance kwargs, so the exact-layout tests read a clean step observation and keep the delay and per-body draws; nothing changes for the seven models, none of which sets the keys. The six-heading paired test (`:357`, through `heading_free_helpers.stance_env`) keeps the full feature block, since it is the check that D44's per-heading draws pair |
| `test_repository_root.py` (`:265-273`, `:280-297`, `:299-308`) | no entry: the derivation script and `view_model.py` import `paths.REPOSITORY_ROOT` and run as modules (§8.1) |
| `test_walk_stand_block.py` `WEIGHT_DIFFERENCES` (`:62`), `SETTING_DIFFERENCES` (`:77`) | `{}`, `{}` (§8.6) |
| `test_tracking_window.py` `WALK_WINDOWS` (`:54`, coverage `:298-301`) | `"triceratops_horridus/robot": (1.10, 55)` |
| `test_none_stage_rollout_golden.py:44`, `:81-100` | the golden must cover every stance (`:81-96`) and was captured on 857b6dd with no modified file, which `:98-100` pins; the robot does not exist there. Design (the implementer's choice): a second fixture for models added after D-D62, with its own spec list (the robot's stance), the coverage test reading the union of the two lists; recapturing the whole fixture would lose its point as a pin taken before D-D62, and an exemption would leave the robot's stance unpinned. **Provenance.** The capture tool records `git rev-parse --short=7 HEAD` and every tracked file that differs from HEAD, untracked files unlisted (`none_stage_rollout_golden.py:379-402`), and commit 5 is one commit (§3), so the fixture cannot name commit 5 itself. The rule: capture last, on a tree whose HEAD is e94efee9 (commit 5's parent, in the branch history) with every file of commit 5 in place and in the index (new files `git add -N` at least), so the fixture records `source_commit = "e94efee"` and, as `source_modified_files`, the files commit 5 adds or changes at that moment, the new fixture itself excepted (it is written after the capture). `--source-commit` is not used: it changes only the recorded commit, not the file list, and would name a commit whose tree is not the one captured. The second fixture's provenance test pins both as recorded, as `:98-100` pins 857b6dd and `[]`, and checks that the robot's asset, species base, robot env, `stages.toml` and `stance.toml` are in the list. A later review fix to any file the robot's stance capture reads is recaptured on the same rule, with HEAD then at commit 5 or its latest fix. A shared test change for main's maintainer (§11) |
| `test_phase_d_command_plumbing.py:529-549` | no entry (the walk `[env]` cites no stance line) |

**Every-model tests with no table**, which the robot's configs must pass: the live walk and its distribution
(`test_phase_d_command_plumbing.py:153-165`; `test_walk_command_distribution.py:113-139`, 400 seeds); the stand block
(`test_walk_stand_block.py:176-277`); the forward reference and the stand-block swap
(`test_walk_command_mechanism.py:325-355`, `:811-875`); the walk gate kind (`test_command_tracking_gate.py:257-276`);
the walk trains, reloads and is judged under PPO and SAC from a 64-step stance (`test_walk_species_training.py:120-234`,
SB3 job, full depth); the behaviour env's 16 tests (`test_behavior_env.py:141-`, among them `pelvis_height` and
`forward_vel_weight` in the signature); the statue-constant freshness (`test_statue_constant_freshness.py:47-80`);
`test_friction_cone.py:34-45`; `test_task_fingerprint_stage.py:56-123`; `test_sb3_notebook_pins.py:4029-4050`.
Optional every-model lists with no coverage check gain TRI to keep their claims true
(`test_warm_start_noise_scan.py:36`, `test_plant_contract_heading_free.py:62`, `:66`, `:80`, `test_cli.py:279`,
`:304`, `:329`, `test_model_id.py:15`, `test_eval_diagnostics.py:1612`, `:1628`,
`test_curriculum_rollout_ledger.py:269`, `test_gait_posture_telemetry.py:36`, `test_gait_body_telemetry.py:391`; and
`test_deliverable_metrics_for_reward_gated_stance_say_so`, `test_species_catalog.py:1794-1831`, by the kind commit 5
chose: on the reward gate TRI joins the Brachiosaurus and Dibothrosuchus loop (`:1805`); on the certificate it joins
`expected` (`:1816-1821`) as `TRI: "stance_quality_heading/v1"`, whose assertions fix the phrases the stance
definition keeps, §8.8).

**An enabled feature, for the first time.** On e94efee9 every `OPTED_IN_*` table is empty
(`test_observation_noise.py:81`), so every every-model test that builds an env from a committed stage config meets an
enabled feature block for the first time with this model, and no shared test exempts an opted-in model. The two
heading-free rows above are the ones found by reading; a text search finds 34 shared test files and helpers that
build an env from a stage config and step it (not all of them on every model), comparing exactly or to a tolerance
(among them `test_heading_free_rollouts.py`, `test_heading_free_checkpoints.py`,
`test_species_integration.py`, `test_gait_*`, `test_walk_command_mechanism.py`). So the whole shared suite runs on the
prototype with the robot's real stage configs before commit 5's test list is final (§8.12), and every further failure
from the feature block is listed here as a shared test change with its reason, for main's maintainer to review.

**CI**: the robot's test directory in the species suite (`python-ci.yml:345-352`) and `pyproject.toml` testpaths
(`:95-103`), both checked by `test_ci_tool_pins.py:291-302`; no SB3 import in the robot's files (`:364-377`). At full
SB3 depth the robot adds one behaviour fixture and a PPO and a SAC walk smoke at 1 ms physics.

### 8.10 Robot tests

New `environments/triceratops_horridus/tests/` (`test_robot_env.py`, `test_robot_features.py`), added to
`pyproject.toml` testpaths and the CI species suite; neither imports SB3 or calls `make_env`, so both run in the
species matrix and stay out of the test-sb3 step; the `make_env` and `train_behaviors` marking is tested once in the
shared SB3 files (§5).

- **Asset**: source commit, path and sha256 in the header; dt 0.001 s; (nq, nv, nu, nkey, nmocap) = (27, 26, 12, 1,
  **0**); the free joint's body is `torso`; key 0 `home`, reproduced by re-running `settle` within 1e-6; the 12 drives
  in order with ctrlrange exactly +-reach; the four spring folds; the provenance record matches the asset and script.
- **Standing**, unchanged: zero action, features off, 1000 steps, truncated, height drift under 2 mm, every foot above
  15 % of weight, foot sum = `model.body_subtreemass[torso] x g` after reset within 2 % (M4). Under every feature, in
  the stance and in the walk at a cruise and at a zero command, training and evaluation, 8 seeds: stands, no weight
  check (M4, R14). Measured on the prototype without D38's draws: 12 of 12 seeds stood the full 1000 steps.
- **Interface**: obs 59, act 12, dt 0.02 s; `_scale_action(a) == a x reach`; filter alpha `1 - exp(-2 pi x 4 x 0.02)`
  within 1e-12; schema, segment table and widths; touch order; morphology labels `fl, fr, rl, rr`; `_scale_action(0)`
  equals the reset key's ctrl (P10); the last 3 entries are 0 under `"none"` and the controller's normalised command on
  the walk.
- **Termination and task**: exactly the three root reasons and `body_contact`; no success key and no success probe
  effector (D50); `timestep=0.002` raises; every stage declares `timestep = 0.001` and `frame_skip = 20`, and every
  recipe resolves to them through `read_recipe` (M8); the walk's command scaling is cruise 0.15 m/s and `speed_scale`
  0.225 (D51); the walk's window is 55 control steps and its comment's L matches the derived asset's gait-module leg
  length within 1 mm; the stand block is drift alone; the zero-action statue, on one cruise and one stand episode of
  the certification plan, survives, and its cruise speed ratio is under the bar (the cheap form of D-D62 (18)).
- **Gates**: the stance gate equals the kind commit 5 chose with its bars (D52), and the walk gate
  `PROVISIONAL_WALK_THRESHOLDS` with the quadruped clause and the lead-swap bar, each comment saying provisional (D40,
  D52). **Removed** (D50): the first design's course tests ("stance never completes; locomotion does; a chassis lifted
  0.1 m never does") and the arrival-bonus twin test (D39).
- **Live-model readers** (§8.12, item 4): the stance report's and the command-tracking panel's body weight equal the
  nominal plant's on a randomised evaluation env.
- **Features**, unchanged but for the schema: stance, walk and every recipe resolved by `read_recipe` carry the one
  block (lists normalised to tuples before comparing, since `load_stage_config` makes tuples and `read_recipe` does
  not), and the noise tables are equal (P12); constructor defaults are off; a `config.build_env` env reports
  `actuation_delay_steps == 10` at every step while the same env marked with `mark_training_rollouts(env, True)` draws
  0-20 over 60 seeds; a `create_behavior_env` env draws 0-20 when marked and holds 10 when not (M10); for one seed the
  whole-robot factor and payload apply in training only and the per-body factors match (D37, D38); friction reaches the
  floor and the four soles only (R5); the noise table equals §8.4; projected-gravity tilt error 1.44 +- 0.1 deg (P2);
  std within `5 / sqrt(2 n)` (P6); both validators pass before the first reset and inside `nominal_physics()` after one
  (R4).

### 8.11 Docs and website

- `environments/triceratops_horridus/README.md`: Models, Run the model, Training (stance then the command-driven walk;
  the features; flat ground (D28); budgets and the measured stance session time, with the environment count D55 sets
  and why, and the samples per update at that count (§8.6); the stance kind commit 5 chose and the check's outcome
  (D52), and, on the reward gate, that the model may have no stance session until its D-D74 revision if main's
  maintainer confirms that reading; the walk bars provisional and not called attainable, with D40's 2000 and 550
  reported beside the walk statue's figures (D52); the owner's review of the stance bars before the first run and how
  that run is published (D54); the noise scan before the first walk, D-D72; Tanh, D43), Task and rewards (the walk's
  command at 0.15 m/s and why (D51), the stand block, no course and no arrival bonus (D50); the inputs the robot will
  not have, D46), Behaviour recipes (main's own command tasks, ported by D53, and inert: no committed walk can parent
  them, D-D62 (17)), Files, Further reading (the build by commit; the adult study in `examples/triceratops_horridus`
  stays outside the manifest). `notebooks/sb3_training.ipynb`: the species list gains `triceratops_horridus`; two
  markdown lines name the robot.
- Website (under `website/`): `docs/models/triceratops_horridus/_category_.json` (position 7) and `robot.mdx`; a robot
  entry in `src/components/AnimalPage/interpretation.ts` (`tests/animal_interpretation.test.mjs`) and a robot page with
  text of its own that claims no walk (D-D75 (2)); **new:** the still image
  `static/img/animals/triceratops_horridus-robot.webp` with its line in `static/img/animals/SOURCES.md`, and the
  turntable poster and light and dark videos (`python website/scripts/render_turntables.py triceratops_horridus-robot`;
  `tests/animal_images.test.mjs:40-80`); the model lists the contract tests compare with the manifest, in manifest
  order (CONTRIBUTING.md, "Adding an embodiment", step 6): in `docs/intro.md` the link
  `[Triceratops horridus (Robot)](/docs/models/triceratops_horridus/robot)` last in the model-page list (`:34-40`;
  `test_model_identity_contract.py:240-244`), and the Key Features line (`:19`) as "Eight training models across seven
  species", naming the Triceratops robot beside the Compsognathus robot (line 23's eight animals stay); in
  `docs/api/overview.md` the bullet `` `MesozoicLabs/TriceratopsHorridus-Robot-v0` - Triceratops horridus (Robot) ``
  last in the Gymnasium ID list (`:25-31`; `test_model_identity_contract.py:255-262`) and "registers all eight"
  (`:33`); in `training/recipes.md` the Triceratops robot beside the four models that commit two nodes (`:139-142`).
  `getting-started/installation.md` needs no edit on e94efee9: its one model-wide line (`:58`, the notebook's
  `BEHAVIOR = "auto"`, "which is walk on every model") stays true, and its examples name the Velociraptor.
- **The animal page: a separate generated model page (D56).** The first design folded the model into the hand-written
  `triceratops` card and skipped its auto route; that now breaks `tests/animal_images.test.mjs:84-92` ("every animal
  of the collection has its image", added with cb1a301e after 2c1039f4), which asks the set of `ANIMALS` slugs that
  have models to equal the catalog's species ids: `src/data/animals.ts:26-40` builds each catalog species' entry with
  `slug: model.species`, here `triceratops_horridus`, while the hand-written card (`:53-60`) has slug `triceratops` and
  `models: []`, and `plugins/animal-pages.cjs:11-23` routes `/animals/<species>` for every catalog species. So, as D56
  decides: the generated `triceratops_horridus` entry and its route are the model's page, with its line in `summaries`
  (`animals.ts:12-19`) naming the robot model; the hand-written Triceratops card keeps `models: []` and its page links
  the model page (`src/pages/animals/triceratops.tsx`, and `src/data/triceratops.ts`); no website test changes. The
  animals list shows two cards that both read "Triceratops / Triceratops horridus" (the generated entry's name comes
  from the species name, `animals.ts:31-35`), told apart by summary and image. The hand-written card's summary ("An
  adult anatomical reference and two robot designs ...", `animals.ts:57`) still reads true of its page's studies (the
  60 cm design, now adopted, and the 75 cm concept); the generated card's summary names the robot model, so the two
  are told apart without a test change. `interpretation.ts` needs a
  species-level `triceratops_horridus` entry (`classification`, `title`, `anatomy`, `alt`, `detailLabel`, `detail`,
  `studies`, which may be empty) for its `robot` text to hang off: `tests/animal_interpretation.test.mjs` "every robot
  model of the catalog has its own page text" (`:97-106`) reads `INTERPRETATIONS[robot.species]?.robot`, and "every
  label names a stage of a model of its species" (`:108-121`) checks every stage key of both. `npm test` in `website/`
  where `node_modules` exists.
- `docs/SPECIES_NAMING.md` row; root `README.md` prose under the generated table; the docstring of
  `environments/shared/scripts/warm_start_noise_scan.py` (`:13`, `:133`: "seven models" becomes eight);
  `docs/NEXT_STEPS.md` §3 gains the robot's item (its first stance session under its gate, after the owner's review of
  the bars, D54, at the environment count D55 sets; then the noise scan and the walk); `docs/ROADMAP.md` Phase 6 (M9);
  `CHANGELOG.md`. The landing itself is recorded only in the consolidation plan's status table and the CHANGELOG
  (`tests/test_landing_records.py`). No new Drive reader or writer, so `docs/DRIVE_LAYOUT.md` is unchanged.
- `docs/hardware/README.md` (D41): the page says only an adopted design becomes a species' robot model (`:7-10`, the
  maintainer's decision of 2026-11-26; D-D75 (3)). D41 adopts the 60 cm design for building, and D32 and D36 make it
  that model, so the page says so, citing them: the header (`:5-10`) names `triceratops_horridus/robot` beside the
  Compsognathus revision B; "Where things stand" (`:12-18`) says the design was adopted on 27 Nov 2026 and that
  adopting it orders nothing (the 21 Nov 2026 not-ready-to-order finding stands, the one-foreleg bench test comes
  first, compute and the rest of the chassis are open); the concepts table's `Triceratops, 60 cm` row (`:31`) becomes
  "adopted design; robot model in simulation", 27 Nov 2026; and the records table's `60 cm Triceratops build` row
  (`:71`) adds the model, derived from build commit f6deec9f.

### 8.12 Measured in commit 5, before it is final

Each item runs on the derived asset with the robot's real stage configs, its output kept and quoted in the PR
description, as §9's gates are. No pass is claimed before a tool's output shows it.

1. **The four-foot check that picks the stance kind (D52).** The `stance_quality_heading/v1` report, through
   `stance_quality_baseline` and the stance gate report path, runs on the four capsule feet with
   `FOOT_SUPPORT["sole"] = None` (no quadruped has declared a floor-truth kind), on the zero-action statue with the
   feature block on, at all six headings, on two panels: the 40 seeds from 3042, on which the five measured bars are
   set (item 2), and an out-of-sample block of 80 seeds from 3082, as the biped certificates use, on which they are
   judged. The criterion is Design, the implementer's proposal, set out for the owner with the bars (item 2). Five
   of the bars come from the statue panel itself, so the check would pass them by construction if it judged them
   there; it does not. **Pass**, all of:
   - (a) the report runs to its verdict on four feet, with a finite value of every required v2 statistic on every
     episode at every heading;
   - (b) each heading of a seed draws the same plant, delay and noise (D37, D44), and the `*_bw` statistics divide by
     the nominal body weight (item 4);
   - (c) on both panels and at every heading, the statue clears the five copied bars as the biped certificates declare
     them (`min_eval_episodes` 40, `min_clean_stance_lcb` 0.80, `min_all_feet_support` 0.98, `max_touchdown_rate`
     0.25, `max_actuator_saturation_fraction` 0.10) and D52's two rails (mean reward at least 900, full-horizon
     fraction at least 0.95), each as declared, with no margin;
   - (d) on the out-of-sample block, at every heading, the statue clears the five measured bars as set on the 40 seeds
     from 3042 with these margins, after the T. rex's certificate
     (`configs/tyrannosaurus_rex/anatomical/stance.toml:757-802`): `settle_steps` 4 x the latest step at which the
     statue's total floor force enters and stays within 5 % of its weight (the T. rex's 200 holds its statue's settle
     4x over); `max_settle_airborne_substeps` the statue's worst count, 0 where it has none (structural, as on the
     T. rex and the Velociraptor); `max_settle_peak_floor_force_bw` the statue's worst settle peak plus 15 % (the
     T. rex's 1.5 is 18 % above its statue's 1.272); `max_window_displacement_m` 4 x the statue's worst window
     travel (the T. rex's 0.10 m against 0.023 m); `min_foot_load_share` 0.8 x the statue's lightest foot share
     (§8.6). `settle_steps` is a split of the episode, not a bar the statue clears: it sets which steps the others
     read.

   **Fail**, and the stance takes `reward_and_length/v1` as D40 until revised, on any of: the report raising, or a
   statistic undefined, on four feet; headings that do not pair; the statue failing a copied bar or a rail at any
   heading of either panel (for example a capsule sole read as lifted, so that all-feet support falls under 0.98,
   touchdown chatter above 0.25 per foot per second, or fewer than 37 of 40 seeds clean at all six headings); or the
   out-of-sample statue failing a measured bar set with its margin. Commit 5 says which, with the output (§8.6).
2. **The stance bars on the statue panel, for the owner's review (D54).** Every bar of the chosen kind, each with the
   statue's measured value on it (the worst heading, in sample and out of sample) and marked by its source, from
   `stance_quality_baseline` and `zero_action_baseline --model-id triceratops_horridus robot --episodes 40 --seed
   3042` (and the 80 seeds from 3082) at the stance's reset noise, written into the stance TOML with their source and
   set out for the owner, who reviews them before the first stance run:
   - under the certificate: the ten required v2 keys, the five **copied** (`min_eval_episodes`,
     `min_clean_stance_lcb`, `min_all_feet_support`, `max_touchdown_rate`, `max_actuator_saturation_fraction`) and
     the five **measured**, with their margins (`settle_steps`, `max_settle_airborne_substeps`,
     `max_settle_peak_floor_force_bw`, `max_window_displacement_m`, `min_foot_load_share`; item 1); D52's two
     **rails** (`min_avg_reward = 900.0`, `min_full_horizon_fraction = 0.95`); and the collapse floor's keys
     (`collapse_peak_floor_reference`, the statue's absolute stance reward; `collapse_peak_floor_fraction = 0.45`;
     `collapse_peak_warmup_timesteps = 1000000`; `statue_constants_physics_revision = 1`);
   - under the fallback: `min_avg_reward = 900.0`, `min_avg_episode_length = 950` and `required_consecutive = 3`,
     against the statue's mean reward and length; `collapse_peak_floor` (0.75 x the statue's return, rounded to the
     nearest 10) and `collapse_peak_warmup_timesteps = 1000000`;
   - open, for the same review: under the certificate, which of the optional keys the three biped certificates
     declare a four-foot stance takes, each with the statue's value on it (or the note that it is not finite on four
     feet) and with whatever scripted hack checks have run on four feet, and whether to add the statue-relative
     reward rail `min_avg_reward_statue_ratio = 0.60`, which D52 does not name (§8.6); under the fallback, whether the
     first stance run waits for the D-D74 revision (§8.6). The candidates are `max_foot_contact_fraction` (the
     stacked-foot bar), `min_support_geom_duty` and `min_support_geom_coverage` (the padless foot's bars, one number
     on this foot), `min_foot_load_share_windowed`, `min_foot_load_share_statue_ratio`, `max_phantom_support_fraction`,
     `max_nonfoot_load_fraction`, `max_settle_stance_width_change_m`, `max_settle_touchdowns`,
     `max_episode_yaw_change_deg`, `max_hop_or_fall_episodes`, `max_window_airborne_substeps` and
     `max_window_peak_floor_force_bw`, the four that count feet or depend on the plant re-derived on the statue. The
     four sole bars (`max_sole_tilt_deg`, `max_sole_corner_lift_m`, `min_sole_contacts`, `max_sole_cop_fore_aft`) are
     excluded, since with no sole they would fail every episode, and the hop watch is not taken, the implementer's
     choice, which the owner may reverse (§8.6, "The certificate's optional keys").

   With them, the walk statue panel (`--stage 2`): its survival and speed ratio (§8.6) and its mean reward and length,
   reported beside D40's 2000 and 550 (D52).
3. **The training speed (D55).** The stance trained through main's trainer at `N_ENVS = 4`, and at 8, for long enough
   to include in-training evaluations at the stance's cadence, on a machine whose CPU count is recorded; env steps/s
   scaled to 11M, plus the certificate and the bundle, against one Colab session of about 24 h and main's 19h31m for the
   Compsognathus robot. If 4 environments would not fit, the README asks for 8 on a runtime with 8 or more CPUs (§8.6).
   The figure is given with its machine: a sandbox's throughput is not Colab's.
4. **The live-model readers handed over by §6** (commit 3's hand-off: §6, "Pushes, settle, readers" and the last item
   of "Built in commit 3"). On the merged tree the readers are `body_weight_n` from `body_subtreemass`
   (`floor_contact/morphology.py:498`), `actuator_force_limits` (`gait/balance_telemetry.py:128-131`),
   `gait/body_telemetry.py:466` and `reporting/stance_report.py:993` (`np.sum(model.body_mass)`), and the layouts
   built from them are `reporting/stance_report.py:294`, `evaluation.py:486`, `eval_diagnostics.py:742`, `:973`,
   `rollout_ledger._layout_of` (`rollout_ledger.py:311-323`) and main's new command-tracking panel
   (`command_tracking_panel.py:313`, built after plant validation and before the first reset). On the robot env each is
   built inside `nominal_physics()` or before the first reset, or documented as per-episode, and a robot test pins the
   stance report's and the panel's body weight to the nominal plant's (§8.10). It bears on D52: the floor-truth `*_bw`
   statistics divide by that weight (`gait/stance_metrics.py:355`, `gait/recorder.py:1091`), while the per-body factors
   move each evaluation episode's true weight by about x0.93-1.07 (§6, measured on the prototype).
5. **Values the tests need**: the leg length L of the window's comment (§8.6, within 1 mm); the `WILD_ACTION_SCALES`
   entry or another route to a terminal state (§8.9); whether the harness's fixed kick topples the robot
   (`_ROLL_STANDS`) and `_POSE_REASONS` from the golden; `PAIRED_ENDS` if the statue falls within 250 steps; the
   standing tests under every feature (§8.10); the whole shared suite with the robot's real configs, each further
   failure from the feature block listed in §8.9; the generated counts (§8.8).

## 9. Verification gates

Every code commit (2-5) passes all of these locally; the outputs are kept and their summary lines quoted in the PR
description. Install with the canonical MuJoCo and tensorboard (the `train` extra): `python -m pip install -e
".[train,test]" "mujoco==3.10.0" "ruff==0.16.9" "mypy==2.3.1" -c constraints.txt`. Before the gates, run `ruff check
--fix` and `ruff format` on the commit's Python files (R9, P11, M2). The base below is 2c1039f4, the base the
commits were built on; from the merge with a232da31 on, read that sha for it (§3, "Baselines at a232da31"). Commit 5
is built on the merged tree and compares with a232da31 throughout: in its digest snapshot, its two plant manifests and
its `--exact` capture only the robot's lines may be added, and no other line may change (gates 2 and 3).

1. Lint and types (`python-ci.yml:101, 104, 107, 118, 430`):

   ```bash
   ruff check environments/ notebooks/colab_install.py
   ruff format --check environments/ notebooks/colab_install.py
   mypy environments/ notebooks/colab_install.py --ignore-missing-imports
   mypy environments/ --ignore-missing-imports          # with SB3 and torch installed
   python environments/shared/tests/notebook_cells.py
   ```

2. Generated files (`python-ci.yml:141-142, 151, 162`; `docs/PLANT_CONTRACT.md:98`):

   ```bash
   python -m environments.shared.plant_contract --check
   git show 2c1039f4:configs/plant_manifest.generated.json > /tmp/base-plant-manifest.json
   python -m environments.shared.plant_contract --check --baseline /tmp/base-plant-manifest.json
   python -m environments.shared.species_catalog --check
   python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --check
   python -m environments.shared.symmetry --check
   git diff --exit-code 2c1039f4 -- configs/digest_snapshot.generated.txt configs/plant_manifest.generated.json \
       environments/shared/data/plant_manifest.generated.json   # commits 2-4
   ```

   Commit 5, against a232da31: every snapshot line it adds names `triceratops_horridus`, and the plant manifests gain
   the robot's entry and nothing else (a JSON entry spans lines that do not all name the model, so it is compared as
   data):

   ```bash
   diff <(git show a232da31:configs/digest_snapshot.generated.txt) \
        <(grep -v triceratops_horridus configs/digest_snapshot.generated.txt)          # empty
   git diff a232da31 -- configs/plant_manifest.generated.json \
       environments/shared/data/plant_manifest.generated.json | grep -E '^-[^-]'      # prints nothing
   for f in configs/plant_manifest.generated.json environments/shared/data/plant_manifest.generated.json; do
     python -c 'import json, subprocess, sys; f = sys.argv[1]; head = json.load(open(f)); base = json.loads(subprocess.check_output(["git", "show", "a232da31:" + f])); head["plants"].pop("triceratops_horridus"); assert head == base, f' "$f"
   done
   ```

3. Bit-exact capture against main (CONTRIBUTING.md step 4; `-m` from each checkout's root, as the harness enforces):

   ```bash
   git worktree add --detach /tmp/base 2c1039f4
   (cd /tmp/base && python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --exact) > /tmp/exact_base.txt
   python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --exact > /tmp/exact_head.txt
   diff /tmp/exact_base.txt /tmp/exact_head.txt                                   # commits 2-4: empty
   diff /tmp/exact_base.txt <(grep -v triceratops_horridus /tmp/exact_head.txt)   # commit 5 (base a232da31): empty
   ```

4. The full suite over `pyproject.toml` testpaths (commit 5 adds the robot's directory), compared by node id with
   main's run the same way in the base worktree:

   ```bash
   PYTHONPATH=$PWD python -m pytest -rfE -p no:cacheprovider 2>&1 | tee /tmp/pytest_head.log
   (cd /tmp/base && PYTHONPATH=/tmp/base python -m pytest -rfE -p no:cacheprovider) 2>&1 | tee /tmp/pytest_base.log
   comm -13 <(grep -oE '^(FAILED|ERROR) [^ ]+' /tmp/pytest_base.log | sort -u) \
            <(grep -oE '^(FAILED|ERROR) [^ ]+' /tmp/pytest_head.log | sort -u)    # must print nothing
   ```

   A head failure that main also has is listed by node id in the PR description. Remove the base worktree after.

   **CI placement.** The new test files import no SB3 and run in the `test` matrix: `test_episode_start.py`,
   `test_actuation_delay.py`, `test_episode_physics.py` and `test_observation_noise.py` in the shared suite, the
   robot's two in the species suite. The SB3 checks (`create_vec_env(training=True)`, `train_behaviors.main`'s marking)
   go in `test_train_base.py` and `test_behavior_checkpoint.py`, which the test-sb3 step already runs, so that step is
   unchanged and `test_ci_tool_pins.py:364-376` holds (§5).

   **Measured on the prototypes**, shared suite only, without tensorboard, failures by node id:
   - Randomisation, full suite (7,305 tests): 12 failures, all also on base: the 9 tensorboard failures
     (`test_compsognathus_longipes_training.py` x4, `test_curriculum_cli_end_to_end.py` x4, `test_train_wiring.py` x1)
     and 3 subprocess imports of `environments` without an install
     (`test_plant_contract_digest_snapshot.py::test_refusals_come_before_the_run` x3).
   - Delay, full suite with `PYTHONPATH` set (the skeptic's run): the 9 tensorboard failures, plus
     `test_behavior_checkpoint.py::test_real_ppo_cli_resume_preserves_recipe_and_releases_stage_warmup`, which passes
     on base. `mark_training_rollouts` fixes it by design (P1).
   - Merged model prototype, full suite: those 10, plus
     `test_species_catalog.py::test_every_species_model_page_is_tracked_by_git`, an artefact of the new files being
     untracked (it passes after `git add -N`).
   - Noise: focused tests only (615 passed, 1 failed; the fix was the noise role helper this design drops).

   No full-suite pass existed for the reconciled design when this section was written; the merged tree's is in §3
   ("Baselines at a232da31"). Commit 5's is run on its own tree against a232da31's, after the whole suite has first
   run on the prototype with the robot's real configs (§8.9, §8.12); every failure it adds in an existing test is
   resolved by a shared test change listed in §8.9 with its reason. With `node_modules` present, `npm test` in
   `website/` must pass with no website test changed (D56).

## 10. Skeptic findings and their resolution

| ID | Finding | Resolution | Section |
|---|---|---|---|
| Delay P1 | `train_behaviors` called `set_training_rollouts` on a plain test env | `mark_training_rollouts` on `env.unwrapped`, no-op without the method | §4.3 |
| Delay P2 | Only 1243 of 7305 shared tests run; no test gate | Full suite over testpaths, compared by node id | §9 |
| Delay P3 | `actuation_delay_steps` stale between reset and the first step | k chosen at reset; property valid from reset | §5 |
| Delay P4 | `k x timestep` is not the decimal seconds | Info carries the steps and the seconds rounded to 12 decimals | §5 |
| Delay P5 | Mirroring copied the role | FIFO and k only | §5 |
| Delay P6 | `behavior_checkpoint` lacks the `actuation_delay` section | Added, with `episode_physics` and `observation_noise` | §5 |
| Delay P7 | "Refused naming env and actuation_delay" overstated | Reworded; switch-off case tested | §5 |
| Delay P8 | Harness docstring says stages are built as `make_env` builds them | Names evaluation envs | §5 |
| Delay P9 | Citations off; 0-20 ms is not the build default | Corrected; D33 and the steer1 presets cited | §5 |
| Delay P10 | Zero-action invariance needs prime == `_scale_action(0)` | Precondition stated; robot test | §5, §8.10 |
| Delay P11 | `warm_start_noise_scan` sees 10 ms, not the training draw | Listed as an evaluation diagnostic; docstring note | §4.3 |
| Rand. R1 | Delay and randomisation differ on RNG and on evaluation-only draws | One stream policy; one role flag read by the whole-robot draws | §4.2, §4.3, §6 |
| Rand. R2 | "No species lists a key" fails once the robot does | Opted-in table with carve-outs | §6, §8.3 |
| Rand. R3 | `episode_physics` missing from `_differing_sections` | Added, and to `behavior_checkpoint` | §6 |
| Rand. R4 | The `nominal_physics()` wrap in validation is unused and weakens it | `validation.py` unchanged and strict | §6 |
| Rand. R5 | Friction on every animal geom changes self-contacts | `_randomized_friction_geoms()`; the robot's 4 soles | §6 |
| Rand. R6 | `np_random_seed` needs gymnasium >= 1.0 | Seed argument first, then `getattr` with fallback | §4.2 |
| Rand. R7 | The state fallback assumes PCG64 | PCG64 read; others raise `ValueError` | §4.2 |
| Rand. R8 | A massless payload body divides by zero | Refused at construction | §6 |
| Rand. R9 | `ruff format` fails on the module | `ruff format` before the gates | §9 |
| Rand. R10 | `_applied` captured and never read | Removed; `nominal_physics()` captures on entry | §6 |
| Rand. R11 | Capability still "planned" on the site | `experimental`, catalog regenerated | §6 |
| Rand. R12 | Randomising stance departs from ROADMAP's "Stage 2+" | Both stages, as the build; noted; owner point | §6, §11 |
| Rand. R13 | Restore test against a fresh compile fails for post-edited models | Compare with an off env of the same kwargs | §6 |
| Rand. R14 | Masses included the mocap target; standing measured on a thin env | Animal subtree masses; standing re-tested on the robot env | §6, §8.10 |
| Rand. R15 | "Same plant in episode k" overstated | Fixed-seed panels only; EvalCallback rounds reset unseeded and see new plants | §6 |
| Rand. R16 | Later writes to managed fields silently undone | Rule in PLANT_CONTRACT.md; record taken at a model's first episode | §6 |
| Noise P1 | Robot table used names its schema lacks | Heading-free names; `observation_segments()` dispatch; on main the robot declares `quadrupedal-heading-free/v1`, so no new schema (§3, §8.4) | §7, §8.4 |
| Noise P2 | Gravity 0.01 halves the build's tilt noise | 0.02; tilt test 1.44 +- 0.1 deg | §7, §8.10 |
| Noise P3 | A perfect accelerometer leaks tilt | 0.2 m/s^2 | §7 |
| Noise P4 | Three features, three rules for who sees them | Option b: training rollouts or the eval flag; bank forces off | §4.3, §7 |
| Noise P5 | Off-by-default test fails for the robot | Off exactly when the stage omits the key | §7 |
| Noise P6 | 3 % tolerance too tight for short episodes | At least 2000 noisy samples, `5 / sqrt(2 n)` | §7 |
| Noise P7 | `_step_count == 1` detection is a heuristic | Stream built in the hook | §4.2, §7 |
| Noise P8 | The wrapper option's cost overstated | Restated as a `Monitor` subclass; a follow-up | §7, §11 |
| Noise P9 | Heading drift framed for a quaternion | Moot: no heading channel | §7 |
| Noise P10 | `observation_noise_manifest()` has no caller | Dropped | §7 |
| Noise P11 | `ruff format` fails on the module | Formatted before the gates | §9 |
| Noise P12 | `extends` replaces a nested table whole | Documented; equal-tables robot test | §7, §8.10 |
| Noise P13 | Provenance by branch path; stale CONTRIBUTING line | Cite by build commit; CONTRIBUTING names the modules | §7 |
| Model M1 | The model cannot land alone | Model last, regenerated on the merged tree | §3 |
| Model M2 | ruff errors and formatting; mypy not run | `ruff check --fix`, `ruff format` before regenerating; mypy in the gates | §8.8, §9 |
| Model M3 | Exempting the robot from `test_species_integration` drops coverage | Added to its tables; not exempt | §8.9 |
| Model M4 | Standing test depends on seed 0's mass draw | Live subtree weight; features-on test without a weight check | §8.10 |
| Model M5 | The reward discourages arriving | D39 arrival bonus, superseded by D50: on main there is no target and nothing ends a walk episode but a fall, so no arrival is left unpaid | §8.5 |
| Model M6 | Delay and mass values attributed to the build's defaults | D33 and D38 cited; whole-robot draws on, training only | §5, §6, §8.6 |
| Model M7 | Dead `symmetry.py` entry; contradictory docstrings | Entry dropped (main's layouts are per schema now); docstrings fixed | §8.1 |
| Model M8 | A TOML missing `timestep` is not caught | Robot test on both stages and on every recipe as `read_recipe` resolves it | §8.2, §8.10 |
| Model M9 | ROADMAP Phase 6 and the capability status go stale | Both updated | §6, §8.11 |
| Model M10 | Behaviour envs held 10 ms without the trainer's mark | `train_behaviors` marks via the helper; robot test | §4.3, §8.10 |

## 11. Not in the follow-up PR (commits 2-5), and points for the owner

**Follow-ups:**

- A training-mode capture in the digest snapshot; a per-episode delay column in `train_episodes.csv`
  (`rollout_ledger.py:390`).
- A previous-action (filter-state) observation segment, as the build has: a policy-interface revision needing fresh
  checkpoints. Until then in-flight targets are hidden state.
- The eleven control-step readers of §8.2 reading the model's own timestep; noise on the reset observation; per-joint
  noise.
- Existing species opting in, with the carve-out at off value `None` (`save_stage_config` converts tuples first,
  `config.py:1137-1152`); measured on the prototype for the Compsognathus robot: every committed digest kept while off.
- Pair-tied draws for a mirror-augmented model; 16 actuators if the head and tail dynamics matter; a smaller reset
  qvel noise.
- The build's steering (its turning schedule, turning on the spot, its heading estimate) and its heat costs, which
  D53 keeps in the build stack; main's whole-episode walking command and main's own command recipes are ported (§8.6,
  §8.7).
- A supported way to name the PPO activation in a stage TOML, so the robot can train with the build's ELU (§8.6); until
  then it trains with SB3's default Tanh, a declared deviation from the build (D43, which covers the activation only).
  The samples per update at main's 4 environments, half the build's, are a further deviation that this design declares
  and does not pin: the implementer's choice, not an owner decision (§8.6).
- An own-sensors quadrupedal schema, if D46's run on the build stack shows that a policy on the robot's own sensors
  walks: a shared policy-interface change (the heading-free helpers, the plant contract, the symmetry layouts, the
  noise kinds and their tests), for main's maintainer (§8.4). Until then the 15 inputs the robot will not have stay in
  the model, a sim-only deviation the implementer accepts, not an owner decision (§8.4).
- A reference walk on this plant (`scripts/reference_walk.py`, from the build's model-based trot, which was written
  for 100 Hz control with the filter off and needs re-tuning for main's 50 Hz env and 4 Hz filter): it would set the
  tracking window from the robot's own stride (§8.6) and allow a matched-basis walk collapse backstop (D-D62 (15),
  D-D65 (1)).
- A `StageWarmupCallback` that skips the SAC reseed at a zero warm-up entropy, a shared change for main's maintainer,
  if the build's zero PPO entropy at the walk's handoff matters (§8.6).

The first design's follow-up on the arrival bonus under `VecNormalize`'s reward clip of 50 (`constants.py:22`) is
moot: there is no arrival bonus (D50).

**Points for the owner.** Points 1, 4 and 5 were decided on 27 Nov 2026 (D41, D44); point 2 on 28 Nov 2026 (D52 and
D54, with D50 and D51; D40 amended); point 3 is moot (D50). Of point 6, the recipes were decided on 27 Nov 2026 (D42;
their D32 scope by D53 on 28 Nov 2026) and the animal page on 28 Nov 2026 (D56); its other two parts, the head-and-tail
fold and the statue warm start, were taken as the design recommended them and never put to the owner as rows. D55
(the environment count) answers no point of the first design. What is still open follows them.

1. `docs/hardware/README.md` says only an adopted design becomes a species' robot model (2026-11-26). Commit 5 makes
   the 60 cm design `triceratops_horridus/robot` in simulation (D32, D36) and updates the page. **Decided (D41):** the
   design is adopted for building too, ordering nothing (§2, §8.11).
2. The gates and course values are provisional (D40), revised after the first main-pipeline run. **Decided (D52,
   D54, with D50 and D51; D40 amended):** no course (D50); the stance on main's six-heading certificate with D40's 900
   reward and 95 % full-horizon rails if commit 5's four-foot check passes, else D40's reward gate until revised; the
   walk on main's shared walk gate with the quadruped clause, D40's 2000 and 550 report-only and its 0.15 m/s the
   fastest command (D51); the walk bars then move only with main's, on this record's reading of D52 (§8.6). The owner
   reviews every stance bar, with the statue's measured value on it, before the first stance run, which is published
   as main's chain publishes it; a revision after it costs nothing if that stance failed and retrains a certified one
   (§8.6).
3. The first design's arrival bonus counted the heading term (1.8 per step), and it left open whether a walker passing
   more than 0.3 m to the side of the target out-earns one that arrives. **Moot (D50):** main has no target and the
   robot no course, so there is no side to pass and no arrival to pay; the walk is judged on command tracking over
   the whole horizon (§8.5).
4. The robot randomises in stance too, as the build does (R12); friction is absolute, not ROADMAP's scale.
   **Decided (D44):** both as planned.
5. The reset observation is noise-free, unlike the build (one observation per episode). **Decided (D44):** as
   planned.
6. Taken as the model design recommended: head and tail folded into springs (nu 12); all 11 recipes on a flat-ground
   robot (D28); a stance the statue passes, kept as a warm start; one animal page on the site. **Decided (D42):** all
   11 ship. Since main a232da31 its other parts read so: the first design's new schema name is moot, as the model uses
   main's `quadrupedal-heading-free/v1` (59 values, §8.4); the D32 scope question of the recipes is answered by D53,
   which ports main's whole-episode walking command and main's own command recipes that run only when started; and
   "one animal page" is replaced by D56, a separate generated model page beside the hand-written Triceratops card,
   since a website test main added refuses the fold (§8.11). **Not decided by any row:** the head-and-tail fold
   (nu 12, §8.1) and a stance the statue passes, kept as the walk's warm start, stand as the design recommended them.
   That the statue passes the stance under either kind of D52 is an expectation that §8.12 (items 1 and 2) measures,
   not a pass (§8.6).

**Still open.** No owner decision is pending now. The owner's next decisions come at two reviews: the stance bars
before the first stance run (D54), which also takes the owner's questions listed below, and D40's revision after it.

- Measured in commit 5, before its design is final (§8.12): the four-foot check, and so the stance kind (item 1,
  D52); the stance bars on the statue panel (item 2, D54); the training speed, and so the environment count the README
  asks for (item 3, D55); the live-model readers, and the stance report's and the command-tracking panel's body weight
  against the nominal plant's (item 4); the values the tests need, among them the window's leg length, the
  `WILD_ACTION_SCALES` entry, whether the harness's fixed kick topples the robot, `PAIRED_ENDS`, the standing tests
  under every feature and the whole shared suite on the robot's real configs (item 5).
- For the owner, at the D54 review before the first stance run: every stance bar with the statue's value on it
  (§8.12, item 2); under the certificate, which of the biped certificates' optional keys a four-foot stance takes, of
  the candidates `max_foot_contact_fraction` (against stacked feet), `min_support_geom_duty` and
  `min_support_geom_coverage` (the padless foot's bars), `min_foot_load_share_windowed`,
  `min_foot_load_share_statue_ratio`, `max_phantom_support_fraction`, `max_nonfoot_load_fraction`,
  `max_settle_stance_width_change_m`, `max_settle_touchdowns`, `max_episode_yaw_change_deg`,
  `max_hop_or_fall_episodes` and the window hop pair (`max_window_airborne_substeps`,
  `max_window_peak_floor_force_bw`), with the four sole bars excluded (no sole to read) and the hop watch not taken
  unless the owner asks for it (§8.6, "The certificate's optional keys"), and whether to add the statue-relative
  reward rail D52 does not name (§8.6); under the fallback, whether the first stance run trains on the reward rail
  after the review or waits for the D-D74 revision (D52 says only "until revised"; this record reads the revision as
  D-D74's, §8.6); and, if D42's "no shared test changes" is read strictly, the one shared test change the recipes
  need, the terrain-scale test's control step (§8.7).
- For main's maintainer, asked with commit 5: whether a new model may land on `reward_and_length/v1` and still get a
  stance session (only if D52's fallback is taken; `docs/NEXT_STEPS.md` §3, D-D74); which branch of D-D64's window
  rule a quadruped with no reference walk takes (the bipeds' Froude branch is used, 1.10 s, §8.6); and the shared test
  changes of §8.9, each with its reason:
  - the heading-free `_stance_env` without the noise keys: the robot's is the first committed stance with noise on,
    and two layout tests compare its step observation with the raw state bit for bit;
  - the `WILD_ACTION_SCALES` entry, or another route to a terminal state: the test needs a terminating rollout, and
    the robot stands at small scales;
  - the second none-stage golden fixture and its provenance rule (§8.9): the first was captured on 857b6dd, before the
    robot existed, and commit 5 cannot name itself, so the fixture names its parent e94efee9 and the files commit 5
    adds or changes;
  - the `frame_skip` rule of `test_behavior_recipes.py`: its hard-coded control step gives a 1 ms model 6.25 s, not
    25 s;
  - `_ROLL_STANDS` and the roll assertion inverted for its members in `test_plant_contract_digest_snapshot.py`, if the
    harness's fixed kick does not topple the robot: new test logic, since today every model's roll must end the
    episode (`:322`);
  - a `NOT_MIRRORED` table in `test_symmetry.py` in place of the Compsognathus robot's special case (`:112-119`): the
    robot is the second asset that is not mirror-symmetric as authored;
  - a TRI branch for the leg tops in `test_gait_morphology.py` (`:231-234`): its leg tops are `<label>_roll_yoke`,
    not `<label>_thigh`;
  - and any further failure the whole suite shows.
- After the first walk: its learned stride against the 1.10 s window, which the bipeds' Froude rule set with no
  reference walk on this plant (a stride longer than the window raises the boxcar's residual sway steeply, D-D64 (1));
  a change to the window trains the walk again (§8.6).
- After the first run, with the owner: D40's revision of the stance bars, as D54 sets out; on this record's reading of
  D52 the walk bars move only with main's.
