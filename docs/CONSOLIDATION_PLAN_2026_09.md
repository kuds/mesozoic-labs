# Consolidation plan: behavior recipes and the direction/terrain pilots (2026-09-17)

**Date**: 2026-09-17 (the review session); migrated into the repository by the
2026-09-19 documentation pass. **Assessed at**: `main` @ 723f58f (the merge
of #541). **Scope**: the recipes program (PRs #528–#539) and the direction/terrain
pilots with their certified library (PRs #540–#541). **Companions**:
[BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) (the design of record this
plan folds the pilots back into; its §6.2 carries the D-D and G series recorded
in §6 and §7 below), [NEXT_STEPS.md](NEXT_STEPS.md) (the training sessions that
run in parallel with this sequence, and the Drive state they rest on),
[investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md)
(the 2026-09-17 Drive survey, the note of record for every run fact cited
here),
[TRAIN_DIRECTION_AND_TERRAIN.md](TRAIN_DIRECTION_AND_TERRAIN.md) and
[CERTIFIED_MODELS.md](CERTIFIED_MODELS.md) (the pilots' own guides; the sequence
shortens the first and deletes the second). Every `file:line` below was re-read
at 723f58f while the plan was written and drifts after it (two merges
already: #542 and #543); read them as anchors, not contracts. Where a first estimate was
corrected on re-reading during the review, the corrected figure is used.

## Status (2026-09-20)

| Item | State |
|---|---|
| Baseline | `main` @ 723f58f. Line numbers in this document were re-read at that commit and drift after it. |
| PR-1 (stop the live disconnect) | **Landed** as #542 on 2026-09-16: notebook default `PUBLISH_CERTIFIED = False`, pin flipped. |
| Automatic trunk selection (D-A25; settles D-D4) | **Landed** as #543 on 2026-09-16: `environments/shared/ancestors.select_trunk`, notebook default `TRUNK_FROM = "auto"`, CLI `curriculum --trunk-from auto`. Canonical chains no longer consult the certified library; widen sessions select no trunk. |
| PR-2 (record the decisions, fix the stale docs) | **Executed by the 2026-09-19 documentation pass**: this document, the D-D and G rows in BEHAVIOR_RECIPES_PLAN.md §6.2, the docs index, the README roadmap bullet, the CHANGELOG, NEXT_STEPS.md, the Drive survey note (investigations/DRIVE_RUN_SURVEY_2026_09.md), four KNOWN_ISSUES.md entries, the template note's appended §6 and one paragraph in website/docs/training/recipes.md. |
| PR-3 .. PR-15 | **Released 2026-09-20** in the notebook-first order of decision D-D13 (§6): PR-3, PR-4, PR-5, PR-6, a notebook-only slice of PR-12, PR-14, then PR-7 .. PR-11, the rest of PR-12, PR-13, PR-15. |
| PR-3 (bound the SB3 CI job) | **In review** (2026-09-20, the session branch): the SB3-free suites leave the `test-sb3` lists (the `test` matrix runs them; verified locally with SB3, torch and ray blocked), the full six-species and four-notebook-parameter sets run on the nightly schedule and under the `full-ci` label, one real-PPO smoke per body of work stays on every PR, `walker` is module-scoped. Measured on the #544 merge run: the job took 52 minutes (notebook smoke 6, behaviors 14, integration 31). |
| Net removal from here | about 9,500 lines. The assessment counted about 10,500 from 723f58f; PR-1 was net zero and #543 added about 1,200 lines including tests. |
| Training | Not on hold. The walker sessions in NEXT_STEPS.md run on the current notebook in parallel with the sequence (G3). |
| Loader change (2026-09-19, outside this sequence) | The Colab image moved to Python 3.13 and both first attempts at NEXT_STEPS.md session 1 died inside the widen tool's self-verification (KNOWN_ISSUES, "SB3 archives are bound to the interpreter that saved them"). `policy_loading.load_sb3_model` is now the one archive loader, `linear_schedule` / `cosine_schedule` are picklable classes, and the notebook's load preflight is a cell right before the widen cell. Consequences for this plan: PR-14 item (d) has two disconnect-before-raise sites left (cell 22's), not three, and the notebook target of §4 gains one ~75-line code cell (preflight) between rows 7 and 8, right before the widen row (it reads `TRUNK_DIR`, which the storage row binds); the disconnect-site numbers are in the plan's `22c1fc8` cell numbering. |

Decisions: the maintainer took D-D1..D-D10 and G1..G4 on 2026-09-17 (§6, §7)
and confirmed D-D11 and D-D12 on 2026-09-20, when D-D13 (the notebook-first
order) and D-D14 (the widen path becomes CLI-only after sessions 1 and 2) were
taken. Ids follow the series recorded
in BEHAVIOR_RECIPES_PLAN.md §6.2; the review's working labels D1..D12 map
one-to-one onto D-D1..D-D12.

The hold lifted on 2026-09-20 with the order of D-D13: PR-3 (no prerequisites,
bounds the CI job that every later PR's validation runs through), then PR-4
and PR-5 in that order (the canonical wrapper imports the library, not the
reverse), PR-6, the notebook-only slice of PR-12 (pulled ahead of PR-11 so the
notebook shrinks first), PR-14, then PR-7 .. PR-11, the rest of PR-12, PR-13
and PR-15. Every later PR names its prerequisites and the decisions it rests
on; none of the decisions it needs is still open.

## 1. Headline

Two bodies of work landed in four days. The recipes program (#528–#539, ~31k
lines) built the canonical machinery: stage manifest v2, per-deliverable result
bundles with `gate_verdict.json` and `ancestors/`, the notebook chain loop with
`TRUNK_FROM` / `WIDEN_FROM` / `RETRAIN_FROM`, gate kinds behind one closed
dispatch, the Phase C interface bump that appended three zero command dims to
every observation, and `widen_checkpoint.py`; it also reserved, on purpose, the
exact hook and kwargs that Phase D was to fill
(`BaseDinoEnv._draw_episode_command`,
`command_mode`, `command_manifest()`, the `command_tracking/v1` gate, two
manifest nodes warm-started from the certified walker;
[BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §4.6). The
direction/terrain
PRs (#540–#541, 117 files, 18,400 lines) delivered the direction sampler,
tracking reward and a real heightfield generator, which are genuinely new and
worth keeping, but delivered them as a second pipeline beside every canonical
concept: two behavior env classes that refuse the reserved `command_mode` and
write `self._command` directly after `super().reset()`
(environments/shared/behavior_env.py:86-87, 397), a second checkpoint preparer
that overwrites the plant identity attribute with its own schema
(environments/shared/behavior_checkpoint.py:259-261), a second PPO trainer with
its own recipe dialect (environments/shared/train_behaviors.py:45-150, 540), 66
TOMLs that are an 11-behavior by 6-species outer product, a second gate outside
`GATE_KINDS` (environments/shared/behavior_certification.py:227-338), a third
identity keyed on source-file hashes, and a "certified library" that re-verifies
chains `find_certified_ancestor` already verifies, recounts replication
`discover_replicates` already counts, copies every parent checkpoint into every
child run, and, under the default `PUBLISH_CERTIFIED = True` that #542 has since
flipped, would have disconnected the Colab runtime the moment a widened stance
passed its gate (no widen session has run; #542 closed the path first; cell
22:298-321; environments/shared/certified_canonical.py:103-107). The
notebook grew to 40 cells and 2,500 lines with a `COMMAND_TERRAIN_BEHAVIOR`
switch guarding 12 whole cells and two partial ones.

The end state is the plan of record, filled in with the new content:
direction-following and terrain traversal are ordinary manifest nodes under
`locomotion`, the direction controller is the body of the reserved hook in
`BaseDinoEnv`, terrain is one generic opt-in env subclass (it cannot enter
`reset()`, whose source is fingerprinted), command-column zeroing is one
primitive in `policy_loading` used by `initialize_next_stage`, nodes are stage
TOMLs judged by a registered gate kind and reused through
`find_certified_ancestor`, the certified library is deleted, and the notebook
loses the mode switch and every library knob while keeping every knob the
maintainer already uses. Fifteen PRs, ordered so that the live workflow is
protected first and each PR stays reviewable; net about 10,500 lines removed
from 723f58f (roughly 60% of what #540–#541 added) with the uncertainty
concentrated in the stage-TOML form (D-D5) and the gate work (D-D6).

Two things the assessment did not know. The 2026-09-17 Drive survey
([investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md),
summarised in NEXT_STEPS.md) established that the certified r13 trex walker the
plan's Phase C½ asked for already exists (run 20260914_123816, stance and
locomotion both `gate_verdict.json` PASS with task and gate digests equal to
those derived from the current stage TOMLs, so `TRUNK_FROM = "auto"` reuses
both nodes), obtained from a from-scratch r13 stance rather than a widened r11
one. And nothing relies on a `mesozoic-labs/certified` library directory on
Drive since #542/#543 (the survey did not inspect it). Both narrow the
consequences noted under PR-4, PR-11 and D-D1 below.

## 2. Target architecture

| Concept | Today: recipes implementation | Today: #540/#541 implementation | After |
|---|---|---|---|
| Env command source | Reserved hook `BaseDinoEnv._draw_episode_command` returns zeros, `command_manifest()` returns None (environments/shared/base_env.py:1063-1083), called at base_env.py:1514 after every reset draw; six `command_*` kwargs stored but never read (base_env.py:195-200, 301-306); `command_frame.validate_command_mode` fails closed on SB3 (command_frame.py:77-94) | `DirectionCommandController` (environments/shared/direction_commands.py, 442 lines) driven from `SpeciesBehaviorMixin.reset/step/set_direction`, which refuse `command_mode != "none"` (behavior_env.py:86-87) and assign `self._command` directly (397, 417, 443); `command_manifest()` overridden (308) but never fed to `compute_task_fingerprint` | `BaseDinoEnv` owns the controller under `command_mode in {"heading","heading_and_speed"}` with one `command_config` kwarg replacing the five dead numeric kwargs (D-D2); hook body = `controller.reset(...)`, new `_update_command()` at the end of `step`, `command_manifest()` = `controller.manifest()` into the task fingerprint. Deleted: the refusals, the three direct writes, the SB3 branch of `validate_command_mode`, the five numeric kwargs in five species constructors and `MJXEnvConfig`. direction_commands.py survives unchanged in substance (imports `COMMAND_WIDTH`/`COMMAND_COMPONENTS`/`COMMAND_RANGE` from command_frame). |
| Terrain | None (base_env.py:1332-1344 documents plane-only settling) | terrain.py (522) + terrain_sampling.py (183) + two model/data pools swapped by `_select_contact_model` (behavior_env.py:177-193); four stacked layers decide an episode's ground: `flat_probability` draw (373-378), `TerrainSamplingMixin.reset` try/finally mutation (terrain_sampling.py:148-163), `_sampled_behavior_env_class` (171-183), `_PanelTerrainMixin` in behavior_certification.py:63-107 | terrain.py kept as the single injection point (`build_terrain_model`, plane named `floor`). One generic opt-in subclass (the surviving `SpeciesBehaviorMixin` minus its command half, built by `make_env` when `[env]` carries terrain keys) holds the two pools, the terrain settle, and ONE per-episode family selector (`select_terrain_family` with a `terrain_contact` family and a `terrain_family` reset option) (D-D10). Deleted: environments/trex/envs/behavior_env.py (497), `TerrainSamplingMixin`, `_sampled_behavior_env_class`, `get_sampled_behavior_env_class`, `_PanelTerrainMixin`, `flat_probability`. `BaseDinoEnv` gains `_ground_height_at(xy)` so species rewards/terminations are terrain-relative without re-derivation. |
| Checkpoint preparation | widen_checkpoint.py (1308): inserts zero columns into an r11/r12 archive, pads Adam moments, `pad_running_stats`, restamps, self-verifies with seed 3042 / atol 1e-6; `load_vecnorm_stats(reseed_command_slice=...)`; `_create_or_load_model` under `resume_same_stage` / `initialize_next_stage` (train_base.py:552-637), which does NOT zero existing command columns | behavior_checkpoint.py (457): `_zero_command_connections` on the same two layers and moments (145-162), same probe (208-231), `BehaviorVecNormalize` passthrough installed by `__class__` swap (58-75, 214) while still calling `reseed_command_slice` (216), `load_behavior_checkpoint` = resume, `adapt_behavior_checkpoint` + three hand-listed frozensets = initialize_next_stage (315-457); marker written into both `MODEL_IDENTITY_ATTRIBUTE` and `MODEL_TASK_ATTRIBUTE` (259-261) | `policy_loading.neutralize_command_columns(model, observation_dim)` (~35 lines) + `assert_command_blind(...)` (~25) sharing widen's probe constants; called by `_create_or_load_model` on `initialize_next_stage` when the parent fingerprint has no `command` section and the child's mode is live; resume = `resume_same_stage`, adapt = `initialize_next_stage` with lineage; command-slice normalisation follows the reseed rule (D-D3). Deleted: behavior_checkpoint.py and its 420-line test. widen_checkpoint.py unchanged. |
| Training entry point | `train_base.train` (980), `train_curriculum` (1999), `make_env` builds `species_cfg.env_class(**env_kwargs)` (204-228); notebook cell 16 `train_stage` (410 lines) re-runs train()'s body | `train_behaviors.main` (601 lines): own `read_recipe`, own env factory, own `model.learn` (540), own run.json/bundle.json, one CPU env; `behavior_notebook.NotebookBehaviorPlan.argv` (94-128) serialises 15 knobs into argv and calls `main()` in-process; trex shim (16) | `train_base.train` for everything; the notebook `train_stage` becomes a ~30-line wrapper over `train()` (which gains the seed line, `parent_run_id`, an eval-seed parameter, duration recording and a richer return) (D-D7). Deleted: train_behaviors.py, behavior_notebook.py, the shim, cells 7/20/34/39. |
| Recipe / stage configuration | Stage TOMLs via `load_stage_config` (config.py:184), `stages.toml` v2 edges, `StageEntry.warm_start_from/deliverable/recipe` (stage_manifest.py:108-128) | 66 `configs/<species>/behaviors/*.toml` (3,030 lines) whose only cross-species differences are 20 keys with one value per species, and whose within-species differences are 7 keys; 8 `configs/trex/behavior_pilots/*.toml` (239) with a `[pilot]` dialect branch in `read_recipe` (train_behaviors.py:51-66, 76, 112) referenced only by pyproject.toml:114, environments/trex/tests/test_behavior_training.py:17 and one doc line | Per species three new stage TOMLs (`follow_direction`, `follow_direction_difficult_terrain`, `difficult_terrain`; the assessment listed a fourth, `follow_direction_speed`, which G1 folds into `follow_direction`) with `[env] command_mode`, `command_config`, `terrain_*` scalars, `[ppo]`, `[curriculum]`, plus `[[stages]]` entries after `behavior`; the nine single-template variants become documented `[env]` overrides for the manual cell. `load_stage_config` gains a ~20-line `extends` key (D-D5). Deleted: all 74 TOMLs, the `[pilot]` branch, the pyproject globs. |
| Gate / certificate | Closed `GATE_KINDS` + `_REQUIRED_THRESHOLD_KEYS` (curriculum/gate_schema.py:53-191), `evaluate_stage_gate` (reporting/gates.py:688-778), `_apply_stage_gate` -> `gate_verdict.json` (stage_artifacts.py:1022-1134), `binomial_lcb`/`paired_difference_lcb` | `judge_behavior_panel` (behavior_certification.py:227-338) reading configs/behavior_certification.toml, its own `gate_sha256` from source-file hashes and package versions (35-61), output `certification/certificate.json`, never `gate_verdict.json`; imports only `binomial_lcb` | One new gate kind, `terrain_command/v1` (D-D6), registered in `GATE_KINDS` with an evidence writer in the `write_recovery_evidence` pattern, judged by `evaluate_stage_gate`, written as `gate_verdict.json` so reuse rules 1-7 apply, first thresholds from G4; first pilots run `none/v1` recorded-not-enforced (gate_schema.py:128). Deleted: behavior_certification.py, the TOML, the certificate schema. |
| Publication and reuse | `gate_verdict.json` hash-bound to the handoff pair, `provenance.deliverables[*].replication/provisional`, `ancestors/` records that never copy checkpoints (ancestors.py:678-686, plan A10), `find_certified_ancestor` rules 1-7, `discover_replicates` | certified_library.py (671), certified_canonical.py (917), certified_comparison.py (153): immutable store re-hashed on every read, second replication counter (258-289), second paired statistic (291-346), flock, complete copies into `certified_inputs/`, `copy_canonical_ancestor` reversing A10, `stamp_canonical_training` written only by the notebook | The recipes publication as it stands, with automatic parent selection provided by `ancestors.select_trunk` (D-A25, landed as #543). Deleted: all three modules, four test files, test_behavior_publication.py, docs/CERTIFIED_MODELS.md, the four notebook knobs and three notebook blocks, `--auto-source/--publish-certified/--certified-library`. No recommendation pointer survives (D-D4). |
| Evaluation | `eval_policy_quality` on `LocomotionMetrics` (evaluation.py:111), `load_sb3_checkpoint` (policy_loading.py:96) | `evaluate_behavior` (behavior_evaluation.py:401-640) importing nothing from evaluation/metrics, `evaluate_saved_panel` wrapper, `benchmark_canonical_stage` fourth rollout loop, three `_sha` helpers and two `_json_value` coercers | `summarize_episode`, `terrain_family_from_reset`, `by_terrain_family` kept as the behavior-specific statistic feeding the gate's evidence writer; one `sha256_file` and one `json_ready`. Deleted: `benchmark_canonical_stage`, `evaluate_saved_panel`, the helper copies, `_PanelTerrainMixin`. No shared rollout generator (refuted as a non-simplification). |
| Replay / video | `record_stage_video` (evaluation.py:221-370): fresh render env at `replay_seed`, mediapy, camera presets, stance CSV the r11/r13 reviews read | `BehaviorReplayRecorder` gym.Wrapper (behavior_replay.py:305-500): same scored trajectory, heightfield snapshot, terrain maps, decoded-MP4 check | Both kept (keep verdict). Only the `_sha`/`_json_value` helpers unify; mediapy stays in the Colab install line (evaluation.py:257, 354 record the canonical replays). |
| Notebook routing | Chain loop (cell 22, 339 lines) + `train_stage` (cell 16:201-610) + widen (11) + resume (26) | `COMMAND_TERRAIN_BEHAVIOR` membership switch (cell 6:64-84) guarding cells 8,10,11,12,14,24,26,28,30,32,36,38 and partials 16:928 / 22:38-47; second storage cell (7), run/display/disconnect cells (20/34/39); 10 `BEHAVIOR_*` + 4 `CERTIFIED_*`/`SOURCE_SELECTION` knobs; second output tree without provenance | One notebook, no switch, no second tree: follow/terrain nodes are values of `BEHAVIOR` walked by the existing chain loop; `train_stage` wraps `train_base.train`; widen-seed check runs before the run directory is minted; `RUN_ID` is a knob. Knobs kept: SPECIES, ALGORITHM, N_ENVS, SEED, VERBOSE, QUICK_TEST, USE_GOOGLE_DRIVE, AUTO_DISCONNECT, BEHAVIOR, TRUNK_FROM, WIDEN_FROM, WIDEN_MAX_REVISION_GAP, RETRAIN_FROM, RUN_LABEL, RUN_ID, REPO_REF. |
| Vocabulary | "behavior" = recipe label and the hunt node id `behavior` (RESERVED_STAGE_IDS, stage_manifest.py:79); "certified" = gate passed with every ancestor passed, hash-bound (`CertifiedAncestor`); "identity" = `PlantIdentity` + task fingerprint | "behavior" = 28 public symbols, 4 checkpoint stamps, 14 knobs, two config trees; "certified" = a recommendation store; a third `behavior_identity` schema (`mesozoic.command-terrain/v1`) hashing five source files; checkpoint stamps 4 -> 12 | "behavior" = recipe label/node only; the surviving env feature is named "command" and "terrain"; "certified" keeps the plan §2 meaning; identity = plant identity + task fingerprint (`command` section via `command_manifest`, terrain via `[env]` kwargs); stamps back to the four recipes attributes. |

## 3. Ranked PR sequence

Sizes: S < 200 changed lines, M < 800, L < 2000, XL above. "Validation" names
the checks that exist today: ruff, mypy on environments/, the shared suite
(`pytest environments/shared/tests/`), the species suites, the notebook
round-trip/parse step (python-ci.yml:81-111) and the SB3 job. "Folds in" names,
in plain words, which of the review's proposals each PR absorbs, so that a
reader of the review's working notes can find where each idea went.

### PR-1. Stop the live disconnect (S, 4 files, +10/-7, no decision needed) — LANDED as #542, 2026-09-16
Goal: prevent the next WIDEN_FROM session of 20260810_145546 / 20260815_205206,
and any resume of the r13 run 20260914_123816, from disconnecting after stance
passes. Chain verified: widen writes `<label>_final.*` and never
`gate_verdict.json` (widen_checkpoint.py:56-61, 1049-1050) so the JUDGE branch
fires; only `train_stage` stamps `mesozoic_canonical_training` (cell 16:464;
certified_canonical.py:45, 59); `publish_canonical_stage` calls
`_training_origin` (certified_canonical.py:669), which raises
`CanonicalLibraryError("Automatic publication requires recorded canonical
training origin")` (103-107); cell 22:319-321 catches, calls
`disconnect_runtime` and re-raises before the gate enforcement (325) and before
`NODE_HANDOFF[NODE.id]` is set (330).
Folds in: the review's stop-gap for the library's publish-time disconnect, the
correction that a widened root (not a trained node) is what trips it, and the
evidence chain above.
Files: notebooks/sb3_training.ipynb cell 6:25 (`PUBLISH_CERTIFIED = False`),
environments/shared/tests/test_sb3_notebook_pins.py:321 (pin flips to `is
False`), environments/shared/tests/test_behavior_notebook.py (two
`PUBLISH_CERTIFIED` asserts) and docs/CERTIFIED_MODELS.md (knob-table row).
Breaks: nothing. Mitigation: none needed. Validation: notebook parse step;
`pytest environments/shared/tests/test_sb3_notebook_pins.py`.
Prerequisites: none. Landed as #542 on 2026-09-16.

### PR-2. Record the decisions in the design of record (S, docs only; net about +100 lines) — THIS PASS (2026-09-19)
Goal: take decisions D-D1..D-D10 (§6; D-D11/D-D12 recorded as recommended)
in docs/BEHAVIOR_RECIPES_PLAN.md before code moves, and fix the stale claims: §5 Phase D row (plan:925) "landed
2026-09-15 as a separate pipeline (#540/#541), consolidation pending"; D-C5
(plan:1030) note that the pilots bypass the hook; new decision block D-D1..
naming which #540/#541 mechanisms are adopted (terrain generator, direction
controller, tracking reward, per-episode settle statistic, replay recorder) and
which are folded; docs/README.md:42 plan row; README.md:511 ("only ... remain
pending"); a CHANGELOG `### Added` / `### Migration` block under v0.3.8
(terrain/direction modules; `certified/` beside `logs/`; pilot bundles become
evaluation-only; `imageio-ffmpeg` in `viz`). The 2026-09-19 pass also records
the goal decisions G1..G4 (§7) and writes NEXT_STEPS.md.
Folds in: the review's decision-and-vocabulary record, the stale-claim half of
its documentation findings, and the CHANGELOG `Added`/`Migration` half (it adds
lines, it does not remove them).
Breaks: nothing; python-ci `docs/**` filter runs the workflow once. Validation:
none beyond the workflow.
Prerequisites: none; every later PR cites the decision numbers.

### PR-3. Bound the SB3 CI job (S, about +10 to +40 workflow lines; about 40 minutes off every PR run) — IN REVIEW (2026-09-20)
Goal: measured test-sb3 went 43 -> 69 minutes (#539 -> #541; steps 7.8 / 18.3 /
40.8 min). Remove from the test-sb3 lists (python-ci.yml:302-305, 313-343) every
suite with no SB3 import that the `test (shared|trex)` matrix already runs three
times (test_behavior_env.py, test_behavior_recipes.py, test_terrain_sampling.py,
test_behavior_evaluation.py, test_certified_library.py,
test_certified_comparison.py, test_behavior_publication.py,
test_sb3_notebook_certified.py, environments/trex/tests/test_behavior_env.py,
the last one currently run five times). Keep one real-PPO smoke per body of work
in the PR gate via `-k`; switch to the full six-species and four-notebook-param
set through a schedule/`full-ci` label condition inside the same job so the
required-check name is untouched. Make `walker` in
test_behavior_species_training.py module-scoped (6 PPO builds instead of 18).
Folds in: the review's CI-length finding in its corrected form (two of its items
dropped: `--eval-episodes 0` is already passed at :85-86, and the remaining
suites are not large contributors).
Breaks: coverage `fail_under=70` may need one re-baseline. Validation: a CI run
on the branch. Prerequisites: none; do before PR-4 so later suite deletions edit
one list.
As executed (2026-09-20, the session branch): ten suites leave the lists, the
nine above plus test_behavior_certification.py, which imports no SB3 either
(all ten verified locally with stable_baselines3, torch, cloudpickle and ray
blocked); the `pull_request` trigger gains the `labeled` activity type so that
adding the `full-ci` label to an open pull request starts the run that reads
it (a re-run replays the original event payload); the `full-ci` label has to
be created once on the repository before the label path can be used; the
nightly schedule runs the whole workflow on `main`, with the SB3 job at full
depth. Measured before the change on the #544 merge run: 52 minutes
(notebook smoke 5:50, behaviors 13:34, integration 30:31).

### PR-4. Delete certified_canonical.py, certified_comparison.py and the notebook library hooks (L by count, mostly file deletion; about -2,400)
Goal: remove the wrapper that publishes canonical stages into the library and
every notebook hook. Delete environments/shared/certified_canonical.py (917),
certified_comparison.py (153), tests test_certified_canonical.py (699),
test_certified_comparison.py (97), test_sb3_notebook_certified.py (438).
Notebook: cell 16:455-479 (stamp block; also stop rewriting stage_config.json),
the `copy_canonical_ancestor` call in the chain loop's reuse branch (still in
cell 22 at 22c1fc8; #543 already removed the `resolve_canonical_parent` branch
that sat beside it at 22:109-131, with its import), 22:295-321 (publish block),
22:39-44 imports, cell 6:21-32 (`CERTIFIED_LIBRARY_ROOT`, `PUBLISH_CERTIFIED`,
`CERTIFIED_COMPARISON_EPISODES`; `SOURCE_SELECTION` stays for the
direction/terrain path until PR-5 removes `--auto-source`), cell 7:14 and 8:24
`CERTIFIED_LIBRARY` lines, cell 5 markdown 13-35 library prose. Pins:
test_sb3_notebook_pins.py:320-322 (defaults), the reuse-branch pin at :694
(#543 re-pointed it: at 22c1fc8 it is :697 and asserts
`resolve_canonical_parent` is absent, so only its `copy_canonical_ancestor`
half is left to drop), :1245-1247 (`PUBLISH_CERTIFIED` if); the rename
`..._and_loads_complete_copies` reverts to the A10 pin
`..._never_copies_checkpoints`; :709
`test_the_library_rule_the_notebook_relies_on` STAYS (it pins
`find_certified_ancestor`).
Folds in: the review's findings on the canonical publish wrapper (the
`starting_updates` pointer idea dropped with D-D4), the library-knob half of the
notebook findings, and the canonical half of the duplicated-provenance finding.
Breaks: `SOURCE_SELECTION="auto"` behaviour for canonical chains (no longer
copies; #543 already replaced it with `select_trunk`); runs since 2026-09-16
holding `certified_inputs/` stay valid because their `ancestors/` records point
at the copy and rule 1 follows it. Mitigation: `TRUNK_FROM = "auto"` and
`find_certified_ancestor(follow_records=True)` already cover the maintainer's
reuse. The r11 parents, the r13 run and their verdicts live in logs/ and are
untouched. Validation: full shared suite; notebook parse;
`test_compsognathus_training.py::test_actual_notebook_training_stance_and_recovery_reports[ppo-compsognathus_robot]`
(the notebook smoke).
Prerequisites: PR-1 (#542); D-D4 (taken 2026-09-17: the library is deleted
outright, its automatic selection replaced by `select_trunk`, which landed
as #543 so the capability never lapses). Land before PR-5 (canonical imports
library, not the reverse).

### PR-5. Delete certified_library.py and its consumers in the behavior trainer (M/L, about -1,400)
Goal: delete environments/shared/certified_library.py (671),
test_certified_library.py (303), test_behavior_publication.py (170; imports
`copy_recommended/resolve_recommended` at :12), docs/CERTIFIED_MODELS.md (198;
its two useful paragraphs on ancestor reuse are already in
docs/RESULT_BUNDLES.md:61-120), the `.gitignore` `/certified/` line; in
train_behaviors.py strip `--auto-source`, `--publish-certified`,
`--certified-library` (275, 340-368), `_copy_explicit_certified_pair` (211-232),
`certified_source.json`; in behavior_certification.py strip
`certify_and_publish_behavior` (395-558), `comparison_from_panel`,
`behavior_library_key`, `save_head_to_head` call; test_behavior_notebook.py
loses its 16 "certified" references; the notebook loses `SOURCE_SELECTION` with
its last reader; CI wheel step drops
`load_certification_rules()["comparison"]["episodes"] == 50` and `RULES_PATH`
asserts (python-ci.yml:180-181).
Folds in: the review's library finding as the "delete outright" option it
recommended, the library half of the duplicated-provenance finding, and the CI
wheel-step assert cleanup.
Breaks: nothing the maintainer uses; a `mesozoic-labs/certified` directory on
Drive exists only if #541's notebook ran with `PUBLISH_CERTIFIED=True` (the
2026-09-17 survey did not inspect it; nothing relies on one since #542/#543);
leave it, nothing reads it.
Validation: shared suite; SB3 job (`test_behavior_species_training` smoke).
Prerequisites: PR-4.

### PR-6. Delete the T. rex pilots, the `[pilot]` dialect and the shim (S, about -275)
Goal: delete configs/trex/behavior_pilots/ (8 files, 239 lines; twins of
configs/trex/behaviors/ differing only in header and omitted defaults,
`trex_gentle_terrain` = `sloped_terrain`), pyproject.toml:114,
environments/trex/scripts/train_behaviors.py (16), the `[pilot]` branch of
`read_recipe` (train_behaviors.py:51-56, 64-66, 76, 112, 296),
`recipe.get("pilot", ...)` at behavior_certification.py:471, the
`mesozoic.trex-command-terrain/v1` string at behavior_checkpoint.py:374 (keep
accepting for one release only if any pilot bundle is worth `adapt`; D-D9 says
none is), `mesozoic.behavior-pilot-run/v1` at behavior_notebook.py:238,
`PILOT_RECIPES` alias (:346); repoint
environments/trex/tests/test_behavior_training.py:17 `PRESETS` to
configs/trex/behaviors and rewrite its two `[pilot]` bad-recipe strings;
docs/TRAIN_DIRECTION_AND_TERRAIN.md:293-298.
Folds in: the pilot-dialect finding, the pilot half of the trex-duplication
finding, and the first step of the config-generator proposal.
Breaks: PR #540 command lines using the pilot paths; exact resume of 2026-09-15
pilot bundles, which is already stranded by the source-hash identity
(behavior_checkpoint.py:277-278 requires `marker["behavior_identity"] ==
expected`, and the identity hashes behavior_env.py itself). Validation: trex
suite, shared suite, wheel step. Prerequisites: none.

### PR-7. One behavior env, part 1: ground-height hook and delete TRexBehaviorEnv (M, about -600)
Goal: (a) add `BaseDinoEnv._ground_height_at(xy) -> 0.0` and `_clearance(xyz)`;
route the species height reward / head-or-snout clearance reward / height
terminations through it (trex_env.py ~770-775, 938-947; brachio_env.py ~435;
dibothrosuchus_env.py ~478; compsognathus_env.py ~232) and have the substep
aggregation in `BaseDinoEnv.step` (base_env.py:1144-1165) include the free-joint
root min and max clearance so the mixin's `_probe_ground_clearance` /
`_check_height_tilt_termination` / `_aggregated_min_height` /
`_current_clearances` overrides (behavior_env.py:227-249) and the re-derived
height block (265-281) go; on the plane every canonical number is `z - 0.0`, so
the golden trajectory fixture, gates and stage fingerprints are unchanged
(species `_get_reward_info`/`_is_terminated` are not fingerprinted:
digests.py:154-166 hashes `_get_obs`, `_scale_action`, `reset`). Keep
`_settle_root_on_ground` and the `lowest_ground_clearance` raise in the mixin
(the terrain settle on the plane copy is a different algorithm;
`mj_geomDistance` is invalid on hfields, terrain.py:498-499). (b) Delete
environments/trex/envs/behavior_env.py (497): `_settle_root_on_ground` is
byte-identical (trex 292-322 vs shared 319-349, re-diffed),
`step`/`reset`/`set_direction`/`_clearance` differ only by `_neck_hit` and a
second `apply_terrain`; move the neck probe pool behind a
`_terrain_contact_probe_geoms = ("neck_geom",)` class attribute on `TRexEnv`
(~60 lines into the mixin), move `_assert_matching_ids`/`_assert_same_animal`
into the mixin (behavior_env.py:174-175 borrow them today), drop the `if species
== "trex"` branch (449-454), the trex file from the `sources` hash list
(196-202), the `mesozoic.trex-command-terrain/v1` schema. Pick the generic
geom-bounds boundary rule. Tests: environments/trex/tests/test_behavior_env.py
(487) and test_behavior_training.py (449) fold into
environments/shared/tests/test_behavior_env.py parametrised over species,
keeping the neck-probe cases; update test_behavior_env.py:54 and
test_terrain_sampling.py:81-85. Delete trex test_behavior_training.py first (it
imports `CommandEnv` from test_behavior_checkpoint.py:384).
Folds in: the ground-height-hook finding (corrected: the settle and the
clearance raise stay), the TRexBehaviorEnv-duplication finding (ordered after
the hook), and the trex test-file collapse.
Breaks: exact resume/adapt of every existing behavior bundle (already stranded
by the source hashes; D-D9). Mitigation: one new test asserting
reward/termination equality between a plane env and a behavior env with
`terrain=None`. Canonical artifacts unaffected. Validation: species suites,
shared suite, `plant_contract --check` (must report no interface change), golden
reset fixture (test_phase_c_interface.py). Prerequisites: PR-6.

### PR-8. One behavior env, part 2: one terrain selector, one command-constant source, one normalisation (M, about -170)
Goal: (a) fold `TerrainSamplingMixin` (terrain_sampling.py:89-168),
`_sampled_behavior_env_class`, `get_sampled_behavior_env_class`, the
factory-by-key logic (train_behaviors.py:161, behavior_certification.py:105) and
`_PanelTerrainMixin` + the `type("BehaviorCertificationPanelEnv", ...)`
(behavior_certification.py:63-107) into one `terrain_sampler` kwarg on the mixin
whose `reset()` calls `select_terrain_family` directly, keeps the
`options={"terrain_family": ...}` override, and gains a `terrain_contact`
(flat-heightfield, `mode = "flat"`) family the sampler cannot express today
(terrain_sampling.py:108-109); map `[terrain]`-without-sampler recipes to block
weights (e.g. `flat=1, <template>=3` for `flat_probability = 0.25`; Bernoulli ->
balanced blocks is a distribution change, stated in §8). Remove
`flat_probability` from `_TRANSITION_TOP_SETTINGS` (behavior_checkpoint.py:350-351).
(b) direction_commands.py imports
`COMMAND_WIDTH`/`COMMAND_COMPONENTS`/`COMMAND_RANGE` from command_frame.py
instead of re-stating them (137, 202, 334). (c) D-D3 (taken): delete
`BehaviorVecNormalize`, the `__class__` swap and the `isinstance` gate
(behavior_checkpoint.py:58-75, 214, 299) so behavior sidecars are plain
`VecNormalize` files and the plan's invariant 8 (reseed) is the one rule; the
passthrough alternative (keep the subclass, delete the redundant
`reseed_command_slice` call at :216, document the pickled-class dependency) is
closed. (d) drop the dead `lateral_speed_scale` (direction_commands.py:321
always emits `v_y = 0`) when the TOMLs are rewritten in PR-11 (D-D12,
recommended).
Folds in: the terrain-selector finding (corrected), the reduced form of the
reward-helpers finding (the move into reward_functions.py is dropped), and the
single-probe-env / terrain-family-property finding.
Breaks: test_terrain_sampling.py mixin cases rewritten against the env;
test_behavior_checkpoint.py passthrough cases; identity of single-template
recipes (already stranded). Validation: shared suite, test_behavior_env
parametrised suite. Prerequisites: PR-7; D-D3.

### PR-9. Phase D through the reserved hook; identity = task fingerprint (M, about -150 net here, more in PR-12)
Goal: `BaseDinoEnv.__init__` takes `command_mode` and ONE `command_config:
DirectionCommandConfig | None` (D-D2), replacing `command_speed_range`,
`command_lateral_range`, `command_yaw_rate_max`, `command_switch_interval`,
`command_switch_jitter` in base_env.py:195-200/301-306, `MJXEnvConfig`
(mjx_env.py:274-279) and the five species constructors
(trex_env.py:195-199/323-327, raptor_env.py:113-117/186-190,
brachio_env.py:109-113/196-200, dibothrosuchus_env.py:131-135/213-217,
compsognathus_env.py:71-75/154-158); the controller is constructed and seeded
only when `command_mode != "none"`; `_draw_episode_command()` returns
`controller.reset(rng, self._heading()).normalized` (the hook already runs after
the settle and the push draw, base_env.py:1514); a new `_update_command()` at
the end of `BaseDinoEnv.step` returns `controller.update(t,
heading).normalized`; `command_manifest()` returns `controller.manifest()` and
`derive_stage_task_fingerprint(command_manifest=...)` carries it; extend the
carve-out at task_fingerprint.py:161-170 to pop `command_config` while the
effective mode is `none` (otherwise every canonical `task_sha256` moves,
breaking test_phase_c_interface.py:348 and the r13 run's identity) and record it
via `asdict` (a ~6-line dataclass branch in `_canonical`, :100-113). Delete the
SB3 branch of `validate_command_mode` (command_frame.py:89-92; MJX keeps failing
closed at mjx_env.py:148-154), the two refusals, the three direct writes, the
mixin's `command_manifest` override, `behavior_identity`
(behavior_env.py:194-221), `BEHAVIOR_IDENTITY_SCHEMA`, `PREPARATION_ATTRIBUTE`,
the source-hash `sources` block, `sampler_source_identity`
(terrain_sampling.py:83-86) and `_evaluation_contract`
(behavior_certification.py:40-48) in favour of versioned strings like
`SCHEDULE_IMPLEMENTATION` (task_fingerprint.py:71). Terrain kwargs
(`TerrainConfig`, `TerrainSamplerConfig`) enter the fingerprint's `env` section
as constructor kwargs of the opt-in subclass. The mixin's zeroing of 20 species
reward weights (behavior_env.py:101-137) becomes explicit `[env]` keys in the
PR-11 TOMLs. `_heading()` (behavior_env.py:223-225) moves to `BaseDinoEnv`.
Folds in: the reserved-hook finding (corrected), the third-identity finding, and
parts (a)-(c) of the cross-cutting identity proposal (part (d), terrain into
`reset()`, is rejected: `home_reset` fingerprints
`inspect.getsource(env.reset)`, policy_layer.py:399 and digests.py:154-166, and
the model swap must precede `mj_resetDataKeyframe` at base_env.py:1452, so it
stays in the subclass; D-D10).
Breaks: `test_sb3_refuses_command_mode_other_than_none_in_phase_c`
(test_phase_c_interface.py:321) and `SB3_COMMAND_REFUSAL` cases in
test_command_frame.py are replaced; test_mjx_phase_c_interface.py loses five
fields; D-D2 amends D-C3 ("(mode, config) replaces six kwargs"). Plant identity,
golden reset stream, the r11/r13 artifacts and every committed stage fingerprint
are unchanged because `command_mode` stays `none` and nothing fingerprinted is
edited. Validation: `plant_contract --check`, golden fixture, shared suite, MJX
suite, species suites, on every species (§8). Prerequisites: PR-7, PR-8; D-D1,
D-D2.

### PR-10. One command-column primitive in the canonical warm-start path (S/M, about +180 now; enables PR-12's -720)
Goal: `policy_loading.neutralize_command_columns(model, *, observation_dim)`
(body of `_zero_command_connections`, behavior_checkpoint.py:145-162,
generalised over widen's `_OPTIMIZER_MEMBERS`) and
`policy_loading.assert_command_blind(model, normalizer, *, seed=3042,
atol=1e-6)` (the probe at 202-230) sharing `VERIFICATION_ROLLOUT_SEED`,
`ACTION_DELTA_ATOL`, `COMMAND_PROBE_VECTOR` with widen's `_verify`.
`_create_or_load_model` (train_base.py:552-637) calls it under
`initialize_next_stage` when the loaded checkpoint's recorded fingerprint has no
`command` section and the child's `command_mode` is live, recording
`zeroed_command_parameters` in the task lineage. This is needed regardless of
consolidation: the from-scratch r13 stance run 20260914_123816 was initialised
with random command columns that never received a gradient, and the plan
requires a zero action delta on non-zero commands at the follow warm-start
(plan:780). Widen's insert-then-verify stays separate (identity gate refuses gap
< 1 by D-C17; it rewrites the serialized archive).
Folds in: the primitive half of the checkpoint-preparer finding (the widen-mode
variant is not adopted: it needs its own identity gate, and the primitive is the
smaller change).
Breaks: nothing today (every current stage TOML has `command_mode = none`).
Validation: ~80 lines of tests in test_policy_loading/test_train_base (zero
columns, zero moments, action equality on a real r13 walker fixture);
test_widen_checkpoint.py untouched. Prerequisites: PR-9 (fingerprint carries the
`command` section); D-D3.

### PR-11. Manifest nodes: stage TOMLs and `[[stages]]` entries, trained by train_base (M, about +370)
Goal: per species, three new stage TOMLs under G1/G2 (the assessment listed
four, with a separate `follow_direction_speed.toml`; G1 folds it into
`follow_direction`): `configs/<species>/follow_direction.toml` (`[env]
command_mode` for the full command set of G2, `command_config` table with
`speed_range` as a fraction `[0.5, 1.0]` of `cruise_speed`, verified identical
on all six species, `[ppo]` = the recipe's learning_rate/ent_coef/target_kl,
`[curriculum] timesteps/warmup_timesteps/warmup_clip_range/gate_kind =
"none/v1"`), `follow_direction_difficult_terrain.toml` (the target deliverable:
the same commands plus terrain_* scalars and `terrain_families`), and
`difficult_terrain.toml` (commands-free, terrain_* scalars, `terrain_families`;
the optional diagnostic sibling); `[[stages]]` entries after `behavior` with
`warm_start_from = "locomotion"` / `"follow_direction"`, `deliverable = true`,
`recipe = "follow"` / `"terrain"` (the manifest allows a second child of
locomotion: `parent_of`, stage_manifest.py:232-237; `resolve_behavior` picks the
deepest deliverable per label, 253-262, so `follow` resolves to
`follow_direction_difficult_terrain`). The 19 per-species scale values
(`cruise_speed`, three scales, `course_distance`, `max_episode_steps`, 13
terrain size keys) come from the TOMLs verbatim. D-D5 takes `extends =
"locomotion"` (~20 lines in `load_stage_config`; no inheritance exists today)
over self-contained `[env]` blocks like configs/trex/behavior.toml. The nodes
are non-advancing (no `legacy_number`), so `train_curriculum --target
follow_direction` refuses them by design (train_base.py:1937-1955) and they run
through the notebook chain loop or `train --stage <id> --load-mode
initialize_next_stage`; the notebook needs no new cell. The final node is
SB3-only (G1). ~40 lines of tests asserting the entries load through
`load_stage_config` and resolve their parent.
Folds in: the target half of the trainer finding (corrected), the stage-TOML
form of the config-dialect finding (not a third dialect), step (b) of the
config-generator proposal (the generator-script variant is superseded by D-D1),
and the target half of the cross-cutting trainer finding.
Breaks: nothing yet (the trainer still reads the 66 files until PR-12). Workflow
consequence to state: under the manifest a follow/terrain node needs a CERTIFIED
locomotion ancestor at the Phase C interface (r13 for trex; each other species
at its own current `policy_interface_revision` in configs/plant_versions.toml).
At the time of the assessment none existed; the
2026-09-17 Drive survey found one for trex (20260914_123816, both nodes reused
by `TRUNK_FROM = "auto"`), and the walker sessions in NEXT_STEPS.md produce the
other five species' in parallel with this sequence (G3). The explicit-load
escape is the manual single-node cell (cell 24). Validation: `load_stage_config`
tests; one real-PPO smoke `train --stage follow_direction` for
compsognathus_robot and trex through the SB3 job. Prerequisites: PR-9, PR-10;
D-D1, D-D5, G1, G2.

### PR-12. Delete the parallel trainer, router, checkpoint module, 66 TOMLs, notebook mode and their tests (XL by count, almost all deletion; about -5,300)
Goal: delete environments/shared/train_behaviors.py (601), behavior_notebook.py
(351), behavior_checkpoint.py (457), configs/*/behaviors/ (66 files, 3,030
lines), pyproject.toml:113 glob; tests test_behavior_checkpoint.py (420),
test_behavior_notebook.py (792), test_behavior_species_training.py (242;
replaced by one real-PPO smoke per species through the manifest node, ~200
lines), test_behavior_recipes.py (179; replaced by PR-11's manifest tests), the
`CommandEnv` fixture; notebook cells 7, 19, 20, 33, 34, 39, the 14 guard sites
(dedent cells 8,10,11,12,14,24,26,28,30,32,36,38; cell 16:928-934 and 22:38-47
keep one print each), cell 6:48-84 (10 `BEHAVIOR_*` knobs,
`COMMAND_TERRAIN_BEHAVIOR`, the `ModuleNotFoundError` shim at 70-76, the
N_ENVS/SEED caveat comments at 14-15), cell 3:7 stale comment; `BEHAVIOR`
dropdown becomes `stand | walk | hunt | follow | terrain` plus stage ids; the
guard-stripping helper `_canonical_source` (test_sb3_notebook_pins.py:78-88) and
the 46 pin lines #540/#541 added; CI collapses the two behavior steps
(python-ci.yml:300-324) into the SB3 job and drops the wheel-step recipe loop
(182-187). docs/TRAIN_DIRECTION_AND_TERRAIN.md shrinks to an operator guide
(~150 lines: behavior table, species-scale table, terrain templates, replay)
with the CLI walkthroughs and notebook-knob paragraph removed.
Folds in: the delete halves of the trainer, checkpoint and cross-cutting
findings, the notebook mode-switch findings (no interim behaviors notebook,
D-D8), the `BEHAVIOR_*` half of the knob finding, the checkpoint-suite collapse,
and the operator-guide cut.
Breaks: every behavior bundle from #540/#541
(bundle.json/run.json/`checkpoints/step-*`) stops loading anywhere (D-D9); the
parallel Drive tree `logs/<species>/ppo/behaviors/` becomes orphaned pilot
output (leave it). The canonical chain, both r11 parents, the r13 run,
WIDEN_FROM/TRUNK_FROM/RETRAIN_FROM and the resume cell are untouched.
Validation: full shared and species suites, notebook parse and pins, the SB3 job
with the new smoke, wheel step. Prerequisites: PR-11.

### PR-13. The gate kind, its evidence writer, and the end of the second gate system (L, about -400 net)
Goal: register `terrain_command/v1` (D-D6) in `GATE_KINDS` and
`_REQUIRED_THRESHOLD_KEYS` in the same commit (parity raise at
gate_schema.py:186-190) with thresholds from configs/behavior_certification.toml
carried once into a `[curriculum]` block per node (never 66 copies), the first
values set by G4; move `episode_measurements`
(behavior_certification.py:181-224) and `summarize_episode`
(behavior_evaluation.py:72-300) into a `curriculum/<kind>_gate.py` as the
per-episode statistic; write an evidence writer in the `write_recovery_evidence`
pattern producing per-episode rows (`command_v_x`, `command_yaw_rate`,
`tracking_error_v`, `tracking_error_yaw`, `terrain_family`, `course_progress_m`,
survival), hash-bound to model.zip/vecnormalize.pkl the way
`_task_success_stage_gate` binds its CSV (gates.py:463-); judge in
reporting/gates.py; the notebook's JUDGE/TRAIN branches then write
`gate_verdict.json` through `_apply_stage_gate` with no new cell. The name is
honest by decision: the certificate samples per EPISODE and lacks the plan's
per-event settle/dwell statistic, worst-of-eight-heading-bins floor and
`paired_difference_lcb` against the command-blind walker (plan:803-815), so it
is registered for what it is; the plan's `command_tracking/v1` with the paired
null is a later second kind, outside this sequence. Delete
behavior_certification.py (558), configs/behavior_certification.toml,
test_behavior_certification.py (212; panel-judging cases re-homed as gate cases
in test_gate_dispatch_fail_closed.py / test_reporting_gates.py),
`evaluate_saved_panel`, the `evaluate_behavior` wrapper layers, the three `_sha`
helpers (`train_behaviors._sha` is gone; `behavior_replay._sha` :50 and
`certified_library._hash_file` are replaced by
`result_bundle.hashing.sha256_file`) and the second `_json_value`
(behavior_replay.py:36-47 vs behavior_evaluation.py:301-314). Keep
`BehaviorReplayRecorder`, `capture_terrain_snapshot`, `write_terrain_maps` (keep
verdict; the canonical `record_stage_video` rolls a different episode on purpose
and carries the stance CSV the r11/r13 reviews read).
Folds in: the gate-kind finding (corrected), the replay-recorder finding
(corrected; recorder deletion rejected), the helper half of the evaluation
finding, and invariant 10 (the fail-closed dispatch test gains a case).
Breaks: `certification/certificate.json` files on Drive stay as history (no
behavior has passed the certificate per #541). Validation: gate schema tests,
dispatch fail-closed test, shared suite, one notebook smoke with `gate_kind`
set. Prerequisites: PR-11, PR-12; D-D6, G4.

### PR-14. Notebook: `train_stage` over `train_base.train`, widen-seed check before minting, one storage and one disconnect path, `RUN_ID` as a knob (M, about -450)
Goal: (a) extend `train_base.train` with the notebook's `alg_kwargs["seed"] =
SEED` line (cell 16:439; train() never seeds model construction, a CLI/notebook
drift the merge resolves in train()'s favour; D-D11), `parent_run_id`
passthrough, an eval-env seed parameter (cell 16:377 `CHECKPOINT_SELECTION_SEED`
vs train_base.py:1176 `seed + 1000`), `read/record_stage_duration` (D-A15;
recipes.md:454 "absent on CLI runs" becomes false), a flag to skip
HPT/post-panel reporting, and a `(model, final_path, model_dir, stage_dir)`
return; cell 16 `train_stage` (201-610, whose comments say "mirrors
train_base.train()" at 385, 486, 536) becomes a ~30-line wrapper with the same
signature so cells 22/24/26 are unchanged; the ~10 AST pins at
test_sb3_notebook_pins.py:762-871 move to test_train_base/test_train_wiring.
`evaluate_stage_checkpoints` (cell 16:613-856) moves beside
`generate_stage_artifacts` with its globals as parameters. (b) Run the resolve
cell (10, depends only on SPECIES) before the storage cell and read the widen
parent's `run.seed` (cell 11:33-85) before `initialize_result_bundle`, so a
wrong SEED refuses with nothing on disk; delete the five copies of the "restart
the runtime (or `del _ACTIVE_RUN_ID`) and delete the stray directory" remedy
(cell 11:88-93, cell 25:30-34, docs/KNOWN_ISSUES.md:160-166, plan:1038,
docs/investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md:63); re-target pins
:1659-1692. (c) `RUN_ID = ""` becomes a knob in cell 6 (cell 25:16-18 already
tells the operator to set it; today it is hardcoded at cell 8:30); the
`_ACTIVE_RUN_ID` memo stays (optional removal, D-D7). (d)
`disconnect_runtime(reason, *, in_colab, auto, flush_drive)` with explicit
parameters and one `halt(reason)` for the three disconnect-before-raise sites
(cell 16:104, 22:320, 22:327); `display_stage_videos` on IPython Video (drop the
`_HAS_MEDIAPY` probe at 16:112-118 only; mediapy stays installed for
`record_stage_video`). Cell 12 (random baseline, 26 lines) deleted; cell 14's
table (70-137) folds into `zero_action_baseline.report()`.
Folds in: the CLI/notebook drift finding (corrected), phase 2 of the
notebook-depth finding (the verbatim move of cells 11/22/24/26 into a package
module is the D-D7 question deferred until after this PR), the `RUN_ID` knob,
the disconnect-path finding, and the widen-seed-first finding ((b) standalone,
(a) optional).
Breaks: pins re-pointed; the Colab hot-edit of `train_stage`'s body is lost (a
change now needs a pushed `REPO_REF`). No on-disk format change; saved runs, the
r11 parents and 20260914_123816 unaffected; every live knob keeps its name.
Validation: notebook parse and pins,
`test_compsognathus_training.py::test_actual_notebook_training_stance_and_recovery_reports`
(all four params once), shared suite. Prerequisites: PR-4 (stamp block gone),
PR-12 (guards gone); D-D7 (taken: wrapper now); D-D11 (confirmed 2026-09-20).
Under D-D14 the widen cell and its two knobs leave the notebook in this PR
when sessions 1 and 2 of NEXT_STEPS.md are both decided by then, otherwise in
a PR right after it.

### PR-15. Docs fold, CHANGELOG Changed/Removed, test helpers and pin budget (M, about -290)
Goal: docs/README.md gains the operator guide under Living reference;
website/docs/training/recipes.md: replace the 2026-09-19 pilot paragraph with
the landed node description and fix the Leaf row (still "today behavior
(hunt)", :27); docs/RESULT_BUNDLES.md
canonical layout stays as is (no `certified_inputs/` after PR-4); CHANGELOG `###
Changed` / `### Removed` filled from PR-4..PR-14; one
`environments/shared/tests/notebook_cells.py` (`code_cells()`, `cell(marker)`)
replacing the four extractor copies (test_sb3_notebook_pins.py:75-97,
test_behavior_notebook.py:30-38 if still present,
test_compsognathus_training.py:386-392); layout literals that survive become
named constants next to `result_bundle.constants.ANCESTORS_DIRNAME`; the exact
PPO update tuple and `mesozoic_behavior_stage_start` pins are replaced by the
invariants they encode. Keep the D-C17 `DEFAULT_MAX_REVISION_GAP == 1` pin and
the empty-default pins for TRUNK_FROM/WIDEN_FROM/RETRAIN_FROM. This document is
then marked complete in the docs index.
Folds in: the remainder of the documentation findings, the CHANGELOG
`Changed`/`Removed` half, the test-helper unification and the pin budget.
Breaks: nothing at runtime. Validation: workflow run. Prerequisites: PR-14.

Running totals (net, using the corrected figures; moves between files count
zero): PR-1 0; PR-2 +100; PR-3 +20; PR-4 -2,400; PR-5 -1,400; PR-6 -275; PR-7
-600; PR-8 -170; PR-9 -150; PR-10 +180; PR-11 +370; PR-12 -5,300; PR-13 -400;
PR-14 -450; PR-15 -290. Net about -10,750 from 723f58f; stated as about 10,500
with a plausible band of 9,000-12,000 (D-D5's TOML form is worth ~700 either
way; D-D7 could remove ~700 more notebook lines while adding them to the
package). From 22c1fc8, with PR-1 landed and #543's ~1,200 lines added, about
9,500.

## 4. Notebook target

Before: 40 cells, 2,500 lines (2,294 code, 206 markdown; 22 code cells; 12 whole
cells and 2 partial cells behind `COMMAND_TERRAIN_BEHAVIOR`). After PR-12 and
PR-14 (the taken form under D-D7: chain loop and widen/resume logic still in
cells as the plan §4.7 pins them): 33 cells, about 1,400 lines. Under the full
package move that D-D7 defers until after PR-14, the same notebook is about 700
lines in 19 sections.

| # | Title | Type | ~lines | Source cells | What moves into the package |
|---|---|---|---|---|---|
| 1 | Title and intro | md | 13 | 0-2 | - |
| 2 | Setup and install | code | 59 | 3 | stale comment 3:7 removed; mediapy stays |
| 3 | Imports and repo root | code | 39 | 4 | - |
| 4 | Configuration | md | 12 | 5 (minus 13-35) | library and behavior prose to the operator guide |
| 5 | Knobs | code | 30 | 6 | drops `SOURCE_SELECTION`, `CERTIFIED_LIBRARY_ROOT`, `PUBLISH_CERTIFIED`, `CERTIFIED_COMPARISON_EPISODES`, ten `BEHAVIOR_*`, `COMMAND_TERRAIN_BEHAVIOR`; gains `RUN_ID` |
| 6 | Resolve chain and table | code | 45 | 10 | moved ahead of storage (widen-seed-first) |
| 7 | Storage, provenance, trunk | code | 70 | 8 | `CERTIFIED_LIBRARY` line gone; parent-seed read before `initialize_result_bundle`; `resolve_parent_run` helper shared with widen |
| 8 | Widen | code | 100 | 11 | seed check gone (done in 7); refusals kept verbatim |
| 9 | Explore + zero-action baseline | md+code | 1+65+40 | 9, 13, 14 | table/report into `zero_action_baseline.report()`; cell 12 deleted |
| 10 | Training infrastructure | code | 250 | 16 | `train_stage` -> 30-line wrapper over `train_base.train`; `evaluate_stage_checkpoints` -> reporting/stage_artifacts; stamp block and mediapy probe deleted; `disconnect_runtime` explicit parameters |
| 11 | Visualization | code | 46 | 18 | - |
| 12 | Chain loop | md+code | 10+270 | 21, 22 | guard, library lookup and publish block deleted (-70); loop stays AST-pinned |
| 13 | Manual single node | md+code | 3+98 | 23, 24 | guard removed |
| 14 | Resume interrupted node | md+code | 45+106 | 25, 26 | remedy prose deleted; guard removed |
| 15 | Evaluate | code | 1+20 | 27, 28 | guard removed |
| 16 | Training curves | code | 1+12 | 29, 30 | guard removed |
| 17 | Replay videos | md+code | 5+11 | 31, 32 | guard removed; IPython Video |
| 18 | Cleanup | code | 1+31 | 35, 36 | guard removed |
| 19 | Auto-disconnect | md+code | 3+2 | 37, 38 | one cell |

Deleted outright: cells 7, 12, 19, 20, 33, 34, 39 and all 14 guard sites. Every
knob the maintainer uses keeps its name and meaning; `BEHAVIOR` gains `follow`
and `terrain`.

## 5. What stays as-is and why

Keep verdicts (each checked against the code during the review):
- The two model/data pools and terrain outside the plant identity. A heightfield
  model is a different plant by construction (physics digest hashes `geom_type`,
  `geom_dataid`, hfield data; plant_contract/physics_layer.py:133-139, 240-246),
  the plane and a flat hfield differ in narrow-phase contact (5/5 vs 0/5 in
  #540), and `build_terrain_model` is already the single injection point.
- `BehaviorReplayRecorder` and the terrain maps stay a distinct exporter;
  `record_stage_video` rolls a fresh episode at `replay_seed` with camera
  presets and the stance CSV the r11/r13 reviews depend on.
- widen_checkpoint, its identity gate (gap >= 1, D-C17), `WIDEN_LINEAGE_KEYS`
  and the reseed plumbing are the single mechanism the maintainer's r11 -> r13
  sessions use; only the probe constants are shared.
- The chain loop is not `train_curriculum` (no JUDGE branch, no frozen-null
  recovery flow, no `generate_stage_artifacts`, no library publication in the
  CLI); the behaviors runner cannot be a branch of the loop today; the
  widen/resume/preflight guards encode real failures (run 20260821_142144; the
  PPO.load segfault).
- direction_commands.py and terrain.py are the genuinely new content; do not
  merge the controller into command_frame.py (amendment A3 keeps command_frame
  numpy-only).
- test_terrain.py, test_terrain_sampling.py, test_direction_commands.py, the
  widen golden pins, the AST notebook pins and (until PR-5) the
  library-immutability tests are distinct; they relocate out of the SB3 lists
  only.

Proposal parts dropped or rejected during the review (no proposal was refuted as
a whole):
- Moving `body_frame_velocity`/`tracking_metrics`/`gaussian_tracking_reward`
  into reward_functions.py deletes nothing; dropped.
- Deleting `_settle_root_on_ground` and the `lowest_ground_clearance` raise;
  rejected (different algorithm; hfield guard).
- A `starting_updates` / `recommend()` pointer in place of the library: a new
  mechanism; dropped under D-D4 (pure deletion, with `select_trunk` as the
  replacement).
- A `[curriculum]` block in all 66 TOMLs (+800 lines) and the name
  `command_tracking/v1` for a gate that does not implement the plan's statistic;
  rejected.
- A shared rollout generator: nets ~0 lines and touches the canonical `evaluate`
  CLI; dropped.
- `train_curriculum --target follow_direction`: refused by
  `_resolve_curriculum_target` for non-advancing nodes; dropped from the target.
- The 11-template + 6-profile dialect and the generator script; superseded by
  stage TOMLs under D-D1.
- Terrain in `BaseDinoEnv.reset()`; rejected (reset source is fingerprinted; it
  would be an r14 bump for all six species; D-D10).
- A `--zero-command-columns` widen mode; replaced by the `policy_loading`
  primitive (widen's gate refuses same-width parents).
- Deleting the replay recorder; rejected (keep verdict above).
- An interim `sb3_behaviors.ipynb`; not built (D-D8): the mode is deleted in
  PR-12 and the split would be maintained for weeks then thrown away.
- Removing the `_ACTIVE_RUN_ID` memo; optional
  (test_sb3_notebook_certified.py:78-81 pins the rerun behaviour and
  jax_training.ipynb keeps the same memo).
- Two of the CI-length items: already the case / not a large contributor.
- The CHANGELOG `Added`/`Migration` block is an addition (+70), kept for its
  content, not counted as removal.

## 6. Decisions (the D-D series, taken 2026-09-17)

Recorded in [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2 under the
same ids. Ordered by how many PRs each blocks, as the review posed them. "Taken"
means decided by the maintainer on 2026-09-17; the last two of that day (D-D11,
D-D12) were recommended then and confirmed on 2026-09-20, when D-D13 and D-D14
were taken.

| Id | Question | Decision | Unblocks |
|---|---|---|---|
| D-D1 | Are direction-following and terrain traversal ordinary manifest nodes under `locomotion` (the plan's Phase D), or a standalone pipeline that only dedupes inside itself? | Taken: manifest nodes, stage TOMLs, gate kinds, ancestors reuse. Consequence accepted: a follow/terrain node needs a certified locomotion ancestor at the Phase C interface (r13 for trex; each other species at its own current revision). Trex has one since the Drive survey (20260914_123816); the other five species get theirs from the NEXT_STEPS.md walker sessions; the manual cell is the escape hatch until then. | PR-9 to PR-15 |
| D-D2 | Fill the six reserved `command_*` kwargs literally (D-C3) or replace the five numeric ones with one `command_config: DirectionCommandConfig \| None`? | Taken: replace. The five (`command_speed_range`, `command_lateral_range`, `command_yaw_rate_max`, `command_switch_interval`, `command_switch_jitter`) have no reader anywhere and the dataclass is already validated and JSON-able; the task-fingerprint carve-out is extended so no canonical `task_sha256` moves while `command_mode` is `"none"`. Amends D-C3. | PR-9 |
| D-D3 | Command-slice normalisation: the plan's invariant 8 (reseed to mean 0 / var 1, statistics keep updating) or the pilots' exact passthrough (`BehaviorVecNormalize`, a pickled class in every sidecar)? | Taken: reseed. The passthrough subclass is deleted; #540/#541-trained policies saw different inputs and are not continuations. | PR-8, PR-10 |
| D-D4 | Keep automatic parent selection when the library goes? | Taken: keep it, as `ancestors.select_trunk` (D-A25, landed as #543): scans the runs under `logs/<species>/<algo>/`, applies the seven reuse rules root-first and picks the run covering the most of the chain; `TRUNK_FROM = "auto"` is the notebook default and `--trunk-from auto` the CLI form. It landed before PR-4/PR-5 delete the library trio, so the capability never lapses. The library is deleted outright; no recommendation pointer survives. | PR-4, PR-5 |
| D-D5 | Stage TOML form for the new nodes: a ~20-line `extends` key in `load_stage_config` (small files, new mechanism) or self-contained `[env]` blocks like configs/trex/behavior.toml (no new mechanism, ~30 restated keys per file)? And the node set: 3-4 per species or all 11? | Taken: `extends`, with four base stage TOMLs per species (`follow_direction`, `follow_direction_difficult_terrain`, `difficult_terrain`, and the existing `locomotion` as the `extends` parent); the terrain templates become a `terrain_families` list inside `difficult_terrain`; single-template runs are documented `[env]` overrides, not files. Refined by G1: three new files per species, `follow_direction_speed` folded into `follow_direction`. | PR-11, PR-12 |
| D-D6 | Gate: implement the plan's `command_tracking/v1` (per-event settle/dwell, heading-bin floor, paired null against the command-blind walker) or register the certificate's per-episode statistic under an honest name? | Taken: honest name first. Pilots run `none/v1` (recorded, not enforced); then the certificate's per-episode statistic is registered as `terrain_command/v1` with one shared threshold block (first values from G4); the plan's `command_tracking/v1` with the paired null is a later second kind, new work outside this sequence. | PR-13 |
| D-D7 | Notebook depth: only the `train_stage` wrapper (the plan §4.7 AST pins on the chain loop stay) or also move cells 11/22/24/26 verbatim into a package module behind a `NotebookSession` (rewrites most of the 2,125-line pin file and plan §4.7:882)? | Taken: wrapper now (PR-14); whether to move the chain loop / widen / resume cells into a package module is decided after PR-14 has settled. | PR-14's scope |
| D-D8 | Build an interim behaviors notebook now, or tolerate the mode switch until PR-12? | Taken: tolerate. #542 removed the dangerous default; the switch is deleted in PR-12. | PR-12 |
| D-D9 | Are any #540/#541 behavior bundles on Drive worth carrying forward? Their identity hashes environments/shared/behavior_env.py itself, so exact resume already breaks on any edit; #540 calls them pilots. | Taken: none. The bundles are evaluation-only; no bundle is carried forward as a training parent. | PR-6, PR-7, PR-9, PR-12 acceptance |
| D-D10 | Terrain in the env: one generic opt-in subclass, or an r14 interface bump that puts the model swap into `reset()`, batched with the queued height-channel removal (plan:668-673)? | Taken: opt-in subclass; no r14 bump (the reset source is fingerprinted). | PR-7, PR-9 |
| D-D11 | May CLI runs record stage duration and seed model construction like the notebook does? | Confirmed 2026-09-20: yes (PR-14 aligns `train()` with the notebook's `alg_kwargs["seed"]` line and its duration recording). | PR-14 |
| D-D12 | Drop the dead `lateral_speed_scale` field (always divides a zero) when the TOMLs are rewritten? | Confirmed 2026-09-20: drop in PR-11/PR-12 (PR-8 item (d)). | PR-11, PR-12 |
| D-D14 | What happens to the widen path (`WIDEN_FROM`, `WIDEN_MAX_REVISION_GAP`, the widen cell, `widen_checkpoint`) after the two pending parents are widened? | Taken 2026-09-20: keep it for NEXT_STEPS.md sessions 1 and 2, then CLI-only — the notebook refactor (PR-14, or a PR right after it once both sessions are decided) deletes the widen cell and both knobs; `widen_checkpoint` stays a command-line tool for the next interface bump. Amends the §4 "knobs kept" list. | PR-14 or its follow-up |
| D-D13 | In which order do PR-3 .. PR-15 land now that the hold is lifted? | Taken 2026-09-20: notebook-first. PR-3, PR-4, PR-5, PR-6, then a notebook-only slice of PR-12 (the `COMMAND_TERRAIN_BEHAVIOR` switch, the ten `BEHAVIOR_*` knobs, cells 7/19/20/33/34/39 and the guard sites, `behavior_notebook.py` with its tests and pins; `train_behaviors.py` stays a CLI-only path) pulled ahead of PR-11, then PR-14, then PR-7 .. PR-11, the rest of PR-12, PR-13, PR-15. Amends D-D8: the switch is tolerated only until that slice, and the direction/terrain pilots have no notebook path between the slice and PR-11 (evaluation-only under D-D9). | the whole sequence |

## 7. Goal decisions G1–G4 (taken 2026-09-17)

The target behavior is: every species follows a direction on difficult terrain.
These four decisions fix what that means for the manifest and the gate; they are
recorded in BEHAVIOR_RECIPES_PLAN.md §6.2 beside the D-D series.

| Id | Decision | What it changes |
|---|---|---|
| G1 | Chain shape `walk -> follow_direction (full command set on flat ground) -> follow_direction_difficult_terrain` (the target deliverable); a commands-free `difficult_terrain` node stays as an optional diagnostic sibling; the recipe label `follow` resolves to the deepest deliverable; `follow_direction_speed` is folded into `follow_direction`; the final node is SB3-only (MJX fails closed on live commands and has no terrain). | PR-11's node set becomes `follow_direction` and `follow_direction_difficult_terrain` (deliverables, `recipe = "follow"`) plus `difficult_terrain` (optional, `recipe = "terrain"`): three new stage files per species, not four; no `follow_direction_speed.toml`. All three new files carry `extends = "locomotion"` (D-D5); only `warm_start_from` differs (`locomotion` / `follow_direction`). The §2 table row for stage configuration reads accordingly. |
| G2 | Command set = heading, speed (half to full cruise), stops and restarts, switching every few seconds: the pilots' combined recipe. | PR-11's `follow_direction` carries the full command set from the start (the assessment had listed a heading-only first node and a separate speed node): `command_config` with `speed_range = [0.5, 1.0]` of `cruise_speed`, stops and restarts, and a switch interval of a few seconds; `follow_direction_difficult_terrain` inherits it. |
| G3 | Walker sessions start now on the current notebook, trex first (its r13 chain 20260914_123816 is selected automatically), the other five species one at a time, in parallel with the consolidation PRs. | Nothing in the PR order changes; it fixes the constraint every PR's "Breaks" line already honours: the notebook chain loop, `TRUNK_FROM = "auto"`, `WIDEN_FROM`, `RETRAIN_FROM` and the resume cell keep working at every step, because the sessions in NEXT_STEPS.md run on whatever `main` is at the time. Those sessions are what give PR-11's nodes their certified locomotion ancestors on the five species that lack one. |
| G4 | The first `terrain_command/v1` gate adopts the pilots' certificate thresholds: 20 episodes per terrain family, 20 s minimum horizon, survival LCB 0.80, success LCB 0.60, tracking and settle fractions 0.60; tightened after the first certified species. | PR-13's `[curriculum]` threshold block is fixed at these values, carried once (never 66 copies) from configs/behavior_certification.toml before that file is deleted; tightening later is a gate-digest change (rule 7), so a node certified under the first thresholds is re-judged, not silently reused, after the change. |

## 8. Risks and open questions

Carried from the review, with the 2026-09-17 additions.

- The stop-gap in PR-1 (#542) is the only change that protected the maintainer's
  next Colab session; everything else waited for the hold, which lifted on
  2026-09-20. Since #543
  canonical chains never consult the library; `SOURCE_SELECTION = "auto"`
  applies only to the direction/terrain path (`train_behaviors --auto-source`,
  deleted in PR-5), which would copy a library version into the pilot bundle's
  `certified_inputs/` (certified_library.py:613; that tree writes bundle.json,
  never `artifact_manifest.json`); the survey did not inspect the library
  directory and nothing relies on one, so that half is dormant today. What
  canonical chains still do until PR-4 is copy a cross-run trunk ancestor's
  bundle into `certified_inputs/` through `copy_canonical_ancestor` (cell 22;
  certified_canonical.py:882), and `artifact_manifest.json` hashes those copies
  into every bundle write (manifest.py:89, 158).
- Does a widened stance reproduce its panel under r13? Unanswered: no widen
  session has run yet. NEXT_STEPS.md sessions 1 (trex seed 44, gap 2) and 2
  (compsognathus seed 42, gap 1) answer it; if the widened seed-44 panel fails
  the duty rail as seed 43 did (0.0323 / UCB 0.0350 against 0.02), the fallback
  is a fresh trex stance with SEED = 45.
- The `ConstantSchedule` `custom_objects` guard in `_load_ppo`
  (behavior_checkpoint.py:108-121, for cloudpickled py3.13 schedule bytecode) is
  deleted with PR-12. Settled 2026-09-19: the canonical `alg_cls.load` path had
  never crossed an interpreter boundary, the first cross-interpreter load (the
  widen tool's self-verification of the r11 parent on the Python 3.13 image)
  killed the kernel, and every load now goes through
  `policy_loading.load_sb3_model`, which makes the guard redundant (KNOWN_ISSUES,
  "SB3 archives are bound to the interpreter that saved them").
- Not verified during the review (flagged, unverified): (i) the two dated
  task-fingerprint valves (`allow_unfingerprinted=True`, train_base.py:593-604;
  the schema-v1 valve in `validate_recorded_task`) may already be dead because
  the plant check runs first on r13 species; deleting them is a ~60-line
  follow-up after both r11 parents are widened; (ii) `EpisodeManifestRecorder`
  (train_behaviors.py:165-176) becomes an info key for the existing
  Monitor/DiagnosticsCallback under train_base, so per-episode terrain manifests
  must be checked to survive PR-12; (iii) `canonical_env_parameters`
  (behavior_env.py:44-55) duplicates `_effective_env_kwargs`; dies with PR-9 but
  its use for the allowed-`[env]`-keys check in `read_recipe` must be re-homed
  in `load_stage_config` (PR-9/PR-11); (iv) the four provenance records of
  seed/parent facts (run block `LOAD_LINEAGE_KEYS`, task lineage,
  `mesozoic_canonical_training`, the four `mesozoic_behavior_training_*` stamps)
  collapse to the first two after PR-4/PR-12, but the task lineage may need
  `parent_normalization_sha256` added (~15 lines) once the pilot stamps go.
- Distribution change in PR-8: single-template recipes move from a Bernoulli
  `flat_probability` draw to balanced blocks; acceptable for pilots
  (evaluation-only, D-D9), stated here as the decision record requires.
- PR-9 touches five species constructors, `MJXEnvConfig` and the fingerprint
  carve-out; the acceptance test is that no committed `task_sha256` moves
  (test_phase_c_interface.py:348) and that `plant_contract --check` reports no
  interface change, run on every species, not only trex.
- MJX reward kernels stay world-z after PR-7 (MJX has no terrain and fails
  closed on live command modes); note the divergence in mjx_env's comment block.
  Under G1 the final node is SB3-only for the same reason.
- CI coverage `fail_under=70` after PR-3 and after the large deletions in
  PR-12/PR-13 will need re-measuring; the excluded trainings mostly cover code
  that is deleted.
- Test-to-test coupling must be untangled in order:
  environments/trex/tests/test_behavior_training.py imports `CommandEnv` from
  test_behavior_checkpoint.py (:384); test_behavior_publication.py imports
  `_identity`/`_report` from test_behavior_certification.py (:14).
- jax_training.ipynb carries the same Drive-mount block and `_ACTIVE_RUN_ID`
  memo (:250, :296); if the D-D7 package move or the memo removal is taken, take
  it for both notebooks so the two Colab drivers do not diverge on the same
  footgun.
- The plan's per-event `command_tracking/v1` statistic and the
  worst-of-heading-bins floor have no implementation anywhere yet; D-D6's second
  kind is new work outside this consolidation.
- Uncertainty on the totals: the largest single figure (PR-12, about -5,300)
  depends on D-D5 (self-contained TOMLs would add ~700 lines back) and on how
  much of test_behavior_species_training.py survives as the real-PPO smoke; the
  band 9,000-12,000 reflects that.
- Velociraptor:
  [investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md](investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md)
  (2026-07-20) left a pending Commit B validation; the fresh r10 stance run in
  NEXT_STEPS.md session 3 is the first data since.
- The r13 trex run's run-level summary.json, provenance.json and
  artifact_manifest.json predate its locomotion verdict (written 2026-09-15
  02:08 UTC, verdict 11:39 UTC); reuse reads per-node files so nothing breaks,
  but the run-level records under-report the run until the bundle cell is
  re-run. PR-14's single storage path does not change this; it is an operator
  step.
- Phase B items the maintainer deferred on 2026-09-13 stay deferred
  (BEHAVIOR_RECIPES_PLAN.md §10); nothing in this sequence reopens them.
