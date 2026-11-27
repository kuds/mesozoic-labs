# Plant contract

The plant contract gives each model a reviewable identity that is more precise than a single model-file hash. A plant is
keyed by its model, the pair (species, embodiment) of [SPECIES_NAMING.md](SPECIES_NAMING.md): its revision counters are
the `[plants.<species>.<embodiment>]` table of `configs/plant_versions.toml` (schema 2), the manifest holds it at
`plants.<species>.<embodiment>` (`mesozoic.plant-manifest/v2`), and its identity record (`mesozoic.plant-identity/v2`)
carries `species` and `embodiment` as two fields, both compared. The committed output is
`configs/plant_manifest.generated.json`; it is generated from the species manifest, MJCF assets, executable
environments, and the human revision counters in `configs/plant_versions.toml`.

The public species catalog reads the committed manifest. It does not recompile fingerprints during a website build.

## Layers

| Layer | Covers | Typical change |
|---|---|---|
| Source closure | Exact root MJCF and recursively referenced asset bytes | Include, mesh, texture, or XML edit |
| Policy interface | Ordered observations, executable observation/action mappings, sensors, actions, actuator ranges, dtype, and control period | Sensor reorder, cached body-ID change, backend mapping change, or control-range change |
| Physics | Compiled topology, inertias, joints, contacts, actuators, solver options, and reset state | Mass, friction, gain, keyframe, or collision change |
| Visual | Render geometry, materials, textures, cameras, and lights | Color, material, or camera change |

Layers can overlap. Changing the size of a visible collision geom can change both physics and visual fingerprints. A
source-only change, such as an XML comment, changes the source closure without changing a semantic layer.

## Revision rules

Each semantic layer has an independent positive revision: `policy_interface_revision`, `physics_revision`, and
`visual_revision`.

- If a semantic fingerprint changes, increase that layer's revision in `configs/plant_versions.toml`.
- If several fingerprints change, increase every affected revision.
- Do not bump a semantic revision for a source-only change.
- A visual-only change does not make a policy incompatible.
- A policy-interface or physics change makes an existing policy incompatible, even when dimensions happen to match.
- A change of a plant's identity key (a renamed species or embodiment) carries its counters forward: the generator pairs
  the old plant with the new key by physics digest (below), and the layer rules apply to the pair. Moving or renaming a
  method the policy layer hashes changes its recorded qualname, which is a policy-interface change and needs that bump
  even when no token moved; decision D-D33 is the example (`configs/plant_versions.toml` note 19).
- A plant leaves the manifest only by an explicit decision: the generator refuses a baseline plant it cannot pair.

The biped home-residual changes are examples where tensor dimensions can stay
fixed while action meaning changes, requiring a policy-interface revision and
fresh checkpoints. See the
[Velociraptor Stage-1 basin investigation](investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md)
and [T-Rex home-equilibrium investigation](investigations/TREX_HOME_EQUILIBRIUM.md).

The generator compares the existing committed manifest with the new one and refuses to write a changed semantic layer
unless its revision increased. It pairs the baseline's plants with the current ones by identity key first, then pairs
the plants left over on both sides by their physics digest (refusing a baseline leftover's physics digest that is shared
by more than one leftover on either side), and refuses any baseline plant still unpaired; a current plant left unpaired
is new. A baseline of `mesozoic.plant-manifest/v1`, keyed by the species alone, is accepted only in this comparison and
only against a v2 manifest (the D-D33 transition), so its six plants pair by physics digest. Pull-request CI also
compares against the base branch, so deleting or replacing the local manifest cannot bypass revision monotonicity.

## Regeneration

The canonical MuJoCo version is recorded in `configs/plant_versions.toml` and pinned exactly in the package metadata.
Use that exact version when writing or checking the manifest. The fingerprint includes all public `MjOption` fields and
compiled physics data covered by the contract, so a different MuJoCo compiler version is not interchangeable.

Fingerprint tool v2 canonicalizes finite numeric values to 12 significant decimal digits. This treats the 1–4 ULP
compiler differences observed across arm64, x86_64, macOS, and Linux as the same plant while providing about `1e-11`
relative resolution. The source-closure layer still hashes MJCF and referenced-asset bytes exactly, so even a
below-resolution source edit remains visible in provenance.

After an intentional model or interface change:

1. Inspect which layers changed and update the affected counters in `configs/plant_versions.toml`.
2. In an environment with the canonical MuJoCo version, regenerate and verify:

   ```bash
   python -m environments.shared.plant_contract --write
   python -m environments.shared.plant_contract --check
   ```

3. Regenerate the public catalog and run the focused tests:

   ```bash
   python -m environments.shared.species_catalog
   pytest environments/shared/tests/test_plant_contract_*.py environments/shared/tests/test_species_catalog.py
   ```

4. Regenerate the digest snapshot (every plant, policy, stage, recovery and behavior digest and each stage's reward,
   info and termination capture, one value per line; D-D22, CU-11) and
   check that its diff moves only the digests you meant to move:

   ```bash
   python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --write
   git diff configs/digest_snapshot.generated.txt
   ```

5. Regenerate the committed mirror map of every model you changed (`configs/<species>/<embodiment>/mirror_map.json`;
   today the anatomical T. rex has one). A map records the plant's physics and policy-interface digests and refuses
   to load for any other, so a stale one stops mirror-symmetric training. Then check them all:

   ```bash
   python -m environments.shared.symmetry --write --species tyrannosaurus_rex --embodiment anatomical
   python -m environments.shared.symmetry --check
   ```

   A T. rex change that moves the statue's legs (their masses, springs, gains or the feet) also re-measures
   `_LEG_SETTLED_QPOS` in `environments/tyrannosaurus_rex/base_env.py`, as its comment describes;
   `test_the_settled_leg_pose_is_the_statues` fails until it is.

The writer updates both `configs/plant_manifest.generated.json` and the byte-identical runtime copy under
`environments/shared/data/`. Commit the version counters, both generated manifests, generated public catalog/README
data, the digest snapshot, the regenerated mirror maps, and the intentional source change together. CI repeats the
check with the canonical MuJoCo version, tests revision monotonicity against the PR base, runs the full digest snapshot
against `configs/digest_snapshot.generated.txt` (a failure names every moved line and prints the `--write` command), and
verifies identity/config loading from an installed wheel. Any other change that moves a digest on purpose (a stage
config, a recovery calibration, a behavior recipe, the environment code it hashes, or the reward, info or termination
code the `reward` lines capture) commits the regenerated snapshot the same way; a change that claims to move no digest
leaves it unchanged. Regenerate it on Linux x86-64 with the canonical MuJoCo and numpy 2, as CI runs it. A refactor that claims to
move no number also diffs the harness's `--exact` output of the base and the head on one machine (see its `--help`):
the `reward` lines are rounded and cannot see a change of a few ulp.

### Widening a checkpoint across a policy-interface bump

A policy-interface revision normally strands every checkpoint minted under the previous one (see "Checkpoints and legacy
artifacts" below). The Phase C revision (BEHAVIOR_RECIPES_PLAN §4.6 "Widening instead of retraining") is the one
exception the repo tools: it appended a 3-dim body-relative command segment at the END of every species' observation
and changed nothing else, so a policy trained one interface-only revision behind is the same function under the new
interface once the first layer of every network that reads the observation gains three zero columns. That mapping is
done by `environments/shared/scripts/widen_checkpoint.py`, never by hand and never by re-labelling:

```bash
python -m environments.shared.scripts.widen_checkpoint --species tyrannosaurus_rex --embodiment anatomical --stage stance \
    --from-stage-dir <parent run>/01_stance --to-stage-dir <new run>/01_stance \
    [--label L] [--parent-run-id ID] [--allow-legacy-plant] [--max-revision-gap N]
# or the explicit pair: --model <zip> --vecnorm <pkl> --algorithm ppo|sac --seed S --n-envs N --timesteps T
```

- **The identity gate.** The parent archive's recorded plant identity must be the current plant at most
  `max_revision_gap` interface-only revisions behind (default 1 — exactly r → r+1, the Phase C bump alone;
  `--max-revision-gap N` on the CLI, `max_revision_gap=N` in the API; decision D-C17): same `species` and
  `embodiment`, `physics_sha256`, `nq` / `nv` / `nu` and `action_dim`,
  `1 <= current - parent.policy_interface_revision <= max_revision_gap` and `observation_dim + 3 == current`.
  Anything else is refused with every differing field named; a parent further behind than the bound is refused with
  both revisions, the measured gap, the bound and the flag named. The fields other than the revision apply whatever
  the bound, so the tool is scoped to the Phase C bump — three appended observation dims with the physics and the
  action mapping untouched — and a bound above 1 can only cross fingerprint-only intermediate revisions (there is
  no ladder of intermediate manifests to walk: the widening is applied once, against the checkout's manifest).
  Setting N > 1 asserts, from `configs/plant_versions.toml`'s numbered notes, that the intermediate bumps changed
  nothing the widening cannot bridge; the report records `revision_gap` / `max_revision_gap`. The two certified
  T-Rex stance runs of 2026-08 are r11 archives, two revisions behind r13 across the fingerprint-only r11 → r12 bump
  (note 11, the perturbation engine): refused under the default, widened under `--max-revision-gap 2`
  (`docs/KNOWN_ISSUES.md`, the Phase C entry). A bump that changed action meaning at fixed dimensions (the biped
  home-residual examples above) is not widenable and still needs fresh checkpoints. A parent without an identity is refused unless
  `--allow-legacy-plant`, which reads its width from the saved observation space. The VecNormalize sidecar must
  record the parent archive's plant; under the legacy allowance an unstamped sidecar is accepted with a warning,
  and a stamped sidecar of a legacy (unstamped) archive must record the parent's species, embodiment and width.
- **What is written.** Into a fresh stage directory named as the stage's directory (`stage_dir_candidates`), outside
  the parent's run: `models/<handoff>.zip` + `models/<handoff>_vecnorm.pkl` under the parent's own handoff name
  (`robust_best_model` or `best_model`, exactly one), byte-identical `models/<stage_label>_final.*` copies (what the
  SB3 notebook's JUDGE branch fires on), `stage_config.json` whose run block carries the parent's `seed` / `n_envs`
  / `timesteps` (and `duration_seconds` when recorded), `hyperparameters_sha256`, the optional `label` and the eight
  `config.WIDEN_LINEAGE_KEYS` (`widened_from_path`, `widened_from_checkpoint_sha256`,
  `widened_from_normalization_sha256`, `widened_from_task_sha256`, `widened_from_policy_interface_sha256`,
  `widened_from_policy_interface_revision`, `widened_from_run_id`, `widened_by`) — never the `LOAD_LINEAGE_KEYS`, a
  widened node is a root — plus `plant_identity.json`, `task_fingerprint.json` and `widen_report.json`. Both
  artifacts are re-stamped with the CURRENT plant identity and the stage's CURRENT task fingerprint (the parent's
  `mesozoic_task_lineage` is kept), and the archive gains a `mesozoic_widen_lineage` attribute recording the parent
  hashes, the parent identity, the padded tensors and the commit.
- **What is not written.** No `gate_verdict.json`, `provenance.json`, `gate_resolution.json`, `evaluations.npz`,
  `metrics.json` or periodic checkpoints: a widened checkpoint carries no certificate and is re-paneled under the
  current gate before it certifies. It is refused into an occupied or non-empty target, or into a run whose
  result bundle is `complete` (or whose manifest cannot be read), and a parent whose `gate_verdict.json` did not
  pass is refused.
- **The reseed rule.** The sidecar's `obs_rms` gains the three dims at mean 0 / variance 1 with the count carried
  (`command_frame.pad_running_stats`) — the values `load_vecnorm_stats(reseed_command_slice=True)` applies to the
  trailing slice on any load into a node whose `command_mode != "none"`. Under `"none"` the command is zero, so the
  widened statistics normalise it to exactly zero and the zero columns see exactly zero input.
- **Self-verification.** Before returning, the tool checks that every padded column is exactly zero, that both
  widened artifacts validate against the current identity through the ordinary loaders, and that over a seeded
  200-step rollout of the real environment the widened policy's deterministic actions match the parent's within
  `1e-6` both with the command slice zero and with `COMMAND_PROBE_VECTOR` in it (the measured deltas are recorded in
  `widen_report.json`); the files' hashes are unchanged by the verification. Any failure deletes the target directory.

This command-line tool is the only widen path (decision D-D14 removed the SB3 notebook's widen cell and knobs). Widen
into a NEW run id, one no run uses and the notebook has not opened yet, written as a timestamp `YYYYMMDD_HHMMSS` like
the ids the storage cell mints (`TRUNK_FROM = "auto"` breaks a coverage tie by the greatest directory name, so an id in
another format would outrank every later run), with
`--to-stage-dir <LOG_BASE>/<species>/<embodiment>/<algo>/<new run id>/<stage_dirname(model_id, root)>` and `--label` when the
session sets `RUN_LABEL`. On Colab: (1) run the notebook's section 1, then a scratch cell
`from google.colab import drive; drive.mount("/content/drive")`, never the storage cell, which would mint a run
directory and its provenance; (2) `LOG_BASE` is `/content/drive/MyDrive/mesozoic-labs/logs`; (3) run
`!cd /content/mesozoic-labs && python -m environments.shared.scripts.widen_checkpoint ...` (the tool's module docstring
lists the same steps). Then run the notebook with `RUN_ID` set to that id, `SEED` to the parent's recorded
`run.seed` and `TRUNK_FROM = ""`, and its chain loop judges the widened root (BEHAVIOR_RECIPES_PLAN §4.6, decision
D-C13). The storage cell refuses any other `SEED` before it writes anything (D-C14), and the chain loop judges the
widened root before it consults any trunk (D-C13 as amended by cleanup ROW-4/6's decision 6 (b)).
`docs/KNOWN_ISSUES.md` lists the pre-Phase-C checkpoints this applies to.

## Training backend and runtime binding

The policy fingerprint includes normalized executable code plus portable, quantized synthetic observation probes.
Stable-Baselines3 is the only training, evaluation and evidence backend (decision D-D17), and every model declares it
alone: `supported_training_backends = ("stable-baselines3",)` on the environment and `training_backends =
["stable-baselines3"]` on its embodiment table in `configs/species_manifest.toml`, which must agree. The plant contract,
the species-manifest reader and the command frame refuse any other value; the plant contract's refusal names a
declared `jax-mjx`. Readers of archived result bundles still accept `jax-mjx` records.
The policy payload records the declaration (`interface_implementations.supported_training_backends`) and hashes the
environment's `_cache_ids` tokens (`interface_implementations.sb3_sensor_cache`), so a token edit there, reward-target
setup included, is a policy-interface revision. Since the Phase C interface revision (BEHAVIOR_RECIPES_PLAN §4.6) the
observation probe injects the non-zero `COMMAND_PROBE_VECTOR = (0.25, -0.5, 0.75)` into the trailing 3-dim command
segment (it sets `env._command` beside the model/data swap), so a dropped or mis-ordered command slot hashes
differently from three zeros.

The JAX/MJX runtime was retired by D-D17, but a frozen MJX interface core (`mjx_env.py`, `jax_setup.py`, `mjx_utils.py`
and `obs_functions.py`, with one `mjx_config.py` registration per dual model) stayed while models still hashed its
tokens into their policy-interface digests, with an MJX observation probe whose parity with SB3 the canonical writer
required. Each model left it by declaring itself SB3-only inside a deliberate `policy_interface_revision` bump
([CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §4.1): the velociraptor at its revision 11 (decision D-D25), the
T-Rex at its revision 14 (decision D-D31), and the brachiosaurus and dibothrosuchus, the last two, at their revisions 11
and 10 (`configs/plant_versions.toml` note 23), when the core, both registrations and the backend itself were removed.
The payload still carries the constant `jax_interface = {"supported": false, "backend": "jax-mjx"}` and
`backend_observation_equal = null` that every SB3-only model recorded, so no other model's digest moved with the core.
The T-Rex keeps its action low-pass filter in its interface (`interface_implementations.action_low_pass_filter` hashes
the cutoff and `action_filter.py`'s two functions, whose tokens `test_action_filter.py` pins).

### The heading-free observation schema

`bipedal-target-heading-free/v1` (the anatomical T-Rex from policy interface 14, the anatomical Compsognathus from 4, decision
D-D31, `configs/plant_versions.toml` note 18; the Velociraptor from 13, the maintainer's decision of 2026-11-06, note 21) carries no world heading: projected gravity replaces the root's world
quaternion, the root's linear velocity is expressed in the gravity-aligned heading frame and the target direction in
pelvis coordinates, all built from the free root's own quaternion `qpos[3:7]` by
`environments/shared/heading_observation.py` ([HEADING_INVARIANCE_DESIGN_2026_10.md](HEADING_INVARIANCE_DESIGN_2026_10.md)).
For this schema only, the policy layer records segments that name each block's frame, source and thresholds
(`pelvis_projected_gravity`, `pelvis_heading_linear_velocity` with its forward/left/up components and vertical-pelvis
fallback, `prey_direction_body`) and describe the foot channels by sensor name, one per foot (each a sum over its
touch sensors) or, where the environment declares `_foot_observation_groups`, one per declared group, which must read
exactly the per-foot groups' sensors (the Velociraptor's per-site channels, one per touch sensor, since its policy
interface 14, the maintainer's decision of 2026-11-08, note 22; no other model declares them, so no other segment
moved), and it refuses a payload whose
segment widths do not sum to the observation width or whose last segment is not the command. It hashes the helpers'
function tokens beside `_get_obs` (`interface_implementations.heading_free_observation`), because `_get_obs` only calls
them. The legacy schemas' segment tables are
unchanged, so no other model's digest moved. The synthetic probe's non-unit quaternion is normalized like any other,
so the probe still exercises a real tilt.

Before any artifact is tagged, training validates the environment that actually runs: SB3 validates the concrete
Gymnasium model and observation interface behind the VecEnv (`validate_environment_plant`). This prevents a stale runtime registry, alternate
model path, or modified sensor/body mapping from being mislabeled with the canonical identity.

`current_plant_identity` keeps one build per model for the life of the process (cleanup CU-14b). Every call still
re-reads `configs/`, re-hashes the model's source closure, re-runs the policy-interface layer on a fresh environment and,
unless `verify_generated=False`, compares the result with the committed manifest; what is reused is the declared model's
compilation and the physics and visual fingerprints, seconds per call for the mesh-heavy Compsognathus robot. The
build is reused only while the model's manifest entry, the plant version, the repository root, the resolved model path, the
source-closure digest, the environment class and the MuJoCo version are unchanged and the policy layer still matches, and only a build equal to
the committed manifest entry, hashed from the bytes the key read, is kept, so a stale plant is rebuilt on every call.
The manifest build, `--write` and `--check` never read it. Python code is not part of the key, so a kept build does not
see environment code patched in a live process so that the compiled model changes (not the MJCF). The live checks
compare the physics digest of every environment they are given instead: `validate_environment_plant` (above) for the
environments training's `make_env` builds with a plant identity and those the evaluation paths build, and
`validate_compiled_plant` for the behavior environments and for every environment `config.build_env` builds, which is
how the frozen recovery gate (its nulls, brace and policy panel), the recovery calibration, the zero-action baseline,
`widen_checkpoint` and the report scripts get theirs. `build_env` reads the identity with `verify_generated=False`; the
callers that certify compare it with the committed manifest in their own calls. A process that patches environment code
and builds environments another way calls `clear_plant_identity_cache()`.

## Opt-in per-episode features

The shared env's opt-in per-episode features are task, not plant (design record
[TRICERATOPS_ROBOT_PORT_2026_11.md](TRICERATOPS_ROBOT_PORT_2026_11.md) §4-§6). They live in `step()` and in an
episode-start hook, outside every method the policy layer hashes, so no feature moves a plant identity, on or off.
Two exist, the actuation delay and the per-episode physics randomisation; every model has both off.

- **The hook.** `BaseDinoEnv` derives from `episode_start.EpisodeStartHook`, whose `reset` calls `_begin_episode(seed)`
  right after gymnasium seeds `np_random` and before the keyframe is loaded. `BaseDinoEnv.reset`, hashed as
  `home_reset`, keeps its source, and no digest hashes class bases. Each feature draws from its own PCG64 stream
  (`episode_start.feature_stream`, a spawn key of the episode's seed), never from `np_random`, so the reset draws and
  `environments/shared/tests/fixtures/phase_c_reset_golden.json` are the same with any feature on or off. With every
  feature off the hook reads no generator and writes nothing.
- **The role.** `training_rollouts` is `False` (an evaluation env) unless a trainer marks the env with
  `episode_start.mark_training_rollouts`: `train_base.make_env` does for `create_vec_env(training=True)`, which only
  the stage body's training env passes, and `train_behaviors.main` does for its env, unmarking it before it scores.
  It refuses a vectorised env, whose `unwrapped` has no `set_training_rollouts`: each env is marked before it is
  vectorised. Every other env (checkpoint selection and the curriculum gate, CLI evaluation, the stage reports,
  `config.build_env` and the digest snapshot harness among them) is an evaluation env. The role is not part of the
  plant, the task fingerprint or the mirrored episode state.
- **The actuation delay.** `actuation_delay_range_s` (`None`, off, or `(lo, hi)` in seconds) and
  `actuation_delay_eval_s` (`None` or seconds), in `environments/shared/actuation_delay.py`. Each is a whole number
  of physics steps and shorter than an episode's `max_episode_steps x frame_skip` physics steps; a longer one is
  refused, named in seconds and in steps. A FIFO of the nu-wide ctrl in `step()` hands each control target to
  `data.ctrl` k physics steps late. The FIFO is primed at the first step with `data.ctrl` as reset left it, the reset
  keyframe's `key_ctrl`, which is `_scale_action(0)` under the home-keyframe mappings, so zero action is
  delay-invariant. A trainer's rollout envs draw k per episode, uniform over the whole steps of the range; every other
  env holds the eval value (or draws when it is `None`). The FIFO never writes `env.model`, so the delay never makes
  `validate_environment_plant` or `validate_compiled_plant` refuse (with the physics randomisation on, see below).
  While the delay is on,
  `info` carries `actuation_delay_steps` and `actuation_delay_s`; the reset info is unchanged. The symmetry module's
  state mirror carries the in-flight targets and refuses two envs whose delay configurations differ. The task
  fingerprint records the two keys in its `env` section and, while the delay is on, an `actuation_delay` section
  naming `ACTUATION_DELAY_IMPLEMENTATION`; `resume_same_stage` and behavior transitions name that section when it
  differs. A species that does not list the keys refuses them with `TypeError`; one that lists them declares the
  opt-in carve-out at off value `None` (`OPT_IN_TASK_KWARGS`), so a stage that leaves the delay off keeps its task
  hash.
- **Per-episode physics randomisation.** Six keys in `environments/shared/episode_physics.py`, each `None` (off) or
  `(low, high)`: `randomize_friction` (an absolute sliding coefficient, written to every static floor geom and to the
  geoms `_randomized_friction_geoms()` returns, every colliding animal geom unless the species narrows it),
  `randomize_body_mass_scale` (per animal body, mass and inertia), `randomize_actuator_kp_scale` (per position servo,
  `gainprm[0]` and `biasprm[1]` together), `randomize_actuator_force_limit_scale` (per force-limited actuator, both
  ends of `forcerange`), and `randomize_total_mass_scale` and `randomize_payload_kg`, which apply in a trainer's
  rollout env only (1 and 0 kg in every other env). Each episode begins by restoring the nominal plant and then
  writes one draw of its own stream into `env.model`, before the keyframe is loaded; when a mass changed,
  `mj_setConst` runs on a scratch `MjData`, never the live one, and the geometric statistics go back to nominal. The
  rules:
  - Randomised values live only between a reset and the next. The nominal plant is the identity: the physics digest
    covers every randomised field, so after a randomised reset `validate_environment_plant` and
    `validate_compiled_plant` refuse the live model (fail-closed). Every live site validates before the first reset;
    a diagnostic that needs the nominal plant later wraps its call in `with env.nominal_physics():`, which restores
    the nominal values for the block and writes the episode's back after it, whether the block returns or raises
    (blocks nest). No env is reset inside the block: a reset inside it raises `RuntimeError` before any of it runs
    (before gymnasium reseeds `np_random` or a behaviour env selects a pool model), since the block's exit would write
    the previous episode's values back over the new one; reconfiguring the randomisation inside it is refused alike.
    `plant_contract/validation.py` itself is unchanged and strict.
  - Randomised values are never fingerprinted. The task fingerprint records the six keys in its `env` section and,
    while any is on, an `episode_physics` section naming `EPISODE_PHYSICS_IMPLEMENTATION`; `resume_same_stage` and
    behavior transitions name that section when it differs. A species that lists the keys declares one carve-out per
    key at off value `None`, `(name, None, (name,))`, so a stage that leaves them off keeps its task hash.
  - No managed field (`episode_physics.RANDOMIZED_FIELDS`, `SETCONST_FIELDS` and `model.stat`) may be written after
    construction while randomisation is on. Each model's nominal copies are taken the first time an episode starts on
    it, so a species edit made after `super().__init__()` is part of the nominal plant, and the restore undoes any
    later write. A behavior env restores every pool model before it selects one and re-runs `_cache_ids`, which reads
    masses. A raise while an episode's values are written restores the nominal plant before it propagates.

  Push forces stay the newtons derived from the nominal plant at construction. The payload body's inertia is scaled
  by `(m + p) / m`, which spreads the payload like the body's own mass, so the rule assumes a payload body heavy
  against the payload (the 60 cm Triceratops robot's 6 kg chassis takes up to 1 kg); only a massless one is refused.
  Refused at construction, and again when a model's nominal copies are taken (its first episode, so after any edit
  made following `super().__init__()`): friction on a model with explicit contact pairs or with no static floor geom,
  or whose `_randomized_friction_geoms()` returns no geom, anything but a 1-D array of integer geom ids (a boolean
  mask included), or a geom that is not a colliding animal geom; the kp key without a position servo; the force key
  without a force-limited actuator; and a payload on a body with no mass. A behavior pool model of another `nbody` or
  `nu` than the constructor's is refused at its first episode. The symmetry module's state mirror refuses an env
  that holds a randomised plant: it mirrors the state, not the plant.

## Checkpoints and legacy artifacts

Current plant identity compares policy and physics revisions, hashes, and interface dimensions. Source-closure and
visual differences are retained for provenance but do not block policy replay.

Artifacts without plant identity are legacy artifacts. Contract validation fails closed by default; loading one requires
an explicit `allow_legacy_plant=True` migration/evaluation choice and emits a warning. That override acknowledges missing
provenance—it does not make the artifact current or verified.

The low-level `load_vecnorm_stats` API (`curriculum/checkpoints.py`) also requires the caller to supply
`current_plant`; omitting it is an error, even when `allow_legacy_plant=True`. `unsafe_skip_plant_validation=True`
exists only for deliberate low-level artifact inspection, emits an explicit warning, and must not be used to resume
training or run evaluation.

A checkpoint or statistics of another observation width are refused by their identity, not by SB3's shape checks
(decision D-D31): SB3's `VecNormalize.load` binds the statistics to the env, asserting equal shapes, before the pickled
identity can be read, so every statistics load goes through `policy_loading.load_vecnormalize`, which validates the
pickled identity when that bind fails and raises `PlantCompatibilityError` naming the revision and width; and the
trainer checks the identity an archive's `data` member records before SB3 rebuilds the model around the training env.

SB3 models, VecNormalize statistics, stage configs,
metrics, and run directories all carry the same identity. Promotion validates the embedded model and normalization
payloads, not only an adjacent sidecar.

The checked-in result summaries predate this contract and remain schema-v2, historical, unverified records with nullable
legacy `model_hash` fields; decision D-D33 amended only their identity in place (`species` and `embodiment`). Do not
translate those null values into current layered fingerprints without a reproducible run and evaluation record.
