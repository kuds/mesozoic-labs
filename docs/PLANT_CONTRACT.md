# Plant contract

The plant contract gives each species a reviewable identity that is more precise than a single model-file hash. The
committed output is `configs/plant_manifest.generated.json`; it is generated from the species registry, MJCF assets,
executable environments, and the human revision counters in `configs/plant_versions.toml`.

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

The biped home-residual changes are examples where tensor dimensions can stay
fixed while action meaning changes, requiring a policy-interface revision and
fresh checkpoints. See the
[Velociraptor Stage-1 basin investigation](investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md)
and [T-Rex home-equilibrium investigation](investigations/TREX_HOME_EQUILIBRIUM.md).

The generator compares the existing committed manifest with the new one and refuses to write a changed semantic layer
unless its revision increased. Pull-request CI also compares against the base branch, so deleting or replacing the local
manifest cannot bypass revision monotonicity.

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

The writer updates both `configs/plant_manifest.generated.json` and the byte-identical runtime copy under
`environments/shared/data/`. Commit the version counters, both generated manifests, generated public catalog/README
data, and the intentional source change together. CI repeats the check with the canonical MuJoCo version, tests revision
monotonicity against the PR base, and verifies identity/config loading from an installed wheel.

### Widening a checkpoint across a policy-interface bump

A policy-interface revision normally strands every checkpoint minted under the previous one (see "Checkpoints and legacy
artifacts" below). The Phase C revision (BEHAVIOR_RECIPES_PLAN §4.6 "Widening instead of retraining") is the one
exception the repo tools: it appended a 3-dim body-relative command segment at the END of every species' observation
and changed nothing else, so a policy trained one interface-only revision behind is the same function under the new
interface once the first layer of every network that reads the observation gains three zero columns. That mapping is
done by `environments/shared/scripts/widen_checkpoint.py`, never by hand and never by re-labelling:

```bash
python -m environments.shared.scripts.widen_checkpoint --species trex --stage stance \
    --from-stage-dir <parent run>/01_stance --to-stage-dir <new run>/01_stance \
    [--label L] [--parent-run-id ID] [--allow-legacy-plant] [--max-revision-gap N]
# or the explicit pair: --model <zip> --vecnorm <pkl> --algorithm ppo|sac --seed S --n-envs N --timesteps T
```

- **The identity gate.** The parent archive's recorded plant identity must be the current plant at most
  `max_revision_gap` interface-only revisions behind (default 1 — exactly r → r+1, the Phase C bump alone;
  `--max-revision-gap N` on the CLI, `max_revision_gap=N` in the API; decision D-C17): same `species`,
  `physics_sha256`, `nq` / `nv` / `nu` and `action_dim`,
  `1 <= current - parent.policy_interface_revision <= max_revision_gap` and `observation_dim + 3 == current`.
  Anything else is refused with every differing field named; a parent further behind than the bound is refused with
  both revisions, the measured gap, the bound and the flag named. The fields other than the revision apply whatever
  the bound, so the tool is scoped to the Phase C bump — three appended observation dims with the physics and the
  action mapping untouched — and a bound above 1 can only cross fingerprint-only intermediate revisions (there is
  no ladder of intermediate manifests to walk: the widening is applied once, against the checkout's manifest).
  Setting N > 1 asserts, from `configs/plant_versions.toml`'s numbered notes, that the intermediate bumps changed
  nothing the widening cannot bridge; the report records `revision_gap` / `max_revision_gap`. The two certified
  trex stance runs of 2026-08 are r11 archives, two revisions behind r13 across the fingerprint-only r11 → r12 bump
  (note 11, the perturbation engine): refused under the default, widened under `--max-revision-gap 2`
  (`docs/KNOWN_ISSUES.md`, the Phase C entry). A bump that changed action meaning at fixed dimensions (the biped
  home-residual examples above) is not widenable and still needs fresh checkpoints. A parent without an identity is refused unless
  `--allow-legacy-plant`, which reads its width from the saved observation space. The VecNormalize sidecar must
  record the parent archive's plant; under the legacy allowance an unstamped sidecar is accepted with a warning,
  and a stamped sidecar of a legacy (unstamped) archive must record the parent's species and width.
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
  current gate before it certifies. It is refused into an occupied or non-empty target, and a parent whose
  `gate_verdict.json` did not pass is refused.
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
`--to-stage-dir <LOG_BASE>/<species>/<algo>/<new run id>/<stage_dirname(species, root)>` and `--label` when the
session sets `RUN_LABEL`. On Colab: (1) run the notebook's section 1, then a scratch cell
`from google.colab import drive; drive.mount("/content/drive")`, never the storage cell, which would mint a run
directory and its provenance; (2) `LOG_BASE` is `/content/drive/MyDrive/mesozoic-labs/logs`; (3) run
`!cd /content/mesozoic-labs && python -m environments.shared.scripts.widen_checkpoint ...` (the tool's module docstring
lists the same steps). Then run the notebook with `RUN_ID` set to that id, `SEED` to the parent's recorded
`run.seed` and `TRUNK_FROM = ""`, and its chain loop judges the widened root (BEHAVIOR_RECIPES_PLAN §4.6, decision
D-C13). The storage cell refuses any other `SEED` before it writes anything (D-C14), and the resolve cell refuses a
trunk until the widened root holds a verdict. `docs/KNOWN_ISSUES.md` lists the pre-Phase-C checkpoints this applies
to. JAX checkpoints are not widened: `jax_checkpoint.load_checkpoint` validates the recorded identity against
`current_plant` and has no widen path, so a pre-bump JAX checkpoint fails closed.

## Backend parity and runtime binding

The policy fingerprint includes normalized executable code plus portable, quantized synthetic observation probes. The
canonical writer requires SB3 and MJX to produce the same ordered observation for the four dual-backend species (the two
compsognathus plants are SB3-only and report parity `None`: `backend_observation_equal` is computed only when the
environment lists `jax-mjx` among its training backends). Since the Phase C interface revision
(BEHAVIOR_RECIPES_PLAN §4.6) both probes inject the non-zero `COMMAND_PROBE_VECTOR = (0.25, -0.5, 0.75)` into the
trailing 3-dim command segment — the SB3 probe sets `env._command` beside the model/data swap and the MJX probe passes
`command=` to `build_mjx_observation` — so the parity assertion covers the appended slot rather than three zeros.
`validate_mjx_environment_plant` also checks the MJX observation width against the identity. MJX registration
values (root-body IDs, sensor offsets, action mapping, frame skip, and control timestep) are versioned alongside the SB3
interface. Curriculum configuration may tune rewards and termination rules, but cannot override these plant-level keys.

Before any artifact is tagged, training validates the environment that actually runs. SB3 validates the concrete
Gymnasium model and observation interface behind the VecEnv. JAX validates both the compiled `mj_model` and the live MJX
interface configuration. This prevents a stale runtime registry, alternate model path, or modified sensor/body mapping
from being mislabeled with the canonical identity.

## Checkpoints and legacy artifacts

Current plant identity compares policy and physics revisions, hashes, and interface dimensions. Source-closure and
visual differences are retained for provenance but do not block policy replay.

Artifacts without plant identity are legacy artifacts. Contract validation fails closed by default; loading one requires
an explicit `allow_legacy_plant=True` migration/evaluation choice and emits a warning. That override acknowledges missing
provenance—it does not make the artifact current or verified.

The low-level JAX `load_checkpoint` and `restore_train_state` APIs also require the caller to supply `current_plant`;
omitting it is an error, even when `allow_legacy_plant=True`. `unsafe_skip_plant_validation=True` exists only for
deliberate low-level artifact inspection, emits an explicit warning, and must not be used to resume training or run
evaluation. The low-level `load_vecnorm_stats` API follows the same rule for SB3 normalization state.

SB3 models, VecNormalize statistics, JAX checkpoints, Ray trial/resume state, promoted sweep winners, stage configs,
metrics, and run directories all carry the same identity. Promotion validates the embedded model and normalization
payloads, not only an adjacent sidecar.

The checked-in result summaries predate this contract and remain schema-v2, historical, unverified records with nullable
legacy `model_hash` fields. Do not translate those null values into current layered fingerprints without a reproducible
run and evaluation record.
