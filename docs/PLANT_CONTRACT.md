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
   pytest environments/shared/tests/test_plant_contract.py environments/shared/tests/test_species_catalog.py
   ```

The writer updates both `configs/plant_manifest.generated.json` and the byte-identical runtime copy under
`environments/shared/data/`. Commit the version counters, both generated manifests, generated public catalog/README
data, and the intentional source change together. CI repeats the check with the canonical MuJoCo version, tests revision
monotonicity against the PR base, and verifies identity/config loading from an installed wheel.

## Backend parity and runtime binding

The policy fingerprint includes normalized executable code plus portable, quantized synthetic observation probes. The
canonical writer requires SB3 and MJX to produce the same ordered observation for all three species. MJX registration
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
