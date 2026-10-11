# Heading-free observations and stance certification

Date: 2026-10-08; revised 2026-10-10 after review. Status: **design with executable research evidence; the observation migration is carried out (D-D31, 2026-10-11), the heading certificate is not implemented**.
Repository baseline: `10f859f96223054932bdc4fa883b1153dde78500` (#604).

## Recommendation

Give anatomical T. rex and anatomical Compsognathus a heading-free observation
contract before their next training runs. Replace the world quaternion with
projected gravity, express linear velocity in the gravity-aligned heading frame,
and express the target direction in the pelvis frame. Keep tilt, relative target
bearing, local angular velocity, contact information and the trailing command
inputs. Train new policies and new normalization statistics.

Make a six-heading, full-floor-truth stance panel a required certificate, with
failure blocking handoff, reuse and publication. The existing heading probe is
useful evidence but is not that certificate: `stance_quality/v2` has since
certified a T. rex stance that falls when turned −45° or −90°
([later runs](#later-runs-on-the-current-interface)). Keep the physical plants
and reward formulas unchanged for the first controlled comparison on the new
interface, so those runs isolate the representation change. That comparison is
not expected to raise v2 clean counts at the training heading, where the
dominant T. rex failures are pad centre-of-pressure (CoP) and settle checks that
the observation does not touch. The CoP/reward work (#608) proceeds in parallel
and is retrained on the new interface.

Decided by the maintainer on 2026-10-11, until the migration and certificate
exist:

- A stance PASS on a pre-migration interface (T. rex policy interface r13,
  anatomical Compsognathus r3) is heading-uncertified and provisional.
  `20261009_155723` is research-only, not a trunk: no recovery, walk or hunt
  starts from it, and the recovery stage already trained from it is
  research-only too.
- Handoff tooling warns, without blocking, when any whole-scene row of
  `stance_heading_probe.json` has a full-horizon fraction below 0.95, the v2
  per-panel full-horizon rail. The probe otherwise stays report-only (D-D27)
  until the certificate below replaces it.

This follows the scope already recorded in [NEXT_STEPS](NEXT_STEPS.md): both
anatomical species together after #604. It does not change Compsognathus robot or
Velociraptor. Their interfaces require separate decisions. Velociraptor feeds its
policy the same three world-frame blocks (a world `framequat` on the `imu` site,
`qvel[0:3]` and the world prey direction), and no Velociraptor heading probe
exists (2026-10-11: by the maintainer's approval its stance, like the
Compsognathus robot's, now declares the report-only probe; no trained
Velociraptor stance has been probed). It stays out of scope here, following
only after the maintainer's two current Velociraptor runs are reviewed (the
2026-10-07 decision). Its
heading-free revision should be coordinated with #608's planned Velociraptor
interface change (per-site contact-load channels), so its checkpoints are
stranded once, not twice.

## What the independent experiments established

The experiments use the current stance configurations, MuJoCo 3.10.0, fresh reset
seeds 18042–18049, and the completed T. rex run `20261007_132026` (training seed
50), the latest completed T. rex stance when this design was written; it stays
the original evidence. Its [model](https://drive.google.com/file/d/1AyCn2LFKbn-Ptujmr67Spaju7Gewvv3a/view)
and [paired VecNormalize](https://drive.google.com/file/d/15lslFyKmlnRSRp9_fXpUvR4_ZJvg4h8S/view)
were loaded with the repository's plant validation. No old policy was fed a new
observation layout.

| Whole-scene rotation | Seed-50 policy: full horizon / 8 | Mean steps | T. rex home-pose controller: full horizon / 8 |
|---|---:|---:|---:|
| −90° | 0 | 75.625 | 8 |
| −45° | 0 | 149.625 | 8 |
| 0° | 8 | 1000.000 | 8 |
| +45° | 2 | 524.625 | 8 |
| +90° | 0 | 64.250 | 8 |
| 180° | 0 | 44.875 | 8 |

The −90°, +90° and 180° policy episodes all terminated on tail contact. The
anatomical Compsognathus home-pose controller also completed 8/8 episodes at
every angle. Every home-pose episode met the current v2 **individual** clean
criteria, for 96 episodes across the two species. Eight seeds are insufficient
for certification: even 8/8 yields a one-sided 95% clean-probability lower bound
of only 0.6877. The seed-50 policy met current individual v2 criteria on 0/8
episodes at every angle, including 0°. Completion is not stance quality.

The saved seed-50 checkpoint predates the latest reward/gate edits. This is a
replay under the current configuration, not a reproduction of its original
publication score or proof that the newer changes made its training worse.

Two independent observation candidates were checked on 100 tilted, moving states
per species and seven rotations per state (1,400 comparisons per candidate).
These kinematic states were lifted clear of the floor to separate the frame
transformation from contact-solver branching. Actual contact trajectories were
tested separately in the rollout sweeps.

| Representation | T. rex max absolute observation difference | Compsognathus max difference |
|---|---:|---:|
| Current world-frame observations | 1.996764 | 1.774866 |
| Projected gravity + transformed vectors | 1.1921 × 10⁻⁷ | 1.1921 × 10⁻⁷ |
| Yaw-removed quaternion + transformed vectors | 1.1921 × 10⁻⁷ | 1.1921 × 10⁻⁷ |

The changing legacy blocks were the quaternion, linear velocity and target
direction. Joint state, local gyro, local accelerometer and the distance/command
tail were unchanged in these probes. Zero contact readings in the lifted-state
test do not establish contact-force invariance by themselves.

Four short PPO experiments used the projected-gravity wrapper: two training
seeds (11, 12) for each species, 8,192 steps each, two vector environments and a
64×64 network. All four updated 13 policy parameter tensors. Each was evaluated
on four fresh reset seeds at −90°, 0°, +90° and 180°, with frozen normalization:
**64/64 evaluation episodes reached 1,000 steps**. The maximum paired reward
difference across headings was 3.57 × 10⁻⁷ for T. rex and 5.27 × 10⁻⁹ for
Compsognathus. All four angles are multiples of 90°, where T. rex's contact
friction is symmetric ([Scope of the symmetry](#scope-of-the-symmetry)); no
smoke policy was evaluated at ±45°.

These smoke runs are plumbing checks. They establish that PPO, normalization,
action selection and stepping work together with the candidate representation:
parameters updated and rotated episodes completed. The policies effectively
stayed at the zero-action home-pose controller: on the paired seeds and
headings, T. rex seed 11 and both Compsognathus seeds return within 1.3 of its
return, and T. rex seed 12 returns 11–35 below it. The 64/64 is that
controller's survival, not learned heading robustness. They do **not**
establish convergence, useful learned balance, production recipe quality, full
v2 stance quality, or a certified trunk. No smoke checkpoint is eligible for
reuse.

Together, the results support a representation-level heading defect rather
than an inability of these plants to stand when rotated. They do not isolate
which of the three leaking blocks contributes most to the old learned failure.

### Later runs on the current interface

Three more T. rex stances on the current interface completed after this design
was written (two started on 2026-10-08, `20261009_155723` on 2026-10-09), all
on policy interface r13 (observation width 64) at `10f859f`, on the
D-D27 task with the D-D28 pad CoP bar: 11,001,856 steps and about 15 h each.
Each bundle carries the report-only heading probe (`stance_heading_probe.json`,
eight episodes per row on reset seeds 3042–3049). The probe columns are
full-horizon fractions:

| Run (training seed) | `stance_quality/v2` at 0° | Whole scene −90° / −45° / +45° / +90° | Animal only −90° / −45° / +45° / +90° |
|---|---|---|---|
| `20261008_163256` (42) | FAIL, 0/40 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| `20261008_163410` (48) | FAIL, 36/40, LCB 0.7856 | 0 / 0 / 0.625 / 0 | 0 / 0 / 0 / 0 |
| `20261009_155723` (52) | **PASS**, 39/40, LCB 0.887 | 0 / 0 / 1.00 / 1.00 | 0 / 0 / 0.875 / 0.125 |

The statue completes 1.00 on every row of all three probes. `20261008_163256`
and `20261008_163410` fall 0/8 at ±90°, like the seed-50 replay;
`20261008_163410` stands 5/8 only at +45° whole-scene.

`20261009_155723` is the newer counterexample, and the case the certificate must
refuse. `stance_quality/v2` certified it on 2026-10-10 (its provenance records
the deliverable as provisional: one of the two certification seeds), and a
recovery stage then trained from it. Yet with the whole scene turned it completes
0/8 at −90° and −45° while it stands 8/8 at 0°, +45° and +90°. Unlike the seed-50
replay's symmetric 0/8 at ±90°, the failure is one-sided, so heading dependence
differs between policies and one or two probe angles cannot stand in for the
grid. The run's VecNormalize statistics, read by the 2026-10-10 review from its
`robust_best_model_vecnorm.pkl`, show how far a turn moves the observation: the
pelvis quaternion's z component has a training std of 0.0415, about ±5° of yaw.
A 45° turn therefore drives the normalised quat_w to the −10 clip and quat_z to
about ±9 (−8.9 at −45°, +9.5 at +45°), roughly 9 standard deviations, and ±90°
clips both. Every turned observation is far outside the training distribution;
this does not by itself predict which side a given policy tolerates.

At the training heading, where the observation is in distribution, the dominant
T. rex failures are pad CoP and settle checks. `20261008_163256` fails
`max_sole_cop_fore_aft` on 40/40 episodes (median 0.99, the pad's front edge)
and nothing else. `20261008_163410` fails CoP on 3/40 and sole tilt on 1/40. The
seed-50 replay above fails CoP on 8/8 and settle peak force on 4/8.
`20261009_155723` passes with a worst CoP of 0.797 against the 0.80 bar. The
T. rex stance reward has no CoP term; D-D28 prices CoP only in the gate.
Heading-free observations do not touch any of this, so the migration is not
expected to raise v2 clean counts at 0°.

## Observation contract

Let `R` map pelvis coordinates to world coordinates, `v` be the root's world
linear velocity, and `d = (target − pelvis) / (distance + 1e−8)`. Let `H` map
world coordinates to the gravity-aligned pelvis-heading frame. Its forward axis
is the normalized horizontal projection of pelvis X.

The candidate outputs `g = Rᵀ(0, 0, −1)`, `v_heading = H v`, and `d_body = Rᵀ d`.
Gravity is a unit direction, not acceleration in m/s². It preserves roll and
pitch while discarding the unobservable world yaw. For a global yaw rotation
`Q`, `R′ = QR`, `v′ = Qv`, and `d′ = Qd`; these three outputs are invariant.
Joint residual actions consequently need no world-frame action transform.

| Segment, in order | Width | New semantics |
|---|---:|---|
| Joint positions and velocities | existing | Existing order, units and normalization |
| `pelvis_projected_gravity` | 3 | Unit down vector in pelvis coordinates |
| `pelvis_angular_velocity` | 3 | Existing gyro sensor in its local site frame |
| `pelvis_heading_linear_velocity` | 3 | Forward, left, vertical velocity; m/s |
| `pelvis_linear_acceleration` | 3 | Existing local accelerometer; m/s² |
| Foot contact | 2 | Existing left/right scalar readings |
| `prey_direction_body` | 3 | Full pelvis-frame normalized relative direction |
| Prey distance | 1 | Existing scalar distance |
| Command | 3 | Existing scaled heading-frame vx, vy, yaw rate; always last |

Suggested schema name: `bipedal-target-heading-free/v1`.

| Species | Physics revision | Policy revision | Observation dimension | Visual revision |
|---|---:|---:|---:|---:|
| T. rex | keep 9 | 13 → 14 | 64 → 63 | keep 4 |
| Anatomical Compsognathus | keep 2 | 3 → 4 | 56 → 55 | keep 2 |

Use the validated free-root quaternion `data.qpos[3:7]`, normalized in float64,
then cast the final observation to float32. Construction calls `_get_obs` before
the normal reset/forward sequence; its orientation sensor can still be zero.
Do not normalize that uninitialized sensor or use a stale `xmat`. The study
checked root/sensor rotation agreement after `mj_forward` on both species.
Production must also test direct construction, reset and terminal observations.

When the horizontal projection of pelvis X has norm below `1e−8`, use the
horizontal projection of pelvis Y rotated clockwise 90° to define heading X.
This fallback rotates with the scene and is finite when pelvis X is vertical.
It is explicitly a terminal-pose convention: the yaw chart cannot be globally
continuous at that singularity. Invalid or zero quaternions should fail with a
clear state error, rather than silently injecting a world-axis default. Tests
cover both vertical signs, near-vertical poses, global rotations and q/−q.

*Carried out 2026-10-11 as T. rex policy interface r14 and anatomical
Compsognathus r4 (decision D-D31, `configs/plant_versions.toml` note 18),
with the contract above unchanged.* `environments/shared/heading_observation.py`
builds the three segments from `qpos[3:7]` in float64, with the 1e−8 fallback
and an `ObservationStateError` for a quaternion that is not four finite
numbers or has norm below 1e−12. `TRexEnv._get_obs` and a new
`CompsognathusBiologicalEnv._get_obs` assemble the table's order (63 and 55
wide); the robot keeps its world-frame layout. The target vector keeps its
origin at `xpos[pelvis]`, so the distance channel keeps its meaning; after a
step `qpos` is post-integration while `xpos` and the sensors come from the
last substep's forward pass, the same one-substep mix the old `qvel[0:3]`
channel had, which leaves the invariance intact. The plant contract's
segments for the schema name each block's frame and threshold and describe
the foot channel per foot (two widths of two, each a sum over its listed
touch sensors), and it checks that the widths sum to the observation width.
Measured on the production environments: a global yaw leaves the float32
observation unchanged on 100 lifted, tilted, moving states × 7 turns per
species (0 against the 2 × 10⁻⁶ budget), `R` matches `mju_quat2Mat` to
4.4 × 10⁻¹⁶ and the fallback holds to 3.3 × 10⁻¹⁶.

Projected gravity is preferred over the same-width yaw-removed quaternion:
there is no quaternion sign ambiguity, fewer channels are needed, and the
interface expresses the balance quantity directly. The quaternion candidate
also removes yaw, but keeping its old width would **not** make old weights or
normalization statistics compatible.

Do not zero yaw-related quaternion components: that damages tilt and leaves
world velocity and target bearing exposed. Do not rotate gyro/accelerometer a
second time; MuJoCo reports them in their sensor frame. MuJoCo free-joint linear
velocity is world-frame but angular velocity is body-frame; the latter is left
unchanged when rotating a state ([MuJoCo coordinate conventions](https://mujoco.readthedocs.io/en/3.5.0/overview.html)).

## Scope of the symmetry

A whole-scene rotation preserves a task only when it rotates all world-frame
inputs to that task. For the initial stance certificate, require a flat,
isotropic floor, `command_mode = "none"`, no applied pushes, and rotation of:

1. Root orientation and world linear velocity, about world vertical through the root.
2. Target/mocap positions about the same pivot, and mocap orientations.
3. Cached world reference directions, followed by `mj_forward` and invalidation
   of the substep aggregates.

A flat, isotropic floor does not make every plant's contact physics
yaw-invariant. T. rex's box pads on the plane get contact tangent frames aligned
with the world axes, and with MuJoCo's default pyramidal friction cone
(`trex.xml` set no `cone` before physics r9) their friction was slightly
anisotropic. A ±45° whole-scene turn was therefore a small physical
perturbation as well as an observation change, and the current `SpawnYaw`
docstring's "changes nothing physical" held for T. rex only at multiples of 90°.
This study's own statue rows show it: per-episode statue rewards in
`rollouts_trex.json` match 0° to 4.2 × 10⁻¹¹ at −90°, +90° and 180° but differ
by up to 0.197 (of about 3,760) at ±45°. The 2026-10-10 review measured the
rest: at multiples of 90° results agree to about 10⁻¹¹; at 45° the statue's v2 metrics barely move (sole tilt by
about 4 × 10⁻³ degrees), though random open-loop actions separate the joint
trajectories by a few hundredths of a radian; with `cone="elliptic"` every
compared case agreed to about 10⁻¹⁴. Velociraptor's floor contacts are capsules,
whose tangent frame turns with the foot, and stay yaw-invariant at 45° under the
pyramidal cone; a fall onto its ellipsoid head did not. Anatomical Compsognathus
already declares `cone="elliptic"`; its statue rows agree to about 3 × 10⁻¹⁰ at
all six headings. *Since D-D30 (2026-10-11, `configs/plant_versions.toml` note
17) the velociraptor (physics r4), the brachiosaurus (r5) and the dibothrosuchus
(r2) declare it too, so every plant has one friction model and every heading of
every species is physically identical: their statues' ±45° rows agree to
1.4 × 10⁻¹² (on the old plants up to 0.84 on the brachiosaurus's ellipsoid feet
and 3.7 × 10⁻² on the dibothrosuchus's box pads), and a later species'
heading-free migration needs no physics revision of its own for the cone.*

The maintainer chose on 2026-10-11 to switch T. rex to `cone="elliptic"` in the
same retrain window. It is a physics revision of its own (a new physics
identity, with statue references, stance bars and the recovery calibration
re-measured), landed before the first fresh heading-free T. rex stance and kept
out of this observation-only migration. Once it lands every T. rex heading is
physically identical, as on Compsognathus. Until then this design treats
T. rex's ±45° cells as slightly perturbed physics, not exact replicates: each
heading is judged against its own statue panel (the per-heading reference in
the certificate below), and paired traces are compared to solver tolerance only
where the physics is invariant.

*Carried out 2026-10-11 as T. rex physics r9 (decision D-D29,
`configs/plant_versions.toml` note 16), before this migration.* Through the
heading probe's own path (`run_panel` with a whole-scene `SpawnYaw`, seeds
3042–3049) the statue's ±45° rewards, which differed from 0° by up to
2.6 × 10⁻² on r8, agree to 3 × 10⁻¹¹ at every heading, and a 45° turn under
fixed random commands repeats the joint trajectory to 5 × 10⁻¹⁴ rad (7.7 × 10⁻²
on r8). Only the cone moved. The statue references were re-measured and carry,
and every stance bar is kept, although the r9 statue is clean on 39/40 of the
certification seeds (one reset-noise nosedive) and its worst out-of-sample pad
tilt is 2.061° against the 2.0° bar (note 16 and KNOWN_ISSUES' fore-aft entry).
T. rex has no recovery calibration file; the recovery evaluation's calibrated
constants, stale since r8, are still re-measured on the first certified stance.
Every T. rex heading is now physically identical, so the provisions above for
its ±45° cells apply only to evidence taken on r8.

Use identical joint state, local angular velocity and RNG draws for each paired
reset. Recreate an environment and reset to the seed for each angle; do not
accumulate turns on a state advanced by another episode. Reset recurrent policy
state, action filtering and any predictor counters per episode.

Rotating only the animal while leaving the target fixed changes relative target
bearing. The policy should see that difference; it is not a failed invariance
test. The prototype explicitly tests this distinction.

The current `_apply_spawn_yaw` handles the stance inputs above but does not
rotate a live command controller's world desired heading, world push schedules,
active external wrenches or arbitrary terrain. Do not silently reuse it to
certify recovery or command tracking. Later protocols must rotate those inputs,
their cached state and recorded events too. The study's live-controller test
confirms equal policy commands when both actual and requested headings rotate,
including scheduled switches and explicit target requests.

Keep world heading available to the planner, diagnostics and command adapter.
The change removes it from the policy's state representation; it does not remove
the ability to request a world direction.

## Changes within the repository

| Location | Required production change |
|---|---|
| New `environments/shared/heading_observation.py` | Pure NumPy frame helpers and explicit segment semantics; normalized wxyz quaternion, projected gravity, heading velocity, pelvis target, terminal fallback. No simulator mutation. |
| `environments/trex/envs/trex_env.py::_get_obs` | Assemble the new schema directly from valid root state and existing local sensors. Declare the supported backend explicitly. |
| `environments/compsognathus/envs/compsognathus_env.py::CompsognathusBiologicalEnv` | Add an anatomical-only `_get_obs` override. Keep the shared parent and robot observation source untouched. |
| `environments/shared/plant_contract/policy_layer.py` | Schema-conditional segment names, widths, frames and gravity convention; hash all new helper implementations as well as `_get_obs`. Preserve existing payload bytes for other schemas. |
| `configs/plant_versions.toml` | The two policy revisions/schema changes above; no physics or visual revision change. T. rex's elliptic friction cone (the maintainer's choice of 2026-10-11) is the separate physics revision r9 (note 16, D-D29), carried out before this migration with re-measured statue constants. |
| `configs/species_manifest.toml` and species documentation | Declare T. rex's new interface SB3-only; expose new dimensions/revisions accurately. Compsognathus is already SB3-only. |
| `environments/trex/mjx_config.py`, `test_plant_contract_frozen_mjx.py`, `pyproject.toml` | Retire T. rex's registration with its policy revision, update the dual-species pin and remove its formatter exclusion. Keep the remaining shared core and registrations unchanged. |
| `configs/plant_manifest.generated.json`, `environments/shared/data/plant_manifest.generated.json` | Regenerate, including schema probe outputs and supported backend declarations. |
| `environments/shared/tests/fixtures/phase_c_reset_golden.json` | Recapture the two species' observation records with revision provenance; verify the reset state, RNG and zero-action trajectory records retain their existing meaning. |
| `configs/compsognathus/recovery_calibration.json` | Restamp the anatomical calibration's interface/task identity only after verifying unchanged physics and measured environment; preserve its original measurement and numeric calibration. |
| New `environments/shared/curriculum/stance_heading_gate.py` | Pure grouped-seed decision logic, strict protocol/evidence validation, full reuse of v2 episode classification. |
| `environments/shared/curriculum/gate_schema.py`, `manager.py`, stage config parsing/fingerprinting | Register the new gate/protocol fields; keep the in-training screen distinct from post-stage certification. |
| `environments/shared/reporting/stance_report.py` | A dedicated heading certificate builder using `run_panel(..., floor_truth=True)` at every angle, paired statue panels and immutable episode evidence. Correct the `SpawnYaw` docstring and probe text that call a whole-scene turn physically neutral; for T. rex that held only at multiples of 90° until physics r9 (D-D29), and it still holds only there for a pyramidal-cone plant on box or ellipsoid feet (Brachiosaurus, Dibothrosuchus). |
| `environments/shared/reporting/gates.py`, `stage_artifacts.py` | Judge the heading certificate before reporting a handoff pass; propagate measurement failures. Leave optional diagnostic probes separately identified. |
| `environments/shared/result_bundle/evidence.py`, gate verdict/reentry/ancestor consumers | Re-derive the decision from bound rows and protocol; refuse missing or mismatched heading evidence for a new-schema stance. |
| `configs/trex/stance.toml`, `configs/compsognathus/stance.toml` | Opt into the new versioned heading gate while retaining the current species-specific stance bars. The `stance_probe_spawn_yaw_deg` comment in `configs/trex/stance.toml`, which says only the observation differs, needs no correction since physics r9 (D-D29) made every T. rex heading physically identical. |
| `configs/digest_snapshot.generated.txt`, species catalog/generated website data | Regenerate after the schema/gate migration and review the exact scope of changed identities. |
| Shared/species tests and docs | Frame algebra, lifecycle, negative evidence cases, old-checkpoint refusal, unchanged robot/other species and end-to-end publication refusal. |

The policy contract currently fingerprints the `_get_obs` call site. Moving
arithmetic into a new helper without fingerprinting the helper body would leave
a compatibility hole. Include that module's function semantics conditionally
for the new schema. Synthetic contract probes also need valid normalized root
quaternions; do not depend on precomputed matrices being initialized there.

### Backend decision

T. rex currently inherits a dual SB3/MJX declaration for historical contract
compatibility. The MJX training runtime has already been retired; SB3 is the
only active training/evaluation backend. Its observation probe is nevertheless
compared with the frozen MJX implementation. Updating only its SB3 observation
while continuing to claim parity is invalid.

Recommended migration: explicitly declare the new T. rex revision
`supported_training_backends = ("stable-baselines3",)` and matching manifest
metadata, following the repository's existing anatomical Compsognathus and
Velociraptor pattern. This deliberately changes the recorded contract, not the
set of currently executable trainers. [CLEANUP_PLAN §4.1](CLEANUP_PLAN_2026_09.md)
and PLANT_CONTRACT explicitly provide for a species leaving the frozen core at
its next policy revision, and document Velociraptor's registration deletion.

Retire T. rex's `mjx_config.py` with that revision, remove its entry from the
frozen-registration pin and its ruff exclusion, and retain the low-pass action
filter pins that still belong to T. rex's active policy contract. The shared
frozen implementations and the Brachiosaurus/Dibothrosuchus registrations remain
byte-identical. Update the corresponding explanatory docs. Merely changing the
manifest without this retirement would fail the existing test that the frozen
registrations are exactly the dual-backend species.

Restoring an MJX training runtime with the new representation would be a
separate project with a versioned backend path and parity tests. Editing the
remaining frozen builders or bypassing their parity checks is not part of this
migration.

*Carried out 2026-10-11 (D-D31):* `TRexEnv` declares
`supported_training_backends = ("stable-baselines3",)`, the species manifest
`training_backends = ["stable-baselines3"]` (its two JAX success metrics
removed), `environments/trex/mjx_config.py` is deleted with its ruff
exclusion, and the frozen-core pin covers the Brachiosaurus and
Dibothrosuchus registrations, which are unchanged; the low-pass filter pins
stay. The plant contract also refuses the heading-free schema on any species
that declares `jax-mjx`. As for every SB3-only species, T. rex's `_cache_ids`
tokens now enter its policy digest, which the maintainer accepted.

## Heading certificate

Gate kind: `stance_quality_heading/v1` (the maintainer's choice of 2026-10-11);
protocol: `whole_scene_yaw/v1`; report schema:
`mesozoic.stance-heading-report/v1`. The obvious `stance_quality/v3` would
clash: the v2 gate already writes reports with schema
`mesozoic.stance-gate-report/v3` (`STANCE_V2_REPORT_SCHEMA` in
`stance_gate_v2.py`, numbered so that "report v2" is never "kind v2"), so a v3
kind would invite misreading a v3 report as a v3 certificate in verdicts and
bundles. Keep the existing v2
floor-truth measurements and species-specific per-episode bars. Version the new
report and CSV schema explicitly rather than changing the meaning of a v2
certificate already in a completed bundle.

The existing `build_stance_gate_report` routes a report with `spawn_yaw` through
its non-certifying probe path, even when v2 was requested. Meanwhile
`stage_artifacts._write_heading_probe` is optional diagnostic output whose
errors are logged. Moving its JSON next to a certificate or renaming it a gate
does not make a failure block anything. A new certificate path is required.

### Protocol and statistic

- Evaluate the fixed heading grid `[−90, −45, 0, 45, 90, 180]` degrees. Canonicalize
  angles and refuse duplicates, including synonymous −180°/180° entries.
- Use 40 independent reset seeds, each repeated at every heading: 240 policy
  episodes, each with the configured 1,000-step horizon and settle window.
- Roll the home-pose controller on the identical 40×6 grid. Compute each
  heading's statue-relative reference with the existing v2 reducer. Do not
  substitute a pooled reference from a different seed block or plant. Where the
  physics is yaw-invariant (every heading of both species since T. rex's
  physics r9, D-D29) these per-heading statue panels are replicates; where it
  is not (T. rex's ±45° cells on r8) they are the correct reference.
- Apply the full v2 decision at each heading, retaining the per-heading
  ≥0.95 full-horizon rail, current reward/statue rails, and T. rex's declared
  ≤1 hop-or-fall rail. Compsognathus does not currently declare that last rail;
  adding one would be a separate threshold decision.
- Define `clean_seed[s] = all(clean_episode[s, angle] for angle in grid)`.
  Require the existing exact one-sided 95% binomial lower bound over those
  **40 seed groups** to be at least 0.80. This requires 37/40 clean groups
  (LCB 0.8174); 36/40 fails (0.7856).
- Report per-angle counts, per-seed failures and the grouped lower bound. The
  primary claim is probability of clean stance across this fixed tested grid,
  not a statistical guarantee over every continuous heading.

The binomial calculation inherits v2's fixed measurement/reference convention;
it does not include uncertainty in the estimated statue reference or compensate
for repeated checkpoint selection on the certification seeds. Preserve a frozen
checkpoint and the separate seed roles when interpreting that bound.

The rotations are not 240 independent trials. A tested counterexample has one
different failing seed at each of six headings: pooling calls it 234/240 and
produces an apparent LCB of 0.9513; only 34/40 seeds pass every angle, whose
correct grouped LCB is 0.7253. Per-angle summaries alone also hide this issue.

### Bound evidence and failure behavior

Every report must identify the model and VecNormalize file hashes, species,
plant/policy identity, task and gate hashes, recorder/measurement identity,
normalization mode, control timestep, horizon, settle window, protocol revision,
ordered angle list, exact reset seed list and policy determinism. Persist each
raw v2 metric row with explicit `seed` and `yaw_deg`; the Cartesian grid must be
complete and unique. Store the corresponding statue rows and reference inputs.

The judge, bundle publisher and reentry/ancestor validator must re-derive the
decision from these bound rows. Refuse missing headings, duplicate pairs, stale
model/normalizer hashes, altered thresholds, another measurement version,
nonfinite values, missing statue data or report-only probes. A claimed `passed`
field is not evidence. A failed rollout or missing heading measurement fails
the certificate; it cannot fall back to the yaw-zero result.

Select/freeze the handoff checkpoint and its normalizer first, then measure the
grid, judge it and publish. Run the full grid at the post-stage certificate,
not at every PPO evaluation. Leave the cheap in-training screen useful for
selection without letting it certify. Publication uses seeds 3042–3081 and the
fresh audit uses 7042–7081 as already planned in NEXT_STEPS; preserve their seed
roles. A cached statue grid is usable only when all physical/task/measurement/
protocol/seed identities match and its evidence is immutable.

Existing v2 artifacts remain readable as historical v2 results. They cannot be
promoted to the new certificate by attaching an optional heading report.

## Migration and acceptance

1. Implement the observation schema and explicit SB3 capability decision for
   the two species; add the helper semantics to the policy digest. Verify other
   species' policy, physics and visual hashes remain unchanged.
2. Implement the grouped heading certificate through production, judge,
   publication and reuse. Add corruption/omission tests before enabling the
   gate in the two stance configs.
3. Regenerate manifests, catalog and digest snapshot together. All stages and
   behavior recipes inheriting either species' policy identity receive new task
   identities, even when their reward arithmetic does not change. Inspect those
   expected changes; reject unexplained cross-species drift. Use
   `environments/shared/tests/reset_golden.py` for the two species' revised
   observation captures, preserving reset/RNG and physical trajectory checks.
   Anatomical Compsognathus's recovery calibration binds the full plant identity:
   use `environments/shared/scripts/restamp_recovery_calibration.py` with an
   explicit heading-interface reason after verifying unchanged physics and
   measured environment. Its quiet-controller numbers do not require a new
   physical calibration for an observation-only change. This restamp does not
   make old learned recovery policies compatible. Leave the robot profile alone.
4. Start fresh PPO weights, optimizer state and VecNormalize for the new policy
   revisions. Old checkpoint/normalizer loads must fail in training, evaluation,
   auto-trunk resolution and manual `TRUNK_FROM` paths. Existing command-channel
   widening cannot migrate nonlinear replacement/reframing of old features.
5. Keep the present reset distribution for the first controlled comparison.
   Invariant features remove yaw algebraically. Random spawn yaw can be a later
   distribution-coverage experiment, but is not a substitute for removing the
   leaks. Adding it also requires a task identity and deterministic RNG-stream
   decision, with targets, references and external inputs transformed together.
6. Train at least two independent production seeds (plan three, as current
   NEXT_STEPS requests), with the unchanged stance recipe including anatomical
   Compsognathus `log_std_init = −2.0`. Each must satisfy the publication grid and
   the separately reported fresh-seed grid before recovery/walk/hunt reuse. This
   is the first controlled comparison; reward changes such as #608's CoP margin
   are compared separately, paired, on the new interface.

Pre-migration runs. After the maintainer's 2026-10-07 decision that the next
T. rex stance waits for this interface, training continued on the current one:
three T. rex stances of about 15 h each (`20261008_163256`, `20261008_163410`
and `20261009_155723`, policy interface r13), the 3,006,464-step recovery stage
trained from `20261009_155723`, and the 15-run, 1,048,576-step reward pilot batch
behind #608 (source `b2c04be`; T. rex interface r13, Velociraptor r11). Step 4
strands every T. rex checkpoint among them; T. rex physics r9 (D-D29) already
refuses them on their physics digest. Label them pre-migration and
research-only: none is a trunk for the new interface, and their reward
conclusions, #608's CoP-margin effect included, are revalidated on the new
interfaces (T. rex's here; Velociraptor's after its own revision), as #608
itself plans.

Acceptance must include:

- Random tilted/moving states across yaw wrap, q/−q, ordinary and singular
  headings; float32 output error ≤2×10⁻⁶ for corresponding states. Changed
  target bearing and changed tilt must remain distinguishable.
- Finite, correctly sized observations at construction/reset/step/terminal
  states; no second rotation of local sensors; command remains trailing width 3.
- Paired full-horizon real-physics rollouts at all six headings, with fixed
  normalizer and deterministic policy. Compare action/state traces and all
  floor-truth metrics to documented contact-solver numerical tolerance where the
  physics is invariant: both species at every heading since T. rex's physics
  r9 (D-D29). On r8 that held for T. rex only at multiples of 90°, and its ±45°
  cells were judged against their own statue panels rather than by trace
  equality.
- Genuine SB3 save/load/evaluation and normalizer round trips on the new
  production environment, including interrupted-run resumption. The wrapper
  smoke experiments do not replace these production-path tests.
- All stale-artifact/cross-schema negative cases, grid completeness, grouped
  confidence counterexamples, and an end-to-end case where yaw-zero passes but
  ±90° fails and handoff/publication/reentry all refuse it.
- Full repository tests, frozen-MJX checks, plant/catalog/digest checks, and the
  existing canonical numeric baseline harness. Rewards/termination/physics
  should match before and after at corresponding states; policy/task/gate
  fingerprints move intentionally. Do not loosen pad, CoP or hop bars to get
  the first new runs through.

*Carried out 2026-10-11 for the observation (D-D31): steps 1, 3 and 4 above,
and the acceptance items that do not involve the certificate.* The tests are
`test_heading_observation.py` (frame algebra), `test_heading_free_observation.py`
(lifecycle, local sensors, whole-scene and animal-only turns, q/−q, and a
250-step six-heading paired rollout per species in the shared legs),
`test_plant_contract_heading_free.py`, `test_heading_free_checkpoints.py`
(SB3 save, load, evaluation and interrupted-run resumption on the new
environments, and old checkpoints and normalizers refused on the training,
evaluation, auto-trunk and `TRUNK_FROM` paths, the widen gate and the
behavior preparation, each by plant identity) and
`test_heading_free_rollouts.py` (full-horizon traces and every floor-truth
metric behind a frozen normalizer and an untrained PPO policy, in the SB3
job's full depth). The digest snapshot moves only T. rex and Compsognathus
lines (649 → 643; 84 reward lines unchanged), the reset golden's reset,
second-reset and trajectory halves re-took byte-identical, and the digest
harness's `--exact` captures of the base and the migration differ only in the
two species' `reset` and `obs` streams (586 of 37,034 lines): rewards,
termination and physics match at corresponding states. The Compsognathus
recovery calibration is restamped. The certificate items (step 2, and grid
completeness, grouped confidence and the end-to-end refusal) remain.

The heading certificate multiplies final panel simulation work by six: 240
policy plus 240 statue episodes per seed block. That is an explicit cost at
certification time, not a sixfold training cost. At the step rates recorded in
this study's rollout files (about 600 steps/s for T. rex, 390 for
Compsognathus), a 480-episode seed block takes about 13 and 20 minutes, against
about 2–3.5 minutes for today's 80-episode v2 panel, and the publication and
fresh-audit blocks double it. An extra off-grid angle panel is useful as an
audit of accidental angle-specific behavior but should be declared before
checkpoint selection if it will be used for admission.

With heading-free observations, the headings of a seed that the physics treats
identically (all six for both species since T. rex's physics r9; on r8,
−90°, 0°, +90° and 180° for T. rex) are deterministic replicates of one
episode, so beyond catching a leak they add only cost. *(Measured on the
migration, D-D31: replicates to solver tolerance under a stable closed loop
-- the same float32 actions and joint state to 4.5 × 10⁻¹³ over 1000 steps,
every floor-truth metric to 2 × 10⁻¹⁰; with single-threaded torch a one-ulp
action difference, carried by the stable loop to 7.3 × 10⁻⁸ in joint state
-- but not exactly in floating point.
An untrained PPO policy behind normalizer statistics gathered on the zero
action alone, whose closed loop on the Compsognathus is unstable, amplified a
one-ulp action difference at step 145 to 0.03 in action and 0.33 rad in
joint state by step 1000 at every heading but 0°. A trained policy with an
unstable loop could therefore still separate headings through rounding,
without any leak.)* The maintainer chose on 2026-10-11 to keep the full
six-heading grid anyway: a few minutes per seed block cost little against a
roughly 15 h stance run, and the grid is the check that would have refused
`20261009_155723`. A cheaper variant was considered and not adopted: the 40-seed
panel at 0° only, plus 0°, ±90° and 180° on a few seeds as leak detectors
(traces compared to documented contact-solver tolerance), adding full panels
only at headings where the physics is anisotropic. The grouped Clopper-Pearson
arithmetic is the same either way.

## Reproduction and evidence boundaries

All research code and raw result summaries are in
[investigations/heading_2026_10](investigations/heading_2026_10/README.md).
The initial experiment code is preserved in commit `221d44e`; each JSON records
the exact script SHA-256 values, arguments, versions and runtime. Subsequent
prototype edits add a finite terminal-pose fallback and additional tests; the
`pointwise_*_terminal_fallback.json` files recheck that version. Pilot and old
checkpoint rollout claims refer to the initial recorded code, not a later
unrecorded training run.

The initial capture used local commit `7b334bc`; published commit `221d44e`
has the identical Git tree `1b860f0690061ef5134a330b5b3331a12f0a8372`.
The recorded `base_commit` values remain the original capture metadata.

Runtime: Python 3.12.14, MuJoCo 3.10.0, NumPy 2.5.3, Gymnasium 1.4.0,
Stable-Baselines3 2.9.0, CPU PyTorch 2.14.1. The repo's current SB3 CI pins
PyTorch 2.13.0; these experiments are not a reproduction of that exact dependency
matrix. No GPU/JAX run, production-budget retraining, recovery/locomotion
certification or trained anatomical-Compsognathus checkpoint replay was done.

Model SHA-256: `e5c9a826de252665d06362d529a507743602d564da4d403b1262358df519685d`.
VecNormalize SHA-256: `2a80c034294e1305ae449b0883944251917d47291f413ad3fc53a6b1c1f64c7b`.

The [later-run](#later-runs-on-the-current-interface) numbers come from those
runs' Drive bundles (verdicts, gate reports and heading probes) and, where
stated, from the 2026-10-10 review's re-measurement; this directory's scripts
did not reproduce them.

The study intentionally does not write these research results into production
bundles or public species result summaries. Heading invariance also does not
solve every stance-quality issue: the T. rex pad CoP and settle failures at the
training heading and the anatomical Compsognathus bounce/flutter concerns
already recorded in NEXT_STEPS remain separate acceptance questions.

### Verification record

| Check | Result |
|---|---|
| Full repository pytest suite | 5,417 passed, 1 skipped; 852.34 seconds |
| Prototype plus existing heading/command suites | 113 passed, including 46 new research tests |
| Documentation landing-record check after adding the design link | 3 passed |
| Plant manifest check | Current |
| Canonical digest snapshot, optional backends blocked | Current; 649 lines, zero errors |
| Generated species catalog | Current |
| Ruff lint/format of the research code; git whitespace check | Passed |
| All ten JSON result files versus their recorded source hashes | Matched the retained source versions |
| 2026-10-10 revision, documentation only (Python 3.13): prototype tests; landing-record check; relative links and anchors of the edited docs; git whitespace check | 46 passed; 3 passed; all resolve; passed |

See the experiment README for commands. The full repository count includes the
67 existing heading/command tests; it does not include the 46 research tests
outside the configured test directories. No new production schema or gate was
implemented or certified in these runs.
