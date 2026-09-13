---
sidebar_position: 3
---

# Custom Models

Learn how to create your own dinosaur models for Mesozoic Labs.

## Architecture Overview

Every species in Mesozoic Labs follows the same structure:

```
environments/<species>/
├── assets/
│   └── <species>.xml          # MuJoCo MJCF model
├── envs/
│   └── <species>_env.py       # Gymnasium environment (subclasses BaseDinoEnv)
├── scripts/
│   ├── train_sb3.py           # Training script
│   └── view_model.py          # Model viewer
└── tests/
    └── test_<species>_env.py  # Environment tests
```

You also need a stage manifest and TOML config files under `configs/<species>/`
for each training stage (see [Stage manifest](#stage-manifest) below).
Every implemented species must also have one entry in
`configs/species_manifest.toml`; the generated public catalog and its CI drift
check use that manifest as their source of truth.

## MuJoCo XML Format

Dinosaur models are defined using MuJoCo's MJCF XML format. Here's a minimal skeleton:

```xml
<mujoco model="custom_dino">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.002" integrator="RK4"/>

  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="50 50 0.1" rgba=".9 .9 .9 1"/>

    <body name="torso" pos="0 0 1.5">
      <joint type="free"/>
      <geom type="capsule" size="0.3 0.5" mass="10"/>

      <!-- Add limbs here -->
      <body name="right_thigh" pos="0.2 0 -0.3">
        <joint name="right_hip_pitch" type="hinge" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.08" mass="2"/>

        <body name="right_shin" pos="0 0 -0.4">
          <joint name="right_knee" type="hinge" axis="0 1 0" range="-120 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.06" mass="1.5"/>
        </body>
      </body>
      <!-- Mirror for left side, add tail, neck, etc. -->
    </body>
  </worldbody>

  <actuator>
    <position name="right_hip_pitch" joint="right_hip_pitch" kp="100"/>
    <position name="right_knee" joint="right_knee" kp="100"/>
    <!-- One actuator per controllable joint -->
  </actuator>
</mujoco>
```

## Model Requirements

1. **Free joint on torso** — The root body must have a `type="free"` joint
2. **Bipedal or quadrupedal stance** — At least 2 legs with hip, knee, and ankle joints
3. **Joint limits** — All hinge joints need `range` attributes to prevent unnatural poses
4. **Actuators** — One position or motor actuator per controllable joint
5. **Appropriate mass distribution** — Heavier torso, lighter extremities for stability
6. **Contact geoms** — Feet need contact geometry for ground interaction

## Existing Species as Reference

Use the generated specifications on the [T-Rex](./trex),
[Velociraptor](./velociraptor), [Brachiosaurus](./brachiosaurus), and
[Dibothrosuchus](./dibothrosuchus) model pages for the current compiled
dimensions. Look at the corresponding MJCF
files under `environments/<species>/assets/` for detailed examples.

## Creating the Environment

Subclass `BaseDinoEnv` and implement species-specific reward components:

```python
from environments.shared.base_env import BaseDinoEnv

class CustomDinoEnv(BaseDinoEnv):
    def __init__(self, **kwargs):
        super().__init__(
            model_path="path/to/custom_dino.xml",
            **kwargs
        )
```

Register with Gymnasium by adding an entry in `environments/__init__.py`:

```python
register(
    id="MesozoicLabs/CustomDino-v0",
    entry_point="environments.custom.envs.custom_env:CustomDinoEnv",
)
```

## Stage Manifest

Create `configs/<species>/stages.toml` declaring the species' recipe graph:
one `[[stages]]` entry per node, in an order where every parent precedes its
children. The committed Velociraptor manifest is the three-node template:

```toml
schema = "mesozoic.stage-manifest/v2"

[[stages]]                       # quiet stance; gate reward-cleared by the statue (plan §4.8)
id = "stance"
config = "stage1_balance.toml"
legacy_number = 1
recipe = "stand"
deliverable = true

[[stages]]
id = "locomotion"
config = "stage2_locomotion.toml"
legacy_number = 2
warm_start_from = "stance"
recipe = "walk"
deliverable = true

[[stages]]                       # id stays "behavior"; the task name (strike) is the TOML's [stage] name
id = "behavior"
config = "stage3_strike.toml"
legacy_number = 3
warm_start_from = "locomotion"
recipe = "hunt"
deliverable = true
```

Per entry: `id` must match `^[a-z][a-z0-9_]*$` (`stance`, `recovery`,
`locomotion` and `behavior` are reserved, and a numbered reserved id must
declare its `legacy_number`); `config` is the stage TOML filename;
`legacy_number` is one of the three historical numbers, absent on an id-only
stage; `warm_start_from` names an **earlier** entry (a self or forward
reference is fatal); `deliverable = true` marks every node whose certified
checkpoint is a published policy (a v2 manifest with no deliverable is fatal);
and `recipe` is the behavior label (`stand`, `walk`, `hunt`). A species with a
recovery node uses the four-entry T-Rex layout, where `recovery` (recipe
`stand`, `warm_start_from = "stance"`, no legacy number) sits between stance
and locomotion. See [Behavior Recipes](/docs/training/recipes) for how the
manifest drives training and publication.

## Config Files

Create the TOML configs the manifest names, one per training stage, under
`configs/<species>/`:

```toml
[stage]
name = "balance"
description = "Learn to stand without falling"

[env]
forward_vel_weight = 0.0
alive_bonus = 1.0
energy_penalty_weight = 0.0005
max_episode_steps = 500

[ppo]
learning_rate = 3e-4
n_steps = 2048
batch_size = 64

[sac]
learning_rate = 3e-4
batch_size = 256

[curriculum]
timesteps = 1000000
min_avg_reward = 50.0
min_avg_episode_length = 400
required_consecutive = 3
```

## Add the Species to the Public Catalog

Copy an existing `[[species]]` block in `configs/species_manifest.toml`
(schema version 2) and update its presentation metadata, environment entry
point, MJCF model path, notebook IDs, and result-summary paths. Add
`[[species.success_metrics]]` entries that state the actual success semantics
for every supported backend. Stage videos and curated result summaries are
optional, but any declared artifact must exist and include the required
backend and provenance metadata.

### Per-deliverable success semantics

Every behavior the species publishes — each stage its `configs/<species>/stages.toml`
flags `deliverable = true` — needs a `[[species.deliverable_metrics]]` entry
stating what "success" means for that checkpoint:

```toml
[[species.deliverable_metrics]]
deliverable = "walk"                 # a recipe label, or a deliverable stage id
backends = ["stable-baselines3"]
key = "forward_velocity_gate"
label = "Gated forward velocity (reward_and_length/v1)"
definition = "The locomotion checkpoint clears the reward_and_length/v1 gate: ..."
```

`deliverable` resolves exactly as the training notebook's `BEHAVIOR` knob
does: a recipe label names its deepest deliverable in manifest order (on the
T-Rex, `"stand"` is the recovery node, so its stance node is addressed by the
id `"stance"`), and a stage id must be a deliverable. The generator fails
closed on an unknown deliverable or label, on a backend the species does not
train, and on two entries for the same stage and backend, and it requires a
`stable-baselines3` entry for every deliverable the stage manifest declares.
A stance stage that is still gated by `reward_and_length/v1` should say so in
its definition rather than claim certified stance quality.

### Stage videos

`[[species.stage_videos]]` entries are keyed by stage id:

```toml
[[species.stage_videos]]
stage = "locomotion"                 # the stage id; a legacy integer (2) is an accepted alias
path = "website/static/videos/<species>_ppo_stage2_best.mp4"
algorithm = "PPO"
backend = "stable-baselines3"
model_revision_status = "historical"
verification_status = "unverified"
```

The id is the spelling that survives renumbering; a legacy integer resolves
through the stage manifest's `legacy_number` mapping, and naming one stage
both ways is rejected as a duplicate. A stage without a legacy number
(`recovery`) is reachable only by id.

Do not copy observation, action, or compiled-model dimensions into the
manifest. The generator derives those values from the environment and MJCF,
and reads stages and gates from the species TOML files.

From the repository root, regenerate the checked-in public data and README
blocks, then verify that nothing is stale:

```bash
python -m environments.shared.species_catalog
python -m environments.shared.species_catalog --check
```

The check also confirms that every implemented species is represented in the
manifest and that all declared configs, notebooks, videos, and summaries are
valid.
