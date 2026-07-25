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

You also need TOML config files under `configs/<species>/` for each training stage.
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

## Config Files

Create TOML configs for each training stage under `configs/<species>/`:

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

Copy an existing `[[species]]` block in `configs/species_manifest.toml` and
update its presentation metadata, environment entry point, MJCF model path,
notebook IDs, and result-summary paths. Add `[[species.success_metrics]]`
entries that state the actual success semantics for every supported backend.
Stage videos and curated result summaries are optional, but any declared
artifact must exist and include the required backend and provenance metadata.

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
