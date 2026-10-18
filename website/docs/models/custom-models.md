---
sidebar_position: 99
---

# Custom Models

Learn how to create your own dinosaur models for Mesozoic Labs.

## Architecture Overview

A model is a species and an embodiment (`anatomical`, the animal-proportioned
model, or `robot`, a buildable robot of the species). Every species in
Mesozoic Labs follows the same structure, with one folder per embodiment:

```
environments/<species>/
├── __init__.py
├── base_env.py                # <Species>Env (subclasses BaseDinoEnv): observation, action, rewards
├── <embodiment>/
│   ├── __init__.py
│   ├── env.py                 # <Species><Embodiment>Env: the embodiment's body data
│   └── assets/
│       └── <species>_<embodiment>.xml   # MuJoCo MJCF model
├── scripts/
│   └── view_model.py          # Model viewer (takes --embodiment)
└── tests/
    └── test_<embodiment>_env.py  # Environment tests
```

You also need a stage manifest and TOML config files under
`configs/<species>/<embodiment>/` for each training stage (see
[Stage manifest](#stage-manifest) below). Every implemented model must also
have an entry in `configs/species_manifest.toml`; the generated public catalog
and its CI drift check use that manifest as their source of truth. Every model
trains through the shared CLI,
`python -m environments.shared.train --species <species> --embodiment <embodiment> ...`;
there is no per-species training script.

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

Use the generated specifications on the [T-Rex](./tyrannosaurus_rex/anatomical),
[Velociraptor](./velociraptor_mongoliensis/anatomical), [Brachiosaurus](./brachiosaurus_altithorax/anatomical), and
[Dibothrosuchus](./dibothrosuchus_elaphros/anatomical) model pages for the current compiled
dimensions. Look at the corresponding MJCF
files under `environments/<species>/<embodiment>/assets/` for detailed examples.

## Creating the Environment

Subclass `BaseDinoEnv` once per species, in `environments/<species>/base_env.py`,
and implement the species' observation, action and reward code there; each
embodiment subclasses that species base in `environments/<species>/<embodiment>/env.py`
with its body data:

```python
# environments/custom_dino/base_env.py
from environments.shared.base_env import BaseDinoEnv

class CustomDinoEnv(BaseDinoEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path=str(self.MODEL_PATH), **kwargs)


# environments/custom_dino/anatomical/env.py
from pathlib import Path

from environments.custom_dino.base_env import CustomDinoEnv
from environments.shared.model_id import ModelId

class CustomDinoAnatomicalEnv(CustomDinoEnv):
    embodiment = "anatomical"
    MODEL_PATH = ModelId("custom_dino", "anatomical").asset_path(Path(__file__).resolve().parents[2])
    # plus FOOT_SUPPORT and REWARD_PROBE, as on the existing embodiment classes
```

Gymnasium registration needs no code: `import environments` registers every
model of `configs/species_manifest.toml` as `MesozoicLabs/<Species>-<Embodiment>-v0`,
here `MesozoicLabs/CustomDino-Anatomical-v0`, with the entry point its
manifest table names, `environments.custom_dino.anatomical.env:CustomDinoAnatomicalEnv`.

## Stage Manifest

Create `configs/<species>/<embodiment>/stages.toml` declaring the model's recipe graph:
one `[[stages]]` entry per node, in an order where every parent precedes its
children. The committed Velociraptor manifest is the three-node template:

```toml
schema = "mesozoic.stage-manifest/v2"

[[stages]]                       # quiet stance; gate reward-cleared by the statue (plan §4.8)
id = "stance"
config = "stance.toml"
legacy_number = 1
recipe = "stance"
deliverable = true

[[stages]]
id = "locomotion"
config = "locomotion.toml"
legacy_number = 2
warm_start_from = "stance"
recipe = "walk"
deliverable = true

[[stages]]                       # id stays "behavior"; the task name (strike) is the TOML's [stage] name
id = "behavior"
config = "strike.toml"
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
and `recipe` is the behavior label (`stance`, `recover`, `walk`, `hunt`), equal
to a stage id only when it resolves to that stage (a label naming another
stage's id is fatal). A model with a recovery node uses the four-entry T-Rex
layout, where `recovery` (recipe `recover`, `warm_start_from = "stance"`, no
legacy number) sits between stance and locomotion; `stance` labels the stance
node alone on every model (decisions D-D35 and D-D38). `stand`, the stance
label until D-D38, is refused at load, naming `stance`. See [Behavior Recipes](/docs/training/recipes) for how the
manifest drives training and publication.

## Config Files

Create the TOML configs the manifest names, one per training stage, under
`configs/<species>/<embodiment>/`, each named by its stage with no number
(`stance.toml`, `locomotion.toml`, `strike.toml`, ...). The stance stage's
`stance.toml`:

```toml
[stage]
name = "stance"
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

## Add the Model to the Public Catalog

`configs/species_manifest.toml` (schema version 3) has one table per species
and one per embodiment. Add the species table and its first embodiment table,
copying an existing pair:

```toml
[species.custom_dino]
display_order = 7
display_name = "Custom Dino"

[species.custom_dino.embodiments.anatomical]
tagline = "Small Bipedal Theropod"
gait = "Bipedal"
specialty = "Non-contact target reaching"
env_entrypoint = "environments.custom_dino.anatomical.env:CustomDinoAnatomicalEnv"
model_path = "environments/custom_dino/anatomical/assets/custom_dino_anatomical.xml"
training_backends = ["stable-baselines3"]
training_notebooks = ["sb3_training"]
result_summary_paths = []
```

Update the presentation metadata, environment entry point, MJCF model path,
notebook IDs, and result-summary paths. Add
`[[species.<species>.embodiments.<embodiment>.success_metrics]]` entries that state the actual success semantics
for every supported backend. Stage videos and curated result summaries are
optional, but any declared artifact must exist and include the required
backend and provenance metadata.

### Per-deliverable success semantics

Every behavior the model publishes — each stage its `configs/<species>/<embodiment>/stages.toml`
flags `deliverable = true` — needs a `[[species.<species>.embodiments.<embodiment>.deliverable_metrics]]` entry
stating what "success" means for that checkpoint:

```toml
[[species.custom_dino.embodiments.anatomical.deliverable_metrics]]
deliverable = "walk"                 # a recipe label, or a deliverable stage id
backends = ["stable-baselines3"]
key = "forward_velocity_gate"
label = "Gated forward velocity (reward_and_length/v1)"
definition = "The locomotion checkpoint clears the reward_and_length/v1 gate: ..."
```

`deliverable` resolves exactly as the training notebook's `BEHAVIOR` knob
does: a recipe label names its deepest deliverable in manifest order (`"stance"`
the stance node and `"recover"` the recovery node, so every committed model
names its deliverables by label), and a stage id must be a deliverable. The generator fails
closed on an unknown deliverable or label, on a backend the model does not
train, and on two entries for the same stage and backend, and it requires a
`stable-baselines3` entry for every deliverable the stage manifest declares.
A stance stage that is still gated by `reward_and_length/v1` should say so in
its definition rather than claim certified stance quality.

### Stage videos

`[[species.<species>.embodiments.<embodiment>.stage_videos]]` entries are keyed by stage id:

```toml
[[species.custom_dino.embodiments.anatomical.stage_videos]]
stage = "locomotion"                 # the stage id; a legacy integer (2) is an accepted alias
path = "website/static/videos/<species>_<embodiment>_ppo_stage2_best.mp4"
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
and reads stages and gates from the model's TOML files.

From the repository root, regenerate the checked-in public data (the site's
JSON, the root README's blocks and `docs/SPECIES_CATALOG.md`), then verify
that nothing is stale:

```bash
python -m environments.shared.species_catalog
python -m environments.shared.species_catalog --check
```

The check also confirms that every implemented model is represented in the
manifest and that all declared configs, notebooks, videos, and summaries are
valid.
