---
sidebar_position: 2
---

# Quick Start

Train your first dinosaur-inspired simulated agent.

## Option 1: Google Colab (Easiest)

Open the unified training notebook in the `notebooks/` directory:

- `notebooks/sb3_training.ipynb` - PPO or SAC training of one behavior with Stable-Baselines3. Pick a model with two dropdowns, `SPECIES` (`velociraptor_mongoliensis`, `tyrannosaurus_rex`, `brachiosaurus_altithorax`, `dibothrosuchus_elaphros` or `compsognathus_longipes`) and `EMBODIMENT` (`anatomical`, or `robot` for the Compsognathus robot), set `BEHAVIOR = "hunt"` (a recipe label `"stand"`, `"walk"` or `"hunt"`, or a deliverable's stage id) and, optionally, `TRUNK_FROM` (an earlier run whose certified stance and walk are reused), `RETRAIN_FROM` and `RUN_LABEL`. One chain-loop cell then walks the behavior's ancestor chain root-first, reusing, judging or training each node. See [Behavior Recipes](/docs/training/recipes).

The notebook handles dependency installation automatically.

## Option 2: Local Setup

```bash
# Clone and setup
git clone https://github.com/kuds/mesozoic-labs.git
cd mesozoic-labs

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package with training dependencies
pip install -e ".[train]"
```

### View the Model

```bash
python environments/velociraptor_mongoliensis/scripts/view_model.py --embodiment anatomical
```

### Train with Curriculum Learning

Every model trains from the repository root through the shared CLI,
`python -m environments.shared.train --species <species> --embodiment <embodiment>`,
which takes both flags and the canonical IDs only. The `curriculum` command
runs the model's advancing stages (stance,
locomotion, behavior) in manifest order in a single call. Each node loads its
own hyperparameters from its TOML config, warm-starts from its declared
`warm_start_from` parent's handoff checkpoint, and writes a `gate_verdict.json`
beside that handoff; a node whose parent has no certified checkpoint stops the
run. Stand, walk and hunt are each published on their own — see
[Behavior Recipes](/docs/training/recipes).

```bash
# The advancing stages in manifest order — one command
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm ppo

# Reuse an earlier run's certified stance and walk; train hunt in a fresh directory
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm ppo --trunk-from logs/<earlier_run> --output-dir logs/<new_run>

# Reuse only the earlier stance; retrain walk and hunt (requires --trunk-from)
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm ppo --trunk-from logs/<earlier_run> --retrain-from locomotion \
  --output-dir logs/<new_run>

# Or control nodes by hand; each command reads its current budget from TOML and
# writes into its --output-dir (without one:
# logs/<species>/<embodiment>/<stage_dir>_<timestamp>/ under the repository root).
# A later node enters from its declared parent's handoff under initialize_next_stage.
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  train --stage 1 --algorithm ppo --output-dir logs/<run>/01_stance
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  train --stage 2 --algorithm ppo --output-dir logs/<run>/02_locomotion \
  --load logs/<run>/01_stance/models/robust_best_model.zip \
  --load-mode initialize_next_stage
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  train --stage 3 --algorithm ppo --output-dir logs/<run>/03_behavior \
  --load logs/<run>/02_locomotion/models/robust_best_model.zip \
  --load-mode initialize_next_stage
```

`--label TEXT` tags a run (train or curriculum). Stage directories inside a run
are named `{position:02d}_{id}` (`01_stance`, `02_locomotion`, `03_behavior`);
the handoff the curriculum promotes is `robust_best_model`, else `best_model`,
with its `_vecnorm.pkl` sidecar. The default `--load-mode resume_same_stage`
requires an exact task match, so a parent's checkpoint must be loaded with
`--load-mode initialize_next_stage`, which in turn refuses a checkpoint whose
recorded stage is not the node's declared parent. A stage directory that
already holds `stage_config.json` or `gate_verdict.json` is refused unless the
load is `--load <checkpoint> --load-mode resume_same_stage`, so a new attempt is
a new run directory. Hand-chained `train` runs are unjudged — `train` writes no
`gate_verdict.json` — so they cannot serve as a later run's `--trunk-from`
until re-judged with
`python -m environments.shared.scripts.backfill_gate_verdict <stage_dir>`. The curriculum walks
to its `--target` (a recipe label, a deliverable's stage id or a legacy
number; default the last advancing stage), so `curriculum --target walk`
certifies a walk-only run on the command line as `BEHAVIOR = "walk"` does in
the notebook.

Pass `--timesteps` only when you intentionally want to override the stage
config. The generated [model pages](/docs/models/velociraptor_mongoliensis/anatomical) show the current
budgets for every model and stage.

### Evaluate a Trained Policy

```bash
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  eval logs/<stage_dir>/models/stage1_final.zip --algorithm ppo
```

### Override Hyperparameters

```bash
# Try a different learning rate without editing the TOML files
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  train --stage 1 --override ppo.learning_rate=1e-3 env.alive_bonus=3.0
```

### Run Tests

```bash
pytest -v
```

## Basic Training Loop (Python)

```python
import gymnasium as gym

# Registers every model of the species manifest in the MesozoicLabs namespace
import environments  # noqa: F401

env = gym.make("MesozoicLabs/VelociraptorMongoliensis-Anatomical-v0")

obs, info = env.reset(seed=42)
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
