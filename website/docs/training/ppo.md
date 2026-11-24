---
sidebar_position: 1
---

# PPO Training

Proximal Policy Optimization (PPO) is a policy gradient method for reinforcement learning.

## Overview

PPO is known for:

- Stable training
- Good sample efficiency
- Easy hyperparameter tuning

## Basic Usage

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

def make_env():
    env = VelociraptorMongoliensisAnatomicalEnv(forward_vel_weight=0.0, alive_bonus=1.0)
    return Monitor(env)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy", vec_env, learning_rate=3e-4)
# Illustrative standalone run, not a committed curriculum-stage budget.
model.learn(total_timesteps=10_000, progress_bar=True)
model.save("velociraptor_mongoliensis_anatomical_stance")
```

Or use the shared training CLI from the repository root with curriculum learning:

```bash
# Single stage; the current budget comes from its TOML config
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  train --stage 1 --algorithm ppo

# The advancing stages in manifest order (per-stage hyperparameters applied automatically)
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm ppo

# Reuse an earlier run's certified stance; train walk in a fresh directory
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm ppo --trunk-from logs/<earlier_run> --output-dir logs/<new_run>
```

## PPO Hyperparameters

PPO settings vary by model and curriculum stage. The authoritative values are
the `[ppo]` sections in the stage TOML files each model's `stages.toml` names,
under `configs/<species>/<embodiment>/` and each named by its stage
(`stance.toml` and `locomotion.toml` on every model, and `recovery.toml` for
the T-Rex and both Compsognathus models);
copied defaults here would quickly become stale. The main fields are
`learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`,
`clip_range`, and `ent_coef`.

## Curriculum and Recipes

The `curriculum` command runs the model's advancing stages in manifest order,
up to its `--target` (a recipe label, a deliverable's stage id or a legacy
number; default the last advancing stage, so `--target walk` stops at walk).
Each node warm-starts from its declared `warm_start_from` parent's handoff
checkpoint and VecNormalize sidecar, a node whose parent has no certified
checkpoint stops the run, and each trained node writes `gate_verdict.json`
beside its handoff. The nodes, by id and recipe label, are the same for PPO
and SAC:

1. **`stance`** (stance; root, historical stage 1): stand upright without falling (`forward_vel_weight=0`, high `alive_bonus`)
2. **`locomotion`** (walk; `warm_start_from = "stance"`, historical stage 2): follow a forward command with heading hold, and stand still on the exactly zero command of about 15% of episodes (a live `[commands]` table with a stand block and a cruise ramp; the tracking term replaces the forward, heading and sideways weights; judged after training by `command_tracking/v1`, decision D-D62)

T-Rex, Compsognathus and the Compsognathus robot add a third, non-advancing
node, **`recovery`** (recover; `warm_start_from = "stance"`): balance under
scheduled pushes. It has no legacy number, the CLI curriculum skips it with a
log line, and it is trained on its own with `train --stage recovery`. Each
`recovery.toml` extends its model's `stance.toml` (a top-level
`extends = { stage = "stance", tables = [...] }`): it inherits the listed
tables, changing only the stage's name and description, the push settings
and the entropy-decay horizon. The T-Rex `recovery.toml` is PPO-only (it
does not inherit stance's `[sac]`); both Compsognathus models inherit it.
Every node is a deliverable: stance and walk (and recover, on the three
models with a recovery node) are each certified and published on their own.
The hunt node that followed walk (`behavior`, historical stage 3) was removed
by decision D-D59, so every chain ends at walk. `--trunk-from RUN_DIR`,
`--retrain-from STAGE_ID` and `--label TEXT` let a run reuse an earlier run's
certified trunk, retrain from a chosen node, and tag its nodes. See
[Behavior Recipes](recipes.md).

Every stage holds the statistics of the observation's last three inputs, the
command, at mean 0 / variance 1 while it trains, so the policy reads the raw
command and every VecNormalize sidecar carries that identity (decision D-D60;
[Hyperparameters](hyperparameters.md#commands)). A stance, whose command is
always zero, trains exactly as it would without the hold.

Within the curriculum, advancement between the numbered stages is judged in
training by the `CurriculumManager` using the thresholds in each stage's TOML
config; its verdict is what the node's `gate_verdict.json` records. Current
stage budgets and gates are shown on the
[generated model pages](/docs/models/velociraptor_mongoliensis/anatomical).

## Published Results

Published PPO summaries are displayed on the
[Velociraptor](/docs/models/velociraptor_mongoliensis/anatomical), [T-Rex](/docs/models/tyrannosaurus_rex/anatomical), and
[Brachiosaurus](/docs/models/brachiosaurus_altithorax/anatomical) pages from the generated catalog.
The available summaries are historical and unverified, so they do not establish
that PPO is faster, slower, better, or worse than SAC under controlled conditions.
