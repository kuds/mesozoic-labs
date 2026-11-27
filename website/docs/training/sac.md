---
sidebar_position: 2
---

# SAC Training

Soft Actor-Critic (SAC) is an off-policy algorithm that optimizes a stochastic policy with entropy regularization.

## Overview

SAC is known for:

- Sample efficiency
- Automatic temperature tuning
- Stable exploration

## Basic Usage

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

def make_env():
    env = VelociraptorMongoliensisAnatomicalEnv(forward_vel_weight=1.0, alive_bonus=0.1)
    return Monitor(env)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = SAC("MlpPolicy", vec_env, learning_rate=3e-4, buffer_size=1_000_000)
# Illustrative standalone run, not a committed curriculum-stage budget.
model.learn(total_timesteps=10_000, progress_bar=True)
model.save("velociraptor_mongoliensis_anatomical_sac")
```

Or use the shared training CLI from the repository root:

```bash
# Single stage; the current budget comes from its TOML config
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  train --stage 1 --algorithm sac

# The advancing stages in manifest order (per-stage hyperparameters applied automatically)
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm sac

# Reuse an earlier SAC run's certified stance; train walk in a fresh directory
python -m environments.shared.train --species velociraptor_mongoliensis --embodiment anatomical \
  curriculum --algorithm sac --trunk-from logs/<earlier_run> --output-dir logs/<new_run>
```

## SAC Hyperparameters

SAC settings vary by model and curriculum stage. The authoritative values are
the `[sac]` sections in the stage TOML files each model's `stages.toml` names,
under `configs/<species>/<embodiment>/` and each named by its stage
(`stance.toml` and `locomotion.toml` on every model; the T-Rex
`recovery.toml` is PPO-only, while both Compsognathus `recovery.toml` files
inherit their `stance.toml`'s `[sac]` through `extends`); copied defaults
here would quickly become stale.
The main fields are `learning_rate`, `batch_size`, `gamma`, `tau`, `ent_coef`,
`buffer_size`, `train_freq`, and `gradient_steps`.

## Curriculum and Recipes

SAC walks the same recipe graph as PPO. The `curriculum` command runs the
model's advancing stages in manifest order up to its `--target` (default
the last advancing stage; `--target walk` stops at walk); each node warm-starts from its
declared `warm_start_from` parent's handoff checkpoint and VecNormalize
sidecar, a node whose parent has no certified checkpoint stops the run, and
each trained node writes `gate_verdict.json` beside its handoff:

1. **`stance`** (stance; root, historical stage 1): stand upright without falling (`forward_vel_weight=0`, high `alive_bonus`)
2. **`locomotion`** (walk; `warm_start_from = "stance"`, historical stage 2): follow a forward command with heading hold, and stand still on the exactly zero command of about 15% of episodes (a live `[commands]` table with a stand block and a cruise ramp; the tracking term replaces the forward, heading and sideways weights; judged after training by `command_tracking/v1`, decision D-D62, whose bars are provisional: no controller has yet passed a committed walk panel, D-D62 (16); a moving command is scored on a one-stride trailing mean since D-D64)

The `recovery` node is PPO-only on the T-Rex, whose `recovery.toml` does
not inherit stance's `[sac]`; both Compsognathus models do. Every node is a
deliverable, so stance and walk (and recover, where a model has a
recovery node) are each certified and published on their own. The hunt
node that followed walk (`behavior`, historical stage 3) was removed by
decision D-D59, so every chain ends at walk. `--trunk-from RUN_DIR`, `--retrain-from STAGE_ID` and
`--label TEXT` work with `--algorithm sac` exactly as with PPO. Keep SAC
trunks for SAC runs: the notebook refuses a `TRUNK_FROM` run of a different
model (species or embodiment), algorithm or backend outright, and the command line reuses a trunk
node only where its recorded task and plant match this run's. See
[Behavior Recipes](recipes.md).

Every stage holds the statistics of the observation's last three inputs, the
command, at mean 0 / variance 1 while it trains, so the policy reads the raw
command and every VecNormalize sidecar carries that identity (decision D-D60;
[Hyperparameters](hyperparameters.md#commands)). The replay buffer still
stores raw observations, terminal ones included, and normalizes them when it
samples. A stance, whose command is always zero, trains exactly as it would
without the hold.

Within the curriculum, advancement between the numbered stages is judged in
training by the `CurriculumManager` using the thresholds in each stage's TOML
config; its verdict is what the node's `gate_verdict.json` records. Current
stage budgets and gates are shown on the
[generated model pages](/docs/models/velociraptor_mongoliensis/anatomical).

## Published Results

The generated [Velociraptor model page](/docs/models/velociraptor_mongoliensis/anatomical) displays the
available SAC summary alongside its provenance status. That historical,
unverified run is not a controlled comparison with PPO and does not establish a
general performance or training-time advantage.
