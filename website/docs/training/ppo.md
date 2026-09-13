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

from environments.velociraptor.envs.raptor_env import RaptorEnv

def make_env():
    env = RaptorEnv(forward_vel_weight=0.0, alive_bonus=1.0)
    return Monitor(env)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy", vec_env, learning_rate=3e-4)
# Illustrative standalone run, not a committed curriculum-stage budget.
model.learn(total_timesteps=10_000, progress_bar=True)
model.save("raptor_stage1")
```

Or use the included training script with curriculum learning:

```bash
cd environments/velociraptor

# Single stage; the current budget comes from its TOML config
python scripts/train_sb3.py train --stage 1 --algorithm ppo

# The advancing stages in manifest order (per-stage hyperparameters applied automatically)
python scripts/train_sb3.py curriculum --algorithm ppo

# Reuse an earlier run's certified stance and walk; train hunt in a fresh directory
python scripts/train_sb3.py curriculum --algorithm ppo \
  --trunk-from logs/<earlier_run> --output-dir logs/<new_run>
```

## PPO Hyperparameters

PPO settings vary by species and curriculum stage. The authoritative values are
the `[ppo]` sections in the stage TOML files each species' `stages.toml` names
(`stage1_*.toml`, `stage2_*.toml`, `stage3_*.toml` for Velociraptor,
Brachiosaurus and Dibothrosuchus; `stance.toml`, `recovery.toml`,
`locomotion.toml`, `behavior.toml` for T-Rex and both Compsognathus variants);
copied defaults here would quickly become stale. The main fields are
`learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`,
`clip_range`, and `ent_coef`.

## Curriculum and Recipes

The `curriculum` command runs the species' advancing stages in manifest order,
up to its `--target` (a recipe label, a deliverable's stage id or a legacy
number; default the last advancing stage, so `--target walk` stops at walk).
Each node warm-starts from its declared `warm_start_from` parent's handoff
checkpoint and VecNormalize sidecar, a node whose parent has no certified
checkpoint stops the run, and each trained node writes `gate_verdict.json`
beside its handoff. The nodes, by id and recipe label, are the same for PPO
and SAC:

1. **`stance`** (stand; root, historical stage 1): stand upright without falling (`forward_vel_weight=0`, high `alive_bonus`)
2. **`locomotion`** (walk; `warm_start_from = "stance"`, historical stage 2): walk and run forward (increase `forward_vel_weight`, add gait rewards)
3. **`behavior`** (hunt; `warm_start_from = "locomotion"`, historical stage 3): species-specific task (strike for Velociraptor, a fixed head-contact "bite" proxy for T-Rex, a head-tip distance-based food-reach proxy for Brachiosaurus, snap for Dibothrosuchus, target reach for Compsognathus)

T-Rex, Compsognathus and the Compsognathus robot add a fourth, non-advancing
node, **`recovery`** (stand; `warm_start_from = "stance"`): balance under
scheduled pushes. It has no legacy number, the CLI curriculum skips it with a
log line, and it is trained on its own with `train --stage recovery`. The
T-Rex `recovery.toml` is PPO-only (it has no `[sac]` section); both
Compsognathus variants declare one.
Every node is a deliverable: stand, walk and hunt are each certified and
published on their own, and `--trunk-from RUN_DIR`, `--retrain-from STAGE_ID`
and `--label TEXT` let a run reuse an earlier run's certified trunk, retrain
from a chosen node, and tag its nodes. See [Behavior Recipes](recipes.md).

These task names are configuration labels. T-Rex has no articulated jaw, and
Brachiosaurus success does not require physical food contact.

Within the curriculum, advancement between the numbered stages is judged in
training by the `CurriculumManager` using the thresholds in each stage's TOML
config; its verdict is what the node's `gate_verdict.json` records. Current
stage budgets and gates are shown on the
[generated model pages](/docs/models/velociraptor).

## Published Results

Published PPO summaries are displayed on the
[Velociraptor](/docs/models/velociraptor), [T-Rex](/docs/models/trex), and
[Brachiosaurus](/docs/models/brachiosaurus) pages from the generated catalog.
The available summaries are historical and unverified, so they do not establish
that PPO is faster, slower, better, or worse than SAC under controlled conditions.
