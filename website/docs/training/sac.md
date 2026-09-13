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

from environments.velociraptor.envs.raptor_env import RaptorEnv

def make_env():
    env = RaptorEnv(forward_vel_weight=1.0, alive_bonus=0.1)
    return Monitor(env)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = SAC("MlpPolicy", vec_env, learning_rate=3e-4, buffer_size=1_000_000)
# Illustrative standalone run, not a committed curriculum-stage budget.
model.learn(total_timesteps=10_000, progress_bar=True)
model.save("raptor_sac")
```

Or use the included training script:

```bash
cd environments/velociraptor

# Single stage; the current budget comes from its TOML config
python scripts/train_sb3.py train --stage 1 --algorithm sac

# The advancing stages in manifest order (per-stage hyperparameters applied automatically)
python scripts/train_sb3.py curriculum --algorithm sac

# Reuse an earlier SAC run's certified stance and walk; train hunt in a fresh directory
python scripts/train_sb3.py curriculum --algorithm sac \
  --trunk-from logs/<earlier_run> --output-dir logs/<new_run>
```

## SAC Hyperparameters

SAC settings vary by species and curriculum stage. The authoritative values are
the `[sac]` sections in the stage TOML files each species' `stages.toml` names
(`stage1_*.toml`, `stage2_*.toml`, `stage3_*.toml` for Velociraptor,
Brachiosaurus and Dibothrosuchus; `stance.toml`, `locomotion.toml`,
`behavior.toml` for T-Rex and both Compsognathus variants — the T-Rex
`recovery.toml` is PPO-only, while both Compsognathus `recovery.toml` files
carry a `[sac]` section); copied defaults here would quickly become stale.
The main fields are `learning_rate`, `batch_size`, `gamma`, `tau`, `ent_coef`,
`buffer_size`, `train_freq`, and `gradient_steps`.

## Curriculum and Recipes

SAC walks the same recipe graph as PPO. The `curriculum` command runs the
species' advancing stages in manifest order; each node warm-starts from its
declared `warm_start_from` parent's handoff checkpoint and VecNormalize
sidecar, a node whose parent has no certified checkpoint stops the run, and
each trained node writes `gate_verdict.json` beside its handoff:

1. **`stance`** (stand; root, historical stage 1): stand upright without falling (`forward_vel_weight=0`, high `alive_bonus`)
2. **`locomotion`** (walk; `warm_start_from = "stance"`, historical stage 2): walk and run forward (increase `forward_vel_weight`, add gait rewards)
3. **`behavior`** (hunt; `warm_start_from = "locomotion"`, historical stage 3): species-specific task (strike for Velociraptor, a fixed head-contact "bite" proxy for T-Rex, a head-tip distance-based food-reach proxy for Brachiosaurus, snap for Dibothrosuchus, target reach for Compsognathus)

The `recovery` node is PPO-only on the T-Rex, whose `recovery.toml` has no
`[sac]` section; both Compsognathus variants declare one. Every node is a
deliverable, so stand, walk and hunt are each certified and published on
their own. `--trunk-from RUN_DIR`, `--retrain-from STAGE_ID` and
`--label TEXT` work with `--algorithm sac` exactly as with PPO. Keep SAC
trunks for SAC runs: the notebook refuses a `TRUNK_FROM` run of a different
species, algorithm or backend outright, and the command line reuses a trunk
node only where its recorded task and plant match this run's. See
[Behavior Recipes](recipes.md).

These task names are configuration labels. T-Rex has no articulated jaw, and
Brachiosaurus success does not require physical food contact.

Within the curriculum, advancement between the numbered stages is judged in
training by the `CurriculumManager` using the thresholds in each stage's TOML
config; its verdict is what the node's `gate_verdict.json` records. Current
stage budgets and gates are shown on the
[generated model pages](/docs/models/velociraptor).

## Published Results

The generated [Velociraptor model page](/docs/models/velociraptor) displays the
available SAC summary alongside its provenance status. That historical,
unverified run is not a controlled comparison with PPO and does not establish a
general performance or training-time advantage.
