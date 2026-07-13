---
sidebar_position: 5
---

# JAX/MJX Training

JAX/MJX is the GPU-oriented training backend for Mesozoic Labs. It is designed to run many parallel environments on a single NVIDIA GPU using [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and trains with a pure-JAX PPO implementation built on [Flax](https://flax.readthedocs.io/) and [Optax](https://optax.readthedocs.io/).

## Overview

The JAX backend provides:

- **Configurable batched simulation** -- parallel environments via `jax.vmap`, sized to the target hardware
- **CUDA acceleration** -- the documented accelerated path targets NVIDIA GPUs
- **Three-stage task sequence** -- balance, locomotion, then a species-specific simulator task
- **All three species** -- T-Rex, Velociraptor, and Brachiosaurus

CPU execution can be useful for smoke tests, but large training runs are intended for a CUDA GPU. TPU execution has not been validated by this project and is not part of the documented setup.

## When to Use JAX vs SB3

| | SB3 (CPU) | JAX/MJX (GPU) |
|---|---|---|
| **Install** | `pip install -e ".[train]"` | `pip install -e ".[jax]"` |
| **Hardware** | Any CPU | NVIDIA CUDA GPU (CPU for small smoke tests) |
| **Parallelism** | Process-based vector environments | Batched environments via `jax.vmap` |
| **Algorithm** | PPO or SAC | PPO (JAX-native) |
| **Best for** | Quick experiments, debugging, no-GPU setups | Large-scale training, hyperparameter sweeps |

Both backends consume the same MJCF assets and TOML stage files. They do not
currently have identical success proxies or curriculum-gate behavior; those
differences are described below.

## Installation

Install the JAX optional dependencies:

```bash
pip install -e ".[jax]"
```

This installs `mujoco-mjx`, `jax[cuda12]`, `flax`, and `optax`. The JAX backend uses lazy imports, so the rest of the codebase works fine without these packages installed.

## Basic Usage

### Config-aware CLI

```bash
# Single stage
python -m environments.shared.jax_training --species trex --stage 1

# Config-driven staged run (stops if a post-stage gate fails)
python -m environments.shared.jax_training --species trex --curriculum
```

The CLI loads the selected stage TOML. The curriculum command also carries
policy and observation-normalization state between stages.

### Python API

For a config-driven Python run, use the curriculum wrapper, which loads and
maps each stage's `[jax]` and `[env]` sections before calling `train_jax`:

```python
from environments.shared.jax_curriculum import run_curriculum
from environments.shared.jax_training import train_jax

results = run_curriculum(
    species="trex",
    train_fn=train_jax,
    seed=42,
)
```

`train_jax(...)` itself is a low-level function: a direct call uses its Python
arguments and built-in defaults and does **not** load the stage TOML. Direct
callers must explicitly pass every desired training argument plus `env_kwargs`,
or reproduce the mapping performed by the CLI/curriculum wrapper.

### Colab Notebook

The `notebooks/jax_training.ipynb` notebook supports all three species. Set the `SPECIES` variable at the top of the notebook:

```python
SPECIES = "trex"  # or "velociraptor" or "brachiosaurus"
```

The notebook handles dependency installation, GPU detection, and stage-config
loading. Stage progression is manual: after evaluating the current stage,
change `CURRENT_STAGE` and rerun the stage cells.

## Architecture

The JAX path mirrors the SB3 path but replaces Python-level loops with JIT-compiled JAX functions:

```
MJXDinoEnv(species="trex", stage=1)
  +-- mjx.put_model(mj_model)       # GPU-resident physics model
  +-- jax.vmap(step_fn)             # Configured environment batch
       +-- JAX PPO (Flax+Optax)    # Pure-JAX policy + optimizer
            +-- Running-mean norm   # JAX-based observation normalization
```

### Key Components

| Module | Description |
|---|---|
| `environments/shared/mjx_env.py` | JAX-native batched environment with functional `step`/`reset` |
| `environments/shared/jax_ppo.py` | Flax `ActorCritic` network with PPO loss, GAE, and update functions |
| `environments/shared/jax_trainer.py` | High-level training loop with hooks for logging and checkpointing |
| `environments/shared/jax_training.py` | Low-level training function and config-aware CLI entry point |
| `environments/shared/jax_normalization.py` | Running-mean observation normalization (equivalent to SB3's `VecNormalize`) |
| `environments/shared/jax_curriculum.py` | Config-driven stage runner with a reward-only post-stage gate |
| `environments/shared/jax_eval.py` | Policy evaluation, metric collection, and video generation |
| `environments/shared/jax_viz.py` | Trajectory rendering and visualization tools |

### Dual-Backend Design

The SB3 and JAX paths share:

- **MJCF model files** (`*.xml`) -- identical physics models, no changes needed
- **Many reward primitives** -- pure functions are reused across NumPy and JAX, while backend-specific wiring still differs
- **Stage config inputs** (`configs/*/stage*.toml`) -- the config-aware JAX paths read the same files
- **Evaluation rendering** -- CPU MuJoCo rendering for both (MJX has no native renderer)

They differ in:

- **Environment wrapper** -- Gymnasium `step()`/`reset()` vs. JAX functional `step_fn`/`reset_fn`
- **Training loop** -- SB3 callbacks vs. JIT-compiled JAX rollout+update
- **Parallelism** -- process-based vector environments vs. a configurable `jax.vmap` batch
- **Stage-3 success** -- Gym/SB3 uses collision contacts for Velociraptor and T-Rex; MJX uses site-to-target distance thresholds
- **Advancement checks** -- the JAX paths use post-stage checks rather than SB3's consecutive in-training evaluations

## PPO Hyperparameters (JAX)

The `[jax]` section in each stage TOML is authoritative. Values vary by species
and stage, so this guide lists the config surface without copying values that
can drift:

| Area | Keys consumed by the CLI/curriculum wrapper |
|---|---|
| Batch and duration | `num_envs`, `rollout_len`, `num_updates`, `minibatch_size`, `ppo_epochs` |
| Optimisation | `learning_rate`, `learning_rate_end`, `max_grad_norm`, `target_kl` |
| PPO objective | `gamma`, `gae_lambda`, `clip_range`, `vf_clip_range`, `ent_coef`, `vf_coef` |
| Transition schedule | `warmup_updates`, `warmup_clip_range`, `warmup_ent_coef`, `ramp_updates`, `ramp_start_fraction` |
| Environment overrides | `fall_penalty`, `reset_noise_scale`, `init_qpos_noise`, `init_yaw_noise` |

The notebook additionally reads `ramp_attr` and
`obs_rms_decay_on_resume`. The committed `[jax.policy_kwargs]` tables are not
currently wired into either JAX network-construction path, so do not treat them
as effective settings. Use the CLI or `run_curriculum` for the mapped keys
above. A direct `train_jax` call does not automatically apply them.

## Three-Stage Task Sequence

The JAX and SB3 paths use the same high-level sequence:

1. **Stage 1 -- Balance**: Stand upright without falling (`forward_vel_weight=0`, high `alive_bonus`)
2. **Stage 2 -- Locomotion**: Walk and run forward (increase `forward_vel_weight`, add gait rewards)
3. **Stage 3 -- Behavior**: Species-specific simulated task (strike, bite proxy, or food reach)

Stage 3 is not behaviorally identical across backends:

| Species | Gym/SB3 success proxy | JAX/MJX success proxy |
|---|---|---|
| Velociraptor | A sickle-claw geom collides with the prey geom | A claw-tip site falls within a configured distance of the prey target |
| T-Rex | A fixed head geom collides with the prey geom | The `head_tip` site falls within a configured distance of the prey target |
| Brachiosaurus | The head tip falls within the configured food-reach distance | The head-tip site falls within the configured food-reach distance |

These names describe simulator proxies. T-Rex has no articulated jaw, and
Brachiosaurus success does not require physical food contact.

Gate behavior also differs:

- `python -m environments.shared.jax_training --curriculum` uses
  `jax_curriculum.run_curriculum`. It trains the configured update budget, then
  performs one post-stage `min_avg_reward` check. It does not check episode
  length, forward velocity, success rate, minimum evaluation episodes, or
  consecutive passes.
- `notebooks/jax_training.ipynb` calls `jax_setup.run_stage_evaluation`. That
  helper checks the enabled reward, episode-length, forward-velocity, and
  success-rate thresholds once over the requested evaluation episodes. It does
  not require consecutive passing evaluations.
- The SB3 curriculum evaluates during training and can advance early only after
  all enabled criteria pass for the required number of consecutive evaluations.

Treat a JAX gate result as backend-specific evidence, not as an SB3-equivalent
curriculum pass.

## Training Logs & Metrics

There is no W&B integration on the JAX path (that is SB3-only for now).
Instead, pass `checkpoint_dir` (or `--checkpoint-dir` on the CLI) and the
trainer writes durable artifacts alongside the checkpoints:

- `<species>_s<stage>_training_log.csv` — per-update metrics (reward,
  episode return/length, losses, KL, gradient norm, fall rate, FPS)
- `<species>_s<stage>_best.pkl` — the best-episode-return parameters, so a
  late-training regression can't cost you the strongest policy

## Checkpointing

Model parameters, optimizer state, and running normalization statistics are
saved periodically during training (rotating checkpoints plus a final one).

```python
# Save checkpoints + training CSV + best-model snapshot
train_jax(
    species="trex",
    stage=1,
    checkpoint_dir="results/trex/jax/stage1",
)

# Resume / warm-start from a saved checkpoint
from environments.shared.jax_checkpoint import restore_train_state

params, opt_state, obs_rms, update = restore_train_state(
    "results/trex/jax/stage1/trex_s1_00450.pkl"
)
params, metrics, obs_stats = train_jax(
    species="trex",
    stage=2,
    init_params=params,
    init_obs_stats=obs_rms,
    checkpoint_dir="results/trex/jax/stage2",
)
```

These checkpoint examples call the low-level API directly. Supply the desired
stage TOML values explicitly (including `env_kwargs`) for a config-matched run.

## GPU Memory Sizing

The number of parallel environments is a major factor in GPU memory use, but the model, species, rollout length, JAX version, and compilation strategy also matter. Start with a conservative batch, measure memory on the target GPU, and increase `num_envs` gradually. The project does not yet publish a validated cross-hardware memory table, so do not assume the committed value fits every GPU and species.

## JIT Compilation

The first training step triggers JAX's JIT compilation, which can take several minutes depending on model complexity, batch size, software versions, and hardware. This one-time pause before steady-state execution is expected behavior.

## Vertex AI with JAX

The following is an illustrative A100 Vertex AI job. Validate the machine type
and environment batch with a short pilot before launching a full run:

```python
job = aiplatform.CustomJob(
    display_name="trex-jax-curriculum",
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": "a2-highgpu-1g",
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE_URI,
                "command": ["python"],
                "args": [
                    "-m", "environments.shared.jax_training",
                    "--species", "trex",
                    "--curriculum",
                ],
            },
        }
    ],
)
job.run(sync=False)
```

See [Training on Vertex AI](vertex-ai.md) for full setup instructions.

## Comparing JAX with SB3

The JAX backend is designed to improve throughput through batched, compiled simulation, while SB3 remains the simpler CPU path for debugging and smaller experiments. The repository does not yet publish a controlled, reproducible cross-backend benchmark, and it should not be assumed that the two implementations reach identical final performance. Compare them with the same species, stage configuration, seed set, evaluation protocol, and hardware description, and report both throughput and policy quality.
