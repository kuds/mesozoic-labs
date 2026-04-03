---
sidebar_position: 5
---

# JAX/MJX Training

JAX/MJX is the GPU-accelerated training backend for Mesozoic Labs. It runs thousands of parallel environments on a single GPU using [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and trains with a pure-JAX PPO implementation built on [Flax](https://flax.readthedocs.io/) and [Optax](https://optax.readthedocs.io/).

## Overview

The JAX backend provides:
- **Massive parallelism** -- 2,048-8,192 environments running simultaneously via `jax.vmap`
- **10-100x speedup** over the SB3/CPU path for large-scale training
- **GPU/TPU support** -- works on NVIDIA GPUs and Google Cloud TPUs
- **Same curriculum** -- identical 3-stage curriculum learning as the SB3 path
- **All three species** -- T-Rex, Velociraptor, and Brachiosaurus

## When to Use JAX vs SB3

| | SB3 (CPU) | JAX/MJX (GPU) |
|---|---|---|
| **Install** | `pip install -e ".[train]"` | `pip install -e ".[jax]"` |
| **Hardware** | Any CPU | NVIDIA GPU or TPU |
| **Parallelism** | 4-32 envs (`SubprocVecEnv`) | 2,048-8,192 envs (`jax.vmap`) |
| **Algorithm** | PPO or SAC | PPO (JAX-native) |
| **Best for** | Quick experiments, debugging, no-GPU setups | Large-scale training, hyperparameter sweeps |

Both backends share the same MJCF model files, TOML stage configs, and reward logic.

## Installation

Install the JAX optional dependencies:

```bash
pip install -e ".[jax]"
```

This installs `mujoco-mjx`, `jax[cuda12]`, `flax`, and `optax`. The JAX backend uses lazy imports, so the rest of the codebase works fine without these packages installed.

## Basic Usage

### Python API

```python
from environments.shared.jax_training import train_jax

# Train T-Rex Stage 1 (balance) with 2048 parallel environments
train_jax(species="trex", stage=1, num_envs=2048, seed=42)
```

### CLI

```bash
# Single stage
python -m environments.shared.jax_training --species trex --stage 1

# Full 3-stage curriculum
python -m environments.shared.jax_training --species trex --curriculum
```

### Colab Notebook

The `notebooks/jax_training.ipynb` notebook supports all three species. Set the `SPECIES` variable at the top of the notebook:

```python
SPECIES = "trex"  # or "velociraptor" or "brachiosaurus"
```

The notebook handles dependency installation, GPU/TPU detection, and curriculum training automatically.

## Architecture

The JAX path mirrors the SB3 path but replaces Python-level loops with JIT-compiled JAX functions:

```
MJXDinoEnv(species="trex", stage=1)
  +-- mjx.put_model(mj_model)       # GPU-resident physics model
  +-- jax.vmap(step_fn)             # 2048+ parallel environments
       +-- JAX PPO (Flax+Optax)    # Pure-JAX policy + optimizer
            +-- Running-mean norm   # JAX-based observation normalization
```

### Key Components

| Module | Description |
|---|---|
| `environments/shared/mjx_env.py` | JAX-native batched environment with functional `step`/`reset` |
| `environments/shared/jax_ppo.py` | Flax `ActorCritic` network with PPO loss, GAE, and update functions |
| `environments/shared/jax_trainer.py` | High-level training loop with hooks for logging and checkpointing |
| `environments/shared/jax_training.py` | Entry point that loads TOML configs and runs the training pipeline |
| `environments/shared/jax_normalization.py` | Running-mean observation normalization (equivalent to SB3's `VecNormalize`) |
| `environments/shared/jax_curriculum.py` | Curriculum stage management with threshold-based advancement |
| `environments/shared/jax_eval.py` | Policy evaluation, metric collection, and video generation |
| `environments/shared/jax_viz.py` | Trajectory rendering and visualization tools |

### Dual-Backend Design

The SB3 and JAX paths share:
- **MJCF model files** (`*.xml`) -- identical physics models, no changes needed
- **Reward logic** -- extracted into pure functions that work with both NumPy and JAX arrays
- **Stage configs** (`configs/*/stage*.toml`) -- same TOML files drive both backends
- **Evaluation rendering** -- CPU MuJoCo rendering for both (MJX has no native renderer)

They differ in:
- **Environment wrapper** -- Gymnasium `step()`/`reset()` vs. JAX functional `step_fn`/`reset_fn`
- **Training loop** -- SB3 callbacks vs. JIT-compiled JAX rollout+update
- **Parallelism** -- `SubprocVecEnv` (4-32 envs) vs. `jax.vmap` (2,048-8,192 envs)

## PPO Hyperparameters (JAX)

The JAX PPO uses the same hyperparameter progression as the SB3 path, loaded from the per-species TOML configs:

| Parameter | Stage 1 | Stage 2 | Stage 3 | Description |
|---|---|---|---|---|
| `learning_rate` | `3e-4` | `1e-4` | `5e-5` | Network learning rate |
| `num_envs` | 2048 | 2048 | 2048 | Parallel environments (via `jax.vmap`) |
| `rollout_len` | 64 | 64 | 64 | Steps per rollout before update |
| `num_minibatches` | 4 | 4 | 4 | Minibatches per update epoch |
| `update_epochs` | 10 | 10 | 10 | Epochs per PPO update |
| `gamma` | 0.99 | 0.99 | 0.995 | Discount factor |
| `gae_lambda` | 0.95 | 0.95 | 0.95 | GAE lambda |
| `clip_range` | 0.2 | 0.2 | 0.1 | PPO clip range |
| `ent_coef` | 0.01 | 0.01 | 0.001 | Entropy coefficient |
| `vf_coef` | 0.5 | 0.5 | 0.5 | Value function coefficient |

With 2,048 environments and a rollout length of 64, each update uses 131,072 transitions -- far more data per update than the SB3 path.

## 3-Stage Curriculum

The JAX path follows the same curriculum as SB3:

1. **Stage 1 -- Balance**: Stand upright without falling (`forward_vel_weight=0`, high `alive_bonus`)
2. **Stage 2 -- Locomotion**: Walk and run forward (increase `forward_vel_weight`, add gait rewards)
3. **Stage 3 -- Behavior**: Species-specific task (strike for Velociraptor, bite for T-Rex, food reach for Brachiosaurus)

Stage transitions use the same threshold-based advancement as the SB3 path. The `jax_curriculum.py` module checks evaluation metrics against the TOML-defined thresholds and advances when criteria are met for consecutive evaluations.

## W&B Integration

Enable Weights & Biases logging by passing `wandb_project`:

```python
train_jax(
    species="trex",
    stage=1,
    num_envs=2048,
    wandb_project="mesozoic-labs",
)
```

The JAX trainer logs per-update metrics (reward, episode length, policy loss, value loss, entropy) to W&B via the built-in `WandbHook`.

## Checkpointing

Model parameters, optimizer state, and running normalization statistics are saved periodically during training. Checkpoints use JAX-compatible serialization so training can be resumed from any saved state.

```python
# Resume from a checkpoint
train_jax(
    species="trex",
    stage=2,
    checkpoint_dir="results/trex/jax/stage1",
)
```

## GPU Memory Requirements

The number of parallel environments is the main factor in GPU memory usage:

| Environments | Approximate VRAM | Recommended GPU |
|---|---|---|
| 512 | ~2 GB | T4 (16 GB) |
| 2,048 | ~6 GB | T4, V100, A100 |
| 4,096 | ~12 GB | V100 (32 GB), A100 |
| 8,192 | ~22 GB | A100 (40/80 GB) |

The default of 2,048 environments works on most modern GPUs. Reduce `num_envs` if you encounter out-of-memory errors.

## JIT Compilation

The first training step triggers JAX's JIT compilation, which takes 1-3 minutes depending on the model complexity and GPU. This is a one-time cost -- subsequent steps run at full speed. This is expected behavior and not an error.

## Vertex AI with JAX

For cloud training with the JAX backend on Vertex AI, use an A100 GPU machine type:

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
                    "--num-envs", "4096",
                ],
            },
        }
    ],
)
job.run(sync=False)
```

See [Training on Vertex AI](vertex-ai.md) for full setup instructions.

## Comparison with SB3 Results

The JAX backend trains significantly faster in wall-clock time due to massive parallelism. Both backends converge to similar final performance since they use the same reward functions and curriculum stages.

| Backend | Envs | Steps/sec | Stage 1 Wall Time |
|---|---|---|---|
| SB3 (CPU, 4 envs) | 4 | ~2,000 | ~3 hours |
| JAX (GPU, 2048 envs) | 2,048 | ~500,000 | ~5 minutes |

*Times are approximate and vary by hardware.*
