# RL Training Plan

> **Date:** 2026-03-26 *(last reviewed 2026-04-18)*
> **Scope:** Remaining SAC & PPO training across all species
> **Notebooks:** `training.ipynb`, `ray_tune_sweep.ipynb`
>
> **Update (2026-04-18):** Active priority shifted to JAX training
> stabilization (Apr 2–3) and then to the SAC VecNormalize fix (Apr 18).
> The status matrix below still reflects the last known SB3 run results —
> no new SB3 runs between Mar 26 and Apr 18. Before resuming SB3 trials,
> factor in (1) the SAC reward-normalization fix (see
> `environments/shared/train_base.py`, which now disables `norm_reward` for
> SAC), and (2) the pending Stage 3 terminal-bonus rescale documented in
> [investigations/REWARD_SCALE_REDESIGN.md](investigations/REWARD_SCALE_REDESIGN.md). Running SAC trials
> before the rescale will test the normalization fix in isolation; running
> them after will confound the two changes.

---

## Training Status Matrix

| Species | Algorithm | Stage 1 (Balance) | Stage 2 (Locomotion) | Stage 3 (Behavior) | Status |
|---------|-----------|-------------------|---------------------|-------------------|--------|
| Velociraptor | PPO | 1960.64 | 2679.18 (3.47 m/s) | 93.3% strike | **COMPLETE** |
| Velociraptor | SAC | 970.19 | 2078.62 (2.91 m/s) | 90.0% strike | **COMPLETE** |
| T-Rex | PPO | 2994.34 | 1941.43 (3.47 m/s) | 96.7% bite | **COMPLETE** |
| T-Rex | SAC | -- | -- | -- | **NOT STARTED** |
| Brachiosaurus | PPO | 3002.52 | 4176.95 (1.12 m/s) | 16.7% food_reach (need 50%) | **STAGE 3 BLOCKED** |
| Brachiosaurus | SAC | -- | -- | -- | **NOT STARTED** |

### Best Completed Runs (reference checkpoints)

| Species | Algorithm | Run Dir | Notes |
|---------|-----------|---------|-------|
| Velociraptor | PPO | `ppo_20260315_041632` | Best overall raptor PPO |
| Velociraptor | SAC | `sac_20260321_170055` | Only full SAC run |
| T-Rex | PPO | `ppo_20260317_120601` | Best overall T-Rex PPO |
| Brachiosaurus | PPO | `ppo_20260321_144730` | Only run to pass Stage 2; Stage 3 at 16.7% |

---

## Remaining Trials (4 total)

### Trial 1: T-Rex SAC

**Notebook:** `training.ipynb`

**Why training.ipynb (not ray tune):** T-Rex PPO hit 96.7% bite success, indicating the
task structure and reward shaping are solid. The SAC hyperparameters in the TOML configs are
well-reasoned. Velociraptor SAC passed all 3 stages on its first serious run with TOML
defaults -- T-Rex should behave similarly.

#### Notebook Setup

| Setting | Value |
|---------|-------|
| Species | `trex` |
| Algorithm | `SAC` |
| n_envs | `8` (SAC benefits from more parallel envs for replay diversity) |
| Stages | All 3 (curriculum auto-advances) |
| Stage 1 timesteps | 6M |
| Stage 2 timesteps | 8M |
| Stage 3 timesteps | 8M |
| GPU | T4 or better (SAC is ~2x slower wall-clock than PPO) |
| Estimated runtime | ~20-24 hrs (based on Velociraptor SAC: 23 hrs for 22M steps) |

#### Key SAC Params (from TOML, already configured)

| Stage | learning_rate | gamma | buffer_size | train_freq | Key Env Param |
|-------|---------------|-------|-------------|------------|---------------|
| 1 (balance) | 3e-4 | 0.99 | 300K | 4 | alive_bonus=2.0, posture_weight=2.0 |
| 2 (locomotion) | 1e-4 | 0.99 | 1M | 4 | forward_vel_weight=2.0 |
| 3 (bite) | 1e-4 | 0.99 | 1M | 4 | bite_bonus=1000, bite_approach_weight=3.0 |

---

### Trial 2: Brachiosaurus PPO Stage 3 Sweep

**Notebook:** `ray_tune_sweep.ipynb`

**Why ray tune (not training.ipynb):** Stage 3 food_reach is stuck at 16.7% success
(target: 50%). The TOML config has already been extensively tuned (widened
food_reach_threshold to 0.8, tripled food_approach_weight to 3.0, extended to 12M steps,
added warmup/ramp), but it's not converging. A systematic sweep over the remaining degrees
of freedom is the right next step.

#### Notebook Setup

| Setting | Value |
|---------|-------|
| Species | `brachiosaurus` |
| Algorithm | `PPO` |
| Stage | `3` only (load Stage 2 checkpoint from `ppo_20260321_144730`) |
| Trials | 30-40 |
| Parallel | 5 (Colab GPU memory constraint) |
| n_envs | 4 |
| Timesteps per trial | 12M |
| ASHA grace period | 3M steps (enough for warmup + ramp to complete) |
| GPU | A100 if possible (long trials) |

#### Recommended Sweep Search Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| `env_food_reach_bonus` | [500, 2000] | log | Current 1000 may not dominate enough vs cumulative per-step rewards |
| `env_food_approach_weight` | [2.0, 6.0] | linear | Strengthen gradient toward food |
| `env_food_head_proximity_weight` | [1.0, 4.0] | linear | The "last mile" signal for head positioning |
| `env_food_reach_threshold` | [0.6, 1.2] | linear | Wider threshold = easier initial discovery |
| `env_food_distance_range_max` | [3.0, 5.0] | linear | Closer food = more attempts per episode |
| `ppo_ent_coef` | [0.005, 0.03] | log | More exploration for novel neck extension |
| `ppo_clip_range` | [0.15, 0.25] | linear | Stability vs expressiveness |
| `curriculum_ramp_timesteps` | [1M, 3M] | discrete | Speed of transition from locomotion to food_reach |
| `ppo_gamma` | [0.99, 0.998] | linear | Higher gamma may help value the sparse bonus |

---

### Trial 3: Brachiosaurus SAC (All Stages)

**Notebook:** `training.ipynb`

**Why training.ipynb (not ray tune):** The SAC configs in the TOMLs are already written with
thoughtful rationale. SAC's replay buffer and auto-entropy tuning may naturally handle two
problems that plague Brachiosaurus PPO:

1. **Catastrophic forgetting between stages** -- buffer retains old transitions
2. **Discovering novel neck-extension behavior in Stage 3** -- auto-entropy maintains exploration

Try the defaults first.

#### Notebook Setup

| Setting | Value |
|---------|-------|
| Species | `brachiosaurus` |
| Algorithm | `SAC` |
| n_envs | `8` |
| Stages | All 3 (curriculum auto-advances) |
| Stage 1 timesteps | 6M |
| Stage 2 timesteps | 16M |
| Stage 3 timesteps | 12M |
| GPU | T4 or better |
| Estimated runtime | ~30-36 hrs (Brachiosaurus is heavier to simulate; SAC ~2x slower) |

#### Key SAC Params (from TOML, already configured)

| Stage | learning_rate | gamma | buffer_size | Key Env Param |
|-------|---------------|-------|-------------|---------------|
| 1 (balance) | 3e-4 | 0.99 | 300K | alive_bonus=2.0 |
| 2 (locomotion) | 1e-4 | 0.99 | 1M | gait_symmetry_weight=2.0, forward_vel_weight=4.0 |
| 3 (food_reach) | 1e-4 | 0.99 | 1M | food_reach_bonus=1000, food_approach_weight=3.0 |

#### Watch For

If Stage 2 locomotion stalls below 0.75 m/s forward vel, the `gait_symmetry_weight=2.0`
setting (critical for PPO's only S2 pass) may interact differently with SAC's entropy.
Be ready to adjust.

---

### Trial 4 (Contingency): Brachiosaurus SAC Stage 3 Sweep

**Notebook:** `ray_tune_sweep.ipynb`

**When to run:** Only if Trial 3 passes Stages 1-2 but fails Stage 3 food_reach (same
pattern as PPO).

#### Notebook Setup

| Setting | Value |
|---------|-------|
| Species | `brachiosaurus` |
| Algorithm | `SAC` |
| Stage | `3` only (load Stage 2 checkpoint from Trial 3) |
| Trials | 20 |
| Parallel | 5 |
| n_envs | 8 |
| Timesteps per trial | 12M |

#### Sweep Search Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| `sac_learning_rate` | [5e-5, 3e-4] | log | |
| `sac_gamma` | [0.985, 0.995] | linear | |
| `sac_buffer_size` | [500K, 2M] | discrete | Larger buffer retains more locomotion memory |
| `sac_train_freq` | [4, 8, 16] | discrete | |
| `env_food_reach_bonus` | [500, 2000] | log | |
| `env_food_approach_weight` | [2.0, 6.0] | linear | |
| `env_food_head_proximity_weight` | [1.0, 4.0] | linear | |
| `env_food_reach_threshold` | [0.6, 1.2] | linear | |

---

## Recommended Execution Order

```
Trial 1: T-Rex SAC (training.ipynb)          <-- high confidence, run first
Trial 2: Brachio PPO S3 sweep (ray_tune)     <-- can run in parallel with Trial 1
Trial 3: Brachio SAC full (training.ipynb)    <-- run after Trial 2 completes
Trial 4: Brachio SAC S3 sweep (ray_tune)     <-- only if Trial 3 S3 fails
```

**Parallelism:** Trials 1 and 2 are independent and can run simultaneously on separate
Colab instances. Trial 3 benefits from any insights gained from Trial 2's sweep results
(reward weight findings apply to SAC too).

## Estimated Total Compute

| Trial | Timesteps | Est. Wall Clock | GPU |
|-------|-----------|-----------------|-----|
| 1: T-Rex SAC | 22M | 20-24 hrs | T4 |
| 2: Brachio PPO S3 sweep | 30-40 x 12M = 360-480M | 48-72 hrs | A100 preferred |
| 3: Brachio SAC full | 34M | 30-36 hrs | T4 |
| 4: Brachio SAC S3 sweep (if needed) | 20 x 12M = 240M | 36-48 hrs | A100 preferred |
