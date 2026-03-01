# Forward Velocity Reward Ramp — Stage 2 (Raptor & T-Rex)

## The Problem

In **Stage 1**, the dinosaur agents learn to balance with `forward_vel_weight = 0.0`
(no incentive to move). Abruptly setting the full forward-velocity weight at the
start of Stage 2 would cause **catastrophic forgetting** — the agent overwrites its
balance skills trying to lunge forward.

## How the Ramp Works

The `RewardRampCallback` (in `environments/shared/curriculum.py`) **linearly
interpolates** `forward_vel_weight` from a small starting value to its target over
the first 500,000 timesteps of Stage 2:

```
current_weight = start_value + (timesteps / ramp_timesteps) * (end_value - start_value)
```

| Config parameter    | Raptor | T-Rex |
|---------------------|--------|-------|
| `ramp_start_value`  | 0.1    | 0.1   |
| `ramp_timesteps`    | 500k   | 500k  |
| Target `forward_vel_weight` | 1.0 | 0.8 |

### Example schedule (raptor)

| Timesteps | `forward_vel_weight` |
|-----------|---------------------|
| 0         | 0.1                 |
| 125k      | 0.325               |
| 250k      | 0.55                |
| 375k      | 0.775               |
| 500k      | 1.0 (final)         |

The callback quantises to 3 decimal places and only pushes updates roughly every
10k steps to avoid per-step overhead.

## Two Complementary Safeguards

When Stage 2 loads a Stage 1 checkpoint, two callbacks fire together (see
`train_sb3.py`, ~line 319):

### 1. `StageWarmupCallback` (first 100k timesteps)

- Shrinks PPO `clip_range` from 0.2 → 0.02 (policy barely changes)
- Boosts `ent_coef` to 0.02 (maintains exploration)
- Purpose: lets the **value function adapt** to the new reward landscape while
  the policy stays nearly frozen, preserving balance

### 2. `RewardRampCallback` (first 500k timesteps)

- Starts forward velocity signal at 10 % strength
- Linearly ramps to full strength
- Purpose: the policy gradually learns to walk **on top of** existing balance skills

The warmup ends at 100k steps (clip range returns to 0.2), but the forward-velocity
weight is still only ~0.28 at that point, keeping gradient magnitudes manageable.

## Forward Velocity Reward Computation

In both `raptor_env.py` and `trex_env.py`, the reward term is computed as:

```python
prey_dir_2d = normalize(prey_pos[:2] - pelvis_pos[:2])
forward_vel  = dot(qvel[0:2], prey_dir_2d)
forward_vel_norm = clip(forward_vel / forward_vel_max, -1.0, 1.0)
reward_forward   = forward_vel_weight * forward_vel_norm
```

The reward is the agent's velocity **toward the prey**, normalised by
`forward_vel_max` (3.0 m/s for both species), then scaled by the weight the ramp
callback is controlling.

At timestep 0 this contributes at most `0.1 × 1.0 = 0.1` to the total reward; by
timestep 500k it contributes up to `1.0` (raptor) or `0.8` (T-Rex).

## Why T-Rex Gets a Lower Final Weight

The T-Rex has a higher centre of mass and needs stronger posture constraints
(`posture_weight = 2.0`, `nosedive_weight = 3.0` vs 1.5 / 1.5 for raptor). A lower
`forward_vel_weight` of 0.8 prevents the T-Rex from over-prioritising speed at the
expense of its more delicate upright posture.

## Key Files

| File | Purpose |
|------|---------|
| `environments/shared/curriculum.py` | `RewardRampCallback` & `StageWarmupCallback` |
| `configs/velociraptor/stage2_locomotion.toml` | Raptor Stage 2 config |
| `configs/trex/stage2_locomotion.toml` | T-Rex Stage 2 config |
| `environments/velociraptor/envs/raptor_env.py` | Raptor reward computation |
| `environments/trex/envs/trex_env.py` | T-Rex reward computation |
| `environments/*/scripts/train_sb3.py` | Training script wiring callbacks |
