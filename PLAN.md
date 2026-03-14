# Environment Randomization Plan

## Goal
Add domain randomization to `BaseDinoEnv` to make trained policies more robust. All randomization is controlled by parameters that default to 0 (disabled), so existing training is unaffected unless explicitly enabled via TOML configs.

## Current State
- **Reset noise**: Small perturbation to joint positions/velocities and starting height (`reset_noise_scale`)
- **Target spawn randomization**: Random prey/food positions within configurable ranges
- **No physics domain randomization**: Friction, mass, gravity, actuator strength are all fixed

## Proposed Randomization Features

### 1. Ground Friction Randomization (per-episode)
**File**: `base_env.py` — `reset()`
**Parameter**: `friction_range: tuple[float, float] = (0.0, 0.0)` — multiplicative range around nominal friction (e.g. `(0.8, 1.2)` = +/-20%)
**Implementation**: On each `reset()`, sample a uniform multiplier and scale all geom friction coefficients. Store nominal values at `__init__` time to restore/re-scale each episode.
**Why**: Teaches the policy to handle slippery and grippy surfaces, critical for locomotion robustness.

### 2. Joint Damping Randomization (per-episode)
**File**: `base_env.py` — `reset()`
**Parameter**: `joint_damping_range: tuple[float, float] = (0.0, 0.0)` — multiplicative range (e.g. `(0.8, 1.2)`)
**Implementation**: On each `reset()`, sample a per-joint uniform multiplier and scale `model.dof_damping`. Store nominal values at init.
**Why**: Simulates wear, temperature variation, and model uncertainty in joint dynamics.

### 3. Gravity Perturbation (per-episode)
**File**: `base_env.py` — `reset()`
**Parameter**: `gravity_range: tuple[float, float] = (0.0, 0.0)` — additive range on Z-gravity (e.g. `(-0.5, 0.5)` adds noise around -9.81)
**Implementation**: On each `reset()`, sample an additive offset and apply to `model.opt.gravity[2]`. Store nominal gravity at init.
**Why**: Prevents overfitting to exact gravity constant; useful for sim-to-real and general robustness.

### 4. Actuator Strength Randomization (per-episode)
**File**: `base_env.py` — `reset()`
**Parameter**: `actuator_strength_range: tuple[float, float] = (0.0, 0.0)` — multiplicative range (e.g. `(0.85, 1.15)`)
**Implementation**: On each `reset()`, sample a per-actuator multiplier and scale `model.actuator_gainprm[:, 0]` and `model.actuator_biasprm[:, 1]`. Store nominals at init.
**Why**: Simulates motor strength variation and wear; forces the policy to not rely on exact torque capabilities.

### 5. External Force Perturbations (during episode)
**File**: `base_env.py` — `step()`
**Parameters**:
- `push_force_scale: float = 0.0` — maximum push force magnitude in Newtons
- `push_interval: int = 0` — apply a random push every N steps (0 = disabled)
**Implementation**: During `step()`, every `push_interval` steps, apply a random 3D force to the root body via `data.xfrc_applied[root_body_id]`. Clear it on the next step. Subclasses declare `_root_body_id` (pelvis/torso).
**Why**: Trains balance recovery; the most impactful single randomization for locomotion robustness.

### 6. Observation Noise (per-step)
**File**: `base_env.py` — `step()`
**Parameter**: `obs_noise_scale: float = 0.0` — std dev of additive Gaussian noise on observations
**Implementation**: After `_get_obs()`, add `np.random.normal(0, obs_noise_scale, obs.shape)` to the observation vector before returning it.
**Why**: Simulates sensor noise; prevents the policy from relying on unrealistically precise state information.

## Files to Modify

| File | Changes |
|------|---------|
| `environments/shared/base_env.py` | Add 7 new `__init__` params, store nominal physics values, randomize in `reset()`, add perturbations/noise in `step()` |
| `environments/velociraptor/envs/raptor_env.py` | Pass new kwargs through to `super().__init__()` |
| `environments/trex/envs/trex_env.py` | Pass new kwargs through to `super().__init__()` |
| `environments/brachiosaurus/envs/brachio_env.py` | Pass new kwargs through to `super().__init__()` |
| `configs/velociraptor/stage2_locomotion.toml` | Add example randomization values (moderate) |
| `configs/velociraptor/stage3_strike.toml` | Add example randomization values (moderate) |
| `configs/trex/stage2_locomotion.toml` | Add example randomization values |
| `configs/trex/stage3_bite.toml` | Add example randomization values |
| `configs/brachiosaurus/stage2_locomotion.toml` | Add example randomization values |
| `configs/brachiosaurus/stage3_food_reach.toml` | Add example randomization values |

Stage 1 (balance) configs are left without randomization — the agent should first learn to stand before being perturbed. Randomization is introduced in stage 2+ where the policy is more mature.

## Suggested Default Values for Stage 2+ Configs

```toml
# Domain randomization (all disabled by default; enable for robustness)
friction_range = [0.8, 1.2]              # +/-20% ground friction
joint_damping_range = [0.9, 1.1]         # +/-10% joint damping
gravity_range = [-0.5, 0.5]             # +/-0.5 m/s^2 around -9.81
actuator_strength_range = [0.9, 1.1]     # +/-10% motor strength
push_force_scale = 5.0                   # 5N random pushes (gentle for a ~15kg raptor)
push_interval = 100                      # Push every 100 steps (~2s at 50Hz)
obs_noise_scale = 0.01                   # Small Gaussian sensor noise
```

## Design Principles

1. **All defaults are 0/disabled** — no impact on existing training runs
2. **Implemented in base class** — all species benefit automatically
3. **Per-episode** for physics params (friction, damping, gravity, actuator strength) — sampled once at `reset()`
4. **Per-step** for perturbations and noise — applied every step or at intervals
5. **Configurable via TOML** — no code changes needed to tune randomization
6. **Nominal values cached at init** — randomization is always relative to the XML-defined defaults
