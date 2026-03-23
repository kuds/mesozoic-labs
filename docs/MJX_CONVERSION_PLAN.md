# MJX Conversion Plan

Detailed plan for adding MuJoCo MJX (JAX-accelerated GPU simulation) support to
Mesozoic Labs while preserving the existing Stable Baselines 3 (SB3) CPU training
path. The goal is a **dual-backend architecture**: developers can train with either
SB3+CPU MuJoCo (accessible, no GPU required) or JAX+MJX (fast, GPU/TPU).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1 — Shared Pure-Function Reward/Obs Layer](#2-phase-1--shared-pure-function-rewardobs-layer)
3. [Phase 2 — MJX Environment Wrapper](#3-phase-2--mjx-environment-wrapper)
4. [Phase 3 — JAX Training Infrastructure](#4-phase-3--jax-training-infrastructure)
5. [Phase 4 — JAX Training Notebooks](#5-phase-4--jax-training-notebooks)
6. [Phase 5 — Dependency & Packaging Updates](#6-phase-5--dependency--packaging-updates)
7. [Phase 6 — Testing & Validation](#7-phase-6--testing--validation)
8. [Phase 7 — Documentation & Developer Guide](#8-phase-7--documentation--developer-guide)
9. [Migration Checklist](#9-migration-checklist)
10. [Risks & Mitigations](#10-risks--mitigations)

---

## 1. Architecture Overview

### Current Stack (SB3 path — preserved as-is)

```
gym.make("MesozoicLabs/TRex-v0")
  └─ TRexEnv(BaseDinoEnv)       # Gymnasium env, CPU MuJoCo
       └─ SB3 PPO/SAC            # PyTorch, SubprocVecEnv
            └─ VecNormalize       # Running-mean obs normalization
```

### New Stack (JAX/MJX path — added)

```
MJXDinoEnv(species="trex", stage=1)
  └─ mjx.put_model(mj_model)     # GPU-resident model
  └─ jax.vmap(step_fn)           # 2048–8192 parallel envs
       └─ JAX PPO (Flax+Optax)   # Pure-JAX policy + optimizer
            └─ Running-mean norm  # JAX-based obs normalization
```

### Dual-Backend Design Principle

The two paths share:
- **MJCF model files** (`*.xml`) — identical, no changes needed.
- **Reward logic** — extracted into pure functions usable by both NumPy and JAX.
- **Stage configs** (`configs/*/stage*.toml`) — identical TOML files drive both paths.
- **Evaluation** — CPU MuJoCo rendering for both (MJX has no native renderer).

They differ in:
- **Environment wrapper** — Gymnasium `step()/reset()` vs. JAX functional `step_fn/reset_fn`.
- **Training loop** — SB3 callbacks vs. JIT-compiled JAX rollout+update.
- **Parallelism** — `SubprocVecEnv` (4–32 envs) vs. `jax.vmap` (2048–8192 envs).

---

## 2. Phase 1 — Shared Pure-Function Reward/Obs Layer

**Goal:** Extract reward and observation computation into backend-agnostic pure
functions that work with both NumPy arrays (SB3 path) and JAX arrays (MJX path).

### 2.1 New File: `environments/shared/reward_functions.py`

Create pure functions for each reward component. Each function takes array inputs
(positions, velocities, sensor data) and returns a scalar reward. No `self`, no
MuJoCo data objects.

```python
def reward_forward_velocity(vel_2d, forward_dir_2d, max_vel, weight):
    """Forward velocity reward, normalized to [-weight, +weight]."""
    forward_vel = vel_2d @ forward_dir_2d  # dot product (works in np and jnp)
    return weight * clip(forward_vel / max_vel, -1.0, 1.0)

def reward_energy(action, n_actuators, weight):
    """Energy penalty: -weight * mean(action^2)."""
    return -weight * sum(action ** 2) / n_actuators

def reward_tail_stability(tail_tip_angvel, weight, max_angvel=10.0):
    ...

def reward_posture(tilt_angle, max_tilt, weight):
    ...

def reward_approach(prev_distance, curr_distance, dt, max_speed, weight):
    ...

# etc. for bite_bonus, gait_symmetry, heading_alignment, nosedive, height
```

**Key constraint:** Use only operations in the intersection of NumPy and JAX
(`+`, `-`, `*`, `/`, `clip`, `sum`, `dot`, `norm`). Avoid Python control flow
(`if/else`, `for` loops) — use `where()` instead for conditional logic.

### 2.2 New File: `environments/shared/obs_functions.py`

```python
def build_obs(qpos, qvel, sensordata, pelvis_xpos, target_pos,
              sensor_layout, root_qpos_dim=7, root_qvel_dim=6):
    """Construct observation vector from raw simulation state.

    Works with both np.ndarray and jnp.ndarray.
    """
    joint_pos = qpos[root_qpos_dim:]
    joint_vel = qvel[root_qvel_dim:]
    quat = sensordata[sensor_layout.quat]
    gyro = sensordata[sensor_layout.gyro]
    linvel = qvel[:3]
    accel = sensordata[sensor_layout.accel]
    foot_contacts = sensordata[sensor_layout.feet]

    target_rel = target_pos - pelvis_xpos
    target_dist = norm(target_rel)
    target_dir = target_rel / (target_dist + 1e-8)

    return concatenate([
        joint_pos, joint_vel, quat, gyro, linvel, accel,
        foot_contacts, target_dir, [target_dist]
    ])
```

### 2.3 Refactor Existing Envs

Update `TRexEnv._get_obs()`, `TRexEnv._get_reward_info()`, etc. to call
the new pure functions, passing in the relevant arrays from `self.data`. This
keeps the Gymnasium API unchanged while sharing logic with the MJX path.

```python
# trex_env.py (refactored)
def _get_obs(self):
    return build_obs(
        qpos=self.data.qpos,
        qvel=self.data.qvel,
        sensordata=self.data.sensordata,
        pelvis_xpos=self.data.xpos[self.pelvis_id],
        target_pos=self.data.mocap_pos[0],
        sensor_layout=TREX_SENSOR_LAYOUT,
    )
```

### 2.4 Files Changed

| File | Action |
|------|--------|
| `environments/shared/reward_functions.py` | **New** — Pure reward functions |
| `environments/shared/obs_functions.py` | **New** — Pure observation builder |
| `environments/trex/envs/trex_env.py` | **Edit** — Call shared functions |
| `environments/velociraptor/envs/raptor_env.py` | **Edit** — Call shared functions |
| `environments/brachiosaurus/envs/brachio_env.py` | **Edit** — Call shared functions |

---

## 3. Phase 2 — MJX Environment Wrapper

**Goal:** Create a JAX-native environment abstraction that mirrors the Gymnasium
env lifecycle but uses pure functions and `jax.vmap` for batched execution.

### 3.1 New File: `environments/shared/mjx_env.py`

```python
@dataclass
class MJXEnvConfig:
    """Immutable environment configuration (JAX-compatible)."""
    species: str
    stage: int
    frame_skip: int
    max_episode_steps: int
    healthy_z_range: tuple[float, float]
    reward_weights: dict[str, float]  # from TOML config
    body_ids: dict[str, int]          # cached MuJoCo name→id mappings
    sensor_layout: SensorLayout

class MJXDinoEnv:
    """JAX-native batched dinosaur environment.

    Unlike Gymnasium envs, this uses functional style:
    - State is explicit (MJX data + step counter + prev_action)
    - All operations are pure functions suitable for jax.jit/vmap
    - No Python-level loops over environments
    """

    def __init__(self, species: str, stage: int = 1, num_envs: int = 2048):
        # Load MJCF and config (one-time, on CPU)
        self.mj_model = mujoco.MjModel.from_xml_path(...)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.config = MJXEnvConfig(...)  # from TOML

        # Build JIT-compiled step/reset functions
        self._step_fn = jax.jit(jax.vmap(self._step_single, in_axes=(None, 0, 0)))
        self._reset_fn = jax.jit(jax.vmap(self._reset_single, in_axes=(None, 0)))

    def _step_single(self, model, env_state, action):
        """Pure-function single-env step. Called via vmap."""
        data = env_state.data
        ctrl = scale_action(action, self.config.ctrl_range)
        data = data.replace(ctrl=ctrl)

        # Frame skip
        data = jax.lax.fori_loop(0, self.config.frame_skip, lambda _, d: mjx.step(model, d), data)

        # Observation and reward via shared pure functions
        obs = build_obs(data.qpos, data.qvel, data.sensordata, ...)
        reward = compute_species_reward(data, action, self.config)
        terminated = is_terminated(data, self.config)

        step_count = env_state.step_count + 1
        truncated = step_count >= self.config.max_episode_steps

        return EnvState(data=data, obs=obs, step_count=step_count, prev_action=action), reward, terminated, truncated

    def step(self, env_states, actions, rng):
        """Batched step across all environments."""
        new_states, rewards, terminated, truncated = self._step_fn(self.mjx_model, env_states, actions)

        # Auto-reset terminated/truncated envs
        dones = terminated | truncated
        new_states = self._auto_reset(new_states, dones, rng)

        return new_states, rewards, terminated, truncated

    def reset(self, rng):
        """Initialize all environments."""
        rngs = jax.random.split(rng, self.num_envs)
        return self._reset_fn(self.mjx_model, rngs)
```

### 3.2 Per-Species Configuration

Each species defines a config module with body IDs, sensor layout, and
species-specific reward function:

```python
# environments/trex/mjx_config.py
TREX_SENSOR_LAYOUT = SensorLayout(gyro=slice(0,3), accel=slice(3,6), ...)

def trex_reward(data, action, config):
    """T-Rex reward: forward + alive + bite + approach + posture + ..."""
    r = reward_forward_velocity(...)
    r += reward_alive(config.reward_weights["alive_bonus"])
    r += reward_bite(...)  # uses contact data
    ...
    return r
```

### 3.3 Contact Detection in JAX

MJX does not expose `data.contact` the same way as CPU MuJoCo. For bite/strike
detection, use geometry-based proximity instead:

```python
def is_bite_contact(head_xpos, prey_xpos, threshold=0.15):
    """Check bite via distance rather than contact pairs."""
    dist = jnp.linalg.norm(head_xpos - prey_xpos)
    return dist < threshold
```

This is simpler and fully compatible with `jax.vmap`. The threshold can be
tuned to match the contact geometry radius from the MJCF model.

### 3.4 Files Changed

| File | Action |
|------|--------|
| `environments/shared/mjx_env.py` | **New** — MJX env wrapper |
| `environments/shared/mjx_utils.py` | **New** — JAX helpers (scaling, normalization) |
| `environments/trex/mjx_config.py` | **New** — T-Rex MJX species config |
| `environments/velociraptor/mjx_config.py` | **New** — Raptor MJX species config |
| `environments/brachiosaurus/mjx_config.py` | **New** — Brachio MJX species config |

---

## 4. Phase 3 — JAX Training Infrastructure

**Goal:** Consolidate the inline PPO code from `jax_training.ipynb` into
reusable modules, add observation normalization, and support curriculum training.

### 4.1 New File: `environments/shared/jax_ppo.py`

Extract and generalize the PPO implementation from the existing notebook:

```python
class ActorCritic(nn.Module):
    """Flax actor-critic network (shared backbone)."""
    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    ...

def sample_action(params, network, obs, rng): ...
def compute_gae(rewards, values, dones, gamma, lam): ...
def ppo_loss(params, network, batch, clip_range, vf_coef, ent_coef): ...
def ppo_update(params, opt_state, optimizer, network, batch, config): ...
```

### 4.2 New File: `environments/shared/jax_normalization.py`

Running-mean observation normalization in JAX (equivalent to SB3's `VecNormalize`):

```python
@struct.dataclass
class RunningMeanStd:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float

def update_running_stats(stats, batch): ...
def normalize_obs(obs, stats, clip=10.0): ...
```

### 4.3 New File: `environments/shared/jax_training.py`

High-level training loop that mirrors `train_base.py` but for the JAX path:

```python
def train_jax(
    species: str,
    stage: int = 1,
    num_envs: int = 2048,
    num_updates: int = 500,
    rollout_len: int = 64,
    seed: int = 42,
    checkpoint_dir: str | None = None,
    wandb_project: str | None = None,
):
    """Train a species with JAX/MJX PPO.

    Loads config from TOML, creates MJX env, runs PPO training loop.
    """
    ...
```

### 4.4 New File: `environments/shared/jax_curriculum.py`

JAX-compatible curriculum manager:

```python
def check_stage_gate(eval_metrics, stage_config):
    """Check if curriculum gate thresholds are met."""
    ...

def run_curriculum(species, stages=(1, 2, 3), ...):
    """Run full curriculum: train each stage, evaluate gate, advance."""
    ...
```

### 4.5 Files Changed

| File | Action |
|------|--------|
| `environments/shared/jax_ppo.py` | **New** — Reusable JAX PPO |
| `environments/shared/jax_normalization.py` | **New** — Running-mean normalization |
| `environments/shared/jax_training.py` | **New** — High-level JAX train loop |
| `environments/shared/jax_curriculum.py` | **New** — JAX curriculum manager |

---

## 5. Phase 4 — JAX Training Notebooks

**Goal:** Refactor the JAX notebook to use the shared modules and make it
species-agnostic so a single notebook supports all species.

### 5.1 Refactored `notebooks/jax_training.ipynb` ✅

The notebook (formerly `jax_trex_training.ipynb`) has been refactored to:
- Import from shared modules (`reward_functions`, `obs_functions`, `jax_ppo`, `mjx_utils`)
- Use a `SPECIES` configuration variable to select any species
- Auto-resolve model paths, body IDs, healthy ranges, and sensor layouts
- Load reward configs from the species' TOML stage files

No per-species notebooks are needed — a single `SPECIES = "trex"` /
`"velociraptor"` / `"brachiosaurus"` selector drives everything.

### 5.2 Existing SB3 Notebook

`notebooks/training.ipynb` (the SB3 Colab notebook) is **not modified**. It
continues to work exactly as before via the Gymnasium API.

### 5.4 Files Changed

| File | Action |
|------|--------|
| `notebooks/jax_training.ipynb` | **Edit** — Refactor to use shared modules |
| `notebooks/jax_raptor_training.ipynb` | **New** — Raptor JAX training |
| `notebooks/jax_brachio_training.ipynb` | **New** — Brachio JAX training |

---

## 6. Phase 5 — Dependency & Packaging Updates

### 6.1 New Optional Dependency Group in `pyproject.toml`

```toml
[project.optional-dependencies]
jax = [
    "mujoco-mjx>=3.0.0",
    "jax[cuda12]>=0.4.20",
    "flax>=0.8.0",
    "optax>=0.1.7",
]
train = [
    "stable-baselines3[extra]>=2.2.0",
    "wandb>=0.16.0",
    "cloudml-hypertune>=0.1.0.dev0",
]
all = [
    "mesozoic-labs[train,jax,viz,dev]",
]
```

### 6.2 Lazy Imports

All JAX/MJX imports in the shared modules must be **lazy** so that users who
only install the `train` (SB3) dependencies don't get import errors:

```python
# environments/shared/mjx_env.py
def _check_jax():
    try:
        import jax
        import mujoco.mjx as mjx
    except ImportError:
        raise ImportError(
            "JAX/MJX training requires: pip install mesozoic-labs[jax]"
        )
```

### 6.3 Files Changed

| File | Action |
|------|--------|
| `pyproject.toml` | **Edit** — Add `[jax]` dependency group |

---

## 7. Phase 6 — Testing & Validation

### 7.1 Shared Reward/Obs Function Tests

```python
# environments/shared/tests/test_reward_functions.py
def test_reward_forward_velocity_positive():
    """Forward motion toward target yields positive reward."""
    ...

def test_reward_functions_numpy_jax_parity():
    """Same inputs produce same outputs with NumPy and JAX arrays."""
    np_result = reward_forward_velocity(np.array(...), ...)
    jax_result = reward_forward_velocity(jnp.array(...), ...)
    assert np.allclose(np_result, jax_result)
```

### 7.2 MJX Environment Tests

```python
# environments/shared/tests/test_mjx_env.py
@pytest.mark.skipif(not _has_jax(), reason="JAX not installed")
def test_mjx_env_step_shape():
    """MJX env step returns correct shapes."""
    ...

def test_mjx_obs_matches_gymnasium():
    """MJX obs matches Gymnasium env obs for the same initial state."""
    ...
```

### 7.3 Regression: SB3 Path Unchanged

Run the existing test suite to confirm no regressions:

```bash
pytest environments/ --tb=short -q
```

### 7.4 Files Changed

| File | Action |
|------|--------|
| `environments/shared/tests/test_reward_functions.py` | **New** |
| `environments/shared/tests/test_obs_functions.py` | **New** |
| `environments/shared/tests/test_mjx_env.py` | **New** |

---

## 8. Phase 7 — Documentation & Developer Guide

### 8.1 Update `README.md`

Add a section showing both training paths:

```markdown
## Training

### SB3 (CPU, no GPU required)
pip install mesozoic-labs[train]
python -m environments.trex.scripts.train_sb3 train --stage 1

### JAX/MJX (GPU, 10-100x faster)
pip install mesozoic-labs[jax]
python -m environments.shared.jax_training --species trex --stage 1
```

### 8.2 Update `docs/ROADMAP.md`

Mark MJX integration as part of the project roadmap.

### 8.3 Files Changed

| File | Action |
|------|--------|
| `README.md` | **Edit** — Add dual-backend training section |
| `docs/ROADMAP.md` | **Edit** — Update with MJX milestone |

---

## 9. Migration Checklist

### Phase 1 — Shared Pure Functions
- [ ] Create `environments/shared/reward_functions.py`
- [ ] Create `environments/shared/obs_functions.py`
- [ ] Refactor `TRexEnv` to use shared functions
- [ ] Refactor `RaptorEnv` to use shared functions
- [ ] Refactor `BrachioEnv` to use shared functions
- [ ] Run existing tests — all pass

### Phase 2 — MJX Environment
- [ ] Create `environments/shared/mjx_env.py`
- [ ] Create `environments/shared/mjx_utils.py`
- [ ] Create species MJX configs (`trex/mjx_config.py`, etc.)
- [ ] Verify MJX env produces same obs as Gymnasium env for same state

### Phase 3 — JAX Training
- [ ] Create `environments/shared/jax_ppo.py`
- [ ] Create `environments/shared/jax_normalization.py`
- [ ] Create `environments/shared/jax_training.py`
- [ ] Create `environments/shared/jax_curriculum.py`
- [ ] Verify PPO converges on T-Rex Stage 1 (balance)

### Phase 4 — Notebooks
- [x] Refactor `notebooks/jax_training.ipynb` (species-agnostic, all 3 species)
- [ ] Test notebook on Colab with GPU runtime for each species

### Phase 5 — Packaging
- [ ] Add `[jax]` optional dependency group to `pyproject.toml`
- [ ] Verify lazy imports (no errors without JAX installed)
- [ ] Verify `pip install mesozoic-labs[jax]` works

### Phase 6 — Testing
- [ ] Write reward/obs function tests (NumPy + JAX parity)
- [ ] Write MJX env tests (shapes, reset, step)
- [ ] Full regression test (`pytest environments/`)

### Phase 7 — Documentation
- [ ] Update README with dual-backend instructions
- [ ] Update ROADMAP

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Contact detection differs in MJX** | Bite/strike rewards may behave differently | Use distance-based proximity instead of `data.contact` for JAX path; tune threshold to match contact geometry |
| **JAX array operations ≠ NumPy** | Shared reward functions may have subtle differences | Parity tests comparing NumPy vs JAX outputs; use only common operations |
| **JIT compilation time** | 1–3 min startup delay for training | Document expected behavior; compilation is one-time per model structure |
| **MJX feature gaps** | Some MuJoCo features unsupported (tendon wrapping, certain constraints) | Current MJCF models use standard hinges/balls — unlikely to hit gaps; validate by loading all 3 models into MJX |
| **GPU memory for large batch sizes** | 8192 envs may exceed VRAM on smaller GPUs | Default to 2048 envs; document memory requirements per GPU tier |
| **SB3 path regression** | Refactoring reward functions may change SB3 training behavior | Existing tests + numerical parity checks before/after refactor |
| **Observation normalization drift** | JAX running-mean may diverge from SB3 VecNormalize | Validate both normalizers produce similar distributions on same data |
| **TF32 precision on Ampere GPUs** | Non-deterministic results on RTX 30/40 series | Document `JAX_DEFAULT_MATMUL_PRECISION=highest` workaround |

---

## Summary of New Files

```
environments/shared/
├── reward_functions.py      # Pure reward functions (NumPy/JAX)
├── obs_functions.py         # Pure observation builder (NumPy/JAX)
├── mjx_env.py               # MJX batched environment wrapper
├── mjx_utils.py             # JAX helpers (action scaling, etc.)
├── jax_ppo.py               # Flax ActorCritic + PPO implementation
├── jax_normalization.py     # Running-mean obs normalization in JAX
├── jax_training.py          # High-level JAX training entry point
├── jax_curriculum.py        # JAX curriculum manager
└── tests/
    ├── test_reward_functions.py
    ├── test_obs_functions.py
    └── test_mjx_env.py

environments/trex/mjx_config.py
environments/velociraptor/mjx_config.py
environments/brachiosaurus/mjx_config.py

notebooks/
└── jax_training.ipynb      # Refactored, species-agnostic (supports all species)
```

## Estimated Scope

- **~15 new files** (modules, configs, tests, notebooks)
- **~6 modified files** (3 species envs, pyproject.toml, README, ROADMAP)
- **0 deleted files** — SB3 path is fully preserved
- Both backends share the same MJCF models and TOML stage configs
