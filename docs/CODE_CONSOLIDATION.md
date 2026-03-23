# Codebase Cleanup & Consolidation — Implementation Plan

> **Target version:** v0.3.0
> **Status:** **COMPLETE** (2026-03-19)
> **Actual reduction:** ~3,000+ lines of duplicated code removed
> **Approach:** Incremental, test-first — each step independently merged

---

## Overview

The cleanup is organized into **7 phases**, ordered by impact and risk. Each phase
is self-contained: it can be committed, tested, and verified independently before
moving to the next. The guiding principle is **extract, don't rewrite** — we pull
duplicated code into shared modules while keeping species-specific behavior
configurable via parameters.

---

## Phase 1: Shared Training Script (`train_base.py`)

**Impact:** ~2,000 lines removed | **Risk:** Medium | **Files touched:** 4

The three `train_sb3.py` files (920–993 lines each) are ~90% identical. Nine
functions are copy-pasted verbatim; only the environment class, species name,
and stage-3 labels differ.

### 1.1 Create `environments/shared/train_base.py`

Extract all shared functions into a single module:

| Function | Lines | Action |
|----------|-------|--------|
| `linear_schedule()` | 7 | Move verbatim |
| `_cast_value()` | 16 | Move verbatim |
| `_apply_overrides()` | 28 | Move verbatim |
| `make_env()` | 13 | Parameterize `env_class` |
| `create_vec_env()` | 16 | Move verbatim |
| `_run_hpt_eval()` | 30 | Parameterize `success_keys` list |
| `train()` | ~210 | Parameterize `species` string |
| `train_curriculum()` | ~260 | Parameterize `species`, fix brachiosaurus missing `eval_callback` |
| `evaluate()` | ~130 | Parameterize `stage3_label`, `height_label`, species-specific metric keys |
| `main()` argparse | ~175 | Parameterize `species`, `stage_descriptions` |

Create a `SpeciesConfig` dataclass to hold all species-specific parameters:

```python
@dataclasses.dataclass
class SpeciesConfig:
    species: str                    # "velociraptor", "trex", "brachiosaurus"
    env_class: type                 # RaptorEnv, TRexEnv, BrachioEnv
    stage3_label: str               # "Strike", "Bite", "Food Reaching"
    stage_descriptions: str         # "1=balance, 2=locomotion, 3=strike"
    height_label: str               # "Pelvis height" or "Torso height"
    success_keys: list[str]         # ["strike_success", "bite_success"] etc.
    hunting_section_label: str      # "Hunting" or "Food Reaching"
```

### 1.2 Reduce species scripts to thin wrappers

Each species' `train_sb3.py` becomes ~20 lines:

```python
"""Velociraptor training script."""
from environments.shared.train_base import SpeciesConfig, main as train_main
from environments.velociraptor.envs.raptor_env import RaptorEnv

CONFIG = SpeciesConfig(
    species="velociraptor",
    env_class=RaptorEnv,
    stage3_label="Strike",
    stage_descriptions="1=balance, 2=locomotion, 3=strike",
    height_label="Pelvis height",
    success_keys=["strike_success", "bite_success"],
    hunting_section_label="Hunting",
)

if __name__ == "__main__":
    train_main(CONFIG)
```

### 1.3 Fix inconsistencies discovered during analysis

- **Brachiosaurus `train_curriculum()`** is missing the `eval_callback=eval_callback`
  parameter in its `CurriculumCallback` instantiation. This is a bug — fix it
  during extraction so all species use the same (correct) callback setup.
- **T-Rex has extra `train_env.close()` / `eval_env.close()`** after `train()`.
  Standardize: all species should close envs after training (add to shared code).

### 1.4 Unit tests for `train_base.py`

Create `environments/shared/tests/test_train_base.py`:

- Test `linear_schedule()` returns correct values at 0%, 50%, 100% progress
- Test `_cast_value()` with int, float, bool, string, list inputs
- Test `_apply_overrides()` merges overrides correctly, handles nested keys
- Test `make_env()` creates a valid Gymnasium env with correct spaces
- Test `create_vec_env()` returns a VecEnv with correct num_envs
- Test `SpeciesConfig` dataclass validates all required fields
- Test `main()` argparse with each subcommand (train, curriculum, evaluate)

---

## Phase 2: Shared Test Utilities (`test_env_base.py`)

**Impact:** ~500 lines removed | **Risk:** Low | **Files touched:** 4

The three `scripts/test_env.py` files share identical test logic with only
the environment class differing. T-Rex is also missing 2 test functions
(`test_reward_components`, `test_observation_bounds`).

### 2.1 Create `environments/shared/test_env_base.py`

Extract parameterized test functions:

```python
def run_basic_functionality(env_class, expected_obs_shape, expected_act_shape, expected_reward_keys):
def run_episode_rollout(env_class, max_steps=1000):
def run_reward_components(env_class, n_steps=500):
def run_determinism(env_class, seed=42, n_steps=50):
def run_observation_bounds(env_class, n_steps=1000):
```

### 2.2 Reduce species test scripts to thin wrappers

Each species' `test_env.py` becomes ~30 lines calling the shared functions
with species-specific parameters.

### 2.3 Add missing T-Rex tests

T-Rex is currently missing `test_reward_components()` and
`test_observation_bounds()`. These get added automatically when wrapping
the shared base.

### 2.4 Unit tests

Add to `environments/shared/tests/test_test_env_base.py`:
- Verify shared test functions execute without error on each species
- Verify reward component keys match species-specific expectations

---

## Phase 3: Environment Method Extraction (~300 lines)

**Impact:** ~300 lines removed + better consistency | **Risk:** Medium | **Files touched:** 4

Common computation patterns are duplicated across species `_env.py` files.
These can be lifted into `BaseDinoEnv` as concrete helper methods that
subclasses call.

### 3.1 Common sensor constants

Move sensor index constants to `BaseDinoEnv`:

```python
class BaseDinoEnv(gym.Env, ABC):
    SENSOR_GYRO_START = 0
    SENSOR_ACCEL_START = 3
    SENSOR_QUAT_START = 6
```

Species `_cache_ids()` methods use these instead of hardcoded `0, 3, 6`.

### 3.2 Shared reward computation helpers

Add concrete (non-abstract) helper methods to `BaseDinoEnv`:

```python
def _compute_forward_velocity(self, direction_2d, velocity_2d, max_vel):
    """Normalized forward velocity toward target."""

def _compute_energy_penalty(self, action):
    """Squared action norm / n_actuators."""

def _compute_approach_shaping(self, prev_dist, curr_dist, max_speed):
    """Distance-delta approach reward, clipped to [-1, 1]."""

def _compute_posture_reward(self, quat, natural_pitch=0.0):
    """Tilt-based posture penalty, normalized to [-1, 0]."""
```

Species environments call these helpers instead of inlining the math.

### 3.3 Move `_spawn_target()` to base class

The target spawning logic is ~95% identical. Move it to `BaseDinoEnv` with
configurable range parameters:

```python
def _spawn_target(self):
    dist = self.np_random.uniform(*self.target_distance_range)
    lateral = self.np_random.uniform(*self.target_lateral_range)
    height = self.np_random.uniform(*self.target_height_range)
    # ... set mocap position
```

Brachiosaurus overrides only `target_height_range` (food is elevated).

### 3.4 Extract termination helpers

Height-check and tilt-check logic is identical across all three species:

```python
def _check_height_termination(self, body_pos_z):
    """Check if body center is outside healthy_z_range."""

def _check_tilt_termination(self, quat):
    """Check if body tilt exceeds max_tilt_angle."""
```

### 3.5 Unit tests

Expand `environments/shared/tests/test_base_env.py`:

- Test `_compute_forward_velocity()` with parallel, perpendicular, backward velocity
- Test `_compute_energy_penalty()` with zero, small, large actions
- Test `_compute_approach_shaping()` with approaching, receding, stationary agent
- Test `_compute_posture_reward()` at upright, tilted, fallen orientations
- Test `_check_height_termination()` at boundaries
- Test `_check_tilt_termination()` at boundaries
- Test `_spawn_target()` generates positions within configured ranges
- Test sensor constants match expected MuJoCo sensor layout

---

## Phase 4: Centralize Constants

**Impact:** Cleaner code, single source of truth | **Risk:** Low | **Files touched:** 5+

### 4.1 Create `environments/shared/constants.py`

```python
"""Simulation-wide default constants."""

# Sensor layout (matches MJCF sensor order)
SENSOR_GYRO_START = 0
SENSOR_ACCEL_START = 3
SENSOR_QUAT_START = 6

# VecNormalize defaults
DEFAULT_NORM_OBS = True
DEFAULT_NORM_REWARD = True
DEFAULT_CLIP_OBS = 10.0
DEFAULT_CLIP_REWARD = 50.0

# Physics defaults
DEFAULT_FRAME_SKIP = 5
TAIL_ANGULAR_VEL_MAX = 10.0  # rad/s
```

### 4.2 Update references

Replace hardcoded values in:
- All three `train_sb3.py` (VecNormalize params) — mostly in `train_base.py` after Phase 1
- All three `*_env.py` (sensor indices, frame skip)
- Species-specific constants stay in their respective files

### 4.3 Unit tests

Add `environments/shared/tests/test_constants.py`:
- Verify constants match expected values (regression guard)
- Verify sensor indices align with MJCF model sensor layout for each species

---

## Phase 5: Shared Reward Test Patterns

**Impact:** ~200 lines removed + test parity across species | **Risk:** Low | **Files touched:** 4

### 5.1 Create shared reward test base

Create `environments/shared/tests/reward_test_helpers.py` with parameterized
test patterns:

```python
def assert_alive_bonus_positive(env):
def assert_energy_penalty_structure(env):
def assert_approach_reward_zero_on_first_step(env):
def assert_reward_components_sum_to_total(env, n_steps=10):
def assert_zero_weights_zero_rewards(env, weight_name, reward_key):
```

### 5.2 Expand T-Rex and Brachiosaurus reward tests

Currently:
- Raptor: 30 reward tests (comprehensive)
- Brachiosaurus: 10 tests (missing gait/posture/smoothness)
- T-Rex: 7 tests (minimal)

After this phase, all three species should have equivalent depth by inheriting
shared test patterns and adding species-specific tests.

### 5.3 New tests to add

**For T-Rex** (`test_trex_rewards.py`):
- Nosedive penalty tests
- Height maintenance reward tests (unique to T-Rex)
- Heading alignment tests
- Gait symmetry tests
- Action smoothness tests
- Curriculum stage reward weight tests (Stage 1 vs Stage 3)

**For Brachiosaurus** (`test_brachio_rewards.py`):
- Gait stability tests
- Food reach distance threshold tests
- Posture/tilt reward tests

---

## Phase 6: Notebook Cleanup

**Impact:** ~90% reduction in notebook duplication | **Risk:** Low

### 6.1 Consolidate training notebooks

The three species training notebooks (`velociraptor_training.ipynb`,
`brachiosaurus_training.ipynb`, `trex_training.ipynb`) are 99% identical.

**Option A — Single parameterized notebook** (recommended):
Create `notebooks/training.ipynb` with a species selector cell at the top:

```python
# === CONFIGURATION ===
SPECIES = "velociraptor"  # Change to: "velociraptor", "trex", "brachiosaurus"
ALGORITHM = "ppo"         # Change to: "ppo", "sac"
```

All subsequent cells use `SPECIES` and `ALGORITHM` variables. The three
original notebooks are archived or removed.

**Option B — Template generation**:
Create `notebooks/generate_training_notebook.py` that generates species-specific
notebooks from a Jinja2 template. This preserves the separate-notebook UX
while eliminating source duplication.

### 6.2 Fix common notebook issues

- **Path resolution**: Replace hardcoded `/content/mesozoic-labs/` with
  dynamic path detection that works in both Colab and local environments
- **Config validation**: Add pre-flight cell that verifies TOML configs exist
  before training starts
- **Error handling**: Add try/except around imports with clear error messages

### 6.3 JAX notebook cleanup (`jax_training.ipynb`)

- Extract hardcoded sensor indices (`S_GYRO`, `S_ACCEL`, etc.) to use
  shared constants from `environments.shared.constants`
- Extract hardcoded environment constants (`FRAME_SKIP=5`, `HEALTHY_Z_MIN=0.4`)
  to use TOML config via `load_stage_config()`
- Add markdown cell documenting this is experimental / not the primary path
- Add pre-flight GPU availability check

### 6.4 Utility notebook cleanup

- **`vertex_ai_sweep.ipynb`**: Add validation for GCP project settings,
  extract duplicated trial submission logic into helper function
- **`google_drive_summary.ipynb`**: Make `LOGS_DIR` configurable via
  environment variable with sensible default

---

## Phase 7: Minor Cleanup

**Impact:** Code hygiene | **Risk:** None

### 7.1 Naming consistency

| Item | Action |
|------|--------|
| `upload_to_gcs()` in `shared/config.py` | Rename to `_upload_to_gcs()` (internal helper) |
| `load_stage_config` in `shared/__init__.py` | Already in `__all__` — confirmed, no action needed |

### 7.2 Website unused exports

| Item | Action |
|------|--------|
| `getConsentStatus()` in CookieConsent | Remove (never imported anywhere) |
| `resetConsentStatus()` in CookieConsent | Either wire to `CookieResetButton` or remove |

### 7.3 Update documentation

- Update `CHANGELOG.md` with v0.3.0 consolidation entries
- Update `REFACTORING.md` to mark completed items
- Update `ROADMAP.md` to check off codebase consolidation

---

## New & Expanded Unit Tests Summary

Across all phases, these are the new test files and test expansions:

### New test files

| File | Tests | What it covers |
|------|-------|----------------|
| `shared/tests/test_train_base.py` | ~15 | Training utility functions, SpeciesConfig, argparse |
| `shared/tests/test_test_env_base.py` | ~6 | Shared test utilities execute correctly per species |
| `shared/tests/test_constants.py` | ~5 | Constants match MJCF models, regression guards |
| `shared/tests/reward_test_helpers.py` | ~8 | Shared reward assertion functions |

### Expanded existing test files

| File | New tests | What's added |
|------|-----------|-------------|
| `shared/tests/test_base_env.py` | +10 | Reward helpers, termination helpers, spawn_target, sensor constants |
| `trex/tests/test_trex_rewards.py` | +15 | Nosedive, height, heading, gait, smoothness, curriculum stages |
| `brachio/tests/test_brachio_rewards.py` | +8 | Gait stability, food reach threshold, posture |
| `trex/tests/test_trex_env.py` | +3 | Termination conditions (fall, tilt, bite success) |
| `brachio/tests/test_brachio_env.py` | +3 | Termination conditions (fall, tilt, food reach) |

### Cross-species integration tests

Add `shared/tests/test_species_integration.py`:
- `test_gymnasium_make_all_species()` — verify `gym.make("MesozoicLabs/Raptor-v0")` etc. work
- `test_config_env_kwargs_match_constructor()` — verify TOML env_kwargs are valid for each species
- `test_all_species_deterministic()` — verify determinism for all three species
- `test_all_species_observation_no_nan()` — run 200 steps, assert no NaN/Inf in any species

**Estimated total new tests: ~70+**
**Estimated total tests after cleanup: ~250+** (up from ~155)

---

## Implementation Order & Dependencies

```
Phase 1 (train_base.py)          ← Biggest impact, do first
  └── Phase 2 (test_env_base.py) ← Can start immediately after Phase 1
Phase 3 (env method extraction)  ← Independent of Phases 1-2
  └── Phase 4 (constants)        ← Depends on Phase 3
Phase 5 (reward test patterns)   ← Independent, can parallel with 3-4
Phase 6 (notebooks)              ← Depends on Phase 1 (uses train_base.py)
Phase 7 (minor cleanup)          ← Independent, do anytime
```

Phases 1-2 and Phases 3-5 can be worked in parallel if desired.

---

## Verification Checklist

After each phase, verify:

- [ ] `pytest` passes (all existing + new tests)
- [ ] `ruff check .` passes (no lint errors)
- [ ] `ruff format --check .` passes (formatting OK)
- [ ] `mypy environments/` passes (no type errors)
- [ ] `pytest --cov` shows ≥70% coverage
- [ ] Each species' `train_sb3.py` still works as a CLI entry point
- [ ] `gym.make("MesozoicLabs/Raptor-v0")` etc. still work
- [ ] No import cycles introduced

---

## What We're NOT Changing

Per the REFACTORING.md analysis, these are confirmed healthy and stay as-is:

- **`BaseDinoEnv` class hierarchy** — sound architecture
- **TOML config system** — well-designed
- **Gymnasium registration** — working correctly
- **Curriculum system** — well-tested
- **MuJoCo MJCF models** — species-specific, no duplication
- **CI/CD pipelines** — functioning
- **Website code** (beyond the two unused exports)
