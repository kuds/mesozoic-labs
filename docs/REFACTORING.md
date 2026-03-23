# Codebase Consolidation Plan (v0.3.0)

> **Status:** **COMPLETE** — all consolidation phases implemented
> **Completed:** 2026-03-19
> **Actual reduction:** ~3,000+ lines of redundant code removed

---

## Summary

The three species directories (`brachiosaurus/`, `trex/`, `velociraptor/`) have
accumulated significant code duplication from rapid parallel development. Training
scripts, test utilities, and environment methods are ~90% identical across species,
with only the environment class and a few labels differing. This document tracks
the consolidation work planned for v0.3.0.

---

## 1. Training Scripts — CRITICAL (~2,000 duplicated lines)

The three `train_sb3.py` files (920-993 lines each) are nearly identical. Nine
functions are copy-pasted verbatim across all three species.

### Duplicated Functions

| Function | Lines (each) | Identical? | Species-specific part |
|----------|-------------|------------|----------------------|
| `linear_schedule()` | 7 | 100% | None |
| `_cast_value()` | 16 | 100% | None |
| `_apply_overrides()` | 28 | 100% | None |
| `make_env()` | 13 | ~95% | Env class name only |
| `create_vec_env()` | 16 | 100% | None |
| `train()` | ~210 | 100% | None |
| `train_curriculum()` | ~260 | 100% | None |
| `evaluate()` | ~130 | ~90% | Metric labels ("Strike" vs "Bite" vs "Food Reaching") |
| `main()` argparse | ~175 | 100% | None |

### Files

- `environments/brachiosaurus/scripts/train_sb3.py` (946 lines)
- `environments/trex/scripts/train_sb3.py` (993 lines)
- `environments/velociraptor/scripts/train_sb3.py` (920 lines)

### Consolidation Plan

Create `environments/shared/train_base.py` containing all shared training logic.
Each species' `train_sb3.py` reduces to a thin wrapper:

```python
"""Brachiosaurus training script."""
from environments.brachiosaurus.envs import BrachioEnv
from environments.shared.train_base import main as train_main

SPECIES = "brachiosaurus"
ENV_CLASS = BrachioEnv
STAGE3_LABEL = "Food Reaching"

if __name__ == "__main__":
    train_main(species=SPECIES, env_class=ENV_CLASS, stage3_label=STAGE3_LABEL)
```

Each species script goes from ~950 lines to ~15 lines.

---

## 2. Test Scripts — HIGH (~500 duplicated lines)

The three `test_env.py` scripts share identical test functions with only the
environment class differing.

### Duplicated Functions

| Function | Brachio | TRex | Raptor | Identical? |
|----------|---------|------|--------|------------|
| `test_basic_functionality()` | Lines 33-75 | Lines 26-65 | Lines 33-80 | Yes |
| `test_episode_rollout()` | Lines 78-120 | Lines 68-109 | Lines 83-126 | Yes |
| `test_reward_components()` | Lines 123-162 | Missing | Lines 129-169 | Yes (2 of 3) |
| `test_determinism()` | Lines 165-198 | Lines 112-142 | Lines 172-208 | Yes |
| `test_observation_bounds()` | Lines 201-242 | Missing | Lines 211-254 | Yes (2 of 3) |

### Files

- `environments/brachiosaurus/scripts/test_env.py` (272 lines)
- `environments/trex/scripts/test_env.py` (169 lines)
- `environments/velociraptor/scripts/test_env.py` (284 lines)

### Consolidation Plan

Create `environments/shared/test_env_base.py` with parameterized test functions
that accept an env class. Species scripts become thin wrappers. Also add the
missing `test_reward_components()` and `test_observation_bounds()` to T-Rex.

---

## 3. Environment Methods — MEDIUM (~300+ duplicated lines)

The species environment files share significant implementation patterns that
could be lifted into `BaseDinoEnv` or shared utility functions.

### Duplicated Patterns

**Sensor index setup** (identical in all 3 `_cache_ids()` methods):
```python
self._sensor_gyro_start = 0
self._sensor_accel_start = 3
self._sensor_quat_start = 6
```

**Observation assembly** (`_get_obs()` — same structure in all 3):
- Joint positions/velocities extraction
- Sensor data slicing (gyro, accel, quat)
- Target-relative direction and distance calculation
- Foot contact array assembly
- Final `np.concatenate(...)` call

**Reward calculations** (`_get_reward_info()` — shared patterns):
- Forward velocity toward target (direction dot product)
- Energy penalty (squared action norm / n_actuators)
- Approach shaping (distance delta / max_delta)
- Gait symmetry (contact asymmetry between left/right feet)
- Alive bonus

**Termination logic** (`_is_terminated()` — shared pattern):
- Height check against `healthy_z_range`
- Tilt angle check against `max_tilt_angle`
- Contact detection loop (`for i in range(self.data.ncon)`)

**Target spawning** (`_spawn_target()` — ~95% identical):
- Random distance/lateral/height from configured ranges
- Mocap position update
- Previous distance reset

### Files

- `environments/brachiosaurus/envs/brachio_env.py` (344 lines)
- `environments/trex/envs/trex_env.py` (495 lines)
- `environments/velociraptor/envs/raptor_env.py` (500 lines)

### Consolidation Plan

1. Move common sensor constants to `BaseDinoEnv` or a shared constants module
2. Add `_compute_forward_velocity()`, `_compute_energy_penalty()`,
   `_compute_approach_shaping()` helper methods to `BaseDinoEnv`
3. Move `_spawn_target()` to `BaseDinoEnv` with override hooks for
   species-specific height ranges
4. Extract contact-checking loop into a shared utility method

---

## 4. Hardcoded Constants — LOW (scattered across files)

Constants repeated across multiple files that should be centralized.

| Constant | Value | Locations |
|----------|-------|-----------|
| Sensor indices | `gyro=0, accel=3, quat=6` | All 3 env files |
| VecNormalize params | `norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0` | All 3 training scripts |
| Tail angular vel max | `10.0 rad/s` | TRex, Raptor env files |
| Frame skip | `5` | Multiple env configs |

### Consolidation Plan

Create `environments/shared/constants.py` for simulation-wide defaults.
Per-species constants stay in their respective env files.

---

## 5. Minor Cleanup Items

### Unused Website Exports

| Item | Location | Action |
|------|----------|--------|
| `getConsentStatus()` | `website/src/components/CookieConsent/index.tsx:118-121` | Remove (never imported) |
| `resetConsentStatus()` | `website/src/components/CookieConsent/index.tsx:123-126` | Either use in `CookieResetButton` or remove |

### Naming Consistency

| Item | Location | Action |
|------|----------|--------|
| `upload_to_gcs()` | `environments/shared/config.py:212` | Rename to `_upload_to_gcs()` (internal helper) |
| `load_stage_config` | `environments/shared/__init__.py` | Add to `__all__` list for export consistency |

### Notebook Review

| Notebook | Status | Action |
|----------|--------|--------|
| `jax_training.ipynb` | Potentially stale | Verify if JAX path is still maintained; archive or mark experimental if not |
| `google_drive_summary.ipynb` | Utility | Verify relevance |

---

## 6. Reward Test Files — MEDIUM (~200 duplicated lines)

The three `test_*_rewards.py` files follow identical testing methodology with
species-specific reward component names.

### Files

- `environments/brachiosaurus/tests/test_brachio_rewards.py`
- `environments/trex/tests/test_trex_rewards.py`
- `environments/velociraptor/tests/test_raptor_rewards.py`

### Consolidation Plan

Extract shared reward test patterns into a parameterized base in
`environments/shared/tests/` that species tests can inherit from.

---

## Implementation Order

| Step | Task | Impact | Risk |
|------|------|--------|------|
| 1 | Extract `train_base.py` from training scripts | ~2,000 lines saved | Medium — training entry points change |
| 2 | Extract `test_env_base.py` from test scripts | ~500 lines saved | Low — test-only change |
| 3 | Move common reward calculations to `BaseDinoEnv` | ~300 lines saved | Medium — touches core env logic |
| 4 | Move `_spawn_target()` to base class | ~50 lines per species | Low — well-isolated method |
| 5 | Centralize sensor constants | Cleaner code | Low — constants only |
| 6 | Clean up website unused exports | Minimal | None |
| 7 | Rename `upload_to_gcs()` to private | Naming consistency | None |
| 8 | Extract shared reward test patterns | ~200 lines saved | Low — test-only change |

**Recommended approach:** Do steps 1-2 first (biggest impact, isolated changes),
then steps 3-5 (env internals), then 6-8 (minor cleanup).

---

## What's NOT a Problem

The analysis confirmed these areas are healthy:

- **No dead code** — All shared module exports are actively used
- **No orphaned files** — Every Python file is imported or a legitimate entry point
- **No commented-out code blocks** — Codebase is clean
- **No unused imports** — Verified across all shared and env modules
- **No deprecated markers** — Nothing flagged for removal
- **Architecture is sound** — `BaseDinoEnv` hierarchy, TOML configs, Gymnasium
  registration, and curriculum system are all well-designed
- **Test coverage is good** — 70% threshold is met, tests are meaningful
