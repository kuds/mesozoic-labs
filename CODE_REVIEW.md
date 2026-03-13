# Code Review: Mesozoic Labs

## Executive Summary

This review covers the entire Mesozoic Labs codebase — a reinforcement learning platform
for training robotic dinosaur agents using MuJoCo physics simulation. The project is
well-architected with a solid shared infrastructure (`BaseDinoEnv`, curriculum manager,
training pipeline), but has accumulated significant duplication across the three species
implementations and their tests. There are also several code quality issues in the shared
utility modules.

**Key findings:**
- ~40-46% duplication across test files (~750 lines recoverable)
- ~95% boilerplate duplication in training scripts
- 12+ duplicated reward/observation helper patterns across species envs
- Several potential bugs in shared utilities (thread-unsafe CSV writes, missing error handling)
- Missing test coverage for critical infrastructure (learning rate schedules, callbacks, VecNormalize persistence)

---

## 1. Code Consolidation Opportunities

### 1.1 Environment Implementations (HIGH IMPACT)

The three species environments (`raptor_env.py`, `brachio_env.py`, `trex_env.py`) share
extensive duplicated code that should be extracted into `BaseDinoEnv`.

#### Duplicated Reward Helpers (Raptor & T-Rex are ~95% identical)

| Helper to extract | Files affected | Lines saved |
|---|---|---|
| `_compute_forward_velocity()` | raptor, trex | ~22 |
| `_compute_backward_penalty()` | raptor, trex | ~8 |
| `_compute_drift_penalty()` | raptor, trex | ~10 |
| `_compute_posture_reward()` | raptor, trex | ~8 |
| `_compute_nosedive_penalty()` | raptor, trex | ~8 |
| `_compute_angular_velocity_penalty()` | all three | ~12 |
| `_compute_tail_stability_reward()` | raptor, trex | ~14 |
| `_compute_heading_alignment()` | raptor, trex | ~18 |
| `_compute_lateral_velocity_penalty()` | raptor, trex | ~6 |
| `_compute_approach_shaping()` | all three | ~24 |

**Total: ~130 lines saved, improved maintainability**

#### Duplicated Quaternion Utilities

- `_quat_to_forward_direction_2d(quat)` — identical in raptor (L473-476) and trex (L404-410)
- `_quat_to_forward_z(quat)` — identical in raptor (L402-404) and trex (L361-363)

These should be static methods or standalone helpers in `BaseDinoEnv`.

#### Duplicated Observation Construction

All three species follow the same pattern in `_get_obs()`:
1. Extract qpos[7:] and qvel[6:]
2. Read gyro, accel, quat from sensor data at standard indices
3. Extract linear velocity from qvel[0:3]
4. Read foot contact sensors
5. Compute relative position/distance to target
6. Normalize and concatenate

A `_construct_base_obs_components()` helper in the base class would eliminate ~30 lines
of duplication per species.

#### Duplicated Termination Logic

`_is_terminated()` in raptor (L519-586) and trex (L460-526) are nearly identical:
- Height check, tilt check, nosedive check
- Contact loop and floor collision detection

The base class should provide the common termination logic with hooks for species-specific
success conditions and configurable geom sets.

#### Duplicated Target Spawning

`_spawn_target()` in raptor (L588-621) and trex (L528-554) are identical. Brachio is
similar but adds a height dimension. A generic implementation in the base class with
an overridable z-height parameter would eliminate ~40 lines.

### 1.2 Training Scripts (QUICK WIN)

The three training scripts are **95% identical boilerplate** (3 x 45 = 135 lines):
- `velociraptor/scripts/train_sb3.py`
- `brachiosaurus/scripts/train_sb3.py`
- `trex/scripts/train_sb3.py`

The only difference is the `SpeciesConfig` instantiation. These could be replaced with a
single config-driven entry point or a species registry, saving ~90 lines.

### 1.3 Test Files (HIGH IMPACT)

#### Static Balance Tests (~924 lines, ~45% duplicated)

**Duplicated helpers** across all three `test_static_balance.py` files:
- `_get_foot_contacts_xy()` — identical logic, only foot geom names differ
- `_com_xy()` — identical, only subtree body ID differs
- `_*_mass()` — identical pattern, only body name differs

**Duplicated test classes:**
- `TestHomePoseCOM` — 4 common tests across raptor/trex (brachio differs for quadrupedal)
- `TestZeroTorqueStability` — raptor/trex identical, brachio uses `TestInitialSettling`
- `TestJointLimitsAtHome` — very high duplication, only joint names differ
- `TestMassDistribution` — very high duplication, only thresholds and body names differ

**Recommendation:** Create a parametrized base test class in `shared/tests/test_static_balance_base.py`
that species-specific tests inherit from, overriding only configuration (body names, thresholds, joint lists).

#### Reward Tests (~561 lines, ~40% duplicated)

- `test_total_reward_is_sum_of_components()` — identical structure across all three, only reward key lists differ
- `test_stage3_*_approach_shaping()` — identical pattern across all three
- Weight-zeroing tests — repetitive `test_zero_X_weight_zeroes_X_reward()` pattern (10+6+2 = 18 tests)

**Recommendation:** Create test factories or parametrized base tests with species-specific reward key maps.

#### Environment Tests (~257 lines, ~15% duplicated)

- Geom ID caching tests follow the same pattern (raptor tail vs trex head)
- Could be parametrized with geom attribute lists

---

## 2. Code Quality Issues

### 2.1 Critical Issues

1. **Thread-unsafe CSV writes** (`config.py:195-207`)
   - `append_stage_result_csv()` rewrites the entire file when new keys are detected
   - Multiple concurrent training processes can corrupt or lose data
   - Fix: Add file locking (fcntl on Unix) or use a queue-based approach

2. **Empty metrics returns broken dict** (`metrics.py:142-144`)
   - `compute()` returns `{"error": float("nan")}` when no data recorded
   - Callers expecting keys like `"mean_forward_velocity"` will crash with KeyError
   - Fix: Return dict with NaN for all expected keys, or raise ValueError

3. **Missing numpy guard in WandB video** (`wandb_integration.py:254-259`)
   - `np.array(frames)` on line 256 will crash if numpy is None (optional import)
   - The check on line 234 only covers the early return case
   - Fix: Add defensive check before use or make numpy mandatory

4. **VecNormalize flag restoration not guarded** (`curriculum.py:467-471`)
   - `finally` block assumes `eval_env` has `.training` attribute
   - If it doesn't, the finally block crashes and masks the original exception
   - Fix: Use `getattr(self.eval_env, 'training', None)`

### 2.2 Design Issues

5. **Config module has too many responsibilities** (`config.py`)
   - Loads configs, serializes to JSON, appends CSV, uploads to GCS, manages type conversions
   - Violates Single Responsibility Principle
   - Consider splitting into: ConfigLoader, ConfigSerializer, ArtifactUploader

6. **Duplicate evaluation loops** (`curriculum.py:422-473` vs `541-610`)
   - `_run_supplementary_eval()` and `_on_step_standalone()` implement nearly identical episode loops
   - Extract shared `_run_eval_episodes()` helper

7. **Hardcoded config key names scattered across modules**
   - "env_kwargs", "ppo_kwargs", "curriculum_kwargs" appear as strings in 10+ files
   - Renaming any key requires updating many files
   - Consider a ConfigSchema class or constants module

8. **Complex nested conditionals in sweep** (`sweep/results.py:139-229`)
   - `_collect_trial_results()` has deeply nested logic
   - Each level (trial iteration → param extraction → metric extraction → gate evaluation)
     should be its own function

9. **Inconsistent error handling patterns**
   - `sweep/search_space.py` uses `logger.error()` + `sys.exit(1)` instead of exceptions
   - `wandb_integration.py` catches broad `except Exception`
   - `evaluation.py` silently catches ImportError
   - Recommendation: Define custom exception hierarchy and catch specific exceptions

10. **Magic string keys in sweep results** (`sweep/results.py:177-194`)
    - "best_mean_reward", "best_mean_episode_length" appear as raw strings in multiple places
    - Create a MetricSchema dataclass or constants module

### 2.3 Moderate Issues

11. **Indiscriminate list→tuple conversion** (`config.py:84-90`)
    - ALL lists in config are converted to tuples — could break if a parameter should remain a list

12. **Long function in diagnostics** (`diagnostics.py:136-198`)
    - `_on_rollout_end()` is 63 lines with repeated `float(np.mean(vals)) if vals else float("nan")` pattern
    - Extract `_safe_mean()` helper

13. **Poor variable names in visualization** (`visualization.py:225-248`)
    - `_w`, `_ri`, `_vals`, `_offset`, `_lbl` in a 100-line function reduce readability

14. **No input validation in public functions**
    - `load_stage_config()` doesn't validate species format
    - `_load_search_space_file()` doesn't check required JSON keys
    - `build_stage_results_from_eval_data()` assumes "env_kwargs" key exists

15. **Temporary directory cleanup is manual** (`sweep/results.py:681-684`)
    - Should use `contextlib.TemporaryDirectory` context manager to prevent leaks on exception

---

## 3. Test Coverage Gaps

### 3.1 Missing Unit Tests (Critical Infrastructure)

| Component | File | What's missing |
|---|---|---|
| Learning rate schedules | `train_base.py:108-130` | `linear_schedule()` and `cosine_schedule()` have zero tests |
| VecNormalize persistence | `train_base.py:310-320` | No tests for save/load of normalization stats |
| Callbacks | `diagnostics.py`, `curriculum.py` | `DiagnosticsCallback`, `EvalCollapseEarlyStopCallback`, `RewardRampCallback`, `StageWarmupCallback` untested |
| HPT metrics reporting | `train_base.py:500-654` | `_report_hpt_metrics()` not tested |
| GCS TensorBoard buffering | `train_base.py:136-175` | `_is_gcs_path()`, `_make_local_tb_dir()`, `_sync_tb_to_gcs()` untested |
| Determinism | All env files | No tests that seeded resets produce identical episodes |
| Evaluation quality | `evaluation.py` | `eval_policy_quality()` not tested |
| Model I/O round-trip | All species | No tests for save/load preserving policy weights |

### 3.2 Missing Integration Tests

- No end-to-end curriculum advancement test (stage 1→2→3 progression)
- No test for `CurriculumManager.advance()` threshold logic
- No test for keyboard interrupt recovery during training
- No test for TensorBoard sync failure graceful degradation

### 3.3 Estimated Test Code Needed

~400-500 lines of new tests to cover the critical gaps listed above.

---

## 4. Recommended Action Plan

### Phase 1: Quick Wins (Low Risk, Immediate Value)

1. Extract duplicated quaternion utilities to `BaseDinoEnv` static methods
2. Extract common reward helper methods to `BaseDinoEnv`
3. Add `_safe_mean()` helper to diagnostics
4. Fix empty metrics `compute()` return value
5. Add numpy guard in WandB video recording
6. Add VecNormalize flag guard in curriculum callbacks

### Phase 2: Test Consolidation (Medium Risk, High Value)

7. Create `shared/tests/test_static_balance_base.py` parametrized base class
8. Create reward test factories for common patterns
9. Add unit tests for learning rate schedules
10. Add unit tests for GCS path utilities

### Phase 3: Environment Refactoring (Medium Risk, High Value)

11. Extract common observation construction to base class
12. Extract common termination logic to base class
13. Extract common target spawning to base class
14. Consolidate training scripts into config-driven entry point

### Phase 4: Architecture Improvements (Higher Risk)

15. Split config.py into focused modules
16. Extract duplicate evaluation loops in curriculum.py
17. Create ConfigSchema constants for shared key names
18. Add file locking to CSV writes
19. Refactor sweep/results.py nested conditionals
