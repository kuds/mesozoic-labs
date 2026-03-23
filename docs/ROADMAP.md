# Mesozoic Labs - Roadmap & Timeline

> Last updated: 2026-03-19

This roadmap organizes the project's growth into six phases. Each phase builds on
the previous one. Items within a phase can often be worked in parallel.

Legend: `[x]` done | `[-]` in progress | `[ ]` not started

---

## Progress Summary

| Phase | Name | Status | Done | Remaining |
|-------|------|--------|------|-----------|
| **0** | Clean Slate (v0.2.0) | **COMPLETE** | 5/5 items | — |
| **1** | First Steps (v0.3.0) | **In Progress** | 7/10 items | 3 training runs |
| **2** | Into the Wild (v0.4.0) | Not Started | 0/9 items | Blocked on Phase 1 training results |
| **3** | Evolution (v0.5.0) | Not Started | 0/8 items | Blocked on Phases 1-2 |
| **4** | The Pack (v0.6.0) | Not Started | 0/6 items | Blocked on Phase 3 species |
| **5** | Hyperdrive (v0.7.0) | **In Progress** | 2/5 items | JAX SAC, large-scale experiments |
| **6** | Life Finds a Way (v1.0.0) | Not Started | 0/5 items | Blocked on Phases 2-5 |

**Current focus:** Phase 1 — all infrastructure is in place (curriculum manager,
W&B tracking, metrics, Dockerfile, Vertex AI guide) and codebase consolidation
is complete. The remaining work is executing full 3-stage training runs for each
species and publishing results/checkpoints.

---

## Phase 0 — Foundation & Code Quality (Weeks 1-3)

Quick wins that pay dividends on everything that follows. Low effort, high leverage.

### Milestone: v0.2.0 — "Clean Slate"

- [x] **Externalize reward configs to TOML files**
  - Create `configs/` directory with per-species, per-stage TOML files
  - Refactor `train_sb3.py` and env constructors to load from config
  - Enables rapid reward experimentation without code changes
  - _Dependency: None_

- [x] **Register Gymnasium entry points**
  - Add `[project.entry-points."gymnasium.envs"]` to `pyproject.toml`
  - Users can then do `gym.make("MesozoicLabs/Velociraptor-v0")`
  - Auto-registration on `import environments`
  - _Dependency: None_

- [x] **Developer tooling**
  - Add `.pre-commit-config.yaml` with Ruff (format + lint), mypy
  - Add `[tool.mypy]` and `[tool.ruff]` sections to `pyproject.toml`
  - Add `pytest-cov` to dev dependencies; set coverage threshold (70%+)
  - Replace `print()` with `logging` module in training scripts
  - _Dependency: None_

- [x] **Package metadata & distribution prep**
  - Add authors, license, URLs, classifiers to `pyproject.toml`
  - Add `CHANGELOG.md` with semantic versioning
  - Add `CONTRIBUTING.md` with code style, PR process, testing requirements
  - Add GitHub issue templates (bug report, feature request, new species)
  - _Dependency: None_

- [x] **Testing improvements**
  - Add reward function unit tests (specific scenarios → expected values)
  - Add curriculum stage transition tests (config loading, reward changes)
  - Add `pytest-cov` coverage reporting to CI
  - _Dependency: None_

**Exit criteria:** All three species pass mypy, Ruff, and 70%+ test coverage.
Config-driven reward weights are loaded for at least one species.

---

## Phase 1 — Complete Core Training (Weeks 3-8)

Finish what's started: all three species fully trained through the 3-stage
curriculum with published results and reproducible checkpoints.

### Milestone: v0.3.0 — "First Steps"

- [ ] **Complete Velociraptor 3-stage training**
  - Run Stages 1-3 with PPO and SAC
  - Record reward curves, training GIFs, final evaluation metrics
  - Publish checkpoints as GitHub release artifacts
  - _Dependency: Phase 0 config externalization (helpful, not blocking)_

- [ ] **Complete Brachiosaurus 3-stage training**
  - Same as above for quadrupedal locomotion + food reach
  - _Dependency: None (parallel with Velociraptor)_

- [-] **T-Rex environment buildout**
  - [x] Implement `TRexEnv` subclass (env, tests, training script)
  - [x] The MJCF model, assets, and Gymnasium registration exist
  - [ ] Complete 3-stage training with PPO and SAC
  - _Dependency: None (parallel with other species)_

- [x] **Automated curriculum transitions**
  - Build `CurriculumManager` class that monitors eval metrics
  - Auto-advance stages when performance thresholds are met
  - `CurriculumCallback` for SB3 integration (stops training on threshold)
  - `curriculum` subcommand in all training scripts for end-to-end runs
  - Per-stage thresholds defined in TOML config `[curriculum]` sections
  - `thresholds_from_configs()` helper to extract thresholds from TOML
  - _Dependency: Phase 0 config externalization_

- [x] **W&B experiment tracking integration**
  - Add `wandb` to optional dependencies
  - `WandbCallback` for SB3 logs per-component rewards and hyperparameters
  - Save git commit hash and full config snapshot per run
  - Video recording of evaluation episodes via `WandbCallback(video_env=...)`
  - _Dependency: None_

- [x] **Expanded evaluation metrics**
  - Gait symmetry (left-right phase difference)
  - Cost of transport (energy / distance / weight)
  - Stride frequency and regularity
  - Forward velocity consistency (std dev)
  - Time-to-target (for Stage 3)
  - `LocomotionMetrics` class with `aggregate_episodes()` for multi-episode reports
  - _Dependency: None_

- [x] **Dockerfile for containerized training**
  - Python 3.11-slim base with MuJoCo headless rendering (`MUJOCO_GL=osmesa`)
  - Installs training dependencies via `pip install -e ".[train]"`
  - Packages `environments/` and `configs/` for cloud deployment
  - _Dependency: None_

- [x] **Vertex AI cloud training guide**
  - Step-by-step guide for single-stage and full curriculum training on GCP
  - GCS checkpoint persistence configuration
  - W&B integration with Vertex AI
  - Parallel multi-species training job submission
  - Machine type recommendations and cost estimates
  - Spot/preemptible VM configuration for cost savings
  - Published at `website/docs/training/vertex-ai.md`
  - _Dependency: Dockerfile_

- [x] **Full linting and type-checking compliance**
  - Fixed 92 Ruff lint errors, formatted 25 files, fixed 40 mypy type errors
  - All checks pass: `ruff check`, `ruff format`, `mypy` (0 errors)
  - All 155 tests passing with pytest
  - _Dependency: Phase 0 developer tooling_

- [x] **Codebase consolidation**
  - Extracted shared training logic into `environments/shared/train_base.py` (~2,000 lines removed)
  - Extracted shared test utilities into `environments/shared/test_env_base.py` (~500 lines removed)
  - Moved common reward calculations to `BaseDinoEnv` helper methods (~300 lines removed)
  - Centralized sensor constants and VecNormalize defaults in `environments/shared/constants.py`
  - Created shared reward test helpers in `environments/shared/tests/reward_test_helpers.py`
  - Cleaned up unused website exports and naming inconsistencies
  - Consolidated training notebooks into single parameterized `notebooks/training.ipynb`
  - See [CODE_CONSOLIDATION.md](CODE_CONSOLIDATION.md) for full implementation plan
  - _Completed: 2026-03-19_

**Exit criteria:** All three species have published training results (PPO + SAC),
downloadable checkpoints, a fully automated end-to-end curriculum pipeline, and
consolidated codebase with shared training/test infrastructure.

---

## Phase 2 — Simulation Realism & Robustness (Weeks 8-14)

Make the simulation closer to reality and the policies more robust. This is the
critical bridge between "cool demo" and "transferable research."

### Milestone: v0.4.0 — "Into the Wild"

- [ ] **Domain randomization**
  All parameters live in `BaseDinoEnv.__init__` with defaults of 0 (disabled),
  configurable via TOML `[env]` sections. Nominal physics values are cached at
  init; randomization is applied relative to XML defaults each `reset()`.
  - **Ground friction** (`friction_range`): Multiplicative per-episode scaling
    of all geom friction coefficients (e.g. `[0.8, 1.2]` = +/-20%). Teaches
    handling of slippery and grippy surfaces.
  - **Joint damping** (`joint_damping_range`): Per-joint multiplicative scaling
    each episode (e.g. `[0.9, 1.1]`). Simulates wear, temperature variation,
    and model uncertainty in joint dynamics.
  - **Gravity perturbation** (`gravity_range`): Additive offset on Z-gravity
    each episode (e.g. `[-0.5, 0.5]` around -9.81 m/s²). Prevents overfitting
    to the exact gravity constant.
  - **Actuator strength** (`actuator_strength_range`): Per-actuator
    multiplicative scaling each episode (e.g. `[0.9, 1.1]`). Forces the policy
    to handle motor strength variation.
  - **External force perturbations** (`push_force_scale`, `push_interval`):
    Random 3D pushes applied to the root body every N steps during the episode
    (e.g. 5 N every 100 steps). Most impactful single technique for locomotion
    balance recovery.
  - Introduce in Stage 2+ configs only — Stage 1 (balance) should train
    without perturbations so the agent learns to stand first.
  - JAX/MJX path will need its own domain randomization approach (functional
    `model.replace()` instead of in-place mutation) — coordinate with Phase 5
  - _Dependency: Phase 1 (need working trained policies to evaluate against)_

- [ ] **Sensor noise & action delay**
  - **Observation noise** (`obs_noise_scale`): Additive Gaussian noise applied
    to the full observation vector each step (e.g. std dev 0.01). Simulates
    sensor imprecision and prevents reliance on unrealistically precise state.
    Implemented in `BaseDinoEnv.step()` after `_get_obs()`.
  - Add accelerometer bias drift model
  - Add touch sensor activation thresholds
  - Simulate 1-3 step action delay to model communication latency
  - All noise parameters in TOML config `[env]` sections, disabled by default
  - _Dependency: Phase 0 config externalization_

- [ ] **Terrain diversity**
  - Implement heightfield-based procedural terrain (sinusoidal, rough, steps)
  - Add slopes (5-15 degrees) as training variants
  - Add varying ground friction zones
  - Parameterize terrain difficulty for curriculum integration
  - _Dependency: None_

- [ ] **MuJoCo model improvements**
  - Tune contact parameters (condim, solref, solimp) per foot geom
  - Add actuator dynamics (time constants, force limits)
  - Improve collision geometry (more detailed foot/claw shapes)
  - _Dependency: None_

- [ ] **Turning and steering**
  - Add yaw velocity reward component and target-relative heading
  - Train policies that can steer toward moving targets
  - Prerequisite for interesting predator-prey dynamics
  - _Dependency: Phase 1 (need locomotion policies as starting point)_

- [ ] **Stage 4: Prey Pursuit (predator species only)**
  - Add a 4th curriculum stage where prey moves on predefined paths
  - `PreyPath` module with four path types: straight retreat, angled
    retreat (30-60 degree offset), smooth arc, and zigzag evasion
  - Path type, speed, and stochastic parameters randomised per episode
    to prevent memorisation and encourage a robust pursuit policy
  - Prey position updated each simulation step via mocap body
  - `prey_velocity` (3D) added to predator observation spaces so the
    agent can anticipate prey movement (zeros in stages 1-3 for
    backward-compatible observation dimensions across all stages)
  - TOML configs: `prey_path_types`, `prey_speed_range`, longer episodes
    (1500 steps), and higher approach reward weight
  - Curriculum threshold: 10% strike/bite success rate on moving prey
  - Applies to Velociraptor and T-Rex only (Brachiosaurus food is static)
  - Bridges the gap between static targets (Stage 3) and fully learned
    prey behaviour (Phase 4), forcing the agent to develop heading
    correction, speed modulation, and sustained pursuit
  - _Dependency: Phase 1 Stages 1-3 + turning and steering_

- [ ] **Reactive scripted prey**
  - Extend Stage 4's predefined prey paths with reactive evasion behaviours
  - Prey behaviors: flee when predator is within range, random wandering,
    dodge on approach (reactive to predator position, not just clock-based)
  - Configurable prey speed, perception radius, and evasion strategy
  - _Dependency: Stage 4 prey pursuit foundation_

- [ ] **Hyperparameter optimization**
  - Add Optuna integration for systematic sweeps
  - Sweep over learning rate, batch size, entropy coefficient, gamma
  - Run for Velociraptor first, then transfer best configs to other species
  - _Dependency: Phase 1 W&B integration (for tracking sweep results)_

**Exit criteria:** Policies can walk on rough terrain, handle noisy observations,
recover from pushes, and chase moving prey (predefined paths in Stage 4,
reactive evasion in scripted prey). Published ablation studies showing the
impact of each robustness technique.

---

## Phase 3 — Advanced RL & New Species (Weeks 14-22)

Deepen the RL capabilities and expand the species roster.

### Milestone: v0.5.0 — "Evolution"

- [ ] **Custom policy networks**
  - Two-stream architecture: proprioception branch + exteroception branch
  - Implement via SB3 `CustomActorCriticPolicy`
  - Benchmark against default MLP on all three species
  - _Dependency: Phase 1 (need baselines to compare against)_

- [ ] **TD3 algorithm support**
  - Add TD3 training configs alongside PPO and SAC
  - Compare sample efficiency and final performance
  - _Dependency: Phase 0 config externalization_

- [ ] **Recurrent policies (LSTM/GRU)**
  - Add `sb3-contrib` with `RecurrentPPO`
  - Useful for partially observable scenarios and memory-dependent tasks
  - _Dependency: None_

- [ ] **Hierarchical RL architecture**
  - High-level controller: selects goals (approach, circle, strike, retreat)
  - Low-level controller: executes locomotion primitives
  - Options framework or feudal architecture
  - _Dependency: Phase 2 turning/steering_

- [ ] **Deinonychus (pack hunter species)**
  - MJCF model: mid-size bipedal raptor, optimized for agility
  - Gymnasium environment with pack-hunting-compatible observation space
  - Single-agent training first, multi-agent in Phase 4
  - _Dependency: Phase 1 (use Velociraptor as template)_

- [ ] **Compsognathus (small fast biped)**
  - MJCF model: small, lightweight, fast
  - Focus on speed and agility benchmarks
  - Good candidate for eventual physical robot (small, cheap)
  - _Dependency: Phase 1 (use Velociraptor as template)_

- [ ] **Stegosaurus (armored quadrupedal defender)**
  - MJCF model: heavy quadruped with dorsal plates and thagomizer tail weapon
  - **First defensive species** — fundamentally different from all existing
    forward-attack species. Agent must orient its *rear* toward threats and
    swing the thagomizer, flipping the reward structure entirely.
  - **Unique biomechanics:**
    - High center of mass from dorsal plates creates a harder balance problem
    - Tail is the active weapon (unlike head-forward predators), requiring
      rotational/pivoting locomotion rather than linear sprinting
    - Asymmetric mass distribution from plates affects turning inertia
    - Slow, deliberate quadrupedal gait — emphasis on stability over speed
  - **Curriculum (3 stages):**
    1. **Balance** — learn stable stance with elevated CoM from dorsal plates
    2. **Locomotion** — slow quadrupedal walking with turning/pivoting emphasis
    3. **Tail Defense** — detect approaching threat, pivot to orient rear
       toward threat, swing thagomizer for contact (defensive strike bonus)
  - **What makes it different from Brachiosaurus:**
    - Brachiosaurus reaches *up* (neck); Stegosaurus strikes *behind* (tail)
    - Brachiosaurus moves toward food; Stegosaurus holds ground against threats
    - Dorsal plates vs. long neck create opposite CoM challenges
    - Introduces rotational control as the primary locomotion objective
  - Reuses quadrupedal infrastructure from `BaseDinoEnv` and Brachiosaurus
  - _Dependency: Phase 1 (use Brachiosaurus as quadrupedal template)_

- [ ] **Benchmark suite**
  - Standardized evaluation scenarios: flat, slope, rough, perturbation
  - Automated benchmark runner producing structured JSON reports
  - Leaderboard across species, algorithms, and training configs
  - _Dependency: Phase 2 terrain diversity_

**Exit criteria:** 6 species in the roster (including first defensive species),
hierarchical policies demonstrating goal-directed behavior, published benchmark
results across algorithms.

---

## Phase 4 — Multi-Agent & Ecosystem (Weeks 22-32)

The signature feature: coordinated pack hunting and interspecies dynamics.

### Milestone: v0.6.0 — "The Pack"

- [ ] **Multi-agent environment framework**
  - PettingZoo-compatible multi-agent wrapper around BaseDinoEnv
  - Shared physics simulation with multiple dinosaurs
  - Configurable number of agents and species mixing
  - _Dependency: Phase 3 Deinonychus species_

- [ ] **2-agent cooperative hunting**
  - Two Deinonychus coordinating to corner prey
  - CTDE (centralized training, decentralized execution)
  - Emergent flanking and herding behaviors
  - _Dependency: Multi-agent framework + Phase 2 scripted prey_

- [ ] **Predator-prey with learned prey**
  - Train prey (Compsognathus) to evade
  - Train predator (Velociraptor/Deinonychus) to catch
  - Co-evolution / self-play training loop
  - _Dependency: Multi-agent framework + Compsognathus species_

- [ ] **Inter-agent communication**
  - Learned latent message passing between pack members
  - Analyze emergent communication protocols
  - _Dependency: 2-agent cooperative hunting_

- [ ] **Pre-trained model zoo**
  - Publish all checkpoints on Hugging Face Hub
  - Per-species, per-stage, per-algorithm checkpoints
  - Include training configs and evaluation results
  - Loading script: `load_pretrained("velociraptor", stage=3, algo="sac")`
  - _Dependency: Phase 3 (need a critical mass of trained models)_

- [ ] **Web-based visualizer**
  - MuJoCo WASM build or Three.js renderer
  - Load and run pre-trained policies in the browser
  - Interactive: change target position, apply perturbations
  - _Dependency: Pre-trained model zoo_

**Exit criteria:** Compelling demo of pack hunting with emergent coordination.
Model zoo with downloadable checkpoints. Browser-based interactive demo.

---

## Phase 5 — JAX/MJX & Performance (Weeks 24-34)

Parallel track that can start alongside Phase 4. GPU-accelerated batch simulation.

### Milestone: v0.7.0 — "Hyperdrive"

- [x] **MJX environment for all species**
  - Implemented `MJXDinoEnv` in `environments/shared/mjx_env.py` with
    functional JAX-native API (`jax.jit`/`jax.vmap` compatible)
  - Batch simulation (2048 parallel environments on GPU by default)
  - All three species (T-Rex, Velociraptor, Brachiosaurus) supported
    via per-species `mjx_config.py` registration modules
  - _Completed_

- [x] **JAX-native PPO training pipeline**
  - Implemented `jax_ppo.py` (Flax actor-critic, GAE, PPO clipped loss)
    and `jax_training.py` (full training loop with curriculum support)
  - `jax_training.ipynb` Colab notebook with A100 GPU support
  - Network architecture matches SB3 configs: `[512, 256]`
  - _Completed_

- [ ] **JAX-native SAC implementation**
  - Build `jax_sac.py` module with replay buffer, twin Q-networks, and
    automatic entropy tuning (analogous to `jax_ppo.py`)
  - Add SAC option to `jax_training.ipynb` notebook alongside PPO
  - Compare sample efficiency and final performance vs JAX PPO and SB3 SAC
  - _Dependency: Brax PPO pipeline (reuse actor-critic infrastructure)_

- [x] **Port remaining species to MJX**
  - All three species (T-Rex, Velociraptor, Brachiosaurus) ported
  - Shared `MJXDinoEnv` with per-species config registration
  - _Completed (included in MJX environment implementation above)_

- [ ] **Large-scale training experiments**
  - Sweep over billions of steps (feasible with MJX speedup)
  - Discover training regimes not possible at CPU scale
  - _Dependency: Brax PPO pipeline_

**Exit criteria:** 100x+ speedup demonstrated. All species available on MJX backend.
Published benchmark comparison (CPU vs. GPU vs. TPU).

---

## Phase 6 — Sim-to-Real & Hardware (Weeks 32-48+)

Physical robot construction and sim-to-real policy transfer.

### Milestone: v1.0.0 — "Life Finds a Way"

- [ ] **Hardware abstraction layer**
  - Unified interface for sim and real: `step(action) → obs`
  - Servo command mapping (simulation units ↔ PWM/position/torque)
  - Sensor data normalization to match simulation observation format
  - _Dependency: Phase 2 sensor noise (policies must handle noisy input)_

- [ ] **ROS 2 bridge**
  - Gymnasium ↔ ROS 2 topic bridge
  - Publish joint commands, subscribe to sensor data
  - Integration with RViz for monitoring
  - _Dependency: Hardware abstraction layer_

- [ ] **Compsognathus physical prototype**
  - 3D-printed frame, servo-driven joints
  - ESP32 or Raspberry Pi onboard controller
  - IMU + joint encoder sensor suite matching simulation
  - Match MJCF model mass/dimension specs
  - _Dependency: Phase 3 Compsognathus species + HAL_

- [ ] **Sim-to-real transfer pipeline**
  - System identification: auto-tune MJCF parameters from real data
  - Policy distillation/quantization for embedded deployment
  - Real-world evaluation protocol and metrics
  - _Dependency: Physical prototype + domain randomization policies_

- [ ] **Documentation & publication**
  - Full technical report on sim-to-real transfer results
  - Open-source hardware design files (STL, BOM, assembly guide)
  - Blog post / video walkthrough
  - _Dependency: Successful sim-to-real demo_

**Exit criteria:** Physical robot walking with a policy trained entirely in
simulation. Open-source hardware designs. Published results.

---

## Timeline Overview

```
         Wk 1     Wk 4     Wk 8     Wk 14    Wk 22    Wk 32    Wk 48
          |        |        |         |         |         |         |
Phase 0:  |████████|        |         |         |         |         |
          | Foundation     |         |         |         |         |
Phase 1:  |    ████|████████|         |         |         |         |
          |     Core Training        |         |         |         |
Phase 2:  |        |    ████|█████████|         |         |         |
          |        | Simulation Realism         |         |         |
Phase 3:  |        |        |     ████|█████████|         |         |
          |        |        |  Advanced RL + Species     |         |
Phase 4:  |        |        |         |     ████|█████████|         |
          |        |        |         |  Multi-Agent + Ecosystem   |
Phase 5:  |        |        |         |       ██|█████████|██       |
          |        |        |         |    JAX/MJX (parallel track) |
Phase 6:  |        |        |         |         |     ████|█████████|
          |        |        |         |         | Sim-to-Real + HW  |
```

## Versioning Plan

| Version | Phase | Name | API Stability |
|---------|-------|------|---------------|
| v0.2.0 | 0 | Clean Slate | Config format may change |
| v0.3.0 | 1 | First Steps | Env API stable, config format stable |
| v0.4.0 | 2 | Into the Wild | Observation space grows (prey_velocity for predators), Stage 4 added |
| v0.5.0 | 3 | Evolution | Multi-species API stable |
| v0.6.0 | 4 | The Pack | Multi-agent API stable |
| v0.7.0 | 5 | Hyperdrive | MJX backend API stable |
| v1.0.0 | 6 | Life Finds a Way | Full API freeze, PyPI publish |

## Key Dependencies (Critical Path)

```
Config externalization (P0) → Automated curriculum (P1) → Domain randomization (P2)
                                                        → Sensor noise (P2)

Velociraptor training (P1) → Custom networks (P3) → Hierarchical RL (P3)
                           → Terrain (P2) → Benchmark suite (P3)

Locomotion (P1) → Turning (P2) → Stage 4 prey pursuit (P2) → Reactive prey (P2) → Multi-agent (P4)
                                                                                 → Learned prey (P4)

Velociraptor trained (P1) → MJX port (P5) → Brax PPO (P5) → Scale experiments (P5)

Compsognathus species (P3) → Physical prototype (P6) → Sim-to-real (P6)
Sensor noise policies (P2) → HAL (P6) → ROS 2 bridge (P6)
```
