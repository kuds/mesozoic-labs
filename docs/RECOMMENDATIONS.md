# Mesozoic Labs - Codebase Review & Recommendations

> Last updated: 2026-03-25

## Codebase Assessment

The project is well-architected for its current stage. The `BaseDinoEnv` abstract
base class provides clean code reuse, type hints are thorough throughout, docstrings
are comprehensive, and the 3-stage curriculum learning design is a sound pedagogical
approach to RL training. CI/CD is functional with per-species test jobs and a
Docusaurus deployment pipeline.

**v0.2.0 completed** the foundational recommendations: TOML config externalization,
Gymnasium entry points, developer tooling (Ruff, mypy, pre-commit), W&B integration,
automated curriculum transitions, evaluation metrics, and testing infrastructure.
See [CHANGELOG.md](../CHANGELOG.md) for details.

**v0.3.0 targets** codebase consolidation (see [REFACTORING.md](REFACTORING.md))
and completing all 3-stage training runs for each species.

The recommendations below cover what remains after the v0.2.0 work.

---

## 1. Reinforcement Learning Improvements

### 1.1 Reward Engineering

- **Implement reward normalization per-component**, not just the total reward via
  `VecNormalize`. Components on different scales (alive bonus at 0.1 vs. strike
  bonus at 500.0) create gradient dominance issues.
- **Consider learned reward shaping** or adversarial reward functions (GAIL/AIRL)
  for more natural locomotion gaits rather than hand-crafting every term.

### 1.2 Algorithm Diversity

- **Add TD3 (Twin Delayed DDPG)** as a third baseline. It often outperforms SAC on
  locomotion tasks and is available in SB3 with minimal additional code.
- **Implement custom policy networks** using SB3's `CustomActorCriticPolicy`.
  A two-stream network processing joint states and prey direction separately
  before merging would likely improve learning.
- **Add recurrent policies (LSTM/GRU)** for tasks requiring memory. SB3-contrib
  provides `RecurrentPPO`.

### 1.3 Curriculum Learning Enhancements

- **Add intermediate difficulty levels** within stages. For example, Stage 2 could
  progressively increase target speed or distance.
- **Implement domain randomization** within each stage: randomize body mass
  (within 10%), joint friction, ground friction, and initial pose perturbation.

### 1.4 Training Infrastructure

- **Create a benchmark suite** that evaluates trained policies on standardized
  scenarios (flat ground, slopes, perturbation recovery) and outputs a structured
  report.

- **Emit the per-joint action envelope as a stage artifact** (added 2026-07-27).
  The run bundle currently captures the action *rate* and a saturation scalar —
  `DiagnosticsCallback` logs `action_delta` and `action_saturation`, aggregated
  across actuators, on the `algo_timesteps` axis — but nothing per-joint, and
  `diagnostics.npz` holds only `tilt_angle`, `forward_vel`, `pelvis_height`,
  `energy`, foot contacts and reward components. The missing third piece is
  *which* joint used *how much of its range*, which is what
  `joint_excursion_report.py` computes today as a manual after-the-fact step.

  The argument for emitting it at stage end is not that it is cheap, though it
  is: 10 deterministic episodes is ~10 000 env steps, roughly 10 s against a
  3.7 h stage (<0.1%), reusing the model and env already in hand. It is that
  **the measurement is plant-version-locked**. Taking the T-Rex baseline after
  PR #464 merged already required pinning `a6c36aa`, re-cloning, and locating
  the checkpoint on Drive by hand; every future plant revision recreates that
  friction, and the T-Rex plant alone has moved through five physics revisions
  in a few months. Emitted at training time it is captured against the plant
  that was actually used, permanently, next to the `plant_identity.json` that
  proves which plant that was.

  Two implementation notes:
  - **Write structured data, not the pretty table.** A `joint_excursion.csv`
    (or extra arrays in `diagnostics.npz`) with per-actuator p5/p95/min/max,
    achieved range, span, used fraction and delta-per-step lets
    `compare_run_diagnostics.py` diff two runs mechanically. Keep the
    human-readable print in the script.
  - **Deterministic rollouts**, matching the eval videos. This matters: the
    same run measures ~2x higher per-step deltas under the stochastic training
    policy than deterministically, so mixing the two silently manufactures
    improvements. See `docs/TREX_LEG_FLEXING_PLAN.md`.

  Worth a cheap cross-check first: `action_saturation` is already logged to
  tensorboard, and the T-Rex baseline measured `used` = 100% on 20 of 21
  actuators. If that metric was already reading high during training, the
  signal existed and was going unread — which would argue for surfacing it in
  `stage_summary.txt` as well as adding the envelope artifact.

---

## 2. Robotics & Simulation Improvements

### 2.1 MuJoCo Model Quality

- **Add tendon and muscle models** instead of relying solely on position/motor
  actuators. MuJoCo supports tendons with routing for more biomechanically
  realistic locomotion.
- **Implement contact property tuning**: tune condim, conaffinity, solref, and
  solimp per-geom for improved foot-ground interaction realism.
- **Add actuator dynamics** (time constants, force limits) to model motor response
  delays.
- **Consider composite objects** for more detailed collision geometry on the bodies.

### 2.2 Terrain and Environment Diversity

- **Implement procedural terrain generation** using MuJoCo's heightfield support.
- **Add obstacle courses** with randomized placement.
- **Implement wind/perturbation forces** applied randomly to the torso during training.
- **Add varying ground friction zones** (mud, ice, rock).

### 2.3 Sim-to-Real Preparation

- **Implement system identification tools** comparing simulated vs. real sensor data.
- **Add sensor noise models** to observations during training (Gaussian noise on
  joint encoders, accelerometer bias drift, touch sensor thresholds).
- **Implement action delay/latency simulation** (1-3 control steps of delay).
- **Add an actuator calibration script** mapping simulation units to real servo
  commands.
- **Create a hardware abstraction layer** presenting the same interface for
  simulation and physical hardware.

---

## 3. Multi-Agent & Behavioral Complexity

### 3.1 Pack Hunting

- **Start with predator-prey with a scripted prey** before adding multi-agent
  complexity. Replace static mocap targets with rule-based evading prey.
- **Implement multi-agent environments** using PettingZoo or a custom multi-agent
  wrapper. Start with 2 velociraptors coordinating to corner prey.
- **Use centralized training with decentralized execution (CTDE)**.
- **Add communication channels** between agents (learned latent messages).

### 3.2 Behavioral Repertoire

- **Implement a behavior switching mechanism** for locomotion primitives (walk,
  trot, gallop, sprint, turn, stop).
- **Add turning and lateral movement** to the action/reward structure.
- **Implement a hierarchical RL architecture** with high-level goal selection and
  low-level locomotion primitives.

---

## 4. Code Quality & Developer Experience

### 4.1 Codebase Consolidation (v0.3.0)

See [REFACTORING.md](REFACTORING.md) for the full consolidation plan. Key items:

- **Extract shared training logic** into `environments/shared/train_base.py`
  (~2,000 lines of duplication across 3 species training scripts)
- **Extract shared test utilities** into `environments/shared/test_env_base.py`
  (~500 lines of duplication across 3 species test scripts)
- **Move common reward calculations** to `BaseDinoEnv` helper methods
- **Centralize constants** (sensor indices, VecNormalize defaults)

### 4.2 Testing

- **Add regression tests for trained policies**: save a reference trajectory from
  a trained checkpoint and verify reproducibility within tolerance.
- **Add performance/benchmark tests** measuring simulation step throughput.
- **Add property-based testing** (Hypothesis) for observation/action space
  invariants.

### 4.3 Package & Distribution

- **Publish to PyPI** once the API stabilizes.
- **Add Dependabot or Renovate** for automated dependency updates.

---

## 5. Documentation

- **Add a reward engineering guide** explaining each reward component's rationale,
  tuning process, and common failure modes.
- **Document the MJCF model design decisions** with references to paleontological
  literature on dinosaur biomechanics.
- **Create an architecture diagram** showing BaseDinoEnv, species environments,
  training scripts, and MuJoCo models.
- **Add a troubleshooting guide** for MuJoCo installation, training divergence,
  and environment debugging.
- **Document observation and action spaces** with labeled diagrams mapping indices
  to joints/sensors.

---

## 6. Long-Term Goals

### 6.1 JAX/MJX Migration

- Start with Velociraptor to measure the JAX/MJX GPU path against an SB3 baseline
- Use Brax's JAX-native PPO rather than adapting SB3
- Benchmark throughput on CPU vs. NVIDIA GPU; evaluate TPU only after adding and validating a supported TPU setup
- Maintain API compatibility with Gymnasium interface

### 6.2 Ecosystem Expansion

- **Model zoo** on Hugging Face Hub with per-species, per-stage checkpoints
- **Leaderboard** for locomotion benchmarks across species and algorithms
- **Web-based visualizer** using MuJoCo WASM or Three.js
- **Paleontology validation** of locomotion patterns against published studies

### 6.3 Real Robot Hardware

- Start with Compsognathus (small, cheap, 3D-printed servo biped)
- ESP32 or Raspberry Pi onboard controller
- ROS 2 bridge for Gymnasium-to-hardware integration
- Match MJCF model specs to physical build

---

## Priority Ranking

| # | Recommendation | Impact | Effort | Status |
|---|---------------|--------|--------|--------|
| 1 | Codebase consolidation | High | Medium | **Done** (v0.3.0) — see [CODE_CONSOLIDATION.md](CODE_CONSOLIDATION.md) |
| 2 | Scripted prey (moving target) | High | Medium | Not started |
| 3 | Sensor noise + action delay | High | Medium | Not started |
| 4 | Domain randomization | High | Medium | Not started |
| 5 | Terrain heightfields | High | Medium | Not started |
| 6 | Custom policy networks | High | Medium | Not started |
| 7 | MJX migration (Velociraptor) | Very High | High | **Done** — all 3 species ported to MJX |
| 8 | Multi-agent pack hunting | Very High | High | Not started |
| 9 | Pre-trained model zoo | Medium | Medium | Not started |
| 10 | Physical robot prototype | Very High | Very High | Not started |

## Completed (v0.2.0)

The following recommendations were implemented in v0.2.0:

- TOML config externalization (`configs/` directory, `environments/shared/config.py`)
- Gymnasium entry point registration (`MesozoicLabs/*-v0`)
- Developer tooling (Ruff format + lint, mypy, pre-commit hooks)
- Automated curriculum transitions (`CurriculumManager`, `CurriculumCallback`)
- W&B experiment tracking (per-component rewards, config snapshots, video recording)
- Evaluation metrics (`LocomotionMetrics` — gait symmetry, cost of transport, etc.)
- Package metadata, CHANGELOG, CONTRIBUTING, issue templates
- Reward function unit tests and curriculum stage transition tests
- `pytest-cov` with 70% coverage threshold
- `logging` module replacing `print()` in training scripts
