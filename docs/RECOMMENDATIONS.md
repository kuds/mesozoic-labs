# Mesozoic Labs - Codebase Review & Recommendations

## Codebase Assessment

The project is well-architected for its current stage. The `BaseDinoEnv` abstract base class provides clean code reuse (~65% shared logic), type hints are thorough throughout, docstrings are comprehensive, and the 3-stage curriculum learning design is a sound pedagogical approach to RL training. CI/CD is functional with per-species test jobs and a Docusaurus deployment pipeline.

That said, there are concrete areas where targeted investment would significantly accelerate the project's trajectory. The recommendations below are organized from foundational improvements to ambitious long-term goals.

---

## 1. Reinforcement Learning Improvements

### 1.1 Reward Engineering Infrastructure

The current reward functions are hand-tuned with hardcoded weights. This is the single biggest bottleneck for training quality.

**Recommendations:**
- ~~**Externalize reward configs to YAML/TOML files** instead of embedding weights in Python code and training scripts. This enables rapid experimentation without code changes and creates a clear audit trail of what was tried.~~ **Done (v0.2.0)** — TOML configs in `configs/` directory, loaded via `environments/shared/config.py`.
- ~~**Add reward logging dashboards** beyond TensorBoard scalars. Plot per-component reward distributions over training, not just totals. This reveals which components dominate and which are being ignored by the policy.~~ **Done (v0.2.0)** — `WandbCallback` logs per-component rewards.
- **Implement reward normalization per-component**, not just the total reward via `VecNormalize`. Components on different scales (alive bonus at 0.1 vs. strike bonus at 500.0) create gradient dominance issues.
- **Consider learned reward shaping** or adversarial reward functions (GAIL/AIRL) for more natural locomotion gaits rather than hand-crafting every term.

### 1.2 Algorithm Diversity

The project currently uses only PPO and SAC from Stable-Baselines3 with default MLP policies.

**Recommendations:**
- **Add TD3 (Twin Delayed DDPG)** as a third baseline. It often outperforms SAC on locomotion tasks and is available in SB3 with minimal additional code.
- **Implement custom policy networks** using SB3's `CustomActorCriticPolicy`. Locomotion benefits from architectures that separate proprioception from exteroception (target tracking). A two-stream network processing joint states and prey direction separately before merging would likely improve learning.
- **Add recurrent policies (LSTM/GRU)** for tasks requiring memory, such as pack hunting or navigating occluded environments. SB3-contrib provides `RecurrentPPO`.
- **Implement hyperparameter sweeps** with Optuna. The current per-stage hyperparameters were likely hand-tuned; systematic search over learning rate, batch size, entropy coefficient, and gamma would yield measurable gains.

### 1.3 Curriculum Learning Enhancements

The 3-stage curriculum is good but static. Each stage requires manual intervention to load the previous checkpoint and start the next phase.

**Recommendations:**
- ~~**Automate stage transitions** with a curriculum manager that monitors performance metrics and automatically advances stages when thresholds are met (e.g., advance from balance to locomotion when average episode length exceeds 900 steps).~~ **Done (v0.2.0)** — `CurriculumManager` and `CurriculumCallback` with per-stage thresholds in TOML `[curriculum]` sections.
- **Add intermediate difficulty levels** within stages. For example, Stage 2 could progressively increase target speed or distance rather than jumping directly to the full locomotion challenge.
- **Implement domain randomization** within each stage: randomize body mass (within 10%), joint friction, ground friction, and initial pose perturbation magnitude. This produces more robust policies.
- ~~**Track and version curriculum configs** alongside model checkpoints so experiments are fully reproducible.~~ **Done (v0.2.0)** — TOML configs versioned in `configs/` and snapshot saved per W&B run.

### 1.4 Training Infrastructure

**Recommendations:**
- ~~**Add Weights & Biases (wandb) integration** alongside TensorBoard. W&B provides experiment comparison, hyperparameter sweeps, and artifact versioning that TensorBoard lacks.~~ **Done (v0.2.0)** — `WandbCallback` with per-component reward logging, config snapshots, and video recording.
- ~~**Implement proper experiment tracking** with run IDs, git commit hashes, and full hyperparameter snapshots saved with each checkpoint.~~ **Done (v0.2.0)** — W&B integration saves git commit hash and full config per run.
- ~~**Add evaluation metrics beyond average reward**: gait symmetry, energy efficiency (cost of transport), stride frequency, forward velocity consistency, and time-to-target.~~ **Done (v0.2.0)** — `LocomotionMetrics` class in `environments/shared/metrics.py`.
- **Create a benchmark suite** that evaluates trained policies on standardized scenarios (flat ground, slopes, perturbation recovery) and outputs a structured report.

---

## 2. Robotics & Simulation Improvements

### 2.1 MuJoCo Model Quality

The MJCF models are functional but could be more physically realistic.

**Recommendations:**
- **Add tendon and muscle models** instead of relying solely on position/motor actuators. MuJoCo supports tendons with routing, which can produce more biomechanically realistic locomotion and transfer better to real hardware.
- **Implement contact property tuning**: the current friction and contact parameters use defaults. Tuning condim, conaffinity, solref, and solimp per-geom would improve foot-ground interaction realism.
- **Add actuator dynamics** (time constants, force limits) to model motor response delays. Real servos have bandwidth limits that the policy should learn to respect.
- **Consider composite objects** for more detailed collision geometry on the bodies, rather than capsule/box approximations.

### 2.2 Terrain and Environment Diversity

All training currently occurs on a flat plane, which produces brittle policies.

**Recommendations:**
- **Implement procedural terrain generation** using MuJoCo's heightfield support. Start with simple sinusoidal surfaces, then progress to rough terrain, steps, and slopes.
- **Add obstacle courses** with randomized placement: rocks, logs, gaps. These force the policy to generalize.
- **Implement wind/perturbation forces** applied randomly to the torso during training. This produces policies that can recover from unexpected disturbances.
- **Add varying ground friction zones** (mud, ice, rock) to teach adaptive gait strategies.

### 2.3 Sim-to-Real Preparation

The README lists sim-to-real transfer as a roadmap item. Groundwork can begin now.

**Recommendations:**
- **Implement system identification tools** that compare simulated sensor readings to real hardware data and auto-tune MJCF parameters.
- **Add sensor noise models** to observations during training: Gaussian noise on joint encoders, accelerometer bias drift, touch sensor thresholds. Policies trained with noisy observations transfer much better.
- **Implement action delay/latency simulation** (1-3 control steps of delay) in the environment. Real communication buses introduce latency that can destabilize policies not trained for it.
- **Add an actuator calibration script** that maps between simulation actuator units and real servo commands (PWM, position, torque).
- **Create a hardware abstraction layer** in code that presents the same interface whether running in simulation or on physical hardware.

---

## 3. Multi-Agent & Behavioral Complexity

### 3.1 Pack Hunting (Roadmap Item)

This is one of the most compelling planned features and would differentiate the project significantly.

**Recommendations:**
- **Start with predator-prey with a scripted prey** before adding multi-agent complexity. The current environments have static mocap targets; replacing these with a simple evading prey (rule-based movement) would be an impactful intermediate step.
- **Implement multi-agent environments** using PettingZoo or a custom multi-agent wrapper around the existing Gymnasium interface. Start with 2 velociraptors coordinating to corner prey.
- **Use centralized training with decentralized execution (CTDE)**: train with a shared critic that sees all agents' observations, but deploy policies that only use local observations.
- **Add communication channels** between agents (learned latent messages) for emergent coordination behavior.

### 3.2 Behavioral Repertoire

Currently each species learns a single behavior chain (stand, walk, attack/feed).

**Recommendations:**
- **Implement a behavior switching mechanism** where the policy selects between locomotion primitives (walk, trot, gallop, sprint, turn, stop) based on context.
- **Add turning and lateral movement** to the action/reward structure. Currently the environments only reward forward velocity, producing policies that can't steer.
- **Implement a hierarchical RL architecture** with a high-level controller selecting goals (approach, circle, strike) and low-level controllers executing locomotion primitives.

---

## 4. Code Quality & Developer Experience

### 4.1 Tooling Gaps

**Recommendations:**
- ~~**Add a code formatter** (Black or Ruff format) and import sorter (isort) with a pre-commit hook. Currently only flake8 linting exists, and it ignores several rules.~~ **Done (v0.2.0)** — Ruff format + lint configured in `pyproject.toml` with pre-commit hooks.
- ~~**Add mypy or pyright** for static type checking in CI. The type hints are already excellent; enforcing them catches bugs early.~~ **Done (v0.2.0)** — mypy configured in `pyproject.toml` with pre-commit hook; all type errors resolved.
- ~~**Add code coverage reporting** (pytest-cov) to CI with a coverage badge on the README. Current tests cover the happy path well but don't measure branch coverage.~~ **Done (v0.2.0)** — `pytest-cov` with 70% threshold.
- ~~**Replace print statements with Python's `logging` module**. This enables configurable log levels, log file output, and integration with monitoring tools.~~ **Done (v0.2.0)** — All `print()` calls replaced with `logging` in training scripts.
- ~~**Add pre-commit hooks** for linting, formatting, and type checking to catch issues before they reach CI.~~ **Done (v0.2.0)** — `.pre-commit-config.yaml` with Ruff and mypy.

### 4.2 Testing Improvements

The test suites are well-structured but limited to basic functionality checks.

**Recommendations:**
- ~~**Add reward function unit tests** that verify specific scenarios produce expected reward values. For example: "raptor at prey position with claw contact should produce strike bonus."~~ **Done (v0.2.0)**
- **Add regression tests for trained policies**: save a reference trajectory from a trained checkpoint and verify that loading the same checkpoint reproduces it within tolerance.
- **Add performance/benchmark tests** that measure simulation step throughput and flag regressions.
- **Add property-based testing** (Hypothesis) for observation/action space invariants.
- ~~**Test curriculum stage transitions** to verify that configs load correctly and rewards change as expected across stages.~~ **Done (v0.2.0)**

### 4.3 Package & Distribution

**Recommendations:**
- ~~**Add metadata to pyproject.toml**: authors, license declaration, repository URL, classifiers. The MIT license is mentioned in the README but not declared in the package metadata.~~ **Done (v0.2.0)**
- ~~**Register Gymnasium environments** using entry points so users can create environments with `gym.make("MesozoicLabs/Velociraptor-v0")` instead of importing directly.~~ **Done (v0.2.0)** — Auto-registration on `import environments`.
- ~~**Adopt semantic versioning** with a CHANGELOG.md. The project is at 0.1.0; define what 0.2.0 and 1.0.0 mean in terms of API stability.~~ **Done (v0.2.0)** — `CHANGELOG.md` and versioning plan in `ROADMAP.md`.
- **Publish to PyPI** once the API stabilizes. The pyproject.toml is already structured for it.
- **Add Dependabot or Renovate** for automated dependency updates.

---

## 5. Documentation Improvements

### 5.1 Technical Documentation

**Recommendations:**
- **Add a reward engineering guide** explaining the rationale behind each reward component, the tuning process, and common failure modes. This is the knowledge most likely to be lost and most valuable to contributors.
- **Document the MJCF model design decisions**: why specific joint types, ranges, and masses were chosen. Include references to paleontological literature on dinosaur biomechanics.
- **Create an architecture diagram** showing the relationship between BaseDinoEnv, species environments, training scripts, and MuJoCo models.
- **Add a troubleshooting guide** covering common MuJoCo installation issues, training divergence, and environment debugging.
- **Document the observation and action spaces** with labeled diagrams showing which indices correspond to which joints/sensors.

### 5.2 Contributor Documentation

**Recommendations:**
- **Create a "Adding a New Species" guide** that walks through the full process: MJCF model creation, environment subclass implementation, test suite, training script, and CI integration. The existing 3 species provide a clear template, but documenting it lowers the barrier to contribution.
- ~~**Add a CONTRIBUTING.md** with code style guidelines, PR process, and testing requirements.~~ **Done (v0.2.0)**
- ~~**Create issue templates** for bug reports, feature requests, and new species proposals.~~ **Done (v0.2.0)**

---

## 6. Ambitious / Long-Term Goals

### 6.1 JAX/MJX Migration

This is already on the roadmap and would be transformative. MJX enables batch simulation on GPU, providing 100-1000x speedup over CPU MuJoCo.

**Recommendations:**
- **Start with a single species** (Velociraptor, as the simplest model) and create a parallel MJX environment. Keep the existing MuJoCo environments as the reference implementation.
- **Use Brax's training infrastructure** (PPO implementation optimized for JAX) rather than adapting SB3.
- **Benchmark throughput** (environment steps/second) on CPU vs. GPU vs. TPU and publish results.
- **Design the MJX environment to be API-compatible** with the existing Gymnasium interface so training scripts work with either backend.

### 6.2 Ecosystem Expansion

**Recommendations:**
- **Build a model zoo** of pre-trained checkpoints for each species at each curriculum stage. Host on Hugging Face Hub or as GitHub releases. This lets users start from proven checkpoints rather than training from scratch.
- **Create a leaderboard** for locomotion benchmarks (speed, energy efficiency, robustness to perturbation) across species and algorithms.
- **Develop a web-based visualizer** using Three.js or MuJoCo's WASM build to let people interact with trained policies in the browser.
- **Partner with paleontology researchers** to validate locomotion patterns against fossil evidence and published biomechanics studies.

### 6.3 Real Robot Hardware

**Recommendations:**
- **Start with a small, inexpensive platform** like the Compsognathus (planned small biped). Servo-driven, 3D-printed bipeds are achievable at low cost.
- **Use ESP32 or Raspberry Pi** as the onboard controller running a distilled/quantized version of the trained policy.
- **Implement a ROS 2 bridge** between the Gymnasium environment interface and ROS 2 topics, enabling integration with the broader robotics ecosystem.
- **Design the hardware with the sim-to-real gap in mind**: match actuator specifications, sensor placements, and mass distribution between the MJCF model and the physical build.

---

## Priority Ranking

| Priority | Recommendation | Impact | Effort | Status |
|----------|---------------|--------|--------|--------|
| 1 | Externalize reward configs to TOML | High | Low | **Done (v0.2.0)** |
| 2 | Register Gymnasium entry points | High | Low | **Done (v0.2.0)** |
| 3 | Add mypy + Ruff + pre-commit | Medium | Low | **Done (v0.2.0)** |
| 4 | Automated curriculum transitions | High | Medium | **Done (v0.2.0)** |
| 5 | Scripted prey (moving target) | High | Medium | Not started |
| 6 | Sensor noise + action delay | High | Medium | Not started |
| 7 | Domain randomization | High | Medium | Not started |
| 8 | Terrain heightfields | High | Medium | Not started |
| 9 | W&B experiment tracking | Medium | Low | **Done (v0.2.0)** |
| 10 | Custom policy networks | High | Medium | Not started |
| 11 | MJX migration (Velociraptor) | Very High | High | Not started |
| 12 | Multi-agent pack hunting | Very High | High | Not started |
| 13 | Pre-trained model zoo | Medium | Medium | Not started |
| 14 | Physical robot prototype | Very High | Very High | Not started |
