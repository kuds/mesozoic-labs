# Mesozoic Labs

Dinosaur-inspired locomotion research using reinforcement learning and MuJoCo physics simulation.

![Trained PPO Agent](results/velociraptor/ppo/stage1_balance.gif)

*Historical, unverified PPO / Stable-Baselines3 artifact (backend version not
recorded); not evidence for the current model or configuration.*

## Overview

Mesozoic Labs is a simulation research project exploring bipedal and quadrupedal locomotion with Mesozoic-inspired articulated models. We use MuJoCo for physics-based simulation and train agents with algorithms such as PPO and SAC. The models are research abstractions rather than validated anatomical reconstructions. Most species here are dinosaurs; Dibothrosuchus is a crocodylomorph, included because its erect, long-limbed terrestrial posture is a distinct locomotion problem from the sprawl a modern crocodilian uses.

**Goals:**
- Develop locomotion controllers for dinosaur-inspired simulated species
- Explore predatory behaviors (hunting, striking, pack coordination)
- Study the requirements for eventual policy transfer to robotic platforms
- Experiment with JAX/MJX for high-performance training

## Repository Structure

```
mesozoic-labs/
├── environments/              # Dinosaur training environments
│   ├── velociraptor/          # Velociraptor (bipedal predator with sickle claws)
│   │   ├── assets/            # MJCF model files
│   │   ├── envs/              # Gymnasium environments
│   │   ├── scripts/           # Training & utility scripts
│   │   ├── tests/             # Pytest test suite
│   │   └── README.md
│   ├── brachiosaurus/         # Brachiosaurus (quadrupedal sauropod)
│   │   ├── assets/            # MJCF model files
│   │   ├── envs/              # Gymnasium environments
│   │   ├── scripts/           # Training & utility scripts
│   │   ├── tests/             # Pytest test suite
│   │   └── README.md
│   ├── trex/                  # T-Rex (large bipedal predator)
│   │   ├── assets/            # MJCF model files
│   │   ├── envs/              # Gymnasium environments
│   │   ├── scripts/           # Training & utility scripts
│   │   ├── tests/             # Pytest test suite
│   │   └── README.md
│   ├── dibothrosuchus/        # Dibothrosuchus (erect-limbed crocodylomorph)
│   │   ├── assets/            # MJCF model files
│   │   ├── envs/              # Gymnasium environments
│   │   ├── scripts/           # Training & utility scripts
│   │   ├── tests/             # Pytest test suite
│   │   └── README.md
│   └── shared/                # Shared base classes and utilities
│       ├── base_env.py        # BaseDinoEnv abstract class
│       ├── config.py          # TOML configuration loading
│       ├── cli.py             # train / curriculum command-line interface
│       ├── stage_manifest.py  # Stage manifest v2: ids, warm-start edges, deliverables, recipes
│       ├── ancestors.py       # Certified-ancestor reuse across runs (--trunk-from / TRUNK_FROM)
│       ├── curriculum/        # Curriculum manager and SB3 callbacks
│       ├── plant_contract/    # Layered MuJoCo plant safety contract
│       ├── reporting/         # Result summaries, CSVs, and stage artifacts
│       ├── result_bundle/     # Provenance, hashing, and bundle validation
│       ├── train_base.py      # Shared SB3 training infrastructure
│       ├── species_registry.py # Species configuration registry
│       ├── metrics.py         # Locomotion evaluation metrics
│       ├── wandb_integration.py # W&B experiment tracking
│       ├── mjx_env.py         # JAX/MJX batched environment
│       ├── jax_ppo.py         # JAX-native PPO implementation
│       ├── jax_training.py    # JAX training loop
│       ├── harnesses/         # Hand-run smoke checks and MJCF viewers
│       └── tests/             # Shared utility tests
├── configs/                   # Per-species stage manifest (stages.toml) and TOML stage configs
├── notebooks/                 # Jupyter training, sweep, and reporting workflows
├── website/                   # Documentation site (Docusaurus)
└── results/                   # Curated historical summaries and available run artifacts
```

## Environments

<!-- BEGIN GENERATED: SPECIES -->
The active-species tables are generated from `configs/species_manifest.toml`, the layered plant contract,
the executable Gymnasium environments, compiled MJCF models, and current stage TOML files. The budgets and
gates shown are for the Stable-Baselines3 curriculum path. Do not edit the generated block by hand.

<a id="velociraptor"></a>
<a id="raptor"></a>

### Velociraptor Mongoliensis

Swift Bipedal Predator. **Specialty:** Sickle-claw contact attacks.

| Generated specification | Value |
|---|---|
| Observation dimension | 70 |
| Action dimension / actuators | 22 |
| Generalized coordinates / velocities | nq=31, nv=30 |
| Compiled dynamic model mass | 13.5 kg |
| Plant contract revisions | policy r10; physics r2; visual r3 ([details](docs/PLANT_CONTRACT.md)) |
| Model | `environments/velociraptor/assets/raptor.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Balance | stand (deliverable) | — | Learn to stand and balance without falling | 6M | reward ≥ 1050; episode length ≥ 950; ≥ 10 episodes/evaluation; 3 consecutive passes |
| 2 — Locomotion | walk (deliverable) | 1 — Balance | Learn forward walking/running | 8M | reward ≥ 100; episode length ≥ 750; avg. velocity ≥ 2 m/s; ≥ 10 episodes/evaluation; 3 consecutive passes |
| 3 — Strike | hunt (deliverable) | 2 — Locomotion | Sprint and strike prey with sickle claw | 12M | reward ≥ 100; task success ≥ 50.0%; ≥ 10 episodes/evaluation; 3 consecutive passes |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Sickle-claw contact success:** A left or right sickle-claw geom contacts the prey geom while the strike reward is enabled.
- **JAX/MJX — Sickle-claw proximity success:** Either claw-tip site comes within 0.20 m of the prey target position while the strike bonus is enabled; physical geom contact is not required.

**Per-deliverable success semantics:**
- **stand (1 — Balance) · Stable-Baselines3 — Reward-gated stance (reward_and_length/v1):** The stance checkpoint clears the reward_and_length/v1 gate: mean evaluation reward at or above the statue-derived collapse rail and a near-full-horizon mean episode length over the required consecutive evaluations. The zero-action statue clears this gate, so stand is labelled by its gate kind here rather than claimed as certified stance quality; stance_quality/v1 waits on a foot-sensor repair (the single toe site reads about 55% of true load) (plan §4.8).
- **walk (2 — Locomotion) · Stable-Baselines3 — Gated forward velocity (reward_and_length/v1):** The locomotion checkpoint clears the reward_and_length/v1 gate: mean forward velocity at or above the stage's configured minimum, with its reward and episode-length floors, over the required consecutive evaluations.
- **hunt (3 — Strike) · Stable-Baselines3 — Sickle-claw contact success:** A left or right sickle-claw geom contacts the prey geom while the strike reward is enabled.
- **hunt (3 — Strike) · JAX/MJX — Sickle-claw proximity success:** Either claw-tip site comes within 0.20 m of the prey target position while the strike bonus is enabled; physical geom contact is not required.

[Full documentation →](environments/velociraptor/README.md)

[Hugging Face models →](https://huggingface.co/kuds/mesozoic-labs-velocipastor)

<a id="trex"></a>
<a id="t-rex"></a>

### Tyrannosaurus Rex

Apex Predator. **Specialty:** Head-contact attack task.

| Generated specification | Value |
|---|---|
| Observation dimension | 64 |
| Action dimension / actuators | 15 |
| Generalized coordinates / velocities | nq=28, nv=27 |
| Compiled dynamic model mass | 85.7 kg |
| Plant contract revisions | policy r13; physics r7; visual r4 ([details](docs/PLANT_CONTRACT.md)) |
| Model | `environments/trex/assets/trex.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Balance | stand (deliverable) | — | Learn to stand and balance without falling | 11M | reward ≥ 2100; full-horizon episodes ≥ 95.0%; unsupported duty ≤ 0.02; unsupported duty 95% upper bound ≤ 0.02; ≥ 40 episodes/evaluation; 3 consecutive passes |
| recovery — Recovery | stand (deliverable) | 1 — Balance | Hold the stance against scheduled external pushes and recover from each | 3M | recovery success LCB95 ≥ 0.3; paired Δ vs each required frozen null LCB95 ≥ 0.2; re-entry ≤ 100 steps + 50-step dwell; ≥ 40 episodes/evaluation; verdict from the frozen gate_resolution.json (post-stage; fail-closed when absent or stale) |
| 2 — Locomotion | walk (deliverable) | 1 — Balance | Learn forward walking/running | 8M | reward ≥ 100; episode length ≥ 750; avg. velocity ≥ 1 m/s; ≥ 10 episodes/evaluation; 3 consecutive passes |
| 3 — Bite | hunt (deliverable) | 2 — Locomotion | Sprint to prey and make contact with the head bite proxy | 8M | task success LCB95 ≥ 0.5; reward rail ≥ 361; ≥ 30 episodes/evaluation; verdict from the selected checkpoint's evaluation_selected.csv (post-stage; fail-closed when absent) |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Head-contact bite proxy:** The head-bite geom contacts the prey geom while the bite reward is enabled; the model has no articulated jaw.
- **JAX/MJX — Head-tip proximity bite proxy:** The head-tip site comes within 0.35 m of the prey target position while the bite bonus is enabled; physical geom contact is not required and the model has no articulated jaw.

**Per-deliverable success semantics:**
- **stance (1 — Balance) · Stable-Baselines3 — Stance quality (stance_quality/v1):** The stance checkpoint clears the stance_quality/v1 gate: the unsupported-duty 95% upper bound and the full-horizon episode fraction meet the configured bounds over 40-episode evaluations, with the reward rail as a collapse floor only. The duty and full-horizon statistics are not exported to summary.json until a later phase (decision D-A9, deferred by D-B15), so the catalog names them with no value.
- **recovery (recovery — Recovery) · Stable-Baselines3 — Recovery under pushes (recovery_quality/v1):** The recovery checkpoint clears the recovery_quality/v1 gate, judged post-stage against the run's frozen gate_resolution.json: recovery-success LCB95 and the paired delta against each required frozen null meet the frozen thresholds, with re-entry inside the configured step budget and dwell. The statistics are not exported to summary.json until a later phase (decision D-A9, deferred by D-B15).
- **walk (2 — Locomotion) · Stable-Baselines3 — Gated forward velocity (reward_and_length/v1):** The locomotion checkpoint clears the reward_and_length/v1 gate: mean forward velocity at or above the stage's configured minimum, with its reward and episode-length floors, over the required consecutive evaluations.
- **hunt (3 — Bite) · Stable-Baselines3 — Head-contact bite success (task_success/v1):** The hunt checkpoint clears the task_success/v1 gate: the one-sided 95% Clopper-Pearson lower bound on per-episode head-contact bite success (the head-bite geom contacts the prey geom while the bite reward is enabled; the model has no articulated jaw) over the selected checkpoint's evaluation_selected.csv at the stage's declared min_eval_episodes, hash-bound to that checkpoint, meets the configured min_success_lcb bar (provisional until the first Phase-B pilot re-freezes it; decision D-B2), with the reward rail as a collapse floor only. The bound is exported to summary.json as selected_model_success_lcb beside the raw selected-checkpoint rate.
- **hunt (3 — Bite) · JAX/MJX — Head-tip proximity bite proxy:** The head-tip site comes within 0.35 m of the prey target position while the bite bonus is enabled; physical geom contact is not required and the model has no articulated jaw. The JAX/MJX path cannot judge task_success/v1 (no MJX hunting panel exists) and refuses the stage rather than certifying it.

[Full documentation →](environments/trex/README.md)

<a id="brachiosaurus"></a>
<a id="brachio"></a>

### Brachiosaurus Altithorax

Gentle Giant Herbivore. **Specialty:** Head-to-food reaching.

| Generated specification | Value |
|---|---|
| Observation dimension | 86 |
| Action dimension / actuators | 30 |
| Generalized coordinates / velocities | nq=38, nv=37 |
| Compiled dynamic model mass | 175.3 kg |
| Plant contract revisions | policy r8; physics r4; visual r2 ([details](docs/PLANT_CONTRACT.md)) |
| Model | `environments/brachiosaurus/assets/brachiosaurus.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Balance | stand (deliverable) | — | Learn to stand on four legs without falling | 6M | reward ≥ 1040; episode length ≥ 950; ≥ 10 episodes/evaluation; 3 consecutive passes |
| 2 — Locomotion | walk (deliverable) | 1 — Balance | Learn coordinated quadrupedal walking | 16M | reward ≥ 100; episode length ≥ 750; avg. velocity ≥ 0.75 m/s; ≥ 10 episodes/evaluation; 3 consecutive passes |
| 3 — Food Reach | hunt (deliverable) | 2 — Locomotion | Move the head tip within the configured distance threshold of food | 12M | reward ≥ 100; task success ≥ 50.0%; ≥ 10 episodes/evaluation; 3 consecutive passes |

**Backend-specific success semantics:**
- **Stable-Baselines3 / JAX/MJX — Head-tip distance-threshold success:** The head-tip site comes within the configured food-reach threshold of the food target while the food-reach bonus is enabled.

**Per-deliverable success semantics:**
- **stand (1 — Balance) · Stable-Baselines3 — Reward-gated stance (reward_and_length/v1):** The stance checkpoint clears the reward_and_length/v1 gate: mean evaluation reward at or above the statue-derived collapse rail and a near-full-horizon mean episode length over the required consecutive evaluations. The zero-action statue clears this gate, so stand is labelled by its gate kind here rather than claimed as certified stance quality; stance_quality/v1 waits on shin instrumentation (a kneeling pose currently reads identically to airborne) (plan §4.8).
- **walk (2 — Locomotion) · Stable-Baselines3 — Gated forward velocity (reward_and_length/v1):** The locomotion checkpoint clears the reward_and_length/v1 gate: mean forward velocity at or above the stage's configured minimum, with its reward and episode-length floors, over the required consecutive evaluations.
- **hunt (3 — Food Reach) · Stable-Baselines3 / JAX/MJX — Head-tip distance-threshold success:** The head-tip site comes within the configured food-reach threshold of the food target while the food-reach bonus is enabled.

[Full documentation →](environments/brachiosaurus/README.md)

<a id="dibothrosuchus"></a>
<a id="dibo"></a>

### Dibothrosuchus Elaphros

Gracile Erect-Limbed Crocodylomorph. **Specialty:** Snout-contact snap task.

| Generated specification | Value |
|---|---|
| Observation dimension | 80 |
| Action dimension / actuators | 27 |
| Generalized coordinates / velocities | nq=35, nv=34 |
| Compiled dynamic model mass | 8.7 kg |
| Plant contract revisions | policy r7; physics r1; visual r1 ([details](docs/PLANT_CONTRACT.md)) |
| Model | `environments/dibothrosuchus/assets/dibothrosuchus.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Balance | stand (deliverable) | — | Hold the erect quadrupedal stance without collapsing into a sprawl | 6M | reward ≥ 1560; episode length ≥ 950; ≥ 10 episodes/evaluation; 3 consecutive passes |
| 2 — Locomotion | walk (deliverable) | 1 — Balance | Learn a coordinated erect-limbed diagonal-pair walk | 12M | reward ≥ 100; episode length ≥ 750; avg. velocity ≥ 0.9 m/s; ≥ 10 episodes/evaluation; 3 consecutive passes |
| 3 — Snap | hunt (deliverable) | 2 — Locomotion | Close on small prey and touch it with the snout snap proxy | 8M | reward ≥ 100; task success ≥ 50.0%; ≥ 10 episodes/evaluation; 3 consecutive passes |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Snout-contact snap proxy:** The snout snap geom contacts the prey geom while the snap reward is enabled; the model has no articulated jaw.
- **JAX/MJX — Snout-tip proximity snap proxy:** The snout-tip site comes within 0.12 m of the prey target position while the snap bonus is enabled; physical geom contact is not required and the model has no articulated jaw.

**Per-deliverable success semantics:**
- **stand (1 — Balance) · Stable-Baselines3 — Reward-gated stance (reward_and_length/v1):** The stance checkpoint clears the reward_and_length/v1 gate: mean evaluation reward at or above the statue-derived collapse rail and a near-full-horizon mean episode length over the required consecutive evaluations. The zero-action statue clears this gate, so stand is labelled by its gate kind here rather than claimed as certified stance quality; stance_quality/v1 waits on a stance-quality and perturbation preflight this plant has not had (plan §4.8).
- **walk (2 — Locomotion) · Stable-Baselines3 — Gated forward velocity (reward_and_length/v1):** The locomotion checkpoint clears the reward_and_length/v1 gate: mean forward velocity at or above the stage's configured minimum, with its reward and episode-length floors, over the required consecutive evaluations.
- **hunt (3 — Snap) · Stable-Baselines3 — Snout-contact snap proxy:** The snout snap geom contacts the prey geom while the snap reward is enabled; the model has no articulated jaw.
- **hunt (3 — Snap) · JAX/MJX — Snout-tip proximity snap proxy:** The snout-tip site comes within 0.12 m of the prey target position while the snap bonus is enabled; physical geom contact is not required and the model has no articulated jaw.

[Full documentation →](environments/dibothrosuchus/README.md)

<a id="compsognathus"></a>
<a id="compso"></a>

### Compsognathus Longipes

Small Bipedal Theropod. **Specialty:** Non-contact target reaching.

| Generated specification | Value |
|---|---|
| Observation dimension | 56 |
| Action dimension / actuators | 14 |
| Generalized coordinates / velocities | nq=24, nv=23 |
| Compiled dynamic model mass | 1.0 kg |
| Plant contract revisions | policy r2; physics r1; visual r1 ([details](docs/PLANT_CONTRACT.md)) |
| Model | `environments/compsognathus/assets/compsognathus.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Balance | stand (deliverable) | — | Hold an upright stance with foot support | 11M | reward ≥ 1800; full-horizon episodes ≥ 95.0%; unsupported duty ≤ 0.02; unsupported duty 95% upper bound ≤ 0.02; ≥ 40 episodes/evaluation; 3 consecutive passes |
| recovery — Recovery | stand (deliverable) | 1 — Balance | Pilot: recover an upright stance after calibrated horizontal pushes | 3M | recovery success LCB95 ≥ 0.5; paired Δ vs each required frozen null LCB95 ≥ 0.1; re-entry ≤ 40 steps + 20-step dwell; ≥ 40 episodes/evaluation; verdict from the frozen gate_resolution.json (post-stage; fail-closed when absent or stale) |
| 2 — Locomotion | walk (deliverable) | 1 — Balance | Move forward while remaining upright and avoiding body-floor contact | 3M | reward ≥ 500; episode length ≥ 900; avg. velocity ≥ 0.08 m/s; ≥ 20 episodes/evaluation; 3 consecutive passes |
| 3 — Target Reach | hunt (deliverable) | 2 — Locomotion | Reach the randomized horizontal target and slow down while upright | 3M | reward ≥ 25; task success ≥ 70.0%; ≥ 20 episodes/evaluation; 3 consecutive passes |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Pelvis target-reaching success:** While the target task is enabled, the upright pelvis enters the configured horizontal target radius at or below the configured speed; non-foot floor contact is forbidden.

**Per-deliverable success semantics:**
- **stance (1 — Balance) · Stable-Baselines3 — Stance quality (stance_quality/v1):** The stance checkpoint clears the stance_quality/v1 gate: the unsupported-duty 95% upper bound and the full-horizon episode fraction meet the configured bounds over 40-episode evaluations, with the reward rail as a collapse floor only. The duty and full-horizon statistics are not exported to summary.json until a later phase (decision D-A9, deferred by D-B15), so the catalog names them with no value.
- **recovery (recovery — Recovery) · Stable-Baselines3 — Recovery under pushes (recovery_quality/v1):** The recovery checkpoint clears the recovery_quality/v1 gate, judged post-stage against the run's frozen gate_resolution.json: recovery-success LCB95 and the paired delta against each required frozen null meet the frozen thresholds, with re-entry inside the configured step budget and dwell. The statistics are not exported to summary.json until a later phase (decision D-A9, deferred by D-B15).
- **walk (2 — Locomotion) · Stable-Baselines3 — Gated forward velocity (reward_and_length/v1):** The locomotion checkpoint clears the reward_and_length/v1 gate: mean forward velocity at or above the stage's configured minimum, with its reward and episode-length floors, over the required consecutive evaluations.
- **hunt (3 — Target Reach) · Stable-Baselines3 — Pelvis target-reaching success:** While the target task is enabled, the upright pelvis enters the configured horizontal target radius at or below the configured speed; non-foot floor contact is forbidden.

[Full documentation →](environments/compsognathus/README.md)

<a id="compsognathus_robot"></a>
<a id="compso-robot"></a>

### Compsognathus Longipes (Robot)

Twelve-Servo Biped Prototype. **Specialty:** Non-contact target reaching with fixed head and tail.

| Generated specification | Value |
|---|---|
| Observation dimension | 46 |
| Action dimension / actuators | 12 |
| Generalized coordinates / velocities | nq=19, nv=18 |
| Compiled dynamic model mass | 1.6 kg |
| Plant contract revisions | policy r2; physics r1; visual r1 ([details](docs/PLANT_CONTRACT.md)) |
| Model | `environments/compsognathus/assets/compsognathus_robot.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Balance | stand (deliverable) | — | Hold an upright stance with foot support | 11M | reward ≥ 1800; full-horizon episodes ≥ 95.0%; unsupported duty ≤ 0.02; unsupported duty 95% upper bound ≤ 0.02; ≥ 40 episodes/evaluation; 3 consecutive passes |
| recovery — Recovery | stand (deliverable) | 1 — Balance | Pilot: recover an upright stance after calibrated horizontal pushes | 3M | recovery success LCB95 ≥ 0.5; paired Δ vs each required frozen null LCB95 ≥ 0.1; re-entry ≤ 40 steps + 20-step dwell; ≥ 40 episodes/evaluation; verdict from the frozen gate_resolution.json (post-stage; fail-closed when absent or stale) |
| 2 — Locomotion | walk (deliverable) | 1 — Balance | Move forward while remaining upright and avoiding body-floor contact | 3M | reward ≥ 500; episode length ≥ 900; avg. velocity ≥ 0.04 m/s; ≥ 20 episodes/evaluation; 3 consecutive passes |
| 3 — Target Reach | hunt (deliverable) | 2 — Locomotion | Reach the randomized horizontal target and slow down while upright | 3M | reward ≥ 25; task success ≥ 70.0%; ≥ 20 episodes/evaluation; 3 consecutive passes |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Pelvis target-reaching success:** While the target task is enabled, the upright pelvis enters the configured horizontal target radius at or below the configured speed; non-foot floor contact is forbidden.

**Per-deliverable success semantics:**
- **stance (1 — Balance) · Stable-Baselines3 — Stance quality (stance_quality/v1):** The stance checkpoint clears the stance_quality/v1 gate: the unsupported-duty 95% upper bound and the full-horizon episode fraction meet the configured bounds over 40-episode evaluations, with the reward rail as a collapse floor only. The duty and full-horizon statistics are not exported to summary.json until a later phase (decision D-A9, deferred by D-B15), so the catalog names them with no value.
- **recovery (recovery — Recovery) · Stable-Baselines3 — Recovery under pushes (recovery_quality/v1):** The recovery checkpoint clears the recovery_quality/v1 gate, judged post-stage against the run's frozen gate_resolution.json: recovery-success LCB95 and the paired delta against each required frozen null meet the frozen thresholds, with re-entry inside the configured step budget and dwell. The statistics are not exported to summary.json until a later phase (decision D-A9, deferred by D-B15).
- **walk (2 — Locomotion) · Stable-Baselines3 — Gated forward velocity (reward_and_length/v1):** The locomotion checkpoint clears the reward_and_length/v1 gate: mean forward velocity at or above the stage's configured minimum, with its reward and episode-length floors, over the required consecutive evaluations.
- **hunt (3 — Target Reach) · Stable-Baselines3 — Pelvis target-reaching success:** While the target task is enabled, the upright pelvis enters the configured horizontal target radius at or below the configured speed; non-foot floor contact is forbidden.

[Full documentation →](environments/compsognathus/README.md)
<!-- END GENERATED: SPECIES -->

### Planned Species
- Deinonychus (pack hunter)
- Compsognathus (small, fast biped)
- Stegosaurus (armored quadrupedal defender)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/kuds/mesozoic-labs.git
cd mesozoic-labs

python -m venv venv
source venv/bin/activate

# Install the package with training dependencies
pip install -e ".[train]"

# View the velociraptor model
python environments/velociraptor/scripts/view_model.py

# Full curriculum — one command walks the manifest's advancing stages in order
# (each stage loads its own hyperparameters from the TOML config)
cd environments/velociraptor
python scripts/train_sb3.py curriculum --algorithm ppo

# Reuse the certified trunk of an earlier run and train only what is missing above it
python scripts/train_sb3.py curriculum --algorithm ppo --trunk-from logs/<earlier_run> --output-dir logs/<new_run>

# Retrain locomotion (and everything after it) on top of the trunk's certified stance
python scripts/train_sb3.py curriculum --algorithm ppo --trunk-from logs/<earlier_run> --retrain-from locomotion --output-dir logs/<new_run> --label lr-sweep-a

# A walk-only run: certify locomotion on the trunk's certified stance and stop there
python scripts/train_sb3.py curriculum --algorithm ppo --target walk --trunk-from logs/<earlier_run> --output-dir logs/<walk_run>
```

Stages are declared by a per-species **stage manifest**
(`configs/<species>/stages.toml`, schema v2 for every species). Each node
names the node it warm-starts from, whether its certified checkpoint is a
published **deliverable**, and the behavior **recipe** it belongs to —
`stand`, `walk`, `hunt`. A recipe is read off those edges: a deliverable plus
the chain of nodes beneath it, so the stages form a small DAG on one shared
trunk rather than a ladder whose last rung is the only product. Integer stage
references always mean their historical stage, so existing artifacts and
commands keep their meaning; stages without a numeric history are addressed
by semantic id.

The `curriculum` command trains the advancing stages of its target's chain
in manifest order — the whole ladder by default — warm-starting each node
from its declared parent's handoff checkpoint and
VecNormalize sidecar. Every trained node is judged and writes a
`gate_verdict.json` beside its handoff; a node whose parent has no certified
checkpoint stops the run with a warning, never trains from scratch. On the
command line that is all a run records — the verdicts and
`curriculum_results.csv`, which is what lets it serve as a later run's trunk;
the per-deliverable result bundle is the notebook's. A notebook run publishes
every deliverable it certified: a failed hunt still publishes the certified
walk and stance it trained beneath it (bundle status `partial`), and the
bundle is `complete` only when the target and every present deliverable are
certified. Ancestors reused from a trunk are not republished — they appear
under `provenance.ancestors` and stay published by the run that certified
them. Across runs:

- `--target BEHAVIOR` names the behavior the run certifies — a recipe label
  (`walk`) or a deliverable's stage id (`locomotion`), resolved as the
  notebook's `BEHAVIOR` knob resolves them, or a legacy number (`2`),
  resolved as `--stage` resolves it — and walks that target's chain and
  stops there, so `--target walk` certifies a walk-only run. The default is
  the last advancing stage. The chain must be the advancing ladder up to the
  target: `--target stand` on the T-Rex, whose `stand` chain runs through the
  non-advancing recovery node, is refused and points at the notebook, as is
  a chain that skips a ladder stage.
- `--trunk-from RUN_DIR` reuses an earlier run's certified ancestors,
  root-first, instead of retraining them (a passed verdict hash-bound to the
  handoff pair, the same task digest, plant and gate configuration — the
  verdict's `gate_sha256` — and each child trained from the very parent
  checkpoint reused before it). The run's target — `--target`,
  the last advancing stage by default — is always trained here; a reused node's verdict, config,
  fingerprint and plant records are copied under `ancestors/<stage_id>/` —
  never the checkpoint pair — and its children record `parent_run_id`. A run
  that itself reused a node resolves it through its ancestor records to the
  run that certified it, one machine-visible run directory away, so trunks
  compose: the notebook's `TRUNK_FROM` follows them the same way, while a
  run's own directory never does.
- `--retrain-from STAGE_ID` (with `--trunk-from`) trains the named stage of
  the target's chain and everything after it even when the trunk holds a
  certified copy, reusing only the ancestors strictly above it. A variant is a new run: a
  stage directory that already records a node is refused, so pair it with a
  fresh `--output-dir`.
- `--label TEXT` (on `train` and `curriculum`) is recorded in each trained
  stage's run block beside its `hyperparameters_sha256` digest, so variants
  can be told apart; an algorithm-block edit that reuse ignores is warned
  about by key.

The SB3 notebook does the same through its `BEHAVIOR` (default `"hunt"`; a
recipe label or a deliverable's stage id), `TRUNK_FROM`, `RETRAIN_FROM` and
`RUN_LABEL` knobs and one behavior chain-loop cell. The design, its
decisions and the phases still to come are in
[docs/BEHAVIOR_RECIPES_PLAN.md](docs/BEHAVIOR_RECIPES_PLAN.md); the site's
[behavior recipes guide](website/docs/training/recipes.md) walks through a
run.

The T-Rex manifest is the DAG in miniature: stance is the root; **recovery**
(`stand`) and locomotion (`walk`) both warm-start from stance; behavior
(`hunt`) warm-starts from locomotion. Recovery holds the certified stance
against scheduled external pushes and is a published deliverable, but it has
no legacy number, so `curriculum` skips it with a log line. Run it as a single
node, or as the `stand` chain (stance → recovery, verdict enforced) with
`BEHAVIOR = "stand"` in the notebook:

```bash
# Run the T-Rex recovery stage, warm-started from a certified stance checkpoint.
# The load is refused unless the checkpoint records stance, recovery's declared parent.
# Its gate (recovery_quality/v1, frozen 2026-08-28) is judged post-stage against
# the stage directory's gate_resolution.json; see docs/STAGE1B_IMPLEMENTATION_PLAN.md.
cd environments/trex
python scripts/train_sb3.py train --stage recovery --load <stance-checkpoint>.zip --load-mode initialize_next_stage
```

## Docker

The repo ships a `Dockerfile` that bundles MuJoCo, Stable-Baselines3, and all training dependencies:

```bash
# Build
docker build -t mesozoic-labs:latest .

# Quick smoke-test (no GPU needed)
docker run --rm mesozoic-labs:latest \
  environments/velociraptor/scripts/train_sb3.py \
  train --stage 1 --timesteps 1000 --n-envs 1

# Full curriculum with GPU, writing outputs to local disk
docker run --rm --gpus all \
  -v "$(pwd)/outputs:/app/outputs" \
  mesozoic-labs:latest \
  environments/velociraptor/scripts/train_sb3.py \
  curriculum --algorithm ppo --n-envs 4 --output-dir /app/outputs/velociraptor
```

See [Vertex AI training docs](website/docs/training/vertex-ai.md) for cloud deployment.

## Training Results

<!-- BEGIN GENERATED: RESULTS -->
The summaries below are historical experiment records generated from the versioned JSON files under
`results/`. They are not evidence for the current model revision unless provenance is marked both current
and verified. Current stage budgets may therefore differ from the steps reported here.

### Velociraptor Mongoliensis (PPO · Stable-Baselines3) — 2026-03-15

**Provenance:** Historical model; unverified; evaluation episode count not recorded; Stable-Baselines3 (version not recorded). **Run total:** 22M steps; 11:25:15. [Source summary](results/velociraptor/ppo/summary.json).

| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |
|---|---:|---:|---:|---:|---:|
| 1 — Balance | 1964.43 | 0.11 m/s | — | 6M | passed retired gate (reward gate) |
| 2 — Locomotion | 2678.68 | 3.47 m/s | — | 8M | passed retired gate (reward gate) |
| 3 — Strike | 1366.19 | 2.02 m/s | 93.3% | 8M | passed retired gate (reward gate) |

**Current Stable-Baselines3 catalog definition for this task label:** A left or right sickle-claw geom contacts the prey geom while the strike reward is enabled.

### Velociraptor Mongoliensis (SAC · Stable-Baselines3) — 2026-03-21

**Provenance:** Historical model; unverified; evaluation episode count not recorded; Stable-Baselines3 (version not recorded). **Run total:** 22M steps; 22:59:18. [Source summary](results/velociraptor/sac/summary.json).

| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |
|---|---:|---:|---:|---:|---:|
| 1 — Balance | 970.19 | -0.64 m/s | — | 6M | passed retired gate (reward gate) |
| 2 — Locomotion | 2078.62 | 2.91 m/s | — | 8M | passed retired gate (reward gate) |
| 3 — Strike | 1195.43 | 1.63 m/s | 90.0% | 8M | passed retired gate (reward gate) |

**Current Stable-Baselines3 catalog definition for this task label:** A left or right sickle-claw geom contacts the prey geom while the strike reward is enabled.

### Tyrannosaurus Rex (PPO · Stable-Baselines3) — 2026-03-18

**Provenance:** Historical model; unverified; evaluation episode count not recorded; Stable-Baselines3 (version not recorded). **Run total:** 22M steps; 13:02:32. [Source summary](results/trex/ppo/summary.json).

| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |
|---|---:|---:|---:|---:|---:|
| 1 — Balance | 3008.66 | 0.02 m/s | — | 6M | passed retired gate (reward gate) |
| 2 — Locomotion | 1936.01 | 3.47 m/s | — | 8M | passed retired gate (reward gate) |
| 3 — Bite | 1294.28 | 1.68 m/s | 96.7% | 8M | passed retired gate (reward gate) |

**Current Stable-Baselines3 catalog definition for this task label:** The head-bite geom contacts the prey geom while the bite reward is enabled; the model has no articulated jaw.

### Brachiosaurus Altithorax (PPO · Stable-Baselines3) — 2026-07-18

**Provenance:** Historical model; unverified; 30 evaluation episodes; Stable-Baselines3 (version not recorded). **Run total:** 34.0214M steps; 19:46:16. [Source summary](results/brachiosaurus/ppo/summary.json).

| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |
|---|---:|---:|---:|---:|---:|
| 1 — Balance | 1740.53 | 0.01 m/s | — | 6.00474M | passed retired gate (reward gate) |
| 2 — Locomotion | 6634.60 | 1.42 m/s | 3.3% | 16.0072M | passed retired gate (reward gate) |
| 3 — Food Reach | 1368.13 | 0.71 m/s | 100.0% | 12.0095M | passed retired gate (reward gate) |

**Current Stable-Baselines3 catalog definition for this task label:** The head-tip site comes within the configured food-reach threshold of the food target while the food-reach bonus is enabled.
<!-- END GENERATED: RESULTS -->

## Notebooks

<!-- BEGIN GENERATED: NOTEBOOKS -->
| Notebook | Description |
|---|---|
| [`notebooks/sb3_training.ipynb`](notebooks/sb3_training.ipynb) | Train and evaluate species with Stable-Baselines3. |
| [`notebooks/jax_training.ipynb`](notebooks/jax_training.ipynb) | Train and evaluate species with the JAX/MJX backend. |
| [`notebooks/ray_tune_sweep.ipynb`](notebooks/ray_tune_sweep.ipynb) | Run distributed hyperparameter sweeps with Ray Tune. |
| [`notebooks/google_drive_summary.ipynb`](notebooks/google_drive_summary.ipynb) | Collect and summarize training artifacts from Google Drive. |
<!-- END GENERATED: NOTEBOOKS -->

## Roadmap

- [x] Publish historical Velociraptor PPO and SAC run summaries
- [x] Publish a historical T-Rex PPO run summary
- [-] Continue Brachiosaurus Stage 3 training and publish a provenance-complete run
- [-] SAC training for T-Rex (a historical, unverified Velociraptor SAC summary is published)
- [ ] Domain randomization (friction, damping, gravity, actuator strength, external pushes, observation noise)
- [ ] Terrain adaptation (uneven ground, obstacles)
- [-] JAX/MJX migration for faster training (PPO pipeline complete, SAC pending)
- [-] mjlab pilot (MuJoCo-Warp + Isaac-Lab manager API) — scaffold landed, velociraptor Stage 1 spike pending
- [-] Behavior recipes on one certified trunk — stand / walk / hunt published per deliverable with cross-run ancestor reuse (Phase A landed 2026-09-12; the measured hunting gate `task_success/v1` landed 2026-09-13 with a provisional 0.5 bar, decision D-B2, and seed replication as provenance landed the same day — `certification_seeds`, provisional labels; trex stance declares a 2-seed bar and publishes provisional at 1 of 2 until the replicate is re-backfilled); only the command-interface bump and the follow-direction leaf remain pending (see [docs/BEHAVIOR_RECIPES_PLAN.md](docs/BEHAVIOR_RECIPES_PLAN.md))
- [ ] Multi-agent pack hunting scenarios
- [ ] Sim-to-real transfer experiments (future work; no hardware-transfer results are published yet)

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full phased timeline, milestones, and dependency graph.

## Resources

- **Documentation:** [mesozoiclabs.com](https://mesozoiclabs.com)
- **Blog:** [From Zero to Dino-Roar](https://www.findingtheta.com/blog/from-zero-to-dino-roar-teaching-a-t-rex-to-walk-with-mujoco-and-reinforcement-learning)

## Development

```bash
# Install with all dev dependencies
pip install -e ".[all]"

# Run tests
pytest

# Lint and type check
ruff check environments/
mypy environments/

# Regenerate public species data and README tables after changing a model,
# stage config, manifest entry, notebook path, video, or result summary
python -m environments.shared.species_catalog

# Verify committed generated data without rewriting it
python -m environments.shared.species_catalog --check
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use Mesozoic Labs in your research, please cite:

```bibtex
@software{mesozoic_labs,
  title     = {Mesozoic Labs: Dinosaur Locomotion via Reinforcement Learning},
  author    = {Michael Kudlaty},
  year      = {2025},
  url       = {https://github.com/kuds/mesozoic-labs},
  license   = {MIT}
}
```

## License

MIT License
