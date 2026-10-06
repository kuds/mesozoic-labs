# Species catalog

<!-- Written whole by `python -m environments.shared.species_catalog`, which CI runs with `--check`;
do not edit by hand. -->

Each species' full entry and the published run summaries. The [root README](../README.md#species)
has one row per species, and each species' page on [mesozoiclabs.com](https://mesozoiclabs.com)
renders the same entry.

## Species

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
| Plant contract revisions | policy r11; physics r3; visual r4 ([details](PLANT_CONTRACT.md)) |
| Model | `environments/velociraptor/assets/raptor.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Balance | stand (deliverable) | — | Learn to stand and balance without falling | 6M | clean stance episodes LCB95 ≥ 0.8 over ≥ 40 episodes (settle 100 steps); an episode is clean when it reaches the horizon with min_all_feet_support ≥ 0.98, max_touchdown_rate ≤ 0.25, max_window_displacement_m ≤ 0.1, min_foot_load_share ≥ 0.4, max_actuator_saturation_fraction ≤ 0.1, max_settle_airborne_substeps ≤ 0, max_settle_peak_floor_force_bw ≤ 2, min_foot_load_share_windowed ≥ 0.35, max_foot_contact_fraction ≤ 0.02, max_phantom_support_fraction ≤ 0.05, max_nonfoot_load_fraction ≤ 0.01, max_settle_stance_width_change_m ≤ 0.05, min_support_geom_duty ≥ 0.5, min_support_geom_coverage ≥ 0.8, min_foot_load_share_statue_ratio ≥ 0.8; full-horizon episodes ≥ 95.0%; reward rail ≥ 1710; reward rail ≥ 0.6 × the statue's; verdict from the floor-truth stance_gate_report.json on the handoff pair (post-stage; fail-closed when absent) |
| 2 — Locomotion | walk (deliverable) | 1 — Balance | Learn forward walking/running | 8M | reward ≥ 100; episode length ≥ 750; avg. velocity ≥ 2 m/s; ≥ 10 episodes/evaluation; 3 consecutive passes |
| 3 — Strike | hunt (deliverable) | 2 — Locomotion | Sprint and strike prey with sickle claw | 12M | reward ≥ 100; task success ≥ 50.0%; ≥ 10 episodes/evaluation; 3 consecutive passes |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Sickle-claw contact success:** A left or right sickle-claw geom contacts the prey geom while the strike reward is enabled.

**Per-deliverable success semantics:**
- **stand (1 — Balance) · Stable-Baselines3 — Stance quality (stance_quality/v2):** The stance checkpoint clears the stance_quality/v2 gate, judged post-stage on the handoff pair: the one-sided 95% lower bound on the share of the 40 certification-panel episodes that are clean on floor truth (horizon reached, no hop or impact in the settle window, both feet down and loaded, both toes and the metatarsal head of each foot loaded, no actuator held at its limit) meets the configured bound, with the reward rails as collapse floors only. The clean count and its bound are recorded in gate_verdict.json but not exported to summary.json until a later phase (decision D-A9, deferred by D-B15), so the catalog names them with no value.
- **walk (2 — Locomotion) · Stable-Baselines3 — Gated forward velocity (reward_and_length/v1):** The locomotion checkpoint clears the reward_and_length/v1 gate: mean forward velocity at or above the stage's configured minimum, with its reward and episode-length floors, over the required consecutive evaluations.
- **hunt (3 — Strike) · Stable-Baselines3 — Sickle-claw contact success:** A left or right sickle-claw geom contacts the prey geom while the strike reward is enabled.

[Full documentation →](../environments/velociraptor/README.md)

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
| Plant contract revisions | policy r13; physics r8; visual r4 ([details](PLANT_CONTRACT.md)) |
| Model | `environments/trex/assets/trex.xml` |

| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |
|---|---|---|---|---:|---:|
| 1 — Balance | stand (deliverable) | — | Learn to stand and balance without falling | 11M | clean stance episodes LCB95 ≥ 0.8 over ≥ 40 episodes (settle 200 steps); an episode is clean when it reaches the horizon with min_all_feet_support ≥ 0.98, max_touchdown_rate ≤ 0.25, max_window_displacement_m ≤ 0.1, min_foot_load_share ≥ 0.4, max_actuator_saturation_fraction ≤ 0.1, max_settle_airborne_substeps ≤ 0, max_settle_peak_floor_force_bw ≤ 1.5, min_foot_load_share_windowed ≥ 0.35, max_foot_contact_fraction ≤ 0.02, max_phantom_support_fraction ≤ 0.05, max_nonfoot_load_fraction ≤ 0.02, max_sole_tilt_deg ≤ 2, max_sole_corner_lift_m ≤ 0.004, min_sole_contacts ≥ 1.5; full-horizon episodes ≥ 95.0%; reward rail ≥ 2260; reward rail ≥ 0.6 × the statue's; verdict from the floor-truth stance_gate_report.json on the handoff pair (post-stage; fail-closed when absent) |
| recovery — Recovery | stand (deliverable) | 1 — Balance | Hold the stance against scheduled external pushes and recover from each | 3M | recovery success LCB95 ≥ 0.3; paired Δ vs each required frozen null LCB95 ≥ 0.2; re-entry ≤ 100 steps + 50-step dwell; ≥ 40 episodes/evaluation; verdict from the frozen gate_resolution.json (post-stage; fail-closed when absent or stale) |
| 2 — Locomotion | walk (deliverable) | 1 — Balance | Learn forward walking/running | 8M | reward ≥ 100; episode length ≥ 750; avg. velocity ≥ 1 m/s; ≥ 10 episodes/evaluation; 3 consecutive passes |
| 3 — Bite | hunt (deliverable) | 2 — Locomotion | Sprint to prey and make contact with the head bite proxy | 8M | task success LCB95 ≥ 0.5; reward rail ≥ 361; ≥ 30 episodes/evaluation; verdict from the selected checkpoint's evaluation_selected.csv (post-stage; fail-closed when absent) |

**Backend-specific success semantics:**
- **Stable-Baselines3 — Head-contact bite proxy:** The head-bite geom contacts the prey geom while the bite reward is enabled; the model has no articulated jaw.
- **JAX/MJX — Head-tip proximity bite proxy:** The head-tip site comes within 0.35 m of the prey target position while the bite bonus is enabled; physical geom contact is not required and the model has no articulated jaw.

**Per-deliverable success semantics:**
- **stance (1 — Balance) · Stable-Baselines3 — Stance quality (stance_quality/v2):** The stance checkpoint clears the stance_quality/v2 gate, judged post-stage on the handoff pair: the one-sided 95% lower bound on the share of the 40 certification-panel episodes that are clean on floor truth (horizon reached, no hop or impact in the settle window, both feet down and loaded, flat pads, no actuator held at its limit) meets the configured bound, with the reward rails as collapse floors only. The clean count and its bound are recorded in gate_verdict.json but not exported to summary.json until a later phase (decision D-A9, deferred by D-B15), so the catalog names them with no value.
- **recovery (recovery — Recovery) · Stable-Baselines3 — Recovery under pushes (recovery_quality/v1):** The recovery checkpoint clears the recovery_quality/v1 gate, judged post-stage against the run's frozen gate_resolution.json: recovery-success LCB95 and the paired delta against each required frozen null meet the frozen thresholds, with re-entry inside the configured step budget and dwell. The statistics are not exported to summary.json until a later phase (decision D-A9, deferred by D-B15).
- **walk (2 — Locomotion) · Stable-Baselines3 — Gated forward velocity (reward_and_length/v1):** The locomotion checkpoint clears the reward_and_length/v1 gate: mean forward velocity at or above the stage's configured minimum, with its reward and episode-length floors, over the required consecutive evaluations.
- **hunt (3 — Bite) · Stable-Baselines3 — Head-contact bite success (task_success/v1):** The hunt checkpoint clears the task_success/v1 gate: the one-sided 95% Clopper-Pearson lower bound on per-episode head-contact bite success (the head-bite geom contacts the prey geom while the bite reward is enabled; the model has no articulated jaw) over the selected checkpoint's evaluation_selected.csv at the stage's declared min_eval_episodes, hash-bound to that checkpoint, meets the configured min_success_lcb bar (provisional until the first Phase-B pilot re-freezes it; decision D-B2), with the reward rail as a collapse floor only. The bound is exported to summary.json as selected_model_success_lcb beside the raw selected-checkpoint rate.
- **hunt (3 — Bite) · JAX/MJX — Head-tip proximity bite proxy:** The head-tip site comes within 0.35 m of the prey target position while the bite bonus is enabled; physical geom contact is not required and the model has no articulated jaw. The JAX/MJX path, retired by D-D17, could not judge task_success/v1 (no MJX hunting panel existed) and refused the stage rather than certifying it.

[Full documentation →](../environments/trex/README.md)

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
| Plant contract revisions | policy r8; physics r4; visual r2 ([details](PLANT_CONTRACT.md)) |
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

[Full documentation →](../environments/brachiosaurus/README.md)

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
| Plant contract revisions | policy r7; physics r1; visual r1 ([details](PLANT_CONTRACT.md)) |
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

[Full documentation →](../environments/dibothrosuchus/README.md)

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
| Plant contract revisions | policy r2; physics r1; visual r1 ([details](PLANT_CONTRACT.md)) |
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

[Full documentation →](../environments/compsognathus/README.md)

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
| Plant contract revisions | policy r2; physics r1; visual r1 ([details](PLANT_CONTRACT.md)) |
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

[Full documentation →](../environments/compsognathus/README.md)
<!-- END GENERATED: SPECIES -->

## Training Results

<!-- BEGIN GENERATED: RESULTS -->
The summaries below are historical experiment records generated from the versioned JSON files under
`results/`. They are not evidence for the current model revision unless provenance is marked both current
and verified. Current stage budgets may therefore differ from the steps reported here.

### Velociraptor Mongoliensis (PPO · Stable-Baselines3) — 2026-03-15

**Provenance:** Historical model; unverified; evaluation episode count not recorded; Stable-Baselines3 (version not recorded). **Run total:** 22M steps; 11:25:15. [Source summary](../results/velociraptor/ppo/summary.json).

| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |
|---|---:|---:|---:|---:|---:|
| 1 — Balance | 1964.43 | 0.11 m/s | — | 6M | passed retired gate (reward gate) |
| 2 — Locomotion | 2678.68 | 3.47 m/s | — | 8M | passed retired gate (reward gate) |
| 3 — Strike | 1366.19 | 2.02 m/s | 93.3% | 8M | passed retired gate (reward gate) |

**Current Stable-Baselines3 catalog definition for this task label:** A left or right sickle-claw geom contacts the prey geom while the strike reward is enabled.

### Velociraptor Mongoliensis (SAC · Stable-Baselines3) — 2026-03-21

**Provenance:** Historical model; unverified; evaluation episode count not recorded; Stable-Baselines3 (version not recorded). **Run total:** 22M steps; 22:59:18. [Source summary](../results/velociraptor/sac/summary.json).

| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |
|---|---:|---:|---:|---:|---:|
| 1 — Balance | 970.19 | -0.64 m/s | — | 6M | passed retired gate (reward gate) |
| 2 — Locomotion | 2078.62 | 2.91 m/s | — | 8M | passed retired gate (reward gate) |
| 3 — Strike | 1195.43 | 1.63 m/s | 90.0% | 8M | passed retired gate (reward gate) |

**Current Stable-Baselines3 catalog definition for this task label:** A left or right sickle-claw geom contacts the prey geom while the strike reward is enabled.

### Tyrannosaurus Rex (PPO · Stable-Baselines3) — 2026-03-18

**Provenance:** Historical model; unverified; evaluation episode count not recorded; Stable-Baselines3 (version not recorded). **Run total:** 22M steps; 13:02:32. [Source summary](../results/trex/ppo/summary.json).

| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |
|---|---:|---:|---:|---:|---:|
| 1 — Balance | 3008.66 | 0.02 m/s | — | 6M | passed retired gate (reward gate) |
| 2 — Locomotion | 1936.01 | 3.47 m/s | — | 8M | passed retired gate (reward gate) |
| 3 — Bite | 1294.28 | 1.68 m/s | 96.7% | 8M | passed retired gate (reward gate) |

**Current Stable-Baselines3 catalog definition for this task label:** The head-bite geom contacts the prey geom while the bite reward is enabled; the model has no articulated jaw.

### Brachiosaurus Altithorax (PPO · Stable-Baselines3) — 2026-07-18

**Provenance:** Historical model; unverified; 30 evaluation episodes; Stable-Baselines3 (version not recorded). **Run total:** 34.0214M steps; 19:46:16. [Source summary](../results/brachiosaurus/ppo/summary.json).

| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |
|---|---:|---:|---:|---:|---:|
| 1 — Balance | 1740.53 | 0.01 m/s | — | 6.00474M | passed retired gate (reward gate) |
| 2 — Locomotion | 6634.60 | 1.42 m/s | 3.3% | 16.0072M | passed retired gate (reward gate) |
| 3 — Food Reach | 1368.13 | 0.71 m/s | 100.0% | 12.0095M | passed retired gate (reward gate) |

**Current Stable-Baselines3 catalog definition for this task label:** The head-tip site comes within the configured food-reach threshold of the food target while the food-reach bonus is enabled.
<!-- END GENERATED: RESULTS -->
