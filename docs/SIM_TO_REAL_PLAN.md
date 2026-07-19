# Sim-to-Real Plan — What It Takes to Build Physical Dino Robots

> **Status:** Scoping / feasibility record. This is an engineering assessment of
> the gap between the current simulation and a walking physical robot, plus a
> phased plan to close it. It is **not** a claim that any hardware exists — per
> [`ROADMAP.md`](ROADMAP.md) Phase 6 and
> [`configs/species_manifest.toml`](../configs/species_manifest.toml),
> `hardware_prototype` is `planned` and `sim_to_real_validation` is
> `not_started`. Nothing here has been built or transferred to hardware.

This document answers a single question: **what would it actually take to turn
the Mesozoic Labs MuJoCo simulation into physical robots?** It is grounded in a
file-by-file audit of the three MJCF models, the observation/action pipeline,
the training stack, and the project's own docs.

---

## Table of Contents

1. [Bottom Line](#1-bottom-line)
2. [Where the Repo Stands Today](#2-where-the-repo-stands-today)
3. [The Six Gaps Between Sim and Hardware](#3-the-six-gaps-between-sim-and-hardware)
4. [Feasibility by Species](#4-feasibility-by-species)
5. [Phased Plan](#5-phased-plan)
6. [Software-Only Quick Wins (start today)](#6-software-only-quick-wins-start-today)
7. [Recommended Path](#7-recommended-path)
8. [Biggest Risks](#8-biggest-risks)

---

## 1. Bottom Line

Reaching a physical robot walking on a sim-trained policy is a **multi-quarter
program**, and the correct first move is **not** to build one of the existing
charismatic species. The honest summary:

- **The plumbing does not exist.** Domain randomization, sensor noise, action
  delay, a hardware abstraction layer (HAL), a ROS 2 bridge, and any
  policy-export path are all absent — `planned`/`not_started` in the repo's own
  capability manifest.
- **There is no buildable artifact yet.** The species the roadmap names as the
  physical target — **Compsognathus** — has no MJCF, no environment, and no
  actuator spec. Phase 6 depends on a model that has not been designed.
- **The deepest technical gap is the observation space.** ~7 of the 67–83
  observation dimensions are *privileged simulator state* (world-frame base
  linear velocity and an absolute prey/food beacon) that **no onboard sensor can
  measure**. Closing this needs a legged base-state estimator *and* an onboard
  perception stack — neither exists in the repo.
- **The heavy species are not servo-buildable.** T-Rex (85 kg) demands
  ~150–375 N·m and Brachiosaurus (175 kg) up to ~900 N·m at the hips — beyond
  all COTS smart servos and most quasi-direct-drive (QDD) units.
- **The good news:** compute is a non-issue (a ~200K-parameter MLP at 100 Hz
  runs on a microcontroller), and the observation-building, action-scaling, and
  normalization code already exist as pure, backend-agnostic modules — so the
  export/runtime work is a thin adapter, not a rewrite. And a large fraction of
  the highest-leverage work (domain randomization, noise, delay, obs reframing)
  is **pure software you can start today** without any hardware.

---

## 2. Where the Repo Stands Today

| Capability | Status | Source |
|---|---|---|
| MuJoCo simulation | implemented | `configs/species_manifest.toml` |
| Domain randomization | **planned** | `configs/species_manifest.toml:7-9` |
| Hardware prototype | **planned** | `configs/species_manifest.toml:11-13` |
| ROS 2 bridge | **planned** | `configs/species_manifest.toml:14-16` |
| Sim-to-real validation | **not_started** | `configs/species_manifest.toml:17-18` |

The training codebase is mature (SB3 + JAX/MJX dual backend, curriculum manager,
W&B, sweeps, a "plant contract" fingerprinting system). Everything downstream of
training toward hardware is greenfield.

Additional maturity caveats that matter for hardware:

- **No reproducible, verified, deployable policy exists.** All four published
  result summaries are self-labelled *historical + unverified* with
  `backend_version = null` (`results/*/summary.json`), and the plant contract
  explicitly refuses to certify legacy artifacts as current
  (`docs/PLANT_CONTRACT.md:99-101`). You cannot deploy a policy you cannot
  reproduce.
- **The sim plant is still churning.** Recent breaking physics changes (actuator
  forcerange resizes, Brachiosaurus leg-spring rework) invalidated prior
  checkpoints by contract (`CHANGELOG.md`, `docs/KNOWN_ISSUES.md:120-166`). There
  is no frozen controller to transfer.
- **Reported "success" is a sim proxy.** T-Rex "bite" is fixed head-geom contact
  (no articulated jaw); Brachiosaurus "food reach" is a head-tip distance
  threshold — not physical contact events a robot would reproduce.

---

## 3. The Six Gaps Between Sim and Hardware

### 3.1 Observation realizability — the deepest gap

All species share one observation builder
(`environments/shared/obs_functions.py:102-115`):

```
[ joint_pos, joint_vel, root_quat(4), root_gyro(3),
  root_linvel(3), root_accel(3), foot_contacts, target_dir(3), target_dist(1) ]
```

~84–90% of the vector is **realizable** on hardware (joint encoders, IMU
gyro/accel/orientation, foot contact). The trap is a small, load-bearing
minority:

- **Base linear velocity** is read straight from `qvel[0:3]` — world-frame
  ground-truth base velocity (`obs_functions.py:94`). No onboard sensor measures
  this; it is the single hardest quantity to estimate on a legged robot. It is
  also the *primary locomotion reward signal*, so the policy is deeply
  conditioned on it.
- **Target direction + distance** are computed from the absolute world position
  of a **mocap body** (`obs_functions.py:98-100`; `base_env.py:700-702`) — the
  simulator equivalent of a motion-capture beacon on the prey/food. There is no
  onboard perception code anywhere in the repo.
- **Absolute yaw.** Orientation is a world-frame `framequat`
  (`raptor.xml:280`) that includes yaw. The models carry no magnetometer, so yaw
  is unobservable onboard. Because base velocity *and* the target vector are also
  world-frame, the whole observation is anchored to a non-realizable absolute
  yaw reference.

**Implication:** before a real robot can even *assemble a valid observation*, you
must (a) re-specify the observation in an egocentric/body frame, (b) build a
base-state estimator, and (c) build onboard target localization (or drop to a
proprioception-only task formulation).

### 3.2 Morphology & actuation

| Species | Mass | Actuated DOF | Peak hip/knee torque (model) | Buildable as servo project? |
|---|---:|---:|---|---|
| Velociraptor | 13.5 kg | 22 | ~145–225 N·m | **With work** (prune DOF, QDD BLDC) |
| T-Rex | 85.4 kg | 21 | ~300–375 N·m | No — Cheetah-3-class / hydraulic |
| Brachiosaurus | 175.3 kg | 30 | ~500–900 N·m | No — industrial / hydraulic |

Key idealizations that don't map to hardware:

- **No motor model.** Every locomotion joint is an ideal MuJoCo `<position>`
  servo applying full `forcerange` torque instantaneously — no torque–speed
  curve, current/thermal limit, back-EMF, backlash, or latency
  (`raptor.xml:232-269`). (Joints do carry `armature` and viscous `damping`, but
  these are passive physics, not an actuator model.)
- **Torque caps are anchored to measured gait, not arbitrary.** The `forcerange`
  values were sized from the actuator-saturation tooling
  (`environments/shared/scripts/actuator_saturation_report.py`,
  `tests/test_actuator_bounds.py`): lower 0.8× kp caps clipped 20–50% of real
  gait torque and broke Stage-2 locomotion, so gait-critical joints were raised
  to 1.5× kp. **Real per-joint peak torque therefore lies between ~0.8× and
  1.5× kp** — the heavy-species torque problem is real, not an artifact of
  oversizing.
- **Uniform-density geometry.** Mass sits at limb geometric centers, not at the
  joints (motors + gearboxes) and trunk (battery) where a real robot carries it —
  so CoM and inertia tensors are miscalibrated for balance transfer.
- **Impractically high DOF density.** 22–30 actuators including per-toe servos,
  dual sickle-claw torque motors, tail-yaw, and forelimb stubs. Most are
  unnecessary to walk; a real build should prune to ~8–10 essential leg DOF.
- **Brachiosaurus offloads static gravity onto lossless passive springs**
  (`stiffness=120 N·m/rad` with `springref` at stance,
  `brachiosaurus.xml:13-27`). On hardware that load becomes continuous motor
  torque (thermal death) or added physical springs/series-elastic elements that
  change the dynamics — the standing pose is non-physical as modeled.

### 3.3 Control loop & action interface

- **100 Hz zero-order-hold control** (timestep 0.002 s × frame_skip 5;
  `base_env.py:78-79`). Modest and embeddable.
- **Actions are normalized `[-1,1]` position targets** mapped to per-actuator
  `ctrlrange` in radians (`base_env.py:783-797`), except the raptor's two
  torque-driven sickle claws — a mixed position+torque interface.
- The servo "D" term is **passive joint damping**, not a servo velocity loop, so
  reproducing sim closed-loop stiffness on hardware requires matching *both* the
  sim `kp` **and** replicating joint damping as an explicit `kd`.
- **No action filtering, latency, or delay modeling** anywhere in the step loop.

### 3.4 Robustness / domain randomization

The only stochasticity implemented is **reset-time initial-state noise**
(`base_env.py:817-823`). Everything that closes the reality gap is **absent**:

| Technique | Status |
|---|---|
| Ground-friction randomization | not implemented |
| Actuator-strength / gain randomization | not implemented |
| Joint-damping / gravity / mass randomization | not implemented |
| Per-step observation / sensor noise | not implemented |
| Action / sensor delay | not implemented |
| External push perturbations | not implemented |
| Terrain diversity (heightfields/slopes) | not implemented |
| Actuator dynamics (time constants / lag) | not implemented |

The mjlab adapter defines default DR ranges but every factory raises
`NotImplementedError` (`mjlab_env.py:172`, `velociraptor/mjlab_config.py:57-63`) —
a scaffold, not a mechanism. **Any policy trainable today is overfit to one
idealized plant and would not survive transfer.**

### 3.5 Deployment / software stack

- **No export path.** Inference only runs inside the training stack — SB3
  `model.predict()` (needs PyTorch + a live VecNormalize env) or JAX
  `network.apply()` (welded to MuJoCo/MJX). No ONNX/TensorRT/tflite/TorchScript.
- **Normalization stats are Python pickles** (SB3 `_vecnorm.pkl`; JAX `obs_rms`
  inside the checkpoint). A C/C++ embedded runtime cannot load these, and
  skipping them silently corrupts the policy input.
- **No hardware abstraction at all** — grep for `ros2|rclpy|dynamixel|servo|pwm|
  serial|HAL` returns zero hits.
- **The network is tiny** — `[512, 256]` tanh MLP (`jax_ppo.py:75-103`) — so
  compute is not the barrier. The barrier is missing plumbing.

### 3.6 Author-stated posture

The authors are explicit and consistent: sim-to-real is Phase 6, `not_started`,
blocked on Phases 2–5; the models are "research abstractions rather than
validated reconstructions" (`README.md:12`); and DR + sensor noise (the two
prerequisites the authors themselves name for the HAL) are unchecked Phase 2
items. This plan agrees with that posture and sequences the work accordingly.

---

## 4. Feasibility by Species

- **Compsognathus — most feasible in principle, zero artifact today.** At true
  animal scale (~1–3 kg, ~1 m), hobby smart servos (Dynamixel XM430 ~4 N·m,
  XH540 ~10 N·m) on a 3D-printed frame with an ESP32/RPi are entirely adequate
  for ~6–8 leg DOF. **Blocker:** it does not exist in sim — no MJCF, env, or
  actuator spec.
- **Velociraptor — feasible with work.** Most buildable of the existing three.
  At 13.5 kg, realistic gait torques are tens of N·m, coverable by
  mini-cheetah/Unitree-class QDD BLDC (~17 N·m cont., ~33–40 N·m peak) with mild
  gearing. The obstacle is DOF density, not torque — prune to ~8–10 leg DOF.
- **T-Rex — impractical.** ~150–375 N·m at hip/knee exceeds all COTS smart
  servos and most QDD units; needs Cheetah-3-class geared actuators or
  hydraulics.
- **Brachiosaurus — impractical.** ~500–900 N·m plus non-physical lossless
  load-bearing springs. Industrial/hydraulic territory only.

---

## 5. Phased Plan

**Phase 0 — Software-only de-risking (start today, no hardware) · months.**
Close the largest gaps that are pure code and stand up the export plumbing:
implement the DR engine, add observation noise and an action-delay buffer,
re-derive real per-joint torque (peak *and* RMS/duty-cycle) from gait rollouts,
build a policy exporter + normalization dumper, fix SB3↔JAX backend divergence,
and produce **one** reproducible, provenance-complete, verified policy bundle.

**Phase 1 — Design the buildable-scale target in sim · quarters.**
Author a ~1–3 kg Compsognathus MJCF with ~6–8 essential leg DOF; replace ideal
position servos with a torque-producing motor model + torque-speed/thermal/
latency limits fit to a chosen real servo; rebuild mass/inertia with concentrated
motor mass at joints and battery in the trunk; add a compliant foot-pad contact
model; stand up its 3-stage curriculum.

**Phase 2 — Robustness training (must precede any transfer) · quarters.**
Re-specify the observation in an egocentric, yaw-invariant body frame and
retrain; turn on friction/damping/mass/gravity/actuator-gain DR + external pushes
+ terrain; retrain with obs noise + action delay; build a robustness eval suite
and gate transfer readiness on it.

**Phase 3 — State estimation & perception (the deep gap) · quarters.**
Build a contact-aided / invariant-EKF base-state estimator (IMU + joint
kinematics + foot contact) to replace world-frame `qvel[0:3]`; build onboard
target localization (camera/LiDAR + detector + bearing/range) to replace the
mocap beacon, *or* retrain on a proprioception-only observation; add and
calibrate foot force sensors.

**Phase 4 — Deployment stack & HAL · quarters.**
Write a standalone MuJoCo-free inference runtime (thin adapter over the existing
pure `obs_functions` / `scale_action` / normalization); build the HAL (action →
servo command mapping; encoder/IMU/contact ingestion normalized to sim format);
design an onboard PD loop matching sim `kp` and emulating joint damping as `kd`;
build the ROS 2 real-time bridge with safety limits; benchmark loop latency
< 10 ms.

**Phase 5 — Hardware build, bring-up & sim-to-real · quarters.**
3D-printed frame, servo joints, ESP32/RPi + IMU/encoder/foot-contact suite
matching the MJCF; system-ID to auto-tune the MJCF from real data; real-robot
eval harness; publish technical report + open hardware files (STL, BOM,
assembly).

---

## 6. Software-Only Quick Wins (start today)

These need no hardware, live inside this repo, and directly de-risk transfer:

1. **DR engine in SB3 `BaseDinoEnv`** — cache nominal friction/damping/mass/
   gravity/actuator-gain at init, resample per reset from opt-in TOML ranges
   (Stage 2+). Highest single leverage. *(M)*
2. **Per-step observation noise** — additive Gaussian + optional IMU bias drift +
   touch threshold, after `_get_obs` / `build_mjx_observation`, disabled by
   default. *(S)*
3. **Action-delay ring buffer** — configurable 1–3 control steps, symmetric in
   SB3 and MJX. *(M)*
4. **Real torque table** — re-derive per-joint peak *and* RMS/duty-cycle torque
   from recorded gait rollouts via the existing
   `actuator_saturation_report.py` + `test_actuator_bounds` tooling, turning
   servo caps into a defensible BOM sizing table. *(M)*
5. **Policy exporter + normalization dumper** — reuse the already-pure
   `obs_functions`, `scale_action_jax`, and `jax_normalization` modules; export a
   deterministic-mean head to ONNX/npy plus a language-neutral mean/var. *(S–M)*
6. **Egocentric observation reframe** — rotate base velocity + target into the
   root frame, express the target as a yaw-invariant bearing, drop absolute yaw;
   retrain. Removes the magnetometer/world-frame dependency that blocks any
   onboard-estimator drop-in. *(L)*
7. **Prune to a buildable DOF variant** — drop toe servos, sickle-claw motors,
   forelimb stubs, tail-yaw → ~8–10 leg DOF; keep the high-DOF research variant
   separate. *(M)*
8. **External push perturbations** — randomized impulses on the root via
   `xfrc_applied`; the roadmap calls this the single most impactful balance
   technique. *(M)*
9. **Backend parity** — MJX reset to the home keyframe, JAX action-clip parity,
   fix `render_fps`, add an SB3↔JAX reward-parity CI test so one canonical
   behavior can be pinned. *(S)*

---

## 7. Recommended Path

1. **Do the Phase 0 software quick wins now** — DR, obs noise, action delay,
   real torque measurement, egocentric obs reframe, and the exporter — inside
   this repo, no hardware required.
2. **Design and commit a Compsognathus MJCF** at true scale (~1–3 kg, ~6–8 leg
   DOF) with actuators parameterized from a *specific* hobby servo and mass
   concentrated at joints/trunk. This is the actual buildable target; nothing
   downstream can start without it.
3. **Retrain with full DR/noise/delay/perturbation on the reframed observation**,
   and build the base-state estimator + a target-localization (or
   proprioception-only) observation so the robot can assemble a valid input.
4. **Build the MuJoCo-free runtime + HAL + ROS 2 loop** and a 3D-printed servo
   prototype on ESP32/RPi, then close the loop with system-ID.

Keep a **pruned ~8–10 DOF Velociraptor** (13.5 kg, QDD BLDC) as a stretch/fallback
demonstrator — but only after the Compsognathus path proves the toolchain. It is
the most buildable of the existing three, yet it is a bigger, costlier machine
that shares every one of the same estimator/DR/HAL prerequisites.

**Do not** target T-Rex or Brachiosaurus as physical robots.

---

## 8. Biggest Risks

1. **Privileged observation** (world-frame base velocity + mocap target) is
   non-measurable onboard and requires two subsystems (base-state estimator +
   perception) that do not exist anywhere in the repo.
2. **Wrong-species commitment.** Aiming hardware at T-Rex/Brachiosaurus is a dead
   end on torque grounds alone.
3. **No reproducible/verified/deployable policy** exists, and the plant is still
   changing — there is no frozen controller to transfer.
4. **Zero robustness training** means direct transfer would fail via slipping,
   servo-lag instability, and sensor-noise divergence. This is a prerequisite,
   not polish.
5. **The named target (Compsognathus) doesn't exist in sim** — the hardware
   roadmap is gated on 0%-started upstream work.
6. **Actuator/control mismatch** — reproducing sim `kp` + passive-damping
   closed-loop stiffness on a real servo's `kp/kd` is unsolved and easy to get
   subtly wrong.

---

*This assessment was produced by auditing the MJCF models, observation/action
pipeline, training stack, and project docs. It aligns with — and sequences —
the existing [`ROADMAP.md`](ROADMAP.md) Phase 2 (robustness) and Phase 6
(sim-to-real) items.*
