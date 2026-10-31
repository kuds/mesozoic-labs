# Reference: Microduck's training stack, read against the T. rex tail habit (2026-10-31)

**Status:** dated reference, report only. It was read on 2026-10-31 at the maintainer's request, as a pointer for when
this project moves to hardware training. Nothing here changes code, configuration, rewards or plans. Nothing was run:
the facts about Microduck below come from reading its training repository and its launch coverage, cited at the end.

**What Microduck is.** Microduck is a ~800 g, ~25 cm bipedal robot from Hugging Face and Pollen Robotics.
- Its reinforcement-learning stack is open source: `pollen-robotics/microduck_rl`, read at commit `bae39f59` of
  2026-10-27.
- It is built on mjlab (MuJoCo Warp) with rsl_rl PPO.
- Policies are trained at 50 Hz, exported to ONNX and run on the robot (`README.md:9`).
- File paths cited below are under `src/mjlab_microduck/` in that repository, except `README.md`, which is at its
  root.

**Why it is here.** The T. rex tail saturation habit
([record](../investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md), §9) points at a training-setup cause rather than
a physical one. The critic cannot see time, yet four stance reward terms switch on only after step 200 of each episode
on the r10 run branch. (On `main` only `stance_width` does;
[#655](https://github.com/kuds/mesozoic-labs/pull/655) §6.2.) The critic then uses the slowly creeping tail angle as a
clock. Microduck's stack avoids that class of problem by
design, and it already uses the remedy being tested next.

**How to read it.** This project's current models are *anatomical* (a realistic T. rex), not robots. Some of
Microduck's practices are about the learning setup and apply to them now. Others exist to bridge simulation and real
hardware, and only matter once hardware training starts. The two are kept apart below.

## A. Applies to the anatomical models now (training setup, not hardware)

1. **No hidden time in the reward.**
   - **Microduck.** Its walking task's rewards do not change over an episode: velocity tracking, pose, action rate,
     air time, foot clearance and similar terms (`tasks/microduck_velocity_env_cfg.py`). Where a task does depend on
     time (crouch, spin, ground pick), the phase is an explicit input, `[cos 2πφ, sin 2πφ, 0]` with a 4 s period,
     carried in the velocity-command slot (`twist`) of the observation (`tasks/microduck_spin_env_cfg.py:8-9,
     374-387`). The command terms go to both the actor and the critic: `tasks/microduck_velocity_env_cfg.py:691-693`
     adds the head and body commands to both groups. That `twist` itself is in both comes from mjlab's base
     configuration and was not traced here.
   - **Here.** The stance task's step-200 gate is a hidden variable that the critic must guess. The candidate fixes
     are:
     - a critic-only phase input (a BS2-style screen is being prepared);
     - making the gated terms stationary.
2. **Critic-only ("privileged") information: asymmetric actor-critic.**
   - **Microduck.** The critic sees clean, simulation-only state the actor does not:
     - the base linear velocity (`microduck_velocity_env_cfg.py:539-540`);
     - true joint positions without the actor's encoder bias (`:629-632`);
     - true IMU readings without the actor's mounting error (`:594`);
     - in the ball-kick task, the ball's position and velocity (`tasks/microduck_ball_kick_env_cfg.py:12-16,
       388-396`, "no ball sensing on the real robot").
   - **Here.** The T. rex critic sees exactly the actor's 63 inputs. A critic-only channel would carry training-only
     information (such as the phase) without changing the deployed actor. It is a general mechanism, broader than the
     phase fix alone.
3. **The previous action as an input (deferred).**
   - **Microduck.** Its actor's 61 inputs include the previous 14 actions, alongside an action-rate penalty whose
     weight ramps in over training (`microduck_velocity_env_cfg.py:347-349, 776-781`).
   - The 61-input layout is from launch coverage. It is consistent with 3 + 3 + 14 + 14 + 14 + 13 = 61, but was not
     traced into mjlab's base configuration.
   - **Here.** #655 notes that the smoothness and jerk penalties depend on an action history that neither network
     sees. That is a candidate cause of #491's chatter. It is **not** part of the tail habit: rate penalties are flat
     across the episode, so they are not a clock.
   - **Decision (2026-10-31):** keep it out of the current phase test (one change at a time). Consider it for a later
     policy-interface revision, ideally bundled with a phase input if that fix goes into a full run. It changes the
     deployed policy's input and needs the checkpoint-widening path.

## B. For hardware training later

These map to [SIM_TO_REAL_PLAN.md](SIM_TO_REAL_PLAN.md) §3.1 (observation realizability) and §3.4 (domain
randomization, all "not implemented" here today).

- **Sensor-realistic actor observations, with the critic kept clean.**
  - Observation noise (`microduck_velocity_env_cfg.py:587-590`).
  - IMU delay of 0–1 control steps (`:565, 569`) and joint-velocity delay (`:612`).
  - Per-environment encoder bias (`:629-632`).
  - IMU mounting error (`:594`).
- **Actuator realism.**
  - A fitted servo friction model with per-environment friction randomization (`actuator/friction_dr_bam.py`;
    `microduck_velocity_env_cfg.py:36`).
  - Backlash variants of the model (`tasks/backlash.py`, `robot/microduck/add_backlash.py`).
- **Plant and contact randomization.**
  - Foot friction 0.7–1.3 (`:377`).
  - Mass and inertia ±5% with the CoM unchanged (`:454-456`).
  - Velocity pushes (`:395`).
- **Deployment.** ONNX export with the observation normaliser baked in (`README.md:176-177`), run on the robot at
  50 Hz (`README.md:9`).

## Caveats

- **Read, not run.** No Microduck training was reproduced here.
- **Scale and plant differ.** Microduck is a 25 cm, 14-servo robot. The models here are anatomical dinosaurs with
  different scale, actuators and reward design.
- **No direct counterpart found.** No public report of a tail-style saturation habit on Microduck was found.
- **mjlab history.** This repository retired its own mjlab scaffold (D-D17; SIM_TO_REAL_PLAN.md §3.4), while Microduck
  is built on mjlab. That is a framework note only, not a recommendation.

## Sources

- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl), read at commit `bae39f59`
  (2026-10-27); file and line citations above refer to that commit, with paths under `src/mjlab_microduck/` except
  `README.md`.
- [How Microduck Learns to Walk: MuJoCo, PPO and Sim-to-Real](https://openelab.io/blogs/learn/how-microduck-learns-to-walk-mujoco-ppo-sim-to-real)
  (the 61-input layout and 50 Hz).
- [Hugging Face launches open-source Microduck robot](https://www.opensourceforu.com/2026/08/hugging-face-launches-open-source-microduck-robotc/)
  and [Microduck pre-orders](https://pasqualepillitteri.it/en/news/13137/microduck-hugging-face-open-source-robot)
  (launch, size, price, open software stack).
