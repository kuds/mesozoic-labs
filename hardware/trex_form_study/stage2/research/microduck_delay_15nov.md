# How Microduck treats control delay, read against the 75 cm panels (15 Nov 2026)

**Status: a dated reference, read and not run. Nothing here is adopted and nothing changes a rule.** The owner asked,
after the 75 cm panels (`STAGE2_75_REPORT.md`), how the open-source Microduck project handles the same problem: a
walk that falls once the servo delay is realistic. An earlier note on the same project, about its training setup, is
`docs/hardware/REFERENCE_MICRODUCK_2026_10.md` at the repository's root.

**What was read.** Microduck's training repository, `pollen-robotics/microduck_rl`, at commit `0daab16`
(10 Nov 2026), and the simulation library it pins, `mujocolab/mjlab` at tag `v1.3.0` (commit `19d6c06`). Both were
cloned read-only; no code of either was run. File paths below are in those repositories.

## What Microduck does

1. **The delay is inside the training simulation, drawn at random, and the policy is trained through it.**
   - Command delay: every servo command reaches the servo model 3 to 6 physics steps late
     (`src/mjlab_microduck/robot/microduck_constants.py:173-174`, `delay_min_lag=3`, `delay_max_lag=6`).
   - The unit is the physics step: mjlab's documentation says "Delays are quantized to physics timesteps"
     (`docs/source/actuators.rst:247-250`; `src/mjlab/actuator/actuator.py:67-77`).
   - The physics step is 5 ms and the policy runs every fourth step, at 50 Hz
     (`src/mjlab/tasks/velocity/velocity_env_cfg.py:446-451`; Microduck's `README.md:9`). So the command delay it
     trains through is **15 to 30 ms**.
   - Sensor delay on top: the IMU readings 0 to 1 control step late (0 to 20 ms), and the joint velocity always one
     control step late (20 ms), because the servo's firmware reports a moving average
     (`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:564-570, 603-613`).
2. **The numbers were set from the real hardware, not assumed.**
   - The IMU delay range was narrowed from 0 to 3 control steps (60 ms at worst) to 0 to 1 after "the 2026-07 audit"
     found that the real robot's IMU path is fast (the comment at `microduck_velocity_env_cfg.py:565`).
   - There is a one-servo test bench in the repository, with its own model and training task, used to match the model
     to the real servo (`src/mjlab_microduck/robot/testbench_constants.py`,
     `src/mjlab_microduck/tasks/testbench_env_cfg.py`).
3. **The servo is modelled in detail, not as an ideal position drive.** It uses the BAM model of its Dynamixel XL330:
   the firmware's voltage control law, back-EMF, and Coulomb, Stribeck and load-dependent friction, with the battery
   voltage, the sag under load, the friction and the command delay drawn per environment (`README.md:114-122`). The
   README's own words: "actuator fidelity is most of the sim2real gap". Each main task also has a twin trained with
   ±1° of gear play in every servo joint (`README.md:100-110`).
4. **A slow outer loop.** The policy sends position targets 50 times a second. The fast loop is the servo's own
   firmware.

## What this does and does not say about the 75 cm build

- **It is the same remedy this study already planned.** `STAGE2_PLAN.md` section 4 makes the judged controller of
  Stage 2 a policy trained with the delay drawn at random from 0 to 24 ms (S2-9), with stiffness, friction, mass and
  damping randomised as well. It also says why the tracker is not kept: "the tracker falls in nearly every 60 cm
  walk at 10 ms". The 75 cm panels of 15 Nov ran that tracker, fitted with no delay, as the baseline; the trained
  policy has not been run on this body yet.
- **It is outside evidence that a delay of 10 ms does not by itself rule out walking.** Microduck walks on real
  hardware after training through 15 to 30 ms of command delay, and it is about 25 cm tall, which tips over faster
  than a 75 cm build. That is another project's robot, read from its code and its public videos, and not a result of
  this study. No public figure was found for how often it succeeds over repeated trials.
- **Two things Microduck has that this study does not yet have.** A servo bench measurement behind its delay and
  friction numbers (this study's counterpart is the delay bench sheet, `stage2/jetson75/BENCH_SHEET.md`, not yet
  run), and a servo model with friction and gear play (`STAGE2_PLAN.md` section 8: "ideal PD servos behind an
  envelope and a delay; no backlash").
- **The robots differ.** Microduck is about 800 g with 14 small servos that each run their own position loop. The
  75 cm build is far heavier and its leg drives are a different kind. Nothing here transfers as a number.
