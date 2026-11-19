# The gait of the trike60-pipe1 policy (simulation)

19 Nov 2026, corrected after the independent review. Analysis of the owner's Colab pipeline-test policy (2,007,040
steps, PR #669's foot), per `PLAN_GAIT_PIPE1.md` (committed before the analysis; changes after a result are listed at
the end). One policy, so every figure is reported, not a verdict. Simulation of a model whose masses are estimates.
Tool output: `results/gait_pipe1_2026-11-19.json` and `.md`, `results/gait_freq_pipe1_2026-11-19.json`.

## Answer first

- **It walks without falling.** At the 0.25 m/s command it holds 0.238 m/s with a quick diagonal, trot-like gait: the
  legs and feet cycle at about **2.4 Hz** (a 0.42 s stride, about 10 cm per stride), with a diagonal pair of feet down
  77 % of the time. **No falls in 25 episodes**, including the stiffer contact setting, the roll-ankle foot it never
  trained on, and the nominal model. It drifts sideways by up to 0.23 m over 18 s (0.43 m on the roll-ankle foot) and
  its heading wanders by up to 5°.
- **It is not yet a gait the robot should use:**
  - **Heat (RMS torque over 18 s against the continuous rating; mean of 5 episodes, worst in brackets):** the two front
    pitch joints 1.02 × (1.09-1.11 ×) and the **left** hind knee 1.10 × (1.15 ×). The other joints are under their
    ratings: right hind knee 0.92 ×, front elbows 0.94-0.96 ×, hind pitch 0.66-0.83 ×, roll 0.47-0.71 ×. The gait is
    asymmetric. RMS over 18 s stands in for steady walking; it is not a thermal simulation.
  - **Torque limits:** the policy uses the full torque available. Against each episode's randomised limit (0.85-1.0 ×
    the model's) the fore pitch joints sit at their limit in 5-10 % of steps, the elbows and hind knees in 2-5 %;
    against the model's nominal limit, which the randomised episodes rarely reach, only the left hind knee does more
    than briefly (0.9 % of steps; the right hind roll 0.1 %); on the nominal model without randomisation, the pitch,
    elbow and knee groups reach it in 3.0, 1.8 and 3.0 % of steps.
  - **Slip:** each loaded foot slides about a quarter of the distance the body travels (slip ratio 0.23-0.29 at
    0.25 m/s, 0.30-0.37 at 0.15 m/s).
  - **Power:** 42 W of joint mechanical power (sum of |torque × speed|, not electrical power) at 0.24 m/s.
- **What it means.** The pipeline works and a first gait exists. The budget's hind-knee warning (F12, about 1.00 × by
  estimate for an average load) now has a simulated counterpart, but for one knee of an asymmetric, lightly penalised
  policy (effort and slip penalties reached full strength only at its last step), and the front pitch joints join it.
  These loads describe this policy, not what a better-trained gait or the model-based walk (D10) would need.

## At the 0.25 m/s target (5 episodes, delay 10 ms, randomisation on)

| Joint | RMS / continuous rating, mean (worst) | At nominal limit, mean (worst) | At the episode's limit, mean |
|---|---|---|---|
| Fore pitch, left / right | 1.02 (1.09) / 1.02 (1.11) | 0 / 0 | 9.5 % / 5.4 % |
| Front elbow (X6), left / right | 0.94 (0.97) / 0.96 (0.99) | 0 / 0 | 4.3 % / 2.4 % |
| Hind knee (X4), left / right | **1.10 (1.15)** / 0.92 (0.94) | 0.9 % (2.8 %) / 0 | 2.9 % / 4.3 % |
| Hind pitch, left / right | 0.83 (0.93) / 0.66 (0.77) | 0 / 0 | 1.1 % / 0.9 % |
| Roll, four joints | 0.47-0.71 (up to 0.78) | 0-0.1 % | 0-0.3 % |

Continuous ratings at the joint: X4 roll and pitch 7.5 N·m, hind knee 6.75 N·m, X6 elbow 10.8 N·m. Peak speeds are
within the budget's allowances (pitch 66 of 83 rpm, elbow 79 of 108, knee 67 of 83).

| | Left fore | Right fore | Left hind | Right hind |
|---|---|---|---|---|
| Duty factor | 0.39 | 0.57 | 0.63 | 0.44 |
| Slip ratio (mean of episodes) | 0.24 | 0.23 | 0.26 | 0.23 |
| Stance slip speed, 95th percentile (worst episode) | 0.59 m/s | 0.35 m/s | 0.32 m/s | 0.43 m/s |

Feet down: two diagonal 77 %, three 9 %, one 8 %, four 2 %, other pairs 3 %, none 0.6 %. Body roll 1.2° RMS, pitch
1.1° RMS.

## Other cases (5 episodes each)

| Case | Speed | Falls | Sideways drift, worst | Worst heat ratio (joint) | Worst slip ratio |
|---|---|---|---|---|---|
| 0.15 m/s command | 0.145 m/s | 0 | 0.15 m | 1.04 (left fore pitch) | 0.37 |
| Stiffer contact (impratio 10), 0.25 m/s | 0.238 m/s | 0 | 0.21 m | 1.17 (left hind knee) | 0.29 |
| Roll-ankle model (unseen), 0.25 m/s | 0.236 m/s | 0 | 0.43 m | 1.18 (left hind knee) | 0.39 |
| Nominal model, no randomisation or noise | 0.240 m/s | 0 | 0.07 m | 1.21 (left hind knee) | 0.24 |

The contact-sensitivity flag is not raised (stiffer contact: the same 0.238 m/s, no falls).

## Footfalls and rhythm (diagnostic added after the first results)

The plan's touchdown count, 13-21 per foot per second, does not count steps. The review traced the extra
"touchdowns" to the contact model: within a stance the sole briefly lifts by a median 0.03 mm (at most 0.38 mm) or
keeps a contact carrying about 0 N, for a few milliseconds. These breaks are 1.6-3.1 % of the window. They are a
solver-scale rebound, not a feature of the gait. Real swings (air gaps) last 137-217 ms (90th percentile). Most joints
and every foot load peak at 2.39 Hz; the right fore roll joint (and once the left hind roll) peaks at 1.5 Hz, the phase
clock's frequency, so the clock is not wholly ignored. Stride and touchdown measures should debounce contact (merge
gaps under about 10 ms).

## Stills

`results/gait_pipe1_side_t6.0.jpg`, `_t6.4.jpg` and the overview pair: one 0.25 m/s episode (seed 100000) at 6.0 and
6.4 s, camera following the body, labelled "SIMULATION".

## What to change before the next training run (proposals)

1. **Make heat and torque-limit time cost something:** a penalty on each joint's torque relative to its continuous
   rating, and on time at the (randomised) torque limit; effort and slip penalties at full strength much sooner.
2. **Reward real steps and stronger traction:** a feet air-time reward on debounced contact (so solver rebounds do not
   count) and a stronger slip penalty. Not a penalty on the contact breaks themselves: they are a solver artefact.
3. **Symmetry and straightness:** watch the left-right asymmetry and the sideways drift in the next run's evaluations.
4. **Train on the roll-ankle foot too** (the direction for D5; this policy already walks on it without falls).
5. **A longer run with the stop and resume test**, on a reviewed commit.

The hardware question stays open until a gait trained with these costs, and the model-based walk (task 5), are
measured: if they still exceed the X4's continuous rating on the hind knees or front pitch joints, the options in F12
apply (a 1.2:1 crank ratio, X6 drives on those joints, or a lighter build).

## Changes after a result (from the independent review)

- **Saturation:** the first version measured the time at the limit against each episode's randomised limit, though the
  plan names the model's nominal limit. Both are now reported; the flags use the plan's (nominal) definition. The
  first version's group figures (3.7-5.3 %) were the randomised-limit ones.
- **Heat is now reported per joint** (mean and worst over episodes), not only as each group's worst joint.
- **Added:** heading change and sideways drift (the plan listed them; the first version computed drift but did not
  report it, and did not compute heading change); a Markdown output (`results/gait_pipe1_2026-11-19.md`), which the
  plan's outputs promised; the hashes of the environment and model files used.
- **Environment:** the script now imports the environment from a copy of `rl/` exported from the run's commit
  (`TRIKE_ENV_DIR`), because `rl/trike_env.py` is being changed for the next run; its hash equals the run commit's.
- **Reading:** "contact chatter" is a solver-scale rebound (above); "every joint at 2.39 Hz" was too strong.
