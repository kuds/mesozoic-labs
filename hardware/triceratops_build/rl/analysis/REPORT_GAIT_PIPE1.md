# The gait of the trike60-pipe1 policy (simulation)

19 Nov 2026. Analysis of the owner's Colab pipeline-test policy (2,007,040 steps, PR #669's foot), per
`PLAN_GAIT_PIPE1.md` (committed before the analysis). One policy, so every figure is reported, not a verdict.
Simulation of a model whose masses are estimates; nothing is measured on hardware.

## Answer first

- **It walks, and robustly.** At the 0.25 m/s command it holds 0.238 m/s with a quick diagonal, trot-like gait: every
  leg joint and every foot load cycles at **2.39 Hz** (a 0.42 s stride, about 10 cm per stride), with a diagonal pair
  of feet down 77 % of the time. **No falls in 25 episodes**, including the stiffer contact setting, the roll-ankle
  foot it never trained on, and the nominal model.
- **But it is not yet a gait the robot should use.** It drives the joints hard and the feet slide:
  - **Heat:** the RMS torque is above the drives' continuous rating on the **pitch joints (1.11 ×)** and the **hind
    knees (1.15 ×)**; the front elbows are at 0.99 ×.
  - **Torque limits:** the pitch, front-elbow and hind-knee drives sit at their limit in 4-5 % of steps (bang-bang
    control).
  - **Slip:** each loaded foot slides about a quarter of the distance the body travels (slip ratio 0.23-0.29).
  - **Contact chatter:** within each stance the foot's contact breaks for about 2 ms many times a second; real swings
    last 140-215 ms.
  - **Power:** 42 W at 0.24 m/s, a cost of transport of 1.44.
- **What it means.** The pipeline works and a first gait exists. The drive loads echo the budget's hind-knee warning
  (F12: 1.00 × by estimate; this gait 1.15 ×) and add the pitch joints. This policy trained with weak effort penalties
  (energy and slip ramped up only to full strength at its last step), so these loads describe this policy, not what a
  better-trained gait or the model-based walk (D10) would need.

## Results at the 0.25 m/s target (5 episodes, delay 10 ms, randomisation on; `results/gait_pipe1_2026-11-19.json`)

| Drive group | RMS torque, worst joint | Continuous rating | Ratio | Peak / limit | At limit | Peak speed / allowance |
|---|---|---|---|---|---|---|
| Roll (X4) | 5.88 N·m | 7.5 | 0.78 | 20.9 / 21 | 0.2 % | 11 / 83 rpm |
| Pitch (X4) | 8.36 N·m | 7.5 | **1.11** | 20.5 / 21 | **5.3 %** | 66 / 83 rpm |
| Front elbow (X6) | 10.72 N·m | 10.8 | 0.99 | 37.1 / 37.8 | **3.7 %** | 79 / 108 rpm |
| Hind knee (X4) | 7.78 N·m | 6.75 | **1.15** | 18.8 / 18.9 | **4.1 %** | 67 / 83 rpm |

| | Left fore | Right fore | Left hind | Right hind |
|---|---|---|---|---|
| Duty factor | 0.39 | 0.57 | 0.63 | 0.44 |
| Slip ratio (mean of episodes) | 0.24 | 0.23 | 0.26 | 0.23 |
| Stance slip speed, 95th percentile (worst episode) | 0.59 m/s | 0.35 m/s | 0.32 m/s | 0.43 m/s |

Feet down: two diagonal 77 %, three 9 %, one 8 %, four 2 %, other pairs 3 %, none 0.6 %. Body roll 1.2° RMS, pitch
1.1° RMS. The gait is asymmetric (right fore and left hind carry longer stances).

## Other cases (5 episodes each)

| Case | Speed | Falls | Pitch RMS ratio | Hind knee RMS ratio | Worst slip ratio |
|---|---|---|---|---|---|
| 0.15 m/s command | 0.145 m/s | 0 | 1.04 | 1.02 | 0.37 |
| Stiffer contact (impratio 10), 0.25 m/s | 0.238 m/s | 0 | 1.14 | 1.17 | 0.29 |
| Roll-ankle model (unseen), 0.25 m/s | 0.236 m/s | 0 | 1.14 | 1.18 | 0.39 |
| Nominal model, no randomisation or noise | 0.240 m/s | 0 | 1.09 | 1.21 | 0.24 |

The contact-sensitivity flag is not raised: with stiffer contact the speed is unchanged (0.238 m/s) and nothing falls.
The front elbow's RMS reaches 1.03 × its rating on the roll-ankle model.

## Stance and frequency (diagnostic added after the first results; `gait_freq_pipe1.py`)

The plan's touchdown count, 13-21 per foot per second, turned out to count contact chatter, so it is not a stride
rate. The added diagnostic (seeds 100000-100002, `results/gait_freq_pipe1_2026-11-19.json`) finds:
- **Stance segments:** median 8-27 ms, separated by air gaps of median 2 ms.
- **Real swings:** 90th-percentile air gaps of 137-217 ms.
- **Rhythm:** every joint's motion and every foot's load peaks at 2.39 Hz, with some roll joints at 1.5 Hz, the
  phase clock's frequency.

Labelled "changed after a result": it was not in the plan.

## Stills

`results/gait_pipe1_side_t6.0.jpg`, `_t6.4.jpg` and the overview pair: one 0.25 m/s episode (seed 100000) at 6.0 and
6.4 s, camera following the body, labelled "SIMULATION".

## What to change before the next training run (proposals)

1. **Make heat and saturation cost something.** Penalise each joint's torque relative to its continuous rating, and
   time spent at the torque limit; apply the effort and slip penalties at full strength sooner.
2. **Reward clean steps.** Penalise contact chatter (contact changes within a stance), require some foot clearance in
   swing, and strengthen the slip penalty.
3. **Gait rhythm.** The policy ignores the 1.5 Hz phase clock and steps at 2.4 Hz; either match the clock to a chosen
   stride frequency or drop it.
4. **Train on the roll-ankle foot too** (the direction for D5; this policy already walks on it without falls).
5. **A longer run with the stop and resume test**, after the two remaining fixes to the checkpoint handling (R1, R2).

The hardware question stays open until a gait trained with these costs, and the model-based walk (task 5), are
measured: if they still exceed the X4's continuous rating on the hind knees or pitch joints, the options in F12 apply
(a 1.2:1 crank ratio, X6 drives on those joints, or a lighter build).
