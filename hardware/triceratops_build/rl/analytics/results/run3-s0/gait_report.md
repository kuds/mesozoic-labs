# Gait quality of trike60-run3-s0 (SIMULATION)

Written by `rl/analytics/gait_report.py` on 2026-11-23. Policy `trike60_10002432_steps.zip` (10,002,432 steps; sha256 f76ac23bfda4...), trained at commit efd1e1c7. Deterministic actions, servo delay 10 ms, seeds 100000-100004, randomisation and observation noise on unless stated, 20 s episodes measured from t = 2 s. One policy: every figure is reported, not judged. Definitions: `rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.

## Answer first

- **Speed:** 0.241 m/s at the 0.25 m/s command and 0.144 m/s at 0.15; falls 0 + 0 of 10 (sweep: 0 falls in 21 episodes).
- **Gait:** trot (trot); a diagonal gait whose RF+LH pair follows the LF+RH pair after 0.53 of a stride (0.5 for an even trot); stride 0.70 Hz, 35.4 cm per stride; phase after LF: RF 0.54, LH 0.54, RH 0.03 of a stride.
- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies outside it 64.9 % of the time; the centre of pressure is within 5 mm of its edge 35 % of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives 66 %.)
- **Drives:** worst RMS torque / continuous rating 0.84 (mean of episodes; left_hind_flexion).
- **Power:** 31 W joint mechanical (sum |torque x speed|); leg electrical estimate 54 W on a shared bus to 67 W per drive with no reuse of braking energy; about 72-84 min walking on one battery set with a 30 W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.
- **Steadiness:** body roll 2.6 deg and pitch 2.6 deg SD, height 36 mm range; head bob 75 mm and sway 52 mm (1-99 % range) at 1.4 Hz; head acceleration 5.0 m/s^2 RMS to 20 Hz, of which 2.8 m/s^2 at or below 5 Hz (the visible bob, 0.29 g) and 4.1 m/s^2 at 5-20 Hz (the step impacts' shake).

## Cases (5 episodes each)

| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W | CoM outside support (step contact) | Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.241 | 0 | trot | 0.70 | 35.4 | 30.9 | 54-67 | 64.9 % | 53 % | 52 |
| 0.15 m/s, randomised | 0.144 | 0 | other | 0.86 | 34.5 | 17.6 | 45-53 | 50.7 % | 79 % | 135 |
| 0.25 m/s, nominal model, observation noise off | 0.242 | 0 | trot | 0.70 | 34.4 | 30.1 | 52-66 | 68.3 % | 12 % | 38 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.243 | 0 | trot | 0.73 | 33.6 | 32.8 | 62-76 | 2.1 % | 74 % | 48 |

Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet shuffle, and the 50 ms step rule splits them less cleanly (README).

## a. Footfalls (`footfall.png`)

Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Duty factor: steps / env debounce / raw | 0.54 / 0.54 / 0.54 | 0.49 / 0.48 / 0.47 | 0.51 / 0.50 / 0.49 | 0.58 / 0.58 / 0.57 |
| Steps per s | 0.70 | 0.66 | 0.73 | 1.13 |
| Touchdowns per s, env debounce (training evaluation's count) | 0.77 | 0.81 | 2.08 | 1.26 |
| Stride frequency, Hz (median period) | 0.68 | 0.67 | 0.69 | 0.88 |
| Stance, ms (median) | 826 | 720 | 746 | 764 |
| Swing, ms (median) | 666 | 778 | 698 | 380 |
| Phase after LF, cycles (R) | 0 | 0.54 (0.97) | 0.54 (0.84) | 0.03 (1.00) |

Gait class: **trot** (trot; largest phase error to the nearest template 0.04 cycles; tolerance 0.125). Per episode: trot, trot, trot, trot, trot. Diagonal timing: Hildebrand limb phase left 0.46, right 0.51 (trot 0.5); each fore foot lands -0.03 (LF after RH) and -0.01 (RF after LH) cycles after its diagonal hind partner; the RF+LH pair follows the LF+RH pair after 0.53 of a stride (0.5 when even). Stride length 35.4 cm (LF touchdown to LF touchdown). Irregular strides (period more than 25 % off the median): LF 0 %, RF 4 %, LH 23 %, RH 53 %.

Feet down (steps): 4 0 %, 3 12 %, 2 diagonal 85 %, 2 same side 3 %, 2 front 0 %, 2 hind 0 %, 1 0 %, 0 0 %.

Symmetry index (Robinson, %, 0 = symmetric): duty factor: fore 11, hind 13, sides 1; stance duration: fore 13, hind 23, sides 18; stance length: fore 13, hind 24, sides 17; load: fore 7, hind 20, sides 6.

## b. Weight distribution (`weight_distribution.png`)

Floor-truth sole normal forces. Load share LF 27 %, RF 26 %, LH 21 %, RH 26 %: front 53 % / rear 47 %, left 49 % / right 51 %. Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: LF 1.83 / 1.02, RF 3.08 / 1.18, LH 2.07 / 1.24, RH 2.06 / 0.97. 58 strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least 100.0 % of samples.

## c. Balance (`balance.png`)

Support polygon: hull of the capsule soles' contact-line ends of the feet in stance on the step contact the footfalls use (the env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. CoM ground projection outside the polygon 64.9 % of the time (by feet in stance: 1: - %, 2: 72 %, 3: 13 %, 4: 0 %; time with one foot in stance 0.1 %); CoM margin median -6 mm. CoP outside: 0.00 % (it cannot leave the polygon by construction; a check); CoP within 5 mm of the edge 35 %, median margin 7 mm. No foot in stance 0.0 % of the time.

| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |
|---|---|---|---|
| step contact (env 10 ms debounce, then gaps and blips under 50 ms merged) | 64.9 % | 0.1 % | 35 % |
| env 10 ms debounce | 65.6 % | 0.6 % | 35 % |
| raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts) | 66.3 % | 2.3 % | 35 % |

Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot 'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the contact model, not the gait. Nominal model: step contact 68.3 %, env 10 ms debounce 68.1 %, raw 68.1 %.

## d. Drives (`drive_envelopes.png`, `drive_loads.png`)

| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |
|---|---|---|---|---|---|
| left_fore_roll | 0.27 (0.30) | 0.0 % (0.0 %) | 17.8 | 6 | 0.2 % |
| left_fore_pitch | 0.73 (0.76) | 0.0 % (0.0 %) | 20.5 | 75 | 6.0 % |
| left_fore_flexion | 0.53 (0.57) | 0.0 % (0.0 %) | 36.9 | 58 | 1.4 % |
| right_fore_roll | 0.42 (0.46) | 0.0 % (0.0 %) | 19.0 | 12 | 0.9 % |
| right_fore_pitch | 0.82 (0.87) | 0.0 % (0.0 %) | 20.4 | 79 | 9.0 % |
| right_fore_flexion | 0.66 (0.68) | 0.0 % (0.0 %) | 37.1 | 91 | 2.6 % |
| left_hind_roll | 0.49 (0.52) | 0.0 % (0.0 %) | 20.8 | 11 | 1.2 % |
| left_hind_pitch | 0.58 (0.60) | 0.0 % (0.0 %) | 19.6 | 39 | 3.4 % |
| left_hind_flexion | 0.84 (0.88) | 0.4 % (1.2 %) | 18.8 | 89 | 11.1 % |
| right_hind_roll | 0.36 (0.37) | 0.0 % (0.0 %) | 18.7 | 9 | 0.4 % |
| right_hind_pitch | 0.73 (0.80) | 0.0 % (0.0 %) | 20.4 | 77 | 7.4 % |
| right_hind_flexion | 0.70 (0.76) | 0.0 % (0.0 %) | 18.1 | 41 | 4.9 % |

Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): roll: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; pitch: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; front elbow: model limit 37.8, maker peak 54.0, maker rated 18.00, continuous 10.80, speed allowance 108 rpm; hind knee: model limit 18.9, maker peak 27.0, maker rated 9.45, continuous 6.75, speed allowance 83 rpm.

## e. Power and energy (`power.png`)

| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W | CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive | Run time, per drive, 25 / 40 W hotel |
|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 30.9 | 15.9 | 50.8 | 53.7 / 67.4 | 1.04 | 2.81 / 3.27 | 84 / 72 | 76 / 66 |
| 0.15 m/s, randomised | 17.6 | 9.4 | 43.0 | 45.2 / 52.8 | 0.99 | 4.23 / 4.66 | 94 / 85 | 91 / 76 |
| 0.25 m/s, nominal model, observation noise off | 30.1 | 15.2 | 49.7 | 52.0 / 65.6 | 1.00 | 2.72 / 3.18 | 86 / 74 | 78 / 67 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 32.8 | 17.5 | 57.4 | 61.7 / 75.8 | 1.09 | 3.06 / 3.53 | 77 / 67 | 70 / 61 |

Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is 3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step. Not included: the head and tail servos. Hotel load 30 W (range 25-40 W) is an assumption, not a parts-list figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC losses. Battery 146.5 Wh x 80% usable. Torque (copper loss) dominates the electrical figures.

## f. Body and head steadiness (`steadiness.png`)

| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 2.59 | 2.65 | 35.7 | 74.9 | 51.8 | 4.95 | 2.81 | 4.08 | 25.9 (8.6) |
| 0.15 m/s, randomised | 3.29 | 2.89 | 27.1 | 55.3 | 36.0 | 3.12 | 1.70 | 2.61 | 21.0 (6.4) |
| 0.25 m/s, nominal model, observation noise off | 1.96 | 2.80 | 35.3 | 71.9 | 43.3 | 4.89 | 2.81 | 4.01 | 24.3 (9.1) |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 2.21 | 2.80 | 32.5 | 78.2 | 52.0 | 4.88 | 3.19 | 3.69 | 27.7 (8.9) |

Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at 20 Hz, gravity excluded, and split at 5 Hz: at or below it the visible bob and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range after removing a 1 s moving average.

## g. Feet (`feet.png`)

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Swing clearance, mm, median (10th pct) | 80.9 (71.0) | 139.8 (117.1) | 17.6 (8.5) | 27.2 (4.4) |
| Swings under 5 mm | 0 % | 0 % | 3 % | 13 % |
| Touchdown speed, m/s, median (95th) | 0.59 (0.67) | 0.82 (0.94) | 0.61 (0.97) | 0.32 (0.59) |
| Peak load rate, body weights/s, median (95th) | 129 (186) | 143 (241) | 220 (290) | 83 (225) |
| Stance slip speed at the sole's origin, mean / 95th pct worst, m/s | 0.03 / 0.11 | 0.05 / 0.24 | 0.07 / 0.38 | 0.05 / 0.26 |
| Slip ratio at the sole's origin, mean (worst) | 0.06 (0.06) | 0.10 (0.12) | 0.15 (0.18) | 0.12 (0.13) |
| Slip ratio at the sole's contacts, mean (worst) | 0.04 (0.04) | 0.08 (0.10) | 0.11 (0.14) | 0.07 (0.08) |

## h. Speed tracking (`speed_tracking.png`)

| Command, m/s | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.27 |
|---|---|---|---|---|---|---|---|
| Speed, m/s (mean of 3 seeds) | -0.000 | 0.044 | 0.093 | 0.144 | 0.194 | 0.242 | 0.260 |
| Falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At 0.25 m/s: mean |step speed - command| 0.015 m/s (control-step speeds swing with the stride, SD 0.016 m/s); |mean speed - command| 0.009 m/s. Sideways drift worst -0.37 m, heading change worst +12.5 deg over 18 s.

## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`

Environment: `hardware/triceratops_build/rl/trike_env.py` (sha256 402819fcc1c3...; the run commit's file: -...).

Not applicable: not the policy gait_pipe1.py analysed.

Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: True.

## What this does not show

Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the hotel load is assumed. Head acceleration is the head body's, not a camera's or a visitor's view of it. Charts: `footfall.png`, `weight_distribution.png`, `balance.png`, `drive_envelopes.png`, `drive_loads.png`, `power.png`, `steadiness.png`, `feet.png`, `speed_tracking.png`.
