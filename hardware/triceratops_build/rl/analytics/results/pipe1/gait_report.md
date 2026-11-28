# Gait quality of trike60-pipe1 (SIMULATION)

Written by `rl/analytics/gait_report.py` on 2026-11-19. Policy `trike60_2007040_steps.zip` (2,007,040 steps; sha256 dc06d30105b2...), trained at commit 6aa3a679. Deterministic actions, servo delay 10 ms, seeds 100000-100004, randomisation and observation noise on unless stated, 20 s episodes measured from t = 2 s. One policy: every figure is reported, not judged. Definitions: `rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.

## Answer first

- **Speed:** 0.238 m/s at the 0.25 m/s command and 0.145 m/s at 0.15; falls 0 + 0 of 10 (sweep: 0 falls in 21 episodes).
- **Gait:** other (nearest trot, 0.13 cycles off); a diagonal gait whose RF+LH pair follows the LF+RH pair after 0.42 of a stride (0.5 for an even trot); stride 2.39 Hz, 10.0 cm per stride; phase after LF: RF 0.43, LH 0.37, RH 0.95 of a stride.
- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies outside it 1.6 % of the time; the centre of pressure is within 5 mm of its edge 7 % of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives 12 %.)
- **Drives:** worst RMS torque / continuous rating 1.10 (mean of episodes; left_hind_flexion).
- **Power:** 42 W joint mechanical (sum |torque x speed|); leg electrical estimate 106 W on a shared bus to 123 W per drive with no reuse of braking energy; about 46-52 min walking on one battery set with a 30 W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.
- **Steadiness:** body roll 0.8 deg and pitch 1.0 deg SD, height 17 mm range; head bob 42 mm and sway 28 mm (1-99 % range) at 2.4 Hz; head acceleration 11.1 m/s^2 RMS to 20 Hz, of which 5.9 m/s^2 at or below 5 Hz (the visible bob, 0.61 g) and 9.4 m/s^2 at 5-20 Hz (the step impacts' shake).

## Cases (5 episodes each)

| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W | CoM outside support (step contact) | Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.238 | 0 | other | 2.39 | 10.0 | 42.4 | 106-123 | 1.6 % | 0 % | 15 |
| 0.15 m/s, randomised | 0.145 | 0 | other | 2.63 | 5.3 | 28.3 | 85-97 | 1.1 % | 27 % | 196 |
| 0.25 m/s, nominal model, observation noise off | 0.240 | 0 | trot | 2.36 | 10.2 | 40.0 | 115-131 | 0.5 % | 0 % | 0 |
| 0.25 m/s, roll-ankle model, randomised | 0.236 | 0 | other | 2.51 | 9.3 | 43.5 | 110-128 | 6.0 % | 8 % | 135 |

Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet shuffle, and the 50 ms step rule splits them less cleanly (README).

## a. Footfalls (`footfall.png`)

Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Duty factor: steps / env debounce / raw | 0.44 / 0.43 / 0.39 | 0.60 / 0.60 / 0.57 | 0.66 / 0.65 / 0.63 | 0.48 / 0.47 / 0.44 |
| Steps per s | 2.39 | 2.39 | 2.38 | 2.40 |
| Touchdowns per s, env debounce (training evaluation's count) | 3.29 | 2.79 | 3.10 | 2.99 |
| Stride frequency, Hz (median period) | 2.38 | 2.40 | 2.39 | 2.39 |
| Stance, ms (median) | 182 | 251 | 275 | 202 |
| Swing, ms (median) | 233 | 166 | 147 | 215 |
| Phase after LF, cycles (R) | 0 | 0.43 (0.98) | 0.37 (0.96) | 0.95 (0.96) |

Gait class: **other** (nearest trot, 0.13 cycles off; largest phase error to the nearest template 0.13 cycles; tolerance 0.125). Per episode: other, other, other, other, trot. Diagonal timing: Hildebrand limb phase left 0.63, right 0.48 (trot 0.5); each fore foot lands +0.05 (LF after RH) and +0.06 (RF after LH) cycles after its diagonal hind partner; the RF+LH pair follows the LF+RH pair after 0.42 of a stride (0.5 when even). Stride length 10.0 cm (LF touchdown to LF touchdown). Irregular strides (period more than 25 % off the median): LF 0 %, RF 0 %, LH 0 %, RH 0 %.

Feet down (steps): 4 3 %, 3 13 %, 2 diagonal 82 %, 2 same side 1 %, 2 front 0 %, 2 hind 0 %, 1 0 %, 0 0 %.

Symmetry index (Robinson, %, 0 = symmetric): duty factor: fore 32, hind 32, sides 1; stance duration: fore 31, hind 31, sides 2; stance length: fore 31, hind 32, sides 2; load: fore 48, hind 40, sides 9.

## b. Weight distribution (`weight_distribution.png`)

Floor-truth sole normal forces. Load share LF 21 %, RF 34 %, LH 27 %, RH 18 %: front 56 % / rear 44 %, left 48 % / right 52 %. Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: LF 3.07 / 1.87, RF 3.33 / 2.22, LH 2.13 / 1.37, RH 2.11 / 1.33. 210 strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least 100.0 % of samples.

## c. Balance (`balance.png`)

Support polygon: hull of the sole corners of the feet in stance on the step contact the footfalls use (the env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. CoM ground projection outside the polygon 1.6 % of the time (by feet in stance: 1: 100 %, 2: 2 %, 3: 0 %, 4: 0 %; time with one foot in stance 0.2 %); CoM margin median 34 mm. CoP outside: 0.00 % (it cannot leave the polygon by construction; a check); CoP within 5 mm of the edge 7 %, median margin 31 mm. No foot in stance 0.0 % of the time.

| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |
|---|---|---|---|
| step contact (env 10 ms debounce, then gaps and blips under 50 ms merged) | 1.6 % | 0.2 % | 7 % |
| env 10 ms debounce | 3.6 % | 1.4 % | 7 % |
| raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts) | 11.5 % | 8.4 % | 9 % |

Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot 'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the contact model, not the gait. Nominal model: step contact 0.5 %, env 10 ms debounce 0.6 %, raw 5.0 %.

## d. Drives (`drive_envelopes.png`, `drive_loads.png`)

| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |
|---|---|---|---|---|---|
| left_fore_roll | 0.47 (0.52) | 0.0 % (0.0 %) | 18.2 | 7 | 1.7 % |
| left_fore_pitch | 1.02 (1.09) | 0.0 % (0.0 %) | 20.5 | 49 | 15.3 % |
| left_fore_flexion | 0.94 (0.97) | 0.0 % (0.0 %) | 36.9 | 79 | 9.6 % |
| right_fore_roll | 0.59 (0.65) | 0.0 % (0.0 %) | 16.2 | 6 | 1.4 % |
| right_fore_pitch | 1.02 (1.11) | 0.0 % (0.0 %) | 20.4 | 66 | 15.6 % |
| right_fore_flexion | 0.96 (0.99) | 0.0 % (0.0 %) | 37.1 | 63 | 7.8 % |
| left_hind_roll | 0.71 (0.78) | 0.0 % (0.0 %) | 20.8 | 6 | 4.9 % |
| left_hind_pitch | 0.83 (0.93) | 0.0 % (0.0 %) | 19.6 | 33 | 10.0 % |
| left_hind_flexion | 1.10 (1.15) | 0.9 % (2.8 %) | 18.8 | 67 | 22.8 % |
| right_hind_roll | 0.58 (0.63) | 0.1 % (0.2 %) | 20.9 | 11 | 5.3 % |
| right_hind_pitch | 0.66 (0.77) | 0.0 % (0.0 %) | 20.4 | 31 | 6.5 % |
| right_hind_flexion | 0.92 (0.94) | 0.0 % (0.0 %) | 18.1 | 45 | 14.7 % |

Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): roll: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; pitch: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; front elbow: model limit 37.8, maker peak 54.0, maker rated 18.00, continuous 10.80, speed allowance 108 rpm; hind knee: model limit 18.9, maker peak 27.0, maker rated 9.45, continuous 6.75, speed allowance 83 rpm.

## e. Power and energy (`power.png`)

| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W | CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive | Run time, per drive, 25 / 40 W hotel |
|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 42.4 | 23.1 | 98.8 | 105.6 / 123.3 | 1.44 | 4.61 / 5.21 | 52 / 46 | 48 / 43 |
| 0.15 m/s, randomised | 28.3 | 15.6 | 80.1 | 84.8 / 96.7 | 1.59 | 6.44 / 7.11 | 62 / 56 | 58 / 52 |
| 0.25 m/s, nominal model, observation noise off | 40.0 | 22.4 | 107.7 | 115.0 / 131.5 | 1.33 | 4.84 / 5.39 | 48 / 44 | 45 / 41 |
| 0.25 m/s, roll-ankle model, randomised | 43.5 | 24.2 | 102.2 | 110.1 / 127.8 | 1.49 | 4.81 / 5.42 | 50 / 45 | 46 / 42 |

Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is 3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step. Not included: the head and tail servos. Hotel load 30 W (range 25-40 W) is an assumption, not a parts-list figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC losses. Battery 146.5 Wh x 80% usable. Torque (copper loss) dominates the electrical figures.

## f. Body and head steadiness (`steadiness.png`)

| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.80 | 1.05 | 16.8 | 41.6 | 28.4 | 11.09 | 5.95 | 9.36 | 39.1 (17.8) |
| 0.15 m/s, randomised | 0.69 | 0.91 | 13.7 | 36.5 | 23.5 | 8.87 | 3.06 | 8.32 | 33.6 (10.0) |
| 0.25 m/s, nominal model, observation noise off | 0.64 | 0.88 | 15.6 | 36.4 | 25.9 | 9.52 | 5.54 | 7.74 | 34.9 (13.3) |
| 0.25 m/s, roll-ankle model, randomised | 1.12 | 1.28 | 16.5 | 48.2 | 25.1 | 12.19 | 4.78 | 11.21 | 40.9 (15.5) |

Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at 20 Hz, gravity excluded, and split at 5 Hz: at or below it the visible bob and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range after removing a 1 s moving average.

## g. Feet (`feet.png`)

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Swing clearance, mm, median (10th pct) | 26.1 (19.3) | 22.1 (15.6) | 13.2 (7.6) | 11.3 (6.8) |
| Swings under 5 mm | 0 % | 0 % | 2 % | 4 % |
| Touchdown speed, m/s, median (95th) | 0.68 (1.13) | 0.68 (0.93) | 0.25 (0.43) | 0.18 (0.35) |
| Peak load rate, body weights/s, median (95th) | 286 (409) | 349 (459) | 163 (296) | 157 (306) |
| Stance slip speed, mean / 95th pct worst, m/s | 0.15 / 0.59 | 0.09 / 0.35 | 0.10 / 0.32 | 0.13 / 0.43 |
| Slip ratio, mean (worst) | 0.24 (0.26) | 0.23 (0.25) | 0.26 (0.29) | 0.23 (0.24) |
| Roll-ankle angle, deg, min / max (roll-ankle model; limit +-20) | -1.1 / +4.6 | -3.0 / +2.8 | -3.0 / +4.3 | -3.2 / +2.9 |

## h. Speed tracking (`speed_tracking.png`)

| Command, m/s | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.27 |
|---|---|---|---|---|---|---|---|
| Speed, m/s (mean of 3 seeds) | 0.003 | 0.053 | 0.100 | 0.144 | 0.192 | 0.238 | 0.256 |
| Falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At 0.25 m/s: mean |step speed - command| 0.018 m/s (control-step speeds swing with the stride, SD 0.020 m/s); |mean speed - command| 0.012 m/s. Sideways drift worst +0.23 m, heading change worst +5.3 deg over 18 s.

## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`

Environment: `hardware/triceratops_build/rl/trike_env.py` (sha256 a82a0c4d206f...; the run commit's file: 4ffec806e794...). This tool's log, passed through `gait_pipe1.measures`, against the committed per-episode figures:

| Case | Episodes | Speed now / then, m/s | Power now / then, W | Largest difference | Identical |
|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 5 | 0.2385 / 0.2385 | 42.40 / 42.40 | pitch_rms_deg 2.2e-16 | True |
| 0.15 m/s, randomised | 5 | 0.1446 / 0.1446 | 28.26 / 28.26 | pitch_rms_deg 1.1e-16 | True |
| 0.25 m/s, nominal model, observation noise off | 5 | 0.2405 / 0.2405 | 39.97 / 39.97 | pitch_rms_deg 1.1e-16 | True |
| 0.25 m/s, roll-ankle model, randomised | 5 | 0.2362 / 0.2362 | 43.49 / 43.49 | forward_speed_mps 0.0e+00 | True |

Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: True.

## What this does not show

Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the hotel load is assumed. Head acceleration is the head body's, not a camera's or an onlooker's view of it. Charts: `footfall.png`, `weight_distribution.png`, `balance.png`, `drive_envelopes.png`, `drive_loads.png`, `power.png`, `steadiness.png`, `feet.png`, `speed_tracking.png`.
