# Gait quality of trike60-run2b-ankle (SIMULATION)

Written by `rl/analytics/gait_report.py` on 2026-11-19. Policy `trike60_8003584_steps.zip` (8,003,584 steps; sha256 00ed354692da...), trained at commit e92fd9ef. Deterministic actions, servo delay 10 ms, seeds 100000-100004, randomisation and observation noise on unless stated, 20 s episodes measured from t = 2 s. One policy: every figure is reported, not judged. Definitions: `rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.

## Answer first

- **Speed:** 0.245 m/s at the 0.25 m/s command and 0.146 m/s at 0.15; falls 0 + 0 of 10 (sweep: 0 falls in 21 episodes).
- **Gait:** trot (trot); a diagonal gait whose RF+LH pair follows the LF+RH pair after 0.42 of a stride (0.5 for an even trot); stride 1.22 Hz, 20.0 cm per stride; phase after LF: RF 0.43, LH 0.44, RH 0.03 of a stride.
- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies outside it 0.2 % of the time; the centre of pressure is within 5 mm of its edge 0 % of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives 1 %.)
- **Drives:** worst RMS torque / continuous rating 0.72 (mean of episodes; left_hind_flexion).
- **Power:** 22 W joint mechanical (sum |torque x speed|); leg electrical estimate 40 W on a shared bus to 50 W per drive with no reuse of braking energy; about 88-101 min walking on one battery set with a 30 W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.
- **Steadiness:** body roll 1.4 deg and pitch 0.9 deg SD, height 28 mm range; head bob 42 mm and sway 29 mm (1-99 % range) at 1.7 Hz; head acceleration 5.1 m/s^2 RMS to 20 Hz, of which 2.7 m/s^2 at or below 5 Hz (the visible bob, 0.27 g) and 4.4 m/s^2 at 5-20 Hz (the step impacts' shake).

## Cases (5 episodes each)

| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W | CoM outside support (step contact) | Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.245 | 0 | trot | 1.22 | 20.0 | 22.1 | 40-50 | 0.2 % | 0 % | 2 |
| 0.15 m/s, randomised | 0.146 | 0 | other | 0.97 | 15.4 | 12.9 | 28-34 | 0.0 % | 51 % | 6 |
| 0.25 m/s, nominal model, observation noise off | 0.247 | 0 | trot | 1.26 | 19.6 | 20.8 | 38-48 | 0.2 % | 0 % | 0 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.244 | 0 | trot | 1.25 | 19.0 | 23.5 | 43-54 | 0.3 % | 1 % | 1 |

Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet shuffle, and the 50 ms step rule splits them less cleanly (README).

## a. Footfalls (`footfall.png`)

Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Duty factor: steps / env debounce / raw | 0.46 / 0.46 / 0.45 | 0.60 / 0.60 / 0.60 | 0.63 / 0.63 / 0.61 | 0.44 / 0.44 / 0.43 |
| Steps per s | 1.22 | 1.21 | 1.21 | 1.22 |
| Touchdowns per s, env debounce (training evaluation's count) | 1.24 | 1.23 | 1.53 | 1.63 |
| Stride frequency, Hz (median period) | 1.22 | 1.22 | 1.22 | 1.22 |
| Stance, ms (median) | 376 | 491 | 517 | 358 |
| Swing, ms (median) | 442 | 323 | 303 | 458 |
| Phase after LF, cycles (R) | 0 | 0.43 (0.99) | 0.44 (0.99) | 0.03 (1.00) |

Gait class: **trot** (trot; largest phase error to the nearest template 0.07 cycles; tolerance 0.125). Per episode: trot, trot, trot, trot, trot. Diagonal timing: Hildebrand limb phase left 0.56, right 0.40 (trot 0.5); each fore foot lands -0.03 (LF after RH) and -0.01 (RF after LH) cycles after its diagonal hind partner; the RF+LH pair follows the LF+RH pair after 0.42 of a stride (0.5 when even). Stride length 20.0 cm (LF touchdown to LF touchdown). Irregular strides (period more than 25 % off the median): LF 0 %, RF 0 %, LH 0 %, RH 0 %.

Feet down (steps): 4 2 %, 3 9 %, 2 diagonal 89 %, 2 same side 0 %, 2 front 0 %, 2 hind 0 %, 1 0 %, 0 0 %.

Symmetry index (Robinson, %, 0 = symmetric): duty factor: fore 27, hind 35, sides 4; stance duration: fore 27, hind 35, sides 4; stance length: fore 27, hind 35, sides 4; load: fore 32, hind 37, sides 1.

## b. Weight distribution (`weight_distribution.png`)

Floor-truth sole normal forces. Load share LF 23 %, RF 32 %, LH 27 %, RH 18 %: front 55 % / rear 45 %, left 50 % / right 50 %. Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: LF 2.54 / 1.52, RF 2.20 / 1.44, LH 1.90 / 1.28, RH 2.19 / 1.31. 105 strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least 100.0 % of samples.

## c. Balance (`balance.png`)

Support polygon: hull of the sole corners of the feet in stance on the step contact the footfalls use (the env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. CoM ground projection outside the polygon 0.2 % of the time (by feet in stance: 1: - %, 2: 0 %, 3: 0 %, 4: 0 %; time with one foot in stance 0.0 %); CoM margin median 40 mm. CoP outside: 0.00 % (it cannot leave the polygon by construction; a check); CoP within 5 mm of the edge 0 %, median margin 49 mm. No foot in stance 0.0 % of the time.

| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |
|---|---|---|---|
| step contact (env 10 ms debounce, then gaps and blips under 50 ms merged) | 0.2 % | 0.0 % | 0 % |
| env 10 ms debounce | 0.2 % | 0.0 % | 0 % |
| raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts) | 1.0 % | 0.4 % | 0 % |

Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot 'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the contact model, not the gait. Nominal model: step contact 0.2 %, env 10 ms debounce 0.2 %, raw 0.5 %.

## d. Drives (`drive_envelopes.png`, `drive_loads.png`)

| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |
|---|---|---|---|---|---|
| left_fore_roll | 0.26 (0.27) | 0.0 % (0.0 %) | 13.9 | 6 | 0.1 % |
| left_fore_pitch | 0.52 (0.59) | 0.0 % (0.0 %) | 20.5 | 28 | 2.7 % |
| left_fore_flexion | 0.61 (0.64) | 0.0 % (0.0 %) | 36.9 | 58 | 3.8 % |
| right_fore_roll | 0.22 (0.23) | 0.0 % (0.0 %) | 9.5 | 5 | 0.0 % |
| right_fore_pitch | 0.66 (0.72) | 0.0 % (0.0 %) | 20.4 | 54 | 4.2 % |
| right_fore_flexion | 0.58 (0.59) | 0.0 % (0.0 %) | 37.1 | 53 | 1.8 % |
| left_hind_roll | 0.32 (0.34) | 0.0 % (0.0 %) | 10.8 | 3 | 0.0 % |
| left_hind_pitch | 0.56 (0.62) | 0.0 % (0.0 %) | 19.6 | 36 | 4.0 % |
| left_hind_flexion | 0.72 (0.77) | 0.2 % (0.5 %) | 18.8 | 45 | 3.5 % |
| right_hind_roll | 0.31 (0.32) | 0.0 % (0.0 %) | 18.5 | 5 | 0.6 % |
| right_hind_pitch | 0.61 (0.63) | 0.0 % (0.0 %) | 20.4 | 31 | 5.6 % |
| right_hind_flexion | 0.62 (0.65) | 0.0 % (0.0 %) | 18.1 | 65 | 3.8 % |

Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): roll: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; pitch: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; front elbow: model limit 37.8, maker peak 54.0, maker rated 18.00, continuous 10.80, speed allowance 108 rpm; hind knee: model limit 18.9, maker peak 27.0, maker rated 9.45, continuous 6.75, speed allowance 83 rpm.

## e. Power and energy (`power.png`)

| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W | CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive | Run time, per drive, 25 / 40 W hotel |
|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 22.1 | 11.3 | 38.0 | 39.9 / 49.9 | 0.73 | 2.31 / 2.65 | 101 / 88 | 94 / 78 |
| 0.15 m/s, randomised | 12.9 | 6.6 | 26.9 | 28.0 / 33.9 | 0.71 | 3.21 / 3.54 | 121 / 110 | 119 / 95 |
| 0.25 m/s, nominal model, observation noise off | 20.8 | 10.8 | 36.3 | 38.3 / 47.7 | 0.68 | 2.22 / 2.53 | 103 / 91 | 97 / 80 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 23.5 | 12.1 | 41.0 | 43.1 / 53.8 | 0.78 | 2.43 / 2.79 | 97 / 84 | 90 / 75 |

Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is 3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step. Not included: the head and tail servos. Hotel load 30 W (range 25-40 W) is an assumption, not a parts-list figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC losses. Battery 146.5 Wh x 80% usable. Torque (copper loss) dominates the electrical figures.

## f. Body and head steadiness (`steadiness.png`)

| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 1.36 | 0.87 | 28.2 | 42.4 | 28.8 | 5.11 | 2.66 | 4.36 | 23.3 (6.7) |
| 0.15 m/s, randomised | 1.69 | 1.04 | 23.5 | 35.1 | 29.3 | 3.40 | 1.88 | 2.84 | 18.9 (5.0) |
| 0.25 m/s, nominal model, observation noise off | 1.16 | 0.76 | 26.6 | 39.6 | 27.2 | 4.76 | 2.57 | 4.01 | 20.7 (6.1) |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.69 | 0.74 | 29.4 | 36.2 | 26.7 | 4.87 | 2.68 | 4.07 | 26.5 (7.9) |

Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at 20 Hz, gravity excluded, and split at 5 Hz: at or below it the visible bob and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range after removing a 1 s moving average.

## g. Feet (`feet.png`)

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Swing clearance, mm, median (10th pct) | 31.1 (24.9) | 30.0 (27.1) | 12.9 (10.7) | 16.0 (13.8) |
| Swings under 5 mm | 0 % | 0 % | 0 % | 0 % |
| Touchdown speed, m/s, median (95th) | 0.49 (0.58) | 0.51 (0.58) | 0.25 (0.41) | 0.39 (0.58) |
| Peak load rate, body weights/s, median (95th) | 217 (291) | 236 (292) | 140 (199) | 116 (225) |
| Stance slip speed, mean / 95th pct worst, m/s | 0.03 / 0.17 | 0.02 / 0.12 | 0.03 / 0.23 | 0.04 / 0.24 |
| Slip ratio, mean (worst) | 0.06 (0.07) | 0.05 (0.06) | 0.09 (0.09) | 0.07 (0.08) |
| Roll-ankle angle, deg, min / max (roll-ankle model; limit +-20) | -3.7 / +1.2 | -2.2 / +3.9 | -3.6 / +2.6 | -0.4 / +5.1 |

## h. Speed tracking (`speed_tracking.png`)

| Command, m/s | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.27 |
|---|---|---|---|---|---|---|---|
| Speed, m/s (mean of 3 seeds) | 0.000 | 0.050 | 0.098 | 0.147 | 0.196 | 0.245 | 0.264 |
| Falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At 0.25 m/s: mean |step speed - command| 0.010 m/s (control-step speeds swing with the stride, SD 0.011 m/s); |mean speed - command| 0.005 m/s. Sideways drift worst +0.23 m, heading change worst -4.4 deg over 18 s.

## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`

Environment: `hardware/triceratops_build/rl/trike_env.py` (sha256 a82a0c4d206f...; the run commit's file: a82a0c4d206f...).

Not applicable: not the policy gait_pipe1.py analysed.

Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: True.

## What this does not show

Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the hotel load is assumed. Head acceleration is the head body's, not a camera's or an onlooker's view of it. Charts: `footfall.png`, `weight_distribution.png`, `balance.png`, `drive_envelopes.png`, `drive_loads.png`, `power.png`, `steadiness.png`, `feet.png`, `speed_tracking.png`.
