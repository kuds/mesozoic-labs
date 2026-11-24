# Gait quality of trike60-fore128-s1 (SIMULATION)

Written by `rl/analytics/gait_report.py` on 2026-11-24. Policy `trike60_10002432_steps.zip` (10,002,432 steps; sha256 dc3090ac8dd2...), trained at commit f84a6919. Deterministic actions, servo delay 10 ms, seeds 100000-100004, randomisation and observation noise on unless stated, 20 s episodes measured from t = 2 s. One policy: every figure is reported, not judged. Definitions: `rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.

## Answer first

- **Speed:** 0.244 m/s at the 0.25 m/s command and 0.147 m/s at 0.15; falls 0 + 0 of 10 (sweep: 0 falls in 21 episodes).
- **Gait:** other (nearest trot, 0.29 cycles off); a diagonal gait whose RF+LH pair follows the LF+RH pair after 0.36 of a stride (0.5 for an even trot); stride 0.69 Hz, 37.0 cm per stride; phase after LF: RF 0.57, LH 0.21, RH 0.06 of a stride.
- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies outside it 52.9 % of the time; the centre of pressure is within 5 mm of its edge 25 % of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives 61 %.)
- **Drives:** worst RMS torque / continuous rating 0.89 (mean of episodes; right_hind_flexion).
- **Power:** 27 W joint mechanical (sum |torque x speed|); leg electrical estimate 59 W on a shared bus to 70 W per drive with no reuse of braking energy; about 70-80 min walking on one battery set with a 30 W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.
- **Steadiness:** body roll 3.9 deg and pitch 1.5 deg SD, height 25 mm range; head bob 58 mm and sway 45 mm (1-99 % range) at 1.3 Hz; head acceleration 4.6 m/s^2 RMS to 20 Hz, of which 2.6 m/s^2 at or below 5 Hz (the visible bob, 0.26 g) and 3.8 m/s^2 at 5-20 Hz (the step impacts' shake).

## Cases (5 episodes each)

| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W | CoM outside support (step contact) | Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.244 | 0 | other | 0.69 | 37.0 | 26.5 | 59-70 | 52.9 % | 69 % | 261 |
| 0.15 m/s, randomised | 0.147 | 0 | other | 0.59 | 35.4 | 14.7 | 42-48 | 37.4 % | 81 % | 205 |
| 0.25 m/s, nominal model, observation noise off | 0.243 | 0 | other | 0.66 | 37.1 | 27.7 | 67-79 | 48.6 % | 38 % | 286 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.250 | 0 | other | 0.92 | 28.0 | 28.5 | 76-88 | 8.3 % | 8 % | 11 |

Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet shuffle, and the 50 ms step rule splits them less cleanly (README).

## a. Footfalls (`footfall.png`)

Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Duty factor: steps / env debounce / raw | 0.59 / 0.59 / 0.59 | 0.43 / 0.43 / 0.43 | 0.67 / 0.63 / 0.59 | 0.61 / 0.59 / 0.58 |
| Steps per s | 0.67 | 0.62 | 1.46 | 0.81 |
| Touchdowns per s, env debounce (training evaluation's count) | 0.67 | 0.66 | 5.09 | 1.67 |
| Stride frequency, Hz (median period) | 0.66 | 0.66 | 1.53 | 0.67 |
| Stance, ms (median) | 898 | 669 | 258 | 900 |
| Swing, ms (median) | 619 | 852 | 176 | 529 |
| Phase after LF, cycles (R) | 0 | 0.57 (0.99) | 0.21 (0.66) | 0.06 (1.00) |

Gait class: **other** (nearest trot, 0.29 cycles off; largest phase error to the nearest template 0.29 cycles; tolerance 0.125). Per episode: other, other, other, other, other. Diagonal timing: Hildebrand limb phase left 0.79, right 0.51 (trot 0.5); each fore foot lands -0.06 (LF after RH) and +0.36 (RF after LH) cycles after its diagonal hind partner; the RF+LH pair follows the LF+RH pair after 0.36 of a stride (0.5 when even). Stride length 37.0 cm (LF touchdown to LF touchdown). Irregular strides (period more than 25 % off the median): LF 0 %, RF 0 %, LH 69 %, RH 35 %.

Feet down (steps): 4 0 %, 3 30 %, 2 diagonal 63 %, 2 same side 6 %, 2 front 0 %, 2 hind 0 %, 1 0 %, 0 0 %.

Symmetry index (Robinson, %, 0 = symmetric): duty factor: fore 31, hind 10, sides 19; stance duration: fore 29, hind 53, sides 6; stance length: fore 28, hind 52, sides 7; load: fore 40, hind 2, sides 22.

## b. Weight distribution (`weight_distribution.png`)

Floor-truth sole normal forces. Load share LF 32 %, RF 21 %, LH 24 %, RH 23 %: front 53 % / rear 47 %, left 56 % / right 44 %. Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: LF 1.81 / 0.99, RF 2.07 / 0.98, LH 1.64 / 0.76, RH 1.83 / 0.99. 55 strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least 100.0 % of samples.

## c. Balance (`balance.png`)

Support polygon: hull of the capsule soles' contact-line ends of the feet in stance on the step contact the footfalls use (the env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. CoM ground projection outside the polygon 52.9 % of the time (by feet in stance: 1: - %, 2: 74 %, 3: 3 %, 4: - %; time with one foot in stance 0.0 %); CoM margin median -2 mm. CoP outside: 0.00 % (it cannot leave the polygon by construction; a check); CoP within 5 mm of the edge 25 %, median margin 8 mm. No foot in stance 0.0 % of the time.

| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |
|---|---|---|---|
| step contact (env 10 ms debounce, then gaps and blips under 50 ms merged) | 52.9 % | 0.0 % | 25 % |
| env 10 ms debounce | 57.7 % | 0.2 % | 28 % |
| raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts) | 60.9 % | 1.0 % | 29 % |

Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot 'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the contact model, not the gait. Nominal model: step contact 48.6 %, env 10 ms debounce 55.2 %, raw 59.1 %.

## d. Drives (`drive_envelopes.png`, `drive_loads.png`)

| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |
|---|---|---|---|---|---|
| left_fore_roll | 0.44 (0.50) | 0.0 % (0.0 %) | 19.1 | 10 | 0.2 % |
| left_fore_pitch | 0.81 (0.85) | 0.0 % (0.0 %) | 20.5 | 47 | 8.7 % |
| left_fore_flexion | 0.70 (0.74) | 0.0 % (0.0 %) | 36.9 | 74 | 2.0 % |
| right_fore_roll | 0.39 (0.44) | 0.0 % (0.0 %) | 19.7 | 13 | 0.7 % |
| right_fore_pitch | 0.63 (0.66) | 0.0 % (0.0 %) | 20.4 | 44 | 4.4 % |
| right_fore_flexion | 0.50 (0.52) | 0.0 % (0.0 %) | 37.1 | 62 | 2.1 % |
| left_hind_roll | 0.43 (0.48) | 0.0 % (0.1 %) | 20.8 | 15 | 1.2 % |
| left_hind_pitch | 0.67 (0.71) | 0.0 % (0.0 %) | 19.6 | 29 | 4.7 % |
| left_hind_flexion | 0.71 (0.74) | 0.1 % (0.2 %) | 18.8 | 49 | 5.3 % |
| right_hind_roll | 0.63 (0.70) | 0.0 % (0.0 %) | 20.1 | 7 | 0.8 % |
| right_hind_pitch | 0.71 (0.78) | 0.0 % (0.0 %) | 20.4 | 37 | 5.0 % |
| right_hind_flexion | 0.89 (0.92) | 0.0 % (0.0 %) | 18.1 | 48 | 10.8 % |

Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): roll: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; pitch: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; front elbow: model limit 37.8, maker peak 54.0, maker rated 18.00, continuous 10.80, speed allowance 108 rpm; hind knee: model limit 18.9, maker peak 27.0, maker rated 9.45, continuous 6.75, speed allowance 83 rpm.

## e. Power and energy (`power.png`)

| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W | CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive | Run time, per drive, 25 / 40 W hotel |
|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 26.5 | 14.0 | 55.6 | 58.6 / 70.4 | 0.88 | 2.93 / 3.32 | 80 / 70 | 74 / 64 |
| 0.15 m/s, randomised | 14.7 | 8.0 | 39.8 | 41.7 / 48.2 | 0.81 | 3.93 / 4.28 | 98 / 90 | 96 / 80 |
| 0.25 m/s, nominal model, observation noise off | 27.7 | 14.5 | 63.9 | 66.8 / 79.2 | 0.91 | 3.19 / 3.59 | 73 / 64 | 68 / 59 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 28.5 | 15.6 | 71.3 | 75.7 / 87.6 | 0.92 | 3.43 / 3.82 | 67 / 60 | 63 / 55 |

Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is 3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step. Not included: the head and tail servos. Hotel load 30 W (range 25-40 W) is an assumption, not a parts-list figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC losses. Battery 146.5 Wh x 80% usable. Torque (copper loss) dominates the electrical figures.

## f. Body and head steadiness (`steadiness.png`)

| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 3.93 | 1.50 | 25.4 | 58.4 | 45.3 | 4.62 | 2.59 | 3.82 | 25.6 (7.7) |
| 0.15 m/s, randomised | 4.22 | 1.45 | 29.2 | 42.3 | 39.3 | 2.91 | 1.54 | 2.47 | 18.2 (5.3) |
| 0.25 m/s, nominal model, observation noise off | 4.16 | 1.48 | 24.5 | 59.0 | 41.7 | 4.48 | 2.65 | 3.62 | 22.6 (7.3) |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 3.48 | 1.12 | 31.1 | 39.6 | 81.1 | 4.46 | 2.46 | 3.72 | 21.4 (7.6) |

Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at 20 Hz, gravity excluded, and split at 5 Hz: at or below it the visible bob and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range after removing a 1 s moving average.

## g. Feet (`feet.png`)

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Swing clearance, mm, median (10th pct) | 94.1 (83.2) | 85.9 (83.8) | 3.3 (1.1) | 28.8 (2.4) |
| Swings under 5 mm | 0 % | 0 % | 66 % | 19 % |
| Touchdown speed, m/s, median (95th) | 0.83 (0.91) | 0.54 (0.70) | 0.17 (0.49) | 0.52 (0.65) |
| Peak load rate, body weights/s, median (95th) | 132 (243) | 169 (251) | 81 (242) | 126 (172) |
| Stance slip speed at the sole's origin, mean / 95th pct worst, m/s | 0.03 / 0.11 | 0.03 / 0.13 | 0.08 / 0.36 | 0.07 / 0.36 |
| Slip ratio at the sole's origin, mean (worst) | 0.08 (0.09) | 0.06 (0.06) | 0.20 (0.25) | 0.17 (0.18) |
| Slip ratio at the sole's contacts, mean (worst) | 0.06 (0.07) | 0.04 (0.05) | 0.14 (0.20) | 0.12 (0.14) |

## h. Speed tracking (`speed_tracking.png`)

| Command, m/s | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.27 |
|---|---|---|---|---|---|---|---|
| Speed, m/s (mean of 3 seeds) | 0.000 | 0.048 | 0.097 | 0.147 | 0.196 | 0.244 | 0.263 |
| Falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At 0.25 m/s: mean |step speed - command| 0.013 m/s (control-step speeds swing with the stride, SD 0.016 m/s); |mean speed - command| 0.006 m/s. Sideways drift worst -0.46 m, heading change worst -10.0 deg over 18 s.

## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`

Environment: `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c39...; the run commit's file: -...).

Not applicable: not the policy gait_pipe1.py analysed.

Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: True.

## What this does not show

Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the hotel load is assumed. Head acceleration is the head body's, not a camera's or a visitor's view of it. Charts: `footfall.png`, `weight_distribution.png`, `balance.png`, `drive_envelopes.png`, `drive_loads.png`, `power.png`, `steadiness.png`, `feet.png`, `speed_tracking.png`.
