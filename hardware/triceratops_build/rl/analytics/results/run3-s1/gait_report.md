# Gait quality of trike60-run3-s1 (SIMULATION)

Written by `rl/analytics/gait_report.py` on 2026-11-23. Policy `trike60_10002432_steps.zip` (10,002,432 steps; sha256 b1c97ffde02c...), trained at commit efd1e1c7. Deterministic actions, servo delay 10 ms, seeds 100000-100004, randomisation and observation noise on unless stated, 20 s episodes measured from t = 2 s. One policy: every figure is reported, not judged. Definitions: `rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.

## Answer first

- **Speed:** 0.239 m/s at the 0.25 m/s command and 0.141 m/s at 0.15; falls 0 + 0 of 10 (sweep: 0 falls in 21 episodes).
- **Gait:** other (phase not locked (LH)); a diagonal gait whose RF+LH pair follows the LF+RH pair after 0.50 of a stride (0.5 for an even trot); stride 0.86 Hz, 28.7 cm per stride; phase after LF: RF 0.60, LH 0.39, RH 1.00 of a stride.
- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies outside it 45.8 % of the time; the centre of pressure is within 5 mm of its edge 20 % of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives 51 %.)
- **Drives:** worst RMS torque / continuous rating 0.80 (mean of episodes; left_hind_pitch).
- **Power:** 24 W joint mechanical (sum |torque x speed|); leg electrical estimate 49 W on a shared bus to 60 W per drive with no reuse of braking energy; about 79-89 min walking on one battery set with a 30 W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.
- **Steadiness:** body roll 1.1 deg and pitch 1.6 deg SD, height 26 mm range; head bob 51 mm and sway 29 mm (1-99 % range) at 1.2 Hz; head acceleration 3.7 m/s^2 RMS to 20 Hz, of which 1.9 m/s^2 at or below 5 Hz (the visible bob, 0.20 g) and 3.1 m/s^2 at 5-20 Hz (the step impacts' shake).

## Cases (5 episodes each)

| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W | CoM outside support (step contact) | Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.239 | 0 | other | 0.86 | 28.7 | 24.2 | 49-60 | 45.8 % | 51 % | 74 |
| 0.15 m/s, randomised | 0.141 | 0 | other | 0.58 | 25.4 | 13.0 | 36-41 | 27.7 % | 80 % | 172 |
| 0.25 m/s, nominal model, observation noise off | 0.239 | 0 | other | 0.81 | 29.8 | 23.0 | 46-56 | 41.6 % | 49 % | 153 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.254 | 0 | walk | 1.32 | 19.3 | 27.8 | 57-69 | 13.5 % | 0 % | 31 |

Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet shuffle, and the 50 ms step rule splits them less cleanly (README).

## a. Footfalls (`footfall.png`)

Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Duty factor: steps / env debounce / raw | 0.64 / 0.63 / 0.63 | 0.40 / 0.40 / 0.39 | 0.54 / 0.52 / 0.47 | 0.68 / 0.67 / 0.67 |
| Steps per s | 0.81 | 0.82 | 1.38 | 0.81 |
| Touchdowns per s, env debounce (training evaluation's count) | 0.87 | 0.83 | 3.33 | 0.94 |
| Stride frequency, Hz (median period) | 0.83 | 0.83 | 1.60 | 0.83 |
| Stance, ms (median) | 755 | 490 | 497 | 804 |
| Swing, ms (median) | 452 | 707 | 355 | 403 |
| Phase after LF, cycles (R) | 0 | 0.60 (0.99) | 0.39 (0.21) | 1.00 (0.94) |

Gait class: **other** (phase not locked (LH); largest phase error to the nearest template 0.11 cycles; tolerance 0.125). Per episode: other, other, other, other, other. Diagonal timing: Hildebrand limb phase left 0.61, right 0.60 (trot 0.5); each fore foot lands +0.00 (LF after RH) and +0.22 (RF after LH) cycles after its diagonal hind partner; the RF+LH pair follows the LF+RH pair after 0.50 of a stride (0.5 when even). Stride length 28.7 cm (LF touchdown to LF touchdown). Irregular strides (period more than 25 % off the median): LF 0 %, RF 0 %, LH 51 %, RH 0 %.

Feet down (steps): 4 1 %, 3 23 %, 2 diagonal 74 %, 2 same side 2 %, 2 front 0 %, 2 hind 0 %, 1 0 %, 0 0 %.

Symmetry index (Robinson, %, 0 = symmetric): duty factor: fore 46, hind 22, sides 9; stance duration: fore 42, hind 68, sides 12; stance length: fore 41, hind 66, sides 12; load: fore 54, hind 31, sides 13.

## b. Weight distribution (`weight_distribution.png`)

Floor-truth sole normal forces. Load share LF 33 %, RF 19 %, LH 20 %, RH 28 %: front 52 % / rear 48 %, left 53 % / right 47 %. Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: LF 1.61 / 0.90, RF 2.55 / 1.16, LH 1.96 / 1.05, RH 1.81 / 0.90. 68 strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least 100.0 % of samples.

## c. Balance (`balance.png`)

Support polygon: hull of the capsule soles' contact-line ends of the feet in stance on the step contact the footfalls use (the env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. CoM ground projection outside the polygon 45.8 % of the time (by feet in stance: 1: - %, 2: 59 %, 3: 2 %, 4: 0 %; time with one foot in stance 0.0 %); CoM margin median 1 mm. CoP outside: 0.00 % (it cannot leave the polygon by construction; a check); CoP within 5 mm of the edge 20 %, median margin 8 mm. No foot in stance 0.0 % of the time.

| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |
|---|---|---|---|
| step contact (env 10 ms debounce, then gaps and blips under 50 ms merged) | 45.8 % | 0.0 % | 20 % |
| env 10 ms debounce | 48.5 % | 0.1 % | 21 % |
| raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts) | 51.4 % | 1.0 % | 21 % |

Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot 'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the contact model, not the gait. Nominal model: step contact 41.6 %, env 10 ms debounce 46.6 %, raw 49.7 %.

## d. Drives (`drive_envelopes.png`, `drive_loads.png`)

| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |
|---|---|---|---|---|---|
| left_fore_roll | 0.37 (0.39) | 0.0 % (0.0 %) | 14.2 | 5 | 0.0 % |
| left_fore_pitch | 0.68 (0.71) | 0.0 % (0.0 %) | 20.5 | 68 | 4.1 % |
| left_fore_flexion | 0.59 (0.63) | 0.0 % (0.0 %) | 36.9 | 63 | 0.9 % |
| right_fore_roll | 0.33 (0.33) | 0.0 % (0.0 %) | 19.3 | 9 | 0.8 % |
| right_fore_pitch | 0.72 (0.76) | 0.0 % (0.0 %) | 20.4 | 25 | 7.3 % |
| right_fore_flexion | 0.51 (0.52) | 0.0 % (0.0 %) | 37.1 | 61 | 2.4 % |
| left_hind_roll | 0.29 (0.32) | 0.0 % (0.0 %) | 19.3 | 10 | 0.6 % |
| left_hind_pitch | 0.80 (0.85) | 0.0 % (0.0 %) | 19.6 | 26 | 9.9 % |
| left_hind_flexion | 0.45 (0.48) | 0.1 % (0.3 %) | 18.8 | 51 | 1.9 % |
| right_hind_roll | 0.49 (0.53) | 0.0 % (0.0 %) | 18.5 | 9 | 0.1 % |
| right_hind_pitch | 0.76 (0.80) | 0.0 % (0.0 %) | 20.4 | 45 | 7.2 % |
| right_hind_flexion | 0.75 (0.81) | 0.0 % (0.0 %) | 18.1 | 39 | 5.2 % |

Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): roll: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; pitch: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; front elbow: model limit 37.8, maker peak 54.0, maker rated 18.00, continuous 10.80, speed allowance 108 rpm; hind knee: model limit 18.9, maker peak 27.0, maker rated 9.45, continuous 6.75, speed allowance 83 rpm.

## e. Power and energy (`power.png`)

| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W | CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive | Run time, per drive, 25 / 40 W hotel |
|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 24.2 | 12.5 | 46.6 | 48.8 / 59.5 | 0.82 | 2.67 / 3.03 | 89 / 79 | 83 / 71 |
| 0.15 m/s, randomised | 13.0 | 7.1 | 33.8 | 35.5 / 41.2 | 0.74 | 3.76 / 4.08 | 107 / 99 | 106 / 87 |
| 0.25 m/s, nominal model, observation noise off | 23.0 | 11.9 | 43.6 | 45.6 / 55.9 | 0.77 | 2.54 / 2.89 | 93 / 82 | 87 / 73 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 27.8 | 14.8 | 53.0 | 56.7 / 68.5 | 0.89 | 2.76 / 3.14 | 81 / 71 | 75 / 65 |

Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is 3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step. Not included: the head and tail servos. Hotel load 30 W (range 25-40 W) is an assumption, not a parts-list figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC losses. Battery 146.5 Wh x 80% usable. Torque (copper loss) dominates the electrical figures.

## f. Body and head steadiness (`steadiness.png`)

| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 1.10 | 1.64 | 26.3 | 50.8 | 29.1 | 3.66 | 1.94 | 3.11 | 20.0 (5.8) |
| 0.15 m/s, randomised | 0.97 | 1.63 | 21.9 | 38.9 | 23.1 | 2.14 | 1.04 | 1.86 | 11.6 (4.0) |
| 0.25 m/s, nominal model, observation noise off | 0.87 | 1.68 | 27.1 | 49.4 | 27.7 | 3.15 | 1.81 | 2.58 | 16.8 (5.5) |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 3.16 | 1.54 | 20.9 | 44.2 | 72.0 | 5.18 | 3.09 | 4.16 | 20.6 (7.1) |

Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at 20 Hz, gravity excluded, and split at 5 Hz: at or below it the visible bob and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range after removing a 1 s moving average.

## g. Feet (`feet.png`)

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Swing clearance, mm, median (10th pct) | 50.6 (47.0) | 82.4 (75.3) | 7.1 (1.2) | 22.6 (18.4) |
| Swings under 5 mm | 0 % | 0 % | 42 % | 0 % |
| Touchdown speed, m/s, median (95th) | 0.36 (0.44) | 0.56 (0.69) | 0.12 (0.64) | 0.15 (0.43) |
| Peak load rate, body weights/s, median (95th) | 109 (175) | 121 (192) | 112 (254) | 58 (156) |
| Stance slip speed at the sole's origin, mean / 95th pct worst, m/s | 0.02 / 0.08 | 0.04 / 0.15 | 0.09 / 0.40 | 0.04 / 0.14 |
| Slip ratio at the sole's origin, mean (worst) | 0.06 (0.06) | 0.07 (0.07) | 0.18 (0.24) | 0.12 (0.13) |
| Slip ratio at the sole's contacts, mean (worst) | 0.04 (0.04) | 0.05 (0.05) | 0.14 (0.20) | 0.06 (0.07) |

## h. Speed tracking (`speed_tracking.png`)

| Command, m/s | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.27 |
|---|---|---|---|---|---|---|---|
| Speed, m/s (mean of 3 seeds) | 0.000 | 0.048 | 0.093 | 0.141 | 0.190 | 0.239 | 0.259 |
| Falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At 0.25 m/s: mean |step speed - command| 0.015 m/s (control-step speeds swing with the stride, SD 0.016 m/s); |mean speed - command| 0.011 m/s. Sideways drift worst +0.54 m, heading change worst +11.9 deg over 18 s.

## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`

Environment: `hardware/triceratops_build/rl/trike_env.py` (sha256 402819fcc1c3...; the run commit's file: -...).

Not applicable: not the policy gait_pipe1.py analysed.

Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: True.

## What this does not show

Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the hotel load is assumed. Head acceleration is the head body's, not a camera's or an onlooker's view of it. Charts: `footfall.png`, `weight_distribution.png`, `balance.png`, `drive_envelopes.png`, `drive_loads.png`, `power.png`, `steadiness.png`, `feet.png`, `speed_tracking.png`.
