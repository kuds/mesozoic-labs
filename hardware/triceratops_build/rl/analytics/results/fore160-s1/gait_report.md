# Gait quality of trike60-fore160-s1 (SIMULATION)

Written by `rl/analytics/gait_report.py` on 2026-11-25. Policy `trike60_10002432_steps.zip` (10,002,432 steps; sha256 85e6ec3b2875...), trained at commit f84a6919. Deterministic actions, servo delay 10 ms, seeds 100000-100004, randomisation and observation noise on unless stated, 20 s episodes measured from t = 2 s. One policy: every figure is reported, not judged. Definitions: `rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.

## Answer first

- **Speed:** 0.246 m/s at the 0.25 m/s command and 0.145 m/s at 0.15; falls 0 + 0 of 10 (sweep: 0 falls in 21 episodes).
- **Gait:** trot (trot); a diagonal gait whose RF+LH pair follows the LF+RH pair after 0.45 of a stride (0.5 for an even trot); stride 1.57 Hz, 15.6 cm per stride; phase after LF: RF 0.43, LH 0.42, RH 0.93 of a stride.
- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies outside it 43.8 % of the time; the centre of pressure is within 5 mm of its edge 31 % of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives 58 %.)
- **Drives:** worst RMS torque / continuous rating 1.01 (mean of episodes; right_hind_flexion).
- **Power:** 36 W joint mechanical (sum |torque x speed|); leg electrical estimate 75 W on a shared bus to 90 W per drive with no reuse of braking energy; about 59-67 min walking on one battery set with a 30 W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.
- **Steadiness:** body roll 1.0 deg and pitch 1.3 deg SD, height 18 mm range; head bob 26 mm and sway 32 mm (1-99 % range) at 1.6 Hz; head acceleration 5.5 m/s^2 RMS to 20 Hz, of which 1.9 m/s^2 at or below 5 Hz (the visible bob, 0.19 g) and 5.2 m/s^2 at 5-20 Hz (the step impacts' shake).

## Cases (5 episodes each)

| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W | CoM outside support (step contact) | Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.246 | 0 | trot | 1.57 | 15.6 | 35.7 | 75-90 | 43.8 % | 48 % | 186 |
| 0.15 m/s, randomised | 0.145 | 0 | other | 1.47 | 9.9 | 21.0 | 55-64 | 25.8 % | 57 % | 53 |
| 0.25 m/s, nominal model, observation noise off | 0.243 | 0 | trot | 1.54 | 15.8 | 36.2 | 80-95 | 47.8 % | 37 % | 232 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.254 | 0 | other | 1.73 | 14.6 | 44.7 | 102-120 | 1.8 % | 13 % | 92 |

Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet shuffle, and the 50 ms step rule splits them less cleanly (README).

## a. Footfalls (`footfall.png`)

Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Duty factor: steps / env debounce / raw | 0.44 / 0.43 / 0.43 | 0.60 / 0.59 / 0.58 | 0.63 / 0.63 / 0.62 | 0.90 / 0.85 / 0.76 |
| Steps per s | 1.56 | 1.58 | 1.58 | 0.87 |
| Touchdowns per s, env debounce (training evaluation's count) | 1.88 | 2.17 | 1.77 | 3.91 |
| Stride frequency, Hz (median period) | 1.57 | 1.57 | 1.57 | 1.46 |
| Stance, ms (median) | 275 | 388 | 408 | 559 |
| Swing, ms (median) | 363 | 250 | 236 | 87 |
| Phase after LF, cycles (R) | 0 | 0.43 (0.99) | 0.42 (0.98) | 0.93 (0.87) |

Gait class: **trot** (trot; largest phase error to the nearest template 0.08 cycles; tolerance 0.125). Per episode: trot, trot, trot, trot, trot. Diagonal timing: Hildebrand limb phase left 0.58, right 0.49 (trot 0.5); each fore foot lands +0.07 (LF after RH) and +0.01 (RF after LH) cycles after its diagonal hind partner; the RF+LH pair follows the LF+RH pair after 0.45 of a stride (0.5 when even). Stride length 15.6 cm (LF touchdown to LF touchdown). Irregular strides (period more than 25 % off the median): LF 0 %, RF 0 %, LH 0 %, RH 48 %.

Feet down (steps): 4 3 %, 3 51 %, 2 diagonal 44 %, 2 same side 1 %, 2 front 0 %, 2 hind 0 %, 1 0 %, 0 0 %.

Symmetry index (Robinson, %, 0 = symmetric): duty factor: fore 32, hind 35, sides 34; stance duration: fore 33, hind 80, sides 64; stance length: fore 33, hind 80, sides 65; load: fore 31, hind 10, sides 19.

## b. Weight distribution (`weight_distribution.png`)

Floor-truth sole normal forces. Load share LF 18 %, RF 24 %, LH 28 %, RH 31 %: front 42 % / rear 58 %, left 45 % / right 55 %. Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: LF 1.48 / 0.82, RF 1.66 / 0.91, LH 1.80 / 0.98, RH 1.95 / 1.04. 135 strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least 100.0 % of samples.

## c. Balance (`balance.png`)

Support polygon: hull of the capsule soles' contact-line ends of the feet in stance on the step contact the footfalls use (the env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. CoM ground projection outside the polygon 43.8 % of the time (by feet in stance: 1: - %, 2: 95 %, 3: 1 %, 4: 0 %; time with one foot in stance 0.2 %); CoM margin median 27 mm. CoP outside: 0.00 % (it cannot leave the polygon by construction; a check); CoP within 5 mm of the edge 31 %, median margin 15 mm. No foot in stance 0.0 % of the time.

| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |
|---|---|---|---|
| step contact (env 10 ms debounce, then gaps and blips under 50 ms merged) | 43.8 % | 0.2 % | 31 % |
| env 10 ms debounce | 49.8 % | 1.1 % | 35 % |
| raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts) | 58.5 % | 3.0 % | 42 % |

Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot 'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the contact model, not the gait. Nominal model: step contact 47.8 %, env 10 ms debounce 52.0 %, raw 58.7 %.

## d. Drives (`drive_envelopes.png`, `drive_loads.png`)

| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |
|---|---|---|---|---|---|
| left_fore_roll | 0.45 (0.48) | 0.0 % (0.0 %) | 18.9 | 10 | 0.7 % |
| left_fore_pitch | 0.79 (0.82) | 0.0 % (0.0 %) | 20.5 | 34 | 8.8 % |
| left_fore_flexion | 0.66 (0.68) | 0.0 % (0.0 %) | 36.9 | 64 | 2.1 % |
| right_fore_roll | 0.48 (0.51) | 0.0 % (0.0 %) | 18.8 | 8 | 1.0 % |
| right_fore_pitch | 0.96 (1.01) | 0.0 % (0.0 %) | 20.4 | 60 | 13.8 % |
| right_fore_flexion | 0.77 (0.80) | 0.0 % (0.0 %) | 37.1 | 46 | 3.4 % |
| left_hind_roll | 0.69 (0.77) | 0.0 % (0.0 %) | 20.8 | 9 | 6.1 % |
| left_hind_pitch | 0.81 (0.86) | 0.0 % (0.0 %) | 19.6 | 28 | 10.6 % |
| left_hind_flexion | 0.65 (0.69) | 0.3 % (0.9 %) | 18.8 | 45 | 5.1 % |
| right_hind_roll | 0.56 (0.63) | 0.0 % (0.0 %) | 20.2 | 12 | 0.9 % |
| right_hind_pitch | 0.74 (0.86) | 0.0 % (0.0 %) | 20.4 | 37 | 7.2 % |
| right_hind_flexion | 1.01 (1.06) | 0.0 % (0.0 %) | 18.1 | 42 | 19.6 % |

Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): roll: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; pitch: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; front elbow: model limit 37.8, maker peak 54.0, maker rated 18.00, continuous 10.80, speed allowance 108 rpm; hind knee: model limit 18.9, maker peak 27.0, maker rated 9.45, continuous 6.75, speed allowance 83 rpm.

## e. Power and energy (`power.png`)

| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W | CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive | Run time, per drive, 25 / 40 W hotel |
|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 35.7 | 19.3 | 70.2 | 75.0 / 90.2 | 1.17 | 3.43 / 3.93 | 67 / 59 | 61 / 54 |
| 0.15 m/s, randomised | 21.0 | 11.9 | 51.4 | 55.1 / 63.6 | 1.17 | 4.72 / 5.19 | 83 / 75 | 80 / 68 |
| 0.25 m/s, nominal model, observation noise off | 36.2 | 19.7 | 74.8 | 79.7 / 95.1 | 1.19 | 3.59 / 4.09 | 64 / 56 | 59 / 52 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 44.7 | 25.7 | 93.6 | 102.3 / 120.2 | 1.43 | 4.23 / 4.80 | 54 / 47 | 49 / 44 |

Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is 3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step. Not included: the head and tail servos. Hotel load 30 W (range 25-40 W) is an assumption, not a parts-list figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC losses. Battery 146.5 Wh x 80% usable. Torque (copper loss) dominates the electrical figures.

## f. Body and head steadiness (`steadiness.png`)

| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 1.01 | 1.28 | 18.4 | 26.0 | 31.6 | 5.54 | 1.88 | 5.21 | 24.4 (5.8) |
| 0.15 m/s, randomised | 0.75 | 0.85 | 12.9 | 15.4 | 26.0 | 3.45 | 1.08 | 3.27 | 15.0 (2.6) |
| 0.25 m/s, nominal model, observation noise off | 0.98 | 1.35 | 18.9 | 26.7 | 31.9 | 5.54 | 1.90 | 5.21 | 19.5 (6.1) |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 1.20 | 1.55 | 14.0 | 40.3 | 43.9 | 4.68 | 2.21 | 4.12 | 20.5 (5.3) |

Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at 20 Hz, gravity excluded, and split at 5 Hz: at or below it the visible bob and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range after removing a 1 s moving average.

## g. Feet (`feet.png`)

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Swing clearance, mm, median (10th pct) | 42.0 (37.5) | 21.0 (14.7) | 13.4 (10.3) | 4.5 (1.5) |
| Swings under 5 mm | 0 % | 0 % | 1 % | 52 % |
| Touchdown speed, m/s, median (95th) | 0.47 (0.67) | 0.47 (0.72) | 0.24 (0.36) | 0.20 (0.43) |
| Peak load rate, body weights/s, median (95th) | 130 (214) | 144 (231) | 63 (99) | 132 (229) |
| Stance slip speed at the sole's origin, mean / 95th pct worst, m/s | 0.07 / 0.20 | 0.08 / 0.20 | 0.09 / 0.28 | 0.15 / 0.61 |
| Slip ratio at the sole's origin, mean (worst) | 0.13 (0.14) | 0.18 (0.19) | 0.22 (0.23) | 0.48 (0.58) |
| Slip ratio at the sole's contacts, mean (worst) | 0.07 (0.08) | 0.09 (0.10) | 0.14 (0.14) | 0.41 (0.53) |

## h. Speed tracking (`speed_tracking.png`)

| Command, m/s | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.27 |
|---|---|---|---|---|---|---|---|
| Speed, m/s (mean of 3 seeds) | 0.002 | 0.040 | 0.093 | 0.144 | 0.194 | 0.246 | 0.267 |
| Falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At 0.25 m/s: mean |step speed - command| 0.014 m/s (control-step speeds swing with the stride, SD 0.018 m/s); |mean speed - command| 0.004 m/s. Sideways drift worst -0.29 m, heading change worst -18.5 deg over 18 s.

## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`

Environment: `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c39...; the run commit's file: -...).

Not applicable: not the policy gait_pipe1.py analysed.

Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: True.

## What this does not show

Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the hotel load is assumed. Head acceleration is the head body's, not a camera's or a visitor's view of it. Charts: `footfall.png`, `weight_distribution.png`, `balance.png`, `drive_envelopes.png`, `drive_loads.png`, `power.png`, `steadiness.png`, `feet.png`, `speed_tracking.png`.
