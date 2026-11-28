# Gait quality of trike60-fore128-s3 (SIMULATION)

Written by `rl/analytics/gait_report.py` on 2026-11-26. Policy `trike60_10002432_steps.zip` (10,002,432 steps; sha256 652082b1aeb2...), trained at commit f84a6919. Deterministic actions, servo delay 10 ms, seeds 100000-100004, randomisation and observation noise on unless stated, 20 s episodes measured from t = 2 s. One policy: every figure is reported, not judged. Definitions: `rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.

## Answer first

- **Speed:** 0.240 m/s at the 0.25 m/s command and 0.143 m/s at 0.15; falls 0 + 0 of 10 (sweep: 0 falls in 21 episodes).
- **Gait:** other (nearest walk (diagonal sequence), 0.19 cycles off); a diagonal gait whose RF+LH pair follows the LF+RH pair after 0.44 of a stride (0.5 for an even trot); stride 1.29 Hz, 18.6 cm per stride; phase after LF: RF 0.31, LH 0.37, RH 0.80 of a stride.
- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies outside it 33.8 % of the time; the centre of pressure is within 5 mm of its edge 27 % of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives 45 %.)
- **Drives:** worst RMS torque / continuous rating 0.95 (mean of episodes; left_hind_pitch).
- **Power:** 34 W joint mechanical (sum |torque x speed|); leg electrical estimate 55 W on a shared bus to 70 W per drive with no reuse of braking energy; about 70-83 min walking on one battery set with a 30 W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.
- **Steadiness:** body roll 1.8 deg and pitch 0.8 deg SD, height 24 mm range; head bob 42 mm and sway 43 mm (1-99 % range) at 1.3 Hz; head acceleration 5.3 m/s^2 RMS to 20 Hz, of which 3.0 m/s^2 at or below 5 Hz (the visible bob, 0.30 g) and 4.4 m/s^2 at 5-20 Hz (the step impacts' shake).

## Cases (5 episodes each)

| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W | CoM outside support (step contact) | Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.240 | 0 | other | 1.29 | 18.6 | 34.4 | 55-70 | 33.8 % | 38 % | 216 |
| 0.15 m/s, randomised | 0.143 | 0 | other | 0.91 | 15.6 | 19.8 | 44-52 | 11.8 % | 63 % | 133 |
| 0.25 m/s, nominal model, observation noise off | 0.239 | 0 | other | 1.25 | 19.0 | 35.3 | 56-71 | 37.5 % | 5 % | 180 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.245 | 0 | trot | 1.54 | 16.0 | 34.6 | 67-83 | 5.6 % | 2 % | 13 |

Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet shuffle, and the 50 ms step rule splits them less cleanly (README).

## a. Footfalls (`footfall.png`)

Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Duty factor: steps / env debounce / raw | 0.31 / 0.28 / 0.27 | 0.71 / 0.71 / 0.70 | 0.67 / 0.67 / 0.66 | 0.86 / 0.78 / 0.64 |
| Steps per s | 1.27 | 1.30 | 1.30 | 1.10 |
| Touchdowns per s, env debounce (training evaluation's count) | 3.64 | 1.30 | 1.37 | 5.86 |
| Stride frequency, Hz (median period) | 1.28 | 1.29 | 1.29 | 1.36 |
| Stance, ms (median) | 241 | 550 | 524 | 590 |
| Swing, ms (median) | 534 | 226 | 252 | 102 |
| Phase after LF, cycles (R) | 0 | 0.31 (1.00) | 0.37 (0.99) | 0.80 (0.74) |

Gait class: **other** (nearest walk (diagonal sequence), 0.19 cycles off; largest phase error to the nearest template 0.19 cycles; tolerance 0.125). Per episode: other, other, other, other, other. Diagonal timing: Hildebrand limb phase left 0.63, right 0.51 (trot 0.5); each fore foot lands +0.20 (LF after RH) and -0.06 (RF after LH) cycles after its diagonal hind partner; the RF+LH pair follows the LF+RH pair after 0.44 of a stride (0.5 when even). Stride length 18.6 cm (LF touchdown to LF touchdown). Irregular strides (period more than 25 % off the median): LF 0 %, RF 0 %, LH 0 %, RH 38 %.

Feet down (steps): 4 2 %, 3 51 %, 2 diagonal 41 %, 2 same side 6 %, 2 front 0 %, 2 hind 0 %, 1 0 %, 0 0 %.

Symmetry index (Robinson, %, 0 = symmetric): duty factor: fore 77, hind 25, sides 46; stance duration: fore 78, hind 37, sides 53; stance length: fore 80, hind 37, sides 54; load: fore 104, hind 25, sides 33.

## b. Weight distribution (`weight_distribution.png`)

Floor-truth sole normal forces. Load share LF 11 %, RF 34 %, LH 31 %, RH 24 %: front 45 % / rear 55 %, left 42 % / right 58 %. Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: LF 1.65 / 0.95, RF 1.86 / 1.11, LH 1.55 / 0.94, RH 1.67 / 1.04. 109 strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least 100.0 % of samples.

## c. Balance (`balance.png`)

Support polygon: hull of the capsule soles' contact-line ends of the feet in stance on the step contact the footfalls use (the env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. CoM ground projection outside the polygon 33.8 % of the time (by feet in stance: 1: - %, 2: 71 %, 3: 0 %, 4: 0 %; time with one foot in stance 0.3 %); CoM margin median 7 mm. CoP outside: 0.00 % (it cannot leave the polygon by construction; a check); CoP within 5 mm of the edge 27 %, median margin 8 mm. No foot in stance 0.0 % of the time.

| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |
|---|---|---|---|
| step contact (env 10 ms debounce, then gaps and blips under 50 ms merged) | 33.8 % | 0.3 % | 27 % |
| env 10 ms debounce | 37.3 % | 3.7 % | 28 % |
| raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts) | 45.3 % | 5.8 % | 33 % |

Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot 'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the contact model, not the gait. Nominal model: step contact 37.5 %, env 10 ms debounce 40.1 %, raw 47.9 %.

## d. Drives (`drive_envelopes.png`, `drive_loads.png`)

| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |
|---|---|---|---|---|---|
| left_fore_roll | 0.37 (0.39) | 0.0 % (0.0 %) | 19.1 | 14 | 1.3 % |
| left_fore_pitch | 0.67 (0.68) | 0.0 % (0.0 %) | 20.5 | 44 | 6.7 % |
| left_fore_flexion | 0.57 (0.60) | 0.0 % (0.0 %) | 36.9 | 97 | 2.5 % |
| right_fore_roll | 0.41 (0.44) | 0.0 % (0.0 %) | 17.4 | 13 | 1.1 % |
| right_fore_pitch | 0.78 (0.81) | 0.0 % (0.0 %) | 20.4 | 86 | 7.2 % |
| right_fore_flexion | 0.61 (0.63) | 0.0 % (0.0 %) | 37.1 | 57 | 1.9 % |
| left_hind_roll | 0.43 (0.51) | 0.0 % (0.0 %) | 20.5 | 8 | 1.4 % |
| left_hind_pitch | 0.95 (1.00) | 0.0 % (0.0 %) | 19.6 | 60 | 13.9 % |
| left_hind_flexion | 0.61 (0.67) | 0.2 % (0.5 %) | 18.8 | 47 | 2.9 % |
| right_hind_roll | 0.42 (0.50) | 0.0 % (0.0 %) | 20.9 | 15 | 1.2 % |
| right_hind_pitch | 0.54 (0.61) | 0.0 % (0.0 %) | 20.4 | 18 | 2.8 % |
| right_hind_flexion | 0.90 (0.95) | 0.0 % (0.0 %) | 18.1 | 32 | 15.2 % |

Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): roll: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; pitch: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; front elbow: model limit 37.8, maker peak 54.0, maker rated 18.00, continuous 10.80, speed allowance 108 rpm; hind knee: model limit 18.9, maker peak 27.0, maker rated 9.45, continuous 6.75, speed allowance 83 rpm.

## e. Power and energy (`power.png`)

| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W | CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive | Run time, per drive, 25 / 40 W hotel |
|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 34.4 | 17.1 | 52.2 | 55.2 / 69.9 | 1.16 | 2.86 / 3.36 | 83 / 70 | 74 / 64 |
| 0.15 m/s, randomised | 19.8 | 10.4 | 41.6 | 43.9 / 52.4 | 1.12 | 4.16 / 4.64 | 95 / 86 | 91 / 76 |
| 0.25 m/s, nominal model, observation noise off | 35.3 | 17.5 | 53.2 | 56.0 / 71.3 | 1.18 | 2.88 / 3.39 | 82 / 69 | 73 / 63 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 34.6 | 17.8 | 64.0 | 67.3 / 82.5 | 1.14 | 3.21 / 3.72 | 72 / 63 | 66 / 57 |

Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is 3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step. Not included: the head and tail servos. Hotel load 30 W (range 25-40 W) is an assumption, not a parts-list figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC losses. Battery 146.5 Wh x 80% usable. Torque (copper loss) dominates the electrical figures.

## f. Body and head steadiness (`steadiness.png`)

| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 1.83 | 0.78 | 23.9 | 42.4 | 43.0 | 5.29 | 2.95 | 4.38 | 18.6 (6.6) |
| 0.15 m/s, randomised | 1.44 | 0.52 | 22.2 | 33.5 | 35.4 | 2.97 | 1.80 | 2.36 | 12.7 (4.7) |
| 0.25 m/s, nominal model, observation noise off | 1.83 | 0.75 | 23.6 | 41.3 | 42.4 | 4.98 | 2.90 | 4.05 | 16.8 (6.1) |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 2.32 | 0.93 | 27.7 | 54.4 | 49.2 | 5.61 | 3.86 | 4.06 | 19.8 (9.7) |

Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at 20 Hz, gravity excluded, and split at 5 Hz: at or below it the visible bob and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range after removing a 1 s moving average.

## g. Feet (`feet.png`)

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Swing clearance, mm, median (10th pct) | 28.6 (25.7) | 37.6 (35.0) | 16.3 (13.9) | 2.8 (1.0) |
| Swings under 5 mm | 0 % | 0 % | 0 % | 78 % |
| Touchdown speed, m/s, median (95th) | 0.35 (0.51) | 0.62 (0.71) | 0.36 (0.43) | 0.07 (0.21) |
| Peak load rate, body weights/s, median (95th) | 185 (228) | 131 (172) | 84 (124) | 34 (165) |
| Stance slip speed at the sole's origin, mean / 95th pct worst, m/s | 0.07 / 0.18 | 0.04 / 0.15 | 0.04 / 0.16 | 0.17 / 0.45 |
| Slip ratio at the sole's origin, mean (worst) | 0.08 (0.08) | 0.12 (0.12) | 0.11 (0.11) | 0.46 (0.53) |
| Slip ratio at the sole's contacts, mean (worst) | 0.06 (0.07) | 0.05 (0.05) | 0.08 (0.09) | 0.43 (0.51) |

## h. Speed tracking (`speed_tracking.png`)

| Command, m/s | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.27 |
|---|---|---|---|---|---|---|---|
| Speed, m/s (mean of 3 seeds) | 0.000 | 0.045 | 0.095 | 0.143 | 0.192 | 0.240 | 0.259 |
| Falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At 0.25 m/s: mean |step speed - command| 0.016 m/s (control-step speeds swing with the stride, SD 0.018 m/s); |mean speed - command| 0.010 m/s. Sideways drift worst -0.63 m, heading change worst -18.1 deg over 18 s.

## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`

Environment: `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c39...; the run commit's file: -...).

Not applicable: not the policy gait_pipe1.py analysed.

Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: True.

## What this does not show

Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the hotel load is assumed. Head acceleration is the head body's, not a camera's or an onlooker's view of it. Charts: `footfall.png`, `weight_distribution.png`, `balance.png`, `drive_envelopes.png`, `drive_loads.png`, `power.png`, `steadiness.png`, `feet.png`, `speed_tracking.png`.
