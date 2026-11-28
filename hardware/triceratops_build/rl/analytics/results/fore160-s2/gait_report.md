# Gait quality of trike60-fore160-s2 (SIMULATION)

Written by `rl/analytics/gait_report.py` on 2026-11-25. Policy `trike60_10002432_steps.zip` (10,002,432 steps; sha256 1f53d26213d8...), trained at commit f84a6919. Deterministic actions, servo delay 10 ms, seeds 100000-100004, randomisation and observation noise on unless stated, 20 s episodes measured from t = 2 s. One policy: every figure is reported, not judged. Definitions: `rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.

## Answer first

- **Speed:** 0.244 m/s at the 0.25 m/s command and 0.148 m/s at 0.15; falls 0 + 0 of 10 (sweep: 0 falls in 21 episodes).
- **Gait:** trot (trot); a diagonal gait whose RF+LH pair follows the LF+RH pair after 0.48 of a stride (0.5 for an even trot); stride 1.65 Hz, 14.9 cm per stride; phase after LF: RF 0.56, LH 0.48, RH 0.08 of a stride.
- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies outside it 36.5 % of the time; the centre of pressure is within 5 mm of its edge 10 % of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives 42 %.)
- **Drives:** worst RMS torque / continuous rating 1.09 (mean of episodes; left_hind_flexion).
- **Power:** 28 W joint mechanical (sum |torque x speed|); leg electrical estimate 77 W on a shared bus to 90 W per drive with no reuse of braking energy; about 59-66 min walking on one battery set with a 30 W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.
- **Steadiness:** body roll 1.0 deg and pitch 0.5 deg SD, height 10 mm range; head bob 22 mm and sway 44 mm (1-99 % range) at 2.3 Hz; head acceleration 4.3 m/s^2 RMS to 20 Hz, of which 2.6 m/s^2 at or below 5 Hz (the visible bob, 0.26 g) and 3.4 m/s^2 at 5-20 Hz (the step impacts' shake).

## Cases (5 episodes each)

| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W | CoM outside support (step contact) | Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.244 | 0 | trot | 1.65 | 14.9 | 28.0 | 77-90 | 36.5 % | 66 % | 311 |
| 0.15 m/s, randomised | 0.148 | 0 | other | 1.54 | 9.7 | 17.9 | 71-79 | 17.7 % | 69 % | 333 |
| 0.25 m/s, nominal model, observation noise off | 0.245 | 0 | trot | 1.66 | 14.8 | 27.6 | 82-94 | 48.4 % | 6 % | 210 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.249 | 0 | other | 2.15 | 19.5 | 30.1 | 87-99 | 4.2 % | 39 % | 195 |

Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet shuffle, and the 50 ms step rule splits them less cleanly (README).

## a. Footfalls (`footfall.png`)

Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Duty factor: steps / env debounce / raw | 0.61 / 0.60 / 0.60 | 0.47 / 0.47 / 0.47 | 0.64 / 0.63 / 0.59 | 0.66 / 0.63 / 0.61 |
| Steps per s | 1.63 | 1.64 | 2.21 | 1.68 |
| Touchdowns per s, env debounce (training evaluation's count) | 1.76 | 1.68 | 5.02 | 3.74 |
| Stride frequency, Hz (median period) | 1.64 | 1.63 | 2.20 | 1.65 |
| Stance, ms (median) | 371 | 288 | 308 | 394 |
| Swing, ms (median) | 242 | 322 | 174 | 204 |
| Phase after LF, cycles (R) | 0 | 0.56 (1.00) | 0.48 (0.52) | 0.08 (1.00) |

Gait class: **trot** (trot; largest phase error to the nearest template 0.08 cycles; tolerance 0.125). Per episode: other, trot, trot, other, other. Diagonal timing: Hildebrand limb phase left 0.52, right 0.48 (trot 0.5); each fore foot lands -0.08 (LF after RH) and +0.08 (RF after LH) cycles after its diagonal hind partner; the RF+LH pair follows the LF+RH pair after 0.48 of a stride (0.5 when even). Stride length 14.9 cm (LF touchdown to LF touchdown). Irregular strides (period more than 25 % off the median): LF 0 %, RF 0 %, LH 66 %, RH 5 %.

Feet down (steps): 4 1 %, 3 35 %, 2 diagonal 58 %, 2 same side 5 %, 2 front 0 %, 2 hind 0 %, 1 0 %, 0 0 %.

Symmetry index (Robinson, %, 0 = symmetric): duty factor: fore 24, hind 2, sides 10; stance duration: fore 24, hind 29, sides 3; stance length: fore 24, hind 28, sides 2; load: fore 28, hind 4, sides 16.

## b. Weight distribution (`weight_distribution.png`)

Floor-truth sole normal forces. Load share LF 30 %, RF 23 %, LH 24 %, RH 23 %: front 52 % / rear 48 %, left 54 % / right 46 %. Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: LF 1.50 / 0.98, RF 1.22 / 0.74, LH 1.65 / 0.77, RH 1.73 / 0.94. 142 strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least 100.0 % of samples.

## c. Balance (`balance.png`)

Support polygon: hull of the capsule soles' contact-line ends of the feet in stance on the step contact the footfalls use (the env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. CoM ground projection outside the polygon 36.5 % of the time (by feet in stance: 1: - %, 2: 58 %, 3: 0 %, 4: 0 %; time with one foot in stance 0.0 %); CoM margin median 5 mm. CoP outside: 0.00 % (it cannot leave the polygon by construction; a check); CoP within 5 mm of the edge 10 %, median margin 9 mm. No foot in stance 0.0 % of the time.

| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |
|---|---|---|---|
| step contact (env 10 ms debounce, then gaps and blips under 50 ms merged) | 36.5 % | 0.0 % | 10 % |
| env 10 ms debounce | 37.8 % | 0.0 % | 10 % |
| raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts) | 42.1 % | 0.1 % | 10 % |

Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot 'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the contact model, not the gait. Nominal model: step contact 48.4 %, env 10 ms debounce 43.3 %, raw 44.1 %.

## d. Drives (`drive_envelopes.png`, `drive_loads.png`)

| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |
|---|---|---|---|---|---|
| left_fore_roll | 0.54 (0.58) | 0.0 % (0.0 %) | 13.7 | 6 | 0.2 % |
| left_fore_pitch | 0.90 (0.91) | 0.0 % (0.0 %) | 20.5 | 44 | 10.6 % |
| left_fore_flexion | 0.83 (0.87) | 0.0 % (0.0 %) | 36.9 | 24 | 2.7 % |
| right_fore_roll | 0.43 (0.46) | 0.0 % (0.0 %) | 14.2 | 5 | 0.1 % |
| right_fore_pitch | 0.78 (0.80) | 0.0 % (0.0 %) | 20.4 | 34 | 7.4 % |
| right_fore_flexion | 0.74 (0.76) | 0.0 % (0.0 %) | 36.4 | 47 | 0.8 % |
| left_hind_roll | 0.58 (0.60) | 0.0 % (0.0 %) | 20.2 | 9 | 2.0 % |
| left_hind_pitch | 0.67 (0.71) | 0.0 % (0.0 %) | 19.6 | 27 | 2.4 % |
| left_hind_flexion | 1.09 (1.16) | 0.0 % (0.0 %) | 18.8 | 25 | 30.4 % |
| right_hind_roll | 0.68 (0.71) | 0.0 % (0.0 %) | 15.8 | 7 | 3.0 % |
| right_hind_pitch | 0.67 (0.69) | 0.0 % (0.0 %) | 20.4 | 31 | 2.6 % |
| right_hind_flexion | 0.94 (0.97) | 0.0 % (0.0 %) | 18.1 | 37 | 17.5 % |

Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): roll: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; pitch: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; front elbow: model limit 37.8, maker peak 54.0, maker rated 18.00, continuous 10.80, speed allowance 108 rpm; hind knee: model limit 18.9, maker peak 27.0, maker rated 9.45, continuous 6.75, speed allowance 83 rpm.

## e. Power and energy (`power.png`)

| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W | CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive | Run time, per drive, 25 / 40 W hotel |
|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 28.0 | 15.1 | 74.2 | 77.2 / 89.7 | 0.92 | 3.52 / 3.94 | 66 / 59 | 61 / 54 |
| 0.15 m/s, randomised | 17.9 | 9.9 | 68.8 | 71.5 / 79.1 | 0.97 | 5.51 / 5.93 | 69 / 65 | 68 / 59 |
| 0.25 m/s, nominal model, observation noise off | 27.6 | 14.9 | 78.5 | 81.5 / 93.8 | 0.90 | 3.62 / 4.02 | 63 / 57 | 59 / 53 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 30.1 | 17.6 | 81.0 | 87.3 / 99.4 | 0.98 | 3.82 / 4.21 | 60 / 55 | 57 / 51 |

Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is 3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step. Not included: the head and tail servos. Hotel load 30 W (range 25-40 W) is an assumption, not a parts-list figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC losses. Battery 146.5 Wh x 80% usable. Torque (copper loss) dominates the electrical figures.

## f. Body and head steadiness (`steadiness.png`)

| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 1.04 | 0.46 | 10.1 | 21.6 | 44.2 | 4.25 | 2.60 | 3.36 | 17.0 (6.7) |
| 0.15 m/s, randomised | 0.82 | 0.32 | 10.1 | 15.2 | 34.7 | 3.04 | 1.69 | 2.52 | 12.4 (4.8) |
| 0.25 m/s, nominal model, observation noise off | 0.87 | 0.39 | 7.9 | 17.9 | 37.5 | 3.81 | 2.15 | 3.14 | 14.9 (7.1) |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 1.46 | 0.40 | 17.7 | 21.0 | 47.4 | 5.52 | 3.73 | 4.06 | 19.0 (7.7) |

Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at 20 Hz, gravity excluded, and split at 5 Hz: at or below it the visible bob and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range after removing a 1 s moving average.

## g. Feet (`feet.png`)

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Swing clearance, mm, median (10th pct) | 11.8 (9.3) | 26.2 (23.7) | 2.4 (0.8) | 6.5 (3.4) |
| Swings under 5 mm | 0 % | 0 % | 86 % | 29 % |
| Touchdown speed, m/s, median (95th) | 0.30 (0.44) | 0.30 (0.43) | 0.13 (0.35) | 0.34 (0.44) |
| Peak load rate, body weights/s, median (95th) | 165 (203) | 67 (141) | 67 (179) | 133 (250) |
| Stance slip speed at the sole's origin, mean / 95th pct worst, m/s | 0.07 / 0.20 | 0.05 / 0.14 | 0.10 / 0.45 | 0.06 / 0.23 |
| Slip ratio at the sole's origin, mean (worst) | 0.16 (0.17) | 0.10 (0.10) | 0.25 (0.29) | 0.14 (0.16) |
| Slip ratio at the sole's contacts, mean (worst) | 0.08 (0.09) | 0.07 (0.08) | 0.18 (0.24) | 0.10 (0.12) |

## h. Speed tracking (`speed_tracking.png`)

| Command, m/s | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.27 |
|---|---|---|---|---|---|---|---|
| Speed, m/s (mean of 3 seeds) | 0.002 | 0.044 | 0.097 | 0.148 | 0.197 | 0.244 | 0.264 |
| Falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At 0.25 m/s: mean |step speed - command| 0.013 m/s (control-step speeds swing with the stride, SD 0.016 m/s); |mean speed - command| 0.006 m/s. Sideways drift worst +0.38 m, heading change worst +6.4 deg over 18 s.

## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`

Environment: `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c39...; the run commit's file: -...).

Not applicable: not the policy gait_pipe1.py analysed.

Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: True.

## What this does not show

Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the hotel load is assumed. Head acceleration is the head body's, not a camera's or an onlooker's view of it. Charts: `footfall.png`, `weight_distribution.png`, `balance.png`, `drive_envelopes.png`, `drive_loads.png`, `power.png`, `steadiness.png`, `feet.png`, `speed_tracking.png`.
