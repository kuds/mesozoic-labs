# Gait quality of trike60-fore128-s2 (SIMULATION)

Written by `rl/analytics/gait_report.py` on 2026-11-25. Policy `trike60_10002432_steps.zip` (10,002,432 steps; sha256 e41f7bd92ddd...), trained at commit f84a6919. Deterministic actions, servo delay 10 ms, seeds 100000-100004, randomisation and observation noise on unless stated, 20 s episodes measured from t = 2 s. One policy: every figure is reported, not judged. Definitions: `rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.

## Answer first

- **Speed:** 0.242 m/s at the 0.25 m/s command and 0.144 m/s at 0.15; falls 0 + 0 of 10 (sweep: 0 falls in 21 episodes).
- **Gait:** other (nearest trot, 0.23 cycles off); a diagonal gait whose RF+LH pair follows the LF+RH pair after 0.44 of a stride (0.5 for an even trot); stride 1.22 Hz, 20.1 cm per stride; phase after LF: RF 0.70, LH 0.27, RH 0.09 of a stride.
- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies outside it 37.8 % of the time; the centre of pressure is within 5 mm of its edge 31 % of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives 46 %.)
- **Drives:** worst RMS torque / continuous rating 0.93 (mean of episodes; left_fore_pitch).
- **Power:** 31 W joint mechanical (sum |torque x speed|); leg electrical estimate 60 W on a shared bus to 74 W per drive with no reuse of braking energy; about 68-78 min walking on one battery set with a 30 W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.
- **Steadiness:** body roll 1.3 deg and pitch 0.9 deg SD, height 13 mm range; head bob 36 mm and sway 40 mm (1-99 % range) at 1.2 Hz; head acceleration 5.2 m/s^2 RMS to 20 Hz, of which 3.4 m/s^2 at or below 5 Hz (the visible bob, 0.35 g) and 4.0 m/s^2 at 5-20 Hz (the step impacts' shake).

## Cases (5 episodes each)

| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W | CoM outside support (step contact) | Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.242 | 0 | other | 1.22 | 20.1 | 31.4 | 60-74 | 37.8 % | 77 % | 173 |
| 0.15 m/s, randomised | 0.144 | 0 | other | 1.01 | 14.3 | 15.2 | 38-44 | 18.5 % | 9 % | 65 |
| 0.25 m/s, nominal model, observation noise off | 0.240 | 0 | other | 1.21 | 19.9 | 33.3 | 65-79 | 38.8 % | 2 % | 74 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.259 | 0 | trot | 1.55 | 16.7 | 32.5 | 72-85 | 6.1 % | 17 % | 186 |

Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet shuffle, and the 50 ms step rule splits them less cleanly (README).

## a. Footfalls (`footfall.png`)

Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Duty factor: steps / env debounce / raw | 0.74 / 0.74 / 0.73 | 0.31 / 0.31 / 0.31 | 0.79 / 0.75 / 0.63 | 0.71 / 0.70 / 0.68 |
| Steps per s | 1.17 | 1.22 | 1.78 | 1.20 |
| Touchdowns per s, env debounce (training evaluation's count) | 1.26 | 1.23 | 4.32 | 2.13 |
| Stride frequency, Hz (median period) | 1.20 | 1.21 | 1.67 | 1.21 |
| Stance, ms (median) | 607 | 259 | 446 | 578 |
| Swing, ms (median) | 220 | 571 | 113 | 239 |
| Phase after LF, cycles (R) | 0 | 0.70 (1.00) | 0.27 (0.94) | 0.09 (1.00) |

Gait class: **other** (nearest trot, 0.23 cycles off; largest phase error to the nearest template 0.23 cycles; tolerance 0.125). Per episode: other, other, other, other, other. Diagonal timing: Hildebrand limb phase left 0.73, right 0.61 (trot 0.5); each fore foot lands -0.09 (LF after RH) and +0.43 (RF after LH) cycles after its diagonal hind partner; the RF+LH pair follows the LF+RH pair after 0.44 of a stride (0.5 when even). Stride length 20.1 cm (LF touchdown to LF touchdown). Irregular strides (period more than 25 % off the median): LF 0 %, RF 0 %, LH 77 %, RH 5 %.

Feet down (steps): 4 3 %, 3 49 %, 2 diagonal 41 %, 2 same side 7 %, 2 front 0 %, 2 hind 0 %, 1 0 %, 0 0 %.

Symmetry index (Robinson, %, 0 = symmetric): duty factor: fore 82, hind 11, sides 40; stance duration: fore 81, hind 24, sides 24; stance length: fore 81, hind 26, sides 23; load: fore 97, hind 20, sides 35.

## b. Weight distribution (`weight_distribution.png`)

Floor-truth sole normal forces. Load share LF 35 %, RF 12 %, LH 24 %, RH 29 %: front 48 % / rear 52 %, left 59 % / right 41 %. Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: LF 1.70 / 1.03, RF 1.38 / 0.75, LH 1.67 / 1.02, RH 2.04 / 1.08. 100 strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least 100.0 % of samples.

## c. Balance (`balance.png`)

Support polygon: hull of the capsule soles' contact-line ends of the feet in stance on the step contact the footfalls use (the env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. CoM ground projection outside the polygon 37.8 % of the time (by feet in stance: 1: - %, 2: 78 %, 3: 0 %, 4: 0 %; time with one foot in stance 0.0 %); CoM margin median 8 mm. CoP outside: 0.00 % (it cannot leave the polygon by construction; a check); CoP within 5 mm of the edge 31 %, median margin 10 mm. No foot in stance 0.0 % of the time.

| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |
|---|---|---|---|
| step contact (env 10 ms debounce, then gaps and blips under 50 ms merged) | 37.8 % | 0.0 % | 31 % |
| env 10 ms debounce | 39.1 % | 0.6 % | 33 % |
| raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts) | 45.6 % | 1.1 % | 36 % |

Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot 'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the contact model, not the gait. Nominal model: step contact 38.8 %, env 10 ms debounce 38.9 %, raw 46.5 %.

## d. Drives (`drive_envelopes.png`, `drive_loads.png`)

| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |
|---|---|---|---|---|---|
| left_fore_roll | 0.49 (0.54) | 0.0 % (0.0 %) | 16.8 | 10 | 0.5 % |
| left_fore_pitch | 0.93 (0.97) | 0.0 % (0.0 %) | 20.5 | 66 | 11.5 % |
| left_fore_flexion | 0.76 (0.79) | 0.0 % (0.0 %) | 36.9 | 53 | 2.2 % |
| right_fore_roll | 0.32 (0.34) | 0.0 % (0.0 %) | 14.8 | 7 | 0.3 % |
| right_fore_pitch | 0.56 (0.58) | 0.0 % (0.0 %) | 20.4 | 32 | 3.5 % |
| right_fore_flexion | 0.45 (0.45) | 0.0 % (0.0 %) | 36.2 | 47 | 0.6 % |
| left_hind_roll | 0.41 (0.45) | 0.0 % (0.0 %) | 20.8 | 13 | 1.1 % |
| left_hind_pitch | 0.68 (0.72) | 0.0 % (0.0 %) | 19.6 | 25 | 5.8 % |
| left_hind_flexion | 0.86 (0.90) | 0.2 % (0.7 %) | 18.8 | 37 | 13.3 % |
| right_hind_roll | 0.64 (0.70) | 0.0 % (0.0 %) | 16.0 | 7 | 0.8 % |
| right_hind_pitch | 0.82 (0.87) | 0.0 % (0.0 %) | 20.4 | 41 | 8.7 % |
| right_hind_flexion | 0.75 (0.79) | 0.0 % (0.0 %) | 18.1 | 41 | 4.5 % |

Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): roll: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; pitch: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; front elbow: model limit 37.8, maker peak 54.0, maker rated 18.00, continuous 10.80, speed allowance 108 rpm; hind knee: model limit 18.9, maker peak 27.0, maker rated 9.45, continuous 6.75, speed allowance 83 rpm.

## e. Power and energy (`power.png`)

| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W | CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive | Run time, per drive, 25 / 40 W hotel |
|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 31.4 | 16.6 | 56.5 | 59.9 / 73.6 | 1.05 | 3.00 / 3.46 | 78 / 68 | 71 / 62 |
| 0.15 m/s, randomised | 15.2 | 8.4 | 35.6 | 37.8 / 44.3 | 0.85 | 3.79 / 4.15 | 104 / 95 | 102 / 83 |
| 0.25 m/s, nominal model, observation noise off | 33.3 | 17.6 | 61.2 | 64.6 / 79.3 | 1.11 | 3.15 / 3.63 | 74 / 64 | 67 / 59 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 32.5 | 18.3 | 66.2 | 71.5 / 85.0 | 1.02 | 3.18 / 3.60 | 69 / 61 | 64 / 56 |

Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is 3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step. Not included: the head and tail servos. Hotel load 30 W (range 25-40 W) is an assumption, not a parts-list figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC losses. Battery 146.5 Wh x 80% usable. Torque (copper loss) dominates the electrical figures.

## f. Body and head steadiness (`steadiness.png`)

| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 1.25 | 0.92 | 12.8 | 35.9 | 40.1 | 5.25 | 3.42 | 3.98 | 22.4 (9.6) |
| 0.15 m/s, randomised | 0.82 | 0.57 | 7.4 | 22.1 | 24.9 | 3.21 | 1.94 | 2.56 | 14.3 (6.1) |
| 0.25 m/s, nominal model, observation noise off | 1.20 | 0.93 | 14.3 | 36.9 | 37.9 | 5.06 | 3.60 | 3.56 | 21.7 (9.5) |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 1.12 | 0.52 | 14.3 | 26.5 | 40.7 | 3.94 | 2.57 | 2.99 | 13.5 (5.9) |

Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at 20 Hz, gravity excluded, and split at 5 Hz: at or below it the visible bob and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range after removing a 1 s moving average.

## g. Feet (`feet.png`)

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Swing clearance, mm, median (10th pct) | 14.3 (11.3) | 48.4 (42.4) | 7.6 (1.1) | 16.1 (13.6) |
| Swings under 5 mm | 0 % | 0 % | 35 % | 3 % |
| Touchdown speed, m/s, median (95th) | 0.49 (0.64) | 0.30 (0.41) | 0.14 (0.28) | 0.42 (0.54) |
| Peak load rate, body weights/s, median (95th) | 185 (263) | 111 (163) | 53 (196) | 156 (265) |
| Stance slip speed at the sole's origin, mean / 95th pct worst, m/s | 0.06 / 0.19 | 0.08 / 0.18 | 0.15 / 0.43 | 0.05 / 0.20 |
| Slip ratio at the sole's origin, mean (worst) | 0.18 (0.19) | 0.10 (0.10) | 0.39 (0.44) | 0.13 (0.14) |
| Slip ratio at the sole's contacts, mean (worst) | 0.09 (0.11) | 0.04 (0.04) | 0.34 (0.41) | 0.08 (0.09) |

## h. Speed tracking (`speed_tracking.png`)

| Command, m/s | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.27 |
|---|---|---|---|---|---|---|---|
| Speed, m/s (mean of 3 seeds) | 0.001 | 0.042 | 0.091 | 0.144 | 0.194 | 0.241 | 0.261 |
| Falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At 0.25 m/s: mean |step speed - command| 0.018 m/s (control-step speeds swing with the stride, SD 0.021 m/s); |mean speed - command| 0.008 m/s. Sideways drift worst +0.65 m, heading change worst +11.2 deg over 18 s.

## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`

Environment: `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c39...; the run commit's file: -...).

Not applicable: not the policy gait_pipe1.py analysed.

Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: True.

## What this does not show

Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the hotel load is assumed. Head acceleration is the head body's, not a camera's or an onlooker's view of it. Charts: `footfall.png`, `weight_distribution.png`, `balance.png`, `drive_envelopes.png`, `drive_loads.png`, `power.png`, `steadiness.png`, `feet.png`, `speed_tracking.png`.
