# Gait quality of trike60-fore128-s0 (SIMULATION)

Written by `rl/analytics/gait_report.py` on 2026-11-24. Policy `trike60_10002432_steps.zip` (10,002,432 steps; sha256 f7bbeb814d0d...), trained at commit f84a6919. Deterministic actions, servo delay 10 ms, seeds 100000-100004, randomisation and observation noise on unless stated, 20 s episodes measured from t = 2 s. One policy: every figure is reported, not judged. Definitions: `rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.

## Answer first

- **Speed:** 0.235 m/s at the 0.25 m/s command and 0.143 m/s at 0.15; falls 0 + 0 of 10 (sweep: 0 falls in 21 episodes).
- **Gait:** trot (trot); a diagonal gait whose RF+LH pair follows the LF+RH pair after 0.52 of a stride (0.5 for an even trot); stride 0.68 Hz, 36.2 cm per stride; phase after LF: RF 0.56, LH 0.54, RH 0.06 of a stride.
- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies outside it 61.8 % of the time; the centre of pressure is within 5 mm of its edge 33 % of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives 66 %.)
- **Drives:** worst RMS torque / continuous rating 0.83 (mean of episodes; left_hind_pitch).
- **Power:** 24 W joint mechanical (sum |torque x speed|); leg electrical estimate 50 W on a shared bus to 60 W per drive with no reuse of braking energy; about 78-88 min walking on one battery set with a 30 W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.
- **Steadiness:** body roll 2.9 deg and pitch 2.5 deg SD, height 41 mm range; head bob 71 mm and sway 73 mm (1-99 % range) at 1.3 Hz; head acceleration 4.5 m/s^2 RMS to 20 Hz, of which 2.5 m/s^2 at or below 5 Hz (the visible bob, 0.25 g) and 3.7 m/s^2 at 5-20 Hz (the step impacts' shake).

## Cases (5 episodes each)

| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W | CoM outside support (step contact) | Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.235 | 0 | trot | 0.68 | 36.2 | 23.9 | 50-60 | 61.8 % | 65 % | 119 |
| 0.15 m/s, randomised | 0.143 | 0 | other | 0.73 | 31.0 | 14.1 | 37-43 | 50.5 % | 80 % | 195 |
| 0.25 m/s, nominal model, observation noise off | 0.233 | 0 | trot | 0.66 | 37.0 | 24.5 | 53-64 | 61.3 % | 63 % | 135 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.236 | 0 | other | 0.68 | 35.1 | 26.7 | 64-75 | 6.7 % | 44 % | 47 |

Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet shuffle, and the 50 ms step rule splits them less cleanly (README).

## a. Footfalls (`footfall.png`)

Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Duty factor: steps / env debounce / raw | 0.57 / 0.57 / 0.57 | 0.45 / 0.45 / 0.45 | 0.57 / 0.55 / 0.52 | 0.58 / 0.58 / 0.56 |
| Steps per s | 0.63 | 0.66 | 0.81 | 0.99 |
| Touchdowns per s, env debounce (training evaluation's count) | 0.73 | 0.66 | 3.03 | 1.44 |
| Stride frequency, Hz (median period) | 0.64 | 0.64 | 0.68 | 0.91 |
| Stance, ms (median) | 906 | 722 | 828 | 791 |
| Swing, ms (median) | 670 | 857 | 592 | 378 |
| Phase after LF, cycles (R) | 0 | 0.56 (0.99) | 0.54 (0.62) | 0.06 (1.00) |

Gait class: **trot** (trot; largest phase error to the nearest template 0.06 cycles; tolerance 0.125). Per episode: trot, trot, trot, other, other. Diagonal timing: Hildebrand limb phase left 0.46, right 0.50 (trot 0.5); each fore foot lands -0.06 (LF after RH) and +0.02 (RF after LH) cycles after its diagonal hind partner; the RF+LH pair follows the LF+RH pair after 0.52 of a stride (0.5 when even). Stride length 36.2 cm (LF touchdown to LF touchdown). Irregular strides (period more than 25 % off the median): LF 0 %, RF 0 %, LH 29 %, RH 65 %.

Feet down (steps): 4 0 %, 3 17 %, 2 diagonal 77 %, 2 same side 6 %, 2 front 0 %, 2 hind 0 %, 1 0 %, 0 0 %.

Symmetry index (Robinson, %, 0 = symmetric): duty factor: fore 23, hind 3, sides 10; stance duration: fore 24, hind 21, sides 22; stance length: fore 25, hind 21, sides 23; load: fore 27, hind 8, sides 10.

## b. Weight distribution (`weight_distribution.png`)

Floor-truth sole normal forces. Load share LF 29 %, RF 22 %, LH 23 %, RH 25 %: front 52 % / rear 48 %, left 52 % / right 48 %. Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: LF 1.95 / 1.13, RF 2.64 / 1.05, LH 1.85 / 0.91, RH 1.59 / 0.91. 52 strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least 100.0 % of samples.

## c. Balance (`balance.png`)

Support polygon: hull of the capsule soles' contact-line ends of the feet in stance on the step contact the footfalls use (the env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. CoM ground projection outside the polygon 61.8 % of the time (by feet in stance: 1: - %, 2: 73 %, 3: 10 %, 4: 0 %; time with one foot in stance 0.0 %); CoM margin median -5 mm. CoP outside: 0.00 % (it cannot leave the polygon by construction; a check); CoP within 5 mm of the edge 33 %, median margin 7 mm. No foot in stance 0.0 % of the time.

| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |
|---|---|---|---|
| step contact (env 10 ms debounce, then gaps and blips under 50 ms merged) | 61.8 % | 0.0 % | 33 % |
| env 10 ms debounce | 64.0 % | 0.2 % | 35 % |
| raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts) | 65.7 % | 1.2 % | 36 % |

Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot 'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the contact model, not the gait. Nominal model: step contact 61.3 %, env 10 ms debounce 64.0 %, raw 65.4 %.

## d. Drives (`drive_envelopes.png`, `drive_loads.png`)

| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |
|---|---|---|---|---|---|
| left_fore_roll | 0.50 (0.52) | 0.0 % (0.0 %) | 18.5 | 8 | 0.9 % |
| left_fore_pitch | 0.69 (0.71) | 0.0 % (0.0 %) | 20.5 | 35 | 5.9 % |
| left_fore_flexion | 0.64 (0.66) | 0.0 % (0.0 %) | 36.9 | 77 | 2.5 % |
| right_fore_roll | 0.26 (0.28) | 0.0 % (0.0 %) | 14.7 | 9 | 0.0 % |
| right_fore_pitch | 0.67 (0.72) | 0.0 % (0.0 %) | 20.4 | 29 | 5.5 % |
| right_fore_flexion | 0.40 (0.41) | 0.0 % (0.0 %) | 37.1 | 56 | 1.1 % |
| left_hind_roll | 0.32 (0.35) | 0.0 % (0.0 %) | 18.3 | 11 | 0.2 % |
| left_hind_pitch | 0.83 (0.88) | 0.0 % (0.0 %) | 19.6 | 32 | 10.4 % |
| left_hind_flexion | 0.55 (0.60) | 0.1 % (0.4 %) | 18.8 | 53 | 4.2 % |
| right_hind_roll | 0.55 (0.58) | 0.0 % (0.0 %) | 18.4 | 7 | 0.1 % |
| right_hind_pitch | 0.61 (0.66) | 0.0 % (0.0 %) | 20.4 | 23 | 4.1 % |
| right_hind_flexion | 0.82 (0.87) | 0.0 % (0.0 %) | 18.1 | 28 | 9.8 % |

Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): roll: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; pitch: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; front elbow: model limit 37.8, maker peak 54.0, maker rated 18.00, continuous 10.80, speed allowance 108 rpm; hind knee: model limit 18.9, maker peak 27.0, maker rated 9.45, continuous 6.75, speed allowance 83 rpm.

## e. Power and energy (`power.png`)

| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W | CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive | Run time, per drive, 25 / 40 W hotel |
|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 23.9 | 12.8 | 46.9 | 49.8 / 60.3 | 0.82 | 2.74 / 3.10 | 88 / 78 | 82 / 70 |
| 0.15 m/s, randomised | 14.1 | 7.6 | 35.4 | 37.3 / 43.4 | 0.80 | 3.81 / 4.15 | 105 / 96 | 103 / 84 |
| 0.25 m/s, nominal model, observation noise off | 24.5 | 13.2 | 50.0 | 53.1 / 64.0 | 0.84 | 2.85 / 3.23 | 85 / 75 | 79 / 68 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 26.7 | 14.9 | 59.3 | 63.8 / 75.0 | 0.91 | 3.22 / 3.60 | 75 / 67 | 71 / 61 |

Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is 3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step. Not included: the head and tail servos. Hotel load 30 W (range 25-40 W) is an assumption, not a parts-list figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC losses. Battery 146.5 Wh x 80% usable. Torque (copper loss) dominates the electrical figures.

## f. Body and head steadiness (`steadiness.png`)

| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 2.93 | 2.51 | 40.7 | 70.8 | 72.5 | 4.49 | 2.48 | 3.74 | 25.8 (8.3) |
| 0.15 m/s, randomised | 3.09 | 2.41 | 45.4 | 49.8 | 66.4 | 3.16 | 1.73 | 2.64 | 20.4 (6.5) |
| 0.25 m/s, nominal model, observation noise off | 3.01 | 2.59 | 40.6 | 71.0 | 72.9 | 4.24 | 2.41 | 3.49 | 24.7 (7.5) |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 2.53 | 2.02 | 47.7 | 77.0 | 84.6 | 4.31 | 2.77 | 3.30 | 21.3 (8.3) |

Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at 20 Hz, gravity excluded, and split at 5 Hz: at or below it the visible bob and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range after removing a 1 s moving average.

## g. Feet (`feet.png`)

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Swing clearance, mm, median (10th pct) | 83.9 (69.0) | 95.1 (81.0) | 11.2 (2.6) | 10.2 (5.3) |
| Swings under 5 mm | 0 % | 0 % | 19 % | 9 % |
| Touchdown speed, m/s, median (95th) | 0.66 (0.78) | 0.66 (0.77) | 0.38 (0.52) | 0.22 (0.41) |
| Peak load rate, body weights/s, median (95th) | 133 (214) | 122 (234) | 174 (269) | 70 (149) |
| Stance slip speed at the sole's origin, mean / 95th pct worst, m/s | 0.04 / 0.12 | 0.04 / 0.12 | 0.07 / 0.44 | 0.05 / 0.31 |
| Slip ratio at the sole's origin, mean (worst) | 0.09 (0.09) | 0.07 (0.07) | 0.16 (0.19) | 0.13 (0.14) |
| Slip ratio at the sole's contacts, mean (worst) | 0.06 (0.06) | 0.05 (0.06) | 0.13 (0.16) | 0.08 (0.10) |

## h. Speed tracking (`speed_tracking.png`)

| Command, m/s | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.27 |
|---|---|---|---|---|---|---|---|
| Speed, m/s (mean of 3 seeds) | 0.001 | 0.047 | 0.095 | 0.142 | 0.189 | 0.235 | 0.255 |
| Falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At 0.25 m/s: mean |step speed - command| 0.018 m/s (control-step speeds swing with the stride, SD 0.016 m/s); |mean speed - command| 0.015 m/s. Sideways drift worst +0.64 m, heading change worst +9.2 deg over 18 s.

## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`

Environment: `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c39...; the run commit's file: -...).

Not applicable: not the policy gait_pipe1.py analysed.

Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: True.

## What this does not show

Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the hotel load is assumed. Head acceleration is the head body's, not a camera's or a visitor's view of it. Charts: `footfall.png`, `weight_distribution.png`, `balance.png`, `drive_envelopes.png`, `drive_loads.png`, `power.png`, `steadiness.png`, `feet.png`, `speed_tracking.png`.
