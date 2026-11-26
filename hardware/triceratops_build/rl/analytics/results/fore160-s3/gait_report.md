# Gait quality of trike60-fore160-s3 (SIMULATION)

Written by `rl/analytics/gait_report.py` on 2026-11-26. Policy `trike60_10002432_steps.zip` (10,002,432 steps; sha256 e0dbaf08d3dd...), trained at commit f84a6919. Deterministic actions, servo delay 10 ms, seeds 100000-100004, randomisation and observation noise on unless stated, 20 s episodes measured from t = 2 s. One policy: every figure is reported, not judged. Definitions: `rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.

## Answer first

- **Speed:** 0.239 m/s at the 0.25 m/s command and 0.144 m/s at 0.15; falls 0 + 0 of 10 (sweep: 0 falls in 21 episodes).
- **Gait:** other (phase not locked (RH)); a diagonal gait whose RF+LH pair follows the LF+RH pair after 0.35 of a stride (0.5 for an even trot); stride 0.87 Hz, 28.3 cm per stride; phase after LF: RF 0.27, LH 0.39, RH 0.96 of a stride.
- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies outside it 42.9 % of the time; the centre of pressure is within 5 mm of its edge 30 % of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives 54 %.)
- **Drives:** worst RMS torque / continuous rating 0.88 (mean of episodes; right_fore_pitch).
- **Power:** 29 W joint mechanical (sum |torque x speed|); leg electrical estimate 54 W on a shared bus to 66 W per drive with no reuse of braking energy; about 73-84 min walking on one battery set with a 30 W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.
- **Steadiness:** body roll 1.9 deg and pitch 2.2 deg SD, height 13 mm range; head bob 47 mm and sway 45 mm (1-99 % range) at 0.8 Hz; head acceleration 4.8 m/s^2 RMS to 20 Hz, of which 2.8 m/s^2 at or below 5 Hz (the visible bob, 0.29 g) and 3.9 m/s^2 at 5-20 Hz (the step impacts' shake).

## Cases (5 episodes each)

| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W | CoM outside support (step contact) | Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.239 | 0 | other | 0.87 | 28.3 | 28.8 | 54-66 | 42.9 % | 85 % | 198 |
| 0.15 m/s, randomised | 0.144 | 0 | other | 0.64 | 22.8 | 16.6 | 38-45 | 20.1 % | 73 % | 143 |
| 0.25 m/s, nominal model, observation noise off | 0.237 | 0 | other | 0.82 | 29.1 | 28.7 | 55-67 | 45.2 % | 23 % | 229 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.246 | 0 | other | 0.80 | 31.4 | 28.7 | 58-70 | 3.6 % | 78 % | 219 |

Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet shuffle, and the 50 ms step rule splits them less cleanly (README).

## a. Footfalls (`footfall.png`)

Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Duty factor: steps / env debounce / raw | 0.29 / 0.29 / 0.28 | 0.74 / 0.74 / 0.74 | 0.63 / 0.63 / 0.62 | 0.76 / 0.71 / 0.62 |
| Steps per s | 0.87 | 0.86 | 0.83 | 1.70 |
| Touchdowns per s, env debounce (training evaluation's count) | 1.12 | 0.86 | 0.93 | 4.63 |
| Stride frequency, Hz (median period) | 0.84 | 0.85 | 0.85 | 1.96 |
| Stance, ms (median) | 336 | 884 | 751 | 372 |
| Swing, ms (median) | 851 | 301 | 436 | 100 |
| Phase after LF, cycles (R) | 0 | 0.27 (1.00) | 0.39 (1.00) | 0.96 (0.49) |

Gait class: **other** (phase not locked (RH); largest phase error to the nearest template 0.23 cycles; tolerance 0.125). Per episode: other, other, other, other, other. Diagonal timing: Hildebrand limb phase left 0.61, right 0.31 (trot 0.5); each fore foot lands +0.04 (LF after RH) and -0.12 (RF after LH) cycles after its diagonal hind partner; the RF+LH pair follows the LF+RH pair after 0.35 of a stride (0.5 when even). Stride length 28.3 cm (LF touchdown to LF touchdown). Irregular strides (period more than 25 % off the median): LF 0 %, RF 0 %, LH 0 %, RH 85 %.

Feet down (steps): 4 1 %, 3 41 %, 2 diagonal 48 %, 2 same side 11 %, 2 front 0 %, 2 hind 0 %, 1 0 %, 0 0 %.

Symmetry index (Robinson, %, 0 = symmetric): duty factor: fore 87, hind 19, sides 48; stance duration: fore 89, hind 50, sides 20; stance length: fore 87, hind 50, sides 19; load: fore 103, hind 6, sides 48.

## b. Weight distribution (`weight_distribution.png`)

Floor-truth sole normal forces. Load share LF 12 %, RF 38 %, LH 26 %, RH 24 %: front 50 % / rear 50 %, left 38 % / right 62 %. Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: LF 2.41 / 0.87, RF 2.20 / 0.99, LH 1.74 / 1.12, RH 1.80 / 0.92. 73 strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least 100.0 % of samples.

## c. Balance (`balance.png`)

Support polygon: hull of the capsule soles' contact-line ends of the feet in stance on the step contact the footfalls use (the env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. CoM ground projection outside the polygon 42.9 % of the time (by feet in stance: 1: - %, 2: 68 %, 3: 7 %, 4: 0 %; time with one foot in stance 0.3 %); CoM margin median 3 mm. CoP outside: 0.00 % (it cannot leave the polygon by construction; a check); CoP within 5 mm of the edge 30 %, median margin 8 mm. No foot in stance 0.0 % of the time.

| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |
|---|---|---|---|
| step contact (env 10 ms debounce, then gaps and blips under 50 ms merged) | 42.9 % | 0.3 % | 30 % |
| env 10 ms debounce | 46.2 % | 0.7 % | 31 % |
| raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts) | 53.7 % | 2.0 % | 33 % |

Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot 'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the contact model, not the gait. Nominal model: step contact 45.2 %, env 10 ms debounce 48.0 %, raw 55.2 %.

## d. Drives (`drive_envelopes.png`, `drive_loads.png`)

| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |
|---|---|---|---|---|---|
| left_fore_roll | 0.34 (0.37) | 0.0 % (0.0 %) | 19.1 | 11 | 0.7 % |
| left_fore_pitch | 0.59 (0.61) | 0.0 % (0.0 %) | 20.5 | 35 | 4.4 % |
| left_fore_flexion | 0.48 (0.50) | 0.0 % (0.0 %) | 36.9 | 60 | 2.0 % |
| right_fore_roll | 0.40 (0.43) | 0.0 % (0.0 %) | 14.5 | 7 | 0.1 % |
| right_fore_pitch | 0.88 (0.92) | 0.0 % (0.0 %) | 20.4 | 79 | 9.7 % |
| right_fore_flexion | 0.69 (0.71) | 0.0 % (0.0 %) | 37.1 | 94 | 2.0 % |
| left_hind_roll | 0.37 (0.39) | 0.0 % (0.2 %) | 20.8 | 21 | 0.6 % |
| left_hind_pitch | 0.75 (0.81) | 0.0 % (0.0 %) | 19.6 | 48 | 7.3 % |
| left_hind_flexion | 0.68 (0.73) | 0.2 % (0.6 %) | 18.8 | 50 | 5.8 % |
| right_hind_roll | 0.33 (0.37) | 0.0 % (0.0 %) | 15.9 | 7 | 0.2 % |
| right_hind_pitch | 0.72 (0.77) | 0.0 % (0.0 %) | 20.4 | 35 | 7.0 % |
| right_hind_flexion | 0.74 (0.79) | 0.0 % (0.0 %) | 18.1 | 31 | 8.5 % |

Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): roll: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; pitch: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; front elbow: model limit 37.8, maker peak 54.0, maker rated 18.00, continuous 10.80, speed allowance 108 rpm; hind knee: model limit 18.9, maker peak 27.0, maker rated 9.45, continuous 6.75, speed allowance 83 rpm.

## e. Power and energy (`power.png`)

| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W | CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive | Run time, per drive, 25 / 40 W hotel |
|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 28.8 | 15.3 | 49.9 | 53.9 / 65.9 | 0.97 | 2.83 / 3.23 | 84 / 73 | 77 / 66 |
| 0.15 m/s, randomised | 16.6 | 8.9 | 35.9 | 38.3 / 45.3 | 0.92 | 3.80 / 4.19 | 103 / 93 | 100 / 82 |
| 0.25 m/s, nominal model, observation noise off | 28.7 | 15.3 | 50.7 | 54.8 / 66.7 | 0.96 | 2.84 / 3.24 | 83 / 73 | 77 / 66 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 28.7 | 15.7 | 53.9 | 58.5 / 70.2 | 0.94 | 2.91 / 3.30 | 80 / 70 | 74 / 64 |

Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is 3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step. Not included: the head and tail servos. Hotel load 30 W (range 25-40 W) is an assumption, not a parts-list figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC losses. Battery 146.5 Wh x 80% usable. Torque (copper loss) dominates the electrical figures.

## f. Body and head steadiness (`steadiness.png`)

| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 1.94 | 2.25 | 13.3 | 47.2 | 45.4 | 4.78 | 2.82 | 3.86 | 21.4 (7.4) |
| 0.15 m/s, randomised | 1.76 | 1.77 | 15.3 | 30.4 | 45.6 | 3.29 | 1.96 | 2.64 | 17.4 (6.3) |
| 0.25 m/s, nominal model, observation noise off | 1.89 | 2.27 | 13.3 | 46.6 | 44.5 | 4.82 | 2.85 | 3.89 | 21.8 (6.6) |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.92 | 1.86 | 14.5 | 49.0 | 37.8 | 3.99 | 2.41 | 3.18 | 19.9 (7.0) |

Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at 20 Hz, gravity excluded, and split at 5 Hz: at or below it the visible bob and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range after removing a 1 s moving average.

## g. Feet (`feet.png`)

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Swing clearance, mm, median (10th pct) | 108.9 (103.7) | 45.9 (38.9) | 45.5 (38.8) | 2.7 (0.9) |
| Swings under 5 mm | 0 % | 0 % | 0 % | 79 % |
| Touchdown speed, m/s, median (95th) | 0.47 (0.60) | 0.59 (0.73) | 0.46 (0.51) | 0.10 (0.43) |
| Peak load rate, body weights/s, median (95th) | 106 (182) | 144 (277) | 112 (173) | 76 (181) |
| Stance slip speed at the sole's origin, mean / 95th pct worst, m/s | 0.06 / 0.24 | 0.03 / 0.09 | 0.05 / 0.21 | 0.14 / 0.51 |
| Slip ratio at the sole's origin, mean (worst) | 0.07 (0.09) | 0.09 (0.09) | 0.14 (0.15) | 0.37 (0.45) |
| Slip ratio at the sole's contacts, mean (worst) | 0.07 (0.09) | 0.06 (0.07) | 0.07 (0.07) | 0.31 (0.40) |

## h. Speed tracking (`speed_tracking.png`)

| Command, m/s | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.27 |
|---|---|---|---|---|---|---|---|
| Speed, m/s (mean of 3 seeds) | 0.001 | 0.049 | 0.097 | 0.144 | 0.192 | 0.239 | 0.256 |
| Falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At 0.25 m/s: mean |step speed - command| 0.018 m/s (control-step speeds swing with the stride, SD 0.019 m/s); |mean speed - command| 0.011 m/s. Sideways drift worst +0.54 m, heading change worst +8.2 deg over 18 s.

## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`

Environment: `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c39...; the run commit's file: -...).

Not applicable: not the policy gait_pipe1.py analysed.

Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: True.

## What this does not show

Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the hotel load is assumed. Head acceleration is the head body's, not a camera's or a visitor's view of it. Charts: `footfall.png`, `weight_distribution.png`, `balance.png`, `drive_envelopes.png`, `drive_loads.png`, `power.png`, `steadiness.png`, `feet.png`, `speed_tracking.png`.
