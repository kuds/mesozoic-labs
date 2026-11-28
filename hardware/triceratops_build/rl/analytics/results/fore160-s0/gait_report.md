# Gait quality of trike60-fore160-s0 (SIMULATION)

Written by `rl/analytics/gait_report.py` on 2026-11-24. Policy `trike60_10002432_steps.zip` (10,002,432 steps; sha256 0eb877a99151...), trained at commit f84a6919. Deterministic actions, servo delay 10 ms, seeds 100000-100004, randomisation and observation noise on unless stated, 20 s episodes measured from t = 2 s. One policy: every figure is reported, not judged. Definitions: `rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.

## Answer first

- **Speed:** 0.242 m/s at the 0.25 m/s command and 0.148 m/s at 0.15; falls 0 + 0 of 10 (sweep: 0 falls in 21 episodes).
- **Gait:** other (phase not locked (LH)); a diagonal gait whose RF+LH pair follows the LF+RH pair after 0.61 of a stride (0.5 for an even trot); stride 0.73 Hz, 33.5 cm per stride; phase after LF: RF 0.65, LH 0.63, RH 0.05 of a stride.
- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies outside it 63.0 % of the time; the centre of pressure is within 5 mm of its edge 30 % of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives 62 %.)
- **Drives:** worst RMS torque / continuous rating 1.01 (mean of episodes; left_fore_pitch).
- **Power:** 37 W joint mechanical (sum |torque x speed|); leg electrical estimate 77 W on a shared bus to 92 W per drive with no reuse of braking energy; about 58-66 min walking on one battery set with a 30 W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.
- **Steadiness:** body roll 2.4 deg and pitch 2.1 deg SD, height 37 mm range; head bob 51 mm and sway 53 mm (1-99 % range) at 1.3 Hz; head acceleration 5.7 m/s^2 RMS to 20 Hz, of which 2.7 m/s^2 at or below 5 Hz (the visible bob, 0.28 g) and 5.0 m/s^2 at 5-20 Hz (the step impacts' shake).

## Cases (5 episodes each)

| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W | CoM outside support (step contact) | Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 0.242 | 0 | other | 0.73 | 33.5 | 36.9 | 77-92 | 63.0 % | 72 % | 149 |
| 0.15 m/s, randomised | 0.148 | 0 | other | 0.54 | 30.0 | 21.0 | 59-68 | 33.3 % | 77 % | 220 |
| 0.25 m/s, nominal model, observation noise off | 0.242 | 0 | other | 0.71 | 34.0 | 37.0 | 82-98 | 69.2 % | 21 % | 108 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 0.250 | 0 | other | 0.77 | 33.3 | 39.7 | 89-105 | 6.2 % | 43 % | 182 |

Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet shuffle, and the 50 ms step rule splits them less cleanly (README).

## a. Footfalls (`footfall.png`)

Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Duty factor: steps / env debounce / raw | 0.66 / 0.66 / 0.66 | 0.36 / 0.36 / 0.35 | 0.48 / 0.51 / 0.47 | 0.65 / 0.64 / 0.63 |
| Steps per s | 0.71 | 0.73 | 1.18 | 0.72 |
| Touchdowns per s, env debounce (training evaluation's count) | 0.79 | 0.81 | 4.20 | 1.01 |
| Stride frequency, Hz (median period) | 0.72 | 0.72 | 1.32 | 0.72 |
| Stance, ms (median) | 910 | 500 | 499 | 890 |
| Swing, ms (median) | 471 | 880 | 514 | 487 |
| Phase after LF, cycles (R) | 0 | 0.65 (0.99) | 0.63 (0.28) | 0.05 (1.00) |

Gait class: **other** (phase not locked (LH); largest phase error to the nearest template 0.15 cycles; tolerance 0.125). Per episode: other, other, other, other, other. Diagonal timing: Hildebrand limb phase left 0.37, right 0.60 (trot 0.5); each fore foot lands -0.05 (LF after RH) and +0.01 (RF after LH) cycles after its diagonal hind partner; the RF+LH pair follows the LF+RH pair after 0.61 of a stride (0.5 when even). Stride length 33.5 cm (LF touchdown to LF touchdown). Irregular strides (period more than 25 % off the median): LF 0 %, RF 0 %, LH 72 %, RH 2 %.

Feet down (steps): 4 0 %, 3 15 %, 2 diagonal 79 %, 2 same side 5 %, 2 front 0 %, 2 hind 0 %, 1 0 %, 0 0 %.

Symmetry index (Robinson, %, 0 = symmetric): duty factor: fore 58, hind 30, sides 12; stance duration: fore 57, hind 73, sides 5; stance length: fore 56, hind 71, sides 5; load: fore 62, hind 27, sides 18.

## b. Weight distribution (`weight_distribution.png`)

Floor-truth sole normal forces. Load share LF 33 %, RF 17 %, LH 22 %, RH 28 %: front 50 % / rear 50 %, left 54 % / right 46 %. Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: LF 1.89 / 1.10, RF 1.94 / 0.99, LH 2.09 / 1.08, RH 1.39 / 0.85. 59 strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least 100.0 % of samples.

## c. Balance (`balance.png`)

Support polygon: hull of the capsule soles' contact-line ends of the feet in stance on the step contact the footfalls use (the env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. CoM ground projection outside the polygon 63.0 % of the time (by feet in stance: 1: - %, 2: 74 %, 3: 2 %, 4: 0 %; time with one foot in stance 0.0 %); CoM margin median -9 mm. CoP outside: 0.00 % (it cannot leave the polygon by construction; a check); CoP within 5 mm of the edge 30 %, median margin 8 mm. No foot in stance 0.0 % of the time.

| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |
|---|---|---|---|
| step contact (env 10 ms debounce, then gaps and blips under 50 ms merged) | 63.0 % | 0.0 % | 30 % |
| env 10 ms debounce | 60.4 % | 0.3 % | 28 % |
| raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts) | 62.0 % | 2.6 % | 28 % |

Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot 'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the contact model, not the gait. Nominal model: step contact 69.2 %, env 10 ms debounce 62.5 %, raw 62.7 %.

## d. Drives (`drive_envelopes.png`, `drive_loads.png`)

| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |
|---|---|---|---|---|---|
| left_fore_roll | 0.73 (0.78) | 0.0 % (0.0 %) | 18.4 | 8 | 3.6 % |
| left_fore_pitch | 1.01 (1.04) | 0.0 % (0.0 %) | 20.5 | 100 | 16.8 % |
| left_fore_flexion | 0.74 (0.77) | 0.0 % (0.0 %) | 36.9 | 116 | 3.3 % |
| right_fore_roll | 0.42 (0.46) | 0.0 % (0.0 %) | 16.7 | 12 | 0.5 % |
| right_fore_pitch | 0.76 (0.79) | 0.0 % (0.0 %) | 20.4 | 75 | 7.6 % |
| right_fore_flexion | 0.61 (0.62) | 0.0 % (0.0 %) | 37.1 | 62 | 2.2 % |
| left_hind_roll | 0.46 (0.51) | 0.0 % (0.0 %) | 20.8 | 12 | 0.4 % |
| left_hind_pitch | 0.67 (0.71) | 0.0 % (0.0 %) | 19.6 | 44 | 6.2 % |
| left_hind_flexion | 0.99 (1.05) | 0.7 % (2.1 %) | 18.8 | 95 | 20.7 % |
| right_hind_roll | 0.90 (0.95) | 0.0 % (0.0 %) | 20.9 | 11 | 12.0 % |
| right_hind_pitch | 0.74 (0.77) | 0.0 % (0.0 %) | 20.4 | 34 | 7.7 % |
| right_hind_flexion | 0.59 (0.62) | 0.0 % (0.0 %) | 18.1 | 32 | 1.1 % |

Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): roll: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; pitch: model limit 21.0, maker peak 30.0, maker rated 10.50, continuous 7.50, speed allowance 83 rpm; front elbow: model limit 37.8, maker peak 54.0, maker rated 18.00, continuous 10.80, speed allowance 108 rpm; hind knee: model limit 18.9, maker peak 27.0, maker rated 9.45, continuous 6.75, speed allowance 83 rpm.

## e. Power and energy (`power.png`)

| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W | CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive | Run time, per drive, 25 / 40 W hotel |
|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 36.9 | 19.0 | 72.6 | 76.5 / 92.4 | 1.22 | 3.53 / 4.06 | 66 / 58 | 60 / 53 |
| 0.15 m/s, randomised | 21.0 | 11.1 | 56.5 | 59.1 / 68.1 | 1.13 | 4.82 / 5.30 | 79 / 72 | 76 / 65 |
| 0.25 m/s, nominal model, observation noise off | 37.0 | 19.0 | 78.2 | 82.2 / 98.1 | 1.22 | 3.68 / 4.21 | 63 / 55 | 57 / 51 |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 39.7 | 22.0 | 81.7 | 88.8 / 104.7 | 1.29 | 3.86 / 4.37 | 60 / 53 | 55 / 49 |

Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is 3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step. Not included: the head and tail servos. Hotel load 30 W (range 25-40 W) is an assumption, not a parts-list figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC losses. Battery 146.5 Wh x 80% usable. Torque (copper loss) dominates the electrical figures.

## f. Body and head steadiness (`steadiness.png`)

| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 m/s (target), randomised | 2.37 | 2.07 | 37.3 | 51.4 | 53.5 | 5.66 | 2.72 | 4.96 | 30.5 (8.7) |
| 0.15 m/s, randomised | 2.33 | 2.01 | 33.7 | 38.2 | 39.6 | 3.78 | 1.55 | 3.44 | 25.6 (6.3) |
| 0.25 m/s, nominal model, observation noise off | 2.34 | 2.03 | 36.4 | 49.3 | 48.9 | 5.42 | 2.52 | 4.79 | 28.6 (8.0) |
| 0.25 m/s, other foot (trike60_C_closed.xml: PR #669's pitch-only ankle), randomised | 2.71 | 1.67 | 37.0 | 52.1 | 54.8 | 4.16 | 2.54 | 3.30 | 23.4 (7.6) |

Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at 20 Hz, gravity excluded, and split at 5 Hz: at or below it the visible bob and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range after removing a 1 s moving average.

## g. Feet (`feet.png`)

| | LF | RF | LH | RH |
|---|---|---|---|---|
| Swing clearance, mm, median (10th pct) | 107.7 (102.7) | 116.5 (110.3) | 11.5 (1.2) | 25.4 (19.1) |
| Swings under 5 mm | 0 % | 0 % | 26 % | 2 % |
| Touchdown speed, m/s, median (95th) | 0.96 (1.19) | 0.89 (1.12) | 0.54 (0.91) | 0.47 (0.64) |
| Peak load rate, body weights/s, median (95th) | 145 (265) | 127 (264) | 142 (324) | 86 (144) |
| Stance slip speed at the sole's origin, mean / 95th pct worst, m/s | 0.05 / 0.19 | 0.07 / 0.28 | 0.11 / 0.49 | 0.06 / 0.21 |
| Slip ratio at the sole's origin, mean (worst) | 0.13 (0.15) | 0.10 (0.12) | 0.22 (0.29) | 0.16 (0.18) |
| Slip ratio at the sole's contacts, mean (worst) | 0.09 (0.11) | 0.09 (0.11) | 0.17 (0.25) | 0.09 (0.11) |

## h. Speed tracking (`speed_tracking.png`)

| Command, m/s | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.27 |
|---|---|---|---|---|---|---|---|
| Speed, m/s (mean of 3 seeds) | -0.000 | 0.049 | 0.100 | 0.149 | 0.196 | 0.242 | 0.259 |
| Falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At 0.25 m/s: mean |step speed - command| 0.016 m/s (control-step speeds swing with the stride, SD 0.017 m/s); |mean speed - command| 0.008 m/s. Sideways drift worst +0.87 m, heading change worst +13.9 deg over 18 s.

## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`

Environment: `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c39...; the run commit's file: -...).

Not applicable: not the policy gait_pipe1.py analysed.

Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: True.

## What this does not show

Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the hotel load is assumed. Head acceleration is the head body's, not a camera's or an onlooker's view of it. Charts: `footfall.png`, `weight_distribution.png`, `balance.png`, `drive_envelopes.png`, `drive_loads.png`, `power.png`, `steadiness.png`, `feet.png`, `speed_tracking.png`.
