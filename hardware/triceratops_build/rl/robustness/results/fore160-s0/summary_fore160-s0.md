# Robustness sweep: trike60-fore160-s0 (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-24. One policy, so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Policy `trike60_10002432_steps.zip` (sha256 0eb877a99151d114...), run commit f84a6919; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c397e0c...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 12.1 min on 2 worker processes (565 episodes).
- Reproduction check (gait_pipe1's target case, randomisation on, seeds 100000-100004): mean speed 0.2420 m/s against the analysis's 0.2385 m/s; per-episode speeds identical: False.

## Headline

- **No fall and no speed drop of more than 20 %** in: delay, 0.15 m/s (0 ms to 60 ms); friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5); step up (flat to step 3 cm); ramp up (flat to ramp 8 deg).
- **delay, 0.25 m/s**: first break at 60 ms (speed 0.151 m/s, 38 % below nominal).
- **Sideways push**: falls start at 1 m/s free-body (impulse 12.8 N s; the centre of mass gained 0.58 m/s), pushed to the left (1/8 fell); 1.25 m/s free-body (impulse 16.0 N s; the centre of mass gained 0.67 m/s), pushed to the right (4/8 fell). Largest push with no fall: 0.75 m/s free-body (impulse 9.6 N s; the centre of mass gained 0.40 m/s), pushed to the left; 1 m/s free-body (impulse 12.8 N s; the centre of mass gained 0.51 m/s), pushed to the right. The feet absorb part of every push: the centre of mass gains 42-66 % of the free-body velocity change (pushes of 0.3 m/s and more).

## Where it breaks

First point outward from nominal with a fall or a speed more than 20 % below the nominal point's (same command). "none": no such point in the tested range.

| Sweep | Nominal | Low side | High side |
|---|---|---|---|
| delay, 0.25 m/s | 10 ms | none | **60 ms**: speed 0.151 m/s, 38 % below nominal |
| delay, 0.15 m/s | 10 ms | none | none |
| friction | 0.85 (model) | none | none |
| mass, every body | x1 | none | none |
| payload on the chassis | +0 kg | (not tested) | none |
| force limit | x1 | none | (not tested) |
| position gain kp | x1 | none | none |
| sideways push | none | **1.25 m/s free-body to the right (impulse 16.0 N s; the centre of mass gained 0.67 m/s)**: 4/8 fell | **1 m/s free-body to the left (impulse 12.8 N s; the centre of mass gained 0.58 m/s)**: 1/8 fell |
| step up | flat | (not tested) | none |
| ramp up | flat | (not tested) | none |

Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.254 (0.253) | +4.2 % | +0.004 | 0.369 (0.465) | 0.25 (left_hind) | 1.12 (left_fore_pitch) | 4.4 % (left_fore_pitch) | 4.4 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.251 (0.250) | +2.8 % | +0.001 | 0.404 (0.522) | 0.23 (left_hind) | 1.10 (left_fore_pitch) | 4.1 % (left_fore_pitch) | 4.1 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.247 (0.246) | +1.3 % | -0.003 | 0.431 (0.559) | 0.20 (left_hind) | 1.08 (left_fore_pitch) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.244 (0.243) | +0.0 % | -0.006 | 0.500 (0.599) | 0.18 (left_hind) | 1.04 (left_fore_pitch) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |
| 15 ms | 0/8 | 0.245 (0.244) | +0.6 % | -0.005 | 0.567 (0.667) | 0.18 (left_hind) | 1.05 (left_hind_flexion) | 4.1 % (left_fore_pitch) | 4.1 % (left_fore_pitch) |
| 20 ms | 0/8 | 0.265 (0.263) | +8.9 % | +0.015 | 0.640 (0.747) | 0.22 (right_hind) | 1.29 (right_hind_pitch) | 12.3 % (right_hind_pitch) | 12.3 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.264 (0.262) | +8.5 % | +0.014 | 0.767 (0.921) | 0.22 (right_hind) | 1.36 (right_hind_pitch) | 15.1 % (right_hind_pitch) | 15.1 % (right_hind_pitch) |
| 40 ms | 0/8 | 0.253 (0.235) | +3.9 % | +0.003 | 2.245 (2.691) | 0.25 (left_hind) | 1.38 (right_hind_pitch) | 16.1 % (right_hind_pitch) | 16.1 % (right_hind_pitch) |
| 60 ms | 0/8 | 0.151 (0.123) | -38.2 % | -0.099 | 2.558 (3.075) | 0.46 (left_hind) | 1.33 (left_hind_pitch) | 12.9 % (right_hind_pitch) | 12.9 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore160-s0_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.156 (0.155) | +3.9 % | +0.006 | 0.125 (0.202) | 0.36 (left_hind) | 1.00 (left_fore_pitch) | 1.8 % (left_fore_pitch) | 1.8 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.153 (0.152) | +2.4 % | +0.003 | 0.126 (0.242) | 0.35 (left_hind) | 0.99 (left_fore_pitch) | 1.7 % (left_fore_pitch) | 1.7 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.151 (0.150) | +0.9 % | +0.001 | 0.139 (0.225) | 0.33 (left_hind) | 0.98 (left_fore_pitch) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.150 (0.149) | +0.0 % | -0.000 | 0.159 (0.208) | 0.30 (left_hind) | 0.96 (left_fore_pitch) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| 15 ms | 0/8 | 0.152 (0.151) | +1.5 % | +0.002 | 0.197 (0.250) | 0.27 (left_hind) | 0.94 (left_fore_pitch) | 1.8 % (left_fore_pitch) | 1.8 % (left_fore_pitch) |
| 20 ms | 0/8 | 0.163 (0.159) | +8.6 % | +0.013 | 0.330 (0.450) | 0.30 (right_hind) | 1.53 (right_hind_pitch) | 18.8 % (right_hind_pitch) | 18.8 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.167 (0.165) | +11.3 % | +0.017 | 0.474 (0.583) | 0.35 (right_hind) | 1.58 (right_hind_pitch) | 22.1 % (right_hind_pitch) | 22.1 % (right_hind_pitch) |
| 40 ms | 0/8 | 0.188 (0.181) | +25.3 % | +0.038 | 1.613 (1.992) | 0.34 (right_hind) | 1.53 (right_hind_pitch) | 22.3 % (right_hind_pitch) | 22.3 % (right_hind_pitch) |
| 60 ms | 0/8 | 0.178 (0.167) | +18.8 % | +0.028 | 1.931 (2.309) | 0.37 (left_hind) | 1.53 (right_hind_pitch) | 23.3 % (right_hind_pitch) | 23.3 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore160-s0_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.241 (0.238) | -1.4 % | -0.009 | 1.052 (1.150) | 0.49 (left_hind) | 0.95 (left_fore_pitch) | 2.6 % (left_fore_pitch) | 2.6 % (left_fore_pitch) |
| 0.2 | 0/8 | 0.240 (0.238) | -1.5 % | -0.010 | 0.998 (1.101) | 0.44 (left_hind) | 0.95 (left_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |
| 0.3 | 0/8 | 0.240 (0.239) | -1.6 % | -0.010 | 0.825 (0.921) | 0.41 (left_hind) | 0.96 (left_fore_pitch) | 2.9 % (right_fore_pitch) | 2.9 % (right_fore_pitch) |
| 0.4 | 0/8 | 0.240 (0.239) | -1.5 % | -0.010 | 0.690 (0.800) | 0.39 (left_hind) | 0.98 (left_fore_pitch) | 3.2 % (left_fore_pitch) | 3.2 % (left_fore_pitch) |
| 0.5 | 0/8 | 0.241 (0.240) | -1.3 % | -0.009 | 0.607 (0.697) | 0.36 (left_hind) | 1.00 (left_fore_pitch) | 3.6 % (left_fore_pitch) | 3.6 % (left_fore_pitch) |
| 0.6 | 0/8 | 0.241 (0.241) | -1.0 % | -0.009 | 0.581 (0.684) | 0.27 (left_hind) | 1.02 (left_fore_pitch) | 3.7 % (left_fore_pitch) | 3.7 % (left_fore_pitch) |
| 0.8 | 0/8 | 0.243 (0.243) | -0.2 % | -0.007 | 0.524 (0.602) | 0.18 (left_hind) | 1.04 (left_fore_pitch) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |
| 0.85 (model) | 0/8 | 0.244 (0.243) | +0.0 % | -0.006 | 0.500 (0.599) | 0.18 (left_hind) | 1.04 (left_fore_pitch) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |
| 1 | 0/8 | 0.245 (0.245) | +0.7 % | -0.005 | 0.430 (0.531) | 0.18 (left_hind) | 1.06 (left_fore_pitch) | 4.0 % (left_fore_pitch) | 4.0 % (left_fore_pitch) |
| 1.2 | 0/8 | 0.247 (0.246) | +1.2 % | -0.003 | 0.380 (0.493) | 0.18 (left_hind) | 1.07 (left_fore_pitch) | 4.1 % (left_fore_pitch) | 4.1 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `fore160-s0_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.243 (0.242) | -0.4 % | -0.007 | 0.574 (0.672) | 0.19 (right_hind) | 0.94 (left_fore_pitch) | 2.9 % (left_fore_pitch) | 2.9 % (left_fore_pitch) |
| x0.9 | 0/8 | 0.243 (0.242) | -0.3 % | -0.007 | 0.542 (0.638) | 0.18 (left_hind) | 0.99 (left_fore_pitch) | 3.3 % (left_fore_pitch) | 3.3 % (left_fore_pitch) |
| x1 | 0/8 | 0.244 (0.243) | +0.0 % | -0.006 | 0.500 (0.599) | 0.18 (left_hind) | 1.04 (left_fore_pitch) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |
| x1.1 | 0/8 | 0.245 (0.243) | +0.4 % | -0.005 | 0.449 (0.548) | 0.18 (left_hind) | 1.12 (left_hind_flexion) | 4.4 % (left_fore_pitch) | 4.4 % (left_fore_pitch) |
| x1.2 | 0/8 | 0.246 (0.244) | +0.7 % | -0.004 | 0.416 (0.532) | 0.18 (left_hind) | 1.20 (left_hind_flexion) | 5.0 % (left_fore_pitch) | 5.0 % (left_fore_pitch) |
| x1.4 | 0/8 | 0.246 (0.245) | +1.0 % | -0.004 | 0.382 (0.475) | 0.18 (left_hind) | 1.35 (left_hind_flexion) | 7.7 % (left_hind_flexion) | 7.7 % (left_hind_flexion) |
| x1.6 | 0/8 | 0.245 (0.244) | +0.5 % | -0.005 | 0.420 (0.523) | 0.19 (left_hind) | 1.50 (left_hind_flexion) | 13.0 % (left_hind_flexion) | 13.0 % (left_hind_flexion) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `fore160-s0_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.244 (0.243) | +0.0 % | -0.006 | 0.500 (0.599) | 0.18 (left_hind) | 1.04 (left_fore_pitch) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |
| +0.5 kg | 0/8 | 0.243 (0.243) | -0.2 % | -0.007 | 0.490 (0.615) | 0.18 (left_hind) | 1.07 (left_hind_flexion) | 3.9 % (left_fore_pitch) | 3.9 % (left_fore_pitch) |
| +1 kg | 0/8 | 0.243 (0.242) | -0.3 % | -0.007 | 0.489 (0.600) | 0.19 (left_hind) | 1.11 (left_hind_flexion) | 4.0 % (left_fore_pitch) | 4.0 % (left_fore_pitch) |
| +2 kg | 0/8 | 0.242 (0.241) | -0.6 % | -0.008 | 0.449 (0.566) | 0.20 (left_hind) | 1.17 (left_hind_flexion) | 4.4 % (left_fore_pitch) | 4.4 % (left_fore_pitch) |
| +3 kg | 0/8 | 0.242 (0.241) | -0.9 % | -0.008 | 0.439 (0.553) | 0.20 (left_hind) | 1.23 (left_hind_flexion) | 5.0 % (left_hind_flexion) | 5.0 % (left_hind_flexion) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `fore160-s0_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.244 (0.242) | -0.1 % | -0.006 | 0.248 (0.370) | 0.29 (left_hind) | 0.85 (right_hind_roll) | 0.0 % (left_fore_roll) | 50.9 % (left_hind_flexion) |
| x0.55 | 0/8 | 0.242 (0.241) | -0.8 % | -0.008 | 0.365 (0.470) | 0.19 (left_hind) | 0.93 (left_hind_flexion) | 0.0 % (left_fore_roll) | 25.7 % (left_hind_flexion) |
| x0.7 | 0/8 | 0.244 (0.243) | +0.1 % | -0.006 | 0.433 (0.535) | 0.18 (left_hind) | 0.99 (left_hind_flexion) | 0.0 % (left_fore_roll) | 10.6 % (left_fore_pitch) |
| x0.85 | 0/8 | 0.244 (0.243) | +0.0 % | -0.006 | 0.492 (0.609) | 0.18 (left_hind) | 1.02 (left_hind_flexion) | 0.0 % (left_fore_roll) | 5.9 % (left_fore_pitch) |
| x1 | 0/8 | 0.244 (0.243) | +0.0 % | -0.006 | 0.500 (0.599) | 0.18 (left_hind) | 1.04 (left_fore_pitch) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `fore160-s0_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.227 (0.225) | -6.9 % | -0.023 | 0.426 (0.490) | 0.31 (left_hind) | 0.92 (left_hind_flexion) | 2.6 % (right_hind_pitch) | 2.6 % (right_hind_pitch) |
| x0.6 | 0/8 | 0.237 (0.236) | -2.9 % | -0.013 | 0.416 (0.506) | 0.22 (left_hind) | 0.97 (left_hind_flexion) | 2.9 % (right_hind_pitch) | 2.9 % (right_hind_pitch) |
| x0.8 | 0/8 | 0.242 (0.241) | -0.9 % | -0.008 | 0.454 (0.525) | 0.19 (left_hind) | 1.00 (left_hind_flexion) | 2.8 % (left_fore_pitch) | 2.8 % (left_fore_pitch) |
| x1 | 0/8 | 0.244 (0.243) | +0.0 % | -0.006 | 0.500 (0.599) | 0.18 (left_hind) | 1.04 (left_fore_pitch) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |
| x1.2 | 0/8 | 0.245 (0.244) | +0.5 % | -0.005 | 0.520 (0.603) | 0.18 (right_hind) | 1.13 (left_fore_pitch) | 5.2 % (left_fore_pitch) | 5.2 % (left_fore_pitch) |
| x1.5 | 0/8 | 0.247 (0.245) | +1.2 % | -0.003 | 0.556 (0.701) | 0.23 (right_hind) | 1.27 (left_fore_pitch) | 8.8 % (left_fore_pitch) | 8.8 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `fore160-s0_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 42-66 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.2 | 192 | 0.84 | 8/8 | all fell | | | | |
| -1.25 m/s | 16.0 | 160 | 0.67 | 4/8 | 0.243 | 0.187 (0.259) | 2.34 (4.82) | 0 | 36.6 |
| -1 m/s | 12.8 | 128 | 0.51 | 0/8 | 0.243 | 0.099 (0.146) | 1.64 (3.63) | 0 | 18.8 |
| -0.75 m/s | 9.6 | 96 | 0.35 | 0/8 | 0.244 | 0.057 (0.069) | 1.71 (2.29) | 0 | 10.2 |
| -0.5 m/s | 6.4 | 64 | 0.22 | 0/8 | 0.244 | 0.047 (0.060) | 0.97 (2.21) | 0 | 9.4 |
| -0.4 m/s | 5.1 | 51 | 0.17 | 0/8 | 0.244 | 0.044 (0.057) | 0.95 (1.67) | 0 | 9.9 |
| -0.3 m/s | 3.8 | 38 | 0.13 | 0/8 | 0.244 | 0.041 (0.052) | 0.67 (1.70) | 0 | 9.3 |
| -0.2 m/s | 2.6 | 26 | 0.09 | 0/8 | 0.244 | 0.032 (0.049) | 0.35 (1.70) | 0 | 9.8 |
| -0.1 m/s | 1.3 | 13 | 0.04 | 0/8 | 0.244 | 0.022 (0.036) | 0.35 (1.79) | 0 | 9.6 |
| none | 0.0 | 0 |  | 0/8 | 0.244 | (no push) | 0.42 (1.74) | 0 | 9.3 |
| +0.1 m/s | 1.3 | 13 | 0.05 | 0/8 | 0.244 | 0.018 (0.026) | 0.56 (1.93) | 0 | 11.0 |
| +0.2 m/s | 2.6 | 26 | 0.10 | 0/8 | 0.244 | 0.024 (0.043) | 0.50 (1.15) | 0 | 12.4 |
| +0.3 m/s | 3.8 | 38 | 0.14 | 0/8 | 0.244 | 0.030 (0.055) | 0.58 (2.58) | 0 | 13.2 |
| +0.4 m/s | 5.1 | 51 | 0.19 | 0/8 | 0.245 | 0.036 (0.061) | 0.59 (2.40) | 0 | 13.5 |
| +0.5 m/s | 6.4 | 64 | 0.24 | 0/8 | 0.244 | 0.043 (0.065) | 0.52 (1.40) | 0 | 13.7 |
| +0.75 m/s | 9.6 | 96 | 0.40 | 0/8 | 0.244 | 0.078 (0.092) | 1.14 (1.88) | 0 | 14.9 |
| +1 m/s | 12.8 | 128 | 0.58 | 1/8 | 0.234 | 0.151 (0.236) | 2.52 (4.23) | 0 | 35.8 |
| +1.25 m/s | 16.0 | 160 | 0.79 | 6/8 | 0.225 | 0.231 (0.278) | 3.13 (3.53) | 0 | 36.5 |
| +1.5 m/s | 19.2 | 192 | 0.99 | 8/8 | all fell | | | | |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 0.42 s median and 1.74 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `fore160-s0_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.244 (0.243) | +0.0 % | -0.006 | 0.500 (0.599) | 0.18 (left_hind) | 1.04 (left_fore_pitch) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |
| step 1 cm | 0/8 | 0.244 (0.243) | -0.0 % | -0.006 | 0.496 (0.590) | 0.18 (left_hind) | 1.05 (left_fore_pitch) | 3.9 % (left_fore_pitch) | 3.9 % (left_fore_pitch) |
| step 2 cm | 0/8 | 0.244 (0.243) | +0.1 % | -0.006 | 0.486 (0.596) | 0.18 (left_hind) | 1.04 (left_fore_pitch) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |
| step 3 cm | 0/8 | 0.244 (0.243) | +0.0 % | -0.006 | 0.456 (0.576) | 0.19 (left_hind) | 1.06 (left_fore_pitch) | 3.9 % (left_fore_pitch) | 3.9 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `fore160-s0_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.244 (0.243) | +0.0 % | -0.006 | 0.500 (0.599) | 0.18 (left_hind) | 1.04 (left_fore_pitch) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |
| ramp 3 deg | 0/8 | 0.243 (0.242) | -0.5 % | -0.007 | 0.550 (0.629) | 0.18 (left_hind) | 1.07 (left_hind_flexion) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |
| ramp 5 deg | 0/8 | 0.241 (0.240) | -1.0 % | -0.009 | 0.597 (0.694) | 0.18 (left_hind) | 1.09 (left_hind_flexion) | 3.9 % (left_fore_pitch) | 3.9 % (left_fore_pitch) |
| ramp 8 deg | 0/8 | 0.240 (0.239) | -1.6 % | -0.010 | 0.557 (0.715) | 0.23 (right_hind) | 1.06 (left_hind_flexion) | 4.5 % (right_fore_pitch) | 4.5 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `fore160-s0_terrain.png`.

