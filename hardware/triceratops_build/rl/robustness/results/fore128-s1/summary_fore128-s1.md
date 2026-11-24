# Robustness sweep: trike60-fore128-s1 (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-24. One policy, so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Policy `trike60_10002432_steps.zip` (sha256 dc3090ac8dd20ad0...), run commit f84a6919; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c397e0c...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 12.2 min on 2 worker processes (565 episodes).
- Reproduction check (gait_pipe1's target case, randomisation on, seeds 100000-100004): mean speed 0.2438 m/s against the analysis's 0.2385 m/s; per-episode speeds identical: False.

## Headline

- **No fall and no speed drop of more than 20 %** in: delay, 0.25 m/s (0 ms to 60 ms); delay, 0.15 m/s (0 ms to 60 ms); friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5); step up (flat to step 3 cm); ramp up (flat to ramp 8 deg).
- **Sideways push**: falls start at 1 m/s free-body (impulse 12.8 N s; the centre of mass gained 0.67 / 0.51 m/s), pushed to the left (7/8 fell) and right (1/8 fell). Largest push with no fall: 0.75 m/s free-body (impulse 9.6 N s; the centre of mass gained 0.49 / 0.35 m/s), pushed to the left and right. The feet absorb part of every push: the centre of mass gains 40-73 % of the free-body velocity change (pushes of 0.3 m/s and more).

## Where it breaks

First point outward from nominal with a fall or a speed more than 20 % below the nominal point's (same command). "none": no such point in the tested range.

| Sweep | Nominal | Low side | High side |
|---|---|---|---|
| delay, 0.25 m/s | 10 ms | none | none |
| delay, 0.15 m/s | 10 ms | none | none |
| friction | 0.85 (model) | none | none |
| mass, every body | x1 | none | none |
| payload on the chassis | +0 kg | (not tested) | none |
| force limit | x1 | none | (not tested) |
| position gain kp | x1 | none | none |
| sideways push | none | **1 m/s free-body to the right (impulse 12.8 N s; the centre of mass gained 0.51 m/s)**: 1/8 fell | **1 m/s free-body to the left (impulse 12.8 N s; the centre of mass gained 0.67 m/s)**: 7/8 fell |
| step up | flat | (not tested) | none |
| ramp up | flat | (not tested) | none |

Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.254 (0.253) | +4.7 % | +0.004 | 0.301 (0.405) | 0.25 (left_hind) | 0.88 (left_fore_pitch) | 1.7 % (left_fore_pitch) | 1.7 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.251 (0.250) | +3.2 % | +0.001 | 0.298 (0.416) | 0.24 (left_hind) | 0.87 (left_fore_pitch) | 1.7 % (left_fore_pitch) | 1.7 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.246 (0.245) | +1.4 % | -0.004 | 0.323 (0.457) | 0.22 (left_hind) | 0.88 (right_hind_flexion) | 1.5 % (left_fore_pitch) | 1.5 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.243 (0.242) | +0.0 % | -0.007 | 0.348 (0.462) | 0.18 (left_hind) | 0.94 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| 15 ms | 0/8 | 0.246 (0.243) | +1.1 % | -0.004 | 0.374 (0.455) | 0.17 (right_hind) | 1.02 (right_hind_flexion) | 2.2 % (left_fore_pitch) | 2.2 % (left_fore_pitch) |
| 20 ms | 0/8 | 0.245 (0.244) | +1.0 % | -0.005 | 0.264 (0.400) | 0.21 (right_hind) | 1.13 (left_fore_pitch) | 4.4 % (right_hind_pitch) | 4.4 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.236 (0.235) | -2.8 % | -0.014 | 0.181 (0.331) | 0.22 (right_hind) | 1.09 (left_fore_pitch) | 4.6 % (right_hind_pitch) | 4.6 % (right_hind_pitch) |
| 40 ms | 0/8 | 0.232 (0.227) | -4.5 % | -0.018 | 0.607 (0.771) | 0.27 (left_fore) | 1.43 (right_hind_pitch) | 18.2 % (right_hind_pitch) | 18.2 % (right_hind_pitch) |
| 60 ms | 0/8 | 0.217 (0.198) | -10.6 % | -0.033 | 0.602 (1.037) | 0.36 (left_fore) | 1.60 (right_hind_pitch) | 25.6 % (right_hind_pitch) | 25.6 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore128-s1_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.155 (0.154) | +5.5 % | +0.005 | 0.199 (0.258) | 0.30 (left_hind) | 0.68 (left_fore_pitch) | 0.4 % (left_fore_pitch) | 0.4 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.152 (0.151) | +3.8 % | +0.002 | 0.213 (0.266) | 0.30 (left_hind) | 0.68 (left_fore_pitch) | 0.4 % (left_fore_pitch) | 0.4 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.149 (0.148) | +1.6 % | -0.001 | 0.212 (0.262) | 0.28 (left_hind) | 0.68 (right_hind_flexion) | 0.3 % (left_fore_pitch) | 0.3 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.147 (0.146) | +0.0 % | -0.003 | 0.240 (0.287) | 0.25 (left_hind) | 0.76 (right_hind_flexion) | 0.3 % (left_fore_pitch) | 0.3 % (left_fore_pitch) |
| 15 ms | 0/8 | 0.149 (0.148) | +1.6 % | -0.001 | 0.228 (0.278) | 0.20 (left_hind) | 0.88 (right_hind_flexion) | 0.4 % (left_fore_pitch) | 0.4 % (left_fore_pitch) |
| 20 ms | 0/8 | 0.147 (0.146) | +0.1 % | -0.003 | 0.222 (0.310) | 0.30 (right_hind) | 1.13 (left_fore_pitch) | 6.1 % (right_hind_pitch) | 6.1 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.144 (0.143) | -1.6 % | -0.006 | 0.224 (0.361) | 0.36 (right_hind) | 1.14 (right_hind_pitch) | 7.7 % (right_hind_pitch) | 7.7 % (right_hind_pitch) |
| 40 ms | 0/8 | 0.141 (0.137) | -4.2 % | -0.009 | 0.375 (0.544) | 0.44 (left_fore) | 1.54 (right_hind_pitch) | 23.2 % (right_hind_pitch) | 23.2 % (right_hind_pitch) |
| 60 ms | 0/8 | 0.139 (0.127) | -5.5 % | -0.011 | 0.625 (0.843) | 0.56 (right_hind) | 1.71 (right_hind_pitch) | 31.1 % (right_hind_pitch) | 31.1 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore128-s1_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.245 (0.244) | +0.9 % | -0.005 | 0.383 (0.449) | 0.42 (left_hind) | 0.69 (left_fore_pitch) | 1.2 % (right_hind_pitch) | 1.2 % (right_hind_pitch) |
| 0.2 | 0/8 | 0.245 (0.243) | +0.7 % | -0.005 | 0.389 (0.514) | 0.40 (left_hind) | 0.71 (left_fore_pitch) | 1.2 % (left_fore_pitch) | 1.2 % (left_fore_pitch) |
| 0.3 | 0/8 | 0.245 (0.243) | +0.8 % | -0.005 | 0.447 (0.569) | 0.36 (left_hind) | 0.73 (left_fore_pitch) | 1.2 % (left_fore_pitch) | 1.2 % (left_fore_pitch) |
| 0.4 | 0/8 | 0.245 (0.243) | +0.8 % | -0.005 | 0.523 (0.655) | 0.30 (left_hind) | 0.76 (right_hind_flexion) | 1.2 % (left_fore_pitch) | 1.2 % (left_fore_pitch) |
| 0.5 | 0/8 | 0.245 (0.243) | +0.8 % | -0.005 | 0.493 (0.583) | 0.25 (left_hind) | 0.81 (right_hind_flexion) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| 0.6 | 0/8 | 0.244 (0.243) | +0.4 % | -0.006 | 0.461 (0.549) | 0.22 (left_hind) | 0.85 (right_hind_flexion) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| 0.8 | 0/8 | 0.243 (0.241) | +0.0 % | -0.007 | 0.355 (0.477) | 0.18 (left_hind) | 0.93 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| 0.85 (model) | 0/8 | 0.243 (0.242) | +0.0 % | -0.007 | 0.348 (0.462) | 0.18 (left_hind) | 0.94 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| 1 | 0/8 | 0.243 (0.241) | -0.0 % | -0.007 | 0.344 (0.496) | 0.18 (left_hind) | 0.96 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| 1.2 | 0/8 | 0.243 (0.241) | -0.1 % | -0.007 | 0.353 (0.466) | 0.18 (left_hind) | 0.97 (right_hind_flexion) | 1.7 % (left_fore_pitch) | 1.7 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `fore128-s1_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.244 (0.243) | +0.3 % | -0.006 | 0.310 (0.422) | 0.16 (right_hind) | 0.81 (right_hind_flexion) | 1.2 % (left_fore_pitch) | 1.2 % (left_fore_pitch) |
| x0.9 | 0/8 | 0.243 (0.242) | +0.2 % | -0.007 | 0.321 (0.419) | 0.17 (left_hind) | 0.87 (right_hind_flexion) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| x1 | 0/8 | 0.243 (0.242) | +0.0 % | -0.007 | 0.348 (0.462) | 0.18 (left_hind) | 0.94 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| x1.1 | 0/8 | 0.242 (0.241) | -0.2 % | -0.008 | 0.357 (0.494) | 0.19 (left_hind) | 1.01 (right_hind_flexion) | 1.8 % (left_fore_pitch) | 1.8 % (left_fore_pitch) |
| x1.2 | 0/8 | 0.242 (0.240) | -0.6 % | -0.008 | 0.374 (0.483) | 0.20 (left_hind) | 1.05 (right_hind_flexion) | 2.1 % (left_fore_pitch) | 2.1 % (left_fore_pitch) |
| x1.4 | 0/8 | 0.240 (0.238) | -1.3 % | -0.010 | 0.372 (0.501) | 0.22 (left_hind) | 1.15 (right_hind_flexion) | 3.0 % (left_fore_pitch) | 3.0 % (left_fore_pitch) |
| x1.6 | 0/8 | 0.237 (0.236) | -2.4 % | -0.013 | 0.349 (0.469) | 0.24 (left_hind) | 1.24 (right_hind_flexion) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `fore128-s1_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.243 (0.242) | +0.0 % | -0.007 | 0.348 (0.462) | 0.18 (left_hind) | 0.94 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| +0.5 kg | 0/8 | 0.242 (0.241) | -0.4 % | -0.008 | 0.353 (0.474) | 0.19 (left_hind) | 0.96 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| +1 kg | 0/8 | 0.241 (0.240) | -0.7 % | -0.009 | 0.376 (0.488) | 0.20 (left_hind) | 0.99 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| +2 kg | 0/8 | 0.240 (0.238) | -1.4 % | -0.010 | 0.379 (0.491) | 0.22 (left_hind) | 1.02 (right_hind_flexion) | 1.7 % (left_fore_pitch) | 1.7 % (left_fore_pitch) |
| +3 kg | 0/8 | 0.238 (0.236) | -2.1 % | -0.012 | 0.400 (0.537) | 0.24 (left_hind) | 1.06 (right_hind_flexion) | 1.8 % (left_fore_pitch) | 1.8 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `fore128-s1_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.248 (0.245) | +2.0 % | -0.002 | 0.081 (0.292) | 0.24 (left_hind) | 0.82 (right_hind_flexion) | 0.0 % (left_fore_roll) | 46.6 % (right_hind_flexion) |
| x0.55 | 0/8 | 0.241 (0.240) | -0.7 % | -0.009 | 0.378 (0.509) | 0.19 (left_hind) | 0.90 (right_hind_flexion) | 0.0 % (left_fore_roll) | 12.0 % (left_fore_pitch) |
| x0.7 | 0/8 | 0.242 (0.241) | -0.3 % | -0.008 | 0.363 (0.472) | 0.18 (left_hind) | 0.91 (right_hind_flexion) | 0.0 % (left_fore_roll) | 5.5 % (left_fore_pitch) |
| x0.85 | 0/8 | 0.243 (0.241) | -0.1 % | -0.007 | 0.349 (0.487) | 0.19 (left_hind) | 0.93 (right_hind_flexion) | 0.0 % (left_fore_roll) | 2.9 % (left_fore_pitch) |
| x1 | 0/8 | 0.243 (0.242) | +0.0 % | -0.007 | 0.348 (0.462) | 0.18 (left_hind) | 0.94 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `fore128-s1_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.230 (0.229) | -5.4 % | -0.020 | 0.442 (0.492) | 0.17 (right_hind) | 0.99 (right_hind_flexion) | 0.2 % (left_fore_pitch) | 0.2 % (left_fore_pitch) |
| x0.6 | 0/8 | 0.237 (0.237) | -2.3 % | -0.013 | 0.080 (0.155) | 0.20 (left_hind) | 0.96 (right_hind_flexion) | 0.5 % (left_fore_pitch) | 0.5 % (left_fore_pitch) |
| x0.8 | 0/8 | 0.241 (0.240) | -0.9 % | -0.009 | 0.205 (0.303) | 0.19 (left_hind) | 0.95 (right_hind_flexion) | 1.0 % (left_fore_pitch) | 1.0 % (left_fore_pitch) |
| x1 | 0/8 | 0.243 (0.242) | +0.0 % | -0.007 | 0.348 (0.462) | 0.18 (left_hind) | 0.94 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| x1.2 | 0/8 | 0.245 (0.243) | +0.7 % | -0.005 | 0.395 (0.500) | 0.19 (left_hind) | 0.91 (right_hind_flexion) | 2.4 % (left_fore_pitch) | 2.4 % (left_fore_pitch) |
| x1.5 | 0/8 | 0.247 (0.245) | +1.7 % | -0.003 | 0.397 (0.496) | 0.20 (left_hind) | 0.98 (left_fore_pitch) | 3.9 % (left_fore_pitch) | 3.9 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `fore128-s1_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 40-73 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.1 | 191 | 0.92 | 8/8 | all fell | | | | |
| -1.25 m/s | 15.9 | 159 | 0.71 | 8/8 | all fell | | | | |
| -1 m/s | 12.8 | 128 | 0.51 | 1/8 | 0.240 | 0.163 (0.224) | 1.27 (2.91) | 0 | 35.0 |
| -0.75 m/s | 9.6 | 96 | 0.35 | 0/8 | 0.243 | 0.061 (0.069) | 0.79 (1.31) | 0 | 13.9 |
| -0.5 m/s | 6.4 | 64 | 0.20 | 0/8 | 0.243 | 0.028 (0.036) | 0.64 (1.80) | 0 | 9.0 |
| -0.4 m/s | 5.1 | 51 | 0.16 | 0/8 | 0.243 | 0.021 (0.030) | 0.58 (0.59) | 0 | 9.2 |
| -0.3 m/s | 3.8 | 38 | 0.12 | 0/8 | 0.243 | 0.018 (0.029) | 0.55 (0.56) | 0 | 9.3 |
| -0.2 m/s | 2.6 | 26 | 0.09 | 0/8 | 0.243 | 0.015 (0.028) | 0.51 (0.56) | 0 | 8.8 |
| -0.1 m/s | 1.3 | 13 | 0.05 | 0/8 | 0.243 | 0.014 (0.038) | 0.10 (0.55) | 0 | 9.3 |
| none | 0.0 | 0 |  | 0/8 | 0.243 | (no push) | 0.10 (0.23) | 0 | 9.2 |
| +0.1 m/s | 1.3 | 13 | 0.06 | 0/8 | 0.243 | 0.028 (0.035) | 0.37 (0.42) | 0 | 9.7 |
| +0.2 m/s | 2.6 | 26 | 0.12 | 0/8 | 0.243 | 0.042 (0.050) | 0.38 (0.86) | 0 | 10.7 |
| +0.3 m/s | 3.8 | 38 | 0.19 | 0/8 | 0.243 | 0.052 (0.061) | 0.38 (0.86) | 0 | 11.4 |
| +0.4 m/s | 5.1 | 51 | 0.25 | 0/8 | 0.244 | 0.062 (0.070) | 1.48 (1.55) | 0 | 12.3 |
| +0.5 m/s | 6.4 | 64 | 0.32 | 0/8 | 0.244 | 0.072 (0.081) | 0.77 (1.49) | 0 | 13.2 |
| +0.75 m/s | 9.6 | 96 | 0.49 | 0/8 | 0.245 | 0.100 (0.105) | 0.85 (1.34) | 0 | 15.3 |
| +1 m/s | 12.8 | 128 | 0.67 | 7/8 | 0.240 | 0.227 (0.242) | 1.42 (1.42) | 0 | 36.1 |
| +1.25 m/s | 15.9 | 159 | 0.88 | 8/8 | all fell | | | | |
| +1.5 m/s | 19.1 | 191 | 1.09 | 8/8 | all fell | | | | |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 0.10 s median and 0.23 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `fore128-s1_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.243 (0.242) | +0.0 % | -0.007 | 0.348 (0.462) | 0.18 (left_hind) | 0.94 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| step 1 cm | 0/8 | 0.244 (0.242) | +0.2 % | -0.006 | 0.316 (0.412) | 0.18 (left_hind) | 0.95 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| step 2 cm | 0/8 | 0.243 (0.242) | +0.1 % | -0.007 | 0.280 (0.386) | 0.18 (left_hind) | 0.95 (right_hind_flexion) | 1.5 % (left_fore_pitch) | 1.5 % (left_fore_pitch) |
| step 3 cm | 0/8 | 0.242 (0.239) | -0.4 % | -0.008 | 0.279 (0.437) | 0.18 (left_hind) | 0.95 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `fore128-s1_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.243 (0.242) | +0.0 % | -0.007 | 0.348 (0.462) | 0.18 (left_hind) | 0.94 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| ramp 3 deg | 0/8 | 0.242 (0.241) | -0.4 % | -0.008 | 0.343 (0.456) | 0.18 (right_hind) | 0.98 (right_hind_flexion) | 1.7 % (left_fore_pitch) | 1.7 % (left_fore_pitch) |
| ramp 5 deg | 0/8 | 0.240 (0.238) | -1.3 % | -0.010 | 0.426 (0.523) | 0.19 (right_hind) | 1.00 (left_fore_pitch) | 2.1 % (left_fore_pitch) | 2.1 % (left_fore_pitch) |
| ramp 8 deg | 0/8 | 0.236 (0.233) | -2.9 % | -0.014 | 0.577 (0.726) | 0.21 (right_hind) | 1.02 (left_fore_pitch) | 2.7 % (left_fore_pitch) | 2.7 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `fore128-s1_terrain.png`.

