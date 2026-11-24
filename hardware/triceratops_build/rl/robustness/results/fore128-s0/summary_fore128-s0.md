# Robustness sweep: trike60-fore128-s0 (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-24. One policy, so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Policy `trike60_10002432_steps.zip` (sha256 f7bbeb814d0d390b...), run commit f84a6919; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c397e0c...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 11.8 min on 2 worker processes (565 episodes).
- Reproduction check (gait_pipe1's target case, randomisation on, seeds 100000-100004): mean speed 0.2348 m/s against the analysis's 0.2385 m/s; per-episode speeds identical: False.

## Headline

- **No fall and no speed drop of more than 20 %** in: delay, 0.15 m/s (0 ms to 60 ms); friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5); step up (flat to step 3 cm); ramp up (flat to ramp 8 deg).
- **delay, 0.25 m/s**: first break at 40 ms (speed 0.086 m/s, 64 % below nominal).
- **Sideways push**: falls start at 1.25 m/s free-body (impulse 15.9 N s; the centre of mass gained 0.67 / 0.80 m/s), pushed to the left (6/8 fell) and right (8/8 fell). Largest push with no fall: 1 m/s free-body (impulse 12.8 N s; the centre of mass gained 0.48 / 0.62 m/s), pushed to the left and right. The feet absorb part of every push: the centre of mass gains 38-66 % of the free-body velocity change (pushes of 0.3 m/s and more).

## Where it breaks

First point outward from nominal with a fall or a speed more than 20 % below the nominal point's (same command). "none": no such point in the tested range.

| Sweep | Nominal | Low side | High side |
|---|---|---|---|
| delay, 0.25 m/s | 10 ms | none | **40 ms**: speed 0.086 m/s, 64 % below nominal |
| delay, 0.15 m/s | 10 ms | none | none |
| friction | 0.85 (model) | none | none |
| mass, every body | x1 | none | none |
| payload on the chassis | +0 kg | (not tested) | none |
| force limit | x1 | none | (not tested) |
| position gain kp | x1 | none | none |
| sideways push | none | **1.25 m/s free-body to the right (impulse 15.9 N s; the centre of mass gained 0.80 m/s)**: 8/8 fell | **1.25 m/s free-body to the left (impulse 15.9 N s; the centre of mass gained 0.67 m/s)**: 6/8 fell |
| step up | flat | (not tested) | none |
| ramp up | flat | (not tested) | none |

Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.248 (0.247) | +5.2 % | -0.002 | 0.537 (0.646) | 0.17 (left_hind) | 0.93 (right_hind_flexion) | 1.2 % (left_fore_pitch) | 1.2 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.244 (0.243) | +3.8 % | -0.006 | 0.512 (0.612) | 0.16 (right_hind) | 0.91 (right_hind_flexion) | 1.1 % (left_fore_pitch) | 1.1 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.239 (0.239) | +1.8 % | -0.011 | 0.470 (0.555) | 0.15 (left_hind) | 0.88 (right_hind_flexion) | 1.0 % (left_fore_pitch) | 1.0 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.235 (0.234) | +0.0 % | -0.015 | 0.337 (0.435) | 0.14 (left_hind) | 0.87 (left_hind_pitch) | 1.0 % (left_hind_pitch) | 1.0 % (left_hind_pitch) |
| 15 ms | 0/8 | 0.236 (0.235) | +0.5 % | -0.014 | 0.287 (0.403) | 0.13 (left_hind) | 0.87 (left_hind_pitch) | 1.1 % (left_hind_pitch) | 1.1 % (left_hind_pitch) |
| 20 ms | 0/8 | 0.241 (0.134) | +2.4 % | -0.009 | 0.226 (0.511) | 0.18 (left_fore) | 1.07 (left_hind_pitch) | 6.4 % (left_hind_pitch) | 6.4 % (left_hind_pitch) |
| 30 ms | 0/8 | 0.246 (0.237) | +4.6 % | -0.004 | 0.438 (0.820) | 0.19 (left_fore) | 1.11 (left_hind_pitch) | 7.0 % (left_hind_pitch) | 7.0 % (left_hind_pitch) |
| 40 ms | 0/8 | 0.086 (0.001) | -63.5 % | -0.164 | 0.142 (0.319) | 0.68 (left_fore) | 1.35 (right_fore_pitch) | 3.5 % (right_fore_pitch) | 3.5 % (right_fore_pitch) |
| 60 ms | 0/8 | 0.019 (0.000) | -92.1 % | -0.231 | 0.131 (0.463) | 3.08 (right_fore) | 1.50 (right_fore_pitch) | 3.3 % (right_fore_pitch) | 3.3 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore128-s0_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.153 (0.153) | +7.6 % | +0.003 | 0.321 (0.383) | 0.26 (left_hind) | 0.88 (right_hind_flexion) | 0.2 % (left_fore_pitch) | 0.2 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.150 (0.150) | +5.4 % | +0.000 | 0.314 (0.370) | 0.25 (right_hind) | 0.86 (right_hind_flexion) | 0.2 % (left_hind_flexion) | 0.2 % (left_hind_flexion) |
| 5 ms | 0/8 | 0.146 (0.146) | +2.8 % | -0.004 | 0.267 (0.329) | 0.23 (left_hind) | 0.81 (right_hind_flexion) | 0.2 % (left_hind_flexion) | 0.2 % (left_hind_flexion) |
| 10 ms | 0/8 | 0.142 (0.142) | +0.0 % | -0.008 | 0.242 (0.283) | 0.21 (right_hind) | 0.80 (left_hind_pitch) | 0.2 % (left_hind_pitch) | 0.2 % (left_hind_pitch) |
| 15 ms | 0/8 | 0.144 (0.143) | +0.9 % | -0.006 | 0.196 (0.250) | 0.20 (left_hind) | 0.81 (left_hind_pitch) | 0.5 % (left_hind_pitch) | 0.5 % (left_hind_pitch) |
| 20 ms | 0/8 | 0.158 (0.154) | +10.7 % | +0.008 | 0.065 (0.109) | 0.28 (left_fore) | 1.29 (right_hind_pitch) | 10.5 % (right_hind_pitch) | 10.5 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.158 (0.150) | +11.0 % | +0.008 | 0.229 (0.403) | 0.32 (left_fore) | 1.28 (right_hind_pitch) | 11.2 % (right_hind_pitch) | 11.2 % (right_hind_pitch) |
| 40 ms | 0/8 | 0.156 (0.142) | +9.7 % | +0.006 | 0.152 (0.393) | 0.44 (left_fore) | 1.32 (right_hind_pitch) | 15.3 % (right_hind_pitch) | 15.3 % (right_hind_pitch) |
| 60 ms | 0/8 | 0.122 (0.078) | -14.2 % | -0.028 | 0.252 (0.395) | 0.55 (left_fore) | 1.31 (right_fore_pitch) | 13.0 % (right_fore_pitch) | 13.0 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore128-s0_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.206 (-0.000) | -12.5 % | -0.044 | 0.338 (0.464) | 0.33 (right_hind) | 0.68 (right_fore_pitch) | 0.5 % (left_fore_pitch) | 0.5 % (left_fore_pitch) |
| 0.2 | 0/8 | 0.235 (0.234) | -0.3 % | -0.015 | 0.411 (0.480) | 0.32 (right_hind) | 0.71 (right_hind_flexion) | 0.6 % (left_fore_pitch) | 0.6 % (left_fore_pitch) |
| 0.3 | 0/8 | 0.235 (0.234) | -0.2 % | -0.015 | 0.431 (0.551) | 0.25 (right_hind) | 0.75 (right_hind_flexion) | 0.7 % (left_fore_pitch) | 0.7 % (left_fore_pitch) |
| 0.4 | 0/8 | 0.236 (0.235) | +0.2 % | -0.014 | 0.370 (0.474) | 0.20 (left_hind) | 0.80 (right_hind_flexion) | 0.8 % (left_fore_pitch) | 0.8 % (left_fore_pitch) |
| 0.5 | 0/8 | 0.236 (0.235) | +0.3 % | -0.014 | 0.352 (0.465) | 0.18 (left_hind) | 0.82 (right_hind_flexion) | 0.8 % (left_fore_pitch) | 0.8 % (left_fore_pitch) |
| 0.6 | 0/8 | 0.235 (0.234) | -0.1 % | -0.015 | 0.381 (0.499) | 0.17 (left_hind) | 0.84 (right_hind_flexion) | 0.8 % (left_hind_pitch) | 0.8 % (left_hind_pitch) |
| 0.8 | 0/8 | 0.235 (0.233) | -0.2 % | -0.015 | 0.363 (0.517) | 0.15 (left_hind) | 0.86 (left_hind_pitch) | 1.0 % (left_hind_pitch) | 1.0 % (left_hind_pitch) |
| 0.85 (model) | 0/8 | 0.235 (0.234) | +0.0 % | -0.015 | 0.337 (0.435) | 0.14 (left_hind) | 0.87 (left_hind_pitch) | 1.0 % (left_hind_pitch) | 1.0 % (left_hind_pitch) |
| 1 | 0/8 | 0.236 (0.234) | +0.3 % | -0.014 | 0.308 (0.466) | 0.14 (left_hind) | 0.88 (left_hind_pitch) | 1.1 % (left_hind_pitch) | 1.1 % (left_hind_pitch) |
| 1.2 | 0/8 | 0.236 (0.234) | +0.2 % | -0.014 | 0.328 (0.487) | 0.14 (left_hind) | 0.89 (left_hind_pitch) | 1.2 % (left_hind_pitch) | 1.2 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `fore128-s0_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.240 (0.238) | +1.9 % | -0.010 | 0.191 (0.299) | 0.13 (left_hind) | 0.72 (left_hind_pitch) | 0.5 % (left_hind_pitch) | 0.5 % (left_hind_pitch) |
| x0.9 | 0/8 | 0.238 (0.236) | +1.0 % | -0.012 | 0.261 (0.435) | 0.14 (left_hind) | 0.79 (left_hind_pitch) | 0.7 % (left_hind_pitch) | 0.7 % (left_hind_pitch) |
| x1 | 0/8 | 0.235 (0.234) | +0.0 % | -0.015 | 0.337 (0.435) | 0.14 (left_hind) | 0.87 (left_hind_pitch) | 1.0 % (left_hind_pitch) | 1.0 % (left_hind_pitch) |
| x1.1 | 0/8 | 0.233 (0.232) | -0.9 % | -0.017 | 0.396 (0.488) | 0.14 (left_hind) | 0.94 (left_hind_pitch) | 1.3 % (left_hind_pitch) | 1.3 % (left_hind_pitch) |
| x1.2 | 0/8 | 0.231 (0.230) | -2.0 % | -0.019 | 0.450 (0.563) | 0.16 (right_hind) | 1.04 (right_hind_flexion) | 1.8 % (left_hind_pitch) | 1.8 % (left_hind_pitch) |
| x1.4 | 0/8 | 0.227 (0.226) | -3.5 % | -0.023 | 0.497 (0.578) | 0.17 (right_hind) | 1.19 (right_hind_flexion) | 4.0 % (left_hind_pitch) | 4.0 % (left_hind_pitch) |
| x1.6 | 0/8 | 0.223 (0.222) | -5.4 % | -0.027 | 0.541 (0.633) | 0.17 (left_hind) | 1.32 (right_hind_flexion) | 7.5 % (left_hind_pitch) | 7.5 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `fore128-s0_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.235 (0.234) | +0.0 % | -0.015 | 0.337 (0.435) | 0.14 (left_hind) | 0.87 (left_hind_pitch) | 1.0 % (left_hind_pitch) | 1.0 % (left_hind_pitch) |
| +0.5 kg | 0/8 | 0.234 (0.233) | -0.6 % | -0.016 | 0.377 (0.475) | 0.15 (left_hind) | 0.89 (left_hind_pitch) | 1.2 % (left_hind_pitch) | 1.2 % (left_hind_pitch) |
| +1 kg | 0/8 | 0.232 (0.232) | -1.3 % | -0.018 | 0.400 (0.504) | 0.15 (left_hind) | 0.94 (right_hind_flexion) | 1.3 % (left_hind_pitch) | 1.3 % (left_hind_pitch) |
| +2 kg | 0/8 | 0.230 (0.229) | -2.2 % | -0.020 | 0.445 (0.536) | 0.17 (right_hind) | 1.02 (right_hind_flexion) | 1.6 % (left_hind_pitch) | 1.6 % (left_hind_pitch) |
| +3 kg | 0/8 | 0.228 (0.227) | -3.2 % | -0.022 | 0.468 (0.570) | 0.17 (right_hind) | 1.10 (right_hind_flexion) | 1.9 % (left_hind_pitch) | 1.9 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `fore128-s0_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.231 (0.226) | -1.8 % | -0.019 | 1.195 (1.312) | 0.38 (left_hind) | 0.89 (right_hind_flexion) | 0.0 % (left_fore_roll) | 58.9 % (right_hind_flexion) |
| x0.55 | 0/8 | 0.232 (0.231) | -1.5 % | -0.018 | 0.379 (0.491) | 0.14 (left_hind) | 0.82 (left_hind_pitch) | 0.0 % (left_fore_roll) | 14.6 % (left_hind_pitch) |
| x0.7 | 0/8 | 0.234 (0.233) | -0.4 % | -0.016 | 0.352 (0.491) | 0.14 (left_hind) | 0.85 (left_hind_pitch) | 0.0 % (left_fore_roll) | 5.5 % (left_hind_pitch) |
| x0.85 | 0/8 | 0.235 (0.233) | -0.2 % | -0.015 | 0.352 (0.468) | 0.14 (left_hind) | 0.86 (left_hind_pitch) | 0.0 % (left_fore_roll) | 2.1 % (left_hind_pitch) |
| x1 | 0/8 | 0.235 (0.234) | +0.0 % | -0.015 | 0.337 (0.435) | 0.14 (left_hind) | 0.87 (left_hind_pitch) | 1.0 % (left_hind_pitch) | 1.0 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `fore128-s0_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.219 (0.218) | -6.9 % | -0.031 | 0.209 (0.325) | 0.14 (left_hind) | 0.84 (right_hind_flexion) | 0.4 % (left_hind_pitch) | 0.4 % (left_hind_pitch) |
| x0.6 | 0/8 | 0.227 (0.226) | -3.6 % | -0.023 | 0.309 (0.398) | 0.15 (left_hind) | 0.83 (right_hind_flexion) | 0.4 % (left_hind_pitch) | 0.4 % (left_hind_pitch) |
| x0.8 | 0/8 | 0.232 (0.231) | -1.5 % | -0.018 | 0.330 (0.468) | 0.14 (left_hind) | 0.83 (right_hind_flexion) | 0.5 % (left_hind_pitch) | 0.5 % (left_hind_pitch) |
| x1 | 0/8 | 0.235 (0.234) | +0.0 % | -0.015 | 0.337 (0.435) | 0.14 (left_hind) | 0.87 (left_hind_pitch) | 1.0 % (left_hind_pitch) | 1.0 % (left_hind_pitch) |
| x1.2 | 0/8 | 0.238 (0.237) | +1.2 % | -0.012 | 0.323 (0.380) | 0.15 (left_hind) | 0.92 (left_hind_pitch) | 1.8 % (left_hind_pitch) | 1.8 % (left_hind_pitch) |
| x1.5 | 0/8 | 0.242 (0.239) | +2.8 % | -0.008 | 0.148 (0.296) | 0.17 (left_hind) | 1.03 (left_hind_pitch) | 4.1 % (left_hind_pitch) | 4.1 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `fore128-s0_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 38-66 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.1 | 191 | 0.99 | 8/8 | all fell | | | | |
| -1.25 m/s | 15.9 | 159 | 0.80 | 8/8 | all fell | | | | |
| -1 m/s | 12.8 | 128 | 0.62 | 0/8 | 0.234 | 0.120 (0.168) | 0.96 (2.47) | 0 | 22.5 |
| -0.75 m/s | 9.6 | 96 | 0.45 | 0/8 | 0.236 | 0.067 (0.082) | 0.64 (4.35) | 0 | 10.5 |
| -0.5 m/s | 6.4 | 64 | 0.29 | 0/8 | 0.237 | 0.057 (0.098) | 0.69 (7.05) | 0 | 7.0 |
| -0.4 m/s | 5.1 | 51 | 0.23 | 0/8 | 0.236 | 0.048 (0.085) | 0.39 (1.25) | 0 | 6.6 |
| -0.3 m/s | 3.8 | 38 | 0.17 | 0/8 | 0.237 | 0.051 (0.088) | 0.52 (8.28) | 0 | 7.3 |
| -0.2 m/s | 2.6 | 26 | 0.11 | 0/8 | 0.236 | 0.039 (0.059) | 0.41 (3.25) | 0 | 7.2 |
| -0.1 m/s | 1.3 | 13 | 0.05 | 0/8 | 0.236 | 0.027 (0.035) | 0.10 (0.29) | 0 | 6.7 |
| none | 0.0 | 0 |  | 0/8 | 0.235 | (no push) | 0.10 (0.97) | 0 | 6.5 |
| +0.1 m/s | 1.3 | 13 | 0.04 | 0/8 | 0.236 | 0.022 (0.034) | 0.87 (2.41) | 0 | 7.4 |
| +0.2 m/s | 2.6 | 26 | 0.07 | 0/8 | 0.236 | 0.028 (0.044) | 0.61 (1.02) | 0 | 8.3 |
| +0.3 m/s | 3.8 | 38 | 0.11 | 0/8 | 0.235 | 0.029 (0.047) | 0.88 (1.05) | 0 | 7.6 |
| +0.4 m/s | 5.1 | 51 | 0.15 | 0/8 | 0.236 | 0.034 (0.046) | 0.95 (2.30) | 0 | 8.6 |
| +0.5 m/s | 6.4 | 64 | 0.20 | 0/8 | 0.235 | 0.037 (0.049) | 0.96 (1.23) | 0 | 8.6 |
| +0.75 m/s | 9.6 | 96 | 0.32 | 0/8 | 0.236 | 0.050 (0.066) | 0.97 (2.21) | 0 | 9.6 |
| +1 m/s | 12.8 | 128 | 0.48 | 0/8 | 0.234 | 0.106 (0.138) | 0.92 (1.07) | 0 | 19.0 |
| +1.25 m/s | 15.9 | 159 | 0.67 | 6/8 | 0.233 | 0.210 (0.251) | 2.46 (2.90) | 0 | 36.6 |
| +1.5 m/s | 19.1 | 191 | 0.87 | 8/8 | all fell | | | | |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 0.10 s median and 0.97 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `fore128-s0_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.235 (0.234) | +0.0 % | -0.015 | 0.337 (0.435) | 0.14 (left_hind) | 0.87 (left_hind_pitch) | 1.0 % (left_hind_pitch) | 1.0 % (left_hind_pitch) |
| step 1 cm | 0/8 | 0.236 (0.234) | +0.2 % | -0.014 | 0.323 (0.450) | 0.14 (left_hind) | 0.87 (left_hind_pitch) | 1.0 % (left_hind_pitch) | 1.0 % (left_hind_pitch) |
| step 2 cm | 0/8 | 0.236 (0.235) | +0.4 % | -0.014 | 0.273 (0.376) | 0.15 (left_hind) | 0.88 (left_hind_pitch) | 1.2 % (left_hind_pitch) | 1.2 % (left_hind_pitch) |
| step 3 cm | 0/8 | 0.235 (0.235) | +0.1 % | -0.015 | 0.271 (0.369) | 0.15 (left_hind) | 0.88 (left_hind_pitch) | 1.1 % (left_hind_pitch) | 1.1 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `fore128-s0_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.235 (0.234) | +0.0 % | -0.015 | 0.337 (0.435) | 0.14 (left_hind) | 0.87 (left_hind_pitch) | 1.0 % (left_hind_pitch) | 1.0 % (left_hind_pitch) |
| ramp 3 deg | 0/8 | 0.234 (0.232) | -0.7 % | -0.016 | 0.396 (0.539) | 0.17 (right_hind) | 0.92 (right_hind_flexion) | 1.4 % (left_hind_pitch) | 1.4 % (left_hind_pitch) |
| ramp 5 deg | 0/8 | 0.234 (0.233) | -0.6 % | -0.016 | 0.352 (0.459) | 0.18 (right_hind) | 0.93 (right_hind_flexion) | 1.7 % (left_hind_pitch) | 1.7 % (left_hind_pitch) |
| ramp 8 deg | 0/8 | 0.234 (0.232) | -0.7 % | -0.016 | 0.277 (0.359) | 0.20 (right_hind) | 0.94 (left_fore_pitch) | 2.0 % (left_hind_pitch) | 2.0 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `fore128-s0_terrain.png`.

