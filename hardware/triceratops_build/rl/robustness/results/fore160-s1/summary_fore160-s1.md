# Robustness sweep: trike60-fore160-s1 (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-25. One policy, so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Policy `trike60_10002432_steps.zip` (sha256 85e6ec3b28752745...), run commit f84a6919; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c397e0c...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 13.9 min on 2 worker processes (565 episodes).
- Reproduction check (gait_pipe1's target case, randomisation on, seeds 100000-100004): mean speed 0.2456 m/s against the analysis's 0.2385 m/s; per-episode speeds identical: False.

## Headline

- **No fall and no speed drop of more than 20 %** in: delay, 0.15 m/s (0 ms to 60 ms); friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5); step up (flat to step 3 cm); ramp up (flat to ramp 8 deg).
- **delay, 0.25 m/s**: first break at 60 ms (2/8 fell).
- **Sideways push**: falls start at 1 m/s free-body (impulse 12.8 N s; the centre of mass gained 0.53 m/s), pushed to the right (2/8 fell); 1.25 m/s free-body (impulse 16.0 N s; the centre of mass gained 0.71 m/s), pushed to the left (4/8 fell). Largest push with no fall: 0.75 m/s free-body (impulse 9.6 N s; the centre of mass gained 0.35 m/s), pushed to the right; 1 m/s free-body (impulse 12.8 N s; the centre of mass gained 0.53 m/s), pushed to the left. The feet absorb part of every push: the centre of mass gains 27-64 % of the free-body velocity change (pushes of 0.3 m/s and more).

## Where it breaks

First point outward from nominal with a fall or a speed more than 20 % below the nominal point's (same command). "none": no such point in the tested range.

| Sweep | Nominal | Low side | High side |
|---|---|---|---|
| delay, 0.25 m/s | 10 ms | none | **60 ms**: 2/8 fell |
| delay, 0.15 m/s | 10 ms | none | none |
| friction | 0.85 (model) | none | none |
| mass, every body | x1 | none | none |
| payload on the chassis | +0 kg | (not tested) | none |
| force limit | x1 | none | (not tested) |
| position gain kp | x1 | none | none |
| sideways push | none | **1 m/s free-body to the right (impulse 12.8 N s; the centre of mass gained 0.53 m/s)**: 2/8 fell | **1.25 m/s free-body to the left (impulse 16.0 N s; the centre of mass gained 0.71 m/s)**: 4/8 fell |
| step up | flat | (not tested) | none |
| ramp up | flat | (not tested) | none |

Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.249 (0.248) | +1.0 % | -0.001 | 0.447 (0.544) | 0.44 (right_hind) | 1.08 (right_hind_flexion) | 3.3 % (right_fore_pitch) | 3.3 % (right_fore_pitch) |
| 2 ms | 0/8 | 0.247 (0.244) | +0.2 % | -0.003 | 0.437 (0.588) | 0.43 (right_hind) | 1.07 (right_hind_flexion) | 3.3 % (right_fore_pitch) | 3.3 % (right_fore_pitch) |
| 5 ms | 0/8 | 0.245 (0.244) | -0.6 % | -0.005 | 0.387 (0.512) | 0.42 (right_hind) | 1.07 (right_hind_flexion) | 3.2 % (right_fore_pitch) | 3.2 % (right_fore_pitch) |
| 10 ms | 0/8 | 0.246 (0.244) | +0.0 % | -0.004 | 0.166 (0.336) | 0.40 (right_hind) | 1.06 (right_hind_flexion) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| 15 ms | 0/8 | 0.248 (0.247) | +0.6 % | -0.002 | 0.044 (0.099) | 0.38 (right_hind) | 1.08 (right_hind_flexion) | 2.9 % (right_fore_pitch) | 2.9 % (right_fore_pitch) |
| 20 ms | 0/8 | 0.270 (0.267) | +9.5 % | +0.020 | 0.401 (0.575) | 0.36 (right_hind) | 1.35 (right_hind_pitch) | 14.7 % (right_hind_pitch) | 14.7 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.254 (0.252) | +3.0 % | +0.004 | 0.160 (0.186) | 0.36 (right_hind) | 1.26 (right_hind_pitch) | 11.4 % (right_hind_pitch) | 11.4 % (right_hind_pitch) |
| 40 ms | 0/8 | 0.255 (0.247) | +3.4 % | +0.005 | 0.412 (0.933) | 0.29 (right_hind) | 1.52 (right_hind_pitch) | 22.7 % (right_hind_pitch) | 22.7 % (right_hind_pitch) |
| 60 ms | 2/8 | 0.241 (0.197) | -2.0 % | -0.009 | 0.250 (0.547) | 0.28 (right_hind) | 1.55 (right_hind_pitch) | 23.8 % (right_hind_pitch) | 23.8 % (right_hind_pitch) |

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore160-s1_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.148 (0.146) | +2.1 % | -0.002 | 0.169 (0.247) | 0.70 (right_hind) | 0.97 (right_hind_flexion) | 1.2 % (right_fore_pitch) | 1.2 % (right_fore_pitch) |
| 2 ms | 0/8 | 0.146 (0.145) | +1.0 % | -0.004 | 0.154 (0.242) | 0.70 (right_hind) | 0.96 (right_hind_flexion) | 1.2 % (right_fore_pitch) | 1.2 % (right_fore_pitch) |
| 5 ms | 0/8 | 0.145 (0.144) | +0.1 % | -0.005 | 0.120 (0.188) | 0.69 (right_hind) | 0.95 (right_hind_flexion) | 1.2 % (right_fore_pitch) | 1.2 % (right_fore_pitch) |
| 10 ms | 0/8 | 0.145 (0.144) | +0.0 % | -0.005 | 0.047 (0.074) | 0.67 (right_hind) | 0.95 (right_hind_flexion) | 1.1 % (right_fore_pitch) | 1.1 % (right_fore_pitch) |
| 15 ms | 0/8 | 0.147 (0.145) | +1.3 % | -0.003 | 0.111 (0.200) | 0.66 (right_hind) | 0.96 (right_hind_flexion) | 1.0 % (right_fore_pitch) | 1.0 % (right_fore_pitch) |
| 20 ms | 0/8 | 0.168 (0.165) | +15.7 % | +0.018 | 0.478 (0.606) | 0.61 (right_hind) | 1.25 (right_hind_pitch) | 10.6 % (right_hind_pitch) | 10.6 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.158 (0.156) | +9.0 % | +0.008 | 0.476 (0.640) | 0.62 (right_hind) | 1.26 (right_hind_pitch) | 10.7 % (right_hind_pitch) | 10.7 % (right_hind_pitch) |
| 40 ms | 0/8 | 0.172 (0.168) | +18.7 % | +0.022 | 0.148 (0.447) | 0.57 (left_hind) | 1.61 (right_hind_pitch) | 25.9 % (right_hind_pitch) | 25.9 % (right_hind_pitch) |
| 60 ms | 0/8 | 0.185 (0.182) | +27.9 % | +0.035 | 0.344 (0.711) | 0.60 (left_hind) | 1.75 (right_hind_pitch) | 33.1 % (right_hind_pitch) | 33.1 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore160-s1_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.237 (0.236) | -3.7 % | -0.013 | 0.411 (0.536) | 0.87 (right_hind) | 0.84 (right_fore_pitch) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| 0.2 | 0/8 | 0.241 (0.240) | -2.1 % | -0.009 | 0.201 (0.326) | 0.81 (right_hind) | 0.85 (right_fore_pitch) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| 0.3 | 0/8 | 0.248 (0.246) | +0.5 % | -0.002 | 0.111 (0.206) | 0.75 (right_hind) | 0.87 (right_fore_pitch) | 2.8 % (right_fore_pitch) | 2.8 % (right_fore_pitch) |
| 0.4 | 0/8 | 0.247 (0.247) | +0.4 % | -0.003 | 0.054 (0.149) | 0.70 (right_hind) | 0.90 (right_fore_pitch) | 2.8 % (right_fore_pitch) | 2.8 % (right_fore_pitch) |
| 0.5 | 0/8 | 0.246 (0.244) | -0.1 % | -0.004 | 0.132 (0.292) | 0.62 (right_hind) | 0.95 (right_hind_flexion) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |
| 0.6 | 0/8 | 0.245 (0.244) | -0.4 % | -0.005 | 0.204 (0.335) | 0.55 (right_hind) | 0.99 (right_hind_flexion) | 2.7 % (right_fore_pitch) | 2.7 % (right_fore_pitch) |
| 0.8 | 0/8 | 0.246 (0.245) | -0.1 % | -0.004 | 0.166 (0.256) | 0.41 (right_hind) | 1.05 (right_hind_flexion) | 3.0 % (right_fore_pitch) | 3.0 % (right_fore_pitch) |
| 0.85 (model) | 0/8 | 0.246 (0.244) | +0.0 % | -0.004 | 0.166 (0.336) | 0.40 (right_hind) | 1.06 (right_hind_flexion) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| 1 | 0/8 | 0.247 (0.246) | +0.4 % | -0.003 | 0.114 (0.214) | 0.34 (right_hind) | 1.09 (right_hind_flexion) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| 1.2 | 0/8 | 0.247 (0.246) | +0.4 % | -0.003 | 0.104 (0.185) | 0.30 (right_hind) | 1.10 (right_hind_flexion) | 3.2 % (right_fore_pitch) | 3.2 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `fore160-s1_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.248 (0.246) | +0.5 % | -0.002 | 0.078 (0.184) | 0.41 (right_hind) | 0.88 (right_hind_flexion) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| x0.9 | 0/8 | 0.247 (0.246) | +0.4 % | -0.003 | 0.086 (0.190) | 0.40 (right_hind) | 0.97 (right_hind_flexion) | 2.7 % (right_fore_pitch) | 2.7 % (right_fore_pitch) |
| x1 | 0/8 | 0.246 (0.244) | +0.0 % | -0.004 | 0.166 (0.336) | 0.40 (right_hind) | 1.06 (right_hind_flexion) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| x1.1 | 0/8 | 0.245 (0.243) | -0.6 % | -0.005 | 0.209 (0.335) | 0.39 (right_hind) | 1.15 (right_hind_flexion) | 3.4 % (right_fore_pitch) | 3.4 % (right_fore_pitch) |
| x1.2 | 0/8 | 0.244 (0.242) | -0.7 % | -0.006 | 0.258 (0.451) | 0.38 (right_hind) | 1.24 (right_hind_flexion) | 3.7 % (right_fore_pitch) | 3.7 % (right_fore_pitch) |
| x1.4 | 0/8 | 0.244 (0.242) | -1.0 % | -0.006 | 0.285 (0.448) | 0.38 (right_hind) | 1.41 (right_hind_flexion) | 4.3 % (right_fore_pitch) | 4.3 % (right_fore_pitch) |
| x1.6 | 0/8 | 0.243 (0.241) | -1.4 % | -0.007 | 0.245 (0.412) | 0.37 (right_hind) | 1.55 (right_hind_flexion) | 7.6 % (right_hind_flexion) | 7.6 % (right_hind_flexion) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `fore160-s1_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.246 (0.244) | +0.0 % | -0.004 | 0.166 (0.336) | 0.40 (right_hind) | 1.06 (right_hind_flexion) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| +0.5 kg | 0/8 | 0.246 (0.244) | -0.3 % | -0.004 | 0.161 (0.301) | 0.39 (right_hind) | 1.10 (right_hind_flexion) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| +1 kg | 0/8 | 0.244 (0.242) | -0.8 % | -0.006 | 0.176 (0.357) | 0.39 (right_hind) | 1.14 (right_hind_flexion) | 3.3 % (right_fore_pitch) | 3.3 % (right_fore_pitch) |
| +2 kg | 0/8 | 0.243 (0.241) | -1.4 % | -0.007 | 0.186 (0.376) | 0.39 (right_hind) | 1.22 (right_hind_flexion) | 3.5 % (right_fore_pitch) | 3.5 % (right_fore_pitch) |
| +3 kg | 0/8 | 0.242 (0.240) | -1.9 % | -0.008 | 0.191 (0.390) | 0.38 (right_hind) | 1.30 (right_hind_flexion) | 3.7 % (right_fore_pitch) | 3.7 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `fore160-s1_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.244 (0.243) | -0.8 % | -0.006 | 0.156 (0.281) | 0.46 (right_hind) | 0.89 (right_hind_flexion) | 0.0 % (left_fore_roll) | 53.5 % (right_hind_flexion) |
| x0.55 | 0/8 | 0.242 (0.240) | -1.7 % | -0.008 | 0.305 (0.482) | 0.43 (right_hind) | 0.97 (right_hind_flexion) | 0.0 % (left_fore_roll) | 20.6 % (right_hind_flexion) |
| x0.7 | 0/8 | 0.244 (0.242) | -1.0 % | -0.006 | 0.264 (0.388) | 0.40 (right_hind) | 1.03 (right_hind_flexion) | 0.0 % (left_fore_roll) | 9.4 % (right_fore_pitch) |
| x0.85 | 0/8 | 0.246 (0.244) | -0.3 % | -0.004 | 0.168 (0.273) | 0.40 (right_hind) | 1.05 (right_hind_flexion) | 0.0 % (left_fore_roll) | 5.2 % (right_fore_pitch) |
| x1 | 0/8 | 0.246 (0.244) | +0.0 % | -0.004 | 0.166 (0.336) | 0.40 (right_hind) | 1.06 (right_hind_flexion) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `fore160-s1_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.236 (0.234) | -4.4 % | -0.014 | 0.121 (0.259) | 0.31 (right_hind) | 1.11 (right_hind_flexion) | 1.3 % (left_hind_pitch) | 1.3 % (left_hind_pitch) |
| x0.6 | 0/8 | 0.236 (0.235) | -4.2 % | -0.014 | 0.187 (0.311) | 0.39 (right_hind) | 1.05 (right_hind_flexion) | 0.7 % (right_fore_pitch) | 0.7 % (right_fore_pitch) |
| x0.8 | 0/8 | 0.241 (0.239) | -2.1 % | -0.009 | 0.209 (0.323) | 0.38 (right_hind) | 1.05 (right_hind_flexion) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| x1 | 0/8 | 0.246 (0.244) | +0.0 % | -0.004 | 0.166 (0.336) | 0.40 (right_hind) | 1.06 (right_hind_flexion) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| x1.2 | 0/8 | 0.251 (0.249) | +1.9 % | +0.001 | 0.130 (0.294) | 0.43 (right_hind) | 1.07 (right_hind_flexion) | 4.2 % (right_fore_pitch) | 4.2 % (right_fore_pitch) |
| x1.5 | 0/8 | 0.258 (0.256) | +4.6 % | +0.008 | 0.076 (0.185) | 0.47 (right_hind) | 1.09 (right_fore_pitch) | 5.7 % (right_fore_pitch) | 5.7 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `fore160-s1_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 27-64 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.2 | 192 | 0.96 | 8/8 | all fell | | | | |
| -1.25 m/s | 16.0 | 160 | 0.74 | 8/8 | all fell | | | | |
| -1 m/s | 12.8 | 128 | 0.53 | 2/8 | 0.247 | 0.137 (0.227) | 6.52 (6.52) | 5 | 35.4 |
| -0.75 m/s | 9.6 | 96 | 0.35 | 0/8 | 0.248 | 0.054 (0.098) | 7.09 (7.31) | 5 | 14.6 |
| -0.5 m/s | 6.4 | 64 | 0.19 | 0/8 | 0.247 | 0.025 (0.046) | 6.09 (8.97) | 4 | 8.1 |
| -0.4 m/s | 5.1 | 51 | 0.13 | 0/8 | 0.247 | 0.020 (0.033) | 9.69 (10.20) | 5 | 6.2 |
| -0.3 m/s | 3.8 | 38 | 0.08 | 0/8 | 0.247 | 0.015 (0.026) | 6.12 (7.44) | 4 | 4.6 |
| -0.2 m/s | 2.6 | 26 | 0.05 | 0/8 | 0.246 | 0.011 (0.025) | 7.09 (9.59) | 5 | 4.3 |
| -0.1 m/s | 1.3 | 13 | 0.02 | 0/8 | 0.246 | 0.007 (0.015) | 4.39 (8.96) | 5 | 4.8 |
| none | 0.0 | 0 |  | 0/8 | 0.246 | (no push) | 3.96 (8.15) | 4 | 3.8 |
| +0.1 m/s | 1.3 | 13 | 0.03 | 0/8 | 0.246 | 0.008 (0.013) | 5.74 (9.67) | 3 | 4.8 |
| +0.2 m/s | 2.6 | 26 | 0.07 | 0/8 | 0.247 | 0.017 (0.031) | 3.85 (4.34) | 6 | 4.8 |
| +0.3 m/s | 3.8 | 38 | 0.10 | 0/8 | 0.247 | 0.019 (0.043) | 4.44 (4.44) | 5 | 4.4 |
| +0.4 m/s | 5.1 | 51 | 0.15 | 0/8 | 0.247 | 0.029 (0.047) | 6.71 (8.29) | 4 | 5.9 |
| +0.5 m/s | 6.4 | 64 | 0.21 | 0/8 | 0.247 | 0.038 (0.057) | 9.11 (10.60) | 5 | 7.2 |
| +0.75 m/s | 9.6 | 96 | 0.36 | 0/8 | 0.248 | 0.065 (0.088) | 9.03 (9.41) | 5 | 10.1 |
| +1 m/s | 12.8 | 128 | 0.53 | 0/8 | 0.247 | 0.099 (0.141) | 4.37 (4.37) | 7 | 19.7 |
| +1.25 m/s | 16.0 | 160 | 0.71 | 4/8 | 0.247 | 0.177 (0.241) | 6.31 (9.09) | 1 | 36.3 |
| +1.5 m/s | 19.2 | 192 | 0.89 | 8/8 | all fell | | | | |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 3.96 s median and 8.15 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `fore160-s1_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.246 (0.244) | +0.0 % | -0.004 | 0.166 (0.336) | 0.40 (right_hind) | 1.06 (right_hind_flexion) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| step 1 cm | 0/8 | 0.246 (0.245) | +0.0 % | -0.004 | 0.156 (0.266) | 0.39 (right_hind) | 1.07 (right_hind_flexion) | 3.0 % (right_fore_pitch) | 3.0 % (right_fore_pitch) |
| step 2 cm | 0/8 | 0.246 (0.245) | +0.1 % | -0.004 | 0.139 (0.250) | 0.39 (right_hind) | 1.07 (right_hind_flexion) | 3.2 % (right_fore_pitch) | 3.2 % (right_fore_pitch) |
| step 3 cm | 0/8 | 0.246 (0.244) | +0.1 % | -0.004 | 0.183 (0.349) | 0.39 (right_hind) | 1.07 (right_hind_flexion) | 3.2 % (right_fore_pitch) | 3.2 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `fore160-s1_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.246 (0.244) | +0.0 % | -0.004 | 0.166 (0.336) | 0.40 (right_hind) | 1.06 (right_hind_flexion) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| ramp 3 deg | 0/8 | 0.251 (0.250) | +1.9 % | +0.001 | 0.121 (0.260) | 0.32 (right_hind) | 1.05 (right_hind_flexion) | 3.5 % (right_fore_pitch) | 3.5 % (right_fore_pitch) |
| ramp 5 deg | 0/8 | 0.254 (0.252) | +3.0 % | +0.004 | 0.210 (0.319) | 0.28 (right_hind) | 1.05 (right_fore_pitch) | 4.4 % (right_hind_pitch) | 4.4 % (right_hind_pitch) |
| ramp 8 deg | 0/8 | 0.255 (0.254) | +3.4 % | +0.005 | 0.358 (0.702) | 0.26 (left_hind) | 1.11 (left_hind_pitch) | 6.1 % (right_hind_pitch) | 6.1 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `fore160-s1_terrain.png`.

