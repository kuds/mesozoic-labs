# Robustness sweep: model-based trot (walk60) (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-19. One controller (the model-based trot, `walk/walk60.py`, frozen parameters), so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Controller `walk/walk60.py (a controller; no checkpoint)` (sha256 5225c64a2006d8b4...), run commit 088ca493; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 4ae8f2d952bfec1f...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 6.8 min on 4 worker processes (560 episodes).

## Headline

- **No fall and no speed drop of more than 20 %** in: delay, 0.25 m/s (0 ms to 60 ms); delay, 0.15 m/s (0 ms to 60 ms); friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5).
- **step up**: first break at step 3 cm (speed 0.193 m/s, 22 % below nominal).
- **ramp up**: first break at ramp 5 deg (speed 0.179 m/s, 27 % below nominal).
- **Sideways push**: falls start at 1.5 m/s free-body (impulse 19.0 N s; the centre of mass gained 0.79 / 0.95 m/s), pushed to the left (8/8 fell) and right (8/8 fell). Largest push with no fall: 1.25 m/s free-body (impulse 15.9 N s; the centre of mass gained 0.61 / 0.76 m/s), pushed to the left and right. The feet absorb part of every push: the centre of mass gains 45-64 % of the free-body velocity change (pushes of 0.3 m/s and more).

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
| sideways push | none | **1.5 m/s free-body to the right (impulse 19.0 N s; the centre of mass gained 0.95 m/s)**: 8/8 fell | **1.5 m/s free-body to the left (impulse 19.0 N s; the centre of mass gained 0.79 m/s)**: 8/8 fell |
| step up | flat | (not tested) | **step 3 cm**: speed 0.193 m/s, 22 % below nominal |
| ramp up | flat | (not tested) | **ramp 5 deg**: speed 0.179 m/s, 27 % below nominal |

Shaded as training ranges on the charts: the learned policies'. The walk was developed and judged with the same randomisation, on friction 0.4 and 1.0. Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.248 (0.247) | +0.7 % | -0.002 | 0.123 (0.253) | 0.11 (left_fore) | 0.66 (right_fore_pitch) | 2.1 % (right_fore_pitch) | 2.1 % (right_fore_pitch) |
| 2 ms | 0/8 | 0.248 (0.247) | +0.6 % | -0.002 | 0.123 (0.237) | 0.10 (left_fore) | 0.64 (right_fore_pitch) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |
| 5 ms | 0/8 | 0.246 (0.245) | +0.0 % | -0.004 | 0.122 (0.236) | 0.11 (right_fore) | 0.66 (right_fore_pitch) | 2.3 % (right_fore_pitch) | 2.3 % (right_fore_pitch) |
| 10 ms | 0/8 | 0.246 (0.246) | +0.0 % | -0.004 | 0.122 (0.243) | 0.11 (left_fore) | 0.68 (right_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |
| 15 ms | 0/8 | 0.245 (0.244) | -0.5 % | -0.005 | 0.119 (0.247) | 0.09 (right_fore) | 0.67 (right_hind_flexion) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |
| 20 ms | 0/8 | 0.245 (0.245) | -0.4 % | -0.005 | 0.116 (0.226) | 0.10 (right_fore) | 0.70 (right_fore_pitch) | 2.7 % (right_fore_pitch) | 2.7 % (right_fore_pitch) |
| 30 ms | 0/8 | 0.238 (0.229) | -3.2 % | -0.012 | 0.118 (0.236) | 0.15 (left_hind) | 0.76 (right_hind_flexion) | 1.7 % (right_fore_pitch) | 1.7 % (right_fore_pitch) |
| 40 ms | 0/8 | 0.240 (0.223) | -2.6 % | -0.010 | 0.103 (0.195) | 0.13 (right_fore) | 0.75 (right_fore_pitch) | 2.9 % (right_fore_pitch) | 2.9 % (right_fore_pitch) |
| 60 ms | 0/8 | 0.243 (0.243) | -1.2 % | -0.007 | 0.099 (0.217) | 0.15 (right_fore) | 0.74 (right_fore_pitch) | 3.7 % (right_fore_pitch) | 3.7 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `walk60_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.150 (0.150) | +0.6 % | +0.000 | 0.065 (0.113) | 0.09 (right_fore) | 0.63 (right_hind_flexion) | 1.2 % (right_fore_pitch) | 1.2 % (right_fore_pitch) |
| 2 ms | 0/8 | 0.150 (0.150) | +0.6 % | +0.000 | 0.076 (0.153) | 0.09 (right_fore) | 0.63 (right_hind_flexion) | 1.2 % (right_fore_pitch) | 1.2 % (right_fore_pitch) |
| 5 ms | 0/8 | 0.150 (0.150) | +0.5 % | +0.000 | 0.080 (0.137) | 0.09 (right_fore) | 0.64 (right_hind_flexion) | 1.1 % (right_fore_pitch) | 1.1 % (right_fore_pitch) |
| 10 ms | 0/8 | 0.149 (0.149) | +0.0 % | -0.001 | 0.073 (0.138) | 0.10 (right_fore) | 0.64 (left_hind_flexion) | 1.5 % (right_fore_pitch) | 1.5 % (right_fore_pitch) |
| 15 ms | 0/8 | 0.149 (0.149) | -0.1 % | -0.001 | 0.077 (0.169) | 0.11 (right_fore) | 0.65 (right_hind_flexion) | 1.7 % (right_fore_pitch) | 1.7 % (right_fore_pitch) |
| 20 ms | 0/8 | 0.150 (0.150) | +0.5 % | +0.000 | 0.063 (0.124) | 0.12 (right_fore) | 0.65 (right_hind_flexion) | 2.2 % (left_fore_pitch) | 2.2 % (left_fore_pitch) |
| 30 ms | 0/8 | 0.152 (0.152) | +2.0 % | +0.002 | 0.067 (0.149) | 0.15 (left_fore) | 0.72 (right_fore_pitch) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| 40 ms | 0/8 | 0.154 (0.154) | +3.1 % | +0.004 | 0.049 (0.129) | 0.21 (right_fore) | 0.74 (right_fore_pitch) | 3.4 % (right_fore_pitch) | 3.4 % (right_fore_pitch) |
| 60 ms | 0/8 | 0.124 (0.122) | -16.8 % | -0.026 | 0.244 (0.478) | 0.43 (right_fore) | 0.92 (right_fore_pitch) | 5.3 % (right_fore_pitch) | 5.3 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `walk60_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.260 (0.259) | +5.5 % | +0.010 | 0.134 (0.248) | 0.22 (right_fore) | 0.80 (right_hind_flexion) | 0.1 % (left_hind_flexion) | 0.1 % (left_hind_flexion) |
| 0.2 | 0/8 | 0.257 (0.256) | +4.4 % | +0.007 | 0.146 (0.248) | 0.20 (right_fore) | 0.78 (right_hind_flexion) | 0.1 % (left_hind_flexion) | 0.1 % (left_hind_flexion) |
| 0.3 | 0/8 | 0.254 (0.254) | +3.1 % | +0.004 | 0.146 (0.242) | 0.18 (right_fore) | 0.74 (left_hind_flexion) | 0.1 % (left_hind_flexion) | 0.1 % (left_hind_flexion) |
| 0.4 | 0/8 | 0.252 (0.251) | +2.3 % | +0.002 | 0.134 (0.209) | 0.16 (right_fore) | 0.71 (right_hind_flexion) | 0.0 % (left_hind_flexion) | 0.0 % (left_hind_flexion) |
| 0.5 | 0/8 | 0.250 (0.249) | +1.5 % | -0.000 | 0.134 (0.251) | 0.13 (right_fore) | 0.69 (right_hind_flexion) | 0.0 % (right_hind_flexion) | 0.0 % (right_hind_flexion) |
| 0.6 | 0/8 | 0.248 (0.247) | +0.6 % | -0.002 | 0.133 (0.246) | 0.11 (right_hind) | 0.67 (right_hind_flexion) | 0.4 % (right_fore_pitch) | 0.4 % (right_fore_pitch) |
| 0.8 | 0/8 | 0.246 (0.245) | -0.2 % | -0.004 | 0.124 (0.264) | 0.11 (left_fore) | 0.66 (right_fore_pitch) | 2.4 % (right_fore_pitch) | 2.4 % (right_fore_pitch) |
| 0.85 (model) | 0/8 | 0.246 (0.246) | +0.0 % | -0.004 | 0.122 (0.243) | 0.11 (left_fore) | 0.68 (right_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |
| 1 | 0/8 | 0.248 (0.246) | +0.6 % | -0.002 | 0.121 (0.248) | 0.09 (left_fore) | 0.73 (left_hind_pitch) | 2.9 % (right_fore_pitch) | 2.9 % (right_fore_pitch) |
| 1.2 | 0/8 | 0.246 (0.245) | -0.1 % | -0.004 | 0.122 (0.254) | 0.09 (left_fore) | 0.77 (left_hind_pitch) | 3.2 % (right_fore_pitch) | 3.2 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `walk60_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.243 (0.243) | -1.3 % | -0.007 | 0.120 (0.264) | 0.10 (left_fore) | 0.63 (right_fore_pitch) | 2.1 % (right_fore_pitch) | 2.1 % (right_fore_pitch) |
| x0.9 | 0/8 | 0.245 (0.244) | -0.6 % | -0.005 | 0.118 (0.247) | 0.10 (left_fore) | 0.65 (right_fore_pitch) | 2.3 % (right_fore_pitch) | 2.3 % (right_fore_pitch) |
| x1 | 0/8 | 0.246 (0.246) | +0.0 % | -0.004 | 0.122 (0.243) | 0.11 (left_fore) | 0.68 (right_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |
| x1.1 | 0/8 | 0.248 (0.247) | +0.6 % | -0.002 | 0.129 (0.261) | 0.11 (left_fore) | 0.71 (right_fore_pitch) | 2.8 % (right_fore_pitch) | 2.8 % (right_fore_pitch) |
| x1.2 | 0/8 | 0.248 (0.247) | +0.9 % | -0.002 | 0.122 (0.256) | 0.11 (left_fore) | 0.73 (right_fore_pitch) | 3.0 % (right_fore_pitch) | 3.0 % (right_fore_pitch) |
| x1.4 | 0/8 | 0.248 (0.248) | +0.8 % | -0.002 | 0.126 (0.246) | 0.11 (left_fore) | 0.78 (left_hind_pitch) | 3.3 % (right_fore_pitch) | 3.3 % (right_fore_pitch) |
| x1.6 | 0/8 | 0.247 (0.246) | +0.3 % | -0.003 | 0.126 (0.245) | 0.10 (left_fore) | 0.88 (right_hind_pitch) | 3.6 % (right_fore_pitch) | 3.6 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `walk60_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.246 (0.246) | +0.0 % | -0.004 | 0.122 (0.243) | 0.11 (left_fore) | 0.68 (right_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |
| +0.5 kg | 0/8 | 0.247 (0.246) | +0.2 % | -0.003 | 0.119 (0.246) | 0.11 (left_fore) | 0.69 (right_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |
| +1 kg | 0/8 | 0.247 (0.246) | +0.3 % | -0.003 | 0.124 (0.241) | 0.11 (left_fore) | 0.69 (right_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |
| +2 kg | 0/8 | 0.246 (0.244) | -0.2 % | -0.004 | 0.128 (0.249) | 0.10 (left_fore) | 0.74 (right_hind_flexion) | 2.2 % (right_fore_pitch) | 2.2 % (right_fore_pitch) |
| +3 kg | 0/8 | 0.244 (0.240) | -1.1 % | -0.006 | 0.128 (0.237) | 0.10 (right_hind) | 0.81 (right_hind_flexion) | 1.6 % (right_fore_pitch) | 1.6 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `walk60_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.249 (0.248) | +1.1 % | -0.001 | 0.122 (0.247) | 0.11 (left_fore) | 0.60 (right_hind_pitch) | 0.0 % (left_fore_roll) | 13.7 % (right_hind_pitch) |
| x0.55 | 0/8 | 0.250 (0.249) | +1.4 % | -0.000 | 0.126 (0.253) | 0.10 (left_fore) | 0.62 (right_hind_pitch) | 0.0 % (left_fore_roll) | 5.3 % (right_fore_pitch) |
| x0.7 | 0/8 | 0.250 (0.249) | +1.5 % | -0.000 | 0.126 (0.261) | 0.10 (left_fore) | 0.63 (left_hind_pitch) | 0.0 % (left_fore_roll) | 4.1 % (right_fore_pitch) |
| x0.85 | 0/8 | 0.249 (0.248) | +1.0 % | -0.001 | 0.119 (0.255) | 0.11 (left_fore) | 0.67 (left_hind_flexion) | 0.0 % (left_fore_roll) | 3.1 % (right_fore_pitch) |
| x1 | 0/8 | 0.246 (0.246) | +0.0 % | -0.004 | 0.122 (0.243) | 0.11 (left_fore) | 0.68 (right_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `walk60_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.216 (0.216) | -12.1 % | -0.034 | 0.097 (0.208) | 0.17 (right_fore) | 0.59 (right_fore_pitch) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| x0.6 | 0/8 | 0.233 (0.231) | -5.4 % | -0.017 | 0.115 (0.232) | 0.11 (right_hind) | 0.60 (right_hind_flexion) | 1.1 % (right_fore_pitch) | 1.1 % (right_fore_pitch) |
| x0.8 | 0/8 | 0.242 (0.241) | -1.8 % | -0.008 | 0.121 (0.241) | 0.11 (left_fore) | 0.64 (right_fore_pitch) | 2.7 % (right_fore_pitch) | 2.7 % (right_fore_pitch) |
| x1 | 0/8 | 0.246 (0.246) | +0.0 % | -0.004 | 0.122 (0.243) | 0.11 (left_fore) | 0.68 (right_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |
| x1.2 | 0/8 | 0.248 (0.247) | +0.6 % | -0.002 | 0.126 (0.274) | 0.09 (right_fore) | 0.77 (left_hind_flexion) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| x1.5 | 0/8 | 0.247 (0.246) | +0.2 % | -0.003 | 0.129 (0.254) | 0.10 (right_fore) | 0.85 (left_hind_flexion) | 3.0 % (right_fore_pitch) | 3.0 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `walk60_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 45-64 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.0 | 190 | 0.95 | 8/8 | all fell | | | | |
| -1.25 m/s | 15.9 | 159 | 0.76 | 0/8 | 0.246 | 0.147 (0.153) | 1.25 (1.87) | 0 | 24.9 |
| -1 m/s | 12.7 | 127 | 0.60 | 0/8 | 0.247 | 0.077 (0.080) | 0.65 (0.69) | 0 | 16.1 |
| -0.75 m/s | 9.5 | 95 | 0.47 | 0/8 | 0.247 | 0.045 (0.046) | 0.18 (0.42) | 0 | 9.6 |
| -0.5 m/s | 6.3 | 63 | 0.32 | 0/8 | 0.246 | 0.031 (0.035) | 0.10 (0.79) | 0 | 6.1 |
| -0.4 m/s | 5.1 | 51 | 0.25 | 0/8 | 0.246 | 0.026 (0.027) | 0.10 (0.10) | 0 | 5.2 |
| -0.3 m/s | 3.8 | 38 | 0.19 | 0/8 | 0.247 | 0.022 (0.022) | 0.10 (0.10) | 0 | 5.1 |
| -0.2 m/s | 2.5 | 25 | 0.13 | 0/8 | 0.247 | 0.016 (0.018) | 0.10 (0.10) | 0 | 3.7 |
| -0.1 m/s | 1.3 | 13 | 0.06 | 0/8 | 0.247 | 0.011 (0.017) | 0.10 (0.10) | 0 | 2.6 |
| none | 0.0 | 0 |  | 0/8 | 0.246 | (no push) | 0.10 (0.10) | 0 | 2.5 |
| +0.1 m/s | 1.3 | 13 | 0.06 | 0/8 | 0.246 | 0.009 (0.014) | 0.10 (0.10) | 0 | 2.6 |
| +0.2 m/s | 2.5 | 25 | 0.13 | 0/8 | 0.247 | 0.016 (0.023) | 0.10 (0.10) | 0 | 3.5 |
| +0.3 m/s | 3.8 | 38 | 0.19 | 0/8 | 0.246 | 0.020 (0.024) | 0.10 (0.10) | 0 | 5.2 |
| +0.4 m/s | 5.1 | 51 | 0.24 | 0/8 | 0.246 | 0.021 (0.025) | 0.10 (0.44) | 0 | 5.9 |
| +0.5 m/s | 6.3 | 63 | 0.28 | 0/8 | 0.247 | 0.024 (0.025) | 0.10 (0.47) | 0 | 5.7 |
| +0.75 m/s | 9.5 | 95 | 0.34 | 0/8 | 0.246 | 0.040 (0.042) | 0.55 (0.58) | 0 | 7.9 |
| +1 m/s | 12.7 | 127 | 0.45 | 0/8 | 0.246 | 0.080 (0.082) | 0.82 (0.83) | 0 | 14.6 |
| +1.25 m/s | 15.9 | 159 | 0.61 | 0/8 | 0.230 | 0.149 (0.154) | 1.25 (2.28) | 0 | 29.3 |
| +1.5 m/s | 19.0 | 190 | 0.79 | 8/8 | all fell | | | | |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 0.10 s median and 0.10 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `walk60_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.246 (0.246) | +0.0 % | -0.004 | 0.122 (0.243) | 0.11 (left_fore) | 0.68 (right_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |
| step 1 cm | 0/8 | 0.245 (0.244) | -0.7 % | -0.005 | 0.117 (0.255) | 0.10 (left_fore) | 0.68 (left_hind_flexion) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |
| step 2 cm | 0/8 | 0.232 (0.211) | -5.7 % | -0.018 | 0.096 (0.200) | 0.12 (right_hind) | 0.73 (right_hind_flexion) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| step 3 cm | 0/8 | 0.193 (0.136) | -21.6 % | -0.057 | 0.110 (0.235) | 0.23 (right_fore) | 0.94 (left_hind_pitch) | 5.2 % (left_hind_pitch) | 5.2 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `walk60_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.246 (0.246) | +0.0 % | -0.004 | 0.122 (0.243) | 0.11 (left_fore) | 0.68 (right_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |
| ramp 3 deg | 0/8 | 0.201 (0.200) | -18.4 % | -0.049 | 0.135 (0.299) | 0.44 (left_hind) | 0.94 (left_hind_flexion) | 5.4 % (left_hind_flexion) | 5.4 % (left_hind_flexion) |
| ramp 5 deg | 0/8 | 0.179 (0.177) | -27.3 % | -0.071 | 0.214 (0.393) | 0.53 (right_hind) | 0.99 (left_hind_flexion) | 6.5 % (left_hind_flexion) | 6.5 % (left_hind_flexion) |
| ramp 8 deg | 0/8 | 0.129 (0.109) | -47.5 % | -0.121 | 0.418 (0.798) | 0.70 (right_hind) | 1.09 (right_hind_flexion) | 7.4 % (right_hind_flexion) | 7.4 % (right_hind_flexion) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `walk60_terrain.png`.

