# Robustness sweep: trike60-pipe1 (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-19. One policy, so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Policy `trike60_2007040_steps.zip` (sha256 dc06d30105b201bf...), run commit 6aa3a679; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 a82a0c4d206fdfb7...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 12.0 min on 2 worker processes (565 episodes).
- Reproduction check (gait_pipe1's target case, randomisation on, seeds 100000-100004): mean speed 0.2385 m/s against the analysis's 0.2385 m/s; per-episode speeds identical: True.

## Headline

- **No fall and no speed drop of more than 20 %** in: delay, 0.15 m/s (0 ms to 60 ms); friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5).
- **delay, 0.25 m/s**: first break at 60 ms (speed 0.177 m/s, 26 % below nominal).
- **step up**: first break at step 3 cm (speed 0.145 m/s, 39 % below nominal).
- **ramp up**: first break at ramp 8 deg (speed 0.188 m/s, 21 % below nominal).
- **Sideways push**: falls start at 1.25 m/s free-body (impulse 15.9 N s; the centre of mass gained 0.68 m/s), pushed to the left (1/8 fell) and right (4/8 fell). Largest push with no fall: 1 m/s free-body (impulse 12.7 N s; the centre of mass gained 0.52 m/s), pushed to the left and right. The feet absorb part of every push: the centre of mass gains 46-58 % of the free-body velocity change (pushes of 0.3 m/s and more).

## Where it breaks

First point outward from nominal with a fall or a speed more than 20 % below the nominal point's (same command). "none": no such point in the tested range.

| Sweep | Nominal | Low side | High side |
|---|---|---|---|
| delay, 0.25 m/s | 10 ms | none | **60 ms**: speed 0.177 m/s, 26 % below nominal |
| delay, 0.15 m/s | 10 ms | none | none |
| friction | 0.85 (model) | none | none |
| mass, every body | x1 | none | none |
| payload on the chassis | +0 kg | (not tested) | none |
| force limit | x1 | none | (not tested) |
| position gain kp | x1 | none | none |
| sideways push | none | **1.25 m/s free-body to the right (impulse 15.9 N s; the centre of mass gained 0.68 m/s)**: 4/8 fell | **1.25 m/s free-body to the left (impulse 15.9 N s; the centre of mass gained 0.68 m/s)**: 1/8 fell |
| step up | flat | (not tested) | **step 3 cm**: speed 0.145 m/s, 39 % below nominal |
| ramp up | flat | (not tested) | **ramp 8 deg**: speed 0.188 m/s, 21 % below nominal |

Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.257 (0.256) | +7.5 % | +0.007 | 0.088 (0.168) | 0.26 (left_hind) | 1.10 (left_hind_flexion) | 7.3 % (left_fore_pitch) | 7.3 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.253 (0.251) | +5.5 % | +0.003 | 0.088 (0.157) | 0.26 (left_hind) | 1.12 (left_hind_flexion) | 7.2 % (left_fore_pitch) | 7.2 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.246 (0.245) | +3.0 % | -0.004 | 0.066 (0.167) | 0.25 (left_hind) | 1.13 (left_hind_flexion) | 7.7 % (left_fore_pitch) | 7.7 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.060 (0.182) | 0.24 (left_hind) | 1.13 (left_hind_flexion) | 8.7 % (left_fore_pitch) | 8.7 % (left_fore_pitch) |
| 15 ms | 0/8 | 0.231 (0.230) | -3.3 % | -0.019 | 0.076 (0.163) | 0.28 (left_fore) | 1.14 (left_fore_pitch) | 10.8 % (left_fore_pitch) | 10.8 % (left_fore_pitch) |
| 20 ms | 0/8 | 0.235 (0.232) | -1.6 % | -0.015 | 0.313 (0.499) | 0.33 (right_fore) | 1.21 (left_fore_pitch) | 12.1 % (left_fore_pitch) | 12.1 % (left_fore_pitch) |
| 30 ms | 0/8 | 0.221 (0.218) | -7.6 % | -0.029 | 0.206 (0.301) | 0.35 (right_fore) | 1.22 (left_fore_pitch) | 12.2 % (left_fore_pitch) | 12.2 % (left_fore_pitch) |
| 40 ms | 0/8 | 0.203 (0.194) | -15.2 % | -0.047 | 0.471 (0.648) | 0.48 (right_fore) | 1.34 (right_fore_pitch) | 15.1 % (left_fore_pitch) | 15.1 % (left_fore_pitch) |
| 60 ms | 0/8 | 0.177 (0.165) | -25.9 % | -0.073 | 0.359 (0.587) | 0.74 (right_fore) | 1.59 (right_fore_pitch) | 22.5 % (right_fore_pitch) | 22.5 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `pipe1_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.156 (0.156) | +8.2 % | +0.006 | 0.042 (0.083) | 0.36 (right_hind) | 1.04 (left_fore_pitch) | 6.4 % (left_fore_pitch) | 6.4 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.153 (0.151) | +5.6 % | +0.003 | 0.042 (0.088) | 0.35 (right_hind) | 1.02 (left_fore_pitch) | 6.2 % (left_fore_pitch) | 6.2 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.149 (0.147) | +2.9 % | -0.001 | 0.041 (0.098) | 0.35 (right_hind) | 1.02 (left_fore_pitch) | 6.0 % (left_fore_pitch) | 6.0 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.144 (0.143) | +0.0 % | -0.006 | 0.043 (0.113) | 0.35 (right_hind) | 1.01 (left_fore_pitch) | 5.9 % (left_fore_pitch) | 5.9 % (left_fore_pitch) |
| 15 ms | 0/8 | 0.141 (0.140) | -2.4 % | -0.009 | 0.037 (0.079) | 0.38 (right_hind) | 1.04 (left_fore_pitch) | 6.5 % (left_fore_pitch) | 6.5 % (left_fore_pitch) |
| 20 ms | 0/8 | 0.149 (0.147) | +2.8 % | -0.001 | 0.057 (0.103) | 0.46 (left_hind) | 1.14 (right_fore_pitch) | 6.8 % (left_fore_pitch) | 6.8 % (left_fore_pitch) |
| 30 ms | 0/8 | 0.141 (0.138) | -2.3 % | -0.009 | 0.171 (0.291) | 0.50 (left_hind) | 1.22 (right_fore_pitch) | 7.8 % (right_fore_pitch) | 7.8 % (right_fore_pitch) |
| 40 ms | 0/8 | 0.140 (0.137) | -2.8 % | -0.010 | 0.068 (0.112) | 0.63 (left_hind) | 1.36 (right_fore_pitch) | 12.4 % (right_fore_pitch) | 12.4 % (right_fore_pitch) |
| 60 ms | 0/8 | 0.134 (0.127) | -6.9 % | -0.016 | 0.096 (0.170) | 0.74 (left_hind) | 1.47 (right_fore_pitch) | 16.9 % (right_fore_pitch) | 16.9 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `pipe1_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.242 (0.240) | +1.2 % | -0.008 | 0.047 (0.153) | 0.57 (left_fore) | 0.93 (left_hind_flexion) | 3.7 % (right_hind_flexion) | 3.7 % (right_hind_flexion) |
| 0.2 | 0/8 | 0.242 (0.240) | +1.0 % | -0.008 | 0.075 (0.156) | 0.47 (left_fore) | 0.94 (left_hind_flexion) | 3.7 % (right_hind_flexion) | 3.7 % (right_hind_flexion) |
| 0.3 | 0/8 | 0.240 (0.239) | +0.5 % | -0.010 | 0.123 (0.264) | 0.38 (left_fore) | 0.98 (left_hind_flexion) | 3.7 % (left_fore_pitch) | 3.7 % (left_fore_pitch) |
| 0.4 | 0/8 | 0.240 (0.239) | +0.4 % | -0.010 | 0.149 (0.300) | 0.32 (left_fore) | 1.02 (left_hind_flexion) | 5.3 % (left_fore_pitch) | 5.3 % (left_fore_pitch) |
| 0.5 | 0/8 | 0.240 (0.238) | +0.4 % | -0.010 | 0.123 (0.260) | 0.29 (left_hind) | 1.05 (left_hind_flexion) | 6.3 % (left_fore_pitch) | 6.3 % (left_fore_pitch) |
| 0.6 | 0/8 | 0.240 (0.238) | +0.2 % | -0.010 | 0.091 (0.234) | 0.27 (left_hind) | 1.07 (left_hind_flexion) | 6.8 % (left_fore_pitch) | 6.8 % (left_fore_pitch) |
| 0.8 | 0/8 | 0.240 (0.238) | +0.2 % | -0.010 | 0.074 (0.179) | 0.24 (left_hind) | 1.13 (left_hind_flexion) | 8.1 % (left_fore_pitch) | 8.1 % (left_fore_pitch) |
| 0.85 (model) | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.060 (0.182) | 0.24 (left_hind) | 1.13 (left_hind_flexion) | 8.7 % (left_fore_pitch) | 8.7 % (left_fore_pitch) |
| 1 | 0/8 | 0.238 (0.237) | -0.6 % | -0.012 | 0.071 (0.186) | 0.23 (left_hind) | 1.17 (left_hind_flexion) | 10.2 % (left_fore_pitch) | 10.2 % (left_fore_pitch) |
| 1.2 | 0/8 | 0.235 (0.233) | -1.6 % | -0.015 | 0.071 (0.190) | 0.22 (left_fore) | 1.21 (right_fore_pitch) | 11.5 % (left_fore_pitch) | 11.5 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `pipe1_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.239 (0.237) | -0.2 % | -0.011 | 0.062 (0.118) | 0.26 (left_hind) | 1.01 (left_fore_pitch) | 7.5 % (left_fore_pitch) | 7.5 % (left_fore_pitch) |
| x0.9 | 0/8 | 0.239 (0.237) | -0.0 % | -0.011 | 0.059 (0.143) | 0.25 (left_hind) | 1.05 (left_fore_pitch) | 8.2 % (left_fore_pitch) | 8.2 % (left_fore_pitch) |
| x1 | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.060 (0.182) | 0.24 (left_hind) | 1.13 (left_hind_flexion) | 8.7 % (left_fore_pitch) | 8.7 % (left_fore_pitch) |
| x1.1 | 0/8 | 0.240 (0.237) | +0.1 % | -0.010 | 0.084 (0.208) | 0.23 (left_hind) | 1.22 (left_hind_flexion) | 9.5 % (left_fore_pitch) | 9.5 % (left_fore_pitch) |
| x1.2 | 0/8 | 0.240 (0.238) | +0.1 % | -0.010 | 0.081 (0.186) | 0.24 (right_hind) | 1.31 (left_hind_flexion) | 10.2 % (left_fore_pitch) | 10.2 % (left_fore_pitch) |
| x1.4 | 0/8 | 0.239 (0.237) | -0.2 % | -0.011 | 0.072 (0.192) | 0.26 (right_hind) | 1.47 (left_hind_flexion) | 12.2 % (left_fore_pitch) | 12.2 % (left_fore_pitch) |
| x1.6 | 0/8 | 0.237 (0.235) | -1.1 % | -0.013 | 0.073 (0.144) | 0.29 (right_hind) | 1.64 (left_hind_flexion) | 16.0 % (left_fore_pitch) | 16.0 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `pipe1_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.060 (0.182) | 0.24 (left_hind) | 1.13 (left_hind_flexion) | 8.7 % (left_fore_pitch) | 8.7 % (left_fore_pitch) |
| +0.5 kg | 0/8 | 0.239 (0.238) | -0.0 % | -0.011 | 0.070 (0.199) | 0.24 (left_hind) | 1.17 (left_hind_flexion) | 8.9 % (left_fore_pitch) | 8.9 % (left_fore_pitch) |
| +1 kg | 0/8 | 0.239 (0.236) | -0.2 % | -0.011 | 0.066 (0.169) | 0.24 (right_hind) | 1.22 (left_hind_flexion) | 9.1 % (left_fore_pitch) | 9.1 % (left_fore_pitch) |
| +2 kg | 0/8 | 0.237 (0.235) | -0.8 % | -0.013 | 0.066 (0.140) | 0.25 (right_hind) | 1.29 (left_hind_flexion) | 10.3 % (left_fore_pitch) | 10.3 % (left_fore_pitch) |
| +3 kg | 0/8 | 0.237 (0.235) | -1.1 % | -0.013 | 0.066 (0.193) | 0.26 (right_hind) | 1.36 (left_hind_flexion) | 10.9 % (left_fore_pitch) | 10.9 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `pipe1_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.225 (0.223) | -5.8 % | -0.025 | 0.070 (0.147) | 0.49 (right_hind) | 0.90 (right_fore_flexion) | 0.0 % (left_fore_roll) | 60.7 % (left_hind_flexion) |
| x0.55 | 0/8 | 0.234 (0.232) | -2.1 % | -0.016 | 0.155 (0.312) | 0.25 (left_hind) | 1.05 (left_hind_flexion) | 0.0 % (left_fore_roll) | 34.5 % (left_hind_flexion) |
| x0.7 | 0/8 | 0.238 (0.236) | -0.6 % | -0.012 | 0.106 (0.268) | 0.24 (left_hind) | 1.08 (left_hind_flexion) | 0.0 % (left_fore_roll) | 16.1 % (left_fore_pitch) |
| x0.85 | 0/8 | 0.239 (0.237) | -0.2 % | -0.011 | 0.089 (0.188) | 0.24 (left_hind) | 1.11 (left_hind_flexion) | 0.0 % (left_fore_roll) | 11.6 % (left_fore_pitch) |
| x1 | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.060 (0.182) | 0.24 (left_hind) | 1.13 (left_hind_flexion) | 8.7 % (left_fore_pitch) | 8.7 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `pipe1_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.222 (0.221) | -7.1 % | -0.028 | 0.218 (0.317) | 0.32 (right_hind) | 1.13 (left_hind_flexion) | 3.9 % (left_fore_pitch) | 3.9 % (left_fore_pitch) |
| x0.6 | 0/8 | 0.234 (0.232) | -2.4 % | -0.016 | 0.077 (0.162) | 0.22 (right_hind) | 1.16 (left_hind_flexion) | 5.1 % (left_fore_pitch) | 5.1 % (left_fore_pitch) |
| x0.8 | 0/8 | 0.238 (0.236) | -0.7 % | -0.012 | 0.058 (0.140) | 0.22 (left_hind) | 1.14 (left_hind_flexion) | 6.7 % (left_fore_pitch) | 6.7 % (left_fore_pitch) |
| x1 | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.060 (0.182) | 0.24 (left_hind) | 1.13 (left_hind_flexion) | 8.7 % (left_fore_pitch) | 8.7 % (left_fore_pitch) |
| x1.2 | 0/8 | 0.241 (0.239) | +0.5 % | -0.009 | 0.092 (0.225) | 0.26 (left_hind) | 1.17 (left_fore_pitch) | 10.6 % (left_fore_pitch) | 10.6 % (left_fore_pitch) |
| x1.5 | 0/8 | 0.242 (0.241) | +1.2 % | -0.008 | 0.166 (0.295) | 0.32 (left_hind) | 1.32 (right_fore_pitch) | 12.8 % (left_fore_pitch) | 12.8 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `pipe1_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 46-58 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.0 | 190 | 0.86 | 8/8 | all fell | | | | |
| -1.25 m/s | 15.9 | 159 | 0.68 | 4/8 | 0.233 | 0.177 (0.228) | 0.91 (1.15) | 0 | 36.2 |
| -1 m/s | 12.7 | 127 | 0.52 | 0/8 | 0.233 | 0.092 (0.134) | 0.84 (1.34) | 0 | 21.8 |
| -0.75 m/s | 9.5 | 95 | 0.37 | 0/8 | 0.239 | 0.055 (0.073) | 0.56 (0.85) | 0 | 9.8 |
| -0.5 m/s | 6.3 | 63 | 0.28 | 0/8 | 0.239 | 0.036 (0.052) | 0.16 (0.35) | 0 | 4.7 |
| -0.4 m/s | 5.1 | 51 | 0.22 | 0/8 | 0.239 | 0.028 (0.044) | 0.10 (0.33) | 0 | 4.0 |
| -0.3 m/s | 3.8 | 38 | 0.16 | 0/8 | 0.239 | 0.022 (0.031) | 0.10 (0.34) | 0 | 3.5 |
| -0.2 m/s | 2.5 | 25 | 0.11 | 0/8 | 0.239 | 0.017 (0.027) | 0.10 (0.10) | 0 | 3.7 |
| -0.1 m/s | 1.3 | 13 | 0.05 | 0/8 | 0.239 | 0.010 (0.016) | 0.10 (0.10) | 0 | 3.7 |
| none | 0.0 | 0 |  | 0/8 | 0.239 | (no push) | 0.10 (0.81) | 0 | 3.6 |
| +0.1 m/s | 1.3 | 13 | 0.05 | 0/8 | 0.239 | 0.014 (0.035) | 0.10 (0.10) | 0 | 3.3 |
| +0.2 m/s | 2.5 | 25 | 0.09 | 0/8 | 0.239 | 0.023 (0.039) | 0.10 (0.10) | 0 | 3.6 |
| +0.3 m/s | 3.8 | 38 | 0.14 | 0/8 | 0.239 | 0.031 (0.046) | 0.10 (0.10) | 0 | 3.9 |
| +0.4 m/s | 5.1 | 51 | 0.19 | 0/8 | 0.239 | 0.037 (0.047) | 0.26 (1.34) | 0 | 4.1 |
| +0.5 m/s | 6.3 | 63 | 0.23 | 0/8 | 0.239 | 0.044 (0.059) | 0.31 (0.63) | 0 | 4.4 |
| +0.75 m/s | 9.5 | 95 | 0.37 | 0/8 | 0.239 | 0.063 (0.072) | 0.47 (0.81) | 0 | 6.1 |
| +1 m/s | 12.7 | 127 | 0.52 | 0/8 | 0.238 | 0.097 (0.115) | 0.64 (0.98) | 0 | 11.5 |
| +1.25 m/s | 15.9 | 159 | 0.68 | 1/8 | 0.227 | 0.173 (0.256) | 1.06 (2.06) | 0 | 35.6 |
| +1.5 m/s | 19.0 | 190 | 0.87 | 8/8 | all fell | | | | |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 0.10 s median and 0.81 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `pipe1_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.060 (0.182) | 0.24 (left_hind) | 1.13 (left_hind_flexion) | 8.7 % (left_fore_pitch) | 8.7 % (left_fore_pitch) |
| step 1 cm | 0/8 | 0.239 (0.237) | -0.1 % | -0.011 | 0.081 (0.172) | 0.24 (left_hind) | 1.14 (left_hind_flexion) | 8.7 % (left_fore_pitch) | 8.7 % (left_fore_pitch) |
| step 2 cm | 0/8 | 0.228 (0.219) | -4.6 % | -0.022 | 0.085 (0.263) | 0.27 (left_hind) | 1.16 (left_hind_flexion) | 9.8 % (left_fore_pitch) | 9.8 % (left_fore_pitch) |
| step 3 cm | 0/8 | 0.145 (0.027) | -39.3 % | -0.105 | 0.116 (0.280) | 1.37 (right_fore) | 1.31 (right_fore_pitch) | 14.9 % (left_fore_pitch) | 14.9 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `pipe1_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.060 (0.182) | 0.24 (left_hind) | 1.13 (left_hind_flexion) | 8.7 % (left_fore_pitch) | 8.7 % (left_fore_pitch) |
| ramp 3 deg | 0/8 | 0.223 (0.221) | -6.9 % | -0.027 | 0.189 (0.331) | 0.35 (right_fore) | 1.17 (left_hind_flexion) | 10.7 % (left_fore_pitch) | 10.7 % (left_fore_pitch) |
| ramp 5 deg | 0/8 | 0.208 (0.206) | -13.2 % | -0.042 | 0.134 (0.268) | 0.48 (right_fore) | 1.23 (right_fore_pitch) | 11.9 % (left_fore_pitch) | 11.9 % (left_fore_pitch) |
| ramp 8 deg | 0/8 | 0.188 (0.186) | -21.4 % | -0.062 | 0.163 (0.306) | 0.67 (right_fore) | 1.32 (right_fore_pitch) | 13.6 % (left_fore_pitch) | 13.6 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `pipe1_terrain.png`.

