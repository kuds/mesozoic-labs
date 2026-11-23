# Robustness sweep: trike60-run3-s0 (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-23. One policy, so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Policy `trike60_10002432_steps.zip` (sha256 f76ac23bfda4dd55...), run commit efd1e1c7; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 402819fcc1c3879d...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 9.9 min on 2 worker processes (565 episodes).
- Reproduction check (gait_pipe1's target case, randomisation on, seeds 100000-100004): mean speed 0.2414 m/s against the analysis's 0.2385 m/s; per-episode speeds identical: False.

## Headline

- **No fall and no speed drop of more than 20 %** in: friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5); step up (flat to step 3 cm); ramp up (flat to ramp 8 deg).
- **delay, 0.25 m/s**: first break at 60 ms (4/8 fell, speed 0.192 m/s, 21 % below nominal).
- **delay, 0.15 m/s**: first break at 40 ms (2/8 fell).
- **Sideways push**: falls start at 1 m/s free-body (impulse 12.7 N s; the centre of mass gained 0.55 / 0.59 m/s), pushed to the left (4/8 fell) and right (5/8 fell). Largest push with no fall: 0.75 m/s free-body (impulse 9.5 N s; the centre of mass gained 0.38 / 0.41 m/s), pushed to the left and right. The feet absorb part of every push: the centre of mass gains 45-67 % of the free-body velocity change (pushes of 0.3 m/s and more).

## Where it breaks

First point outward from nominal with a fall or a speed more than 20 % below the nominal point's (same command). "none": no such point in the tested range.

| Sweep | Nominal | Low side | High side |
|---|---|---|---|
| delay, 0.25 m/s | 10 ms | none | **60 ms**: 4/8 fell; speed 0.192 m/s, 21 % below nominal |
| delay, 0.15 m/s | 10 ms | none | **40 ms**: 2/8 fell |
| friction | 0.85 (model) | none | none |
| mass, every body | x1 | none | none |
| payload on the chassis | +0 kg | (not tested) | none |
| force limit | x1 | none | (not tested) |
| position gain kp | x1 | none | none |
| sideways push | none | **1 m/s free-body to the right (impulse 12.7 N s; the centre of mass gained 0.59 m/s)**: 5/8 fell | **1 m/s free-body to the left (impulse 12.7 N s; the centre of mass gained 0.55 m/s)**: 4/8 fell |
| step up | flat | (not tested) | none |
| ramp up | flat | (not tested) | none |

Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.258 (0.257) | +6.2 % | +0.008 | 0.131 (0.281) | 0.19 (left_hind) | 0.85 (left_hind_flexion) | 4.0 % (right_fore_pitch) | 4.0 % (right_fore_pitch) |
| 2 ms | 0/8 | 0.254 (0.253) | +4.8 % | +0.004 | 0.210 (0.327) | 0.16 (left_hind) | 0.84 (right_fore_pitch) | 3.8 % (right_fore_pitch) | 3.8 % (right_fore_pitch) |
| 5 ms | 0/8 | 0.249 (0.247) | +2.5 % | -0.001 | 0.293 (0.493) | 0.14 (left_hind) | 0.86 (right_hind_pitch) | 3.7 % (right_fore_pitch) | 3.7 % (right_fore_pitch) |
| 10 ms | 0/8 | 0.243 (0.241) | +0.0 % | -0.007 | 0.232 (0.343) | 0.13 (left_hind) | 0.86 (right_fore_pitch) | 4.2 % (right_fore_pitch) | 4.2 % (right_fore_pitch) |
| 15 ms | 0/8 | 0.239 (0.238) | -1.4 % | -0.011 | 0.084 (0.202) | 0.14 (left_hind) | 0.94 (right_fore_pitch) | 4.6 % (right_fore_pitch) | 4.6 % (right_fore_pitch) |
| 20 ms | 0/8 | 0.252 (0.248) | +3.7 % | +0.002 | 0.603 (0.734) | 0.18 (right_hind) | 1.24 (right_hind_pitch) | 10.2 % (right_hind_pitch) | 10.2 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.240 (0.235) | -1.3 % | -0.010 | 0.533 (0.814) | 0.19 (right_hind) | 1.26 (right_hind_pitch) | 11.9 % (right_hind_pitch) | 11.9 % (right_hind_pitch) |
| 40 ms | 0/8 | 0.224 (0.208) | -7.6 % | -0.026 | 0.943 (1.102) | 0.30 (left_hind) | 1.28 (left_hind_pitch) | 12.7 % (left_hind_pitch) | 12.7 % (left_hind_pitch) |
| 60 ms | 4/8 | 0.192 (0.175) | -21.0 % | -0.058 | 0.845 (0.903) | 0.41 (left_hind) | 1.29 (left_hind_pitch) | 12.6 % (left_hind_pitch) | 12.6 % (left_hind_pitch) |

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `run3-s0_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.154 (0.154) | +7.1 % | +0.004 | 0.040 (0.112) | 0.28 (left_hind) | 0.87 (right_hind_pitch) | 1.7 % (right_fore_pitch) | 1.7 % (right_fore_pitch) |
| 2 ms | 0/8 | 0.152 (0.151) | +5.1 % | +0.002 | 0.033 (0.135) | 0.26 (left_hind) | 0.88 (right_hind_pitch) | 1.6 % (right_fore_pitch) | 1.6 % (right_fore_pitch) |
| 5 ms | 0/8 | 0.148 (0.147) | +2.5 % | -0.002 | 0.038 (0.096) | 0.24 (left_hind) | 0.87 (right_hind_pitch) | 1.5 % (right_fore_pitch) | 1.5 % (right_fore_pitch) |
| 10 ms | 0/8 | 0.144 (0.143) | +0.0 % | -0.006 | 0.031 (0.096) | 0.23 (left_hind) | 0.88 (left_hind_flexion) | 1.6 % (right_fore_pitch) | 1.6 % (right_fore_pitch) |
| 15 ms | 0/8 | 0.143 (0.142) | -1.1 % | -0.007 | 0.068 (0.110) | 0.22 (left_hind) | 0.88 (left_hind_flexion) | 1.6 % (right_fore_pitch) | 1.6 % (right_fore_pitch) |
| 20 ms | 0/8 | 0.153 (0.147) | +5.9 % | +0.003 | 0.388 (0.477) | 0.30 (left_hind) | 1.24 (right_hind_pitch) | 11.2 % (right_hind_pitch) | 11.2 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.147 (0.141) | +2.2 % | -0.003 | 0.362 (0.732) | 0.35 (left_hind) | 1.21 (right_hind_pitch) | 11.4 % (right_hind_pitch) | 11.4 % (right_hind_pitch) |
| 40 ms | 2/8 | 0.142 (0.135) | -1.7 % | -0.008 | 0.523 (0.611) | 0.46 (left_hind) | 1.29 (left_hind_pitch) | 12.6 % (left_hind_pitch) | 12.6 % (left_hind_pitch) |
| 60 ms | 7/8 | 0.118 (0.118) | -17.9 % | -0.032 | 0.419 (0.419) | 0.63 (left_hind) | 1.50 (left_hind_pitch) | 20.7 % (left_hind_pitch) | 20.7 % (left_hind_pitch) |

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `run3-s0_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.237 (0.236) | -2.3 % | -0.013 | 0.346 (0.480) | 0.32 (right_hind) | 0.83 (right_fore_pitch) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |
| 0.2 | 0/8 | 0.237 (0.235) | -2.3 % | -0.013 | 0.320 (0.426) | 0.27 (left_hind) | 0.82 (right_fore_pitch) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |
| 0.3 | 0/8 | 0.240 (0.238) | -1.2 % | -0.010 | 0.152 (0.275) | 0.23 (left_hind) | 0.79 (right_fore_pitch) | 2.4 % (right_fore_pitch) | 2.4 % (right_fore_pitch) |
| 0.4 | 0/8 | 0.241 (0.240) | -0.8 % | -0.009 | 0.140 (0.240) | 0.20 (left_hind) | 0.80 (right_fore_pitch) | 2.8 % (right_fore_pitch) | 2.8 % (right_fore_pitch) |
| 0.5 | 0/8 | 0.241 (0.240) | -0.7 % | -0.009 | 0.130 (0.268) | 0.18 (left_hind) | 0.80 (right_fore_pitch) | 3.1 % (right_fore_pitch) | 3.1 % (right_fore_pitch) |
| 0.6 | 0/8 | 0.242 (0.240) | -0.5 % | -0.008 | 0.081 (0.158) | 0.16 (left_hind) | 0.82 (right_fore_pitch) | 3.3 % (right_fore_pitch) | 3.3 % (right_fore_pitch) |
| 0.8 | 0/8 | 0.242 (0.240) | -0.3 % | -0.008 | 0.186 (0.306) | 0.13 (left_hind) | 0.86 (right_fore_pitch) | 4.0 % (right_fore_pitch) | 4.0 % (right_fore_pitch) |
| 0.85 (model) | 0/8 | 0.243 (0.241) | +0.0 % | -0.007 | 0.232 (0.343) | 0.13 (left_hind) | 0.86 (right_fore_pitch) | 4.2 % (right_fore_pitch) | 4.2 % (right_fore_pitch) |
| 1 | 0/8 | 0.243 (0.241) | +0.1 % | -0.007 | 0.293 (0.461) | 0.13 (left_hind) | 0.87 (right_fore_pitch) | 4.1 % (right_fore_pitch) | 4.1 % (right_fore_pitch) |
| 1.2 | 0/8 | 0.244 (0.241) | +0.4 % | -0.006 | 0.309 (0.368) | 0.13 (left_hind) | 0.89 (right_fore_pitch) | 4.3 % (right_fore_pitch) | 4.3 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `run3-s0_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.244 (0.242) | +0.6 % | -0.006 | 0.288 (0.445) | 0.14 (left_hind) | 0.78 (right_fore_pitch) | 2.7 % (right_fore_pitch) | 2.7 % (right_fore_pitch) |
| x0.9 | 0/8 | 0.244 (0.241) | +0.4 % | -0.006 | 0.287 (0.396) | 0.13 (left_hind) | 0.82 (right_fore_pitch) | 3.2 % (right_fore_pitch) | 3.2 % (right_fore_pitch) |
| x1 | 0/8 | 0.243 (0.241) | +0.0 % | -0.007 | 0.232 (0.343) | 0.13 (left_hind) | 0.86 (right_fore_pitch) | 4.2 % (right_fore_pitch) | 4.2 % (right_fore_pitch) |
| x1.1 | 0/8 | 0.241 (0.239) | -0.6 % | -0.009 | 0.107 (0.180) | 0.14 (left_hind) | 0.94 (left_hind_flexion) | 5.0 % (right_fore_pitch) | 5.0 % (right_fore_pitch) |
| x1.2 | 0/8 | 0.241 (0.240) | -0.8 % | -0.009 | 0.081 (0.118) | 0.15 (left_hind) | 1.03 (left_hind_flexion) | 5.3 % (right_fore_pitch) | 5.3 % (right_fore_pitch) |
| x1.4 | 0/8 | 0.241 (0.241) | -0.6 % | -0.009 | 0.137 (0.198) | 0.17 (left_hind) | 1.22 (left_hind_flexion) | 7.0 % (right_fore_pitch) | 7.0 % (right_fore_pitch) |
| x1.6 | 0/8 | 0.241 (0.240) | -0.8 % | -0.009 | 0.225 (0.293) | 0.18 (left_hind) | 1.35 (left_hind_flexion) | 8.8 % (right_fore_pitch) | 8.8 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `run3-s0_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.243 (0.241) | +0.0 % | -0.007 | 0.232 (0.343) | 0.13 (left_hind) | 0.86 (right_fore_pitch) | 4.2 % (right_fore_pitch) | 4.2 % (right_fore_pitch) |
| +0.5 kg | 0/8 | 0.241 (0.239) | -0.7 % | -0.009 | 0.164 (0.284) | 0.14 (left_hind) | 0.88 (left_hind_flexion) | 4.4 % (right_fore_pitch) | 4.4 % (right_fore_pitch) |
| +1 kg | 0/8 | 0.240 (0.239) | -1.2 % | -0.010 | 0.090 (0.162) | 0.15 (left_hind) | 0.92 (left_hind_flexion) | 4.5 % (right_fore_pitch) | 4.5 % (right_fore_pitch) |
| +2 kg | 0/8 | 0.239 (0.238) | -1.4 % | -0.011 | 0.118 (0.180) | 0.17 (left_hind) | 1.01 (left_hind_flexion) | 4.7 % (right_fore_pitch) | 4.7 % (right_fore_pitch) |
| +3 kg | 0/8 | 0.238 (0.237) | -1.9 % | -0.012 | 0.120 (0.170) | 0.18 (left_hind) | 1.08 (left_hind_flexion) | 5.0 % (right_fore_pitch) | 5.0 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `run3-s0_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.235 (0.234) | -3.2 % | -0.015 | 0.104 (0.214) | 0.21 (right_hind) | 0.76 (left_hind_flexion) | 0.0 % (left_fore_roll) | 40.4 % (left_hind_flexion) |
| x0.55 | 0/8 | 0.239 (0.239) | -1.4 % | -0.011 | 0.057 (0.149) | 0.17 (left_hind) | 0.84 (left_hind_flexion) | 0.0 % (left_fore_roll) | 14.8 % (right_fore_pitch) |
| x0.7 | 0/8 | 0.241 (0.240) | -0.9 % | -0.009 | 0.052 (0.106) | 0.16 (left_hind) | 0.86 (left_hind_flexion) | 0.0 % (left_fore_roll) | 8.1 % (right_fore_pitch) |
| x0.85 | 0/8 | 0.241 (0.240) | -0.7 % | -0.009 | 0.166 (0.292) | 0.14 (left_hind) | 0.84 (left_hind_flexion) | 0.0 % (left_fore_roll) | 5.9 % (right_fore_pitch) |
| x1 | 0/8 | 0.243 (0.241) | +0.0 % | -0.007 | 0.232 (0.343) | 0.13 (left_hind) | 0.86 (right_fore_pitch) | 4.2 % (right_fore_pitch) | 4.2 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `run3-s0_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.232 (0.230) | -4.6 % | -0.018 | 0.405 (0.530) | 0.09 (left_hind) | 0.80 (right_hind_pitch) | 2.1 % (right_hind_pitch) | 2.1 % (right_hind_pitch) |
| x0.6 | 0/8 | 0.238 (0.235) | -2.1 % | -0.012 | 0.315 (0.459) | 0.11 (left_hind) | 0.78 (right_hind_pitch) | 1.6 % (right_hind_pitch) | 1.6 % (right_hind_pitch) |
| x0.8 | 0/8 | 0.242 (0.241) | -0.5 % | -0.008 | 0.263 (0.353) | 0.12 (left_hind) | 0.81 (right_fore_pitch) | 3.0 % (right_fore_pitch) | 3.0 % (right_fore_pitch) |
| x1 | 0/8 | 0.243 (0.241) | +0.0 % | -0.007 | 0.232 (0.343) | 0.13 (left_hind) | 0.86 (right_fore_pitch) | 4.2 % (right_fore_pitch) | 4.2 % (right_fore_pitch) |
| x1.2 | 0/8 | 0.243 (0.241) | +0.1 % | -0.007 | 0.227 (0.334) | 0.15 (left_hind) | 0.94 (right_fore_pitch) | 5.0 % (right_fore_pitch) | 5.0 % (right_fore_pitch) |
| x1.5 | 0/8 | 0.243 (0.242) | +0.2 % | -0.007 | 0.322 (0.414) | 0.18 (left_hind) | 1.16 (right_fore_pitch) | 7.3 % (right_fore_pitch) | 7.3 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `run3-s0_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 45-67 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.0 | 190 | 1.00 | 8/8 | all fell | | | | |
| -1.25 m/s | 15.9 | 159 | 0.79 | 8/8 | all fell | | | | |
| -1 m/s | 12.7 | 127 | 0.59 | 5/8 | 0.244 | 0.168 (0.221) | 0.89 (1.93) | 0 | 35.9 |
| -0.75 m/s | 9.5 | 95 | 0.41 | 0/8 | 0.244 | 0.072 (0.088) | 0.70 (1.33) | 0 | 12.6 |
| -0.5 m/s | 6.3 | 63 | 0.25 | 0/8 | 0.244 | 0.051 (0.069) | 0.38 (1.36) | 0 | 9.9 |
| -0.4 m/s | 5.1 | 51 | 0.19 | 0/8 | 0.244 | 0.046 (0.060) | 0.37 (1.25) | 0 | 8.9 |
| -0.3 m/s | 3.8 | 38 | 0.14 | 0/8 | 0.243 | 0.042 (0.052) | 0.36 (1.24) | 0 | 8.3 |
| -0.2 m/s | 2.5 | 25 | 0.09 | 0/8 | 0.243 | 0.037 (0.055) | 0.35 (0.48) | 0 | 7.5 |
| -0.1 m/s | 1.3 | 13 | 0.05 | 0/8 | 0.243 | 0.032 (0.044) | 0.28 (1.27) | 0 | 6.7 |
| none | 0.0 | 0 |  | 0/8 | 0.243 | (no push) | 0.10 (0.13) | 0 | 6.2 |
| +0.1 m/s | 1.3 | 13 | 0.05 | 0/8 | 0.243 | 0.016 (0.030) | 0.10 (0.77) | 0 | 5.7 |
| +0.2 m/s | 2.5 | 25 | 0.10 | 0/8 | 0.243 | 0.026 (0.040) | 0.10 (0.81) | 0 | 6.6 |
| +0.3 m/s | 3.8 | 38 | 0.14 | 0/8 | 0.243 | 0.028 (0.047) | 0.10 (0.78) | 0 | 5.6 |
| +0.4 m/s | 5.1 | 51 | 0.18 | 0/8 | 0.242 | 0.031 (0.056) | 0.10 (0.79) | 0 | 6.2 |
| +0.5 m/s | 6.3 | 63 | 0.23 | 0/8 | 0.243 | 0.038 (0.062) | 0.29 (0.82) | 0 | 6.7 |
| +0.75 m/s | 9.5 | 95 | 0.38 | 0/8 | 0.242 | 0.064 (0.080) | 0.85 (1.62) | 0 | 11.9 |
| +1 m/s | 12.7 | 127 | 0.55 | 4/8 | 0.240 | 0.178 (0.252) | 1.06 (1.09) | 0 | 36.1 |
| +1.25 m/s | 15.9 | 159 | 0.74 | 8/8 | all fell | | | | |
| +1.5 m/s | 19.0 | 190 | 0.95 | 8/8 | all fell | | | | |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 0.10 s median and 0.13 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `run3-s0_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.243 (0.241) | +0.0 % | -0.007 | 0.232 (0.343) | 0.13 (left_hind) | 0.86 (right_fore_pitch) | 4.2 % (right_fore_pitch) | 4.2 % (right_fore_pitch) |
| step 1 cm | 0/8 | 0.243 (0.241) | -0.1 % | -0.007 | 0.194 (0.333) | 0.13 (left_hind) | 0.86 (right_fore_pitch) | 4.1 % (right_fore_pitch) | 4.1 % (right_fore_pitch) |
| step 2 cm | 0/8 | 0.243 (0.242) | +0.1 % | -0.007 | 0.169 (0.301) | 0.13 (left_hind) | 0.88 (right_fore_pitch) | 4.2 % (right_fore_pitch) | 4.2 % (right_fore_pitch) |
| step 3 cm | 0/8 | 0.242 (0.240) | -0.1 % | -0.008 | 0.173 (0.270) | 0.13 (left_hind) | 0.88 (right_fore_pitch) | 4.3 % (right_fore_pitch) | 4.3 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `run3-s0_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.243 (0.241) | +0.0 % | -0.007 | 0.232 (0.343) | 0.13 (left_hind) | 0.86 (right_fore_pitch) | 4.2 % (right_fore_pitch) | 4.2 % (right_fore_pitch) |
| ramp 3 deg | 0/8 | 0.242 (0.241) | -0.3 % | -0.008 | 0.147 (0.265) | 0.15 (left_hind) | 0.88 (right_fore_pitch) | 4.3 % (right_fore_pitch) | 4.3 % (right_fore_pitch) |
| ramp 5 deg | 0/8 | 0.243 (0.242) | +0.0 % | -0.007 | 0.144 (0.298) | 0.17 (left_hind) | 0.90 (right_fore_pitch) | 4.3 % (right_fore_pitch) | 4.3 % (right_fore_pitch) |
| ramp 8 deg | 0/8 | 0.243 (0.241) | -0.1 % | -0.007 | 0.075 (0.160) | 0.20 (left_hind) | 0.92 (right_hind_pitch) | 3.8 % (right_fore_pitch) | 3.8 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `run3-s0_terrain.png`.

