# Robustness sweep: trike60-fore160-s3 (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-26. One policy, so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Policy `trike60_10002432_steps.zip` (sha256 e0dbaf08d3dd9449...), run commit f84a6919; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c397e0c...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 14.1 min on 2 worker processes (565 episodes).
- Reproduction check (gait_pipe1's target case, randomisation on, seeds 100000-100004): mean speed 0.2385 m/s against the analysis's 0.2385 m/s; per-episode speeds identical: False.

## Headline

- **No fall and no speed drop of more than 20 %** in: delay, 0.15 m/s (0 ms to 60 ms); friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5); step up (flat to step 3 cm); ramp up (flat to ramp 8 deg).
- **delay, 0.25 m/s**: first break at 60 ms (4/8 fell).
- **Sideways push**: falls start at 0.5 m/s free-body (impulse 6.4 N s; the centre of mass gained 0.32 m/s), pushed to the right (1/8 fell); 1 m/s free-body (impulse 12.8 N s; the centre of mass gained 0.69 m/s), pushed to the left (1/8 fell). Largest push with no fall: 0.4 m/s free-body (impulse 5.1 N s; the centre of mass gained 0.25 m/s), pushed to the right; 0.75 m/s free-body (impulse 9.6 N s; the centre of mass gained 0.51 m/s), pushed to the left. The feet absorb part of every push: the centre of mass gains 61-72 % of the free-body velocity change (pushes of 0.3 m/s and more).

## Where it breaks

First point outward from nominal with a fall or a speed more than 20 % below the nominal point's (same command). "none": no such point in the tested range.

| Sweep | Nominal | Low side | High side |
|---|---|---|---|
| delay, 0.25 m/s | 10 ms | none | **60 ms**: 4/8 fell |
| delay, 0.15 m/s | 10 ms | none | none |
| friction | 0.85 (model) | none | none |
| mass, every body | x1 | none | none |
| payload on the chassis | +0 kg | (not tested) | none |
| force limit | x1 | none | (not tested) |
| position gain kp | x1 | none | none |
| sideways push | none | **0.5 m/s free-body to the right (impulse 6.4 N s; the centre of mass gained 0.32 m/s)**: 1/8 fell | **1 m/s free-body to the left (impulse 12.8 N s; the centre of mass gained 0.69 m/s)**: 1/8 fell |
| step up | flat | (not tested) | none |
| ramp up | flat | (not tested) | none |

Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.251 (0.250) | +5.0 % | +0.001 | 0.149 (0.271) | 0.43 (right_hind) | 0.97 (right_fore_pitch) | 2.8 % (right_fore_pitch) | 2.8 % (right_fore_pitch) |
| 2 ms | 0/8 | 0.247 (0.246) | +3.3 % | -0.003 | 0.174 (0.280) | 0.41 (right_hind) | 0.95 (right_fore_pitch) | 2.7 % (right_fore_pitch) | 2.7 % (right_fore_pitch) |
| 5 ms | 0/8 | 0.242 (0.242) | +1.4 % | -0.008 | 0.195 (0.330) | 0.38 (right_hind) | 0.93 (right_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |
| 10 ms | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.242 (0.366) | 0.30 (right_hind) | 0.91 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| 15 ms | 0/8 | 0.239 (0.238) | -0.0 % | -0.011 | 0.362 (0.451) | 0.23 (right_hind) | 0.93 (right_fore_pitch) | 2.7 % (right_fore_pitch) | 2.7 % (right_fore_pitch) |
| 20 ms | 0/8 | 0.242 (0.240) | +1.2 % | -0.008 | 0.249 (0.423) | 0.26 (right_hind) | 1.10 (left_hind_pitch) | 5.2 % (left_hind_pitch) | 5.2 % (left_hind_pitch) |
| 30 ms | 0/8 | 0.231 (0.230) | -3.3 % | -0.019 | 0.097 (0.184) | 0.21 (left_hind) | 1.16 (left_hind_pitch) | 7.0 % (left_hind_pitch) | 7.0 % (left_hind_pitch) |
| 40 ms | 0/8 | 0.243 (0.239) | +1.7 % | -0.007 | 0.132 (0.368) | 0.24 (left_hind) | 1.32 (left_hind_pitch) | 14.8 % (left_hind_pitch) | 14.8 % (left_hind_pitch) |
| 60 ms | 4/8 | 0.224 (0.208) | -6.1 % | -0.026 | 0.153 (0.398) | 0.29 (right_fore) | 1.33 (left_hind_pitch) | 16.1 % (left_hind_pitch) | 16.1 % (left_hind_pitch) |

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore160-s3_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.153 (0.152) | +6.2 % | +0.003 | 0.043 (0.072) | 0.57 (right_hind) | 0.81 (left_hind_pitch) | 1.4 % (right_fore_pitch) | 1.4 % (right_fore_pitch) |
| 2 ms | 0/8 | 0.151 (0.150) | +4.2 % | +0.001 | 0.043 (0.083) | 0.55 (right_hind) | 0.80 (left_hind_pitch) | 1.3 % (right_fore_pitch) | 1.3 % (right_fore_pitch) |
| 5 ms | 0/8 | 0.147 (0.146) | +1.8 % | -0.003 | 0.038 (0.082) | 0.53 (right_hind) | 0.78 (left_hind_pitch) | 1.2 % (right_fore_pitch) | 1.2 % (right_fore_pitch) |
| 10 ms | 0/8 | 0.144 (0.144) | +0.0 % | -0.006 | 0.074 (0.117) | 0.48 (right_hind) | 0.73 (left_hind_pitch) | 1.1 % (right_fore_pitch) | 1.1 % (right_fore_pitch) |
| 15 ms | 0/8 | 0.145 (0.144) | +0.4 % | -0.005 | 0.127 (0.159) | 0.43 (right_hind) | 0.74 (right_fore_pitch) | 1.3 % (right_fore_pitch) | 1.3 % (right_fore_pitch) |
| 20 ms | 0/8 | 0.148 (0.147) | +2.4 % | -0.002 | 0.090 (0.137) | 0.31 (right_hind) | 1.14 (left_hind_pitch) | 3.2 % (left_hind_pitch) | 3.2 % (left_hind_pitch) |
| 30 ms | 0/8 | 0.143 (0.140) | -1.3 % | -0.007 | 0.053 (0.113) | 0.35 (left_hind) | 1.21 (left_hind_pitch) | 5.6 % (left_hind_pitch) | 5.6 % (left_hind_pitch) |
| 40 ms | 0/8 | 0.146 (0.139) | +1.0 % | -0.004 | 0.149 (0.306) | 0.41 (right_fore) | 1.52 (left_hind_pitch) | 21.8 % (left_hind_pitch) | 21.8 % (left_hind_pitch) |
| 60 ms | 0/8 | 0.142 (0.137) | -2.0 % | -0.008 | 0.195 (0.302) | 0.44 (left_hind) | 1.52 (left_hind_pitch) | 22.7 % (left_hind_pitch) | 22.7 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore160-s3_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.236 (0.235) | -1.2 % | -0.014 | 0.183 (0.317) | 0.54 (right_hind) | 0.79 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| 0.2 | 0/8 | 0.237 (0.236) | -1.0 % | -0.013 | 0.064 (0.134) | 0.55 (right_hind) | 0.79 (right_fore_pitch) | 2.3 % (right_fore_pitch) | 2.3 % (right_fore_pitch) |
| 0.3 | 0/8 | 0.237 (0.237) | -0.7 % | -0.013 | 0.268 (0.349) | 0.52 (right_hind) | 0.82 (right_fore_pitch) | 2.4 % (right_fore_pitch) | 2.4 % (right_fore_pitch) |
| 0.4 | 0/8 | 0.238 (0.237) | -0.5 % | -0.012 | 0.421 (0.538) | 0.48 (right_hind) | 0.85 (right_fore_pitch) | 2.4 % (right_fore_pitch) | 2.4 % (right_fore_pitch) |
| 0.5 | 0/8 | 0.238 (0.238) | -0.4 % | -0.012 | 0.422 (0.508) | 0.46 (right_hind) | 0.88 (right_fore_pitch) | 2.4 % (right_fore_pitch) | 2.4 % (right_fore_pitch) |
| 0.6 | 0/8 | 0.239 (0.238) | -0.2 % | -0.011 | 0.350 (0.455) | 0.44 (right_hind) | 0.88 (right_fore_pitch) | 2.4 % (right_fore_pitch) | 2.4 % (right_fore_pitch) |
| 0.8 | 0/8 | 0.239 (0.238) | -0.1 % | -0.011 | 0.236 (0.318) | 0.32 (right_hind) | 0.91 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| 0.85 (model) | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.242 (0.366) | 0.30 (right_hind) | 0.91 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| 1 | 0/8 | 0.239 (0.238) | -0.1 % | -0.011 | 0.284 (0.380) | 0.27 (right_hind) | 0.92 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| 1.2 | 0/8 | 0.239 (0.238) | -0.1 % | -0.011 | 0.310 (0.401) | 0.26 (right_hind) | 0.92 (right_fore_pitch) | 2.6 % (right_fore_pitch) | 2.6 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `fore160-s3_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.240 (0.239) | +0.4 % | -0.010 | 0.258 (0.311) | 0.29 (right_hind) | 0.83 (right_fore_pitch) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |
| x0.9 | 0/8 | 0.240 (0.239) | +0.2 % | -0.010 | 0.261 (0.373) | 0.30 (right_hind) | 0.87 (right_fore_pitch) | 2.2 % (right_fore_pitch) | 2.2 % (right_fore_pitch) |
| x1 | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.242 (0.366) | 0.30 (right_hind) | 0.91 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| x1.1 | 0/8 | 0.239 (0.237) | -0.2 % | -0.011 | 0.256 (0.349) | 0.29 (right_hind) | 0.94 (right_fore_pitch) | 2.7 % (right_fore_pitch) | 2.7 % (right_fore_pitch) |
| x1.2 | 0/8 | 0.238 (0.237) | -0.5 % | -0.012 | 0.291 (0.378) | 0.27 (right_hind) | 0.97 (right_fore_pitch) | 2.9 % (right_fore_pitch) | 2.9 % (right_fore_pitch) |
| x1.4 | 0/8 | 0.236 (0.236) | -1.2 % | -0.014 | 0.330 (0.420) | 0.27 (right_hind) | 1.09 (left_hind_pitch) | 3.8 % (left_hind_pitch) | 3.8 % (left_hind_pitch) |
| x1.6 | 0/8 | 0.235 (0.234) | -1.7 % | -0.015 | 0.373 (0.464) | 0.27 (right_hind) | 1.27 (left_hind_pitch) | 6.1 % (left_hind_pitch) | 6.1 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `fore160-s3_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.242 (0.366) | 0.30 (right_hind) | 0.91 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| +0.5 kg | 0/8 | 0.238 (0.238) | -0.3 % | -0.012 | 0.254 (0.373) | 0.30 (right_hind) | 0.91 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| +1 kg | 0/8 | 0.238 (0.237) | -0.6 % | -0.012 | 0.249 (0.332) | 0.31 (right_hind) | 0.92 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| +2 kg | 0/8 | 0.236 (0.235) | -1.2 % | -0.014 | 0.230 (0.313) | 0.31 (right_hind) | 0.93 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| +3 kg | 0/8 | 0.235 (0.233) | -1.8 % | -0.015 | 0.209 (0.302) | 0.31 (right_hind) | 0.97 (left_hind_pitch) | 2.7 % (left_hind_pitch) | 2.7 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `fore160-s3_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.235 (0.234) | -1.9 % | -0.015 | 0.425 (0.513) | 0.31 (right_hind) | 0.73 (right_fore_pitch) | 0.0 % (left_fore_roll) | 24.8 % (right_fore_pitch) |
| x0.55 | 0/8 | 0.237 (0.236) | -0.9 % | -0.013 | 0.361 (0.459) | 0.31 (right_hind) | 0.81 (right_fore_pitch) | 0.0 % (left_fore_roll) | 10.3 % (right_fore_pitch) |
| x0.7 | 0/8 | 0.238 (0.238) | -0.3 % | -0.012 | 0.301 (0.394) | 0.30 (right_hind) | 0.85 (right_fore_pitch) | 0.0 % (left_fore_roll) | 5.6 % (right_fore_pitch) |
| x0.85 | 0/8 | 0.239 (0.238) | -0.1 % | -0.011 | 0.291 (0.397) | 0.30 (right_hind) | 0.88 (right_fore_pitch) | 0.0 % (left_fore_roll) | 3.5 % (right_fore_pitch) |
| x1 | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.242 (0.366) | 0.30 (right_hind) | 0.91 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `fore160-s3_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.224 (0.222) | -6.5 % | -0.026 | 0.367 (0.452) | 0.21 (right_hind) | 0.71 (right_hind_flexion) | 0.7 % (left_hind_pitch) | 0.7 % (left_hind_pitch) |
| x0.6 | 0/8 | 0.230 (0.229) | -3.6 % | -0.020 | 0.094 (0.216) | 0.29 (right_hind) | 0.72 (right_hind_flexion) | 0.9 % (right_fore_pitch) | 0.9 % (right_fore_pitch) |
| x0.8 | 0/8 | 0.236 (0.235) | -1.3 % | -0.014 | 0.117 (0.186) | 0.30 (right_hind) | 0.80 (right_fore_pitch) | 1.7 % (right_fore_pitch) | 1.7 % (right_fore_pitch) |
| x1 | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.242 (0.366) | 0.30 (right_hind) | 0.91 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| x1.2 | 0/8 | 0.242 (0.241) | +1.0 % | -0.008 | 0.392 (0.496) | 0.29 (right_hind) | 1.01 (right_fore_pitch) | 3.3 % (right_fore_pitch) | 3.3 % (right_fore_pitch) |
| x1.5 | 0/8 | 0.244 (0.243) | +2.1 % | -0.006 | 0.495 (0.618) | 0.31 (right_hind) | 1.15 (right_fore_pitch) | 4.6 % (right_fore_pitch) | 4.6 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `fore160-s3_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 61-72 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.2 | 192 | 1.09 | 8/8 | all fell | | | | |
| -1.25 m/s | 16.0 | 160 | 0.87 | 8/8 | all fell | | | | |
| -1 m/s | 12.8 | 128 | 0.67 | 8/8 | all fell | | | | |
| -0.75 m/s | 9.6 | 96 | 0.48 | 7/8 | 0.237 | 0.186 (0.227) | 1.60 (1.60) | 0 | 36.7 |
| -0.5 m/s | 6.4 | 64 | 0.32 | 1/8 | 0.240 | 0.081 (0.167) | 1.54 (1.69) | 1 | 35.8 |
| -0.4 m/s | 5.1 | 51 | 0.25 | 0/8 | 0.239 | 0.051 (0.067) | 1.56 (1.65) | 1 | 16.9 |
| -0.3 m/s | 3.8 | 38 | 0.18 | 0/8 | 0.239 | 0.036 (0.050) | 1.37 (3.84) | 1 | 14.1 |
| -0.2 m/s | 2.6 | 26 | 0.12 | 0/8 | 0.239 | 0.025 (0.039) | 1.60 (1.74) | 2 | 13.0 |
| -0.1 m/s | 1.3 | 13 | 0.07 | 0/8 | 0.239 | 0.019 (0.032) | 0.53 (2.87) | 1 | 12.1 |
| none | 0.0 | 0 |  | 0/8 | 0.239 | (no push) | 0.67 (1.85) | 1 | 11.7 |
| +0.1 m/s | 1.3 | 13 | 0.07 | 0/8 | 0.239 | 0.022 (0.041) | 0.45 (0.76) | 1 | 11.8 |
| +0.2 m/s | 2.6 | 26 | 0.14 | 0/8 | 0.239 | 0.040 (0.060) | 0.47 (1.39) | 0 | 11.9 |
| +0.3 m/s | 3.8 | 38 | 0.20 | 0/8 | 0.239 | 0.051 (0.068) | 0.44 (1.70) | 0 | 12.0 |
| +0.4 m/s | 5.1 | 51 | 0.27 | 0/8 | 0.239 | 0.059 (0.079) | 0.42 (0.58) | 0 | 12.0 |
| +0.5 m/s | 6.4 | 64 | 0.33 | 0/8 | 0.239 | 0.062 (0.087) | 0.42 (2.82) | 0 | 11.8 |
| +0.75 m/s | 9.6 | 96 | 0.51 | 0/8 | 0.239 | 0.085 (0.104) | 0.40 (1.63) | 1 | 12.2 |
| +1 m/s | 12.8 | 128 | 0.69 | 1/8 | 0.238 | 0.124 (0.234) | 0.41 (3.94) | 0 | 35.5 |
| +1.25 m/s | 16.0 | 160 | 0.88 | 2/8 | 0.233 | 0.179 (0.262) | 1.10 (5.41) | 0 | 35.3 |
| +1.5 m/s | 19.2 | 192 | 1.08 | 7/8 | 0.231 | 0.271 (0.324) | 3.56 (3.56) | 0 | 36.7 |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 0.67 s median and 1.85 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `fore160-s3_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.242 (0.366) | 0.30 (right_hind) | 0.91 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| step 1 cm | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.253 (0.358) | 0.30 (right_hind) | 0.91 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| step 2 cm | 0/8 | 0.239 (0.238) | +0.1 % | -0.011 | 0.233 (0.324) | 0.30 (right_hind) | 0.90 (right_fore_pitch) | 2.4 % (right_fore_pitch) | 2.4 % (right_fore_pitch) |
| step 3 cm | 0/8 | 0.239 (0.238) | -0.1 % | -0.011 | 0.222 (0.348) | 0.29 (right_hind) | 0.91 (right_fore_pitch) | 2.4 % (right_fore_pitch) | 2.4 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `fore160-s3_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.242 (0.366) | 0.30 (right_hind) | 0.91 (right_fore_pitch) | 2.5 % (right_fore_pitch) | 2.5 % (right_fore_pitch) |
| ramp 3 deg | 0/8 | 0.239 (0.239) | +0.1 % | -0.011 | 0.133 (0.217) | 0.28 (right_hind) | 0.89 (right_fore_pitch) | 2.3 % (right_fore_pitch) | 2.3 % (right_fore_pitch) |
| ramp 5 deg | 0/8 | 0.240 (0.238) | +0.2 % | -0.010 | 0.069 (0.155) | 0.27 (right_hind) | 0.91 (left_hind_pitch) | 2.1 % (right_fore_pitch) | 2.1 % (right_fore_pitch) |
| ramp 8 deg | 0/8 | 0.239 (0.238) | +0.0 % | -0.011 | 0.279 (0.545) | 0.24 (right_hind) | 1.01 (left_hind_pitch) | 3.6 % (right_hind_pitch) | 3.6 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `fore160-s3_terrain.png`.

