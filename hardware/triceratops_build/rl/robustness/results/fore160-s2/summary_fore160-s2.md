# Robustness sweep: trike60-fore160-s2 (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-25. One policy, so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Policy `trike60_10002432_steps.zip` (sha256 1f53d26213d8a90d...), run commit f84a6919; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c397e0c...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 13.5 min on 2 worker processes (565 episodes).
- Reproduction check (gait_pipe1's target case, randomisation on, seeds 100000-100004): mean speed 0.2441 m/s against the analysis's 0.2385 m/s; per-episode speeds identical: False.

## Headline

- **No fall and no speed drop of more than 20 %** in: delay, 0.15 m/s (0 ms to 60 ms); friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5); step up (flat to step 3 cm); ramp up (flat to ramp 8 deg).
- **delay, 0.25 m/s**: first break at 60 ms (speed 0.084 m/s, 66 % below nominal).
- **Sideways push**: falls start at 1 m/s free-body (impulse 12.8 N s; the centre of mass gained 0.37 / 0.55 m/s), pushed to the left (1/8 fell) and right (2/8 fell). Largest push with no fall: 0.75 m/s free-body (impulse 9.6 N s; the centre of mass gained 0.23 / 0.40 m/s), pushed to the left and right. The feet absorb part of every push: the centre of mass gains 14-58 % of the free-body velocity change (pushes of 0.3 m/s and more).

## Where it breaks

First point outward from nominal with a fall or a speed more than 20 % below the nominal point's (same command). "none": no such point in the tested range.

| Sweep | Nominal | Low side | High side |
|---|---|---|---|
| delay, 0.25 m/s | 10 ms | none | **60 ms**: speed 0.084 m/s, 66 % below nominal |
| delay, 0.15 m/s | 10 ms | none | none |
| friction | 0.85 (model) | none | none |
| mass, every body | x1 | none | none |
| payload on the chassis | +0 kg | (not tested) | none |
| force limit | x1 | none | (not tested) |
| position gain kp | x1 | none | none |
| sideways push | none | **1 m/s free-body to the right (impulse 12.8 N s; the centre of mass gained 0.55 m/s)**: 2/8 fell | **1 m/s free-body to the left (impulse 12.8 N s; the centre of mass gained 0.37 m/s)**: 1/8 fell |
| step up | flat | (not tested) | none |
| ramp up | flat | (not tested) | none |

Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.257 (0.256) | +4.8 % | +0.007 | 0.088 (0.167) | 0.19 (left_hind) | 1.14 (left_hind_flexion) | 2.2 % (left_fore_pitch) | 2.2 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.254 (0.253) | +3.4 % | +0.004 | 0.069 (0.146) | 0.19 (left_hind) | 1.14 (left_hind_flexion) | 2.2 % (left_fore_pitch) | 2.2 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.249 (0.248) | +1.6 % | -0.001 | 0.084 (0.144) | 0.19 (left_hind) | 1.13 (left_hind_flexion) | 2.1 % (left_fore_pitch) | 2.1 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.245 (0.244) | +0.0 % | -0.005 | 0.093 (0.162) | 0.20 (left_hind) | 1.13 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| 15 ms | 0/8 | 0.246 (0.245) | +0.2 % | -0.004 | 0.073 (0.175) | 0.22 (left_hind) | 1.15 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| 20 ms | 0/8 | 0.258 (0.257) | +5.2 % | +0.008 | 0.144 (0.299) | 0.22 (left_fore) | 1.13 (left_hind_pitch) | 8.8 % (left_hind_pitch) | 8.8 % (left_hind_pitch) |
| 30 ms | 0/8 | 0.249 (0.247) | +1.5 % | -0.001 | 0.145 (0.379) | 0.25 (left_fore) | 1.13 (left_hind_pitch) | 9.0 % (left_hind_pitch) | 9.0 % (left_hind_pitch) |
| 40 ms | 0/8 | 0.214 (0.063) | -12.8 % | -0.036 | 0.368 (0.521) | 0.44 (left_fore) | 1.32 (right_fore_pitch) | 13.1 % (right_fore_pitch) | 13.1 % (right_fore_pitch) |
| 60 ms | 0/8 | 0.084 (0.013) | -65.9 % | -0.166 | 0.387 (0.723) | 1.78 (left_fore) | 1.58 (right_fore_pitch) | 14.7 % (right_fore_pitch) | 14.7 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore160-s2_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.152 (0.151) | +2.7 % | +0.002 | 0.044 (0.089) | 0.31 (left_hind) | 1.09 (left_hind_flexion) | 0.8 % (left_fore_pitch) | 0.8 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.151 (0.150) | +1.8 % | +0.001 | 0.050 (0.102) | 0.32 (left_hind) | 1.09 (left_hind_flexion) | 0.8 % (left_fore_pitch) | 0.8 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.149 (0.149) | +0.8 % | -0.001 | 0.036 (0.095) | 0.32 (left_hind) | 1.09 (left_hind_flexion) | 0.8 % (left_fore_pitch) | 0.8 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.148 (0.147) | +0.0 % | -0.002 | 0.049 (0.115) | 0.34 (left_hind) | 1.11 (left_hind_flexion) | 0.7 % (left_fore_pitch) | 0.7 % (left_fore_pitch) |
| 15 ms | 0/8 | 0.149 (0.148) | +0.4 % | -0.001 | 0.042 (0.113) | 0.31 (left_hind) | 1.14 (left_hind_flexion) | 0.8 % (left_fore_pitch) | 0.8 % (left_fore_pitch) |
| 20 ms | 0/8 | 0.151 (0.148) | +1.8 % | +0.001 | 0.046 (0.090) | 0.31 (left_fore) | 1.01 (left_hind_flexion) | 3.9 % (left_hind_pitch) | 3.9 % (left_hind_pitch) |
| 30 ms | 0/8 | 0.147 (0.145) | -1.2 % | -0.003 | 0.054 (0.081) | 0.36 (left_fore) | 1.02 (left_hind_flexion) | 3.9 % (left_hind_pitch) | 3.9 % (left_hind_pitch) |
| 40 ms | 0/8 | 0.184 (0.180) | +24.1 % | +0.034 | 0.894 (1.021) | 0.47 (right_fore) | 1.46 (left_hind_pitch) | 18.3 % (left_hind_pitch) | 18.3 % (left_hind_pitch) |
| 60 ms | 0/8 | 0.195 (0.180) | +31.3 % | +0.045 | 0.931 (1.118) | 0.44 (right_fore) | 1.39 (left_hind_pitch) | 16.5 % (left_hind_pitch) | 16.5 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore160-s2_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.240 (0.239) | -2.1 % | -0.010 | 0.189 (0.279) | 0.67 (left_hind) | 0.82 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| 0.2 | 0/8 | 0.240 (0.238) | -2.3 % | -0.010 | 0.197 (0.294) | 0.60 (left_hind) | 0.83 (left_hind_flexion) | 1.8 % (left_fore_pitch) | 1.8 % (left_fore_pitch) |
| 0.3 | 0/8 | 0.241 (0.240) | -1.9 % | -0.009 | 0.199 (0.320) | 0.52 (left_hind) | 0.86 (left_hind_flexion) | 1.7 % (left_fore_pitch) | 1.7 % (left_fore_pitch) |
| 0.4 | 0/8 | 0.242 (0.242) | -1.1 % | -0.008 | 0.162 (0.278) | 0.42 (left_hind) | 0.93 (left_hind_flexion) | 1.9 % (left_fore_pitch) | 1.9 % (left_fore_pitch) |
| 0.5 | 0/8 | 0.242 (0.241) | -1.2 % | -0.008 | 0.123 (0.198) | 0.32 (left_hind) | 1.02 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| 0.6 | 0/8 | 0.244 (0.243) | -0.4 % | -0.006 | 0.094 (0.170) | 0.29 (left_hind) | 1.08 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| 0.8 | 0/8 | 0.245 (0.244) | -0.1 % | -0.005 | 0.089 (0.151) | 0.21 (left_hind) | 1.13 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| 0.85 (model) | 0/8 | 0.245 (0.244) | +0.0 % | -0.005 | 0.093 (0.162) | 0.20 (left_hind) | 1.13 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| 1 | 0/8 | 0.245 (0.244) | +0.0 % | -0.005 | 0.114 (0.174) | 0.19 (left_hind) | 1.13 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| 1.2 | 0/8 | 0.245 (0.244) | +0.0 % | -0.005 | 0.138 (0.226) | 0.18 (left_hind) | 1.13 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `fore160-s2_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.244 (0.243) | -0.6 % | -0.006 | 0.074 (0.154) | 0.19 (left_hind) | 0.96 (left_hind_flexion) | 1.9 % (left_fore_pitch) | 1.9 % (left_fore_pitch) |
| x0.9 | 0/8 | 0.245 (0.244) | -0.3 % | -0.005 | 0.073 (0.171) | 0.21 (left_hind) | 1.05 (left_hind_flexion) | 1.9 % (left_fore_pitch) | 1.9 % (left_fore_pitch) |
| x1 | 0/8 | 0.245 (0.244) | +0.0 % | -0.005 | 0.093 (0.162) | 0.20 (left_hind) | 1.13 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| x1.1 | 0/8 | 0.245 (0.244) | -0.0 % | -0.005 | 0.135 (0.279) | 0.20 (left_hind) | 1.20 (left_hind_flexion) | 2.2 % (left_fore_pitch) | 2.2 % (left_fore_pitch) |
| x1.2 | 0/8 | 0.244 (0.244) | -0.3 % | -0.006 | 0.198 (0.285) | 0.19 (left_hind) | 1.26 (left_hind_flexion) | 2.1 % (left_fore_pitch) | 2.1 % (left_fore_pitch) |
| x1.4 | 0/8 | 0.243 (0.243) | -0.9 % | -0.007 | 0.273 (0.366) | 0.18 (left_hind) | 1.37 (left_hind_flexion) | 2.2 % (left_fore_pitch) | 2.2 % (left_fore_pitch) |
| x1.6 | 0/8 | 0.243 (0.242) | -1.0 % | -0.007 | 0.266 (0.381) | 0.18 (left_fore) | 1.53 (left_hind_flexion) | 3.1 % (left_hind_flexion) | 3.1 % (left_hind_flexion) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `fore160-s2_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.245 (0.244) | +0.0 % | -0.005 | 0.093 (0.162) | 0.20 (left_hind) | 1.13 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| +0.5 kg | 0/8 | 0.245 (0.243) | -0.3 % | -0.005 | 0.110 (0.218) | 0.21 (left_hind) | 1.16 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| +1 kg | 0/8 | 0.244 (0.243) | -0.4 % | -0.006 | 0.121 (0.204) | 0.21 (left_hind) | 1.19 (left_hind_flexion) | 2.1 % (left_fore_pitch) | 2.1 % (left_fore_pitch) |
| +2 kg | 0/8 | 0.243 (0.242) | -0.9 % | -0.007 | 0.167 (0.295) | 0.22 (left_hind) | 1.25 (left_hind_flexion) | 2.1 % (left_fore_pitch) | 2.1 % (left_fore_pitch) |
| +3 kg | 0/8 | 0.242 (0.241) | -1.4 % | -0.008 | 0.192 (0.307) | 0.21 (left_hind) | 1.31 (left_hind_flexion) | 2.1 % (left_fore_pitch) | 2.1 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `fore160-s2_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.251 (0.250) | +2.5 % | +0.001 | 0.041 (0.126) | 0.27 (left_hind) | 0.91 (left_hind_flexion) | 0.0 % (left_fore_roll) | 62.4 % (left_hind_flexion) |
| x0.55 | 0/8 | 0.245 (0.244) | -0.1 % | -0.005 | 0.103 (0.183) | 0.17 (left_hind) | 1.08 (left_hind_flexion) | 0.0 % (left_fore_roll) | 38.4 % (left_hind_flexion) |
| x0.7 | 0/8 | 0.245 (0.244) | -0.2 % | -0.005 | 0.103 (0.168) | 0.19 (left_hind) | 1.11 (left_hind_flexion) | 0.0 % (left_fore_roll) | 6.1 % (left_fore_pitch) |
| x0.85 | 0/8 | 0.245 (0.244) | -0.1 % | -0.005 | 0.112 (0.192) | 0.20 (left_hind) | 1.13 (left_hind_flexion) | 0.0 % (left_fore_roll) | 3.3 % (left_fore_pitch) |
| x1 | 0/8 | 0.245 (0.244) | +0.0 % | -0.005 | 0.093 (0.162) | 0.20 (left_hind) | 1.13 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `fore160-s2_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.231 (0.230) | -5.8 % | -0.019 | 0.078 (0.175) | 0.18 (left_fore) | 0.94 (left_hind_flexion) | 1.0 % (left_hind_flexion) | 1.0 % (left_hind_flexion) |
| x0.6 | 0/8 | 0.240 (0.238) | -2.2 % | -0.010 | 0.070 (0.179) | 0.18 (left_hind) | 1.10 (left_hind_flexion) | 0.2 % (left_fore_pitch) | 0.2 % (left_fore_pitch) |
| x0.8 | 0/8 | 0.243 (0.242) | -0.8 % | -0.007 | 0.061 (0.117) | 0.20 (left_hind) | 1.12 (left_hind_flexion) | 1.0 % (left_fore_pitch) | 1.0 % (left_fore_pitch) |
| x1 | 0/8 | 0.245 (0.244) | +0.0 % | -0.005 | 0.093 (0.162) | 0.20 (left_hind) | 1.13 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| x1.2 | 0/8 | 0.246 (0.245) | +0.5 % | -0.004 | 0.112 (0.188) | 0.20 (left_hind) | 1.13 (left_hind_flexion) | 3.3 % (left_fore_pitch) | 3.3 % (left_fore_pitch) |
| x1.5 | 0/8 | 0.247 (0.246) | +0.8 % | -0.003 | 0.115 (0.270) | 0.20 (left_hind) | 1.14 (left_hind_flexion) | 5.8 % (left_fore_pitch) | 5.8 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `fore160-s2_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 14-58 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.2 | 192 | 0.87 | 8/8 | all fell | | | | |
| -1.25 m/s | 16.0 | 160 | 0.70 | 7/8 | 0.245 | 0.197 (0.219) | 0.68 (0.68) | 0 | 37.1 |
| -1 m/s | 12.8 | 128 | 0.55 | 2/8 | 0.245 | 0.138 (0.226) | 1.04 (1.36) | 0 | 35.4 |
| -0.75 m/s | 9.6 | 96 | 0.40 | 0/8 | 0.245 | 0.066 (0.090) | 0.73 (1.03) | 0 | 13.4 |
| -0.5 m/s | 6.4 | 64 | 0.25 | 0/8 | 0.245 | 0.037 (0.052) | 0.20 (0.79) | 0 | 8.1 |
| -0.4 m/s | 5.1 | 51 | 0.19 | 0/8 | 0.245 | 0.027 (0.042) | 0.10 (0.50) | 0 | 7.7 |
| -0.3 m/s | 3.8 | 38 | 0.13 | 0/8 | 0.245 | 0.020 (0.031) | 0.10 (0.13) | 0 | 7.2 |
| -0.2 m/s | 2.6 | 26 | 0.08 | 0/8 | 0.245 | 0.013 (0.018) | 0.10 (0.10) | 0 | 7.0 |
| -0.1 m/s | 1.3 | 13 | 0.04 | 0/8 | 0.245 | 0.006 (0.010) | 0.10 (0.10) | 0 | 6.6 |
| none | 0.0 | 0 |  | 0/8 | 0.245 | (no push) | 0.10 (0.10) | 0 | 6.2 |
| +0.1 m/s | 1.3 | 13 | 0.01 | 0/8 | 0.245 | 0.004 (0.006) | 0.10 (0.10) | 0 | 6.2 |
| +0.2 m/s | 2.6 | 26 | 0.02 | 0/8 | 0.245 | 0.008 (0.015) | 0.10 (0.10) | 0 | 6.7 |
| +0.3 m/s | 3.8 | 38 | 0.04 | 0/8 | 0.245 | 0.009 (0.016) | 0.10 (0.10) | 0 | 7.5 |
| +0.4 m/s | 5.1 | 51 | 0.07 | 0/8 | 0.245 | 0.012 (0.024) | 0.10 (0.10) | 0 | 8.8 |
| +0.5 m/s | 6.4 | 64 | 0.10 | 0/8 | 0.245 | 0.015 (0.032) | 0.10 (0.39) | 0 | 9.8 |
| +0.75 m/s | 9.6 | 96 | 0.23 | 0/8 | 0.245 | 0.031 (0.070) | 0.52 (0.80) | 0 | 14.5 |
| +1 m/s | 12.8 | 128 | 0.37 | 1/8 | 0.245 | 0.073 (0.189) | 1.12 (1.58) | 0 | 35.0 |
| +1.25 m/s | 16.0 | 160 | 0.54 | 3/8 | 0.241 | 0.142 (0.192) | 1.47 (1.55) | 0 | 36.8 |
| +1.5 m/s | 19.2 | 192 | 0.72 | 8/8 | all fell | | | | |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 0.10 s median and 0.10 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `fore160-s2_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.245 (0.244) | +0.0 % | -0.005 | 0.093 (0.162) | 0.20 (left_hind) | 1.13 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| step 1 cm | 0/8 | 0.245 (0.244) | -0.1 % | -0.005 | 0.091 (0.164) | 0.20 (left_hind) | 1.13 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| step 2 cm | 0/8 | 0.245 (0.244) | -0.1 % | -0.005 | 0.092 (0.150) | 0.20 (left_hind) | 1.14 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| step 3 cm | 0/8 | 0.245 (0.244) | -0.3 % | -0.005 | 0.096 (0.174) | 0.20 (left_hind) | 1.14 (left_hind_flexion) | 2.1 % (left_fore_pitch) | 2.1 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `fore160-s2_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.245 (0.244) | +0.0 % | -0.005 | 0.093 (0.162) | 0.20 (left_hind) | 1.13 (left_hind_flexion) | 2.0 % (left_fore_pitch) | 2.0 % (left_fore_pitch) |
| ramp 3 deg | 0/8 | 0.242 (0.242) | -1.2 % | -0.008 | 0.133 (0.238) | 0.24 (left_hind) | 1.15 (left_hind_flexion) | 2.2 % (left_fore_pitch) | 2.2 % (left_fore_pitch) |
| ramp 5 deg | 0/8 | 0.243 (0.242) | -0.7 % | -0.007 | 0.177 (0.299) | 0.20 (left_hind) | 1.14 (left_hind_flexion) | 2.5 % (left_fore_pitch) | 2.5 % (left_fore_pitch) |
| ramp 8 deg | 0/8 | 0.248 (0.248) | +1.1 % | -0.002 | 0.273 (0.399) | 0.18 (left_fore) | 1.10 (left_hind_flexion) | 3.8 % (left_fore_pitch) | 3.8 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `fore160-s2_terrain.png`.

