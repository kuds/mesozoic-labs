# Robustness sweep: trike60-run2b-ankle (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-19. One policy, so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Policy `trike60_8003584_steps.zip` (sha256 00ed354692da0a39...), run commit e92fd9ef; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 a82a0c4d206fdfb7...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 12.1 min on 2 worker processes (560 episodes).

## Headline

- **No fall and no speed drop of more than 20 %** in: friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5).
- **delay, 0.25 m/s**: first break at 60 ms (speed 0.171 m/s, 31 % below nominal).
- **delay, 0.15 m/s**: first break at 60 ms (speed 0.104 m/s, 29 % below nominal).
- **step up**: first break at step 3 cm (speed 0.114 m/s, 54 % below nominal).
- **ramp up**: first break at ramp 8 deg (speed 0.162 m/s, 34 % below nominal).
- **Sideways push**: falls start at 1.25 m/s free-body (impulse 15.9 N s; the centre of mass gained 0.73 / 0.65 m/s), pushed to the left (7/8 fell) and right (4/8 fell). Largest push with no fall: 1 m/s free-body (impulse 12.7 N s; the centre of mass gained 0.56 / 0.48 m/s), pushed to the left and right. The feet absorb part of every push: the centre of mass gains 43-61 % of the free-body velocity change (pushes of 0.3 m/s and more).

## Where it breaks

First point outward from nominal with a fall or a speed more than 20 % below the nominal point's (same command). "none": no such point in the tested range.

| Sweep | Nominal | Low side | High side |
|---|---|---|---|
| delay, 0.25 m/s | 10 ms | none | **60 ms**: speed 0.171 m/s, 31 % below nominal |
| delay, 0.15 m/s | 10 ms | none | **60 ms**: speed 0.104 m/s, 29 % below nominal |
| friction | 0.85 (model) | none | none |
| mass, every body | x1 | none | none |
| payload on the chassis | +0 kg | (not tested) | none |
| force limit | x1 | none | (not tested) |
| position gain kp | x1 | none | none |
| sideways push | none | **1.25 m/s free-body to the right (impulse 15.9 N s; the centre of mass gained 0.65 m/s)**: 4/8 fell | **1.25 m/s free-body to the left (impulse 15.9 N s; the centre of mass gained 0.73 m/s)**: 7/8 fell |
| step up | flat | (not tested) | **step 3 cm**: speed 0.114 m/s, 54 % below nominal |
| ramp up | flat | (not tested) | **ramp 8 deg**: speed 0.162 m/s, 34 % below nominal |

Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.257 (0.256) | +4.4 % | +0.007 | 0.092 (0.236) | 0.08 (left_hind) | 0.70 (left_hind_flexion) | 1.0 % (right_hind_flexion) | 1.0 % (right_hind_flexion) |
| 2 ms | 0/8 | 0.253 (0.253) | +2.8 % | +0.003 | 0.086 (0.181) | 0.08 (left_hind) | 0.71 (left_hind_flexion) | 0.9 % (right_hind_flexion) | 0.9 % (right_hind_flexion) |
| 5 ms | 0/8 | 0.249 (0.249) | +1.1 % | -0.001 | 0.079 (0.115) | 0.08 (left_hind) | 0.71 (left_hind_flexion) | 0.8 % (right_hind_flexion) | 0.8 % (right_hind_flexion) |
| 10 ms | 0/8 | 0.246 (0.245) | +0.0 % | -0.004 | 0.078 (0.161) | 0.09 (left_hind) | 0.72 (left_hind_flexion) | 0.9 % (right_fore_pitch) | 0.9 % (right_fore_pitch) |
| 15 ms | 0/8 | 0.246 (0.245) | -0.3 % | -0.004 | 0.266 (0.327) | 0.09 (left_hind) | 0.78 (right_fore_pitch) | 1.4 % (right_fore_pitch) | 1.4 % (right_fore_pitch) |
| 20 ms | 0/8 | 0.234 (0.232) | -4.8 % | -0.016 | 0.297 (0.382) | 0.11 (right_hind) | 0.87 (right_hind_pitch) | 3.8 % (right_hind_pitch) | 3.8 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.229 (0.227) | -7.1 % | -0.021 | 0.323 (0.395) | 0.12 (right_hind) | 0.86 (right_hind_pitch) | 3.6 % (right_hind_pitch) | 3.6 % (right_hind_pitch) |
| 40 ms | 0/8 | 0.211 (0.206) | -14.3 % | -0.039 | 0.216 (0.361) | 0.22 (right_hind) | 1.12 (left_hind_pitch) | 8.4 % (right_hind_pitch) | 8.4 % (right_hind_pitch) |
| 60 ms | 0/8 | 0.171 (0.159) | -30.7 % | -0.079 | 0.084 (0.191) | 0.35 (left_hind) | 1.19 (left_fore_pitch) | 10.2 % (right_hind_pitch) | 10.2 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `run2b_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.154 (0.154) | +4.9 % | +0.004 | 0.110 (0.150) | 0.09 (left_hind) | 0.64 (left_hind_flexion) | 0.7 % (right_hind_flexion) | 0.7 % (right_hind_flexion) |
| 2 ms | 0/8 | 0.152 (0.151) | +3.2 % | +0.002 | 0.104 (0.168) | 0.09 (left_hind) | 0.64 (left_hind_flexion) | 0.7 % (right_hind_flexion) | 0.7 % (right_hind_flexion) |
| 5 ms | 0/8 | 0.149 (0.148) | +1.1 % | -0.001 | 0.118 (0.158) | 0.09 (left_hind) | 0.64 (left_hind_flexion) | 0.7 % (right_hind_flexion) | 0.7 % (right_hind_flexion) |
| 10 ms | 0/8 | 0.147 (0.146) | +0.0 % | -0.003 | 0.090 (0.136) | 0.10 (left_hind) | 0.64 (left_hind_flexion) | 0.7 % (right_hind_flexion) | 0.7 % (right_hind_flexion) |
| 15 ms | 0/8 | 0.147 (0.146) | +0.2 % | -0.003 | 0.056 (0.099) | 0.11 (left_hind) | 0.66 (left_hind_flexion) | 0.7 % (right_hind_flexion) | 0.7 % (right_hind_flexion) |
| 20 ms | 0/8 | 0.142 (0.141) | -3.2 % | -0.008 | 0.179 (0.210) | 0.15 (right_hind) | 0.78 (left_hind_pitch) | 1.8 % (left_hind_pitch) | 1.8 % (left_hind_pitch) |
| 30 ms | 0/8 | 0.140 (0.139) | -4.8 % | -0.010 | 0.216 (0.267) | 0.17 (right_hind) | 0.80 (left_hind_pitch) | 2.0 % (left_hind_pitch) | 2.0 % (left_hind_pitch) |
| 40 ms | 0/8 | 0.131 (0.127) | -10.8 % | -0.019 | 0.221 (0.331) | 0.35 (left_hind) | 1.25 (left_hind_pitch) | 8.4 % (left_hind_pitch) | 8.4 % (left_hind_pitch) |
| 60 ms | 0/8 | 0.104 (0.095) | -29.4 % | -0.046 | 0.118 (0.237) | 0.69 (left_hind) | 1.30 (left_hind_pitch) | 11.7 % (left_hind_pitch) | 11.7 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `run2b_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.223 (0.220) | -9.3 % | -0.027 | 0.781 (0.991) | 0.30 (right_fore) | 0.76 (left_hind_flexion) | 2.2 % (right_hind_flexion) | 2.2 % (right_hind_flexion) |
| 0.2 | 0/8 | 0.229 (0.226) | -7.0 % | -0.021 | 0.564 (0.745) | 0.22 (right_fore) | 0.74 (left_hind_flexion) | 1.7 % (right_hind_flexion) | 1.7 % (right_hind_flexion) |
| 0.3 | 0/8 | 0.235 (0.234) | -4.7 % | -0.015 | 0.319 (0.384) | 0.12 (right_fore) | 0.74 (left_hind_flexion) | 1.2 % (right_hind_flexion) | 1.2 % (right_hind_flexion) |
| 0.4 | 0/8 | 0.238 (0.236) | -3.6 % | -0.012 | 0.206 (0.293) | 0.09 (left_hind) | 0.73 (left_hind_flexion) | 1.0 % (right_hind_flexion) | 1.0 % (right_hind_flexion) |
| 0.5 | 0/8 | 0.240 (0.239) | -2.7 % | -0.010 | 0.129 (0.221) | 0.09 (left_hind) | 0.73 (left_hind_flexion) | 0.9 % (right_hind_flexion) | 0.9 % (right_hind_flexion) |
| 0.6 | 0/8 | 0.242 (0.241) | -1.8 % | -0.008 | 0.064 (0.129) | 0.09 (left_hind) | 0.71 (left_hind_flexion) | 0.9 % (right_hind_flexion) | 0.9 % (right_hind_flexion) |
| 0.8 | 0/8 | 0.245 (0.244) | -0.4 % | -0.005 | 0.060 (0.142) | 0.08 (left_hind) | 0.72 (left_hind_flexion) | 0.9 % (right_fore_pitch) | 0.9 % (right_fore_pitch) |
| 0.85 (model) | 0/8 | 0.246 (0.245) | +0.0 % | -0.004 | 0.078 (0.161) | 0.09 (left_hind) | 0.72 (left_hind_flexion) | 0.9 % (right_fore_pitch) | 0.9 % (right_fore_pitch) |
| 1 | 0/8 | 0.247 (0.246) | +0.4 % | -0.003 | 0.118 (0.189) | 0.09 (left_hind) | 0.74 (left_hind_flexion) | 1.1 % (right_fore_pitch) | 1.1 % (right_fore_pitch) |
| 1.2 | 0/8 | 0.247 (0.245) | +0.2 % | -0.003 | 0.059 (0.141) | 0.08 (left_hind) | 0.74 (left_hind_flexion) | 1.3 % (right_fore_pitch) | 1.3 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `run2b_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.247 (0.247) | +0.3 % | -0.003 | 0.346 (0.476) | 0.08 (right_hind) | 0.69 (right_fore_pitch) | 1.0 % (right_hind_flexion) | 1.0 % (right_hind_flexion) |
| x0.9 | 0/8 | 0.247 (0.246) | +0.3 % | -0.003 | 0.145 (0.245) | 0.08 (left_hind) | 0.67 (right_fore_pitch) | 0.9 % (right_fore_pitch) | 0.9 % (right_fore_pitch) |
| x1 | 0/8 | 0.246 (0.245) | +0.0 % | -0.004 | 0.078 (0.161) | 0.09 (left_hind) | 0.72 (left_hind_flexion) | 0.9 % (right_fore_pitch) | 0.9 % (right_fore_pitch) |
| x1.1 | 0/8 | 0.245 (0.244) | -0.4 % | -0.005 | 0.060 (0.106) | 0.09 (left_hind) | 0.77 (left_hind_flexion) | 1.0 % (right_fore_pitch) | 1.0 % (right_fore_pitch) |
| x1.2 | 0/8 | 0.245 (0.244) | -0.6 % | -0.005 | 0.067 (0.135) | 0.09 (left_hind) | 0.84 (left_hind_flexion) | 1.1 % (right_hind_pitch) | 1.1 % (right_hind_pitch) |
| x1.4 | 0/8 | 0.243 (0.242) | -1.4 % | -0.007 | 0.142 (0.212) | 0.10 (left_hind) | 0.94 (left_hind_flexion) | 2.5 % (right_hind_pitch) | 2.5 % (right_hind_pitch) |
| x1.6 | 0/8 | 0.241 (0.240) | -2.4 % | -0.009 | 0.180 (0.230) | 0.11 (left_hind) | 1.06 (left_hind_flexion) | 4.5 % (right_hind_pitch) | 4.5 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `run2b_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.246 (0.245) | +0.0 % | -0.004 | 0.078 (0.161) | 0.09 (left_hind) | 0.72 (left_hind_flexion) | 0.9 % (right_fore_pitch) | 0.9 % (right_fore_pitch) |
| +0.5 kg | 0/8 | 0.245 (0.245) | -0.4 % | -0.005 | 0.062 (0.127) | 0.09 (left_hind) | 0.76 (left_hind_flexion) | 1.0 % (right_fore_pitch) | 1.0 % (right_fore_pitch) |
| +1 kg | 0/8 | 0.244 (0.243) | -0.9 % | -0.006 | 0.052 (0.139) | 0.09 (left_hind) | 0.79 (left_hind_flexion) | 1.0 % (right_fore_pitch) | 1.0 % (right_fore_pitch) |
| +2 kg | 0/8 | 0.242 (0.241) | -1.8 % | -0.008 | 0.047 (0.116) | 0.10 (left_hind) | 0.84 (left_hind_flexion) | 1.0 % (right_fore_pitch) | 1.0 % (right_fore_pitch) |
| +3 kg | 0/8 | 0.239 (0.239) | -2.9 % | -0.011 | 0.071 (0.143) | 0.10 (left_hind) | 0.91 (left_hind_flexion) | 1.5 % (right_hind_pitch) | 1.5 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `run2b_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.242 (0.241) | -1.8 % | -0.008 | 0.049 (0.079) | 0.08 (left_hind) | 0.66 (left_hind_flexion) | 0.0 % (left_fore_roll) | 13.5 % (left_hind_flexion) |
| x0.55 | 0/8 | 0.245 (0.244) | -0.4 % | -0.005 | 0.066 (0.138) | 0.08 (left_hind) | 0.69 (left_hind_flexion) | 0.0 % (left_fore_roll) | 5.5 % (right_hind_pitch) |
| x0.7 | 0/8 | 0.246 (0.245) | -0.2 % | -0.004 | 0.075 (0.173) | 0.08 (left_hind) | 0.70 (left_hind_flexion) | 0.0 % (left_fore_roll) | 2.6 % (right_fore_pitch) |
| x0.85 | 0/8 | 0.246 (0.245) | -0.0 % | -0.004 | 0.090 (0.217) | 0.08 (left_hind) | 0.72 (left_hind_flexion) | 0.0 % (left_fore_roll) | 1.5 % (right_fore_pitch) |
| x1 | 0/8 | 0.246 (0.245) | +0.0 % | -0.004 | 0.078 (0.161) | 0.09 (left_hind) | 0.72 (left_hind_flexion) | 0.9 % (right_fore_pitch) | 0.9 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `run2b_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.205 (0.204) | -16.8 % | -0.045 | 0.216 (0.272) | 0.07 (left_hind) | 0.60 (left_hind_flexion) | 0.1 % (right_hind_flexion) | 0.1 % (right_hind_flexion) |
| x0.6 | 0/8 | 0.228 (0.227) | -7.6 % | -0.022 | 0.312 (0.383) | 0.08 (left_hind) | 0.62 (left_hind_flexion) | 0.5 % (right_hind_flexion) | 0.5 % (right_hind_flexion) |
| x0.8 | 0/8 | 0.240 (0.239) | -2.8 % | -0.010 | 0.210 (0.282) | 0.08 (left_hind) | 0.67 (left_hind_flexion) | 0.7 % (right_hind_flexion) | 0.7 % (right_hind_flexion) |
| x1 | 0/8 | 0.246 (0.245) | +0.0 % | -0.004 | 0.078 (0.161) | 0.09 (left_hind) | 0.72 (left_hind_flexion) | 0.9 % (right_fore_pitch) | 0.9 % (right_fore_pitch) |
| x1.2 | 0/8 | 0.251 (0.250) | +1.8 % | +0.001 | 0.074 (0.148) | 0.09 (left_hind) | 0.77 (right_fore_pitch) | 1.6 % (right_fore_pitch) | 1.6 % (right_fore_pitch) |
| x1.5 | 0/8 | 0.253 (0.250) | +2.7 % | +0.003 | 0.128 (0.264) | 0.16 (left_hind) | 1.08 (right_fore_pitch) | 6.1 % (right_hind_pitch) | 6.1 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `run2b_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 43-61 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.0 | 190 | 0.84 | 8/8 | all fell | | | | |
| -1.25 m/s | 15.9 | 159 | 0.65 | 4/8 | 0.232 | 0.198 (0.230) | 2.25 (2.79) | 0 | 35.2 |
| -1 m/s | 12.7 | 127 | 0.48 | 0/8 | 0.243 | 0.090 (0.111) | 1.05 (1.67) | 0 | 17.4 |
| -0.75 m/s | 9.5 | 95 | 0.32 | 0/8 | 0.246 | 0.049 (0.054) | 0.88 (1.00) | 0 | 8.8 |
| -0.5 m/s | 6.3 | 63 | 0.21 | 0/8 | 0.246 | 0.033 (0.040) | 0.80 (0.92) | 0 | 5.2 |
| -0.4 m/s | 5.1 | 51 | 0.18 | 0/8 | 0.246 | 0.030 (0.039) | 0.82 (0.86) | 0 | 4.7 |
| -0.3 m/s | 3.8 | 38 | 0.15 | 0/8 | 0.246 | 0.025 (0.038) | 0.78 (1.49) | 0 | 4.5 |
| -0.2 m/s | 2.5 | 25 | 0.10 | 0/8 | 0.246 | 0.020 (0.029) | 0.57 (0.77) | 0 | 3.6 |
| -0.1 m/s | 1.3 | 13 | 0.05 | 0/8 | 0.246 | 0.017 (0.026) | 0.38 (1.37) | 0 | 3.9 |
| none | 0.0 | 0 |  | 0/8 | 0.246 | (no push) | 0.10 (0.10) | 0 | 3.6 |
| +0.1 m/s | 1.3 | 13 | 0.06 | 0/8 | 0.246 | 0.018 (0.026) | 0.10 (0.32) | 0 | 4.1 |
| +0.2 m/s | 2.5 | 25 | 0.11 | 0/8 | 0.246 | 0.025 (0.032) | 0.10 (0.52) | 0 | 5.0 |
| +0.3 m/s | 3.8 | 38 | 0.16 | 0/8 | 0.246 | 0.029 (0.034) | 0.10 (0.50) | 0 | 5.5 |
| +0.4 m/s | 5.1 | 51 | 0.22 | 0/8 | 0.246 | 0.034 (0.042) | 0.26 (0.52) | 0 | 5.8 |
| +0.5 m/s | 6.3 | 63 | 0.28 | 0/8 | 0.246 | 0.041 (0.058) | 0.32 (1.36) | 0 | 6.7 |
| +0.75 m/s | 9.5 | 95 | 0.40 | 0/8 | 0.242 | 0.060 (0.068) | 1.47 (2.81) | 0 | 9.8 |
| +1 m/s | 12.7 | 127 | 0.56 | 0/8 | 0.240 | 0.109 (0.129) | 0.93 (1.93) | 0 | 19.3 |
| +1.25 m/s | 15.9 | 159 | 0.73 | 7/8 | 0.227 | 0.222 (0.236) | 1.92 (1.92) | 0 | 35.8 |
| +1.5 m/s | 19.0 | 190 | 0.92 | 8/8 | all fell | | | | |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 0.10 s median and 0.10 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `run2b_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.246 (0.245) | +0.0 % | -0.004 | 0.078 (0.161) | 0.09 (left_hind) | 0.72 (left_hind_flexion) | 0.9 % (right_fore_pitch) | 0.9 % (right_fore_pitch) |
| step 1 cm | 0/8 | 0.245 (0.244) | -0.4 % | -0.005 | 0.133 (0.234) | 0.08 (left_hind) | 0.73 (left_hind_flexion) | 1.2 % (right_fore_pitch) | 1.2 % (right_fore_pitch) |
| step 2 cm | 0/8 | 0.202 (0.193) | -18.1 % | -0.048 | 1.128 (1.259) | 0.21 (right_fore) | 1.17 (right_fore_pitch) | 6.0 % (right_fore_pitch) | 6.0 % (right_fore_pitch) |
| step 3 cm | 0/8 | 0.114 (0.060) | -53.9 % | -0.136 | 1.152 (1.331) | 0.89 (right_fore) | 1.40 (right_fore_pitch) | 13.0 % (right_fore_pitch) | 13.0 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `run2b_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.246 (0.245) | +0.0 % | -0.004 | 0.078 (0.161) | 0.09 (left_hind) | 0.72 (left_hind_flexion) | 0.9 % (right_fore_pitch) | 0.9 % (right_fore_pitch) |
| ramp 3 deg | 0/8 | 0.234 (0.233) | -5.0 % | -0.016 | 0.224 (0.362) | 0.09 (left_hind) | 0.74 (left_hind_flexion) | 2.3 % (right_hind_pitch) | 2.3 % (right_hind_pitch) |
| ramp 5 deg | 0/8 | 0.218 (0.214) | -11.7 % | -0.032 | 0.660 (0.719) | 0.12 (right_hind) | 0.72 (right_hind_pitch) | 2.8 % (right_hind_pitch) | 2.8 % (right_hind_pitch) |
| ramp 8 deg | 0/8 | 0.162 (0.158) | -34.3 % | -0.088 | 0.868 (0.943) | 0.27 (right_hind) | 0.78 (right_hind_pitch) | 2.5 % (right_hind_pitch) | 2.5 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `run2b_terrain.png`.

