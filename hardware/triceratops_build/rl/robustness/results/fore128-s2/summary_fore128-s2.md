# Robustness sweep: trike60-fore128-s2 (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-25. One policy, so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Policy `trike60_10002432_steps.zip` (sha256 e41f7bd92dddcd28...), run commit f84a6919; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c397e0c...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 15.1 min on 2 worker processes (565 episodes).
- Reproduction check (gait_pipe1's target case, randomisation on, seeds 100000-100004): mean speed 0.2419 m/s against the analysis's 0.2385 m/s; per-episode speeds identical: False.

## Headline

- **No fall and no speed drop of more than 20 %** in: delay, 0.25 m/s (0 ms to 60 ms); delay, 0.15 m/s (0 ms to 60 ms); friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5); step up (flat to step 3 cm); ramp up (flat to ramp 8 deg).
- **Sideways push**: falls start at 1.25 m/s free-body (impulse 15.9 N s; the centre of mass gained 0.67 / 0.69 m/s), pushed to the left (8/8 fell) and right (1/8 fell). Largest push with no fall: 1 m/s free-body (impulse 12.8 N s; the centre of mass gained 0.48 / 0.58 m/s), pushed to the left and right. The feet absorb part of every push: the centre of mass gains 15-61 % of the free-body velocity change (pushes of 0.3 m/s and more).

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
| sideways push | none | **1.25 m/s free-body to the right (impulse 15.9 N s; the centre of mass gained 0.69 m/s)**: 1/8 fell | **1.25 m/s free-body to the left (impulse 15.9 N s; the centre of mass gained 0.67 m/s)**: 8/8 fell |
| step up | flat | (not tested) | none |
| ramp up | flat | (not tested) | none |

Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.264 (0.263) | +8.7 % | +0.014 | 0.571 (0.663) | 0.39 (left_hind) | 0.98 (left_fore_pitch) | 3.6 % (left_fore_pitch) | 3.6 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.259 (0.257) | +6.7 % | +0.009 | 0.552 (0.674) | 0.37 (left_hind) | 0.97 (left_fore_pitch) | 3.5 % (left_fore_pitch) | 3.5 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.251 (0.250) | +3.6 % | +0.001 | 0.480 (0.591) | 0.37 (left_hind) | 0.98 (left_fore_pitch) | 3.4 % (left_fore_pitch) | 3.4 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.242 (0.242) | +0.0 % | -0.008 | 0.354 (0.420) | 0.35 (left_hind) | 0.97 (left_fore_pitch) | 3.1 % (left_fore_pitch) | 3.1 % (left_fore_pitch) |
| 15 ms | 0/8 | 0.239 (0.238) | -1.2 % | -0.011 | 0.292 (0.381) | 0.32 (left_hind) | 0.98 (left_fore_pitch) | 3.0 % (left_fore_pitch) | 3.0 % (left_fore_pitch) |
| 20 ms | 0/8 | 0.255 (0.254) | +5.3 % | +0.005 | 0.419 (0.531) | 0.26 (left_hind) | 1.20 (right_hind_pitch) | 7.6 % (right_hind_pitch) | 7.6 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.249 (0.244) | +2.6 % | -0.001 | 0.503 (0.620) | 0.29 (left_fore) | 1.22 (right_hind_pitch) | 10.0 % (left_hind_pitch) | 10.0 % (left_hind_pitch) |
| 40 ms | 0/8 | 0.260 (0.232) | +7.3 % | +0.010 | 0.481 (1.163) | 0.28 (left_fore) | 1.42 (right_hind_pitch) | 17.2 % (right_hind_pitch) | 17.2 % (right_hind_pitch) |
| 60 ms | 0/8 | 0.221 (0.171) | -8.9 % | -0.029 | 0.405 (0.640) | 0.36 (left_fore) | 1.37 (right_hind_pitch) | 15.4 % (right_hind_pitch) | 15.4 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore128-s2_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.156 (0.156) | +8.0 % | +0.006 | 0.315 (0.369) | 0.66 (left_hind) | 0.74 (left_fore_pitch) | 1.1 % (left_fore_pitch) | 1.1 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.153 (0.152) | +5.8 % | +0.003 | 0.310 (0.372) | 0.65 (left_hind) | 0.74 (left_fore_pitch) | 1.1 % (left_fore_pitch) | 1.1 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.149 (0.148) | +3.0 % | -0.001 | 0.305 (0.350) | 0.63 (left_hind) | 0.73 (left_fore_pitch) | 1.0 % (left_fore_pitch) | 1.0 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.145 (0.144) | +0.0 % | -0.005 | 0.304 (0.349) | 0.55 (left_hind) | 0.72 (left_fore_pitch) | 0.9 % (left_fore_pitch) | 0.9 % (left_fore_pitch) |
| 15 ms | 0/8 | 0.144 (0.143) | -0.4 % | -0.006 | 0.290 (0.337) | 0.43 (left_hind) | 0.73 (left_fore_pitch) | 0.9 % (left_fore_pitch) | 0.9 % (left_fore_pitch) |
| 20 ms | 0/8 | 0.153 (0.152) | +5.8 % | +0.003 | 0.237 (0.295) | 0.46 (left_hind) | 1.25 (right_hind_pitch) | 6.9 % (right_hind_pitch) | 6.9 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.149 (0.146) | +3.1 % | -0.001 | 0.211 (0.255) | 0.43 (left_fore) | 1.33 (right_hind_pitch) | 10.5 % (right_hind_pitch) | 10.5 % (right_hind_pitch) |
| 40 ms | 0/8 | 0.181 (0.167) | +25.4 % | +0.031 | 0.155 (0.444) | 0.53 (right_hind) | 1.60 (right_hind_pitch) | 23.9 % (right_hind_pitch) | 23.9 % (right_hind_pitch) |
| 60 ms | 0/8 | 0.177 (0.144) | +22.6 % | +0.027 | 0.384 (0.601) | 0.62 (right_hind) | 1.65 (right_hind_pitch) | 26.0 % (right_hind_pitch) | 26.0 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore128-s2_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.240 (0.239) | -0.9 % | -0.010 | 0.763 (0.862) | 0.73 (left_hind) | 0.88 (left_fore_pitch) | 3.4 % (left_fore_pitch) | 3.4 % (left_fore_pitch) |
| 0.2 | 0/8 | 0.241 (0.239) | -0.8 % | -0.009 | 0.758 (0.893) | 0.67 (left_hind) | 0.89 (left_fore_pitch) | 3.3 % (left_fore_pitch) | 3.3 % (left_fore_pitch) |
| 0.3 | 0/8 | 0.242 (0.241) | -0.1 % | -0.008 | 0.647 (0.745) | 0.56 (left_hind) | 0.90 (left_fore_pitch) | 3.1 % (left_fore_pitch) | 3.1 % (left_fore_pitch) |
| 0.4 | 0/8 | 0.243 (0.242) | +0.2 % | -0.007 | 0.521 (0.576) | 0.51 (left_hind) | 0.93 (left_fore_pitch) | 3.1 % (left_fore_pitch) | 3.1 % (left_fore_pitch) |
| 0.5 | 0/8 | 0.243 (0.242) | +0.4 % | -0.007 | 0.464 (0.591) | 0.46 (left_hind) | 0.94 (left_fore_pitch) | 3.1 % (left_fore_pitch) | 3.1 % (left_fore_pitch) |
| 0.6 | 0/8 | 0.243 (0.242) | +0.3 % | -0.007 | 0.409 (0.484) | 0.43 (left_hind) | 0.94 (left_fore_pitch) | 3.0 % (left_fore_pitch) | 3.0 % (left_fore_pitch) |
| 0.8 | 0/8 | 0.242 (0.241) | -0.0 % | -0.008 | 0.358 (0.424) | 0.37 (left_hind) | 0.97 (left_fore_pitch) | 3.1 % (left_fore_pitch) | 3.1 % (left_fore_pitch) |
| 0.85 (model) | 0/8 | 0.242 (0.242) | +0.0 % | -0.008 | 0.354 (0.420) | 0.35 (left_hind) | 0.97 (left_fore_pitch) | 3.1 % (left_fore_pitch) | 3.1 % (left_fore_pitch) |
| 1 | 0/8 | 0.242 (0.241) | -0.3 % | -0.008 | 0.338 (0.425) | 0.33 (left_hind) | 0.99 (left_fore_pitch) | 3.2 % (left_fore_pitch) | 3.2 % (left_fore_pitch) |
| 1.2 | 0/8 | 0.241 (0.240) | -0.5 % | -0.009 | 0.326 (0.382) | 0.31 (left_hind) | 1.00 (left_fore_pitch) | 3.3 % (left_fore_pitch) | 3.3 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `fore128-s2_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.243 (0.242) | +0.2 % | -0.007 | 0.342 (0.419) | 0.36 (left_hind) | 0.88 (left_fore_pitch) | 2.9 % (left_fore_pitch) | 2.9 % (left_fore_pitch) |
| x0.9 | 0/8 | 0.243 (0.241) | +0.0 % | -0.007 | 0.353 (0.448) | 0.36 (left_hind) | 0.93 (left_fore_pitch) | 3.0 % (left_fore_pitch) | 3.0 % (left_fore_pitch) |
| x1 | 0/8 | 0.242 (0.242) | +0.0 % | -0.008 | 0.354 (0.420) | 0.35 (left_hind) | 0.97 (left_fore_pitch) | 3.1 % (left_fore_pitch) | 3.1 % (left_fore_pitch) |
| x1.1 | 0/8 | 0.242 (0.241) | -0.2 % | -0.008 | 0.339 (0.424) | 0.35 (left_hind) | 1.02 (left_fore_pitch) | 3.3 % (left_fore_pitch) | 3.3 % (left_fore_pitch) |
| x1.2 | 0/8 | 0.241 (0.241) | -0.5 % | -0.009 | 0.329 (0.438) | 0.34 (left_hind) | 1.06 (left_fore_pitch) | 3.5 % (left_fore_pitch) | 3.5 % (left_fore_pitch) |
| x1.4 | 0/8 | 0.240 (0.239) | -1.0 % | -0.010 | 0.292 (0.384) | 0.30 (left_hind) | 1.13 (left_fore_pitch) | 4.3 % (left_fore_pitch) | 4.3 % (left_fore_pitch) |
| x1.6 | 0/8 | 0.238 (0.238) | -1.7 % | -0.012 | 0.218 (0.324) | 0.28 (left_hind) | 1.17 (left_fore_pitch) | 5.2 % (left_fore_pitch) | 5.2 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `fore128-s2_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.242 (0.242) | +0.0 % | -0.008 | 0.354 (0.420) | 0.35 (left_hind) | 0.97 (left_fore_pitch) | 3.1 % (left_fore_pitch) | 3.1 % (left_fore_pitch) |
| +0.5 kg | 0/8 | 0.242 (0.240) | -0.4 % | -0.008 | 0.343 (0.432) | 0.36 (left_hind) | 0.99 (left_fore_pitch) | 3.2 % (left_fore_pitch) | 3.2 % (left_fore_pitch) |
| +1 kg | 0/8 | 0.241 (0.240) | -0.7 % | -0.009 | 0.339 (0.413) | 0.36 (left_hind) | 1.00 (left_fore_pitch) | 3.2 % (left_fore_pitch) | 3.2 % (left_fore_pitch) |
| +2 kg | 0/8 | 0.240 (0.238) | -1.1 % | -0.010 | 0.339 (0.429) | 0.36 (left_hind) | 1.02 (left_fore_pitch) | 3.3 % (right_hind_pitch) | 3.3 % (right_hind_pitch) |
| +3 kg | 0/8 | 0.239 (0.238) | -1.5 % | -0.011 | 0.321 (0.410) | 0.36 (left_hind) | 1.05 (left_hind_flexion) | 3.6 % (right_hind_pitch) | 3.6 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `fore128-s2_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.240 (0.239) | -1.1 % | -0.010 | 0.366 (0.446) | 0.39 (left_hind) | 0.75 (left_fore_pitch) | 0.0 % (left_fore_roll) | 28.8 % (left_hind_flexion) |
| x0.55 | 0/8 | 0.241 (0.240) | -0.6 % | -0.009 | 0.382 (0.455) | 0.36 (left_hind) | 0.83 (left_fore_pitch) | 0.0 % (left_fore_roll) | 13.8 % (left_fore_pitch) |
| x0.7 | 0/8 | 0.241 (0.241) | -0.4 % | -0.009 | 0.372 (0.472) | 0.36 (left_hind) | 0.89 (left_fore_pitch) | 0.0 % (left_fore_roll) | 7.8 % (left_fore_pitch) |
| x0.85 | 0/8 | 0.242 (0.241) | -0.2 % | -0.008 | 0.360 (0.439) | 0.36 (left_hind) | 0.94 (left_fore_pitch) | 0.0 % (left_fore_roll) | 4.8 % (left_fore_pitch) |
| x1 | 0/8 | 0.242 (0.242) | +0.0 % | -0.008 | 0.354 (0.420) | 0.35 (left_hind) | 0.97 (left_fore_pitch) | 3.1 % (left_fore_pitch) | 3.1 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `fore128-s2_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.228 (0.227) | -5.8 % | -0.022 | 0.284 (0.357) | 0.29 (left_hind) | 0.84 (left_hind_flexion) | 0.2 % (right_hind_pitch) | 0.2 % (right_hind_pitch) |
| x0.6 | 0/8 | 0.235 (0.234) | -3.2 % | -0.015 | 0.324 (0.399) | 0.34 (left_hind) | 0.86 (left_hind_flexion) | 0.7 % (left_fore_pitch) | 0.7 % (left_fore_pitch) |
| x0.8 | 0/8 | 0.239 (0.238) | -1.3 % | -0.011 | 0.373 (0.456) | 0.36 (left_hind) | 0.91 (left_fore_pitch) | 1.9 % (right_hind_pitch) | 1.9 % (right_hind_pitch) |
| x1 | 0/8 | 0.242 (0.242) | +0.0 % | -0.008 | 0.354 (0.420) | 0.35 (left_hind) | 0.97 (left_fore_pitch) | 3.1 % (left_fore_pitch) | 3.1 % (left_fore_pitch) |
| x1.2 | 0/8 | 0.245 (0.244) | +0.9 % | -0.005 | 0.316 (0.380) | 0.36 (left_hind) | 1.03 (left_fore_pitch) | 4.5 % (left_fore_pitch) | 4.5 % (left_fore_pitch) |
| x1.5 | 0/8 | 0.248 (0.247) | +2.4 % | -0.002 | 0.235 (0.353) | 0.36 (left_hind) | 1.13 (left_fore_pitch) | 6.4 % (left_fore_pitch) | 6.4 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `fore128-s2_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 15-61 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.1 | 191 | 0.84 | 8/8 | all fell | | | | |
| -1.25 m/s | 15.9 | 159 | 0.69 | 1/8 | 0.242 | 0.151 (0.251) | 0.89 (5.76) | 0 | 35.6 |
| -1 m/s | 12.8 | 128 | 0.58 | 0/8 | 0.243 | 0.087 (0.127) | 0.67 (0.89) | 0 | 16.9 |
| -0.75 m/s | 9.6 | 96 | 0.46 | 0/8 | 0.243 | 0.056 (0.079) | 0.53 (0.64) | 0 | 10.1 |
| -0.5 m/s | 6.4 | 64 | 0.28 | 0/8 | 0.243 | 0.038 (0.054) | 0.14 (0.55) | 0 | 6.6 |
| -0.4 m/s | 5.1 | 51 | 0.22 | 0/8 | 0.243 | 0.033 (0.046) | 0.13 (0.53) | 0 | 5.6 |
| -0.3 m/s | 3.8 | 38 | 0.15 | 0/8 | 0.242 | 0.025 (0.038) | 0.10 (0.76) | 0 | 5.6 |
| -0.2 m/s | 2.6 | 26 | 0.09 | 0/8 | 0.242 | 0.014 (0.026) | 0.10 (0.21) | 0 | 5.1 |
| -0.1 m/s | 1.3 | 13 | 0.03 | 0/8 | 0.242 | 0.008 (0.016) | 0.10 (0.10) | 0 | 4.9 |
| none | 0.0 | 0 |  | 0/8 | 0.242 | (no push) | 0.10 (0.10) | 0 | 5.2 |
| +0.1 m/s | 1.3 | 13 | 0.01 | 0/8 | 0.242 | 0.005 (0.007) | 0.10 (0.70) | 0 | 5.2 |
| +0.2 m/s | 2.6 | 26 | 0.02 | 0/8 | 0.242 | 0.007 (0.015) | 0.10 (0.67) | 0 | 5.0 |
| +0.3 m/s | 3.8 | 38 | 0.05 | 0/8 | 0.242 | 0.009 (0.013) | 0.29 (0.73) | 0 | 5.1 |
| +0.4 m/s | 5.1 | 51 | 0.09 | 0/8 | 0.242 | 0.013 (0.016) | 0.45 (1.45) | 0 | 5.1 |
| +0.5 m/s | 6.4 | 64 | 0.14 | 0/8 | 0.242 | 0.019 (0.022) | 0.55 (1.47) | 0 | 5.3 |
| +0.75 m/s | 9.6 | 96 | 0.30 | 0/8 | 0.242 | 0.045 (0.051) | 1.01 (1.05) | 0 | 9.1 |
| +1 m/s | 12.8 | 128 | 0.48 | 0/8 | 0.238 | 0.115 (0.154) | 1.31 (5.99) | 0 | 23.6 |
| +1.25 m/s | 15.9 | 159 | 0.67 | 8/8 | all fell | | | | |
| +1.5 m/s | 19.1 | 191 | 0.88 | 8/8 | all fell | | | | |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 0.10 s median and 0.10 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `fore128-s2_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.242 (0.242) | +0.0 % | -0.008 | 0.354 (0.420) | 0.35 (left_hind) | 0.97 (left_fore_pitch) | 3.1 % (left_fore_pitch) | 3.1 % (left_fore_pitch) |
| step 1 cm | 0/8 | 0.243 (0.242) | +0.0 % | -0.007 | 0.365 (0.447) | 0.35 (left_hind) | 0.98 (left_fore_pitch) | 3.2 % (left_fore_pitch) | 3.2 % (left_fore_pitch) |
| step 2 cm | 0/8 | 0.242 (0.240) | -0.2 % | -0.008 | 0.324 (0.385) | 0.36 (left_hind) | 0.99 (left_fore_pitch) | 3.3 % (left_fore_pitch) | 3.3 % (left_fore_pitch) |
| step 3 cm | 0/8 | 0.241 (0.237) | -0.4 % | -0.009 | 0.258 (0.385) | 0.36 (left_hind) | 0.99 (left_fore_pitch) | 3.4 % (left_fore_pitch) | 3.4 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `fore128-s2_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.242 (0.242) | +0.0 % | -0.008 | 0.354 (0.420) | 0.35 (left_hind) | 0.97 (left_fore_pitch) | 3.1 % (left_fore_pitch) | 3.1 % (left_fore_pitch) |
| ramp 3 deg | 0/8 | 0.244 (0.242) | +0.5 % | -0.006 | 0.298 (0.353) | 0.35 (left_hind) | 1.04 (left_fore_pitch) | 4.9 % (right_hind_pitch) | 4.9 % (right_hind_pitch) |
| ramp 5 deg | 0/8 | 0.243 (0.241) | +0.1 % | -0.007 | 0.206 (0.280) | 0.36 (left_hind) | 1.07 (left_fore_pitch) | 6.9 % (right_hind_pitch) | 6.9 % (right_hind_pitch) |
| ramp 8 deg | 0/8 | 0.240 (0.238) | -1.1 % | -0.010 | 0.329 (0.452) | 0.39 (left_hind) | 1.16 (right_hind_pitch) | 9.0 % (right_hind_pitch) | 9.0 % (right_hind_pitch) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `fore128-s2_terrain.png`.

