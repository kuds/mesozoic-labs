# Robustness sweep: trike60-fore128-s3 (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-26. One policy, so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Policy `trike60_10002432_steps.zip` (sha256 652082b1aeb26cf4...), run commit f84a6919; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 6774a9622c397e0c...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 13.8 min on 2 worker processes (565 episodes).
- Reproduction check (gait_pipe1's target case, randomisation on, seeds 100000-100004): mean speed 0.2399 m/s against the analysis's 0.2385 m/s; per-episode speeds identical: False.

## Headline

- **No fall and no speed drop of more than 20 %** in: delay, 0.15 m/s (0 ms to 60 ms); friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5); step up (flat to step 3 cm); ramp up (flat to ramp 8 deg).
- **delay, 0.25 m/s**: first break at 40 ms (1/8 fell).
- **Sideways push**: falls start at 0.75 m/s free-body (impulse 9.6 N s; the centre of mass gained 0.42 m/s), pushed to the right (2/8 fell); 1 m/s free-body (impulse 12.8 N s; the centre of mass gained 0.56 m/s), pushed to the left (3/8 fell). Largest push with no fall: 0.5 m/s free-body (impulse 6.4 N s; the centre of mass gained 0.24 m/s), pushed to the right; 0.75 m/s free-body (impulse 9.6 N s; the centre of mass gained 0.41 m/s), pushed to the left. The feet absorb part of every push: the centre of mass gains 38-71 % of the free-body velocity change (pushes of 0.3 m/s and more).

## Where it breaks

First point outward from nominal with a fall or a speed more than 20 % below the nominal point's (same command). "none": no such point in the tested range.

| Sweep | Nominal | Low side | High side |
|---|---|---|---|
| delay, 0.25 m/s | 10 ms | none | **40 ms**: 1/8 fell |
| delay, 0.15 m/s | 10 ms | none | none |
| friction | 0.85 (model) | none | none |
| mass, every body | x1 | none | none |
| payload on the chassis | +0 kg | (not tested) | none |
| force limit | x1 | none | (not tested) |
| position gain kp | x1 | none | none |
| sideways push | none | **0.75 m/s free-body to the right (impulse 9.6 N s; the centre of mass gained 0.42 m/s)**: 2/8 fell | **1 m/s free-body to the left (impulse 12.8 N s; the centre of mass gained 0.56 m/s)**: 3/8 fell |
| step up | flat | (not tested) | none |
| ramp up | flat | (not tested) | none |

Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.259 (0.258) | +8.1 % | +0.009 | 0.294 (0.394) | 0.45 (right_hind) | 0.98 (right_hind_flexion) | 2.2 % (right_fore_pitch) | 2.2 % (right_fore_pitch) |
| 2 ms | 0/8 | 0.253 (0.252) | +5.6 % | +0.003 | 0.337 (0.470) | 0.46 (right_hind) | 0.98 (right_hind_flexion) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |
| 5 ms | 0/8 | 0.247 (0.245) | +2.8 % | -0.003 | 0.406 (0.545) | 0.45 (right_hind) | 0.96 (left_hind_pitch) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |
| 10 ms | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.470 (0.550) | 0.42 (right_hind) | 1.00 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| 15 ms | 0/8 | 0.237 (0.236) | -1.2 % | -0.013 | 0.552 (0.633) | 0.35 (right_hind) | 1.00 (left_hind_pitch) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |
| 20 ms | 0/8 | 0.258 (0.257) | +7.7 % | +0.008 | 0.450 (0.595) | 0.23 (right_hind) | 1.20 (left_hind_pitch) | 7.6 % (right_hind_pitch) | 7.6 % (right_hind_pitch) |
| 30 ms | 0/8 | 0.245 (0.241) | +2.0 % | -0.005 | 0.572 (0.804) | 0.24 (right_fore) | 1.22 (left_hind_pitch) | 8.3 % (left_hind_pitch) | 8.3 % (left_hind_pitch) |
| 40 ms | 1/8 | 0.245 (0.234) | +2.1 % | -0.005 | 1.593 (1.784) | 0.25 (right_fore) | 1.24 (left_hind_pitch) | 12.5 % (right_hind_pitch) | 12.5 % (right_hind_pitch) |
| 60 ms | 3/8 | 0.106 (0.021) | -56.0 % | -0.144 | 0.709 (1.467) | 1.58 (right_fore) | 1.49 (left_hind_pitch) | 20.5 % (left_hind_pitch) | 20.5 % (left_hind_pitch) |

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore128-s3_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.153 (0.151) | +7.3 % | +0.003 | 0.251 (0.308) | 0.70 (right_hind) | 0.89 (right_hind_flexion) | 1.1 % (right_fore_pitch) | 1.1 % (right_fore_pitch) |
| 2 ms | 0/8 | 0.149 (0.148) | +5.0 % | -0.001 | 0.270 (0.346) | 0.70 (right_hind) | 0.89 (right_hind_flexion) | 1.0 % (right_fore_pitch) | 1.0 % (right_fore_pitch) |
| 5 ms | 0/8 | 0.146 (0.145) | +2.6 % | -0.004 | 0.296 (0.348) | 0.70 (right_hind) | 0.89 (right_hind_flexion) | 1.0 % (right_fore_pitch) | 1.0 % (right_fore_pitch) |
| 10 ms | 0/8 | 0.142 (0.141) | +0.0 % | -0.008 | 0.354 (0.394) | 0.70 (right_hind) | 0.87 (right_hind_flexion) | 1.0 % (right_fore_pitch) | 1.0 % (right_fore_pitch) |
| 15 ms | 0/8 | 0.141 (0.139) | -1.2 % | -0.009 | 0.397 (0.457) | 0.68 (right_hind) | 0.85 (right_hind_flexion) | 1.0 % (right_fore_pitch) | 1.0 % (right_fore_pitch) |
| 20 ms | 0/8 | 0.151 (0.149) | +6.2 % | +0.001 | 0.253 (0.313) | 0.49 (right_hind) | 1.10 (left_hind_pitch) | 2.7 % (left_hind_pitch) | 2.7 % (left_hind_pitch) |
| 30 ms | 0/8 | 0.144 (0.143) | +1.5 % | -0.006 | 0.299 (0.387) | 0.44 (right_hind) | 1.13 (left_hind_pitch) | 4.1 % (right_hind_pitch) | 4.1 % (right_hind_pitch) |
| 40 ms | 0/8 | 0.159 (0.151) | +11.6 % | +0.009 | 0.742 (1.055) | 0.42 (right_fore) | 1.37 (left_hind_pitch) | 12.2 % (left_hind_pitch) | 12.2 % (left_hind_pitch) |
| 60 ms | 0/8 | 0.181 (0.162) | +27.5 % | +0.031 | 1.120 (1.582) | 0.40 (left_hind) | 1.43 (left_hind_pitch) | 16.9 % (left_hind_pitch) | 16.9 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `fore128-s3_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.235 (0.233) | -2.3 % | -0.015 | 0.072 (0.129) | 0.65 (right_hind) | 0.86 (right_fore_pitch) | 2.3 % (right_fore_pitch) | 2.3 % (right_fore_pitch) |
| 0.2 | 0/8 | 0.236 (0.235) | -1.6 % | -0.014 | 0.223 (0.363) | 0.65 (right_hind) | 0.85 (right_fore_pitch) | 2.3 % (right_fore_pitch) | 2.3 % (right_fore_pitch) |
| 0.3 | 0/8 | 0.238 (0.236) | -0.9 % | -0.012 | 0.453 (0.551) | 0.61 (right_hind) | 0.82 (right_fore_pitch) | 2.1 % (right_fore_pitch) | 2.1 % (right_fore_pitch) |
| 0.4 | 0/8 | 0.239 (0.237) | -0.4 % | -0.011 | 0.531 (0.628) | 0.58 (right_hind) | 0.82 (left_hind_pitch) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |
| 0.5 | 0/8 | 0.240 (0.238) | -0.2 % | -0.010 | 0.544 (0.669) | 0.55 (right_hind) | 0.88 (left_hind_pitch) | 1.8 % (right_fore_pitch) | 1.8 % (right_fore_pitch) |
| 0.6 | 0/8 | 0.240 (0.238) | +0.0 % | -0.010 | 0.514 (0.621) | 0.53 (right_hind) | 0.94 (left_hind_pitch) | 1.8 % (right_fore_pitch) | 1.8 % (right_fore_pitch) |
| 0.8 | 0/8 | 0.240 (0.239) | +0.1 % | -0.010 | 0.458 (0.557) | 0.45 (right_hind) | 0.99 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| 0.85 (model) | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.470 (0.550) | 0.42 (right_hind) | 1.00 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| 1 | 0/8 | 0.239 (0.238) | -0.3 % | -0.011 | 0.497 (0.559) | 0.37 (right_hind) | 1.00 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| 1.2 | 0/8 | 0.239 (0.237) | -0.5 % | -0.011 | 0.508 (0.617) | 0.33 (right_hind) | 1.02 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `fore128-s3_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.239 (0.237) | -0.5 % | -0.011 | 0.601 (0.708) | 0.41 (right_hind) | 0.87 (left_hind_pitch) | 1.6 % (right_fore_pitch) | 1.6 % (right_fore_pitch) |
| x0.9 | 0/8 | 0.239 (0.238) | -0.2 % | -0.011 | 0.560 (0.674) | 0.42 (right_hind) | 0.93 (left_hind_pitch) | 1.8 % (right_fore_pitch) | 1.8 % (right_fore_pitch) |
| x1 | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.470 (0.550) | 0.42 (right_hind) | 1.00 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| x1.1 | 0/8 | 0.240 (0.239) | +0.1 % | -0.010 | 0.386 (0.469) | 0.42 (right_hind) | 1.06 (left_hind_pitch) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |
| x1.2 | 0/8 | 0.240 (0.239) | +0.1 % | -0.010 | 0.339 (0.406) | 0.42 (right_hind) | 1.12 (left_hind_pitch) | 2.2 % (right_fore_pitch) | 2.2 % (right_fore_pitch) |
| x1.4 | 0/8 | 0.240 (0.239) | -0.1 % | -0.010 | 0.190 (0.256) | 0.42 (right_hind) | 1.28 (right_hind_flexion) | 3.0 % (left_hind_pitch) | 3.0 % (left_hind_pitch) |
| x1.6 | 0/8 | 0.239 (0.237) | -0.5 % | -0.011 | 0.056 (0.146) | 0.42 (right_hind) | 1.43 (right_hind_flexion) | 5.5 % (left_hind_pitch) | 5.5 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `fore128-s3_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.470 (0.550) | 0.42 (right_hind) | 1.00 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| +0.5 kg | 0/8 | 0.239 (0.238) | -0.3 % | -0.011 | 0.439 (0.535) | 0.43 (right_hind) | 1.02 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| +1 kg | 0/8 | 0.239 (0.238) | -0.4 % | -0.011 | 0.425 (0.514) | 0.43 (right_hind) | 1.06 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| +2 kg | 0/8 | 0.238 (0.237) | -0.6 % | -0.012 | 0.366 (0.449) | 0.44 (right_hind) | 1.11 (right_hind_flexion) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |
| +3 kg | 0/8 | 0.237 (0.235) | -1.0 % | -0.013 | 0.326 (0.464) | 0.45 (right_hind) | 1.19 (right_hind_flexion) | 2.3 % (left_hind_pitch) | 2.3 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `fore128-s3_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.236 (0.235) | -1.4 % | -0.014 | 0.301 (0.384) | 0.42 (right_hind) | 0.84 (left_hind_pitch) | 0.0 % (left_fore_roll) | 45.5 % (left_hind_pitch) |
| x0.55 | 0/8 | 0.239 (0.237) | -0.6 % | -0.011 | 0.406 (0.493) | 0.44 (right_hind) | 0.93 (left_hind_pitch) | 0.0 % (left_fore_roll) | 18.0 % (left_hind_pitch) |
| x0.7 | 0/8 | 0.239 (0.238) | -0.2 % | -0.011 | 0.443 (0.549) | 0.42 (right_hind) | 0.97 (left_hind_pitch) | 0.0 % (left_fore_roll) | 6.6 % (left_hind_pitch) |
| x0.85 | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.455 (0.532) | 0.42 (right_hind) | 0.98 (left_hind_pitch) | 0.0 % (left_fore_roll) | 2.7 % (left_hind_pitch) |
| x1 | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.470 (0.550) | 0.42 (right_hind) | 1.00 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `fore128-s3_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.217 (0.214) | -9.7 % | -0.033 | 1.515 (1.660) | 0.14 (right_hind) | 0.95 (right_hind_flexion) | 1.9 % (left_hind_pitch) | 1.9 % (left_hind_pitch) |
| x0.6 | 0/8 | 0.224 (0.222) | -6.6 % | -0.026 | 0.821 (0.923) | 0.32 (right_hind) | 0.92 (right_hind_flexion) | 0.6 % (right_fore_pitch) | 0.6 % (right_fore_pitch) |
| x0.8 | 0/8 | 0.233 (0.231) | -2.8 % | -0.017 | 0.684 (0.788) | 0.41 (right_hind) | 0.93 (right_hind_flexion) | 1.2 % (right_fore_pitch) | 1.2 % (right_fore_pitch) |
| x1 | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.470 (0.550) | 0.42 (right_hind) | 1.00 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| x1.2 | 0/8 | 0.245 (0.244) | +2.2 % | -0.005 | 0.237 (0.375) | 0.42 (right_hind) | 1.09 (left_hind_pitch) | 2.7 % (right_fore_pitch) | 2.7 % (right_fore_pitch) |
| x1.5 | 0/8 | 0.251 (0.250) | +4.7 % | +0.001 | 0.096 (0.226) | 0.40 (right_hind) | 1.21 (left_hind_pitch) | 5.7 % (left_hind_pitch) | 5.7 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `fore128-s3_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 38-71 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.1 | 191 | 1.06 | 8/8 | all fell | | | | |
| -1.25 m/s | 15.9 | 159 | 0.84 | 8/8 | all fell | | | | |
| -1 m/s | 12.8 | 128 | 0.62 | 8/8 | all fell | | | | |
| -0.75 m/s | 9.6 | 96 | 0.42 | 2/8 | 0.240 | 0.104 (0.220) | 6.14 (9.74) | 0 | 34.8 |
| -0.5 m/s | 6.4 | 64 | 0.24 | 0/8 | 0.240 | 0.043 (0.088) | 5.43 (6.76) | 2 | 15.1 |
| -0.4 m/s | 5.1 | 51 | 0.17 | 0/8 | 0.240 | 0.031 (0.053) | 4.61 (6.26) | 1 | 13.6 |
| -0.3 m/s | 3.8 | 38 | 0.12 | 0/8 | 0.240 | 0.022 (0.033) | 6.45 (9.24) | 2 | 11.3 |
| -0.2 m/s | 2.6 | 26 | 0.07 | 0/8 | 0.240 | 0.016 (0.020) | 4.65 (7.75) | 1 | 10.5 |
| -0.1 m/s | 1.3 | 13 | 0.04 | 0/8 | 0.240 | 0.011 (0.015) | 6.03 (6.98) | 1 | 10.1 |
| none | 0.0 | 0 |  | 0/8 | 0.240 | (no push) | 3.79 (8.57) | 3 | 9.4 |
| +0.1 m/s | 1.3 | 13 | 0.04 | 0/8 | 0.240 | 0.011 (0.019) | 1.96 (6.88) | 2 | 9.4 |
| +0.2 m/s | 2.6 | 26 | 0.09 | 0/8 | 0.240 | 0.023 (0.030) | 4.10 (6.94) | 2 | 9.2 |
| +0.3 m/s | 3.8 | 38 | 0.14 | 0/8 | 0.240 | 0.036 (0.046) | 6.85 (9.13) | 2 | 9.1 |
| +0.4 m/s | 5.1 | 51 | 0.19 | 0/8 | 0.240 | 0.049 (0.061) | 6.36 (9.84) | 2 | 9.5 |
| +0.5 m/s | 6.4 | 64 | 0.24 | 0/8 | 0.241 | 0.065 (0.075) | 5.89 (8.98) | 3 | 9.3 |
| +0.75 m/s | 9.6 | 96 | 0.41 | 0/8 | 0.241 | 0.087 (0.102) | 7.32 (10.40) | 3 | 9.2 |
| +1 m/s | 12.8 | 128 | 0.56 | 3/8 | 0.241 | 0.158 (0.219) | 5.72 (6.34) | 2 | 35.6 |
| +1.25 m/s | 15.9 | 159 | 0.74 | 8/8 | all fell | | | | |
| +1.5 m/s | 19.1 | 191 | 0.92 | 8/8 | all fell | | | | |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 3.79 s median and 8.57 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `fore128-s3_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.470 (0.550) | 0.42 (right_hind) | 1.00 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| step 1 cm | 0/8 | 0.240 (0.239) | +0.1 % | -0.010 | 0.478 (0.541) | 0.42 (right_hind) | 1.00 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| step 2 cm | 0/8 | 0.241 (0.239) | +0.3 % | -0.009 | 0.468 (0.529) | 0.41 (right_hind) | 0.99 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| step 3 cm | 0/8 | 0.241 (0.240) | +0.4 % | -0.009 | 0.429 (0.537) | 0.41 (right_hind) | 1.02 (left_hind_pitch) | 2.0 % (right_fore_pitch) | 2.0 % (right_fore_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `fore128-s3_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.470 (0.550) | 0.42 (right_hind) | 1.00 (left_hind_pitch) | 1.9 % (right_fore_pitch) | 1.9 % (right_fore_pitch) |
| ramp 3 deg | 0/8 | 0.242 (0.241) | +0.7 % | -0.008 | 0.524 (0.618) | 0.41 (right_hind) | 1.10 (left_hind_pitch) | 2.1 % (right_fore_pitch) | 2.1 % (right_fore_pitch) |
| ramp 5 deg | 0/8 | 0.243 (0.242) | +1.4 % | -0.007 | 0.481 (0.548) | 0.40 (right_hind) | 1.15 (left_hind_pitch) | 2.2 % (left_hind_pitch) | 2.2 % (left_hind_pitch) |
| ramp 8 deg | 0/8 | 0.245 (0.244) | +2.1 % | -0.005 | 0.392 (0.560) | 0.39 (right_hind) | 1.22 (left_hind_pitch) | 3.0 % (left_hind_pitch) | 3.0 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `fore128-s3_terrain.png`.

