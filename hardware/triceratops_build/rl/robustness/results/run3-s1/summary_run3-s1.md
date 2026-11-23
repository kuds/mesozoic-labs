# Robustness sweep: trike60-run3-s1 (SIMULATION)

Written by `rl/robustness/sweep.py` on 2026-11-23. One policy, so this is a report, not a verdict: a simulation of a model whose masses are estimates. Definitions in `rl/robustness/README.md`.

- Policy `trike60_10002432_steps.zip` (sha256 b1c97ffde02c8781...), run commit efd1e1c7; environment `hardware/triceratops_build/rl/trike_env.py` (sha256 402819fcc1c3879d...).
- Every point: 8 episodes (seeds 200000-200007), 20 s, deterministic actions, observation noise and heading drift on, no domain randomisation. The nominal point is the **nominal model with observation noise on** (the model as built, delay 10 ms, 0.25 m/s command); the gait report's "nominal model" case has the noise off, so the two are not directly comparable. Speeds, drift and drive figures are over the episodes that did not fall, from t = 2 s.
- "At ... limit" columns give the **worst single joint's** share of 1 ms steps at or above 99 % of the limit; the gait report quotes per-joint and per-group figures.
- Wall time 9.8 min on 2 worker processes (565 episodes).
- Reproduction check (gait_pipe1's target case, randomisation on, seeds 100000-100004): mean speed 0.2393 m/s against the analysis's 0.2385 m/s; per-episode speeds identical: False.

## Headline

- **No fall and no speed drop of more than 20 %** in: friction (0.15 to 1.2); mass, every body (x0.8 to x1.6); payload on the chassis (+0 kg to +3 kg); force limit (x0.4 to x1); position gain kp (x0.4 to x1.5); step up (flat to step 3 cm); ramp up (flat to ramp 8 deg).
- **delay, 0.25 m/s**: first break at 60 ms (speed 0.059 m/s, 75 % below nominal).
- **delay, 0.15 m/s**: first break at 60 ms (speed 0.070 m/s, 51 % below nominal).
- **Sideways push**: falls start at 1 m/s free-body (impulse 12.7 N s; the centre of mass gained 0.54 m/s), pushed to the left (2/8 fell); 1.25 m/s free-body (impulse 15.9 N s; the centre of mass gained 0.62 m/s), pushed to the right (8/8 fell). Largest push with no fall: 0.75 m/s free-body (impulse 9.5 N s; the centre of mass gained 0.37 m/s), pushed to the left; 1 m/s free-body (impulse 12.7 N s; the centre of mass gained 0.45 m/s), pushed to the right. The feet absorb part of every push: the centre of mass gains 26-62 % of the free-body velocity change (pushes of 0.3 m/s and more).

## Where it breaks

First point outward from nominal with a fall or a speed more than 20 % below the nominal point's (same command). "none": no such point in the tested range.

| Sweep | Nominal | Low side | High side |
|---|---|---|---|
| delay, 0.25 m/s | 10 ms | none | **60 ms**: speed 0.059 m/s, 75 % below nominal |
| delay, 0.15 m/s | 10 ms | none | **60 ms**: speed 0.070 m/s, 51 % below nominal |
| friction | 0.85 (model) | none | none |
| mass, every body | x1 | none | none |
| payload on the chassis | +0 kg | (not tested) | none |
| force limit | x1 | none | (not tested) |
| position gain kp | x1 | none | none |
| sideways push | none | **1.25 m/s free-body to the right (impulse 15.9 N s; the centre of mass gained 0.62 m/s)**: 8/8 fell | **1 m/s free-body to the left (impulse 12.7 N s; the centre of mass gained 0.54 m/s)**: 2/8 fell |
| step up | flat | (not tested) | none |
| ramp up | flat | (not tested) | none |

Training ranges: delay 0-10 ms, friction 0.6-1.0, mass and inertia x0.9-1.1 per body (drawn per body, so the robot's total varied less), kp x0.9-1.1 and force limits x0.85-1.0 per drive; flat floor, no pushes.

## delay, 0.25 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.257 (0.256) | +7.2 % | +0.007 | 0.353 (0.469) | 0.17 (left_hind) | 0.77 (right_hind_pitch) | 1.4 % (left_fore_pitch) | 1.4 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.252 (0.252) | +5.2 % | +0.002 | 0.340 (0.450) | 0.17 (left_hind) | 0.78 (left_hind_pitch) | 1.4 % (left_fore_pitch) | 1.4 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.246 (0.246) | +2.7 % | -0.004 | 0.307 (0.430) | 0.17 (left_hind) | 0.79 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.255 (0.348) | 0.16 (left_hind) | 0.82 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| 15 ms | 0/8 | 0.237 (0.237) | -1.1 % | -0.013 | 0.244 (0.327) | 0.16 (left_hind) | 0.84 (left_hind_pitch) | 1.4 % (left_fore_pitch) | 1.4 % (left_fore_pitch) |
| 20 ms | 0/8 | 0.248 (0.245) | +3.5 % | -0.002 | 0.527 (0.848) | 0.18 (left_hind) | 1.20 (left_hind_pitch) | 10.7 % (left_hind_pitch) | 10.7 % (left_hind_pitch) |
| 30 ms | 0/8 | 0.236 (0.232) | -1.5 % | -0.014 | 0.477 (0.579) | 0.19 (left_hind) | 1.20 (left_hind_pitch) | 11.0 % (left_hind_pitch) | 11.0 % (left_hind_pitch) |
| 40 ms | 0/8 | 0.215 (0.000) | -10.3 % | -0.035 | 0.538 (0.897) | 0.22 (left_fore) | 1.26 (left_fore_pitch) | 12.2 % (left_hind_pitch) | 12.2 % (left_hind_pitch) |
| 60 ms | 0/8 | 0.059 (0.003) | -75.4 % | -0.191 | 0.142 (0.359) | 0.69 (left_fore) | 1.38 (left_fore_pitch) | 4.8 % (left_fore_pitch) | 4.8 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `run3-s1_delay.png`.

## delay, 0.15 m/s

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0 ms | 0/8 | 0.153 (0.152) | +8.5 % | +0.003 | 0.143 (0.182) | 0.41 (left_hind) | 0.74 (left_hind_pitch) | 0.6 % (left_fore_pitch) | 0.6 % (left_fore_pitch) |
| 2 ms | 0/8 | 0.150 (0.149) | +6.2 % | -0.000 | 0.128 (0.187) | 0.40 (left_hind) | 0.75 (left_hind_pitch) | 0.5 % (left_fore_pitch) | 0.5 % (left_fore_pitch) |
| 5 ms | 0/8 | 0.146 (0.145) | +3.3 % | -0.004 | 0.116 (0.160) | 0.39 (left_hind) | 0.78 (left_hind_pitch) | 0.5 % (left_fore_pitch) | 0.5 % (left_fore_pitch) |
| 10 ms | 0/8 | 0.141 (0.140) | +0.0 % | -0.009 | 0.101 (0.153) | 0.38 (left_hind) | 0.78 (left_hind_pitch) | 0.5 % (left_fore_pitch) | 0.5 % (left_fore_pitch) |
| 15 ms | 0/8 | 0.139 (0.138) | -1.5 % | -0.011 | 0.075 (0.141) | 0.35 (left_hind) | 0.81 (left_hind_pitch) | 0.5 % (left_fore_pitch) | 0.5 % (left_fore_pitch) |
| 20 ms | 0/8 | 0.146 (0.144) | +3.6 % | -0.004 | 0.326 (0.417) | 0.33 (left_hind) | 1.20 (left_hind_pitch) | 10.1 % (left_hind_pitch) | 10.1 % (left_hind_pitch) |
| 30 ms | 0/8 | 0.143 (0.141) | +1.3 % | -0.007 | 0.409 (0.514) | 0.31 (left_hind) | 1.27 (left_hind_pitch) | 13.2 % (left_hind_pitch) | 13.2 % (left_hind_pitch) |
| 40 ms | 0/8 | 0.123 (-0.002) | -13.1 % | -0.027 | 0.402 (0.821) | 0.42 (left_fore) | 1.25 (left_fore_pitch) | 13.0 % (left_hind_pitch) | 13.0 % (left_hind_pitch) |
| 60 ms | 0/8 | 0.070 (-0.002) | -50.7 % | -0.080 | 0.188 (0.395) | 0.79 (left_fore) | 1.39 (left_fore_pitch) | 9.5 % (left_fore_pitch) | 9.5 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): 30 ms, 40 ms, 60 ms.

Chart: `run3-s1_delay.png`.

## friction

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0/8 | 0.237 (0.236) | -1.3 % | -0.013 | 0.127 (0.225) | 0.32 (left_hind) | 0.74 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| 0.2 | 0/8 | 0.238 (0.237) | -0.7 % | -0.012 | 0.045 (0.168) | 0.29 (left_hind) | 0.75 (right_hind_flexion) | 1.6 % (left_fore_pitch) | 1.6 % (left_fore_pitch) |
| 0.3 | 0/8 | 0.240 (0.239) | +0.1 % | -0.010 | 0.063 (0.119) | 0.25 (left_hind) | 0.76 (right_hind_flexion) | 1.5 % (left_fore_pitch) | 1.5 % (left_fore_pitch) |
| 0.4 | 0/8 | 0.241 (0.239) | +0.3 % | -0.009 | 0.058 (0.146) | 0.23 (left_hind) | 0.77 (right_hind_flexion) | 1.5 % (left_fore_pitch) | 1.5 % (left_fore_pitch) |
| 0.5 | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.162 (0.306) | 0.22 (left_hind) | 0.77 (left_hind_pitch) | 1.4 % (left_fore_pitch) | 1.4 % (left_fore_pitch) |
| 0.6 | 0/8 | 0.240 (0.239) | -0.1 % | -0.010 | 0.225 (0.335) | 0.20 (left_hind) | 0.81 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| 0.8 | 0/8 | 0.240 (0.239) | -0.1 % | -0.010 | 0.259 (0.356) | 0.17 (left_hind) | 0.82 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| 0.85 (model) | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.255 (0.348) | 0.16 (left_hind) | 0.82 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| 1 | 0/8 | 0.240 (0.239) | +0.1 % | -0.010 | 0.236 (0.315) | 0.15 (left_hind) | 0.83 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| 1.2 | 0/8 | 0.240 (0.239) | +0.1 % | -0.010 | 0.222 (0.320) | 0.15 (left_hind) | 0.84 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): 0.15, 0.2.

Chart: `run3-s1_friction.png`.

## mass, every body

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.8 | 0/8 | 0.239 (0.239) | -0.2 % | -0.011 | 0.220 (0.327) | 0.15 (left_hind) | 0.70 (left_hind_pitch) | 1.2 % (left_fore_pitch) | 1.2 % (left_fore_pitch) |
| x0.9 | 0/8 | 0.240 (0.239) | -0.1 % | -0.010 | 0.248 (0.314) | 0.16 (left_hind) | 0.76 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| x1 | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.255 (0.348) | 0.16 (left_hind) | 0.82 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| x1.1 | 0/8 | 0.240 (0.239) | -0.0 % | -0.010 | 0.253 (0.373) | 0.16 (left_hind) | 0.88 (left_hind_pitch) | 1.4 % (left_fore_pitch) | 1.4 % (left_fore_pitch) |
| x1.2 | 0/8 | 0.240 (0.239) | -0.2 % | -0.010 | 0.264 (0.366) | 0.16 (left_hind) | 0.93 (left_hind_pitch) | 1.5 % (left_hind_pitch) | 1.5 % (left_hind_pitch) |
| x1.4 | 0/8 | 0.238 (0.238) | -0.7 % | -0.012 | 0.251 (0.350) | 0.16 (left_hind) | 1.03 (left_hind_pitch) | 3.0 % (left_hind_pitch) | 3.0 % (left_hind_pitch) |
| x1.6 | 0/8 | 0.237 (0.235) | -1.4 % | -0.013 | 0.288 (0.399) | 0.16 (left_hind) | 1.11 (left_hind_pitch) | 4.8 % (left_hind_pitch) | 4.8 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): x1.4, x1.6.

Chart: `run3-s1_mass.png`.

## payload on the chassis

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| +0 kg | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.255 (0.348) | 0.16 (left_hind) | 0.82 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| +0.5 kg | 0/8 | 0.239 (0.238) | -0.3 % | -0.011 | 0.241 (0.369) | 0.17 (left_hind) | 0.85 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| +1 kg | 0/8 | 0.239 (0.238) | -0.5 % | -0.011 | 0.229 (0.328) | 0.17 (left_hind) | 0.87 (left_hind_pitch) | 1.4 % (left_fore_pitch) | 1.4 % (left_fore_pitch) |
| +2 kg | 0/8 | 0.238 (0.237) | -0.9 % | -0.012 | 0.204 (0.317) | 0.18 (left_hind) | 0.91 (left_hind_pitch) | 1.5 % (left_hind_pitch) | 1.5 % (left_hind_pitch) |
| +3 kg | 0/8 | 0.236 (0.235) | -1.5 % | -0.014 | 0.188 (0.295) | 0.19 (left_hind) | 0.97 (right_hind_flexion) | 1.7 % (left_hind_pitch) | 1.7 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): +2 kg, +3 kg.

Chart: `run3-s1_mass.png`.

## force limit

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.235 (0.234) | -2.0 % | -0.015 | 0.320 (0.396) | 0.18 (left_hind) | 0.70 (right_hind_flexion) | 0.0 % (left_fore_roll) | 26.7 % (left_hind_pitch) |
| x0.55 | 0/8 | 0.238 (0.238) | -0.6 % | -0.012 | 0.282 (0.404) | 0.17 (left_hind) | 0.76 (left_hind_pitch) | 0.0 % (left_fore_roll) | 10.9 % (left_hind_pitch) |
| x0.7 | 0/8 | 0.239 (0.239) | -0.2 % | -0.011 | 0.268 (0.356) | 0.17 (left_hind) | 0.80 (left_hind_pitch) | 0.0 % (left_fore_roll) | 4.3 % (left_hind_pitch) |
| x0.85 | 0/8 | 0.240 (0.239) | -0.0 % | -0.010 | 0.262 (0.362) | 0.17 (left_hind) | 0.82 (left_hind_pitch) | 0.0 % (left_fore_roll) | 2.0 % (left_hind_pitch) |
| x1 | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.255 (0.348) | 0.16 (left_hind) | 0.82 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.55.

Chart: `run3-s1_drive.png`.

## position gain kp

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| x0.4 | 0/8 | 0.217 (0.216) | -9.7 % | -0.033 | 0.214 (0.302) | 0.19 (left_hind) | 0.67 (right_hind_pitch) | 0.3 % (left_hind_pitch) | 0.3 % (left_hind_pitch) |
| x0.6 | 0/8 | 0.229 (0.228) | -4.6 % | -0.021 | 0.232 (0.301) | 0.18 (left_hind) | 0.74 (right_hind_pitch) | 0.4 % (left_hind_pitch) | 0.4 % (left_hind_pitch) |
| x0.8 | 0/8 | 0.236 (0.235) | -1.7 % | -0.014 | 0.249 (0.315) | 0.17 (left_hind) | 0.78 (right_hind_pitch) | 0.8 % (left_fore_pitch) | 0.8 % (left_fore_pitch) |
| x1 | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.255 (0.348) | 0.16 (left_hind) | 0.82 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| x1.2 | 0/8 | 0.242 (0.241) | +0.9 % | -0.008 | 0.219 (0.280) | 0.16 (left_hind) | 0.90 (left_hind_pitch) | 1.9 % (left_fore_pitch) | 1.9 % (left_fore_pitch) |
| x1.5 | 0/8 | 0.244 (0.242) | +1.6 % | -0.006 | 0.104 (0.174) | 0.19 (left_hind) | 1.02 (left_hind_pitch) | 4.3 % (left_hind_pitch) | 4.3 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): x0.4, x0.6, x1.5.

Chart: `run3-s1_drive.png`.

## sideways push

Each push is quoted three ways: the free-body velocity change dv (the velocity the whole robot would gain if nothing resisted), the impulse (robot mass x dv, applied as a constant force for 0.1 s), and the velocity change its centre of mass actually got, measured against the same seed's unpushed episode (identical until the push). The feet's friction absorbs the rest: the centre of mass gains 26-62 % of dv for pushes of 0.3 m/s and more.

| Push: free-body dv | Impulse N s | Force N for 0.1 s | CoM dv achieved m/s | Falls | Speed m/s | Extra sideways displacement m, mean (max) | Recovery s, median (max) | Not recovered | Peak roll deg |
|---|---|---|---|---|---|---|---|---|---|
| -1.5 m/s | 19.0 | 190 | 0.80 | 8/8 | all fell | | | | |
| -1.25 m/s | 15.9 | 159 | 0.62 | 8/8 | all fell | | | | |
| -1 m/s | 12.7 | 127 | 0.45 | 0/8 | 0.240 | 0.096 (0.128) | 0.84 (0.92) | 0 | 13.9 |
| -0.75 m/s | 9.5 | 95 | 0.30 | 0/8 | 0.240 | 0.054 (0.071) | 0.34 (0.64) | 0 | 7.5 |
| -0.5 m/s | 6.3 | 63 | 0.17 | 0/8 | 0.240 | 0.039 (0.064) | 0.32 (0.56) | 0 | 7.3 |
| -0.4 m/s | 5.1 | 51 | 0.12 | 0/8 | 0.240 | 0.042 (0.059) | 0.33 (0.37) | 0 | 7.2 |
| -0.3 m/s | 3.8 | 38 | 0.08 | 0/8 | 0.240 | 0.029 (0.059) | 0.27 (1.26) | 0 | 6.8 |
| -0.2 m/s | 2.5 | 25 | 0.04 | 0/8 | 0.240 | 0.022 (0.044) | 0.18 (0.32) | 0 | 6.2 |
| -0.1 m/s | 1.3 | 13 | 0.02 | 0/8 | 0.240 | 0.012 (0.033) | 0.10 (0.18) | 0 | 6.8 |
| none | 0.0 | 0 |  | 0/8 | 0.240 | (no push) | 0.10 (1.31) | 0 | 5.9 |
| +0.1 m/s | 1.3 | 13 | 0.03 | 0/8 | 0.240 | 0.007 (0.019) | 0.10 (1.30) | 0 | 5.8 |
| +0.2 m/s | 2.5 | 25 | 0.06 | 0/8 | 0.240 | 0.011 (0.023) | 0.10 (1.29) | 0 | 6.0 |
| +0.3 m/s | 3.8 | 38 | 0.11 | 0/8 | 0.240 | 0.015 (0.023) | 0.59 (1.30) | 0 | 6.4 |
| +0.4 m/s | 5.1 | 51 | 0.16 | 0/8 | 0.240 | 0.021 (0.030) | 0.10 (1.29) | 0 | 7.6 |
| +0.5 m/s | 6.3 | 63 | 0.22 | 0/8 | 0.240 | 0.027 (0.040) | 0.51 (1.25) | 0 | 9.3 |
| +0.75 m/s | 9.5 | 95 | 0.37 | 0/8 | 0.240 | 0.056 (0.075) | 0.86 (1.02) | 0 | 14.8 |
| +1 m/s | 12.7 | 127 | 0.54 | 2/8 | 0.239 | 0.129 (0.189) | 1.24 (1.49) | 0 | 35.3 |
| +1.25 m/s | 15.9 | 159 | 0.72 | 8/8 | all fell | | | | |
| +1.5 m/s | 19.0 | 190 | 0.93 | 8/8 | all fell | | | | |

Noise floor: the unpushed episodes, measured the same way from t = 8 s, score a recovery time of 0.10 s median and 1.31 s at most (the gait's own sideways sway can exceed the threshold), so recovery times below that are not distinguishable from walking unpushed.

Fall terminations: `tilt`.

Beyond the requested set (added to find the edge): -1.5 m/s, -1.25 m/s, -1 m/s, -0.75 m/s, +0.75 m/s, +1 m/s, +1.25 m/s, +1.5 m/s.

Chart: `run3-s1_push.png`.

## step up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.255 (0.348) | 0.16 (left_hind) | 0.82 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| step 1 cm | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.240 (0.343) | 0.16 (left_hind) | 0.83 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| step 2 cm | 0/8 | 0.240 (0.239) | -0.0 % | -0.010 | 0.255 (0.371) | 0.17 (left_hind) | 0.84 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| step 3 cm | 0/8 | 0.239 (0.238) | -0.3 % | -0.011 | 0.260 (0.358) | 0.17 (left_hind) | 0.86 (left_hind_pitch) | 1.5 % (left_fore_pitch) | 1.5 % (left_fore_pitch) |

Beyond the requested set (added to find the edge): step 3 cm.

Chart: `run3-s1_terrain.png`.

## ramp up

| Point | Falls | Speed m/s, mean (min) | vs nominal | Speed error m/s | Sideways drift m, mean (worst) | Slip ratio, worst foot | Heat: worst joint RMS / rating | At nominal limit, worst joint | At this point's limit, worst joint |
|---|---|---|---|---|---|---|---|---|---|
| flat | 0/8 | 0.240 (0.239) | +0.0 % | -0.010 | 0.255 (0.348) | 0.16 (left_hind) | 0.82 (left_hind_pitch) | 1.3 % (left_fore_pitch) | 1.3 % (left_fore_pitch) |
| ramp 3 deg | 0/8 | 0.240 (0.239) | -0.1 % | -0.010 | 0.265 (0.418) | 0.17 (left_hind) | 0.89 (left_hind_pitch) | 1.8 % (left_hind_pitch) | 1.8 % (left_hind_pitch) |
| ramp 5 deg | 0/8 | 0.240 (0.240) | +0.2 % | -0.010 | 0.213 (0.311) | 0.17 (left_hind) | 0.91 (left_hind_pitch) | 2.7 % (left_hind_pitch) | 2.7 % (left_hind_pitch) |
| ramp 8 deg | 0/8 | 0.240 (0.239) | +0.1 % | -0.010 | 0.216 (0.405) | 0.19 (left_hind) | 0.97 (right_hind_pitch) | 3.5 % (left_hind_pitch) | 3.5 % (left_hind_pitch) |

Beyond the requested set (added to find the edge): ramp 8 deg.

Chart: `run3-s1_terrain.png`.

