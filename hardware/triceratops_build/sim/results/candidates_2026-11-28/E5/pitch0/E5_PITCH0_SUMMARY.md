# E5 pitch-0 branch: summary (SIMULATION)

28 Nov 2026. **Changed after a result:** this branch was planned after E5's pitched held-out results and review finding E5-R1. Plan: `E5_PITCH0_PLAN.md` (sha256 recorded before any episode; unchanged, see `e5p0_harness_checks.json`). It runs prereg section 3 item 9's alternative: plan.json's stage-1 points with body pitch set to 0, as a contingency for the lead's open ruling. It does not make that ruling. Nothing measured; nothing adopted.

## Outcome (plan item 7: SUCCEEDS only if all four held-out sets pass)

| Size | Chosen pitch-0 setting (seeds 101-110 only) | 301-310 | 401-410 | 501-510 | 601-610 | Lead's sets 301-410 | Rerun sets 501-610 | Pitch-0 retune |
|---|---|---|---|---|---|---|---|---|
| 60 cm | H0.60-p0-s1-24: period 0.845 s, duty 0.628, body pitch +0.00 deg, lift 25.0 mm, reach 87.0 mm | pass: falls 0, J3 0.2556 m/s, slow-gait max 0, knee 0.829 | pass: falls 0, J3 0.2564 m/s, slow-gait max 0, knee 0.759 | pass: falls 0, J3 0.2509 m/s, slow-gait max 0, knee 0.961 | pass: falls 0, J3 0.2597 m/s, slow-gait max 0, knee 0.680 | pass | pass | **SUCCEEDS** |
| 50 cm | H0.50-p0-s1-15: period 0.638 s, duty 0.532, body pitch +0.00 deg, lift 23.1 mm, reach 81.9 mm | pass: falls 0, J3 0.2491 m/s, slow-gait max 0, knee 0.680 | pass: falls 0, J3 0.2490 m/s, slow-gait max 0, knee 0.709 | pass: falls 0, J3 0.2479 m/s, slow-gait max 0, knee 0.681 | pass: falls 0, J3 0.2476 m/s, slow-gait max 0, knee 0.707 | pass | pass | **SUCCEEDS** |

Per set: falls in J1-J4; J3 mean speed 2-20 s (rule >= 0.225 m/s); the largest slow-gait count in any condition (rule <= 1 of 10); the worst hind knee RMS over its 6.75 N m continuous rating (rule <= 1.0). One controller (the trot), as the lead's rule is written.


## Tuning (seeds 101-110 only)

**0.60 m.** 33 distinct settings, 39 setting runs (budget 60; plan item 5 caps this branch at 47). Stage 3 not run: a stage-2 setting met the rule. Chosen: H0.60-p0-s1-24 (s2), source point H0.60-s1-24; meets the rule on 101-110: yes.

| Rank | Setting | Values (pitch 0) | Falls J1+J3 | Slow-gait (of 20) | Worst hind knee | J3 speed, m/s |
|---|---|---|---|---|---|---|
| 1 | H0.60-p0-s1-24 | period 0.845 s, duty 0.628, body pitch +0.00 deg, lift 25.0 mm, reach 87.0 mm | 0 | 0 | 0.647 | 0.2604 |
| 2 | H0.60-p0-s1-12 | period 0.706 s, duty 0.603, body pitch +0.00 deg, lift 20.4 mm, reach 66.7 mm | 0 | 0 | 0.696 | 0.2494 |
| 3 | H0.60-p0-s1-02 | period 0.668 s, duty 0.596, body pitch +0.00 deg, lift 24.5 mm, reach 79.1 mm | 0 | 0 | 0.703 | 0.2554 |
| 4 | H0.60-p0-s1-30 | period 0.583 s, duty 0.618, body pitch +0.00 deg, lift 21.8 mm, reach 99.4 mm | 0 | 0 | 0.711 | 0.2553 |
| 5 | H0.60-p0-s1-13 | period 0.770 s, duty 0.548, body pitch +0.00 deg, lift 26.3 mm, reach 63.6 mm | 0 | 0 | 0.720 | 0.2527 |
| 6 | H0.60-p0-s1-07 | period 0.657 s, duty 0.538, body pitch +0.00 deg, lift 22.0 mm, reach 71.2 mm | 0 | 0 | 0.721 | 0.2489 |
| 31 | H0.60-p0-s1-00 | period 0.700 s, duty 0.550, body pitch +0.00 deg, lift 35.0 mm, reach 80.0 mm | 0 | 15 | 1.141 | 0.2181 |

| Stage | Setting | Falls J1-J4 | J3 speed, m/s | Slow-gait total (max) | Worst hind knee | Meets rule |
|---|---|---|---|---|---|---|
| s2 | H0.60-p0-s1-24 | 0 | 0.2603 | 0 (0) | 0.675 | yes |
| s2 | H0.60-p0-s1-12 | 0 | 0.2497 | 0 (0) | 0.700 | yes |
| s2 | H0.60-p0-s1-02 | 0 | 0.2555 | 0 (0) | 0.703 | yes |
| s2 | H0.60-p0-s1-30 | 0 | 0.2553 | 0 (0) | 0.711 | yes |
| s2 | H0.60-p0-s1-13 | 0 | 0.2523 | 0 (0) | 0.720 | yes |
| s2 | H0.60-p0-s1-07 | 0 | 0.2488 | 0 (0) | 0.765 | yes |

**0.50 m.** 33 distinct settings, 39 setting runs (budget 60; plan item 5 caps this branch at 47). Stage 3 not run: a stage-2 setting met the rule. Chosen: H0.50-p0-s1-15 (s2), source point H0.50-s1-15; meets the rule on 101-110: yes.

| Rank | Setting | Values (pitch 0) | Falls J1+J3 | Slow-gait (of 20) | Worst hind knee | J3 speed, m/s |
|---|---|---|---|---|---|---|
| 1 | H0.50-p0-s1-15 | period 0.638 s, duty 0.532, body pitch +0.00 deg, lift 23.1 mm, reach 81.9 mm | 0 | 0 | 0.663 | 0.2479 |
| 2 | H0.50-p0-s1-12 | period 0.644 s, duty 0.603, body pitch +0.00 deg, lift 17.0 mm, reach 55.6 mm | 0 | 0 | 0.666 | 0.2427 |
| 3 | H0.50-p0-s1-07 | period 0.600 s, duty 0.538, body pitch +0.00 deg, lift 18.3 mm, reach 59.3 mm | 0 | 0 | 0.675 | 0.2425 |
| 4 | H0.50-p0-s1-05 | period 0.672 s, duty 0.522, body pitch +0.00 deg, lift 26.7 mm, reach 71.4 mm | 0 | 0 | 0.699 | 0.2483 |
| 5 | H0.50-p0-s1-13 | period 0.703 s, duty 0.548, body pitch +0.00 deg, lift 21.9 mm, reach 53.0 mm | 0 | 0 | 0.708 | 0.2414 |
| 6 | H0.50-p0-s1-17 | period 0.711 s, duty 0.533, body pitch +0.00 deg, lift 19.5 mm, reach 68.9 mm | 0 | 0 | 0.750 | 0.2336 |
| 13 | H0.50-p0-s1-00 | period 0.639 s, duty 0.550, body pitch +0.00 deg, lift 29.2 mm, reach 66.7 mm | 0 | 0 | 0.886 | 0.2456 |

| Stage | Setting | Falls J1-J4 | J3 speed, m/s | Slow-gait total (max) | Worst hind knee | Meets rule |
|---|---|---|---|---|---|---|
| s2 | H0.50-p0-s1-15 | 0 | 0.2485 | 0 (0) | 0.663 | yes |
| s2 | H0.50-p0-s1-12 | 0 | 0.2415 | 0 (0) | 0.718 | yes |
| s2 | H0.50-p0-s1-07 | 0 | 0.2428 | 0 (0) | 0.677 | yes |
| s2 | H0.50-p0-s1-05 | 0 | 0.2482 | 0 (0) | 0.712 | yes |
| s2 | H0.50-p0-s1-13 | 0 | 0.2405 | 0 (0) | 0.708 | yes |
| s2 | H0.50-p0-s1-17 | 0 | 0.2355 | 0 (0) | 0.750 | yes |

## The frozen controller on the rerun seeds (supporting, not judged)

- 0.60 m, seeds 501-510: would fail (FAIL: falls 0, J3 0.2205 m/s, slow-gait max 10, knee 1.141); slow-gait per condition {'J1': 0, 'J2': 0, 'J3': 7, 'J4': 0, 'push_back_0.3': 10, 'push_back_0.4': 10, 'push_back_0.5': 10, 'push_right_0.3': 10, 'push_right_0.4': 10, 'push_right_0.5': 7}.
- 0.60 m, seeds 601-610: would fail (FAIL: falls 0, J3 0.2121 m/s, slow-gait max 10, knee 1.124); slow-gait per condition {'J1': 2, 'J2': 0, 'J3': 8, 'J4': 0, 'push_back_0.3': 9, 'push_back_0.4': 10, 'push_back_0.5': 10, 'push_right_0.3': 10, 'push_right_0.4': 10, 'push_right_0.5': 8}.
- 0.50 m, seeds 501-510: would pass (pass: falls 0, J3 0.2444 m/s, slow-gait max 0, knee 0.911); slow-gait per condition {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0, 'push_back_0.3': 0, 'push_back_0.4': 0, 'push_back_0.5': 0, 'push_right_0.3': 0, 'push_right_0.4': 0, 'push_right_0.5': 0}.
- 0.50 m, seeds 601-610: would pass (pass: falls 0, J3 0.2452 m/s, slow-gait max 0, knee 0.913); slow-gait per condition {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0, 'push_back_0.3': 0, 'push_back_0.4': 0, 'push_back_0.5': 0, 'push_right_0.3': 0, 'push_right_0.4': 0, 'push_right_0.5': 0}.

## Reported, not judged

- 0.60 m: X4 drive speed peak 80.5 rpm, 99th percentile 65.3 rpm over the four sets (83 rpm cap; reported). Condition-mean 10-20 s pitch -0.26 to +0.01 deg (+ nose down; the setpoint is 0). Episodes slow or hot over 2-8 s (speed under 0.225 m/s or hind knee >= 0.9; review E5-R2, context only): 3 of 400: J3 seed 302 (2-8 s: 0.2075 m/s, knee 0.991; 10-20 s 0.2568 m/s); J3 seed 401 (2-8 s: 0.2210 m/s, knee 0.886; 10-20 s 0.2588 m/s); J3 seed 507 (2-8 s: 0.2119 m/s, knee 0.943; 10-20 s 0.2084 m/s).
- 0.50 m: X4 drive speed peak 59.4 rpm, 99th percentile 49.7 rpm over the four sets (83 rpm cap; reported). Condition-mean 10-20 s pitch -0.22 to -0.00 deg (+ nose down; the setpoint is 0). Episodes slow or hot over 2-8 s (speed under 0.225 m/s or hind knee >= 0.9; review E5-R2, context only): 0 of 400.
- 0.60 m: "slow" episodes (10-20 s speed under 0.225 m/s; the slow-gait rule also needs the hind knee at or over 1.0, so these are not slow-gait episodes): 2 of 400: seeds 501-510 J3 seed 505 (10-20 s 0.2136 m/s, worst hind knee 0.906, slip ratio 0.200, pitch -0.31 deg); seeds 501-510 J3 seed 507 (10-20 s 0.2084 m/s, worst hind knee 0.961, slip ratio 0.242, pitch -0.33 deg).
- 0.50 m: "slow" episodes (10-20 s speed under 0.225 m/s; the slow-gait rule also needs the hind knee at or over 1.0, so these are not slow-gait episodes): 0 of 400.

## Harness checks

- 0.60 m: setting 00 (the frozen controller) reproduces E5's stored stage-1 episodes: 0 differing fields over 20 episodes.
- 0.50 m: setting 00 (the frozen controller) reproduces E5's stored stage-1 episodes: 0 differing fields over 20 episodes.
- 0.60 m: stage 2 reran its 120 shared stage-1 episodes: 0 mismatching fields.
- 0.50 m: stage 2 reran its 120 shared stage-1 episodes: 0 mismatching fields.
- 0.60 m: 1860 episode files, every one at body pitch 0 with an allowed seed: yes.
- 0.50 m: 1860 episode files, every one at body pitch 0 with an allowed seed: yes.
- Choice files unchanged since written: 0.60 yes (6a3dd65c7ee6f7c5), 0.50 yes (e395a948468b32d7); plan unchanged: yes (1eb4903d1d1b254a).
- Choice written before any held-out episode file existed: 0.60 yes, 0.50 yes.
- Wall time on 2 workers (s; episodes): 0.60 s1 1372 (660), 0.60 s2 1256 (600), 0.60 heldout 809 (400), 0.60 heldout_frozen 393 (200), 0.50 s1 1307 (660), 0.50 s2 1366 (600), 0.50 heldout 856 (400), 0.50 heldout_frozen 455 (200).

## Against E5's pitched choice (supporting)

- 0.60 m, seeds 301-310 / 401-410: worst hind knee 0.818 / 0.719 with H0.60-s1-24 (pitch +1.08 deg) against 0.829 / 0.759 with H0.60-p0-s1-24 (pitch 0); J3 speed 0.2530 / 0.2565 against 0.2556 / 0.2564 m/s.
- 0.50 m, seeds 301-310 / 401-410: worst hind knee 0.670 / 0.678 with H0.50-s1-15 (pitch +0.82 deg) against 0.680 / 0.709 with H0.50-p0-s1-15 (pitch 0); J3 speed 0.2485 / 0.2465 against 0.2491 / 0.2490 m/s.

