# E5 retuned trot: summary (SIMULATION)

28 Nov 2026. Pre-registration: `design/PREREG_CANDIDATES_2026_11_28.md` section 4 E5 (the lead's rule, quoted there, binds). Runner: the session scratch `candidates/E5/code/run_e5.py` (sha256 in every result file's `provenance`), calling only `sim/candidate_checks.py`'s frozen E5 code. MuJoCo 3.10.0, the build's env at 1 ms. Nothing measured; nothing adopted: any trot setting is the owner's decision.

## Outcome of the pre-registered rule

| Size | Chosen setting (tuning seeds 101-110 only) | Seeds 301-310 | Seeds 401-410 | Retune |
|---|---|---|---|---|
| 60 cm | H0.60-s1-24: period 0.845 s, duty 0.628, body pitch +1.08 deg, lift 25.0 mm, reach 87.0 mm | pass: falls J1-J4 0, J3 0.2530 m/s, slow-gait max 0 of 10, worst hind knee 0.818 | pass: falls J1-J4 0, J3 0.2565 m/s, slow-gait max 0 of 10, worst hind knee 0.719 | **SUCCEEDS** |
| 50 cm | H0.50-s1-15: period 0.638 s, duty 0.532, body pitch +0.82 deg, lift 23.1 mm, reach 81.9 mm | pass: falls J1-J4 0, J3 0.2485 m/s, slow-gait max 0 of 10, worst hind knee 0.670 | pass: falls J1-J4 0, J3 0.2465 m/s, slow-gait max 0 of 10, worst hind knee 0.678 | **SUCCEEDS** |

Rule (lead, 28 Nov 2026): on both held-out sets, no falls in J1-J4, J3 mean speed (2-20 s) >= 0.225 m/s, slow-gait episodes (`fore128_slow_gait.py`'s slow-and-hot rule, a fall counting as one) <= 1 of 10 in each of the 10 conditions, and the larger hind knee's RMS <= 1.0 of 6.75 N m (the worse of 2-20 s and 10-20 s) in every episode. One controller (the trot), as the rule is written. Hind knee: RMS over continuous; speeds m/s; pitch deg, + nose down.

## Harness checks (code, not hypotheses)

- Before stage 1, a smoke test on tuning seed 101; then, from the stored files: E5's frozen 60 cm setting reproduces E1's 16 frozen-trot episodes (J1, J3, push back 0.5, push right 0.5; seeds 101-104) with 0 differing record fields.
- 0.60 m: stage 2 reran the 120 episodes it shares with stage 1: 0 mismatching fields (the episodes are deterministic).
- 0.50 m: stage 2 reran the 120 episodes it shares with stage 1: 0 mismatching fields (the episodes are deterministic).
- **Post hoc** (decided after the 60 cm held-out result was seen, because the chosen setting's push episodes looked almost alike; it changes no result): four held-out push episodes rerun in a separate folder equal the stored ones, and the push was applied: 0.60 m push_back_0.5 seed 301: 100 steps, 6.3753 N s; 0.60 m push_right_0.5 seed 301: 100 steps, 6.3753 N s; 0.50 m push_back_0.5 seed 301: 100 steps, 5.7305 N s; 0.50 m push_right_0.5 seed 301: 100 steps, 5.7305 N s.
- Choice files unchanged since written: 0.60 21b623ce50d6fbfb at 2026-11-28T07:36:07Z, now 21b623ce50d6fbfb, 0.50 80223d678a9686a7 at 2026-11-28T08:34:16Z, now 80223d678a9686a7.
- Wall time on 2 workers (s; episodes): 0.60 s1 1352 (660), 0.60 s2 1238 (600), 0.60 heldout 413 (200), 0.60 heldout_frozen 406 (200), 0.50 s1 1330 (660), 0.50 s2 1304 (600), 0.50 heldout 428 (200), 0.50 heldout_frozen 428 (200).

## 60 cm, the adopted model

Tuning (seeds 101-110 only): 33 distinct settings, 39 setting runs over the stages (budget 60). Stage-2 determinism check: 0 mismatching fields over 120 episodes shared with stage 1.

Stage 1 (J1, J3, push back 0.5, push right 0.5; seeds 101-105; 20 episodes each), the six best and the frozen setting (score: falls in J1+J3, slow-gait episodes, worst hind knee, J3 speed m/s):

| Rank | Setting | Values | Falls J1+J3 | Slow-gait (of 20) | Worst hind knee | J3 speed, m/s |
|---|---|---|---|---|---|---|
| 1 | H0.60-s1-24 | period 0.845 s, duty 0.628, body pitch +1.08 deg, lift 25.0 mm, reach 87.0 mm | 0 | 0 | 0.623 | 0.2591 |
| 2 | H0.60-s1-13 | period 0.770 s, duty 0.548, body pitch +2.18 deg, lift 26.3 mm, reach 63.6 mm | 0 | 0 | 0.693 | 0.2527 |
| 3 | H0.60-s1-17 | period 0.779 s, duty 0.533, body pitch +2.94 deg, lift 23.4 mm, reach 82.7 mm | 0 | 0 | 0.696 | 0.2490 |
| 4 | H0.60-s1-12 | period 0.706 s, duty 0.603, body pitch +1.58 deg, lift 20.4 mm, reach 66.7 mm | 0 | 0 | 0.696 | 0.2504 |
| 5 | H0.60-s1-10 | period 0.627 s, duty 0.624, body pitch +2.71 deg, lift 33.7 mm, reach 91.4 mm | 0 | 0 | 0.722 | 0.2516 |
| 6 | H0.60-s1-02 | period 0.668 s, duty 0.596, body pitch -1.14 deg, lift 24.5 mm, reach 79.1 mm | 0 | 0 | 0.738 | 0.2553 |
| 31 | H0.60-s1-00 | period 0.700 s, duty 0.550, body pitch +0.00 deg, lift 35.0 mm, reach 80.0 mm | 0 | 15 | 1.141 | 0.2181 |

Stages 2 and 3 (all 10 conditions, seeds 101-110, 100 episodes each), the rule on the tuning seeds:

| Stage | Setting | Values | Falls J1-J4 | J3 speed, m/s | Slow-gait total (max per condition) | Worst hind knee | Failed checks | Meets rule |
|---|---|---|---|---|---|---|---|---|
| s2 | H0.60-s1-24 | period 0.845 s, duty 0.628, body pitch +1.08 deg, lift 25.0 mm, reach 87.0 mm | 0 | 0.2585 | 0 (0) | 0.642 | 0 | yes |
| s2 | H0.60-s1-13 | period 0.770 s, duty 0.548, body pitch +2.18 deg, lift 26.3 mm, reach 63.6 mm | 0 | 0.2527 | 0 (0) | 0.693 | 0 | yes |
| s2 | H0.60-s1-17 | period 0.779 s, duty 0.533, body pitch +2.94 deg, lift 23.4 mm, reach 82.7 mm | 0 | 0.2487 | 0 (0) | 0.696 | 0 | yes |
| s2 | H0.60-s1-12 | period 0.706 s, duty 0.603, body pitch +1.58 deg, lift 20.4 mm, reach 66.7 mm | 0 | 0.2507 | 0 (0) | 0.696 | 0 | yes |
| s2 | H0.60-s1-10 | period 0.627 s, duty 0.624, body pitch +2.71 deg, lift 33.7 mm, reach 91.4 mm | 0 | 0.2526 | 0 (0) | 0.765 | 0 | yes |
| s2 | H0.60-s1-02 | period 0.668 s, duty 0.596, body pitch -1.14 deg, lift 24.5 mm, reach 79.1 mm | 0 | 0.2552 | 0 (0) | 0.738 | 0 | yes |

**Chosen** (`e5_choose`, written to `e5_choice_0.60.json` before any held-out seed ran): H0.60-s1-24 (s2), period 0.845 s, duty 0.628, body pitch +1.08 deg, lift 25.0 mm, reach 87.0 mm; meets the rule on the tuning seeds: yes.

### Held-out, judged once: SUCCEEDS

Seeds 301-310: **pass**
- no falls in J1-J4: falls {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0} -> pass
- J3 mean speed 2-20 s >= 0.225 m/s: 0.2530 m/s -> pass
- slow-gait episodes <= 1 of 10 per condition: {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0, 'push_back_0.3': 0, 'push_back_0.4': 0, 'push_back_0.5': 0, 'push_right_0.3': 0, 'push_right_0.4': 0, 'push_right_0.5': 0} -> pass
- worst hind knee <= 1.0 of continuous in every episode: max per condition {'J1': 0.605, 'J2': 0.658, 'J3': 0.818, 'J4': 0.61, 'push_back_0.3': 0.593, 'push_back_0.4': 0.573, 'push_back_0.5': 0.58, 'push_right_0.3': 0.592, 'push_right_0.4': 0.604, 'push_right_0.5': 0.586} -> pass

Seeds 401-410: **pass**
- no falls in J1-J4: falls {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0} -> pass
- J3 mean speed 2-20 s >= 0.225 m/s: 0.2565 m/s -> pass
- slow-gait episodes <= 1 of 10 per condition: {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0, 'push_back_0.3': 0, 'push_back_0.4': 0, 'push_back_0.5': 0, 'push_right_0.3': 0, 'push_right_0.4': 0, 'push_right_0.5': 0} -> pass
- worst hind knee <= 1.0 of continuous in every episode: max per condition {'J1': 0.627, 'J2': 0.666, 'J3': 0.719, 'J4': 0.621, 'push_back_0.3': 0.582, 'push_back_0.4': 0.57, 'push_back_0.5': 0.576, 'push_right_0.3': 0.595, 'push_right_0.4': 0.6, 'push_right_0.5': 0.595} -> pass

Per condition (chosen setting; both held-out sets, 10 episodes each):

| Condition | Set | Falls | Slow-gait | Slow | Mean speed 2-20 s, m/s | Worst hind knee | Pitch 10-20 s, deg (+ nose down) | X4 peak / p99 rpm | X6 knee peak / p99 rpm | Slip ratio max |
|---|---|---|---|---|---|---|---|---|---|---|
| J1 | 301-310 | 0 | 0 | 0 | 0.2529 | 0.605 | 1.01 | 62.7 / 53.0 | 46.2 / 38.7 | 0.128 |
| J1 | 401-410 | 0 | 0 | 0 | 0.2524 | 0.627 | 1.01 | 61.5 / 53.7 | 45.4 / 39.8 | 0.140 |
| J2 | 301-310 | 0 | 0 | 0 | 0.2561 | 0.658 | 1.11 | 63.7 / 53.1 | 44.5 / 38.0 | 0.152 |
| J2 | 401-410 | 0 | 0 | 0 | 0.2560 | 0.666 | 1.12 | 62.7 / 53.7 | 44.8 / 38.9 | 0.160 |
| J3 | 301-310 | 0 | 0 | 0 | 0.2530 | 0.818 | 0.86 | 73.6 / 62.2 | 52.4 / 41.3 | 0.160 |
| J3 | 401-410 | 0 | 0 | 0 | 0.2565 | 0.719 | 0.86 | 75.1 / 58.9 | 49.2 / 40.8 | 0.136 |
| J4 | 301-310 | 0 | 0 | 0 | 0.2582 | 0.610 | 1.09 | 63.6 / 52.7 | 44.4 / 37.8 | 0.147 |
| J4 | 401-410 | 0 | 0 | 0 | 0.2580 | 0.621 | 1.10 | 61.7 / 53.2 | 44.7 / 38.6 | 0.155 |
| push_back_0.3 | 301-310 | 0 | 0 | 0 | 0.2540 | 0.593 | 1.03 | 61.9 / 53.2 | 47.1 / 38.3 | 0.123 |
| push_back_0.3 | 401-410 | 0 | 0 | 0 | 0.2541 | 0.582 | 1.03 | 60.2 / 52.8 | 46.1 / 38.2 | 0.124 |
| push_back_0.4 | 301-310 | 0 | 0 | 0 | 0.2540 | 0.573 | 1.03 | 61.9 / 52.9 | 47.1 / 38.1 | 0.124 |
| push_back_0.4 | 401-410 | 0 | 0 | 0 | 0.2541 | 0.570 | 1.03 | 60.2 / 53.0 | 46.1 / 38.2 | 0.124 |
| push_back_0.5 | 301-310 | 0 | 0 | 0 | 0.2540 | 0.580 | 1.03 | 61.9 / 52.9 | 47.1 / 38.2 | 0.125 |
| push_back_0.5 | 401-410 | 0 | 0 | 0 | 0.2541 | 0.576 | 1.03 | 60.2 / 52.9 | 46.1 / 38.3 | 0.127 |
| push_right_0.3 | 301-310 | 0 | 0 | 0 | 0.2540 | 0.592 | 1.02 | 61.9 / 52.8 | 47.1 / 38.2 | 0.126 |
| push_right_0.3 | 401-410 | 0 | 0 | 0 | 0.2541 | 0.595 | 1.03 | 60.2 / 53.0 | 52.7 / 38.4 | 0.126 |
| push_right_0.4 | 301-310 | 0 | 0 | 0 | 0.2541 | 0.604 | 1.03 | 61.9 / 53.6 | 47.1 / 38.4 | 0.129 |
| push_right_0.4 | 401-410 | 0 | 0 | 0 | 0.2540 | 0.600 | 1.02 | 60.2 / 53.1 | 53.8 / 38.3 | 0.124 |
| push_right_0.5 | 301-310 | 0 | 0 | 0 | 0.2537 | 0.586 | 1.03 | 61.9 / 53.3 | 47.1 / 38.2 | 0.125 |
| push_right_0.5 | 401-410 | 0 | 0 | 0 | 0.2535 | 0.595 | 1.03 | 60.2 / 52.9 | 54.8 / 38.4 | 0.125 |

Drive speeds, seeds 301-310 (reported, not judged): X4 peak 73.6 rpm, p99 62.2 rpm (83 rpm cap; peak line 91.3); X6 fore knees peak 52.4, p99 41.3 rpm (83 rpm; 108 rpm beside).
Drive speeds, seeds 401-410 (reported, not judged): X4 peak 75.1 rpm, p99 58.9 rpm (83 rpm cap; peak line 91.3); X6 fore knees peak 54.8, p99 40.8 rpm (83 rpm; 108 rpm beside).

### The frozen controller on the same held-out seeds (supporting, not judged)

Seeds 301-310: the rule would fail
- no falls in J1-J4: falls {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0} -> pass
- J3 mean speed 2-20 s >= 0.225 m/s: 0.2254 m/s -> pass
- slow-gait episodes <= 1 of 10 per condition: {'J1': 0, 'J2': 0, 'J3': 7, 'J4': 0, 'push_back_0.3': 9, 'push_back_0.4': 10, 'push_back_0.5': 10, 'push_right_0.3': 10, 'push_right_0.4': 10, 'push_right_0.5': 8} -> FAIL
- worst hind knee <= 1.0 of continuous in every episode: max per condition {'J1': 0.963, 'J2': 0.819, 'J3': 1.155, 'J4': 0.814, 'push_back_0.3': 1.047, 'push_back_0.4': 1.051, 'push_back_0.5': 1.057, 'push_right_0.3': 1.048, 'push_right_0.4': 1.059, 'push_right_0.5': 1.053} -> FAIL

Seeds 401-410: the rule would fail
- no falls in J1-J4: falls {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0} -> pass
- J3 mean speed 2-20 s >= 0.225 m/s: 0.2236 m/s -> FAIL
- slow-gait episodes <= 1 of 10 per condition: {'J1': 0, 'J2': 0, 'J3': 5, 'J4': 0, 'push_back_0.3': 9, 'push_back_0.4': 10, 'push_back_0.5': 10, 'push_right_0.3': 10, 'push_right_0.4': 10, 'push_right_0.5': 9} -> FAIL
- worst hind knee <= 1.0 of continuous in every episode: max per condition {'J1': 0.718, 'J2': 0.807, 'J3': 1.148, 'J4': 0.818, 'push_back_0.3': 1.044, 'push_back_0.4': 1.051, 'push_back_0.5': 1.061, 'push_right_0.3': 1.052, 'push_right_0.4': 1.053, 'push_right_0.5': 1.057} -> FAIL

## 50 cm, build_xml(0.50, 'D0', 'adopted', layout='L1')

Tuning (seeds 101-110 only): 33 distinct settings, 39 setting runs over the stages (budget 60). Stage-2 determinism check: 0 mismatching fields over 120 episodes shared with stage 1.

Stage 1 (J1, J3, push back 0.5, push right 0.5; seeds 101-105; 20 episodes each), the six best and the frozen setting (score: falls in J1+J3, slow-gait episodes, worst hind knee, J3 speed m/s):

| Rank | Setting | Values | Falls J1+J3 | Slow-gait (of 20) | Worst hind knee | J3 speed, m/s |
|---|---|---|---|---|---|---|
| 1 | H0.50-s1-15 | period 0.638 s, duty 0.532, body pitch +0.82 deg, lift 23.1 mm, reach 81.9 mm | 0 | 0 | 0.641 | 0.2467 |
| 2 | H0.50-s1-13 | period 0.703 s, duty 0.548, body pitch +2.18 deg, lift 21.9 mm, reach 53.0 mm | 0 | 0 | 0.649 | 0.2439 |
| 3 | H0.50-s1-07 | period 0.600 s, duty 0.538, body pitch -0.49 deg, lift 18.3 mm, reach 59.3 mm | 0 | 0 | 0.673 | 0.2437 |
| 4 | H0.50-s1-12 | period 0.644 s, duty 0.603, body pitch +1.58 deg, lift 17.0 mm, reach 55.6 mm | 0 | 0 | 0.709 | 0.2410 |
| 5 | H0.50-s1-17 | period 0.711 s, duty 0.533, body pitch +2.94 deg, lift 19.5 mm, reach 68.9 mm | 0 | 0 | 0.722 | 0.2350 |
| 6 | H0.50-s1-24 | period 0.772 s, duty 0.628, body pitch +1.08 deg, lift 20.8 mm, reach 72.5 mm | 0 | 0 | 0.740 | 0.2583 |
| 13 | H0.50-s1-00 | period 0.639 s, duty 0.550, body pitch +0.00 deg, lift 29.2 mm, reach 66.7 mm | 0 | 0 | 0.886 | 0.2456 |

Stages 2 and 3 (all 10 conditions, seeds 101-110, 100 episodes each), the rule on the tuning seeds:

| Stage | Setting | Values | Falls J1-J4 | J3 speed, m/s | Slow-gait total (max per condition) | Worst hind knee | Failed checks | Meets rule |
|---|---|---|---|---|---|---|---|---|
| s2 | H0.50-s1-15 | period 0.638 s, duty 0.532, body pitch +0.82 deg, lift 23.1 mm, reach 81.9 mm | 0 | 0.2472 | 0 (0) | 0.650 | 0 | yes |
| s2 | H0.50-s1-13 | period 0.703 s, duty 0.548, body pitch +2.18 deg, lift 21.9 mm, reach 53.0 mm | 0 | 0.2427 | 0 (0) | 0.669 | 0 | yes |
| s2 | H0.50-s1-07 | period 0.600 s, duty 0.538, body pitch -0.49 deg, lift 18.3 mm, reach 59.3 mm | 0 | 0.2438 | 0 (0) | 0.704 | 0 | yes |
| s2 | H0.50-s1-12 | period 0.644 s, duty 0.603, body pitch +1.58 deg, lift 17.0 mm, reach 55.6 mm | 0 | 0.2416 | 0 (0) | 0.709 | 0 | yes |
| s2 | H0.50-s1-17 | period 0.711 s, duty 0.533, body pitch +2.94 deg, lift 19.5 mm, reach 68.9 mm | 0 | 0.2345 | 0 (0) | 0.722 | 0 | yes |
| s2 | H0.50-s1-24 | period 0.772 s, duty 0.628, body pitch +1.08 deg, lift 20.8 mm, reach 72.5 mm | 0 | 0.2578 | 0 (0) | 0.741 | 0 | yes |

**Chosen** (`e5_choose`, written to `e5_choice_0.50.json` before any held-out seed ran): H0.50-s1-15 (s2), period 0.638 s, duty 0.532, body pitch +0.82 deg, lift 23.1 mm, reach 81.9 mm; meets the rule on the tuning seeds: yes.

### Held-out, judged once: SUCCEEDS

Seeds 301-310: **pass**
- no falls in J1-J4: falls {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0} -> pass
- J3 mean speed 2-20 s >= 0.225 m/s: 0.2485 m/s -> pass
- slow-gait episodes <= 1 of 10 per condition: {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0, 'push_back_0.3': 0, 'push_back_0.4': 0, 'push_back_0.5': 0, 'push_right_0.3': 0, 'push_right_0.4': 0, 'push_right_0.5': 0} -> pass
- worst hind knee <= 1.0 of continuous in every episode: max per condition {'J1': 0.655, 'J2': 0.67, 'J3': 0.648, 'J4': 0.651, 'push_back_0.3': 0.607, 'push_back_0.4': 0.603, 'push_back_0.5': 0.593, 'push_right_0.3': 0.593, 'push_right_0.4': 0.602, 'push_right_0.5': 0.605} -> pass

Seeds 401-410: **pass**
- no falls in J1-J4: falls {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0} -> pass
- J3 mean speed 2-20 s >= 0.225 m/s: 0.2465 m/s -> pass
- slow-gait episodes <= 1 of 10 per condition: {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0, 'push_back_0.3': 0, 'push_back_0.4': 0, 'push_back_0.5': 0, 'push_right_0.3': 0, 'push_right_0.4': 0, 'push_right_0.5': 0} -> pass
- worst hind knee <= 1.0 of continuous in every episode: max per condition {'J1': 0.67, 'J2': 0.678, 'J3': 0.666, 'J4': 0.673, 'push_back_0.3': 0.61, 'push_back_0.4': 0.596, 'push_back_0.5': 0.602, 'push_right_0.3': 0.605, 'push_right_0.4': 0.609, 'push_right_0.5': 0.603} -> pass

Per condition (chosen setting; both held-out sets, 10 episodes each):

| Condition | Set | Falls | Slow-gait | Slow | Mean speed 2-20 s, m/s | Worst hind knee | Pitch 10-20 s, deg (+ nose down) | X4 peak / p99 rpm | X6 knee peak / p99 rpm | Slip ratio max |
|---|---|---|---|---|---|---|---|---|---|---|
| J1 | 301-310 | 0 | 0 | 0 | 0.2472 | 0.655 | 0.68 | 57.1 / 48.8 | 42.9 / 38.6 | 0.116 |
| J1 | 401-410 | 0 | 0 | 0 | 0.2463 | 0.670 | 0.70 | 55.4 / 48.7 | 43.8 / 38.9 | 0.115 |
| J2 | 301-310 | 0 | 0 | 0 | 0.2539 | 0.670 | 0.82 | 56.6 / 50.2 | 43.1 / 39.0 | 0.142 |
| J2 | 401-410 | 0 | 0 | 0 | 0.2541 | 0.678 | 0.85 | 57.5 / 50.4 | 43.4 / 39.5 | 0.147 |
| J3 | 301-310 | 0 | 0 | 0 | 0.2485 | 0.648 | 0.69 | 58.5 / 48.7 | 46.3 / 39.0 | 0.110 |
| J3 | 401-410 | 0 | 0 | 0 | 0.2465 | 0.666 | 0.74 | 60.7 / 49.1 | 45.5 / 39.3 | 0.114 |
| J4 | 301-310 | 0 | 0 | 0 | 0.2553 | 0.651 | 0.80 | 56.3 / 50.4 | 42.7 / 38.9 | 0.139 |
| J4 | 401-410 | 0 | 0 | 0 | 0.2551 | 0.673 | 0.84 | 56.9 / 50.1 | 43.1 / 39.3 | 0.142 |
| push_back_0.3 | 301-310 | 0 | 0 | 0 | 0.2496 | 0.607 | 0.65 | 57.8 / 47.1 | 43.9 / 38.1 | 0.103 |
| push_back_0.3 | 401-410 | 0 | 0 | 0 | 0.2496 | 0.610 | 0.65 | 56.3 / 46.9 | 42.0 / 38.1 | 0.106 |
| push_back_0.4 | 301-310 | 0 | 0 | 0 | 0.2492 | 0.603 | 0.66 | 56.7 / 46.8 | 41.8 / 37.9 | 0.103 |
| push_back_0.4 | 401-410 | 0 | 0 | 0 | 0.2492 | 0.596 | 0.65 | 56.8 / 46.9 | 42.1 / 37.9 | 0.104 |
| push_back_0.5 | 301-310 | 0 | 0 | 0 | 0.2488 | 0.593 | 0.66 | 57.0 / 47.3 | 42.3 / 38.0 | 0.105 |
| push_back_0.5 | 401-410 | 0 | 0 | 0 | 0.2490 | 0.602 | 0.65 | 57.5 / 47.1 | 43.5 / 38.1 | 0.105 |
| push_right_0.3 | 301-310 | 0 | 0 | 0 | 0.2486 | 0.593 | 0.65 | 59.2 / 47.2 | 43.8 / 38.0 | 0.104 |
| push_right_0.3 | 401-410 | 0 | 0 | 0 | 0.2488 | 0.605 | 0.65 | 59.5 / 47.0 | 44.5 / 38.2 | 0.103 |
| push_right_0.4 | 301-310 | 0 | 0 | 0 | 0.2479 | 0.602 | 0.66 | 59.4 / 47.5 | 65.5 / 38.2 | 0.107 |
| push_right_0.4 | 401-410 | 0 | 0 | 0 | 0.2476 | 0.609 | 0.67 | 58.0 / 47.2 | 65.2 / 38.2 | 0.105 |
| push_right_0.5 | 301-310 | 0 | 0 | 0 | 0.2482 | 0.605 | 0.65 | 58.6 / 47.2 | 66.6 / 38.0 | 0.104 |
| push_right_0.5 | 401-410 | 0 | 0 | 0 | 0.2482 | 0.603 | 0.65 | 57.4 / 47.0 | 66.3 / 38.1 | 0.108 |

Drive speeds, seeds 301-310 (reported, not judged): X4 peak 59.4 rpm, p99 50.4 rpm (83 rpm cap; peak line 91.3); X6 fore knees peak 66.6, p99 39.0 rpm (83 rpm; 108 rpm beside).
Drive speeds, seeds 401-410 (reported, not judged): X4 peak 60.7 rpm, p99 50.4 rpm (83 rpm cap; peak line 91.3); X6 fore knees peak 66.3, p99 39.5 rpm (83 rpm; 108 rpm beside).

### The frozen controller on the same held-out seeds (supporting, not judged)

Seeds 301-310: the rule would fail
- no falls in J1-J4: falls {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0} -> pass
- J3 mean speed 2-20 s >= 0.225 m/s: 0.2437 m/s -> pass
- slow-gait episodes <= 1 of 10 per condition: {'J1': 0, 'J2': 0, 'J3': 1, 'J4': 0, 'push_back_0.3': 0, 'push_back_0.4': 0, 'push_back_0.5': 0, 'push_right_0.3': 0, 'push_right_0.4': 0, 'push_right_0.5': 0} -> pass
- worst hind knee <= 1.0 of continuous in every episode: max per condition {'J1': 0.89, 'J2': 0.894, 'J3': 1.019, 'J4': 0.887, 'push_back_0.3': 0.777, 'push_back_0.4': 0.772, 'push_back_0.5': 0.778, 'push_right_0.3': 0.783, 'push_right_0.4': 0.772, 'push_right_0.5': 0.781} -> FAIL

Seeds 401-410: the rule would pass
- no falls in J1-J4: falls {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0} -> pass
- J3 mean speed 2-20 s >= 0.225 m/s: 0.2457 m/s -> pass
- slow-gait episodes <= 1 of 10 per condition: {'J1': 0, 'J2': 0, 'J3': 0, 'J4': 0, 'push_back_0.3': 0, 'push_back_0.4': 0, 'push_back_0.5': 0, 'push_right_0.3': 0, 'push_right_0.4': 0, 'push_right_0.5': 0} -> pass
- worst hind knee <= 1.0 of continuous in every episode: max per condition {'J1': 0.907, 'J2': 0.909, 'J3': 0.891, 'J4': 0.89, 'push_back_0.3': 0.764, 'push_back_0.4': 0.766, 'push_back_0.5': 0.765, 'push_right_0.3': 0.784, 'push_right_0.4': 0.778, 'push_right_0.5': 0.764} -> pass

## Readings, deviations and what was decided when

- **Run as pre-registered.** Settings, boxes, seeds, conditions, the screen score, the rule and the choice are
  `candidate_checks.py`'s frozen code (sha256 `008aa308debdf1b6`, unchanged through every stage). The 33 stage-1
  settings per size were checked against `plan.json` before running. Stage 3 was not run at either size: all six
  stage-2 settings met the rule on seeds 101-110 (the pre-registration runs stage 3 only if none does).
- **Body pitch (prereg item 9, flagged for the lead).** No decision to hold body pitch at 0 was on record before the
  first E5 run, so `PitchedTrot` was used as pre-registered, and the lead's own rule lists body pitch among the tuned
  parameters. Both chosen settings use it (+1.08 deg at 60 cm, +0.82 deg at 50 cm, nose down). If the lead reads it as
  new controller structure, this result does not answer the pitch-0 question. Pitch-0 settings were not tuned.
- **Operational readings fixed in the runner before any E5 episode ran** (not "changed after a result"): R1 stage 3's
  centre would have been `e5_choose`'s first stage-2 setting; R2 stage 2 reruns its 20 shared episodes rather than
  reusing them, as a determinism check; R3 stage-1 ties broken by index, stage-3 ids from 00; R4 the supporting frozen
  run is `e5_default` unrounded (exactly `WalkParams()` at 60 cm and `scaled_walk_params(0.50 / 0.60)` at 50 cm;
  stage 1's setting 00 is the rounded copy); R5 drive speeds come from a `StepLogger` histogram (0.1 rpm bins, from
  2 s), and slip from pass-through wrappers that keep what `gait_metrics.measures` and the sweep record return; R6
  the 50 cm model is written once to the scratch folder, and its sha256 is checked against the build's `xml_sha256`.
- **Changed after a result:** no rule, reading, seed or setting. The only step decided after a result is the post-hoc
  push check above, which changes no figure. The extra files (`e5_harness_checks.json`, `e5_run_record.json`, this
  summary) are reporting.
- **Not recorded:** the trot's RMS on drives other than the hind knees, and C1 / C2 for the trot. E5's rule does not
  ask for them, and the per-episode files keep only the hind knee, drive speeds, slip and pitch.

## What it means (SIMULATION; nothing adopted)

- **Candidate A (60 cm).** This is the pre-registered outcome "SUCCEEDS at 60 cm". With setting H0.60-s1-24, A's
  scripted fallback avoids the slow gait in J1-J4 and after 0.3-0.5 m/s pushes back and to the right, on 20 held-out
  seeds. It is a candidate, not adopted. On the same seeds the frozen trot fell into the slow gait in 5-7 of 10 J3
  episodes and 8-10 of 10 after every push, with the worst hind knee per condition at 1.04-1.15 of rating in J3 and the pushes (supporting). The cost is
  speed at the fore hip pitch: the fastest X4 drive (the fore hip pitch) peaked at 73.6 / 75.1 rpm, p99 62.2 / 58.9
  rpm, against 62.4 / 63.3 rpm peak and 48.9 / 53.8 rpm p99 for the frozen trot, still under 83 rpm. DESK arithmetic
  for context only (no rule covers the trot here): at a 1.25 hip-pitch reduction, joint speed x r gives a 93.9 rpm
  peak. That is over E2's 91.3 rpm line, so this trot setting and a hip-pitch reduction pull against each other.
- **Candidate B (50 cm, today's drives, L1, adopted gains).** This is the pre-registered outcome "SUCCEEDS at 50
  cm". It removes one of the study's cautions for B, "a retune does not remove the trot's slow gait at 50 cm", for
  these conditions, with the adopted gains and L1. Supporting context: here the frozen trot was already close. It had
  1 slow-gait episode in J3 on seeds 301-310 and none on 401-410, and it failed the rule on 301-310 only by its hind
  knee (1.019 in J3). The retune's main effect at 50 cm is the hind-knee load: 0.59-0.68 against 0.76-1.02. The
  study's 50 cm slow-gait counts came from its S2 plant (friction 0.8, its seeds) and are not directly comparable.
- **What this does not show.** It shows one controller, the scripted trot, at 0.25 m/s, straight, on a flat floor,
  with the build's ideal frictionless servo, a 10 ms delay and 20 s episodes. It has no turning, starts or stops, no
  stops from E1 in place, and no hardware. The chosen 60 cm period (0.845 s) is 5 ms inside the pre-registered box's
  upper edge (0.85 s), so longer periods were not explored. Heat on the policies (E3) and the reduction question (E2)
  are not touched by this result.
