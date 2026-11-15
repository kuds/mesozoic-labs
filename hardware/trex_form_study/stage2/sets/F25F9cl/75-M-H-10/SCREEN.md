# Stage 0 screen of set F25F9cl on 75-M-H-10

Simulation only: every reference is priced as a prescribed motion that the joints follow exactly, on the round-two model. Made by `python -m designlab.stage2.setscreen md --set F25F9cl --body 75-M-H-10` from `screen.json`; the rules are in its `rules` block (STAGE2_75_PLAN.md A12).

A ratio is a load over its limit: cont = RMS torque over time / sustained torque (each frame weighted by its duration; on a walk, whose frames are uniform in time, that is the RMS over the frames), peak = 1.25 x peak torque / planning peak, speed = 1.15 x peak speed / speed at the bus. A ratio above 1 is over the limit. Ratios and margins are printed to four decimals; the verdict is read on the unrounded numbers in `screen.json`.

## Limits

| drive | actuator | sustained (N m) | planning peak (N m) | speed at the bus (rad/s) | bus (V) |
|---|---|---|---|---|---|
| hip roll | myactuator_rmd_x4_p36_36_e_l | 5.9850 | 21.0000 | 8.6917 | 24.0 |
| hip pitch | myactuator_rmd_x6_p20_60_e | 11.4000 | 42.0000 | 8.0111 | 24.0 |
| thigh twist | myactuator_rmd_x4_p36_36_e_l | 5.9850 | 21.0000 | 8.6917 | 24.0 |
| knee | myactuator_rmd_x4_p36_36_e_l | 5.9850 | 21.0000 | 8.6917 | 24.0 |
| ankle A | myactuator_rmd_x4_p36_36_e_l | 5.9850 | 21.0000 | 8.6917 | 24.0 |
| ankle B | myactuator_rmd_x4_p36_36_e_l | 5.9850 | 21.0000 | 8.6917 | 24.0 |
| neck | robotis_xm540_w270_t | 1.7702 | 3.1058 | 1.8850 | - |
| tail | feetech_sts3215_c018 | 0.4913 | 0.8620 | 2.8274 | - |

## Every reference

Columns: the governing ratio (largest of all drives); the worst ratio of the drives other than the neck's and the tail's (the legs'; on a body with the expressive head, its two drives as well); neck and tail ratios (cont / peak / speed); the single-support CoP margin (mm inside the pad; negative: outside) and the rule it is read against (mm); the single-support grip need mu_req and whether it is within mu 0.9 and mu 0.54; the mid-swing pad clearance (H; the gate is 0.04 H); the joint-range margin that the gate reads (rad; the gate is 0.10 rad for a walk and 0.05 rad for a start, change, stop or turn): the tightest gated joint, and for a turn the tightest joint other than the thigh twist; the thigh-twist use of the usable range (a turn's gate is 0.87; for a turn the twist's own tightest margin is beside it); the forward CoM acceleration low-passed at 1 Hz (m/s^2; the gate of a start, change or stop is 1.5 m/s^2), read on the file alone: a peak within 1 s of the file's first or last frame may be an end effect of the filter and is marked, with the largest value away from the ends beside it.

| reference | kind | frames | governing ratio | worst ratio but neck and tail | neck c / p / s | tail c / p / s | CoP margin (mm) | CoP rule (mm) | grip need mu_req | within mu 0.9 | within mu 0.54 | mid-swing clearance (H) | gated joint margin (rad) | thigh twist use | 1 Hz forward accel (m/s^2) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nominal` | walk | 160 | 0.7090 (thigh twist cont) | 0.7090 (thigh twist cont) | 0.6607 / 0.4947 / 0.2469 | 0.2186 / 0.3217 / 0.2059 | 45.0770 | -10 | 0.2452 | yes | yes | 0.0484 | 0.1189 (left_hip_roll max) | 0.0000 | 0.2111 | within the Stage 0 limits |
| `v0.3` | walk | 160 | 0.7040 (thigh twist cont) | 0.7040 (thigh twist cont) | 0.6617 / 0.5097 / 0.3047 | 0.2703 / 0.4020 / 0.2541 | 45.0262 | -10 | 0.3249 | yes | yes | 0.0484 | 0.1493 (left_hip_roll max) | 0.0000 | 0.1079 | within the Stage 0 limits |
| `v0.35` | walk | 160 | 0.7059 (thigh twist cont) | 0.7059 (thigh twist cont) | 0.6631 / 0.5262 / 0.3555 | 0.3186 / 0.4786 / 0.2965 | 45.0663 | -10 | 0.4074 | yes | yes | 0.0484 | 0.1717 (left_hip_roll max) | 0.0000 | 0.0598 | within the Stage 0 limits |
| `v0.4` | walk | 160 | 0.7426 (knee speed) | 0.7426 (knee speed) | 0.6643 / 0.5367 / 0.3855 | 0.3498 / 0.5264 / 0.3215 | 45.0314 | -10 | 0.4954 | yes | yes | 0.0484 | 0.1831 (left_hip_roll max) | 0.0000 | 0.0449 | within the Stage 0 limits |
| `v0.5` | walk | 160 | 0.9285 (knee speed) | 0.9285 (knee speed) | 0.6673 / 0.5589 / 0.4407 | 0.4158 / 0.6239 / 0.3675 | 44.9868 | -10 | 0.7008 | yes | no | 0.0484 | 0.2009 (left_hip_roll max) | 0.0000 | 0.0270 | within the Stage 0 limits |
| `v0.45` | walk | 160 | 0.8343 (knee speed) | 0.8343 (knee speed) | 0.6657 / 0.5475 / 0.4137 | 0.3820 / 0.5739 / 0.3450 | 44.9972 | -10 | 0.5931 | yes | no | 0.0484 | 0.1927 (left_hip_roll max) | 0.0000 | 0.0345 | within the Stage 0 limits |
| `start` | start | 800 | 0.6596 (neck cont) | 0.5302 (thigh twist cont) | 0.6596 / 0.5021 / 0.2469 | 0.1633 / 0.3245 / 0.2059 | 44.8120 | -10 | 0.2564 | yes | yes | 0.0484 | 0.0950 (left_hip_roll max) | 0.0000 | 0.4971 (at the file's end; 0.4743 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.4` | change | 960 | 0.7428 (knee speed) | 0.7428 (knee speed) | 0.6624 / 0.5367 / 0.3855 | 0.2869 / 0.5272 / 0.3215 | 44.1940 | -10 | 0.4974 | yes | yes | 0.0484 | 0.1189 (left_hip_roll max) | 0.0000 | 0.6515 (at the file's end; 0.3182 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.3` | change | 960 | 0.7068 (thigh twist cont) | 0.7068 (thigh twist cont) | 0.6613 / 0.5097 / 0.3047 | 0.2471 / 0.4025 / 0.2541 | 44.3721 | -10 | 0.3250 | yes | yes | 0.0484 | 0.1189 (left_hip_roll max) | 0.0000 | 0.6511 (at the file's end; 0.2373 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.35` | change | 960 | 0.7080 (thigh twist cont) | 0.7080 (thigh twist cont) | 0.6619 / 0.5262 / 0.3555 | 0.2717 / 0.4793 / 0.2965 | 44.2909 | -10 | 0.4083 | yes | yes | 0.0484 | 0.1189 (left_hip_roll max) | 0.0000 | 0.6519 (at the file's end; 0.2745 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.45` | change | 960 | 0.8346 (knee speed) | 0.8346 (knee speed) | 0.6629 / 0.5476 / 0.4137 | 0.3022 / 0.5748 / 0.3450 | 44.1231 | -10 | 0.5976 | yes | no | 0.0484 | 0.1189 (left_hip_roll max) | 0.0000 | 0.6520 (at the file's end; 0.3644 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.5` | change | 960 | 0.9290 (knee speed) | 0.9290 (knee speed) | 0.6634 / 0.5590 / 0.4407 | 0.3181 / 0.6251 / 0.3675 | 44.1071 | -10 | 0.7099 | yes | no | 0.0484 | 0.1189 (left_hip_roll max) | 0.0000 | 0.6515 (at the file's end; 0.4124 at least 1 s from both ends) | within the Stage 0 limits |
| `stop2_v0.4` | stop | 560 | 0.7644 (knee speed) | 0.7644 (knee speed) | 0.6604 / 0.5391 / 0.3855 | 0.1912 / 0.5481 / 0.3215 | 45.0455 | 10 | 0.5011 | yes | yes | 0.0484 | 0.1355 (left_hip_roll max) | 0.0000 | 0.6835 (at the file's end; 0.2990 at least 1 s from both ends) | within the Stage 0 limits |
| `stop1_v0.4` | stop | 800 | 0.7686 (knee speed) | 0.7686 (knee speed) | 0.6616 / 0.5391 / 0.3855 | 0.2495 / 0.5477 / 0.3215 | 44.6722 | -10 | 0.4953 | yes | yes | 0.0484 | 0.1350 (left_hip_roll max) | 0.0000 | 0.6549 | within the Stage 0 limits |
| `turn90_0.35_left` | turn | 1120 | 0.8040 (hip pitch speed) | 0.8040 (hip pitch speed) | 0.6653 / 0.5377 / 0.3855 | 0.3487 / 0.5562 / 0.3215 | 39.1604 | 10 | 0.5547 | yes | no | 0.0484 | 0.0861 (left_hip_roll max) | 0.5357 (margin 0.1393 rad, left_hip_yaw min) | 0.5999 (at the file's end; 0.0744 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.35_right` | turn | 1200 | 0.8041 (hip pitch speed) | 0.8041 (hip pitch speed) | 0.6653 / 0.5377 / 0.3855 | 0.3488 / 0.5562 / 0.3215 | 39.1616 | 10 | 0.5555 | yes | no | 0.0484 | 0.0862 (right_hip_roll min) | 0.5358 (margin 0.1393 rad, right_hip_yaw max) | 0.5998 (at the file's end; 0.0744 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.5_left` | turn | 960 | 0.8387 (hip pitch speed) | 0.8387 (hip pitch speed) | 0.6658 / 0.5382 / 0.3855 | 0.3640 / 0.6425 / 0.3215 | 34.2274 | 10 | 0.5889 | yes | no | 0.0484 | 0.0596 (left_hip_roll max) | 0.7452 (margin 0.0764 rad, right_hip_yaw min) | 0.6057 (at the file's end; 0.0865 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.5_right` | turn | 1040 | 0.8388 (hip pitch speed) | 0.8388 (hip pitch speed) | 0.6657 / 0.5382 / 0.3855 | 0.3629 / 0.6424 / 0.3215 | 34.2292 | 10 | 0.5885 | yes | no | 0.0484 | 0.0597 (right_hip_roll min) | 0.7451 (margin 0.0765 rad, left_hip_yaw max) | 0.6057 (at the file's end; 0.0864 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_pi4_left` | turn | 960 | 0.9449 (hip pitch speed) | 0.9449 (hip pitch speed) | 0.6668 / 0.5472 / 0.3855 | 0.4009 / 0.7900 / 0.3215 | 17.8944 | 10 | 0.8583 | yes | no | 0.0484 | 0.0777 (left_hip_roll max) | 0.8190 (margin 0.0543 rad, right_hip_yaw min) | 0.5882 (at the file's end; 0.0716 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_pi4_right` | turn | 1040 | 0.9449 (hip pitch speed) | 0.9449 (hip pitch speed) | 0.6665 / 0.5472 / 0.3855 | 0.3965 / 0.7899 / 0.3215 | 17.8974 | 10 | 0.8578 | yes | no | 0.0484 | 0.0778 (right_hip_roll min) | 0.8189 (margin 0.0543 rad, left_hip_yaw max) | 0.5882 (at the file's end; 0.0716 at least 1 s from both ends) | within the Stage 0 limits |

## What blocks a freeze

No screened reference is over a gate that blocks its freeze.

## Turns over a neck or tail limit (`a1_over`)

No screened turn is over a neck or tail planning limit.

## Reported, not gated

The frames of a move are not uniform in time where its stride period changes; the cont ratio by frames (every frame the same weight) is beside the gated one, which is over time.

| reference | frames uniform in time | neck cont over time / by frames | tail cont over time / by frames | neck with passive c / p | tail with passive c / p | double-support min mu | sway per stride (H; spec 0.12 H) | CoM horizontal accel peak (m/s^2) | near a limit (within 0.02 below 1) | record re-check equal |
|---|---|---|---|---|---|---|---|---|---|---|
| `nominal` | yes | 0.6607 / 0.6607 | 0.2186 / 0.2186 | 0.6719 / 0.5874 | 0.2920 / 0.3339 | 0.1938 | 0.2310 | 2.2116 | - | yes |
| `v0.3` | yes | 0.6617 / 0.6617 | 0.2703 / 0.2703 | 0.6754 / 0.5969 | 0.3013 / 0.3108 | 0.2090 | 0.1876 | 2.3464 | - | yes |
| `v0.35` | yes | 0.6631 / 0.6631 | 0.3186 / 0.3186 | 0.6778 / 0.6057 | 0.3267 / 0.3406 | 0.2202 | 0.1559 | 2.5171 | - | yes |
| `v0.4` | yes | 0.6643 / 0.6643 | 0.3498 / 0.3498 | 0.6803 / 0.6146 | 0.3513 / 0.3647 | 0.2305 | 0.1398 | 2.6772 | - | yes |
| `v0.5` | yes | 0.6673 / 0.6673 | 0.4158 / 0.4158 | 0.6848 / 0.6331 | 0.4137 / 0.4200 | 0.2475 | 0.1150 | 3.0218 | - | yes |
| `v0.45` | yes | 0.6657 / 0.6657 | 0.3820 / 0.3820 | 0.6824 / 0.6237 | 0.3805 / 0.3913 | 0.2454 | 0.1264 | 2.8460 | - | yes |
| `start` | yes | 0.6596 / 0.6596 | 0.1633 / 0.1633 | 0.6651 / 0.5873 | 0.2486 / 0.4269 | 0.1937 | 0.2562 | 2.1815 | - | yes |
| `nominal_to_v0.4` | no | 0.6624 / 0.6631 | 0.2869 / 0.3011 | 0.6767 / 0.6147 | 0.3206 / 0.3649 | 0.2325 | 0.2311 | 2.7092 | - | yes |
| `nominal_to_v0.3` | no | 0.6613 / 0.6614 | 0.2471 / 0.2500 | 0.6742 / 0.5968 | 0.2971 / 0.3352 | 0.2086 | 0.2310 | 2.3413 | - | yes |
| `nominal_to_v0.35` | no | 0.6619 / 0.6624 | 0.2717 / 0.2808 | 0.6757 / 0.6057 | 0.3093 / 0.3416 | 0.2203 | 0.2310 | 2.5130 | - | yes |
| `nominal_to_v0.45` | no | 0.6629 / 0.6639 | 0.3022 / 0.3224 | 0.6777 / 0.6238 | 0.3337 / 0.3915 | 0.2442 | 0.2311 | 2.9052 | - | yes |
| `nominal_to_v0.5` | no | 0.6634 / 0.6648 | 0.3181 / 0.3450 | 0.6787 / 0.6332 | 0.3485 / 0.4203 | 0.2559 | 0.2311 | 3.0968 | - | yes |
| `stop2_v0.4` | yes | 0.6604 / 0.6604 | 0.1912 / 0.1912 | 0.6635 / 0.6177 | 0.2309 / 0.6134 | 0.2258 | 0.1018 | 2.6560 | - | yes |
| `stop1_v0.4` | yes | 0.6616 / 0.6616 | 0.2495 / 0.2495 | 0.6689 / 0.6178 | 0.2730 / 0.6130 | 0.2304 | 0.1398 | 2.6725 | - | yes |
| `turn90_0.35_left` | yes | 0.6653 / 0.6653 | 0.3487 / 0.3487 | 0.6820 / 0.6163 | 0.3728 / 0.5519 | 0.2458 | 0.1483 | 2.8820 | - | yes |
| `turn90_0.35_right` | yes | 0.6653 / 0.6653 | 0.3488 / 0.3488 | 0.6821 / 0.6163 | 0.3714 / 0.5519 | 0.2459 | 0.1429 | 2.8818 | - | yes |
| `turn90_0.5_left` | no | 0.6658 / 0.6659 | 0.3640 / 0.3644 | 0.6820 / 0.6160 | 0.3734 / 0.5607 | 0.2593 | 0.1458 | 2.9809 | - | yes |
| `turn90_0.5_right` | no | 0.6657 / 0.6658 | 0.3629 / 0.3633 | 0.6823 / 0.6160 | 0.3716 / 0.5607 | 0.2589 | 0.1398 | 2.9806 | - | yes |
| `turn90_pi4_left` | no | 0.6668 / 0.6673 | 0.4009 / 0.4038 | 0.6820 / 0.6222 | 0.3427 / 0.5279 | 0.2645 | 0.1397 | 3.4463 | - | yes |
| `turn90_pi4_right` | no | 0.6665 / 0.6671 | 0.3965 / 0.3999 | 0.6821 / 0.6222 | 0.3433 / 0.5280 | 0.2631 | 0.1418 | 3.4457 | - | yes |

## Not available

- `turn90_1_left`: no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)
- `turn90_1_right`: no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)
