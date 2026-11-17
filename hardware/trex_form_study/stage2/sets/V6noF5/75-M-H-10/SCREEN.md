# Stage 0 screen of set V6noF5 on 75-M-H-10

Simulation only: every reference is priced as a prescribed motion that the joints follow exactly, on the round-two model. Made by `python -m designlab.stage2.setscreen md --set V6noF5 --body 75-M-H-10` from `screen.json`; the rules are in its `rules` block (STAGE2_75_PLAN.md A12).

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
| `nominal` | walk | 1440 | 0.8504 (knee speed) | 0.8504 (knee speed) | 0.6683 / 0.7493 / 0.2139 | 0.3575 / 0.5594 / 0.3437 | 46.0065 | -10 | 0.6550 | yes | no | 0.0421 | 0.1975 (left_hip_roll max) | 0.0103 | 0.0483 | within the Stage 0 limits |
| `v0.3` | walk | 1600 | 0.8457 (knee speed) | 0.8457 (knee speed) | 0.6700 / 0.7472 / 0.2143 | 0.2930 / 0.4946 / 0.3862 | 45.8180 | -10 | 0.6489 | yes | no | 0.0443 | 0.2103 (left_hip_roll max) | 0.0103 | 0.0424 | within the Stage 0 limits |
| `v0.35` | walk | 1920 | 0.8333 (knee speed) | 0.8333 (knee speed) | 0.6695 / 0.6946 / 0.2150 | 0.3744 / 0.6017 / 0.4429 | 45.5297 | -10 | 0.6167 | yes | no | 0.0466 | 0.2245 (left_hip_roll max) | 0.0107 | 0.0354 | within the Stage 0 limits |
| `v0.4` | walk | 1920 | 0.8317 (knee speed) | 0.8317 (knee speed) | 0.6732 / 0.7034 / 0.2155 | 0.3905 / 0.6273 / 0.4573 | 45.2745 | -10 | 0.6457 | yes | no | 0.0472 | 0.2280 (left_hip_roll max) | 0.0107 | 0.0369 | within the Stage 0 limits |
| `v0.45` | walk | 1920 | 0.8278 (knee speed) | 0.8278 (knee speed) | 0.6776 / 0.7360 / 0.2156 | 0.4156 / 0.6654 / 0.4720 | 45.2938 | -10 | 0.6729 | yes | no | 0.0469 | 0.2313 (left_hip_roll max) | 0.0107 | 0.0381 | within the Stage 0 limits |
| `v0.5` | walk | 2080 | 0.8281 (knee speed) | 0.8281 (knee speed) | 0.6829 / 0.7601 / 0.2157 | 0.4493 / 0.7139 / 0.4831 | 44.0653 | -10 | 0.7089 | yes | no | 0.0474 | 0.2348 (left_hip_roll max) | 0.0107 | 0.0388 | within the Stage 0 limits |
| `nominal_to_v0.4` | change | 960 | 0.9192 (knee speed) | 0.9192 (knee speed) | 0.6694 / 0.7569 / 0.2270 | 0.3610 / 0.5682 / 0.3711 | 44.4681 | -10 | 0.7288 | yes | no | 0.0440 | 0.1979 (left_hip_roll max) | 0.0106 | 0.1681 | within the Stage 0 limits |
| `stop2_v0.4` | stop | 560 | 0.8281 (knee speed) | 0.8281 (knee speed) | 0.6649 / 0.7101 / 0.2238 | 0.2274 / 0.5730 / 0.3638 | 43.5256 | 10 | 0.6497 | yes | no | 0.0490 | 0.1631 (left_hip_roll max) | 0.0107 | 0.6385 (at the file's end; 0.1826 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.3` | change | 960 | 0.8777 (knee speed) | 0.8777 (knee speed) | 0.6694 / 0.7507 / 0.2210 | 0.3157 / 0.5330 / 0.3180 | 45.4123 | -10 | 0.6767 | yes | no | 0.0440 | 0.1979 (left_hip_roll max) | 0.0104 | 0.1109 (at the file's end; 0.0772 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.35` | change | 960 | 0.9025 (knee speed) | 0.9025 (knee speed) | 0.6686 / 0.7509 / 0.2247 | 0.3541 / 0.5499 / 0.3593 | 45.0147 | -10 | 0.6958 | yes | no | 0.0440 | 0.1979 (left_hip_roll max) | 0.0105 | 0.1195 | within the Stage 0 limits |
| `nominal_to_v0.45` | change | 960 | 0.9356 (knee speed) | 0.9356 (knee speed) | 0.6705 / 0.7624 / 0.2276 | 0.3726 / 0.6009 / 0.3830 | 44.8498 | -10 | 0.7689 | yes | no | 0.0440 | 0.1979 (left_hip_roll max) | 0.0107 | 0.2206 | within the Stage 0 limits |
| `stop1_v0.4` | stop | 800 | 0.8302 (knee speed) | 0.8302 (knee speed) | 0.6673 / 0.7096 / 0.2173 | 0.2812 / 0.5844 / 0.3713 | 44.5338 | -10 | 0.6407 | yes | no | 0.0490 | 0.1628 (left_hip_roll max) | 0.0108 | 0.6221 | within the Stage 0 limits |
| `turn90_0.35_left` | turn | 1280 | 0.8545 (knee speed) | 0.8545 (knee speed) | 0.6744 / 0.7040 / 0.2270 | 0.3943 / 0.7523 / 0.3711 | 38.4173 | 10 | 0.7115 | yes | no | 0.0490 | 0.1564 (left_hip_roll max) | 0.4469 (margin 0.1659 rad, right_hip_yaw min) | 0.2489 (at the file's end; 0.0507 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.35_right` | turn | 1360 | 0.8545 (knee speed) | 0.8545 (knee speed) | 0.6743 / 0.7045 / 0.2270 | 0.3950 / 0.7521 / 0.3711 | 39.2746 | 10 | 0.7125 | yes | no | 0.0490 | 0.1606 (right_hip_roll min) | 0.4466 (margin 0.1660 rad, left_hip_yaw max) | 0.2446 (at the file's end; 0.0507 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.5_left` | turn | 1120 | 0.8742 (thigh twist peak) | 0.8742 (thigh twist peak) | 0.6750 / 0.7059 / 0.2270 | 0.3737 / 0.5875 / 0.3711 | 35.3733 | 10 | 0.7412 | yes | no | 0.0490 | 0.1271 (left_hip_roll max) | 0.6472 (margin 0.1058 rad, right_hip_yaw min) | 0.2606 (at the file's end; 0.0716 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.5_right` | turn | 1040 | 0.8750 (thigh twist peak) | 0.8750 (thigh twist peak) | 0.6751 / 0.7063 / 0.2270 | 0.3744 / 0.5877 / 0.3711 | 36.2832 | 10 | 0.7422 | yes | no | 0.0490 | 0.1311 (right_hip_roll min) | 0.6468 (margin 0.1060 rad, left_hip_yaw max) | 0.2557 (at the file's end; 0.0716 at least 1 s from both ends) | within the Stage 0 limits |

## What blocks a freeze

No screened reference is over a gate that blocks its freeze.

## Turns over a neck or tail limit (`a1_over`)

No screened turn is over a neck or tail planning limit.

## Reported, not gated

The frames of a move are not uniform in time where its stride period changes; the cont ratio by frames (every frame the same weight) is beside the gated one, which is over time.

| reference | frames uniform in time | neck cont over time / by frames | tail cont over time / by frames | neck with passive c / p | tail with passive c / p | double-support min mu | sway per stride (H; spec 0.12 H) | CoM horizontal accel peak (m/s^2) | near a limit (within 0.02 below 1) | record re-check equal |
|---|---|---|---|---|---|---|---|---|---|---|
| `nominal` | yes | 0.6683 / 0.6683 | 0.3575 / 0.3575 | 0.6727 / 0.8120 | 0.3134 / 0.4653 | 0.1962 | 0.1594 | 2.1923 | - | yes |
| `v0.3` | yes | 0.6700 / 0.6700 | 0.2930 / 0.2930 | 0.6723 / 0.7547 | 0.3465 / 0.4192 | 0.2510 | 0.1374 | 2.2934 | - | yes |
| `v0.35` | yes | 0.6695 / 0.6695 | 0.3744 / 0.3744 | 0.6713 / 0.7329 | 0.3763 / 0.5132 | 0.2531 | 0.1165 | 2.3767 | - | yes |
| `v0.4` | yes | 0.6732 / 0.6732 | 0.3905 / 0.3905 | 0.6742 / 0.7750 | 0.3953 / 0.5393 | 0.2708 | 0.1109 | 2.5164 | - | yes |
| `v0.45` | yes | 0.6776 / 0.6776 | 0.4156 / 0.4156 | 0.6782 / 0.8053 | 0.4118 / 0.5668 | 0.2784 | 0.1056 | 2.6872 | - | yes |
| `v0.5` | yes | 0.6829 / 0.6829 | 0.4493 / 0.4493 | 0.6814 / 0.8261 | 0.4359 / 0.5900 | 0.2837 | 0.0998 | 3.9136 | - | yes |
| `nominal_to_v0.4` | no | 0.6694 / 0.6697 | 0.3610 / 0.3633 | 0.6799 / 0.8065 | 0.3307 / 0.4341 | 0.2176 | 0.1602 | 2.6204 | - | yes |
| `stop2_v0.4` | no | 0.6649 / 0.6648 | 0.2274 / 0.2275 | 0.6548 / 0.7806 | 0.2513 / 0.7152 | 0.2125 | 0.0861 | 2.6546 | - | yes |
| `nominal_to_v0.3` | no | 0.6694 / 0.6696 | 0.3157 / 0.3142 | 0.6816 / 0.8068 | 0.3099 / 0.4198 | 0.2388 | 0.1602 | 2.3230 | - | yes |
| `nominal_to_v0.35` | no | 0.6686 / 0.6688 | 0.3541 / 0.3553 | 0.6801 / 0.8067 | 0.3221 / 0.4198 | 0.2039 | 0.1602 | 2.4669 | - | yes |
| `nominal_to_v0.45` | no | 0.6705 / 0.6710 | 0.3726 / 0.3768 | 0.6816 / 0.8064 | 0.3382 / 0.4537 | 0.2307 | 0.1602 | 2.7959 | - | yes |
| `stop1_v0.4` | no | 0.6673 / 0.6673 | 0.2812 / 0.2814 | 0.6779 / 0.7823 | 0.2963 / 0.7149 | 0.2135 | 0.1076 | 2.5710 | - | yes |
| `turn90_0.35_left` | no | 0.6744 / 0.6744 | 0.3943 / 0.3944 | 0.6720 / 0.7767 | 0.3954 / 0.5238 | 0.2457 | 0.1199 | 2.7594 | - | yes |
| `turn90_0.35_right` | no | 0.6743 / 0.6743 | 0.3950 / 0.3950 | 0.6722 / 0.7767 | 0.3962 / 0.5565 | 0.2458 | 0.1166 | 2.7564 | - | yes |
| `turn90_0.5_left` | no | 0.6750 / 0.6750 | 0.3737 / 0.3738 | 0.6718 / 0.7785 | 0.4065 / 0.5794 | 0.2349 | 0.1241 | 2.8976 | - | yes |
| `turn90_0.5_right` | no | 0.6751 / 0.6751 | 0.3744 / 0.3745 | 0.6714 / 0.7786 | 0.4093 / 0.6192 | 0.2356 | 0.1206 | 2.8928 | - | yes |

## Not available

- `start`: no transition within the gates
- `turn90_pi4_left`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit
- `turn90_pi4_right`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit
- `nominal_to_v0.5`: no transition within the gates
- `turn90_1_left`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit; also outside heading_at_slowest_clock
- `turn90_1_right`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit; also outside heading_at_slowest_clock
