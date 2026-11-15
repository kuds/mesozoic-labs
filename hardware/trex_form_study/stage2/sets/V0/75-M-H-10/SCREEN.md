# Stage 0 screen of set V0 on 75-M-H-10

Simulation only: every reference is priced as a prescribed motion that the joints follow exactly, on the round-two model. Made by `python -m designlab.stage2.setscreen md --set V0 --body 75-M-H-10` from `screen.json`; the rules are in its `rules` block (STAGE2_75_PLAN.md A12).

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
| `v0.3` | walk | 160 | 0.6639 (neck cont) | 0.6533 (knee speed) | 0.6639 / 0.5408 / 0.3813 | 0.4631 / 0.5989 / 0.3053 | 45.1220 | -10 | 0.4569 | yes | yes | 0.0540 | 0.1856 (left_hip_roll max) | 0.0000 | 0.0364 | within the Stage 0 limits |
| `nominal` | walk | 160 | 0.6622 (neck cont) | 0.6221 (thigh twist cont) | 0.6622 / 0.5256 / 0.3372 | 0.4009 / 0.5152 / 0.2700 | 45.3036 | -10 | 0.3612 | yes | yes | 0.0500 | 0.1689 (left_hip_roll max) | 0.0000 | 0.0531 | within the Stage 0 limits |
| `v0.35` | walk | 160 | 0.7403 (knee speed) | 0.7403 (knee speed) | 0.6664 / 0.5605 / 0.4308 | 0.5340 / 0.6963 / 0.3449 | 44.9433 | -10 | 0.5564 | yes | no | 0.0518 | 0.2015 (left_hip_roll max) | 0.0000 | 0.0231 | within the Stage 0 limits |
| `v0.4` | walk | 160 | 0.8042 (knee speed) | 0.8042 (knee speed) | 0.6673 / 0.5647 / 0.4450 | 0.5565 / 0.7280 / 0.3562 | 44.8662 | -10 | 0.6289 | yes | no | 0.0561 | 0.2054 (left_hip_roll max) | 0.0000 | 0.0222 | within the Stage 0 limits |
| `v0.45` | walk | 160 | 0.8708 (knee speed) | 0.8708 (knee speed) | 0.6683 / 0.5693 / 0.4592 | 0.5793 / 0.7610 / 0.3677 | 44.4494 | -10 | 0.7079 | yes | no | 0.0552 | 0.2092 (left_hip_roll max) | 0.0000 | 0.0212 | within the Stage 0 limits |
| `v0.5` | walk | 160 | 0.9412 (knee speed) | 0.9412 (knee speed) | 0.6695 / 0.5757 / 0.4769 | 0.6073 / 0.8007 / 0.3818 | 44.1502 | -10 | 0.7975 | yes | no | 0.0540 | 0.2137 (left_hip_roll max) | 0.0000 | 0.0194 | within the Stage 0 limits |
| `start` | start | 800 | 0.6608 (neck cont) | 0.5671 (knee speed) | 0.6608 / 0.5273 / 0.3372 | 0.3007 / 0.5153 / 0.2700 | 45.0669 | -10 | 0.3643 | yes | yes | 0.0500 | 0.1272 (left_hip_roll max) | 0.0000 | 0.3834 | within the Stage 0 limits |
| `nominal_to_v0.4` | change | 960 | 0.8093 (knee speed) | 0.8093 (knee speed) | 0.6645 / 0.5648 / 0.4450 | 0.4857 / 0.7287 / 0.3562 | 44.2782 | -10 | 0.6416 | yes | no | 0.0500 | 0.1689 (left_hip_roll max) | 0.0000 | 0.2023 | within the Stage 0 limits |
| `nominal_to_v0.3` | change | 960 | 0.6630 (neck cont) | 0.6550 (knee speed) | 0.6630 / 0.5408 / 0.3813 | 0.4359 / 0.5986 / 0.3053 | 44.5048 | -10 | 0.4595 | yes | yes | 0.0500 | 0.1689 (left_hip_roll max) | 0.0000 | 0.1753 (at the file's end; 0.1041 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.35` | change | 960 | 0.7435 (knee speed) | 0.7435 (knee speed) | 0.6643 / 0.5605 / 0.4308 | 0.4740 / 0.6973 / 0.3449 | 44.0759 | -10 | 0.5629 | yes | no | 0.0500 | 0.1689 (left_hip_roll max) | 0.0000 | 0.1544 | within the Stage 0 limits |
| `nominal_to_v0.45` | change | 960 | 0.8781 (knee speed) | 0.8781 (knee speed) | 0.6648 / 0.5693 / 0.4592 | 0.4975 / 0.7622 / 0.3677 | 43.3480 | -10 | 0.7292 | yes | no | 0.0500 | 0.1689 (left_hip_roll max) | 0.0000 | 0.2510 | within the Stage 0 limits |
| `nominal_to_v0.5` | change | 960 | 0.9502 (knee speed) | 0.9502 (knee speed) | 0.6652 / 0.5757 / 0.4769 | 0.5117 / 0.8015 / 0.3818 | 43.6217 | -10 | 0.8324 | yes | no | 0.0496 | 0.1689 (left_hip_roll max) | 0.0000 | 0.3015 | within the Stage 0 limits |
| `stop2_v0.4` | stop | 560 | 0.8319 (knee speed) | 0.8319 (knee speed) | 0.6606 / 0.5680 / 0.4450 | 0.2905 / 0.7507 / 0.3562 | 44.6138 | 10 | 0.6393 | yes | no | 0.0561 | 0.1503 (left_hip_roll max) | 0.0000 | 0.6463 (at the file's end; 0.1964 at least 1 s from both ends) | within the Stage 0 limits |
| `stop1_v0.4` | stop | 800 | 0.8385 (knee speed) | 0.8385 (knee speed) | 0.6627 / 0.5680 / 0.4450 | 0.3887 / 0.7340 / 0.3562 | 44.1998 | -10 | 0.6289 | yes | no | 0.0561 | 0.1495 (left_hip_roll max) | 0.0000 | 0.6278 | within the Stage 0 limits |
| `turn90_0.5_left` | turn | 1120 | 0.8483 (knee speed) | 0.8483 (knee speed) | 0.6689 / 0.5670 / 0.4450 | 0.5556 / 0.8275 / 0.3562 | 35.2170 | 10 | 0.7208 | yes | no | 0.0561 | 0.0933 (left_hip_roll max) | 0.6316 (margin 0.1105 rad, right_hip_yaw min) | 0.3048 (at the file's end; 0.0681 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.5_right` | turn | 1040 | 0.8484 (knee speed) | 0.8484 (knee speed) | 0.6690 / 0.5670 / 0.4450 | 0.5555 / 0.8274 / 0.3562 | 35.2187 | 10 | 0.7217 | yes | no | 0.0561 | 0.0934 (right_hip_roll min) | 0.6315 (margin 0.1105 rad, left_hip_yaw max) | 0.3048 (at the file's end; 0.0681 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.35_left` | turn | 1280 | 0.8330 (knee speed) | 0.8330 (knee speed) | 0.6683 / 0.5659 / 0.4450 | 0.5513 / 0.7629 / 0.3562 | 38.3399 | 10 | 0.6945 | yes | no | 0.0561 | 0.1244 (left_hip_roll max) | 0.4504 (margin 0.1649 rad, left_hip_yaw min) | 0.2969 (at the file's end; 0.0493 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.35_right` | turn | 1360 | 0.8331 (knee speed) | 0.8331 (knee speed) | 0.6682 / 0.5659 / 0.4450 | 0.5516 / 0.7629 / 0.3562 | 38.3412 | 10 | 0.6953 | yes | no | 0.0561 | 0.1245 (right_hip_roll min) | 0.4504 (margin 0.1649 rad, right_hip_yaw max) | 0.2969 (at the file's end; 0.0493 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_pi4_left` | turn | 960 | 1.0510 (tail peak) | 0.9229 (hip pitch speed) | 0.6700 / 0.5703 / 0.4450 | 0.6081 / 1.0510 / 0.3562 | 23.0606 | 10 | 0.9490 | no | no | 0.0561 | 0.0851 (left_hip_roll max) | 0.8154 (margin 0.0554 rad, right_hip_yaw min) | 0.3001 (at the file's end; 0.0729 at least 1 s from both ends) | a1_over: tail peak 1.0510 |
| `turn90_pi4_right` | turn | 880 | 1.0510 (tail peak) | 0.9229 (hip pitch speed) | 0.6702 / 0.5703 / 0.4450 | 0.6131 / 1.0510 / 0.3562 | 23.0621 | 10 | 0.9484 | no | no | 0.0561 | 0.0852 (right_hip_roll min) | 0.8153 (margin 0.0554 rad, left_hip_yaw max) | 0.3001 (at the file's end; 0.0729 at least 1 s from both ends) | a1_over: tail peak 1.0510 |

## What blocks a freeze

No screened reference is over a gate that blocks its freeze.

## Turns over a neck or tail limit (`a1_over`)

Frozen with `a1_over` and read under the plan's third reading only:

- `turn90_pi4_left`: tail peak 1.0510; neck c / p / s 0.6700 / 0.5703 / 0.4450; tail c / p / s 0.6081 / 1.0510 / 0.3562; thigh twist use 0.8154 (margin 0.0554 rad, right_hip_yaw min); gated joint margin 0.0851 (left_hip_roll max)
- `turn90_pi4_right`: tail peak 1.0510; neck c / p / s 0.6702 / 0.5703 / 0.4450; tail c / p / s 0.6131 / 1.0510 / 0.3562; thigh twist use 0.8153 (margin 0.0554 rad, left_hip_yaw max); gated joint margin 0.0852 (right_hip_roll min)

## Reported, not gated

The frames of a move are not uniform in time where its stride period changes; the cont ratio by frames (every frame the same weight) is beside the gated one, which is over time.

| reference | frames uniform in time | neck cont over time / by frames | tail cont over time / by frames | neck with passive c / p | tail with passive c / p | double-support min mu | sway per stride (H; spec 0.12 H) | CoM horizontal accel peak (m/s^2) | near a limit (within 0.02 below 1) | record re-check equal |
|---|---|---|---|---|---|---|---|---|---|---|
| `v0.3` | yes | 0.6639 / 0.6639 | 0.4631 / 0.4631 | 0.6789 / 0.6145 | 0.8728 / 0.8340 | 0.2042 | 0.1396 | 2.4127 | - | yes |
| `nominal` | yes | 0.6622 / 0.6622 | 0.4009 / 0.4009 | 0.6757 / 0.6025 | 0.8004 / 0.7446 | 0.1977 | 0.1641 | 2.1386 | - | yes |
| `v0.35` | yes | 0.6664 / 0.6664 | 0.5340 / 0.5340 | 0.6836 / 0.6344 | 0.9517 / 0.9394 | 0.1944 | 0.1168 | 2.6653 | - | yes |
| `v0.4` | yes | 0.6673 / 0.6673 | 0.5565 / 0.5565 | 0.6843 / 0.6393 | 0.9779 / 0.9731 | 0.2058 | 0.1112 | 2.8387 | - | yes |
| `v0.45` | yes | 0.6683 / 0.6683 | 0.5793 / 0.5793 | 0.6874 / 0.6449 | 1.0043 / 1.0074 | 0.2174 | 0.1059 | 3.0084 | - | yes |
| `v0.5` | yes | 0.6695 / 0.6695 | 0.6073 / 0.6073 | 0.6894 / 0.6527 | 1.0355 / 1.0484 | 0.2231 | 0.0997 | 3.1706 | - | yes |
| `start` | yes | 0.6608 / 0.6608 | 0.3007 / 0.3007 | 0.6677 / 0.6036 | 0.5903 / 0.7445 | 0.1982 | 0.1982 | 2.1618 | - | yes |
| `nominal_to_v0.4` | no | 0.6645 / 0.6652 | 0.4857 / 0.4942 | 0.6811 / 0.6393 | 0.8957 / 0.9734 | 0.2154 | 0.1642 | 2.8730 | - | yes |
| `nominal_to_v0.3` | no | 0.6630 / 0.6632 | 0.4359 / 0.4372 | 0.6784 / 0.6155 | 0.8417 / 0.8342 | 0.2090 | 0.1641 | 2.4218 | - | yes |
| `nominal_to_v0.35` | no | 0.6643 / 0.6648 | 0.4740 / 0.4803 | 0.6805 / 0.6343 | 0.8826 / 0.9397 | 0.2135 | 0.1642 | 2.7044 | - | yes |
| `nominal_to_v0.45` | no | 0.6648 / 0.6656 | 0.4975 / 0.5084 | 0.6817 / 0.6449 | 0.9086 / 1.0076 | 0.2422 | 0.1642 | 3.0807 | - | yes |
| `nominal_to_v0.5` | no | 0.6652 / 0.6663 | 0.5117 / 0.5260 | 0.6825 / 0.6527 | 0.9237 / 1.0491 | 0.2296 | 0.1642 | 3.2808 | - | yes |
| `stop2_v0.4` | yes | 0.6606 / 0.6606 | 0.2905 / 0.2905 | 0.6643 / 0.6430 | 0.4903 / 0.9996 | 0.2156 | 0.0854 | 2.8354 | tail peak with the plant's damping and friction 0.9996 | yes |
| `stop1_v0.4` | yes | 0.6627 / 0.6627 | 0.3887 / 0.3887 | 0.6709 / 0.6432 | 0.6735 / 0.9993 | 0.2170 | 0.1111 | 2.8449 | tail peak with the plant's damping and friction 0.9993 | yes |
| `turn90_0.5_left` | yes | 0.6689 / 0.6689 | 0.5556 / 0.5556 | 0.6871 / 0.6422 | 0.9639 / 1.0550 | 0.2296 | 0.1234 | 3.1625 | - | yes |
| `turn90_0.5_right` | yes | 0.6690 / 0.6690 | 0.5555 / 0.5555 | 0.6873 / 0.6422 | 0.9628 / 1.0549 | 0.2303 | 0.1218 | 3.1621 | - | yes |
| `turn90_0.35_left` | yes | 0.6683 / 0.6683 | 0.5513 / 0.5513 | 0.6866 / 0.6410 | 0.9727 / 1.0069 | 0.2247 | 0.1193 | 3.0729 | - | yes |
| `turn90_0.35_right` | yes | 0.6682 / 0.6682 | 0.5516 / 0.5516 | 0.6866 / 0.6410 | 0.9730 / 1.0068 | 0.2264 | 0.1181 | 3.0727 | - | yes |
| `turn90_pi4_left` | no | 0.6700 / 0.6706 | 0.6081 / 0.6095 | 0.6874 / 0.6446 | 0.9887 / 1.3387 | 0.2389 | 0.1112 | 3.5278 | tail cont with the plant's damping and friction 0.9887 | yes |
| `turn90_pi4_right` | no | 0.6702 / 0.6709 | 0.6131 / 0.6141 | 0.6875 / 0.6447 | 0.9891 / 1.3386 | 0.2391 | 0.1118 | 3.5273 | tail cont with the plant's damping and friction 0.9891 | yes |

Over a neck or tail planning limit once the plant's joint damping and friction are added (outside Stage 0's rule):

- `v0.45`: tail cont with the plant's damping and friction 1.0043; tail peak with the plant's damping and friction 1.0074
- `v0.5`: tail cont with the plant's damping and friction 1.0355; tail peak with the plant's damping and friction 1.0484
- `nominal_to_v0.45`: tail peak with the plant's damping and friction 1.0076
- `nominal_to_v0.5`: tail peak with the plant's damping and friction 1.0491
- `turn90_0.5_left`: tail peak with the plant's damping and friction 1.0550
- `turn90_0.5_right`: tail peak with the plant's damping and friction 1.0549
- `turn90_0.35_left`: tail peak with the plant's damping and friction 1.0069
- `turn90_0.35_right`: tail peak with the plant's damping and friction 1.0068
- `turn90_pi4_left`: tail peak with the plant's damping and friction 1.3387
- `turn90_pi4_right`: tail peak with the plant's damping and friction 1.3386

## Not available

- `turn90_1_left`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit
- `turn90_1_right`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit
