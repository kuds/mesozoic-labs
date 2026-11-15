# Stage 0 screen of set F25F9 on 75-M-H-10

Simulation only: every reference is priced as a prescribed motion that the joints follow exactly, on the round-two model. Made by `python -m designlab.stage2.setscreen md --set F25F9 --body 75-M-H-10` from `screen.json`; the rules are in its `rules` block (STAGE2_75_PLAN.md A12).

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
| `v0.4` | walk | 160 | 0.7234 (thigh twist cont) | 0.7234 (thigh twist cont) | 0.6643 / 0.5356 / 0.3855 | 0.3772 / 0.5417 / 0.3215 | 43.7301 | -10 | 0.4131 | yes | yes | 0.0405 | 0.1833 (left_hip_roll max) | 0.0000 | 0.0485 | within the Stage 0 limits |
| `v0.5` | walk | 160 | 0.7571 (hip pitch speed) | 0.7571 (hip pitch speed) | 0.6670 / 0.5565 / 0.4407 | 0.4440 / 0.6311 / 0.3675 | 43.6278 | -10 | 0.5790 | yes | no | 0.0405 | 0.2009 (left_hip_roll max) | 0.0000 | 0.0294 | within the Stage 0 limits |
| `v0.45` | walk | 160 | 0.7287 (thigh twist cont) | 0.7287 (thigh twist cont) | 0.6656 / 0.5459 / 0.4137 | 0.4100 / 0.5858 / 0.3450 | 43.6767 | -10 | 0.4918 | yes | yes | 0.0405 | 0.1928 (left_hip_roll max) | 0.0000 | 0.0374 | within the Stage 0 limits |
| `start_v0.4` | start | 800 | 0.6612 (neck cont) | 0.6462 (hip pitch speed) | 0.6612 / 0.5389 / 0.3855 | 0.2781 / 0.5520 / 0.3215 | 43.2332 | -10 | 0.4222 | yes | yes | 0.0405 | 0.1361 (left_hip_roll max) | 0.0000 | 0.6187 | within the Stage 0 limits |
| `stop2_v0.4` | stop | 560 | 0.6606 (neck cont) | 0.6215 (knee speed) | 0.6606 / 0.5376 / 0.3855 | 0.1979 / 0.5647 / 0.3215 | 43.2136 | 10 | 0.4182 | yes | yes | 0.0405 | 0.1331 (left_hip_roll max) | 0.0000 | 0.7408 (at the file's end; 0.3274 at least 1 s from both ends) | within the Stage 0 limits |
| `stop1_v0.4` | stop | 800 | 0.6617 (neck cont) | 0.6247 (knee speed) | 0.6617 / 0.5375 / 0.3855 | 0.2646 / 0.5649 / 0.3215 | 43.2280 | -10 | 0.4131 | yes | yes | 0.0405 | 0.1327 (left_hip_roll max) | 0.0000 | 0.7018 (at the file's end; 0.6568 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.35_left` | turn | 1120 | 0.7168 (thigh twist cont) | 0.7168 (thigh twist cont) | 0.6653 / 0.5371 / 0.3855 | 0.3759 / 0.5724 / 0.3215 | 39.4639 | 10 | 0.4596 | yes | yes | 0.0405 | 0.0934 (left_hip_roll max) | 0.5096 (margin 0.1471 rad, left_hip_yaw min) | 0.7255 (at the file's end; 0.0786 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.35_right` | turn | 1200 | 0.7172 (thigh twist cont) | 0.7172 (thigh twist cont) | 0.6652 / 0.5371 / 0.3855 | 0.3760 / 0.5724 / 0.3215 | 39.4651 | 10 | 0.4602 | yes | yes | 0.0405 | 0.0935 (right_hip_roll min) | 0.5096 (margin 0.1471 rad, right_hip_yaw max) | 0.7255 (at the file's end; 0.0785 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.5_left` | turn | 960 | 0.7407 (tail peak) | 0.7275 (hip pitch speed) | 0.6658 / 0.5383 / 0.3855 | 0.3974 / 0.7407 / 0.3215 | 36.8399 | 10 | 0.4783 | yes | yes | 0.0405 | 0.0589 (left_hip_roll max) | 0.7396 (margin 0.0781 rad, right_hip_yaw min) | 0.7385 (at the file's end; 0.0994 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.5_right` | turn | 1040 | 0.7406 (tail peak) | 0.7275 (hip pitch speed) | 0.6657 / 0.5383 / 0.3855 | 0.3959 / 0.7406 / 0.3215 | 36.8417 | 10 | 0.4790 | yes | yes | 0.0405 | 0.0590 (right_hip_roll min) | 0.7395 (margin 0.0782 rad, left_hip_yaw max) | 0.7385 (at the file's end; 0.0994 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_pi4_left` | turn | 960 | 0.7915 (hip pitch speed) | 0.7915 (hip pitch speed) | 0.6665 / 0.5399 / 0.3855 | 0.3968 / 0.6948 / 0.3215 | 17.2036 | 10 | 0.7030 | yes | no | 0.0405 | 0.0769 (left_hip_roll max) | 0.8322 (margin 0.0503 rad, right_hip_yaw min) | 0.7109 (at the file's end; 0.0896 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_pi4_right` | turn | 880 | 0.7916 (hip pitch speed) | 0.7916 (hip pitch speed) | 0.6667 / 0.5399 / 0.3855 | 0.3988 / 0.6947 / 0.3215 | 17.2069 | 10 | 0.7025 | yes | no | 0.0405 | 0.0770 (right_hip_roll min) | 0.8321 (margin 0.0504 rad, left_hip_yaw max) | 0.7109 (at the file's end; 0.0896 at least 1 s from both ends) | within the Stage 0 limits |

## What blocks a freeze

No screened reference is over a gate that blocks its freeze.

## Turns over a neck or tail limit (`a1_over`)

No screened turn is over a neck or tail planning limit.

## Reported, not gated

The frames of a move are not uniform in time where its stride period changes; the cont ratio by frames (every frame the same weight) is beside the gated one, which is over time.

| reference | frames uniform in time | neck cont over time / by frames | tail cont over time / by frames | neck with passive c / p | tail with passive c / p | double-support min mu | sway per stride (H; spec 0.12 H) | CoM horizontal accel peak (m/s^2) | near a limit (within 0.02 below 1) | record re-check equal |
|---|---|---|---|---|---|---|---|---|---|---|
| `v0.4` | yes | 0.6643 / 0.6643 | 0.3772 / 0.3772 | 0.6792 / 0.6117 | 0.3492 / 0.3343 | 0.2557 | 0.1418 | 2.8543 | - | yes |
| `v0.5` | yes | 0.6670 / 0.6670 | 0.4440 / 0.4440 | 0.6832 / 0.6293 | 0.4155 / 0.3949 | 0.2724 | 0.1168 | 3.0686 | - | yes |
| `v0.45` | yes | 0.6656 / 0.6656 | 0.4100 / 0.4100 | 0.6810 / 0.6179 | 0.3805 / 0.3608 | 0.2678 | 0.1284 | 2.9633 | - | yes |
| `start_v0.4` | yes | 0.6612 / 0.6612 | 0.2781 / 0.2781 | 0.6689 / 0.6128 | 0.2886 / 0.4122 | 0.2556 | 0.1801 | 2.8017 | - | yes |
| `stop2_v0.4` | yes | 0.6606 / 0.6606 | 0.1979 / 0.1979 | 0.6630 / 0.6157 | 0.2331 / 0.5792 | 0.2518 | 0.0999 | 2.9645 | - | yes |
| `stop1_v0.4` | yes | 0.6617 / 0.6617 | 0.2646 / 0.2646 | 0.6682 / 0.6155 | 0.2735 / 0.5790 | 0.2545 | 0.1418 | 2.8854 | - | yes |
| `turn90_0.35_left` | yes | 0.6653 / 0.6653 | 0.3759 / 0.3759 | 0.6809 / 0.6132 | 0.3705 / 0.5176 | 0.2720 | 0.1498 | 3.0059 | - | yes |
| `turn90_0.35_right` | yes | 0.6652 / 0.6652 | 0.3760 / 0.3760 | 0.6810 / 0.6132 | 0.3691 / 0.5177 | 0.2726 | 0.1442 | 3.0060 | - | yes |
| `turn90_0.5_left` | yes | 0.6658 / 0.6658 | 0.3974 / 0.3974 | 0.6813 / 0.6145 | 0.3718 / 0.5670 | 0.2785 | 0.1539 | 3.0723 | - | yes |
| `turn90_0.5_right` | yes | 0.6657 / 0.6657 | 0.3959 / 0.3959 | 0.6814 / 0.6145 | 0.3700 / 0.5670 | 0.2783 | 0.1560 | 3.0724 | - | yes |
| `turn90_pi4_left` | no | 0.6665 / 0.6674 | 0.3968 / 0.4008 | 0.6811 / 0.6276 | 0.3624 / 0.8406 | 0.2799 | 0.1417 | 3.1574 | - | yes |
| `turn90_pi4_right` | no | 0.6667 / 0.6677 | 0.3988 / 0.4029 | 0.6812 / 0.6276 | 0.3638 / 0.8406 | 0.2786 | 0.1420 | 3.1573 | - | yes |

## Not available

- `v0.3`: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped from the ladder with its numbers
- `nominal`: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped from the ladder with its numbers
- `v0.35`: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped from the ladder with its numbers
- `nominal_to_v0.4`: the walk nominal is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped from the ladder with its numbers)
- `nominal_to_v0.3`: the walk nominal, v0.3 is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped from the ladder with its numbers; v0.3: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped from the ladder with its numbers)
- `nominal_to_v0.35`: the walk nominal, v0.35 is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped from the ladder with its numbers; v0.35: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped from the ladder with its numbers)
- `start`: the walk nominal is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped from the ladder with its numbers); the set starts from standing straight into v0.4 (start_v0.4; STAGE2_75_PLAN.md A8 as amended)
- `nominal_to_v0.45`: the walk nominal is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped from the ladder with its numbers); no change from the neighbouring rung within the gates
- `nominal_to_v0.5`: the walk nominal is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382: blocks the freeze (clearance swing_clearance); a walk that is not frozen is dropped from the ladder with its numbers); no change from the neighbouring rung within the gates
- `turn90_1_left`: no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)
- `turn90_1_right`: no reference within the thigh twist and the leg drives: no stride factor of the grid passes the kinematic rule (thigh twist at most 0.85 of its range, every other gated joint at least 0.05 rad of range, ik_ok, in both directions)
