# Stage 0 screen of set V6 on 75-M-H-10-X

Simulation only: every reference is priced as a prescribed motion that the joints follow exactly, on the round-two model. Made by `python -m designlab.stage2.setscreen md --set V6 --body 75-M-H-10-X` from `screen.json`; the rules are in its `rules` block (STAGE2_75_PLAN.md A12).

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
| `nominal` | walk | 1440 | 0.8020 (neck peak) | 0.7552 (knee speed) | 0.7179 / 0.8020 / 0.2139 | 0.3574 / 0.5748 / 0.3437 | 45.9923 | -10 | 0.4034 | yes | yes | 0.0421 | 0.1981 (left_hip_roll max) | 0.0767 | 0.0546 | within the Stage 0 limits |
| `v0.3` | walk | 1600 | 0.8006 (neck peak) | 0.7621 (knee speed) | 0.7212 / 0.8006 / 0.2143 | 0.2925 / 0.5109 / 0.3862 | 45.7599 | -10 | 0.4543 | yes | yes | 0.0443 | 0.2110 (left_hip_roll max) | 0.0420 | 0.0457 | within the Stage 0 limits |
| `v0.35` | walk | 1920 | 0.7767 (neck peak) | 0.7325 (knee speed) | 0.7198 / 0.7767 / 0.2150 | 0.3741 / 0.6119 / 0.4429 | 45.3803 | -10 | 0.5217 | yes | yes | 0.0466 | 0.2252 (left_hip_roll max) | 0.0157 | 0.0363 | within the Stage 0 limits |
| `v0.4` | walk | 1920 | 0.7780 (neck peak) | 0.7729 (knee speed) | 0.7236 / 0.7780 / 0.2155 | 0.3904 / 0.6368 / 0.4573 | 45.1510 | -10 | 0.5698 | yes | no | 0.0472 | 0.2288 (left_hip_roll max) | 0.0123 | 0.0376 | within the Stage 0 limits |
| `v0.45` | walk | 1920 | 0.8096 (knee speed) | 0.8096 (knee speed) | 0.7281 / 0.7841 / 0.2156 | 0.4155 / 0.6753 / 0.4720 | 45.0884 | -10 | 0.6139 | yes | no | 0.0469 | 0.2321 (left_hip_roll max) | 0.0107 | 0.0385 | within the Stage 0 limits |
| `v0.5` | walk | 2080 | 0.8293 (knee speed) | 0.8293 (knee speed) | 0.7337 / 0.8113 / 0.2157 | 0.4490 / 0.7219 / 0.4831 | 43.9361 | -10 | 0.6634 | yes | no | 0.0474 | 0.2355 (left_hip_roll max) | 0.0107 | 0.0391 | within the Stage 0 limits |
| `start` | start | 800 | 0.8052 (neck peak) | 0.7948 (knee speed) | 0.7138 / 0.8052 / 0.2172 | 0.2629 / 0.5300 / 0.2812 | 45.4687 | -10 | 0.5883 | yes | no | 0.0440 | 0.1320 (left_hip_roll max) | 0.0740 | 0.4223 | within the Stage 0 limits |
| `nominal_to_v0.4` | change | 960 | 0.9044 (knee speed) | 0.9044 (knee speed) | 0.7193 / 0.8098 / 0.2270 | 0.3609 / 0.5775 / 0.3711 | 44.0576 | -10 | 0.7236 | yes | no | 0.0440 | 0.1986 (left_hip_roll max) | 0.0762 | 0.1670 | within the Stage 0 limits |
| `nominal_to_v0.3` | change | 960 | 0.8653 (knee speed) | 0.8653 (knee speed) | 0.7199 / 0.8035 / 0.2210 | 0.3154 / 0.5502 / 0.3180 | 45.2123 | -10 | 0.6743 | yes | no | 0.0440 | 0.1986 (left_hip_roll max) | 0.0762 | 0.1408 (at the file's end; 0.0783 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.35` | change | 960 | 0.8894 (knee speed) | 0.8894 (knee speed) | 0.7186 / 0.8035 / 0.2247 | 0.3538 / 0.5621 / 0.3593 | 44.5628 | -10 | 0.6923 | yes | no | 0.0440 | 0.1986 (left_hip_roll max) | 0.0762 | 0.1282 (at the file's end; 0.1204 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.45` | change | 960 | 0.9322 (knee speed) | 0.9322 (knee speed) | 0.7205 / 0.8158 / 0.2276 | 0.3725 / 0.6095 / 0.3830 | 44.4588 | -10 | 0.7551 | yes | no | 0.0440 | 0.1986 (left_hip_roll max) | 0.0762 | 0.2194 | within the Stage 0 limits |
| `stop2_v0.4` | stop | 560 | 0.7809 (knee speed) | 0.7809 (knee speed) | 0.7149 / 0.7594 / 0.2238 | 0.2275 / 0.5739 / 0.3638 | 43.5387 | 10 | 0.5754 | yes | no | 0.0490 | 0.1636 (left_hip_roll max) | 0.0115 | 0.6421 (at the file's end; 0.1833 at least 1 s from both ends) | within the Stage 0 limits |
| `stop1_v0.4` | stop | 800 | 0.7837 (neck peak) | 0.7735 (knee speed) | 0.7175 / 0.7837 / 0.2173 | 0.2811 / 0.5950 / 0.3713 | 44.1210 | -10 | 0.5670 | yes | no | 0.0490 | 0.1634 (left_hip_roll max) | 0.0161 | 0.6211 | within the Stage 0 limits |
| `turn90_0.35_left` | turn | 1280 | 0.8444 (knee speed) | 0.8444 (knee speed) | 0.7249 / 0.7779 / 0.2270 | 0.3942 / 0.7535 / 0.3711 | 38.1718 | 10 | 0.6939 | yes | no | 0.0490 | 0.1560 (left_hip_roll max) | 0.4469 (margin 0.1659 rad, right_hip_yaw min) | 0.3001 (at the file's end; 0.0517 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.35_right` | turn | 1360 | 0.8445 (knee speed) | 0.8445 (knee speed) | 0.7249 / 0.7779 / 0.2270 | 0.3950 / 0.7537 / 0.3711 | 38.1629 | 10 | 0.6949 | yes | no | 0.0490 | 0.1603 (right_hip_roll min) | 0.4466 (margin 0.1660 rad, left_hip_yaw max) | 0.2956 (at the file's end; 0.0517 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.5_left` | turn | 1120 | 0.8685 (thigh twist peak) | 0.8685 (thigh twist peak) | 0.7255 / 0.7779 / 0.2270 | 0.3735 / 0.5888 / 0.3711 | 35.0772 | 10 | 0.7241 | yes | no | 0.0490 | 0.1264 (left_hip_roll max) | 0.6474 (margin 0.1058 rad, right_hip_yaw min) | 0.3117 (at the file's end; 0.0733 at least 1 s from both ends) | within the Stage 0 limits |
| `turn90_0.5_right` | turn | 1040 | 0.8692 (thigh twist peak) | 0.8692 (thigh twist peak) | 0.7257 / 0.7779 / 0.2270 | 0.3742 / 0.5888 / 0.3711 | 35.0137 | 10 | 0.7251 | yes | no | 0.0490 | 0.1305 (right_hip_roll min) | 0.6469 (margin 0.1059 rad, left_hip_yaw max) | 0.3060 (at the file's end; 0.0733 at least 1 s from both ends) | within the Stage 0 limits |

## What blocks a freeze

No screened reference is over a gate that blocks its freeze.

## Turns over a neck or tail limit (`a1_over`)

No screened turn is over a neck or tail planning limit.

## Reported, not gated

The frames of a move are not uniform in time where its stride period changes; the cont ratio by frames (every frame the same weight) is beside the gated one, which is over time.

| reference | frames uniform in time | neck cont over time / by frames | tail cont over time / by frames | neck with passive c / p | tail with passive c / p | double-support min mu | sway per stride (H; spec 0.12 H) | CoM horizontal accel peak (m/s^2) | near a limit (within 0.02 below 1) | record re-check equal |
|---|---|---|---|---|---|---|---|---|---|---|
| `nominal` | yes | 0.7179 / 0.7179 | 0.3574 / 0.3574 | 0.7223 / 0.8646 | 0.3152 / 0.4641 | 0.1970 | 0.1594 | 2.2616 | - | yes |
| `v0.3` | yes | 0.7212 / 0.7212 | 0.2925 / 0.2925 | 0.7229 / 0.8171 | 0.3442 / 0.4070 | 0.2513 | 0.1373 | 2.4036 | - | yes |
| `v0.35` | yes | 0.7198 / 0.7198 | 0.3741 / 0.3741 | 0.7206 / 0.7701 | 0.3756 / 0.5126 | 0.2523 | 0.1162 | 2.4223 | - | yes |
| `v0.4` | yes | 0.7236 / 0.7236 | 0.3904 / 0.3904 | 0.7240 / 0.8207 | 0.3953 / 0.5388 | 0.2696 | 0.1107 | 2.5134 | - | yes |
| `v0.45` | yes | 0.7281 / 0.7281 | 0.4155 / 0.4155 | 0.7283 / 0.8532 | 0.4120 / 0.5672 | 0.2767 | 0.1054 | 2.6173 | - | yes |
| `v0.5` | yes | 0.7337 / 0.7337 | 0.4490 / 0.4490 | 0.7317 / 0.8775 | 0.4363 / 0.5913 | 0.2819 | 0.0995 | 3.8419 | - | yes |
| `start` | no | 0.7138 / 0.7138 | 0.2629 / 0.2633 | 0.7197 / 0.8659 | 0.2426 / 0.4539 | 0.3601 | 0.1897 | 2.2256 | - | no |
| `nominal_to_v0.4` | no | 0.7193 / 0.7197 | 0.3609 / 0.3631 | 0.7296 / 0.8588 | 0.3313 / 0.4346 | 0.3655 | 0.1602 | 2.6244 | - | no |
| `nominal_to_v0.3` | no | 0.7199 / 0.7201 | 0.3154 / 0.3139 | 0.7318 / 0.8591 | 0.3094 / 0.4183 | 0.5301 | 0.1602 | 2.3849 | - | no |
| `nominal_to_v0.35` | no | 0.7186 / 0.7188 | 0.3538 / 0.3550 | 0.7296 / 0.8590 | 0.3223 / 0.4183 | 0.3922 | 0.1602 | 2.4690 | - | no |
| `nominal_to_v0.45` | no | 0.7205 / 0.7210 | 0.3725 / 0.3767 | 0.7313 / 0.8586 | 0.3388 / 0.4544 | 0.3655 | 0.1602 | 2.7992 | - | no |
| `stop2_v0.4` | no | 0.7149 / 0.7148 | 0.2275 / 0.2276 | 0.7046 / 0.8299 | 0.2514 / 0.7150 | 0.2103 | 0.0859 | 2.6986 | - | yes |
| `stop1_v0.4` | no | 0.7175 / 0.7174 | 0.2811 / 0.2813 | 0.7278 / 0.8287 | 0.2962 / 0.7149 | 0.3622 | 0.1073 | 2.6149 | - | no |
| `turn90_0.35_left` | no | 0.7249 / 0.7249 | 0.3942 / 0.3943 | 0.7222 / 0.8255 | 0.3962 / 0.5208 | 0.3541 | 0.1194 | 2.7988 | - | no |
| `turn90_0.35_right` | no | 0.7249 / 0.7248 | 0.3950 / 0.3950 | 0.7224 / 0.8255 | 0.3971 / 0.5532 | 0.3554 | 0.1166 | 2.7912 | - | no |
| `turn90_0.5_left` | no | 0.7255 / 0.7255 | 0.3735 / 0.3736 | 0.7220 / 0.8273 | 0.4073 / 0.5758 | 0.3541 | 0.1236 | 2.9398 | - | no |
| `turn90_0.5_right` | no | 0.7257 / 0.7256 | 0.3742 / 0.3743 | 0.7216 / 0.8273 | 0.4102 / 0.6161 | 0.3546 | 0.1204 | 2.9257 | - | no |

## Not available

- `inplace_0.5_left`: the turn in place has no reference within the gates (inplace_0.5_left: Stage 0: over thigh twist use above 0.87 0.9215: blocks the freeze (thigh twist use above 0.87); start_inplace_0.5_left: Stage 0: over knee speed 1.1089; joint range margin under 0.05 rad (left_hip_yaw max) 0.0235: blocks the freeze (knee speed; joint range margin under 0.05 rad (left_hip_yaw max)))
- `inplace_0.5_right`: the turn in place has no reference within the gates (inplace_0.5_right: Stage 0: over thigh twist use above 0.87 0.9216: blocks the freeze (thigh twist use above 0.87); start_inplace_0.5_right: Stage 0: over knee speed 1.1705; joint range margin under 0.05 rad (right_hip_yaw min) 0.0235: blocks the freeze (knee speed; joint range margin under 0.05 rad (right_hip_yaw min)))
- `nominal_to_v0.5`: no transition within the gates
- `turn90_pi4_left`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit
- `turn90_pi4_right`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit
- `turn90_1_left`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit; also outside heading_at_slowest_clock, post_window_zmp
- `turn90_1_right`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit; also outside heading_at_slowest_clock
- `start_inplace_0.5_left`: the turn in place has no reference within the gates (inplace_0.5_left: Stage 0: over thigh twist use above 0.87 0.9215: blocks the freeze (thigh twist use above 0.87); start_inplace_0.5_left: Stage 0: over knee speed 1.1089; joint range margin under 0.05 rad (left_hip_yaw max) 0.0235: blocks the freeze (knee speed; joint range margin under 0.05 rad (left_hip_yaw max)))
- `stop_inplace_0.5_left`: the turn in place has no reference within the gates (inplace_0.5_left: Stage 0: over thigh twist use above 0.87 0.9215: blocks the freeze (thigh twist use above 0.87); start_inplace_0.5_left: Stage 0: over knee speed 1.1089; joint range margin under 0.05 rad (left_hip_yaw max) 0.0235: blocks the freeze (knee speed; joint range margin under 0.05 rad (left_hip_yaw max)))
- `start_inplace_0.5_right`: the turn in place has no reference within the gates (inplace_0.5_right: Stage 0: over thigh twist use above 0.87 0.9216: blocks the freeze (thigh twist use above 0.87); start_inplace_0.5_right: Stage 0: over knee speed 1.1705; joint range margin under 0.05 rad (right_hip_yaw min) 0.0235: blocks the freeze (knee speed; joint range margin under 0.05 rad (right_hip_yaw min)))
- `stop_inplace_0.5_right`: the turn in place has no reference within the gates (inplace_0.5_right: Stage 0: over thigh twist use above 0.87 0.9216: blocks the freeze (thigh twist use above 0.87); start_inplace_0.5_right: Stage 0: over knee speed 1.1705; joint range margin under 0.05 rad (right_hip_yaw min) 0.0235: blocks the freeze (knee speed; joint range margin under 0.05 rad (right_hip_yaw min)))
