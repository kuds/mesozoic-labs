# Stage 0 screen of set V6 on 75-M-H-10

Simulation only: every reference is priced as a prescribed motion that the joints follow exactly, on the round-two model. Made by `python -m designlab.stage2.setscreen md --set V6 --body 75-M-H-10` from `screen.json`; the rules are in its `rules` block (STAGE2_75_PLAN.md A12).

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
| `nominal` | walk | 1440 | 0.7575 (knee speed) | 0.7575 (knee speed) | 0.6678 / 0.7496 / 0.2139 | 0.3579 / 0.5733 / 0.3437 | 45.9931 | -10 | 0.4054 | yes | yes | 0.0421 | 0.1976 (left_hip_roll max) | 0.0763 | 0.0543 | within the Stage 0 limits |
| `v0.3` | walk | 1600 | 0.7479 (neck peak) | 0.7414 (knee speed) | 0.6709 / 0.7479 / 0.2143 | 0.2929 / 0.5096 / 0.3862 | 45.7615 | -10 | 0.4570 | yes | yes | 0.0443 | 0.2104 (left_hip_roll max) | 0.0420 | 0.0454 | within the Stage 0 limits |
| `v0.35` | walk | 1920 | 0.7249 (neck peak) | 0.7145 (knee speed) | 0.6696 / 0.7249 / 0.2150 | 0.3744 / 0.6114 / 0.4429 | 45.3818 | -10 | 0.5352 | yes | yes | 0.0466 | 0.2247 (left_hip_roll max) | 0.0157 | 0.0361 | within the Stage 0 limits |
| `v0.4` | walk | 1920 | 0.7522 (knee speed) | 0.7522 (knee speed) | 0.6731 / 0.7253 / 0.2155 | 0.3905 / 0.6370 / 0.4573 | 45.1523 | -10 | 0.5846 | yes | no | 0.0472 | 0.2282 (left_hip_roll max) | 0.0123 | 0.0374 | within the Stage 0 limits |
| `v0.45` | walk | 1920 | 0.7864 (knee speed) | 0.7864 (knee speed) | 0.6774 / 0.7337 / 0.2156 | 0.4154 / 0.6756 / 0.4720 | 45.0913 | -10 | 0.6303 | yes | no | 0.0469 | 0.2316 (left_hip_roll max) | 0.0103 | 0.0383 | within the Stage 0 limits |
| `v0.5` | walk | 2080 | 0.8045 (knee speed) | 0.8045 (knee speed) | 0.6826 / 0.7590 / 0.2157 | 0.4488 / 0.7230 / 0.4831 | 43.9404 | -10 | 0.6819 | yes | no | 0.0474 | 0.2350 (left_hip_roll max) | 0.0103 | 0.0388 | within the Stage 0 limits |
| `nominal_to_v0.4` | change | 960 | 0.9188 (knee speed) | 0.9188 (knee speed) | 0.6691 / 0.7569 / 0.2270 | 0.3612 / 0.5770 / 0.3711 | 44.0642 | -10 | 0.7493 | yes | no | 0.0440 | 0.1980 (left_hip_roll max) | 0.0758 | 0.1666 | within the Stage 0 limits |
| `stop2_v0.4` | stop | 560 | 0.7582 (knee speed) | 0.7582 (knee speed) | 0.6648 / 0.7102 / 0.2238 | 0.2276 / 0.5732 / 0.3638 | 43.5284 | 10 | 0.5903 | yes | no | 0.0490 | 0.1631 (left_hip_roll max) | 0.0115 | 0.6425 (at the file's end; 0.1829 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.3` | change | 960 | 0.8775 (knee speed) | 0.8775 (knee speed) | 0.6697 / 0.7510 / 0.2210 | 0.3159 / 0.5486 / 0.3180 | 45.2169 | -10 | 0.6958 | yes | no | 0.0440 | 0.1980 (left_hip_roll max) | 0.0758 | 0.1401 (at the file's end; 0.0780 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.35` | change | 960 | 0.9022 (knee speed) | 0.9022 (knee speed) | 0.6684 / 0.7510 / 0.2247 | 0.3542 / 0.5612 / 0.3593 | 44.5695 | -10 | 0.7155 | yes | no | 0.0440 | 0.1980 (left_hip_roll max) | 0.0758 | 0.1272 (at the file's end; 0.1203 at least 1 s from both ends) | within the Stage 0 limits |
| `nominal_to_v0.45` | change | 960 | 0.9353 (knee speed) | 0.9353 (knee speed) | 0.6702 / 0.7625 / 0.2276 | 0.3727 / 0.6101 / 0.3830 | 44.4647 | -10 | 0.7838 | yes | no | 0.0440 | 0.1980 (left_hip_roll max) | 0.0758 | 0.2192 | within the Stage 0 limits |
| `stop1_v0.4` | stop | 800 | 0.7528 (knee speed) | 0.7528 (knee speed) | 0.6673 / 0.7303 / 0.2173 | 0.2813 / 0.5938 / 0.3713 | 44.1269 | -10 | 0.5819 | yes | no | 0.0490 | 0.1630 (left_hip_roll max) | 0.0161 | 0.6220 | within the Stage 0 limits |

## What blocks a freeze

No screened reference is over a gate that blocks its freeze.

## Turns over a neck or tail limit (`a1_over`)

No screened turn is over a neck or tail planning limit.

## Reported, not gated

The frames of a move are not uniform in time where its stride period changes; the cont ratio by frames (every frame the same weight) is beside the gated one, which is over time.

| reference | frames uniform in time | neck cont over time / by frames | tail cont over time / by frames | neck with passive c / p | tail with passive c / p | double-support min mu | sway per stride (H; spec 0.12 H) | CoM horizontal accel peak (m/s^2) | near a limit (within 0.02 below 1) | record re-check equal |
|---|---|---|---|---|---|---|---|---|---|---|
| `nominal` | yes | 0.6678 / 0.6678 | 0.3579 / 0.3579 | 0.6727 / 0.8123 | 0.3144 / 0.4644 | 0.1973 | 0.1597 | 2.2534 | - | yes |
| `v0.3` | yes | 0.6709 / 0.6709 | 0.2929 / 0.2929 | 0.6731 / 0.7682 | 0.3433 / 0.4060 | 0.2521 | 0.1376 | 2.3904 | - | yes |
| `v0.35` | yes | 0.6696 / 0.6696 | 0.3744 / 0.3744 | 0.6708 / 0.7248 | 0.3744 / 0.5109 | 0.2536 | 0.1165 | 2.4228 | - | yes |
| `v0.4` | yes | 0.6731 / 0.6731 | 0.3905 / 0.3905 | 0.6740 / 0.7723 | 0.3940 / 0.5370 | 0.2716 | 0.1109 | 2.5135 | - | yes |
| `v0.45` | yes | 0.6774 / 0.6774 | 0.4154 / 0.4154 | 0.6780 / 0.8028 | 0.4106 / 0.5651 | 0.2791 | 0.1056 | 2.6353 | - | yes |
| `v0.5` | yes | 0.6826 / 0.6826 | 0.4488 / 0.4488 | 0.6811 / 0.8252 | 0.4348 / 0.5890 | 0.2842 | 0.0998 | 3.8704 | - | yes |
| `nominal_to_v0.4` | no | 0.6691 / 0.6695 | 0.3612 / 0.3634 | 0.6798 / 0.8068 | 0.3302 / 0.4329 | 0.3727 | 0.1605 | 2.6203 | - | no |
| `stop2_v0.4` | no | 0.6648 / 0.6648 | 0.2276 / 0.2277 | 0.6547 / 0.7807 | 0.2510 / 0.7152 | 0.2124 | 0.0860 | 2.6991 | - | yes |
| `nominal_to_v0.3` | no | 0.6697 / 0.6699 | 0.3159 / 0.3143 | 0.6819 / 0.8071 | 0.3085 / 0.4186 | 0.5393 | 0.1605 | 2.3751 | - | no |
| `nominal_to_v0.35` | no | 0.6684 / 0.6686 | 0.3542 / 0.3554 | 0.6798 / 0.8070 | 0.3213 / 0.4186 | 0.4012 | 0.1605 | 2.4668 | - | no |
| `nominal_to_v0.45` | no | 0.6702 / 0.6707 | 0.3727 / 0.3769 | 0.6814 / 0.8067 | 0.3377 / 0.4525 | 0.3727 | 0.1605 | 2.7957 | - | no |
| `stop1_v0.4` | no | 0.6673 / 0.6672 | 0.2813 / 0.2814 | 0.6778 / 0.7795 | 0.2954 / 0.7151 | 0.3693 | 0.1076 | 2.6105 | - | no |

## Not available

- `start`: no transition within the gates
- `turn90_pi4_left`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit; also outside join_command
- `turn90_pi4_right`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit; also outside join_command
- `nominal_to_v0.5`: no transition within the gates
- `turn90_0.35_left`: no reference within the gates (entry gates: join_command)
- `turn90_0.35_right`: no reference within the gates (entry gates: join_command)
- `turn90_0.5_left`: no reference within the gates (entry gates: join_command)
- `turn90_0.5_right`: no reference within the gates (entry gates: join_command)
- `turn90_1_left`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit; also outside heading_at_slowest_clock
- `turn90_1_right`: no reference within the thigh twist and the leg drives: the built reference is over a leg drive's Stage 0 limit; also outside heading_at_slowest_clock
