# Stage 0 screen of set V6 on 60-N-T-10-X

Simulation only: every reference is priced as a prescribed motion that the joints follow exactly, on the round-two model. Made by `python -m designlab.stage2.setscreen md --set V6 --body 60-N-T-10-X` from `screen.json`; the rules are in its `rules` block (STAGE2_75_PLAN.md A12).

A ratio is a load over its limit: cont = RMS torque over time / sustained torque (each frame weighted by its duration; on a walk, whose frames are uniform in time, that is the RMS over the frames), peak = 1.25 x peak torque / planning peak, speed = 1.15 x peak speed / speed at the bus. A ratio above 1 is over the limit. Ratios and margins are printed to four decimals; the verdict is read on the unrounded numbers in `screen.json`.

## Limits

| drive | actuator | sustained (N m) | planning peak (N m) | speed at the bus (rad/s) | bus (V) |
|---|---|---|---|---|---|
| hip roll | cubemars_ak45_36_v3 | 4.5600 | 16.8000 | 4.1888 | 24.0 |
| hip pitch | myactuator_rmd_x4_p36_36_e_l | 5.9850 | 21.0000 | 8.6917 | 24.0 |
| thigh twist | cubemars_ak45_36_v3 | 4.5600 | 16.8000 | 4.1888 | 24.0 |
| knee | myactuator_rmd_x4_p36_36_e_l | 5.9850 | 21.0000 | 8.6917 | 24.0 |
| ankle A | myactuator_rmd_x4_p12_10_e | 2.2800 | 7.0000 | 24.9233 | 24.0 |
| ankle B | myactuator_rmd_x4_p12_10_e | 2.2800 | 7.0000 | 24.9233 | 24.0 |
| neck | feetech_sts3250 | 0.8188 | 1.4366 | 4.7124 | - |
| tail | feetech_sts3215_c018 | 0.4913 | 0.8620 | 2.8274 | - |

## Every reference

Columns: the governing ratio (largest of all drives); the worst ratio of the drives other than the neck's and the tail's (the legs'; on a body with the expressive head, its two drives as well); neck and tail ratios (cont / peak / speed); the single-support CoP margin (mm inside the pad; negative: outside) and the rule it is read against (mm); the single-support grip need mu_req and whether it is within mu 0.9 and mu 0.54; the mid-swing pad clearance (H; the gate is 0.04 H); the joint-range margin that the gate reads (rad; the gate is 0.10 rad for a walk and 0.05 rad for a start, change, stop or turn): the tightest gated joint, and for a turn the tightest joint other than the thigh twist; the thigh-twist use of the usable range (a turn's gate is 0.87; for a turn the twist's own tightest margin is beside it); the forward CoM acceleration low-passed at 1 Hz (m/s^2; the gate of a start, change or stop is 1.5 m/s^2), read on the file alone: a peak within 1 s of the file's first or last frame may be an end effect of the filter and is marked, with the largest value away from the ends beside it.

| reference | kind | frames | governing ratio | worst ratio but neck and tail | neck c / p / s | tail c / p / s | CoP margin (mm) | CoP rule (mm) | grip need mu_req | within mu 0.9 | within mu 0.54 | mid-swing clearance (H) | gated joint margin (rad) | thigh twist use | 1 Hz forward accel (m/s^2) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nominal` | walk | 1600 | 0.8771 (neck peak) | 0.7443 (knee speed) | 0.8544 / 0.8771 / 0.0853 | 0.1444 / 0.2582 / 0.3805 | 31.0894 | -10 | 0.5526 | yes | no | 0.0414 | 0.2096 (left_hip_roll max) | 0.0503 | 0.0377 | within the Stage 0 limits |
| `v0.3` | walk | 2080 | 0.8555 (neck cont) | 0.7757 (knee speed) | 0.8555 / 0.8387 / 0.0862 | 0.2079 / 0.3456 / 0.4668 | 30.7454 | -10 | 0.7004 | yes | no | 0.0465 | 0.2355 (left_hip_roll max) | 0.0123 | 0.0282 | within the Stage 0 limits |
| `v0.35` | walk | 2080 | 0.8837 (neck peak) | 0.8465 (knee speed) | 0.8603 / 0.8837 / 0.0863 | 0.2337 / 0.3902 / 0.4988 | 30.5427 | -10 | 0.7796 | yes | no | 0.0475 | 0.2438 (left_hip_roll max) | 0.0103 | 0.0272 | within the Stage 0 limits |
| `v0.4` | walk | 2240 | 0.9228 (neck peak) | 0.8668 (knee speed) | 0.8658 / 0.9228 / 0.0864 | 0.2492 / 0.4169 / 0.5165 | 30.3897 | -10 | 0.8374 | yes | no | 0.0473 | 0.2478 (left_hip_roll max) | 0.0103 | 0.0287 | within the Stage 0 limits |
| `v0.45` | walk | 2240 | 0.9463 (neck peak) | 0.8949 (knee speed) | 0.8726 / 0.9463 / 0.0866 | 0.2721 / 0.4554 / 0.5391 | 29.5225 | -10 | 0.9031 | no | no | 0.0473 | 0.2527 (left_hip_roll max) | 0.0103 | 0.0288 | within the Stage 0 limits |
| `v0.5` | walk | 2560 | 0.9954 (knee speed) | 0.9954 (knee speed) | 0.8798 / 0.9724 / 0.0865 | 0.3179 / 0.5530 / 0.6157 | 29.4537 | -10 | 1.0429 | no | no | 0.0471 | 0.2647 (left_hip_roll max) | 0.0127 | 0.0232 | within the Stage 0 limits |

## What blocks a freeze

No screened reference is over a gate that blocks its freeze.

## Turns over a neck or tail limit (`a1_over`)

No screened turn is over a neck or tail planning limit.

## Reported, not gated

The frames of a move are not uniform in time where its stride period changes; the cont ratio by frames (every frame the same weight) is beside the gated one, which is over time.

| reference | frames uniform in time | neck cont over time / by frames | tail cont over time / by frames | neck with passive c / p | tail with passive c / p | double-support min mu | sway per stride (H; spec 0.12 H) | CoM horizontal accel peak (m/s^2) | near a limit (within 0.02 below 1) | record re-check equal |
|---|---|---|---|---|---|---|---|---|---|---|
| `nominal` | yes | 0.8544 / 0.8544 | 0.1444 / 0.1444 | 0.8692 / 0.9944 | 0.4088 / 0.4098 | 0.2113 | 0.1628 | 2.2768 | neck peak with the plant's damping and friction 0.9944 | yes |
| `v0.3` | yes | 0.8555 / 0.8555 | 0.2079 / 0.2079 | 0.8666 / 0.9478 | 0.4336 / 0.4777 | 0.2487 | 0.1257 | 2.5318 | - | yes |
| `v0.35` | yes | 0.8603 / 0.8603 | 0.2337 / 0.2337 | 0.8720 / 1.0410 | 0.4514 / 0.5146 | 0.2615 | 0.1140 | 2.7185 | - | yes |
| `v0.4` | yes | 0.8658 / 0.8658 | 0.2492 / 0.2492 | 0.8771 / 1.0692 | 0.4615 / 0.5381 | 0.2609 | 0.1080 | 2.9084 | - | yes |
| `v0.45` | yes | 0.8726 / 0.8726 | 0.2721 / 0.2721 | 0.8789 / 1.0832 | 0.4749 / 0.5711 | 0.2672 | 0.1008 | 3.7562 | - | yes |
| `v0.5` | yes | 0.8798 / 0.8798 | 0.3179 / 0.3179 | 0.8823 / 1.0968 | 0.5225 / 0.6823 | 0.2778 | 0.0837 | 3.9988 | knee speed 0.9954 | yes |

Over a neck or tail planning limit once the plant's joint damping and friction are added (outside Stage 0's rule):

- `v0.35`: neck peak with the plant's damping and friction 1.0410
- `v0.4`: neck peak with the plant's damping and friction 1.0692
- `v0.45`: neck peak with the plant's damping and friction 1.0832
- `v0.5`: neck peak with the plant's damping and friction 1.0968

## Not available

- `start`: no transition within the gates
- `nominal_to_v0.4`: no transition within the gates
