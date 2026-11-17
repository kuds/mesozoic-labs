# Stage 0 screen of set V6 on 60-N-T-10

Simulation only: every reference is priced as a prescribed motion that the joints follow exactly, on the round-two model. Made by `python -m designlab.stage2.setscreen md --set V6 --body 60-N-T-10` from `screen.json`; the rules are in its `rules` block (STAGE2_75_PLAN.md A12).

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
| `nominal` | walk | 1600 | 0.7889 (neck peak) | 0.7353 (knee speed) | 0.7644 / 0.7889 / 0.0853 | 0.1450 / 0.2581 / 0.3805 | 31.0890 | -10 | 0.5591 | yes | no | 0.0414 | 0.2087 (left_hip_roll max) | 0.0503 | 0.0375 | within the Stage 0 limits |
| `v0.3` | walk | 2080 | 0.7880 (knee speed) | 0.7880 (knee speed) | 0.7654 / 0.7541 / 0.0862 | 0.2083 / 0.3460 / 0.4668 | 30.7005 | -10 | 0.7313 | yes | no | 0.0465 | 0.2346 (left_hip_roll max) | 0.0123 | 0.0279 | within the Stage 0 limits |
| `v0.35` | walk | 2080 | 0.8617 (knee speed) | 0.8617 (knee speed) | 0.7698 / 0.7970 / 0.0863 | 0.2339 / 0.3912 / 0.4988 | 30.5222 | -10 | 0.8135 | yes | no | 0.0475 | 0.2429 (left_hip_roll max) | 0.0103 | 0.0269 | within the Stage 0 limits |
| `v0.4` | walk | 2240 | 0.8845 (knee speed) | 0.8845 (knee speed) | 0.7749 / 0.8328 / 0.0864 | 0.2492 / 0.4181 / 0.5165 | 30.3752 | -10 | 0.8751 | yes | no | 0.0473 | 0.2470 (left_hip_roll max) | 0.0103 | 0.0285 | within the Stage 0 limits |
| `v0.45` | walk | 2240 | 0.9147 (knee speed) | 0.9147 (knee speed) | 0.7810 / 0.8534 / 0.0866 | 0.2721 / 0.4568 / 0.5391 | 29.5480 | -10 | 0.9459 | no | no | 0.0473 | 0.2519 (left_hip_roll max) | 0.0103 | 0.0286 | within the Stage 0 limits |

## What blocks a freeze

No screened reference is over a gate that blocks its freeze.

## Turns over a neck or tail limit (`a1_over`)

No screened turn is over a neck or tail planning limit.

## Reported, not gated

The frames of a move are not uniform in time where its stride period changes; the cont ratio by frames (every frame the same weight) is beside the gated one, which is over time.

| reference | frames uniform in time | neck cont over time / by frames | tail cont over time / by frames | neck with passive c / p | tail with passive c / p | double-support min mu | sway per stride (H; spec 0.12 H) | CoM horizontal accel peak (m/s^2) | near a limit (within 0.02 below 1) | record re-check equal |
|---|---|---|---|---|---|---|---|---|---|---|
| `nominal` | yes | 0.7644 / 0.7644 | 0.1450 / 0.1450 | 0.7819 / 0.9087 | 0.4082 / 0.4095 | 0.2126 | 0.1633 | 2.2732 | - | yes |
| `v0.3` | yes | 0.7654 / 0.7654 | 0.2083 / 0.2083 | 0.7794 / 0.8704 | 0.4327 / 0.4760 | 0.2510 | 0.1262 | 2.5523 | - | yes |
| `v0.35` | yes | 0.7698 / 0.7698 | 0.2339 / 0.2339 | 0.7844 / 0.9546 | 0.4504 / 0.5124 | 0.2634 | 0.1145 | 2.7302 | - | yes |
| `v0.4` | yes | 0.7749 / 0.7749 | 0.2492 / 0.2492 | 0.7890 / 0.9793 | 0.4604 / 0.5358 | 0.2632 | 0.1084 | 2.9251 | - | yes |
| `v0.45` | yes | 0.7810 / 0.7810 | 0.2721 / 0.2721 | 0.7901 / 0.9903 | 0.4738 / 0.5686 | 0.2694 | 0.1013 | 3.8101 | neck peak with the plant's damping and friction 0.9903 | yes |

## Not available

- `v0.5`: dropped from the ladder: over knee speed 1.0163: blocks the freeze (knee speed); a walk that is not frozen is dropped from the ladder with its numbers
- `start`: no transition within the gates
- `nominal_to_v0.4`: no transition within the gates
