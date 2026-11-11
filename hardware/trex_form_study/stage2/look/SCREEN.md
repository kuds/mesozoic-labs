# Stage 2 look ladder: the Stage 0 screen (before any render)

Numbers from `python -m designlab.stage2.screen run` (`designlab/stage2/screen.py`) at commit `f4bc82b7680c` (clean tree), in `stage2/look/screen.json`, each block under the source function its `sources` names; this page is rendered from that file alone (`screen.markdown`; `python -m designlab.stage2.screen md`). Simulation only: prescribed references with perfect tracking on the round-two models (STAGE2_PLAN 4 A.2-A.3; research_look 4.1 item 2). Nothing is adopted (P24).

Inputs: `look_refs.json` (sha256 `368a33723d01...`) and `tail_phase.json` (`908a5f7eebd0...`) as committed; the 73 rungs' arrays (git-ignored, `stage2/look/refs/`) each checked against its recorded sha256. Ratios are against the record's drives with the sizing margins: cont = RMS / sustained (A1), peak = 1.25 x peak / planning peak, speed = 1.15 x speed / speed at the bus; over 1 is over the limit. Kind letters: c cont, p peak, s speed.

## In brief

- **Tail phase (research Bottom line 4): confirmed.** Round two's straight-run reading of the stance pad's peak single-support yaw moment |Mz| and friction need mu_req, changes (nominal / 0.40 m/s / animal pace): 60-N-T-10: today's tail over no tail, peak +9.5 / +8.1 / +7.6 %, mu_req +8.5 / +7.8 / +7.4 %; F9's phase instead of today's on the same walk, peak -17.6 / -15.4 / -14.4 %, mu_req -15.8 / -14.8 / -14.0 %; on the look walk (V4 to V4 with F9's tail), peak -16.4 / -13.1 / -13.0 %, mu_req -4.1 / -12.4 / -12.4 %; 75-M-H-10: today's tail over no tail, peak +11.5 / +10.8 / +9.4 %, mu_req +9.5 / +9.8 / +8.9 %; F9's phase instead of today's on the same walk, peak -16.9 / -19.7 / -17.6 %, mu_req -14.1 / -18.0 / -16.6 %; on the look walk (V4 to V4 with F9's tail), peak -17.3 / -16.3 / -15.7 %, mu_req -4.0 / -14.8 / -14.5 %. F9's phase lowers the friction need on every V0 walk. The reading reproduces the friction need of round two's recorded straight runs (nominal and brisk, both bodies) to 5.8e-13 relative (on 75-M-H-10 one touchdown frame is classified differently by the two generators, which moves a minimum such as r_cop_min, not the friction need: section 2).
- **Over a Stage 0 limit:** 75-M-H-10 animal pace (Fr 0.05) V6 (knee speed 1.012; the knee's peak comes in swing at 0.78 of it, on the shortest swing (0.345 s; swings 0.345-0.366 s; correlation of swing time and peak speed -1.00)); 75-M-H-10 animal pace (Fr 0.05) V6S (knee speed 1.009; the knee's peak comes in swing at 0.78 of it, on the shortest swing (0.345 s; swings 0.345-0.366 s; correlation of swing time and peak speed -1.00)). Under the plan's rule (4 A.3) each keeps its features and loses that speed rung unless the owner decides otherwise (recorded per rung, section 5).
- **Within 0.02 of a limit:** 60-N-T-10 nominal (Fr 0.012) V3T (knee speed 0.986); 75-M-H-10 nominal (Fr 0.012) V3T (knee speed 0.993).
- **Over a planning limit once the plant's joint damping and friction are added** (research Bottom line 3; outside Stage 0's rule, so no rung is lost; section 3b): 60-N-T-10 0.40 m/s neck peak on V1, V1B100, V2, V3, V3T, V4, V4L (up to 1.125); 60-N-T-10 animal pace (Fr 0.05) neck peak on V1B50, V1, V1B100, V2, V3, V3T, V4, V4L (up to 1.168); 75-M-H-10 0.40 m/s tail peak on V1B100, V3, V3T, V4, V4L (up to 1.105); 75-M-H-10 0.40 m/s tail cont on V4 (up to 1.014); 75-M-H-10 animal pace (Fr 0.05) tail cont on V0, V1B50, V1, V1B100, V2, V3, V3T, V4 (up to 1.071); 75-M-H-10 animal pace (Fr 0.05) tail peak on V0, V1B50, V1, V1B100, V2, V3, V3T, V4, V4L (up to 1.178).
- **Neck and tail with the plant's joint damping and friction** (nominal / 0.40 m/s / animal pace) (research Bottom line 3; Stage 0 leaves them out, so they cost no rung): 60-N-T-10: tail cont 0.684 / 0.830 / 0.850 on V0 (Stage 0 alone 0.242 / 0.347 / 0.362; servo power 0.038 / 0.064 / 0.068 W) and 0.408 / 0.459 / 0.469 with F9's phase on V5 (power -0.034 / -0.064 / -0.068 W); neck peak 0.805 / 0.860 / 0.868 on V0, up to 0.922 / 1.086 / 1.106 on V1-V4 (F1/F2's bob with today's nod) and 0.903 / 0.976 / 0.989 on V5 (F10's stabilised head); 75-M-H-10: tail cont 0.800 / 0.978 / 1.030 on V0 (Stage 0 alone 0.401 / 0.557 / 0.602; servo power 0.056 / 0.091 / 0.102 W) and 0.312 / 0.391 / 0.426 with F9's phase on V5 (power -0.065 / -0.092 / -0.102 W); neck peak 0.603 / 0.639 / 0.651 on V0, up to 0.734 / 0.850 / 0.888 on V1-V4 (F1/F2's bob with today's nod) and 0.807 / 0.771 / 0.822 on V5 (F10's stabilised head). With today's phase the servo drives the tail (positive power) and the damping adds to its load; with F9's the body's sway drives it and the servo brakes, which the damping helps.
- **Each feature's step on the friction need** (mu_req, relative change (nominal / 0.40 m/s / animal pace)): 60-N-T-10: F1/F2 -5.9 / -15.3 / -18.6 %, F3 +60.9 / +4.1 / +0.0 %, F4/F5 -27.7 / +15.1 / +18.2 %, F6 -0.2 / -0.1 / -0.0 %, F9/F10 -5.0 / -11.4 / -11.5 %, F14 +12.0 / +1.0 / +1.0 %; 75-M-H-10: F1/F2 -5.5 / -12.2 / -17.9 %, F3 +55.1 / +6.9 / +0.0 %, F4/F5 -28.0 / +13.6 / +18.7 %, F6 -0.4 / -0.2 / -0.1 %, F9/F10 -5.1 / -13.9 / -13.0 %, F14 +12.6 / +1.5 / +2.3 %. F3 changes nothing at the animal pace (its swing time is that walk's own) and at the nominal walk shortens the swing to 0.32-0.36 s, which accelerates the swing leg harder at lift-off: the stance pad's peak yaw moment rises with it.
- **The CoM's upward acceleration peak** (m/s^2 (nominal / 0.40 m/s / animal pace); the stance legs carry it): 60-N-T-10: V0 0.91 / 1.73 / 1.89, V1 1.67 / 2.74 / 3.11 (F2's 0.1 s drop), V3 2.34 / 4.46 / 4.61 (F5's short roll), V6 2.28 / 4.45 / 4.57; 75-M-H-10: V0 0.84 / 1.51 / 1.85, V1 2.20 / 3.13 / 4.03 (F2's 0.1 s drop), V3 2.40 / 5.15 / 5.58 (F5's short roll), V6 2.36 / 5.28 / 5.95.
- **Grip (F21's friction need on the straight walk, against the plan's mu 0.54 and 0.9):** over 0.9: 60-N-T-10 0.40 m/s: V0, V3, V3T, V4, V4L; 60-N-T-10 animal pace (Fr 0.05): V0, V1B50, V3, V3T, V4, V4L, V5, V6, V6S. Over 0.54: 60-N-T-10 nominal (Fr 0.012): V2, V6; 60-N-T-10 0.40 m/s: V0, V1B50, V1, V1B100, V2, V3, V3T, V4, V4L, V5, V6, V6S; 60-N-T-10 animal pace (Fr 0.05): V0, V1B50, V1, V1B100, V2, V3, V3T, V4, V4L, V5, V6, V6S; 75-M-H-10 0.40 m/s: V0, V1B50, V1, V2, V3, V3T, V4, V4L, V5, V6, V6S; 75-M-H-10 animal pace (Fr 0.05): V0, V1B50, V1, V1B100, V2, V3, V3T, V4, V4L, V5, V5F13, V6, V6S.
- **V0 to V6** (nominal / 0.40 m/s / animal pace) (section 6 for every number): 60-N-T-10: governing 0.761 / 0.865 / 0.929 to 0.789 / 0.885 / 0.911; knee speed 0.559 / 0.865 / 0.929 to 0.735 / 0.885 / 0.911; hip pitch 0.525 / 0.832 / 0.899 to 0.464 / 0.616 / 0.672; thigh twist 0.576 / 0.643 / 0.661 to 0.519 / 0.720 / 0.736; tail cont with passive 0.684 / 0.830 / 0.850 to 0.408 / 0.460 / 0.471; neck peak with passive 0.805 / 0.860 / 0.868 to 0.909 / 0.979 / 0.992; mu_req 0.481 / 0.964 / 1.092 to 0.559 / 0.875 / 0.938; sway 0.168 / 0.108 / 0.103 to 0.164 / 0.110 / 0.104 H; CoM upward accel 0.91 / 1.73 / 1.89 to 2.28 / 4.45 / 4.57 m/s^2; 75-M-H-10: governing 0.662 / 0.804 / 0.935 to 0.757 / 0.752 / 1.012; knee speed 0.555 / 0.804 / 0.935 to 0.757 / 0.752 / 1.012; hip pitch 0.567 / 0.716 / 0.834 to 0.566 / 0.646 / 0.697; thigh twist 0.622 / 0.674 / 0.709 to 0.572 / 0.747 / 0.775; tail cont with passive 0.800 / 0.978 / 1.030 to 0.314 / 0.394 / 0.388; neck peak with passive 0.603 / 0.639 / 0.651 to 0.812 / 0.772 / 0.761; mu_req 0.361 / 0.629 / 0.787 to 0.405 / 0.585 / 0.683; sway 0.164 / 0.111 / 0.101 to 0.161 / 0.112 / 0.116 H; CoM upward accel 0.84 / 1.51 / 1.85 to 2.36 / 5.28 / 5.95 m/s^2.

## 1. Limits, speeds and method

| Drive group | 60-N-T-10: part, sustained / planning peak (N m), speed (rad/s) | 75-M-H-10: part, sustained / planning peak (N m), speed (rad/s) |
|---|---|---|
| hip roll | cubemars_ak45_36_v3: 4.560 / 16.800, 4.19 at 24 V | myactuator_rmd_x4_p36_36_e_l: 5.985 / 21.000, 8.69 at 24 V |
| hip pitch | myactuator_rmd_x4_p36_36_e_l: 5.985 / 21.000, 8.69 at 24 V | myactuator_rmd_x6_p20_60_e: 11.400 / 42.000, 8.01 at 24 V |
| thigh twist | cubemars_ak45_36_v3: 4.560 / 16.800, 4.19 at 24 V | myactuator_rmd_x4_p36_36_e_l: 5.985 / 21.000, 8.69 at 24 V |
| knee | myactuator_rmd_x4_p36_36_e_l: 5.985 / 21.000, 8.69 at 24 V | myactuator_rmd_x4_p36_36_e_l: 5.985 / 21.000, 8.69 at 24 V |
| ankle A | myactuator_rmd_x4_p12_10_e: 2.280 / 7.000, 24.92 at 24 V | myactuator_rmd_x4_p36_36_e_l: 5.985 / 21.000, 8.69 at 24 V |
| ankle B | myactuator_rmd_x4_p12_10_e: 2.280 / 7.000, 24.92 at 24 V | myactuator_rmd_x4_p36_36_e_l: 5.985 / 21.000, 8.69 at 24 V |
| neck | feetech_sts3250: 0.819 / 1.437, 4.71 | robotis_xm540_w270_t: 1.770 / 3.106, 1.88 |
| tail | feetech_sts3215_c018: 0.491 / 0.862, 2.83 | feetech_sts3215_c018: 0.491 / 0.862, 2.83 |

The plant's MJCF gives the neck pitch and tail yaw 60-N-T-10: damping 0.2 / 0.2 N m s/rad, friction loss 0.15 / 0.15 N m; 75-M-H-10: damping 0.2 / 0.2 N m s/rad, friction loss 0.15 / 0.15 N m (neck / tail); Stage 0's inverse dynamics leave them out, the `passive` columns add them.

**0.40 m/s** (the agility bar's speed) is defined from the existing rules: the gait the brisk_to_fr05 speed change passes through at 0.40 m/s: period, duty and lift mixed with the speed's weight (transitions.py) (`look.speed_spec`, documented in `look.py`).

| Body | Speed (m/s) | Period (s) | Stride (H) | Duty | Lift (H) | Duty with F3 (V2+) | Weight w | Brisk / animal pace (the two walks mixed) |
|---|---|---|---|---|---|---|---|---|
| 60-N-T-10 | 0.400 | 0.751 | 0.742 | 0.567 | 0.110 | 0.574 | 0.651 | 0.315 m/s, T 0.796 s, duty 0.58 / 0.446 m/s, T 0.727 s, duty 0.56 |
| 75-M-H-10 | 0.400 | 0.861 | 0.686 | 0.573 | 0.110 | 0.586 | 0.338 | 0.351 m/s, T 0.887 s, duty 0.58 / 0.496 m/s, T 0.809 s, duty 0.56 |

Method, per rung (`screen.screen_rung`; full list in screen.json `sources`): `look.price` re-run on the arrays (`look.look_inverse_dynamics`, i.e. `gait.inverse_dynamics` with F5's rolling contact, then `sizing._motor_space` and `envelope._ratios`; equal to the price `look_refs.json` records), `gait.summarize`, the plant's joint damping and friction on the neck and tail (`screen.passive_loads`), round two's straight-run friction reading (`screen.straight_run`: `turning.inverse_dynamics` and `turning._foot_stats`), sway, CoM and foot accelerations, and the clearance screen's walking gates (`clearance.screen`, gates r2).

## 2. The tail-phase check (research Bottom line 4; plan 4 A.2)

Round two's straight-run reading on each walk: the stance pad's free yaw moment about its CoP over single support (peak and RMS, N m), the friction need mu_req (F21's: limit surface with the pads' uniform-pressure radius) and mu_req_cop (the CoP-based radius), and the double-support requirement over the frames where no foot rolls (frames read / all). Tail: amplitude (rad) at phase (rad); today's is 0.12 at pi/2 = 1.571; F9's phase is the ladder's (the V4 minimiser of the peak, `tail_phase.json`).

### 60-N-T-10 (60 cm Tapered)

| Speed | Walk | Tail | Peak \|Mz\| | RMS \|Mz\| | mu_req | mu_req_cop | DS min mu (read) |
|---|---|---|---|---|---|---|---|
| nominal (Fr 0.012) | V0 | 0.120 at +1.571 | 1.774 | 0.947 | 0.481 | 0.480 | 0.203 (32/32) |
| nominal (Fr 0.012) | V0_no_tail | 0.000 at +1.571 | 1.620 | 0.882 | 0.443 | 0.443 | 0.205 (32/32) |
| nominal (Fr 0.012) | V0_F9_phase | 0.120 at -1.702 | 1.462 | 0.831 | 0.405 | 0.405 | 0.208 (32/32) |
| nominal (Fr 0.012) | V0_F9 | 0.125 at -1.702 | 1.456 | 0.829 | 0.403 | 0.403 | 0.208 (32/32) |
| nominal (Fr 0.012) | V0_look_path | 0.120 at +7.854 | 1.774 | 0.947 | 0.481 | 0.480 | 0.203 (32/32) |
| nominal (Fr 0.012) | V4 | 0.120 at +1.571 | 2.057 | 1.155 | 0.525 | 0.511 | 0.183 (30/58) |
| nominal (Fr 0.012) | V4_no_tail | 0.000 at +1.571 | 1.892 | 1.110 | 0.490 | 0.476 | 0.184 (30/58) |
| nominal (Fr 0.012) | V4_F9 | 0.125 at -1.702 | 1.720 | 1.079 | 0.504 | 0.489 | 0.186 (30/58) |
| nominal (Fr 0.012) | V5 | 0.125 at -1.702 | 1.723 | 1.075 | 0.499 | 0.485 | 0.178 (30/58) |
| nominal (Fr 0.012) | V6 | 0.125 at -1.702 | 1.809 | 1.085 | 0.559 | 0.542 | 0.213 (305/599) |
| 0.40 m/s | V0 | 0.120 at +1.571 | 3.737 | 1.937 | 0.964 | 0.964 | 0.219 (20/20) |
| 0.40 m/s | V0_no_tail | 0.000 at +1.571 | 3.457 | 1.800 | 0.895 | 0.894 | 0.219 (20/20) |
| 0.40 m/s | V0_F9_phase | 0.120 at -1.780 | 3.163 | 1.690 | 0.822 | 0.821 | 0.223 (20/20) |
| 0.40 m/s | V0_F9 | 0.125 at -1.780 | 3.151 | 1.686 | 0.819 | 0.818 | 0.223 (20/20) |
| 0.40 m/s | V0_look_path | 0.120 at +7.854 | 3.737 | 1.937 | 0.964 | 0.964 | 0.219 (20/20) |
| 0.40 m/s | V4 | 0.120 at +1.571 | 4.622 | 1.847 | 0.978 | 0.977 | 0.216 (10/22) |
| 0.40 m/s | V4_no_tail | 0.000 at +1.571 | 4.329 | 1.715 | 0.919 | 0.917 | 0.216 (10/22) |
| 0.40 m/s | V4_F9 | 0.125 at -1.780 | 4.018 | 1.609 | 0.856 | 0.855 | 0.219 (10/22) |
| 0.40 m/s | V5 | 0.125 at -1.780 | 4.112 | 1.642 | 0.867 | 0.865 | 0.210 (10/22) |
| 0.40 m/s | V6 | 0.125 at -1.780 | 4.157 | 1.651 | 0.875 | 0.874 | 0.263 (159/329) |
| animal pace (Fr 0.05) | V0 | 0.120 at +1.571 | 4.229 | 2.163 | 1.092 | 1.092 | 0.229 (18/18) |
| animal pace (Fr 0.05) | V0_no_tail | 0.000 at +1.571 | 3.931 | 2.012 | 1.017 | 1.017 | 0.231 (18/18) |
| animal pace (Fr 0.05) | V0_F9_phase | 0.120 at -1.780 | 3.618 | 1.890 | 0.939 | 0.939 | 0.236 (18/18) |
| animal pace (Fr 0.05) | V0_F9 | 0.125 at -1.780 | 3.605 | 1.885 | 0.936 | 0.935 | 0.236 (18/18) |
| animal pace (Fr 0.05) | V0_look_path | 0.120 at +7.854 | 4.229 | 2.163 | 1.092 | 1.092 | 0.229 (18/18) |
| animal pace (Fr 0.05) | V4 | 0.120 at +1.571 | 4.954 | 1.995 | 1.050 | 1.048 | 0.236 (8/18) |
| animal pace (Fr 0.05) | V4_no_tail | 0.000 at +1.571 | 4.643 | 1.847 | 0.987 | 0.985 | 0.236 (8/18) |
| animal pace (Fr 0.05) | V4_F9 | 0.125 at -1.780 | 4.313 | 1.722 | 0.920 | 0.919 | 0.240 (8/18) |
| animal pace (Fr 0.05) | V5 | 0.125 at -1.780 | 4.402 | 1.763 | 0.929 | 0.928 | 0.233 (8/18) |
| animal pace (Fr 0.05) | V6 | 0.125 at -1.780 | 4.453 | 1.770 | 0.938 | 0.937 | 0.281 (128/268) |

| Speed | F9 phase (cycles from today) | V0: no tail to today (peak / mu) | V0: today to F9's phase | V0: today to F9's tail | V4: today to F9's tail | V4: no tail to F9's tail | V4 to V5 (F9 + F10) |
|---|---|---|---|---|---|---|---|
| nominal (Fr 0.012) | -1.702 (+0.479) | +9.5 % / +8.5 % | -17.6 % / -15.8 % | -17.9 % / -16.1 % | -16.4 % / -4.1 % | -9.1 % / +2.8 % | -16.2 % / -5.0 % |
| 0.40 m/s | -1.780 (+0.467) | +8.1 % / +7.8 % | -15.4 % / -14.8 % | -15.7 % / -15.1 % | -13.1 % / -12.4 % | -7.2 % / -6.8 % | -11.0 % / -11.4 % |
| animal pace (Fr 0.05) | -1.780 (+0.467) | +7.6 % / +7.4 % | -14.4 % / -14.0 % | -14.7 % / -14.3 % | -13.0 % / -12.4 % | -7.1 % / -6.8 % | -11.1 % / -11.5 % |

Record check (`round2/refine/narrow_carpet_60/review/screening/60-N-T-10_turning.json`): V0 at the nominal walk against `runs.straight` (27 fields; friction fields to 1.6e-13, every field to 1.6e-13; mu_req 0.4807 against 0.4807) and today's brisk walk against `runs.straight_brisk` (friction fields to 4.4e-13; mu_req 0.7540). Control (today's phase through the look path): nominal (Fr 0.012) joints 2.3e-15 rad, |Mz| 1.8e-13 N m; 0.40 m/s joints 2.2e-15 rad, |Mz| 4.1e-14 N m; animal pace (Fr 0.05) joints 2.2e-15 rad, |Mz| 3.8e-13 N m.

### 75-M-H-10 (75 cm Hybrid)

| Speed | Walk | Tail | Peak \|Mz\| | RMS \|Mz\| | mu_req | mu_req_cop | DS min mu (read) |
|---|---|---|---|---|---|---|---|
| nominal (Fr 0.012) | V0 | 0.120 at +1.571 | 2.479 | 1.345 | 0.361 | 0.361 | 0.198 (31/31) |
| nominal (Fr 0.012) | V0_no_tail | 0.000 at +1.571 | 2.223 | 1.243 | 0.330 | 0.330 | 0.199 (31/31) |
| nominal (Fr 0.012) | V0_F9_phase | 0.120 at -0.925 | 2.060 | 1.140 | 0.310 | 0.310 | 0.203 (31/31) |
| nominal (Fr 0.012) | V0_F9 | 0.125 at -0.925 | 2.053 | 1.136 | 0.309 | 0.309 | 0.203 (31/31) |
| nominal (Fr 0.012) | V0_look_path | 0.120 at +7.854 | 2.479 | 1.345 | 0.361 | 0.361 | 0.198 (31/31) |
| nominal (Fr 0.012) | V4 | 0.120 at +1.571 | 2.726 | 1.679 | 0.379 | 0.370 | 0.174 (30/58) |
| nominal (Fr 0.012) | V4_no_tail | 0.000 at +1.571 | 2.455 | 1.610 | 0.384 | 0.373 | 0.176 (30/58) |
| nominal (Fr 0.012) | V4_F9 | 0.125 at -0.925 | 2.256 | 1.530 | 0.364 | 0.354 | 0.182 (30/58) |
| nominal (Fr 0.012) | V5 | 0.125 at -0.925 | 2.265 | 1.523 | 0.360 | 0.350 | 0.172 (30/58) |
| nominal (Fr 0.012) | V6 | 0.125 at -0.925 | 2.503 | 1.532 | 0.405 | 0.393 | 0.197 (270/534) |
| 0.40 m/s | V0 | 0.120 at +1.571 | 4.627 | 2.487 | 0.629 | 0.628 | 0.206 (22/22) |
| 0.40 m/s | V0_no_tail | 0.000 at +1.571 | 4.177 | 2.286 | 0.573 | 0.572 | 0.208 (22/22) |
| 0.40 m/s | V0_F9_phase | 0.120 at -1.772 | 3.718 | 2.135 | 0.516 | 0.515 | 0.211 (22/22) |
| 0.40 m/s | V0_F9 | 0.125 at -1.772 | 3.699 | 2.129 | 0.513 | 0.513 | 0.211 (22/22) |
| 0.40 m/s | V0_look_path | 0.120 at +7.854 | 4.627 | 2.487 | 0.629 | 0.628 | 0.206 (22/22) |
| 0.40 m/s | V4 | 0.120 at +1.571 | 5.848 | 2.425 | 0.669 | 0.668 | 0.221 (12/26) |
| 0.40 m/s | V4_no_tail | 0.000 at +1.571 | 5.386 | 2.244 | 0.621 | 0.620 | 0.221 (12/26) |
| 0.40 m/s | V4_F9 | 0.125 at -1.772 | 4.895 | 2.111 | 0.570 | 0.569 | 0.224 (12/26) |
| 0.40 m/s | V5 | 0.125 at -1.772 | 4.993 | 2.134 | 0.576 | 0.575 | 0.212 (12/26) |
| 0.40 m/s | V6 | 0.125 at -1.772 | 5.086 | 2.144 | 0.585 | 0.584 | 0.272 (160/332) |
| animal pace (Fr 0.05) | V0 | 0.120 at +1.571 | 5.826 | 3.079 | 0.787 | 0.787 | 0.223 (18/18) |
| animal pace (Fr 0.05) | V0_no_tail | 0.000 at +1.571 | 5.324 | 2.841 | 0.723 | 0.722 | 0.224 (18/18) |
| animal pace (Fr 0.05) | V0_F9_phase | 0.120 at -1.772 | 4.802 | 2.653 | 0.656 | 0.656 | 0.228 (18/18) |
| animal pace (Fr 0.05) | V0_F9 | 0.125 at -1.772 | 4.781 | 2.646 | 0.654 | 0.653 | 0.229 (18/18) |
| animal pace (Fr 0.05) | V0_look_path | 0.120 at +7.854 | 5.826 | 3.079 | 0.787 | 0.787 | 0.223 (18/18) |
| animal pace (Fr 0.05) | V4 | 0.120 at +1.571 | 6.807 | 2.836 | 0.767 | 0.766 | 0.240 (8/18) |
| animal pace (Fr 0.05) | V4_no_tail | 0.000 at +1.571 | 6.288 | 2.606 | 0.713 | 0.711 | 0.239 (8/18) |
| animal pace (Fr 0.05) | V4_F9 | 0.125 at -1.772 | 5.735 | 2.423 | 0.655 | 0.654 | 0.242 (8/18) |
| animal pace (Fr 0.05) | V5 | 0.125 at -1.772 | 5.946 | 2.490 | 0.667 | 0.666 | 0.232 (8/18) |
| animal pace (Fr 0.05) | V5F13 | 0.125 at -1.772 | 5.876 | 2.801 | 0.678 | 0.676 | 0.239 (14/30) |
| animal pace (Fr 0.05) | V6 | 0.125 at -1.772 | 5.931 | 2.816 | 0.683 | 0.682 | 0.300 (172/377) |

| Speed | F9 phase (cycles from today) | V0: no tail to today (peak / mu) | V0: today to F9's phase | V0: today to F9's tail | V4: today to F9's tail | V4: no tail to F9's tail | V4 to V5 (F9 + F10) |
|---|---|---|---|---|---|---|---|
| nominal (Fr 0.012) | -0.925 (-0.397) | +11.5 % / +9.5 % | -16.9 % / -14.1 % | -17.2 % / -14.3 % | -17.3 % / -4.0 % | -8.1 % / -5.1 % | -16.9 % / -5.1 % |
| 0.40 m/s | -1.772 (+0.468) | +10.8 % / +9.8 % | -19.7 % / -18.0 % | -20.0 % / -18.4 % | -16.3 % / -14.8 % | -9.1 % / -8.2 % | -14.6 % / -13.9 % |
| animal pace (Fr 0.05) | -1.772 (+0.468) | +9.4 % / +8.9 % | -17.6 % / -16.6 % | -17.9 % / -17.0 % | -15.7 % / -14.5 % | -8.8 % / -8.0 % | -12.6 % / -13.0 % |

Record check (`round2/refine/feet_75/review/screening/75-M-H-10_turning.json`): V0 at the nominal walk against `runs.straight` (27 fields; friction fields to 2.7e-13, every field to 1.6e-02; mu_req 0.3612 against 0.3612; differing: right frames 65 against 64; right r_cop_min 0.0492956 against 0.0493385 (gait.generate and turning.generate classify a touchdown frame differently (turning docstring, Events): one frame more or less in single support)) and today's brisk walk against `runs.straight_brisk` (friction fields to 5.8e-13; mu_req 0.5582). Control (today's phase through the look path): nominal (Fr 0.012) joints 2.5e-15 rad, |Mz| 1.3e-13 N m; 0.40 m/s joints 2.3e-15 rad, |Mz| 8.3e-14 N m; animal pace (Fr 0.05) joints 2.8e-15 rad, |Mz| 2.3e-13 N m.

## 3. Drives: ratios against the planning limits (1.15 speed / 1.25 peak margins)

Governing: the largest ratio. Each group: its worst ratio and kind. Neck and tail: Stage 0's (inverse dynamics) / with the plant's joint damping and friction. Status: the plan's rule (4 A.3); near = within 0.02 of a limit.

### 60-N-T-10, nominal (Fr 0.012)

| Rung | Features | Governing | Knee s | Knee c | Hip pitch | Ankle A | Ankle B | Hip roll | Thigh twist | Neck c | Tail c | Tail p | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | today | neck cont 0.761 | 0.559 | 0.460 | 0.525 s | 0.524 c | 0.524 c | 0.511 c | 0.576 c | 0.761 / 0.818 | 0.242 / 0.684 | 0.313 / 0.569 | within |
| V1B50 | F1, F2, F1 at 50 % | neck cont 0.763 | 0.556 | 0.466 | 0.513 s | 0.524 c | 0.524 c | 0.510 c | 0.582 c | 0.763 / 0.821 | 0.241 / 0.683 | 0.315 / 0.568 | within |
| V1 | F1, F2 | neck cont 0.764 | 0.554 | 0.469 | 0.508 s | 0.525 c | 0.525 c | 0.509 c | 0.584 c | 0.764 / 0.824 | 0.241 / 0.682 | 0.317 / 0.568 | within |
| V1B100 | F1, F2, F1 at 100 % | neck cont 0.766 | 0.553 | 0.474 | 0.503 s | 0.525 c | 0.525 c | 0.509 c | 0.587 c | 0.766 / 0.827 | 0.241 / 0.682 | 0.318 / 0.568 | within |
| V2 | F1, F2, F3 | knee speed 0.776 | 0.776 | 0.454 | 0.700 s | 0.509 c | 0.509 c | 0.476 c | 0.556 c | 0.765 / 0.827 | 0.243 / 0.658 | 0.310 / 0.598 | within |
| V3 | F1, F2, F3, F4, F5 | neck cont 0.765 | 0.715 | 0.471 | 0.462 c | 0.515 c | 0.515 c | 0.471 c | 0.538 c | 0.765 / 0.825 | 0.238 / 0.656 | 0.334 / 0.603 | within |
| V3T | F1, F2, F3, F4, F5, F5 at 0.40 rad (the research's middle) | knee speed 0.986 | 0.986 | 0.481 | 0.563 s | 0.520 c | 0.520 c | 0.472 c | 0.538 c | 0.765 / 0.825 | 0.238 / 0.656 | 0.326 / 0.602 | near: knee speed 0.986 |
| V4 | F1, F2, F3, F4, F5, F6 | neck cont 0.763 | 0.716 | 0.471 | 0.462 c | 0.515 c | 0.515 c | 0.468 c | 0.535 c | 0.763 / 0.824 | 0.260 / 0.680 | 0.351 / 0.620 | within |
| V4L | F1, F2, F3, F4, F5, F6 (swing side drops) | neck cont 0.766 | 0.713 | 0.471 | 0.462 c | 0.515 c | 0.515 c | 0.475 c | 0.541 c | 0.766 / 0.826 | 0.216 / 0.632 | 0.317 / 0.585 | within |
| V5 | F1, F2, F3, F4, F5, F6, F9, F10 | neck peak 0.785 | 0.719 | 0.470 | 0.465 c | 0.513 c | 0.513 c | 0.475 c | 0.523 c | 0.764 / 0.792 | 0.144 / 0.408 | 0.243 / 0.355 | within |
| V6 | F1, F2, F3, F4, F5, F6, F9, F10, F14 | neck peak 0.789 | 0.735 | 0.469 | 0.464 c | 0.513 c | 0.514 c | 0.470 c | 0.519 c | 0.764 / 0.782 | 0.145 / 0.408 | 0.258 / 0.409 | within |
| V6S | F1, F2, F3, F4, F5, F6, F9, F10, F14, F7 | neck peak 0.789 | 0.744 | 0.464 | 0.463 c | 0.540 p | 0.537 c | 0.460 c | 0.525 c | 0.765 / 0.782 | 0.130 / 0.420 | 0.233 / 0.423 | within |

### 60-N-T-10, 0.40 m/s

| Rung | Features | Governing | Knee s | Knee c | Hip pitch | Ankle A | Ankle B | Hip roll | Thigh twist | Neck c | Tail c | Tail p | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | today | knee speed 0.865 | 0.865 | 0.478 | 0.832 s | 0.541 c | 0.541 c | 0.547 c | 0.643 c | 0.768 / 0.841 | 0.347 / 0.830 | 0.462 / 0.747 | within |
| V1B50 | F1, F2, F1 at 50 % | knee speed 0.852 | 0.852 | 0.486 | 0.809 s | 0.538 c | 0.538 c | 0.542 c | 0.657 c | 0.772 / 0.855 | 0.349 / 0.828 | 0.465 / 0.752 | within |
| V1 | F1, F2 | knee speed 0.845 | 0.845 | 0.494 | 0.799 s | 0.538 c | 0.538 c | 0.540 c | 0.664 c | 0.777 / 0.865 | 0.350 / 0.827 | 0.467 / 0.756 | within |
| V1B100 | F1, F2, F1 at 100 % | neck peak 0.856 | 0.838 | 0.506 | 0.790 s | 0.540 c | 0.540 c | 0.538 c | 0.673 c | 0.784 / 0.876 | 0.351 / 0.827 | 0.470 / 0.760 | within |
| V2 | F1, F2, F3 | knee speed 0.868 | 0.868 | 0.494 | 0.819 s | 0.536 c | 0.536 c | 0.539 c | 0.665 c | 0.777 / 0.867 | 0.350 / 0.826 | 0.470 / 0.754 | within |
| V3 | F1, F2, F3, F4, F5 | knee speed 0.870 | 0.870 | 0.502 | 0.583 s | 0.547 c | 0.547 c | 0.535 c | 0.736 p | 0.778 / 0.866 | 0.349 / 0.824 | 0.531 / 0.800 | within |
| V3T | F1, F2, F3, F4, F5, F5 at 0.40 rad (the research's middle) | knee speed 0.852 | 0.852 | 0.503 | 0.578 s | 0.550 p | 0.550 p | 0.535 c | 0.732 p | 0.778 / 0.866 | 0.348 / 0.824 | 0.530 / 0.799 | within |
| V4 | F1, F2, F3, F4, F5, F6 | knee speed 0.872 | 0.872 | 0.502 | 0.585 s | 0.547 c | 0.547 c | 0.532 c | 0.735 p | 0.776 / 0.864 | 0.369 / 0.852 | 0.544 / 0.814 | within |
| V4L | F1, F2, F3, F4, F5, F6 (swing side drops) | knee speed 0.868 | 0.868 | 0.502 | 0.582 s | 0.547 c | 0.547 c | 0.539 c | 0.738 p | 0.780 / 0.867 | 0.330 / 0.797 | 0.518 / 0.786 | within |
| V5 | F1, F2, F3, F4, F5, F6, F9, F10 | knee speed 0.877 | 0.877 | 0.504 | 0.589 c | 0.545 c | 0.545 c | 0.536 c | 0.717 p | 0.775 / 0.794 | 0.248 / 0.459 | 0.382 / 0.453 | within |
| V6 | F1, F2, F3, F4, F5, F6, F9, F10, F14 | knee speed 0.885 | 0.885 | 0.503 | 0.616 s | 0.543 c | 0.542 c | 0.534 c | 0.720 p | 0.775 / 0.789 | 0.249 / 0.460 | 0.418 / 0.536 | within |
| V6S | F1, F2, F3, F4, F5, F6, F9, F10, F14, F7 | knee speed 0.889 | 0.889 | 0.499 | 0.619 s | 0.608 p | 0.607 p | 0.525 c | 0.724 p | 0.775 / 0.789 | 0.233 / 0.479 | 0.392 / 0.552 | within |

### 60-N-T-10, animal pace (Fr 0.05)

| Rung | Features | Governing | Knee s | Knee c | Hip pitch | Ankle A | Ankle B | Hip roll | Thigh twist | Neck c | Tail c | Tail p | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | today | knee speed 0.929 | 0.929 | 0.486 | 0.899 s | 0.545 c | 0.545 c | 0.556 c | 0.661 c | 0.769 / 0.843 | 0.362 / 0.850 | 0.483 / 0.772 | within |
| V1B50 | F1, F2, F1 at 50 % | knee speed 0.910 | 0.910 | 0.495 | 0.873 s | 0.541 c | 0.541 c | 0.551 c | 0.679 c | 0.774 / 0.860 | 0.365 / 0.848 | 0.493 / 0.780 | within |
| V1 | F1, F2 | knee speed 0.900 | 0.900 | 0.506 | 0.862 s | 0.542 c | 0.542 c | 0.549 c | 0.689 c | 0.781 / 0.872 | 0.366 / 0.847 | 0.498 / 0.785 | within |
| V1B100 | F1, F2, F1 at 100 % | neck peak 0.891 | 0.890 | 0.521 | 0.852 s | 0.545 c | 0.545 c | 0.547 c | 0.701 c | 0.790 / 0.887 | 0.368 / 0.847 | 0.503 / 0.791 | within |
| V2 | F1, F2, F3 | knee speed 0.900 | 0.900 | 0.506 | 0.862 s | 0.542 c | 0.542 c | 0.549 c | 0.689 c | 0.781 / 0.872 | 0.366 / 0.847 | 0.498 / 0.785 | within |
| V3 | F1, F2, F3, F4, F5 | knee speed 0.890 | 0.890 | 0.514 | 0.634 s | 0.552 c | 0.552 c | 0.542 c | 0.754 p | 0.781 / 0.872 | 0.365 / 0.846 | 0.550 / 0.825 | within |
| V3T | F1, F2, F3, F4, F5, F5 at 0.40 rad (the research's middle) | knee speed 0.878 | 0.878 | 0.515 | 0.631 s | 0.552 c | 0.552 c | 0.542 c | 0.751 p | 0.781 / 0.872 | 0.364 / 0.846 | 0.549 / 0.825 | within |
| V4 | F1, F2, F3, F4, F5, F6 | knee speed 0.892 | 0.892 | 0.513 | 0.635 s | 0.552 c | 0.552 c | 0.539 c | 0.753 p | 0.779 / 0.870 | 0.384 / 0.873 | 0.561 / 0.840 | within |
| V4L | F1, F2, F3, F4, F5, F6 (swing side drops) | knee speed 0.888 | 0.888 | 0.514 | 0.632 s | 0.552 c | 0.552 c | 0.546 c | 0.755 p | 0.783 / 0.873 | 0.346 / 0.819 | 0.538 / 0.811 | within |
| V5 | F1, F2, F3, F4, F5, F6, F9, F10 | knee speed 0.898 | 0.898 | 0.516 | 0.634 s | 0.551 c | 0.551 c | 0.543 c | 0.733 p | 0.780 / 0.802 | 0.265 / 0.469 | 0.408 / 0.474 | within |
| V6 | F1, F2, F3, F4, F5, F6, F9, F10, F14 | knee speed 0.911 | 0.911 | 0.514 | 0.672 s | 0.548 c | 0.548 c | 0.541 c | 0.736 p | 0.780 / 0.790 | 0.266 / 0.471 | 0.447 / 0.561 | within |
| V6S | F1, F2, F3, F4, F5, F6, F9, F10, F14, F7 | knee speed 0.915 | 0.915 | 0.510 | 0.674 s | 0.609 p | 0.608 p | 0.533 c | 0.741 p | 0.781 / 0.791 | 0.250 / 0.489 | 0.420 / 0.577 | within |

### 75-M-H-10, nominal (Fr 0.012)

| Rung | Features | Governing | Knee s | Knee c | Hip pitch | Ankle A | Ankle B | Hip roll | Thigh twist | Neck c | Tail c | Tail p | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | today | neck cont 0.662 | 0.555 | 0.522 | 0.567 c | 0.420 s | 0.420 s | 0.606 c | 0.622 c | 0.662 / 0.676 | 0.401 / 0.800 | 0.515 / 0.745 | within |
| V1B50 | F1, F2, F1 at 50 % | neck cont 0.663 | 0.552 | 0.532 | 0.567 c | 0.420 s | 0.420 s | 0.605 c | 0.630 c | 0.663 / 0.678 | 0.400 / 0.799 | 0.519 / 0.749 | within |
| V1 | F1, F2 | neck cont 0.665 | 0.550 | 0.539 | 0.567 c | 0.420 s | 0.420 s | 0.605 c | 0.633 c | 0.665 / 0.680 | 0.400 / 0.798 | 0.521 / 0.751 | within |
| V1B100 | F1, F2, F1 at 100 % | neck peak 0.680 | 0.548 | 0.548 | 0.568 c | 0.420 s | 0.420 s | 0.604 c | 0.637 c | 0.667 / 0.683 | 0.400 / 0.798 | 0.523 / 0.753 | within |
| V2 | F1, F2, F3 | knee speed 0.773 | 0.773 | 0.542 | 0.648 s | 0.551 s | 0.551 s | 0.559 c | 0.601 c | 0.666 / 0.682 | 0.403 / 0.765 | 0.504 / 0.739 | within |
| V3 | F1, F2, F3, F4, F5 | knee speed 0.749 | 0.749 | 0.574 | 0.563 c | 0.532 s | 0.532 s | 0.555 c | 0.586 c | 0.666 / 0.681 | 0.395 / 0.761 | 0.536 / 0.773 | within |
| V3T | F1, F2, F3, F4, F5, F5 at 0.40 rad (the research's middle) | knee speed 0.993 | 0.993 | 0.592 | 0.633 s | 0.639 s | 0.639 s | 0.555 c | 0.587 c | 0.666 / 0.681 | 0.395 / 0.761 | 0.524 / 0.757 | near: knee speed 0.993 |
| V4 | F1, F2, F3, F4, F5, F6 | knee speed 0.750 | 0.750 | 0.574 | 0.563 c | 0.532 s | 0.532 s | 0.550 c | 0.582 c | 0.664 / 0.680 | 0.434 / 0.804 | 0.565 / 0.803 | within |
| V4L | F1, F2, F3, F4, F5, F6 (swing side drops) | knee speed 0.748 | 0.748 | 0.574 | 0.563 c | 0.533 s | 0.533 s | 0.560 c | 0.590 c | 0.667 / 0.682 | 0.357 / 0.718 | 0.507 / 0.743 | within |
| V5 | F1, F2, F3, F4, F5, F6, F9, F10 | knee speed 0.754 | 0.754 | 0.569 | 0.568 c | 0.529 s | 0.529 s | 0.553 c | 0.574 c | 0.668 / 0.677 | 0.357 / 0.312 | 0.519 / 0.418 | within |
| V6 | F1, F2, F3, F4, F5, F6, F9, F10, F14 | knee speed 0.757 | 0.757 | 0.569 | 0.566 c | 0.529 s | 0.557 s | 0.550 c | 0.572 c | 0.668 / 0.673 | 0.358 / 0.314 | 0.573 / 0.464 | within |
| V6S | F1, F2, F3, F4, F5, F6, F9, F10, F14, F7 | knee speed 0.763 | 0.763 | 0.560 | 0.567 c | 0.522 s | 0.550 s | 0.540 c | 0.584 c | 0.668 / 0.673 | 0.334 / 0.330 | 0.539 / 0.462 | within |

### 75-M-H-10, 0.40 m/s

| Rung | Features | Governing | Knee s | Knee c | Hip pitch | Ankle A | Ankle B | Hip roll | Thigh twist | Neck c | Tail c | Tail p | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | today | knee speed 0.804 | 0.804 | 0.539 | 0.716 s | 0.603 s | 0.603 s | 0.633 c | 0.674 c | 0.667 / 0.684 | 0.557 / 0.978 | 0.728 / 0.973 | within |
| V1B50 | F1, F2, F1 at 50 % | knee speed 0.792 | 0.792 | 0.551 | 0.696 s | 0.602 s | 0.602 s | 0.627 c | 0.686 c | 0.670 / 0.691 | 0.558 / 0.976 | 0.734 / 0.988 | within |
| V1 | F1, F2 | knee speed 0.786 | 0.786 | 0.565 | 0.687 s | 0.602 s | 0.602 s | 0.624 c | 0.693 c | 0.674 / 0.697 | 0.559 / 0.975 | 0.738 / 0.995 | within |
| V1B100 | F1, F2, F1 at 100 % | knee speed 0.780 | 0.780 | 0.582 | 0.679 s | 0.601 s | 0.601 s | 0.621 c | 0.700 c | 0.680 / 0.704 | 0.561 / 0.974 | 0.743 / 1.002 | within |
| V2 | F1, F2, F3 | knee speed 0.825 | 0.825 | 0.570 | 0.719 s | 0.625 s | 0.625 s | 0.620 c | 0.693 c | 0.675 / 0.698 | 0.562 / 0.971 | 0.744 / 0.995 | within |
| V3 | F1, F2, F3, F4, F5 | tail peak 0.823 | 0.724 | 0.586 | 0.643 c | 0.579 s | 0.579 s | 0.617 c | 0.771 p | 0.676 / 0.698 | 0.557 / 0.968 | 0.823 / 1.080 | within |
| V3T | F1, F2, F3, F4, F5, F5 at 0.40 rad (the research's middle) | tail peak 0.820 | 0.724 | 0.593 | 0.642 c | 0.579 s | 0.579 s | 0.616 c | 0.759 p | 0.676 / 0.698 | 0.556 / 0.968 | 0.820 / 1.077 | within |
| V4 | F1, F2, F3, F4, F5, F6 | tail peak 0.846 | 0.725 | 0.586 | 0.643 c | 0.579 s | 0.579 s | 0.612 c | 0.769 p | 0.674 / 0.697 | 0.593 / 1.014 | 0.846 / 1.105 | within |
| V4L | F1, F2, F3, F4, F5, F6 (swing side drops) | tail peak 0.800 | 0.722 | 0.587 | 0.643 c | 0.579 s | 0.579 s | 0.622 c | 0.773 p | 0.677 / 0.700 | 0.524 / 0.923 | 0.800 / 1.056 | within |
| V5 | F1, F2, F3, F4, F5, F6, F9, F10 | thigh twist peak 0.741 | 0.720 | 0.590 | 0.647 c | 0.579 s | 0.579 s | 0.618 c | 0.741 p | 0.673 / 0.675 | 0.388 / 0.391 | 0.571 / 0.431 | within |
| V6 | F1, F2, F3, F4, F5, F6, F9, F10, F14 | knee speed 0.752 | 0.752 | 0.590 | 0.646 c | 0.592 s | 0.613 s | 0.615 c | 0.747 p | 0.673 / 0.674 | 0.390 / 0.394 | 0.637 / 0.537 | within |
| V6S | F1, F2, F3, F4, F5, F6, F9, F10, F14, F7 | thigh twist peak 0.756 | 0.754 | 0.583 | 0.647 c | 0.594 s | 0.613 s | 0.608 c | 0.756 p | 0.673 / 0.674 | 0.369 / 0.412 | 0.607 / 0.560 | within |

### 75-M-H-10, animal pace (Fr 0.05)

| Rung | Features | Governing | Knee s | Knee c | Hip pitch | Ankle A | Ankle B | Hip roll | Thigh twist | Neck c | Tail c | Tail p | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | today | knee speed 0.935 | 0.935 | 0.564 | 0.834 s | 0.705 s | 0.705 s | 0.654 c | 0.709 c | 0.669 / 0.689 | 0.602 / 1.030 | 0.793 / 1.040 | within |
| V1B50 | F1, F2, F1 at 50 % | knee speed 0.909 | 0.909 | 0.579 | 0.809 s | 0.702 s | 0.702 s | 0.645 c | 0.729 c | 0.675 / 0.699 | 0.607 / 1.029 | 0.818 / 1.063 | within |
| V1 | F1, F2 | knee speed 0.896 | 0.896 | 0.601 | 0.799 s | 0.701 s | 0.701 s | 0.642 c | 0.742 c | 0.682 / 0.709 | 0.611 / 1.029 | 0.830 / 1.074 | within |
| V1B100 | F1, F2, F1 at 100 % | knee speed 0.883 | 0.883 | 0.631 | 0.789 s | 0.700 s | 0.700 s | 0.639 c | 0.756 c | 0.692 / 0.722 | 0.615 / 1.029 | 0.850 / 1.086 | within |
| V2 | F1, F2, F3 | knee speed 0.896 | 0.896 | 0.601 | 0.799 s | 0.701 s | 0.701 s | 0.642 c | 0.742 c | 0.682 / 0.709 | 0.611 / 1.029 | 0.830 / 1.074 | within |
| V3 | F1, F2, F3, F4, F5 | tail peak 0.886 | 0.783 | 0.613 | 0.681 c | 0.655 s | 0.655 s | 0.634 c | 0.820 p | 0.682 / 0.709 | 0.607 / 1.025 | 0.886 / 1.155 | within |
| V3T | F1, F2, F3, F4, F5, F5 at 0.40 rad (the research's middle) | tail peak 0.885 | 0.783 | 0.617 | 0.680 c | 0.655 s | 0.655 s | 0.633 c | 0.815 p | 0.682 / 0.709 | 0.606 / 1.025 | 0.885 / 1.154 | within |
| V4 | F1, F2, F3, F4, F5, F6 | tail peak 0.907 | 0.784 | 0.613 | 0.681 c | 0.655 s | 0.655 s | 0.629 c | 0.818 p | 0.680 / 0.707 | 0.641 / 1.071 | 0.907 / 1.178 | within |
| V4L | F1, F2, F3, F4, F5, F6 (swing side drops) | tail peak 0.866 | 0.781 | 0.614 | 0.681 c | 0.654 s | 0.655 s | 0.640 c | 0.821 p | 0.684 / 0.710 | 0.575 / 0.980 | 0.866 / 1.132 | within |
| V5 | F1, F2, F3, F4, F5, F6, F9, F10 | thigh twist peak 0.794 | 0.771 | 0.619 | 0.693 c | 0.654 s | 0.654 s | 0.635 c | 0.794 p | 0.682 / 0.688 | 0.439 / 0.426 | 0.642 / 0.471 | within |
| V5F13 | F1, F2, F3, F4, F5, F6, F9, F10, F13 | knee speed 0.957 | 0.957 | 0.632 | 0.697 c | 0.730 s | 0.730 s | 0.629 c | 0.772 p | 0.679 / 0.684 | 0.367 / 0.386 | 0.589 / 0.407 | within |
| V6 | F1, F2, F3, F4, F5, F6, F9, F10, F13, F14 | knee speed 1.012 | 1.012 | 0.631 | 0.697 c | 0.748 s | 0.773 s | 0.626 c | 0.775 p | 0.679 / 0.680 | 0.369 / 0.388 | 0.645 / 0.508 | **over**: knee speed 1.012 |
| V6S | F1, F2, F3, F4, F5, F6, F9, F10, F13, F14, F7 | knee speed 1.009 | 1.009 | 0.624 | 0.696 c | 0.750 s | 0.773 s | 0.619 c | 0.784 p | 0.679 / 0.680 | 0.348 / 0.406 | 0.605 / 0.531 | **over**: knee speed 1.009 |

## 3b. Neck and tail with the plant's joint damping and friction (research Bottom line 3)

Stage 0's inverse-dynamics torque (ID) against the servo's torque with the plant's damping and friction loss (total = ID + b omega + f sign omega). RMS and peak in N m; c and p: ratios to the sustained (A1) and planning peak (x 1.25). Power: the servo's mean power on the ID torque (W; negative: the body's motion drives the joint and the servo brakes it, so the damping helps).

### 60-N-T-10, nominal (Fr 0.012)

| Rung | Neck RMS ID / total | Neck c | Neck p | Tail RMS ID / passive alone / total | Tail c | Tail p | Tail power (W) |
|---|---|---|---|---|---|---|---|
| V0 | 0.624 / 0.670 | 0.761 / 0.818 | 0.601 / 0.805 | 0.119 / 0.248 / 0.336 | 0.242 / 0.684 | 0.313 / 0.569 | +0.038 |
| V1B50 | 0.624 / 0.673 | 0.763 / 0.821 | 0.660 / 0.881 | 0.118 / 0.248 / 0.335 | 0.241 / 0.683 | 0.315 / 0.568 | +0.037 |
| V1 | 0.625 / 0.675 | 0.764 / 0.824 | 0.700 / 0.922 | 0.118 / 0.248 / 0.335 | 0.241 / 0.682 | 0.317 / 0.568 | +0.037 |
| V1B100 | 0.627 / 0.677 | 0.766 / 0.827 | 0.741 / 0.962 | 0.118 / 0.248 / 0.335 | 0.241 / 0.682 | 0.318 / 0.568 | +0.037 |
| V2 | 0.626 / 0.677 | 0.765 / 0.827 | 0.729 / 0.868 | 0.119 / 0.248 / 0.323 | 0.243 / 0.658 | 0.310 / 0.598 | +0.031 |
| V3 | 0.626 / 0.676 | 0.765 / 0.825 | 0.729 / 0.868 | 0.117 / 0.248 / 0.322 | 0.238 / 0.656 | 0.334 / 0.603 | +0.031 |
| V3T | 0.626 / 0.675 | 0.765 / 0.825 | 0.730 / 0.868 | 0.117 / 0.248 / 0.322 | 0.238 / 0.656 | 0.326 / 0.602 | +0.031 |
| V4 | 0.625 / 0.674 | 0.763 / 0.824 | 0.729 / 0.868 | 0.128 / 0.248 / 0.334 | 0.260 / 0.680 | 0.351 / 0.620 | +0.036 |
| V4L | 0.627 / 0.677 | 0.766 / 0.826 | 0.729 / 0.868 | 0.106 / 0.248 / 0.311 | 0.216 / 0.632 | 0.317 / 0.585 | +0.025 |
| V5 | 0.626 / 0.648 | 0.764 / 0.792 | 0.785 / 0.903 | 0.071 / 0.253 / 0.200 | 0.144 / 0.408 | 0.243 / 0.355 | -0.034 |
| V6 | 0.626 / 0.640 | 0.764 / 0.782 | 0.789 / 0.909 | 0.071 / 0.253 / 0.201 | 0.145 / 0.408 | 0.258 / 0.409 | -0.034 |
| V6S | 0.626 / 0.640 | 0.765 / 0.782 | 0.789 / 0.909 | 0.064 / 0.253 / 0.206 | 0.130 / 0.420 | 0.233 / 0.423 | -0.030 |

### 60-N-T-10, 0.40 m/s

| Rung | Neck RMS ID / total | Neck c | Neck p | Tail RMS ID / passive alone / total | Tail c | Tail p | Tail power (W) |
|---|---|---|---|---|---|---|---|
| V0 | 0.629 / 0.688 | 0.768 / 0.841 | 0.646 / 0.860 | 0.171 / 0.284 / 0.408 | 0.347 / 0.830 | 0.462 / 0.747 | +0.064 |
| V1B50 | 0.632 / 0.700 | 0.772 / 0.855 | 0.720 / 0.988 | 0.171 / 0.284 / 0.407 | 0.349 / 0.828 | 0.465 / 0.752 | +0.063 |
| V1 | 0.636 / 0.708 | 0.777 / 0.865 | 0.788 / 1.057 | 0.172 / 0.284 / 0.406 | 0.350 / 0.827 | 0.467 / 0.756 | +0.063 |
| V1B100 | 0.642 / 0.717 | 0.784 / 0.876 | 0.856 / 1.125 | 0.172 / 0.284 / 0.406 | 0.351 / 0.827 | 0.470 / 0.760 | +0.062 |
| V2 | 0.637 / 0.710 | 0.777 / 0.867 | 0.794 / 1.059 | 0.172 / 0.284 / 0.406 | 0.350 / 0.826 | 0.470 / 0.754 | +0.062 |
| V3 | 0.637 / 0.709 | 0.778 / 0.866 | 0.821 / 1.086 | 0.171 / 0.284 / 0.405 | 0.349 / 0.824 | 0.531 / 0.800 | +0.062 |
| V3T | 0.637 / 0.709 | 0.778 / 0.866 | 0.830 / 1.095 | 0.171 / 0.284 / 0.405 | 0.348 / 0.824 | 0.530 / 0.799 | +0.062 |
| V4 | 0.635 / 0.708 | 0.776 / 0.864 | 0.821 / 1.086 | 0.181 / 0.284 / 0.418 | 0.369 / 0.852 | 0.544 / 0.814 | +0.071 |
| V4L | 0.638 / 0.710 | 0.780 / 0.867 | 0.821 / 1.086 | 0.162 / 0.284 / 0.392 | 0.330 / 0.797 | 0.518 / 0.786 | +0.052 |
| V5 | 0.634 / 0.650 | 0.775 / 0.794 | 0.832 / 0.976 | 0.122 / 0.290 / 0.225 | 0.248 / 0.459 | 0.382 / 0.453 | -0.064 |
| V6 | 0.634 / 0.646 | 0.775 / 0.789 | 0.833 / 0.979 | 0.122 / 0.291 / 0.226 | 0.249 / 0.460 | 0.418 / 0.536 | -0.064 |
| V6S | 0.635 / 0.646 | 0.775 / 0.789 | 0.833 / 0.979 | 0.115 / 0.291 / 0.235 | 0.233 / 0.479 | 0.392 / 0.552 | -0.056 |

### 60-N-T-10, animal pace (Fr 0.05)

| Rung | Neck RMS ID / total | Neck c | Neck p | Tail RMS ID / passive alone / total | Tail c | Tail p | Tail power (W) |
|---|---|---|---|---|---|---|---|
| V0 | 0.630 / 0.690 | 0.769 / 0.843 | 0.652 / 0.868 | 0.178 / 0.289 / 0.417 | 0.362 / 0.850 | 0.483 / 0.772 | +0.068 |
| V1B50 | 0.634 / 0.704 | 0.774 / 0.860 | 0.735 / 1.012 | 0.179 / 0.289 / 0.417 | 0.365 / 0.848 | 0.493 / 0.780 | +0.067 |
| V1 | 0.640 / 0.714 | 0.781 / 0.872 | 0.813 / 1.090 | 0.180 / 0.289 / 0.416 | 0.366 / 0.847 | 0.498 / 0.785 | +0.066 |
| V1B100 | 0.647 / 0.726 | 0.790 / 0.887 | 0.891 / 1.168 | 0.181 / 0.289 / 0.416 | 0.368 / 0.847 | 0.503 / 0.791 | +0.065 |
| V2 | 0.640 / 0.714 | 0.781 / 0.872 | 0.813 / 1.090 | 0.180 / 0.289 / 0.416 | 0.366 / 0.847 | 0.498 / 0.785 | +0.066 |
| V3 | 0.640 / 0.714 | 0.781 / 0.872 | 0.832 / 1.106 | 0.179 / 0.289 / 0.416 | 0.365 / 0.846 | 0.550 / 0.825 | +0.065 |
| V3T | 0.640 / 0.714 | 0.781 / 0.872 | 0.840 / 1.114 | 0.179 / 0.289 / 0.416 | 0.364 / 0.846 | 0.549 / 0.825 | +0.065 |
| V4 | 0.638 / 0.712 | 0.779 / 0.870 | 0.832 / 1.105 | 0.189 / 0.289 / 0.429 | 0.384 / 0.873 | 0.561 / 0.840 | +0.076 |
| V4L | 0.641 / 0.715 | 0.783 / 0.873 | 0.832 / 1.106 | 0.170 / 0.289 / 0.402 | 0.346 / 0.819 | 0.538 / 0.811 | +0.055 |
| V5 | 0.639 / 0.656 | 0.780 / 0.802 | 0.852 / 0.989 | 0.130 / 0.295 / 0.230 | 0.265 / 0.469 | 0.408 / 0.474 | -0.068 |
| V6 | 0.639 / 0.647 | 0.780 / 0.790 | 0.853 / 0.992 | 0.131 / 0.296 / 0.231 | 0.266 / 0.471 | 0.447 / 0.561 | -0.068 |
| V6S | 0.639 / 0.647 | 0.781 / 0.791 | 0.853 / 0.992 | 0.123 / 0.296 / 0.240 | 0.250 / 0.489 | 0.420 / 0.577 | -0.059 |

### 75-M-H-10, nominal (Fr 0.012)

| Rung | Neck RMS ID / total | Neck c | Neck p | Tail RMS ID / passive alone / total | Tail c | Tail p | Tail power (W) |
|---|---|---|---|---|---|---|---|
| V0 | 1.172 / 1.196 | 0.662 / 0.676 | 0.526 / 0.603 | 0.197 / 0.237 / 0.393 | 0.401 / 0.800 | 0.515 / 0.745 | +0.056 |
| V1B50 | 1.174 / 1.200 | 0.663 / 0.678 | 0.592 / 0.690 | 0.197 / 0.237 / 0.393 | 0.400 / 0.799 | 0.519 / 0.749 | +0.056 |
| V1 | 1.177 / 1.204 | 0.665 / 0.680 | 0.636 / 0.734 | 0.197 / 0.237 / 0.392 | 0.400 / 0.798 | 0.521 / 0.751 | +0.056 |
| V1B100 | 1.180 / 1.208 | 0.667 / 0.683 | 0.680 / 0.778 | 0.197 / 0.237 / 0.392 | 0.400 / 0.798 | 0.523 / 0.753 | +0.055 |
| V2 | 1.178 / 1.208 | 0.666 / 0.682 | 0.662 / 0.726 | 0.198 / 0.237 / 0.376 | 0.403 / 0.765 | 0.504 / 0.739 | +0.046 |
| V3 | 1.178 / 1.206 | 0.666 / 0.681 | 0.663 / 0.726 | 0.194 / 0.237 / 0.374 | 0.395 / 0.761 | 0.536 / 0.773 | +0.046 |
| V3T | 1.178 / 1.206 | 0.666 / 0.681 | 0.663 / 0.727 | 0.194 / 0.237 / 0.374 | 0.395 / 0.761 | 0.524 / 0.757 | +0.046 |
| V4 | 1.176 / 1.204 | 0.664 / 0.680 | 0.662 / 0.726 | 0.213 / 0.237 / 0.395 | 0.434 / 0.804 | 0.565 / 0.803 | +0.055 |
| V4L | 1.180 / 1.208 | 0.667 / 0.682 | 0.663 / 0.726 | 0.175 / 0.237 / 0.353 | 0.357 / 0.718 | 0.507 / 0.743 | +0.037 |
| V5 | 1.182 / 1.199 | 0.668 / 0.677 | 0.744 / 0.807 | 0.175 / 0.242 / 0.154 | 0.357 / 0.312 | 0.519 / 0.418 | -0.065 |
| V6 | 1.182 / 1.191 | 0.668 / 0.673 | 0.750 / 0.812 | 0.176 / 0.242 / 0.154 | 0.358 / 0.314 | 0.573 / 0.464 | -0.065 |
| V6S | 1.182 / 1.191 | 0.668 / 0.673 | 0.750 / 0.812 | 0.164 / 0.242 / 0.162 | 0.334 / 0.330 | 0.539 / 0.462 | -0.059 |

### 75-M-H-10, 0.40 m/s

| Rung | Neck RMS ID / total | Neck c | Neck p | Tail RMS ID / passive alone / total | Tail c | Tail p | Tail power (W) |
|---|---|---|---|---|---|---|---|
| V0 | 1.181 / 1.211 | 0.667 / 0.684 | 0.565 / 0.639 | 0.273 / 0.266 / 0.480 | 0.557 / 0.978 | 0.728 / 0.973 | +0.091 |
| V1B50 | 1.187 / 1.224 | 0.670 / 0.691 | 0.635 / 0.752 | 0.274 / 0.266 / 0.479 | 0.558 / 0.976 | 0.734 / 0.988 | +0.090 |
| V1 | 1.194 / 1.234 | 0.674 / 0.697 | 0.700 / 0.816 | 0.275 / 0.266 / 0.479 | 0.559 / 0.975 | 0.738 / 0.995 | +0.089 |
| V1B100 | 1.204 / 1.247 | 0.680 / 0.704 | 0.764 / 0.880 | 0.276 / 0.266 / 0.479 | 0.561 / 0.974 | 0.743 / 1.002 | +0.088 |
| V2 | 1.195 / 1.236 | 0.675 / 0.698 | 0.711 / 0.824 | 0.276 / 0.266 / 0.477 | 0.562 / 0.971 | 0.744 / 0.995 | +0.087 |
| V3 | 1.196 / 1.236 | 0.676 / 0.698 | 0.737 / 0.850 | 0.274 / 0.266 / 0.476 | 0.557 / 0.968 | 0.823 / 1.080 | +0.087 |
| V3T | 1.196 / 1.236 | 0.676 / 0.698 | 0.745 / 0.858 | 0.273 / 0.266 / 0.476 | 0.556 / 0.968 | 0.820 / 1.077 | +0.087 |
| V4 | 1.193 / 1.233 | 0.674 / 0.697 | 0.737 / 0.850 | 0.292 / 0.266 / 0.498 | 0.593 / 1.014 | 0.846 / 1.105 | +0.101 |
| V4L | 1.198 / 1.238 | 0.677 / 0.700 | 0.737 / 0.850 | 0.257 / 0.266 / 0.453 | 0.524 / 0.923 | 0.800 / 1.056 | +0.073 |
| V5 | 1.192 / 1.195 | 0.673 / 0.675 | 0.723 / 0.771 | 0.191 / 0.272 / 0.192 | 0.388 / 0.391 | 0.571 / 0.431 | -0.092 |
| V6 | 1.192 / 1.193 | 0.673 / 0.674 | 0.725 / 0.772 | 0.192 / 0.272 / 0.194 | 0.390 / 0.394 | 0.637 / 0.537 | -0.092 |
| V6S | 1.192 / 1.193 | 0.673 / 0.674 | 0.725 / 0.772 | 0.181 / 0.272 / 0.202 | 0.369 / 0.412 | 0.607 / 0.560 | -0.083 |

### 75-M-H-10, animal pace (Fr 0.05)

| Rung | Neck RMS ID / total | Neck c | Neck p | Tail RMS ID / passive alone / total | Tail c | Tail p | Tail power (W) |
|---|---|---|---|---|---|---|---|
| V0 | 1.185 / 1.219 | 0.669 / 0.689 | 0.574 / 0.651 | 0.296 / 0.274 / 0.506 | 0.602 / 1.030 | 0.793 / 1.040 | +0.102 |
| V1B50 | 1.194 / 1.237 | 0.675 / 0.699 | 0.671 / 0.792 | 0.298 / 0.274 / 0.505 | 0.607 / 1.029 | 0.818 / 1.063 | +0.100 |
| V1 | 1.207 / 1.255 | 0.682 / 0.709 | 0.754 / 0.876 | 0.300 / 0.274 / 0.505 | 0.611 / 1.029 | 0.830 / 1.074 | +0.098 |
| V1B100 | 1.226 / 1.278 | 0.692 / 0.722 | 0.838 / 0.959 | 0.302 / 0.274 / 0.505 | 0.615 / 1.029 | 0.850 / 1.086 | +0.097 |
| V2 | 1.207 / 1.255 | 0.682 / 0.709 | 0.754 / 0.876 | 0.300 / 0.274 / 0.505 | 0.611 / 1.029 | 0.830 / 1.074 | +0.098 |
| V3 | 1.208 / 1.255 | 0.682 / 0.709 | 0.767 / 0.888 | 0.298 / 0.274 / 0.504 | 0.607 / 1.025 | 0.886 / 1.155 | +0.098 |
| V3T | 1.208 / 1.255 | 0.682 / 0.709 | 0.771 / 0.892 | 0.298 / 0.274 / 0.503 | 0.606 / 1.025 | 0.885 / 1.154 | +0.098 |
| V4 | 1.204 / 1.252 | 0.680 / 0.707 | 0.767 / 0.888 | 0.315 / 0.274 / 0.526 | 0.641 / 1.071 | 0.907 / 1.178 | +0.113 |
| V4L | 1.210 / 1.257 | 0.684 / 0.710 | 0.767 / 0.888 | 0.282 / 0.274 / 0.482 | 0.575 / 0.980 | 0.866 / 1.132 | +0.082 |
| V5 | 1.208 / 1.218 | 0.682 / 0.688 | 0.756 / 0.822 | 0.216 / 0.280 / 0.209 | 0.439 / 0.426 | 0.642 / 0.471 | -0.102 |
| V5F13 | 1.202 / 1.211 | 0.679 / 0.684 | 0.797 / 0.759 | 0.180 / 0.269 / 0.189 | 0.367 / 0.386 | 0.589 / 0.407 | -0.085 |
| V6 | 1.201 / 1.203 | 0.679 / 0.680 | 0.801 / 0.761 | 0.181 / 0.269 / 0.191 | 0.369 / 0.388 | 0.645 / 0.508 | -0.085 |
| V6S | 1.202 / 1.204 | 0.679 / 0.680 | 0.802 / 0.761 | 0.171 / 0.269 / 0.199 | 0.348 / 0.406 | 0.605 / 0.531 | -0.077 |

## 4. Balance, grip, sway, accelerations and the walking clearance gates

SS CoP: the single-support CoP's margin inside the stance pad (mm; envelope rule over -10). ZMP: the multibody ZMP's margin inside the support (mm). mu_req and peak |Mz| (N m): the straight-run reading (section 2); DS: the double-support requirement. Sway (H, peak to peak; spec 0.12): Stage 0's (pendulum CoM) and the model CoM's largest per stride. Accelerations (m/s^2): the CoM's horizontal peak, forward part at 1 Hz, vertical range; the feet's vertical peak. Swing: the lowest pad point over the middle half of each swing (H; gate 0.04). Joint: the tightest margin inside the usable ranges (rad; gate 0.10).

### 60-N-T-10, nominal (Fr 0.012)

| Rung | SS CoP | ZMP | mu_req | Peak \|Mz\| | DS mu | Sway S0 | Sway CoM | a horiz | a fwd 1 Hz | a vert | a foot | Swing (H) | Joint | Gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 30.3 | 30.3 | 0.481 | 1.77 | 0.203 | 0.168 | 0.168 | 2.21 | 0.03 | -0.89 / +0.91 | 4.8 | 0.0500 | 0.182 (left_hip_roll) | pass |
| V1B50 | 30.7 | 30.7 | 0.461 | 1.73 | 0.218 | 0.167 | 0.167 | 2.15 | 0.03 | -1.05 / +1.10 | 4.8 | 0.0500 | 0.182 (left_hip_roll) | pass |
| V1 | 30.7 | 30.7 | 0.452 | 1.70 | 0.227 | 0.167 | 0.167 | 2.18 | 0.03 | -1.56 / +1.67 | 4.8 | 0.0500 | 0.183 (left_hip_roll) | pass |
| V1B100 | 30.8 | 30.8 | 0.445 | 1.69 | 0.237 | 0.167 | 0.167 | 2.21 | 0.03 | -2.07 / +2.25 | 4.8 | 0.0500 | 0.183 (left_hip_roll) | pass |
| V2 | 31.1 | 31.1 | 0.728 | 2.85 | 0.183 | 0.161 | 0.161 | 2.17 | 0.02 | -1.70 / +1.67 | 7.8 | 0.0476 | 0.188 (left_hip_roll) | pass |
| V3 | 31.1 | 7.0 | 0.527 | 2.06 | 0.182 | 0.162 | 0.162 | 2.21 | 0.02 | -2.09 / +2.34 | 18.3 | 0.0440 | 0.187 (left_hip_roll) | pass |
| V3T | 31.1 | 7.0 | 0.505 | 1.69 | 0.182 | 0.162 | 0.162 | 2.21 | 0.03 | -2.86 / +3.53 | 27.3 | 0.0440 | 0.187 (left_hip_roll) | pass |
| V4 | 31.1 | 7.0 | 0.525 | 2.06 | 0.183 | 0.162 | 0.162 | 2.21 | 0.03 | -2.09 / +2.34 | 18.3 | 0.0440 | 0.216 (left_hip_roll) | pass |
| V4L | 31.1 | 7.0 | 0.528 | 2.07 | 0.182 | 0.162 | 0.162 | 2.21 | 0.02 | -2.09 / +2.34 | 18.3 | 0.0440 | 0.159 (left_hip_roll) | pass |
| V5 | 31.1 | 7.0 | 0.499 | 1.72 | 0.178 | 0.162 | 0.162 | 2.18 | 0.02 | -2.09 / +2.24 | 18.3 | 0.0440 | 0.212 (left_hip_roll) | pass |
| V6 | 31.1 | 5.2 | 0.559 | 1.81 | 0.213 | 0.164 | 0.163 | 2.27 | 0.04 | -2.12 / +2.28 | 18.6 | 0.0414 | 0.209 (left_hip_roll) | pass |
| V6S | 22.0 | 5.2 | 0.530 | 1.73 | 0.198 | 0.142 | 0.141 | 2.09 | 0.04 | -2.12 / +2.29 | 18.6 | 0.0414 | 0.226 (left_hip_roll) | pass |

### 60-N-T-10, 0.40 m/s

| Rung | SS CoP | ZMP | mu_req | Peak \|Mz\| | DS mu | Sway S0 | Sway CoM | a horiz | a fwd 1 Hz | a vert | a foot | Swing (H) | Joint | Gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 29.9 | 29.9 | 0.964 | 3.74 | 0.219 | 0.108 | 0.108 | 3.05 | 0.01 | -1.75 / +1.73 | 8.3 | 0.0548 | 0.225 (left_hip_roll) | pass |
| V1B50 | 30.3 | 30.3 | 0.858 | 3.56 | 0.221 | 0.107 | 0.107 | 3.03 | 0.01 | -1.74 / +1.79 | 8.3 | 0.0548 | 0.226 (left_hip_roll) | pass |
| V1 | 30.3 | 30.3 | 0.817 | 3.50 | 0.224 | 0.107 | 0.107 | 3.08 | 0.01 | -1.94 / +2.74 | 8.3 | 0.0548 | 0.226 (left_hip_roll) | pass |
| V1B100 | 30.2 | 30.2 | 0.780 | 3.44 | 0.227 | 0.107 | 0.107 | 3.13 | 0.01 | -2.27 / +3.69 | 8.3 | 0.0548 | 0.227 (left_hip_roll) | pass |
| V2 | 30.5 | 30.5 | 0.850 | 3.63 | 0.217 | 0.107 | 0.107 | 2.95 | 0.01 | -1.94 / +2.74 | 8.6 | 0.0554 | 0.227 (left_hip_roll) | pass |
| V3 | 30.4 | 5.8 | 0.979 | 4.63 | 0.217 | 0.107 | 0.107 | 2.96 | 0.01 | -1.69 / +4.46 | 19.0 | 0.0510 | 0.226 (left_hip_roll) | pass |
| V3T | 30.3 | 5.7 | 0.965 | 4.55 | 0.216 | 0.107 | 0.107 | 2.96 | 0.01 | -1.67 / +5.01 | 20.3 | 0.0510 | 0.226 (left_hip_roll) | pass |
| V4 | 30.4 | 5.8 | 0.978 | 4.62 | 0.216 | 0.107 | 0.107 | 2.96 | 0.01 | -1.69 / +4.46 | 19.0 | 0.0510 | 0.255 (left_hip_roll) | pass |
| V4L | 30.3 | 5.8 | 0.979 | 4.63 | 0.216 | 0.107 | 0.107 | 2.96 | 0.01 | -1.69 / +4.46 | 19.0 | 0.0510 | 0.197 (left_hip_roll) | pass |
| V5 | 30.4 | 5.8 | 0.867 | 4.11 | 0.210 | 0.107 | 0.107 | 2.86 | 0.01 | -1.62 / +4.40 | 19.0 | 0.0510 | 0.252 (left_hip_roll) | pass |
| V6 | 30.4 | 5.5 | 0.875 | 4.16 | 0.263 | 0.110 | 0.108 | 2.93 | 0.03 | -1.72 / +4.45 | 19.1 | 0.0473 | 0.247 (left_hip_roll) | pass |
| V6S | 21.4 | 5.4 | 0.842 | 4.01 | 0.250 | 0.094 | 0.093 | 2.75 | 0.03 | -1.71 / +4.46 | 19.1 | 0.0473 | 0.259 (left_hip_roll) | pass |

### 60-N-T-10, animal pace (Fr 0.05)

| Rung | SS CoP | ZMP | mu_req | Peak \|Mz\| | DS mu | Sway S0 | Sway CoM | a horiz | a fwd 1 Hz | a vert | a foot | Swing (H) | Joint | Gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 29.7 | 29.7 | 1.092 | 4.23 | 0.229 | 0.103 | 0.103 | 3.25 | 0.01 | -1.89 / +1.89 | 8.6 | 0.0540 | 0.229 (left_hip_roll) | pass |
| V1B50 | 30.0 | 30.0 | 0.946 | 4.00 | 0.233 | 0.102 | 0.102 | 3.46 | 0.01 | -1.88 / +2.02 | 8.6 | 0.0540 | 0.230 (left_hip_roll) | pass |
| V1 | 29.9 | 29.9 | 0.889 | 3.91 | 0.236 | 0.101 | 0.101 | 3.66 | 0.01 | -2.21 / +3.11 | 8.6 | 0.0540 | 0.231 (left_hip_roll) | pass |
| V1B100 | 29.7 | 29.7 | 0.841 | 3.83 | 0.239 | 0.101 | 0.101 | 3.82 | 0.01 | -2.62 / +4.20 | 8.6 | 0.0540 | 0.231 (left_hip_roll) | pass |
| V2 | 29.9 | 29.9 | 0.889 | 3.91 | 0.236 | 0.101 | 0.101 | 3.66 | 0.01 | -2.21 / +3.11 | 8.6 | 0.0540 | 0.231 (left_hip_roll) | pass |
| V3 | 29.7 | 4.9 | 1.051 | 4.96 | 0.236 | 0.101 | 0.101 | 3.59 | 0.01 | -1.97 / +4.61 | 19.1 | 0.0498 | 0.230 (left_hip_roll) | pass |
| V3T | 29.7 | 4.8 | 1.041 | 4.91 | 0.236 | 0.101 | 0.101 | 3.57 | 0.01 | -1.96 / +5.13 | 19.8 | 0.0498 | 0.230 (left_hip_roll) | pass |
| V4 | 29.7 | 4.9 | 1.050 | 4.95 | 0.236 | 0.101 | 0.101 | 3.59 | 0.01 | -1.97 / +4.61 | 19.1 | 0.0498 | 0.259 (left_hip_roll) | pass |
| V4L | 29.7 | 4.9 | 1.051 | 4.96 | 0.236 | 0.102 | 0.102 | 3.59 | 0.01 | -1.97 / +4.61 | 19.1 | 0.0498 | 0.201 (left_hip_roll) | pass |
| V5 | 29.7 | 4.9 | 0.929 | 4.40 | 0.233 | 0.101 | 0.101 | 3.58 | 0.01 | -1.90 / +4.53 | 19.1 | 0.0498 | 0.256 (left_hip_roll) | pass |
| V6 | 29.7 | 4.9 | 0.938 | 4.45 | 0.281 | 0.104 | 0.103 | 3.58 | 0.03 | -2.04 / +4.57 | 19.3 | 0.0472 | 0.251 (left_hip_roll) | pass |
| V6S | 20.7 | 5.0 | 0.903 | 4.30 | 0.268 | 0.090 | 0.088 | 3.38 | 0.03 | -2.04 / +4.58 | 19.3 | 0.0472 | 0.262 (left_hip_roll) | pass |

### 75-M-H-10, nominal (Fr 0.012)

| Rung | SS CoP | ZMP | mu_req | Peak \|Mz\| | DS mu | Sway S0 | Sway CoM | a horiz | a fwd 1 Hz | a vert | a foot | Swing (H) | Joint | Gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 45.3 | 45.3 | 0.361 | 2.48 | 0.198 | 0.164 | 0.164 | 2.14 | 0.05 | -0.82 / +0.84 | 4.8 | 0.0500 | 0.169 (left_hip_roll) | pass |
| V1B50 | 45.6 | 45.6 | 0.348 | 2.42 | 0.214 | 0.164 | 0.164 | 2.10 | 0.05 | -1.51 / +1.45 | 4.8 | 0.0500 | 0.169 (left_hip_roll) | pass |
| V1 | 45.6 | 45.6 | 0.341 | 2.39 | 0.224 | 0.163 | 0.163 | 2.11 | 0.05 | -2.26 / +2.20 | 4.8 | 0.0500 | 0.170 (left_hip_roll) | pass |
| V1B100 | 45.6 | 45.6 | 0.336 | 2.37 | 0.235 | 0.163 | 0.163 | 2.13 | 0.05 | -3.01 / +2.96 | 4.8 | 0.0500 | 0.170 (left_hip_roll) | pass |
| V2 | 46.0 | 46.0 | 0.530 | 3.94 | 0.175 | 0.158 | 0.158 | 2.14 | 0.04 | -2.22 / +2.20 | 7.8 | 0.0476 | 0.174 (left_hip_roll) | pass |
| V3 | 46.0 | 8.4 | 0.381 | 2.74 | 0.175 | 0.159 | 0.159 | 2.19 | 0.04 | -2.22 / +2.40 | 21.4 | 0.0440 | 0.174 (left_hip_roll) | pass |
| V3T | 46.0 | 8.4 | 0.388 | 2.53 | 0.174 | 0.159 | 0.159 | 2.20 | 0.04 | -2.68 / +3.45 | 31.2 | 0.0440 | 0.174 (left_hip_roll) | pass |
| V4 | 46.0 | 8.4 | 0.379 | 2.73 | 0.174 | 0.158 | 0.158 | 2.21 | 0.04 | -2.22 / +2.40 | 21.4 | 0.0440 | 0.203 (left_hip_roll) | pass |
| V4L | 46.0 | 8.4 | 0.383 | 2.75 | 0.174 | 0.159 | 0.159 | 2.21 | 0.04 | -2.22 / +2.40 | 21.4 | 0.0440 | 0.145 (left_hip_roll) | pass |
| V5 | 46.0 | 8.4 | 0.360 | 2.26 | 0.172 | 0.158 | 0.158 | 2.17 | 0.04 | -2.36 / +2.28 | 21.4 | 0.0440 | 0.201 (left_hip_roll) | pass |
| V6 | 46.0 | 5.2 | 0.405 | 2.50 | 0.197 | 0.161 | 0.160 | 2.25 | 0.05 | -2.52 / +2.36 | 21.5 | 0.0421 | 0.198 (left_hip_roll) | pass |
| V6S | 36.9 | 5.2 | 0.385 | 2.39 | 0.186 | 0.142 | 0.141 | 2.10 | 0.05 | -2.52 / +2.37 | 21.5 | 0.0421 | 0.211 (left_hip_roll) | pass |

### 75-M-H-10, 0.40 m/s

| Rung | SS CoP | ZMP | mu_req | Peak \|Mz\| | DS mu | Sway S0 | Sway CoM | a horiz | a fwd 1 Hz | a vert | a foot | Swing (H) | Joint | Gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 44.9 | 44.9 | 0.629 | 4.63 | 0.206 | 0.111 | 0.111 | 2.84 | 0.02 | -1.49 / +1.51 | 8.1 | 0.0561 | 0.205 (left_hip_roll) | pass |
| V1B50 | 45.2 | 45.2 | 0.575 | 4.46 | 0.217 | 0.110 | 0.110 | 2.64 | 0.02 | -1.48 / +2.06 | 8.1 | 0.0561 | 0.206 (left_hip_roll) | pass |
| V1 | 45.2 | 45.2 | 0.552 | 4.39 | 0.223 | 0.110 | 0.110 | 2.60 | 0.02 | -1.73 / +3.13 | 8.1 | 0.0561 | 0.206 (left_hip_roll) | pass |
| V1B100 | 45.3 | 45.3 | 0.533 | 4.33 | 0.230 | 0.110 | 0.110 | 2.62 | 0.02 | -2.81 / +4.21 | 8.1 | 0.0561 | 0.207 (left_hip_roll) | pass |
| V2 | 45.3 | 45.3 | 0.590 | 4.70 | 0.221 | 0.110 | 0.110 | 2.60 | 0.02 | -1.67 / +3.13 | 8.6 | 0.0532 | 0.207 (left_hip_roll) | pass |
| V3 | 45.2 | 5.8 | 0.670 | 5.86 | 0.221 | 0.110 | 0.110 | 2.56 | 0.02 | -1.80 / +5.15 | 21.4 | 0.0490 | 0.207 (left_hip_roll) | pass |
| V3T | 45.1 | 5.7 | 0.652 | 5.67 | 0.220 | 0.110 | 0.110 | 2.56 | 0.02 | -2.61 / +5.80 | 29.6 | 0.0490 | 0.207 (left_hip_roll) | pass |
| V4 | 45.2 | 5.8 | 0.669 | 5.85 | 0.221 | 0.110 | 0.110 | 2.56 | 0.02 | -1.80 / +5.15 | 21.4 | 0.0490 | 0.236 (left_hip_roll) | pass |
| V4L | 45.2 | 5.8 | 0.672 | 5.87 | 0.221 | 0.110 | 0.110 | 2.56 | 0.02 | -1.80 / +5.15 | 21.4 | 0.0490 | 0.178 (left_hip_roll) | pass |
| V5 | 45.2 | 5.8 | 0.576 | 4.99 | 0.212 | 0.110 | 0.110 | 2.48 | 0.02 | -1.70 / +5.12 | 21.4 | 0.0490 | 0.233 (left_hip_roll) | pass |
| V6 | 45.2 | 4.7 | 0.585 | 5.09 | 0.272 | 0.112 | 0.111 | 2.51 | 0.04 | -2.02 / +5.28 | 23.7 | 0.0472 | 0.228 (left_hip_roll) | pass |
| V6S | 36.2 | 4.7 | 0.564 | 4.94 | 0.261 | 0.099 | 0.098 | 2.41 | 0.04 | -2.02 / +5.28 | 23.7 | 0.0472 | 0.238 (left_hip_roll) | pass |

### 75-M-H-10, animal pace (Fr 0.05)

| Rung | SS CoP | ZMP | mu_req | Peak \|Mz\| | DS mu | Sway S0 | Sway CoM | a horiz | a fwd 1 Hz | a vert | a foot | Swing (H) | Joint | Gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 44.2 | 44.2 | 0.787 | 5.83 | 0.223 | 0.101 | 0.101 | 3.16 | 0.02 | -1.75 / +1.85 | 8.6 | 0.0540 | 0.213 (left_hip_roll) | pass |
| V1B50 | 44.4 | 44.4 | 0.687 | 5.55 | 0.234 | 0.100 | 0.100 | 3.45 | 0.02 | -1.77 / +2.63 | 8.6 | 0.0540 | 0.214 (left_hip_roll) | pass |
| V1 | 44.2 | 44.2 | 0.647 | 5.44 | 0.240 | 0.100 | 0.100 | 3.70 | 0.02 | -2.43 / +4.03 | 8.6 | 0.0540 | 0.214 (left_hip_roll) | pass |
| V1B100 | 43.9 | 43.9 | 0.612 | 5.34 | 0.247 | 0.099 | 0.099 | 3.95 | 0.02 | -3.86 / +5.43 | 8.6 | 0.0540 | 0.214 (left_hip_roll) | pass |
| V2 | 44.2 | 44.2 | 0.647 | 5.44 | 0.240 | 0.100 | 0.100 | 3.70 | 0.02 | -2.43 / +4.03 | 8.6 | 0.0540 | 0.214 (left_hip_roll) | pass |
| V3 | 44.0 | 6.2 | 0.768 | 6.82 | 0.240 | 0.100 | 0.100 | 3.70 | 0.02 | -2.21 / +5.58 | 21.8 | 0.0498 | 0.214 (left_hip_roll) | pass |
| V3T | 44.0 | 6.2 | 0.759 | 6.72 | 0.240 | 0.100 | 0.100 | 3.69 | 0.02 | -2.20 / +6.19 | 28.7 | 0.0498 | 0.214 (left_hip_roll) | pass |
| V4 | 44.1 | 6.2 | 0.767 | 6.81 | 0.240 | 0.099 | 0.099 | 3.70 | 0.02 | -2.21 / +5.58 | 21.8 | 0.0498 | 0.242 (left_hip_roll) | pass |
| V4L | 44.0 | 6.2 | 0.769 | 6.82 | 0.240 | 0.100 | 0.100 | 3.67 | 0.02 | -2.20 / +5.57 | 21.8 | 0.0498 | 0.185 (left_hip_roll) | pass |
| V5 | 44.1 | 6.1 | 0.667 | 5.95 | 0.232 | 0.100 | 0.100 | 3.64 | 0.02 | -2.33 / +5.48 | 21.8 | 0.0498 | 0.239 (left_hip_roll) | pass |
| V5F13 | 45.2 | 8.6 | 0.678 | 5.88 | 0.239 | 0.114 | 0.114 | 2.97 | 0.02 | -3.02 / +5.92 | 25.3 | 0.0518 | 0.229 (left_hip_roll) | pass |
| V6 | 45.2 | 5.0 | 0.683 | 5.93 | 0.300 | 0.116 | 0.115 | 3.04 | 0.05 | -3.06 / +5.95 | 25.9 | 0.0469 | 0.225 (left_hip_roll) | pass |
| V6S | 36.2 | 5.0 | 0.662 | 5.77 | 0.292 | 0.103 | 0.102 | 2.90 | 0.05 | -3.04 / +5.95 | 25.9 | 0.0469 | 0.235 (left_hip_roll) | pass |

## 5. Over-limit and near-limit rungs: the plan's rule

Rule: over a drive limit (or a Stage 0 gate), a variant keeps its feature and loses that speed rung (character first) unless the owner decides otherwise.

- **75-M-H-10, animal pace (Fr 0.05), V6** (features F1, F2, F3, F4, F5, F6, F9, F10, F13, F14): over (knee speed): keeps its features (F1, F2, F3, F4, F5, F6, F9, F10, F13, F14) and loses the animal pace (Froude 0.05) on 75-M-H-10 unless the owner decides otherwise (plan 4 A.3).
  - Cause (knee speed): the peak, 7.651 rad/s, comes on the right knee in swing at 0.78 of the swing; over the window's 24 swings (durations 0.345-0.366 s) the knee's peak speed runs 6.924-7.651 rad/s, correlation with the swing's duration -0.996: the fastest swing is the shortest (0.345 s).
- **75-M-H-10, animal pace (Fr 0.05), V6S** (features F1, F2, F3, F4, F5, F6, F9, F10, F13, F14, F7): over (knee speed): keeps its features (F1, F2, F3, F4, F5, F6, F9, F10, F13, F14, F7) and loses the animal pace (Froude 0.05) on 75-M-H-10 unless the owner decides otherwise (plan 4 A.3).
  - Cause (knee speed): the peak, 7.627 rad/s, comes on the right knee in swing at 0.78 of the swing; over the window's 24 swings (durations 0.345-0.366 s) the knee's peak speed runs 6.895-7.627 rad/s, correlation with the swing's duration -0.995: the fastest swing is the shortest (0.345 s).

- Near (60-N-T-10, nominal (Fr 0.012), V3T): knee speed 0.986.
- Near (75-M-H-10, nominal (Fr 0.012), V3T): knee speed 0.993.
- Over with the plant's damping and friction (60-N-T-10, 0.40 m/s, neck peak): V1 1.057, V1B100 1.125, V2 1.059, V3 1.086, V3T 1.095, V4 1.086, V4L 1.086.
- Over with the plant's damping and friction (60-N-T-10, animal pace (Fr 0.05), neck peak): V1B50 1.012, V1 1.090, V1B100 1.168, V2 1.090, V3 1.106, V3T 1.114, V4 1.105, V4L 1.106.
- Over with the plant's damping and friction (75-M-H-10, 0.40 m/s, tail peak): V1B100 1.002, V3 1.080, V3T 1.077, V4 1.105, V4L 1.056.
- Over with the plant's damping and friction (75-M-H-10, 0.40 m/s, tail cont): V4 1.014.
- Over with the plant's damping and friction (75-M-H-10, animal pace (Fr 0.05), tail cont): V0 1.030, V1B50 1.029, V1 1.029, V1B100 1.029, V2 1.029, V3 1.025, V3T 1.025, V4 1.071.
- Over with the plant's damping and friction (75-M-H-10, animal pace (Fr 0.05), tail peak): V0 1.040, V1B50 1.063, V1 1.074, V1B100 1.086, V2 1.074, V3 1.155, V3T 1.154, V4 1.178, V4L 1.132.
- Near with the plant's damping and friction (60-N-T-10, 0.40 m/s, neck peak): V1B50 0.988.
- Near with the plant's damping and friction (60-N-T-10, animal pace (Fr 0.05), neck peak): V5 0.989, V6 0.992, V6S 0.992.
- Near with the plant's damping and friction (75-M-H-10, 0.40 m/s, tail peak): V1B50 0.988, V1 0.995, V2 0.995.
- Near with the plant's damping and friction (75-M-H-10, animal pace (Fr 0.05), tail cont): V4L 0.980.

The passive-inclusive loads are outside Stage 0's rule (round two's screens left the joint damping and friction out), so they cost no rung; the plan's reading 3, 'shown but for A1', and Track D's levers 2 (an STS3250 tail) and 4 (damping and friction x0.25) are for them. They are recorded for the owner.

## 6. Changes from V0 to V6 (and to every rung: screen.json `vs_V0`)

V6 carries every feature of the ladder (F1-F6, F9, F10, F14; F13 at 75 cm's animal pace). Each cell: V0, V6 (change).

### 60-N-T-10 (60 cm Tapered)

| Number | nominal (Fr 0.012) | 0.40 m/s | animal pace (Fr 0.05) |
|---|---|---|---|
| governing ratio | 0.761, 0.789 (+0.027) | 0.865, 0.885 (+0.020) | 0.929, 0.911 (-0.018) |
| knee speed | 0.559, 0.735 (+0.177) | 0.865, 0.885 (+0.020) | 0.929, 0.911 (-0.018) |
| knee cont | 0.460, 0.469 (+0.009) | 0.478, 0.503 (+0.025) | 0.486, 0.514 (+0.028) |
| hip pitch worst | 0.525, 0.464 (-0.061) | 0.832, 0.616 (-0.215) | 0.899, 0.672 (-0.228) |
| hip pitch speed | 0.525, 0.391 (-0.134) | 0.832, 0.616 (-0.215) | 0.899, 0.672 (-0.228) |
| hip roll worst | 0.511, 0.470 (-0.040) | 0.547, 0.534 (-0.013) | 0.556, 0.541 (-0.016) |
| thigh twist worst | 0.576, 0.519 (-0.057) | 0.643, 0.720 (+0.077) | 0.661, 0.736 (+0.075) |
| ankle A worst | 0.524, 0.513 (-0.011) | 0.541, 0.543 (+0.001) | 0.545, 0.548 (+0.003) |
| ankle B worst | 0.524, 0.514 (-0.011) | 0.541, 0.542 (+0.001) | 0.545, 0.548 (+0.003) |
| neck cont | 0.761, 0.764 (+0.003) | 0.768, 0.775 (+0.007) | 0.769, 0.780 (+0.011) |
| neck cont with passive | 0.818, 0.782 (-0.036) | 0.841, 0.789 (-0.052) | 0.843, 0.790 (-0.053) |
| tail cont | 0.242, 0.145 (-0.097) | 0.347, 0.249 (-0.098) | 0.362, 0.266 (-0.096) |
| tail peak | 0.313, 0.258 (-0.055) | 0.462, 0.418 (-0.044) | 0.483, 0.447 (-0.036) |
| tail cont with passive | 0.684, 0.408 (-0.276) | 0.830, 0.460 (-0.369) | 0.850, 0.471 (-0.379) |
| tail peak with passive | 0.569, 0.409 (-0.159) | 0.747, 0.536 (-0.211) | 0.772, 0.561 (-0.211) |
| neck peak with passive | 0.805, 0.909 (+0.104) | 0.860, 0.979 (+0.120) | 0.868, 0.992 (+0.124) |
| tail power, Stage 0 (W) | 0.038, -0.034 (-0.072) | 0.064, -0.064 (-0.128) | 0.068, -0.068 (-0.136) |
| tail RMS with passive (N m) | 0.336, 0.201 (-0.135) | 0.408, 0.226 (-0.181) | 0.417, 0.231 (-0.186) |
| single-support CoP margin (mm) | 30.3, 31.1 (+0.8) | 29.9, 30.4 (+0.5) | 29.7, 29.7 (-0.0) |
| ZMP margin (mm) | 30.3, 5.2 (-25.2) | 29.9, 5.5 (-24.4) | 29.7, 4.9 (-24.8) |
| friction need mu_req | 0.481, 0.559 (+0.078) | 0.964, 0.875 (-0.089) | 1.092, 0.938 (-0.154) |
| stance-pad peak \|Mz\| (N m) | 1.77, 1.81 (+0.04) | 3.74, 4.16 (+0.42) | 4.23, 4.45 (+0.22) |
| double-support min mu | 0.203, 0.213 (+0.009) | 0.219, 0.263 (+0.044) | 0.229, 0.281 (+0.052) |
| sway Stage 0 (H) | 0.168, 0.164 (-0.003) | 0.108, 0.110 (+0.002) | 0.103, 0.104 (+0.002) |
| sway CoM per stride (H) | 0.168, 0.163 (-0.004) | 0.108, 0.108 (+0.000) | 0.103, 0.103 (+0.000) |
| CoM horizontal accel (m/s^2) | 2.21, 2.27 (+0.06) | 3.05, 2.93 (-0.13) | 3.25, 3.58 (+0.33) |
| CoM vertical accel max (m/s^2) | 0.91, 2.28 (+1.37) | 1.73, 4.45 (+2.72) | 1.89, 4.57 (+2.68) |
| CoM vertical accel min (m/s^2) | -0.89, -2.12 (-1.23) | -1.75, -1.72 (+0.03) | -1.89, -2.04 (-0.15) |
| foot vertical accel (m/s^2) | 4.8, 18.6 (+13.8) | 8.3, 19.1 (+10.8) | 8.6, 19.3 (+10.7) |
| mid-swing clearance (H) | 0.050, 0.041 (-0.009) | 0.055, 0.047 (-0.007) | 0.054, 0.047 (-0.007) |
| tightest joint margin (rad) | 0.182, 0.209 (+0.027) | 0.225, 0.247 (+0.022) | 0.229, 0.251 (+0.022) |

### 75-M-H-10 (75 cm Hybrid)

| Number | nominal (Fr 0.012) | 0.40 m/s | animal pace (Fr 0.05) |
|---|---|---|---|
| governing ratio | 0.662, 0.757 (+0.095) | 0.804, 0.752 (-0.052) | 0.935, 1.012 (+0.078) |
| knee speed | 0.555, 0.757 (+0.202) | 0.804, 0.752 (-0.052) | 0.935, 1.012 (+0.078) |
| knee cont | 0.522, 0.569 (+0.047) | 0.539, 0.590 (+0.051) | 0.564, 0.631 (+0.067) |
| hip pitch worst | 0.567, 0.566 (-0.000) | 0.716, 0.646 (-0.070) | 0.834, 0.697 (-0.137) |
| hip pitch speed | 0.488, 0.452 (-0.036) | 0.716, 0.516 (-0.200) | 0.834, 0.690 (-0.144) |
| hip roll worst | 0.606, 0.550 (-0.056) | 0.633, 0.615 (-0.017) | 0.654, 0.626 (-0.027) |
| thigh twist worst | 0.622, 0.572 (-0.051) | 0.674, 0.747 (+0.073) | 0.709, 0.775 (+0.066) |
| ankle A worst | 0.420, 0.529 (+0.108) | 0.603, 0.592 (-0.011) | 0.705, 0.748 (+0.043) |
| ankle B worst | 0.420, 0.557 (+0.136) | 0.603, 0.613 (+0.010) | 0.705, 0.773 (+0.068) |
| neck cont | 0.662, 0.668 (+0.006) | 0.667, 0.673 (+0.006) | 0.669, 0.679 (+0.009) |
| neck cont with passive | 0.676, 0.673 (-0.003) | 0.684, 0.674 (-0.010) | 0.689, 0.680 (-0.009) |
| tail cont | 0.401, 0.358 (-0.043) | 0.557, 0.390 (-0.166) | 0.602, 0.369 (-0.233) |
| tail peak | 0.515, 0.573 (+0.058) | 0.728, 0.637 (-0.091) | 0.793, 0.645 (-0.148) |
| tail cont with passive | 0.800, 0.314 (-0.486) | 0.978, 0.394 (-0.584) | 1.030, 0.388 (-0.641) |
| tail peak with passive | 0.745, 0.464 (-0.280) | 0.973, 0.537 (-0.436) | 1.040, 0.508 (-0.532) |
| neck peak with passive | 0.603, 0.812 (+0.210) | 0.639, 0.772 (+0.133) | 0.651, 0.761 (+0.110) |
| tail power, Stage 0 (W) | 0.056, -0.065 (-0.122) | 0.091, -0.092 (-0.183) | 0.102, -0.085 (-0.187) |
| tail RMS with passive (N m) | 0.393, 0.154 (-0.239) | 0.480, 0.194 (-0.287) | 0.506, 0.191 (-0.315) |
| single-support CoP margin (mm) | 45.3, 46.0 (+0.7) | 44.9, 45.2 (+0.3) | 44.2, 45.2 (+1.0) |
| ZMP margin (mm) | 45.3, 5.2 (-40.1) | 44.9, 4.7 (-40.2) | 44.2, 5.0 (-39.2) |
| friction need mu_req | 0.361, 0.405 (+0.044) | 0.629, 0.585 (-0.044) | 0.787, 0.683 (-0.104) |
| stance-pad peak \|Mz\| (N m) | 2.48, 2.50 (+0.02) | 4.63, 5.09 (+0.46) | 5.83, 5.93 (+0.11) |
| double-support min mu | 0.198, 0.197 (-0.000) | 0.206, 0.272 (+0.066) | 0.223, 0.300 (+0.077) |
| sway Stage 0 (H) | 0.164, 0.161 (-0.003) | 0.111, 0.112 (+0.001) | 0.101, 0.116 (+0.015) |
| sway CoM per stride (H) | 0.164, 0.160 (-0.004) | 0.111, 0.111 (-0.000) | 0.101, 0.115 (+0.014) |
| CoM horizontal accel (m/s^2) | 2.14, 2.25 (+0.11) | 2.84, 2.51 (-0.33) | 3.16, 3.04 (-0.12) |
| CoM vertical accel max (m/s^2) | 0.84, 2.36 (+1.52) | 1.51, 5.28 (+3.76) | 1.85, 5.95 (+4.10) |
| CoM vertical accel min (m/s^2) | -0.82, -2.52 (-1.71) | -1.49, -2.02 (-0.53) | -1.75, -3.06 (-1.32) |
| foot vertical accel (m/s^2) | 4.8, 21.5 (+16.7) | 8.1, 23.7 (+15.6) | 8.6, 25.9 (+17.3) |
| mid-swing clearance (H) | 0.050, 0.042 (-0.008) | 0.056, 0.047 (-0.009) | 0.054, 0.047 (-0.007) |
| tightest joint margin (rad) | 0.169, 0.198 (+0.029) | 0.205, 0.228 (+0.023) | 0.213, 0.225 (+0.012) |

## 7. Checks

- Re-price: on 73 of 73 rungs `look.price` on the arrays equals the price `look_refs.json` records, number for number.
- G2 (gait.py with every look field at its default reproduces today's references; `screen.g2` on the code that ran the screen): 60-N-T-10: nominal bit-identical, brisk bit-identical, fr05 bit-identical, record leaves match; 75-M-H-10: nominal bit-identical, brisk bit-identical, fr05 bit-identical, record leaves match (against `stage1/runs/2026-11-08/<body>/refs.json`; one BLAS thread). The ten references of each body, turns included, are `tests/test_stage2_look.py`'s G2 test.
- Round two's split on the F5 rungs (`gait.inverse_dynamics`, a rolling foot's CoP kept in its pad box, 43 rungs): its governing ratio differs from the screen's by +0.000 to +0.000 and no group's worst ratio by more than 0.016; a single ratio below its group's worst moves by up to 0.112 (75-M-H-10 animal pace (Fr 0.05) V6); no verdict changes. The rolling contact matters frame by frame, not for the governing limits.
- No rolling foot in single support on any rung (the friction reading's single-support wrench is unique); the double-support friction LP is not read on rolling frames (counted in section 2's tables).
- Provenance: {"commit": "f4bc82b7680c4960c4b33f5b28f875ad614bdeeb", "dirty": false, "dirty_paths": []}; versions {"python": "3.13.18", "numpy": "2.5.4", "mujoco": "3.10.0", "scipy": "1.18.1"}; 3 processes, 9 min.
