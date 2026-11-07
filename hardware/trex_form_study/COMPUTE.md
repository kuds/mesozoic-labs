# Onboard compute: Jetson, or something smaller?

The owner asked (P9) whether compute other than the Jetson makes sense, such as a Raspberry Pi or
other smaller computers and processors. This file answers that from a sourced survey run on
2026-11-07. The full record is `designlab/data/research/compute_2026-11-07.json`, and contains:
* the workload derived from the repository;
* 79 platform options in five categories, each with source URLs;
* an independent fact-check of each category (108 corrections, plus confirmations, unverifiable
  claims and missed options);
* a synthesis challenged by two critics and then revised.

Runtime effects are computed by `designlab/compute_options.py` on the round-one reference robots.

## Short answer

**Yes for 60 and 75 cm. Mostly no for 100 and 120 cm. The first fix is the power budget, not the
computer.**

* **The study's Jetson budget is about twice the measured draw.** The study books the Jetson at 31 W
  (25 W module plus 6 W carrier, storage and cooling), which is 35 W from the battery.
  * Independent measurements of the same module: about 5–6 W idle, 10–12 W running a mid-size
    detector end to end, and 12.5–14 W with a small detector flat out, all uncapped.
  * In its 15 W mode, the dev kit should draw no more than about 15 W at the board while following,
    with a ceiling near 18 W. These are estimates; nobody has measured light load in that mode.
  * Re-budgeting alone takes the 60 cm robot from about 34 to about 42 minutes of walking.
* **60 and 75 cm: a Raspberry Pi 5 makes sense.** The recommended start is a Pi 5 8 GB with
  Raspberry Pi's $70 AI HAT+, a 13 TOPS Hailo-8L that runs the person detector and the
  re-identification network. The OAK-D Lite in the head keeps computing depth and compressing the
  operator video. That set is about 8 W, about 100 g and $290, against about 15 W, 174 g and $439
  for the Jetson kit. A cheaper version drops the HAT and runs detection on the OAK-D's own chip
  (about 6–7 W, 75 g, $232). Nobody has measured that chip doing depth, video, detection and
  re-identification at once, so it waits for a bench test.
* **100 and 120 cm: stay with NVIDIA.** There, compute is about 9–19 % of walking power, and the
  growth features matter most: mapping, voice and bigger vision models. The owner has said price is
  not a constraint at these sizes.
  * At 100 cm: an Orin Nano 8 GB. Prototype on the dev kit, then move to the same module on a
    rugged carrier that takes battery voltage directly (about $929).
  * At 120 cm: an Orin NX 16 GB, only if mapping, voice and bigger models are committed. Its extra
    GPU speed needs the 40 W mode, which costs 13.6 W more than today's budget. Otherwise use the
    100 cm Orin Nano setup.
  * NVIDIA has announced an Orin Nano 2 for the first half of 2027. Re-check it before a production
    buy.
* **The interesting non-Jetson experiment is a camera that is also the computer.** Luxonis's OAK 4
  D Pro has a 48 TOPS processor and could replace both the head camera and the Jetson.
  * Vendor figures put it at roughly 8–14 W more than the 3 W the study books for the camera; no
    independent measurement exists.
  * It costs $1,249 and puts 674 g in the head.
  * It needs a battery-powered PoE+ injector that has not yet been sourced.
  * It runs Luxonis's own Linux, so the follow software would have to be rebuilt as apps on the
    camera.
* **Other boards.** RK3588 boards such as the Orange Pi 5 Max (about $219) draw about the same power
  as the Pi with the Hailo. They depend on vendor kernels, their re-identification path is untested,
  and stock is patchy.
* **Avoid** Google Coral, whose software is archived, and the big Jetsons (AGX Orin, Thor). Those are
  overkill: 7–16 W idle and $3,499–5,499.

## What stays the same whatever is chosen

* **The real-time layer stays on the microcontroller** (Teensy 4.1, already budgeted at 4 W). It runs
  the walking policy, the CAN motor loop to the 12 leg drives, IMU sampling, the ToF stop,
  velocity-request expiry and the watchdogs. That firmware still has to be written for every option.
  Until it exists, a Linux crash does not end in a controlled stop.
* **The current person detector has to change.** PR #662's OpenCV HOG pedestrian detector takes
  80–105 ms per frame on a 2.1 GHz x86 core (measured during this survey) and an estimated 140–320 ms
  on the ARM cores of any of these boards. That misses the 100 ms frame period and the 200 ms
  deadline. Detection moves to a small neural network on the camera, a Hailo chip or a GPU.
* **The tracker must handle a moving camera.** The camera now rides on the moving head, so the
  tracker composes the camera pose from the neck encoder and IMU on every frame (`BEHAVIOR_SPEC.md`
  §3).
* **Camera supply is the shared risk.** Every option except the OAK 4 depends on the OAK-D's Myriad X
  chip, which Intel discontinued in 2025. Luxonis says its end-of-life buys cover about five years
  from 2025, and the OAK-D Lite rose from $169 to $269. Buy spare cameras, and watch the lighter
  OAK 4 Lite, an early prototype today, as a successor.
* **2026 prices:** the Jetson kit went from $249 to $399 (the study's cost model already uses $399),
  and the Pi 5 8 GB from $80 at launch to $175.

## What the onboard computer has to do

Derived from the repository by the survey's requirements agent. Sources: the PR #662 autonomy stack,
`BEHAVIOR_SPEC.md`, `docs/hardware/SIM_TO_REAL_PLAN.md`, and this study's power and hardware code.

| Task | Rate | Budget | Where it should run |
|---|---|---|---|
| Stereo depth, rectification, RGB–depth alignment | 10–30 Hz | ≤ 60 ms acquisition age, ≤ 60 ms skew | On the OAK-D (free to the host) |
| Camera pose per frame (the camera is in the head) | Every frame | Neck angle and torso attitude aligned within 10–20 ms | Microcontroller timestamps, Linux composes the pose |
| Person detection | ≥ 10 Hz | Detection, tracking and planning within about 140 ms | Neural accelerator: OAK-D chip, Hailo, RK3588 NPU or Jetson GPU |
| Re-identification (no silent target switching) | Per person, 5–10 Hz | Stop and report "re-select" within 500 ms | Same accelerator |
| Ground-frame tracking, local obstacle grid, planner | 10–15 Hz | ≤ 200 ms from the oldest sample to a motion request | Linux CPU (about 1–1.5 cores in C++) |
| Operator video and detections | About 5 Hz view | Operator lease 0.8 s | Camera's hardware encoder (the Pi 5 has none) |
| Locomotion policy (small network) | 50–100 Hz | Trained with an 8 ms command delay | Microcontroller |
| CAN loop, IMU, ToF stop, watchdogs | 200–1000 Hz | Deterministic | Microcontroller |
| Voice commands (stretch) | Continuous audio | About 0.5–1.5 s to intent | Linux CPU or accelerator, with a beamforming microphone array |

**Minimum spec:**
* the Teensy-class microcontroller;
* a neural accelerator of about 1 TOPS or more for the detector and a light re-identification
  network;
* a Linux host with four Cortex-A76-class cores, 8 GB of RAM, NVMe or SSD storage, USB 3 for the
  camera and Ubuntu with ROS 2.

**Growth spec:** 20–70 TOPS, 6–8 cores, 8–16 GB and 256 GB or more of NVMe. That covers bigger
detectors and pose, full re-identification for up to 10 people, 2D-lidar SLAM, and on-device speech
recognition.

## Architectures compared

Power is for the compute set at the board: host, accelerator, and the camera's extra inference share.
It excludes the camera's base power, which the study already books, the motors, and the Teensy.
Prices are US list on 2026-11-07, without the camera unless the architecture replaces it.

| Architecture | Person detection and re-identification | Typical / max power | Mass | Price | Minimum spec | Growth | Porting effort | Main risks |
|---|---|---|---|---|---|---|---|---|
| A. Jetson Orin Nano Super dev kit (current), capped at 15 W | Jetson GPU (TensorRT) | 15 / 18 W (estimate from measurements) | 174 g | $439 | yes | moderate | low: the platform PR #662 assumes | dev-kit carrier not for production; no hardware video encoder |
| B. Orin Nano 8 GB module on a 10–60 V carrier (Connect Tech) | Jetson GPU | 15 / 18 W | not published | $929 | yes | moderate (takes an Orin NX later) | low–moderate: vendor BSP trails NVIDIA | about twice A's price for the same compute |
| C. Raspberry Pi 5 8 GB + AI HAT+ 13 TOPS (Hailo-8L); OAK-D for depth and video | Hailo-8L | 8 / 18 W (estimate; storage extra) | about 100 g | $290 | yes | partial (26 or 40 TOPS HAT upgrades) | moderate: Hailo toolchain, ROS 2 on Ubuntu | detector and re-identification share one PCIe lane; HAT rated 0–50 °C |
| D. Raspberry Pi 5 8 GB; detection on the OAK-D's own chip | OAK-D (Myriad X) | 6.7 / 10.8 W | about 75 g | $232 | provisional: the combined camera load is unmeasured | none | low–moderate | most dependent on the discontinued camera chip; the on-camera tracker ignores head motion |
| E. Orange Pi 5 Max (RK3588, 6 TOPS NPU); OAK-D for depth | RK3588 NPU | 8 / 15.7 W | 82 g | $219 | yes | partial | moderate–high: vendor kernel | thin stock; re-identification untested; NVMe idle penalty |
| F. Luxonis OAK 4 D Pro (camera = computer) plus the Teensy | OAK 4 NPU (48 TOPS) | 12 / 22 W over the camera's booked 3 W (vendor) | 674 g (replaces the head camera) | $1,249 | provisional | mostly | high: rebuild as camera apps on Luxonis OS | 674 g and 10–15 W of heat in the skull; PoE+ power path not sourced |
| G. Jetson Orin NX 16 GB (Super) on a 19–54 V carrier | Deep-learning accelerator plus GPU | 28 / 43 W (25 W / 40 W modes) | not published | $1,342 | yes | full, with the 40 W mode | low | under a 15–25 W cap its GPU is slower than A's |

## Runtime with each architecture, by size

Reference robots at balanced width, Substantial body, 10°, on each size's selected pack. First order:
drive power and the other auxiliaries are held fixed, and the compute set replaces the 31 W Jetson
allowance through the same 0.88 converter.
| Architecture | 60 cm | 75 cm | 100 cm | 120 cm |
|---|---|---|---|---|
| Study budget today (31 W Jetson allowance) | 34 min | 29 min | 29 min | 31 min |
| A. Jetson Orin Nano Super dev kit (current), 15 W mode | 42 min (+8) | 35 min (+6) | 33 min (+3) | 34 min (+3) |
| B. Orin Nano 8 GB module on a 10–60 V carrier | 42 min (+8) | 35 min (+6) | 33 min (+3) | 34 min (+3) |
| C. Raspberry Pi 5 8 GB + AI HAT+ 13 TOPS (Hailo-8L) | 47 min (+13) | 38 min (+9) | 34 min (+5) | 36 min (+4) |
| D. Raspberry Pi 5 8 GB, detection on the OAK-D | 48 min (+14) | 39 min (+10) | 35 min (+6) | 36 min (+5) |
| E. Orange Pi 5 Max (RK3588) | 47 min (+13) | 38 min (+9) | 34 min (+5) | 36 min (+4) |
| F. OAK 4 D Pro (camera = computer) | 44 min (+10) | 36 min (+7) | 33 min (+4) | 35 min (+4) |
| G. Jetson Orin NX 16 GB (Super) | 35 min (+1) | 30 min (+1) | 30 min (+1) | 32 min (+1) |
The mass change is small at every size, −74 to −99 g for C, D and E against the Jetson kit. The
0.12 kg tray and mounting allowance stays. The exception is F, which adds about 0.5 kg to the head.

## Recommendation by size

| Size | Primary | Alternative | Why |
|---|---|---|---|
| 60 cm | C: Pi 5 + AI HAT+ 13 TOPS | D: drop the HAT once the camera passes the combined-load bench test | Compute is the largest discretionary load: 31 W of a 96 W walk. C carries the minimum spec at about 8 W with measured Pi-plus-Hailo detector rates |
| 75 cm | C | D | Same camera, sensors and software as 60 cm |
| 100 cm | A, then B: Orin Nano, dev kit first, then a rugged carrier | G if mapping, voice and larger detectors will run together | Growth matters, price is not a constraint, and compute is a smaller share of walking power |
| 120 cm | G: Orin NX 16 GB, only if growth loads are committed; otherwise the 100 cm setup | F as an experiment | Flagship size, where voice, mapping and larger models would live |

## Bench tests before committing

1. **Jetson re-budget, no purchase needed.** Measure the current dev kit at the converter input and
   with `tegrastats`, running the real pipeline in the 15 W and 25 W modes.
2. **Camera acceptance test.** On an OAK-D Lite and an OAK-D Pro W, run stereo depth, alignment, a
   YOLOv6n-class detector and video encoding together. Pass: detection at 10 Hz or more and 25 or
   more re-identification crops per second. This test gates the step from C to D.
3. **Pi 5 host load** with the full follow stack: Ubuntu 24.04, ROS 2 Jazzy, C++ ingest, obstacle
   grid, planner and the ground-frame tracker.
4. **Hailo end to end:** the detector compiled at 416 px plus OSNet re-identification, with pipelined
   asynchronous inference.
5. **Detector accuracy on the real head geometry:** truncated bodies at 1.0–2.5 m, people out to
   5.5 m, and the Pro W's wide lens.
6. **Re-identification and no silent switching:** crossing and occlusion trials while the robot walks
   and moves its head.
7. **Camera power from the robot's 5 V rail.** An OAK-D Lite powered from a Pi has browned the Pi
   out.
8. **Teensy firmware and link:** clock sync, frame-aligned neck and IMU timestamps, and failure
   handling. This gates every option.
9. **Supply rails:** 6S sag and 12S full charge against each carrier's input range, and the real
   converter efficiency.
10. **Storage under walking vibration.**

## Round-one status

The round-one sweep keeps the 31 W Jetson allowance for every candidate, so all round-one runtimes are
conservative by roughly 3–14 minutes, depending on size and computer. Round two adopts the owner's
chosen architecture and its measured or bench budget in `designlab/power.py`, and re-sizes the
batteries.
