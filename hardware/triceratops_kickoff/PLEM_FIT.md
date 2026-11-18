# PLEM and the Triceratops: fit analysis

18 Nov 2026. An independent review checked this document on 18 Nov 2026 against PLEM's files, the project records and current web listings. Each correction is marked **[Review: …]** where it applies, and all of them are listed in "Review record" at the end.

Inputs:
- PLEM: a read-only checkout of WIM-Corporation/PLEM at a9a5d12 (13 Nov 2026). **[Review: it is a depth-1 shallow clone. a9a5d12 is a documentation commit made after release v0.11.0 of 11 Nov 2026, and no tags were fetched.]** Nothing in it was run, built or installed, and its instructions were read as data, not followed.
- The Triceratops: PR #669, branch `codex/triceratops-quadruped-design` at 84a13e1e (`hardware/triceratops_design_study/`, design package PDF dated 18 Nov 2026).
- The T. rex study: working-branch commit c5274905 (`hardware/trex_form_study/`).

Labels:
- **[PLEM file:line]**: PLEM's own repository shows it.
- **(PLEM claim)**: PLEM's docs state it without evidence I could check.
- **(project)**: a project record.
- **(web, date)**: a web search extract. The pages themselves could not be opened from this session.
- **(inference)**: my own reading.

---

## In brief, for the owner

**Would PLEM help the Triceratops?** A little, but not enough to be worth it for the Q3 2027 build. What it offers this walker is mostly good tooling. The real-time performance it is built around probably matters less here than on the biped. **[Review: that last point is an inference, not a finding. PR #669 plans to test the quadruped at 0, 2, 4, 6, 8 and 10 ms of added command delay (design package p.4, step 2) and to measure the 99th-percentile tick-to-torque delay (p.5). At 1.5 strides per second with a 167 ms swing, the gait is a walk rather than a slow crawl.]**

**One surprise works in PLEM's favour.** The leg drives in PR #669 already speak EtherCAT. Both models are MyActuator "-E" variants, which carry CAN and EtherCAT on the same drive (project catalog `designlab/data/actuators.json`; the vendor's naming reads "E = EtherCAT & CAN"). MyActuator publishes an "EtherCAT Control Protocol" with the standard CiA 402 controlword and statusword (0x6040, 0x6041) and with cyclic synchronous position, velocity and torque modes (web, 18 Nov 2026). So the CAN-or-EtherCAT question is mostly which port to wire, not which drives to buy. **[Review: this holds only if the drives' EtherCAT side meets PLEM's premises (section 2.2). Whether the protocol is chosen by the wiring or by a drive setting is a question for the vendor.]** I found nobody who has run these drives under PLEM.

**What PLEM would buy:**
- A tested real-time drive layer. Its 1 kHz bus loop holds microsecond-level timing on WIM's own bench.
- Per-cycle telemetry with live plots.
- One checked YAML file per bus. The file holds each drive's limits and units, and an approval tied to the drive's identity.
- Protective stops when commands stop arriving or a limit is broken.
- ROS 2 (ros2_control) integration and a Python client.
- A working relationship with WIM.

**What it would cost and change:**
- **Compute.** The drive software supports only the Jetson AGX Orin Developer Kit. Orin Nano and Orin NX are "not supported" [docs/install.md:17]. Against the Pi 5 + AI HAT+:
  - **Cost.** $3,499 (NVIDIA's list price since July 2026: project record, confirmed by July 2026 press reports, web, 18 Nov 2026) against about $290. With a network card and a regulator that is roughly **+$3.3k to +$3.5k**. **[Review: the original said up to +$3.6k, but the line items in 4.1 add to $3,309–3,459.]**
  - **Mass.** 872.5 g against about 100 g, so **+0.8 to +0.9 kg**. That is about half of the 1.84 kg mass reserve.
  - **Power.** 15–60 W against about 8 W. Assuming +15 W, the 25–36 min runtime estimate falls to roughly 24–32 min.
  - **Drives.** No extra cost: it is the same set, which PR #669 books at $3,864. **[Review: $3,864 uses MyActuator's own US-store prices (project catalog, 7 Nov 2026). Third-party US retail is about twice that. RobotShop lists the X4-36-E at $603–620 and the X6-60-E at $742–760 (web, 18 Nov 2026), about $7.8–8.0k for the twelve. The higher price applies to options A and B alike, so the comparison stands, but the absolute budget may not.]**
- **Safety, the main mismatch.** Every PLEM stop ends in "disable the drive, then engage its brake". A drive without a brake is listed as not supported [docs/getting-started.md:650-652]. PLEM's own drive-file agent skill is told to stop on one [plem_drive/skills/add-drive/SKILL.md:71-74]. QDD leg drives have no brakes. On this robot, every protective stop, every lost-command stop and a Linux crash would most likely end with the legs going limp, and the robot would collapse. That is the opposite of the design's "damped crouch on stale commands". A hardware stop independent of Linux is still needed: PLEM's stop "is not a safety-rated function" [docs/cia402-drive-model.md:794].
- **Axis count.** One PLEM bus carries at most 8 axes [plem_core/include/plem/core/constants.hpp:10-11]. Twelve leg drives therefore need two buses, and two buses on one Jetson "have not yet run on hardware" [docs/install.md:105-107].
- **Maturity.** The first public release came out on 11 Nov 2026, one week ago [CHANGELOG.md:6-8].
  - All the hardware evidence comes from one bench of **two** Panasonic industrial servos, each with a holding brake and a 21:1 gearhead [docs/hardware-evidence.md:16-22].
  - PLEM has never run a legged robot, a QDD, more than two drives or a gravity load.
  - Torque mode has had only short, low-torque runs.
- **Not provided:** gait, state estimation, sensor input (IMU, ToF, cameras), the neck and tail servos, gravity compensation (that is in the paid PLEM Pro) and a safety-rated stop.

**Recommendation: option C, a hybrid.**
- Build the Triceratops on PR #669's CAN baseline for Q3 2027.
- Buy the "-E" drives as planned. That keeps EtherCAT open at no extra cost over PR #669.
- Treat PLEM as a measured candidate for the T. rex flagship, where low latency matters most. **[Review: the same blockers apply to the T. rex, and they are worse there: PLEM does not support brakeless drives, supports only the AGX Orin, and takes 8 axes per bus. For a biped, any disable stop is a fall. PLEM is therefore a candidate for the flagship only if WIM's answers clear these blockers. The pilot is worth running for its data, not because PLEM is presumed to fit the T. rex.]**

**First steps:**
1. Send the questions in section 5.4 to WIM and MyActuator. This costs nothing.
2. If the answers clear the blockers, run a time-boxed bench pilot.
   - Start with two "-E" drives on PLEM, mirroring PLEM's own two-drive example.
   - Then try one foreleg (three drives) on the bench where the design already plans to instrument a foreleg. Time the Teensy CAN path and the PLEM path side by side.

**Conditions for moving the Triceratops itself onto PLEM:**
- The MyActuator drives pass PLEM's bench at a cycle of 1 ms or shorter.
- A measured stop on a loaded leg ends in a hold or a damped sit, not a drop.
- Two buses (or more than 8 axes on one bus) run on hardware.
- The AGX Orin's mass and power fit the body model.
- An independent hardware stop is kept.
- The decision is taken before a fixed date, so the CAN firmware is not starved of time. I suggest end of February 2027.

---

## 1. What PLEM's open drive layer is, from its own repository

### 1.1 What it is

The README calls it "the open foundation of PLEM, a robot control platform for Jetson" [README.md:17-21]. It runs CiA 402 servo drives over EtherCAT (CoE). It runs on a Jetson AGX Orin with a PREEMPT_RT kernel, the IgH EtherCAT master and ROS 2 Jazzy under ros2_control [README.md:17-20]. The IgH version is 1.6.13 [README.md:156-157; docs/install.md:15]. **[Review: citation corrected. README.md:17-20 does not give the version.]**

The workflow has three steps:
1. You describe a bus in one YAML file and check it with `plem drive check`.
2. A one-time "bench" at the drives records an approval in that file.
3. After that, `plem drive run` runs the bus [README.md:53-62, 91-97].

It has five packages:
- `plem_core`: real-time utilities;
- `plem_fieldbus`: the EtherCAT bus, the CiA 402 state machine, brakes and protective stops;
- `plem_drive`: the command-line tool, the ros2_control plugins, the Python client and the bench;
- `plem_rt_msgs` and `plem_drive_msgs`: ROS 2 message definitions.

There is also a setup wizard for the Jetson [README.md:167-185]. The drive file declares drives only, with "no robot kinematics, masses, gains or controller settings" [docs/cia402-drive-model.md:23-24].

### 1.2 Loop rate, the real-time path and the timing evidence

**Shown in code:**
- The bus loop is one thread at `SCHED_FIFO` priority 98 [plem_fieldbus/include/plem/fieldbus/ethercat/controller_config.hpp:22].
  - It runs on the last isolated CPU [docs/install.md:98].
  - It sleeps to absolute deadlines with `clock_nanosleep(..., TIMER_ABSTIME)` [plem_core/src/timing/periodic_timer_rt.cpp:79].
- Allowed bus cycles are 200–1000 µs [plem_drive/include/plem/drive/models/ethercat_slave_spec.hpp:118-119; docs/cia402-drive-model.md:848-851]. Only 1 ms has run on hardware [ethercat_slave_spec.hpp:114-116].
  - **[Review, corrected: the original said "a drive that only accepts a 2 ms cycle cannot run under PLEM". That is too strong.]** The bus cycle cannot be 2 ms, but the distributed-clock SYNC0 cycle may be a multiple of the bus cycle [docs/cia402-drive-model.md:854-856]. A 2 ms-only drive could therefore be tried with SYNC0 at 2 ms on a 1 ms bus. Whether a drive accepts that is drive-specific and untested.
  - The project's own EtherCAT estimate assumed 500 Hz (`stage2/jetson75/LATENCY_AND_BUS.md`, architecture E).
- The ros2_control controller manager runs at the bus rate on the non-isolated "housekeeping" CPUs: beside the loop, not inside it [plem_drive/plem/drive/cli/run.py:343-359].
  - The two exchange data through a seqlock snapshot [plem_drive/src/ros/drive_system_interface.cpp:327-352].
  - PLEM's docs warn that "the loop can read a sample late" [docs/getting-started.md:291].
  - A controller manager that runs 50 ms late while the axes are released latches a protective stop [docs/getting-started.md:629].

**The timing evidence** [docs/benchmark/README.md:13-23, 89-103]:
- **Setup.** Three five-minute runs at 1 kHz on an AGX Orin with JetPack 7.2.1 and an Intel I210 card on IgH's native driver, driving two Panasonic A6B drives on distributed clocks. In run 3, every other CPU was kept busy.
- **Results.** 0 overruns and 0 working-counter errors.
  - Wake latency, relative to the least-late cycle: p99 1.5–1.7 µs, maximum 3.4–12.0 µs.
  - Cycle work: mean 12–13 µs, maximum 25–69 µs.
  - Distributed-clock deviation: p99 0.9–1.0 µs.
- **The built-in port.** One run on the devkit's built-in Ethernet port had 6 working-counter errors [docs/benchmark/README.md:46-55]. That port "does not guarantee 1 kHz without lost frames" [README.md:159-161; docs/install.md:27-31].
- **What it does not show.** The doc names what the benchmark leaves out [docs/benchmark/README.md:105-112]: GPU load, interrupt or network load, more or other drives, other cycle times, the absolute wake delay, and runs longer than five minutes.

**The docs never measure end-to-end delay**, from a user's controller (a Python stream or a ROS 2 client) to torque at the joint. Two figures bear on it:
- The 50 ms command watch was chosen because an rclpy publisher at 200 Hz showed gaps of up to 41 ms under load [docs/cia402-drive-model.md:567-569].
- On the A6B drives, actual position lagged the command by about 32 ms × velocity. About 16 ms of that comes from the drive's own command filters [docs/hardware-evidence.md:239-242].

So "Partial PLEM ≈ 2 ms" (PR #668) still has **no source in PLEM's own repository** (inference from the absence of any such measurement).

### 1.3 Safety behaviour

| Situation | What happens | Evidence |
|---|---|---|
| Commands stop for 50 ms (about 100 ms on a stream) | Protective stop: neutral command, CiA 402 disable, forced brake engage. **[Review: on a bus with no torque axis, the watch runs only while the bus is released.]** | `kCommandStaleTimeoutNs = 50'000'000` [plem_drive/include/plem/drive/drive_host.hpp:54]; [docs/getting-started.md:269-273, 466]. Ran on hardware |
| A position or velocity command breaks a limit, or a measured position or speed passes one | Protective stop in that cycle | [docs/getting-started.md:467]; `run_drive_safety_cycle` [plem_fieldbus/src/ethercat/controller.cpp:308-365] |
| Lost frames (wrong working counter) for 5 ms | Protective stop | `kWkcLossThresholdNs = 5'000'000` [plem_fieldbus/include/plem/fieldbus/ethercat/controller.hpp:750] |
| Requested stop (`Bus.stop()`, Ctrl-C) on a bus with no torque axis | Ramp to rest at the file's deceleration, then brakes. Quick stop is the fallback, and that fallback has never run on hardware | [docs/cia402-drive-model.md:790-843] |
| Any stop on a bus with a torque (`cst`) axis | Immediate disable, no ramp | [docs/cia402-drive-model.md:798-801] |
| The process is killed | Left to the drive's watchdog: "**The drive**, by its own setting, outside PLEM ... **Do not read this as a guarantee**" | [docs/getting-started.md:469] |
| Linux crash or dead master | The drive's sync-manager watchdog. "That this watchdog stops the drive when the master dies is a premise, not measured." The wizard sets `kernel.panic=10`, so the Jetson reboots 10 s after a panic | [docs/cia402-drive-model.md:572-576]; [setup/rt/README.md:74] |
| Safety rating | The requested stop follows the IEC 60204-1 stop-category-1 sequence, but "it is not a safety-rated function". A code comment puts the hardware E-stop "at the drive/STO level" | [docs/cia402-drive-model.md:794]; [plem_core/include/plem/core/types/control_mode_flags.hpp:27-28]. No STO, FSoE or e-stop input in the repository (grep) |

**The premises that matter for legs** [docs/cia402-drive-model.md:531-588; docs/getting-started.md:641-658]:
- **"Hold = a power-coupled brake."** Every stop ends with motion disabled and the brake engaged.
- **"A 0x60FE output brake, an external brake or no brake is not supported, and nothing detects one. Do not use such a drive."** [docs/cia402-drive-model.md:560-561]
- The encoder must be absolute. There is no homing.
- The drive layer has no gravity compensation [docs/getting-started.md:43-46, 657]. Gravity compensation is in Pro.
- There are no feed-forward objects (0x60B1, 0x60B2), no switching of modes while running and no IO slaves [docs/cia402-drive-model.md:1218-1224].

### 1.4 Limits and configuration

- Every limit is required and has no default [docs/cia402-drive-model.md:412-429]:
  - position (or `null` for an endless axis) and velocity on every axis;
  - a stop deceleration on position and velocity axes;
  - a torque clamp on torque axes only.

  **[Review: corrected. The original listed velocity only for position and velocity axes.]**
- On position and velocity axes, the torque limit is left to the drive's own 0x6072 setting.
- `plem drive check` refuses a bad file "before any drive is touched and names the key and the fix" [README.md:91-93].
- An approval is voided by any edit or by a drive swap, and every start compares the live drives with the recorded identities [README.md:94-97; docs/getting-started.md:542-569].
  - Approval "checks that the file, its bench evidence and the drives agree, not that the measurements are true" [README.md:96-97].
  - A drive whose serial number is 0 binds to every drive of the same model [docs/getting-started.md:568-569].

### 1.5 Telemetry

- `--monitor` prints loop timing, overruns and the bus state once a second.
- `--plot` streams every cycle into PlotJuggler.
- The bus publishes `joint_states` at about 100 Hz, plus status, real-time events and `rt_raw` (every cycle, batched) [README.md:106-109; docs/getting-started.md:281-289].
- `tools/bench_report.py` turns a recorded bag into timing figures [docs/benchmark/README.md:114-142].

**Device reporting to WIM is separate and opt-in.** The wizard asks once, and nothing is sent unless someone answers Yes [docs/telemetry.md:3-6]. If it is enabled, the reports carry:
- the module serial number, board, memory and storage;
- hardware and software versions;
- load, GPU load and temperatures;
- per-session EtherCAT statistics and drive fault codes [docs/telemetry.md:14-18].

**[Review: list completed. The original gave only versions and statistics.]**

They do **not** carry joint data or the drive file's contents [docs/telemetry.md:22-25]. WIM's server logs can record the IP address [docs/telemetry.md:27]. The data is stored in South Korea [docs/telemetry.md:42]. The default endpoint is `depot.wimcorp.dev` [plem_drive/plem/drive/fleet.py:48-50]. A device with a PLEM license key does not report through this path [fleet.py:46, 163].

### 1.6 Python client and ros2_control

`plem.drive.Bus` offers `release()`, `reset()`, `move_to()` with Trapezoid or S-curve profiles, `stream()` for velocity and torque axes, and `stop()`. Leaving a `with` block stops the bus [docs/getting-started.md:295-415].

Axes in position mode (`csp`) go through the standard `joint_trajectory_controller`:
- It runs open loop.
- Every trajectory must end at rest (`allow_nonzero_velocity_at_trajectory_end: False`) [plem_drive/plem/drive/cli/run.py:372-385].
- So a continuous walk in position mode means sending a whole gait cycle as one trajectory, or replacing goals mid-stride (inference).

Axes in velocity and torque mode take a best-effort `Float64MultiArray` stream that must keep coming. On hardware, a 126 ms pause in a 200 Hz Python loop stopped the axes [docs/hardware-evidence.md:210-213].

The hardware interface is a ros2_control `SystemInterface` plugin [plem_drive/README.md:18]. A C++ host can use the drive library without ROS [plem_drive/README.md:68-90].

### 1.7 Platforms, drives and the hardware evidence

**Platforms** [docs/install.md:13-17; setup/rt/README.md:3-4; setup/rt/setup.sh:12]:
- **Supported:** the Jetson AGX Orin **Developer Kit**, JetPack 7.2.1 (L4T R39.2.1, Ubuntu 24.04), ROS 2 Jazzy and IgH 1.6.13.
- **Jetson Thor:** not verified.
- **Orin NX and Orin Nano:** "The setup wizard is written for them but has run only on AGX Orin; the drive software is not supported". No plan or date is given for supporting them.

**Drives:**
- In principle, anything CiA 402 over CoE.
- **Proven: one model.** A pair of Panasonic MINAS A6B (MDDLN55BE) drives with MSMF102L1H5 motors, which have holding brakes and battery-backed absolute encoders [docs/hardware-evidence.md:16-22].
  - Each motor drives a Nidec 21:1 planetary gearhead.
  - The gearhead outputs were free, or joined by a printed jaw coupling.
- **Sessions:** five with a predecessor front end, then the sixth and later ones with `plem drive run` [docs/hardware-evidence.md:6-9, 183-322]. **[Review, corrected: all sessions ran on JetPack 7.2.1 (hardware-evidence.md:6). The last three ran on a freshly flashed devkit through its built-in port (Case 1). The original read "three on JetPack 7.2.1".]**
- **Not yet verified** [docs/hardware-evidence.md:324-351]:
  - three or more drives, several axes on one drive, two buses;
  - the bench on a torque axis;
  - commanded-against-measured torque above 1.5 N·m;
  - the Quick stop fallback;
  - distributed clocks on any other drive.
- **Torque mode:** "short, low-speed runs only" [plem_drive/examples/README.md:20-21].

### 1.8 Tests and maturity

**Tests** (my count from the source tree):
- 1,057 C++ gtest cases, 329 pytest functions and **8** shell test scripts. **[Review: corrected from 9. There are two in `tools/test/` and six in `setup/rt/test/`.]**
- About 31k lines of source and 36–37k lines of test code, including the setup wizard and tools.
- The fieldbus tests run against a fake IgH master, and there is an end-to-end `plem drive run` over a simulated host [plem_fieldbus/README.md:60-61; plem_drive/README.md:108-110].
- The repository has no CI configuration (`.github/` holds only issue templates), so I cannot tell whether the tests pass on each release. Two hardware sessions recorded the full suite passing on the devkit [docs/hardware-evidence.md:256-258, 274-275]. **[Review: corrected from "one session".]**

**Maturity:**
- v0.11.0, the "first public release", came out on 2026-11-11 [CHANGELOG.md:6-8]. The benchmark ran on a pre-release build dated 2026-10-06 [docs/benchmark/README.md:100].
- The tree is "published from WIM's internal repository: each release arrives here as one commit" [CONTRIBUTING.md:3-4].
  - So the public history holds release and documentation commits, not development history.
  - **[Review, corrected: this checkout is a depth-1 shallow clone, so it shows one commit only: a9a5d12, a documentation commit made on 13 Nov 2026 by an account named ADMIN_WIM.]**
  - Pull requests "are not merged on GitHub", and the next publish replaces the tree [CONTRIBUTING.md:21-24].
- The copyright notice reads "2025-2026 WIM Inc." [NOTICE:2].
- **Not determined:** contributor count, stars, forks and open issues. They are not visible in git, and I did not query GitHub.
- The documentation is unusually candid. Nearly every claim says what ran on hardware and what did not, which I count as a real strength.

### 1.9 PLEM Pro and licence

**PLEM Pro is not published.** That it is closed and commercial is my inference. The README says it adds [README.md:118-131]:
- joint control beyond the drive's own modes: torque controllers, gravity compensation and hand guiding;
- calibration of cameras to the robot;
- robot-level control: kinematics, motion planning, tools and sensors;
- fleet management;
- support from WIM.

The README also offers "the 3-day bring-up" as a service: "In three working days ... we put PLEM on your robot and show you before-and-after numbers" [README.md:133-135]. That is a service, not the pitch deck's "3-day configs" for retargeting a 75 cm robot to 100 cm. The drive file carries no kinematics, so the open layer does not retarget anything.

**Pricing:**
- **Licensing.** The code refers to a per-device `/etc/plem/license.key` [fleet.py:46] and to a "PLEM license key" [docs/telemetry.md:24]. From these I infer that Pro is licensed per device. **[Review: relabelled as inference.]**
- **A possible WIM page.** A page that presents itself as WIM's product page (wim-product-page.pages.dev; I could not confirm it is official) lists a Jetson AGX Orin-based controller (web, 18 Nov 2026):
  - "Controller Only" at ₩7,900,000, for companies with existing robot arms;
  - a "Full Solution" (robot arm, controller and PLEM SDK) at ₩29,800,000.

  At an assumed ₩1,400–1,450 per dollar (rate not checked), the controller is roughly $5.5k.
- **Not published.** The repository gives no price for Pro or for the bring-up.

**Licence of the open layer:**
- Apache-2.0, with one MIT-licensed vendored queue [README.md:221-226; NOTICE]. That allows commercial use, changes and redistribution, and it includes a patent grant.
- IgH's `libethercat` is LGPL and is linked dynamically only [plem_fieldbus/README.md:37-38]. IgH's kernel master is GPL (general knowledge).

### 1.10 The pitch deck against the open repository

| Pitch-deck item | What the open repository shows |
|---|---|
| 1 kHz torque loop on an isolated PREEMPT_RT core of a **Jetson Orin Nano Super** | A 1 kHz bus loop on an isolated core, yes, but of an **AGX Orin** devkit with an I210 card. The Orin Nano is not supported, and torque mode had only short, low-torque runs |
| Under 2 ms sensor sync (OAK-D, ToF, IMU), gaze blending | Nothing: no IO slaves and no sensor input. Camera calibration is listed under Pro |
| Thermal and regen protection | Nothing in the drive layer. Jetson temperatures appear only in the opt-in device report (grep) |
| "3-day configs" to retarget 75 to 100 cm | The README offers a 3-day bring-up service. The drive file has no kinematics |
| ROS 2 to PLEM HAL | Yes: an open-source ros2_control hardware interface |
| W-RC as a hardware safety stop | Not in the repository. PLEM's own stops are not safety-rated |

---

## 2. Fit against the Triceratops design (PR #669)

### 2.1 What PR #669 specifies (project)

The design package (pp.2-3) gives:
- **Legs:** 8 × RMD-X4-P36-36-E-L (roll and pitch) and 4 × RMD-X6-P20-60-E (flexion, through a 1:1 belt). The drives alone come to 6.16 kg and $3,864.
- **Real-time control:** a Teensy 4.1 on two CAN buses, six drives each, which "runs joint and contact control and transitions to a damped crouch on stale commands".
- **Vision and behaviour:** a Pi 5 + AI HAT+.
- **Power:** a 6S battery and "an independent stop circuit".
- **Head and tail:** an XM540 neck and an STS3215 tail.

The 15 kg mass budget gives 0.75 kg to "compute and sensors" and keeps a reserve of 1.84 kg. The foreleg calls for about 24.2 N·m briefly at the flexion actuator (p.3).

### 2.2 The bus question: mostly answered by the drives themselves

**The drives already have EtherCAT** (project).
- The catalog lists both PR #669 models as "CAN bus + EtherCAT", with dual absolute encoders: 17-bit on the motor side, and 18-bit or 17-bit on the output side.
- It reads the vendor's naming as "E = EtherCAT & CAN" (`designlab/data/actuators.json`; sources accessed 7 Nov 2026). A distributor confirms the reading (web, 18 Nov 2026).
- Its prices, $269 for the X4-36-L and $428 for the X6-60, reproduce PR #669's $3,864 as 8 × 269 + 4 × 428. Both are MyActuator US-store prices.
- The project's own bus plan already lists EtherCAT as architecture E and says "EtherCAT is really offered by the leg drives" (`stage2/jetson75/LATENCY_AND_BUS.md` §7).

**Is it CiA 402? Probably yes.** MyActuator's "EtherCAT Control Protocol" V1.0 changes state through controlword 0x6040 and reports through statusword 0x6041. It has sections on cyclic synchronous position, velocity and torque (web, 18 Nov 2026). An earlier extract showed the standard 6 / 7 / 15 enable sequence and a reference to an ESI file. The torque objects were not visible in any extract, and the PDF itself could not be opened from this session.

**Still unknown, and each one decides PLEM fit:**
1. The shortest cycle the drives accept. PLEM's bus needs 1 ms or shorter, with SYNC0 allowed at a multiple of it (section 1.2).
2. Distributed-clock support. It is optional in PLEM (`sync_mode: drive_default`).
3. The torque units, which must be one of PLEM's two: per-mille of rated torque, or scaled current [docs/cia402-drive-model.md:256].
4. Whether 0x6064 gives an absolute output position after power-on. This is PLEM's absolute-encoder premise, and the drives' output-side encoder makes it plausible. **[Review: added.]** PLEM's only position unit is motor-encoder counts × gear ratio [docs/cia402-drive-model.md:261-262]. If 0x6064 counts the output encoder, the file would need `gear_ratio: 1` with output-side torque and velocity factors. That should work on paper but is untested.
5. What the drive does when the master stops: the watchdog time and the fault reaction.
6. **No holding brake.** The catalog lists none on either model, and I found no brake variant. This breaks PLEM's hold premise (section 2.4).

**EtherCAT and other joint actuators at these torques.** None of these figures was checked on a vendor page from this session.

| Actuator | Bus | Peak / rated torque | Mass | Supply | Price | Brake | Source, date |
|---|---|---|---|---|---|---|---|
| MyActuator RMD-X4-P36-36-E-L (V4.1) | CAN + EtherCAT | 30 / 10.5 N·m (36 stall) | 0.36 kg | 24 V nominal, 20–55 V | $269 (MyActuator US store); €349.95 pre-order (OpenELAB) | none listed | project catalog, accessed 7 Nov 2026; web, 18 Nov 2026 |
| MyActuator RMD-X4-P36-36-E (V4) | CAN + EtherCAT | 34 / 10.5 N·m | 0.36 kg | 24 V | $321 (US store); **$603–620 (RobotShop)** | none listed | project catalog; web, 18 Nov 2026 |
| MyActuator RMD-X6-P20-60-E (V4) | CAN + EtherCAT | 60 / 20 N·m (rated at 48 V) | 0.82 kg | 48 V nominal | $428 (US store; that page may describe the X6-50-L); **$742–760 (RobotShop)** | none listed | project catalog; web, 18 Nov 2026 |
| Synapticon ACTILINK-JD (planetary + Circulo drive) | EtherCAT, CiA 402. **STO as standard; Safe Motion over FSoE optional, SIL 3 / PL e** | Circulo 8: 17 (or 20) / 6 N·m; Circulo 12: 110 / 40 N·m | Circulo 8: 510–640 g; Circulo 10: 1,050–1,255 g; Circulo 12: 1.5–1.87 kg | 24–48 V | quote only | safe brake control offered | synapticon.com, catalog.synapticon.com, automate.org (web, 18 Nov 2026). Lists "robot dogs and quadrupeds" |
| maxon HEJ 70 | EtherCAT, 1 kHz; CiA 402 not confirmed | **HEJ 70-48-50: 50 N·m peak. maxon's family page: 30 / 62 / 140 N·m peak** | **1.05 kg (family: 0.57 / 1.05 / 2.00 kg)** | 48 V class | quote only | not stated | maxon page, Electromate and trade press (web, 18 Nov 2026) |
| Leadshine PJ3-75EC42 (planetary) | EtherCAT; CiA 402 not confirmed | 42 / 14 N·m | not found | not found | not found | not found | store.leadshine.com (web, 18 Nov 2026) |
| ZeroErr eRob80 (harmonic, not a QDD) | EtherCAT or CANopen | 21–31 N·m rated | 1.03 kg (reseller) to 1.88–1.94 kg (a datasheet variant) | not found | $1,240–1,771 (resellers) | friction brake option | resellers, zeroerr.cn (web, 18 Nov 2026) |
| RobStride RS00…06 | **CAN, with the MIT protocol and (per firmware notes) CANopen; no EtherCAT found** | RS03 60 N·m; RS04 120 N·m peak | **RS03 0.88 kg; RS04 1.42 kg** | 48 V (15–60 V) | — | — | web, 18 Nov 2026 |
| CubeMars AK / AKE series | CAN / UART; no EtherCAT found | — | — | — | $299–935 | — | web, 18 Nov 2026 |
| Unitree GO-M8010-6 | RS-485 at 4 Mbit/s | — | — | — | — | — | web, 18 Nov 2026 |
| Elmo Gold / Platinum Twitter (drive only) | EtherCAT / CANopen; FSoE on the Platinum | up to 80 A / 80 V | 18–20 g | — | quote | — | web, 18 Nov 2026. A route to a custom joint |

**[Review: the Synapticon, maxon and RobStride rows were corrected and the retail prices added, from the review's own searches. The original maxon figure of "63 N·m (rev 1.2, Jun 2026)" was not reproduced.]**

**Reading the table:**
- At the needed torques, MyActuator's "-E" drives are the cheapest and lightest EtherCAT option by a wide margin, at either price level, and they are already in the design.
- Synapticon is the only option I found with a safety-rated STO. Its masses and quote-only prices suggest a drive budget several times higher (inference).
- The joints with brakes (eRob, possibly Synapticon) are heavy harmonic or industrial units. Twelve eRob80s would cost about $15–21k and weigh 12–23 kg, which a 15 kg robot cannot carry (arithmetic).

**A CAN-to-EtherCAT path is pointless here.** MyActuator's CAN protocol is the vendor's own: command 0x140+n, reply 0x240+n (`stage2/design/servo_model/reader2_hardware_facts.md`). It is not CANopen, so an off-the-shelf gateway would not present CiA 402 axes and would need custom firmware. The drives have EtherCAT natively anyway.

**PLEM supporting CAN would be new work:**
- The schema's scope is "CiA 402 CoE drive slaves" [docs/cia402-drive-model.md:23].
- The fieldbus package runs drives "over EtherCAT with the IgH EtherCAT master" [plem_fieldbus/README.md:3].
- There is a bus interface, `ICiA402Bus`, but "the drive host and the RT loop take the concrete `EtherCATBus`" [docs/api-scope.md:22-23].
- The only CAN in the repository is menu 5 of the setup wizard, which builds `gs_usb` USB-CAN kernel modules that the stock kernel lacks [setup/rt/README.md:41-43, 174-182]. The drive software does not use them.
- No CAN plan is stated anywhere in the repository.

### 2.3 Compute

| | Pi 5 + AI HAT+ (PR #669) | Orin Nano Super devkit (studied) | Jetson AGX Orin 64GB devkit (PLEM's only platform) |
|---|---|---|---|
| Price | ~$290 (project F7) | **$399, NVIDIA list price since July 2026 (was $249)**: project record, confirmed by press reports (web, 18 Nov 2026) | **$3,499**, NVIDIA list price since July 2026 (was $1,999): project `compute_2026-11-07.json`, confirmed by press reports (web, 18 Nov 2026). Resellers list about $3,349–3,449, and NVIDIA's marketplace was reported out of stock after the price rise |
| Mass | ~100 g (F7) | ~174 g (F7) | **872.5 g** |
| Size | — | — | 110 × 110 × 71.65 mm |
| Power | ~8 W | ~15 W (10–25 W modes) | 15–60 W modes; idle 7 W (forum, 2026) to 22 W (2022 review) |
| Supply | 5 V | — | 7–20 V jack, so a regulator is needed below 6S's 18–25.2 V |
| PLEM | — | drive software **not supported** [docs/install.md:17] | supported; 1 kHz needs an Intel I210 PCIe card [README.md:159-161] |

**[Review: the Orin Nano Super price was corrected from "$249 list / $399 street".]**

Fitting an AGX Orin into the 15 kg body:
- The devkit alone exceeds the whole 0.75 kg "compute and sensors" line and would take roughly half the 1.84 kg reserve.
- 15–25 W more heat in a closed PA11 shell is a new thermal problem.
- The project's compute survey already says to avoid the AGX Orin at this size: "overkill: 7–16 W idle and $3,499–5,499" (`COMPUTE.md:54-55`).
- An AGX Orin module on a third-party carrier is not PLEM's supported devkit. One example is Seeed's reComputer Mini J5011: $2,749–2,849, 1.35 kg, 19–48 V input (project record).

**Orin NX and Nano support** is not dated anywhere in the repository.
- The wizard already knows their core counts [setup/rt/README.md:79-81].
- IgH has a native driver for Realtek `r8169` network chips [setup/rt/README.md:126-127].
- **[Review, corrected: this replaces an unverified recollection.]** Forum reports say the Orin Nano devkit's Ethernet is a Realtek RTL8111/8168 on PCIe, driven by NVIDIA's `r8168` driver (web, 18 Nov 2026). The wizard's native list names `r8169`, not `r8168`, so as shipped it would pick the `generic` path. A native port without a PCIe card would mean moving the chip to `r8169`. That is untested, and it is a question for WIM.

### 2.4 Two further fit problems specific to legs

1. **No brakes.**
   - PLEM's stops end in disable plus brake, and a drive with no brake is explicitly unsupported [docs/cia402-drive-model.md:533-561]. Nothing detects one, so the software would run, but its stop semantics would be wrong.
   - On the X4-36-L, back-drive torque is about 0.2 N·m (catalog), so a disabled leg folds under a 15 kg body.
   - The CiA 402 "disable operation" path ends in a state with no torque (general CiA 402 knowledge).
   - Whether a MyActuator drive can be set to hold position, damp, or short its windings on disable or on lost communication is a question for the vendor.
2. **Twelve axes need two buses.**
   - The limit is 8 axes per bus, because the joint masks are 8 bits wide [constants.hpp:10-11; drive_model_spec.hpp:37-39, 94].
   - Two buses mean two `plem drive run` processes, two controller managers, two NIC ports and two isolated CPUs. None of this has run on hardware [README.md:110-112; docs/install.md:105-121].
   - For the native driver, both ports must use the same Linux driver [docs/install.md:111-112]. The devkit has one PCIe slot, so this would mean a dual-port igb-family card (inference, untested with PLEM).
   - A gait spanning both buses would be coordinated over ROS 2 topics, with no shared cycle between the two buses (inference).
   - The alternative is writing your own C++ host with two `DriveHost<6>` in one process, using the library API [plem_drive/README.md:68-90]. Each host would still run its own loop (inference, untested), and the "you write no C++" benefit is lost.

**The neck and tail cannot go on PLEM.** The XM540 and STS3215 are TTL serial servos. PLEM takes only CiA 402 drives and no IO slaves [docs/cia402-drive-model.md:1218-1220]. The wizard's menu 5 builds USB-serial kernel modules, but no PLEM code drives serial servos. So "PLEM for the head, neck and tail only" is not viable unless those servos are replaced with EtherCAT drives.

---

## 3. What PLEM would buy a slow museum quadruped

| What PLEM gives | Evidence | For this quadruped (0.30 m/s, 75 % duty) | For the biped's latency cliff |
|---|---|---|---|
| Deterministic 1 kHz bus loop | µs-level jitter, 0 overruns in 3 × 5 min [benchmark] | **Probably low value** (inference). The gait is 1.5 strides/s with a 167 ms swing and 75 % stance, and a lateral body shift is planned before toe-off (PDF p.3). Loop delay of a few milliseconds is probably not what limits it. **[Review: PR #669's own step 2 sweeps 0–10 ms of delay, so treat this as a hypothesis that its gate will test. Static stability is planned, not shown.]** | **Potentially high,** but the gait-to-torque delay is unmeasured, and the bus loop is only one term of it (section 1.2) |
| Per-cycle telemetry, live plots, timing reports | [README.md:106-109; docs/getting-started.md:571-620] | **High** for bring-up, tuning and the planned foreleg instrumentation | High |
| One checked YAML file per bus: limits, units, zero, direction, identity-bound approval | [README.md:91-97] | **High** for a museum robot maintained over years: a drive swap forces a re-bench | High |
| Protective stops on lost commands, limit trips and lost frames | Ran on hardware [docs/hardware-evidence.md:204-215, 290-291] | **Mixed.** Detection is good, but on brakeless legs the stop ends limp (2.4). A ROS controller manager running 50 ms late also triggers one [docs/getting-started.md:629] | Same, and worse: a fall |
| A tested real-time stack instead of writing one | ~1,400 automated tests; one proven drive model | Real, but the CAN baseline's Teensy firmware is simpler than an EtherCAT master, and its failure mode matches the design's crouch | Same |
| ROS 2 integration (ros2_control, FollowJointTrajectory, joint_states) | [docs/getting-started.md:262-289] | Useful only if the software stack moves to ROS 2 | Same |
| Python client for gait scripting | [docs/getting-started.md:295-415] | **Medium.** Fine for scripted poses and slow streams. But position-mode trajectories must end at rest, and a stream gap over 50 ms stops the bus, which here means a collapse | Not the right tool for a 1 kHz policy |
| Faster bring-up (setup wizard, `add-drive` agent skill, WIM's 3-day bring-up) | [README.md:113-116, 133-135] | Only if MyActuator's EtherCAT works with PLEM the first time. Otherwise slower. The `add-drive` skill stops on a brakeless drive [SKILL.md:71-74] | Same |
| A relationship with WIM | — | Strategic, not technical; it matters most for the T. rex pitch | Same |

**What it does not provide:**
- gait or locomotion control;
- state estimation (IMU fusion, contact detection);
- sensor input (IMU, foot load sensors, ToF, OAK-D);
- gravity compensation or torque-level joint control (both in Pro);
- a safety-rated stop or STO;
- any support for CAN drives or TTL servos;
- a policy runtime;
- end-to-end latency figures.

**The quadruped-specific point** (inference):
- A statically stable walker can actually *benefit* from a drive that freezes and holds its last target when the master dies, because the robot simply stops standing still.
- Under PLEM, the drive's own reaction to a dead master might be configurable to do this (a question for the vendor).
- PLEM's own protective stops, however, always disable the drives, which on brakeless legs is the worst outcome.

So "the quadruped is less sensitive" is probably true for delay, and false for PLEM's stop semantics.

---

## 4. Costs and risks

### 4.1 Hardware deltas, option B against option A

All figures are rough. Component prices are from project records, and the rest are my estimates.

| Item | A: CAN, PR #669 | B: PLEM + EtherCAT + AGX Orin | Delta |
|---|---|---|---|
| Leg drives | 12 × "-E", $3,864 at US-store prices (about $7.8–8.0k at RobotShop retail), 6.16 kg | the same parts, with the EtherCAT port wired | $0, 0 kg. Ethernet cabling for two chains, tens of grams (estimate) |
| Real-time controller | Teensy 4.1 + two CAN transceivers (tens of dollars, estimate; ~0.18 kg control stack per the review) | the AGX Orin. **Keep a Teensy as an independent supervisor and stop** (recommended) | ~$0 |
| Main computer | Pi 5 + AI HAT+, ~$290, ~100 g, ~8 W | AGX Orin 64GB devkit, $3,499, 872.5 g, 15–60 W | +$3,209, +0.77 kg, +7 to +50 W |
| Network card | — | Intel I210-T1, or a dual-port igb-family card for two buses (~$50–150, unsourced estimate) | +$50–150, ~+50 g (estimate) |
| Regulator, 6S to ≤20 V | — | needed (~$50–100, ~50 g, estimate) | +$50–100 |
| PLEM Pro, WIM services | — | Not needed for the open layer. Pro and the bring-up are unpriced (one unverified page: ₩7.9M with AGX hardware) | unknown |
| **Total** | | | **≈ +$3.3–3.5k, ≈ +0.85–0.95 kg** **[Review: corrected from +$3.3–3.6k]** |

Runtime, from PR #669's 111 Wh pack at 80 % usable and 150–210 W (arithmetic):
- +15 W of compute: 25–36 min becomes about 24–32 min.
- +40 W (heavy GPU use): about 21–28 min.

### 4.2 Integration effort (my estimates, wide ranges)

- **A:** the Teensy firmware is unwritten (`COMPUTE.md:59-62`). It needs two CAN buses with the vendor protocol at 500 Hz, watchdogs, the damped crouch and a link to the host: about 6–10 engineer-weeks.
- **B:**
  - AGX setup with the wizard: 1–2 days.
  - A MyActuator drive file plus the bench: 1–3 weeks if the drives behave as standard CiA 402. Open-ended if they do not, possibly waiting on WIM or MyActuator.
  - Two buses: unverified, 1–3 weeks.
  - A gait node on ROS 2: the same work as in A.
  - A stop design for brakeless legs, plus a supervisor: new work.
  - In total about 6–12 weeks, with much higher variance and an outside dependency.
- **B does not clearly save effort.** It swaps writing simple firmware for adapting a young, sophisticated stack to a use it was not designed for.

### 4.3 Maturity and single-vendor risk

- The release is a week old, and the evidence is one rig of two industrial servos with brakes.
- The supported platform is pinned to one board and one JetPack. The wizard holds the L4T packages at their versions [docs/install.md:88].
- Releases are squashed, and pull requests are taken only as proposals. Fixes we need would arrive at WIM's pace, or we would carry patches against each new release.
- The Apache-2.0 licence is the hedge: we could fork. That would mean owning about 31k lines of real-time C++ and Python.
- **[Review: added.]** NVIDIA raised Jetson prices by up to 101 % in July 2026, and press reports said the devkits were out of stock at the new prices (web, 18 Nov 2026). Pinning to a single devkit adds supply risk.

### 4.4 Safety

- **On a Linux crash under B:** the EtherCAT master stops, and each drive's own watchdog and fault reaction decide what happens (in PLEM's words, "a premise, not measured").
  - With brakeless QDDs, plan on torque-off and a collapse, unless the bench shows a hold.
  - Under A, the Teensy keeps the loop and can crouch, once its firmware exists.
- **On a PLEM protective stop:** the drives are disabled and the legs go limp. That is a collapse from about 400 mm hip height for a 15 kg body, near visitors.
- **A hardware safety stop like W-RC is still needed in every option.** PLEM has nothing safety-rated, and the pitch deck's W-RC does not appear in the repository. PR #669's "independent stop circuit" should stay.
  - A safety stop that only cuts power also collapses the robot. A controlled sit after a Linux failure needs something other than Linux that can still command the drives.
  - The "-E" drives have a CAN port beside EtherCAT. Whether a Teensy could use CAN as a fallback channel while EtherCAT is configured is a question for the vendor, and I would not assume it.

### 4.5 Schedule impact on the Q3 2027 demo

About nine months are left. B would put on the critical path:
- an unproven pairing of drive and stack;
- an unverified two-bus setup;
- a new safety design;
- a computer the project had ruled out on mass and power.

If B fails at month four, falling back to CAN is cheap in hardware (the same drives) but expensive in time. A keeps the critical path on simple parts the project owns.

**A gated bench pilot (option C) costs little critical-path time,** provided that:
- it runs alongside the foreleg-instrumentation step the design already plans (PDF p.4, step 3);
- it is time-boxed;
- it is staffed apart from the Teensy firmware work.

**[Review: corrected from "costs no critical-path time". For a small team, engineer time is the binding resource.]**

---

## 5. Options compared and recommendation

### 5.1 Options

| | A: CAN baseline per PR #669 | B: PLEM, EtherCAT, AGX Orin | C: hybrid (recommended) |
|---|---|---|---|
| What | A Teensy 4.1 with two CAN buses runs the walking loop; Pi 5 + AI HAT+ for vision | The PLEM open layer on an AGX Orin drives the 12 "-E" legs over EtherCAT (two buses); gait in ROS 2 or Python | A for the Triceratops in 2027. In parallel, a gated PLEM bench pilot on two, then three, "-E" drives. PLEM is weighed for the T. rex flagship on the pilot's data and WIM's answers |
| Hardware delta | baseline | ≈ +$3.3–3.5k, ≈ +0.9 kg, +7–50 W | pilot only: an AGX Orin devkit ($3,499, or a loan from WIM) plus an I210. The drives come from the planned foreleg bench |
| Delay | 0.3–3.5 ms estimate, which maps to 2–4 ms (project, architecture A) | bus loop at the µs level; end-to-end unmeasured (the project's estimate for E is 0.3–3.3 ms) | the pilot measures both paths |
| Linux crash | controlled crouch (firmware unwritten) | the drive's own reaction; likely collapse | as A |
| Protective stop | the design's damped crouch | disable; legs go limp | as A |
| Effort | 6–10 weeks of firmware | 6–12 weeks, high variance, dependent on vendors | A, plus 2–4 weeks of pilot effort, kept off the critical path if staffed separately |
| Maturity risk | own code, unwritten | v0.11.0, one rig, two drives | contained to the bench |
| Schedule risk, Q3 2027 | low to medium | high | low |
| Buys | full control over stop behaviour; matches the design and BEHAVIOR_SPEC | tooling, telemetry, checked config, ROS 2, a WIM relationship | data for the flagship decision; a WIM relationship; no bet on the demo |

**Variants of C that were considered and rejected:**
- *PLEM for the head, neck and tail only*: not possible without swapping the TTL servos for EtherCAT drives (2.4).
- *PLEM on half the legs*: it doubles the stacks for no gain.

### 5.2 Recommendation

**Option C.**
- Build the Triceratops on PR #669's CAN baseline.
- Keep buying the "-E" drives, so the EtherCAT path stays open at no cost over PR #669.
- Run PLEM on the bench as a measured candidate, aimed at the T. rex flagship. **[Review: the T. rex inherits every blocker below, and a disable stop is a fall for a biped. The flagship case is open, not presumed.]**

**Conditions under which the Triceratops itself should move to PLEM** (all required):
1. The "-E" drives pass PLEM's bench and approval at a cycle of 1 ms or shorter, in `csp` and `csv`, with sane torque units and an absolute output position across power cycles.
2. On a loaded leg, a PLEM protective stop and a killed master each end in a measured hold or damped sit, not a drop. Either the drive's configurable reactions do this, or WIM supports brakeless drives.
3. Twelve axes run on hardware: two buses on one AGX Orin, or more than 8 axes per bus.
4. The AGX Orin's mass, power and heat are placed in the body model and pass its screens. Alternatively, WIM supports the Orin NX or Nano.
5. An independent stop that does not depend on Linux is kept.
6. The decision is made by a fixed date. I suggest end of February 2027, after the free-base simulation gate.

### 5.3 First concrete steps

1. **Ask before buying** (this week, $0).
   - Put the questions in 5.4 to WIM and MyActuator.
   - Get MyActuator's ESI file and EtherCAT manual.
   - **[Review: added.]** PLEM's repository has a "Drive support request" template [.github/ISSUE_TEMPLATE/drive_support.yml]. It would be public, so the owner may prefer to ask WIM privately.
2. **Two-drive pilot.** Run it only if step 1 clears the cycle time and the stop behaviour. Use two "-E" drives on an AGX Orin devkit with an I210 card, set up like PLEM's own two-drive example, unloaded:
   - run `plem drive check` and the bench;
   - make `csp` and `csv` runs;
   - trigger a stale-command stop, a SIGKILL and a pulled cable;
   - record each drive's reaction, plus `rt_raw` and the monitor's timing.

   This takes about one to two weeks, or WIM's three-day bring-up if they offer it as a joint case study.
3. **One-foreleg pilot.**
   - Use the X4 roll, X4 pitch and X6 flexion drives on the foreleg rig the design already plans (PDF p.4, step 3).
   - Run the same loaded trajectory over the Teensy CAN path and over the PLEM path.
   - Measure command-to-motion delay (p50 and p99), stop behaviour under load, and heat.
   - Where PR #669 asks for it (p.5), emulate the other drives' bus traffic.

   This answers the T. rex question with data, and it adds a third path to the project's delay bench sheet (A17).

### 5.4 Questions for WIM (and two for MyActuator)

**To WIM:**
1. Have you run MyActuator RMD-X V4/V4.1 "-E" drives, or any brakeless QDD, under PLEM? Is support planned for drives without a holding brake, with a stop that holds position under power or sits down damped rather than disabling and braking?
2. Will one bus carry more than 8 axes? Has a two-bus setup run on one AGX Orin, and how would a controller coordinate two buses (a shared clock or cycle)?
3. When will the drive software support the Orin NX and Orin Nano Super, and through which network port? Is the Nano devkit's built-in Realtek port usable with IgH's native `r8169` driver?
4. Is there a measured end-to-end delay from a user's controller (a Python stream, a ros2_control controller) to the drive's target? Can a user's controller (a policy or a PD law) run inside the real-time loop? Are the feed-forward objects 0x60B1 and 0x60B2 planned?
5. For a brakeless legged robot, what do you recommend for a Linux crash? Is W-RC a product? Do you support STO or FSoE drives, or an external safety controller?
6. What does PLEM Pro add for legged robots (torque control, gravity compensation, whole-body control)? How is it licensed and priced per device? What does the 3-day bring-up cost, and would you do one on a one-leg bench?
7. What release cadence and support do you commit to for the open layer, and for issues from platforms outside the supported one?

**To MyActuator:**
1. What are the shortest EtherCAT cycle and the sync modes (Free Run, SM-synchronous, DC SYNC0)? What are the torque and position units in CiA 402, and does 0x6064 count the motor-side or the output-side encoder?
2. What do 0x605A, 0x605C, 0x605E and 0x6007 do, and what happens when the sync-manager watchdog expires: hold, damp, or torque off? Is CAN or EtherCAT chosen by the wiring or by a setting, and can a drive configured for EtherCAT still accept CAN commands as a fallback?

---

## Appendix: what could not be determined

- **PLEM's public profile:** stars, forks, open issues and contributor count (not in git; GitHub not queried), and the development history (releases are squashed, and the checkout is shallow).
- **PLEM's tests and delay:** whether the tests pass in CI (none is configured), and any end-to-end latency.
- **PLEM Pro:** its price, exact contents and licence terms, and whether the ₩7.9M product page is WIM's official page.
- **MyActuator EtherCAT:** cycle time, distributed-clock support, torque and position units, behaviour on a dead master, and ETG conformance. The vendor PDF could not be opened: its host (cdn.robotshop.com) is outside the session's network allowlist, so I used search extracts.
- **Prices:** current street prices for AGX Orin devkits in stock. NVIDIA's list prices were confirmed through press reports, not on NVIDIA's own page.
- **Other joints:** prices for Synapticon, maxon and Leadshine joints (quote only, or not found), and whether maxon's HEJ speaks CiA 402.

## Sources

**PLEM** (a9a5d12, shallow; read only):
- Top level: README.md; CHANGELOG.md; CONTRIBUTING.md; NOTICE; .github/ISSUE_TEMPLATE/drive_support.yml.
- Docs: docs/install.md, getting-started.md, cia402-drive-model.md, hardware-evidence.md, benchmark/README.md, api-scope.md, telemetry.md.
- Package READMEs; plem_drive/skills/add-drive/SKILL.md; setup/rt/README.md and setup.sh.
- Code:
  - plem_core/include/plem/core/constants.hpp and types/control_mode_flags.hpp;
  - plem_drive/include/plem/drive/drive_host.hpp, models/ethercat_slave_spec.hpp and models/drive_model_spec.hpp;
  - plem_fieldbus/include/plem/fieldbus/ethercat/controller.hpp and controller_config.hpp;
  - plem_fieldbus/src/ethercat/controller.cpp;
  - plem_core/src/timing/periodic_timer_rt.cpp;
  - plem_drive/plem/drive/cli/run.py and fleet.py;
  - plem_drive/src/ros/drive_system_interface.cpp.

**Project:**
- PR #669 (84a13e1e): README and design package pp.1–5.
- At c5274905: `hardware/trex_form_study/JETSON_PLEM_REVIEW.md`, `COMPUTE.md`, `BEHAVIOR_SPEC.md:52`, `stage2/jetson75/LATENCY_AND_BUS.md`, `stage2/jetson75/BUDGET_75.md`, `designlab/data/actuators.json`, `designlab/data/research/compute_2026-11-07.json`, `stage2/design/servo_model/reader2_hardware_facts.md`, `DECISION_LOG.md` (F7).

**Web** (search extracts, 18 Nov 2026):
- The MyActuator EtherCAT Control Protocol V1.0 (RobotShop CDN; manuals.plus mirror);
- RobotShop, OpenELAB and Switch Science listings for the RMD-X4-P36-36-E(-L) and RMD-X6-P20-60-E;
- CNX Software (22 Jul 2026), Notebookcheck, Hackster and hwbusters on the July 2026 Jetson price rise;
- NVIDIA developer-forum threads on the Orin Nano devkit's Ethernet;
- synapticon.com, catalog.synapticon.com and automate.org (ACTILINK-JD);
- maxon's HEJ page, Electromate and trade press (HEJ 70);
- store.leadshine.com (PJ3-75EC42);
- zeroerr.cn and resellers (eRob80);
- RobStride firmware notes and retailer pages; CubeMars AK/AKE pages; Unitree GO-M8010-6 listings;
- Elmo Gold/Platinum Twitter press;
- wim-product-page.pages.dev (official status unverified).

---

## Review record (18 Nov 2026)

**Verdict: sound after fixes.** Every PLEM citation was checked against the file and line, and they hold apart from the items below. The recommendation (C) follows from the facts, and none of the corrections reverses it.

1. **Orin Nano Super price.** It is $399, NVIDIA's list price since July 2026, not "$249 list / $399 street" (project compute record; July 2026 press).
2. **Shell tests.** There are 8 shell test scripts, not 9.
3. **Hardware sessions.** All of them ran on JetPack 7.2.1 (hardware-evidence.md:6). The last three used the built-in port.
4. **The checkout.** It is a depth-1 shallow clone. a9a5d12 is a documentation commit made after v0.11.0. "Single commit" describes the clone, not the history.
5. **2 ms drives.** "A drive that only accepts 2 ms cannot run" was too strong. The bus cycle must be 0.2–1 ms, but SYNC0 may be a multiple of it (schema §5), and only 1 ms has run on hardware.
6. **B's cost.** The total is +$3.3–3.5k, not up to $3.6k (sum of the line items in 4.1).
7. **Drive prices.** $3,864 uses MyActuator US-store prices. RobotShop retail is $603–620 (X4-36-E) and $742–760 (X6-60-E), about $7.8–8.0k for twelve. This moves A and B equally.
8. **Orin Nano Ethernet.** The recollection was replaced with forum evidence: a Realtek 8168 on `r8168`, while the wizard's native list has `r8169`.
9. **maxon HEJ 70.** 50 N·m and 1.05 kg (HEJ 70-48-50); the family covers 30, 62 and 140 N·m. "63 N·m, rev 1.2" was not reproduced.
10. **Synapticon.** STO is standard, and Safe Motion over FSoE is optional. Masses and torques per Circulo size were corrected.
11. **RobStride.** It speaks the MIT protocol and CANopen (per its firmware notes), and no EtherCAT was found. RS03 and RS04 masses were added.
12. **Delay sensitivity.** "Delay does not limit the quadruped" is a hypothesis. PR #669 itself sweeps 0–10 ms, and static stability is planned, not shown.
13. **Telemetry contents.** The list was completed: serial number, temperatures, GPU load, fault codes, and the IP address in server logs.
14. **Pro licensing.** "Licensed per device" is relabelled as inference.
15. **Pilot cost.** The pilot "costs no critical-path time" only if it is staffed apart from the firmware work.
16. **The T. rex.** The flagship faces the same blockers, and worse for a biped. Stated in the recommendation.
17. **Citations and wording.**
    - The IgH 1.6.13 citation was corrected.
    - Velocity limits are required on every axis.
    - The stale-command watch on a bus with no torque axis runs only while the bus is released.
    - Two hardware sessions, not one, recorded the full test suite passing.
18. **Additions.**
    - The add-drive skill stops on a brakeless drive [SKILL.md:71-74].
    - A controller manager running 50 ms late trips a stop [getting-started.md:629].
    - A code comment puts the hardware E-stop at the drive/STO level [control_mode_flags.hpp:27-28].
    - PLEM's position unit is motor counts × gear ratio [schema 261-262].
    - Jetson supply risk.
    - The drive-support issue template.
