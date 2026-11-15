"""Stage 2, the 75 cm + Jetson track: a latency budget and bus plan per compute architecture (``JETSON_PLEM_REVIEW.md``
section 5, task 7) and the machine-made tables of the delay bench sheet (task 8, decision-log row A17).

Planning arithmetic only: nothing here is simulated, built or measured.  It writes ``stage2/jetson75/latency_bus.json``;
``LATENCY_AND_BUS.md`` and ``BENCH_SHEET.md`` embed the tables that ``md`` prints.

Every number carries its kind (``KINDS``):

* (a) a prescribed-walk or static model result of this study;
* (b) a closed-loop simulation result of this study;
* (c) a sourced fact: a catalog id in ``designlab/data``, a document or record of this study, or a web page
  (``SOURCES``; the web pages were read on 14 Nov 2026 through the web-search tool's extracts, ``WEB_READ``);
* (d) an estimate or assumption, with whose it is and how it was derived (``basis``).

What it computes:

1. ``frames``: the length of one CAN frame of the leg drives (classic CAN, 11-bit identifier, 8 data bytes: 8 n + 47
   bits with the 3-bit intermission, plus at most floor((34 + 8 n - 1) / 4) stuff bits) and its time at the drives' bit
   rate; a CAN FD frame is priced beside it as a hypothetical, since no CAN FD mode of the leg drives was found; and
   one EtherCAT frame carrying all 12 drives, from an assumed process-data size.
2. ``bus_load``: 12 leg drives, each a command and a reply per cycle, on 1, 2 or 3 buses at 250, 500 and 1000 Hz:
   frames per cycle x frame time, against the cycle.
3. ``architectures``: for A (Raspberry Pi 5 + Teensy, two CAN buses), A3 (the same with three), B (Jetson + Teensy),
   C1 and C2 (Jetson-direct over USB-CAN, one and two channels), D (the Orin's own CAN controller) and E (EtherCAT):
   the rate the bus holds, its load, an estimated delay from the control tick to the torque change (terms: compute,
   host jitter, the host's path to the bus, bus transmit, the drive's own loop, and the hold of a slower target rate),
   the age of the feedback at the tick, and the whole-step simulated delay the total maps onto (``k_of``).
4. ``aux``: the neck (ROBOTIS XM540-W270) and tail (Feetech STS3215) buses: packet times at an assumed baud rate.
5. ``model``: how the simulation's delay parameter is defined, with file and line (resolved here from the code's text).
6. ``ledger``: Stage 1's recorded delay and stiffness readings of the 75 cm nominal walk (b), read from the records.
7. ``bench``: the bench sheet's parts (catalog ids and prices, the allowances, and the cost model's bench lot from
   the round-two record), the gains of the delay steps on a bare shaft, the rows of the results and soak tables, the
   model's holding gains, the map from a measured delay to the simulated delay to re-run the bar at, the constant
   speeds of the friction sheet, what the prescribed walks need of the floor, the tail servo's holds and the bench
   time.

Run from ``hardware/trex_form_study`` (no MuJoCo, about a second)::

    python stage2/jetson75/latency_bus.py          # write latency_bus.json
    python stage2/jetson75/latency_bus.py md       # print every table (md NAME: one table)
    python stage2/jetson75/latency_bus.py sync     # rewrite the tables embedded in the two documents
    python stage2/jetson75/latency_bus.py check    # the JSON and the embedded tables are current (exit 1 if not)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
STUDY = RUN.parents[1]
OUT = RUN / "latency_bus.json"
DOCS = (RUN / "LATENCY_AND_BUS.md", RUN / "BENCH_SHEET.md")
DATE = "2026-11-14"
BODY = "75-M-H-10"

KINDS = {
    "a": "a prescribed-walk or static model result of this study",
    "b": "a closed-loop simulation result of this study",
    "c": "a sourced fact (a catalog id, a document or record of this study, or a web page)",
    "d": "an estimate or assumption (whose, and how derived, is in its basis)",
}
WEB_READ = ("read on 14 Nov 2026 through the web-search tool's extract of the page; the page itself could not be "
            "opened from the session, whose network allows the search tool only, so confirm it on the page before "
            "relying on it")

PLANT_JSON = "stage1/runs/2026-11-08/75-M-H-10/plant.json"
LEDGER = {
    "R2/43": "stage1/runs/2026-11-08/75-M-H-10/sweeps/R2/43/latency.json",
    "R3/42": "stage1/runs/2026-11-08/75-M-H-10/sweeps/R3/42/latency.json",
}
KP_LEDGER = "stage1/runs/2026-11-08/75-M-H-10/sweeps/R2/43/kp.json"
CANDIDATE = "round2/refine/feet_75/candidates/75-M-H-10.json"
LOOK_SCREEN = "stage2/look/screen.json"
ACTUATORS = "designlab/data/actuators.json"
PARTS = "designlab/data/parts.json"

# ------------------------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------------------------
SOURCES = {
    # --- this study
    "plant_json": dict(type="study", ref=PLANT_JSON,
                       says="the Stage 1 plant of 75-M-H-10: its 14 drives by catalog part, the physics step (2 ms) "
                            "and the holding gains"),
    "catalog_actuators": dict(type="catalog", ref=ACTUATORS, says="the drives' interfaces, prices and notes"),
    "catalog_parts": dict(type="catalog", ref=PARTS, says="compute, adapters, regulators and sensors with prices"),
    "actuator_research": dict(type="study", ref="designlab/data/research/actuators_2026-11-07.json",
                              says="the vendor's model naming, read on 7 Nov 2026 at "
                                   "https://www.myactuator.com/x4-36details: E = EtherCAT & CAN"),
    "control_py": dict(type="study", ref="designlab/stage1/control.py",
                       says="the tracker: servo targets every physics step, feedback every 5 steps, ramped in"),
    "env_py": dict(type="study", ref="designlab/stage1/env.py", says="the latency queue (_latency_on)"),
    "plant_py": dict(type="study", ref="designlab/stage1/plant.py",
                     says="the servo command, the physics step, the sensors' age and the holding-gain rule"),
    "carpet_py": dict(type="study", ref="designlab/carpet.py", says="SERVO_ERROR_RAD 0.01 (an assumption), zeta 1"),
    "mjcf_py": dict(type="study", ref="designlab/mjcf.py",
                    says="the default joint damping 0.2 N m s/rad and friction loss 0.15 N m (unsourced)"),
    "s1b_n8": dict(type="study", ref="stage1/runs/2026-11-10-speed/STAGE1B_SPEED.md",
                   says="engineering note N8: the delay ledger, the model's delay and what a bench figure compares "
                        "with; servo stiffness"),
    "s1_correction": dict(type="study", ref="stage1/runs/2026-11-08/STAGE1_PROBE.md",
                          says="the 10 Nov correction: the 75 cm nominal walk meets every limit to 6 ms, stays up to "
                               "8 ms, falls at 10 ms and stays up again at 20 ms"),
    "ledger_r2": dict(type="study", ref=LEDGER["R2/43"], says="the delay ledger of 75-M-H-10 under R2/43"),
    "ledger_r3": dict(type="study", ref=LEDGER["R3/42"], says="the delay ledger of 75-M-H-10 under R3/42"),
    "ledger_kp": dict(type="study", ref=KP_LEDGER, says="the servo-stiffness ledger of 75-M-H-10 under R2/43"),
    "compute_md": dict(type="study", ref="COMPUTE.md",
                       says="the real-time layer on the microcontroller; bench test 1 measures 'the current dev kit' "
                            "with no purchase; the firmware is unwritten"),
    "behavior_spec": dict(type="study", ref="BEHAVIOR_SPEC.md",
                          says="section 3: heartbeat loss, controlled stop within 1.0 s, then stand; operator stop "
                               "within 0.3 s; at most 200 ms from the oldest sensor sample to a motion request"),
    "decision_log": dict(type="study", ref="DECISION_LOG.md",
                         says="P3, P4, P14, P21, P26, P29, E6, F21, F22, A1, A2, A14"),
    "review": dict(type="study", ref="JETSON_PLEM_REVIEW.md", says="the review of PR #668 and this track's tasks"),
    "lessons": dict(type="study", ref="LESSONS_LEARNED.md",
                    says="section 2.3 (delay), 2.4 (damping and friction), 3.12 (the Python tracker's rate)"),
    "cost_py": dict(type="study", ref="designlab/cost.py",
                    says="the control lot ($125 / 180 / 270) and the bench lot of the cost model: a tether supply, a "
                         "charger, a cradle and instrumentation, $550 / 850 / 1,300, or $900 / 1,400 / 2,200 for a "
                         "body whose H is above 0.6 m"),
    "candidate_json": dict(type="study", ref=CANDIDATE,
                           says="the round-two record of 75-M-H-10: its cost lines, the bench lot among them"),
    "materials_md": dict(type="study", ref="round2/refine/MATERIALS.md",
                         says="section 5: the recommended tread (cast polyether polyurethane, 50-60 Shore A on a "
                              "70-80 A core, 1.0-1.5 mm round studs; a TPU 85 A print for a fast trial) and the pads' "
                              "pressure in static single support, 41 kPa on the 75 cm body with the larger pads"),
    "look_screen": dict(type="study", ref=LOOK_SCREEN,
                        says="the Stage 0 look screen's tail-phase check: the stance pad's peak twisting moment over "
                             "single support and the friction it needs, per walk and speed (prescribed-walk results)"),
    "ledger_py": dict(type="study", ref="designlab/stage1/ledger.py",
                      says="the ledger's own latency queue (_latency), its range (LATENCY_MAX_STEPS) and the "
                           "stiffness rule (KP_RULE: kp x f, kv unchanged)"),
    "compute_research": dict(type="study", ref="designlab/data/research/compute_2026-11-07.json",
                             says="the compute survey: PR #662's 60 cm platform modelled its drives as position "
                                  "servos with an 8 ms command delay and a 50 Hz hold"),
    "build_py": dict(type="study", ref="designlab/build.py",
                     says="the control stack: 0.18 kg, 'Teensy 4.1, CAN x2, IMU', 4 W; an IMU breakout at $40"),
    "s1b_a1": dict(type="study", ref="stage1/runs/2026-11-10-speed/STAGE1B_SPEED.md",
                   says="section 8, the A1 update: bench-test the tail servo at 0.51-0.64 N m at 75 cm"),
    # --- web (search extracts)
    "myactuator_protocol": dict(
        type="web",
        ref="https://cdn.robotshop.com/rbm/acdeb3e1-04dc-4138-8b99-88509ddc4e94/4/"
            "4501bc19-2742-44e8-8d9e-8a4b398c18c4/e6052050_rmd-x-motor-motion-protocol-v4.01.pdf",
        also=["https://cdn.shopify.com/s/files/1/0673/6848/5000/files/Motor_Motion_Protocol.pdf?v=1756635504"],
        says="MyActuator Servo Motor Control Protocol V4.01 (and a V4.3 copy): bus interface CAN, baud rate 1 Mbps; "
             "single-motor command 0x140 + ID, reply 0x240 + ID, multi-motor command 0x280; data frames, standard "
             "frame, 8 data bytes (DLC 8); a 'Motion Mode Control Command_CAN (0x400 + ID)'"),
    "myactuator_sibling_manual": dict(
        type="web", ref="https://manuals.plus/myactuator/rmd-x4-p36-24-rmd-series-motor-manual",
        also=["https://github.com/2b-t/myactuator_rmd/issues/9"],
        says="the manual of a sibling model (RMD-X4-P36-24): a 'force position hybrid control' frame with KP in 12 "
             "bits (0-4095 for 0.0-500.0), KD in 9 bits (0-511 for 0.0-5.0), position 16 bits, speed 12 bits and "
             "feedforward torque 12 bits; a three-loop servo whose current loop the user cannot tune. An issue of a "
             "community driver describes the motion-mode command on 0x400 + ID the same way: a desired position and "
             "velocity, kp, kd and a feedforward torque"),
    "myactuator_mc_driver": dict(
        type="web", ref="https://manuals.plus/myactuator/mc-series-brushless-servo-driver-manual",
        also=["https://cdn.robotshop.com/rbm/acdeb3e1-04dc-4138-8b99-88509ddc4e94/0/"
              "052b77d7-e3c8-45a8-a14d-df19bf957084/73c10167_mc-series-brushless-servo-driver-manual-24.2.1.pdf",
              "https://cdn.robotshop.com/media/m/mya/rbc-mya-04/pdf/mc300a_series_servo_motor_driver_manual_v2.1.pdf",
              "https://cdn.robotshop.com/media/m/mya/rbc-mya-09/pdf/"
              "myactuator-rmd-x6-v3-can-bus-18-mc-x-300-o-brushless-servo-driver-manual.pdf"],
        says="MyActuator's MC-series drivers, the drivers of the earlier RMD-X generation (a distributor files an "
             "MC-X-300-O manual under the RMD-X6 V3). The V3.0 manual, which covers the MC-X-500-O: current loop "
             "15 kHz, speed loop 5 kHz, position loop 1 kHz. An older V2.1 manual of the MC300A series: torque "
             "15 kHz, speed 5 kHz, position mode 500 Hz. No loop rate of the V4 and V4.1 drives was found"),
    "myactuator_setup": dict(
        type="web", ref="https://wiki.seeedstudio.com/myactuator_series/",
        also=["https://cdn.robotshop.com/rbm/acdeb3e1-04dc-4138-8b99-88509ddc4e94/7/"
              "709f6435-daac-4a0f-9e2a-de033477cc27/9af65af7_setup-software-instructionmanual-v3.pdf",
              "https://uk.robotshop.com/products/myactuator-can-adapter-analyzer-for-rmd-x-v4-rh-and-h-series"],
        says="the vendor's set-up software runs on a Windows PC and needs a USB-CAN interface of its own for CAN "
             "drives (a distributor's wiki; the software's package holds a driver for one adapter family); a "
             "retailer lists 'MyActuator CAN Adapter Analyzer for RMD-X V4, RH and H Series' at about 55 pounds. "
             "Whether the software accepts the PCAN-USB FD was not found"),
    "myactuator_ethercat": dict(
        type="web", ref="https://manuals.plus/m/afac3f8b251f8f42b18fa2bec083396ebc325afc4b36fc1d655cee2cdf580301",
        also=["https://cdn.robotshop.com/rbm/acdeb3e1-04dc-4138-8b99-88509ddc4e94/3/"
              "34224137-babc-4373-9ac9-9d1fbf7ad526/bbabf271_ethercat-control-protocol-v1.0-250430.pdf"],
        says="a mirror of MyActuator's 'EtherCAT Control Protocol' V1.0: cyclic synchronous position, velocity and "
             "torque modes (CSP, CSV, CST); no supported cycle time was found in the extract"),
    "myactuator_naming": dict(
        type="web", ref="https://www.switch-science.com/products/10516",
        says="a distributor's page for the RMD-X6-P20-60-E: in the model name, E = EtherCAT & CAN BUS and C = CAN "
             "BUS"),
    "myactuator_timeout": dict(
        type="web",
        ref="https://docs.ros.org/en/ros2_packages/humble/api/myactuator_rmd/generated/"
            "typedef_responses_8hpp_1abb5dc20fefee579cda714dd829f56c52.html",
        says="a community ROS 2 driver documents the drives' 'communication interruption protection time setting' "
             "command; what the drive does when that time runs out was not in the extract"),
    "can_frame_bits": dict(
        type="web", ref="https://lkml.iu.edu/hypermail/linux/kernel/2306.1/08769.html",
        also=["https://elearning.vector.com/mod/page/view.php?id=349"],
        says="a Linux kernel patch lists a classic CAN standard-format frame with 8 data bytes as 111 bits without "
             "bit stuffing and 135 bits with the worst stuffing; Vector's course gives at most 24 stuff bits and a "
             "132-bit longest frame before the 3-bit intermission"),
    "teensy41": dict(
        type="web", ref="https://pjrc.com/store/teensy41.html",
        also=["https://www.pjrc.com/teensy/card11b_rev4_web.pdf"],
        says="Teensy 4.1: ARM Cortex-M7 at 600 MHz; 3 CAN buses (1 with CAN FD). Its pinout card: 'Teensy 4.1 pins "
             "are not 5 volt tolerant. Do not apply more than 3.3V to any pin, except VIN & VUSB'"),
    "teensy41_price": dict(
        type="web", ref="https://www.sparkfun.com/teensy-4-1.html",
        says="SparkFun's listing of the Teensy 4.1 (DEV-16771): $31.50 each (seen on 14 Nov 2026; distributors' "
             "prices run higher). An earlier draft's $35.50 from PJRC's own page could not be confirmed"),
    "jetson_can": dict(
        type="web", ref="https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/HR/ControllerAreaNetworkCan.html",
        also=["https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/HR/ControllerAreaNetworkCan.html",
              "https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/"
              "jetson_orin_nano_devkit_carrier_board_specification_sp.pdf"],
        says="Jetson Linux developer guide: Orin NX and Nano have 1 CAN controller (driver mttcan, 50 MHz clock); the "
             "interface needs an external transceiver of at least 3.3 V; CAN FD mode up to 15 Mbps, and 'all types of "
             "transceivers can achieve' 5 Mbps; the dev kit's carrier has an optional 4-pin CAN header, J17"),
    "pcan_usb_fd": dict(
        type="web", ref="https://www.peak-system.com/products/hardware/external-pc-interfaces/pcan-usb-fd/",
        says="PCAN-USB FD: High-speed USB 2.0; CAN 25 kbit/s to 1 Mbit/s; CAN FD data field up to 12 Mbit/s; time "
             "stamp resolution 1 µs; galvanic isolation up to 500 V"),
    "rt_jetson": dict(
        type="web", ref="https://www.wimcorp.co.kr/tech/en/posts/embedded/preempt-rt-jetson-orin/",
        also=["https://proventusnova.com/blog/jetson-orin-preempt-rt-real-time-kernel",
              "https://forums.developer.nvidia.com/t/how-to-improve-real-time-performance-of-the-rt-kernal/338651",
              "https://wiki.seeedstudio.com/deploy_preempt_rt_kernel_with_prebuilt_deb_package_on_recomputer_jetson/",
              "https://forums.developer.nvidia.com/t/jetpack-6-2-agx-orin-64gb-preempt-rt-large-cyclictest-latency-"
              "variation-between-different-modules-up-to-90-us-even-with-cpu-isolation/381288",
              "https://forums.developer.nvidia.com/t/observing-latency-discrepancies-on-nvidia-jetson-nano-with-"
              "preempt-rt/329935"],
        says="third-party wake-up latency tests of a PREEMPT_RT kernel on other Jetson Orin modules: below 20 µs "
             "maximum on an AGX Orin only with an isolated core as well; 50-120 µs worst case on a quiet isolated "
             "core; one AGX module about 10 µs and another above 90 µs; an Orin Nano thread reports averages of "
             "about 25 µs (cores 0-3) and 40 µs (cores 4-5); forum threads report a maximum of about 200 µs on "
             "the real-time kernel, in one case down from 400 µs once the clocks were pinned. A page to read in "
             "full: a vendor's wiki tests the real-time kernel on a carrier with the Orin Nano Super module (6 "
             "cores, the CPU fully loaded: wake-up latency, pin jitter and an EtherCAT master at 1 kHz); its "
             "figures were not in the extract"),
    "xm540": dict(
        type="web", ref="https://emanual.robotis.com/docs/en/dxl/x/xm540-w270",
        says="XM540-W270 control table: Return Delay Time, default 250, range 0-254, unit 2 µs (500 µs by default); "
             "Baud Rate at address 8, initial value 1, range 0-7"),
    "xm540_baud": dict(
        type="web", ref="https://manualshive.com/ru/manual/robotis/xm540-w150/1471413?page=2",
        also=["https://robosavvy.com/store/robotis-dynamixel-xm540-w150-r.html"],
        says="a copy of the XM540-W150 manual, the same servo family: Baud Rate value 1 = 57,600 bit/s (the "
             "default), 3 = 1 Mbit/s; a retailer's listing gives the default as 57600 bps. Confirm on ROBOTIS's own "
             "page for the W270"),
    "dxl_ttl_level": dict(
        type="web", ref="https://emanual.robotis.com/docs/en/reference/dxl-selection-guide/",
        says="ROBOTIS's selection guide: the TTL bus is 0 to 5 V logic; its circuit 'is designed for 5V or 5V "
             "tolerant MCU. Otherwise, use a Level Shifter to match the voltage of MCU'"),
    "sts3215_serial": dict(
        type="web", ref="https://www.dfrobot.com/product-2961.html",
        also=["https://core-electronics.com.au/attachments/uploads/sts3215-smart-servo-datasheet-translated.pdf"],
        says="a distributor's listing of the ST3215: a TTL-level single-wire bus, high level 2-5 V, low level "
             "0-0.45 V; a translated datasheet gives the factory bit rate as 1 Mbit/s. The idle voltage of the data "
             "line was not found"),
    "adxl335": dict(
        type="web", ref="https://www.analog.com/en/products/adxl335",
        also=["https://community.element14.com/products/devtools/technicallibrary/w/documents/10547/"
              "analog-devices-eval-adxl335z-three-axis-accelerometer-evaluation-board"],
        says="ADXL335 analog accelerometer: the bandwidth is set by a capacitor on each output, 1 / (2 pi x 32 kOhm "
             "x C); its evaluation board's 100 nF capacitors give 50 Hz"),
    "u2d2": dict(
        type="web", ref="https://emanual.robotis.com/docs/en/parts/interface/u2d2/",
        also=["https://stanfordasl.github.io/trunk-stack/motors"],
        says="ROBOTIS U2D2 USB adapter: set the serial chip's latency timer to 1 ms (on Linux through "
             "/sys/bus/usb-serial/devices/ttyUSB0/latency_timer); a lab's notes give its default as 16 ms"),
    "sts3215_registers": dict(
        type="web", ref="https://docs.rs/crate/st3215/latest/source/src/registers.rs",
        also=["https://blog.youkuaiyun.com/gitblog_00860/article/details/151443061"],
        says="a third-party driver's register notes for the ST3215: return delay in units of 2 µs, at most 508 µs, "
             "initial value 0; another write-up gives a factory default of 500 µs: the two disagree"),
    "vl53l5cx": dict(
        type="web",
        ref="https://www.st.com/resource/en/user_manual/um2884-a-guide-to-using-the-vl53l5cx-multizone-timeofflight-"
            "ranging-sensor-with-a-wide-field-of-view-ultra-lite-driver-uld-stmicroelectronics.pdf",
        says="ST user manual UM2884: ranging frequency 1-60 Hz at 4 x 4 zones and 1-15 Hz at 8 x 8 zones"),
    "ethercat_rt": dict(
        type="web", ref="https://www.forlinx.net/article_view_776.html",
        also=["https://www.acontis.com/en/running-ethercat-in-real-time-on-linux.html"],
        says="EtherCAT masters on PREEMPT_RT Linux at a 1 ms cycle, on other boards: about 4-6 µs of jitter with the "
             "IgH master on a small network; about 22 µs with a vendor's master on boards including a Jetson AGX "
             "Orin"),
}

# ------------------------------------------------------------------------------------------------
# Inputs (each with its kind; (c) with sources, (d) with a basis)
# ------------------------------------------------------------------------------------------------
def _inp(value, unit, kind, source=(), basis="", note=""):
    return dict(value=value, unit=unit, kind=kind, source=list(source), basis=basis, note=note)


INPUTS = {
    # the model
    "model_step_ms": _inp(2.0, "ms", "c", ["plant_json", "mjcf_py"], note="the physics step, timestep_s 0.002"),
    "model_target_rate_hz": _inp(500, "Hz", "c", ["control_py"], note="servo targets every physics step"),
    "model_feedback_every_steps": _inp(5, "steps", "c", ["control_py"], note="FEEDBACK_EVERY: 100 Hz"),
    "model_sensor_age_ms": _inp(2.0, "ms", "c", ["plant_py"], note="sensors describe the last step's start"),
    "model_delay_steps_max": _inp(10, "steps", "c", ["env_py", "ledger_py"],
                                  note="latency_steps (0, 10) in the environment and LATENCY_MAX_STEPS = 10 in the "
                                       "ledger: 0-20 ms"),
    "judged_delay_ms": _inp(10, "ms", "c", ["decision_log"], note="P29: the bar is judged at 10 ms"),
    # the leg bus
    "leg_drive_count": _inp(12, "drives", "c", ["plant_json"], note="2 hip-pitch drives and 10 others"),
    "frames_per_drive_per_cycle": _inp(2, "frames", "c", ["myactuator_protocol"],
                                       note="a command (0x140 + ID) and its reply (0x240 + ID)"),
    "can_bit_rate_bps": _inp(1_000_000, "bit/s", "c", ["myactuator_protocol"]),
    "can_data_bytes": _inp(8, "bytes", "c", ["myactuator_protocol"], note="DLC 8, standard frame"),
    "can_frame_bits_plain": _inp(111, "bits", "c", ["can_frame_bits"],
                                 note="8 n + 47 for n = 8 data bytes, with the 3-bit intermission"),
    "can_frame_bits_stuffed": _inp(135, "bits", "c", ["can_frame_bits"],
                                   note="plus floor((34 + 8 n - 1) / 4) = 24 stuff bits at worst"),
    "leg_drive_can_fd": _inp(False, "", "d", basis="the author's reading: the vendor's protocol extracts list classic "
                             "CAN only (standard frames, DLC 8, 1 Mbit/s) and no CAN FD mode was found; the absence "
                             "is an inference, to confirm with the vendor"),
    "can_fd_data_bit_rate_bps": _inp(5_000_000, "bit/s", "d", basis="the author's choice for a hypothetical CAN FD "
                                     "drive: the data rate NVIDIA's guide says every transceiver reaches (jetson_can)"),
    "utilisation_ceiling": _inp(1.0, "of the cycle", "d", basis="the author's rule: the cycle fits if the worst-case "
                                "frame time is at most the cycle; a polled bus has one master and no other traffic, "
                                "but 100 % leaves no room for a resend or a status read"),
    # hosts
    "teensy_can_buses": _inp(3, "buses", "c", ["teensy41"], note="1 of them with CAN FD"),
    "jetson_can_controllers": _inp(1, "controllers", "c", ["jetson_can"], note="Orin NX and Nano"),
    "pcan_usb_fd_channels": _inp(1, "channels", "c", ["catalog_parts", "pcan_usb_fd"],
                                 note="catalog id pcan_usb_fd, a one-channel adapter"),
    "one_bus_rate_hz": _inp(250, "Hz", "d", basis="the author's choice: the fastest whole multiple of the model's "
                            "2 ms step that one bus carries with 12 drives at the worst stuffing (4 ms cycle)"),
    "drive_reply_turnaround_ms": _inp(None, "ms", "d", basis="not found by the author in the vendor's protocol "
                                      "extracts: the time a drive takes to start its reply after a command. The "
                                      "exchange times assume it is shorter than the command burst; the bench "
                                      "stopwatch measures it"),
    "drive_max_command_rate_hz": _inp(None, "Hz", "d", basis="not found by the author in the vendor's documents: the "
                                      "highest rate at which a drive accepts a command and answers each one. 500 Hz "
                                      "is assumed to be within it; the bench soak shows whether it is"),
    # EtherCAT
    "ethercat_offered": _inp(True, "", "c", ["catalog_actuators", "actuator_research", "myactuator_naming",
                                             "myactuator_ethercat"],
                             note="both leg-drive models are -E variants: CAN bus + EtherCAT"),
    "ethercat_bit_rate_bps": _inp(100_000_000, "bit/s", "d", basis="EtherCAT's usual 100 Mbit/s Ethernet, from the "
                                  "author's general knowledge; not re-read"),
    "ethercat_process_bytes_per_drive": _inp(24, "bytes", "d", basis="the author's assumption of a cyclic synchronous "
                                             "mapping: control word 2, target position 4, velocity offset 4 and "
                                             "torque offset 2 out; status word 2, position 4, velocity 4 and torque "
                                             "2 in; the vendor's mapping was not read"),
    "ethercat_overhead_bytes": _inp(52, "bytes", "d", basis="the Ethernet and EtherCAT frame layout from the "
                                    "author's general knowledge: preamble 8, Ethernet header 14, EtherCAT header 2, "
                                    "datagram header 10, working counter 2, checksum 4, inter-frame gap 12"),
    "ethercat_forward_us_per_drive": _inp(1.0, "µs", "d", basis="the author's allowance for each drive's forwarding "
                                          "delay; not sourced"),
    # the neck and tail buses
    "xm540_return_delay_default_us": _inp(500, "µs", "c", ["xm540"], note="250 x 2 µs; settable to 0"),
    "xm540_baud_default_bps": _inp(57_600, "bit/s", "c", ["xm540", "xm540_baud"],
                                   note="Baud Rate (address 8) initial value 1; 1 Mbit/s is value 3"),
    "sts3215_baud_default_bps": _inp(1_000_000, "bit/s", "c", ["sts3215_serial"], note="as delivered; read it back"),
    "teensy_pin_max_v": _inp(3.3, "V", "c", ["teensy41"], note="no pin takes more, except VIN and VUSB"),
    "dxl_ttl_logic_v": _inp(5.0, "V", "c", ["dxl_ttl_level"], note="the neck servo's data line"),
    "sts3215_return_delay_us": _inp([0, 500], "µs", "d", basis="two third-party write-ups disagree on the factory "
                                    "default (0 or 500 µs; sts3215_registers); read it from the servo"),
    "aux_baud_bps": _inp(1_000_000, "bit/s", "d", basis="the author's choice inside both servos' catalog ranges "
                         "(XM540 9.6 kbps-4.5 Mbps, STS3215 38.4 kbps-1 Mbps)"),
    "aux_bits_per_byte": _inp(10, "bits", "d", basis="8 data bits with a start and a stop bit, the usual serial "
                              "framing of both protocols; from the author's general knowledge"),
    "aux_rate_hz": _inp(100, "Hz", "d", basis="the author's plan: the neck and tail are not in the balance loop, so "
                        "their targets go out every fifth leg cycle; the model updates all 14 targets at 500 Hz"),
    "u2d2_latency_timer_ms": _inp([1, 16], "ms", "c", ["u2d2"], note="set to 1 ms; 16 ms by default"),
    # other parts
    "tof_rate_hz": _inp({"8x8": 15, "4x4": 60}, "Hz", "c", ["vl53l5cx"], note="the most the sensor is rated for"),
    "rt_latency_reported_ms": _inp([0.01, 0.2], "ms", "c", ["rt_jetson"],
                                   note="worst-case wake-up latency reported on Orin modules with the real-time "
                                        "kernel; 0.4 ms in one report before tuning"),
    "sts3215_vendor_rated_nm": _inp(0.980665, "N m", "c", ["catalog_actuators"],
                                    note="the spec sheet's rated torque, 10 kg cm at 12 V and 0.9 A, with no thermal "
                                         "definition (the catalog's vendor_rated_definition)"),
    "sts3215_stall_nm": _inp(2.94, "N m", "c", ["catalog_actuators"], note="30 kg cm at 12 V, 2.7 A (catalog notes)"),
    "box_servo_sustained_of_stall": _inp(0.167, "of the 12 V stall torque", "d",
                                         basis="PR #662's planning rule for box servos, which the study's catalog "
                                               "follows (designlab/data/build_catalog.py); not the author's and not "
                                               "a vendor figure"),
    "video_frame_ms": _inp([4.17, 8.33], "ms", "d", basis="the author's arithmetic: one frame of a phone's slow "
                           "motion at 240 and at 120 frames a second"),
    "motion_mode_kp_max": _inp(500.0, "per the manual's scale", "d",
                               basis="the author's reading of the KP range in a sibling model's manual "
                                     "(myactuator_sibling_manual); its units, and whether the 75 cm build's V4 and "
                                     "V4.1 drives share the frame, were not confirmed"),
    "motion_mode_kd_max": _inp(5.0, "per the manual's scale", "d",
                               basis="the author's reading of the KD range in the same sibling manual "
                                     "(myactuator_sibling_manual); units and applicability not confirmed"),
}

# ------------------------------------------------------------------------------------------------
# Delay terms: (low, high) in ms, each with its kind and basis.  ``bus_transmit`` and ``rate_hold`` are computed.
# ------------------------------------------------------------------------------------------------
def _term(lo, hi, kind, basis, source=()):
    return dict(lo_ms=lo, hi_ms=hi, kind=kind, basis=basis, source=list(source))


#: The outer-loop period of the leg drives if it is the slower of the two the vendor's manuals give (500 Hz), in ms
#: beyond the 1 ms the ``drive_can`` term takes: a sensitivity, printed beside the totals.
DRIVE_LOOP_SLOW_EXTRA_MS = 1.0

TERMS = {
    "compute_mcu": _term(0.10, 0.50, "d", "the author's estimate for one 500 Hz step of the tracker on a 600 MHz "
                         "microcontroller: the reference look-up, the targets, the ankle cap and a share of the "
                         "100 Hz feedback. No firmware exists (COMPUTE.md). The study's Python tracker takes about "
                         "0.5 ms per physics step, physics included, on the cloud machine (LESSONS_LEARNED.md 3.12)",
                         ["compute_md", "lessons", "teensy41"]),
    "compute_jetson": _term(0.02, 0.20, "d", "the author's estimate for the same step in compiled code on one "
                            "isolated Jetson core; not measured"),
    "jitter_mcu": _term(0.00, 0.05, "d", "the author's estimate for a timer-interrupt loop on a microcontroller with "
                        "serial and USB interrupts beside it; not measured"),
    "jitter_jetson_rt": _term(0.02, 0.50, "d", "low end: third-party tests of a PREEMPT_RT kernel on Orin modules "
                              "report about 0.01-0.2 ms at worst (rt_jetson). High end: the author's allowance for "
                              "the Orin Nano with the GPU loaded; no published figure for this board with the GPU "
                              "loaded was found",
                              ["rt_jetson"]),
    "path_mcu_can": _term(0.00, 0.02, "d", "the author's estimate: the microcontroller's own CAN controller, a "
                          "transmit mailbox away from the bus"),
    "path_usb_can": _term(0.20, 1.00, "d", "the author's estimate for one USB hop: a High-speed USB 2.0 transfer "
                          "waits for a 0.125 ms microframe, then the adapter's firmware and the Linux USB stack; no "
                          "measured latency of this adapter was found", ["pcan_usb_fd"]),
    "path_native_can": _term(0.02, 0.20, "d", "the author's estimate for a socket write through the Linux CAN driver "
                             "to the Orin's own controller", ["jetson_can"]),
    "path_ethercat": _term(0.02, 0.10, "d", "the author's estimate for a raw Ethernet send from a real-time thread; "
                           "published jitter of EtherCAT masters on other boards is 0.004-0.022 ms (ethercat_rt)",
                           ["ethercat_rt"]),
    "drive_can": _term(0.10, 1.50, "d", "the author's estimate from a frame's end to the torque change: up to one "
                       "period of the drive's outer loop, taken as 1 ms, plus the current's rise (winding L / R: "
                       "0.18 mH / 0.4 ohm = 0.45 ms on the X4-36-L and 0.51 mH / 0.41 ohm = 1.24 ms on the X6-60, "
                       "catalog notes; a closed current loop is faster). The vendor's driver manuals give a "
                       "position loop of 1 kHz (V3.0) or 500 Hz (V2.1) for the MC-X drivers of the earlier RMD-X "
                       "generation, with 5 kHz speed and 15 kHz current loops (myactuator_mc_driver); which loop "
                       "closes the motion-mode gains on the V4 and V4.1 drives is not known. With a 500 Hz loop "
                       "the high end would be 2.50 ms",
                       ["myactuator_mc_driver", "catalog_actuators"]),
    "drive_ethercat": _term(0.25, 2.50, "d", "the author's estimate: a cyclic synchronous drive takes its setpoint at "
                            "its next sync event, up to one 2 ms cycle, then the torque rises as over CAN; the "
                            "vendor's cycle times were not found", ["myactuator_ethercat"]),
}

# ------------------------------------------------------------------------------------------------
# Architectures
# ------------------------------------------------------------------------------------------------
FAIL_MCU_STOP = dict(
    short="Controlled stop, then stand (once the firmware exists)",
    detail="The microcontroller keeps the walking loop, the host's velocity request expires and the stop runs from "
           "the reference: a controlled stop, then a stand. Meets the requirement once the firmware exists; it is "
           "unwritten (COMPUTE.md).")
FAIL_MCU_CRASH = dict(
    short="Torque-off or frozen drives; a fall when walking",
    detail="No commands reach the drives until the microcontroller's watchdog restarts it, and Linux has no path to "
           "the leg bus. What the drives do without commands is not confirmed (they have a time-out setting): plan "
           "on torque-off, which for a walking biped is a fall. Standing, the robot may stay up if the restart is "
           "quick and the drives hold.")
FAIL_LINUX_DIRECT = dict(
    short="Torque-off or frozen drives; a fall when walking",
    detail="The walking loop is the thing that crashed, so the drives get no new targets: torque-off or a frozen "
           "last command, then a fall when walking. Does not meet the requirement. A supervisor can only cut power "
           "unless it can drive the leg bus itself, and then it is architecture B's microcontroller.")
FAIL_LINUX_ETHERCAT = dict(
    short="Torque-off or the drives' fault reaction; a fall when walking",
    detail="The EtherCAT master is the thing that crashed: torque-off or the drives' own fault reaction, then a "
           "fall when walking. Does not meet the requirement, unless the master runs on a microcontroller, which "
           "nobody here has shown.")
FAIL_NO_MCU = dict(
    short="None in the leg loop",
    detail="No microcontroller in the leg loop. If a supervisor is added and it hangs, its fail-safe output "
           "decides: an enable line that drops means torque-off.")

ARCHITECTURES = {
    "A": dict(name="Raspberry Pi 5 + Teensy 4.1, two CAN buses (COMPUTE.md's design, the baseline)",
              host="Raspberry Pi 5 + AI HAT+ (P14)", realtime="Teensy 4.1", bus="can", buses=2, rate_hz=500,
              terms=dict(compute="compute_mcu", host_jitter="jitter_mcu", host_path="path_mcu_can",
                         drive="drive_can"),
              rate_note="Two of the Teensy's three CAN ports, one leg (6 drives) on each.",
              linux_crash=FAIL_MCU_STOP, mcu_crash=FAIL_MCU_CRASH),
    "A3": dict(name="the same with three CAN buses (an option: the Teensy has three ports)",
               host="Raspberry Pi 5 + AI HAT+ (P14)", realtime="Teensy 4.1", bus="can", buses=3, rate_hz=500,
               terms=dict(compute="compute_mcu", host_jitter="jitter_mcu", host_path="path_mcu_can",
                          drive="drive_can"),
               rate_note="4 drives on each bus; a third transceiver and harness branch.",
               linux_crash=FAIL_MCU_STOP, mcu_crash=FAIL_MCU_CRASH),
    "B": dict(name="Jetson Orin Nano Super + Teensy 4.1, two CAN buses (the same real-time layer, another host)",
              host="Jetson Orin Nano Super dev kit", realtime="Teensy 4.1", bus="can", buses=2, rate_hz=500,
              terms=dict(compute="compute_mcu", host_jitter="jitter_mcu", host_path="path_mcu_can",
                         drive="drive_can"),
              rate_note="As A: the host is outside the servo loop, so the budget is A's.",
              linux_crash=FAIL_MCU_STOP, mcu_crash=FAIL_MCU_CRASH),
    "C1": dict(name="Jetson-direct, one USB-CAN channel (PR #668's layout, unverified)",
               host="Jetson Orin Nano Super dev kit", realtime="an isolated PREEMPT_RT core", bus="can", buses=1,
               rate_hz=None,
               terms=dict(compute="compute_jetson", host_jitter="jitter_jetson_rt", host_path="path_usb_can",
                          drive="drive_can"),
               rate_note="One PCAN-USB FD (catalog pcan_usb_fd). One bus cannot carry 12 drives at 500 Hz.",
               linux_crash=FAIL_LINUX_DIRECT, mcu_crash=FAIL_NO_MCU),
    "C2": dict(name="Jetson-direct, two USB-CAN channels",
               host="Jetson Orin Nano Super dev kit", realtime="an isolated PREEMPT_RT core", bus="can", buses=2,
               rate_hz=500,
               terms=dict(compute="compute_jetson", host_jitter="jitter_jetson_rt", host_path="path_usb_can",
                          drive="drive_can"),
               rate_note="Two adapters (2 x pcan_usb_fd) or one two-channel adapter, which the catalog does not hold.",
               linux_crash=FAIL_LINUX_DIRECT, mcu_crash=FAIL_NO_MCU),
    "D": dict(name="Jetson-direct, the Orin's own CAN controller with a transceiver",
              host="Jetson Orin Nano Super dev kit", realtime="an isolated PREEMPT_RT core", bus="can", buses=1,
              rate_hz=None,
              terms=dict(compute="compute_jetson", host_jitter="jitter_jetson_rt", host_path="path_native_can",
                         drive="drive_can"),
              rate_note="The Orin Nano has one CAN controller, so one bus: it cannot carry 12 drives at 500 Hz.",
              linux_crash=FAIL_LINUX_DIRECT, mcu_crash=FAIL_NO_MCU),
    "E": dict(name="EtherCAT from the Jetson's Ethernet port (the leg drives are -E variants)",
              host="Jetson Orin Nano Super dev kit", realtime="an isolated PREEMPT_RT core", bus="ethercat", buses=1,
              rate_hz=500,
              terms=dict(compute="compute_jetson", host_jitter="jitter_jetson_rt", host_path="path_ethercat",
                         drive="drive_ethercat"),
              rate_note="One frame carries all 12 drives. The drives' shortest supported cycle was not found.",
              linux_crash=FAIL_LINUX_ETHERCAT, mcu_crash=FAIL_NO_MCU),
}
ARCH_ORDER = ("A", "A3", "B", "C1", "C2", "D", "E")

# ------------------------------------------------------------------------------------------------
# The neck and tail buses (packet sizes from the protocols' layouts as the author knows them: kind d)
# ------------------------------------------------------------------------------------------------
AUX = {
    "neck": dict(part="robotis_xm540_w270_t", joint="neck pitch", protocol="DYNAMIXEL Protocol 2.0, TTL half-duplex",
                 write_bytes=(16, 11), read_bytes=(14, 21),
                 packets="write goal position (4 bytes): 16 bytes out, 11 back; read current, velocity and position "
                         "(10 bytes): 14 bytes out, 21 back",
                 return_delay_us=(0, 500), usb_adapter="ROBOTIS U2D2 (not in the catalog)",
                 factory_baud="xm540_baud_default_bps"),
    "tail": dict(part="feetech_sts3215_c018", joint="tail yaw", protocol="Feetech STS serial protocol, TTL half-duplex",
                 write_bytes=(13, 6), read_bytes=(8, 12),
                 packets="write position, time and speed (6 bytes): 13 bytes out, 6 back; read position, speed and "
                         "load (6 bytes): 8 bytes out, 12 back",
                 return_delay_us=(0, 500), usb_adapter="Feetech FE-URT-1 (not in the catalog)",
                 factory_baud="sts3215_baud_default_bps"),
}

# ------------------------------------------------------------------------------------------------
# The model's delay, with file and line (the line is found from the text when the record is made)
# ------------------------------------------------------------------------------------------------
MODEL = (
    dict(id="rate", says="The controller sets all 14 servo targets every physics step of 2 ms (500 Hz).",
         refs=(("designlab/stage1/control.py", "Servo target (plan section 2), every physics step (500 Hz)"),
               ("designlab/mjcf.py", 'timestep="0.002"'))),
    dict(id="target", says="A target is an angle, a rate and a feedforward torque folded into one number per drive: "
                           "ctrl = q + (kv qd + tau) / kp, in front of an ideal PD servo at the holding gains.",
         refs=(("designlab/stage1/control.py", "ctrl = q_ref + dq + (kv qd_ref + tau_ff + dtau) / kp"),
               ("designlab/stage1/plant.py", "def command(self, q=None, qd=None, tau=None) -> None:"))),
    dict(id="queue", says="The delay: every target, feedforward included, passes through a queue of the last k + 1 "
                          "targets, so the servos get the target set k physics steps (2 k ms) earlier.",
         refs=(("designlab/stage1/env.py", "def _latency_on(self, steps: int) -> None:"),
               ("designlab/stage1/env.py", "queue = deque([p.d.ctrl.copy() for _ in range(steps)], maxlen=steps + 1)"),
               ("designlab/stage1/env.py", "ctrl[:] = queue[0]"))),
    dict(id="ledger_queue", says="The ledger's delay readings quoted below were made by the ledger's own copy of the "
                                 "same queue, refilled with the reset's targets at every reset.",
         refs=(("designlab/stage1/ledger.py", "def _latency(p: plant.Plant, steps: int):"),)),
    dict(id="range", says="k is a whole number of steps, 0 to 10 (0-20 ms): the training environment's random range "
                          "and the ledger's search range.",
         refs=(("designlab/stage1/env.py", "latency_steps: tuple = (0, 10)"),
               ("designlab/stage1/ledger.py", "LATENCY_MAX_STEPS = 10"))),
    dict(id="order", says="Each step: set the target, step the physics, then read the sensors for the next step.",
         refs=(("designlab/stage1/control.py", "p.command(q_cmd, qd_c, tau_cmd)"),
               ("designlab/stage1/control.py", "        p.step()", "p.command(q_cmd, qd_c, tau_cmd)"),
               ("designlab/stage1/control.py", "self.est.step(self.sensors.read())"),
               ("designlab/stage1/plant.py", "mujoco.mj_step(m, d)"))),
    dict(id="sensor_age", says="The sensors the controller reads describe the state one physics step (2 ms) earlier.",
         refs=(("designlab/stage1/plant.py", "that step's start, one physics step (2 ms) before ``d.qpos``"),)),
    dict(id="feedback", says="The balance feedback is computed every fifth step (100 Hz) and ramped into the targets: "
                             "one fifth in the step it is computed, all of it four steps (8 ms) later.",
         refs=(("designlab/stage1/control.py", "FEEDBACK_EVERY = 5"),
               ("designlab/stage1/control.py", 'FEEDBACK_HOLD = "ramp"'),
               ("designlab/stage1/control.py", "j = self.k % FEEDBACK_EVERY"),
               ("designlab/stage1/control.py", "a = (j + 1) / FEEDBACK_EVERY"))),
    dict(id="gains", says="The holding gains: kp = the drive's sustained planning torque / 0.01 rad (an assumption), "
                          "kv = 2 sqrt(kp I), critical damping on the inertia I the drive sees at home with "
                          "every other drive held: the leg's, not the bare shaft's.",
         refs=(("designlab/carpet.py", "SERVO_ERROR_RAD = 0.01"),
               ("designlab/stage1/plant.py", "kp = np.array([dr.sustained for dr in drives]) / carpet.SERVO_ERROR_RAD"),
               ("designlab/stage1/plant.py", "kv = 2.0 * carpet.SERVO_ZETA * np.sqrt(kp * inertia)"))),
    dict(id="passive", says="Every joint has the generator's default damping, 0.2 N m s/rad, and friction loss, "
                            "0.15 N m: unsourced.",
         refs=(("designlab/mjcf.py", 'damping="0.2", frictionloss="0.15"'),)),
    dict(id="n8", says="Stage 1b's note N8 states the same and what a bench figure is compared with.",
         refs=(("stage1/runs/2026-11-10-speed/STAGE1B_SPEED.md",
                "**N8. Servo delay: Stage 1's ledger, the model and the bench**"),)),
)

# ------------------------------------------------------------------------------------------------
# The bench sheet's parts
# ------------------------------------------------------------------------------------------------
BENCH_PARTS = (
    # (catalog file, id or None, what, run 1 quantity, run 2 quantity, note); a None id is not in the catalog
    ("actuators", "myactuator_rmd_x6_p20_60_e", "leg drive, hip pitch", 1, 2, ""),
    ("actuators", "myactuator_rmd_x4_p36_36_e_l", "leg drive, every other leg joint", 1, 10, ""),
    ("actuators", "robotis_xm540_w270_t", "neck servo", 1, 1, ""),
    ("actuators", "feetech_sts3215_c018", "tail servo", 1, 1, "a spare is cheap: the thermal test may age one"),
    ("parts", "nvidia_orin_nano_super_devkit", "Jetson dev kit (path 2)", 0, 0,
     "on hand: COMPUTE.md's bench test 1 measures 'the current dev kit' with no purchase"),
    ("parts", "pcan_usb_fd", "USB-CAN adapter (path 2)", 2, 2,
     "one channel each, so two for the two buses that 500 Hz needs; bench tools that this sheet uses to measure "
     "the Jetson path (their price is in the totals row)"),
    ("parts", "pololu_d36v50f12", "24 V to 12 V regulator for the neck and tail servos", 1, 1, ""),
    ("parts", "st_vl53l5cx_carrier", "chest ToF sensor (follow-on sheet C)", 1, 2, ""),
    ("parts", "lipo_6s_3300", "the build's 6S pack (optional: repeat run 2 on a sagging supply)", 0, 0,
     "not needed for the delay itself"),
)
BENCH_EXTRA = (
    # not in the catalog: (what, run 1 quantity, run 2 quantity, price, kind, source keys, basis)
    ("Teensy 4.1: one runs path 1's loop, the other is the stopwatch on both paths", 2, 2, 31.50, "c",
     ["teensy41", "teensy41_price"], "SparkFun's listing, seen on 14 Nov 2026"),
    ("3.3 V CAN transceiver boards for the Teensys, 1 Mbit/s: three for the loop, three for the stopwatch", 6, 6, 10.0,
     "d", [], "the author's allowance per board; not in the catalog and not priced"),
    ("ROBOTIS U2D2 USB adapter for the neck servo's bus: path 2, and ROBOTIS's set-up tool", 1, 1, 40.0, "d", [],
     "the author's allowance; not in the catalog and not priced"),
    ("Feetech FE-URT-1 USB adapter for the tail servo's bus: path 2, and Feetech's set-up tool", 1, 1, 20.0, "d", [],
     "the author's allowance from retail listings of about $19 seen in a web search; not in the catalog"),
    ("3.3 V to 5 V level-shifting half-duplex buffer for a Teensy serial port: path 1, one per servo bus", 2, 2, 10.0,
     "d", ["teensy41", "dxl_ttl_level"],
     "the author's allowance. Needed: the Teensy's pins take 3.3 V at most and the neck servo's data line is 5 V "
     "logic"),
    ("the vendor's USB-CAN adapter for the leg drives' set-up software (which also needs a Windows PC)", 1, 1, 70.0,
     "d", ["myactuator_setup"],
     "the author's allowance from a retail listing at about 55 pounds seen in a web search; whether the software "
     "accepts the PCAN-USB FD instead was not found"),
    ("piezo disc for method M2, with a 1 megohm load resistor, a series resistor and two clamp diodes, on a clamp-on "
     "arm", 1, 1, 10.0, "d", [], "the author's allowance"),
    ("dial indicator (0.01 mm) on a stand, or a laser pointer, for sheet A", 1, 1, 30.0, "d", [],
     "the author's allowance"),
    ("bus wire (twisted pair), two D-Sub 9 breakouts for the USB-CAN adapters, 120 ohm terminators, an indicator "
     "light", 1, 1, 30.0, "d", [], "the author's allowance"),
    ("emergency-stop switch in the drive supply, fuse, clamps, lever arms and weights", 1, 1, 60.0, "d", [],
     "the author's allowance"),
    ("tread sample for sheet D: cast polyurethane to the study's specification, or a TPU 85 A print for a fast trial",
     1, 1, 30.0, "d", ["materials_md"], "the author's allowance; the specification is in round2/refine/MATERIALS.md"),
    ("24 V bench supply with a current limit, 10 A or more", 1, 1, None, "d", ["cost_py", "candidate_json"],
     "not priced here: a tether supply is part of the cost model's bench lot, which is {bench_lot} for this body "
     "(low / target / high: the round-two record's cost.lines.bench)"),
)

#: Follow-on sheet A: the lever arm of the stiffness test; sheet E: the tail servo's holds (N m) and their sources.
LEVER_M = 0.20
THERMAL_LEVER_M = 0.10
THERMAL_HOLDS = (
    ("the model's sustained planning torque: 0.167 x the 12 V stall torque (PR #662's rule for box servos; A1)", None,
     "c", ["plant_json"]),
    ("the 75 cm brisk walk's closed-loop load, upper figure (Stage 1, R2; A1's update)", 0.52, "b", ["s1b_a1"]),
    ("the 75 cm animal-pace walk's closed-loop load, upper figure (Stage 1, R2; A1's update)", 0.64, "b",
     ["s1b_a1"]),
)
G = 9.81

#: The delay steps (sheet section 4.1): the step, the pointer and the bench supply's first current limit.
STEP_RAD = 0.01
POINTER_M = 0.20
SUPPLY_LIMIT_A = 2.0
SUPPLY_V = 24.0

#: Follow-on sheet B: constant speeds per drive type (rad/s), below each servo's own no-load speed.
FRICTION_SPEEDS = {
    "myactuator_rmd_x6_p20_60_e": (0.5, 1.0, 2.0, 4.0),
    "myactuator_rmd_x4_p36_36_e_l": (0.5, 1.0, 2.0, 4.0),
    "robotis_xm540_w270_t": (0.25, 0.5, 1.0, 1.5),
    "feetech_sts3215_c018": (0.5, 1.0, 1.5, 2.5),
}

#: The results table's rows (sheet section 5): (drive, path, buses, load condition, target rate Hz, run).
_LEG = ("X6-P20-60-E", "X4-P36-36-E-L")
RESULT_ROWS = tuple(
    (drive, path, buses, load, rate, "1 and 2")
    for drive in _LEG
    for path, buses, load, rate in (("1: Teensy", "2", "idle", 500), ("1: Teensy", "2", "loaded", 500),
                                    ("1: Teensy", "3", "loaded", 500), ("2: Jetson", "2", "idle", 500),
                                    ("2: Jetson", "2", "loaded", 500), ("2: Jetson", "1", "loaded", 250))
) + tuple(
    (drive, path, "own", "loaded", 100, "1")
    for drive in ("XM540-W270", "STS3215") for path in ("1: Teensy", "2: Jetson")
)
SOAK_ROWS = (("1: Teensy", "2", "idle"), ("1: Teensy", "2", "loaded"), ("1: Teensy", "3", "loaded"),
             ("2: Jetson", "2", "idle"), ("2: Jetson", "2", "loaded"))
SOAK_MIN = 30
COMMANDS = 1000
THERMAL_HOLD_MIN = 20

#: Bench time, hours (low, high): the author's estimates (d); none is measured.
BENCH_TIME = (
    ("run1", "Unpack, set identifiers and bit rates with the vendors' tools, set each drive's own torque limit, clamp "
             "and wire one drive of each type", 1.5, 2.5),
    ("run1", "Path 1 set-up: the loop Teensy and the stopwatch Teensy (which replays the absent drives), three "
             "transceivers each, the two level-shifting buffers", 1.0, 2.0),
    ("run1", "Path 2 set-up: real-time kernel, one isolated core, two USB-CAN adapters, the tick lines to the "
             "stopwatch, a wake-up latency check", 2.0, 4.0),
    ("run1", "Delay runs: the {n_rows} rows of the results table x {commands:,} commands (about 10 min a row with its "
             "set-up), method M2 on one row per path and drive type, and the 300-command check of M1 fine", 3.5, 5.0),
    ("run1", "Soak logs: the {n_soak} rows of the soak table x {soak_min} min, levers off; may be left alone once the "
             "drives' torque limits are set", 2.5, 3.0),
    ("run2", "Wire all 12 leg drives on two buses (then three), with terminators", 2.0, 4.0),
    ("run2", "Repeat the {n_leg_rows} leg-drive rows and the {n_soak} soak logs with 12 real drives", 4.5, 6.0),
    ("follow", "A. Servo stiffness: 4 drive types", 2.0, 3.0),
    ("follow", "B. Joint friction and damping: 4 drive types", 2.0, 3.0),
    ("follow", "C. Chest ToF stop timing: 2 modes x 100 trials on video", 1.5, 2.5),
    ("follow", "D. Tread grip: 4 floors x 10 pulls, clean and dusty, and the spin test", 2.0, 3.0),
    ("follow", "E. Tail servo thermal: {n_holds} holds of {hold_min} min (3 static, 3 swinging) with cooling between, "
               "attended", 3.5, 5.0),
)

DELAY_MAP = (
    # (label, k or None, what to re-run when k is None, reading)
    ("above 0 and at most 2 ms", 1, None,
     "Not simulated: no delay search on any body ran 2 ms. Run the bar at 2 ms before reading anything; its "
     "neighbours, 0 and 4 ms, are not a reading of it."),
    ("above 2 and at most 4 ms", 2, None,
     "In the model the 75 cm nominal walk met every limit here. The 0.40 m/s bar has no record at this delay: re-run "
     "it at 4 ms."),
    ("above 4 and at most 6 ms", 3, None,
     "The largest delay at which the model's 75 cm nominal walk met every limit. Re-run the bar at 6 ms."),
    ("above 6 and at most 8 ms", 4, None,
     "The model's walk stays up and misses a limit. The fitted tracker is at its edge; re-run the bar at 8 ms."),
    ("above 8 and at most 10 ms", 5, None,
     "The model's walk falls in 3 of 3 episodes under both controllers. This is today's judged delay (P29). A "
     "reading here would count against the fitted tracker as it stands, not against every controller (Stage 1b, "
     "note N8)."),
    ("above 10 and at most 18 ms", None, "the whole 2 ms step above it (k 6–9)",
     "Never simulated on any body (12-18 ms). No reading until the bar is run at the whole step above the "
     "measurement."),
    ("above 18 and at most 20 ms", 10, None,
     "The model's walk stayed up without meeting every limit. The response is not monotonic and is unexplained, so "
     "this says nothing about 12-18 ms."),
    ("above 20 ms", None, "nothing: outside the simulation's range (k stops at 10)",
     "Cannot be mapped onto a simulated delay. Report it as measured; the delay range of the model would have to be "
     "widened first."),
)


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------
def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _r(x, nd=4):
    return round(float(x), nd)


def _load(rel: str):
    return json.loads((STUDY / rel).read_text())


def find_line(rel: str, text: str, after: str | None = None) -> int:
    """The 1-based number of the only line of ``rel`` that contains ``text``; with ``after``, the first such line
    below the only line that contains ``after``."""
    lines = (STUDY / rel).read_text().splitlines()
    start = find_line(rel, after) if after else 0
    hits = [i for i, line in enumerate(lines, 1) if text in line and i > start]
    if not hits or (after is None and len(hits) != 1):
        raise ValueError(f"{rel}: {len(hits)} lines contain {text!r}; the reference needs exactly one")
    return hits[0]


def k_of(delay_ms: float, step_ms: float = 2.0) -> int:
    """The whole number of physics steps a delay maps onto: rounded up, never read between delays."""
    return int(math.ceil(delay_ms / step_ms - 1e-9))


# ------------------------------------------------------------------------------------------------
# 1. Frames
# ------------------------------------------------------------------------------------------------
def classic_can_bits(n: int) -> dict:
    """Bits of a classic CAN data frame with an 11-bit identifier and ``n`` data bytes, intermission included."""
    plain = 8 * n + 47
    stuff = (34 + 8 * n - 1) // 4
    return dict(plain=plain, stuff_max=stuff, stuffed=plain + stuff)


def can_fd_bits(n: int = 8) -> dict:
    """A CAN FD data frame (11-bit identifier, ``n`` <= 16 data bytes, 17-bit CRC) by phase: (d), the author's count
    from the frame layout.  Arbitration phase: SOF 1, identifier 11, RRS 1, IDE 1, FDF 1, reserved 1, BRS 1 = 17 bits,
    then CRC delimiter 1, ACK 1, ACK delimiter 1, EOF 7 and intermission 3 = 13 bits at the nominal rate.  Data phase:
    ESI 1, DLC 4, data 8 n, stuff count 4, CRC 17 and 6 fixed stuff bits at the data rate.  Dynamic stuff bits apply
    from SOF to the end of the data: at most floor((17 - 1) / 4) = 4 in the arbitration part and the rest of
    floor((22 + 8 n - 1) / 4) in the data part."""
    arb, data = 17 + 13, 1 + 4 + 8 * n + 4 + 17 + 6
    stuff_total = (17 + 5 + 8 * n - 1) // 4
    stuff_arb = (17 - 1) // 4
    return dict(arbitration=arb, data=data, stuff_arbitration_max=stuff_arb, stuff_data_max=stuff_total - stuff_arb)


def frames() -> dict:
    v = {k: INPUTS[k]["value"] for k in INPUTS}
    n, rate = v["can_data_bytes"], v["can_bit_rate_bps"]
    c = classic_can_bits(n)
    assert (c["plain"], c["stuffed"]) == (v["can_frame_bits_plain"], v["can_frame_bits_stuffed"])
    classic = dict(bits_plain=c["plain"], stuff_bits_max=c["stuff_max"], bits_stuffed=c["stuffed"], bit_rate_bps=rate,
                   us_plain=_r(c["plain"] / rate * 1e6, 3), us_stuffed=_r(c["stuffed"] / rate * 1e6, 3), kind="c",
                   arithmetic=f"8 x {n} + 47 = {c['plain']} bits; + floor((34 + 8 x {n} - 1) / 4) = {c['stuff_max']} "
                              f"stuff bits = {c['stuffed']} bits; at {rate / 1e6:g} Mbit/s one bit is "
                              f"{1e6 / rate:g} µs")
    f = can_fd_bits(n)
    drate = v["can_fd_data_bit_rate_bps"]
    fd_lo = f["arbitration"] / rate + f["data"] / drate
    fd_hi = (f["arbitration"] + f["stuff_arbitration_max"]) / rate + (f["data"] + f["stuff_data_max"]) / drate
    fd = dict(f, nominal_bit_rate_bps=rate, data_bit_rate_bps=drate, us_plain=_r(fd_lo * 1e6, 3),
              us_stuffed=_r(fd_hi * 1e6, 3), kind="d", available_on_leg_drives=v["leg_drive_can_fd"],
              arithmetic=f"{f['arbitration']} bits at {rate / 1e6:g} Mbit/s + {f['data']} bits at {drate / 1e6:g} "
                         f"Mbit/s; at worst + {f['stuff_arbitration_max']} and + {f['stuff_data_max']} stuff bits")
    nd = v["leg_drive_count"]
    ebytes = v["ethercat_overhead_bytes"] + nd * v["ethercat_process_bytes_per_drive"]
    wire = ebytes * 8 / v["ethercat_bit_rate_bps"] + nd * v["ethercat_forward_us_per_drive"] * 1e-6
    ecat = dict(bytes=ebytes, us=_r(wire * 1e6, 3), kind="d",
                arithmetic=f"{v['ethercat_overhead_bytes']} + {nd} x {v['ethercat_process_bytes_per_drive']} = "
                           f"{ebytes} bytes x 8 / 100 Mbit/s + {nd} x {v['ethercat_forward_us_per_drive']:g} µs")
    return dict(classic_can=classic, can_fd_hypothetical=fd, ethercat=ecat)


# ------------------------------------------------------------------------------------------------
# 2. Bus load
# ------------------------------------------------------------------------------------------------
def bus_row(drives_on_bus: int, rate_hz: float, us_plain: float, us_stuffed: float) -> dict:
    n = INPUTS["frames_per_drive_per_cycle"]["value"] * drives_on_bus
    lo, hi = n * us_plain / 1000.0, n * us_stuffed / 1000.0
    cycle = 1000.0 / rate_hz
    return dict(drives_on_bus=drives_on_bus, frames_per_cycle=n, rate_hz=rate_hz, cycle_ms=_r(cycle),
                ms_per_cycle_plain=_r(lo), ms_per_cycle_stuffed=_r(hi), load_plain=_r(lo / cycle),
                load_stuffed=_r(hi / cycle), fits=bool(hi <= cycle * INPUTS["utilisation_ceiling"]["value"] + 1e-12),
                max_rate_hz_stuffed=_r(1000.0 / hi, 1), max_rate_hz_plain=_r(1000.0 / lo, 1))


def bus_load(fr: dict) -> list:
    c = fr["classic_can"]
    nd = INPUTS["leg_drive_count"]["value"]
    rows = []
    for buses in (1, 2, 3):
        for rate in (250, 500, 1000):
            rows.append(dict(bus="classic CAN, 1 Mbit/s", buses=buses, kind="c",
                             **bus_row(nd // buses, rate, c["us_plain"], c["us_stuffed"])))
    f = fr["can_fd_hypothetical"]
    rows.append(dict(bus="CAN FD, 1 and 5 Mbit/s (hypothetical: not offered by the leg drives as far as found)",
                     buses=1, kind="d", **bus_row(nd, 500, f["us_plain"], f["us_stuffed"])))
    return rows


# ------------------------------------------------------------------------------------------------
# 3. Architectures
# ------------------------------------------------------------------------------------------------
def architecture(key: str, fr: dict) -> dict:
    a = ARCHITECTURES[key]
    nd = INPUTS["leg_drive_count"]["value"]
    step = INPUTS["model_step_ms"]["value"]
    model_age = INPUTS["model_sensor_age_ms"]["value"]
    n_bus = nd // a["buses"]
    t = {name: TERMS[tk] for name, tk in a["terms"].items()}
    if a["bus"] == "can":
        c = fr["classic_can"]
        at_500 = bus_row(n_bus, 500, c["us_plain"], c["us_stuffed"])
        rate = a["rate_hz"] or INPUTS["one_bus_rate_hz"]["value"]
        row = bus_row(n_bus, rate, c["us_plain"], c["us_stuffed"])
        holds_500 = at_500["fits"]
        f_lo, f_hi = c["us_plain"] / 1000.0, c["us_stuffed"] / 1000.0
        # the commands go out back to back, lowest identifier first; the replies (higher identifiers) follow
        bus_lo, bus_hi = f_lo, n_bus * f_hi
        bus_basis = (f"computed: the first drive has its command after 1 frame without stuffing ({_r(f_lo, 3)} ms), "
                     f"the last after {n_bus} frames at the worst stuffing ({_r(bus_hi, 3)} ms). The frame times "
                     "are sourced (c); the order is the author's assumption (d): the commands are sent back to "
                     "back and win arbitration over the replies (lower identifiers)")
        bus_kind = "d"
        if holds_500:
            rate_held = dict(hz=rate, kind="c", text=f"{rate:g} Hz fits the bus (c)",
                             basis="bus-time arithmetic on sourced frame lengths; it takes the drives' reply "
                                   "turnaround as shorter than the command burst and assumes the drives accept a "
                                   "command each at this rate, neither of which was found (d)")
        else:
            rate_held = dict(hz=rate, kind="d",
                             text=(f"**500 Hz does not fit** (c: {_pct(at_500['load_plain'], at_500['load_stuffed'])}"
                                   f" load); {rate:g} Hz chosen (d)"),
                             basis="that 500 Hz does not fit one bus is bus-time arithmetic on sourced frame "
                                   "lengths (c); the slower rate is the author's choice (d: one_bus_rate_hz)")
    else:
        e = fr["ethercat"]
        rate = a["rate_hz"]
        cycle = 1000.0 / rate
        row = dict(drives_on_bus=nd, frames_per_cycle=1, rate_hz=rate, cycle_ms=_r(cycle),
                   ms_per_cycle_plain=_r(e["us"] / 1000.0), ms_per_cycle_stuffed=_r(e["us"] / 1000.0),
                   load_plain=_r(e["us"] / 1000.0 / cycle), load_stuffed=_r(e["us"] / 1000.0 / cycle), fits=True,
                   max_rate_hz_stuffed=_r(1e6 / e["us"], 1), max_rate_hz_plain=_r(1e6 / e["us"], 1))
        at_500, holds_500 = row, True
        f_lo = f_hi = e["us"] / 1000.0
        bus_lo = bus_hi = f_lo
        bus_basis = ("computed from the author's assumed frame: one pass of one Ethernet frame through all 12 "
                     "drives")
        bus_kind = "d"
        rate_held = dict(hz=rate, kind="d",
                         text=f"{rate:g} Hz assumed (d): the drives' supported cycle was not found",
                         basis="the author's assumption: the frame is an assumed layout and the drives' shortest "
                               "supported cycle was not found")
    cycle = 1000.0 / rate
    hold_hi = max(0.0, cycle - step)
    terms = dict(
        compute=dict(t["compute"], term=a["terms"]["compute"]),
        host_jitter=dict(t["host_jitter"], term=a["terms"]["host_jitter"]),
        host_path=dict(t["host_path"], term=a["terms"]["host_path"]),
        bus_transmit=dict(lo_ms=_r(bus_lo), hi_ms=_r(bus_hi), kind=bus_kind, basis=bus_basis, source=[],
                          term="computed"),
        drive=dict(t["drive"], term=a["terms"]["drive"]),
        rate_hold=dict(lo_ms=0.0, hi_ms=_r(hold_hi), kind="d", source=[], term="computed",
                       basis=("none: the targets update at the model's 500 Hz" if hold_hi == 0 else
                              f"computed: a target held for {_r(cycle, 3)} ms instead of the model's {step:g} ms is "
                              f"up to {_r(hold_hi, 3)} ms older at the end of its hold")),
    )
    order = ("compute", "host_jitter", "host_path", "bus_transmit", "drive", "rate_hold")
    cmd_lo = sum(terms[k]["lo_ms"] for k in order)
    cmd_hi = sum(terms[k]["hi_ms"] for k in order)
    # the age of the feedback at the tick that uses it: a reply is sampled when its command arrives and is used at
    # the first tick after it is back at the controller
    pre_lo = sum(terms[k]["lo_ms"] for k in ("compute", "host_jitter", "host_path"))
    pre_hi = sum(terms[k]["hi_ms"] for k in ("compute", "host_jitter", "host_path"))
    back_lo, back_hi = t["host_path"]["lo_ms"], t["host_path"]["hi_ms"]
    ages, exchange_end_hi = [], 0.0
    for pre, f, back in ((pre_lo, f_lo, back_lo), (pre_hi, f_hi, back_hi)):
        for i in ((1, n_bus) if a["bus"] == "can" else (1,)):
            if a["bus"] == "can":
                sampled = pre + i * f
                returned = pre + n_bus * f + i * f + back
            else:
                sampled = pre + f
                returned = sampled + back
            used = max(1, math.ceil(returned / cycle - 1e-9)) * cycle
            ages.append(used - sampled)
            exchange_end_hi = max(exchange_end_hi, returned)
    age_lo, age_hi = min(ages), max(ages)
    excess_lo, excess_hi = max(0.0, age_lo - model_age), max(0.0, age_hi - model_age)
    total_lo, total_hi = cmd_lo + excess_lo, cmd_hi + excess_hi
    k_lo, k_hi = max(1, k_of(total_lo, step)), max(1, k_of(total_hi, step))
    slow = None
    if a["terms"]["drive"] == "drive_can":
        slow_hi = total_hi + DRIVE_LOOP_SLOW_EXTRA_MS
        slow = dict(extra_ms=DRIVE_LOOP_SLOW_EXTRA_MS, total_hi_ms=_r(slow_hi, 3), k_hi=max(1, k_of(slow_hi, step)),
                    delay_ms_hi=max(1, k_of(slow_hi, step)) * step, kind="d",
                    basis="the author's sensitivity: the high end if the drives' outer loop runs at 500 Hz, the "
                          "slower of the two rates the vendor's driver manuals give (myactuator_mc_driver)")
    return dict(
        name=a["name"], host=a["host"], realtime_layer=a["realtime"], bus=a["bus"], buses=a["buses"],
        drives_on_fullest_bus=n_bus, rate_note=a["rate_note"],
        holds_500_hz=holds_500, load_at_500_hz=dict(plain=at_500["load_plain"], stuffed=at_500["load_stuffed"]),
        rate_held=rate_held,
        planned_rate_hz=rate, planned_cycle_ms=_r(cycle), bus_at_planned_rate=row,
        terms=terms,
        tick_to_torque_ms=dict(lo=_r(cmd_lo, 3), hi=_r(cmd_hi, 3), kind="d"),
        exchange_end_hi_ms=_r(exchange_end_hi, 3), exchange_fits_cycle=bool(exchange_end_hi <= cycle + 1e-9),
        sensor_age_ms=dict(lo=_r(age_lo, 3), hi=_r(age_hi, 3), kind="d",
                           basis="computed from the terms: a drive samples its reply when its command arrives; the "
                                 "reply is used at the first tick after it is back at the controller"),
        sensor_age_beyond_model_ms=dict(lo=_r(excess_lo, 3), hi=_r(excess_hi, 3), model_ms=model_age, kind="d"),
        total_ms=dict(lo=_r(total_lo, 3), hi=_r(total_hi, 3), kind="d"),
        if_drive_loop_500_hz=slow,
        maps_to=dict(k_lo=k_lo, k_hi=k_hi, delay_ms_lo=k_lo * step, delay_ms_hi=k_hi * step, kind="d",
                     rule="k = the total / 2 ms, rounded up to a whole step, at least 1",
                     target_rate_matches_model=bool(rate == INPUTS["model_target_rate_hz"]["value"])),
        failure=dict(linux_crash=a["linux_crash"], microcontroller_crash=a["mcu_crash"],
                     operator_link_lost="Controlled stop within 1.0 s, then stand: a software rule in whatever runs "
                                        "the walking loop, in every architecture (BEHAVIOR_SPEC.md section 3)."),
    )


# ------------------------------------------------------------------------------------------------
# 4. The neck and tail buses
# ------------------------------------------------------------------------------------------------
def aux_buses(actuators: dict) -> dict:
    baud, bits = INPUTS["aux_baud_bps"]["value"], INPUTS["aux_bits_per_byte"]["value"]
    rate = INPUTS["aux_rate_hz"]["value"]
    out = {}
    for key, s in AUX.items():
        act = actuators[s["part"]]
        byte_us = bits / baud * 1e6
        wire = sum(s["write_bytes"] + s["read_bytes"]) * byte_us
        lo = wire + 2 * s["return_delay_us"][0]
        hi = wire + 2 * s["return_delay_us"][1]
        factory = INPUTS[s["factory_baud"]]["value"]
        at_factory = sum(s["write_bytes"] + s["read_bytes"]) * bits / factory * 1e6
        out[key] = dict(part=s["part"], joint=s["joint"], protocol=s["protocol"], catalog_interface=act["interface"],
                        packets=s["packets"], baud_bps=baud, wire_us=_r(wire, 1),
                        exchange_ms=dict(lo=_r(lo / 1000.0), hi=_r(hi / 1000.0)), rate_hz=rate,
                        factory_baud_bps=factory,
                        exchange_at_factory_baud_ms=dict(lo=_r((at_factory + 2 * s["return_delay_us"][0]) / 1000.0, 2),
                                                         hi=_r((at_factory + 2 * s["return_delay_us"][1]) / 1000.0, 2)),
                        fits_period_at_factory_baud=bool(at_factory + 2 * s["return_delay_us"][1] < 1e6 / rate),
                        own_bus_occupancy=dict(lo=_r(lo * 1e-6 * rate), hi=_r(hi * 1e-6 * rate)),
                        usb_adapter=s["usb_adapter"], kind="d",
                        basis="packet sizes from the protocols' layouts as the author knows them, at the assumed "
                              "baud rate and with the reply delay at 0 and at 500 µs; not measured")
    return out


# ------------------------------------------------------------------------------------------------
# 5-7. The model, the ledger and the bench tables
# ------------------------------------------------------------------------------------------------
def model_block() -> list:
    return [dict(id=m["id"], says=m["says"],
                 refs=[dict(file=ref[0], line=find_line(*ref), text=ref[1].strip()) for ref in m["refs"]])
            for m in MODEL]


def ledger_block() -> dict:
    rows = {}
    for row, rel in LEDGER.items():
        d = _load(rel)
        assert d["body"] == BODY and d["seeds"] == [3042, 3043, 3044]
        per_k = {}
        for ev in d["evaluations"].values():
            k = ev["variant"]["value"] or 0
            eps = ev["episodes"]
            per_k[k] = dict(delay_ms=2 * k, episodes=len(eps), up=sum(not e["fell"] for e in eps),
                            meets_every_limit=sum(bool(e["succeeded"]) for e in eps),
                            fall_times_s=[_r(e["fall_time_s"], 2) for e in eps if e["fell"]],
                            failed_terms=sorted({x for e in eps for x in e["failed_terms"]}))
        rows[row] = dict(record=rel, sha256=sha256(STUDY / rel), by_k={str(k): per_k[k] for k in sorted(per_k)})
    kp = _load(KP_LEDGER)["summary"]["searches"]["walk_nominal"]
    return dict(kind="b", body=BODY, condition="walk/nominal, seeds 3042-3044 (3 episodes per delay: a margin "
                                               "reading, not a 20-episode verdict)",
                rows=rows, never_run_k=[1, 6, 7, 8, 9],
                stiffness=dict(record=KP_LEDGER, sha256=sha256(STUDY / KP_LEDGER), kind="b",
                               meets_every_limit_down_to=kp["success"]["threshold"],
                               stays_up_down_to=kp["survival"]["threshold"],
                               note="every drive's kp x f, kv unchanged, the controller keeping the nominal gains "
                                    "(ledger.KP_RULE)"))


def gains_block(plant: dict) -> list:
    kp_cap, kd_cap = INPUTS["motion_mode_kp_max"]["value"], INPUTS["motion_mode_kd_max"]["value"]
    groups = {}
    for dr in plant["drives"]:
        g = plant["gains"]["drives"][dr["actuator"]]
        key = (dr["part"], dr["group"])
        groups.setdefault(key, dict(part=dr["part"], group=dr["group"], count=0, kp=_r(g["kp"], 2), kv=_r(g["kv"], 2),
                                    sustained_nm=_r(dr["sustained"], 4)))
        groups[key]["count"] += 1
    ankle = plant["gains"]["ankle_joint_space"]["left"]["stiffness_Nm_per_rad"]
    out = []
    for (part, group), g in groups.items():
        leg = part.startswith("myactuator")
        g["deflection_at_sustained_rad"] = _r(g["sustained_nm"] / g["kp"], 4)
        g["share_of_model_kp_at_sibling_manual_max"] = _r(kp_cap / g["kp"], 3) if leg else None
        g["share_of_model_kv_at_sibling_manual_max"] = _r(kd_cap / g["kv"], 3) if leg else None
        g["mass_kg_on_lever"] = _r(g["sustained_nm"] / (G * LEVER_M), 2)
        g["tip_mm_per_0p01_rad"] = _r(LEVER_M * 0.01 * 1000.0, 1)
        g["note"] = (f"each ankle's two motors carry half each: {_r(ankle['pitch'], 1)} N m/rad on pitch and on roll"
                     if group.startswith("ankle") else "the thigh twist" if group == "hip_yaw" else "")
        out.append(g)
    return out


def bench_parts_block(actuators: dict, parts: dict) -> dict:
    lot = _load(CANDIDATE)["cost"]["lines"]["bench"]
    assert _load(CANDIDATE)["id"] == BODY and len(lot) == 3
    lot_text = "$" + " / ".join(f"{v:,.0f}" for v in lot)
    rows = []
    for cat, pid, what, q1, q2, note in BENCH_PARTS:
        e = (actuators if cat == "actuators" else parts)[pid]
        rows.append(dict(what=what, catalog_id=pid, catalog=ACTUATORS if cat == "actuators" else PARTS,
                         model=e["model"], price_usd=e["price_usd"], run1_qty=q1, run2_qty=q2, kind="c", note=note))
    for what, q1, q2, price, kind, src, basis in BENCH_EXTRA:
        rows.append(dict(what=what, catalog_id=None, catalog=None, model=None, price_usd=price, run1_qty=q1,
                         run2_qty=q2, kind=kind, source=src, note=basis.format(bench_lot=lot_text)))
    tot = lambda q: _r(sum((r["price_usd"] or 0.0) * r[q] for r in rows), 2)  # noqa: E731
    cat_tot = lambda q: _r(sum(r["price_usd"] * r[q] for r in rows if r["catalog_id"]), 2)  # noqa: E731
    leg = sum(r["price_usd"] * r["run2_qty"] for r in rows if (r["catalog_id"] or "").startswith("myactuator"))
    adapters = sum(r["price_usd"] * r["run2_qty"] for r in rows if r["catalog_id"] == "pcan_usb_fd")
    return dict(rows=rows, run1_total_usd=tot("run1_qty"), run2_total_usd=tot("run2_qty"),
                run1_catalog_usd=cat_tot("run1_qty"), run2_catalog_usd=cat_tot("run2_qty"),
                leg_drives_usd=_r(leg, 2), usb_can_adapters_usd=_r(adapters, 2),
                run1_share_of_run2=_r(tot("run1_qty") / tot("run2_qty"), 3),
                bench_lot_usd=dict(low=lot[0], target=lot[1], high=lot[2], source=CANDIDATE, kind="c"),
                unpriced=[r["what"] for r in rows if r["price_usd"] is None],
                note="run 1: one drive of each type with the other 10 leg drives' frames replayed on the bus; run 2: "
                     "all 12 leg drives, which the build needs whatever the compute")


def step_gains_block(plant: dict, actuators: dict) -> list:
    """The gains of the delay steps on a bare shaft (d): the model's kp, and a kv that is critical on the drive's own
    rotor inertia (the catalog's armature estimate), since the model's kv is sized for the leg's inertia.  For a
    critically damped step x0 from rest, with w = sqrt(kp / I): the travel is x0 (1 - (1 + w t) e^(-w t)), the
    torque demand kp x0 (1 - w t) e^(-w t) (it crosses zero at t = 1 / w) and the peak speed x0 w / e."""
    frame_ms = INPUTS["video_frame_ms"]["value"][0]
    out = []
    for part in ("myactuator_rmd_x6_p20_60_e", "myactuator_rmd_x4_p36_36_e_l"):
        drives = [d for d in plant["drives"] if d["part"] == part and not d["group"].startswith("ankle")]
        gains = [plant["gains"]["drives"][d["actuator"]] for d in drives]
        kps = sorted({_r(g["kp"], 2) for g in gains})
        assert len(kps) == 1, "one kp per drive type outside the ankles"
        kp = kps[0]
        inertia = actuators[part]["armature_kg_m2"]
        w = math.sqrt(kp / inertia)
        kv = 2.0 * math.sqrt(kp * inertia)
        wt = w * frame_ms / 1000.0
        share = 1.0 - (1.0 + wt) * math.exp(-wt)
        model_kv = sorted(g["kv"] for g in gains)
        leg_inertia = sorted(g["inertia_kg_m2"] for g in gains)
        res = actuators[part]["phase_resistance_ohm"]
        kt = actuators[part]["kt_nm_per_a"]
        amps = math.sqrt(SUPPLY_LIMIT_A * SUPPLY_V / (3.0 * res))
        out.append(dict(
            part=part, kind="d", kp=kp, sustained_nm=_r(drives[0]["sustained"], 3),
            rotor_inertia_kg_m2=_r(inertia, 5), bench_kv=_r(kv, 2),
            model_kv=dict(lo=_r(model_kv[0], 2), hi=_r(model_kv[-1], 2)),
            model_kv_over_bench_kv=dict(lo=_r(model_kv[0] / kv, 1), hi=_r(model_kv[-1] / kv, 1)),
            leg_inertia_kg_m2=dict(lo=_r(leg_inertia[0], 3), hi=_r(leg_inertia[-1], 3)),
            time_constant_ms=_r(1000.0 / w, 2), torque_demand_nm=_r(kp * STEP_RAD, 2),
            peak_speed_rad_s=_r(STEP_RAD * w / math.e, 2),
            creep_time_constant_at_model_kv_ms=dict(lo=_r(1000.0 * model_kv[0] / kp, 0),
                                                    hi=_r(1000.0 * model_kv[-1] / kp, 0)),
            share_of_step_in_one_frame=_r(share, 2), frame_ms=frame_ms,
            tip_mm_in_one_frame=_r(share * STEP_RAD * POINTER_M * 1000.0, 2),
            stall_torque_at_supply_limit_nm=_r(kt * amps, 1), stall_phase_current_a_rms=_r(amps, 1),
            backdrive_nm={"myactuator_rmd_x6_p20_60_e": 1.6, "myactuator_rmd_x4_p36_36_e_l": 0.2}[part],
            basis="the author's estimate: an ideal PD loop on the catalog's rotor-inertia estimate, the pointer's "
                  "own inertia left out; the stall figure is the supply limit's power, "
                  f"{SUPPLY_LIMIT_A:g} A x {SUPPLY_V:g} V, as heat in three phase resistances, inverter losses "
                  "ignored, times the catalog's torque constant per ampere rms"))
        assert f"backdrive {out[-1]['backdrive_nm']:g} N" in actuators[part]["notes"], "the catalog's backdrive torque"
    return out


def grip_block() -> dict:
    """Follow-on sheet D: what the prescribed walks need of the floor (a), from the look screen's tail-phase check of
    today's walk (V0), and the pads' pressure (c)."""
    d = _load(LOOK_SCREEN)
    rows = []
    for speed in ("nominal", "v040", "fr05"):
        ss = d["tail_phase"][BODY][speed]["walks"]["V0"]["single_support"]
        rows.append(dict(speed=speed, walk=d["speeds"][speed], peak_twist_nm=_r(ss["mz_abs_max_nm"], 2),
                         friction_needed=_r(ss["mu_req_max"], 2), kind="a"))
    return dict(source=LOOK_SCREEN, sha256=sha256(STUDY / LOOK_SCREEN), walk="V0, today's walk and tail", rows=rows,
                pad_pressure_kpa=41, pad_pressure_kind="c", pad_pressure_source="materials_md",
                sample=dict(side_mm=40, mass_kg=_r(41e3 * 0.040 ** 2 / G, 2), kind="d",
                            basis="the author's example: a 40 mm x 40 mm sample under the mass that gives 41 kPa"))


def friction_block(plant: dict) -> list:
    """Follow-on sheet B's constant speeds against each drive's speeds on the build's supply (the plant's envelope)."""
    env = {e["part"]: e for e in plant["envelope"]["drives"].values()}
    out = []
    for part, speeds in FRICTION_SPEEDS.items():
        e = env[part]
        assert max(speeds) <= 0.9 * e["rated_speed_rad_s"], "inside the usable speed, with room"
        out.append(dict(part=part, speeds_rad_s=list(speeds), usable_speed_rad_s=_r(e["rated_speed_rad_s"], 2),
                        no_load_speed_rad_s=_r(e["no_load_speed_rad_s"], 2), supply_v=e["bus_V"] or 12.0, kind="c"))
    return out


def thermal_block(plant: dict) -> dict:
    tail = [d for d in plant["drives"] if d["group"] == "tail"][0]
    rated = INPUTS["sts3215_vendor_rated_nm"]["value"]
    stall, frac = INPUTS["sts3215_stall_nm"]["value"], INPUTS["box_servo_sustained_of_stall"]["value"]
    assert abs(frac * stall - tail["sustained"]) < 1e-3, "the sustained planning torque is 0.167 x the stall torque"
    rows = []
    for what, tau, kind, src in THERMAL_HOLDS:
        tau = tail["sustained"] if tau is None else tau
        rows.append(dict(what=what, torque_nm=_r(tau, 4), ratio_to_a1=_r(tau / tail["sustained"], 3),
                         ratio_to_vendor_rated=_r(tau / rated, 2), kind=kind, source=src,
                         mass_kg_on_lever=_r(tau / (G * THERMAL_LEVER_M), 3)))
    return dict(part=tail["part"], lever_m=THERMAL_LEVER_M, minutes=THERMAL_HOLD_MIN, holds=2 * len(rows), rows=rows,
                vendor_rated_nm=rated, stall_nm=stall, sustained_of_stall=frac)


def results_block() -> dict:
    rows = [dict(drive=d, path=p, buses=b, load=load, target_rate_hz=r, run=run)
            for d, p, b, load, r, run in RESULT_ROWS]
    soak = [dict(path=p, buses=b, load=load) for p, b, load in SOAK_ROWS]
    return dict(commands=COMMANDS, rows=rows, soak_minutes=SOAK_MIN, soak=soak,
                soak_cycles=int(SOAK_MIN * 60 * INPUTS["model_target_rate_hz"]["value"]),
                leg_rows=sum(r["drive"] in _LEG for r in rows))


def time_block(results: dict, thermal: dict) -> dict:
    fill = dict(n_rows=len(results["rows"]), n_leg_rows=results["leg_rows"], n_soak=len(results["soak"]),
                soak_min=results["soak_minutes"], commands=results["commands"], n_holds=thermal["holds"],
                hold_min=thermal["minutes"])
    rows = [dict(group=g, what=w.format(**fill), hours_lo=lo, hours_hi=hi, kind="d") for g, w, lo, hi in BENCH_TIME]
    tot = {g: dict(lo=_r(sum(r["hours_lo"] for r in rows if r["group"] == g), 1),
                   hi=_r(sum(r["hours_hi"] for r in rows if r["group"] == g), 1)) for g in ("run1", "run2", "follow")}
    # the unattended parts must fit inside their rows: the soak logs and the thermal holds
    soak_h = len(results["soak"]) * results["soak_minutes"] / 60.0
    holds_h = thermal["holds"] * thermal["minutes"] / 60.0
    assert [r for r in rows if r["what"].startswith("Soak")][0]["hours_lo"] >= soak_h
    assert [r for r in rows if r["what"].startswith("E.")][0]["hours_lo"] >= holds_h
    return dict(rows=rows, totals=tot, soak_hours=_r(soak_h, 2), thermal_hold_hours=_r(holds_h, 2),
                basis="the author's estimates for one person at a bench with the test programs already written; "
                      "none is measured")


def delay_map_block(ledger: dict) -> list:
    out = []
    for label, k, rerun, reading in DELAY_MAP:
        cells = {}
        for row, r in ledger["rows"].items():
            e = r["by_k"].get(str(k)) if k is not None else None
            cells[row] = None if e is None else dict(up=e["up"], meets_every_limit=e["meets_every_limit"],
                                                     episodes=e["episodes"], fall_times_s=e["fall_times_s"])
        out.append(dict(measured=label, k=k, simulated_delay_ms=None if k is None else 2 * k, rerun=rerun,
                        model=cells, reading=reading))
    return out


# ------------------------------------------------------------------------------------------------
# The record
# ------------------------------------------------------------------------------------------------
def build() -> dict:
    plant = _load(PLANT_JSON)
    actuators = {a["id"]: a for a in _load(ACTUATORS)["actuators"]}
    parts = {p["id"]: p for p in _load(PARTS)["parts"]}
    # the inputs read from records are checked against them
    assert plant["body"] == BODY and plant["timestep_s"] * 1000.0 == INPUTS["model_step_ms"]["value"]
    legs = [d for d in plant["drives"] if d["part"].startswith("myactuator")]
    assert len(legs) == INPUTS["leg_drive_count"]["value"] and len(plant["drives"]) == 14
    leg_parts = sorted({d["part"] for d in legs})
    for pid in leg_parts:
        assert "EtherCAT" in actuators[pid]["interface"] and "CAN" in actuators[pid]["interface"]
    fr = frames()
    ledger = ledger_block()
    results, thermal = results_block(), thermal_block(plant)
    doc = dict(
        generated_by="stage2/jetson75/latency_bus.py",
        script_sha256=sha256(__file__),
        date=DATE,
        status="planning arithmetic and estimates; nothing here is simulated anew, built or measured",
        kinds=KINDS,
        web_read=WEB_READ,
        body=BODY,
        drives=dict(source=PLANT_JSON, sha256=sha256(STUDY / PLANT_JSON),
                    by_part={pid: dict(count=sum(d["part"] == pid for d in plant["drives"]),
                                       joints=sorted({d["group"] for d in plant["drives"] if d["part"] == pid}),
                                       interface=actuators[pid]["interface"], price_usd=actuators[pid]["price_usd"])
                             for pid in sorted({d["part"] for d in plant["drives"]})}),
        inputs=INPUTS,
        terms=TERMS,
        frames=fr,
        bus_load=bus_load(fr),
        architectures={k: architecture(k, fr) for k in ARCH_ORDER},
        aux=aux_buses(actuators),
        model=model_block(),
        ledger=ledger,
        bench=dict(parts=bench_parts_block(actuators, parts), gains=gains_block(plant),
                   step_gains=step_gains_block(plant, actuators), results=results,
                   delay_map=delay_map_block(ledger), friction=friction_block(plant), grip=grip_block(),
                   thermal=thermal, time=time_block(results, thermal)),
        sources=SOURCES,
    )
    return json.loads(json.dumps(doc))  # plain JSON types, as a reader of the file gets them


def dumps(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


# ------------------------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------------------------
def _rng(lo, hi, nd=2, unit=""):
    f = lambda x: f"{x:.{nd}f}"  # noqa: E731
    return f"{f(lo)}{unit}" if f(lo) == f(hi) else f"{f(lo)}–{f(hi)}{unit}"


def _pct(lo, hi):
    f = lambda x: f"{100 * x:.0f}"  # noqa: E731
    return f"{f(lo)} %" if f(lo) == f(hi) else f"{f(lo)}–{f(hi)} %"


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def t_frames(doc):
    f = doc["frames"]
    c, fd, e = f["classic_can"], f["can_fd_hypothetical"], f["ethercat"]
    return _table(
        ["Frame", "Arithmetic", "Time, no stuffing to worst stuffing", "Kind"],
        [["Classic CAN, 11-bit identifier, 8 data bytes (the leg drives' protocol)", c["arithmetic"],
          f"{c['us_plain']:.0f}–{c['us_stuffed']:.0f} µs", "(c)"],
         ["CAN FD, 8 data bytes, 1 Mbit/s and 5 Mbit/s (hypothetical: no CAN FD mode of the leg drives was found)",
          fd["arithmetic"], f"{fd['us_plain']:.1f}–{fd['us_stuffed']:.1f} µs", "(d)"],
         ["EtherCAT, one frame for all 12 drives (assumed 24 bytes of process data per drive)", e["arithmetic"],
          f"{e['us']:.1f} µs", "(d)"]])


def t_bus_load(doc):
    rows = []
    for r in doc["bus_load"]:
        arith = (f"{r['frames_per_cycle']} frames: {r['ms_per_cycle_plain']:.3f}–{r['ms_per_cycle_stuffed']:.3f} ms")
        rows.append([r["bus"], r["buses"], r["drives_on_bus"], f"{r['rate_hz']:g} Hz ({r['cycle_ms']:g} ms)", arith,
                     _pct(r["load_plain"], r["load_stuffed"]), "yes" if r["fits"] else "**no**",
                     f"{r['max_rate_hz_stuffed']:.0f}–{r['max_rate_hz_plain']:.0f} Hz", f"({r['kind']})"])
    return _table(["Bus", "Buses", "Drives on each", "Loop rate (cycle)", "Frames per cycle × frame time",
                   "Load = that ÷ cycle", "Fits", "Fastest loop the bus carries (worst–no stuffing)", "Kind"], rows)


def t_architectures(doc):
    rows = []
    for k, a in doc["architectures"].items():
        b = a["bus_at_planned_rate"]
        rate = a["rate_held"]["text"]
        m = a["maps_to"]
        sim = (f"{m['delay_ms_lo']:g}–{m['delay_ms_hi']:g} ms (k {m['k_lo']}–{m['k_hi']})"
               if m["k_lo"] != m["k_hi"] else f"{m['delay_ms_lo']:g} ms (k {m['k_lo']})")
        if not m["target_rate_matches_model"]:
            sim += ", at a target rate the model has not been run at"
        linux = a["failure"]["linux_crash"]["short"]
        mcu = a["failure"]["microcontroller_crash"]["short"]
        rows.append([f"**{k}**: {a['name']}", rate, _pct(b["load_plain"], b["load_stuffed"]),
                     _rng(a["tick_to_torque_ms"]["lo"], a["tick_to_torque_ms"]["hi"], 1, " ms"),
                     _rng(a["sensor_age_ms"]["lo"], a["sensor_age_ms"]["hi"], 1, " ms"),
                     _rng(a["total_ms"]["lo"], a["total_ms"]["hi"], 1, " ms"), sim, linux, mcu])
    return _table(["Architecture", "Loop rate with 12 leg drives (kind in each cell)",
                   "Bus load at that rate (c; E: d)", "Control tick to torque change (d)",
                   "Age of the feedback at the tick (d; the model's is 2 ms)",
                   "End-to-end total: tick to torque, plus any feedback age beyond 2 ms (d)",
                   "Simulated delay it maps onto (d)", "On a Linux crash", "On a microcontroller crash"], rows)


TERM_LABELS = dict(compute="compute step", host_jitter="host jitter", host_path="host's path to the bus",
                   bus_transmit="bus transmit", drive="drive's own loop", rate_hold="hold of a slower target rate")


def t_terms(doc):
    rows = []
    for k, a in doc["architectures"].items():
        cells = [f"**{k}**"]
        for name in TERM_LABELS:
            t = a["terms"][name]
            cells.append(f"{_rng(t['lo_ms'], t['hi_ms'])} ({t['kind']})")
        cells.append(f"**{_rng(a['tick_to_torque_ms']['lo'], a['tick_to_torque_ms']['hi'])}**")
        cells.append(f"{_rng(a['sensor_age_ms']['lo'], a['sensor_age_ms']['hi'])}")
        cells.append(f"{_rng(a['sensor_age_beyond_model_ms']['lo'], a['sensor_age_beyond_model_ms']['hi'])}")
        cells.append(f"**{_rng(a['total_ms']['lo'], a['total_ms']['hi'])}**")
        cells.append("yes" if a["exchange_fits_cycle"] else f"no: up to {a['exchange_end_hi_ms']:.2f} ms")
        rows.append(cells)
    head = ["Arch."] + [f"{v}, ms" for v in TERM_LABELS.values()]
    head += ["Tick to torque, ms (d)", "Feedback age at the tick, ms (d)", "Age beyond the model's 2 ms, ms (d)",
             "Total, ms (d)", "Whole exchange ends inside the cycle at worst (d)"]
    return _table(head, rows)


def t_term_basis(doc):
    labels = dict(compute_mcu="compute step, microcontroller", compute_jetson="compute step, Jetson real-time core",
                  jitter_mcu="host jitter, microcontroller", jitter_jetson_rt="host jitter, Jetson real-time core",
                  path_mcu_can="path to the bus, microcontroller's CAN", path_usb_can="path to the bus, USB-CAN",
                  path_native_can="path to the bus, the Orin's CAN controller",
                  path_ethercat="path to the bus, EtherCAT", drive_can="drive's own loop, over CAN",
                  drive_ethercat="drive's own loop, over EtherCAT")
    return _table(["Term", "Estimate, ms", "Kind", "Whose estimate and how derived"],
                  [[labels[k], _rng(t["lo_ms"], t["hi_ms"]), f"({t['kind']})",
                    t["basis"] + (f" [{', '.join(t['source'])}]" if t["source"] else "")]
                   for k, t in doc["terms"].items()])


def t_aux(doc):
    rows = []
    for k, s in doc["aux"].items():
        rows.append([f"{s['joint']}: `{s['part']}`", s["catalog_interface"], s["protocol"], s["packets"],
                     f"{s['exchange_ms']['lo']:.2f}–{s['exchange_ms']['hi']:.2f} ms",
                     f"{_pct(s['own_bus_occupancy']['lo'], s['own_bus_occupancy']['hi'])} at {s['rate_hz']:g} Hz",
                     s["usb_adapter"]])
    return _table(["Servo", "Catalog interface (c)", "Protocol", "One write and one read (d)",
                   "Time at 1 Mbit/s, reply delay 0 to 500 µs (d)", "Share of its own bus (d)",
                   "USB adapter on a Jetson-direct path"], rows)


def t_failure(doc):
    rows = []
    for k, a in doc["architectures"].items():
        rows.append([f"**{k}**", a["realtime_layer"], a["failure"]["linux_crash"]["detail"],
                     a["failure"]["microcontroller_crash"]["detail"]])
    return _table(["Arch.", "Walking loop runs on", "Linux crash (kernel panic, hang, or the real-time thread dies)",
                   "Microcontroller crash"], rows)


def t_model(doc):
    rows = []
    for m in doc["model"]:
        refs = "; ".join(f"`{r['file']}:{r['line']}`" for r in m["refs"])
        rows.append([m["says"], refs])
    return _table(["What the model does", "Where (file:line)"], rows)


def t_ledger(doc):
    led = doc["ledger"]
    ks = sorted({int(k) for r in led["rows"].values() for k in r["by_k"]})
    rows = []
    for k in ks:
        cells = [f"{2 * k} ms (k {k})"]
        for row in led["rows"]:
            e = led["rows"][row]["by_k"][str(k)]
            ft = e["fall_times_s"]
            fall = f"; falls at {min(ft):.1f}–{max(ft):.1f} s" if ft else ""
            miss = f"; misses {', '.join(e['failed_terms'])}" if e["failed_terms"] and not ft else ""
            cells.append(f"{e['up']} of {e['episodes']} · {e['meets_every_limit']} of {e['episodes']}{fall}{miss}")
        rows.append(cells)
    return _table(["Simulated delay"] + [f"{r}: up · meeting every limit (b)" for r in led["rows"]], rows)


def t_bench_parts(doc):
    rows = []
    for r in doc["bench"]["parts"]["rows"]:
        ident = f"`{r['catalog_id']}`" if r["catalog_id"] else "not in the catalog"
        price = "not priced" if r["price_usd"] is None else f"${r['price_usd']:,.2f}"
        rows.append([r["what"], ident, price, r["run1_qty"], r["run2_qty"], f"({r['kind']})", r["note"]])
    p = doc["bench"]["parts"]
    rows.append(["**Total, catalog parts only**", "", "", f"**${p['run1_catalog_usd']:,.2f}**",
                 f"**${p['run2_catalog_usd']:,.2f}**", "(c)",
                 f"of which the 12 leg drives ${p['leg_drives_usd']:,.2f} and the two USB-CAN adapters "
                 f"${p['usb_can_adapters_usd']:,.2f}"])
    rows.append(["**Total with the Teensys and the allowances**", "", "", f"**${p['run1_total_usd']:,.2f}**",
                 f"**${p['run2_total_usd']:,.2f}**", "(d)", "the bench supply is not priced; the dev kit is on hand"])
    return _table(["Part", "Catalog id (`designlab/data`)", "Price each", "Run 1 qty", "Run 2 qty", "Kind", "Note"],
                  rows)


def t_bench_gains(doc):
    rows = []
    for g in doc["bench"]["gains"]:
        x, y = g["share_of_model_kp_at_sibling_manual_max"], g["share_of_model_kv_at_sibling_manual_max"]
        joint = g["group"].replace("_", " ") + (" motor" if g["group"].startswith("ankle") else "")
        rows.append([f"`{g['part']}`", joint, g["count"], f"{g['kp']:.2f}", f"{g['kv']:.2f}",
                     f"{g['sustained_nm']:.2f}", f"{g['mass_kg_on_lever']:.2f}",
                     f"{g['deflection_at_sustained_rad']:.3f}",
                     "n/a" if x is None else f"{x:.2f}", "n/a" if y is None else f"{y:.2f}", g["note"]])
    return _table(["Drive (catalog id)", "Joint", "Count", "Model kp, N·m/rad (c)", "Model kv, N·m·s/rad (c)",
                   "Sustained planning torque, N·m (c)", f"Mass on a {LEVER_M:.2f} m lever for that torque, kg",
                   "Deflection the model expects there, rad", "500 ÷ model kp (d)", "5 ÷ model kv (d)", "Note"],
                  rows)


def t_step_gains(doc):
    rows = []
    for g in doc["bench"]["step_gains"]:
        mk, ratio, creep = g["model_kv"], g["model_kv_over_bench_kv"], g["creep_time_constant_at_model_kv_ms"]
        rows.append([f"`{g['part']}`", f"{g['kp']:.1f}", f"{g['rotor_inertia_kg_m2']:.5f}", f"**{g['bench_kv']:.1f}**",
                     f"{_rng(mk['lo'], mk['hi'], 1)} ({_rng(ratio['lo'], ratio['hi'], 1)} × the bench kv)",
                     f"{_rng(creep['lo'], creep['hi'], 0)}",
                     f"{g['torque_demand_nm']:.1f}", f"{g['time_constant_ms']:.1f}", f"{g['peak_speed_rad_s']:.1f}",
                     f"{g['tip_mm_in_one_frame']:.1f}"])
    f = doc["bench"]["step_gains"][0]
    return _table(["Leg drive", "kp for the delay steps, N·m/rad (c: the model's)",
                   "Rotor inertia at the output, kg·m² (d: the catalog's estimate)",
                   "Bench kv = 2 √(kp × that inertia), N·m·s/rad (d)",
                   "The model's kv for this drive type, N·m·s/rad (c): sized for the leg",
                   "Creep time constant kv ÷ kp if the model's kv were set on the bare shaft, ms (d)",
                   f"Torque the {STEP_RAD:g} rad step asks for at first, N·m (c: kp × the step)",
                   "Time constant with the bench kv, ms (d): the torque demand lasts about this long",
                   "Peak speed of the step, rad/s (d)",
                   f"Travel of a {POINTER_M:.2f} m pointer's tip in the first {f['frame_ms']:.1f} ms, mm (d)"], rows)


def t_results(doc):
    res = doc["bench"]["results"]
    blank = [""] * 14
    rows = [[r["drive"], r["path"], r["buses"], r["load"], r["target_rate_hz"], r["run"]] + blank +
            (["n/a", "n/a"] if r["target_rate_hz"] != 500 else ["", ""]) for r in res["rows"]]
    return _table(["Drive", "Path", "Buses", "Load", "Target rate, Hz", "Run", "Achieved cycle: p50 / max", "N",
                   "H p50", "H p99", "H max", "D p50", "D p99", "D method and level", "T p50", "T p99", "T max",
                   "T by M2: p50 / p99 / max", "S p99", "Reading", "k", "k +"], rows)


def t_soak(doc):
    res = doc["bench"]["results"]
    rows = [[r["path"], r["buses"], r["load"]] + [""] * 8 for r in res["soak"]]
    return _table(["Path", "Buses", "Load", "Run", "H p99, ms", "H p99.9, ms", "H max, ms",
                   "Reply turnaround: p50 / max, ms", "Late cycles (tick to tick above 2.5 ms)",
                   "Cycles whose replies missed the next tick", "Frames lost or resent"], rows)


def t_friction(doc):
    names = {"myactuator_rmd_x6_p20_60_e": "X6-P20-60-E", "myactuator_rmd_x4_p36_36_e_l": "X4-P36-36-E-L",
             "robotis_xm540_w270_t": "XM540-W270", "feetech_sts3215_c018": "STS3215"}
    rows = []
    for r in doc["bench"]["friction"]:
        rows.append([names[r["part"]], ", ".join(f"{v:g}" for v in r["speeds_rad_s"]),
                     f"{r['usable_speed_rad_s']:.2f}", f"{r['no_load_speed_rad_s']:.2f} at {r['supply_v']:g} V"])
    return _table(["Drive", "Constant speeds to run, rad/s", "The model's usable speed, rad/s (c: `plant.json`)",
                   "No-load speed on the build's supply, rad/s (c: `plant.json`)"], rows)


def t_grip(doc):
    g = doc["bench"]["grip"]
    rows = [[r["walk"], f"{r['peak_twist_nm']:.1f}", f"{r['friction_needed']:.2f}", "(a)"] for r in g["rows"]]
    return _table(["Prescribed walk (today's walk and tail)", "Peak twisting moment under the stance pad, N·m",
                   "Friction it needs with that moment", "Kind"], rows)


def t_delay_map(doc):
    rows = []
    for r in doc["bench"]["delay_map"]:
        cells = []
        for row, e in r["model"].items():
            cells.append("not run" if e is None else f"{e['up']} of {e['episodes']} · {e['meets_every_limit']} of "
                                                     f"{e['episodes']}")
        sim = f"{r['simulated_delay_ms']} ms (k {r['k']})" if r["k"] is not None else r["rerun"]
        rows.append([r["measured"], sim] + cells + [r["reading"]])
    heads = [f"Model, {row}: up · every limit (b)" for row in doc["ledger"]["rows"]]
    return _table(["Your reading (tick to torque at its p99, plus any feedback age beyond 2 ms)",
                   "Re-run the bar at"] + heads + ["What it would mean, from what the study has shown"], rows)


def t_thermal(doc):
    th = doc["bench"]["thermal"]
    rows = [[r["what"], f"{r['torque_nm']:.2f}", f"{r['ratio_to_a1']:.2f}", f"{r['ratio_to_vendor_rated']:.2f}",
             f"{r['mass_kg_on_lever']:.2f}", f"({r['kind']})"] for r in th["rows"]]
    return _table(["Hold", "Torque, N·m", "× the planning torque",
                   f"× the vendor's rated {th['vendor_rated_nm']:.2f} N·m",
                   f"Mass on a {th['lever_m']:.2f} m lever, kg", "Kind"], rows)


def t_time(doc):
    tm = doc["bench"]["time"]
    names = dict(run1="Run 1", run2="Run 2", follow="Follow-on")
    rows = [[names[r["group"]], r["what"], _rng(r["hours_lo"], r["hours_hi"], 1, " h")] for r in tm["rows"]]
    for g, n in names.items():
        rows.append([f"**{n}, total**", "", f"**{_rng(tm['totals'][g]['lo'], tm['totals'][g]['hi'], 1, ' h')}**"])
    return _table(["Part", "What", "Bench time (d)"], rows)


def t_sources(doc):
    rows = []
    for k, s in doc["sources"].items():
        if s["type"] != "web":
            continue
        refs = " ; ".join([s["ref"]] + s.get("also", []))
        rows.append([f"`{k}`", refs, s["says"]])
    return _table(["Key", "URL", "What it says (as extracted)"], rows)


TABLES = dict(frames=t_frames, bus_load=t_bus_load, architectures=t_architectures, terms=t_terms,
              term_basis=t_term_basis, aux=t_aux, failure=t_failure, model=t_model, ledger=t_ledger,
              sources=t_sources, bench_parts=t_bench_parts, step_gains=t_step_gains, results=t_results,
              soak=t_soak, bench_gains=t_bench_gains, delay_map=t_delay_map, friction=t_friction, grip=t_grip,
              thermal=t_thermal, time=t_time)
BEGIN, END = "<!-- latency_bus:{} begin -->", "<!-- latency_bus:{} end -->"


def block(doc: dict, name: str) -> str:
    return "\n".join([BEGIN.format(name)] + TABLES[name](doc) + [END.format(name)])


def markdown(doc: dict, names=None) -> str:
    return "\n\n".join(block(doc, n) for n in (names or TABLES)) + "\n"


_BLOCK = re.compile(r"<!-- latency_bus:(\w+) begin -->\n.*?<!-- latency_bus:\1 end -->", re.S)


def embedded(text: str) -> dict:
    """The tables embedded in a document, by name."""
    return {m.group(1): m.group(0) for m in _BLOCK.finditer(text)}


def sync(doc: dict) -> list:
    """Rewrite the tables embedded in the two documents; returns the documents changed."""
    changed = []
    for path in DOCS:
        if not path.exists():
            continue
        old = path.read_text()
        new = _BLOCK.sub(lambda m: block(doc, m.group(1)), old)
        if new != old:
            path.write_text(new)
            changed.append(path.name)
    return changed


def check(doc: dict) -> list:
    problems = []
    if not OUT.exists() or OUT.read_text() != dumps(doc):
        problems.append(f"{OUT.name} is stale: run python stage2/jetson75/latency_bus.py")
    for path in DOCS:
        if not path.exists():
            problems.append(f"{path.name} is missing")
            continue
        for name, text in embedded(path.read_text()).items():
            if text != block(doc, name):
                problems.append(f"{path.name}: table {name} is stale: run python stage2/jetson75/latency_bus.py sync")
    return problems


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    doc = build()
    if not argv:
        OUT.write_text(dumps(doc))
        print(f"wrote {OUT.relative_to(STUDY)}")
        return 0
    if argv[0] == "md":
        sys.stdout.write(markdown(doc, argv[1:] or None))
        return 0
    if argv[0] == "sync":
        OUT.write_text(dumps(doc))
        print("rewrote the tables in: " + (", ".join(sync(doc)) or "nothing to change"))
        return 0
    if argv[0] == "check":
        problems = check(doc)
        print("\n".join(problems) or "current")
        return 1 if problems else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
