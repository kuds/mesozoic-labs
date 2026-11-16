# Latency budget and bus plan for the 75 cm build, per compute architecture

14 Nov 2026. **Planning arithmetic and estimates only.** Nothing here is simulated anew, built or measured. This is task
7 of `JETSON_PLEM_REVIEW.md` section 5, for the body `75-M-H-10`. It follows the owner's decision of 14 Nov (P30): the
Jetson is studied, the microcontroller keeps the walking loop as the baseline, the bar stays judged at 10 ms of servo
delay until a bench value exists, and the delay bench runs first (`BENCH_SHEET.md`).

Paths are relative to `hardware/trex_form_study/`. P, E, F and A rows are in `DECISION_LOG.md`.

**Kinds of evidence.** Every number is labelled:
- (a) a prescribed-walk or static model result of this study;
- (b) a closed-loop simulation result of this study;
- (c) a sourced fact: a catalog id in `designlab/data`, a document or record of this study, or a web page (section 10);
- (d) an estimate or assumption. Here "the author's estimate" means an estimate made for this document on 14 Nov 2026
  from the sources named beside it, not a vendor figure and not a measurement.

**How the web pages were read.** Through the web-search tool's extracts on 14 Nov 2026. The session's network does not
allow opening the pages themselves, so each web fact should be confirmed on its page before anything is bought or built
on it. Where an extract did not settle a fact, the fact is labelled (d).

**Sources.** Every number in the tables is a field of `stage2/jetson75/latency_bus.json`, written by
`stage2/jetson75/latency_bus.py`. The tables are machine-made: `python stage2/jetson75/latency_bus.py md` prints them
and `sync` rewrites them here. Tests: `tests/test_stage2_latency_bus.py`.

**Words used once and then abbreviated.**
- CAN: Controller Area Network, the two-wire bus the leg drives speak. A *frame* is one message on it.
- CAN FD: CAN with a flexible data rate, a faster variant.
- EtherCAT: an industrial Ethernet bus in which one frame passes through every drive.
- USB-CAN: a USB adapter that gives a computer a CAN port.
- PREEMPT_RT: the Linux kernel's real-time option. An *isolated core* is a processor core kept free for one task.
- TTL half-duplex: a one-wire serial bus, the kind the neck and tail servos use.
- PD servo: a position servo with a stiffness (kp) and a damping (kv) gain.
- IMU: inertial measurement unit. ToF: time-of-flight distance sensor. GPU: graphics processor.
- Tick: the instant the control loop takes its sensor snapshot and sets its targets, 500 times a second in the model.
- k: the simulation's delay, a whole number of 2 ms physics steps.

## In brief, for the owner

1. **One CAN bus cannot carry the 12 leg drives at the model's 500 Hz.**
   - Each drive needs a command and a reply every cycle (c). That is 24 frames of 111–135 µs each: 2.66–3.24 ms of bus
     time per cycle, against a 2 ms cycle. The load is 133–162 % (c, arithmetic on sourced facts).
   - One bus tops out at 309–375 Hz with 12 drives.
   - That rules out, at 500 Hz, both single-bus Jetson-direct layouts: one USB-CAN channel, which is PR #668's layout
     (unverified), and the Orin's own CAN controller, of which the Orin Nano has one (c).
   - They could run at 250 Hz. The simulation has never been run at that target rate.
   - No CAN FD mode of these drives was found (d), so CAN FD does not rescue a single bus.
2. **Two buses carry it at 67–81 % load, three at 44–54 %** (c).
   - `COMPUTE.md`'s design, a Teensy with two CAN buses, holds 500 Hz on this arithmetic.
   - It is tight: in the worst case the whole exchange of one cycle takes 2.2 ms, longer than the 2 ms cycle (d).
   - The Teensy has a third CAN port (c). Three buses end the exchange inside the cycle (d). Worth trying at the bench.
   - These rates are bus-time arithmetic only. How fast a drive answers, and the highest command rate it accepts,
     were not found (d); the bench shows both.
3. **Estimated end-to-end delay, all (d).** This is the time from the control tick to the change of torque at the
   joint, plus any age of the feedback beyond the model's 2 ms. Each total maps onto a simulated delay in whole 2 ms
   steps:
   - Teensy with two buses, under the Raspberry Pi 5 or the Jetson (A, B): 0.3–3.5 ms, which maps onto 2–4 ms.
   - Jetson-direct with two USB-CAN channels (C2): 0.5–5.5 ms, which maps onto 2–6 ms.
   - Jetson-direct with one USB-CAN channel (C1), at 250 Hz: 0.9–11.0 ms, which maps onto 2–12 ms.
   - Jetson-direct on the Orin's own CAN controller (D), at 250 Hz: 0.9–9.5 ms, which maps onto 2–10 ms.
   - EtherCAT from the Jetson (E): 0.3–3.3 ms, which maps onto 2–4 ms, if the drives accept a 2 ms cycle. Their
     supported cycle times were not found.
   - One known soft spot: the vendor's driver manuals give the position loop of an earlier drive generation as
     1 kHz in one edition and 500 Hz in another. The figures above take 1 kHz. At 500 Hz every CAN upper end rises by
     1 ms: the Teensy paths (A, B) would map onto 2–6 ms and C2 onto 2–8 ms.
4. **For scale, what the simulation has shown (b).** The 75 cm nominal walk (0.243 m/s) meets every limit to 6 ms,
   stays up to 8 ms and falls at 10 ms (`stage1/runs/2026-11-08/STAGE1_PROBE.md`, the correction note). These are
   3-episode margin readings, 2 ms was never run, and the 0.40 m/s bar has no record at any of these delays yet.
5. **The host computer does not change the Teensy paths' delay.** With the walking loop on the Teensy, the Raspberry
   Pi 5 or the Jetson sits outside the servo loop. The delay estimate is the same for both, so it does not favour
   either host.
6. **Failure behaviour separates the architectures more than delay does.**
   - On a Linux crash the Teensy paths end in a controlled stop and a stand, once the firmware exists (it is
     unwritten: `COMPUTE.md`).
   - Jetson-direct and EtherCAT from the Jetson end in torque-off or frozen drives. For a walking biped that is a fall.
   - A microcontroller crash fells a walking robot in the Teensy paths too. Every architecture has one such point.
7. **EtherCAT is really offered by the leg drives** (c): both models are "-E" variants. What is missing is the
   supported cycle time, a second cable chain through the legs, and a controlled stop on a Linux crash.
8. **The neck and tail servos do not load the leg buses.** Each has its own serial bus. At 100 Hz each would use
   4–16 % of its own bus at 1 Mbit/s (d). The neck servo ships at 57,600 bit/s (c), which does not fit 100 Hz, and
   its 5 V data line needs a level shifter on a Teensy (c).
9. **Which numbers are estimates.** Every delay term: the compute step, the host's jitter, the USB hop, the order of
   the frames on the bus, the drive's own loop, and so every total and every "maps onto". Only the bench test can
   replace them (section 8). The frame times and the bus loads alone are arithmetic on sourced facts. Until the bench
   reports, the bar stays judged at 10 ms (P29, P30).

## 1. What is compared

| | Host computer | Where the walking loop runs | Leg bus |
|---|---|---|---|
| **A** | Raspberry Pi 5 + AI HAT+ (P14) | Teensy 4.1 microcontroller | two CAN buses, one leg (6 drives) on each: `COMPUTE.md`'s design, the baseline |
| **A3** | as A | as A | three CAN buses, 4 drives on each: an option, since the Teensy has three ports (c) |
| **B** | Jetson Orin Nano Super dev kit | Teensy 4.1 | as A: the same real-time layer under another host |
| **C1** | Jetson | an isolated PREEMPT_RT core on the Jetson | one USB-CAN channel (catalog `pcan_usb_fd`): PR #668's layout (unverified) |
| **C2** | Jetson | as C1 | two USB-CAN channels |
| **D** | Jetson | as C1 | the Orin's own CAN controller with a transceiver: one bus |
| **E** | Jetson | as C1 | EtherCAT from the Jetson's Ethernet port |

The 75 cm body has 14 drives (c: `stage1/runs/2026-11-08/75-M-H-10/plant.json`):
- 12 leg drives: 2 × MyActuator RMD-X6-P20-60-E (hip pitch, `myactuator_rmd_x6_p20_60_e`) and 10 × RMD-X4-P36-36-E-L
  (hip roll, thigh twist, knee and the two ankle motors of each leg, `myactuator_rmd_x4_p36_36_e_l`);
- the neck: ROBOTIS XM540-W270 (`robotis_xm540_w270_t`);
- the tail: Feetech STS3215 (`feetech_sts3215_c018`).

**The servo form assumed for the hardware (d).** In the model each drive is a PD servo that receives an angle, a rate
and a feedforward torque (section 4). The matching hardware form is the drive's own PD loop, fed targets over the bus.
The vendor's protocol lists a "motion mode" command (c: `myactuator_protocol`). A sibling model's manual and a
community driver describe it as carrying a position, a speed, two gains and a feedforward torque (d for these drives:
`myactuator_sibling_manual`). The other form, a PD loop on the controller that sends torque commands, would put the
whole bus delay and the feedback's age inside the servo loop. The model does not represent that, so this document
assumes the drive's own loop. Whether its gain range reaches the model's gains is open (`BENCH_SHEET.md`, sheet A).

## 2. The leg bus: frames and load

The vendor's protocol documents list a CAN bus at 1 Mbit/s with standard frames of 8 data bytes: a command to drive
*n* on identifier 0x140 + *n*, and its reply on 0x240 + *n* (c: `myactuator_protocol`). So every drive costs two
frames per cycle, as the review assumed.

<!-- latency_bus:frames begin -->
| Frame | Arithmetic | Time, no stuffing to worst stuffing | Kind |
|---|---|---|---|
| Classic CAN, 11-bit identifier, 8 data bytes (the leg drives' protocol) | 8 x 8 + 47 = 111 bits; + floor((34 + 8 x 8 - 1) / 4) = 24 stuff bits = 135 bits; at 1 Mbit/s one bit is 1 µs | 111–135 µs | (c) |
| CAN FD, 8 data bytes, 1 Mbit/s and 5 Mbit/s (hypothetical: no CAN FD mode of the leg drives was found) | 30 bits at 1 Mbit/s + 96 bits at 5 Mbit/s; at worst + 4 and + 17 stuff bits | 49.2–56.6 µs | (d) |
| EtherCAT, one frame for all 12 drives (assumed 24 bytes of process data per drive) | 52 + 12 x 24 = 340 bytes x 8 / 100 Mbit/s + 12 x 1 µs | 39.2 µs | (d) |
<!-- latency_bus:frames end -->

A frame's length depends on its content: the bus inserts a "stuff" bit after five equal bits. The table gives the
frame without stuff bits and with the most there can be.

**CAN FD.** No CAN FD mode of these drives was found: the protocol extracts list classic CAN only (d: an absence, to
confirm with the vendor). The hypothetical row prices what a CAN FD leg drive would change. The Orin's controller and
the catalog's USB adapter both speak CAN FD (c: `jetson_can`, `pcan_usb_fd`), so the limit is the drives.

**Bus load** is the frames per cycle times the frame time, divided by the cycle:

<!-- latency_bus:bus_load begin -->
| Bus | Buses | Drives on each | Loop rate (cycle) | Frames per cycle × frame time | Load = that ÷ cycle | Fits | Fastest loop the bus carries (worst–no stuffing) | Kind |
|---|---|---|---|---|---|---|---|---|
| classic CAN, 1 Mbit/s | 1 | 12 | 250 Hz (4 ms) | 24 frames: 2.664–3.240 ms | 67–81 % | yes | 309–375 Hz | (c) |
| classic CAN, 1 Mbit/s | 1 | 12 | 500 Hz (2 ms) | 24 frames: 2.664–3.240 ms | 133–162 % | **no** | 309–375 Hz | (c) |
| classic CAN, 1 Mbit/s | 1 | 12 | 1000 Hz (1 ms) | 24 frames: 2.664–3.240 ms | 266–324 % | **no** | 309–375 Hz | (c) |
| classic CAN, 1 Mbit/s | 2 | 6 | 250 Hz (4 ms) | 12 frames: 1.332–1.620 ms | 33–40 % | yes | 617–751 Hz | (c) |
| classic CAN, 1 Mbit/s | 2 | 6 | 500 Hz (2 ms) | 12 frames: 1.332–1.620 ms | 67–81 % | yes | 617–751 Hz | (c) |
| classic CAN, 1 Mbit/s | 2 | 6 | 1000 Hz (1 ms) | 12 frames: 1.332–1.620 ms | 133–162 % | **no** | 617–751 Hz | (c) |
| classic CAN, 1 Mbit/s | 3 | 4 | 250 Hz (4 ms) | 8 frames: 0.888–1.080 ms | 22–27 % | yes | 926–1126 Hz | (c) |
| classic CAN, 1 Mbit/s | 3 | 4 | 500 Hz (2 ms) | 8 frames: 0.888–1.080 ms | 44–54 % | yes | 926–1126 Hz | (c) |
| classic CAN, 1 Mbit/s | 3 | 4 | 1000 Hz (1 ms) | 8 frames: 0.888–1.080 ms | 89–108 % | **no** | 926–1126 Hz | (c) |
| CAN FD, 1 and 5 Mbit/s (hypothetical: not offered by the leg drives as far as found) | 1 | 12 | 500 Hz (2 ms) | 24 frames: 1.181–1.358 ms | 59–68 % | yes | 736–847 Hz | (d) |
<!-- latency_bus:bus_load end -->

Reading it:
- "Fits" means the worst-case frame time is no longer than the cycle (d: the author's rule). The leg bus is polled:
  one master sends, the drives answer, nothing else talks. Such a bus can run close to full. But at 81 % there is
  little room for a resent frame or a status read (temperatures, error flags).
- **1000 Hz does not fit on one, two or three buses** with a command and a reply per drive. `COMPUTE.md`'s workload
  table gives the CAN loop 200–1000 Hz; with these drives the upper end is out of reach.
- The reviewer's estimate in `JETSON_PLEM_REVIEW.md` section 3 (2.7–3.2 ms, 133–162 % on one bus) is confirmed by
  this arithmetic with sourced frame lengths.
- Two ways to cut the load, neither confirmed: the protocol's multi-motor command on identifier 0x280 (c: it exists;
  d: what it carries on these drives was not read), and a third bus.

## 3. The one table: rate, load, delay and failure, per architecture

<!-- latency_bus:architectures begin -->
| Architecture | Loop rate with 12 leg drives (kind in each cell) | Bus load at that rate (c; E: d) | Control tick to torque change (d) | Age of the feedback at the tick (d; the model's is 2 ms) | End-to-end total: tick to torque, plus any feedback age beyond 2 ms (d) | Simulated delay it maps onto (d) | On a Linux crash | On a microcontroller crash |
|---|---|---|---|---|---|---|---|---|
| **A**: Raspberry Pi 5 + Teensy 4.1, two CAN buses (COMPUTE.md's design, the baseline) | 500 Hz fits the bus (c) | 67–81 % | 0.3–2.9 ms | 1.2–2.6 ms | 0.3–3.5 ms | 2–4 ms (k 1–2) | Controlled stop, then stand (once the firmware exists) | Torque-off or frozen drives; a fall when walking |
| **A3**: the same with three CAN buses (an option: the Teensy has three ports) | 500 Hz fits the bus (c) | 44–54 % | 0.3–2.6 ms | 0.9–1.8 ms | 0.3–2.6 ms | 2–4 ms (k 1–2) | Controlled stop, then stand (once the firmware exists) | Torque-off or frozen drives; a fall when walking |
| **B**: Jetson Orin Nano Super + Teensy 4.1, two CAN buses (the same real-time layer, another host) | 500 Hz fits the bus (c) | 67–81 % | 0.3–2.9 ms | 1.2–2.6 ms | 0.3–3.5 ms | 2–4 ms (k 1–2) | Controlled stop, then stand (once the firmware exists) | Torque-off or frozen drives; a fall when walking |
| **C1**: Jetson-direct, one USB-CAN channel (PR #668's layout, unverified) | **500 Hz does not fit** (c: 133–162 % load); 250 Hz chosen (d) | 67–81 % | 0.5–6.8 ms | 2.4–6.2 ms | 0.9–11.0 ms | 2–12 ms (k 1–6), at a target rate the model has not been run at | Torque-off or frozen drives; a fall when walking | None in the leg loop |
| **C2**: Jetson-direct, two USB-CAN channels | 500 Hz fits the bus (c) | 67–81 % | 0.5–4.0 ms | 1.1–3.5 ms | 0.5–5.5 ms | 2–6 ms (k 1–3) | Torque-off or frozen drives; a fall when walking | None in the leg loop |
| **D**: Jetson-direct, the Orin's own CAN controller with a transceiver | **500 Hz does not fit** (c: 133–162 % load); 250 Hz chosen (d) | 67–81 % | 0.3–6.0 ms | 2.6–5.5 ms | 0.9–9.5 ms | 2–10 ms (k 1–5), at a target rate the model has not been run at | Torque-off or frozen drives; a fall when walking | None in the leg loop |
| **E**: EtherCAT from the Jetson's Ethernet port (the leg drives are -E variants) | 500 Hz assumed (d): the drives' supported cycle was not found | 2 % | 0.3–3.3 ms | 1.2–1.9 ms | 0.3–3.3 ms | 2–4 ms (k 1–2) | Torque-off or the drives' fault reaction; a fall when walking | None in the leg loop |
<!-- latency_bus:architectures end -->

Kinds:
- **The CAN loads, and whether 500 Hz fits a bus, are (c):** bus-time arithmetic on sourced frame lengths. That
  arithmetic takes each drive's reply turnaround as shorter than the command burst, and assumes the drives accept a
  command each at 500 Hz. Neither figure was found in the vendor's documents (d); the bench sheet measures both.
- **The 250 Hz of C1 and D is the author's choice (d):** the fastest whole multiple of the model's 2 ms step that one
  bus carries.
- **E's rate and load are (d):** the EtherCAT frame is an assumed layout and the drives' supported cycle was not
  found.
- **Every delay, feedback age, total and simulated delay is (d).**
- **The failure columns** are engineering readings of each layout against `COMPUTE.md` and `BEHAVIOR_SPEC.md`
  (section 6); none has been tested.

For comparison only:
- PR #668 (unverified) put its Jetson-direct path at about 2 ms. `JETSON_PLEM_REVIEW.md` calls that an unsourced,
  host-side estimate. This budget's range for the same layout (C1) is 0.9–11.0 ms at half the model's target rate.
- `COMPUTE.md`'s workload table lists the locomotion policy as "trained with an 8 ms command delay" (c: a modelling
  figure of PR #662's 60 cm platform, whose drives were modelled with an 8 ms command delay and a 50 Hz hold:
  `designlab/data/research/compute_2026-11-07.json`; not a measurement, and not this budget's estimate).

### 3.1 The terms

<!-- latency_bus:terms begin -->
| Arch. | compute step, ms | host jitter, ms | host's path to the bus, ms | bus transmit, ms | drive's own loop, ms | hold of a slower target rate, ms | Tick to torque, ms (d) | Feedback age at the tick, ms (d) | Age beyond the model's 2 ms, ms (d) | Total, ms (d) | Whole exchange ends inside the cycle at worst (d) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** | 0.10–0.50 (d) | 0.00–0.05 (d) | 0.00–0.02 (d) | 0.11–0.81 (d) | 0.10–1.50 (d) | 0.00 (d) | **0.31–2.88** | 1.23–2.62 | 0.00–0.62 | **0.31–3.50** | no: up to 2.21 ms |
| **A3** | 0.10–0.50 (d) | 0.00–0.05 (d) | 0.00–0.02 (d) | 0.11–0.54 (d) | 0.10–1.50 (d) | 0.00 (d) | **0.31–2.61** | 0.89–1.79 | 0.00 | **0.31–2.61** | yes |
| **B** | 0.10–0.50 (d) | 0.00–0.05 (d) | 0.00–0.02 (d) | 0.11–0.81 (d) | 0.10–1.50 (d) | 0.00 (d) | **0.31–2.88** | 1.23–2.62 | 0.00–0.62 | **0.31–3.50** | no: up to 2.21 ms |
| **C1** | 0.02–0.20 (d) | 0.02–0.50 (d) | 0.20–1.00 (d) | 0.11–1.62 (d) | 0.10–1.50 (d) | 0.00–2.00 (d) | **0.45–6.82** | 2.43–6.17 | 0.43–4.17 | **0.88–10.98** | no: up to 5.94 ms |
| **C2** | 0.02–0.20 (d) | 0.02–0.50 (d) | 0.20–1.00 (d) | 0.11–0.81 (d) | 0.10–1.50 (d) | 0.00 (d) | **0.45–4.01** | 1.09–3.49 | 0.00–1.49 | **0.45–5.50** | no: up to 4.32 ms |
| **D** | 0.02–0.20 (d) | 0.02–0.50 (d) | 0.02–0.20 (d) | 0.11–1.62 (d) | 0.10–1.50 (d) | 0.00–2.00 (d) | **0.27–6.02** | 2.61–5.48 | 0.61–3.48 | **0.88–9.50** | no: up to 4.34 ms |
| **E** | 0.02–0.20 (d) | 0.02–0.50 (d) | 0.02–0.10 (d) | 0.04 (d) | 0.25–2.50 (d) | 0.00 (d) | **0.35–3.34** | 1.16–1.90 | 0.00 | **0.35–3.34** | yes |
<!-- latency_bus:terms end -->

How the columns are built:
- **Tick to torque** is the sum of the six terms. It is the time from the tick to the change of torque at the joint.
- **Bus transmit** is computed: the first drive on a bus has its command after one frame, the last after as many
  frames as the bus has drives. The frame times are sourced (c); the order is an assumption (d), so the term is
  labelled (d). It assumes the commands go out back to back and win the bus over the replies, which they do when
  their identifiers are lower (0x140 + *n* against 0x240 + *n*). The identifier of the motion-mode command is
  0x400 + *n* (c) and that of its reply was not read: if replies outrank commands they interleave with them. The
  bench stopwatch shows the real order.
- **Reply turnaround.** The time a drive takes to start its reply was not found (d). The exchange times take it as
  shorter than the rest of the command burst: 0.56–0.68 ms with six drives on a bus. A longer turnaround lengthens
  the exchange and the feedback's age.
- **Hold of a slower target rate.** At 250 Hz a target is held for 4 ms instead of the model's 2 ms, so it is up to
  2 ms older at the end of its hold (d).
- **Feedback age at the tick.** A drive samples its reply when its command arrives. The reply is used at the first
  tick after it is back at the controller. The range covers the first and the last drive on a bus, under the low and
  the high estimates. In the model this age is 2 ms.
- **Total** is tick to torque plus any feedback age beyond 2 ms.
- **Whole exchange ends inside the cycle.** If the last reply can arrive after the next tick, the loop must use
  feedback one cycle older for that drive, or wait. This is why A and B show "no" with two buses and A3 shows "yes".

Where each estimate comes from:

<!-- latency_bus:term_basis begin -->
| Term | Estimate, ms | Kind | Whose estimate and how derived |
|---|---|---|---|
| compute step, microcontroller | 0.10–0.50 | (d) | the author's estimate for one 500 Hz step of the tracker on a 600 MHz microcontroller: the reference look-up, the targets, the ankle cap and a share of the 100 Hz feedback. No firmware exists (COMPUTE.md). The study's Python tracker takes about 0.5 ms per physics step, physics included, on the cloud machine (LESSONS_LEARNED.md 3.12) [compute_md, lessons, teensy41] |
| compute step, Jetson real-time core | 0.02–0.20 | (d) | the author's estimate for the same step in compiled code on one isolated Jetson core; not measured |
| host jitter, microcontroller | 0.00–0.05 | (d) | the author's estimate for a timer-interrupt loop on a microcontroller with serial and USB interrupts beside it; not measured |
| host jitter, Jetson real-time core | 0.02–0.50 | (d) | low end: third-party tests of a PREEMPT_RT kernel on Orin modules report about 0.01-0.2 ms at worst (rt_jetson). High end: the author's allowance for the Orin Nano with the GPU loaded; no published figure for this board with the GPU loaded was found [rt_jetson] |
| path to the bus, microcontroller's CAN | 0.00–0.02 | (d) | the author's estimate: the microcontroller's own CAN controller, a transmit mailbox away from the bus |
| path to the bus, USB-CAN | 0.20–1.00 | (d) | the author's estimate for one USB hop: a High-speed USB 2.0 transfer waits for a 0.125 ms microframe, then the adapter's firmware and the Linux USB stack; no measured latency of this adapter was found [pcan_usb_fd] |
| path to the bus, the Orin's CAN controller | 0.02–0.20 | (d) | the author's estimate for a socket write through the Linux CAN driver to the Orin's own controller [jetson_can] |
| path to the bus, EtherCAT | 0.02–0.10 | (d) | the author's estimate for a raw Ethernet send from a real-time thread; published jitter of EtherCAT masters on other boards is 0.004-0.022 ms (ethercat_rt) [ethercat_rt] |
| drive's own loop, over CAN | 0.10–1.50 | (d) | the author's estimate from a frame's end to the torque change: up to one period of the drive's outer loop, taken as 1 ms, plus the current's rise (winding L / R: 0.18 mH / 0.4 ohm = 0.45 ms on the X4-36-L and 0.51 mH / 0.41 ohm = 1.24 ms on the X6-60, catalog notes; a closed current loop is faster). The vendor's driver manuals give a position loop of 1 kHz (V3.0) or 500 Hz (V2.1) for the MC-X drivers of the earlier RMD-X generation, with 5 kHz speed and 15 kHz current loops (myactuator_mc_driver); which loop closes the motion-mode gains on the V4 and V4.1 drives is not known. With a 500 Hz loop the high end would be 2.50 ms [myactuator_mc_driver, catalog_actuators] |
| drive's own loop, over EtherCAT | 0.25–2.50 | (d) | the author's estimate: a cyclic synchronous drive takes its setpoint at its next sync event, up to one 2 ms cycle, then the torque rises as over CAN; the vendor's cycle times were not found [myactuator_ethercat] |
<!-- latency_bus:term_basis end -->

**The drive's own loop, the soft spot.** The 1.50 ms upper end takes an outer loop of 1 kHz. If it is 500 Hz, the
slower of the two rates in the vendor's driver manuals, every CAN total's upper end rises by 1 ms (d): A and B to
4.5 ms (2–6 ms), A3 to 3.6 ms (still 2–4 ms), C2 to 6.5 ms (2–8 ms), C1 to 12.0 ms and D to 10.5 ms (both 2–12 ms).
The record holds these under `if_drive_loop_500_hz`.

The sensor age above is the leg encoders'. The IMU is read by the same controller at 1 kHz or faster in every
architecture (d: the author's plan), so it is never older than the encoders and does not add a term.

## 4. How a total maps onto the simulation's delay

What the model does, with file and line:

<!-- latency_bus:model begin -->
| What the model does | Where (file:line) |
|---|---|
| The controller sets all 14 servo targets every physics step of 2 ms (500 Hz). | `designlab/stage1/control.py:15`; `designlab/mjcf.py:113` |
| A target is an angle, a rate and a feedforward torque folded into one number per drive: ctrl = q + (kv qd + tau) / kp, in front of an ideal PD servo at the holding gains. | `designlab/stage1/control.py:18`; `designlab/stage1/plant.py:662` |
| The delay: every target, feedforward included, passes through a queue of the last k + 1 targets, so the servos get the target set k physics steps (2 k ms) earlier. | `designlab/stage1/env.py:513`; `designlab/stage1/env.py:516`; `designlab/stage1/env.py:522` |
| The ledger's delay readings quoted below were made by the ledger's own copy of the same queue, refilled with the reset's targets at every reset. | `designlab/stage1/ledger.py:708` |
| k is a whole number of steps, 0 to 10 (0-20 ms): the training environment's random range and the ledger's search range. | `designlab/stage1/env.py:235`; `designlab/stage1/ledger.py:75` |
| Each step: set the target, step the physics, then read the sensors for the next step. | `designlab/stage1/control.py:1429`; `designlab/stage1/control.py:1430`; `designlab/stage1/control.py:1434`; `designlab/stage1/plant.py:692` |
| The sensors the controller reads describe the state one physics step (2 ms) earlier. | `designlab/stage1/plant.py:704` |
| The balance feedback is computed every fifth step (100 Hz) and ramped into the targets: one fifth in the step it is computed, all of it four steps (8 ms) later. | `designlab/stage1/control.py:149`; `designlab/stage1/control.py:152`; `designlab/stage1/control.py:1410`; `designlab/stage1/control.py:1416` |
| The holding gains: kp = the drive's sustained planning torque / 0.01 rad (an assumption), kv = 2 sqrt(kp I), critical damping on the inertia I the drive sees at home with every other drive held: the leg's, not the bare shaft's. | `designlab/carpet.py:176`; `designlab/stage1/plant.py:858`; `designlab/stage1/plant.py:864` |
| Every joint has the generator's default damping, 0.2 N m s/rad, and friction loss, 0.15 N m: unsourced. | `designlab/mjcf.py:120` |
| Stage 1b's note N8 states the same and what a bench figure is compared with. | `stage1/runs/2026-11-10-speed/STAGE1B_SPEED.md:404` |
<!-- latency_bus:model end -->

Put plainly:
- The simulation's delay parameter is a whole number k of physics steps. Every joint target (the angle, the rate and
  the feedforward torque together) reaches the servos k steps of 2 ms after the controller set it.
- That delay sits on top of a loop that already lags in two ways. The sensors the controller reads are one physics
  step (2 ms) old. And the balance feedback is recomputed only every fifth step (100 Hz) and then ramped into the
  targets: one fifth in the step it is computed, all of it 8 ms later.
- Behind the delay the servo is ideal: a PD loop at the model's holding gains, evaluated every 2 ms, with no delay of
  its own.
- Seen from the controller, the model's round trip (set a target, see the joint respond) is k + 2 steps: k steps of
  delay, one step for the physics to move the joint and one step for the sensors' age.

So the real quantity that corresponds to 2 k ms is **the time from the control tick to the change of torque at the
joint**, on three conditions:
1. the targets go out at 500 Hz;
2. the feedback the loop uses is no more than 2 ms old at the tick. Any age beyond 2 ms is added to the delay, since it
   delays the feedback path, the one balance depends on;
3. the drive closes its own PD loop on the delayed target, at the model's gains.

**The rule (d):** k = (tick to torque + feedback age beyond 2 ms) ÷ 2 ms, rounded up to a whole step. Delays are never
read between steps (P29, P30): a total of 2.1 ms is run at 4 ms. The smallest k a real loop maps onto is 1.

What the mapping cannot carry:
- **A slower target rate.** The model has only been run with targets at 500 Hz. The 250 Hz layouts (C1, D) need a
  model variant with slower targets before their k means anything.
- **Neck and tail targets at 100 Hz.** The model sends all 14 targets at 500 Hz. A neck or tail target held for
  10 ms, plus that servo's own bus and delay, has not been simulated (section 5). These targets are known ahead of
  time, so a measured constant delay could be led in firmware (d).
- **Jitter.** A constant k stands in for a high percentile. Rare late cycles are not modelled.
- **A softer or less damped servo** than the model's. That is a separate reading (`BENCH_SHEET.md`, sheet A).

What the simulation has shown at each delay, for the 75 cm nominal walk (b). These are Stage 1's ledger records, 3
episodes per delay under two controllers: margin readings, not 20-episode verdicts.

<!-- latency_bus:ledger begin -->
| Simulated delay | R2/43: up · meeting every limit (b) | R3/42: up · meeting every limit (b) |
|---|---|---|
| 0 ms (k 0) | 3 of 3 · 3 of 3 | 3 of 3 · 3 of 3 |
| 4 ms (k 2) | 3 of 3 · 3 of 3 | 3 of 3 · 3 of 3 |
| 6 ms (k 3) | 3 of 3 · 3 of 3 | 3 of 3 · 3 of 3 |
| 8 ms (k 4) | 3 of 3 · 0 of 3; misses toe_clearance | 3 of 3 · 0 of 3; misses grf, toe_clearance |
| 10 ms (k 5) | 0 of 3 · 0 of 3; falls at 4.4–6.5 s | 0 of 3 · 0 of 3; falls at 4.6–4.8 s |
| 20 ms (k 10) | 3 of 3 · 0 of 3; misses cadence, drive_speed, grf, speed, stride | 3 of 3 · 0 of 3; misses drive_speed, duty, grf, slip_median, speed, stride, toe_clearance |
<!-- latency_bus:ledger end -->

R2/43 is the balance tracker on the robot's own sensors, in its fitted version 43. R3/42 is the same tracker with a
learned add-on. No delay search on any body ran 2 ms or 12–18 ms. The walk falls at 10 ms yet stays up at 20 ms, which
nobody has explained; 10 ms is also one period of the 100 Hz feedback (`LESSONS_LEARNED.md` section 2.3).

## 5. The neck and tail servos

<!-- latency_bus:aux begin -->
| Servo | Catalog interface (c) | Protocol | One write and one read (d) | Time at 1 Mbit/s, reply delay 0 to 500 µs (d) | Share of its own bus (d) | USB adapter on a Jetson-direct path |
|---|---|---|---|---|---|---|
| neck pitch: `robotis_xm540_w270_t` | TTL half-duplex multidrop (-T), 9.6 kbps-4.5 Mbps | DYNAMIXEL Protocol 2.0, TTL half-duplex | write goal position (4 bytes): 16 bytes out, 11 back; read current, velocity and position (10 bytes): 14 bytes out, 21 back | 0.62–1.62 ms | 6–16 % at 100 Hz | ROBOTIS U2D2 (not in the catalog) |
| tail yaw: `feetech_sts3215_c018` | TTL half-duplex serial bus, 38.4 kbps-1 Mbps, IDs 0-253; 12-bit magnetic encoder | Feetech STS serial protocol, TTL half-duplex | write position, time and speed (6 bytes): 13 bytes out, 6 back; read position, speed and load (6 bytes): 8 bytes out, 12 back | 0.39–1.39 ms | 4–14 % at 100 Hz | Feetech FE-URT-1 (not in the catalog) |
<!-- latency_bus:aux end -->

- **Two protocols, so two buses (d).** Both are one-wire serial buses, but the ROBOTIS and Feetech packet formats
  differ. The plan gives each servo its own serial port instead of mixing them on one wire.
- **They avoid the leg cycle entirely.** Neither touches a CAN bus. On the Teensy paths (A, A3, B) each has its own
  serial port, serviced between the CAN frames. On the Jetson-direct paths (C1, C2, D, E) each needs a USB adapter and
  runs in a separate thread at lower priority.
- **Rate (d).** The model updates all 14 targets at 500 Hz. The neck and tail are not in the balance loop, so the plan
  sends their targets every fifth leg cycle, at 100 Hz. One write and one read then take 0.4–1.6 ms.
- **Settings that matter (c).** Three, all to change before the bench:
  - The XM540's bit rate is 57,600 bit/s as delivered (`xm540`, `xm540_baud`). At that rate the write and read above
    take 10.8–11.8 ms (d: the same packet sizes), longer than the 10 ms period of 100 Hz. Set it to 1 Mbit/s.
  - The XM540's reply delay is 500 µs by default and can be set to 0 (`xm540`).
  - The ROBOTIS USB adapter's latency timer is 16 ms by default and must be set to 1 ms (`u2d2`): left at its
    default it alone would add up to 16 ms to every read on a Jetson-direct path.
- **Voltage levels (c).** The Teensy's pins take 3.3 V at most (`teensy41`) and the DYNAMIXEL TTL bus is 5 V logic
  (`dxl_ttl_level`). On the Teensy paths each servo bus therefore needs a level-shifting half-duplex buffer. The
  STS3215's line is listed at a high level of 2–5 V (`sts3215_serial`); its idle voltage was not found.
- **Not known (d).** The STS3215's default reply delay (two write-ups disagree), the delay of either servo from a
  target to motion, and whether their own position loops add a profile. The bench sheet measures all three.
- **Why their delay matters less, and where it does.** They do not balance the robot. But the model delays their
  targets with the rest, and look feature F9's tail phase, which cuts the tail servo's load, is a timing between the
  tail and the stride (`stage2/look/SCREEN.md`).

## 6. What happens on a crash

The requirement (`BEHAVIOR_SPEC.md` section 3): an operator stop is honoured within 0.3 s; on a lost heartbeat the
robot makes a controlled stop within 1.0 s, then stands, and does not move until reconnection plus an explicit start.
`COMPUTE.md` adds the design rule that P30 keeps as the baseline: the walking loop, the CAN buses, the ToF stop and
the watchdogs run on the microcontroller, so that a Linux crash still ends in a controlled stop.

A lost operator link is handled in software by whatever runs the walking loop, so every architecture can meet that
line while its loop is alive. The architectures differ in what happens when a computer itself fails:

<!-- latency_bus:failure begin -->
| Arch. | Walking loop runs on | Linux crash (kernel panic, hang, or the real-time thread dies) | Microcontroller crash |
|---|---|---|---|
| **A** | Teensy 4.1 | The microcontroller keeps the walking loop, the host's velocity request expires and the stop runs from the reference: a controlled stop, then a stand. Meets the requirement once the firmware exists; it is unwritten (COMPUTE.md). | No commands reach the drives until the microcontroller's watchdog restarts it, and Linux has no path to the leg bus. What the drives do without commands is not confirmed (they have a time-out setting): plan on torque-off, which for a walking biped is a fall. Standing, the robot may stay up if the restart is quick and the drives hold. |
| **A3** | Teensy 4.1 | The microcontroller keeps the walking loop, the host's velocity request expires and the stop runs from the reference: a controlled stop, then a stand. Meets the requirement once the firmware exists; it is unwritten (COMPUTE.md). | No commands reach the drives until the microcontroller's watchdog restarts it, and Linux has no path to the leg bus. What the drives do without commands is not confirmed (they have a time-out setting): plan on torque-off, which for a walking biped is a fall. Standing, the robot may stay up if the restart is quick and the drives hold. |
| **B** | Teensy 4.1 | The microcontroller keeps the walking loop, the host's velocity request expires and the stop runs from the reference: a controlled stop, then a stand. Meets the requirement once the firmware exists; it is unwritten (COMPUTE.md). | No commands reach the drives until the microcontroller's watchdog restarts it, and Linux has no path to the leg bus. What the drives do without commands is not confirmed (they have a time-out setting): plan on torque-off, which for a walking biped is a fall. Standing, the robot may stay up if the restart is quick and the drives hold. |
| **C1** | an isolated PREEMPT_RT core | The walking loop is the thing that crashed, so the drives get no new targets: torque-off or a frozen last command, then a fall when walking. Does not meet the requirement. A supervisor can only cut power unless it can drive the leg bus itself, and then it is architecture B's microcontroller. | No microcontroller in the leg loop. If a supervisor is added and it hangs, its fail-safe output decides: an enable line that drops means torque-off. |
| **C2** | an isolated PREEMPT_RT core | The walking loop is the thing that crashed, so the drives get no new targets: torque-off or a frozen last command, then a fall when walking. Does not meet the requirement. A supervisor can only cut power unless it can drive the leg bus itself, and then it is architecture B's microcontroller. | No microcontroller in the leg loop. If a supervisor is added and it hangs, its fail-safe output decides: an enable line that drops means torque-off. |
| **D** | an isolated PREEMPT_RT core | The walking loop is the thing that crashed, so the drives get no new targets: torque-off or a frozen last command, then a fall when walking. Does not meet the requirement. A supervisor can only cut power unless it can drive the leg bus itself, and then it is architecture B's microcontroller. | No microcontroller in the leg loop. If a supervisor is added and it hangs, its fail-safe output decides: an enable line that drops means torque-off. |
| **E** | an isolated PREEMPT_RT core | The EtherCAT master is the thing that crashed: torque-off or the drives' own fault reaction, then a fall when walking. Does not meet the requirement, unless the master runs on a microcontroller, which nobody here has shown. | No microcontroller in the leg loop. If a supervisor is added and it hangs, its fail-safe output decides: an enable line that drops means torque-off. |
<!-- latency_bus:failure end -->

Notes:
- **What a drive does when commands stop is not confirmed (d).** The drives have a "communication interruption
  protection time" setting (c: `myactuator_timeout`). What happens when it runs out was not read. The planning
  assumption is torque-off. The bench sheet's safety step records the real behaviour.
- **The ToF stop.** In A, A3 and B it runs on the microcontroller and survives a Linux crash. In C1, C2, D and E it
  would run on the Jetson beside the walking loop and fail with it.
- **A USB cable is one more thing to fail** in C1 and C2: a pulled or shaken connector is a Linux-side bus loss.
- **The firmware is the gate for the baseline too.** Until the Teensy firmware exists, no architecture ends a Linux
  crash in a controlled stop (`COMPUTE.md`).

## 7. EtherCAT

- **Do the 75 cm leg drives offer it? Yes (c).** The catalog lists both as "CAN bus + EtherCAT"
  (`designlab/data/actuators.json`). The vendor's naming reads the "-E" in both model names as EtherCAT and CAN
  (`actuator_research`, `myactuator_naming`). The vendor publishes an EtherCAT control protocol with cyclic
  synchronous position, velocity and torque modes (`myactuator_ethercat`).
- **On paper it has the most room (d).** One frame of about 340 bytes carries all 12 drives in about 0.04 ms, 2 % of
  a 2 ms cycle, under an assumed process-data size.
- **What was not found.** The drives' shortest supported cycle; their behaviour when the master stops; whether the
  cyclic modes can carry the model's servo form (a position target with a rate and a feedforward torque, at set
  gains); and the mass and routing of a second cable chain through the legs.
- **Its safety story is C's, not A's.** With the master on the Jetson, a Linux crash stops the frames. An EtherCAT
  master on a microcontroller would restore the baseline's behaviour, but nobody here has shown one.
- **A caution from the catalog (c).** The US store page for the hip-pitch drive conflicts with its datasheet; the
  catalog's note asks to confirm which variant ships.

## 8. Which numbers only the bench test can replace

Sourced or computed, and not expected to move: the drive counts, the bit rate, the frame lengths, the bus loads, the
number of CAN ports on each computer.

Estimates that stand until measured (`BENCH_SHEET.md`):
1. the drive's own delay from a frame to torque, per drive type: 0.1–1.5 ms assumed for the leg drives;
2. the compute step of the walking loop on each computer: no firmware exists;
3. the host's jitter on the Jetson's real-time core with the GPU loaded: no published figure for this board with
   the GPU loaded was found;
4. the USB hop of the USB-CAN adapter, each way;
5. each leg drive's reply turnaround, the highest command rate it accepts, and the order of commands and replies on
   the bus;
6. whether the two-bus exchange really overruns the 2 ms cycle, and whether three buses cure it;
7. the neck and tail servos' delays and reply settings;
8. the drives' behaviour when commands stop;
9. whether the drives' own PD loop reaches the model's gains, in stiffness and in damping.

Not measurable at the bench, and needing a decision or a simulation instead: a 250 Hz target rate in the model, and
the EtherCAT cycle the drives support (a vendor question).

## 9. What this does not show

- **No measurement.** Every delay here is an estimate. The totals are sums of ranges: their ends are unlikely
  corners, and nothing says where inside the range the truth lies.
- **No new simulation.** The mapping onto k uses Stage 1's model as it stands. The readings quoted are 3-episode
  margin readings of the nominal walk; the 0.40 m/s bar has no record at these delays.
- **No firmware.** The Teensy paths' failure behaviour is a design intent until the firmware is written and tested.
- **No choice.** Nothing here chooses a computer, a bus or a power mode (P30). P14 stays adopted.
- **Web facts read through extracts**, not on the pages (see the top).
- **Not covered:** the Jetson body's mass, heat and power (task 6), the budget (task 9) and the ToF speed governor
  (task 10).

## 10. Web sources

Each was read on 14 Nov 2026 through the search tool's extract. The study's own sources (records, documents and
catalog files) are listed in `latency_bus.json` under `sources`.

<!-- latency_bus:sources begin -->
| Key | URL | What it says (as extracted) |
|---|---|---|
| `myactuator_protocol` | https://cdn.robotshop.com/rbm/acdeb3e1-04dc-4138-8b99-88509ddc4e94/4/4501bc19-2742-44e8-8d9e-8a4b398c18c4/e6052050_rmd-x-motor-motion-protocol-v4.01.pdf ; https://cdn.shopify.com/s/files/1/0673/6848/5000/files/Motor_Motion_Protocol.pdf?v=1756635504 | MyActuator Servo Motor Control Protocol V4.01 (and a V4.3 copy): bus interface CAN, baud rate 1 Mbps; single-motor command 0x140 + ID, reply 0x240 + ID, multi-motor command 0x280; data frames, standard frame, 8 data bytes (DLC 8); a 'Motion Mode Control Command_CAN (0x400 + ID)' |
| `myactuator_sibling_manual` | https://manuals.plus/myactuator/rmd-x4-p36-24-rmd-series-motor-manual ; https://github.com/2b-t/myactuator_rmd/issues/9 | the manual of a sibling model (RMD-X4-P36-24): a 'force position hybrid control' frame with KP in 12 bits (0-4095 for 0.0-500.0), KD in 9 bits (0-511 for 0.0-5.0), position 16 bits, speed 12 bits and feedforward torque 12 bits; a three-loop servo whose current loop the user cannot tune. An issue of a community driver describes the motion-mode command on 0x400 + ID the same way: a desired position and velocity, kp, kd and a feedforward torque |
| `myactuator_mc_driver` | https://manuals.plus/myactuator/mc-series-brushless-servo-driver-manual ; https://cdn.robotshop.com/rbm/acdeb3e1-04dc-4138-8b99-88509ddc4e94/0/052b77d7-e3c8-45a8-a14d-df19bf957084/73c10167_mc-series-brushless-servo-driver-manual-24.2.1.pdf ; https://cdn.robotshop.com/media/m/mya/rbc-mya-04/pdf/mc300a_series_servo_motor_driver_manual_v2.1.pdf ; https://cdn.robotshop.com/media/m/mya/rbc-mya-09/pdf/myactuator-rmd-x6-v3-can-bus-18-mc-x-300-o-brushless-servo-driver-manual.pdf | MyActuator's MC-series drivers, the drivers of the earlier RMD-X generation (a distributor files an MC-X-300-O manual under the RMD-X6 V3). The V3.0 manual, which covers the MC-X-500-O: current loop 15 kHz, speed loop 5 kHz, position loop 1 kHz. An older V2.1 manual of the MC300A series: torque 15 kHz, speed 5 kHz, position mode 500 Hz. No loop rate of the V4 and V4.1 drives was found |
| `myactuator_setup` | https://wiki.seeedstudio.com/myactuator_series/ ; https://cdn.robotshop.com/rbm/acdeb3e1-04dc-4138-8b99-88509ddc4e94/7/709f6435-daac-4a0f-9e2a-de033477cc27/9af65af7_setup-software-instructionmanual-v3.pdf ; https://uk.robotshop.com/products/myactuator-can-adapter-analyzer-for-rmd-x-v4-rh-and-h-series | the vendor's set-up software runs on a Windows PC and needs a USB-CAN interface of its own for CAN drives (a distributor's wiki; the software's package holds a driver for one adapter family); a retailer lists 'MyActuator CAN Adapter Analyzer for RMD-X V4, RH and H Series' at about 55 pounds. Whether the software accepts the PCAN-USB FD was not found |
| `myactuator_ethercat` | https://manuals.plus/m/afac3f8b251f8f42b18fa2bec083396ebc325afc4b36fc1d655cee2cdf580301 ; https://cdn.robotshop.com/rbm/acdeb3e1-04dc-4138-8b99-88509ddc4e94/3/34224137-babc-4373-9ac9-9d1fbf7ad526/bbabf271_ethercat-control-protocol-v1.0-250430.pdf | a mirror of MyActuator's 'EtherCAT Control Protocol' V1.0: cyclic synchronous position, velocity and torque modes (CSP, CSV, CST); no supported cycle time was found in the extract |
| `myactuator_naming` | https://www.switch-science.com/products/10516 | a distributor's page for the RMD-X6-P20-60-E: in the model name, E = EtherCAT & CAN BUS and C = CAN BUS |
| `myactuator_timeout` | https://docs.ros.org/en/ros2_packages/humble/api/myactuator_rmd/generated/typedef_responses_8hpp_1abb5dc20fefee579cda714dd829f56c52.html | a community ROS 2 driver documents the drives' 'communication interruption protection time setting' command; what the drive does when that time runs out was not in the extract |
| `can_frame_bits` | https://lkml.iu.edu/hypermail/linux/kernel/2306.1/08769.html ; https://elearning.vector.com/mod/page/view.php?id=349 | a Linux kernel patch lists a classic CAN standard-format frame with 8 data bytes as 111 bits without bit stuffing and 135 bits with the worst stuffing; Vector's course gives at most 24 stuff bits and a 132-bit longest frame before the 3-bit intermission |
| `teensy41` | https://pjrc.com/store/teensy41.html ; https://www.pjrc.com/teensy/card11b_rev4_web.pdf | Teensy 4.1: ARM Cortex-M7 at 600 MHz; 3 CAN buses (1 with CAN FD). Its pinout card: 'Teensy 4.1 pins are not 5 volt tolerant. Do not apply more than 3.3V to any pin, except VIN & VUSB' |
| `teensy41_price` | https://www.sparkfun.com/teensy-4-1.html | SparkFun's listing of the Teensy 4.1 (DEV-16771): $31.50 each (seen on 14 Nov 2026; distributors' prices run higher). An earlier draft's $35.50 from PJRC's own page could not be confirmed |
| `jetson_can` | https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/HR/ControllerAreaNetworkCan.html ; https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/HR/ControllerAreaNetworkCan.html ; https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf | Jetson Linux developer guide: Orin NX and Nano have 1 CAN controller (driver mttcan, 50 MHz clock); the interface needs an external transceiver of at least 3.3 V; CAN FD mode up to 15 Mbps, and 'all types of transceivers can achieve' 5 Mbps; the dev kit's carrier has an optional 4-pin CAN header, J17 |
| `pcan_usb_fd` | https://www.peak-system.com/products/hardware/external-pc-interfaces/pcan-usb-fd/ | PCAN-USB FD: High-speed USB 2.0; CAN 25 kbit/s to 1 Mbit/s; CAN FD data field up to 12 Mbit/s; time stamp resolution 1 µs; galvanic isolation up to 500 V |
| `rt_jetson` | https://www.wimcorp.co.kr/tech/en/posts/embedded/preempt-rt-jetson-orin/ ; https://proventusnova.com/blog/jetson-orin-preempt-rt-real-time-kernel ; https://forums.developer.nvidia.com/t/how-to-improve-real-time-performance-of-the-rt-kernal/338651 ; https://wiki.seeedstudio.com/deploy_preempt_rt_kernel_with_prebuilt_deb_package_on_recomputer_jetson/ ; https://forums.developer.nvidia.com/t/jetpack-6-2-agx-orin-64gb-preempt-rt-large-cyclictest-latency-variation-between-different-modules-up-to-90-us-even-with-cpu-isolation/381288 ; https://forums.developer.nvidia.com/t/observing-latency-discrepancies-on-nvidia-jetson-nano-with-preempt-rt/329935 | third-party wake-up latency tests of a PREEMPT_RT kernel on other Jetson Orin modules: below 20 µs maximum on an AGX Orin only with an isolated core as well; 50-120 µs worst case on a quiet isolated core; one AGX module about 10 µs and another above 90 µs; an Orin Nano thread reports averages of about 25 µs (cores 0-3) and 40 µs (cores 4-5); forum threads report a maximum of about 200 µs on the real-time kernel, in one case down from 400 µs once the clocks were pinned. A page to read in full: a vendor's wiki tests the real-time kernel on a carrier with the Orin Nano Super module (6 cores, the CPU fully loaded: wake-up latency, pin jitter and an EtherCAT master at 1 kHz); its figures were not in the extract |
| `xm540` | https://emanual.robotis.com/docs/en/dxl/x/xm540-w270 | XM540-W270 control table: Return Delay Time, default 250, range 0-254, unit 2 µs (500 µs by default); Baud Rate at address 8, initial value 1, range 0-7 |
| `xm540_baud` | https://manualshive.com/ru/manual/robotis/xm540-w150/1471413?page=2 ; https://robosavvy.com/store/robotis-dynamixel-xm540-w150-r.html | a copy of the XM540-W150 manual, the same servo family: Baud Rate value 1 = 57,600 bit/s (the default), 3 = 1 Mbit/s; a retailer's listing gives the default as 57600 bps. Confirm on ROBOTIS's own page for the W270 |
| `dxl_ttl_level` | https://emanual.robotis.com/docs/en/reference/dxl-selection-guide/ | ROBOTIS's selection guide: the TTL bus is 0 to 5 V logic; its circuit 'is designed for 5V or 5V tolerant MCU. Otherwise, use a Level Shifter to match the voltage of MCU' |
| `sts3215_serial` | https://www.dfrobot.com/product-2961.html ; https://core-electronics.com.au/attachments/uploads/sts3215-smart-servo-datasheet-translated.pdf | a distributor's listing of the ST3215: a TTL-level single-wire bus, high level 2-5 V, low level 0-0.45 V; a translated datasheet gives the factory bit rate as 1 Mbit/s. The idle voltage of the data line was not found |
| `adxl335` | https://www.analog.com/en/products/adxl335 ; https://community.element14.com/products/devtools/technicallibrary/w/documents/10547/analog-devices-eval-adxl335z-three-axis-accelerometer-evaluation-board | ADXL335 analog accelerometer: the bandwidth is set by a capacitor on each output, 1 / (2 pi x 32 kOhm x C); its evaluation board's 100 nF capacitors give 50 Hz |
| `u2d2` | https://emanual.robotis.com/docs/en/parts/interface/u2d2/ ; https://stanfordasl.github.io/trunk-stack/motors | ROBOTIS U2D2 USB adapter: set the serial chip's latency timer to 1 ms (on Linux through /sys/bus/usb-serial/devices/ttyUSB0/latency_timer); a lab's notes give its default as 16 ms |
| `sts3215_registers` | https://docs.rs/crate/st3215/latest/source/src/registers.rs ; https://blog.youkuaiyun.com/gitblog_00860/article/details/151443061 | a third-party driver's register notes for the ST3215: return delay in units of 2 µs, at most 508 µs, initial value 0; another write-up gives a factory default of 500 µs: the two disagree |
| `vl53l5cx` | https://www.st.com/resource/en/user_manual/um2884-a-guide-to-using-the-vl53l5cx-multizone-timeofflight-ranging-sensor-with-a-wide-field-of-view-ultra-lite-driver-uld-stmicroelectronics.pdf | ST user manual UM2884: ranging frequency 1-60 Hz at 4 x 4 zones and 1-15 Hz at 8 x 8 zones |
| `ethercat_rt` | https://www.forlinx.net/article_view_776.html ; https://www.acontis.com/en/running-ethercat-in-real-time-on-linux.html | EtherCAT masters on PREEMPT_RT Linux at a 1 ms cycle, on other boards: about 4-6 µs of jitter with the IgH master on a small network; about 22 µs with a vendor's master on boards including a Jetson AGX Orin |
<!-- latency_bus:sources end -->

## 11. Draft decision-log rows

Drafts for the owner. Nothing is adopted here, and the row numbers are for the lead to assign.

- **Draft E row (the leg bus).** "With the vendor's CAN protocol (classic CAN at 1 Mbit/s, 8 data bytes, a command and
  a reply per drive per cycle), one bus cannot carry the 12 leg drives at 500 Hz: 24 frames take 2.66–3.24 ms, 133–162 %
  of the cycle, and the bus tops out at 309–375 Hz. Two buses carry 500 Hz at 67–81 % load and three at 44–54 %;
  1000 Hz fits on none of them. The Orin Nano has one CAN controller; the Teensy 4.1 has three." Evidence:
  `stage2/jetson75/latency_bus.json` (`frames`, `bus_load`); web sources `myactuator_protocol`, `can_frame_bits`,
  `jetson_can`, `teensy41`, read through search extracts. Status: open, to confirm on the vendor's pages.
- **Draft A row (the delay estimates).** "Until the delay bench test reports (A17), the end-to-end delay of each
  compute architecture is an estimate: 0.3–3.5 ms for the Teensy with two CAN buses under either host; 0.5–5.5 ms for
  the Jetson with two USB-CAN channels; 0.9–11.0 ms and 0.9–9.5 ms, at 250 Hz, for the Jetson with one USB-CAN channel
  and with its own CAN controller; 0.3–3.3 ms for EtherCAT if the drives accept a 2 ms cycle. Each CAN upper end is
  1 ms higher if the drives' position loop runs at 500 Hz, not 1 kHz (`stage2/jetson75/LATENCY_AND_BUS.md`). The bar
  stays judged at 10 ms (P29, P30)." Replace with: the bench sheet's median, 99th-percentile and largest reading per
  drive type and path.
- **Draft A row (the servo form).** "The hardware servo is assumed to be the drive's own PD loop with a feedforward
  torque, fed targets at 500 Hz, at the model's holding gains. Whether the 75 cm leg drives' loop reaches those gains
  is unconfirmed: a sibling model's manual gives gain ranges that, if they applied in the model's units, would cap the
  hip pitch at 0.44 of the model's stiffness, and the damping at 0.16–0.66 of the model's on the hip and knee drives,
  a case no simulation has run (the stiffness search kept the damping unchanged)." Replace with: the vendor's protocol
  for these drives and the stiffness bench (`stage2/jetson75/BENCH_SHEET.md`, sheet A).
- **Draft note for the simulation plan.** "A leg loop slower than 500 Hz (any single-bus layout) is outside what the
  model has been run at; its delay cannot be read from the 500 Hz ledger."
