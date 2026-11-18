# Delay bench sheet for the 75 cm build (decision-log row A17)

**Changed, 15 Nov 2026 (P32).** The owner does not expect to run this bench test before a 75 cm build works in
simulation. The sheet is unchanged otherwise and stays ready; one sentence below is corrected (where this sheet said
the bench "runs first"). Until a bench value exists, the agility bar stays judged at the assumed 10 ms of servo delay.

**Changed, 14 Nov 2026 (P31).** The two USB-CAN adapters ($736, c) are stated as bench tools that this sheet uses. The
request that the owner confirm them is removed in four places: "In brief", the parts table's note, section 2 and
section 10's drafts. No price or quantity changed.

14 Nov 2026. **A sheet to run by hand. Nothing on it has been measured yet.** It is task 8 of `JETSON_PLEM_REVIEW.md`
section 5, for the body `75-M-H-10`. The owner's decision of 14 Nov (P30) was that the delay bench runs first, on both
paths, before any compute purchase; on 15 Nov the owner said it is not expected before a 75 cm build works in
simulation (P32). The order of the two paths and "before any compute purchase" stand for whenever it is run.

**Purpose, in three lines.**
1. Measure, for each drive type of the 75 cm build, how long a command takes to become motion, end to end, on both
   control paths: the Teensy's CAN loop, and a real-time core on the Jetson with USB-CAN adapters.
2. Map each reading onto the simulation's delay, in whole 2 ms steps, so that the agility bar can be re-run at the
   measured value instead of the assumed 10 ms.
3. Then measure the other things the simulation assumes and nobody here has measured: servo stiffness, joint
   friction, the chest ToF stop, tread grip and the tail servo's heat.

Paths are relative to `hardware/trex_form_study/`. The budget behind this sheet is `stage2/jetson75/LATENCY_AND_BUS.md`
("the budget" below). Its tables and this sheet's are machine-made by `stage2/jetson75/latency_bus.py` from
`stage2/jetson75/latency_bus.json`. P, A and F rows are in `DECISION_LOG.md`.

**Kinds of evidence**, as in the budget: (a) a prescribed-walk or static model result; (b) a closed-loop simulation
result; (c) a sourced fact (a catalog id, a document or record of this study, or a web page listed in the budget's
section 10, read through search extracts on 14 Nov 2026); (d) an estimate or assumption, the author's unless said.
The blank cells of this sheet will be the study's first hardware measurements. They are not model results, and they
must never be mixed with the targets beside them.

**Words used once and then abbreviated.**
- Teensy 4.1: the microcontroller board that runs the walking loop in the baseline design. Jetson: NVIDIA's Orin Nano
  Super dev kit. Host: the Linux computer that sends the walking request.
- CAN: the two-wire bus the leg drives speak; a *frame* is one message on it. USB-CAN: a USB adapter that gives a
  computer a CAN port. PREEMPT_RT: the Linux kernel's real-time option.
- Tick: the instant the control loop takes its sensor snapshot and sets its targets, 500 times a second.
- p50, p99, p99.9, max: the median, the values that 99 of 100 and 999 of 1,000 readings stay under, and the largest.
- k: the simulation's delay, a whole number of 2 ms steps.
- The bar: the agility test of `STAGE2_PLAN.md` section 5 (walk at 0.40 m/s, turn 90° in 2.0 s, take a nudge).
- ToF: time-of-flight distance sensor. GPU: graphics processor. 6S: a battery of six cells in series, 22.2 V.
- PD servo: a position servo with a stiffness gain (kp) and a damping gain (kv).
- Look features (F9 and others, where the text says "look feature") are those of `stage2/look/SCREEN.md`, not the
  decision log's F rows.

## In brief, for the owner

- **What you measure.** Per drive type and per path: the time from the control tick to the change of torque at the
  joint (p50, p99 and max over 1,000 commands), at 500 Hz, with all 12 leg drives' traffic on two buses, with the
  computer idle and loaded.
- **What you need.** Run 1 uses one drive of each type and replays the other 10 leg drives' traffic: $1,998 of
  catalog parts (c), about $2,431 with the two Teensys and the small parts (d). Run 2 uses all 12 leg drives: $4,867
  of catalog parts (c), of which $3,546 is the 12 leg drives that the build needs whatever the compute; about $5,300
  with the Teensys and small parts (d). Both runs include two USB-CAN adapters ($736): bench tools that this sheet
  uses to measure the Jetson path, and that only a Jetson-direct layout would use on the robot. Add a 24 V, 10 A
  bench supply if you have none (not priced) and a Windows PC for the leg drives' set-up software. The Jetson dev
  kit is on hand (`COMPUTE.md`, bench test 1).
- **What it takes.** About 10.5–16.5 hours at the bench for run 1, 6.5–10 hours more for run 2, and 11–16.5 hours for
  the five follow-on sheets (d). Two test programs have to be written first, and four small ones for the follow-on
  sheets (section 3.1); none exists yet.
- **A method that needs no special equipment.** The drives' own replies, time-stamped by a second Teensy on the bus:
  2 ms resolution as they come, and 0.2 ms bins with an extra read that the loop times (d: to be shown at the bench).
  A piezo disc on the output, sampled every 0.1 ms, times the motion itself; its own lag has to be checked. A phone's
  slow-motion video gives one frame, 4.2 ms at 240 frames a second: a sanity check only.
- **How to read it.** Take the p99 under the worst load condition, add any feedback age beyond 2 ms, round up to a
  whole 2 ms step, and re-run the bar at that step (section 6). Never read between steps. The reading does not yet
  hold the walking controller's own compute step: no firmware exists.
- **What the simulation has shown so far (b).** The 75 cm nominal walk meets every limit to 6 ms, stays up to 8 ms
  and falls at 10 ms; 2 ms was never run. On paper the Teensy path lands at 2–4 ms and the Jetson path with two
  USB-CAN channels at 2–6 ms (d, the budget); at 2–6 and 2–8 ms if the drives' position loop is the slower of the two
  rates the vendor's manuals give. The bench decides.

## 1. What is measured, exactly

### 1.1 The events

- **Start, t0: the tick.** The instant the control loop begins a cycle: it takes its sensor snapshot and hands the new
  targets to the bus. In the simulation this is the instant the controller sets a target.
- **End, t1: the torque change.** The first instant the drive under test pushes with the new torque. The output
  moves only after any play in the gears is taken up, so the procedure keeps the gear on its driving flank (section
  4.1, step 2); then the torque change and the first motion coincide.
- **The delay, T = t1 − t0.** It has two parts that are measured separately and added:
  - **H, the host-side time:** from the tick to the end of the last command frame on the drive's bus. It contains the
    computer's jitter, its path to the bus and the frames queued ahead. It is where the two paths differ, and where
    a loaded computer could show.
  - **D, the drive-side time:** from the end of the command frame to the torque change. It belongs to the drive and
    should be the same on both paths.
- **What H does not contain: the walking controller's compute step.** No firmware exists, and the loop program of
  this bench computes nothing. The budget estimates that step at 0.10–0.50 ms on the Teensy and 0.02–0.20 ms on the
  Jetson (d). The reading is short by that much; section 6 says how to carry it.
- **The feedback age, S.** How old a drive's reply is at the tick that uses it. In the simulation it is 2 ms. Any age
  beyond 2 ms is added to T before reading the result.
- **The reading:** T at its p99, plus the p99 of S beyond 2 ms, under the worst load condition. Adding two p99 values
  overstates the p99 of the sum a little; that is the safe side.

### 1.2 The stopwatch

A second Teensy 4.1, the same one on both paths, so that the two paths are timed by one instrument:
- it listens on each CAN bus (up to three) through its own transceiver and time-stamps the start and the end of every
  frame;
- it takes two logic lines from the computer that runs the loop: "tick" (raised at every tick) and "replies in"
  (raised when the loop has every reply of the cycle);
- it reads the piezo disc of method M2 on an analog input;
- in run 1 it also plays the 10 absent leg drives: it answers their commands with reply frames, so the bus carries the
  traffic of 12 drives;
- it logs to a laptop over USB.

Its clock runs at 600 MHz (c: `teensy41`). Its timing error is expected to be a few tens of microseconds (d). Check it
once, two ways. Time a square wave of known period: the loop Teensy's timer output at 1 kHz. And time the spacing of
two back-to-back frames, start to start: 111–135 µs at 1 Mbit/s, the exact value following from the frames' content
(a frame alone is 3 bit times shorter on the wire: the rest is the pause before the next one).

### 1.3 The methods

| Method | What marks the torque change | Extra equipment | Resolution | Use it for |
|---|---|---|---|---|
| **M1, coarse** | The drive's normal reply, once per cycle: its torque current, speed or angle leaves its rest value | none | one cycle: 2 ms at 500 Hz, the simulation's own step | every run; an upper bound |
| **M1, fine** | An extra status read to the drive under test, sent by the loop program itself a set time after each step command; that time is swept in 0.2 ms steps over the 1,000 commands, and each read is binned by the stopwatch's own stamp of it | none | 0.2 ms bins (d), to be shown at the bench: a read can only start at a frame boundary, 0.11–0.135 ms apart | D on every leg-drive run |
| **M2** | A piezo disc taped to the pointer arm on the output, sampled by the stopwatch at 10 kHz | the disc and four small parts, about $10 (d) | 0.1 ms sampling; the sensor's own lag is checked once (below) | T in one piece; the neck and tail servos |
| **M3** | A light that the loop switches on at the step command's tick, and a pointer on the output, filmed together in a phone's slow motion | a phone, a light | one video frame: 4.2 ms at 240 frames a second, 8.3 ms at 120 | a sanity check only: it cannot separate 2 ms steps |

Notes on the methods:
- **H needs no method of its own:** the stopwatch measures it on every cycle, to its own resolution.
- **M1** relies on the drives' replies carrying the torque current, the speed and the angle (d: expected from the
  vendor's protocol; confirm the reply's fields and scales in the protocol document before the run). The threshold
  is: the first reply in which any of the three differs from its mean at rest by more than five times its scatter at
  rest. Record the threshold used. With the bench gains the torque demand lasts only about 2 ms (section 3.2), so
  the speed is the surer field.
- **M1 fine: one sender only.** The loop program sends the extra read; the stopwatch never sends on a bus under
  test. Two nodes sending frames with the same identifier is not allowed on CAN (d: the CAN specification's rule
  as the author knows it; not re-read), and a second sender would push the loop's own frames back. The read is used
  in the step runs only, never in the soak. A cycle that carries an extra read, and the cycle after it, are left out
  of S's statistics: the read and its answer add two frames (0.22–0.27 ms) to a bus that is busy 1.33–1.62 ms of
  every 2 ms. M1 fine also relies on a status read returning fresh values (d).
- **M2: the sensor.** Use the piezo disc, with a 1 megohm resistor across it, a series resistor to the Teensy's pin
  and two diodes clamping the pin to 3.3 V and to ground: a bare disc can swing above 3.3 V and below zero, and the
  Teensy's pins take 3.3 V at most (c: `teensy41`). An accelerometer will do only if its stated bandwidth is 1 kHz
  or more: the evaluation board of a common analog part is filtered to 50 Hz (c: `adxl335`), a lag of about 3 ms.
  The threshold is the first sample more than five times the rest scatter from the rest mean; record it. Check
  M2's own lag once: tap the arm with a wire that closes a contact on a stopwatch input, and compare the two times.
- **M3: the phone.** Check its real slow-motion rate in the file's details. Put the light beside the pointer, at the
  same height in the picture: a phone reads the picture row by row, so two things at different heights are caught
  at different times, up to about a frame apart (d).
- **If two methods disagree.** If D by M1 fine and by M2 differ by more than 2 ms, report both and find the cause
  (play in the gears, the sensor's lag, a status read that returns old values) before choosing.

The sums:
- **Send order.** The loop sends its commands in a fixed, written order, the same on both paths, with the drive
  under test addressed last on its bus.
- **H** = the end of the last command frame on the bus − the tick. That is the worst position: the first frame
  ends five frames (0.56–0.68 ms) earlier on a six-drive bus. Log the first frame's H beside it.
- **D by M1 coarse.** The reply to the step command itself is sampled before the torque changes. Count the cycles, c,
  from that cycle to the first reply that shows the change. The torque changed within c cycles of the command, so
  record D = c × the cycle (2 ms at 500 Hz, 4 ms at 250 Hz). It is an upper bound, in whole cycles.
- **D by M1 fine.** Bin each read by the stopwatch's stamp of the end of the read frame, counted from the end of the
  step command's frame, in 0.2 ms bins from 0 to 4 ms (50 reads a bin). D p50 is the bin in which half the reads show
  the change. The "all-changed bin" is the first bin from which every read shows it. Fifty of fifty shows only that
  at least 94 % of the delays are shorter (95 % confidence): report it as "all of 50", not as a p99. To claim a
  p99, send 300 more commands with the read at that bin: 300 of 300 shows at least 99 %.
- **T.** For M1 coarse and for M2, compute T for each command (H + D of that command; by M2, the sensor's threshold
  crossing − the tick), then take its percentiles: with 1,000 commands the p99 is the 990th smallest and the max the
  largest. For M1 fine, T p99 = H p99 + D at its stated level, and T max is left blank.
- **H's figures** in the results table are over the 1,000 step-command cycles. The soak table holds the every-cycle
  figures.
- **S** = the tick at which the loop first holds a drive's reply − the end of that drive's command frame, which is
  when the drive is taken to sample the reply (d).
- **Which T feeds the reading.** T by M2, where it was measured and its lag checked. Otherwise H p99 plus D by M1
  fine, labelled with its level. M1 coarse is the upper bound: use it for k only if nothing finer worked, and say so.
  It cannot return less than k 2, whatever the truth.

## 2. Parts and tools

<!-- latency_bus:bench_parts begin -->
| Part | Catalog id (`designlab/data`) | Price each | Run 1 qty | Run 2 qty | Kind | Note |
|---|---|---|---|---|---|---|
| leg drive, hip pitch | `myactuator_rmd_x6_p20_60_e` | $428.00 | 1 | 2 | (c) |  |
| leg drive, every other leg joint | `myactuator_rmd_x4_p36_36_e_l` | $269.00 | 1 | 10 | (c) |  |
| neck servo | `robotis_xm540_w270_t` | $482.89 | 1 | 1 | (c) |  |
| tail servo | `feetech_sts3215_c018` | $22.00 | 1 | 1 | (c) | a spare is cheap: the thermal test may age one |
| Jetson dev kit (path 2) | `nvidia_orin_nano_super_devkit` | $399.00 | 0 | 0 | (c) | on hand: COMPUTE.md's bench test 1 measures 'the current dev kit' with no purchase |
| USB-CAN adapter (path 2) | `pcan_usb_fd` | $368.00 | 2 | 2 | (c) | one channel each, so two for the two buses that 500 Hz needs; bench tools that this sheet uses to measure the Jetson path (their price is in the totals row) |
| 24 V to 12 V regulator for the neck and tail servos | `pololu_d36v50f12` | $39.95 | 1 | 1 | (c) |  |
| chest ToF sensor (follow-on sheet C) | `st_vl53l5cx_carrier` | $19.95 | 1 | 2 | (c) |  |
| the build's 6S pack (optional: repeat run 2 on a sagging supply) | `lipo_6s_3300` | $107.19 | 0 | 0 | (c) | not needed for the delay itself |
| Teensy 4.1: one runs path 1's loop, the other is the stopwatch on both paths | not in the catalog | $31.50 | 2 | 2 | (c) | SparkFun's listing, seen on 14 Nov 2026 |
| 3.3 V CAN transceiver boards for the Teensys, 1 Mbit/s: three for the loop, three for the stopwatch | not in the catalog | $10.00 | 6 | 6 | (d) | the author's allowance per board; not in the catalog and not priced |
| ROBOTIS U2D2 USB adapter for the neck servo's bus: path 2, and ROBOTIS's set-up tool | not in the catalog | $40.00 | 1 | 1 | (d) | the author's allowance; not in the catalog and not priced |
| Feetech FE-URT-1 USB adapter for the tail servo's bus: path 2, and Feetech's set-up tool | not in the catalog | $20.00 | 1 | 1 | (d) | the author's allowance from retail listings of about $19 seen in a web search; not in the catalog |
| 3.3 V to 5 V level-shifting half-duplex buffer for a Teensy serial port: path 1, one per servo bus | not in the catalog | $10.00 | 2 | 2 | (d) | the author's allowance. Needed: the Teensy's pins take 3.3 V at most and the neck servo's data line is 5 V logic |
| the vendor's USB-CAN adapter for the leg drives' set-up software (which also needs a Windows PC) | not in the catalog | $70.00 | 1 | 1 | (d) | the author's allowance from a retail listing at about 55 pounds seen in a web search; whether the software accepts the PCAN-USB FD instead was not found |
| piezo disc for method M2, with a 1 megohm load resistor, a series resistor and two clamp diodes, on a clamp-on arm | not in the catalog | $10.00 | 1 | 1 | (d) | the author's allowance |
| dial indicator (0.01 mm) on a stand, or a laser pointer, for sheet A | not in the catalog | $30.00 | 1 | 1 | (d) | the author's allowance |
| bus wire (twisted pair), two D-Sub 9 breakouts for the USB-CAN adapters, 120 ohm terminators, an indicator light | not in the catalog | $30.00 | 1 | 1 | (d) | the author's allowance |
| emergency-stop switch in the drive supply, fuse, clamps, lever arms and weights | not in the catalog | $60.00 | 1 | 1 | (d) | the author's allowance |
| tread sample for sheet D: cast polyurethane to the study's specification, or a TPU 85 A print for a fast trial | not in the catalog | $30.00 | 1 | 1 | (d) | the author's allowance; the specification is in round2/refine/MATERIALS.md |
| 24 V bench supply with a current limit, 10 A or more | not in the catalog | not priced | 1 | 1 | (d) | not priced here: a tether supply is part of the cost model's bench lot, which is $550 / 850 / 1,300 for this body (low / target / high: the round-two record's cost.lines.bench) |
| **Total, catalog parts only** |  |  | **$1,997.79** | **$4,866.74** | (c) | of which the 12 leg drives $3,546.00 and the two USB-CAN adapters $736.00 |
| **Total with the Teensys and the allowances** |  |  | **$2,430.79** | **$5,299.74** | (d) | the bench supply is not priced; the dev kit is on hand |
<!-- latency_bus:bench_parts end -->

- **Run 1 or run 2?** Run 1 gives each drive type's delay for under half the cost of run 2 (0.46 of it), with the
  other 10 leg drives' traffic replayed. It is an emulation: run 2, with 12 real drives, is the result that counts
  for the leg drives. Both runs carry two buses on both paths, so both give the 500 Hz reading with all 12 drives'
  traffic.
- **On hand (c).** `COMPUTE.md`'s first bench test measures "the current dev kit" with no purchase, so the Jetson
  Orin Nano Super dev kit is taken as on hand. Nothing else is assumed to be.
- **The USB-CAN adapters and the Teensys are bench tools.** P30 says the bench runs "before any compute purchase".
  This sheet reads that as being about the robot's computer. The two USB-CAN adapters ($736, c) and the two Teensys
  are tools it uses to measure the two paths; only a Jetson-direct layout would use the adapters on the robot.
- **The Raspberry Pi 5 is assumed not to be needed (d).** On the Teensy path the host is outside the servo loop (the
  budget, section 1), so the dev kit plays the busy host for both paths. If the owner wants the baseline's own host
  on the bench, it is one more purchase.
- **The leg drives' set-up software (c, d).** The vendor's tool runs on a Windows PC and wants a USB-CAN adapter of
  its own (c: `myactuator_setup`). Whether it accepts the PCAN-USB FD was not found (d), so the vendor's adapter is
  on the list. The other way is to set identifiers and limits by bus command from the test program.
- **Check on delivery (c).** The catalog notes that the US store page for the hip-pitch drive conflicts with its
  datasheet. Write down the label of the unit that arrives.
- **Tools you probably have:** a laptop, a Windows PC, a multimeter, a phone with slow-motion video, a kitchen scale,
  a luggage scale (sheet D), a ruler, tape, a thermometer.

## 3. Set-up

### 3.1 Before the bench: the test programs

They are test tools, not the robot's firmware, and none is written yet. A later session writes them. Writing them is
not in the time table.
1. **The loop program**, in two builds with the same logic: one for the Teensy (path 1), one for the Jetson's
   real-time core (path 2). Every 2 ms it raises "tick", sends one command frame to each of the 12 leg drives in the
   written order, collects the replies, raises "replies in", and logs. It computes no control law. On request it
   adds a step to one drive's target at a random tick, sends the extra read of M1 fine, returns the drive by a slow
   ramp, and sets the gains.
2. **The stopwatch program** (section 1.2).

For the follow-on sheets, four small programs: a constant-speed runner that logs the torque current (sheet B); a
reader for the ToF sensor with ST's driver and the stop rule (sheet C); a logger of the tail servo's temperature and
current that cuts its torque at a set temperature (sheet E); and a step-and-hold commander for the neck and tail
servos (sheets A and B).

### 3.2 Common to both paths

1. Clamp each drive's body to the bench (section 9). Leave the output free, with a light pointer arm 0.20 m long for
   M2 and M3.
2. Power the leg drives from the 24 V bench supply, current limit set low (section 9), through the emergency-stop
   switch and a fuse. Feed the 12 V regulator of the neck and tail servos from behind the same emergency stop, so
   that one switch cuts everything.
3. Settings, with each vendor's own tool. Note the firmware version and every setting changed.
   - Every drive: its identifier.
   - Leg drives: the CAN bit rate, 1 Mbit/s, and the drive's own torque or current limit (section 9).
   - Neck servo: its Baud Rate (address 8) from the factory 57,600 bit/s to 1 Mbit/s (value 3; c: `xm540`,
     `xm540_baud`). At the factory rate one write and one read take 10.8–11.8 ms (d: the budget's packet sizes) and
     do not fit the planned 100 Hz.
   - Tail servo: read back its bit rate; a datasheet gives 1 Mbit/s as delivered (c: `sts3215_serial`).
4. Wire each bus as a line, not a star, with a 120 ohm terminator at each end. Two buses: 6 drives on each. In run 1
   put both real leg drives on bus 1.
5. Wire the stopwatch: a transceiver on each bus, the "tick" and "replies in" lines, a common ground.
6. **Gains for the delay runs.** Use the drive's own PD mode (the vendor's "motion mode"; d) with the model's kp and
   the bench kv of the table below, and record what the drive accepted. The model's kv values are sized for the
   leg's inertia: on a bare shaft the hip and knee values are 3 to 9 times critical damping, and the shaft would
   creep to its target over tens of milliseconds instead of stepping. If a drive offers only a position mode, set
   its speed and acceleration limits to their maximum and record them: a planned motion would lengthen D.
7. **Reply turnaround, once per leg-drive type.** With that drive the only one addressed, log for a minute the gap
   from the end of its command frame to the start of its reply: p50 and max. The budget's exchange times assume it
   is shorter than the rest of the command burst (0.56–0.68 ms with six drives on a bus; d); it was not found in the
   vendor's documents.

<!-- latency_bus:step_gains begin -->
| Leg drive | kp for the delay steps, N·m/rad (c: the model's) | Rotor inertia at the output, kg·m² (d: the catalog's estimate) | Bench kv = 2 √(kp × that inertia), N·m·s/rad (d) | The model's kv for this drive type, N·m·s/rad (c): sized for the leg | Creep time constant kv ÷ kp if the model's kv were set on the bare shaft, ms (d) | Torque the 0.01 rad step asks for at first, N·m (c: kp × the step) | Time constant with the bench kv, ms (d): the torque demand lasts about this long | Peak speed of the step, rad/s (d) | Travel of a 0.20 m pointer's tip in the first 4.2 ms, mm (d) |
|---|---|---|---|---|---|---|---|---|---|
| `myactuator_rmd_x6_p20_60_e` | 1140.0 | 0.00430 | **4.4** | 30.8 (6.9 × the bench kv) | 27 | 11.4 | 1.9 | 1.9 | 1.3 |
| `myactuator_rmd_x4_p36_36_e_l` | 598.5 | 0.00324 | **2.8** | 7.6–24.5 (2.7–8.8 × the bench kv) | 13–41 | 6.0 | 2.3 | 1.6 | 1.1 |
<!-- latency_bus:step_gains end -->

All (d) but the model's own gains: an ideal PD loop on the catalog's estimate of the rotor's inertia, with the
pointer's own inertia left out. If the step overshoots, raise kv until it does not, and record it.

### 3.3 Path 1: the Teensy's CAN loop

1. The loop Teensy with one transceiver per bus. It is the only sender on the leg buses, the extra reads of M1 fine
   included.
2. The host (the Jetson dev kit) connects to the loop Teensy by USB and sends a velocity request 50 times a second.
   It is outside the servo loop.
3. Conditions: host idle; host loaded (the load of section 3.4, step 5).
4. Then repeat with three buses, 4 drives on each: the budget expects two buses to overrun the 2 ms cycle in the worst
   case and three to cure it (d).
5. The neck and tail servos' buses go to two of the loop Teensy's serial ports **through the level-shifting
   buffers. Never wire a servo's data line straight to a Teensy pin:** the Teensy's pins take 3.3 V at most
   (c: `teensy41`) and the neck servo's line is 5 V logic (c: `dxl_ttl_level`). The tail servo's line is listed at a
   high level of 2–5 V (c: `sts3215_serial`): measure its idle voltage with the multimeter before trusting it
   anywhere without a buffer.

### 3.4 Path 2: an isolated real-time core on the Jetson, with USB-CAN

Tool and setting names below are the usual Linux ones (d): confirm each on the board's own software release.
1. Install the real-time (PREEMPT_RT) kernel for the dev kit's software release. Keep one processor core free of
   other work with the kernel's boot settings (`isolcpus`, `nohz_full` and `rcu_nocbs` for that core). Run the loop
   program on it at real-time priority, with its memory locked. Write down the priority and the core of the USB host
   controller's interrupt thread and of the CAN receive work: run first as delivered, then with both moved to the
   isolated core, and say which a reading belongs to.
2. Before any drive is connected, run the wake-up latency test `cyclictest` on that core for 10 minutes idle and 10
   minutes loaded. Write down its p99 and max. Third-party tests on Orin modules report about 0.01–0.2 ms at worst
   (c: `rt_jetson`); no published figure for this board with the GPU loaded was found.
3. Connect the two USB-CAN adapters (`pcan_usb_fd`) directly to the dev kit, not through a hub. One adapter carries
   one bus. Bring each up as a CAN port at 1 Mbit/s (`ip link set can0 up type can bitrate 1000000`, then `can1`).
4. Wire two of the dev kit's header pins to the stopwatch as "tick" and "replies in".
5. Conditions: idle; loaded. **Loaded means all of these at once,** after 10 minutes of warm-up:
   - a named GPU load, the perception pipeline if it runs, held at 90 % or more on the board's own readout
     (`tegrastats`);
   - the five other processor cores busy;
   - continuous USB 3 traffic on the same board: the head camera streaming if it is on hand, otherwise a USB disk
     copy. The USB-CAN adapters share the board's USB with the camera on the robot (`COMPUTE.md`).

   Log the board's readout beside the run. No power mode is chosen (P30): run the 25 W mode, and the 15 W mode if
   time allows, and say which.
6. One bus, for the record: with one adapter and all 12 drives on one bus, 500 Hz does not fit (the budget, section
   2: 133–162 % load). Note what the bus does at 500 Hz, then run 250 Hz, labelled as a different rate. A 250 Hz
   reading cannot be mapped onto k (section 6).

## 4. Procedure

### 4.1 Leg drives: RMD-X6-P20-60-E (hip pitch) and RMD-X4-P36-36-E-L

For each leg-drive row of the results table (section 5):
1. Start the loop at the row's target rate with all 12 drives' traffic on the bus. Let it run a minute. Note the
   achieved cycle: the tick-to-tick interval's p50 and max.
2. **Steps.** Command 1,000 steps of 0.01 rad in the target angle, **always in the same direction**, one every quarter
   second at a random tick. After each step return by a slow ramp to 0.02 rad behind the start, then come forward
   slowly to the start, so that the gear already rests on its driving flank when the next step comes (d). A step
   that reverses the torque has to cross the gear's play first (at most 15 arcmin, 0.0044 rad, on the X6; c:
   catalog), and the output then moves later than the torque changes. At the model's kp the step asks at first for
   the drive's sustained planning torque: 6.0 N·m on the X4, 11.4 N·m on the X6 (c: `plant.json`).
3. Read D by M1 coarse and by M1 fine from the same 1,000 commands. Then run the 300-command check of M1 fine on the
   row the reading will use.
4. On the loaded row of each path and drive type, read T by M2 as well.
5. Film ten steps with M3 as a sanity check. With the bench gains the pointer's tip travels about 1 mm in the first
   video frame after the torque changes (table above, d). If T is under 4 ms, as the budget estimates (d), that
   movement shows in the frame after the light's, or the one after that. A laser pointer taped to the arm, shining on
   a wall 2 m away, shows ten times the travel.
6. Fill in the row.

**Soak, once per row of the soak table** (not once per drive type). Take the levers and pointer arms off and check
that the drives' own torque limits are set; the soak may then be left alone. Log 30 minutes of plain running, no steps
and no extra reads. The stopwatch records H and S on every cycle (900,000 cycles at 500 Hz). This catches rare late
cycles that 1,000 commands would miss.

### 4.2 Neck servo: ROBOTIS XM540-W270

1. Its own serial bus at 1 Mbit/s (set in section 3.2): on path 1 a serial port of the loop Teensy through the
   level-shifting buffer, on path 2 the U2D2 adapter. On path 2, set the adapter's latency timer to 1 ms; its
   default is 16 ms (c: `u2d2`).
2. Note the servo's reply delay setting: 500 µs by default, settable to 0 (c: `xm540`). Run with 0 and say so.
3. Send its target and read its position, speed and current every fifth leg cycle (100 Hz), as the budget plans, with
   the leg loop running beside it at 500 Hz.
4. 1,000 steps of 0.05 rad, always in the same direction with the slow return of section 4.1, with any motion
   profile in the servo switched off. Read T by M2 (the arm's sensor), and by M1 coarse from its own replies (10 ms
   resolution at 100 Hz).
5. Its reading is assumed (d) not to set k: the neck does not balance the robot. But the simulation delays the neck
   and tail targets by the same k as the legs', and no run has unequal delays. If this reading is above the legs'
   simulated delay, report it as its own finding.

### 4.3 Tail servo: Feetech STS3215

1. Its own serial bus at 1 Mbit/s: on path 1 another serial port through the second buffer, on path 2 its own USB
   adapter.
2. Read and write down its reply delay setting and bit rate as delivered: two write-ups disagree on the reply
   delay's default (d).
3. As section 4.2, steps 3 and 4.
4. As section 4.2, step 5. The tail's timing also sets its phase against the stride (look feature F9), which
   changes the twisting load under the stance foot: a late tail is a finding of its own.

## 5. Results (to fill in)

One row per drive type, path, bus count and load condition. All times in ms. The leg-drive rows are filled in once
in run 1 and again in run 2: use a copy of the table for each. "D p99" holds, for M1 fine, the all-changed bin, and
"D method and level" says which method and, for M1 fine, "all of 50" or "300 of 300".
"Reading" = T p99 + (S p99 − 2 ms, if positive). "k +" is the k of the reading plus the budget's upper estimate of
the compute step (section 6).

<!-- latency_bus:results begin -->
| Drive | Path | Buses | Load | Target rate, Hz | Run | Achieved cycle: p50 / max | N | H p50 | H p99 | H max | D p50 | D p99 | D method and level | T p50 | T p99 | T max | T by M2: p50 / p99 / max | S p99 | Reading | k | k + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X6-P20-60-E | 1: Teensy | 2 | idle | 500 | 1 and 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| X6-P20-60-E | 1: Teensy | 2 | loaded | 500 | 1 and 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| X6-P20-60-E | 1: Teensy | 3 | loaded | 500 | 1 and 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| X6-P20-60-E | 2: Jetson | 2 | idle | 500 | 1 and 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| X6-P20-60-E | 2: Jetson | 2 | loaded | 500 | 1 and 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| X6-P20-60-E | 2: Jetson | 1 | loaded | 250 | 1 and 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | n/a | n/a |
| X4-P36-36-E-L | 1: Teensy | 2 | idle | 500 | 1 and 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| X4-P36-36-E-L | 1: Teensy | 2 | loaded | 500 | 1 and 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| X4-P36-36-E-L | 1: Teensy | 3 | loaded | 500 | 1 and 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| X4-P36-36-E-L | 2: Jetson | 2 | idle | 500 | 1 and 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| X4-P36-36-E-L | 2: Jetson | 2 | loaded | 500 | 1 and 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| X4-P36-36-E-L | 2: Jetson | 1 | loaded | 250 | 1 and 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | n/a | n/a |
| XM540-W270 | 1: Teensy | own | loaded | 100 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | n/a | n/a |
| XM540-W270 | 2: Jetson | own | loaded | 100 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | n/a | n/a |
| STS3215 | 1: Teensy | own | loaded | 100 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | n/a | n/a |
| STS3215 | 2: Jetson | own | loaded | 100 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | n/a | n/a |
<!-- latency_bus:results end -->

Soak logs, 30 minutes each (900,000 cycles at 500 Hz). A late cycle is a tick-to-tick interval above 2.5 ms. The
reply turnaround here is that of the drive addressed last on a bus: the start of its reply − the end of its command.

<!-- latency_bus:soak begin -->
| Path | Buses | Load | Run | H p99, ms | H p99.9, ms | H max, ms | Reply turnaround: p50 / max, ms | Late cycles (tick to tick above 2.5 ms) | Cycles whose replies missed the next tick | Frames lost or resent |
|---|---|---|---|---|---|---|---|---|---|---|
| 1: Teensy | 2 | idle |  |  |  |  |  |  |  |  |
| 1: Teensy | 2 | loaded |  |  |  |  |  |  |  |  |
| 1: Teensy | 3 | loaded |  |  |  |  |  |  |  |  |
| 2: Jetson | 2 | idle |  |  |  |  |  |  |  |  |
| 2: Jetson | 2 | loaded |  |  |  |  |  |  |  |  |
<!-- latency_bus:soak end -->

Also write down: the date, each drive's label and firmware version, the gains each drive accepted, the send order,
each leg-drive type's reply turnaround alone on a bus (section 3.2, step 7), the thresholds used, M2's measured lag,
the phone's frame rate, the supply voltage, the room temperature, the Jetson's software release and power mode, what
the load was and its readout, where the USB and CAN interrupt work ran, and the wake-up latency test's p99 and max.

## 6. How to read your numbers

**The rule.** For each path, take the reading (section 5) of its worst leg-drive type under its worst load
condition. Round it up to a whole 2 ms step: that is k. The bar is re-run at that k. Delays are never read between
steps (P29, P30): a reading of 2.1 ms is run at 4 ms; it is not estimated from results at other delays.

**The missing compute step.** The reading does not hold the walking controller's own compute step (section 1.1). So
give a second figure beside k: the k of the reading plus the budget's upper estimate of that step, 0.50 ms on the
Teensy and 0.20 ms on the Jetson (d). If the two k differ, the bar is re-run at both. When the firmware exists, its
measured step time replaces the estimate and k is read again.

<!-- latency_bus:delay_map begin -->
| Your reading (tick to torque at its p99, plus any feedback age beyond 2 ms) | Re-run the bar at | Model, R2/43: up · every limit (b) | Model, R3/42: up · every limit (b) | What it would mean, from what the study has shown |
|---|---|---|---|---|
| above 0 and at most 2 ms | 2 ms (k 1) | not run | not run | Not simulated: no delay search on any body ran 2 ms. Run the bar at 2 ms before reading anything; its neighbours, 0 and 4 ms, are not a reading of it. |
| above 2 and at most 4 ms | 4 ms (k 2) | 3 of 3 · 3 of 3 | 3 of 3 · 3 of 3 | In the model the 75 cm nominal walk met every limit here. The 0.40 m/s bar has no record at this delay: re-run it at 4 ms. |
| above 4 and at most 6 ms | 6 ms (k 3) | 3 of 3 · 3 of 3 | 3 of 3 · 3 of 3 | The largest delay at which the model's 75 cm nominal walk met every limit. Re-run the bar at 6 ms. |
| above 6 and at most 8 ms | 8 ms (k 4) | 3 of 3 · 0 of 3 | 3 of 3 · 0 of 3 | The model's walk stays up and misses a limit. The fitted tracker is at its edge; re-run the bar at 8 ms. |
| above 8 and at most 10 ms | 10 ms (k 5) | 0 of 3 · 0 of 3 | 0 of 3 · 0 of 3 | The model's walk falls in 3 of 3 episodes under both controllers. This is today's judged delay (P29). A reading here would count against the fitted tracker as it stands, not against every controller (Stage 1b, note N8). |
| above 10 and at most 18 ms | the whole 2 ms step above it (k 6–9) | not run | not run | Never simulated on any body (12-18 ms). No reading until the bar is run at the whole step above the measurement. |
| above 18 and at most 20 ms | 20 ms (k 10) | 3 of 3 · 0 of 3 | 3 of 3 · 0 of 3 | The model's walk stayed up without meeting every limit. The response is not monotonic and is unexplained, so this says nothing about 12-18 ms. |
| above 20 ms | nothing: outside the simulation's range (k stops at 10) | not run | not run | Cannot be mapped onto a simulated delay. Report it as measured; the delay range of the model would have to be widened first. |
<!-- latency_bus:delay_map end -->

The model columns are simulation results (b): Stage 1's ledger of the 75 cm nominal walk (0.243 m/s) under two
controllers, 3 episodes per delay (`stage1/runs/2026-11-08/STAGE1_PROBE.md`, the correction note at its top). R2/43 is
the balance tracker on the robot's own sensors, in its fitted version 43; R3/42 is the same tracker with a learned
add-on. They are margin readings, not 20-episode verdicts, and they are not the 0.40 m/s bar. In full:

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

**Why this mapping.** In the simulation every joint target reaches the servos k steps of 2 ms after the controller
sets it. That is added to a loop that already lags: its sensors are one step (2 ms) old, and its balance feedback is
computed 100 times a second and ramped in over the following steps. Behind the delay the simulated servo is an ideal
PD loop. The file and line of each statement is in the budget, section 4. So your T corresponds to 2 k ms, as long as
the loop ran at 500 Hz, the feedback was no more than 2 ms old (hence the S term) and the drive ran its own PD loop.

**Three more readings.**
- **Path 1 against path 2.** If both map onto the same k, the delay does not separate the architectures. What else
  separates them is for the owner. By design the microcontroller path ends a Linux crash in a controlled stop, once
  its firmware exists; that firmware is unwritten and untested (`COMPUTE.md`; the budget, section 6). If the paths
  map onto different k, that difference is the first measured evidence for or against moving the walking loop.
- **Idle against loaded.** The reading for a path is always that of its worst load condition: the robot runs
  loaded. The idle figure is reported beside it, to show what the load costs.
- **A max far above the p99.** A constant k does not represent rare late cycles. If the max is more than one step
  above the p99, or the soak log shows late cycles, report it as its own finding: the simulation has no such test yet.

**What a reading does not settle.**
- A reading above 8 ms and at most 10 ms maps onto the delay at which the model's nominal walk fell in 3 of 3
  episodes under both controllers (b). That counts against the fitted tracker as it stands, not against every
  controller.
- A reading above 10 ms maps onto delays never simulated (12–18 ms), or onto 20 ms, where the model's walk stayed up
  without meeting every limit, or beyond the model's range. It has no reading until the bar is run there.
- A reading at or below 8 ms still leaves grip and the controller to check (Stage 1b, note N8).
- A 250 Hz reading cannot be mapped until the simulation has a 250 Hz variant.
- Every model figure here is a 3-episode margin reading of the 0.243 m/s walk, not the 0.40 m/s bar.

## 7. Follow-on sheets

Shorter, and independent of each other. Each replaces an assumption of the simulation that nobody here has measured.

### A. Servo stiffness

**What the model assumes (c: `plant.json`; the rule is an assumption, d).** Each drive holds its joint like a spring
that reaches the drive's sustained planning torque at 0.01 rad of error, critically damped on the leg's inertia. Each
ankle motor reaches it at 0.02 rad. The rule is in `designlab/stage1/plant.py` and `designlab/carpet.py`; the
budget's section 4 gives the lines.

<!-- latency_bus:bench_gains begin -->
| Drive (catalog id) | Joint | Count | Model kp, N·m/rad (c) | Model kv, N·m·s/rad (c) | Sustained planning torque, N·m (c) | Mass on a 0.20 m lever for that torque, kg | Deflection the model expects there, rad | 500 ÷ model kp (d) | 5 ÷ model kv (d) | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| `robotis_xm540_w270_t` | neck | 1 | 177.02 | 4.16 | 1.77 | 0.90 | 0.010 | n/a | n/a |  |
| `feetech_sts3215_c018` | tail | 1 | 49.13 | 2.72 | 0.49 | 0.25 | 0.010 | n/a | n/a |  |
| `myactuator_rmd_x4_p36_36_e_l` | hip roll | 2 | 598.50 | 24.48 | 5.99 | 3.05 | 0.010 | 0.83 | 0.20 |  |
| `myactuator_rmd_x6_p20_60_e` | hip pitch | 2 | 1140.00 | 30.78 | 11.40 | 5.81 | 0.010 | 0.44 | 0.16 |  |
| `myactuator_rmd_x4_p36_36_e_l` | hip yaw | 2 | 598.50 | 7.59 | 5.99 | 3.05 | 0.010 | 0.83 | 0.66 | the thigh twist |
| `myactuator_rmd_x4_p36_36_e_l` | knee | 2 | 598.50 | 11.51 | 5.99 | 3.05 | 0.010 | 0.83 | 0.43 |  |
| `myactuator_rmd_x4_p36_36_e_l` | ankle a motor | 2 | 299.25 | 2.08 | 5.99 | 3.05 | 0.020 | 1.67 | 2.40 | each ankle's two motors carry half each: 598.5 N m/rad on pitch and on roll |
| `myactuator_rmd_x4_p36_36_e_l` | ankle b motor | 2 | 299.25 | 2.08 | 5.99 | 3.05 | 0.020 | 1.67 | 2.40 | each ankle's two motors carry half each: 598.5 N m/rad on pitch and on roll |
<!-- latency_bus:bench_gains end -->

The last two numeric columns compare the model's gains with the largest gains in a sibling model's manual (KP up to
500, KD up to 5; d: its units, and whether the 75 cm drives share it, are not confirmed). Below 1 the drive's own loop
could not reach the model's gain, if the units were the model's. This is a question for the vendor's protocol
document and for step 1 below, not a finding.

**What the simulation has shown (b).** With every drive's stiffness scaled together, the 75 cm nominal walk met every
limit down to 0.71 of the model's stiffness and stayed up down to 0.55
(`stage1/runs/2026-11-08/75-M-H-10/sweeps/R2/43/kp.json`, 3 episodes per value). That search ran with no servo
delay, kept kv unchanged and scaled the feedforward torque with kp.

**Measure, per drive type:**
1. Leg drives: set the model's kp and the hip kv from the table (X4: the hip-roll row; X6: the hip-pitch row). With
   the lever and mass of step 2 the drive sees an inertia of the leg's order (0.12 kg·m² on the X4, 0.23 on the X6;
   d), so these two sets are at or a little above critical damping here; the X4's other sets would ring.
   Write down the largest values the drive accepts and their units. Neck and tail servos: their gains are register
   values without physical units, so their stiffness is found by this measurement, not set. Note the register
   values.
2. Fit the 0.20 m lever, horizontal. Take zero with the lever fitted and no mass. Hang masses in four steps up to the
   table's mass. At each step read the deflection twice:
   - from the drive's own encoder. The leg drives' encoders have 17 bits or more (c: catalog), far finer than
     needed. The neck and tail servos' have 12 bits (c for the tail: catalog; d for the neck): 0.01 rad is 6.5
     counts, good to about 15 %;
   - at the lever, with a dial indicator at the tip (0.01 mm; 2.0 mm of tip travel is 0.01 rad) or a laser pointer
     taped to the lever and a mark on a wall 2 m away (20 mm per 0.01 rad). A ruler at the tip reads 0.5 mm in
     2 mm and is not enough. This second reading catches play and flex in the gears and the mount, which the encoder
     may not see.
3. Stiffness = torque ÷ deflection, in N·m/rad. Torque = mass × 9.81 × 0.20.
4. With the mass on, command a 0.01 rad step and note any overshoot. The model's servo has none. Judge damping again
   once a leg exists.

| Drive | Largest kp accepted (units) | Largest kv accepted (units) | Stiffness by encoder, N·m/rad | Stiffness at the lever, N·m/rad | ÷ model kp | Overshoot of a 0.01 rad step, % |
|---|---|---|---|---|---|---|
| X6-P20-60-E | | | | | | |
| X4-P36-36-E-L | | | | | | |
| XM540-W270 | | | | | | |
| STS3215 | | | | | | |

**Reading it.** At or above 0.71 of the model's stiffness on every drive, the reading is inside the range where the
simulated walk met every limit, at no servo delay (b). Three limits on that reading:
- It holds only if the drive also reaches the model's kv. The simulation never lowered kv, and the table shows that
  the sibling manual's damping cap, if it applied, would be the larger shortfall (0.16–0.66 of the model's kv on the
  hip and knee drives, against 0.44 of the hip pitch's kp). A drive that reaches kp but not kv needs its own run.
- Stiffness and delay together were never simulated.
- One soft drive is not the case the simulation ran (it scaled every drive together), so it too needs its own run.

Below 0.71, the simulation is re-run with the measured stiffness.

### B. Joint friction and damping (decision-log row A18)

**What the model assumes (d).** Every joint, legs included, has a damping of 0.2 N·m·s/rad and a friction loss of
0.15 N·m: the generator's defaults (`designlab/mjcf.py`, line 120), unsourced. They dominate the tail servo's load
(`LESSONS_LEARNED.md` section 2.4). The vendor gives a backdrive torque of 0.2 N·m for the X4-36-L and 1.6 N·m for
the X6-60 (c: catalog notes), about ten times the model's friction on the hip pitch.

**Measure, per drive type:**
1. **Breakaway, unpowered.** Add small masses until the shaft starts to turn. Both directions, five times each. Use a
   balanced two-sided arm, or a pulley and a string, so that the lever's own weight does not count. With a
   one-sided lever add its own moment: its mass × 9.81 × the distance to its centre. Torque = mass × 9.81 × the
   lever length.
2. **Running, powered.** Run the free shaft at the constant speeds of the table below, in both directions. Log the
   torque current and convert with the catalog's torque constant: 1.8 N·m/A on the X4-36-L, 2.1 N·m/A on the X6-60,
   both per ampere rms (c). Check in the protocol whether the drive reports amperes rms or peak: the wrong one is a
   1.4-fold error. Better, calibrate the current reading once against a known torque, with sheet A's lever and a
   mass, and use that factor. Fit a line: torque = friction + damping × speed.
3. For the neck and tail servos use step 1, and step 2 with their own current or load reading if it is fine enough.
4. Repeat on the assembled joint once a leg exists: bearings, linkages and cables add their own.

<!-- latency_bus:friction begin -->
| Drive | Constant speeds to run, rad/s | The model's usable speed, rad/s (c: `plant.json`) | No-load speed on the build's supply, rad/s (c: `plant.json`) |
|---|---|---|---|
| X6-P20-60-E | 0.5, 1, 2, 4 | 8.01 | 9.22 at 24 V |
| X4-P36-36-E-L | 0.5, 1, 2, 4 | 8.69 | 11.62 at 24 V |
| XM540-W270 | 0.25, 0.5, 1, 1.5 | 1.88 | 3.14 at 12 V |
| STS3215 | 0.5, 1, 1.5, 2.5 | 2.83 | 4.71 at 12 V |
<!-- latency_bus:friction end -->

| Drive | Breakaway, N·m (+ / −) | Friction from the fit, N·m | Damping from the fit, N·m·s/rad | Model: 0.15 N·m and 0.2 N·m·s/rad |
|---|---|---|---|---|
| X6-P20-60-E | | | | |
| X4-P36-36-E-L | | | | |
| XM540-W270 | | | | |
| STS3215 | | | | |

**Reading it.** The tail's load in the Stage 0 screen depends on these two numbers more than on any part: with them,
look feature F9's tail phase cuts the tail servo's sustained load by 40–61 %; without them by 11–40 % (a:
`stage2/look/SCREEN.md`). The measured values go into the model before that cut is relied on.

### C. The chest ToF stop

**What the design says (c).** The stop runs on the microcontroller (`COMPUTE.md`). The catalog's sensor
(`st_vl53l5cx_carrier`) ranges at up to 15 Hz with 8 × 8 zones or 60 Hz with 4 × 4 zones (`vl53l5cx`): one frame every
66.7 ms or 16.7 ms. An operator stop must be honoured within 0.3 s (`BEHAVIOR_SPEC.md` section 3). No time has been
set for the ToF stop itself; the speed governor (task 10) will need this number.

**Measure:**
1. Mount the sensor as on the robot: 45° down (F4). Have the Teensy read it and switch on a light when its stop rule
   fires (say, any zone nearer than a set distance in two frames running; write the rule down).
2. Mark a line where a card first enters the zone the stop rule watches. Swing the card into the beam across that
   line. Film the card and the light in slow motion, the light beside the line (4.2 ms a frame at 240 frames a
   second, fine at these time scales). Count frames from the card crossing the line to the light.
3. 100 trials in each mode. Report the median, the largest and the second-largest of the 100. Call a figure a p99
   only with 300 or more trials: with 100 it rests on two readings.
4. Add the leg drives' delay from section 5 for the time from the stop flag to the first braking torque.

| Mode | Stop rule used | Card to stop flag: p50, ms | Second-largest, ms | Largest, ms | Distance walked in the largest at 0.40 m/s, mm |
|---|---|---|---|---|---|
| 8 × 8 zones, 15 Hz | | | | | |
| 4 × 4 zones, 60 Hz | | | | | |

**Reading it.** No limit exists yet for this time: task 10, the speed governor, sets it. Until then the figure is
recorded, with the distance it means: at 0.40 m/s the robot walks 4 mm in every 10 ms.

### D. Tread grip on your floors (P21)

**What the model assumes.** The bar runs at friction 0.9 and at a slippery 0.54 (c: `STAGE2_PLAN.md` section 5). With
the stance foot's twisting moment counted, the 75 cm body needs 0.36 to walk at the nominal pace, 0.41 in the target
turn and 0.56 at the brisk pace (a: F22). The look screen prices today's walk at three speeds (a):

<!-- latency_bus:grip begin -->
| Prescribed walk (today's walk and tail) | Peak twisting moment under the stance pad, N·m | Friction it needs with that moment | Kind |
|---|---|---|---|
| the nominal walk (Froude 0.012) | 2.5 | 0.36 | (a) |
| 0.40 m/s (the agility bar's speed) | 4.6 | 0.63 | (a) |
| the animal pace (Froude 0.05) | 5.8 | 0.79 | (a) |
<!-- latency_bus:grip end -->

The sourced estimates for a soft polyurethane tread are 0.42–0.8 on vinyl and laminate and 0.70–0.85 on carpet (c,
as estimates: F22); none is measured on your floors.

**Measure, on each floor of P26 (hardwood, vinyl, a museum-type floor, thin compact carpet):**
1. Make a tread sample to the study's specification (c: `round2/refine/MATERIALS.md` section 5): cast polyether
   polyurethane, 50–60 Shore A on a 70–80 A core, with round studs 1.0–1.5 mm high; a TPU 85 A print will do for a
   first trial. Glue it under a flat block and load it to the pads' pressure in single support, 41 kPa (c: the same
   section). For example a sample 40 mm × 40 mm under 6.7 kg (d: arithmetic).
2. **Slide.** Pull horizontally with a luggage scale, slowly. Note the peak before it moves and the steady pull while
   it slides. Friction = pull ÷ weight. Ten pulls, along and across the floor's grain, on a clean floor and a dusty
   one.
3. **Spin.** This needs a whole foot's pads under the robot's whole weight, 14.9 kg (c: `plant.json`), so it waits
   for a printed foot. Twist the loaded foot about the vertical with a lever and the scale. Note the moment at
   which it lets go (pull × lever length, in N·m). Stance-foot spin is the grip limit that binds (F21).

| Floor | Clean: peak / sliding | Dusty: peak / sliding | Lowest peak | Spin moment, N·m | Against 0.36 / 0.41 / 0.56, the table above and the 0.54 test grip |
|---|---|---|---|---|---|
| Hardwood | | | | | |
| Vinyl | | | | | |
| Museum-type floor | | | | | |
| Thin carpet | | | | | |

**Reading it.** On a floor whose lowest peak is above 0.56, the three needs of F22 are covered (a); above 0.63, so is
today's walk at the bar's 0.40 m/s (a, the table above). Below 0.36 the nominal walk is not covered. In between, say
which needs are covered and which are not. The spin moment is read against the table's peak twisting moments: a foot
that lets go below a walk's figure does not hold that walk. These needs come from prescribed walks; no controller
has been run against a measured floor.

### E. The tail servo's heat (decision-log row A1)

**What the model assumes.** The STS3215's sustained planning torque is 0.49 N·m. It is 0.167 × the vendor's 12 V
stall torque of 2.94 N·m: PR #662's rule for box servos (d), which the catalog writes as 0.57 × a derived "rated"
torque of 0.86 N·m (A1). The vendor's own rated torque is 0.98 N·m, 10 kg·cm at 0.9 A, with no thermal definition
(c: catalog). In closed loop the 75 cm speed walks load the servo above the planning torque (b):

<!-- latency_bus:thermal begin -->
| Hold | Torque, N·m | × the planning torque | × the vendor's rated 0.98 N·m | Mass on a 0.10 m lever, kg | Kind |
|---|---|---|---|---|---|
| the model's sustained planning torque: 0.167 x the 12 V stall torque (PR #662's rule for box servos; A1) | 0.49 | 1.00 | 0.50 | 0.50 | (c) |
| the 75 cm brisk walk's closed-loop load, upper figure (Stage 1, R2; A1's update) | 0.52 | 1.06 | 0.53 | 0.53 | (b) |
| the 75 cm animal-pace walk's closed-loop load, upper figure (Stage 1, R2; A1's update) | 0.64 | 1.30 | 0.65 | 0.65 | (b) |
<!-- latency_bus:thermal end -->

**Measure:**
1. Clamp the servo. Fit a 0.10 m lever, horizontal, with the table's mass.
2. Before the first hold, read the servo's own temperature-limit setting and write it down. Stop any hold 5 °C below
   it, or at once if the servo's own protection trips.
3. Hold for 20 minutes, the top of P4's 15–20 minute runtime. Every 10 s log the servo's own temperature and
   current readings and the room temperature. Let the servo cool to room temperature between holds.
4. Repeat each hold while swinging ±0.2 rad at two swings a second: closer to the walk, and it adds the gear's
   friction heat. Six holds in all. Stay with the bench during each hold (section 9).

| Hold, N·m | Static: temperature at 20 min, °C | Static: rise over the last 5 min, °C | Swinging: temperature at 20 min, °C | Swinging: rise over the last 5 min, °C | Tripped or sagged? |
|---|---|---|---|---|---|
| 0.49 | | | | | |
| 0.52 | | | | | |
| 0.64 | | | | | |

**Reading it.** If the temperature rises by less than 1 °C over the last 5 minutes at 0.64 N·m, this sample, in open
air, does not contradict A1 at the loads the simulation has shown (b). That is not yet "safe": on the robot the servo
sits at the tail base under a covering, which is warmer, and it still needs its own test there. If the temperature
is still rising at 0.49 N·m, A1 is optimistic for this servo, and the tail's load has to come down (look feature F9's
phase, sheet B's friction) or the servo has to change. Neither is decided here.

## 8. Bench time

<!-- latency_bus:time begin -->
| Part | What | Bench time (d) |
|---|---|---|
| Run 1 | Unpack, set identifiers and bit rates with the vendors' tools, set each drive's own torque limit, clamp and wire one drive of each type | 1.5–2.5 h |
| Run 1 | Path 1 set-up: the loop Teensy and the stopwatch Teensy (which replays the absent drives), three transceivers each, the two level-shifting buffers | 1.0–2.0 h |
| Run 1 | Path 2 set-up: real-time kernel, one isolated core, two USB-CAN adapters, the tick lines to the stopwatch, a wake-up latency check | 2.0–4.0 h |
| Run 1 | Delay runs: the 16 rows of the results table x 1,000 commands (about 10 min a row with its set-up), method M2 on one row per path and drive type, and the 300-command check of M1 fine | 3.5–5.0 h |
| Run 1 | Soak logs: the 5 rows of the soak table x 30 min, levers off; may be left alone once the drives' torque limits are set | 2.5–3.0 h |
| Run 2 | Wire all 12 leg drives on two buses (then three), with terminators | 2.0–4.0 h |
| Run 2 | Repeat the 12 leg-drive rows and the 5 soak logs with 12 real drives | 4.5–6.0 h |
| Follow-on | A. Servo stiffness: 4 drive types | 2.0–3.0 h |
| Follow-on | B. Joint friction and damping: 4 drive types | 2.0–3.0 h |
| Follow-on | C. Chest ToF stop timing: 2 modes x 100 trials on video | 1.5–2.5 h |
| Follow-on | D. Tread grip: 4 floors x 10 pulls, clean and dusty, and the spin test | 2.0–3.0 h |
| Follow-on | E. Tail servo thermal: 6 holds of 20 min (3 static, 3 swinging) with cooling between, attended | 3.5–5.0 h |
| **Run 1, total** |  | **10.5–16.5 h** |
| **Run 2, total** |  | **6.5–10.0 h** |
| **Follow-on, total** |  | **11.0–16.5 h** |
<!-- latency_bus:time end -->

All (d): the author's estimates for one person, with the test programs already written and tested. Writing those
programs is not bench time and is not estimated here.

## 9. Safety, for powered drives on a bench

- **Clamp the drive's body, not its output.** The hip-pitch drive's vendor peak is 60 N·m and the X4's 30 N·m (c:
  catalog). At the end of a 0.20 m lever 60 N·m is 300 N, enough to throw the drive off the bench or break a finger.
- **An emergency stop in the supply line, within reach, that cuts power** to the leg drives and to the 12 V
  regulator alike. Never rely on a software stop. Test it before the first run.
- **Set each leg drive's own torque or current limit before a lever is fitted.** Low for the first power-up, then
  the sustained planning torque: 6.0 N·m on the X4, 11.4 N·m on the X6 (c: `plant.json`). The bench supply's current
  limit protects the wiring; it does not limit torque. A stalled drive turns 2 A at 24 V into about 9 A in its
  windings: about 16 N·m on the X4 and 19 N·m on the X6 (d: arithmetic from the catalog's resistances and torque
  constants), which is 80–93 N at the end of a 0.20 m lever.
- **A current limit and a fuse in the supply as well.** Start with the bench supply limited to 2 A and raise it only
  as a test needs (d). A supply in current limit sags and can reset the drives, and it may trip when a drive brakes;
  follow the vendor's guidance on the supply.
- **Keep out of the lever's arc.** Hands, hair, sleeves and cables. Put a soft end stop on both sides of the arc.
  Secure hanging masses so that they cannot swing off, and keep feet clear of them.
- **First power-up of each drive with the output free,** at low gains and a small step, with the vendor's tool.
- **Find out what a drive does when commands stop,** on purpose and early: pull the bus plug with the output free.
  Write it down; the budget's failure table needs it.
- **Never wire a servo's data line straight to a Teensy pin** (section 3.3): 5 V on a 3.3 V pin can destroy the
  Teensy that runs the loop.
- **The neck and tail servos share one regulator.** Their stall currents at 12 V are 4.4 A and 2.7 A (c: catalog),
  7.1 A together, against the regulator's 4.5 A (c). Limit both servos' current in their own settings.
- **Heat.** A servo in sheet E gets hot. Stay with a held load; do not leave it.
- **Before a soak is left alone:** levers and pointer arms off, the drives' torque limits set, nothing loose near
  the outputs.
- **If the battery pack is used:** on a fire-safe surface, never charged unattended, with its own fuse.
- **Eye protection** whenever a lever or a mass is fitted.

## 10. Decision-log rows this sheet serves, and drafts

Rows already in the log (approved by the owner on 14 Nov with P30): **A17**, the servo delay and stiffness (sections
1–6 and sheet A); **A18**, the joints' damping and friction (sheet B); **A1**'s update, the tail servo first (sheet
E). P21's tread measurement is sheet D.

Drafts for the owner. Nothing is adopted here, and the row numbers are for the lead to assign.

- **Draft wording for A17's "replace with".** "The p99 of the time from the control tick to the change of torque at
  the joint, plus any feedback age beyond 2 ms, per leg-drive type, at 500 Hz with all 12 leg drives' traffic on two
  buses, under the worse of the idle and the loaded condition, on the Teensy's CAN loop and on a Jetson real-time
  core with USB-CAN; rounded up to a whole 2 ms step, at which the bar is re-run. The reading lacks the walking
  controller's compute step until the firmware exists (`stage2/jetson75/BENCH_SHEET.md`)."
- **Draft A row (the bench host).** "On the delay bench the Jetson dev kit stands in for the Raspberry Pi 5 as the
  Teensy path's host, on the assumption that a host outside the servo loop does not change the delay."
- **Draft A row (the ToF stop).** "The chest ToF stop's time from an obstacle entering the beam to the stop flag is
  unmeasured; the sensor's frame takes 16.7 ms (4 × 4 zones) or 66.7 ms (8 × 8 zones)." Replace with: sheet C.
- **Draft A row (bench emulation).** "Run 1 of the delay bench replays 10 of the 12 leg drives' frames; its leg-drive
  readings are provisional until run 2 repeats them with 12 drives."
