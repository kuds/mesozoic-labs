# A speed governor from the chest ToF sensors (75 cm + Jetson track, task 10)

**Changed, 14 Nov 2026.** Decision-log rows P30 and A17 are in the log now. The opening paragraph no longer says that
P30 is "being added today", and A17 is no longer called a draft in four places: "In brief", section 3, section 10's
heading and the `servo_delay` row of the inputs table. No number changed.

14 Nov 2026. Simulation and planning only: nothing here is a hardware measurement, and nothing here is a speed the
robot has shown. Nothing is adopted (P24); the draft rows in section 13 are for the owner. This answers task 10 of
`JETSON_PLEM_REVIEW.md` section 5, "a speed governor, not a speed claim", under the owner's decisions of 14 Nov 2026
(decision-log row P30): the Jetson is studied as a refine case, the real-time walking loop stays on the microcontroller
as the baseline, and the agility bar is judged at 10 ms of servo delay until a bench value exists.

Paths are relative to `hardware/trex_form_study/`. The 75 cm body `75-M-H-10` comes first and the 60 cm body
`60-N-T-10` beside it. P, F, E and A rows are in `DECISION_LOG.md`. S1 is the Stage 1 report
(`stage1/runs/2026-11-08/STAGE1_PROBE.md`).

Words used here:
- **ToF** (time of flight): the two ranging sensors under the chest, each a VL53L5CX on a carrier board (catalog id
  `st_vl53l5cx_carrier`). Each looks 45° down and 12° outward over one foot path (F4, P22) and reports a grid of
  distances, 8x8 or 4x4 zones. **ST** is STMicroelectronics, which makes the sensor.
- **Ahead of the toes**: distance along the floor from the toe tips of the home stance, as in the study's sensor
  screen (`designlab/sensors.py`). This toe line rides with the trunk. **The foot path** is the strip the feet walk
  on, their outer edges plus 0.05 m.
- **Far row**: the top row of a sensor's zones, the one that looks farthest ahead.
- **H**: hip height, 0.502 m on the 75 cm body and 0.405 m on the 60 cm body. **Walking height**: every reference walk
  holds the trunk 0.03 H below the home pose (0.015 / 0.012 m).
- **I2C**: the two-wire bus the sensors are read over. **CAN**: the bus to the leg drives. **IMU**: the motion sensor
  in the trunk.
- **Jetson**: the Jetson Orin Nano Super, the computer under study. **Pi 5**: the Raspberry Pi 5, the computer adopted
  meanwhile (P14). **Host**: whichever of them runs Linux and the planner.
- **PR #668**: a pull request, a proposed change to the repository. This one holds an unverified study of the Jetson
  (`stage2/jetson_plem/`).
- **Governor**: the highest speed at which autonomy could still stop for something that only these sensors see. It is
  a ceiling from this one constraint. The robot's speed is set by the lowest of all its limits.
- **Froude 0.05 walk**: the study's fastest reference walk, the animal-like pace: 0.496 m/s at 75 cm, 0.446 m/s at 60
  cm. The **nominal walk** is 0.243 / 0.218 m/s, the fastest walk shown in closed loop.
- **RMS**: root mean square, the typical size of a quantity that varies. **lux**: a unit of light level; 5000 lux is
  roughly an overcast day outdoors, and room light is a few hundred lux.

Kinds of evidence, as in the study's reviews:
- (a) a prescribed-walk (Stage 0) or static model result of this study;
- (b) a closed-loop simulation result (none is used here);
- (c) a sourced fact: a catalog id, a document of this study, or a vendor document;
- (d) an estimate or assumption, with whose it is and how it was derived.

Every governor speed in this document is kind (a): a static and kinematic model result. Its range comes from (d)
assumptions about timing, so each cell is a range, not one number. A kind in square brackets beside a stop reading is
the kind of that reading, not of the speed. The spec's 1.5 m/s², 1.5 strides and 200 ms, and the bar's "a step or
two", are targets, not values the robot has met.

**Sources.** Every number in the tables is a field of `stage2/jetson75/tof_governor.json`, written by
`stage2/jetson75/tof_governor.py` in about a second from committed records only: each body's sensor screen and
envelope screen (`round2/refine/<case>/review/screening/`), its candidate record and its Stage 1 stop reference
(`stage1/runs/2026-11-08/<body>/transitions.json`). The record holds a sha256 checksum of each, and of the two ToF
entries of the parts catalog. Before computing anything new, the module recomputes the 64 zone centres the screen
recorded for each sensor and agrees with them to 0.17 mm. Tests: `tests/test_stage2_tof_governor.py`.

## In brief, for the owner

- **The answer.** The chest ToF sensors are not what limits walking speed, whichever computer runs the stop. Over
  three ToF modes, three stop paths and six readings of the stop, the lowest governor is 0.57 m/s at 75 cm and 0.53
  m/s at 60 cm (a, on (d) timing). That is above the fastest walk the study has a reference for (0.496 / 0.446 m/s)
  and above the top of the speed ladder (0.50 m/s).
  - By the review's method, with the bar's two-step stop and 8x8 zones, the ToF stop would allow 0.92–1.21 m/s on the
    75 cm body when the microcontroller or the Jetson's real-time core decides it, and 0.77–1.21 m/s when the host's
    planner does (a). At 60 cm: 0.85–1.13 and 0.70–1.13 m/s (a).
  - A longer stop allows less. Today's stop builder at its worst phase (2.4 steps) gives 0.80–1.02 and 0.68–1.02 m/s
    at 75 cm. The spec's own stop limit, 1.5 strides, gives 0.67–0.83 and 0.58–0.83 m/s (0.62–0.78 and 0.54–0.78 m/s
    at 60 cm) (a).
  - A harder stop allows more, up to 1.68 m/s (a) with the hardest stop the spec allows, 1.5 m/s². A one-step stop is
    capped by that same limit, so it allows no more.
  - Read all of these as "this constraint does not bind", not as speeds. No walk faster than the nominal one (0.243 /
    0.218 m/s) is shown in closed loop.
- **At the bar's 0.40 m/s, 75 cm, 8x8 mode.** At walking height the sensors see the floor 0.98–1.07 m ahead of the
  toes (a). The robot covers 0.03–0.10 m while it reacts on the microcontroller path, or 0.03–0.19 m on the host path
  (a, on (d) timing). It covers 0.10–0.52 m while it stops, depending on the stop reading (a).
  - With a two-step stop at least 0.41 m is left over on any path and in any ToF mode (0.32 m at 60 cm). With the
    spec's 1.5-stride stop at least 0.23 m is left (0.17 m at 60 cm).
  - At the ladder's top rung, 0.50 m/s, at least 0.31 m is left with two steps and 0.11 m with 1.5 strides (0.22 and
    0.04 m at 60 cm, where that rung is above the body's fastest reference walk).
- **The computer is not what moves it.**
  - The microcontroller and the Jetson's real-time core get the same range: no sourced timing separates them. Each
    loop period, the bus read and the command delay move the governor by about 0.02 m/s or less across their whole
    range.
  - Routing the stop through the host's planner can only add time: 0–205 ms (d). The 0 is a bound, and the 200 ms is
    the spec's deadline, assumed met, plus a link of up to 5 ms. It lowers the low end of the two-step governor at 75
    cm from 0.92 to 0.77 m/s and leaves the high end where it is. No latency has been measured on either host, so
    nothing here separates the Pi 5 from the Jetson.
  - The paths differ in what happens when Linux fails. By design only the microcontroller path keeps its ToF stop
    then; that firmware is not written (`COMPUTE.md`).
- **What does move it** (75 cm, 8x8, microcontroller path, two-step stop, 1.05 m/s at the reference values):
  - the stop reading itself: 1.48 m/s if the stance foot alone brakes the robot, 1.05 m/s with two steps, 0.90 m/s
    with today's stop builder at its worst phase, 0.74 m/s with the spec's 1.5 strides;
  - the trunk pitching nose-down by the spec's peak limit of 8°: −0.35 m/s; by its RMS limit of 4°: −0.20 m/s;
  - not trusting the far row of zones: −0.21 m/s;
  - the number of ToF frames needed to detect, 1 to 3: 0.14 m/s. This is the largest timing term on this path, and it
    belongs to the sensor, not to the computer. On the host path the planner's 0–200 ms is larger: 0.18 m/s.
- **Every adverse reading at once** (trunk at 8°, the hazard must fill a zone, a 0.10 m gap, every timing term high; a
  (d) stack, not a likely case). At 75 cm:
  - with a two-step stop the governor falls to 0.40–0.46 m/s on the microcontroller and real-time-core paths and to
    0.30–0.38 m/s on the host path;
  - with today's stop builder at its worst phase, to 0.30–0.39 and 0.24–0.30 m/s; with the spec's 1.5 strides, to
    0.22–0.27 and 0.20–0.23 m/s;
  - with a stop as hard as the spec allows, to 0.90–0.96 and 0.69–0.74 m/s.
  - So in that case the bar's 0.40 m/s keeps its ToF stop with a stop of at most two steps only if the real-time layer
    decides it, and on no path with today's stop builder at its worst phase or with a 1.5-stride stop. At 60 cm the
    same holds, except that 4x4 at 30 Hz falls just short even on the real-time layer (0.38 m/s).
- **ToF mode.** By geometry alone, at 75 cm 8x8 zones see 0.09–0.13 m farther than 4x4 zones (a), and a 0.10 m
  obstacle can fill one of their far-row zones (it needs 0.095 m, a). 4x4 at 60 Hz reacts in 29–101 ms instead of
  81–259 ms (a, on (d) timing), but a far-row zone needs 0.158 m to be filled. Whether either mode ranges the floor
  that far out in the owner's light is not established: ST states range per zone count and rate, those tables were not
  read, and no vendor range was found for 4x4 at 60 Hz. No mode brings the governor into the ladder by the main
  method.
- **Not established.** How many frames the sensor needs to report an edge; whether it ranges the floor 1.05 m away at
  about 23–25° to the floor in the owner's light, in each mode; one I2C packet size; the trunk's pitch in closed-loop
  walking; and any stop from a walk faster than the nominal one, since no such stop reference exists yet (section 9).
- **Nothing to decide now.** Section 10 lists six readings for the delay bench sheet (decision-log row A17). Section 13
  has three draft decision-log rows.

## 1. The question and the method

The head camera cannot see the floor nearer than 2.395 m ahead of the toes at 75 cm (2.396 m at 60 cm) (a; each body's
sensor screen, `summary.centre_column_nearest_visible_floor_ahead_of_toes_m`). A drop, or a low obstacle that comes
into view inside that distance, is seen only by the ToF pair (P22). Autonomy must be able to stop for it.

For a body, a ToF mode, a stop path and a stop reading, the governor speed v is the largest speed with

    d_see − v × T_react ≥ d_stop(v)

- **`d_see`** is the farthest floor point ahead of the toes that the sensors see on the foot path. It comes from the
  body's committed sensor screen: the zone-centre rays of the far row on a flat floor.
  - The record holds the 8x8 zones at the home pose. Its summary value, 1.110 m at 75 cm and 0.943 m at 60 cm, is the
    farthest of them. The module reproduces the recorded zones with the screen's own lens model (`sensors.Lens`). No
    git-ignored file is needed.
  - The main table takes `d_see` at the walking height, not at the home pose. Every reference walk holds the trunk
    0.03 H lower than the home pose (`designlab/gait.py`; the Stage 1 stop record carries the same crouch), and the
    sensors ride on the trunk. That takes about 0.03 m off: 0.982–1.075 m instead of 1.013–1.110 m at 75 cm (a).
  - The 4x4 zones come from the same computation. The 4x4 grid is taken to cover the same field with zones twice as
    wide (d).
  - `d_see` is a range. Its far end is the first far-row zone centre to reach a hazard that spans the path. Its near
    end is the last one, so that a hazard anywhere on the foot path has been reached.
- **`T_react`** is the time from the hazard reaching `d_see` to the drives starting the stop: the ToF frames needed to
  detect, the I2C read, the stop logic's loop period, the walking loop's period and the command delay to the drives.
  The host path adds the planner and its link. Section 3 gives each term's kind and range.
- **`d_stop(v)`** is how far the front of the feet travels in the stop, measured from the toe line at the instant the
  stop begins. Six readings are used, all from what the study has (section 4). Two floors apply:
  - **The spec's deceleration limit.** A stop counted in steps is never taken shorter than v² / (2 × 1.5 m/s²). The
    spec limits the trunk's deceleration in every stop, and the bar's stop gates both the step count and that limit
    (c, `STAGE2_PLAN.md` section 5).
  - **The lead of a foot on the ground.** The toe line rides with the trunk, but in a walk a stance foot lands ahead
    of it by duty / 2 × stride (a, `designlab/gait.py`, `_foot_x`): 0.083 m in the nominal walk, 0.099 m at 0.40 m/s
    and 0.112 m in the Froude 0.05 walk at 75 cm (0.067, 0.085 and 0.091 m at 60 cm). No stop is taken shorter than
    that lead. It is longer than a stop at 1.5 m/s² only below 0.68 m/s at 75 cm and 0.61 m/s at 60 cm, so it changes
    the distances at the bar and on the ladder (section 5) and no cell of the governor table. A step reading already
    covers it, since a step is longer than the lead.

The low end of each range takes the near `d_see` with every timing term high. The high end takes the far `d_see` with
every term low. The main table keeps no gap between the toes and the hazard, as the method is stated; section 6 prices
a 0.10 m gap.

The three stop paths:
1. **The microcontroller reads the ToF pair and stops.** This is `COMPUTE.md`'s design and the owner's baseline (P30).
2. **An isolated real-time core on the Jetson reads the ToF pair and stops.** This is the alternative the delay bench
   test measures (P30). PR #668 (unverified) calls it Partial PLEM, a term its report does not define.
3. **The host's planner decides the stop through the camera pipeline.** This is PR #668's (unverified) routing.
   `COMPUTE.md` does not route the ToF stop this way.

## 2. What the sensors see, per ToF mode

| Body | ToF mode | Frame period, ms (c) | Zone, deg (c) | d_see at the home pose, the screen's, m (a) | d_see at the walking height, used here, m (a) | Slant range to it, m (a) | Angle to the floor there, deg (a) | Vendor's range for this mode | Second row, m (a) | Far-row zone filled, m (a) | Height that fills a far-row zone, m (a) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `75-M-H-10` | 8x8 zones at 15 Hz | 66.7 | 5.62 | 1.013–1.110 | 0.982–1.075 | 1.052 | 23–25 | Not read (datasheet tables 17, 18) | 0.784–0.867 | 0.882–0.961 | 0.095 |
| `75-M-H-10` | 4x4 zones at 60 Hz | 16.7 | 11.25 | 0.921–0.974 | 0.892–0.944 | 0.927 | 26–28 | None found for this rate | 0.597–0.654 | 0.738–0.777 | 0.158 |
| `75-M-H-10` | 4x4 zones at 30 Hz | 33.3 | 11.25 | 0.921–0.974 | 0.892–0.944 | 0.927 | 26–28 | Not read (datasheet tables 17, 18) | 0.597–0.654 | 0.738–0.777 | 0.158 |
| `60-N-T-10` | 8x8 zones at 15 Hz | 66.7 | 5.62 | 0.862–0.943 | 0.837–0.915 | 0.874 | 23–25 | Not read (datasheet tables 17, 18) | 0.673–0.742 | 0.755–0.820 | 0.079 |
| `60-N-T-10` | 4x4 zones at 60 Hz | 16.7 | 11.25 | 0.786–0.830 | 0.763–0.806 | 0.771 | 26–28 | None found for this rate | 0.518–0.565 | 0.635–0.667 | 0.132 |
| `60-N-T-10` | 4x4 zones at 30 Hz | 33.3 | 11.25 | 0.786–0.830 | 0.763–0.806 | 0.771 | 26–28 | Not read (datasheet tables 17, 18) | 0.518–0.565 | 0.635–0.667 | 0.132 |

- **Frame period (c).** ST's user manual UM2884, table 2, gives the ranging frequency limits: 1–15 Hz at 8x8 and 1–60
  Hz at 4x4. The first two modes run at the fastest rate allowed. The third, 4x4 at 30 Hz, is inside the limits; it is
  here because the datasheet's range table for 4x4 is given at 30 Hz, as the review's search extract has it (d; this
  document's own searches did not return the table). The catalog note on `st_vl53l5cx_carrier` calls the 15 Hz figure
  unverified; section 12 says how it was read.
- **Zone (c).** The catalog's 45° × 45° field divided by the zones per side.
- **`d_see` at the home pose (a).** Recomputed zone centres on the foot path, both sensors. At 8x8 the far end equals
  the screen's recorded value.
- **`d_see` at the walking height (a).** The same rays with the trunk 0.03 H lower. Every other column, and every
  table after this one, uses this height.
- **Slant range and angle (a).** The distance from the sensor to the farthest far-row floor point on the foot path,
  and the angle between the beam and the floor across the far row.
- **Vendor's range for this mode.** What is sourced does not settle whether the far row returns a range:
  - The catalog gives a 4 m maximum (c, `st_vl53l5cx_carrier`). Pololu's page says that figure assumes an ideal dark
    environment (c).
  - In 5000 lux the catalog note gives about 1.7 m for this sensor (c, `pololu_vl53l8cx`, from Pololu's page; an ST
    presentation gives the same 170 cm). That figure is for a large reflective target, and neither source states the
    zone count or the rate.
  - ST's datasheet has two tables of maximum range, 17 and 18 (c; an ST forum thread names them). The review's search
    extract has them per mode: 8x8 at 15 Hz and 4x4 at 30 Hz, in the dark and in 5000 lux, for white and grey targets
    (d). Their values could not be read by the review or by this document. The review recalls the 8x8 figures as the
    lower ones (d, unconfirmed). No vendor range was found for 4x4 at 60 Hz.
  - ST measures with the target square to the sensor (c for its field-of-view measurement; d, this document's reading,
    for its range tables). Here the beam meets the floor at 23–28°.
  - So "8x8 sees farther" is geometry only. Whether the far row returns a range on the owner's floors, in either mode,
    is not established.
- **Second row (a).** The zone centres of the next row down, nearest to farthest, for the case where the far row is
  not trusted.
- **Far-row zone filled (a).** Where the far row's lower edge meets the floor. A hazard edge there has a whole far-row
  zone beyond it.
- **Height that fills a far-row zone (a).** An obstacle standing at that lower edge fills the zone only if it is this
  tall. The spec's smallest obstacle is 0.10 m (c, `BEHAVIOR_SPEC.md` section 3). It fills an 8x8 far-row zone at both
  sizes (0.095 and 0.079 m), and it does not fill a 4x4 one (0.158 and 0.132 m). A zone that is only part filled
  reports a mix; what the sensor then reports is for the bench.

These are the screen's assumptions: geometry only, a flat floor, no noise, multipath or ambient light. On these two
bodies the ToF pair is exposed: the screen's records count no ray through a cover window for either sensor, so no
window is assumed here.

## 3. Reaction time per stop path

| Timing term | Kind | 8x8 zones at 15 Hz: low–high, ms | 4x4 zones at 60 Hz: low–high, ms | 4x4 zones at 30 Hz: low–high, ms | In stop paths |
|---|---|---|---|---|---|
| ToF frames to detect | (c period x d frames) | 66.7–200.0 | 16.7–50.0 | 33.3–100.0 | 1, 2, 3 |
| I2C read of the frame | (d) | 2.8–14.2 | 1.1–5.6 | 1.1–5.6 | 1, 2, 3 |
| Microcontroller loop period | (c) | 1.0–5.0 | 1.0–5.0 | 1.0–5.0 | 1, 3 |
| Real-time core loop period | (d) | 1.0–5.0 | 1.0–5.0 | 1.0–5.0 | 2 |
| Walking loop period | (c paths 1 and 3; d path 2, taken equal) | 10.0–20.0 | 10.0–20.0 | 10.0–20.0 | 1, 2, 3 |
| Host planner, sample to motion request | (d: low a bound, high the c deadline assumed met) | 0.0–200.0 | 0.0–200.0 | 0.0–200.0 | 3 |
| Host to real-time layer link | (d) | 0.0–5.0 | 0.0–5.0 | 0.0–5.0 | 3 |
| Command delay to the drives | (d) | 0.0–20.0 | 0.0–20.0 | 0.0–20.0 | 1, 2, 3 |

- **ToF frames to detect.** The frame period is (c). The number of frames is (d), this document's: 1 if the hazard
  arrives as a frame begins, 2 if it arrives just after one began, 3 if the stop logic also wants a second frame to
  confirm. The firmware is not written, so none of these is chosen.
- **I2C read (d).** Bytes × 9 bits ÷ bus rate. The bus rate, 400 kHz to 1 MHz, is (c) from the sensor's datasheet. The
  packet sizes come from ST's I2C bandwidth table (section 12). Read with their rows: 124 bytes for 4x4 with one
  target per zone and three outputs, and the largest packets, 1008 bytes at 4x4 and 3360 at 8x8, with every output.
  The 316 bytes used for 8x8 is taken to be the 124-byte row's outputs at 8x8, which the extract did not show, and
  that the stop needs only those outputs is assumed; so the term stays (d). Low is one sensor at 1 MHz. High is both
  sensors read back to back at 400 kHz.
- **Microcontroller loop (c).** `COMPUTE.md`: "CAN loop, IMU, ToF stop, watchdogs" at 200–1000 Hz on the
  microcontroller. One period, 1–5 ms, is the longest wait for the next pass.
- **Real-time core loop (d).** No document of the study fixes it, and PR #668 (unverified) gives no measured period.
  It is taken equal to the microcontroller's.
- **Walking loop (c on paths 1 and 3; d on path 2).** `COMPUTE.md` puts the locomotion policy on the microcontroller
  at 50–100 Hz, so there the stop reaches the walk within 10–20 ms. On path 2 the walking loop runs on the Jetson's
  real-time core, where no document of the study fixes its rate; it is taken equal.
- **Host planner (d).** The high end, 200 ms, is `BEHAVIOR_SPEC.md` section 3's deadline from the oldest sensor sample
  to a motion request (c). It is a target, assumed met, not a measured latency. The low end is 0 ms, a bound: a
  planner at its best phase waits almost nothing, and no compute time is sourced. One planner period at `COMPUTE.md`'s
  10–15 Hz is 67–100 ms, which is a typical wait, not a floor. PR #668's (unverified) 45–60 ms for the Jetson and
  175–200 ms for the Pi 5 all lie inside the range, so this document can neither confirm nor exclude them; only the
  bench can (section 10, reading 5). No latency has been measured on the Pi 5 or on the Jetson, so both hosts take the
  same range.
- **Host link (d).** A serial or USB hop from the host to the real-time layer, 0–5 ms.
- **Command delay (d).** The study's assumption: the bar is judged at 10 ms until a bench value exists (P29, P30;
  A17). The range is the 0–20 ms Stage 1 simulated.

| Stop path | Reaction time, 8x8 zones at 15 Hz: low–high, ms (d) | Reaction time, 4x4 zones at 60 Hz: low–high, ms (d) | Reaction time, 4x4 zones at 30 Hz: low–high, ms (d) | If Linux fails |
|---|---|---|---|---|
| 1. Microcontroller reads the ToF pair and stops | 80.5–259.2 | 28.8–100.6 | 45.4–150.6 | By design the ToF stop keeps working, since the real-time layer does not need Linux; that firmware is not written (COMPUTE.md) |
| 2. Isolated real-time core on the Jetson reads the ToF pair and stops | 80.5–259.2 | 28.8–100.6 | 45.4–150.6 | No ToF stop: the walking loop itself runs on Linux, and a Linux failure ends in torque-off (JETSON_PLEM_REVIEW.md section 3, Safety) |
| 3. Host planner decides the stop through the camera pipeline | 80.5–464.2 | 28.8–305.6 | 45.4–355.6 | No ToF stop from the planner; by design the microcontroller's own stops remain (velocity-request expiry, watchdogs; that firmware is not written: COMPUTE.md) |

Paths 1 and 2 come out the same because nothing sourced separates them. PR #668's (unverified) 2 ms of bus delay for
the real-time core lies inside the command delay's 0–20 ms range, and that whole range moves the governor by 0.02 m/s
(section 6). Path 3 shares their low end because the planner's low end is a bound of 0. The paths do differ in the
last column, which no governor speed shows; and what that column says of paths 1 and 3 is the design's intent, since
the microcontroller's firmware is not written (`COMPUTE.md`: "Until it exists, a Linux crash does not end in a
controlled stop").

## 4. The governor

Each cell is a governor speed in m/s, kind (a), with its range from (d) timing. The square bracket in a column's
heading is the kind of the stop reading, not of the speed.

| Body | ToF mode | Stop path | Spec limit, 1.5 m/s² [reading: c, a target]: governor, m/s (a, range from (d) timing) | One-foot hold [reading: a]: governor, m/s (a, range from (d) timing) | One step [reading: count c, a target; step a]: governor, m/s (a, range from (d) timing) | Two steps [reading: count c, a target; step a]: governor, m/s (a, range from (d) timing) | Today's stop builder, worst phase (2.4 steps) [reading: a]: governor, m/s (a, range from (d) timing) | Spec stop limit, 1.5 strides [reading: count c, a target; step a]: governor, m/s (a, range from (d) timing) |
|---|---|---|---|---|---|---|---|---|
| `75-M-H-10` | 8x8 zones at 15 Hz | 1. Microcontroller | 1.37–1.68 † | 1.34–1.63 † | 1.37¶–1.68¶ † | 0.92–1.21 † | 0.80–1.02 † | 0.67–0.83 † |
| `75-M-H-10` | 8x8 zones at 15 Hz | 2. Real-time core | 1.37–1.68 † | 1.34–1.63 † | 1.37¶–1.68¶ † | 0.92–1.21 † | 0.80–1.02 † | 0.67–0.83 † |
| `75-M-H-10` | 8x8 zones at 15 Hz | 3. Host planner | 1.16–1.68 † | 1.13–1.63 † | 1.13–1.68¶ † | 0.77–1.21 † | 0.68–1.02 † | 0.58–0.83 † |
| `75-M-H-10` | 4x4 zones at 60 Hz | 1. Microcontroller | 1.49–1.64 † | 1.45–1.59 † | 1.49¶–1.64¶ † | 0.98–1.13 † | 0.83–0.94 † | 0.68–0.76 † |
| `75-M-H-10` | 4x4 zones at 60 Hz | 2. Real-time core | 1.49–1.64 † | 1.45–1.59 † | 1.49¶–1.64¶ † | 0.98–1.13 † | 0.83–0.94 † | 0.68–0.76 † |
| `75-M-H-10` | 4x4 zones at 60 Hz | 3. Host planner | 1.24–1.64 † | 1.21–1.59 † | 1.24¶–1.64¶ † | 0.80–1.13 † | 0.70–0.94 † | 0.59–0.76 † |
| `75-M-H-10` | 4x4 zones at 30 Hz | 1. Microcontroller | 1.43–1.62 † | 1.39–1.57 † | 1.43¶–1.62¶ † | 0.93–1.10 † | 0.80–0.93 † | 0.65–0.75 † |
| `75-M-H-10` | 4x4 zones at 30 Hz | 2. Real-time core | 1.43–1.62 † | 1.39–1.57 † | 1.43¶–1.62¶ † | 0.93–1.10 † | 0.80–0.93 † | 0.65–0.75 † |
| `75-M-H-10` | 4x4 zones at 30 Hz | 3. Host planner | 1.19–1.62 † | 1.16–1.57 † | 1.17–1.62¶ † | 0.77–1.10 † | 0.67–0.93 † | 0.57–0.75 † |
| `60-N-T-10` | 8x8 zones at 15 Hz | 1. Microcontroller | 1.24–1.54 † | 1.16–1.41 † | 1.24¶–1.54¶ † | 0.85–1.13 † | 0.74–0.96 † | 0.62–0.78 † |
| `60-N-T-10` | 8x8 zones at 15 Hz | 2. Real-time core | 1.24–1.54 † | 1.16–1.41 † | 1.24¶–1.54¶ † | 0.85–1.13 † | 0.74–0.96 † | 0.62–0.78 † |
| `60-N-T-10` | 8x8 zones at 15 Hz | 3. Host planner | 1.03–1.54 † | 0.98–1.41 † | 1.01–1.54¶ † | 0.70–1.13 † | 0.63–0.96 † | 0.54–0.78 † |
| `60-N-T-10` | 4x4 zones at 60 Hz | 1. Microcontroller | 1.37–1.51 † | 1.26–1.38 † | 1.37¶–1.51¶ † | 0.92–1.07 † | 0.78–0.89 † | 0.64–0.72 † |
| `60-N-T-10` | 4x4 zones at 60 Hz | 2. Real-time core | 1.37–1.51 † | 1.26–1.38 † | 1.37¶–1.51¶ † | 0.92–1.07 † | 0.78–0.89 † | 0.64–0.72 † |
| `60-N-T-10` | 4x4 zones at 60 Hz | 3. Host planner | 1.12–1.51 † | 1.05–1.38 † | 1.12¶–1.51¶ † | 0.74–1.07 † | 0.65–0.89 † | 0.55–0.72 † |
| `60-N-T-10` | 4x4 zones at 30 Hz | 1. Microcontroller | 1.30–1.49 † | 1.21–1.36 † | 1.30¶–1.49¶ † | 0.87–1.04 † | 0.75–0.88 † | 0.61–0.71 † |
| `60-N-T-10` | 4x4 zones at 30 Hz | 2. Real-time core | 1.30–1.49 † | 1.21–1.36 † | 1.30¶–1.49¶ † | 0.87–1.04 † | 0.75–0.88 † | 0.61–0.71 † |
| `60-N-T-10` | 4x4 zones at 30 Hz | 3. Host planner | 1.07–1.49 † | 1.01–1.36 † | 1.06–1.49¶ † | 0.70–1.04 † | 0.62–0.88 † | 0.53–0.71 † |

† The whole range lies above the body's fastest reference walk (0.496 m/s at 75 cm, 0.446 m/s at 60 cm). Above it the
step readings hold that walk's period, an extrapolation (d): at these speeds a two-step stop implies steps of
0.62–0.97 H at 75 cm, against 0.40 H in the Froude 0.05 walk. Read such a cell as "the ToF stop does not bind", not as
a speed. Every cell carries the mark.

¶ At this end the spec's 1.5 m/s² sets the stop, not the step count: one step from that speed would need a harder stop
than the spec allows. Where both ends carry the mark the cell equals the spec-limit cell. The other step readings stay
under the limit: a two-step stop implies 0.47–0.78 m/s² at its governor speeds (a).

The lowest governor per reading, over the paths and modes, at 75 cm / 60 cm (m/s; a, on (d) timing): spec limit 1.16 /
1.03; one-foot hold 1.13 / 0.98; one step 1.13 / 1.01; two steps 0.77 / 0.70; today's stop builder at its worst phase
0.67 / 0.62; the spec's 1.5 strides 0.57 / 0.53. The lowest of all is the 1.5-stride stop decided by the host's
planner with 4x4 zones at 30 Hz.

Which stop reading is which:
- **Spec limit, 1.5 m/s² [c, a target].** `v² / (2 × 1.5)`. The behaviour spec's limit on trunk acceleration in a
  start or stop (`BEHAVIOR_SPEC.md` section 1; `STAGE1_PLAN.md` decision 10), held from the first instant. It is the
  hardest stop the spec allows, and the bar's stop gates it too (`STAGE2_PLAN.md` section 5). One foot cannot hold it
  (F12): it needs both feet down or a step.
- **One-foot hold [a].** `v² / (2a)`, with `a` the Stage 0 one-foot stopping hold: 1.41 m/s² at 75 cm with the bigger
  feet, 1.25 m/s² at 60 cm (each body's envelope screen; F12, F23). It is what the stance foot's ankle can hold in a
  static model with no controller.
- **One step, two steps [count c, a target; step a].** One or two step lengths of the study's reference walk at speed
  v, and never less than the spec's limit allows (section 1). This is the alternative reading: the bar asks for a stop
  within a step or two (`STAGE2_PLAN.md` section 5). A step is half a stride. The stride at speed v is the reference
  walks' own: the period is linear in speed between the nominal, brisk and Froude 0.05 walks (`look.speed_spec`). At
  0.40 m/s a step is 0.172 m at 75 cm and 0.150 m at 60 cm.
- **Today's stop builder, worst phase [a].** The one stop reference the study has built is from the nominal walk
  (`transitions.json`, `stop`). Its speed ramps down over 0.4 stride, so the path covers 0.2 stride after the stop
  command: 0.055 m at 75 cm, 0.045 m at 60 cm. It begins at a left lift-off only, since no mirrored stop is built. A
  request at the worst phase therefore waits a full stride first: 1.2 strides, or 2.4 steps. The column applies those
  2.4 steps at every speed, which is this document's extension (d): no stop reference exists from a faster walk
  (`STAGE1_PLAN.md` decision 31), and Stage 2's two-step stop is deferred (`designlab/stage2/look.py`, `DEFERRED`).
- **Spec stop limit, 1.5 strides [count c, a target; step a].** Three steps. The behaviour spec asks for a stop within
  1.5 strides (`BEHAVIOR_SPEC.md` section 1), which decision 18 of `STAGE1_PLAN.md` reads as the forward speed staying
  under 5 % of the command within 1.5 strides of the stop command. It is the longest stop the spec allows, so it gives
  the lowest governor of the six.

None of the six is a closed-loop result (b). Stage 1 shows the stop from the nominal walk at no servo delay (S1 §2),
but its records were not read for distances here. The record holds, for each end of each cell, the stop distance, what
set it, the constant deceleration it implies and the step as a fraction of H.

## 5. Against the bar and the ladder

The same condition, read at the bar's 0.40 m/s as distances (kind (a); the reaction and left-over columns rest on (d)
timing):

| Body | ToF mode | Stop path | d_see, m (a) | Travelled while reacting at 0.40 m/s, m (a, from (d) timing) | Stop distance, m (a): spec limit, 1.5 m/s²; one-foot hold; one step; two steps; today's stop builder, worst phase (2.4 steps); spec stop limit, 1.5 strides | Left over, worst–best, m (a, from (d) timing): two steps | Left over, worst–best, m (a, from (d) timing): spec stop limit, 1.5 strides |
|---|---|---|---|---|---|---|---|
| `75-M-H-10` | 8x8 zones at 15 Hz | 1. Microcontroller | 0.982–1.075 | 0.032–0.104 | 0.099◊; 0.099◊; 0.172; 0.344; 0.413; 0.516 | 0.534–0.699 | 0.361–0.526 |
| `75-M-H-10` | 8x8 zones at 15 Hz | 2. Real-time core | 0.982–1.075 | 0.032–0.104 | 0.099◊; 0.099◊; 0.172; 0.344; 0.413; 0.516 | 0.534–0.699 | 0.361–0.526 |
| `75-M-H-10` | 8x8 zones at 15 Hz | 3. Host planner | 0.982–1.075 | 0.032–0.186 | 0.099◊; 0.099◊; 0.172; 0.344; 0.413; 0.516 | 0.452–0.699 | 0.280–0.526 |
| `75-M-H-10` | 4x4 zones at 60 Hz | 1. Microcontroller | 0.892–0.944 | 0.011–0.040 | 0.099◊; 0.099◊; 0.172; 0.344; 0.413; 0.516 | 0.508–0.589 | 0.336–0.416 |
| `75-M-H-10` | 4x4 zones at 60 Hz | 2. Real-time core | 0.892–0.944 | 0.011–0.040 | 0.099◊; 0.099◊; 0.172; 0.344; 0.413; 0.516 | 0.508–0.589 | 0.336–0.416 |
| `75-M-H-10` | 4x4 zones at 60 Hz | 3. Host planner | 0.892–0.944 | 0.011–0.122 | 0.099◊; 0.099◊; 0.172; 0.344; 0.413; 0.516 | 0.426–0.589 | 0.254–0.416 |
| `75-M-H-10` | 4x4 zones at 30 Hz | 1. Microcontroller | 0.892–0.944 | 0.018–0.060 | 0.099◊; 0.099◊; 0.172; 0.344; 0.413; 0.516 | 0.488–0.582 | 0.316–0.410 |
| `75-M-H-10` | 4x4 zones at 30 Hz | 2. Real-time core | 0.892–0.944 | 0.018–0.060 | 0.099◊; 0.099◊; 0.172; 0.344; 0.413; 0.516 | 0.488–0.582 | 0.316–0.410 |
| `75-M-H-10` | 4x4 zones at 30 Hz | 3. Host planner | 0.892–0.944 | 0.018–0.142 | 0.099◊; 0.099◊; 0.172; 0.344; 0.413; 0.516 | 0.406–0.582 | 0.234–0.410 |
| `60-N-T-10` | 8x8 zones at 15 Hz | 1. Microcontroller | 0.837–0.915 | 0.032–0.104 | 0.085◊; 0.085◊; 0.150; 0.300; 0.360; 0.451 | 0.433–0.582 | 0.283–0.432 |
| `60-N-T-10` | 8x8 zones at 15 Hz | 2. Real-time core | 0.837–0.915 | 0.032–0.104 | 0.085◊; 0.085◊; 0.150; 0.300; 0.360; 0.451 | 0.433–0.582 | 0.283–0.432 |
| `60-N-T-10` | 8x8 zones at 15 Hz | 3. Host planner | 0.837–0.915 | 0.032–0.186 | 0.085◊; 0.085◊; 0.150; 0.300; 0.360; 0.451 | 0.351–0.582 | 0.201–0.432 |
| `60-N-T-10` | 4x4 zones at 60 Hz | 1. Microcontroller | 0.763–0.806 | 0.011–0.040 | 0.085◊; 0.085◊; 0.150; 0.300; 0.360; 0.451 | 0.422–0.494 | 0.272–0.344 |
| `60-N-T-10` | 4x4 zones at 60 Hz | 2. Real-time core | 0.763–0.806 | 0.011–0.040 | 0.085◊; 0.085◊; 0.150; 0.300; 0.360; 0.451 | 0.422–0.494 | 0.272–0.344 |
| `60-N-T-10` | 4x4 zones at 60 Hz | 3. Host planner | 0.763–0.806 | 0.011–0.122 | 0.085◊; 0.085◊; 0.150; 0.300; 0.360; 0.451 | 0.340–0.494 | 0.190–0.344 |
| `60-N-T-10` | 4x4 zones at 30 Hz | 1. Microcontroller | 0.763–0.806 | 0.018–0.060 | 0.085◊; 0.085◊; 0.150; 0.300; 0.360; 0.451 | 0.402–0.487 | 0.252–0.337 |
| `60-N-T-10` | 4x4 zones at 30 Hz | 2. Real-time core | 0.763–0.806 | 0.018–0.060 | 0.085◊; 0.085◊; 0.150; 0.300; 0.360; 0.451 | 0.402–0.487 | 0.252–0.337 |
| `60-N-T-10` | 4x4 zones at 30 Hz | 3. Host planner | 0.763–0.806 | 0.018–0.142 | 0.085◊; 0.085◊; 0.150; 0.300; 0.360; 0.451 | 0.320–0.487 | 0.170–0.337 |

◊ The lead of a foot on the ground sets this distance: at 0.40 m/s it is longer than the stop the deceleration alone
would give (0.053 m at 1.5 m/s²).

The distance left over in the worst case at each rung of the speed ladder. Each cell gives it for a two-step stop,
then for the spec's 1.5-stride stop:

| Body | ToF mode | Stop path | Left over at 0.30 m/s, m (a, from (d) timing): two steps / 1.5 strides | Left over at 0.35 m/s, m (a, from (d) timing): two steps / 1.5 strides | Left over at 0.40 m/s, m (a, from (d) timing): two steps / 1.5 strides | Left over at 0.45 m/s, m (a, from (d) timing): two steps / 1.5 strides | Left over at 0.50 m/s, m (a, from (d) timing): two steps / 1.5 strides |
|---|---|---|---|---|---|---|---|
| `75-M-H-10` | 8x8 zones at 15 Hz | 1. Microcontroller | 0.603 / 0.452 | 0.580 / 0.424 | 0.534 / 0.361 | 0.490 / 0.302 | 0.447 / 0.245 ‡ |
| `75-M-H-10` | 8x8 zones at 15 Hz | 2. Real-time core | 0.603 / 0.452 | 0.580 / 0.424 | 0.534 / 0.361 | 0.490 / 0.302 | 0.447 / 0.245 ‡ |
| `75-M-H-10` | 8x8 zones at 15 Hz | 3. Host planner | 0.541 / 0.390 | 0.508 / 0.352 | 0.452 / 0.280 | 0.397 / 0.210 | 0.345 / 0.142 ‡ |
| `75-M-H-10` | 4x4 zones at 60 Hz | 1. Microcontroller | 0.561 / 0.410 | 0.546 / 0.391 | 0.508 / 0.336 | 0.472 / 0.284 | 0.438 / 0.235 ‡ |
| `75-M-H-10` | 4x4 zones at 60 Hz | 2. Real-time core | 0.561 / 0.410 | 0.546 / 0.391 | 0.508 / 0.336 | 0.472 / 0.284 | 0.438 / 0.235 ‡ |
| `75-M-H-10` | 4x4 zones at 60 Hz | 3. Host planner | 0.500 / 0.349 | 0.474 / 0.319 | 0.426 / 0.254 | 0.380 / 0.192 | 0.335 / 0.133 ‡ |
| `75-M-H-10` | 4x4 zones at 30 Hz | 1. Microcontroller | 0.546 / 0.395 | 0.529 / 0.373 | 0.488 / 0.316 | 0.450 / 0.262 | 0.413 / 0.210 ‡ |
| `75-M-H-10` | 4x4 zones at 30 Hz | 2. Real-time core | 0.546 / 0.395 | 0.529 / 0.373 | 0.488 / 0.316 | 0.450 / 0.262 | 0.413 / 0.210 ‡ |
| `75-M-H-10` | 4x4 zones at 30 Hz | 3. Host planner | 0.485 / 0.334 | 0.457 / 0.301 | 0.406 / 0.234 | 0.357 / 0.170 | 0.310 / 0.108 ‡ |
| `60-N-T-10` | 8x8 zones at 15 Hz | 1. Microcontroller | 0.510 / 0.385 | 0.474 / 0.338 | 0.433 / 0.283 | 0.393 / 0.230 ‡ | 0.344 / 0.162 ‡ |
| `60-N-T-10` | 8x8 zones at 15 Hz | 2. Real-time core | 0.510 / 0.385 | 0.474 / 0.338 | 0.433 / 0.283 | 0.393 / 0.230 ‡ | 0.344 / 0.162 ‡ |
| `60-N-T-10` | 8x8 zones at 15 Hz | 3. Host planner | 0.448 / 0.324 | 0.402 / 0.266 | 0.351 / 0.201 | 0.301 / 0.138 ‡ | 0.241 / 0.060 ‡ |
| `60-N-T-10` | 4x4 zones at 60 Hz | 1. Microcontroller | 0.483 / 0.359 | 0.456 / 0.320 | 0.422 / 0.272 | 0.391 / 0.227 ‡ | 0.349 / 0.168 ‡ |
| `60-N-T-10` | 4x4 zones at 60 Hz | 2. Real-time core | 0.483 / 0.359 | 0.456 / 0.320 | 0.422 / 0.272 | 0.391 / 0.227 ‡ | 0.349 / 0.168 ‡ |
| `60-N-T-10` | 4x4 zones at 60 Hz | 3. Host planner | 0.422 / 0.297 | 0.384 / 0.248 | 0.340 / 0.190 | 0.298 / 0.135 ‡ | 0.247 / 0.065 ‡ |
| `60-N-T-10` | 4x4 zones at 30 Hz | 1. Microcontroller | 0.469 / 0.344 | 0.438 / 0.302 | 0.402 / 0.252 | 0.368 / 0.205 ‡ | 0.324 / 0.143 ‡ |
| `60-N-T-10` | 4x4 zones at 30 Hz | 2. Real-time core | 0.469 / 0.344 | 0.438 / 0.302 | 0.402 / 0.252 | 0.368 / 0.205 ‡ | 0.324 / 0.143 ‡ |
| `60-N-T-10` | 4x4 zones at 30 Hz | 3. Host planner | 0.407 / 0.282 | 0.366 / 0.230 | 0.320 / 0.170 | 0.276 / 0.113 ‡ | 0.222 / 0.040 ‡ |

‡ The rung is above the body's Froude 0.05 walk, so the stride is extrapolated (d).

- **The bar.** At the bar's 0.40 m/s walk the ToF stop leaves at least 0.41 m (75 cm) and 0.32 m (60 cm) of its sight
  distance unused with a two-step stop, and 0.23 m and 0.17 m with the spec's 1.5-stride stop, in the main table's
  worst case on every path and in every mode (section 7's stack is lower), so by the main method it sets no limit
  below the bar; this is a limit autonomy sets on speed, not a walk the robot has shown.
- **The ladder.** On the speed ladder (0.30–0.50 m/s) it leaves at least 0.31 m (75 cm) and 0.22 m (60 cm) at the top
  rung with a two-step stop, and 0.11 m and 0.04 m with the spec's 1.5-stride stop, in the main table's worst case, so
  by the main method it sets no limit inside the ladder either (section 7's stack does); no walk faster than the
  nominal one (0.243 / 0.218 m/s) is shown in closed loop.

## 6. What moves the answer

One term at a time, with the others at their reference values (the midpoint of each range; 2 frames; 10 ms of command
delay):

| Term, one at a time (`75-M-H-10`, 8x8 zones at 15 Hz, path 1. Microcontroller, stop: two steps) | Kind of the term | Range | Governor at the range's ends, m/s (a) | Swing, m/s |
|---|---|---|---|---|
| ToF frames to detect | (c period x d frames) | 66.7–200.0 ms | 0.98–1.13 | 0.14 |
| Far row: nearest to farthest zone centre on the foot path | (a) | 0.982–1.075 m | 1.00–1.10 | 0.10 |
| Command delay to the drives | (d) | 0.0–20.0 ms | 1.04–1.06 | 0.02 |
| I2C read of the frame | (d) | 2.8–14.2 ms | 1.04–1.06 | 0.01 |
| Walking loop period | (c) | 10.0–20.0 ms | 1.04–1.06 | 0.01 |
| Microcontroller loop period | (c) | 1.0–5.0 ms | 1.05–1.05 | 0.00 |

| Term, one at a time (`75-M-H-10`, 8x8 zones at 15 Hz, path 3. Host planner, stop: two steps) | Kind of the term | Range | Governor at the range's ends, m/s (a) | Swing, m/s |
|---|---|---|---|---|
| Host planner, sample to motion request | (d: low a bound, high the c deadline assumed met) | 0.0–200.0 ms | 0.87–1.05 | 0.18 |
| ToF frames to detect | (c period x d frames) | 66.7–200.0 ms | 0.90–1.01 | 0.12 |
| Far row: nearest to farthest zone centre on the foot path | (a) | 0.982–1.075 m | 0.91–0.99 | 0.09 |
| Command delay to the drives | (d) | 0.0–20.0 ms | 0.94–0.96 | 0.02 |
| I2C read of the frame | (d) | 2.8–14.2 ms | 0.95–0.96 | 0.01 |
| Walking loop period | (c) | 10.0–20.0 ms | 0.95–0.96 | 0.01 |
| Host to real-time layer link | (d) | 0.0–5.0 ms | 0.95–0.95 | 0.00 |
| Microcontroller loop period | (c) | 1.0–5.0 ms | 0.95–0.95 | 0.00 |

Other readings of the same case, one at a time. A reading of what the sensors see is taken at the midpoint of its own
range, as the reference is, so each row changes one thing:

| What-if (`75-M-H-10`, 8x8 zones at 15 Hz, path 1. Microcontroller, stop: two steps; reference 1.05 m/s) | Kind of the reading | Governor, m/s (a) | Change, m/s |
|---|---|---|---|
| Trunk pitched 8 deg nose-down (the spec's peak limit) | (a) | 0.70 | -0.35 |
| Stop reading: spec stop limit, 1.5 strides | (count c, a target; step a) | 0.74 | -0.31 |
| The far row is not trusted: the second row's zone centres | (a) | 0.84 | -0.21 |
| Trunk pitched 4 deg nose-down (the spec's RMS limit) | (a) | 0.85 | -0.20 |
| Stop reading: today's stop builder, worst phase (2.4 steps) | (a) | 0.90 | -0.15 |
| The two sensors range in turn (each frame period doubles) | (d) | 0.92 | -0.13 |
| The hazard must fill a far-row zone on the foot path (the row's near edge) | (a) | 0.94 | -0.11 |
| A 0.10 m gap kept between the toes and the hazard | (d) | 0.95 | -0.10 |
| Every output is read from both sensors at 1 MHz (the largest packet) | (d) | 1.00 | -0.05 |
| Standing height: the screen's home pose, 0.03 H above the walk's | (a) | 1.08 | +0.03 |
| Stop reading: one-foot hold | (a) | 1.48 | +0.43 |
| Stop reading: spec limit, 1.5 m/s² | (c, a target) | 1.52 | +0.47 |
| Stop reading: one step | (count c, a target; step a) | 1.52 | +0.47 |

- **Among the timing terms, the sensor's frames move the governor most on the microcontroller path**: 0.14 m/s. On the
  host path the planner's 0–200 ms moves it most, 0.18 m/s, and the frames 0.12 m/s. The loop periods, the I2C read,
  the link and the command delay each move it by about 0.02 m/s or less.
- **What the sensors see moves it more than any timing term on the microcontroller path.** The trunk pitching
  nose-down swings the far row toward the feet: `d_see` falls from 0.982–1.075 m to 0.792–0.873 m at 4° and to
  0.658–0.719 m at 8° (a). Those angles are the spec's RMS and peak limits on trunk pitch while walking (c,
  `BEHAVIOR_SPEC.md` section 1). The reference walk holds the trunk level, so this is a controller question. The pitch
  axis is taken at the hip (d).
- **The stop reading moves it most of all**: between the one-foot hold and the spec's 1.5 strides lie 0.74 m/s.
- **The standing-height row** shows what the screen's own pose would give: +0.03 m/s. It is not an adverse reading;
  the walk is simply lower than the pose the screen was run at.
- The record holds the same tables for the 60 cm body, the other modes and paths and the one-foot hold
  (`sensitivity`).

## 7. Every adverse reading at once

This stacks the adverse readings of section 6: the far row's lower edge on the foot path, with the trunk at the spec's
peak pitch, less a 0.10 m gap, and every timing term high, at the walking height as everywhere. It is this document's
stack (d), not a likely case. It shows where the governor would sit under each stop reading if all of them held
together. Speeds are kind (a) on a (d) stack.

| Body | ToF mode | Stop path | d_see less the gap, m (a, with the (d) gap) | Reaction time, ms (d) | Spec limit, 1.5 m/s²: governor, m/s (a, from a (d) stack) | One-foot hold: governor, m/s (a, from a (d) stack) | One step: governor, m/s (a, from a (d) stack) | Two steps: governor, m/s (a, from a (d) stack) | Today's stop builder, worst phase (2.4 steps): governor, m/s (a, from a (d) stack) | Spec stop limit, 1.5 strides: governor, m/s (a, from a (d) stack) |
|---|---|---|---|---|---|---|---|---|---|---|
| `75-M-H-10` | 8x8 zones at 15 Hz | 1. Microcontroller | 0.503 | 259.2 | 0.90 | 0.88 | 0.76 | 0.46 | 0.39 ↓ | 0.27 ↓ |
| `75-M-H-10` | 8x8 zones at 15 Hz | 2. Real-time core | 0.503 | 259.2 | 0.90 | 0.88 | 0.76 | 0.46 | 0.39 ↓ | 0.27 ↓ |
| `75-M-H-10` | 8x8 zones at 15 Hz | 3. Host planner | 0.503 | 464.2 | 0.72 | 0.70 | 0.58 | 0.38 ↓ | 0.30 ↓ | 0.23 § |
| `75-M-H-10` | 4x4 zones at 60 Hz | 1. Microcontroller | 0.406 | 100.6 | 0.96 | 0.94 | 0.80 | 0.43 | 0.34 ↓ | 0.23 § |
| `75-M-H-10` | 4x4 zones at 60 Hz | 2. Real-time core | 0.406 | 100.6 | 0.96 | 0.94 | 0.80 | 0.43 | 0.34 ↓ | 0.23 § |
| `75-M-H-10` | 4x4 zones at 60 Hz | 3. Host planner | 0.406 | 305.6 | 0.74 | 0.72 | 0.57 | 0.32 ↓ | 0.24 ↓ | 0.20 § |
| `75-M-H-10` | 4x4 zones at 30 Hz | 1. Microcontroller | 0.406 | 150.6 | 0.90 | 0.88 | 0.73 | 0.40 | 0.30 ↓ | 0.22 § |
| `75-M-H-10` | 4x4 zones at 30 Hz | 2. Real-time core | 0.406 | 150.6 | 0.90 | 0.88 | 0.73 | 0.40 | 0.30 ↓ | 0.22 § |
| `75-M-H-10` | 4x4 zones at 30 Hz | 3. Host planner | 0.406 | 355.6 | 0.69 | 0.68 | 0.53 | 0.30 ↓ | 0.24 § | 0.20 § |
| `60-N-T-10` | 8x8 zones at 15 Hz | 1. Microcontroller | 0.426 | 259.2 | 0.81 | 0.76 | 0.68 | 0.43 | 0.36 ↓ | 0.26 ↓ |
| `60-N-T-10` | 8x8 zones at 15 Hz | 2. Real-time core | 0.426 | 259.2 | 0.81 | 0.76 | 0.68 | 0.43 | 0.36 ↓ | 0.26 ↓ |
| `60-N-T-10` | 8x8 zones at 15 Hz | 3. Host planner | 0.426 | 464.2 | 0.63 | 0.60 | 0.51 | 0.34 ↓ | 0.28 ↓ | 0.21 § |
| `60-N-T-10` | 4x4 zones at 60 Hz | 1. Microcontroller | 0.346 | 100.6 | 0.88 | 0.81 | 0.75 | 0.41 | 0.33 ↓ | 0.21 § |
| `60-N-T-10` | 4x4 zones at 60 Hz | 2. Real-time core | 0.346 | 100.6 | 0.88 | 0.81 | 0.75 | 0.41 | 0.33 ↓ | 0.21 § |
| `60-N-T-10` | 4x4 zones at 60 Hz | 3. Host planner | 0.346 | 305.6 | 0.66 | 0.62 | 0.52 | 0.31 ↓ | 0.23 ↓ | 0.19 § |
| `60-N-T-10` | 4x4 zones at 30 Hz | 1. Microcontroller | 0.346 | 150.6 | 0.82 | 0.76 | 0.67 | 0.38 ↓ | 0.30 ↓ | 0.21 § |
| `60-N-T-10` | 4x4 zones at 30 Hz | 2. Real-time core | 0.346 | 150.6 | 0.82 | 0.76 | 0.67 | 0.38 ↓ | 0.30 ↓ | 0.21 § |
| `60-N-T-10` | 4x4 zones at 30 Hz | 3. Host planner | 0.346 | 355.6 | 0.62 | 0.59 | 0.48 | 0.28 ↓ | 0.22 ↓ | 0.18 § |

↓ Below the bar's 0.40 m/s. § Below the body's nominal walk.

- **Two steps.** At 75 cm the microcontroller and real-time-core paths keep 0.40–0.46 m/s. The host path keeps
  0.30–0.38 m/s.
- **Longer stops.** With today's stop builder at its worst phase the paths keep 0.30–0.39 and 0.24–0.30 m/s; with the
  spec's 1.5 strides, 0.22–0.27 and 0.20–0.23 m/s. A stop that could begin at either foot's lift-off would wait half a
  stride at most: 0.7 stride, or 1.4 steps, which is inside the two-step reading. No such stop is built.
- **Shorter stops.** With a stop as hard as the spec allows the paths keep 0.90–0.96 and 0.69–0.74 m/s, and with one
  step 0.73–0.80 and 0.53–0.58 m/s: above the bar on every path, at both sizes.
- **Against the bar.** In this stack the bar's 0.40 m/s keeps its ToF stop with a stop of at most two steps only if
  the real-time layer decides it, and on no path with today's stop builder at its worst phase or with a 1.5-stride
  stop. At 60 cm, 4x4 at 30 Hz falls short even on the real-time layer with two steps (0.38 m/s).
- **ToF mode.** For a stop of two steps or longer, 8x8 is the better mode on every path in this stack: its longer
  sight outweighs its slower frames.

## 8. How this differs from PR #668's version

PR #668 (unverified), in its section 2.2 on the chest ToF safe-stop horizon, gave a worst-case stride-boundary safe
speed of 0.540 → 0.610 m/s at 75 cm and 0.485 → 0.550 m/s at 60 cm, and a mean-phase safe speed of 0.815 → 0.900 and
0.710 → 0.840 m/s. In each pair the first number is for the Pi 5 and the second for the Jetson: at 200 and 60 ms in
the worst case, and at 175 and 45 ms in the mean-phase case (its script). The review found every timing term assumed.
This document differs in five ways:

1. **The Pi's figure.** PR #668 took 200 ms as the Pi's latency. It is the spec's deadline, a target. Here it is only
   the high end of the host planner's range, for either host.
2. **No credit for the Jetson.** PR #668 gave the Jetson 45–60 ms against the Pi's 175–200 ms with no source. Here
   both hosts take the same range, 0–200 ms, until one is timed (`BEHAVIOR_SPEC.md`: results do not transfer between
   computers without re-timing).
3. **The route.** PR #668 routed the ToF stop through the camera pipeline. Here that is path 3 of three, and the
   baseline is `COMPUTE.md`'s: the stop runs on the microcontroller.
4. **The stop.** PR #668 used a 1.75-stride commitment plus a braking distance in its worst case, and 1.25 strides in
   its mean-phase case, with a stride rule of its own. Here each stop reading is one the study has, and the stride is
   the reference walks'.
5. **The gap.** PR #668 took 0.10 m off the sight distance. Here the main table keeps none, as the method is stated,
   and section 6 prices it (−0.10 m/s).

Where its numbers fall beside this document's host path with 8x8 zones: its worst-case pair straddles the low end of
the longest stop here, the spec's 1.5 strides (0.58–0.83 m/s at 75 cm, 0.54–0.78 m/s at 60 cm), and lies above the
stacked case with a stop of two steps or longer; its mean-phase pair lies inside the two-step range (0.77–1.21 and
0.70–1.13 m/s). Neither set is a speed the robot has shown.

## 9. What this does not show, and what could not be established

- **It is not a speed.** The drives, grip and the controller bind far below these ceilings. The fastest walk shown in
  closed loop is the nominal one.
- **Detection is assumed, not modelled.** The sensor is taken to report a hazard once a zone-centre ray reaches it.
  Range noise, a part-filled zone, a dark or glossy floor and sunlight are not modelled.
- **Frames to detect** could not be established: 1 to 3 is this document's range.
- **Range on the floor** could not be established. ST's range tables, per zone count and rate, were not read. The one
  sourced figure in bright light, 1.7 m in 5000 lux, is for a large reflective target in an unstated mode. The
  farthest far-row point here is 0.77–1.05 m from the sensor, by body and mode, and the beam meets the floor at
  23–28°, not square on. No vendor range was found for 4x4 at 60 Hz.
- **I2C packet sizes.** Three of the four sizes were read with their rows. The 316 bytes used for 8x8, and that the
  stop needs only the smallest set of outputs, are assumed (d). The term they feed moves the governor by 0.01 m/s;
  reading every output from both sensors would cost 0.05 m/s.
- **The field of view.** The catalog and the screen use 45° × 45°, 65° on the diagonal. The datasheet's revision 2
  said 63° on the diagonal; its revision 13, ST's product page and ST's application note AN5894 say 65°, with the same
  45° × 45° sides (c, search extracts). The sides set the zone rays used here, and both revisions agree on them. The
  screen's lens model gives 63.6° on the diagonal (`designlab/sensors.py`), between the two.
- **Two sensors at once.** Their floor patches overlap: 0.65 of their union on the 75 cm foot path within 0.8 m of the
  toes (a; the sensor screen's near strip). The far rows lie beyond that strip, 0.98–1.07 m ahead, and their overlap
  was not computed. ST's forum answers say sensors set at different angles interfere little (d; forum answers, not a
  specification). If they had to range in turn, each frame period would double and the governor would fall by 0.13
  m/s.
- **The trunk's pitch while walking** was not read from the closed-loop records. The spec's limits stand in for it.
- **The walking height** is the reference walks' own, 0.03 H below the home pose. A controller that walks lower sees
  less far: the 0.015 m between the two heights is worth 0.03 m/s here.
- **Stops at speed.** No stop reference exists from a walk faster than the nominal one, and no closed-loop stop
  distance was read. The step readings are the bar's and the spec's targets, not stops the robot has made, and the
  stop builder's 2.4 steps are applied at speeds it was not built for.
- **The stop's start is idealised.** The deceleration readings brake from the first instant at any gait phase. The
  foot-lead floor covers a foot already on the ground. It does not cover a swing foot that still has to land; the step
  readings stand for that.
- **Turning, slopes and moving people** are outside this document. So are hazards the camera sees first, which give
  more room.

## 10. For the delay bench sheet (decision-log row A17)

Task 8 of the review asks the sheet to carry the ToF stop's timing, and the bench tests run first, on both paths,
before any compute purchase (P30). Six readings would replace this document's assumptions:

- Readings 1, 3, 5 and 6 replace inputs of `tof_governor.json`: `tof_frames`; `i2c_read`; `host_planner`; and
  `rt_core_loop`, `rt_core_walk_loop` and `host_link`.
- Reading 2 replaces the computed sight distance (`bodies.*.horizons` in the record), which is not one input. It also
  tests the assumed 4x4 layout (`tof_4x4_layout`).
- Reading 4 settles the "sensors range in turn" what-if.
- The pitch axis (`pitch_axis`) is not a bench reading; the closed-loop panels answer it.

1. **Frames to first report** (`tof_frames`). Slide a 0.10 m box face, and then a 0.10 m drop edge, across a marked
   line under one sensor at the chest's height and aim. Log the time from the crossing to the first frame that reports
   it, at 8x8 15 Hz, 4x4 60 Hz and 4x4 30 Hz.
2. **Where it is first reported** (the sight distance). The same, logging the distance, at each mode and rate, on
   samples of the owner's floors set at the far row's 23–28° to the beam, with the sensor level and pitched 4° and 8°
   nose-down, in room light and near a window.
3. **I2C read time** (`i2c_read`). Per frame, for the outputs the stop needs, both sensors on one bus, at 400 kHz and
   1 MHz, on the microcontroller and on the Jetson's header.
4. **Both sensors ranging together.** Count invalid or extra targets in the overlap.
5. **Planner time, if path 3 is ever used** (`host_planner`). Sample to motion request, on each host.
6. **The real-time core's loops and the host link** (`rt_core_loop`, `rt_core_walk_loop`, `host_link`). The period and
   jitter of the stop loop and of the walking loop on the Jetson's isolated core, and the delay of the hop from the
   host to the real-time layer.

From simulation, with no hardware: read the trunk's pitch from the closed-loop panels, and the stop distance once the
two-step stop reference exists (review tasks 2 to 4).

## 11. Inputs, with kinds and sources

| Input | Kind | Value | Unit | Source; for (d), whose and how derived |
|---|---|---|---|---|
| `75-M-H-10.tof_site` | (a) | {"site_pos_m": [0.2851, 0.0151, 0.4272], "axis": [0.6917, 0.147, -0.7071], "axis_depression_deg": 45.0} | m | round2/refine/feet_75/review/screening/75-M-H-10_sensors.json tof.tof_l.summary (tof_r mirrored) |
| `75-M-H-10.screen_far_floor` | (a) | 1.11 | m | round2/refine/feet_75/review/screening/75-M-H-10_sensors.json summary.tof.tof_l.floor_ahead_of_toes_m[1] |
| `75-M-H-10.toe_tip_x` | (a) | 0.1662 | m | round2/refine/feet_75/review/screening/75-M-H-10_sensors.json model.toe_tip_x_m |
| `75-M-H-10.foot_path_half_width` | (a) | 0.2067 | m | round2/refine/feet_75/review/screening/75-M-H-10_sensors.json tof.near_coverage.foot_path_half_width_m |
| `75-M-H-10.hip_height_screen` | (a) | 0.4998 | m | round2/refine/feet_75/review/screening/75-M-H-10_sensors.json model.hip_height_m |
| `75-M-H-10.H` | (a) | 0.5018535496646683 | m | round2/refine/feet_75/candidates/75-M-H-10.json H |
| `75-M-H-10.one_foot_decel` | (a) | 1.4137222751065168 | m/s^2 | round2/refine/feet_75/review/screening/75-M-H-10_envelope.json stopping.max_one_foot_decel_ms2 (designlab/envelope.py; DECISION_LOG.md F12, F23) |
| `75-M-H-10.stop_reference` | (a) | {"ramps_path": [0.0, 0.4], "speed_m_s": 0.24306007460404672, "period_s": 1.1356017756730012, "crouch_m": 0.01505560648994005} | - | stage1/runs/2026-11-08/75-M-H-10/transitions.json transitions.stop (ramps, gaits.a, steps) |
| `60-N-T-10.tof_site` | (a) | {"site_pos_m": [0.2273, 0.0121, 0.3546], "axis": [0.6917, 0.147, -0.7071], "axis_depression_deg": 45.0} | m | round2/refine/narrow_carpet_60/review/screening/60-N-T-10_sensors.json tof.tof_l.summary (tof_r mirrored) |
| `60-N-T-10.screen_far_floor` | (a) | 0.9428 | m | round2/refine/narrow_carpet_60/review/screening/60-N-T-10_sensors.json summary.tof.tof_l.floor_ahead_of_toes_m[1] |
| `60-N-T-10.toe_tip_x` | (a) | 0.1071 | m | round2/refine/narrow_carpet_60/review/screening/60-N-T-10_sensors.json model.toe_tip_x_m |
| `60-N-T-10.foot_path_half_width` | (a) | 0.1728 | m | round2/refine/narrow_carpet_60/review/screening/60-N-T-10_sensors.json tof.near_coverage.foot_path_half_width_m |
| `60-N-T-10.hip_height_screen` | (a) | 0.4004 | m | round2/refine/narrow_carpet_60/review/screening/60-N-T-10_sensors.json model.hip_height_m |
| `60-N-T-10.H` | (a) | 0.40468919988052376 | m | round2/refine/narrow_carpet_60/candidates/60-N-T-10.json H |
| `60-N-T-10.one_foot_decel` | (a) | 1.2498989107790124 | m/s^2 | round2/refine/narrow_carpet_60/review/screening/60-N-T-10_envelope.json stopping.max_one_foot_decel_ms2 (designlab/envelope.py; DECISION_LOG.md F12, F23) |
| `60-N-T-10.stop_reference` | (a) | {"ramps_path": [0.0, 0.4], "speed_m_s": 0.21826592177876797, "period_s": 1.019760932537567, "crouch_m": 0.012140675996415712} | - | stage1/runs/2026-11-08/60-N-T-10/transitions.json transitions.stop (ramps, gaits.a, steps) |
| `tof_part` | (c) | st_vl53l5cx_carrier | - | designlab/data/parts.json id st_vl53l5cx_carrier (the records' selection.tof) |
| `tof_fov` | (c) | {"hfov_deg": 45.0, "vfov_deg": 45.0, "dfov_deg": 65.0, "range_m": [0.02, 4.0]} | deg | designlab/data/parts.json id st_vl53l5cx_carrier, fov; designlab/sensors.py SPECS. The datasheet's revision 2 said 63 deg on the diagonal and revision 13 says 65 deg (web sources st_vl53l5cx_datasheet, st_an5894); the screen's equidistant lens gives 63.6 deg |
| `tof_4x4_layout` | (d) | same field, zones twice as wide | - | This document's: the 4x4 zones tile the same 45 x 45 deg field, each one four of the 8x8 zones; the catalog says only '4x4 or 8x8 zones' |
| `tof_mount_aim` | (a) | {"down_deg": 45.0, "outward_deg": 12.0} | deg | designlab/build.py (two ToF sensors under the chest); DECISION_LOG.md F4 |
| `tof_rate_8x8` | (c) | 15.0 | Hz | ST UM2884 table 2 (web source st_um2884): 8x8 ranges at 1 to 15 Hz |
| `tof_rate_4x4` | (c) | low 1 / high 60 | Hz | ST UM2884 table 2 (web source st_um2884): 4x4 ranges at 1 to 60 Hz |
| `tof_rate_4x4_table` | (d) | 30.0 | Hz | This document's choice of a second 4x4 rate, allowed by UM2884's 1 to 60 Hz: the review's search extract has the datasheet's 4x4 range table at 30 Hz; this document's own searches did not return the table |
| `tof_range_ambient` | (c) | 1.7 | m | designlab/data/parts.json id pololu_vl53l8cx, fov note; web source pololu_3419: in 5000 lux about 1.7 m for the VL53L5CX, for a large reflective target, with no resolution or rate given. The vendor's own range tables, per resolution and rate, were not read |
| `i2c_rate` | (c) | low 400000 / high 1000000 | Hz | ST VL53L5CX datasheet (web source st_vl53l5cx_datasheet) |
| `i2c_packet_bytes` | (d) | {"4x4": [124, 1008], "8x8": [316, 3360]} | bytes | web source st_i2c_bandwidth_table. This document's: the smallest and largest packets in ST's table. Read with their rows: 124 (4x4, one target, three outputs), 1008 (4x4, every output), 3360 (8x8, every output); 316 is taken as the 8x8 counterpart of 124. That the stop needs only the smallest set of outputs is assumed |
| `i2c_read` | (d) | {"8x8_15hz": [0.002844, 0.008532, 0.01422], "4x4_60hz": [0.001116, 0.003348, 0.00558], "4x4_30hz": [0.001116, 0.003348, 0.00558]} | s | This document's: bytes x 9 bits / bus rate, the smallest packet; low = one sensor at 1 MHz, high = both sensors read back to back at 400 kHz |
| `tof_frames` | (d) | low 1 / ref 2 / high 3 | frames | This document's: 1 = one frame at the best phase, 2 = one frame at the worst phase, 3 = a two-frame confirmation at the worst phase; the firmware is not written |
| `mcu_loop` | (c) | low 200 / high 1000 | Hz | COMPUTE.md, 'What the onboard computer has to do': CAN loop, IMU, ToF stop, watchdogs at 200-1000 Hz on the microcontroller |
| `rt_core_loop` | (d) | low 200 / high 1000 | Hz | This document's: taken equal to the microcontroller's 200-1000 Hz; no document of the study fixes it and PR #668 (unverified) gives no measured period |
| `walk_loop` | (c) | low 50 / high 100 | Hz | COMPUTE.md, same table: locomotion policy at 50-100 Hz, on the microcontroller (stop paths 1 and 3) |
| `rt_core_walk_loop` | (d) | low 50 / high 100 | Hz | This document's: on stop path 2 the walking loop runs on the Jetson's real-time core; its rate is taken equal to COMPUTE.md's 50-100 Hz for the microcontroller, which no document of the study fixes for the Jetson |
| `planner_rate` | (c) | low 10 / high 15 | Hz | COMPUTE.md, same table: ground-frame tracking, obstacle grid, planner at 10-15 Hz. Not used in the ranges; given for scale (one period is 67-100 ms) |
| `host_deadline` | (c) | 0.2 | s | BEHAVIOR_SPEC.md section 3, Compute timing: at most 200 ms from the oldest sensor sample to a motion request. A deadline (a target), not a measured latency |
| `host_planner` | (d) | low 0 / high 0.2 | s | This document's: low = 0, a bound (a planner at its best phase waits almost nothing, and no compute time is sourced); high = the spec's deadline, assumed met. No latency has been measured on the Pi 5 or the Jetson, so both hosts take the same range |
| `host_link` | (d) | low 0 / high 0.005 | s | This document's: a serial or USB hop from the host to the real-time layer |
| `servo_delay` | (d) | low 0 / ref 0.01 / high 0.02 | s | DECISION_LOG.md P29 (and the owner's 14 Nov decision, P30): the bar is judged at 10 ms until a bench value exists; A17. The study's assumption: reference 10 ms, range 0-20 ms as Stage 1 simulated (LESSONS_LEARNED.md 2.3) |
| `spec_decel` | (c) | 1.5 | m/s^2 | BEHAVIOR_SPEC.md section 1, Start / stop: peak horizontal trunk acceleration at most 1.5 m/s^2; STAGE1_PLAN.md decision 10; STAGE2_PLAN.md section 5 gates it in the bar's stop; designlab/envelope.py STOP_DECEL. A limit (a target), not a deceleration the robot has shown |
| `stop_steps` | (c) | low 1 / high 2 | steps | STAGE2_PLAN.md section 5, the bar: stop within a step or two (at most two touchdowns after the stop command). A target the bar sets, not a stop the robot has shown |
| `spec_stop_strides` | (c) | 1.5 | strides | BEHAVIOR_SPEC.md section 1, Start / stop: stop within 1.5 strides; STAGE1_PLAN.md decision 18. A target, not a stop the robot has shown |
| `reference_walks` | (a) | period and duty linear in speed between the three walks | - | designlab/sizing.py NOMINAL, BRISK; designlab/envelope.py FAST; designlab/stage2/look.py speed_spec |
| `stride_extrapolation` | (d) | flagged where used | - | This document's: outside the three walks the nearest walk's period and duty are held |
| `foot_lead` | (a) | duty / 2 | stride | designlab/gait.py _foot_x: a stance foot is placed duty x stride / 2 ahead of its home position, so it lands that far ahead of the home-stance toe line |
| `bar_speed` | (c) | 0.4 | m/s | STAGE2_PLAN.md section 5; DECISION_LOG.md P29 |
| `speed_ladder` | (c) | [0.3, 0.35, 0.4, 0.45, 0.5] | m/s | STAGE2_PLAN.md section 5, Ladders |
| `trunk_pitch_limits` | (c) | [4.0, 8.0] | deg | BEHAVIOR_SPEC.md section 1, Body motion: at most 4 deg RMS, 8 deg peak |
| `pitch_axis` | (d) | hip axis | - | This document's: the trunk is pitched about the hip axis, taken at x = 0 and the screen record's hip height |
| `crouch` | (a) | 0.03 | H | designlab/gait.py GaitSpec.crouch default: every reference walk holds the trunk this far below the home height, so the main table's sight distance is taken there |
| `standoff` | (d) | 0.1 | m | This document's what-if: a gap kept between the toes and the hazard; the main table keeps none |
| `obstacle_height` | (c) | 0.1 | m | BEHAVIOR_SPEC.md section 3: no contact with static obstacles at least 0.10 m tall |

## 12. Web sources

Searched on 14 Nov 2026 with generic part numbers and document names only. The vendor's site (www.st.com) and the
distributors' file hosts (cdn.sparkfun.com, static6.arrow.com) could not be opened from this session, so each source
below was read through the search tool's extract of the page, not the whole document. The frame-rate limits came back
the same from several searches, with the table number. The range tables did not come back at all, and one packet-size
row came back without its configuration; both are marked.

- `st_um2884`: https://st.com/resource/en/user_manual/um2884-a-guide-to-using-the-vl53l5cx-multizone-timeofflight-ranging-sensor-with-a-wide-field-of-view-ultra-lite-driver-uld-stmicroelectronics.pdf (also https://core-electronics.com.au/attachments/localcontent/um2884-a-guide_75701dc7b2e.pdf)
  Says: ST user manual UM2884 for the VL53L5CX driver. Table 2, minimum and maximum ranging frequencies: 4x4 1 to 60
  Hz, 8x8 1 to 15 Hz; the default is 1 Hz; the resolution is set before the frequency. Continuous mode keeps the
  emitter on, so range and ambient immunity are better; autonomous mode is the default and takes an integration time.
  Changing the resolution also changes the I2C traffic when results are read. Read: through the web search tool's
  extract on 14 Nov 2026; the vendor's site could not be opened directly from this session.
- `st_vl53l5cx_datasheet`: https://static6.arrow.com/aropdfconversion/65108bc7ab8c2686fe29d25aca394c39e7548213/vl53l5cx.pdf (also https://cdn.sparkfun.com/assets/6/e/3/0/6/vl53l5cx-datasheet.pdf)
  Says: ST VL53L5CX datasheet, DS13754 (distributors' copies). I2C at 400 kHz to 1 MHz, address 0x52; up to 400 cm
  ranging; the integration time applies in autonomous mode only; the detection volume depends on reflectance, ambient
  light, resolution, sharpener, ranging mode and integration time. Field of view: revision 2 says 63 deg diagonal;
  revision 13 lists 45 deg horizontal, 45 deg vertical and 65 deg diagonal. The field of view is measured with a white
  88 % target square to the sensor at 1 m, in the dark, at 8x8. Read: through the web search tool's extract on 14 Nov
  2026; the maximum-range tables (17 and 18) were not in any extract, and the file hosts could not be opened directly.
- `st_an5894`: https://www.st.com/resource/en/application_note/an5894-description-of-the-fields-of-view-of-stmicroelectronics-timeofflight-sensors-stmicroelectronics.pdf
  Says: ST application note AN5894 on the fields of view of its ToF sensors: for the VL53L5CX the field is given on
  the diagonal (65 deg) and by side (45 x 45 deg), and the side cannot be converted to the diagonal with Pythagoras'
  theorem. Read: through the web search tool's extract on 14 Nov 2026.
- `st_range_tables_thread`: https://community.st.com/imaging-sensors-49/vl53l5cx-datasheet-table-17-18-definition-of-typical-and-minimum-values-4794
  Says: An ST community question on how to read 'typical' and 'minimum' in the VL53L5CX datasheet's tables 17 and 18
  of maximum range capability (table 18 is on page 20). It shows that the tables exist; it gives no value, and the
  extract held no answer. Read: through the web search tool's extract on 14 Nov 2026; used only to name the tables
  that were not read.
- `st_tof_presentation_2023`: https://st.com/resource/en/application_presentation/oct24th-2023-latest-time-of-flight-sensor-solutions-and-applications.pdf
  Says: An ST presentation of October 2023 lists the VL53L5CX at 400 cm in the dark and 170 cm in ambient light, with
  no target, resolution or rate beside the figures. Read: through the web search tool's extract on 14 Nov 2026.
- `pololu_3419`: https://www.pololu.com/product/3419
  Says: Pololu's page for its VL53L8CX carrier: in 5000 lux the VL53L8CX is expected to detect a large reflective
  target out to about 2.8 m, against about 1.7 m for the VL53L5CX; the 4 m maximum of both assumes an ideal dark
  environment; effective range and update rate depend on the configuration, the target and the environment. No
  resolution or rate is given with the 1.7 m. Read: through the web search tool's extract on 14 Nov 2026; the catalog
  note on pololu_vl53l8cx cites the same page.
- `st_i2c_bandwidth_table`: https://manuals.plus/m/4fea81abf81933d02393c56515009feef4d6015b81bc2f2bccf14262e8fcc3b4.pdf
  Says: A copy of an ST guide for the VL53L5CX, which the search tool names UM2887 revision 1, the software
  integration guide. Its table of I2C bandwidth during ranging lists packet sizes of 124, 316, 1008 and 3360 bytes and
  bandwidths of 124, 316, 63000 and 50909 bytes per second. Three rows were read with their configuration: 1 Hz, 4x4,
  one target per zone, distance, target status and number of targets only: 124 bytes; 60 Hz, 4x4, four targets per
  zone, all output enabled: 1008 bytes, 63000 bytes per second; 15 Hz, 8x8, four targets per zone, all output enabled:
  3360 bytes, 50909 bytes per second. The 316-byte row did not come back with its configuration. Read: through the web
  search tool's extract on 14 Nov 2026; the 316-byte row is taken to be the 124-byte row's outputs at 8x8, which is
  this document's reading (d).
- `st_um3109`: https://www.pololu.com/file/0J2030/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf
  Says: ST user manual UM3109 for the VL53L8CX (the catalog's newer part, pololu_vl53l8cx): the same limits, 4x4 1 to
  60 Hz and 8x8 1 to 15 Hz. Read: through the web search tool's extract on 14 Nov 2026.
- `st_forum_multi_sensor`: https://community.st.com/imaging-sensors-49/vl53l5cx-mutual-influence-of-multiple-sensors-34808
  Says: ST community answers by an ST employee: several VL53L5CX sensors interfere a little (an extra target with an
  invalid status); ST tested sensors set at different angles and found the effect minimal. Forum answers, not a
  specification. Read: through the web search tool's extract on 14 Nov 2026; used only as a note, kind (d).

## 13. Draft decision-log rows (not adopted; ids for the owner to assign)

| Draft | Kind | Statement | Evidence | Status |
|---|---|---|---|---|
| ToF governor | F | By the stopping-horizon method (sight distance on the foot path at the walking height, less the distance covered while reacting, at least the stopping distance), the chest ToF pair does not limit autonomous walking speed inside the study's reference walks or its speed ladder. Over three ToF modes, three stop paths and six stop readings the lowest governor is 0.57 m/s at 75 cm and 0.53 m/s at 60 cm (the spec's 1.5-stride stop through the host's planner). With the bar's two-step stop and 8x8 zones it is 0.92–1.21 m/s at 75 cm and 0.85–1.13 m/s at 60 cm when the real-time layer decides the stop, and 0.77–1.21 and 0.70–1.13 m/s through the host's planner; at the bar's 0.40 m/s at least 0.41 / 0.32 m of sight distance is left with two steps and 0.23 / 0.17 m with 1.5 strides. No sourced timing separates the microcontroller from the Jetson's real-time core, or the Pi 5 from the Jetson as host. With every adverse reading stacked the 75 cm governor is 0.40–0.46 m/s with a two-step stop decided by the real-time layer, 0.20–0.39 m/s with a two-step stop through the host or a longer stop on any path, and 0.53–0.96 m/s with one step or a stop as hard as the spec allows. Static and kinematic model results (a) on assumed timing (d); not a speed shown. | `stage2/jetson75/TOF_GOVERNOR.md`, `tof_governor.json` | draft, open (information for the compute choice; the bench replaces its assumptions) |
| ToF frame rates | E | The VL53L5CX ranges at up to 15 Hz with 8x8 zones and up to 60 Hz with 4x4 zones (ST UM2884, table 2); the VL53L8CX has the same limits (ST UM3109). This would replace the catalog note that calls the 15 Hz figure unverified. It was read through a search extract of the manuals; confirm against the documents before the catalog changes. `tof_governor.json` carries a checksum of the two ToF catalog entries, so it is regenerated with that change. | section 12 | draft |
| ToF stop timing | A | Until the bench: 1–3 ToF frames to detect; an I2C read of 1.1–5.6 ms (4x4) or 2.8–14.2 ms (8x8); the real-time core's stop loop and walking loop equal to the microcontroller's 200–1000 Hz and 50–100 Hz; 0–200 ms from sample to motion request through the host's planner, the same for either host; the sight distance at the reference walks' height; no gap kept between the toes and the hazard. | section 3 | draft; replace with the delay bench sheet's ToF readings (section 10) |

## Reproduce

From `hardware/trex_form_study`, with the study's Python environment:

    MUJOCO_GL=disable PYTHONPATH=. python stage2/jetson75/tof_governor.py        # writes tof_governor.json
    MUJOCO_GL=disable PYTHONPATH=. python stage2/jetson75/tof_governor.py md     # prints this document's tables
    MUJOCO_GL=disable PYTHONPATH=. python -m pytest -p no:cacheprovider tests/test_stage2_tof_governor.py
