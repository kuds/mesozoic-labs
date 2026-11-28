# Readiness review: will the 60 cm design work as a real robot?

27 Nov 2026, build commit aa150f89. **A review of the records, not a judged result:** no pre-registration covers it.
- **The design reviewed** is the adopted one (D25, D41): `sim/generated/trike60_C_closed_capsule_fore128.xml`,
  12.751 kg, with option C's drives (10 X4 + 2 X6 on 12S), the capsule sole and the 128 mm upper foreleg.
- **Nothing is built, bought, weighed or measured** (`README.md:4-5`). Every figure below is either simulation or
  desk work (an estimate, a datasheet figure or a calculation). None is measured. Published work on other robots is
  context, not evidence about this one.
- **How it was checked:** every claim of the day's working review was checked against the lines it cites and its
  arithmetic redone. The heat at 35 C was rerun. What changed is listed in section 7.
- **After the review,** the owner decided D45-D48 (last section). Work on them started the same day ("This batch",
  at the end).

**Citations.** Every file:line below is at the commit that adds this record, the last of the same day's batch (listed at
the end), unless it names another commit (aa150f89 is the build before the batch). Files of the batch are marked
"working tree" where cited: they were uncommitted when this was written and are committed with it. Paths are relative to
this folder unless they start with `hardware/` or `docs/` (the repository root). A bare `:NN` is a line of the file
named last before it in the same paragraph, list or table. Short names:
- "the model": `sim/generated/trike60_C_closed_capsule_fore128.xml`;
- "the D22 report" and "the appendix": `rl/analysis/REPORT_FORELEG_D22.md` and
  `rl/analysis/REPORT_FORELEG_D22_APPENDIX.md`;
- "the summaries": `rl/robustness/results/fore128-s<seed>/summary_fore128-s<seed>.md`;
- "the bench plan": `bench/BENCH_PLAN_FORELEG.md` (18 Nov 2026);
- "the kickoff list": `hardware/triceratops_kickoff/PARTS_PREMIUM.md`, the 75 cm kickoff's parts list (18 Nov 2026);
- "the catalog": `hardware/trex_form_study/designlab/data/actuators.json`, maker datasheets and store pages accessed
  7 Nov 2026;
- "main's lessons": `docs/hardware/LESSONS_LEARNED.md` at main commit 2c1039f4;
- "the 21 Nov review", cited as `REVIEW_60CM.md`: `hardware/triceratops_build/REVIEW_60CM.md` at commit 2ac57608
  (branch `review/triceratops-60cm-review`).

**Prices** are desk figures: store pages, list prices, and the parts-list and budget estimates of 18 Nov 2026
(`budget/size_budget_out.txt:1`, `design/PARTS_60CM.md:3`). Each carries its source and the date it was read.

## The answer

The owner's expectation is that the 60 cm design should work with some fine-tuning. The records support it
**partly.**
- **Supported, in simulation, for straight walking on a flat floor.** All four 128 mm policies walk at the 0.25 m/s
  command without falling. With one factor changed at a time, they keep walking through wide changes in delay,
  mass, drive strength, gains and friction.
- **Not yet supported for the rest of a demo.** Three things a flat-floor demo needs are thin, failing or untested,
  even in simulation:
  - **heat margin:** thin in a 35 C room (D48), where two of four demo loops are above B3's 90 C figure (a comparison
    only, desk model; section 2), and untested inside a shell, which no model covers;
  - **turning:** no controller in this build has turned;
  - **sideways shoves:** the learned policies fail the push test on all four seeds. P8 keeps people away from the
    robot, so a shove stands in for a stumble or a bump, not a demo item.
- **Both controllers read signals the robot will not have.** The learned policies read the simulator's true body
  velocity, the foot loads and the passive ankle angles: 11 of their 68 inputs. The model-based trot reads the foot
  loads for its velocity estimate.
- **Where "fine-tuning" fits.** It fits most torque, speed, gain and friction questions (section 5). It understates
  heat and sensing. It also leaves out work not yet done: CAD and the chassis (D5), firmware, the stop chain's sit and
  the battery cut-off (D47).

**Confidence.** This is a judgement, not a result.
- Moderate that a robot of this design can be made to walk flat-floor demos.
- Low that it will do so without a retrain on the robot's own sensors, plus some change for heat in the hardware or
  in the demo's schedule.

**What that rests on:**
- **An idealised model.** Its position servos run on assumed gains (A2). Every hinge joint has viscous damping and
  armature, but there is no dry friction (no `frictionloss` anywhere in the model), no backlash and no torque-speed
  curve:
  - damping 0.14 N·m·s/rad by default, which the roll and pitch drives use; 0.20 on the flexion drives; 0.04-0.09 on
    the neck, tail and passive ankles (model lines 40, 148, 193, 199);
  - armature 0.002 kg·m² by default, which no hinge joint keeps: the drive joints carry the drive's reflected
    armature, 0.00324 kg·m² on the ten X4s and 0.004304 on the two X6s (model lines 180-287; the catalog's estimate,
    `sim/data/drives.json:2`, `:9`, `:18`), and the neck, tail and ankles 0.0002.

  The drive ratings are open-air figures from the maker's temperature-rise tables
  (`budget/SIZE_BUDGET_OPTIONS.md:80`, 18 Nov 2026; tables read 7 Nov 2026, `sim/data/drives.json:2`).
- **An estimated mass.** 6.02 of the 12.75 kg (47 %) is estimated and none is weighed. The rest is catalog or reseller
  figures, summed over the `basis` fields of `sim/generated/mass_ledger_trike60_C_closed_capsule_fore128.json`.
  The total includes about 1.08 kg of reserve: 0.95 kg of growth (x1.2088 on the estimated parts) and a 0.138 kg block.
- **Outside work on robots with very different drives.**

## 1. What simulation and desk work show

**Straight flat walking (simulation, four seeds).**
- Speeds of 0.235 / 0.244 / 0.242 / 0.240 m/s at the 0.25 m/s command, with 0 falls in 5 episodes each
  (`rl/analysis/REPORT_FORELEG_D22.md:11-15`, `:135`).
- Conditions: randomised model and floor, 10 ms delay, sensor noise on.
- Seed 0 clears the 0.225 m/s rule by 0.010 m/s.

**Falls are rare when one factor changes at a time (simulation).**
- The sweep's design: 8 episodes per point, no randomisation, and its break rule counts falls and speed only
  (`rl/robustness/README.md:111-123`).
- No break on any seed, at 0.25 m/s, for (the summaries `:13`):
  - friction 0.15-1.2;
  - every body x0.8-1.6, or 0-3 kg of payload;
  - force limit x0.4-1, or kp x0.4-1.5.
- Delay at 0.25 m/s: no break up to 60 ms on seeds 1 and 2. Seeds 0 and 3 first break at 40 ms: seed 0 slows to
  0.086 m/s, and 1 of 8 seed 3 episodes fall (the summaries, s0 and s3, `:14`).
- What "no fall" leaves out:
  - At force x0.4 the worst joint sits at its reduced limit 28.8-58.9 % of the time, and seed 0 drifts 1.20 m.
  - At kp x1.2 two seeds run over their rating (1.03, 1.09); at x1.5 three do (1.03, 1.13, 1.21).
  - At kp x0.4 seed 3 drifts 1.52 m.
  - All from each summary's force-limit and kp tables.

**Heat is the binding limit, and its margin is thin (simulation torques).** Worst drive, RMS torque over its
continuous rating, at 0.25 m/s (`rl/analysis/REPORT_FORELEG_D22.md:167-170`; stress rows from the
appendix and the summaries):

| Case | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| Judged case (randomised, noise on) | 0.828 | 0.888 | 0.931 | 0.945 |
| Nominal model, noise off | 0.878 | 0.989 | 1.005 | 0.966 |
| Demo course, walking segment | 0.87 | 0.90 | 1.00 | 1.01 |
| 15 ms delay (appendix `:186`) | 0.87 | 1.02 | 0.98 | 1.00 |
| 20 ms delay, no falls (appendix `:187`) | 1.07 | 1.13 | 1.20 | 1.20 |
| Every body x1.1 (appendix `:228`) | 0.94 | 1.01 | 1.02 | 1.06 |
| 2 kg payload (summaries `:114`, `:113`, `:113`, `:116`) | 1.02 | 1.02 | 1.02 | 1.11 |

- **Which drive runs hottest.** In the judged case it is a pitch drive on 3 of 4 seeds: the fore pitch on s2, the hind
  pitch on s0 and s3. On s1 it is a hind knee (appendix `:76-86`).
- **Worst episode.** Seed 3's reaches 1.00 (`rl/analytics/results/fore128-s3/gait_report.md:72`).
- **The report's own reading:** with other seeds or settings either length "could land on either side"
  (the D22 report `:387-389`).
- **Walking slower buys no delay margin.** At 0.15 m/s with 20-30 ms of delay the worst drive is at 1.10-1.33
  (the D22 report `:193-195`).

**Demo heat (simulation torques; desk thermal model `rl/telemetry/thermal.py`, one node per drive, open air).**
- The demo loop is 90 s walking at 0.25 m/s and 30 s standing, repeated for one pack, starting from 25 C. The hottest
  winding peaks at 62.9-85.9 C with the 8 min time constant, and 62.5-87.0 C over 4-15 min (the D22 report
  `:209-211`; appendix `:153-154`).
- All four policies stand on four feet, the worst standing drive at 0.58-0.69 of rating (the D22 report `:219`).
- Walking with no breaks: 99.6 C (s2) and 101.2 C (s3) (the D22 report `:221-224`). The margin rests on the standing
  breaks.
- At 35 C: section 2.

**A second controller works on the adopted model (simulation).**
- **What it uses.** The model-based trot (`walk/walk60.py`) estimates body velocity from the stance legs, not the
  simulator (`:9-12`). It reads the foot loads to pick which legs enter that leg-odometry velocity estimate (`:67`,
  `:239`). Its step timing is fixed (`:15`) and its load-triggered early touchdown is off (`:84`).
- **On the 128 mm model,** untuned and with no randomisation, seeds 101-110: 0 falls in 10, 0.250-0.252 m/s, worst
  drive 0.66-0.70 of rating (`renders/README.md:53-66`).
- **Its sweep was on the older 188 mm roll-ankle model:** no break up to 60 ms of delay, no fall at +-1.25 m/s pushes
  (`walk/REPORT_WALK60.md:126`, `:131`). It has not been swept on the adopted model (D45 puts that first).

**The foot (simulation).** The capsule sole is the only one of four feet that passes every registered rule under both
contact settings (`sim/REPORT_FEET60.md:11-13`). Three limits:
- it was tested on the 188 mm model;
- the replication is weak (`:140-141`);
- the simulated sole is a full rigid shape, where a real part is the lower half only, a rocker pad (`:110-112`).

**Power and bus (desk unless marked).**
- **Bus load:** 33-41 % at 500 Hz with one bus of three drives per leg. This is the 75 cm kickoff's arithmetic, for
  the same count of twelve drives, three per bus, at 500 Hz (the kickoff list `:280-281`). Its drives are other types
  (X8-150 and X6-60 flexion drives, `:12`).
- **Delay:** 2-4 ms on the Teensy path (2-6 ms read cautiously), "every term of it unmeasured"; up to 12 ms on a
  Jetson with one USB-CAN channel (main's lessons `:89-91`).
- **Peak pack current:** 26.6 A (simulation, seed 3;
  `rl/analysis/results/d22/demo/fore128-s3/demo_course_printout.txt:8`), against the kickoff's 100 A main fuse (a
  design rating, the kickoff list `:81`).
- **Runtime per pack:** 84-96 min on the demo loop at a 30 W hotel load, and 75-84 min at 40 W (simulation power,
  desk battery; appendix `:160`). The budget's desk rule gives 43-60 min (`budget/size_budget_out.txt:54`), 0.45-0.71
  of the demo figure.

**Packaging (desk, from the model's geometry; collisions between these parts are off, so the simulation never shows
them).**
- **The 128 mm forelegs fix F18:** each X6 elbow housing clears its upper strut by 8.7 mm, on both sides.
- **New: the hind knee drive sits inside its own strut.** Each hind X4 knee housing (`*_hind_x4_flexion_stator`) is
  centred on its upper strut's centreline (model lines 250-251) and overlaps the strut by 38.7 mm, on both sides.
  - Both figures are MuJoCo's `mj_geomDistance` on the model (MuJoCo 3.10.0), run for this record on 27 Nov 2026:
    +8.70 mm and -38.70 mm. Each pair is in one body, so the figure is the same in every pose. The batch's packaging
    check gives the same (`sim/results/packaging60_2026-11-27/packaging60_report.txt:19`, `:42`, working tree).
  - It is the kind of overlap F18 found at the front (44.9 mm). No record reports it, and `sim/build_foreleg60.py
    --packaging` checks the forelegs only.
- **The crossmembers still pass through the drives:** at the shoulders (F19) and at the hips (F21).

**Outside work (context; checked at search-summary level only, arxiv.org cannot be reached from this session).**
- **Mini Cheetah** (MIT's 9 kg robot) sustained 3.9 m/s on flat ground with a policy trained in simulation (arXiv
  2205.02824; search summary, 27 Nov 2026). Its drives' ratio, about 6:1, is unchecked (search summary, 28 Nov 2026).
  This build's X4 is 36:1 and its X6 19.6:1 (maker datasheets read 7 Nov 2026; `sim/data/drives.json:6`, `:14`).
- **Not rechecked today:** the working review's further papers (arXiv 2202.05481, 2204.03897, 2410.08650,
  2503.01255) and its GitHub search for this drive line.

## 2. Heat in a 35 C room: two methods, one consistent

**The rerun.** The build's thermal model was rerun on the four 128 mm policies' stored demo torques
(`rl/analysis/results/d22/demo/fore128-s<seed>/demo_course.json` and the `demo_walkonly` set). Only the room
temperature was changed, with the 8 min time constant.
- **Inputs:** Kt and phase resistance from the catalog (maker datasheets accessed 7 Nov 2026; read through
  `rl/analytics/gait_report.py:121-136`): X4-36-L 1.8 N·m/A, 0.2 ohm (`myactuator_rmd_x4_p36_36_e_l`); X6-60
  2.1 N·m/A, 0.205 ohm (`myactuator_rmd_x6_p20_60_e`). The thermal-equilibrium torques, 7.5 N·m (X4-36-L) and
  12.0 N·m (the X6-50-L's table standing in for the X6-60), are the maker's temperature-rise tables, read 7 Nov 2026
  (`sim/data/drives.json:6`, `:14-15`).
- **Check:** at 25 C the rerun reproduces the stored peaks exactly: 64.7 / 62.9 / 81.1 / 85.9 C (appendix `:153`).

Hottest winding, demo loop, C (desk model, simulated torques):

| Room | Method | s0 (left hind pitch) | s1 (left fore pitch) | s2 (left fore pitch) | s3 (left hind pitch) |
|---|---|---|---|---|---|
| 25 C | stored | 64.7 | 62.9 | 81.1 | 85.9 |
| 35 C | **A:** the drive as fitted at 25 C, room raised | **76.3** | **74.4** | **93.3** | **98.3** |
| 35 C | **B:** refitted so the maker's figure holds from 35 C | 70.1 | 68.5 | 84.1 | 88.2 |
| 40 C | A | 82.1 | 80.1 | 99.4 | 104.5 |

Walking with no breaks (s2, s3):
- **25 C:** 99.6 and 101.2 C.
- **35 C:** 112.5 and 114.2 C under A; 99.6 and 101.0 C under B.

**A is the physically consistent method.**
- The model's thermal resistance and heat capacity (R_th and C) describe the drive: its winding, its housing and how
  its surface sheds heat to the air. To first order they do not change when the room warms.
- They are fitted to the maker's temperature-rise test, run in one room. The X4-36-L's table starts at about
  26-27 C (`hardware/trex_form_study/designlab/data/actuators.json`, X4-36-L `vendor_rated_definition`); the model
  takes 25 C (`rl/telemetry/thermal.py:10-12`).
- B refits as if that test had been run from 35 C. It then asks the drive to reach the same 100 C after 90 min from a
  10 K warmer start. That needs 13 % lower thermal resistance (X4-36-L: 4.82 against 5.56 K/W, for a 65 K rise
  instead of 75 K; X6-60: 2.50 against 2.89 K/W): a drive that cools better in a hotter room.
  - Source (desk): `DriveThermal.calibrate` at the 8 min time constant, fitted from 25 and from 35 C
    (`rl/telemetry/thermal.py:74-93`), with the inputs above. At 4 and 15 min the X4-36-L's are 5.56 / 4.82 and
    5.60 / 4.85 K/W; the ratio is 0.866-0.867 at all three.
- The no-break runs show the error. Under B the s2 winding peaks at 99.6 C in a 25, 30 or 35 C room, and 99.7 C at
  40 C. The room has almost stopped mattering.
- Under A the rise from 25 to 35 C is 11.5-12.4 K, a little more than the 10 K of room. The extra is the copper's
  resistance rising as it warms, which the model includes.

**Steady loads (desk, the same model).** A constant load settles at 90 C:
- at 0.945 of the continuous rating in a 25 C room;
- at **0.870** in a 35 C room, where 0.931 reaches 100 C;
- at 0.829 in a 40 C room.

These hold for the X4 and the X6 alike at 4-8 min time constants (0.866 at 15 min). Three of the four judged-case
worst drives (0.888, 0.931, 0.945) are above 0.870, so walked without breaks in a 35 C room they would settle above
90 C.

**Beside the bench rule.** B3 passes a drive at "Winding ≤ 90 °C at 35 °C" (`bench/BENCH_PLAN_FORELEG.md:61`). Under
A, two of the four policies, s2 (93.3 C) and s3 (98.3 C), are over that figure. This is a comparison, not a fail:
- B3 was written for a measured winding on the 75 cm T20 foreleg, over a 30-minute loop, with the shell on;
- these figures are a desk model of simulated torques, over a whole pack, in open air.

A shell would add heat that no model has (`rl/telemetry/thermal.py:33-35`).

**Provenance.**
- The rerun was a scratch script. At aa150f89 `rl/demo_course.py` has no room-temperature setting, though
  `DriveThermal` has `ambient_c`; D48's rule, every heat rule in degrees checked at 35 C, needs one.
- This batch adds both (working tree, 27 Nov 2026, uncommitted when this was written): a room setting, `--room-c` in
  `rl/demo_course.py` and `room_c` in `rl/telemetry/thermal.py`, and `rl/analysis/thermal_ambient.py`, whose tables
  give the figures above (`rl/analysis/results/heat35/thermal_ambient_2026-11-27.md`: the demo peaks under A and B at
  `:14-15`, the no-break runs at `:43-44`, the steady loads at `:71-79`).

## 3. What only hardware can show (none of it measured)

1. **Drive heat in a closed shell over a full pack.**
   - The model is "ESTIMATES, not measurements" (`rl/telemetry/thermal.py:3`). It leaves out the housing, the
     electronics and the airflow inside a shell, which it calls worse than open air (`:26-28`).
   - B3 runs 30-minute loops (`bench/BENCH_PLAN_FORELEG.md:61`); a pack lasts 84-96 min (simulation).
2. **The real command delay, feedback age and jitter, and where the policy runs.**
   - Heat is at or over the rating on two seeds already at 15 ms (section 1). B5 measures the delay, pass at 10 ms or
     less (`bench/BENCH_PLAN_FORELEG.md:63`).
   - The kickoff list puts the policy on the Jetson (`hardware/triceratops_kickoff/PARTS_PREMIUM.md:279`). Main's
     lessons say "Keep the walking loop on a microcontroller" (main's lessons `:127`).
3. **The drives as servos.**
   - The model's kp of 598.5 N·m/rad is "an assumption there too, not a maker figure" (`DECISION_LOG.md`, A2).
   - A sibling drive's manual caps the gains at "KP 0 to 500, KD 0 to 5", with units not confirmed
     (`hardware/trex_form_study/designlab/stage2/servo.py:229-231`; the manual read through web-search extracts on
     14 Nov 2026, `hardware/trex_form_study/stage2/jetson75/LATENCY_AND_BUS.md:17`, `:442`). The model's kv is
     11.51-24.48 N·m·s/rad (model lines 311-322).
   - A kp cap of 500 N·m/rad is 0.84 of the model's 598.5, inside the swept x0.4-1.5. kv was never swept: if the
     units match, a KD cap of 5 is 0.20-0.43 of the model's kv (5 / 24.48, 5 / 11.51), which no sweep has covered.
   - Friction, backlash and torque at speed at 42-49 V are unknown. The X4's 83 rpm cap above 24 V is "reseller text
     quoting MyActuator" (`DECISION_LOG.md` (E1), E1, from the kickoff list's F1, 18 Nov 2026).
4. **What the drives do when they overheat or lose commands.**
   - "What the drives do without commands is not confirmed"
     (`hardware/trex_form_study/stage2/jetson75/LATENCY_AND_BUS.md:365`, `:375-377`).
   - The drives have no brakes (main's lessons `:99-100`).
5. **Which X6 ships for $428** (store page read 18 Nov 2026, F9, `DECISION_LOG.md` (F9)): the page shows the X6-50-L's
   figures, 0.78 kg and 50 N·m peak (bench B2).
6. **The real mass and centre of mass:** "nothing here is quoted, bought, weighed or measured"
   (`design/PARTS_60CM.md:3`).
7. **The rubber rocker on the floors of public demos:** grip, spin and wear.
   - The model's sole friction of 0.85 is assumed (model line 54).
   - The project's only floor figure, 0.42-0.8, is for the T. rex's polyurethane treads and is "unmeasured" (main's
     lessons `:141-142`).
8. **The BNO085 IMU's latency, and its bias under foot impacts.**
   - The simulation adds Gaussian noise to every input and a random walk to the heading (`rl/README.md:145-150`).
   - The repository's own notes call the part "prototype only until measured"
     (`docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md:494`).

## 4. Risks that could stop the robot or cost real money (ranked)

The ranking is a judgement.

**1. Drive heat: a thin margin that a warm room, a shell, a longer delay or extra mass would use up.**
- **Why.**
  - The figures are in sections 1 and 2.
  - No model of a shell or its airflow exists, and the parts list's $90 of blowers, ducts and filters
    (`design/PARTS_60CM.md:156`, a parts-list estimate, 18 Nov 2026) are unsized.
  - Set beside the steering run's sketched heat rule (written for another policy, not yet fixed), today's policies
    miss it. The rule asks for the worst drive under about 0.85, and under the rating at x1.1 mass and at 20 ms
    (`rl/PLAN_STEERING_RUN.md:86-88`). Seeds miss these on 3, 3 and 4 of 4. No steering policy is trained (`:10`).
  - The battery cut-off (D47) adds an estimated 0.30 kg, 2.4 % of the model's 12.751 kg, that the simulation model
    leaves out, so no policy was trained or swept with it (desk estimate; `design/PARTS_60CM.md:25-26`, working
    tree). In the sweep, +0.5 kg on the chassis raises the worst drive by 0.02 of its rating on every seed, from
    0.87 / 0.94 / 0.97 / 1.00 to 0.89 / 0.96 / 0.99 / 1.02 (simulation; the summaries' payload tables, s0
    `:111-112`, s1 `:110-111`, s2 `:110-111`, s3 `:113-114`).
- **Cheapest test.** The steering dev runs with the hottest-drive cost (D29), read at 5 M steps: about 2.3-2.6 h each
  on Colab (`rl/PLAN_STEERING_RUN.md:122`; FINAL sheet `rl/runs/trike60-steer1-dev_STEPS.md`). Then B3 with the shell
  on, at 25 and 35 C.
- **Bench plan: covers it partly.** B3 is written for the 75 cm T20 foreleg ("elbow 16.2 N·m at T20") and replays
  foreleg duty only (the bench plan `:61`). The demo's hottest winding is a hind pitch drive on 2 of 4 seeds (appendix
  `:153`).
- **If it fails.**
  - **Software first:** a stronger heat cost, standing breaks or shorter walks.
  - **What the records name.** For heat, B3's fail line names ducting, a larger drive, a shorter held pose or short
    trots only (the bench plan `:61`). F12's options are for the hind knees: a 1.2:1 crank, X6 knees (+$318 of
    hardware, +0.92 kg; parts-list estimate, 18 Nov 2026) or a lighter build (`design/PARTS_60CM.md:187`). B4 names
    X6 pitch drives, but for the X4's speed cap, not for heat (the bench plan `:62`).
  - **X6 pitch drives as a heat fix** (this review's reading, desk): +0.46 kg and +$159 each (the bench plan `:62`;
    the list-price difference $428 - $269, catalog read 7 Nov 2026, `sim/data/drives.json:10`, `:19`, with the store
    page showing the same prices on 18 Nov 2026, F9), so +1.84 kg and +$636 for four. 1.84 kg is about 14 % of
    12.75 kg. In the sweep, +2 kg of payload puts the worst drive at 1.02-1.11 of rating, a hind knee on 3 of 4 seeds.

**2. Both controllers rely on signals the robot will not have.**
- **Why.** 11 of the 68 inputs (counted from `rl/README.md:132-137`; D46):
  - **true body velocity,** 3 inputs, "simulation only; no sensor gives it" (`rl/trike_env.py:127`);
  - **foot loads,** 4 inputs: "no foot load cells in the core build" (`sim/build_model60.py:12`), an option at +$254
    (`budget/SIZE_BUDGET_OPTIONS.md:88-89`; parts-list estimate, 18 Nov 2026);
  - **passive ankle pitch angles,** 4 inputs (`rl/trike_env.py:126`), with no encoder in `design/PARTS_60CM.md`.
  - The trot reads the foot loads to pick which legs enter its leg-odometry velocity estimate (`walk/walk60.py:67`,
    `:239`).
- **How likely.** It is certain that this must be resolved. How well a policy walks without these inputs is unknown,
  because no such run exists.
- **Cheapest test.** One 10 M-step run on the robot's own sensors (D46): about 4.5-5 h on Colab, the judged-run time
  (`rl/PLAN_STEERING_RUN.md:75-76`). For the trot, a stance estimate from drive torque.
- **Bench plan: not covered.** B9 tests an FX29 foot cell's reading, its I2C link and the sole's roll (the bench plan
  `:67`).
- **If it fails.** Foot cells at +$254 with a 21-23 week lead (budget estimate, 18 Nov 2026,
  `budget/size_budget_out.txt:108`); ankle sensors, unpriced; or a velocity estimator.

**3. Shoves and turning: failing or untested.**
- **Why.**
  - The push test (T5) fails on all four seeds. Falls of 8 at -1.25 / -1.0 / +1.0 / +1.25 m/s
    (the D22 report `:271-276`):

    | Seed | -1.25 | -1.0 | +1.0 | +1.25 |
    |---|---|---|---|---|
    | s0 | 8 | 0 | 0 | 6 |
    | s1 | 8 | 1 | 7 | 8 |
    | s2 | 1 | 0 | 0 | 8 |
    | s3 | 8 | 8 | 3 | 8 |

  - Seed 3 already falls 2 of 8 at -0.75 m/s (appendix `:271`).
  - Pushes are only ever sideways, in training and in the sweep (`rl/trike_env.py:232-234`;
    `rl/robustness/README.md:123`).
  - Run 3's report cannot say whether the training or the line-contact capsule foot is to blame
    (`rl/analysis/REPORT_RUN3.md:84-86`).
  - **Turning.** No controller in this build has turned (`walk/REPORT_WALK60.md:179`; `rl/PLAN_STEERING_RUN.md:10`).
    PR #697's scripted controllers turned and pivoted on the 128 and 160 mm models, with higher foot loads at 128 mm
    (the D22 report `:250-253`).
  - **Falls.** A belly skid is in the frame's cost line (`design/PARTS_60CM.md:158`) but in no model or design.
    Nothing protects the head, frill or shell.
- **Cheapest test.** The trot's sweep (`walk/sweep_walk60.py`) on the 128 mm capsule model with pushes both ways
  (D45). Then the steering dev runs, which train pushes and turns.
- **Bench plan: not covered.**
- **If it fails.** Retraining, if the training is the cause; a foot change, if the capsule is; impact protection.

**4. The stop and safety chain on drives with no brakes.**
- **Why.**
  - Every stop must "first command a damped sit" (main's lessons `:99-100`).
  - The stop behaviour is "not modelled yet" (`bench/BENCH_PLAN_FORELEG.md:66`). The whole-robot sit's regen energy,
    which B7 needs as its input, is "not modelled" (`:65`).
  - At aa150f89 the 60 cm list has battery alarms only (D47's row). The kickoff list says the robot needs an
    automatic cut-off, not only an alarm (`:83`).
  - "the watchdog MCU is not" a safety-rated part (the kickoff list `:302`).
  - If a drive's over-temperature cut-out removes torque mid-demo, that leg drops (section 3, item 4).
- **Cheapest test.** Simulate a damped sit of the 12.751 kg model and compute its regen energy (D45).
- **Bench plan: covers it partly.** B8 tests one leg with the radio stop (pressed, switched off, blocked), the
  mushroom and the pendant. It has no watchdog or BMS trip and no lost-CAN test. B7 waits for the sit energy.
- **What it takes.** An Orion BMS Jr 2, about $440 (maker's list price, web search 18 Nov 2026, the kickoff list
  `:392`), with its separate current sensor (`:83`), now adopted (D47). A certified stop adds +$2,450 of hardware if
  public demos require one (budget estimate, 18 Nov 2026, `budget/size_budget_out.txt:107`; D15).

**5. The drive line at 12S, and how realistic the actuator model is.**
- **Why.** The X4 is a 24 V drive (maker datasheet read 7 Nov 2026, `sim/data/drives.json:7`) run on a 42-49 V bus,
  and its cap rests on reseller text (E1), so how it behaves above its rated voltage is not confirmed. The model has
  no dry friction, backlash or torque-speed curve.
- **Cheapest test.** B1 and B4 at 42 and 49 V, plus a row identifying gains and friction.
- **Bench plan: covers it partly.**
  - B4's pass mark of 69 rpm is 1.15 times the 75 cm 0.30 m/s walk's 60 rpm (`bench/BENCH_PLAN_FORELEG.md:62`). Seed
    3's right fore pitch peaks at 86.3 rpm (`rl/analytics/results/fore128-s3/gait_report.md:69`).
  - B5 uses the T. rex delay sheet's sections 1, 3, 4.1 and 5 (the bench plan `:63`), not its servo-stiffness and
    friction sheets (`hardware/trex_form_study/stage2/jetson75/BENCH_SHEET.md:489`, `:556`).
- **If it fails.** Weeks of work in simulation: identify, fit, retrain. If the X4 cannot run at 48 V, all ten X4s
  need a drive or bus change.

**6. The mechanical design is not done.**
- **Why.**
  - The 21 Nov review rated parts, CAD, safety and packaging "Not Order-Ready" (`REVIEW_60CM.md:29`), and D41 says
    that finding stands (`DECISION_LOG.md`, D41).
  - The crossmembers pass through the drives (F19, F21), and the hind knee drives sit inside their struts (section 1).
  - **The pushrod is unsized.** On the 188 mm leg the 21 Nov review found 1,979 N of pull at the 37.8 N·m clamp and
    3,142 N at the X6-60's 60 N·m peak (28 mm crank, 43 deg transmission angle at -47 deg of flexion;
    `REVIEW_60CM.md:117`). The 128 mm leg keeps the 28 mm crank radius (model lines 188-189), but its four-bar's
    angles have not been worked out.
    Even at a 90 deg angle the pull would be 1,350 N and 2,143 N (desk: 37.8 / 0.028 and 60 / 0.028).
- **Cheapest test.** A packaging check of all four legs, parts in the same body included (D45), and a pushrod and
  bearing sizing.
- **Bench plan.** B6 measures play and torsional stiffness up to 35 N·m (the bench plan `:64`), with no proof load on
  the pushrod.
- **If it fails.** Chassis CAD under D5, then a frame quote. The frame line is an estimate, $2,460 (the budget's frame
  cost scaled by area, 18 Nov 2026; `design/PARTS_60CM.md:158`, `:179`).

**7. Software, budget and schedule.**
- **Software.** No firmware or bench test programs exist: "None of this is written yet"
  (`bench/BENCH_PLAN_FORELEG.md:101`). The microcontroller firmware is unwritten
  (`hardware/trex_form_study/stage2/jetson75/LATENCY_AND_BUS.md:365`).
- **Budget** (`budget/size_budget_out.txt`, planning estimates of 18 Nov 2026, `:1`):
  - $20,105-$22,988 landed (`:97-103`) against the $20k ceiling (D12);
  - $17,934-$18,773 with half the bench gear counted against the T. rex and the spares bought later (`:100`, `:104`);
  - the estimate predates D47's BMS (about $440, maker's list price, web search 18 Nov 2026). At aa150f89 the BMS is
    in neither the parts list nor the budget script; this batch adds it to the parts list (`design/PARTS_60CM.md`,
    `design/parts60.py`);
  - it leaves out foot cells (+$254, parts-list estimate, 18 Nov 2026) and a certified stop (+$2,450, above).
- **Schedule.** The 75 cm kickoff's milestones have no line for firmware
  (`hardware/triceratops_kickoff/CONSOLIDATION_PLAN.md:204-214`).
- **Cheapest test.** A frame quote and a check of drive stock (both the owner's), and a schedule that includes
  firmware.

## 5. Likely tuning only

- **Peak torque.** Every seed walks with every force limit at x0.4, with no fall (speed -1.8 to +2.0 % across the
  four seeds; the summaries, s0 `:125`, s1 `:124`, s2 `:124`, s3 `:127`). The worst drive then spends 29-59 % of the
  time at its reduced limit.
- **An X6-50-L elbow** (45 against 60 N·m peak, x0.75; the catalog's `myactuator_rmd_x6_p20_50_e_l` and
  `myactuator_rmd_x6_p20_60_e`, maker datasheets accessed 7 Nov 2026; the US store page lists 50 N·m peak, F9, read
  18 Nov 2026). At x0.7 on every drive there are 0 falls in 8 on all seeds (the summaries, s2 `:126`, and the other
  three headlines).
- **The 83 rpm cap.** It is exceeded in 0.01 % of steps on average and 0.05 % in the worst episode, by seed 3's right
  fore pitch, which peaks at 86.3 rpm (`rl/analytics/results/fore128-s3/gait_report.json`,
  `above_speed_allowance_fraction`). The fix is to add the cap to the model and retrain.
- **Gain errors.** kp x0.4-1.5 gives no falls, but heat goes over the rating at x1.2 on two seeds (section 1).
- **Drift.** At friction 0.3 (T4) the robot drifts 0.43 / 0.45 / 0.65 / 0.45 m, against 0.34 / 0.35 / 0.35 / 0.47 m on
  the nominal floor (the D22 report `:258`; appendix `:212`, `:217`). That is mostly drift already present,
  except on s2, where it nearly doubles. The steering run closes the heading loop.
- **Gait details:** irregular strides, hind feet clearing only 10-15 mm (appendix `:98`) and the ankle's stop range.
  These are reward or geometry choices.
- **Battery runtime:** 75-84 min per pack at a 40 W hotel load (appendix `:160`).
- **Small fixes.**
  - An RS-485 converter for the TTL neck servo bus (`design/PARTS_60CM.md:190`).
  - Regenerating the parts list from the adopted model. At aa150f89 `design/parts60.py:53` reads the 188 mm model's
    ledgers, and `design/PARTS_60CM.md:21` showed 12.694 kg there. This batch regenerates both from the adopted
    model's ledger: 12.751 kg for the model, and 13.053 kg with the BMS and its current sensor (+0.30 kg, an estimate
    the simulation model does not carry; `design/PARTS_60CM.md:21`, `:25-26`, working tree; see "This batch").

**The caveat on all of these:** "tuning" assumes retraining succeeds, and the heat cost cannot simply be raised. At
weight 3, a gait with exploration noise earns only +0.01-0.09 per step more than standing, against the README's
0.3 rule (`rl/PLAN_STEERING_RUN.md:57-61`).

## 6. Order of work before the large spend

1. **No-spend checks (days of work; D45).** Each has a file in this batch (working tree, 27 Nov 2026, uncommitted
   and not read for this record unless marked; see "This batch"):
   - The trot's sweep on the 128 mm model, with pushes both ways (`walk/REPORT_WALK60_FORE128.md`,
     `walk/fore128_walk60.py`, `walk/results/walk60_fore128_2026-11-27/`).
   - A 16-19 ms delay sweep. Heat jumps between 15 ms (0.87-1.02) and 20 ms (1.07-1.20). On trained policies D45
     makes it a Colab step for the owner. The batch's step sheet (`rl/runs/trike60-ownsensors_STEPS.md`, part 2,
     working tree) leaves open whether it can run in the session instead, for the lead to settle: its section 7,
     "WHAT YOU DO", has no owner step yet. `rl/robustness/sweep.py` at aa150f89 has no 16-19 ms points (its delays are
     fixed, `:127`).
   - The thermal model at 35 C: section 2, with a room setting in `rl/demo_course.py` (its change read here),
     `rl/analysis/thermal_ambient.py`, its tables in `rl/analysis/results/heat35/`, and
     `rl/analysis/REPORT_HEAT_35C.md`.
   - A whole-robot damped sit and its regen energy (`sim/REPORT_SIT60.md`, `sim/sit60.py`,
     `sim/results/sit60_2026-11-27/`).
   - A packaging check of all four legs, with the hind knees and both crossmembers (`sim/check_packaging60.py`,
     `sim/results/packaging60_2026-11-27/`; its strut figures checked here, section 1).
   - The pushrod's sizing (`design/pushrod60.py`, `design/PACKAGING_PUSHROD_60CM.md`).
   - The parts list regenerated (`design/parts60.py`, `design/PARTS_60CM.md`; its model total, 12.751 kg, and its
     BMS lines, +0.302 kg for 13.053 kg in all, seen here).
2. **Training.**
   - The steering dev runs, as their FINAL sheet sets them (D35).
   - One run on the robot's own sensors (D46), with fore-aft pushes worth adding (step sheet in this batch,
     `rl/runs/trike60-ownsensors_STEPS.md`, with `rl/analysis/PREREG_OWNSENSORS.md`).
   - The heat rule pre-registered in degrees at 35 C (D48) before the judged runs.
3. **Rewrite the bench plan for the 128 mm article** (`README.md:95`, next step 6). Add:
   - pass marks derived from what the walk needs;
   - hind pitch and knee duty in B3, over a loop as long as a pack, at 35 C;
   - gain and friction identification;
   - a loaded speed sweep, and the X6's speed;
   - watchdog and BMS trips in B8, and the drives' reaction to lost CAN;
   - an IMU log;
   - a pushrod proof load.
4. **Owner decisions before ordering:**
   - D3, compute, including whether the policy runs on the Teensy or the Jetson;
   - the foot cells, after the D46 run;
   - the demo's duty cycle: standing breaks, walk length, tether or battery;
   - the budget ceiling.
5. **Write the minimum Teensy firmware, then order and run the one-foreleg bench** (kickoff milestones T2 and T4).
   Then "the simulation is re-run at the measured values" before the four-leg order
   (`bench/BENCH_PLAN_FORELEG.md:84-85`).
6. **Before the four-leg order (T5):** design the chassis under D5, get a frame quote, weigh the first parts as they
   arrive (`design/PARTS_60CM.md:188`) and re-cost.

## 7. What changed from the working review

**Found as cited.** Every other quotation was found at its cited line, and every other number reproduced. That
includes the 47 % estimated mass, the 3-of-4 counts against the steering sketch and the 1.84 kg / $636 sum.

**Corrected or qualified:**
- **Joint friction.** "No joint friction" now reads no dry friction. Viscous damping and armature are modelled: the
  default at model line 40, and on the drive joints the drive's reflected armature (see "What that rests on").
- **The bench rule.** "2 of 4 policies fail the bench's own rule" now reads: are over B3's 90 C figure. B3 is a
  measured-winding rule for another leg and loop (section 2).
- **Pushes.** They are not a demo item (P8); the push test stands in for a stumble or a bump.
- **Turning.** "No controller has turned" is true in this build only; PR #697's scripted controllers turned.
- **Fall protection.** "No fall-protection design": a belly skid is costed but not designed.
- **Fallbacks.** "The records' only named fallback" was X6 hind knees. F12 also names a 1.2:1 crank and a lighter
  build, all for the hind knees. For heat, B3 names ducting or a larger drive; B4 names X6 pitch drives, for the speed
  cap.
- **Drift.** "T4 drift close to the nominal drift" holds on three seeds, not on s2.
- **"Weight 3 does not".** This is the reward margin of a noisy gait over standing, not a heat margin.
- **The 83 rpm cap.** 0.05 % is the worst episode; the mean is 0.01 %.
- **Runtime.** "About 2x apart" now reads 0.45-0.71 (1.4-2.2x).
- **Pushrod.** The 1,979 / 3,142 N figures are the 188 mm leg's. The 128 mm leg's angles are unchecked; it has at least
  1,350 / 2,143 N.
- **IMU noise.** The simulation adds Gaussian noise plus a heading random walk, not "white noise only".
- **Force-limit and kp tolerance** is for falls and speed only: drift reaches 1.20-1.52 m and heat goes over at kp x1.2.
- **Delay.** Two seeds are already at or over the rating at 15 ms (1.02, 1.00).
- **Budget.** The $17.9k-$18.8k case and the missing BMS are added.
- **Bus load.** The figure is the 75 cm kickoff's arithmetic, for the same count of drives per bus, not the same
  drives.
- **Gain caps.** If the units match, the sibling manual's KD cap is 0.20-0.43 of the model's kv, a range no sweep
  covers.
- **Bench coverage.** B9 is not "an FX29 reading only": it also tests the link and the sole's roll. B8's three inputs
  are the radio stop, the mushroom and the pendant.
- **Outside work.** Mini Cheetah's 50 Hz policy and onboard computer were not confirmed today; the other papers and
  the GitHub search were not rechecked.
- **Real-robot sensing.** It is no longer "only the velocity, the run after": see D46.

**Added:** the hind knee overlap, the method B heat figures and why A is the one to use, the 35 C steady-state
line (0.870 of rating), and the battery cut-off's estimated 0.30 kg beside the heat margin.

## After the review: the owner's decisions (27 Nov 2026)

Recorded in `DECISION_LOG.md` (D45-D48), question tool, 27 Nov 2026:
- **D45, start the no-spend checks now,** alongside the main-pipeline PR:
  - pushes both ways on the scripted trot;
  - a 16-19 ms delay sweep;
  - the energy of an emergency sit;
  - a packaging check of all four legs;
  - the foreleg pushrod's sizing;
  - the parts list regenerated from the adopted model.

  Each result lands on this branch after an independent check. The delay sweep on trained policies becomes a Colab
  step for the owner.
- **D46, train without the three simulation-only signals first.** A step sheet for one 10 M-step run on the robot's
  own sensors, which the owner runs on Colab. The foot sensors are decided after it.
- **D47, an automatic cut-off.** A battery management system wired into the stop chain (for example an Orion BMS Jr 2,
  about $440, maker's list price, web search 18 Nov 2026), added to the parts list. Nothing is ordered.
- **D48, heat rules assume a 35 C room,** a warm indoor room, until the real room's temperature is known. Every heat
  rule in degrees not yet judged is checked at 35 C. Its row's 93-98 C figures are method A's (section 2).

**This batch.** Work on D45-D48 started the same day. When this was written its files were in the working tree,
uncommitted; each package then had its independent check (D45). The batch changes some files cited above, so their
lines move. It is six packages; every file is listed as it is committed, by its path from this folder:
- **This record:** new `READINESS_REVIEW_2026_11_27.md`.
- **The trot on the 128 mm model, pushes both ways (D45):** new `walk/REPORT_WALK60_FORE128.md`,
  `walk/fore128_walk60.py`, `walk/fore128_slow_gait.py`, `walk/tests/test_fore128.py` and the folder
  `walk/results/walk60_fore128_2026-11-27/` (`RESULTS.md`, `summary.json`, `judged_101-110.json.gz`,
  `judged_201-210.json.gz`, `sideways_200000/`, `sideways_300000/`, `fore_aft_200000/`, `fore_aft_300000/`,
  `slow_gait/` with 60 files, and `capsule188_check/`).
- **The emergency sit (D45):** new `sim/REPORT_SIT60.md`, `sim/sit60.py`, `sim/tests/test_sit60.py` and
  `sim/results/sit60_2026-11-27/` (`sit60_summary.json`, `sit60_tables.md`).
- **Packaging, the pushrod and the parts list (D45, D47):** new `sim/check_packaging60.py`,
  `sim/tests/test_check_packaging60.py`, `sim/results/packaging60_2026-11-27/` (`packaging60.json`,
  `packaging60_report.txt`), `design/PACKAGING_PUSHROD_60CM.md`, `design/pushrod60.py`, `design/pushrod60_out.json`
  and `design/tests/test_parts60_pushrod60.py`; changed `design/parts60.py` and `design/PARTS_60CM.md` (regenerated
  from the adopted model's ledger: 12.751 kg for the model; 13.053 kg with the BMS and its current sensor, +0.30 kg,
  estimate, not in the simulation model; `design/PARTS_60CM.md:21`, `:25-26`).
- **Heat in a 35 C room (D48):** new `rl/analysis/REPORT_HEAT_35C.md`, `rl/analysis/thermal_ambient.py`,
  `rl/analysis/results/heat35/thermal_ambient_2026-11-27.json` and `.md`, and `rl/tests/test_thermal_room.py`;
  changed `rl/telemetry/thermal.py` (a room paragraph and `room_c`), `rl/demo_course.py` (`--room-c`) and
  `rl/PLAN_STEERING_RUN.md` (a D48 note after its line 90).
- **The run on the robot's own sensors (D46):** new `rl/runs/trike60-ownsensors_STEPS.md` (a draft sheet),
  `rl/analysis/PREREG_OWNSENSORS.md`, `rl/configs/run3_terrain_ownsensors.json` and
  `rl/tests/test_obs_own_sensors.py`; changed `rl/trike_env.py` (an `obs_foot_loads` flag, default on) and
  `rl/trike60_training.ipynb` (the preset for this run).

This record checked only:
- the batch's 35 C tables (they match section 2);
- the packaging report's two strut figures (they match section 1);
- the regenerated parts list's total and BMS lines (section 5);
- `rl/demo_course.py`'s `--room-c` change (section 2);
- the step sheet's part 2, section 7 (section 6).

The rest, the three reports included, is not read here. The parts list, the 35 C tables, the packaging report and
the step sheet were rewritten after this record was first written. On 28 Nov 2026, once the other five packages
were committed (6196720c, 1abc6f01, 2958f5e1, 7b2db13b and d21c64ca), every line this
record cites in a file the batch changed was rechecked against the committed file and moved where it had moved;
`design/parts60.py:53` and the 12.694 kg total are cited as they were at aa150f89.
