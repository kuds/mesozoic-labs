# Emergency stop of the adopted 60 cm model: damped sit, hard sit and passive collapse (simulation, with a desk estimate of the regen)

27 Nov 2026, under D45 (run the no-spend checks now; `DECISION_LOG.md` (D45)); rerun and revised 28 Nov 2026. Why: the
leg drives have no brakes, so every stop must first command a damped sit (main's
`docs/hardware/LESSONS_LEARNED.md:99-100`, at a232da31), and the bench plan's B7 needs the regen of a hard sit of
twelve drives as its input
(`bench/BENCH_PLAN_FORELEG.md:65`). Script `sim/sit60.py`; tables `sim/results/sit60_2026-11-27/sit60_tables.md`;
summary `sim/results/sit60_2026-11-27/sit60_summary.json` (both written by `sit60.py summarise`; the folder keeps its
27 Nov name). Model: the adopted `sim/generated/trike60_C_closed_capsule_fore128.xml` (12.751 kg; D25, D41), read and
never written; the skid variants are test copies written to a temporary folder at run time. Run: `sit60.py run
--workers 2`, 888 stops in 665 s on 28 Nov 2026, MuJoCo 3.10.0, at commit db43e595 with `sit60.py` untracked (its
sha256, 799e0417..., is in the summary's meta, for the run and for the summary). **Reported, not judged:** no
pre-registration covers this study, so nothing here is a verdict; where a rule exists (the bench plan's B7 and B8 pass
marks) the measurement is shown beside it, with the case the rule was written for. Two seeds per start; one sit
routine.

Evidence kinds, kept apart throughout:

- **Simulation:** every motion, force, torque, speed and time (MuJoCo, the build's training plant).
- **Desk:** the electrical side (the build's drive loss model applied to the simulated torques and speeds), the energy
  ceilings, and every comparison with the power board, the bus and the pack (catalog and parts-list figures).
- **Measured:** none. Nothing is built or measured.

Dates of the catalog and parts-list figures:

- **Catalog figures** (the drives' torque constants, phase resistances, efficiencies, back-drive torques, back-EMF
  constants, gear ratios and peak torques): maker data read 7 Nov 2026, as recorded in
  `hardware/trex_form_study/designlab/data/actuators.json` (its sources are marked "accessed 2026-11-07").
- **Parts-list figures** (the 52.5 V clamp, the two 50 W resistors, the drives' 55 V maximum as the parts list
  states it, the bus capacitors, the 49.2 V pack ceiling, the about 50 V tether setting, the 12-60 V rail input, the
  relay's 0.5-1.5 s delays and the pack's "60C"): `hardware/triceratops_kickoff/PARTS_PREMIUM.md` as of its last
  change, aa3420e9, 18 Nov 2026. In that version the power board line (`:393`) is an estimate (EST), with its
  ideal-diode controller's rating from a web search of 18 Nov 2026; the rails' 12-60 V input (`:394`) is a
  web-search result of 18 Nov 2026; the pack line (`:391`) is a catalog listing read on 7 Nov 2026.

## Answer first

- **The adopted model has nothing to sit on, so every stop still drops the body after the power cut (simulation).**
  - The damped sit (1.0 s) and the hard sit (0.3 s) lower the chassis from 0.316 to 0.258 m while the drives are
    powered.
  - After the cut the legs, which have no brakes, fold until they rest on their hard stops, and the body drops again.
  - How far it drops depends on the start and on the contact setting. The build checks every result under both
    settings, and both were run at the same starts: standing, and the trot at all 14 phases of one stride, on two
    seeds.
  - **From standing:**
    - Stiffer contact (impratio 10): drop after the cut 0.038-0.039 m; the soles carry the whole robot, at 0.220 m;
      at rest 1.7-2.7 s after the stop input.
    - The model's own contact: drop 0.069-0.070 m; at rest 4.5-5.5 s after the stop input. For the first second
      after the cut the two settings agree within 6 mm (0.220 and 0.225 m).
    - The rest of the model contact's drop, 0.031-0.032 m between 1 s and about 4 s after the cut, is the soft
      contact's creep. The soles slide apart (the fore feet forward, the hind feet back) by up to 66-69 mm while
      their friction use |Ft| / Fn is 0.37-0.44 (median over the sliding samples; 95 % of them at or under 0.79),
      under the soles' friction coefficient of 0.85. Under impratio 10 the same phase sinks 4-6 mm, with 9-10 mm of
      sole slide.
    - Only in that creep do the fore upper struts reach the floor: at 0.08-0.09 m/s, peak 2.3-2.6 body weights,
      3.7 s after the cut. From standing, the strut landing is a soft-contact result.
  - **From the trot** (28 starts per stop under each contact):
    - Drop after the cut: 0.039-0.052 m under impratio 10 (median 0.050 m), 0.050-0.071 m under the model's contact
      (median 0.055 m). The chassis ends at 0.208-0.220 m and 0.188-0.209 m.
    - The fore upper struts reach the floor under both contacts: in all 84 sits under the model's contact, and in
      53 of 84 under impratio 10, at 11 of the 14 phases. From the trot the strut landing is not a soft-contact
      artefact.
    - Paired at the same start, 1 s after the cut the model's contact has the chassis a median 4-6 mm lower than
      impratio 10 (per stop); the gap runs from 2 mm the other way to 18.5 mm, depending on the phase.
    - The drops beyond 0.052 m come only under the model's contact. Its largest, 0.069-0.071 m, are at the two
      headline phases (5.94 and 6.29 s), where its creep adds up to 0.033 m after the first second, as from standing.
    - Time to rest: under impratio 10 a median 1.7-2.9 s per stop over the sits that came to rest (1.7-3.0 s
      counting the sits still moving at 8 s as over 8 s), up to 7.4 s, and 9 of 84 sits still moving slowly at 8 s;
      under the model's contact up to 6.6 s, also 9 of 84 still moving.
  - **Across both settings:** drop after the cut 0.038-0.071 m; time to rest from standing 1.7-5.5 s. Under
    impratio 10 the drop is 0.038-0.052 m; the top of the range, 0.069-0.071 m, is the model contact's creep.
- **A passive collapse: the drives lose their commands and all 12 leg drives go to zero torque (simulation).**
  - From standing the body falls 0.097 m under impratio 10 (to 0.220 m, at rest in 1.0 s), and 0.128 m under the
    model's contact (to 0.188 m, 4.0 s, the fore struts on the floor). The soles take up to 4.4 body weights.
  - From the trot the chassis ends at 0.207-0.234 m under impratio 10 and 0.191-0.224 m under the model's contact.
    The fore struts land in 12 and 17 of 28 collapses, and the leg links take up to 7.2 body weights under
    impratio 10 (5.5 under the model's contact).
  - The hind knees reach 77 rpm, 0.93 of their speed allowance (74 rpm, 0.89, under impratio 10). From the trot the
    joints' hard stops take up to 43.1 N m under the model's contact and 43.8 N m under impratio 10 (up to 25.9 N m
    from standing).
  - It exchanges no electrical energy in this model. A drive delivering no torque carries no current, and the
    drives' back-EMF at these speeds stays under any bus held at 42-50 V (desk, below).
- **With a skid under the belly (test copies of the model, not the adopted file; simulation).**
  - The body lands on the skid and stays within 5 mm of its sit height after the cut, except where it tipped.
  - The landing force depends mostly on how far below the soles the skid reaches (5, 10 or 20 mm, a choice of this
    study) and on its stiffness. Peak skid force over the runs: 1.0-3.3 body weights in a damped sit and 1.4-7.1 in a
    hard sit (every skid copy), 10-17 in a collapse onto a rigid skid and 4.0-4.8 onto the soft one.
  - The hard sit tipped the 10 mm skid copy onto its side in 3 of 28 trot starts under the model's contact and in
    1 of 28 under impratio 10 (seed 1 at 6.34 s, one of the three). No other skid tipped. That is a sensitive
    outcome, not a rate.
  - The skid has to reach 0.194 m below the belly shell, because the legs do not fold far enough to bring the belly
    near the floor.
- **The B7 input, the regen of a hard sit of twelve drives (desk on simulation; an upper bound; 216 runs per sit).**
  - At most 2.93 J goes back onto the shared bus, over 0.230 s. That run (adopted model, impratio 10, standing,
    seed 2) peaks at 25.9 W over 1 ms, 25.8 W over 10 ms and 21.7 W over 100 ms.
  - The largest peaks of any hard sit are 26.7, 26.0 and 21.7 W, and they come from other runs. Taken together,
    these figures are an **envelope** for the bench pulse, not one simulated pulse.
  - The bounding 0.2 s sit gives 2.84 J over 0.099 s (its own peak 51.6 W; the largest of any 0.2 s sit is
    68.5 W).
  - The damped sit gives 1.48 J over 0.511 s (its own peak 4.5 W; the largest 18.2 W).
  - The impratio 10 trot sits at all 14 phases stay under these: at most 1.78 J (hard sit) and 1.97 J (0.2 s sit).
- **Against the power board (desk; parts-list figures as of 18 Nov 2026).**
  - 2.93 J is 0.029 s of the two 50 W resistors' combined continuous rating
    (`hardware/triceratops_kickoff/PARTS_PREMIUM.md:393`, an estimate line of 18 Nov 2026). B7's ten pulses in a row
    are 29.3 J.
  - The largest peaks need at most 103 ohm (26.7 W) and 40 ohm (68.5 W) of clamp resistance at 52.5 V. The records
    give the resistors' rating, not their resistance.
  - Bus capacitance alone would need 22.8 mF to keep 2.93 J under 52.5 V from a 50 V bus (11.1 mF under the drives'
    55 V). The records give no capacitance, so the clamp has to take it.
  - A 30 W hotel load on the same bus would take the whole hard sit, and all but 0.68 J of the 0.2 s sit.
- **A ceiling for any sit routine that reaches the same height before the cut (desk energy balance).**
  - The drives cannot take more than 7.10 J net while powered. That is the largest, over the runs, of each run's
    kinetic energy at the stop input plus the potential energy released before the cut (that run: 0.71 + 6.39 J,
    adopted model, impratio 10, trot, damped sit, seed 2 at 6.29 s). The two largest parts come from different
    runs: the largest kinetic energy, 0.82 J, is from the impratio 10 trot start of seed 1 at 6.24 s.
  - A routine that lowered the body all the way while powered would have a ceiling of 16.7 J, 0.17 s at 100 W.
  - The modelled bus surplus stays under the powered ceiling in all 648 sits. The gross braking at the joints does
    not (290 of 648 are over it), because a drive can brake out work it put in itself.
- **The 12S pack's charge acceptance is not given in the records.** The ideal diodes block regen from reaching the
  pack by design (`PARTS_PREMIUM.md:311`), so the pack takes none.
- **B8's timing (simulation, beside a bench rule written for one leg on a catch rig).**
  - On the adopted model, under both contacts, the chassis is within 5 mm of its height at the cut 0.70-0.73 s before
    the cut in the damped sit, 0.23-0.24 s in the hard sit and 0.30-0.32 s in the 0.2 s sit. B8 asks for "at least
    0.2 s before power drops" (`bench/BENCH_PLAN_FORELEG.md:66`).
  - The hard sit's last target reaches the drives 0.19 s before the cut (arithmetic from the routine's timing:
    0.5 - 0.3 - 0.01 s for the 10 ms delay).
  - B8's fail line is "never ship with a stop that drops the leg". With no skid the whole body drops 0.038-0.071 m
    after the cut, onto the soles and, at most trot phases, the fore struts, under both contacts.

## What was run

- **Plant (simulation).** The training environment `rl/trike_env.py`, built as the scripted trot's plant
  (`walk/walk60.py`, `plant_config`) on the adopted model:
  - domain randomisation off, sensor noise on, 10 ms servo delay (D9, `DECISION_LOG.md` (D9)), 1 ms physics;
  - a flat floor (the demo is on flat ground: the D28 note, `DECISION_LOG.md` (D28));
  - drives: the model's position servos (kp 598.5 N m/rad, kv 11.51-24.48 N m s/rad; A2, `DECISION_LOG.md` (A2)),
    clamped at 0.70 of the maker's peak;
  - seeds 1 and 2.
- **Starts.**
  - Standing: the trot controller at 0 m/s for 2 s.
  - Trotting: the trot at the 0.25 m/s command, stopped at 14 moments 0.05 s apart over one 0.70 s stride
    (5.94-6.59 s), so every phase of the stride is covered. 0.40 m/s is not a walking target and is not used.
  - The headline trot starts, 5.94 and 6.29 s, are each pair of legs' nearest control step to mid-swing (phases
    0.771 and 0.271). Their traces are kept, and the sensitivity cases run there.
- **The sit (`sit_1.0`, `sit_0.3`).**
  - At the stop input each ankle moves on a minimum-jerk path, from where the controller last sent it to the same
    fore-aft and sideways place at the sit depth. The move takes 1.0 s (the damped sit) or 0.3 s (the hard sit).
  - The joint targets come from the trot's own inverse kinematics and go through the same 10 ms delay queue, with the
    drives' gains and torque limits.
  - The sit depth is the deepest the forelegs reach with the ankle under the hip and the elbow at most 32 deg, 4 deg
    short of its +36 deg stop: the ankle 0.2234 m below the chassis origin, the chassis 0.2594 m over the floor,
    the body level. The hind legs could go 17 mm deeper.
  - Power is cut 1.5 s (damped) or 0.5 s (hard) after the stop input: every actuator, head and tail included, goes to
    zero torque. The Pilz relay's delayed contacts are set between 0.5 and 1.5 s (`PARTS_PREMIUM.md:301`, as of
    18 Nov 2026).
  - `sit_0.2` (0.2 s, cut at 0.5 s) is a bounding case for B7, faster than the hard sit.
- **The collapse.**
  - The 12 leg drives go to zero torque at the stop input; the head and tail servos keep their targets.
  - What the real drives do when commands stop is not confirmed: they have a time-out setting, planned as torque-off
    (`hardware/trex_form_study/stage2/jetson75/LATENCY_AND_BUS.md:365`).
  - Two sensitivity cases run at the headline starts (standing, and the trot triggers 5.94 and 6.29 s), under the
    model's contact only:
    - `collapse_friction` adds each drive's back-drive torque from the catalog as joint friction: X4 0.2 N m, X6
      1.6 N m (`hardware/trex_form_study/designlab/data/actuators.json:125`, `:208`; maker data read 7 Nov 2026).
    - `collapse_shorted` adds the damping of a drive with shorted phases, Kt^2 / (3 R) at the output: X4 5.4, X6
      7.2 N m s/rad. Inductance is neglected, so this is an upper bound.
  - Whether the drives do either on a lost command is not known. These are desk constants, simulated.
- **Test copies with a skid.**
  - The adopted model has no skid. The parts list says "the skid and bumpers take the sit" (`PARTS_PREMIUM.md:304`),
    and the 60 cm parts list prices a belly skid inside the frame line (`design/PARTS_60CM.md:120` at aa150f89,
    27 Nov 2026), but
    neither the model nor the mass ledger has one.
  - Each test copy is the adopted XML with one added massless box, `belly_skid_TEST`, under the belly:
    - it has the chassis's contact class;
    - its top is at the belly shell's lowest point, 0.075 m below the chassis origin;
    - its bottom is 10 mm deeper below the chassis than the soles at the sit pose (0.2694 m), so the box is 0.194 m
      tall.
  - Variants:
    - `skid`: 0.30 x 0.10 m, centred under the standing centre of mass;
    - `skid_e5mm` and `skid_e20mm`: 5 and 20 mm early contact;
    - `skid_wide`: 0.18 m wide, its front end at x = 0.110 m so that it clears the fore elbow drives' X6 housings;
    - `skid_soft`: `skid` with its floor contact at solref 0.03 s instead of the model's 0.008 s, a stand-in for a
      foam pad (standing and the two headline trot starts only).
  - The adopted file is never written; the tests check it by its sha256.
- **Stiffer contact.**
  - The four main stops of the adopted model and of `skid` are repeated at impratio 10, the setting the build checks
    every result under, at every start the model's contact runs them at: standing and all 14 trot triggers, both
    seeds (28 trot runs per stop under each contact).
  - The first runs (27 Nov 2026) repeated only the two headline trot triggers, two of the phases with the smallest
    drop under impratio 10, and set those 4 runs beside the 28 runs over all 14 phases under the model's contact.
    That mixed a phase effect into the contact comparison, so the impratio 10 trot cases were rerun at every trigger
    on 28 Nov 2026, and every impratio 10 statement here comes from that rerun.
  - The other skid copies and the sensitivity cases run under the model's contact only.
- **Measures.** All are taken from the stop input over 8 s; the definitions are in `sit60.measures` and
  `sit60.settling`.
  - Time to rest: the body under 5 mm/s and 0.02 rad/s for the last 0.5 s or more, and not before the cut.
  - The chassis's height at the cut, at fixed times after it and at the end; the drop after the cut.
  - Peak torque and speed per drive group, against the model's limit and the gait report's speed allowances (X4
    83 rpm, X6 108 rpm at the joint; `rl/analysis/gait_pipe1.py:40`).
  - Peak floor force per part group (soles, skid, leg links, shell), and the first body part's landing speed.
  - Self-contact, and the skid's contact with the floor or a leg before the stop input (checked at each 10 ms control
    step).
  - Torque on the joints' hard stops, during the stop, and which joints rest on them at its end.
  - From 1 s after the cut (or after the stop input, for a collapse) to the end: the chassis's further sink, the
    largest distance one loaded sole slides along the floor, and the soles' friction use |Ft| / Fn in the samples
    where a loaded sole slides faster than 2 mm/s.
- **Electrical side (desk, on the simulated torques and speeds).**
  - Per 1 ms sample, each drive's bus power is its mechanical power at the drive plus its copper loss 3 R
    (tau_drive / Kt)^2 at 25 C. The mechanical power at the drive is the joint's power x 0.9 through the flexion
    linkages while braking, and / 0.9 while motoring (`sim/data/drives.json:40`).
  - Kt and R are the values the thermal model is given (`rl/analytics/gait_report.drive_table`, which
    `rl/demo_course.py` passes to `rl/telemetry/thermal.DriveThermal`): X4-36-L 1.8 N m/A rms and 0.2 ohm per phase
    (`actuators.json:104-105`), X6-60 2.1 N m/A rms and 0.205 ohm (`actuators.json:186-187`); maker data read 7 Nov
    2026.
  - The copper-loss formula is the thermal model's own (`rl/telemetry/thermal.py:61-64` at aa150f89), checked in the
    tests.
  - Energy pushed back onto the bus is counted two ways:
    - per drive alone: the gross regen;
    - on the shared bus, where one drive's braking feeds another's motoring: the positive part of the sum, which is
      what the clamp must take (the bus surplus).
  - The bus surplus is also given less a 25-40 W hotel load (`rl/analytics/gait_report.py:81`).
  - **Left out, so the regen is an upper bound:**
    - gearbox and bearing losses in the drives. The maker's efficiency figures, 64.7 % for the X4-36-L and 72.7 % for
      the X6-60 (`actuators.json:125`, `:208`; read 7 Nov 2026), would cut the return well below this estimate.
      The model's own joint damping, 0.14 N m s/rad, is not counted as drive power.
    - inverter switching and conduction losses, and iron losses;
    - the drives' own standby draw, and wiring and capacitor losses;
    - the rise of winding resistance with temperature. Copper is taken at 25 C; at 3.93e-3 per K (`thermal.py:40` at
      aa150f89), a winding in D48's 35 C room, or hotter, loses more and returns less.
  - Also not modelled: the drives' own current loops and what they do with braking current, motor inductance, and
    the drives' DC-link capacitors.

## The adopted model (no skid)

Simulation, both contact settings, at the same starts. How to read the table:

- Time to rest: median / max over the runs that came to rest within the 8 s window; the runs still moving at 8 s
  are counted in brackets and left out of the median.
- Chassis end and drop after the cut: min / max over the row's runs.
- First body part on the floor: its vertical landing speed, median / max over the runs in which a body part touched
  (counted where that is not every run).
- Peak floor force, drive torque, worst speed and hard stop: the largest over the row's runs. The soles' and the leg
  links' force cells are each part's own maximum, so the two can come from different runs. (In
  `sit60_tables.md` too, the floor-force cells give one value per part, the maximum over the row's runs, and the
  end-support cells one value per part, the median, although that file's opening line says median / max.)
- BW = body weights (125.1 N).
- Drive torque is the largest share of the model's limit from 20 ms after the stop input on. That leaves out the
  walking controller's own torque at the stop input and the 10 ms delay.
- The worst speed is the drive group with the largest share of its speed allowance.
- Hard stop is the largest torque on a joint's range limit during the stop.

| start | stop | contact | runs | time to rest s | chassis end m | drop after cut m | first body part on the floor | peak floor force BW, max over runs: soles / leg links | worst drive torque N m (share of limit) | worst speed (share of allowance) | hard stop N m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stand | sit_1.0 | model | 2 | 5.52 / 5.53 | 0.189 / 0.189 | 0.069 / 0.069 | fore upper struts, 0.08 / 0.08 m/s | 1.88 / 2.55 | knee 3.8 (0.20) | knee 41 rpm (0.50) | pitch 17.7 |
| stand | sit_1.0 | impratio 10 | 2 | 2.71 / 2.74 | 0.220 / 0.220 | 0.038 / 0.038 | none | 1.90 / 0.00 | knee 3.9 (0.21) | knee 41 rpm (0.50) | pitch 18.7 |
| stand | sit_0.3 | model | 2 | 4.52 / 4.52 | 0.188 / 0.188 | 0.070 / 0.070 | fore upper struts, 0.08 / 0.09 m/s | 1.94 / 2.51 | knee 5.1 (0.27) | knee 41 rpm (0.50) | pitch 17.9 |
| stand | sit_0.3 | impratio 10 | 2 | 1.73 / 1.78 | 0.220 / 0.220 | 0.038 / 0.039 | none | 1.90 / 0.00 | knee 5.2 (0.27) | knee 41 rpm (0.50) | pitch 18.6 |
| stand | sit_0.2 | model | 2 | 4.51 / 4.51 | 0.188 / 0.188 | 0.070 / 0.070 | fore upper struts, 0.08 / 0.09 m/s | 1.95 / 2.58 | knee 7.4 (0.39) | knee 41 rpm (0.50) | pitch 17.9 |
| stand | sit_0.2 | impratio 10 | 2 | 1.72 / 1.76 | 0.220 / 0.220 | 0.038 / 0.039 | none | 1.90 / 0.00 | knee 7.7 (0.41) | knee 41 rpm (0.50) | pitch 18.7 |
| stand | collapse | model | 2 | 3.99 / 4.00 | 0.188 / 0.188 | - | fore upper struts, 0.08 / 0.09 m/s | 4.44 / 2.37 | none (zero torque) | knee 67 rpm (0.81) | knee 25.7 |
| stand | collapse | impratio 10 | 2 | 1.00 / 1.03 | 0.220 / 0.220 | - | none | 4.44 / 0.00 | none (zero torque) | knee 67 rpm (0.81) | knee 25.9 |
| stand | collapse_friction | model | 2 | 7.06 / 7.15 | 0.190 / 0.191 | - | fore upper struts, 0.07 / 0.07 m/s | 3.20 / 2.29 | none (zero torque) | knee 61 rpm (0.73) | knee 25.5 |
| stand | collapse_shorted | model | 2 | 7.31 / 7.33 | 0.182 / 0.182 | - | none | 1.16 / 0.00 | none (zero torque) | knee 5 rpm (0.06) | pitch 5.7 |
| trot | sit_1.0 | model | 28 | 2.72 / 5.98 (3 of 28 not at rest) | 0.188 / 0.206 | 0.053 / 0.071 | fore upper struts, 0.06 / 0.17 m/s | 2.27 / 3.51 | pitch 12.9 (0.61) | pitch 48 rpm (0.58) | elbow 21.7 |
| trot | sit_1.0 | impratio 10 | 28 | 2.90 / 7.40 (5 of 28 not at rest) | 0.208 / 0.220 | 0.039 / 0.052 | fore upper struts, 0.05 / 0.10 m/s, in 15 of 28 | 2.37 / 2.09 | pitch 14.9 (0.71) | roll 47 rpm (0.57) | pitch 23.0 |
| trot | sit_0.3 | model | 28 | 1.76 / 6.55 (3 of 28 not at rest) | 0.188 / 0.209 | 0.050 / 0.071 | fore upper struts, 0.07 / 0.14 m/s | 2.86 / 2.65 | knee 18.9 (1.00) | pitch 48 rpm (0.58) | pitch 22.5 |
| trot | sit_0.3 | impratio 10 | 28 | 1.84 / 7.35 (3 of 28 not at rest) | 0.208 / 0.219 | 0.041 / 0.052 | fore upper struts, 0.05 / 0.15 m/s, in 20 of 28 | 2.73 / 2.33 | knee 18.9 (1.00) | roll 48 rpm (0.58) | pitch 24.2 |
| trot | sit_0.2 | model | 28 | 1.76 / 4.30 (3 of 28 not at rest) | 0.188 / 0.209 | 0.050 / 0.071 | fore upper struts, 0.06 / 0.12 m/s | 3.44 / 2.35 | knee 18.9 (1.00) | pitch 48 rpm (0.58) | pitch 23.2 |
| trot | sit_0.2 | impratio 10 | 28 | 1.66 / 7.31 (1 of 28 not at rest) | 0.208 / 0.220 | 0.040 / 0.052 | fore upper struts, 0.06 / 0.09 m/s, in 18 of 28 | 4.11 / 2.32 | knee 18.9 (1.00) | pitch 47 rpm (0.57) | pitch 22.6 |
| trot | collapse | model | 28 | 1.94 / 7.46 (11 of 28 not at rest) | 0.191 / 0.224 | - | fore upper struts, 0.05 / 0.13 m/s, in 17 of 28 | 4.43 / 5.50 | none (zero torque) | knee 77 rpm (0.93) | pitch 43.1 |
| trot | collapse | impratio 10 | 28 | 1.35 / 5.79 (1 of 28 not at rest) | 0.207 / 0.234 | - | fore upper struts, 0.04 / 0.16 m/s, in 12 of 28 | 4.52 / 7.21 | none (zero torque) | knee 74 rpm (0.89) | pitch 43.8 |
| trot | collapse_friction | model | 4 | 0.66 / 0.79 (1 of 4 not at rest) | 0.202 / 0.230 | - | fore upper struts, 0.03 m/s, in 1 of 4 | 3.00 / 2.63 | none (zero torque) | knee 64 rpm (0.77) | knee 21.2 |
| trot | collapse_shorted | model | 4 | not at rest | 0.191 / 0.191 | - | none | 1.65 / 0.00 | none (zero torque) | pitch 25 rpm (0.30) | elbow 6.8 |

**From standing, the slow part of the model contact's drop is creep.** Source: the settling table, section
"Settling from 1 s after the reference" in `sit60_tables.md`.

- The chassis is at 0.237 m 0.1 s after the cut, and at 0.220 m (model contact) or 0.225 m (impratio 10) 1 s after
  it.
- Under the model's contact it then keeps sinking: 0.214-0.215 m at 2 s, 0.192 m at 4 s, 0.188-0.189 m at the end.
  Seed 1's hard sit, for example: 0.2583 m at the cut, 0.2368 m at +0.1 s, 0.2261 m at +0.5 s, 0.2200 m at +1 s,
  0.2145 m at +2 s, 0.2081 m at +3 s, 0.1917 m at +4 s, 0.1884 m at 8 s.
- Over that phase the most-travelled sole slides 66-69 mm. The soles slide apart: in seed 1's hard sit, from 1 s
  after the cut to the end, the fore soles move 51-54 mm forward and the hind soles 38-44 mm back, while the chassis
  moves 20 mm back (a scratch probe that records the soles' positions in that run; simulation).
- In the sits the soles slide faster than 2 mm/s in 88-91 % of the phase's sole-samples (four soles, 1 ms each; 84 %
  in the collapse).
- Their median friction use is 0.37-0.44 in the sits (0.34 in the collapse), with 95 % at or under 0.79, against the
  coefficient of 0.85. So the soles creep while held well inside the friction limit. MuJoCo's soft contact allows
  that; a real sole on a dry floor would not be expected to do it.
- Under impratio 10 the same phase sinks 4-6 mm. The soles slide 9-10 mm in all, about 1.5 mm/s, mostly under the
  2 mm/s threshold (164-213 sliding sole-samples), and they carry the robot alone.
- The fore upper struts first touch the floor 3.7 s after the cut (4.18 s after the stop input for the hard sit,
  5.18 s for the damped sit), inside the creep. So from standing the strut landing exists only under the model's own
  contact.

**From the trot, the struts land under both contacts.** Source: the same settling table and the adopted rows of the
tables, 28 starts per stop under each contact.

- 1 s after the cut the chassis is at 0.208-0.222 m (median 0.212 m) under the model's contact and 0.209-0.227 m
  (median 0.221 m) under impratio 10.
- Paired at the same start (seed and trigger), the model's contact is a median 4.0-5.6 mm lower at that time,
  depending on the stop. The gap depends on the phase:
  - at 6.14 and 6.49 s both contacts have the chassis at 0.208-0.211 m;
  - at 6.04, 6.09, 6.39, 6.44 and 6.54 s impratio 10 has it 11-18.5 mm higher in 16 of the 30 sits there (the
    0.2 s sit, seed 2 at 6.39 s: 18.5 mm; the hard sit, seed 2 at 6.39 s: 0.2086 m under the model's contact,
    0.2261 m under impratio 10);
  - in four runs (at 6.14 and 6.59 s) impratio 10 has it 0.6-2.1 mm lower.
- The fore upper struts land in all 84 sits under the model's contact, 0.6-3.6 s after the cut (median 1.0 s). Under
  impratio 10 they land in 53 of 84 sits, 0.5-7.3 s after the cut (median 1.0 s), at 11 of the 14 phases: never at
  5.94, 5.99 or 6.29 s, in all six sits at six of the phases.
- Where they land, they carry a median 0.41 of the weight at the end under the model's contact and 0.28 under
  impratio 10 (up to 0.52 and 0.42).
- After the first second the chassis sinks a further 7 mm (median) under both contacts, up to 33 mm under the
  model's contact and 18 mm under impratio 10.
- Under impratio 10 the soles also slide after the first second from the trot: a median 15-16 mm per stop, up to
  39.5 mm. They do so at a friction use close to the coefficient (median 0.70-0.71 per stop, 95th percentiles up to
  0.85), nearer a real slip than the model contact's creep (median 0.35).
- The model contact's largest drops are at the two headline phases (5.94 and 6.29 s): 0.059-0.071 m, of which
  0.018-0.033 m is further sink after the first second, as from standing. Impratio 10 drops 0.039-0.045 m there.
  At the other twelve phases the two contacts' drops are close (0.041-0.052 m under impratio 10, 0.050-0.058 m
  under the model's contact, in the hard sit).

**What holds the unpowered legs (simulation, the end of the 8 s window).** In every adopted-model stop, some of the
folded legs' joints rest on their hard stops. The counts below are of runs in which at least one joint of the kind
is on its stop, as in the tables' column "runs ending with joints on their range limits". Both sides together are
counted separately (from the raw run, sha256 49f06e2c...), and from the trot they are often fewer, most of all under
impratio 10:

- At least one elbow and one knee on the +36 deg fold stop: all runs, except the elbows in 2 of 28 trot collapses
  under the model's contact.
  - Both elbows: in every standing stop except `collapse_friction` (one elbow in each of its two runs). From the
    trot, in 23, 25 and 28 of 28 sits (damped, hard, 0.2 s) and 20 of 28 collapses under the model's contact, but
    only in 9, 7 and 11 of 28 sits and 16 of 28 collapses under impratio 10.
  - Both knees: in every standing stop. From the trot, in 23-28 of 28 sits and 18 of 28 collapses under the model's
    contact, and 19-21 of 28 sits and 20 of 28 collapses under impratio 10.
- At least one hind pitch joint on its -38 deg stop: all runs, except 4 of 28 trot collapses and 3 of 4 trot
  `collapse_friction` runs under the model's contact, and 5 of 28 trot collapses and 1 of 28 trot damped sits under
  impratio 10. Both hind pitch joints: in every standing stop; from the trot in 20-25 of 28 sits and 20 of 28
  collapses under the model's contact, and 23-25 of 28 sits and 15 of 28 collapses under impratio 10.
- At least one roll joint on a range limit: all runs except the standing shorted-phase case. All four roll joints
  at once: in no run under the model's contact, and in at most 2 of 28 trot runs per stop under impratio 10.

The two contacts differ in which other joints reach a stop:

- Under the model's contact, from standing, both fore pitch joints also reach their -38 deg stop in every stop but
  `collapse_friction`, as the forelegs swing forward relative to the body in the creep (in seed 1's hard sit, from 0
  to -38.2 deg; the probe above). From the trot at least one does in 5-17 of 28 stops (both in 4-14).
- Under impratio 10 the fore pitch joints do not reach their stop from standing. From the trot at least one does in
  10-18 of 28 stops, both in at most 2. At least one ankle reaches its 22 deg limit in every standing impratio 10
  stop (never all four) and in 8-14 of 28 per stop from the trot. Under the model's contact an ankle reaches it in
  one run only (a trot `collapse_friction`).

The largest static torque on any stop at the end is 4.2-4.9 N m from standing and 5.1-9.1 N m from the trot. With the
10 mm skid almost nothing rests on a stop (at most 1.5 N m; at most 4.9 N m on any skid copy). So without a skid, once
power is off the robot's weight is held partly by its joint stops. The stops that carry it must take the folded
robot's static load (at most 9.1 N m per joint here) and catch the fold (peaks up to 26 N m from standing and 44 N m
from the trot).

**Drives during the sit (simulation).**

- From standing, every drive stays at or under 0.41 of its torque limit (hind knee 7.7 N m in the 0.2 s sit) and at
  or under 0.50 of its speed allowance (hind knee 41 rpm).
- From the trot, after the first 20 ms, the hind knees reach their limit (18.9 N m, 1.00) in the hard and 0.2 s sits
  under both contacts. A drive group is at its torque limit for at most 0.06 % of its samples over the stop (0.04 %
  under the model's contact). Speeds stay at or under 0.58 of the allowance (the pitch drives at 48 rpm, and under
  impratio 10 the roll drives at 48 rpm).
- In the first 20 ms the pitch drives are at their 21.0 N m limit. That is the trot's own stance torque at the stop
  input.
- No sit exceeds a limit: the model clamps each drive at 0.70 of the maker's peak.

**The collapse (simulation).**

- Zero torque lets the legs fold fastest: up to 67 rpm at the hind knee from standing (0.81 of the allowance, under
  both contacts), and 77 rpm from the trot (0.93; 74 rpm and 0.89 under impratio 10).
- These speeds are not commanded, and they stay under the allowance.
- The joints' range limits catch the folding legs at up to 25.9 N m from standing, and from the trot at up to
  43.1 N m under the model's contact and 43.8 N m under impratio 10 (the pitch joints, seed 2 at 6.14 s). In the sits
  the largest is 17.7-24.2 N m.
- From the trot the fore struts land in 17 of 28 collapses under the model's contact and in 12 of 28 under
  impratio 10, where the leg links take up to 7.2 body weights (seed 2 at 6.04 s; 5.5 under the model's contact).
- From the trot, 11 of 28 collapses are still moving after 8 s under the model's contact, against 1 of 28 under
  impratio 10, where the other 27 are at rest by 5.8 s (median of those 27, 1.35 s). That difference is mostly the
  soft contact's.

The two sensitivity cases, run under the model's contact only:

- **Back-drive friction** (X4 0.2, X6 1.6 N m). Standing, the body ends at 0.190-0.191 m after 7.1 s, and the struts
  land at 0.07 m/s. From the trot the body ends at 0.202-0.230 m.
- **Shorted phases** (5.4 and 7.2 N m s/rad).
  - The fold slows to at most 5 rpm from standing (25 rpm from the trot), and nothing lands.
  - The creep goes on, and the body ends lower, at 0.182 m.
  - By the body measure both standing runs are at rest at 7.3 s, but the legs of one of them are still moving at 8 s.
  - All four trot runs are still moving at 8 s.

## With a skid under the belly (test copies; simulation)

Each cell gives the peak skid floor force in body weights and the body's vertical speed when it first touches, as
median / max over the row's runs. Model contact unless marked; 28 trot starts unless marked.

| skid (test copy) | standing, damped sit | standing, hard sit | standing, 0.2 s sit | standing, collapse | trot, hard sit | trot, collapse |
|---|---|---|---|---|---|---|
| 5 mm | 1.26 / 1.28 BW, 0.05 / 0.05 m/s | 3.91 / 3.92 BW, 0.15 / 0.15 m/s | 5.00 / 5.16 BW, 0.21 / 0.22 m/s | 14.39 / 14.57 BW, 0.58 / 0.59 m/s | 3.15 / 4.59 BW, 0.16 / 0.30 m/s | 13.10 / 16.65 BW, 0.60 / 0.65 m/s |
| 10 mm | 1.73 / 1.78 BW, 0.07 / 0.07 m/s | 5.28 / 5.40 BW, 0.21 / 0.22 m/s | 8.09 / 8.22 BW, 0.30 / 0.31 m/s | 13.42 / 13.57 BW, 0.56 / 0.57 m/s | 3.91 / 5.45 BW, 0.22 / 0.29 m/s; 3 tipped | 12.73 / 16.45 BW, 0.58 / 0.60 m/s |
| 10 mm, impratio 10 | 1.70 / 1.72 BW, 0.07 / 0.07 m/s | 5.34 / 5.56 BW, 0.21 / 0.21 m/s | 7.26 / 7.65 BW, 0.31 / 0.32 m/s | 13.70 / 13.92 BW, 0.56 / 0.56 m/s | 3.89 / 5.40 BW, 0.21 / 0.27 m/s; 1 tipped | 12.53 / 15.98 BW, 0.59 / 0.63 m/s |
| 20 mm | 2.09 / 2.31 BW, 0.08 / 0.09 m/s | 6.44 / 6.47 BW, 0.28 / 0.28 m/s | 8.99 / 9.00 BW, 0.39 / 0.40 m/s | 13.08 / 13.69 BW, 0.50 / 0.50 m/s | 5.36 / 7.14 BW, 0.26 / 0.31 m/s | 11.86 / 15.40 BW, 0.53 / 0.57 m/s |
| 10 mm, wide | 1.76 / 1.78 BW, 0.06 / 0.06 m/s | 5.22 / 5.34 BW, 0.21 / 0.21 m/s | 8.08 / 8.21 BW, 0.30 / 0.31 m/s | 13.42 / 13.57 BW, 0.56 / 0.57 m/s | 4.22 / 5.11 BW, 0.22 / 0.29 m/s | 12.25 / 15.15 BW, 0.58 / 0.61 m/s |
| 10 mm, soft (4 trot starts) | 1.10 / 1.10 BW, 0.08 / 0.08 m/s | 2.03 / 2.05 BW, 0.25 / 0.25 m/s | 2.43 / 2.52 BW, 0.37 / 0.38 m/s | 4.16 / 4.35 BW, 0.68 / 0.69 m/s | 1.51 / 1.53 BW, 0.25 / 0.26 m/s | 4.72 / 4.75 BW, 0.73 / 0.73 m/s |

- **Nothing touched before the stop.** In all 636 runs with a skid, the skid touched neither the floor nor any leg
  before the stop input. This was checked at each 10 ms control step of the walk and the stand.
- **After the cut the body stays on the skid.** The chassis ends at 0.264 m (5 mm skid), 0.269 m (10 mm) or 0.279 m
  (20 mm). The drop after the cut is at most 0.005 m in every run that did not tip, and 0.107-0.108 m in those that
  did.
- **The early-contact depth sets the landing.**
  - A skid that reaches deeper is met earlier, while the sit is still moving fast. Hard sit from standing at 5 / 10 /
    20 mm: 0.15 / 0.21 / 0.28 m/s and 3.9 / 5.4 / 6.5 BW.
  - Landing earlier also takes load off the drives before the cut, so the bus surplus falls: 2.20 / 1.83 / 1.09 J.
  - Sources: the tables' sections "skid_e5mm", "skid", "skid_e20mm" and "The skid's early-contact depth".
- **Tip-overs.**
  - Under the model's contact the hard sit tipped the 10 mm skid copy onto its side in 3 of 28 trot starts: seed 1 at
    6.34 s, and seed 2 at 5.99 and 6.34 s (phases 0.843 and 0.343). The roll reached 104 deg and the chassis ended at
    0.166 m. The fore X6 elbow housings struck the floor at up to 806 N and the hind X4 pitch housings at up to 694 N.
  - Under impratio 10, at the same 28 starts, it tipped in 1: seed 1 at 6.34 s (roll 104.2 deg, chassis 0.166 m; fore
    X6 elbow housing 796 N, hind X4 pitch housing 628 N). Seed 2 at 5.99 and 6.34 s does not tip there (roll 6.5 and
    6.9 deg), nor does seed 1 at 5.99 s (7.4 deg).
  - No other skid tipped: not the wide, 5 mm or 20 mm skid in 28 trot starts each, nor the soft skid in 4, nor any
    other stop on the 10 mm skid.
  - This is one routine, one skid and one to three starts out of 28: a sensitive outcome, not a rate.
- **The wide skid** (model contact) touched the hind leg struts in 10 of 28 trot collapses, up to 99 N (0.79 BW; the
  right hind lower strut took 89 N), and in one hard sit (13 N, 0.10 BW). A full-length 0.18 m skid, tried in the
  first run, touched the fore elbow drives' housings in every stop from the trot, up to 1.3 kN. That is why this
  skid's front end is cut back.
- **The soft skid** (contact at 0.03 s) cuts the landing force by more than half: 2.05 BW in the hard sit from
  standing (5.40 BW on the rigid skid), and 4.35-4.75 BW in a collapse (13.6-16.5 BW rigid).

## The electrical side (desk on simulation)

**The pulse for B7.** B7 injects "a regen pulse equal to the walk simulation's estimate for twelve drives in a hard
sit, with the ideal diodes blocking the pack", its "energy and duration ... computed in the walk simulation and
committed before B7 runs" (`BENCH_PLAN_FORELEG.md:65`). Over every variant and both contact settings:

| sit | runs | largest bus surplus J (the run) | its span s | its own peak W (1 / 10 / 100 ms) | largest peak W of any run (1 / 10 / 100 ms) | largest gross regen J | largest one-leg gross regen J |
|---|---|---|---|---|---|---|---|
| damped, 1.0 s | 216 | 1.48 (adopted, impratio 10, standing, seed 2) | 0.511 | 4.5 / 4.5 / 4.4 | 18.2 / 13.4 / 4.4 | 1.99 | 0.84 |
| hard, 0.3 s | 216 | 2.93 (adopted, impratio 10, standing, seed 2) | 0.230 | 25.9 / 25.8 / 21.7 | 26.7 / 26.0 / 21.7 | 3.71 | 1.53 |
| bounding, 0.2 s | 216 | 2.84 (adopted, impratio 10, standing, seed 2) | 0.099 | 51.6 / 49.5 / 28.4 | 68.5 / 66.0 / 28.4 | 3.55 | 1.79 |

- The largest-energy run and the largest peaks come from different runs:
  - The 26.7 W peak is from an adopted-model trot hard sit under the model's contact (seed 1, 6.14 s), which returns
    2.15 J.
  - The 68.5 W peak is from an impratio 10 trot 0.2 s sit (seed 2, 6.29 s), which returns 1.65 J.
- So a bench pulse of 2.93 J over 0.23 s at up to 26.7 W, or 2.84 J over 0.10 s at up to 68.5 W, is an envelope of
  the simulated hard sits, not a copy of one of them.
- Under impratio 10 the trot sits at all 14 phases return at most 1.78 J (hard sit, seed 2 at 6.09 s) and 1.97 J
  (0.2 s sit, seed 2 at 6.14 s), and peak at up to 26.6 W (hard sit) and 68.5 W (0.2 s sit). So the envelope's
  energy still comes from the standing start.
- The adopted model (no skid) gives the largest energies, because nothing stops the body before the sit ends. Every
  skid lowers them: the 10 mm skid's hard sit from standing returns 1.82-1.83 J under the model's contact
  (1.92-2.33 J under impratio 10).

**Against the power board (desk).** The relevant records, all from `PARTS_PREMIUM.md` as of its last change, aa3420e9,
18 Nov 2026 (the power board line `:393` is an estimate in that version):

- The board has "an active clamp at 52.5 V, under the drives' 55 V maximum, dumping into two 50 W resistors. The
  ideal diodes block regen from reaching the pack, so the clamp must absorb it" (`PARTS_PREMIUM.md:311`; line item
  `:393`).
- The pack's ceiling is 49.2 V (4.10 V per cell, `:307`). The tether is set to about 50 V (`:313`).
- B7 passes at a bus of 54 V or less, with the 52.5 V clamp and the resistors under their rating after 10 pulses in a
  row (`BENCH_PLAN_FORELEG.md:65`).

The comparisons:

- **Energy.** 2.93 J is 0.029 s of the resistors' combined 100 W continuous rating. Ten pulses are 29.3 J (28.4 J for
  the 0.2 s sit, 14.8 J for the damped sit). Whether the resistors stay under their rating after ten pulses depends
  on how fast the pulses come and on the resistors' thermal mass, which the records do not give. That is B7's
  measurement.
- **Power.** At 52.5 V the largest hard-sit peak (26.7 W) needs at most 103 ohm of clamp resistance (0.51 A). The
  0.2 s sit's 68.5 W needs at most 40 ohm (1.31 A). The records give the resistors' power rating, not their
  resistance.
- **Capacitance.**
  - 1 mF of bus capacitance takes 0.128 J from 50.0 to 52.5 V, 0.168 J from the pack's 49.2 V, and 0.263 J from 50.0
    to 55 V.
  - Keeping the worst hard sit under 52.5 V by capacitance alone would take 22.8 mF (11.1 mF under 55 V).
  - The parts list has "bus caps" but gives no value (`PARTS_PREMIUM.md:393`), so the clamp, not the capacitors, must
    take the pulse.
- **Hotel load.**
  - The 12 V rails take 12-60 V input (`PARTS_PREMIUM.md:312`, `:394`; a web-search result of 18 Nov 2026), so they
    presumably sit on the bus with the Jetson and
    the logic, and stay powered until the contactor opens.
  - At the build's 25-40 W hotel range (`gait_report.py:81`) they would take all of the hard sit's surplus but 0.01 J
    (at 25 W), or all of it (at 30 W).
  - Of the 0.2 s sit's surplus they would leave 0.90 J (25 W), 0.68 J (30 W) or 0.42 J (40 W).
- **Pack charge acceptance.** Not given in the records; the "60C" in the pack's name (`PARTS_PREMIUM.md:391`, a
  catalog listing read on 7 Nov 2026) is a discharge rating. The ideal diodes keep regen out of the pack, so the pack
  takes none by design.

**Ceilings, for a sit routine not simulated here (desk energy balance on the simulated motion).**

- The most the drives can take net while powered is the kinetic energy at the stop input plus the potential energy
  released before the cut. The largest over each sit's runs: 7.10 J (damped), 6.96 J (hard), 7.00 J (0.2 s).
- This holds for any sit routine that reaches the same height before the cut.
- The drives' net take, braking less their own positive work, is under it in every run (the largest are 6.00, 5.43
  and 4.99 J), and so is the modelled bus surplus.
- It is not a ceiling on the gross braking at the joints, because a drive can brake out work it put in. The gross
  braking exceeds it in 181 of 216 damped sits, 54 of 216 hard sits and 55 of 216 0.2 s sits (the largest are 7.37,
  8.20 and 8.17 J). A routine that pushes and brakes in turn could, likewise, return more than the ceiling.
- A routine that lowered the body all the way to where the cut leaves it, while still powered, has a ceiling of
  16.7 J (0.17 s of the resistors' rating).

**In this model the collapse puts nothing on the bus.** Zero torque means no current. The question for the real drives
is whether their back-EMF can push current through their bridge diodes.

- A drive's line back-EMF reaches a bus held at 42 V at 271 rpm (X4-36-L) and 134 rpm (X6-60) at the output. That is
  from the maker's 4.3 and 16 V per 1000 motor rpm and the gear ratios 36 and 19.612 (`actuators.json:103`, `:125`,
  `:185`, `:208`; maker data read 7 Nov 2026).
- At 49.2-50 V the thresholds are 318-323 rpm (X4) and 157-159 rpm (X6). Worked from Kt instead, at 42 V, they are
  273 and 234 rpm.
- The fastest collapse speeds are 77 rpm (X4, hind knee) and 76 rpm (X6, front elbow). So no current flows into a
  bus held at 42-50 V by the pack or the tether.
- Into a DC link that has decayed (the contactor open and the capacitors drained), a drive with its bridge off
  rectifies at any speed, up to its back-EMF: about 12 V for the X4 and 14-24 V for the X6 at those speeds.
- Where the clamp sits relative to the contactor and the drives' own links is not in the records.

## B8 and the drop after the cut

B8 times the stop chain on one leg on a catch rig. It asks that "the leg reaches the sit pose at least 0.2 s before
power drops", and its fail line is "never ship with a stop that drops the leg" (`BENCH_PLAN_FORELEG.md:66`). The rule
was written for one leg on a catch rig, not for the whole robot in simulation. The simulated figures beside it:

- **Time in the sit pose before the cut** (the chassis within 5 mm of its height at the cut; adopted model, both
  contact settings): 0.70-0.73 s for the damped sit (cut at 1.5 s), 0.23-0.24 s for the hard sit and 0.30-0.32 s for
  the 0.2 s sit (both cut at 0.5 s).
  - Measured by the last target instead: with the 10 ms delay it reaches the drives 0.49 s, 0.19 s and 0.29 s before
    the cut (arithmetic from the routine's timing: the cut time less the sit's length less 0.01 s).
  - By the same arithmetic, with the relay at 0.5 s a sit of 0.29 s or shorter would have its last target at the
    drives 0.2 s or more before the cut. That is desk arithmetic, not a simulated result: only the 0.3 s and 0.2 s
    sits were simulated, and by the chassis measure both were in the pose 0.23 s or more before the cut.
- **The drop after the cut.**
  - With no skid the body drops 0.038-0.071 m after the cut, and the legs fold on to their stops. The body comes to
    rest on the soles and, from the trot at most phases, also on the fore upper struts, under both contact settings.
  - Under impratio 10 the drop is 0.038-0.052 m. The top of the range, 0.069-0.071 m, comes only under the model's
    contact, from its creep (standing, and the two headline trot phases).
  - With a test-copy skid reaching 10 mm below the sit-pose soles, it drops at most 0.005 m, except in the tip-overs.
  - So in this model the sit-then-cut sequence avoids dropping the body only if something under the body takes the
    sit. As modelled, B8's fail line would apply to the whole robot.
- **Not covered by B8 (the readiness review's note):** the watchdog's and the battery management system's trips.
  - D47 adds an automatic battery cut-off. The parts list wires the BMS's discharge-enable into the watchdog "so a BMS
    trip runs the same sit-then-cut sequence" (`PARTS_PREMIUM.md:303`).
  - A cut-off that opened the pack directly, without the sit, would be the collapse case above.

## What this does not show

- **Anything measured.** The drives, the floor, the soles, the relay and the power board are all models or catalog
  figures. Nothing here is a bench result.
- **The real contact.** The two contact settings bracket the result, and neither is a measured floor. The creep under
  the model's contact is a property of the solver's soft friction. A slippery floor was not run. The soles' friction
  use while they slide after the cut (each run's median, adopted model) is up to 0.51 under the model's contact and
  up to 0.84 under impratio 10, over a 0.4 floor's limit, so on such a floor the feet could slide out for real (desk
  inference, not simulated).
- **The real drives.** The drives are ideal position servos with a torque clamp: no current loop, no field weakening,
  no internal braking or regen handling. What they do on a lost command (torque-off, brake by shorting, hold) is not
  known; the collapse and its two sensitivity cases bracket it.
- **The sit routine.** One routine (level body, ankles straight down, minimum jerk), at three speeds. A routine that
  leans the body, folds the hind legs deeper first, or lowers further before the cut would give other numbers. If it
  reaches the same height before the cut, the ceilings above bound its net energy.
- **The skid.** The skid is a test geometry (a massless box), not a design. Its depth, width and stiffness, and the
  10 mm early contact, are this study's choices. The model has no bumpers, and the head and tail stay powered until
  the cut.
- **Statistics.** Two seeds. For the adopted model and the 10 mm skid, 28 trot starts per stop (14 phases x 2 seeds)
  under both contact settings; for the other skid copies 28 (4 for the soft one) under the model's contact only; the
  sensitivity cases at 4 trot starts. Nothing here is a result in the build's sense, and no rule judges it.

## For the lead (the records this would change; not edited here)

- **Proposed finding (DECISION_LOG, next free F number):** "Emergency stop of the adopted 128 mm model (simulation,
  2 seeds, standing and 14 trot phases under both contact settings; desk regen; `sim/REPORT_SIT60.md`, 27-28 Nov
  2026): the damped (1.0 s) and hard (0.3 s) sits lower the chassis 0.316 to 0.258 m and are in the pose at least
  0.70 and 0.23 s before the cut; with no skid the body drops a further 0.038-0.071 m after the cut, onto the soles
  and, at most trot phases, the fore upper struts, under both contacts (0.038-0.052 m under impratio 10; the
  0.069-0.071 m upper end is the model contact's creep). Time to rest from standing 1.7-5.5 s. A passive collapse
  falls 0.097-0.128 m from standing; its hard stops take up to 26 N m from standing and 44 N m from the trot. The
  hard sit returns at most 2.93 J to the shared bus over 0.230 s (largest peak of any hard sit 26.7 W; envelope,
  upper bound), 0.029 s of the 2 x 50 W resistors' rating; no sit reaching the same height before the cut can make
  the drives take more than 7.1 J net while powered. With a test-copy skid the drop after the cut is at most 5 mm,
  but the hard sit tipped the 10 mm skid copy in 3 of 28 trot starts under the model's contact and 1 of 28 under
  impratio 10."
- **BENCH_PLAN_FORELEG.md, B7 input:** the committed estimate in this report.
  - Hard sit: 2.93 J over 0.230 s, at up to 26.7 W over 1 ms.
  - Bounding 0.2 s sit: 2.84 J over 0.099 s, at up to 68.5 W.
  - Both are envelopes and upper bounds. 7.1 J is the ceiling for any sit that reaches the same height before the
    cut.
  - B7's "what it replaces in the model" cell ("None (not modelled ...)") and B8's ("The stop behaviour (not modelled
    yet)") could point here. B8's note would add that, with no skid, the body drops 0.038-0.071 m after the cut, onto
    the soles and, at most trot phases, the fore struts, under both contact settings.
- **README "Where things are":** a row for `sim/REPORT_SIT60.md`, and `sit60.py` in the list of simulation code.
- **The design question this raises (for the owner; not decided here): what takes the sit.** As modelled, the legs
  cannot fold far enough to put the belly near the floor. Either a skid would have to reach about 0.19 m below the
  belly shell, or the sit routine and the joint stops would have to carry the robot.

## Reproducing

```sh
python hardware/triceratops_build/sim/sit60.py run --out /tmp/sit60 --workers 2   # 888 runs, about 11 min on 2 workers; 3.8 MB, keep it out of the repository
python hardware/triceratops_build/sim/sit60.py summarise --raw /tmp/sit60/sit60_raw.json.gz \
    --out hardware/triceratops_build/sim/results/sit60_2026-11-27
python -m pytest hardware/triceratops_build/sim/tests/test_sit60.py   # 17 passed, 28 Nov 2026
```

The runs are deterministic. The 28 Nov 2026 run reproduced every one of the 700 runs of the last 27 Nov 2026 run
exactly (the same measures in each), and added the 188 impratio 10 trot runs at the triggers not run before. The raw
file is not committed.

## Revised after the independent checks

This is the first saved version. The checkers' corrections were applied before it was saved.

27 Nov 2026:

- **Every adopted-model figure is given across both contact settings** (SIT-8): the drop after the cut
  (0.038-0.071 m), the standing collapse (0.097-0.128 m) and the time to rest (1.7-5.5 s).
  - The slow phase after about 1 s past the cut is identified as soft-contact creep. New settling measures in the
    script show it: the chassis's further sink, the soles' slide and their friction use.
  - The proposed finding and the B8 section use the full range.
- **The B7 pulse is labelled an envelope** (SIT-9). The worst-energy run's own peaks (25.9 W, 51.6 W) stand beside the
  largest peaks of any run, and the summary and the tables now carry both.
- **Smaller fixes** (SIT-10):
  - it is the legs, not the body, of the shorted-phase case that are still moving;
  - the skid's contact with the floor and with the legs before the stop input is now logged;
  - the usage line names the raw file;
  - the worst speed is chosen by its share of the allowance in every row;
  - the 7.1 J ceiling is stated for routines that reach the same height before the cut;
  - the test file skips cleanly without MuJoCo.
- **Also changed:**
  - the landing-speed cell in the tables now says in how many runs a body part touched, instead of "not at rest";
  - the joints resting on their stops at the end of each stop are now recorded.

28 Nov 2026:

- **Impratio 10 at every trot start** (SIT-11). The earlier version ran the stiffer contact from the trot at only the
  two headline phases and compared it with all 14 phases under the model's contact. The impratio 10 trot cases of the
  adopted model and the 10 mm skid now run at all 14 phases on both seeds, and every impratio 10 statement was
  rewritten from the rerun. What changed:
  - from the trot, the drop after the cut under impratio 10 is 0.039-0.052 m (it was given as 0.039-0.045 m), and the
    chassis ends at 0.208-0.220 m;
  - the fore struts land from the trot under both contacts (53 of 84 sits under impratio 10), so the strut landing is
    a soft-contact result only from standing;
  - the gap between the contacts 1 s after the cut is a median 4-6 mm when paired at the same start, and depends on
    the phase;
  - from the trot the hard stops take up to 43.1 N m under the model's contact and 43.8 N m under impratio 10, and
    the impratio 10 trot rest times reach 7.4 s;
  - the 10 mm skid tipped in 1 of 28 trot hard sits under impratio 10 (the separate re-run of the four tip-over
    starts is no longer needed and was dropped);
  - the run counts, the ceilings' tallies and the largest gross regen and braking figures follow the larger run.
- **Smaller fixes** (SIT-12): the skid force ranges are over runs; the creep's direction is the fore feet forward and
  the hind feet back (the forelegs swing forward relative to the body); the standing impratio 10 sliding samples are
  164-213; the standing strut landing peaks at 2.3-2.6 body weights; the sliding share and friction use are given for
  the sits apart from the collapse; the exceptions to the hind pitch stops are listed; the friction use beside a
  0.4 floor is given as a maximum; the 10 mm skid's standing hard sit is given under both contacts; the opening cites
  the lesson that motivates the study; and the skid's contact checks have their own unit test.
- **After the final check** (SITC-1 to SITC-5; text only, no number from the run changed):
  - the catalog and parts-list figures carry their dates (maker data read 7 Nov 2026; the parts list as of
    18 Nov 2026);
  - the joints resting on their stops are counted as at least one of a kind, with both sides counted beside them;
  - the adopted-model table says which cells are maxima, and that its time-to-rest medians are over the runs that
    came to rest;
  - the standing impratio 10 collapse falls 0.097 m (it was rounded to 0.096 m); the 44 N m hard stop is from the
    trot (26 N m from standing); the 7.10 J ceiling's two largest parts come from different runs;
  - the 0.29 s sit's timing is labelled desk arithmetic, not a simulated result.
  - `sim/sit60.py`'s docstring, left byte-identical so the hash the summary records stays valid, still words the
    0.29 s figure as one that "meets" B8 and gives the catalog and parts-list figures without dates: the figure is
    desk arithmetic on the last-target measure, not a B8 result, and the figures carry the dates given above.
