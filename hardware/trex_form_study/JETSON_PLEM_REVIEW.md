# Review of PR #668: the 75 cm build with a Jetson Orin Nano Super

14 Nov 2026. This review covers the results of pull request kuds/mesozoic-labs#668, "Stage 2 study: Jetson Orin Nano
Super + Partial PLEM, walking speed ceilings, and 75 cm build budget". Its branch, `jetson-orin-plem-study`, has two
commits (f836eae and 6162b45), is stacked on PR kuds/mesozoic-labs#667 at 4571826, and merges cleanly onto it.

The owner asked for the review and said that its results need not be verified now, so nothing was re-run. Four
independent reviewers read its files against this study's records and rules. Each took one area: the closed-loop
results; the speed ceilings; the hardware and budget; and behaviour and governance.

Paths are relative to `hardware/trex_form_study/`. The PR's files are on its branch under `stage2/jetson_plem/`:
- `JETSON_PLEM_AND_SPEED.md` (the report, "JPS");
- `BUILD_BUDGET_75_JETSON.md`;
- `jetson_plem_study.py` and `jetson_plem_study.json`;
- `tests/test_stage2_jetson_plem.py`;
- a note added to `COMPUTE.md`.

Kinds of evidence, as in the reviews:
- (a) a prescribed-walk (Stage 0) or static model result;
- (b) a closed-loop simulation;
- (c) a sourced fact;
- (d) an estimate or assumption.

## In brief, for the owner

**What it is.** A study of the 75 cm Hybrid (and the 60 cm Tapered) with an NVIDIA Jetson Orin Nano Super dev kit in
place of the Raspberry Pi 5 + AI HAT+. The Jetson would drive the 12 leg motors directly over USB-CAN FD from an
isolated real-time CPU core, which the PR calls "Partial PLEM". It also covers walking-speed ceilings, closed-loop tests
under servo delay, and an itemised budget for a 75 cm + Jetson prototype.

**The ideas worth carrying into the next session**, in order of value to your goal (character with some agility, then
long training runs):

1. **A better reference walk for the 75 cm body.**
   - F25's tuned walk (the gait tuning of `round2/gait_tune/`) combined with look feature F9's counter-phased tail, plus
     a small duty and lift margin, may let Stage 1's existing tracker walk the 75 cm body at 0.40–0.50 m/s at low servo
     delay.
   - It attacks exactly the two limits that stopped that body at speed in Stage 1: the tail servo's sustained limit
     (A1), and duty.
   - If it holds under the study's official panels, it would be the first 75 cm walk faster than the nominal one to
     pass any reading.
2. **Servo delay as a design target.** An isolated real-time core with a direct CAN path to the leg drives is a
   concrete architecture to measure in the delay bench test (draft A17). The delay is the study's largest unknown.
3. **The 75 cm + Jetson package fits** (a):
   - no internal overlaps, with a 20.7 mm gap at the power box, and the exterior unchanged;
   - +78.7 g, with the centre of mass 1.68 mm forward;
   - runtime above 20 min on the 3,300 mAh pack in either Jetson power mode (an estimate);
   - the 75 cm XM540 neck servo keeps headroom for the expressive head.
4. **The Jetson's spare compute for character.** Looking at a person, tracking several people and voice: the headroom
   is sourced in the study's own compute survey (c), though not yet benchmarked for this robot.
5. **A stopping-horizon view of speed.** It ties the chest ToF sensors' floor window to a top speed, which is useful as
   a speed governor for autonomous walking.
6. **Two smaller pieces.** A sourced 6S-to-12 V regulator choice for the Jetson's carrier, and the first single budget
   table for the 75 cm build that includes the materials and power items you adopted in P21.

**What to treat with care:**

1. **The most impressive numbers are not backed by records.**
   - JPS §2.3, the 20-episode closed-loop table (0.40–0.55 m/s passing on both controllers), was added in a
     report-only commit. Neither the script nor the JSON contains it.
   - The only closed-loop result on record is a 3-episode, survival-only probe at 0.40 m/s.
2. **The headline is read at 0–2 ms of delay.**
   - By the PR's own table, every walk falls at 6–10 ms. Under your rule (P29: judge the bar at 10 ms until a bench
     value exists), the 75 cm walk at 0.40 m/s is not shown.
   - "Partial PLEM ≈ 2 ms" is an unsourced, host-side estimate. The model's delay is end to end, command to motion.
3. **Its "safe maintainable" speeds are ceilings, not demonstrated speeds.**
   - These are 0.45–0.48 m/s at 60 cm and 0.54–0.65 m/s at 75 cm, set by drive, grip and stopping limits of
     prescribed walks.
   - They are mostly interpolated or extrapolated (a, d), and are not speeds the robot has shown.
4. **The architecture changes the safety story.**
   - With the walking controller on Linux, a Linux crash ends in torque-off, which for a biped means a fall. The study's
     design keeps the real-time loop on a microcontroller so that a crash still ends in a controlled stop
     (`COMPUTE.md`; `BEHAVIOR_SPEC.md`).
   - One CAN channel may not carry 12 drives at 500–1000 Hz (a reviewer's estimate, d).
5. **The budget needs corrections.**
   - It uses a stale $249 Jetson price; the catalog and `COMPUTE.md` use $399.
   - It cites two catalog ids that do not exist.
   - It reads P4's $10–20k against the self-assembled total ($17,216). The turnkey total, $20,030, is just over P4, and
     P4 includes outsourced integration.
6. **The behaviour gains are graded beyond the evidence.**
   - "Transformative" re-identification and "< 5 ms" jaw-audio sync are estimates (d).
   - The character moves are limited by the servos, not by compute.

**Recommendation.** Keep PR #668 open and unmerged for now. The next session merges its branch into its own working
branch, records its key 75 cm results under the study's rules (section 5) and takes its decisions to you (section 6).
Merge it later, relabelled and with records, or let the next session's PR supersede it.

## 1. What the PR found, by kind

| Claim (where) | Kind | Reading |
|---|---|---|
| The two bodies rebuild from their round-two records: `xml_sha256` and Stage 1's prefix match, with 25 mass, hold and RMS comparisons (JPS intro) | (a), recorded | Sound; the script's sha256 is tested against the JSON |
| The dev kit fits the 75 cm chest: 0 same-body overlaps, +20.70 mm at the power box, crown and widths unchanged, +78.7 g (JPS §1.1) | (a), recorded | Sound for home-pose overlaps. The round-two screens (walking clearance, battery path, joint ranges, sensors, turning, carpet, envelope) were not re-run on the Jetson body, and no refine case or record exists. The doc's crown is 752.71 mm against the JSON's 750.80 mm |
| At 60 cm the default 110 mm power box hits the left hip-roll motor (−6.95 mm); a 94 mm box clears (JPS §1.1) | (a), recorded | A real conflict; the 94 mm box is a geometric placeholder, not a laid-out power bay |
| 6S-to-12 V: `pololu_d36v50f12` for the 15 W mode, `pololu_d42v110f12` for both modes (JPS §1.2) | (c) | Sourced in the catalog, and consistent with E7 and PR #662's own feed. Converter efficiency is taken as 0.93 against A6's 0.88 |
| Runtime 34.5 min (75 cm, 15 W, Partial PLEM, 3,300 mAh) and 30.7 min in the 25 W mode (JPS §1.3) | (d) | `designlab/power.py` method, reproducing the round-two record. The "+17 g" mass is arithmetic that leaves out the 68 g PCAN adapter, the supervisor and the IMU. The 67 TOPS perception figure needs the 25 W mode, which costs 6.0 min at 75 cm |
| F25 + F9: 75 cm tail at the animal pace 1.030 → 0.413 of A1; friction need down 25–36 % (JPS §2.1) | (a), recorded | Agrees with `stage2/look/SCREEN.md` (V5: 0.426). Its V0 rows equal SCREEN.md's exactly. The tuned walk at 0.496 m/s still needs μ 0.572, above 0.54 |
| Stage 0 speed ceilings: 0.481 / 0.531 m/s on today's walk (knee speed); 21 V drive ceilings; μ 0.9 grip ceilings; "25.2 V burst" (JPS §2.1) | (a, d) | Hand interpolations and extrapolations of six evaluated speeds. The 75 cm values lie beyond the last one. The 21 V and 25.2 V ceilings drop the speed margin; the "burst" does not exist in the model |
| Chest ToF safe-stop ceilings: 0.485 → 0.550 m/s (60 cm) and 0.540 → 0.610 m/s (75 cm) with the Jetson (JPS §2.2) | (d) | Every timing term is assumed. The Pi's 200 ms is the spec's deadline, not a measured latency. The gain comes from routing the ToF stop through the camera pipeline, which the study's design does not do (`COMPUTE.md`: the stop runs on the microcontroller) |
| Recorded probe: F25 + F9 at 0.40 m/s, 3 episodes, survival only; stays up at 0 / 2 / 6 / 10 ms in 3, 3, 2, 0 (60 cm, R2s) and 3, 3, 3, 1 (75 cm, R2) of 3 (JSON) | (b), recorded | A hint, not a verdict. 3 of 3 bounds survival only above 0.37, at 95 % (20 of 20: 0.86). Its start transition is hand-built (2 ZMP corrections instead of 6, `ik_ok` forced) and its seeds are not the panel seeds |
| 20-episode decomposition on 75 cm with R2/42 and R2/43 (JPS §2.3) | (b), **not recorded** | Shown on both grips at 0.40 / 0.45 / 0.496 m/s at 0 ms (F25_F9_cl: duty 0.59, lift 0.095 H); 0.55 m/s at 0 ms; 2 ms passes; 4 ms meets every limit in 5 · 1 (toe clearance); 6–10 ms 0 / 20; left nudge at 0.40 m/s and 2 ms keeps 15 · 15 of 20 up; no 90° turn run. None of this has code or a record |
| "Jetson mass does not matter": +397 g changes speed by less than 0.001 m/s (JPS §2.3) | (b), **not recorded** | The study has no `plant.trunk_mass`; `env.set_trunk_mass` scales mass without moving the centre of mass. The +397 g has no derivation |
| Behaviour impact grades: re-ID "transformative", camera-pose skew "< 1–2 ms", jaw-audio sync "< 5 ms" (JPS §3) | (d) | Unsourced. The compute survey records that no Orin Nano re-ID benchmark exists. The character moves are servo-limited: about 0.40 s for 0.6 rad on the STS3215, and no quick snap (`stage2/x/X_COSTS.md`) |
| Budget: 75 cm + Jetson $12,586 / $17,216 / $24,077 self-assembled; $14,282 / $20,030 / $28,215 turnkey; swap delta +$50 to +$373 (`BUILD_BUDGET_75_JETSON.md`) | (d), cost model | Reuses `designlab/cost.py`'s lines and A8. Uses $249 for the Jetson (catalog $399). Two ids do not exist (`feetech_sts3215_12v`, `robotis_xm540_w270`; the catalog has `feetech_sts3215_c018`, `robotis_xm540_w270_t`) |

## 2. Strengths

- **It holds to the study's provenance practice.** Everything sits in a new directory, and no `designlab`, `stage1` or
  `round2` file changes. Both bodies are rebuilt from their records, with sha256 checks, and the JSON is pinned to the
  script's hash.
- **Its reference-versus-compute decomposition is the right experiment.** It holds the body, both Stage 1 controllers
  and the 20 seeds fixed, and changes one thing per row: reference, then duty and lift, then delay, then mass.
- **Its F9 is a faithful shortcut.** A negative tail amplitude at today's phase is a half-cycle shift. The study's own
  tail check puts the best phase 0.468 cycles from today's, and the plain flip comes within 0.2 % of it in peak yaw
  moment.
- **It treats servo delay as the deciding unknown, as the study does.** Its recorded probe agrees in direction with
  Stage 1's 75 cm delay ledger (up to 8 ms, falls at 10 ms).
- **The 75 cm packaging result is clean and useful.** The 60 cm conflict it found is real.
- **It reports its own neutral and negative findings.** These cover grip, turn range, carpet and bump recovery.
- **It takes up the combination the owner now wants explored.** P3 originally asked for Jetson autonomy, and
  `COMPUTE.md`'s first bench test measures "the current dev kit" with no purchase.

## 3. Caveats

- **Unrecorded evidence.** JPS §2.3, the trunk-mass rows and every "3/3" cell in §2.1 arrived in commit 6162b45, which
  changes only the report. The script runs one speed at delays of 0, 1, 3 and 5 physics steps. `F25_F9_cl` exists
  only in the text. None of it can be traced to code or record keys, against the study's rule that reports are made
  from records.
- **The judged delay.**
  - P29 and S2-7 judge the bar at 10 ms until a bench value exists and never read between delays. At 10 ms the PR's
    own §2.3 has every walk falling.
  - The PR calls 6–10 ms "serial-bridge latencies" and equates 2 ms with Partial PLEM. Neither is sourced.
  - In `COMPUTE.md`'s design the Teensy runs the walking loop and the CAN bus, so there is no bridge hop inside the
    servo loop.
- **The judging rules.**
  - The claimed rows use Stage 1's tuner measure (`tune.measure`), not Stage 2's bar:
    - toe clearance is read on the reference's clock, so the 4 ms misses may partly be that artefact (S1b N5);
    - drift is reported beside success, not inside it;
    - the stop uses Stage 1's 1.5-stride rule;
    - nudges were run forward and left only;
    - μ 0.54 was run at 0 ms only;
    - no 90° turn was run.
  - The PR touches 5 of the bar's 16 gated conditions, one of them at 10 ms.
- **Measured speeds quoted as targets.** "0.513–0.516 m/s" and "0.514 m/s" are medians achieved in a 0.496 m/s walk
  commanded 5 % fast. The study corrected the same mixing in its own Stage 1b report.
- **Tuned after seeing results.** F25_F9_cl's duty (0.59) and lift (0.095 H) were set after F25 + F9's results on the
  evaluation seeds. They are sensible, but the change has to be labelled, and that walk was never screened at Stage 0.
- **The bus.**
  - Twelve drives, each with a command and a reply, make 24 classic CAN frames per cycle: about 2.7–3.2 ms at 1
    Mbit/s. One channel would therefore run at about 133–162 % load at 500 Hz (a reviewer's estimate, d).
  - The study's build uses two CAN buses.
  - The neck and tail run on a TTL bus through a USB adapter, whose latency is unmeasured.
- **Safety.**
  - Under Partial PLEM a Linux failure ends in torque-off, which means a fall. `BEHAVIOR_SPEC.md` asks for a controlled
    stop and a stand when the heartbeat is lost.
  - The PR does not discuss this.
- **Modelling gaps in the Jetson body.**
  - Every dynamics result uses the Pi body.
  - The 0.18 kg Teensy control stack stays in the model.
  - The PCAN adapter, the regulator and the supervisor are never placed.
  - Heat (15–25 W in a closed PA11 chest, against the Pi's 8–9.5 W) is not addressed, nor is a separate 12 V rail for
    the Jetson and the servos.
- **Small slips.**
  - The doc and the JSON disagree on the crown (752.71 against 750.80 mm) and on the 60 cm mass delta (+74.2 against
    +79.4 g).
  - The doc gives delays "0, 2, 4, 6, 10 ms" where the probe ran 0, 2, 6 and 10.
  - "PLEM" and "WIM-Corporation" are not defined or sourced.
  - The tests read back the JSON. None recomputes a number or covers §2.3, §3 or §4.

## 4. How it fits this study's findings

- **Agrees.**
  - The 75 cm brisk walk with today's reference stays up and misses only the tail's A1 (records: 20/20 up, 0/20 meeting
    every limit, tail 1.043–1.053).
  - F9's phase cuts the tail load and the friction need (`stage2/look/SCREEN.md`).
  - F25's own caveats predicted the duty and toe-clearance trouble: lift at 0.08 H with no margin, and duty at the
    spec's minimum.
  - Duty 0.59 sits inside Stage 2's expected 0.57–0.62 at 0.40 m/s.
  - The walking-nudge edge matches what Stage 1's tracker recovered.
- **Agrees on which limits bind, not on every number.** At the 75 cm animal pace the PR gives median speeds of 0.506 /
  0.490 m/s. The records have 0.5054 / 0.4949, and R2/43's duty is 0.5204, not 0.518. The PR also leaves out R2/43's
  speed, toe-clearance and flight misses.
- **Disagrees.**
  - The judged delay (P29, S2-7).
  - The rule that reports come from records.
  - The rule to report targets, not measured speeds.
  - `COMPUTE.md`'s real-time layer on the microcontroller.
  - Stage 2's bar readings and coverage (`STAGE2_PLAN.md` §5).

## 5. The next session's 75 cm + Jetson track

This is in order. Each task says when it is done. Everything here is simulation or planning, apart from the bench sheet
the owner may run.

1. **Bring the PR in.**
   - Merge `origin/jetson-orin-plem-study` into the working branch; it adds files only.
   - Do not cite its numbers except through this review's labels until task 3 has records.
   - Done when its files are in the branch and the full test suite still passes.
2. **Record its references.**
   - For `75-M-H-10`, generate F25 + F9 (duty 0.55, lift 0.081 H) and F25_F9_cl (duty 0.59, lift 0.095 H, crouch
     0.05 H). Use F9's recorded phase from `stage2/look/tail_phase.json`, with the PR's flipped tail as a check.
   - Generate them at the nominal speed and at 0.30–0.50 m/s in 0.05 m/s steps.
   - Build their starts, changes and two-step stops with the official builders (`transitions.generate`), not by hand.
   - Done when every rung has hashed arrays, transitions with `ik_ok` and ZMP error recorded, and a
     `designlab.stage2.screen` block with every gate.
3. **The official panels for the walk** (this turns JPS §2.3 into evidence).
   - On `75-M-H-10`, run F25_F9_cl, with F25 + F9 and V0 as controls, under R2/42 and R2/43 unchanged.
   - Seeds 3042–3061, commanded 5 % above target, `WALK_RULE`'s plan, μ 0.9 and 0.54.
   - Delays 0, 2, 4, 6, 8 and 10 ms (whole physics steps), with decision 20's attribution wherever there are 5 or more
     falls, and Stage 2's readings (toe clearance over each foot's own swing, drift inside success).
   - Label the duty and lift margin as set after results.
   - Done when code and records with keys exist for every delay and grip, each condition has its counts, lower bound
     and verdict, and a seed-by-seed table compares the result with JPS §2.3.
4. **The rest of the bar on 75 cm** at 10 ms (also 0 and 4 or 6 ms), both grips:
   - the 90° walking turn from 0.40 m/s (new π/4 rad/s references in both directions, with the thigh twist's range
     checked first);
   - nudges in four directions at 0.25 × the Stage 0 bound;
   - the stop read as at most two touchdowns.
   - Done when all 16 gated conditions have records under both controllers.
5. **R2L on `75-M-H-10`** (`STAGE2_PLAN.md` B.2): Stage 1b's speed fit with 0–10 ms in its mix and fixed delay draws,
   on the approved look reference and, if the owner allows it, on F25_F9_cl. Then repeat task 3's 10 ms walk. Done when
   R2L's walk at 0, 5 and 10 ms is judged beside R2/42 and R2/43 on the same seeds.
6. **A Jetson body**, built as the expressive head was: refine case `round2/refine/jetson_75`, body `75-M-H-10-J`, or
   `-XJ` with the head.
   - Put the dev kit in the chest.
   - Model the control layer the owner chooses (decision 2 below): the PCAN adapter, the regulator and the supervisor
     placed, and the Teensy stack kept or removed.
   - Record a compute power row for the 15 W and 25 W modes as a draft A row.
   - Run every round-two screen, then G1–G3.
   - Done when the body passes its gates, and a paired-seed table against the Pi body reports each difference (or
     "below what this round resolves").
7. **A latency budget and bus plan per architecture:**
   - Pi 5 + Teensy with two CAN buses (`COMPUTE.md`);
   - Jetson + Teensy;
   - Jetson-direct over USB-CAN (one or two channels), over the Orin's own CAN controller with a transceiver, or over
     EtherCAT, which the catalog lists on the leg drives.
   - Cover CAN load with 12 drives, the TTL bus for the neck and tail, and what each does on a Linux crash.
   - Done when one table gives each architecture's estimated end-to-end delay, bus load and failure behaviour.
8. **The delay bench sheet (draft A17)** for the owner.
   - Measure the end-to-end command-to-motion delay (p50, p99, max) for each drive type: X6-P20-60-E, X4-P36-36-E-L,
     XM540-W270 and STS3215.
   - Measure it on both paths: the Teensy CAN loop, and an isolated PREEMPT_RT core on the Jetson with USB-CAN.
   - Run at the model's 500 Hz target rate with all 12 drives on the bus, and with the GPU loaded.
   - Add servo stiffness, the ToF stop's timing and tread grip on the owner's floors.
   - Done when the sheet maps each reading to a simulated delay and lists parts and time. `COMPUTE.md`'s first bench
     test already uses the current dev kit.
9. **The budget, rebuilt with `designlab/cost.py`.**
   - Use catalog prices ($399 Jetson) and corrected ids.
   - Put P21's items as their own lines, and the `-X` yaw bearing and the audio as optional lines.
   - Compare the turnkey nominal with P4.
   - Done when a recorded table and a list of quotes to ask for exist.
10. **A speed governor, not a speed claim.**
    - Recompute the ToF stopping horizon per stop path (microcontroller, the Jetson's real-time core, the host planner)
      and per ToF mode, from sourced timings.
    - Label it as the limit autonomy sets on speed.
    - Done when a governor table exists per architecture.

## 6. Owner decisions it raises

Recommended options come first.

1. **Compute for the 75 cm build.**
   - Keep P14 (Pi 5 + AI HAT+), or move toward P3's original Jetson (the Orin Nano Super dev kit)?
   - Recommended: study the Jetson as a refine case (task 6) and decide after the screens and the delay bench test. The
     PR's closed-loop evidence shows the walk's sensitivity to delay, not an advantage for the Jetson as such.
   - If the Jetson is chosen, decide its power mode: 15 W, or 25 W, which costs about 6 min of walking.
2. **Where the real-time loop lives.**
   - A microcontroller, as in `COMPUTE.md` (a controlled stop survives a Linux crash).
   - An isolated real-time core on the Jetson with a supervisor (Partial PLEM; a crash ends in torque-off).
   - EtherCAT.
   - Recommended: keep the microcontroller as the baseline, and let the bench measure the alternatives.
3. **The judged delay.**
   - Keep judging the bar at 10 ms until a bench value exists (P29, S2-7), and treat 2 and 4 ms results as sensitivity
     readings.
   - Recommended: yes.
4. **When the bench tests run** (`STAGE2_PLAN.md` decision 8).
   - The delay measurement now decides both the agility bar and the compute architecture.
   - Recommended: first, on both paths, before any compute purchase.
5. **References.**
   - May Stage 2's frozen reference use F25's tuned walk with F9's tail phase and the duty and lift margin, as a
     labelled candidate shown at L1 beside the look ladder?
   - Recommended: yes, as a candidate, with its Stage 0 price and clips.
   - Its costs: duty near the band's edge, toe clearance near the 0.04 H gate, flat hard floors only, and peak
     centre-of-mass acceleration about twice the spec's (F25).
6. **Scope.**
   - Concentrate Stage 2 on `75-M-H-10` (and its `-X`), or keep training both picks as P29 decided?
   - Concentrating would cut Colab runs roughly in half.
   - P25's 60 cm question stays open either way.
7. **The budget under P4.**
   - With P21's items, the 75 cm build is about $19.6k nominal turnkey before the Jetson, and about $20.0–20.2k with it
     at catalog price ($28.2k high).
   - Accept it, trim it (defer the boom or the audio), or raise the ceiling?
8. **PR #668 itself.**
   - Recommended: keep it open and unmerged. The next session records its results (tasks 2–4).
   - Then either merge it relabelled, with records, or let the next session's PR supersede it.

## 7. If PR #668 is merged later

- Back JPS §2.3, the trunk-mass rows and every "3/3" cell with committed code and records, or remove them.
- Put the 10 ms reading first. Restate every speed as a target, with the delay, grips and seeds it was judged at.
- Label every number in §3 and §4 with its kind. Replace the grades ("transformative") with what was measured.
- Fix the Jetson price ($399), the two catalog ids, the crown and mass slips and the P4 reading.
- Add a dated note to the `COMPUTE.md` pointer naming its conflict with the microcontroller real-time layer.
- Make the tests recompute at least the packaging and power numbers.
