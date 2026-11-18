<!-- Working paper, 16 Nov 2026: the design note for the servo model of gate 1 (BUY_GATE_DRAFT.md; DECISION_LOG.md P39, P40). It was reviewed, revised, checked a second time, and revised again; the second check's fixes and the lead's rulings are in and were not checked a third time. It proposes a design that is not built. It sets no rule, and nothing in it is a result or a measurement on hardware. -->

<!-- Working paper, 16 Nov 2026: the designer's note for the servo model of gate 1 (BUY_GATE_DRAFT.md; DECISION_LOG.md P39), written from the three readers' reports beside it. REVISED after two independent checks on 16 Nov 2026 (REVIEW.md, RECHECK.md); the second check's fixes and the lead's rulings of that day (LEAD_RULINGS.md) are in and were not checked a third time. It proposes; it sets no rule, and nothing in it is a result or a measurement on hardware. -->

# Servo model for gate 1: design note

**Status.** Working paper for the study's lead, 16 Nov 2026. It proposes. It sets no rule and changes no file. Revised after two independent checks on 16 Nov 2026 (`REVIEW.md` beside this note: "usable after the listed fixes"; `RECHECK.md`: "ready after the listed fixes"). The second check's fixes are in and were not checked a third time. The lead's rulings of 16 Nov 2026 on the open choices (`LEAD_RULINGS.md`) are written into section 8 and carried through the note; they are the lead's proposals for a design that is not built, and what is the owner's stays the owner's. What changed is listed finding by finding in section 10. Body `75-M-H-10` only, as its record has it: 14 drives and the Raspberry Pi 5 as computer. Not covered: the 60 cm body (section 8, choice 14); the optional expressive head's neck-yaw servo, a second STS3215 with no row here (`stage2/jetson75/BUDGET_75.md:318`; the head as decided has the neck-yaw servo only, `:312-314`); the Jetson body's record, whose auxiliary power differs (section 2c).

**Three kinds of statement, kept apart.**
- *Prospective design*: sections 2 to 6, 8 and 9, their summary in plain words in section 11, and the questions of appendix A. None of it is built.
- *Simulation evidence*: what the study's code and records show. Marked [S], or [P] where I printed it today.
- *Measured on hardware*: the study has none. No number here was measured on a part of this build.

**Marks on numbers.**

| Mark | Kind |
|---|---|
| [M] | The maker's catalogue value, as the study's research record transcribes it. I did not re-open the maker's pages. |
| [T] | Measured or identified by a third party on its own unit. "(file)": I read the saved file. "(summary)": reader 3's web-search summary only. |
| [A] | An assumption of this study, with its decision-log id where it has one. |
| [E] | My own estimate or arithmetic. |
| [S] | Simulation: the study's code or a record. |
| [P] | Printed by my read-only probes today. They load the model and call the forward pass. No physics step, no episode, no tracked file changed (`git status` clean). |
| [R] | Printed or computed by the independent reviewer (`REVIEW.md`). "(checked)": the fixer repeated the arithmetic or reran the probe on 16 Nov. "(not rerun)": taken from the review as it stands. |
| "second revision" | A number in a sentence so marked was recomputed by the finisher on 16 Nov (`calc_final.py`, arithmetic only) or printed by a read-only probe rerun that day. None is taken from the second check unchecked. |
| no source | Nothing found. The value is a placeholder with a wide range. |

**Files.** Paths are relative to `hardware/trex_form_study/`. RES = `designlab/data/research/actuators_2026-11-07.json`. PLANT = `stage1/runs/2026-11-08/75-M-H-10/plant.json`. REC = `round2/refine/feet_75/candidates/75-M-H-10.json`. LAT = `stage2/jetson75/LATENCY_AND_BUS.md`. BENCH = `stage2/jetson75/BENCH_SHEET.md`. MAT = `round2/refine/MATERIALS.md`. DL = `DECISION_LOG.md`. R1, R2, R3 = the readers' reports in `stage2/design/servo_model/`. Code lines are those of the merged files on the working branch at commit `82ba1ad`, read on 16 Nov for this revision (`git diff 26b057e 82ba1ad -- designlab/` is empty, so they are also the first version's lines). At the second revision the head is `cd68f2c`: `git diff 82ba1ad cd68f2c -- designlab/` adds two files (`designlab/stage2/bundle.py`, `vecpool.py`) and changes none, so the cited lines hold. They are not the lines of the training env's worktree branch, which has moved since (see the first row of the check table).

**My scripts** (in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_design/designer/`): `calc.py` (arithmetic), `probe_runtime_arrays.py`, `probe_deadzone.py`, `probe_joint_frames.py` (forward pass only), `probe_layer_cost2.py` (numpy timing, no model), `probe_ref_accel.py` (reads stored reference files). The note's first version is `DESIGN_NOTE.md` in the same folder.

**The revision's checks** (16 Nov, in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_fix/`): `calc_fix.py` (arithmetic only: the reviewer's figures repeated, and the pass probabilities of section 4); the designer's `probe_ref_accel.py` (stored references only) and `probe_runtime_arrays.py` (forward pass only) rerun as they stand. Every code, record and bench-sheet line the review cites was opened. No episode, no physics step, no fit. The branch head moved from `b57fdd0` to `82ba1ad` while this was done; the one file that changed is `stage2/design/checkpoint/STATUS.md`.

**The second revision's checks** (16 Nov, in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_final/`): `calc_final.py` (arithmetic only; it reads the tail's gains from PLANT). The second checker's `probe_ref_speed.py` (stored references only) rerun as it stands. Every code, record, plan and sheet line the second check cites was opened before its finding was applied, and every number the second revision writes was recomputed first. No episode, no physics step, no fit, no web page.

**Who "I" is.** The first version was written by the designer. The revision was made by a second agent, the fixer. The second check's fixes and the lead's rulings were written in by a third agent, the finisher. "I" and "my" are the designer's, except in section 10.1 and in the sentences marked "in the revision", "[R, checked]" or "rerun on 16 Nov", where they are the fixer's. They are the finisher's in section 10.2, in appendix A, in the sentences marked "second revision", and in section 8 where a point is marked "open for the lead" or reads a ruling (choices 3, 7 and 27). Section 11 was written by the fixer and rewritten by the finisher.

**What I checked in the readers' reports, and what I correct.**

| Claim | Check | Result |
|---|---|---|
| R1: the training env exists only on the worktree branch | `cmp` of `r5env.py`, `r5core.py`, `r5lib.py`, `g8a.py` | Merged since (commit `9ecefa5`). R1's line numbers hold in the merged files in `designlab/stage2/`, and this note cites those. The worktree branch has moved on and is not merged (`wt/wpc-ce`, at `cdc4762` at the second revision; its `r5env.py` was last changed by `4523ed6`): its `r5env.py` differs from the merged one in 148 lines, the reset's lines 431 and 504-505 are 468 and 541-542 there, and it adds `RESUME_KEYS`, the keys a resumed run compares. The `servo` field of section 5 must pass through that comparison and through `config_from` (`r5env.py:936`). |
| R1: `Plant.step` is the only physics step; the envelope is called before it | `plant.py:687-692`; grep of `mj_step` | Confirmed. |
| R1: a new row's key carries its own `describe()` and `sources()` | `bar.py:880-918`, `1677-1705`; `panels.py:347-359` | Confirmed. |
| R1: backlash could be done by a zero force range | `control.py:2296-2365` (`_episode`) | Not usable: the judge counts a force at its range as a drive at its torque limit. Section 2b uses a command shift instead. |
| R1: a servo layer adds very roughly 4 to 12 % | my timing | 28 µs against 2.2 µs today per physics step [P]: about 10 % (section 6). |
| R2: the maker's numbers and their lines in RES | RES:55-86, 122-155, 186, 1482-1512, 1546-1579 | Confirmed as transcribed. |
| R2: no bench sheet measures gear play | BENCH:327-331, 484-594 | Confirmed. `BUY_GATE_DRAFT.md:59-61` says the bench measures it. |
| R3: third-party servo fits | the saved files in `dl_bam/`, `dl_misc/` | Confirmed for the STS3215 7.4 V fit, the MX-106 fit, the Feetech firmware law and the small biped's model. Corrected in the revision: the MX-106 file's 0.122 N·m is its low-speed (Stribeck) term; its constant term is 1.0e-5 N·m (`dl_bam/pin_bam_params_mx106_m6.json`). |
| R3 §1.2: the X6's back-EMF figure does not fit its no-load speed | RES:135, 138, 154 | Confirmed in the revision: 16 Vdc/krpm at 176 rpm × 19.612 is 55 V on a 48 V bus [E]. The first version left this out. Sections 1.2 and 3 now carry it. |
| The draft's corners (friction ×0.5 and ×1.5, play 0 and 1°, mass 0.9 and 1.1) | against section 3 | Friction corrected. Play kept at the draft's 1° on the leg drives (the first version lowered it to 0.5°; the review's B2). Mass confirmed. Section 4. |

Two things the readers found are not in the draft's list and are added here: the **reflected rotor inertia** (a 6 to 120-fold open range) and the **gain range of the leg drives' own loop**.

---

## 1. What is modelled today and what is missing

Common to all 14 drives today [S]:
- A MuJoCo position actuator: torque = kp (ctrl − q) − kv q̇, on the true state, every 2 ms physics step (`plant.py:865-868`; [P]).
- The command folds three targets into one number: ctrl = G q* + (kv G q̇* + G⁻ᵀ τ_ff) / kp (`plant.py:662-681`).
- A pure delay of k whole 2 ms steps on the whole command (`speedfit.py:324-361`; `STAGE2_PLAN.md:351-360`).
- Damping 0.2 N·m·s/rad and dry friction 0.15 N·m on every hinge, unsourced [A18] (`designlab/mjcf.py:120`; DL:124).
- Bus fixed at 24.0 V (PLANT:18). It scales the speed axis only (`designlab/catalog.py:77-82`).
- The controller reads joint angles rounded to 14 bits, rates with 0.02 rad/s of noise, one step (2 ms) old [A, no decision-log id] (`estimator.py:76-93`; `plant.py:703-705`).
- The stand-in controller's feedforward knows the simulated friction exactly (`control.py:1204-1205`, `1403-1404`).

### 1.1 Leg drive RMD-X4-P36-36-E-L: hip roll, hip yaw, knee, ankle motors a and b (10 drives)

| Aspect | Modelled today [S] | Known about the real part | Missing |
|---|---|---|---|
| Servo law | as above | CAN "motion mode": position, speed, kp, kd, feedforward torque, described for a sibling model [M via search extract, LAT:104-110, 442] | whether these drives have it; its loop rate (none found, LAT:443) |
| Gains | kp 598.5 N·m/rad (each ankle motor 299.25); kv 24.48 hip roll, 7.59 hip yaw, 11.51 knee, 2.08 ankle motor [A] (PLANT:126-192; `plant.py:858-864`) | sibling manual: KP 0 to 500, KD 0 to 5 [M via search extract, LAT:442]; units not confirmed | the caps. With them the damping ratio is 0.22 hip roll, 0.72 hip yaw, 0.48 knee [P] |
| Torque limit | ±21.0 N·m = 0.70 × maker's peak [A2] (PLANT:287) | peak 30 N·m at 10 rpm; 21 N·m lasts 44 s at standstill [M] (RES:64-65) | how long a peak lasts |
| Torque against speed | 21 N·m up to 8.69 rad/s, then a line to 0 at 11.62 rad/s; braking at peak (`drive.py:22-35`; PLANT:289-290) | rated 10.5 N·m at 83 rpm, no-load 111 rpm, 24 V [M] (RES:63-67); no curve published | a line through the maker's two points gives 21 N·m only below 5.76 rad/s [E] |
| Bus voltage | 24.0 V fixed, no sag | supply window 20 to 55 V [M] (RES:69); pack swing 21.0 to 25.2 V [A11] | sag; the 20 V floor |
| Friction | 0.15 N·m + 0.2 N·m·s/rad per hinge [A18]; at the ankle on pitch and roll, none on the motors [P, R1] | backdrive torque 0.2 N·m; efficiency 64.7 % [M] (RES:85). The V4 sibling: 1.14 N·m [M] (RES:52) | friction that grows with load; friction on the ankle motors |
| Reflected inertia | 0.00324 kg·m² [A7] (PLANT:291) | "inertia 0.000082 kg·m²", side of the gear not stated [M] (RES:85); × 36² = 0.106 if rotor-side [E]. The V4 sibling's sheet: 0.3 kg·cm² [M] (RES:52); × 36² = 0.039 [E] | a 12 to 33-fold open range above today's value. If the figures are output-side, today's value is 40 times too high [E from M] |
| Gear play | none (`STAGE2_PLAN.md:306`) | ≤ 15 arcmin [M] (RES:75): an upper limit, not a typical value. [T, summary]: an older, larger sibling measured 0.334° against 0.083° advertised, 4.0 times. Ankle push rods: no source | all of it |
| Command path | delay k × 2 ms; torque follows at once | winding L/R = 0.45 ms [E from M, RES:85]; drive-side delay 0.10 to 1.50 ms [A] (LAT:241) | torque rise; no measurement |
| What the controller reads | 14-bit angle, 2 ms old [A] | two encoders, 17 and 18 bit; accuracy 0.4°, repeatability 0.3° [M] (RES:76, 85) | zero error; which encoder is reported and which closes the loop: no source |

### 1.2 Leg drive RMD-X6-P20-60-E: hip pitch (2 drives)

| Aspect | Modelled today [S] | Known about the real part | Missing |
|---|---|---|---|
| Gains | kp 1140, kv 30.78 [A] (PLANT:132) | same sibling caps, 500 and 5 [M via search extract] | with the caps the damping ratio is 0.25 [P] |
| Torque limit | ±42.0 N·m [A2] (PLANT:299) | peak 60 N·m; 40 N·m heats 29 °C in 10 s [M] (RES:132-133). The X6-50-L, which may be the part that ships: peak 45 N·m [M] (RES:167) | peak duration. With the X6-50-L the planning peak would be 31.5 N·m by A2's rule [E] |
| Torque against speed | 42 N·m up to 8.01 rad/s, zero at 9.22 rad/s: the 48 V speeds halved (PLANT:301-302; `catalog.py:77-82`) | rated 20 N·m at 153 rpm, no-load 176 rpm, both at **48 V** [M] (RES:131-135). No data at 24 V. Back-EMF 16 Vdc/krpm [M] (RES:154), which at 176 rpm is 55 V on a 48 V bus [E]: the two maker figures do not fit each other (R3 §1.2) | Two estimates, neither a catalogue value. (1) My line through the maker's two 48 V points, kept at its slope (8.30 N·m·s/rad) and shifted so that its zero is at the halved no-load speed: 42 N·m only below 4.16 rad/s, 10 N·m at 8.01 rad/s [E]. (2) From the back-EMF figure: no-load 8.01 rad/s at 24 V; with the copper slope (7.17 N·m·s/rad) the peak only below 2.15 rad/s [E from M; R, checked] |
| Bus voltage | as 1.1 | window 20 to 55 V, nominal 48 V [M] (RES:136-137) | as 1.1 |
| Friction | 0.15 N·m + 0.2 N·m·s/rad [A18] | backdrive 1.6 N·m; efficiency 72.7 % [M] (RES:154). The V4.1 sibling X6-50-L: 0.2 N·m [M] (RES:186). Which part ships is open (RES:154) | a ten-fold gap to the maker's figure |
| Reflected inertia | 0.004304 kg·m² [A7] (PLANT:303) | "0.66 kg·cm²", side not stated [M] (RES:154); × 19.612² = 0.0254 if rotor-side [E] | a 6-fold open range |
| Gear play | none | ≤ 15 arcmin; the store page says 10 [M] (RES:143, 154) | all of it |
| Command path | as 1.1 | L/R = 1.24 ms [E from M, RES:154] | as 1.1 |
| What the controller reads | as 1.1 | 17 and 17 bit [M] (RES:144) | as 1.1 |

### 1.3 Neck servo DYNAMIXEL XM540-W270-T (1 drive)

| Aspect | Modelled today [S] | Known about the real part | Missing |
|---|---|---|---|
| Servo law and gains | the same ideal servo with feedforward; kp 177.02, kv 4.16 [A] (PLANT:114) | a box servo with register gains and no torque input; control law not in the repository (R2) | the real law; its stiffness would be found by sheet A's measurement, not set (BENCH:523-525); nothing is measured yet |
| Torque limit | ±3.106 N·m = 0.293 × stall [A] (PLANT:263) | stall 10.6 N·m at 12 V; no continuous rating [M] (RES:1491-1492) | — |
| Torque against speed | peak to 1.885 rad/s (0.6 × no-load), zero at 3.142 [A] (PLANT:265-266) | no-load 30 rpm at 12 V [M] (RES:1494) | a straight line from stall to no-load gives the peak up to 2.22 rad/s [E]: today's line is inside it |
| Friction | 0.15 + 0.2 [A18] | none published. Another model of the maker [T, file]: MX-106 low-speed (Stribeck) term 0.122 N·m, constant term 1.0e-5 N·m, viscous 0.051, load term 0.20 | everything for this model |
| Reflected inertia | 0.000201 kg·m² [A, rule of A7] (PLANT:267) | not published. Other models [T]: 0.0058 (XH540-V150, summary), 0.026 (MX-106, file) | a 29 to 130-fold open range |
| Gear play | none | 15 arcmin [M] (RES:1502) | all of it; and which side of the play the position sensor is on: no source (section 2b) |
| Command path | 500 Hz targets, delay k | planned at 100 Hz on its own serial port [A] (LAT:335-336); delay not known (LAT:346-347) | the 10 ms hold; its own response |
| What the controller reads | 14 bit | 12-bit sensor [M, summary; R3] | coarser by 4 than modelled; not in the balance loop |

### 1.4 Tail servo Feetech STS3215 C018, 12 V (1 drive)

| Aspect | Modelled today [S] | Known about the real part | Missing |
|---|---|---|---|
| Servo law and gains | the same ideal servo; kp 49.13, kv 2.72 [A] (PLANT:120) | [T, file, on the 7.4 V version]: a proportional loop into motor voltage; default gain 32 × 0.166 per rad of duty. With the 12 V stall torque that is 15.6 N·m/rad, 0.32 of the model's [E] | the real law; damping from back-EMF only: stall / no-load = 0.62 N·m·s/rad, 0.23 of the model's kv today [E from M] |
| Torque limit | ±0.862 N·m [A] (PLANT:275) | stall 2.94 N·m; "rated" 0.98 N·m, no duty definition [M] (RES:1554-1556) | the continuous rating (A1; BENCH sheet E) |
| Torque against speed | peak to 2.827 rad/s, zero at 4.712 [A] (PLANT:277-278) | no-load 45 rpm [M] (RES:1558) | straight line: peak up to 3.33 rad/s [E]; today's line is inside it |
| Friction | 0.15 + 0.2 [A18]. They dominate this servo's load (DL:124) | [T, file, 7.4 V version]: dry 0.052 N·m, viscous 0.059 N·m·s/rad | about a third of today's values, if the 12 V part is alike |
| Reflected inertia | 0.000226 kg·m² [A] (PLANT:279) | [T, file, 7.4 V version]: 0.0284; another project's model: 0.027 | about 120 to 126 times today's value |
| Gear play | none | ≤ 0.5° [M] (RES:1566); [T, summary]: 0.6 to 0.9° unloaded, 1.3° loaded | all of it; and which side of the play the position sensor is on: no source (section 2b) |
| Command path | as 1.3 | as 1.3; target slew 5.2 rad/s in the 7.4 V firmware [T, file] | as 1.3 |
| What the controller reads | 14 bit | 12-bit magnetic encoder [M] (RES:1567) | as 1.3 |

---

## 2. The proposed model

One run-time layer on the unchanged Stage 1 plant. It writes model arrays and shifts the command. It adds no joint, no state to MuJoCo and no file to a hashed package (section 5).

Symbols, per drive, on the drive's own coordinate (the joint, or the motor's tendon at the ankle): L its angle, ω its speed, c the command that reaches it (after the delay), τ its motor torque from the last physics step, dt = 2 ms.

### 2a. Friction

| Item | Proposal |
|---|---|
| Equation | Moving: τ_fric = −b_v ω − F sign(ω). At rest: holds up to F. F = m_j (T_c + μ_L τ̄). τ̄ is \|τ\| of the last step through a 10 ms low-pass (my choice; it breaks the loop between friction and torque). |
| In MuJoCo | Per run: damping arrays = m_j b_v. Per step: friction-loss arrays = F. Joints use `dof_damping`, `dof_frictionloss`; the ankle motors use `tendon_damping`, `tendon_frictionloss`, and the ankle joints' own values go to 0. MuJoCo reads all of these at run time [P]. Friction rows stay 14 [P]. |
| Parameters | `fric_c` T_c, `fric_v` b_v, `fric_load` μ_L per class; `fric_joint` m_j per joint. |
| Why it is needed | Today every joint has the same small, load-free friction, and the controller's feedforward cancels it exactly. The maker's own figure for the hip pitch is ten times larger (BENCH:560-561). Friction that grows with load holds a loaded joint, resists every reversal in stance and adds to the motor torque that the duty reading counts. |
| Against which rating it is counted | The maker's peak, rated torque and torque constant are output-side values (RES:85, 154: "Kt side: output"), so they are already net of gear friction. The model puts friction outside the motor torque (section 5.2, trap 4). Two consequences, both on the harsh side and both kept. At the torque limit the net driving torque is the planning peak less the friction: at standstill 19.1 N·m (X4) and 38.3 N·m (X6) at `N`, and 17.25 and 35.4 N·m at `F+` [E; R, checked]. And the duty reading counts the torque that overcomes friction against a rating that was already net of it. The rules text must say both. |
| Is the load term worth its cost | Yes for the leg drives. Cost: six array operations per step. Evidence that it exists: [T] fits of geared servos need it (0.20 to 0.31, other models); the maker's efficiency leaves 3.2 N·m (X4) and 4.0 N·m (X6) unexplained at the rated point [E from M: efficiency, rated current, resistance]. No value for these drives has a source. |
| Is a stiction (Stribeck) term worth its cost | No, left out. No data for any of the four parts; the one third-party fit that has such a term is of another model (MX-106: 0.122 N·m, fading above about 1.7 rad/s; file). MuJoCo's friction constraint is soft at 2 ms (solref 0.02 s [P]) and could not show a sharp break-away. The range of T_c covers a higher static value. |
| Feedforward | Stays today's (the A18 values), on both the training and the judging side. The controller does not know the drawn friction (section 5, trap 1). |

### 2b. Gear play (backlash): a dead zone, no extra joint

| Item | Proposal |
|---|---|
| Representation chosen | A stateless dead zone in the drive's own loop. With u = kp (c − L) − kv ω (the torque today's servo asks for) and b = half the play: torque = u − kp b if u > kp b; u + kp b if u < −kp b; 0 in between. |
| In MuJoCo | The layer sends c − clip(u / kp, −b, +b) instead of c. MuJoCo's own servo then gives exactly the torque above: checked on this plant, joints and ankle tendons, to 1e-14 N·m [P]. The reported angle is the true one plus the same clip (the motor-side reading). |
| Where it applies | The 12 leg drives only. The neck and tail servos get no dead zone and no play shift of their reading until N6 says which side of the play their sensor is on (the encoder zero error of 2e still applies to all 14 drives). Reason: the dead zone is only right for a loop that closes on the motor side. The cloned small biped's code says of its servo, a small model of the neck servo's maker, that "the magnetic encoder sits on the OUTPUT side of that play, so the firmware position loop closes on main+backlash" [T, file: `/home/user/pollen-robotics/microduck_rl/src/mjlab_microduck/actuator/friction_dr_bam.py:72-73`, commit `0daab16`; another model]. No source was found for the sensor's side on the XM540 or the STS3215. With the first version's 1.5° on the tail the dead zone would also have been 0.64 N·m of that servo's 0.862 N·m peak [R, checked]. |
| What it assumes | The leg drive's loop closes on the motor-side encoder. This has no source: the record says only "dual encoder" (RES:76, 144), and which encoder closes the loop and which one the reply reports was not found (R3 §6). It is question 5 to the maker (section 8, choice 1; appendix A) and N6. The rotor crosses the gap within a step. |
| At the ankle | The play and the offset of 2e act on each motor's own coordinate: L_a = q_p + q_r, L_b = q_p − q_r (`drive.py:5-17`). The encoders the controller reads are joint sensors, one per hinge (`plant.py:372-374`, `707-708`). So the layer maps the two motors' reading errors e_a, e_b (clip plus offset) back to the joints: pitch reads q_p + (e_a + e_b) / 2 and roll reads q_r + (e_a − e_b) / 2. |
| Size | kp b at the maker's limit: 1.3 N·m on the X4 joints, 2.5 N·m on the X6, 0.65 N·m per ankle motor [P]. At 1° of play: 5.2, 9.9 and 2.6 N·m [P]. A joint whose servo asks for less torque than that gets none from it. |
| Cost in physics steps | None. No extra degree of freedom, no extra constraint row, five array operations per step. |
| Why it is needed | A loaded leg joint sits on one flank and shifts by the play when the load reverses: at touchdown and lift-off. At 0.5 m that is 2.2 mm per joint at the maker's limit and 8.7 mm at 1° [E]. The controller does not see it. |
| What it cannot show | The time the rotor takes to cross (for 1° on the X4: 2 to 11 ms with today's inertia, 13 to 61 ms with the maker's figure read rotor-side [E]). The knock when the gap closes. Rattle or a limit cycle. The link running free of the rotor's inertia inside the gap. A joint that is truly free inside the gap: MuJoCo's integrator is `implicitfast` (`designlab/mjcf.py:113`) and applies the servo's rate gain to the change of speed within a step even when the servo torque is zero. That acts as an added inertia of 2 ms × kv: 0.023 kg·m² at the knee beside the knee's own 0.055, so a load speeds the knee up 29 % less per step than a free joint [R: a one-joint toy model, not rerun; the formula h T / (M + h kv) checked by arithmetic]. Play in the ankle's push rods (no source; not included). Anything at all if the loop closes on the output encoder. |
| The alternative, not chosen now | A motor-side degree of freedom per drive with a limited gap (two open projects do this for small servos, R3 §2a). It needs a second plant builder outside the hashed files, or an edit that changes every key. nv goes from 20 to 34 [P, R1]. The settled stance, mid-task starts and the estimator's structure check change. With play 0 it is still another model, so no step-for-step identity with today. Its cost and stability at 2 ms are unmeasured. Section 8, choice 4. |

### 2c. Torque limit against speed and bus voltage; voltage sag

| Item | Proposal |
|---|---|
| Line | Driving side: limit = clip(s (ω0(V) − \|ω\|), 0, τ_peak). Braking side: τ_peak, as today (`drive.py:28-31`). s = τ_peak / (ω0(24 V) − `ts_corner`). On the X6, ω0(24 V) is itself drawn (`ts_zero`) if the maker has not answered question 3 when the layer is built: ruled by the lead on 16 Nov 2026 (section 8, choice 3). The slope then follows both draws, so the build needs a rule that keeps the corner below the zero (same choice). |
| Voltage | The zero-torque speed follows today's rule (`catalog.py:77-82`), every step instead of once: ω0(V) = ω0_catalogue × min(1, V / V_nominal). V_nominal is 24 V on the X4 and 48 V on the X6 [M]. The slope s stays, so the line shifts with the voltage. That is not today's rule for the corner: today the corner and the zero are both scaled by the same factor, so the two agree at 24.0 V only (`drive.py:22-35`). On the X4 at 21 V today's rule puts the corner at 7.61 rad/s and this model at 7.24 rad/s [E]; that example has `ts_corner` at today's 8.69, and with the nominal 5.76 this model's corner at 21 V is 4.31 rad/s [E, second revision]. A shift is what a fixed winding resistance gives; N2 would decide. Where the X6's `ts_zero` is drawn, it takes the place of the halved catalogue value: ω0(V) = `ts_zero` × V / 24. |
| Sag | V is the higher root of V = V0 − R_bus P / V: V = (V0 + √(V0² − 4 R_bus P)) / 2, never above V0. P = Σ over the 12 leg drives of [max(τ ω, 0) + c_cu τ²] / 0.92, plus the record's auxiliary input, from the last step's torques and speeds (one step of lag, on P only). If 4 R_bus P exceeds V0² there is no root: the bus has collapsed. That step is counted as a fault, V is set to V0 / 2 for it so that the arithmetic stays finite, and the run is recorded as having left the supply window. The collapse power is 735 W at `V-` and 2,464 W at `N`; ten X4 at their 21 N·m limit draw 888 W in copper loss alone [R, checked]. The first version iterated one step at a time, which oscillates when R_bus P / V² exceeds 1, and clipped at 12 V. |
| Constants | c_cu = 3 R_phase / Kt² = 0.185 W/(N·m)² on the X4 and 0.139 on the X6 [E from M, RES:71-72, 139-140]. 0.92 is the drive efficiency [A6] (`designlab/power.py:21-27`). The auxiliary input is read from the body's record: 20.3 W here [S] (REC:298), which is the Raspberry Pi 5's; the Jetson body's record has 44.8 W [S] (`round2/refine/jetson_75/candidates/75-M-H-10.json:299`). No credit for braking power, as A6. |
| Parameters | `ts_corner` per leg-drive class; `ts_zero` on the X6 (drawn only while question 3 is unanswered); `bus_v0` V0 and `bus_r` R_bus per run. |
| Box servos | Keep today's lines. They sit on a 12 V regulator, not on the bus. |
| A new reading | Steps with V below 20 V, the leg drives' lower supply limit [M] (RES:69, 137). A run with any such step is recorded as "left the supply window" beside its counts (section 4). Ruled by the lead on 16 Nov 2026: reported in every record, and a judged run is not failed on it yet (section 8, choice 10). The pack's end-of-charge 21.0 V [A11] is 1.0 V above the limit. At the nominal walk's summed peak (214 W: the record's summed drive peak of 194 W [S] (REC:301) plus its auxiliary input of 20.3 W [S] (REC:298); 8.9 A at 24 V [E]. MAT:205-206 prints 194 W and 8.9 A "with the auxiliaries" for the 75 cm leader it names; prescribed motion) that allows 0.09 Ω at constant power, or 0.11 Ω if the current stayed at 8.9 A [E]; with the Jetson record's 242 W, 0.08 to 0.10 Ω [E]. The top of `bus_r`'s range, 0.15 Ω, is above all of these: from 21.0 V it puts the bus at 19.3 to 19.7 V under that peak [R, checked]. |
| What the judge reads | The nominal corner speeds and peaks, unchanged, so the registered limits do not move (section 5, trap 3). |
| Why it is needed | Today the X6 is given 42 N·m up to 8.0 rad/s on half its nominal voltage. My line through the maker's two 48 V points, shifted to 24 V, gives about 10 N·m there; the catalogue has no 24 V data [E from M]. The study flags this itself (`LESSONS_LEARNED.md:113`; A2). A push recovery is where speed and torque peak together with the current. |
| Sense of scale [E] | With my shifted line at 22.2 V the X6 has about 22.5 N·m at the 5.8 rad/s its animal-pace reference asks for, against that reference's own peak of 16.6 N·m. On the back-EMF reading of section 1.2 it has 15.8 N·m there at 24 V and 11.5 N·m at 22.2 V, below that peak [R, checked]. At the brisk walk's 4.6 rad/s (0.35 m/s [S], `STAGE1_PLAN.md:85`; [P, second revision]: the stored stride is 0.311 m in 0.887 s; gate 1's 0.30 m/s lies between this stored walk and the nominal one at 0.24 m/s) the back-EMF reading leaves 20.4 N·m at 22.2 V against a peak of 14.2 N·m. The X4 has 13 N·m at the knee's 7.1 rad/s, against 7.2 N·m. Prescribed references (`stage1/runs/2026-11-08/75-M-H-10/refs/brisk.npz`, `fr05.npz`), not closed loop; peaks need not coincide. |

### 2d. The drive's own loop, and the delay in the parts the bench sheet plans to measure

The delay itself is unchanged: k whole physics steps on the whole command, through `speedfit.servo_delay` (`STAGE2_PLAN.md:351-360`; P40's delays). The model adds what sits around it. Nothing on the bench sheet has been measured (BENCH:11); the "bench reading" column names the reading the sheet plans.

| Part | What it is | Estimate today [A] (LAT:196-199 and its term table; baseline wiring) | Bench reading the sheet plans | In the model |
|---|---|---|---|---|
| C | the controller's compute step | 0.10 to 0.50 ms | none until firmware exists (BENCH:90-92) | inside k |
| H | tick to the end of the last command frame | 0.11 to 0.88 ms | H, every cycle (BENCH:85-87) | inside k |
| D | frame end to the torque change | 0.10 to 1.50 ms; 2.50 if the drive's loop runs at 500 Hz | D, in 0.2 ms bins (BENCH:88-89, 119-120) | its dead part inside k; its rise is `cmd_lag_ms` |
| S | age of the feedback at the tick | 1.23 to 2.62 ms; 0 to 0.62 ms beyond the model's 2 ms | S (BENCH:93-94) | `sense_steps` (2e) |
| T | tick to torque | 0.31 to 2.88 ms; 0.31 to 3.50 ms with S | T = H + D (BENCH:84) | k × 2 ms: T rounded up to whole steps. The sheet's own reading rule also adds S beyond 2 ms to T; the rule proposed below does not |

| Added part | Equation | Parameters | Why |
|---|---|---|---|
| Command lag | y ← y + (1 − exp(−dt / T_r)) (c − y); the drive acts on y | `cmd_lag_ms` T_r per class | the winding's rise (0.45 and 1.24 ms [E from M]) and the drive's outer loop. Nothing is measured. The budget's D already holds this rise (LAT:241): see "The judged delay" below |
| Box servos' update rate | their y changes only every 5th physics step; the phase is drawn | `cmd_hold` | the bus plan sends the neck and tail targets at 100 Hz [A] (LAT:335-336); the model updates them at 500 Hz |
| Box servos' gains | kp × `kp_scale`, kv × `kv_scale`, per run; the plant's own copies stay (table "Gain copies" below). The tail's kv that `kv_scale` multiplies is its nominal kv under the new inertia, 3.59 (today 2.72; 2f): ruled by the lead on 16 Nov 2026, one rule for every drive | two per box servo | their stiffness is a register value that sheet A would find (BENCH:523-525); the one third-party figure is a third of the model's (1.4) |
| Leg drives' gain limit | setting `gain_cap`: kp ≤ 500, kv ≤ 5, with the plant's own copies (`plant.py:575-576`) set too, so the feedforward torque is not scaled | none drawn | the sibling manual's ranges [M via search extract]. If they apply, hip and knee are at a damping ratio of 0.22 to 0.72 [P] |

**Gain copies: who knows a changed gain.** `Plant.command` folds the rate and torque targets into the command with the plant's own copies of the gains (`plant.py:575-576`, `662-681`). If only the model's kv changes, the servo gets an extra torque of (kv of the plant − kv of the model) × the commanded speed. The tracker's ankle cap holds a third copy of both gains, taken when the tracker is built (`control.py:1240-1248`). So each part that changes a gain must say which copies follow:

| Part | The plant's copy | The tracker's ankle-cap copy |
|---|---|---|
| `armature` (2f): the draw, `J+`, and the tail's nominal inertia in every setting but `today`; kv by the plant's rule | Follows: set to the run's kv and restored on exit. Without it the knee at `J+` (kv 11.51 → 19.47) would get 33 N·m at the nominal walk's 4.2 rad/s, against a 21 N·m peak [R, checked] | Amended by the lead's third list (17 Nov 2026) to say what the code does. Nominal in the env, the run's on the judging side. The layer sets the plant's copies when it is entered and the model's gains at the first physics step after it (the split, kept by that list), and a tracker takes its ankle-cap copy from the plant's copies when it is built (`control.py:1240-1248`). The env builds every tracker before the layer is entered (5.1), so its copy is nominal; it passes no cap anyway (`r5env.py:450`; `r5core.py:354`). A stand-in row builds its tracker inside the episode (`panels.py:844`), after the entry, so a served R2 row whose tracker is built there takes the run's kv in its ankle cap (`panels.py:808`). Of the named settings only `J+` and `W1` move the ankle motors' kv copy: 11.28 against 2.08 (S1's check; printed again from `servo.Layer` on 17 Nov). Both are reported only, and no judged setting moves an ankle motor's gain. The R0 rows pass no cap (`panels.py:880-881`); the policy rows are not built yet. The first wording, "cannot follow … nominal on both sides", was not what the code does. Accepted by the lead (section 8, choice 21, and its note) |
| `gain_cap` and `G5` | Follows | No difference: the ankle motors' gains (299.25 and 2.08) are below both caps |
| `kp_scale`, `kv_scale` (neck, tail) | Does not follow, on purpose. A box servo takes a position target, and the controller shifts that target with its own, nominal gains; the servo's real stiffness then scales the feedforward, as the env's stiffness factor does today (`r5env.py:504-506`) | Not concerned |

Rule for the build, for `armature` and `gain_cap` only: no command is served by a kv other than the one it was folded with. As built (amended by the lead's third list of 17 Nov 2026), the plant's copy changes when the layer is entered and the model's gain at the first physics step after it, before any command is served; the first wording had both "in the same call". That includes the commands a delay queue already holds at a mid-task start (`r5env.py:672-684`): they are folded with the plant's copies when the queue is filled (`r5env.py:679-680`, through `r5env.py:467-472`), so the copies must be set before that (5.1), and the gains are set at the first physics step, which serves the queue's first command. `kp_scale` and `kv_scale` are the stated exception of the row above: there the servo's gain differs from the one the command was folded with, on purpose. Test T14 (section 5.5).

**The judged delay, and where a measured value would go.**
- Under `N` a judged "10 ms" is 10 ms of pure delay plus, on the leg drives, the command lag (time constant 0.5 ms on the X4, 1.2 ms on the X6) and, on the neck and tail, a held target that is up to 8 ms old. P40 approved gate 1's delays (DL:55), and a delay in the study's model is a pure delay. Gate 1 under this model is therefore a little harsher than the approved numbers say. Ruled by the lead on 16 Nov 2026: the command lag and the 100 Hz hold stay in the nominal model, and the note says so (sections 9 and 11; section 8, choice 19).
- In the drawn panel `D` the delay-like draws come on top of the record's delay: `sense_steps` adds 0, 2 or 4 ms of feedback age, and `cmd_lag_ms` a lag of up to 4 ms on the leg drives and up to 20 ms on the neck and tail. Ruled by the lead on 16 Nov 2026: kept, and said (section 4; section 8, choice 25).
- The budget's drive-side estimate D (0.10 to 1.50 ms) already contains the winding's rise (LAT:241). Against the budget the rise is thus counted twice, once inside k and once as the lag. The error is on the harsh side and at most the lag's time constant.
- The bench sheet's reading adds the feedback age beyond 2 ms to T (BENCH:93-96). This model carries that age as `sense_steps`. When a bench value goes in, only one of the two may carry it.
- **Rule for a measured value** (gate 2; nothing is measured today). Ruled by the lead on 16 Nov 2026 for the feedback age: a measured feedback age goes to `sense_steps`. The rest is this note's proposal, which no ruling names. H and the dead part of D, up to the first change of torque, which is what the sheet's D is (BENCH:81-83, 88-89), go to k: T = H + D, rounded up to whole 2 ms steps. The rise after the first change (new measurement N4) goes to `cmd_lag_ms`. The feedback age beyond 2 ms goes to `sense_steps`, rounded up to whole steps, and is then not added to T. For a run without the servo layer the sheet's own rule stands. Section 8, choice 19.

Not modelled in the loop: its sampling of the feedback. MuJoCo evaluates the servo on the true state every 2 ms, with its damping treated implicitly. A lag on that path needs the servo law in code and gives up the implicit damping (R1 §3.3). The drive's loop is quoted at 500 Hz to 1 kHz for an earlier generation [M via search extract, LAT:443], which is no slower than the physics step.

### 2e. Sensor side

The readers found two things that matter, and one that does not.

| Part | Equation | Parameters | Why, or why left out |
|---|---|---|---|
| Encoder zero error | The drive holds L + o = target, so the layer sends c − o. The reported angle is L + o. | `enc_offset` o per joint | Every sim-to-real project R3 read draws it (±0.01 to ±0.05 rad). The maker's accuracy is 0.4° = 0.007 rad on the X4 [M]. It moves the foot and the controller cannot see it. |
| Play in the reading | on the leg drives the reported angle also carries the clip of 2b; at the ankle it is mapped to the joints as 2b says | none new | The sign-dependent error of a motor-side encoder. Not applied to the neck and tail (2b). |
| Feedback age | joint angles and rates are those of `sense_steps` steps earlier; the box servos' refresh every 5th step | `sense_steps` 0 to 2 | The budget's S is up to 0.62 ms beyond the model on the baseline wiring and up to 4.17 ms on a Jetson USB path [A] (LAT:196-199). Where a measured S would go: the rule in 2d. |
| Encoder quantisation | left as today (14 bit) | — | The leg drives' 17 and 18 bit [M] are finer than modelled. The box servos' 12 bit is coarser, but they are not in the balance loop. |
| IMU delay | left out | — | Read by the same controller at 1 kHz or faster in every wiring [A] (LAT:250-251). |

One artefact: the reset's own first reading is exact; the offset appears with the first physics step (section 5).

### 2f. Reflected rotor inertia (added; not in the draft's list)

| Item | Proposal |
|---|---|
| Equation | Per run: `dof_armature` (joints) and `tendon_armature` (ankle motors) = the drawn J, then MuJoCo's constants recomputed (`mj_setConst`, as the env's mass change does). Both arrays are read at run time [P]. kv = 2 √(kp (I_home + J − J_today)): the plant's own rule (`plant.py:864`) on the drawn inertia. The plant's own copy of kv follows (2d, "Gain copies"). The same rule gives the tail's nominal kv under its new nominal inertia: 3.59 N·m·s/rad with the table's 0.028 kg·m² (3.60 with the fit's unrounded 0.0284), against today's 2.72 [E, second revision: PLANT's kp 49.13 and inertia 0.0377]. Ruled by the lead on 16 Nov 2026: recomputed by the plant's rule, one rule for every drive (section 8, choices 7 and 23). |
| Parameter | `armature` per class. |
| Why it is needed | Today's value is a rule of thumb (A7). The maker's own figure, read on the rotor side, is 33 times larger on the X4 and 6 times on the X6 [E from M]; the V4 sibling's figure, read the same way, is 12 times larger on the X4 [E from M, RES:52]. Third-party fits of the tail servo's 7.4 V version give about 120 to 126 times today's value (0.027 and 0.0284 against 0.000226 kg·m²) [T, file; the ratio E, second revision]. |
| The tail, on the stored references [P, rerun on 16 Nov] | With the 7.4 V fit's 0.0284 kg·m² the tail's RMS torque rises from 0.20 to 0.25 N·m in the nominal walk, from 0.26 to 0.36 in the brisk walk and from 0.30 to 0.41 at the animal pace, against its 0.491 N·m planning limit. Same arithmetic as the next row; prescribed motion, no closed loop. The same fit supplies the tail's friction nominals, so its inertia is taken as the tail's nominal too (section 3). Ruled by the lead on 16 Nov 2026: condition 4 is read with the tail's nominal inertia, and that nominal stays at the top of its range (section 8, choices 16 and 24). |
| What it would mean, on the stored references [E] | With the high value the knee's RMS torque rises from 0.52 to 0.79 of its planning limit in the nominal walk, from 0.53 to 1.08 in the brisk walk and from 0.56 to 1.41 at the animal pace. Its peak there is 22.2 N·m against the 21.0 N·m limit. My arithmetic: (J_high − J_today) × the reference's own acceleration, added to the reference's own torque (`refs/nominal.npz`, `brisk.npz`, `fr05.npz`). Prescribed motion, no closed loop. |
| What stays nominal | The references' torques and the controller's feedforward, as with the mass (`designlab/stage1/env.py:126-133`). |

### 2g. Left out, and why

| Left out | Why |
|---|---|
| A torque-scale (motor strength) draw | Not drawn, and A2's margin is used up, not spare. A2's 0.70 of the maker's peak is the study's one allowance on a peak that the maker defines at 10 rpm (RES:64-65; DL:108). The one third-party figure found, on an older and larger sibling, has the torque constant at 0.67 of the quoted value (1.393 against 2.09 N·m/A; summary): that alone takes the whole 0.70. Gear friction then comes off the planning peak as well (2a). The env's stiffness draw × 0.6 to 1.2 (`r5env.py:226`) scales the position and feedforward terms, not the limit. The envelope can take a scale (`torque_scale`, `drive.py:195-196`). Ruled by the lead on 16 Nov 2026: no torque-scale draw now, and the note says that A2's 0.70 is used up (section 8, choice 20). N2 would measure it. |
| Heat | No thermal model. The judge keeps its RMS limit [A1]. Section 7. |
| A learned actuator network | Needs torque data from the real drive (R3 §2c). |
| A voltage-loop law for the box servos | Their firmware scaling is known for one 7.4 V unit only. A gain scale and the update rate stand in. |
| The 12 V regulator's droop | One 4.5 A regulator against 7.1 A of summed stall current (BENCH:745-746). Outside the legs. |
| Dropped or late bus frames | The sheet's soak test is planned to count them (BENCH:342-345). Nothing is measured; no figure to draw from. |
| Gear play on the neck and tail servos | No source says which side of the play their sensor is on (2b). Their maker and third-party play figures stay in section 3 for the day N6 answers. |

---

## 3. The parameter table

"Today" is what the simulation uses now [S]. "Nominal" is the value gate 1's condition 1 would be judged at. Every draw is once per run. The "Source and kind" column says what each nominal is. It is the maker's own figure for the quantity in one row only (the pack's nominal voltage). In four rows it is a maker's figure for a related quantity, taken as the value by this note: the two leg drives' backdrive torque as their dry friction, and their backlash limit as their play. In six rows it comes from a third party's fit of another model or another voltage version. In the other 28 rows it is an assumption or estimate of this study, or today's value kept; four of those are my arithmetic on maker figures (the two torque-speed corners and the two command lags). The table has 39 rows: the first revision's 38 (1 + 4 + 6 + 27; the fixer's count, repeated by the second check and by the finisher) and one added in the second revision, the X6's zero-torque speed `ts_zero`, whose nominal is today's value kept. Where the column says "no source", the nominal is a placeholder, labelled **unsourced**. No nominal is measured on this build's parts. Five cells named no kind until the lead's third list of 17 Nov 2026 (option (c) there): the neck's `armature`, the tail's `kp_scale`, the neck's and the tail's `kv_scale`, and `latency_steps`. Their kind is the study's assumption [A], which is what the code (`servo.py`) already writes; the counts above do not change, and the code's table and its sha256 (`servo.TABLE_SHA256`) do not move.

No bench measurement replaces any value today. Nothing on the bench sheet has been measured (BENCH:11), and the owner does not expect to run the bench before a build works in simulation (P32, DL:47). The last column says what would replace each value: a step that is on the sheet (with its lines), or "proposed" with the id of an addition N1 to N6. The additions are listed under the table; none is on the sheet.

Classes: X4 = `myactuator_rmd_x4_p36_36_e_l` (10 drives), X6 = `myactuator_rmd_x6_p20_60_e` (2), neck = `robotis_xm540_w270_t`, tail = `feetech_sts3215_c018`.

| Parameter | Class | Today | Nominal | Range per run | Distribution | Source and kind | What would replace it: a step on the bench sheet, or a proposed addition |
|---|---|---|---|---|---|---|---|
| `fric_c` dry friction, N·m | X4 | 0.15 [A18] | 0.2 | 0.1 to 0.6 | uniform | nominal [A, this note]: the maker's backdrive torque, 0.2 N·m [M] RES:85. That is another quantity (the torque that turns the unpowered output); it is taken as the dry term. Range [E]; the V4 sibling's 1.14 [M] RES:52 is outside it | sheet B step 1 (BENCH:564-567) |
| | X6 | 0.15 | 1.6 | 0.2 to 2.4 | uniform | nominal [A, this note]: the maker's backdrive torque, 1.6 N·m [M] RES:154, taken as on the X4. Low end: the X6-50-L's backdrive 0.2 [M] RES:186. High end [E] | sheet B step 1 |
| | neck | 0.15 | 0.12 | 0.03 to 0.30 | uniform | no source for this model. Nominal [T, file, another model]: the MX-106 fit's low-speed (Stribeck) term, 0.122; its constant term is 1.0e-5. Range [E] | sheet B steps 1 and 3 |
| | tail | 0.15 | 0.05 | 0.03 to 0.15 | uniform | the 7.4 V version's fit, 0.052 [T, file]. Top = A18. No source for the 12 V part | sheet B steps 1 and 3 |
| `fric_v` viscous, N·m·s/rad | X4 | 0.2 [A18] | 0.10 | 0 to 0.25 | uniform | no source. Nominal [E], set so that the total at the maker's rated point equals today's (1.9 N·m) | sheet B step 2 (BENCH:568-572); it stops at 4 rad/s, the model uses 8.7 |
| | X6 | 0.2 | 0.05 | 0 to 0.10 | uniform | no source. Nominal [E], same rule (3.4 against today's 3.35 N·m) | sheet B step 2 |
| | neck | 0.2 | 0.05 | 0.01 to 0.20 | uniform | another model's fit, 0.051 [T, file]. Top = A18 | sheet B step 3 |
| | tail | 0.2 | 0.06 | 0.03 to 0.20 | uniform | the 7.4 V version's fit, 0.059 [T, file]. Top = A18 | sheet B step 3 |
| `fric_load` share of motor torque | X4 | 0 | 0.08 | 0 to 0.15 | uniform | no source. [E]. Upper bound 0.30 [E from M] | proposed N5 |
| | X6 | 0 | 0.05 | 0 to 0.10 | uniform | no source. [E]. Upper bound 0.20 [E from M] | proposed N5 |
| | neck | 0 | 0.20 | 0 to 0.30 | uniform | other models' fits: 0.198, 0.311 [T, file] | proposed N5 |
| | tail | 0 | 0 | 0 to 0.25 | uniform | no source (the third-party fit has no load term) | proposed N5 |
| `fric_joint` multiplier per joint | all | 1 (the env draws one factor 0.5 to 1.5 for all hinges, `r5env.py:229-230`) | 1 | 0.8 to 1.25 | uniform, per joint | no source. [E]: left and right differ | sheet B on each unit; step 4 on the leg |
| `play_deg` total play, degrees | X4 | 0 | 0.25 | 0 to 1.0 | uniform, per joint | nominal [A, this note]: the maker's upper limit, 15 arcmin [M] RES:75, taken as the value. Top [E from T, summary]: 4.0 × that limit, the ratio the one third-party measurement found on an older, larger sibling (0.334° against 0.083° advertised). It is also the buy gate draft's 1°. Ankle push rods: no source, not included | proposed N1 |
| | X6 | 0 | 0.25 | 0 to 1.0 | uniform, per joint | as the X4; the limit is [M] RES:143 (the store page says 10 arcmin, RES:154) | proposed N1 |
| | neck | 0 | 0, not modelled | fixed | — | left out until it is known which side of the play the sensor is on (2b). For that day: 15 arcmin [M] RES:1502; [T, summary]: 0.11 to 0.29° on a family member | proposed N6, then N1 |
| | tail | 0 | 0, not modelled | fixed | — | left out, as the neck. For that day: ≤ 0.5° [M] RES:1566; [T, summary]: 0.6 to 0.9° unloaded, 1.3° loaded | proposed N6, then N1 |
| `ts_corner` speed to which the peak is available at 24 V, rad/s | X4 | 8.69 [A] | 5.76 | 5.76 to 8.69 | uniform | nominal [E from M]: my line through the maker's rated and no-load points (RES:63-67). Top: today's rule | proposed N2 |
| | X6 | 8.01 [A] | 4.16 | 2.15 to 8.01 | uniform | nominal [E from M]: my line through the maker's two 48 V points (RES:131-135), kept at its slope and shifted to the halved no-load speed. Low end [E from M; R, checked]: the corner on the back-EMF reading, 24 V over 16 Vdc/krpm (RES:154) with the copper slope Kt² / 3R; the first version had 3.36. No 24 V data exist and the maker's figures do not fit each other (1.2): open, question 3 to the maker. With the zero-torque speed fixed at 9.215 rad/s this one parameter does not reach the back-EMF line itself: at 5.8 rad/s and 24 V it gives 20.3 N·m against that line's 15.8 [E]; hence the row `ts_zero` below (section 8, choice 3) | proposed N2 |
| | neck, tail | 1.885, 2.827 [A] | same | fixed | — | today's lines are inside the maker's stall-to-no-load lines [E from M] | none planned |
| `ts_zero` zero-torque speed at 24 V, rad/s | X6 | 9.215 [A] | 9.215 | 8.01 to 9.215, drawn only if the maker has not answered question 3 when the layer is built; else fixed | uniform, kept above the drawn corner by a rule the build must state (section 8, choice 3) | ruled by the lead on 16 Nov 2026: also drawn if the maker has not answered. Nominal and top [A]: today's rule, the maker's 48 V no-load speed of 176 rpm halved (RES:135; `catalog.py:77-82`). Low end [E from M; second revision]: the no-load speed the back-EMF figure gives at 24 V, 8.01 rad/s (RES:154). With the corner at 2.15 and the zero at 8.01 the line is the back-EMF line itself: 15.8 N·m at 5.8 rad/s [E] | the maker's answer (question 3); proposed N2 |
| `armature` reflected inertia, kg·m² | X4 | 0.00324 [A7] | 0.00324 | 0.00324 to 0.106 | log-uniform | nominal and low end [A7]. High [E from M]: 0.000082 × 36² (RES:85), if rotor-side. Between them: the V4 sibling's 0.3 kg·cm² × 36² = 0.039 [E from M, RES:52]. The nominal is at the bottom of a 33-fold range (section 8, choice 1) | the maker's answer (question 1); proposed N5 |
| | X6 | 0.0043 [A7] | 0.0043 | 0.0043 to 0.0254 | log-uniform | nominal and low end [A7]. High [E from M]: 0.66 kg·cm² × 19.612² (RES:154) | the maker's answer (question 1); proposed N5 |
| | neck | 0.000201 [A] | same | 0.0002 to 0.026 | log-uniform | no source for this model. Nominal: the study's assumption [A], the study's rule (A7). High: other models [T]: 0.0058 (summary), 0.026 (file). The nominal keeps the study's rule although the friction nominals come from another model's fit: flagged (section 8, choice 16) | proposed N5 |
| | tail | 0.000226 [A] | 0.028 | 0.0002 to 0.028 | log-uniform | nominal [T, file, the 7.4 V version]: the fit's 0.0284, the same fit that gives this servo's friction nominals; another project's model of that version has 0.027 [T, file]. Low end: today's rule. No source for the 12 V part. The nominal is the top of the range (ruled by the lead on 16 Nov 2026: it stays there; nothing sources a higher value). So a training draw lies at or below the judged value (the range's median is 0.0024 [E, second revision]), and `J+` equals `N` for the tail. With this nominal the tail's nominal kv is 3.59, not today's 2.72 (2f) | proposed N5 |
| `cmd_lag_ms` | X4 | 0 | 0.5 | 0 to 4 | uniform | nominal [E from M]: L/R 0.45 ms. Top [E]. No source for the response. The budget's D already holds this rise (2d) | proposed N4 |
| | X6 | 0 | 1.2 | 0 to 4 | uniform | nominal [E from M]: L/R 1.24 ms | proposed N4 |
| | neck, tail | 0 | 0 | 0 to 20 | uniform | no source (LAT:346-347) | BENCH:347-370 |
| `cmd_hold` box servos' update | neck, tail | every step | every 5th step, phase 0 | phase 0 to 4 | integer uniform | the bus plan [A] LAT:335-336 | BENCH:347-370 |
| `kp_scale` | neck | 1 | 1 | 0.3 to 1.2 | uniform | no source (register gains) | sheet A steps 1 to 3 (BENCH:519-535) |
| | tail | 1 | 1 | 0.3 to 1.2 | uniform | nominal: the study's assumption [A], the model's stiffness kept until sheet A (choice 17). Low end [E from T (file) and M]: 15.6 N·m/rad at the default register | sheet A |
| `kv_scale` | neck | 1 | 1 | 0.5 to 1.2 | uniform | nominal: the study's assumption [A], the model's damping kept. [E from M]: stall / no-load = 3.4 N·m·s/rad, 0.81 of the model's | sheet A step 4 |
| | tail | 1 | 1, of the nominal kv under the new inertia (3.59; ruled by the lead on 16 Nov 2026) | 0.15 to 1.2 | uniform | nominal: the study's assumption [A], 1 of the nominal kv under the new inertia. [E from M]: stall / no-load = 0.62 N·m·s/rad; 0.23 of today's kv (2.72) and 0.17 of the nominal kv under the new inertia [E, second revision]. The first revision's low end of 0.2 no longer reached it (0.2 × 3.59 = 0.72); 0.15 × 3.59 = 0.54 does, and is about the old low end's damping (0.2 × 2.72 = 0.54). The top is 4.3 N·m·s/rad, where it was 3.3 [E] | sheet A step 4 |
| `gain_cap` | X4, X6 | none | none | a setting, not drawn: none, or kp ≤ 500 and kv ≤ 5 | — | nominal "none" [A, this note]: the caps are from a sibling's manual [M via search extract] LAT:442 and their units are not confirmed, so they are not taken as this part's until the maker's document is read (question 2) | sheet A step 1 (BENCH:519-525); the maker's protocol document |
| `enc_offset` rad | all | 0 | 0 | −0.015 to +0.015 | uniform, per joint | nominal 0 [A, this note]: an offset has no preferred sign. Range: a third-party default [T, code: R3 §2b]; the maker's accuracy on the X4 is 0.4° = 0.007 rad [M] RES:85, inside it | proposed N6 |
| `sense_steps` | all | 0 | 0 | 0 to 2 | integer uniform | the budget's S [A] LAT:196-199 | bench S (BENCH:93-94) |
| `bus_v0` V | bus | 24.0 fixed | 22.2 | 21.0 to 25.2 | uniform | nominal: the pack's nominal voltage [M] `designlab/data/parts.json:207`. Range [A11] | not on the sheet (the pack is an optional part there, BENCH:186): a log of the pack's voltage over a run, with proposed N3 |
| `bus_r` Ω | bus | 0 | 0.05 | 0.02 to 0.15 | uniform | no source (the pack's entry has none, `parts.json:193-216`). [E]. For scale: 5 m of tether cable is 0.13 Ω [E from MAT:233]. The top of the range takes a 21.0 V bus below the drives' 20 V limit at the walk's summed peak (2c) | proposed N3 |
| `latency_steps` | all | training 0 to 12; judged 0 to 5 | unchanged | unchanged | integer uniform | the study's assumption [A], not the layer's: the env's training draw and the record's delay (`r5env.py:227`; `bar.py:134-136`) | bench T (BENCH:84) |

The friction sets, as (T_c, b_v, μ_L):

| Class | Low | Nominal | High | Friction torque at the maker's rated point: low / nominal / high; today; my bound [E] |
|---|---|---|---|---|
| X4 | 0.1, 0, 0 | 0.2, 0.10, 0.08 | 0.6, 0.25, 0.15 | 0.10 / 1.91 / 4.35; 1.89; 3.16 N·m |
| X6 | 0.2, 0, 0 | 1.6, 0.05, 0.05 | 2.4, 0.10, 0.10 | 0.20 / 3.40 / 6.00; 3.35; 4.05 N·m |
| neck | 0.03, 0.01, 0 | 0.12, 0.05, 0.20 | 0.30, 0.20, 0.30 | no rated point |
| tail | 0.03, 0.03, 0 | 0.05, 0.06, 0 | 0.15, 0.20, 0.25 | no thermal rated point |

The high sets of the leg drives are above my bound on purpose: the bound assumes the maker's efficiency was taken at the rated point, which the record does not say. `F+` is labelled "above the estimate's bound" wherever it is read.

**Proposed additions to the bench sheet** (`stage2/jetson75/BENCH_SHEET.md`). None is on the sheet today. Today the sheet plans the delay (sections 1 to 6), stiffness (sheet A), friction and damping (sheet B), the ToF stop (C), tread grip (D) and the tail servo's heat (E). The ids are this note's. Ruled by the lead on 16 Nov 2026: the six are proposed additions to the sheet, and the gate draft's sentences on what the bench measures are reworded with them once the owner has seen them (section 8, choice 15).

| Id | Measurement proposed | Pins |
|---|---|---|
| N1 | Lost motion per drive type: a lever and a dial indicator, the load reversed, read at the lever and on both encoders; loaded and unloaded. On the leg later, with the push rods. | `play_deg`; the push rods' play |
| N2 | Peak torque held at 2, 4, 6 and 8 rad/s against a brake or a second drive, at 21, 24 and 25.2 V. A2's dyno curves, reduced. | `ts_corner`; the X6's line at 24 V; whether the line shifts or scales with voltage; a torque scale |
| N3 | Bus voltage at the drives before and during a known load step, at two currents; a log of the pack's voltage over a run. | `bus_r`, `bus_v0` |
| N4 | Torque current after a feedforward-torque step, in 0.2 ms bins; a small sine sweep from 1 to 50 Hz. | `cmd_lag_ms`; whether a torque lag must be modelled at all |
| N5 | A free-shaft chirp read by the encoders (inertia, viscous, dry friction); a pendulum with two or three masses (the load term). The two methods R3 §2a and §2c describe. | `armature`, `fric_load` |
| N6 | Zero repeatability against a jig. For each drive type, which side of the gear play the reported position is on and which one the loop closes on: from the maker's document or answer for the leg drives (question 5), or, if neither says, by N1's load reversal with the drive holding a position, read at the lever and on both encoders; for the neck and tail by a load reversal read at the lever and in the servo's own position. | `enc_offset`; the assumption of 2b; whether the neck and tail get a dead zone |

---

## 4. How it is drawn

| Question | Proposal |
|---|---|
| When | Once per run (episode), at reset. |
| Per class | `fric_c`, `fric_v`, `fric_load`, `ts_corner`, `ts_zero` (X6 only, while question 3 is unanswered), `armature`, `cmd_lag_ms`, `kp_scale`, `kv_scale`. |
| Per joint | `fric_joint`, `play_deg`, `enc_offset`. Left and right legs differ. |
| Per run, global | `bus_v0`, `bus_r`, `sense_steps`, the box servos' hold phase, and the existing `latency_steps`. |
| Fixed within a run | Every parameter. Only the layer's states move: the bus voltage, the load filter, the lagged command, the reading buffer. |
| Seed, training | Its own generator: `numpy.random.SeedSequence([episode seed, 2])`. The env already uses `[episode, 0]` for the start and `[episode, 1]` for the sensors (`designlab/stage1/env.py:119-123`; `r5env.py:648-649`). The env's own generator is not touched, so the task, the factors and the push of an episode are the same with the option on or off. With `options["seeds"]` and no `episode_seed` the env's episode seed is None (`r5env.py:637-638`): no draw is possible then, the layer refuses it, and a named setting must be given. |
| Draw order | Fixed and written as a rule text. The table's sha256 goes into the env's description and the row's `describe()`. |
| Named settings on a judged panel | Draw nothing. Every parameter is at a named value; `fric_joint` is 1 and `enc_offset` is 0. A record is then repeatable from the bar's own 20 seeds per block (`STAGE2_75_PLAN.md:621`), as today. |
| The drawn panel `D` | For the run with bar seed s (the block's 20 seeds: 3042 to 3061 on the headline block, `bar.py:87-88`) the servo generator is `numpy.random.SeedSequence([s, 3])`, in the rule text's draw order. The tag 3 is the panel's own. With the training env's tag 2 a bar seed would give the parameter set of the training episode whose seed has the same number, and an episode seed is 1,000,000 × the reset's seed + k (`designlab/stage1/env.py:117-123`), so a stream reset with seed 0 reaches the bar's numbers (the second check's finding 11). The panels' own seeds for the start, the sensor noise and the push (`panels.py:505-508`) are not touched.<br>**Drawn:** every drawn parameter of section 3, except three. `latency_steps` is not drawn: the delay is the record's, 4 or 10 ms. `armature` is at nominal: on the leg drives until the maker answers question 1 of section 8, on the neck and tail until N5. `bus_r` is at nominal: it has no source, and its top leaves the supply window (2c). The trunk mass stays 1: it is not a servo parameter, and `M-`, `M+` cover it.<br>**Delay-like draws** (ruled by the lead on 16 Nov 2026: kept, and said). On top of the record's delay a `D` run has 0, 2 or 4 ms of feedback age (`sense_steps`) and a command lag of up to 4 ms on the leg drives and up to 20 ms on the neck and tail (`cmd_lag_ms`).<br>**Sets.** Same seed and same table hash give the same 20 parameter sets, on both sides. The same 20 sets serve both delays and both policies, so the four `D` records of a block test 20 sets, not 80.<br>**Second block** (ruled by the lead on 16 Nov 2026). A `D` record that passes is also run on the bar's second block, seeds 5042 to 5061 (`bar.py:87-88`). By the same rule those seeds give 20 other parameter sets. What a second-block record that fails does to the verdict is not in the ruling: section 8, choice 27. |
| The env's old draws | With the option on, `joint_damping` and `joint_friction` must be (1, 1): the layer owns friction. The calls stay, so the stream does not shift. Stiffness × 0.6 to 1.2, mass, floor friction and noise stay as they are. |

**Named settings.**

| Name | Values | Use |
|---|---|---|
| `today` | every part neutral: A18's friction on the joints, play 0, today's line, 24.0 V and 0 Ω, no lag, no offset, A7's inertia | the identity test (section 5); a link to today's records |
| `N` nominal | the Nominal column of section 3 | condition 1 |
| `F-`, `F+` | every class at its low, or high, friction set. On the leg drives `F+` is above my estimate of the bound (section 3) and is labelled so | condition 2, judged |
| `P0`, `P+` | play 0 on every leg drive; or 1.0° on every leg drive. No play on the neck and tail in either (2b) | condition 2, judged |
| `M-`, `M+` | trunk mass and inertia × 0.9, × 1.1 (the env's own factor, `r5env.py:225`), applied before the reset on both sides (5.1), not by the layer | condition 2, judged |
| `D` | the drawn panel defined above | condition 2, judged by the bar's whole rule on the headline block; a record that passes is run again on the second block |
| `V-` | `bus_v0` 21.0 V with `bus_r` 0.15 Ω | reported. By this note's own arithmetic the pair is outside the drives' supply window at the walk's summed peak (19.3 to 19.7 V, 2c), and `bus_r` has no source. A `V-` run with a step below 20 V is recorded as "left the supply window", not as stayed up |
| `J+` | `armature` at the top of each class's range, kv by the plant's rule. For the tail the top is the nominal, so `J+` equals `N` there | reported until the maker answers which side of the gear its figure is on (section 8, choice 1) |
| `G5` | `gain_cap` on | reported until the maker's document is read |
| `T+` | `ts_corner` at today's values (and the X6's `ts_zero` at today's 9.215) | reported: the easy end |
| `X6L` | hip pitch as if the X6-50-L ships: force range ±31.5 N·m (0.70 × the maker's 45 N·m, RES:167) and dry friction 0.2 N·m (RES:186); the rest as `N`; the judge's registered limits unchanged | reported until the maker says which part ships |
| `W1`, `W2` | `F+ V- J+ M+` together; `F- P+ M-` together | reported |

Each corner moves one factor and leaves the rest at `N`.

**The draft's corners, confirmed or corrected.**

| Draft (`BUY_GATE_DRAFT.md:41-44`) | Against the table | Proposed |
|---|---|---|
| friction at half and at one and a half times nominal | The open range is not ±50 % of one number. On the X4 the maker's two generations differ 5.7-fold; on the X6 the two candidate parts differ 8-fold; the load term has no source at all. | `F-` and `F+`: each class's low and high set |
| gear play at 0 and at 1° | 1° is four times the leg drives' maker limit. That is the ratio the one third-party measurement found on a sibling (summary). The first version lowered it to 0.5° with no source; the review put it back (B2). The neck and tail get no play (2b). | confirmed for the leg drives: `P0`; `P+` at 1.0° |
| trunk mass at 0.9 and 1.1 | Not a servo parameter; it is the env's range. | confirmed: `M-`, `M+` |
| — | The bus and the rotor inertia are not in the draft. Each rests on a number nobody has: the bus resistance, and which side of the gear the maker's inertia figure is on. | `V-` and `J+` are run and reported, not judged |
| — | The draft judges corners only. The inside of the ranges is never judged, and this body's response to one factor is known not to be monotone: with delay it falls at 10 ms and stays up at 20 ms (A17, DL:123). | add the drawn panel `D` |
| 2 × 2 × 2 corners at two delays: 16 records | | 6 one-factor corners and `D` at two delays: 14 records per policy, 28 for the two; and up to 4 second-block records of `D` |

**Condition 2 as proposed: what is run, and what "passes" means.** Still a proposal: condition 2 has not been put to the owner. Which settings are judged is the owner's (section 8, choice 9; the question is in section 11). The pass rule at a corner and the second block are ruled by the lead (choices 22 and 27).

| Item | Proposal |
|---|---|
| Walk, floor, delays | The 0.30 m/s walk of the chosen look, on the grippy floor (μ 0.9), at 4 ms and at 10 ms of delay (k = 2 and k = 5). |
| Policies | Both training seeds' policies. A condition counts only if both pass it (gate 1's condition 5, P40). |
| Runs per record | 20: the bar's headline block, seeds 3042 to 3061 (`bar.py:87-88`; H1, `STAGE2_75_PLAN.md:621`). No seed is dropped or exchanged. A run that cannot be read is run again and is never a count (C8, `STAGE2_75_PLAN.md:613-617`). A `D` record that passes is run once more on the second block, seeds 5042 to 5061 (ruled by the lead on 16 Nov 2026). |
| Judged settings | Seven: `F-`, `F+`, `P0`, `P+`, `M-`, `M+` and `D`. In each, every parameter not named is at `N`. |
| Records and runs | 7 settings × 2 delays × 2 policies = 28 records, 560 runs. If all four `D` records pass: 4 second-block records, 80 more runs [E, second revision]. |
| What passes | The bar's own rule is H1: 20 of 20 runs stay up and at least 17 meet every limit. **Ruled by the lead on 16 Nov 2026:** at a corner (`F-` to `M+`) the robot must stay up in 20 of 20; the other limits are recorded, not judged. The drawn panel is judged by the bar's whole rule. The reason given in this note: a corner's friction or play is a guessed extreme, and the duty limits read the torque that overcomes it (5.2, trap 4). Condition 2 holds when all 28 records pass.<br>This is an exception to the gate's sentence that its conditions use the bar's own pass rule (`BUY_GATE_DRAFT.md:34-36`, `:109`; P40, DL:55). The lead ruled that it is said in the owner's question (section 11) and written into `BUY_GATE_DRAFT.md` only if the owner approves it (the text is in section 9). |
| The second block of `D` | In the bar's own form (H4, `STAGE2_75_PLAN.md:642-652`) the headline stays the first block; the second-block record is printed beside it as "replicated" or "not replicated", is never pooled, and gives no verdict alone (H2, `:626-630`). This note takes that form. Whether "not replicated" stops a purchase is not in the ruling and is open for the lead (section 8, choice 27). |
| Reported beside it, no verdict | The same 28 records on the slippery floor (μ 0.54). The settings `J+`, `V-`, `G5`, `T+`, `X6L`, `W1`, `W2` on the grippy floor at the same delays: 28 more records. In every record: the number of runs with a step below 20 V ("left the supply window"; ruled by the lead on 16 Nov 2026: reported in every record, and a judged run is not failed on it yet) and the share of steps a leg drive spent inside its play. |
| Repeatable | Named settings draw nothing. `D` draws from the table at its recorded sha256 with one fixed seed per run. The same code, table and seeds give the same counts, so a rerun cannot change a verdict. |

**What these counts can carry** [E: my arithmetic, `calc_fix.py`, repeated and extended in the second revision by `calc_final.py`; it treats runs as independent with one true rate, which is only roughly so, since the same 20 seeds recur in every record].

| True stay-up rate per run | Chance that all 560 runs stay up (the recommended option) | 640 runs (the first version's rule; also 560 plus the second block's 80, if those were judged) | 720 runs (the stricter option: eight corners and `D`) | 80 runs (the lightest option: `D` alone) |
|---|---|---|---|---|
| 99.99 % | 95 % | 94 % | 93 % | 99 % |
| 99.9 % | 57 % | 53 % | 49 % | 92 % |
| 99.8 % | 33 % | 28 % | 24 % | 85 % |
| 99.5 % | 6 % | 4 % | 3 % | 67 % |
| 99 % | 0.4 % | 0.2 % | 0.1 % | 45 % |

- A policy pair that truly falls once in 1,000 runs fails condition 2 about four times in ten. P40 stops the purchase after two failed attempts at the training setup.
- **What a pass leaves open.** Passing all 28 records does not show that the policy stays up everywhere inside the ranges. The corners move one factor at a time. `D` tests 20 parameter sets on the headline block, and 20 more on the second block when the first passes, with the rotor inertia, the bus resistance and the trunk mass at nominal. The combined corners `W1` and `W2` are only reported (the second check's finding 2).
- The other way round, a pass is weak evidence at any one setting: 20 of 20 runs of one policy at one setting and one delay show, at 95 % confidence, only that the stay-up rate there is above 86 % (the panels' own bound, `panels.py:286-288`).
- The second half of H1 on the four `D` records: if a run meets every limit 90 % of the time, one record passes 87 % of the time and all four 57 %; at 95 %, 98 % and 94 %. If the four second-block records were judged too, all eight would pass 32 % of the time at a 90 % rate and 88 % at a 95 % rate [E, second revision].
- The first version gave 73 % at a 99.9 % rate. That was for 320 runs; with both policies its rule had 640 (the review's minor 12).
- The seeds are fixed, so a fall is not noise to be redrawn: it is one run of one policy at one setting, and it can be replayed and studied.

---

## 5. Where it attaches in the code

### 5.1 The hooks

One new module, `designlab/stage2/servo.py`, used as a context manager on a `plant.Plant`. It uses three hooks that exist today, each with a precedent:

| Hook | What the layer does there | Precedent |
|---|---|---|
| Model arrays | per-run arrays (damping, inertia, gains), per-step friction arrays; restored exactly on exit | `r5env.py:493-511`; `designlab/stage1/ledger.py:801-839` |
| `p.envelope` replaced | its `update()` runs before every physics step (`plant.py:687-692`): sag, the line, friction, the lag, the offset and the dead zone | the ledger swaps the envelope with an undo (`ledger.py:777-799`) |
| `p.step` as an instance attribute | after each step: the controller's own command back into `ctrl`, the reported angles patched in `sensordata` | `designlab/stage1/policy_rows.py:841-852` |

The layer **arms** at entry and **acts from the first physics step** after a reset. So the reset's forward pass, the first sensor reading and every tracker's copy of the friction are today's, on both sides.

Two things do not follow that rule. The trunk mass of `M-` and `M+` is not the layer's: the env applies its factor at reset, before the start state (`r5env.py:501`, `629`), and on the judging side the wrapper row makes the same call (`env.set_trunk_mass`, `designlab/stage1/env.py:317`) before the inner episode and undoes it after. And the gains and their copies follow the rule of 2d.

| Side | How it enters |
|---|---|
| Training env | `R5Config.servo` (default `None`: off). The env leaves the layer first thing in `reset` (`r5env.py:583`), before it restores its own arrays (`r5env.py:591`). It enters it after the delay is entered (`r5env.py:668-671`) and before a mid-task start fills the delay queue (`r5env.py:679-680`): the queued commands are folded with the plant's gain copies there, so the run's copies must already be set (2d; the second check's finding 11). Amended by the lead's fourth list (17 Nov 2026) to match 2d's split: the layer sets the plant's gain copies when it is entered and the model's gains at the first physics step, which serves the queue's first command; the first wording had "the run's gains and their copies must already be set". `reset(options={"servo": ...})` gives a named setting instead of a draw, as `options["factors"]` does today. The `r5env.py` line references of this note are those of the file at package S3 (refreshed on 17 Nov 2026, the same list), except where the check table at the top and section 10 record what an earlier revision found; the note's other code lines are still those of `82ba1ad`. |
| Judging side: the row | A wrapper row `servo.Served(inner, setting)` in the layer's own module, in the form of `bar.Delayed` (`bar.py:880-918`): `with served(p, setting): return inner.episode(...)`. Registered through `panels.register_rows` (`panels.py:865-872`); `bar.rows` then wraps it in the delay like any registered row (`bar.py:921-929`). |
| Judging side: the row's name | A controller name of its own. `register_rows` refuses `R0`, `R1` and `R2` (`panels.py:869-870`), and `R2s` is taken by the speed round (`speedfit.py:1213`), so the first version's `R0s<setting>` and `R5sN` forms are dropped. The bar's branches key on the controller name: no verdict for `R0` only (`bar.py:138`, `2437`), fall attribution for `R2` only (`bar.py:1603`, `1648`, `2287`), verdict pairs for `R2` and `R3` (`panels.py:82`). A served row under a new name is none of these, so whatever reads its records must carry its own lists: a served stand-in row gives no verdict, and no served row is attributed, because the relax labels cannot lift play, friction or sag and would mislead (`control.py:2061-2099`). Example names, for the lead to settle: `SV0/<setting>`, `SV2/<seed>/<setting>`, `SV5/<policy>/<setting>` (section 8, choice 18; the ruling of 16 Nov settles the route, not the names). |
| Judging side: the route | **Ruled by the lead on 16 Nov 2026:** a runner outside `bar.py`, so that the bar's hash and every registered record stay as they are. Why a route was needed: the bar's own runner cannot run these rows today. Its tasks are 3 and 4 only (`bar.py:140`, `3468`). On the three sets of the 75 cm round a row that is not `R0` runs only as a task's unit, and on any other reference set only `R0` rows run (`bar.py:333-340`, `2196-2222`; the check is inside `bar.evaluate`, `bar.py:2362`; only the pseudo-set `S1` of a Stage 1 run directory is exempt, `bar.py:2199-2200`). On the ruled route `bar.py` is not edited (section 8, choice 18). |
| Same model on both sides | Both call the one context manager with the one parameter object. A gate in G8a's form (`g8a.py:12-19`) proves it: the env with the zero action and a setting against the served `R0` row of that setting, the sha256 of the state after every physics step. |

### 5.2 Traps, and how the design meets them

| Trap | Answer |
|---|---|
| 1. On the judging side the tracker is built inside the episode and copies the plant's friction then (`panels.py:844`; `control.py:1204-1205`) | The layer writes its arrays at the first physics step, after that copy. The env's trackers are built at construction (`r5env.py:339-340`) and its reset check (`r5env.py:598-600`) still holds. |
| 2. `servo_delay` refuses a plant whose `command` or `reset` is replaced (`speedfit.py:336-338`) | The layer replaces neither. Either order works. |
| 3. The judge reads the envelope and the force ranges (`control.py:2304`, `2296-2365`) | The replacement keeps `moment`, `peak`, `rated`, `no_load`, `active`, `enabled`, `bind`, `restore`, `limits` and `describe`, with the nominal `peak` and `rated`. `p.drives` is not touched. Play never touches a force range. |
| 4. Friction is outside `actuator_force`, so the torque that overcomes it counts as load | Kept, and said in the rules text: gear friction is overcome by motor torque. It matters for the draft's condition 4 (the tail). The ratings it is read against are output-side and already net of friction, so the reading is on the harsh side (2a). |
| 5. A worker's plant is shared across rows (`panels.py:979-988`) | Exact undo on exit, tested as the ledger tests it (`ledger.py:833-837`). |
| 6. Snapshots hold MuJoCo's state and `ctrl` only (`control.py:1347-1388`) | The judged rows take none (`confirm` appears in neither `panels.py` nor `bar.py`). The layer is not offered to `control.run(confirm=True)` or to the ledger. |
| 7. New: on the judging side the estimator's model is the plant's live model (`estimator.py:247-252`; `control.py:1337`); the env gives it a nominal copy (`r5env.py:451`, `837`) | Concerns `M-` and `M+` only: the judging side's estimator would know the changed mass, and so would the tracker's own mass copy (`control.py:1207`). Ruled by the lead on 16 Nov 2026: policy rows need a nominal-model estimator before the two mass corners can be judged, and it joins the build's work packages (section 8, choice 11; package S8). |
| 8. New: `policy_rows.digesting` removes `p.step` on exit (`policy_rows.py:841-852`) | Digests are taken through the recorder or the env's step hook, as G8a does. The layer refuses to enter over a step hook it did not install (`p.step` already an instance attribute: a digest or another layer), and on exit it removes the step hook, raising after it if the one there is not its own. Amended by the lead's third list (17 Nov 2026), which keeps this stricter refusal: the first wording, "the layer restores whatever `step` it found", was not what the code does. |
| 9. New: the env restores gains, damping and friction at the start of `reset` (`r5env.py:498-500`) | Hence "leave first" above; the entry comes after the delay and before the delay queue is filled (5.1). |
| 10. From the review (M4): the plant and the tracker hold their own copies of the gains (`plant.py:575-576`; `control.py:1240-1248`) | The table "Gain copies" in 2d says which copy follows for each part. Test T14. |
| 11. From the review (M5): the bar keys its branches on the controller's name, and its runner admits only tasks 3 and 4 | Names and route in 5.1. The route is ruled (a runner outside `bar.py`); the names are still for the lead (section 8, choice 18). |
| 12. From the review (minor 5): `R5Config.describe()` is a plain `asdict` (`r5env.py:250-254`), and the env's `describe()` prints `FACTOR_RULE`, `RESET_ORDER` and `DRAW_ORDER` (`r5env.py:816-818`), which would be false with the layer on | `R5Config.describe()` drops the `servo` key when it is `None`, so today's dictionary is unchanged (T2), and `config_from` (`r5env.py:936`) reads the field back. With the layer on, the env prints servo variants of the three texts, kept as new constants beside the old strings, with the table's sha256 and the layer's own draw order. The old strings are not edited. |
| 13. From the review (minor 6): a G8a result hashes `r5env.py` (`g8a.py:254-255`) | No stored record holds a G8a result today [R; my grep of the JSON files under `stage2/` found only a review file that names it]. One made before the edit reads as made by other code afterwards, so G8a is run again after package S2. |

### 5.3 Files

| File | Change | In a record key? |
|---|---|---|
| `designlab/stage2/servo.py` | new: the table, the parameter object, the draw, the named settings, the layer, the wrapper row and its factories | only in the new rows' own keys, through their `sources()` |
| `designlab/stage2/g8s.py` | new: the env-against-row comparison, G8a's form | no |
| `designlab/stage2/r5env.py` | edit: the `servo` field; the two lines in `reset`; the check on the old friction draws; `R5Config.describe()` drops the key when the field is `None` and `config_from` reads it back; servo variants of the three rule texts the env prints (5.2, trap 12) | Not in a bar or panel key: no row imports it (`stage2/design/ROUND_C_DESIGN.md:465`). It is hashed into a G8a result (`g8a.py:254-255`): none is stored today, and one made before the edit goes stale (5.2, trap 13). The worktree branch `wt/wpc-ce` (at `cdc4762` at the second revision) holds unmerged edits to this file (last changed there by `4523ed6`; among them the keys a resumed run compares): merge after it, and pass the `servo` field through that comparison |
| `tests/test_stage2_servo.py`, `tests/test_stage2_g8s.py` | new | no |
| `designlab/stage1/*.py`, `designlab/*.py`, `designlab/stage2/readings.py`, `setio.py`, `r5core.py`, `r5lib.py` | none | they are the hashed sources |
| `designlab/stage2/bar.py` | none: ruled by the lead on 16 Nov 2026 (a runner outside `bar.py`; section 8, choice 18). The route not taken, a new bar task, would have edited it and moved the bar's hash `45719d03…` | the bar's hash is in every bar record's key |
| the runner of gate 1's records (name for the lead, for example `designlab/stage2/gate1.py`) | new, on the ruled route: it imports the bar's rows, conditions, seeds, readings and counts (the pieces round C already relies on, `ROUND_C_DESIGN.md:754-757`) and carries its own rules text and budget | only in its own records |
| the screened MJCF, the references, every record | none | — |

### 5.4 Why no existing record's key or content changes

1. The source hashes that records and the regress check hold are: the panels' (`panels.py:1572-1583`), the bar's (`bar.py:1677-1687`: `bar.py`, `readings.py`, `setio.py`, `speedfit.py`), the Stage 1 package's (`plant.py:262-275`: every `designlab/stage1/*.py`), the tuner's and the speed fit's (`designlab/stage2/regress.py:84`, `214-219`). No file in any of them is edited, and no file is added under `designlab/stage1/`. This holds on the route the lead ruled on 16 Nov 2026 for gate 1's records: a runner outside `bar.py` (section 8, choice 18, route c). The route not taken, a new bar task (route b), would have edited `bar.py` and moved the bar's hash.
2. A row added in its own module leaves the other rows' keys as they are: the panels' own rule says so (`panels.py:347-359`).
3. A served row has a new name, its own `describe()` (the setting and the table's sha256) and its own `sources()`. Its records are new records.
4. The model file on disk is not touched. The layer changes arrays in memory and restores them.
5. With `servo=None` the env runs today's code path and prints today's description.
6. No existing record is rerun or extended (`NEXT_SESSION_PROMPT.md:223-225`).

### 5.5 Tests that would prove it

| Id | Test | Proves |
|---|---|---|
| T1 | `panels.panel_sources()` and `bar.bar_sources()` give `c0027992…` and `45719d03…` ([P] today) after the change; `regress check` passes. Both hashes must stay: the route that would have moved the bar's hash was not taken (choice 18, ruled) | no key moved |
| T2 | With `servo=None`: G8a's existing tests pass unchanged; `R5Config(...).describe()` equals today's dictionary; `config_from(describe())` gives back the same config with the field `None` and with it set; with it set, the env's description prints the servo variants of its three rule texts | option off = today, bit for bit; the description is true with the option on |
| T3 | With the setting `today`: the state digests of the R0 row and of the env equal those without the layer, at 0 and 6 ms | the layer's arithmetic adds nothing of its own |
| T4 | After exit: every model array equals a copy taken before; `p.envelope` is the old object, bound; `p.__dict__` holds no `step` | exact undo |
| T5 | Friction: the array written equals m_j (T_c + μ_L τ̄) for a set τ̄; on a one-joint model the break-away torque is T_c | part a |
| T6 | Play: `actuator_force` equals the dead-zone formula at random states (my probe's form); on a one-joint model the static deflection under a load τ is τ / kp + b; at the ankle the reported pitch and roll carry the two motors' errors as 2b maps them; the neck and tail get no dead zone and no play shift of their reading (their `enc_offset` still applies, T9) | part b |
| T7 | Line and sag: the force range at set (ω, V) equals the formula, for example 7.4 N·m on the X4 at 8.69 rad/s and 22.2 V, and 0.4 N·m on the X6 at 8.01 rad/s and 21 V [E]; V equals the higher root of V = V0 − R P / V at set (V0, R, P); above V0² / 4R the step is flagged as a collapse and the run as having left the supply window; a run with a step below 20 V carries that mark | part c |
| T8 | Lag: a step in the command gives 1 − exp(−n dt / T_r); the box servos' command changes every 5th step | part d |
| T9 | Offset: the servo rests at target − o; the reported angle is the true one + o; the buffer returns the reading of n steps before | part e |
| T10 | Inertia: the mass matrix's diagonal rises by the drawn difference; `mj_setConst` after the undo leaves the model equal bit for bit | part f |
| T11 | The judge's `rated` and `peak` are the nominal arrays under every setting | registered limits do not move |
| T12 | Same seed, same parameters; the env's task, factors and push are equal with the option on and off; with `options["seeds"]` and no episode seed a draw is refused; in `D`, bar seed s gives the same parameter set on every call, from `SeedSequence([s, 3])`, with `armature` and `bus_r` at nominal and `latency_steps` not drawn; that set differs from the training draw of episode seed s; seeds 5042 to 5061 give 20 other sets | repeatable draws |
| T13 | G8s: the env with the zero action against the served `R0` row of the same setting, for `N`, `F+`, `P+`, `V-`, `J+`, `M-`, `M+` and three runs of `D`, at 0 and 10 ms | training and judging run one model; the mass is applied at the same instant on both sides |
| T14 | Gain copies: under `J+`, under `G5` and, for the tail, under `N` (kv 3.59) the force MuJoCo applies equals kp (q* − q) + kv (q̇* − q̇) + τ_ff with the run's gains, at random states and for the commands a delay queue holds at a mid-task start; under `kp_scale` s and `kv_scale` v it equals s [kp (q* − q) + kv q̇* + τ_ff] − v kv q̇; the plant's copies are restored on exit | part d: no command is served by a gain other than the one it was folded with (the review's M4) |

---

## 6. What it costs

| Quantity | Before | After | Basis |
|---|---|---|---|
| Layer arithmetic per physics step | 2.2 µs (today's `Envelope.update`, common case) | 28 µs | [P] `probe_layer_cost2.py`: the layer's numpy calls on stand-in arrays of this body's sizes, idle machine. Not measured in the env |
| Env steps per second, hard floor | 382 (builder) and 385 (reviewer), one env, load about 3, provisional [S] (`stage2/design/checkpoint/reviews/roundC__build_CE.md:11`; `roundC__review_CE.json:76`) | about 350 [E] | 260 µs per physics step all-in, plus 26 µs: about 10 % |
| Env steps per second, carpet | 307 and 302 [S], same lines | about 280 [E] | 326 µs plus 26 µs: about 8 % |
| Physics itself | 39 µs per step [S] (PLANT:1041) | unchanged in size: 14 friction rows, no new joint [P] | Unmeasured: whether larger friction and inertia change the solver's iteration count |
| Per run | — | two `mj_setConst` calls and a few array writes | the mass change already does the same |
| Condition 2's runs, per attempt at gate 1 | — | 28 judged records (560 runs); 56 reported records (1,120 runs); up to 4 second-block records of `D` (80 runs) | [E] the count of section 4, on the option recommended to the owner. Their wall time is not measured: package S4 |

Build effort, in work packages. Sizes are my estimates.

| Package | Content | Size | Needs |
|---|---|---|---|
| S1 | `servo.py`: table, parameters, draw, settings, layer, exact undo; tests T4 to T12 and T14 on the plant | about 600 lines of code and 500 of tests; a build, an independent review, a fix round | this note reviewed |
| S2 | Env option; `g8s.py`; tests T1 to T3, T13, and T14's case of a delay queue at a mid-task start | about 80 lines in `r5env.py`, 150 in `g8s.py`, 300 of tests | S1; wave 2a's edits to `r5env.py` merged |
| S3 | `Served` row, factories and names; the reported readings (runs that left the supply window; share of steps a leg drive spent inside its play); the named and drawn panels, the second block of `D` among them, through the route the lead ruled on 16 Nov 2026: a runner outside `bar.py` (section 8, choice 18), about 300 more lines and its own review [E] | about 250 lines and 250 of tests, plus the runner | S1; the lead's ruling on the names. Policy rows need `s2rows.py`; served R0 and R2 rows need nothing more |
| S4 | Measure before use: `r5env.throughput` with the layer; the served stand-in rows (R0 and R2 inside) at `N` and at each setting, with attribution off | runs only; rules first; listed as development episodes; through the chosen route, since the bar's own runner refuses these rows today (5.1) | S2, S3 |
| S5 | Rules text for gate 1's servo conditions; draft decision-log rows for each unsourced nominal; the buy gate edits | text | the lead |
| S6 | Bench sheet additions N1 to N6, as proposed additions (ruled by the lead on 16 Nov 2026); the gate draft's sentences on the bench reworded with them | text | the owner has seen them (choice 15) |
| S7 | The five questions to the maker (section 8, choice 1): written, in appendix A | five short messages | the owner sends them (ruled by the lead on 16 Nov 2026) |
| S8 | A nominal model copy for the policy row's estimator, and for the tracker's mass copy, when `s2rows.py` is built (ruled by the lead on 16 Nov 2026; 5.2, trap 7). Until it exists `M-` and `M+` run on R0 rows only and cannot be judged on policy rows | size not estimated | `s2rows.py`; condition 2's mass corners wait for it (choice 11) |

---

## 7. What this model still cannot show

Gate 1 on this model is a harsher simulation. It is not proof.

- **What a pass shows, in numbers.** 20 of 20 runs up, of one policy at one setting and one delay, bounds the stay-up rate there above 86 % at 95 % confidence, no more. And a policy pair that falls once in 1,000 runs fails the proposed condition 2 about four times in ten (section 4).
- **Inside the ranges.** Passing all 28 records does not show that the policy stays up everywhere inside the ranges. The corners move one factor at a time. `D` tests 20 parameter sets on the headline block, and 20 more on the second block when the first passes, with the rotor inertia, the bus resistance and the trunk mass at nominal. The combined corners `W1` and `W2` are only reported.
- **Heat.** No winding temperature and no limit on how long a peak lasts. The judge's RMS limit is 0.57 of the maker's rating, unverified (A1).
- **Structure.** No flex in links, mounts or push rods. The ankle's cranks are unit ratios (`drive.py:5-17`).
- **Gear dynamics.** No crossing time, knock or rattle in the play; no stick-slip; no cogging or torque ripple.
- **The play model's assumption.** The leg drives' dead zone assumes that the drive closes its loop on the motor-side encoder. No source says so (the record says only "dual encoder", RES:76, 144). If the loop closes on the output side, the model shows nothing of the play. The neck and tail servos have no play in the model at all, for the same reason, and the ankle's push rods have none either (no source).
- **Ratings and friction.** Friction is taken off output-side ratings that are already net of it (2a): the model is harsh there by up to the friction. A2's 0.70 of the maker's peak is no spare margin (2g), and no torque scale is drawn.
- **The hip-pitch drive at 24 V.** The catalogue has no data at 24 V. The nominal line is my estimate; the maker's back-EMF figure gives a lower one (1.2). The drawn corner alone does not reach it; the draw does once the zero-torque speed is drawn too, which the lead ruled on 16 Nov 2026 for the case that the maker has not answered (section 3). The judged nominal stays the estimate either way.
- **The drive's inside.** No sampling or filtering of its own feedback; no torque bandwidth on the feedback path (a third party measured about 10 Hz on an older, larger sibling; summary only); no velocity estimation.
- **Faults.** No under-voltage cut-out (steps below 20 V are only counted), no over-current or over-temperature trip, no time-out, no lost or late frame.
- **Braking.** The envelope gives full peak torque when braking and takes no account of the bus rising.
- **The neck and tail servos' own control law**, and the droop of their shared regulator.
- **The feet.** Floor compliance, tread grip, carpet (gate 1 is hard floors only, P40).
- **Cables.** No tether drag, no harness stiffness across joints.
- **Sensors beyond the joints.** IMU mounting error, vibration, temperature drift.
- **Change over time.** Wear, warm-up friction, the pack's state across a session.
- **Parts.** Which X6 ships (RES:154): the X6-50-L has a 45 N·m peak, so 31.5 N·m by A2's rule against the model's 42 N·m. `X6L` reports that case; nothing judges it. Units that differ more than the per-joint factor. The optional expressive head's neck-yaw servo has no row.
- **Five open facts the maker can answer now** (section 8, choice 1; appendix A). Until they are answered, the rotor inertia, the gain limits, the X6's line at 24 V, the X6's part and the play model's assumption stay open. A bench could find four of them later (N5, sheet A step 1, N2, N6); which X6 ships has no bench route.
- **The delay.** Under `N` a judged delay is the pure delay plus the model's command lag, and the budget's estimate already holds that rise (2d). In `D` the drawn feedback age (0 to 4 ms) and the drawn lag (up to 4 ms on the leg drives) come on top of the record's delay.
- **Every range marked "no source".** A pass inside a guessed range says the policy tolerates that range. It does not say the real part is inside it.

---

## 8. The open choices, after the lead's rulings of 16 Nov 2026

The rulings are in `LEAD_RULINGS.md` beside this note. They are the lead's proposals for a design that is not built; nothing in them is a result. Each choice below starts with its state, one of three:

- **Ruled by the lead on 16 Nov 2026**, with the ruling.
- **The owner's**, where the ruling says so.
- **Ruled, second list**: the nine choices the first list did not name were ruled by the lead later on 16 Nov 2026 (`LEAD_RULINGS.md`, second list): each recommendation stands. That change to this section was made by the lead and was not checked by a reviewer.

After the state, the recommendation and reason stand as they were written before the rulings ("my" there is the designer's or the fixer's), so that each ruling can be read against them. In the points marked "open for the lead" and in choice 7, "I" is the finisher's.

Count [second revision]. Of the first revision's 22 choices, 11 are ruled in whole or in part (1, 3, 10, 11, 15, 16, 18, 19, 20, 21, 22), one is the owner's (9), one is covered by a ruling made for the tail (7), and nine are named by no ruling (2, 4, 5, 6, 8, 12, 13, 14, 17). Choices 23 to 27 come from the second check and are all ruled. Three points that the rulings leave open are marked "open for the lead": in choices 3, 18 and 27.

1. **Five questions to the maker.** **Ruled by the lead on 16 Nov 2026:** asked before gate 1 is judged. Sending them is **the owner's**; the note carries the five questions as a text ready to send (appendix A).
   - (1) Which side of the gear the inertia figures are on: 0.000082 kg·m² on the X4-36-L, 0.66 kg·cm² on the X6-60.
   - (2) The motion-mode kp and kd ranges and their units on these two models.
   - (3) The X6-60's torque against speed at 24 V, and how 16 Vdc/krpm fits a no-load speed of 176 rpm at 48 V.
   - (4) Which X6 ships at the listed price: the X6-60 or the X6-50-L.
   - (5) Which encoder closes the position loop in motion mode, and which one the reply reports.
   - Until they are answered: A7 stays the nominal inertia, and `J+`, `G5` and `X6L` are reported. If (1) is "rotor side", the references and the Stage 0 screen need the real value, which is outside this note; the nominal then sits at the bottom of a 12 to 33-fold range (the V4 sibling's figure gives 0.039, the part's own 0.106).
   - As it stood: they cost nothing, and each closes a number that otherwise sits in a setting. The options were all five before gate 1 is judged (the reviewer's; ruled so), only (1) as a precondition (the first version), or none. Neither the reviewer nor I read the maker's protocol document: the reviewer found `cdn.robotshop.com` outside the session's allowed hosts [R, not rerun].
2. **Gain limits.** **Ruled by the lead on 16 Nov 2026 (second list, `LEAD_RULINGS.md`): the recommendation stands.** Question 2 of choice 1, which is ruled, asks the maker for the ranges and units. Recommendation as it stood: read the maker's V4.1 motion-mode document for the kp and kd ranges and units. If the caps are real, make `G5` the nominal; until then report it. Costs nothing; with the caps the hip and knee are far from critically damped, at a damping ratio of 0.22 to 0.72 [P; R, checked], and `N` would use gains the drive cannot hold.
3. **Nominal torque-speed line, and the X6's low end.**
   - The nominal line: **not ruled** (no ruling names it). Recommendation as it stood: my line through the maker's rated and no-load points; on the X6, the two 48 V points shifted to 24 V. It is an estimate, not a catalogue value: the X6 has no 24 V data, and the maker's back-EMF figure gives a lower line (1.2). Today's line stays as the easy end `T+`.
   - The X6's low end (the review's M2): **ruled by the lead on 16 Nov 2026:** its zero-torque speed is also drawn if the maker has not answered. That is (b) of the two forms that were offered: (a) `ts_corner` down to 2.15 rad/s with the zero-torque speed fixed; (b) also draw the X6's zero-torque speed between 8.01 and 9.215 rad/s, which reaches the back-EMF line itself (15.8 N·m at 5.8 rad/s and 24 V, against 20.3 with (a)) [E]. Section 3 now has the row `ts_zero`; 2c, section 4 and section 7 carry it.
   - **Open for the lead** (new; it comes from carrying the ruling through). The corner is drawn from 2.15 to 8.01 rad/s and the zero from 8.01 to 9.215. A draw can then put the two next to each other, and the slope τ_peak / (zero − corner) has no bound. The build needs a rule. Options: (i) draw the corner first, then the zero uniformly between the larger of 8.01 and (corner + 1.20) and 9.215; 1.20 rad/s is today's gap between the two (9.215 − 8.011), so no drawn line is steeper than about today's 35 N·m·s/rad [E, second revision]; (ii) draw one number and set both from it, from the back-EMF line (2.15, 8.01) to today's line (8.01, 9.215); simpler, but the nominal line (4.16, 9.215) is then outside the drawn family. I recommend (i).
4. **Backlash.** **Ruled by the lead on 16 Nov 2026 (second list, `LEAD_RULINGS.md`): the recommendation stands.** Its assumption, a loop closed on the motor side, is question 5 of choice 1, which is ruled. Recommendation as it stood: the run-time dead zone now, on the leg drives only; build the gap joint only if `P+` changes the counts against `P0`. It keeps every key and today's model, and the gap joint's cost and stability are unmeasured. Condition 2 measures the sensitivity that would justify it. The neck and tail wait for N6 (the review's M1).
5. **Friction's load term.** **Ruled by the lead on 16 Nov 2026 (second list, `LEAD_RULINGS.md`): the recommendation stands.** Recommendation as it stood: on, for all classes. It is the main candidate for "friction ten times the model's" under load and costs six array operations per step.
6. **Feedforward.** **Ruled by the lead on 16 Nov 2026 (second list, `LEAD_RULINGS.md`): the recommendation stands.** Recommendation as it stood: it keeps today's friction values under every setting. At gate 1 nobody knows the real friction, so the controller cannot either; it is also the env's existing rule.
7. **kv under a drawn inertia.** **Covered by a ruling made for the tail** (choice 23): "recomputed by the plant's rule under the new inertia … one rule for every drive". The ruling's words are about the tail's kv. I read "one rule for every drive" as taking this recommendation for every drive; the lead may say otherwise. Recommendation as it stood: recompute it by the plant's rule. A build would tune damping on the real drive; keeping kv would test a tuning mistake, not the servo. `G5` shows the case where the drive cannot reach it.
8. **Nominal bus.** **Ruled by the lead on 16 Nov 2026 (second list, `LEAD_RULINGS.md`): the recommendation stands.** Recommendation as it stood: 22.2 V with 0.05 Ω. The pack's nominal voltage is nearer the middle of a run than 24 V; the resistance is a placeholder until N3. The auxiliary power comes from the body's record: 20.3 W on this one (the Raspberry Pi 5), 44.8 W on the Jetson body's (2c).
9. **Which settings can stop a purchase (condition 2).** **The owner's** (the lead's ruling of 16 Nov 2026: it is the owner's question; the question is in section 11). Also ruled by the lead that day: the stricter option keeps the drawn panel, so that the owner's options differ only in the two harsh corners (choice 26). The three options, on the grippy floor at 4 and 10 ms, with the chance that a pair of policies with a true stay-up rate of 99.9 % per run passes [E, second revision; runs taken as independent]:
   - Option (a), recommended to the owner (the reviewer's form, B2): the six one-factor corners `F-`, `F+`, `P0`, `P+`, `M-`, `M+` and the drawn panel `D`. 28 records, 560 runs, 57 %. `J+` and `V-` are reported. It is not "only what has a stated basis": `F+` is above this note's own bound, and `P+` is four times the maker's limit, from one outside test of another model (section 3).
   - Option (b), the stricter one: the same, plus `J+` and `V-` judged. 36 records, 720 runs, 49 %. A purchase could then stop on one catalogue number that may be misread, or on a bus resistance nobody has; and `V-` as defined is outside the drives' supply window at the walk's summed peak (2c).
   - Option (c), the lightest: only `D` is judged in this condition (`N` is condition 1) and every corner is reported. 4 records, 80 runs, 92 %. A fall at a corner informs and does not block.
   - Under every option: `G5`, `T+`, `X6L`, `W1`, `W2` and the slippery floor are reported; section 4 defines the runs and the pass; a `D` record that passes is run again on the second block (choice 27).
10. **Steps below 20 V.** **Ruled by the lead on 16 Nov 2026:** reported in every record; judged runs are not failed on it yet. As it stood: `bus_r` has no source; after N3 it can become a limit. A run with such a step is marked "left the supply window" beside the counts, and a `V-` run so marked is not called "stayed up" (the review's B2). The option not taken: count any such run as not up in judged records too, which would make an unsourced resistance able to stop a purchase.
11. **Mass corners on the judging side.** **Ruled by the lead on 16 Nov 2026:** policy rows need a nominal-model estimator before the two mass corners can be judged. It joins the build's work packages (package S8, section 6). As it stood: give the policy row's estimator a nominal model copy when `s2rows.py` is built; until then run `M-` and `M+` on R0 rows only. Otherwise training and judging differ in what the estimator and the tracker know (trap 7). The mass is applied before the reset on both sides (5.1; the review's minor 8). Condition 2 as proposed needs the two mass corners, so it cannot be judged in full before S8 is built.
12. **Critic inputs.** **Ruled by the lead on 16 Nov 2026 (second list, `LEAD_RULINGS.md`): the recommendation stands.** Recommendation as it stood: do not add the servo draw to the critic's observation now. It would change `r5core.py`'s layout while other packages build on it; revisit if training under the draw converges slowly.
13. **Encoder offset.** **Ruled by the lead on 16 Nov 2026 (second list, `LEAD_RULINGS.md`): the recommendation stands.** Recommendation as it stood: patch the reading after each step, and accept that the reset's own reading is exact. The alternative, turning each child body by the offset [P: every hinge is alone in its body at the body's origin], is cleaner at reset but edits the model's geometry per run.
14. **The 60 cm body.** **Ruled by the lead on 16 Nov 2026 (second list, `LEAD_RULINGS.md`): the recommendation stands.** Recommendation as it stood: its own table before its gate 1, same code. Its AK45-36, X4-P12-10 and STS3250 have their own maker figures (R2 §6) and no table here.
15. **The bench sheet and gate 2's wording.** **Ruled by the lead on 16 Nov 2026:** the six added measurements (N1 to N6) are proposed additions to `stage2/jetson75/BENCH_SHEET.md`; the gate draft's sentences on what the bench measures are reworded with them once the owner has seen them. Those sentences are `BUY_GATE_DRAFT.md:59-61` and `:67`. The draft says the bench measures gear play; no sheet does (BENCH:327-331 keeps the gear on one flank on purpose). Section 9 has the text. The options not taken: add N1 only, which is the one the draft already promises; change the draft's wording and add nothing.
16. **Condition 4 (the tail).** **Ruled by the lead on 16 Nov 2026:** read at the worse of nominal and high friction, with the tail's nominal inertia. The neck's same asymmetry is kept and flagged. Condition 4 stays a proposal until put to the owner. The tail's nominal inertia stays at the top of its range (choice 24).
   - As it stood (changed by the review's M3): the first version took the third-party fit's friction, a third of today's values, and left out its inertia, about 126 times today's (0.0284 against 0.000226 kg·m²) [E, second revision; the first revision wrote 120]. On the stored references the inertia alone adds 0.05 to 0.11 N·m to the tail's RMS torque, against a 0.491 N·m limit [P] (2f).
   - Ruled so: the tail's nominal inertia is 0.028 kg·m², as the table has it. Condition 1 is then trained and judged with it too, and with the tail's kv that follows from it (choice 23).
   - Not taken, the reviewer's first form: keep today's inertia as the tail's nominal and read condition 4 at the worst of `N`, `F+` and a tail-inertia-high setting.
   - The neck: friction from another model's fit, inertia by the study's rule. It is in no condition of gate 1 by name, but its sustained load is one of the bar's limits. Kept and flagged until N5, because the two third-party inertias are of other models and differ 4.5-fold.
17. **The box servos' nominal stiffness.** **Ruled by the lead on 16 Nov 2026 (second list, `LEAD_RULINGS.md`): the recommendation stands.** Recommendation as it stood: keep the model's until sheet A is run; the draw reaches down to 0.3. No source decides it; the one third-party figure is for another voltage version at its default register. (This is about kp. The tail's nominal kv is ruled: choice 23.)
18. **Route and names of gate 1's records** (the review's M5).
   - The route: **ruled by the lead on 16 Nov 2026:** a runner outside `bar.py`, so that the bar's hash and every registered record stay as they are. That is route (c) below. Why a route was needed: the bar's own runner cannot run a served row, or any row that is not `R0`, on a reference set outside its tasks 3 and 4 (5.1).
   - (a), not taken: run them as units of task 3 or 4. No change to the runner; but those budgets were registered for the 75 cm round, which has run, and the bar's name branches would still read a served row as a row with a verdict.
   - (b), not taken: a new task in `bar.py`. The bar's hash `45719d03…` would move once, and every bar record made before then would name the old hash in its key. P40 says the bar and its registered rules are unchanged (DL:55).
   - (c), ruled: a runner of its own outside `bar.py` that imports the bar's rows, conditions, seeds, readings and counts, with its own rules text, budget and directory. No existing hash or record moves, and P40's sentence holds as written. Cost: its own run rules and its own review.
   - The names: **not ruled, open for the lead.** A controller name of its own for each kind of served row; 5.1 has examples.
19. **The judged delay under the model** (the review's M7). **Ruled by the lead on 16 Nov 2026:** the command lag and the 100 Hz hold stay in the nominal model and the note says so. A measured feedback age goes to `sense_steps`.
   - Where the note says so: 2d, section 7, section 9 ("Two things to know") and section 11. It is a little harsher than the numbers P40 approved.
   - Not taken: no lag and no hold in `N`, both drawn only in training and in `D`, which would have made gate 1's delays exactly the pure delays P40 approved. And, for a measured value, the bench sheet's own rule (the feedback age beyond 2 ms added to T, `sense_steps` held at 0), which treats an old reading as a late command.
   - The rest of 2d's rule for a measured value (H and the dead part of D to k; the rise to `cmd_lag_ms`) is this note's proposal; no ruling names it.
   - In `D` the drawn feedback age and the drawn lag come on top of the record's delay (choice 25).
20. **A torque-scale draw** (the review's M6). **Ruled by the lead on 16 Nov 2026:** none now. The note says the 0.70 of assumption A2 is used up (2a, 2g, section 7). Not taken: (b) draw the leg drives' limit × 0.85 to 1.0 through the envelope's `torque_scale` [E, no source]; (c) one reported setting with the limit × 0.95, that third-party shortfall applied to the maker's peak, 0.667 × 30 N·m against the planning 21 N·m [E]. The one third-party figure is of an older, larger model, and N2 would measure it.
21. **The stand-in rows' ankle cap under a drawn inertia** (from the review's M4). **Ruled by the lead on 16 Nov 2026:** accepted. **Note, amended by the lead's third list (17 Nov 2026) to say what the code does:** the layer sets the plant's gain copies when it is entered and the model's gains at the first physics step (the split, kept), so a served R2 row whose tracker is built inside the episode takes the run's kv in its ankle cap at `J+` and `W1` (11.28 against 2.08 on the ankle motors), and both are reported only (2d). No judged setting moves an ankle motor's gain. The env's trackers are built before the layer is entered and keep the nominal copies; they use no cap, and the policy rows, not built yet, should pass none either. The note's first wording, that an R2 stand-in row's ankle cap keeps the nominal gains at `J+`, was not what the code does. Not taken: served R2 rows whose tracker is made after the gains are set; the wrapper would then have to reach inside the inner row (as built, the copies set at entry reach a tracker built inside the episode without that).
22. **What "passes" means at a corner.** **Ruled by the lead on 16 Nov 2026:** at an extreme the robot must stay up in 20 of 20; the other limits are recorded, not judged. The drawn panel is judged by the bar's whole rule. This is an exception to the gate's sentence that its conditions use the bar's own pass rule, so it is said in the owner's question and written into `BUY_GATE_DRAFT.md` only if the owner approves it.
   - The facts behind the ruling (the second check's finding 3). The draft is not of one mind. Its condition 2 says "stays up in 20 of 20" (`BUY_GATE_DRAFT.md:41`); its preamble (`:34-36`), its line 109 and P40 (DL:55) say the gate's conditions use the bar's own pass rule, which also asks that 17 of 20 meet every limit. The first revision cited only the first of these. Stay-up only at the corners is therefore an exception to a sentence of the gate. That sentence stands in the draft the owner answered on and in P40's row; the draft marks it neither approved nor proposed (`BUY_GATE_DRAFT.md:6-10`).
   - What follows: the owner's question says it (section 11), and the draft's lines 34-36 and 109 get the exception only if the owner approves (the text is in section 9).
   - Not taken: the whole rule at every judged setting. The duty limits would then be read against guessed extremes of friction, and `F+` adds to every drive's load by construction (5.2, trap 4).
23. **The tail's kv** (new; the second check's finding 7). **Ruled by the lead on 16 Nov 2026:** recomputed by the plant's rule under the new inertia, and `kv_scale` re-based on it: one rule for every drive.
   - The numbers [E, second revision]: 3.59 N·m·s/rad with the table's 0.028 kg·m² (3.60 with the fit's unrounded 0.0284), against today's 2.72. The maker's stall over no-load, 0.62 N·m·s/rad, is 0.17 of it; `kv_scale`'s low end is now 0.15.
   - Carried through: 2d (the box servos' gains; the gain copies), 2f, section 3 (`armature` and `kv_scale` of the tail), T14.
   - Not taken: keep the box servos' kv at today's.
24. **The tail's inertia range** (new; the second check's finding 7). **Ruled by the lead on 16 Nov 2026:** the nominal stays at the top of the range, and the note says so (section 3; `J+` in section 4). Not taken: widen the range.
25. **Delay-like draws in the drawn panel** (new; the second check's finding 8). **Ruled by the lead on 16 Nov 2026:** kept, and said (2d, section 4, section 7, section 9). Not taken: hold `sense_steps` at 0 in `D`.
26. **Does the stricter option keep the drawn panel** (new; the second check's finding 6). **Ruled by the lead on 16 Nov 2026:** the stricter option keeps the drawn panel, so that the owner's options differ only in the two harsh corners. It is then 720 runs, not 640 (choice 9, option (b); section 11).
27. **Replication of the drawn panel** (new; the second check's choice 6). **Ruled by the lead on 16 Nov 2026:** a drawn-panel record that passes is also run on the bar's second block (seeds 5042 to 5061). The reason the second check gave: `D`'s 20 parameter sets are known after a first attempt. Carried through: section 4, T12, sections 6, 9 and 11. If all four `D` records pass that is 80 more runs [E, second revision].
   - **Open for the lead:** the ruling does not say what a second-block record that fails does to the verdict.
   - (a) The bar's own form (H4, `STAGE2_75_PLAN.md:642-652`): the headline stays the first block; the second block is printed beside it as "replicated" or "not replicated", is never pooled and gives no verdict alone. "Not replicated" then goes to the owner with the verdict and does not stop a purchase by rule. The runs that must all stay up remain 560 on the recommended option.
   - (b) Judged: a `D` record counts as passed only if its second-block record passes too. On the recommended option 640 runs must then stay up (53 % at a true rate of 99.9 %), and eight records must each have 17 of 20 within every limit (32 % if a run meets every limit 90 % of the time, against 57 % for four) [E, second revision].
   - I recommend (a): it is the bar's own registered form, to which P40's sentence points, and (b) would be a second departure from that sentence for the owner to approve. Until the lead rules, sections 4, 9 and 11 word it as (a) and say that it is open.

---

## 9. Proposed text for the buy gate draft

Proposals only. Each block is written to be true to the tables above. It calls a value the maker's only where section 3 does, and it names no bench measurement that the bench sheet does not hold.

**Replacement for the bullet at `BUY_GATE_DRAFT.md:30-31`:**

> - **A servo model, drawn at random per run** (`stage2/design/servo_model/SERVO_MODEL_DESIGN.md`). Per drive type: joint friction (dry, viscous and a part that grows with load); gear play on the leg drives, as a dead zone in the drive's own loop; the torque limit against speed and bus voltage, with the voltage sagging under the legs' current; a short lag on the command; the encoder's zero error and age; the rotor's reflected inertia. Today's model has ideal servos and one unsourced friction value (`DECISION_LOG.md` A18).
>   - *Where the numbers come from.* The leg drives' dry friction, gear play and torque-speed line start from the maker's catalogue figures as this study reads them: the backdrive torque; the backlash limit; the rated and no-load points, which for the hip-pitch drive exist at 48 V only. The rotor inertia, the drives' gain limits, the encoder zero error and the bus resistance do not, and the note's table says why. Six values are borrowed from third-party tests of other servo models or versions. Values without a source are labelled "unsourced". No value is measured on this build's hardware.
>   - *What would replace them.* Nothing does today. The bench sheet plans the delay, stiffness, friction and the tail servo's heat, and it has not been run (P32). Gear play, torque against speed and voltage, the bus's resistance, the torque's rise, the rotor inertia with the load-dependent friction, and the encoders' zero and side would each need a measurement added to the sheet (the note's proposals N1 to N6). The maker can answer five open facts now, before any part is bought (the note's section 8, choice 1, and its appendix A); a bench could find four of them later.
>   - *Two things to know.* The play model assumes that the leg drive closes its loop on its motor-side encoder; no source says so. And under this model a judged delay of, say, 10 ms is 10 ms of pure delay plus a command lag of 0.5 or 1.2 ms on the leg drives, with the neck and tail targets refreshed 100 times a second instead of 500. In the drawn panel of condition 2 a run also draws a feedback age of 0, 2 or 4 ms and a command lag of up to 4 ms on the leg drives, on top of the record's delay.

**Replacement for condition 2 at `BUY_GATE_DRAFT.md:41-44`** (the option recommended to the owner; which settings are judged is the owner's, section 8, choice 9, and the question is in section 11. If the owner takes another option, the list of settings and the counts below change with it):

> 2. **Robust to the servo (*proposed*).** The 0.30 m/s walk on the grippy floor, at 4 ms and at 10 ms, under both policies, on the bar's 20 seeds, at seven settings of the servo model.
>    - Six settings are named corners. Each moves one factor and leaves the rest nominal: friction low and high (each drive type's low and high set, not a multiple of one number; the high set is above the study's own estimate of the bound); gear play none and 1° on the leg drives; trunk mass 0.9 and 1.1. At a corner all 20 runs must stay up. The number that also meet every limit is recorded and not judged there: this is the one exception to the gate's sentence that its conditions use the bar's own pass rule.
>    - The seventh is a drawn panel: each of its 20 runs draws its servo values from the model's table, except the rotor inertia and the bus resistance, which stay nominal, with a seed fixed by the run's own seed; the same 20 sets serve both delays and both policies. It must pass the bar's whole rule: 20 of 20 stay up and at least 17 meet every limit. A drawn-panel record that passes is run once more on the bar's second block of seeds (5042 to 5061), which draws 20 other sets; the result is printed beside the first as "replicated" or "not replicated".
>    - That is 28 records and 560 runs, and every run must stay up; with the second block, up to 4 more records and 80 more runs.
>    - Reported beside it, with no verdict: the slippery floor; the rotor inertia at its high end, which waits for the maker to say which side of the gear its figure is on; a low bus (21.0 V with 0.15 Ω), which by the study's own arithmetic falls below the drives' 20 V supply limit and whose resistance has no source; the drives' gain limits; today's torque-speed line; the other hip-pitch part that may ship; two combined corners; and, in every record, the number of runs in which the bus fell below 20 V.
>    - The settings' values are the table of `stage2/design/servo_model/SERVO_MODEL_DESIGN.md`, at the hash recorded with the rows. Values without a source are labelled so there. When the maker's answers or a bench value arrive, the settings are proposed again.
>    - What the numbers can carry: a pair of policies that truly stays up in 999 runs of 1,000 passes all 560 about 57 % of the time; and 20 clean runs of one policy at one setting and one delay show, with 95 % confidence, only that it stays up more than 86 % of the time there. A pass does not show that the policy stays up everywhere inside the ranges, or with two extremes at once.

The second-block sentence above is worded in the bar's own form; what a failed second block does to the verdict is open for the lead (section 8, choice 27).

**The pass rule's exception at `BUY_GATE_DRAFT.md:34-36` and `:109`.** Ruled by the lead on 16 Nov 2026: written into the draft only if the owner approves it. The text for that case:

> At lines 34-36, after "with the chosen look's walk": "One exception, in condition 2: at a named corner of the servo model only the first half of the rule is judged (20 of 20 runs stay up); the number of runs that also meet every limit is recorded."
>
> At line 109: "The gate's conditions use the bar's own pass rule, with the one exception stated in condition 2."

**Condition 4 (the tail) at `BUY_GATE_DRAFT.md:47`.** No new text is proposed here. Ruled by the lead on 16 Nov 2026 (section 8, choice 16): the tail's continuous load is read at the worse of the nominal and the high-friction setting, with the tail's nominal inertia, which is taken from the same third-party fit as its friction. Condition 4 stays a proposal until it is put to the owner.

**Gate 2's sentences on what the bench measures (`BUY_GATE_DRAFT.md:59-61` and `:67`).** The draft says the bench measures "each drive type's friction and gear play". Friction is on the sheet (sheet B). Gear play is not. Ruled by the lead on 16 Nov 2026: N1 to N6 are proposed additions to the sheet, and these sentences are reworded with them once the owner has seen them. That is form (i) of the two that were offered:

> (i) "… each drive type's friction and, once that measurement is added to the bench sheet, its gear play …", together with the addition of N1 to `stage2/jetson75/BENCH_SHEET.md` (and of N2 to N6; section 3).
>
> (ii) "… each drive type's friction …", with "gear play" also taken out of gate 2's condition 2 until a sheet measures it.

Form (ii) was not taken (section 8, choice 15).

---

## 10. Changes after the two independent checks

### 10.1 After the first review

This part is the first revision's record. It stands as the fixer wrote it, except for two additions marked "second revision"; where it says "for the lead", section 8 now has the lead's ruling.

The review is `REVIEW.md` beside this note (16 Nov 2026; verdict "usable after the listed fixes"). One line per finding, by its id there. Every file, line and record the review cites was opened before acting on it. No finding was rejected. Five details of the review are corrected or not taken; each is marked in its line. "For the lead" means that the note now states the facts and my recommendation, and that the choice is open in section 8.

| Finding | What was done |
|---|---|
| B1 | Fixed. Section 9 no longer says that a maker's value is the nominal; section 3 says for each row what the nominal is (1 the maker's own figure, 4 a maker's figure for a related quantity, 6 third-party, 27 the study's; second revision: 28, since a row was added). "The bench measures it" is gone from sections 3 and 9; the missing measurements are named as proposed additions N1 to N6. One sub-point not taken as an error of the table: a zero nominal for a signed encoder error does not contradict the maker's 0.4° accuracy, which bounds its size; the row now says so. |
| B2 | Fixed, and for the owner. Section 4 defines the corner rule: settings, values, seeds, runs, what passes, what is reported, and the pass probabilities. `J+` and `V-` are reported; `P+` is back at 1.0° on the leg drives; `F+` is labelled; the drawn panel `D` is added; a `V-` run below 20 V is "left the supply window". Which settings can stop a purchase is the owner's choice: section 8, choice 9 (three options, the reviewer's form recommended), choice 22, and section 11. Not repeated from the review: that the third-party unit was "worn". Reader 3's summary does not say it, and I could not open the paper. |
| M1 | Fixed. No dead zone and no reading shift on the neck and tail until N6 (2b, 2e, section 3, `P+`). The cited third-party lines were opened and say what the review quotes. Sections 2b, 7 and 9 say that the leg drives' dead zone assumes a motor-side loop, and that this has no source. |
| M2 | Fixed, with a sub-choice for the lead. The back-EMF check is in 1.2; the X6's low end is 2.15 rad/s; the wording is "my line … shifted to 24 V … no 24 V data"; `X6L` is a reported setting. Found while checking: with the zero-torque speed fixed, the one drawn corner does not reach the back-EMF line itself (20.3 against 15.8 N·m at 5.8 rad/s). Choice 3 therefore offers to draw that speed too; recommended if the maker has not answered. Also made plain: the nominal line keeps the 48 V slope and shifts; the first version's "speeds halved" was loose. |
| M3 | Fixed, with options for the lead. The tail's nominal inertia is now the same fit's 0.028 kg·m². The reviewer's table was reproduced by rerunning the probe (2f). Choice 16 keeps the reviewer's other form as option (b), and flags the same asymmetry on the neck (recommended: keep until N5). |
| M4 | Fixed. 2d has a table of which gain copy follows for each part, a rule for the build, and test T14. One point for the lead: the stand-in rows' ankle cap cannot follow a drawn inertia (choice 21; recommended: accept). |
| M5 | Fixed in the text; the route is for the lead. "`bar.py` is not edited" now holds only on the route recommended in choice 18, a runner of its own; 5.1, 5.3, 5.4, T1 and packages S3 and S4 say so. Served rows get controller names of their own; the first version's `R0s…` and `R5s…` forms are dropped. Options (a) to (c) are in choice 18. Checked on the way: the regress check does not hold the bar's hash. |
| M6 | Fixed by stating it, with options for the lead. 2a says that friction is counted against output-side ratings and gives the net torques; 2g says that A2's 0.70 is used up; section 7 repeats both. Whether to draw a torque scale is choice 20 (recommended: not now). |
| M7 | Fixed, with options for the lead. 2d states the three points and proposes a rule for where a measured D and S go. Section 9 tells the owner that a judged delay carries the lag. Choice 19 (recommended: keep the lag in the nominal model and say so). |
| M8 | For the lead. Choice 1 now recommends all five questions to the maker before gate 1 is judged (options: all five; the inertia only; none). Section 7 and package S7 carry it. Neither the reviewer nor I read the maker's protocol document. |
| Minor 1 | Fixed. "Floats" is gone; 2b says that the joint is not free inside the gap (an added inertia of 2 ms × kv). The toy probe was not rerun; its formula was checked by arithmetic. |
| Minor 2 | Fixed. The sag is the closed-form root; above V0² / 4R the step is a collapse and a fault, not a clip to 12 V (2c, T7). |
| Minor 3 | Fixed. 0.122 N·m is the MX-106 fit's Stribeck term; its constant term is 1.0e-5 (1.3, 2a, section 3). |
| Minor 4 | Fixed, with one correction to the review. The worktree's edits are committed there now (`wt/wpc-ce` at `4523ed6`; second revision: the branch is at `cdc4762`, its `r5env.py` unchanged since `4523ed6`) and still unmerged; they are not "uncommitted". `config_from` is already in the merged file; `RESUME_KEYS` is not. The check table says which files the line numbers are of. |
| Minor 5 | Fixed. 5.2 trap 12, 5.3 and T2 say how `describe()` and the three rule texts change. |
| Minor 6 | Fixed. 5.2 trap 13 and 5.3 say that a G8a result hashes `r5env.py` and is run again after the edit. |
| Minor 7 | Fixed. The V4 sibling's 0.3 kg·cm² (0.039 kg·m² reflected, 12 times A7) is in 1.1, 2f and section 3. That the nominal sits at the bottom of the range stays the lead's choice (choice 1). |
| Minor 8 | Fixed. 5.1 says when the mass is applied on each side; `M-`, `M+` and three `D` runs are in T13; choice 11 says that condition 2 needs the ruling. |
| Minor 9 | Fixed, with one correction to the review. In the merged file the lines are `r5env.py:475-476`, not 478-480. Section 4 and T12 say that a named setting is then required. |
| Minor 10 | Fixed. 2b states the mapping from the two motors' errors to the ankle's joint readings; T6 tests it. |
| Minor 11 | Fixed. 2c says that the model shifts the line while today's rule scales both speeds, with the X4's corner at 21 V under each (7.61 against 7.24 rad/s). |
| Minor 12 | Fixed. Section 4 gives the pass probabilities with both policies (560 runs: 57 % at a 99.9 % rate; the first version's 640 runs: 53 %). |
| Minor 13 | Fixed. Section 9 cites `SERVO_MODEL_DESIGN.md`; no served row is named `R2s…` (5.1). |
| Minor 14 | Fixed by stating the scope. The status line and section 7 say that the expressive head's neck-yaw servo has no row. 2c says that 20.3 W is the Raspberry Pi 5 record's auxiliary input and gives the Jetson record's 44.8 W. |
| The owner's question | Fixed: section 11. The review's proposed words "anywhere in the table's ranges" are not used: 20 or 40 drawn runs do not show that. The counts of the table's rows are my own (section 3). |

Count: 25 findings (2 blockers, 8 major, 14 minor, the owner's question). Rejected: none. Fixed with nothing left open: 17 (B1, M1, the 14 minor ones, the owner's question). Fixed in the text with a choice left to the lead or the owner: 7 (B2, M2, M3, M4, M5, M6, M7). Handed to the lead as a choice alone: 1 (M8).

Not verified in this revision: any maker's page; the third-party paper behind the sibling's play and torque-constant figures (summary only); the reviewer's one-joint toy probe (not rerun); the designer's timing probe (not rerun); whether the friction put on the ankle motors' tendons acts as section 2a assumes once the physics runs (the review lists it as unchecked too; only the count of friction rows in a forward pass was rerun on 16 Nov: 14, 18 with tendon friction added, 14 with the ankle joints' own set to 0). No episode was run.

### 10.2 After the second check, and the lead's rulings

The second check is `RECHECK.md` beside this note (16 Nov 2026; verdict "ready after the listed fixes"). The lead's rulings on the open choices are `LEAD_RULINGS.md` (16 Nov 2026). One line per finding of the check, by its number there. Before a finding was applied, the lines it cites were opened and its numbers were recomputed (`calc_final.py`; the second checker's `probe_ref_speed.py` rerun). These fixes were not checked a third time.

| Finding | What was done |
|---|---|
| 1 | Fixed. Section 11's "what a pass would tell you" is the check's text, with four things added: the other limits are recorded at an extreme; each delay carries the model's lag; a passing random panel is run once more; conditions 3 and 4 are named. Section 9's drawn-panel bullet has the check's addition. |
| 2 | Fixed. The sentence is in section 4 (under the table of chances), in section 7 and, in plain words, in section 11. One clause is changed from the check's text: `D` tests 20 more sets on the second block when the first passes (the lead's ruling on replication). |
| 3 | Fixed, and ruled. Choice 22 has the check's facts and the lead's ruling; section 4 states the exception; section 9 has the text for the draft's lines 34-36 and 109, to be written only if the owner approves; section 11 says it in the owner's question, in option 1 and for every option. One clause not taken: "a sentence the owner approved". The draft marks that sentence neither approved nor proposed (`BUY_GATE_DRAFT.md:6-10`), so the note says "a sentence of the gate" that stands in the draft the owner answered on and in P40's row. |
| 4 | Fixed. 2c: the brisk reference is 0.35 m/s (`STAGE1_PLAN.md:85`; the probe prints a stride of 0.311 m in 0.887 s, which is 0.351 m/s, and 0.243 m/s for the nominal walk). |
| 5 | Fixed. Sections 7, 9 and 11 no longer say "only the maker". They say that the maker can answer five facts now and that a bench could find four of them later. To keep the second sentence true to the table, N6 got one clause: if neither the maker's document nor the maker's answer says, the leg drives' encoder side is read on the bench by N1's load reversal. Before, N6 sent the leg drives to the maker's document only. |
| 6 | Fixed, with the lead's ruling that the stricter option keeps the drawn panel. Section 11 has the check's three options with their runs. Option 2 is 720 runs and 49 %: the check's main text had 640, and its last line gave 720 and 49 % for the panel kept. Choice 9 no longer says "what has a stated basis". One phrase differs from the check's: "a panel of runs with 20 randomly drawn sets of motor values", where the check had "20 runs with randomly drawn motors"; the panel is 80 of the 560 runs. |
| 7 | Fixed, and ruled (choices 23 and 24). The tail's nominal kv by the plant's rule is 3.59 with the table's 0.028 kg·m²; the check's 3.60 comes from the fit's unrounded 0.0284. The tail's `kv_scale` is re-based: 0.15 to 1.2, with the maker's 0.62 N·m·s/rad at 0.17 of the nominal. The tail's `armature` row says that the nominal is the top of the range and that `J+` equals `N` there. `D` holds `armature` at nominal on the leg drives until question 1 and on the neck and tail until N5. |
| 8 | Fixed, and ruled (choice 25). `D` does not draw `latency_steps`. The delay-like draws are kept and said in 2d, section 4, section 7 and section 9's "Two things to know". |
| 9 | Fixed. Section 11: "full strength up to their rated speed" and "joint sensors with no zero error". |
| 10 | Fixed. The 86 % sentences of sections 4, 7, 9 and 11 carry "95 % confidence" and "one policy at one setting and one delay". Section 11 says "condition 2 alone needs 560 runs". |
| 11 | Fixed, all nine points. (1) `BUY_GATE_DRAFT.md:59-61` in the check table and in choice 15. (2) T14 is in packages S1 and S2. (3) 2d's rule is limited to `armature` and `gain_cap`; 5.1 and trap 9 say that the layer is entered before the delay queue is filled. (4) `wt/wpc-ce` is at `cdc4762`; its `r5env.py` was last changed by `4523ed6` (the check table, 5.3, 10.1). (5) "No play shift" in 2b and T6. (6) 2c: the 7.24 rad/s example has `ts_corner` at today's 8.69. (7) 2c: 214 W is REC:301 plus REC:298. (8) The status line cites `BUDGET_75.md:318`. (9) `D` draws from `SeedSequence([s, 3])`. |

Count: 11 findings. Fixed: 11. Not taken: none in whole. Three details differ from the check's own words or numbers, each marked in its line: "approved" in finding 3; 640 runs for option 2 in finding 6; 3.60 in finding 7.

**The lead's rulings, where they changed the note.** Section 8 states each ruling. Carried beyond it:

| Ruling | Carried to |
|---|---|
| The pass rule at a corner; the exception is said to the owner and written into the draft only if the owner approves | section 4 ("What passes"), section 9 (condition 2; the text for the draft's lines 34-36 and 109), section 11 |
| The tail's kv by the plant's rule, `kv_scale` re-based | 2d (box servos' gains; gain copies), 2f, section 3 (the tail's `armature` and `kv_scale`), T14 |
| The tail's nominal inertia stays at the top of its range | section 3, `J+` in section 4 |
| Delay-like draws kept in the drawn panel | 2d, section 4, section 7, section 9 |
| The stricter option keeps the drawn panel | choice 9, the table of chances in section 4, section 11 |
| A passing drawn-panel record is run on the second block | section 4, T12, section 6, section 9, section 11 |
| The five questions are asked; sending is the owner's | choice 1, section 7, section 9, section 11, package S7, appendix A |
| A runner outside `bar.py` | 5.1, 5.2 (trap 11), 5.3, 5.4, T1, package S3 |
| The lag and the 100 Hz hold stay in the nominal model; a measured feedback age goes to `sense_steps` | 2d, section 7, section 9, section 11 |
| The X6's zero-torque speed is also drawn if the maker has not answered | 2c, section 3 (the new row `ts_zero`), section 4, section 7 |
| Condition 4 read at the worse of nominal and high friction, with the tail's nominal inertia | 2f, section 9 |
| No torque-scale draw; the ankle cap accepted; steps below 20 V reported | 2g, 2d, 2c and section 4 |
| Policy rows need a nominal-model estimator before the mass corners are judged | trap 7, the new package S8 |
| N1 to N6 are proposed additions; the draft's bench sentences are reworded once the owner has seen them | section 3, section 9, package S6 |

**Changes no finding asked for**, made while the rulings were carried through:

- Section 3 has a 39th row, `ts_zero`. The counts are now 39 = 1 + 4 + 6 + 28 (section 3, section 11). The second check counted 38 = 1 + 4 + 6 + 27, which was right for the first revision.
- The tail's inertia against today's is "about 120 to 126 times" in 1.4 and 2f and "about 126 times" in choice 16, where the first revision had 120 (0.0284 against 0.000226 kg·m² is 126; 0.027 is 119).
- Section 11 says "in training, each run draws these at random": on a judged panel a named setting draws nothing (section 4).
- The branch head is `cd68f2c`; no cited code file changed since `82ba1ad`.

**Left open for the lead by this revision** (section 8): the rule that keeps the X6's drawn corner below its drawn zero (choice 3); the names of the served rows (choice 18); what a second-block record that fails does to the verdict (choice 27). Nine choices are named by no ruling and stay open (2, 4, 5, 6, 8, 12, 13, 14, 17), and choice 7 is read as covered by the ruling on the tail's kv.

Not verified in the second revision: any maker's page or document; the third-party papers and files; the two source hashes; the timing probe and the reviewer's toy probe; whether the proposed questions reach the right address. Nothing was run but arithmetic and one read-only probe of the stored references.

---

## 11. For the owner, in plain words

Simulation and design only. Nothing here was measured on a motor of this build.

**What today's simulation leaves out.** Its motors are ideal. They push exactly as told, with one small friction value that nobody sourced, no looseness in the gears, full strength up to their rated speed, a battery that never dips, and joint sensors with no zero error.

**What the proposed model adds, one part per line.** In training, each run draws these at random within a table of ranges, so the walking controller cannot rely on one lucky value.
- Friction: each motor type gets its own, and it grows when the joint carries load.
- Gear looseness: a leg motor does nothing until the slack is taken up, and its sensor misreads by the slack. Leg motors only; the neck and tail wait until we know where their sensor sits.
- Strength at speed: a motor pushes less the faster it turns, and less again on a low battery.
- Battery dip: the voltage drops while the legs draw current.
- Response: the push builds up over a millisecond or so instead of at once, and the neck and tail get new targets 100 times a second, not 500.
- Sensors: each joint sensor can be off by up to about a degree, and its reading can be a few milliseconds old.
- Rotor weight: the spinning part inside each motor resists speeding up; on the main leg motor it may be up to 33 times heavier than today's guess.

**Where the numbers come from.** The table has 39 lines. One is a maker's own figure (the voltage printed on the battery). Four start from a maker's figure for something related (how hard a leg motor is to turn by hand; the upper limit on its gear looseness). Six are borrowed from other people's tests of similar servos, not these. The other 28 are the study's estimates or guesses, and the table says so line by line. The real values would come from a bench, and six measurements it would need are not on the bench sheet yet (gear looseness, strength at speed, battery dip, how fast the push builds, rotor weight, sensor zero); this note proposes adding them. The maker can give five facts for free, before any part is bought; a bench could find four of them later. The lead has ruled that the maker is asked before gate 1 is judged. Sending the questions is yours: the five are in appendix A, ready to send.

**What a pass at gate 1 would tell you, and what it would not.** It would tell you that two separately trained walking controllers keep the simulated robot walking at 0.30 m/s in two kinds of test. First, with every motor value at the table's expected value, on both hard floors, at each delay from 0 to 10 ms in 2 ms steps (condition 1, which you approved). Second, on the grippy floor at 4 and 10 ms (condition 2, proposed here): with 20 sets of motor values drawn at random from the table, and with friction, gear looseness and trunk weight each put at its extreme in turn. At an extreme the robot only has to stay up; the other limits are recorded there, not judged. The 20 random sets keep the rotor weight and the wiring resistance at their expected values. A pass does not show that the controller stays up everywhere between these points, or with two extremes at once.

Three more things about these tests.
- Under the model each delay carries a small extra lag in the motors' response, about 0.5 to 1.2 ms on the leg motors. The tests are therefore a little harsher than the plain delay numbers you approved. The lead has ruled that the lag stays in and that you are told.
- A random-panel result that passes is run once more with 20 other random sets, and you are shown whether it repeats.
- This note does not cover gate 1's pushes, turn and stop (condition 3). The tail servo's load (condition 4) would be read with this model; it is still a proposal and is not asked here.

A pass would not tell you that the real motors are inside the table's ranges. It says nothing about heat, flexing parts, worn gears or a motor switching off on low voltage. The test is strict in one way and weak in another: condition 2 alone needs 560 runs that all stay up, so a pair of controllers that truly falls once in 1,000 runs would still fail it about four times in ten; yet 20 clean runs of one controller at one setting and one delay show, with 95 % confidence, only that it stays up more than 86 % of the time there.

**The question for you (gate 1, condition 2): which motor settings may stop the purchase of the bench parts?** Each count of runs below is the number of settings × 2 delays × 2 controllers × 20 runs.
1. *Recommended.* Six extremes and the random panel can stop it: friction low and high, gear looseness none and 1°, trunk weight 10 % down and up, and a panel of runs with 20 randomly drawn sets of motor values. 560 runs must all stay up. At these extremes the robot must stay up; the other limits are recorded there, not judged. The high-friction extreme is above the study's own estimate, and 1° is four times the maker's limit. Two harsher settings are run and shown to you but cannot stop a purchase yet: the heavy rotor, which hangs on a catalogue number the maker must explain, and the weak battery, which uses a wiring resistance nobody has measured.
2. The same, plus the heavy rotor and the weak battery: 720 runs must all stay up. Stricter; the purchase could then stop on a misread catalogue number or on a wiring resistance nobody has measured, and the weak-battery setting as defined falls below the motors' own minimum voltage at the walk's peak draw.
3. Only the expected values and the random panel can stop it: 80 runs in this condition must all stay up. Lightest: a controller that falls at an extreme would still pass. Every extreme is still run and shown to you.

Under every option:
- **At an extreme the robot must only stay up. The other limits, such as how hard and how long each motor works, are recorded there and not judged.** On the random panel the bar's whole rule applies: all 20 runs stay up and at least 17 meet every limit. This is an exception to the gate's sentence that its conditions use the bar's own pass rule. Options 1 and 2 judge extremes, so choosing one of them approves the exception; the buy gate's text is changed only if you approve.
- For scale, by the study's arithmetic with the runs taken as independent: a pair of controllers that truly stays up in 999 runs of 1,000 would pass option 1 about 57 % of the time, option 2 about 49 % and option 3 about 92 %.
- Each random-panel record that passes is run once more on 20 other seeds: 80 more runs if all four pass. The result is shown beside the verdict. Whether a failed repeat may also stop the purchase is not settled; it is open for the lead (section 8, choice 27).
- In this condition the slippery floor, the motors' gain limits, today's strength-at-speed line, the other hip motor that may ship and two combined extremes are run and shown without a verdict.

---

## Appendix A. Five questions for the makers

Ruled by the lead on 16 Nov 2026: the five questions are asked before gate 1 is judged, and sending them is the owner's. Each text below stands alone and can be sent to a maker's support address as it is. None names a project, a person or an organisation. All five concern the two leg drives, which have one maker. Nothing is asked of the makers of the neck and tail servos: their open point, the side of the position sensor, is the bench's (N6).

The figures quoted in the questions are the maker's catalogue values as the study's research record transcribes them [M]. I did not re-open the maker's pages; the record read the store listing of question 4 on 7 Nov 2026 (RES:146). The line under each text says where its figures stand and what the answer changes in this note. Those lines are not part of the message.

**Question 1. The rotor inertia: before or after the gearbox.**

> Subject: RMD-X4-P36-36-E-L and RMD-X6-P20-60-E: is the listed inertia before or after the gearbox?
>
> Hello. Your data for the RMD-X4-P36-36-E-L (X4-36-L, V4.1) list an inertia of 0.000082 kg·m². Your data for the RMD-X6-P20-60-E (X6-60, V4) list 0.66 kg·cm². For each part: is this the inertia of the rotor alone, before the gearbox, or the inertia seen at the output shaft, after the gearbox (ratio 36 and 19.612)? Please also give the inertia seen at the output shaft of each part, in kg·m². Thank you.

Figures: RES:85 and RES:70 (X4); RES:154 and RES:138 (X6; the record gives "0.66 kg·cm²" without the word inertia). The answer sets the `armature` rows of the two leg drives and decides whether `J+` stays a setting (section 3; section 8, choice 1).

**Question 2. The gain ranges of the motion mode, and their units.**

> Subject: RMD-X4-P36-36-E-L and RMD-X6-P20-60-E: range and units of KP and KD in the Motion Mode Control Command
>
> Hello. I plan to drive the RMD-X4-P36-36-E-L and the RMD-X6-P20-60-E with the Motion Mode Control Command over CAN (0x400 + ID). For each part: (1) Does it accept this command? (2) What are the smallest and the largest KP and KD it accepts? (3) In which units do KP and KD act? I need them as N·m/rad and N·m·s/rad at the output shaft. If they are defined on the motor side or as a current, please give the conversion. The manual of another model of the series, the RMD-X4-P36-24, gives KP 0 to 500 and KD 0 to 5 with no units: do those ranges apply to these two parts? Thank you.

Figures: LAT:441-442 [M via search extract]. The answer sets `gain_cap` and decides whether `G5` becomes the nominal (section 3; section 8, choice 2).

**Question 3. The RMD-X6-P20-60-E at 24 V.**

> Subject: RMD-X6-P20-60-E: torque against speed at 24 V, and the back-EMF constant
>
> Hello. The RMD-X6-P20-60-E (X6-60, V4) is listed with a rated torque of 20 N·m at 153 rpm and a no-load speed of 176 rpm, both at 48 V, and with a supply range of 20 to 55 V. I plan to run it from a 24 V supply. (1) At 24 V, what is the no-load speed in rpm, and up to which speed in rpm can it give its peak torque of 60 N·m? A torque-speed curve at 24 V, or a few pairs of N·m and rpm, would answer this. (2) The listed back-EMF constant is 16 Vdc/krpm. At 176 rpm at the output and a gear ratio of 19.612 that gives about 55 V, which is more than the 48 V supply. Is the constant per 1,000 rpm of the motor or of the output shaft, and how is its voltage defined? Please give it in V per 1,000 rpm of the motor, with the definition. Thank you.

Figures: RES:131-138, RES:154; 55 V is my arithmetic [E]. The answer sets the X6's `ts_corner` and `ts_zero` and ends the draw of the zero-torque speed (section 3; section 8, choice 3).

**Question 4. Which hip-pitch part ships.**

> Subject: Store listing "RMD-X6-P20-60": is it the X6-60 (V4) or the X6-50-L (V4.1)?
>
> Hello. The store listing "RMD-X6-P20-60" gives a mass of 0.78 kg, a peak torque of 50 N·m and a backlash of 10 arcmin. The manual and the data sheet of the RMD-X6-P20-60-E (X6-60, V4) give 0.82 kg, 60 N·m and at most 15 arcmin. The listing's figures look like those of the RMD-X6-P20-50-E-L (X6-50-L, V4.1). Which part is shipped today when this listing is ordered? Please answer with the full model name and version, and with its mass in kg, its peak torque in N·m and its backlash in arcmin. Thank you.

Figures: RES:146, RES:154. The answer decides whether `X6L` becomes the model of the hip pitch: a planning peak of 31.5 N·m in place of 42 N·m, and another friction (sections 1.2, 4 and 7).

**Question 5. Which encoder closes the position loop.**

> Subject: RMD-X4-P36-36-E-L and RMD-X6-P20-60-E: which encoder closes the position loop, and which one is reported?
>
> Hello. Both parts are listed with two encoders (17 and 18 bit on the RMD-X4-P36-36-E-L, 17 and 17 bit on the RMD-X6-P20-60-E) and with a backlash of up to 15 arcmin. (1) In position control and in the Motion Mode Control Command, does the position loop close on the encoder on the motor side of the gearbox, or on the one on the output side? (2) Which encoder's angle does the drive report in its replies: motor side or output side? In which unit, and with how many counts per turn of the output shaft? (3) Can the user choose between the two? Thank you.

Figures: RES:75-76, RES:143-144. The answer decides whether the dead zone of 2b is the right model of the leg drives' gear play, and whether the reported angle carries the play (2b, 2e; N6).
