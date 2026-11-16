<!-- Working paper, 16 Nov 2026: the designer's note for the servo model of gate 1 (BUY_GATE_DRAFT.md; DECISION_LOG.md P39), written from the three readers' reports beside it. NOT YET REVIEWED: an independent review is running and its fixes are not in. It proposes; it sets no rule, and nothing in it is a result or a measurement on hardware. -->

# Servo model for gate 1: design note

**Status.** Working paper for the study's lead, 16 Nov 2026. It proposes. It sets no rule and changes no file. Not yet reviewed. Body `75-M-H-10` only (the 60 cm body: section 8, choice 14).

**Three kinds of statement, kept apart.**
- *Prospective design*: sections 2 to 6, 8 and 9. None of it is built.
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
| no source | Nothing found. The value is a placeholder with a wide range. |

**Files.** Paths are relative to `hardware/trex_form_study/`. RES = `designlab/data/research/actuators_2026-11-07.json`. PLANT = `stage1/runs/2026-11-08/75-M-H-10/plant.json`. REC = `round2/refine/feet_75/candidates/75-M-H-10.json`. LAT = `stage2/jetson75/LATENCY_AND_BUS.md`. BENCH = `stage2/jetson75/BENCH_SHEET.md`. MAT = `round2/refine/MATERIALS.md`. DL = `DECISION_LOG.md`. R1, R2, R3 = the readers' reports in `stage2/design/servo_model/`. Code lines are those of the branch head `26b057e`.

**My scripts** (in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_design/designer/`): `calc.py` (arithmetic), `probe_runtime_arrays.py`, `probe_deadzone.py`, `probe_joint_frames.py` (forward pass only), `probe_layer_cost2.py` (numpy timing, no model), `probe_ref_accel.py` (reads stored reference files). A copy of this note is `DESIGN_NOTE.md` in the same folder.

**What I checked in the readers' reports, and what I correct.**

| Claim | Check | Result |
|---|---|---|
| R1: the training env exists only on the worktree branch | `cmp` of `r5env.py`, `r5core.py`, `r5lib.py`, `g8a.py` | Merged since (commit `9ecefa5`). The files in `designlab/stage2/` are identical to the worktree's, so R1's line numbers hold. |
| R1: `Plant.step` is the only physics step; the envelope is called before it | `plant.py:687-692`; grep of `mj_step` | Confirmed. |
| R1: a new row's key carries its own `describe()` and `sources()` | `bar.py:880-918`, `1677-1705`; `panels.py:347-359` | Confirmed. |
| R1: backlash could be done by a zero force range | `control.py:2296-2365` (`_episode`) | Not usable: the judge counts a force at its range as a drive at its torque limit. Section 2b uses a command shift instead. |
| R1: a servo layer adds very roughly 4 to 12 % | my timing | 28 µs against 2.2 µs today per physics step [P]: about 10 % (section 6). |
| R2: the maker's numbers and their lines in RES | RES:55-86, 122-155, 186, 1482-1512, 1546-1579 | Confirmed as transcribed. |
| R2: no bench sheet measures gear play | BENCH:327-331, 484-594 | Confirmed. `BUY_GATE_DRAFT.md:59-60` says the bench measures it. |
| R3: third-party servo fits | the saved files in `dl_bam/`, `dl_misc/` | Confirmed for the STS3215 7.4 V fit, the MX-106 fit, the Feetech firmware law and the small biped's model. |
| The draft's corners (friction ×0.5 and ×1.5, play 0 and 1°, mass 0.9 and 1.1) | against section 3 | Friction and play corrected (section 4). Mass confirmed. |

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
| Reflected inertia | 0.00324 kg·m² [A7] (PLANT:291) | "inertia 0.000082 kg·m²", side of the gear not stated [M] (RES:85); × 36² = 0.106 if rotor-side [E] | a 33-fold open range |
| Gear play | none (`STAGE2_PLAN.md:306`) | ≤ 15 arcmin [M] (RES:75). Ankle push rods: no source | all of it |
| Command path | delay k × 2 ms; torque follows at once | winding L/R = 0.45 ms [E from M, RES:85]; drive-side delay 0.10 to 1.50 ms [A] (LAT:241) | torque rise; no measurement |
| What the controller reads | 14-bit angle, 2 ms old [A] | two encoders, 17 and 18 bit; accuracy 0.4°, repeatability 0.3° [M] (RES:76, 85) | zero error; which encoder is reported and which closes the loop: no source |

### 1.2 Leg drive RMD-X6-P20-60-E: hip pitch (2 drives)

| Aspect | Modelled today [S] | Known about the real part | Missing |
|---|---|---|---|
| Gains | kp 1140, kv 30.78 [A] (PLANT:132) | same sibling caps, 500 and 5 [M via search extract] | with the caps the damping ratio is 0.25 [P] |
| Torque limit | ±42.0 N·m [A2] (PLANT:299) | peak 60 N·m; 40 N·m heats 29 °C in 10 s [M] (RES:132-133) | peak duration |
| Torque against speed | 42 N·m up to 8.01 rad/s, zero at 9.22 rad/s: the 48 V speeds halved (PLANT:301-302; `catalog.py:77-82`) | rated 20 N·m at 153 rpm, no-load 176 rpm, both at **48 V** [M] (RES:131-135). No data at 24 V | the line through the maker's points, speeds halved, gives 42 N·m only below 4.16 rad/s and 10 N·m at 8.01 rad/s [E] |
| Bus voltage | as 1.1 | window 20 to 55 V, nominal 48 V [M] (RES:136-137) | as 1.1 |
| Friction | 0.15 N·m + 0.2 N·m·s/rad [A18] | backdrive 1.6 N·m; efficiency 72.7 % [M] (RES:154). The V4.1 sibling X6-50-L: 0.2 N·m [M] (RES:186). Which part ships is open (RES:154) | a ten-fold gap to the maker's figure |
| Reflected inertia | 0.004304 kg·m² [A7] (PLANT:303) | "0.66 kg·cm²", side not stated [M] (RES:154); × 19.612² = 0.0254 if rotor-side [E] | a 6-fold open range |
| Gear play | none | ≤ 15 arcmin; the store page says 10 [M] (RES:143, 154) | all of it |
| Command path | as 1.1 | L/R = 1.24 ms [E from M, RES:154] | as 1.1 |
| What the controller reads | as 1.1 | 17 and 17 bit [M] (RES:144) | as 1.1 |

### 1.3 Neck servo DYNAMIXEL XM540-W270-T (1 drive)

| Aspect | Modelled today [S] | Known about the real part | Missing |
|---|---|---|---|
| Servo law and gains | the same ideal servo with feedforward; kp 177.02, kv 4.16 [A] (PLANT:114) | a box servo with register gains and no torque input; control law not in the repository (R2) | the real law; stiffness is found on the bench, not set (BENCH:523-525) |
| Torque limit | ±3.106 N·m = 0.293 × stall [A] (PLANT:263) | stall 10.6 N·m at 12 V; no continuous rating [M] (RES:1491-1492) | — |
| Torque against speed | peak to 1.885 rad/s (0.6 × no-load), zero at 3.142 [A] (PLANT:265-266) | no-load 30 rpm at 12 V [M] (RES:1494) | a straight line from stall to no-load gives the peak up to 2.22 rad/s [E]: today's line is inside it |
| Friction | 0.15 + 0.2 [A18] | none published. Other models of the maker [T]: MX-106 dry 0.12 N·m, viscous 0.051, load term 0.20 (file) | everything for this model |
| Reflected inertia | 0.000201 kg·m² [A, rule of A7] (PLANT:267) | not published. Other models [T]: 0.0058 (XH540-V150, summary), 0.026 (MX-106, file) | a 29 to 130-fold open range |
| Gear play | none | 15 arcmin [M] (RES:1502) | all of it |
| Command path | 500 Hz targets, delay k | planned at 100 Hz on its own serial port [A] (LAT:335-336); delay not known (LAT:346-347) | the 10 ms hold; its own response |
| What the controller reads | 14 bit | 12-bit sensor [M, summary; R3] | coarser by 4 than modelled; not in the balance loop |

### 1.4 Tail servo Feetech STS3215 C018, 12 V (1 drive)

| Aspect | Modelled today [S] | Known about the real part | Missing |
|---|---|---|---|
| Servo law and gains | the same ideal servo; kp 49.13, kv 2.72 [A] (PLANT:120) | [T, file, on the 7.4 V version]: a proportional loop into motor voltage; default gain 32 × 0.166 per rad of duty. With the 12 V stall torque that is 15.6 N·m/rad, 0.32 of the model's [E] | the real law; damping from back-EMF only: stall / no-load = 0.62 N·m·s/rad, 0.23 of the model's kv [E from M] |
| Torque limit | ±0.862 N·m [A] (PLANT:275) | stall 2.94 N·m; "rated" 0.98 N·m, no duty definition [M] (RES:1554-1556) | the continuous rating (A1; BENCH sheet E) |
| Torque against speed | peak to 2.827 rad/s, zero at 4.712 [A] (PLANT:277-278) | no-load 45 rpm [M] (RES:1558) | straight line: peak up to 3.33 rad/s [E]; today's line is inside it |
| Friction | 0.15 + 0.2 [A18]. They dominate this servo's load (DL:124) | [T, file, 7.4 V version]: dry 0.052 N·m, viscous 0.059 N·m·s/rad | about a third of today's values, if the 12 V part is alike |
| Reflected inertia | 0.000226 kg·m² [A] (PLANT:279) | [T, file, 7.4 V version]: 0.0284; another project's model: 0.027 | about 120 times today's value |
| Gear play | none | ≤ 0.5° [M] (RES:1566); [T, summary]: 0.6 to 0.9° unloaded, 1.3° loaded | all of it |
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
| Is the load term worth its cost | Yes for the leg drives. Cost: six array operations per step. Evidence that it exists: [T] fits of geared servos need it (0.20 to 0.31, other models); the maker's efficiency leaves 3.2 N·m (X4) and 4.0 N·m (X6) unexplained at the rated point [E from M: efficiency, rated current, resistance]. No value for these drives has a source. |
| Is a stiction (Stribeck) term worth its cost | No, left out. No data for any of the four parts. MuJoCo's friction constraint is soft at 2 ms (solref 0.02 s [P]) and could not show a sharp break-away. The range of T_c covers a higher static value. |
| Feedforward | Stays today's (the A18 values), on both the training and the judging side. The controller does not know the drawn friction (section 5, trap 1). |

### 2b. Gear play (backlash): a dead zone, no extra joint

| Item | Proposal |
|---|---|
| Representation chosen | A stateless dead zone in the drive's own loop. With u = kp (c − L) − kv ω (the torque today's servo asks for) and b = half the play: torque = u − kp b if u > kp b; u + kp b if u < −kp b; 0 in between. |
| In MuJoCo | The layer sends c − clip(u / kp, −b, +b) instead of c. MuJoCo's own servo then gives exactly the torque above: checked on this plant, joints and ankle tendons, to 1e-14 N·m [P]. The reported angle is the true one plus the same clip (the motor-side reading). |
| What it assumes | The drive's loop closes on the motor-side encoder. The rotor crosses the gap within a step. |
| Size | kp b at the maker's limit: 1.3 N·m on the X4 joints, 2.5 N·m on the X6, 0.65 N·m per ankle motor [P]. At 1° of play: 5.2, 9.9 and 2.6 N·m [P]. A joint with less torque asked than that floats. |
| Cost in physics steps | None. No extra degree of freedom, no extra constraint row, five array operations per step. |
| Why it is needed | A loaded leg joint sits on one flank and shifts by the play when the load reverses: at touchdown and lift-off. At 0.5 m that is 2.2 mm per joint at the maker's limit and 4.4 mm at 0.5° [E]. The controller does not see it. |
| What it cannot show | The time the rotor takes to cross (for 1° on the X4: 2 to 11 ms with today's inertia, 13 to 61 ms with the maker's figure read rotor-side [E]). The knock when the gap closes. Rattle or a limit cycle. The link running free of the rotor's inertia inside the gap. Anything at all if the loop closes on the output encoder. |
| The alternative, not chosen now | A motor-side degree of freedom per drive with a limited gap (two open projects do this for small servos, R3 §2a). It needs a second plant builder outside the hashed files, or an edit that changes every key. nv goes from 20 to 34 [P, R1]. The settled stance, mid-task starts and the estimator's structure check change. With play 0 it is still another model, so no step-for-step identity with today. Its cost and stability at 2 ms are unmeasured. Section 8, choice 4. |

### 2c. Torque limit against speed and bus voltage; voltage sag

| Item | Proposal |
|---|---|
| Line | Driving side: limit = clip(s (ω0(V) − \|ω\|), 0, τ_peak). Braking side: τ_peak, as today (`drive.py:28-31`). s = τ_peak / (ω0(24 V) − `ts_corner`). |
| Voltage | ω0(V) = ω0_catalogue × min(1, V / V_nominal): today's rule (`catalog.py:77-82`), applied every step instead of once. V_nominal is 24 V on the X4 and 48 V on the X6 [M]. |
| Sag | V = V0 − R_bus P / V_last, kept between 12 V and V0 (the 12 V floor is my choice, to keep the arithmetic finite). P = Σ over the 12 leg drives of [max(τ ω, 0) + c_cu τ²] / 0.92, plus 20.3 W. One step of lag. |
| Constants | c_cu = 3 R_phase / Kt² = 0.185 W/(N·m)² on the X4 and 0.139 on the X6 [E from M, RES:71-72, 139-140]. 0.92 is the drive efficiency [A6] (`designlab/power.py:21-27`). 20.3 W is the record's auxiliary input [S] (REC:298). No credit for braking power, as A6. |
| Parameters | `ts_corner` per leg-drive class; `bus_v0` V0 and `bus_r` R_bus per run. |
| Box servos | Keep today's lines. They sit on a 12 V regulator, not on the bus. |
| A new reading | Steps with V below 20 V, the leg drives' lower supply limit [M] (RES:69, 137). Reported, not judged. The pack's end-of-charge 21.0 V [A11] is 1.0 V above it: at the walk's summed peak of 8.9 A [S] (MAT:205-206) that allows 0.11 Ω in all [E]. |
| What the judge reads | The nominal corner speeds and peaks, unchanged, so the registered limits do not move (section 5, trap 3). |
| Why it is needed | Today the X6 is given 42 N·m up to 8.0 rad/s on half its nominal voltage. The catalogue supports about 10 N·m there [E from M]. The study flags this itself (`LESSONS_LEARNED.md:113`; A2). A push recovery is where speed and torque peak together with the current. |
| Sense of scale [E] | With the maker-points line at 22.2 V the X6 has 22 N·m at the 5.8 rad/s its animal-pace reference asks for, against that reference's own peak of 16.6 N·m. The X4 has 13 N·m at the knee's 7.1 rad/s, against 7.2 N·m. Prescribed references (`stage1/runs/2026-11-08/75-M-H-10/refs/fr05.npz`), not closed loop; peaks need not coincide. |

### 2d. The drive's own loop, and the delay in the parts the bench measures

The delay itself is unchanged: k whole physics steps on the whole command, through `speedfit.servo_delay` (`STAGE2_PLAN.md:351-360`; P40's delays). The model adds what sits around it.

| Part | What it is | Estimate today [A] (LAT:196-199 and its term table; baseline wiring) | Bench reading | In the model |
|---|---|---|---|---|
| C | the controller's compute step | 0.10 to 0.50 ms | none until firmware exists (BENCH:90-92) | inside k |
| H | tick to the end of the last command frame | 0.11 to 0.88 ms | H, every cycle (BENCH:85-87) | inside k |
| D | frame end to the torque change | 0.10 to 1.50 ms; 2.50 if the drive's loop runs at 500 Hz | D, in 0.2 ms bins (BENCH:88-89, 119-120) | its dead part inside k; its rise is `cmd_lag_ms` |
| S | age of the feedback at the tick | 1.23 to 2.62 ms; 0 to 0.62 ms beyond the model's 2 ms | S (BENCH:93-94) | `sense_steps` (2e) |
| T | tick to torque | 0.31 to 2.88 ms; 0.31 to 3.50 ms with S | T = H + D (BENCH:84) | k × 2 ms, by the bench sheet's own reading rule |

| Added part | Equation | Parameters | Why |
|---|---|---|---|
| Command lag | y ← y + (1 − exp(−dt / T_r)) (c − y); the drive acts on y | `cmd_lag_ms` T_r per class | the winding's rise (0.45 and 1.24 ms [E from M]) and the drive's outer loop. Nothing is measured |
| Box servos' update rate | their y changes only every 5th physics step; the phase is drawn | `cmd_hold` | the bus plan sends the neck and tail targets at 100 Hz [A] (LAT:335-336); the model updates them at 500 Hz |
| Box servos' gains | kp × `kp_scale`, kv × `kv_scale`, per run | two per box servo | their stiffness is a register value found on the bench (BENCH:523-525); the one third-party figure is a third of the model's (1.4) |
| Leg drives' gain limit | setting `gain_cap`: kp ≤ 500, kv ≤ 5, with the plant's own copies (`plant.py:575-576`) set too, so the feedforward torque is not scaled | none drawn | the sibling manual's ranges [M via search extract]. If they apply, hip and knee are at a damping ratio of 0.22 to 0.72 [P] |

Not modelled in the loop: its sampling of the feedback. MuJoCo evaluates the servo on the true state every 2 ms, with its damping treated implicitly. A lag on that path needs the servo law in code and gives up the implicit damping (R1 §3.3). The drive's loop is quoted at 500 Hz to 1 kHz for an earlier generation [M via search extract, LAT:443], which is no slower than the physics step.

### 2e. Sensor side

The readers found two things that matter, and one that does not.

| Part | Equation | Parameters | Why, or why left out |
|---|---|---|---|
| Encoder zero error | The drive holds L + o = target, so the layer sends c − o. The reported angle is L + o. | `enc_offset` o per joint | Every sim-to-real project R3 read draws it (±0.01 to ±0.05 rad). The maker's accuracy is 0.4° = 0.007 rad on the X4 [M]. It moves the foot and the controller cannot see it. |
| Play in the reading | reported angle also carries the clip of 2b | none new | The sign-dependent error of a motor-side encoder. |
| Feedback age | joint angles and rates are those of `sense_steps` steps earlier; the box servos' refresh every 5th step | `sense_steps` 0 to 2 | The budget's S is up to 0.62 ms beyond the model on the baseline wiring and up to 4.17 ms on a Jetson USB path [A] (LAT:196-199). |
| Encoder quantisation | left as today (14 bit) | — | The leg drives' 17 and 18 bit [M] are finer than modelled. The box servos' 12 bit is coarser, but they are not in the balance loop. |
| IMU delay | left out | — | Read by the same controller at 1 kHz or faster in every wiring [A] (LAT:250-251). |

One artefact: the reset's own first reading is exact; the offset appears with the first physics step (section 5).

### 2f. Reflected rotor inertia (added; not in the draft's list)

| Item | Proposal |
|---|---|
| Equation | Per run: `dof_armature` (joints) and `tendon_armature` (ankle motors) = the drawn J, then MuJoCo's constants recomputed (`mj_setConst`, as the env's mass change does). Both arrays are read at run time [P]. kv = 2 √(kp (I_home + J − J_today)): the plant's own rule (`plant.py:864`) on the drawn inertia. |
| Parameter | `armature` per class. |
| Why it is needed | Today's value is a rule of thumb (A7). The maker's own figure, read on the rotor side, is 33 times larger on the X4 and 6 times on the X6 [E from M]. Third-party fits of the tail servo's 7.4 V version give 120 times today's value [T, file]. |
| What it would mean, on the stored references [E] | With the high value the knee's RMS torque rises from 0.52 to 0.79 of its planning limit in the nominal walk, from 0.53 to 1.08 in the brisk walk and from 0.56 to 1.41 at the animal pace. Its peak there is 22.2 N·m against the 21.0 N·m limit. My arithmetic: (J_high − J_today) × the reference's own acceleration, added to the reference's own torque (`refs/nominal.npz`, `brisk.npz`, `fr05.npz`). Prescribed motion, no closed loop. |
| What stays nominal | The references' torques and the controller's feedforward, as with the mass (`designlab/stage1/env.py:126-133`). |

### 2g. Left out, and why

| Left out | Why |
|---|---|
| A torque-scale (motor strength) draw | A2 already takes 0.70 of the maker's peak. The one third-party shortfall found is of that size (a sibling's torque constant 33 % low; summary). The env already draws stiffness × 0.6 to 1.2 (`r5env.py:136`). |
| Heat | No thermal model. The judge keeps its RMS limit [A1]. Section 7. |
| A learned actuator network | Needs torque data from the real drive (R3 §2c). |
| A voltage-loop law for the box servos | Their firmware scaling is known for one 7.4 V unit only. A gain scale and the update rate stand in. |
| The 12 V regulator's droop | One 4.5 A regulator against 7.1 A of summed stall current (BENCH:745-746). Outside the legs. |
| Dropped or late bus frames | The soak test measures them (BENCH:342-345). No figure to draw from. |

---

## 3. The parameter table

"Today" is what the simulation uses now [S]. "Nominal" is the value gate 1's condition 1 would be judged at. Every draw is once per run. Where the source column says "no source", the nominal is a placeholder: **unsourced; the bench measures it**.

Classes: X4 = `myactuator_rmd_x4_p36_36_e_l` (10 drives), X6 = `myactuator_rmd_x6_p20_60_e` (2), neck = `robotis_xm540_w270_t`, tail = `feetech_sts3215_c018`.

| Parameter | Class | Today | Nominal | Range per run | Distribution | Source and kind | Bench measurement that replaces it |
|---|---|---|---|---|---|---|---|
| `fric_c` dry friction, N·m | X4 | 0.15 [A18] | 0.2 | 0.1 to 0.6 | uniform | nominal: backdrive torque [M] RES:85. Range [E]; the V4 sibling's 1.14 [M] RES:52 is outside it | sheet B step 1 (BENCH:564-567) |
| | X6 | 0.15 | 1.6 | 0.2 to 2.4 | uniform | nominal [M] RES:154. Low end: the X6-50-L's 0.2 [M] RES:186. High end [E] | sheet B step 1 |
| | neck | 0.15 | 0.12 | 0.03 to 0.30 | uniform | no source for this model. Nominal: another model's fit [T, file: MX-106, 0.122]. Range [E] | sheet B steps 1 and 3 |
| | tail | 0.15 | 0.05 | 0.03 to 0.15 | uniform | the 7.4 V version's fit, 0.052 [T, file]. Top = A18. No source for the 12 V part | sheet B steps 1 and 3 |
| `fric_v` viscous, N·m·s/rad | X4 | 0.2 [A18] | 0.10 | 0 to 0.25 | uniform | no source. Nominal [E], set so that the total at the maker's rated point equals today's (1.9 N·m) | sheet B step 2 (BENCH:568-572); it stops at 4 rad/s, the model uses 8.7 |
| | X6 | 0.2 | 0.05 | 0 to 0.10 | uniform | no source. Nominal [E], same rule (3.4 against today's 3.35 N·m) | sheet B step 2 |
| | neck | 0.2 | 0.05 | 0.01 to 0.20 | uniform | another model's fit, 0.051 [T, file]. Top = A18 | sheet B step 3 |
| | tail | 0.2 | 0.06 | 0.03 to 0.20 | uniform | the 7.4 V version's fit, 0.059 [T, file]. Top = A18 | sheet B step 3 |
| `fric_load` share of motor torque | X4 | 0 | 0.08 | 0 to 0.15 | uniform | no source. [E]. Upper bound 0.30 [E from M] | new N5 |
| | X6 | 0 | 0.05 | 0 to 0.10 | uniform | no source. [E]. Upper bound 0.20 [E from M] | new N5 |
| | neck | 0 | 0.20 | 0 to 0.30 | uniform | other models' fits: 0.198, 0.311 [T, file] | new N5 |
| | tail | 0 | 0 | 0 to 0.25 | uniform | no source (the third-party fit has no load term) | new N5 |
| `fric_joint` multiplier per joint | all | 1 (the env draws one factor 0.5 to 1.5 for all hinges, `r5env.py:139-140`) | 1 | 0.8 to 1.25 | uniform, per joint | no source. [E]: left and right differ | sheet B on each unit; step 4 on the leg |
| `play_deg` total play, degrees | X4 | 0 | 0.25 | 0 to 0.5 | uniform, per joint | nominal: ≤ 15 arcmin [M] RES:75. Top [E]: twice that. [T, summary]: a sibling measured 0.33° against 0.083° advertised | new N1 |
| | X6 | 0 | 0.25 | 0 to 0.5 | uniform, per joint | [M] RES:143 | new N1 |
| | neck | 0 | 0.25 | 0 to 0.5 | uniform | [M] RES:1502. [T, summary]: 0.11 to 0.29° on a family member | new N1 |
| | tail | 0 | 0.5 | 0 to 1.5 | uniform | nominal [M] RES:1566. Top [T, summary]: 1.3° loaded | new N1 |
| `ts_corner` speed to which the peak is available at 24 V, rad/s | X4 | 8.69 [A] | 5.76 | 5.76 to 8.69 | uniform | nominal [E from M]: the line through the maker's rated and no-load points (RES:63-67). Top: today's rule | new N2 |
| | X6 | 8.01 [A] | 4.16 | 3.36 to 8.01 | uniform | nominal [E from M]: the 48 V points, speeds × 24/48 (RES:131-135). Low [E]: copper-loss slope Kt² / (3 R). No 24 V data exist | new N2 |
| | neck, tail | 1.885, 2.827 [A] | same | fixed | — | today's lines are inside the maker's stall-to-no-load lines [E from M] | none planned |
| `armature` reflected inertia, kg·m² | X4 | 0.00324 [A7] | 0.00324 | 0.00324 to 0.106 | log-uniform | low [A7]. High [E from M]: 0.000082 × 36² (RES:85), if rotor-side | the maker's answer; new N5 |
| | X6 | 0.0043 [A7] | 0.0043 | 0.0043 to 0.0254 | log-uniform | high [E from M]: 0.66 kg·cm² × 19.612² (RES:154) | the maker's answer; new N5 |
| | neck | 0.000201 [A] | same | 0.0002 to 0.026 | log-uniform | no source for this model. High: other models [T]: 0.0058 (summary), 0.026 (file) | new N5 |
| | tail | 0.000226 [A] | same | 0.0002 to 0.028 | log-uniform | high: the 7.4 V version's fit, 0.0284 [T, file] | new N5 |
| `cmd_lag_ms` | X4 | 0 | 0.5 | 0 to 4 | uniform | nominal [E from M]: L/R 0.45 ms. Top [E]. No source for the response | new N4 |
| | X6 | 0 | 1.2 | 0 to 4 | uniform | nominal [E from M]: L/R 1.24 ms | new N4 |
| | neck, tail | 0 | 0 | 0 to 20 | uniform | no source (LAT:346-347) | BENCH:347-370 |
| `cmd_hold` box servos' update | neck, tail | every step | every 5th step, phase 0 | phase 0 to 4 | integer uniform | the bus plan [A] LAT:335-336 | BENCH:347-370 |
| `kp_scale` | neck | 1 | 1 | 0.3 to 1.2 | uniform | no source (register gains) | sheet A steps 1 to 3 (BENCH:519-535) |
| | tail | 1 | 1 | 0.3 to 1.2 | uniform | low end [E from T (file) and M]: 15.6 N·m/rad at the default register | sheet A |
| `kv_scale` | neck | 1 | 1 | 0.5 to 1.2 | uniform | [E from M]: stall / no-load = 3.4 N·m·s/rad, 0.81 of the model's | sheet A step 4 |
| | tail | 1 | 1 | 0.2 to 1.2 | uniform | [E from M]: 0.62 N·m·s/rad, 0.23 of the model's | sheet A step 4 |
| `gain_cap` | X4, X6 | none | none | a setting, not drawn: none, or kp ≤ 500 and kv ≤ 5 | — | a sibling's manual [M via search extract] LAT:442; units not confirmed | sheet A step 1; the maker's protocol document |
| `enc_offset` rad | all | 0 | 0 | −0.015 to +0.015 | uniform, per joint | a third-party default [T, code: R3 §2b]. The X4's accuracy is 0.007 rad [M] RES:85 | new N6 |
| `sense_steps` | all | 0 | 0 | 0 to 2 | integer uniform | the budget's S [A] LAT:196-199 | bench S (BENCH:93-94) |
| `bus_v0` V | bus | 24.0 fixed | 22.2 | 21.0 to 25.2 | uniform | nominal: the pack's nominal voltage [M] `designlab/data/parts.json:207`. Range [A11] | a pack log |
| `bus_r` Ω | bus | 0 | 0.05 | 0.02 to 0.15 | uniform | no source (the pack's entry has none, `parts.json:193-216`). [E]. For scale: 5 m of tether cable is 0.13 Ω [E from MAT:233] | new N3 |
| `latency_steps` | all | training 0 to 12; judged 0 to 5 | unchanged | unchanged | integer uniform | `r5env.py:137`; `bar.py:134-136` | bench T |

The friction sets, as (T_c, b_v, μ_L):

| Class | Low | Nominal | High | Friction torque at the maker's rated point: low / nominal / high; today; my bound [E] |
|---|---|---|---|---|
| X4 | 0.1, 0, 0 | 0.2, 0.10, 0.08 | 0.6, 0.25, 0.15 | 0.10 / 1.91 / 4.35; 1.89; 3.16 N·m |
| X6 | 0.2, 0, 0 | 1.6, 0.05, 0.05 | 2.4, 0.10, 0.10 | 0.20 / 3.40 / 6.00; 3.35; 4.05 N·m |
| neck | 0.03, 0.01, 0 | 0.12, 0.05, 0.20 | 0.30, 0.20, 0.30 | no rated point |
| tail | 0.03, 0.03, 0 | 0.05, 0.06, 0 | 0.15, 0.20, 0.25 | no thermal rated point |

The high sets are above my bound on purpose: the bound assumes the maker's efficiency was taken at the rated point, which the record does not say.

New bench measurements named above (none is on the sheet today):

| Id | Measurement | Pins |
|---|---|---|
| N1 | Lost motion per drive type: a lever and a dial indicator, the load reversed, read at the lever and on both encoders; loaded and unloaded. On the leg later, with the push rods. | `play_deg` |
| N2 | Peak torque held at 2, 4, 6 and 8 rad/s against a brake or a second drive, at 21, 24 and 25.2 V. A2's dyno curves, reduced. | `ts_corner` |
| N3 | Bus voltage at the drives before and during a known load step, at two currents. | `bus_r` |
| N4 | Torque current after a feedforward-torque step, in 0.2 ms bins; a small sine sweep from 1 to 50 Hz. | `cmd_lag_ms`; whether a torque lag must be modelled at all |
| N5 | A free-shaft chirp read by the encoders (inertia, viscous, dry friction); a pendulum with two or three masses (the load term). The two methods R3 §2a and §2c describe. | `armature`, `fric_load` |
| N6 | Zero repeatability against a jig; from the maker's document, which encoder the reply reports and which one closes the loop. | `enc_offset`; the assumption of 2b |

---

## 4. How it is drawn

| Question | Proposal |
|---|---|
| When | Once per run (episode), at reset. |
| Per class | `fric_c`, `fric_v`, `fric_load`, `ts_corner`, `armature`, `cmd_lag_ms`, `kp_scale`, `kv_scale`. |
| Per joint | `fric_joint`, `play_deg`, `enc_offset`. Left and right legs differ. |
| Per run, global | `bus_v0`, `bus_r`, `sense_steps`, the box servos' hold phase, and the existing `latency_steps`. |
| Fixed within a run | Every parameter. Only the layer's states move: the bus voltage, the load filter, the lagged command, the reading buffer. |
| Seed, training | Its own generator: `numpy.random.SeedSequence([episode seed, 2])`. The env already uses `[episode, 0]` for the start and `[episode, 1]` for the sensors (`designlab/stage1/env.py:119-123`; `r5env.py:486-487`). The env's own generator is not touched, so the task, the factors and the push of an episode are the same with the option on or off. |
| Draw order | Fixed and written as a rule text. The table's sha256 goes into the env's description and the row's `describe()`. |
| Judged panels | Draw nothing. Every parameter is at a named value; `fric_joint` is 1 and `enc_offset` is 0. A record is then repeatable from the bar's own 20 seeds per block (`STAGE2_75_PLAN.md:621`), as today. |
| A drawn panel | Optional and reported only: the servo seed is `SeedSequence([panel seed, 2])`. |
| The env's old draws | With the option on, `joint_damping` and `joint_friction` must be (1, 1): the layer owns friction. The calls stay, so the stream does not shift. Stiffness × 0.6 to 1.2, mass, floor friction and noise stay as they are. |

**Named settings.**

| Name | Values | Use |
|---|---|---|
| `today` | every part neutral: A18's friction on the joints, play 0, today's line, 24.0 V and 0 Ω, no lag, no offset, A7's inertia | the identity test (section 5); a link to today's records |
| `N` nominal | the Nominal column of section 3 | condition 1 |
| `F-`, `F+` | every class at its low, or high, friction set | condition 2 |
| `P0`, `P+` | play 0; or 0.5° on X4, X6 and the neck and 1.5° on the tail | condition 2 |
| `V-` | `bus_v0` 21.0 V with `bus_r` 0.15 Ω | condition 2 |
| `J+` | `armature` at the top of each class's range, kv by the plant's rule | condition 2 |
| `M-`, `M+` | trunk mass and inertia × 0.9, × 1.1 (the env's own factor, `r5env.py:135`) | condition 2 |
| `G5` | `gain_cap` on | reported until the maker's document is read |
| `T+` | `ts_corner` at today's values | reported: the easy end |
| `W1`, `W2` | `F+ V- J+ M+` together; `F- P+ M-` together | reported |

Each corner moves one factor and leaves the rest at `N`.

**The draft's corners, confirmed or corrected.**

| Draft (`BUY_GATE_DRAFT.md:41-44`) | Against the table | Proposed |
|---|---|---|
| friction at half and at one and a half times nominal | The open range is not ±50 % of one number. On the X4 the maker's two generations differ 5.7-fold; on the X6 the two candidate parts differ 8-fold; the load term has no source at all. | `F-` and `F+`: each class's low and high set |
| gear play at 0 and at 1° | 1° is four times the leg drives' maker limit and below a third-party loaded figure for the tail (1.3°). | `P0`; `P+` at 0.5° (legs, neck) and 1.5° (tail) |
| trunk mass at 0.9 and 1.1 | Not a servo parameter; it is the env's range. | confirmed: `M-`, `M+` |
| — | The bus and the rotor inertia are not in the draft. | add `V-` and `J+` |
| 2 × 2 × 2 corners at two delays: 16 records | | 8 one-factor corners at two delays: 16 records |

---

## 5. Where it attaches in the code

### 5.1 The hooks

One new module, `designlab/stage2/servo.py`, used as a context manager on a `plant.Plant`. It uses three hooks that exist today, each with a precedent:

| Hook | What the layer does there | Precedent |
|---|---|---|
| Model arrays | per-run arrays (damping, inertia, gains), per-step friction arrays; restored exactly on exit | `r5env.py:376-394`; `designlab/stage1/ledger.py:801-839` |
| `p.envelope` replaced | its `update()` runs before every physics step (`plant.py:687-692`): sag, the line, friction, the lag, the offset and the dead zone | the ledger swaps the envelope with an undo (`ledger.py:777-799`) |
| `p.step` as an instance attribute | after each step: the controller's own command back into `ctrl`, the reported angles patched in `sensordata` | `designlab/stage1/policy_rows.py:841-852` |

The layer **arms** at entry and **acts from the first physics step** after a reset. So the reset's forward pass, the first sensor reading and every tracker's copy of the friction are today's, on both sides.

| Side | How it enters |
|---|---|
| Training env | `R5Config.servo` (default `None`: off). The env leaves the layer first thing in `reset`, before it restores its own arrays (`r5env.py:431`), and enters it last, after the delay (`r5env.py:504-505`). `reset(options={"servo": ...})` gives a named setting instead of a draw, as `options["factors"]` does today. |
| Judging runner | A wrapper row `servo.Served(inner, setting)` in the layer's own module, in the form of `bar.Delayed` (`bar.py:880-918`): `with served(p, setting): return inner.episode(...)`. Registered through `panels.register_rows` (`panels.py:865-872`) under controller names such as `R5sN`, `R5sFhi`; `bar.rows` then wraps it in the delay like any registered row (`bar.py:921-929`). `bar.py` is not edited. |
| Same model on both sides | Both call the one context manager with the one parameter object. A gate in G8a's form (`g8a.py:12-19`) proves it: the env with the zero action and a setting against the `R0s<setting>` row, the sha256 of the state after every physics step. |

### 5.2 Traps, and how the design meets them

| Trap | Answer |
|---|---|
| 1. On the judging side the tracker is built inside the episode and copies the plant's friction then (`panels.py:844`; `control.py:1204-1205`) | The layer writes its arrays at the first physics step, after that copy. The env's trackers are built at construction (`r5env.py:239-240`) and its reset check (`r5env.py:438-440`) still holds. |
| 2. `servo_delay` refuses a plant whose `command` or `reset` is replaced (`speedfit.py:336-338`) | The layer replaces neither. Either order works. |
| 3. The judge reads the envelope and the force ranges (`control.py:2304`, `2296-2365`) | The replacement keeps `moment`, `peak`, `rated`, `no_load`, `active`, `enabled`, `bind`, `restore`, `limits` and `describe`, with the nominal `peak` and `rated`. `p.drives` is not touched. Play never touches a force range. |
| 4. Friction is outside `actuator_force`, so the torque that overcomes it counts as load | Kept, and said in the rules text: gear friction is overcome by motor torque. It matters for the draft's condition 4 (the tail). |
| 5. A worker's plant is shared across rows (`panels.py:979-988`) | Exact undo on exit, tested as the ledger tests it (`ledger.py:833-837`). |
| 6. Snapshots hold MuJoCo's state and `ctrl` only (`control.py:1347-1388`) | The judged rows take none (`confirm` appears in neither `panels.py` nor `bar.py`). The layer is not offered to `control.run(confirm=True)` or to the ledger. |
| 7. New: on the judging side the estimator's model is the plant's live model (`estimator.py:247-252`; `control.py:1337`); the env gives it a nominal copy (`r5env.py:334`, `623`) | Concerns `M-` and `M+` only: the judging side's estimator would know the changed mass. Needs a ruling (section 8, choice 11). |
| 8. New: `policy_rows.digesting` removes `p.step` on exit (`policy_rows.py:841-852`) | Digests are taken through the recorder or the env's step hook, as G8a does. The layer restores whatever `step` it found. |
| 9. New: the env restores gains, damping and friction at the start of `reset` (`r5env.py:381-383`) | Hence "leave first, enter last" above. |

### 5.3 Files

| File | Change | In a record key? |
|---|---|---|
| `designlab/stage2/servo.py` | new: the table, the parameter object, the draw, the named settings, the layer, the wrapper row and its factories | only in the new rows' own keys, through their `sources()` |
| `designlab/stage2/g8s.py` | new: the env-against-row comparison, G8a's form | no |
| `designlab/stage2/r5env.py` | edit: the `servo` field, the two lines in `reset`, the check on the old friction draws. `describe()` leaves the key out when the field is `None` | no. No row imports it (`stage2/design/ROUND_C_DESIGN.md:465`); no JSON record names it (grep today). The session's task list shows another package editing this file (wave 2a): merge after it |
| `tests/test_stage2_servo.py`, `tests/test_stage2_g8s.py` | new | no |
| `designlab/stage1/*.py`, `designlab/*.py`, `designlab/stage2/bar.py`, `readings.py`, `setio.py`, `r5core.py`, `r5lib.py` | none | they are the hashed sources |
| the screened MJCF, the references, every record | none | — |

### 5.4 Why no existing record's key or content changes

1. The source hashes that records and the regress check hold are: the panels' (`panels.py:1572-1583`), the bar's (`bar.py:1677-1687`: `bar.py`, `readings.py`, `setio.py`, `speedfit.py`), the Stage 1 package's (`plant.py:262-275`: every `designlab/stage1/*.py`), the tuner's and the speed fit's (`designlab/stage2/regress.py:84`, `214-219`). No file in any of them is edited, and no file is added under `designlab/stage1/`.
2. A row added in its own module leaves the other rows' keys as they are: the panels' own rule says so (`panels.py:347-359`).
3. A served row has a new name, its own `describe()` (the setting and the table's sha256) and its own `sources()`. Its records are new records.
4. The model file on disk is not touched. The layer changes arrays in memory and restores them.
5. With `servo=None` the env runs today's code path and prints today's description.
6. No existing record is rerun or extended (`NEXT_SESSION_PROMPT.md:223-225`).

### 5.5 Tests that would prove it

| Id | Test | Proves |
|---|---|---|
| T1 | `panels.panel_sources()` and `bar.bar_sources()` give `c0027992…` and `45719d03…` ([P] today) after the change; `regress check` passes | no key moved |
| T2 | With `servo=None`: G8a's existing tests pass unchanged; `R5Config(...).describe()` equals today's dictionary | option off = today, bit for bit |
| T3 | With the setting `today`: the state digests of the R0 row and of the env equal those without the layer, at 0 and 6 ms | the layer's arithmetic adds nothing of its own |
| T4 | After exit: every model array equals a copy taken before; `p.envelope` is the old object, bound; `p.__dict__` holds no `step` | exact undo |
| T5 | Friction: the array written equals m_j (T_c + μ_L τ̄) for a set τ̄; on a one-joint model the break-away torque is T_c | part a |
| T6 | Play: `actuator_force` equals the dead-zone formula at random states (my probe's form); on a one-joint model the static deflection under a load τ is τ / kp + b | part b |
| T7 | Line and sag: the force range at set (ω, V) equals the formula, for example 7.4 N·m on the X4 at 8.69 rad/s and 22.2 V, and 0.4 N·m on the X6 at 8.01 rad/s and 21 V [E]; V after one step equals V0 − R P / V | part c |
| T8 | Lag: a step in the command gives 1 − exp(−n dt / T_r); the box servos' command changes every 5th step | part d |
| T9 | Offset: the servo rests at target − o; the reported angle is the true one + o; the buffer returns the reading of n steps before | part e |
| T10 | Inertia: the mass matrix's diagonal rises by the drawn difference; `mj_setConst` after the undo leaves the model equal bit for bit | part f |
| T11 | The judge's `rated` and `peak` are the nominal arrays under every setting | registered limits do not move |
| T12 | Same seed, same parameters; the env's task, factors and push are equal with the option on and off | repeatable draws |
| T13 | G8s: the env with the zero action against `R0s<setting>@<ms>ms`, for `N`, `F+`, `P+`, `V-`, `J+`, at 0 and 10 ms | training and judging run one model |

---

## 6. What it costs

| Quantity | Before | After | Basis |
|---|---|---|---|
| Layer arithmetic per physics step | 2.2 µs (today's `Envelope.update`, common case) | 28 µs | [P] `probe_layer_cost2.py`: the layer's numpy calls on stand-in arrays of this body's sizes, idle machine. Not measured in the env |
| Env steps per second, hard floor | 382 (builder) and 385 (reviewer), one env, load about 3, provisional [S] (`stage2/design/checkpoint/reviews/roundC__build_CE.md:11`; `roundC__review_CE.json:76`) | about 350 [E] | 260 µs per physics step all-in, plus 26 µs: about 10 % |
| Env steps per second, carpet | 307 and 302 [S], same lines | about 280 [E] | 326 µs plus 26 µs: about 8 % |
| Physics itself | 39 µs per step [S] (PLANT:1041) | unchanged in size: 14 friction rows, no new joint [P] | Unmeasured: whether larger friction and inertia change the solver's iteration count |
| Per run | — | two `mj_setConst` calls and a few array writes | the mass change already does the same |

Build effort, in work packages. Sizes are my estimates.

| Package | Content | Size | Needs |
|---|---|---|---|
| S1 | `servo.py`: table, parameters, draw, settings, layer, exact undo; tests T4 to T12 | about 600 lines of code and 500 of tests; a build, an independent review, a fix round | this note reviewed |
| S2 | Env option; `g8s.py`; tests T1 to T3, T13 | about 80 lines in `r5env.py`, 150 in `g8s.py`, 300 of tests | S1; wave 2a's edits to `r5env.py` merged |
| S3 | `Served` row, factories and names; the reported readings (steps below 20 V; share of steps a leg drive floats in its play); the corner panel as a bar task | about 250 lines and 250 of tests | S1. Policy rows need `s2rows.py`; R0 and R2 rows need nothing |
| S4 | Measure before use: `r5env.throughput` with the layer; the R0 and R2 rows at `N` and at each corner | runs only; rules first; listed as development episodes | S2, S3 |
| S5 | Rules text for gate 1's servo conditions; draft decision-log rows for each unsourced nominal; the buy gate edits | text | the lead |
| S6 | Bench sheet additions N1 to N6 | text | — |

---

## 7. What this model still cannot show

Gate 1 on this model is a harsher simulation. It is not proof.

- **Heat.** No winding temperature and no limit on how long a peak lasts. The judge's RMS limit is 0.57 of the maker's rating, unverified (A1).
- **Structure.** No flex in links, mounts or push rods. The ankle's cranks are unit ratios (`drive.py:5-17`).
- **Gear dynamics.** No crossing time, knock or rattle in the play; no stick-slip; no cogging or torque ripple.
- **The drive's inside.** No sampling or filtering of its own feedback; no torque bandwidth on the feedback path (a third party measured about 10 Hz on an older, larger sibling; summary only); no velocity estimation.
- **Faults.** No under-voltage cut-out (steps below 20 V are only counted), no over-current or over-temperature trip, no time-out, no lost or late frame.
- **Braking.** The envelope gives full peak torque when braking and takes no account of the bus rising.
- **The neck and tail servos' own control law**, and the droop of their shared regulator.
- **The feet.** Floor compliance, tread grip, carpet (gate 1 is hard floors only, P40).
- **Cables.** No tether drag, no harness stiffness across joints.
- **Sensors beyond the joints.** IMU mounting error, vibration, temperature drift.
- **Change over time.** Wear, warm-up friction, the pack's state across a session.
- **Parts.** Which X6 ships (RES:154). Units that differ more than the per-joint factor.
- **Every range marked "no source".** A pass inside a guessed range says the policy tolerates that range. It does not say the real part is inside it.

---

## 8. Open choices for the lead

Each starts with my recommendation and one line of reason.

1. **Rotor inertia: ask the maker which side of the gear its inertia figures are on, before gate 1 is judged. Until then keep A7 as nominal and judge `J+`.** The range is 6 to 33-fold on the leg drives and the high end changes the knee's load by more than any other parameter here (2f). If the maker confirms the rotor side, the references and the Stage 0 screen need the real value, which is outside this note.
2. **Gain limits: read the maker's V4.1 motion-mode document for the kp and kd ranges and units. If the caps are real, make `G5` the nominal; until then report it.** Costs nothing; with the caps the hip and knee are far from critically damped [P].
3. **Nominal torque-speed line: the line through the maker's rated and no-load points.** It is the only line the catalogue supports, and the X6 has no 24 V data at all. Today's line stays as the easy end `T+`.
4. **Backlash: the run-time dead zone now. Build the gap joint only if `P+` changes the counts against `P0`.** It keeps every key and today's model, and the gap joint's cost and stability are unmeasured. Condition 2 measures the sensitivity that would justify it.
5. **Friction's load term: on, for all classes.** It is the main candidate for "friction ten times the model's" under load and costs six array operations per step.
6. **Feedforward: keeps today's friction values under every setting.** At gate 1 nobody knows the real friction, so the controller cannot either; it is also the env's existing rule.
7. **kv under a drawn inertia: recompute it by the plant's rule.** A build would tune damping on the real drive; keeping kv would test a tuning mistake, not the servo. `G5` shows the case where the drive cannot reach it.
8. **Nominal bus: 22.2 V with 0.05 Ω.** The pack's nominal voltage is nearer the middle of a run than 24 V; the resistance is a placeholder until N3.
9. **Judged corners: the eight one-factor corners on the grippy floor, 4 and 10 ms; the slippery floor, `W1`, `W2`, `G5` and `T+` reported.** That is the draft's own count of 16 records. Note the rule's weight: 320 runs per policy must all stay up; at a true stay-up rate of 99.9 % per run that happens 73 % of the time [E].
10. **Steps below 20 V: report and count, do not fail a run on them yet.** `bus_r` has no source; after N3 it can become a limit.
11. **Mass corners on the judging side: give the policy row's estimator a nominal model copy when `s2rows.py` is built; until then run `M-` and `M+` on R0 rows only.** Otherwise training and judging differ in what the estimator knows (trap 7).
12. **Critic inputs: do not add the servo draw to the critic's observation now.** It would change `r5core.py`'s layout while other packages build on it; revisit if training under the draw converges slowly.
13. **Encoder offset: patch the reading after each step, and accept that the reset's own reading is exact.** The alternative, turning each child body by the offset [P: every hinge is alone in its body at the body's origin], is cleaner at reset but edits the model's geometry per run.
14. **The 60 cm body: its own table before its gate 1, same code.** Its AK45-36, X4-P12-10 and STS3250 have their own maker figures (R2 §6) and no table here.
15. **Gate 2's wording: add N1 to N6 to the bench sheet, or change `BUY_GATE_DRAFT.md:59-60`.** The draft says the bench measures gear play; no sheet does.
16. **Condition 4 (the tail): read it at `N` and at `F+`.** The tail's load is mostly joint friction (A18), and the third-party figures for its 7.4 V version are a third of today's values.
17. **The box servos' nominal stiffness: keep the model's until sheet A is run; the draw reaches down to 0.3.** No source decides it; the one third-party figure is for another voltage version at its default register.

---

## 9. Proposed text for the buy gate draft

**Replacement for the bullet at `BUY_GATE_DRAFT.md:30-31`:**

> - **A servo model, drawn at random per run** (`stage2/design/servo_model/DESIGN.md`). Per drive type: joint friction (dry, viscous and a part that grows with load); gear play, as a dead zone in the drive's own loop; the torque limit against speed and bus voltage, with the voltage sagging under the legs' current; a short lag on the command; the encoder's zero error and age; the rotor's reflected inertia. Where the maker gives a value, that value is the nominal. Where nothing has a source, the range is wide and is labelled "unsourced; the bench measures it". Today's model has ideal servos and one unsourced friction value (`DECISION_LOG.md` A18). No value in the model is measured on this build's hardware.

**Replacement for condition 2 at `BUY_GATE_DRAFT.md:41-44`:**

> 2. **Robust to the servo (*proposed*).** The 0.30 m/s walk stays up in 20 of 20 on the grippy floor, at 4 ms and at 10 ms, at each of eight named corners of the servo model. A corner moves one factor and leaves the rest nominal: friction low and high (`F-`, `F+`: each drive type's low and high set, not a multiple of one number); gear play none and high (`P0`, `P+`: 0°, and 0.5° on the leg drives and the neck with 1.5° on the tail); bus low (`V-`: 21.0 V with 0.15 Ω in series); rotor inertia high (`J+`: the maker's figure read on the rotor side); trunk mass 0.9 and 1.1 (`M-`, `M+`). Reported beside it, with no verdict: the slippery floor, the two combined corners (`W1`, `W2`), the drives' gain limits (`G5`), today's torque-speed line (`T+`), and the number of steps the bus spent below the drives' 20 V supply limit. The corners' values are the table of `stage2/design/servo_model/DESIGN.md`, at the hash recorded with the rows. Values without a source are labelled so there; the bench (gate 2) replaces them, and the corners are then set again.
