<!-- Working paper, 16 Nov 2026: one of three readers' reports gathered for the design of the servo model of gate 1 (BUY_GATE_DRAFT.md). Not reviewed. Nothing here is a rule or a result. Outside sources are quoted as found and were not checked a second time. -->

# Reader 3: outside sources for a realistic servo model (75 cm build, `75-M-H-10`)

## 0. Scope, marks, limits

- **Three kinds of statement, kept apart.** Prospective design: section 5 only. Simulation evidence: rows labelled "study's model". Measured on hardware: the study has none. Every hardware number below is a maker's claim or a third party's measurement on its own unit.
- **Marks.** M = maker's catalogue value. T = measured or identified by a third party. A = assumption of this study. E = my estimate. (C) = I read the code or file today. (S) = web-search summary only; the page itself was not read.
- **Network limit.** Only `github.com` and `raw.githubusercontent.com` could be fetched. Blocked by the session's network policy: `arxiv.org`, `ar5iv.labs.arxiv.org`, `emanual.robotis.com`, `www.myactuator.com`, `www.feetechrc.com`, `bam.readthedocs.io`, `mujoco.readthedocs.io`, `huggingface.co`. These hosts would have to be allowed to read the papers and datasheets directly.
- **Consequence for M values.** I could not re-read any maker page. M values are cited from the study's own catalogue of 7 Nov 2026, which holds the maker's URL per record.
- **Deviation from the standing rules.** Four JSON-reading Python one-offs ran directly with the venv python (`-I`), not through `tools/lowprio.sh`. No model was loaded, no simulation or training ran, no tracked file changed.
- **Abbreviations** (paths relative to `/home/user/mesozoic-labs/hardware/trex_form_study`):
  - RES = `designlab/data/research/actuators_2026-11-07.json`
  - BS = `stage2/jetson75/BENCH_SHEET.md`
  - LB = `stage2/jetson75/LATENCY_AND_BUS.md`
  - PLANT = `stage1/runs/2026-11-08/75-M-H-10/plant.json`
  - MD = `/home/user/pollen-robotics/microduck_rl`
  - MJ = `/home/user/mujocolab/mjlab`
- **Downloads** (untrusted data, read only):
  - `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_design/dl_bam`
  - `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_design/dl_misc`
- **Build's actuators** (`stage2/jetson75/BUDGET_75.md:380, 418-420`; BS:178-181): 10 × RMD-X4-P36-36-E-L, 2 × RMD-X6-P20-60-E (hip pitch), 1 × DYNAMIXEL XM540-W270-T (neck), 1 × Feetech STS3215 C018 12 V (tail).

## 1. Actuator tables

### 1.1 MyActuator RMD-X4-P36-36-E-L (V4.1): every leg joint except hip pitch

| Quantity | Value | Mark | Source |
|---|---|---|---|
| Rated torque and speed | 10.5 N·m at 83 rpm, 24 V. "Rated" is a duty cycle: 90° swings with 5 s holds | M | RES:63-66 |
| Peak; stall | 30 N·m at 10 rpm; 36 N·m at 21.2 A rms | M | RES:65, 74, 85 |
| No-load speed | 111 rpm at 24 V | M | RES:67 |
| Voltage | 20–55 V, nominal 24 V | M | RES:68-69 |
| Gear | planetary, 36:1; number of stages: no source found | M | RES:70 |
| Backlash | ≤15 arcmin (manual); 10 arcmin on the US store page of the V4 sibling | M | RES:75; RES:47-48 |
| Torque constant (output) | 1.8 N·m/A rms = 1.3 N·m/A peak | M | RES:71, 85 |
| Resistance, inductance, poles | 0.4 Ω line-to-line, 0.18 mH, 11 pole pairs | M | RES:85 |
| Back-EMF | 4.3 Vdc/krpm, motor side | M | RES:85 |
| Rotor inertia | "0.000082 kg·m²"; which side of the gear is not stated in the record | M | RES:85 |
| Backdrive torque; efficiency | 0.2 N·m; 64.7 % | M | RES:85 |
| Encoders | 17-bit input, 18-bit output; absolute accuracy 0.4°, repeatability 0.3° | M | RES:76, 85 |
| Thermal | equilibrium after 90 min at 7.5–8.5 N·m; 10.5 N·m reaches 100 °C in 5.4–12.4 min; at 0 rpm 21 N·m lasts 44 s, 31.5 N·m 7 s, 34 N·m 2 s | M | RES:64 |
| CAN | 1 Mbit/s; command 0x140+ID, reply 0x240+ID; "motion mode" 0x400+ID (position, velocity, kp, kd, feed-forward torque). A sibling's manual gives KP in 12 bits for 0–500 and KD in 9 bits for 0–5 | M | LB:441-442 |
| Loop rates | none found for V4 or V4.1. Earlier MC-series drivers: current 15 kHz, speed 5 kHz, position 1 kHz (older manual 500 Hz) | M, other generation | LB:443 |
| Friction terms, torque bandwidth, torque accuracy, response delay | no published figure found, maker or third party | | |
| Third party, older and larger sibling only (RMD X8 V2) | torque constant 1.393 N·m/A measured against 2.09 quoted; backlash 0.3336° measured against 0.083° advertised; output torque bandwidth 10.32 Hz by sine sweep | T (S) | https://arxiv.org/abs/2605.24127 |
| Study's model | kp 598.5 N·m/rad (each ankle motor 299.25), kv 2.08–24.48; sustained 5.985 and peak 21.0 N·m; usable speed 8.69 and no-load 11.62 rad/s at 24 V; armature 0.00324 kg·m² from a rule | A | BS:501-506; PLANT `drives`; `designlab/data/build_catalog.py:14-15`; `DECISION_LOG.md:106-107` |

### 1.2 MyActuator RMD-X6-P20-60-E (V4): hip pitch

| Quantity | Value | Mark | Source |
|---|---|---|---|
| Rated torque and speed | 20 N·m at 153 rpm, 48 V, 320 W; no thermal condition stated | M | RES:131-134 |
| Peak | 60 N·m | M | RES:133 |
| No-load speed | 176 rpm at 48 V | M | RES:135 |
| Voltage | 20–55 V, nominal 48 V. The build runs it on a 6S bus of 21.0–25.2 V | M; A | RES:136-137; `stage2/jetson75/BUDGET_75.md:200` |
| Gear | planetary, 19.612:1; number of stages: no source found | M | RES:138 |
| Backlash | ≤15 arcmin (manual and motor sheet); US store page says 10 | M | RES:143, 154; BS:330 |
| Torque constant (output) | 2.1 N·m/A rms | M | RES:139, 154 |
| Resistance, inductance, poles | 0.41 Ω line-to-line, 0.51 mH, 10 pole pairs | M | RES:154 |
| Back-EMF | 16 Vdc/krpm | M | RES:154 |
| Rotor inertia | 0.66 kg·cm²; side of the gear not stated | M | RES:154 |
| Backdrive torque; efficiency | 1.6 N·m; 72.7 % | M | RES:154 |
| Encoders | 17-bit and 17-bit | M | RES:144 |
| Thermal (stall table) | 30 N·m: +17 °C in 15 s; 40: +29 °C in 10 s; 50: +37 °C in 8 s; 60: +24 °C in 5 s | M | RES:132 |
| Which variant ships at $428 | open: the store page lists 0.78 kg, 50 N·m and 10 arcmin, which match the V4.1 X6-50-L | M, conflict | RES:154 |
| CAN, loop rates | as 1.1 | M | LB:441-443 |
| Friction terms, bandwidth, torque accuracy, delay | no published figure found | | |
| Internal consistency | at the 176 rpm no-load speed, 16 Vdc/krpm gives about 55 V of back-EMF, above the 48 V bus; the same check on the X4-36-L closes (17 V at 24 V). One X6-60 figure uses another convention or is wrong | E | from RES:135, 138, 154 and RES:67, 70, 85 |
| Study's model | kp 1140 N·m/rad, kv 30.78; sustained 11.4, peak 42.0 N·m; usable 8.01, no-load 9.22 rad/s at 24 V; armature 0.00430 kg·m² from a rule | A | BS:502; PLANT `drives` |

### 1.3 ROBOTIS DYNAMIXEL XM540-W270-T: neck

| Quantity | Value | Mark | Source |
|---|---|---|---|
| Stall torque | 10.0 N·m at 11.1 V; 10.6 N·m at 12 V (4.4 A); 12.9 N·m at 14.8 V (5.5 A). No continuous rating is published | M | RES:1491-1492, 1511 |
| No-load speed | 28 / 30 / 37 rpm at 11.1 / 12 / 14.8 V | M | RES:1494, 1511 |
| Voltage | 10.0–14.8 V | M | RES:1496 |
| Gear ratio | 272.5:1; gear type not in the record | M | RES:1497 |
| Backlash | 15 arcmin | M | RES:1502 |
| Torque constant | 2.409 N·m/A, which is stall torque over stall current | M, derived | RES:1498 |
| Resistance, rotor inertia | not published | | RES:1499 |
| Position sensor | contactless absolute encoder, 12 bit over 360° | M (S) | https://emanual.robotis.com/docs/en/dxl/x/xm540-w270 |
| Bus | TTL half duplex, 9.6 kbit/s–4.5 Mbit/s; factory 57,600 bit/s; reply delay 500 µs by default, settable to 0 | M | RES:1503; LB:454-455 |
| Position-loop gains and scaling, loop period | not retrieved (host blocked) | | |
| Third party, this model | none found | | |
| Third party, XH540-V150 (same family, other motor and ratio) | kP 5.0, kD 0.2, max torque 4.8 N·m, static friction 0.05, viscous 0.009, backlash 0.002–0.005 rad, encoder error up to 0.02 rad, reflected inertia 0.0058 kg·m² | T (S) | https://arxiv.org/abs/2501.05204, Table VI |
| Third party, Dynamixel MX-106 (fit on a pendulum bench) | kt 2.21 N·m/A, R 2.03 Ω, armature 0.0260 kg·m², viscous 0.0507, Stribeck 0.122 N·m, load-friction (motor side) 0.198 | T (C) | https://raw.githubusercontent.com/Rhoban/bam/62bd8ce12154340be97e06f7f41a0ca8f116d967/bam/params/mx106/m6.json |
| Third party, Dynamixel MX-64 | armature 0.0123 kg·m², load-friction (motor side) 0.311 | T (C) | same commit, `bam/params/mx64/m6.json` |
| Study's model | kp 177.02 N·m/rad, kv 4.16; sustained 1.77, peak 3.11 N·m; usable 1.88, no-load 3.14 rad/s; armature 0.000201 kg·m² from a rule | A | BS:499; PLANT `drives` |

### 1.4 Feetech STS3215 C018, 12 V: tail

| Quantity | Value | Mark | Source |
|---|---|---|---|
| Stall | 2.94 N·m at 12 V, 2.7 A | M | RES:1556, 1565, 1578 |
| "Rated" | 0.981 N·m at 0.9 A; no duty definition | M | RES:1554-1555, 1564 |
| No-load speed | 45 rpm | M | RES:1558 |
| Voltage | 4–14 V | M | RES:1560 |
| Gear ratio | 1:345; gear type not in the record | M | RES:1561 |
| Backlash | ≤0.5° | M | RES:1566, 1578 |
| Torque constant; terminal resistance | 1.079 N·m/A (output); 1.0 Ω | M | RES:1562-1563 |
| Rotor inertia | not published | | |
| Bus | TTL, 38.4 kbit/s–1 Mbit/s, 12-bit magnetic encoder; reply-delay default disputed between two write-ups | M | RES:1567; LB:457, 460 |
| Third party, backlash | about 0.77° and 0.87° unloaded in two tests; 0.62° unloaded and 1.30° loaded in a third; position deviation about 2° at 10 kg·cm and 2.6° at 15 kg·cm | T (S) | https://robonine.com/smart-serial-servo-motor-comparison-robotics/ ; https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/ ; https://robonine.com/backlash-compensation-in-sts3215-servo-actuators/ |
| Third party, 7.4 V version, pendulum-bench fit (Coulomb-viscous model only) | kt 1.2116 N·m/A, R 2.676 Ω, armature 0.0284 kg·m², Coulomb 0.0524 N·m, viscous 0.0591 N·m·s/rad | T (C) | https://raw.githubusercontent.com/Rhoban/bam/mjlab_frictionloss/bam/params/feetech_sts3215_7_4V/m1.json |
| Third party, firmware model, 7.4 V version | P-only voltage loop: duty = clip(kp × 0.166 × error in rad, ±0.97), default kp 32 (the 0.166 "determined using an oscilloscope"); target slewed at 3400 steps/s = 5.22 rad/s | T (C) | https://raw.githubusercontent.com/Rhoban/bam/62bd8ce12154340be97e06f7f41a0ca8f116d967/bam/feetech/actuator.py lines 27-41, 80-92 |
| Third party, a small biped's model of the 7.4 V servo | joint damping 0.56, friction loss 0.068, armature 0.027; position kp 13.37 (earlier 17.12), force range ±3.23 N·m; backlash joint ±0.5° | T (C) | https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/xmls/open_duck_mini_v2.xml lines 45-54 |
| Study's model | kp 49.13 N·m/rad, kv 2.72; sustained 0.491, peak 0.862 N·m; usable 2.83, no-load 4.71 rad/s; armature 0.000226 kg·m² from a rule | A | BS:500; PLANT `drives` |
| Every joint, study's model | damping 0.2 N·m·s/rad and friction loss 0.15 N·m, unsourced (A18) | A | `designlab/mjcf.py:120`; `DECISION_LOG.md:123` |

## 2. How others model such actuators

### 2a. The small-biped project (`microduck_rl` at `0daab16`) and BAM

**Origin.** The actuator model is the BAM library, https://github.com/Rhoban/bam, branch `mjlab_frictionloss`, pinned at commit `62bd8ce` (MD `pyproject.toml:90`, `uv.lock:106`). Its paper is "Extended Friction Models for the Physics Simulation of Servo Actuators", ICRA 2025, https://arxiv.org/abs/2410.08650 (cited in BAM's README (C); paper not read).

**Servo law, voltage-controlled with a firmware P loop** (`bam/actuator.py` lines 251-269 and 293-296 (C)):
- duty = clip(kp_fw × error_gain × (q_target − q), −max_pwm, +max_pwm).
- Optional firmware current limit, applied first: duty is clamped to (kt·dq ∓ R·I_max)/V_in.
- V = V_in × duty.
- Motor torque tau_m = kt·V/R − kt²·dq/R. The second term is back-EMF.
- For the XL330: error_gain = (4096/2π)/(256×885), V_in 7.4 V, max_pwm 1.0 (MD `scripts/validate_bam_testbench.py:39-41`). Microduck sets kp_fw = 200 (MD `src/mjlab_microduck/robot/microduck_constants.py:167`).

**Voltage and sag** (MD `microduck_constants.py:169-171`; pinned `bam/mjlab.py` lines 85-88 and 603-605):
- V_in per environment is drawn once from U(6.5, 8.2) V.
- V_eff = max(6.0 V, V_in − g × Σ over joints of |tau|), with g drawn from U(0, 0.2) V per N·m.
- The branch head renames this to a resistance: V_drop = R_drop × Σ|tau|/kt.

**Friction budget, model M6** (`bam/model.py` lines 114-197; `docs/theory/models.rst` lines 148-163 (C)):
- s = exp(−|dq/dq_s|^alpha), the Stribeck factor.
- tau_fm = K_c + |K_m·tau_m − K_e·tau_e| + s × (K_cs + |K_ms·tau_m − K_es·tau_e| + Q) + K_v·|dq|.
- Q = K_eq·tau_e² if |tau_m| > |tau_e|, else K_mq·tau_m². `model.py:173` also requires tau_m and tau_e to have opposite signs; the mjlab adapter at the branch head has no such test.
- tau_e is the external load on the gearbox: gravity, Coriolis and constraint forces, minus the friction force itself.
- M1 is K_c + K_v·|dq| only. M2 adds Stribeck, M3 load, M4 both, M5 the motor-side and load-side split.

**How the friction is applied** (`bam/simulate.py` lines 49-79, pinned (C)):
- tau_stop = −((J/dt)·dq + tau_m + tau_e); friction = clip(tau_stop, −tau_fm, +tau_fm).
- In mjlab the velocity-independent part is written each step into `dof_frictionloss` and K_v into `dof_damping`; the solver does the clipping (branch-head `bam/mjlab.py` lines 599-700).
- The friction constraint is stiffened with solref (−5e4, −2e2), because MuJoCo Warp has no no-slip solver (line 248).

**XL330 parameters used** (pinned `bam/params/xl330/m6.json` (C)): kt 0.366, R 2.811 Ω, armature 0.00181 kg·m², K_c 0.00477, K_cs 0.00468, K_m 0.267, K_e 8.5e-6, K_ms 1.1e-5, K_es 0.0808, K_mq 0.00997, K_eq 0.00490, dq_s 2.89 rad/s, alpha 8.68, K_v 0.00536.

**Per-environment randomisation on top** (MD `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` (C)):

| Item | Range | Lines |
|---|---|---|
| Friction budget scale (Coulomb + Stribeck + load part only) | ×0.9–1.1 | 76; MD `src/mjlab_microduck/actuator/friction_dr_bam.py:37-47` |
| Armature | ×0.9–1.1 | 78 |
| kp and kd randomisation | switched off | 33-34 |
| Mass and inertia | ×0.95–1.05 | 73 |
| Centre of mass | ±3 mm, ramped to ±15 mm | 57, 863-866 |
| Encoder bias | ±0.015 rad | 85 |
| IMU mounting error | up to 6° | 84 |
| Foot friction | 0.7–1.3 | 377 |
| Pushes | ±0.3 m/s every 3–6 s | 79-80 |
| Command delay | 3–6 physics steps of 5 ms = 15–30 ms | MD `microduck_constants.py:173-174`; MJ `src/mjlab/tasks/velocity/velocity_env_cfg.py:446-451` |
| IMU delay | 0–1 control step (0–20 ms) | 564-570 |
| Joint velocity delay | fixed 1 control step (20 ms) | 611-613 |
| Observation noise | gyro ±0.03, gravity ±0.01, joint position ±0.001, joint velocity ±0.25 | 587-590 |

**Backlash.** Not part of BAM. Microduck adds it separately (MD `src/mjlab_microduck/robot/microduck/add_backlash.py:38-57, 63-72`; `robot_walk_backlash.xml:90`):
- An unactuated hinge in series with each servo joint, range ±1° (2° total).
- Joint settings: damping 0.01, armature 0.001, no friction loss; limit solref "0.01 1" (2 × the 5 ms step), solimp "0.95 0.999 0.0001 0.5 2".
- The encoder is on the output side, so the firmware loop and the policy both read servo angle + backlash angle; back-EMF and friction use the motor-side velocity (MD `friction_dr_bam.py:64-104`; MD `src/mjlab_microduck/tasks/backlash.py:1-27`).
- Backlash joints are excluded from the soft-limit penalty.

**Identification on a one-servo bench** (BAM docs at the branch head (C): `docs/identification/setup.rst`, `acquisition.rst` lines 27-56 and 208-225, `fitting.rst`; `bam/fit.py` lines 31-40, 190-198):
- Bench: a pendulum on the output shaft with known mass, arm mass and length; tau_e = m·g·l·sin(theta). Several mass and length combinations.
- Four trajectories of 6 s each: sin(t²); lift then torque off, so the arm drops (separates viscous friction from back-EMF damping); slow up-and-down; a multi-frequency sine.
- Each trajectory at several firmware P gains. XL330: 50, 100, 200, 300, 400. Its full data set is 240 recordings of 6 s.
- Logged at the firmware's rate: position, speed, load, input volts, goal, torque-enable. Resampled to 5 ms.
- Fit: CMA-ES through Optuna, 100,000 trials by default. Objective: mean absolute position error between the BAM simulator and the logs, with the firmware law simulated. One P gain held out for validation. All of M1 to M6 are fitted and compared.
- Fitted parameters: kt, R, armature, a bench angle offset and the friction parameters.
- Microduck's own bench: one XL330 with a 120 g arm (MD `src/mjlab_microduck/robot/testbench_constants.py:20-21, 48-56`). One script replays BAM's recordings in MuJoCo against the kernel (MD `scripts/validate_bam_testbench.py:22-24, 61-175`). Another runs the same exported policy in simulation and on the real servo at 50 Hz and compares traces (MD `scripts/testbench_sim2real.py:1-45`).
- Paper's validation, (S): four actuators (MX-64, MX-106, eRob80:50, eRob80:100) and 2R arms; position error "over twice as low" as Coulomb-viscous in one result.

### 2b. mjlab v1.3.0 (`19d6c06`)

- **Per-actuator-group fields** (MJ `src/mjlab/actuator/actuator.py:46-85`): `armature`; `frictionloss` (constant, load-independent dry friction as a constraint); `viscous_damping` (passive); `delay_min_lag`, `delay_max_lag`, `delay_hold_prob`, `delay_update_period`, `delay_per_env_phase`.
- **Delay semantics** (MJ `docs/source/actuators.rst:216-250`): the lag is drawn uniformly each step unless held, in whole physics steps. Only the command target is delayed; the control law sees fresh joint state. Observation delay is a separate mechanism on observation terms. `set_lags` at `actuator.py:304`.
- **Built-in classes**: position, velocity, motor and muscle actuators, integrated implicitly by the engine (`actuators.rst:62-67`).
- **Explicit classes** (`actuators.rst:69-76, 153-165`):
  - `IdealPdActuator`: tau = Kp·e + Kd·de, clamped to `effort_limit`.
  - `DcMotorActuator`: adds a linear torque-speed clip. Upper limit = min(effort_limit, saturation_effort × (1 − v/velocity_limit)); lower limit = max(−effort_limit, saturation_effort × (−1 − v/velocity_limit)) (MJ `src/mjlab/actuator/dc_actuator.py:137-163`).
  - `LearnedMlpActuator`: a TorchScript network on a history (default 3) of position errors and velocities, inside the DC clip (MJ `src/mjlab/actuator/learned_actuator.py:20-58`).
  - `XmlActuator`: wraps actuators already in the XML.
- **Not offered**: voltage, load-dependent or Stribeck friction, backlash (the word does not occur in `src` or `docs`).
- **Randomisation functions** (MJ `src/mjlab/envs/mdp/dr/`): `pd_gains` and `effort_limits` (`actuator.py:23, 118`); `joint_damping`, `joint_armature`, `joint_friction`, `encoder_bias` (`joint.py:26, 57, 88, 209`); `pseudo_inertia` (`body.py:422`); `geom_friction` (`geom.py:117`).
- **Default velocity task** (MJ `src/mjlab/tasks/velocity/velocity_env_cfg.py:223-271`): foot friction 0.3–1.2; encoder bias ±0.015 rad; base centre of mass ±0.025 to ±0.03 m; pushes every 1–3 s.
- **Reflected inertia from datasheet values**: a helper and a worked example with rotor inertia 0.489e-4 kg·m² (`actuators.rst:313-386`).

### 2c. Published approaches for geared and quasi-direct-drive legged robots

1. **Learned actuator network.**
   - Origin: Hwangbo et al. 2019, https://arxiv.org/abs/1901.08652. A network maps a history of position errors and velocities to joint torque, trained on torque-measured data from the real drive. Validation error 0.740 N·m against 3.55 N·m for an ideal model (S).
   - In code (C), ANYmal-C: an LSTM on (position error, velocity), with added base mass ±5 kg and floor friction 0.5–1.25. https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/anymal_c/anymal.py lines 52-77; `.../anymal_c/mixed_terrains/anymal_c_rough_config.py` lines 62-81; `.../envs/base/legged_robot_config.py` lines 121-128.
   - In code (C), Go1: a network on position error and velocity at three time steps, then torque × motor strength. Randomised: motor strength ×0.9–1.1, motor offset ±0.02 rad, a fixed lag of 6 simulation steps of 5 ms (30 ms, my reading of the code), floor friction 0.1–3.0, restitution 0–0.4, payload −1 to +3 kg, gravity ±1 m/s². https://raw.githubusercontent.com/Improbable-AI/walk-these-ways/master/scripts/train.py lines 30-65; `.../go1_gym/envs/base/legged_robot.py` lines 922-945, 1154, 1242-1250.
   - Needs torque data from the real drive, so it cannot be built before hardware.
2. **Analytic motor model with a torque-speed envelope and friction.**
   - Minitaur (Tan et al. 2018, https://arxiv.org/abs/1804.10332; code (C): https://raw.githubusercontent.com/bulletphysics/bullet3/master/examples/pybullet/gym/pybullet_envs/bullet/motor.py):
     - pwm = clip(−kp(q − q*) − kd·dq, ±1); V_net = clip(pwm·V − (Kt + b)·dq, ±50 V); I = V_net/R.
     - Torque from a measured current-to-torque table, which carries the saturation.
     - Randomised ranges are in section 3.
   - Berkeley Humanoid (planetary drives; code (C)):
     - PD torque through a DC-motor clip, minus friction = F_s·tanh(dq/v_act) + F_d·dq.
     - Per actuator: F_s 0.1–1.0 N·m, v_act 0.1 rad/s, F_d 0.005–0.02 N·m·s/rad; armature = rotor inertia (6.1e-6 to 1.5e-4 kg·m²) × 81.
     - Randomised: friction model ×0.9–1.1, armature ×1.0–1.05, default joint position ±0.05 rad, link masses ×0.9–1.1, torso ±1 kg, floor friction 0.2–1.25, pushes ±0.5 m/s every 10–15 s. No delay term in this configuration.
     - https://raw.githubusercontent.com/HybridRobotics/isaac_berkeley_humanoid/main/exts/berkeley_humanoid/berkeley_humanoid/actuators/actuator_pd.py lines 35-36; `.../assets/berkeley_humanoid.py` lines 12-75; `.../tasks/locomotion/velocity/velocity_env_cfg.py` lines 165-254.
     - Its paper says cogging torque is folded into the friction term (S), https://arxiv.org/abs/2407.21781.
   - PACE (https://arxiv.org/abs/2509.06342 (S); code (C): https://raw.githubusercontent.com/leggedrobotics/pace-sim2real/main/source/pace_sim2real/pace_sim2real/utils/pace_actuator.py lines 19-75, and `.../scripts/pace/data_collection.py` lines 17-19):
     - DC-motor model with an encoder bias and a command delay.
     - CMA-ES fits, per joint, armature, viscous damping, Coulomb friction and encoder bias, plus one delay, from about 20 s of in-air chirp (0.1–10 Hz) using joint encoders only.
     - The torque constant is left out as not identifiable from position data (S).
     - A follow-up on a quadruped with planetary reducers above 40:1 adds a Stribeck term for low-speed friction (S), https://doi.org/10.3390/machines14091031.
   - A bipedal robotic character (Grandia et al., RSS 2024, https://arxiv.org/abs/2501.05204 (S)):
     - PD torque law, static and viscous friction, reflected rotor inertia.
     - Backlash as an encoder offset: the motor sits on the side of the gap that the applied torque pushes toward.
     - Identified on a one-actuator bench that measures output torque; parameters randomised within their observed ranges.
     - Table VI, in the units the summary gave:

       | Parameter | Unitree A1 | Unitree Go1 | XH540-V150 |
       |---|---|---|---|
       | Static friction | 0.45 | 0.15 | 0.05 |
       | Viscous friction | 0.023 | 0.016 | 0.009 |
       | Backlash, rad | 0.005–0.015 | 0.002–0.005 | 0.002–0.005 |
       | Encoder error, rad | 0.02 | 0.02 | 0.02 |
       | Reflected inertia, kg·m² | 0.011 | 0.0043 | 0.0058 |
3. **Randomised strength, gains and delay with no explicit motor model.** Cassie, Unitree G1, Booster T1 and a 1 kHz humanoid configuration; all ranges are in section 3.

## 3. Randomisation ranges reported for sim-to-real walking

All (C) unless marked. Keys:
- TAN = https://raw.githubusercontent.com/bulletphysics/bullet3/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/env_randomizers/minitaur_env_randomizer_config.py lines 9-18
- CAS-B = https://raw.githubusercontent.com/HybridRobotics/cassie_rl_walking/master/rlenv/env_randomlizer.py lines 31-38, 54, 68, 73, 77-85, 100-106
- CAS-O = https://raw.githubusercontent.com/osudrl/apex/master/cassie/cassie.py lines 111-112, 152-160, 392
- BH = Berkeley Humanoid files in 2c
- T1 = https://raw.githubusercontent.com/BoosterRobotics/booster_gym/main/envs/T1.yaml lines 208-248
- HG = https://raw.githubusercontent.com/roboterax/humanoid-gym/main/humanoid/envs/custom/humanoid_config.py lines 149-160, and `.../humanoid_env.py` lines 194-196
- G1 = https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/g1/g1_config.py lines 28-35
- WTW and LG = files in 2c
- ODM = https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/common/randomize.py lines 44-91, and `.../open_duck_mini_v2/joystick.py` lines 51-65
- MDK = microduck, section 2a
- MJL = mjlab defaults, section 2b
- PLAN = this study's `STAGE2_PLAN.md:125-126, 360`

| Parameter | Ranges in the literature | This study's plan (A) |
|---|---|---|
| Joint Coulomb friction | 0–0.05 N·m (TAN, a 6 N·m-class direct drive); +0–2.0 N·m (T1); model ×0.9–1.1 (BH, MDK, ODM) | ×0.5–1.5 of 0.15 N·m (PLAN) |
| Joint damping | 0–0.05 N·m·s/rad (TAN); ×0.7–4.0 (CAS-B); ×0.3–5.0 (CAS-O); switched off under BAM (MDK) | ×0.5–1.5 of 0.2 N·m·s/rad (PLAN) |
| Motor strength | ×0.8–1.2 (TAN); ×0.9–1.1 (WTW); training 0.90–1.10 in RMA (S), https://arxiv.org/abs/2107.04034 | none stated in PLAN:125-126 |
| Stiffness and damping gains | ×0.7–1.3 (CAS-B); ×0.95–1.05 (T1); kp ×0.9–1.1 (ODM); switched off (MDK); defined but off: Kp ×0.8–1.3, Kd ×0.5–1.5 (WTW) | stiffness ×0.6–1.2 (PLAN) |
| Action delay | 0–40 ms (TAN); 0.45–28.6 ms (CAS-B); 15–30 ms (MDK); 0–3 control steps of 20 ms (ODM); fixed 30 ms (WTW); a blend of 0–50 % of the previous action at 100 Hz (HG) | 0–24 ms in 2 ms steps (PLAN:360) |
| Control-step jitter | 3–20 ms (TAN); policy period 15–30 ms around 25 ms (CAS-O) | none |
| Sensor delay | state 0.45–28.6 ms (CAS-B); IMU 0–20 ms and joint velocity fixed 20 ms (MDK); IMU 0–3 control steps of 20 ms (ODM) | sensors one physics step old (`DECISION_LOG.md:122`) |
| Backlash | 0.002–0.015 rad as an encoder offset (bipedal character, (S)); ±1° joint (MDK); ±0.5° joint (ODM) | none in the model (`STAGE2_PLAN.md:306`); gate draft corners 0 and 1° (`BUY_GATE_DRAFT.md:36`) |
| Encoder or joint offset | ±0.015 rad (MDK, MJL); ±0.02 rad (WTW); ±0.01 rad (CAS-O); ±0.05 rad (BH); ±0.03 rad (ODM) | none stated |
| Armature | ×0.9–1.1 (MDK); ×1.0–1.05 (BH, ODM) | none stated |
| Link mass | ×0.8–1.2 (TAN); ×0.5–1.5 (CAS-B, CAS-O); ×0.9–1.1 (BH, ODM); ×0.95–1.05 with inertia (MDK); base ×0.8–1.2 and other links ×0.98–1.02 (T1); base ±5 kg (HG, LG ANYmal); −1 to +3 kg (G1, WTW) | trunk ×0.9–1.1 (PLAN) |
| Inertia | ×0.5–1.5 (TAN); ×0.7–1.3 defined (CAS-B) | none |
| Centre of mass | ±0.1 m base and ±0.01 m links (CAS-B); ±0.1 m base (T1); ±0.05 m (ODM); ±3 to ±15 mm (MDK) | none stated |
| Ground friction | 0.5–1.25 (TAN, LG); 0.3–3.0 (CAS-B); 0.4–1.1 (CAS-O); 0.2–1.25 (BH); 0.1–2.0 (T1, HG); 0.1–1.25 (G1); 0.1–3.0 (WTW); 0.5–1.0 (ODM); 0.7–1.3 (MDK); 0.3–1.2 (MJL) | 0.4–1.2 (PLAN) |
| Supply voltage | 14.0–16.8 V (TAN); 6.5–8.2 V with sag of 0–0.2 V per N·m (MDK) | fixed 24.0 V (PLANT `options.bus_V`) |

Not retrieved, hosts blocked: the tables of Tan 2018 (paper), RMA beyond motor strength, the IJRR Cassie paper (https://arxiv.org/abs/2401.16889, Table 4), the OP3 soccer paper (https://arxiv.org/abs/2304.13653), and the equations and randomisation table of the bipedal-character paper.

## 4. Gear types of this build: backlash, friction and how to represent backlash

- **Gear types.** X4 and X6: planetary (the "P" in the model name), 36:1 and 19.612:1; stage count: no source found. XM540 (272.5:1) and STS3215 (1:345): gear type not in the study's records; no source found in this session.
- **Backlash figures.**

  | Drive | Maker | Third party |
  |---|---|---|
  | X4, X6 | ≤15 arcmin = 0.25° = 0.0044 rad; store pages say 10 | 20 arcmin on an older sibling against 5 advertised (S) |
  | XM540 | 15 arcmin | 0.11–0.29° identified on the XH540-V150 (S) |
  | STS3215 | ≤0.5° | 0.6–0.9° unloaded, 1.3° loaded (S) |

- **Effect at the foot (E).** 15 arcmin at a hip moves a foot 0.5 m away by 2.2 mm; the gate's 1° corner moves it 8.7 mm.
- **Friction of the planetary drives.** Maker's backdrive torque: 0.2 N·m (X4-36-L), 1.6 N·m (X6-60). For comparison the V4 X4-36 is 1.14 N·m and the V4.1 X6-50-L 0.2 N·m (RES:52, 186).
- **Loss split at the rated point (E), from the maker's efficiency, current and resistance:**

  | Drive | Loss left after copper loss | As a torque at rated speed | Share of delivered torque |
  |---|---|---|---|
  | X4-36-L | 27 W | 3.2 N·m | 30 % |
  | X6-60 | 65 W | 4.0 N·m | 20 % |

  These are upper bounds for gear friction: iron and drive losses are included. The size matches BAM's load-friction coefficient of 0.20–0.31 for geared Dynamixels. The study's model gives 0.15 N·m + 0.2 N·m·s/rad × speed, with no dependence on load.
- **Unrelated reference points (S):** a 3D-printed single-stage planetary measured 60–80 % efficiency and 0.59° backlash; one maker lists a 9 arcmin 8:1 drive.

**Three ways to represent backlash in a rigid-body simulator:**

| Way | Who uses it | What it captures | Cost |
|---|---|---|---|
| Passive hinge in series with a hard-limited range (deadband joint with a stiff stop) | microduck, Open Duck Mini | free play, impact on reversal, encoder reading through the play | one more degree of freedom and limit constraint per joint (this body: 20 to 34, from PLANT `nv`, `nu`); needs a limit stiffness near 2 × the step and a small armature; observations, firmware feedback and limit penalties must be rewired; which encoder closes the drive's loop must be known (X4 and X6 have both; no source found for which the motion mode uses) |
| Encoder offset toward the side the torque pushes, magnitude drawn per joint | bipedal character | the sign-dependent position error | no extra degrees of freedom; no free-play dynamics; needs the measured range |
| Constant random joint or encoder offset per episode | mjlab default, Berkeley Humanoid, Go1, Cassie | calibration error only | cheapest; not sign-dependent |

## 5. Prospective design points, all my estimates, to be checked on the bench

1. **Reflected rotor inertia is probably far too low in the model.**

   | Drive | Model (rule at `build_catalog.py:14`) | Outside evidence | Ratio |
   |---|---|---|---|
   | Tail servo | 0.000226 kg·m² | 0.027–0.028 fitted for the 7.4 V STS3215 | about 120× |
   | Neck servo | 0.000201 | 0.0058–0.026 on family members | 29–130× |
   | X4 | 0.00324 | maker's 0.3–0.82 kg·cm², if rotor-side, × 36² = 0.039–0.106 | 12–33× |
   | X6 | 0.00430 | maker's 0.66 kg·cm² × 19.612² = 0.025 | 6× |

   - The rotor-side reading is the physically plausible one; the maker should confirm it.
   - Rotor inertias read in code for comparable robot motors are 6e-6 to 1.5e-4 kg·m²; the rule assumes 2.5e-6 for the X4.
   - Leg-drive values of this size are comparable to the leg's own 0.12–0.23 kg·m² (BS:521).
2. **Torque at speed on a 24 V bus looks optimistic.** The model holds peak torque up to the usable speed (PLANT `envelope.rule`).

   | Drive, speed | Model | Line through the maker's rated and no-load points | Copper-loss-only line | At 21 V |
   |---|---|---|---|---|
   | X6, 8.01 rad/s | 42 N·m | about 10 N·m | — | about 0.4 N·m |
   | X4, 8.69 rad/s | 21 N·m | 10.5 N·m | 15.9 N·m | 5.3–8.0 N·m |

   - On the maker-points line the X6's 42 N·m is available only below about 4.2 rad/s.
   - The envelope should take the bus voltage as a drawn variable over 21.0–25.2 V.
   - Pack and wiring resistance for the sag: no source found.
3. **Friction should have a load-dependent term**, not only a constant and a viscous one. Coefficient of order 0.2–0.3 of transmitted torque as an upper bound, plus the maker's backdrive torque as the constant.
4. **Torque scale should be drawn wider on the low side than ±10 %.** One third-party test of a sibling found the torque constant 33 % below the quoted value (S). BS:569-572 already warns of a 1.4-fold rms-versus-peak ambiguity.
5. **The two hobby-class servos are not ideal position servos.**
   - Both run a voltage loop: a P term into duty cycle, with back-EMF as the only damping.
   - With the 7.4 V unit's firmware scaling and default gain, a 12 V STS3215 would give about 15.6 N·m/rad, against the model's 49.13.
   - Its damping would be about kt²/R: 0.55 N·m·s/rad on the 7.4 V unit, against the model's kv of 2.72.
   - A BAM-style model can be written for the STS3215 today from the published 7.4 V fit, with kt and R rescaled for the 12 V winding.
   - For the XM540 the firmware scaling was not retrieved.
6. **Backlash corners.** 0.25–0.35° covers the leg drives' maker and third-party figures. The gate's 1° corner is below the third-party loaded figure for the tail servo (1.3°).
7. **Identification methods for gate 2.** A free-shaft chirp with encoders only (PACE) gives armature, damping, Coulomb friction, bias and delay. A loaded pendulum with several masses and gains (BAM) adds load-dependent friction. Both fit with CMA-ES against position traces.

## 6. Not found
- For the X4-36-L and X6-60: friction curves, torque bandwidth, torque accuracy, command-to-torque delay, loop rates of the V4 drives, which encoder closes the position loop, gear stage count.
- For the XM540-W270: rotor inertia, resistance, friction, any third-party identification.
- For the STS3215 C018 (12 V): any third-party identification; the published fit is for the 7.4 V version and for the Coulomb-viscous model only.
