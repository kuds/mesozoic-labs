<!-- Working paper, 16 Nov 2026: one of three readers' reports gathered for the design of the servo model of gate 1 (BUY_GATE_DRAFT.md). Not reviewed. Nothing here is a rule or a result. Outside sources are quoted as found and were not checked a second time. -->

# Reader 1 (the code): what the simulation models per actuator today, and where a servo model can attach

Paths: `M/` = `/home/user/mesozoic-labs/hardware/trex_form_study` (branch `claude/mesozoic-trex-stage2-refine-4jffre`); `W/` = `/home/user/mesozoic-wt/wpc-ce/hardware/trex_form_study` (branch `wt/wpc-ce`, head 16890e8).
`git diff` between the two branches under `designlab/` shows only four new files (`stage2/r5env.py`, `r5core.py`, `r5lib.py`, `g8a.py`), so every Stage 1 file reference holds in both.

Marks on numbers:
- **[cat]** maker's catalogue value as transcribed in `M/designlab/data/actuators.json` (I did not re-open the maker's files).
- **[cat-d]** a catalogue entry whose own `basis` field says derived or estimate.
- **[A#]** an assumption of this study, by decision-log id.
- **[sim]** what the study's code or records do.
- **[probe]** printed today by my read-only probes.
- **[est]** my own arithmetic, untested.

Measured on hardware: nothing, anywhere. Everything in sections 1 to 4 is simulation code or simulation records; section 5 is prospective design only.

Probes (no physics step, no tracked file changed, `git status` clean afterwards) are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_design/`: `probe_sources.py`, `probe_plant.py` (`plant.build(settle=False)`), `probe_rows.py` (one `mj_forward` per variant).

## 1. The plant today (simulation)

### 1.1 Build chain

| Step | Where |
|---|---|
| Screened MJCF, written by the round-two generator | `M/designlab/mjcf.py:107-216` (`write_mjcf`) |
| Model file (git-ignored by `round2/.gitignore:3`), sha256 `0517e4d3…` | `M/round2/refine/feet_75/models/75-M-H-10.xml` |
| Design record (`selection` at line 16, `bus` 24.0) | `M/round2/refine/feet_75/candidates/75-M-H-10.json` |
| Hash check against the record | `M/designlab/stage1/plant.py:136-146` |
| Text edit of a copy: sensors, ankle tendons, ranges | `plant.py:322-378` (`augment`) |
| Compile, then gains, envelope, settle | `plant.py:872-989` (`build`) |
| Stored description of this body's plant | `M/stage1/runs/2026-11-08/75-M-H-10/plant.json` |

The body has 14 hinges and 14 actuators and no jaw (nq 21, nv 20, nu 14, na 0, 4 tendons) [probe]. A jaw and neck yaw exist only on the `-X` bodies.

### 1.2 Global settings

| Item | Value | Source |
|---|---|---|
| Physics step | 2 ms, `implicitfast`, elliptic cone | `mjcf.py:113` [sim] |
| Solver | Newton, 100 iterations, tolerance 1e-8, MuJoCo 3.10.0 | [probe] |
| Servo target written | every physics step (500 Hz) | `M/designlab/stage1/control.py:15-18`, `1429-1430` |
| Tracker feedback | 100 Hz, ramped over 10 ms | `control.py:149-152` |
| Policy step | 50 Hz (10 physics steps) | `M/designlab/stage1/env.py:63`; `W/designlab/stage2/r5core.py:71` |
| Action filter | first-order low-pass, 10 Hz | `env.py:64`, `532`; `W/…/r5env.py:153`; `r5core.py:409`, `588-593` |
| Servo's own loop | MuJoCo evaluates kp(ctrl − q) − kv·q̇ on the true state each 2 ms step | actuator arrays, [probe] |
| The only `mj_step` call in an episode | `Plant.step`: `envelope.update()` then `mj_step` | `plant.py:687-692` |

### 1.3 Per actuated joint class

Every actuator is a MuJoCo `position` actuator: no activation state, fixed gain, affine bias, gear 1 [probe].

| Class (count) | Drive part | Acts on | kp N·m/rad | kv N·m·s/rad | Force range N·m | Corner / no-load speed rad/s | Armature kg·m² |
|---|---|---|---|---|---|---|---|
| neck_pitch (1) | `robotis_xm540_w270_t` | joint | 177.02 | 4.159 | ±3.1058 | 1.885 / 3.142 | 0.000201 |
| tail_yaw (1) | `feetech_sts3215_c018` | joint | 49.13 | 2.723 | ±0.8620 | 2.827 / 4.712 | 0.000226 |
| hip_roll (2) | `myactuator_rmd_x4_p36_36_e_l` | joint | 598.5 | 24.48 | ±21 | 8.692 / 11.624 | 0.00324 |
| hip_yaw (2) | same X4 | joint | 598.5 | 7.586 | ±21 | 8.692 / 11.624 | 0.00324 |
| knee (2) | same X4 | joint | 598.5 | 11.51 | ±21 | 8.692 / 11.624 | 0.00324 |
| hip_pitch (2) | `myactuator_rmd_x6_p20_60_e` | joint | 1140 | 30.78 | ±42 | 8.011 / 9.215 | 0.004304 |
| ankle motors a, b (4) | same X4, two per ankle | fixed tendon q_p ± q_r | 299.25 | 2.081 / 2.083 | ±21 each | 8.692 / 11.624 (motor speed) | 0.00324 on each tendon |

Values are from `plant.json` (`gains.drives`, `envelope.drives`) and agree with my probe. Where each column comes from:

- **kp** = sustained torque / 0.01 rad (`plant.py:858`; `M/designlab/carpet.py:176`, itself marked an assumption). Sustained = 0.57 × vendor rated [A1] (`M/designlab/catalog.py:27`). Each ankle motor gets half (`plant.py:859-863`).
- **kv** = 2·ζ·√(kp·I), ζ = 1, I the drive's inertia at home (`plant.py:857`, `864`; `carpet.py:177`). The MJCF's own gains (kp 84 / 160, kv 0.05 kp) are overwritten (`mjcf.py:204-207`, `243-252`; `plant.py:865-867`).
- **Force range** = 0.70 × vendor peak [A2] (`catalog.py:29`; `mjcf.py:250`). Vendor peak is 30 N·m (X4) and 60 N·m (X6) [cat] (`actuators.json:97`, `179`). The neck's and tail's "peak" are [cat-d], derived from stall.
- **Speeds**: the corner is `Actuator.speed_at(bus)`, the zero is the no-load speed on the bus (`catalog.py:77-82`; `drive.py:77-81`).
  - X4: 83 and 111 rpm at 24 V [cat] (`actuators.json:98-99`).
  - X6: 153 and 176 rpm at 48 V [cat] (`actuators.json:180-181`), halved on the 24 V bus.
  - Neck and tail corners are 0.6 × no-load [cat-d].
- **Armature**: [cat-d], "estimate (rule in build_catalog.py)". Written per joint by `mjcf.py:153-155`, `219-224`.
- **Ankle**: the plant sets the ankle joints' own armature to 0 and puts 0.00324 on each motor tendon (`plant.py:336-337`, `345-352`). In joint space that is 0.00648 on pitch and on roll (`plant.json` `ankle.armature`).

Joint passive terms, identical on all 14 hinges:

| Term | Value | Source |
|---|---|---|
| Damping | 0.2 N·m·s/rad | `mjcf.py:120`, unsourced [A18] (`M/DECISION_LOG.md:123`) |
| Friction loss | 0.15 N·m | same line, [A18] |
| Friction constraint softness | solref (0.02, 1), MuJoCo's default | [probe] |
| On the four ankle tendons | damping 0, friction loss 0 | [probe] |

- The ankle's friction is therefore in joint space (pitch, roll), not on the motors.
- The catalogue notes hold maker figures that the simulation does not use:
  - backdrive torque 0.2 N·m (X4) and 1.6 N·m (X6-60) [cat] (`actuators.json:125`, `208`);
  - backlash ≤15 arcmin (X6-60) [cat] (`actuators.json:208`) and ≤0.5° (STS3215) [cat] (`actuators.json:1566`, `1571`);
  - X4-36-L and XM540 backlash: no source found in their entries.

### 1.4 How the limits are enforced

| Limit | Mechanism | Where |
|---|---|---|
| Torque | `actuator_forcerange` clamps the actuator's total force (PD plus feedforward). `Envelope.bind` sets `forcelimited` on every actuator and restores the peaks at every reset | `drive.py:236-250`; `plant.py:649`, `658` |
| Torque falling with speed | Per step, in code: driving side is peak up to the corner, then linear to zero at no-load; braking side stays at peak. Written into `m.actuator_forcerange` only while some drive is past its corner | `drive.py:84-98`, `253-271` |
| Speed | No clamp. Driving torque reaches zero at the no-load speed. Separately the judge fails an episode if peak drive speed exceeds `p.envelope.rated` | `control.py:2441`; `panels.py:1084-1087` |
| Bus voltage | Fixed at build: 24 V from the record, or a `bus` argument. It scales only the speed axis by min(1, bus / nominal), leg drives only. No dependence of torque on voltage and no sag with current | `plant.py:874`, `895`, `936`; `catalog.py:77-82` |
| Command range | The MJCF's ctrlrange is switched off, because ctrl carries the feedforward: `ctrl = G q + (kv G q̇ + G⁻ᵀ τ) / kp` | `plant.py:868`, `359`, `662-681` |
| Joint range | MuJoCo joint limits at the clearance screen's usable ranges | `plant.py:331-333`, `890` |
| Ankle linkage | Two motors at ±21 each give \|τ_p\| + \|τ_r\| ≤ 42 | `drive.py:8-17` |

The stand-in controller's feedforward includes the model's own passive terms: `tau + damp·q̇ + fric·tanh(q̇ / 0.05)`. It copies them from the plant's model when the tracker is built (`control.py:1204-1205`, `1403-1404`). So today the controller knows the simulated friction exactly.

## 2. Delay, noise and randomisation today

**Delay model** (one model, three copies of the same queue):
- Copies: `speedfit.servo_delay` (`M/designlab/stage1/speedfit.py:324-361`), `env.Stage1Env._latency_on` (`env.py:513-524`), `ledger._latency` (`M/designlab/stage1/ledger.py:708-735`).
- After `p.command` writes `d.ctrl`, the queue replaces it with the value from k physics steps earlier.
- The whole command vector is delayed together: position target, rate feedforward and torque feedforward.
- The servo's own PD on q and q̇ is not delayed.
- k is a whole number of 2 ms steps (`M/STAGE2_PLAN.md:351-360`).
- At reset the queue is refilled with the reset's targets.

**Sensor or observation delay**: none is modelled. The only lag is MuJoCo's own: after a step, sensors describe the state at that step's start, 2 ms earlier (`plant.py:703-705`).

| Factor | Stage 1 env (`env.py:217-241`, `437-458`) | R5 / R6 env (`W/…/r5env.py:116-157`, `364-394`) | Bar's judging rows (`M/designlab/stage2/bar.py:880-918`) |
|---|---|---|---|
| Servo delay | k uniform on 0–10 (0–20 ms) per episode | k uniform on 0–12 (0–24 ms), through `speedfit.servo_delay` (`r5env.py:504-505`) | fixed per row name, even ms, 0–20 (`bar.py:134-136`) |
| Floor friction | 0.54–1.25 | 0.4–1.2 | the condition's |
| Trunk mass and inertia | ×0.9–1.1 (`mj_setConst`) | ×0.9–1.1 | 1 |
| Servo kp | ×0.8–1.2 | ×0.6–1.2 | 1 |
| Joint damping | not drawn | ×0.5–1.5, one factor for all hinges (`r5env.py:390-391`) | 1 |
| Joint friction loss | not drawn | ×0.5–1.5, one factor for all hinges (`r5env.py:392-393`) | 1 |
| Sensor noise scale | ×0.5–1.5 | ×0.5–1.5 | 1 |
| Pushes | probability 0.5, 0–1 × Stage 0 bound, 0.1 s | probability 0.5; 0–0.5 walking, 0–1 standing | the condition's |
| Start noise | 0.01 rad, 0.02 m/s | same, enforced (`r5env.py:204-207`) | same |
| Mid-task starts | none | half of episodes | none |
| Floors per job | hard, hard, carpet, low pile | hard ×3, carpet | the condition's |

- The kp factor scales `gainprm[0]` and `biasprm[1]` only; kv is not scaled (`env.py:455-457`; `r5env.py:387-389`).
- Stage 1b's refit drew k on 0–5 (0–10 ms) (`speedfit.py:107`, `282-285`).
- The R5 env builds every tracker on the nominal model and refuses a reset if a tracker's damping or friction copy is not nominal (`r5env.py:239-240`, `438-440`). The feedforward therefore stays nominal while the plant's friction is drawn.
- `r5lib.py` randomises nothing; it builds the reference scripts.

Sensor noise (`M/designlab/stage1/estimator.py:83-93`, applied in `154-171`). The file says of its own values "assumed MEMS-class figures" (`estimator.py:76-80`); they are assumptions of this study with no decision-log id.

| Channel | Value |
|---|---|
| Gyro | white 0.003 rad/s per sample; bias within ±2e-4 rad/s per episode |
| Accelerometer | white 0.03 m/s²; bias within ±0.05 m/s² |
| Joint angle | rounded to 14 bits per turn |
| Joint rate | white 0.02 rad/s |
| Foot load | white, 2 % of the weight |

- `scale` multiplies the noises and biases, not the encoder bits.
- The quantisation applies to what the controller or policy reads. The servo's own loop sees the exact angle.
- Catalogue encoder strings: "dual encoder 17-bit/18-bit" (X4), "17/17-bit" (X6), "12-bit magnetic encoder" (STS3215) [cat] (`actuators.json:108`, `190`, `1552`). XM540: no source found in its entry.

## 3. Record keys, and where a servo model can attach

### 3.1 What is hashed (run today on `M/`, `probe_sources.py`)

**`panels.panel_sources()`**, sha256 `c0027992…` (`M/designlab/stage1/panels.py:347-359`, `1559-1583`):
- `designlab/stage1/`: `__init__.py`, `control.py`, `drive.py`, `estimator.py`, `legs.py`, `metrics.py`, `panels.py`, `plant.py`, `refs.py`, `transitions.py`, `tune.py`.
- `designlab/`: `__init__.py`, `build.py`, `carpet.py`, `catalog.py`, `clearance.py`, `envelope.py`, `gait.py`, `geometry.py`, `hardware.py`, `mjcf.py`, `params.py`, `shapes.py`, `sizing.py`, `skeleton.py`, `statics.py`, `template.py`, `turning.py`.
- Left out: `env.py` and `gates.py` (`panels.py:1503-1504`). From `env.py` the key holds by value the push bound, `RESIDUAL_RATE`, `ENV_SUBSTEPS`, the carpet walk's plan and friction, and the source hash of `env.foot_tables` (`panels.py:1601-1607`).

**`bar.bar_sources()`**, sha256 `45719d03…` (`bar.py:1677-1687`): `designlab/stage2/bar.py`, `readings.py`, `setio.py`, and `designlab/stage1/speedfit.py`.

**Also in every key** (`panels.py:1609-1651`; `bar.py:1693-1705`):
- the row's `describe()` and `sources()`;
- the sha256 of the body's record, MJCF, ledger and clearance screen;
- the library versions.

**Other hashes that go stale**: `plant.sources_sha256()` covers every `designlab/stage1/*.py` (`plant.py:262-275`). It is in the trainer's records and is one of the four hashes `regress check` compares (`M/designlab/stage2/regress.py:84`, `214-219`). Adding any file under `designlab/stage1/` changes it.

**The rules**:
- "Change the generator only behind options that default to today's behaviour, and test that bit for bit" (`M/LESSONS_LEARNED.md:252-254`).
- "Records hash their code too … Any edit there … makes earlier records read as made by other code" (`LESSONS_LEARNED.md:260-262`).
- "Never rerun or extend Stage 1 or 1b records with newer code" (`M/NEXT_SESSION_PROMPT.md:223-225`).

**Consequence for (c)**:
- An option that defaults to today's behaviour keeps results bit-identical, but it still changes the key if it lives in a hashed file.
- For no existing key to change, the servo model must be a new module under `designlab/stage2/`, not under `designlab/stage1/`, that edits none of the files listed above.
- It must leave the screened MJCF on disk untouched.
- `r5env.py`, `r5core.py`, `r5lib.py` and `g8a.py` are hashed only into a G8a gate record (`W/…/g8a.py:254-260`). No committed JSON on `wt/wpc-ce` names them by hash (grep), so they can be edited without touching an existing key.

### 3.2 Hook points that exist today

| Hook | What it gives | Precedent |
|---|---|---|
| H1. Compile time | Extra joints, tendons, actuator states. `plant.augment` and `plant.build` are hashed; a new builder would have to repeat `build`'s sequence around its own XML edit | `plant.py:322-378`, `872-989` |
| H2. Model arrays per episode | `dof_damping`, `dof_frictionloss`, `tendon_damping`, `tendon_frictionloss`, gain and bias parameters, masses. MuJoCo reads these at run time: friction rows go 14 → 18 with tendon friction and 14 → 0 with none [probe] | `r5env.py:376-394`; `ledger.py:801-839` |
| H3. Replace `p.envelope` | Its `update()` is called before every `mj_step` and its `bind()` at every reset and `set_state`. It holds `m` and `d`, so it can rewrite force ranges (and `d.ctrl`) and keep state | `plant.py:649`, `658`, `689-691`; swap with undo in `ledger.py:777-799` |
| H4. Instance override of `p.command` / `p.reset` | Arithmetic on the command | `speedfit.py:324-361` |
| H5. Instance override of `p.step` | Arithmetic before and after each physics step | `M/designlab/stage1/policy_rows.py:841-852` |
| H6. Sensor path | `tk.noise = SensorNoise(scale=…)` per episode; `NoisySensors.read` is hashed | `r5env.py:340`; `estimator.py:154-171` |

**Training env side (a)**: the R5 env leaves the delay at reset step 10 and enters it at step 12 through an `ExitStack` (`r5env.py:260`, `416-418`, `493`, `504-505`). A servo-model context manager can enter the same stack.

**Judging side (b)**:
- A row is any `panels.Row` whose `episode(ctx, cond, seed, relax)` runs on `ctx.built(cond).p` (`panels.py:732-742`).
- `bar.Delayed` wraps an inner row in `speedfit.servo_delay` (`bar.py:915-918`).
- Rows from another module are registered with `panels.register_rows` (`panels.py:865-872`).
- The worker rebuilds rows from their specs on one shared context (`bar.py:2140-2141`).
- A new wrapper row's key carries its own `describe()` and `sources()`, so its records are new and old ones are untouched.

**Precedent for proving env and judge run the same plant**: `policy_rows.equivalence_check` (`policy_rows.py:169-185`, `866-868`) and gate G8a (`W/…/g8a.py:12-19`). Both compare the sha256 of qpos, qvel, ctrl and sensordata after every physics step.

### 3.3 What each hook can and cannot express

| Effect | H2 arrays | H3 / H4 / H5 per-step arithmetic | H1 compile time |
|---|---|---|---|
| Friction as MuJoCo friction loss and damping | Yes, per joint. Ankle motor-side friction through the tendon arrays. The R5 env already scales both by one factor | Speed- or load-dependent friction only by rewriting the arrays each step, or as a torque added through ctrl (Δτ / kp) [est] | Not needed |
| Backlash as a deadband between motor and joint | No | Approximation only: a hysteresis offset on ctrl, or zero force range while a code-side rotor state crosses the gap. The rotor inertia stays rigidly on the link | Yes: a motor-side degree of freedom with a limited play joint or tendon. The related project inserts an unactuated hinge after each servo joint, range ± play/2, with a stiffened limit (`/home/user/pollen-robotics/microduck_rl/src/mjlab_microduck/robot/microduck/add_backlash.py:1-27`, `38-58`, commit 0daab16; read, not run) |
| Torque–speed line | n/a | Exists today (`drive.Envelope`) | n/a |
| Line moving with bus voltage | n/a | Speed-axis scaling exists at build time only. Per step it needs a replacement envelope that rescales its corner and no-load arrays | n/a |
| Voltage sag from total current | n/a | Yes: current from the previous step's `d.actuator_force` and kt [cat] (`actuators.json:104`, `186`, `1462`, `1548`), with one step of lag. The formula for drive input power is in `M/designlab/power.py:68-92` (efficiencies [A6]). Pack and wiring resistance: no source found (`parts.json:193`, entry `lipo_6s_3300`, has none) | n/a |
| First-order position response | n/a | Yes, a lag on ctrl at the same place as the delay queue | Also possible (`position` actuator time constant), but it adds activation states; na is 0 today [probe] |
| First-order torque response | n/a | Not while MuJoCo computes the PD: only the ctrl term can be filtered. It needs the PD in code with raw-torque actuators (array edit precedent: `M/designlab/stage1/gates.py:199-201`). That gives up the integrator's implicit treatment of kv; kv·dt/I reaches 1.15 on the ankle motors [est from `plant.json`]. Third-party statement of the trade-off: `/home/user/mujocolab/mjlab/docs/source/actuators.rst:62-86` | Same limit |
| Encoder quantisation | n/a | Controller side exists (14 bit). In the servo's own loop: shift ctrl by (q − q rounded) each step [est] | n/a |

A sense of scale for backlash: crossing 1° of play with the X4's rotor alone (0.00324 kg·m² [cat-d]) takes about 2.3 ms at 21 N·m and about 10.6 ms at 1 N·m [est, frictionless]. That is 1 to 5 physics steps, so per-step arithmetic resolves it coarsely.

### 3.4 Traps found in the code

1. **Friction leaks into the feedforward on the judging side.** `panels.TrackerRow.episode` builds its tracker inside the episode (`panels.py:844`), and the tracker copies the plant's damping and friction then (`control.py:1204-1205`). A wrapper that scales friction around `inner.episode` therefore hands the true friction to the feedforward. The R5 env does the opposite (section 2).
2. **Order of wrappers.** `servo_delay` refuses a plant whose `command` or `reset` is already replaced and wraps the class's own methods (`speedfit.py:336-338`). A servo layer must be entered after the delay, or use H3 or H5.
3. **The judge reads the envelope.** It reads `p.envelope.rated`, `.peak` and `.active`, `m.actuator_forcerange`, `d.actuator_force` and `p.drives[].sustained` (`control.py:2304`, `2314`, `2335`, `2366-2367`, `2440-2441`; `M/designlab/stage1/metrics.py:122`, `139`). A replacement envelope must keep the nominal `rated` and `peak`, or the registered limits move.
4. **Friction counts as load.** Joint friction and damping are passive forces outside `actuator_force`, so the motor torque that overcomes them counts in the duty reading. That includes gate 1's tail-servo condition (A18: they dominate the tail's load).
5. **The worker's plant is shared** across rows and conditions (`panels.py:979-988`). Every change must be undone exactly; the ledger checks this (`ledger.py:833-837`).
6. **Snapshots miss servo state.** `Tracker.save` / `load` and `control.relaxed` assume MuJoCo's state plus ctrl is the whole plant state (`control.py:1347-1388`, `2061-2100`). Lag, backlash or sag state inside a servo layer is outside it. `relaxed("envelope")` uses the envelope's `enabled` setter.

## 4. Cost

| Measurement | Value | Source |
|---|---|---|
| R5 env, one env in one process, hard floor | 382 env steps/s; carpet 307. Provisional (load average about 3); body not stated on the line | `W/stage2/design/checkpoint/reviews/roundC__build_CE.md:11` |
| Same, reviewer's own run (load 2.4–3.2 on 4 cores) | hard 385 with action noise, 401 with the zero action; carpet 302 and 413 | `W/…/reviews/roundC__review_CE.json:76` |
| Plant alone, zero command, 75-M-H-10 | 25,358 physics steps/s hard; 18,753 carpet (8 Nov record, different session and load) | `M/stage1/runs/2026-11-08/75-M-H-10/plant.json:1041`, `1103` |

- From these, the env spends about 260 µs per physics step all-in, of which physics is about 39 µs, roughly 15 % [est; the two measurements are not from the same session].
- **Per-step arithmetic** lands in the other 85 %. Today's `Envelope.update` is two numpy calls and a comparison in the common case (`drive.py:253-258`). A servo layer of 10 to 30 small numpy calls per physics step would add very roughly 4–12 % [est, not measured].
- **Per-episode array changes** cost a few writes at reset.
- **Extra joints for backlash** add physics cost:
  - nv goes from 20 to 34 (all 14 drives) or 32 (legs only), with up to 14 more limit rows, beside today's 14 friction rows [probe].
  - If physics time rose 1.5 to 2 times, env time would rise about 8–15 % [est]. I ran no step, so this is not measured.
  - The larger unknown is stability at 2 ms: inside the gap the ankle joints' own inertia is 0.00125 (pitch) and 0.00028 (roll) kg·m² without the motors' 0.00648 [probe]. The related project stiffens the limit to twice its 5 ms step (`add_backlash.py:47-58`).

## 5. Workable attachment designs (prospective; no recommendation)

**A. Run-time servo layer on the unchanged plant.** A new module under `designlab/stage2/`, used as a context manager on a `plant.Plant`: per-episode array factors (H2), a replacement envelope for the torque–speed–voltage line and sag (H3), and per-step ctrl arithmetic for lag, inner-loop quantisation and an approximated backlash.
- Keys: none changes. The env enters it beside the delay; the bar gets a wrapper row like `bar.Delayed` from its own row module.
- Its off state can be proven identical to today by step digests, in G8a's form.
- Backlash is not physical: at the ankle the rotors are about 84 % (pitch) and 96 % (roll) of the joint-space inertia [est from probe].
- A torque lag cannot be expressed.
- Traps 1 to 6 of section 3.4 all apply.

**B. Compile-time servo plant.** A new module builds a variant model with a motor-side degree of freedom per drive and a limited play, without editing `plant.py`, and returns a `plant.Plant` whose `joints` are the 14 actuated hinges. A's layer can sit on top.
- Gives physical backlash, rotor inertia on the motor side, and separate motor-side and joint-side friction.
- Keys: unchanged only if `plant.build`'s roughly 110 lines are repeated outside the hashed file, which can drift from the original.
- nq and nv change, which touches:
  - the keyframe;
  - `Plant.set_state` with reference frames (`plant.py:652-659`), so mid-task starts;
  - the estimator's structure check (`estimator.py:248-250`);
  - the hinge-order and transmission checks (`plant.py:918`, `923-928`).
- The gain rule (kv from inertia including armature) and the settled stance need redefining.
- Play of 0 is still a different model, so no step-for-step identity with today's records.
- Judging rows need their own plant and context (precedent for a derived context: `ledger.LedgerContext.of`, `ledger.py:750-754`).
- Physics cost and stability at 2 ms are unmeasured.

**C. Options inside the hashed Stage 1 files.** For example a `servo=` argument to `plant.build`, options on `drive.Envelope` and a hook in `Plant.step`, all defaulting to off.
- One implementation serves env and bar by construction, with the least code. It can host A's arithmetic or B's compile edits.
- Results stay bit-identical by default, but it fails (c) as stated: `plant.py` and `drive.py` are in `panel_sources()` and in the Stage 1 package hash, so every existing panel and bar record's key names other code.
- The regress record must then be re-made (full tier: 256 comparisons of 248 episodes, `regress.py:24-26`).

## 6. Not verified, or no source found

- Battery and wiring resistance, for sag: no source found in the repository.
- Backlash of the X4-36-L and the XM540; XM540 encoder resolution: no source found in their catalogue entries. Whether the X4's and X6's second encoder is on the output shaft: not stated there.
- The catalogue's maker values were read from `actuators.json` only; the maker's files were not re-opened.
- MuJoCo's own documentation could not be reached (`mujoco.readthedocs.io` is outside this session's allowed network domains). The statement about implicit integration of actuator damping rests on the third-party mjlab document cited above and is not re-verified.
- No physics step was run, so every cost figure for a servo layer or for extra joints is an estimate.
- The body behind the builder's throughput line is not stated in that report.
