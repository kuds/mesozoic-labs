# Map 05: plant, tracker, env and training code (for R5 / R6)

Read-only map made on 2026-11-14. Paths are relative to `/home/user/mesozoic-labs/hardware/trex_form_study` unless
absolute. `file:line` refers to the working tree as read (the stage1 package's hash at the time:
`plant.sources_sha256()` = `cc2a5c418b94a7ed...`). Nothing was run beyond imports and reading committed JSON.
"not found" means searched for and absent.

Files read in full: `designlab/stage1/{env,control,estimator,plant,policy,train,policy_rows,gates}.py`,
`tests/test_stage1_ppo.py`, `tests/conftest.py`; in part: `tests/test_stage1_policy_rows.py`, `tests/test_stage2_x.py`,
`designlab/stage1/{panels,speedfit,ledger,__main__,__init__}.py`, `STAGE2_PLAN.md` §4 B, §5, §6,
`LESSONS_LEARNED.md` §4.5, §5.3, `/home/user/mesozoic-labs/environments/shared/direction_commands.py`.

---

## 0. The five facts that shape everything else

1. **Every hashed source set, and what an edit to each file invalidates** (measured with the code's own functions):

   | Edit to | `train.py` resume hash (`plant.sources_sha256`, all `designlab/stage1/*.py`, non-recursive glob, `plant.py:267-275`) | tuner sources (`tune.tuner_modules`) | panel record keys (`panels.key_modules`) | speed-fit sources (`speedfit.speed_sources`) | policy rows' own extra sources (`policy_rows._policy_sources`, `policy_rows.py:666-667`) |
   |---|---|---|---|---|---|
   | `control.py`, `estimator.py`, `plant.py`, `drive.py`, `metrics.py`, `refs.py`, `transitions.py`, `legs.py`, `tune.py`, `__init__.py` | yes | yes | yes | yes | no |
   | `panels.py` | yes | no | yes | yes | no |
   | `env.py` | yes | no | **no**, but see below | **yes** | yes |
   | `gates.py` | yes | no | no (attribution records hold its sha256) | yes | no |
   | `policy.py`, `policy_rows.py` | yes | no | no | no | yes |
   | `speedfit.py` | yes | no | no | yes | no |
   | `train.py`, `__main__.py`, `ledger.py`, `report.py`, `speedreport.py`, `replay.py` | yes | no | no | no | no |
   | **a new `.py` file added to `designlab/stage1/`** | **yes** (the glob picks it up) | only if imported by a hashed module | same | same | no |
   | a file under `designlab/stage2/` | no | no | no | no | no |

   Module lists as computed today: tuner = `__init__, control, drive, estimator, legs, metrics, plant, refs,
   transitions, tune`; panels = those + `panels`; speed fit = panels' + `env, gates, speedfit`.
   R3/R4 records are panel records as well, so their keys carry the panel sources **and** the last column's three
   files (`env.py`, `policy_rows.py`, `policy.py`).
   Panels exclude `env.py` and `gates.py` by name (`panels.py:1504` `KEY_EXCLUDED = ("env.py", "gates.py")`) but hold
   **by value** (`panels.env_values`, `panels.py:1600-1606`): `env.PUSH_BOUND_M_S[body]`, `env.RESIDUAL_RATE`,
   `env.ENV_SUBSTEPS`, `env.TASKS["carpet_walk"][0]`, `env.TASK_FRICTION["carpet_walk"]`, and the sha256 of the
   **source text of `env.foot_tables`** (`env.py:328-343`). Changing any of those six changes every R0-R2 panel key.
   `panels.CLOCK_MAX = RATE * (1 + PHASE_RATE_LIMIT) * (1 + env.RESIDUAL_RATE)` (`panels.py:122`) also sizes panel
   plans from `env.RESIDUAL_RATE`.
   **The panel key also hashes designlab modules outside `designlab/stage1`** (`PANEL_SOURCES_RULE`,
   `panels.py:347-359`; `panels.panel_sources(...)["designlab_modules"]` as computed today): `__init__, build,
   carpet, catalog, clearance, envelope, gait, geometry, hardware, mjcf, params, shapes, sizing, skeleton, statics,
   template, turning`. So an edit to `designlab/gait.py` or `designlab/turning.py` (build items A, B, D), even behind
   a default-off option, changes every panel record key, R2/42's and R2/43's included. The tuner's and the speed
   fit's hashes do not cover these modules (they are named by commit and dirty paths only, `tune.py:374-381`).
2. **The env refuses any plant that does not have 14 hinges** (`env.py:361-364`), and a test pins the refusal
   (`tests/test_stage2_x.py:629-637`). `-X` bodies have 16 hinges in a different order (§1.2).
3. **Delay exists only as whole 2 ms physics steps**, implemented three times as an instance-level replacement of
   `Plant.command` (§1.5).
4. **The "R0" the panels run is not `Tracker(mode="R0")`.** It is `Tracker(mode="R1", terms=(), ankle_cap=())` at
   clock rate 1.05 (`panels.py:879-882`, `ROWS_RULE` `panels.py:165-174`). `Tracker` mode `"R0"` refuses any rate but
   1 (`control.py:1180-1181`).
5. **`train.py` never sets `MUJOCO_GL`** ("not found" in `train.py`, `__main__.py`, `env.py`). `designlab/__init__.py:15`
   does `setdefault("MUJOCO_GL", "egl" if no DISPLAY and /dev/dri exists else "osmesa")`; with libOSMesa loaded torch
   segfaults while building an SB3 policy (`tests/conftest.py:1-13`, `LESSONS_LEARNED.md:428-437`). Only
   `tests/conftest.py:13` (`setdefault("MUJOCO_GL", "disable")`) protects test processes.

---

## 1. `designlab/stage1/plant.py` (1006 lines)

### 1.1 Entry points and signatures

- `build(body, floor="hard", seed=0, envelope=True, speed_scale=1.0, torque_scale=1.0, pile_grid=None, settle=True,
  ankle_gains="joint", ankle_armature="linkage", bus=None, braking="peak") -> Plant` (`plant.py:872-989`).
  `FLOORS = ("hard", "robot_floor_carpet") + carpet.PILES` (`plant.py:86`; piles `low, medium, high`).
- **Hazard (read-only work):** `model_files(body)` (`plant.py:116-133`) rebuilds a missing round-two MJCF **into the
  checkout** at `stage1/models/<body>/` via `pipeline.build_candidate(..., write=True)`. Any `plant.build` on a clone
  without `round2/refine/<case>/models/<id>.xml` writes files. Rebuilt models carry another ledger JSON, so every
  panel key differs (`policy_rows.py:19-23`).
- `verify(body, xml)` checks sha256 against the record's `xml_sha256` and `SHA256_PREFIX[body]`
  (`plant.py:136-146`; prefixes in `__init__.py:69-71`). Raises `PlantError`.
- `class Plant` (`plant.py:549-843`). Attributes used by everything downstream:
  `m, d` (MjModel, MjData; `d` is a property whose setter renews `_v = MjViews(d)`, `plant.py:608-617`),
  `joints` (hinge names, model order), `drives`, `envelope`, `meta`, `qadr`, `dofs` (int arrays, one per hinge),
  `G` (nu x nj motor matrix), `_GinvT`, `armature_extra` (nj x nj), `kp`, `kv` (per actuator), `trunk` (body id),
  `feet` ([left, right] body ids), `q_home`, `ctrl_home`, `z_home`, `_R_home`, `z_floor`, `q_settled`, `rng`.
- `Plant.reset(joint_noise=0.0, root_velocity_noise=0.0, seed=None, allow_unsettled=False)` (`plant.py:630-650`):
  `mj_resetData`, `qpos = q_settled`, uniform joint noise `+-joint_noise` on `qadr`, uniform root linear velocity
  noise on `qvel[:3]`, `ctrl = ctrl_home`, `envelope.bind`, `mj_forward`. `seed` replaces `self.rng` with
  `default_rng(seed)`. Raises `PlantError` if the stance did not settle on this floor.
- `Plant.set_state(qpos, qvel)` (`plant.py:652-659`): `mj_resetData`, sets the **full** `qpos`/`qvel` (nq / nv
  vectors), `ctrl = ctrl_home`, `envelope.bind`, `mj_forward`. No noise, no seed, no settled-stance check. It leaves
  the servo targets at **home**, not at the given pose.
- `Plant.command(q=None, qd=None, tau=None)` (`plant.py:662-681`): joint-space targets to motor coordinates,
  `ctrl = G q + (kv G qd + G^-T tau) / kp`, written in place into `self._v.ctrl`.
- `Plant.step(n=1)` (`plant.py:687-692`): `envelope.update()` then `mj_step`, n times. Timestep 0.002 s
  (`plant.json` `timestep_s`).
- `Plant.sensors()` (`plant.py:699-712`) dict: `gyro(3) rad/s`, `accel(3) m/s^2`, `quat(4)`, `q(nj)`, `qd(nj)`,
  `foot_load(2) N`, `foot_contacts(2)`, `foot_force(2,3)`. Sensors describe the state **at the last step's start**
  (2 ms before `d.qpos`), `plant.py:703-705`.
- `Plant.fall_cause()` (`plant.py:737-744`): `"tilt"` (trunk z axis more than `FALL_TILT_RAD = 0.6` rad from
  vertical), `"height"` (`qpos[2] < 0.7 * z_home`), `"proxy <geom>"` (a `cl_*` geom below the floor), else None.
- `Plant.foot_loads()` (`plant.py:776-777`), `toe_lowest()` (`plant.py:749-774`), `tilt_from_home()`
  (`plant.py:718-721`), `settle(...)` (`plant.py:791-843`).
- `describe(p)` (`plant.py:1004-1006`) = `plant.json`: `p.meta` + `provenance()` + `environment.omp_num_threads`.

### 1.2 Joint list per body (how `-X` adds neck yaw and jaw)

`joints = _hinges(m0)` = every hinge of the screened MJCF **in model order** (`plant.py:381-383`, `894`). Nothing
lists joints by hand. From the committed `plant.json` files:

| Body | nq / nv / nu | `joints` (order = joint-space index) |
|---|---|---|
| `60-N-T-10` (`stage1/runs/2026-11-08/60-N-T-10/plant.json`) | 21 / 20 / 14 | `neck_pitch, tail_yaw, left_hip_roll, left_hip_pitch, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll, right_*` (6) |
| `60-N-T-10-X`, `75-M-H-10-X` (`stage2/x/<body>/plant.json`) | 23 / 22 / 16 | **`neck_yaw, neck_pitch, jaw, tail_yaw`**, then the 12 leg hinges |

- On `-X` the legs start at index 4, not 2, and `tail_yaw` moves from index 1 to 3. Anything indexing joints by
  position breaks; the code indexes by name (`control.py:1208-1212`, `LEGS` `control.py:175`).
- Ranges (plant): `neck_yaw [-0.6, 0.6]`, `jaw [0.0, 0.6]`, `neck_pitch [-0.35, 0.35]`, `tail_yaw [-0.45, 0.45]`,
  `hip_yaw [-0.3, 0.3]` (the thigh twist of STAGE2_PLAN lever 6).
- The jaw **rests on its lower stop by design**: `clearance.REST_ON_STOP = {"jaw": "min"}` (`designlab/clearance.py:145`),
  `plant.rest_ends(joints)` (`plant.py:174-177`) returns -1 for it, and `REST_ON_STOP_RULE` (`plant.py:164-171`) takes
  that end out of the joint-range margins, `control.limit_bounds` (`control.py:2098-2105`) and the joint_range
  relaxation (`control.py:2084-2087`). `plant.json` `joint_ranges` gains two keys on `-X`: `rest_on_stop`,
  `not_screened_at_seam` (`plant.range_notes`, `plant.py:180-196`).
- Leg / non-leg masks: `plant.leg_joints(joints)` (`plant.py:199-202`) = not in
  `mjcf.HEAD_TAIL_ROLES = {"neck_pitch", "tail_yaw", "neck_yaw", "jaw"}` (`designlab/mjcf.py:229`);
  `plant.leg_drives(drives)` (`plant.py:205-208`) = group not in
  `sizing.SINGLE_GROUPS = ("neck", "tail", "neck_yaw", "jaw")` (`designlab/sizing.py:41-44`).
- Drives follow the joints with each ankle's pitch and roll replaced by motors a and b
  (`drive.motor_matrix`, `designlab/stage1/drive.py:144-160`; generic in the joint list).
- Sensors: 34 (sum of dims 50) on 14 hinges, 38 (54) on 16: `imu_gyro(3)`, `imu_accel(3)`, `imu_quat(4)`,
  `enc_q_<joint>` x nj, `enc_qd_<joint>` x nj, `<side>_foot_load(4)` x 2 (`plant.augment`, `plant.py:365-377`).
- Body registry: `BODIES` (`__init__.py:47-55`), `PROBE_BODIES` (`:57`), `HEAD_X_BODIES` (`:60`), `CANDIDATE_IDS`
  (`:64-65`; `-K` and `-X` map to their pick's id inside another case directory).

### 1.3 `plant.json` / `meta` keys

`body, floor, seed, model{file, file_sha256, record_sha256, plan_prefix, matches_record, matches_plan}, H_m,
leg_layout, options{ankle_gains, ankle_armature, bus_V, bus_source, braking, envelope, speed_scale, torque_scale,
pile_grid_m}, mass_kg, timestep_s, integrator, nq, nv, nu, ngeom, imu_site_trunk_m, floor_params, floor_info,
gains{rule, ankle_gains, ..., drives{<actuator>: kp, kv, inertia_kg_m2, coordinate}, mjcf_own}, drives[...],
envelope, ankle{linkage, armature}, joint_ranges{source, reading, mjcf, plant, dropped_inter_leg_limits[,
rest_on_stop, not_screened_at_seam]}, ctrl{limited, home}, sensors[{name, type, dim}], falls{...}, settle{status,
settled, seconds, ...}` (`plant.py:937-985`, `824-841`), plus `provenance{versions, git{commit, dirty, dirty_paths},
stage1_sources{sha256, rule, files}}` (`plant.py:294-316`). Stage 1's committed plant.json also has `floors`,
`floors_coarse_pile`, `throughput` (written by the CLI, not `describe`); its `provenance` predates
`stage1_sources` (keys `versions, git` only).

### 1.4 Hashes in plant.py

- `sha256(path)` (`plant.py:112-113`): file bytes.
- `sources_sha256(files=None)` (`plant.py:267-275`), `SOURCES_RULE` (`plant.py:263-264`): "sha256 over
  designlab/stage1/*.py in path order, each file as its path relative to the package, a NUL byte, its bytes and a NUL
  byte". `files = {p.name: p.read_bytes() for p in SOURCES.glob("*.py")}`: top-level only, 22 files today.
- `sources_sha256_at(commit)` (`plant.py:278-291`); `provenance()` (`plant.py:294-316`) reads versions with
  `importlib.metadata` for `scipy, gymnasium, stable-baselines3, torch` (note the hyphenated key
  `"stable-baselines3"`, unlike `train._versions`).

### 1.5 The delay model

The plant has **no delay of its own**. Three wrappers replace `p.command` on the instance (`p.__dict__["command"]`):

| Where | Signature | Queue fill | Also wraps |
|---|---|---|---|
| `env.Stage1Env._latency_on(steps)` / `_latency_off()` (`env.py:513-527`) | per episode, called after `_begin` (`env.py:502-505`) | `deque([p.d.ctrl.copy()] * steps, maxlen=steps+1)` = the targets left by the reset (**home**) | nothing |
| `ledger._latency(p, steps) -> undo` (`ledger.py:708-735`) | until `undo()` | `fill()` at creation and at every `p.reset` | `p.reset` |
| `speedfit.servo_delay(p, steps)` context manager (`speedfit.py:324-362`) | for the block | same | `p.reset`; raises if `command` or `reset` is already replaced |

All three: after the real `command` writes `ctrl`, append a copy, then `ctrl[:] = queue[0]`: the servos get the
target set `steps` physics steps earlier. Controller, estimator and sensors are untouched ("the delay is the
plant's", `speedfit.LATENCY_RULE` `speedfit.py:159-168`).

- Whole steps only: `EnvConfig.latency_steps = (0, 10)` = 0-20 ms, drawn with `rng.integers(lo, hi + 1)`
  (`env.py:235`, `442-443`); `speedfit.LATENCY_STEPS = (0, 5)` (`speedfit.py:107`), `ms_steps` rejects odd ms
  (`speedfit.py:1116-1121`). The plan's 5, 15 and 25 ms cannot run (`LESSONS_LEARNED.md:575-578`).
- **None of the three wraps `Plant.set_state`.** A start from a frame (`Tracker.reset(start="frame")` calls
  `p.set_state`, `control.py:1296`) under `servo_delay` or `ledger._latency` keeps whatever the queue held; under the
  env the queue is rebuilt after `_begin`, but from `p.d.ctrl`, which `set_state` sets to `ctrl_home`.
- The wrappers do not stack: `orig = type(p).command.__get__(p)` bypasses an existing override, and
  `_latency_off` pops `"command"` whoever put it there.
- `speedfit.latency_steps(episode_seed, j)` (`speedfit.py:282-285`): `default_rng(tune.derived_seed(episode_seed, j,
  3)).integers(0, 6)`. This is "Stage 1b's delay draw".
- Test: `test_latency_applies_the_command_of_steps_ago` (`tests/test_stage1_ppo.py:212-227`): with 3 steps, the
  first three commands leave `ctrl` at the held value, then command k-3 arrives; afterwards `"command" not in
  p.__dict__`.

---

## 2. `designlab/stage1/estimator.py` (473 lines)

- `SensorNoise` dataclass (`estimator.py:82-100`): `gyro_sigma 0.003`, `gyro_bias 2e-4`, `accel_sigma 0.03`,
  `accel_bias 0.05`, `encoder_bits 14`, `joint_rate_sigma 0.02`, `foot_load_sigma_frac 0.02`, `scale 1.0`;
  `SensorNoise.none()` = `scale 0, encoder_bits 0`.
- `NoisySensors(p, noise, seed, weight=None)` (`estimator.py:114-172`); `.read()` returns
  `dict(gyro(3), accel(3), q(nj), qd(nj), foot_load(2))`. One `rng.standard_normal(8 + nj)` call per physics step
  (`estimator.py:167`): **the noise stream's length depends on nj**, so the same seed gives other noise on `-X`.
  The docstring still says "3 + 3 + 14 + 2" (`estimator.py:157-159`); the code is generic.
- `EstimatorGains` (`estimator.py:223-230`): `k_leg_tilt 2.0`, `k_accel_tilt 0.05`, `k_leg_velocity inf`,
  `contact_on 0.20`, `contact_off 0.10`, `stance_point "loaded_pad_contact_point"`.
- `Estimator(p, gains=None, model=None)` (`estimator.py:233-268`). `model` must match the plant's `nq, nv, nbody,
  ngeom, nsite, njnt` (`estimator.py:248-251`).
  - `reset(qpos, qvel, reading=None)` (`estimator.py:271-290`): start from a **known full pose and velocity**.
  - `reset_at_rest(reading, yaw=0.0, base_xy=(0, 0))` (`estimator.py:292-308`): from sensors, body at rest on both
    feet.
  - `step(reading)` every physics step (`estimator.py:364-374`); `update(dt_update)` at 100 Hz
    (`estimator.py:376-455`).
  - State the env reads: `q, qd, R (3x3), gyro, load` (`env.py:733-735`); the tracker also `com, com_vel, foot (2x3),
    foot_yaw, omega_world, contact`.
- Generic in nj. Assumes exactly two feet and pads named `*_pad_*` (`estimator.py:205-220`).

---

## 3. `designlab/stage1/control.py` (2493 lines)

### 3.1 Constants

`FEEDBACK_EVERY = 5` (`control.py:149`; 100 Hz on 500 Hz physics), `FEEDBACK_HOLD = "ramp"` (`:152`),
`FRICTION_SMOOTH = 0.05` (`:153`), `COP_INSET = 0.005` (`:154`), `PAD_CLEARANCE = 0.005` (`:157`),
`ANKLE_CAP = ("box", "pad")`, `ANKLE_CAPS = (ANKLE_CAP, ("box",), ("pad",), ())` (`:160-161`),
`INWARD_BOUNDS = ("pad_clearance", None)` (`:163`), `PHASE_RATE_LIMIT = 0.05` (`:164`), `SWING_FREEZE_U = 0.8`,
`CONTACT_LOAD_FRAC = 0.02` (`:170`), `LEGS = ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_pitch",
"ankle_roll")` (`:175`), `TERMS = ("ankle", "step", "trunk", "heading", "phase", "capture")` (`:176`),
`MODES = ("R0", "R1", "R2")` (`:177`), `GAIN_SPEC` 14 named gains (`:183-203`), `GAIN_SETS = {"start",
"earlier_defaults"}` (`:208`).

### 3.2 `Track`, `Segment`, `build_track`

- `Segment` dataclass (`control.py:749-766`): `kind` ("periodic" / "finite" / "hold"), `name, t0, duration`,
  `S` (frames x 4 nj: `[q, qd, tau, qdd]`, each nj wide in the plant's joint order, `_joint_stack`
  `control.py:939-940`), `info` (dict of per-frame arrays, below), `step, n, t, period, duty, yaw_per_stride`.
- `frame_info(...)` (`control.py:847-922`) keys: `anchor` (0 left, 1 right, 2 midpoint), `swing` (-1 none),
  `u` (swing progress 0..1), `dcm_rel(2)`, `comv_rel(2)`, `zmp_rel(2)` (all relative to the anchor, in its frame, m
  and m/s), `poly`, `poly_n`, `poly_raw`, `poly_raw_n`, `weights(2)`, `roll_pitch(2)`, `yaw` (unwrapped, rad),
  `period`, `stand` (bool), `land_rel(2)`, `land_yaw`, `poly_hp`, `poly_raw_hp`.
- `Track(p, omega)` (`control.py:952-1096`): `add_periodic(ref, strides, name=None)` (requires
  `list(ref.joints) == p.joints`, `:969`), `add_transition(tr, next_frame=None, next_dt=None)`, `add_hold(q, tau,
  zmp, seconds, name="stand", period=1.0)`; `joint(t) -> 4 nj vector`; `info(t) -> (Segment, dict)`; `events`
  (stop commands: `kind, name, t_ref, command_phase, speed_m_s, period_s`, `:1000-1001`); `t_end`; **`q0`, `qd0` =
  the first frame's full state only** (`:963`, `:1021`).
- A stop event is added only when `TRANSITIONS.get(tr.name, (None, None))[1] == "stand" and "phase" in tr.arrays`
  (`control.py:994`): the lookup is **by transition name in `transitions.TRANSITIONS`**, so a transition for a new
  GaitSpec under a new name gets no stop event unless that table knows it.
- `build_track(p, run_dir, plan, omega)` (`control.py:1099-1132`): plan items `("ref", name, strides)`,
  `("transition", name)`, `("hold", seconds)`; files `<run_dir>/refs/<name>.npz`, `<run_dir>/transitions/<name>.npz`.
- `omega0(p, q_home, crouch_frac=0.03)` (`control.py:943-949`): `sqrt(g / (home CoM height - 0.03 H))`.

### 3.3 `Tracker`

```python
Tracker(p, track, mode="R1", gains=None, terms=TERMS, noise=None, seed=0, est_gains=None,
        inward_bound=INWARD_BOUNDS[0], ankle_cap=ANKLE_CAP, rate=1.0, model=None)      # control.py:1174-1177
```

- Modes (`control.py:1178-1192`, `1329-1339`):
  - **R0**: `terms` forced to `()`; clock is `p.d.time` (`:1388`); `rate` must be exactly 1 (`:1180-1181`:
    `"rate {rate!r}: a positive clock rate, and 1 for R0 (G2's replay clock)"`); `p.command(q, qd, ff)` (`:1403`);
    no estimator; `state = TrueState`.
  - **R1**: feedback on `TrueState(p).view()` (`control.py:1138-1153`).
  - **R2**: `NoisySensors` + `Estimator`, feedback on `EstimatedState` (`control.py:1156-1165`); `est.step` after
    every physics step (`:1428-1429`), `est.update` in `feedback` (`:1687-1688`).
- `model` (`control.py:1183-1186`, `1202`): the controller's own model (mass, R2's estimator, sensor weight); the
  env passes the nominal model while it randomises the plant.
- `Tracker.step()` (`control.py:1386-1429`), one physics step:
  1. `x = track.joint(t_ref)`; `q, qd, tau = x[:nj], x[nj:2nj], x[2nj:3nj]`.
  2. `s = rate * (1 + delta)`; `if self.rate_factor != 1.0: s = s * self.rate_factor`.
  3. `ff = tau + damp * (s qd) + fric * tanh(s qd / 0.05) [+ armature_extra @ (s^2 qdd)]`
     (`damp`, `fric` are read from the model once at construction, `:1199-1200`).
  4. Non-R0: every 5th step `feedback(t_ref, q)`; with `hold == "ramp"`, `dq` and `dtau` move linearly from the
     previous update's values to the new ones over the next 5 steps (`:1405-1413`).
  5. `q_cmd = q + dq`; `if self.residual is not None: q_cmd = q_cmd + self.residual` (`:1416-1418`).
  6. `tau_cmd = ff + dtau`; `if self.ankle_cap:` cap (`:1420-1423`); `p.command(q_cmd, s qd, tau_cmd)`.
  7. `p.step()`; `k += 1`; `tau_ref += dt * s`; R2: `est.step(sensors.read())`.
- Residual hooks (`control.py:1273-1277`): `tk.residual: np.ndarray | None` (rad, joint order, **nj wide**) and
  `tk.rate_factor: float`. "With None and 1 the tracker is exactly R2 (the offsets are not added, the rate not
  multiplied)."
- **"Zero feedback"** = `terms=()` and `ankle_cap=()`: `feedback` still runs every 10 ms (it updates the estimator,
  the heading buffer and `self.e`) but leaves `dq = zeros(nj)`, `dtau = zeros(nj)`, `delta = 0.0`
  (`control.py:1708`, `1813-1840`), and the cap is skipped. So the command is `q_ref + 0`, `rate * qd_ref`, `ff`:
  R0's targets on the commanded clock. This is `panels.ROWS_RULE`'s R0 and `policy_rows.PolicyOnlyTracker`'s base
  (`policy_rows.py:508`: `super().__init__(built.p, built.track, "R2", terms=(), ankle_cap=(), rate=cfg.rate)`).
- Terms (`feedback`, `control.py:1678-1840`): `ankle` (ZMP shift, `:1720-1755`), `trunk` (`:1757-1769`), `heading`
  (`:1785-1790`), `step` (`:1794-1811`), `phase` (clock rate PI within +-5 %, `:1829-1839`), `capture`
  (`_capture`, `:1843-1918`, standing only).
- `ankle_cap` (`_ankle_cap`, `control.py:1475-1564`): each ankle's total torque (feedforward + predicted servo PD +
  feedback) limited to one motor's peak per axis ("box") and, on a loaded foot, to torques whose quasi-static CoP is
  5 mm inside the pad ("pad"). Needs each ankle's pitch and roll joints adjacent (`:1257-1258`).
- `reset(start="frame", velocity_offset=None, joint_noise=0.0, seed=None, root_velocity_noise=0.0, noise_seed=None)`
  (`control.py:1282-1339`): `"frame"` = `p.set_state(track.q0, qd0)` and, in R2, `est.reset(qpos, qvel, reading)`;
  `"stance"` = `p.reset(...)` and `est.reset_at_rest(...)`. Always `tau_ref = 0.0`, `track._i = 0`.
  **There is no start at an arbitrary reference time.**
- `save(prev_state)` / `load(snap)` (`control.py:1342-1383`): full episode state incl. the noise generator.
- `run(tk, seconds, start="frame", velocity_offset=None, push=None, seed=None, period=None, confirm=False, relax=(),
  recorder=None, joint_noise=0.0, root_velocity_noise=0.0, noise_seed=None) -> dict` (`control.py:2263-2291`);
  `push = (t0, duration, fx, fy)` N on the trunk, applied by `_apply_push` before each step
  (`control.py:2108-2110`). Outcome keys at `control.py:2437-2466` (`mode, terms, start, rate, seconds, strides,
  reference_strides, fell, fall_time_s, fall_cause, fall_onset_s, ..., stops, wall_s, steps_per_s`).
  Recorder protocol: `begin(tk, n)`, `step(k)` after every physics step, `update(k, loads, slip_v, Rr, yaw)` at
  feedback updates, `end(steps, out)`.
- `relaxed(tk, names)` (`control.py:2055-2095`), `set_friction(p, mu)` (`:1988-1992`), `foot_slip_speed`
  (`:2016-2052`), `foot_geoms(p)` (`:1995-2000`).

### 3.4 How the panels' R0 targets are produced (`panels.ROWS_RULE`, `panels.py:165-174`)

> "R0: the reference with feedforward and servo PD only, played on the commanded clock: control.Tracker with no
> feedback term and no ankle cap (terms=(), ankle_cap=(); its R1 mode, whose state view then sets nothing), which
> commands exactly R0's targets (control.Tracker's R0 mode refuses another clock than the references' own; at rate 1
> the two are the same bit for bit)."

Built at `panels.py:879-882`: `TrackerRow("R0", "R1", None, dict(source=...), terms=(), ankle_cap=(),
hard_only=True, controller="R0: feedforward + PD on the commanded clock (ROWS_RULE)")`. Its episode
(`panels.TrackerRow.episode`, `panels.py:841-852`):

```python
tk = control.Tracker(p, b.track, self.mode, gains=self.gains, terms=self.terms, ankle_cap=self.ankle_cap, rate=self.rate)
out = control.run(tk, b.seconds, start="stance", seed=s["start"], noise_seed=s["noise"], recorder=hook,
                  joint_noise=tune.START_JOINT_NOISE_RAD, root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S,
                  push=b.push(seed), relax=tuple(relax))
```

with `rate = panels.RATE = tune.COMMAND_RATE = 1.05` (`panels.py:79`, `tune.py:111-112`), start noise 0.01 rad and
0.02 m/s (`tune.py:113-114`), seeds `panels.episode_seeds(cond, seed)` = `tune.derived_seed(seed, slot, purpose)`
(`panels.py:505-508`), panel seeds 3042-3061 (`panels.py:75`). The R0 row is `hard_only=True`: it has no
robot_floor_carpet or pile episodes. `panels.Policy` / `ZeroResidual` / `_Acting` (`panels.py:751-801`) are the
hooks a learned controller acts through (`act` after physics step k when `(k + 1) % every == 0`, `every =
env.ENV_SUBSTEPS`).

---

## 4. `designlab/stage1/env.py` (914 lines)

### 4.1 Classes

`Stage1Env(gym.Env)` (`env.py:346-699`), `ResidualEnv` = R3 (`env.py:702-735`), `PolicyOnlyEnv` = R4
(`env.py:738-790`), `make(mode, config, seed=None)` (`env.py:793-798`, `"R3"` or `"R4"` only). Hooks a subclass
provides (`env.py:417-434`): `_setup()`, `_begin(seeds)`, `_apply_action(a)`, `_physics_step()`, `_clock()`,
`_estimate()`; `_gains()` may be overridden (R4 does).

### 4.2 Observation and action layout (as computed: `ACTOR_DIM 57`, `OBS_DIM 90`, `ACTION_DIM 15`)

`ACTOR_OBS` (`env.py:138-139`) then `CRITIC_OBS` (`env.py:141-142`); `obs_layout()` (`env.py:275-281`):

| Block | Slice | Content (`_observation`, `env.py:623-659`) |
|---|---|---|
| `q` | 0:14 | estimator's encoder angles (rad) |
| `qd` | 14:28 | encoder rates (rad/s) |
| `gravity` | 28:31 | `-est.R[2]` (gravity direction in the trunk frame) |
| `gyro` | 31:34 | rad/s |
| `foot_load` | 34:36 | loads / **nominal** weight |
| `phase` | 36:38 | `sin, cos` of `2 pi clock / period[task]` |
| `support` | 38:40 | `[sw != 0, sw != 1]` from the reference's swing foot |
| `command` | 40:42 | `[v_cmd, w_cmd] / COMMAND_SCALE` (0.5; `env.py:74`) |
| `prev_action` | 42:57 | the previous **clipped, unfiltered** action (`env.py:559`) |
| `root_vel` | 57:60 | true root velocity in the heading frame |
| `height` | 60:61 | `(qpos[2] - z_floor) / H` |
| `roll_pitch` | 61:63 | about home |
| `dcm_error` | 63:65 | true DCM error (`_true_dcm_error`, `env.py:661-677`) |
| `joint_error` | 65:79 | `qpos[qadr] - q_ref` |
| `heading_error` | 79:80 | |
| `friction` | 80:81 | the drawn friction, 0.9 when none |
| `floor` | 81:84 | one-hot `hard / robot_floor_carpet / any pile` (`FLOOR_CODES`, `env.py:143`) |
| `push` | 84:87 | `[Fx / m, Fy / m, active]` while the pulse acts |
| `plant_factors` | 87:90 | `[mass, kp, latency_steps * dt / 0.02]` |

Action (`Box(-1, 1, (15,))`, `env.py:399`): `a[:nj]` joint values, `a[nj]` the clock-rate value.

### 4.3 Every place that assumes 14 joints (or a fixed width)

| file:line | What |
|---|---|
| `env.py:138-139` | `ACTOR_OBS`: `("q", 14)`, `("qd", 14)`, `("prev_action", 15)` |
| `env.py:141-142` | `CRITIC_OBS`: `("joint_error", 14)` |
| `env.py:284-286` | module constants `ACTOR_DIM`, `OBS_DIM`, `ACTION_DIM = 15` |
| `env.py:290` | `WIDTH_JOINTS = dict(ACTOR_OBS)["q"]` |
| `env.py:361-364` | the refusal: `if len(p.joints) != WIDTH_JOINTS: raise ValueError(... "is pending")` |
| `env.py:398-399` | spaces built from the module constants |
| `env.py:499-500` | `a_prev`, `a_filt = np.zeros(ACTION_DIM)` |
| `env.py:275-281` | `obs_layout()` takes no body: one layout for the module |
| `env.py:685` | `describe()` writes `ACTOR_OBS`, `CRITIC_OBS`, `ACTION_DIM` |
| `env.py:861`, `898` | `zero_residual_check`, `throughput`: `np.zeros(ACTION_DIM)`, `rng.uniform(-1, 1, ACTION_DIM)` |
| `train.py:355` | monitor's zero action `np.zeros(env_mod.ACTION_DIM)` |
| `train.py:511` | `model_kwargs(env_mod.ACTOR_DIM, ...)` |
| `train.py:683` | noise probe `sigma * rng.standard_normal(env_mod.ACTION_DIM)` |
| `policy_rows.py:143` | rule text `a_f[14]` |
| `policy_rows.py:572-578` | description records `env.ACTOR_DIM`, `OBS_DIM`, `ACTION_DIM` |
| `policy_rows.py:607-613` | checkpoint refused unless `actor_dim == env.ACTOR_DIM`, obs shape `(env.OBS_DIM,)`, action shape `(env.ACTION_DIM,)`, VecNormalize mean shape `(env.OBS_DIM,)` |
| `policy_rows.py:644-645` | `a_prev`, `a_filt = np.zeros(env.ACTION_DIM)` |
| `tests/test_stage2_x.py:635` | `assert env.WIDTH_JOINTS == 14 and env.ACTION_DIM == env.WIDTH_JOINTS + 1` |
| `tests/test_stage1_ppo.py:67-75, 80, 82, 248` | shapes against the module constants |

Already generic (index by `self.nj` or by name): `_apply_action` (`env.py:723-725`, `772-773`: `a[:self.nj]`,
`a[self.nj]`), `_reward` (`env.py:576-578`), `_observation`'s `x[:nj]` (`env.py:654`), `residual_scale`
(`env.py:710`, via `plant.leg_joints`), the span (`env.py:749-754`), `Tracker`, `Estimator`, `NoisySensors`, `Plant`.
Docstrings that say 14: `env.py:8`, `13`; `estimator.py:5`, `157-159`; `plant.py:11`, `552`, `555`.

### 4.4 Step timing, filter, latency

- `ENV_SUBSTEPS = 10` (`env.py:63`): 10 physics steps of 2 ms per env step = 50 Hz.
- `FILTER_HZ = 10.0` (`env.py:64`); `alpha = 1 - exp(-2 pi FILTER_HZ dt_env)` (`env.py:397`) = 0.7154;
  `a_filt += alpha * (clip(a, -1, 1) - a_filt)` (`env.py:531-532`), then `_apply_action(a_filt)`. The filter state
  starts at zero each episode (`env.py:500`). `STAGE2_PLAN.md:112` says the filter "is re-chosen in validation".
- Latency queue: §1.5. Drawn per episode in `_draw_factors` (`env.py:437-445`), switched on after `_begin`
  (`env.py:502-505`).
- `step()` (`env.py:530-564`): per substep `_apply_push`, `_physics_step`, optional `_step_hook(p)`, foot loads;
  **falls are read once, at the env step's end** (`env.py:553`), plus any MuJoCo warning (`env.py:554-555`).

### 4.5 R3 residual vs R4 policy-only

| | R3 `ResidualEnv` | R4 `PolicyOnlyEnv` |
|---|---|---|
| Controller | one `control.Tracker(p, track, "R2", gains, rate=cfg.rate, model=nominal_model)` per task (`env.py:708-709`) | none; `Plant.command(q_target)` alone (`env.py:777`) |
| Joint action | `tk.residual = a[:nj] * scale`, `None` when all zero (`env.py:723-724`); scale `RESIDUAL_LEG_RAD = 0.2` on legs, `RESIDUAL_NECK_TAIL_RAD = 0.1` elsewhere (`env.py:65-66`, `710`) | `q_target = clip(q_home + a[:nj] * span, lo, hi)` (`env.py:772`); `span = max(1.25 x largest excursion from home over the env's tracks, 0.10 rad)` (`env.py:747-756`) |
| Clock action | `tk.rate_factor = 1 + RESIDUAL_RATE * a[nj]`, `RESIDUAL_RATE = 0.25` (`env.py:67`, `725`) | `rate_factor` likewise; `tau += dt * rate * rate_factor`, capped at `track.t_end` (`env.py:773`, `780`) |
| Feedforward | kept (the tracker's) | **none** (`command(q)` only: no rate, no torque) |
| Clock | `tk.tau_ref` (`env.py:731`) | `self.tau` (`env.py:786`) |
| Estimator | the tracker's | its own `NoisySensors` + `Estimator(p, model=nominal_model)`, stepped every physics step, updated every 5 (`env.py:762-765`, `781-783`) |
| Start | `tk.reset("stance", joint_noise, seed=seeds["start"], root_velocity_noise, noise_seed=seeds["noise"])` (`env.py:718-719`) | `p.reset(...)`, `est.reset_at_rest` (`env.py:760-765`) |
| Gains | `tuned_gains` (`env.py:182-213`, `GAINS_RULE` `env.py:162-167`) | none |

### 4.6 Reward (`REWARD_RULE`, `env.py:149-159`; `_reward`, `env.py:571-611`)

`r = 0.55 * imitation + 0.35 * tracking - sum(PENALTIES[k] * pen[k])`

- `IMITATION_W, TRACKING_W = 0.55, 0.35`; `IMITATION_SCALE_RAD = 0.1` (`env.py:147-148`).
- `imitation = exp(-mean_j(((q_j - q_ref_j) / 0.1)^2))` over **all nj joints**, true joints, reference at the env
  step's end (`env.py:578`). On `-X` this mean includes neck yaw and jaw.
- `tracking = gaussian_tracking_reward(|speed - v_cmd|, |yaw_rate - w_cmd|)` (`env.py:588`); speed is the CoM's
  displacement along the previous heading over the env step, yaw rate the trunk's (`env.py:579-586`).
- `PENALTIES = dict(duty=0.02, slip=0.02, flight=0.05, impact=0.02, clearance=0.02, tilt=0.02)` (`env.py:146`), each
  term in [0, 1]:
  - `duty`: each foot's contact share over the last stride outside 0.55-0.65, over 0.05, while the reference walks
    (`env.py:592-594`);
  - `slip`: a foot carrying 20 % of the weight sliding faster than 0.05 m/s (`env.py:597-600`);
  - `flight`: share of substeps with neither foot above 2 % of the weight (`env.py:601`);
  - `impact`: `(grf_peak / weight - 1.6) / 1.6` (`env.py:602`);
  - `clearance`: the **reference's** swing foot at reference swing progress 0.25-0.75 below 0.04 H
    (`env.py:603-606`; this is the reference-clock reading STAGE2_PLAN §5 says misreads a late lift-off);
  - `tilt`: roll or pitch about home beyond 4 deg (`env.py:607-609`).
- Weight and BW are the episode's true ones (`self.weight_true`, `env.py:458`, `595`).
- `info["reward_terms"] = dict(imitation, tracking, duty, slip, flight, impact, clearance, tilt)`.
- Not present today: foot or contact imitation, head attitude, action rate, neck/tail torque terms ("not found").

### 4.7 The command and `direction_commands`

- Import (`env.py:170-179`): tries `from environments.shared.direction_commands import gaussian_tracking_reward`;
  on `ImportError` inserts `str(STUDY.parents[1])` (the repository root, `/home/user/mesozoic-labs`) at
  `sys.path[0]` and retries. Called once per env in `__init__` (`env.py:396`). So the env depends on the main
  repository's `environments/shared/direction_commands.py` being on disk two levels above the study.
- `gaussian_tracking_reward(velocity_error, yaw_rate_error, *, velocity_sigma=0.25, yaw_rate_sigma=0.15) -> float`
  (`/home/user/mesozoic-labs/environments/shared/direction_commands.py:421-444`) =
  `exp(-((ev / 0.25)^2 + (ew / 0.15)^2))`; raises on negative or non-finite errors. Only this function is used;
  `DirectionCommandController` etc. are not.
- **The command is the reference's, not an input** (`_command`, `env.py:613-620`):
  `v_cmd = rate * ref["comv_rel"][0]`, `w_cmd = rate * (yaw(clock + 0.02) - yaw(clock)) / 0.02`. There is no
  commanded speed or yaw rate independent of the track.

### 4.8 Tasks, episode sampling, floors, pushes

- `TASKS` (`env.py:78-99`): name -> (track plan, stride reference). Nine tasks: `walk`, `walking_turn_left/right`
  (`walk_0.35_*`), `in_place_turn_left/right` (`in_place_0.50_*`), `brisk`, `fr05`, `stand` (`[("hold", 12.0),
  ("transition", "start")]`, length `STAND_SECONDS = 10.0`), `carpet_walk` (reference `carpet`, friction 0.54 by
  `TASK_FRICTION`, `env.py:103`). **Reference and transition names are literals here.**
- Built at construction: `tracks[t] = control.build_track(p, run_dir, TASKS[t][0], omega)` (`env.py:384`),
  `period[t]` from `refs/<TASKS[t][1]>.npz` (`env.py:385`), `omega` from `refs/nominal.npz` (`env.py:380`).
  `run_dir = cfg.run_dir or tune.RUN / cfg.body` (`env.py:359`).
- `EnvConfig` (`env.py:216-244`), all fields with defaults:
  `body`, `floor="hard"`, `pile_grid=0.008`, `run_dir=None`, `gains="tuned"`, `tune_json=None`,
  `tasks=tuple(TASKS)`, `task_weights=None`, `rate=1.05`, `max_seconds=30.0`, `randomize=True`,
  `friction=(0.54, 1.25)`, `mass=(0.9, 1.1)`, `kp=(0.8, 1.2)`, `latency_steps=(0, 10)`, `noise_scale=(0.5, 1.5)`,
  `push_prob=0.5`, `push_frac=(0.0, 1.0)`, `push_s=0.1`, `start_joint_noise=0.01`,
  `start_root_velocity_noise=0.02`. `describe()` = `asdict`. There is no field for reward weights.
- `reset(seed=None, options=None)` (`env.py:470-510`); `options` keys: `task`, `factors` (overrides of
  `friction, mass, kp, latency_steps, noise_scale`), `push` (a tuple or None), `episode_seed`.
  Task drawn with `rng.choice(tasks, p=task_p)`; `seconds = min(track.t_end / rate, max_seconds)`;
  `n_steps = round(seconds / dt_env)`.
- Episode seeds (`EPISODE_SEED_RULE`, `env.py:118-123`, `488-497`): `ep = 1_000_000 * reset_seed + k`; start noise
  seed `SeedSequence([ep, 0])`, sensor noise seed `SeedSequence([ep, 1])`; task, factors and push from the env's own
  generator.
- Floors: one per env (`EnvConfig.floor`); `FLOOR_SPREAD = ("hard", "hard", "robot_floor_carpet", "low")`
  (`env.py:110`), assigned `i % 4` by `train.env_configs` (`train.py:289-302`); training pile grid 0.008 m.
  STAGE2_PLAN B.1 wants hard on three envs in four, robot_floor_carpet on one, no low pile.
- Pushes (`_draw_push`, `env.py:460-468`): with probability `push_prob`, if the episode is longer than 4 s; time
  uniform in `[2, seconds - 2]`; `dv = uniform(push_frac) * PUSH_BOUND_M_S.get(cfg.body, 0.5)`; direction uniform;
  `F = true mass * dv / push_s`. Tuple `(t0, push_s, Fx, Fy)`.
  **Hazard:** `PUSH_BOUND_M_S` (`env.py:257-272`) lists only the three probe bodies; `.get(body, 0.5)` reads only
  the list, so `-K` and `-X` bodies silently get 0.5 m/s, while `PUSH_BOUND_M_S[body]` (used by `panels.env_values`)
  reads the body's envelope screen.
- Termination: `terminated = fall is not None`; `truncated = k_env >= n_steps` (`env.py:560-561`).

### 4.9 Randomisation hooks

- `_draw_factors(rng)` (`env.py:437-445`) -> `dict(friction, mass, kp, latency_steps, noise_scale)`.
- `_apply_factors(f)` (`env.py:447-458`): restores `geom_friction`, `actuator_gainprm[:, 0]`,
  `actuator_biasprm[:, 1]`; `set_trunk_mass`; `control.set_friction`; kp factor on gain and bias.
- `set_trunk_mass(m, trunk, factor, nominal, scratch)` (`env.py:317-325`, `MASS_RULE` `env.py:126-133`): trunk mass
  and inertia x factor, then `mujoco.mj_setConst(m, scratch)`. The env checks at build that `mj_setConst` leaves the
  nominal model unchanged (`setconst_changes`, `env.py:310-314`, `368-371`).
- Noise scale: `tk.noise = SensorNoise(scale=...)` (`env.py:716`).
- Not randomised today: joint damping and frictionloss, gaze or jaw events ("not found"). `Tracker.damp` / `.fric`
  are copied at construction (`control.py:1199-1200`), so a later change to `m.dof_damping` is a model error the
  feedforward does not follow.

### 4.10 G6a's check and its digest

- `_state_digest(d)` (`env.py:812-815`): concatenated sha256 of `d.qpos`, `d.qvel`, `d.ctrl`, `d.sensordata` bytes.
- `zero_residual_check(body, task="walk", seconds=4.0, episode_seed=7, run_dir=None, gains="tuned", tune_json=None,
  factors=None, floor="hard") -> dict` (`env.py:837-884`), `ZERO_RESIDUAL_RULE` (`env.py:804-809`). Runs
  `control.run` with an R2 tracker on the env's own plant, then the env with a zero action on the same plant, seeds
  and gains, and compares the digest after every physics step. `factors` may hold only `mass, kp, friction`
  ("latency and the noise scale are R3's alone and are not compared", `env.py:840-845`).
  Result keys: `rule, body, task, floor, seconds, episode_seed, factors, model_rule, reference_steps,
  reference_fell, reference_fall_cause, reference_fall_time_s, fall_env_step, physics_steps, env_steps,
  r3_terminated, r3_fall_cause, compared_steps, first_difference, ends_alike, identical, gains_info`.
- `throughput(mode, body, env_steps=500, seed=0, run_dir=None, config=None, policy=None)` (`env.py:887-908`).

---

## 5. `designlab/stage1/policy.py` (167 lines)

- `DESIGN2` (`policy.py:35-36`): `net_arch=(256, 256), n_envs=4, n_steps=2048, batch_size=256, n_epochs=5,
  gamma=0.99, gae_lambda=0.95, clip_range=0.2, lr_start=1e-4, lr_end=3e-5, target_kl=0.03, clip_obs=10.0,
  clip_reward=50.0`. **`target_kl` lives here** and reaches PPO through `model_kwargs` (`policy.py:162`).
- `ACTION_NOISE_SIGMA = 0.02`, `LOG_STD_INIT = math.log(0.02)` (-3.912) (`policy.py:40-41`);
  `ACTION_NOISE_RECORD = "stage1/runs/2026-11-08/ppo_action_noise.json"` (`:42`); **`LOG_STD_RULE`**
  (`policy.py:43-49`).
- `LR_START, LR_END = 1e-5, 3e-6` (`policy.py:53`), `LR_RULE` (`:55-62`), `LR_PROBE_RECORD` (`:54`).
- `PrivilegedMlpExtractor(feature_dim, net_arch, activation_fn, device="auto", *, actor_dim)` (`policy.py:69-103`);
  `PrivilegedCriticPolicy(observation_space, action_space, lr_schedule, *args, actor_dim, **kwargs)`
  (`policy.py:106-141`): the actor reads `features[..., :actor_dim]`, the critic all; `actor_dim` is saved with the
  constructor parameters (`:126-129`); `obs_to_tensor` pads actor-only observations with zeros (`:131-141`).
  Generic in every width.
- **`linear_schedule(start, end)` (`policy.py:144-151`) returns the nested function `schedule`, a closure.**
  `model_kwargs` passes `learning_rate=linear_schedule(LR_START, LR_END)` (`policy.py:162`), so every `model_*.zip`
  pickles it. Loading under another Python minor version segfaults (`LESSONS_LEARNED.md:596-602`). The repository's
  safe loader is `environments/shared/policy_loading.load_sb3_model(model_path, *, algorithm=None, env=None,
  device=None, custom_objects=None, **load_kwargs)` (`/home/user/mesozoic-labs/environments/shared/policy_loading.py:501`);
  neither `train.py:497` nor `policy_rows.py:601` uses it (both plain `PPO.load`).
- `model_kwargs(actor_dim, *, net_arch=DESIGN2["net_arch"], log_std_init=LOG_STD_INIT) -> dict` (`policy.py:154-162`).

---

## 6. `designlab/stage1/train.py` (714 lines)

### 6.1 CLI (`designlab/stage1/__main__.py:528-561`, dispatch `:357-391`, `:696`)

```
python -m designlab.stage1 train --body B --seed N [--mode R3|R4] [--steps N] [--resume] [--n-envs N]
       [--vec dummy|subproc] [--checkpoint-every 250000] [--no-g6] [--log-std-init X] [--allow-untuned]
       [--allow-source-change] [--fix-task-weights w1,...] [--fix-reason TEXT] [--tune-json PATH]
       [--out stage1/runs/2026-11-08]
python -m designlab.stage1 judge [--mode R3|R4] [--budget 5000000] [--final] [--out DIR]
python -m designlab.stage1 noise [--body 75-M-H-10] [--tune-json PATH] [--out DIR]
```

- `--body` choices are all of `BODIES`; `--mode` choices are `("R3", "R4")` (`__main__.py:531`).
- `--steps` default `train.BUDGET_ASKED` (4,000,000), rounded up to whole rollouts.
- `--fix-task-weights` must have exactly `len(env.TASKS)` values (`__main__.py:379-381`).
- `judge --budget` defaults to 5,000,000 (`__main__.py:555`), not `BUDGET_STEPS`.
- The CLI cannot pass `config` (beyond `tune_json`) or `ppo` overrides; those exist only on the function.

### 6.2 `train.train` signature

```python
train(body, seed, mode="R3", steps=BUDGET_STEPS, out=None, resume=False, n_envs=None, vec="dummy",
      checkpoint_every=CHECKPOINT_EVERY, g6=True, config=None, log_std_init=None, ppo=None, allow_untuned=False,
      allow_source_change=False, fix=None, fix_reason=None, log=print) -> dict            # train.py:377-381
```

`mode not in ("R3", "R4")` raises (`train.py:395-396`). `n_envs = n_envs or DESIGN2["n_envs"]` = 4
(`train.py:400`). `steps = whole_rollouts(steps_asked, n_envs * n_steps)` (`train.py:403`).
`torch.set_num_threads(1)` (`train.py:468`); `device="cpu"` (`train.py:497`, `513`).
Vec env: `DummyVecEnv` (default) or `SubprocVecEnv(fns, start_method="spawn")` (`train.py:476`); `VecNormalize(
norm_obs=True, norm_reward=True, clip_obs=10, clip_reward=50, gamma=0.99)` (`train.py:509-510`).

### 6.3 Run directory layout

`run_dir_of(out, body, mode, seed)` (`train.py:158-159`) =
`<out>/<body>/("ppo" if mode == "R3" else "policy_only")/<seed>/`. **Any mode other than R3 lands in
`policy_only`.**

```
<out>/<body>/ppo/<seed>/run.json
<out>/<body>/ppo/<seed>/checkpoints/model_<steps>.zip
<out>/<body>/ppo/<seed>/checkpoints/vecnormalize_<steps>.pkl
```

- **`STATUS.json`: not found in any code.** It appears only in `STAGE2_PLAN.md:225`, `LESSONS_LEARNED.md:482` and
  `stage2/research/research_pipeline.md` as something the notebook is to write.
- `checkpoints(run_dir)` (`train.py:186-194`): step counts for which **both files exist**. Existence only; no size
  or hash check (the plan's "pair check on resume" is not built).
- Checkpoints are written at the first rollout start at or past each multiple of `checkpoint_every`, and once after
  `learn()` returns (`CHECKPOINT_RULE`, `train.py:81-84`; `_Callback._save`, `train.py:551-558`; `:629-630`). Order
  in `_save`: model, then VecNormalize, then `run.json` (atomic via `.tmp` + `os.replace`, `train.py:179-183`).
- Checkpoints are git-ignored.

### 6.4 `run.json` (schema 3; keys as in `stage1/runs/2026-11-08/75-M-H-10/ppo/42/run.json`)

Top: `schema, body, seed, mode, settings, env_configs, gains, final_rule, judged_rule, g6_rule, seed_rule,
budget_rule, sessions, checkpoints, throughput, g6, status, budget, budget_asked, num_timesteps[, fix, stopped,
judged_checkpoint]` (`train.py:514-517`, `555-556`, `639-640`).

- `settings` (`train.py:421-425`): `body, seed, mode, n_envs, design2 (the whole DESIGN2 dict), learning_rate{start,
  end, schedule, rule}, log_std_init, ppo_overrides, config, checkpoint_every, checkpoint_rule, floor_rule,
  seed_rule`. **Four rule strings are inside it** (`LR_RULE`, `CHECKPOINT_RULE`, `env.FLOOR_RULE`, `SEED_RULE`).
- `env_configs`: one `EnvConfig.describe()` per env, **including `run_dir`** (the committed run holds the relative
  string `'stage1/runs/2026-11-08/75-M-H-10'`) and `floor`.
- `gains` (R3 only, `train.py:445-448`): `values, candidate, cma_seed, status, flag, source, rule, tune_json,
  tune_json_sha256, allow_untuned`.
- `sessions[k]` (`train.py:481-485`, `533`): `index, started_at, provenance, versions, ppo_venv, matches_ppo_venv,
  vec, env_processes, n_envs, torch_threads, solve2, gains, env (env 0's describe()), other_envs (diffs), floors,
  seed_base, env_seeds, episode_streams[, resumed_from, source_change, fix, ended_at]`.
- `throughput[]`: `first_rollouts?, session, env_steps, n_envs, vec, env_processes, wall_s, cpu_s_main,
  cpu_s_env_workers, cpu_s, env_steps_per_wall_s, env_steps_per_cpu_s, load_average[, cores]`.
- `g6`: `G6a_zero_residual` (the `zero_residual_check` dict), `G6a_rate`, `G6b`, `G6b_after_fix`, `G6c`
  (each `at_steps, pass_, consequence_if_failed, monitor`).
- `status`: `"running"`, `"done"`, `"partial"`, `"stopped_g6b"` (`train.py:516`, `639`).

### 6.5 Resume checks (all `SystemExit`; `train.py:426-465`, `486-498`)

| Check | Line | Refuses when |
|---|---|---|
| run exists, no `--resume` | `426-427` | `run.json` present |
| stopped run | `431-432` | `doc["stopped"]` set |
| **settings** | `433-434` | `doc["settings"] != settings` (dict equality: `n_envs`, `design2`, `log_std_init`, `ppo_overrides`, `config`, `checkpoint_every`, and the four rule strings) |
| **env configs** | `435-436` | `doc["env_configs"] != cfgs` (every field of every env, **`run_dir` included**) |
| gains (R3) | `451-456` | `values, candidate, cma_seed, status` changed |
| **versions** | `457-460` | `sessions[0]["versions"] != _versions()` (exact dict; no override flag exists) |
| **source hash** | `461-465` | `sessions[0].provenance.stage1_sources.sha256 != plant.provenance()[...]`, unless `--allow-source-change` (then `session["source_change"] = dict(allowed=True, first_session, this_session)`, `train.py:491-494`) |
| checkpoint | `487-490` | no complete pair |

- Resume loads `VecNormalize.load(...)` then `PPO.load(str(model), env=vecnorm, device="cpu")` (`train.py:496-497`)
  from the newest pair; `learn(total_timesteps=remaining, reset_num_timesteps=False)` (`train.py:626`).
- Seeds (`SEED_RULE`, `train.py:132-139`; `env_seeds`, `train.py:280-286`): session k of PPO seed s resets env i
  with `10_000 s + 100 k + i`, via `model.set_random_seed(seed_base)` (`train.py:528`) and a check of
  `venv._seeds` (`train.py:529-532`). A resumed run is therefore not the uninterrupted run bit for bit.
- The committed 75-M-H-10 R3/42 run has 3 sessions, two resumed with `--allow-source-change`.
- **A resume across library versions is not possible** (no flag). STAGE2_PLAN B.4 asks for one.

### 6.6 Versions: `torch.__version__` vs `importlib.metadata`

- `train._versions()` (`train.py:162-172`) records the imported modules' own `__version__`:
  `python, numpy, mujoco, scipy, torch, stable_baselines3, gymnasium`.
- `policy_rows.library_versions()` (`policy_rows.py:398-409`) reads `importlib.metadata.version(name.replace("_",
  "-"))` for `RUN_VERSIONS[1:]` (`policy_rows.py:78`).
- `policy_rows.check_versions(run, where)` (`policy_rows.py:412-419`) compares the two key by key and raises
  `PolicyRowError("...the libraries differ from the run's (...): run the policy rows in the PPO venv")`. Called in
  `resolve` (`policy_rows.py:380`) and `TrainedPolicy.load` (`policy_rows.py:586`).
- Measured in this venv: both read `torch 2.15.0+cpu` (a download.pytorch.org wheel), so they agree here. With
  PyPI's default build, metadata reads `2.15.0` and `torch.__version__` `2.15.0+cu...`, so a run would be refused by
  the policy rows in its own venv (`LESSONS_LEARNED.md:579-588`).
- `plant.provenance()` reads torch a third way, also `importlib.metadata`, under the key `"stable-baselines3"`
  (`plant.py:299-304`).
- This venv is `torch 2.15.0+cpu, stable_baselines3 2.9.2, gymnasium 1.4.0`. `train.PPO_VENV` (`train.py:109`) is
  `torch 2.9.0+cpu, stable_baselines3 2.9.0, gymnasium 1.3.0` (`tools/requirements-stage1-ppo.txt`), the versions
  every committed R3/R4 run recorded. So Stage 1's existing checkpoints are refused in this venv, and a session here
  records `matches_ppo_venv: False` (recorded only, never enforced, `train.py:482`).

### 6.7 Monitor and G6 (`G6_RULE`, `train.py:140-155`)

- Constants: `SEEDS = (42, 43)`, `R4_SEEDS = (42,)` (`train.py:85-86`), `G6A_ENV_STEPS_PER_S = 600`,
  `G6A_MEASURE_STEPS = 16_384`, `G6A_CHECK = dict(task="walk", seconds=30.0, episode_seed=7)`,
  `G6B_STEPS = 1_000_000`, `G6C_STEPS = 2_500_000`, `G6C_MIN_GAIN = 0.01`,
  `G6B_TOLERANCE_S = ENV_SUBSTEPS * 0.002` (`train.py:87-95`), `ROLLOUT_STEPS = 4 * 2048` (`train.py:66`),
  `BUDGET_ASKED = 4_000_000`, `BUDGET_STEPS = 4_005_888` (`train.py:76-77`), `CHECKPOINT_EVERY = 250_000` (`:64`).
- `MONITOR` (`train.py:113-116`): 8 conditions `(task, floor)` x episode seeds `(5001, 5002, 5003)`, nominal plant,
  no push, **latency 0**.
- `monitor(model, vecnorm, body, mode, config) -> dict(rows, summary, mean_survival_s, g6c_by_condition,
  policy_beats_zero_on)` (`train.py:338-374`): for each condition builds a fresh env, runs the deterministic policy
  and the zero action.
- **G6a** (R3 only, before training, `train.py:519-523`): `env.zero_residual_check(body, run_dir, gains, tune_json,
  **G6A_CHECK)`; and the rate over the first 16,384 steps against 600 env steps per wall second
  (`train.py:602-619`). The committed run measured 74 steps/s (failed; decision 23 cut the budget).
- **G6b** at the first rollout start at or past 1M steps: `g6b_pass` (`train.py:98-100`), R3's mean survival at
  least R2's minus one env step. **G6c** at 2.5M: `beats_r2` (`train.py:323-335`).
- **FIX_RULE** (`train.py:102-107`): one fix, `FIX_FIELDS = frozenset({"task_weights"})`, on resume only;
  `G6b_after_fix` at the fix's checkpoint + 1M; a second failure sets `doc["stopped"]` and halts
  (`train.py:581-594`).
- All G6 monitoring is guarded by `mode == "R3"` (`train.py:519`, `568`, `582`). R4 gets only `G6a_rate`.
- Judging: `expected_runs(out, mode)` (`train.py:208-212`) = `PROBE_BODIES` x seeds only;
  `judged_with_r2`, `judged_checkpoint`, `judge` (`train.py:223-277`).
- `action_noise_survival(body="75-M-H-10", sigmas=NOISE_SIGMAS, episodes=NOISE_EPISODES, run_dir=None,
  tune_json=None, log=print)` (`train.py:664-708`, `NOISE_RULE` `:654-661`): the measurement behind `LOG_STD_RULE`
  (18 of 20 nominal walks survive under action noise sigma).

---

## 7. `designlab/stage1/policy_rows.py` (1223 lines)

- Registers `R3` and `R4` with `panels.register_rows` at import (`policy_rows.py:823-825`); CLI
  `python -m designlab.stage1.policy_rows {verdicts|check|costs}` (`policy_rows.py:1154-1219`).
- Environment variables: `STAGE1_POLICY_CHECKPOINT` (`judged`, `newest`, a step count, or `R3=...,R4=...`),
  `STAGE1_POLICY_RUNS` (`policy_rows.py:70-71`, `227-253`).
- `resolve(body, mode, seed, run_dir, choice=None) -> dict(mode, body, seed, steps, selection, final, judged, root,
  files{model{file, sha256}, vecnormalize{file, sha256}}, run)` (`policy_rows.py:321-385`). Refuses a run whose
  `rate != panels.RATE`, whose start noise differs from tune's, whose tasks are unknown to `env.TASKS`
  (`policy_rows.py:366-379`).
- `ResidualView` (`policy_rows.py:474-489`) and `PolicyOnlyTracker(control.Tracker)` (`policy_rows.py:492-541`)
  **borrow env.py's unbound methods** (`env.Stage1Env._observation`, `_command`, `_true_dcm_error`,
  `ResidualEnv._estimate/_clock/_apply_action`, `PolicyOnlyEnv._begin/_physics_step/...`) and feed them the
  attributes set by `episode_attrs` (`policy_rows.py:453-471`): `nj, dt, dt_env, H, weight, omega, _R_home, cfg,
  task, period, factors, push`. A new attribute read by `_observation` must be added there too.
- `TrainedPolicy(panels.Policy)` (`policy_rows.py:547-660`): `load()` checks file sha256, then plain `PPO.load` and
  `pickle.load` of the VecNormalize (`:601-603`); `action(obs)` = `model.predict(vecnorm.normalize_obs(obs[None]),
  deterministic=True)`.
- `equivalence_check(body, row="R3/42", condition="walk/nominal", seed=3042, run_dir=None, seconds=None,
  choice=None, variant=None, relax=(), overrides=None)` (`policy_rows.py:872-979`), `EQUIVALENCE_RULE`
  (`:169-188`): a panel-row episode against the training env, digest per physics step, incl. the ledger's
  `latency_steps` variant (`VARIANT_FACTORS`, `:869`). This is the existing template for "an env episode equals a
  panels episode".

---

## 8. `designlab/stage1/gates.py` and the other gates

`gates.py` holds **G1-G4 only**. G5 is `tune.g5` (`tune.py:1442`), G6 is in `train.py` (§6.7), G7 is `panels.g7`
(`panels.py:2160-2251`). G8a, G8b, G9: **not found in code** (only `STAGE2_PLAN.md:118-123`, `243-246`).

| Gate | Function | What it checks |
|---|---|---|
| **G1** plant | `g1(body, out_dir=None, **plant_kw)` (`gates.py:377-396`) | model hashes match record and plan (`plant.verify`); zero-command stand 20 s on hard and low pile with tilt below 12 deg (`stand`, `:151-188`); compiled tendon map equals `sizing._motor_space` to 1e-9 and the ankle reaches but never exceeds 2 x peak (`tendon_map`, `:191-244`); seeded runs repeat bit for bit (`repeatability`, `:262-271`); at least 20,000 physics steps/s per core (`step_rate`, `:352-364`; provisional if fewer processes than cores) |
| **G2** references | `g2(body, out_dir, strides=60, rule="panel", procs=None, **plant_kw)` (`gates.py:638-712`) | every reference matches its records to 6 digits; the R0 replay (`replay`, `:406-537`) of the nominal walk walks at least 10 strides, judged on the median of 20 starts offset by up to 1 mm/s |
| **G3** transitions | `g3(body, out_dir)` (`gates.py:718-744`) | from `transitions.json`: each transition's last stride within `G3_COM_MM` / `G3_ZMP_MM` of the periodic walk, every frame's IK solved; **`missing = [n for n in tr.TRANSITIONS if n not in rows]`** (`:734`) fails the gate if any name in `transitions.TRANSITIONS` is absent |
| **G4** tracker | `g4(body, out_dir, procs=None, attribution=True)` (`gates.py:950-1082`) | at the tuning start: R1 walks at least 40 strides on hard floor and 20 at mu 0.54, R2 at least 20, by the 20-start panel's median; failing conditions attributed (`attribute`, `:919-947`, sign test p < 0.05). Does not apply to `-X` (`__init__.py:58-59`) |
| **G5** tuner | `tune.g5` | selected gains succeed on at least 17 of 20 validation episodes per condition (`tune.py:120-121`) |
| **G6** PPO | `train.py` | G6a zero residual = R2 exactly and 600 env steps/s; G6b at 1M; G6c at 2.5M |
| **G7** panels | `panels.g7` | the metrics on the references reproduce Stage 0's sway and cadence (`STAGE1_PLAN.md:75`); only the three probe bodies have figures (`speedfit.py:251-254`) |

`gates.replay` (`gates.py:406-537`) is the original R0: `p.set_state(ref.q[0], qd0)`, clock `d.time`,
`ff = tau + damp * qd + fric * tanh(qd / 0.05) [+ extra @ qdd]`, rate 1.

---

## 9. Tests that pin behaviour

`tests/conftest.py` (13 lines): `os.environ.setdefault("MUJOCO_GL", "disable")`.

`tests/test_stage1_ppo.py` (body `60-N-T-10`, run `stage1/runs/2026-11-08`; env tests skip without the generated
refs and transitions):

| Test (line) | Asserts |
|---|---|
| `test_observation_layout_puts_the_actor_first` (67) | `actor_dim == ACTOR_DIM`, blocks contiguous from 0, the critic names present |
| `test_r3_steps_at_50_hz_and_returns_finite_values` (78) | obs shape `(OBS_DIM,)` float32; one step advances `10 * dt`; reward terms in [0, 1] |
| `test_zero_residual_reproduces_r2_exactly` (89) | 1 s walk: `identical`, 500 physics steps compared |
| `test_zero_residual_check_compares_to_r2s_fall_and_its_env_step` (97) | on robot_floor_carpet R2 falls; R3 matches to the fall step and terminates in the env step holding it |
| `test_a_heavier_or_lighter_trunk_keeps_the_plant_consistent_and_r3_is_r2_on_it` (108) | mass 1.1 / 0.9: model equals an independently recompiled one; only `MASS_FIELDS` differ; tracker keeps the nominal model; zero residual identical |
| `test_a_residual_moves_the_servo_targets` (139) | `tk.residual[left_knee] == alpha * RESIDUAL_LEG_RAD` after one step of action 1 |
| `test_the_rate_residual_scales_the_reference_clock` (154) | `rate_factor == 1 + RESIDUAL_RATE * alpha` |
| `test_randomisation_is_drawn_per_episode_and_restored` (164) | factors land in the plant; the next nominal episode's model is nominal bit for bit |
| `test_the_carpet_walk_runs_the_carpet_swing_at_mu_054` (183) | friction 0.54; segments `start, carpet, stop`; phase period = nominal's |
| `test_a_pile_env_steps_and_its_feet_skip_the_probes` (196) | `env.foot_tables` equals `metrics._lowest_point_tables` on hard floor; floor code 2 on a pile |
| `test_latency_applies_the_command_of_steps_ago` (212) | §1.5 |
| `test_episode_seeds_follow_the_reset_seed` (230) | `EPISODE_SEED_RULE` |
| `test_policy_only_env_sets_targets_within_the_ranges` (243) | targets within ranges, span at least 0.10 rad |
| `test_tuned_gains_are_the_body_level_g5_selection` (269) | `GAINS_RULE` cases |
| `test_the_actor_reads_its_columns_only_and_the_critic_all` (316), `test_predict_takes_the_actors_columns_alone` (327), `test_the_actor_gets_no_gradient_from_the_privileged_inputs` (338), `test_a_saved_model_keeps_its_actor_width` (345) | the privileged critic |
| `test_log_std_init_is_the_measured_noise` (360) | `ACTION_NOISE_SIGMA` equals the record's; `model_kwargs(4)` carries `LOG_STD_INIT` |
| `test_the_budget_is_4m_steps_in_whole_rollouts` (421) | `BUDGET_STEPS == 4_005_888`; divisible by `n_envs * n_steps` |
| `test_the_learning_rate_is_design_2s_x_01` (427) | `(LR_START, LR_END) == (1e-5, 3e-6)`, `target_kl == 0.03`, schedule endpoints |
| `test_g6c_needs_fewer_falls_or_a_paired_gain_beyond_the_margin` (439), `test_the_monitor_runs_each_condition_on_its_floor` (455) | G6c, monitor |
| `test_env_configs_spread_the_floors` (484) | floors `hard, hard, robot_floor_carpet, low, hard, hard`; `pile_grid == 0.008`; `kp == (0.8, 1.2)` |
| `test_env_seeds_are_disjoint_across_seeds_and_sessions` (492) | `SEED_RULE` |
| `test_r3_is_refused_without_a_g5_pass` (504) | `--allow-untuned` |
| `test_training_checkpoints_and_resumes` (515) | checkpoints `[16, 32]`; `run_dir` strings in `env_configs`; resume refused on other settings, on other sources (allowed and flagged with the flag), on changed gains |
| `test_g6bs_one_fix_changes_the_task_weights_and_stops_on_a_second_failure` (582), `test_a_fixed_run_keeps_its_fix_on_later_resumes` (620) | `FIX_RULE` |
| `test_two_ppo_seeds_and_a_resume_draw_disjoint_episode_seeds` (637), `test_subproc_envs_report_their_processes_and_cpu` (674) | seeds, subproc |
| `test_cli_trains_and_records` (687) | CLI argument mapping; `tune.RUN.name == "2026-11-08"` |

`tests/test_stage1_policy_rows.py` (needs checkpoints; most tests skip without them): registration leaves the
panels' and tuner's source sets unchanged (91); checkpoint choice (104, 143, 177, 203); rows name checkpoint, run
and sources, and their source set is exactly `{env.py, policy_rows.py, policy.py}` (232); gains must equal the run's
to the last bit (267); a changed file is refused (280); `PERIOD_RULE` (291); **R3 and R4 rows equal the training env
step for step** on the walk, the carpet walk on low pile, under `mass=1.1`, `kp=0.8`, `latency_steps=3`, on pushes
and with relaxations (313-406); sensitivity checks (386, 414, 463).

`tests/test_stage2_x.py:629-637` `test_the_env_refuses_an_x_plant_until_its_widths_are_generalised`: asserts
`env.WIDTH_JOINTS == 14 and env.ACTION_DIM == env.WIDTH_JOINTS + 1` and that `env.Stage1Env(EnvConfig(body=
"60-N-T-10-X", ...))` raises `ValueError` matching `"pending"`. **This test must change when the widths are
generalised.** `tests/test_stage2_x.py:622-626` checks `Tracker.leg_act` on `-X` leaves out exactly
`sizing.SINGLE_GROUPS`.

---

## 10. What an R5 mode on `-X` bodies needs generalised (precise list)

Plan text: `STAGE2_PLAN.md:93-113`. R5 = `control.Tracker` in R2 mode, `terms=()`, `ankle_cap=()`; 50 Hz joint
offsets of +-0.35 rad (legs), +-0.15 rad (head and tail), +-25 % clock rate; feedforward kept.

1. **Widths per body.** `ACTOR_OBS`, `CRITIC_OBS`, `ACTION_DIM`, `ACTOR_DIM`, `OBS_DIM`, `obs_layout()`,
   `WIDTH_JOINTS` and the refusal (§4.3). On `-X`: nj 16, so `q`, `qd`, `joint_error` 16 wide and the action 17.
   Every consumer of the module constants in §4.3 (`train.py:355, 511, 683`; `policy_rows.py:572-578, 607-613,
   644-645`).
2. **New actor inputs:** the last three actions (today one, `env.py:139`, `499`, `559`) and, on `-X`, the head
   command. Neither exists.
3. **Commands as inputs.** `_command` derives speed and yaw rate from the reference (`env.py:613-620`); R5 is
   "commanded by speed and yaw rate". The phase input needs a period per task (`env.py:385`, `629`;
   `policy_rows.PERIOD_RULE`).
4. **Action scales.** `RESIDUAL_LEG_RAD 0.2`, `RESIDUAL_NECK_TAIL_RAD 0.1` (`env.py:65-66`) vs 0.35 / 0.15.
   `RESIDUAL_RATE 0.25` already matches, **and the panels hold it by value** (§0.1).
5. **The controller under it.** `ResidualEnv._setup` hard-codes `control.Tracker(..., "R2", gains=self.gains, ...)`
   with default terms and cap (`env.py:708-709`); `_gains` calls `tuned_gains` (`env.py:408-414`). R5 needs
   `terms=()`, `ankle_cap=()` and no gains. `train.train` refuses R3 without a G5 pass (`train.py:439-448`) and
   compares `describes[i]["gains"]` (`train.py:478-480`).
6. **Tasks and references.** `TASKS` literals (`env.py:78-99`), `omega` from `refs/nominal.npz` (`env.py:380`),
   `TASK_FRICTION`. New look references, 0.30-0.50 m/s walks, 0.35-1.0 rad/s turns, two-step stops and changes need
   entries; `--fix-task-weights` counts `len(env.TASKS)` (`__main__.py:380`); `policy_rows.resolve` refuses tasks
   outside `env.TASKS` (`policy_rows.py:372-373`); `control.Track.add_transition` finds a stop only through
   `transitions.TRANSITIONS[tr.name]` (`control.py:994`).
7. **Mid-task starts** ("half start mid-task (`Plant.set_state`, `Estimator.reset`, the delay queue filled from the
   reference)"). Missing pieces: `Tracker.reset` always sets `tau_ref = 0` and starts from the track's first frame
   or the stance (`control.py:1292-1303`); a `Track` keeps full `qpos`/`qvel` only for its first frame
   (`control.py:963`, `1021`) and `Segment.S` holds joint arrays only, so the root pose at a mid-track time is not
   in the track; `_latency_on` fills the queue from `p.d.ctrl`, which `set_state` leaves at home (§1.5).
8. **Reward.** New terms (foot and contact imitation, head attitude in the world, action rate, neck and tail torque
   over their limit) and "weights recorded in the env config": `EnvConfig` has no weight fields; `PENALTIES`,
   `IMITATION_W`, `TRACKING_W` are module constants written into `describe()` (`env.py:686-687`). The imitation mean
   runs over all nj joints (`env.py:578`).
9. **Randomisation.** Ranges are `EnvConfig` fields (changeable without code): delay 0-25 ms needs a whole-step
   choice (12 = 24 ms or 13 = 26 ms); stiffness 0.6-1.2; friction 0.4-1.2. Not built: joint damping and
   frictionloss x 0.5-1.5; pushes split by walking / standing fraction (`push_frac` is one range); gaze and jaw
   events. The critic's `plant_factors` block is 3 wide with the delay scaled by 0.02 s (`env.py:658`).
10. **Floors.** `FLOOR_SPREAD` and `FLOOR_RULE` (`env.py:110-115`) give low pile to one env in four; R5 wants none.
    `FLOOR_RULE` is inside `settings` (§6.4).
11. **Push bound.** `PUSH_BOUND_M_S.get(cfg.body, 0.5)` (`env.py:465`) for unlisted bodies (§4.8).
12. **`train.py`.** Mode check (`train.py:395`); `run_dir_of` (`:158-159`); all G6 logic under `mode == "R3"`;
    `MONITOR` at latency 0 with Stage 1's tasks (STAGE2_PLAN §6's monitor is 18 episodes at 10 ms every 2.5M
    steps); `expected_runs` over `PROBE_BODIES` (`:208-212`); `n_envs` default 4 and `ROLLOUT_STEPS = 4 * 2048`
    (plan: 8 envs); `LOG_STD_INIT` and `LR_START/END` are module constants in `policy.py` (plan: sigma 0.1, a probed
    rate); `FIX_FIELDS`; no `MUJOCO_GL`; no cross-version resume; no fine-tune from another run's checkpoint; the
    closure schedule; `tools/memguard.sh:11` protects only `designlab.stage1 train`.
13. **Evaluation rows.** `policy_rows` is built around R3/R4 (`MODES = ("R3", "R4")`, `policy_rows.py:72`;
    `ROW_FACTORIES`, `:823`); `TrainedPolicy.load` checks the widths in §4.3; `episode_attrs` fixes
    `latency_steps=0` in the view's factors (`policy_rows.py:470`), while the bar runs at 10 ms.
14. **Noise streams.** `NoisySensors` draws `8 + nj` normals per step (`estimator.py:167`): `-X` episodes are not
    noise-paired with the plain body's on the same seed.
15. **Tests.** `tests/test_stage2_x.py:629-637` pins the refusal.

Already generic and needing no change for 16 hinges: `plant.build`, `Plant`, `drive.motor_matrix`, `Estimator`,
`NoisySensors`, `control.Track`, `control.Tracker` (name-indexed; `leg_drives` knows `neck_yaw` and `jaw`),
`PrivilegedCriticPolicy`.

---

## 11. What G8a would compare

Plan: "G8a (R5's zero action reproduces the panels' R0 exactly, at 0 and 10 ms)" (`STAGE2_PLAN.md:118-119`);
"G8a fails: no run" (`:243`). No code exists. From the existing pieces:

- **Reference side: the panels' R0 episode** (§3.4):
  `control.Tracker(p, track, "R1", gains=None, terms=(), ankle_cap=(), rate=1.05)` under
  `control.run(tk, seconds, start="stance", seed=start_seed, noise_seed=noise_seed, joint_noise=0.01,
  root_velocity_noise=0.02, push=..., relax=())`.
- **At 10 ms:** the same episode inside `speedfit.servo_delay(p, 5)` (5 physics steps). **No R0 row with a delay
  exists**: `speedfit.DelayedRow` is R2 only (`speedfit.py:1081-1112`, `super().__init__(name, MODE, gains, ...)`,
  `MODE = tune.MODE`, `speedfit.py:91`),
  and `zero_residual_check` rejects a latency factor (`env.py:843-845`).
- **R5 side:** the R5 env stepped with a zero action on the same plant, floor, friction, track, start seed, start
  noise and delay.
- **Compared quantity:** `env._state_digest` (sha256 of `qpos, qvel, ctrl, sensordata`) after every physics step, to
  the reference's end or fall, plus the same end (`ZERO_RESIDUAL_RULE`, `env.py:804-809`; or `EQUIVALENCE_RULE`'s
  shorter-horizon reading, `policy_rows.py:169-188`: `control.run` runs `round(seconds / 2 ms)` physics steps, the
  env `round(seconds / 20 ms)` env steps of 10, so they can differ by up to 5 physics steps, and the env reads falls
  only at env-step ends).
- **Why a zero action can be exact (by reading; no existing test compares an R2-mode no-term tracker with the
  R1-mode R0 row):** in R2 mode with `terms=()`, `ankle_cap=()`, the tracker commands
  `q_ref + zeros`, `1.05 qd_ref`, `ff`, the same expressions as the R1-mode R0 (§3.3). The env must leave
  `tk.residual = None` (as `env.py:724` does: `off if off.any() else None`) and `tk.rate_factor == 1.0` exactly
  (`1.0 + 0.25 * 0.0`), so `control.py:1393-1394` and `1417-1418` are skipped. R2 mode additionally builds sensors
  and an estimator, which only read the plant.
- **Seeds must be aligned by the harness:** the panels derive start and noise seeds with `tune.derived_seed`
  (`panels.py:505-508`), the env with `SeedSequence([ep, 0/1])` (`env.py:496-497`). `policy_rows._aligned`
  (`policy_rows.py:856-864`) shows the existing way (override `_begin`).
- **Delay equivalence:** both sides fill the queue with the post-reset `ctrl` (home) before the first command
  (`env.py:516`; `speedfit.py:341-343`). `tests/test_stage1_policy_rows.py:341-356` already checks this equality
  for R3 at `latency_steps=3` through the ledger's variant.
- **Scope limits to state in the gate's record:** the panels' R0 row is hard-floor only; it is defined on a
  condition's own track (`ctx.built(cond)`), so the env must run that track, not an `env.TASKS` plan, unless the two
  plans are identical (`policy_rows.env_task`, `policy_rows.py:431-438`).

---

## 12. Extension points and hazards (short)

Extension points that exist today:
- `Tracker.residual` / `Tracker.rate_factor` (`control.py:1273-1277`) with `terms=()`, `ankle_cap=()`: R5's
  mechanism needs **no edit to `control.py`**.
- `Stage1Env`'s six hooks (`env.py:417-434`): a subclass in another module can supply R5's behaviour.
- `panels.register_rows(controller, factory)` (`panels.py:865-871`) and `eval --row-module`: new rows without
  touching `panels.py`; a row's `sources()` enter its own keys only.
- `EnvConfig` ranges and `train.train(config=..., ppo=...)` (function arguments, recorded in `settings`).
- `options` of `reset` (`task, factors, push, episode_seed`) and `_step_hook` for digests.

Hazards:
1. **Any edit to `env.py`** changes `train.py`'s resume hash, the speed fit's sources (Stage 1b's fit records then
   read as made by other code) and the R3/R4 rows' keys; six of its values and one function's source text are also
   in every panel key (§0.1). **Any new file in `designlab/stage1/`** changes `train.py`'s resume hash. Code under
   `designlab/stage2/` is in none of these sets, so it needs its own provenance.
2. `control.py`, `estimator.py`, `plant.py` are in the tuner's, the panels' and the speed fit's sources: editing
   them makes R2/42, R2/43 and every earlier panel record read as other code. The panels' key also covers
   `designlab/gait.py` and `designlab/turning.py` (§0.1). New official panels (build item E) are new records
   under whatever the sources are when they run; earlier records then have other keys and are cited, not reused.
3. Rule strings sit inside `run.json` `settings` (`LR_RULE`, `CHECKPOINT_RULE`, `FLOOR_RULE`, `SEED_RULE`): rewording
   one blocks every resume of an older run, with no override.
4. `env_configs` hold `run_dir` as given: keep one path spelling for a run's life (`LESSONS_LEARNED.md:592-593`).
5. Delay is whole 2 ms steps; `set_state` is not covered by any delay wrapper; wrappers do not stack (§1.5).
6. `MUJOCO_GL` must be `disable` before `designlab` or `mujoco` is imported in a training process; nothing in
   `train.py` does it (§0.5).
7. `plant.build` on a clone without round-two models writes into `stage1/models/` and changes every record key.
8. `linear_schedule` is a pickled closure in every archive; both loaders use plain `PPO.load` (§5).
9. torch's version is read three ways (§6.6); `check_versions` has no cross-version path.
10. `checkpoints()` accepts a pair by existence alone; a truncated file from a killed session passes.
11. `PUSH_BOUND_M_S.get(body, 0.5)` silently substitutes 0.5 m/s for `-K` and `-X` bodies in training.
12. The env's clearance penalty and the reward's swing reading use the reference's clock (`env.py:603-606`), the
    reading the Stage 2 bar replaces.
13. `-X` joint order differs (legs from index 4); `NoisySensors`' draw count depends on nj.
14. `gates.g3` fails on any name of `transitions.TRANSITIONS` missing from `transitions.json`, and stop events are
    found by transition name.
