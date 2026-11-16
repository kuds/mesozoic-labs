# Map 03: turn references and the thigh-twist range

Study root: `/home/user/mesozoic-labs/hardware/trex_form_study` (all paths relative to it). Read-only survey made
14 Nov 2026 at commit `7e9ea0d`. Code line numbers are of that commit; no code file changed while I read. Four
documents were edited in the working tree during the survey by another session (`DECISION_LOG.md`,
`STAGE2_PLAN.md`, `LESSONS_LEARNED.md`, `JETSON_PLEM_REVIEW.md`: owner decision P30). `STAGE2_PLAN.md` and
`DECISION_LOG.md` line numbers below are the working tree's after that edit (the plan gained 11 lines at its top).

P30 (`DECISION_LOG.md:45`) matters here in three ways: the 75 cm track comes first, so section 5.3's 75 cm reading
is the first thing to settle; the reference to freeze is V6, every look feature, so a look-aware turn has to carry
F5 and F14 too (section 7); and the 75 cm candidate reference is "F25's tuned walk with look feature F9's
counter-phased tail and a small duty and lift margin", which sets `tail_phase`, a look field, so `turning.generate`
refuses it as well.

Everything marked **[my run]** is a quick kinematic check I ran in this session (single process, nothing written in
the checkout). It is not a record and must not be cited as one; the settings are given so it can be redone properly.

## 0. The five things to know first

1. `designlab/turning.py` builds **one periodic stride of a steady turn** at a constant yaw rate. It has no ramps, no
   entry and no exit (`turning.py:124`: "No transitions: steady turning only"). All ramps live in
   `designlab/stage1/transitions.py` (`enter_*`, `exit_*`, `start_in_place_*`, `stop_in_place_*`).
2. A Stage 1 "walking turn" track is `start, enter_<case>_<dir>, ("ref", <case>_<dir>, 5), exit_<case>_<dir>, stop`.
   The enter and exit transitions alone carry 3 + 2 strides of steady turning in their output windows
   (`transitions.py:109`, `WINDOWS["walk"] = (-2, 4)`, ramp over phase [0, 1]). At pi/4 rad/s from 0.40 m/s one stride
   turns 33.8 deg (60 cm) or 38.7 deg (75 cm), so **the existing enter + periodic + exit construction cannot make a
   90 degree turn**: its minimum is about 5 strides of steady turning plus two ramps.
3. `turning.generate` refuses any look spec (`turning.py:380-382`), and `transitions.gait_of` only knows
   `refs.walk_specs` names and `sizing.NOMINAL` turns (`transitions.py:165-179`). F25's tuned walk changes only
   `lift_frac`, `duty`, `stride_rel`, `crouch` (`designlab/gaittune.py:45-48`), none of them a look field, so it passes
   the guard as it is. With F9's tail phase added (P30's candidate) it does not: `tail_phase` is in `LOOK_FIELDS`.
4. Thigh twist is `<side>_hip_yaw`, range +-0.30 rad in the MJCF and in the plant (`plant.usable_ranges` returns
   (-0.3, 0.3) for both bodies). `statics.leg_ik` clips to the joint range (`statics.py:124-127`), so a turn that
   needs more shows up as an IK failure with the joint pinned at 1.00 of range, not as a number above 1.
5. **[my run]** Kinematics only (`zmp_iters=0`, n = 160, r2 accuracy, left turn, `look.speed_spec(H)` at 0.40 m/s):
   the pi/4 rad/s turn uses 0.90 of the range on `60-N-T-10` (IK holds) and **saturates on `75-M-H-10`** (right
   hip yaw at 1.00, IK fails on 6 of 160 frames). Table in section 5.3.

## 1. Files

| File | Lines | What it holds for this subsystem |
|---|---|---|
| `designlab/turning.py` | 1740 | steady-turn generator, inverse dynamics, friction need, screen, CLI |
| `tests/test_turning.py` | 365 | 14 tests, on `60-M-T-10` at round-one accuracy |
| `designlab/stage1/refs.py` | 387 | names, generation and storage of the six turn references |
| `designlab/stage1/transitions.py` | 984 | turn entries, exits, in-place starts and stops (the ramps) |
| `designlab/stage1/legs.py` | 94 | `LegGap` |
| `designlab/stage1/plant.py` | 1006 | `usable_ranges` (211-237), source hash (262-275) |
| `designlab/stage1/tune.py` | 1593 | turn episode plans, decision 14 margin, turn success terms |
| `designlab/stage1/metrics.py` | 451 | yaw rate over 180 deg (316-325), `joint_margins`, `leg_gap` |
| `designlab/stage1/panels.py` | 2260 | turn conditions (453-474), templates (519-547), source hash (1572) |
| `designlab/stage1/control.py` | 2493 | `Track` (952-1096), `build_track` (1099), `TURN_CASES` (228) |
| `designlab/gait.py` | 1065 | `GaitSpec` (98-146), `look_active` (165-167), `_generate_look` (856-1065) |
| `designlab/stage2/look.py` | 1625 | `DEFERRED` (179-189), `speed_spec` (295-304), `turning_traj` (679-689) |

## 2. `designlab/turning.py`

### 2.1 Constants (`turning.py:149-190`)

```python
CASES = {"walk_0.35": dict(yaw_rate=0.35, in_place=False, role="target"),
         "in_place_0.50": dict(yaw_rate=0.50, in_place=True, role="target"),
         "walk_0.50": dict(yaw_rate=0.50, in_place=False, role="stretch")}      # 149-153
MU = 0.8                # 155  friction assumed at the pads (the compiled MJCF uses 0.9)
FN_MIN_FRAC = 0.10      # 161
INCREASE_FLAG = 0.10    # 163
NEAR_FLAG_BAND = 0.02   # 166
CLEARANCE_GATES = "r2"  # 168
CLEARANCE_PROBES = {"walk": (0.40, 0.45), "in_place": (0.35, 0.40, 0.45)}       # 170
IK_TOL = 1e-4           # 176  foot pose error (m + rad) above which a frame is an IK failure
COM_TOL = 1e-6; COM_ITERS = 12   # 182-183
PHASE_TOL = 1e-9        # 185
DS_DIRECTIONS = 256     # 188
LEG_GROUPS = ("hip_roll", "hip_yaw", "hip_pitch", "knee", "ankle_a", "ankle_b")  # 190
```

### 2.2 Signatures

```python
path_frame(t, speed: float, yaw_rate: float, origin)                       # 216 -> (P (n,2), th (n,))
turning_centre(info: dict)                                                 # 230 -> origin + [0, v/w], None if w == 0
move_pose(q, phi: float, a, b)                                             # 238 root x -> b + R(phi)(x - a), quat pre-yawed
move_velocity(qd, phi: float)                                              # 246
derivatives(M, qs, dt, transform)                                          # 254 exact stride-boundary unwrap
foot_schedule(t, T, duty, lift, feet0: dict, origin, speed, yaw_rate, phase_tol=PHASE_TOL) -> dict   # 275
zmp_reference(t, T, duty, sched: dict, pad_c: dict) -> np.ndarray          # 314
lipm_com(zref, P, theta, dt, w0, yaw_rate, speed, sway_gain=1.0)           # 338 -> (com (n,2), u (n,2))
generate(M: statics.Model, q_home, H: float, spec: gait.GaitSpec = sizing.NOMINAL, yaw_rate: float = 0.0,
         in_place: bool = False, com_tol=COM_TOL, com_iters=COM_ITERS, phase_tol=PHASE_TOL,
         unwrap: str = "exact") -> dict                                    # 366
contact_torques(M, q, qd, qdd, support, split: str = "exact")             # 630 -> (tau, wrenches, how)
ds_min_mu(M, q, qd, qdd, r_eff: dict, centroid: dict, max_iter: int = 30) -> dict   # 666
inverse_dynamics(M, traj: dict, mu: float = MU, split: str = "exact", ds_friction: bool = True) -> dict  # 755
_foot_stats(F: dict, idx: np.ndarray, phases) -> dict                      # 853
summarize(M, traj: dict, idr: dict) -> dict                                # 869
run_groups(names, idr: dict, layout: str) -> dict                          # 1010
_range_margins(cl_summary, M, trajs, joints=("hip_yaw", "hip_roll")) -> dict   # 1020
clearance_screen(xml, M, q_home, H, trajs, cases, rates=None) -> dict      # 1059
screen_model(M, q_home, design, spec=sizing.NOMINAL, cases=None, straight=None, mu=MU, com_tol=COM_TOL,
             com_iters=COM_ITERS, xml=None, validate=True, brisk=None, probes=None) -> dict   # 1179
screen_built(cid, M, q_home, design, xml, rec, out, record, spec=sizing.NOMINAL) -> dict      # 1513
screen_record(cid, rec, xml, q_home, design, scr, spec=sizing.NOMINAL) -> dict                # 1543 round-two hook
screen_candidate(cid, out, stage_a=Path("round1/stage_a"), spec=sizing.NOMINAL, round2=None)  # 1578
record_summary(doc) -> dict                                                # 1602
headline(doc) -> dict                                                      # 1644
main(argv=None)                                                            # 1705
```

### 2.3 How `generate` builds a turn (`turning.py:366-467`)

A steady turn, one stride, periodic in the path frame. There are no ramps here.

* Timing from the spec: `base = spec.describe(H)`, `T = base["period"]`, `v = 0.0 if in_place else base["speed"]`,
  `n = spec.n`, `dt = T / n` (384-389). **It reads `spec.duty` directly** (396, 399, 400, 459), not
  `describe(H)["duty"]`, which is where F3's `swing_time` acts (`gait.py:143-144`).
* Path frame (216-227): origin moves at v on a circle of radius v / omega, heading `theta = yaw_rate * t`, starting at
  `origin` = midpoint of the two `*_hip_pitch` joint anchors at home (393). Positive yaw rate turns left.
* Heading per stride is `yaw_rate * T` (`info["heading_per_stride"]`, 458). The stride motion is
  `transform = (yaw_rate * T, origin, PT)` with `PT = path_frame([T], ...)` (407-408): the next stride is
  `move_pose(q, phi, origin, PT)`.
* Footprints (`foot_schedule`, 275-311): "the home foot pose carried by the path frame to mid-stance". Left lifts at
  phase 0, right at 0.5. A stance foot is fixed at the path-frame pose of its mid-stance time (301). A swing foot
  moves along the chord between consecutive footprints and yaws between their headings, both with
  `gait._min_jerk(u / sw)` (305-307), and lifts by `lift * sin(pi u / sw)**2` (308). Feet stay flat: targets are
  `_rotz(yaw)` only (409).
* Events: a frame within `PHASE_TOL` of a lift-off is swing, of a touchdown stance (298-300). `phase_tol=0` is
  `gait.generate`'s rule.
* ZMP reference (314-335): stance pad centroid (rotated by the foot's yaw) in single support, linear hand-over in
  double support.
* CoM (`lipm_com`, 338-354): periodic LIPM solved in the rotating frame by FFT over the n frames of one stride;
  `U[0] += 1j * yaw_rate * speed * n` is the inward lean. `sway_gain` scales the lateral oscillation.
* Per frame (417-434): `q[2] = root0[2] - spec.crouch * H`; trunk quaternion = home quaternion yawed by theta (home roll
  and pitch kept); `q[neck] = spec.head_amp * sin(2 w t)`; `q[tail] = spec.tail_amp * sin(w t + pi/2)`; then up to
  `com_iters` passes of `statics.leg_ik(M, q, targets, iters=120)` with the pelvis shifted onto the LIPM path.
  Only `neck_pitch` and `tail_yaw` are set by name (403-404); any other hinge (the `-X` neck yaw and jaw) stays at its
  `q_home` value.
* Multibody-ZMP correction: `spec.zmp_iters + 1` passes with gain `spec.zmp_gain` (439-449).
* Thigh twist is not prescribed. It is whatever `leg_ik` needs to hold the foot's yaw against the yawing trunk. The
  analytic part is in `info`: `foot_yaw_vs_trunk_amp = |yaw_rate| * spec.duty * T / 2` (459) and
  `swing_yaw_rate_peak = 1.875 * |yaw_rate| / sw` (460).

Return dict (465-467), n = `spec.n`, nq = 7 + hinges, nv = 6 + hinges:

| Key | Shape | Meaning |
|---|---|---|
| `t` | (n,) | s |
| `q` | (n, nq) | full qpos |
| `qd`, `qdd` | (n, nv) | exact-unwrap central differences |
| `phases` | list of n tuples | support set per frame, e.g. `("left",)`, `("left", "right")` |
| `dt` | float | s |
| `zmp_ref`, `com_ref`, `zmp_mb`, `path` | (n, 2) | m, world |
| `heading` | (n,) | rad, trunk yaw |
| `transform` | `(phi, P0 (2,), PT (2,))` | stride motion |
| `schedule` | dict | `pos[side]` (n,3), `yaw[side]` (n,), `swing[side]` (n,) bool, `phases` |
| `info` | dict | `spec.describe(H)` plus `speed, stride, period, cadence_hz, yaw_rate, in_place, radius, heading_per_stride, origin, foot_yaw_vs_trunk_amp, swing_yaw_rate_peak, sway_amp, lean_inward, lift, omega, zmp_error_history_m, zmp_error_exact_final_m, zmp_error_exact_final_frame, com_tol, com_iters, phase_tol, unwrap` |
| `ik_ok` | bool | no frame with foot error above `IK_TOL` |
| `ik_fail_frames` | list[int] | |
| `foot_error` | (n, 2) | per foot, m + rad |
| `com_error` | (n,) | m |

### 2.4 Why it refuses a look spec

`turning.py:380-382`, the exact check:

```python
    if gait.look_active(spec):
        raise ValueError("turning.generate builds today's walk: a spec with look fields set (gait.look_active) would "
                         "lose them (the look's turns are deferred, designlab/stage2/look.py DEFERRED)")
```

`gait.look_active` (`gait.py:165-167`) is true when any of the 27 `LOOK_FIELDS` (`gait.py:150-153`) differs from
`LOOK_DEFAULTS`. The reason is in the docstring (377-379): "this path reads ``spec.duty`` and today's head and tail
laws and would silently drop the look". `look.DEFERRED["why"]` (`look.py:184-188`) says the same of
`transitions.gait_of`. Pinned by `tests/test_stage2_look.py:113` `test_turning_generate_refuses_a_look_spec`
(`pytest.raises(ValueError, match="look")`).

### 2.5 The "straight runs"

`screen_model` runs the same generator at yaw rate 0 as a like-for-like baseline (`turning.py:1214-1216`):
`runs["straight"]` with `spec` (the nominal walk) and `runs["straight_brisk"]` with `brisk` (default `sizing.BRISK`).
They appear as `runs.straight` and `runs.straight_brisk` in `<case>/review/screening/<id>_turning.json`. Uses:

* the "more than 10 % above the straight walk" flag (1313-1317);
* DECISION_LOG F21's straight-walk friction need: `runs.straight.feet.<side>.single_support.mu_req_max`;
* Stage 2's tail-phase check: `look.turning_traj` (`look.py:679-689`) dresses a `gait.generate` trajectory as a
  yaw-rate-0 turning trajectory (adds `transform`, `path`, `heading`, `info.yaw_rate=0`, `info.unwrap="exact"`,
  `info.origin`) so that `turning.inverse_dynamics` can read it; `look.stance_yaw_moment` (`look.py:1210`) and
  `screen.straight_run` use it.

Recorded straight-run single-support `mu_req_max`: `60-N-T-10` 0.480 nominal, 0.753 brisk; `75-M-H-10` 0.361, 0.558.

### 2.6 `inverse_dynamics` and the friction need

`inverse_dynamics(M, traj, mu=MU, split="exact", ds_friction=True)` (755-838) returns

| Key | Shape | Meaning |
|---|---|---|
| `tau`, `omega` | (n, n_act) | joint torque N m, joint speed rad/s |
| `qd`, `qdd` | (n, nv) | |
| `cop_margin`, `normal` | (n, 2) | m inside the pad polygon; N |
| `friction` | (n,) | largest F_t / F_n |
| `feet[side][key]` | (n,) each | keys `fn, ft, mz, r_cop, util, util_cop, mu_req, mu_req_cop, mu_req_centroid, margin`, plus `nsup` (int) |
| `r_eff[side]` | float | m, mean distance from the pad-area centroid (`uniform_radius`, 487) |
| `mu` | float | the assumed mu |
| `total_force` | (n, 3) | N, world |
| `moment_about_centre` | (n,) | N m about the turning axis (NaN for a straight run) |
| `u_com` | (n, 2) | CoM in the path frame |
| `split`, `split_how` | str, list | per frame `unique`, `equality_optimum`, `exact_qp`, `infeasible`, `statics` |
| `ds` | dict or None | `min_mu, min_mu_lower, fn_left, fn_right` (n,), `feasible` (n,) bool |

Per-frame formulas (820-824), single support being the requirement:

```python
F["util"][k] = abs(mz) / (mu * fn * r_eff[side])
F["mu_req"][k] = math.hypot(ft / fn, mz / (fn * r_eff[side]))            # does not depend on the assumed mu
F["mu_req_cop"][k] = math.hypot(ft / fn, mz / (fn * r_cop)) if r_cop > 0 else np.inf
F["mu_req_centroid"][k] = math.hypot(ft / fn, mz_cen / (fn * r_eff[side]))
```

`mz` is the free vertical moment about the CoP (814), `mz_cen` about the pad-area centroid (815). Double support:
`ds_min_mu` (666-752), the smallest mu for which some split of the total wrench keeps both CoPs on their pad hulls
with `||(F_x, F_y, M_z / r_eff)|| <= mu F_n` per foot; `mu` is an upper bound, `mu_lower` the polyhedral lower bound.

`_foot_stats(F, idx, phases)` (853-866) reduces one foot's arrays over the frames `idx` to
`frames, fn_min, ft_over_fn_max, mz_abs_max, util_max, util_cop_max, mu_req_max, mu_req_cop_max, mu_req_centroid_max,
r_cop_min, frames_cop_outside, cop_margin_min, util_max_frame, util_max_support`. `summarize` (869-932) calls it per
side over single support (`feet[side]["single_support"]`) and over double support where the foot carries at least
`FN_MIN_FRAC` of the weight (`double_support_qp_allocation`, a diagnostic), and adds `double_support`
(`min_mu_max`, ...), `range_use`, `checks`, `split`.

Grip need of a case = `_case_stats(doc, c)["mu_needed"]` (1591-1599) = max of the single-support `mu_req_max` over
both directions and feet and the double-support `min_mu_max`.

Recorded single-support `mu_req_max` (worse foot): `60-N-T-10` walk 0.35: 0.544, walk 0.50: 0.566, in place 0.50:
0.281; `75-M-H-10`: 0.409, 0.427, 0.252. Double support 0.17-0.22 everywhere.

## 3. How Stage 1 names and stores turn references

### 3.1 Names

`refs.turn_cases()` (`refs.py:85-87`) crosses `turning.CASES` with `DIRECTIONS = (("left", 1.0), ("right", -1.0))`
(`refs.py:66`):

```python
    return {f"{case}_{label}": dict(case=case, yaw_rate=sign * c["yaw_rate"], in_place=c["in_place"], role=c["role"])
            for case, c in turning.CASES.items() for label, sign in DIRECTIONS}
```

Six names: `walk_0.35_left`, `walk_0.35_right`, `in_place_0.50_left`, `in_place_0.50_right`, `walk_0.50_left`,
`walk_0.50_right`. `refs.reference_names(body)` (90-91) = `nominal, brisk, fr05, carpet` + these six.

Transitions (`transitions.py:120-131`): `enter_walk_0.35_{left,right}`, `enter_walk_0.50_{...}`,
`exit_walk_0.35_{...}`, `exit_walk_0.50_{...}` (all from or to `nominal`), `start_in_place_0.50_{...}`,
`stop_in_place_0.50_{...}` (from or to `stand`). There is no entry into a turn from any walk but `nominal`.

Panel condition keys (`panels.py:460-466`): `turn/walking_0.35_left`, `turn/walking_0.35_right`,
`turn/in_place_0.50_left`, `turn/in_place_0.50_right` (gated), `turn/walking_0.50_left`, `turn/walking_0.50_right`
(reported). `tune.TURN = {"walking_turn": "walk_0.35", "in_place_turn": "in_place_0.50"}` (`tune.py:124`).

### 3.2 Generation

`refs.generate(src, name)` (`refs.py:113-134`), turn branch (125-133): always `spec = sizing.NOMINAL` (127),
`turning.generate(M, q0, H, spec, yaw_rate=c["yaw_rate"], in_place=c["in_place"])`,
`turning.inverse_dynamics(M, tr, turning.MU, split="exact", ds_friction=True)`, `turning.summarize`, then
`s["groups"] = turning.run_groups(M.names, idr, src["layout"])`. Accuracy is r2 (`refs.ACCURACY`, line 46).
`refs.source(body)` (97-110) rebuilds the body from its record in a temp directory and checks the MJCF's sha256.

### 3.3 Files

`<run>/<body>/refs/<name>.npz` (`refs.arrays`, 145-166). Shapes from `stage1/runs/2026-11-08/60-N-T-10/refs/walk_0.35_left.npz`
(14 hinges, so nq 21, nv 20):

| Key | Shape | Note |
|---|---|---|
| `t` | (160,) | s |
| `q` | (160, 21) | qpos |
| `qd`, `qdd` | (160, 20) | |
| `tau`, `omega` | (160, 14) | feedforward torque, joint speed, in `joints` order |
| `support` | (160, 2) bool | left, right |
| `com_ref`, `zmp_ref`, `zmp_mb` | (160, 2) | |
| `cop_margin`, `normal` | (160, 2) | |
| `friction` | (160,) | |
| `feet` | (160, 2, 3) | foot body positions |
| `com` | (160, 3) | model CoM |
| `q_home` | (21,) | |
| `joints` | (14,) `<U17` | `neck_pitch, tail_yaw, left_hip_roll, left_hip_pitch, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll, right_...` |
| `dt` | () | s |
| `stride_motion` | (5,) | `(phi, a_x, a_y, b_x, b_y)` |
| turns only: `path` | (160, 2) | |
| turns only: `heading` | (160,) | |
| turns only: `foot_target` | (160, 2, 3) | |
| turns only: `foot_yaw` | (160, 2) | |
| turns only: `swing` | (160, 2) bool | |

The last five are written only when the trajectory has a `schedule` (`refs.py:160-165`), which only
`turning.generate` produces. There is no foot pitch array.

`<run>/<body>/refs.json` (`refs.prepare`, 307-328): top keys `body, accuracy, model, H_m, q_home, joints, feedforward,
match_rule, references, matches_records, seconds, provenance`. A turn's entry (`refs.write`, 289-304) has
`name, file, sha256, frames, dt_s, period_s, speed_m_s, stride_m, lift_m, cadence_steps_s, yaw_rate_rad_s,
stride_motion, min_cop_margin_single_m, max_friction_ratio, ik_ok, generator, dynamics, spec, case, yaw_rate, in_place,
role, info, seconds, record_checks, matches_records`. `generator` is `"turning.generate"`, `dynamics`
`"turning.inverse_dynamics (exact split, mu 0.8)"`. `yaw_rate_rad_s` is signed; `tune.build_spec` reads it
(`tune.py:501`).

Record check for a turn (`refs.record_checks`, 275-278): the summary is compared leaf by leaf (6 significant digits)
with `<case>/review/screening/<cid>_turning.json` `runs.<name>` (232 leaves for `walk_0.35_left`). A new turn has no
such record, so this branch raises `KeyError` on `doc["runs"][name]` unless the screen record is extended or the
check is skipped for it.

### 3.4 How a track plays them

`control.build_track(p, run_dir, plan, omega)` (`control.py:1099-1132`) takes `("ref", name, strides)`,
`("transition", name)`, `("hold", seconds)`. `Track.add_periodic` (967-978) reads `yaw_per_stride` from
`ref.stride_motion[0]`; the reference yaw is carried across segments by `_append` and `_yaw_at_end` (1017-1035).
Walking-turn plans: `tune.episode_plan` (`tune.py:275-285`), `panels.template` (`panels.py:533-536`), `env.TASKS`
(`env.py:78-88`).

## 4. The ramps (`designlab/stage1/transitions.py`)

All plan parameters are functions of gait phase (`_param_fns`, 256-293); `_ramp` (182-185) is the minimum-jerk step
`u**3 * (10 - 15 u + 6 u**2)`.

* Gait to gait, which is what `enter_*` and `exit_*` are (kind `"walk"`, 278-286): T, duty, lift, v, yaw rate w, head
  and tail amplitudes and crouch all go from gait a's to gait b's over phases `[0, WALK_RAMP_STRIDES]`,
  `WALK_RAMP_STRIDES = 1.0` (106). Output window `WINDOWS["walk"] = (-2, 4)` (109): 2 strides of a, the ramp stride, 3
  strides of b. `transitions.json` records `ramps: {"all": [0.0, 1.0]}`, 960 frames.
* `start_in_place_*` (kind `"start"`, 259-267): v and w ramp from 0 over `[phi_c - 0.2, phi_c + 0.2]`,
  `phi_c = 0.5 - duty / 2`, `PATH_HALF_WIDTH = 0.2` (104). `stop_in_place_*` ramps down over the same window.
* Path frame (302-314): `heading' = w T`, `position' = v T (cos heading, sin heading)`, integrated on a grid of
  `SUB = 16` steps per frame.
* Footprints (333-342): the home foot pose at the path pose of the stance's mid phase,
  `pose(0.5 * (stance_from + stance_to))`. Swings use `_ramp(u)` for position and yaw and `lift * sin(pi u)**2`
  (374-380).
* Trunk and appendages (417-422): heading from the path frame, `root_z = root0_z - amp * crouch(phase)`,
  `neck = amp * head_amp * sin(4 pi phase)`, `tail = amp * tail_amp * sin(2 pi phase + pi/2)`.
* CoM by ZMP preview control (`com_path`, 492-524); poses by `solve_frames` (555-584) with foot targets
  `turning._rotz(pl.foot_yaw[k, j])` (568); inverse dynamics by `turning.contact_torques(..., split="exact")` when
  either gait is a turn (665, 690-691).
* The first computed frame must fall on a stride boundary of gait a (616-617).
* Gate G3 (`compare`, 719-757): last output stride within 1 mm (CoM) and 2 mm (ZMP) of gait b's periodic stride.

`Gait` (143-162) is `name, period, duty, lift, speed, yaw_rate, head_amp, tail_amp, crouch, zmp_iters, zmp_gain,
split`. `gait_of(body, name, H)` (165-179) builds it from `refs.walk_specs(body)[name]` or, for a turn, from
`sizing.NOMINAL` (176) and `refs.turn_cases()[name]`.

Tune's judged window for a walking turn is `(enter phase 1.0, exit phase 0.0)` (`tune.py:504-505`): 3 + 5 + 2 = 10
strides of steady turning, 204 deg at 0.35 rad/s, which is why the yaw-rate reading can be "over the 180 deg turn".

No code defines a turn's "command" instant. The stop has one (`stop_command_phase`, `transitions.py:783-785`, and the
track event at `control.py:994-1001`); a turn does not. **Not found.**

## 5. The thigh-twist range

### 5.1 Where it is computed today

* `turning.summarize`, `turning.py:904-911`:

```python
    for side in SIDES:
        for jn, qi in zip(statics.LEG_JOINTS, M.leg_q[side], strict=True):
            lo, hi = m.jnt_range[statics.jid(m, f"{side}_{jn}")]
            qv = traj["q"][:, qi]
            rng[f"{side}_{jn}"] = float(max(np.max(qv) / hi, np.min(qv) / lo))
    out["range_use"] = rng  # largest fraction of the joint limit reached (1.0 = at the limit)
```

  The range is the designlab model's MJCF range. It assumes `lo < 0 < hi`.
* `_case_stats` (1599): `hip_yaw_range_use` = max over both directions and sides. `record_summary` (1624):
  `hip_yaw_range_used_max` = max over the target cases; `shortlist.py:203` reads it.
* `clearance_screen` (1113): `cases[c]["range_margins"]` from `_range_margins` (1020-1042), per `left_hip_yaw`,
  `left_hip_roll`, `right_hip_yaw`, `right_hip_roll`: `min, max, usable, margin_min, margin_max` (rad) against the
  clearance screen's usable range.
* The screen is run as `turning.screen_model(...)` per body; its record is
  `<case>/review/screening/<cid>_turning.json`, `runs.<name>.range_use`.

### 5.2 Recorded values (nominal walk, +-0.30 rad)

From `round2/refine/{narrow_carpet_60,feet_75}/review/screening/*_turning.json`, worse leg (the outer one):

| Body | walk 0.35 | in place 0.50 | walk 0.50 | margin left, walk 0.50 (rad) |
|---|---|---|---|---|
| `60-N-T-10` | 0.522 | 0.682 | 0.766 | 0.070 |
| `75-M-H-10` | 0.577 | 0.773 | 0.847 | 0.046 |

`DECISION_LOG.md:70` (F11) gives the ranges over all bodies: walking 0.35 "0.50-0.52 and 0.57-0.59", in place
"0.67-0.69 (60 cm) and 0.78-0.79 (75 cm)", stretch "0.73-0.77 and 0.84-0.87". `STAGE2_PLAN.md:197-198` scales these
linearly to pi/4 at the nominal walk: "about 1.1-1.4 of the thigh twist's range".

Peak hip yaw is about 1.45-1.50 times `foot_yaw_vs_trunk_amp` in the stored nominal walking turns (0.1566 rad against
0.1071 on `60-N-T-10`; 0.1731 against 0.1192 on `75-M-H-10`) and 1.34-1.36 times in place. The factor is not constant:
it rises with speed and rate (5.3), so the linear scaling under-reads fast turns.

### 5.3 Computing it for pi/4 rad/s from 0.40 m/s

The 0.40 m/s walk is `look.speed_spec(H)` (`look.py:295-304`), the point on the brisk-to-Froude-0.05 change at
0.40 m/s: a plain `GaitSpec` (no look field), so `turning.generate` accepts it. H is the hip height: 0.40469 m
(`60-N-T-10`), 0.50185 m (`75-M-H-10`).

| Body | T (s) | duty | stride (m) | lift_frac | heading per stride at pi/4 | radius (m) |
|---|---|---|---|---|---|---|
| `60-N-T-10` | 0.7510 | 0.5670 | 0.3004 | 0.11 | 0.590 rad (33.8 deg) | 0.509 |
| `75-M-H-10` | 0.8606 | 0.5732 | 0.3442 | 0.11 | 0.676 rad (38.7 deg) | 0.509 |

The call (what `refs.generate` does for a turn, with another spec and rate):

```python
gait.set_accuracy("r2")
src = refs.source(body); M, q0, H = src["M"], src["q_home"], src["H"]
tr = turning.generate(M, q0, H, look.speed_spec(H), yaw_rate=+math.pi / 4)   # and -math.pi / 4
idr = turning.inverse_dynamics(M, tr, turning.MU, split="exact", ds_friction=True)
s = turning.summarize(M, tr, idr)
s["range_use"]["left_hip_yaw"], s["range_use"]["right_hip_yaw"], s["ik_fail_frames"]
```

The range use needs only `tr["q"]` (the formula in 5.1), so it can be read before the inverse dynamics. Check
`tr["ik_ok"]` first: at the limit the value is pinned at 1.00.

**[my run]** `zmp_iters=0` (no multibody-ZMP pass), n = 160, r2, left turns, hip-yaw range use (left, right):

| Spec | Rate (rad/s) | `60-N-T-10` | IK | `75-M-H-10` | IK |
|---|---|---|---|---|---|
| 0.40 m/s | 0.35 | 0.356, 0.380 | ok | 0.403, 0.430 | ok |
| 0.40 m/s | 0.50 | 0.503, 0.552 | ok | 0.569, 0.626 | ok |
| 0.40 m/s | pi/4 | 0.774, 0.902 | ok | 0.877, 1.000 | fails, 6 frames |
| 0.40 m/s | 1.00 | 0.972, 1.000 | fails, 18 | 1.000, 1.000 | fails, 42 |
| 0.40 m/s, F3's duty | pi/4 | 0.779, 0.909 | ok | 0.888, 1.000 | fails, 8 |
| nominal | 0.35 | 0.474, 0.519 | ok | 0.523, 0.573 | ok |
| nominal | 0.50 | 0.668, 0.762 | ok | 0.738, 0.841 | ok |
| nominal | pi/4 | 1.000, 1.000 | fails, 31 | not run | |

"F3's duty" is `speed_spec` with `duty = clip(1 - look.swing_time(H) / T, 0.55, 0.72)` = 0.5743 (60 cm), 0.5863
(75 cm), set by hand because `generate` refuses `swing_time`. The nominal rows land within 0.005 of the records
(0.522, 0.766, 0.577, 0.847), so the missing ZMP passes move the reading little. One generate call took 1 to 5 s at
these settings.

Reading: at 60 cm the pi/4 turn from 0.40 m/s leaves about 0.03 rad of hip yaw; at 75 cm it does not fit +-0.30 rad
with today's footprint rule. 1.0 rad/s fits on neither. The clearance and look screens' walking margin is 0.10 rad
(`clearance.RANGE_MARGIN_RAD`, `clearance.py:140`); the turns' rule is decision 14 (5.4).

### 5.4 How the range is judged in closed loop (decision 14)

`STAGE1_PLAN.md:210-214`: "In turns a success must reach no joint limit and keep at least the reference's own smallest
margin less 0.02 rad." Code, `tune.build_spec`, `tune.py:510-520`:

```python
    q = np.vstack([seg.S[:, :nj] for seg in track.segments])
    rest = plant.rest_ends(p.joints)
    ref_margins = metrics.joint_margins(q, lo, hi, rest)
    ...
        required = np.maximum(np.minimum(WALK_RANGE_RAD, ref_margins - TURN_RANGE_SLACK_RAD), AT_LIMIT_RAD)
```

with `WALK_RANGE_RAD = 0.10`, `TURN_RANGE_SLACK_RAD = 0.02`, `AT_LIMIT_RAD = 1e-3` (`tune.py:162-164`). `lo, hi` are
the plant's ranges, which are `plant.usable_ranges(body)` (`plant.py:890`). The margin is taken over the whole episode
track, ramps included. Success term `joint_range = shortfall <= 0.0` (`tune.py:586, 629`). A reference at 0.90 of range
(margin 0.029 rad) asks the closed loop to keep 0.009 rad.

`plant.usable_ranges(body, inter_leg="mjcf")` (`plant.py:211-237`) reads `summary.joint_range_usable` from
`<case>/review/screening/<cid>_clearance.json`: first new interference with the trunk or the same leg, else the MJCF
limit; a limit set by the two legs touching falls back to the MJCF limit. For hip yaw it returns (-0.3, 0.3) on both
bodies. Pinned by `tests/test_stage1.py:316`.

## 6. Reaching 80 degrees by 2.0 s

The bar (`STAGE2_PLAN.md:187`): "The reference turns at pi/4 rad/s (5 % fast) between its ramps to the start heading +
90 deg, itself reaching 80 deg by 2.0 s (G9 ...)"; the trunk's heading within 10 deg of start + 90 deg 2.0 s after
the command and for 3 s of walking after.

"5 % fast" is the commanded clock: `tune.COMMAND_RATE = 1.05` (`tune.py:111-112`) plays the track 5 % faster, so
simulated 2.0 s is reference time 2.1 s and the yaw rate in simulated time is 0.8247 rad/s.

Arithmetic (mine; constant T, symmetric minimum-jerk ramps of length R seconds, the command at the start of the
ramp in, total exactly 90 deg, steady rate pi/4):

* A minimum-jerk ramp of length R contributes `omega R / 2` of heading. Total: `omega (R + steady) = pi / 2`, so
  `R + steady = 2.0 s` of reference time.
* On the reference's own clock the heading at 2.0 s is `90 deg - omega R / 2`. At least 80 deg needs **R <= 0.444 s**
  (0.59 stride at 60 cm, 0.52 at 75 cm).
* On the 5 % fast clock (reference time 2.1 s) it needs **R <= 0.643 s** (0.86 stride at 60 cm, 0.75 at 75 cm).
* With today's 1-stride ramps (`WALK_RAMP_STRIDES = 1.0`: 0.751 s, 0.861 s) the reference is at 73.1 deg (60 cm) and
  70.6 deg (75 cm) at 2.0 s on its own clock, 77.6 deg and 75.1 deg on the fast clock. It reaches 80 deg at simulated
  2.06 s and 2.11 s. So **a 1-stride ramp misses the 80 deg rule on both bodies**; half-stride ramps give 85.9 deg and
  84.7 deg on the fast clock.
* Whole turn in strides: ramp + steady = 2.0 s / T = 2.66 strides (60 cm), 2.32 strides (75 cm). Neither is a whole
  number, so the turn cannot be "N whole periodic strides between two stride-long ramps" at exactly pi/4 and exactly
  90 deg. A finite plan whose `w(phase)` rises and falls inside one output window can do it; `plan()` already takes
  any phase function for `w` (the start and stop ramps are not on stride boundaries either).
* Peak thigh twist comes from the steady part (1.3 to 1.8 strides here, longer than one stance), so section 5.3's
  steady-turn numbers are the ones to check first; the finite reference's own `q` then gives the exact value
  (`metrics.joint_margins`, or 5.1's formula on the transition's `q`), with `ik_ok` (`transitions.py:677`).

The yaw-rate reading itself assumes 180 deg: `metrics.window`, `metrics.py:316-325`, takes the time to
`turned >= math.pi` and otherwise reports `turned / (tb - ta)` with `turned_180 = False`. A 90 degree turn never
reaches it.

## 7. What a look-aware turn needs, feature by feature

`gait._generate_look` (`gait.py:856-1065`) is written for a straight line: x is forward, y is lateral, both in the
world. What each look feature touches in the turn path (`turning.generate` lines; `transitions.plan` has the same
laws at the lines in section 4):

| Feature (fields) | Straight-walk code | What the turn path does today | Interaction with turning |
|---|---|---|---|
| F3 `swing_time` | duty from `describe(H)` (`gait.py:143-144, 861`) | reads `spec.duty` (396, 399, 400, 459) | A longer stance raises the foot's yaw against the trunk (`amp = omega duty T / 2`): +0.7 to +1 % of range at pi/4 **[my run]**. |
| F1/F2 `bob`, `bob_skew`, `bob_drop`, `bob_phase` | `dz` on `q[2]` (`gait.py:959-964, 1001`) | fixed `q[2] = root0[2] - crouch H` (419) | Keyed to phase and touchdowns, so it carries over unchanged in the path frame. The LIPM height `z_c` stays constant in both. Slows the ZMP correction (`look.LOOK_ZMP_ITERS = 12`). |
| F4 `lift_peak` | `lift_profile(s, lift_peak)` (`gait.py:897`) | `sin(pi u / sw)**2` (308) | Independent of the turn. |
| F5 `toe_off*`, `toe_reach` | foot pitch `_rot_y(pitch)`, `roll_offset` in world x-z, `rolling_entry` from the straight stride (`gait.py:919-956`) | flat feet, `_rotz(yaw)` only (409) | The largest change. The pitch must be about the yawed foot's own lateral axis and the roll offset rotated by the foot's yaw. `rolling_entry` assumes two parallel feet a stride apart. `turning._qp_matrices` (548-581) and `ds_min_mu` (666-752) assume flat pad boxes and a torsion radius `r_eff`; a foot on its toe edge or claw tip has neither (the look's straight-walk version is `look.look_inverse_dynamics`, `look.py:719-844`, and `screen.straight_run` leaves rolling frames out of the double-support need). The roll happens while the trunk yaws over a planted, rolling foot. |
| F6 `roll_amp`, `roll_phase` | trunk quaternion = home x roll about the long axis (`gait.py:965-966`) | home roll and pitch kept, yaw only (420) | Compose as yaw(theta) x home x roll. Uses hip-roll range: the recorded turns leave 0.10-0.12 rad of inward hip roll at 0.35 rad/s and 0.07-0.10 rad at 0.50 (`range_margins`); 1.5 deg of roll is 0.026 rad. The turn's own inward lean is a CoM shift, not a trunk roll. |
| F7 `zmp_inset_y` | `inset` along world y (`gait.py:887`) | none | Must be along the yawed foot's lateral axis, as `pad_c` is (`turning.py:324`). |
| F9 `tail_phase` (and `tail_amp`) | `sin(2 pi phi + tail_phase)` (`gait.py:969`) | `+ np.pi / 2` hard-coded (422) | `look.tail_check` picks the phase that minimises the straight walk's stance-pad yaw moment. A turn adds its own yaw moment (single-support `mu_req_max` 0.544 in the 0.35 turn against 0.480 straight on `60-N-T-10`), so the best phase may differ; nothing in the code checks it for a turn. |
| F10 `head_mode`, `head_gain`, `head_settle`, `settle_lag` | neck cancels trunk pitch, dips after touchdowns (`gait.py:972-980`) | `head_amp sin(2 w t)` (421) | Trunk pitch is at home in a turn, so only the settle dips remain (plus F6's small effect on `_nose_up`). |
| F14 `strides`, `phase_jitter`, `asym`, `gaze_var`, `tail_var` | K-stride window with a warped phase (`_Warp`, `gait.py:735, 883-885`) | one stride of `spec.n` frames, uniform phase, FFT-periodic LIPM over that stride (338-354), rigid stride motion `omega T` (408) | The periodic turn assumes identical strides. A finite turn in `transitions.plan` is already parametrised by phase with `T(phase)`, the same idea as the warp. |
| F13 `stride_rel` | via `describe` | via `describe` (384-386) | None beyond the larger T: more thigh twist per stride. |
| F11 head-first turn | not a field (`gait.py:57-58`) | not built | `research_look.md:193`: lead 0.2-0.4 s, up to 0.3-0.5 rad of neck yaw, tail flick inward at the start and trailing at the end; needs the `-X` neck yaw. `turning.generate` leaves `neck_yaw` at home. |
| F12 anticipation, settle | not a field | not built | `research_look.md:194`: belongs to the transition generator and a head and tail overlay. |

Also flat-foot assumptions downstream of the references: `refs.arrays` stores `foot_target` and `foot_yaw` only
(`refs.py:160-165`); `transitions.arrays` the same (`transitions.py:865-876`).

`STAGE2_PLAN.md:114-116` already excludes F5 from R2L ("without F5's toe-off, which its flat-pad ankle cap and
estimator cannot follow").

## 8. `LegGap` (`designlab/stage1/legs.py`)

```python
class LegGap:
    def __init__(self, body: str, cap: float = clearance.REPORT_CAP)   # 32; REPORT_CAP = 0.025 m
    def at(self, qpos) -> np.ndarray                                  # 55  capped signed distance of every leg pair
    def pair(self, k: int) -> list[str]                               # 60  [geom name, geom name]
    def per_pose(self, frames) -> np.ndarray                          # 63  (len, 4)
    def summary(self, rows: np.ndarray) -> dict                       # 76
    def trace(self, frames) -> dict                                   # 92  = summary(per_pose(frames))
```

* What it checks: signed distance (negative = interference) between every screened geom on a `left_*` body and every
  screened geom on a `right_*` body (36-41), on the **screened** MJCF from `plant.model_files(body)`, with
  `clearance._Screen`'s geometry and distance oracle (the `hw_*` hardware and `cl_*` shell proxies). In the plant only
  the foot pads collide, so this is the only leg-to-leg check.
* `self.home` = distances at `key_qpos[0]`; `self.moving = self.home >= cap` marks pairs that are not within 25 mm at
  home (42-43). On the 60 cm bodies the two hip brackets sit 4 mm apart at every pose and would otherwise mask the rest.
* `per_pose` row: `(smallest gap, its pair index, smallest gap of the moving pairs, its pair index)`, index -1 when
  nothing is within the cap; distances beyond the cap are reported as the cap.
* `summary` keys: `poses, cap_mm, interference_poses, tight_poses, interference_free, min_gap_all_mm,
  min_gap_all_pair, min_gap_all_pose, min_gap_moving_mm, min_gap_moving_pair, min_gap_moving_pose,
  home_pairs_within_cap` (mm). Tight is `0 <= d < clearance.TIGHT` (3 mm).
* Poses are full qpos vectors of the plant ("the plant's and the screened model's qpos agree", line 56).

Where it is judged:

* Sampling: `metrics.Recorder(legs_every=...)` stores `qpos` every `legs_every` feedback updates
  (`metrics.py:172-173`); `tune.LEGS_EVERY = 2` (`tune.py:119`) gives 50 Hz. "a brief touch between samples can be
  missed" (`legs.py:17-18`).
* `metrics.leg_gap(rec, lg)` (`metrics.py:386-392`) = `lg.trace(rec.poses)` reduced to 8 keys.
* `tune.measure` (`tune.py:595`): `legs = metrics.leg_gap(rec, lg) if (lg is not None and kind != "walk") else None`,
  so **only turns are checked**. Limit term `leg_interference` from `min_gap_all_mm` (643-644, scale 5 mm); success
  term `no_leg_interference = bool(legs and legs["interference_free"])` (645). Both `SUCCESS["walking_turn"]` and
  `SUCCESS["in_place_turn"]` include it (`tune.py:187-192`).
* Panels: `Built.legs` is true only for `cond.judge in ("walking_turn", "in_place_turn")` (`panels.py:926-927`); the
  recorder gets `legs_every` only then (929-931); `Context.lg` builds one `LegGap` per body lazily (972-977).
* `tune.Evaluator.__init__` builds `LegGap(body)` when `legs=True` (`tune.py:793-814`).
* Related but separate: `control.turn_gap(body)` (`control.py:235-240`) is the turning screen's smallest gap in the
  target turns, a pelvis-to-thigh pair, which bounds no step since decision 20.
* Test: `tests/test_stage1.py:593` `test_leg_gap_on_the_reference_walk` (40 poses, interference-free, smallest pair
  the two hip brackets at 4.0 mm, moving gap positive).

The bar's "no leg interference" for the 90 degree turn (`STAGE2_PLAN.md:187`) therefore needs the new condition's
judge to be one that turns the recorder's `legs_every` on.

## 9. Hashes and what an edit breaks

| Hash | Where | Covers |
|---|---|---|
| Stage 1 sources | `plant.sources_sha256`, `plant.py:267-275`; rule 263-264 | every `designlab/stage1/*.py` (name, NUL, bytes, NUL, in name order). In every `provenance()` (294-316), so in `refs.json` and `transitions.json`. `train.py:463-465` refuses a resume when it changed, unless `--allow-source-change`. |
| Panel sources | `panels.panel_sources`, `panels.py:1572-1583`; rule 347 | stage1 modules `__init__, control, drive, estimator, legs, metrics, panels, plant, refs, transitions, tune` (not `env.py`, `gates.py`) and designlab modules `__init__, build, carpet, catalog, clearance, envelope, gait, geometry, hardware, mjcf, params, shapes, sizing, skeleton, statics, template, turning`. Current sha256 starts `6a316fc82bf5`. |
| Tuner sources | `tune.tuner_sources`, `tune.py:431`; rule 374 | stage1 modules `__init__, control, drive, estimator, legs, metrics, plant, refs, transitions, tune`. |
| Input files | `tune.INPUTS_RULE`, `tune.py:294` | each reference and transition npz a track loads, by sha256, checked before loading. |
| Reference files | `refs.write`, `refs.py:297` | each npz's sha256 in `refs.json`; gate G2 compares regenerated files with these. |
| Turning screen | `turning.screen_built`, `turning.py:1526-1533` | the MJCF's sha256 against the record's `xml_sha256`. |

Consequences for this subsystem:

* **Any edit to `designlab/turning.py` or `designlab/gait.py` changes the panel sources key**, and any edit to
  `designlab/stage1/{refs,transitions,legs,metrics,tune,control,plant,panels}.py` changes that key, the tuner key and
  the Stage 1 sources hash. `look.py:107-110` records that the look branch already did this once (parent key
  `733f2b80...` at `e59ab7d`).
* `turning.generate` must keep producing the six Stage 1 turn files bit for bit (G2). The npz sha256 depends on every
  array `refs.arrays` writes, so even a new key in the trajectory dict that `arrays` picks up would change it.
* `designlab/stage2/*.py` is in none of these three hashes.

## 10. Tests that pin behaviour

`tests/test_turning.py` (fixture: `60-M-T-10` from `round1/stage_a` if present; `gait.set_accuracy("r1")` for the
module; `FAST = replace(sizing.NOMINAL, n=80, zmp_iters=1)`):

| Test (line) | Asserts |
|---|---|
| `test_path_frame_is_a_circle_with_the_straight_and_in_place_limits` (79) | path on a circle of radius v/w, heading w t; yaw 0 is exactly `v t`; v 0 is a fixed point |
| `test_footprints_lie_on_concentric_arcs_and_are_yawed_with_the_body` (96) | stance feet on circles about the centre and yawed with the bearing; left foot inside in a left turn; in-place footprints rotate about the hip midpoint |
| `test_frames_on_a_lift_off_or_touchdown_are_classified_by_the_event` (123) | with `PHASE_TOL`, left swings frames 0-63 and right 80-143 at n 160, duty 0.6, for 41 stride periods |
| `test_rotating_frame_pendulum_solution_satisfies_the_pendulum_in_the_world` (143) | `lipm_com` residual below 1e-3 of the acceleration |
| `test_exact_unwrap_matches_explicit_neighbouring_strides` (162) | `derivatives` equals three explicit strides (1e-9, 1e-6) |
| `test_zero_yaw_rate_reproduces_gait_generate` (182) | with `com_tol=2e-4, com_iters=4, phase_tol=0.0, unwrap="gait"`, q equals `gait.generate`'s to 1e-12; `unwrap="gait"` with a yaw rate raises |
| `test_left_turn_mirrors_right_turn` (203) | joint RMS, peak, speed mirror within 1 %; no IK failure |
| `test_double_support_split_is_heading_invariant_and_exact` (217) | rotating a frame rotates its split; exact QP no worse than SLSQP |
| `test_double_support_friction_requirement_is_a_minimum` (252) | `min_mu` between its lower bound and 1.02 x it; below single support's |
| `test_turn_balances_the_expected_mean_forces_and_moments` (272) | mean vertical force = weight; centripetal force; mean trunk yaw rate 0.35; `left_hip_yaw` speed peak above 0.5 rad/s |
| `test_torsion_radius_for_disc_and_off_centre_pressure` (285) | `uniform_radius` = 2R/3 on a disc; `pressure_radius` 0 outside |
| `test_motor_space_by_side_stacks_to_sizing` (299) | equals `sizing._motor_space` when stacked |
| `test_compare_flags_limits_increases_and_near_misses` (313) | `compare`'s flags |
| `test_round_two_hook_feeds_the_envelope_and_the_shortlist` (330) | `screen_record`'s first six parameters; `record_summary` keys `targets_within_limits, worst_ratio_targets, hip_yaw_range_used_max, mu_needed_max, targets_interference_free, targets_pass`; `0 < hip_yaw_range_used_max < 1`; the literal call text in `round2._turning` |

Elsewhere:

| Test | Asserts |
|---|---|
| `tests/test_stage2_look.py:113` `test_turning_generate_refuses_a_look_spec` | a look spec raises; today's spec at 0.35 rad/s runs; `"turns"` and `"starts_and_stops"` in `look.DEFERRED` |
| `tests/test_stage2_look.py:140` `test_g2_stage1_references_regenerate_bit_for_bit` | all ten Stage 1 references of both bodies, the six turns included, regenerate to the recorded sha256; `set(G2_NAMES) == set(rec["references"])` |
| `tests/test_stage2_look.py:665` | `LADDER["deferred"] == look.DEFERRED` (the stored `look_refs.json` against the constant) |
| `tests/test_stage2_screen.py:98` `test_straight_run_reproduces_the_turning_screens_record` | `screen.straight_run` on V0 equals the turning record's straight run |
| `tests/test_stage1.py:761` `test_transition_plans_are_continuous` | every plan in `TRANSITIONS`: heading rate at most the larger gait's yaw rate, no jumps |
| `tests/test_stage1.py:852` `test_transition_records_pass_g3` | `set(rows) == set(transitions.TRANSITIONS)` for the stored records; G3 values |
| `tests/test_stage1.py:316` | `plant.usable_ranges` as the screens read them |
| `tests/test_stage1.py:927-940` | `control.turn_gap` 3.43 mm on the test body; a pelvis-to-thigh pair |
| `tests/test_stage1.py:1897-1898` | `tune.directions` parity rule |
| `tests/test_stage1_panels.py:60-65` | the reported set includes `turn/walking_0.50_*`; `len(by["turn"]) == 6` |
| `tests/test_stage1_panels.py:137-139` | the 0.35 and in-place turn plans are tune's; the 0.50 one is not |
| `tests/test_stage1_speedfit.py:42` | `speedfit.EPISODES == ("walk", "walking_turn", "in_place_turn", "brisk", "fr05")` |

Three of these compare a code constant with a stored record's key set (`TRANSITIONS`, `G2_NAMES`, `len(by["turn"])`),
so adding a name to `turning.CASES`, `transitions.TRANSITIONS` or `panels.conditions()` fails them as well as moving
the hashes.

## 11. CLI entry points

* `python -m designlab.turning <ids...> --out <dir> [--stage-a round1/stage_a] [--round2 <dir>] [--workers N]`
  (`turning.py:1705-1736`). Writes `<out>/<id>_turning.json`, `.png`, `<out>/turning_index.json`, models under
  `<out>/mjcf`; refuses an `--out` under `round1/`; at most 2 workers; exit 1 if a candidate failed. Always
  `sizing.NOMINAL` and `turning.CASES`: no argument selects a spec, a rate or a case.
* `python -m designlab.stage1 prepare --body <b> --out stage1/runs/<run>`: references and transitions (other flags not
  read here).
* `python -m designlab.stage1 eval --body <b> --rows R0,R1,R2/42,R2/43 [--behaviors ...] [--conditions turn/...]
  [--workers N] [--episodes 20] [--inputs <run>] --out <run>` (`__main__.py:562-581`).
* `python -m designlab.stage2.look g2 --body <b> [--name walk_0.35_left ...]` (`look.py:114, 1578`): regenerates
  Stage 1 references and prints their sha256.
* Round two calls `turning.screen_record(cid, rec, str(xml), q_home, design, scr)` from `round2._turning`
  (`round2.py:304-309`).

## 12. Hard-coded assumptions the build list would break

| # | Assumption | Where | Breaks |
|---|---|---|---|
| 1 | Turn cases are the three of `CASES` | `turning.py:149-153`; `refs.py:85-87` | B |
| 2 | Every stored turn uses `sizing.NOMINAL` | `refs.py:127`; `transitions.py:176` | A, B |
| 3 | `generate` refuses a look spec | `turning.py:380-382` | B |
| 4 | `generate` reads `spec.duty`, fixed crouch height, nod and `+pi/2` tail, flat feet, `sin**2` lift | `turning.py:396-400, 419-422, 409, 308` | B |
| 5 | One stride of `spec.n` frames; `spec.strides` ignored | `turning.py:387` | B with F14 |
| 6 | Only `neck_pitch` and `tail_yaw` are driven | `turning.py:403-404`; `transitions.py:560-561` | B on `-X` (F11) |
| 7 | Transitions exist only between the names in `TRANSITIONS`; turn entries only from `nominal` | `transitions.py:120-131` | A, B |
| 8 | `gait_of` reads `refs.walk_specs` and `spec.duty` | `transitions.py:165-179` | A |
| 9 | Gait-to-gait ramp is one stride; window (-2, 4) | `transitions.py:106, 109` | B (the 90 deg turn, the 80 deg rule) |
| 10 | Plan laws: nod, `+pi/2` tail, crouch only, flat yawed feet, `sin**2` lift | `transitions.py:378, 420-422, 568` | A, B |
| 11 | Stop is one step (left lifts at phase 0, nothing after) | `transitions.py:329-330`; `stop_time_arrays` returns `single_step=True` (825) | A (two-step stop) |
| 12 | Turn record check reads `<cid>_turning.json` `runs.<name>` | `refs.py:275-278` | B |
| 13 | Yaw rate is read over 180 deg | `metrics.py:316-325`; `tune.py:197-198, 228` | C |
| 14 | Turn window is enter phase 1.0 to exit phase 0.0; segment names `enter_<c>`, `exit_<c>` | `tune.py:500-505` | B, C |
| 15 | Walking turn's speed target is the `gait or "nominal"` reference's | `tune.py:490, 506` | B, E |
| 16 | `tune.TURN`, `tune.episode_plan`, `env.TASKS`, `panels.conditions`, `control.TURN_CASES` name 0.35 and 0.50 | `tune.py:124, 275-285`; `env.py:78-88`; `panels.py:460-466`; `control.py:228` | B, E, F, G |
| 17 | Leg check only for judges `walking_turn`, `in_place_turn` | `panels.py:926-927`; `tune.py:595` | C, E |
| 18 | `screen_built` plots `runs["walk_0.35_left"]` and `runs["in_place_0.50_left"]` | `turning.py:1535-1536` | B if `CASES` or `cases` changes |
| 19 | `envelope.turning_rms_from` picks the first run whose name has "walk" and "0.35" | `envelope.py:225` | B if run names change |
| 20 | `range_use` assumes `lo < 0 < hi` | `turning.py:908-910` | any joint whose range does not straddle 0 (the jaw, 0 to 0.6) if the loop were widened beyond `LEG_JOINTS` |
| 21 | Contact model: flat pad box, `r_eff` of the flat pads | `turning.py:548-581, 666-752, 777` | B with F5 |
| 22 | Left foot always lifts first; left and right turns differ in which leg is inside at the first step | `turning.py:296`; `transitions.py:325` | B (left and right are not mirror tracks in a finite turn) |

## 13. Extension points and hazards

* Put new behaviour beside the old path, behind arguments whose defaults leave the six Stage 1 turns and the 18
  transitions bit-identical. `designlab/stage2/` is outside all three source hashes; `turning.py`, `gait.py` and every
  `stage1/*.py` are inside at least one.
* The 90 degree turn fits `transitions.plan`'s machinery (phase functions for v, w, T, duty, lift; footprints at the
  mid-stance path pose; preview control; G3 against the walk on the far side) as a walk-to-walk plan whose `w(phase)`
  rises and falls. It does not fit `enter + ref + exit`.
* Check the thigh twist before building anything: steady-turn kinematics with `zmp_iters=0` take seconds per case
  (5.3). On `75-M-H-10` the pi/4 turn from 0.40 m/s is at the limit in my run; `STAGE2_PLAN.md:154-155` (lever 6,
  range beyond +-0.30 rad) and line 198 ("its reference takes quicker steps") are the plan's own outs, and a shorter
  stance or period lowers `omega duty T / 2`.
* A look-aware turn's hardest part is F5 (rolling contact under a yawed foot, the friction need on an edge contact).
  F1-F4, F6, F7, F9, F10 are law substitutions in the path frame. F14 needs the multi-stride window.
* An IK failure reads as range use 1.00, never more. Always report `ik_ok` and `ik_fail_frames` beside the fraction.
* The new turn needs: a defined command instant (none exists for turns), a judge with the leg check on, a heading
  reading that is not the 180 deg yaw-rate rule, and a record-check path that does not look for a round-two turning
  record.
* `refs.source(body)` rebuilds the model in a temp directory and is safe to call read-only. `legs.LegGap(body)` and
  `plant.build` go through `plant.model_files`, which writes `stage1/models/<body>/` if the round-two model files are
  missing (they are present now).

## 14. Not found

* A turn's command instant or a "time to within 10 deg" reading: not found in code.
* Any generator of a finite turn (entry, part turn, exit in one reference): not found.
* A two-step stop: not found (the stop is one step).
* A tail-phase check on a turning run: not found (`look.tail_check` runs straight walks only).
* The -X bodies' turning screen values were not read here; `head_x_60` and `head_x_75` have their own case
  directories and I did not open their `_turning.json`.
* Full-setting (6 ZMP passes, friction LPs) values for the pi/4 turn: not computed.
