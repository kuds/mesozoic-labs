# Map 02: Stage 1 references and transitions

Subsystem: `designlab/stage1/refs.py` (387 lines), `designlab/stage1/transitions.py` (984 lines), their records
(`refs.json`, `transitions.json`), `stage1/runs/2026-11-08/stage1_inputs.json`, `tools/restore_inputs.py`.
Paths are relative to `/home/user/mesozoic-labs/hardware/trex_form_study` unless absolute. Line numbers are those of
the working tree at commit `7e9ea0d` (14 Nov 2026).

How this map was made, and its limits:

- Read-only. No file in the checkout was edited.
- **No `*.npz` file was present** in the checkout while I mapped (the background `restore_inputs.py restore` job was
  still sizing models). Array shapes below are therefore read from the code and from the `frames` fields of the JSON
  records, not from loaded arrays. Dimensions `nq = 21`, `nv = 20`, 14 joints are from `refs.json` (`q_home` has 21
  entries, `joints` 14); the -X bodies have `nq = 23`, 16 joints (`stage2/x/75-M-H-10-X/refs.json`).
- I ran `transitions.plan` on a synthetic `Geometry` with hand-made `Gait` objects (no model, no IK) to count swings
  and touchdowns. Those results are marked "checked on a synthetic plan".
- Arithmetic of mine (not a record) is marked "my arithmetic".

---

## 1. The references the Stage 1 plant uses

### 1.1 Names (`refs.py:80-91`)

```python
def walk_specs(body: str) -> dict:                                   # refs.py:80
    return {"nominal": sizing.NOMINAL, "brisk": sizing.BRISK, "fr05": envelope.FAST,
            "carpet": replace(sizing.NOMINAL, lift_frac=CARPET_LIFT[body])}

def turn_cases() -> dict:                                            # refs.py:85
    return {f"{case}_{label}": dict(case=case, yaw_rate=sign * c["yaw_rate"], in_place=c["in_place"], role=c["role"])
            for case, c in turning.CASES.items() for label, sign in DIRECTIONS}

def reference_names(body: str) -> list[str]:                         # refs.py:90
    return list(walk_specs(body)) + list(turn_cases())
```

Ten references per body, in this order:
`nominal, brisk, fr05, carpet, walk_0.35_left, walk_0.35_right, in_place_0.50_left, in_place_0.50_right,
walk_0.50_left, walk_0.50_right`.

Straight walks (`gait.GaitSpec`; the 11 pre-look fields, every look field at its default):

| name | froude | duty | lift_frac | stride_rel | crouch | head_amp | tail_amp | zmp_iters x gain | n |
|---|---|---|---|---|---|---|---|---|---|
| `nominal` (`sizing.NOMINAL`) | 0.012 | 0.60 | 0.10 | 0.55 | 0.03 | 0.05 | 0.12 | 6 x 0.5 | 160 |
| `brisk` (`sizing.BRISK`) | 0.025 | 0.58 | 0.11 | 0.62 | 0.03 | 0.05 | 0.12 | 6 x 0.5 | 160 |
| `fr05` (`envelope.FAST`) | 0.05 | 0.56 | 0.11 | 0.80 | 0.03 | 0.05 | 0.12 | 6 x 0.5 | 160 |
| `carpet` | as nominal, `lift_frac = CARPET_LIFT[body]` (`refs.py:57-58`): 0.19 (60-N-T-10, -K, -X), 0.195 (60-N-H-10, -K), 0.18 (75-M-H-10, -X) |

Turns (`turning.CASES`, `turning.py:149`; `DIRECTIONS = (("left", 1.0), ("right", -1.0))`, `refs.py:66`), all on
`sizing.NOMINAL` (hard-coded, `refs.py:127`):

| case | yaw_rate (rad/s) | in_place | role |
|---|---|---|---|
| `walk_0.35` | 0.35 | False | target |
| `in_place_0.50` | 0.5 | True | target |
| `walk_0.50` | 0.5 | False | stretch |

There is **no turn reference at any speed other than the nominal walk's**, and none at pi/4 rad/s.

Recorded values (`stage1/runs/2026-11-08/<body>/refs.json`):

| body (H) | reference | period s | dt ms (rate) | speed m/s | stride m | lift m |
|---|---|---|---|---|---|---|
| 60-N-T-10 (0.4047 m) | nominal | 1.0198 | 6.374 (157 Hz) | 0.2183 | 0.2226 | 0.0405 |
| | brisk | 0.7964 | 4.978 (201 Hz) | 0.3150 | 0.2509 | 0.0445 |
| | fr05 | 0.7267 | 4.542 (220 Hz) | 0.4455 | 0.3238 | 0.0445 |
| | carpet | 1.0198 | 6.374 | 0.2183 | 0.2226 | 0.0769 |
| 60-N-H-10 (0.4015 m) | nominal / brisk / fr05 | 1.0157 / 0.7933 / 0.7238 | 6.348 / 4.958 / 4.524 | 0.2174 / 0.3138 / 0.4438 | 0.2208 / 0.2489 / 0.3212 | 0.0401 / 0.0442 / 0.0442 |
| 75-M-H-10 (0.5019 m) | nominal | 1.1356 | 7.098 (141 Hz) | 0.2431 | 0.2760 | 0.0502 |
| | brisk | 0.8869 | 5.543 (180 Hz) | 0.3508 | 0.3111 | 0.0552 |
| | fr05 | 0.8092 | 5.058 (198 Hz) | 0.4961 | 0.4015 | 0.0552 |
| | carpet | 1.1356 | 7.098 | 0.2431 | 0.2760 | 0.0903 |

Every turn has the nominal walk's period and lift; walking turns have its speed, in-place turns speed 0.
Heading per stride (`stride_motion[0]`) on 75-M-H-10: 0.3975 rad (0.35 rad/s), 0.5678 rad (0.5 rad/s); on 60-N-T-10:
0.3569 and 0.5099 rad. Every reference of the three picks has `ik_ok: true` and `matches_records: true`.

### 1.2 How a reference is made (`refs.py:97-134`)

```python
def source(body: str) -> dict            # refs.py:97  -> dict(body, M, q_home, design, H, layout, record, model)
def generate(src: dict, name: str) -> dict   # refs.py:113 -> dict(name, traj, idr, summary, how, seconds)
```

- `source` rebuilds the body with `pipeline.build_candidate(..., write=False)` in a temporary directory and checks
  the rebuilt MJCF against the record's `xml_sha256` (`plant.verify`). `q_home` is the solver's full-precision stance.
- `generate` first calls `gait.set_accuracy("r2")` (`ACCURACY = "r2"`, `refs.py:46`; process-wide module state).
  - Name in `walk_specs(body)`: `gait.generate` + `gait.inverse_dynamics` + `gait.summarize` (`refs.py:119-124`).
  - Any other name: `c = turn_cases()[name]` (`refs.py:126`, a `KeyError` for an unknown name), then
    `turning.generate(M, q0, H, sizing.NOMINAL, yaw_rate=..., in_place=...)`,
    `turning.inverse_dynamics(M, tr, turning.MU, split="exact", ds_friction=True)` (`MU = 0.8`), `turning.summarize`.
- `gait.generate` sends any spec with a look field set to `gait._generate_look` (`gait.py:193-194`);
  `turning.generate` refuses such a spec with `ValueError` (`turning.py:380-382`).

### 1.3 The reference npz layout (`refs.arrays`, `refs.py:145-166`)

Written by `np.savez(path, **arrays(src, g))` (`refs.py:293`), uncompressed. **The key order and set are part of the
file's sha256.** `n = 160` frames (one stride), uniform `dt = period / 160`; frame 0 is a left lift-off (phase 0), the
right foot lifts at phase 0.5.

| key | shape | meaning, unit |
|---|---|---|
| `t` | (160,) | s, `arange(n) * dt` |
| `q` | (160, nq) | qpos: root xyz, root quaternion (w first), then the hinges in `joints` order |
| `qd` | (160, nv) | qvel, central differences with the exact stride unwrap |
| `qdd` | (160, nv) | qacc |
| `tau` | (160, n_joints) | inverse-dynamics joint torque, N m (the feedforward; no MJCF damping or friction) |
| `omega` | (160, n_joints) | `qd[:, M.act_dofs]`, rad/s |
| `support` | (160, 2) bool | [left, right] in support |
| `com_ref` | (160, 2) | planned (LIPM) CoM xy, m |
| `zmp_ref` | (160, 2) | the ZMP **target** (pad centroid / hand-over), not the corrected reference |
| `zmp_mb` | (160, 2) | multibody ZMP of the solved poses |
| `cop_margin` | (160, 2) | per-foot CoP margin inside its pad, m; NaN for a foot not in support |
| `normal` | (160, 2) | per-foot normal force, N |
| `friction` | (160,) | largest tangential / normal ratio over the feet |
| `feet` | (160, 2, 3) | foot body positions, m |
| `com` | (160, 3) | model CoM, m |
| `q_home` | (nq,) | home pose |
| `joints` | (n_joints,) str | `M.names` |
| `dt` | () | s |
| `stride_motion` | (5,) | `(phi, a_x, a_y, b_x, b_y)`: next stride's pose is `turning.move_pose(q, phi, a, b)` |
| turns only: `path` | (160, 2) | path-frame origin |
| turns only: `heading` | (160,) | rad |
| turns only: `foot_target` | (160, 2, 3) | scheduled foot positions |
| turns only: `foot_yaw` | (160, 2) | rad |
| turns only: `swing` | (160, 2) bool | |

`stride_motion` for a straight walk is `[0, 0, 0, info["stride"], 0]` (`refs.py:142`): one stride, not a multi-stride
look window.

Joint order on the picks (`refs.json` `joints`): `neck_pitch, tail_yaw, left_hip_roll, left_hip_pitch, left_hip_yaw,
left_knee, left_ankle_pitch, left_ankle_roll, right_...` (14). On -X: `neck_yaw, neck_pitch, jaw, tail_yaw, ...` (16).

`Reference` (`refs.py:334-387`): `load(path)` (name = file stem), attribute access to arrays, `n`, `step` (= `dt`),
`period` (= `n * step`), `joint_series(qadr, dofs)`, `qdd_series(dofs)`, `at(t, series)`, `interp(t, stacked)` (linear,
wrapped periodically in the joint coordinates).

Keys the consumers read from a `Reference` (grep of `designlab/stage1/*.py`): `q, qd, qdd, tau, support, zmp_mb,
normal, stride_motion, joints, q_home, n, step, period, name` (`control.py:967-978`, `control.py:1099-1132`). A file
without `tau` and `normal` cannot be put on a `control.Track`.

**The Stage 2 look ladder's npz (`look.arrays`, `designlab/stage2/look.py:581-648`) has a different layout**: it
carries `t, dt, q, qd, qdd, support, com_ref, zmp_ref, zmp_mb, feet, com, q_home, joints, stride_motion` plus the look
keys (`phase, swing, swing_u, foot_pitch, ...`), but **no `tau`, `omega`, `cop_margin`, `normal`, `friction`**, and
its `stride_motion` spans `K` strides with `K * 160` frames.

### 1.4 `refs.prepare`, `refs.write` and `refs.json`

```python
def write(src: dict, g: dict, out_dir: Path) -> dict                 # refs.py:289
def prepare(body: str, out_dir: Path, names: list[str] | None = None) -> dict   # refs.py:307
```

`prepare` writes `<out_dir>/refs/<name>.npz` and `<out_dir>/refs.json`.

`refs.json` top level: `body, accuracy ("r2"), model {file, file_sha256, record_sha256, plan_prefix, matches_record,
matches_plan}, H_m, q_home [nq], joints [n], feedforward (text), match_rule (text), references {name: entry},
matches_records, seconds, provenance`.

Each entry (`refs.py:297-304`): `name, file ("refs/<name>.npz"), sha256 (of the npz), frames, dt_s, period_s,
speed_m_s, stride_m, lift_m, cadence_steps_s, yaw_rate_rad_s, stride_motion [5], min_cop_margin_single_m,
max_friction_ratio, ik_ok, generator, dynamics, spec {asdict(GaitSpec)}, [turns: case, yaw_rate, in_place, role], info
{the generator's info, with zmp_error_history_m of 7 values}, seconds, record_checks [..], matches_records`.

- `ik_ok` is `summary["ik_ok"]`: for walks `statics.leg_ik`'s convergence on every frame, for turns no foot pose error
  above `turning.IK_TOL = 1e-4`.
- ZMP error: `info.zmp_error_history_m` (m, the largest |multibody ZMP - target| after each of the 7 solves); turns
  add `zmp_error_exact_final_m` and `zmp_error_exact_final_frame`.
- `record_checks`: each `{record, leaves, mismatched, n_mismatched, missing, n_missing, noise_floor_matches,
  n_noise_floor_matches, n_noise_level_leaves, max_rel_diff, max_rel_diff_at, match}` (`refs.compare`,
  `refs.py:225-245`; 6 significant digits, `REL_TOL = 5e-7`, `NOISE_FLOORS` `refs.py:71-77`).

Facts a builder trips over:

1. `record_checks` (`refs.py:256-279`) treats every name that is not `nominal/brisk/fr05/carpet` as a turn of the
   turning screen and reads `review/screening/<cid>_turning.json` `runs[name]`. **`refs.write` raises `KeyError` for a
   new reference name.**
2. `prepare` **overwrites** `refs.json` with only the names it was given (`references=entries`, `refs.py:324-327`);
   it does not merge. (`transitions.prepare` does merge, section 2.6.)
3. The committed `refs.json` files hold `spec` with **11 keys**. Today `asdict(spec)` has 38 (11 + 27 look fields), so
   a `refs.json` written today is not the committed text even when every npz is bit-identical.
4. `walk_specs(body)` raises `KeyError` for a body missing from `CARPET_LIFT` (`refs.py:82`).

---

## 2. Transitions

### 2.1 The table and the name binding

```python
TRANSITIONS = {                                                      # transitions.py:120-131 (the last four lines abbreviated)
    "start": ("stand", "nominal"), "stop": ("nominal", "stand"),
    "nominal_to_brisk": ("nominal", "brisk"), "brisk_to_nominal": ("brisk", "nominal"),
    "brisk_to_fr05": ("brisk", "fr05"), "fr05_to_brisk": ("fr05", "brisk"),
    enter_{walk_0.35,walk_0.50}_{left,right}: ("nominal", turn),  exit_...: (turn, "nominal"),
    start_in_place_0.50_{left,right}: ("stand", turn),  stop_in_place_0.50_{left,right}: (turn, "stand"),
}
def kind_of(a: str, b: str) -> str     # transitions.py:134: "start" if a == "stand", "stop" if b == "stand", else "walk"
```

18 transitions. **The only walk stop is `stop` (nominal to stand) and the only walk start is `start` (stand to
nominal).** Brisk and Fr 0.05 "have no stop reference of their own" (`STAGE1_PLAN.md:384`, decision 31).

### 2.2 `Gait` and `gait_of`: why only `refs.walk_specs` walks are accepted

```python
@dataclass(frozen=True)
class Gait:                                                          # transitions.py:143-162
    name: str; period: float = nan; duty: float = nan; lift: float = 0.0   # m
    speed: float = 0.0; yaw_rate: float = 0.0; head_amp: float = 0.0; tail_amp: float = 0.0
    crouch: float = 0.0   # m
    zmp_iters: int = 0; zmp_gain: float = 0.0; split: str = "statics"   # or "exact" (turns)
    stand -> name == "stand"

def gait_of(body: str, name: str, H: float) -> Gait:                 # transitions.py:165-179
    if name == "stand": return Gait("stand")
    walks = refs.walk_specs(body)                                    # :169
    if name in walks:
        spec = walks[name]; info = spec.describe(H)
        return Gait(name, info["period"], spec.duty, spec.lift_frac * H, info["speed"], 0.0, spec.head_amp,
                    spec.tail_amp, spec.crouch * H, spec.zmp_iters, spec.zmp_gain, "statics")     # :173-174
    c = refs.turn_cases()[name]                                      # :175  KeyError for any other name
    spec = sizing.NOMINAL                                            # :176
    ...
```

The restriction has three layers:

1. **Lookup by name.** `gait_of` resolves a name against `refs.walk_specs(body)` (`:169`) and then
   `refs.turn_cases()` (`:175`); anything else is a `KeyError`. `generate` takes a transition *name* and does
   `a_name, b_name = TRANSITIONS[name]` then `gait_of(body, a_name, H)` (`transitions.py:641-642`); `_job` and
   `prepare` do the same (`:926-927`, `:941`), and `_load_ref` loads `<out>/refs/<name>.npz` (`:920-921`).
2. **`Gait` cannot carry a look.** It has 11 numbers: no look field, no `sway_gain`, no `tail_phase`, no lift shape.
   `gait_of` reads `spec.duty` (`:173`), not `describe(H)["duty"]`, so F3 (`swing_time`, which changes the duty in
   `GaitSpec.describe`, `gait.py:143-144`) would be dropped silently.
3. **The plan and the solver hard-code today's walk** (section 4, items T8-T14): the lift is `sin^2`, the tail is
   `sin(2 pi phase + pi/2)`, the neck is the nod, the trunk only yaws, the feet stay flat, the ZMP target is the pad
   centroid.

`designlab/stage2/look.py:184-188` (`DEFERRED["why"]`) states the same reason: "the turn and transition generators
... read spec.duty and today's head nod and tail law, so a look spec would silently give today's walk; turning.generate
now refuses a look spec (gait.look_active) and transitions.gait_of takes only refs.walk_specs' walks. Building these
rungs needs look-aware turn and transition paths, beside the old ones (G2 keeps those unchanged)".

Everything below `generate` already takes `Gait` objects, not names: `plan(name, a, b, geo, n)`, `com_path`,
`solve_frames`, `derivatives`, `multibody_zmp`, `idyn`, `arrays`. Checked on a synthetic plan: `plan("stop", Gait(
"v040", 0.861, 0.573, ...), Gait("stand"), geo)` runs. `kind` comes from the names (`a.name == "stand"`).

A precedent that bypasses `gait_of` exists and is **not** an official builder:
`stage2/jetson_plem/jetson_plem_study.py:505-575` builds a start to a hand-made `Gait("tuned_040", ...)` by calling
`plan / initial_state / com_path / solve_frames / derivatives / multibody_zmp / idyn / arrays` itself, with 2 ZMP
corrections instead of 6, `ik_ok=True` and `zmp_error_history_m=[0.0]` written in by hand (`:564-570`).
`JETSON_PLEM_REVIEW.md:103` records it as "hand-built (2 ZMP corrections instead of 6, `ik_ok` forced)".

### 2.3 The plan (`plan`, `transitions.py:296-426`; `_param_fns`, `:256-293`)

All timing is in gait phase, `N_FRAMES = 160` frames per stride (`:91`). Phase 0 is the transition's own stride
boundary, a **left** lift-off; left lifts at integer phases, right at half-integers (`:325`). A swing that lifts at `p`
lands at `p + 1 - duty(p)` with the lift `lift(p)`, both sampled at the lift-off (`:331`).

Constants (`:89-117`): `SUB = 16`, `PHASE_TOL = 1e-9`, `PREVIEW_QE = 1.0`, `PREVIEW_R = 1e-10`, `PREVIEW_S = 2.5`,
`SHIFT_STRIDES = 0.25`, `SETTLE_STRIDES = 0.25`, `PATH_HALF_WIDTH = 0.2`, `WALK_RAMP_STRIDES = 1.0`,
`WINDOWS = {"start": (-2, 3), "stop": (-2, 3), "walk": (-2, 4)}`, `PRE_STRIDES = 1.0`, `POST_STRIDES = 0.25`,
`G3_COM_MM = 1.0`, `G3_ZMP_MM = 2.0`, `COM_TOL = 1e-6`, `COM_ITERS = 12`, `IK_ITERS = 120`.

Common construction:

- Fine grid of `160 * 16` points per stride; `t = integral T dphase`, `heading' = w T`,
  `position' = v T (cos heading, sin heading)` by cumulative trapezoid, zero at phase `first`, starting at
  `geo.origin` (the hip-pitch midpoint of the home pose) (`:303-314`).
- Footprint of a stance = the home foot pose carried by the path frame to the stance's **mid-stance phase**; a stance
  that began in the initial stand takes the path pose at rest, a stance that never ends takes the path's final pose
  (`footprint`, `:333-342`).
- Swing: min-jerk position and yaw between consecutive footprints, lift `lift * sin(pi u)^2` (`:374-380`).
- ZMP target: the stance pad centroid in single support; in double support a linear hand-over from the foot about to
  lift to the foot that just landed (`:396-416`).
- Trunk: `root_z = root0_z - amp * crouch(phase)`, `neck = amp * head_amp * sin(4 pi phase)`,
  `tail = amp * tail_amp * sin(2 pi phase + pi/2)` (`:419-422`).
- Computed window `[first - 1, last + 0.25)`; output window `[first, last)`.
  Start and stop: 1000 frames computed, **800 output** (5 strides). Walk-to-walk: 1160 computed, **960 output**
  (6 strides).

**Start from standing** (`kind == "start"`, `:259-267`), `g` = the walk:

- Period, duty and lift are the walk's constants for the whole window, so the stand is measured in walk periods: the
  output begins 2 walk periods before the first lift-off.
- Only swings with lift-off `>= 0` exist (`:328`): the **left foot lifts first, at phase 0**.
- Weight shift: the ZMP target moves from the home CoM to the first stance foot (the right) over
  `[-SHIFT_STRIDES, 0]` (`:403-407`).
- Path speed and yaw rate ramp 0 to full (min-jerk) over `[phi_c - 0.2, phi_c + 0.2]`, `phi_c = 0.5 - duty / 2`.
  The ramp is antisymmetric about `phi_c`, so every footprint from the second step on is the periodic walk's: the
  left foot's first step is **half a stride**, the right's first is a full one.
- Neck, tail and crouch amplitude ramps 0 to 1 over `[-0.25, 1.0]`.
- Recorded ramps for `start` (duty 0.6): `{"path": [0.0, 0.4], "amplitude": [-0.25, 1.0], "weight_shift": [-0.25, 0.0]}`.

**Speed change, e.g. `brisk_to_fr05`** (`kind == "walk"`, `:278-286`):

- Every parameter (`T, duty, lift, v, w, head_amp, tail_amp, crouch`) is mixed
  `xa + (xb - xa) * minjerk(phase / 1.0)` over phases `[0, 1]`; amplitude 1 throughout. Recorded ramps:
  `{"all": [0.0, 1.0]}`.
- Output `[-2, 4)`: 2 strides of the first gait, the ramp stride, 3 strides of the second. 13 swings in the window.
- The frame interval shrinks through the ramp (non-uniform time grid); `derivatives` (`:530-552`) is second order on
  it. The saved `dt` is only the **mean** frame interval (`:876`).
- `T`, `duty`, `lift` and `v` share one weight. `look.speed_spec` (`designlab/stage2/look.py:295-304`) builds the
  0.40 m/s "v040" spec from exactly that mix, between brisk and fr05.
- Turn entries and exits are the same kind: the yaw rate ramps over one stride, then holds for 3 more strides of
  output.

**Stop** (`kind == "stop"`, `:268-277`), `g` = the walk:

- Only swings with lift-off `<= 0` exist (`:330`): **the left foot's lift-off at phase 0 is the last step**. It lands
  at `1 - duty` beside the right foot, in the home stance; nothing lifts after it.
- Path speed and yaw rate ramp to 0 over the same window `[phi_c - 0.2, phi_c + 0.2]`. The final path pose is the
  periodic walk's at `phi_c`, so the right foot's last footprint is the periodic one and the left's last step is half
  a stride.
- Amplitude ramps 1 to 0 over `[0, end]`, `end = 1 - duty + SETTLE_STRIDES`.
- After the last touchdown the ZMP target moves from the other foot's centroid to the final stand's CoM over
  `SETTLE_STRIDES` (`:408-411`).
- Recorded ramps for `stop` (duty 0.6): `{"path": [0.0, 0.4], "amplitude": [0.0, 0.65], "settle": [0.4, 0.65]}`.

### 2.4 CoM, poses, ZMP corrections, inverse dynamics (`generate`, `transitions.py:634-678`)

```python
def generate(src: dict, name: str, ref_a: refs.Reference | None = None, verbose: bool = False) -> dict
```

1. `gait.set_accuracy(refs.ACCURACY)`; `a, b = gait_of(...)`; `walk = b if a.stand else a`;
   `geo = Geometry.of(M, q_home, walk.crouch)` (`:643-644`); `pl = plan(name, a, b, geo)`.
2. `x0 = initial_state(pl, ref_a, geo)` (`:603-631`): at rest for a stand; otherwise the first gait's `com_ref` one
   stride before the output window, by 4th-order differences across the stride boundary using `ref_a.stride_motion`.
   **`ref_a` must be the saved periodic reference of the first gait.**
3. Loop (`:651-663`):

```python
iters = walk.zmp_iters if b.stand else b.zmp_iters      # :648
gain = walk.zmp_gain if b.stand else b.zmp_gain         # :649
for it in range(iters + 1):
    com_t = com_path(pl, zref, x0, geo.omega)           # ZMP preview control
    qs, foot_err, com_err = solve_frames(M, q_home, pl, com_t, geo)
    qd, qdd = derivatives(M.m, qs, pl.t)
    zmb = multibody_zmp(M, qs, qd, qdd)
    e = np.nan_to_num(zmb - pl.zmp)
    history.append(float(np.max(np.linalg.norm(e[pl.out], axis=1))))
    if it == iters: break
    zref = zref - gain * e
```

   **Number of ZMP corrections: 6, at gain 0.5 (7 solves)** for every existing transition, because every existing
   spec has `zmp_iters = 6`, `zmp_gain = 0.5`. `zmp_error_history_m` in `transitions.json` has 7 entries. The error
   is measured on the output frames only; the correction is applied to every computed frame. The Stage 2 look rungs
   use `LOOK_ZMP_ITERS = 12` (`designlab/stage2/look.py:168`).
4. `com_path` (`:492-524`): preview control on a uniform grid at the smallest frame interval, gains from
   `scipy.linalg.solve_discrete_are` (`preview_gains`, `:432-454`), 2.5 s of preview; past the last frame the ZMP
   reference is the last stride carried on by the second gait's stride motion with `pl.b.period` (a constant for a
   stand).
5. `solve_frames` (`:555-584`): per frame, root z, trunk yaw, `neck_pitch` and `tail_yaw` from the plan, the pelvis
   shifted until the model CoM is within `COM_TOL` of the path (at most `COM_ITERS` passes), legs by
   `statics.leg_ik(M, q, targets, iters=120)`.
6. `idyn` (`:681-703`): `turning.contact_torques(..., split="exact")` if either gait is a turn, else
   `statics.contact_torques`.
7. `ik_ok = bool(np.all(foot_err[o] <= turning.IK_TOL))` over the output frames (`:677`).

Run time (the records' `seconds`): 43-74 s per transition on 75-M-H-10, 56-93 s on the 60 cm picks; 18.5-23 min
summed over a body's 18.

### 2.5 Gate G3 (`compare`, `transitions.py:719-757`; `summary`, `:883-907`)

The last output stride against the second gait's reference (or the home stance at the final path pose): model CoM
(3D) within 1 mm, multibody ZMP within 2 mm, and `ik_ok`. `g3.pass_ = com_ok and zmp_ok and ik_ok`. The first output
stride is compared with the first gait the same way (`join_a`); it is recorded, not gated. `gates.g3`
(`designlab/stage1/gates.py:718-744`) re-reads `transitions.json` and adds `missing = [n for n in tr.TRANSITIONS if n
not in rows]`; `pass_` needs no failure and nothing missing.

Records: all 18 transitions pass on all three picks, both K bodies and both -X bodies. Largest G3 values on the picks:
CoM 0.28-0.31 mm, ZMP 0.96-1.08 mm (`brisk_to_fr05` is the worst ZMP, 1.083 mm on 75-M-H-10).

### 2.6 `transitions.prepare`, the npz layout and `transitions.json`

```python
def prepare(body: str, out_dir: Path, names: list[str] | None = None, procs: int | None = None,
            verbose: bool = False) -> dict                           # transitions.py:932
```

- Needs `<out_dir>/refs/<name>.npz` of both gaits already there (`_load_ref`).
- `procs = min(procs or os.cpu_count() or 1, len(jobs))` (`:943`): **by default one spawned process per CPU**; each
  job calls `refs.source(body)` again.
- **Merges into an existing `transitions.json` and rewrites it** (`:950-960`).

npz (`arrays`, `transitions.py:865-876`), `N` = output frames (800 or 960); key order as written:

| key | shape | meaning |
|---|---|---|
| `t` | (N,) | s from 0; non-uniform through a speed change |
| `phase` | (N,) | strides, from `first` in steps of 1/160 |
| `q`, `qd`, `qdd` | (N, nq), (N, nv), (N, nv) | |
| `tau` | (N, n_joints) | N m |
| `omega` | (N, nv - 6) | `qd[:, 6:]` |
| `support`, `swing` | (N, 2) bool | `support = ~swing` |
| `swing_u` | (N, 2) | swing progress 0-1, NaN in stance |
| `com_ref` | (N, 2) | preview-control CoM path |
| `zmp_ref` | (N, 2) | the plan's ZMP target |
| `zmp_mb` | (N, 2) | multibody ZMP |
| `zref` | (N, 2) | the corrected ZMP reference fed to the preview controller |
| `cop_margin`, `normal` | (N, 2) | |
| `friction` | (N,) | |
| `feet` | (N, 2, 3) | solved foot positions |
| `com` | (N, 3) | model CoM |
| `foot_target` | (N, 2, 3) | planned foot positions |
| `foot_yaw` | (N, 2) | rad |
| `path` | (N, 2) | path-frame origin |
| `heading` | (N,) | rad |
| `stand_point` | (2, 2) | [initial, final] stand CoM xy |
| `q_home` | (nq,) | |
| `joints` | (n_joints,) str | |
| `dt` | () | mean frame interval, s |

`Transition` (`:964-984`): `load`, attribute access, `n`. Keys consumers read: `t, q, qd, qdd, tau, support, zmp_mb,
normal, swing_u, phase, path` (`control.py:980-1003`), `heading, com` (`ledger.py`).

`transitions.json` top level: `body, transitions {name: entry}, method, preview {qe, r, preview_s}, shift_strides,
settle_strides, path_half_width, walk_ramp_strides, windows, g3_com_mm, g3_zmp_mm, seconds, provenance`, and
`stop_reading {rule, reread_from_arrays, provenance}` where `reread_stops` ran (the three picks; not K or -X).

Each entry (`summary` + `write`, `:895-917`):

- `name, kind, from_, to, gaits {a: asdict(Gait), b: asdict(Gait)}, output_phase [first, last], frames, duration_s,
  ramps`;
- `steps`: each `{side, lift_off_phase, touchdown_phase, lift_m, from_xy, from_yaw, to_xy, to_yaw, length_m}`, for
  the swings with `first - 0.5 <= lift_off < last`;
- `zmp_iters, zmp_gain, zmp_error_history_m [iters + 1] (m), split, ik_ok, com_tracking_max_m`.
  `zmp_error_history_m[k]` is the largest |multibody ZMP - ZMP target| over the output frames after solve `k`. Its
  last value is 1.6-1.7 mm for the 75 cm `start` and `stop` and at most 3.3 / 3.2 / 3.8 mm on 60-N-T-10 / 60-N-H-10 /
  75-M-H-10 (`brisk_to_fr05` each time). It is a different quantity from G3's `zmp_max_mm` (against the periodic
  walk's multibody ZMP);
- `min_cop_margin_single_m, min_cop_margin_m, max_friction_ratio`. `min_cop_margin_m` is negative in every record
  (-0.001 to -0.031 m on 75-M-H-10, -0.006 to -0.022 m on the 60 cm picks): it includes the per-foot split in double
  support. The single-support value is the meaningful one (0.043-0.045 m on 75-M-H-10, 0.029-0.031 m at 60 cm);
- `stop` (null unless `kind == "stop"`): `{rule, command_speed_m_s, stride_s, forward_stopped_strides, within,
  at_rest_strides, at_rest_s, command_phase, last_touchdown_phase, single_step}`;
- `join_a`, `join_b`: `{stride_phase, reference, com_max_mm, com_mean_mm, zmp_max_mm, zmp_mean_mm, joint_max_rad,
  joint_rate_max_rad_s, at_join {frame, com_mm, zmp_mm, joint_rad, joint_rate_rad_s[, tau_Nm]}[, tau_max_Nm], com_ok,
  zmp_ok}`;
- `g3 {rule, com_max_mm, zmp_max_mm, pass_}`, `seconds`, `file ("transitions/<name>.npz")`, `sha256`;
- on the picks' stops, `stop_before_decision_18` (the earlier 1 % reading).

Consumers read `gaits.b.duty` of the first transition (`tune.py:484`, `panels.py:656`) and `gaits.a.{speed, period,
duty}` (`transitions.py:851-853`).

### 2.7 The stop reading and the "1.5-stride rule"

`STAGE1_PLAN.md:236-238`, decision 18: "'Stopped within 1.5 strides' means forward speed below 5 % of the command
within 1.5 strides of the stop command. The stop reference ends forward motion after 0.76-0.79 strides and reaches
full rest after 1.55-1.57 strides; the time to rest is reported."

```python
STOP_SPEED_FRAC = 0.05; STOP_WITHIN_STRIDES = 1.5; REST_SPEED_M_S = 1e-3; REST_DISTANCE_M = 1e-3   # :762-765
REST_SPEED_R2_M_S = 3.5e-3                                                                         # :771
def stop_command_phase(duty: float) -> float: return 0.5 - duty / 2 - PATH_HALF_WIDTH              # :783
def stop_reading(t, v_forward, command_speed, t_command, period, rest=None) -> dict                # :796
def stop_time_arrays(phase, t, com_ref, heading_final, stand_final, speed, period, duty) -> dict   # :813
def stop_time(g: dict) -> dict | None                                                              # :828
def reread_stops(out_dir: Path) -> dict                                                            # :837
```

- The **stop command** is defined as the phase where the path speed starts to ramp down.
- "Stopped" = `|v_forward| < 0.05 * command_speed` from then on; `within = forward_stopped_strides <= 1.5`.
- The stride is the first gait's period.
- `stop_time_arrays` returns `single_step=True` and `last_touchdown_phase = 1 - duty` as constants (`:825`).
- `STOP_WITHIN_STRIDES = 1.5` is defined three times: `transitions.py:763`, `gates.py:749`, `tune.py:166`.
- `STOP_RULE`'s text says the command is "the start of the stop's speed ramp, the last lift-off" (`:776-780`).
  That equality holds only at duty 0.6.

Recorded stop readings of the references (decision 18):

| body | `stop` forward stopped (strides) | at rest (strides) | command phase |
|---|---|---|---|
| 60-N-T-10, 60-N-H-10 | 0.519 | 1.55 | 0.0 |
| 75-M-H-10 | 0.525 | 1.569 | 0.0 |
| 75-M-H-10-X | 0.531 | 1.569 | 0.0 |

In-place stops have no forward reading (`None`); at rest after 1.544-1.575 strides. The K bodies read 0.519-0.525 and
1.544-1.55. The plan's "0.76-0.79 strides" is the earlier reading at 1 % of the walk's speed, kept in the picks'
records as `stop_before_decision_18` (0.7875 on 75-M-H-10).

Command phase against duty (`stop_command_phase`, computed): 0.60 -> 0.0; 0.59 -> 0.005; 0.58 -> 0.010;
0.573 -> 0.0135; 0.56 -> 0.020; 0.55 -> 0.025; 0.69 -> -0.045. **For any duty below 0.6 the command falls after the
last lift-off**: the last swing is already in the air when the "command" is given.

`control.Track.add_transition` (`designlab/stage1/control.py:994-1001`) does not read the command from the record. It
recognises a stop **by name** and recomputes the command from the arrays:

```python
if TRANSITIONS.get(tr.name, (None, None))[1] == "stand" and "phase" in tr.arrays:
    n0 = min(N_FRAMES, len(tr.t) - 1)
    duty = float(np.mean(tr.support[:n0, 0]))
    ph = stop_command_phase(duty)
    speed = float(np.linalg.norm(tr.path[1] - tr.path[0]) / (tr.t[1] - tr.t[0]))
    self.events.append(dict(kind="stop", name=tr.name, t_ref=..., command_phase=ph, speed_m_s=speed,
                            period_s=float(tr.t[n0] - tr.t[0])))
```

**A stop transition whose file name is not a key of `transitions.TRANSITIONS` registers no stop event**, so it gets no
stop reading. `tune.build_spec` then raises "expected one stop" unless it is a custom walk plan (`tune.py:486-488`).

### 2.8 A "two-step stop" from the existing stop builder

What the bar asks (`STAGE2_PLAN.md:178`): "at most two touchdowns (`metrics.py`'s 2 % load rule) after the stop
command before the forward speed stays under 5 % of the command (decision 18); deceleration (1 Hz) at most 1.5 m/s^2
(decision 10)". That is a reading on the simulated episode's contacts; the reference only has to permit it.

What the existing builder gives (checked on a synthetic plan, duties 0.55, 0.573, 0.59, 0.60, 0.69):

- **Exactly one touchdown after the command, at every duty**: the left foot's, at phase `1 - duty`. The right foot's
  last touchdown is at `0.5 - duty`, before the command. So the existing one-step stop, built for a 0.40 m/s gait,
  already has at most two touchdowns in the reference.
- Its cost at 0.40 m/s is the deceleration. The path speed ramps down over `2 * PATH_HALF_WIDTH = 0.4` strides. The
  peak of a min-jerk ramp is 1.875 x the mean, so the peak path deceleration is `1.875 v / (0.4 T)` (my arithmetic):
  1.00 m/s^2 for the 75 cm nominal stop, **2.18 m/s^2** for 75 cm at 0.40 m/s (T 0.861 s), about 2.5 m/s^2 for 60 cm
  (T 0.751 s). This is the path frame's, not the 1 Hz low-passed CoM value the bar judges. No record measures it at
  0.40 m/s.

Where a second step would come from, in the builder's own terms:

1. The swing schedule needs no change. A stop has two touchdowns after its command as soon as the command phase is
   earlier than the right foot's last touchdown, `0.5 - duty`. The command is the start of the speed ramp, so this
   means starting (or widening) the ramp of `_param_fns`' `down` (`:269-271`) before phase `0.5 - duty`.
2. `footprint` (`:333-342`) already handles it: the right foot's last stance never ends, so its footprint is the final
   path pose. With an earlier ramp the right foot's last step becomes a shortened step into the final stance, and the
   left then lands beside it.
3. What has to move with the ramp:
   - `stop_command_phase` (`:783-785`), which three other modules import (`control.py:135`, `panels.py:72`,
     `tune.py:99`) and which `tune.py:485` and `panels.py:657` also use for the **start** command;
   - `control.py:994-1001` (above), which recomputes the command from the duty;
   - `stop_time_arrays`' constants `single_step=True`, `last_touchdown_phase` (`:825`);
   - the amplitude ramp's `end` and the `ramps` record (`:272-277`).
4. Constraints from the window: `WINDOWS["stop"] = (-2, 3)` leaves 2 strides of steady walking before phase 0. The
   first output stride's swings land on footprints with mid-stance up to phase -0.8 (duty 0.6), so a ramp that starts
   before about -0.8 changes the first output stride and the `join_a` comparison.
5. `PATH_HALF_WIDTH`'s comment (`:102-104`) says "at most 0.2 keeps the first regular footprint, mid-stance phi_c +
   0.5, outside the ramp". That bound is the start's; the constant is shared by start and stop.

---

## 3. Hashing and provenance

### 3.1 What `refs.json` and `transitions.json` themselves record

- Per array: `sha256` of the npz file (`refs.py:297`, `transitions.py:916`).
- `provenance = plant.provenance()` (`designlab/stage1/plant.py:294-316`): library versions, git commit, dirty flag
  and dirty paths, and `stage1_sources {sha256, rule, files}` = sha256 over **every** `designlab/stage1/*.py` (22
  files; `SOURCES_RULE`, `plant.py:263-264`).
- **Neither record hashes its own module or its inputs separately.** The inputs are recorded by value: `spec`
  (refs), `gaits` (transitions), the model's `xml_sha256`.
- The three picks' committed `refs.json` / `transitions.json` predate `stage1_sources`: their `provenance` has only
  `versions` and `git`. The K and -X records have it.

### 3.2 Records that hash these modules or these files

| hash | covers | where |
|---|---|---|
| `plant.sources_sha256()` | all 22 `designlab/stage1/*.py` | `plant.py:267-275`; every `provenance()`; `train.py:461-465` refuses a resume on a difference (unless `--allow-source-change`) |
| `tune.tuner_sources()` | `__init__, control, drive, estimator, legs, metrics, plant, refs, transitions, tune` | `tune.py:374-435`; resume guard and G5 |
| `panels.panel_sources()` | the tuner's ten + `panels.py`, and designlab modules `__init__, build, carpet, catalog, clearance, envelope, gait, geometry, hardware, mjcf, params, shapes, sizing, skeleton, statics, template, turning` | `panels.py:1559-1583`; in every record key |
| `speedfit.speed_sources()` | `__init__, control, drive, env, estimator, gates, legs, metrics, panels, plant, refs, speedfit, transitions, tune` | `speedfit.py:525-529` |
| inputs | sha256 of every `refs/*.npz` and `transitions/*.npz` a run loads, **and of `refs.json` and `transitions.json` themselves** | `tune.read_inputs`, `tune.py:320-347`; in every panel record key (`panels.py:1649`), every tuner and speed-fit session |

Consequences, as facts:

1. **Any byte change to `refs.py` or `transitions.py` changes four code hashes** (package, tuner, panels, speed fit).
2. **Any byte change to `designlab/gait.py` or `designlab/turning.py` changes the panels' hash** (its `designlab`
   part).
3. **Rewriting `refs.json` or `transitions.json` in a run directory changes the input hash of every record made from
   that directory**, even if no array changes. `transitions.prepare` rewrites the file on every call; `refs.prepare`
   overwrites it; `reread_stops` rewrites it.
4. The committed records already read as made by other code. Current `panel_sources` sha256 begins `6a316fc8`
   (stage1 part `0306af7e`, designlab part `57b564b6`); the 2026-11-08 panel records hold `7516b927`
   (`5aae9e9b` / `16dd446d`), the 2026-11-10-speed ones `733f2b80` (`a5a8657c` / `16dd446d`). Current tuner hash
   `1d936548`; the 2026-11-08 `tune.json` holds `c8b112cf`.
5. Current file hashes: `refs.py` sha256 `e3f76701...`, `transitions.py` `72dfe5ee...`. `transitions.py` was last
   changed in commit `6864d10` (10 Nov), `refs.py` in `c324645` (11 Nov, the -X carpet lifts).

### 3.3 `stage1_inputs.json` and `tools/restore_inputs.py`

`stage1/runs/2026-11-08/stage1_inputs.json`: `{rule (text), bodies [3], cases {body: case}, files {path relative to
the study: sha256}}`. 90 files: 6 model files (each pick's `round2/refine/<case>/models/<id>.xml` and `.json`) and 84
arrays (per pick 10 `refs/*.npz` + 18 `transitions/*.npz`).

How the arrays are regenerated (`restore_picks_arrays`, `tools/restore_inputs.py:610-643`):

- References: one child process per body, `refs.prepare(body, <scratch>/refs/<body>, names)` (`_refs`, `:539-545`).
- Transitions: one body at a time, `transitions.prepare(body, <scratch>, names, procs=procs)` with `<scratch>/refs`
  a symlink to the run's own `refs/` (`_transitions`, `:548-555`).
- Child environment (`_env`, `:493-502`): `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1`,
  `MUJOCO_GL=disable`, `PYTHON_CPU_COUNT = procs`. Docstring `:31-33`: "with several, the inverse dynamics'
  double-support QP differs in the 15th digit".
- A regenerated file is put in place **only if its sha256 is the listed one** (`put_matching`, `:193-208`), through a
  temporary name. The scratch `refs.json` / `transitions.json` are discarded; no committed file is written.
- Order `GROUPS = ("picks-models", "picks-arrays", "k", "x", "look")`; `NEEDS = {"look": ("picks-models",)}`.
- The K and -X bodies' arrays are restored by their own helpers (`stage1/runs/2026-11-10-speed/arm_k_inputs.py`,
  `stage2/x/x_inputs.py`), which call the same two `prepare` functions.

CLI: `python tools/restore_inputs.py {check | restore | copy-from DIR | not-restored | manifest [--from DIR]}`
with `--group G ...`, and for `restore` `--procs 2`, `--scratch DIR`.

---

## 4. Hard-coded assumptions the build list would break

`refs.py`:

| # | where | assumption | breaks |
|---|---|---|---|
| R1 | `:80-82` | four walks, from three module constants and `CARPET_LIFT[body]` | A, D |
| R2 | `:126-127` | any non-walk name is a `turning.CASES` turn on `sizing.NOMINAL` | B |
| R3 | `:142` | a straight walk advances one stride per file | look windows with `strides > 1` |
| R4 | `:155-165` | the array key set and order (sha256) | any added key |
| R5 | `:261-279` | every name has a round-two record to compare with | A, B (new names raise `KeyError`) |
| R6 | `:317-327` | `prepare` owns the whole `refs.json` | adding one reference to a run directory |
| R7 | `:46`, `:115` | `gait.set_accuracy("r2")` is set process-wide and left set | callers mixing accuracies |

`transitions.py`:

| # | where | assumption | breaks |
|---|---|---|---|
| T1 | `:91` | 160 frames per stride; also `control.py:985-996, 1130` | a reference with another `n`, or a `K`-stride window |
| T2 | `:109` | windows: 2 strides before, 3 after (walk), 5 in all (start, stop) | B (below), the two-step stop's lead-in |
| T3 | `:120-131`, `:641`, `:926`, `:941` | a transition is a name in `TRANSITIONS` | A, B |
| T4 | `:165-179` | a gait is a `refs.walk_specs` walk or a `turning.CASES` turn | A, B, D |
| T5 | `:173` | `spec.duty` is the duty | F3 `swing_time` |
| T6 | `:143-158` | 11 numbers describe a gait; `sway_gain` is not among them | look fields; any `sway_gain != 1` |
| T7 | `:325-331` | left lifts first; a start begins and a stop ends with the left foot | mirrored starts and stops |
| T8 | `:378` | lift `sin^2` | F4 `lift_peak` |
| T9 | `:385-387` | ZMP target = pad centroid | F7 `zmp_inset_y`, F5's rolling support |
| T10 | `:420` | `root_z = root0_z - amp * crouch` | F1/F2 bob |
| T11 | `:421-422` | nod `sin(4 pi phase)`; tail phase `pi/2` | F9, F10, F14 |
| T12 | `:560-561`, `:571` | only `neck_pitch` and `tail_yaw` are driven; other hinges stay at `q_home` | F: -X neck yaw and jaw motion in a transition |
| T13 | `:568`, `:570` | feet flat (yaw only), trunk yaw only | F5 toe-off and toe reach, F6 roll |
| T14 | `:359`, phase grid | phase advances uniformly per frame | F14 jitter and asymmetry |
| T15 | `:643-644` | one LIPM frequency per transition, from the **first** gait's crouch | a change between crouches (V0 0.03 H to F25 0.05 / 0.07 H) |
| T16 | `:648-649` | pass count and gain come from the second gait's spec | look rungs' 12 passes are a spec value, so they would follow |
| T17 | `:665`, `:681-703` | contact split is `statics` or `exact` | F5's rolling contact (`look.look_inverse_dynamics`) |
| T18 | `:713-716` | a reference without `path` is a straight walk starting at the plan's origin | |
| T19 | `:724-743` | `compare` slices 160 frames and compares with the whole reference | a reference with `n != 160` |
| T20 | `:783-785`, `:825` | the stop command is `phi_c - 0.2`; one step | C, the two-step stop |
| T21 | `:869-876` | array key set and order | any added key |
| T22 | `:920-921` | references live at `<out>/refs/<name>.npz` with `refs.arrays`' keys | look npz (no `tau`, `normal`) |
| T23 | `:943` | a pool of `os.cpu_count()` processes by default | shared machines |
| T24 | `:950-960` | `prepare` merges into and rewrites `transitions.json` | 3.2 item 3 |

T2 and turns, in numbers. An `enter_*` transition's output is 2 straight strides, the 1-stride yaw ramp (worth half a
stride of yaw) and 3 strides at the full rate: `3.5 w T` of heading. `exit_*` adds `2.5 w T`. Recorded:
`enter_walk_0.35_left` on 75-M-H-10 ends with its last footprint at 1.4706 rad (84.3 deg), `enter_walk_0.50_left` at
120.4 deg. At 0.40 m/s on 75 cm (T 0.861 s) and pi/4 rad/s, `w T` = 0.676 rad = 38.7 deg (my arithmetic), so an entry
plus an exit with no steady stride between them turns `6 w T` = 232 deg. **A 90 degree turn at pi/4 rad/s cannot be
assembled from `enter_*` + `exit_*` with these windows and ramps.**

Outside these two files, but bound to their names:

- `control.py:994`: a stop is a name whose `TRANSITIONS` target is `"stand"` (section 2.7).
- `gates.py:734`: G3 fails if any `TRANSITIONS` name is missing from `transitions.json`.
- `tune.py:278-285`, `panels.py:523-545`, `env.py:79-94`: plans are lists of `("transition", name)`,
  `("ref", name, strides)`, `("hold", seconds)` with these literal names.
- `tune.py:553`, `panels.py:1200-1204`: stop readings are found by the segment name `"stop"`.
- Six call sites take `omega0` from `refs/nominal.npz` with the default `crouch_frac = 0.03`
  (`control.py:943-949`; `env.py:380`, `gates.py:862-863`, `metrics.py:432`, `panels.py:996`, `speedfit.py:432`,
  `tune.py:805`).

---

## 5. Tests that pin this behaviour

`tests/test_stage1.py`:

| test | asserts |
|---|---|
| `test_nominal_reference_matches_the_record` (`:524`) | 60-N-T-10 nominal: speed 0.218266 m/s, single-support CoP margin 0.030316 m; every record check matches, more than 80 leaves, no noise-floor match |
| `test_record_comparison_catches_a_difference` (`:533`) | a 2e-6 relative change fails, 1e-7 passes |
| `test_noise_level_leaves_match_within_their_floor` (`:543`) | `NOISE_FLOORS` behaviour |
| `test_reference_file_round_trip` (`:558`) | 160 frames, `period`, `tau`, `joints`, `at` / `interp` interpolation and periodic wrap |
| `test_transition_plans_are_continuous` (`:761`) | every `TRANSITIONS` plan: `diff(phase) == 1/160`, no jump in feet, ZMP, heading, height, tail; each swing starts and lands on its footprints |
| `test_start_and_stop_steps` (`:788`) | start: first step `("left", 0.0)` of half a stride, then full strides; stand then shift. Stop: last swing `("left", 0.0)` of half a stride; final feet in the home stance; ZMP ends at the final stand; no swing in the last 160 frames |
| `test_transition_plan_steady_part_is_the_periodic_walk` (`:807`) | the start's last stride equals the nominal reference's ZMP target (1e-9), feet (2e-4), support |
| `test_preview_control_reproduces_the_periodic_lipm` (`:820`) | preview control within 2e-5 m of `gait.generate`'s CoM |
| `test_transition_derivatives_on_a_nonuniform_grid` (`:838`) | exact on a quadratic; NaN at the ends |
| `test_transition_records_pass_g3` (`:851`) | committed `transitions.json`: `set(rows) == set(TRANSITIONS)`, `(from_, to)` match, `ik_ok`, G3 limits, `join_a` CoM < 0.1 mm; stops: `at_rest_strides < 1.6`, last `lift_off_phase == 0.0`, `stop.rule == STOP_RULE`, `command_phase == 0.0`; `gates.json` G3; walk stops 0.4-0.6 strides, at rest 1.5-1.6 |
| `test_g3_cli_reproduces_the_record` (`:880`) | `gates --only g3` on the committed `transitions.json` equals the committed G3 exactly |
| `test_stop_reading_is_decision_18` (`:1468`) | `STOP_SPEED_FRAC == 0.05`, `STOP_WITHIN_STRIDES == 1.5`, the reading's arithmetic, `stop_command_phase(0.6) == 0.0`, `control._stop_result` |
| `test_tuner_sources_are_the_modules_the_tuner_imports` (`:1937`) | the ten modules, `refs.py` and `transitions.py` among them; any byte change changes the hash |
| `test_inputs_are_checked_against_their_records` (`:1984`) | 15 mix files; each npz against its listed sha256; both JSON files hashed |
| `test_g2_references_match_and_the_nominal_replay_reproduces_the_gate_record` (`:2334`, `STAGE1_SLOW=1`) | `refs.prepare` on a temporary directory matches the records; replay strides equal the gate record |

Other files:

| test | asserts |
|---|---|
| `tests/test_stage2_look.py::test_g2_stage1_references_regenerate_bit_for_bit` (`:140`) | all 10 references of 60-N-T-10 and 75-M-H-10 regenerate to the npz sha256 in `refs.json` (`python -m designlab.stage2.look g2`, one BLAS thread) |
| `tests/test_stage2_look.py::test_turning_generate_refuses_a_look_spec` (`:113`) | `ValueError` on a look spec; `"turns"` and `"starts_and_stops"` in `look.DEFERRED` |
| `tests/test_stage2_look.py::test_look_fields_default_to_todays_walk` (`:75`) | the 11 old `GaitSpec` fields come first, unchanged; defaults are not a look |
| `tests/test_stage1_bodies.py` (`:200-206`), `tests/test_stage2_x.py` (`:459-464`) | K and -X records: `set(references) == set(refs.reference_names(body))`, `set(transitions) == set(transitions.TRANSITIONS)` |
| `tests/test_restore_inputs.py` | `stage1_inputs.json` lists 6 + 84 files; each array's sha256 is the one `refs.json` / `transitions.json` records |
| `tests/test_stage1_panels.py:693` | `transitions.py` is among the panels' source modules |

**No pytest regenerates a transition npz and compares its sha256.** The transitions' bit-identity is checked only by
the restore tools (`restore_inputs.py restore --group picks-arrays`, `arm_k_inputs.py arrays`, `x_inputs.py arrays`),
which refuse a regenerated file whose sha256 differs. Stage 2's G2 covers references only.

Adding a key to `TRANSITIONS` fails `test_transition_records_pass_g3`, `test_g3_cli_reproduces_the_record` (through
`gates.g3`'s `missing`) and the two set-equality tests on the K and -X records.

---

## 6. CLI entry points

`python -m designlab.stage1 prepare` (`designlab/stage1/__main__.py:667-743`):

| argument | meaning |
|---|---|
| `--body B` (repeatable, required) | a key of `stage1.BODIES` |
| `--out RUN` (required) | run directory; files go to `RUN/<body>/` |
| `--parts plant,refs,transitions` | products to make; `stops` re-reads the saved stops (`reread_stops`) |
| `--refs a,b` | reference names (default: all 10) |
| `--transitions a,b` | transition names (default: all 18) |
| `--no-gates` | skip the gates; **without it `prepare` goes on to run G1-G4** (`:738-740`) |
| `--ankle-gains`, `--ankle-armature`, `--g2-rule`, `--seed` | plant and gate options |

`python -m designlab.stage1 gates --body B --out RUN [--only g1|g2|g3|g4]`: G3 reads `RUN/<body>/transitions.json`.

`python -m designlab.stage2.look g2 --body B --name N ...`: regenerates Stage 1 references and prints their sha256
(`look.g2_hashes`, `designlab/stage2/look.py:1578-1591`).

`refs.py` and `transitions.py` have no `__main__` of their own.

---

## 7. Extension points and hazards (short)

Extension points that exist today:

1. `transitions.plan`, `com_path`, `solve_frames`, `derivatives`, `multibody_zmp`, `idyn`, `arrays`, `compare`,
   `summary` and `Transition` take `Gait` objects or plain arrays. A caller outside `designlab/stage1/` can drive them
   with its own `Gait`, as `jetson_plem_study.py` did, without changing a byte of `transitions.py`. Only `generate`,
   `_job`, `prepare` and `gait_of` are name-bound.
2. `turning.generate(M, q_home, H, spec, yaw_rate, in_place)` already accepts any non-look spec and any yaw rate. A
   0.40 m/s, pi/4 rad/s periodic turn of today's walk needs no edit to it; a look-aware one does.
3. `refs.arrays(src, dict(traj=..., idr=...))` and `refs.Reference(name, arrays)` build a loadable reference from any
   `gait.generate` trajectory with its inverse dynamics.
4. A module under `designlab/stage2/` is in none of the four Stage 1 code hashes unless a Stage 1 module imports it.

Hazards:

1. Bit-identity of existing transitions is not under pytest (section 5). An edit to `transitions.py`, `statics.py`,
   `turning.py` or `gait.py` that changes them shows only in a restore run.
2. The npz sha256 depends on key order, dtypes, the numpy / mujoco / scipy versions and one BLAS thread per process.
3. New references and transitions written into an existing run directory change that directory's `refs.json` /
   `transitions.json` hashes (3.2 item 3).
4. A new stop needs `control.py:994` to see it as a stop (section 2.7); a new name must not be added to
   `TRANSITIONS` (section 5).
5. `stop_command_phase` is shared by starts and stops and by three consumers; at duty below 0.6 it already falls
   after the last lift-off.
6. `Geometry.of` takes one crouch; a transition between walks of different crouch runs its preview control at the
   first walk's LIPM frequency (T15).
7. Look npz files are not Stage 1 references: no `tau`, no `normal`, `K * 160` frames (section 1.3).
8. `transitions.prepare` defaults to one process per CPU, and `prepare` on the CLI runs the gates unless
   `--no-gates` is given.

Not found:

- Any builder for a stop from a walk other than nominal, a start to a walk other than nominal, a change to 0.40 m/s
  as a transition, or a turn reference at 0.40 m/s.
- Any record of the stop reference's deceleration at 0.40 m/s.
- Any option or parameter in `refs.py` or `transitions.py` that selects a different `GaitSpec`; both are name-driven
  throughout.
