# Map 07: Stage 0 screen, motion measures, clips, the look page, the envelope/clearance screens and refine cases

Read-only map of `hardware/trex_form_study` (all paths relative to it). Line numbers are those of the checkout on
2026-11-14. "not found" means searched for and absent. Nothing here was run beyond short inspections of JSON files and
signatures.

Files read in full: `designlab/stage2/screen.py`, `motion.py`, `clips.py`, `designlab/envelope.py`,
`designlab/refine.py`, `stage2/x/HEAD_X.md`, `X_COSTS.md`, `x_inputs.py`, `run_x.sh`, `tests/test_stage2_screen.py`,
`tests/test_stage2_motion.py`. Read in part: `designlab/stage2/look.py` (140-1625, everything the screen calls),
`look_page.py` (docstring, constants, loader, build, CLI), `designlab/clearance.py` (docstring, constants, swing,
`screen`, `_range_margins`, `_summary`), `designlab/round2.py` (screens), `designlab/stage1/{__init__,refs,plant,env,
gates,panels}.py` (the parts quoted).

## 0. State of this checkout when mapped

- `stage2/look/refs/` (the ladder's 73 npz files) is **absent** (git-ignored by the repository's `.gitignore:49`
  `*.npz`). `screen.load_arrays` would raise on every rung; the tests marked `needs_arrays` skip. `look_refs.json`,
  `tail_phase.json`, `screen.json`, `SCREEN.md`, `metrics.json`, `clips.json`, `RENDERS.md`, `look_page.json` are
  committed and present.
- `round2/refine/{narrow_knee_60,head_x_60,head_x_75}/models/` are **absent** (`round2/.gitignore:3` `models/`);
  `narrow_carpet_60/models` and `feet_75/models` are present. Without a case's model files `plant.model_files`
  rebuilds into `stage1/models/<body>/` with another ledger JSON, "so every record key would differ"
  (`stage2/x/x_inputs.py:13-14`). `stage2/x/<body>/{refs,transitions}/*.npz` are absent too (60 git-ignored inputs
  listed in `stage2/x/x_inputs.json`).
- Import direction: `designlab/stage2/*` imports `designlab/stage1` (`plant`, `refs`, `metrics`, `candidate_id`,
  `case_dir`); nothing in `designlab/stage1` imports `designlab/stage2`. So no Stage 1 record hash covers any Stage 2
  module (section 9).

## 1. `designlab/stage2/screen.py` (1405 lines): the Stage 0 screen of the look ladder

### 1.1 Constants

| Name | Line | Value |
|---|---|---|
| `OUT` | 110 | `look.OUT` = `stage2/look` |
| `SCREEN_JSON`, `SCREEN_MD` | 111-112 | `"screen.json"`, `"SCREEN.md"` |
| `GRIPS` | 114 | `(0.9, 0.54)` |
| `SWAY_SPEC_H`, `ROLL_SPEC_RMS_DEG`, `ACCEL_SPEC` | 117-119 | 0.12 H, 4.0 deg, 1.5 m/s^2 (reported, not gated) |
| `NEAR_BAND` | 122 | `turning.NEAR_FLAG_BAND` = 0.02 |
| `KINDS` | 123 | `("cont_ratio", "peak_ratio", "speed_ratio")` |
| `SPEED_LABEL` | 124-125 | keys `nominal`, `v040`, `fr05` only |
| `ORDER` | 127 | `("V0","V1B50","V1","V1B100","V2","V3","V3T","V4","V4L","V5","V5F13","V6","V6S")` |
| `TAIL_WALKS` | 129-130 | the tail check's 11 walk names |
| `SOURCES` | 132-184 | block name -> prose source; written to `screen.json["sources"]` |
| `LOOK_KEYS` | 459-462 | the 16 `look.measure` keys copied into each rung's `look` block |
| `FRICTION_FIELDS` | 559-560 | fields of `turning._foot_stats` compared with round two's record |
| `HEADLINES` | 684-716 | label -> path in a rung entry, for `vs_V0` |
| `G2_NAMES` | 775 | `("nominal", "brisk", "fr05")` |
| `GROUP_LABEL`, `KIND_LETTER`, `BODY_LABEL`, `SPEED_SHORT` | 901-905 | 8 groups; 2 bodies; 3 speeds |

### 1.2 Inputs and how rungs are enumerated

- `load_arrays(e: dict, out_dir: Path = OUT) -> dict` (194): opens `<out_dir>/<e["file"]>`, raises `ValueError` unless
  its sha256 equals `e["sha256"]`; returns `{key: ndarray}` (`np.load(..., allow_pickle=False)`).
- `_docs(out_dir) -> (look_refs.json, tail_phase.json)` (208), cached per resolved `out_dir` in `_DOCS`.
- Rungs come only from `look_refs.json["bodies"][body][speed][variant]` and only for variants in `ORDER`:

  ```python
  jobs += [("rung", b, s, v, str(out_dir)) for b in bodies for s in speeds for v in ORDER
           if v in ladder_doc["bodies"][b][s]]                                   # screen.py:825-826
  ```
  A rung present in `look_refs.json` under a name not in `ORDER` is silently not screened (and then
  `test_every_rung_is_screened_from_its_own_arrays` fails on the set comparison). `bodies` and `speeds` default to
  `look.BODIES = ("60-N-T-10", "75-M-H-10")` (look.py:145) and `look.SPEEDS = ("nominal", "v040", "fr05")` (look.py:146);
  `run()` takes them as arguments, the CLI does not expose them.
- The committed screen holds 73 rungs: 12 per body and speed (`V0,V1B50,V1,V1B100,V2,V3,V3T,V4,V4L,V5,V6,V6S`) plus
  `V5F13` at `75-M-H-10`/`fr05`.

### 1.3 `screen_rung`

```python
def screen_rung(src: dict, speed: str, variant: str, e: dict, out_dir: Path = OUT) -> dict     # screen.py:465
```

- `src`: `look.source(body)` (look.py:418) = `stage1.refs.source(body)` (refs.py:97) plus `geometry`
  (`gait.foot_geometry`) and `markers`. Keys: `body`, `M` (`statics.Model`), `q_home`, `design`, `H` (m), `layout`,
  `record` (the candidate record), `model` (the `plant.verify` check), `geometry`, `markers`.
  `look._source_cached(body)` (look.py:1453) caches it per process.
- `e`: one `look_refs.json` entry (section 2). Keys read: `file`, `sha256`, `frames`, `info` (`speed`, `period`,
  `duty`, `stride_rel`, `stride`, `sway_amp`, optional `strides`, `window_s`, `swing_time`), `price` (compared),
  `measured` (`zmp_margin_min_mm`, `foot_below_floor_max_mm`, `trunk_roll_rms_deg`, `LOOK_KEYS`), `features`, `flags`.
- Array keys read (directly or through `look.trajectory`, `look.price`, `straight_run`, `sway`, `accelerations`,
  `peak_speed_cause`): `t, dt, q, support, zmp_ref, com_ref, zmp_mb, ik_ok, ik_fail_frames, rolling, toe_edge, phase,
  com, hip_mid, stride_motion, feet, swing, swing_u, pad_floor_z, toe_floor_z`. A Stage 1 reference npz
  (`refs.arrays`, refs.py:145-166) lacks `rolling, toe_edge, phase, hip_mid, ik_ok, ik_fail_frames, pad_floor_z,
  toe_floor_z, swing_u`, so it cannot be passed as is (`look.trajectory` refuses arrays without `ik_ok`,
  look.py:669-670).

Blocks computed, in order (all through `look._plain`, so the result is JSON-plain):

| Block | Lines | How |
|---|---|---|
| re-price | 472-473 | `pr = look.price(src, arrs, info)` (look.py:891); `_max_abs_diff(pr, e["price"])` -> `reprice = {max_abs_diff, other_differences, equal_to_recorded}` |
| `drives` | 474-480 | `look.look_inverse_dynamics(M, tr, arrs["rolling"], arrs["toe_edge"])` under `look.accuracy()` ("r2") -> `sizing._motor_space(M.names, tau, omega, src["layout"])` -> `envelope._ratios(design, bus, ms)`; **raises `AssertionError`** if any leaf differs from `pr["drives"]` |
| `governing` | 390-392 | `[group, "cont"/"peak"/"speed", ratio]` of the largest ratio |
| `passive` | 481-486 | `passive_loads` (307); **raises `AssertionError`** unless `cont_ratio_total`/`peak_ratio_total` equal `look._passive_ratios` to rel 1e-12 |
| `summary` | 487-489 | `gait.summarize(M, tr, idr)`: `min_cop_margin_mm`, `min_cop_margin_single_mm`, `max_friction_ratio`, `ik_ok`, `ik_fail_frames` |
| `friction` | 490-491 | `straight_run(src, tr, rolling, arrs["phase"])` (261) |
| `contact` | 493 | `pr["contact"]` + `measured.zmp_margin_min_mm` |
| `clearance` | 494-495 | `pr["clearance"]` (`look._clearance_gates`, look.py:867) + `measured.foot_below_floor_max_mm`, without `joint_range_margins` |
| `plain_gait_id` | 496-503 | only if any foot rolls: `gait.inverse_dynamics(M, tr)` ratios (`worst`, `governing`, `over`, `max_ratio_diff`); else `None` |
| `over` | 504 | `pr["over"]` (look.py:926-942): each drive ratio > 1.0, `"IK"`, `"single-support CoP off the pad (envelope rule, -10 mm)"`, `"rolling-contact split did not solve"`, `"clearance <gate>"` for each of the six gates not `True` |
| `near` | 505-506 | drive ratios with `1 - 0.02 < r <= 1` |
| `over_with_passive`, `near_with_passive` | 507-512 | neck/tail `cont_ratio_total`/`peak_ratio_total` over 1 / within 0.02 below 1 |
| `rung` | 514-519 | the plan's rule as text; uses `SPEED_LABEL[speed]` when over (**KeyError for any other speed key**) |
| `cause` | 520-524 | `peak_speed_cause` (395) for each over-limit `"<group> speed"` with `group in sizing.JOINT_OF` (hip_roll, hip_pitch, hip_yaw, knee); `None` if none |
| `sway` | 532 | `sway(arrs, info, src["H"], m)` (339) |
| `accel` | 532 | `accelerations(arrs)` (363) |
| `look` | 535 | `{k: measured.get(k) for k in LOOK_KEYS}` |
| `vs_V0` | 753 (in `assemble`) | `changes(r, ents["V0"])`: `{label: [V0 value, rung value, difference]}` over `HEADLINES` |

Return keys (screen.py:525-537): `body, speed, variant, features, flags, arrays{file, sha256, frames, window_s,
strides}, gait{speed_m_s, period_s, duty, stride_rel, swing_time_s}, drives, governing, passive, summary, contact,
friction, sway, accel, clearance, plain_gait_id, over, near, over_with_passive, near_with_passive, verdict, rung,
within_limits, cause, look, reprice, seconds` (+ `vs_V0` after `assemble`).

Drive ratios (`envelope._ratios`, envelope.py:61-72), per group of `sizing.GROUPS` (+ `sizing.HEAD_X_GROUPS` on a model
that has them, envelope.py:55-58):

```python
out[g] = dict(actuator=a.id, rms=rms, peak=peak, speed=speed, cont_ratio=rms / a.sustained,
              peak_ratio=sizing.PEAK_MARGIN * peak / a.peak,
              speed_ratio=sizing.SPEED_MARGIN * speed / max(spd, 1e-6))
out[g]["worst"] = max(out[g]["cont_ratio"], out[g]["peak_ratio"], out[g]["speed_ratio"])
```
`rms`, `peak` in N m and `speed` in rad/s of the motor-space series (both sides stacked); `a.sustained` = A1 (0.57 x
rated), `a.peak` the planning peak, `spd = a.speed_at(bus if a.box is None else None)`; `PEAK_MARGIN = 1.25`
(sizing.py:55), `SPEED_MARGIN = 1.15` (sizing.py:56), `CONT_MARGIN = 1.0` (sizing.py:53). Groups: `GROUPS = ("hip_roll",
"hip_pitch", "hip_yaw", "knee", "ankle_a", "ankle_b", "neck", "tail")` (sizing.py:35).

Readings in detail:

- `straight_run(src, traj, rolling=None, phase=None) -> dict` (261-304): `turning.inverse_dynamics(M,
  look.turning_traj(traj, M, q0), turning.MU, split="exact", ds_friction=True)` (267); per foot `turning._foot_stats`
  over its single support. Raises `ValueError("a foot rolls in single support")` (278-279). Returns
  `single_support{frames, mz_abs_max_nm, mz_rms_nm, mz_peak_stride_frac, mz_peak_stance, mu_req_max,
  mu_req_max_stride_frac, mu_req_max_stance, mu_req_cop_max, util_max, fn_min_frac_weight,
  mu_req_peak_frame_is_mz_peak_frame}`, `feet{left,right: _foot_stats (14 keys: frames, fn_min, ft_over_fn_max,
  mz_abs_max, util_max, util_cop_max, mu_req_max, mu_req_cop_max, mu_req_centroid_max, r_cop_min, frames_cop_outside,
  cop_margin_min, util_max_frame, util_max_support)}`, `double_support{frames, read, rolling_frames_not_read,
  min_mu_max, min_mu_lower_max, infeasible_frames}` (only the double-support frames where no foot rolls are read),
  `grips{"0.9": mu_req_max <= 0.9, "0.54": mu_req_max <= 0.54}`, `mu_assumed` (0.8).
- `passive_loads(M, design, tau, omega) -> dict` (307-336): hard-coded `(("neck", "neck_pitch"), ("tail",
  "tail_yaw"))` (312). Per group 20 keys: `joint, actuator, damping, frictionloss, sustained_nm, planning_peak_nm,
  rms_id_nm, rms_passive_alone_nm, rms_total_nm, peak_id_nm, peak_total_nm, cont_ratio_id, cont_ratio_total,
  peak_ratio_id, peak_ratio_total, rms_total_over_id, speed_rms_rad_s, moving_frames_frac, power_mean_id_w,
  power_mean_total_w`. `total = tau + b omega + f sign(omega)` with the MJCF's `dof_damping` and `dof_frictionloss`
  (0.2 N m s/rad and 0.15 N m on both bodies).
- `sway(arrs, info, H, measured=None) -> dict` (339-352): `stage0_pp_H = 2 * info["sway_amp"] / H`; `com_pp_H`;
  `com_pp_per_stride_mean_H`, `com_pp_per_stride_max_H` (strides indexed by `floor(phase)`); `hip_mid_pp_H`; `spec_H`;
  `strides`; with `measured`: `trunk_roll_rms_deg`, `trunk_roll_spec_rms_deg`.
- `accelerations(arrs) -> dict` (363-383), m/s^2, periodic second differences with the window's x advance
  (`stride_motion[3]`) unwrapped: `com_horizontal_peak, com_forward_peak, com_lateral_peak, com_forward_lp1hz_peak`
  (`stage1.metrics.lowpass(x, fs, hz=ACCEL_LP_HZ, order=ACCEL_LP_ORDER)`, metrics.py:351, on the window tiled to at
  least 20 s each side), `com_vertical_min, com_vertical_max, foot_vertical_peak, foot_vertical_up_peak,
  spec_start_stop` (1.5).
- `peak_speed_cause(src, arrs, info, omega, joint) -> dict` (395-425): `joint, peak_rad_s, side, t_s, stride_frac,
  state ("swing" | "rolling stance" | "flat stance"), swing_frac, swings, swing_s_range, peak_rad_s_range,
  fastest_swing_s, swing_vs_peak_corr`; swings from `look.contact_events(arrs)`.
- `limits(src) -> dict` (216-225): per group of `sizing.GROUPS` only (220): `actuator, sustained_nm, peak_nm,
  speed_rad_s, bus_v`.
- `plant_joint_friction(body) -> dict` (239-249): hard-coded `("neck_pitch", "tail_yaw")` (246).
- `v040_between(H) -> dict` (228-236): `weight_w`, `brisk{speed_m_s, period_s, duty, stride_rel}`, `fr05{...}`.

Clearance gates inside the price (`look._clearance_gates`, look.py:867-888): `clearance.screen(str(xml), src["q_home"],
{"nominal": dict(q=traj["q"], phases=traj["phases"], ik_ok=traj["ik_ok"])}, H=src["H"], gates="r2")` on the plant's
MJCF (`plant.model_files`, checked by `plant.verify`); the rung "stands in for the nominal walk". Gates kept:
`swing_clearance, joint_range_margin, interference_free_home, interference_free_nominal_gait,
component_overlap_free, tail_clearance` (look.py:881-882; `interference_free_brisk_gait` and `battery_path` are not
read). Block keys: `gates, walking_gates, mid_swing_min_frac_H, mid_swing_min_mm, mid_swing_threshold_frac_H (0.04),
lateral_gap_mm, tightest_joint [name, rad], joint_range_margin_rule_rad (0.10), joint_range_margins (dropped by the
screen), nominal_gait_interference_pairs, tail_min_frac_H, screen`.

### 1.4 The tail-phase check and G2

- `tail_phase(src, speed, out_dir=OUT) -> dict` (610-677): needs rungs `V0`, `V4`, `V5` in `look_refs.json` (620-622,
  625, 632, 660) and `tail_phase.json["bodies"][body][speed]["tail_phase_F9_rad"]`; generates six walks by
  `look.generate` (`V0_no_tail`, `V0_F9_phase`, `V0_F9`, `V0_look_path`, `V4_no_tail`, `V4_F9`) and, at
  `speed == "nominal"`, the brisk walk for the record check against
  `<case>/review/screening/<id>_turning.json` `runs.straight` / `runs.straight_brisk`. Returns `body, speed,
  f9_phase_rad, f9_amp_rad, today_phase_rad, today_amp_rad, f9_phase_cycles_from_today, f9_f13_phase_rad, walks{name:
  straight_run + tail{amp_rad, phase_rad}, source, spec_differs_from[, look_path]}, look_path_control, changes{...},
  today_adds_to_leg_yaw_moment, f9_phase_lowers_peak_yaw_moment, f9_phase_lowers_friction_need[, record_check],
  seconds`.
- `g2(body) -> dict` (778-791): `look.g2_hashes(body, G2_NAMES)` (look.py:1578: `refs.generate` at r2 + `refs.write`
  in a temp dir) against `stage1/runs/2026-11-08/<body>/refs.json["references"][name]["sha256"]`. Keys: `record,
  accuracy, references{name: sha256, recorded, bit_identical, record_leaves_match}, passes, blas_threads, seconds`.

### 1.5 `run`, partial files, `screen.json`

```python
def run(out_dir: Path = OUT, procs: int = 1, resume: bool = False, bodies=look.BODIES, speeds=look.SPEEDS) -> dict   # 811
```

- Jobs: `("g2", body, "-", None)`, `("tail", body, speed, None)`, `("rung", body, speed, variant)`; sorted long first;
  `multiprocessing` `fork` pool of `procs` (854).
- Each finished job is kept in `<out>/partial/screen/<kind>__<body>__<speed>__<variant or ->.json` as
  `{"inputs": [sha256(look_refs.json), sha256(tail_phase.json)], "result": ...}` (807-808, 849). `--resume` reuses a
  partial only if `r.get("inputs") == [refs_sha, tail_sha]` (836). All partials are deleted at the end (890-894).
- Writes `<out>/screen.json` (through `.tmp`, `indent=1`) and `<out>/SCREEN.md` (`markdown(doc)`).
- `screen.json` top-level keys: `what, note, method, inputs{look_refs_json_sha256, tail_phase_json_sha256,
  arrays_checked}, speeds, v040_rule, v040{body: speed_m_s, period_s, duty, stride_rel, lift_frac, duty_with_F3,
  weight_w, brisk{}, fr05{}}, limits{body: {group: ...}}, plant_joint_friction, margins{peak, speed, cont, near_band},
  grips, specs{sway_H, trunk_roll_rms_deg, start_stop_accel_ms2}, limit_rule, g2{body: ...}, provenance
  (`plant.provenance()`), versions, procs, sources, tail_phase{body: {speed: ...}}, rungs{body: {speed: {variant:
  screen_rung}}}, over_limit[], near_limit[], over_with_passive[], near_with_passive[], seconds`.
- Keyed by `rungs[body][speed][variant]`; no key is a hash. What is hashed: the two input JSON files, each rung's npz
  (`arrays.sha256`, checked before use), the G2 references. The screen's own code is not hashed; `provenance` records
  the git commit, `dirty`, `dirty_paths` and `stage1_sources.sha256` (all 22 `designlab/stage1/*.py`,
  `plant.sources_sha256`, plant.py:267-275). The committed screen: commit `f4bc82b7...`, clean, 556 s on 3 processes;
  its two over-limit rungs are `75-M-H-10 fr05 V6` (knee speed 1.012) and `V6S` (1.009).
- Hard-coded rung names inside `run`/`assemble`/`markdown` (a directory lacking them fails): `V0` in every body and
  speed (`assemble`, 753); `rungs[b]["v040"]["V0"]` and `["V2"]` and `ladder_doc[...]["v040"]["V0"]["spec"]` (873-876);
  `_brief` reads `V0, V1, V2, V3, V4, V5, V6` at every speed (1046-1052, 1058-1067, 1077-1078, 1098-1099); section 6 of
  `markdown` reads `e["V6"]["vs_V0"]` (1344).

### 1.6 CLI

```
python -m designlab.stage2.screen run [--out stage2/look] [--procs 1] [--resume]     # screen.json and SCREEN.md
python -m designlab.stage2.screen md  [--out stage2/look]                             # SCREEN.md again from screen.json
```
(`main`, 1386-1401). When run as `__main__` it sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` to 1 by
default (90-92). `md` needs no arrays.

### 1.7 Screening an arbitrary new reference

(a) A periodic straight walk from any `gait.GaitSpec` (an F25+F9 walk, a look spec):

```python
from designlab.stage2 import look, screen
src  = look._source_cached(body)                 # any body of stage1.BODIES whose model has the markers' geoms
g    = look.generate(src, spec)                  # look.py:470: gait.generate at "r2" + gait.derivatives; a default-field
                                                 # spec gets its look block from _complete_v0 (485)
e    = look.write(src, speed, variant, spec, g, out_dir)   # look.py:1349: <out_dir>/refs/<body>/<speed>/<variant>.npz,
                                                           # entry with sha256, spec, info, measured, price, flags
r    = screen.screen_rung(src, speed, variant, e, out_dir)
r["vs_V0"] = screen.changes(r, v0_entry)         # only assemble() adds it
```
Conditions: `speed` must be a `SPEED_LABEL` key if the rung is over a limit (516); `look.features_of(variant, ...)`
returns `[]` for an unknown variant (look.py:405-406); no foot may roll in single support (278); F9's phase needs
`look.tail_check(src, spec, amp)` first (look.py:1226; six generated walks). Passing an `out_dir` other than
`stage2/look` leaves the committed ladder and screen untouched; `look_refs.json` is written only by `look._merge`
(look.py:1459, reached through `look.prepare`), `look.remeasure` (1532) and `look.reflag` (1551), never by
`look.write` or `screen_rung`. `look.write` needs `look.generate`'s output (`g["look"]["rolls"]`, `info["window_s"]`,
`info["zmp_error_history_m"]`, `info["zmp_margin_min_m"]`). A V6-size window took about 300 s to generate and 17 s to
price in the committed run (`seconds`, `price_seconds` of `60-N-T-10 v040 V6`); its screen took 28 s.

(b) A Stage 1-style reference (a walk or turn of `refs.generate(src, name)`): the pattern is
`stage2/x/x_costs.py:355 walk_screen`: `sizing._motor_space(M.names, tau, omega, src["layout"])` ->
`envelope._ratios(src["design"], bus, ms)`, and `clearance.screen(xml, q_home, {"nominal": dict(q, phases, ik_ok)},
H=..., gates="r2")`. For a turn, `refs.generate` (refs.py:126-131) uses `turning.generate`,
`turning.inverse_dynamics(M, tr, turning.MU, split="exact", ds_friction=True)`, `turning.summarize`,
`turning.run_groups(M.names, idr, layout)`; the thigh-twist range reading is `turning._range_margins(cl_summary, M,
trajs, joints=("hip_yaw", "hip_roll"))` (turning.py:1020) on a `clearance.screen` summary, and
`turning.clearance_screen` (turning.py:1059). `screen_rung` cannot take a turn: `look.turning_traj` (look.py:679-689)
writes `transform=(0.0, origin, origin + [adv, 0])`, `heading = 0` and `yaw_rate = 0.0`;
`look.look_inverse_dynamics` differentiates with `gait.derivatives` (look.py:796-799), whose unwrap is "a translation
along x: straight walks only" (gait.py:337-338); `sway` and `accelerations` unwrap along x.

(c) A non-periodic window (start, stop, speed change): nothing in `screen.py`, `motion.py`, `clips.py` or
`look.measure` handles one. Every derivative and event wraps the window: `np.roll` (screen.py:356, motion.py:177),
`% n` (gait.py:309, 332; motion.py:199), `% W` (look.py:989-1010, 1040-1180; screen.py:413-414), tiling (screen.py:375,
motion.py:157, 329). Window-agnostic pieces that do apply to any `(tau, omega)` series: `sizing._motor_space`,
`envelope._ratios`, `passive_loads`. `clearance.screen` takes any `q` and `phases`, but reads swings as cyclic runs
(`_swing_runs`, clearance.py:739-758) and pairs the last frame with the first for `max_step_mm` and substeps
(clearance.py:1013).

## 2. `look_refs.json` entries and the rung arrays (what the screen, motion and clips consume)

`stage2/look/look_refs.json` top level: `bodies, body_info, checks, accuracy, values, variants, extras, speeds,
provenance, versions, generator, note, limit_rule, deferred, price_rule, seconds, reflag`.

Entry `bodies[body][speed][variant]` (`look.write`, look.py:1363-1368): `body, speed, variant, features, file
("refs/<body>/<speed>/<variant>.npz"), sha256 (of the npz), frames, spec (the 38 `GaitSpec` fields), info (24 keys:
speed, stride, period, froude, duty, stride_rel, cadence_hz, sway_amp, lift, omega, zmp_error_history_m, accuracy,
strides, window_s, swing_time, bob_m, bob_trough_step, bob_drop_step, step_deviations, gaze_freqs_hz, tail_freqs_hz,
rolling_entry, zmp_margin_min_m, zmp_margin_ref_min_m), ik_ok, ik_fail_frames (count), zmp_error_final_mm,
zmp_margin_min_mm, rolls{left,right: [...]}, measured (49 keys, `look.measure`), price (11 keys: dynamics, drives_at,
drives, worst_group, worst_ratio, passive, contact, clearance, over, within_limits, verdict), flags, seconds,
price_seconds`.

`body_info[body]`: `H_m, foot_frame, feet, markers{name: {body, point}}, toe_clearance, claw_question`.
`tail_phase.json["bodies"][body][speed]`: `tail_phase_F9_rad[, tail_phase_F9_F13_rad], V0, V4[, V4F13]` (each a
`look.tail_check` result).

Arrays (`look.arrays`, look.py:581-648; n frames, 2 feet ordered left, right; shapes read from the code, the files
being absent):

| Key | Shape | Unit / meaning |
|---|---|---|
| `t` | (n,) | s |
| `dt` | () | s (0.00469 s on 60 cm at 0.40 m/s: 160 frames per stride) |
| `q`, `qd`, `qdd` | (n, nq), (n, nv), (n, nv) | MuJoCo qpos (free root 7 + hinges), qvel, qacc |
| `support` | (n, 2) bool | planned support flags |
| `com_ref`, `zmp_ref`, `zmp_mb` | as `gait.generate` returns them | m |
| `feet` | (n, 2, 3) | foot body origins, m |
| `com` | (n, 3) | model CoM, m |
| `q_home` | (nq,) | |
| `joints` | (hinges,) str | `M.names` |
| `stride_motion` | (5,) | `(phi, a_x, a_y, b_x, b_y)` (refs.py:137-143); a look window: `[0, 0, 0, K * stride, 0]` |
| `phase` | (n,) | strides (0 at the left foot's lift-off; warped by F14) |
| `swing` | (n, 2) bool; `swing_u` (n, 2) | swing flag; swing fraction, NaN in stance |
| `foot_pitch`, `foot_pitch_solved` | (n, 2) | target and solved heel-up pitch, rad (`atan2(-R[2,0], R[0,0])`) |
| `rolling`, `on_claw` | (n, 2) bool | |
| `toe_edge` | (n, 2, 2, 3) | the front contact's two end points, m; `toe_pivot` (n, 2, 3) their midpoint |
| `foot_quat` | (n, 2, 4) | |
| `ik_ok` | () bool; `ik_fail_frames` (k,) int64 | |
| `foot_target` | from the look block | |
| `support_hull` | (n, hmax, 2) | NaN-padded; `zmp_margin`, `zmp_ref_margin` (n,) m |
| `hip_mid` | (n, 3) | midpoint of the two hip-roll axes' anchors, m |
| `bob`, `trunk_roll` | (n,) | m, rad; `trunk_quat` (n, 4) |
| `neck`, `tail` | (n,) | joint angles, rad; `head_settle`, `gaze_var`, `tail_var` (n,) |
| `head_pitch_world` | (n,) | rad (`gait._nose_up` of the head's quaternion) |
| `pad_floor_z`, `toe_floor_z` | (n, 2) | lowest contact-pad point; lowest toe/claw point (NaN if none), m |
| `points` | (n, 13, 3) | markers `look.POINTS` (look.py:277); `point_names` (13,) |
| `lift_off_t`, `touchdown_t` | (2, K) | planned event times, s (`look.events`, look.py:651) |

## 3. `designlab/stage2/motion.py` (836 lines): the look measures

All readings are on a rung's arrays (a periodic kinematic window). The docstring says they are "defined on the
frames ... so that the same code can read the closed-loop traces later" (motion.py:4-7); an adapter from a Stage 1
trace or `metrics.Recorder` to these arrays: **not found** (no Stage 1 module imports Stage 2; no function in
`motion.py` or `look.py` takes a trace).

Constants: `BANDS = dict(bob_rel=0.30, head_world_pitch_rms_deg=1.5, tail_phase_cycles=0.1, toe_off_rel=0.70)`
(motion.py:94; "Research 4.2's look-fidelity bands (for the closed loop; reported beside the reference's readings)").
The bands are only written out (`metrics.json["bands_closed_loop"]`, `head.band_world_pitch_rms_deg`,
`tail.band_cycles`); code that judges a run against `bob_rel` or `toe_off_rel`: **not found**. (Stage 1's
`panels.LOOK_BANDS = {"head": (0.05, 0.15), "tail": (0.10, 0.20)}`, panels.py:142, are other bands.)

```python
def metrics(body: str, speed: str, variant: str, entry: dict, arrs: dict) -> dict        # motion.py:358
```
`H = info["stride"] / entry["spec"]["stride_rel"]`; calls `look.measure(arrs, dict(info, look=entry["spec"]), H,
crouch=entry["spec"]["crouch"])` (366). Returns `body, speed, variant, features, rung_sha256, H_m, window{frames,
dt_s, strides, seconds}, bob, trunk, sway, head, tail, toe_off, gait, bird_curves, step_times, smoothness,
mechanical_tells, extremes`.

The four look measures the bar's bands name:

| Measure | Value | Computed from | Where |
|---|---|---|---|
| Hip bob | `bob.pp_mm`, `bob.pp_H` (peak to peak of `hip_mid[:, 2]` over the window), `per_step_pp_mm` (mean peak to peak per step, steps indexed by `floor(2 * phase)`) | `hip_mid`, `phase` | motion.py:368-375; `look.measure` `bob_pp_H` look.py:1032-1033 |
| Head pitch in the world, RMS | `head.world_pitch_rms_deg = degrees(sqrt(mean((hp - mean(hp))**2)))` | `head_pitch_world` | motion.py:387-390; look.py:1169-1170 |
| Tail phase | least squares `tail = A sin(2 pi phase + psi) + c` (`look._fit_phase`, look.py:961-965); `tail.phase_rad = psi`, `tail.amp_rad = A`, `phase_from_today_cycles = ((psi - pi/2) / (2 pi) + 0.5) % 1 - 0.5` | `tail` (the tail-yaw angle), `phase` | motion.py:403-407; look.py:1144-1146 |
| Toe-off | `toe_off.at_lift_off_rad` = mean over the planned lift-offs of `foot_pitch_solved` interpolated at `lift_off_t`; `at_lift_off_range_rad`, `peak_rad`, `reach_rad`, `ground_roll_s` | `foot_pitch_solved`, `lift_off_t`, `touchdown_t`, `rolling` | motion.py:409-411; look.py:1080-1106 |

Other readings: `trunk{roll_rms_deg, pitch_rms_deg, yaw_rms_deg, roll_pp_deg, pitch_pp_deg, roll_toward_stance}`
(`relative_euler`, 208); `head{angular_rate_rms_deg_s, trunk_angular_rate_rms_deg_s, stabilisation_ratio,
pitch_rate_rms_deg_s, trunk_pitch_rate_rms_deg_s, pitch_stabilisation_ratio, nod_amp_rad, settle_lag_s}` (body
attitudes from the plant's MJCF by `body_quats(model(body), arrs["q"])`, 223-242, so `q` must have that model's `nq`);
`sway` (`screen.sway`); `gait{duty, duty_planned, swing_time_s, swing_time_spread_s, cadence_steps_s, stride_H,
lift_peak_frac_swing, speed_m_s, froude, period_s}` (duty, swing time and cadence from `look.contact_events`: a foot is
on the floor where `min(pad_floor_z, toe_floor_z) <= CONTACT_TOL = 3e-4` m, look.py:970, 983-1011); `step_times{n,
mean_s, cv, max_dev, left_to_right_over_right_to_left, alternate}` (245); `smoothness{foot{left, right, mean, lowest,
fixed_band_mean}, head{sparc, cutoff_hz, repeats, fixed_band}}` (`sparc` 128, `sparc_periodic` 147, `foot_sparc`
264); `mechanical_tells{constant_hip_height (< 1 mm), flat_feet_at_lift_off (< 0.01 rad), identical_steps (cv < 1e-4),
metronome_nod (> 0.01 rad), synchronous_extremes}`; `extremes` (311; parts `PARTS`, 102-109); from `screen.json`:
`stance_pad_yaw{peak_nm, rms_nm, mu_req_max, source}`, `stage0{within_limits, over, verdict}` (`stage0_screen`, 586).

`look.measure(arrs, info, H, crouch=gait.GaitSpec.crouch) -> dict` (look.py:1014) is all or nothing: it reads `t, dt,
phase, support, lift_off_t, touchdown_t, hip_mid, pad_floor_z, toe_floor_z, foot_pitch_solved, rolling, toe_pivot,
on_claw, q, q_home, com, stride_motion, tail, points, neck, head_settle, head_pitch_world, feet, zmp_mb,
support_hull` and `info["period"], ["duty"], ["stride_rel"], ["stride"]`, `info["look"]["head_settle"]`, with periodic
arithmetic throughout.

`run(out_dir=OUT, media=None, bodies=look.BODIES) -> dict` (614): iterates **every** entry of `look_refs.json` for the
bodies (no `ORDER` filter); if `screen.json` exists it requires `sdoc["rungs"][b][s][v]` for each (629, `KeyError`
otherwise); writes `<out>/metrics.json` (`what, method, inputs, v040, bands_closed_loop, sparc, parts, bodies{body:
{speed: {variant: compact(metrics)}}}, plots, provenance, versions, seconds`) and, with `media`, the PNG plots of
`plot_set` (596; hard-coded `V0`..`V6` and `v040`). `markdown(mdoc, cdoc)` (768) -> `RENDERS.md`; its columns are
`look.VARIANTS + ("V5F13",)` (801); `notes` (722) reads `V2, V3, V5, V6` at every speed.

CLI (`main`, 818-832):
```
python -m designlab.stage2.motion run [--media DIR] [--out stage2/look] [--body B ...]   # B in look.BODIES
python -m designlab.stage2.motion md  [--out stage2/look]                                # RENDERS.md
```

## 4. `designlab/stage2/clips.py` (694 lines): render entry points

- `render_clip(body, speed, variant, view, mode, media: Path, out_dir: Path = OUT, clip_s: float = CLIP_S, fps: int =
  FPS, max_bytes: int = MAX_BYTES) -> dict` (543): one MP4 to `<media>/<body>/<name>.mp4` with a poster PNG and a
  contact sheet; `name = f"{body}_{speed}_{variant}_{view}" + ("_pointlight" if mode == "pointlight" else "")` (529).
  Manifest entry keys: `file, poster, sheet, body, speed, variant, view, mode, features, rung_file, rung_sha256, bytes,
  sha256, encode, probe, frames, fps, seconds, size, labels, label_lines, stage0_over, camera, track, window{frames,
  dt_s, window_s, advance_m, repeats}, model{file, sha256}, render_s, encode_s`.
- `run(media: Path, jobs=None, procs: int = 1, out_dir: Path = OUT, record: Path | None = None) -> dict` (634): renders
  `jobs` (default `review_set()`), writes `<media>/clips.json` (merging entries of clips this run did not make) and,
  with `record`, the same manifest to `stage2/look/clips.json`.
- `review_set(bodies=look.BODIES) -> list[(body, speed, variant, view, mode)]` (593): 42 clips (`REVIEW`, 105-106).
- `Playback(arrs)` (164): `q_at(tau)`, `com_at(tau)`; refuses a window whose `stride_motion` is not a pure x advance:
  `raise ValueError(f"stride_motion {sm.tolist()}: not a straight window (a pure advance along x)")` (174-175); the
  window repeats with its advance.
- `rung(body, speed, variant, out_dir) -> (entry, arrays)` (125), `stage0_over` (132), `labels` (432; `SPEED_TEXT[speed]`
  at 436, three keys only), `ClipScene` (306; `plant.model_files` + `plant.verify`, `render.Scene`), `frame_margin`
  (277), `encode` (485; libx264 CRF 23, capped 1100 kbit/s, two-pass fallback, limit 1.5 MB), `probe` (469).
- Constants: `FPS = 30`, `CLIP_S = 10.0`, `SIZE = (854, 480)`, `LABEL = "kinematic reference, not a simulation"`,
  `CAMERAS` (`visitor`, `side`), `MODES = ("covered", "pointlight")`.

CLI (`main`, 664-690; as `__main__` it defaults `LP_NUM_THREADS=1` and `MUJOCO_GL=osmesa`, 52-54):
```
python -m designlab.stage2.clips render --media DIR [--body B ...] [--procs N] [--out stage2/look] [--no-record]
python -m designlab.stage2.clips one --media DIR --body B --speed S --variant V --view visitor|side
                                     [--mode covered|pointlight] [--out stage2/look]
```
`--body` is limited to `look.BODIES`, `--speed` to `look.SPEEDS`; `--variant` is free text.

## 5. `designlab/stage2/look_page.py` (2469 lines, skimmed)

- `load(look_dir=OUT) -> Docs(metrics, screen, clips, refs)` (153-177): reads `metrics.json`, `screen.json`,
  `clips.json`, `look_refs.json`.
- `build(media: Path, out: Path, look_dir: Path = OUT, commit=None, viewers_url=None) -> dict` (2389): stages the media
  (each clip checked against the sha256 in `clips.json`), writes `<out>/index.html` and `<out>/viewers/index.html`;
  returns the manifest (`inputs` = sha256 of the four JSON files; every staged file's `sha256`). Limit `MAX_PAGE_MB =
  60.0`.
- CLI (2443): `python -m designlab.stage2.look_page --media DIR --out DIR [--look stage2/look] [--manifest
  stage2/look/look_page.json] [--viewers-url URL]`.
- Hard-coded: `BODIES = look.BODIES`, `LADDER = look.VARIANTS`, `LADDER_SPEED = "v040"`, `BODY_NAME`/`BODY_SHORT`/
  `BODY_ID` for the two bodies, `GROUPS` (8), `BLIND` (V0 and V6 at 0.40 m/s), `RUNG_TITLE`/`RUNG_TAG`/`RUNG_ADDS` for
  `V0`..`V6`, `DATE = "11 Nov 2026"` (61-125). The page states F25 is not used (look_page.py:1083, 1892).

## 6. `designlab/envelope.py` (262 lines): ratios and the one-step push bound

Constants: `G = 9.81`, `FAST_FROUDE = 0.05`, `FAST = gait.GaitSpec(froude=0.05, stride_rel=0.80, duty=0.56,
lift_frac=0.11)` (46), `STOP_DECEL = 1.5`, `SWING_OVERHEAD_S = 0.08`, `SESSION = dict(standing=0.25, nominal=0.50,
brisk=0.10, turning=0.15)`.

- `_ratios(design, bus, ms: dict) -> dict` (61): section 1.3.
- `fast_walk(M, q_home, design, bus, layout) -> dict` (75): `turning.generate(M, q_home, H, FAST, yaw_rate=0.0)`;
  `reference_cop_on_pad = cop_min > -0.010` m; `drives_ok` is `None` when the CoP is off the pad.
- `stopping(M, q_home) -> dict` (106): `max_one_foot_decel_ms2 = G * front / z`.
- `push_recovery(M, q_home, design, bus) -> dict` (147-169), the Stage 0 forward push bound:

  ```python
  omega = np.sqrt(G / z)                                  # z: CoM height at home
  front, _ = _pad_reach(M, q_home, "right")               # pad's reach ahead of its centroid
  l_max, q_step = _max_step(M, q_home, H, "left")         # furthest flat-footed forward placement by IK (bisection to 1.2 H)
  dt = max(2 * |dq| / drive speed at the bus  for hip_pitch, knee) + SWING_OVERHEAD_S
  d1 = front + l_max * np.exp(-omega * dt)
  v1 = omega * d1                                         # max_push_dv_ms
  ```
  Keys: `com_height_m, omega, pad_front_reach_m, max_step_m, max_step_over_H, swing{hip_pitch, knee: excursion_rad,
  drive_speed_rad_s, time_s}, swing_time_s, capture_offset_m, max_push_dv_ms, max_push_impulse_ns, max_push_froude`.
  Forward only; no lateral or backward bound exists (**not found**).
- `screen(M, q_home, design, rec, trajs=None, turning=None) -> dict` (198): `module, fast_walk, stopping,
  push_recovery, session_heat, summary{..., push_max_dv_ms, push_max_impulse_ns, push_max_step_over_H,
  push_swing_time_s, ...}`; `screen_record(cid, rec, M, q_home, design, trajs, scr, turning=None)` (232) writes
  `<scr>/<cid>_envelope.json` and returns the summary.
- CLI (239): `python -m designlab.envelope <round2 dir> <ids...>` (prints summaries; sets `gait.set_accuracy("r2")`).

Where the bound is recorded and read:

- Recorded: `<case>/review/screening/<candidate id>_envelope.json` -> `push_recovery.max_push_dv_ms` (also
  `summary.push_max_dv_ms`, and `<case>/review/review_index.json[<id>]["envelope"]["push_max_dv_ms"]`).
- Read by Stage 1: `env.PUSH_BOUND_M_S = _PushBounds({"60-N-T-10": 0.487, "60-N-H-10": 0.478, "75-M-H-10": 0.589})`
  (env.py:272); an unlisted body's `[body]` reads its own envelope screen "to 3 decimals" (`__missing__`,
  env.py:264-268); `.get` "reads only the list".
- Panels: `PUSH_S = 0.1`, `PUSH_FRAC = 0.5` (panels.py:96-97); `F = self.mass * PUSH_FRAC * self.push_bound_m_s /
  PUSH_S` (panels.py:947); the bound enters every record key by value (`env_values`, panels.py:1603).
- Env: `dv = rng.uniform(*cfg.push_frac) * PUSH_BOUND_M_S.get(cfg.body, 0.5)` (env.py:465): a body outside the listed
  three gets 0.5 m/s there, not its screen's value.

Values in the committed screens (m/s):

| Body | Screen file | `max_push_dv_ms` | to 3 decimals | 0.25 x |
|---|---|---|---|---|
| `60-N-T-10` | `round2/refine/narrow_carpet_60/review/screening/60-N-T-10_envelope.json` | 0.486785 | 0.487 | 0.12175 |
| `60-N-H-10` | `.../narrow_carpet_60/.../60-N-H-10_envelope.json` | 0.478230 | 0.478 | 0.1195 |
| `75-M-H-10` | `round2/refine/feet_75/review/screening/75-M-H-10_envelope.json` | 0.588520 | 0.589 | 0.14725 |
| `60-N-T-10-K` | `round2/refine/narrow_knee_60/.../60-N-T-10_envelope.json` | 0.584249 | 0.584 | 0.146 |
| `60-N-H-10-K` | `.../narrow_knee_60/.../60-N-H-10_envelope.json` | 0.573707 | 0.574 | 0.1435 |
| `60-N-T-10-X` | `round2/refine/head_x_60/.../60-N-T-10_envelope.json` | 0.476293 | 0.476 | 0.119 |
| `75-M-H-10-X` | `round2/refine/head_x_75/.../75-M-H-10_envelope.json` | 0.579363 | 0.579 | 0.14475 |

## 7. `designlab/clearance.py` (1604 lines): the walking gates, gate set "r2"

```python
def screen(xml_path, q_home, gait_trajs: dict[str, dict], out_png: str | None = None, *, H: float | None = None,
           frame_stride: int = 1, substeps: int = 0, gates: str = "r1") -> dict            # clearance.py:942
```
`gait_trajs`: `{name: {"q": (n, nq), "phases": [support tuples], optional "ik_ok"}}`; the entry named `"nominal"`
carries the walk's interference gate, `"brisk"` the brisk one.

Constants: `TIGHT = 0.003` m, `REPORT_CAP = 0.025` m, `SWING_MIN_FRAC = 0.04` (136), `TAIL_MIN_FRAC = 0.25` (137),
`RANGE_MARGIN_RAD = 0.10` (140), `REST_ON_STOP = {"jaw": "min"}` (145), `SEAM_JOINTS = ("neck_pitch", "tail_yaw",
"neck_yaw", "jaw")` (150), `MID_SWING = (0.25, 0.75)` (164), `SWEEP_STEP = 0.05` rad, `LEG_JOINTS`, `CORNERS`.

`summary["pass"]` (`_summary`, 1431-1460):

| Gate | Rule |
|---|---|
| `interference_free_home` | no interfering pair at home |
| `interference_free_nominal_gait` | none along `gait_nominal` (`None` without a `"nominal"` entry) |
| `swing_clearance` | `min(mid_swing_min_frac_H over the gaits) >= 0.04` |
| `tail_clearance` | lowest tail point `>= 0.25 H` (home, tail-yaw sweep, every gait frame) |
| `battery_path` | at least one clear direction |
| r2: `interference_free_brisk_gait` | none along `gait_brisk` (`None` without a `"brisk"` entry) |
| r2: `component_overlap_free` | no same-body component overlap at home |
| r2: `joint_range_margin` | `min(gated_margins) >= 0.10` rad (`gated_margins`, 1393: all ends but those in `REST_ON_STOP`) |
| `all` | every gate `True` (a `None` `battery_path` excepted) |

`round2.WALK_GATES = ("interference_free_home", "interference_free_nominal_gait", "interference_free_brisk_gait",
"component_overlap_free", "joint_range_margin")` (round2.py:60-61).

The swing gate's reading (`_swing_report`, 761-790): a foot's swing frames are those "whose support tuple does not
contain it" (`_swing_runs`, 739-758, cyclic runs on the **planned** support flags); pad height = the lowest point of
that foot's contact pads only (`geom_contype == 1` on `<side>_foot`, `_foot_geoms` 729-736; toes and claws are not
pads); "middle half" = frames with `0.25 <= j / L <= 0.75` by frame index within the run (776); the value compared with
0.04 is `round(mid / H, 4)` (781, 786).

Joint-range margins (`_range_margins`, 1401-1428): per swept joint, the walks' excursion against the usable range
(first new interference with the trunk or the same leg in the single-joint sweep from home, else the MJCF limit; a
limit set by the other leg falls back to the joint limit). Sweeps (`_sweep_list`, 703-726): every leg joint, the
`hip_pitch x knee` and `hip_roll x hip_yaw` corners, `neck_pitch`, `tail_yaw`, and `neck_yaw`, `jaw` on a model that
has them. On the two picks the thigh twist is `left/right_hip_yaw` usable `[-0.3, 0.3]` rad, with no interference in
its sweep; round two's target turns (`walk_0.35`, `in_place_0.50`, at the nominal walk) reach 0.682 (60 cm) / 0.773
(75 cm) of that limit (`review_index.json[...]["turning"]["hip_yaw_range_used_max"]`: "largest fraction of the joint
limit reached (1.0 = at the limit)", turning.py:911, 1624; the stretch case `walk_0.50` is not in that figure).

`seam_sweep(xml_path, joints=SEAM_JOINTS, q_home=None, step=SWEEP_STEP, ...)` (1326): the parent/child seams the
body-pair screen leaves out.

## 8. Refine cases, the `-X` bodies and what a new body variant needs

### 8.1 A refine case (`designlab/refine.py`)

`CASES: dict[str, dict]` (68-139); a case's fields: `ids` (round-two candidate ids), `overrides` (tuple of `(key,
value)` for `build.Design` via `Candidate.override`), `headroom`, `fr05` (bool), `min_gap` (mm), `label`, optional
`walks` (e.g. `("fr05", "lift19")`), `headroom_by_group`, `screen` (False: sized only), `base_case`, `head_x`.

`_init(case)` (202-211) sets process state before any build: `gait.set_accuracy(round2.GAIT_ACCURACY)`,
`sizing.HEADROOM`, `sizing.HEADROOM_BY_GROUP`, `sizing.EXTRA_WALKS = extra_walks(names)`,
`round2.TURN_SEARCH_MIN_GAP_MM`.

Two ways a case's record comes to be (`_size`, 214-228):

1. Re-sized (every case but the head's): `round2.size_one(cid, out, str(Path(out) / "models"),
   base_overrides=CASES[case]["overrides"])`, then the record gains `refine_case, headroom, headroom_by_group,
   fr05_sizing, turn_min_gap_mm`.
2. Not re-sized (`CASES[case].get("head_x")`, 219): `head_x_one(case, cid, out)` (428-514). The head cases are added
   by the loop at 177-181 from `HEAD_X_BASE = {"head_x_60": ("narrow_carpet_60", "60-N-T-10"), "head_x_75":
   ("feet_75", "75-M-H-10")}` (146): `CASES[_case] = dict(CASES[_base], ids=(_cid,), overrides=base overrides +
   HEAD_X_OVERRIDES, base_case=_base, head_x=True, label=...)`. `HEAD_X_OVERRIDES = (("neck_yaw_joint", 1),
   ("jaw_joint", 1))` (145); `HEAD_X_PART = "feetech_sts3215_c018"` (144).
   - `head_x_base(case, out) -> (path, record)` (242): the pick's record `<out>/../<base>/candidates/<id>.json`.
   - `head_x_candidate(case, base_rec) -> (candidate, selection, B)` (250): the base record's overrides (with the
     packaging search's) + `HEAD_X_OVERRIDES`; `selection = dict(base_rec["selection"], **{g: HEAD_X_PART for g in
     sizing.HEAD_X_GROUPS})`.
   - `head_x_build(case, base_rec, models_dir) -> dict` (291): `pipeline.build_candidate(cand, models_dir,
     selection=sel, B=B, write=False)`, `sizing.requirements`, `_evaluate`, `mjcf.write_ledger(...)`; writes
     `<models_dir>/<id>.xml` and `<id>.json`.
   - `head_x_at_pick_pose` (378), `head_x_seams` (357), `_against` (312), `_stance_effect` (327).
   - Record (`<out>/candidates/<id>.json`) keys: `id, rev, candidate, B, H, xml_sha256, selection, metrics, margins,
     holds, power, runtime, cost, gaits, sizing_history, walk_feasible, geometry_checks, notes, overrides,
     packaging_search, mass_by_category, mass_by_volume_class, gait_accuracy, seconds, ok, refine_case, headroom,
     headroom_by_group, fr05_sizing, turn_min_gap_mm, head_x`; `head_x` keys: `rule, case, base_case, base_record,
     base_record_sha256, base_xml_sha256, part, overrides_added, groups, parts, added_mass_kg, mass_kg,
     com_ahead_of_hip_m, home_pose, head_neck_mass_kg, groups_over_planning_limit, meets_case_headroom, against_base,
     against_base_at_pick_pose, loads_between_stances, joint_ranges, selection_if_resized`.

Screened "exactly like the round-two candidates": `_screen` (231-236) -> `round2._screen_star((cid, Path(out)))` ->
`round2.screen_one(cid, out)` (round2.py:248-284): rebuilds from the record into `<case>/review/mjcf`, sets
`matches_sized = xml_sha256 == sized_sha256`, generates the nominal and brisk walks, and runs five screens, each
writing under `<case>/review/screening/`:

| Screen | Function | File |
|---|---|---|
| clearance (gates r2) | `round2._clearance` -> `clearance.screen` | `<id>_clearance.json` (+ `.png`) |
| sensors | `sensors.screen` | `<id>_sensors.json` (+ `.png`) |
| turning | `turning.screen_record(cid, rec, xml, q_home, design, scr)` | `<id>_turning.json` (+ `.png`) |
| carpet | `carpet.screen_record(cid, rec, xml, q_home, design, trajs, scr)` | `<id>_carpet.json` (+ `.png`) |
| envelope | `envelope.screen_record(...)` on a fresh build | `<id>_envelope.json` |

The summaries go to `<case>/review/review_index.json[<id>]` (`id, xml_sha256, sized_sha256, gait_accuracy,
matches_sized, clearance, sensors, turning, carpet, envelope`). Plus the carpet-lift pass (`carpet_lift_one`, 520)
whose file `refs.CARPET_LIFT_RECORD[body]` names.

CLI (`main`, 763-851):
```
python -m designlab.refine [--out round2/refine] [--workers 3] [--cases C ...] [--resume]
       [--carpet-lift [--lift-cases C ...] [--lifts L ...] [--nominal-only]]
       [--fixed [--fixed-cases C ...]] [--plates] [--plates-picks] [--report-only]
```
Without `--cases` every case in `CASES` is sized and screened (785). `--resume` skips a body whose
`candidates/<id>.json` (sizing) or `review/screening/<id>_envelope.json` (screens) exists. Every invocation ends by
rewriting `<out>/refine.json` and `<out>/REFINE.md` (848-850). Re-running a case without `--resume` rewrites its
committed candidate record (each "holds its run time", x_inputs.py:16-17).

### 8.2 Registered as a Stage 1 body (`designlab/stage1/__init__.py`)

```python
BODIES = {"60-N-T-10": "round2/refine/narrow_carpet_60", "60-N-H-10": "round2/refine/narrow_carpet_60",
          "75-M-H-10": "round2/refine/feet_75", "60-N-T-10-K": "round2/refine/narrow_knee_60",
          "60-N-H-10-K": "round2/refine/narrow_knee_60", "60-N-T-10-X": "round2/refine/head_x_60",
          "75-M-H-10-X": "round2/refine/head_x_75"}                                              # :47-55
PROBE_BODIES = ("60-N-T-10", "60-N-H-10", "75-M-H-10")                                           # :57
HEAD_X_BODIES = ("60-N-T-10-X", "75-M-H-10-X")                                                   # :60
CANDIDATE_IDS = {"60-N-T-10-K": "60-N-T-10", "60-N-H-10-K": "60-N-H-10",
                 "60-N-T-10-X": "60-N-T-10", "75-M-H-10-X": "75-M-H-10"}                         # :64-65
SHA256_PREFIX = {...,"60-N-T-10-X": "9718b3b5", "75-M-H-10-X": "66f531bb"}                       # :69-71
def case_dir(body) -> Path          # :74   STUDY / BODIES[body]
def candidate_id(body) -> str       # :81   CANDIDATE_IDS.get(body, body)
```
A registered body is read as: record `<case>/candidates/<id>.json` (`plant.record`, plant.py:108), model
`<case>/models/<id>.xml` + ledger `<id>.json` (`plant.model_files`, plant.py:116; rebuilt into
`stage1/models/<body>/` if absent), checked by `plant.verify` (plant.py:136) against the record's `xml_sha256` and
`SHA256_PREFIX[body]`; screens `<case>/review/screening/<id>_*.json`.

Other per-body tables a new body must be in:

- `refs.CARPET_LIFT` and `refs.CARPET_LIFT_RECORD` (refs.py:57-65); `refs.walk_specs(body)` indexes `CARPET_LIFT[body]`
  (refs.py:80-83).
- `env.PUSH_BOUND_M_S` only if the listed value is wanted (section 6).
- `gates.DESIGN_STUDY_STRIDES` (gates.py:99) lists the three probe bodies only.
- `stage1.ledger` holds per-body tables (`MASSES`, `STAGE0`) for the probe bodies only: "Arm K's and the expressive
  heads' bodies are not margin-ledger bodies" (`__init__.py:61-63`).
- Stage 2: `look.BODIES` (look.py:145), `screen.BODY_LABEL` (screen.py:904), `look_page.BODY_NAME` etc.

### 8.3 Prepare and gates G1-G3 (`stage2/x/run_x.sh`, `designlab/stage1/gates.py`)

`run_x.sh` (modelled on arm K's `stage1/runs/2026-11-10-speed/run_arm_k.sh`): `[PY=python] [LOG=DIR] [PROCS=2]
[G1_PROCS=N] bash stage2/x/run_x.sh [build|check|copy-from DIR|run|g1|gates|records|costs]` (default `run`). It exports
`PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHON_CPU_COUNT=$PROCS` (run_x.sh:54).

- `build` (98-125): `python -m designlab.refine --out round2/refine --workers $PROCS --cases <missing cases>
  --resume`, then the carpet-lift pass per case with its pick's lifts (`--cases --carpet-lift --lift-cases head_x_60
  --lifts 0.19 0.195 --nominal-only`; `head_x_75 --lifts 0.175 0.18 0.19`).
- `run` (127-182): `x_inputs.py models`; `python -m designlab.stage1 prepare --body B --out stage2/x --parts
  plant,refs --no-gates`; `... --parts transitions --no-gates`; then `python -m designlab.stage1 gates --body B --out
  stage2/x --only g1|g2|g3` for each gate `gates.json` lacks; `x_inputs.py check`.
- Outputs per body in `stage2/x/<body>/`: `plant.json`, `refs.json`, `transitions.json`, `gates.json` (committed);
  `refs/*.npz` (10) and `transitions/*.npz` (18) git-ignored.

Gates (gates.py docstring 1-65; `g1` 377, `g2` 638, `g3` 718):

| Gate | Passes when |
|---|---|
| G1 plant | hashes match (`plant.verify`, `refs.source`); stands `STAND_S = 20` s on hard floor and low pile with zero command, no fall, tilt below 12 deg; the tendon map equals `sizing._motor_space` to `TENDON_TOL = 1e-9`; seeded runs repeat bit for bit; at least `MIN_STEPS_PER_S = 20_000` physics steps/s per core, alone and with one process per core (fewer processes than cores leaves G1 provisional, `PROVISIONAL_RATE_RULE`, gates.py:302) |
| G2 references | every reference matches its records to 6 digits (`refs.compare`); the R0 feedforward replay of the nominal walk on the plant with the envelope: median of `PANEL_RUNS = 20` starts (offsets up to `PANEL_OFFSET_M_S = 1e-3`, seed 0) walks at least `MIN_REPLAY_STRIDES = 10` strides |
| G3 transitions | every transition of `transitions.TRANSITIONS`: last output stride within `G3_COM_MM = 1.0` mm of the periodic walk's CoM and `G3_ZMP_MM = 2.0` mm of its multibody ZMP, every frame's IK solved (transitions.py:113-114) |
| G4 tracker | "does not apply" to the `-X` bodies (`__init__.py:58-59`) |

The `-X` records read "G2 and G3 pass; G1 is provisional on its step rate" (`HEAD_X.md` section 3); `gates.json` keys:
`body, G1, same_plant_options, pass, G2, G3, provisional`.

`stage2/x/x_inputs.py`: `check`, `copy-from DIR`, `models`, `arrays BODY refs|transitions`, `records`, `manifest`
(329-344); `x_inputs.json` = `{rule, bodies, cases, files: {path relative to the study: sha256}}` (60 files: 4 model
files + 2 x (10 + 18) arrays). `models` regenerates a case's model files by `refine.head_x_build` from a scratch
working directory and puts them in place only if both sha256 match the list (196-234).

### 8.4 What the `-X` work changed outside the case (what a new joint or part costs in code)

From `HEAD_X.md` and the code: `build.py` options behind overrides that default off (`neck_yaw_joint`, `jaw_joint`;
build.py:162-163; `NECK_YAW_RANGE`, `JAW_RANGE` build.py:76-77); `sizing.HEAD_X_GROUPS = ("neck_yaw", "jaw")`,
`HEAD_X_JOINT`, `SINGLE_GROUPS` (sizing.py:41-44); `mjcf.HEAD_TAIL_ROLES` (mjcf.py:229), `mjcf._head_x` and the
integration reserve leaving the added parts out (`mjcf.body_inertials`, mjcf.py:76-80); `clearance.REST_ON_STOP`,
`SEAM_JOINTS`, `seam_sweep`; `plant.leg_drives`/`leg_joints` by role; `plant.REST_ON_STOP_RULE`. Tests check that every
earlier registry model still regenerates with its recorded `xml_sha256` (section 10).

Open on the `-X` bodies (`HEAD_X.md` section 7): G1 with one process per core; the env's 14-joint widths
(`env.ACTOR_OBS` has `("q", 14), ("qd", 14)`, env.py:138; `CRITIC_OBS` has `("joint_error", 14)`, env.py:141;
`ACTION_DIM = 15`, env.py:286; `WIDTH_JOINTS = dict(ACTOR_OBS)["q"]`, env.py:290; the envs raise `ValueError` for a
plant with another hinge count, env.py:361-363); the jaw cover's overhang (trimming it "is a model change, so every
`-X` record would change"); the yaw stage's bearing is not modelled.

### 8.5 A "Jetson body" by analogy (facts only)

- Plan wording (`JETSON_PLEM_REVIEW.md:239-246`): "refine case `round2/refine/jetson_75`, body `75-M-H-10-J`, or `-XJ`
  with the head"; dev kit in the chest; "the PCAN adapter, the regulator and the supervisor placed, and the Teensy
  stack kept or removed"; "a compute power row for the 15 W and 25 W modes as a draft A row"; "Run every round-two
  screen, then G1-G3"; a paired-seed table against the Pi body.
- Compute is a selection field, not an override: records hold `selection["compute"] = "rpi5_8gb_ai_hat_plus_13tops"`
  (`round2.COMPUTE`, round2.py:43); the build places `parts[sel["compute"]]` in the chest tray (build.py:681-694) and,
  with the battery at the tail base, the power box against the tray's length (build.py:667-674). The catalog has
  `nvidia_orin_nano_super_devkit` (parts.json:6), `pcan_usb_fd` (parts.json:474), `pololu_d42v110f12` (371),
  `pololu_d36v50f12` (397). Placement of a PCAN adapter, a regulator or a supervisor in `build.py`: **not found**; a
  supervisor part in `parts.json`: **not found**. The control stack is a fixed 0.18 kg box "Teensy 4.1, CAN x2, IMU"
  (build.py:699-700).
- Power: `power.COMPUTE_BUDGET["nvidia_orin_nano_super_devkit"] = {"jetson_module": 25.0, "carrier_storage_cooling":
  6.0}` (power.py:34-36, round one's allowance); a 15 W row: **not found**. `AUX_FIXED = {"mcu_comms": 4.0}`.
- The only existing Jetson build is a study script: `stage2/jetson_plem/jetson_plem_study.py:156 _build_with_compute`
  (swaps `sel["compute"]`, optionally monkeypatches `build.Design._place_trunk_hardware` to shorten the power box,
  builds in a temp dir and reads home-pose overlaps only). `JETSON_PLEM_REVIEW.md:166-170`: "Every dynamics result
  uses the Pi body ... The PCAN adapter, the regulator and the supervisor are never placed."
- The not-re-sized path in `refine.py` is specific to the head: `head_x_candidate` adds `HEAD_X_OVERRIDES` and an
  `HEAD_X_PART` per `sizing.HEAD_X_GROUPS` (256-257); `head_x_one` records head-specific blocks (`joint_ranges`,
  seams). A generic "pick plus parts, not re-sized" builder: **not found**.

## 9. Hashes: where computed and what they cover

| Record | Hash | Computed at | Covers |
|---|---|---|---|
| `look_refs.json` entry | `sha256` | `look.write` (look.py:1364, `_sha(path)`) | the rung's npz bytes |
| same | `spec` | look.py:1364 | compared by `look._entry_current` (1382-1385) with the file's sha256 to decide whether a rung is regenerated on `--resume` |
| `screen.json` | `inputs.look_refs_json_sha256`, `inputs.tail_phase_json_sha256` | screen.py:821, 869 | the two input files' bytes; also the partial files' resume key (836, 849) |
| same | `rungs[..]["arrays"]["sha256"]` | screen.py:527 | copied from the entry; `load_arrays` re-hashes the file (197-200) |
| same | `g2[body]["references"][name]["sha256"]` vs `recorded` | screen.py:786-788 | Stage 1's regenerated reference npz |
| same, `metrics.json`, `clips.json`, `look_page.json`, `look_refs.json` | `provenance` | `plant.provenance()` (plant.py:294-316) | git commit, `dirty`, `dirty_paths`, `stage1_sources.sha256` over all `designlab/stage1/*.py` (`SOURCES_RULE`, plant.py:263-264) |
| `metrics.json` rung | `rung_sha256` | motion.py:429 | the entry's npz sha256 |
| `clips.json` clip | `sha256`, `rung_sha256`, `model.sha256` | clips.py:580-589 | the MP4, the rung's npz, the plant's MJCF |
| `look_page.json` | `inputs{metrics.json, screen.json, clips.json, look_refs.json}`, each file's `sha256` | look_page.py:2437 | file bytes |
| candidate record | `xml_sha256` | the build's summary | the MJCF; `plant.verify` and `SHA256_PREFIX` check it |
| head-X record | `head_x.base_record_sha256`, `base_xml_sha256` | refine.py:460-461 | the pick's record file and model |
| `x_inputs.json` | `files[path]` | x_inputs.py:103-118 | each git-ignored input |
| `x_costs.json` | `script_sha256` | x_costs.py:799 | the script itself |

No Stage 2 record hashes `designlab/stage2/*.py`. The Stage 1 panels' code hash (`panels.panel_sources()`,
panels.py:1572-1583; inspected by calling it) covers:

- `designlab/stage1`: `__init__.py, control.py, drive.py, estimator.py, legs.py, metrics.py, panels.py, plant.py,
  refs.py, transitions.py, tune.py` (excluded: `env.py`, `gates.py`);
- `designlab`: `__init__.py, build.py, carpet.py, catalog.py, clearance.py, envelope.py, gait.py, geometry.py,
  hardware.py, mjcf.py, params.py, shapes.py, sizing.py, skeleton.py, statics.py, template.py, turning.py`.

Not in it: `refine.py`, `round2.py`, `pipeline.py`, `power.py`, `cost.py`, `gaittune.py`, everything under
`designlab/stage2/`. `plant.provenance()`'s `stage1_sources` covers all 22 Stage 1 files.

## 10. Tests that pin behaviour

`tests/test_stage2_screen.py` (`needs_screen` = `screen.json` and `look_refs.json` exist):

| Test | Asserts |
|---|---|
| `test_passive_loads_equal_look_and_add_damping_and_friction` | `passive_loads` ratios equal `look._passive_ratios` (rel 1e-12); damping 0.2 and friction loss 0.15 on the 60 cm body; passive terms vanish at rest |
| `test_accelerations_of_a_known_periodic_path` | `accelerations` on an analytic periodic CoM path (peaks to 1e-3; 1 Hz low pass of a 3 Hz wobble under 5 %) |
| `test_stage0_sway_reproduces_stage1_g7` | `rungs[b]["nominal"]["V0"]["sway"]["stage0_pp_H"]` rounds to 0.168 / 0.164 |
| `test_straight_run_reproduces_the_turning_screens_record` | `straight_run` on regenerated V0 nominal equals `<case>/review/screening/<id>_turning.json` `runs.straight` field by field (< 1e-9); grips follow `mu_req_max` |
| `test_every_rung_is_screened_from_its_own_arrays` | the set of `(body, speed, variant)` in `screen.json` equals that of `look_refs.json`; `inputs.arrays_checked == len(entries)`; per rung `arrays.sha256`/`file`, `reprice.equal_to_recorded`, `drives == price.drives`, `over == price.over`, `features`, `flags` equal the entry's |
| `test_g2_on_the_code_that_ran_the_screen` | `g2[b]["passes"]`, accuracy "r2", all three references bit-identical |
| `test_over_limit_rungs_follow_the_plans_rule` | every ratio > 1 listed; `rung` text contains "loses", `SPEED_LABEL[s]`, "unless the owner" and every feature; a `cause` for speed limits; `near` and `over_with_passive` follow the ratios; `passive.*.cont_ratio_id == drives.*.cont_ratio` |
| `test_tail_phase_check` | per body and speed: `f9_phase_rad` equals `tail_phase.json`; today's tail adds to and F9's phase lowers the peak yaw moment; V0 tail `{amp_rad: 0.12, phase_rad: pi/2}`; look-path control < 1e-9; rung readings equal the rungs' own friction blocks; the record check's friction fields < 1e-9 |
| `test_readings_are_consistent` | grips; `read + rolling_frames_not_read == frames`; rolling frames and `plain_gait_id` present iff `spec.toe_off`; **`clearance.walking_gates is True` and `summary.ik_ok` on every rung**; V0's `vs_V0` differences zero; every block name in `screen.SOURCES` |
| `test_screen_md_regenerates_from_screen_json` | **`SCREEN.md == screen.markdown(SCREEN)`**; `provenance.git.dirty is False`; `turning.MU == 0.8`; margins 1.25 and 1.15 |

`tests/test_stage2_motion.py`:

| Test | Asserts |
|---|---|
| `test_sparc_of_a_bell_is_scale_free_and_smoother_than_two_submovements` | a minimum-jerk bell reads -1.40 +- 0.01, scale and duration free |
| `test_sparc_periodic_of_sinusoids_harmonics_and_corners` | a sinusoid -1.95; `repeats == 20` for a 1 s period |
| `test_angular_velocity_and_relative_attitude_of_known_rotations` | `angular_velocity`, `relative_euler` |
| `test_extremes_and_flat_signals_on_a_periodic_window` | maxima/minima phases, flat parts, a synchronous pair |
| `test_crest_hold_of_a_sinusoid` | `CREST_SINUSOID` = 0.1436 |
| `test_bird_curves` | T. rex cadence 1.83 (H 0.405) and 1.64 (H 0.502) steps/s, tinamou stride 0.98 H, swing 0.32 s |
| `test_todays_walk_shows_every_mechanical_tell_and_v6_none` (arrays) | V0 at 60 cm 0.40 m/s shows the four tells, V6 none; V6 trunk roll RMS 1.5/sqrt(2) deg; no trunk pitch; V0 tail phase 0 cycles from today, V6 > 0.4 |
| `test_readings_shared_with_look_measure_equal_its` (arrays) | `metrics` equals the recorded `measured` for swing time, duty, toe-off, tail phase, head pitch RMS, bob |
| `test_feet_sparc_reads_each_swing` (arrays) | one swing per stride and foot; ranges |
| `test_plots_are_written` (arrays) | `path_plot`, `phase_chart` write PNGs |
| `test_metrics_json_reads_every_rung_from_its_recorded_arrays` | every rung of `look_refs.json` is in `metrics.json` with its sha256 and features; duty within 0.5-0.75; `constant_hip_height == (v == "V0")`; **`identical_steps == (v not in ("V6", "V6S"))`**; `set(v040.per_body) == set(look.BODIES)` |
| `test_renders_md_regenerates_from_the_json_files` | **`RENDERS.md == motion.markdown(METRICS, clips_doc)`** |
| `test_summary_cells`, `test_notes_read_metrics_json` | cell formats; the notes name "75-M-H-10 fr05 V6 (knee speed 1.012)" |

`tests/test_stage2_clips.py`: `test_playback_refuses_a_turning_window` (a `stride_motion` with a rotation raises
`ValueError`); `test_review_set_is_the_tasks_list` (42 clips, the exact set); `test_clip_times_are_ten_seconds_at_thirty_fps`;
`test_every_clip_says_kinematic_reference_not_a_simulation`; `test_every_recorded_clip_keeps_the_robot_in_the_picture`;
others on cameras, labels, encoding.

`tests/test_stage2_look.py` (neighbouring; names only): `test_the_ladder_has_every_rung` pins
`set(LADDER["bodies"][b][s]) == set(look.VARIANTS + look.EXTRAS)` (minus `V5F13` where F13 does not apply);
`test_turning_generate_refuses_a_look_spec`; `test_g2_stage1_references_regenerate_bit_for_bit`;
`test_every_variant_regenerates_to_its_recorded_hash`.

`tests/test_stage2_x.py`: `test_an_earlier_registry_model_regenerates_with_its_recorded_sha256` (every earlier body's
MJCF byte for byte, 14 hinges, no head-X rows); `test_the_options_default_to_todays_bodies`;
`test_the_head_x_bodies_are_registered_beside_the_earlier_ones` (**`set(BODIES) == set(EARLIER) | set(HEAD_X)`** and
`set(SHA256_PREFIX) == set(BODIES)`, `CANDIDATE_IDS` exactly the four); `test_a_head_x_model_regenerates_with_its_recorded_sha256`;
`test_a_head_x_case_is_its_pick_with_the_head_added`; `test_the_locked_head_x_model_has_todays_mass_properties_plus_the_added_parts`;
`test_a_head_x_carpet_swing_is_its_picks`; `test_the_head_x_prepare_records`; `test_the_head_x_gate_records`;
`test_a_rest_end_is_no_range_limit_for_stage1s_judges`; `test_the_env_refuses_an_x_plant_until_its_widths_are_generalised`
(**`env.WIDTH_JOINTS == 14 and env.ACTION_DIM == env.WIDTH_JOINTS + 1`**, and `Stage1Env` on `60-N-T-10-X` raises
`ValueError` matching "pending"). `tests/test_stage2_look_page.py`, `test_stage2_x_costs.py`,
`test_stage2_jetson_plem.py` exist (not read).

## 11. Hard-coded assumptions the build list would break

| Build item | Assumption | Where |
|---|---|---|
| A, B | the look's turns and transitions are recorded as deferred; `turning.generate` refuses any look spec | look.py:179-189 (`DEFERRED`, also written into `look_refs.json`; pinned by `test_the_ladder_has_every_rung`); turning.py:380-382 |
| A, B, C | the screen reads a periodic straight window: `look.turning_traj` yaw rate 0 and an x advance; `gait.derivatives` wraps; second differences wrap; `stride_motion[3]` is the only advance read | look.py:679-689, 796-799; gait.py:300-333; screen.py:355-383, 372 |
| A | a non-periodic window has no reader in `screen.py`, `motion.py`, `clips.py`, `look.measure`, `look.contact_events` | section 1.7 (c) |
| B | `clips.Playback` refuses a turning window | clips.py:174-175; `test_playback_refuses_a_turning_window` |
| B | thigh twist: no `hip_yaw` check inside the screen; round two's reading is `turning._range_margins` on cases at the nominal walk only (0.35 and 0.50 rad/s); 0.40 m/s at pi/4 rad/s is not a case | turning.py:149-153, 1020 |
| C | Stage 0's toe clearance reads the contact pads only, over the middle half of each swing by frame index on the planned support flags, rounded to 4 decimals of H | clearance.py:729-736, 739-790 |
| C | detected (not planned) lift-off and touchdown exist only for a periodic window, with a 0.3 mm contact tolerance on the reference's geometry | look.py:970, 983-1011 |
| C | sideways drift, a stop read as touchdowns, a turn heading rule: no Stage 0 or motion reading of them | **not found** in `screen.py`, `motion.py` |
| C | the look bands exist as four numbers only; no code judges a run against them; no trace adapter | motion.py:94; section 3 |
| C, E | pushes are `0.5 x` the bound (`PUSH_FRAC`), a module constant in a hashed module | panels.py:97, 947 |
| D | screened variants are the fixed tuple `ORDER`; unknown names in `look_refs.json` are skipped | screen.py:127, 825-826 |
| D | the ladder is `look.VARIANTS + look.EXTRAS`; `base_spec` is `sizing.NOMINAL` / `speed_spec(H)` / `envelope.FAST`; `features_of` is a fixed dict; `look prepare --variant` choices are those names | look.py:149-150, 307-315, 395-412, 1600 |
| D | `screen.run`/`markdown` need `V0`..`V6` at every body and speed, `V0` and `V2` at `v040`; `tail_phase` needs `V0, V4, V5` | screen.py:620-660, 753, 873-876, 1046-1099, 1344 |
| D | speeds are `nominal, v040, fr05`; label dicts are indexed without a default | screen.py:124-125, 516; clips.py:100-101, 436; motion.py:639 uses `.get` |
| D | `motion.run` requires every `look_refs.json` rung in `screen.json` | motion.py:629 |
| D | the committed pages regenerate byte for byte from their JSON | `test_screen_md_regenerates_from_screen_json`, `test_renders_md_regenerates_from_the_json_files` |
| D | F25's tuned walk sits on the 0.04 H gate: `toe_mid_swing_mm` 16.18731 against 16.18757 (60 cm) and 20.07400 against 20.07414 (75 cm), accepted by `gaittune` with a 0.01 mm tolerance (`toe_ok`, gaittune.py:98; reading `carpet.toe_catch`, generator `turning.generate` at yaw 0, gaittune.py:76, 84-85); the clearance gate compares `round(mid / H, 4)` with 0.04 | round2/gait_tune/{60-N-T-10,75-M-H-10}.json `chosen`; clearance.py:781, 786, 1448 |
| D | F25's `chosen.spec`: 60 cm `{froude 0.05, lift_frac 0.08, duty 0.55, stride_rel 0.7125, crouch 0.07}`; 75 cm `{froude 0.05, lift_frac 0.08, duty 0.55, stride_rel 0.8625, crouch 0.05}`; the bob rule and `look.measure` use `crouch` | look.py:318-321, 1064 |
| F | `-X` in the screen: `passive_loads`, `look._passive_ratios`, `plant_joint_friction` name only `neck_pitch`/`tail_yaw`; `limits` iterates `sizing.GROUPS`; `GROUP_LABEL` has 8 groups; `look.BODIES` has two bodies | screen.py:220, 246, 312, 901-902; look.py:145, 854 |
| F | the env's widths are for 14 hinges and it refuses another count; its push uses `.get(cfg.body, 0.5)` | env.py:138, 141, 286, 290, 361-363, 465 |
| F, G, E | registering a body edits `designlab/stage1/__init__.py` and `refs.py`, both inside the panels' code hash and `stage1_sources` | section 9; `__init__.py:47-71`; refs.py:57-65 |
| Jetson | new parts need `build.py` (and perhaps `hardware.py`, `mjcf.py`), all inside the panels' code hash; `test_the_head_x_bodies_are_registered_beside_the_earlier_ones` pins the registry's exact contents | section 9; tests/test_stage2_x.py:399-406 |
| H | notebook, Drive bundle, kill-and-resume in this subsystem: **not found** (the only resume logic here is `screen.run --resume` and `look.prepare --resume` partial files) | screen.py:807-850; look.py:1371-1447 |

## 12. CLI entry points (this subsystem)

```
python -m designlab.stage2.screen run [--out DIR] [--procs N] [--resume]
python -m designlab.stage2.screen md  [--out DIR]
python -m designlab.stage2.motion run [--media DIR] [--out DIR] [--body B ...]
python -m designlab.stage2.motion md  [--out DIR]
python -m designlab.stage2.clips render --media DIR [--body B ...] [--procs N] [--out DIR] [--no-record]
python -m designlab.stage2.clips one --media DIR --body B --speed S --variant V --view visitor|side [--mode M] [--out DIR]
python -m designlab.stage2.look_page --media DIR --out DIR [--look DIR] [--manifest FILE] [--viewers-url URL]
python -m designlab.stage2.look prepare [--body B ...] [--speed S ...] [--variant V ...] [--out DIR] [--procs N] [--resume]
python -m designlab.stage2.look g2 --body B [--name N ...] | remeasure [--out DIR] | reflag [--out DIR]
python -m designlab.envelope <round2 dir> <ids...>
python -m designlab.clearance <id> --out <dir>                       (docstring, clearance.py:8; arguments not read)
python -m designlab.refine [--out round2/refine] [--workers 3] [--cases ...] [--resume] [--carpet-lift ...] [--fixed ...] [--plates] [--plates-picks] [--report-only]
bash stage2/x/run_x.sh [build|check|copy-from DIR|run|g1|gates|records|costs]      (PY, LOG, PROCS, G1_PROCS)
python stage2/x/x_inputs.py check|copy-from DIR|models|arrays BODY refs|transitions|records|manifest
python stage2/x/x_costs.py [--procs 2] [--out stage2/x/x_costs.json]
python -m designlab.stage1 prepare --body B --out DIR --parts plant,refs,transitions --no-gates      (as run_x.sh calls it)
python -m designlab.stage1 gates --body B --out DIR --only g1|g2|g3                                   (as run_x.sh calls it)
```

## 13. Extension points and hazards (short)

- `screen_rung`, `look.generate`, `look.write`, `look.price`, `motion.metrics` all take `out_dir` or plain dicts, so a
  new periodic straight reference can be generated, priced, screened and measured in a separate directory without
  touching `stage2/look` or any hashed module (section 1.7 a). `screen.run` on such a directory needs the full
  `V0`..`V6` ladder and a `tail_phase.json`.
- Adding rungs to `stage2/look/look_refs.json` changes its sha256, so the whole committed screen (73 rungs, the tail
  checks, G2) is re-run and `screen.json`, `SCREEN.md`, `metrics.json`, `RENDERS.md` are rewritten; five tests compare
  rung sets or regenerate pages from those files (section 10).
- Anything edited in `clearance.py`, `envelope.py`, `gait.py`, `turning.py`, `sizing.py`, `build.py`, `hardware.py`,
  `mjcf.py`, `stage1/__init__.py`, `stage1/refs.py`, `stage1/metrics.py`, `stage1/panels.py` changes the panels' code
  hash; `refine.py`, `round2.py`, `power.py` and all of `designlab/stage2/` do not (section 9).
- A new body is three things in order: a refine case with a committed record and five screens; a registry entry
  (`BODIES`, `CANDIDATE_IDS`, `SHA256_PREFIX`, `CARPET_LIFT`, `CARPET_LIFT_RECORD`); prepare and G1-G3. The model files
  are git-ignored and must match the record's `xml_sha256`; `x_inputs.py` is the template for listing, checking and
  regenerating them.
- The push bound is forward only and per body in the envelope screen; the panels' fraction is a constant; the env
  falls back to 0.5 m/s for bodies outside the listed three.
- The two bodies' ladder arrays and three cases' model files were absent when this was written; every statement about
  array shapes comes from the code, not from a file.
