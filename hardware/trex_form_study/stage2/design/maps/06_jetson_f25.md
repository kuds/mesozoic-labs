# Map 06: PR #668's Jetson study script, the F25 + F9 reference, the budget and the ToF horizon

Read-only map, made 14 Nov 2026 on branch `claude/mesozoic-trex-stage2-refine-4jffre` (read at 7e9ea0d, the merge of
PR #668's branch). Paths are relative to `hardware/trex_form_study/` unless absolute. Nothing was run except JSON
reads and one evaluation of `_tuned_f9_spec` (a dataclass constructor). "Not found" means searched for and absent.
The branch moved to 167dce4 (a commit to four documents) while this was written; the `DECISION_LOG.md` and
`JETSON_PLEM_REVIEW.md` line numbers below were rechecked at 167dce4. Code and record line numbers are unaffected.

Naming trap: "F9" has two meanings. `DECISION_LOG.md:68` F9 is "round-two packaging rules". In "F25 + F9" it is
**look feature F9**, the counter-phased tail (`designlab/gait.py:123`, `stage2/research/research_look.md` §3.1).
F25 is `DECISION_LOG.md:84`, the gait tuning of `round2/gait_tune/`.

## 1. Files, entry points, hashes

| File | What it is | Lines |
|---|---|---|
| `stage2/jetson_plem/jetson_plem_study.py` | The only code of PR #668 | 700 |
| `stage2/jetson_plem/jetson_plem_study.json` | Its one record (both bodies) | 1551 |
| `stage2/jetson_plem/JETSON_PLEM_AND_SPEED.md` | Report ("JPS"); §2.3 is report-only | 127 |
| `stage2/jetson_plem/BUILD_BUDGET_75_JETSON.md` | Hand-built budget table | 94 |
| `tests/test_stage2_jetson_plem.py` | 9 tests, all read the JSON back | 139 |
| `JETSON_PLEM_REVIEW.md` | This study's review; §5 is the task list, §6 the owner's decisions (P30) | 333 |

**CLI** (`jetson_plem_study.py:671-696`), run from `hardware/trex_form_study` with `PYTHONPATH=.`:

```
MUJOCO_GL=disable OMP_NUM_THREADS=1 python stage2/jetson_plem/jetson_plem_study.py [--procs 2] [--out PATH]
```

- `--procs` (int, default 2): a `spawn` pool of `min(procs, 2)` workers, one per body (`:679-684`).
- `--out` (Path, default `stage2/jetson_plem/jetson_plem_study.json`).
- Recorded wall time: 313 s total (60 cm 312.5 s, 75 cm 224.7 s). Over the 2-minute limit, so do not run it here.

**Hash.** One hash only: `script_sha256=sha256(Path(__file__))` (`:689`), the script's own bytes. Committed value
`500e48ffb12eadca5393c4a963fb2f31803d95b90d041c7e77797bb280424dc2`, equal to the file on disk today.
- It covers no input: not `designlab/*`, not the Stage 1 modules, not the gains files, not the catalog, not the
  round-two screening records it reads. There is no provenance block, no git commit, no versions.
- `tests/test_stage2_jetson_plem.py:46-51` pins it: any edit to the script, even a comment, fails
  `test_study_record_matches_committed_script` until the 5-minute run is redone.

**Module constants** (`:50-56`): `PAIRS = (("60-N-T-10","narrow_carpet_60"), ("75-M-H-10","feet_75"))`;
`VOLTAGES = (25.2, 24.0, 22.2, 21.0)`; `PDB_SHORT_LEN_M = 0.094`. `STUDY_RULES` (`:58-86`) is prose copied into
the JSON's `rules`.

**Process-global side effects the script relies on** (hazards if its functions are imported and reused):
- `refine._init(case)` (`:121`, `:160`) sets `sizing.HEADROOM`, `sizing.HEADROOM_BY_GROUP`, `sizing.EXTRA_WALKS`,
  `round2.TURN_SEARCH_MIN_GAP_MM` and the gait accuracy (`designlab/refine.py:202-211`).
- `gait.set_accuracy("r2")` (`:122`, `:398`) is module state. The 0.40 m/s probe reference (`:501`) is generated
  with no `set_accuracy` call of its own; it inherits "r2" from the earlier `_eval_speed_point` calls.
- `_build_with_compute` monkeypatches the class method `dl_build.Design._place_trunk_hardware` (`:166-192`),
  restored in a `finally`.

## 2. Functions of `jetson_plem_study.py`

| Function | Lines | Signature and return |
|---|---|---|
| `verify_baseline_reproduction` | 117-153 | `(src: dict, case: str) -> dict`: 25 relative differences (mass; per group `hold`, `double_support`, `rms` against `rec["margins"]`), `max_rel_diff`, `all_match`, baseline exterior and power |
| `_build_with_compute` | 156-248 | `(src, case, compute_id: str, pdb_len_m: float\|None=None) -> dict`: rebuilds the body with another compute part in a temp dir |
| `evaluate_power_and_regulators` | 251-344 | `(src) -> dict(regulators=..., stacks=...)` |
| `_std_spec` | 347-366 | `(H, v) -> GaitSpec`: today's walk at speed v |
| `_tuned_f9_spec` | 369-390 | `(body: str, H: float, v: float) -> GaitSpec` (section 3) |
| `_eval_speed_point` | 393-467 | `(src, spec, usable_map: dict) -> dict`: one Stage 0 row |
| `evaluate_speed_and_latency` | 470-634 | `(src, case) -> dict`: speed rows, the probe, the stopping horizon |
| `_body_worker` | 637-668 | `(pair) -> (body, dict)` |
| `main` | 671-696 | writes the JSON |

`src` is `designlab.stage1.refs.source(body)` (`designlab/stage1/refs.py:97-110`): keys `body, M, q_home, design,
H, layout, record, model`; `model` holds `matches_record` and `matches_plan` (the sha256 check of the rebuilt MJCF
against the record's `xml_sha256` and `stage1.SHA256_PREFIX`, `designlab/stage1/__init__.py:69-71`).

## 3. `_tuned_f9_spec`: the exact GaitSpec

`jetson_plem_study.py:369-390`, quoted in full where the wording matters:

```python
fr = v * v / (gait.G * H)                                   # gait.G = 9.81
crouch = 0.07 if body.startswith("60") else 0.05
duty = 0.58 if v < 0.36 else 0.55
base_sr = 0.713 if body.startswith("60") else 0.863
sr_cap = 0.90 if body.startswith("60") else 0.94
sr = round(min(sr_cap, max(0.62, base_sr * ((max(fr, 0.025) / 0.05) ** 0.20))), 4)
return gait.GaitSpec(froude=fr, duty=duty, lift_frac=0.081, head_amp=0.05, tail_amp=-0.125, sway_gain=1.0,
                     stride_rel=sr, crouch=crouch, zmp_iters=6, zmp_gain=0.5, n=160)
```

Facts a builder needs:

- **Body is matched by `startswith("60")`.** Every other name, `75-M-H-10-X` included, takes the 75 cm values. A
  60 cm `-X` or `-K` body takes the 60 cm values.
- **Crouch is 0.05 H at 75 cm but 0.07 H at 60 cm** (`:373`). The task brief's "crouch 0.05 H" holds for 75 cm only.
- **Duty**: 0.58 for v < 0.36 m/s, 0.55 from 0.36 m/s up (`:374`). A step, not a ramp. At 0.35 m/s it is 0.58.
- **Lift** 0.081 H (`:381`). F25's record has 0.080 H (`round2/gait_tune/75-M-H-10.json` `chosen.spec.lift_frac`).
- **Stride rule** (`:375-377`): `base_sr * (max(Fr, 0.025) / 0.05) ** 0.20`, clamped to [0.62, cap], rounded to 4
  places. Below Froude 0.025 the stride is constant: 0.7513 H at 75 cm, 0.6207 H at 60 cm. F25's record has
  `stride_rel` 0.8625 at 75 cm; the PR anchors at 0.863.
- **Flipped tail**: `tail_amp=-0.125` with `tail_phase` left at its default pi/2 (`gait.py:103`, `:123`). The tail
  law is `tail_amp * sin(w t + pi/2)` in today's generator (`gait.py:265`), so the sign flip is a half-cycle shift,
  equal to `tail_amp=+0.125, tail_phase=-pi/2`.
- **`gait.look_active(spec)` is False** (checked): no look field differs from its default, so `gait.generate` takes
  today's code path (`gait.py:193`), not `_generate_look`. Setting `tail_phase` instead of flipping the sign makes
  the spec look-active and routes it through `_generate_look` (`gait.py:856`, tail law at `:969`). That is a
  different generator; whether the two give bit-identical arrays for the same tail was not checked.
- **Froude comes from the speed**, so the rungs are at the rounded speeds, not the study's Froude-defined ones:
  the "animal pace" tuned row is at 0.4960 m/s (Fr 0.04997), while `envelope.FAST` is Fr 0.05 = 0.49614 m/s. The
  nominal row is 0.2430 against `sizing.NOMINAL`'s 0.24306 m/s; brisk 0.3510 against 0.35083.
- `head_amp=0.05`, `sway_gain=1.0`, `zmp_iters=6`, `zmp_gain=0.5`, `n=160` are today's defaults.
- F25's record holds one speed only (Fr 0.05). Every other speed of this spec is the PR's extrapolation.

Values at 75 cm (H = 0.5018535 m), evaluated with the committed function:

| v, m/s | Froude | duty | stride_rel | period, s | cadence, steps/s | In the JSON |
|---|---|---|---|---|---|---|
| 0.243 | 0.011994 | 0.58 | 0.7513 | 1.5516 | 1.289 | yes |
| 0.30 | 0.018281 | 0.58 | 0.7513 | 1.2568 | 1.591 | no |
| 0.35 | 0.024882 | 0.58 | 0.7513 | 1.0773 | 1.857 | no |
| 0.351 | 0.025025 | 0.58 | 0.7514 | 1.0743 | 1.862 | yes |
| 0.40 | 0.032499 | 0.55 | 0.7918 | 0.9934 | 2.013 | yes |
| 0.45 | 0.041132 | 0.55 | 0.8300 | 0.9256 | 2.161 | no |
| 0.496 | 0.049971 | 0.55 | 0.8629 | 0.8731 | 2.291 | yes |
| 0.50 | 0.050780 | 0.55 | 0.8657 | 0.8689 | 2.302 | no |
| 0.55 | 0.061444 | 0.55 | 0.8993 | 0.8206 | 2.437 | no (0.56 and 0.62 are) |

At the nominal speed the cadence, 1.29 steps/s, is below `BEHAVIOR_SPEC.md:16`'s 1.6-2.2 steps/s and the stride,
0.75 H, is above its 0.5-0.65 H. 60 cm values: 0.218 and 0.315 m/s give duty 0.58, stride 0.6207; 0.40 gives 0.55,
0.6829; 0.446 gives 0.55, 0.7133; all with crouch 0.07.

**`F25_F9_cl` is not in any code or record.** It appears only in `JETSON_PLEM_AND_SPEED.md:88`,
`JETSON_PLEM_REVIEW.md` and `NEXT_SESSION_PROMPT.md`. Its text definition: "`duty = 0.59, lift_frac = 0.095` to
`F25 + F9`", so `dataclasses.replace(_tuned_f9_spec(body, H, v), duty=0.59, lift_frac=0.095)`. JPS uses it only at
0.40, 0.45, 0.496 and 0.55 m/s; whether its duty also steps at 0.36 m/s is not stated.

**F9 tail phase, from `stage2/look/tail_phase.json`** (written by `look._merge`, `designlab/stage2/look.py:1459-1462`;
method `look.tail_check`, `look.py:1226-1262`). Structure:
`bodies.<body>.<speed>.{tail_phase_F9_rad, V0: {...}, V4: {...}}` with speeds `nominal`, `v040`, `fr05`; the 75 cm
`fr05` entry adds `tail_phase_F9_F13_rad` and `V4F13`. Each `V0`/`V4` block has `amp_rad`, `no_tail`, `runs`
(four phases), `fit_max_residual_nm`, `predicted_peak_nm{today, flipped, best}`, `peak_vs_no_tail_rel`,
`best_phase_rad`, `best_phase_cycles_from_today`, `confirmed{peak_nm, rms_nm, mu_req_max}`. Top-level keys:
`bodies, accuracy, values, variants, extras, speeds, provenance, versions, method, convention`.

`tail_phase_F9_rad` is the **V4** minimiser (the look ladder's F9). The review asks for the **V0** sweep's best
phase (`V0.best_phase_rad`), which is on today's walk at amplitude 0.12 rad:

| Body | Speed | `V0.best_phase_rad` | Cycles from today | Predicted peak, N m: today / flipped / best | `tail_phase_F9_rad` (V4) |
|---|---|---|---|---|---|
| 60-N-T-10 | nominal | -1.8500490071139892 | 0.4556 | 1.7735 / 1.4663 / 1.4603 | -1.7016960206944713 |
| 60-N-T-10 | v040 | -1.8849555921538759 | 0.4500 | 3.7372 / 3.1758 / 3.1614 | -1.7802358370342162 |
| 60-N-T-10 | fr05 | -1.8849555921538759 | 0.4500 | 4.2293 / 3.6318 / 3.6165 | -1.7802358370342162 |
| 75-M-H-10 | nominal | -1.8500490071139892 | 0.4556 | 2.4794 / 1.9658 / 1.9558 | -0.9250245035569944 |
| 75-M-H-10 | v040 | -1.8413223608540177 | 0.4569 | 4.6260 / 3.7324 / 3.7153 | -1.7715091907742444 |
| 75-M-H-10 | fr05 | -1.8849555921538759 | 0.4500 | 5.8255 / 4.8236 / 4.7983 | -1.7715091907742444 |

- The flip is phase -pi/2 = -1.5708 rad; the V0 best is 0.27-0.31 rad later. The flip is within about 0.5 % of the
  best in predicted peak.
- These sweeps are on V0 (`sizing.NOMINAL`, `look.speed_spec(H)`, `envelope.FAST`), not on the F25 walk. No tail
  sweep exists for the F25 walk: not found.
- `tail_phase.json` is a hashed input of the screen: `screen.py:821` and `:869` store `tail_phase_json_sha256` in
  `screen.json`, and `tests/test_stage2_screen.py:177-200` and `tests/test_stage2_look.py:318-331` read it. Adding
  F25 entries to that file changes the hash.

## 4. How the script made its references and its start transition

**Stage 0 rows** (`_eval_speed_point`, `:393-467`): `gait.generate(M, q0, H, spec)` then `gait.inverse_dynamics`.
No arrays are saved. No `refs.write`, no npz, no sha256 of any array.

**The probe's periodic reference** (`:500-503`):

```python
sp_40 = _tuned_f9_spec(body, H, 0.400)
tr_40 = gait.generate(src["M"], src["q_home"], H, sp_40)
idr_40 = gait.inverse_dynamics(src["M"], tr_40)
ref_40 = refs.Reference("tuned_040", refs.arrays(src, dict(traj=tr_40, idr=idr_40)))
```

- Built in memory with `refs.arrays` (`refs.py:145-166`); keys `t, q, qd, qdd, tau, omega, support, com_ref,
  zmp_ref, zmp_mb, cop_margin, normal, friction, feet, com, q_home, joints, dt, stride_motion`.
- It bypasses `refs.generate` (`refs.py:113-134`), which accepts only names from `refs.walk_specs(body)`
  (`nominal, brisk, fr05, carpet`; `refs.py:80-82`) or `turn_cases()`. No summary, no record check, no `ik_ok` read.

**The hand-built start** (`:505-575`). It copies the body of `transitions.generate`
(`designlab/stage1/transitions.py:634-678`) because that function takes a name from `TRANSITIONS`
(`transitions.py:120-131`) and resolves it with `gait_of(body, name, H)` (`:165-179`), which reads only
`refs.walk_specs` and, for turns, `sizing.NOMINAL`.

| Step | Script | Official `transitions.generate` |
|---|---|---|
| Gait record | `transitions.Gait(name="tuned_040", period, duty, lift=lift_frac*H, speed, yaw_rate=0, head_amp, tail_amp, crouch=crouch*H, zmp_iters=2, zmp_gain=0.5, split="statics")` (`:507-520`) | `gait_of`: the spec's own `zmp_iters` (6) |
| Plan | `transitions.plan("start", Gait("stand"), g_40, Geometry.of(M, q_home, g_40.crouch))` (`:521-522`) | same call |
| ZMP loop | `for it in range(3)`: three passes, the correction `zref -= 0.5 * nan_to_num(zmb - pl.zmp)` applied when `it < 2`, so **2 corrections** (`:525-531`) | `iters + 1` passes, `iters` corrections: 6 for this spec (`:648-663`) |
| ZMP error | `zmp_error_history_m=[0.0]` (`:564`), a placeholder | the per-pass maximum, recorded |
| IK | `ik_ok=True` hard-coded (`:570`); `ferr` is never compared with anything | `bool(np.all(foot_err[o] <= turning.IK_TOL))`, `IK_TOL = 1e-4` (`:677`) |
| Timing | `seconds=0.0` | measured |
| Accuracy | none set here (inherits "r2") | `gait.set_accuracy(refs.ACCURACY)` (`:638`) |
| Output | `transitions.Transition("start", transitions.arrays(dict(...)))` (`:540-575`), in memory | `transitions.write` saves the npz and its sha256 and a `summary` with G3 (`:883-917`) |

- `transitions.arrays` (`:865-876`) reads only `plan, q, qd, qdd, tau, com_ref, zmp_ref, zmp_mb, zref, cop_margin,
  normal, friction, feet, com, q_home, joints`. The forced `ik_ok`, the fake history and `x0` are dropped, so the
  transition was never IK-checked or G3-gated (`G3_COM_MM = 1.0`, `G3_ZMP_MM = 2.0`, `:113-114`).
- The transition is named `"start"`, which in `TRANSITIONS` means stand to `nominal`. `control.Track.add_transition`
  looks the name up to detect a stop (`designlab/stage1/control.py:994`), so the name carries meaning there.
- No speed change and no stop were built. The start goes from standing straight to the 0.40 m/s walk.
- `transitions.Gait` (`:143-158`) has 11 fields and none of the look fields; the tail law inside `plan` is today's.
  A negative `tail_amp` passes through it; a `tail_phase` would be lost (see `tail_phase.json` `values.deferred.why`).

## 5. The closed-loop probe (`:489-599`)

| Item | Value | Where |
|---|---|---|
| Plant | `plant.build(body, "hard")`: the **Pi body** from the round-two record, default grip, no Jetson mass | `:489` |
| Controller, 60 cm | `speedg5.json` `seeds["42"]["selected_gains"]`, label `"R2s"` | `:490-494` |
| Controller, 75 cm | no `speedg5.json` exists for 75 cm, so `env.tuned_gains(run_dir, body, tune.json)`, label `"R2"`. It returns the better-ranked seed: **CMA-ES seed 43, candidate `seed43_gen40`** (checked; `designlab/stage1/env.py:182-212`). So the 75 cm probe is R2/43 only; R2/42 was not run | `:495-498` |
| Tracker | `control.Tracker(p24, track, "R2", gains=gains, rate=tune.COMMAND_RATE)`; rate 1.05 | `:586` |
| LIPM frequency | `control.omega0(p24, ref_40.q_home, crouch_frac=sp_40.crouch)`: the walk's own crouch | `:576` |
| Track | the hand-built start, then `add_periodic(ref_40, strides=5)`; the join uses `next_frame` = the reference's frame 0 and `next_dt=ref_40.step` | `:577-580` |
| Episode length | `track.t_end / COMMAND_RATE`: start window (phases -2 to 3, `transitions.WINDOWS`) plus 5 strides, about 9.9 s of reference at 75 cm | `:590` |
| Delays | `d_steps in (0, 1, 3, 5)` physics steps of 2 ms: 0, 2, 6, 10 ms. **4 and 8 ms were not run** | `:583` |
| Delay model | `with speedfit.servo_delay(p24, d_steps)` (`designlab/stage1/speedfit.py:325-361`) | `:587` |
| Episodes | 3 per delay | `:585` |
| Seeds | `seed=tune.derived_seed(4100 + s_i, 0, 0)`, `noise_seed=tune.derived_seed(4100 + s_i, 0, 1)`, `s_i` 0-2. Not the panel seeds 3042-3061 | `:592-593` |
| Start | `start="stance"`, `joint_noise=tune.START_JOINT_NOISE_RAD` (0.01), `root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S` (0.02) | `:591-595` |
| Reading | `out_ep["fell"]` only. No `metrics.Recorder`, no `tune.measure`, no speed, toe clearance, duty or drift | `:597-598` |

Recorded result (`speed_and_latency.closed_loop_latency_probe_040ms`; the key says "040ms" but means 0.40 m/s):

| Body | Label | 0 ms | 2 ms | 6 ms | 10 ms |
|---|---|---|---|---|---|
| 60-N-T-10 | R2s | 3/3 | 3/3 | 2/3 | 0/3 |
| 75-M-H-10 | R2 | 3/3 | 3/3 | 3/3 | 1/3 |

Each cell is `{delay_ms, survived, total, rate}` under keys `"0ms"`, `"2ms"`, `"6ms"`, `"10ms"`.

**The probe's omega differs from every official evaluator's.** `control.omega0`'s default is `crouch_frac=0.03`
(`control.py:943`), and the panels, tuner, speed fit, env and metrics all call it with the default on
`refs/nominal.npz`'s home pose: `panels.py:996` and `:2173`, `tune.py:805`, `speedfit.py:432`, `env.py:380`,
`metrics.py:432`, `gates.py:863`. An official panel on an F25 reference would therefore use the 0.03 H crouch in the
tracker's LIPM frequency where the probe used 0.05 H (about 1.2 % in omega at 75 cm).

## 6. Structure of `jetson_plem_study.json`

Top level: `study` ("stage2_jetson_orin_plem_and_speed"), `script`, `script_sha256`, `wall_s`, `rules` (4 strings),
`bodies` (2 keys). Per body:

```
body, refine_case, H_m, B_m, wall_s
baseline_reproduction: xml_matches_record, xml_matches_stage1_plan, n_checks (25), max_rel_diff, all_match,
    baseline_exterior_mm{crown_mm, torso_width_mm, hardware_width_mm, thigh_width_mm, max_width_mm, com_xyz_mm[3]},
    baseline_mass_kg, baseline_drives_w, baseline_total_w, baseline_runtime_min
jetson_packaging:
    drop_in_devkit: compute_part, pdb_length_mm, mass_kg, added_mass_over_pi5_g, exterior_mm{as above},
        max_chest_profile_flare_mm, compute_pos_trunk_mm[3], pdb_pos_trunk_mm[3], same_body_overlap_count,
        same_body_overlaps[{geoms[2], gap_mm}], min_pdb_gap_mm, min_pdb_pair[2]
    shortened_94mm_pdb_fix: the same block, or null (75 cm)
power_and_regulators:
    regulators: mode_15w{id, max_vin_v, vout_v, max_power_w, mass_g, price_usd, clears_6s_full_25_2v,
        load_fraction_at_18w_peak}, mode_25w{..., load_fraction_at_28_5w_peak}, usb_can_fd{id, mass_g, power_typ_w, price_usd}
    stacks.<pi5_hailo_teensy_r2 | jetson_15w_teensy | jetson_15w_partial_plem | jetson_25w_partial_plem>:
        description, regulator_id, drives_w, aux_input_w, total_walking_w, net_mass_delta_g,
        runtime_6s_3300_min, runtime_6s_5000_min, meets_20min_spec_on_3300
speed_and_latency:
    stage0_standard[6], stage0_tuned_f9[6]: speed_ms, froude, duty, stride_rel, crouch, tail_amp, gates_ok,
        mid_swing_pad_mm, min_joint_margin_rad, worst_group_24v, worst_planning_ratio_24v,
        voltage_speed{"25.2V"|"24.0V"|"22.2V"|"21.0V": {worst_group, raw_speed_ratio, planning_speed_ratio,
        over_no_load, knee_planning_ratio}}, tail_cont_ratio_total, neck_cont_ratio_total, mu_req_max
    closed_loop_latency_probe_040ms: controller, results{"0ms","2ms","6ms","10ms"}
    stopping_horizon: camera_near_floor_ahead_of_toes_m, tof_floor_window_ahead_of_toes_m[2],
        blind_gap_tof_to_camera_m, standoff_m, usable_tof_stopping_budget_m, max_one_foot_decel_ms2,
        max_tof_safe_speed_{pi5,jetson_plem}_{worst,mean}_ms
```

Units are in the key names (mm, g, kg, W, min, m, m/s as `_ms`, m/s² as `_ms2`). No arrays are stored. `H_m` and
`B_m` are rounded to 5 places (0.50185); the record's H is 0.5018535496646683.

Speeds evaluated (`:481-485`): 60 cm `[0.218, 0.315, 0.400, 0.446, 0.520, 0.580]`; 75 cm
`[0.243, 0.351, 0.400, 0.496, 0.560, 0.620]`. The tests index rows by `speed_ms` rounded to 4 places
(`std_map[0.4]`).

75 cm rows, standard / tuned + F9:

| v, m/s | duty | stride_rel | pad, mm (gate 20.07) | joint margin, rad | worst at 24 V | tail A1 | neck A1 | mu needed |
|---|---|---|---|---|---|---|---|---|
| 0.243 | 0.60 / 0.58 | 0.55 / 0.7513 | 25.09 / 20.70 | 0.169 / 0.120 | neck 0.662 / thigh twist 0.716 | 0.800 / 0.288 | 0.676 / 0.672 | 0.361 / 0.234 |
| 0.351 | 0.58 / 0.58 | 0.62 / 0.7514 | 28.12 / 20.70 | 0.202 / 0.173 | knee 0.742 / thigh twist 0.710 | 0.953 / 0.324 | 0.683 / 0.679 | 0.558 / 0.387 |
| 0.400 | 0.573 / 0.55 | 0.6859 / 0.7918 | 28.14 / 20.32 | 0.205 / 0.183 | knee 0.804 / thigh twist 0.723 | 0.978 / 0.349 | 0.684 / 0.679 | 0.629 / 0.413 |
| 0.496 | 0.56 / 0.55 | 0.80 / 0.8629 | 28.59 / 20.32 | 0.213 / 0.200 | knee 0.935 / hip pitch 0.751 | 1.030 / 0.413 | 0.689 / 0.686 | 0.787 / 0.572 |
| 0.560 | 0.56 / 0.55 | 0.852 / 0.9058 | 28.59 / 20.32 | 0.220 / 0.209 | knee 1.056 / knee 0.854 | 1.078 / 0.461 | 0.692 / 0.689 | 0.941 / 0.695 |
| 0.620 | 0.56 / 0.55 | 0.8983 / 0.94 | 28.59 / 20.32 | 0.225 / 0.217 | knee 1.178 / knee 0.953 | 1.120 / 0.514 | 0.695 / 0.691 | 1.103 / 0.824 |

Every row has `gates_ok: true`. The tuned walk's mid-swing pad height is 20.32 mm against the 20.07 mm target:
0.25 mm of margin. At 0.243 m/s the tuned walk's joint margin, 0.120 rad, is close to the 0.10 rad gate.

**What `_eval_speed_point` computes** (`:393-467`):
- `ms = sizing._motor_space(M.names, idr["tau"], idr["omega"], layout)`; `envelope._ratios(design, bus, ms)`
  (`designlab/envelope.py:61-72`) gives `worst` per group at the record's 24 V bus, with `sizing.SPEED_MARGIN`
  (1.15) and `PEAK_MARGIN`.
- Per voltage: `wpk / act.speed_at(vb)` raw, x 1.15 planning, and `wpk / drive.no_load_speed_at(...)`. `speed_at`
  takes no credit above nominal voltage (`designlab/catalog.py:77-82`), so 25.2 V equals 24.0 V for the 24 V drives.
- `screen.passive_loads(M, design, tau, omega)` (`designlab/stage2/screen.py:307-336`), which hard-codes the joints
  `(("neck", "neck_pitch"), ("tail", "tail_yaw"))` (`:312`).
- Friction: `turning.inverse_dynamics(M, look.turning_traj(tr, M, q0), turning.MU, split="statics",
  ds_friction=False)`, then the maximum single-support `mu_req` (`:425-427`).
- Clearance: `carpet.swing_profiles(M, tr)`, minimum `z_pad` over `carpet.MID_SWING` against `0.04 * H`
  (`:429-434`). `MID_SWING = (0.25, 0.75)`, "fraction of each swing's duration" (`designlab/clearance.py:164`): the
  middle half of the reference's own swing, read on the prescribed walk, lowest pad point. Joint range: margins against `usable_map` from the round-two `<body>_clearance.json`
  `summary.joint_range_margins`, gate `clearance.RANGE_MARGIN_RAD` = 0.10 (`designlab/clearance.py:140`).
- `gates_ok = tr.get("ik_ok", True) and pad >= target - 0.05 mm and margin >= 0.10 - 1e-6` (`:450`). It does not
  run the walking clearance screens, the CoP margin, the sway or acceleration specs, or `designlab.stage2.screen`.

`_std_spec` (`:347-366`): `sizing.NOMINAL`, `sizing.BRISK`, `envelope.FAST` at their own speeds (within 1.5 mm/s);
`look.speed_spec(H, v, lo, hi)` between them (`look.py:295-304`); above `FAST` it is
`replace(envelope.FAST, froude=fr, stride_rel=min(0.96, 0.80 * (fr / 0.05) ** 0.26))`, the PR's own rule.

## 7. How the script rebuilt the Jetson body

`_build_with_compute(src, case, "nvidia_orin_nano_super_devkit", pdb_len_m)` (`:156-248`):

1. `sel = dict(rec["selection"]); sel["compute"] = compute_id` (`:163-164`). Nothing else in the selection changes.
2. `pipeline.build_candidate(cand, tmpdir, selection=sel, B=float(rec["B"]), write=True)` (`:190`;
   `designlab/pipeline.py:24-44`). The body is not re-sized, the stance is re-solved, nothing is written outside the
   temp dir, and no refine case or record is made.
3. Readings: `metrics.measure` (exterior), `clearance._Screen(xml)` + `clearance._same_body_overlaps` at home only,
   the power box's smallest gap to other trunk hardware, `geometry_checks["trunk_widened_for_hardware"]`.

**Where the dev kit goes** (`designlab/build.py:681-694`, unchanged code; the part only changes its size):
- trunk station `cx = 0.60 * B`, height `cz = min(z_wide + 0.02 B, dorsal - tray_z / 2 - 0.014)`, rotated with the
  chest bend;
- the tray is the board plus `(0.017, 0.020, 0.030)` m: 120 x 110.5 x 64.77 mm for the dev kit;
- the board's mass is the catalog's (0.174 kg); the tray item is a fixed 0.12 kg, `collide=False`.
- Recorded `compute_pos_trunk_mm`: `[313.84, 0.0, -12.79]` (75 cm), `[252.1, 0.0, -4.47]` (60 cm).

**The power box** (`power_distribution`, `build.py:663-680`): 110 x 70 x 50 mm and 0.30 kg on a 24 V bus. With the
battery at the tail base it sits in the abdomen at
`px = min(0.22 B + pd/2, 0.60 B - tray_x/2 - pd/2 - 0.010)`: 10 mm behind the compute tray, so a longer tray pushes
it rearward.
- 75 cm: `pdb_pos_trunk_mm [186.91, 0.0, -76.21]`, gap +20.70 mm (to `hw_compute_tray`), 0 overlaps.
- 60 cm drop-in: `[117.81, 0.0, -36.22]`, **-6.95 mm** into `hw_left_hip_roll_motor`, 1 overlap.
- 60 cm fix: the monkeypatch (`:168-183`) replaces the item with a 94 x 70 x 50 mm box at the same rule's position:
  `[138.45, 0.0, -34.14]`, gap +10.00 mm, 0 overlaps. The box's mass stays 0.30 kg.

**Masses** (model totals):

| Body | Pi 5 record | Dev kit | Delta | CoM x, mm |
|---|---|---|---|---|
| 75-M-H-10 | 14.854912 kg | 14.9336 kg | +78.7 g | 67.07 to 68.75 |
| 60-N-T-10 | 10.811683 kg | 10.8918 kg | +80.1 g | 33.91 to 35.20 |
| 60-N-T-10, 94 mm box | | 10.8911 kg | +79.4 g | 36.00 |

The part difference is 74 g (0.174 - 0.100 kg); the rest is the ledger's integration reserve, 6 % of installed mass
(`designlab/mjcf.py:100-102`). Exterior (`crown_mm`, widths) is identical to the baseline in the JSON.
`max_chest_profile_flare_mm` is 11.78 (75 cm) and 21.77 (60 cm), the same as the committed Pi records'
`geometry_checks.trunk_widened_for_hardware`, so the dev kit adds none. The test file's docstring says "zero chest
profile flare"; no test asserts it.

**Never placed or modelled**: the regulator, the PCAN-USB FD adapter, the supervisor, the NVMe drive. The 0.18 kg
`control_stack` (Teensy 4.1, CAN x2, IMU; `build.py:695-701`) stays in the model in every stack. Every dynamics
result in the JSON (speed rows, the probe) uses the Pi body: `src` and `plant.build(body, "hard")`.

Slips between the report and the JSON: JPS gives the 75 cm crown as 752.71 mm (JSON 750.80) and the 60 cm fix's mass
delta as +74.2 g (JSON +79.4).

**Power and runtime** (`evaluate_power_and_regulators`, `:251-344`):
- `drives_w = rec["power"]["drives_w"]` (69.425 W at 75 cm, 48.978 W at 60 cm) and
  `sensors_w = rec["power"]["aux"]["loads"]["sensors"]` (4.4 W) are read from the round-two record. The drives' power
  is not recomputed for the heavier body.
- Two converter efficiencies: `eta_12v = 0.93` (hard-coded, `:267`) and `eta_5v = power.ETA_DCDC` (0.88).
- Stack inputs, W at the bus (`:270-295`):

| Stack | Formula | 75 cm total | Runtime 3300 / 5000 mAh |
|---|---|---|---|
| `pi5_hailo_teensy_r2` | `(9.5 + 4.0 + sensors) / 0.88` = 20.34 | 89.77 W | 36.73 / 55.64 min |
| `jetson_15w_teensy` | `17.5 / 0.93 + (4.0 + sensors) / 0.88` = 28.36 | 97.79 W | 33.71 / 51.08 min |
| `jetson_15w_partial_plem` | `17.5 / 0.93 + (2.0 + sensors) / 0.88` = 26.09 | 95.51 W | 34.52 / 52.30 min |
| `jetson_25w_partial_plem` | `28.5 / 0.93 + (2.0 + sensors) / 0.88` = 37.92 | 107.34 W | 30.71 / 46.53 min |

- 17.5 W is 15 W plus 2.5 W "NVMe/fan"; 28.5 W is 25 W plus 3.5 W; 2.0 W is 0.5 W USB-CAN plus 1.5 W "IMU/E-stop
  watchdog". All are literals with no source in the script.
- Runtime: `power.runtime(pack, total)["minutes"]` = `60 * capacity_wh * 0.75 / total_w`
  (`designlab/power.py:104-107`) on `lipo_6s_3300` (73.26 Wh) and `lipo_6s_5000` (111 Wh).
- `net_mass_delta_g` is arithmetic, not a model mass: `orin.mass - pi5.mass + regulator.mass`, less a literal
  `0.064` kg for the Partial PLEM stacks (`:286`, `:292`). The 0.064 has no derivation, and the 68 g adapter is not
  added. Results: +81.0, +17.0, +25.0 g.
- The regulator block's `max_vin_v`, `vout_v`, `max_power_w` and `clears_6s_full_25_2v=True` are literals
  (`:316-335`), not read from the catalog. Only `id`, `mass_g`, `price_usd` come from `catalog.part`.
- It does **not** use `power.COMPUTE_BUDGET` or `power.aux_power` (section 11).

## 8. The stopping horizon (the ToF "safe speed") as coded

`evaluate_speed_and_latency`, `:601-633`. Inputs, all from committed round-two screening records:

| Quantity | Source | 60 cm | 75 cm |
|---|---|---|---|
| `cam_near_m` | `<body>_sensors.json` `camera.poses.home.centre_column.visible.ahead_of_toes_m` | 2.3956 | 2.395 |
| `tof_in_m, tof_out_m` | `<body>_sensors.json` `summary.tof.tof_l.floor_ahead_of_toes_m` | 0.25, 0.9428 | 0.2754, 1.11 |
| `a_1foot` | `<body>_envelope.json` `stopping.max_one_foot_decel_ms2` | 1.2499 | 1.4137 |

Files are in `round2/refine/<case>/review/screening/`. Then (`:604-616`):

```python
standoff_m = 0.10
usable_tof_m = tof_out_m - standoff_m
def safe_stop_speed(t_tof_plus_host, commit_strides):
    for v in np.linspace(0.20, 0.90, 141):
        fr = v * v / (9.81 * H)
        sr = min(0.92, max(0.55, 0.80 * ((max(fr, 0.012) / 0.05) ** 0.18)))
        l_str = sr * H
        d_stop = (commit_strides + 0.75) * l_str + (v * v) / (2.0 * min(a_1foot, 1.50))
        if v * t_tof_plus_host + d_stop <= usable_tof_m:
            best = float(round(v, 3))
```

Called four times (`:629-632`): `(0.133 + 0.200, 1.00)` Pi worst; `(0.133 + 0.060, 1.00)` Jetson worst;
`(0.067 + 0.175, 0.50)` Pi mean; `(0.067 + 0.045, 0.50)` Jetson mean. Results: 60 cm 0.485 / 0.550 / 0.710 / 0.840;
75 cm 0.540 / 0.610 / 0.815 / 0.900 m/s.

Every timing and distance term is a literal: 0.133 s (2 frames at 15 Hz), 0.067 s (1 frame), 175-200 ms and
45-60 ms host latency, the 0.10 m standoff, the commitment of 1.0 or 0.5 stride plus 0.75 stride, the 1.5 m/s² cap,
and a third stride rule (exponent 0.18) that matches neither `_std_spec` (0.26) nor `_tuned_f9_spec` (0.20). The
stop is not simulated; `transitions.py` is cited in the rules text but not called here.

## 9. `JETSON_PLEM_AND_SPEED.md` §2.3 (lines 86-101): what has no code or record

§2.3 arrived in commit 6162b45, which changed the report only. Neither the script nor the JSON holds any of it. The
PR's only closed-loop record is section 5's probe. Row by row:

| Row (JPS line) | Numbers | Backing |
|---|---|---|
| Framing (88) | "20-seed panel", "`tune.measure` 10 s steady window", `F25_F9_cl` = duty 0.59, lift 0.095 | No code. `F25_F9_cl` is defined only in this sentence |
| V0 brisk, 0.351 m/s, 0 ms (92) | 20/20 · 20/20 up; 0/20 · 0/20; median 0.360 · 0.361; tail A1 1.05x | **Backed by Stage 1's records, not by the PR**: `stage1/runs/2026-11-08/75-M-H-10/eval/R2/{42,43}/walk/brisk.json` (checked: survived 20, succeeded 0, median `window.speed_m_s` 0.3603 / 0.3605, all 20 fail `drive_sustained`) |
| V0 animal pace, 0.496 m/s, 0 ms (93) | 20/20 · 20/20; 0/20 · 0/20; 0.506 · 0.490; tail 1.18-1.31x; duty 0.546 / 0.518; R2/43 ankle speed 1.17x | Counts backed by `.../walk/fr05.json`. The medians are slips: the records give 0.5054 / 0.4949. The review gives R2/43's duty as 0.5204. R2/43's other misses are left out (records: `speed` 11, `toe_clearance` 14, `drive_speed` 14, `no_flight` 4 of 20) |
| F25 + F9, duty 0.55, 0.496 m/s, 0 ms (94) | 20/20 · 20/20; 20/20 · 0/20; 0.511 · 0.510; tail 0.58x; ankle speed 0.74x; R2/43 duty 0.542 | **None** |
| `F25_F9_cl` at 0.400 / 0.450 / 0.496, 0 ms, mu 0.90 and 0.54 (95) | 20/20 · 20/20 twice; 0.415 / 0.466 / 0.514; "18-20/20" keep 0.25 m drift | **None** |
| `F25_F9_cl` at 0.550, 0 ms (96) | 20/20 · 20/20 twice; 0.570 · 0.568; hip pitch 0.966x of the 24 V speed limit | **None** |
| `F25_F9_cl` + "+397 g trunk mass", 0.496, 0 / 2 ms (97) | 20/20 · 20/20 twice; 0.515 · 0.516; "< 0.001 m/s" | **None.** No `plant.trunk_mass` exists; the only mass hook is `env.set_trunk_mass(m, trunk, factor, nominal, scratch)` (`designlab/stage1/env.py:317`), which scales mass and inertia by a factor and does not move the centre of mass. "+397 g" has no derivation anywhere |
| `F25_F9_cl` at 0.400 and 0.496, 2 ms (98) | 20/20 · 20/20 twice; 0.416 / 0.515; toe 22.8-23.6 mm | **None** |
| `F25_F9_cl` at 0.496, 4 ms against 6-10 ms (99) | 20·20 / 0·0 up; 5·1 / 0·0; 0.512; toe 18.9-19.3 mm; "0/20 at both 0.40 and 0.50" | **None.** It conflicts with the recorded probe, where the 75 cm walk at 0.40 m/s stays up 3/3 at 6 ms |
| "Stage 2 `stop2`" at 0.40 and 0.50, 2 ms (100) | 20/20 · 20/20 twice; 0.67 / 0.73 strides; 0.79 / 0.92 m/s² | **None.** No `stop2` and no two-step stop exists in code (`look.py:182` lists it as deferred). The rule quoted (1.5 strides, 1.5 m/s²) is Stage 1's |
| Walking nudge "~22 N", forward / left, 0.40 against 0.50, 2 ms (101) | 20·20 (15·15 left) to 20·17 (13·11 left); 18·15 to 19·13 (9·7 left); 0.416 / 0.515 | **None** |

Also unbacked outside §2.3: every "3/3" cell and the "0, 2, 4, 6, 10 ms" list in §2.1-2.2 beyond the one probe; the
"+17 g up to +397 g into `plant.trunk_mass`" claim (JPS:83); the interpolated and extrapolated ceilings 0.481,
0.531, about 0.48, about 0.65, 0.559, 0.651, about 0.645 and about 0.760 m/s (JPS:70-74); the `-X` note (JPS:79);
all of §3's timings.

## 10. Budget: `designlab/cost.py`, the PR's table, the catalog

### 10.1 `cost.estimate`

`estimate(design, rows: list[dict], selection: dict) -> dict` (`designlab/cost.py:36-99`). Callers:
`pipeline.py:202`, `refine.py:288` and `:648`, `sizestudy.py:121`.

- **Inputs.** `design`: `act(g)` for the six leg groups, `neck`, `tail` and any `head_x_groups` (`:37-41`);
  `sk.H`; optional `struct_conv_al`, `struct_al_equiv_extra`, `struct_mat`, `bracket_mat`, `cover_mat_override`,
  `power_mode`. `rows`: the model ledger (`mjcf.ledger(design)`, `designlab/mjcf.py:37`), each with `category` and
  `mass_kg`. `selection`: `compute`, `camera`, `tof`, `battery`, optional `lidar`, `bus`.
- **Return.** `dict(lines={name: [low, nominal, high]}, low, nominal, high, drivers={structure_kg, cover_cm3,
  frame_factor, cover_factor, integration_factor, bus_48v})`. Totals are rounded to whole dollars, lines to cents.
- **No self-assembled or turnkey split exists in `cost.py`.** `integration` is always a line and the total always
  includes it. The PR's "self-assembled" total is its own construction (below).

**The 16 lines** (`:71-94`), with A8's scaling (`DECISION_LOG.md:104`) and the 75 cm record's values
(`round2/refine/feet_75/candidates/75-M-H-10.json` `cost`):

| Line | Rule | 75-M-H-10 low / nominal / high |
|---|---|---|
| `leg_actuators` | sum of `2 * price` over six groups; high x 1.05 | 3546 / 3546 / 3723.30 |
| `expression_actuators` | neck + tail (+ head-X servos); high x 1.05 | 504.89 / 504.89 / 530.13 |
| `compute` | `price + 60 / + 120 / + 200` | 350 / 410 / 490 |
| `rgbd_camera` | `price + 15 / + 35 / + 80` | 284 / 304 / 349 |
| `tof` | `2 * price`, `+ 10`, `+ 30` | 39.90 / 49.90 / 69.90 |
| `lidar` | only if selected | absent |
| `battery` | `price + 30 / + 60 / + 100`; tether mode `(120, 180, 280)` | 137.19 / 167.19 / 207.19 |
| `control` | `REF["control"]` (125, 180, 270); x 1.3 on a 48 V bus | 125 / 180 / 270 |
| `foot_sensing` | `REF` (100, 180, 300) x `max(1, size ** 0.5)` | 113.44 / 204.19 / 340.31 |
| `power` | `REF["power24"]` (275, 420, 650); 48 V `(450, 650, 950)` | 275 / 420 / 650 |
| `harness` | `REF` (150, 240, 350) x `max(1, size)` (x 1.3 at 48 V) | 193.02 / 308.83 / 450.38 |
| `frame` | `REF` (1800, 2750, 4000) x `max(1, (structure_kg / 1.79) ** 0.6)` | 2947.58 / 4503.25 / 6550.19 |
| `covers` | `REF` (350, 600, 1050) x `max(0.6, cover_cm3 / 480) ** 0.85` | 1102.87 / 1890.63 / 3308.60 |
| `bench` | `(900, 1400, 2200)` if `H > 0.6` else `REF` (550, 850, 1300) | 550 / 850 / 1300 |
| `spares` | `2 * dearest leg drive + pack price + 150 / 350 / 650` | 1113.19 / 1313.19 / 1613.19 |
| `integration` | `REF` (1350, 2200, 3200) x `max(1, size ** 0.7)` | 1610.62 / 2624.71 / 3817.76 |
| `contingency` | 5.3 / 7.2 / 8.4 % of the sum of all lines above | 683.31 / 1243.93 / 1988.28 |
| **total** | | **13,576 / 18,521 / 25,658** |

- `size = design.sk.H / 0.39` (`:69`): 1.2868 at 75 cm. `structure` sums rows of category `structure`, `bearings`,
  `linkage`, `foot` (`:49`): 4.0723 kg. `cover_cm3` is the `cover` rows' mass over 1.01e-3 kg/cm³ (`:50`): 1852.08.
  Drivers at 75 cm: frame 1.6375, covers 3.1510, integration 1.1931.
- Low / nominal / high are three columns of every line; there is no other uncertainty model.
- Material hooks: a frame cost factor when `design.struct_conv_al > 0` (`:54-60`); a cover factor
  `cost_per_kg * density / (180 * 1010)` when `design.cover_mat_override` is set (`:61-67`), with
  `build.COVER_MATERIALS` (`designlab/build.py:104-114`: `PA12_MJF` 180 $/kg, `PA11_MJF` 225 $/kg at 1050 kg/m³).
- The ledger also carries per-item `cost_nom` and `cost_high` (`build.py:655-703`, `mjcf.py:33-63`; for example
  `power_distribution` 260, `compute_tray` 120, `control_stack` 260, `imu` 40). `cost.estimate` does not read them.

### 10.2 What `BUILD_BUDGET_75_JETSON.md` did with those lines

Its table is hand arithmetic on the record's `cost.lines`; no code produces it.

| Budget line | Low / nominal / high | Against the record |
|---|---|---|
| 1 Leg actuators | 3546 / 3546 / 3723.30 | equal |
| 2 Expression actuators | 504.89 / 504.89 / 530.13 | equal |
| 3 Compute and NVMe | 335 / 395 / 599 | hand-entered at $249. `cost.py` with the catalog's $399 gives 459 / 519 / 599 |
| 4 Partial PLEM bus and IMU | 175 / 498 / 560 | replaces `control` (125 / 180 / 270); hand-entered |
| 5 Vision, ToF, foot sensors | 437.34 / 568.09 / 759.21 | `rgbd_camera + tof + foot_sensing` = 437.34 / 558.09 / 759.21; nominal has +$10 for the VL53L8CX |
| 6 Character audio | 35 / 65 / 120 | new, unsourced |
| 7 Power bus, 12 V buck, P21 input | 515 / 740 / 1120 | `power` 275 / 420 / 650 plus 240 / 320 / 470 by hand |
| 8 Battery, cassette, dummy puck | 167.19 / 217.19 / 277.19 | `battery` plus 30 / 50 / 70 |
| 9 Harness | 193.02 / 308.83 / 450.38 | equal |
| 10 Frame | 2947.58 / 4503.25 / 6550.19 | equal |
| 11 PA11 skins | 1132.87 / 1950.63 / 3408.60 | `covers` plus 30 / 60 / 100 by hand (not `cover_mat_override`) |
| 12 Bench, charger, boom | 850 / 1450 / 2500 | `bench` plus 300 / 600 / 1200 |
| 13 Spares | 1113.19 / 1313.19 / 1613.19 | equal |
| Subtotal | 11,952.08 / 16,060.07 / 22,211.19 | |
| Contingency | 633.46 / 1,156.33 / 1,865.74 | 5.3 / 7.2 / 8.4 % of the subtotal |
| **"Self-assembled"** | **12,585.54 / 17,216.40 / 24,076.93** | the PR's construction: no `integration` line |
| 14 Integration | +1,695.98 / +2,813.68 / +4,138.45 | the record's `integration` x (1 + contingency) |
| **"Turnkey"** | **14,281.52 / 20,030.08 / 28,215.38** | `cost.py`'s structure: (subtotal + integration) x (1 + contingency) |

P4 (`DECISION_LOG.md:19`) is "$10-20 k prototype including outsourced fabrication, integration and spares", so the
turnkey column is the one to compare. Line 3 at the catalog price adds $124 nominal before contingency, about
$20,163 turnkey.

### 10.3 Catalog ids and prices

`designlab/catalog.py`: `Part` (`:94-111`) fields `id, category, vendor, model, dims (m), mass (kg), power_typ,
power_max, capacity_wh, voltage, price, fov, max_current, basis, sources, notes`; `Actuator` (`:35-57`);
`catalog.part(id)` (`:163`), `catalog.actuator(id)` (`:159`), both `lru_cache`d. JSON keys map at `:124-151`
(`price_usd` to `price`, `mass_kg` to `mass`, `dims_m` to `dims`, `voltage_v` to `voltage`). Both JSON files are
generated by `designlab/data/build_catalog.py` (`PART_MAP` `:124-145`; the Pi set by `_compute_from_survey`
`:187-206`) from `designlab/data/research/*.json`. Hand edits to the JSON are overwritten by a rebuild.

| Item | Catalog id | Price | Mass | Other | Where |
|---|---|---|---|---|---|
| Jetson Orin Nano Super dev kit | `nvidia_orin_nano_super_devkit` | **$399** | 0.174 kg | 103 x 90.5 x 34.77 mm; typ 15 W, max 25 W; `voltage_v` 19.0; category `compute` | `data/parts.json:6-36` (price `:22`) |
| Pi 5 8 GB + AI HAT+ 13 TOPS | `rpi5_8gb_ai_hat_plus_13tops` | $290 (175 + 5 + 70 + 40) | 0.100 kg (survey estimate) | 85 x 56 x 30 mm; typ 8.0 W, max 18 W; 5 V / 5 A | `data/parts.json:552-578` |
| PCAN-USB FD | `pcan_usb_fd` | $368 | 0.068 kg | 75 x 43 x 22 mm; typ 0.51 W, max 0.85 W; category `can_interface` | `data/parts.json:474-499` |
| 12 V 9 A regulator | `pololu_d42v110f12` | $69.95 | 0.015 kg | 60 V max in; 108 W; category `dcdc` | `data/parts.json:371-395` |
| 12 V 4.5 A regulator | `pololu_d36v50f12` | $39.95 | 0.007 kg | 50 V max in; 54 W | `data/parts.json:397-421` |
| STS3215, 12 V | `feetech_sts3215_c018` | $22 | 0.055 kg | `rated_torque_nm` 0.86195 (planning sustained 0.57 x = 0.491 N m); 27 rpm | `data/actuators.json:1529-1573` |
| XM540-W270 | `robotis_xm540_w270_t` | $482.89 | 0.165 kg | `rated_torque_nm` 3.1056; 18 rpm | `data/actuators.json:1443-1484` |
| X4-P36-36-E-L (10 leg drives) | `myactuator_rmd_x4_p36_36_e_l` | $269 | 0.36 kg | 24 V nominal | `data/actuators.json:89-127` |
| X6-P20-60-E (hip pitch) | `myactuator_rmd_x6_p20_60_e` | $428 | 0.82 kg | 48 V nominal, run at 24 V | `data/actuators.json:171-210` |
| STS3250 (60 cm neck) | `feetech_sts3250` | $89 | 0.0745 kg | | `data/actuators.json:1575` |
| ToF carrier (selected) | `st_vl53l5cx_carrier` | $19.95 | | typ 0.5 W | `data/parts.json:90-113` |
| ToF, newer | `pololu_vl53l8cx` | $24.95 | | | `data/parts.json:115-138` |
| Packs | `lipo_6s_3300`, `lipo_6s_5000` | $107.19, $136.99 | 0.506, 0.755 kg | 73.26, 111 Wh | `data/parts.json:193-242` |

- For a `dcdc` part, `Part.voltage` is the **maximum input** voltage, and the tests assert it
  (`tests/test_stage2_jetson_plem.py:92-95`).
- The $249 in the PR is the NVIDIA datasheet's price inside the catalog's source list (`data/parts.json:26`); the
  catalog's `price_usd` is the $399 reseller price (`:32`). `COMPUTE.md:74`: "the Jetson kit went from $249 to $399
  (the study's cost model already uses $399)". `compute_options.py:30`: `JETSON_USD = 399 + 120`.
- The two ids the PR cites, `feetech_sts3215_12v` and `robotis_xm540_w270`, do not exist.
- **P21 items are not catalog parts.** Not found in `parts.json`: the LTC4417 power path, the regen clamp, the
  E-stop, the dummy puck, the Mean Well RSP-750-24 supply, the cable, the boom, the audio parts, an NVMe drive, a
  Teensy, an IMU, a CANable, a U2D2. Their only prices are prose estimates in `round2/refine/MATERIALS.md`: "about
  $0.2-0.4k for the input, power path, clamp and puck, plus $0.3-1.5k for a boom" (`:253`); PA11 walls x 0.75 at
  75 cm +$30 nominal / +$60 high (`:93`, simulated on the targeted 75 cm body, not the pick); graded PA11
  +$81 / +$130 (`:137`, a research estimate on `60-N-H-10` / `75-M-T-10-bighead`). `MATERIALS.md` §6 gives no part
  prices for the supply or the cable. No refine case applies P21's materials to `75-M-H-10` itself: the material
  cases are `mat60_*` and `mat75_*` (`round2/refine/`), built on other bodies.
- The `-X` head: `stage2/x/X_COSTS.md:71` gives the record's nominal $18,521 to $18,596 (+$75) at 75 cm, and `:76`
  says the yaw stage's bearing "is in neither the mass nor the cost". The budget's "+$195 (2x STS3215 $44 +
  thin-section yaw bearing $75 + brackets/cables $76)" has no source there.

## 11. `designlab/power.py` and what a compute power row needs

Constants (`:21-32`): `ETA_GEAR 0.75`, `ETA_DRIVE 0.92`, `IDLE_QDD_W 1.4`, `IDLE_SERVO_W 0.5`, `SERVO_V 12.0`,
`ETA_DCDC 0.88`, `USABLE 0.75`, `USABLE_RANGE (0.65, 0.85)`, `TARGET_MIN 20.0`, `MIN_MIN 15.0`.

```python
COMPUTE_BUDGET = {  # W at the compute rail, before the DC-DC, by selected compute part      (power.py:34-40)
    "nvidia_orin_nano_super_devkit": {"jetson_module": 25.0, "carrier_storage_cooling": 6.0},
    "rpi5_8gb_ai_hat_plus_13tops": {"pi5_ai_hat_plus": 8.0, "usb_ssd": 1.5},
}
AUX_FIXED = {"mcu_comms": 4.0}                                                              # (power.py:41-43)
```

- `aux_power(design)` (`:73-81`): `loads = dict(COMPUTE_BUDGET[design.sel["compute"]], **AUX_FIXED,
  sensors=<sum of power_nom of items whose category is "sensors">)`; returns
  `dict(loads, load_w, input_w=load_w / ETA_DCDC)`.
- `walking_power(design, req, motor_series)` (`:84-101`) returns
  `dict(drives, drives_w, aux, total_w, peak_drives_w, flags)`; `runtime(pack, total_w)` (`:104-107`) returns
  `dict(pack, energy_wh, minutes, minutes_range{"0.65","0.75","0.85"})`; `select_battery(bus, total_w, peak_w)`
  (`:110-126`).

So a compute power row is: **a key in `COMPUTE_BUDGET` equal to the selected compute part's catalog id, mapping
load names to watts at the compute rail before the converter.** Consequences:

1. A part id with no entry raises `KeyError` at `:79`.
2. The dev kit's entry today is 31 W (A13, round one's allowance; `DECISION_LOG.md:109`). A refine case that simply
   selects the dev kit gets `(31 + 4 + 4.4) / 0.88 = 44.8 W` of auxiliaries, not the PR's 26-38 W.
3. There is one key per part: no power-mode dimension (15 W against 25 W), and one efficiency (0.88) for every
   rail. The PR's 0.93 for a 12 V rail has no place in the model.
4. `AUX_FIXED["mcu_comms"]` 4.0 W is always added. A stack without the Teensy cannot be expressed.
5. Changing the dev kit's entry changes round one's reproduction (A13) and `compute_options.JETSON_W`
   (`designlab/compute_options.py:28`), which feeds `round1/compute_options.json` and `COMPUTE.md`'s tables.
6. The part's own `power_typ_w` / `power_max_w` (15 / 25 for the dev kit) reach only the ledger item
   (`build.py:690`); `power.py` does not read them.
7. `power.py` is not one of the Stage 1 hashed modules, but its output is stored in every round-two record
   (`power`, `runtime`), so a changed constant makes re-sized records differ from committed ones.

`COMPUTE.md` on the Jetson's draw: about 5-6 W idle, 10-12 W with a mid-size detector, at most about 15 W (ceiling
near 18 W) in the 15 W mode, labelled estimates (`:19-25`); architecture A "15 / 18 W", 174 g, $439 (`:113`); bench
test 1 measures the current dev kit in both modes with no purchase (`:150-151`); A14's 9.5 W for the Pi set is
unmeasured (`:176-183`).

## 12. ToF speed governor: where the geometry and timing assumptions live

**Geometry (code).**
- Mount: `designlab/build.py:707-718`. Two sensors under the chest at trunk station `tx = 0.50 * B`,
  `y = +-0.030 * B`, on the ventral line + 14 mm, aimed 45° down and each 12° outward
  (`tR = sk.R_p.T @ rz(radians(12 * sgn)) @ ry(radians(-45))`); site names `tof_l`, `tof_r`; box 8 x 20 x 20 mm.
- Field of view: `designlab/sensors.py:129-132`, `SPECS["st_vl53l5cx_carrier"]` and `["pololu_vl53l8cx"]`:
  `hfov 45, vfov 45, dfov 65, zones (8, 8), range_m (0.02, 4.0)`; equidistant projection (docstring `:28-32`).
- The floor window: `_Screen.tof` (`sensors.py:704-749`) casts the 64 **zone-centre** rays at the **home pose** on
  a flat floor; `summary.floor_ahead_of_toes_m = [min, max]` over valid floor zones (`:731`), copied to
  `summary.tof.<name>.floor_ahead_of_toes_m` (`:1050-1053`). "Ahead of toes" is from the largest x of any foot geom
  at home. `NEAR_STRIP = 0.8` m (`:148`).
- Recorded (`round2/refine/<case>/review/screening/<body>_sensors.json`, `tof.tof_l.summary`):

| Body | site_pos_m | axis depression | floor_range_m | floor_ahead_of_toes_m | foot path covered (union) |
|---|---|---|---|---|---|
| 60-N-T-10 | [0.2273, 0.0121, 0.3546] | 45.0° | [0.3927, 0.9053] | [0.25, 0.9428] | 0.683 |
| 75-M-H-10 | [0.2851, 0.0151, 0.4272] | 45.0° | [0.4731, 1.0907] | [0.2754, 1.11] | 0.648 |

- Braking: `designlab/envelope.py:106-112`, `stopping(M, q_home)`:
  `max_one_foot_decel_ms2 = G * pad_front_reach / com_height`; `STOP_DECEL = 1.5` (`:47`). At 75 cm: com height
  0.4359 m, pad front reach 0.0628 m, ratio 1.061, `one_foot: false`.
- The screen's own limits (`sensors.py:100-106`): geometry only; no noise, multipath or ambient light; flat floor;
  windows assumed. The window is for the floor, not for a 0.10 m obstacle, and it does not move with the gait's
  trunk pitch or the crouch.

**Timing.**
- **No ToF frame rate is sourced anywhere.** The only mention outside the PR is `data/parts.json:112`: "A 15 Hz
  maximum at 8x8 appeared only in a search snippet and is unverified." Not found in
  `designlab/data/research/parts_2026-11-07.json`, in `sensors.py`, or in `SCREENING.md`.
- `COMPUTE.md:59-62`: the real-time layer, the ToF stop included, stays on the microcontroller. `:91`: "CAN loop,
  IMU, ToF stop, watchdogs | 200-1000 Hz | Deterministic | Microcontroller". `:90`: locomotion policy 50-100 Hz,
  "Trained with an 8 ms command delay". `:88`: planner 10-15 Hz, at most 200 ms from the oldest sample to a motion
  request. The 14 Nov note (`:190-197`) records the conflict with Partial PLEM.
- `BEHAVIOR_SPEC.md:50`: no contact with static obstacles at least 0.10 m tall. `:56`: detection at 10 Hz or more
  and at most 200 ms to a motion request; "Results do not transfer between computers without re-timing". `:52`:
  heartbeat loss ends in a controlled stop within 1.0 s. `:51`: operator stop within 0.3 s. `:19`: stop within 1.5
  strides, peak horizontal trunk acceleration at most 1.5 m/s².
- The PR's timings (section 8) are literals in `jetson_plem_study.py:629-632`; its 200 ms for the Pi is the spec's
  deadline, not a measured latency, and it routes the ToF stop through the host, which `COMPUTE.md` does not.
- Review task 10 (`JETSON_PLEM_REVIEW.md:268-272`) asks for a governor table per stop path and per ToF mode from
  sourced timings; A17 (`DECISION_LOG.md:113`) is the bench row.

## 13. Tests that pin this subsystem (`tests/test_stage2_jetson_plem.py`)

The module imports `jetson_plem_study` (so the script must stay importable) and loads the JSON at import. Nine tests
with the parametrisation; none recomputes a number.

| Test | Asserts |
|---|---|
| `test_study_record_matches_committed_script` (`:46`) | JSON `script_sha256` equals the script's sha256; `bodies` keys are the two bodies; `PAIRS` order |
| `test_baseline_reproduces_round2_records_and_stage1_hashes[body]` (`:54`) | `xml_matches_record`, `xml_matches_stage1_plan`, `all_match` true; `max_rel_diff <= 1e-9`; `n_checks >= 10` |
| `test_jetson_packaging_dropin_and_60cm_pdb_fix` (`:64`) | 75 cm: 0 overlaps, torso and max width equal to baseline within 0.05 mm, `added_mass_over_pi5_g` 78.7 ± 1.5. 60 cm drop-in: at least 1 overlap, `min_pdb_gap_mm` -6.95 ± 0.25, widths unchanged. Fix: `pdb_length_mm` 94.0, 0 overlaps, gap at least 8.5 mm |
| `test_power_regulators_and_runtime_margins[body]` (`:86`) | regulator ids; `catalog.part(...).voltage` is 50.0 and 60.0 and "12V" is in each model; Partial PLEM draws less and runs longer than the Teensy stack; every stack meets 20 min and has at least 28.0 min on the 3300 mAh pack |
| `test_speed_tuning_f9_friction_and_stopping_bounds[body]` (`:105`) | at 0.40 m/s the tuned row has `gates_ok` and a lower `worst_planning_ratio_24v`, `mu_req_max` and `tail_cont_ratio_total` than standard; probe `2ms` survived at least `10ms` survived; camera near floor above 2.30 m; usable ToF budget above 0.80 m; Jetson safe speeds above the Pi's |
| `test_reports_cite_study_artifacts` (`:128`) | JPS contains "jetson_plem_study.json", "jetson_plem_study.py", "BUILD_BUDGET_75_JETSON.md"; the budget file contains "75-M-H-10.json", "pololu_d42v110f12", "PCAN-USB FD"; `COMPUTE.md` contains both report paths |

Related pins elsewhere: `tests/test_stage2_x_costs.py:72` compares the `-X` report's nominal costs with the records'
`cost.nominal`; `tests/test_stage2_screen.py:177-200` and `tests/test_stage2_look.py:318-331` read
`tail_phase.json`. No test calls `cost.estimate` or `power.aux_power` directly: not found.

## 14. Hard-coded assumptions the build list would break

| # | Assumption | Where | Breaks |
|---|---|---|---|
| 1 | Transitions exist only for names in `TRANSITIONS`, between gaits from `refs.walk_specs` | `stage1/transitions.py:120-131`, `:165-179`, `:641-642` | (A): F25 and look specs have no name; the PR hand-copied `generate` |
| 2 | `refs.generate` accepts only `nominal, brisk, fr05, carpet` and the turn cases | `stage1/refs.py:80-91`, `:118-133` | (A), (E): no saved, hashed F25 reference can be made through it |
| 3 | `transitions.Gait` has no look fields and no `tail_phase`; turn gaits always use `sizing.NOMINAL` | `stage1/transitions.py:143-158`, `:175-179` | (A), (B) |
| 4 | A stop is recognised by `TRANSITIONS.get(tr.name)[1] == "stand"` | `stage1/control.py:994` | (A), (C): a stop with a new name registers no stop event |
| 5 | `omega0` defaults to crouch 0.03 H and every evaluator uses the default on `refs/nominal.npz` | `stage1/control.py:943`; `panels.py:996`, `:2173`; `tune.py:805`; `speedfit.py:432`; `env.py:380`; `metrics.py:432`; `gates.py:863` | (E), (F), (G): F25's crouch is 0.05 H (0.07 H at 60 cm); the PR's probe used the walk's own crouch |
| 6 | `_tuned_f9_spec` picks its constants by `body.startswith("60")` | `jetson_plem_study.py:373-376` | `-X`, `-K` and any new body name |
| 7 | The tuned spec exists only at speeds the caller passes; duty steps at 0.36 m/s; `F25_F9_cl` is text | `jetson_plem_study.py:374`; JPS:88 | (A), (D), (E) |
| 8 | The tail is flipped by sign, leaving the spec not look-active | `jetson_plem_study.py:383`; `gait.py:165-167`, `:193` | (D): a `tail_phase` rung goes through `_generate_look` instead |
| 9 | The probe ran 0, 2, 6, 10 ms, 3 episodes, seeds 4100-4102, one controller per body, survival only | `jetson_plem_study.py:583-599` | (E) needs 0, 2, 4, 6, 8, 10 ms, 20 episodes, R2/42 and R2/43, both grips |
| 10 | 75 cm has no `speedg5.json`; `tuned_gains` returns seed 43 only | `jetson_plem_study.py:490-498`; `stage1/env.py:182-212` | (E), (G) |
| 11 | `screen.passive_loads` reads `neck_pitch` and `tail_yaw` only | `stage2/screen.py:312` | (D), (F) on `-X` bodies |
| 12 | `power.COMPUTE_BUDGET` is keyed by part id; the dev kit is 31 W; one converter efficiency; the MCU's 4 W is fixed | `power.py:34-43`, `:79-81` | the Jetson body's compute power row |
| 13 | The power box sits 10 mm behind the compute tray; a 24 V box is 110 mm long | `build.py:665-673` | a Jetson body at 60 cm (-6.95 mm); the 94 mm fix is a monkeypatch |
| 14 | The control stack (Teensy) is always built, 0.18 kg up to 75 cm | `build.py:695-701` | a body without it, or with an adapter and regulator placed |
| 15 | `cost.py` has no line for the regulator, the CAN adapter, audio or P21's power path and boom; `bench` switches on `H > 0.6`; `control` and `power` are fixed lots | `cost.py:79-87` | the budget rebuild |
| 16 | The script's own sha256 pins the JSON | `tests/test_stage2_jetson_plem.py:46-51` | any edit to the PR's script |
| 17 | Three different stride rules for "the walk at speed v" | `jetson_plem_study.py:365`, `:377`, `:611` | a governor that should use the reference's own stride |
| 18 | The ToF window is zone centres at the home pose; the frame rate is unsourced | `sensors.py:704-749`; `data/parts.json:112` | the ToF governor |

## 15. Extension points and hazards (short)

- **Do not edit `stage2/jetson_plem/jetson_plem_study.py`.** Importing `_tuned_f9_spec` from it is safe (the test
  module does); note that the import sets thread and `MUJOCO_GL` environment defaults (`:40-42`). A new module can
  hold the F25 specs and cite the PR's function by path and sha256.
- **F25 rungs are new GaitSpecs, not new generator behaviour.** With the sign flip they run through today's
  `gait.generate` unchanged. Choosing a `tail_phase` from `tail_phase.json`'s V0 sweep changes the generator path;
  decide and label which is used, and note that no tail sweep exists on the F25 walk itself.
- **The official builders need a way in for an arbitrary spec** (items 1-4 above); those modules are hashed by
  `train.py` and their own records, so the entry has to be an option that defaults to today's behaviour, or a new
  module. Item 5 (omega) is the least visible mismatch between the PR's probe and an official panel.
- **Label what was tuned after results.** `F25_F9_cl`'s duty 0.59 and lift 0.095 H were set after the evaluation
  seeds were seen (`JETSON_PLEM_REVIEW.md:155-156`), and that walk has no Stage 0 screen.
- **Jetson body.** The head-X case is the precedent: `refine.HEAD_X_BASE`, `head_x_candidate`, `_evaluate`
  (`designlab/refine.py:146`, `:250-285`) build once from the base record without re-sizing, and Stage 1 registers
  the body in `stage1.BODIES`, `CANDIDATE_IDS`, `SHA256_PREFIX` (`designlab/stage1/__init__.py:47-71`).
  `designlab/stage1/__init__.py` is inside `train.py`'s hashed set.
- **Budget rebuild.** `BUILD_BUDGET_75_JETSON.md` must keep the three strings the test reads, or the rebuild lives
  in a new file. The catalog JSON is generated; new parts go through `build_catalog.py` and a research record.
- **Records to cite, never rerun**: `jetson_plem_study.json` (sha-pinned to its script), the round-two screening
  records it read, `stage1/runs/2026-11-08/75-M-H-10/eval/R2/{42,43}/walk/{brisk,fr05}.json` for JPS §2.3's two V0
  rows, and `round2/gait_tune/75-M-H-10.json` for F25's own walk (`chosen.spec`: Froude 0.05, lift 0.08, duty 0.55,
  stride_rel 0.8625, crouch 0.05, sway_gain 1.0).
