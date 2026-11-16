# Map 01: the gait generator and the look ladder

Study root: `/home/user/mesozoic-labs/hardware/trex_form_study` (all paths below are relative to it).
Read for this map: `designlab/gait.py` (1065 lines), `designlab/stage2/look.py` (1625), `tests/test_stage2_look.py` (906),
`designlab/gaittune.py` (272), `round2/gait_tune/GAIT_TUNE.md` and its three JSON records, `stage2/look/look_refs.json`
and `tail_phase.json` (structure). Read in part, because the ladder depends on them: `designlab/stage1/refs.py`,
`designlab/stage2/screen.py`, `designlab/turning.py:364-423`, `designlab/stage1/transitions.py:165-179`,
`designlab/stage1/panels.py:1503-1583`, `tools/restore_inputs.py:311-388, 669-687`,
`stage2/jetson_plem/jetson_plem_study.py:347-390`.

Checked by running (short, read-only): the field lists, `look_active` on F25 specs, the v040 specs, array shapes on a
16-frame walk of `75-M-H-10`, `panels.panel_sources()`. Nothing else was run; no pytest.

Line numbers are of the checkout at commit `7e9ea0d`.

---

## 1. `designlab/gait.py`

### 1.1 `GaitSpec` (gait.py:97-146): every field

A plain (not frozen) `@dataclass`. 38 fields: 11 old ones, then 27 look fields. Field order is pinned by a test
(section 7).

| # | Field | Default | Unit / meaning | Look field |
|---|---|---|---|---|
| 1 | `froude` | 0.012 | Fr = v^2 / (g H); speed = sqrt(froude G H) | no |
| 2 | `duty` | 0.60 | stance fraction of the stride | no |
| 3 | `lift_frac` | 0.10 | swing lift / H | no |
| 4 | `head_amp` | 0.05 | rad, neck pitch at step frequency | no |
| 5 | `tail_amp` | 0.12 | rad, tail yaw at stride frequency | no |
| 6 | `sway_gain` | 1.0 | multiplies the LIPM sway (y only) | no |
| 7 | `stride_rel` | 0.55 | stride / H | no |
| 8 | `crouch` | 0.03 | hip height reduction, fraction of H | no |
| 9 | `zmp_iters` | 6 | multibody-ZMP correction passes | no |
| 10 | `zmp_gain` | 0.5 | correction gain per pass | no |
| 11 | `n` | 160 | samples per stride | no |
| 12 | `bob` | 0.0 | F1: pelvis dip, fraction of H, peak to peak | yes |
| 13 | `bob_phase` | 0.0 | F1: stride fraction | yes |
| 14 | `bob_skew` | None | F2: trough after touchdown, fraction of stance | yes |
| 15 | `swing_time` | None | F3: s; duty then follows (see `describe`) | yes |
| 16 | `lift_peak` | 0.5 | F4: fraction of the swing at peak lift | yes |
| 17 | `toe_off` | 0.0 | F5: rad, peak heel-up pitch | yes |
| 18 | `toe_off_start` | 0.825 | F5: fraction of stance | yes |
| 19 | `toe_off_margin` | 0.005 | F5: m | yes |
| 20 | `toe_reach` | 0.0 | F5: rad, toes-up before landing | yes |
| 21 | `roll_amp` | 0.0 | F6: rad, toward the stance leg | yes |
| 22 | `roll_phase` | 0.0 | F6: stride fraction | yes |
| 23 | `zmp_inset_y` | 0.0 | F7: m | yes |
| 24 | `tail_phase` | pi/2 | F9: tail = tail_amp sin(2 pi phase + tail_phase) | yes |
| 25 | `head_mode` | "nod" | F10: "nod" or "stabilise" | yes |
| 26 | `head_gain` | 0.9 | F10 | yes |
| 27 | `head_settle` | 0.0 | F10: rad | yes |
| 28 | `settle_lag` | 0.075 | F10: s | yes |
| 29 | `strides` | 1 | F14: strides in the periodic window | yes |
| 30 | `phase_jitter` | 0.0 | F14: fraction | yes |
| 31 | `asym` | 0.0 | F14: fraction | yes |
| 32 | `gaze_var` | 0.0 | F14: rad | yes |
| 33 | `tail_var` | 0.0 | F14: fraction of tail_amp | yes |
| 34 | `var_band` | (0.05, 0.2) | F14: Hz (a tuple; JSON turns it into a list) | yes |
| 35 | `noise_seed` | 0 | F14 | yes |
| 36 | `bob_drop` | None | F2: s | yes |
| 37 | `toe_off_rate` | None | F5: rad/s cap | yes |
| 38 | `toe_off_accel` | None | F5: rad/s^2 cap | yes |

* `LOOK_FIELDS` (gait.py:150-153) lists the 27 look names; `LOOK_DEFAULTS` (154) is built from the dataclass defaults.
* `look_active(spec) -> bool` (165-167): true if any look field differs from its default. `_same_value` (157-162)
  lets a list or array equal the tuple default of `var_band`, so a default spec read back from JSON stays today's walk.
* `tail_amp`, `head_amp`, `crouch`, `lift_frac`, `duty`, `stride_rel`, `zmp_iters`, `n` are NOT look fields. Changing
  them keeps the default code path. Checked: `GaitSpec(froude=0.03, tail_amp=-0.125)` gives `look_active == False`;
  `tail_phase=-pi/2` gives True.
* F8, F11, F12, F13 are not fields (docstring, gait.py:57-58): F13 is `stride_rel`; F11 and F12 "belong to the turn and
  transition generators".
* A new field that is not added to `LOOK_FIELDS` is silently ignored by both paths and does not trigger the look path.

`describe(H) -> dict` (139-146): keys `speed` (m/s), `stride` (m), `period` (s), `froude`, `duty`, `stride_rel`,
`cadence_hz` (= 2 v / stride, steps per second). With `swing_time` set:

```python
duty = float(np.clip(1.0 - self.swing_time / (stride / v), *DUTY_RANGE))   # DUTY_RANGE = (0.55, 0.72), gait.py:74
```

So the effective duty is `describe(H)["duty"]`, not `spec.duty`, whenever `swing_time` is set. `speed` and `period` come
back as numpy floats.

### 1.2 Accuracy (module state)

```python
ACCURACY = "r1"; EXACT_UNWRAP = False; COM_TOL = 2e-4; COM_ITERS = 4          # gait.py:83-86
_ACCURACY_LEVELS = {"r1": (False, 2e-4, 4), "r2": (True, 1e-6, 12)}            # gait.py:87
def set_accuracy(level: str = "r1") -> None                                    # gait.py:90-94
```

* "r2": exact stride-boundary unwrap in `derivatives`, per-frame CoM tolerance 1e-6 m, up to 12 CoM passes per frame.
* It is process-wide state. `info["accuracy"]` records the level a trajectory was made at (gait.py:296, 1043).
* Who sets it: `stage1.refs.generate` and `refs.prepare` (refs.py:115, 310; not restored), `stage1.transitions`
  (transitions.py:638), `gaittune.Body.__init__` (gaittune.py:63; not restored), `look.generate` and `look.price`
  through the `look.accuracy()` context manager, which restores the previous level (look.py:281-289, 475, 897).
  Observed: after `look.source()` and `look.generate()` the process is still at "r1".

### 1.3 `generate(M, q_home, H, spec)` (gait.py:182-297): the default path

`if look_active(spec): return _generate_look(...)` (193-194); otherwise the code that has always run.

1. `info = spec.describe(H)`; `n = spec.n`; `dt = T / n`; `t = arange(n) * dt`.
2. Home pose: foot origins `feet0`, pad centroids, `com0`, `root0 = q_home[:3]`.
3. Footprints and ZMP reference (216-238). Phase `ph = t / T mod 1`; left foot offset 0.0, right 0.5 (219). A foot
   swings while `u < 1 - duty`, so the left foot lifts at phase 0 and lands at phase `1 - duty`. Swing x is min-jerk
   (`_foot_x`, 174-179). ZMP reference: the stance pad centroid in single support, a linear hand-over in double support.
4. LIPM: `z_c = com0[2] - spec.crouch * H`, `omega = sqrt(G / z_c)` (239-240). The CoM path is solved harmonic by
   harmonic with an FFT; y is multiplied by `sway_gain` (250-251).
5. Per frame (`solve_frames`, 249-280): swing height `lift * sin(pi u / sw) ** 2` (260); pelvis height
   `root0[2] - crouch * H` (262); trunk orientation `q_home[3:7]`; then

   ```python
   q[m.jnt_qposadr[neck]] = spec.head_amp * np.sin(2 * w * tk)              # gait.py:264
   q[m.jnt_qposadr[tail]] = spec.tail_amp * np.sin(w * tk + np.pi / 2)       # gait.py:265
   ```

   then up to `COM_ITERS` passes of `statics.leg_ik(M, q, targets, iters=120)` with the pelvis shifted until the model's
   CoM is within `COM_TOL` of the LIPM path (268-275).
6. Multibody-ZMP correction (282-292): `for it in range(spec.zmp_iters + 1)`: solve the frames, `derivatives`,
   `statics.total_zmp` per frame, `e = zmb - zmp`, append `max |e|` (m) to the history, then
   `zref = zref - spec.zmp_gain * np.nan_to_num(e)`. So 6 corrections means 7 solves and a 7-entry history.

Returned trajectory dict (294-297):

| Key | Shape / type | Meaning |
|---|---|---|
| `t` | (n,) | s, `k * dt` |
| `q` | (n, nq) | qpos: root x y z, root quaternion, hinges. nq 21 on the 14-joint bodies |
| `phases` | list of n tuples | the stance sides, e.g. `("right",)` or `("left", "right")` |
| `dt` | float | s |
| `zmp_ref`, `com_ref`, `zmp_mb` | (n, 2) | world x y, m |
| `info` | dict | `describe`'s keys plus `sway_amp`, `lift`, `omega`, `zmp_error_history_m`, `accuracy` |
| `ik_ok` | bool | every frame's IK converged |
| `ik_fail_frames` | list[int] | |

Hard-coded in this path: joints named `neck_pitch` and `tail_yaw` must exist (244-245); every other non-leg joint
stays at its `q_home` value, so the generator itself does not depend on the joint count; straight walk along +x on a
flat floor; left foot lifts first; no check that `duty > 0.5` (the look path checks, 862-863).

### 1.4 `_generate_look` (gait.py:856-1065)

Same construction, vectorised, with the features added. Differences a builder must know:

* Uses `info["duty"]` (so F3 works), `K = spec.strides`, `n = spec.n * K` frames, window `W = K * T` (861-873).
* Refuses `duty` outside (0.5, 1), an unknown `head_mode`, `strides < 1` (862-868).
* Seeds: `np.random.SeedSequence(spec.noise_seed).spawn(3)` gives the step, gaze and tail generators (882).
* F5 raises `ValueError` if the ZMP reference never enters the rolling support (925-927), if a foot rolls in single
  support (1035-1036), or if the reference comes within `toe_off_margin` of the rolling support's edge (1039-1041).
* Tail: `tail_amp * (1 + tail_var * g_tail) * sin(2 pi phi + tail_phase)` (969). Head: nod, or stabilise with the
  settle dip (972-981).
* Pelvis z gets the bob (1001); trunk quaternion gets the roll (966, 1002).
* `derivatives(M, qs, dt, stride=K * stride)` (1025).
* Extra `info` keys (1043-1052): `strides`, `window_s`, `swing_time`, `look` (the 27 look values), `bob_m`,
  `bob_trough_step`, `bob_drop_step`, `step_deviations`, `gaze_freqs_hz`, `tail_freqs_hz`, `rolling_entry`,
  `zmp_margin_min_m`, `zmp_margin_ref_min_m`.
* Extra top-level key `look` (1053-1063), a dict: `phase` (n,), `swing` (n,2) bool, `swing_u` (n,2), `foot_pitch`
  (n,2), `rolling` (n,2) bool, `rolling_contact` (n,2,2,3), `on_claw` (n,2), `foot_base` and `foot_target` (n,2,3),
  `bob`, `trunk_roll` (n,), `trunk_quat` (n,4), `neck`, `tail`, `head_settle`, `gaze_var`, `tail_var`, `zmp_margin`,
  `zmp_ref_margin` (n,), `support_hull` (list of arrays), `touchdowns` (list of s), `rolls` (per side, list of dicts),
  `geometry` (`foot_geometry`), `warp` (`_Warp`). Column order of every (n,2,...) array is left, right.

The look path at today's values is not the default path bit for bit. `screen.json`'s `look_path_control` records the
difference for `tail_phase = pi/2 + 2 pi`: joints differ by 2.2e-15 to 2.8e-15 rad. So a default-equivalent walk must
go through the default path to keep its sha256.

### 1.5 Other signatures

```python
derivatives(M, qs, dt, stride=None) -> (qd, qdd)          # gait.py:300; central differences, periodic; exact unwrap
                                                          # only when EXACT_UNWRAP and stride is given
inverse_dynamics(M, traj) -> dict                         # gait.py:335; keys tau (n, na), omega (n, na),
                                                          # cop_margin (n,2), friction (n,), normal (n,2), qd, qdd
summarize(M, traj, idr) -> dict                           # gait.py:365
lift_profile(s, lift_peak=0.5)                            # gait.py:426 (F4)
bob_profile(tau, trough=0.5, drop=None)                   # gait.py:435 (F1/F2)
bob_trough(spec, duty) -> float                           # gait.py:456
toe_off_bump(t, t_r, t_lo, t_e, amp) -> (pitch, t_peak)   # gait.py:471
roll_amplitude(spec, t_r, t_lo) -> float                  # gait.py:490; MIN_JERK_ACC at 487
foot_geometry(M) -> {side: dict}                          # gait.py:538; call at home with feet flat
roll_offset(gs, th) -> (offset, contact(2,3), part)       # gait.py:670; part in "edge", "arc", "claw", "heel"
rolling_entry(geo, pad_c, inset, feet0, stride, margin)   # gait.py:692
class _Warp(T, K, e=None): time(phi), rate(phi), phase(t) # gait.py:735
step_deviations(spec, rng) -> ndarray(2 K)                # gait.py:781
slow_variation(rng, window_s, band, t) -> (g, freqs)      # gait.py:794
support_margins(M, qs, phases, zmb, zmp, rolling=None, contact=None) -> (margin, margin_ref, hulls)   # gait.py:825
```

`inverse_dynamics` reads `info["stride"]`, `info.get("yaw_rate")` and `info.get("strides", 1)` to choose the unwrap
(339-341): a trajectory with a yaw rate gets no exact unwrap from this function.

gait.py has no command line.

### 1.6 The named walks

| Name | Definition | 60-N-T-10 (H 0.40469 m) | 75-M-H-10 (H 0.50185 m) |
|---|---|---|---|
| `sizing.NOMINAL` (sizing.py:58) | Fr 0.012, stride 0.55 H, duty 0.60, lift 0.10 | 0.2183 m/s, T 1.0198 s | 0.2431 m/s, T 1.1356 s |
| `sizing.BRISK` (sizing.py:59) | Fr 0.025, stride 0.62 H, duty 0.58, lift 0.11 | 0.3150 m/s, T 0.7964 s | 0.3508 m/s, T 0.8869 s |
| `envelope.FAST` (envelope.py:46) | Fr 0.05, stride 0.80 H, duty 0.56, lift 0.11 | 0.4455 m/s, T 0.7267 s | 0.4961 m/s, T 0.8092 s |

All three keep crouch 0.03, head_amp 0.05, tail_amp 0.12, zmp_iters 6, n 160. Exact hip heights, as the tests pin
them (`H_BODY`, test_stage2_look.py:53): 0.40468919988052376 m and 0.5018535496646683 m.

Frame time is `T / n`, so it changes with the walk: on `75-M-H-10`, 7.10 ms nominal, 5.38 ms at 0.40 m/s, 5.06 ms at
the animal pace.

---

## 2. Stage 1's reference files (the interface the ladder copies)

`designlab/stage1/refs.py` (another map covers it; only what the ladder uses is given here).

* `walk_specs(body)` (refs.py:80-82): exactly `nominal`, `brisk`, `fr05`, `carpet`. `generate(src, name)` (113-134)
  takes a name, not a spec.
* `source(body) -> dict` (97-110): keys `body`, `M`, `q_home`, `design`, `H`, `layout`, `record`, `model`. The model
  is rebuilt from the record and checked against its `xml_sha256`.
* `arrays(src, g)` (145-166), the npz keys of a Stage 1 reference: `t, q, qd, qdd, tau, omega, support, com_ref,
  zmp_ref, zmp_mb, cop_margin, normal, friction, feet, com, q_home, joints, dt, stride_motion`; turns add `path,
  heading, foot_target, foot_yaw, swing`.
* `Reference.joint_series` reads `arrays["tau"]` (refs.py:364-366).

**A look rung's npz has no `tau`, `omega`, `cop_margin`, `normal` or `friction`** (section 3.5). It cannot be loaded as
a Stage 1 `Reference` for the tracker's feedforward as it is.

---

## 3. `designlab/stage2/look.py`

### 3.1 Constants

```python
ACCURACY = "r2"                                        # look.py:144
BODIES = ("60-N-T-10", "75-M-H-10")                    # look.py:145
SPEEDS = ("nominal", "v040", "fr05")                   # look.py:146
AGILITY_SPEED = 0.40                                   # look.py:148
VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5", "V6")  # look.py:149
EXTRAS = ("V1B50", "V1B100", "V3T", "V4L", "V5F13", "V6S")   # look.py:150
CLIP_S = 10.0; BOB_SHARE = 0.75; BOB_DROP_S = 0.10; F13_STRIDE = 0.875
LOOK_ZMP_ITERS = 12                                    # look.py:168
COP_RULE_M = -0.010                                    # look.py:272
POINTS = (13 marker names)                             # look.py:277
```

`VALUES` (191-267), `LIMIT_RULE` (269), `DEFERRED` (179-189) and `CLAW_QUESTION` (170-176) are text that is copied
into `look_refs.json`; two tests compare the file with the module (section 7).

`DEFERRED` has keys `turns`, `starts_and_stops`, `expressive_head`, `why`. Its `why` states the blocker for build
items (A) and (B):

> "the turn and transition generators (turning.generate; stage1.transitions.gait_of and its plans) read spec.duty and
> today's head nod and tail law, so a look spec would silently give today's walk; turning.generate now refuses a look
> spec (gait.look_active) and transitions.gait_of takes only refs.walk_specs' walks."

### 3.2 How 0.40 m/s is defined: `speed_spec` (look.py:295-304)

```python
def speed_spec(H, v=AGILITY_SPEED, lo=sizing.BRISK, hi=envelope.FAST) -> gait.GaitSpec:
    a, b = lo.describe(H), hi.describe(H)
    w = (v - a["speed"]) / (b["speed"] - a["speed"])
    if not 0.0 <= w <= 1.0: raise ValueError(...)
    T = a["period"] + w * (b["period"] - a["period"])
    return replace(lo, froude=v * v / (G * H), stride_rel=v * T / H, duty=lo.duty + w * (hi.duty - lo.duty),
                   lift_frac=lo.lift_frac + w * (hi.lift_frac - lo.lift_frac))
```

* It is the gait Stage 1's `brisk_to_fr05` speed change passes through at 0.40 m/s: period, duty and lift mixed with
  the speed's weight.
* Only `froude`, `stride_rel`, `duty`, `lift_frac` are set. `crouch`, `tail_amp`, `head_amp`, `zmp_iters` and every
  look field are copied from `lo`. It reads `lo.duty` and `hi.duty`, not `describe()["duty"]`.
* It raises outside `lo`'s to `hi`'s speeds: 0.3150 to 0.4455 m/s at 60 cm, 0.3508 to 0.4961 m/s at 75 cm.
* The result is not a look spec (`look_active` False), so it takes the default path.

| Body | w | froude | stride_rel | duty | lift | T | stride |
|---|---|---|---|---|---|---|---|
| 60-N-T-10 | 0.651 | 0.040302 | 0.742303 | 0.566979 | 0.11 | 0.75101 s | 0.30040 m |
| 75-M-H-10 | 0.338 | 0.032499 | 0.685946 | 0.573232 | 0.11 | 0.86061 s | 0.34424 m |

`base_spec(H, speed)` (307-315) returns `sizing.NOMINAL`, `envelope.FAST` or `speed_spec(H)`; any other speed key
raises `KeyError`.

### 3.3 `feature_values` and `ladder`

`feature_values(body, H, speed, tail_phase=None) -> dict` (334-352), keyed by feature group:

| Key | Fields it sets |
|---|---|
| `F1F2` | `bob = 0.75 * enforced_shortening(V0)`, `bob_skew=0.125`, `bob_drop=0.10` |
| `F3` | `swing_time = (1 - FAST.duty) * FAST.describe(H)["period"]` (0.320 s at 60 cm, 0.356 s at 75 cm) |
| `F4F5` | `lift_peak=0.375, toe_off=0.30, toe_off_rate=3.4, toe_off_accel=220.0, toe_off_start=0.825, toe_reach=0.075, toe_off_margin=0.005` |
| `F6` | `roll_amp = radians(1.5)` |
| `F9F10` | `tail_amp=0.125, head_mode="stabilise", head_gain=0.9, head_settle=0.015, settle_lag=0.075`, plus `tail_phase` when given |
| `F14` | `phase_jitter=0.015, asym=0.025, gaze_var=0.015, tail_var=0.2, var_band=(0.05, 0.2), noise_seed=2026` |
| `F13` | `stride_rel=0.875` only where `f13_applies(body, speed)` (fr05 and a body starting with "75"), else `{}` |
| `F7` | `zmp_inset_y=0.010` |
| `F5full` | `toe_off=0.40, toe_off_rate=4.5, toe_off_accel=290.0` |

`enforced_shortening(spec)` (318-321) reads `spec.duty`, `spec.stride_rel`, `spec.crouch`.

`ladder(body, H, speed, tail_phase=None, extras=True, tail_phase_f13=None) -> {variant: GaitSpec}` (360-392). Each rung
is `dataclasses.replace` of the one before:

```
V0 = base_spec(H, speed)
V1 = replace(V0, **F1F2, zmp_iters=12)        V1B50 / V1B100 = replace(V1, bob=share * enforced_shortening(V0))
V2 = replace(V1, **F3)
V3 = replace(V2, **F4F5)                       V3T = replace(V3, **F5full)
V4 = replace(V3, **F6)                         V4L = replace(V3, roll_amp=-F6)
V5 = replace(V4, **F9F10)                      only when tail_phase is given
V5F13 = replace(V5, **F13, tail_phase=tail_phase_f13)     only where F13 applies and tail_phase_f13 is given
V6 = replace(V5 or V5F13, **F14), then strides = window_strides(...)     V6S = replace(V6, **F7)
```

Without `tail_phase`, V5, V6, V6S are absent. `features_of(variant, body, speed)` (395-412) holds its own hard-coded
tables; an unknown variant returns `[]`.

### 3.4 Generation

```python
source(body) -> dict            # look.py:418; refs.source plus "geometry" (gait.foot_geometry) and "markers"
generate(src, spec) -> dict     # look.py:470; gait.generate at r2, adds "qd", "qdd", "seconds";
                                # a default-path walk gets its look block from _complete_v0 (look.py:485-518)
```

`generate` accepts any `GaitSpec`. `markers` (524-573) needs bodies `head` and `tail` and geoms `cover_skull`,
`hw_eye_0`, `hw_<side>_claw_0`, `cover_tail*`; it asserts the names equal `POINTS`.

### 3.5 `arrays(src, g)` (look.py:581-648): the npz of a rung

Verified shapes on `75-M-H-10` (nq 21, nv 20, 14 hinges). `n = spec.n * spec.strides`. Units m, rad, s. Column order
left, right. Key order below is the file's.

| Key | Shape | Note |
|---|---|---|
| `t` | (n,) | |
| `dt` | () | float64 |
| `q`, `qd`, `qdd` | (n, 21), (n, 20), (n, 20) | |
| `support` | (n, 2) bool | stance flags; equals `~swing` |
| `com_ref`, `zmp_ref`, `zmp_mb` | (n, 2) | |
| `feet` | (n, 2, 3) | foot body origins, solved pose |
| `com` | (n, 3) | |
| `q_home` | (21,) | |
| `joints` | (14,) str | `M.names`: neck_pitch, tail_yaw, then left and right hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll |
| `stride_motion` | (5,) | `[0, 0, 0, K * stride, 0]` |
| `phase` | (n,) | strides; 0 at the left foot's lift-off |
| `swing` | (n, 2) bool | planned swing flags |
| `swing_u` | (n, 2) | swing fraction, NaN in stance |
| `foot_pitch`, `foot_pitch_solved` | (n, 2) | target and solved heel-up pitch |
| `rolling`, `on_claw` | (n, 2) bool | |
| `toe_pivot` | (n, 2, 3) | midpoint of `toe_edge` |
| `toe_edge` | (n, 2, 2, 3) | front contact's two end points |
| `foot_quat` | (n, 2, 4) | |
| `ik_ok` | () bool | |
| `ik_fail_frames` | (k,) int64 | |
| `foot_target` | (n, 2, 3) | |
| `support_hull` | (n, hmax, 2) | NaN-padded; hmax 10 in the check |
| `zmp_margin`, `zmp_ref_margin` | (n,) | m, positive inside |
| `hip_mid` | (n, 3) | midpoint of the hip-roll axes |
| `bob`, `trunk_roll`, `neck`, `tail`, `head_settle`, `gaze_var`, `tail_var`, `head_pitch_world` | (n,) | |
| `trunk_quat` | (n, 4) | |
| `pad_floor_z`, `toe_floor_z` | (n, 2) | lowest pad point, lowest toe or claw point, m above the floor |
| `points`, `point_names` | (n, 13, 3), (13,) | markers |
| `lift_off_t`, `touchdown_t` | (2, K) | s, planned events from the phase (`events`, look.py:651-662); row 0 left |

`trajectory(arrs, info)` (665-676) rebuilds a `gait.generate`-style dict and refuses arrays without `ik_ok`.
`turning_traj(traj, M, q_home)` (679-689) dresses a straight walk for `turning.inverse_dynamics`.

**The npz bytes depend on this key set and order.** Adding, removing or renaming a key in `arrays` changes the sha256
of every rung.

Events for the bar's readings, as they exist on references:
* planned: `lift_off_t`, `touchdown_t`;
* detected: `contact_events(arrs)` (983-1011) from `pad_floor_z` and `toe_floor_z` with `CONTACT_TOL = 3e-4` m (970);
  returns `on_floor` (n,2), `duty` [2], `swings` (per foot, list of (lift-off, touchdown) in s).
* The clearance gate's "middle half" is `clearance.MID_SWING = (0.25, 0.75)` (clearance.py:164), threshold
  `SWING_MIN_FRAC = 0.04` H (clearance.py:136), value rounded to 4 decimals (clearance.py:781, 786).

These read prescribed walks. Closed-loop readings are not in this subsystem.

### 3.6 `price(src, arrs, info, with_clearance=True) -> dict` (look.py:891-947)

Steps: `trajectory`; `look_inverse_dynamics(M, tr, arrs["rolling"], arrs["toe_edge"])` at r2 (785-846);
`sizing._motor_space(M.names, tau, omega, src["layout"])`; `envelope._ratios(design, bus, ms)`; contact block;
`_clearance_gates` (867-888: `clearance.screen(xml, q_home, {"nominal": ...}, H, gates="r2")` on the round-two MJCF from
`plant.model_files(body)`).

Returned keys: `dynamics`, `drives_at` {`bus_v`, `margins`}, `drives` {group: `actuator, rms, peak, speed, cont_ratio,
peak_ratio, speed_ratio, worst`}, `worst_group`, `worst_ratio`, `passive` {neck, tail: `damping, frictionloss,
cont_ratio, peak_ratio`}, `contact` (11 keys, e.g. `single_support_cop_margin_min_mm`, `friction_ratio_max`,
`total_friction_need_max`, `split_failed_frames`), `clearance` {`gates` (6 booleans), `walking_gates`,
`mid_swing_min_frac_H`, `mid_swing_min_mm`, `mid_swing_threshold_frac_H`, `lateral_gap_mm`, `tightest_joint`,
`joint_range_margin_rule_rad`, `joint_range_margins`, `nominal_gait_interference_pairs`, `tail_min_frac_H`, `screen`},
`over` (list of {`limit`, `value`}), `within_limits`, `verdict`.

* `walking_gates` here needs all six gates, `swing_clearance` and `tail_clearance` included (881-883).
  `round2.WALK_GATES`, which gaittune uses, has neither.
* `joint_range_margins[<joint>]` = {`walk_min, walk_max, usable, margin_min, margin_max`}. Recorded for both bodies:
  `left_hip_yaw` and `right_hip_yaw` usable `[-0.3, 0.3]` rad, straight-walk excursion 0. The gate needs 0.10 rad of
  margin (`clearance.RANGE_MARGIN_RAD`). This is where a thigh-twist check reads from.
* `_passive_ratios` is hard-coded to `neck_pitch` and `tail_yaw` (854).

### 3.7 `measure(arrs, info, H, crouch=0.03) -> dict` (look.py:1014-1204)

Up to 49 keys (`trunk_roll_toward_stance`, `settle_lag_s` and `settle_depth_rad` appear only with a roll or a
settle), each measured on the frames: `bob_pp_H`, `duty`, `swing_time_s`, `cadence_steps_s`, `lift_peak_frac_swing`,
`toe_off_at_lift_off_rad`, `foot_below_floor_max_mm`, `trunk_roll_amp_rad`, `sway_amp_H`,
`com_accel_horizontal_peak_ms2`, `tail_amp_rad`, `tail_phase_rad`, `nod_amp_rad`, `stride_H`, `step_time_dev_max`,
`zmp_margin_min_mm`, `extremes_phase`, `mechanical_tells` {`constant_hip_height, flat_feet_at_lift_off,
identical_steps, metronome_nod`}, and others. The `crouch` default is `GaitSpec.crouch`; callers pass the spec's.

`tail_phase_rad` comes from a fit `y = A sin(2 pi phase + psi)` with A >= 0 (`_fit_phase`, 961-965). A walk with a
negative `tail_amp` therefore measures `psi = -pi/2` while its spec still says `tail_phase = pi/2`.

### 3.8 Records: `write`, `_job`, `_merge`, `prepare`

`write(src, speed, variant, spec, g, out_dir) -> dict` (1349-1368):

* saves `<out>/refs/<body>/<speed>/<variant>.npz` with `np.savez(path, **arrays(src, g))`;
* `sha256 = _sha(path)`: sha256 of the npz file's bytes (1309-1310);
* returns the rung's entry.

**A rung's entry in `look_refs.json`** (`bodies[body][speed][variant]`), 19 keys:

| Key | Content |
|---|---|
| `body`, `speed`, `variant` | names |
| `features` | `features_of(...)`, e.g. `["F1", "F2", "F3"]` |
| `file` | `"refs/<body>/<speed>/<variant>.npz"`, relative to the out dir |
| `sha256` | of that file |
| `frames` | n |
| `spec` | `_plain(asdict(spec))`: all 38 fields, `var_band` as a list |
| `info` | the generator's info without `look` |
| `ik_ok`, `ik_fail_frames` | bool, count |
| `zmp_error_final_mm`, `zmp_margin_min_mm` | |
| `rolls` | {left, right: list of roll dicts inside the window} |
| `measured` | `measure(...)` |
| `price` | `price(...)` |
| `flags` | list of strings for the owner (`flags`, 1322-1346) |
| `seconds`, `price_seconds` | wall time; they differ on every run |

`_job((body, speed, out_dir, variants, resume))` (1402-1447), one body and speed:
1. tail checks: `tail_check` on V0 and on V4 (and V4 with F13's stride where it applies); `tail_phase` is V4's
   `best_phase_rad`. They always run unless `--resume` finds a partial file, whichever variants were asked for.
2. `checks["zmp_passes"] = zmp_passes_check(src, V0)`.
3. `lad = ladder(body, H, speed, tail_phase, tail_phase_f13=...)`; for each requested variant that `lad` holds:
   `generate`, `write`. A rung is skipped only when the partial file already holds it with the same spec and file
   (`_entry_current`, 1382-1385). **Entries already in `look_refs.json` are not consulted**: asking for an existing
   variant regenerates it and replaces its entry.

`_merge(out_dir, results, infos, t0, prov)` (1459-1493):
* `doc["bodies"][body][speed].update(entries)`: rung by rung, other rungs kept;
* `doc["checks"][body][speed]` replaced; `tail_phase.json`'s `bodies[body][speed]` replaced;
* `doc["body_info"].update(infos)`;
* top-level keys overwritten on every call: `accuracy`, `values`, `variants`, `extras`, `speeds`, `provenance`,
  `versions`, `generator`, `note`, `limit_rule`, `deferred`, `price_rule`, `seconds`.

`prepare(bodies=BODIES, out_dir=OUT, speeds=SPEEDS, variants=VARIANTS + EXTRAS, procs=1, resume=False)` (1500-1529)
takes `plant.provenance()` first, computes `body_info` for the bodies and calls `_merge` once before any job. So the
file is rewritten as soon as `prepare` starts.

`remeasure(out_dir)` (1532-1548) and `reflag(out_dir)` (1551-1566) rewrite `look_refs.json` in place. `reflag` adds a
top-level `reflag` block; `_merge` never removes it.

### 3.9 `stage2/look/look_refs.json` as committed

Top-level keys: `bodies`, `body_info`, `checks`, `accuracy` ("r2"), `values`, `variants`, `extras`, `speeds`,
`provenance`, `versions`, `generator`, `note`, `limit_rule`, `deferred`, `price_rule`, `seconds`, `reflag`.

* 73 rungs: 2 bodies x 3 speeds x 12, plus `V5F13` on `75-M-H-10` `fr05`.
* `provenance` = `plant.provenance()`: `versions`, `git` {`commit` 11d95dc5..., `dirty` false, `dirty_paths`},
  `stage1_sources` {`sha256` 4ad0f657..., `rule`, `files`}.
* `checks[body][speed]["zmp_passes"]` = {`passes` [6, 12], `com_ref_max_diff_mm`, `root_xyz_max_diff_mm`,
  `joint_max_diff_rad`, `zmp_error_final_mm` [2]}.
* `body_info[body]` = {`H_m`, `foot_frame`, `feet` {side: `toe_pad, toe_edge, heel_pad, heel_edge, pads, claw`},
  `markers`, `toe_clearance`, `claw_question`}.
* Frames: 160 for one-stride rungs; V6 and V6S 1440 to 2240 (9 to 14 strides).
* Cost per rung on record (75 cm, v040): V0 9 s plus 7 s price; V1 to V5 about 15 s plus 7 s; V6 187 s plus 15 s.
  The file's `seconds` is 3058 (the run's wall time; the process count is not recorded).
* One rung pair is over a limit: `75-M-H-10` `fr05` V6 (knee speed 1.012) and V6S (1.009).
* The arrays are git-ignored (`*.npz`, repository `.gitignore:49`) and were not on disk when this map was made.

`stage2/look/tail_phase.json`: `bodies[body][speed]` = {`tail_phase_F9_rad`, optional `tail_phase_F9_F13_rad`, `V0`,
`V4`, optional `V4F13`}; each check has `amp_rad, no_tail, runs, fit_max_residual_nm, predicted_peak_nm,
best_phase_rad, confirmed, today_adds_to_leg_yaw_momentum, flipped_lowers_peak, ...`.

| Body, speed | F9 phase (V4's minimiser), rad | V0 sweep's best phase, rad |
|---|---|---|
| 60-N-T-10 nominal | -1.7017 | -1.8500 |
| 60-N-T-10 v040 and fr05 | -1.7802 | -1.8850 |
| 75-M-H-10 nominal | -0.9250 | -1.8500 |
| 75-M-H-10 v040 | -1.7715 | -1.8413 |
| 75-M-H-10 fr05 | -1.7715 (with F13 also -1.7715) | -1.8850 |

### 3.10 Command line (look.py:1594-1621)

```
python -m designlab.stage2.look prepare [--body B]... [--speed S]... [--variant V]... [--out DIR] [--procs N] [--resume]
python -m designlab.stage2.look g2 --body B [--name NAME]...      # JSON on stdout
python -m designlab.stage2.look remeasure [--out DIR]
python -m designlab.stage2.look reflag [--out DIR]
```

`--body`, `--speed`, `--variant` use `choices=BODIES`, `SPEEDS`, `VARIANTS + EXTRAS`. Run as `__main__`, the module
sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` to 1 (121-123); imported, it does not.

---

## 4. Where hashes are computed and what they cover

| Hash | Computed at | Covers |
|---|---|---|
| a rung's `sha256` | look.py:1364 via `_sha` (1309) | the npz file's bytes |
| Stage 1 reference `sha256` | refs.py:297 | the npz of `refs.arrays` |
| G2 | `g2_hashes(body, names)`, look.py:1578-1591 | regenerates Stage 1 references with `refs.generate` and `refs.write` in a temp dir; returns `{name: {sha256, matches_records}}` |
| `provenance.stage1_sources.sha256` | plant.py:267, 294 | every `designlab/stage1/*.py` |
| `screen.json` `inputs.look_refs_json_sha256`, `tail_phase_json_sha256` | screen.py:821, 869 | the whole committed files |
| `look_page.json` `inputs` | look_page.py:2437 | whole files: metrics, screen, clips, look_refs |
| `metrics.json` `rung_sha256`, `clips.json` `rung_sha256`, `screen.json` `rungs...arrays.sha256` | motion.py:429, clips.py:583, screen.py:527 | each rung's npz sha256, copied |
| panel sources key | `panels.panel_sources()`, panels.py:1572-1583 | see below |

* `look_refs.json` does **not** hash `designlab/gait.py` or `designlab/stage2/look.py`. Its code is named only by the
  git commit, the dirty flag and the Stage 1 sources hash. Not found: any per-module hash of the look code.
* The committed `look_refs.json` has sha256 `368a3372...`; `screen.json` and `look_page.json` both record that value.
  `tail_phase.json` is `908a5f7e...`, recorded in `screen.json`.
* **`gait.py` is inside the panels' key.** `panel_sources()` today lists these designlab modules: `__init__, build,
  carpet, catalog, clearance, envelope, gait, geometry, hardware, mjcf, params, shapes, sizing, skeleton, statics,
  template, turning`. Any edit to `gait.py`, `turning.py`, `sizing.py` or `envelope.py` changes that key.
* `gaittune.py` and everything under `designlab/stage2/` are outside it. `designlab_files()` globs `designlab/*.py`
  only (panels.py:1510-1512), and an import of `..stage2` from a Stage 1 module resolves to `stage2.py`, which is not
  in that set and is dropped (panels.py:1549). So Stage 2 code imported by a keyed Stage 1 module would run unhashed.
* The tuner's rule hashes Stage 1 modules only (tune.py:374-381).
* The current `designlab/stage1/*.py` hash is `cc2a5c41...`, already different from the ladder's `4ad0f657...`.

---

## 5. F25: the tuned walk

### 5.1 How `gaittune.py` expresses it

* `BODIES = {"60-N-T-10": "narrow_carpet_60", "60-N-H-10": "narrow_carpet_60", "75-M-H-10": "feet_75"}`
  (gaittune.py:44). No -K or -X body.
* The base is `envelope.FAST` (gaittune.py:138). A candidate is

  ```python
  def _spec(base, x): return replace(base, **{k: float(v) for k, v in x.items()})     # gaittune.py:131-132
  ```

  with `x` over exactly four fields: `lift_frac`, `duty`, `stride_rel`, `crouch` (`GRID`, 45-46; `BOUNDS`, 47:
  lift 0.08-0.12, duty 0.55-0.65, stride 0.60-0.95, crouch 0.01-0.07). Froude stays 0.05.
* The walk is generated by `turning.generate(M, q0, H, spec, yaw_rate=0.0)` and `turning.inverse_dynamics`
  (gaittune.py:76-77), not `gait.generate`.
* Feasibility (98-104): IK, single-support CoP margin above -10 mm, `carpet.toe_catch` hard-floor mid-swing toe at
  least 0.04 H less 0.01 mm, duty at least 0.55. The chosen walk must also pass `round2.WALK_GATES` on
  `clearance.screen(..., gates="r2")` (`feasible_full`, 105-113). All three chosen walks record
  `clearance_pass.swing_clearance` true as well.

### 5.2 The records: `round2/gait_tune/<body>.json`

Top-level keys: `id`, `case`, `H_m`, `bus`, `drives`, `baseline`, `grid` (144), `finalists` (6), `pattern` (24 to 32),
`chosen`. Each walk entry: `spec`, `speed_ms`, `cadence_steps_s`, `stride_m`, `ik_ok`, `min_cop_margin_single_mm`,
`cop_on_pad`, `toe_mid_swing_mm`, `toe_target_mm`, `toe_ok`, `duty_ok`, `sway_pp_H`, `com_accel_peak_ms2`, `ratios`
{group: `cont_ratio, peak_ratio, speed_ratio, worst`}, `worst_group`, `worst_ratio`, `feasible`, `seconds`; baseline,
finalists and chosen add `clearance_pass`, `walking_gates`, `tightest_joint`, `feasible_full`.

`spec` holds six fields only: `froude, lift_frac, duty, stride_rel, crouch, sway_gain` (gaittune.py:92-93).

The records hold **no arrays, no sha256 and no provenance block**. `round2/gait_tune/.gitignore` ignores `models/`.
No test covers `gaittune.py` or these files (not found in `tests/`).

### 5.3 The chosen walks

| Body | froude | lift_frac | duty | stride_rel | crouch | speed | stride | cadence | worst ratio | toe, mid-swing vs target |
|---|---|---|---|---|---|---|---|---|---|---|
| 60-N-T-10 | 0.05 | 0.08 | 0.55 | 0.7125 | 0.07 | 0.44553 m/s | 0.28834 m | 3.090 /s | hip_pitch 0.7981 | 16.18731 vs 16.18757 mm |
| 60-N-H-10 | 0.05 | 0.08 | 0.55 | 0.725 | 0.07 | 0.44376 m/s | 0.29108 m | 3.049 /s | hip_pitch 0.7935 | 16.05905 vs 16.05931 mm |
| 75-M-H-10 | 0.05 | 0.08 | 0.55 | 0.8625 | 0.05 | 0.49614 m/s | 0.43285 m | 2.292 /s | hip_yaw 0.7542 | 20.07400 vs 20.07414 mm |

`GAIT_TUNE.md` prints the strides rounded (0.713, 0.725, 0.863). The toe clearance is below its target by 0.0003 mm
and passes on the 0.01 mm tolerance: no margin. Lift 0.08 H with the sin^2 swing gives exactly 0.5 x 0.08 = 0.04 H at
25 % and 75 % of the swing.

### 5.4 Can a `GaitSpec` reproducing F25 be built with existing fields? Yes

```python
replace(envelope.FAST, lift_frac=0.08, duty=0.55, stride_rel=0.7125, crouch=0.07)   # 60-N-T-10
replace(envelope.FAST, lift_frac=0.08, duty=0.55, stride_rel=0.8625, crouch=0.05)   # 75-M-H-10
```

Checked: these give `look_active == False`, with head_amp 0.05, tail_amp 0.12, zmp_iters 6, n 160 from `FAST`. They
take the default path of `gait.generate`, and `turning.generate` accepts them.

Limits of that statement:
* F25 is defined at Froude 0.05 only: 0.4455 m/s at 60 cm, 0.4961 m/s at 75 cm. There is **no F25 at 0.40 m/s** in
  `designlab`.
* The recorded F25 numbers came through `turning.generate`. `gait.generate` gives the same walk by definition, not
  the same arrays; the two generators agree on V0's knee speed ratio to 0.2 % (`test_v0_price_reproduces_...`).

### 5.5 "F25 + F9" and "F25_F9_cl" (the PR's walks)

`_tuned_f9_spec(body, H, v)` lives only in `stage2/jetson_plem/jetson_plem_study.py:369-390`:

```python
fr = v * v / (gait.G * H)
crouch = 0.07 if body.startswith("60") else 0.05
duty = 0.58 if v < 0.36 else 0.55
base_sr = 0.713 if body.startswith("60") else 0.863
sr_cap = 0.90 if body.startswith("60") else 0.94
sr = round(min(sr_cap, max(0.62, base_sr * ((max(fr, 0.025) / 0.05) ** 0.20))), 4)
return gait.GaitSpec(froude=fr, duty=duty, lift_frac=0.081, head_amp=0.05, tail_amp=-0.125, sway_gain=1.0,
                     stride_rel=sr, crouch=crouch, zmp_iters=6, zmp_gain=0.5, n=160)
```

* It is not F25's record: lift 0.081 (not 0.080), stride base 0.713 / 0.863 (not 0.7125 / 0.8625).
* F9 is expressed as a **negative `tail_amp`**, not the `tail_phase` field. So it is not a look spec and runs the
  default path. The ladder's F9 is `tail_amp=0.125` with `tail_phase` near -1.77 rad, which runs the look path.
* On `75-M-H-10`: 0.2431 m/s gives duty 0.58, stride 0.7513 H; 0.30 and 0.35 the same; 0.40 m/s gives froude
  0.032499, duty 0.55, stride 0.7918 H, T 0.9934 s; 0.45 gives 0.83 H; 0.4961 gives 0.863 H; 0.50 gives 0.8657 H.
* `F25_F9_cl` (that spec with duty 0.59, lift 0.095 H) exists in text only: `JETSON_PLEM_REVIEW.md:104, 134, 213`,
  `NEXT_SESSION_PROMPT.md:129-130`. No code builds it.

---

## 6. Places that hard-code an assumption the build list would break

| # | Where | Assumption | Breaks |
|---|---|---|---|
| 1 | turning.py:380-382 | `turning.generate` raises `ValueError` for any look spec | B |
| 2 | turning.py:396-400, 421-422 | reads `spec.duty`; neck `head_amp sin(2wt)`; tail `tail_amp sin(wt + pi/2)` | B |
| 3 | transitions.py:165-179 | `gait_of(body, name, H)` takes names from `refs.walk_specs` or turn cases only; reads `spec.duty`, `lift_frac`, `head_amp`, `tail_amp`, `crouch` | A |
| 4 | transitions.py:422 | tail law fixed at `+ np.pi / 2` | A |
| 5 | refs.py:80-82, 113-134 | four named walks; `generate` takes a name | A, E |
| 6 | look.py:295-304 | `speed_spec` mixes only froude, stride, duty, lift; copies crouch, tail and look fields from `lo`; valid only between `lo`'s and `hi`'s speeds | A, D |
| 7 | look.py:307-315, 146 | `base_spec` knows three speed keys; `SPEEDS` is the CLI's choices | D |
| 8 | look.py:145 | `BODIES` is two bodies; CLI choices | D, F |
| 9 | look.py:149-150, 1500, 1600 | `VARIANTS + EXTRAS` is the default and the CLI's choices | D |
| 10 | look.py:395-412 | `features_of` tables; unknown rung gives `[]` | D |
| 11 | look.py:360-392 | V0 is always `base_spec`; the chain is fixed | D |
| 12 | look.py:1416-1428 | every job runs the tail checks on V0 and V4 | D (cost; rewrites `tail_phase.json`) |
| 13 | look.py:1418-1422, 1278-1289 | at "nominal" it reads the body's round-two `<id>_turning.json` | D on a new body |
| 14 | look.py:546-571 | marker geoms and bodies by name | F (-X heads) |
| 15 | look.py:854; sizing.py:84-85 | `neck_pitch`, `tail_yaw` by name | F |
| 16 | gait.py:244-245, 989-990 | the same two joints set by the generator; other joints stay at home | F (neck yaw and jaw are never moved) |
| 17 | look.py:581-648 | npz has no `tau`; `Reference.joint_series` needs it (refs.py:366) | E |
| 18 | screen.py:127 | `ORDER`: a rung not listed is silently not screened (825-826) | D |
| 19 | screen.py:620-622 | raises unless `ladder()`'s V0, V4, V5 `__dict__` equal the recorded specs, key for key | any new `GaitSpec` field |
| 20 | screen.py:873-875 | needs `rungs[b]["v040"]["V0"]` and `["V2"]` | D in a separate out dir |
| 21 | look_page.py:100-105, 113 | `RUNG_TITLE`, `RUNG_TAG`, `RUNG_ADDS` keyed V0-V6; `LADDER_X = LADDER + look.EXTRAS` | D |
| 22 | motion.py:725, 801 | `v in look.VARIANTS`; `("V5F13",)` | D |
| 23 | gaittune.py:44 | three bodies | D |
| 24 | look.py:1314-1315 | `DUTY_BAND = (0.55, 0.65)`; F25's duty sits on its edge | D |
| 25 | tools/restore_inputs.py:24, 679-681 | "73 rungs"; restores by calling `look prepare --variant ...` | D (new names must be CLI choices) |

---

## 7. Tests that pin behaviour (`tests/test_stage2_look.py` unless said)

G2 and the default path:

| Test (line) | Asserts |
|---|---|
| `test_look_fields_default_to_todays_walk` (75) | `LOOK_FIELDS` are fields; `LOOK_DEFAULTS` keys equal them; **the first 11 field names in order**; NOMINAL, BRISK, FAST and `GaitSpec()` are not look specs; `describe` unchanged; `tail_phase == pi/2`, `head_mode == "nod"` |
| `test_a_default_spec_read_back_from_json_stays_todays_walk` (93) | JSON round trip stays non-look |
| `test_default_fields_take_todays_code_path` (102) | with `_generate_look` patched to raise, a default walk still generates |
| `test_turning_generate_refuses_a_look_spec` (113) | `turning.generate(..., replace(NOMINAL, bob=0.01, n=20))` raises `ValueError` matching "look"; a default spec with `yaw_rate=0.35` works; `"turns"` and `"starts_and_stops"` are in `look.DEFERRED` |
| `test_g2_stage1_references_regenerate_bit_for_bit` (139; fixture 124) | both `look.BODIES`, ten references each (`G2_NAMES`, line 55: nominal, brisk, fr05, carpet, six turns), regenerated by `python -m designlab.stage2.look g2` in subprocesses with one BLAS thread; each sha256 equals `stage1/runs/2026-11-08/<body>/refs.json` and `matches_records` is true |

Unit pieces: `test_lift_profile_...` (155), `test_bob_profile_...` (178), `test_toe_off_bump_...` (199),
`test_warp_...` (220), `test_variation_is_seeded_and_bounded` (234).

The ladder's definition:

| Test (line) | Asserts |
|---|---|
| `test_speed_spec_is_the_speed_change_at_0_40` (249) | speed 0.40 to 1e-12; w, T, stride, duty per body; lift 0.11; not a look spec |
| `test_ladder_values_and_one_change_per_rung` (264) | `lad["V0"] is sizing.NOMINAL`; every feature value; which fields may change between consecutive rungs; `zmp_iters` 12 from V1; F13 rules; `"V5F13" not in lad` at 60 cm nominal |

Generated walks (fixture `walks`, 327: 60-N-T-10 nominal V0, V1, V3, V5 and a 3-stride V6): tests at 342-581, 632;
`test_hybrid_animal_pace_with_every_feature` (583); `test_markers_on_both_bodies` (615), which pins `H_BODY`.

The committed ladder (skipped if `look_refs.json` is absent; `ENTRIES` is every rung in the file):

| Test (line) | Asserts |
|---|---|
| `test_the_ladder_has_every_rung` (661) | per body and speed, the rung set **equals** `VARIANTS + EXTRAS` (less `V5F13` where F13 does not apply); `LADDER["deferred"] == look.DEFERRED`; `LADDER["limit_rule"] == look.LIMIT_RULE` |
| `test_every_variant_solves_...` (670, per rung) | IK; ZMP margin > 0; final ZMP error <= 1.25 x V0's at that body and speed and < 3.5 mm; measured bob, duty, lift peak, roll, stride against the spec; `cos(measured tail phase - spec.tail_phase) == 1` (692); `frames == n * strides`; speed equals a dict keyed `nominal`, `v040`, `fr05` (696-697) |
| `test_every_rung_is_priced_...` (716, per rung) | swing clearance gate true and >= 0.04 H; all walking gates true; tightest joint >= 0.10 rad; split solved; single-support CoP above -10 mm; over-limit list matches the ratios |
| `test_v0_price_reproduces_the_round_two_envelope_screen` (743) | fr05 V0 knee and hip-pitch speed ratios within 0.2 % of the envelope record |
| `test_flags_for_the_owner` (756) | flags per rung |
| `test_checks_zmp_passes_and_bob_drift` (776) | per body and speed |
| `test_tail_check_...` (791) | `tail_phase.json` against V5, V5F13, V6, V6S specs |
| `test_body_info_records_the_claw` (824) | `claw_question == look.CLAW_QUESTION` |
| `test_recorded_measures_recompute_from_the_arrays` (842) | only when all arrays are on disk |
| `test_a_variant_regenerates_to_its_recorded_hash_and_price` (891) | 60-N-T-10 nominal V4 regenerated in a subprocess through `look.ladder(...)`: sha256, price and flags equal |
| `test_every_variant_regenerates_to_its_recorded_hash` (902) | the same for every rung; only with `STAGE2_SLOW=1` |

Other files that read the ladder:

| Test | Asserts |
|---|---|
| `test_stage2_screen.py::test_every_rung_is_screened_from_its_own_arrays` (124) | the rungs in `screen.json` **equal** the ladder's entries; `arrays_checked == len(ENTRIES)`; each sha256, price, features, flags equal |
| `test_stage2_screen.py::test_g2_on_the_code_that_ran_the_screen` (136) | `screen.json`'s G2 block passes |
| `test_stage2_screen.py::test_screen_md_regenerates_from_screen_json` (232) | `SCREEN.md` equals `screen.markdown(SCREEN)`; the screen ran on a clean tree |
| `test_stage2_motion.py::test_metrics_json_reads_every_rung_...` (199) | every ladder rung is in `metrics.json` with its sha256; `constant_hip_height == (v == "V0")`; `identical_steps == (v not in ("V6", "V6S"))` |
| `test_stage2_clips.py` (246, 295) | clips' `rung_sha256` equal the ladder's |

No test compares `screen.json`'s or `look_page.json`'s recorded `look_refs.json` sha256 with the file (not found).

---

## 8. Adding rungs such as "F25F9" and "F25F9cl": what the code allows

### 8.1 What fixes an existing rung's bytes

1. Its spec: `ladder(body, H, speed, tail_phase, tail_phase_f13)[variant]`, hence `base_spec`, `speed_spec`,
   `feature_values`, the constants, and the phases in `tail_phase.json`.
2. `gait.generate` (the path the spec takes) and `gait.derivatives` at r2; `statics.leg_ik`, `statics.total_zmp`.
3. `_complete_v0` for default-path rungs; `look.arrays` (key set and order), `markers`, `events`, `foot_geometry`,
   `gait._contact_in_foot`, `carpet.foot_geoms`, `carpet.lowest_point`, `carpet._is_toe`.
4. The body's model; numpy, mujoco, scipy versions; one BLAS thread.

Not in the npz: `features_of`, `measure`, `price`, `flags`, `VALUES`, `DEFERRED`, membership of `VARIANTS` or
`EXTRAS`, and the JSON file.

So new keys added to `ladder()`'s returned dict, new names in `EXTRAS` or a new tuple, and new `features_of` rows
leave every existing rung's spec and npz sha256 as they are, provided the existing `replace` chain and
`feature_values` entries are not touched.

### 8.2 What changes anyway if the rungs are merged into `stage2/look/look_refs.json`

* The per-rung entries of existing rungs are kept by value (`_merge`'s `update`), sha256 included.
* The file itself changes: `provenance`, `versions`, `values`, `variants`, `extras`, `seconds`, `body_info` and the
  job's `checks` block are rewritten, and `tail_phase.json`'s block for that body and speed is replaced by a rerun.
* Its whole-file sha256 (`368a3372...`) then no longer matches what `screen.json` and `look_page.json` record.
* `prepare --out <another directory>` writes its own `look_refs.json` and `tail_phase.json` there and leaves the
  committed pair alone; `tools/restore_inputs.py:669-687` already uses it that way. The tests in section 7 read only
  `stage2/look/`, and `screen.run` on such a directory needs V0, V2, V4, V5 and a `tail_phase.json` in it (table 6,
  rows 19-20).

### 8.3 Which tests would notice, case by case

| Change | Tests that fail or guard |
|---|---|
| Any edit that alters the default path, `derivatives`, `inverse_dynamics`, `turning.generate`, `refs.arrays` | G2 (139); `test_default_fields_take_todays_code_path` (102) |
| An existing rung's spec or bytes change | `test_a_variant_regenerates_...` (891) for one rung; all rungs only with `STAGE2_SLOW=1` (902); `test_ladder_values_...` (264); `screen.tail_phase` raises at run time (screen.py:621) |
| A new rung in the committed file but not in `VARIANTS + EXTRAS`, or the reverse, or missing at some body or speed | `test_the_ladder_has_every_rung` (661) |
| A new rung under a new speed key | `KeyError` at 696-697; `base_spec` |
| A new rung with F9 as a negative `tail_amp` | 692: `cos(-pi/2 - pi/2) = -1` |
| A new rung at lift 0.08 H | 723 passes only at exactly 0.04 H after rounding; with F4's `lift_peak=0.375` the lift kept at 75 % of the swing is 0.45-0.47 of it, about 0.037 H, which fails |
| A new rung with a ZMP error above 1.25 x V0's at the same body and speed | 679 |
| New rungs in the file, `screen.json` not rerun, or rerun without extending `ORDER` | `test_stage2_screen.py:124` |
| New rungs in the file, `metrics.json` not rerun | `test_stage2_motion.py:199` (`KeyError`) |
| A new rung without a bob (F25 has none) | `test_stage2_motion.py:199`: `constant_hip_height == (v == "V0")` |
| `DEFERRED`, `LIMIT_RULE` or `CLAW_QUESTION` text edited | 665, 828 against the committed file |
| A new rung that `look.ladder(body, H, speed, tail_phase, tail_phase_f13=...)` cannot build from those arguments | `_regenerate` (858-872) under `STAGE2_SLOW=1` |
| A new `GaitSpec` field | 75-91 if it is placed among the first 11 fields, or named in `LOOK_FIELDS` without being a field; `screen.tail_phase` at run time (recorded specs have 38 keys); `_entry_current` on resume. No test fails for a field appended at the end with today's default |

---

## 9. Extension points and hazards (short)

Extension points that exist today:
* `GaitSpec` plus `LOOK_FIELDS`: a field whose default is today's behaviour, dispatched by `look_active`.
* `look.generate(src, spec)`, `look.arrays`, `look.write`, `look.price`, `look.measure` all take an arbitrary spec and
  out dir; only `ladder`, `base_spec`, `features_of`, `_job` and the CLI are tied to the fixed rung and speed names.
* `speed_spec(H, v, lo, hi)` takes any pair of non-look walks.
* `prepare(out_dir=...)` and `write(..., out_dir)` for records beside, not inside, the committed ladder.
* `price`'s `clearance.joint_range_margins` for the thigh-twist range.

Hazards:
1. `gait.py`, `turning.py`, `sizing.py`, `envelope.py` are in the panels' source key (section 4). Editing them makes
   earlier panel records read as made by other code even when G2 passes. `designlab/stage2/*` and `gaittune.py` are
   not in any key, so code placed there and called from Stage 1 modules is not hashed at all.
2. `look.arrays`' key set is part of every rung's sha256. A reference format with `tau` for the tracker cannot be had
   by adding a key there.
3. The look path is not bit-identical to the default path at equal values (1e-15 rad). F9 as a negative `tail_amp`
   and F9 as `tail_phase` are different arrays and take different paths.
4. At least four places read `spec.duty` where `describe(H)["duty"]` is the truth under F3: `turning.generate`
   (turning.py:396), `transitions.gait_of` (transitions.py:173, 178), `look.speed_spec` (look.py:303),
   `look.enforced_shortening` (look.py:321).
5. The accuracy level is process state that `refs.generate` and `gaittune.Body` set and do not restore.
6. F25 has no arrays, no hash and no test today, exists for three bodies at Froude 0.05 only, and sits at the
   0.04 H clearance threshold with no margin. The 0.40 m/s version exists only as the PR's stride rule in a study
   script, with a lift and stride that differ from F25's record.
7. `test_turning_generate_refuses_a_look_spec` pins the refusal, so look-aware turns need their own path or option.
8. Running `prepare` on `stage2/look` for one new variant still reruns the tail checks and rewrites both JSON files
   immediately; running it for an existing variant regenerates that rung.
