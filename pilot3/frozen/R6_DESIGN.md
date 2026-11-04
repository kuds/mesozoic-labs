# Walking pilot 3: how R6 and the -2.0 reset enter the branch (DESIGN stage, 2026-11-04)

Stage folder: `SP/work/pilot3/r6_design/` (SP = `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad`).
Read only: nothing in `SP/wt_p3` or any other stage folder was written. Every command is in `CMDLOG.txt` (UTC).
The one computation this stage ran is an offline check on recorded episodes (no simulation, no training):
`scripts/r6_ref.py`, results in `out/` (section 11).

Labels: [measured: source] = read from a file or computed by this stage's script; [inferred: basis] = reasoning.
Line numbers are those of the branch head `3ae1c95967547ada569ad71aebaa88999afa02cd` (worktree clean, `git status
--porcelain` empty [measured]).

Inputs: the maintainer's decisions (8)-(11) of 2026-11-04 (`SP/work/run_log.txt`, last entries); the SCREEN stage
(`SP/work/overnight/screen/RESULT.md`, `scripts/screen.py` sha256 `52f408ce...`); its verifier
(`SP/work/overnight/verify_screen/RESULT.md`); the config-only screen and the log_std scan (`overnight/configonly`,
`overnight/logstd`); the morning report; the frozen candidate (`SP/work/pilot3/frozen_candidate/criteria_pilot3.txt`,
sha256 `6daef03d...`, 965 lines); the pinned tools (`SP/work/pilot3/tools/`).

---

## 0. The decision in one paragraph

R6 = four parts. Two are existing T. rex knobs set in `locomotion.toml` only: `airborne_substep_weight = 1.0`
(decision 9b) and `forward_vel_max = 1.5` (9c). Two are new code in `environments/tyrannosaurus_rex/base_env.py`, each
behind its own constructor key whose default is off: the compsognathus support gate on alive, posture and height
(9a; key `walk_support_gate_bw`, pilot value 0.04) and the SCREEN stage's alternating-touchdown event reward in its
probe-debounce form R5p (9d; key `walk_touchdown_weight`, pilot value 0.1). Both keys are declared in the class's
existing `OPT_IN_TASK_KWARGS`, so while off they are carved out of every task fingerprint and every recorded stage
config: no stance, recovery or behavior digest moves, and only `locomotion.toml` on this branch turns them on. The
`[curriculum] warm_start_log_std` goes from -1.5 to -2.0 (decision 8). Nothing in the physics, the observation, the
action path, the gate (decision 10) or the inputs (decision 11) changes.

---

## 1. R6 as the env will compute it (per control step)

Notation: step index m = 0, 1, ... counts completed control steps since reset (m = `self._step_count - 1` inside
`_get_reward_info`); dt = `self.dt` = 0.01 s; the per-step substep block is `self._substep_foot_force_block()`
(shape (5, 2), right foot first: pad + three digit touch sensors per foot, one row per physics substep).

1. **Support gate (9a, new).** `walk_support = float(r_contact + l_contact > walk_support_gate_bw * self._body_weight_n)`,
   with `r_contact, l_contact = self._aggregated_foot_contact_forces()` (the per-foot MIN over the step's 5
   substeps, the values the env already records as `r_foot_contact` / `l_foot_contact`, base_env.py:933-935). This is
   compsognathus_longipes/base_env.py:200 and :250 (`support = float(right + left > self._contact_threshold)`,
   threshold body_mass x |g| x 0.04) with the T. rex's own body weight: `self._body_weight_n` = 840.9132 N, threshold
   33.6365 N [measured: SCREEN npz `body_weight_n`; verify_screen RESULT "Definitions checked"; out/r6_ref_stdout.txt
   first line]. `reward_alive`, `reward_posture` and `reward_height` are each multiplied by `walk_support`, in place,
   before `info` records them (so `info` carries the paid values and the total is the sum of the recorded terms).
   The comparison is strict (`>`), as in compsognathus and SCREEN's R1. `0.04 * BW` and `BW * 0.04` are the same
   double (IEEE multiplication is commutative), so the threshold is SCREEN's to the bit.
2. **Airborne penalty (9b, existing knob).** `airborne_substep_weight = 1.0`: -1.0 x (substeps on which every foot's
   touch is <= 0.1 N) / 5, added after the 29-term sum (base_env.py:1227-1229, 1323-1324). Unchanged code.
3. **Forward cap (9c, existing knob).** `forward_vel_max = 1.5`: `reward_forward = w * clip(v / 1.5, -1, 1)`
   (shared/reward_functions.py:61-63). It caps the pay at 1.5 m/s and raises the slope below the cap by 2.5/1.5.
   It also enters `reward_backward`'s normalisation, whose weight is 0 in locomotion (no effect) [measured:
   shared/base_env.py:651-653; locomotion.toml sets no backward weight].
4. **Alternating-touchdown event (9d, new), SCREEN's R5p made causal:**
   - **Foot down** on step m: `d_i[m] = mean over the substeps of (touch_i > 0.1 N) >= 0.5`, from the same substep
     block the airborne term reads (base_env.py:1219), with the env's own `_AIRBORNE_TOUCH_N = 0.1` (the probe's
     `REPO_CONTACT_THRESHOLD_N`). This is the probe's "floor" rule (`floor_contact_substep_frac >= 0.5`) applied to
     the env's touch sensors instead of the probe's contact-force hook (section 2.3 says why, section 11 Q1 measures
     the agreement). Steps before reset read as up.
   - **Touchdown of foot i at step k** (k >= 1): `d_i[k-2] = 0, d_i[k-1] = 0, d_i[k] = 1` and
     `(d_i[k+1] = 1 or d_i[k+2] = 1)`. This is exactly the frozen probe's `_events(debounce(d_i))` touchdown set
     (MIN_RUN_STEPS 2: fill one-step gaps, then drop one-step blips; gait_probe.py:155, 173-189) for every
     k <= T-3; Appendix A derives it, test B3 checks it exhaustively. It is decided (registered) at step k+2.
   - **At registration**, touchdowns in (k, right-then-left) order, as SCREEN sorts `(k, i)`:
     `alt` = the previous registered touchdown (of either foot) was by the other foot j;
     `T_i = (k - k_i_prev) * dt`, or `k * dt` for foot i's first touchdown (timed from the reset, as SCREEN);
     `dx_i = (p_i - p_j)[:2] @ h` at the state after step k+1, where p are the `r_foot` / `l_foot` site positions
     (the probe's primary sites: the pad touch sensors' sites, MJCF lines 515-516) and h is the pelvis heading from
     the orientation sensor quaternion (`sensordata[_sensor_quat_start:+4]`, the probe's quat source and the
     `pelvis_quat` the posture term already reads, base_env.py:1038), computed with SCREEN's `fwd2d` verbatim;
     `S = clip(dx_i / (0.1 L), 0, 1)` with L = 0.8766 m (section 2.4);
     `base = min(T_i, 1.2) / (2.0 * dt) * S` (SCREEN's expression and evaluation order).
   - **Paid at step k+4**: `reward_walk_touchdown = walk_touchdown_weight * base` if `alt` and foot j has no
     registered touchdown in [k-2, k+2] (the probe's MERGE_STEPS = 2 rule for a simultaneous `B` event, two-sided as
     SCREEN's corrected R5); else nothing. At most one touchdown is paid per step [inferred: two touchdowns due at the
     same step have the same k; if they are of different feet each is "near" the other, and one foot cannot have two
     at one k; measured: the reference's `assert npay <= 1` held on all 660 recorded episodes, section 11], so the
     pay is one product, bitwise SCREEN's `W_TD * E[k]`.
   - **Why k+4 and not SCREEN's k+3.** SCREEN pays at k+3, but a touchdown of foot j at k+2 is decided only at
     step k+4 when j's step k+3 is a one-step gap (raw j: 0, 0, 1, 0 at k..k+3). k+4 is the first step at which the
     ±2 test is final in every case, so the env reads no future step. It changes nothing but (a) every pay arrives
     one control step (10 ms) later and (b) a touchdown at k = T-4 is paid by SCREEN on the episode's last step and
     never by the env [inferred: Appendix A; measured on recorded episodes: section 11 Q2, Q3].

The total: the existing 29-term sum in its order, with the three gated terms in place; then the existing optional
terms (floor impact, airborne, lateral lean) as now; then `reward_walk_touchdown` last, only when its weight is > 0
(the pattern at base_env.py:1320-1326, "added only when enabled, so the legacy sum keeps its exact order"); the fall
penalty is added by `BaseDinoEnv.step` as now.

What R6 does not change: the observation, the action mapping, the filter, reset and its RNG draws, terminations,
`_cache_ids`, the substep loop, `_substep_probe_hook`, the shared modules. The reward never enters the dynamics, so
a trajectory under R6 equals the same seed's trajectory under main's reward given the same actions [inferred: the
reward is computed after the physics and read by nothing that steps it; measured as recorded for main's reward: the
SCREEN stage's weight-0.2 runs had trajectories identical to its weight-2.0 runs].

---

## 2. Constructor keys and where the code goes (environments/tyrannosaurus_rex/base_env.py)

### 2.1 Two new keys (defaults reproduce the branch head bit for bit)

| key | default (off) | pilot value | meaning | validation |
|---|---|---|---|---|
| `walk_support_gate_bw` | `0.0` | `0.04` | > 0: alive, posture and height are paid only on steps whose summed per-foot substep-MIN touch exceeds this multiple of body weight (compsognathus rule) | finite and >= 0 (`ValueError` naming the key otherwise) |
| `walk_touchdown_weight` | `0.0` | `0.1` | > 0: the alternating-touchdown event reward (R5p, section 1 item 4) at this weight | finite and >= 0 |

- Placed in the signature after `leg_home_pose_reference` (base_env.py:279), stored with the other attributes
  (after :392), validated after the `leg_home_pose_reference` check (:454-458), in the house style of
  `airborne_substep_weight` (:443-444).
- `OPT_IN_TASK_KWARGS` (:193-196) gains two entries:
  `("walk_support_gate_bw", 0.0, ("walk_support_gate_bw",))` and
  `("walk_touchdown_weight", 0.0, ("walk_touchdown_weight",))`.
  While a switch holds its off value, `task_fingerprint._effective_env_kwargs` and `config.save_stage_config` drop
  the key (task_fingerprint.py:110-142, 216; config.py:936-943), so every stage that does not set it keeps its
  `task_sha256` and its recorded `stage_config.json` (hence `stage_config_view_sha256`) [measured by code reading;
  test A3 and the digest snapshot at commit 4 measure it]. Without these entries every T. rex stage's task digest
  and stage-config view would move (the effective config hashes every constructor default) [measured by code
  reading: task_fingerprint.py:185-208].
- The event's free parameters are SCREEN's, fixed before any SCREEN number was computed and not tuned (SCREEN
  RESULT "Limits"); they are class constants, not keys, to keep the change small: `_WALK_TOUCHDOWN_T_CAP_S = 1.2`,
  `_WALK_TOUCHDOWN_STRIDE_FRACTION = 0.1` (of L), `_WALK_TOUCHDOWN_MERGE_STEPS = 2`, `_WALK_DOWN_SUBSTEP_FRACTION
  = 0.5`, and the existing `_AIRBORNE_TOUCH_N = 0.1`. [inferred: a later change to a constant is a code change the
  task digest cannot see; the branch is pilot-only and the frozen text pins the head, which covers the code.]
- Rejected: reusing `gait_symmetry_weight` (pays the statue 997.6 of 1000, counts reset contacts and two-footed
  landings: configonly RESULT [measured as recorded]); `support_conditioned_alive_fraction` (pays alive only while
  BOTH feet carry load, ranks standing first: configonly RESULT [measured as recorded]); the gait plan's shared kit
  (`gait_phase_weight`, `environments/shared/gait_rewards.py`, GAIT_QUALITY_PLAN section 5.3): a shared-code change,
  and its P and A factors are not what was screened; its name is left free for it.

### 2.2 The code, piece by piece (about 90 lines of code plus docstrings)

1. **A small pure class at module level** (same file, before `TyrannosaurusRexEnv`), `_WalkTouchdownEvents`, no
   MuJoCo, unit-testable on synthetic sequences:
   - `__init__(self, dt: float, stride_scale_m: float)`; `reset()`;
   - `update(self, step: int, down_r: bool, down_l: bool, dx_r: float, dx_l: float) -> tuple[float, int]`
     returning (unit-weight pay delivered at `step`, touchdowns registered at `step`). `dx_r = (p_r - p_l)[:2] @ h`
     and `dx_l = (p_l - p_r)[:2] @ h` of the state after `step`, each computed as SCREEN computes it (two products,
     no sign trick).
   - State: the last 5 down flags per foot (padding before step 0 reads as up), the previous step's `(dx_r, dx_l)`
     (registration at step m needs the state after step m-1 = k+1), each foot's last touchdown step, the last
     registered foot, each foot's touchdown steps not older than m-8, the pending registrations `(k, foot, alt,
     base)` (at most 2 per step, kept 2 steps), the last step folded in and its output.
   - Step discipline (fail closed): `step == last + 1` folds the step in; `step == last` (the same step scored again
     out of band) returns the stored output and changes nothing; anything else raises `RuntimeError` naming both
     steps (a gap means the step history is broken; it cannot happen through `reset`/`step`). The env never calls it
     with `step < 0` (below).
   - Pseudocode of `update` (m = step): append `down_r`, `down_l`; k = m - 2; if k >= 1, for f in (right, left): with
     `w = d_f[k-2..k+2]` (the five stored flags), if `not w0 and not w1 and w2 and (w3 or w4)`: j = other foot;
     `alt = last_foot == j`; `t = (k - last[f]) * dt` if `last[f] is not None` else `k * dt`;
     `s = float(np.clip(dx_prev[f] / stride_scale_m, 0.0, 1.0))`; `base = min(t, 1.2) / (2.0 * dt) * s`; append
     `(k, f, alt, base)` to pending and k to `recent[f]`; `last[f] = k`; `last_foot = f`. Then for the pending entry
     with `k == m - 4`: `near = any(abs(x - k) <= 2 for x in recent[j])`; pay = base if `alt and not near`. Drop
     entries with `k <= m - 4`, prune `recent` below m - 8, store `dx_prev = (dx_r, dx_l)`.
2. **`__init__`** (state before `super().__init__`, like `_init_gait_state` at :499):
   `self._walk_touchdowns: _WalkTouchdownEvents | None = None`. After `super().__init__(...)` (:511-529), only when
   `walk_touchdown_weight > 0`: compute L (section 2.4) and build
   `_WalkTouchdownEvents(self.dt, _WALK_TOUCHDOWN_STRIDE_FRACTION * leg_length)` (SCREEN: `ell = ELL_FRAC *
   leg_length`, the same product). Nothing is computed when off.
3. **`_spawn_target`** (:1378-1394; called by every `reset`, the T. rex's own method, not hashed): after
   `self._reset_gait_state()`, `if self._walk_touchdowns is not None: self._walk_touchdowns.reset()`.
4. **`_get_reward_info`** (:910-1329):
   - after :935 (`info["l_foot_contact"] = l_contact`), when `walk_support_gate_bw > 0`:
     `walk_support = float(r_contact + l_contact > self.walk_support_gate_bw * self._body_weight_n)`;
     `info["walk_support"] = walk_support`;
   - alive (:956-959): after `reward_alive = raw_alive * alive_gate`, when on: `reward_alive = reward_alive *
     walk_support`, then `info["reward_alive"]` as now (`raw_alive` and `alive_gate` stay as recorded);
   - posture (:1039-1041): after `_compute_posture_reward`, when on: `reward_posture = reward_posture *
     walk_support`;
   - height (:1063-1082): after both branches, before `info["reward_height"]`, when on: `reward_height =
     reward_height * walk_support`;
   - event, in 9b after `foot_force_block` and `airborne_substeps` (:1219-1221), when `self._walk_touchdowns is not
     None`: `m = self._step_count - 1`; if `m < 0` (no step since reset: the constructor's and reset's own reads,
     the digest harness's posed-state probes, which call `_get_reward_info` directly after `reset`:
     digest_snapshot.py:584-591): pay 0, nothing registered, the tracker untouched. Else
     `down = (foot_force_block > self._AIRBORNE_TOUCH_N).mean(axis=0) >= _WALK_DOWN_SUBSTEP_FRACTION` (the probe's
     `(fl > 0.1).mean(axis=0) >= 0.5`), the two foot sites' positions, `h = _screen_heading(pelvis_quat)` (SCREEN's
     `fwd2d` on a (1, 4) array: `v = [1 - 2(y^2 + z^2), 2(xy + wz)]`, divided by `max(norm(v, axis=-1), 1e-9)`;
     the shared `quat_to_forward_2d` normalises through a different norm call and a 1e-6 guard, so it is not used),
     then `pay, n = self._walk_touchdowns.update(m, down[0], down[1], dx_r, dx_l)`;
     `reward_walk_touchdown = self.walk_touchdown_weight * pay`;
   - total (:1320-1326): after the lateral lean line, `if self.walk_touchdown_weight > 0.0: total_reward +=
     reward_walk_touchdown`.
5. Docstrings: the module docstring's reward list (:35-65) gains the two items; each new block gets a comment naming
   the pilot ("walking pilot 3, pilot-only, never merged"), decision 9, the SCREEN evidence and the k+4 reason.

Nothing else in the repository changes for R6: not `shared/base_env.py`, not `shared/diagnostics.py` (see 3.2), not
the probe, not the plant contract, not any other species.

### 2.3 Touch sensors, not a floor-truth hook

The env has no per-foot floor-truth force today; the probe gets it from `_substep_probe_hook` (gait_probe.py:537-562),
a slot the env must not take (shared/base_env.py:197, 1012-1018), and the documented species hook for a per-substep
reward quantity is `_accumulate_substep` (shared/base_env.py:1008-1018). A floor-truth version would add a Python loop
over `data.contact` with `mj_contactForce` at every physics substep (the cost the step loop's vectorised strike scan
was written to avoid, "~50 us per substep", shared/base_env.py:1297-1299) and more code. The touch block already
exists, is what the airborne term reads, and its summed value equals floor truth to 1e-12 N on the recorded r8 stance
panels (base_env.py:1208-1214 comment) [measured as recorded]. Per foot, a touch volume can also see a non-floor
contact (one sole on the other: gait_probe.py:695-698). Section 11 Q1 measures the per-foot majority-substep
agreement on 660 recorded episodes; if a later stage finds a disagreement that matters, the fallback is the
`_accumulate_substep` version, not a change of the rule.

### 2.4 The leg length L

S's length scale is 0.1 L with SCREEN's L = 0.8766 m, the probe's: the mean world z of the `hip_pitch` joint anchors
at the home keyframe (gait_probe.py:405-450: `mj_resetDataKeyframe(m, data, m.key("home").id)`, `mj_forward`, `z =
data.xanchor[j][2]` for `r_hip_pitch` and `l_hip_pitch`, `float(np.mean(...))`). The env computes it the same way on
a scratch `MjData` in `__init__` after `super().__init__`, only when the event is on (never in `_cache_ids`, which the
plant contract hashes: policy_layer.py:551). Test B7 pins it bitwise to `RobotDescription(env).leg_length`
(0.8765999999999999 in the recorded episodes [measured: verify_screen npz `leg_length`]). The stance and locomotion
envs never swap their compiled model; the behavior env (terrain) does, but the event is off there.

---

## 3. What R6 records

### 3.1 Info keys (each exists only while its switch is on, so an off env's info dict is the branch head's)

| key | when | value |
|---|---|---|
| `walk_support` | gate on | 1.0 or 0.0, this step's flag |
| `reward_alive`, `reward_posture`, `reward_height` | always (existing) | the paid (gated) values; `raw_alive` and `alive_gate` stay the ungated inputs |
| `r_foot_down`, `l_foot_down` | event on | 1.0 or 0.0, this step's majority-substep touch flags |
| `walk_touchdowns_registered` | event on | 0, 1 or 2 (float): touchdowns registered this step (at k = this step - 2) |
| `walk_touchdown_unit` | event on | the unit-weight pay delivered this step (`base`, or 0.0) |
| `reward_walk_touchdown` | event on | `walk_touchdown_weight * walk_touchdown_unit` |
| `reward_airborne_substeps`, `reward_forward` | always (existing) | now non-zero / capped at 1.5 m/s by the config |

The forfeit on an unsupported step is recoverable offline: `raw_alive * alive_gate * (1 - walk_support)` for alive,
and the posture and height inputs (`tilt_angle`, `height_quality`) are recorded as now.

### 3.2 Where they show up (no shared code edited)

- The in-training evaluation ledger sums every `reward_*` key by prefix (shared/eval_diagnostics.py:1309-1319), so
  `eval_episodes.csv` gains the `reward_walk_touchdown` sum for locomotion with no code change; the re-smoke checks
  that `pilot3_habit.py` still finds every column it reads.
- `DiagnosticsCallback.REWARD_KEYS` (shared/diagnostics.py:111-157) and visualization's weight map are NOT edited:
  adding a key there records an all-NaN series in every other run's `diagnostics.npz` (the lateral-lean comment,
  base_env.py:1184-1187, and diagnostics.py:283-289 `_safe_mean` of an empty list), which would put a shared change on
  the stance path. The pilot reads the term from the ledger and the probe. [inferred; a deliberate pilot-only
  omission, recorded in the commit message.]
- The probe's rollout records every `reward_*` key (gait_probe.py:603-609); its outcome metrics read only contact and
  kinematics, so judging is unchanged.

---

## 4. locomotion.toml on this branch (pilot values)

`[env]` (each line with a `# WALKING PILOT 3 (pilot-only, never merged; ...)` comment naming decision 9 of 2026-11-04
and the SCREEN evidence, in the style of the `[curriculum]` pilot keys):

```toml
forward_vel_max = 1.5                  # was 2.5 (decision 9c)
airborne_substep_weight = 1.0          # was unset (env default 0.0) (decision 9b; the stance stage's own value)
walk_support_gate_bw = 0.04            # new key (decision 9a)
walk_touchdown_weight = 0.1            # new key (decision 9d)
```

`forward_vel_max`'s existing comment (locomotion.toml:13-16, "saturate just past the 2.0 m/s gate") is replaced by
the pilot comment (the gate is 1.0 m/s and stays: decision 10; SCREEN: caps at or below 1.4 conflict with the
episode-mean gate, 1.5 does not [measured as recorded: run log 07:46, configonly RESULT]).

`[curriculum]`: `warm_start_log_std = -2.0` (written `-2.0`, not `-2`: the TOML integer `-2` would be recorded as
`-2` in `curriculum.warm_start_log_std` and hash differently in the recipe digest's canonical JSON, while
`declared_warm_start_log_std` returns `float` for the run record; gate_schema.py:538-556 [measured by code reading]).
Its comment: decision (8) of 2026-11-04 replaces decision (c)'s -1.5; std exp(-2.0) = 0.1353; the log_std scan: the
stochastic parent stands 359/360 at -2.0 against 9/120 at -1.5 [measured as recorded: logstd RESULT, run log 06:06].

Unchanged: every gate key (decision 10), `collapse_peak_floor_reference = 1092.6` and its pair (a collapse key; see
section 7 V4), `[ppo]`, every other `[env]` key, the cadence keys.

---

## 5. Digests

`hyperparameters_sha256` (the recipe digest) hashes the algorithm table plus the `warmup_`, `ramp_` and `warm_start_`
curriculum keys only (config.py:638, 667-686); env keys never enter it. `task_sha256` hashes the effective `[env]`
(constructor defaults + TOML, opt-ins carved out while off). `stage_config_view_sha256` hashes the recorded
`stage_config.json` (minus run-specific keys: digest_snapshot.py:142-145, 387-401), so it moves with either. The
digest snapshot's `reward` lines run each stage's env through scripted actions and posed states
(digest_snapshot.py:458-560).

Expected, to be measured by `digest_snapshot --block-optional-backends --check` at each commit (section 9):

| commit | moves (all `tyrannosaurus_rex anatomical locomotion`) | must not move |
|---|---|---|
| 4 (code, off) | nothing | every line (649), incl. every T. rex stance/recovery/locomotion/behavior line, the plant and policy lines |
| 5 (R6 on) | `task_sha256` (5f835d65... -> new), `stage_config_view_sha256.PPO/.SAC`, `reward summary`, `reward shape_sha256` (new info keys), `reward rounded_values_sha256` (`reward poses` may move only if an end changes) | `hyperparameters_sha256.PPO/.SAC`, `gate_sha256` 02602cb0..., every non-locomotion line |
| 6 (-2.0) | `hyperparameters_sha256.PPO/.SAC` (8263f82f... / a958a8fb... -> new), `stage_config_view_sha256.PPO/.SAC` | `task_sha256` (commit 5's), `gate_sha256`, the reward lines, every non-locomotion line |

`plant_contract` and `species_catalog` checks are expected to stay current at every commit [inferred: no plant,
interface, MJCF or budget change; `_cache_ids`, `_get_obs`, `_scale_action`, `reset` and the filter are untouched;
the source closure covers the MJCF's dependencies only, plant_contract/manifest.py:100-101; measured at each commit
by the implement stage].

The judge derives the recipe digest from the tree (`pilot3_judge.py:372, 400`), so it needs no new constant for it;
the new PPO digest goes into the criteria text's first-hour check (section 10).

---

## 6. Tests (one new file, `environments/tyrannosaurus_rex/tests/test_anatomical_walk_r6.py`, plus two small fixtures)

Module docstring names the pilot, "pilot-only, never merged", decision 9 and the SCREEN stage. Fixed seeds; no test
longer than a few seconds except B3's exhaustive set and C1's two rolled episodes (under 30 s together on this CPU
[inferred: 1,000 env steps take about 1-2 s here]).

**A. Off is the branch head, bit for bit**
- A1 `test_the_r6_keys_default_to_off_and_only_locomotion_sets_them`: signature defaults 0.0 and 0.0; of the
  committed T. rex stages and behavior recipes only `locomotion` sets them (0.04, 0.1), and only on this branch (the
  docstring says so: on main the set would be empty).
- A2 `test_off_r6_changes_no_reward_no_info_key_and_no_trajectory`: 240 steps (seeded uniform actions, amplitude
  0.05, seeds 3042 and 3043) of (i) the stance config, (ii) the locomotion config with the four R6 lines reverted to
  the head's values (`forward_vel_max` 2.5, `airborne_substep_weight` 0.0, both new keys absent), each against the
  same config with the new keys explicitly 0.0: rewards, observations, terminated/truncated and every info key and
  value equal (`==`), and no R6 info key present. (The precedent test_anatomical_lean_and_settled_legs.py:68-75.)
- A3 `test_off_r6_moves_no_task_fingerprint_and_no_recorded_config`: for every committed T. rex stage except
  locomotion, `stage_task_fingerprint` with the keys explicitly 0.0 equals the committed one and the env section
  lacks them; `save_stage_config` records neither key while off and both (with defaults) when on; turning either on
  moves the fingerprint, and the two on-values give different fingerprints. (Precedent ibid. :78-123.)
- A4 `test_off_r6_is_the_branch_head_bitwise` (the golden): a JSON fixture
  `tests/fixtures/walk_r6/off_golden_3ae1c959.json` holding, per scenario, sha256 digests of the per-step
  observation (float32 bytes), reward (float64), terminated, truncated and the sorted `repr` of every info item,
  recorded on a clean clone of `3ae1c959` by a generator script the implement stage keeps in its folder and names in
  the commit message. Scenarios: the stance config, the recovery config (pushes), the locomotion config as at
  `3ae1c959` (the new file's `[env]` minus the four R6 lines), a default-constructed env; seeds 3042 and 3043; 300
  steps of seeded uniform actions plus 100 zero-action steps. Compared bitwise in the recording environment (Python,
  platform, numpy, MuJoCo, gymnasium versions and the CPU capability, recorded in the JSON); elsewhere it skips and
  names the fields that differ, as test_critic_phase_off_path.py does (:9-11, :97-110).
- A5 `test_invalid_r6_values_fail_fast`: -0.1, inf and nan for each key raise `ValueError` naming the key.

**B. Units on synthetic sequences (pure, no MuJoCo except B7)**
- B1 `test_the_support_flag_is_strict_and_reads_four_percent_of_body_weight`: with `_foot_contact_forces`
  monkeypatched (the precedent TestStageOneStanceRewardPrimitives, test_anatomical_rewards.py:309-330) and
  `_get_reward_info` called after reset: (16.0, 17.0) N -> flag 0, the three terms 0.0 (posture -0.0 allowed),
  every other term equal to the gate-off env's; (20.0, 20.0) -> flag 1, all equal to gate off; the threshold is
  `0.04 * env._body_weight_n` exactly; a sum equal to the threshold is unsupported.
- B2 `test_a_foot_is_down_on_at_least_three_of_five_substeps`: 3 of 5 rows > 0.1 N -> down, 2 of 5 -> up, exactly
  0.1 N -> up; a one-row block (no aggregate) works.
- B3 `test_registration_is_the_probes_debounced_touchdown_set`: for every binary sequence of length 4-12 (8,176
  sequences) and 2,000 random sequences of length 200 with flicker, the tracker's registered steps (one foot, the
  other always up) equal `_events(debounce(seq))[0]` restricted to 1 <= k <= len-3, with `debounce` and `_events`
  loaded from `docs/investigations/gait_2026_09/gait_probe.py` by path (importlib, read only).
- B4 `test_the_event_rule_on_hand_written_gaits`: (a) a clean alternating walk (R and L each down 30 steps, up 20,
  half a cycle apart, dx 0.2 m) pays every touchdown after the first two exactly `min(T_i, 1.2) / (2 dt)` at k+4
  (the first of each foot timed from reset); (b) both feet landing within 0, 1, 2 steps pays nothing for either, 3
  steps apart pays the later; (c) the same foot twice in a row is not paid; (d) dx <= 0 pays 0, dx = 0.04383 m pays
  half, dx >= 0.08766 m pays the full base; (e) T_i above 1.2 s is capped; (f) a one-step bounce before a landing
  (down, up, down, down) is one touchdown at the first step (the probe's gap fill); a one-step blip (up, down, up,
  up) is none; (g) k = 0 is never a touchdown; (h) a touchdown whose pay step is at or beyond the episode end is
  never paid; (i) `reset()` forgets everything.
- B5 `test_the_tracker_is_screens_events_probe_one_step_later`: SCREEN's `events_probe` ported verbatim into the test
  with its pay delay as a parameter (3 = SCREEN); on 500 random two-foot sequences (length 300, random dx in [-0.1,
  0.2]) the tracker's per-step pays equal the port's with delay 4 (`==`, every step), and its total equals the
  port's with delay 3 minus the pays of touchdowns at k = T-4.
- B6 `test_the_tracker_reads_no_future_step`: on random sequences, changing every flag after step m changes no pay at
  or before m; and `test_rescoring_a_step_is_idempotent_and_a_gap_fails`: `update(m)` twice returns the same and
  changes nothing; `update(m + 2)` raises; through the env, `_get_reward_info` after `reset` (m = -1) registers
  nothing and pays 0, and calling it twice in one step gives the same info.
- B7 `test_the_stride_scale_is_a_tenth_of_the_probes_leg_length`: `env._walk_touchdowns` built with
  `0.1 * RobotDescription(env).leg_length` bitwise, L = 0.8766 m to 1e-4.

**C. Integration**
- C1 `test_the_env_r6_terms_equal_an_offline_recompute_of_its_own_episode`: the locomotion stage env as committed
  (R6 on), two seeded open-loop action patterns that produce touchdowns (the smoke's scripted alternating pattern
  and its two-leg hop pattern, smoke/scripted/scripted_rollout.py, grid 2), each rolled to termination or 300 steps,
  recording `info`, `_substep_foot_force_block()` (a fresh copy), the two foot sites and the pelvis quaternion after
  every step; the offline side is a whole-episode, non-causal recompute (the probe's `debounce` and `_events`, the
  ported `events_probe` with delay 4, SCREEN's `fwd2d`) and SCREEN's R1 flag from `r_foot_contact + l_foot_contact`:
  `reward_walk_touchdown`, `walk_support`, `r_foot_down`, `l_foot_down` equal per step (`==`), and each step's
  reward equals the env-order sum of its recorded terms plus the fall penalty (`==`).
- C2 `test_r6_returns_equal_screens_offline_r6p_on_recorded_hand_built_episodes`: fixture
  `tests/fixtures/walk_r6/screen_cases.npz`, built by the implement stage (script in its folder, sha256 in the commit
  message) from the SCREEN stage's own recorded npz for seed 3042 of FINAL3, V24, V18, V35, W02, H23, H20 and the
  statue: per step the floor-truth down flags (`floor_frac >= 0.5`, uint8), dx_r and dx_l computed as `screen.py`
  does (float64), `r_foot_contact + l_foot_contact`, the recorded `reward_alive + reward_posture + reward_height`,
  `airborne_substeps` (uint8), `forward_vel`, the env's main reward and fall; plus SCREEN's published R6p returns at
  w 0.2 and 2.0 (`screen_main.json`) [about 45 bytes per step, about 0.36 MB raw for 8 episodes; inferred]. The test
  drives the env's own pure pieces (the tracker with the floor flags, the support threshold of a constructed env,
  the forward term at 1.5) and asserts each episode's R6 return equals SCREEN's published R6p within 1e-9 after
  removing the pays SCREEN delivers for touchdowns at k = T-4 (recomputed in the test with the ported
  `events_probe`); the per-step touchdown pays equal SCREEN's shifted by one step (`==`). The live version, with the
  env's own touch flags on re-rolled episodes, is the validate stage's V2 (section 7).

**D. Existing suites.** The shared suite and the T. rex suite are run at each new commit. At commit 4 nothing new
may fail. At commits 5 and 6 any test that fails only because the committed locomotion config now sets R6 or -2.0
(pins main's locomotion values, e.g. the 14 ids already deselected, any test reading locomotion's task digest or the
digest snapshot before regeneration) is deselected by id with its cause, exactly as commit 2 did; any other failure
is a bug. Known pins found by grep: none in the T. rex suite reads locomotion's reward values or task digest; the
digest snapshot golden is regenerated with `--write` [measured: grep for 5f835d65, 1561b27a, 228.9660 finds only
the golden and one stance-5 investigation note]. ruff, ruff format --check, mypy (with and without SB3 and torch)
and `notebook_cells.py` clean at each commit.

---

## 7. What the tests cannot show (the validate stage's list, CPU, one simulation at a time)

- **V1 two-tree off-path harness:** a clean clone at `3ae1c959` and one at the new head (commit 4, and again at the
  head); stance, recovery, locomotion-as-at-3ae1c959 and two behavior recipes; 3 seeds x 1,000 steps of seeded
  random and zero actions: observations, rewards, terminations and every info item bitwise equal; reset draws equal;
  `digest_snapshot --exact` lines equal for every non-locomotion stage.
- **V2 SCREEN equality, live:** re-roll SCREEN's hand-built behaviours (the 6 stable walks, the 9 F hops, statue,
  parent, W02; seeds 3042-3061, and the held-out panel's subset) in the R6 env at the new head through the frozen
  probe's rollout (its floor-truth hook): trajectories (`forward_vel`, foot sites) bitwise equal to SCREEN's
  records; `reward_walk_touchdown` per step equal to the ported `events_probe` (delay 4) on the env's own touch
  flags; each episode's R6 return equal to SCREEN's published R6p up to the documented differences (touchdowns at
  k = T-4; touch-versus-floor flag differences, each listed; summation order, <= 1e-9); every R6p verdict of
  SCREEN's table (worst walk against standing and against its matched hop, both weights, every seed) reproduced
  from the env's own returns.
- **V3 the parent at -2.0 under R6:** the stochastic parent (log_std -2.0 in memory, as the log_std scan) in the R6
  locomotion env, seeds 3042-3081: falls, return, gate forfeits, registered and paid touchdowns (the -1.5 parent's
  micro-lifts earned 68 unit-weight per episode under R5 [measured as recorded: SCREEN RESULT "What it pays"]).
- **V4 the statue under R6:** `zero_action_baseline.py --model-id tyrannosaurus_rex anatomical --stage 2` on the new
  head (the command in locomotion.toml:113-115). SCREEN's offline R6 statue is 1091.4 at w 2.0 against main's 1092.7
  (seeds 3042-3061) [measured as recorded: SCREEN RESULT tables]. Recommendation: keep
  `collapse_peak_floor_reference = 1092.6` (decision 10 keeps the gate; the floor is 0.45 x the reference, so a 1.3
  change moves it by 0.6) and record the measured value beside it in the criteria; moving it is a gate-side change
  the maintainer did not decide.
- **V5 the re-smoke** (the pre-registered local CPU smoke, re-run at the new head): the second Warm start line
  (section 8), `stage_config.json` (the 17 checks with -2.0, plus `reward_weights` recording `forward_vel_max` 1.5,
  `airborne_substep_weight` 1.0, `walk_support_gate_bw` 0.04 and `walk_touchdown_weight` 0.1), the ledger's columns
  (`reward_walk_touchdown` added, every column `pilot3_habit.py` reads present), train/std near 0.135, the judge
  validation J1-J10 with the re-pinned judge, the video tool, the RESUME check.

---

## 8. The -2.0 reset

- Config: `[curriculum] warm_start_log_std = -2.0` (section 4). The code path is PR #658's, unchanged
  (`train_base._apply_warm_start_log_std`, :324-366).
- The expected second Warm start line, from the format string `{value:g}` and `{math.exp(value):.3f}`
  (train_base.py:360-363): `Warm start: policy log_std set to -2 (std 0.135) on all 15 actuators (the parent's std:
  mean 0.057, min 0.007, max 0.111; Adam moments kept)` [inferred from the code: `format(-2.0, "g")` is `-2`; the
  re-smoke measures it string-equal]. The first line (the drop) is unchanged.
- Records: `run.warm_start_log_std` -2.0 (float) and `curriculum.warm_start_log_std` -2.0.
- Ratios for the text (from the printed parent values 0.057 / 0.007 / 0.111): exp(-2) = 0.1353, about x2.4 on the
  mean, x19 on the smallest, x1.2 on the largest (at -1.5: x3.9, x32).
- `pilot_checkpoints/stance7_r10LTSBpc_s42_final/SOURCE.txt` lines 20-21, 63 and 68 (and the x3.9/x32 remark at
  :92) state -1.5; they are updated in commit 6 (line 45, the parent's own `log_std_init -1.5` from its stance
  training, stays).

---

## 9. Commits (local `claude/pilot-walk-r10-p3`, on `3ae1c959`; never pushed here)

Each title ends "(walking pilot 3, pilot-only, never merged)"; each message ends with exactly the two lines
`Co-Authored-By: Claude <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_01Y8Bg5B54ULHsgvHfZKSkRr`.

4. "Add a default-off R6 walking reward to the T. rex env": base_env.py (section 2), the new test file and its two
   fixtures. Digest snapshot unchanged; suites pass with the existing 14 deselections and nothing new.
5. "Turn R6 on in the T. rex locomotion stage": locomotion.toml `[env]` (four lines), the regenerated digest
   snapshot (moved lines as section 5, listed in the message), deselections if any (section 6 D).
6. "Reset the warm-start log_std to -2.0": locomotion.toml `[curriculum]` (one value and its comment), SOURCE.txt
   (section 8), the regenerated snapshot (moved lines as section 5).

Three commits keep each digest move attributable to one cause. The head changes; the criteria's head, commit list
and every head-derived check follow it (section 10).

---

## 10. The pre-registration text (criteria_pilot3.txt) and the tools

### 10.1 Sections of criteria_pilot3.txt that must change (line numbers at sha256 6daef03d...)

1. Title and status (1-12): a revision note (R6 and -2.0, 2026-11-04, the amendment file beside it).
2. Maintainer decisions (18-43): add (8)-(11) of 2026-11-04 with the run-log entry; mark (c) as replaced by (8);
   (d) keeps the declared exception to GQ-3 (a) and GQ-17 (the reward is now neither today's nor gait-r1's: a
   pilot-only R6), and the order-of-work departure gains commits 4-6.
3. Changes against pilot 1 (45-75): C1's "launch check reads main's r10 locomotion task digest 5f835d65..." becomes
   the branch's new task digest; C3 adds the R6 code; C5's value becomes -2.0; a new item for R6 (the four parts,
   off-by-default code, SCREEN and verify_screen evidence, k+4, touch flags).
4. Code (104-118): commits 4-6 with full hashes; the head.
5. What is trained (119-158): "main's locomotion [env] reward" -> main's with R6 (the four `[env]` lines); the
   `[curriculum]` list's `warm_start_log_std = -2.0` (std 0.1353); the second expected Warm start line (section 8),
   re-measured by the re-smoke (the "No amendment of the two lines is needed" paragraph is replaced).
6. How it is judged (196-306): the judge's description ("-1.5" -> "-2.0") and every changed sha256 (10.2); a sentence
   that the probe runs the R6 env and its outcome metrics read no reward.
7. Run handling (422-470): the head in the "Repository ref ... commit:" line; the first-Eval note's statue figure
   (V4's R6 statue beside 1092.60); the first-hour checks: -2.0 twice, the new hyperparameters_sha256 and
   task_sha256, and the four R6 `reward_weights`.
8. What each outcome means (471-538): "the reset to 0.2231 ... on today's walk reward" -> 0.1353 on the R6 walk
   reward; HOPS (the hop is learned despite R6's flight penalty and gate, against SCREEN's ranking); STILL ("noise
   near 0.22" -> 0.135; the forward term is on a 5/3 steeper scale below 1.5 m/s, so pilot 1's 0.5-3.2 is compared
   after scaling by 0.6, or as forward_vel); FALLS (x3.9 / x32 -> x2.4 / x19).
9. The locomotion gate (539-543): unchanged by decision 10; one sentence that R6 does not change it.
10. Report only (596-638): noise against 0.1353; the forward-term comparisons on the new scale; the hidden-clock
    audit must change: R6's touchdown term reads unobserved history (the time since that foot's previous touchdown;
    a foot's first touchdown is timed from the reset, capped at 1.2 s = 120 steps, so it reads the episode step in
    [0, 120)), and pays 4 steps after the touchdown; the expectation "no step-localised critic error" becomes
    "possible structure in [0,100) and [100,200)" [inferred]; still report only.
11. Known limits (639-667): one noise value (-2.0); R6 screened offline on 21 fixed behaviours, how PPO responds is
    unmeasured; gating posture forgives tilt in the air (SCREEN's R1np differs by at most 3 per episode [measured as
    recorded]); touch flags against SCREEN's floor truth (section 11 Q1); pays one step later than SCREEN; the
    re-smoke's train/std.
12. Pre-launch checks (668-938): the tree (commits 1-6), moved digest lines per commit (section 5), the deselect
    list re-measured, the R6 validation (V1-V4), the re-smoke (V5), the re-pinned tool hashes and final-run counts.
13. Recorded exception text (939-953): -1.5 (0.2231) -> -2.0 (0.1353); "with a pilot-only walking reward R6
    (support-gated alive, posture and height; airborne penalty 1.0; forward_vel_max 1.5; alternating-touchdown event
    0.1)"; the head.
14. The September walkers' table (955-965): a note that its forward-term column is on the 2.5 m/s scale.
Unchanged: Parent, Seeds and runs, Budget, Episode labels, the probe, Video review, seed and pilot outcome rules,
Habit's rules. BRANCH_PLAN.md (commits 4-6, the deselect list, the "Not on the branch" rows for the judge and the
noise tool) and NOTES.md (its I6 "task and gate unchanged" now holds for commit 6 only) follow.

### 10.2 Tools that hard-code -1.5 (and must be changed and re-pinned)

Grep of `SP/work/pilot3/tools` [measured: CMDLOG 12:05:4xZ-12:06:46Z]:
- `pilot3_judge.py:85` `WARM_START_LOG_STD = -1.5` (and the docstring :38-39) -> -2.0. The head is not hard-coded
  (`--expect-head`, :345, :366); the recipe digest is derived from the tree (:372, :400); the parent sha256 is
  unchanged. Its tests `tests/test_pilot3_judge.py` :39, :44, :65, :84, :347, :377 (records and asserts of -1.5;
  keep -1.5 as a refused value) and `tests/e2e_parent_cell.py` :4, :85, :105.
- `pilot3_noise.py:39` `WARM_START_STD = math.exp(-1.5)` (and :8) -> `math.exp(-2.0)`; `tests/test_pilot3_noise.py:130`
  (0.2231 -> 0.1353).
- Value-agnostic, no change needed: `tests/test_nb_pilot_runner3.py:141` (a synthetic record), `tests/
  test_pilot3_habit.py:305` (a synthetic head), `test_pilot3_judge.py:279-284` (commit-prefix cases).
- No hit: `pilot3_score.py`, `pilot3_habit.py`, `pilot3_vg.py` (its G uses the env's reward, so R6 automatically),
  `pilot3_video.py`, `iso.py`, `gait_probe_pilot.py`, `nb_pilot_runner3.py` (prints the keys, no value).
- The tools' tree clone (`SP/work/pilot3/tools/tree`, at `3ae1c959`) moves to the new head, since
  `tests/_tree_facts.py` reads the tree's recipe digest and curriculum.
- Re-pin `SHA256SUMS.txt` (73 entries) after a final run, keeping the old list as `before/`.

---

## 11. Offline check of this design on recorded episodes (`scripts/r6_ref.py`) [measured]

What it is: a reference implementation of section 1 (the script's `Tracker` class is the tracker of section 2.2,
taking foot sites and quaternions instead of precomputed dx), run over the verify_screen stage's recorded re-rolls of
SCREEN's behaviours (`SP/work/overnight/verify_screen/out/runs/`: 660 episodes, panel A = 20 behaviours x seeds
3042-3061, panel B = 13 behaviours x seeds 3062-3081; each with per-step info, the env's own per-substep touch block
`_substep_foot_force_block()`, the probe's floor-truth fractions, foot sites and pelvis quaternion), against
`screen.py`'s `events_probe` ported verbatim (pay delay 3 as SCREEN, and 4) and SCREEN's published R6p returns
(`screen_main.json`, `screen_HO.json`, w_td 0.1). No simulation; `nice -n 10`, `OMP_NUM_THREADS=1`, TMPDIR in this
folder. [measured: out/r6_ref_stdout.txt, out/r6_ref_rows.json, out/r6_verdict_stdout.txt]

- **Reproduction first.** Main's per-step reward rebuilt in the env's order equals the recorded `env.step` reward on
  651,967 of 651,967 steps; the recomputed SCREEN R6p equals the published value on all 1,280 episode-weight pairs
  (difference 0.0). Body weight 840.9132000000001 N, threshold 0.04 x BW = 33.636528000000006 N.
- **Q1, touch flags against floor truth.** The env's majority-substep touch flag and the probe's majority-substep
  floor flag disagree on 8 of 1,303,934 foot-steps, in 2 of 660 episodes (one V29, the 1.5 m/s walk with landing
  bounces; one V33, the failed fast walk). The event pay differs on 1 step of 651,967 (V33 seed 3043: +1.0 unit,
  +0.1 at w_td 0.1). No stable walk, hop, standing or stepping-in-place episode has a disagreement.
- **Q2, the causal rule against SCREEN's.** On floor flags, the tracker's per-step pay equals `events_probe` with pay
  delay 4 on every step of all 660 episodes (0 steps differ). SCREEN pays 9,455 touchdowns, the env rule 9,443; the
  12 missing are exactly SCREEN's pays for touchdowns at k = T-4 (W02 7, V35 2, V33 2, V29 1). Every other touchdown
  is paid the same amount one step later.
- **Q3, how often SCREEN's k+3 pay used a later step.** 1 of SCREEN's 9,455 paid touchdowns (a V29 episode) had the
  other foot's 0, 0, 1, 0 pattern at k..k+3, so its pay at k+3 depended on step k+4. Rare in these behaviours, but
  a learned policy's flicker is unknown, which is why the env pays at k+4. [count measured; reason inferred]
- **Q4, returns.** The env-order R6 (gated terms in place, then airborne, then the event; fall last) against SCREEN's
  expression with the k+4 pays: at most 9.1e-13 per episode and 2.8e-14 per step (summation order only). Against
  SCREEN's published R6p, 1,254 of 1,280 episode-weight pairs agree within 1e-9; the other 26 are the 12 episodes
  with a k = T-4 pay (x 2 weights; -0.22 to -2.5 per episode) and V33 seed 3043 (x 2; +0.1, the touch flag).
- **SCREEN's verdict under the env rule.** Every stable walk is above the best of statue, parent and W02 and above
  its speed-matched hop on every seed at w 0.2 and 2.0, with SCREEN's own per-seed minima in all three columns
  (published R6p, env rule on floor flags, env rule on touch flags): panel A +93.2 over standing and +969.1 over the
  hop at w 0.2, +309.6 and +823.3 at w 2.0; panel B +89.7 and +957.3, +307.3 and +782.8 (walk-seed pairs above
  standing 120/120 and 80/80 per weight; above the matched hop 120/120 and 60/60).
- **Limits.** These are fixed behaviours re-rolled once (SCREEN's own limits apply: what PPO learns is not measured);
  the flags come from recorded touch blocks, so the live env path (the env computing them during `step`) is the
  validate stage's V2.

---

## 12. Risks and open points for the maintainer

- The event term pays history the policy cannot observe (each foot's time since its previous touchdown); the policy
  has no memory. SCREEN accepted this for R5p; it is the gait plan's GQ-15 (a) form. [inferred]
- The first touchdown of each foot is timed from the reset (SCREEN's rule): within the first 1.2 s the pay grows with
  the episode step. Report only (hidden-clock audit). [inferred]
- SCREEN's tie order (right before left at the same step) makes the rule not exactly mirror-symmetric after a
  simultaneous landing (the next touchdown's alternation depends on which foot was processed last). Inherited
  unchanged; it affects only the touchdown after a simultaneous pair. [inferred: from the rule]
- Gating posture forgives tilt while airborne (decided: compsognathus's rule). [measured as recorded: SCREEN R1np]
- A policy that taps in place with feet landing ahead earns the event term (W02 earned 80.5 unit per episode; S keeps
  it small). [measured as recorded: SCREEN]
- Touch versus floor truth: section 11 Q1.

---

## 13. Slips (this stage)

1. **CMDLOG literalness.** Several logged lines are not the literal command that ran: shell variables defined at
   the start of each command are not logged; the first six lines (11:58-12:00Z) were typed after the fact with
   approximate times; twelve read-only greps and seds were logged with a different pattern or range than ran, and
   three seds ran with no line of their own. A NOTE in `CMDLOG.txt` (12:35Z) lists each one with what actually ran,
   and a further NOTE corrects that list's own backslashes.
2. **A wrong typed time.** A NOTE line (the Write of `scripts/r6_ref.py`) carries a typed time of 12:31:00Z; the
   real time was about 12:11-12:12Z. Corrected by a NOTE at 12:24:45Z.
3. **A long run.** `scripts/r6_ref.py` took about 17 minutes instead of the minutes expected, because it re-read
   compressed npz members inside its per-step loop; the shell tool moved it to the background at its 600 s limit.
   It wrote only to `out/`, ran under `nice -n 10`, `OMP_NUM_THREADS=1` and TMPDIR in this folder, and was the only
   computation running (no simulation). Read-only greps ran meanwhile.
4. **Text fixed after drafting.** Before this file was final: the episode count in section 2.3 (675 -> 660: the
   verifier's 675 include its X panel, which the check did not read), five cited line numbers in the probe and the
   base env, one label ("measured" -> "measured as recorded") and one source were corrected (`scripts/edit1.py`,
   `edit2.py`, one logged `sed -i`).
No simulation, training, write outside this folder, push or config change.

---

## Appendix A. The probe's debounce as a causal rule

`debounce` (gait_probe.py:173-182) on a 0/1 array b of length T: pass 1 sets every interior 0-run of length 1 to 1
(b1); pass 2 sets every interior 1-run of length 1 of b1 to 0 (b2); "interior" = not touching index 0 or T-1.
`_events` reports k where b2 goes 0 -> 1. For 1 <= k <= T-3:
- b2[k] = 1 needs b1[k] = 1 and, since b1[k-1] must be 0 (below), b1[k+1] = 1 (a 1-run of length 1 is dropped).
- b2[k-1] = 0 with b1[k] = 1 needs b1[k-1] = 0 (pass 2 clears b1[k-1] only when it is an isolated 1, which
  b1[k] = 1 rules out).
- b1[k-1] = 0 and b1[k] = 1: b1[k] = raw[k] or (raw[k-1] and raw[k+1]); with raw[k-1] = 0 this is raw[k] = 1; and
  b1[k-1] = 0 needs raw[k-1] = 0 and not (raw[k-2] = 1 and raw[k] = 1), so raw[k-2] = 0 (for k = 1, index 0 is not
  interior, b1[0] = raw[0], which the padding "raw[-1] = 0" reproduces).
- b1[k+1] = raw[k+1] or (raw[k] and raw[k+2]) = raw[k+1] or raw[k+2] (k+1 is interior when k <= T-3).
So a touchdown at k is `raw[k-2] = raw[k-1] = 0, raw[k] = 1, raw[k+1] or raw[k+2]`, decided at step k+2, and it
depends on raw[k-2..k+2] only. SCREEN pays touchdown (k, i) at k+3 when `alt` and no touchdown of j in [k-2, k+2]; j's
touchdown at k+2 is decided at k+4 in general (at k+3 only when raw_j[k+3] = 1). Every touchdown that can affect a
pay at k+4 <= T-1 has its index <= T-3, where the rule above is exact; so the env's pays equal SCREEN's shifted by one
step, except SCREEN's pays for k = T-4 (delivered at T-1), which the env never makes.
