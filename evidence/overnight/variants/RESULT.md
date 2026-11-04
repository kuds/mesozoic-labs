# VARIANTS: two-sided screen of HYPOTHETICAL plant variants (T. rex, physics r10)

Stage folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/variants/` (below, `out/...` and `scripts/...` are relative to it). Written at about 03:10 UTC (final; every planned run finished); time box 05:30 UTC.

**None of this changes the plant.** V1, V2 and V3 are HYPOTHETICAL in-memory copies of the model that the export's own factory builds. They exist only inside the analysis processes. No model file, repository file or export file was written. Every output that belongs to V1-V3 carries the label HYPOTHETICAL. Whether to change the real plant is the maintainer's decision; this screen only gives evidence for it.

## 1. Answer in brief

Variants: V0 is the real r10 plant (the baseline). V1 sets the foot contact solref to 0.01 s and V2 to 0.005 s (r10 uses 0.02 s). V3 doubles the hip-roll servo kp and keeps its forcerange.

1. **Stance side: no variant breaks the stance parent.** It ran 40/40 full 1000-step episodes on every variant, with no falls and no actuator at abs(cmd) >= 0.99. Its tail_3 command mean stays between +0.005 and +0.039 [measured: out/stance_policy40_V*.json].
   - On main's own stance gate cell at heading 0, V0, V1 and V3 are 40/40 clean.
   - **V2 lands exactly on the bar: 37/40 clean, LCB 0.817 against a bar of 0.80.** Two bars that were calibrated on the real r10 plant now fail: settle peak floor force above 1.5 BW, and sole tilt above 2.0 deg. The zero-action statue also fails 2/40 on V2, on settle peak force (up to 1.558 BW) [measured: out/gate_h0_V*.json, out/gate_detail.md].
   - Main's six-heading stance certificate (stance_quality_heading/v1) passes on all four variants. V0, V1 and V3 have 40/40 seeds clean at all six headings (LCB 0.928). V2 has 37/40 (LCB 0.817), the smallest count the bar admits: the config's comment says 37/40 is admitted and 36/40 refused. The three failing V2 seeds are the same at every heading: 3047 and 3059 (sole tilt 2.18 and 2.33 deg, bar 2.0) and 3069 (settle peak 1.575 BW, bar 1.5) [measured: out/heading_cert.json, out/gate_h0_V2.json, out/gate_h5_V2.json; configs/tyrannosaurus_rex/anatomical/stance.toml min_clean_stance_lcb comment].
2. **Walking side: no variant lifts the hand-built controller's speed ceiling.**
   - The faster attempt (v_des 0.35, seeds 0-4) fell on 5/5 seeds on every variant, at 3.0-5.0 s [measured: out/walk_f2v0.35_V*.json].
   - FINAL3 and f2v0.25 still walk 10/10 with alternation 1.000 on every variant. They walk at about the same speed or a little slower than on V0: FINAL3 0.176-0.207 m/s on V1-V3 against 0.206-0.211 m/s on V0 [measured: out/walk_final3_V*.json, out/walk_f2v0.25_V*.json].
   - V3 is the only variant that kept any seed up at v_des 0.30 (2/5). At v_des 0.35 its hip roll reaches the 480 N.m forcerange (58 substeps at >= 0.99), so forcerange becomes the next limit [measured: out/walk_f2v0.30_V3.json, out/walk_f2v0.35_V3.json].
3. **Single support (the SUPPORT stage's open-loop hold of its best pose, E): stiffer foot contact removes the pad-roll creep. Hip-roll kp does nothing to the hold.**
   - Real plant: right 2.21 s, left 1.83 s.
   - V1: right holds the full 10 s, left 3.45 s.
   - V2: right holds the full 10 s, left 5.26 s.
   - V3: right 2.14 s, left 1.80 s, the same as V0 [measured: out/hold_V*.json].
   - The stance-pad roll grows with an e-folding time of 0.45-0.52 s on V0 and V3. On V1 and V2 it does not grow: abs roll stays at or below 0.42 deg [measured: out/diag_V*_E_*.json].
4. **Two-sided reading.**
   - V1 is the only variant that improves one side without measurably costing the other. Its single-leg hold gets longer, while the stance parent, the statue and the gate cell at heading 0 stay within every bar. They did move toward two of the bars: settle peak 1.29 BW against 1.21 on V0 (bar 1.5), and statue sole tilt 1.68 deg against 1.60 (bar 2.0) [measured: rows above, out/gate_detail.md].
   - V2 improves support further but erodes main's stance gate, both for the parent and for the statue. The gate's bars are calibrated on r10's soft contact, so adopting V2 would also mean re-validating those bars [inferred: the bar comments in configs/tyrannosaurus_rex/anatomical/stance.toml cite r10 statue values].
   - V3 leaves stance and support unchanged and helps walking only marginally (v_des 0.30).
   - None of the three brings walking anywhere near the 1.0 m/s locomotion gate with this controller [measured: section 4.1]. The controllers were tuned on V0 and not re-tuned on any variant, so the walking rows are a lower bound on what a variant allows, not its ceiling [inferred: controller provenance].

## 2. Combined table: variant x (walking metrics, stance metrics)

All rows are generated from the saved outputs by `scripts/combined.py` (`out/combined_table.md`). Rows marked (extra) go beyond the stage brief; they were added by the first attempt or by this resume.

| metric | V0 real r10 (baseline) | V1 HYPOTHETICAL foot solref 0.01 | V2 HYPOTHETICAL foot solref 0.005 | V3 HYPOTHETICAL hip-roll kp x2 | source |
|---|---|---|---|---|---|
| **WALKING SIDE** (hand-built walker through env.step, locomotion-stage env, reset noise 0.05) | | | | | |
| FINAL3 (v_des 0.15), seeds 0-9: full 10 s episodes | 10/10 | 10/10 | 10/10 | 10/10 | out/walk_final3_V*.json |
| FINAL3 (v_des 0.15): walking-phase alternation (min over seeds) | 1.000 | 1.000 | 1.000 | 1.000 | out/walk_final3_V*.json |
| FINAL3 (v_des 0.15): speed m/s (from t = 1 s, range over seeds) | 0.206 to 0.211 | 0.197 to 0.205 | 0.180 to 0.207 | 0.176 to 0.186 | out/walk_final3_V*.json |
| FINAL3 (v_des 0.15): stride / L (floor touchdowns, range) | 0.231 to 0.240 | 0.226 to 0.305 | 0.240 to 0.320 | 0.217 to 0.231 | out/walk_final3_V*.json |
| FINAL3 (v_des 0.15): hip-roll abs(action) p95 (mean over seeds) | 0.65 | 0.69 | 0.75 | 0.44 | out/walk_final3_V*.json |
| FINAL3 (v_des 0.15): hip-roll share of steps with abs(action) = 1 | 0.013 | 0.014 | 0.016 | 0.001 | out/walk_final3_V*.json |
| FINAL3 (v_des 0.15): hip-roll servo error p95 deg | 11.6 | 12.0 | 12.4 | 6.1 | out/walk_final3_V*.json |
| FINAL3 (v_des 0.15): hip-roll force / forcerange, max | 0.37 | 0.37 | 0.37 | 0.37 | out/walk_final3_V*.json |
| f2v0.25 (v_des 0.25), seeds 0-9: full 10 s episodes | 10/10 | 10/10 | 10/10 | 10/10 | out/walk_f2v0.25_V*.json |
| f2v0.25 (v_des 0.25): walking-phase alternation (min over seeds) | 1.000 | 1.000 | 1.000 | 1.000 | out/walk_f2v0.25_V*.json |
| f2v0.25 (v_des 0.25): speed m/s (from t = 1 s, range over seeds) | 0.306 to 0.324 | 0.287 to 0.329 | 0.300 to 0.359 | 0.283 to 0.297 | out/walk_f2v0.25_V*.json |
| f2v0.25 (v_des 0.25): stride / L (floor touchdowns, range) | 0.335 to 0.385 | 0.375 to 0.401 | 0.344 to 0.427 | 0.317 to 0.332 | out/walk_f2v0.25_V*.json |
| f2v0.25 (v_des 0.25): hip-roll abs(action) p95 (mean over seeds) | 0.75 | 0.83 | 0.78 | 0.50 | out/walk_f2v0.25_V*.json |
| f2v0.25 (v_des 0.25): hip-roll share of steps with abs(action) = 1 | 0.020 | 0.024 | 0.020 | 0.001 | out/walk_f2v0.25_V*.json |
| f2v0.25 (v_des 0.25): hip-roll servo error p95 deg | 12.3 | 12.9 | 12.6 | 6.3 | out/walk_f2v0.25_V*.json |
| f2v0.25 (v_des 0.25): hip-roll force / forcerange, max | 0.37 | 0.37 | 0.37 | 0.38 | out/walk_f2v0.25_V*.json |
| (extra) FINAL3, held-out seeds 100-119: full episodes; falls | 20/20; none | 20/20; none | 20/20; none | 20/20; none | out/walk_ho_final3_V*.json |
| (extra) f2v0.25, held-out seeds 100-119: full episodes; falls | 20/20; none | 19/20; seed 117 at 5.82 s | 18/20; seed 101 at 5.75 s; seed 113 at 9.81 s | 20/20; none | out/walk_ho_f2v0.25_V*.json |
| faster attempt v_des 0.35, seeds 0-4: full episodes | 0/5 | 0/5 | 0/5 | 0/5 | out/walk_f2v0.35_V*.json |
| v_des 0.35: fall times s (range) | 3.15 to 3.72 | 3.25 to 3.77 | 3.02 to 3.76 | 3.56 to 4.98 | out/walk_f2v0.35_V*.json |
| v_des 0.35: hip-roll substeps at >= 0.99 forcerange (sum) | 0 | 0 | 0 | 58 | out/walk_f2v0.35_V*.json |
| (extra) v_des 0.30, seeds 0-4: full episodes; speed of full ones m/s | 0/5; - | 0/5; - | 0/5; - | 2/5; 0.375 to 0.511 | out/walk_f2v0.30_V*.json |
| SUPPORT open-loop single-leg hold, pose E (planned CoM margin +0.0213 m), right stance: hold time s | 2.21 (tail_contact) | 10.00 (held 10 s) | 10.00 (held 10 s) | 2.14 (tail_contact) | out/hold_V*.json |
| same, left stance: hold time s | 1.83 (tail_contact) | 3.45 (tail_contact) | 5.26 (tail_contact) | 1.80 (tail_contact) | out/hold_V*.json |
| same, cap-20 pose (margin +0.0048 m), right: hold time s | 1.59 (fallen) | 1.86 (fallen) | 2.10 (fallen) | 1.58 (fallen) | out/hold_V*.json |
| (extra) stance-pad roll e-folding time s, E pose (r / l) | 0.52 / 0.46 | no growth / no growth | no growth / no growth | 0.50 / 0.45 | out/diag_V*_E_*.json |
| **STANCE SIDE** (main's anatomical stance-stage env, committed config) | | | | | |
| zero-action statue (seed 0, no reset noise): held 10 s | True | True | True | True | out/stance_statue0_V*.json |
| statue: settled pelvis height m (steps 200-999) | 0.9261 | 0.9264 | 0.9265 | 0.9261 | out/stance_statue0_V*.json |
| statue: right-foot load share (lighter-foot min) | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) | out/stance_statue0_V*.json |
| statue: CoM margin in two-foot hull m (mean / min) | +0.0632 / +0.0579 | +0.0629 / +0.0581 | +0.0626 / +0.0584 | +0.0632 / +0.0579 | out/stance_statue0_V*.json |
| statue: head-tip height m (mean / min) | 0.958 / 0.875 | 0.959 / 0.877 | 0.959 / 0.877 | 0.958 / 0.875 | out/stance_statue0_V*.json |
| (extra) statue with reset noise 0.05, seeds 3042-3081: full horizon | 40/40 | 40/40 | 40/40 | 40/40 | out/stance_statue40_V*.json |
| stance parent (actor only, deterministic), seeds 3042-3081: full 1000-step episodes | 40/40 | 40/40 | 40/40 | 40/40 | out/stance_policy40_V*.json |
| parent: falls | none | none | none | none | out/stance_policy40_V*.json |
| parent: tail_3 command, mean of episode means (range) | +0.0074 (-0.0109 to +0.0543) | +0.0275 (+0.0090 to +0.0749) | +0.0393 (+0.0128 to +0.0899) | +0.0053 (-0.0109 to +0.0352) | out/stance_policy40_V*.json |
| parent: tail_3 max abs(cmd) | 0.4518 | 0.4522 | 0.4532 | 0.4519 | out/stance_policy40_V*.json |
| parent: episodes with any actuator at abs(cmd) >= 0.99 | 0/40 | 0/40 | 0/40 | 0/40 | out/stance_policy40_V*.json |
| parent: head-tip height m, mean (range of episode means) / min | 0.903 (0.894 to 0.918) / 0.881 | 0.911 (0.900 to 0.931) / 0.886 | 0.918 (0.905 to 0.942) / 0.889 | 0.901 (0.895 to 0.908) / 0.882 | out/stance_policy40_V*.json |
| parent: foot slide m, steps 200-999 (median / max) | 0.0064 / 0.0217 | 0.0068 / 0.0231 | 0.0066 / 0.0164 | 0.0054 / 0.0204 | out/stance_policy40_V*.json |
| parent: lighter-foot load share mean (min) | 0.499 (0.464) | 0.498 (0.437) | 0.493 (0.422) | 0.499 (0.476) | out/stance_policy40_V*.json |
| (extra) parent: foot-contact observation channels, mean abs(z) (max) under its frozen normalizer | 0.064 (5.78) | 0.077 (7.97) | 0.126 (14.08) | 0.061 (5.78) | out/stance_policy40_V*.json |
| (extra) main's stance gate cell, heading 0 (stance_report.run_panel, floor truth) | parent 40/40 clean, LCB 0.9278, passed True; statue 40/40 | parent 40/40 clean, LCB 0.9278, passed True; statue 40/40 | parent 37/40 clean, LCB 0.8174, passed True, failures {'max_settle_peak_floor_force_bw': 1, 'max_sole_tilt_deg': 2}; statue 38/40 | parent 40/40 clean, LCB 0.9278, passed True; statue 40/40 | out/gate_h0_V*.json |
| (extra) main's six-heading stance certificate (stance_quality_heading/v1) | passed True, 40/40 seeds clean at all six headings, LCB 0.9278 | passed True, 40/40 seeds clean at all six headings, LCB 0.9278 | passed True, 37/40 seeds clean at all six headings, LCB 0.8174 | passed True, 40/40 seeds clean at all six headings, LCB 0.9278 | out/heading_cert.json |

Metric definitions [measured: scripts/walk.py, scripts/stance.py, scripts/summarize.py]:
- **Speed** is the CoM displacement along the initial heading from t = 1 s to the end of a full episode, divided by the elapsed time.
- **Alternation** is the share of alternating consecutive floor touchdowns after t_settle (touchdown = foot floor load above 5 % of body weight, debounced).
- **Hip-roll use** is the walking-phase abs(action) of r/l_hip_roll_act, with these readouts: the share of steps at abs(action) = 1; the servo error abs(ctrl - q); and actuator_force / forcerange per substep.
- **Stance window** is steps 200-999, the gate's settle split.
- **Foot slide** is the larger pad-centre displacement from step 200 to the end.
- **Head-tip height** is the env's info["head_tip_z"].
- **CoM margin** is the signed distance of the CoM from the edge of the hull of both feet's sole points (positive inside).

## 3. What was run

All runs used the export of main 4d5f4e7 via `environments.shared.config.build_env` (stance or locomotion stage, committed configs).

**Variants** [measured: scripts/vlib.py, out/check_variants_HYPOTHETICAL.json, out/check_variants_stdout.txt]:
- Each variant is applied in place to `env.model` right after the factory builds the env, and before the first reset.
- **V1/V2:** `geom_solref[:, 0]` of the eight foot geoms (r/l_plantar_geom and r/l_toe_d2/d3/d4_geom) is set to 0.01 or 0.005 s. dampratio 1 and solimp (0.9, 0.95, 0.001, 0.5, 2) are kept. The metatarsus capsules, the floor and every other geom keep 0.02 s.
  - Every geom of this plant has solmix 1 and priority 0, so MuJoCo would otherwise average the foot and floor solref (V1 would give 0.015 s). The foot geoms therefore also get geom_priority 1, so that a foot contact takes its parameters from the foot geom.
  - The foot and floor solimp, friction (1.2), condim (3), margin and gap are identical (asserted), so priority changes only the time constant. Live floor contacts after 0.5 s read solref 0.01 (V1), 0.005 (V2) and 0.02 (V0, V3) on the stance and locomotion envs.
  - 0.005 s is 2.5 timesteps (timestep 0.002 s), above MuJoCo's refsafe floor of 2 timesteps.
- **V3:** `actuator_gainprm[a, 0]` and `actuator_biasprm[a, 1]` of r/l_hip_roll_act are doubled (kp 600 -> 1200 N.m/rad). Forcerange (+-480 N.m), ctrlrange, kv (0) and joint damping are unchanged. The deliverable torque at home with ctrl at the range end goes from 261.8 to 480.0 N.m (the forcerange cap).
- V0 changes no array (fingerprint check).

**Walking side** [measured: scripts/walk.py, scripts/walk_all.sh, scripts/extra_all.sh]:
- The controller is the EXISTENCE stage's hand-built walker `wlib.Walker`, copied unchanged (sha256 matches the source).
- It runs on the locomotion-stage env (default reset noise 0.05), acting only through `env.step`.
- The walker is constructed after the variant is applied, so its model-based parts (Jacobians, gravity compensation, ctrl = q + tau/kp) see the variant.
- The V3 "real kp table" rows are a control in which the walker is not told about the kp change.
- Parameter sets match `feas/existence/out/variant_ledger.json`:
  - FINAL3 = {kz 8, dz 2, v_des 0.15, step_w 0.35, kby 1.5, max_w 0.8}.
  - f2v0.25 = the same with v_des 0.25 and k_yp 0.05.
  - The faster attempt = f2v0.25 with v_des 0.35.
- Seeds 0-9, plus 0-4 for the faster attempt. Extras: held-out seeds 100-119, and a v_des 0.30 rung on seeds 0-4.

**Single-leg hold** [measured: scripts/hold_variant.py, scripts/diag_variant.py]:
- These are the SUPPORT stage's `hold_test.py` 'placed' mode and `diag_creep.py`, copied unchanged.
- Pose: the optimiser poses from `feas/support/out/opt_full.json`: E (planned CoM margin +0.0213 m, right and left stance) and cap-20 (+0.0048 m), with swing-foot lift h = 0.0438 m (0.05 L).
- The pose is placed at zero velocity and the static action is held open loop for 10 s (SUPPORT used 5 s) on the locomotion-stage env (reset noise 0, seed 0).
- The variant is applied both to the planning plant (so V3's static ctrl uses kp 1200) and to the stepped env.

**Stance side** [measured: scripts/stance.py, scripts/stance_all.sh, scripts/gate_variant.py]:
- Env: main's anatomical stance-stage env (committed stance.toml: max_episode_steps 1000, reset_noise_scale 0.05, dt 0.01 s).
- **Statue:** zero action, noise-free reset, seed 0, 1000 steps = 10 s. Extra: zero action with the config's noise on seeds 3042-3081.
- **Parent:** the stance checkpoint `fetch/stance7_r10LTSBpc_s42/01_stance/models/stage1_final.zip` with `stage1_final_vecnorm.pkl` (sha256 1c9d87c3... and 5e417fef..., re-checked OK at resume: out/provenance_recheck_resume.txt).
  - It is loaded by the export's `load_sb3_checkpoint`, validated against the real plant identity (policy class CriticPhaseActorCriticPolicy, obs 63, clip_obs 10).
  - Actor only, `model.predict(normalized obs, deterministic=True)` with the normalizer frozen. Seeds 3042-3081, full 1000-step episodes.
- **Extra:** main's own gate machinery (`stance_report.run_panel`, floor truth, `SpawnYaw(yaw, whole_scene=True)`, `stance_gate_v2` / `stance_heading_gate` evaluators, thresholds from the committed config). Only one thing differs: the panel's env class applies the variant right after construction.
  - The parent and the statue each get 40 seeds from 3042, at heading 0 and at the five other headings of the certificate grid.
  - For V1-V3 the plant identity is not passed to run_panel, because a variant is by construction not the committed plant. For V0 it is passed and validated.

**Rules.** No training of any kind. One simulation process at a time, with OMP_NUM_THREADS=1, nice -n 10 and TMPDIR inside the folder. The repository and the export were not touched. Nothing under the inputs was modified after the stage began (find -newer: 0 files, logged in CMDLOG at resume).

## 4. Details by side

### 4.1 Walking

- **Baseline reproduces EXISTENCE.**
  - FINAL3 on V0: 10/10 full, alternation 1.000, 0.206-0.211 m/s. EXISTENCE gives 0.209-0.212 m/s by its own displacement measure [measured: out/walk_final3_V0.json; feas/existence/out/summary_final3.json].
  - f2v0.25 on V0: 0.306-0.324 m/s [measured: out/walk_f2v0.25_V0.json].
  - Stride from floor touchdowns: 0.231-0.240 L. EXISTENCE gives 0.224-0.229 L with its probe's own stride definition. This is a different measure, not a discrepancy [inferred: definitions differ].
- **Contact variants (V1, V2).**
  - Both controllers still walk every seed 0-9.
  - FINAL3 is slightly slower and more variable: V1 0.197-0.205 m/s, V2 0.180-0.207 m/s.
  - f2v0.25 is more variable: V1 0.287-0.329 m/s, V2 0.300-0.359 m/s.
  - Hip-roll use rises a little: abs(action) p95 0.65 -> 0.69 / 0.75 for FINAL3 [measured: out/walk_*_V1.json, out/walk_*_V2.json].
  - On the held-out seeds, f2v0.25 loses 1/20 on V1 (seed 117, 5.82 s) and 2/20 on V2 (seeds 101 at 5.75 s and 113 at 9.81 s); V0 loses none [measured: out/walk_ho_f2v0.25_V*.json]. On V1/V2 the floor-touchdown stride reads longer than speed x stride period. For FINAL3 on V2 it is 0.24-0.32 L per seed, while speed x 2 x median step period ("controller-clock stride") gives 0.197-0.227 L. On V0 the two agree within 0.01 L (0.231-0.240 vs 0.221-0.231) [measured: out/walk_final3_V*.json, out/tables.md]. The floor touchdown count equals the controller's step count on 19 of 20 FINAL3 V1/V2 runs (off by one on the other), so the gap is not from missed or extra touchdowns. Its cause (where on the pad the load first registers, or slip) was not resolved [measured: out/summary.json; inferred: cause open].
- **Hip-roll kp x2 (V3).**
  - Hip-roll servo error p95 halves: 11.6 -> 6.1 deg (FINAL3). abs(action) p95 drops 0.65 -> 0.44. The share of clipped steps drops 0.013 -> 0.001. Peak force / forcerange is unchanged at 0.37 [measured: out/walk_final3_V3.json].
  - Speed drops to 0.176-0.186 m/s (FINAL3) and 0.283-0.297 m/s (f2v0.25). With the walker not told about the kp change, 0.190-0.202 m/s [measured: out/walk_final3_V3.json, out/walk_final3_V3realctl.json].
  - So at 0.2-0.3 m/s, hip-roll servo authority does not bind for this controller [inferred: force fraction <= 0.38 and clipping <= 2 % of steps on V0].
- **Faster attempt (v_des 0.35, seeds 0-4).**
  - Every variant fell on every seed: V0 3.15-3.72 s, V1 3.25-3.77 s, V2 3.02-3.76 s, V3 3.56-4.98 s.
  - On V0-V2 the hip-roll command is clipped on 12.6-14.4 % of steps.
  - On V3 it is clipped on 9.2 % of steps, and the force reaches the 480 N.m forcerange on 58 substeps [measured: out/walk_f2v0.35_V*.json, out/tables.md].
  - Extra rung, v_des 0.30: V0-V2 0/5; V3 2/5, at 0.375-0.511 m/s on the two surviving seeds [measured: out/walk_f2v0.30_V*.json].
- The plant's own speed ceiling is still unmeasured; these are controller-limited results [inferred: as in FEAS_RESULTS].

### 4.2 Single-leg hold and creep

- **Baseline reproduces SUPPORT.**
  - E right 2.21 s and E left 1.83 s, both ending in tail_contact; cap-20 right 1.59 s, fallen. SUPPORT measured right 1.83-2.39 s, left 1.72-2.03 s and cap-20 1.59 s.
  - CoM drift in the first 0.5 s: 0.0115 m. SUPPORT measured 0.0115-0.0142 m.
  - Pad-roll e-folding: 0.52 s (r) and 0.46 s (l), matching "about 0.4-0.5 s" [measured: out/hold_V0.json, out/diag_V0_E_*.json; feas/support/RESULT.md section 5].
- **V1.**
  - E right holds 10 s; the CoM margin falls from +0.0213 to +0.0033 m and the CoM drifts 0.083 m.
  - E left falls at 3.45 s: the CoM leaves the pad at 2.71 s.
  - abs pad roll is at most 0.29 deg (right) and 0.42 deg (left) over the whole 5 s diagnostic trace. On V0 it reaches 14-23 deg before the fall [measured: out/hold_V1.json, out/diag_V1_E_*.json, out/diag_V0_E_*.json].
- **V2.**
  - E right holds 10 s with a margin of +0.0191 m and a drift of 0.031 m.
  - E left falls at 5.26 s: the CoM leaves the pad at 4.41 s.
  - abs pad roll is at most 0.09 deg (right) and 0.14 deg (left) [measured: out/hold_V2.json, out/diag_V2_E_*.json].
  - This is consistent with SUPPORT's HYPOTHETICAL all-geom solref 0.005 control, which held E right for 5 s [measured: feas/support/RESULT.md section 6].
  - The left-side falls on V1/V2 happen without pad-roll growth, so a second, slower drift remains. Its cause was not identified [inferred: from the roll traces].
- **Cap-20 pose** (margin +0.0048 m): falls in 1.58-2.10 s on every variant [measured: out/hold_V*.json].
- **V3:** hold times and creep match V0 within 0.07 s [measured: out/hold_V3.json, out/diag_V3_E_*.json]. This is consistent with SUPPORT's finding that servo kp x2/x4 changes nothing [measured: feas/support/RESULT.md].
- The hand-built walker never relies on this static hold: its step period is about 0.47 s, within one e-folding time. That would explain why the large support gain on V1/V2 does not show up as walking speed [inferred: step rate 2.08-2.22 Hz, out/tables.md].

### 4.3 Stance

- **Statue (seed 0, no reset noise).** It holds 10 s on every variant [measured: out/stance_statue0_V*.json]:
  - pelvis height 0.9261-0.9265 m;
  - right-foot load share 0.500;
  - CoM margin +0.0626 to +0.0632 m;
  - head tip 0.958-0.959 m;
  - foot slide at most 0.1 mm.

  With reset noise on seeds 3042-3081 it is 40/40 on every variant [measured: out/stance_statue40_V*.json].
- **Parent.** 40/40 on every variant [measured: out/stance_policy40_V*.json].
  - Stiffer contact shifts its posture slightly. The tail_3 mean goes +0.0074 (V0) -> +0.0275 (V1) -> +0.0393 (V2). The head-tip mean goes 0.903 -> 0.911 -> 0.918 m. The lighter-foot load share minimum goes 0.464 -> 0.437 -> 0.422.
  - Foot slide is unchanged (median 5.4-6.8 mm, max 16-23 mm).
  - Max abs(cmd) is at most 0.71 on any actuator (tail_1_pitch, V2) [measured: out/policy40_detail_stdout.txt]. Max abs(tail_3 cmd) is 0.452-0.453 on all variants.
  - V3 is indistinguishable from V0.
- **Observation shift.** The parent's frozen normalizer sees larger foot-contact readings on the stiff-contact variants: mean abs(z) 0.064 (V0), 0.077 (V1), 0.126 (V2); max 5.8, 8.0 and 14.1 (clip 10) [measured: out/stance_policy40_V*.json, out/tables.md]. On V2 the parent is acting partly outside its training distribution [inferred: z above the clip].
- **Main's gate cells.** Heading-0 rows for the parent and the statue, with the plant-calibrated metrics. The rows for the other five headings are in out/gate_detail.md and match these to the precision shown [measured: out/gate_detail.md]:

| variant | heading deg | who | clean/40 | nonzero criterion failures | max settle peak floor force BW (bar 1.5) | max sole tilt deg (bar 2.0) | max sole corner lift mm (bar 4) | max foot slide mm | max window displacement mm | min windowed foot-load share (bar 0.35) |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 real r10 | +0 | parent | 40/40 | none | 1.205 | 1.065 | 1.752 | 22.278 | 9.342 | 0.470 |
| V0 real r10 | +0 | statue | 40/40 | none | 1.207 | 1.603 | 2.626 | 10.308 | 18.945 | 0.469 |
| V1 HYPOTHETICAL solref 0.01 | +0 | parent | 40/40 | none | 1.291 | 1.411 | 2.306 | 22.835 | 15.388 | 0.442 |
| V1 HYPOTHETICAL solref 0.01 | +0 | statue | 40/40 | none | 1.293 | 1.675 | 2.696 | 10.578 | 17.195 | 0.443 |
| V2 HYPOTHETICAL solref 0.005 | +0 | parent | 37/40 | {'max_settle_peak_floor_force_bw': 1, 'max_sole_tilt_deg': 2} | 1.575 | 2.327 | 3.732 | 16.664 | 12.937 | 0.424 |
| V2 HYPOTHETICAL solref 0.005 | +0 | statue | 38/40 | {'max_settle_peak_floor_force_bw': 2} | 1.558 | 1.902 | 3.008 | 6.990 | 14.900 | 0.443 |
| V3 HYPOTHETICAL hip-roll kp x2 | +0 | parent | 40/40 | none | 1.213 | 0.547 | 0.931 | 20.761 | 6.920 | 0.484 |
| V3 HYPOTHETICAL hip-roll kp x2 | +0 | statue | 40/40 | none | 1.240 | 1.187 | 1.916 | 13.494 | 18.517 | 0.461 |

**Six-heading stance certificate** (stance_quality_heading/v1, the export's own evaluator over the saved cells)

| variant | complete | certificate passed | clean seeds (all six headings) | LCB | per-heading clean | failures |
|---|---|---|---|---|---|---|
| V0 real r10 (baseline) | yes | True | 40/40 | 0.9278 | -90: 40, -45: 40, +0: 40, +45: 40, +90: 40, +180: 40 | none |
| V1 HYPOTHETICAL foot solref 0.01 | yes | True | 40/40 | 0.9278 | -90: 40, -45: 40, +0: 40, +45: 40, +90: 40, +180: 40 | none |
| V2 HYPOTHETICAL foot solref 0.005 | yes | True | 37/40 | 0.8174 | -90: 37, -45: 37, +0: 37, +45: 37, +90: 37, +180: 37 | none |
| V3 HYPOTHETICAL hip-roll kp x2 | yes | True | 40/40 | 0.9278 | -90: 40, -45: 40, +0: 40, +45: 40, +90: 40, +180: 40 | none |

[measured: out/heading_cert.json, from scripts/combine_cert.py over out/gate_h0_V*.json and out/gate_h5_V*.json]. V2 seed failures (seed: headings): 3047: all six; 3059: all six; 3069: all six. On V2 the statue itself fails settle peak floor force on seeds 3051 (1.558 BW) and 3069 (1.517 BW) at every heading [measured: out/gate_h0_V2.json, out/gate_h5_V2.json].

## 5. What was kept from the first attempt, and what this resume added

A container restart at about 01:52 UTC stopped the first attempt (CMDLOG from 01:11:16 to 01:50:52). Its last command, `gate_variant.py V2 heading 0`, had left an empty stdout and no JSON. This resume appended `RESUME after container restart` at 01:54:56 UTC.

**Kept as complete and valid.** I checked these by reading every script and parsing every output: entry counts, seeds, variant labels, which arrays each run changed, and that the V0 baselines reproduce EXISTENCE, SUPPORT and the stance config's r10 statue values (settle peak 1.207 BW, sole tilt max 1.603 deg against the config comment's 1.207 and 1.60). Inputs and copied scripts were re-hashed OK. The kept outputs are:
- `out/check_variants_HYPOTHETICAL.json`, `out/inspect_plant_stdout.txt`
- all 27 `out/walk_*.json` with their stdout: final3, f2v0.25, f2v0.35, f2v0.30, ho_final3 and ho_f2v0.25 on V0-V3, V3realctl for final3 and f2v0.25, and the smoke run
- `out/hold_V0..V3.json`, `out/diag_V0..V3_E_r/l.json`
- `out/stance_statue0_V*.json`, `out/stance_statue40_V*.json`, `out/stance_policy40_V*.json`, `out/policy40_detail_stdout.txt`
- `out/gate_h0_V0.json`, `out/gate_h0_V1.json`
- the provenance files and the driver logs

**Superseded but kept as records:** `out/summary_partial_walk.json` and `out/tables_partial_walk.md` (the walking-only interim), and `out/gate_h0_driver.txt` (cut off at V2).

**Redone:** `gate_h0_V2` (the interrupted run).

**New in this resume:**
- `gate_h0_V3` and `gate_h5_V0..V3` (the other five headings) through `scripts/gate_resume.sh`
- `out/heading_cert.json` (`scripts/combine_cert.py`, no simulation)
- `scripts/assemble_result.py`, which builds this file from `tmp/RESULT_template.md` and the generated tables
- `scripts/gate_detail.py` -> `out/gate_detail.md`
- `scripts/combined.py` -> `out/combined_table.md`
- a patch to `scripts/summarize.py`: the gate table now shows only nonzero criterion failures, and the certificate table is added; the pre-patch copy is `tmp/summarize_before_resume_patch.py`
- regenerated `out/summary.json` and `out/tables.md`
- this RESULT.md

## 6. Limits

- **The walking controllers are V0-tuned.** No variant got re-tuned gains. A variant's walking rows are therefore a lower bound on what that plant permits, not its ceiling. The plant's speed ceiling is still unmeasured on every variant [inferred].
- **No learned policy was run on the walking side.** The stance parent is the only learned policy here, and it was trained on V0. Its numbers on V1-V3 say how a V0-trained policy transfers; they do not say what a policy trained on the variant would do [inferred].
- **V1/V2 change the foot geoms only.** Through priority 1 they also govern any foot contact with a non-floor geom (for example foot on foot or foot on tail). No such contacts were recorded in the stance panels (nonfoot load 0 N; the gate's foot-on-foot criterion, max_foot_contact_fraction, never failed), but they were not separately audited on the walking side [measured: out/policy40_detail_stdout.txt, out/gate_detail.md; inferred for walking].
- **Seed counts are small:** 10 (walking), 5 (faster attempt), 40 (stance). The V1/V2 held-out falls (1-2 of 20) are too few to separate a real robustness loss from chance [inferred].
- **The hold tests are open loop by design.** A feedback balancer was not tried, as in SUPPORT.
- **The six headings are not six independent samples here.** With the whole scene rotated (`SpawnYaw(whole_scene=True)`), the per-episode gate metrics at the five other headings equal heading 0's to a relative difference of at most 1e-9 for the statue and at most 0.5 % for the parent (floating-point divergence under feedback) on every variant [measured: python3 comparison logged in CMDLOG at about 03:05 UTC over out/gate_h0_V*.json and out/gate_h5_V*.json]. So the certificate is, in effect, the heading-0 cell repeated, and the same three V2 seeds fail at every heading [measured: out/heading_cert.json].
- **The gate bars' calibration.** For V2, a statue failing settle peak floor force says the bar, calibrated on r10, would need re-validation on any stiffer-contact plant. That is a reading of the config comments, not a re-calibration [inferred].

## 7. Slips

**First attempt (from its CMDLOG):**
1. Several read-only commands ran without the logger and were recorded after the fact as NOTE lines (01:14:13, 01:17:06, 01:24:28, and the wait loops at 01:22:06, 01:23:47, 01:25:18, 01:37:01, 01:45:31 and 01:48:22).
2. `scripts/gate_all.sh` was first written with an argument typo (`$v=-90,...`) in its five-heading branch. It was fixed with sed at 01:40:39, before that branch ever ran; the branch never ran before the restart.
3. Read-only analysis processes (`summarize.py`, two `python -c` readers of saved JSON) ran at the same time as a simulation process (01:23:47, 01:38:00, 01:45:48). They did not simulate, so the one-simulation-at-a-time rule held. I report it for completeness.

**This resume:**
1. My first five inspection commands (01:54:05 to 01:54:56) and several progress checks (01:57 to 02:03) ran without being logged at the time. They are recorded after the fact in a NOTE at 02:04:10.
2. Some read-only `python3 -I -c` inspections are logged as a description of what they printed rather than the literal command text.
3. One of those inspections stopped with an IndexError at about 02:00. It was read-only and wrote nothing.
4. The first attempt's `tables.md` printed all-zero criterion-failure dicts in the gate table. I fixed this in the summarize patch; it was a presentation issue, not a data issue.
5. Five read-only foreground waits (02:07 to 03:04) ran without separate log lines at the time. They are recorded after the fact in a NOTE line in CMDLOG.
6. The stage venv was created at 01:14:13 by the first attempt with `python3.13 -m venv --without-pip`. That tool also wrote shell-activation scripts, and the bash one contained two device-path tokens in redirects. Nothing in this stage ever used those scripts: every run calls `venv/bin/python` directly. The resume found them at about 03:06 UTC and deleted the four activation scripts. A scan of the folder now finds no device path other than the generic search pattern in CMDLOG's own grep lines. The venv still imports mujoco 3.10.0, scipy 1.18.1 and numpy 2.5.4 (tmp/venv_check_after_activate_removal.txt).

## 8. Files

- **Folder:** `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/variants/`
- **Tables:** `out/combined_table.md` (the variant x walking/stance table), `out/tables.md` (all detail tables), `out/gate_detail.md`, `out/summary.json`, `out/heading_cert.json`
- **Scripts:** `scripts/vlib.py` (variants), `walk.py`, `hold_variant.py`, `diag_variant.py`, `stance.py`, `gate_variant.py`, `combine_cert.py`, `summarize.py`, `combined.py`, `gate_detail.py`; the drivers `walk_all.sh`, `stance_all.sh`, `extra_all.sh`, `gate_all.sh`, `gate_resume.sh`; the copied inputs in `scripts/copied/` (sha256 in `out/provenance_copied_sha256.txt`)
- **Log:** `CMDLOG.txt`
