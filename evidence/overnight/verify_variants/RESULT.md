# VERIFY_VARIANTS: independent adversarial check of the VARIANTS stage (T. rex, physics r10)

Verifier folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/verify_variants/` (paths below are relative to it unless absolute).
Stage checked: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/variants/` (its RESULT.md, its outputs, scripts and CMDLOG).
Work ran 03:07-03:52 UTC. Write-up at about 03:52 UTC; time box 07:30 UTC.

**None of this changes the plant.** V1, V2 and V3 are HYPOTHETICAL copies, held only in memory, of the model that the export's own factory builds. Neither the stage nor this check wrote any model, repository or export file. Any real change to the plant is the maintainer's decision.

**Resume.** When I started (`ls` at 03:07:04 UTC), my folder did not exist. The first attempt left no CMDLOG, scripts or outputs, so there was nothing to keep. Everything here is new. `RESUME after container restart` was appended to CMDLOG.txt at 03:07:12 UTC.

## 1. Verdict: CONFIRMED WITH CORRECTIONS

I re-ran every headline number with my own episode loops and metric code. The EXISTENCE walker and SUPPORT's hold code were imported read-only, and only to produce actions. Each headline number matched the stage's saved outputs exactly. The per-seed and per-episode values are bitwise identical: speed, fall time, hold time, tail_3 mean, head tip, foot slide, and all 43 numeric fields of main's gate metrics, with differences of 0.0 [measured: out/cmp_walk.txt, out/cmp_stance.txt, out/cmp_gate.txt, out/hold.json, out/stance_statue0.json].

The corrections concern wording and labels, not data:
1. **"No variant lifts the hand-built controller's (speed) ceiling" is stated too strongly and labelled [measured].** On V3 the controller survives v_des 0.30 on 2/5 seeds. Those seeds walk 0.375-0.511 m/s (mean 0.443) with alternation 1.000. That is faster than anything this controller holds on V0: V0's best is 0.324 m/s with f2v0.25, and V0 survives v_des 0.30 on 0/5 seeds [measured: out/walk_main.json, out/misc_stats.txt]. Suggested wording: "no variant lets this controller survive v_des 0.35 (0/5 on each). V3 alone survives v_des 0.30 on 2/5 seeds at 0.375-0.511 m/s, a small-sample gain that is still far below the 1.0 m/s gate." The stage reports the 2/5 itself; only the summary sentence over-reaches [inferred: comparison of the measured rows].
2. **Some interpretive sentences carry a [measured] label.** The stage's RESULT.md section 1 has three:
   - item 4: "V1 is the only variant that improves one side without measurably costing the other", labelled [measured: rows above, out/gate_detail.md]. This is a judgement. V3 also improves one walking row (v_des 0.30) at no stance cost, while slowing FINAL3 from 0.210 to 0.180 m/s mean. The key-results line labels the same sentence "[measured + inferred]".
   - item 2: "so forcerange becomes the next limit [measured]".
   - The summary sentence in correction 1.

   These should carry [inferred].
3. **Rule slips the stage did not report** (section 6): three V1-V3 output files lack the HYPOTHETICAL label, and the stage often used an interpreter other than the designated venv.

## 2. Headlines, one by one

| stage headline claim | my re-measurement | status |
|---|---|---|
| None of this changes the plant; V1-V3 were in-memory copies; nothing written | My own variant code (`scripts/vv.py`) produces models bitwise equal, in every MjModel array, to the stage's `vlib.apply_variant`, on both the stance and locomotion envs. V0 changes no array. V1/V2 change only `geom_solref` and `geom_priority`; V3 changes only `actuator_gainprm` and `actuator_biasprm`. `env.reset` changes nothing back. No file under feas/, fetch/stance7_r10LTSBpc_s42/, the stage folder or the shared venvs changed after 03:07 UTC [measured: out/vcheck.json, logs/vcheck.txt; `find -newermt` in CMDLOG at 03:48:40] | CONFIRMED |
| Stance: parent 40/40 full 1000-step episodes on every variant, no falls, no actuator at abs(cmd) >= 0.99 | 40/40 full and 0 falls on V0-V3. Max abs(a) is 0.653 / 0.680 / 0.714 / 0.651, all on tail_1_pitch_act. 0 episodes have any actuator at >= 0.99 [measured: out/stance_policy40.json, logs/stance_policy40.txt] | CONFIRMED |
| Main's gate: V0, V1, V3 40/40 clean at all six headings; V2 37/40, LCB 0.817 vs 0.80; statue fails 2/40 on V2 (settle peak > 1.5 BW) | I re-ran headings 0 and +90 for all four variants with main's run_panel (floor truth, SpawnYaw whole_scene, committed thresholds). V0, V1, V3: parent 40/40 clean, LCB 0.9278; statue 40/40. V2: parent 37/40 clean, LCB 0.8174, passed True. The failing seeds are 3047 (sole tilt 2.181 deg), 3059 (2.327 deg) and 3069 (settle peak 1.575 BW). The V2 statue is 38/40 clean, failing 3051 (1.558 BW) and 3069 (1.517 BW), with 40/40 full horizon. Both headings are bitwise equal to the stage's cells. I rebuilt the six-heading certificate with main's evaluator from my two headings plus the stage's other four; it gives the same verdicts: V0, V1, V3 40/40 at all six (LCB 0.9278), and V2 37/40 (LCB 0.8174), failing the same three seeds at all six [measured: out/gate_h0.json, out/gate_h90.json, out/cmp_gate.txt, out/gate_h0_maxima.txt] | CONFIRMED (4 of 6 headings from the stage's saved cells, not re-run) |
| Walking: FINAL3 and f2v0.25 10/10, alternation 1.000 on all variants | 10/10 and alternation 1.000 for both controllers on V0-V3. Alternation comes from my own hysteresis touchdown detector: on above 5 % BW, off below 2 % BW, 2-step hold. FINAL3 speed mean / range in m/s: V0 0.210 / 0.206-0.211, V1 0.201 / 0.197-0.205, V2 0.200 / 0.180-0.207, V3 0.180 / 0.176-0.186. f2v0.25: 0.313 / 0.306-0.324, 0.304 / 0.287-0.329, 0.313 / 0.300-0.359, 0.292 / 0.283-0.297 [measured: out/walk_main.json, out/misc_stats.txt] | CONFIRMED |
| v_des 0.35 seeds 0-4: every variant fell 5/5 (3.0-5.0 s); V3 hip roll hits the 480 N.m forcerange | 0/5 on each variant. Falls: V0 3.15-3.72 s, V1 3.25-3.77 s, V2 3.02-3.76 s, V3 3.56-4.98 s. V3 hip-roll force / forcerange reaches 1.000 on 58 substeps at >= 0.99. V0-V2 have 0 such substeps, and their hip-roll command is clipped on 12.6 / 14.4 / 12.8 % of steps [measured: out/cmp_walk.txt] | CONFIRMED |
| V3 alone kept 2/5 seeds up at v_des 0.30 | V3 2/5 full, at 0.375-0.511 m/s; V0-V2 0/5 [measured] | CONFIRMED (see correction 1 for the "ceiling" sentence) |
| No variant lifts the controller's ceiling | Contradicted in part by the row above | CORRECTED (wording and label) |
| Hold of SUPPORT's pose E: right 2.21 / 10 / 10 / 2.14 s; left 1.83 / 3.45 / 5.26 / 1.80 s (V0 / V1 / V2 / V3) | Identical: right 2.21 (tail_contact), 10.00 held, 10.00 held, 2.14 s; left 1.83, 3.45, 5.26, 1.80 s (all tail_contact). Cap-20 pose: 1.59, 1.86, 2.10, 1.58 s (fallen) [measured: out/hold.json, logs/hold.txt] | CONFIRMED |
| Stiffer foot contact removes the pad-roll creep (e-folding 0.45-0.52 s on V0/V3; abs roll at most 0.42 deg on V1/V2) | My own read-only probe of the stance-pad orientation gives these roll e-folding times, fitted over abs roll 0.5-10 deg: V0 0.52 s (r) / 0.46 s (l), V3 0.50 / 0.45 s. Max abs roll in the first 5 s: V1 0.29 / 0.42 deg, V2 0.09 / 0.14 deg. **Nuance:** on the left the total pad tilt still reaches 3.32 deg (V1) and 1.60 deg (V2) before those falls, while roll stays at most 0.42 deg. So the left-side drift involves a non-roll rotation of the pad (pitch) [measured: out/hold.json "pad"]; whether it causes the fall is not established [inferred] | CONFIRMED (with nuance) |
| Hip-roll kp x2 changes nothing in the hold | 2.14 / 1.80 s against 2.21 / 1.83 s on V0 [measured] | CONFIRMED |

Other key-results numbers I re-measured, all matching [measured: out/cmp_walk.txt, out/cmp_stance.txt, out/stance_statue0.json, out/walk_ho_f2v025.json, out/gate_h0_maxima.txt]:
- **FINAL3 hip roll** (V0 / V1 / V2 / V3):
  - abs(a) p95: 0.65 / 0.69 / 0.75 / 0.44.
  - servo error p95: 11.6 / 12.0 / 12.4 / 6.1 deg.
  - force / forcerange max: 0.372 / 0.371 / 0.372 / 0.373.
- **Held-out f2v0.25 seeds 100-119:** V0 20/20, V1 19/20 (seed 117 falls at 5.82 s), V2 18/20 (101 at 5.75 s, 113 at 9.81 s), V3 20/20.
- **Statue, seed 0, noise 0:**
  - pelvis 0.9261 / 0.9264 / 0.9265 / 0.9261 m;
  - load share 0.500;
  - CoM margin mean / min +0.0632 / +0.0579, +0.0629 / +0.0581, +0.0626 / +0.0584, +0.0632 / +0.0579 m. My own sole-corner hull gives these values, and a hull of the live contact points gives the same means;
  - head tip 0.958 / 0.959 / 0.959 / 0.958 m;
  - held 10 s.
- **Parent:**
  - tail_3 mean of episode means: +0.0074 / +0.0275 / +0.0393 / +0.0053;
  - max abs(tail_3): 0.4518 / 0.4522 / 0.4532 / 0.4519;
  - head-tip window mean: 0.903 / 0.911 / 0.918 / 0.901 m;
  - foot slide median / max: 6.4 / 21.7, 6.8 / 23.1, 6.6 / 16.4 and 5.4 / 20.4 mm;
  - lighter-foot share min: 0.464 / 0.437 / 0.422 / 0.476.
- **Gate maxima:** V1 settle peak 1.29 BW (V0 1.21) and V1 statue sole tilt 1.675 deg (V0 1.603). The V0 statue values 1.207 BW and 1.603 deg match the stance.toml comment for r10.

## 3. The checks the brief asked for

- **Only env.step, actions in [-1, 1].** Each check below was made at every step.
  - **Walking**, 140 episodes on seeds 0-9 / 0-4 plus 80 held-out episodes:
    - max abs(a) is 1.000 and 0 actions are non-finite;
    - the controller call never wrote env.data. I compared qpos, qvel, ctrl, act, xfrc_applied, qfrc_applied and mocap_pos before and after each call: 0 writes;
    - no model array changed during any episode;
    - nothing but the feet touched the floor.

    [measured: out/walk_main.json, out/walk_ho_f2v025.json]
  - **Stance parent:** max abs(a) is 0.714 [measured].
  - **Hold:** actions are clipped in hold_test (max abs(a_target) 0.95). Its 'placed' initial state, however, is written directly to qpos/qvel followed by mj_forward. This is SUPPORT's documented initialisation, and the stage discloses it ("placed at zero velocity"). Strictly, though, it is a state write, not env.step [measured: feas/support/scripts/hold_test.py lines 209-224].
- **Rewards add up.**
  - Walking: r equals info["reward_total"] exactly on every step (max error 0.0). r equals the sum of the info reward_* terms to 4.4e-16, except on the terminating step of a fallen episode. There r minus the sum is exactly -150, the committed `fall_penalty = -150.0` (configs/tyrannosaurus_rex/anatomical/locomotion.toml line 10), which `BaseDinoEnv.step` adds after the terms and writes into reward_total (environments/shared/base_env.py lines 1355-1359) [measured: out/walk_main.json; code read].
  - Stance (parent and statue): r equals the sum of terms to 1.8e-15 [measured: out/stance_policy40.json, out/stance_statue0.json].
  - The gate panels' per-episode `reward` field is bitwise equal to the stage's [measured: out/cmp_gate.txt].
- **Seeds and configs** are the ones claimed. Per-seed results are bitwise equal on the walking seeds 0-9, 0-4 and 100-119 and on the stance seeds 3042-3081 (40 per cell, PUBLICATION_SEED_START), so the stage used these seeds [measured].
  - Walking uses the committed locomotion config (dt 0.01 s, 1000 steps, reset noise 0.05).
  - Stance uses the committed stance config (dt 0.01 s, 1000 steps, noise 0.05). The seed-0 statue overrides reset_noise_scale to 0 through an env kwarg, which the stage discloses.
  - The hold overrides reset noise to 0 and max_episode_steps to 2000.

  [measured: logs/smoke_info.txt]
  - The controller parameters match feas/existence/out/variant_ledger.json: FINAL3 {kz 8, dz 2, v_des 0.15, step_w 0.35, kby 1.5, max_w 0.8}, and f2v0.25 the same with v_des 0.25 and k_yp 0.05.
  - The copied wlib.py, defaults.py, hold_test.py, sslib.py and diag_creep.py have the same sha256 as their EXISTENCE and SUPPORT sources.
  - The stance checkpoint and vecnorm (sha256 1c9d87c3..., 5e417fef...) and opt_full.json (72165a3a...) match the stage's records.

  [measured: out/provenance_sha256.txt, CMDLOG 03:08:24 and 03:11:14]
  - The policy loads through the export's `load_sb3_checkpoint`, validated against the real plant identity. Its class is CriticPhaseActorCriticPolicy with clip_obs 10.
  - It runs actor only, `predict(deterministic=True)`. I normalised with the frozen VecNormalize's own `normalize_obs`, not the stage's hand-written formula, and the results are bitwise equal [measured].
- **No HYPOTHETICAL model drove a claim about the real plant.** The stage's real-plant statements rest on V0 runs or on config text. These are: the V0 baselines reproducing EXISTENCE, SUPPORT and the r10 statue values in stance.toml; "the plant's ceiling is unmeasured"; and "hip-roll authority does not bind at 0.2-0.3 m/s", inferred from V0 force fractions. None of them uses a V1-V3 number as evidence about r10 [checked by reading RESULT.md against the outputs].
- **HYPOTHETICAL label.** Every V1-V3 JSON of the stage carries it. Three stdout files do not: `out/stance_statue0_V1_stdout.txt`, `_V2_`, `_V3_` [measured: `grep -L` at 03:35:34].
- **Variant construction (worth knowing for any real change).**
  - With the stage's construction (priority 1 on the eight foot geoms), live foot-floor contacts carry solref 0.01 (V1) or 0.005 (V2). Non-foot floor contacts keep 0.02, and the timestep is 0.002 s.
  - If only the foot geoms' solref is changed, without the priority change, MuJoCo mixes it with the floor's 0.02 (solmix 1 / 1). The live contact solref is then 0.015 for a nominal 0.01, and 0.0125 for a nominal 0.005 (HYPOTHETICAL side-check).
  - V3: deliverable hip-roll torque at q = 0 with ctrl at its range end rises from 261.8 to 480.0 N.m (the forcerange cap).

  [measured: out/vcheck.json, out/vmix.txt]
- **Labels on interpretive sentences.** Most are labelled correctly. The exceptions are the three in correction 2 and the "ceiling" sentence in correction 1. "V3 is indistinguishable from V0" (stage section 4.3) sits under a [measured] header. The measured differences are small (tail_3 +0.0053 against +0.0074; foot slide median 5.4 against 6.4 mm), but "indistinguishable" is a judgement [inferred].

## 4. What I did not re-run (claims taken from the stage's saved outputs)

- Gate headings -90, -45, +45 and 180. The headings I did re-run (0 and +90) are bitwise equal to the stage's cells, and the stage itself shows that the headings are near-replicas.
- FINAL3 held-out seeds 100-119.
- The V3 "walker not told about kp" control (FINAL3 0.190-0.202 m/s).
- The stride and controller-clock stride comparison.
- SUPPORT's full creep diagnostic traces (diag_creep). My own pad-orientation probe on the same holds reproduces the e-folding times and the roll maxima.
- The foot-contact observation z-scores of the parent (V2 max 14.1 against clip 10).
- statue40 through the stage's stance.py. The gate's statue panel covers the same 40 seeds with the config noise: 40/40 full horizon on every variant at headings 0 and 90 [measured: out/misc_stats.txt].
- A scan of the stage folder for AI model names: it would have required writing those names into my own log.

## 5. Two-sided table (re-measured values; V1-V3 HYPOTHETICAL)

| metric | V0 real r10 | V1 HYPOTHETICAL foot solref 0.01 | V2 HYPOTHETICAL foot solref 0.005 | V3 HYPOTHETICAL hip-roll kp x2 |
|---|---|---|---|---|
| FINAL3 seeds 0-9: full; alternation min; speed m/s | 10/10; 1.000; 0.206-0.211 | 10/10; 1.000; 0.197-0.205 | 10/10; 1.000; 0.180-0.207 | 10/10; 1.000; 0.176-0.186 |
| f2v0.25 seeds 0-9: full; alternation min; speed m/s | 10/10; 1.000; 0.306-0.324 | 10/10; 1.000; 0.287-0.329 | 10/10; 1.000; 0.300-0.359 | 10/10; 1.000; 0.283-0.297 |
| f2v0.25 held-out 100-119: full | 20/20 | 19/20 | 18/20 | 20/20 |
| v_des 0.30 seeds 0-4: full (speed) | 0/5 | 0/5 | 0/5 | 2/5 (0.375-0.511) |
| v_des 0.35 seeds 0-4: full; fall s | 0/5; 3.15-3.72 | 0/5; 3.25-3.77 | 0/5; 3.02-3.76 | 0/5; 3.56-4.98 |
| FINAL3 hip roll abs(a) p95; servo err p95 deg | 0.65; 11.6 | 0.69; 12.0 | 0.75; 12.4 | 0.44; 6.1 |
| hold pose E right / left s | 2.21 / 1.83 | 10.00 / 3.45 | 10.00 / 5.26 | 2.14 / 1.80 |
| statue seed 0: held 10 s; pelvis m; head tip m | yes; 0.9261; 0.958 | yes; 0.9264; 0.959 | yes; 0.9265; 0.959 | yes; 0.9261; 0.958 |
| parent 3042-3081: full; falls; tail_3 mean; max abs(tail_3); any actuator >= 0.99 | 40/40; 0; +0.0074; 0.4518; 0 | 40/40; 0; +0.0275; 0.4522; 0 | 40/40; 0; +0.0393; 0.4532; 0 | 40/40; 0; +0.0053; 0.4519; 0 |
| parent head tip mean m; slide median / max mm | 0.903; 6.4 / 21.7 | 0.911; 6.8 / 23.1 | 0.918; 6.6 / 16.4 | 0.901; 5.4 / 20.4 |
| main's gate h0 and h90: parent clean (LCB); statue clean | 40/40 (0.928); 40/40 | 40/40 (0.928); 40/40 | 37/40 (0.817); 38/40 | 40/40 (0.928); 40/40 |

[measured: out/walk_main.json, out/walk_ho_f2v025.json, out/hold.json, out/stance_statue0.json, out/stance_policy40.json, out/gate_h0.json, out/gate_h90.json]. Reading [inferred]: the stage's two-sided reading holds as data. V1 lengthens the hold with gate metrics that move toward the bars but stay inside them. V2 lengthens it more but sits on the stance gate's bar. V3 leaves stance and the hold unchanged and walks slower at v_des 0.15-0.25, yet survives v_des 0.30 on 2/5 seeds. None of the variants brings this V0-tuned controller near 1.0 m/s.

## 6. Rule slips

**Stage, not in its own slip list:**
1. Three V1-V3 output files carry no HYPOTHETICAL label: `out/stance_statue0_V1_stdout.txt`, `out/stance_statue0_V2_stdout.txt` and `out/stance_statue0_V3_stdout.txt`. Their JSON companions do carry it.
2. Many read-only inspections, the patch heredocs and the RESULT.md assembly ran under the system `python3` instead of the designated venv interpreter, some without `-I` (stage CMDLOG lines 43, 70, 85, 107, 225 and 244-315). None of them was a simulation.
3. Some interpretive sentences are labelled [measured] (section 1, corrections 1 and 2).

**Mine:**
1. My first walking smoke (03:13:19) computed the CoM over all bodies, including body 1, the 65.45 kg mocap prey. It read 0.119 m/s against the stage's 0.210 m/s. I fixed it before any batch run, to use the animal's free-root subtree, and logged the fix at 03:13:57. The smoke output was overwritten by the fixed re-run.
2. Several commands are logged as descriptions rather than literal text: the Write-tool file creations; the python heredoc patches to vwalk.py and vv.py (03:13:57), vgate.py (03:15:53) and vhold.py (03:19:13); the sed edit to vstance.py (03:15:19); and a read-only stdin heredoc that printed the body tree (03:13:43, logged right after it ran).
3. The wait command at 03:35:56 is logged without its stderr redirect to `tmp/wait_err.txt`.
4. vhold.py was patched at 03:19:13 to add a read-only pad-orientation probe. The background chain that later ran it had already started on its stance steps; vhold.py itself had not yet started.
5. Read-only analysis (cmp_stance.py, sha256sum, grep) ran at 03:27-03:36 while a simulation process was running. None of it simulated, so the one-simulation-at-a-time rule held.
6. I made the venv by hand: `pyvenv.cfg`, a `bin/python` symlink, and a `.pth` to the optB and venv_sb3 site-packages. I did not run `python -m venv --without-pip`, because that tool writes activation scripts that contain a device-path token (the stage's slip 6). It is functionally a pip-less venv, but it departs from the literal instruction.
7. Background runs append their EXIT lines to CMDLOG when they finish, so those lines are interleaved with later commands.
8. CMDLOG lines 36 (03:09:46) and 109 (03:51:03) hold my two folder scans for device paths. They use only the generic directory prefix as the pattern and name no device. My folder holds no device path; the stage folder has no '/dev/' string at all.

## 7. Files

- Scripts:
  - `scripts/vv.py`: my variant implementation and helpers;
  - `scripts/vcheck.py`, `scripts/vmix.py`: variant checks;
  - `scripts/vwalk.py`, `scripts/vstance.py`, `scripts/vhold.py`, `scripts/vgate.py`: re-runs;
  - `scripts/cmp_walk.py`, `scripts/cmp_stance.py`, `scripts/cmp_gate.py`, `scripts/misc_stats.py`: comparisons, no simulation;
  - `scripts/smoke_info.py`.
- Outputs:
  - `out/vcheck.json`, `out/vmix.txt`;
  - `out/walk_main.json`, `out/walk_ho_f2v025.json`, `out/walk_smoke.json`;
  - `out/stance_statue0.json`, `out/stance_policy40.json`;
  - `out/hold.json`;
  - `out/gate_h0.json`, `out/gate_h90.json`, `out/gate_h0_maxima.txt`;
  - `out/cmp_walk.txt`, `out/cmp_stance.txt`, `out/cmp_gate.txt`, `out/cmp_gate.json`, `out/misc_stats.txt`, `out/provenance_sha256.txt`.
- Logs: `logs/*.txt`, `CMDLOG.txt`.
- Folder size: about 3 MB.
