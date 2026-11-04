# HOP: a hand-built two-footed hop on the real r10 T. rex

Stage folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/hop/`. Paths below are relative to it.

- **When:** 04:24-05:02 UTC on 2026-11-04. The time box was 06:45 UTC or 30 controller variants. I reached 30 variants at 04:49 and built no new controller after that. The later runs are validations: held-out seeds, deterministic replays and re-runs of earlier tags with the final code.
- **Rules kept:**
  - Nothing was trained or fitted.
  - The repository, the export (main 4d5f4e7), the configs and the physics are unchanged. No HYPOTHETICAL plant was built.
  - The plant is the export's own `build_env(ModelId("tyrannosaurus_rex","anatomical"), "locomotion")`. It was driven only through `env.step(action)`, and every action was asserted to lie in [-1, 1] on every step.
  - Every simulation ran as one process at a time through `scripts/sim.sh`, with `nice -n 10`, `OMP_NUM_THREADS=1` and `TMPDIR=tmp/`.
- **Interpreter:** `optB/venv/bin/python -I -B`. No extras were needed (nothing from the sb3 site-packages is imported), so I made no venv.
- **Units:** m, s, m/s, deg, BW (body weight 840.9 N). L = 0.8766 m (the probe's leg length). Froude Fr = v²/(gL).
- **Probe:** the frozen gait probe `docs/investigations/gait_2026_09/gait_probe.py`, loaded read-only from the export the way `feas/existence/scripts/probe_eval.py` loads it (sha256 in `out/SHA256SUMS_scripts_and_probe.txt`). I used its own `rollout_episode`, `analyze_episode` and floor-truth hook. I did not use the `stance_2026_10/pilot/gait_probe_pilot.py` adapter: it only wraps the same probe's `main()` (CLI and checkpoint path) for today's API.
- **Success at a speed (the stage's definition):**
  - the full 10 s (1000 steps) without termination, and
  - the probe's `floor_gait_class` over t >= 1 s is `hop`.

  `classify_biped` returns `hop` only when the simultaneous-touchdown fraction is at least 0.5 and the flight fraction is at least 0.05. I also report alternation.

## Bottom line

1. **A stable two-footed hop exists on the real plant at all three requested speeds, and up to 2.0 m/s.**
   - One parameter set ("F") covers every speed; only the speed command `v_mps` changes.
   - On every successful episode the probe classes the gait as `hop`. Over the 157 successful episodes:
     - simultaneous touchdowns >= 0.977 on 156. The exception is 0.816, on one 2.0 m/s held-out episode (seed 3067, alternation 0.125).
     - alternation 0.000 on 156
     - flight fraction 0.55-0.60
     - double support 0.39-0.45

   The probe classes all 160 episodes (falls included) `hop`. [measured: out/an_success_ranges.txt, out/summary_F.txt, out/summary_HV.txt]

   | speed command | tag (seeds 3042-3046) | success 3042-3046 | success 3047-3061 | success 3062-3081 | 3-10 s speed (m/s), all full episodes | Fr | hop frequency (Hz) | distance per hop (L) |
   |---|---|---|---|---|---|---|---|---|
   | 0.7 | H23_v070_F | **5/5** | 15/15 | 20/20 | 0.705-0.723 | 0.058-0.061 | 5.05-5.13 | 0.147-0.162 |
   | 1.0 | H24_v100_F | **5/5** | 15/15 | 20/20 | 1.007-1.036 | 0.118-0.125 | 4.86-5.15 | 0.218-0.242 |
   | 1.4 | H25_v140_F | **5/5** | 15/15 | 20/20 | 1.402-1.464 | 0.229-0.249 | 4.64-4.80 | 0.316-0.352 |
   | 2.0 | H20_v200_v010_T008 | **5/5** | 14/15 (3056 fell at 6.73 s) | 18/20 (falls at 7.60 and 8.31 s) | 1.983-2.121 | 0.457-0.523 | 4.45-4.93 | 0.481-0.517 |

   [measured: out/summary_F.txt, out/summary_HV.txt; the held-out tags are HO*_s3047_3061 and HV*_s3062_3081]
   - On the 40-seed panel (3042-3081) the hop succeeds on 40/40 at 0.7, 40/40 at 1.0, 40/40 at 1.4 and 37/40 at 2.0. [measured: same files]

2. **The fastest stable hop is 2.0 m/s (Fr 0.46-0.47 on seeds 3042-3046).** Above it:
   - 2.25 m/s: 1/5 (H26)
   - 2.5 m/s: 0/5 (H21) and 2/5 (H22)

   [measured: out/summary_all.txt] Its gait and actuator use are in the next two sections.

3. **What fails, at 2.0 m/s and above and in the early variants: the flight-phase leg swing cannot bring the feet forward in time.**
   - The feet land 0.1-0.2 m behind their touchdown target, behind the CoM. The next stance then accelerates the body.
   - The speed runs past the command (to 3-4 m/s), the pelvis pitches nose-down to 20-29 deg, the flight phase shrinks, and the animal falls.
   - Seen in: H03/H07 at 0.7 m/s (before the fix), H16 at 1.6 m/s, H26 at 2.25 m/s, and the held-out 2.0 m/s fall on seed 3056. [measured: out/an_H03_s3042.txt, out/an_H04_s3042_b.txt, out/an_H07_tds.txt, out/an_H16_tds.txt, out/an_H26_s3042_tds.txt, out/an_HO20_s3056_tds.txt]
   - The other two held-out 2.0 m/s falls (seeds 3066 and 3072, run without per-step records) were not traced.
   - [inferred: same mechanism each time, and the stance and torque budgets are not exhausted (actuator section)] The 2.0-2.25 m/s ceiling belongs to this controller, not to the plant.

4. **A hop earns more from main's locomotion reward than a walk at the same speed.**
   - Same seeds 3042-3046, F hop at 0.71 m/s against the SPEED stage's V24 walk at 0.69 m/s:
     - +64.5 per episode at forward weight 2.0 (+54 to +76 per seed)
     - +36.7 per episode at weight 0.2

     [measured: out/an_diff_hop070_vs_walk070.txt]
   - The gain at weight 2.0, by term:
     - forward: +30.9
     - height: +27.0 (the hop's pelvis rides at 0.91 m)
     - lateral sway: +3.0
     - heading: +2.9
     - nosedive: +1.9
     - tail: -1.2
   - No term charges the hop's flight or its landings, even though:
     - 55-60% of physics substeps have both feet unloaded (env `airborne_substeps` 2.76-2.98 of 5 per step)
     - the per-step peak summed foot force is 3.5-4.3 BW at p95 and up to 9.0 BW, against 1.54 BW p95 for the walk

     [measured: out/ana_main.txt] Every support, impact and airborne reward term is 0 in the recorded components. [measured: per-step `comp` arrays in out/*_steps.npz]
   - Returns at weight 2.0 on seeds 3042-3046:

     | behaviour | return |
     |---|---|
     | statue | 1090.8-1096.1 |
     | FINAL3 walk | 1225.0-1245.3 |
     | f2v0.25 walk | 1291.9-1309.6 |
     | V24 walk, 0.69 m/s | 1467.7-1496.9 |
     | F hop, 0.7 m/s | 1537.2-1553.5 |
     | F hop, 1.0 m/s | 1725.0-1754.2 |
     | F hop, 1.4 m/s | 1947.7-2006.4 |
     | F hop, 2.0 m/s | 2263.2-2292.1 |

     [measured: out/an_reward_stage_refs.txt (the REWARD stage's own JSONs, read-only), out/summary_all.txt]
   - At weight 0.2 (the start of the ramp), the V24 walk earns 1075.3-1084.4, below the statue's 1091.7-1097.6. The 0.7 m/s hop earns 1113.4-1117.7, above the statue. [measured: same files]

5. **The benchmark gate (episode-mean forward_vel >= 1.0 m/s, length >= 750, no gait check) passes the hop at the 1.4 command but not at 1.0.**
   - 1.4 command: 40/40 on 3042-3081 (episode means 1.080-1.185 m/s)
   - 1.0 command: 0/40 (0.783-0.858; the 0.5 s settle and the 2 s start ramp pull the mean down)
   - 2.0 command: 39/40 (two of its falls came after step 750)

   [measured: out/an_gate.txt, out/an_gate_HV.txt]

6. **Stepping in place (the walker, near-zero speed): done.**
   - W02 (V24 with v_mps 0, ki_v 0, v_des -0.016) runs 5/5 full episodes.
   - Probe class `stand (stepping / tapping in place)`: alternation 1.000, 1.79 Hz strides, 0.025 m/s, no flight.
   - Return at weight 2.0: 1038.6-1050.6, below the statue.
   - W01 (v_des 0) creeps at 0.045 m/s, so the probe classes it `walk` (its threshold is 0.03 m/s).

   [measured: out/summary_all.txt, out/ana_main.txt]

7. **Shuffle (feet slide, no lift): one attempt, failed.** W03 (V24 at 0.3 m/s with swing clearance 0) fell on 5/5 seeds between 5.0 and 8.2 s, class `irregular / other`. Its trajectories are saved. [measured: out/summary_all.txt]

8. **Records.** Every episode in the main runs has a per-step file `out/<tag>_s<seed>_steps.npz` holding:
   - the raw actions
   - the env reward
   - all 31 `reward_*` components
   - all 85 scalar info entries (forward_vel, pelvis_height, foot contacts, airborne_substeps, peak_foot_force_bw, bilateral_support_quality, ...)
   - the fall penalty, the probe's floor forces, root position, CoM, joint angles and the controller phase

   Checks:
   - On every step of every episode: the env reward equals the sum of the components plus the fall penalty, to 1.3e-15 or better, and the return equals the probe's own total exactly. [measured: out/summary_all.txt "reward resid max"]
   - Re-rolling six saved episodes open loop from their recorded actions reproduces reward, all components, all info entries and the root trajectory **bit for bit**. Two of them are falls. [measured: out/replay_check.json]
   - Recomputing the forward term from the recorded forward_vel as w·clip(v/2.5, -1, 1) matches the env's term exactly (difference 0.0). [measured: out/replay_check.json]
   - The weight-0.2 returns are this exact recomputation at w = 0.2. [inferred: the forward term is the only weight-dependent term, and weights do not enter the dynamics]

## The controller (scripts/hoplib.py, class Hopper)

It reuses the walker's machinery: the EXISTENCE `wlib.Walker`, unchanged (sha256 in out/copied_sha256.txt). That covers:
- read-only sensing on a private MjData copy
- damped-least-squares leg IK
- the ctrl to action map
- inversion of the 10 Hz command filter
- the SPEED stage's swing feed-forward recipe: gravity, passive damping from the desired joint speed, M·qdd_des, kd 45, sagittal joints only

Both legs act as one virtual leg, in the style of a Raibert hopper:

- **Stance (both feet down).** A virtual-model wrench on the body, split half to each pad; each leg gets tau = -Jᵀ·W, and the servo ctrl = q + (tau + spring + 0.6·c·qd)/kp.
  - Vertical: a half-sine impulse feed-forward m·π(v0 - hd0)/(2T)·sin(πt/T) over T = T_st, plus a weak PD on CoM height above the mean pad level (h_to = 0.95 L).
  - Horizontal: the GRF points from the mid-pad point to the CoM.
  - Pelvis roll/pitch PD couple: the roll part is made as a left/right vertical-force difference across the stance width; the pitch part goes through the pads.
- **Flight (from t = T_st).** Each foot follows a min-jerk path, relative to the CoM in the heading frame, from its lift-off position to the touchdown target, with a 4 cm mid-flight retraction.
  - x = v·T_st/2 + k_v·(v - v_cmd) - k_i·v_int (Raibert neutral point plus speed feedback). v_int is a per-hop speed integral, active after the 2 s start ramp.
  - y = ±0.15 m + lateral-velocity terms
  - z = -h_to
  - Both legs go through IK with the pad orientation left free (w_pitch_ik 0).
- **Touchdown:** either foot's touch force above 0.08 W after half the predicted flight time.

Nothing reads world position: every quantity is a difference of two world points or a velocity. [inferred: by construction; not tested with a shifted-root shadow]

The F set (out/an_params_F.txt) changes these from the defaults: vmode 1, v0 1.0 m/s (the commanded takeoff speed; the measured CoM vertical speed at the end of stance is 0.43-0.59 m/s, per-episode medians over the successful F episodes [measured: out/an_success_ranges.txt]), T_st 0.08 s, k_v 0.25 s, ki_v 0.15, k_i 0.25 s, w_pitch_ik 0. Start: 0.5 s at the home pose, then a 0.4 s crouch-and-push first stance and a speed ramp from 0 over 2 s.

## Fastest stable hop: H20, 2.0 m/s, seeds 3042-3046

Gait, probe window t = 3-10 s unless marked [measured: out/summary_all.txt, out/ana_main.txt]:
- speed 1.985-2.011 m/s (forward_vel 1.972-2.004); 5-10 s 2.001-2.018 m/s; Fr 0.458-0.470
- hop frequency 4.53-4.62 Hz; distance per hop 0.493-0.502 L (0.43-0.44 m); duty 0.40-0.41 per foot
- t >= 1 s:
  - simultaneous 1.000, alternation 0.000
  - flight fraction 0.576-0.586, double support 0.409-0.414
  - probe class `hop` on 5/5
- touchdowns: L and R within 1 control step on 220 of 224 events. Lift-offs: same step on 207 of 224, within 1 step on 15, 5 steps apart on 2.
- flight in 215 of 217 cycles; airborne control steps per cycle median 11 (0.11 s), maximum 23
- swing clearance 0.087-0.092 m; foot slip in contact 0.21-0.23 m/s
- pelvis pitch std 3.8-4.9 deg, roll std 0.1-0.7 deg; heading change -0.6 to +0.9 deg; pelvis height 0.934-0.938 m
- the per-second speed oscillates between about 1.7 and 2.2 m/s (speed integral against hop-to-hop placement). [measured: per-second v in out/H20_v200_v010_T008_stdout.txt]
- peak summed foot force per step: 3.97-4.23 BW at p95, maximum 6.2-7.9 BW. Airborne substeps 2.90-2.94 of 5 per step.
- leg joint speed p95: hip pitch 8.6-9.3, knee 9.9-10.5, ankle 9.3-9.9, hip roll ≤ 0.3 rad/s

Actuator use, both legs, seeds 3042-3046 [measured: out/an_audit_both_legs.txt; definitions as in the SPEED stage: "deliverable" = kp·(ctrl-range end - q) toward the force, clipped to forcerange]:

| | hip pitch | hip roll | knee | ankle |
|---|---|---|---|---|
| peak force / forcerange | 0.72-0.80 | 0.04-0.21 | 0.52-0.57 | 0.52-0.55 |
| p95 force / forcerange | 0.47-0.51 | 0.03-0.04 | 0.24-0.26 | 0.34-0.37 |
| p95 force / deliverable | 0.62-0.65 | 0.05-0.07 | 0.40-0.46 | 0.66-0.69 |
| peak force / deliverable | 0.86-0.91 | 0.08-0.36 | 0.63-0.90 | 0.88-0.93 |
| steps with demand beyond the ctrl range | 0.3-1.7 % | 0 | 0-0.9 % | 6.1-9.0 % |
| steps with the raw action clipped at ±1 | 5.4-8.1 % | 0 | 1.9-4.8 % | 18.7-22.1 % |
| minimum margin to a joint stop (deg) | 27.5 | 20.9 | 6.7 | 18.3 |

- [inferred: from the table] No servo reaches its forcerange: the peak is hip pitch at 0.80.
- [inferred: from the table] The binding actuator is again the ankle: about 9 in 10 of its deliverable torque at peak, demand beyond its range on 6-9 % of steps, and its raw action clipped on about a fifth of steps.
- [inferred: from the table] Hip roll is almost idle (p95 4 % of forcerange): the two-footed stance needs almost no roll torque, unlike the alternating walk, where hip roll used 50-55 % of its deliverable torque at p95 (SPEED, 1.2 m/s).
- At 0.7 m/s (H23) the same table reads: peak/forcerange ≤ 0.43, ankle deliverable p95 0.64, ankle raw clip 16-18 %, knee margin ≥ 24.9 deg. [measured: out/an_audit_both_legs.txt]

## How the hop was found (30 controller variants, in run order)

Seeds 3042-3046 unless a count says otherwise; "falls" are times in s. [measured: out/summary_all.txt; H03 from out/H03_v070_kv25_5seed_rerun.json]

| # | tag | change from the previous | result |
|---|---|---|---|
| 1 | H01_v070_base | first hopper (cubic height reference, T_st 0.18, v0 0.45, k_v 0.10), 1 seed | full 10 s, class hop, but 0.34 m/s |
| 2 | H02_v070_ki | + speed integral (ki_v 0.5, k_i 0.1) | 0/5, falls 4.9-5.1 (speed runaway after the ramp) |
| 3 | H03_v070_kv25 | no integral, k_v 0.25 | 0/5, falls 4.1-4.4 |
| 4 | H04_v070_sine | half-sine vertical impulse (vmode 1), v0 0.6, T_st 0.16; 2 seeds | 0/2, falls 4.1, 4.6 |
| 5 | H05_v070_kx4 | + direct stance speed force kx_st 4 | 0/5, falls 3.7-4.0 (worse) |
| 6 | H06_v070_kx4_v010 | + v0 1.0 | 0/5, falls 3.5 |
| 7 | H07_v070_T010 | H04 with T_st 0.10 | 0/5, falls 4.2-5.1 |
| 8 | H08_v070_nopitchik | H07 + no pad-pitch term in the flight IK (w_pitch_ik 0); 2 seeds | **2/2, 0.89 m/s** |
| 9 | H09_v070_noff | H07 without swing feed-forward and kd; 2 seeds | 2/2, 0.46 m/s |
| 10-14 | H1x_v{070,100,140,180,220}_ki | H08 + ki_v 0.15, k_i 0.25 | 5/5 at 0.7, 1.0, 1.4 (0.73-0.76, 0.99-1.07, 1.40-1.47 m/s); 0/5 at 1.8 and 2.2 |
| 15 | H15_v180_ki005 | 1.8, ki_v 0.05 | 0/5 |
| 16 | H16_v160_ki015 | 1.6 | 1/5 |
| 17 | H17_v160_v010 | 1.6, v0 1.0 | 5/5, 1.57-1.66 m/s |
| 18 | H18_v160_T008 | 1.6, T_st 0.08 | 5/5, 1.63-1.65 m/s |
| 19 | H19_v200_v010 | 2.0, v0 1.0, T_st 0.10 | 0/5 |
| 20 | **H20_v200_v010_T008** | 2.0, v0 1.0, T_st 0.08 (= the F set) | **5/5, 1.99-2.01 m/s** |
| 21 | H21_v250_v010_T008 | 2.5, F | 0/5 |
| 22 | H22_v250_v012_T007 | 2.5, v0 1.2, T_st 0.07 | 2/5 |
| 23-25 | H23/H24/H25 (_F) | F set at 0.7, 1.0, 1.4 | 5/5 each |
| 26 | H26_v225_F | F at 2.25 | 1/5 |
| 27 | W00_V24_walk_v070 | SPEED's V24 walker unchanged, as the walk reference on these seeds (counted to be safe) | 5/5 walk, 0.686-0.692 m/s |
| 28 | W01_step_in_place | V24 with v_mps 0, v_des 0, ki_v 0 | 5/5 full, class walk (0.045 m/s creep) |
| 29 | W02_step_in_place_vdesm016 | + v_des -0.016 | 5/5 full, class stand (stepping in place), 0.025 m/s |
| 30 | W03_shuffle_v030_noclear | V24 at 0.3 m/s, h_clear 0, land_push 0 | 0/5, falls 5.0-8.2, irregular |

The decisive change was #8, removing the pad-pitch term from the flight IK. The evidence chain:
- Before it, the feet at touchdown sat 0.04-0.20 m behind the flight target, and the gap grew with speed. [measured: out/an_H07_tds.txt]
- Shorter stance (#7) and a stronger push (#6) did not remove the runaway; a direct stance speed force (#5) made it worse.
- With the term removed, the same settings survive. [measured: out/summary_all.txt]
- [inferred] The pad-level IK constraint forced large ankle motions that the ankle servo could not track within a 0.07-0.12 s flight, holding the whole leg back.

Not done (I was at the variant cap): yaw control; a shorter-flight, higher-cadence hop like the September policies' 7.7-9.2 Hz (this hop runs at 4.5-5.2 Hz); a second shuffle design.

## Re-run and determinism checks (no new controller)

- H01 was run before the speed-integral patch, and H03 before the vmode patch.
  - H01 re-run with the final code: same final root xy and same return, bit for bit.
  - H03 re-run on 5 seeds: identical lengths 444/408/441/436/413; seed 3042 bit for bit.

  [measured: out/an_rerun_check.txt, out/H03_v070_kv25_5seed_rerun.json]
- H02 (run between the two patches, with vmode 0, whose code path the vmode patch left unchanged) was not re-run. [inferred: unchanged code path]
- Six episodes re-rolled open loop from recorded actions: bit for bit. [measured: out/replay_check.json]

## Limits

- One controller family, hand-tuned within 30 variants. 2.0 m/s is a lower bound on what the plant can hop. The ceiling between 2.0 and 2.25 m/s is this controller's. [inferred: the failure mechanism is the flight-phase swing; no actuator is at forcerange]
- The 2.0 m/s hop is near its edge: 3 falls in 40 seeds, per-second speed swinging ±0.25 m/s, and one held-out success with simultaneous fraction 0.816 (seed 3067). The 1.4 m/s hop is 40/40, but its per-second speed also oscillates (about 1.0-1.7 m/s on seed 3042). [measured: out/summary_HV.txt, out/H25_v140_F_stdout.txt]
- Seeds are reset seeds of the locomotion config (reset noise 0.05). There are no pushes, no yaw control and no command changes. Episodes are 10 s.
- The controller keeps internal state (hop phase timer, speed integral) and reads the env's MjData read-only. The observation has no clock.
- The hop's landings reach 3.5-4.3 BW p95 per step and up to 9 BW. The stance reward priced exactly this (floor impact 2.0 above 1.4 BW, airborne substeps 1.0). I did not compute what those terms would cost: that is a HYPOTHETICAL reward setting, outside this stage. The recorded `peak_foot_force_bw`, `airborne_substeps` and `bilateral_support_quality` per step are there for it.
- The height-term advantage of the hop over the walk (+27 per episode) partly reflects the walker's chosen crouch (CoM target 0.93 L) against the hop's 0.95 L takeoff height. It is not intrinsic to hopping. [inferred]
- Physics and behaviour: no repository code, config or physics was changed, so the stance behaviour is not affected. The controller is an analysis artefact.

## Slips

1. Read-only commands from 04:24 to 04:29, before CMDLOG.txt existed, are listed after the fact in the first NOTE line. These were reads of the SPEED and VERIFY reports, the walker scripts, FACTS.md, the REWARD rollout script and the probe source, plus one read-only JSON print.
2. The system `python3` was used, without -I -B, for string-replace patches of my own scripts:
   - hoplib.py twice (speed integral; vmode 1)
   - runh.py twice (controller-trace fields; `nosave` option)
   - replay.py once (root-position definition)

   They are summarised in NOTE lines, not quoted literally. One sed patch of hoplib.py (the `mj_fullM` argument order) is quoted in a NOTE. Scripts written with the file-write tool (hoplib.py, runh.py, summarize.py, replay.py, ana.py) are NOTE lines too. No simulation ran outside `scripts/sim.sh`.
3. Three bugs of my own:
   - The first `inspect.py` run crashed (wrong argv order). I renamed it to inspect_env.py and fixed it.
   - hoplib.py first used the old `mj_fullM` argument order. I fixed it before the first run.
   - replay.py first compared qpos[0:3] with the probe's `root_pos`, which is `xpos` of the pelvis. That gave a spurious 2-9 mm "mismatch". After the fix the comparison is bit for bit; the first output was overwritten.
4. H03's 5-seed JSON and stdout were overwritten by a 1-seed re-run under the same tag, done to save a controller trace. The 5-seed outcome was restored by a deterministic re-run under the tag `H03_v070_kv25_5seed_rerun`, with matching lengths. The original 5-seed file is gone.
5. Some commands piped sim.sh's console copy into display filters (cut, awk, a read-only JSON formatter). The logged sim.sh line is literal, and each run's full stdout file is in out/. The two held-out loops ran in the shell tool's background, still one process at a time.
6. Validation tags HV* (seeds 3062-3081) and the two re-runs were run with `nosave`, so they have JSONs but no per-step npz. H08, H09 and H04 ran on 2 seeds and H01 on 1 seed.
7. W00 is the SPEED stage's V24, unchanged. I counted it as a variant, which may have used one variant early.
8. My analysis scripts summarise the right leg in `summary_*.txt`. The both-leg table above comes from `out/an_audit_both_legs.txt`.

## Files

- `CMDLOG.txt`: literal commands with UTC times, plus NOTE lines.
- `scripts/hoplib.py`: the hop controller. `scripts/runh.py`: the runner, with the frozen probe, actuator audit and per-step recorder. `scripts/sim.sh` and `scripts/x.sh`: the wrappers.
- Analysis scripts (no simulation): `scripts/summarize.py`, `ana.py`, `gate.py`, `tds.py`, `hops.py`, `grf.py`.
- `scripts/replay.py`: open-loop re-roll of recorded actions.
- Unchanged copies of `wlib.py`, `wlib2.py`, `defaults.py`.
- `out/<tag>.json` and `out/<tag>_stdout.txt` for every run. `out/<tag>_s<seed>_steps.npz` holds the per-step record (see 8 above). Episodes can be regenerated from the params in the JSON plus the seed, or re-rolled from the saved actions.
- Summaries:
  - `out/summary_all.txt|json`: every tag
  - `out/summary_F.txt|json`: the F family with seeds 3047-3061
  - `out/summary_HV.txt|json`: seeds 3062-3081
  - `out/ana_main.txt|json`: cycles, forces and reward terms
  - `out/an_gate*.txt`, `out/an_diff_hop070_vs_walk070.txt`, `out/an_audit_both_legs.txt`, `out/an_*_tds.txt` (per-touchdown traces)
  - `out/replay_check.json`, `out/an_rerun_check.txt`
- `out/SHA256SUMS_scripts_and_probe.txt`.
- Folder size: about 170 MB.
