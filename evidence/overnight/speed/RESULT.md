# SPEED: speed ceiling of a hand-built alternating walk, real r10 T. rex

Stage folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/speed/` (all paths below are relative to it).
Worked 00:42-01:37 UTC on 2026-11-04 (time box 04:45 UTC). I stopped at 37 of the 40 counted variants: 35 controller variants on the real plant and 2 HYPOTHETICAL plant probes (ledger: `out/variant_ledger.json`).
Nothing was trained or fitted. The repository, the main 4d5f4e7 export and the physics are unchanged. Every simulation ran one process at a time through `scripts/sim.sh` (nice -n 10, OMP_NUM_THREADS=1, TMPDIR inside the folder). The plant was the export's own `build_env(ModelId("tyrannosaurus_rex","anatomical"), "locomotion")`, driven only by `env.step(action)` with every action in [-1, 1] (asserted on every step). No world-position feedback was used: `k_yp` = 0 throughout, and every controller quantity is a difference of two world points or a velocity.

Units: m, s, m/s, deg, rad/s, N.m. L = 0.8766 m, the probe's leg length (mean hip-pitch anchor height at the home keyframe). Froude Fr = v^2/(g L).

## Bottom line

1. **Fastest speed that passes the stage's success test: 1.2 m/s** (Fr 0.17).
   - V35: 5/5 on reset seeds 0-4; 15/15 full 10 s episodes on seeds 0-14 (14/15 pass every criterion; seed 8 has one airborne control step). 5-10 s speed 1.194-1.211 m/s. [measured: out/V35_v120_T030.json, out/summary_ladder.json]
   - V19: 5/5 on seeds 0-4 at 1.184-1.227 m/s; 9/10 on held-out seeds 5-14 (seed 11 fell at 8.15 s). [measured: out/V19_v120.json, out/H19_v120_seeds5_14.json]
   - The 0.5, 0.7 and 1.0 m/s rungs all pass. The gate speed of 1.0 m/s passes on 15/15 seeds (V18 + held-out). [measured: out/summary_ladder.json]
2. **The EXISTENCE failures at 0.88 m/s and above were controller-limited, as suspected.** Eight controller changes, each traced to a failure mechanism seen in a trace, took the stable walk from 0.31 m/s to 1.2 m/s. The list is under "What changed from FINAL3" below. [measured: out/variant_ledger.json and the per-variant JSONs]
3. **What fails first above 1.2 m/s depends on the swing time; it is the controller first, not hip roll, pad roll, torque or joint range.**
   - Same swing time (T_sw 0.32 s) at 1.35 m/s: 0/5. A speed overshoot, irregular steps, the step-length cap and late touchdowns produce a runaway.
   - Shorter swing (0.28 s): 1.35 m/s survives 15/15, but brief landing-bounce airborne steps appear (1-3 per episode, at most 0.33% of control steps on 7/15 seeds). They fail the strict no-flight test first.
   - T_sw 0.25 s: 1.5 m/s survives 15/15 with airborne steps on all 15 (0.2-1.4%).
   - 1.8 m/s: 0/5.
   - Plant-side signals grow with speed (swing-leg ankle servo span against passive joint damping), but two HYPOTHETICAL plant relaxations did not move the limit. The plant's own ceiling is therefore still unmeasured; it is at least 1.2 m/s for a strict walk and 1.5 m/s for a walk with rare bounces. [measured: out/summary_ladder.json, out/an_*.txt; inferred: from the HYPOTHETICAL probes P1/P2]
4. **No running gait appeared.** Every surviving episode is probe class "walk", with double support 0.06-0.21 (0.13-0.21 at 1.0 m/s and above). Flight shows up only as isolated landing bounces and in the runaway just before a fall. [measured: out/summary_ladder.json, out/an_flight_vs_speed.txt]

## Definitions used for "success"

All definitions use the project's frozen gait probe (`docs/investigations/gait_2026_09/gait_probe.py`, loaded read-only from the export; its own `rollout_episode` / `analyze_episode` / floor-truth hook).

- **full**: 1000 control steps (10 s), not terminated.
- **alt**: floor `alternation_index` >= 0.9 over the walking phase. The probe window starts at t = 1 s; the walk starts at t = 0.5 s.
- **no flight**: floor `flight_fraction` = 0 over the same window. A control step is airborne when neither foot touches the floor on at least half of its five physics substeps (the probe's rule).
- **strict** = full and alt and no flight.
- **Steady numbers** (speed, stride, cadence, duty, double support, Fr) come from the probe window t = 5-10 s, after every speed ramp has ended. Variants run before V18 have only the 3-10 s window.

## Speed ladder (real plant)

| speed command | variant | T_sw (s) | seeds | full | alt>=0.9 | no flight | 5-10 s speed (m/s) | stride (L) | stride freq (Hz) | duty | double support | Fr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v_des 0.25 (no k_yp) | V01 | 0.45 | 0-4 | 5/5 | 5/5 | 5/5 | 0.306-0.311 (3-10 s) | 0.32-0.33 | 1.07 | 0.55-0.57 | 0.12 | 0.011 |
| 0.5 | V05 | 0.32 | 0-4 | 5/5 | 5/5 | 5/5 | 0.526-0.532 (3-10 s) | 0.42 | 1.42 | 0.52-0.54 | 0.06 | 0.032 |
| 0.5 | V23 (final controller) | 0.32 | 0-4 | 5/5 | 5/5 | 5/5 | 0.479-0.490 (2-4 % under) | 0.30 | 1.83-1.85 | 0.54-0.57 | 0.11 | 0.027 |
| 0.7 | V24 | 0.32 | 0-4 | 5/5 | 5/5 | 5/5 | 0.699-0.703 | 0.42-0.43 | 1.85 | 0.58-0.60 | 0.18 | 0.057 |
| 1.0 | V18 + H18 | 0.32 | 0-14 | 15/15 | 15/15 | 15/15 | 0.984-1.010 | 0.51-0.57 | 1.98-2.22 | 0.58-0.61 | 0.17-0.21 | 0.11-0.12 |
| 1.2 | V19 + H19 | 0.32 | 0-14 | 14/15 | 15/15 | 14/15 | 1.184-1.227 | 0.61-0.69 | 1.97-2.22 | 0.58-0.61 | 0.19-0.21 | 0.16-0.18 |
| 1.2 | **V35** | 0.30 | 0-14 | **15/15** | 15/15 | 14/15 | **1.194-1.211** | 0.58-0.67 | 2.06-2.35 | 0.56-0.60 | 0.15-0.18 | 0.166-0.171 |
| 1.2 | V26 | 0.28 | 0-14 | 15/15 | 15/15 | 9/15 | 1.172-1.203 | 0.53-0.61 | 2.23-2.46 | 0.57-0.60 | 0.16-0.18 | 0.16-0.17 |
| 1.35 | V21 | 0.32 | 0-4 | 0/5 (falls 5.9-8.5 s) | - | - | - | - | - | - | - | - |
| 1.35 | V27 + H27 | 0.28 | 0-14 | 15/15 | 15/15 | 8/15 | 1.338-1.354 | 0.66-0.70 | 2.18-2.32 | 0.57-0.59 | 0.14-0.17 | 0.21 |
| 1.35 | V34 | 0.30 | 0-4 | 4/5 (fall 9.24 s) | 5/5 | 4/5 | 1.357-1.361 | 0.76-0.77 | 2.01-2.05 | 0.57-0.59 | 0.16-0.17 | 0.21 |
| 1.5 | V20 / V28 | 0.32 / 0.28 | 0-4 | 0/5 / 2/5 | - | - | - | - | - | - | - | - |
| 1.5 | V29 + H29 | 0.25 | 0-14 | 15/15 | 14/15 | 0/15 | 1.494-1.546 | 0.46-0.75 | 2.30-2.62 | 0.55-0.60 | 0.13-0.17 | 0.26-0.28 |
| 1.8 | V33 | 0.25 | 0-4 | 0/5 (falls 6.7-9.7 s) | - | - | - | - | - | - | - | - |

[measured: out/summary_ladder.json (rows built by scripts/ladder.py from out/<tag>.json); V01, V05 3-10 s numbers from out/V01_f2v025_noyp_stdout.txt and out/V05_v050_stdout.txt]

- Cadence in steps per second is twice the stride frequency: about 4.2 steps/s (0.24 s per step) at 1.2 m/s.
- The FINAL3 baseline (0.21 m/s) reproduced bitwise with both the unchanged EXISTENCE controller and this stage's controller with every new option off. Final root xy matches the EXISTENCE probe JSON exactly on seeds 0-2. [measured: out/base_final3_orig.json, out/base_final3_w2.json]
- Whole-episode mean CoM forward speed, including the 0.5 s settle and the start ramp, from the controller's own log (not the env's `info['forward_vel']`): 0.76-0.78 m/s at the 1.0 command, 0.89-0.91 at 1.2, 1.006-1.021 at 1.35, 1.13-1.15 at 1.5. [measured: out/an_episode_mean.txt]
  - [inferred: from the gate's episode-mean definition in locomotion.toml:139] A walk at a steady 1.0 m/s, started from rest the way I start it, would not by itself average 1.0 m/s over a 10 s episode.

## Fastest success in detail: V35, 1.2 m/s, 15 seeds

Gait, steady window 5-10 s [measured: out/summary_ladder.json, out/an_show_budget.txt, out/an_steps_stats.txt]:
- speed 1.194-1.211 m/s; Fr 0.166-0.171
- stride 0.58-0.67 L (mean 0.64 L, 0.56 m) at 2.06-2.35 Hz; step 0.24 s
- duty 0.56-0.60; double support 0.15-0.18
- floor alternation (t >= 1 s) 0.923-1.000; probe class "walk" on 15/15
- one airborne control step (seed 8) in 15 x 900; isolated airborne physics substeps 8-18 per episode (16-36 ms in total) on all seeds
- swing clearance 0.048-0.068 m; foot slip in contact 0.17-0.21 m/s
- pelvis roll std 0.8-1.0 deg, pitch std 2.0 deg
- heading drift -22 to +6 deg over 10 s (no yaw control); minimum lateral foot separation 0.25-0.31 m
- 0 forced (time-out) stance switches

Actuator use on the leg servos, min-max over 15 seeds [measured: out/an_show_budget.txt, from out/V35_v120_T030.json]:

| | hip pitch | hip roll | knee | ankle |
|---|---|---|---|---|
| peak force / forcerange | 0.45-0.52 | 0.31-0.43 | 0.42-0.44 | 0.42-0.48 |
| p95 force / forcerange | 0.32-0.33 | 0.26-0.28 | 0.19-0.22 | 0.32-0.34 |
| p95 force / deliverable servo torque at that angle | 0.44-0.45 | 0.50-0.55 | 0.43-0.46 | 0.58-0.64 |
| peak force / deliverable | 0.52-0.58 | 0.62-0.97 | 0.59-0.78 | 0.91-0.99 |
| control steps with demand beyond the ctrl range | 0.8-1.8 % | 0-0.2 % | 0.1-1.0 % | 1.6-2.3 % |
| control steps with the raw action clipped at +-1 (filter inversion) | 4.4-5.2 % | 0.8-1.6 % | 4.8-6.9 % | 8.8-11.2 % |
| minimum margin to a joint stop (deg) | 29.6 | 11.8 | 9.1 | 12.1 |

"Deliverable" means kp x (ctrl-range end - q) in the direction of the force, clipped to forcerange. That is the position servo's real authority, as flagged by VERIFY for hip roll. The stance-pad roll angle stays within p5/p95 -5.5/+5.0 deg.

- [inferred: from the table] At 1.2 m/s the T. rex walks with every leg servo below its forcerange (peak at most 0.52). The ankle is the actuator that uses most of its deliverable envelope. Hip roll keeps about 45-50% of its deliverable torque in reserve at p95 and almost never asks for more than its range.
- Swing-leg joint speeds at 1.2 m/s, p95: hip pitch 9.0, knee 10.6, ankle 10.0 rad/s. [measured: out/an_swing_speed.txt]

## What fails first above 1.2 m/s, with evidence

**A. Same controller (T_sw 0.32 s) at 1.35 m/s: controller runaway, not a plant limit.**
- V21 is 0/5. Before each fall the steps become irregular, alternating long and short between 0.32 and 0.59 L. The swing target sits on the controller's 0.6 L step cap. A late touchdown (0.41 s step, sph 1.28) lets the capture point run ahead, the speed climbs to 1.75-1.97 m/s, and the robot falls. [measured: out/an_limits_V21.txt]
- In that window the stance hip-roll demand |b| stays at or below 0.89 and pad roll within +-4 deg. They exceed their range only during the final collapse (|b| 3.4-3.6). [measured: out/an_limits_V21.txt]
- More reach (deeper crouch CoM 0.90 L, reach clamp 0.58 L, step cap 0.8 L; V22) made it worse: 1/5. [measured: out/an_limits_V22.txt, out/an_V22_window.txt]
- A shorter swing (T_sw 0.28 s) at the same speed survives 15/15 (V27 + H27). [measured: out/summary_ladder.json]
- [inferred: same plant, controller timing changed] The 1.35 m/s failure at T_sw 0.32 is controller-limited (step timing).

**B. Faster cadence: the strict no-flight criterion fails first, through landing bounces.**
- At 1.35 m/s (T_sw 0.28) 7/15 seeds have 1-3 airborne control steps (0.11-0.33%). At 1.5 m/s (T_sw 0.25) all 15 do (0.22-1.44%). [measured: out/summary_ladder.json]
- Where they occur (V29, 1.5 m/s, seed 0): about 0.03 s after a stance switch (sph 0.12). The new stance pad touches the floor on only 2 of 5 substeps while the trailing foot has already lifted, and the CoM moves up at 0.13-0.21 m/s. [measured: out/an_air_events_V29.txt]
- They are not a flight phase: double support stays at 0.13-0.17 and the probe classes every episode "walk". [measured: out/summary_ladder.json]
- Cause not identified within the variant cap. Three candidates did not remove them:
  - a stiffer foot contact (HYPOTHETICAL P1, solref 0.005): worse, 2/5 survive, airborne steps 0.7-4.9% [measured: out/P1_HYPOTHETICAL_stiffpad_V29_v150.json]
  - doubled vertical damping in the stance height loop (V36): airborne steps on 4/5 [measured: out/V36_v135_T028_dz4.json]
  - switching stance only at 0.4 W of new-foot load (V32): 3/5 survive [measured: out/V32_v135_td04.json]
  - [inferred: from these three] The soft pad is not the cause. The bounces most likely come from the landing/stance-switch strategy at high cadence (a controller matter), but this is not shown.

**C. Survival ceiling between 1.5 and 1.8 m/s for this controller family.**
- 1.5 m/s holds 15/15 at T_sw 0.25 s (V29 + H29). [measured: out/summary_ladder.json]
- 1.8 m/s fails 0/5 (V33) with the same runaway pattern as A: the capture point outruns the swing, the target hits the cap, then the fall. [measured: out/V33_v180_T025.json]
- A shorter swing with an explicit 0.03 s trailing-foot hold (V30) failed 0/5 at 1.5 m/s. [measured: out/V30_v150_T025_ds.json]

**D. Plant-side resources as speed rises (real plant), and what is not limiting.**

Swing-leg ankle servo span. The ankle is the first leg servo near its authority:

| speed (m/s) | ankle deliverable p95 | ankle demand beyond range | ankle raw action clipped | source |
|---|---|---|---|---|
| 1.2 | 0.58-0.64 | 1.6-2.3 % | 9-11 % | out/an_show_budget.txt |
| 1.35 | 0.67-0.80 | 2.7-3.9 % | 12-16 % | out/an_budget.txt |
| 1.5 | 0.83-0.90 | 6-9 % | 21-25 % | out/an_budget.txt |

- At 1.5 m/s this is a swing-leg effect: ankle demand above +1 on 11.6% of swing steps vs 1.3% of stance steps; swing knee beyond its range on 6.5% (5.2% below -1, 1.3% above +1). [measured: out/an_sat_split_V29.txt]
- Swing joint speeds rise with speed (p95 ankle 10.0 / 11.0 / 11.5 rad/s, knee 10.6 / 12.2 / 17.3 rad/s at 1.2 / 1.35 / 1.5 m/s). The 45 N.m.s/rad passive damping alone, fed forward by the controller, then takes 0.55 / 0.59 / 0.65 of the ankle's ctrl span at p95 (up to 0.91). [measured: out/an_swing_speed.txt]
- [inferred: kp x ctrl span against c x qdot] Cadence is bounded on the plant side by servo span against passive damping and inertia.
- A HYPOTHETICAL leg with half the joint damping (P2) did not rescue 1.8 m/s: 0/5, worse. So damping is not the single binding limit for this controller. [measured: out/P2_HYPOTHETICAL_halfdamp_V33_v180.json]

Not limiting at or below 1.5 m/s on surviving episodes:
- **Hip roll.** Demand beyond range at most 0.2% of steps at 1.2-1.35 m/s and at most 3.2% (one seed) at 1.5 m/s; deliverable p95 0.50-0.74. [measured: out/an_show_budget.txt, out/an_budget.txt]
  - An earlier 20-33% swing-hip-roll clipping came from my own swing derivative/inertia terms. It went away when those were restricted to the sagittal joints (V14, V15). [measured: out/an_hiproll_split.txt, out/V15_v070_ground.json]
- **Pad roll.** Stance-pad roll p5/p95 within -5.5/+7.2 deg up to 1.5 m/s. [measured: out/summary_ladder.json]
- **Torque.** Force/forcerange peak at most 0.80, p95 at most 0.52 at 1.5 m/s. [measured: out/an_budget.txt]
- **Joint range.** Minimum margins at 1.5 m/s: knee 8.5 deg, ankle 13.1 deg, hip roll 11.5 deg, hip pitch 24.8 deg from their stops. [measured: out/an_budget.txt]
  - In the 1.35 m/s runaway the stance ankle reached 122-125 deg (7-10 deg from its 132.4 deg stop) before the fall. [measured: out/an_limits_V21.txt]

## Running gait?

No. Across all real-plant variants the probe never classes an episode "run" (flight >= 0.05 with double support < 0.05). [measured: out/summary_ladder.json] Airborne control steps occur only in two situations:
- as isolated landing bounces in otherwise walking episodes at 1.35 m/s and above (0.1-1.4%)
- in the uncontrolled acceleration before a fall. In the V20 and V21 runaway traces, airborne steps (zero floor force on both feet for the whole step) appear only at instantaneous CoM speeds of 1.7 m/s and above. In the V29 and V33 traces (probe rule), the landing bounces fall at instantaneous CoM speeds of 1.0-1.5 m/s. [measured: out/an_flight_vs_speed.txt]

By construction the controller cannot plan a flight phase: it switches stance at touchdown. [inferred: wlib2.py]

## What changed from FINAL3, and the evidence for each

The controller is `scripts/wlib2.py` (class Walker2, a subclass of the unchanged EXISTENCE `wlib.Walker`; every option defaults to off). Final parameters are in `out/V35_v120_T030.json` (and V18, V24, V27, V29 for the other rungs). In order of discovery:

1. **Feed-forward of the joints' passive damping** (stance 0.6 x measured joint speed; swing 1.0 x desired joint speed). [measured: out/an_xi_check_base.txt, out/V04_ramp_T032_damp.json]
   - Mechanism found: the LIP placement was accurate at touchdown (capture point to new pad 0.045-0.056 m against the rule's 0.055 m). The CoM still gained much less speed per stance than the LIP predicts, and the shortfall grew with speed, consistent with the uncompensated 45 N.m.s/rad leg damping.
   - With the damping feed-forward, actual/commanded speed rose from about 0.5 to 0.8-1.0.
2. **Shorter swing** (T_sw 0.45 -> 0.32 s, later 0.30 / 0.28 / 0.25 s). The original failure (EXISTENCE fail_analysis): with a 0.45 s swing, the predicted capture point at touchdown runs past the step cap. [measured: out/an_V02_steps.txt, out/an_V03_steps.txt]
3. **IK restart from the home leg posture, and the pad-pitch IK weight ramped in over the first 40% of swing.** At lift-off from far behind, the damped-least-squares IK folded the swing leg backward (hip +62 deg at its stop, knee -74 deg, IK error 0.5-0.7 m) and never swung it forward. [measured: out/an_V06_window.txt]
4. **Hip-relative reach clamp (0.52 L) and a slight crouch (CoM target 0.93 L instead of 0.985 L).** At about 0.7 m/s and above, the swing foot was sent toward the forward reach limit and hovered there (forced switches at sph 1.5). [measured: out/an_V09_steps.txt]
   - V10 (crouch) held 0.7 m/s 5/5. [measured: out/V10_v070_crouch.json]
5. **Speed command in m/s with a start ramp, a step-to-step speed integral (only after the ramp; reset on overspeed), a proportional term against the true command (0.2 s), and the model step time matched to the measured touchdown (0.80 T_sw).** The integral wound up during ramps in V02/V03/V09 and the step-time model was biased (V17 vs V18). [measured: out/an_V02_steps.txt, out/an_V09_steps.txt, out/summary_ladder.json]
6. **Swing inertial feed-forward and derivative feedback on the sagittal joints only.** Swing overshoot of about 0.2 L and late touchdown at 0.9-1.0 m/s [measured: out/an_V07_window.txt]. Applied to hip roll as well, they caused 30% swing hip-roll clipping [measured: out/an_hiproll_split.txt].
7. **Landing height from the stance pad's lowest corner** (not its centre). On the long and forced steps the swing pad tracked its target to 0-1.9 cm below the stance pad's centre without loading [measured: out/an_forced_check_V14.txt]. [inferred: from that geometry] The stance pad pitches at speed, so its centre is above the floor.
   - With this fix the step time became regular: 0.270 +- 0.003 s, zero forced switches, against 0.38-0.42 s with 1-4 forced switches per seed before (V14). [measured: out/an_steps_stats.txt]
8. **Touchdown window from sph 0.75 and clearance 0.08 L** (V25 onward). The held-out 1.2 m/s fall (seed 11) started at an early toe-scuff touchdown (sph 0.72): the stance switched to a still-folded leg (stance knee -40 to -47 deg), giving short steps and a runaway. [measured: out/an_limits_H19_s11.txt]

Tried and dropped (each worse, evidence in its JSON):
- swing-target freeze late in swing (V12)
- a DCM end-of-step CoP regulator (V13; the CoP sat on its +-0.04 m clip) [measured: out/an_V13_cs.txt]
- deeper crouch with more reach (V22)
- trailing-foot hold (V30)
- higher stance-switch threshold (V32)
- more vertical damping (V36)

## Limits

- One controller family, hand-tuned within 37 counted variants, not optimised. The ladder is a lower bound on what the plant can walk. The plant's own walking-speed ceiling is unmeasured: above 1.5 m/s every failure I traced began in the controller.
- Seeds 0-4 for every variant, plus held-out seeds 5-14 for the 1.0, 1.2, 1.35 and 1.5 m/s parameter sets, at the locomotion config's reset noise. Episodes are 10 s, not the 40-episode publication panel.
- Success at a speed means the steady 5-10 s displacement speed at or above the rung, with one exception: V23 at the 0.5 command reached only 0.479-0.490 m/s. The 0.5 rung is passed by V05, an earlier variant of the same controller (0.526-0.532 m/s over 3-10 s).
- "No flight" uses the probe's per-control-step rule. Isolated physics substeps with both feet off the floor exist even where the probe counts none: 0-1 per episode at 0.7 m/s, 1-7 at 1.0 m/s, 8-18 at 1.2 m/s. [measured: out/summary_ladder.json] A stricter substep-level reading would put the strict ceiling at 0.7 m/s.
- The controller reads the env's MjData read-only (as in EXISTENCE) and does its kinematics on private MjData copies. Equivalence to the policy observation is by construction (differences and velocities only; k_yp = 0), not tested numerically. The controller keeps internal state (phase timer, stance side, speed integral); the observation has no clock.
- Heading is uncontrolled: yaw drift up to +-22 deg per 10 s at 1.0 m/s and above. Foot slip in contact rises to 0.15-0.25 m/s at 1.0-1.5 m/s. [measured: out/summary_ladder.json]
- HYPOTHETICAL probes P1 and P2 edited the env's compiled model in memory in their own process only (foot geom solref/solmix; leg dof_damping). They are labelled HYPOTHETICAL in their tags, JSONs and printed lines and were never written anywhere.
- Both behaviours (stance and walking): this stage changes no repository code, config, reward or physics, so the stance behaviour is not affected. The controller is an analysis artefact.
- [inferred: from the gait numbers, for the walking pilot] A walk at the gate speed exists on the real plant, but it is not the slow FINAL3 gait:
  - about 2x FINAL3's cadence (stride 2.0-2.2 Hz, 0.24-0.25 s steps)
  - a 5-6 cm crouch (CoM target 0.93 L)
  - swing joint speeds near 10 rad/s, with the swing ankle near its servo span
  - speed held by step-to-step placement
  A policy without a clock would have to produce this rhythm from state alone.

## Slips

1. Commands before the logging wrapper existed (00:42-00:45 UTC) and a number of later read-only commands were not logged literally. They are listed after the fact in a CMDLOG NOTE:
   - date, mkdir, ls, df
   - reads of FEAS_RESULTS.txt, the existence scripts and FACTS.md
   - system-python JSON prints
   - reads of the crashed V08 stdout
   - greps of my own scripts
   - the heredoc creation of 16 analysis scripts
2. The system python3, not the specified venv python and without -I -B, was used for:
   - string-replace patches of my own scripts (wlib2.py, run2.py, air_events.py, flight_vs_speed.py, forced_check.py via sed)
   - three read-only JSON prints

   Those patches are summarised in CMDLOG NOTE lines, not quoted literally. No simulation ran outside `scripts/sim.sh` / `scripts/x.sh`.
3. V08's first attempt crashed in its first episode: wrong `mj_fullM` argument order for mujoco 3.10, my bug. Its stdout was overwritten by the corrected re-run under the same tag, and it counts as one variant.
4. Analysis-script bugs, caught before any conclusion was drawn from them:
   - `forced_check.py`'s first output read the swing phase from the stance-switch row, which carries the previous swing's phase. Fixed; the first table was discarded.
   - `flight_vs_speed.py` first used a stricter airborne rule (zero force on the whole step) than the probe's, so it reported no airborne steps where the probe counted some. Fixed to the probe's rule.
   - The speed-binned double-support column it prints is confounded with gait phase and was not used.
5. The audit hook was chained after the probe's floor-truth hook by replacing `env.unwrapped._substep_probe_hook` at the first controller call of each episode. The probe's hook still runs first and unchanged, and the probe restores its previous hook at episode end. This is an in-memory change to the probe's runtime object, not to its code or to physics.
6. The 5-10 s probe window was added after V17. V01-V17 report steady speeds over 3-10 s (V05 in the table).
7. The ramp diagnostics V02-V04 ran on 2 seeds only.
8. One CMDLOG line (the RESULT.md corrections) abbreviated the interpreter path; the NOTE line right after it gives the literal path.
9. All analysis outputs were first printed to the terminal. They were regenerated into files (`out/an_*.txt`, by `scripts/analysis_all.sh`, no simulation) so that every [measured] label points to a file.

## Files

- `CMDLOG.txt`: literal commands with UTC times, plus NOTE lines.
- `scripts/wlib2.py` (controller), `scripts/run2.py` (runner: frozen probe + audit + optional HYPOTHETICAL edits), `scripts/sim.sh`, `scripts/x.sh`; copies of the EXISTENCE `wlib.py`, `defaults.py`, `run.py`, `probe_eval.py` (unchanged).
- `out/<tag>.json` and `out/<tag>_stdout.txt` for every run; `out/<tag>_s<seed>_trace.npz` for saved traces.
- `out/summary_ladder.json` (all ladder rows and per-seed criteria), `out/variant_ledger.json` (37 counted variants in run order, parameter differences from FINAL3, outcomes; plus the non-counted re-runs).
- `out/an_*.txt` (every trace/JSON analysis quoted above), `out/contact_params_stdout.txt` (foot and floor contact parameters, read-only).
