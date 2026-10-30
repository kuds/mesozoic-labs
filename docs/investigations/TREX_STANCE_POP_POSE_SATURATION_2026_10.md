# Diagnostic Report — T-Rex Stage 1 (`stance`): reset "pop", displaced pose (#489), saturation & chatter (#491)

**Date:** 2026-10-30

**Status:** dated record, report only. It was written at the maintainer's request as a root-cause investigation of #489 and #491. It follows the six tracks of the maintainer's investigation brief of 2026-10-30 ("the brief"), and §7 scores the brief's hypotheses one by one. The brief asked for no code patches and no reward or architecture solutions, so this note proposes no fix. Nothing in it trained or loaded a policy. It changes no code, config, reward, plant or gate.

- **What comes with it.** The scripts and outputs behind every [M] number are in [stance_2026_10/pop_pose_saturation/](stance_2026_10/pop_pose_saturation/README.md). Their provenance is in [SOURCES.txt](stance_2026_10/SOURCES.txt) and their sums in [SHA256SUMS](stance_2026_10/SHA256SUMS) (see the Appendix).
- **Related records.** [TREX_TAIL_SATURATION_HABIT_2026_10.md](TREX_TAIL_SATURATION_HABIT_2026_10.md) (the "habit doc"), against which this note was reconciled row by row, and [TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md).

**Scope.** Root-cause investigation only. No fixes are proposed. Code is at `358651f` (main, after the #654 merge). The investigation made no repository changes: this note and its evidence folder are its only additions.

**Method.** I read the code, config, MJCF, issues and the repo's own investigation docs. I also ran scripted and open-loop rollouts of the real `TyrannosaurusRexAnatomicalEnv`, built from `configs/tyrannosaurus_rex/anatomical/stance.toml` (MuJoCo 3.10, isolated venv). No trained checkpoints are available locally (they live on Drive only) and torch is not installed. Trained-policy numbers therefore come from the repo's committed telemetry, with citations. The main sources are the tail-saturation habit investigation (`docs/investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md`, cited as "habit doc" with its ledger row IDs such as L145), the r9 satdiag report (`docs/investigations/stance_2026_10/stance4/satdiag/report.md`) and the r10 CoM revisit (`docs/investigations/stance_2026_10/com_revisit/REPORT.txt`, "COMREV"). Scripts and outputs are in `stance_2026_10/pop_pose_saturation/` (see the Appendix).

**Plant revisions matter throughout.** #489/#491 were measured on physics **r6**, the October satdiag screens on **r9**, and today's checkout is **r10** (`configs/plant_versions.toml:607-609`). Every [M] number in this report is r10. r6 and r9 facts are [D] and labelled as such.

**Evidence labels:**
- **[M]** measured for this note on r10 (scripts and outputs in `stance_2026_10/pop_pose_saturation/`).
- **[D]** documented in a repo issue or investigation doc (cited).
- **[C]** code or config fact (cited as `path:line`).
- **[I]** inference, not directly measured.

---

## 0. Bottom line

1. **The brief's numbers come from a plant that no longer exists.**
   - #489 (3004.3 vs 3274.4) and #491 (12 of 21 actuators on the rail) were measured on run `20260805_011234`, on physics r6. That plant had 21 actuators, including 5 toe actuators that were railed. It had no command filter and no saturation term.
   - Today's checkout is physics r10 / policy interface r15. It has **15 actuators**, a 10 Hz command filter, a saturation ramp, and a tail sprung to rest level. The statue there scores **≈3,766–3,796**.
   - Since then the habit has shrunk to **one actuator per run** [D]:
     - On r9, 2 of 5 satdiag runs railed `tail_3_pitch` at +1 (4 of 10 runs in the wider family alarmed).
     - In r10's first round, s42 railed tail_3, s60 parked it at +0.64, and s80 alarmed on `neck_pitch` at −1.
2. **The pop is real, but it neither costs much nor triggers the rails.**
   - `qpos0` is not a static equilibrium. The neck and head have no gravity preload, the ankles carry an intentional 5.5° preload, and the root is authored 0.5 mm into the floor.
   - Under `action = 0` the body settles in about 155–190 steps (the neck and head by about step 130). Excluding `stance_width`, which pays 0 before step 200 by design, this costs about 3 points without reset noise and about 18 (12–26) with native reset noise.
   - The trained tail rail is acquired by a slow climb that starts **after** the pop [D]. On r10 the command is about 0 during the pop and reaches 0.9 at about step 250. On r9 the median first |a| ≥ 0.9 is at step 178. Onset on r6 was never measured.
3. **The 270-point gap is a joint-space gap, not a pelvis gap.** Shares of the r6 gap (#489's term table, regrouped):

   | Source | Share |
   |---|---|
   | Rate penalties (smoothness + jerk) | 39% |
   | Joint home-pose terms | 34% |
   | L2 energy | 20% |
   | Pelvis-rate terms (drift + speed + spin) | 9% |

   Pelvis attitude, height and neck/head terms were a **net gain** for the policy (+26.2).
4. **Tail and head rails are not counterweights.**
   - Open-loop on r10, a tail rail moves the CoM ≤ 2.1 mm and pelvis tilt ≤ 0.23°. It is pure cost (−97 to −102 per episode each). It is PD-target windup at mid-range, never at a mechanical stop.
   - The repo's closed-loop tests agree [D]. Clamping the tail_3 rail to 0 in 12 trained r10 policies leaves the stance unchanged and gains +63.6 per episode. On r6, four tail joints held at +1 stood the full horizon at 3254.0 against the statue's 3274.4.
   - The **neck** is the only effective counterweight: −1 moves the CoM 25–32 mm back.
   - Head and tail *do* balance passively: about the hips the tail offsets 78% of the head + neck moment, and the CoM sits 153 mm ahead of the hips, over the feet. But a level tail is already within 0.2 mm of its maximum backward reach, so tail pitch can only move the CoM *forward* (§2.3a).
   - No leg actuator can be held on a rail: 0/6 survive for every leg actuator, both signs.
5. **There is no "unwinding valley" for pure rails, but a real trap for counterbalanced pairs.**
   - Releasing a tail or neck rail is immediately profitable (+5 to +48, no transient dip).
   - Releasing only the counterweight half of a lean + neck pair is catastrophic (−1,450 to −1,720). The nosedive arrives 170–350 steps later, which is beyond the γ = 0.98 horizon.
   - Only coordinated or slow unwinds pay.
   - **What holds the r9/r10 tail rail is the policy itself** [D]: a learned positive-feedback loop through the tail's own angle observation. Its gain is 0.63 in 12 of 12 railed r10 ends, and its fixed point lies past +1. How training forms it is unexplained.
   - Flat reward regions are secondary. `tail_home_pose` is a narrow Gaussian. The action-clip dead zone of the un-squashed Gaussian policy only halves the slope at the observed pre-clip means (+1.01 to +1.12, σ ≈ 0.14, about 79% of samples clipped), and bounding the mean on r9 did not remove the rail.
6. **The reward is not rate-blind.**
   - Smoothness (Δa²) and jerk (Δ²a²) penalties existed at #489 time (−65.0 and −41.4 there) and still exist. Since policy interface r11 (during physics r7), a 10 Hz filter sits in front of the plant (`configs/plant_versions.toml:370-388`).
   - The measured chatter (22.6 Hz effective, r6) is a **4–5-step** limit cycle at 100 Hz control, not 1-step Nyquist bang-bang.
   - What is genuinely unpriced or unobserved:
     - joint velocities (`qvel[6:]`);
     - chatter that stays beyond the clip;
     - the action and filter history (the rate penalties are non-Markov with respect to the observation).
7. **Exploration noise forces active feedback.**
   - The statue cannot survive PPO's own initial exploration noise: σ = 0.22 (`log_std_init = −1.5`) on the legs gives **0/10** survivors, and σ ≥ 0.11 gives ≤ 1/10.
   - So a policy must learn leg feedback before it can find the passive basin. This is the most plausible common origin of both the displaced pose and the chatter [I].
8. **Under the committed config, the statue is not the reward optimum** (r10, 3 seeds) [M].
   - `leg_home_pose_reference` defaults to `"keyframe"`, and `stance.toml` does not override it. The keyframe leg pose is not an equilibrium.
   - A small static lean back toward it beats the statue by **+17 to +24.5** points.
   - The static return surface peaks just short of a stability cliff: lean 0.25–0.5 gains, while lean 0.75 survives only 1 of 3 seeds. With the `"settled"` reference, the same lean *loses* 12 to 47 points.
9. **VecNormalize, critic and transfer:**
   - **VecNormalize.** In a deterministic statue rollout, the first 50 steps (5% of samples) carry 69–92% of all velocity and IMU variance. Under training-level noise (σ ≥ 0.14), the pop's share drops to 13–25% and the pop is no longer separable (AUC 0.48–0.65). Contamination is therefore a late-training effect.
   - **Critic.** V(s) and GAE across regimes were **not measurable**. Two critic signals are logged: SB3's pooled per-update explained variance (`environments/shared/diagnostics.py:81-83,416-436`), and the policy probe's V(s) at ten fixed statue and lean states from steps 199–599 (`environments/shared/policy_probe.py:10-18,73-87`). Neither covers the pop or the policy's own episodes by step, and no checkpoints are available here.
   - **Transfer.** Stage transfer carries the whole SB3 model (actor, critic, `log_std`, Adam state) plus `obs_rms`. `ret_rms` is reset, and the observation width must match exactly.

---

## Premise corrections

| Brief premise | What the code and evidence say | Status |
|---|---|---|
| 21 actuators, ~10–11 pinned at ±1 | There are **15 actuators** today: toes are passive (since r7), and arms plus tail_4/5 are welded. The observation carries 21 hinge angles. #491's 12 rails were on the August 2026 21-actuator plant (physics r6), and 5 of them were toe actuators that no longer exist. On r9, `tail_3_pitch` +1 railed in s42 and stance3 Run 2, while s52, s60 and Run 3 had no actuator at \|a\| ≥ 0.9 [D `stance_2026_10/stance4/satdiag/report.md:8-60,98-120`]. In r10's first round, s42 railed tail_3, s60 parked it at +0.642, and s80 alarmed on `neck_pitch` at −1 [D `TREX_TAIL_SATURATION_HABIT_2026_10.md:621,625`] | **Outdated** |
| `k = 40` (legs, toes) and `k = 80` (tail) are actuator stiffnesses | These are **passive joint stiffnesses**. The leg springs rest at the stance pose (MJCF springref, e.g. lines 258, 267, 274). Since r10 the tail pitch springs are preloaded (springref 52.5/25.9/10.4°, lines 199, 205, 210) so the tail rests level at a = 0; through r9 they rested at 0, and gravity drooped the tail. Actuator kp: hip_pitch 1200, knee 1500, ankle 900, hip_roll 600, neck_pitch 200, neck_yaw 100, head_pitch 80, tail 100/80/80/60. The tail spring (80) is **stiffer than the tail_3 servo (60)**, so a full command moves that joint only about ±0.09 rad: 0.086 on r9 [D satdiag:110-114] and 0.091 on r10 [M]. That sets the joint's range per unit command. On r9, what put a level tail out of reach within ±1 was gravity, not the spring (§2.4) | **Misattributed** |
| Weights alive 1.75, posture 1.5, nosedive 1.5, height 1.0 | Current: `alive_bonus = 1.0` (half support-conditioned), posture 1.5, nosedive 1.5, **height 0.6**, drift 0.3, speed 0.3 (only above 0.10 m/s), spin 0.1 [C `stance.toml:8-12,67,317,321,370`]. The #489 table implies alive ≈ 1.0 and height 0.6 on r6 as well | **Outdated** |
| No Δa penalty; reward is blind to chatter | Smoothness 2.0 (Δa²) and jerk 3.0 (Δ²a²) are priced on the raw clipped command [C `reward_functions.py:115,136`; `stance.toml:32,242,337`]. A saturation ramp of 0.5 applies above \|a\| = 0.9 [C `reward_functions.py:522-546`; `stance.toml:283,290`]. A 10 Hz first-order filter is active [C `tyrannosaurus_rex/base_env.py:144`; `shared/base_env.py:248,1226-1242`]. At #489 time smoothness and jerk already cost −65.00 and −41.41 | **Refuted** |
| Chatter is near-Nyquist, 1-step bang-bang | 22.6 Hz effective (a DC-blind estimator, #489) at 100 Hz control is a **4.4-step period**; Nyquist is 50 Hz | **Refuted** |
| Trained ≈ 3,008 vs statue ≈ 3,278 | r6: 3004.3 vs 3274.4. r10 statue: 3796.4 noise-free and **3765.6 ± 15.7** over 100 reset-noise seeds [M]. The railed r9 s42 scores 3559.5–3562.7 in deterministic play [D satdiag:77-78], about 195 below r9's statue (3755.7 over 40 reset-noise seeds; 3788.93 noise-free) [D `configs/plant_versions.toml:454,468`]. Different seed panels, so only approximate | **r6 only** |
| `qpos0` may be a static equilibrium | It is not (Track 1) | **Refuted** |
| No phase or time signal reaches the actor or critic | Confirmed. The 63-dim observation has no step count, previous action or filter state [C `tyrannosaurus_rex/base_env.py:853`] | **Confirmed** |

---

## Track 1: Initial "pop" vs true static equilibrium

### 1.1 Physical source of the pop [C, M]

The keyframe (`anatomical/assets/tyrannosaurus_rex_anatomical.xml:582-621`, `home` key at lines 613-621) is a hand-authored pose, not a solved equilibrium. `action = 0` maps to the keyframe ctrl (`tyrannosaurus_rex/base_env.py:819-833`, `shared/base_env.py:405`), so the statue means "hold keyframe ctrl".

| Element | Keyframe qpos vs ctrl | Consequence at t = 0 |
|---|---|---|
| neck_pitch, neck_yaw, head_pitch | ctrl = qpos = 0, **no gravity preload** | The PD holds zero torque against the head's weight, so the head sags |
| hip_pitch | ctrl = qpos (−2.6°), no preload | The hips sag under trunk weight |
| ankle | qpos 76.9° vs ctrl 82.4°, an **intentional 5.5° preload** (asserted in `test_stance_gate_report.py:2551-2552`) | The ankles push |
| root z | 0.9266 m, **0.5 mm into the floor**; the settled 0.9261 m is documented in the same XML comment and used as constants in `tyrannosaurus_rex/base_env.py:140-195` | Contact pushes the root up |

Initial generalised accelerations at reset (noise-free, `action = 0`) [M]:

| Joint | Acceleration (rad/s²) |
|---|---|
| hips | −45.7 |
| ankles | +52.3 |
| knees | +27.4 |
| neck | +16.2 |
| head | −17.1 |
| tail_1 | −22.5 |

### 1.2 Settling timeline under `action = 0` [M, `t1_pop.py`]

| Metric | Value |
|---|---|
| Kinetic-energy peak | 4.37 J at **step 15** |
| KE < 10% of peak | step 67 |
| KE < 1% of peak | step 155 |
| Root speed > 2 cm/s until | step 167 |
| max \|q̇\| > 0.02 rad/s until | step 192 |
| Steady KE | 0.0026 J |
| Peak foot force | 1.10 body weight |
| Neck / head sag | +12.26° / +12.63°; overshoot ≈ step 50, settled ≈ step 130; the actuators then carry 27% of their forcerange |
| Hip pitch | −2.6° → −5.2° |
| Ankle | 76.9° → 78.85° (+1.95°) |
| Knee / toes / tail | −0.25° / ≈ −0.2° / ≈ 0° |
| Pelvis z, tilt, forward_z | 0.92614 m, 0.94°, −0.0164 (the documented constants are 0.9261 and −0.017) |

With native reset noise (scale 0.05), steps 0–10 reach a KE peak of 9.8 J and a peak foot force of 1.58 BW. Out of 20 seeds, one nosedived (seed 4 at step 201). Seed 4 also fails under almost every open-loop pin in Track 2, so it is a reset draw near the basin edge. Over 100 seeds (1000–1099), **100/100 survive at 3765.6 ± 15.7**.

Settled poses across reset-noise seeds differ by only **0.25° rms**. There is one passive basin, and `q_statue` is well defined.

### 1.3 Reward during the pop: steps 0–50 vs 50–200 vs 200–1000 [M, statue, 5 surviving seeds]

Per-step reward by window:

| Window | Total per step | of which `stance_width` |
|---|---|---|
| 0–50 | 3.489 | 0 |
| 50–200 | 3.692 | 0 |
| 200–1000 | 3.816 | 0.099 |

- `stance_width` pays 0 before step 200 by design (`stance_width_settle_steps = 200`, `stance.toml:188`). This is a structural −20 points for every policy.
- Excluding `stance_width`, the pop costs **3.0 points noise-free** and **18.0 on average with native reset noise** (11.6–26.1 over seeds 0–3; 16.4 over 20 qpos-noise-only seeds, `t1_pop.py`). About 14 of the 18 fall in steps 0–50 and about 4 in steps 50–200. **That is 1–10% of a 270-point gap.**
- The loss runs through support and contact terms, not pelvis kinematics. Per-step deficit vs late:
  - bilateral_support −0.038
  - alive −0.031 (the support-conditioned gate)
  - head_clearance −0.034
  - foot_load_balance −0.030
  - foot_flatness −0.029
  - tail_home_pose −0.019
  - neck_posture −0.016
  - speed −0.007
  - leg_home_pose −0.006

### 1.4 Policy behaviour in steps 0–50 vs 50–1000, and when saturation starts

All trained-policy data here are [D], and the plant revision matters. Today's checkout is physics r10 (`configs/plant_versions.toml:607-609`). The step-resolved onset tables were measured on r9, and #491 on r6. Control runs at 100 Hz, so 0.25 s is 25 steps.

**Physics r10 (today's plant) [D: `docs/investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md:670` (L144), `:671` (L145)].** These are the 12 railed s42 entropy-screen continuation ends, 240 episodes per condition, in 0.25 s bins:
- **Own tail_3 command:**

  | Steps | Command |
  |---|---|
  | 0–25 | −0.01 |
  | 25–50 | +0.21 |
  | 50–75 | +0.41 |
  | 75–100 | +0.53 |
  | 250 (2.5 s) | **+0.90** |
  | later | +0.94 |

  Over the same period the tail_3 angle goes from −0.08° to +4.84°.
- **With the tail_3 joint clamped at 0,** the command climbs to +0.23, then +0.37, then holds at **+0.35** from about 0.5 s on.
- **Clamped at 0, at 0.5, and free,** the tail sits at 0, 2.6 and 4.9–5.2°, and the own command at 0.29–0.33, 0.66–0.68 and 0.94–1.0. The relation is linear.
- **[I] What this means.** On r10 the command is about 0 at the pop's KE peak (step 15, §1.1), so there is no slam during the pop.
  - A posture-driven offset of about +0.3 to +0.4 is in place by about step 50.
  - The loop through the tail's own angle then amplifies it to the bound over about 2.5 s, overlapping the late settling (the body is quiet by steps 155–190).
  - The habit doc traces that offset to the s42 lineage reading its nose-down pelvis pitch and leg angles (L162, `docs/investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md:682`). That is steady posture, not pop velocity.

**Physics r9 (stance4 s42; not today's plant) [D: `stance4/satdiag/report.md:21,29-39`; `satdiag/rollout/tables/onset_within_episode.txt`].** s42 at the 2.1M checkpoint (17 of 20 episodes flagged), seeds 3042–3061:
- The root settles by a median of **step 16**, so the policy damps the pop faster than the statue does.
- The tail_3 command sits at **+0.16 over steps 40–60**, then ramps:

  | Steps | Command |
  |---|---|
  | 60–80 | +0.29 |
  | 100–125 | +0.59 |
  | 200–225 | +0.92 |
  | 400+ | +0.97 |

- First crossings: **0.5 at median step 101.5, 0.9 at step 178, 0.99 at step 216** (range 155–822). The half-change step is 110 [IQR 93–121]. In the final checkpoint the command starts higher (+0.31) and crosses 0.5 at a median of step 56.
- r9 stance3 Run 2 (seed 60) behaves the same way: its first 0.99 step comes at a median of step 212 (range 23–345) [D `docs/investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md:554` (L027)].
- Onset-aligned means (n = 11) show the tail_3 joint rising from −0.094 to −0.020 rad as the command goes from +0.16 to +0.89. Meanwhile:
  - the tail_1 command moves **the other way** (+0.09 → −0.15);
  - the CoM ahead of MTP III stays flat at 0.067–0.068 m;
  - pelvis pitch drifts from −1.12° to −1.27°.
- During the reset transient (steps 0–36), only tail_1_pitch (s52/s60 finals) and head_pitch (Run 2) touch the rail, and only briefly.
- Swapping **only the tail_3 joint-angle observation** moves the command mean from +0.998 to +0.290, and from +0.244 to +0.943. Zeroing tail_3 velocity does nothing. **The rail is reached by observation-to-command positive feedback on the tail's own angle** [D, inferred and probe-supported]. The same loop was later measured directly on r10; see §4.6.

**Physics r6 (#491).** There is no step-resolved onset telemetry for `20260805_011234`, only time-mean DC/AC per actuator. Whether those 12 rails were acquired during the pop is **unverified**.

### 1.5 `qpos0` vs the settled statue (`q_statue`) [M]

| Joint | Change from `qpos0` to `q_statue` |
|---|---|
| neck_pitch / head_pitch | **+12.26° / +12.63°** |
| hip_pitch | −2.6° |
| ankle | +1.95° |
| knee | −0.25° |
| toes | ≈ −0.2° |
| tail | ≈ 0 |
| pelvis z | −0.46 mm |
| forward_z | −0.0164 |

Reward references [C]:
- `neck_posture` and `tail_home_pose` are centred on the **settled** pose (`stance.toml:271,305`), as is `stance_width` (`stance.toml:165`).
- **`leg_home_pose` is centred on the keyframe** by default (`tyrannosaurus_rex/base_env.py:195,279,1089`), which `stance.toml` does not override. Section 3.4 covers the consequence.

### Track 1 verdicts

| Hypothesis | Verdict |
|---|---|
| A pop or settling transient exists at reset | **Confirmed.** KE peaks at step 15 and the body is quiet by about steps 155–190 |
| `qpos0` is static equilibrium | **Refuted.** Neck/head sag 12°; hip and ankle preload/sag about 2° |
| Actuators slam to ±1 during the pop and then fail to unwind | **Refuted on r9 and r10.** The command is about 0 at the KE peak. The rail is reached by a monotone climb: 0.9 at median step 178 on r9, and about step 250 on r10. That climb is driven by the tail's own angle (§4.6). **Untested on r6**, the plant behind #491 |
| The pop explains a material share of the gap | **Refuted.** About 3 points noise-free and 18 (12–26) with reset noise, through support and contact terms |

---

## Track 2: Kinematic null space, counterweights, PD windup

### 2.1 Which actuators were railed

**Physics r6** (#491, `20260805_011234`, time-mean commanded DC; every rail had AC = 0.000, so these are static rails, not chatter):

| Group | Railed actuators |
|---|---|
| Toes (5) | r_toe_d2 +1, r_toe_d3 −1, r_toe_d4 +1, l_toe_d3 +1, l_toe_d4 −1 |
| Tail (4) | tail_1_pitch +1, tail_1_yaw +1, tail_2_pitch +1, tail_3_pitch +1 |
| Head (1) | head_pitch −1 |
| Hip roll (2, mirrored) | r_hip_roll +1, l_hip_roll −1 |

- No hip_pitch, knee or ankle was railed.
- #491 states that the 8 `leg_home_pose` joints carried only 1.2% of total DC power.

**Physics r9 (the October 2026 screens; not today's plant):**
- tail_3_pitch +1 in 2 of the 5 satdiag runs: stance4 s42 and stance3 Run 2 (seed 60) [D satdiag:9-21,116].
- Across all ten r9 L+T-family runs, the habit raised an alarm in 4 [D `TREX_TAIL_SATURATION_HABIT_2026_10.md:589` (L064)].
- Without any command rail, tail_1 sits on its lower **joint stop** in every r9 policy, and tail_2 is at or near its stop through 4.2M [D satdiag:46-47].
- The legs stay within about 0.014 rad of the statue, with foot loads 0.50/0.50 [D satdiag:53].
- Unflagged neck and head parking also occurred. Run 3 held `neck_pitch` at −0.700, and Run 2 held `head_pitch` at −0.52 beside its tail rail [D `TREX_TAIL_SATURATION_HABIT_2026_10.md:557` (L155)].

**Physics r10 (today's plant), first r10 round, 3 runs × 12M [D `TREX_TAIL_SATURATION_HABIT_2026_10.md:621,625` (L100, L106)]:**
- **s42:** `tail_3_pitch` at +1, with 22 ledger alarms from step 3,932,160. Gate FAIL 38/40 on actuator saturation.
- **s60:** `tail_3_pitch` **parked** at a window-mean command of +0.642, just under the alarm line. No alarm; gate PASS.
- **s80:** no tail rail, but 6 alarms on `neck_pitch_act` at **−1** (steps 5,505,024–7,077,888). Gate PASS.
- So on r9 and r10 the habit involves **one actuator per run**, not the 10–12 of #491 (r6).
- Two of three r10 runs alarmed, against 2 of 5 for the same arm on r9 (one-sided Fisher p ≈ 0.93). Whether r10 changed the rate is unknown.

### 2.2 Single actuator railed open-loop from the statue [M, `t2_pins.py` → `t2/pins.json`]

Setup: 6 seeds, with ΔReturn measured against the paired statue. Term deltas are taken over steps 200–1000. "Survive" counts all 6 seeds, but seed 4 also nosedives as a statue, so *5/6 = every seed on which the statue survives*.

| Rail | Survive | Joint angle | At stop? | Force / limit | ΔCoM x (mm) | ΔReturn | Main term deltas |
|---|---|---|---|---|---|---|---|
| tail_1_pitch +1 / −1 | 5/6, 5/6 | +6.84° / −6.43° (≈55% of the 12° commanded) | no | 0.11–0.12 | +2.1 / +0.4 | −99.3 / −101.8 | tail_home_pose −49.8, saturation −26.7, energy −4.0 |
| tail_1_yaw ±1 | 5/6 | ±4.1° (51% of 8°) | no | 0.08 | ≤ 0.8 | −100.6 / −102.0 | tail_home_pose −43.4, saturation −26.7, foot_load_balance −8.1 |
| tail_2_pitch ±1 | 5/6 | +6.13° / −5.90° | no | 0.13 | ≤ 1.4 | −99.5 / −99.7 | tail_home_pose −49.4, saturation −26.7 |
| tail_3_pitch ±1 | 5/6 | +5.23° / −5.11° | no | 0.14 | ≤ 0.6 | −97.5 / −97.3 | tail_home_pose −48.1, saturation −26.7 |
| neck_pitch +1 | 6/6 | 40.1° | **yes** | 0.00 | −3.3 | −446.0 | head_clearance −280, neck_posture −47 |
| neck_pitch −1 | **2/6** (falls at steps 250–402) | −20.6° | no | 0.20 | **−25.1** | −99.6 (survivors) | neck_posture −49.6, saturation −26.7 |
| neck_yaw ±1 | 5/6 | ±18.5° | no | 0.03 | −4 to −6 | −106.0 / −103.0 | neck_posture −30.6, foot_load_balance −21 |
| head_pitch +1 | 5/6 | 35.1° | **yes** | 0.00 | −9.1 | −242.2 | head_clearance −128, neck_posture −38 |
| head_pitch −1 | 5/6 | −11.3° | no | 0.29 | +3.6 | −88.6 | neck_posture −40.5, nosedive −2.8 |
| Each of 16 leg rails (hip_pitch, hip_roll, knee, ankle × L/R × ±1) | **0/6 each** | — | — | — | — | falls at steps 37–196 | — |

For the tail, every rail changes pelvis tilt by ≤ 0.23° and pelvis height by 0.0 mm.

### 2.3 Does any rail act as a counterweight? [M, `t2b.py`, 5 seeds]

- **LEAN52 / LEAN60** are leg-command DC offsets modelled on the forward-CoP drift of seeds s52/s60 (`stance_2026_10/diagnosis/findings_v3.txt:73-81`).
  - That file (2026-10-17) calls its plant "r10", but the name belongs to the foot plant of the parked draft #623, which never merged. Today's r10 is a different plant (`configs/plant_versions.toml:600-601`).
  - The leans are used here only as generic forward-lean patterns on today's plant.
  - LEAN52 = hip_pitch +0.104 and ankle −0.084 on both legs.
  - LEAN60 = hip_pitch +0.114 and ankle −0.081.

| Pattern | Survive | ΔReturn | Reading |
|---|---|---|---|
| LEAN52 alone | 0/5 (nosedive around step 190) | — | The lean alone is open-loop unstable |
| neck_pitch −1 alone | 2/5 | −100 | Shifts the CoM 25–32 mm back |
| LEAN52 + neck −1 | **4/5** | −130 | **The neck counterbalances the lean** |
| LEAN60 + neck −1 | **5/5** | −141 | Same |
| LEAN52 + tail_3 +1 | 0/5 | — | The tail does not |
| LEAN52 + head −1 | 0/5 | — | The head does not (head −1 moves the CoM +3.6 mm, *forward*) |

Closed-loop corroboration [D]:
- **r10 (today's plant).** The CoM revisit's tail clamp test covers the 12 railed s42 entropy-screen ends (`stance_2026_10/com_revisit/REPORT.txt:92-131`).
  - On the plant alone, the rail moves the CoM 0.2 mm forward and 1.6 mm up, and the pelvis 0.01°.
  - Removing the rail leaves lateral lean, roll, sway, creep and margins unchanged over 20 s, and return rises +63.6 per episode (12/12).
  - Its verdict: "It is not a counterweight."
  - tail_3 at its ±12° limits moves the whole-body CoM only 0.19–0.24 mm [`TREX_TAIL_SATURATION_HABIT_2026_10.md:657` (L130)]. L130's "151.17 kg body" is MuJoCo's world subtree, which includes the fixed 65.45 kg `prey` body. For the 85.72 kg robot alone the same keyframe check gives +0.33/+0.42 mm [M, §2.3a]. That is still negligible.
  - My open-loop pins agree: ≤ 0.6 mm for tail_3.
- **r9.** Through s42's within-episode tail ramp, the CoM ahead of MTP III stays at 0.067–0.068 m (`onset_within_episode.txt`, onset-aligned block).
- **r9, policy reaction.** satdiag §3 (`satdiag/report.md:55-58`) shows a forced tail_3 = 0.9–1.0 moving the CoM about 10–13 mm forward, and pitch from −1.1° to −2.6/−2.8°. Open-loop, a tail moves the CoM < 1 mm on r10. **That closed-loop shift is most likely the policy's coupled posture, not the tail's mass moment** [I; the habit doc reaches the same reading, `TREX_TAIL_SATURATION_HABIT_2026_10.md:762-765`, but the two numbers come from different plants].
- **r6.** All four tail joints held at +1, with everything else at home, stood the full horizon at 3254.0 against the statue's 3274.4: "The tail is a bystander" [`TREX_TAIL_SATURATION_HABIT_2026_10.md:523` (L005)].
- **[I] r10 s80.** The neck_pitch −1 alarm (§2.1) is the one rail direction that my pins show to be a real counterweight. Whether it counterbalances anything in s80 was not measured (§9, OQ 13).

### 2.3a Do the head and tail counterbalance each other? [M, `t8_headtail.py`, settled statue, noise-free]

**Statically, about the hips: mostly.** These are gravity moments about the hip-pitch axis; + means ahead of the hips (nose-down).

| Group | Mass (kg) | Share of 85.72 kg | CoM vs hip axis | Moment (N·m) |
|---|---|---|---|---|
| Neck + head | 13.70 | 16.0% | +921 mm | **+123.7** |
| Tail (5 segments) | 15.60 | 18.2% | −628 mm | **−96.1** |
| Trunk (pelvis geoms) | 35.00 | 40.8% | +233 mm | +80.1 |
| Legs (both) | 20.52 | 23.9% | +81 mm | +16.4 |
| Arms (both) | 0.90 | 1.0% | +531 mm | +4.6 |
| **Whole body** | 85.72 | | **+153 mm** | +128.7 |

- The tail offsets **78%** of the head + neck moment. The trunk's +80 N·m has no counterpart behind the hips.
- So the CoM sits 153 mm ahead of the hips, over the feet: 3.8 mm from the CoP, with about 61 mm of support behind it.
- An exact head–tail balance would move the CoM back only 32.8 mm. It would still sit about 120 mm ahead of the hips.
- The mass shares match the MJCF comments (head + neck "~16%", tail "~18%"; `tyrannosaurus_rex_anatomical.xml:78,81`). Its "skull and tail balance each other" (`:88-89`) holds only in this partial sense.
- They fall inside the ranges in the repo's literature notes: tail 10–21% and head + neck 7–18% (Hutchinson et al. 2011). Those notes also say the CoM stays ahead of the hip "in every reconstruction" [D `stance_2026_10/literature/research_notes/trex_foot_anatomy_posture.md:145-150`].

**Actively, in pitch: no.**
- A level tail is already within 0.2 mm of its maximum backward reach. Pitching it either way shortens its horizontal lever and moves the CoM **forward**, a second-order (cosine) effect.
- Swinging the whole tail rigidly gives these CoM shifts:

  | Swing | ΔCoM fore-aft |
  |---|---|
  | ±12° | +2.95 / +0.86 mm |
  | ±30° | +14.2 / +9.2 mm |
  | ±60° | +47.9 / +39.2 mm |

  The ±30° and ±60° cases are hypothetical, beyond the joint range.
- At a = ±1 the servos reach only about 5–7° per joint against the preloaded springs (§2.4).
- So **no tail command can move the CoM backward** by more than a fraction of a millimetre.
- The neck is the only adjustable fore-aft mass.
  - At the statue (+12.2°) its lever is also near horizontal (+0.08 mm/deg).
  - Its range reaches −30°, which lifts the head back over the body: −16.3 mm kinematically, and −25.6 mm measured once the body responds (pin, noise-free).

**Laterally, the tail is a first-order lever.**
- Yaw moves the tail mass sideways along its full length.
- tail_1_yaw at a = +1 settles at +4.08° (pin), which moves the CoM about −6.2 mm sideways. The full ±8° range gives ±12.1 mm.
- That is comparable to neck_yaw: ±10° gives ±8.4 mm and ±20° gives ±16.6 mm.
- Stage 1's symmetric stance does not need lateral CoM authority.

**Not measured:** dynamic use of the tail, i.e. reaction torque or angular-momentum exchange during fast head or leg motion.

### 2.4 Mechanical stop or PD-target windup?

- **At a stop:** only neck_pitch +1 (40.1°) and head_pitch +1 (35.1°). Both are expensive (−446 and −242) through `head_clearance`. Neither appears in any trained policy.
- **Windup (mid-range):**
  - The tail rails reach only 44–57% of the commanded angle, on 8–14% of forcerange.
  - neck_pitch −1, neck_yaw and head_pitch −1 use 3–29% of forcerange.
  - The joint is held off its target by its passive spring and gravity, not by a stop or a force limit.
- **r9 s42 tail_3 (stance4; not today's plant)** [D satdiag:19,111-114]:
  - The joint is at −0.007 rad, which is **0.216 rad below its +0.2094 stop**. It carries 0.26 of its 50 N·m limit, and since |kp·(ctrl − q)| ≤ 60 × 0.419 = 25 N·m it cannot force-clip.
  - A statue sweep gives q ≈ −0.093 + 0.086·a. The slope, kp/(kp + k) = 60/140 of the commanded angle, is set by the spring being stiffer than the servo.
  - The offset is gravity. At q3 = 0 gravity loads the joint with 13.48 N·m against the servo's 12.57 N·m at a = +1, and the spring, resting at 0, adds nothing there. So **on r9 a level tail needed a = +1.073, just past the bound**, and +1 held it at −0.007 [D `TREX_TAIL_SATURATION_HABIT_2026_10.md:698-720` (L061)].
  - Satdiag's own reading, "the spring is stiffer than the servo" (`satdiag/report.md:110-114`), was superseded on this point [`TREX_TAIL_SATURATION_HABIT_2026_10.md:566` (L040), `:1858`].
  - Holding tail_3 at 0 in the trained policy gains **+73 to +86 every episode, with no falls**: tail_home_pose +46.3, saturation +21.8, leg_home_pose +8.1, energy +3.9 [D satdiag:74-84]. The rail buys nothing.
- **r10 tail_3 (today's plant).**
  - Physics r10 preloads the tail pitch springs to springref 52.5°, 25.9° and 10.4° (MJCF lines 199, 205, 210; D-D44), so the statue's tail rests level at a = 0.
  - A command of +1 holds tail_3 at +5.19° with 7.1 N·m of servo torque against ±50 [D `TREX_TAIL_SATURATION_HABIT_2026_10.md:664` (L138)]. That is 43% of the 12° range, set by kp/(kp + k) = 60/140. My open-loop pin gives +5.23° (+0.091 rad) [M].
  - **So on r10 the rail is windup about 5° *above* level, not a requirement for holding the tail level.** The habit doc rules out "the plant forces the pin" as the cause on both plants (its §4.1, `TREX_TAIL_SATURATION_HABIT_2026_10.md:691-725`).
  - Holding tail_3 at 0 gains +78.29 per window at r10 s42's 3,670,016 checkpoint (its onset; 20/20 positive) [`TREX_TAIL_SATURATION_HABIT_2026_10.md:624` (L103)], and +63.6 per episode at the 12 railed entropy-screen ends (COMREV §4 (i)).
  - The exception is r10 s42's *final*. Its deterministic command was parked, not railed, and there the hold *loses* 64.46 (2/20 positive) [L103]. The habit doc lists this sign change as open (its §7 Q10).
- **History: stops were real on physics r7** [D]:
  - Run `20260809_155206` (r7, 15 actuators, `leg_home_pose_tolerance` 0.10) had 5 to 6 of 15 actuators parked at or within 2% of a stop by 10M. These were head_pitch, both knees, and the tail chain; the sources disagree on the exact count.
  - A constant hold of that pose fell within about 3 s, 9 of 10 times backward onto the tail, so the parked pose still needed feedback.
  - Sources: `TREX_TAIL_SATURATION_HABIT_2026_10.md:528,1856`; `stance.toml:267-270`.
  - The #504 shaping pack (August, r7) added the `tail_home_pose` and `action_saturation` terms. The r7 gate-pass run and its replicates then saturated nothing [`TREX_TAIL_SATURATION_HABIT_2026_10.md:222-224` (L013–L015)].
  - Through r9, the passive tail itself drooped onto tail_1 and tail_2's ventral stops (`stance.toml:271-275`). The r10 preload removed that droop.
  - For #491's run (`20260805_011234`) only commanded values are reported ("commanding them past their limits and `_scale_action`'s clip is holding them"). **Joint positions were never reported, so stop-vs-windup is unknown on that plant.**

### 2.5 Leg channels [M, `t2b.py`]

**Static tolerance from the statue:**
- ±0.02 is tolerated on every leg actuator.
- Knee and ankle at ±0.05–0.1 mostly fall.
- Mirrored hip roll at ±0.1 costs only −4, a near-null direction.
- hip_pitch +0.02/+0.05 scores **better** than the statue (+15/+16, see 3.4).

**Static leg "saturation" is physically impossible on this plant.** Any leg value seen near ±1 must be chatter or a time-average.

**r6's mirrored hip-roll rails, mapped onto r10:**
- P491 = head −1, hip rolls ±1 and four tail +1. It survives **0/6** open-loop, falling at steps 124–781 (median about 563).
- Without the rolls (P491_noroll) it survives 5/6 at −493: tail_home_pose −193, saturation −133, neck_posture −41, energy −20.
- So the roll rails need active stabilisation.

### 2.6 Where the "null space" actually is

- The statue already satisfies the pelvis terms. **No pelvis term requires internal counter-torque** (Track 3: pelvis attitude terms were a net gain for the displaced policy).
- Mechanically free directions (moving the CoM < 2–6 mm, not toppling): the four tail channels, neck_yaw, and small mirrored hip roll.
- **r6:** the toes and tail had no pose term (#491), so 9 of the 12 rails sat in directions priced only by energy. That is 0.075/21 ≈ 0.0036 per step, or about 3.6 points per episode per rail [C/D; toe mechanics on r6 not measured, I].
- **Today (r10):** the toes are passive (since r7, 15 actuators). Each free direction now costs about 100 points per episode (tail_home_pose + saturation + energy) [M]. The r9 and r10 habits involve one actuator per run (§2.1).

### Track 2 verdicts

| Hypothesis | Verdict |
|---|---|
| The k = 80 tail is locked at an extreme to counterbalance an off-nominal leg/hip posture | **Refuted on r10 and r6.** Open-loop ΔCoM ≤ 2.1 mm and LEAN + tail survives 0/5 [M]. Removing the rail from 12 railed r10 s42 ends leaves the stance unchanged (COMREV §4). On r6, four tail joints at +1 stood at 3254.0 vs 3274.4 (L005). r9's closed-loop CoM shift is most likely policy posture [I] |
| (Follow-up) Head and tail counterbalance each other | **Statically, mostly:** about the hips the tail offsets 78% of the head + neck moment, and the CoM sits 153 mm ahead of the hips, over the feet. **Actively, not in pitch:** a level tail is within 0.2 mm of its maximum backward reach, so tail pitch can only move the CoM forward. Laterally, tail yaw is a real lever (±12 mm) [M, §2.3a] |
| The neck/head is locked as a counterweight | **Neck: confirmed open-loop.** It is the only effective counterweight [M]. r10 s80's neck −1 habit points this way, but its role there is unmeasured. **Head: refuted.** |
| The saturated joints rest against MJCF stops | **Refuted on r10** except neck +1 and head +1, which no trained policy uses [M]. **True on r7** (head_pitch, knees, tail chain) [D]. On r9, tail_1 rested on its stop with *no* command rail [D]. Unknown for #491's plant (r6) |
| Rails are PD-target windup holding spring tension | **Confirmed on r9 and r10, with different meanings.** On r9, +1 held tail_3 near level against gravity (−0.007 rad, 12.9 N·m), with the spring adding almost nothing there [D]. On r10, a = 0 is already level, and +1 holds tail_3 about 5° above level against its spring (7.1 N·m) [D, M] |
| Pelvis-only rewards leave the internal DOFs unconstrained | **Partly.** True on r6: toes and tail had no pose term, and 9 of 12 rails were there. Largely closed since r7 (toes passive; #504 prices tail pose and saturation), but the tail kernel is narrow (Track 4) and **no term prices joint velocity** |

---

## Track 3: Deconstructing the 270-point gap (−0.27 points per step)

### 3.1 The #489 term table (physics r6), grouped by mechanism [D #489; grouping and arithmetic M]

Δ = policy − statue, over 1,000 steps. Statue 3274.4, policy 3004.3.

| Group | Terms | Σ Δ | Per step | Share of −270 |
|---|---|---|---|---|
| **A. Rate penalties** (the chatter's own price) | smoothness −65.00, action_jerk −41.41 | **−106.41** | −0.106 | **39.4%** |
| **B. Joint posture** | leg_home_pose −84.98, tail −6.46 | **−91.44** | −0.091 | **33.9%** |
| **C. L2 energy** | energy −53.68 | **−53.68** | −0.054 | **19.9%** |
| D. Pelvis rates | drift −10.19, speed −8.29, spin −6.55 | −25.03 | −0.025 | 9.3% |
| E. Support | foot_load_balance −11.63, bilateral_support −4.34, alive −3.62 | −19.59 | −0.020 | 7.3% |
| F. Pelvis attitude and height | height −2.26, heading −0.25, posture −0.16, nosedive **+6.78** | +4.11 | +0.004 | −1.5% |
| G. Neck and head | neck_posture **+17.85**, head_clearance **+4.19** | +22.04 | +0.022 | −8.2% |
| **Total** | | **−270.0** | **−0.270** | 100% |

### 3.2 The brief's three candidate sources

1. **Chronic L2 energy from holding rails: confirmed but secondary, about 20% of the gap (the DC holds alone are about 18%).**
   - Energy = 0.075 × mean(a²) × 1000.
   - #489's closed-form inversion gives a per-actuator DC of 0.800 and AC of 0.277. Then mean(a²) = 0.640 + 0.077 = 0.717, which gives 53.8, matching the −53.68 measured.
   - About 89% of this term (≈ −48, matching #491's "~48/episode") is the DC holds, and about −6 is the tremor.
2. **Micro-velocity and pelvis jitter through `speed` and `drift`: largely refuted.**
   - Speed + drift = −18.5 (6.8%); with spin, −25.0 (9.3%).
   - The mechanism does not fit jitter [C]:
     - Speed = −0.3·min(max(0, |v_xy| − 0.10)/1, 1) (`reward_functions.py:632-645`). It is **zero below 0.10 m/s**.
     - Drift = −0.3·(d/2)², where d is the distance from spawn (`reward_functions.py:84-99`). A 1 cm offset costs 7.5×10⁻⁶ per step.
     - Chatter-level pelvis jitter (mm/s) is effectively free under both.
   - If r6 used today's formulas (unverified, because the clone is shallow):
     - drift −10.19 implies an **rms distance from spawn of about 0.37 m**;
     - speed −8.29 implies a mean excess of about 0.028 m/s above 0.10 m/s.
     - That is **macroscopic creep or shuffling**, not micro-jitter.
3. **Steady-state pelvis height or orientation offset (`posture`, `nosedive`, `height`): refuted.**
   - Group F is a net +4.1 in the policy's favour, and posture is −0.16.
   - Together with the neck/head terms (G), the displaced policy holds trunk and head **better** than the statue (+26.2).

**What dominates instead:** the two terms the brief did not list.
- Rate penalties (A, 39%) are the chatter's own price.
- Joint posture (B, 34%) is the displaced joints' own price.
- Adding energy (C), **93% of the gap is priced in joint and actuator space, not at the pelvis.**

This matches #489's own split: static displacement 49%, tremor 42% [D].

### 3.3 What a rail or chatter costs on r10 [M, `t5_price.py`]

Per actuator, under today's weights (n = 15 actuators). Smoothness, jerk and saturation are priced on the raw clipped command; energy is priced on the filtered command.

| Signal on one actuator | Cost per step | Per 1,000 steps |
|---|---|---|
| DC +1 (a rail) | 0.0383 (saturation 0.0333 + energy 0.005) | 38.3 |
| DC +0.9 | 0.004 | 4 |
| ±1 square wave at 50 Hz (1-step bang-bang) | 0.367 | 367 |
| ±1 square wave at 25 / 16.7 / 5 / 1 Hz | 0.151 / 0.112 / 0.060 / 0.043 | 151 / 112 / 60 / 43 |
| #489 pattern (DC 0.8 + AC 0.277 at 22.6 Hz) | 0.0206 | 20.6 |
| Same AC at 1.2 Hz | 0.0004 | 0.4 |
| Exploration noise σ = 0.22 at μ = 0 | 0.0069 | 6.9 |

Joint pose terms come on top of these. For a tail rail, tail_home_pose costs −43 to −50 over steps 200–1000. A measured single tail rail totals **−97 to −102 per episode** (2.2). The #489 pattern on all 15 actuators would cost about 309 per episode in action penalties alone.

### 3.4 On r10, the statue is *not* the reward optimum under the committed config [M, `t4d_ref.py`, `t2b.py`, `t4c_grid.py`]

`leg_home_pose_reference` defaults to `"keyframe"` (`tyrannosaurus_rex/base_env.py:195,279,1089`) and `stance.toml` does not set it. The keyframe leg pose is not an equilibrium: settling moves the hips −2.6° and the ankles +1.95°.

| Case (3 seeds) | Keyframe reference (default) | Settled reference |
|---|---|---|
| Statue | 3781.7 | 3808.4 |
| 0.25 × LEAN52 | **+24.5** (leg_home_pose +23.9) | −12.2 |
| 0.5 × LEAN52 | **+22.9** | −47.0 |
| hip_pitch +0.02 | **+17.1** (t2b: +0.02/+0.05 give +15/+16) | +0.4 (leg_home_pose −4.6) |

Along the same lean axis, the static return surface (neck = 0, t4c) is:

| Lean | Return | Survivors |
|---|---|---|
| 0 | 3782 | 3/3 |
| 0.25 | 3806 | 3/3 |
| 0.5 | 3805 | 3/3 |
| **0.75** | **1661** | **1/3** |
| 1.0 | 494 | 0/3 |

Implications:
- **[I]** At the statue, the reward gradient points along a leg direction toward a non-equilibrium reference, and the static optimum sits just short of a stability cliff. A policy that follows it is rewarded for holding a pose that needs active stabilisation.
- **[I]** The magnitude (+17 to +25 per episode) is small next to chatter costs. This is a plausible **drift** mechanism into a displaced pose, not on its own an explanation of #489's −270.
- **Not supporting evidence:** the s52/s60 forward-CoP drift (−97/−163, mostly leg_home_pose) in `stance_2026_10/diagnosis/findings_v3.txt:73-81` ran on the draft-#623 foot plant (which that file calls "r10"; §2.3), under an older config with a lean term, and went the other way.
- **Discrepancy:** `docs/PLANT_VALIDATION_AND_STAGE1_OBJECTIVE.md:210-236` states that the statue is the global optimum. Under the default keyframe reference this is off by +17 to +25 points. See Open Question 9.

### Track 3 verdicts

| Hypothesis | Verdict |
|---|---|
| (1) Chronic L2 energy from rails | **Confirmed, secondary**: −53.7 (20%), of which about −48 is DC holds |
| (2) Micro-velocity jitter via speed and drift | **Largely refuted**: −18.5 (7%). The speed dead zone and quadratic drift make jitter almost free; the r6 losses imply creep |
| (3) Steady-state pelvis offset (posture, nosedive, height) | **Refuted**: net +4.1, and the policy beats the statue there |
| (Not in brief) Rate penalties | **Largest single source**: −106.4 (39%) |
| (Not in brief) Joint home-pose terms | **Second largest**: −91.4 (34%) |

---

## Track 4: The "unwinding valley" (why PPO stays trapped)

### 4.1 Pure rails have no valley [M, `t2_pins.py` → `t2/patterns.json`, 6 seeds]

Each pattern is held from reset and then released at step 400 with a linear ramp over T steps. Δ is measured against continuing to hold.

| Pattern released at step 400 | Survive | Δ over steps 400–520, T1 / T200 | Worst cumulative dip | Δ full episode (survivors), T1 / T200 |
|---|---|---|---|---|
| TAIL3_RAIL | 5/6 (same as the statue) | +10.4 / +4.8 | −0.01 | +57.9 / +50.6 |
| NECK_RAIL (neck −1) | 2/6 (the hold falls at steps 250–402 on 4 seeds) | +5.4 / +5.6 | −0.01 | +50.6 / +53.1 |
| P491_noroll (head −1 + four tail +1) | 5/6 | **+48.1** / +24.5 | −0.05 | +287.7 / +258.4 |
| P491 (with mirrored hip-roll rails) | **3/6** (T1–T50) and 2/6 (T200), vs 0/6 held | — | — | Rescues half the seeds |

The reverse direction, applying a rail to the statue at step 400:

| Re-pin at step 400 | Survive | Δ over steps 400–520 | Δ full episode |
|---|---|---|---|
| TAIL3, T1 / T50 | 5/6 | −10.9 / −7.5 | −58.6 / −55.0 |
| NECK, T1 / T50 | **1/6 / 0/6**, falling **277–335 steps after** the re-pin | −17.8 (T1) | — |

- A leg-only lean (LEAN60) released at step 60 rescues 1–2 of 5 seeds; released at step 100, 0 of 5 [M, `t2b.py`].
- So the point of no return is about 100 steps before the nosedive, which comes around step 190.
- **Releasing a pure rail produces no reaction-torque penalty.** The gain is immediate and monotone.
- **The same holds in closed loop on r10, with trained policies** [D, `stance_2026_10/com_revisit/REPORT.txt:105-112`; `TREX_TAIL_SATURATION_HABIT_2026_10.md:624` (L103)]. All rollouts are deterministic:
  - Clamping tail_3 to 0 in the 12 railed s42 ends gains +63.6 per episode (12/12), with lean, roll, sway, creep and margins unchanged.
  - At r10 s42's onset checkpoint, the same hold gains +78.29 per window (20/20).
  - Imposing the rail on the 12 unrailed s80 ends costs −180 (12/12), about half directly and half through the policies' reaction.
  - The one exception is the parked s42 final (§2.4).

### 4.2 Counterbalanced pairs: a delayed cliff, not an immediate barrier [M, `t4b.py`, 5 seeds]

- Pair A = 0.5×LEAN52 + neck −1, which holds at 3710.7.
- Pair B = LEAN60 + neck −1, which holds at 3639.8.
- The statue scores 3780.6.
- Each pair is released starting at step 400 over T steps.

| Release | ΔReturn vs hold (A / B) | Worst cumulative dip | Break-even | ΔV, discounted at γ = 0.98 |
|---|---|---|---|---|
| Both halves, abrupt (T1) | +35 / +71 | −2.1 / −5.0 (head_clearance, speed, foot_flatness) | 86–125 steps | **+0.08 / −0.51** |
| Both halves, smooth (T50, T200) | +36 to +80 | none | 0 | +2 to +4.5 |
| **Neck only, fast (T1, T50)** | **−1,450 to −1,720** | First **gains** +2 to +3.8 over 50 steps (neck_posture + saturation + energy) | never | −2 to −9.6 |
| Neck only, slow (T200) | +51 (survives 5/5) | none | 0 | — |
| Legs only | A: −12 to −17, never breaks even. B: T1 −194 (one fall); T50/T200 +22 to +26 | — | — | — |

- Releasing only the counterweight looks good for about 50 steps, then **nosedives 173–346 steps later**. That is beyond the reach of the discount:
  - The effective horizon at γ = 0.98 is about 50 steps.
  - GAE's γλ = 0.931 gives about a 14-step credit time constant.
  - A fall 200 steps out is worth about 0.98²⁰⁰ × (189 + 100) ≈ **5 value units**. Here 189 ≈ 3.78/(1 − γ) is the forfeited stream and 100 is `fall_penalty` (`stance.toml:372`).
  - That is the same order as the +2 to +3.8 early gain.
- An abrupt *coordinated* release does produce a small transient dip. At γ = 0.98 it is roughly value-neutral (+0.08 / −0.51), even though it pays +35 to +71 undiscounted.

### 4.3 Static return surface over lean × neck [M, `t4c_grid.py`, 3 seeds]

Mean return, with the survivor count in brackets where not all survived.

| Lean \ neck | 0 | −0.25 | −0.5 | −0.75 | −0.9 | −1.0 |
|---|---|---|---|---|---|---|
| 0 | 3782 | 3776 | 3755 | 3732 | 2771 [2/3] | 2698 [2/3] |
| 0.25 | **3806** | 3801 | 3781 | 3758 | 3748 | 3710 |
| 0.5 | 3805 | 3801 | 3782 | 3761 | 3751 | 3713 |
| 0.75 | 1661 [1/3] | 2779 [2/3] | 3757 | 3737 | 3729 | 3691 |
| 1.0 | 494 [0/3] | 628 [0/3] | 1799 [1/3] | 3697 | 3690 | 3653 |

Lean is a scale of LEAN52. CoM x is 92.8 mm for the statue; neck −1 shifts it −24 mm and lean 1.0 shifts it +49 mm.

- From the displaced corner (lean 1.0, neck −1):
  - A **neck-first** path falls off a cliff: survival goes 1/3 at −0.5, then 0/3 at −0.25 and 0.
  - **Lean-first or diagonal** paths rise monotonically to about 3806.
  - This is a **ridge that requires coordinated moves**.
- The local static slope per unit action is about 370 along the neck (+37 per 0.1) and about 150 along the lean (+38 per 0.25).
- So the steepest-ascent direction, which is what a coordinate-wise gradient step would follow, **points at the cliff** [I].

### 4.4 Gradient magnitudes at a rail [C, M]

Per unit action per step, with n = 15:

| Source pulling a rail toward 0 | Gradient |
|---|---|
| L2 energy, 2·w·a/n = 2·0.075/15 (r6: 2·0.075/21 = 0.0071) | 0.010 |
| Saturation ramp, \|a\| ∈ (0.9, 1] (`reward_functions.py:522-546`), *since #504 (r7); absent at #489 time* | 0.333 |
| tail_home_pose at the rail displacement (0.09–0.12 rad = 1.8–2.4 tolerance widths; weight 0.25, tolerance 0.05, single Gaussian, mean of 4 joints; `stance.toml:267-271`) | 0.002–0.015 |
| neck_posture at neck −1 (tolerance 0.35 rad, broad) | ≈ 0.024 |

- **Exponential-tail flatness is confirmed for `tail_home_pose`.**
- `leg_home_pose` already mixes in a 6× broad Gaussian (`broad_fraction = 0.25`, `stance.toml:91-100`; rationale in the docstring at `reward_functions.py:401-440`) against exactly this flatness.

### 4.5 A secondary flatness: the action-clip dead zone [C, M, D]

- SB3 PPO uses a diagonal Gaussian with **no tanh squash**. Actions are clipped to [−1, 1] in the env before pricing (`shared/base_env.py:1237`, `tyrannosaurus_rex/base_env.py:819-833`).
- For a railed channel, the expected cost as a function of the pre-clip mean μ has this slope:

| μ | dCost/dμ at σ = 0.14 | dCost/dμ at σ = 0.22 |
|---|---|---|
| 0.9 | 0.087 | 0.051 |
| 1.0 | 0.084 | 0.051 |
| 1.11 | 0.048 | 0.041 |
| 1.3 | **0.0046** | 0.016 |

- Beyond μ ≈ 1 + 2σ, nearly every sample clips to exactly +1, so **the reward carries no information about μ**.
- **But a rail does not sit that deep.** On r9, s42's tail_3 raw mean was +1.01 to +1.12 on pinned steps, with σ ≈ 0.14, so about 79% of samples clip [D satdiag:84,124].
  - At those means my table gives a slope of 0.084 down to 0.048. That is at most about half of the in-range slope lost, not a flat reward.
  - The habit doc reached the same result independently: the expected reward slope is cut only 1.3–1.9× at means 1.05–1.11.
  - Its status for the dead zone: "contributes at most beyond about mu 1.2" (§4.5, `TREX_TAIL_SATURATION_HABIT_2026_10.md:823-851`; narrowing of L003).
- **Bounding the mean in training did not remove the rail** [D, same section]:
  - P1 left only 2.5% of training rows beyond +1, yet the deterministic mean stayed beyond +1 on 21–46% of window steps.
  - BOUND90 cut training rows above 0.90 in every pair and still read RAIL 6/6.
  - Both tests ran on r9 only.
- **So the dead zone is secondary.** It deepens a rail that has already formed and drifted past μ ≈ 1.2, but it is not what holds the rail in place. §4.6 covers what does.
- The same clip makes **chatter in the pre-clip command of a railed actuator invisible to every term**, rate penalties included [C]. This part stands.

### 4.6 What actually holds the r9/r10 tail rail: a learned positive-feedback loop [D]

This was measured by the repo's clamp and normaliser probes on r10 (habit doc §4.7–§4.8 and §4.11, `TREX_TAIL_SATURATION_HABIT_2026_10.md:906-1079,1153-1262`). I did not reproduce it: it needs the checkpoints.
- **The loop** [`TREX_TAIL_SATURATION_HABIT_2026_10.md:671` (L145)]. On stored states, only the tail_3 angle observation is swept, and the command slope is converted to a loop gain G through the plant's 5.183° per unit command.
  - In the 12 railed s42 ends, the command rises with the observed angle in 12/12 policies.
  - Median G is **+0.630** (range 0.480–0.910), against +0.268 at their start checkpoint. The s80 ends sit at −0.255.
  - G < 1, so the loop settles at a fixed point near c0/(1 − G) instead of running away. With the command at 0° (c0) between +0.148 and +0.644, that point lies **past +1**, so the command parks on the bound.
  - Closed-loop holds agree: the linear relation in §1.4 (0.29–0.33 at a clamped tail, 0.94–1.0 free).
- **The offset** [`TREX_TAIL_SATURATION_HABIT_2026_10.md:682` (L162)].
  - In the s42 lineage, c0 ≈ 0.40 comes from the network reading the lineage's nose-down pelvis pitch and leg angles: −0.157 per degree of nose-up.
  - Giving the s42 ends the s80 ends' posture inputs moves c0 to about 0 and the rail count from 9/12 to 0/12, with the gain unchanged.
  - Showing them a pelvis 1.1° more nose-up cuts their railed episodes from 0.671 to 0.154.
- **The order** [`TREX_TAIL_SATURATION_HABIT_2026_10.md:684` (L164)].
  - In r10 s42, the tail bias rose first, at about 1.20M steps, and the pitch about 0.17M later.
  - At 1.05M the loop was *negative* (G −0.250) and the posture pathway absent. Both grew between 2.1M and 3.15M.
  - The posture is an amplifier learned together with the habit, not its origin.
- **Where it lives** [`TREX_TAIL_SATURATION_HABIT_2026_10.md:680` (L158)]. The gain's growth is in the second hidden layer and the output layer.
  - At fixed weights, the drift of the observation normaliser *lowers* the gain (12/12; see §6.3).
- **What is not known** (habit doc §7 Q1, Q2 and Q14, `TREX_TAIL_SATURATION_HABIT_2026_10.md:1624-1642,1678-1680`):
  - why the loop forms, and what changed at about 1.2M;
  - why some seeds or lineages and not others (s60 parked with the *opposite* pitch sensitivity);
  - whether the r6 many-actuator rails of #491 were held the same way, since no r6 policy was probed.
- **[I] How this fits my measurements.**
  - A rail held by a learned fixed point needs no reward barrier. My open-loop releases (§4.1) and the r10 clamp agree that unwinding a tail rail is immediately profitable. Yet a deterministic policy whose fixed point lies beyond +1 returns to the bound whenever its tail rises.
  - For training to undo it, gradients must lower c0 or G. They act through per-step margins of about 0.02–0.27 (§4.7), the clip-thinned slope (§4.5), and the posture inputs that feed c0 (L162).

### 4.7 Signal vs noise [I, M]

- The undiscounted gains from unwinding are 0.024–0.27 points per step, which is only about 1–13 value units at γ = 0.98.
- They must be found through advantage estimates in a regime where the exploration noise itself costs −48 to −531 points per episode (Track 5).

### Track 4 verdicts

| Hypothesis | Verdict |
|---|---|
| Unwinding triggers immediate speed, drift, posture or nosedive penalties (reaction torques, CoM motion) | **Refuted** for pure rails: worst dip −0.05 [M]. On r10 in closed loop, removing the tail rail gains +63.6 with the stance unchanged [D]. Abrupt coordinated release of a pair dips only −2 to −5 [M] |
| The policy is trapped behind a reward barrier | **Pure rails: refuted.** Release is monotone [M, D], and the r9/r10 tail rail is held by a learned fixed point, not a barrier [D]. **Counterbalanced pairs: confirmed in modified form [M].** It is a coordination ridge plus a delayed cliff beyond the γ = 0.98 horizon, where partial release of the counterweight is locally attractive and fatal 170–350 steps later |
| The energy gradient (−2·w·a/21) is too weak | **Confirmed**: 0.0071 per unit per step on r6. Since #504 (r7) the saturation ramp adds 0.333 per unit in the band |
| Exponential kernels are flat far from nominal | **Confirmed for `tail_home_pose`.** Not for neck (broad) or legs (broad component) |
| *(New)* Clip dead zone | **Secondary.** It halves the slope at μ 1.11 (σ 0.14) and becomes near-total only beyond μ ≈ 1.2–1.3 [M, D]. Bounding the mean on r9 did not remove the rail [D]. It still hides pre-clip chatter [C] |
| *(New)* Learned positive feedback through the tail's own angle observation | **Measured on r10 as the persistence mechanism of the tail rail [D].** Gain 0.63 (0.48–0.91) in 12/12 railed s42 ends vs 0.27 at their start, with the fixed point past +1. How training forms it is unexplained. Untested on r6 |

---

## Track 5: Near-Nyquist oscillation and rate-blindness (#491)

### 5.1 The reward does price rates [C]

| Term | Form and weight | Where |
|---|---|---|
| smoothness | −w·Σ(a_t − a_{t−1})²/(4n), w = 2.0 | `reward_functions.py:115-135`; `stance.toml:337` |
| action_jerk | −w·Σ(a_t − 2a_{t−1} + a_{t−2})²/(16n), w = 3.0 | `reward_functions.py:136+`; `stance.toml:32`; called before smoothness (`shared/base_env.py:496-508`) |
| action_saturation | 0.5 × linear ramp from \|a\| = 0.9 | `reward_functions.py:522-546`; `stance.toml:283,290` |
| energy | −0.075·Σa²/n, on the **filtered** command | `shared/base_env.py:472-480`; `stance.toml:315` |
| Command filter | first order, 10 Hz, α = 0.386 at 100 Hz | `tyrannosaurus_rex/base_env.py:144`; `shared/base_env.py:248,1226-1242` |

- Smoothness, jerk and saturation price the **raw clipped** command (`action_penalty_source = "raw"`, `stance.toml:242`).
- The brief's example (a² = 1 whether held at +1 or flipped ±1 every step) holds **only for energy**.
  - A per-step ±1 flip costs smoothness 4·2.0/(4·15) = 0.133 plus jerk 16·3.0/(16·15) = 0.200 per actuator per step.
  - Measured: **0.367 per step for a flip vs 0.038 for a held rail**, about 10× [M, t5].
- At #489 time (r6) smoothness and jerk already cost −65.00 and −41.41, together 39% of the gap.

### 5.2 What the measured "chatter" is [D, M]

- **#489 (r6, no filter):**
  - Per-actuator AC is 0.277 rms at an **effective 22.6 Hz**.
  - That comes from J/D = (2 sin πfΔt)², a DC-blind estimator under which white noise reads 33.3 Hz.
  - At 100 Hz control this is a **4.4-step period**. Nyquist is 50 Hz, a 2-step period.
- **Phase-locked bounces:**
  - Two failed runs on the same r6 plant converged to a vertical bounce at **exactly 1/6 and 1/5 of the control rate** (16.7 and 20.0 Hz) (`docs/investigations/TREX_STAGE1_BOUNCE_2026_08.md:3,13-22`).
  - The bounce was not reward-preferred: it scored 450 points worse (2554.3 vs the passing policy's 3004.3 and the statue's 3271.8 under one reward; `TREX_STAGE1_BOUNCE_2026_08.md:75-86`).
- **Load-bearing AC:**
  - #489 shows that injecting 0.277-rms open-loop tremor at any frequency from 2 to 40 Hz topples the statue (falls within 38–80 steps).
  - The policy stands with that AC, so **the AC is state-correlated feedback, not noise.**
- **#491 filter probe:** a post-hoc filter at 35/20/10/5 Hz leads to falls at steps 351/289/189/96. #491 itself cautions that this shows *delay* intolerance rather than a bandwidth need.
- **Behind the 10 Hz filter** (#503, policy interface r11, introduced during physics r7 and still active on r10; `configs/plant_versions.toml:370-388`):
  - Its gain is **0.35** at 22.6 Hz and 0.44 at 16.7 Hz (10 Hz: 0.62, 50 Hz: 0.24), with about 36° of phase lag at 10–20 Hz [M, r10].
  - The first gate-pass run trained behind it was `20260810_145546` (physics r7, with the #504 shaping pack; `TREX_STAGE1_GATE_PASS_RUN_2026_08.md:5-6`). It had **zero saturation**.
  - That run still stood when low-passed post hoc at 5 Hz. Its "residual fast content is decoration, not life support" (`TREX_STAGE1_GATE_PASS_RUN_2026_08.md:112-143`) [D].
- **[I]** Oscillations with 4–6-step periods (4.4 effective in #489; exactly 5 and 6 in the bounces) in a loop with one step of policy delay (plus filter lag since interface r11) are the signature of a **high-gain feedback limit cycle at its phase crossover**. They are not 1-step bang-bang, and not Nyquist.

### 5.3 Does the displaced pose need feedback while the statue does not? [M]

**The statue is a passive basin.** 100/100 reset-noise seeds (1000–1099) survive, or 119/120 counting seeds 0–19, where seed 4 nosedives (§1.2). The settled poses agree to 0.25° rms.

**The basin is narrow:**

| Open-loop perturbation | Survive | Notes |
|---|---|---|
| Random static DC offsets on all 15 actuators, rms 0.1 / 0.2 / 0.3 / 0.5 (12 rollouts each) | **0/48** | Median fall step 196 / 100 / 112 / 103 |
| Any single leg rail | 0/6 each | — |
| Knee or ankle offset of ±0.05–0.1 | Mostly fall | — |
| r6 (#489): offset rms 0.10 | — | Falls in 107–268 steps |

- Displaced patterns that stand open-loop either do not move the CoM (tail, neck_yaw and head −1 rails) or pair a leg lean with the neck counterweight.
- **Any displaced leg pose of ≥ 0.05 needs active correction**, which confirms the slope-vs-basin hypothesis for the legs.

### 5.4 Exploration noise forces active feedback [M, `t7_noise.py`, 10 seeds each]

White Gaussian noise is added to the statue's zero mean:

| σ | Channels | Survive | ΔReturn vs statue |
|---|---|---|---|
| 0.02 | all | 10/10 | −48 |
| 0.05 | all | 10/10 | −247 |
| 0.08 | all | 9/10 | −531 |
| 0.11 | all | **1/10** | — |
| 0.14 | all | **0/10** (median fall step 389) | — |
| **0.22** (= `log_std_init = −1.5`, `stance.toml:466`) | all | **0/10** (median fall step 236) | — |
| 0.22 | **legs only** | same as all channels | — |
| 0.22 | neck and head only | 10/10 | −31 |
| 0.22 | tail only | 10/10 | −33 |

**[I] A mechanism linking #489 and #491:**
1. At initialisation, a zero-mean policy falls in about 236 steps under its own exploration noise, and the leg channels alone are equally lethal.
2. PPO must therefore learn **high-gain leg feedback before it can benefit from the passive basin**.
3. A high-gain loop through a one-step delay (plus filter lag) is the classic source of a 4–6-step limit cycle.
4. Once active balancing exists, a displaced operating point costs little extra stability, so nothing pulls the policy back into the narrow passive basin.

**Testable prediction:** feedback gain (∂a/∂obs) and chatter amplitude should track σ over training. The existing data fit, but they are correlational [D]:
- **r6:** policy std against the number of saturated actuators gives r = −0.967. "Saturation appears exactly as noise decays" [`TREX_TAIL_SATURATION_HABIT_2026_10.md:525` (L007)].
- **r9:** before onset, tail_3's σ was equal across stance4's three runs (0.211, 0.209, 0.217). After onset the railed s42 is lowest in all 15 dimensions (`satdiag/report.md:71`).
- **r10:** s42's final tail_3 σ is 0.095, against 0.114 (s60) and 0.123 (s80). It has been below its other tail actuators' since 3,145,728 [`TREX_TAIL_SATURATION_HABIT_2026_10.md:885-897`].
- The habit doc reads the lower σ as *following* the rail on r9 and r10, with cause and effect unknown. The tail loop gain has been measured on r10 (§4.6: 0.27 → 0.63 over a continuation), but not as a function of σ. Neither has it been measured for the **leg** channels, which are the ones this mechanism concerns.

### 5.5 Genuine blind spots [C]

- **Joint velocities are not priced.** Drift and speed read the root `qvel[0:2]` only (`tyrannosaurus_rex/base_env.py:917`; `reward_functions.py:84-99,632-645`). Spin reads pelvis angular velocity. `qvel[6:]` appears in no term.
- **The rate penalties are non-Markov with respect to the observation.** The 63-dim observation has no previous action and no filter state (`tyrannosaurus_rex/base_env.py:853`; `satdiag/report.md:37`).
- **Pre-clip chatter on a railed actuator is invisible** to every term (4.5).
- **Price and actuation diverge behind the filter (since interface r11).** The rate penalties price the raw command while the plant receives the filtered one, which carries 35% of the amplitude at 22.6 Hz. The policy pays full price for chatter of which about a third reaches the joints.

### Track 5 verdicts

| Hypothesis | Verdict |
|---|---|
| No penalty on action deltas (slew rate) | **Refuted.** Smoothness and jerk exist, and existed at #489 time |
| The reward is blind to chatter on saturated actuators | **Mostly refuted.** A ±1 flip costs about 10× a held rail. **True only for chatter that stays beyond the clip** |
| Chatter is near-Nyquist, 1-step bang-bang | **Refuted.** It is a 4.4-step effective period, with 5- and 6-step phase-locked bounces |
| The displaced pose needs active feedback while the statue is passive | **Confirmed.** But the statue's basin is too narrow to survive PPO's own exploration noise |
| *(New)* No joint-velocity penalty | **Confirmed** |
| *(New)* Exploration noise forces leg feedback | **Confirmed** (σ 0.22 gives 0/10; legs alone behave the same) |

---

## Track 6: Critic and actor phase aliasing, VecNormalize, curriculum transfer

### 6.1 What the networks see and how they learn [C]

**Observation.** 63 dims (`tyrannosaurus_rex/base_env.py:853`):
- body state: 21 hinge angles, velocities and IMU channels;
- a 3-dim command segment that is constant zero in Stage 1;
- **no** step count, phase, previous action or filter state.

**PPO** (`stance.toml:390-466`):

| Setting | Value |
|---|---|
| Networks | Separate policy and value MLPs [512, 256] (first layers `mlp_extractor.policy_net.0` / `value_net.0`, `policy_loading.py:890`) |
| Discount | γ = 0.98 (horizon about 50 steps) |
| GAE | λ = 0.95 (γλ = 0.931, about a 14-step credit time constant) |
| Rollout | n_steps 2048 × 4 envs |
| Optimisation | batch 128, 10 epochs, clip 0.2, target_kl 0.03 |
| Learning rate | 3e-5 → 1e-5 |
| Entropy coefficient | 0.005 → 0 over 7M steps |
| Exploration noise | `log_std_init = −1.5` (σ = 0.22), state-independent, no gSDE |

1000-step truncation is bootstrapped by SB3.

### 6.2 Sources of phase aliasing [M, C, I]

1. **The pop.**
   - Pop and quiet states are perfectly separable from the observation in a deterministic rollout (AUC 1.0).
   - Under training noise (σ 0.14–0.22) they are **not** (AUC 0.48–0.65) [M, t6].
   - During training, steps 0–150 and 150+ therefore look alike to the critic, which must blend them [I].
2. **The `stance_width` switch at step 200** (`stance.toml:188`). This is a stronger and *guaranteed* aliasing source:
   - Reward steps up by +0.099 per step [M].
   - It is driven by an unobserved counter, and the states on either side are only weakly separable (AUC 0.71 for steps 150–200 vs 200–250) [M].
   - At γ = 0.98 the value difference approaches 0.099/(1 − γ) ≈ **5 value units**. That is a systematic V(s) error the critic cannot remove in roughly steps 150–200 [I, magnitude M].
3. **Unobserved action and filter history.**
   - Smoothness, jerk and the filter output depend on a_{t−1} and a_{t−2}, which neither network sees.
   - The critic has to infer them from joint velocities. This is partial observability that inflates value variance [I].
4. **Time limit.** It is bootstrapped and is not a meaningful aliasing source [C].
5. **Not measured.**
   - Nothing in the repo logs V(s_t), returns-to-go, GAE or advantages along the policy's own episodes by step index.
   - The closest signal is the policy probe. Every update it records the critic's value at a fixed bank of ten states: the statue at steps 199 and 599, and ankle leans at steps 400 and 599, on reset seeds 3042–3043. It does so under both the current and the session-start normalisation (`environments/shared/policy_probe.py:10-18,73-87`). No bank state lies in the pop, and the bank is built from scripted rollouts, not the policy's.
   - No checkpoints are available locally and torch is not installed here.
   - **The critic's behaviour in steps 0–40 vs 40–1000 could not be observed directly** (Open Question 1).

### 6.3 Is VecNormalize contaminated by the pop? [C, M, `t6_vecnorm.py`]

**Configuration:**
- `norm_obs` and `norm_reward` are True, `clip_obs = 10`, `clip_reward = 50`, and reward normalisation uses γ = 0.98 (`environments/shared/constants.py:19-22`; `train_base.py:220-272`).
- RunningMeanStd is **cumulative with no decay**.
- Evaluation loads the stats frozen (`training = False`, `norm_reward = False`; `policy_loading.py:652-653`).

**Statistics fitted on deterministic statue rollouts.** The first 50 steps are 5% of samples.

| Feature group | Share of variance from steps 0–50 | Overall / quiet std | Quiet \|z\| p99 |
|---|---|---|---|
| qpos | 69% | — | — |
| qvel | **81%** | **14.3×** | 0.40 |
| gyro | 70% | 4.8× | 1.08 |
| linvel | 69% | — | — |
| accelerometer | **92%** | 5.8× | 0.78 |

- Pop samples reach |z| p99 of 16–17, and **2.8–3.5% of pop samples clip at 10**.
- Under σ = 0.14/0.22 noise, the pop's share of qvel variance falls to 13–25%, the overall/quiet std ratio is about 1, and pop-vs-quiet AUC is 0.48–0.65. **Exploration noise dominates the statistics.**

Readings:
- **[I]** Pop contamination is negligible early in training.
- **[I]** Because the stats are cumulative, the late-training normaliser stays a blend of early exploration noise and pop spikes.
- **[I]** As σ shrinks (entropy reaches 0 at 7M, `stance.toml:399-400,432`), quiet-standing features occupy an ever smaller slice of the normalised range. Even a pure-statue fit puts quiet qvel at |z| p99 = 0.40.

**What the repo has already measured about normaliser drift [D]:**
- **Per-checkpoint statistics.** The training-history page `18_obs_normalization` (#645) reads VecNormalize statistics per checkpoint from the saved pickles [`TREX_TAIL_SATURATION_HABIT_2026_10.md:260`].
- **The suspected link.** In the first r10 round, the observation dimension that drifted most was the tail_3 joint angle, the joint that later saturated. The CoM revisit called this an "inferred link, untested" (`stance_2026_10/com_revisit/REPORT.txt:126-128`).
- **The probe** [`TREX_TAIL_SATURATION_HABIT_2026_10.md:680` (L158)]. The normaliser probe tested that link on r10 s42.
  - The tail_3 angle's normaliser scale fell from 1.281 to 0.560 per degree.
  - Yet at fixed weights this drift *lowers* the loop gain (12/12; log-log slope −0.81, where a pure scale effect would give +1).
  - The gain's growth sits in the downstream weights (§4.6).
- **Still open.** Did the drift shape training, with PPO growing those weights partly to compensate for a compressed input? Answering that needs a continuation with frozen statistics (ECS add-on N) [`TREX_TAIL_SATURATION_HABIT_2026_10.md:1643-1648`].
- **Plant transfer.** Eight r9 checkpoints run zero-shot on r10 saw their tail observations sit 5–8 SD outside the r9 normaliser's statistics [`TREX_TAIL_SATURATION_HABIT_2026_10.md:611` (L089)].
- **[I] How this relates to the pop.** These drifts are slow and come from steady standing, driven by the policy's own posture (a tail rising to +5°), not from pop spikes. Under training noise my numbers put the pop's share at 13–25%, which is consistent with a minor late-training role. The pop's per-feature effect on a *trained* normaliser was not measured.

### 6.4 Curriculum transfer: what crosses from Stage 1 to Stages 2 and 3 [C]

**Weights:**
- `load_sb3_model(load_path, env=train_env, …)` (`train_base.py:596`) restores the **entire archive**: actor, critic, state-independent `log_std` and the Adam moments.
- The new stage's `policy_kwargs` (net_arch, `log_std_init`) are popped (`train_base.py:563`) and not applied to a loaded model, so **Stage 2 inherits Stage 1's σ and its critic**.

**Observation width:**
- It must match exactly. Plant identity is validated before the load (`train_base.py:582-602`), and SB3 refuses another observation space.
- If the parent never saw live commands, the first-layer command columns of **both** the policy and value nets, plus their Adam moments, are zeroed (`policy_loading.py:899-930`; `train_base.py:618-636`). Command-blindness is then asserted (`policy_loading.py:956+`).

**Widening precedent.** `environments/shared/scripts/widen_checkpoint.py` appends zero columns to `policy_net.0` and `value_net.0`, and pads the Adam moments and the VecNormalize sidecar, across one policy-interface bump. This is the Phase C command segment, BEHAVIOR_RECIPES_PLAN §4.6.

**VecNormalize:**
- **Only `obs_rms` transfers; `ret_rms` is reset** (`curriculum/checkpoints.py:619+`; `train_base.py:429-512`).
- Stats keep updating in the new stage, and a missing sidecar fails closed.
- The command slice is reseeded only for live command modes (`train_base.py:461-473`; every committed stage is `"none"`, so no load reseeds today).

**Stage-entry warm-up (PPO).** clip_range 0.02 and ent_coef 0.02 for the first 100k steps (`curriculum/advancement.py:476-510`).

**Constraints any phase context would face** [I; descriptive, not a proposal]:
1. A new observation feature changes the width, so it must go through the identity and widening path, or the load is refused.
2. Its `obs_rms` would be fitted on Stage 1's 1000-step stance episodes and carried forward.
3. The inherited critic's value scale was learned under Stage 1's `ret_rms`, which is reset at transfer. The critic is therefore mis-scaled at Stage 2 entry regardless of any phase input.

### Track 6 verdicts

| Hypothesis | Verdict |
|---|---|
| The critic blends value estimates across the pop and settle regimes | **Not measurable here.** Nothing logs V(s) or GAE along episodes by step; the probe's critic bank starts at step 199; no checkpoints. Structurally likely under training noise, because the pop is not separable from the observation (AUC 0.48–0.65). The stronger, guaranteed aliasing is the **unobservable `stance_width` switch at step 200** (about 5 value units) and the **unobserved action history** |
| VecNormalize is contaminated by pop spikes | **Conditional, and unmeasured on trained stats.** Dominant in a deterministic fit (69–92% of variance, 14.3× qvel std inflation, about 3% of pop samples clipped). Negligible under σ ≥ 0.14 [M]. The one normaliser drift measured on trained stats (tail_3 angle, r10) is posture-driven, and at fixed weights it opposes the rail's loop [D] |
| Transfer constraints | **Documented above.** Full model plus `obs_rms`; `ret_rms` reset; exact observation width; a zero-column widening path exists |

---

## 7. Hypothesis scorecard (all tracks)

| # | Hypothesis (from the brief unless marked *new*) | Verdict | Evidence |
|---|---|---|---|
| 1a | A pop or settling transient exists at reset | **Confirmed** | §1.1–1.2 |
| 1b | `qpos0` is a static equilibrium | **Refuted** | §1.1, §1.5 |
| 1c | Actuators slam to ±1 during the pop and fail to unwind | **Refuted on r9 and r10.** The command is about 0 at the KE peak, then climbs monotonically to 0.9 by about steps 178–250, driven by the tail's own angle. Untested on r6 | §1.4, OQ 5 |
| 1d | The pop explains a material share of the gap | **Refuted** (≈3–26 pts) | §1.3 |
| 2a | The k = 80 tail is railed as a counterweight | **Refuted** on r10 [M, D] and r6 [D] | §2.3 |
| 2b | The neck is railed as a counterweight | **Confirmed open-loop**; the only effective one. r10 s80's neck −1 habit is unexamined | §2.3, OQ 13 |
| 2c | The head is railed as a counterweight | **Refuted** | §2.3 |
| 2d | Saturated joints rest on MJCF stops | **Refuted on r10**, except neck +1 and head +1, which no policy uses (on r9, tail_1 rested on its stop with *no* command rail). **True historically on r7**; unknown for #491's plant (r6) | §2.4 |
| 2e | Rails are PD-target windup holding spring tension | **Confirmed as windup.** On r9 it held tail_3 near level against gravity, with the spring adding almost nothing; on r10 it holds the tail about 5° above level against spring tension | §2.4 |
| 2f | Pelvis-centred rewards leave the internal DOFs unconstrained | **Partly**: yes on r6. Pose terms have closed it since r7, but joint velocity is unpriced | §2.6, §5.5 |
| 3a | The gap comes from chronic L2 energy | **Confirmed, secondary** (20%) | §3.1–3.2 |
| 3b | The gap comes from speed and drift jitter | **Largely refuted** (7%; implies creep) | §3.2 |
| 3c | The gap comes from a steady-state pelvis offset | **Refuted** (net +4.1 for the policy) | §3.2 |
| 3d | *New:* rate penalties dominate the gap | **Confirmed** (39%) | §3.1 |
| 3e | *New:* joint home-pose terms are the second source | **Confirmed** (34%) | §3.1 |
| 3f | *New:* the statue is the reward optimum under the committed config | **Refuted** (+17 to +24.5 for a small lean; keyframe reference) | §3.4, OQ 9 |
| 4a | Unwinding triggers immediate penalties | **Refuted**: open-loop [M], and in closed loop for the r10 tail [D] | §4.1–4.2 |
| 4b | The policy is trapped behind a reward barrier | **Pure rails: refuted**; they are held by a learned loop, not a barrier. **Counterbalanced pairs: confirmed in modified form** (coordination ridge plus a cliff 170–350 steps later, beyond the γ = 0.98 horizon) | §4.1–4.3, §4.6 |
| 4c | The energy gradient is too weak to pull rails back | **Confirmed** | §4.4 |
| 4d | Exponential kernels are flat far from nominal | **Confirmed for `tail_home_pose` only** | §4.4 |
| 4e | *New:* the action-clip dead zone removes the gradient at the rail | **Secondary.** The slope is about halved at μ 1.11 and near zero only beyond μ ≈ 1.2–1.3. Bounding the mean on r9 did not remove the rail | §4.5 |
| 4f | *New:* a learned positive-feedback loop through the tail's own angle holds the rail | **Measured on r10 [D]**: gain 0.63, with the fixed point past +1. Its origin in training is unexplained | §4.6 |
| 5a | No penalty on Δa | **Refuted** | §5.1 |
| 5b | The reward is blind to chatter on saturated actuators | **Mostly refuted**; true only for chatter that stays beyond the clip | §5.1, §5.5 |
| 5c | Chatter is near-Nyquist 1-step bang-bang | **Refuted** (4.4-step effective period; 5- and 6-step locked bounces) | §5.2 |
| 5d | The displaced pose needs active feedback while the statue is passive | **Confirmed, with caveat**: the statue's basin does not survive σ = 0.22 | §5.3–5.4 |
| 5e | *New:* joint velocities are unpriced | **Confirmed** | §5.5 |
| 5f | *New:* exploration noise forces leg feedback | **Confirmed** | §5.4 |
| 6a | The critic blends values across the pop and settle regimes | **Not measured.** Structurally plausible under noise; the step-200 `stance_width` switch and the unobserved action history are stronger, guaranteed sources | §6.2, OQ 1 |
| 6b | Pop spikes contaminate VecNormalize | **Conditional**: dominant in a deterministic fit, negligible at σ ≥ 0.14. The only drift measured on trained stats is posture-driven (tail_3 angle, r10) | §6.3, OQ 10 |
| 6c | What constrains Stage 1 → 2/3 transfer | **Documented** | §6.4 |

---

## 8. Synthesis: the causal chain most consistent with the evidence [I]

This is an inference that links measured pieces, not a measurement. Each link cites its support.

1. **Noise first.** With σ = 0.22, a zero-mean policy falls in about 236 steps, and the leg channels alone are lethal (§5.4). PPO must learn **high-gain leg feedback** before the passive basin is reachable.
2. **Feedback produces chatter (#491).** A high-gain loop with one step of policy delay oscillates at its phase crossover; since interface r11 the filter adds lag. That matches the 4–6-step periods seen: 22.6 Hz effective, and bounces locked at 16.7 and 20 Hz (§5.2). Smoothness and jerk price this (39% of the r6 gap) without removing it, because the observation lacks the action history that the price depends on (§5.5).
3. **Active balancing makes displacement cheap (#489).** Once the body is balanced actively, a displaced operating point costs little extra stability.
   - Today (r10): the keyframe-referenced `leg_home_pose` *rewards* a small lean (§3.4). The neck is the one counterweight that makes larger leans statically stable (§2.3).
   - r6: toes and tail had no pose term, so those DOFs were free (§2.6).
4. **Lock-in comes in two kinds.**
   - **Counterbalanced pairs** can only be unwound jointly [M]. Partial release fails 170–350 steps later, beyond the effective horizon, and the steepest static ascent points at the cliff (§4.2–4.3).
   - **Single-actuator rails** (the r9/r10 tail habit) are held by the policy itself [D]. A learned positive-feedback loop runs through the actuator's own angle observation, and its fixed point lies past +1 (§4.6).
   - The reward favours unwinding at every tested railed checkpoint, so this is a trap in the policy's dynamics, not a reward barrier. The clip dead zone (§4.5) and the flat tail kernel (§4.4) only weaken the restoring signal.
5. **The r9/r10 tail rail is a learned habit, not a balance strategy or a plant requirement** [D].
   - On r9, gravity put a level tail just out of reach (a = +1.07), so any lasting tail-up preference ended on the rail. In the habit doc's words, "The plant enables the habit but does not cause it" (`TREX_TAIL_SATURATION_HABIT_2026_10.md:721-725`).
   - r10 removed that constraint: a = 0 now holds the tail level. The rail still formed (s42; +1 now holds the tail about 5° above level) or parked (s60, at +0.64) (§2.1, §2.4).
   - It is not a counterweight on r10 or r6 (§2.3).
   - Why some seeds or lineages form it is unexplained. The earliest marker found is a posture-independent rise of the tail bias at about 1.2M steps in r10 s42 (§4.6; OQ 12).
6. **[I] One kind of mechanism, two outcomes?** The leg feedback (steps 1–2) and the tail loop (step 4) are both learned observation-to-command gains.
   - In the legs the gain is needed for balance, and it shows up as chatter. In the tail it is not needed, and it shows up as a rail.
   - Saturation appears as σ falls (r6: r = −0.967), and railed runs end with the lowest σ on r9 and r10 (§5.4).
   - Whether chatter amplitude and leg gain track σ the same way is unmeasured (OQ 7). This is a hypothesis linking the two issues, not a finding.

**Ruled out as primary drivers:** the reset pop (§1.3–1.4), pelvis offsets (§3.2), immediate unwinding penalties (§4.1), the tail as a counterweight (§2.3), and the clip dead zone as the cause of persistence (§4.5).

---

## 9. Open questions and missing instrumentation

These are ordered by how much each would change the picture. Each is a measurement, not a fix.

**Tooling that already exists** (the habit doc's detector table, `TREX_TAIL_SATURATION_HABIT_2026_10.md:244-262`):
- **#638 per-step trace tool** (`environments/shared/scripts/episode_trace_report.py:1-27`): raw mean, command, `data.ctrl`, joint angle and velocity, and force, per step and actuator. Options `--hold NAME=VALUE` and `--stochastic SEED`.
- **#640:** per-episode actuator naming in the ledger, and per-actuator clip shares of the training rollouts in `policy_probe.npz`.
- **The policy probe** (`policy_probe.npz`, `environments/shared/policy_probe.py:10-18,73-87`): every update, the per-actuator σ, the VecNormalize statistics, and the critic's value and deterministic mean action at ten fixed bank states, under both the current and the session-start normalisation.
- **#644/#645:** window-mean commands per actuator, at-rail and parked actuators, and the `18_obs_normalization` page.
- **Training diagnostics:** `diagnostics/raw_action_saturation`, and SB3's per-update `train/explained_variance` (`environments/shared/diagnostics.py:81-83,416-436`).

The tail habit has its own 17 open questions in the habit doc's §7 (`TREX_TAIL_SATURATION_HABIT_2026_10.md:1603-1726`). The items below avoid repeating them.

1. **Critic error by episode step.** *(Resolves 6a; tests the step-200 aliasing.)*
   - Record per-step V(s_t), return-to-go and GAE advantage with the episode step index, at about 5 checkpoints.
   - Bucket by step: 0–50, 50–150, 150–200, 200–250, 250–1000.
   - Prediction: a systematic V − G residual of up to about 5 value units in steps 150–200.
   - **Exists:** pooled per-update explained variance, and the probe's V(s) at ten fixed bank states (steps 199–599).
   - **Missing:** V(s), return-to-go and advantage along the policy's own episodes by step, including any state in the pop.
2. **Pre-clip statistics against onset.** *(Dates the clip-zone entry, 4e.)*
   - **Exists:** `policy_probe.npz` stores per-actuator shares of sampled actions and of Gaussian means beyond each bound (added by #640), plus `std_per_actuator`.
   - **Missing:** a read of that series against each run's onset, per actuator, leg channels included.
   - **Missing:** a summary of the ECS per-actuator training-row shares, which were never read (habit doc §7 Q12).
3. **Per-actuator spectra.** *(Separates load-bearing from decorative AC on r10.)*
   - **Exists:** the #638 traces carry the signals.
   - **Missing:** the PSD of the raw command, the filtered command and `qvel[6:]`.
   - **Missing:** cross-coherence and phase between channel pairs (L/R hip pitch, knee vs ankle, neck vs leg lean), to detect antagonistic co-contraction that cancels at the pelvis.
4. **Closed-loop unwind probes on real checkpoints.** *(Tests 4b on trained policies.)*
   - **Exists:** #638 `--hold` does constant holds. tail_3 = 0 has been measured: +73 to +86 on r9, and +63.6 to +78.3 on r10, except −64.5 at the parked s42 final.
   - **Missing:** ramped releases (T = 1, 50, 200) instead of constant holds.
   - **Missing:** lean + neck pair releases, joint and one-half.
   - **Missing:** discounted against undiscounted return over the remaining horizon.
   - **Missing:** tail_1/tail_2 clamps and stochastic clamps (habit doc §7 Q13).
5. **r6 onset timing on `20260805_011234`.** *(The only open half of 1c.)*
   - Find the first step at which each of the 12 rails crosses |a| ≥ 0.9 within an episode, and check whether any falls in steps 0–50.
   - The r6 checkpoint drives 21 actuators against today's 15. The trace tool's `--allow-legacy-plant` (`episode_trace_report.py:95-99`) waives only the plant check, so this needs the r6-era code.
6. **Tail antagonism.** On r9 the tail_1 command moved opposite to tail_3 during the ramp, with tail_1 on its stop (§1.4). On r10, tail_1 is preloaded off its stop.
   - Does the counter-move persist? The tail_1–3 commands are in the #638 traces.
   - Joint-limit and passive torques (`qfrc_constraint`, `qfrc_passive`) are not recorded and would need adding.
7. **Leg feedback gain against σ over training.** *(Tests §8 steps 1, 2 and 6.)*
   - **Exists:** the tail_3 loop gain was swept at 146 r10 checkpoints. The s42 medians rise from 0.268 to 0.630, and s80 goes from −0.370 to −0.255 [`TREX_TAIL_SATURATION_HABIT_2026_10.md:680` (L158)]. The per-actuator σ series is in `policy_probe.npz`.
   - **Missing:** the same observation sweep for the **leg** actuators against pelvis pitch, pitch rate and CoM-related inputs, recorded with σ and the AC rms of the raw command.
   - Prediction: gain and chatter are set early, while σ is large.
8. **Loop identification.** Measure the plant's frequency response from command (through the 10 Hz filter) to pelvis pitch and CoM. Combine it with the measured policy gain to predict the limit-cycle frequency with a describing function. Check that against the 16.7–22.6 Hz observed on r6.
9. **Keyframe vs settled leg reference.** *(Resolves 3f and the discrepancy with `PLANT_VALIDATION_AND_STAGE1_OBJECTIVE.md:210-236`.)*
   - Re-score existing checkpoints per term under both references.
   - Check whether the trained leg offsets point from the settled pose toward the keyframe.
10. **Pop and quiet-stance features under trained statistics.** *(Resolves 6b in practice.)*
    - **Exists:** the `18_obs_normalization` page and the normaliser probe cover per-checkpoint drift (§6.3).
    - **Missing:** the quiet-standing |z| distribution of qvel, gyro and accelerometer under each checkpoint's stats.
    - **Missing:** the pop's share inside those stats.
    - Whether drift shapes training needs the frozen-statistics continuation (ECS add-on N; habit doc §7 Q4).
11. **Old drift and speed definitions.** #489's drift and speed losses imply about 0.37 m rms creep, if the formulas matched today's. The shallow clone prevents checking the old code here. Verify at the #489 commit, or replay the r6 checkpoint's pelvis XY track.
12. **Seed and lineage selection.** This is open in the habit doc too (§7 Q1–Q2). Its earliest marker is a posture-independent rise of the tail bias at about 1.2M steps in r10 s42. Neither document compares seeds around that window on:
    - σ by dimension (logged: `policy_probe.npz` `std_per_actuator`);
    - termination histories (logged per rollout, `environments/shared/diagnostics.py:85-86`);
    - critic explained variance (logged per update);
    - the sign of tail_3's advantage, which nothing logs.
13. **r10 s80's neck habit.** *(New; links Track 2's only counterweight to a trained policy.)*
    - s80 alarmed on `neck_pitch` at −1 (§2.1). Is it acting as a counterweight there?
    - Clamp it to 0 in closed loop, as the tail clamp test did, and read the CoM shift, falls and return.
    - Check whether s80 holds a lean + neck pair of the §4.2 kind.
14. **The r6 rails and the loop.** *(New.)* Were #491's 10–12 rails held by the same kind of observation-to-command loop as the r10 tail? An observation sweep on the r6 checkpoint, with r6-era code, would answer it.

---

## Appendix: reproduction

**Environment.**
- Python 3.12 venv with the repo's core dependencies only: `python3 -m venv /tmp/mesovenv && /tmp/mesovenv/bin/pip install mujoco==3.10.0 gymnasium numpy` (here: gymnasium 1.4.0, numpy 2.5.4).
- No torch or SB3 is needed. Nothing is trained or loaded.
- `common.py` puts the repo on `sys.path` and builds the real env through `environments.shared.config.build_env(ModelId("tyrannosaurus_rex", "anatomical"), 1)`. That is Stage 1, `stance.toml`: `TyrannosaurusRexAnatomicalEnv`, 15 actuators, a 63-dim observation.
- The scripts are in [stance_2026_10/pop_pose_saturation/](stance_2026_10/pop_pose_saturation/README.md), as they ran except for two path substitutions (SOURCES.txt's **prefix** and **checkout**). The session folder they read and wrote (`D`) is now `scratchpad`, and the checkout (`REPO` in `common.py`) is now `checkout`. Both are relative to the working directory.
- To rerun a script, copy that folder to `<work>/scratchpad` and link the checkout as `<work>/checkout`. Then, from `<work>`, run `/tmp/mesovenv/bin/python scratchpad/<script> [arg]`. Rerun this way on 2026-10-30, `t4d_ref.py`, `t5_price.py` and `t8_headtail.py` reproduced their kept outputs byte for byte.
- Rollouts are 1000 steps, deterministic given the seed. Seed `None` means a noise-free reset (`custom_reset(env, u, 0)` with zero noise); integer seeds use the env's native reset noise. Pools use 2 worker processes.

**Scripts and outputs:**

| Script | Tracks | What it measures | Seeds | Output |
|---|---|---|---|---|
| `common.py` | — | Env factory, joint and actuator tables, `custom_reset`, `rollout` with per-term reward capture | — | — |
| `t1_pop.py [NSEED]` | 1, 6 | Pop timeline, KE and foot force, reset penetration, joint settling vs `qpos0`, per-window rewards; native, qpos-only and qvel-only reset noise | noise-free plus `reset(seed=0..NSEED−1)`, NSEED = 20 | `t1/t1_results.json`, `t1/t1_traces.npz` |
| `t2_pins.py pins` | 2, 4 | Each of 15 actuators held at ±1 open-loop from reset: survival, joint deflection, ΔCoM, tilt, per-term ΔReturn | None, 0–4 | `t2/pins.json` |
| `t2_pins.py patterns` | 2, 4, 5 | #491 and satdiag patterns, random static DC offsets (rms 0.1–0.5, `default_rng(7)`), release starting at step 400, ramped to 0 over 1 step (T1) or 200 steps (T200) | None, 0–4 | `t2/patterns.json` |
| `an_pins.py <json>` | — | Summariser for the two files above | — | stdout |
| `t2b.py` | 2, 4, 5 | Leg-offset tolerance, neck/tail counterweights for a leg lean, early lean unwinds | None, 0–3 | `t2b.json` |
| `t4b.py` | 4 | Counterbalanced lean + neck pair: joint vs one-half release, abrupt vs slow | None, 0–3 | `t4b.json` (23 MB, full traces) |
| `t4c_grid.py` | 4 | Static return surface over lean scale × neck command | None, 0, 1 | `t4c_grid.json` |
| `t4d_ref.py` | 3, 4 | Statue vs small leans under `leg_home_pose_reference` = keyframe vs settled (set on the unwrapped env) | None, 0, 1 | `t4d_ref.txt` |
| `t5_price.py` | 3, 4, 5 | Per-actuator price of DC rails, flips and sinusoids under the current weights (energy, smoothness, jerk, saturation) | — (analytic, current config) | `t5_price.json` |
| `t6_vecnorm.py` | 6 | Observation statistics: pop vs quiet variance shares, std ratios, \|z\| quantiles, clip fraction, pop and step-200 separability (AUC) at σ = 0, 0.14, 0.22 | 100–111 | `t6_vecnorm.json` |
| `t7_noise.py` | 4, 5 | Statue under per-step Gaussian action noise by channel group and σ; 100-seed statue baseline | noise 200–209; statue 1000–1099 | `t7_noise.json` |
| `t8_headtail.py` | 2 | Settled-statue gravity moments about the hip axis by body group; CoP and contact span; kinematic CoM authority of the neck, head and tail joints (pitch and yaw, with the root frozen); whole-tail swing ±12–60° | None | `t8_headtail.json`, `t8_headtail.txt` |

**Outputs kept.** Every output named above is in the evidence folder, with three exceptions:
- `t2/pins.json`, `t2/patterns.json` and `t2b.json` are kept gzipped (`.json.gz`), because each is over the repository's 500 KB limit for an added file.
- `t1/t1_traces.npz` (3.4 MB) and `t4b.json` (23 MB of full traces) are not kept. Their sha256 is in the folder's README, and their scripts rewrite them.

**Caveats on the measurements:**
- All of them are open-loop: scripted commands, no trained network.
- Trained-policy behaviour is taken only from committed repo telemetry and the #489/#491 issue bodies, cited inline: `TREX_TAIL_SATURATION_HABIT_2026_10.md`, `stance_2026_10/stance4/satdiag/`, `stance_2026_10/com_revisit/`, `stance_2026_10/diagnosis/`, `TREX_STAGE1_BOUNCE_2026_08.md` and `TREX_STAGE1_GATE_PASS_RUN_2026_08.md`.
- The r6 21-actuator plant (`20260805_011234`) and the r9 plant cannot be rebuilt from this checkout. Claims about them are [D] only, and every [M] number is r10.
