# Saturation and foot placement at 4,194,304 steps (Run 2 first; Runs 1 and 3 and the statue for comparison)

This is a report only. Nothing was changed in either clone. Both trees are clean at 7e6b557d (L) and 1af2d4d7 (L+T).

## How the numbers were made
- `roll_sat.py` was run through `tools/iso.py`, with `ISO_WT` set to the arm's clone. It builds the judge's `Ctx(arm)`, so it uses the clone's stance stage with its `[env]` kwargs as loaded. It loads the policy with `policy_loading.load_sb3_checkpoint`, using the VecNormalize stats, and predicts with `model.predict(deterministic=True)`. That is `stance_report._load_policy`'s predict, and SB3 clips its output to [-1, 1]. The rollout goes through `stance_report.run_panel(floor_truth=True)` on seeds 3042-3081 at heading 0. The statue is `_zero_action_predict`, rolled on the same seeds.
- The raw Gaussian mean (before the clip) is recomputed from the same normalised observation. It never reaches `env.step`.
- Readings are taken after each `env.step`, in a wrapper around `SubstepContactRecorder.end_step`, as the judge does.
- Pass 1 is in `out/`. Pass 2 is in `out2/` and adds sole pitch and root height. The tables come from `out2/`. The two passes are identical on every shared field.
- **Reproduction check** (`tables/reproduction_check.txt`):
  - For all 3 runs and the statue, every seed's length and termination reason equals the judge JSON's. Run 1 falls on 3059 (339 steps) and 3063 (300), both `tail_contact`. Run 3 falls on 3050 (218) and 3063 (281), both `fallen`. The statue falls on 3081 (292, `nosedive`).
  - The per-foot `foot_cop_fore_aft` and `max_sole_cop_fore_aft` are bit-identical to the judge's.
  - Run 2's per-episode `max_actuator_saturation_fraction` equals the judge's reason value in all 40 episodes. My per-actuator recomputation also equals `stance_metrics`.
  - `model.predict` equals clip(raw mu), and the recorder's action equals `model.predict`, both with max difference 0.
  - The statue trajectories in the L and L+T clones are identical (max qpos difference 0).
- Window: steps 200 up to the episode length, i.e. 200-999 when the episode is full horizon. Means are pooled over every window step of every episode. The feet tables use full-horizon episodes only.

## 1. Per-actuator saturation (`tables/per_actuator_saturation.csv`, `per_episode_saturation.csv`)
- **The gate's threshold is 0.99, not 0.9.** `SATURATION_ABS = 0.99` in `environments/shared/gait/constants.py:98` of both clones. `stance_metrics._saturation_metrics` uses `|action| >= SATURATION_ABS`.
- The 0.9 is the env's reward ramp. It is `action_saturation_threshold = 0.9`, with `action_saturation_weight` 0.5 and `action_penalty_source` "raw".
- **The gate reads clipped actions.** `SubstepContactRecorder.end_step` stores `np.clip(action, -1, 1)` (`recorder.py:644`). For any threshold up to 1, |raw| >= thr exactly when |clipped| >= thr, so the fractions are the same for raw and clipped.
- The table gives both thresholds, the mean raw mu, the mean clipped action, the mean |clipped| and the sign.
- **Run 2: one actuator, `tail_3_pitch_act`, in all 40 episodes.**
  - Its per-episode fraction at 0.99 runs from 0.786 to 1.000 (mean 0.963). At 0.9 the mean is 0.9925.
  - It is the argmax actuator in 40 of 40 episodes.
  - No other actuator is above 0.10 at either threshold in any episode.
  - Raw mu: mean +1.0455 (p5 +1.0045, p50 +1.0519, max +1.1335). It is above 1 on 95.5% of window steps.
  - Clipped: mean +0.9974, so the sign is +, which raises the tail tip.
  - Within an episode, the first step at 0.99 has a median of 212 (range 23-345). The mean clipped action per 100-step block is +0.36, +0.80, +0.98, +1.00, then +1.00 to the end.
  - In training (`eval_episodes.csv`, keys only), 6-10 of 10 evaluation episodes list the saturation rail at every evaluation from 2,228,224 steps onward, all of them full horizon. Before that, only one evaluation lists it: one episode at 1,310,720, a 137-step fall (`tables/eval_saturation_history.txt`).
  - Run 2's other large mean: `head_pitch` -0.52.
- **Run 1 and Run 3: 0 window steps at 0.99 or at 0.9 on any actuator.**
  - Run 1's largest |mean| values are `tail_3_pitch` -0.375 and `tail_2_pitch` -0.270. Its tail_3 action points the opposite way from Run 2's.
  - Run 3's largest are `neck_pitch` -0.700 and `head_pitch` -0.298. Its `tail_3_pitch` mean is +0.035.

## 2. The saturating actuator: angle, target and force (`tables/joint_target_force.csv`, `extra_stats.txt`)
- **Model data for `tail_3_pitch`:** position servo, kp 60. Joint range = ctrlrange = ±0.2094 rad. forcerange ±50 N·m. Keyframe home 0.
- **Run 2, `tail_3_pitch`:**
  - The mean applied ctrl target (after the 10 Hz filter) is +0.2088 rad, at the upper stop.
  - The joint's mean angle is -0.0066 rad (-0.38°, sd 0.0023). That is 48% of the way through its range, mid-range and about 0.21 rad short of the target.
  - The mean force is +12.9 N·m, 0.26 of the 50 N·m limit. No window step reaches 0.99 of the limit.
  - **So the action is pinned while the force is not.** Within the joint range, the servo can give at most kp × (0.2094 − q) ≈ 13 N·m at this angle. The tail tip hangs where that torque balances its weight. Pinning the action buys the maximum lift torque, not a reachable target (inferred from the servo law; the numbers are measured).
- **What pulls on this joint:**
  - The L+T **leg_home_pose** term covers the 8 leg joints only (r/l hip pitch, hip roll, knee, ankle), so it has no direct pull on the tail.
  - Its settled targets are: hip pitch -0.09319, hip roll ∓0.00737 (r −, l +), knee -0.43976, ankle 1.37137 rad. Tolerance 0.05 rad, broad component 0.25 × 6 (width 0.30), weight 1.0. Run 1's term has tolerance 0.10 and weight 0.5.
  - The term on the tail is **tail_home_pose**: weight 0.25, tolerance 0.05 rad, no broad component. Its targets are the settled droop: tail_1_pitch -0.2107, tail_1_yaw 0, tail_2_pitch -0.2029, tail_3_pitch -0.0926 rad.
  - Run 2's tail_3 sits +0.086 rad above its target. Its per-joint quality is 0.053, against 1.000 for the statue. The whole tail term's quality is 0.743, against 0.9998 for the statue.
  - By arithmetic from the measured quality and weight 0.25, that costs about 0.059 reward per step. The reward ramp adds 0.5/15 = 0.033 per step for one pinned actuator. The policy pays both, so it is buying something else; what that is was not measured.
- **The other runs and the statue:**
  - Statue: tail_3 at -0.0925, force 5.55 N·m.
  - Run 1: -0.1244.
  - Run 3: -0.0884.
  - tail_1 and tail_2 pitch rest on their ventral stops (about -0.21 rad) in every run and in the statue.
- **Run 2's leg joints** sit within 0.005 rad of the settled targets:
  - hip pitch -0.0896, knee -0.4432 / -0.4436, ankle 1.3753 / 1.3762 (78.80° / 78.85°);
  - the statue is at 78.57°;
  - the leg term's quality is 0.973 (statue 0.982 under the same term).
  - The leg forces are far from their limits: hip pitch 58 of 1800 N·m, ankle 59-60 of 1350.
  - Run 1, with the looser term, sits further off: hip pitch -0.151 (0.058 from the target), ankle 1.402.

## 3. Run 2's split fore-aft loading (`tables/feet_summary.csv`, `feet_per_episode.csv`, `feet_split_stats.txt`)
- **On average the feet are level.**
  - The foot sites are 0.1263 m (r) and 0.1262 m (l) ahead of the root body along the pelvis heading. The mean stagger is +0.02 mm.
  - Foot yaw against the heading: r +0.035°, l -0.026°.
  - Ankles: 78.80° and 78.85°.
  - MTP d2/d3/d4: 12.36/12.37/12.30° (r) and 12.34/12.36/12.30° (l).
- **Episode by episode, one foot sits ahead.** The stagger has sd 11.9 mm, a mean |stagger| of 9.6 mm, and a range of -21 to +23 mm.
  - The forward foot is heel-loaded and the rear foot is toe-loaded:
    - corr(r−l CoP, stagger) = -0.972, with slope -0.086 span half-lengths per mm;
    - in 37 of 40 episodes the two feet's signed CoPs have opposite signs;
    - in 36 of 40 the rear foot carries the more forward CoP.
  - The stagger is not a pelvis twist: the sum of the two feet's yaws has sd 0.14°.
  - It goes with a hip-pitch difference (sd 0.85°, corr +0.97), a smaller ankle difference (0.048° per mm of stagger), and a sole-pitch difference of +0.0345° per mm, with the forward foot toe-up (corr with the CoP split -0.99).
  - The passive MTP d3 difference has sd 0.30° (corr -0.98).
  - All 14 K2b-outside episodes have |stagger| >= 10.3 mm (mean 17.5, against 5.3 for the in-range episodes), and in all 14 the rear foot is the toe-loaded one. The 10 K2a episodes have mean |stagger| 17.9 mm (min 10.3).
- **The stagger is left over from the spawn.**
  - At step 0 the stagger sd is 42 mm in both Run 2 and the statue.
  - The statue removes it: 8.7 mm at step 200, 0.5 mm at 999.
  - Run 2 keeps part of it: 15.9 mm at 200, 10.6 mm at 999.
  - Run 3: 5.4 and 3.6 mm. Run 1: 23.2 and 22.2 mm.
- **Comparison across runs:**

  | | Stagger sd | CoP split sd | CoP split per mm | Ankle diff per mm | Sole-pitch diff per mm | Mean signed CoP r / l |
  |---|---|---|---|---|---|---|
  | Statue | 2.5 mm | 0.16 | -0.064 | 0.066° | +0.0215° | +0.054 / +0.061 |
  | Run 3 | 3.8 mm | 0.34 | -0.077 | 0.061° | +0.025° | -0.20 / -0.15 |
  | Run 2 | 11.9 mm | 1.05 | -0.086 | 0.048° | +0.0345° | +0.003 / +0.056 |
  | Run 1 | 21.2 mm | 0.15 | +0.005 | 0.098° | -0.0025° | -0.24 / -0.29 |

  - Run 3 has 11 of 38 opposite-sign episodes.
  - Run 1 has the largest stagger but almost no split: its ankle difference tracks the stagger at about 1.5 times the statue's rate and keeps the soles parallel. Both of its feet are heel-side.
- **Inferred, not tested:** Run 2's split is the spawn stagger that the tight leg term lets persist, combined with less ankle compensation than even the statue's. The legs are held within about 0.005 rad of a single settled pose, so the forward foot tips toe-up and the rear foot toe-down. The both-feet CoP stays centred (x_c median 63.8 mm, statue 64.0, from the judge).

## 4. exp(log_std) per actuator (`tables/log_std.csv`; read from `model.policy.log_std` and equal to the zip's policy.pth)

| Actuator | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| hip pitch | 0.0444 | 0.0383 | 0.0368 |
| hip roll | 0.1109 | 0.1053 | 0.1061 |
| knee | 0.0188 | 0.0168 | 0.0168 |
| ankle | 0.0233 | 0.0224 | 0.0236 |
| tail_3_pitch | 0.1762 | 0.1532 | 0.1732 |

- Left equals right to 4 decimals in every run.
- The neck and head are 0.136-0.164, and the other tail actuators 0.171-0.180.
- Hip pitch, knee and ankle are all below the 0.08 floor that criteria.txt's recipe 2 would add, in all three runs.
