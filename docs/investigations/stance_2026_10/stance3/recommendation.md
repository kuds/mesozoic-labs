# T. rex stance: three screen runs for 2026-10-18

Offline diagnosis of 2026-10-18 (no training) and the recommended three runs. Report only; every physics, reward, task, gate or policy-interface change still needs your approval, and screen runs certify nothing.

## What the offline work found

In plain terms: the T. rex learns to stand. Then, a few million steps in, it slowly leans forward onto its toes by closing its ankles, and sometimes it leans back again. Tonight's offline work found five things that point to tomorrow's runs. It was offline only: no training, just rolls of existing checkpoints, the zero-action statue and scripted poses.

1. The committed stance reward pays for a small forward lean. Its leg-pose term is centred on the authored keyframe, which the servos cannot hold under the animal's weight, so moving closer to the keyframe means leaning forward. The reward's best posture sits about 23 mm ahead of the statue, and leaning all the way to the gate's toe bar costs almost nothing (T/reward_landscape/out/derived.txt). Arm L's 'settled' leg target removes that payment; it is one of #624's options, off by default.

2. Under arm L the drift is not paid for, but it is barely resisted. A ±14 mm lean costs at most 0.013 per step, about 0.3% of the reward, and only the leg-pose term pushes back (derived.txt). Seed 60's drift went against its own reward, scoring 114 per episode below the statue at 5M (T/reward_landscape/out/ckpt_tables.txt).

3. The lean lives in the ankles. Each degree of ankle closure moves the body about 10.7 mm forward, so the gate's toe bar is only about 4 degrees away (T/onset/out/xstats.json). The leg-pose term watches the ankles, and its value tracked the lean almost exactly (r = -0.95; T/onset/out/timeline.txt). But at its current width of 0.10 rad, 4 degrees is cheap.

4. Other suspects look weaker:
- Exploration noise falls smoothly and does not time the drift (T/noise_robustness/work/training_noise_bins.txt).
- Checkpoints picked by reward always come from the first 1M steps, and they are fragile under their own noise (T/onset/out/selection.txt).
- Among physics changes, only a stiffer ankle spring set at the settled angle helped, and only when rolled with policies already trained on the old plant. Pulling the toe edge back made the statue fall, and the other foot changes did nothing for the lean (T/mechanics/out/tables.txt).

5. The gate's fore-aft bar reads only the foot pad and ignores direction. A signed whole-foot measure per foot, [-5, +100] mm ahead of the MTP III joint, separates the drifted stances from the good ones more cleanly (T/cop_statistic/out/analysis2.txt).

So tomorrow, all three runs are config-only on main's r9 plant:
- arm L for the full 12M, as the control and as a candidate to commit;
- arm L with the leg-pose term made twice as heavy and twice as narrow (L+T), on two seeds that drifted in opposite directions before.

The signed per-foot measure is added to the judging. Every kept checkpoint is judged by fixed rules that were written before training. None of this certifies anything; it decides what to commit and certify next.

## Recommended runs

PLAIN SUMMARY. Run three full 12M stance trainings tomorrow. All three are config-only, all on main's physics r9, and each runs on a never-merged screen branch built on the stance-balance-telemetry branch (04ced045). Each run changes one thing against a named baseline:
- Run 1, 'armL_s60': arm L exactly as #624 ships it (off by default on main), seed 60. It is the control and the zero-new-code commit candidate. Its baseline is the committed stance.toml with #624's switches on, nothing else.
- Run 2, 'armLT_s60': arm L with the settled leg-pose term made twice as heavy and twice as narrow ('L+T': leg_home_pose_weight 0.5 -> 1.0, leg_home_pose_tolerance 0.10 -> 0.05 rad), seed 60. Its baseline is Run 1.
- Run 3, 'armLT_s52': the same config as Run 2 on seed 52. It is the within-recipe replicate.

WHY THIS SHAPE.
- All three judges included an arm L 12M control. Two of the three proposed exactly this hybrid.
- Run 1 answers the question nobody has answered yet: does the commit candidate hold from 6M to 12M through the notebook path?
- Runs 2-3 test the one config-only lever the offline work ties to the failure: a stronger joint-space restoring price on the ankles and hips that carry the lean.
- Two seeds under one digest is also the shape certification needs.
- Seeds 60 and 52 drifted in opposite directions under scratch arm L: s60 +52 mm forward at 5M, s52 -44 mm heel-ward at 4M (T/onset/out/tables.txt). Together they test whether the price works both ways.
- Run 1 and Run 2 share seed 60, so they share initial weights and the reset stream. It is a same-seed contrast, not a replay.

RUN SETUP.
- Branches: Run 1 on claude/screen-trex-r9-arml-12m; Runs 2-3 on claude/screen-trex-r9-armlt.
- Each run: one Colab runtime, launched in parallel, BEHAVIOR "stance", N_ENVS 4, 12,000,000 steps (final 12,001,280).
- Screen cadence: one evaluation of 10 episodes every 131,072 steps; a checkpoint every 524,288 steps, all kept.
- Expected wall time is about 12 h per run on a CPU runtime: the r10 arm L screen's 12M runs on Colab CPU ran 17:47 to 05:24-05:48, from the investigation timeline. The telemetry overhead is measured in the smoke run.
- Mechanics, the leg exploration-noise floor and a new fore-aft reward term are held in reserve for the next round (see the decision table).

Paths: T = scratchpad/foot/tonight, SP = scratchpad.

### Run 1 - armL_s60: arm L on r9 through the notebook to the full 12M (control and config-only commit candidate)

**Change:** One change against the committed stance stage (stance.toml at the telemetry branch 04ced045, the same as main's for these keys): #624's arm L recipe, switched on exactly as shipped. This is three [env] options (the lateral lean term and the settled leg_home_pose target) and four [ppo] keys (mirror-symmetric PPO). The collapse reference moves with the statue reward (3756.8 -> 3981.4). Screen-only cadence and retention keys are added; they enter no task, gate or hyperparameter digest. Everything else is unchanged: reward weights, PPO schedules, budget, gate, plant and reset noise. Against the scratch r9 arm L screen it is the same recipe; the only differences are the production notebook path and 12M instead of 6.29M.

**Why:**

1. Arm L is the only family with certificate passes. Offline six-heading certificates at 6.29M: s60 40/40, s52 38/40, s42 29/40 (SP/foot/next_signals/howclose/offline_certificates.txt).
2. The committed keyframe leg target pays for a forward 'level lean'. Its static optimum is about +23 mm of COM, paying +26 per episode over the statue. Leaning to pad 0.75 costs only 0.008 per step. The settled target moves the optimum to within 2 mm of the statue (T/reward_landscape/out/derived.txt). So the control must be arm L, not production. Production on r9 was clean 1/40 (run A); heading-free controls were clean 0/2/1 of 40 at 3M (history findings, docs/investigations/stance_2026_10/run_a and cap_screens).
3. Arm L has never been trained through the notebook or past 6.29M on r9, so its 6-12M behaviour is unknown. On r10, s42's over-bar fraction rose from 0.035 to 0.169 between 6M and 12M (records-branch investigation section 6.4-6.7).
4. Prior, an inference: it may well fail something. Scratch arm L failed K2 on 2 of 3 seeds (s42 9/8/11, s60 8/39/0 over the pad bar at 4/5/6M). It failed K3 on s60 (6/20 joint-noise falls at 6M) (SP/fix/records_1018b/docs/investigations/stance_2026_10/r10_screen/score.txt).
5. Without this run, Runs 2-3 have no concurrent baseline, and a pass would be confounded with the harness change.

**Exact settings:**

Branch claude/screen-trex-r9-arml-12m = claude/stance-balance-telemetry 04ced045ba8990e4ad108c597536101ce1a89f3f, plus a cherry-pick of 412f19aec200f0366c09f50803d64079a15796be (the cadence keys), plus one config commit. Never merged.

configs/tyrannosaurus_rex/anatomical/stance.toml:
- [env], added: lateral_lean_weight = 0.25; lateral_lean_tolerance_m = 0.015; leg_home_pose_reference = "settled".
- [env], unchanged: leg_home_pose_weight 0.5; leg_home_pose_tolerance 0.10; broad fraction 0.25; broad scale 6.0; reset_noise_scale 0.05.
- [ppo], added: symmetry = "mirror_kl"; symmetry_kappa = 0.3; symmetry_sigma_floor = 0.01; symmetry_mirror_critic = true.
- [ppo], unchanged: learning_rate 3e-5 -> 1e-5 linear; n_steps 2048; batch_size 128; n_epochs 10; gamma 0.98; gae_lambda 0.95; clip_range 0.2; ent_coef 0.005 -> 0.0 over 7,000,000; target_kl 0.03; net_arch [512, 256]; log_std_init -1.5.
- [curriculum], added: eval_freq = 131072; eval_episodes = 10; checkpoint_freq = 524288.
- [curriculum], changed: max_checkpoints 5 -> 0; collapse_peak_floor_reference 3756.8 -> 3981.4. 3981.4 is production plus the measured arm L statue gain of 224.6: 3980.37 against 3755.72, 39/40 standing, in T/refine_plan/out/statue_ref.json; the r9 screen used 3981.45.
- [curriculum], unchanged: timesteps 12000000; the gate stance_quality_heading/v1 and every bar; min_avg_reward 2250; statue_constants_physics_revision 9; checkpoint_archive_every_timesteps 500000 (moot with keep-all).

Notebook:
- REPO_REF = the branch, or its pushed head SHA.
- SPECIES "tyrannosaurus_rex"; EMBODIMENT "anatomical"; ALGORITHM "ppo"; N_ENVS 4; SEED 60.
- QUICK_TEST False; USE_GOOGLE_DRIVE True; AUTO_DISCONNECT True.
- BEHAVIOR "stance"; TRUNK_FROM ""; RETRAIN_FROM "".
- RUN_LABEL "stance screen armL s60"; RUN_ID "stance3_armL_s60".

Preflight: the zero-action baseline row reads about 3980. About 3757 means arm L is off: stop.

**Code needed:** None in the environment, reward or trainer.
- Cherry-pick 412f19ae (about 42 lines, already used by the r10 screen and the walking pilots). It has a one-hunk conflict in environments/shared/curriculum/gate_schema.py, resolved by hand as in the build spec.
- Add a small test file for the cadence keys (412f19ae shipped none).
- Make the config commit.
- Regenerate configs/digest_snapshot.generated.txt.
- Re-measure the statue gain.
- Run a notebook smoke run on CPU.
- The scratch judge and scorer are adapted from the r10 screen's.

**Category:** refinement (config-only recipe on; the control and commit candidate)

### Run 2 - armLT_s60: arm L with a stronger settled leg target (weight 1.0, tolerance 0.05 rad)

**Change:** One conceptual change against Run 1, set through two existing [env] keys: leg_home_pose_weight 0.5 -> 1.0 and leg_home_pose_tolerance 0.10 -> 0.05 rad. The broad component keeps fraction 0.25 and scale 6, so its width becomes 0.30 rad. The collapse reference moves with the statue reward (3981.4 -> 4460.5). Nothing else differs from Run 1: same branch base, [ppo], cadence, gate, plant and seed.

**Why:**

1. Under arm L the drift is not paid for; it climbs a weak restoring slope.
- Only the settled leg target restores: -0.051 per step per unit of whole-foot CoP.
- A lean of +/-14 mm of COM costs at most 0.013 per step, about 0.3% of the 4.08 per-step reward (T/reward_landscape/out/derived.txt, held_tables.txt).
- s60 at 5M scored 114 per episode below the arm L statue (T/reward_landscape/out/ckpt_tables.txt).

2. The lean lives in the joints this term governs.
- The ankle term explains the COM shift: slope 1.00, r 0.92 (arm L alone r 0.99), about 10.7 mm per degree. The pad bar is therefore about 4 degrees of ankle away (T/onset/out/xstats.json).
- In-training leg_home_pose fell as signed CoP rose: r = -0.95 (T/onset/out/timeline.txt).

3. L+T is strongest where drift starts and the gate still passes. Per-ankle restoring slope, arithmetic on the term's form in environments/shared/reward_functions.py:401-440 (an inference about gradients, not a training result):
- 7.3x arm L's at 1 deg, 5.6x at 2 deg, 3.6x at 3 deg, 1.9x at 4 deg, about 0.9x at 5 deg and 0.54x at 5.8 deg.
- So it is a strong centring price inside the gate's range and a weaker backstop past about 5 deg; this is a stated risk.

4. Measured cost against the statue per 800-step window, deterministic, 8 spawns (T/refine_plan/out/leg_tax.jsonl):
- s60 3M: -190.6 under L+T against -39.2 under arm L.
- s60 6M: -118.8 against -22.2.

5. Other properties.
- It is two-sided: it also prices s52's heel-ward ankle opening.
- Neck or head counterweights cannot pay it, unlike arm C's COM term, whose runs saturated actuators on 16-20 of 20 episodes (T/onset/out/tables.txt).
- A zero action holds it, so it adds no saturation pressure (inference).
- The policy can sense what it prices: leg joint angles decode the COM's fore-aft position with R2 0.987 (T/cop_statistic/out/analysis2.txt).
- Its physics analogue, a stiffer ankle spring at the settled angle, moved a drifted stance 24 mm back, with a dose-response (T/mechanics/out/tables.txt). L+T applies the same idea as a reward, without a plant revision.

6. The old objections do not apply.
- Tolerance 0.05 was 'past the knee' only against the keyframe reference (stance.toml:81-90). Under the settled reference the statue keeps 975.7 of 1000 on the term and stands 39/40 (T/refine_plan/out/statue_ref.json).
- #491's weight-1.5 failure was about the tail and neck pose, which this term does not govern (stance.toml:69-80).

**Exact settings:**

Branch claude/screen-trex-r9-armlt = Run 1's branch plus one commit. Never merged.

configs/tyrannosaurus_rex/anatomical/stance.toml:
- [env] leg_home_pose_weight = 1.0 (telemetry-branch line 68; was 0.5).
- [env] leg_home_pose_tolerance = 0.05 (line 81; was 0.10).
- leg_home_pose_broad_fraction 0.25 and leg_home_pose_broad_scale 6.0 unchanged.
- Arm L's three [env] keys and four [ppo] keys as in Run 1.
- [curriculum] collapse_peak_floor_reference = 4460.5, which is 3756.8 + 703.7. That gain is measured: statue 4459.40 +/- 34.6 against 3755.72, 39/40, in T/refine_plan/out/statue_ref.json; the build re-measures it. The collapse floor becomes 0.45 x 4460.5 = 2007.2.
- Cadence keys and everything else as Run 1.

The [ppo] table is identical to Run 1's, so its hyperparameter digest should equal Run 1's. Its task digest differs.

Notebook: as Run 1, with REPO_REF = claude/screen-trex-r9-armlt, SEED 60, RUN_LABEL "stance screen armLT s60", RUN_ID "stance3_armLT_s60". Preflight: the baseline row reads about 4459. About 3980 or 3757 means L+T is off: stop.

**Code needed:** None. One config commit, a digest snapshot regeneration and a statue re-measurement, on a second clone.

**Category:** reward setting (refinement of arm L)

### Run 3 - armLT_s52: the L+T replicate

**Change:** Identical to Run 2: same branch, same byte-identical config, so the task, hyperparameter, gate and plant digests are identical. Only the training seed differs (52).

**Why:**

1. Per-seed outcomes swing.
- Scratch arm L s60 was clean on 40, 31, 1 and 40 of 40 at 3, 4, 5 and 6M (T/onset; investigation section 5.4).
- s42 failed on r9 and passed on r10, while s52 and s60 did the reverse (r10_screen/score.txt).
- One seed cannot separate a recipe effect from seed luck. 2/2 still has a 95% CI of [0.16, 1.0] for the per-seed rate.

2. Seed 52 drifted heel-ward under scratch arm L: -29 to -44 mm at 3-6M, with the ankle 2.4-3.5 deg open (T/onset/out/tables.txt). Seed 60 drifted toe-ward. Together the pair tests whether a two-sided joint-space price centres the stance both ways. Seed labels carry only the initial weights and environment streams, not outcome (inference).

3. Two runs under one digest is the shape certification_seeds = 2 needs (stance.toml:902). Whether screen runs could ever count is a maintainer ruling, not assumed here.

4. Frozen K3 control sigma vectors exist for seed 52, so its noise tests are comparable (SP/foot/symmetry/design/frozen_params.json, sha256 58b2c8fb...).

**Exact settings:**

As Run 2 (branch claude/screen-trex-r9-armlt, same stance.toml), with SEED 52, RUN_LABEL "stance screen armLT s52" and RUN_ID "stance3_armLT_s52". Preflight baseline about 4459.

**Code needed:** None beyond Run 2's branch.

**Category:** reward setting (replicate of Run 2)

## Fixed criteria (written before training)

```
T. REX STANCE, THREE SCREEN RUNS OF 2026-10-19: FIXED CRITERIA

Write this to SP/foot/stance3/criteria.txt before any training step. Record its sha256 in both config commit messages and in colab_steps.txt. Do not edit it after the first training step.

WHAT IS TRAINED
All three runs share these settings:
- physics r9; the stance stage; the SB3 notebook; n_envs 4;
- 12,000,000 steps (final 12,001,280);
- an evaluation of 10 episodes every 131,072 steps;
- a checkpoint every 524,288 steps, all kept;
- the committed PPO schedules.

The runs:
- Run 1 (stance3_armL_s60, branch claude/screen-trex-r9-arml-12m): arm L as #624 ships it (lateral_lean_weight 0.25, lateral_lean_tolerance_m 0.015, leg_home_pose_reference "settled"; mirror_kl kappa 0.3, sigma floor 0.01, mirrored critic). Seed 60.
- Run 2 (stance3_armLT_s60, branch claude/screen-trex-r9-armlt): Run 1 plus leg_home_pose_weight 1.0 (was 0.5) and leg_home_pose_tolerance 0.05 rad (was 0.10). Seed 60.
- Run 3 (stance3_armLT_s52): Run 2's branch and config. Seed 52.

HOW IT IS JUDGED
The judge is judge_stance3.py: the r10 screen's judge, ported to r9 and to each branch's clone, with all binding checks on (plant, task, symmetry record, commit).
- Deterministic, heading 0. The 40-episode panel on seeds 3042-3081 runs under the branch's own stance [env] and r9 bars, scored by stance_quality/v2.
- The zero-action statue runs on the same seeds under the same env.
- Noise tests: spawns 3042-3061. Joint-space noise uses frozen_params.json control_sigma[seed] (sha256 58b2c8fb...); the own-sigma test uses the checkpoint's own exp(log_std). A fall is an episode shorter than 1000 steps.
- Judged checkpoints J: 4,194,304; 5,242,880; 6,291,456; 8,388,608; 10,485,760; and final (stage1_final, 12,001,280).
- A missing or invalid judge file fails every criterion that needs it.

CRITERIA, PER RUN

K1 gate held:
- v2 PASS at 6,291,456 and at final. PASS means clean >= 37/40 with exact Clopper-Pearson LCB >= 0.80, full horizon >= 38/40, hop-or-fall <= 1, and the repository verdict PASS.
- Across J, at most one checkpoint below 37/40 clean and none below 33/40.

K2 fore-aft held, at every checkpoint in J:
- K2a, pad rail: at most 3 of 40 episodes with max_sole_cop_fore_aft > 0.80. A non-full-horizon episode counts as over.
- K2b, signed per-foot whole-foot CoP: at most 3 of 40 episodes in which either foot's <label>_foot_cop_fore_aft lies outside [-0.929, +0.571] span half-lengths. The statistic is the mean over loaded steps in the window, steps 200-999. On r9 the range is [-5, +100] mm ahead of that foot's MTP III, with x = 60 + 70 w mm. A non-full-horizon episode counts as outside.
- K2c, no drift: the panel median of the both-feet whole-foot CoP (which equals the COM projection to within 0.5 mm) lies within [-60, +30] mm of the statue's median on the same seeds, and its range over the six checkpoints is at most 30 mm.

K3 robustness, at 6,291,456 and at final: joint-space noise falls <= 2 of 20, and own-sigma falls <= 1 of 20.

K4 roll, at 6,291,456 and at final: at most 3 of 40 full-horizon episodes with max_sole_tilt_deg > 2.0 or max_sole_corner_lift_m > 0.004.

Run verdicts:
- HOLDS: K1-K4 all hold.
- FORE-AFT HOLDS: K1 and K2 hold.

WHERE SCRATCH R9 ARM L STOOD
Reference only: a scratch harness, 6.29M, the same tests where they exist.
- K1 at 6M: clean 29/38/40, so s42 fails.
- K2a at 4M/5M/6M: s42 9/8/11, s52 3/1/2, s60 8/39/0.
- K2b: s60 at 5M had 40 episodes outside; s60 at 6M had 1 (seed 3080); s52 at 6M had 1 (seed 3050).
- K2c: s60 at +26/+52/+3 mm fails; s52 at -44/-37/-38 mm passes; s42 at +11/+3/+18 mm passes.
- K3: joint-noise falls 1/1/6 of 20, so s60 fails; own-sigma 0.
- K4: 2/0/0.
Sources: r10_screen/score.txt, T/cop_statistic/out/analysis2.txt, T/onset/out/tables.txt.

RECIPE VERDICTS AND WHAT FOLLOWS (decided now)
1. L+T HOLDS on both Run 2 and Run 3: propose committing L+T to the maintainer (arm L's switches plus the two leg keys, collapse reference 4460.5), then certify it on 2 seeds under the committed config.
2. L+T FORE-AFT HOLDS on both but fails K3 or K4: the drift is fixed. The next round adds the leg exploration-noise floor to L+T (new PPO-side code: log_std >= ln 0.08 on the hip-pitch, knee and ankle actuators).
3. L+T fails K2 on either seed:
   a. If Run 1 HOLDS, arm L (config-only) stays the candidate. Replicate it on 2 more seeds before committing.
   b. If Run 1 also fails K2, a joint-space price does not hold the stance. The next round tests either a fore-aft term that needs code (the ankle anchor, w 0.4, tolerance 0.10 rad; or the COM band penalty with a neck guard) or the v6b ankle spring.
4. L+T fails K1 through new failure modes (saturation, hop, nosedive, head contact) while K2 holds: the dose is too strong. Next: weight 0.75 at tolerance 0.05 (statue 4215.5, measured in T/refine_plan/out/statue_ref.json).
5. Exactly one L+T seed HOLDS: the recipe is not separated from seed luck. Replicate before any commit.

In every case, the best final checkpoint of the three runs is offered as the stance parent for the parked walking pilot 2: most clean panel episodes, with ties going to fewer joint-space plus own-sigma falls.

Fails stay fails. Runs are never stopped or changed on their readouts. Early reads at 1M and 2M are report-only.

PRE-REGISTERED READOUTS (report only, never in a verdict)
- K5 trajectory, from eval_episodes.csv: after 3,145,728 steps, any 4 consecutive evaluations whose full-horizon mean com_fore_aft_mean_m lies outside [statue - 0.060, statue + 0.030] m. The statue reads about -0.016 m in that foot-site frame.
- The late-fall fraction (200 < length < 1000) per 262,144-step bin, from train_episodes.csv. Flag anything above 5%. Arm L s60 peaked at 15.8% (T/noise_robustness/work/falls_by_phase.txt).
- A 20-episode quick panel (3042-3061) at every 524,288-step checkpoint; 3,145,728 judged like J.
- Common noise: sigma = exp(log_std) of scratch arm L s60 robust_best (legs about 0.152), 40 seeds, falls only. Reference: arm L s60 at 6M 38/40.
- Forward-start tail panel: the 32 seeds of T/noise_robustness/work/tail_seeds.txt. References: statue 0, arm L s60 6M 3, s52 6M 7, s42 6M 26.
- At final: the fresh panel 7042-7081 and the six-heading certificate.
- Per-term reward decomposition; ankle, knee and hip DC; neck and tail posture; per-actuator saturation; leg log_std.
- Selection comparison, each candidate on the 40-seed panel and own-noise falls:
  - S0: robust_best.
  - S1: best.
  - S2: final.
  - S3, late-stable: the latest periodic checkpoint at or after 3,145,728 steps whose preceding 8 evaluations were all 10/10 full horizon, with pad and whole-foot over-bar fractions of 0.
```

## Alternative layout

Alternative layout: three different levers on one seed, the incentive design. Run 1 arm L s60 and Run 2 L+T s60 stay as recommended. Run 3 becomes arm L plus a new opt-in ankle-anchor penalty on seed 60: -0.4 x mean over the ankles of (1 - exp(-(e/0.10 rad)^2)), centred on the settled ankle angle 1.3713665 rad, paid after step 200. That is about 40 lines plus tests, and it moves the task digest. Unlike L+T, it keeps its restoring slope across 3-6 degrees: +0.166 per step from 2.87 to 5.77 degrees (T/design_incentive/costs.txt). If the maintainer specifically wants 'slightly different mechanics', Run 3 can instead be arm L on a screen-only plant with the v6b passive ankle spring: stiffness 250 N.m/rad, damping 90, springref 78.6 deg, physics revision 11 on the branch. Its statue was already measured: 40/40 clean, reward 3982.7 against 3980.4, settled pose within 0.083 deg (T/design_mech/out/statue_variants.json), so no constant has to move. This layout spreads the slots over three distinct mechanisms and tells more if everything fails. The costs: there is no replicate, so every contrast is n=1 and only large effects show. New reward code or a plant revision lands on tonight's critical path, and the build grows from about 4 h to about 6-7 h before launch. A win still needs a code PR or a physics revision before any certification, and the spring would strand every r9 checkpoint, including walking pilot 2's parent. The spring's evidence is also off-distribution only, and a milder cousin of it (capped sagittal authority) crept and failed at 6M (cap_screens/screen_6m_authority_analysis.txt).

## Concerns

- Nothing here can be certified. The configs are not committed, so the official judge refuses these runs on task fingerprint. The notebook's end-of-stage certificate judges robust_best, which will again come from under 1M steps: it did in 6 of 6 earlier runs, and those checkpoints were clean on 0 of 40 under their own action noise (T/onset/out/selection.txt; SP/foot/battery/own_sigma/own_sigma_report.txt). Treat that certificate as report-only.
- Sample size is small. L+T gets two seeds and arm L one, so only large effects can be read. Even 2/2 passes leave a 95% CI of [0.16, 1.0] on the per-seed pass rate. If exactly one L+T seed holds, the recipe cannot be told apart from seed luck.
- The L+T dose has not been screened. By arithmetic on the reward form (reward_functions.py:401-440), its per-ankle restoring slope is 2-7 times arm L's within 4 degrees of the statue. Beyond about 5 degrees it is only about 0.5-0.9 times arm L's. So it is a strong centring price but a weaker backstop once a seed has already leaned past about 5 degrees.
- L+T also charges centred, passing stances: s60 at 6M pays -119 per window against -22 under arm L (T/refine_plan/out/leg_tax.jsonl). Most of that is the policies' 2-3 degree hip-pitch nose-up margin, so watch for nosedive and head-contact terminations.
- L+T charges the policy's own leg noise about 3.7 times more than arm L does (T/refine_plan/out/leg_tax.jsonl). Leg exploration noise could therefore shrink faster, which would hurt K3.
- The #491 precedent: weight 1.5 on an older plant took about 2M extra steps to escape an early collapse. The collapse floor here is 2007.2.
- Run 1 will probably fail something. Scratch arm L failed K2 on 2 of 3 seeds and K3 on s60, with 6/20 joint-noise falls at 6M (r10_screen/score.txt). That is fine for a control, but a pass by arm L alone would be a surprise, not the expectation.
- These are the first arm-L-family runs through the production notebook on r9. Earlier evidence came from a scratch harness that stopped at 6.29M. The two setups have been shown equivalent as environments (statue 3981 against 3980), not as training runs.
- Runtime type. The mirror-symmetric trainer has trained 12M only on Colab CPU runtimes (the r10 screen; its colab_steps.txt says 'CPU is enough'). It has never run on CUDA. Pilot 1's L4 runs used stock PPO. If L4 is chosen, the first 'Eval num_timesteps=131072' line shows that the update path runs on the GPU. Either way, all three runs should use the same runtime type, because CPU and GPU numerics differ.
- The telemetry branch is an open PR (#630), and the screens pin it at 04ced045. This is the first time its ledgers and policy probe run alongside MirrorKLPPO and the cadence keys; the CPU smoke runs check that combination.
- Expected test failures on the screen branches. These come from pins that no stage enables symmetry, that the collapse reference is 3756.8, that the reward rails are 0.60 x the reference, and that the arm L options are off. They are listed in the build spec. Any other failure blocks launch.
- K2b (the signed per-foot whole-foot bar, [-5, +100] mm ahead of MTP III) and K2c (no drift) are criteria for this screen only. Putting either into the gate is a gate change: it moves gate_sha256 and needs approval. On the r9 certificates so far it would flip no verdict (T/cop_statistic/out/analysis2.txt).
- The cause of the drift's onset is still unexplained, as is why s60 reversed at about 6M. L+T tests whether a stronger restoring price suppresses the drift, not why it starts. A posture at 12M may still not be an equilibrium.
- No mechanics run is included. The one physics candidate that moved a drifted stance back was the v6b ankle spring, and that evidence comes from already-trained policies only. It is held in reserve because it needs a plant revision, which would strand every r9 checkpoint.
- Records. Two of tonight's records treat reward changes as out of scope: SP/foot/next_signals/recommendation.md section 3 and SP/foot/battery/battery_report.md section 7. Tomorrow's decision supersedes the 2026-10-16 rule ('no more stance-only reward experiments unless walking points back to stance'); pilot 1 read STILL, so the rule's own condition was never met. Record the decision, for example as D-D37 beside D-D36 in docs/BEHAVIOR_RECIPES_PLAN.md.
- Resources. Three runtimes for about 12 h each, plus about 30 min each for the end-of-stage certificate and videos. About 100-150 MB of Drive per run (23 checkpoints of about 4 MB each, plus telemetry). A disconnect costs up to about 35 min of retraining after a resume. Judging afterwards takes about 1 CPU-hour per run.
- GitHub. The two screen branches have to be pushed in the morning before Colab can clone them. Tonight's rules keep GitHub read-only, so the build stays local until the maintainer says go.

## Decisions for you

- Approve the three-run layout: arm L s60, L+T s60, L+T s52. Or choose the alternative, which replaces Run 3 with an ankle-anchor reward arm (new code) or a v6b ankle-spring mechanics arm (screen-only plant revision). Record the choice as a new decision beside D-D36 that supersedes the 2026-10-16 'no more stance-only reward experiments' rule.
- Approve training a reward-weight change, L+T (leg_home_pose_weight 1.0, leg_home_pose_tolerance 0.05), on a never-merged screen branch. Committing it to main later would be a reward and task change that needs separate approval and re-measured statue constants (collapse reference 4460.5; min_avg_reward rail).
- Choose the Colab runtime type. CPU is recommended: it is the path the mirror-symmetric trainer has already trained 12M on, at about 12 h per run. L4 is untested with this trainer. Whichever is chosen, use the same type for all three runs.
- Approve the pre-registered criteria as written: K1-K4, with K2b (the signed per-foot whole-foot bar) and K2c (no drift) as screen-only criteria, plus the decision table. Fails stay fails and runs are never stopped on their readouts.
- Rule now, before training, on whether screen runs could count toward certification_seeds = 2 if their config were later committed byte-identically. replication.py checks digests, not branches. The recommended ruling is no: certify with fresh runs under the committed config.
- Keep robust_best as the official handoff. The S0-S3 selection comparisons, including the late-stable rule, are report-only, and any change to selection needs approval.
- Approve pushing two never-merged screen branches to GitHub in the morning: claude/screen-trex-r9-arml-12m and claude/screen-trex-r9-armlt.
- Walking pilot 2 stays parked. Decide whether it restarts from the best final stance checkpoint of these runs once they are judged.

## Build spec

```
BUILD SPEC: THREE T. REX STANCE SCREEN RUNS
Tonight's build is local only. Nothing is pushed until the maintainer approves in the morning.

PATHS AND RULES
- SP = scratchpad.
- TEL = SP/fix/balance_telemetry: branch claude/stance-balance-telemetry, head 04ced045ba8990e4ad108c597536101ce1a89f3f.
- Work folder: SP/foot/stance3/, holding criteria.txt, colab_steps.txt, build.json, tools/, judge/, smoke/ and logs/.
- Never write under /home/user/mesozoic-labs. No git worktree. Make clones with git clone.
- Every Python process runs with OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, torch.set_num_threads(1), PYTHONPYCACHEPREFIX=SP/caches/pyc and MPLCONFIGDIR=SP/caches/mpl.
- Isolation: run through SP/foot/r10screen/tools/iso.py with ISO_WT=<clone>. It asserts that environments.__file__ resolves inside the clone.
- Run one simulation process at a time.

1. CLONES AND BRANCHES
1a. Run 1 branch:
- git clone --no-hardlinks TEL SP/fix/screen_s3_L
- git -C SP/fix/screen_s3_L checkout -b claude/screen-trex-r9-arml-12m 04ced045ba8990e4ad108c597536101ce1a89f3f
- git -C SP/fix/screen_s3_L fetch SP/fix/screen_r10L 412f19aec200f0366c09f50803d64079a15796be
- git -C SP/fix/screen_s3_L cherry-pick -x 412f19ae

1b. Resolve the one conflict, in environments/shared/curriculum/gate_schema.py (TEL line 325):
- Keep TEL's `_RETENTION_KEYS = frozenset({"max_checkpoints", "checkpoint_archive_every_timesteps"})`.
- Directly after it, add 412f19ae's `_CADENCE_KEYS = frozenset({"eval_freq", "eval_episodes", "checkpoint_freq"})` with its comment.
- In validate_gate_config's key union (TEL around line 554), add `| _CADENCE_KEYS` after `| _RETENTION_KEYS`.
- environments/shared/train_base.py applies with an offset and needs no hand edits. It brings the eval_episodes override in _eval_episodes_for_stage, _positive_cadence, _stage_cadence, and the `eval_freq, save_freq = _stage_cadence(stage_config, eval_freq, save_freq)` line at the top of _build_core_callbacks. The incentive design checked this with git apply --check (T/design_incentive/412f19ae.patch).

1c. Add environments/shared/tests/test_stage_cadence.py (about 40 lines). It checks that:
- a declared eval_freq or checkpoint_freq replaces the caller's value;
- absent keys keep the caller's 50,000 and 100,000;
- a declared eval_episodes is used exactly, even below min_eval_episodes;
- 0, -1, 1.5 and True are refused;
- validate_gate_config accepts the three keys;
- adding the keys moves only stage_config_view_sha256, never task_sha256, gate_sha256 or hyperparameters_sha256.

1d. Run 1 config commit, in configs/tyrannosaurus_rex/anatomical/stance.toml (line numbers are TEL's):
- Before [stage], add a header block: 'SCREEN, NEVER MERGED. Branch claude/screen-trex-r9-arml-12m: stance screen Run 1 (arm L as #624 ships it) on physics r9, 12M. Criteria SP/foot/stance3/criteria.txt, sha256 <hash>. recovery.toml inherits [env] and [ppo] here; only stance is trained.'
- [env], after the reset_noise_scale note and before [ppo] at line 372: lateral_lean_weight = 0.25, lateral_lean_tolerance_m = 0.015, leg_home_pose_reference = "settled".
- [ppo], after target_kl (line 444) and before [ppo.policy_kwargs] (line 446): symmetry = "mirror_kl", symmetry_kappa = 0.3, symmetry_sigma_floor = 0.01, symmetry_mirror_critic = true.
- [curriculum]: change max_checkpoints = 5 to 0 (line 914). Beside it add eval_freq = 131072, eval_episodes = 10 and checkpoint_freq = 524288.
- [curriculum]: change collapse_peak_floor_reference from 3756.8 to 3981.4 (line 985), with the note '3756.8 + arm L statue gain 224.6'.
- Leave everything else untouched, in particular timesteps 12000000 (line 514), the gate and every bar, min_avg_reward 2250 (line 799), checkpoint_archive_every_timesteps 500000 (line 921) and statue_constants_physics_revision 9 (line 1003).

1e. Runs 2-3 branch:
- git clone --no-hardlinks SP/fix/screen_s3_L SP/fix/screen_s3_LT
- git -C SP/fix/screen_s3_LT checkout -b claude/screen-trex-r9-armlt
- Make one commit:
  - leg_home_pose_weight 0.5 -> 1.0 (line 68);
  - leg_home_pose_tolerance 0.10 -> 0.05 (line 81);
  - collapse_peak_floor_reference 3981.4 -> 4460.5, i.e. 3756.8 + 703.7;
  - a header naming Runs 2-3.
- Leave leg_home_pose_broad_fraction 0.25 and leg_home_pose_broad_scale 6.0 unchanged.
- Both commit messages cite the criteria sha256.

2. STATUE REFERENCES (each branch, before its reference is committed)
- Adapt SP/foot/r10screen/v11/v11_pr_c.py, the V11 method: train_base.make_env, zero action, seeds 3042-3081, reset noise 0.05, mean over full-horizon episodes, minus production. Point it at the clone.
- Expected gains: +224.6 for Run 1 and +703.7 for L+T, each within ±1. Reference values: 3980.37 and 4459.40 against 3755.72, 39/40 standing, seed 3081 falls (T/refine_plan/out/statue_ref.json).
- If a gain differs by more than 1, stop and re-derive the reference.
- About 3 CPU-minutes each.

3. GENERATED FILES (each branch, after its config edit)
- python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --write, then the same with --check.
- Only the T. rex stance and recovery lines should move: task_sha256, hyperparameters_sha256.PPO, stage_config_view_sha256 and the reward-capture lines. List the diff in build.json. Any other line moving means stop.
- Cross-check: stance hyperparameters_sha256.PPO should read sha256:054ab48303167eda1bd66c8894a64ef516bedd041a9e12f57dbd20e75c026b9c on both branches. That is the r10 arm L screen's value, from the same [ppo] table (r10 commit 026eb833).
- Check-only, expected unchanged on r9:
  - python -m environments.shared.plant_contract --check
  - python -m environments.shared.species_catalog --check
  - python -m environments.shared.symmetry --check

4. TESTS (each clone, through iso.py --pytest)
- First record TEL's own baseline at 04ced045 for environments/shared/tests and environments/tyrannosaurus_rex/tests.
- Expected new failures, caused by the screen config (as on the r10 screen). Record each one with its reason in build.json:
  - test_mirror_kl_ppo.py::test_no_stage_config_enables_symmetry
  - test_stance_gate_config.py::test_the_reward_rails_are_the_same_fraction_of_the_statue
  - test_curriculum_manager.py, the assertion that collapse_peak_floor_reference == 3756.8 (around line 504)
  - In environments/tyrannosaurus_rex/tests/test_anatomical_lean_and_settled_legs.py:
    - test_the_options_default_to_off
    - test_off_the_options_change_no_reward_and_no_info_key
    - test_off_the_options_move_no_task_fingerprint_and_no_recorded_config
    - test_the_lateral_offset_flips_under_the_mirror
    - test_the_settled_reference_centres_leg_home_pose_on_the_settled_pose
- Any other new failure blocks launch.
- Run ruff check and ruff format --check on the changed Python files.

5. CPU SMOKE RUNS (one at a time)
Setup:
- Copy SP/foot/r10screen/tools/{iso.py,nb_runner.py,smoke_evidence.py,cadence_check.py} to SP/foot/stance3/tools.
- Set PARAMS: REPO_REF to the branch, SEED, BEHAVIOR "stance", QUICK_TEST True, USE_GOOGLE_DRIVE False, AUTO_DISCONNECT False, RUN_LABEL "stance3 smoke ...".
- Invocation: python3 iso.py nb_runner.py --log-base SP/foot/stance3/smoke/<L|LT> --out runner_<L|LT>.json --budget 262144 --checkpoint-freq 131072, with the clone as working directory.

The two smokes:
- Smoke L: clone L, SEED 60.
- Smoke LT: clone LT, SEED 52.
- Each runs the whole notebook through the end-of-stage certificate. That is about 45 min each: about 2 min for the baseline, about 12.5 min of training at about 350 steps/s (the r10 smoke, SP/foot/r10screen/build.json) and about 26 min for the certificate.

Assertions (extend smoke_evidence.py), on each run directory:
- stage_config.json:
  - reward_weights and task_fingerprint options carry the three arm L keys; on LT also leg_home_pose_weight 1.0 and leg_home_pose_tolerance 0.05.
  - hyperparameters carry the four symmetry keys.
  - curriculum carries eval_freq 131072, eval_episodes 10, max_checkpoints 0, collapse_peak_floor_reference 3981.4 (L) or 4460.5 (LT), and timesteps 12000000.
  - checkpoint_freq reads the runner's 131072. Confirm the committed 524288 separately with cadence_check.py through load_all_stages.
  - run.seed matches the smoke's SEED.
  - task_sha256 and hyperparameters_sha256 equal the regenerated snapshot.
  - plant physics_revision is 9 and physics_sha256 is sha256:3d68aec73fbc4642681ed601bedaf3524459d693d6a7afce036bb14bac872d67.
- zero_action_baseline.json mean is about 3980 (L) or about 4459 (LT).
- models/stage1_131072_steps.zip and stage1_262144_steps.zip are both kept, with their vecnormalize pickles. Each carries a mesozoic_symmetry record: mirror_kl, kappa 0.3, sigma floor 0.01, mirrored critic, and the committed mirror_map sha.
- evaluations.npz has timesteps [131072, 262144] with 10 episodes each.
- The telemetry outputs exist and are filled:
  - eval_episodes.csv: 20 rows, with com_fore_aft_mean_m and both <label>_foot_cop_fore_aft columns;
  - train_episodes.csv;
  - policy_probe.npz;
  - progress.json.
- Tensorboard has train/mirror_kl_loss and train/std.

Also:
- Record steps/s against the r10 smoke's 352 and update the wall-time estimate.
- Then run judge_stance3.py --smoke-episodes 4 on each smoke's stage1_262144_steps to prove that the binding checks pass against each branch.

6. CRITERIA
- Write SP/foot/stance3/criteria.txt with the text given in this recommendation.
- Compute its sha256 and put it in both config commit messages and in colab_steps.txt.
- No edits after the first training step.

7. JUDGE, SCORER AND TESTS (scratch, SP/foot/stance3/judge/)
This can be finished after launch: the first judged checkpoint, 4,194,304, lands about 4 h in.

judge_stance3.py, from SP/fix/records_1018b/docs/investigations/stance_2026_10/r10_screen/judge_r10.py:
- --clone and --expect-head per branch; --seed {52, 60}.
- --steps takes one of 3145728, 4194304, 5242880, 6291456, 8388608, 10485760, or final. Final loads models/stage1_final.zip and stage1_final_vecnorm.pkl.
- r9 rails: K2a is max_sole_cop_fore_aft, counted over all 40 episodes, with a non-full-horizon episode counting as over.
- Per episode, add both <label>_foot_cop_fore_aft (signed, mean over loaded steps in the window, from the pad telemetry) and the both-feet COM ahead of the mean MTP III. Port com_mtp3_head from T/cop_statistic/scripts/cop_roll.py. Record both for the policy panel and for the statue.
- The H4 noise tests stay unchanged (frozen_params.json, sha256 58b2c8fb...).
- Report-only modes:
  - --quick: 20 episodes, seeds 3042-3061.
  - --common-noise: sigma is exp(log_std) of SP/foot/symmetry/screen/runs/sym_lean_settled_s60/robust_best_model.zip, sha256 pinned; 40 seeds; falls only.
  - --tail: the seeds in T/noise_robustness/work/tail_seeds.txt, deterministic.
  - --fresh: seeds 7042-7081.

score_stance3.py, from score_r10.py:
- Scores K1-K4, FORE-AFT, the recipe verdicts and the decision-table line.
- Reads K5 from each run's eval_episodes.csv and train_episodes.csv.
- Produces the S0-S3 selection comparison.
- Refuses smoke files and relaxed-binding files. A missing checkpoint fails every criterion that needs it.

test_score_stance3.py, from test_score_r10.py, uses synthetic judge files at every edge:
- 3 against 4 episodes over a rail;
- K2b at -0.929 and +0.571;
- the K2c window and range;
- K1 durability at 33 and 37;
- missing files.

Validation, with task binding in report mode, on known checkpoints. The judge must reproduce:
- Statue: 39/40, falling on seed 3081.
- Scratch arm L s60 at 6M: clean 40/40, K2a 0, K2b 1 outside (seed 3080), joint-noise falls 6/20, own-sigma 0/20.
- s60 at 5M: clean 1/40, K2a 39, K2b 40 outside.
- s52 at 6M: K2b 1 outside (seed 3050), median COM about 38 mm behind the statue.
- Sources: r10_screen/score.txt, T/cop_statistic/out/analysis2.txt, T/onset/out/tables.txt.
- About 15 CPU-minutes.

Six-heading certificate of each final, report-only: SP/foot/next_signals/howclose/scripts/run_cert.py, pointed at the branch clone.

8. COLAB STEPS (SP/foot/stance3/colab_steps.txt; adapted from the r10 screen's and pilot 2's)
'T. rex stance, three screen runs.'
- Three runs, one Colab runtime each, all started at the same time.
- Runtime type: [CPU, recommended, or L4: the maintainer's choice], the same for all three.
- The runs:
  - Run 1, 'armL s60': branch claude/screen-trex-r9-arml-12m (commit <sha>), SEED 60, RUN_ID stance3_armL_s60.
  - Run 2, 'armLT s60': branch claude/screen-trex-r9-armlt (commit <sha>), SEED 60, RUN_ID stance3_armLT_s60.
  - Run 3, 'armLT s52': the same branch as Run 2, SEED 52, RUN_ID stance3_armLT_s52.
- The criteria were fixed before training: criteria.txt, sha256 <hash>.

FOR EACH RUN:
1. Open https://colab.research.google.com/github/kuds/mesozoic-labs/blob/<branch>/notebooks/sb3_training.ipynb. If that fails, open the same path on main: it is the same notebook, and REPO_REF picks the code. Use File > Save a copy in Drive, named for the run.
2. Runtime > Change runtime type: the chosen type.
3. In section 1, set REPO_REF = "<branch>".
4. In the section 2 configuration cell, set:
   SPECIES "tyrannosaurus_rex", EMBODIMENT "anatomical", ALGORITHM "ppo", N_ENVS 4, SEED 60 / 60 / 52, QUICK_TEST False, USE_GOOGLE_DRIVE True, AUTO_DISCONNECT True, BEHAVIOR "stance" (type it), TRUNK_FROM "", RETRAIN_FROM "", RUN_LABEL "stance screen armL s60" / "stance screen armLT s60" / "stance screen armLT s52", RUN_ID as above.
   Leave everything else as it is. RESUME_STAGE stays None.
5. Runtime > Run all, and allow Drive access.
6. Check the output for:
   - 'Repository ref: <branch>; commit: <sha>';
   - 'Run directory: .../ppo/<RUN_ID> (new run: RUN_ID names no existing run)';
   - 'Behavior 'stance' -> deliverable 'stance'; chain: stance';
   - the zero-action baseline row reading about 3980 for Run 1, or about 4459 for Runs 2-3. About 3757 means the screen config is not on: stop and tell me.
   - 'Timesteps: 12,000,000';
   - an 'Eval num_timesteps=...' line every 131,072 steps, the first within about 15 minutes. On L4 that first line also shows that the mirror-symmetric trainer runs on the GPU. An error mentioning cuda or device: stop and tell me.
7. Leave it running: about 12 h, then about 30 min of end-of-stage evaluation, certificate and videos. A PASS or a FAIL at the end is fine: it is report-only. Ignore 'never beaten the zero-action baseline'. Tell me if you see 'collapse' or an early stop.
8. Run section 9, Auto-Disconnect, by hand if it has not run.

IF COLAB DISCONNECTS:
9. Open the same copy on a fresh runtime with the same settings and the same RUN_ID. Set RESUME_STAGE = "stance" in section 5 and Run all. It continues from the newest 524,288-step checkpoint, so up to about 35 min is redone. If MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/<RUN_ID>/01_stance/models/ has no stage1_*_steps.zip yet, restart instead under RUN_ID '<RUN_ID>_b' with RESUME_STAGE None, and leave the first folder alone.

WHEN TO TELL ME:
10. Tell me when all three have started (with the commit and the baseline row), when each passes 6.3M steps, and when each finishes, stops or resumes. There is nothing to send: I read the checkpoints and telemetry from Drive.

9. MORNING, AFTER APPROVAL
- Push both branches (never merged, no PR).
- Write the head SHAs into colab_steps.txt and the criteria header.
- In the next records PR, record the decision (D-D37 beside D-D36 in docs/BEHAVIOR_RECIPES_PLAN.md) and the superseded 2026-10-16 rule.

TIME
- To launch-ready: about 4 h. Clones, config, statue and digests take about 1.5 h; tests about 1 h; the two smokes about 1.5 h of CPU.
- Judge, scorer and validation: about 3 h more, which can be done while the runs train.
```
