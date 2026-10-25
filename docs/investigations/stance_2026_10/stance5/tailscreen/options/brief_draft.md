# Tail saturation habit: decision brief (draft, 2026-10-24)

Report only. Nothing trained, nothing fetched from Drive, no criterion, config or run changed. The band clone `scratch_copband` is still clean at cf80ecb1.

Labels: **[measured]** means computed or read from a file (the source is named). **[inferred]** means reasoned from measured numbers. Short names for sources:
- SAT: `checkout/docs/investigations/stance_2026_10/stance4/satdiag/report.md`
- TAILEXP: `scratchpad/foot/stance5/tailexp/report.md` (its skeptic checks govern)
- K3DIAG: `scratchpad/foot/stance5/k3diag/report.md`
- SCORE: `scratchpad/foot/stance5/replication/scores/score_final.txt`
- CRIT: `scratchpad/foot/stance5/criteria.txt`
- MECH: `scratchpad2/tailround/mech/` (probe_plant, probe_tail12, probe_reward, deadzone; .py and .out)
- LAND: `scratchpad2/tailround/landscape/` (plant_probe, plant_probe2, clip_arith; .py and .out)

All CPU probes ran on a copy of the r9 MJCF and a git archive of cf80ecb1, without checkpoints.

## 1. Answer first

**The habit is a training artefact on a weak channel, not something the plant forces or the reward pays for.** Four parts, ranked by evidence:

1. **Persistence comes from the clipped Gaussian's dead zone** [measured code and arithmetic; its role inferred].
   - In SB3 2.9.0, the action is clipped only for `env.step`. The rollout buffer stores the unclipped sample and its log-prob (MECH sb3src `on_policy_algorithm.py` lines 199-254). Nothing in MirrorKLPPO's loss bounds the mean (`mirror_kl_ppo.py`).
   - Once the mean is past +1, only samples below +1 carry a reward slope. That share is 0.26-0.32 at mu 1.1 and 0.10-0.17 at mu 1.2, for sigma 0.153-0.21 (MECH deadzone.out). The gradient on the expected command is cut about 4x and becomes noisy, but it is not zero (LAND clip_arith.out).
   - Deterministic play uses clip(mu) = +1, which is exactly flat in the mean. The pinned states are also rarely visited in training: 3.7% of rows at stance5 s42 4.19M (TAILEXP lambda section).
2. **A feedback loop through the tail_3 joint-angle observation amplifies a small bias into the pin** [inferred from measured parts].
   - SAT's swap probe moves the mean by about 0.70 per 0.085 rad of observed angle. The plant gain is 0.0863 rad per action unit (MECH probe_plant.out).
   - That gives a loop gain of about 0.72 and an amplification of about 3.5x. The loop's time constant, about 75 control steps, fits the measured in-episode ramp: the command reaches 0.99 at a median of step 216 (SAT §2).
3. **Onset is trajectory-dependent drift of a low-stakes output** [inferred].
   - The habit appeared in 4 of 11 recent runs: stance3 Run 2 (s60), stance4 s42, stance5 s42 and stance5r s80 [measured: SAT, SCORE].
   - Seed 42 took it in 2 of 2 recipes. Two no-penalty continuations from one 4.19M state read frac99 0.92 and 0.17 at 5.24M (TAILEXP check 2) [measured].
   - The channel is worth about 0.03-0.10 per step out of about 4.5 [measured: MECH probe_reward].
4. **The plant enables the habit but does not cause it** [measured].
   - tail_3's spring (80 N.m/rad) is stiffer than its servo (kp 60). On the statue, q3 = -0.0924 + 0.0863a. A level distal tail (q3 = 0) needs a = +1.073, which is outside the range (MECH, LAND plant probes).
   - So any lasting tail-up preference lands on the rail. Yet the reward's target pose needs a = 0, and tail_2 can lift the tail tip as high with no channel past 0.9 (MECH probe_tail12.out).
5. **No reward incentive found** [measured, open-loop only].
   - The per-step reward falls monotonically with the tail_3 command. Deterministically it is -0.0971 per step at +1, which is -77.7 per 800-step window. That matches SAT's +73 to +86 gain from holding tail_3 at 0.
   - With sigma 0.16 noise on tail_3 the reward is still monotone (MECH probe_reward.out).
   - **Not tested:** the posture's value in closed loop under full 15-channel training noise. That is the one remaining route for a reward incentive (§4, item 1).

**Where the two research passes disagree.**
- LAND reads the pinned raw mean (+1.077 to +1.114) through the plant map as aiming for a level distal tail (q ≈ 0). This is weak evidence [inferred].
- MECH finds no term that pays for a level tail.
- Both predict the same practical outcome: **a fix that only bounds the command is expected to clear the rail and may leave the tail-up posture in place** [inferred]. stance3 Run 2's final did exactly that. It commanded +0.841 with q3 -0.019 and 0/40 flagged (SAT §5) [measured].

**What it cost the band arm** [inferred from SCORE, measured inputs].
- s80 failed only K1, and its K1 failure modes are all the saturation rail apart from one tilt episode. Without the habit, s80 would very likely have HELD, making H = 3 of 5.
- s42 would still fail K3. Its noise falls are hip-pitch noise, not the tail (K3DIAG: the tail claim holds in full).

**Anatomy** [measured geometry; plausibility inferred].
- The whole tail droops in this plant. tail_1 sits on its ventral stop in every policy and cannot leave it: it would need a = +1.45.
- The distal segment points -27° in the statue and -22.6° at a = +1. It would point -10.4° even at the unreachable joint stop (MECH probe_plant parts C-E).
- A tyrannosaur tail is usually reconstructed as carried roughly level [inferred, general literature]. So the tail-up drift is not anatomically wrong in direction. The plant simply cannot hold that posture except on the rail.

## 2. Options

Four candidates are worth a round. Every one needs the maintainer's explicit approval.

### A. Mean bound below the gate (P1 with bound 0.90) - **Recommended**

- **Class:** training-loss change. A new opt-in `[ppo]` key, off by default, hashed into hyperparameters_sha256. The checkpoint stays a stock MlpPolicy zip, r9 and r15 are unchanged, and every existing checkpoint still loads [inferred from TAILEXP "Exact change" and LAND].
- **What it changes:** it adds coef × mean over rows and 15 actuators of relu(|mu| - 0.90)² to MirrorKLPPO's loss through the existing `_mirror_loss` / `_mirror_end_update` hooks. This is P1's code, re-created from its exact description in TAILEXP, with the bound moved from 1.0 to 0.90 and coef kept at 41.
- **Evidence for:**
  - It targets the dead zone directly. Kept below the clip, the mean stays where the reward slope is alive.
  - P1 against its paired control: mean frac99 0.26 vs 0.86 over 8 checkpoints, and lower at 8 of 8 [measured, TAILEXP check 2].
  - No cost was measured: 0 falls, K3 0/20 at 18 checkpoints, other actuators inert [measured, TAILEXP check 3].
  - Coef 0 is bitwise identical to no bound [measured, TAILEXP].
- **Evidence against:**
  - P1 relapsed (0.488 at final). It did not keep the deterministic mean under its bound: beyond +1 on 21-46% of window steps at 7 of 9 checkpoints, while training rows were beyond +1 on only 2.5%.
  - Its effect against training alone is not established: a no-penalty draw read 0.171 [measured, TAILEXP checks 1-2].
  - It was one seed, one lambda and a 1M-step continuation. It tested persistence, never onset.
- **Why 0.90:**
  - P1's bound of 1.0 sits above the gate's 0.99 by construction.
  - Off the training distribution, P1's deterministic mean overshot its bound by up to 0.033 (window maximum 1.022-1.033) [measured, TAILEXP check 1]. The same overshoot at 0.90 leaves about 0.06 below 0.99 [inferred].
  - 0.90 is also where the saturation reward ramp starts, so the loss wall and the reward ramp meet [measured: stance.toml action_saturation threshold 0.9, MECH].
- **Risk:**
  - It may park the command at about 0.85-0.9 with q3 near -0.015. That clears K1 and keeps the posture (the Run 2 final pattern). This is a gate fix, not a posture fix [inferred].
  - An all-actuator bound at 0.90 also touches the brief reset-transient rail hits of tail_1 and head_pitch (steps 0-36, outside the gate window) [measured that they occur, SAT §1; effect inferred small].
- **Cost:**
  - Code: about 106 lines plus a 245-line test, re-created from TAILEXP, plus the config key, hashing, and a test that no shipped stage sets it.
  - Training: 5 runs × about 14 h on the L4, roughly 70 GPU-hours [measured throughput 230.9-245.2 steps/s, SCORE].
- **How a round shows it worked:**
  - No saturation alarm, and no saturation-rail K1 failure mode, across the seeds that took the habit before (s42, s80).
  - K1-K4 no worse than the recorded band runs.
  - A report-only posture readout says whether the tail parked or came down (§3).

### B. Give tail_3 reach: ctrlrange about ±0.255 to ±0.30 rad

- **Class:** physics plus policy interface (r10 and r16). ctrlrange is hashed in both payloads [measured: LAND, `policy_layer.py` 609-614, `identity.py` 135-159].
  - **Every r9/r15 checkpoint stops loading**, including stance3 Run 2's final, walking pilot 2's parent.
  - It may also need a reward change if the intended tail pose moves: tail_home_pose is centred on the drooped settled pose, -0.0926.
- **What it changes:** a level distal tail becomes reachable at a = 0.85-0.88 (±0.255) or 0.75 (±0.30). The zero-command statue stays bit-identical on the ctrlrange route. A kp of 73 would do the same but moves the statue [measured, LAND plant_probe2.out].
- **Evidence for:**
  - It removes the reason any tail-up preference must sit on the rail.
  - It is the anatomically supported direction: a real tail is held about level [inferred].
- **Evidence against:**
  - No reward term pays for a level tail (MECH).
  - If the drift means "as high as possible" rather than "level", the habit moves to the new rail. At a = +1, q3 would be +0.031 (±0.30) or +0.052 (±0.35) [measured map; outcome inferred].
  - It leaves the dead zone and the feedback loop untouched.
  - Fixing tail_3 alone leaves tail_1 on its stop and the tail drooped. The distal segment points -10° even at the tail_3 joint stop [measured, MECH]. A plausible tail is a tail-chain revision, not a tail_3 tweak [inferred].
- **Risk:** a new lineage, so comparisons with every recorded run reset. The control has to be new runs.
- **Cost:**
  - New statue constants, a new manifest and a regenerated mirror map.
  - New r10/r16 runs for both arms: about 140 GPU-hours for 5+5 seeds.
  - A maintainer decision on the intended tail posture first.
- **How a round would show it worked:** no alarm, and the tail_3 command settling inside (-0.9, +0.9) with q3 near the chosen target.
- **Recommendation:** hold. Decide it together with the tail-chain anatomy question, not as a fix for this habit.

### C. Broad tail_home_pose component (reward)

- **Class:** reward. It changes task_sha256, so a same-stage resume is refused and only a warm start through `initialize_next_stage` works [measured, LAND, `train_base.py` 562-566].
- **What it changes:** it adds a wide-tolerance share to tail_home_pose, as `leg_home_pose_broad_fraction` does for the legs.
- **Evidence for:** at the pinned pose, the narrow Gaussian gives only 0.217 per rad, about 0.019-0.025 per action unit per step. About 90% of the restoring slope in 0.9-1.0 comes from the saturation ramp [measured, MECH].
- **Evidence against:**
  - s42 kept the habit for 10M steps while losing 73-86 per window [measured, SAT §4].
  - Beyond +1, tail_home_pose is flat too, because q3 stops moving at -0.006. So it does not touch the dead zone [measured].
  - It is symmetric, so it also pushes the tail-down seeds: s52 pays 0.053 per step the other way [measured, SAT §6].
- **Risk:** a reward change on all seeds for a habit in about a third of them.
- **Cost:** the same GPU budget as A, plus new task identities in the judge.
- **How a round would show it worked:** the same readouts as A.

### D. Clipped-action policy gradient (CAPG; Fujita and Maeda 2018)

- **Class:** training-loss.
- **What it changes:** for samples beyond ±1, it replaces the log-density with the log tail probability. That gives an unbiased gradient with lower variance than SB3's.
- **Evidence for:** the persistence signal is buried in noise. At 4.19M the policy-gradient pull on tail_3's output bias averaged -0.122 ± 0.037 against an RMS of 0.602 per mini-batch [measured, TAILEXP lambda section].
- **Evidence against:**
  - The expected gradient is unchanged and there is no restoring force. The deterministic mode stays flat beyond +1 [inferred].
  - There is no prior run.
  - It needs a DiagGaussian subclass and a wider exact-type check in `enable_mirror_kl` [measured, LAND].
- **Risk:** low.
- **Cost:** more code than A.
- **Verdict:** a possible later companion to A, not a round on its own.

### Not worth a round

- **Gate exemption or a physical-saturation statistic** (gate): it hides the habit. The joint sits 0.216 rad below its stop on 0.26 of its force, so physical measures cannot see the pin [measured, SAT §1].
- **Previous-action observation** (policy interface, r16): the wrong lever. The loop runs through the joint angle, and a last-action input could make parking easier [inferred].
- **tail_3 log_std floor** (training-loss): weak. Sigma was equal across runs before onset (0.211, 0.209, 0.217), and noise does not move the deterministic fixed point [measured, SAT §3].
- **Store clipped actions, or hard-clamp the mean** (training-loss): biased, or a dead gradient [inferred, LAND].
- **tanh squashing** (policy parameterisation):
  - SB3 asserts that squashing needs gSDE, and `enable_mirror_kl` refuses any non-DiagGaussian distribution [measured, LAND].
  - tanh reaches 0.99 at a pre-tanh value of 2.647, where its slope is 0.02, so the dead zone comes back [measured, clip_arith.out].

**Why A is recommended:**
- It is the only candidate that acts on the measured persistence mechanism, the dead zone.
- It has partial prior evidence in the right direction.
- It keeps r9/r15, every checkpoint and walking pilot 2's parent.
- It is opt-in and bit-identical when off.
- A from-scratch round tests onset as well as persistence, which P1 never did.
- Its main weakness, parking under the rail, is something the round can measure, and it would point to option B's posture question rather than hide it.

## 3. Round sketch for option A

**Arms.**
- Treatment: arm LTSB (the band recipe at cf80ecb1) plus the mean bound (bound 0.90, coef 41, all 15 actuators, training rows only, as P1). From scratch, physics r9, the stance stage, n_envs 4, Colab L4, 12,000,000 steps (final 12,001,280), eval 10 episodes every 131,072 steps, a checkpoint every 524,288.
- Control: **the five recorded band runs** (stance5 s42, s52, s60 and stance5r s70, s80; same HEAD, runtime and judge).
  - This is valid only if the treatment branch at coef 0 is re-measured bit-identical to cf80ecb1.
  - It saves about 70 GPU-hours, which are better spent on treatment seeds.
  - Matched seeds do not pair runs, because GPU training is not bit-reproducible (CRIT line 91). Compare counts, not seeds.
  - A new concurrent control is worth paying for only if the Colab or SB3 stack has changed since 2026-10-23.

**Seeds and what each size can show.** These are the chances of a "clean" result if the bound did nothing. They use the base rate 4/11 = 0.36, and a one-sided Fisher test against the recorded controls [measured: scipy, this brief's arithmetic].

| Treatment seeds | All clean (0 alarms) by chance | Fisher p: 0/n vs recorded 2/5 | vs pooled 4/11 | What it can conclude |
|---|---|---|---|---|
| 3 | 0.26 | 0.36 | 0.33 | Only a clear failure (2 or more alarms). A clean result means nothing. |
| 5 (s42, s52, s60, s70, s80) | 0.10 | 0.22 | 0.18 | It refutes the bound if 2 or more alarm. 0/5 is suggestive, and s42 (2/2 before) is the sharpest single run. It checks K2-K4 cost on the same seeds as the control. |
| 8 (+3 new) | 0.027 | 0.13 | 0.085 | 0/8 is fairly strong against the base rate and still weak against the historical control. |
| 10 | 0.011 | 0.095 | 0.055 | About the most an affordable screen can show. |

**Proposal.** Run 5 matched seeds (about 70 GPU-hours). If the result is "adopt-candidate", add 3 new seeds before any commit, with the extension's rule fixed in the same criteria file. No affordable round proves a difference in the habit rate. The round is a screen for a large effect plus a cost check, and the report must say so [inferred].

**Primary readout, fixed in advance.**
- A = the number of treatment runs with a saturation alarm on any actuator, by `score_stance4.py`'s `saturation_alarm` (two evaluations in a row with at least 5 of 10 episodes listing max_actuator_saturation_fraction), with the actuator named by `satroll5.py`.
- The recorded control has A = 2 of 5: s42 at 3,670,016 and s80 at 4,456,448 [measured, SCORE].

**Secondary readouts, pre-registered and report only.**
- Saturation-rail episodes summed over the six J checkpoints, out of 240 per run, compared with the recorded runs' K1 failure modes.
- tail_3 window-mean command and q3 at each J checkpoint. **PARKED** = a window-mean command of at least 0.6 (SAT's T2 watch threshold) at 2 or more J checkpoints.
- The share of window steps at |a| >= 0.9 for every actuator.
- Training telemetry: `train/mean_bound_loss`, the training-row share of |mu| > 0.9, and tail_3 sigma.

**K1-K4** are as in the replication criteria and stance5 (CRIT "CRITERIA, PER RUN").
- K1: v2 PASS at 6,291,456 and at final. Across J, at most one checkpoint below 37/40 and none below 33/40.
- K2a, K2b and K2c at every J checkpoint.
- K3 at 6.3M and final: joint-noise falls ≤ 2/20 and own-sigma falls ≤ 1/20. Sigma comes from frozen_params for s42, s52 and s60, and from `sigma_s70_s80.json` for s70 and s80.
- K4: roll ≤ 3/40.
- Judged checkpoints J: 4.19M, 5.24M, 6.29M, 8.39M, 10.49M and final. Fails stay fails, and no run is stopped on a readout.

**Decision table** (a proposal; the maintainer sets it before training).
1. **ADOPT-candidate.** A = 0, no run lists max_actuator_saturation_fraction among its K1 failure modes, and the treatment's pass counts are no worse than the recorded band's on K2 (5/5), K3 (3/5) and K4 (5/5).
   - Extend to 3 new seeds. If they also give A = 0 with no K regression, propose the opt-in key, on in stance.toml.
   - If PARKED applies to 2 or more runs, adopt it only as a gate guard and send the posture to the option B decision.
2. **NOT SEPARATED.** A = 1. Extend by 3 seeds, or stop. The maintainer picks which in advance.
3. **NOT ADOPTED.** A ≥ 2 (no better than the recorded 2/5), or a K regression beyond the rule in line 1.

**What must be fixed in the criteria file before any training.**
- The bound (0.90), coef (41), scope (15 actuators), and that the bound applies to training rows only.
- The config key name and its hashing. The coef-0 bit-identity test on the new branch.
- Seeds, sigma sources, runtime, steps and cadence.
- The judge: the stance5r judge plus the new arm identity (HEAD, schema, hyperparameters digest).
- The control's identity: the five runs and the score_final.json sha256 938da843...
- The definitions of A, PARKED and the secondary readouts.
- The decision table, including the K-rule tolerance. The stance5 rule (b) needs K2 and K4 at 5/5. A one-run tolerance may suit n = 5, and the maintainer chooses.
- What happens at A = 1, and the 3-seed extension's rule.
- "Fails stay fails; runs are never stopped or changed on readouts."
- **Recommended before the round** (telemetry, no behaviour change; a code PR): make the eval ledger name the actuator (SAT T1, also the maintainer's open item 3 in the HANDOFF), so A needs no checkpoint fetches.

## 4. Still unknown, and cheap to learn before the round

1. **What the tail-up posture is worth under training noise.**
   - Roll one pinned checkpoint (stance5 s42 at 4.19M, or s80 at 4.72M) with its own sigma on all 15 actuators, three ways: tail_3 free, held at 0, held at 0.9.
   - If the noisy policy gains from tail-up, a bound fights the objective, and B or C moves up.
   - Cost: one Drive checkpoint fetch, since no checkpoint is local [measured: none under tailexp/ or the scratchpads], then CPU only. SB3 2.9.0 and torch 2.13.0+cpu are now importable in the venv [measured].
2. **Level or as high as possible?**
   - On the same checkpoint, a scratch rollout that lets tail_3's deterministic mean drive the servo past the clip (ctrl = 0.2094·mu, no clip).
   - If mu settles near 1.07 with q3 ≈ 0, the "aims for level" reading holds and B is a real cure. If mu keeps climbing, B would only move the rail.
   - CPU, the same fetch, and a scratch copy only.
3. **Does a 0.90 bound hold off the training distribution?**
   - A 1M-step CPU continuation from the same state, as P1 did, at bound 0.90, measuring deterministic frac99 and the overshoot.
   - This is training (a training-loss change in scratch), so it needs the maintainer's OK. It costs about one CPU night and catches P1's failure mode before 70 GPU-hours are spent.
4. **Code readiness, no training.**
   - Re-create P1 with the bound as a parameter, the config key and the tests.
   - Show coef 0 is bit-identical on the band branch.
   - Check that no shipped stage sets the key.
5. **Other actuators near 0.9.**
   - Whether tail_1 and head_pitch reset transients (SAT §1) would feel an all-actuator 0.90 bound. This can be read from the same rollouts as items 1 and 2.
6. **Decisions only the maintainer can make, at no compute cost.**
   - Does "rail cleared but tail parked near 0.9" (PARKED) count as success?
   - Is a drooped tail (tail_1 on its ventral stop, distal segment at -27°) acceptable, or is a tail-chain physics revision (r10/r16, a new lineage) wanted later?
   - Is the recorded band arm an acceptable control?
7. **Not cheap (Drive):**
   - Early checkpoints (1.05M, 1.57M) to compare loop gain and intrinsic bias between habit seeds and clean seeds.
   - `gate_progress.npz` for exact onset timing.
   - Useful for mechanism, not needed for the round.
