# Tail saturation habit: decision brief (v2, 2026-10-24)

Report only. Nothing was trained or fetched from Drive, and no criterion, config or run was changed. The band clone `scratch_copband` is clean at cf80ecb1 [measured: `git --no-optional-locks status --porcelain --ignored` empty, 16:03 UTC].

This version applies two adversarial skeptics' corrections to `brief_draft.md`, which is unchanged. Appendix A gives each finding and its verdict. Appendix B lists what did not hold as stated.

Labels: **[measured]** means computed or read from a file, with the source named. **[inferred]** means reasoned from measured numbers.

Sources (short names):
- SAT: `checkout/docs/investigations/stance_2026_10/stance4/satdiag/report.md`
- TAILEXP: `scratchpad/foot/stance5/tailexp/report.md` (its skeptic checks govern). In the same folder: TCSV (`results.csv`), TTEL (`train_telemetry.csv`), LAMBDA (`lambda.json`).
- K3DIAG: `.../foot/stance5/k3diag/report.md`. SCORE: `.../foot/stance5/replication/scores/score_final.txt`.
- CRIT: `.../foot/stance5/criteria.txt`. REPL: `.../foot/stance5/replication/criteria.txt`. S3CRIT: `.../foot/stance3/criteria.txt`.
- RUNLOG: `.../foot/stance4/live/run_log.txt`. (`...` = `scratchpad`)
- MECH, LAND, SKF, SKD, V2: `scratchpad2/tailround/` `mech/`, `landscape/`, `skeptic-facts/`, `skeptic-design/`, and `v2check/` (this revision's re-runs).

**Maintainer decisions already recorded for this round** [measured: RUNLOG 2026-10-24 16:00 UTC]. (1) Probes first, then a CPU continuation screen: bound against no bound, 3 replicates per arm from both states, with its criteria brought back before it runs. (2) Bound scope: tail_3, upper side only. (3) "Rail cleared but tail parked" (command about 0.6-0.9) counts as NOT ADOPTED and goes to the posture decision. This brief is consistent with all three.

## 1. Answer first

**The habit is a training artefact on a weak channel. The plant does not force it and no reward term was found that pays for it.** Ranked by evidence:

1. **Persistence: the deterministic fixed point lies off the training distribution** [measured parts; mechanism inferred].
   - Deterministic play reaches the fixed point through the tail_3 joint-angle observation. Swapping only that observation moves the raw mean from +0.998 to +0.290, and from +0.244 to +0.943. Zeroing tail_3's velocity does nothing [measured, SAT §2].
   - With the plant gain of 0.0863 rad per action unit, that is a loop gain of about 0.72 and an amplification of about 3.5x. The loop's time constant, about 75 control steps, fits the in-episode ramp, which reaches 0.99 at a median of step 216 [inferred; MECH probe_plant, SAT §2].
   - Training rarely samples that state. At the stance5 s42 4,194,304 start, 3.7% of training rows had tail_3 mu > 1, with a median of 0.83-0.87. Deterministic play was beyond +1 on 29% of window steps [measured, LAMBDA, TCSV start row].
   - Removing the dead zone from training did not remove the pin. P1 kept training rows beyond +1 to 2.5% (1.4% over the last 16 rollouts), yet its deterministic mean was beyond +1 on 46% of window steps at final [measured, TAILEXP check 1].
   - Under noise the pinned state goes away. With the judge's joint noise, s42's tail_3 is at >= +0.9 on at most 0.012 of window steps at all 7 checkpoints, against up to 0.90 in deterministic play [measured, K3DIAG check 3 item 4].
2. **An unexplained outward drift** [measured; cause unknown].
   - The no-penalty CONTROL drifted further out despite a live restoring slope. Its pinned-step raw mean went from 1.02 to 1.09 (1.07 at final), and its raw maximum reached 1.17 [measured, TCSV].
   - Its training-row share with mu > 1 went 0.06 -> 0.53 (peak) -> 0.25 in 131,072-step trailing windows, 27.5% on average [measured, TAILEXP Findings, TCSV `train_w16_t3_mu_beyond`, TTEL; re-derived in V2 `tele_blocks.out`].
   - At 4,194,304, the mean policy-gradient pull on tail_3's output bias was -0.122 ± 0.037 per mini-batch, negative in all four rollouts (-0.028 to -0.207). LAMBDA's caveat reads that sign as raising the mean [measured, LAMBDA]. The open-loop reward falls as the command rises (item 6), so what drives the outward pull is not known [inferred].
3. **The clipped Gaussian's dead zone contributes at most beyond about mu 1.2** [measured code and arithmetic; role inferred].
   - SB3 2.9.0 clips only for `env.step`. The buffer stores the unclipped sample and its log-prob, and MirrorKLPPO's loss does not bound the mean [measured, MECH sb3src `on_policy_algorithm.py` 199-254, `mirror_kl_ppo.py`].
   - About 4x is the cut in the clipped-command sensitivity, dE[clip a]/dmu = 0.22-0.27 at mu 1.11 [measured, LAND clip_arith.out]. The expected reward slope is cut far less: 1.3-1.9x at mu 1.05-1.11 and 2.6-4.2x at mu 1.2, for sigma 0.153-0.21 [measured: MECH probe_reward.json finite differences at sigma 0.16 (1.29x over mu 1.0-1.1, 2.36x over 1.1-1.2), plus SKF's analytic model, which reproduces the probe's terms; SKF slopes.py re-run identical; V2 ratios.out]. These figures are open loop, with noise on tail_3 only.
   - The measured pinned means are 1.01-1.11 (SAT §1, TAILEXP). There the restoring slope keeps about half or more of its in-range value [inferred from the above].
   - Deterministic play uses clip(mu), which is exactly flat beyond +1 [measured, MECH sb3src `common/policies.py` line 379].
4. **Onset is a trajectory-dependent drift of a low-stakes output** [inferred].
   - The habit appeared in 4 of 10 L+T-family runs: stance3 Run 2 (s60), stance4 s42, stance5 s42 and stance5r s80 [measured: SAT §6, SCORE, CRIT line 173]. stance3 Run 1 (arm L, s60) used a different recipe, was never checked for an alarm, and is not counted.
   - Seed 42 took the habit in 2 of 2 recipes, and s60 in 1 of 3 [measured, SAT, SCORE]. Two no-penalty continuations from one 4.19M state read frac99 0.92 (CPU) and 0.17 (the original GPU run) at 5.24M [measured, TAILEXP check 2].
   - Every recorded onset came before 4.5M, with alarms at 2,228,224 (stance4 s42), 2,359,296 (stance3 Run 2, applied afterwards), 3,670,016 (stance5 s42) and 4,456,448 (s80) [measured, CRIT line 173, SCORE].
   - The channel is worth about 0.03-0.10 per step out of about 4.5 [measured, MECH probe_reward].
5. **The plant enables the habit but does not cause it** [measured].
   - On the statue, q3 = -0.0924 + 0.0863a. A straight tail_3 joint (q3 = 0) would need a = +1.073, which is outside the range [measured, MECH probe_plant C-D, LAND].
   - The limit is the servo's authority against gravity. At q3 = 0, gravity loads the joint with 13.48 N.m, the spring adds nothing, and the servo can give at most 60 × 0.2094 = 12.57 N.m [measured, MECH probe_plant D].
   - The spring (k 80, stiffer than kp 60) narrows the reachable band to about ±0.086 rad around -0.092 [measured map, MECH probe_plant C]. Below q3 = 0 it helps lift the tail: at stiffness 40, q3 at a = +1 falls to -0.0085, against r9's -0.0062 [measured, LAND plant_probe.out]. So softening the spring would not help [inferred].
   - So any lasting tail-up preference ends on the rail. Yet the reward's target pose needs a = 0, and tail_2 can lift the tip as high without any channel going past 0.9 [measured, MECH probe_tail12.out].
6. **No reward incentive found** [measured, open loop only].
   - The per-step reward falls monotonically as the tail_3 command rises. Deterministically it is -0.0971 per step at +1, or -77.7 per 800-step window, which matches SAT's +73 to +86 gain from holding tail_3 at 0. With sigma 0.16 noise on tail_3 it is still monotone [measured, MECH probe_reward.out].
   - **Not tested:** what the posture is worth in closed loop under full 15-channel noise. Item 2's outward pull makes this the main open question (§5 item 1) [inferred].

**Where the two research passes disagree.**
- LAND reads the pinned raw mean (+1.077 to +1.114) through the plant map as aiming for q3 ≈ 0. This is weak evidence [inferred]. MECH finds no reward term that pays for that posture.
- Both expect that a fix which only bounds the command may clear the rail and keep the tail-up posture [inferred].
- stance3 Run 2's final is the nearest precedent, by analogy only. That run had no bound, and its final commanded +0.841 with q3 -0.019 and 0/40 flagged [measured, SAT §5].
- Its path was not monotone: tail down at 8.39M (+0.191), 28/40 judged saturation failures at 9.96M and 6/40 at 11.53M [measured, SAT §5]. Its clean final is a single checkpoint [inferred].

**What it cost the band arm** [inferred from SCORE, measured inputs].
- s80 failed only K1. Its K1 failure modes were all the saturation rail, apart from one tilt episode (36, 25+1 and 14) [measured, SCORE]. Without the habit it would very likely have HELD, making H = 3 of 5 [inferred].
- s42 would still fail K3. Its noise falls come from hip-pitch noise, not the tail [measured, K3DIAG].

**Anatomy** [measured geometry; plausibility inferred].
- tail_1 sits on its ventral stop in every policy. Leaving it would need a = +1.45 [measured, MECH probe_plant E].
- The distal segment points about -27° on the statue, -22.6° at a = +1 and -22.4° at q3 = 0. Even at the joint stop it would point -10.4° [measured, MECH probe_plant C-D]. **q3 = 0 means straight with tail_2, not level in the world.**
- Tyrannosaur tails are usually reconstructed as carried roughly level [inferred, general literature]. So the tail-up drift is not wrong in direction. This plant cannot hold a level tail at all, because tail_1 sits on its stop [inferred].

## 2. Options

Every option needs the maintainer's explicit approval.

### A. Mean bound on tail_3 (training-loss) - **recommended, for a CPU screen only**

- **Class:** training-loss change, through an opt-in `[ppo]` key that is off by default.
  - hyperparameters_sha256 records provenance and is not checked at load, so every checkpoint still loads [measured: grep of `scratch_copband`; SKD].
  - Turning the key on in stance.toml moves the stance recipe digest [inferred].
  - Off, the key must change nothing. P1 at coef 0 was bitwise identical to no bound [measured, TAILEXP].
- **What it changes:** it adds coef × relu(mu - b)² over training rows to MirrorKLPPO's loss through the existing hooks. This is P1's code, re-created from TAILEXP's exact description [inferred from TAILEXP "Exact change"]. The scope is a choice (table below).
- **Evidence for:**
  - P1 (bound 1.0, symmetric on all 15 actuators, coef 41) against its paired CONTROL: mean frac99 0.26 vs 0.86 over 8 checkpoints, lower at all 8 [measured, TAILEXP check 2].
  - P1 had no measured cost in its window: 0 falls, and K3 0/20 at 18 checkpoints. **That window cannot show a K3 cost**, because s42's first K3 fall came at 7,340,032 and P1 ended at 5,242,880. The other actuators were inert, **at bound 1.0 on one seed** [measured, TAILEXP check 3].
- **Evidence against:**
  - **It acts on training rows only, and P1 showed that this does not carry over to deterministic states.** Over P1's last 6 rollouts, the training-row tail_3 q95 was 0.91-0.99, yet deterministic frac99 was 0.470 at 5,242,880 and 0.488 at final [measured, TTEL, TCSV; V2 tele.out].
  - P1 relapsed. The no-penalty GPU original read 0.171 at 5.24M, so P1's effect against training alone is not established. It was one seed, one lambda and a 1M-step continuation, which tested persistence but never onset [measured, TAILEXP checks 1-2].
- **Bound level. "Why 0.90" is withdrawn as evidence.**
  - P1's window maximum of 1.022-1.033 was the start state's own level (1.028), not an overshoot of its bound. P1 never brought the deterministic mean under 1.0 at 7 of its 9 checkpoints [measured, TCSV].
  - No data predict where the deterministic mean will sit under a 0.90 bound. The screen has to measure it [inferred].
  - The only grounds for 0.90 are that the saturation ramp starts there (stance.toml action_saturation threshold 0.9) and that it sits below the gate's 0.99 [measured, MECH; inferred].
  - Coef 41 at 0.90 is a stronger, untested dose. About 25-37% of training rows lay above 0.90 at 4,194,304, while P1's calibration saw only 3.7% above 1.0 [inferred from LAMBDA quantiles]. Either re-calibrate coef by P1's method (`lambda_calib.py`) for the chosen bound and scope, or state that the dose is untested.
- **Risk: parking.** The command may settle just under the bound with the posture kept.
  - At a = 0.9 the deterministic cost is -0.0604 per step (48.3 per window), against -0.0971 (77.7) at +1. About 62% of the cost would remain [measured, MECH probe_reward.out].
  - q3 would be -0.0148 against -0.0061, effectively the same posture [measured, MECH probe_plant C].
  - That outcome counts as NOT ADOPTED and goes to the posture decision (maintainer, RUNLOG 16:00 UTC). It would hide the habit from the gate, which is why this brief rejects a gate exemption [inferred].
- **Untested variant:** the same penalty applied to deterministic eval-rollout states (LAND variant c). It targets the gap in §1 item 1 directly. The stored states go stale as the policy moves, and the variant edges toward training on the gate statistic [inferred, LAND]. It is a candidate if the training-row bound fails the screen.
- **Cost:**
  - Code: about 106 lines plus a 245-line test, re-created from TAILEXP. P1's module no longer exists: `fix/scratch_tailfix` is gone, and only `run_continuation.py` references MeanBoundConfig [measured: `fix/` listing; SKD].
  - `run_continuation.py` hard-codes seed 42 and refuses a run with another seed (lines 59 and 131-132). The screen needs a replicate-seed parameter and support for s80 [measured].

**Bound scope (choice).** The maintainer has chosen tail_3 upper side only (RUNLOG 16:00 UTC). Both scopes are shown for the record.

| | tail_3, upper side only (mu_t3 > b) | All 15 actuators, symmetric (abs(mu) > b; P1's form) |
|---|---|---|
| What it touches | Tail-up excess on tail_3 only. | All of these [measured, SAT §1]: head_pitch at >= 0.9 on 0.134 of window steps (stance3 Run 2, 8.39M, max 0.920); neck_pitch >= 0.9 for 702 window steps (one s52 8.4M episode); ledger flags with an unknown actuator (s52 at 7,864,320, stance3 Run 3 at 4,718,592); reset-transient hits of tail_1 and head_pitch (steps 0-36). Also tail_3 <= -0.9 (tail down) on 50% and 43% of the last 100 steps of falls at 7.3M and 8.4M [measured, K3DIAG check 3 item 4]. |
| In deterministic play | A cap on tail-up only [inferred]. | A soft cap near b on every servo setpoint [inferred]. |
| Prior evidence | None at this scope. | Other 14 actuators inert at bound 1.0, one seed, 4.19-5.24M only [measured, TAILEXP check 3]. |
| What to disclose | A tail-specific special case. | A recipe-wide change. |
| Extra checks | None. | The per-actuator training-row share of abs(mu) > b, and a K3 own-noise read at screen checkpoints. |

### B. Give tail_3 reach: ctrlrange about ±0.255 to ±0.30 rad

- **Class:** physics plus policy interface (r10 and r16). ctrlrange is hashed into both payloads, so **every r9/r15 checkpoint stops loading**, including stance3 Run 2's final, walking pilot 2's parent [measured: LAND, `policy_layer.py` 609-614, `identity.py` 135-159]. It may also need a reward change, because tail_home_pose is centred on the drooped settled pose (-0.0926).
- **What it changes:**
  - q3 = 0 (straight with tail_2, and still about 22° below horizontal) becomes reachable at a = 0.85-0.88 with ±0.255, or a = 0.75 with ±0.30 [measured, LAND plant_probe and plant_probe2, MECH probe_plant D].
  - The zero-command statue stays bit-identical [measured, SKF plant.out]. A kp of 73 would do the same but moves the statue [measured, LAND].
- **Evidence for:** it removes the reason a tail-up preference must end on the rail [inferred].
- **Evidence against:**
  - **B alone does not give a level tail.** That needs authority on tail_1 and tail_2, which means a tail-chain revision [inferred from MECH probe_plant C-E].
  - No reward term pays for q3 = 0 (MECH).
  - If the drift means "as high as possible", the habit moves to the new rail. At a = +1, q3 would be +0.031 with ±0.30, or +0.052 with ±0.35 [measured map; outcome inferred].
  - It leaves the fixed-point gap in §1 item 1 untouched. Softening the spring would not help (§1 item 5).
- **Risk and cost:** it starts a new lineage, so the control has to be new runs. That means new statue constants, a new manifest, a regenerated mirror map, about 140 GPU-hours for 5+5 seeds, and a prior maintainer decision on the intended tail posture [inferred].
- **Recommendation:** hold. Decide it together with the tail-chain anatomy question.

### C. Broad tail_home_pose component (reward)

- **Class:** reward. It changes task_sha256, so only a warm start through `initialize_next_stage` works [measured, LAND, `train_base.py` 562-566].
- **Evidence for:** at the pinned pose the narrow Gaussian gives only 0.217 per rad. About 90% of the restoring slope in 0.9-1.0 comes from the saturation ramp [measured, MECH].
- **Evidence against:**
  - s42 kept the habit for 10M steps while losing 73-86 per window [measured, SAT §4].
  - It is flat beyond +1, because q3 stops moving at -0.006 [measured].
  - It is symmetric, so it also pushes the tail-down seeds: s52 pays 0.053 per step the other way [measured, SAT §6].
- **Cost:** the same as a later round of A, plus new task identities in the judge [inferred].

### D. Clipped-action policy gradient (CAPG; Fujita and Maeda 2018)

- **Class:** training-loss. For samples beyond ±1 it uses the log tail probability, which lowers the gradient variance.
- **Evidence for:** the per-mini-batch pull on tail_3's bias is noisy: a mean of -0.122 ± 0.037 against an RMS of 0.602 [measured, LAMBDA].
- **Evidence against:**
  - It targets the dead zone, which §1 now ranks as a minor contributor [inferred].
  - It adds no restoring force, and the deterministic mode stays flat beyond +1 [inferred].
  - There is no prior run, and it needs a DiagGaussian subclass [measured, LAND].
- **Verdict:** not a round on its own.

### Not worth a round

- **Gate exemption, or a physical-saturation statistic:** it hides the habit [inferred]. The joint sits 0.216 rad below its stop on 0.26 of its force, so physical measures cannot see the pin [measured, SAT §1]. "Rail cleared, tail parked" has the same effect, so it is NOT ADOPTED too [inferred].
- **Previous-action observation:** the wrong lever. The loop runs through the joint angle [measured, SAT §2], and a previous-action input could make parking easier [inferred].
- **tail_3 log_std floor:** weak. Sigma was equal across runs before onset (0.211, 0.209, 0.217) [measured, SAT §3].
- **Storing clipped actions, or hard-clamping the mean:** these give a biased gradient or a dead one [inferred, LAND].
- **tanh squashing:**
  - SB3 requires gSDE for squashing, and `enable_mirror_kl` refuses any distribution other than DiagGaussian [measured, LAND].
  - tanh reaches 0.99 where its slope is 0.02 [measured, LAND clip_arith.out].

**Why A is recommended for the screen.**
- It is the cheapest lever. It is opt-in, keeps r9/r15, every checkpoint and walking pilot 2's parent, and changes nothing when off [measured, TAILEXP; inferred].
- It is the only candidate with prior evidence in the right direction, from P1 against CONTROL [measured].
- **It is not a fix for the measured persistence mechanism.** P1 showed that a bound on training rows does not reach deterministic play. The screen asks whether a lower, tail_3-only bound does, with replicates, before any GPU time is spent [inferred].
- If it does not, the next candidates are the deterministic-state variant and the posture decision (B with a tail-chain revision) [inferred].

## 3. First step: a replicated CPU continuation screen

This replaces the draft's 5-seed from-scratch round as the first step. It is training with a training-loss change in scratch, so it needs the maintainer's OK on fixed criteria.

**Design (proposal).**
- **Arms:** coef 0 against the bound on tail_3, upper side. The bound b and the re-calibrated coef are fixed in the criteria.
- **Replicates:** K = 3 per arm per state. Each replicate gets its own RNG seed for the training env and the action sampling, and the arms use the same seeds.
- **States:**
  - Onset: stance5r s80 at 4,194,304. It was judged 40/40 clean, its first flagged evaluation came at 4,325,376 and its alarm at 4,456,448 [measured, SCORE].
  - Persistence: stance5 s42 at 4,194,304. It is pinned: frac99 0.338, beyond +1 on 0.293 of window steps, raw max 1.028 [measured, TCSV start row].
- **Length:** 0.5-1M steps per continuation, on CPU with n_envs 4, as P1 ran [measured, `run_continuation.py`]. That makes 12 continuations.

**Readouts per replicate**, at every 131,072-step checkpoint, from a deterministic panel (seeds 3042-3061, window steps 200-999, as in TAILEXP):
- **Rail:** tail_3 frac99, and flagged episodes out of 20.
- **Posture:** the window-mean tail_3 command and q3. **PARKED** means a window-mean command of at least 0.6 (SAT's T2 watch threshold) at both of the last two checkpoints.
- **Reach:** the raw maximum, and the share of window steps where the raw mean is beyond b and beyond +1. These show whether the bound reaches deterministic play.
- **Training telemetry:** the share of rows beyond b and beyond +1, the penalty, and tail_3 sigma.

**Outcome per replicate**, read at the last two checkpoints (thresholds proposed; the maintainer fixes them):
- **CURED:** 0/20 flagged and frac99 <= 0.05 at both checkpoints, and not PARKED.
- **PARKED:** the rail is cleared but the tail is parked. This counts as **NOT ADOPTED** and goes to the posture decision (B/C). It is not a gate guard.
- **RAIL:** flagged at either checkpoint.

**Reading rule (proposal).**
- **Persistence (s42):** the bound passes if at least 2 of 3 bound replicates are CURED and at most 1 of 3 control replicates is.
- **Onset (s80):** this state is informative only if at least 2 of 3 control replicates end on RAIL. If so, the bound passes when no bound replicate ends on RAIL and at most 1 is PARKED. If fewer than 2 control replicates take the rail, onset was not tested, and the report says so.
- A from-scratch proposal follows only if every informative state passes. With K = 3, this screen can detect only a large effect [inferred].
- Fails stay fails. No replicate is stopped or changed because of a readout.

**Why the screen comes first.**
- The control replicates give a base rate within each state, which a from-scratch round lacks [inferred].
- A from-scratch round cannot separate a real effect from chance. Treatment at 0/5 against the recorded 2/5 gives a Fisher p of 0.22, and 0/8 gives 0.13 [measured, SKF slopes.out].
- One draw per arm cannot measure an effect size: two no-penalty draws from one state differed by 0.75 [measured, TAILEXP check 2].
- The screen tests both onset (s80) and persistence (s42). It needs only CPU time and the two checkpoint fetches, which have already started [measured, RUNLOG 16:01 UTC].

**Compute estimate on this container** [inferred, from measured inputs].
- This container has 4 CPUs, 15 GB of RAM and 26 GB of free disk [measured: `nproc`, `free`, `df`, 16:03 UTC].
- P1's two 1M-step continuations, their checkpoint reads and the K3 diagnosis ran in one workflow from 15:52 to 21:04 UTC on 2026-10-22, 5.2 h in all, beside two other builds [measured, RUNLOG lines 378-380 and 433-439]. Each continuation is one process with n_envs 4 and one torch thread [measured, `run_continuation.py` lines 59 and 151].
- So one 1M-step continuation, with its reads, took at most about 5 h on about one core. This assumes that container was similar to this one, which was not recorded.
- Running 3 continuations at a time leaves one core for reads. The 12 continuations then take 4 waves: about 10 h at 0.5M steps, or about 20 h at 1M. Four at a time would cut this to about 7.5 h and 15 h but would slow the reads.
- Disk is not a constraint: P1's two arms used 117 MB (RUNLOG line 439), so the screen needs about 0.7 GB.
- Plan: run 0.5M steps first, about one night. Extend to 1M only for states where the arms have not separated.

**What the criteria file must fix before the screen.** The bound b, the coef and its calibration, and the scope. The start checkpoints and their sha256. The replicate seeds, steps, cadence and panels. The readouts, PARKED, the outcome definitions and the reading rule. A coef-0 bit-identity test on the screen code. "Fails stay fails."

## 4. Later, only if the screen passes: a from-scratch GPU round

- **Treatment:** the band recipe at cf80ecb1 plus the bound as screened. From scratch on r9, stance stage, n_envs 4, Colab L4, 12,000,000 steps, with stance5's cadences.
- **Control:** the five recorded band runs (stance5 s42, s52, s60 and stance5r s70, s80).
  - They share HEAD cf80ecb1, the L4 runtime and the K1-K4 code. They were judged under **two judge schemas**, stance5-judge/v1 and stance5r-judge/v1 [measured, SCORE line 2].
  - The comparison is valid only if coef 0 is bit-identical on the final treatment HEAD, telemetry included. Otherwise the telemetry must land as its own behaviour-neutral commit, measured bit-identical first [inferred, CRIT "one difference"].
  - Matched seeds do not pair runs, because Colab GPU training is not bit-reproducible. The source labels this inferred [CRIT line 103; REPL line 123]. Pre-register one analysis: either exchangeable runs, or seed-matched with a stated seed-propensity model.
- **Cost:** 5 runs × about 14 h on the L4, roughly 70 GPU-hours [measured throughput 230.9-245.2 steps/s, SCORE].

**Chance of a clean treatment arm if the bound did nothing** [measured arithmetic, SKF slopes.out; the models are inferred]:

| Treatment runs | All clean, exchangeable at 4/10 | Fisher p, 0/n vs recorded 2/5 | Fisher p, vs pooled 4/10 |
|---|---|---|---|
| 3 | 0.22 | 0.36 | 0.29 |
| 5 | 0.078 | 0.22 | 0.15 |
| 8 | 0.017 | 0.13 | 0.069 |
| 10 | 0.006 | 0.095 | 0.043 |

- Under a seed-specific model (Laplace on each seed's record), a 0/5 result on the five matched seeds has a chance of 0.027. Three new seeds, at the non-s42 rate of 2/9, would all be clean by chance with probability 0.47 [measured arithmetic, SKD arith.out, re-run identical in V2].
- So the draft's 3-new-seed extension adds little. Replace it with more s42 and s80 runs, or drop it [inferred].

**Co-primary readouts, fixed in advance.**
- **A**, the alarm count: the number of treatment runs with a saturation alarm on any actuator, by `score_stance4.py`'s `saturation_alarm`, with the actuator named by `satroll5.py`.
  - The recorded control has A = 2 of 5: s42 at 3,670,016 and s80 at 4,456,448 [measured, SCORE].
  - **A measures whether the bound holds in deterministic play, not whether the habit is gone.** If a tail_3 bound at 0.90 holds, A = 0 for tail_3 by construction [inferred].
- **Posture:** PARKED (a window-mean tail_3 command of at least 0.6 at 2 or more J checkpoints), plus the window-mean tail_3 command and q3 at each J checkpoint, read against the recorded runs' values.
- **Secondary, report only:** saturation-rail episodes summed over J (out of 240 per run), the share of window steps at abs(a) >= 0.9 per actuator, and training telemetry.

**K1-K4** are as in CRIT "CRITERIA, PER RUN". The judged checkpoints J are 4.19M, 5.24M, 6.29M, 8.39M, 10.49M and final. Fails stay fails.

**Decision table** (a proposal; the maintainer sets it before training).
1. **ADOPT-candidate:** A = 0, PARKED in at most 1 run, no saturation-rail K1 failure mode, and the harm rule passes.
2. **NOT ADOPTED (rail cleared, tail parked):** A = 0, but PARKED in 2 or more runs. This goes to the posture decision (B/C) and is not a gate guard.
3. **NOT SEPARATED:** A = 1. Extend with more s42 and s80 runs, or stop. The maintainer picks which in advance.
4. **NOT ADOPTED:** A >= 2, or the harm rule fails.

**Harm rule (the K-rule tolerance).**
- The draft's rule, "no worse than the recorded band on K2 (5/5), K3 (3/5) and K4 (5/5)", needs every run to pass K2 and K4.
- A bound with no cost at all would pass that rule only 0.15-0.68 of the time, for assumed true pass rates of 0.857-1.0 on K2 and K4 and 0.6 on K3 [measured arithmetic, SKD arith.out; the rates are inferred].
- So fix a rule with a stated false-veto rate. For example: at most one more K2 or K4 failure than the recorded band, and K3 judged either on the four non-s42 seeds or with a one-run tolerance. s42's K3 falls are a seed-42 hip-pitch effect seen in both arms [measured, K3DIAG].
- Alternatively, make K1-K4 a report-only cost check, and require a separate cost round before the bound is turned on by default.
- Precedent: the replication round already set stance5's rule (b) aside, because it made adoption impossible [measured, RUNLOG 2026-10-22 15:11 UTC].

## 5. Still unknown, and cheap to learn

1. **What the tail-up posture is worth under training noise** (closed loop, all 15 channels).
   - Roll stance5 s42 at 4.19M with its own sigma three ways: tail_3 free, held at 0, and held at 0.9.
   - If tail-up pays under noise, a bound fights the objective, and B or C moves up [inferred].
   - The checkpoint fetch has started [measured, RUNLOG 16:01 UTC]. After that it is CPU only.
2. **Level, or as high as possible?** On the same checkpoint, run a scratch rollout in which tail_3's deterministic mean drives the servo past the clip (ctrl = 0.2094·mu, no clip).
   - If mu settles near 1.07 with q3 ≈ 0, the "aims for q3 = 0" reading holds.
   - If mu keeps climbing, B would only move the rail [inferred].
3. **The screen (§3)** replaces the draft's single continuation at bound 0.90.
4. **Code readiness, no training.**
   - Re-create P1 with the bound, the scope and the replicate seed as parameters.
   - Show coef 0 is bit-identical, and check that no shipped stage sets the key.
5. **Maintainer decisions still open:**
   - Is a drooped tail acceptable, or is a tail-chain revision (r10/r16) wanted later?
   - Is the recorded band arm an acceptable control for a later round?
   - Which bound level and coef should the screen use?
   - The scope and the treatment of PARKED are already decided (RUNLOG 16:00 UTC).
6. **Not cheap (Drive):** early checkpoints (1.05M, 1.57M) and `gate_progress.npz`, for onset timing. They help explain the mechanism and are not needed for the screen.

## Appendix A. Skeptic findings and verdicts

Every finding was checked against its source. All of them hold at least in substance. Where a finding held only in part, Appendix B says what did not hold.

| # | Finding | Verdict and check | Applied in |
|---|---|---|---|
| F1 | The dead zone does not explain persistence; re-rank the mechanism. | Holds. SKF slopes.py re-run identical (V2 slopes_rerun.out). The reward-slope cut is 1.29-1.94x at mu 1.05-1.11 for sigma 0.153-0.21 (V2 ratios.out). Also P1's 2.5% of training rows vs 46% of deterministic steps (TAILEXP check 1), and K3DIAG's noise readout. Wording narrowed (B2). | §1 items 1-3; A rationale |
| F2 | "Why 0.90" overshoot is mislabelled measured. | Holds. TCSV: start raw max 1.0281; P1's maxima 1.022-1.033 on pinned checkpoints; beyond +1 at 7 of 9. | §2 A, bound level |
| F3 | The cost claim omits "this window cannot show a K3 cost", and 41 at 0.90 is a new dose. | Holds. TAILEXP check 3 corrected text; LAMBDA quantiles. | §2 A |
| F4 | "3.7% of rows" generalised from one state. | Holds. TTEL/TCSV: CONTROL 0.06 -> 0.53 -> 0.25, average 27.5%. Windowing stated (B3). | §1 items 1-2 |
| F5 | 4 of 11 should be 4 of 10. | Holds. S3CRIT lines 14-16 (Run 1 is arm L); CRIT line 173; SAT "2 affected of 5 L+T runs". | §1 item 4; §4 table |
| F6 | Wrong CRIT line for GPU non-reproducibility. | Holds. Line 91 is the control's passes; the claim is at line 103, marked inferred; REPL line 123 says the same. | §4 control |
| F7 | Spring wrongly blamed for the unreachable q3 = 0. | Holds. MECH probe_plant D: gravity 13.48, spring 0, servo maximum 12.57 N.m. LAND stiff40: -0.0085 vs -0.0062. | §1 item 5; B |
| F8 | q3 = 0 is not world-level. | Holds. MECH probe_plant D: -22.37° at q3 = 0, -10.37° at the stop. | §1 anatomy; B |
| F9 | In-window head/neck events omitted. | Holds. SAT §1 lines 22-25. | Scope table |
| F10 | "Run 2 did exactly that" is an analogy. | Holds. SAT §5 table and verification item 2. | §1 disagreement |
| F11 | "Same judge" is wrong. | Holds. SCORE line 2: stance5-judge/v1 and stance5r-judge/v1. | §4 control |
| D1 | Withdraw "Why 0.90"; nothing shows a training-row bound keeps deterministic play under 0.99. | Holds. TTEL last 6 P1 rollouts: q95 0.912-0.993, mean 0.566-0.816. TCSV: frac99 0.470/0.488. Amplification estimate not adopted (B1). | §2 A |
| D2 | A is near a manipulation check; posture co-primary; PARKED is NOT ADOPTED. | Holds. MECH: 48.3 vs 77.7 per window; q3 -0.0148 vs -0.0061. Matches the maintainer's decision (3). | §3 outcomes; §4 table |
| D3 | A from-scratch round cannot separate from chance; screen first. | Holds. SKD arith.py re-run identical (V2 arith_rerun.out); SCORE onsets and s80 40/40 at 4.19M. One supporting point swapped (B4). | §3 |
| D4 | The strict K rule vetoes a no-cost bound. | Holds. Same re-run: 0.146-0.683. | §4 harm rule |
| D5 | Risk paragraph incomplete; scope and dose. | Holds. SAT §1; K3DIAG check 3 item 4; TTEL first row q50 0.869, q95 0.989. | Scope table; §2 A |
| D6 | Understated class scope; telemetry confound. | Holds. The digest is not checked at load; wider file list (B5). | §2 A class; §4 control |

## Appendix B. What did not hold as stated

- **B1. Design skeptic's "0.90 + 0.116 = 1.016" prediction.** It amplifies P1's 0.033 "overshoot", which F2 shows is the start state's own level, not an overshoot. It is not used as a prediction. "Why 0.90" is still withdrawn, on P1's telemetry alone (D1).
- **B2. Facts skeptic's "under training noise".** The slope ratios come from an open-loop statue probe with noise on tail_3 only (sigma 0.16), plus an analytic model over sigma 0.153-0.21. The 1.3-1.9x figure reproduces exactly (V2 ratios.out: 1.34-1.37x at mu 1.05, 1.67-1.94x at mu 1.11; measured finite differences 1.29x over 1.0-1.1). The brief narrows the wording to what was measured. Full 15-channel closed-loop noise was never measured.
- **B3. "0.06 -> 0.53 (peak) -> 0.25" and "P1 stays <= 0.05".** These are trailing 16-rollout (131,072-step) means at the checkpoints: TCSV `train_w16_t3_mu_beyond` gives 0.0619, 0.5266 and 0.2502, with a rolling peak of 0.533.
  - Per rollout, CONTROL peaked at 0.664 and P1 at 0.182.
  - P1's trailing mean was 0.057 at its first checkpoint, then <= 0.050, ending at 0.014.
  - The overall means, 27.5% and 2.5%, reproduce (V2 tele.out, tele_blocks.out).
  - The claim holds in substance, and the brief states the windowing.
- **B4. Design skeptic's "s60 took the habit in stance3 Run 2 but not in Run 1 (same seed)".** Run 1 is arm L, a different recipe that was never checked for an alarm (F5). The conclusion that a seed does not fix the outcome still holds, on two other grounds: s60 took the habit in 1 of 3 recipes, and two draws from one state diverged.
- **B5. Design skeptic's list of files using hyperparameters_sha256.** It also appears in `cli.py` (help text), `mirror_kl_ppo.py` and `symmetry.py` (comments), a harness and tests. None of them checks it at load, so the conclusion holds.
- **No draft text was kept against a skeptic.** Every correction held at least in substance. The draft's seeds-table arithmetic was correct for its 4/11 base; only the base was wrong.
