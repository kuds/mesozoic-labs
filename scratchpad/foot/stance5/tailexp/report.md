# stance5 scratch tail experiment (P1, a bound on the policy mean): report (report only)

Written to file by the coordinator on 2026-10-22 from workflow wf_a5ac2185-de4: the synthesis returned its findings, but its report text was a placeholder, and the subagent could not write .md files. Report only. The scratch clone scratch_tailfix (branch scratch/tail-mean-bound at 1d2d22b4, push URL disabled) was never pushed. Nothing trains on main and no run, criterion or config changed.

**Read the skeptic checks first.** The headline claim was corrected: P1 did not stop tail_3's Gaussian mean from sitting beyond +1 in deterministic rollouts (it stopped it drifting further out; beyond +1 on 21-46% of window steps at 7 of 9 checkpoints). It kept the mean within +1 only in the stochastic training batches (2.5% of rows against CONTROL's 27.5%). The effect on the gate statistic was not monotone and relapsed by the end. The size and cost claims hold.

## Skeptic checks (written after the synthesis below; where they differ, the check governs)

Each claim of the synthesis was checked by an independent agent that re-rolled the judge's own noise loop or re-reduced the raw files (scratch in the verify_* folders). "Holds: no" means the claim as written is wrong in part; the corrected text replaces it.

### Check 1: holds = no

**Claim as written:** Answer: against the CONTROL arm it ran beside, yes, the mean bound (P1) unpinned the tail, but the effect did not last. Against training alone in general, it is not established. P1 stopped tail_3_pitch_act's Gaussian mean from parking beyond +1, and CONTROL parked it there. P1's gate statistic (tail_3 frac99, the share of window steps at |cmd| >= 0.99) was lower at 8 of 8 periodic checkpoints. P1 was below the start's 0.338 by 4,456,448 (+262,144 steps). It was at 0.000 with 20/20 clean at 4,980,736 (+786,432) and 5,111,808. It then relapsed to 0.470 and 0.488 (12 and 14 of 20 flagged) at 5,242,880 and final. CONTROL never went below 0.338 at any of its 9 checkpoints.

**Corrected:** Answer: against the CONTROL arm it ran beside, yes, the mean bound (P1) unpinned the tail for a time, but the effect did not last and was not monotone. Against training alone in general, it is not established. This is one seed and one pair of runs. The original run, which had training alone (on Colab), read 0.171 at 5,242,880, lower than P1's 0.470 there.

P1 did not stop tail_3_pitch_act's Gaussian mean from sitting beyond +1. In the 20 deterministic episodes it stopped the mean drifting further out. P1's window maximum stayed at 1.022-1.033 (start 1.028), and its mean on pinned steps at 1.009-1.018 (start 1.015). CONTROL parked it further out: maximum 1.044-1.171, pinned-step mean 1.024-1.092, beyond +1 on 42-95% of window steps. P1's mean was still beyond +1 on 21-46% of window steps at 7 of its 9 checkpoints (0% only at 4,980,736 and 5,111,808). At 5,242,880 and final it was beyond +1 on 46%, above the start's 29%. Only in the stochastic training batches did P1 keep it within +1 almost always: beyond +1 on 2.5% of rows on average, against CONTROL's 27.5%.

P1's gate statistic (tail_3 frac99, the share of window steps at |cmd| >= 0.99) was lower than CONTROL's at 8 of 8 periodic checkpoints and at final. Its path:
- above the start's 0.338 at 4,325,376 (0.378);
- below it at 4,456,448 (+262,144 steps; 0.277) and 4,587,520 (0.255);
- back above it at 4,718,592 (0.487, 13 of 20 flagged);
- 0.217 at 4,849,664;
- 0.000 with 20/20 clean at 4,980,736 (+786,432) and 5,111,808;
- relapsed to 0.470 and 0.488 (12 and 14 of 20 flagged) at 5,242,880 and final.

So P1 was below the start at 5 of its 9 checkpoints. CONTROL never went below 0.338 at any of its 9 checkpoints (minimum 0.479 at 4,325,376).

<details><summary>Checker's reasoning</summary>

I re-derived the numbers three ways. Scratch is under /tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/tailexp/verify_4/.

(1) npz_reduce.py re-reduces tailexp/measure/*.npz. It agrees with results.csv. Each episode's gate max-saturation equals the max over actuators of frac99. Flagged episodes all come from tail_3; no other actuator had any step at |cmd| >= 0.99.

(2) reroll.py is my own episode loop: env.reset(seed=3042+i), loaded through judge_stance5.load_checkpoint, deterministic predict with frozen VecNormalize, window 200..L-1. It re-rolled all 19 checkpoints (start, 9 CONTROL, 9 P1) under the clean band clone at cf80ecb1. Every tail_3 frac99, frac90, raw>1 share and raw max matches results.csv to its 4-decimal rounding (max diff 5e-5). Flagged and falls match exactly.

(3) reroll_clean.py runs the judge's run_panel(floor_truth) and classify_stance_episode against the statue reference read from the judged judge_4194304.json. It gives P1 20/20 clean and 0 flagged at 4,980,736 and 5,111,808, 8 clean / 12 flagged at 5,242,880, 6 / 14 at final, and 10 / 10 at the start.

judged_xcheck.py checks measure.py against the judge itself. Per-episode clean flags and saturation-rail flags agree on seeds 3042-3061 at the original run's 4,194,304, 5,242,880 and 6,291,456. The start's 0.33825 also equals satroll5's independent value.

Both arms are MirrorKLPPO from identical start digests. The only difference is coef 41 vs 0, and the clone diff touches only mirror_kl_ppo.py and its test.

Checked and holding:
- P1's frac99 was lower than CONTROL's at 8 of 8 periodic checkpoints, and at final too (0.488 vs 0.921).
- P1 was first below 0.338 at 4,456,448 (0.277; +262,144).
- P1 read 0.000 with 20/20 clean at 4,980,736 (+786,432) and 5,111,808.
- P1 read 0.470 / 12 flagged at 5,242,880 and 0.488 / 14 flagged at final.
- CONTROL's minimum was 0.479, so it never went below 0.338 at any of its 9 checkpoints.
- "Not established in general" holds, and more strongly than stated: the original run (training alone, Colab) read 0.171 at 5,242,880, below P1's 0.470 there, then 0.565 at 6,291,456.

Refuted: "P1 stopped tail_3_pitch_act's Gaussian mean from parking beyond +1". In the same deterministic rollouts, P1's raw mean was beyond +1 on these shares of window steps: 0.361, 0.238, 0.244, 0.463, 0.212, 0, 0, 0.460 and 0.458. That is 7 of 9 checkpoints, and at 5,242,880 and final the share was above the start's 0.293. P1's mean on pinned steps was 1.009-1.018 (start 1.015).

What P1 did was stop the mean drifting further out. Its window max stayed at 1.022-1.033 when pinned (start 1.028). CONTROL's max reached 1.044-1.171, with pinned-step mean 1.024-1.092 and a beyond-+1 share of 0.42-0.95. Only in the stochastic training batches was P1's tail_3 mean rarely beyond +1: 2.5% of rows on average (last 16 rollouts 1.4%), against CONTROL's 27.5%.

Omission: P1 was not steadily below the start. It was above 0.338 at 4,325,376 (0.378) and again at 4,718,592 (0.487, 13 flagged), so it was below the start at 5 of 9 checkpoints, and the end relapse was its second. Inference, not measured: the penalty acted on the training-state distribution, and the deterministic posture's states are seldom visited there.

</details>

### Check 2: holds = yes

**Claim as written:** By how much, as means over the 8 periodic checkpoints (deterministic, 20 episodes): frac99 was 0.86 for CONTROL and 0.26 for P1. Flagged episodes averaged 17.9 vs 6.6, and clean episodes 2.1 vs 13.4. At final the readings were 0.921 with 19 flagged (CONTROL) and 0.488 with 14 flagged (P1). The original GPU run had no penalty and read 0.171 with 6 flagged at 5,242,880, which is better than P1's final. CONTROL and the original are both no-penalty draws from the same state, and they differ by 0.75 at that step. So the size of P1's effect against training alone is open.

<details><summary>Checker's reasoning</summary>

The claim holds. Every number in it comes back the same when I recompute it from the raw per-episode data, and an independent re-roll of the checkpoints gives exactly the same readings.

1) Recomputing from the measure npz files (/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/tailexp/verify_5/recompute.py). "frac99" is tail_3_pitch_act's share of window steps (200..L-1) at |cmd| >= 0.99, averaged over the 20 deterministic episodes. "Flagged" means the gate's max-actuator saturation is above 0.10; in every episode that is exactly the tail_3 share above 0.10. "Clean" is the gate's per-episode flag, and clean + flagged = 20 every time.
- CONTROL at the 8 periodic checkpoints (4,325,376..5,242,880): frac99 0.4793 0.8506 0.9113 0.9424 0.9339 0.9483 0.8718 0.9254, mean 0.8579. Flagged 11 18 19 19 19 19 19 19, mean 17.875. Clean mean 2.125.
- P1: frac99 0.3784 0.2767 0.2545 0.4868 0.2165 0 0 0.4698, mean 0.2603. Flagged 10 7 6 13 5 0 0 12, mean 6.625. Clean mean 13.375.
- So 0.86 vs 0.26, 17.9 vs 6.6 and 2.1 vs 13.4 are all correct.
- Final: CONTROL 0.920875 with 19 flagged; P1 0.4876 with 14 flagged.
- Original run at 5,242,880: 0.1712 with 6 flagged and 14 clean. That is better than P1's final on both readings, and also better than P1's own 5,242,880 checkpoint (0.470 with 12 flagged).
- CONTROL minus original: 0.754 at the 5,242,880 checkpoint and 0.750 at CONTROL's final. Both round to 0.75.
- No falls in any of these.

2) Independent re-roll (/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/tailexp/verify_5/reroll5.py; outputs in verify_5/out/*.json and the logs verify_5/reroll_a.log and reroll_b.log). This is my own episode loop: checkpoints load through judge_stance5.load_checkpoint, the policy is deterministic, the frozen VecNormalize is applied, and episodes start from env.reset(seed=3042+i). It covers the original's 5,242,880 and 6,291,456 (verify_4 had not re-rolled the original's checkpoints), CONTROL at 5,242,880 and final, and P1's final. Every reading matches the measure files exactly:
- original 5,242,880: 0.1711875, 6 flagged
- CONTROL 5,242,880: 0.925375, 19 flagged
- CONTROL final: 0.920875, 19 flagged
- P1 final: 0.4875625, 14 flagged
- original 6,291,456: 0.5648, 14 flagged
verify_4's earlier re-roll of all of CONTROL's and P1's checkpoints also matches.

3) Provenance checks:
- The original run trained on GPU (stage_config gpu: NVIDIA L4) with n_envs 4 and seed 42, and its checkpoint carries no mesozoic_mean_bound, so it had no penalty.
- CONTROL ran on CPU with coef 0.0, which the unit test says is bit-for-bit the same as no penalty. P1 ran with coef 41.0.
- Both arms started from the same 4,194,304 checkpoint: same model sha256 and the same digest of parameters, Adam state and VecNormalize. That is the same starting state the original continued from.
- Caveats (inference): the continuation resets the environments and draws new random numbers. It also skips the update the original made on the rollout that ended at 4,194,304, because the periodic checkpoint is taken before that update. None of this undercuts the phrase "draws from the same state".

4) Context that supports "the size is open" (inference; this is a different panel: the run's own 10-episode eval ledger on different seeds). The original's clean counts at the same 8 steps were 3, 5, 0, 0, 7, 10, 0 and 10 out of 10, so 35/80 = 44% clean. CONTROL was 17/160 = 11% and P1 107/160 = 67%. The no-penalty original therefore swings between 0/10 and 10/10 from one checkpoint to the next and sits between the two arms.

P1 reads below CONTROL at all 8 checkpoints. With one draw per arm, though, that settles which way the effect goes, not how big it is. The claim says only that the size is open, which is right.

</details>

### Check 3: holds = yes

**Claim as written:** Cost: none measured. Falls were 0 in every deterministic panel of both arms. K3 joint-noise falls were 0/20 and own-sigma falls 0/20 at all 18 checkpoints, but this window cannot show a K3 cost. P1's deterministic reward was higher at 7 of 8 slots, by -3 to +64 per episode. The other 14 actuators never came near the bound: max frac90 was 0.000, max |raw mean| was <= 0.20, and their training-state share beyond 1 was 0.0000. Policy std was the same in both arms (tail_3 0.161 -> 0.153). Balance did not get worse with P1. Deterministic K2a, K2b and the roll rail were 0 everywhere. xc median was 64.7-68.9 mm for P1 (statue about 64) against CONTROL's drift from 65.6 to 72.3 mm. Under joint noise, K2a was 0-4 for P1 and 0-7 for CONTROL.

**Corrected:** Cost: none measured. Falls were 0 in every deterministic panel of both arms (20 episodes, 18 checkpoints). K3 joint-noise falls were 0/20 and own-sigma falls 0/20 at all 18 checkpoints. This window cannot show a K3 cost: the original s42 run had no K3 falls through 6,291,456; its first came at 7,340,032. P1's deterministic reward was higher at 8 of 9 checkpoints (7 of the 8 periodic slots, plus final), by -3 to +64 per episode. The other 14 actuators never came near the bound: max frac90 was 0.000, the panel-mean window |raw mean| was at most 0.20 (per-step |mu| at most 0.35 in the window), and their training-state share beyond 1 was 0.0000. Policy std was the same in both arms, within 1.5% on every actuator (tail_3 0.162 -> 0.153 for CONTROL, 0.152 for P1). Balance did not get worse with P1 (inference). Deterministic K2a, K2b and the roll rail were 0 everywhere. xc median was 64.7-68.9 mm for P1 (statue about 64) against CONTROL's drift from 65.6 to 72.3 mm (peak 72.8). Under joint noise, K2a was 0-4 for P1 and 0-7 for CONTROL (start 3).

<details><summary>Checker's reasoning</summary>

The claim holds. Every number in it reproduced, both from the existing outputs and from my own re-roll. Three small precision points are listed at the end; none of them changes the claim.

Method (scratch: /tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/tailexp/verify_6/):
- reduce6.py recomputes the numbers from the measure summaries and npz files, the balance JSONs and both arms' telemetry.npz without using collect.py. It also checks that each checkpoint's sha256 on disk matches the one measured.
- reroll6.py re-rolled all 19 checkpoints: start, plus 9 per arm (4,325,376 to 5,242,880 and final). It ran under the clean band clone at cf80ecb1, which is still clean, and loaded each checkpoint through judge_stance5.load_checkpoint (PPO class, frozen VecNormalize). For each checkpoint it ran:
  (A) its own deterministic episode loop on seeds 3042-3061, recording the raw mu of all 15 actuators at every step;
  (B) judge noise_matched with sigma 0 in joint space, with a hook on action_net;
  (C) judge noise_matched with control_sigma["42"] in joint space (K3);
  (D) judge noise_matched with the checkpoint's own sigma in policy space (K3).
- cmp6.py compares the re-roll with the existing outputs. Zero mismatches: per-episode rewards match to 1e-6, mu from (A) equals mu from (B), and falls, K2a, K2b, the roll rail and xc all match.

Clause by clause:
- Falls: 0/20 in every deterministic panel at all 18 checkpoints, by three paths (run_panel, my own loop, the judge's sigma-0 path).
- K3: joint-noise falls 0/20 and own-sigma falls 0/20 at all 18 checkpoints, and at start. The caveat is a fair inference: the judged original s42 had K3 joint falls of 0 at 4.19M, 5.24M and 6.29M, and the first ones (9) at 7,340,032. So a window ending at 5.24M cannot reveal a K3 cost.
- Reward (P1 minus CONTROL): -2.8, +20.0, +21.9, +16.5, +33.0, +57.5, +63.9, +53.3 at the 8 periodic slots, and +49.0 at final. That is 7 of 8 periodic slots, and the range -3 to +64 is right.
- Other 14 actuators (tail_3 is index 14 of 15):
  - Deterministic frac90 is 0.000 at every checkpoint. The panel-mean window raw mean peaks at |0.1993| (CONTROL 4,325,376), so "<= 0.20" holds.
  - Training share of |mu| > 1 is 0.0 in all 128 rollouts of each arm.
  - For context: per-episode window means reach 0.34, per-step |mu| peaks at 0.347 in the window and 0.596 over whole episodes including settle. All are far from 1.
- Std: start 0.16158; CONTROL final 0.1531; P1 final 0.1524. Across all 15 actuators the two arms differ by at most 1.5% relative (0.0025 absolute).
- Balance: deterministic K2a, K2b and the roll rail are 0 at all 18 checkpoints.
  - P1 xc runs 64.72 to 68.90 mm. The statue is 64.05 mm on 40 seeds (63.97 on 20).
  - CONTROL xc goes 65.63 to 72.31 at final, with a peak of 72.75 at 5,111,808.
  - Joint-noise K2a: P1 [2,1,4,3,0,3,3,1,1], range 0-4, total 18. CONTROL [0,1,2,3,7,4,4,1,1], range 0-7, total 23. Start was 3. "Did not get worse" is an inference, and the data support it. The COM foot-site readout agrees: P1 ends at -15.1 mm, CONTROL at -7.7 mm, statue about -16.

Precision points (wording only, not refutations):
1. "7 of 8 slots" leaves out final, while the K3 sentence counts final among its 18 checkpoints. Over all 9 checkpoint pairs, P1 was higher at 8 of 9, the only exception being 4,325,376 (-2.8).
2. The start tail_3 std is 0.1616, which rounds to 0.162, not 0.161. P1's final is 0.1524, which rounds to 0.152, not 0.153.
3. "max |raw mean| <= 0.20" is the panel-mean statistic. Per-episode and per-step values reach about 0.35 in the window.

</details>


---

## Findings (as returned)
- Answer: against the CONTROL arm it ran beside, yes, the mean bound (P1) unpinned the tail, but the effect did not last. Against training alone in general, it is not established. P1 stopped tail_3_pitch_act's Gaussian mean from parking beyond +1, and CONTROL parked it there. P1's gate statistic (tail_3 frac99, the share of window steps at |cmd| >= 0.99) was lower at 8 of 8 periodic checkpoints. P1 was below the start's 0.338 by 4,456,448 (+262,144 steps). It was at 0.000 with 20/20 clean at 4,980,736 (+786,432) and 5,111,808. It then relapsed to 0.470 and 0.488 (12 and 14 of 20 flagged) at 5,242,880 and final. CONTROL never went below 0.338 at any of its 9 checkpoints.
- By how much, as means over the 8 periodic checkpoints (deterministic, 20 episodes): frac99 was 0.86 for CONTROL and 0.26 for P1. Flagged episodes averaged 17.9 vs 6.6, and clean episodes 2.1 vs 13.4. At final the readings were 0.921 with 19 flagged (CONTROL) and 0.488 with 14 flagged (P1). The original GPU run had no penalty and read 0.171 with 6 flagged at 5,242,880, which is better than P1's final. CONTROL and the original are both no-penalty draws from the same state, and they differ by 0.75 at that step. So the size of P1's effect against training alone is open.
- Cost: none measured. Falls were 0 in every deterministic panel of both arms. K3 joint-noise falls were 0/20 and own-sigma falls 0/20 at all 18 checkpoints, but this window cannot show a K3 cost. P1's deterministic reward was higher at 7 of 8 slots, by -3 to +64 per episode. The other 14 actuators never came near the bound: max frac90 was 0.000, max |raw mean| was <= 0.20, and their training-state share beyond 1 was 0.0000. Policy std was the same in both arms (tail_3 0.161 -> 0.153). Balance did not get worse with P1. Deterministic K2a, K2b and the roll rail were 0 everywhere. xc median was 64.7-68.9 mm for P1 (statue about 64) against CONTROL's drift from 65.6 to 72.3 mm. Under joint noise, K2a was 0-4 for P1 and 0-7 for CONTROL.
- Exact change, in scratch_tailfix at 1d2d22b4 on branch scratch/tail-mean-bound (parent cf80ecb1, push URL disabled, never pushed). environments/shared/mirror_kl_ppo.py gains MeanBoundConfig(coef, bound=1.0), mean_bound_penalty(mu, bound) = relu(|mu| - bound)^2 averaged over rows and the 15 actuators, and enable_mean_bound(model, cfg|None). mu is the Gaussian mean before SB3's clip. MirrorKLPPO._mean_bound_loss adds coef * L_B beside the mirror loss through the existing _mirror_loss and _mirror_end_update hooks, so train() is unchanged. The bound is off by default. With coef 0, L_B is only logged, and a run is bitwise identical to one without the bound (equal digest, measured). The checkpoint stays a stock PPO zip plus a JSON record, mesozoic_mean_bound, and loading a checkpoint never turns the bound back on. 6 new unit tests pass. No stage config key exists: only tailexp/run_continuation.py --coef turns the bound on.
- lambda = 41 (recorded in tailexp/lambda.json). At the 4,194,304 checkpoint, the policy-gradient pull on tail_3's output bias has RMS 0.602 over 256 mini-batches of 128 (theory 1/(sigma sqrt B) = 0.547 at sigma 0.162). Its mean is -0.122 +- 0.037. On a mini-batch whose tail_3 means all sit at +1.11, the penalty's pull is lambda * 2 * 0.11 / 15 = 0.01467 lambda, which equals the RMS at 41.05. The gradient clip (median norm 53 against 0.5) and Adam scale both terms alike. A per-row calibration, which would give about 224, was rejected. Measured at the start: tail_3's mean is above +1 on only 3.7% of training rows, so the realised pull at the start was 0.0056, 0.9% of the RMS. In P1, train/mean_bound_loss averaged 2.1e-4, 2.5e-4, 1.1e-4 and 4.2e-4 by quarter (max 3.6e-3), and the dimension reported each update was always 14 (tail_3).
- Mechanism (measured). CONTROL drifted into the parked regime. Its pinned-step raw mean went from 1.02 to 1.09 and its window raw mean from 0.93 to 1.05-1.06; its training-row share with mu > 1 went 0.06 -> 0.53 (peak) -> 0.25. In P1 the pinned-step raw mean stayed at 1.009-1.018, and its training-row share with mu > 1 stayed <= 0.05 (0.014 at the end). Inferred: P1 works as a wall, not a pull. It cannot hold the command under the gate's 0.99, because the bound is 1.0. It also acts only on states visited in training: at final, P1's raw mean is beyond +1 on 46% of deterministic window steps but on only 1.4% of training rows.
- Limits: the arms trained on CPU, while the screens trained on GPU. Each arm is one draw on one seed (s42), and the 9 checkpoints of an arm are one autocorrelated trajectory, so 8/8 is not a significance test. Only one lambda was run. The experiment tests whether an existing habit persists, not whether it forms: it started from a checkpoint already pinned after the K1 alarm at 3,670,016. It covered only 1M steps and ends at 5,242,880. That is before the K3 slot at 6,291,456 and before s42's joint-noise falls began (5 at 8.4M, 9 at 10.5M, 13/20 at final).
- A real round would need three things. (1) An opt-in config key, such as [ppo] mean_bound_coef / mean_bound, read where train_base applies mirror_kl. It must be off when absent, hashed into hyperparameters_sha256, and covered by a test that no shipped stage config sets it. (2) Criteria fixed before training: the gate statistic, whether the bound sits at 1.0 or below 0.99, the lambda, all three seeds from scratch over 12M steps on the screens' GPU, a paired no-bound control, and K3 plus the no-worse rule. (3) The maintainer's explicit approval, because the bound is a change to the PPO training loss. This scratch result is evidence for that decision; it is not an approval.

## Caveats (as returned)
- report.md was not written: this subagent's rules forbid writing report .md files, which the task itself foresees. The full text is in report_md, ready to be placed at /tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/tailexp/report.md. I wrote no files in this step.
- This report synthesises the build and run steps. I re-checked their key numbers on disk: results.csv (21 rows), lambda.json, both arms' run_meta.json and progress.csv, and the scratch_tailfix diff. I did not re-run any training or measurement.
- The training-state penalty quarter means here are 2.1e-4, 2.5e-4, 1.1e-4 and 4.2e-4, recomputed from p1/progress.csv. The run step reported '1.2e-4 to 4.1e-4'. The difference is rounding or windowing; the max of 3.6e-3 agrees.
- How much faster P1 unpins than training alone is not established. One draw per arm, plus the large spread between the two no-penalty draws at 5,242,880 (0.921 for CONTROL vs 0.171 for the original GPU run), leaves the effect size open. Only 'P1 prevents the mean from parking beyond +1' is established, for this one draw.
- P1's reward gain is not independent evidence. Inferred: the reward includes action_saturation, which ramps from 0.9, and reward correlates with tail_3 frac99 at r = -0.83 across all rows.
- State: scratch_tailfix 1d2d22b4 (push URL DISABLED-never-push), scratch_copband cf80ecb1 and screen_s4_LTS c1e146a8 are clean. /home/user/mesozoic-labs shows no changes. Nothing was pushed. tailexp is 117 MB, and about 2.3 GB of disk is free.

## Files

Under this folder: results.csv (21 rows), lambda.json, train_telemetry.csv, control/ and p1/ (each arm's run_meta.json, progress.csv, checkpoints), measure/, balance/, logs/, smoke/, and the checks' verify_4..verify_6. The scorers and judges were imported, never edited.
