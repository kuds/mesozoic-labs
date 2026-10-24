# stance5 K3 diagnosis: report (report only)

Written to file by the coordinator on 2026-10-22 from workflow [id removed]: the subagent could not write .md files. Report only; no verdict, criterion or run changes.

**Read the skeptic checks first.** Three of the four checked headline claims were corrected:
- the stagger before a fall comes mostly from the pelvis yawing over planted feet (scissored hip pitch), not the feet drifting apart, and it is associated with falls, not a threshold;
- hip-pitch noise is the main cause, but the "differential 17-19/20" figures came from a boosted test (twice the natural variance); at the judge's real amplitude the differential part alone gives 5, 2, 7 and 15/20 at 7.3M, 8.4M, 10.5M and final, and control s42 needs knee and hip-pitch noise together;
- the -3.2 to -0.75 "correction gain" was a regression artefact; the policy's direct response to a left/right hip-pitch difference halves between 6.3M and 7.3M (-1.04 to -0.51 per rad), but restoring it from outside does not remove the falls, so it is a correlate, not the mechanism. The mechanism is open.
The tail claim (s42's K3 falls are not the tail habit) holds in full. The band-against-control readings in sections 3 and 6 (43 against 11 paired falls, the kick probe) were not independently checked.

## Skeptic checks (written after the synthesis below; where they differ, the check governs)

Each claim of the synthesis was checked by an independent agent that re-rolled the judge's own noise loop or re-reduced the raw files (scratch in the verify_* folders). "Holds: no" means the claim as written is wrong in part; the corrected text replaces it.

### Check 1: holds = no

**Claim as written:** Answer [measured]: the K3 joint-noise falls are slow topples, with a median of 0.9-1.5 s from the start of the lean to the end of the episode. The trigger is the feet drifting apart fore-aft: every fall is preceded by a stagger of at least 81 mm, against 5-16 mm p95 without noise. Then one foot unloads and slides, and the pelvis drops below 0.70 m.

**Corrected:** K3 falls [measured; 48 joint-noise falls re-rolled to match the judge files exactly: band s42 at 7.3M, 8.4M, 10.5M and final, band s60 final, control s42 and s60 final]: they are slow topples. With the report's onset definition (|roll| or |pitch - ref| > 3 deg, or pelvis more than 2 cm low, ref from the no-noise roll), the per-slot medians are 0.9-1.5 s from onset to the end (pooled 1.17 s). With a 5 deg tilt threshold they are 0.55-1.0 s. The last 5 cm of pelvis drop takes only about the final 0.23 s.

Before onset, every one of the 48 falls shows a heading-frame foot stagger (env foot_stagger_m) of at least 81 mm (sample minimum 81 mm, s60 final seed 3059). Without noise the stagger p95 in these slots is 5-16 mm. With noise, survivors in the same slots have p95 19-75 mm, and 8 of 87 survivors also pass 81 mm (up to 175 mm) without falling. At other checkpoints falls occur at 73-78 mm (band s60 10.5M seed 3057, control s60 11.5M seed 3059). So the stagger is strongly associated with falls but is neither a threshold nor sufficient.

The stagger is mostly not the feet drifting apart. In 40 of 45 falls it comes mainly from the pelvis yawing (median 24 deg, up to 56 deg) over nearly planted feet: the median fixed-frame fore-aft change is 17 mm, and the fixed-frame offset reaches 81 mm in only 8 of 45. At the stagger peak the hip-pitch angles are split (scissored legs, inferred). Two falls (seed 3061 at band s42 and s60 final) have their stagger peak at spawn.

After onset, one foot drops below 0.02 body weight in 45 of 48 falls and is mostly swung (a recovery step) while the loaded foot slides about 35 mm. The pelvis goes below 0.70 m only in the 31 'fallen' endings, where that is the termination test itself. The 17 nosedive, tail-contact and head-contact falls end with the pelvis no lower than 0.705-0.87 m. Of the 16 band falls at the K3 slots, 15 are 'fallen' and 1 is a nosedive.

<details><summary>Checker's reasoning</summary>

I re-rolled the judge's joint-noise loop myself (verify_0/scripts/vroll.py, run through iso.py, judge-style checkpoint loading, joint-space sigma from frozen_params, default_rng(seed), recorder attached the same way the judge attaches it). I rolled 18 checkpoints with noise, 12 of them also without noise. Every noisy roll matched its judge file exactly: lengths, termination reasons, rewards to 0.0, fall seeds. The 48 falls the claim covers are band s42 at 7.3M, 8.4M, 10.5M and final, band s60 final, and control s42 and s60 final. The figures 81 mm, 5-16 mm and 0.9-1.5 s match the earlier k3diag tables for these 48 falls. Outputs: verify_0/vsum.json, vreport.txt, roll/*.npz.

What reproduces:
(a) Time to fall. With the report's onset definition (|roll| > 3 deg, |pitch - ref| > 3 deg or pelvis more than 2 cm low, ref taken from the no-noise roll), the per-slot medians are 0.92-1.53 s (pooled 1.17 s). This depends on the definition. A 5-degree tilt threshold gives slot medians of 0.55-1.0 s. The final 5 cm of pelvis drop takes only the last 0.23 s (median).
(b) Stagger. The maximum |foot_stagger_m| in the 2 s before onset is at least 81 mm in all 48 falls. The minimum is exactly 81 mm (s60 final, seed 3059).
(c) Baseline. Without noise, the stagger p95 in those slots is 5.1-15.6 mm.

What fails:
1. "The feet drifting apart fore-aft" is the wrong mechanism. foot_stagger_m is the r_foot minus l_foot offset measured along the pelvis heading. I split each stagger peak into a part from the feet moving on the ground (measured in the step-200 heading frame) and a part from the pelvis turning. In 40 of 45 falls the pelvis-turning part is larger. The pelvis yaws a median 24 deg (up to 56 deg). The feet's fore-aft offset in that fixed frame changes by a median of only 17 mm. That fixed-frame offset reaches 81 mm in only 8 of 45 falls. Example: band s42 final, seed 3044. The world-frame foot offset stays about 5-20 mm while the pelvis yaws 27 deg. The heading-frame metric reads 119 mm. The other analyst's per-joint data for the 13 s42 final falls shows the hip-pitch angles splitting by 0.06-0.18 rad (right minus left) at the stagger peak. So the legs scissor under a yawing pelvis while the feet stay nearly planted (inferred). The feet move clearly (77-195 mm) only in the few falls where a foot steps or the spawn landing shifts it. Two of the 48 falls (seed 3061 at band s42 final and band s60 final) have their stagger peak at spawn, steps 6-7.
2. "Trigger" overstates the evidence. Noisy survivors in the same slots have stagger p95 of 19-75 mm, and 8 of 87 survivors also pass 81 mm (up to 175 mm) without falling. At other checkpoints, falls happen with lower stagger. Band s60 10.5M seed 3057 peaks at 78 mm over the whole episode, and control s60 11.5M seed 3059 at 73 mm. So 81 mm is the minimum of this sample, not a threshold. The stagger is strongly associated with the falls (48 of the 56 episodes that pass 81 mm in those slots fell), but it is not sufficient on its own.
3. "The pelvis drops below 0.70 m" is false for 17 of the 48 falls: every nosedive, tail-contact and head-contact ending. Those episodes end with the pelvis no lower than 0.705-0.87 m. For the 31 'fallen' endings the statement just restates the termination test (healthy_z_range lower bound 0.70). Among the 16 band falls at the K3 slots (s42 final and s60 final), it is 15 'fallen' and 1 nosedive.
4. "One foot unloads and slides" is imprecise. After onset a foot drops below 0.02 body weight in 45 of 48 falls; 3 never unload. The unloaded foot mostly swings, travelling a median 278 mm while unloaded, which looks like a recovery step. The still-loaded feet slide a median of about 35 mm after the unload.

</details>

### Check 2: holds = no

**Claim as written:** Answer [measured]: the cause is hip-pitch noise in its left-against-right (differential) mode. Tail-only and neck/head-only noise give 0/20 at all 12 checkpoints. On band s42 from 7.3M, hip pitch alone gives 8-15/20, the legs without hip pitch 0-2, differential hip pitch 17-19 and common-mode hip pitch 0-3. Control s42 final is the exception: its falls need knee noise (the legs without knee give 1/20, knee alone 4/20).

**Corrected:** Answer [measured]: hip-pitch noise is the main cause, mostly its left-against-right (differential) mode. Tail-only and neck/head-only noise give 0/20 at all 12 checkpoints. On band s42 at 7.3M, 8.4M, 10.5M and final, hip pitch alone gives 8, 3, 9 and 15 of 20. The full noise without hip pitch gives 0, 0, 0 and 2, and the legs without hip pitch give 0, 0, 1 and 2. The split was done on the same draws. The differential part of the actual hip-pitch noise alone gives 5, 2, 7 and 15 of 20; the common-mode part alone gives 0, 0, 0 and 1. The full noise without the differential part gives 1, 0, 0 and 5; without the common-mode part, 7, 3, 9 and 8. At the final, the common mode together with the other actuators still causes 5/20 falls. The counterfactual where the left hip copies the right draw at full sigma, which doubles one mode's variance, gives 17, 4, 17 and 19 for the differential mode and 1, 0, 0 and 3 for the common mode. Control s42 final needs knee and hip-pitch noise together: knee alone gives 4/20 and the legs without knee 1/20; hip pitch alone gives 2/20 and its differential part alone 3/20; the full noise without knee, without hip pitch, or without the differential hip-pitch part each gives 2/20, against 8 with full noise.

<details><summary>Checker's reasoning</summary>

The main point stands, but two of the numbers are wrong, the "differential" and "common-mode" figures come from boosted test conditions rather than the actual noise, and the control exception is overstated.

How I checked:
- I read the fall counts straight from all 186 rollout files in k3diag/rollouts/npz. They match tables/ablation_matrix.csv, and every "full" run matches its judge file.
- I wrote my own loop, verify_1/vroll.py, separate from k3roll.py. It copies the judge's noise_matched in joint space: frozen control sigma, default_rng(episode seed), a = clip(clip(mu) + n), recorder attached, checkpoints loaded with J.load_checkpoint. It reproduces the judge's lengths and termination reasons exactly at band s42 7.3M, 8.4M, 10.5M and final, band s60 final and control s42 final.
- The new conditions use the same draws. hpD and hpC split the actual hip-pitch noise n into its differential part (d, -d) with d = (n_r - n_l)/2 and its common part (c, c) with c = (n_r + n_l)/2; the two add back to n exactly. fullD and fullC are the full noise with the hip-pitch pair replaced by one of those parts. nohp and noknee are the full noise minus those actuators.
- In the mirror map hip pitch has sign +1, so z_l = z_r is the common mode and z_l = -z_r the differential mode.
- Outputs are in verify_1 (p*.jsonl and q*.jsonl, plus logs). Both clones are still clean.

What holds:
- Tail-only and neck/head-only noise give 0/20 at all 12 checkpoints. I re-rolled 8.4M myself and also got 0 and 0.
- Band s42 legs without hip pitch give 0, 0, 1, 2 at 7.3M, 8.4M, 10.5M and final. The full noise minus hip pitch gives 0, 0, 0, 2.
- Control s42 final: knee alone 4/20 and legs without knee 1/20, both reproduced.

What fails:
1. "Hip pitch alone gives 8-15/20" from 7.3M. The counts are 8, 3, 9 and 15. At 8.4M it is 3/20 (seeds 3046, 3059, 3061) against 5 under full noise. Found in the matrix and reproduced by my own run.
2. "Differential 17-19, common-mode 0-3." These numbers come from k3roll's hp_anti and hp_sym, where the left hip draw is replaced by minus or plus the right draw at full sigma. Each mode then has twice its natural variance, so this is a boosted test, not a split of the actual noise. They were also only run at 7.3M, 10.5M and final (17/17/19 and 1/0/3). I ran them at 8.4M: hp_anti 4/20, hp_sym 0/20. So "17-19 from 7.3M" is false.
3. Splitting the actual hip-pitch noise supports the cause, though less strongly. At 7.3M, 8.4M, 10.5M and final:
   - differential part alone (hpD): 5, 2, 7, 15
   - common part alone (hpC): 0, 0, 0, 1
   - full noise without the differential part (fullC): 1, 0, 0, 5
   - full noise without the common part (fullD): 7, 3, 9, 8
   At the final, taking out only the differential part still leaves 5/20, above the K3 limit of 2. Taking out all hip pitch leaves 2/20.
4. Control s42 final is not a "knee instead of hip pitch" exception. Full noise without knee gives 2/20, without hip pitch 2/20, and without the differential hip-pitch part (fullC) also 2/20, against 8 under full noise. Hip pitch alone gives 2/20 and its differential part alone 3/20. Its falls need both knee and hip-pitch noise. [inference] This was one draw set of 20 episodes per condition, so a difference of 2 or 3 falls between conditions is within binomial noise.

</details>

### Check 3: holds = no

**Claim as written:** Answer [measured; mechanism inferred]: s42's rise is not the tail habit. It tracks a step change between 6.3M and 7.3M in how strongly the policy corrects a left/right hip-pitch difference: the gain goes from -3.2 to -0.75 (-0.19 at the final). Over the same step the stagger sd under noise goes from 11 to 50 mm and differential-noise falls from 0 to 17/20.

**Corrected:** Answer [measured; mechanism not identified]: s42's rise is not the tail habit. Holding tail_3 at 0 leaves the falls at 0, 9 and 12 (against 0, 9 and 13 at 6.3M, 10.5M and final). Tail-only noise gives 0 falls at every checkpoint. The tail_3 share at |cmd| >= 0.9 does not track the falls. The falls need hip-pitch noise: leg noise without hip pitch gives 0, 0, 1 and 2 falls at 7.3M, 8.4M, 10.5M and final, and hip-pitch noise alone gives 8, 3, 9 and 15. Three things change together between 6.3M and 7.3M, when the judged falls go from 0 to 9/20. First, the policy's direct response to a left/right hip-pitch difference halves (measured by nudging that difference in its input): from -1.04 to -0.51 per rad. It is then -0.55 at 8.4M, -0.17 at 10.5M and -0.34 at the final. Second, the stagger sd under noise goes from 11 to about 50 mm (30 mm at 8.4M, 34 at 10.5M, 44-51 at the final). Third, falls when the left hip's noise is the negative of the right's go from 0 to 17/20 (then 4, 17 and 19). That test doubles the judge's left/right differential noise variance. At the judge's real amplitude, the differential part alone gives 0 to 5/20 (then 2, 7 and 15). The earlier figures of -3.2, -0.75 and -0.19 come from a regression, not from the policy's gain. That regression pairs each command with the angle recorded after the step, uses two predictors that move almost together (correlation -0.96), and explains little (R² 0.02-0.06). Paired with the angle the policy actually saw, the figures are -2.8, -0.98 and -0.73. The same regression puts 10.5M at -1.6 to -1.9 while it has 9 falls. The step in the gain does not explain the falls. An outside left/right hip-pitch correction that restores the 6.3M-level response leaves 7/20 falls at 7.3M, 9/20 at 10.5M and 12/20 at the final. Even 2.5 per rad leaves 3, 13 and 10. Cutting the 6.3M policy's response to the 10.5M level gives only 1-2/20. Band s52's final has a similar response (-0.59) and 0 falls. So the step in the hip-pitch correction is a correlate of what changed between 6.3M and 7.3M, not its mechanism, and the mechanism is still open.

<details><summary>Checker's reasoning</summary>

The claim does not hold. The tail part and the step itself are right, but the gain figures do not measure the policy's correction, and the gain step is not shown to cause the falls.

**What I checked.** Every new run used the judge's own noise loop, imported read-only from judge_stance5.py, with the checkpoint loaded through J.load_checkpoint. Every "full" condition matched the judge files exactly: lengths, terminations and rewards (difference 0.0). The s42 falls are 0, 0, 0, 9, 5, 9, 13 at 4.2M, 5.2M, 6.3M, 7.3M, 8.4M, 10.5M and final, so the rise is not monotonic. Scratch outputs are in scratchpad/foot/stance5/k3diag/verify_2/.

**1. "Not the tail habit": holds.**
- Holding the tail_3 command at 0 leaves the falls unchanged: 0, 9 and 12 against 0, 9 and 13 at 6.3M, 10.5M and final.
- Tail-only noise gives 0 falls at every checkpoint.
- The tail_3 share at |cmd| >= 0.9 does not track the falls. With no noise it is 0.75 at 6.3M (0 falls) and 0.09 at 10.5M (9 falls). Under the judge's noise it is at most 0.04 at every checkpoint.
- Leg noise without hip pitch gives 0, 0, 1 and 2 falls at 7.3M, 8.4M, 10.5M and final. Hip-pitch noise alone gives 8, 3, 9 and 15.

**2. The gain figures reproduce, but only with the report's method.**
- Using k3gain's own window (from its episodes.csv onsets) and the stored angles, I get exactly -3.198, -0.745 and -0.189.
- That regression pairs each command with the hip angle recorded after the step, which the command itself just moved. With the angle the policy actually saw, the figures are -2.80, -0.98 and -0.73 at the final.
- The two predictors, stagger and hip-pitch difference, move almost together (correlation -0.96 at 7.3M). The fit explains little: R² is 0.06 at 7.3M, 0.02 at 8.4M and 0.02 at the final.
- The claim leaves out 10.5M, where this regression gives -1.6 to -1.9. That is a stronger gain than at 7.3M, yet both checkpoints have 9 falls.
- I also measured the policy's own response directly, by nudging the left/right hip-pitch angle difference in its input and reading the change in its command. Per rad, on the noisy states:

| Checkpoint | 4.2M | 5.2M | 6.3M | 7.3M | 8.4M | 10.5M | final |
|---|---|---|---|---|---|---|---|
| Direct response | -1.17 | -0.95 | -1.04 | -0.51 | -0.55 | -0.17 | -0.34 |

  So there is a real step between 6.3M and 7.3M, but it is a halving, not -3.2 to -0.75.

**3. Stagger sd under noise: holds.** It goes from 10.7 to 48-50 mm (the exact figure depends on the window). It is then 30 mm at 8.4M, 34 at 10.5M and 44-51 at the final.

**4. "Differential-noise falls 0 to 17/20": true, but needs a label.** The 17/20 comes from a counterfactual where the left hip's noise is the negative of the right's. That doubles the left/right differential noise variance of the judge's test. My re-run gives 0, 0, 0, 17, 4, 17 and 19 across the checkpoints. At the judge's real amplitude, the differential part of the hip-pitch noise alone gives 0, 0, 0, 5, 2, 7 and 15.

**5. The mechanism is not supported (counterfactual tests).** I added an outside left/right hip-pitch correction on top of the policy's command, and checked that it was applied at the intended strength.
- Restoring the 6.3M-level response gives 7/20 falls at 7.3M (from 9), 4 at 8.4M (from 5), 9 at 10.5M (from 9) and 12 at the final (from 13).
- Adding a correction for the rate of change as well gives 5, 9 and 11.
- Even a correction of 2.5 per rad, more than twice the 6.3M level, gives 3, 13 and 10.
- Going the other way, cutting the 6.3M policy's response down to the 10.5M level gives only 1/20 falls, or 2/20 when the rate response is cut too.
- Across runs, band s52's final has about the same direct response (-0.59) as s42 at 7.3M. Yet it has 0 falls under the judge's noise and 0 under the doubled-variance test.
- What does track the full-noise falls is the falls under left/right hip-pitch noise (rank correlation 0.99 to 1.0). The weakness is to that kind of disturbance, but this hip-pitch correction is not the lever.

**Labels.** The policy probe and the outside correction are counterfactual tools. All conclusions about mechanism are inferences.

</details>

### Check 4: holds = yes

**Claim as written:** Tail [measured]: the deterministic tail_3 share does not track the falls (0.77, 0.67, 0.75, 0.90, 0.59, 0.09, 0.75 against falls of 0, 0, 0, 9, 5, 9, 13; Spearman -0.15). Under the noise the habit vanishes (share <= 0.012). Holding tail_3 at 0 under the same noise still leaves 12 of 13 falls at the final and 9 of 9 at 10.5M. Control s42 was 35-40/40 flagged with 0-1 falls.

**Corrected:** Tail [measured]: the deterministic tail_3 share (clip(mu) >= 0.9, window steps 200-999, 20 episodes, seeds 3042-3061) does not track the joint-noise falls. At 4.2, 5.2, 6.3, 7.3, 8.4, 10.5M and the final, the shares are 0.77, 0.67, 0.75, 0.90, 0.59, 0.09, 0.75 and the falls are 0, 0, 0, 9, 5, 9, 13 (Spearman -0.15, n = 7). Under the noise the habit vanishes: the policy's own tail_3 command is at >= +0.9 on <= 0.012 of window steps (<= 0.06 if counted on the applied command after noise). In the final second of a fall the policy instead drives tail_3 to -0.9 or below (43-50% of those steps at 7.3M and 8.4M). Holding the tail_3 command at 0 under the same noise draws leaves:
- 12 of the 13 final falls (the same seeds minus 3052);
- the same 9 of 9 at 10.5M;
- the same 9 of 9 at 7.3M and 5 of 5 at 8.4M, where the deterministic habit is strongest.
With no noise on tail_3 either (a = 0 exactly), all 13, 9, 9 and 5 remain. Control s42 (stance4) was 35-40/40 saturation-flagged with 0-1 joint-noise falls from 2.1M to 9.4M. At 10.5M, 11.5M and the final it was 23-24/40 flagged with 1, 2 and 8 falls.

<details><summary>Checker's reasoning</summary>

I re-derived every number and they all hold. Two parts need a qualifier: the control sentence needs its checkpoint range, and the "<= 0.012" share depends on how the share is defined.

How I checked (inside the folder named below):
- I wrote my own rollout script, v3roll.py. It loads the judge file (never edited) and uses the judge's own isolate, Ctx, checkpoint_paths and load_checkpoint. The policy is deterministic with VecNormalize applied, and the judge's recorder is attached.
- Settings: iso.py with ISO_WT=scratch_copband (still clean at cf80ecb1), nice 19, one thread.
- I rolled s42 at all 7 checkpoints in three ways:
  - det: no noise.
  - full: the judge's joint-noise test, a = clip(clip(mu) + sigma z), z from default_rng(seed), seeds 3042-3061.
  - t3hold0: the same z draws with cmd[14] = 0 before the noise is added.
- At 7.3M, 8.4M, 10.5M and the final I also ran t3hold0nz: a[14] = 0 exactly, with no tail_3 noise.

Results:
1. The "full" rolls reproduce the judge files at all 7 checkpoints [measured]: same lengths and termination reasons, reward difference 0. Falls are 0, 0, 0, 9, 5, 9, 13, and the fall seeds match the judge files.
2. The deterministic tail_3 share (clip(mu) >= 0.9, window steps 200-999) is 0.7737, 0.6697, 0.7509, 0.9001, 0.5879, 0.0865, 0.7455 [measured]. This rounds to the claimed figures and matches the satroll frac90 at 4.2, 5.2 and 6.3M. For |cmd| the numbers are identical.
3. Spearman against falls is -0.1497 using average ranks for ties (n = 7, p = 0.75). Kendall tau is -0.16. So "-0.15" holds.
4. Under the noise, the share with the policy's tail_3 at >= +0.9 is 0.0116, 0.0096, 0.0008, 0.0017, 0.0080, 0.0004, 0.0004 [measured]. That is <= 0.012, so the claim holds. Two other definitions give higher numbers:
   - Counting |cmd| over the whole window, which includes the last second of each fall: up to 0.037 at 7.3M and 0.023 at 8.4M. In the last 100 steps of falls the policy commands tail_3 <= -0.9 (tail down) on 50% of steps at 7.3M and 43% at 8.4M.
   - Counting the applied command after noise: up to 0.059 at 4.2M.
   Either way the habit is gone against the deterministic 0.59-0.90. The window-mean command drops from 0.89-0.96 to 0.11-0.44.
5. Holding tail_3 at 0 under the same draws [measured]:
   - Final: 12 falls. These are the original 13 seeds minus 3052.
   - 10.5M: the identical 9 seeds.
   - Not in the claim: the identical 9 of 9 at 7.3M and 5 of 5 at 8.4M. These are the checkpoints where the deterministic habit is strong (0.90 and 0.59). At 10.5M it is only 0.09, so that test is weak on its own.
   - With a[14] = 0 exactly (no tail_3 noise), the falls stay at 13 (final), 9 (10.5M), 9 (7.3M) and 5 (8.4M), same seeds.
   - At 4.2-6.3M the hold gives 0 falls.
6. Control s42, from the stance4 judge JSONs [measured]:
   - 2.1M to 9.4M (8 checkpoints): panel saturation-flagged 35, 39, 40, 37, 39, 39, 37, 35 of 40. Joint-noise falls 0, 0, 0, 1, 0, 0, 0, 0; own-sigma falls 0.
   - 10.5M, 11.5M and the final: 23, 24 and 23 of 40 flagged, with 1, 2 and 8 joint-noise falls.
   So "35-40/40 flagged with 0-1 falls" is true only for 2.1-9.4M and needs that range. Its final failed K3 with 8 falls.
   I did not re-roll which actuator the control's flags name; the stance4 satdiag report says tail_3.

Outputs (4.5 MB): scratchpad/foot/stance5/k3diag/verify_3/ (v3roll.py, run.sh, run2.sh, p1-p4.jsonl, q1-q4.jsonl, npz/ with per-step tail_3 mu, act and q).

</details>


---

## The synthesis as written (before the checks)


2026-10-22. This is a report only: no verdict, criterion or run changes. It combines two passes:
- a file-only pass over the judge files and ledgers (`k3diag/files/`);
- a rollout pass that re-runs the judge's noise test with ablations (`k3diag/rollouts/`). It reproduces all 12 judged conditions exactly.

**[measured]** means counted or rolled. **[inferred]** means reasoning from those numbers.

## Answer

- **What the falls are [measured].** Slow topples, about 1-1.5 s from the start of the lean to the end of the episode. They start when the two feet drift apart fore-aft: the stagger reaches at least 81 mm, against 5-16 mm without noise.
- **Which noise causes them [measured].** Hip-pitch noise in its left-against-right (differential) mode. Tail and neck/head noise cause no falls at any checkpoint. Control s42 is the exception: its falls need knee noise.
- **s42's rise is not the tail habit [measured; mechanism inferred].** It tracks a step change between 6.3M and 7.3M. The policy largely stops correcting left/right hip-pitch differences: the gain goes from -3.2 to -0.75. Differential-noise falls go from 0 to 17/20.
- **For the replication [inferred]:**
  - The data support a seed-42 effect shared by both arms, not one that works through the tail.
  - They also allow a band trade-off that three seeds per arm cannot confirm.
  - s60's 3/20 is within luck.

**Joint-noise falls of 20 [measured, judge files].**

| run | 6.3M (K3) | 7.3M (report-only) | 8.4M | 10.5M | final (K3) |
|---|---|---|---|---|---|
| band s42 | 0 | 9 | 5 | 9 | 13 |
| band s52 | 0 | 0 | 1 | 1 | 0 |
| band s60 | 0 | 0 | 1 | 1 | 3 |
| control s42 | 0 | 0 | 0 | 1 | 8 |
| control s52 | 0 | 0 | 0 | 1 | 0 |
| control s60 | 0 | 0 | 0 | 0 | 1 |

- Band s42 also had 0 falls at 4.2M and 5.2M.
- The own-sigma test gives 0 falls everywhere except band s42 final (1/20, seed 3046, not examined).

## 1. What the falls look like

**The sequence [measured]:**
1. Under noise every run rocks on its pad edges, including runs that never fall. The CoP sits on the toe edge for 32-46% of loaded foot-steps and on the heel edge for 14-29%, against about 0 without noise.
2. In fallers the stagger then wanders to 0.1-0.15 m, and the loaded feet creep at 0.11-0.17 m/s (survivors' p99 is 0.08-0.10).
3. In the second before the lean starts, the feet stand split (one on its heel, one on its toe) or both toe-loaded.
4. Roll or pitch builds over about 1 s.
5. One foot unloads and slides at 0.6-1.4 m/s. The pelvis then drops below 0.70 m ('fallen') or pitches ('nosedive').

**Timing [measured]:**
- Lean start to termination: median 0.9-1.5 s per checkpoint (range 0.66-1.84 s).
- Band s42 episodes end at a median step of 576 (range 282-999); control s42 at 464.
- Spawn 3061 falls early (143-332 steps), and falls in all six runs at some checkpoint.

**Direction depends on the checkpoint [measured]:**
- Band s42 at 7.3M falls forward, nosedive-like (pitch -29 deg).
- Band s42 at the final leans forward, then topples sideways: to the right in 9 of 13, with the left foot unloaded in 10.
- Band s42 at 10.5M falls backward onto both heels, with a tail strike in 6 of 9.
- s60 in both arms topples sideways with the right foot off the floor.
- Control s42 is mixed.

The file pass, which only has window averages, labelled 26 of 36 band s42 falls 'forward' because the feet were toe-loaded. The per-step traces show that the final-checkpoint falls end sideways.

**Contacts and feet [measured]:**
- No head, knee or shin, thigh, metatarsus, torso or arm touches the floor before any fall.
- The tail touches the floor only as the last event of backward falls.
- The control arm's 2 head_contact endings come from the env's head-height check, not from a recorded floor contact.
- Every faller's sole tilts past 3.7 deg; no band survivor passes 3.0 deg. The median sole tilt when the lean starts is 6-11 deg.
- Ankle force is at most 0.17 of its limit everywhere, so the ankle is never torque-limited.

## 2. Which noise causes them

**Falls of 20 with the same noise draws, noise kept only on the named group [measured].** A blank cell was not rolled.
- "Differential" sets the left hip-pitch draw to minus the right one (L = -R).
- "Common" sets it equal to the right one (L = R).

| checkpoint | full | hip pitch only | legs without hip pitch | tail only | neck/head only | differential hip pitch | common hip pitch | full, tail_3 held at 0 |
|---|---|---|---|---|---|---|---|---|
| band s42 6.3M | 0 | 0 | 0 | 0 | 0 | 0 | | 0 |
| band s42 7.3M | 9 | 8 | 0 | 0 | 0 | 17 | 1 | |
| band s42 8.4M | 5 | 3 | 0 | 0 | 0 | | | |
| band s42 10.5M | 9 | 9 | 1 | 0 | 0 | 17 | 0 | 9 |
| band s42 final | 13 | 15 | 2 | 0 | 0 | 19 | 3 | 12 |
| band s52 final | 0 | 1 | 0 | 0 | 0 | 0 | 0 | |
| band s60 final | 3 | 2 | 0 | 0 | 0 | 3 | 0 | |
| control s42 final | 8 | 2 | 3 | 0 | 0 | 3 | 0 | |
| control s52 final | 0 | 0 | 0 | 0 | 0 | 0 | | |
| control s60 final | 1 | 0 | 0 | 0 | 0 | 0 | | |

**Band runs [measured]:**
- Hip-pitch noise is necessary and nearly sufficient. Hip roll, knee or ankle noise alone gives 0-1 falls.
- The differential mode is what matters: s42 falls 17-19/20 from 7.3M on, but 0/20 at 5.2M and 6.3M.

**Control s42 [measured].** Knee noise is necessary: the legs without knee noise give 1/20, and knee noise alone gives 4/20.

**Noise scale [measured]:**
- The K3 noise is 5-10x the policies' own late-training noise on hip pitch and knee, about 4.5x on ankle and 3x on hip roll.
- The band arm's own hip-pitch noise is about 30% below control's at the same training step in all three seeds (0.020-0.022 against 0.030 at 6.3M).
- K3's hip-pitch noise is therefore 6.9-10.2x the band policies' own, against 4.7-7.9x for control.

## 3. Band against control under the same noise

- **Band falls more [measured].** From 6.3M to the final, band runs fall 43 times and control runs 11, over 300 paired episodes per arm. Without s42 it is 7 against 2. At s42 final, all 8 control falls are among band's 13.
- **Static balance does not explain it [measured].**
  - Band policies stand more centred: their left/right CoP split is 0.04-0.34, against 0.26-0.98 for control.
  - Their toe and heel margins are equal or larger, with and without noise.
  - Under noise, edge occupancy, the toe margin (p5 of -0.04 to -0.05 everywhere) and ankle headroom are alike in both arms.
- **Recovery separates runs, not arms [measured].** Stagger sd and roll persistence under noise:

  | run | stagger sd (mm) | roll persistence |
  |---|---|---|
  | band s52 final | 15 | 0.04 |
  | control s52 final | 17 | 0.02 |
  | control s60 final | 10 | 0.00 |
  | band s60 final | 21 | 0.18 |
  | band s42, 7.3M on | 32-51 | 0.11-0.42 |
  | control s42 final | 41 | 0.35 |

  Band s52 behaves like the control finals, and both s42 runs lose stagger control. Band s60 sits between.
- **Kick probe [measured; not part of K3].** A deterministic differential hip-pitch kick of +/-0.3 for 0.2 s:
  - all three band finals fall 20/20;
  - control finals fall 0, 1 and 11/20;
  - band s42 at 6.3M holds (0/20);
  - the runs that hold counter the kick 2-5x more strongly.

  [inferred] This hints at a band effect but does not show one, with three runs per arm.

## 4. s42's rise and the tail habit

**Not the tail [measured]:**
- The tail habit does not track the falls:

  | checkpoint | 4.2M | 5.2M | 6.3M | 7.3M | 8.4M | 10.5M | final |
  |---|---|---|---|---|---|---|---|
  | deterministic tail_3 share at \|cmd\| >= 0.9 | 0.77 | 0.67 | 0.75 | 0.90 | 0.59 | 0.09 | 0.75 |
  | falls of 20 | 0 | 0 | 0 | 9 | 5 | 9 | 13 |

  Spearman -0.15.
- Under the noise the habit disappears: the share is at most 0.012, with a tail_3 mean command of 0.14-0.44.
- Tail-only noise gives 0 falls. Holding tail_3 at 0 under full noise still leaves 12 of 13 falls at the final and 9 of 9 at 10.5M.
- Within a checkpoint, spawns flagged for saturation and unflagged spawns fall at similar rates.
- Control s42 was flagged on 35-40 of 40 episodes through 9.4M with 0-1 falls.

**Not std, COM or posture [measured]:**
- Policy std declines smoothly in every run. Band s52 holds with a lower hip-pitch std than s42 (0.0139 against 0.0148).
- The deterministic COM, 53-70 mm ahead of MTP III, does not track the falls (Spearman -0.54).
- Leg posture moves by at most 0.02 rad.

**What does track [measured], from 6.3M to 7.3M:**
- Stagger sd under noise: 11 to 50 mm.
- Left-right hip-pitch sd: 0.020 to 0.057 rad (Spearman 0.95 with falls over 7 checkpoints).
- Differential-noise falls: 0 to 17/20.
- Correction gain d(mu_R - mu_L)/d(q_R - q_L): -4.4, -4.3 and -3.2 up to 6.3M, then -0.75, -0.69, -1.59 and -0.19 at the final.

**Training ledger [measured].** In training (own noise), band s42 falls stay at 0-0.4% per 524k-step bin through 9.96M, then rise to 2.1-5.4%. K3 sees the fragility about 3M steps earlier.

**Direction [inferred].** Fall direction seems to follow the checkpoint's stance rather than the tail command. 10.5M has the most rearward COM (53 mm) and almost no tail habit, and it falls backward.

## 5. What is measured and what is inferred

**Measured:**
- fall counts, the fall sequence, contacts and tilt;
- the group ablations;
- edge shares and ankle headroom;
- gains, correlations and the kick probe;
- the train ledger.

The rollouts match the judge files on episode length, termination, reward and sole readouts in all 12 judged conditions.

**Inferred:**
- That s42's policy stopped countering left/right hip-pitch differences, so noise builds up into a split stance that topples. The gain is closed-loop and descriptive (R^2 at most 0.36).
- That the kick probe hints at a band effect.
- That fall direction follows the stance, not the tail.
- That the band's lower own hip-pitch noise is linked to its extra falls. This was not tested.

**Unknown:** why training dropped the correction between 6,291,456 and 7,340,032.

**Own definitions:** the start of the lean, direction sectors, the pad edge (|CoP| >= 0.9), excursions, persistence and gain. With n = 20 per condition, differences of 2-3 falls are noise.

## 6. What it means for the replication, and next steps

**Which reading the data support:**
- **Seed-42 effect: supported.**
  - Both s42 runs end fragile. In the common-noise test at the final, band s42 falls 31/40 and control s42 30/40, against 1-6/40 for the other runs.
  - Both lose stagger control, through different joints: hip pitch for band, knee for control.
- **Tail side effect: not supported as the mechanism.** The habit is a seed-42 trait in both arms but plays no part in these falls. [inferred] It may be another symptom of the same training path.
- **Band trade-off: possible, not shown.**
  - For it: 43 against 11 paired falls; band s42 breaks about 3M steps before control s42; the band's own hip-pitch noise is about 30% lower in all three seeds; all three band finals fall to the +/-0.3 kick, against 0, 1 and 11 for control.
  - Against it: band s52 holds and matches the controls; band s42 at 6.3M resisted the same kick; without s42 it is 7 against 2; static margins and ankle headroom are alike.
- **Luck: fits s60, not s42.** s60's miss (3/20 against a limit of 2; control s60 1/20) is within binomial noise. s42's is not: it persists over four checkpoints and has a mechanism.

The replication therefore has to decide between "seed effect only" and "band trade-off". The readouts that separate them are the differential hip-pitch tests and each arm's own hip-pitch noise.

**Candidate next steps.** Reward, policy and physics changes need the maintainer's approval; telemetry does not.
1. **[telemetry]** Run the report-only readouts from `k3roll.py` on each replication checkpoint (6.3M, 7.3M, final): hip-pitch-only and differential noise, the +/-0.2 and +/-0.3 kick, the correction gain, own noise per joint and stagger sd. The judge is not edited and K3 is unchanged.
2. **[telemetry]** Find what changed in band s42 between 6,291,456 and 7,340,032: reward per term, band penalty activity, per-joint log_std and update statistics. Also measure what the band does under noise; only deterministic band readouts exist.
3. **[telemetry]** Examine the own-sigma fall at band s42 final (seed 3046).
4. **[policy]** Only if the replication shows the trade-off: keep the hip-pitch and knee action noise at least at the control arm's level, so band runs keep training against that noise.
5. **[policy]** Train with occasional differential hip-pitch or joint-space kicks at a fraction of the K3 noise. This is set up in the env.
6. **[reward]** Penalise stagger growth, or the left/right hip-pitch difference, beyond today's 45 mm tolerance; falls follow staggers of 81 mm or more. Without noise the stagger is only 5-16 mm, so training may rarely trigger this penalty; pair it with step 5.
7. **[physics]** No change indicated. The ankle has at least 83% torque headroom, no body part touches the floor before a fall, and the foot slide follows unloading.

## Files

Everything is under `scratchpad/foot/stance5/k3diag/`.

**File pass (`files/`):**
- `summary.json`
- `noise_episodes.csv.gz`
- `ledger_bins.csv`
- `extract.json.gz`
- `scripts/extract.py`, `scripts/analyze.py`, `scripts/build_summary.py`

**Rollout pass (`rollouts/`):**
- `scripts/k3roll.py` and the analysis scripts
- `tables/`: `ablation_matrix.csv`, `conditions.csv`, `episodes.csv`, `q4_trend.csv`, `policy_gain.csv`, `pulse.csv`, `q4_spearman_s42.json`
- `npz/`: 186 files, 209 MB. The ablation npz files (about 85 MB) can be deleted, since the tables hold their reductions.
- `logs/`

Both clones are clean. The judges were imported, never edited.
