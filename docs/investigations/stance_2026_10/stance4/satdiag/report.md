# stance4 s42: actuator saturation, offline diagnosis (report only, decision 5a)

2026-10-21. O = `scratchpad/foot/stance4/satdiag`.
Nothing here changes any verdict, criterion or training. Labels: [measured] means rolled or counted here; [inferred] means reasoning from those numbers.

## Answer

- **Actuator [measured].** `tail_3_pitch_act`, the distal tail pitch servo, commanded at +1 (ctrl +0.2094 rad, tail tip up). stance3 Run 2 had the same actuator and sign. Inside the gate window (steps 200-999), no other actuator reaches |a| >= 0.99 in any of 29 rolled checkpoints. In s42, no other actuator reaches 0.99 at any step (largest 0.836).
- **Onset [measured].** Between the placed checkpoints 1,048,576 (clean; tail_3 window-mean command +0.065) and 2,097,152 (17 of 20 flagged; +0.967). Eval ledger: 3 of 10 flagged at 1,835,008, 8 of 10 at 2,097,152, and the alarm (two evaluations in a row with >= 5 of 10) at 2,228,224. Within an episode the command ramps up from the settled pose and first reaches 0.99 at a median of step 216.
- **What precedes it [measured].** No contact, termination, noise or force signal. There are two early signs. From 1.57M, s42's eval centre of mass moved forward (about -8 mm, against -17 and -21 mm for s52 and s60). From 1.44M, its training return fell behind the sister runs. [Inferred, counterfactual-supported] Both are probably the tail command rising below the 0.99 rail: forcing tail_3 to 0.9 reproduces the forward shift with no flag.
- **Cost [measured, deterministic].** Holding tail_3 at 0 in the s42 policy gains +85.9 (2.1M) and +73.1 (final) per 800-step window, with no falls. Why training kept the habit is not measured.
- **s52 and s60 [measured].** In the window, no actuator reaches 0.9. Their tail_3 wandered and ended tail-down (-0.49, -0.52). Both passed through tail-up postures mid-run but never came near the rail. Nothing in the data explains which posture a seed ends in (2 affected of 5 L+T runs).

## 1. What saturates [measured]

- Every saturated window step of tail_3 is positive, in s42 and in Run 2.
- Share of s42 window steps at |a| >= 0.99: 0.779 (2.10M), 0.904 (3.15M), 0.972 (4.19M), 0.892 (6.29M), 0.520 (10.49M), 0.567 (final). The clipped window-mean command runs from +0.888 (final) to +0.998 (4.19M).
- The raw Gaussian mean sits beyond the clip: +0.91 to +1.11 as a window mean, +1.01 to +1.12 on pinned steps (+1.077 to +1.114 at 3.1-4.2M).
- **The command is pinned, not the joint or the motor.** At 4.19M the joint sits at -0.007 rad, 0.216 rad below its +0.2094 stop, and carries 0.26 of its 50 N·m limit (max 0.30). [Inferred from the XML] It cannot be force-clipped: |kp·(ctrl - q)| <= 60 × 0.419 = 25 N·m.
- So the ledger's force columns cannot see it. actuator_force_frac_max reads 0.272 before the habit and 0.283 during it, and it is held by head_pitch or tail_1, not tail_3. actuator_force_limited_fraction is 0 in every s42 episode.
- Outside the window (reset transient, steps 0-36), other actuators touch the rail briefly: tail_1_pitch_act in the s52 and s60 finals, and head_pitch_act once each in Run 2 at 4.19M and 6.29M. The gate does not count these.
- Other in-window events:
  - head_pitch_act at |a| >= 0.9 on 0.134 of steps in Run 2 at 8.39M (max 0.920).
  - neck_pitch_act at >= 0.9 for 702 window steps in one s52 8.4M episode.
  - Ledger: s52 flagged 2 of 10 at 7,864,320, and stance3 Run 3 flagged 5 of 10 once at 4,718,592. The actuator is unknown for both.

## 2. Onset [measured]

- **Ledger.** s42 flags 3, 3, 8 and 10 of 10 at 1.84M, 1.97M, 2.10M and 2.23M. 70 of 91 evaluations have >= 5 flagged, in three spells. After the first flag only 2 evaluations are clean (10,878,976 and 11,010,048).
- **Judged panels.** 35-40 of 40 flagged from 2.1M to 8.4M; 23-24 of 40 from 10.5M to the final.
- **Rollouts.** No placed checkpoint lies between 1,048,576 and 2,097,152; 1,572,864 is only on Drive.
- **Within an episode** (s42 2.1M, seeds 3042-3061):
  - The root settles by a median of step 16.
  - The command sits near +0.16 over steps 40-60, then climbs: +0.29 (60-80), +0.59 (100-125), +0.92 (200-225), +0.97 from step 400.
  - It first reaches 0.5, 0.9 and 0.99 at median steps 101.5 (20/20 episodes), 178 (17/20) and 216 (17/20, range 155-822).
  - [Inferred] This is a preferred resting posture reached by a slow ramp, not a reaction to a disturbance.
- **Feedback [inferred, probe-supported].** The 63-value observation has no previous-action channel.
  - Holding tail_3 at 0 drops the policy's own raw tail_3 mean from +0.998 to +0.244 (2.1M) and from +0.908 to +0.188 (final).
  - Swapping only the tail_3 joint-angle observation between conditions moves the mean from +0.998 to +0.290, and from +0.244 to +0.943. Zeroing the tail_3 velocity does nothing.
  - So the higher the tail, the higher the command, which lets the ramp run to the stop.

## 3. What goes with it, and what comes before it

**Posture [measured].**
- tail_3 joint at -0.007 to -0.016 rad (statue -0.093). Tail tip at 0.45-0.46 m (0.39 at 1.05M).
- tail_1 sits on its lower stop in every policy.
- tail_2 is at or near its stop through 4.2M. It lifts 0.014-0.026 rad off it in s42 from 6.3M, and 0.034 in the s52 final. So tail_3 is the main moving tail pitch joint, not the only one.
- The head is nose-up compared with s42's own 1.05M checkpoint and the statue. Skull pitch is -26.5 deg at 1.05M, -20.8 at 2.1M and -11.7 at the final; the head tip goes from 0.946 m to 0.98-1.08 m.
  - **Not specific:** s52 and s60 at 1.05M carry the head the same way and never saturate. Within a checkpoint, unflagged episodes hold the head as high or higher.
- Pelvis pitch goes -1.10, -1.44, -1.61 deg at 1.05, 2.1 and 4.2M, toward the statue's -1.54.
  - **Not specific:** s52 and s60 do the same by 4.2M.
- COM sits 61-69 mm ahead of MTP III over the saturated checkpoints (63.1 before onset; statue 63.8). It does not separate flagged from unflagged episodes within a checkpoint.
- Legs stay within about 0.014 rad, foot loads are 0.50/0.50, and stagger is within 2 mm.

**A raised tail moves the COM forward [measured; closed-loop counterfactual, tail_3 command only].**
- s42 at 1.05M: COM goes from -16.8 mm to -6.8 mm with tail_3 forced to 0.9 (0 of 20 flagged), and to -5.7 mm at 1.0 (20 of 20 flagged). Pitch goes from -1.10 to -2.63 and -2.81 deg.
- s42 at 2.1M with tail_3 held at 0.065: COM goes from -10.8 to -15.7 mm.
- s52 at 2.1M with tail_3 forced to 1.0: COM goes from -18.7 to -5.5 mm.

**Before onset [measured].**
- Training return, s42 minus the mean of s52 and s60 (131,072-step bins, labelled by start): +22 (1.05M), +15 (1.31M), -5.1 (1.44M), -30.3 (1.84M, first flag), -38.8 (2.10M, first evaluation >= 5/10), -43.8 (2.23M, alarm). The deterministic eval return gap was already -50 to -62 at 1.57-1.70M.
- Eval COM (ahead of the mean foot site): s42 moved from -16.9 mm (1.05M) to -11.6 (1.44M) and -8.1 (1.57M), then held at -8.0 to -8.7 through 1.97M. Its mean over that window was -8.9 mm, against -16.9 for s52 and -21.2 for s60. Its pitch was -1.57 deg against -0.79 and -1.11, and its ankle 0.99 against 1.44 and 1.94. The COM did not move further when the flags came.
- stance3 Run 2 against Run 3 shows the same pattern before Run 2's first full-horizon flag at 2.23M: COM -11.9 against -20.2 mm, pitch -1.25 against -0.57 deg, ankle 1.10 against 1.83. The return gap was -2.7, -42.5 and -56.4 at 1.44, 1.84 and 2.10M.
- [Inferred] A 0.9 command gives nearly the full forward shift without a flag. So the 1.3-1.7M drift is probably the tail command rising below the rail, not a separate trigger. The timing is unchecked; it needs gate_progress.npz or the 1,572,864 checkpoint.

**Not triggers [measured].**
- Contacts and terminations:
  - tail_contact falls end at 739,472 steps.
  - In 1.57-2.62M, s42 had the fewest early terminations of the five runs (3 of 1,052, all head_contact).
  - Its head_contact terminations at 1,420,912 and 1,928,292 fall before or inside the onset window. s52 had as many without saturating, so [inferred] they are not distinctive.
- Exploration noise: tail_3 sigma at 1.05M is equal across the runs (0.211, 0.209, 0.217). At 2.1M, after onset, it is 0.179, 0.194 and 0.202, and s42 is lowest in all 15 dimensions.

## 4. Cost under deterministic play [measured]

| s42 checkpoint | as trained | tail_3 held at 0 | gain |
|---|---|---|---|
| 2,097,152 | 3559.5 | 3645.4 | +85.9 (every episode, +53 to +143) |
| final | 3562.7 | 3635.7 | +73.1 (every episode, +9 to +126) |

- All 20 episodes run the full horizon either way, and none is flagged when tail_3 is held.
- Split at 2.1M: tail_home_pose +46.3, action_saturation +21.8, leg_home_pose +8.1, energy +3.9, neck_posture +2.3, lateral_lean +2.2, other +1.3. The first two follow largely from the hold itself.
- For reference, s42 at 1.05M scores 3643.7 and s52 at 2.1M scores 3649.3.
- Not shown: what the posture is worth under training noise.
- [Inferred, untested] Why it persists: beyond +1, the clipped command and the reward are flat in the mean. With the mean at +1.11 and sigma about 0.14, roughly 79% of training samples clip to +1 and carry no signal to lower it.

## 5. stance3 Run 2 [measured]

- Same actuator, sign and posture: +0.975 to +0.999 at 3.1-5.2M, with 17-20 of 20 flagged.
- Its onset was abrupt (0 to 9 of 10 at 2.23M), where s42's was gradual.
- It did not settle under the ramp. tail_3 window mean and judged-panel saturation failures:

| checkpoint | tail_3 | rolled flagged | judged sat. failures |
|---|---|---|---|
| 6.29M | +0.873 | 2/20 | 2/40 |
| 8.39M | +0.191 (tail down) | 0/20 | 0/40 |
| 9.96M | +0.906 | 12/20 | 28/40 |
| 10.49M | +0.866 | 0/20 | 0/40 |
| 11.53M | +0.896 | 1/20 | 6/40 |
| final | +0.841 (q -0.019; statue -0.093) | 0/20 | 0/40 |

- [Inferred] Its clean final is a command just under the 0.9 ramp with the tail-up posture kept, not a cure. Run 2 failed K1 in the stance3 record.

## 6. Why s52 and s60 did not

- **[Measured]** In the window, at 2.1M, 4.2M and the final, neither run puts any actuator at |a| >= 0.9.
- **[Measured] tail_3 path.**
  - s52: +0.065 (2.1M), -0.32 and -0.25 (3.1 and 4.2M), -0.03 (7.3M), +0.20 (8.4M), -0.49 (final).
  - s60: +0.142 (2.1M), +0.224 (4.2M), +0.29 (5.2M), +0.34 (6.3M), then -0.33, -0.50 and -0.52 from 8.4M.
  - Both passed through tail-up, head-up postures (head tip 1.03 m) without nearing the rail. Both ended tail-down and nose-down (head_pitch 0.56-0.57 rad, against a settled 0.22).
- **[Measured] tail_home_pose at the final.** It pays 0.191 (s42), 0.197 (s52) and 0.215 (s60) per step out of 0.25, so the costs are 0.059, 0.053 and 0.035. s52 is about level with s42. tail_3's own score is 0.19 for s42 against 0.45 and 0.49; in s52 and s60 that is offset by tail_2 and tail_1_yaw.
- **[Measured] Plant.** A statue sweep with only tail_3 commanded gives q = -0.178, -0.135, -0.092, -0.049 and -0.006 rad at a = -1, -0.5, 0, +0.5 and +1, i.e. q ≈ -0.093 + 0.086a. The servo (kp 60) works against a passive spring (80 N·m/rad about 0) and about 13 N·m of gravity.
- **[Inferred]** The spring is stiffer than the servo, so the full command range moves the joint only ±0.086 rad around -0.093.
  - s42's level tail (-0.007 to -0.022) is at the top of that range, so it can only be held on the rail.
  - s52's and s60's -0.135 is half the downward range.
  - What differs is the size of the excursion from home (+0.07 to +0.085 rad against -0.043), not gravity. An equally large tail-down excursion would also need the -1 rail.
- n = 2 affected of 5 L+T runs (s42 and Run 2, against s52, s60 and Run 3). Nothing in the data explains which posture a seed ends in.

## 7. Candidate fixes for the next round (none acted on)

| # | Fix | Type | For | Against |
|---|---|---|---|---|
| T1 | Name the actuator in the eval ledger (per-actuator window share at 0.99 and window-mean command). Until then, fetch `gate_progress.npz` (`action_dc_per_actuator`) at every check-in. | Telemetry | 5a's alarm must name the actuator. The ledger cannot, and its force columns cannot see it (0.27 to 0.28). | Changes no behaviour. |
| T2 | Watch for: window-mean tail_3 command above about 0.6; training return more than about 25 below the sister runs; eval COM drifting forward. | Telemetry | Every saturated checkpoint of both affected runs sits at +0.87 to +1.0, and the unaffected runs never exceeded +0.34. Both affected runs showed the lag and the drift 2-3 evaluations before the first flag. | n = 2; thresholds not validated [inferred]. A 0.9 watch would miss s42's final (+0.888, 13/20 flagged) and Run 2 at 11.53M (+0.896). |
| P1 | Bound the policy mean (a loss on \|mu\| > 1, or squashed actions). | Policy / training (approval) | The raw mean is +0.91 to +1.11 as a window mean and +1.01 to +1.12 on pinned steps. Beyond +1 the reward is flat in the mean, so the habit decays slowly (s42 from +1.11 at 4.2M to +0.91 at the final). | Addresses persistence, not onset. Touches all 15 channels. Untested. |
| R1 | Lower the saturation ramp start, or weight tail_3 more. | Reward (approval) | s42 pays roughly 0.02-0.03 per step in saturation. | s42 paid about 86 per window for 10M steps and kept the habit. Run 2 hovered at +0.84 to +0.91 around the 0.9 start, in and out of saturation; a lower start would likely move where it parks, not change the posture [inferred]. Adds a per-channel special case. |
| R2 | Stronger or one-sided tail_home_pose (tail-up charged more). | Reward (approval) | Largest term in the counterfactual (+46.3 of +85.9), though partly mechanical. | Already costs s42 0.059 per step (24% of the term). s52 pays 0.053 in the other direction, so a symmetric increase pushes every seed. No evidence a one-sided term breaks the habit. |
| Y1 | Plant: raise tail_3's kp, lower its passive stiffness or widen its ctrl range, so a level tail can be held off the rail; or revisit the tail_1 stop. | Physics (approval) | The spring (80) is stiffer than the servo (60): full command moves the joint only ±0.086 rad, and the joint sits 0.216 rad below its stop on 0.26 of its force. | A new physics revision resets comparisons. It makes the tail-up posture cheaper and hides it from the rail rather than discouraging it. |
| N0 | No training change; T1 and T2 only (the current 5a). | None | 2 of 5 L+T runs; the judge's rail catches it; Run 2's final was clean. | s42 never cleared (14/40 clean at the final). |

## 8. Cheapest next checks (report only)

1. Fetch `gate_progress.npz` and `diagnostics.npz` for s42, s52 and s60 from Drive. They give the tail_3 command at every evaluation, the exact onset, the training-time reward terms and algo_std. This tests whether the 1.3-1.7M drift is the tail rising below the rail.
2. Fetch the 1,572,864 checkpoint and roll it.
3. Run stochastic (training-noise) rollouts of s42 at 2.1M with and without tail_3 held, to see what the posture is worth under noise.

## Sources

- Rollouts: O/rollout/tables/ (summary_tables.md, checkpoint_table.txt, tail3_timecourse.txt, posture.csv, posture_compact.txt, per_actuator.csv, flagged_episodes.txt, reproduction_check.txt).
- Ledgers: O/ledger/timeline.csv and O/ledger/out_v2/.
- Counterfactual: O/synth/recheck_s42_tail3_zero.npz and O/synth/logs/recheck_ov0.log (script O/verify_0/roll_v.py); independent check in O/verify_4/.
- Verifier re-derivations:
  - O/verify_0/r2: 37 checkpoints.
  - O/verify_1/claimA: actuator, sign, transient hits.
  - O/verify_2/pinned: force.
  - O/verify_3/v3b: onset and crossings.
  - O/verify_5/out: posture, terminations, sigma.
  - O/verify_6/out: return lag, COM counterfactual.
  - O/verify_7/out: s52 and s60 paths, tail_home_pose, plant sweep.
- Alarm rule: `saturation_alarm` in foot/stance4/judge/score_stance4.py. Run 2 judged counts: foot/stance3/judged/stance3_armLT_s60/judge_*.json.

## Verification

**What was checked.**
- Eight verifiers re-rolled checkpoints through the judge's own loading path: Ctx("LTS"), `stance_report._load_policy` and `load_checkpoint`, deterministic policy, VecNormalize applied, nice 19, one thread.
- Per-episode returns and saturation flags match the judge files for every s42 and Run 2 checkpoint.
- The tail_3-at-0 counterfactual was reproduced independently.
- Ledger counts were recounted from eval_episodes.csv; the runs/ and fetch/final copies are identical.
- New probes: tail_3 command sweeps for the COM and the joint map, and an observation-swap probe for the feedback.

**What held.** Actuator and sign; share-at-0.99 numbers; onset bracket; ledger counts and the alarm at 2,228,224; pinned-command and force figures; the +85.9 / +73.1 counterfactual and its split; sigma values; return lag; early terminations.

**What changed from the synthesis draft.**
1. "No other actuator reaches 0.99 on any step" now holds only inside the gate window. Brief reset-transient hits exist outside it, plus the s52 (2/10) and Run 3 (5/10) ledger events.
2. Run 2 did not park under the ramp. It went in and out of saturation through 11.53M (28/40 and 6/40 judged failures at 9.96M and 11.53M). R1 loses that support.
3. The tail_home_pose figures were rewards, not costs; the costs are 0.059, 0.053 and 0.035.
4. s52 and s60 did not simply drift down; both passed through tail-up postures mid-run.
5. The plant asymmetry is not gravity. The joint map is linear and symmetric about home, with a passive spring stiffer than the servo; the difference is excursion size.
6. Head nose-up and pelvis nose-down are not specific to the habit, and tail_2 is not always on its stop.
7. The 0.99 crossing has a median of 216, range 155-822. The old 214 and 155-351 came from a file cut at step 399.
8. Corrected: the raw-mean range, the within-episode ramp numbers, the run count (2 of 5, not 2 of 6), sigma (equal before onset), and head_contact terminations before onset.
9. One verifier called 2.10M the alarm. score_stance4.py defines the alarm as two flagged evaluations in a row, so it is 2,228,224; 2.10M is the first evaluation with >= 5/10.
10. T2's tail_3 watch was lowered from 0.9 to about 0.6, since 0.9 misses s42's final (+0.888) and Run 2 at 11.53M (+0.896).

**Not re-checked.** The T2 COM threshold; the 79% clip arithmetic; whether the timing of the pre-onset drift matches the tail command (needs Drive data).
