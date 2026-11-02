# Phase-critic Colab round (PC, STANCE7): result

2026-11-02 (UTC). Six 12M-step training runs on Colab L4 runtimes; report only; nothing merged. The reading is the
pinned scorer's (score_pc.py 3c55bf21, run 19:12:31-19:12:44 UTC). The flagged deviation reading is the approved
deviation tool's (score_pc_dev.py efb37b20, run 19:13:15-19:13:22 UTC). Two independent checks found no wrong number
and no blocking process problem. The numbers check also found that the three switch-off runs are bit-for-bit
reproductions of the STANCE6 runs with the same seeds, trained on 2026-10-27/28: retrained from scratch, yet
byte-identical. [measured] cites a file and line or field; [inferred] marks reasoning.

**In short.**
- Pinned reading (the reading of record): line 4, no tail rail or park in 3 of 3 phase-critic runs, gate certificate
  passed in 3 of 3, no saturating habit elsewhere: "CONSISTENT WITH THE PHASE CRITIC REMOVING THE HABIT, NOT
  CONFIRMATORY".
- THE ONE TEST, pre-registered: NOT SEPARATED (p = 20/20). A FAIL, caused by the 17:44 UTC opening on 2026-11-01.
- The flagged deviation reading (approved before F2, not the reading of record): LOWER (p = 1/20).
- Every phase-critic TD (0.008-0.017) is below every switch-off TD (0.154-0.635). But the switch-off arm repeats
  STANCE6 byte for byte, so it adds no new sample, and its values were known when the test was pinned.
- Negatives: PHASE s60 fell in 9 of 910 training-evaluation episodes; PHASE s80's tail_2 command crept up to +0.055;
  M4, the critic readouts, M2 posture and D-1 to D-15 were not run (owed work, listed under "Open items").
- These bullets compress the sections below, where every claim carries its label and source.

Short names. ST = /home/user/work/pc_study. RUNLOG = S1/foot/stance4/live/run_log.txt; S1 and S2 are the two
scratchpad folders that ST/round/criteria.txt lines 38-39 define (line 40 defines RUNLOG). RUNLOG was read at 971
lines (sha256 89425ea9...); the scorers read it at 966 lines (8c05d9ba...). `<prefix>/` stands for the agent branch
prefix, left out under the rule against product names. PHASE = the three runs with the phase input; OFF = the three
switch-off runs. STANCE6 = the first r10 round's three runs (stance6_r10LTSB_s60/_s42/_s80). PS = the phase screen
(/home/user/work/phase_screen, its REPORT.md).

Words used (one line each):
- **F0 / F2:** F0 = the launch-time copy of five run files from Drive, for the launch check. F2 = the end-of-run copy
  of every file the reading uses, at least 6 h after the last run ended, each checked against the run's own manifest.
- **Ledger:** the run's own evaluation file (eval_episodes.csv): 10 episodes every 131,072 steps, 91 evaluations.
- **M1 (traced):** deterministic replays of a saved checkpoint that record every step's commands.
- **M3 (gate certificate):** the six-heading stance test on the final checkpoint, 40 seeded episodes per heading.
- **M4:** the offline K1-K4 stance judge (a clean test panel, a centre-of-pressure rail, falls under added noise, a
  roll rail). Its record at each final feeds the parent rule.
- **Parent rule ("stance3's parent rule"):** the pinned rule, first used in the stance3 round, that names the best
  POSITIVE final: most clean panel episodes in its M4 record, then fewest noise falls. It names; it never offers.
- **On-policy command:** the policy's mean tail_3 command on its own training states, last 8 updates (policy_probe.npz).
- **C1-C8:** the completeness checks: one launch, the manifest, every file equal to it, one session from step 0 in
  the round's configuration, the end, the ledger whole, the traces, and the manipulation check (is the phase in force,
  and is the actor blind to it).
- **PRESENCE:** a rule that can count a rail from the files of a run that was lost before it finished. 0 here.
- **M2:** the round's posture panel (head height, tail angles, pelvis pitch and similar) at the judged checkpoints.
  **D-1 to D-15:** the stance6 round's pre-registered report-only diagnostic readouts (a rail timeline and others).
- **TD (tail drive):** defined under "Design" below. **V − G:** the critic's value minus the real discounted return.

## The reading

### Pinned reading of record: line 4 (STATUS: FINAL)

> READING: line 4: NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, THE STANCE GATE CERTIFICATE PASSED IN 3 OF 3
> (GATE CERTIFICATE: final checkpoint, measured on a screen task; not certification), NO SATURATING HABIT ON ANOTHER
> ACTUATOR: CONSISTENT WITH THE PHASE CRITIC REMOVING THE HABIT, NOT CONFIRMATORY (descriptive: 3 phase-critic runs
> beside 3 concurrent switch-off runs; KNOWN LIMITS)

[measured: ST/f2/score/score_pc.txt line 357; "STATUS: FINAL" at line 365]

### THE ONE TEST, pinned: NOT SEPARATED (STATUS: FINAL; a failed pre-registered test)

> THE ONE TEST (phase vs the concurrent switch-off runs): NOT SEPARATED (p = 20/20)

[measured: score_pc.txt line 362]

- **Why it failed.** At 17:44 UTC on 2026-11-01 a PHASE run's notebook output was read before any OFF run had started
  (RUNLOG line 902). Under the pinned rule that is an "opening". Every OFF run launched after it is NOT BLIND and enters
  the test as missing, ranked lowest (criteria.txt lines 297-302, 353-361, 371-373). [measured: score_pc.txt line 40]
- **It could not have passed.** With all three controls ranked lowest, p is 20/20 whatever the six numbers.
  [measured: ST/amend2/AMENDMENT2.txt section 4 (iii); inferred: the rank argument]
- This fail stays a fail. The pinned reading is the round's reading of record.

### Flagged deviation reading: LOWER (approved before F2; not the reading of record)

> DEVIATION READING (flagged; not the pinned reading): THE ONE TEST computed as if the three OFF runs were blind;
> maintainer decision 2026-11-02 13:52 UTC, recorded before F2; the pinned reading above stands

> THE ONE TEST (phase vs the concurrent switch-off runs): LOWER THAN EVERY CONCURRENT SWITCH-OFF RUN (exact
> permutation p = 1/20)

[measured: ST/f2/score/score_pc_dev.txt lines 373 and 393; "[deviation] STATUS: FINAL" at line 396]

- **Decided** by the maintainer at RUNLOG line 949 (2026-11-02 13:52 UTC, "Pin both readings"). RUNLOG line 949
  records that this was before any F2 file and before any tail-command figure of any run was seen. **Approved**, with
  its exact bytes, at line 954 (15:59:28 UTC, "Approve all (Recommended)"). [measured]
  - Caveat 1: for the three files possibly viewed at 17:50-17:51 UTC (line 905), that statement names only the
    tail-command figure, and what those files showed is not on record. [measured: AMENDMENT2.txt lines 162-168]
  - Caveat 2: the control values were in fact known at the decision. The OFF runs equal STANCE6, whose TDs are
    printed in criteria.txt (lines 229 and 309). Only the PHASE values were unseen. [measured: criteria.txt; the
    identity below]
- **How it works.** The pinned scorer runs unchanged on a RUNLOG copy with only line 902's content removed (the removed
  line's sha256 is 0e30b26e...). [measured: score_pc_dev.txt lines 374-376]
- **What differs.** Only the blindness lines, THE ONE TEST line and one C1 listing: 6 output lines and 26 JSON values,
  0 unexpected. Exit codes 0 and 0. [measured: score_pc_dev.txt lines 442-443]
- **The wording fixed in advance for this case:** "the pre-registered test could not be passed because of a
  procedural slip; a reading decided after the slip, which assumes the slip had no effect on the OFF runs, reads
  LOWER." [measured: AMENDMENT2.txt lines 38-40]
- **New since the approval.** The OFF runs equal STANCE6 bit for bit (see the checks below). That supports the
  deviation's assumption: the look could not have changed the OFF runs. It also undercuts the claim that the control
  values were unseen at the pin: they were printed in criteria.txt. [inferred]

### Design, in brief (pinned 2026-11-01 03:07 UTC, RUNLOG lines 880-881)

- **The change tested.** The critic (the network that scores states) gets the episode phase through its own input from
  step 0: a ramp to step 200, a "settled" flag and a discount (criteria.txt lines 95-98). The actor (the network that
  picks actions) still reads the same 63 inputs.
- **Runs.** Six runs of 12,001,280 steps: T. rex, physics r10, arm LTSB, 4 environments, branch head 1dfe2327.
  - PHASE: stance7_r10LTSBpc_s60 / _s42 / _s80, switch on.
  - OFF: stance7_r10LTSBoff_s60 / _s42 / _s80, switch off; everything else the same (criteria.txt lines 49-52).
- **Waves.** "Phase first" (RUNLOG line 876). PHASE launched 03:26, 03:27 and 03:28 UTC on 11-01 (lines 885-887). OFF
  launched 18:12 (s60), 18:20 (s80) and 18:56 (s42) (lines 907, 911, 923). OFF s60 and OFF s42 each started in the
  runtime its same-seed PHASE run had freed (lines 906 and 913; 922 and 930) [measured]. For OFF s80 this is
  inferred: PHASE s80's manifest landed at 18:04:50 UTC (line 908), and no line names the runtime [inferred].
- **TAIL DRIVE (TD), the test's statistic.** For each of the run's 91 training evaluations, average the tail_3 pitch
  command over its 10 episodes; TD is the mean size of that number. 0 = the tail motor idle on average; 1 = held at
  full command. (criteria.txt lines 211-218)
- **THE ONE TEST.** An exact one-sided permutation test. Six TDs can be split 3 + 3 in 20 ways; p is the share of
  splits whose sum is at or below the PHASE sum. p = 1/20 only if every PHASE TD is below every OFF TD.
  (criteria.txt lines 289-296)
- **All six runs are COMPLETE** (C1-C8 PASS): one launch each, whole manifests, configuration pins, the ledger whole to
  the end, traces, and the manipulation check. [measured: score_pc.txt lines 44-319; ST/f2/measure/*/complete_post.txt]
  - PHASE: the phase input was in force at all 6 checkpoints checked, the actor's output changed by exactly 0.0
    when the phase was perturbed, and the critic still read the tail angle (observation 6). OFF: a stock critic with
    no phase input. [measured: score_pc.json runs.*.completeness.C8.value.results; e.g.
    f2/measure/stance7_r10LTSBpc_s42/complete_post.txt line 68]

## Per-run table

Each PHASE row sits above its same-seed OFF row. The pairing is in part by construction (same commit, seed and
starting networks) and in part [inferred from the OFF runs' determinism]; the criteria had pinned "Seeds do not pair
runs" (criteria.txt line 149), and THE ONE TEST is unpaired. See "Consequences" 3. TD is in tail-command units (a
fraction of full command). The J window is the 60 evaluations from 4,194,304 steps on.

| Run | TD (91 evals) | J-window TD | Tail class | Gate certificate (final; seeds clean of 40) | Outcome | Habit elsewhere | Alarms: ANY / TAIL (tail-flagged evals) |
|---|---|---|---|---|---|---|---|
| PHASE s60 | 0.0079 | 0.0053 | CLEAR | PASS (40/40) | POSITIVE | FREE | 0 / 0 (0) |
| OFF s60 | 0.3882 | 0.5777 | PARKED | PASS (40/40) | NEGATIVE | FREE | 0 / 0 (0) |
| PHASE s42 | 0.0173 | 0.0113 | CLEAR | PASS (40/40) | POSITIVE | FREE | 0 / 0 (0) |
| OFF s42 | 0.6353 | 0.8088 | RAIL (ledger, persistent) | **FAIL**: 2 hop-or-fall episodes at each heading (limit 1); 38/40 seeds clean | NEGATIVE | FREE | 22 / 22 (26); first 3,932,160 |
| PHASE s80 | 0.0122 | 0.0132 | CLEAR | PASS (40/40) | POSITIVE | FREE | 0 / 0 (0) |
| OFF s80 | 0.1543 | 0.2179 | CLEAR | PASS (40/40) | POSITIVE | **SATURATING** neck_pitch; DRIVE ONLY head_pitch | 6 (neck_pitch, 5,505,024-7,077,888) / 0 (0) |

[measured: score_pc.txt lines 321, 325, 329 (PHASE), 334, 338, 342 (OFF), 345-346 (OFF s80 habits); full-precision
PHASE TDs at line 362, OFF TDs at lines 334, 338, 342; seeds clean at
ST/round/panels/<RUN_ID>/final/m3/stance_heading_report.txt line 10; OFF s42: 38/40 seeds clean passes its own bar
(LCB95 0.8508 against 0.80, line 10), but 2 hop-or-fall episodes at every heading exceed the limit of 1 (lines 15-20,
63); its two unclean seeds 3079 and 3080 had actuator saturation 0.539 and 0.515 against a limit of 0.10 (lines
76-77)]

- **RAIL:** the tail motor held at its limit in training evaluations, raising TAIL ALARMs. A TAIL ALARM is the second
  of two evaluations in a row with at least 5 of 10 episodes over the saturation bar. ANY ALARM is the same for any
  motor.
- **PARKED:** the tail_3 command averaging +0.60 or more without a rail. **CLEAR:** neither.
- **Gate certificate:** the six-heading stance test on the final checkpoint, 40 seeded episodes per heading. It is a
  screen measurement, not certification.
- **Habit elsewhere:** SATURATING = another motor at its limit (alarms or traces). DRIVE ONLY = another motor's mean
  command 0.60 or more in size without saturation. FREE = neither.

Report only (these decide nothing):

| Run | tail_3 mean command, traced | Ledger tail_3 maximum (step) | Evaluations with ledger tail_3 at or above 0.60 | On-policy tail_3, last 8 updates | Mean return, 910 eval episodes | Falls | Clean rows of 910 |
|---|---|---|---|---|---|---|---|
| PHASE s60 | -0.006 | 0.021 (4,849,664) | 0 | -0.007 | 4442.7 | **9** | 892 |
| OFF s60 | +0.642 | 0.894 (8,650,752) | 37 | +0.536 | 4412.6 | 1 | 864 |
| PHASE s42 | +0.019 | 0.030 (7,864,320) | 0 | +0.008 | 4481.0 | 0 | 899 |
| OFF s42 | +0.915 | 1.000 (4,587,520) | 57 | +0.813 | 4377.7 | 1 | 645 |
| PHASE s80 | +0.019 | 0.031 (9,961,472) | 0 | +0.017 | 4482.3 | 0 | 903 |
| OFF s80 | +0.505 | 0.616 (11,796,480) | 1 | +0.517 | 4415.8 | 1 | 778 |

[measured: score_pc.txt lines 322-324, 326-328, 330-332, 335-337, 339-341, 343-347 (MEAN_a, early signs, Q4,
on-policy tail_3)]

- **Separation [measured, my arithmetic on score_pc.txt lines 334, 338, 342, 362].** Every PHASE TD is below every OFF
  TD. The largest PHASE TD (0.0173) is one ninth (1/8.9) of the smallest OFF TD (0.1543). Within each seed the PHASE TD
  is 2.0%, 2.7% and 7.9% of the OFF TD (s60, s42, s80); put the other way, OFF TD is 49.1, 36.7 and 12.6 times PHASE
  TD.
- **Every other tail measure splits the arms the same way:** the J window, the count of evaluations at or above 0.60
  (0/0/0 against 37/57/1), and the on-policy command. [measured: the tables above, from score_pc.txt; the numbers check
  re-derived the J window and the counts, but not the PHASE on-policy values]
- **No early signs in PHASE.** The ledger tail_3 command never reached 0.30 in any PHASE run, and no evaluation had a
  tail over the bar. [measured: score_pc.txt lines 322, 326, 330]
- **Other tail joints: small in PHASE, with one slow rise.** PHASE tail_1 and tail_2 mean commands at the traced
  checkpoints (MEAN_a) stay within ±0.051 (largest: s80 tail_2 +0.051) [measured: score_pc.txt lines 323, 327, 331].
  But PHASE s80's tail_2 ledger command rose from +0.014 at 4,194,304 steps to +0.055 at 11,927,552, up at 4 of 5
  steps (+0.014, +0.016, +0.015, +0.022, +0.034, +0.055) [measured: score_pc.txt line 332, "guard"]. That is the drift
  the phase screen's guard warned of (PS SEC-TAIL2, RUNLOG line 857; criteria.txt lines 81-82 and 338). It is far
  below OFF, whose runs reach MEAN_a +0.168 (s60 tail_2) and +0.186 (s80 tail_1) [measured: lines 336, 344].
- **Returns.** PHASE was higher in all 3 pairs, by +30.1, +103.4 and +66.5. [measured: Q4, score_pc.txt lines 324,
  328, 332, 337, 341, 347; my arithmetic]
- **Falls, a negative (report only): none at the final gate; one PHASE run fell 9 times in training.** PHASE s60 had
  9 falls in its 910 training-evaluation episodes, against 1 in each OFF run and 0 in the other two PHASE runs
  [measured: score_pc.txt lines 324, 328, 332, 337, 341, 347]. The falls did not stop early in training. They came at
  3,407,872; 3,538,944; 3,801,088; 4,194,304; 4,587,520; 7,733,248; 9,830,400; 10,616,832 and 11,665,408 steps: 6 of
  the 9 at or after 4,194,304, the last at 11,665,408. The episodes lasted 150-378 steps; 7 of the 9 ended between
  step 211 and step 378 of the episode, soon after the step-200 switch-on [measured: ST/final/verify_report/evidence/
  falls.txt; ST/final/next_steps_work/short_eps.out]. Whether the two are linked is not tested [inferred]. The two at
  3,407,872 and 3,538,944 steps (lengths 192 and 150) left a blank tail cell, which TD skips as the criteria say
  [measured: numbers check, journal result.checkA (a)]. Its final checkpoint still passed the gate 40/40. Not looked
  into further: an open item.
- **The pinned plug-in figure (report only):** "if the phase critic changed nothing, no RAIL in 3 runs has chance (1 -
  1/3)^3 = 0.296". [measured: score_pc.txt line 359] So "no rail in 3 of 3" alone would happen about 3 times in 10 with
  no effect. [inferred]
- **Statistic A (report only):** runs with any ALARM, PHASE 0 of 3 against OFF 2 of 3, one-sided Fisher p 0.2.
  [measured: score_pc.txt line 363]
- **No parent named.** Line 4 asks the parent rule to name the best POSITIVE final. It printed "not named: an input is
  absent": there is no M4 judge record at any of the three PHASE finals. No checkpoint is offered. [measured:
  score_pc.txt line 361] M4 at the finals was owed F2 work once line 4 read (criteria.txt lines 269-272); see "Open
  items".

## What the independent checks found

Both checks ran on 2026-11-02 in workflow wf_cd81846d-bc1, read only, on the F2 copies. [measured: its journal,
result.checkA and result.checkB; RUNLOG line 970]

### Numbers (ST/f2/check_numbers, 19:15-19:24 UTC): no wrong number

- All 6 TDs and all 6 J-window TDs match the scorer to the last digit. [out/td_check.json]
- Its own permutation code gives 20/20 NOT SEPARATED with the pinned blindness, and 1/20 LOWER as if blind.
  [out/one_test_check.json]
- Its own reduction of the traces and its own ledger reading give line 4 (r 0, u 0, p 0, g 3, hs 0, hd 0). The OFF
  classes, alarms and counts come out as printed. [out/m1_check.json, out/ledger_check.json]
- Statistic A, the plug-in figure and the replication totals recompute.
- Info: an exact-fraction TD mean differs from the tools' value in the last digit in a few cases. The test result is
  the same either way.
- **Not checked by it:** the gate certificates were not re-simulated (PHASE verdicts were taken from m3_summary.json;
  OFF verdicts were cross-checked only through byte-identity with STANCE6). Also not checked: C1-C8 and the
  manipulation records, the M4 parent rule, the PHASE on-policy tail_3 values, and the deviation tool's other JSON
  differences.

### Process (ST/f2/check_process, 19:15-19:30 UTC): no blocking finding

- **Pinned files unchanged:** criteria e612d523, APPROVED ce886520, F2_PINS ed3e29c5, COLAB_STEPS 98abcaa6, all 14
  tools, and the approved amendment files; 1338 of 1338 amend2 sums OK. [pinned_end_sha256.txt,
  amend2_sums_check.txt]
- **F2 order:** six F2-START lines at 16:39 UTC (RUNLOG lines 960-965), 6.12 h after the last ending note (10:32 UTC).
  F2 rows ran 16:45:01-17:32:44 UTC: 410 copies, each equal to its run manifest. [manifest_check.out] Drive was used
  read only: 410 downloads and folder listings, no write tool. [drive_scan.out line 40, from the agents' transcripts;
  journal result.checkB item 2]
- **Inputs:** 78 measurement records (6 runs × 13), each bound to its own manifest-checked pair. [inputs_check.out]
- **Measuring copy:** seal intact (tree digest 9417b765..., 1410 files; archive babc6812...), no bytecode.
  [seal_check.out, archive_check.out]
- **Writes:** none outside the allowed folders, none in the repositories. [outside_writes.txt, wider_writes.txt,
  repo_writes.txt]
- **Deviation run as designed:** line 902 blanked with its line feed kept; 6 lines and 26 values differ, none
  unexpected. [dev_diff_check.out]
- **Concurrency:** at most 2 simulation processes at once. [concurrency_check.out]
- **Minor M1:** Python ran with -I but without -B in the six fetch helper scripts and in four inline reads. No bytecode
  file was written anywhere after 16:39 UTC. [py_flag_scan.out; journal]

### MEASURED: the three OFF runs are bit-for-bit reproductions of STANCE6's same-seed runs

The OFF runs were trained again from scratch on 2026-11-01/02. Nothing was copied. Yet their outputs equal STANCE6's.
Only the numbers check found this; the process check does not mention STANCE6 [measured: journal result.checkA,
result.checkB].

Evidence:

- **Same records.** For all three seeds, the OFF run's eval_episodes.csv, train_episodes.csv, evaluations.npz,
  k3_panel.csv, gate_progress.npz, replay CSVs and stance_heading_report.txt are sha256-equal to STANCE6's same-seed F2
  copies. They came from different Drive files. [measured: check_numbers, journal result.checkA]
- **Same networks.** In the final checkpoints the policy weights, optimizer state, torch variables and system info have
  equal CRC32 and size. num_timesteps (12,001,280) and the update counts (11,265 / 10,607 / 11,052) are equal. Every
  policy-probe array is equal except its wall-clock build time. [measured: check_numbers/out/zipcrc_off_vs_stance6.out,
  out/zipdata_off_vs_stance6.out]
- **What is not equal in the final zips.** 5 of the 43 keys in the zip's "data" member differ, in all three seeds:
  ep_info_buffer, policy_class, rollout_buffer_class, start_time and tensorboard_log [measured:
  zipdata_off_vs_stance6.out lines 2-6, 12-16, 22-26]. policy_class and rollout_buffer_class differ only in memory
  addresses inside printed function names [measured: ST/final/verify_report/evidence/identity_spot.txt, s60].
  start_time and tensorboard_log carry the wall clock and the run's name; ep_info_buffer holds recent episodes'
  wall-clock times [inferred]. None of them is a learned weight.
- **Really new runs.** Commit 1dfe2327 against 8737cd3d; different run ids; start times 2026-11-01 against 2026-10-27;
  different wall-clock totals. [measured: the same two files; journal]
- **Spot checks.** The coordinator (RUNLOG line 971) and this report (sha256sum, 19:37-19:42 UTC) both found eval_episodes.csv
  copy 1 of OFF s60 / s42 / s80 equal to STANCE6's (1821fd2e..., d1bcb789..., b677ce76...), while PHASE differs
  (d7022a3e..., 92135e33..., 7f939be3...). This report also found OFF s42's gate certificate text byte-identical to
  STANCE6 s42's (cmp). [measured]

Consequences:

1. **No new OFF sample.** The OFF arm repeats STANCE6; it adds no run independent of it. [inferred] The OFF training
   took 44.6 L4-hours (s60 14.9 h, s80 15.0 h, s42 14.7 h, session start to manifest). What it bought is a measurement:
   same-configuration training is deterministic on this setup. [measured: start times in score_pc.txt C4 values,
   manifest times at RUNLOG lines 941, 943, 945; my arithmetic] The reproduction spans two sets of Colab sessions, on
   2026-10-27/28 and 2026-11-01/02. All nine runs (STANCE6's three and this round's six) record the same GPU and
   software fields: stage_config.json's gpu block reads NVIDIA L4 and CUDA 13.0, and provenance.json reads Python
   3.13.16, NumPy 2.1.3 and the same Linux kernel string [measured: ST/final/verify_report/evidence/env_fields.txt;
   ST/final/next_steps_work/env_fields.out]. No file records the torch, cuDNN or GPU driver version [measured: same].
2. **The OFF TDs were known at the pin.** STANCE6's TDs 0.388227 / 0.635343 / 0.154339 are printed in criteria.txt at
   lines 229 and 309. So "the six runs' values are unseen at the pin" (criteria.txt lines 305-307, 311 and 507;
   printed at score_pc.txt line 362) is false for the OFF three. TD's form was chosen with these exact values in view.
   The PHASE values were unseen. [measured]
3. **Paired, in part.** Each PHASE/OFF pair shares commit 1dfe2327, the seed and the starting networks (A5, shown on
   CPU: criteria.txt lines 112-125) [measured]. A6 shows only that the switch-off path equals 8737cd3d (lines
   126-134); the PHASE runs take the switch-on path. The recorded GPU and software fields are equal; the Colab image
   itself is not measured (lines 497-498) [measured]. [inferred] So each pair's recorded settings differ only by the
   critic's phase input. That does not make one pair a controlled experiment: training is deterministic but
   sensitive, so any change, even an input that carries no information, would send a run down a different path. No
   sham-input arm was run, and PHASE's own determinism was not measured. The criteria had pinned the opposite premise:
   "Seeds do not pair runs" (line 149).
4. **The peek could not have changed the OFF outcomes.** They equal runs trained on 2026-10-27/28, before any PHASE run
   existed. [inferred from the identity]
5. **A premise of the criteria is contradicted:** "GPU training is not bit-reproducible" (criteria.txt line 149), "GPU
   nondeterminism can move any single run" (lines 83-84) and "GPU nondeterminism within today's image remains" (lines
   315 and 496). [measured: the identity] The numbers check cites lines 313 and 500 for the last one; the text stands
   at 315 and 496. [measured: grep of criteria.txt] The premise was never measured. The stance6 round's evidence file
   labels it "[inferred, as REPL]" (S2/r10round/final_lean/evidence_r10round_v12.txt line 256), but the stance6 review
   labels the same claim "[measured, EVID l250-l258]" (S2/r10round/f2/REVIEW.txt line 511) [measured]. That label is
   wrong, and the records should correct it. An inferred premise is now contradicted by a measurement.

## What it means [inferred]

- **In 3 of 3 PHASE runs, no run formed the tail habit.** With the critic told the episode phase from step 0, no run
  formed a rail or a park in 12M steps. The tail_3 command stayed near zero: TD 0.008-0.017, against 0.15-0.64 in
  the same-seed OFF runs.
- **Same-seed pairs are close to, but not, a controlled experiment.** Each pair's recorded settings differ only by the
  phase input, and the OFF configuration trains deterministically. In every seed the OFF run had a habit (a rail, a
  park, or a saturating neck habit with tail drive 0.154) and the PHASE run had none. But a deterministic system can still be
  chaotic: any change, even a sham input, would send a run elsewhere, so one pair alone shows little, and no sham arm
  was run. All three pairs moved the same way, to near-zero tail drive. That is a pattern across 3 seeds, not a
  measured per-seed effect of the phase information.
- **It fits the explanation the phase screen tested.** The explanation: four stance reward terms switch on at step 200
  (PS REPORT.md line 95; RUNLOG line 804); a critic that cannot see time reads the slowly creeping tail angle as a
  clock; the policy gradient then pays for raising the tail. PS found the tail climb stopped from s42's 2.10M state on
  CPU; the explanation itself is [inferred] there (PS REPORT.md lines 91 and 94-97). This round's tail results fit
  it, from scratch on GPU for 12M steps, but its mechanism readouts were not run.
- **Stance cost: none at the final gate; one PHASE run fell 9 times in training.** All three PHASE finals passed the
  gate 40/40, and no habit formed on another motor. Mean returns were higher than the same-seed OFF run in all 3
  pairs. But PHASE s60 fell in 9 of 910 training-evaluation episodes, against 0 or 1 in the other five runs, and its
  falls went on until 11.7M steps. That is not explained.
- **The formal case is weaker than the raw numbers.**
  - The pre-registered test failed on procedure.
  - The flagged deviation reads LOWER, but its controls were values known at the pin.
  - A paired sign test on 3 pairs cannot go below p = 1/8 (criteria.txt line 316).
  - The pinned plug-in says "no rail in 3" happens 30% of the time with no effect.
  - Statistic A (runs with any alarm), PHASE 0 of 3 against OFF 2 of 3, reads one-sided Fisher p 0.2: not significant.
  - The strength lies in the size of the gap and in the same direction across 3 seeds. Both were weighed only after
    the data.
- **Post hoc, not pre-registered, deciding nothing:** if a rail or a park happened in 2 of 3 runs, as in the OFF arm,
  0 of 3 would happen with chance (1/3)^3 = 0.037. A rate from 3 runs is very rough. [inferred, my arithmetic]

## What it does not show

- **Other seeds.** Only seeds 60, 42 and 80, chosen with their earlier outcomes known; 3 runs measure no rate
  (criteria.txt lines 492, 507-508).
- **Other images or hardware.** Determinism was measured only for the switch-off configuration, across Colab L4
  sessions on 2026-10-27/28 and 2026-11-01/02 whose recorded GPU and software fields are equal (NVIDIA L4, CUDA 13.0
  in stage_config.json's gpu block; Python 3.13.16, NumPy 2.1.3). No file records the torch, cuDNN or GPU driver
  version, and the image itself is not measured (criteria.txt lines 497-498, which name provenance.json only).
- **Certification.** The gate is a screen-task measurement on the final checkpoint only, not certification.
- **A chosen checkpoint.** M4 was not run at the PHASE finals, so the parent rule named nothing and nothing is offered
  (score_pc.txt line 361). This is owed work, not only a limit: see "Open items".
- **Walking.** Not tested. The phase features belong to this stage (settle step 200, discount 0.98); walking stages
  need their own (criteria.txt lines 513-514).
- **Main's configuration.** On main only one stance term, stance_width, is gated (#655). This branch switches on four
  stance terms at step 200. Whether the habit, or this fix, carries over to main is untested.
- **That it is the phase.** No sham-input arm: "any extra critic input" is not separated from "the phase"
  (criteria.txt line 513).
- **The mechanism in these runs.** The report-only critic readouts (tail-angle slope, V − G by step bucket, loop gain;
  criteria.txt lines 332-337) were not run in F2 (RUNLOG line 967).
- **Later or in between.** A rail after 12M steps, or between evaluations, is not seen; the gate reads the final only
  (criteria.txt line 515).

## Open items (work the criteria call for that was not done)

Each is a gap in this round's record. None changes the pinned reading, and none is a reason to rerun anything.

1. **M4 at the three PHASE finals.** The criteria make it F2 work once line 4 reads (criteria.txt lines 269-272). F2
   did not run it (RUNLOG line 967 lists C8, M1 and M3 only), so the parent rule printed "not named" [measured]. The
   pinned operating steps for F2 (COLAB_STEPS.txt lines 264-269) list the manipulation check, M1, M3, the completeness
   tool and the scorer, and no M4; the process check had no M4 item [measured]. [inferred] The slip sits in a gap
   between two pinned texts, and the independent process check did not catch it. It stays a failure of process.
2. **The critic readouts** (tail-angle slope, V − G by step bucket, loop gain; report only; criteria.txt lines
   332-337). Not run [measured: RUNLOG line 967; panels hold only c8, m1 and m3].
3. **M2 posture at the J checkpoints, and D-1 to D-15 "as computable"** (report only; criteria.txt lines 329-331).
   Neither appears in score_pc.txt [measured: grep].
4. **PHASE s60's 9 training-evaluation falls.** Not looked into (their recorded reasons and the in-run K3 panel).
5. **PHASE s80's slow tail_2 rise** (+0.014 to +0.055). Seen in the ledger only.
6. **The stance6 review's wrong "[measured]" label** on the non-reproducibility premise (Consequences 5), for the
   records to correct.
7. **The records.** PC has no records PR yet; the unpushed draft holds a placeholder (RUNLOG lines 871-872).

## Process notes

### Timeline (UTC; RUNLOG line in brackets)

| When | What |
|---|---|
| 10-31 20:54 | Maintainer switches to a Colab run with a separate critic-only input [867] |
| 10-31 20:57 | Build workflow wf_9b7ee473-01d launched [869] |
| 11-01 01:12 | Build done; re-review READY; commit trailers reworded; head 1dfe2327 [873] |
| 11-01 01:15 | Maintainer adds 3 switch-off runs as the test's control [874] |
| 11-01 01:41 | Maintainer: 3 runs at a time, "Phase first" [876] |
| 11-01 02:57 | Amendment ready to pin (blocking B1 fixed by THE BLIND PERIOD rule) [878] |
| 11-01 03:07 | PIN: APPROVED ce886520 [880-881] |
| 11-01 03:26-03:28 | PHASE launches [885-887]; launch checks no-fault at 03:50 [893-895] |
| 11-01 17:44 | SNAPSHOT: PHASE s60 notebook output read [902] |
| 11-01 17:47 | Maintainer: "You can ignore the results for now"; decision deferred [903] |
| 11-01 18:01 | Possible views of three PHASE s60 files at 17:50-17:51 noted, inferred [905] |
| 11-01 18:12 / 18:20 / 18:56 | OFF launches s60 / s80 / s42 [907, 911, 923]; order deviation [910] |
| 11-01 18:56 | THE BLIND PERIOD ends [924] |
| 11-01 19:15 | All six launch checks no-fault [933] |
| 11-02 10:32 | All six runs ENDED; none lost, stopped or relaunched [947] |
| 11-02 13:52 | Maintainer: "Pin both readings" [949] |
| 11-02 14:45-14:47 | Deviation review returned; disk deletions [951-952] |
| 11-02 15:59 | Maintainer approves the deviation reading and F2 pins [954-957] |
| 11-02 16:39 | Six F2-START lines [960-965] |
| 11-02 16:45-19:10 | F2 fetch, then measurements [967] |
| 11-02 19:12-19:13 | Pinned scorer, then deviation tool (score_pc_rc.txt, score_pc_dev_rc.txt) |
| 11-02 19:15-19:30 | Two independent checks [970] |
| 11-02 19:32-19:33 | Results noted; this review started [967-971] |
| 11-02 19:56-20:20 | Adversarial verification of this report and its two companions (ST/final/verify_*) |
| 11-02 from 20:14 | Finalizing: each finding checked against its source (ST/final/FINALIZE_NOTES.txt) |

### The 17:44 opening, and lines 905 and 910

- **What was shown (line 902).** The maintainer pasted the PHASE s60 notebook output into chat. It held in-training
  evaluations at 11,796,480 and 11,927,552 steps (reward, length), the final-model evaluation, the best-model step
  (6,815,744), the selected checkpoint and its 30-episode evaluation. No tail command, TD or ledger figure. [measured:
  RUNLOG line 902; what was on screen is the operator's record]
  - [inferred] "No TD shown" is not quite "no outcome information shown". The opening showed PHASE s60's rewards, and
    returns differ between the arms (PHASE higher in all 3 pairs). So it carried a weak hint about the outcome. The
    hint is weak: the s60 gap is only +30.1.
- **Line 903 (17:47).** The maintainer: "So I was just trying to confirm if there should be more results in the google
  drive notebook. You can ignore the results for now". The figures were not used or repeated. [measured]
- **Line 905 (18:01).** Drive metadata showed "viewed" times on three PHASE s60 files at 17:50:58-17:51:20 UTC
  (stance_gate_probe_constant.txt, gate_verdict.json, stage_summary.txt): after the opening, before every OFF launch.
  RUNLOG records this as inferred, not established. They may be views in the Drive web page or reads by the notebook.
  No SNAPSHOT line was added, because the 17:44 opening already governed that run. What they showed is not on
  record. [measured: RUNLOG line 905; AMENDMENT2 section 4 (i)]
  - gate_verdict.json holds the run's own gate verdict. If a person viewed it, more than line 902 was seen before the
    OFF launches. [inferred] With the OFF runs equal to STANCE6, none of it could have changed them. [inferred]
- **Line 910.** See the order deviation below.

### The deviation decision and its approval

- The keep-or-amend decision was left to the maintainer at 17:47 (line 903) and was still open at lines 924, 947 and
  948. [measured]
- **13:52 (line 949):** "Pin both readings". The pinned reading stays exactly as pinned; a second, flagged reading
  computes THE ONE TEST as if the OFF runs were blind. Both were to be pinned, reviewed and approved before F2.
- **13:56 (line 950):** build wf_8a4486a0-a57 started, with an independent adversarial review.
- **14:45 (line 951):** review: 1 blocking text finding (AMENDMENT2 must name RUNLOG lines 905 and 910) and 5 minor.
- **15:58 (line 953):** fixer and re-review: every finding fixed; 35 tests pass; 30 of 30 mutants killed; re-review
  READY with two minor text points, fixed in AMENDMENT2 only.
- **15:59:28 (line 954):** "Approve all (Recommended)". Pinned read-only: AMENDMENT2 c1ec55be, score_pc_dev.py
  efb37b20, amend2 SHA256SUMS 1a4786e8; F2_PINS ed3e29c5 (line 955); APPROVED2 0c034e90 (lines 956-957).
- **Its own stated limit:** it was decided about 20 h after the opening, by people who knew what the opening showed. So
  it is not pre-registered relative to the opening. [measured: AMENDMENT2 section 4 (iv)]

### The order deviation (s80 before s42)

- The Colab steps start the OFF runs in seed order s60, s42, s80 (COLAB_STEPS.txt lines 75-77). OFF s80 started at
  18:20, before OFF s42 at 18:56. RUNLOG recorded it as a deviation at once (line 910). [measured]
- No scoring rule reads the order. The start gaps are printed: 886, 929 and 892 minutes for s60, s42 and s80.
  [measured: score_pc.txt lines 41-43]
- [inferred] PHASE s80 freed its runtime before PHASE s42 (manifests 18:04:50 and 18:37:39 UTC; lines 908, 919).

### Disk deletions

- **14:45 UTC (line 951).** F2 needed about 6.5 GB; 1.37 GiB was free at 14:41. The coordinator deleted reviewer A's
  scratch in ST/amend2/review (mutant tool copies, test roots, probe work folders). DELETED_SCRATCH.txt lists 2,381,764
  KiB. Findings were kept in the workflow journal and review/*.txt. Free after: 3,914,477,568 B. [measured]
- **14:47 UTC (line 952).** The maintainer chose "Caches, clean copies + ECS raw (Recommended)". After re-checking each
  git copy clean (0 unpushed commits, 0 stashes), the coordinator deleted:
  - five type-check caches and a lint environment;
  - the ECS tmp, runs, validate and traces folders (the ECS text record is pushed at `<prefix>/handoff-ecs` 067b8a10);
  - two pull-request clones and seven worktrees (git worktree remove, all exit 0).
  - Free space went from 3,913,723,904 B to 7,625,555,968 B. ST/f2prep/DELETIONS.txt lists 3,948,700 KiB. [measured]
- No file of this round and no pinned file is in either list. The ECS raw outputs deleted belong to an earlier screen
  whose text record is pushed. [measured: both lists; RUNLOG line 952] Before F2: 7,616,159,744 B free (line 959);
  lowest during measuring: 5,075,619,840 B (process check). [measured]

### Rule slips recorded for this round

| When (UTC) | Who | What | Effect | Record |
|---|---|---|---|---|
| 10-31 20:57 to 11-01 01:12 | Build agents | All six branch commits carried a trailer naming an AI model, against the maintainer's rule (the agents followed a harness reminder) | Reworded locally with git filter-branch; trees byte-identical, 6 of 6; head e4d222bb became 1dfe2327 | RUNLOG 873 |
| 11-01, about 02:50 | Coordinator | A read-only grep sent its error output to a system device path ("rule slip AGAIN") | Nothing written | RUNLOG 878 |
| 11-01 03:32-03:50 | Coordinator and maintainer | Gaps in the PHASE launch-check record: the three CHECK lines say new_run=not-reported and were never updated (the OFF runs' were, lines 925-927); no cell-9 row was pasted for PHASE s42 (lines 889 and 891 cover s60 and s80); the cell-6 rows came from an unnamed run (line 890) | C4 (one session from step 0) and C8 (switch ON) pass for all three, so the substance is covered | RUNLOG 889-895; score_pc.txt C4, C8 |
| 11-01 17:44 | Maintainer and coordinator | THE BLIND PERIOD opening: PHASE s60's notebook output read before any OFF launch | THE ONE TEST cannot pass (NOT SEPARATED, p = 20/20): a FAIL | RUNLOG 902; "Why it failed" above |
| 11-01 17:50-17:51 | Unknown (inferred) | Possible views of three PHASE s60 files before the OFF launches; content not on record | Covered by the 17:44 opening; caveat on the deviation decision | RUNLOG 905; "The 17:44 opening" below |
| 11-01 18:20 | Maintainer | Order deviation: OFF s80 started before OFF s42 | No scoring rule reads the order | RUNLOG 910; "The order deviation" below |
| 11-01 19:23 | Coordinator | Told the maintainer a wrong end time for the OFF runs (15 h added twice) | Corrected at once | RUNLOG 935 |
| 11-02, 13:56-14:45 | Deviation reviewer B | One read-only du sent its error output to a system device path; self-reported | Nothing written | RUNLOG 951 |
| 11-02, 14:47-15:58 | Deviation re-reviewer | One read-only cat sent its error output to a system device path; self-reported | Nothing written | RUNLOG 953 |
| 11-02, F2 | F2 plan (pinned texts) | M4 at the PHASE finals, owed once line 4 read, was not in the pinned F2 step list and was not run | No parent named | "Open items" 1 |
| 11-02, 16:44-18:07 | F2 fetch and measuring agents | Python with -I but without -B: the six fetch helper scripts, and four inline reads (OFF s42 fetch at 16:57:32; OFF s60 measuring at 17:33:07, 17:33:12 and 18:06:52) | No bytecode written anywhere after 16:39 | RUNLOG 970; process check M1 |
| 11-02, 19:15-19:30 | Process checker (its own) | One `python3 -` call without -I -B on empty input; its scanner first held the device paths as a literal pattern | Ran nothing; no device opened | Process check I6 (journal only) |
| 10-31, 21:00-22:31 | Parallel records draft (phase-screen records, not PC data) | Its checks found slips not stated and an AI product name in an evidence copy | Fixed by its fixer | RUNLOG 872 |

The phase screen's own slip at RUNLOG line 860 (10-31 20:31 UTC) came before this round; PS REPORT.md covers it.

### Other notes

- **F0 files in an agent's context (line 892).** The Drive connector returned the launch-check files as inline encoded
  text in the fetch agent's context, undecoded. These are the launch check's allowed reads, so no SNAPSHOT. [measured]
- **A blind-period listing (line 897).** A liveness listing also showed the best-model files' names and modification
  times. These are metadata, which the blind period allows. They were not interpreted, and later checks listed fewer
  names. [measured]
- **A wrong estimate (line 935; also in the slip table).** The coordinator told the maintainer the OFF runs would end
  at 9:15-10:00 am CST; that added 15 h twice. It was corrected. The manifests landed at 09:05-09:36 UTC. [measured:
  lines 935, 941-945]
- **Launch times.** LAUNCH times are Drive folder creation times. Each run's session began within 0.1-0.8 min of its
  LAUNCH time. [measured: score_pc.txt C4 "minutes_from_launch"]
- **The design's predictions (criteria.txt lines 75-84):** no PHASE rail likelier (it happened); PHASE gate passes
  likely (3 of 3); a habit in about 2 of 3 OFF runs (2 had a tail habit, the third a neck habit). [measured]
- **This report.** Drafted 2026-11-02 19:33-19:45 UTC (date -u; last draft write 19:45:43 UTC), checked by an
  adversarial verifier 19:56-20:10 UTC (ST/final/verify_report/NOTES.txt), and finalized from 20:14 UTC with every
  finding checked against its source (ST/final/FINALIZE_NOTES.txt; the draft is kept as ST/final/drafts/PC_REPORT.md.v1).
  Written from the files listed, read only; it writes only under ST/final/. Slips of its own: during drafting, a
  read-only grep over RUNLOG held a system device path as text in its search pattern; during finalizing, a read-only
  sed used a pattern that matches system device paths to mask them on screen. No device was opened and nothing was
  written. The tool harness also saved long command outputs to its own results folder, outside ST/final/.

## Files (sha256)

Round and pins (ST/):
- e612d52362b4336b7bbaa437a217e18909b86d4f2f04a0135a9c58c9e21d61ec round/criteria.txt
- ce8865205a218541a742d7acf0cee479e87d63c7b53cbc52ae47a8892569873a round/APPROVED
- ed3e29c5711946d81ab3f365c4dba6b34c9a1a4fd0f95844b4df70e377a2a356 round/F2_PINS.json
- 98abcaa6efd68af38ca257d7ab14c5aa6c75eb8940f68764a2f9c9874e7311e1 COLAB_STEPS.txt
- 69b5acf9cd6bde0cf40c86e1757d14b6bb35e64e1db41798b20547ec62cfc7f1 design/DESIGN_NOTES.txt
- 3c55bf21f121d30bc31d78a7228c904f83179ece8dabdaa8530c45f54d203986 tools/pc/score_pc.py
- c1ec55be6254b939fc751fbcef9f984f170e3cf463c5d5b55f9d2f5517f884b5 amend2/AMENDMENT2.txt
- efb37b2099bf77d38c8e585cbc0b86f7eb215e21598a1b93a92633e3b13c62a3 amend2/tools/score_pc_dev.py
- 0c034e90024adec82542658150009a3597795a97bb81142b0f7efcbeeaa31b0c amend2/APPROVED2.json
- 1a4786e88b89a009e692f19a5d3782ea89d83eed9badf30d6295690216960822 amend2/SHA256SUMS.txt

Readings (ST/f2/score/; its SHA256SUMS.txt: 33 of 33 OK at about 19:36 UTC):
- c3736b1fbda069641a72caedee6105ab203d0442d236fc465b2d2ae3e5011ffb score_pc.txt
- 915f661ec97107998f759ab10a1050f0f8ea9b48c30ad760a496b618c6079f68 score_pc.json (= dev/pinned.json)
- 483eba8fe925cd0c9edfee6deff6c14dd1a5f783f5ba1fcd3b1b31994b68883f score_pc_dev.txt
- 89f15e9f1543078439a5dcc1cc35d9cf3a33dc820f724bcb9a61aa9e88d56ee2 dev/deviation.json
- 8c05d9ba257fad9ada011b417d70d05b56c267ca8fa42998448e508e6e7193e5 dev/runlog_asread.txt (RUNLOG as scored, 966 lines)
- 4ea162967b8e8458cd16332581736688aa46ac9d72238980f9762e48f26f0c93 dev/runlog_deviation.txt
- 1f91dbf37bb146d4ae663f0680695dfa2776a4ae1703cfc5c9ac4b36fcbc681b SHA256SUMS.txt

Per-run completeness and gate (ST/):
- 4ff1c8860da14daa6da07e4bf393afb7bb3a0c8fcdd61331294de0ed12054b67 f2/measure/stance7_r10LTSBpc_s60/complete_post.txt
- 76d03b5a43fc0be45dade2d81693034b69c59193bcc461e114de5830f36ab187 f2/measure/stance7_r10LTSBpc_s42/complete_post.txt
- 2f51b1ddd1a69b60f9cf6d617b9c90f3c8b06b483c6fe15119e5c4ac03953548 f2/measure/stance7_r10LTSBpc_s80/complete_post.txt
- 762862bfebdb9646746bfdd37569cc990680bc17b88290a7d58979fddca28786 f2/measure/stance7_r10LTSBoff_s60/complete_post.txt
- 9bb045f5ee117b9313a7ec4cb2608ddcaf881d7d5e68435222267f22f60c3777 f2/measure/stance7_r10LTSBoff_s42/complete_post.txt
- 52d31fc420272037f1ff6dca7edbe70240c92a66ea2e21dc36150e6a26f9d6f5 f2/measure/stance7_r10LTSBoff_s80/complete_post.txt
- 201e7e27162c4eadd3d7305605f0a155b4ae76c3d298bb75cde86a3005ae7e97 round/panels/stance7_r10LTSBpc_s60/final/m3/stance_heading_report.txt
- 25c37ba44859445f7373bf4fb2b8b458d890e3c249c2c5b937d23257c85b36a7 round/panels/stance7_r10LTSBpc_s42/final/m3/stance_heading_report.txt
- b21ec5c7cbf24e8308ce0b2f01be79b599bd5916e88a9c90dd0a2337cbbf5440 round/panels/stance7_r10LTSBpc_s80/final/m3/stance_heading_report.txt
- 95fe810ea65f1fdba577ec92d62dc88505ba3ccc2d28f234144de6247562b387 round/panels/stance7_r10LTSBoff_s60/final/m3/stance_heading_report.txt
- e7bb9e5e56b2ec708ff51ba6613d7d22ef29748389efd26337768fc008ef3259 round/panels/stance7_r10LTSBoff_s42/final/m3/stance_heading_report.txt
- d8e6c5a4cf682f552c4b460157dd05d55ffafedf5c66a0d85fec56ee0aa83abc round/panels/stance7_r10LTSBoff_s80/final/m3/stance_heading_report.txt

Checks (ST/f2/; each SHA256SUMS.txt fully OK at about 19:36 UTC: 16 of 16 and 41 of 41):
- 3099f69c914bbede120df2ee8900ef5e9de565b7315b43acbc8dcfcc332f6a61 check_numbers/SHA256SUMS.txt
- de5cb588126f704b737c29c0a396badd2bb1658df51af0ae2c908667f015db4b check_numbers/out/td_check.json
- 8d67fc46c42e8c31e76e56582ae4fdb346d3147abbed5a6ebbe1530a5667ddbd check_numbers/out/one_test_check.json
- 777771d326317b53761c8a8f8f01a4cb9c10d24ae615dbe9e625e946c587578f check_numbers/out/m1_check.json
- c47e7dd8bc4af05205723392d085608644b1eb794ea7b89a86930880f20ea697 check_numbers/out/ledger_check.json
- 65d58d71547956cfe0d24f05ef589f2222cf23c7061b11f6382b57f25ad47fd4 check_numbers/out/zipcrc_off_vs_stance6.out
- 3a78e8e8dbe8c93eecb5082855f0db5743683b7bc839bee04aa006209a33fa53 check_numbers/out/zipdata_off_vs_stance6.out
- ffe3c3f4cb452945dfa7088ed1a9752c189e7f31e45450e2c125e421d59b82e9 check_process/SHA256SUMS.txt
- b06229c289141cb5c2c1fd3ebebcf974fb73756b69eac45b19ff7cb7ae777056 check_process/manifest_check.out
- 0951cabf8991345086c16df218ffad7fcf8b9fa20409facc75a1a82ca959237f check_process/drive_scan.out
- 0b5a89271a213601600078d7cdace841264da2ab704db01a1a96a527bec28707 check_process/inputs_check.out
- fc70b7e4fd319f55b0e5e6d9965badc3006a78ba2b978a05d1b763561f34dbc6 check_process/seal_check.out
- 9c81528a6d4b555c1d3322c9f02e1f4c08221599546340fda0ce8756a579c330 check_process/dev_diff_check.out
- afb5362fa10613b65277c9b86df05cc115d24205dbe3ebc70bff9607eb9dc12b check_process/concurrency_check.out
- ea00b5612ad4fe9017a2642acb0c1a20f5f0ae6a3e14f420d7e56a7a6db4cfe1 check_process/py_flag_scan.out
- 102a3c4d68b1c2f2dc5e9ef5b853c174d6b2bce8d23594b42042cbc083e82fbf check_process/transcript_scan.out

Other:
- 89425ea97aff2bb93ab9b0643a33ae7580f0877f98db8761b3f451508f4d5f16 RUNLOG as read for this report (971 lines)
- 85f43e8c5ad810b34e6154e355ee0c888724e3b3e9392fe02dce7ebffa9c9790 /home/user/work/phase_screen/REPORT.md (the shape)
- Check summaries: the journal of workflow wf_cd81846d-bc1, result.checkA (numbers) and result.checkB (process).
- Review evidence written in ST/final/ (read only by this report):
  - 7dc7c40d91521be9b4ebc08725af4761b5160e4d16540bae6d120016b8bedde8 verify_report/evidence/falls.txt
  - d5e9f4d75eda2d4306a0e59643df73ee1c5ec8d2f82e68432727c7f3e74b38e1 verify_report/evidence/identity_spot.txt
  - c33c06dc5e46e8581648386f04241dae165a7735f8913ca2b8a6ca809f99914f verify_report/evidence/env_fields.txt
  - 1f7484f03e4afc045de4ed2438d270cd4abc7aacdf0f8195b5632a5ef6961178 next_steps_work/short_eps.out
  - 13338ae2469d1d97ac27db6bf3adb652743c749012bfe2b069a7d17e80fb7553 next_steps_work/env_fields.out
- Durable copies (names only, not fetched): `<prefix>/handoff-phase-critic-prereg` (dd0e5ffd, RUNLOG line 883),
  `<prefix>/handoff-phase-screen`, `<prefix>/handoff-ecs`, `<prefix>/handoff-tailscreen`.
