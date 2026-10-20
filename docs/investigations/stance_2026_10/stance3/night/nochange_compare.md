# Night continuations: A and B against the local no-change run (report only)

Written 2026-10-19, 09:35-10:15 UTC. This is a report only.
- Nothing on Drive or GitHub changed. Nothing ran in or wrote to /home/user/mesozoic-labs.
- The clones were checked after the last rollout and are clean: screen_s3_LT at 1af2d4d7 and exp_night at 80328abf.
- "Measured" means read from a file or computed from a rollout I ran. "Inferred" is marked.
- Paths: S3 = scratchpad/foot/stance3, C = S3/night/cont.

## 0. Bottom line

| question | answer (measured) |
|---|---|
| Does the local no-change run close the stagger (sd <= 10 mm) and reach K2a <= 3 by 9,961,472? | **No.** Stagger sd **25.9 mm** and **K2a 15**. At 9,437,184 it was 29.4 mm with K2a 10. |
| Is A better than the local no-change run on stagger sd, CoP split sd, K2a and K2b? | **Yes, on all four, at both checkpoints.** Every paired seed-bootstrap interval excludes 0. At 9.96M: stagger -18.5 mm [-23.3, -12.9], split -0.471 [-0.563, -0.358], K2a -15 [-21, -9], K2b -6 [-11, -2]. |
| Is B worse than the local no-change run? | **Only on K2b.** At 9.96M, B is worse on K2b (+7 [+3, +12]) and about level on split sd (+0.087 [-0.012, +0.185]). It is better on stagger sd (-7.8 mm [-11.0, -4.2]), K2a (-5 [-10, 0]) and clean count (30 against 25). |
| Is the stagger term's effect attributable? | **Unclear.** The result rules out "no-change matches A", but it does not show that the term caused the gap. See below. |

**Why "unclear" and not "attributable":**
- **Against the harness-matched control, A is clearly better.** So the earlier reading "the unchanged recipe did what A did" (interpretation.md, sections 0 and 4) does not hold in this harness.
- **Two no-change runs disagree by as much as A and the local no-change run do.** The Colab no-change run and the local no-change run start from the same checkpoint and reach the same step. Measured with the same paired bootstrap at 9.96M (Colab minus local):

  | readout | Colab minus local no-change | A minus local no-change |
  |---|---|---|
  | stagger sd | -16.3 mm [-22.8, -9.1] | -18.5 mm |
  | split sd | -0.453 [-0.607, -0.282] | -0.471 |
  | K2a | -15 [-21, -9] | -15 |
  | K2b | -6 [-11, -2] | -6 |

  A and the Colab run are level on every one of the four: stagger -2.2 mm [-4.9, +0.5], K2a and K2b 0 against 0.
- **What the bootstrap measures.** It covers only which 40 spawns the panel draws. It does not cover the run-to-run variance of training, and with one run per arm that variance is as large as the effect.
- **Where A does differ (inferred, not tested).** A's closure held. It held at both judged checkpoints, and in every 10-episode ledger evaluation from 8.91M to 9.96M (per-evaluation stagger sd 5.6-10.9 mm). Neither no-change run held it:
  - the local run's evaluations ranged 11.4-43.5 mm;
  - the Colab run's ranged 10.2-51.9 mm.

  This favours the term, but it is n = 1 per arm.
- **What would settle it:** more seeds for both arms, not a longer single run.

## 1. Setup and validity

**Lineages.** All three runs start from the Colab Run 3 checkpoint at 8,388,608 (zip sha256 c67576b6...). All use arm LT, seed 52, n_envs 4, the exp_night clone at 80328abf and the same cadence. The learning rate matches at the same steps: 1.6005e-5 at 8,404,992 and 1.4258e-5 at 9,453,568 in all three.

| run | resume points | train task |
|---|---|---|
| A | 8,912,896 (05:31) | d4a3b8f8 + stagger (w 0.30, tol 45 mm) |
| B | 8,912,896 (05:31) | d4a3b8f8 + anchor (w 0.4, tol 0.10) |
| no-change | 9,437,184 (09:34) | d4a3b8f8, no added kwargs (`added_env_kwargs {}`) |

The resume points differ: 8,912,896 for A and B, 9,437,184 for the no-change run. Each resume reseeds the env streams and collects a fresh rollout. Inferred: this adds noise but no systematic bias.

**R1.**
- The 9,437,184 files: C/judged94/<run>/judge_9437184.json. These were made by the orchestrator's judge_exp.sh queue.
- The no-change 9,961,472 file: C/judged/night_R3_nochange/judge_9961472.json. I ran the command given (rc 0, 296 s).
- The A and B 9,961,472 files are the earlier ones.
- All files are relaxed-binding and report only: task_binding "report" and allow_commit_mismatch. The rolled task is d4a3b8f8 in every file.

**R2.**
- **Rollouts.** I ran roll_night.py, which calls roll_split.py (sha256 dbb6aeea..., unchanged), in screen_s3_LT through tools/iso.py. One rollout ran at a time, each about 2 minutes:
  - A, B and the no-change run at 9,437,184;
  - the no-change run at 9,961,472.
- **Symlink.** I added C/r2/runs/night_R3_nochange/01_stance -> S3/night/runs/night_R3_nochange.
- **Readouts.** They use analyze_night.py's repro() and readouts(), imported unchanged.
- **Reproduction.** Each of the 6 sets reproduces its judge file on all 40 seeds:
  - seeds, lengths and terminations are equal;
  - |foot_cop_fore_aft - judge| is 0 on both feet, and the pad metric |diff| is 0;
  - the K2a and K2b flags are equal;
  - the npz model sha256 equals the judge file's.
- **Short episodes.** No set has one.
- **Reference sets.** The two reference sets also reproduce: the R3 start at 8,388,608 and the Colab control at 9,961,472.

**Process count.**
- From 09:34 to 09:39, the orchestrator's three 9.44M judges ran beside the training run. That is 4 processes, and they were not mine.
- After that I kept to 2 or fewer:
  - while training ran, my rollouts ran one at a time beside it;
  - the judge and the last rollout ran after training ended at 09:59.

**Timing.** The 9,961,472 checkpoint did not exist when step 1 finished (about 09:46). I waited for it, because step 3 needs it. It landed at 09:59.

## 2. R1 at 9,437,184 and 9,961,472

Spec 7.2 safety flags are checked against the Run 3 start (clean 34, roll 3; start reasons: pad, sole tilt, corner lift).

| | A 9.44M | B 9.44M | NC 9.44M | A 9.96M | B 9.96M | NC 9.96M |
|---|---|---|---|---|---|---|
| verdict | PASS | FAIL | FAIL | PASS | FAIL | FAIL |
| clean / 40 (LCB) | 40 (0.928) | 35 (0.755) | 30 (0.613) | 40 (0.928) | 30 (0.613) | **25 (0.483)** |
| K2a | 0 | 5 | 10 | 0 | 10 | **15** |
| K2b | 0 | 5 | 7 | 0 | 13 | **6** |
| roll | 0 | 0 | 4 | 0 | 1 | 4 |
| x_c - statue (mm) | -15.2 | -6.1 | -16.5 | -13.7 | -5.4 | -16.3 |
| reward ratio | 0.995 | 0.992 | 0.983 | 0.983 | 0.985 | 0.982 |
| full / hop-or-fall / panel falls | 40/0/0 | 40/0/0 | 40/0/0 | 40/0/0 | 40/0/0 | 40/0/0 |
| noise falls, joint / own (of 20) | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| saturation: episodes > 0.10; worst share | 0; 0.000 | 0; 0.000 | 0; 0.000 | 0; 0.000 | 0; 0.000 | 0; 0.000 |
| criterion failures (episodes) | none | pad 5 | tilt 4, corner 4, pad 10 | none | corner 1, pad 10 | tilt 4, corner 4, pad 15 |
| spec 7.2 safety flags | none | none | none | none | none | none |

**K2a seeds.**
- No-change at 9.96M: 3042, 3050, 3051, 3052, 3053, 3061, 3062, 3067, 3068, 3069, 3071, 3073, 3074, 3075, 3081.
- No-change at 9.44M: 3042, 3046, 3050, 3051, 3052, 3053, 3067, 3071, 3073, 3081.
- B at 9.44M: 3050, 3052, 3071, 3072, 3081.

**Safety flags in detail.**
- **Clean below 34.** B and the no-change run fall below 34, but only for the start's own reasons (pad, tilt, corner lift), so no new reason.
- **Side door.** Touchdown, all-feet support, settle touchdowns and yaw change are 0 in all six.
- **Roll.** At most +1 against the start (no-change: 4).
- **Stops.** No hop_watch.json or ENDED_ON_ITS_OWN.json exists.
- **The one non-panel fall.** It is in the no-change ledger: an evaluation episode at 9,830,400, length 573, "fallen".

## 3. R2 at 9,437,184 and 9,961,472

| readout | A 9.44M | B 9.44M | NC 9.44M | A 9.96M | B 9.96M | NC 9.96M | Colab NC 9.96M (ref) |
|---|---|---|---|---|---|---|---|
| window stagger sd (mm) | **8.5** | 10.9 | **29.4** | **7.5** | 18.1 | **25.9** | 9.7 |
| six-seed mean \|s\| (mm) | 15.3 | 21.1 | 62.0 | 13.3 | 34.8 | 51.3 | 10.0 |
| six seeds, s each (mm) | -5.6, -13.8, +22.7, -17.4, -15.8, +16.4 | -23.2, -13.7, +22.3, -22.3, -14.3, +30.5 | -87.6, -27.6, +69.0, -52.9, -45.5, +89.7 | -6.1, -8.9, +19.4, -15.5, -11.8, +18.2 | -39.1, -18.2, +35.4, -32.8, -27.4, +56.1 | -33.9, -32.4, +63.3, -49.5, -55.1, +73.9 | +1.1, -16.2, +22.2, -4.3, -15.8, -0.4 |
| stagger sd i200 -> i999 (mm) | 12.3 -> 7.3 | 15.9 -> 9.1 | 23.4 -> **36.6** | 10.7 -> 5.6 | 17.7 -> 19.0 | 20.8 -> **33.7** | 19.3 -> 6.1 |
| r-l hip servo slope | +0.54 | +0.96 | +0.64 | +0.52 | +1.09 | +0.88 | +0.21 |
| CoP split sd | 0.453 | 0.811 | 0.855 | 0.471 | 1.029 | 0.943 | 0.490 |
| common CoP | -0.160 | -0.043 | -0.206 | -0.133 | -0.039 | -0.196 | -0.179 |
| ankle offset from settled (deg) | +1.28 | +0.52 | +1.32 | +1.02 | +0.13 | +1.16 | +1.17 |
| mean \|ankle L-R\| (deg) | 0.46 | 0.40 | 1.03 | 0.38 | 0.64 | 1.01 | 0.49 |
| heel-foot min CoP | -0.772 | -0.948 | -0.994 | -0.726 | -0.963 | -0.989 | -0.795 |
| corr(yaw change, bearing) | +0.601 | +0.034 | +0.727 | +0.154 | +0.778 | +0.932 | +0.705 |

Start values (8,388,608), for reference: 20.9; 44.0; 20.1 -> 24.2; +0.62; 0.685; -0.253; +2.00; 0.99; -0.974; +0.880.

**What the local no-change run did:**
- It stayed at or above the start on every split readout:
  - stagger sd 20.9 -> 29.4 -> 25.9 mm;
  - the stagger grew inside the window at both checkpoints;
  - split sd 0.685 -> 0.855 -> 0.943;
  - K2a 6 -> 10 -> 15.
- It still moved the ankle level and the common CoP about halfway back, as A and the Colab run did: ankle +2.00 -> +1.16 deg, common CoP -0.253 -> -0.196.

**What follows for A's "ankle unchanged" line.**
- In this harness, A's ankle (+1.02) is within 0.14 deg of the no-change run (+1.16). So the ankle move belongs to the recipe, not to the term.
- Read against the matched control, spec 7.2's expectation for A ("ankle and common unchanged: not this term's target") holds for the ankle.
- The common CoP is 0.063 better in A.

**B's stagger line.** Spec 7.2's line "window stagger sd <= 10: the stagger closed without its term, so A's attribution fails" is **not** met by the matched no-change run (25.9 mm).

**At 9.44M B was nearly as closed as A** (10.9 against 8.5 mm), then reopened to 18.1 mm. So a single-checkpoint closure is not unique to the stagger term.

## 4. Paired seed bootstrap and McNemar

Method: 40 panel seeds, 10,000 resamples with the same indices for both runs of a pair, default_rng(20261019), 95% percentile intervals. This covers panel sampling only.

| readout | step | A | B | NC | **A - NC** | **B - NC** |
|---|---|---|---|---|---|---|
| stagger sd (mm) | 9.44M | 8.5 [6.5, 10.3] | 10.9 [8.1, 13.4] | 29.4 [18.8, 38.2] | **-20.9 [-29.1, -11.6]** | -18.5 [-25.3, -10.3] |
| | 9.96M | 7.5 [5.6, 9.1] | 18.1 [13.1, 22.6] | 25.9 [18.7, 32.1] | **-18.5 [-23.3, -12.9]** | -7.8 [-11.0, -4.2] |
| CoP split sd | 9.44M | 0.453 [0.349, 0.543] | 0.811 [0.652, 0.943] | 0.855 [0.713, 0.962] | **-0.403 [-0.499, -0.293]** | -0.044 [-0.177, +0.092] |
| | 9.96M | 0.471 [0.368, 0.559] | 1.029 [0.849, 1.169] | 0.943 [0.794, 1.050] | **-0.471 [-0.563, -0.358]** | +0.087 [-0.012, +0.185] |
| K2a | 9.44M | 0 [0, 0] | 5 [1, 10] | 10 [5, 15] | **-10 [-15, -5]** | -5 [-10, 0] |
| | 9.96M | 0 [0, 0] | 10 [5, 16] | 15 [9, 21] | **-15 [-21, -9]** | -5 [-10, 0] |
| K2b | 9.44M | 0 [0, 0] | 5 [1, 10] | 7 [3, 12] | **-7 [-12, -3]** | -2 [-6, +2] |
| | 9.96M | 0 [0, 0] | 13 [7, 19] | 6 [2, 11] | **-6 [-11, -2]** | **+7 [+3, +12]** |

**Exact McNemar on the paired flags** (arm-only / NC-only, two-sided p):

| pair | 9.44M | 9.96M |
|---|---|---|
| A vs NC, K2a | 0/10, p 0.002 | 0/15, p 0.0001 |
| A vs NC, K2b | 0/7, p 0.016 | 0/6, p 0.031 |
| B vs NC, K2a | 1/6, p 0.125 | 1/6, p 0.125 |
| B vs NC, K2b | 1/3, p 0.625 | **7/0, p 0.016** |

**The between-control yardstick at 9.96M** (C/nochange/two_nochange.txt), with the same code:

| | stagger sd | split sd | K2a | K2b |
|---|---|---|---|---|
| Colab NC - local NC | -16.3 [-22.8, -9.1] | -0.453 [-0.607, -0.282] | -15 [-21, -9]; McNemar 0/15, p 6e-5 | -6 [-11, -2] |
| A - Colab NC | -2.2 [-4.9, +0.5] | -0.019 [-0.141, +0.104] | 0 | 0 |

## 5. Eval ledgers, 8.39M -> 9.96M (no simulation; C/nochange/ledger_nochange.txt)

Per-evaluation stagger sd (10 episodes, ledger FK map):

| run | evaluations after 8.91M (mm) | block to 9.96M: stagger sd / clean / pad over / K2b out |
|---|---|---|
| A | 7.4, 9.7, 9.7, 8.5, 10.9, 7.1, 6.8, 5.6 | 8.8 / 39 / 1 / 0 |
| B | 17.1, 20.5, 17.6, 9.4, 16.1, 11.5, 8.9, 9.6 | 13.0 / 36 / 4 / 7 |
| no-change (local) | 32.2, 11.4, 14.5, 12.9, 27.5, 36.1, 43.5, 21.8 | **33.3 / 25 / 15 / 12** |
| Colab no-change (from readouts_control.md) | 19.3, 51.9, 30.1, 32.0, 18.2, 10.2, 15.7, 14.8 | 15.9 / 35 / 5 / 3 |

The local no-change ledger has 12 evaluations, each with 10 rows and no duplicates across the restart.

## 6. Reading

1. **The matched control does not recover.** The verifier's question, "is A's recovery what the unchanged recipe does in this harness?", is answered no. At 9.96M the local no-change run has stagger 25.9 mm, K2a 15, clean 25 and FAIL, against A's 7.5 mm, K2a 0, clean 40 and PASS.
2. **The unchanged recipe can still reach A's endpoint.** The Colab run from the same checkpoint did. So the gap between A and the local no-change run (n = 1 each) is no larger than the gap between two no-change runs.
   - Under spec section 6 item 3 ("one control against one arm reads only effects larger than the natural swing"), the effect is not shown to exceed that swing.
   - The verdict is therefore **unclear**. It is not "not attributable": the matched control does not match A.
3. **The most term-specific signal is persistence (inferred).** A is the only one of the four continuations whose stagger stayed closed at both checkpoints and in every ledger evaluation after 8.91M. The other three are B, the local no-change run and the Colab no-change run.
4. **B against the matched control.**
   - B is worse only on K2b: 13 against 6 (p 0.016). Its split sd is level.
   - It is better on K2a, stagger and clean.
   - The earlier statement that "the anchor is associated with 10 more K2a episodes" was made against the Colab control. Against the local no-change run, B has 5 **fewer**.
   - H_B is still refuted by its own pre-registered line: ankle +0.13 and common -0.039, yet K2a 10.
5. **What would resolve A** (inferred cost): 2-3 more seeds each of a local no-change run and A, from the same checkpoint. Each run takes about 65-80 minutes. Alternatively, the from-scratch screen with an unchanged arm on the same seeds, which interpretation.md already recommends.

## Files

| item | path |
|---|---|
| R1, 9,437,184 (orchestrator's queue) | C/judged94/{night_R3_stagger,night_R3_anchor,night_R3_nochange}/judge_9437184.json (+ .exp.json, .log) |
| R1, no-change 9,961,472 (run here) | C/judged/night_R3_nochange/judge_9961472.json (+ .exp.json, .log); queue line in C/judged/queue.log |
| R2 rollouts (run here) | C/nochange/out/{night_R3_stagger,night_R3_anchor,night_R3_nochange}_9437184.npz, C/nochange/out/night_R3_nochange_9961472.npz (logs log_*.txt) |
| R2 symlink (added) | C/r2/runs/night_R3_nochange/01_stance -> S3/night/runs/night_R3_nochange |
| comparison script and output | C/nochange/scripts/nochange_compare.py -> C/nochange/nochange_compare.txt, C/nochange/nochange_compare.json |
| between-control bootstrap | C/nochange/scripts/two_nochange.py -> C/nochange/two_nochange.txt |
| ledgers | C/nochange/scripts/ledger_nochange.py -> C/nochange/ledger_nochange.txt |
