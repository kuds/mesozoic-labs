# Replicate study, night of 2026-10-20: does the stagger term's continuation effect survive more seeds?

Pre-registration. Written 2026-10-20, 00:10-00:35 UTC, before any training step of this study. Amended 01:36-01:50 UTC
after a review, still before any training step (section 12 lists every amendment; N4/review_handling.md gives the
reasons). Frozen at 01:50 UTC: sha256 in N4/spec.sha256, file mode 444. Report only:
- nothing here enters or changes a stance4 verdict, and nothing changes the stance4 criteria
  (SP/foot/stance4/criteria.txt, sha256 978a93da8da469c1acde7b9f7261d9236e4799eb2111b3af0f0cbeaa153f0f38, checked
  unchanged at 00:31 UTC, build/checks/final_checks.txt);
- fails stay fails;
- no Drive or GitHub writes, nothing in checkout, no commits.

"Measured" means read from a file or computed by a script named here. "Inferred" is marked.
Paths: SP = scratchpad,
S3 = SP/foot/stance3, C = S3/night/cont, N4 = SP/foot/stance4/night.

**The from-scratch stance4 screen remains the deciding evidence for the stagger term.** This study can only say whether
last night's continuation result replicates.

## 0. In short

- **Question.** Last night, from one checkpoint, the stagger continuation (A) closed the split and the local no-change
  run did not. But the Colab no-change run from the same checkpoint closed it too. With one run per arm, the effect was
  no larger than the gap between two no-change runs. Is A's advantage real, or run-to-run noise?
- **Design** (fixed by the orchestrator). Four new continuations from the same parent, to the same endpoint, with the
  same harness: two more no-change runs (NC2, NC3) and two more stagger runs (ST2, ST3), on two new seeds. With last
  night's three, that gives stagger n = 3 and no change n = 4.
- **Primary readouts.** Window stagger sd (mm) and K2a count, at E = 9,961,472.
- **Test.** Exact one-sided permutation (rank-sum) of stagger against no change, 3 against 4.
- **One limit found while building (section 6.3).** The Colab no-change run already has K2a 0 (measured). So on K2a
  the stagger arm cannot be completely separated from the no-change arm, and the smallest attainable p there is
  4/35 = 0.114. As fixed, reading line (1) cannot be reached. **Decided before launch (section 7): option (a), the
  lines stay as fixed.** The strongest attainable reading is line (2).
- **A second limit (section 6.4, added before launch).** Last night's three runs generated the hypothesis, and their
  order was known. So the 35-split p overstates the evidence. Two report-only numbers are added: the p conditional on
  that known order (3/35 = 0.086 at best), and the new runs alone (2 against 2, 1/6 = 0.167 at best).

## 1. Background (measured; C/nochange_compare.md and C/interpretation.md)

All from Colab stance3 Run 3 (stance3_armLT_s52) at 8,388,608, read at E = 9,961,472:

| run | harness | seed | clean / 40 | K2a | K2b | window stagger sd (mm) | CoP split sd |
|---|---|---|---|---|---|---|---|
| A, stagger (w 0.30, tol 45 mm) | local, exp_night | 52 | 40 | 0 | 0 | 7.5 | 0.471 |
| local no-change | local, exp_night | 52 | 25 | 15 | 6 | 25.9 | 0.943 |
| Colab no-change (Run 3's own continuation) | Colab | 52 (mid-stream) | 40 | 0 | 0 | 9.7 | 0.490 |
| start (8,388,608) | | | 34 | 6 | 3 | 20.9 | 0.685 |

- A minus local no-change: stagger sd -18.5 mm, K2a -15.
- Colab minus local no-change: stagger sd -16.3 mm, K2a -15.
- The paired seed bootstrap covers panel sampling only. It does not cover training-run variance, which is what is in
  question.
- nochange_compare.md's conclusion: "attribution unclear ... what would settle it: more seeds for both arms".

## 2. The runs

| run | arm | training seed | added env kwargs | parent | endpoint | run dir |
|---|---|---|---|---|---|---|
| NC2 | L+T, no change | 152 | {} | Run 3 @ 8,388,608 | 9,961,472 | N4/runs/NC2 |
| ST2 | L+T + stagger | 152 | stance_stagger_weight 0.3, stance_stagger_tolerance_m 0.045 | the same | the same | N4/runs/ST2 |
| NC3 | L+T, no change | 252 | {} | the same | the same | N4/runs/NC3 |
| ST3 | L+T + stagger | 252 | as ST2 | the same | the same | N4/runs/ST3 |

With last night's runs (read from their existing files, not re-run):

| group | runs | n |
|---|---|---|
| stagger (ST) | A (seed 52), ST2 (152), ST3 (252) | 3 |
| no change (NC) | local no-change (52), Colab no-change (52), NC2 (152), NC3 (252) | 4 |

- **Parent.** S3/runs/stance3_armLT_s52/01_stance/models/stage1_8388608_steps.zip, sha256
  c67576b67ec3c6b08e276e440fb13d7054f9041a85124aba21fa7ec4c1330785, and stage1_vecnormalize_8388608_steps.pkl, sha256
  49ce291a1c734a4e61c7af9737d3d930afeea5d156b95c4e6b1fab968cbe798f. Both re-measured at 00:31 UTC. continue.py checks
  them again on every first start.
- **Endpoint.** E = 9,961,472 (3 checkpoints of 524,288 past the parent), the same step as last night's readouts.
- **Order.** At most 2 runs train at once: NC2 and ST2 first, then NC3 and ST3.

## 3. Seeds

How a seed S reaches the streams (measured from the exp_night code at 80328abf and SB3 2.9.0):
- **Training envs.** environments/shared/train_base.py make_env: env rank i calls set_random_seed(S + i) and
  env.reset(seed=S + i). On the model load, SB3's _setup_model calls set_random_seed(S), which also calls
  VecEnv.seed(S): the 4 training envs reset with seeds S, S+1, S+2, S+3 at the first reset of learn().
- **Policy sampling and minibatch order.** model.set_random_seed(S) seeds Python, numpy and torch globally. It runs
  after the env constructors, so the global streams start from S. (train_base: alg_kwargs.setdefault("seed", seed);
  the stance [ppo] block names no seed.)
- **Evaluation env.** eval_env_seed(S) = S + 1000. It drives the 10-episode evaluations and the ledgers only; the
  evaluations are deterministic.
- **On a resume,** continue.py passes the same S again, so each stream restarts from its start. Last night's A and
  local no-change runs both had such a restart (A at 8,912,896; local no-change at 9,437,184). Inferred: noise, not
  bias.

Chosen seeds: **152** (NC2, ST2) and **252** (NC3, ST3).
- Training env streams {152..155} and {252..255}; eval streams 1152 and 1252.
- Disjoint from last night's seed 52 ({52..55}, 1052), from each other, from the screen seeds 42 and 60 and their
  offsets, and from every judge panel (3042-3201, 5042-5081, 7042-7081, 9042-9081).
- NC2 and ST2 share every stream; so do NC3 and ST3, and A and the local no-change run (seed 52). Each seed pair
  therefore starts from the same first rollout, and differs only by the reward term from the first update on.
  Inferred: the trajectories part within the first update, so the pairing matters little.

## 4. Harness (the same as last night's A and local no-change runs)

The same, by construction:
- **Trainer.** S3/night/build/continue.py, unchanged, sha256 42398f26b61fbb949db366a28d4e79a89c7cbce92a913142425da69f47b58e69,
  through S3/tools/iso.py (sha256 7749ca3f...) with ISO_WT = SP/fix/exp_night.
- **Clone.** SP/fix/exp_night at 80328abf955cfe762d814784d018e6f859defb0a, clean (measured 00:31 UTC; screen_s3_LT at 1af2d4d7 clean too). It carries the
  stagger term and the anchor; only the stagger term is switched on, and only in ST2/ST3.
- **Arm.** L+T as committed (armcfg.py sha256 32117dba...; the LT overlay changes nothing). The stance task_sha256
  with the term off is d4a3b8f8... (checked by continue.py).
- **n_envs 4**; the arm's [curriculum] cadence: an evaluation of 10 episodes every 131,072 steps, a checkpoint every
  524,288 steps.
- **Learning rate.** train_base gets the remaining stage budget (12,000,000 minus the loaded steps), so SB3's progress
  keeps Colab's 12,000,000 anchor: lr = 1e-5 + 2e-5 x (1 - n / 12e6), 1.60e-5 at 8.39M to 1.34e-5 at 9.96M.
  ent_coef is 0. clip_range 0.2 and target_kl 0.03 are constant.
- **Stop.** --stop-at 9,961,472, as last night's local no-change run. (A had --stop-at 10,485,760 and was stopped by a
  container restart after E; the stop step does not change the E checkpoint, because the LR schedule and budget do not
  depend on it.)
- **Task crossing.** continue.py's wrapper lets through exactly the added stagger kwargs (ST runs) or nothing (NC runs).
- **Tensorboard** from S3/cf/tbsite on PYTHONPATH, as launch.sh did.

What differs from last night (each recorded):
1. Run root N4/runs and names NC2, ST2, NC3, ST3.
2. Concurrency: 2 trainers plus at most 1 judge or rollout. Last night A ran beside 2 other trainers; the local
   no-change run ran alone beside rollouts and judges. Measured rates were 349-376 steps/s per process in both cases,
   so CPU sharing does not change the training arithmetic (inferred: the numerics are per process, single-threaded).
3. **Judge seed flag.** judge_stance3.py accepts --seed 52 or 60 only. Its --seed does two things: the binding check
   against the checkpoint's recorded seed, and the choice of the frozen joint-noise sigma (control_sigma[seed]).
   judge_rep.sh therefore passes **--seed 52 --allow-seed-mismatch**. The noise test then uses control_sigma[52],
   exactly as for A and the local no-change run. The relaxation is written into the judge file (checks.relaxed); such
   files are report-only and score_stance3.py refuses them. Measured in a 2-episode smoke: the judge accepts the
   flags and records allow_seed_mismatch true.
4. **R2 wrapper.** roll_rep.py is roll_night.py with RUNS as an argument (N4/r2/runs). roll_split.py itself is
   unchanged (sha256 dbb6aeea..., checked at every call). Measured in a 2-episode smoke on A's E checkpoint: 147 of
   149 per-episode arrays are bitwise equal to last night's rollout, including qpos and site_ahead. The two that differ
   are root_xy and reset__root_xy: roll_split stores a view of mjData there, so each file holds one constant from the
   end of its process. No readout reads them.

## 5. Readouts at E (last night's R1 and R2, exactly)

**R1, the full judge** (judge_rep.sh -> S3/night/build/judge_exp.py -> S3/judge/judge_stance3.py, sha256
5dd3cf4d..., both unchanged; exp_night clone; --task-binding report --allow-commit-mismatch --stage-subdir .; the
40-episode panel on seeds 3042-3081, deterministic, heading 0; the statue on the same seeds; the joint and own-sigma
noise tests on spawns 3042-3061). The judged env is the L+T task with the term off (d4a3b8f8...), so the K numbers are
the screen's. Read with C/r1_summary.py summ() (unchanged):
- verdict, clean / 40 (and LCB), K2a, K2b, roll rail, x_c minus statue, reward ratio to the statue, full horizon,
  hop-or-fall, panel falls, noise falls (joint, own), saturation episodes, criterion failures.

**R2, the split rollout** (roll_split.py in screen_s3_LT at 1af2d4d7, the same panel). Read with
C/nochange/scripts/nochange_compare.py r2_set(), which calls analyze_night.py repro() and readouts() (all unchanged).
A set is used only if, on all 40 seeds, the seeds, lengths, terminations, each foot's foot_cop_fore_aft, the pad
metric and the K2a/K2b flags equal the judge file's, and the model sha256 equals the judge file's. The readouts:
- window stagger sd, window mean |stagger|, the six former K2a seeds' mean |s|, stagger sd at reset / i200 / i999;
- CoP split sd, common CoP, heel-foot min CoP, corr(split, stagger);
- ankle offset from settled (mean, r, l), mean |ankle L-R|;
- r-l hip servo slope and force per mm; yaw-creep readouts; foot slide; episode yaw change; COM ahead of MTP III;
  knee / hip offsets; the term costs; saturation; K2a / K2b recomputed.

**Safety flags** (spec 7.2 of last night, nochange_compare.flags(), against the Run 3 start judge file): clean below
the start for a new reason; hop-or-fall >= 2; any panel fall; joint-noise falls >= 3/20 or own-sigma >= 2/20; any
touchdown, all-feet-support, settle-touchdown or yaw-change failure; saturation >= 2 episodes; roll up by >= 2; x_c
outside [-60, +30] mm of the statue; reward ratio < 0.95; hop_watch.json or ENDED_ON_ITS_OWN.json present.

Last night's three runs enter with their existing files:
- A: C/judged/night_R3_stagger/judge_9961472.json and C/r2/out/night_R3_stagger_9961472.npz;
- local no-change: C/judged/night_R3_nochange/judge_9961472.json and C/nochange/out/night_R3_nochange_9961472.npz;
- Colab no-change: S3/judged/stance3_armLT_s52/judge_9961472.json (strict binding, screen clone) and
  C/control/out/r3ctl_9961472.npz.

readout.py's dry run reproduces all three (REPRODUCED, model sha256 equal) and their numbers in section 1.

## 6. Primary comparison and test

### 6.1 Primary readouts
- **Window stagger sd** (mm, R2), over the full-horizon episodes of the 40-seed panel.
- **K2a count** (R1), of 40.
Lower is better on both. Each is one number per run.

### 6.2 Test
- Exact one-sided permutation test on ranks (Wilcoxon rank-sum), stagger (n = 3) against no change (n = 4).
- Statistic: the sum of the stagger runs' ranks over all 7 values (rank 1 = lowest).
- p = the share of the C(7,3) = 35 ways to choose 3 of the 7 runs whose rank sum is at most the observed one.
- Reported for both primary readouts. No correction for two tests (report only).

### 6.3 Ties, and the K2a limit
- Ties take mid-ranks. p counts splits with a rank sum at most the observed one.
- **Complete separation** = every stagger value strictly below every no-change value. **Weak separation** = every
  stagger value at or below every no-change value. Both are reported; the reading lines use complete separation.
- Without ties, complete separation gives p = 1/35 = 0.029. One inversion gives 2/35 = 0.057; then 4/35 = 0.114,
  7/35 = 0.200.
- **K2a.** The Colab no-change run has K2a 0 and A has K2a 0 (measured). A K2a count cannot go below 0. So complete
  separation on K2a is impossible, whatever the new runs do.
  - Best case: ST2 = ST3 = 0 and NC2, NC3 > 0. The four zeros share mid-rank 2.5, the stagger rank sum is 7.5, and
    p = 4/35 = 0.114 (computed by readout.perm_test; build/checks/perm_test_check.txt).
  - If NC2 or NC3 is also 0: p = 10/35 = 0.286.
- **Window stagger sd.** Complete separation is still possible: it needs ST2 and ST3 below 9.7 mm (the Colab run) and
  NC2 and NC3 above every stagger value.

### 6.4 What the test does and does not cover
- It covers run-to-run variation, which last night's panel bootstrap did not.
- All seven runs share one parent checkpoint. So it is replication within one lineage, not across parents.
- The no-change group mixes harnesses (one Colab run, three local). The stagger group is all local. The Colab run is
  the one at 9.7 mm and K2a 0. Inferred: including it makes separation harder for the stagger arm.
- **The 35-split p is optimistic (amended before launch).** Last night's three runs (A, the local no-change run, the
  Colab no-change run) generated the hypothesis. Their values were seen first, and they already sit in the
  hypothesised order (stagger sd: A 7.5 < Colab 9.7 < local 25.9 mm, measured). The 35-split p treats them as new
  evidence. The arithmetic is right; read as the strength of the evidence, it is not. So p = 1/35 = 0.029 for complete
  separation overstates it.
- Two report-only numbers are added (readout.py; they do not enter the reading lines, which use separation, not p):
  - **p conditional on last night's order.** Every assignment of the seven observed values to the seven runs that gives
    the old three their observed order (<, =, >) is equally likely under no effect; p is the share of those with an ST
    rank sum at most the observed one. Measured by enumeration: complete separation on stagger sd gives
    72/840 = 3/35 = 0.086, not 1/35. The K2a best case (ST2 = ST3 = 0, NC2 and NC3 > 0) gives 144/864 = 1/6 = 0.167.
  - **The new runs alone:** ST2 and ST3 against NC2 and NC3, the same rank-sum test, 2 against 2. Its smallest p is
    1/6 = 0.167. With it, the within-seed signs (stagger minus no change) for seeds 152 and 252.
- Inferred: the evidence this study can add is two new runs per arm. Its strength is closer to the conditional and
  new-runs-only numbers than to 1/35.

### 6.5 Descriptive only (not tested, not in the reading)
- Within-seed differences, stagger minus no change, for seeds 52 (A - local), 152 (ST2 - NC2) and 252 (ST3 - NC3).
- All other R1 and R2 readouts, per run, and the safety flags.

## 7. Reading lines (fixed by the orchestrator; report only, never in a stance4 verdict)

1. **Complete separation on both primary readouts in the stagger direction:** the term's continuation effect is real
   (inferred for from-scratch training).
2. **Complete separation on one only:** partial.
3. **Neither:** the continuation evidence for the term does not survive replication.

In every case: the from-scratch stance4 screen remains the deciding evidence.

**Consequence of section 6.3 (decision needed before launch, not taken here).** As fixed, line (1) cannot occur,
because K2a cannot separate completely. The best attainable reading is line (2), through the stagger sd. Two options:
- (a) keep the lines as they are. Then line (2) with complete separation on stagger sd and weak separation on K2a is the
  strongest possible result. readout.py reports this as it stands.
- (b) amend line (1) before any training step to read "complete separation on stagger sd and weak separation on K2a
  (p 4/35 = 0.114 at best)". That would be a weaker bar for K2a than the orchestrator intended.
This file implements (a). If (b) is chosen, it must be written into this file, with a timestamp, before the queue is
launched.

**Decision, 2026-10-20T01:48Z, before the queue was launched: option (a).** The reading lines stay exactly as fixed
above. Reasons:
- The lines were fixed by the orchestrator. Option (b) would lower the K2a bar below what was intended.
- (b) would set the bar after last night's K2a values were known (A 0, Colab 0, local 15). That is the same problem as
  section 6.4: fitting the bar to data already seen.
- Under (a) the outcome space is still informative: line (2) through stagger sd, line (3), or "incomplete".
Consequences:
- Line (1) cannot occur. The strongest attainable reading is line (2) with complete separation on stagger sd and weak
  separation on K2a. readout.py adds that qualifier to line (2) when it applies; the line itself does not change.
- readout.py line logic is unchanged (complete separation on each readout).

Inferred, for the reader: if this study gives line (2) through stagger sd, it is the best result the design allows. It
still rests on 2 new runs per arm (section 6.4).

A safety flag on a stagger run (a side door: touchdown, yaw, roll up, falls) is reported beside the reading. It does
not change the line, but a line (1) or (2) with such a flag is reported as "met through a flagged run".

## 8. Missing, stopped and failed runs (fixed now)

- **Ended on its own** (hop watch or collapse stop before E; ENDED_ON_ITS_OWN.json). A recorded outcome, never
  restarted, not judged at E. It takes the worst value on both primary readouts (ranked above every finished run).
  The test is also reported with that run dropped, as a sensitivity. Collapse cannot fire before 20 evaluations
  (this window has 12). The hop watch is live after 7M.
- **Crash** (the process dies before E without that file). queue.sh restarts it from its newest intact checkpoint.
  After 3 crashes it writes QUEUE_FAILED.json and stops restarting it. Such a run is left out; the study is reported
  below plan, with the test on the remaining runs.
- **Container restart.** Everything dies. Re-running queue.sh resumes each unfinished run from its newest checkpoint
  (at most 524,288 steps, about 25 minutes, are lost per run).
- **R2 does not reproduce the judge file.** Delete that npz and its .done marker and re-run queue.sh once. If it still
  does not reproduce, that run's stagger sd is missing: the stagger-sd test runs without it (below plan); its K2a still
  counts.
- **Pending.** While any judge or R2 file is missing, the reading is "incomplete".
- **Stagger sd not defined (amended before launch).** The window stagger sd is taken over the full-horizon episodes
  (last night's definition, unchanged). If the judge file has fewer than 2 full-horizon episodes, or the value is not a
  finite number, the run takes the worst stagger sd (ranked above every finished run, as an ended run). Its K2a counts
  as measured. readout.py shows the full-horizon count per run beside the primary table. Inferred: very unlikely (the
  worst run last night had 25 of 40).
- **Ended-run sensitivity (amended before launch).** It drops only the ended runs. It keeps the main test's filter
  otherwise, so a run whose R2 did not reproduce still counts on K2a there too.
- **Orphan judge file (amended before launch).** A container restart between the judge's write (judge_E.json) and the
  wrapper's (judge_E.exp.json) would leave a judge file the judge refuses to overwrite. queue.sh moves such a file
  aside to judge_E.orphan.<UTC>.json and judges the run again. The judge is deterministic, so the new file replaces it
  with the same content (inferred).

## 9. Compute, time and disk

- **CPU (4 cores, shared).** At most 2 training processes. At most 1 judge or rollout, and only while fewer than 3
  python processes machine-wide are busy (>= 50% CPU), so one core stays free for judging the Colab checkpoints.
  OMP/MKL/OPENBLAS 1 thread; iso.py pins torch to 1 thread.
- **The night's budget (amended before launch):** 2 trainers (this queue), 1 panels process, 1 core free for judging the
  Colab checkpoints. This queue's judge/R2 is a panels-type process. It takes the panels slot only when no other panels
  job is alive: queue.sh starts a post job only when pgrep finds no queue_controls.sh and no panels4.py process
  (QUEUE_PANELS_RE), in addition to the CPU check above. Tonight the control panels queue
  (SP/foot/stance4/panels/queue_controls.sh, nice 5) holds that slot first, so the replicate judges and rollouts wait
  for it to finish. Any other panels job started tonight should check that N4/logs/post.lock is free before it starts.
- **Rate.** Measured last night: 349-376 steps/s per training process (3 concurrent: 374-376; alone beside judges:
  356-359).
- **Wall time (inferred from those rates).**
  - One run: 1,572,864 steps at 350-375 steps/s is 70-75 minutes, plus about 1 minute to start.
  - First pair (NC2, ST2): about 71-76 minutes. Second pair (NC3, ST3): the same.
  - Judge (measured last night: 296 s) and R2 (130 s): about 7 minutes per run. The first pair's run during the second
    pair's training; the second pair's add about 14 minutes at the end.
  - **Total: about 2 h 40 min (range 2 h 35 min to 2 h 50 min)**, plus up to about 25 minutes per container restart and
    the time until queue.sh is re-run.
  - **Amended before launch (inferred):** the first pair's judges and rollouts now wait for the control panels queue
    (estimated 115-150 minutes). They then run during the second pair's training. The total becomes about 2 h 50 min to
    3 h, and longer if a Colab judge holds the free core when a post job would start.
- **Disk.** Measured on last night's local no-change run: models 23.7 MB, start copy 4.0 MB, tensorboard 1.6 MB, other
  records 1.3 MB; an R2 npz is 62 MB; a judge file 0.35 MB.
  - Peak, all four: about 4 x (30.5 + 62.4) = 372 MB.
  - After pruning: about 4 x (7 + 62.4) = 278 MB.
  - Free on / at 00:19 UTC: 5.28 GB.
  - Pruning (after a run is judged and rolled): only the E pair is kept (stage1_9961472_steps.zip and its vecnormalize
    pkl). The 8.91M and 9.44M pairs, best / robust_best, stage1_final and the start copy are removed; the Colab parent
    stays in S3/runs. QUEUE_PRUNE=0 keeps everything (about 20 MB more per run).
  - Amended before launch: every poll also re-runs the prune for finished runs (it is idempotent), so a prune cut short
    by a restart is completed.

## 10. Commands

Launch (not run here):

```
mkdir -p N4/logs && cd N4 && setsid nohup bash N4/build/queue.sh >> N4/logs/queue.out 2>&1 < /dev/null &
```

After a container restart, run the same line again. Dry run: `bash N4/build/queue.sh --dry-run` (starts nothing).
Readout at any time: `python3 -I N4/build/readout.py` (writes N4/readout.md and N4/readout.json).

The commands the queue runs, per run NAME with seed S (absolute paths in build/R1.json):

```
# train (detached; ST runs add: env.stance_stagger_weight=0.3 env.stance_stagger_tolerance_m=0.045 after --override)
cd N4 && ISO_WT=SP/fix/exp_night PYTHONPATH=S3/cf/tbsite setsid nohup python3 S3/tools/iso.py S3/night/build/continue.py \
  --name NAME --arm LT --seed S --start-zip <parent zip> --start-vecnorm <parent pkl> \
  --start-zip-sha256 c67576b6... --start-vecnorm-sha256 49ce291a... --override --stop-at 9961472 --runs-root N4/runs \
  >> N4/runs/NAME/train.log 2>&1 &
# judge (R1)
bash N4/build/judge_rep.sh NAME 9961472     # -> N4/judged/NAME/judge_9961472.json (+ .exp.json, .log)
# R2
cd N4 && ISO_WT=SP/fix/screen_s3_LT python3 S3/tools/iso.py N4/build/roll_rep.py --runs-root N4/r2/runs -- \
  --arm LT --run NAME --steps 9961472 --out N4/r2/out/NAME_9961472.npz
```

## 11. Files

- This pre-registration: N4/spec.md.
- Build: N4/build/queue.sh, N4/build/judge_rep.sh, N4/build/roll_rep.py, N4/build/readout.py, N4/build/R1.json.
- Checks (no training): N4/build/checks/ (queue dry run, continue.py argument checks for all four runs, a 2-episode
  judge smoke, a 2-episode R2 smoke and its comparison with last night's rollout, the permutation-test check,
  readout.py's dry run on last night's three runs, and a simulation of queue.sh's state machine with stand-in trainer,
  judge and rollout scripts: order, at most 2 trainers, crash restart, QUEUE_FAILED after 3 crashes, resume after a
  simulated restart, idempotent re-run).
- Made by the queue: N4/runs/<run>/, N4/judged/<run>/, N4/r2/{runs,out}/, N4/logs/queue.log, N4/readout.md,
  N4/readout.json.
- **Build files, full sha256, measured 2026-10-20T01:49Z after the amendments and before launch.** queue.sh pins the
  last three in its preflight (with last night's six harness files: 9 in all), so a changed file stops the queue.
  queue.sh cannot pin itself; its sha256 is recorded here.
  - N4/build/queue.sh      eb11c48af8a387153ac36fea00f108be467971812935739a2f7462919722760b
  - N4/build/judge_rep.sh  ac144468549c9e93ab7a7cef40086da40d8805115bd94774010d1d6c4cfbc537
  - N4/build/roll_rep.py   166e1b0d5a1ad24722e86e5ce9bb9d7b8edc8bc0b15123c83c2764af279959dd
  - N4/build/readout.py    6ef7ec98a5f10f36ddd2c16e450a9db69f3eb100d093e49a12f8904a164872e3
  - criteria.txt unchanged: 978a93da8da469c1acde7b9f7261d9236e4799eb2111b3af0f0cbeaa153f0f38 (measured 01:49Z).
  - build/R1.json still records the pre-amendment 12-character prefixes; the lines above supersede it.
- Checks of the amendments (no training): N4/build/checks/queue_dry_run_v2.txt (preflight ok with 9 pinned files; the
  commands are identical to the first dry run), N4/build/checks/queue_sim2/ (the new queue.sh with stand-in trainer,
  judge and rollout: posts wait while a fake panels4.py is alive; an orphan judge file is moved aside and re-judged;
  leftover checkpoints of a finished run are pruned on the next poll), N4/build/checks/review_fix/ (readout.py's dry
  run on last night's three runs, unchanged numbers; the conditional and new-runs-only tests on hypothetical values).

## 12. Amendments before launch (2026-10-20, 01:36-01:50 UTC; no training step had run)

From a review of this file and the build (N4/review_handling.md has each finding, the check and the decision):
1. Section 7: option (a) decided, with reasons. No change to the lines or to readout.py's line logic; readout.py adds a
   qualifier to line (2) when the other readout is weakly separated.
2. Section 6.4 (and 0): the 35-split p is optimistic because last night's three runs generated the hypothesis. readout.py
   adds two report-only numbers: the p conditional on last night's order, and the new runs alone (2 against 2) with the
   within-seed signs. Text after the readout's primary table says the same.
3. Section 8: orphan judge file handling (queue.sh).
4. Section 8: the ended-run sensitivity keeps the main test's filter (readout.py).
5. Section 8: an undefined or non-finite stagger sd ranks worst; the full-horizon count is shown per run (readout.py).
6. Section 9: the prune is re-run every poll for finished runs (queue.sh).
7. Section 9: the replicate judge/R2 waits while another panels job holds the night's panels slot (queue.sh).
8. Section 11: full sha256 of the build files; queue.sh pins judge_rep.sh, roll_rep.py and readout.py.
Nothing in the training commands, seeds, endpoint, readout definitions or reading lines changed.
