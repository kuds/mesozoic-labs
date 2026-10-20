# Night2: does the CoP band remove the loading split in continuation? A paired test from the stance4 parents

Pre-registration. Written 2026-10-20, 05:50-06:40 UTC, before any training step of this study. The only training so
far is the build smoke (section 12): 32,768 steps in two throw-away runs whose outputs were deleted. Report only:
- nothing here enters or changes a stance4 verdict, and nothing changes the stance4 criteria
  (SP/foot/stance4/criteria.txt, sha256 978a93da8da469c1acde7b9f7261d9236e4799eb2111b3af0f0cbeaa153f0f38, checked);
- fails stay fails;
- no Drive or GitHub writes, nothing in checkout, no commits, no pushes.

"Measured" means read from a file or computed by a script named here. "Inferred" is marked.
Paths: SP = scratchpad,
S3 = SP/foot/stance3, S4 = SP/foot/stance4, N2 = SP/foot/stance4/night2.

**This is a continuation test with n = 1 run per arm per parent. It is not a from-scratch screen.** At most it can
support putting a CB arm into the next from-scratch screen.

## 0. In short

- **Question.** In the stance4 runs at 4,194,304 the stagger term removed the positional stagger, but a left-right
  loading split remains (one foot on its heel, one on its toe). Does the per-foot CoP band remove that split when
  training continues?
- **Design** (fixed by the orchestrator). Two parents, stance4_LTS_s52 and stance4_LTS_s60, each at 4,194,304. Two
  arms per parent, with the parent's own training seed: CB (the parent's LTS task plus the CoP band at the
  prototype's proposed dose) and NC (no change). Four runs: CB52, NC52, CB60, NC60. Endpoint E2 = 5,767,168.
- **Primary comparison.** Within each parent, CB minus NC on K2a and on CoP split sd. Lower is better.
- **Reading lines** (section 7). (1) CB lower on both readouts for both parents, with no new safety flag: supports a
  CB arm in the next from-scratch screen. (2) Partial. (3) No support.
- **Limit found while building (section 7.3, inferred).** K2a cannot go below 0. If an NC run reaches K2a 0, its CB
  pair cannot be lower on K2a, and line (1) is out of reach for that parent.

## 1. Background (measured)

The parents at 4,194,304 (S4/judged/stance4_LTS_s*/judge_4194304.json; S4/panels/out/stance4_LTS_s*/panels_4194304.json):

| parent | verdict | clean / 40 | K2a | K2b | window stagger sd (mm) | CoP split sd | common CoP | heel min |
|---|---|---|---|---|---|---|---|---|
| stance4_LTS_s52 | FAIL | 28 | 12 | 16 | 4.8 | 0.989 | +0.067 | -0.840 |
| stance4_LTS_s60 | FAIL | 29 | 11 | 16 | 2.3 | 1.091 | -0.014 | -0.919 |

- Mirrored legs plus prey ahead remove the split (K2a 0 and 1; split sd 0.161 and 0.074). Inferred (tonight's panels
  note): the split persists as a per-foot loading split, driven by spawn asymmetry and the prey bearing.
- Tonight's replicate study (S4/night/readout.md): in continuation from stance3 Run 3, the stagger term separated
  window stagger sd completely from no change (n 3 against 4), but not K2a.
- The CoP band prototype (S4/copband/design.md; clone SP/fix/scratch_copband, branch scratch/cop-band, commit f94452ad
  on c1e146a8, never pushed): off by default and byte-identical when off; proposed dose w 0.30, tolerance 0.357,
  band [-0.6, +0.4], pad band [-0.65, +0.65], minimum load share 0.35; +45% env step time.
- By the fixed stance4 criteria all three Colab runs fail K1 at 4,194,304, and s52 and s60 fail K2. Decision line 5
  is determined. This study does not touch that.

## 2. The runs

| run | parent (Colab run @ 4,194,304) | arm | training seed | added env kwargs | run dir |
|---|---|---|---|---|---|
| CB52 | stance4_LTS_s52 | LTS + CoP band | 52 | the seven band keys below | N2/runs/CB52 |
| NC52 | stance4_LTS_s52 | LTS, no change | 52 | {} | N2/runs/NC52 |
| CB60 | stance4_LTS_s60 | LTS + CoP band | 60 | the seven band keys below | N2/runs/CB60 |
| NC60 | stance4_LTS_s60 | LTS, no change | 60 | {} | N2/runs/NC60 |

- **The band keys (CB only), all seven, at the prototype's proposed values:** stance_cop_band_weight 0.3,
  stance_cop_band_lower -0.6, stance_cop_band_upper 0.4, stance_cop_band_pad_lower -0.65, stance_cop_band_pad_upper
  0.65, stance_cop_band_tolerance 0.357, stance_cop_band_min_load_share 0.35.
- **Tasks (measured, build/checks/task_shas_*.txt and the smoke).** NC: sha256:3fa467083e34fa280ccb6d1bd04eebccdd3aac2b820fd89901b8d418ef32f2db,
  equal to the parents' LTS task. CB: sha256:7f95bf70a389c6d6c4263d5d1d805ec8ee6692facfb519543a00302df8864315.
- **Parents** (placed locally, A = B pkl, re-measured 05:5x UTC):
  - s52: S4/runs/stance4_LTS_s52/01_stance/models/stage1_4194304_steps.zip, sha256
    1773f3cf3f671360b5f474e6220a0588809e49aa322eb7625a72e297fda977bb; stage1_vecnormalize_4194304_steps.pkl, sha256
    532b7ca7bb60a85cd155e5b1955439405ae6ed989ab3675a67d3d93801c1b6bf.
  - s60: the same names under stance4_LTS_s60; zip 3ab1709a4abea8211403662c419b7f4c1ce95217a17aece52e0d9d35e0228c32,
    pkl d31199db3f2a2e67030e34bf82be2148906e08a40c88b608c066b467ef1f4f3b.
  - queue2.sh checks both pairs at every start; continue2.py checks them again when it copies them.
- **Endpoint.** E2 = 5,767,168 = 4,194,304 + 1,572,864 (three checkpoints of 524,288: 4,718,592, 5,242,880,
  5,767,168), the same span as last night's continuations.
- **Order.** At most 2 runs train at once, in the order CB52, NC52, CB60, NC60: the s52 pair first; each s60 run
  starts when a slot frees.

## 3. Seeds

How a seed S reaches the streams is unchanged from tonight's replicate spec (S4/night/spec.md section 3): training envs
S..S+3, global python/numpy/torch S, eval env S + 1000; a restart passes the same S again.
- CB52 and NC52 use 52; CB60 and NC60 use 60. These are the parents' own training seeds.
- So within a parent, CB and NC share every stream and start from the same first rollout. They differ by the reward
  term from the first update on. That is what makes the comparison paired.
- The Colab parents ran these seeds from step 0, so at 4,194,304 their streams are mid-stream. The local runs restart
  them at 4,194,304. Inferred: this does not favour either arm.
- The judge's --seed is the run's own seed, so its frozen joint-noise sigma is control_sigma[52] or [60], as in the
  Colab judge files of the same run. No seed relaxation is needed (unlike tonight's replicate study).

## 4. Harness (last night's, with the differences recorded)

- **Trainer.** build/continue2.py = S3/night/build/continue.py (sha256 42398f26...) with these changes and nothing else
  (build/continue2.diff): --arm LTS only, with the arm table from build/armcfg2.py (arm LTS = the clone's own stance
  stage; task 3fa46708...; the overlay changes nothing, as arm LT's did); default runs root N2/runs; schema and label
  names; the progress line also averages the band's info keys when present. Through S3/tools/iso.py (sha256 7749ca3f...).
- **Clone.** SP/fix/scratch_copband at f94452ad880933ca91c9aab422873ba72d1dec4c, clean, push URL disabled (measured).
  All four runs train in it. With the band off it is byte-identical to c1e146a8 (copband design 6.2, measured there).
- **n_envs 4**, DummyVecEnv (one process per run). The stance [curriculum] cadence: an evaluation of 10 episodes every
  131,072 steps, a checkpoint every 524,288 steps, keep all.
- **Learning rate and entropy.** train_base gets the remaining stage budget (12,000,000 minus the loaded steps), so the
  Colab schedules continue: lr = 1e-5 + 2e-5 (1 - n / 12e6), 2.30e-5 at 4.19M to 2.04e-5 at 5.77M; ent_coef =
  0.005 (1 - n / 7e6), 0.0020 at 4.19M to 0.0009 at 5.77M (measured in the smoke's progress lines at 4.21M).
- **Stop.** --stop-at 5,767,168. continue.py's callback ends learn() there; the E2 pair is saved first.
- **Task crossing.** continue.py's wrapper lets through exactly the seven added band keys (CB) or nothing (NC). Each
  crossing is recorded in <run>/scratch_task_change.json (measured in the smoke: parent 3fa46708 -> child 7f95bf70,
  exactly the seven keys).
- **Tensorboard** from S3/cf/tbsite on PYTHONPATH, as last night.

## 5. Readouts at E2

### 5.1 R1, the full judge (report mode)

build/judge_rep2.sh = S4/night/build/judge_rep.sh with these changes (build/judge_rep2.diff): the stance4 judge
(S4/judge/judge_stance4.py, sha256 3c8c0759..., unchanged) through build/judge_exp4.py (= S3/night/build/judge_exp.py
pointed at it; build/judge_exp4.diff), in the training clone scratch_copband at f94452ad, --seed = the run's own seed,
smoke cadence 8,192. Unchanged: --arm LTS, --stage-subdir ., --task-binding report, --allow-commit-mismatch, the
40-episode panel on seeds 3042-3081 (deterministic, heading 0), the statue on the same seeds, the joint and own-sigma
noise tests on spawns 3042-3061.
- **The LTS stance task binding.** The clone's stance.toml sets no band key, so the judged env is arm LTS's task with
  the band off (3fa46708...). The K numbers are the stance4 screen's.
- **Every relaxation is recorded** in the judge file: checks.relaxed (task_binding report, allow_commit_mismatch), and
  what they let through: checks.task (CB: the checkpoint and run record 7f95bf70, match false), checks.stage_config
  (CB: env difference stance_cop_band_weight; commit and seed match). readout2.py prints them per run. Measured in the
  smoke: NC exercises none of them; CB exercises the task binding only.
- Read with S3/night/cont/r1_summary.py summ() (unchanged): verdict, clean / 40 and LCB, K2a, K2b, roll rail, x_c
  minus statue, reward ratio, full horizon, hop-or-fall, panel falls, noise falls (joint, own), saturation episodes,
  criterion failures.

### 5.2 R2, the split rollout

build/roll_rep2.py = S4/night/build/roll_rep.py with three changes (build/roll_rep2.diff): roll_split.py's JUDGE is set
to judge_stance4.py (pinned; arm LTS is in its table; for L and LT it is judge_stance3's code), an optional
--expect-head check of the clone, and the dry-run line. roll_split.py itself is unchanged (sha256 dbb6aeea...). It
rolls in the screen clone SP/fix/screen_s4_LTS at c1e146a8 (as last night's R2 rolled in screen_s3_LT), on the judge's
panel. Read with S3/night/cont/nochange/scripts/nochange_compare.py r2_set() (analyze_night.py repro() and readouts(),
unchanged). A set is used only if, on all 40 seeds, the seeds, lengths, terminations, each foot's foot_cop_fore_aft,
the pad metric and the K2a/K2b flags equal the judge file's, and the model sha256 equals the judge file's.
Readouts: window stagger sd, CoP split sd, common CoP, heel min, ankle offsets (mean, r, l, |L-R|), and the rest of
analyze_night.ROWS.

### 5.3 Guard readouts (new)

A third rollout per run, the **guard rollout**: roll_rep2.py in scratch_copband at f94452ad, the same panel, with the
seven band keys added through roll_split.py's own --env-override. The band only observes (it is a reward term; it
changes no dynamics: copband design 6.1, measured). readout2.py requires the guard rollout's seeds, lengths,
terminations, qpos, actions, whole-foot and pad CoP, loaded flags and model sha256 to equal the R2 rollout's; otherwise
its guard numbers are not used. The same is done for NC runs, so the CB and NC numbers are measured the same way.

Over the window foot-steps (recorder index >= 200, the band's gate; every episode; both feet), as shares and counts:
- unloaded (the limb carries no floor force by the gate's rule);
- below the 0.35 load share;
- digit-only stance (the limb is loaded but its pad is not);
- not counted (any guard; the band's own `<s>_cop_band_counted`);
- pad band hit (pad CoP outside [-0.65, +0.65]) and pad band binding (the pad excess sets the foot's price);
- whole-foot CoP outside [-0.6, +0.4];
- the mean digit share; the band's cost per window step and per full-horizon window.
Self-checks: the band's counted flag equals the guard rule on the recorder's trace; its whole-foot and pad readings
equal the trace's; it pays nothing before the gate; the env's step counter agrees with the gate index.

### 5.4 Safety flags

nochange_compare.flags(), against the run's parent judge file at 4,194,304: clean below the parent for a new reason;
hop-or-fall >= 2; any panel fall; joint-noise falls >= 3/20 or own >= 2/20; any touchdown, all-feet-support,
settle-touchdown or yaw-change failure; saturation >= 2 episodes; roll up by >= 2; x_c outside [-60, +30] mm of the
statue; reward ratio < 0.95; hop_watch.json or ENDED_ON_ITS_OWN.json present. Plus two guard flags for CB runs: the
not-counted share, or the pad-band-binding share, at least 1.0 percentage point above its NC pair's.

### 5.5 Extra no-change references (no new training; context only)

The Colab runs' own judge files at 4,194,304 (the parents), 5,242,880 and 6,291,456 for s52 and s60, when they exist
(S4/judged/stance4_LTS_s*/judge_*.json), read with summ(), and the stance4 panels' standard-variant numbers at those
steps when they exist. They show where no change went on Colab. They are not paired controls (Colab hardware,
mid-stream seeds) and do not enter the reading.

### 5.6 Diagnostic panels (lowest priority, report only)

At E2, per run, the four variants of S4/panels/panels4.py (standard, mirror_legs, prey_ahead, mirror_prey_ahead;
sha256 4695ffdf..., unchanged), in screen_s4_LTS at c1e146a8, --no-statue (the statue does not depend on the
checkpoint; the parents' panels files hold it for these seeds), --allow-task-mismatch for CB runs only (recorded).
They run through build/panels_rep2.py, which only checks panels4.py's sha256 and runs it, so the stance4 panels queue
(which waits on any process named panels4.py) is not held back by them. A panels job starts only when no run waits for
its judge or rollouts, and only if it can end before 11:30 UTC at an estimated 15 minutes; otherwise that run's panels
are skipped and recorded (N2/panels/<run>/SKIPPED.json).

## 6. Primary comparison

Within each parent p (52, 60), CB minus NC on:
- **K2a** count (R1, of 40);
- **CoP split sd** (R2, sd over the full-horizon episodes of r - l whole-foot CoP).
Lower is better on both. "CB lower" means strictly lower. A tie is not lower.

Descriptive only (not in the reading): the differences themselves; "CB at or below NC"; a paired panel-seed bootstrap
of each difference (nochange_compare.boot(): 10,000 resamples of the 40 seeds, 95% percentile intervals) and exact
McNemar tests on the paired K2a and K2b flags. These cover panel sampling only, not run-to-run variance (n = 1 per arm).

## 7. Reading lines (fixed by the orchestrator; report only, never in a stance4 verdict)

1. **CB lower than NC on both K2a and CoP split sd, for both parents, with no new safety flag:** the band fixes the
   loading split in continuation (supports a CB arm in the next from-scratch screen).
2. **Lower on both readouts for one parent only, or on one readout for both parents:** partial.
3. **Otherwise:** no support.

In every case: n = 1 per arm per parent, and a continuation is not a from-scratch screen.

### 7.1 What the lines leave open, decided here before launch

- **"New safety flag"** = a flag in the reading's set that the CB run raises and its paired NC run does not. The set,
  from the orchestrator's list (guards firing, falls, saturation, reward ratio < 0.95):
  - guards: the two guard flags of section 5.4;
  - falls: any panel fall; hop-or-fall >= 2; joint-noise >= 3/20 or own >= 2/20; hop_watch.json or
    ENDED_ON_ITS_OWN.json present;
  - saturation >= 2 episodes;
  - reward ratio < 0.95.
  The other flags of section 5.4 are reported beside the reading and do not change it.
- **"On one readout for both"** means the same readout is lower for both parents. A crossed result (K2a lower for one
  parent, split sd lower for the other) is line (3).
- **Line (1)'s pattern with a new safety flag** reads as line (2), and the flag is named.
- **A line (2) met through a run with a new safety flag** is reported as such ("met through a run with a new safety
  flag").
- Implemented in readout2.reading_line(); 13 hand-made cases pass (build/checks/reading_line_check.txt).

### 7.2 Why these readouts

K2a is the judged statistic the split fails (K2a: pad CoP beyond 0.80). CoP split sd is the split itself. The band
prices each foot's whole-foot CoP and its pad CoP, so it can lower both. The guard readouts and the stagger and ankle
readouts are the ones the band cannot pay directly (copband design 5 and 8.6).

### 7.3 A limit (inferred)

K2a cannot go below 0. If an NC run reaches K2a 0 by E2 (tonight's stance3 Colab no-change run did, from K2a 6, over
the same span), CB cannot be lower on K2a for that parent, and line (1) cannot be reached there. The reading lines are
the orchestrator's and stay as fixed. readout2.py also prints "CB at or below NC" (report only), so such a case is
visible. Measured at 06:30 UTC (the stance4 judge queue's new files): the Colab runs' own no-change continuation at
5,242,880 still has K2a 12 (s52) and 11 (s60), so the floor did not bind there.

## 8. Missing, stopped and failed runs (fixed now)

- **Ended on its own** (hop watch or collapse stop before E2; ENDED_ON_ITS_OWN.json). A recorded outcome, never
  restarted, not judged. It takes the worst value on both primary readouts: a CB run that ended is never lower; an NC
  run that ended is beaten by any finished CB run. It also raises the falls flag. Inferred: collapse cannot fire here
  (fewer than 20 evaluations in the window) and the hop watch is live only after 7M.
- **Crash** (the process dies before E2 without that file). queue2.sh restarts it from its newest intact checkpoint.
  After 3 crashes it writes QUEUE_FAILED.json and stops. That run has no E2 readouts. Its parent's comparison counts
  as "not lower" on both readouts, and the study is reported below plan.
- **Container restart.** Everything dies. Re-running build/launch2.sh resumes each unfinished run from its newest
  checkpoint (at most 524,288 steps lost per run) and skips finished work.
- **R2 does not reproduce the judge file.** Delete that npz and its .done marker and re-run launch2.sh once. If it
  still does not reproduce, that run's split sd is missing (counts as "not lower"); its K2a still counts. Below plan.
- **Guard rollout differs from R2, or is missing.** Its guard readouts and guard flags are not used, and readout2.py
  says so. The reading is computed without the guard flags and the note is printed beside it.
- **Pending.** While any judge or R2 file is missing, the reading is "incomplete".
- **Split sd not defined.** If the judge file has fewer than 2 full-horizon episodes, or the value is not finite, the
  run takes the worst split sd. Its K2a counts as measured.
- **Orphan files.** A judge file without its .exp.json, or an incomplete panels file, is moved aside and redone.

## 9. Compute, time and disk

- **CPU (4 cores, shared).** At most 2 training processes (this study). One post job at a time (judge, then R2, then
  guard rollout; or a panels job), and only while fewer than 3 python processes are busy (>= 50% CPU), not counting the
  stance4 panels queue's panels4.py process (matched by '^python3 <iso.py> SP/foot/stance4/panels/panels4.py '). The
  queue never waits for the panels queue. Already running elsewhere: the stance4 judge queue (1 process when busy) and
  the stance4 panels queue (1 process). All process matching is anchored on the interpreter and the script path (no
  match on shells whose command text names a script). OMP/MKL/OPENBLAS 1 thread; iso.py pins torch to 1 thread.
- **Rates.** NC: 378-398 steps/s per process with 2 concurrent trainers (measured last night on the LT harness; the LTS
  stagger term is cheap, inferred). CB: about 260-320 steps/s (inferred from the band's +0.62 ms per env step, and
  from the smoke, which ran beside other jobs).
- **Wall time (inferred).**
  - NC run: 1,572,864 steps in about 66-70 minutes. CB run: about 82-101 minutes.
  - Training: CB52 and NC52 start together; CB60 starts when NC52 ends (about 70 min); NC60 when CB52 ends (82-101
    min). Both s60 runs end at about 150-170 min.
  - Post job per run: judge about 4.6 min (measured 275-281 s on the stance4 J checkpoints), R2 about 2 min (measured
    115-120 s last night), guard about 2.5 min (inferred: the smoke's guard roll took 1.2x the R2 roll). The s52 pair's
    run during the s60 pair's training when the CPU cap allows; the s60 pair's add about 20 min at the end.
  - Panels: about 8-13 min per run (inferred: 4 variants, no statue); the s52 pair's can run during the s60 training.
  - **Total: about 3 h 10 min to 3 h 40 min after launch** (inferred), plus waits for the stance4 judge queue and up
    to about 30 min per container restart. Launched by 07:00 UTC, the core readout lands by about 10:40 and the panels
    by 11:30 at the latest (the panels deadline).
- **Disk** (free on /: 4.4 GB, measured 06:2x UTC). Per run at peak about 35 MB of checkpoints and records; after the
  prune about 8 MB. An R2 npz is about 63 MB and a guard npz about 66 MB (extrapolated from the 2-episode smoke files:
  3.15 and 3.29 MB). Peak, all four: about 4 x (35 + 129) = 660 MB; after pruning about 550 MB. The queue starts
  nothing new below 1 GB free.
- **Pruning** (after a run's judge, R2 and guard rollout succeed): only the E2 pair is kept (stage1_5767168_steps.zip
  and its vecnormalize pkl; the panels read it). The 4.72M and 5.24M pairs, best, robust_best, stage1_final and the
  start copy are removed; the parents stay in S4/runs. QUEUE_PRUNE=0 keeps everything.

## 10. Commands

Launch (not run here):

```
bash N2/build/launch2.sh            # = cd N2 && setsid nohup bash N2/build/queue2.sh >> N2/logs/queue2.out 2>&1 < /dev/null &
bash N2/build/launch2.sh --dry-run  # preflight, state and every command; starts nothing
python3 -I N2/build/readout2.py     # readout at any time (writes N2/readout2.md and N2/readout2.json)
```

After a container restart, run launch2.sh again. The per-run commands (absolute paths in build/B1.json and in the dry
run):

```
# train (detached); CB runs add after --override: env.stance_cop_band_weight=0.3 env.stance_cop_band_lower=-0.6
#   env.stance_cop_band_upper=0.4 env.stance_cop_band_pad_lower=-0.65 env.stance_cop_band_pad_upper=0.65
#   env.stance_cop_band_tolerance=0.357 env.stance_cop_band_min_load_share=0.35
cd N2 && ISO_WT=SP/fix/scratch_copband PYTHONPATH=S3/cf/tbsite setsid nohup python3 S3/tools/iso.py N2/build/continue2.py \
  --name NAME --arm LTS --seed S --start-zip <parent zip> --start-vecnorm <parent pkl> --start-zip-sha256 <..> \
  --start-vecnorm-sha256 <..> --override [band keys] --stop-at 5767168 --runs-root N2/runs >> N2/runs/NAME/train.log 2>&1 &
# R1
bash N2/build/judge_rep2.sh NAME 5767168                    # -> N2/judged/NAME/judge_5767168.json (+ .exp.json, .log)
# R2 (screen clone, band off)
cd N2 && ISO_WT=SP/fix/screen_s4_LTS python3 S3/tools/iso.py N2/build/roll_rep2.py --runs-root N2/r2/runs \
  --expect-head c1e146a8... -- --arm LTS --run NAME --steps 5767168 --out N2/r2/out/NAME_5767168.npz
# guard rollout (scratch_copband, the band observing)
cd N2 && ISO_WT=SP/fix/scratch_copband python3 S3/tools/iso.py N2/build/roll_rep2.py --runs-root N2/r2/runs \
  --expect-head f94452ad... -- --arm LTS --run NAME --steps 5767168 --out N2/guard/out/NAME_5767168.npz \
  --env-override stance_cop_band_weight=0.3 ... (all seven)
# panels (lowest priority)
cd N2 && ISO_WT=SP/fix/screen_s4_LTS python3 S3/tools/iso.py N2/build/panels_rep2.py N2/r2/runs/NAME --steps 5767168 \
  --clone SP/fix/screen_s4_LTS --out N2/panels/NAME/panels_5767168.json \
  --variants standard mirror_legs prey_ahead mirror_prey_ahead --no-statue --expect-head c1e146a8... [--allow-task-mismatch for CB]
```

## 11. Files and frozen sha256

- This pre-registration: N2/spec.md (mode 444); its sha256 in N2/spec.sha256. queue2.sh refuses to start if spec.md
  changed.
- Build: N2/build/{queue2.sh, launch2.sh, continue2.py, armcfg2.py, judge_exp4.py, judge_rep2.sh, roll_rep2.py,
  panels_rep2.py, readout2.py} and the diffs {continue2, judge_exp4, judge_rep2, roll_rep2}.diff against the files
  they come from. Record: N2/build/B1.json.
- **Build files, full sha256, measured before launch.** queue2.sh pins all of them, plus the reused harness files
  (iso.py, judge_stance4.py, roll_split.py, panels4.py) and the four files the diffs are taken against. queue2.sh
  pins spec.md, so its own sha256 (and launch2.sh's) is in N2/build/FROZEN.sha256, not here.
  - N2/build/continue2.py    3381292e1ff3096ffd637e0d8f66856c1b92a58c1fc69764494edb324b01b203
  - N2/build/armcfg2.py      d803780f41fa92779db9c936dbb37b91e03fac2c58ae900a5c9ac683b25f83f4
  - N2/build/judge_exp4.py   b7e5eead4bdb3b650c3bd885f2473c24d4854970e09d31e4e4d2f210329779c0
  - N2/build/judge_rep2.sh   60cb10bf1493bbb85ae6a4413be098ea1afdb0437f00b83931e18a6876cf1a34
  - N2/build/roll_rep2.py    ff80b1eae5bb479b508bdd622690fa5811adfc393dae240fb19fdf4032ee0caa
  - N2/build/panels_rep2.py  a9aabfc708ed3ae508cedc6bfef03c87177fd18fd6f5dbe9e3e4155ab667499f
  - N2/build/readout2.py     f91e447066725bc82740c4464896254c9b972fd76d11ca02ec42bb91ddd5bb30
  - criteria.txt unchanged: 978a93da8da469c1acde7b9f7261d9236e4799eb2111b3af0f0cbeaa153f0f38.
- Made by the queue: N2/runs/<run>/, N2/judged/<run>/, N2/r2/{runs,out}/, N2/guard/out/, N2/panels/<run>/,
  N2/logs/queue2.log, N2/readout2.md, N2/readout2.json.

## 12. Checks made while building (measured; N2/build/checks/)

- **Task shas** (task_shas.py in both clones): NC = 3fa46708... (= the parents' task), CB = 7f95bf70...; the plant
  identity (physics 3d68aec7..., policy interface ffbf6631...) is the parents'.
- **Training smoke** (CB52 into a throw-away runs root, --smoke-cadence 8,192; outputs deleted afterwards, records kept
  in checks/smoke_records/): session 1, one PPO rollout (8,192 steps, 2,048 per env) from the copied parent pair
  (sha256 checked); session 2 resumed from its 4,202,496 checkpoint (the restart path) and ran two rollouts to
  4,218,880. The progress line of session 2's first rollout: mean reward_stance_cop_band -0.0696 per step; the share
  of steps with a non-zero band reward 0.619 (the rest are the first 200 steps of each episode and in-band steps);
  counted share r 0.995, l 0.994; band quality 0.689. So the band's info keys reach the training env and the term is
  non-zero after the settle. The task crossing was recorded with exactly the seven keys. An NC52 smoke (24,576 steps)
  logged no band key (the band is off there).
- **Judge smoke** (judge_rep2.sh --smoke-episodes 2 on both smoke runs at 4,218,880; rc 0, about 30 s each): the judge
  rolled 3fa46708; CB recorded its checkpoint and run task 7f95bf70 (match false) and the env difference
  stance_cop_band_weight; commit, seed, n_envs, hyperparameters, mirror map and plant all match; NC matched on
  everything.
- **R2 and guard smokes** (2 episodes each): R2 reproduced the judge files (model sha256 equal); the guard rollouts'
  trajectories equal the R2 rollouts' on every compared array; the guard self-checks pass.
- **Panels smoke** (CB52, 2 episodes, four variants, no statue, --allow-task-mismatch): ran; its standard variant
  equals the R2 numbers (stagger sd 1.673 mm, split sd 0.616, common +0.049, heel min -0.404).
- **readout2.py** on the smoke pair: ran end to end (R1, relaxations, R2, guard, flags, McNemar, Colab references,
  parents' panels). The bootstrap needs 40 episodes and is skipped, with a note, on smaller sets.
- **Reading lines:** 13 hand-made cases (reading_line_check.txt).
- **queue2.sh:** a dry run (queue2_dry_run.txt: preflight ok, 15 pinned files) and a simulation of the frozen queue
  with stand-in trainer, judge, rollout, panels and readout scripts (queue_sim/; queue2_sim.diff shows the only changed
  lines: paths and no pins). All seven cases behaved as specified: order CB52, NC52, CB60, NC60 with at most 2
  trainers; a crash restarted; 3 crashes give QUEUE_FAILED; post jobs one at a time, judge then R2 then guard, then the
  panels; posts wait while 3 or more python processes are busy and start when the count drops, with a stand-in stance4
  panels-queue process alive and not counted; a shell whose command text names the scripts is not taken for a trainer;
  a past deadline skips the panels with a record; a killed and relaunched queue resumes without duplicate trainers; an
  orphan judge file and an incomplete panels file are moved aside and redone; a rerun after completion only re-reads.
- readout2.py on the empty study tree (readout_dry/): "incomplete", with the parents and the Colab references filled.
