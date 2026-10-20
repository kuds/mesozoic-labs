# Night2: how the review findings were handled

Written 2026-10-20, 06:48-06:55 UTC, before launch. Report only. Nothing here changes the stance4 criteria
(S4/criteria.txt, sha256 978a93da..., re-checked by the dry run at 06:50 UTC) or any stance4 verdict. Fails stay fails.

Paths: SP = scratchpad,
S4 = SP/foot/stance4, N2 = SP/foot/stance4/night2.

## What stayed fixed

- The review verdict was "clean": eight findings, all minor. Each was checked against the files. All eight are correct.
  None is rejected.
- No finding needs a change to the design or to a frozen file, and none was made. Unchanged:
  - N2/spec.md, mode 444, sha256 b1606f741ac0354b43ab28e82227550882d98064441b923b767992dba4fedc42 (= N2/spec.sha256;
    measured 06:48 UTC). The spec was frozen at 06:29 UTC and is not unfrozen for prose.
  - N2/build/FROZEN.sha256: all 14 entries OK (sha256sum -c, 06:48 UTC). queue2.sh pins spec.md and 15 files; the dry
    run at 06:50 UTC passed its preflight.
  - N2/build/B1.json (the build record) is not edited. Corrections are here.
- Points the spec leaves open are decided below, now, before launch. They are rules for the write-up. They do not touch
  the reading lines (spec section 7) or the primary readouts.

## Findings

### F1. Smoke step count (spec line 4). Accepted. Spec not unfrozen.

- Measured (build/checks/smoke_records/runs/*/sessions.json, loaded_steps and stop_at; train.log):
  CB52 ran 4,194,304 to 4,202,496 (8,192), then 4,202,496 to 4,218,880 (16,384). NC52 ran 4,194,304 to 4,218,880
  (24,576). Total 49,152 steps in two throw-away runs.
- Spec line 4 says 32,768. That is wrong. Section 12's per-run numbers are right.
- Action: the correct total (49,152) is recorded here and goes into the readout notes. If the spec is ever re-frozen
  for another reason, line 4 gets fixed then. The smoke outputs were deleted (checks/smoke_records/deleted_outputs.sha256),
  so no smoke step enters the study.

### F2. "CB and NC share every stream and start from the same first rollout" (spec section 3). Accepted.

- The spec states this as fact. It is inferred. The smoke did not test it.
- Measured: the CB smoke's session 1 printed no progress line (its stop at 4,202,496 fired at the end of its first
  rollout; train.log line 73). Session 2 restarted the seed streams at 4,202,496, so its first line (n=4,210,688) is
  not comparable with NC.
- Inferred support (code read): --seed goes to train_base.train(seed=...) (continue2.py line 385), which the
  reviewer traced to the PPO load, set_random_seed and the env seeds S..S+3; the band commit
  (c1e146a8..f94452ad) adds no random-number call outside its tests (grep of the diff); the band reads contacts only
  and does not change observations, so VecNormalize sees the same observations.
- Rule: the claim is written as inferred until measured. Free test after launch: the first progress line of each run
  (n=4,202,496) is computed from the parent policy's first rollout, before any update. For CB52 and NC52,
  mean_abs_foot_stagger_mm and mean_reward_stance_stagger must be equal, and NC52 must equal the NC smoke's
  4.9718 / -0.0019255 (smoke train.log line 65). The same equality is checked for CB60 and NC60 when they start (no
  smoke reference for s60).
- If the values match: the pairing claim is reported as measured for the first rollout. If they differ: the write-up
  drops "same first rollout" and says so. The reading lines do not change either way.

### F3. A post job that fails twice ("post_failed"). Accepted. Rule fixed now.

- Measured: post_failed is a terminal state in queue2.sh (state(), terminal()). readout2.py marks a run without its
  judge file, or without its R2 npz and .done marker, as "pending"; then the reading is "incomplete" and
  reading_line() returns no line. Spec section 8 has no rule for this case.
- Rule (handled like spec section 8):
  - The queue's own two attempts are the only attempts. No manual retry.
  - Judge failed: K2a is missing, and R2 cannot be checked against a judge file. The parent's comparison counts as
    "not lower" on both readouts (as section 8's crash case).
  - R2 failed (judge done): split sd counts as "not lower" for that parent; K2a counts as measured (as section 8's
    "R2 does not reproduce" case).
  - Guard rollout failed (judge and R2 done): section 8 already covers it. Guard readouts and guard flags are not
    used, and the note is printed. readout2.py handles this case itself (the run is not "pending").
  - In the first two cases the study is reported below plan. readout2.py will print "incomplete"; the write-up then
    applies reading_line()'s own rules by hand (a missing comparison counts as "not lower"), shows the inputs, and
    says that it did so.
- This rule can only remove support. It never makes a CB run "lower".

### F4. Guard readouts whose self-checks fail. Accepted. Rule fixed now.

- Measured (readout2.py lines 390-411): row["guard"] is set before checks_ok is read; a failed self-check only adds a
  note; the guard-flag block uses the guard numbers whenever both runs of the pair have them.
- Rule: a guard readout that fails any self-check of spec 5.3 is treated like a guard rollout that differs from R2
  (spec section 8). Its numbers are shown as report only, with the failed check named. The two guard flags of that
  pair are not used (a failure on either run of the pair voids them, since each flag compares the two). The reading
  is computed without them, and the note is printed beside it. If readout2.py's printed line used such flags, the
  write-up gives the line recomputed without them and says why.
- Reason: the self-checks test that the guard numbers mean what they say. A number that fails them is not a valid
  measurement, and section 8 already says "not used, note printed" for invalid guard data. This affects only the
  flag gate of line (1). It never changes K2a or split sd.

### F5. "Met through a run with a new safety flag" can point at the wrong parent. Accepted.

- Measured (readout2.py reading_line(), lines 136-144): `flagged` holds new flags from both parents, and the note is
  added whenever it is non-empty. The printed dict is keyed by the parent seed, so the flag itself is attributed,
  but the words "met through" are wrong when line (2) was met only through the other parent.
- Rule for the write-up: name the parent that carries the flag. Say "met through a run with a new safety flag" only
  if that parent is one through which line (2) was met (lower on both readouts for that parent, or the same readout
  lower for both parents). The line number is correct as computed. No code change (readout2.py is frozen).

### F6. busy_count matches arguments that start with "python". Accepted. No change.

- Measured: queue2.sh busy_count() tests `args ~ /^python[0-9.]* /`; tonight's S4/night/build/queue.sh line 130 used
  comm. A busy python started by an absolute path (/usr/bin/python3 ...) would not be counted by queue2.sh.
- Measured at 06:49-06:50 UTC: the only python process on the box was the stance4 panels worker (pid 31991,
  'python3 SP/foot/stance3/tools/iso.py SP/foot/stance4/panels/panels4.py ...'), which queue2.sh excludes on purpose.
  The dry run read 0 busy and named pid 31991 as the panels queue. The stance4 judge (S4/tools/judge_one4.sh lines 39
  and 89) and every job of this study start as 'python3 ...'. iso.py runs its script with runpy, so the argv stays.
- Rule: any extra manual python work tonight is started as 'python3 ...'. This difference from queue.sh is recorded
  here and in logs/launch.json.

### F7. Robustness gaps. Accepted. No code change; operating rules.

- (a) Every queue2.sh start re-checks the parent pairs and refuses on a mismatch. Measured: S4/tools/queue4.sh's only
  removal is its own stop file; nothing in S4 prunes S4/runs/stance4_LTS_s*/01_stance/models (all 4,194,304 pairs
  present at 06:49 UTC). Rule: the parent pairs stay in place until queue2.sh logs "queue2 done", so a relaunch always
  passes the preflight.
- (b) No timeouts on post jobs or trainers; a hung job holds logs/post.lock. Rule: read logs/queue2.log at each
  wake-up. Expected: judge about 4.6 min, R2 about 2 min, guard about 2.5 min (spec section 9). If a post job runs
  longer than about 20 min, or a panels job longer than about 25 min, look at its log and process by hand. A job
  killed by hand counts as a failed attempt (the queue bumps its counter); F3's rule then applies.

### F8. The launch command lacks the leading cd. Accepted.

- B1.json facts.commands.launch is 'bash .../launch2.sh' without a cd. launch2.sh cd's itself, so behaviour is the
  same; only the standing rule is affected.
- Launch and relaunch as:
  `cd SP/foot/stance4/night2 && bash SP/foot/stance4/night2/build/launch2.sh` (absolute paths in N2/RELAUNCH.txt).

## Summary

| finding | verdict | file change | rule added |
|---|---|---|---|
| F1 smoke steps 49,152, not 32,768 | correct | none (spec stays frozen) | correction carried in the notes |
| F2 pairing claim is inferred | correct | none | measured after launch from the first progress lines |
| F3 post_failed has no rule | correct | none | missing readout = "not lower", below plan, no manual retry |
| F4 failed guard self-check | correct | none | guard flags not used, note printed (as section 8) |
| F5 flag note can name the wrong parent | correct | none | write-up names the flagged parent |
| F6 busy_count uses args, not comm | correct | none | manual python work starts as 'python3 ...' |
| F7 parents and hung jobs | correct | none | keep parents; look by hand past 20 min |
| F8 launch without cd | correct | none | launch with cd (RELAUNCH.txt) |

## Added after launch (measured, 07:02 UTC)

The text above was written before launch. Its copy is N2/logs/review_handling.prelaunch.md (mode 444, sha256
bb6f4f9020d7ec4c1a30757b4138f7eeecbbdf8da8d5322de83df5e7a395fbbe). Only this section was added after.

- Launched 06:52:16 UTC as in F8: queue2.sh pid 745. Trainers: CB52 pid 932 (seed 52, the seven band keys, task
  7f95bf70...), NC52 pid 971 (seed 52, no override, task 3fa46708...). Details in N2/logs/launch.json.
- F2, s52: the first progress lines (n=4,202,496, the parent policy's first rollout) are equal. CB52 4.9718 /
  -0.0019255; NC52 4.9718 / -0.0019255; the NC smoke 4.9718 / -0.0019255 (mean_abs_foot_stagger_mm /
  mean_reward_stance_stagger). Equal at the 5 printed significant figures. The pairing claim is measured for s52's
  first rollout at that precision. s60 is still to check when CB60 and NC60 start.
