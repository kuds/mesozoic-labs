# review:CT (roundC, wave 3): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

REVIEW OF WORK PACKAGE CT (train2.py, monitor2.py, clip2.py and their tests), branch wt/wpc-ct at bcaea22
Simulation only; nothing here is a result. I changed no tracked file and committed nothing.

VERDICT: merge after the listed fixes (findings 1 and 2). Kill and resume, the rules on records and the acceptance all hold; the two major findings are in the probes (rules O3, O4, K8), are small, and lie inside the package's own files. Findings 3 to 10 can follow in the same round or the next.

FINDINGS, most serious first

1. MAJOR. A later probe call overwrites the validation record and drops what earlier calls decided (train2.py 1507-1521 probe_lr, 1601-1608 filter_choice; read back at 290-306; used at 733-735).
   Rule O3 as settled has heads probed at different times (the -X head later, 60 cm later) and "not run again". Rule O4 has 60-N-T-10 measured later, "reported".
   Evidence (my case C8, stub): after probing 'stub': rates_by_body {'stub': {'rate': 1e-06, 'heads': {'stub': 1e-06}}}; after probing 'stub-X' into the same record: {'stub': {'rate': 1e-07, 'heads': {'stub-X': 1e-07}}}. The base head is gone, so the body's rate is no longer "the smallest over its probed heads".
   By the code, filter_choice for a body other than 75-M-H-10 writes choice None. _validation then returns no filter and the envs take R5Config's default 10 Hz without a word. The same happens when filter.json is missing: a real run starts on 10 Hz, whereas a missing rate is refused.
   Fix: probe_lr and filter_choice merge into an existing record, refuse a head already probed, and recompute rates_by_body over all recorded heads; the rule body's choice is kept. A run that is not tiny refuses without a filter record unless filter_hz is given by --config. One test each.

2. MAJOR. Rule K8's probe builds R5 envs and today's default configs, whatever the fine-tune is (train2.py 1487-1488; called from 901-902 without the job's mode or config).
   Evidence (C7): train(mode='R6', parent=<R6 run>) made the calls ('env_configs','R5'), ('env_specs','R5',['R5']) for the probe, then ('env_specs','R6',['R6']) for the job. The action width is nj + 1 in both modes, so an R6 policy loads and is probed on R5 envs. --config overrides of the job are ignored the same way.
   Fix: probe_lr(parent=...) takes the mode and the env overrides of the fine-tune; a test on the seam.

3. MINOR, and a rule point for the lead. A monitor child that refuses because the sources moved is counted like a crash (train2.py 1106-1113; monitor2.py 615-623, exit 1).
   Evidence (C9): attempts at 512 [(1, False), (1, False)], counted_without_record True, G8b pass False, all six mapped tasks doubled, run status done, exit 0. That is rule M1 to the letter, but the refusal is deterministic. In a local run (S2-4, the 60 cm run) a commit touching a source file would change a reading.
   I would have monitor2 leave with its own exit code for a refusal and the trainer halt (exit 5) on it.

4. MINOR. RESUME_RULE, the text stored in every run.json, gives the order settings, sources, versions, inputs (train2.py 144-158). The code and the design refuse inputs before sources (592-596). C6: with a changed file and changed sources the refusal is "other inputs". Fix the text.

5. MINOR. After a kill between a checkpoint's last file and run.json, run.json's checkpoints list loses that step for good (train2.py 1004, 1162-1173). C1: [0, 256, 512, 1024, 1280, 1536] against the uninterrupted [0, 256, 512, 768, 1024, 1280, 1536]. The final checkpoint is still equal. Fix: _rollback adds the step it resumes from.

6. MINOR. A monitor_<n>.json that was cut makes every resume fail (train2.py 1086-1088; r0l.json likewise at 1030). C3: both resumes exit 1 with JSONDecodeError. By M1 such a file is no record: move it aside and run the check.

7. MINOR. If the trainer alone is killed during a check, the monitor child lives on and the resumed trainer starts a second one on the same check (train2.py 1015-1024). C4: 3 s after the kill ['monitor child'] is alive, and it is still alive after the resumed run ended. The workers do exit (C5). A group kill is clean. Fix: record the child's pid in the attempt and end a live one at the resume.

8. MINOR. survival_credit gives the full survival time when planned_s or the clock block is missing (monitor2.py 229-232). My table: a clock at 0.5 is credited 6.0 with the clock and 12.0 without. It fails open where the credit exists to stop a slowed clock. For a policy row, refuse the table or flag the episode.

9. MINOR. progress.json does not say that a check was counted without a record, nor a halt (train2.py 1193-1204). The launcher sees only N 0, S 0, and M1 says "the lead is told".

10. MINOR, small.
   - PYTHONHASHSEED is checked only in the CLI's run (1776), not in train() nor in the probe subcommands.
   - _baseline makes a lost r0l.json again and replaces run.json's baseline without comparing the episodes (1031-1050).
   - Commit e52b153's title says "and their tests"; it holds monitor2.py only.

CHECKED AND FOUND RIGHT
- Diff against the working branch: the five files only (4,887 insertions). Each of the four merge commits equals the working branch outside those files.
- All 11 commits not in the working branch end with the two required lines and name no model elsewhere.
- git merge-tree --write-tree onto fb421ba: exit 0, the tree equals HEAD's, five files added.
- Hash line, before and after my runs: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1.
- git status --porcelain is empty in the worktree and in the main checkout. The builder's remaining outputs are a stub run, tiny R5 and R6 runs on S1 and one clip.
- The interfaces, run.json, checkpoint run state, updates.jsonl, PPO construction, budget rounding (2,007,040; 5,005,312), boundary seeds and the four pool rulings match the design, the brief and the rulings.
- External SIGKILL of the process group at random moments on the stub: 23 trials, 34 kills that landed (before run.json, in the baseline, in rollouts, in the monitor children at 512, 1,024 and 1,536, at the end), resumed with 0 to 3 workers. 0 problems: final checkpoint, updates, fix, gates, kept files equal to the uninterrupted run; nothing left alive; no .tmp file.
- A kill after the monitor wrote its record and before the attempt was closed (C2): the record is read, not made again; the result is equal.
- Real env, S1 view, tiny, 2 workers, through the CLI:
  - R6 uninterrupted: exit 0, "done at 4096 steps".
  - R6 killed past the check at 2,048 and resumed: exit 0; checkpoint 4,096 EQUAL in every tensor, optimizer state, statistics with counts and fingerprint; updates equal.
  - R5 killed while the monitor child of 2,048 ran (attempts [(0, interrupted), (1, exit 0, record)]): resumed, exit 0, EQUAL to the uninterrupted tiny R5 run.
- Would the tests fail if the behaviour broke? 11 mutations on a scratch copy, each caught by its test: no reseed at a boundary; weights not restored at a resume; updates not cut back; no boundary at a resume; attempt not logged before the child; monitor runs on other sources; checkpoint saved again at a resume; a final check applies the stop; fix recorded again at a replay; second G8b reading at "fix + 5M"; R0L dropped from N*, S*.
- Rule cases on decide: S2-4's 500k checks make no G8b reading; a failed 5M then a 10M check without a record stops; the 60 cm run's 5M reading is its final check.

TEST SUMMARY LINES (mine, through tools/lowprio.sh)
- tests/test_stage2_monitor2.py: "21 passed in 18.65s", exit 0
- tests/test_stage2_train2.py: "33 passed, 3 skipped, 1 warning in 155.92s (0:02:35)", exit 0
- STAGE2_SLOW=1 ... -k tiny: "4 passed, 32 deselected in 270.29s (0:04:30)", exit 0
- Tiny R5 at 2 workers, load about 1.9, so provisional: 408 env steps/s first rollout, 448 first three; trainer 518-526 MB, each worker 523-530 MB.

THE BUILDER'S POINTS FOR THE LEAD
- a (a fix never changes the settings): keep. The design's wording would change the settings hash that K4 and every checkpoint compare; correct design 3.6.
- b (second G8b reading at the multiple 10M, only after a failed first): keep; the rule as corrected wins over the design. Confirmed: pass at 5M, down at 10M gives no reading and no consequence. Option: record the 10M reading as reported.
- c (the final check makes no fix and no stop): keep; the budget ended the run. Tables should name the ending "budget" and show final_check beside it.
- d, e, f, h, i, j, k, l: agree as built. For e the launcher must repeat every non-default argument on a resume. For j the budget arithmetic must count checkpoint and check time separately.
- g (one rate per body from rates_by_body): right by rule O3 as settled, but see finding 1.
- --config, --retry-check, exit codes 4 and 5, "not_run": accept. CW must return "not_run", and CN must not retry 4 or 5.
- The stand-in in monitor2: keep it for this merge, since acceptance needs a baseline and S1 episodes are allowed. Delete it and its test in the round that merges CW; after that it is dead and untested.
- Rule M3's "r0l.json must equal the committed baseline, else stop before the first rollout": add --expect-baseline PATH to train2, compared by monitor2.baseline_equal after the baseline, halting with exit 5. The other option, copying the committed file into the job, makes the equality empty.
- No timeout on the monitor child: keep it so. A timeout would turn a hang into a closed attempt and then a failed reading. Let the launcher watch the open attempt's started_at and kill the group, which is an interruption.
- "R0L exactly" holds to rounding (seen: S 19.490963612274165 against mean survival 19.490963612274705). Either say "to rounding" in the rule or clamp the ratio within 1e-9 of 1.
- Fingerprint clipped_rows 256 of 256 at 4,096 steps: confirmed. It is the policy-files package's matter; look at it on a long run.

FOR THE NEXT PACKAGES
- CN: the trainer raises if a monitor record that run.json names is missing, so copy monitor/ with run.json.
- CG: checkpoint_every is not on the command line, so K3's gate needs its own driver around train().

NOT CHECKED
- monitor2 against the real lookbar and s2rows; any frozen set (none was run); g8a.short and bar_form through the trainer; probes on a real set; -X bodies.
- More than 3 workers on the stub or 2 on the real env; other library versions; a Drive mount; clip2 on a policy row's trace.
- The full test suite (not run, by rule); s2load, vecpool and r5env beyond the interfaces the trainer uses.
- The stub's rollouts last milliseconds, so most random kills fell in the baseline or a monitor child; kills inside rollouts rest on the package's own tests and my two real-env kills.

Everything of mine is in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_ct_wave3/ (scripts kill_fuzz.py, checks2.py, mutate.py, real_kill.py, decide_cases.py; logs fuzz.log, checks2.log, mutate.log, mutate2.log, real_kill.log, monitor_tests.log, train_tests.log, slow_tests.log).
