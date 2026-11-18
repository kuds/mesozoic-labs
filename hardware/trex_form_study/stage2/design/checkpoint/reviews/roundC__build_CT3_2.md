# build:CT3 (roundC, 17 Nov, second attempt): the agent's final report

Working paper. Nothing here is a rule.

CT THIRD FOLLOW-UP (rulings 10 and 11 of 17 Nov; sixth list, ruling 3). Branch wt/wpc-ct, worktree /home/user/mesozoic-wt/wpc-ct. Simulation tooling only: no run, no episode on a look set, no training. Tests ran on the stub env and the stub monitor only.

1. MERGE OF THE WORKING BRANCH
- I merged claude/mesozoic-trex-stage2-refine-4jffre (fc505b0d) as 6999e818, with the message written to a file and passed by -F.
- No conflicts. CT's six files were identical on both sides before the merge. 22 files of other packages came in, as they are on the working branch.

2. PER ITEM
(a) The flag is passed to the row.
- What changed (designlab/stage2/train2.py):
  - new `ALLOW_SOURCE_ENV = "STAGE2_ALLOW_SOURCE_CHANGE"`;
  - new `monitor_env(allow_source_change, base=None)`;
  - `_Run._child` starts the monitor child with `monitor_env(self.allow_source)` instead of `tbenv.child_env("disable")`;
  - the help text of `--allow-source-change` and the module docstring say so.
- The child gets "1" exactly when the session was given `--allow-source-change`. A value in the trainer's own environment is removed, not passed on.
- Test: tests/test_stage2_train2.py::test_the_trainer_passes_its_allow_source_change_to_the_row. A new stub-monitor hook `STUB_MON_ENV` logs what each child saw:
  - session 0 has no flag and the variable set in the trainer's environment: baseline and the check at 512 see none;
  - session 1 resumes with the flag: the checks at 1024 and 1536 see "1".

(b) A row's RowError before any episode is a refusal.
- What changed (designlab/stage2/monitor2.py):
  - `_episodes` builds the rows (`s2rows.configure` and `lookbar.rows`) in their own try block, before `lookbar.episodes`.
  - `_row_refusal` turns an exception into a refusal kind through `ROW_REFUSALS`: `s2rows.RowError` gives kind "row", `s2load.CheckpointError` gives kind "checkpoint".
  - The narrow rule: the class is looked up in sys.modules and is never imported for this, and the test is isinstance against that class. A class of the same name from another module is not a refusal.
- These stay non-refusals:
  - whatever `lookbar.episodes` raises;
  - a row's refusal at a further delay (rule M6, which runs after the check's own episodes). `_also` turns that into a MonitorError, exit 1.
- `REFUSALS` gains "run" and "row". `RULES["refusal"]` and the docstrings are updated.
- Test: test_stage2_monitor2.py::test_a_rows_refusal_before_any_episode_is_the_monitors, with stub `lookbar` and `s2rows` modules and a stub RowError of that name and module.

(c) The other exits before any episode are refusals.
- Missing run.json: `_job_doc(job_dir, refuse=True)` in `run` and `baseline` gives kind "run". Beyond the ruling's words, I also made "cannot be read" and "no JSON object" refusals (for the lead, item 3). `s2-10` keeps exit 1.
- Checkpoint record of the wrong shape: `_checkpoint` now checks `files`. Each of model, vecnormalize and fingerprint must be a dict with an int `bytes` and a str `sha256`. Otherwise kind "checkpoint". Before, this was an AttributeError and exit 1.
- Fingerprint not reproduced at the row's load:
  - `s2load.CheckpointError` raised from `lookbar.rows` is caught by the rule in (b);
  - in `_stand_in`, the S1 view's stand-in, the load is wrapped the same way.
- Tests:
  - test_the_other_exits_before_any_episode_are_refusals (three run.json cases at a check and at the baseline; 10 shapes of record);
  - the row test above (a CheckpointError with s2load's fingerprint text);
  - test_the_stand_ins_load_refuses_as_the_row (with `s2load.load_checkpoint` patched to raise).
- One earlier test pinned exit 1 for a missing run.json. That assertion is removed; the new test pins the refusal.

(d) The halt after a refusal under --retry-check names that parameter.
- What changed (train2.py, `_Run.check`): when the refused attempt was the one `--retry-check` let run (`after_halt`), the halt ends "with the fault repaired, a resume with --retry-check runs the check again (...)". Otherwise it still ends "a resume runs the check again".
- The fallback reason in `_refusal` and the docstrings now name the new refusals.
- Tests: the kill test below asserts the new text. In test_a_monitor_that_refuses_to_run_is_no_attempt I added an assertion that a plain refusal's halt does not name --retry-check.

(e) A kill around a refusal.
- I took in the reviewer's third scratch test as test_a_kill_around_a_refusal_and_the_halt_under_retry_check, with the drive hook `STUB_KILL_IN_REFUSAL=before_close` (only that mode). The sequence:
  1. two SIGKILLs between the monitor's exit 8 and the attempt's closing leave two interruptions;
  2. a plain resume halts "never closed";
  3. with --retry-check the monitor refuses and the run halts again; the refusal is not counted and the halt names --retry-check;
  4. a plain resume halts "never closed" again;
  5. a --retry-check resume ends at the run's end, bit for bit the uninterrupted run (checkpoints 1024 and 1536, updates.jsonl).

Each change has a test that fails without it. I ran a mutation script on a scratch git-archive copy: "15 of 15 mutations caught".
- 3 for (a), 4 for (b), 6 for (c), 2 for (d).
- Script and log: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rerun17/ct3b/mutate.py and /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rerun17/ct3b/mutate.log.

FOR THE LEAD
1. Which session's flag reaches the row. I built the ruling's literal reading: the session's own `--allow-source-change`.
   - Consequence: K9 at the row compares every session's sources, while the trainer's resume compares only the last session. So on a run once resumed on moved sources, a later resume without the flag (which the trainer accepts) will have its first check refused by the row. The run halts with exit 5, quoting the row's reason, until it is resumed with the flag again.
   - Alternative B: also pass the variable when `run.json` `flags.source_change` is set by an earlier session.
   - I recommend B. It needs your word.
2. A value the trainer's own environment holds is dropped. The alternative, inheriting it, would let a shell variable judge rows flagged without the trainer's flag. I recommend keeping it dropped.
3. A run.json that cannot be read or is no JSON object is a refusal, not only a missing one. The alternative keeps those as exit 1, a counted attempt that can end as a failed G8b. I recommend keeping them as refusals. A run.json without `mode` or `seed` is unchanged: a traceback, exit 1.
4. Every CheckpointError at the row's load is kind "checkpoint", the row's "another run's" case included. Mapping that case to "other_run" would need matching on the message text. Not done.
5. A row's refusal at a further delay (rule M6) is exit 1, a counted attempt. The alternative: build the rows of every delay before any episode, so that such a refusal comes first.
6. The session record does not note that the flag was given when the sources equal the last session's. The row's description holds `source_change` when it matters.

Owed outside this package:
- CW: s2rows must keep `RowError` under that name and module, and `SOURCE_CHANGE_ENV = "STAGE2_ALLOW_SOURCE_CHANGE"`.
- The Track B draft or design text: the new refusal kinds "run" and "row" in the note `monitor_<n>.refused.json` and in rule M1's list.
- CN: the notebook reads the new halt text.

3. TEST SUMMARY LINES (worktree, standing env, through tools/lowprio.sh, final code)
- tests/test_stage2_monitor2.py: "26 passed in 30.67s" (exit 0)
- tests/test_stage2_train2.py: "50 passed, 3 skipped, 2 warnings in 211.21s (0:03:31)" (exit 0). The 3 skipped are the STAGE2_SLOW tests, not run. Log: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rerun17/ct3b/train2_tests.log
- At the middle commit 9c348ca7 (git archive, `-k "refus or allow_source or monitor_env"`): "14 passed, 64 deselected in 68.24s (0:01:08)"
- ruff on the five package files: "All checks passed!"
- Record-key hashes: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1, the registered pair.
- New output paths: none in the worktree (status is empty). Scratch outputs are only under the scratchpad's rerun17/ct3b/. I had first put them in scratchpad/ct3b/ and moved them there. That is outside any repository, so `git check-ignore` does not apply.

4. DIFF AND STATUS
git diff claude/mesozoic-trex-stage2-refine-4jffre --stat:
 .../trex_form_study/designlab/stage2/monitor2.py   | 145 +++++++++++++----
 .../trex_form_study/designlab/stage2/train2.py     |  63 +++++--
 .../trex_form_study/tests/test_stage2_monitor2.py  | 181 ++++++++++++++++++++-
 .../trex_form_study/tests/test_stage2_train2.py    | 104 +++++++++++-
 4 files changed, 435 insertions(+), 58 deletions(-)
git status --porcelain: (empty)
clip2.py is unchanged.

5. COMMITS OF THIS ROUND, OLDEST FIRST
- 6999e818796c0e9a8e766ccc14e2f43a3c9ccf22: the merge of the working branch
- 695e84292bcc6c33859443c1adfdacc303c06089: the monitor's other exits before any episode are refusals
- 9c348ca7779e046783bacccf6f91ef5fb3052fe9: the trainer passes its --allow-source-change to the row
- eee35062f0dcae1e9f109b098a8df97358148b79: a refusal under --retry-check names it; the kill test

Branch head: eee35062f0dcae1e9f109b098a8df97358148b79. Each message ends with the two given lines and names no model elsewhere.

Commits on the branch not in the working branch that lack attribution lines at their end: none.
- 57b5bf36 and 2fca27bf end with the earlier model's two lines.
- The four of this round end with the given two.

NOT DONE
- No test against the real s2rows or lookbar: they are not merged.
- No real fingerprint mismatch on a real checkpoint: the CheckpointError was raised by a stub or a patched load.
- No STAGE2_SLOW tests and no real env.
- The reviewer's other two scratch kill tests are not taken in.
- Nothing pushed.
