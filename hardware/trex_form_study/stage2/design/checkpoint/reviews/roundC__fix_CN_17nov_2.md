# fix:CN (roundC, 17 Nov, second attempt): the agent's final report

Working paper. Nothing here is a rule.

WP CN (the notebook): fix round report, 17 Nov 2026. Worktree /home/user/mesozoic-wt/wpc-cn, branch wt/wpc-cn, head a003c31e566601d1f0e248aeb8342874b946eb88. Simulation tooling only. Nothing here is a result, and there are no hardware measurements. Nothing was pushed. No history was rewritten, no network call was made, no real Drive was touched. Nothing was written in the main checkout. The acceptance run was NOT run.

Note: the earlier fixer left four commits, not one: ef80550, cc6cef5, 36347d4 and b54c0aa. I built on all four.

1. MERGE OF THE WORKING BRANCH
- Merge commit 5bdd5aa8 brought in the working branch at fc505b0.
- It touched 12 document files: the 17 Nov rulings, STATUS.md, and the CN/CW/CT3 build, review, fix and check reports.
- No conflicts, and no package file changed.
- The record-key hash pair is the registered one: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`.

2. PER FINDING
Reproductions:
- I ran the review's scripts (exp1 to exp5, copied to `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rerun17/cn/repro/`) on scratch copies of 73a5112 (before any fix) and of 5bdd5aa.
- I wrote one more script, exp7_finish_keep.py, and ran it on 4d2ce5f2.
- Logs: run_1_2.log, run_2_head.log, run_3_4_5.log in that directory.

MAJOR 1, a job's last copy (ruling 1)
- Reproduced on 73a5112:
  - exp1: "result state: done | record: DONE.json", "Drive checkpoints: [1024, 2048, 3072]"; the local copy holds up to 4096.
  - exp4: "launch state: done", then "finish RAISED: OSError [Errno 107]".
- On 5bdd5aa (the earlier fixer's ef80550):
  - exp1: Drive holds [1024, 2048, 3072, 4096].
  - exp4: "launch state: unfinished"; finish does not raise.
- Still open there, shown with exp7 on 4d2ce5f2:
  - (a) A run that halted after an earlier job's last save failed to reach Drive: the finish step returned exit code 5, not 6.
  - (b) Cell 8, given exit code 6: "released: True".
  - (c) A finish run after Drive came back: "unfinished on_drive False", and checkpoint 4096 was not on Drive.
- Fixed in d97f9383 (nb.py):
  - `end_copy_waits` and `_Launcher.settle`: fenced tries with a growing wait of 30, 60, 120, 240, 300, 300 and 150 s at the 60 s period, 20 minutes in all. This replaces END_COPY_TRIES.
  - `finish` (now `finish` plus `_finish`):
    - It tries the missing checkpoint and tar copies again, through `_drive`.
    - It writes DONE.json when the whole queue is finished and on Drive.
    - It never raises.
    - It records each run in the session's `finishes` and moves `last_status_at`.
  - The pack tool's try count is kept in the session state.
  - `_step_finish`: exit code 6 now comes before 5.
  - New `release_or_keep` and `KEEP_RUNTIME_TEXT`. Cell 8 calls them: on exit code 6 the runtime is kept and one plain paragraph is printed.
  - `hours`: lists the sessions whose last finish kept the runtime (`kept_runtime`).
- Tests:
  - test_the_end_of_job_copies_wait_longer_each_time_for_at_most_20_minutes
  - test_finish_makes_the_last_copy_and_ends_the_run
  - test_the_finish_keeps_the_runtime_while_a_save_is_not_on_drive (runs cell 8's own source with each exit code)
  - test_hours (extended)
  - All fail on 4d2ce5f2. exp7 on the fix: (a) exit code 6, (b) "released: False", (c) "done on_drive True | DONE.json: True | 4096 on Drive: True".

MAJOR 2, what is final and what is resumable (ruling 2)
- Reproduced:
  - exp2 on 5bdd5aa: after the trainer's exit 4 and exit 5, every resume was refused with "the run is finished (HALT.json: ...)", whatever flag was given.
  - exp3 (broken pipe) and exp5(a) (a failed read during the copy back) on 73a5112 both wrote HALT.json. On 5bdd5aa they end DONE with no HALT (ef80550).
- Fixed in 4d2ce5f2:
  - The trainer's exit 4 raises `_Refusal` and ends the session "refused": no HALT.json, LIVE.json closed, "S2 launch: REFUSED: ...", exit code 4.
  - The trainer's exit 5 and any launcher error write a HALT.json with `resumable: true`, a `halt_id` and a `remedy` (["retry_check"] when the trainer's halt text names --retry-check).
  - "crashed twice" writes `resumable: false`.
  - New `read_lead` reads runs/<run_id>/LEAD.json. Allowed keys: retry_check, allow_source_change, allow_version_change, budget_by, budget_set_on, resume_halt, note, schema, written_at. A key it does not know, or a value of the wrong type, refuses the session.
  - `_halt_to_resume`: a resumable halt is resumed only when LEAD.json names its halt_id and holds its remedy. The old halt is then kept as HALT_<k>.json.
  - `session` records the LEAD.json document and the values in force. budget_by and budget_set_on go to QUEUE.json when the run is made.
  - `_cmd` takes retry_check from the session record.
  - ALLOW_VERSION_CHANGE and ALLOW_SOURCE_CHANGE left cell 1 (and the sheet's tables); they stay as S2_PARAMS-only parameters (EXTRA).
- Tests:
  - test_a_refusal_of_the_trainer_writes_no_halt
  - test_a_halt_is_resumed_only_with_the_leads_remedy
  - test_a_launcher_error_is_a_resumable_halt
  - test_lead_json
  - test_job_rules (updated)
  - A sheet-table check in test_the_steps_sheet_quotes_the_programs_own_lines
  - All fail on 5bdd5aa.
- The `_say`, `copy_back` and first-beat parts were already fixed in ef80550; tests for them exist.

Step 6 of the acceptance run (design 3.10), built in db1409f0
- `_remedies` makes a run of its own, <RUN_ID>-remedies, with one tiny job. Its sessions run cells 1-3 and 7-8 (`REMEDY_CELLS`).
- Its first session is killed after the job's first checkpoint is on the stand-in Drive, and the local run directory is wiped.
- It then fakes two records: QUEUE.json's commit, and the numpy version in the job's run.json last session.
- The sequence it checks:
  1. A resume is refused by the session for the other commit (rule K6).
  2. With LEAD.json `allow_source_change`, the session accepts and the trainer refuses (exit 4): no HALT.json.
  3. With `allow_version_change` added, the run reaches DONE, both remedies are recorded in the session, and the trainer records `version_change`.
- `acceptance` runs it in the second clone and records `remedies` and `step6_note`. The old `not_done` entry is gone.
- `local_run` runs it too by default; `--no-remedies` leaves it out.
- Tests: test_design_3_10_step_6_the_leads_remedies_after_a_faked_record (stub trainer, through the notebook's cells) and test_the_acceptance_runs_plumbing_on_stubs (updated). Both fail on c98c01a4.
- The acceptance run itself was not run.

MAJOR 3, a resume that would start from step 0 (ruling 3)
- Reproduced exp5(b) on 73a5112: "'copied back from Drive' printed: False", resumed_from [None, None], "stub session 1 from 0 steps".
- On 5bdd5aa: copied back, resumed_from [None, 3072]. That is ef80550, and the sheet's "from 0 steps" text is from 36347d4.
- Added in 6297190d, the marker the ruling asks for:
  - work/whole/<job>.json is written after a whole copy back, and when this machine's launcher starts the trainer.
  - The copy back is skipped only with the marker present, plus run.json present and a local whole checkpoint at least as new as Drive's newest.
- Test: test_the_copy_back_is_skipped_only_on_its_marker (fails on d97f9383).

MINOR CODE POINTS (ruling 4)
- Built in cc6cef5: launch refuses a closed session; release_runtime always unassigns; torch must be the cpu build on Colab; apt-get update; line 2 fails on another bundle; the smoke's Y5 trace; the acceptance scratch, wipe and record fields; the REPO_REF note; the mutation-survivor tests.
- Built now in feaac5d0:
  - S2_NB_SEAMS is refused on Colab and in any run that is not tiny, by session and by launch. The two non-tiny tests now patch the seams in-process (fixture `inproc_stubs`).
  - A Colab launch is refused when tracked files are modified. Untracked files get a note. `repo_state` now gives `modified` and `untracked`.
- Tests: test_the_stub_hook_is_refused_on_colab_and_in_a_run_that_is_not_tiny and test_a_colab_launch_on_a_checkout_with_modified_tracked_files_is_refused. Both fail on 6297190d.
- No code change for the review's risk note (text records re-uploaded whole may fill Drive's bin). It is for the lead: look at the bin after the smoke.

STEPS SHEET (ruling 5), c98c01a4
- QUEUE is a text field (`QUEUE = ""`). `parse_queue` drops brackets and straight or curly quotation marks.
- Every "(inferred)" became "(expected, not yet seen)".
- RULES_FILE field: "TODAY STAGE2_TRACKB_PLAN.md".
- The RUNTIME_KEPT field is gone: ruling 1 decided it, and the sheet now explains "THIS MACHINE IS KEPT ON PURPOSE".
- Added: "S2 launch: REFUSED"; a halt that waits for the lead; 5.2 and 5.5 notes on halts and "S2 NOTE:" lines.
- Kept plain and numbered, ASCII, at most 115 columns except the one URL.
- Tests: test_queue_is_a_text_field and test_the_steps_sheet_quotes_the_programs_own_lines. Both fail on feaac5d0.

RULING 6: nothing changed. Points a to k stay as built; retries are counted per job across sessions.

Also: ruff F841 fixed in 6297190d; docstrings updated in a003c31e.

FOR THE LEAD (each built as described; alternatives in brackets)
a. Cell 1 now has 13 parameters, not the design's 15: the two ALLOW boxes were removed because "the owner is never asked". [Keep the boxes and simply never mention them.]
b. A key the ruling does not list, `resume_halt`: every resumable halt must be named in LEAD.json by its halt_id, so a plain resume or an old LEAD.json cannot reopen a later halt. The retry_check remedy is found by matching "--retry-check" in the trainer's halt text, because train2 has no structured remedy field. [No acknowledgment needed for launcher errors; or make allow_source_change their remedy.]
c. Cell 8 keeps the runtime only on finish's exit code 6. That code also covers a finish that could not tell whether the save is on Drive. Any other code, or none, releases it (D19). [Also keep it on a crash or other unknown codes.]
d. finish itself writes DONE.json once the queue is finished and the last copies are on Drive. [Leave DONE to a resume; but a new runtime would not have the save.]
e. The dirty-checkout refusal applies to Colab launches only, not to smokes. [Refuse smokes too.]
f. Several "expected, not yet seen" marks can only be confirmed on Colab, so the sheet says "the acceptance run or, where only Colab can show it, the smoke".
g. In step 6, "a faked session record" is read as faking the run's QUEUE.json commit and the run.json last session's versions. A real exit-5 resumable halt is exercised only on stubs. local_run now runs one more tiny job for step 6.
h. On a resume, budget_by and budget_set_on from LEAD.json are recorded in the session only; QUEUE.json is not changed.
i. Hours: idle time after the last finish is unknown; such sessions are listed as `kept_runtime`.
j. Texts outside the package may need updating (not my files): rule K7's Check (SUPERSEDED_<k>.json, HALT_<k>.json, LEAD.json) in STAGE2_TRACKB_PLAN.md, and the run plan's V8 command. No other module must change.

3. TEST AND ACCEPTANCE OUTPUTS (through tools/lowprio.sh with the standing environment; scratch basetemp)
- Baseline after the merge: "50 passed, 1 skipped in 48.59s".
- Head a003c31e:
  - `pytest -p no:cacheprovider tests/test_stage2_nb.py`: exit 0, "61 passed, 1 skipped in 57.57s".
  - `STAGE2_SLOW=1 ... -k local_run`: exit 0, "1 passed, 61 deselected in 282.72s (0:04:42)".
  - ruff: "All checks passed!".
- `python -m designlab.stage2.nb local-run --drive <scratch>/lr/drive --record <scratch>/lr/LOCAL_RUN.json` on a003c31e:
  - Exit 0: "S2 LOCAL RUN local-20261117-023601: OK in 294 s", 23 of 23 checks.
  - Session 0 was killed after 60.7 s at "checkpoint 1024 of R5-75-M-H-10-s42 is on the stand-in Drive".
  - The resume without TAKEOVER was refused in 0.6 s: "...: still running (LIVE.json written 1 s ago); nothing to do".
  - The resume with TAKEOVER reached DONE in 141.6 s. A further resume was refused in 0.6 s.
  - Lines printed:
    - `S2 LINE 1/4 commit: a003c31e566601d1f0e248aeb8342874b946eb88 (REPO_REF is a commit: yes; checkout clean: yes)`
    - `S2 LINE 2/4 inputs: bundle S1-view, manifest 80a0aafa2177, 28 of 28 files match their sha256: OK`
    - `S2 LINE 3/4 G8a 75-M-H-10 set S1: zero action = R0 at 0 ms and 10 ms, 2 of 2 episodes identical (20596 physics steps): PASS`
    - `Run directory: .../runs/local-20261117-023601 (resuming session 1: R5-75-M-H-10-s42 from 1024 steps; 0 jobs done)`
    - `S2 LINE 4/4 first rate R5-75-M-H-10-s42: 467 env steps/s (8 envs, 2 workers, 4 vCPUs); 0.004096M steps in about 0.0 h`
    - `S2 LINE 4/4 first rate R6-75-M-H-10-s42: 578 env steps/s (8 envs, 2 workers, 4 vCPUs); 0.004096M steps in about 0.0 h`
  - Step 6 with the real trainer:
    - The remedies session was killed after 27.0 s at checkpoint 1024.
    - The resume was refused: "another commit is accepted only with allow_so[urce_change]".
    - With allow_source_change: "S2 launch: REFUSED: job R5-75-M-H-10-s42 was refused by the trainer (exit code 4): train2 run: Refused: ...: other library versions than the run's last session".
    - With both remedies: "S2 launch: DONE: the queue is finished"; the remedies line 4 was 439 env steps/s.
  - All rates are tiny jobs on a shared machine: provisional.
- git status of the worktree was the same before and after the run.
- Episodes: only the tiny R5 and R6 jobs on the S1 view of 75-M-H-10 (local-run, the slow test, and their remedies job). None on a look set.
- Outputs:
  - Everything is under `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rerun17/cn/`, which is outside the repository, so `git check-ignore` does not apply.
  - I removed two ignored `__pycache__` folders I had created in the worktree. Ruff added two ignored cache files under the existing `.ruff_cache/`.

4. DIFF AND STATUS
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 hardware/trex_form_study/designlab/stage2/nb.py    | 4018 ++++
 .../trex_form_study/stage2/stage2_training.ipynb   |  248 ++
 hardware/trex_form_study/stage2/tb/colab_steps.txt |  347 ++
 .../stage2/tb/requirements-colab.txt               |   19 +
 hardware/trex_form_study/tests/test_stage2_nb.py   | 2617 +++
 5 files changed, 7249 insertions(+)
```
`git status --porcelain`: empty.

5. COMMITS OF THIS ROUND (oldest first)
- 5bdd5aa8 merge of the working branch
- 4d2ce5f2 resumable halts and LEAD.json (ruling 2)
- d97f9383 last copy: growing wait, finish that copies, runtime kept (ruling 1)
- 6297190d copy-back marker (ruling 3)
- feaac5d0 test hook only in tiny runs; dirty Colab launch refused (ruling 4)
- c98c01a4 steps sheet; QUEUE as a text field (ruling 5)
- db1409f0 design 3.10 step 6
- a003c31e docstrings

Branch head: a003c31e566601d1f0e248aeb8342874b946eb88.

All 20 commits on the branch that are not in the working branch end with attribution lines; none lack them. The 12 earlier ones carry the earlier model's Co-Authored-By line, as briefed. All of mine, the merge included, end with the two 17 Nov lines and name no model anywhere else.
