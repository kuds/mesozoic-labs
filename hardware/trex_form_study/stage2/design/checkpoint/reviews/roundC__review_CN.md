# review:CN (roundC, wave 4): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule. The package is not merged: its fix round was paused for the usage limit.

REVIEW OF WORK PACKAGE CN (the notebook), branch wt/wpc-cn at 73a5112, against the working branch at fdb31d5

VERDICT: merge after the listed fixes (majors 1 to 3). Nothing is a blocker for the other packages: the branch adds five new files and touches nothing else. The three majors are all on failure paths of the launcher that a real Colab session can reach, and two of them lose a trained job. They should be fixed, with the lead's ruling on major 2, before the acceptance run. The sheet's points can wait for its final text.

My scripts and logs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_cn_wave4/ (exp1 to exp6, mutants.py, the test logs). All experiments ran on the package's own stub trainer in scratch; no tracked file was changed, nothing was committed.

FINDINGS, most serious first

1. MAJOR. A job's last copy to Drive gets one try; the run is then written DONE and the runtime released without its final checkpoint. With Drive gone for good the whole job is lost.
/home/user/mesozoic-wt/wpc-cn/hardware/trex_form_study/designlab/stage2/nb.py: `_drive` 1714-1724, `run_job` 2095-2102, `_end` 2151-2177, `finish` 2300-2355 (unguarded write at 2349); notebook cell 8.
- exp1_last_copy.py: one OSError on the copy of the final checkpoint in the end-of-job cycle. Output: "S2 WARNING: a copy to Drive failed (...); it is tried again in the next cycle", then "S2 launch: DONE: the queue is finished". Drive checkpoints [1024, 2048, 3072], local [2048, 3072, 4096]; DONE.json says last_checkpoint 4096, last_checkpoint_copied 3072; no ckpt_4096.tar. There is no next cycle at a job's end, and an earlier job's checkpoint is never retried while the next job runs.
- exp4_drive_gone.py: every Drive access fails with errno 107 after the first checkpoint. The launcher trains the whole budget (116 warnings), ends "DONE: the queue is finished (DONE.json could NOT be written ...)", state done, exit code 0. `finish` then raises an uncaught OSError. Cell 8's `finally` still calls `release_runtime`. Drive held checkpoint [1024] of 30720 steps.
Fix: at a job's end repeat the copy (fenced, bounded) until the final checkpoint and each check's tar read back; if they do not, write no DONE.json and say so. Guard `finish`'s writes and have it return whether every job's last checkpoint is on Drive; cell 8 releases the runtime only then. Whether an unreleased runtime is acceptable against the hours cap is the lead's call.

2. MAJOR. HALT.json is final, and the launcher writes it for cases the rules treat as resumable, and for its own faults.
nb.py 2104-2107 (trainer exit 4 and 5), 2226-2228 (any exception), 1153-1156 (a run with HALT.json is never resumed), 238-239 (`_say` can raise), 2080 and 1867-1890 (`copy_back` outside `_drive`), 2202 (first `beat` unguarded).
- exp2_halt_final.py: after the stub trainer's exit 4 ("refused") or exit 5 ("halt") every resume is refused, with ALLOW_VERSION_CHANGE, ALLOW_SOURCE_CHANGE or RETRY_CHECK alike: "the run is finished (HALT.json: ...)". So rule K4's flags after a trainer refusal, rule M1's retry parameter (fifth list, ruling 2) and "a resume runs the check again" (fifth list ruling 6, sixth list ruling 3) cannot be applied through the notebook. Design 3.10 step 6 cannot be performed as built; `acceptance` leaves it out (3087).
- exp5_copy_back.py (a): one failed read from Drive during the copy back gives HALT "launcher error: OSError: [Errno 5]"; a further resume is refused.
- exp3_broken_pipe.py: the reader of the launcher's output goes away (a kernel that died alone): HALT "launcher error: BrokenPipeError" at 1536 of 204800 steps; resume refused.
Fix (the lead rules the first part): trainer exit 4 writes no HALT.json (the session ends, LIVE.json closed, as an interrupt does). Exit 5 and launcher errors write a halt marked resumable, which `session` accepts only with the named remedy and then archives as HALT_<k>.json. Only "crashed twice" stays final. `_say` must never raise; `copy_back` and the first beat must handle OSError. Then build step 6 of the acceptance run.

3. MAJOR (silent; narrow trigger). A resume can restart a job from step 0 although Drive holds its checkpoints, and the sheet tells the owner that this needs nothing.
nb.py 1880-1881 (`copy_back` is skipped whenever a local run.json exists); colab_steps.txt 216-219.
- exp5_copy_back.py (b): the local job directory holds only run.json (what a copy back cut after its first file leaves). The session line reads "resuming session 1: R5-B1-s42 from 3072 steps". No copy back happens, the trainer starts "from 0 steps", resumed_from is [None, None], and the run ends DONE.
Fix: make the copy back all or nothing (verify each checkpoint against its record, write a marker, skip only on the marker). Refuse to start a `--resume` job when Drive holds a readable checkpoint and the local directory has none. In the sheet, "restarting <job> from step 0" is fine only when the Run directory line said "from 0 steps"; otherwise stop and tell me.

4. MINOR, code.
- `launch` (2244-2279) does not require lines 2 and 3 of its own session; only the cell order does. exp6: after a failed line 2 the same session's launch ran to done.
- `release_runtime` (2358-2380): if `drive.flush_and_unmount()` raises, `runtime.unassign()` is never reached. exp6: calls ['flush', 'flush'], not unassigned. Design D19 wants both.
- `setup` (814-834) does not assert that torch's build label is `cpu` after the installs (rule N2). `apt-get install` runs without `apt-get update`.
- `_step_inputs` (3122-3131): line 2 is printed and stored as OK when the step then refuses "RESUME with another bundle".
- Smoke Y5 (2671-2676): the trace is looked for only before the kill; it may be copied while being written.
- `acceptance`: `shutil.rmtree(scratch)` (3007) on any path outside the checkout; the wipe (3046) keeps `<scratch>/local/inputs`, which a dead VM would lose; the record has no clone HEAD value and no per-body G8a counts as fields.
- S2_NB_SEAMS is honoured in any session. Suggest refusing it on Colab or when the run is not tiny.
- The REPO_REF note in `session` (1167-1169) cannot trigger for a launch.
- Risk, not measured: every cycle re-uploads each changed text record whole (updates.jsonl, run.json, logs). If Drive keeps each replaced file in its bin, this may reach gigabytes per job. Look at the bin after the smoke.

5. MINOR, tests. Mutation check on scratch copies (mutants.log): 26 of 30 mutants killed. Survivors: nb's own comparison of the bundle id and of the content hash from `verify --json` (the bundle tool refuses the same cases first, so the bundle ruling's item 3 has no test of its own); collect's `logs/` skip; one equivalent mutant. No test covers a Drive error at a job's last cycle, a failed copy back, or a resume after a trainer exit 4 or 5.

6. MINOR, the steps sheet (/home/user/mesozoic-wt/wpc-cn/hardware/trex_form_study/stage2/tb/colab_steps.txt), read as a careful person who is not a programmer.
- Line 181 "You can close the tab; the run goes on" and line 197 are Colab behaviour not seen here: mark them inferred, and say where background execution is switched on if it must be.
- 5.4 (line 205) deletes the old runtime; 5.6 (226) then says "leave the old one alone". Say: wait ten minutes and do 5.4 again.
- QUEUE typed on a phone gets curly quotes: `SyntaxError invalid character '“'` (exp6). Say "copy and paste", or make QUEUE a text field: `parse_queue` already accepts `R5:75-M-H-10:42, R6:75-M-H-10:42`.
- Line 28 gives the rules file as `stage2/tb/PREREG_TRACKB.md` as if final; the settled text is `STAGE2_TRACKB_PLAN.md`. Make it a field.
- The Drive window appears only after the setup, minutes after Run all; say to stay until line (d).
- Section 6 says "anything in red": git's own lines in box 2 may show on red (inferred). The RuntimeError text "S2 step ... stopped with exit code" and "S2 launch: interrupted" are not listed.
- 5.5: the trainer's own <k> and <n> can differ from the session line's.
- Otherwise the sheet is plain, numbered, in the design's order, says what to expect and what to do, and every run-dependent value is a marked field listed again at the end.

FOR THE LEAD: the builder's points, with my choice
a. A superseded launcher writes no HALT.json on Drive. Options: as built, or the design's literal HALT. Keep as built: a fenced launcher must write nothing into a run it no longer owns, and HALT.json would end the run the takeover continues. Rule K7's Check should name SUPERSEDED_<k>.json.
b. The resume right after a kill needs TAKEOVER. Options: as built, or wait 600 s. Keep as built; the fast test shows the path without TAKEOVER on a LIVE.json older than ten minutes.
c. QUEUE.json is written by `session`. Keep: it matches rule K6 and rule Y10 ("written when the queue is made").
d. REPO_REF on a resume. Keep: cell 2 checks out before Drive is mounted, so the cell's ref must name the run's commit; another commit needs the flag.
e. A dirty checkout only prints "no". I would refuse a Colab launch when tracked files are modified and only report untracked ones.
f. Checkpoint 0 is not copied, the newest check gets the clip, the queue waits up to 300 s for it at a job's end. Keep.
g. Exit 4 and 5 are not retried. Keep, but see major 2.
h. HALT.json final, RETRY_CHECK only through a parameter file. Not acceptable as it stands (major 2). I would have the lead write a small file on Drive that `session` reads and records, rather than add a parameter the owner could set. The same file can carry BUDGET_BY and BUDGET_SET_ON, which rule Y10's Check needs when Y4 fails and which the notebook cannot set on Colab today.
i. Line 3's steps are the sum over episodes; line 4 reads `first_three`; Y4 reads three updates. Keep. Line 4's rate is slightly higher than Y4's by construction (two updates against three); the sheet's <<RATE>> must come from line 4.
j. REBUILD_OK.json. Keep; note that one body's Y8 then permits the fallback for the whole bundle (every hash is still verified).
k. A smoke is not resumed; collect never replaces. Keep.
Not handed up: retries are counted per job across sessions (JOB_RULE, nb.py 184-185, 2133-2135); per session is the other reading. The run plan's V8 command (`nb launch --run-id ... --queue ...`) does not exist: it is a parameter file with `nb session` first. A failed cell 2 to 6 leaves the runtime allocated. Fine-tunes (rule K8) have no queue entry: defer until it is decided where they run. High-RAM in the metadata: add only after the smoke shows what the account offers.

CHECKED AND FOUND RIGHT
- `git diff claude/mesozoic-trex-stage2-refine-4jffre --name-status`: the five files, all added. `git status --porcelain` is empty in the worktree, before and after my runs; the main checkout has no tracked change.
- The hash line prints the two registered values.
- All eight commits not in the working branch, the two merges included, end with the two required lines and name no model elsewhere.
- `git merge-tree --write-tree` against fdb31d5: exit 0, the five files only; the working branch's tip is an ancestor (fast-forward).
- nb.py imports the standard library and tbenv only; the notebook has nine cells, no accelerator, the scaffold's Git block with `designlab` in the refusal; the nine pins equal the venv.
- The lines, the STATUS, SESSION, DONE and HALT schemas, the K2 copy cadence, rules K6 and K7, the fence, the job rules, collect, hours, and rule Y11's budget match the design and the rules.
- The command lines and record keys of train2, bundle, s2load, clip2, g8a, r5env, setio and monitor2 that nb.py uses exist as it uses them; seed 6004 is walk/v0.4's first monitor seed.
- ruff passes on both Python files.

TESTS (HEAD 73a5112, through tools/lowprio.sh)
- `pytest -p no:cacheprovider tests/test_stage2_nb.py`: "33 passed, 1 skipped in 28.13s", exit 0.
- `STAGE2_SLOW=1 ... -k local_run`: "1 passed, 33 deselected in 201.68s (0:03:21)", exit 0. Its record: ok, 17 of 17 checks; killed after 60.0 s at "checkpoint 1024 of R5-75-M-H-10-s42 is on the stand-in Drive"; refused in 0.5 s; resumed to DONE in 139.1 s; line 4 at 478 and 622 env steps/s (tiny jobs on a loaded machine, provisional).
Episodes: only that one drill's tiny R5 and R6 jobs on the S1 view of 75-M-H-10 (G8a stock form, R0L baseline, two checks each). None on a real set, no record, no network call, nothing uploaded.

NOT CHECKED
`acceptance` (never run); anything only Colab does (pip, apt, the clone and fetch of cell 2, the Drive mount, the disconnect, how Drive treats replaced files, red output); the smoke with the real trainer; Y8's `sets build` on a look set; preflight's bar form and envs on a real set; a real clip; the data-branch fallback beyond the package's own scratch test.
