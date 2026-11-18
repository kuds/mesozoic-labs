# review:CW (roundC, wave 4): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule. The package is not merged: its fix round was paused for the usage limit.

REVIEW OF WORK PACKAGE CW (lookbar.py, s2rows.py and their two test files), branch wt/wpc-cw at 5d04b20

VERDICT: merge after the listed fixes. One major fix (finding 1, two lines and one assertion) must be in before code state B, because it changes every Track B row's description and so its record key. Findings 2 to 7 are minor. No blocker. Nothing I ran contradicts the builder's report; three things it does not mention are findings 1, 2 and 4.

FINDINGS, MOST SERIOUS FIRST

1. MAJOR. One of the look reading's four rule texts is dropped from every row description and every companion record.
   Where: designlab/stage2/lookbar.py:329 (`rule=dict(lookread.RULES, **RULES)`) and :835 (`rules=dict(lookread.RULES, **RULES)`); lookbar.RULES has a key "reference" (:143) and so has lookread.RULES.
   Evidence (my script scen_records.py): "B: shared keys ['reference']", "describe's 'reference' is lookbar's: True | lookread's text anywhere in the description: False", "rule texts in the description: 12 of 13". The lost text is lookread's rule U5 wording with the module's own reading after its mark (the 0.1 s lead, too_few_steps, a playback that lacks a value), which the rulings of 18:21 UTC say stays as marked. The tests do not see it: tests/test_stage2_lookbar.py:161 and :518 compare the union of the key sets, which is the same with or without the collision.
   Fix: rename lookbar's key (for example "reference_use"), or nest the two tables (`rule=dict(lookread=..., lookbar=...)`); add `assert not set(lookread.RULES) & set(lookbar.RULES)` and an assertion that the description holds lookread's text. Do it before any Track B record: the description is in the key.

2. MINOR. The test "no round-A or Stage 1 file differs from state A" is not there, and the report does not say so.
   Where: design section 4, row "The look reading and its wrapper" and decision D18 name it for tests/test_stage2_lookbar.py; the brief asks for "every item of that row that names it". `grep -n "state A\|git.*diff\|45719d03\|c0027992" tests/test_stage2_lookbar.py tests/test_stage2_s2rows.py` prints nothing, and no other test file holds it.
   Fix: a fast test that `bar.bar_sources()["sha256"]` and `panels.panel_sources()["sha256"]` are the registered pair (it also runs in an export without git).

3. MINOR. Rule K4's check at the episode (designlab/stage2/s2rows.py:365-372, `S2Policy._check_files`) has no test.
   Evidence: with the check switched off in memory, the fast file still passes ("no_files_check exit=0 :: none failed"); the slow tests change no file either. The check itself works (scen_k4_episode.py: a job built against root A, the same set name under root B with another 0.40 m/s walk file): in force, "episode on root B: error ... RowError: R5/42: refs/v0.4.npz in .../B/T0/75-M-H-10 are not the files the run recorded (rule K"; switched off, "episode on root B: fell | steps 7126 | error: None". It is the only guard when `--set-root` (or `episodes(root=)`) is not the root the row was built against.
   Fix: that scenario as a test.

4. MINOR. After a refusal, `collect` leaves a stale companion on disk and rewrites the summary without the record.
   Where: lookbar.py:880-884 and :904-913.
   Evidence (scen_records.py): after the sidecars are removed, "second collect raised LookBarError", "the companion file is there and unchanged: True | summary records 0 refused 1". After a bar record changed (one episode now a fall) with a sidecar that is not its episode's: "the old companion is still on disk: True | it names the bar record's old sha256: True | its counts_look succeeded 20 (the bar record now holds a fall)". The exit code is 1, so it fails closed for a caller that reads the code; a reader that takes companions by path gets an old one.
   Fix: on a refusal move the record's existing companion aside (or delete it) when its `bar_record.sha256` is no longer the file's, and keep a cell for the refused record in the summary.

5. MINOR. A fault of the reading is recorded as a traceback with this machine's absolute paths (lookbar.py:430, `traceback.format_exc(limit=4)`), into the sidecar and the companion. The other faults are `Type: message`. Fix: the same form there.

6. MINOR. `set_check` fails open on a run.json that records no plant inputs or no task names (s2rows.py:203-206: `was is not None and ...`, `settings.get("task_names") or []`). The trainer always writes both (I checked train2.py:953-955), so this needs a malformed run.json. Fix: refuse when either is absent.

7. MINOR, and a rule reading for the lead. An episode during which MuJoCo warned is "a fall at its start" (tune.warned_fall: fell, fall_time_s 0), but its `survival_s` is the time simulated (lookbar.py:405, as the bar's own record), and the companion episode carries no `fall_time_s`. scen_records.py: "C: outcome fell survival_s 20.0 fall_cause MuJoCo warning BADQACC (x1) | monitor's credit 20.0". Options: the monitor credits 0 for it (my choice; the companion episode then needs a `warned` flag or `fall_time_s`), or it is left as the bar counts it.

INTERFACES OWED BY OTHER PACKAGES (not this package's files)

- The trainer does not pass its `--allow-source-change` to the row. train2.py:1131 starts the monitor with `tbenv.child_env("disable")`; s2rows.py:253 reads STAGE2_ALLOW_SOURCE_CHANGE from the environment. On a flagged run every later check would end in `RowError` ("not judged, rule K9"), which monitor2.main does not catch (it catches MonitorRefusal and MonitorError only): a traceback, a closed attempt without a record, and after two the check counts as without a gain. Not run by me; read from the code. Owed with the trainer's next change, together with the refusals of ruling 3 of the sixth list (`s2load.CheckpointError`, `s2rows.RowError` from `lookbar.rows`).
- A row is a policy row when its inner row has `provenance()` (lookbar.py:350-353, :551). The R2L package must not give its rows a method of that name, or their unreadable episodes become "unread" instead of "error".
- `look_partial/` is not ignored under stage2/tb/bar/ (`git check-ignore` exit 1 there, exit 0 under stage2/tb/work/).

THE BUILDER'S POINTS FOR THE LEAD

a. `lookbar run` cannot run R0L, R5, R6 or R2L rows on a look set (bar.py:2202-2205; pinned by the builder's test). Confirmed by reading; the monitor's path (`lookbar.episodes`) and S1 are not affected, and my tiny trainer runs went through it. Options: (1) edit `bar._check_rule`: moves the bar's source hash, against D18; no. (2) set `bar.SYNTHETIC_EPISODES`: misuse of a test switch; no. (3) a runner of lookbar's own: some 400 lines copied from `bar.evaluate`; no. (4) `lookbar.run` puts a rule of its own in place of the bar's check for the one call (in memory, restored in a `finally`), checks rule X1 itself (G9-ref passed for the set and body) and writes its rule into the companion and the summary. I would choose (4), with a ruling and one sentence in the Track B rules, because the bar record will still carry the bar's own `rules.checks` text. Until then only `--rows R0@10ms` works on a look set.
b. Comparing the Stage 1 sources as a whole and the panels' sources: keep. A run records nothing finer, and it is stricter than the design's list.
c. Add `look_partial/` to stage2/tb/.gitignore; the companion holds every episode's block, so the sidecars need not be committed.
d. `on_command_at_stop` against 0.9 x 0.42 = 0.378 m/s: keep. STAGE2_75_PLAN C7 itself calls "the command" 0.42 m/s. On V0 I read 0.4098 under R0L.
e. On a set that is no look set the counts are the bar's: this is the ruling. Note that an unread episode there has outcome "unread" and is counted as an error.
f. Keep. On a look set a fall on an item with a faulty reference stays a fall, so such a record can be judged; gate Z2 (b) must have caught the definition.
g, h, i, j, k. Keep as built. (j) is rule E4's own text.
l. LOOK_SETS by exact name: "V6.2" is then no look set and reading 1 silently stops counting ("D: V6.2 look set: False"). Options: exact; or the bar's convention, the name before the first dot. I would take the bar's convention, decided before state B.
Also flagged by the builder: the bar runs a policy row's unread episode again on every run of the same key and `lookbar run` exits 4. It cannot be changed without bar.py; accept it and say so in the steps for bar runs.

WHAT I CHECKED AND FOUND RIGHT

- Tests, run by me at 5d04b20, environment as the standing rules give it, through tools/lowprio.sh:
  - fast, both files: 44 passed and 11 skipped by the progress line, exit 0 (my `-q` suppressed the summary line);
  - `STAGE2_SLOW=1 ... tests/test_stage2_lookbar.py`: "35 passed in 99.18s (0:01:39)";
  - `STAGE2_SLOW=1 ... tests/test_stage2_s2rows.py` (whole file: equivalence at k = 5 on 75-M-H-10 and on -X with H1, R6, two spawned workers): "20 passed in 266.53s (0:04:26)";
  - an export of the tip without the ignored inputs: "20 passed, 35 skipped in 6.52s", file list unchanged;
  - ruff on the four files: "All checks passed!".
- Other packages against the real rows: tests/test_stage2_monitor2.py "22 passed, 1 skipped in 21.40s" (the skip: "lookbar is merged: the stand-in is no longer reached"); `STAGE2_SLOW=1 ... tests/test_stage2_train2.py -k "tiny_r5_run_resumes or tiny_r6_run_resumes"`: "2 passed, 49 deselected in 211.92s (0:03:31)". Their monitor records say `episodes_by lookbar.episodes`, rows R5/42@10ms and R0L@10ms, every episode with its clock times and `planned_s`, no error outcome.
- Would the tests fail if the behaviour were broken: 27 in-memory mutations of lookbar.py (recorder not restored, lent to all or none, no sidecar, unread for any row or after a fall, the clock rule, unread counted as a fall, reading 1 on other sets, the stop term, the sidecar checks, a1_over, the heading in the stop term, and others): all 27 fail a fast test. 13 of s2rows.py: 9 fail a fast test, the H1 seed fails the slow -X test, the two set-name checks are each covered by the other (both off: fails), and the files check survives (finding 3).
- Check on the frozen set V0 of 75-M-H-10 (read from the main checkout, freeze index 0, manifest 34a820e5...; no record; traces under my scratch directory): "walk/v0.4 @10ms seed 3042: steps 10539, digests equal True, records equal True, trace sha256 equal True"; equivalence at k = 5, freshly initialised policy: "'identical': True ... 'reference_steps': 8683, 'compared_steps': 8683, 'reference_fell_step': 8683, 'env_terminated_step': 869". A turn item is read end to end under R0L (window [15.701, 18.701], window_ref [16.486, 19.636]); the stop item's term reads 0.4098 against 0.378.
  My check episodes on V0, 8 in all: R0@10ms and R0L@10ms on walk/v0.4 seed 3042, each once for digests and once for record and trace (4); R5/42@10ms and the env on walk/v0.4 (2); R0L@10ms on turn/q90_left and on stop/two_step (2).
- Kill and resume: `lookbar run` on S1 (walk/brisk, 3 episodes, one worker) killed with SIGKILL in its third episode, run again: exit 0; the bar record names the two kept episodes and the new one; "companion: outcomes {'read': 3, ...} counts equal bar True sidecars ['3042', '3043', '3044']".
- Rules: `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`: the four files, "4 files changed, 3455 insertions(+)". The two hashes are the registered pair (printed at the start and at the end). All 7 commits not in the working branch, the merge bb28c45 included, end with the two required lines and name no model elsewhere. `git status --porcelain` is empty in the worktree and in the main checkout. No network call.
- Merge readiness: the working branch's tip fdb31d5 is the branch's merge base; `git merge-tree --write-tree` exit 0, and the result differs from the working branch by the four files only.
- Read against the brief, design 2.2, 3.5 and 3.11, rules U4-U6, M1, K4, K9, E4, E5 and the rulings: LookRow, R0L, rows, episodes, collect, run, the CLI, the three build checks, describe, sources, spec, the turn in place by E4, what monitor2.py expects. Apart from the findings I found no difference.

WHAT I DID NOT CHECK

- tests/test_stage2_train2.py's fast tests and its probe test; any other package's tests.
- A 60 cm body, a -J body; R6 and R5G anywhere but S1; any episode on a look set; `lookbar run` through `bar.evaluate` on V0 (it would write records) or with `--task` and a budget; a Colab runtime or a mounted Drive.
- Item 6 of the checklist does not apply: the package has no text a person follows.
- In the worktree I created and removed a `.ruff_cache/` and three `__pycache__` directories (ignored paths; my own slip of one command without PYTHONDONTWRITEBYTECODE).

Files are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_cw_wave4:
- mutplug.py
- run_muts.sh
- muts_lookbar.log
- muts_s2rows.log
- scen_records.py
- scen_k4_episode.py
- extra1.log
- v0_review_check.py
- v0check.log
- v0_turn_stop.py
- kill_resume.sh
- kill_resume.log
- slow_lookbar.log
- slow_s2rows.log
- other_monitor2.log
- other_train2_slow.log
