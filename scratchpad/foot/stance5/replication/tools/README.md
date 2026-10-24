# Stance5 replication: fetch, judge and watch the two new band runs (s70, s80)

Everything here is tooling. It changes no criterion and no verdict. The criteria are `R/criteria.txt`; the judge
(`R/judge/judge_stance5r.py`) and the scorer (`R/judge/score_stance5r.py`) decide. These scripts only fetch, place,
queue and report. Each one is the stance5 tool of the same name (`S5/tools`, read-only, never edited) with the
replication identity. `CHANGES.diff` lists every changed line, and `R/build/make_tools.py` writes the tools from the
stance5 originals (`--check` confirms that the files on disk are exactly that).

`S` = `/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad`, `S5` = `$S/foot/stance5`,
`R` = `$S5/replication`, `T` = `$R/tools`, `C` = `$S/fix/scratch_copband`. `C` is the screen clone, on branch
`claude/screen-trex-r9-armlts-copband` with HEAD cf80ecb14d1630ce3b4dcdea28a05835d143470b. It is read-only and must stay
clean at that HEAD.

## The runs (Drive, read-only)

| Run | Seed | RUN_ID (the Drive folder name) | RUN_LABEL | Drive folder ids |
|---|---|---|---|---|
| Run 4 | 70 | stance5_LTSB_s70 | stance screen LTSB s70 | folder 12uAVYvh..., 01_stance 1Z4yOUFc..., models 1KeutiFc... (filled 2026-10-22) |
| Run 5 | 80 | stance5_LTSB_s80 | stance screen LTSB s80 | folder 1IMKbSxM..., 01_stance 1bxcvmsv..., models 1MyRe7Q2... (filled 2026-10-22) |

Run `stance5r-find-folders.js` once the runs have started (args `{}`). Fill `DRIVE_IDS` in both fetch workflows with
what it returns, or pass `args.drive` per call. Until then both fetch workflows refuse to run (measured, dry run).

The find workflow also looks for a folder titled with the RUN_LABEL. On stance5 the label was once typed into RUN_ID;
such a folder is never used. It reports a `<RUN_ID>_b` (or `_c`) folder with its ids in notes: that is the run's restart
(Run handling, below), and it is never put into `DRIVE_IDS` in place of the RUN_ID's own ids.

Within about 10 minutes of the maintainer's start message, read each run's provenance.json on Drive (run_id
stance5_LTSB_s70 / s80, training_seed 70 / 80, repository commit cf80ecb1, not dirty) and look for a label-titled or
duplicate folder. Report any mismatch to the maintainer at once; stopping or restarting is their decision. (The first
check by `place5r.py config` comes only with the first checkpoint fetch, about 36 minutes in; stance5 caught its
label-in-RUN_ID error 2 minutes in by reading provenance.json, run_log 2026-10-21 13:28 UTC.)

After the first check-in, place the run records once with `place5r.py config RUN_ID DIR DIR2`, from a fetch with
`configs: true`. `place5r.py` refuses them unless `stage_config.json` reads all of these:
- git_commit cf80ecb1;
- the run's seed (70 or 80);
- n_envs 4 and 12,000,000 steps;
- stagger 0.30 / 0.045;
- all eight band keys;
- collapse reference 4457.4;
- task_sha256 8c6998d2....

`task_fingerprint.json` must carry the same task. These are place5.py's checks. One note is added, report only: a
`run.label` that is not the RUN_LABEL is noted, never refused. If it refuses the run's own records (a check above fails,
or a record is absent on Drive) and A and B agree, that is the run's side: place them with `--run-records-fail` (Run
handling rule 2).

## Files and what changed against stance5

| File | From (S5/tools) | Changes (all in CHANGES.diff) |
|---|---|---|
| `stance5r-find-folders.js` | `stance5-find-folders.js` | The two runs and their seeds, and the replication tools path. The prompt adds a look for a folder titled with the RUN_LABEL (not used, reported), and reports a `_b` / `_c` folder as the run's restart, with its ids, in notes. |
| `stance5r-fetch-checkpoint.js` | `stance5-fetch-checkpoint.js` | The two runs, with null `DRIVE_IDS`. Root `R` (`R/fetch/r<steps>/<RUN>/...`), `place5r.py`, and the "stance5 replication" wording in the prompt. Run handling: `args.source` (below); the tie-break transcriber names the same Drive folder as A and B. |
| `stance5r-fetch-named.js` | `stance5-fetch-named.js` | The same changes (`R/fetch/<tag>/<RUN>/...`), `args.source` included. |
| `place5r.py` | `place5.py` | `RUNS` s70/s80, root `R` (test root env `PLACE5R_TEST_ROOT`), and the report-only `run.label` note. Every check is unchanged. Run handling: `--source-folder`, `--supersede`, a ledger replaced after a resume is moved aside, `placed_after_N` after a resume, and `config ... --run-records-fail` for run records that fail the checks (below). |
| `judge_one5r.sh` | `judge_one5.sh` | `judge_stance5r.py`, root `R` (`R/runs`, `R/judged`), RUN_ID:SEED s70:70 or s80:80, and the completeness check's schema `stance5r-judge/v1`. Run handling: the refusal record `refused_<steps>.json` and exit 5, and a refusal held for review, `refused_<steps>.review.json` and exit 6 (below); test root env `JUDGE5R_TEST_ROOT`. |
| `panels5r_one.sh` | `panels5_one.sh` | The two runs, `RUN_DIR=R/runs/RUN_ID` and `PANELS_OUT=R/panels/out`. It calls stance5's `S5/panels/panels_one5.sh` and `panels5.py`, unchanged; they write only under `PANELS_OUT`. |
| `queue5r.sh` | `queue5.sh` | The two runs, root `R`, and `judge_one5r.sh` / `panels5r_one.sh`. The header adds how to run it here (below). Run handling: the status "refused" (settled; not rerun; its panels and final extras skipped) and "review" (held: not rerun, not settled). |
| `satwatch5r.py` | `satwatch5.py` | The two runs, root `R`, `satroll5r.py`, `place5r.py` and its own state file `T/state/satwatch5r_state.json`. The alarm rule is still read from stance5's `score_stance5.py`, unchanged. |
| `satroll5r.py` | `satroll5.py` | Imports `judge_stance5r.py`, and `--seed` also takes 70 and 80. |
| `cert5r_one.sh` | `cert5_one.sh` | Root `R` (`R/cert/<RUN>`), and it refuses a RUN_ID that is not s70 or s80. It creates `R/cert/<RUN>` only after it finds the final pair placed, so a refused call leaves no empty directory. |
| `wfcheck/harness.mjs` | `wfcheck/harness.mjs` | The scenario runs: `notyet` on s80, `differ` on s70. |

## What to fetch, and when

The slots are stance5's (the replication criteria, by reference):

- J (the verdict): 4,194,304; 5,242,880; 6,291,456; 8,388,608; 10,485,760; final (12,001,280).
- The report-only 1M grid, judged like J: 1,048,576; 2,097,152; 3,145,728; 7,340,032; 9,437,184; 11,534,336.
  stance5 did not fetch 1.0M, 2.1M, 3.1M, 9.4M or 11.5M. The queue lists them as "missing" until they are fetched.
- The report-only quick panel (20 episodes): the other 524,288-step checkpoints. Lowest priority.

Arrival times (measured on the stance5 runs, same code and runtime type): 230.9 to 245.2 steps/s on the L4. Computed
from those rates (not timed): a checkpoint every 36-38 minutes and 6.3M about 7.1-7.6 h after the start. Measured: the
final about 13.7-14.5 h after the start (stance5: started 13:32-13:35 UTC, ended 03:14-04:03 UTC). The end-of-stage evaluation adds about 30 minutes. Read each
run's progress.json (`timesteps`, `steps_per_s`, `checkpoints.periodic_timesteps`) before each fetch.

## Each check-in

1. Ledgers and progress (report only): `stance5r-fetch-named.js` with
   `{"tag": "ledgers_<hhmm>", "csv": true, "extras": ["progress.json"]}`.
2. Checkpoints: `stance5r-fetch-checkpoint.js` with `{"steps": N}` once both runs list it. The first one also takes
   `"configs": true`, so the run records are placed. The queues pick the checkpoints up.
3. The saturation watch, at every check-in (D-D40 5a), after steps 1 and 2:

       cd $T && python3 -I satwatch5r.py --roll

   Exit 11 means something new to report. Tell the maintainer at once with the run, the step and the actuator, or
   "actuator not yet named". If the actuator is not named yet, fetch the named checkpoint for that run alone
   (`{"steps": N, "runs": ["RUN_ID"]}`) and re-run the watch. Exit 3 means a check-in problem, never "no alarm".
   Nothing is stopped, restarted or changed on an alarm. Fails stay fails. (Stance5's step 3 text applies word for
   word.)

   The stance5 lesson: `stance5-fetch-checkpoint.js` does not place progress.json. Without step 1 the watch reports a
   LEDGER PROBLEM (no progress.json), so run step 1 at every check-in.
4. After a run ends (`01_stance/gate_verdict.json` exists): `stance5r-fetch-named.js` with
   `{"tag": "final", "csv": true, "extras": ["progress.json", "gate_verdict.json"], "items": [["stage1_final.zip",
   "stage1_final_vecnorm.pkl"], ["best_model.zip", "best_model_vecnorm.pkl"], ["robust_best_model.zip",
   "robust_best_model_vecnorm.pkl"]]}`. gate_verdict.json can land on Drive a few minutes after the final pair (stance5
   s42 and s60): fetch it again if it is missing.
5. At final (report only): `$T/cert5r_one.sh RUN_ID`, the six-heading certificate, about 16 minutes.

The stance5 fetches had their place step cut off by a safety classifier three times. When that happens, place by
hand with the workflow's own command: `cd $T && python3 -I place5r.py <steps> RUN_ID $R/fetch/r<steps>/RUN_ID` (with
`--source-folder <RUN_ID>_b` for a restarted run, below).

## Running the queues in this container (stance5's lesson)

A detached worker (`--watch`, setsid nohup) is not tracked by the harness, and the idle container is reclaimed about
10 minutes after a turn ends. On stance5 that killed both workers twice (run_log 2026-10-21 21:33 and 23:48 UTC). Run
each worker as a harness-tracked background command, by its absolute path, under a time limit, and relaunch it when it
exits:

    timeout 7000 /tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/replication/tools/queue5r.sh judge --workerwatch
    timeout 7000 /tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/replication/tools/queue5r.sh panels --workerwatch

Use one Bash `run_in_background` command each. A job cut by the time limit leaves no output, and the next worker
reruns it. A container restart stops them too: relaunch both, and finished jobs are skipped. Never launch it as
`bash queue5r.sh`, because the worker relaunches "$0" and needs a path (stance5, 2026-10-21 20:32 UTC). The budget is one
judge process and one panels process at a time, under nice -n 19, with OMP/MKL/OPENBLAS at 1 thread. Disk is tight:
the queue stops below 1 GB free.

`cd $T && ./queue5r.sh judge --dry-run` lists the plan (writes nothing).

## After a resume

As stance5's README, section "After a resume", with `stance5r-fetch-named.js`, `place5r.py` and `satwatch5r.py`:

    stance5r-fetch-named.js {"tag": "ledgers_<hhmm>", "csv": true, "extras": ["progress.json"], "runs": ["RUN_ID"]}
    cd $T && python3 -I place5r.py csv RUN_ID $R/fetch/ledgers_<hhmm>/RUN_ID --resumed-from N
    cd $T && python3 -I satwatch5r.py --roll

The placed ledgers that `--resumed-from N` replaces are moved aside, never deleted (Run handling, below). A checkpoint
the resumed session saved again (a step after N that was fetched before the resume) is refused by the fetch's place
step ("exists and differs ... --resumed-from N --supersede"). Place it by hand:

    cd $T && python3 -I place5r.py <steps> RUN_ID $R/fetch/r<steps>/RUN_ID --resumed-from N --supersede

The csv placement above prints `placed_after_N`: the periodic pairs already placed after N. Each of them was fetched
before the resume, so it is the old session's. Nothing else flags it, and its judge file would stand. Fetch it again
once the resumed session has saved it (its progress.json lists the step), and place it with the command above. "The
same RUN_ID" here and in rules 3 and 4 means the run's Drive folder: the RUN_ID, or its `_b` / `_c` folder after a
restart (rule 1; criteria lines 44-45 read that way, inference from line 42).

## Run handling (criteria RUN HANDLING, the maintainer's rules of 2026-10-22 16:32 UTC)

What each rule means for these tools. "Moved aside" is always `R/superseded/<RUN_ID>/<UTC time>/<the path under R>`,
listed in `SUPERSEDED.json` there. Nothing is deleted.

1. **A restart before the first checkpoint** (the Colab steps' case d) IS the run. Check on Drive that the first
   folder's `01_stance/models` lists no `stage1_*_steps.zip` (the find workflow reports it). Then fetch from the
   restart folder and place under the RUN_ID: pass its ids and its name on every fetch call,
   `{"drive": {"RUN_ID": {folder, stance, models of RUN_ID_b}}, "source": {"RUN_ID": "RUN_ID_b"}}`. The fetch workflows
   check that folder's title and pass `--source-folder RUN_ID_b` to `place5r.py`. The first such placement switches
   the run (`R/runs/<RUN_ID>/source_folder.json`), and whatever the first folder had placed (records, ledgers,
   progress.json) is moved aside. The switch is refused once a periodic checkpoint is placed. After it, a call without
   the folder, or with an earlier one, is refused. The judge, the queue and the scorer work under the RUN_ID as
   always. The scorer prints the folder as a note. A second restart is `_c`, likewise. Anything else: stop and ask.
2. **A judge refusal on the run's own records** (seed, commit, n_envs, hyperparameters, task, env kwargs, gate bars,
   plant, a record missing on Drive) is an INVALID J file. `judge_one5r.sh` (full mode) writes
   `R/judged/<RUN_ID>/refused_<steps>.json` from the judge's "refusing (nothing written): ..." line and exits 5. The
   queue shows the slot as "refused" and never reruns it. The scorer counts the record as invalid.
   - `place5r.py` checks most of these records first, so on its own the judge would never see them. Without the next
     step, the slots would read "missing", and the run could never meet "Exists" (rule 5): its checkpoints are fetched
     but cannot be placed. When `place5r.py config` (or a `configs: true` fetch) refuses the run's own records, a
     check failing or a record absent on Drive, and A and B agree: confirm on Drive that the run's 01_stance folder
     holds those very files (or lacks the absent one), and write it in the run log. Then run
     `cd $T && python3 -I place5r.py config RUN_ID DIR_A DIR_B --run-records-fail`. It places the agreed records as
     they are (never over a different placed copy) and writes `R/runs/<RUN_ID>/01_stance/records_check_failed.json`.
     That file lists the problems, the records absent on Drive and their sha256s. While it stands, the fetches place
     the run's pairs, and the judge refuses on the records. `judge_one5r.sh` then writes a refusal record at each slot,
     an invalid J file. A record listed there as absent on Drive is the run's side when the judge finds it missing.
     The criteria name "a missing stage_config.json". A plant_identity.json or task_fingerprint.json absent on Drive
     is read as the criteria's "plant" and "task" (inference). Any other missing record is a local file not placed:
     the operator's side. `place5r.py` checks two things the
     judge does not refuse on: run.timesteps and the curriculum's collapse_peak_floor_reference. Neither is among the
     judge's run bindings, and the reference is not a StanceV2Thresholds field (measured, by reading the judge and
     the clone). A run whose only problems are these is judged as it stands, and its final is read by its
     num_timesteps (rule 4). Either one differing without a commit mismatch is not expected (inference).
   - A refusal that names only the checkpoint's own attributes is held for review. Those attributes are its seed,
     plant, task or `mesozoic_symmetry` record, and the placed records pass. Its most likely cause is the operator's
     side: a zip fetched from another run's models folder (a wrong id in `args.drive`; inference).
     `judge_one5r.sh` writes `refused_<steps>.review.json` instead and exits 6. The scorer does not read that file.
     The queue shows "review", and nothing is judged while it stands. Confirm with get_file_metadata that the fetched
     zip's parent is the run's models folder id. If it is, rename the file to `refused_<steps>.json` (run-side), and
     note it in the run log. If it is not, the error is on the operator's side: move the review file and the placed pair
     aside by hand under `R/superseded/<RUN_ID>/<UTC time>/` (never deleted), note it in the run log, and fetch the
     slot again with the right ids.
   - An operator-side refusal (a wrong path or clone, a missing local file, a symmetry check the judge could not run)
     writes no record. Fix it, and the queue judges the slot again. A record later found to be the operator's side:
     move it aside by hand, and record why in the run log.
3. **A resume** is the same run, in the same folder (the RUN_ID, or its `_b` / `_c` folder after a restart). Its J
   checkpoints are judged as they stand. A file fetched before the resume and saved again by the resumed session:
   `place5r.py <steps> RUN_ID DIR --resumed-from N --supersede` (After a resume). That moves aside the placed pair, its
   judge files, refusal record and review hold (every mode), its panels output, its saturation roll and, at final, its
   certificate folder. Then it places the new pair, and the queue judges the slot again. `placed_after_N` (After a
   resume) names the placed pairs this applies to.
4. **A cut-short final save**: the run resumes from the newest intact periodic checkpoint in the same folder. The
   final slot is only a `stage1_final` pair at 12,001,280 in that folder. The judge records the checkpoint's
   num_timesteps, and the scorer counts any other in the final slot as invalid (score_stance3.py's check, unchanged).
   Without one, the final J file is missing.
   The notebook alone cannot do this. Its section 5 measures the shortfall against its default 100,000-step cadence:
   12,000,000 - 11,534,336 = 465,664 is more than that. So it calls a broken final pair "an early stop whose final
   save was cut short" and asks for a fresh RUN_ID, which this rule forbids. How to finish it (inference, from the
   notebook and `environments/shared/curriculum/checkpoints.py` read at cf80ecb1, not run):
   - First read the run's ledgers (`stance5r-fetch-named.js` with `csv: true`). The budget was spent when
     `train_episodes.csv` has rows past the last evaluation before the end (11,927,552), up to about 12,000,000, and
     the session reported no collapse early stop.
   - If training stopped short (a collapse early stop whose final save was cut), do not go on. Tell the maintainer.
     The notebook's own refusal is right there, and rule 5 reads an early stop only from a gate_verdict.json, which
     the run lacks.
   - Otherwise ask the maintainer to rename two files in the run's `01_stance/models/` on Drive, deleting nothing:
     `stage1_final.zip` to `stage1_final.zip.cut`, and `stage1_final_vecnorm.pkl` to `stage1_final_vecnorm.pkl.cut`
     if it is there. Then resume as in step 9 (`RESUME_STAGE = "stance"`). With no `stage1_final.zip`, section 5
     finds no final pair to check and skips the cadence refusal. Its walk (`newest_intact_periodic_pair`, glob
     `stage1_*_steps.zip`) and the chain loop's `*.zip` glob both ignore the `.cut` names. It resumes from the newest
     intact periodic pair, 11,534,336 when that one is intact. The final save comes after 57 rollouts of 8,192 steps,
     at 12,001,280. The fetches take files by exact title, so the `.cut` files are never fetched or judged.
5. **Ended and exists.** A run has ended when it finished at 12,001,280; or stopped early, meaning `gate_verdict.json`
   was written before 12,001,280 (the scorer reads the placed copy's `stage_result.timesteps`: the final J file is then
   invalid, and the placed `stage1_final` is not a J checkpoint); or was lost or never started and declared abandoned by
   the maintainer (run log). Use `--runs-ended` only once every J checkpoint in each run's Drive `01_stance/models` has
   been fetched, placed and judged, or its fetch failed for good (A and B disagree and no tie-break copy matches; write
   it in the run log). That folder is the one the run's files come from: the RUN_ID, or its `_b` / `_c` folder after a
   restart. The earlier folder is never judged. "Every J checkpoint" means the J checkpoints that exist there (criteria
   line 47), not a `.cut` file (rule 4). "Judged" includes a refusal record (rule 2: for records `place5r.py` refused,
   only after `--run-records-fail`). A review hold (`refused_<steps>.review.json`) is not judged yet.
6. **A scorer refusal on the stance5 runs** ("refusing to score: ... do not reproduce"): tell the maintainer at once.
   Read no line and act on none. Once the cause is found, run the scorer again with every pinned file unchanged.
7. **No L4 for a resume**: the maintainer tells the coordinator before resuming on another runtime (the Colab steps,
   step 9 and the short version). Report the runtime from that session's stage_config.json "gpu" block, as for every run.

## How to score

    cd $R/judge && python3 -I score_stance5r.py

- The stance5 runs are checked first. If their recorded verdicts no longer reproduce, the scorer refuses (no decision).
  `python3 -I score_stance5r.py --validate` shows that check alone; it must print "reproduced: yes".
- If the scorer refuses because the stance5 runs do not reproduce, follow Run handling rule 6: tell the maintainer
  at once, and read no line.
- Until every new-run J file exists, the decision is "incomplete". A refusal record and an early stop's final count as
  invalid files, not missing ones. After both runs have ended (Run handling rule 5, and only then), check that every J
  checkpoint placed under `R/runs/<RUN>/01_stance/models` has its `R/judged/<RUN>/judge_<steps>.json` or
  `refused_<steps>.json` (a `refused_<steps>.review.json` does not count: settle it first, rule 2). Then run `python3 -I score_stance5r.py --runs-ended --json $R/scores/score_final.json`
  (`mkdir -p $R/scores` first).
- Line 1 is ADOPT. Lines 2 and 3 are not adopted: the scorer prints the new runs' failing tests and K1 failure modes,
  and the maintainer chooses the next round.
- Tests: `cd $R/judge && python3 -I -B -m pytest -p no:cacheprovider -q test_score_stance5r.py test_tools_run_handling.py --basetemp=$S/caches/pytest_r/t`.
  `test_tools_run_handling.py` covers the tools' side of the run-handling rules: `place5r.py` on scratch roots,
  `judge_one5r.sh`'s refusal classification and `--check`, the queue's status, and both build checks.

## Checks made while building this (2026-10-22, measured)

- `judge_stance5r.py --seed 52` on stance5_LTSB_s52 at 4,194,304 reproduces stance5's judge file field for field. The
  only differences are schema, judge path and sha256, argv, `control_sigma_file` (+ sha256) and timings
  (`R/judge/validation/compare_v1_s52.json`).
- `judge_stance5r.py --seed 70 --allow-seed-mismatch` on the same checkpoint (validation only) reads the sidecar. Its
  noise test records the sidecar's vector and source, and only the joint-noise block and the seed relaxations differ
  from the record (`R/judge/validation/compare_v2_as_s70.json`). Joint-noise falls were 0 of 20 under the mean sigma,
  as under s52's own (report only, one checkpoint).
- `score_stance5r.py --validate`: the three stance5 runs' criteria, J cells and K1 failure modes reproduce
  score_final.json. The test suite passes (`test_score_stance5r.py`).
- `satroll5r.py` on stance5_LTSB_s42 at 4,194,304 reproduces stance5's roll exactly: tail_3_pitch_act at +1, frac90
  0.774, 10 of 20 over the bar (`T/validation/satroll5r_stance5_LTSB_s42_4194304.json`).
- `satwatch5r.py --validate-on-stance4 --no-state`: s42's alarm at 2,228,224, s52 and s60 none. With nothing placed it
  exits 3 with a LEDGER PROBLEM for each run.
- `place5r.py` on a scratch root (`PLACE5R_TEST_ROOT`): it refuses a wrong seed and a stance5 RUN_ID, notes a label
  typed as the RUN_ID, places records, a pair and a ledger, and reports status.
- The workflows dry-run under `wfcheck/harness.mjs` (agree, differ, notyet; placeholders refused; a stance5 RUN_ID
  refused). `judge_one5r.sh`, `panels5r_one.sh` and `queue5r.sh` pass their dry runs and checks. `cert5r_one.sh` refuses
  a stance5 run.

- Review fixes (2026-10-22, about 16:30 UTC): criteria text corrections re-pinned (CRITERIA_SHA256_R), the scorer's
  next-round line, the steps' seed and fresh-copy checks, `cert5r_one.sh`'s mkdir order, and both build scripts
  refusing unknown arguments. Re-run record: `R/validation/review_fixes/` (tests 37 passed, `--validate` reproduced,
  the synthetic decision cases all as expected, `make_tools.py --check` and `make_colab_steps.py --check` ok).

- Run handling (2026-10-22, about 17:00 UTC; the maintainer's rules of 16:32 UTC): criteria section RUN HANDLING and
  the sigma confirmation, re-pinned; the scorer's refusal records, early stop and source-folder note, with tests for
  both rules; `place5r.py --source-folder` and `--supersede`; `judge_one5r.sh`'s refusal record; the queue's
  "refused"; `args.source` in the fetch workflows; the Colab steps' `_b` restart, cut-short final and no-L4 lines. Re-run
  record: `R/validation/run_rules/` (place5r.py and judge_one5r.sh on scratch roots, the wfcheck harness, tests,
  `--validate`, both build checks, pins).
- Review of the run handling (2026-10-22, about 17:45 UTC; criteria, judge and sidecar unchanged, scorer re-pinned in
  the steps). Changes:
  - `place5r.py config ... --run-records-fail`, so that run records `place5r.py` refuses reach the judge as rule 2
    says.
  - `judge_one5r.sh`: a missing record is the run's side only when listed absent on Drive, and a refusal on the
    checkpoint's own attributes is held for review (exit 6, the queue's "review").
  - `placed_after_N` after a resume.
  - The scorer reads the placed gate_verdict.json under `--ledgers` too, and notes the two new files.
  - The tie-break prompt names the `args.source` folder.
  - The Colab steps carry `_c` through, and say how a cut-short final save is finished (rule 4).
  - `make_sigma.py`'s docstring records the 16:32 confirmation.
  - `judge/test_tools_run_handling.py`.

  Re-run record: `R/validation/skeptic_fixes/`. It holds each item checked before and after, rule 2 end to end with
  the real judge on a scratch root, the wfcheck prompt comparison, tests (66 passed), `--validate`, the eight
  synthetic decision cases, both build checks and the pins.

- Time correction (2026-10-22, after the run log's TIME CORRECTION entry stamped 17:47 UTC; judge and sidecar
  unchanged). The maintainer-decision times cited from the run log now read as corrected there: decisions 15:08, reading
  rule and sigma default 15:11, sigma confirmation and run-handling rules 16:32 (UTC). Changed in the criteria
  (re-pinned), the scorer's docstring and printed parent line, `make_sigma.py`'s docstring, `make_tools.py` (so
  `judge_one5r.sh` and `CHANGES.diff`), this README and the Colab steps. The sidecar's `computation.decision` string,
  and the `make_sigma.py` code that wrote it, keep the pre-correction time; the criteria say so. Re-run record:
  `R/validation/time_fix/` (tests 66 passed, `--validate` reproduced, both build checks, pins).

## Re-pin cascade (only before the first training step)

If `criteria.txt`, `judge_stance5r.py` or `sigma_s70_s80.json` changes (for example after a review), go through these
steps in order.

1. Criteria changing: `chmod u+w criteria.txt` (it is kept 0444), then edit it.
2. Sidecar changed: update `CONTROL_SIGMA_SIDECAR_SHA256` in the judge and `SIDECAR_SHA256` in the scorer.
3. Judge changed (step 2 changes it too): rewrite `R/judge/judge_diff.txt` with
   `diff -u S5/judge/judge_stance5.py R/judge/judge_stance5r.py`, then update `JUDGE_SHA256_S5R` in the scorer.
4. Criteria changed: update `CRITERIA_SHA256_R` in the scorer.
5. `cd R/build && python3 -I make_colab_steps.py --force`. It embeds the criteria and the four sha256s; the scorer's own
   sha256 changes at every pin. A change to the scorer alone (criteria, judge and sidecar unchanged) needs steps 5 and
   6 only, and the new `colab_steps.txt` / `colab_short.md` go to the maintainer.
6. Run the test suite. `test_pins_are_the_files_as_they_stand` and `test_judge_diff_is_the_recorded_one` fail until
   steps 2-5 are done.
7. `chmod 0444` on `criteria.txt` and `sigma_s70_s80.json`, and rewrite `R/criteria.sha256`
   (`cd R && sha256sum criteria.txt > criteria.sha256`). Send the maintainer the new `colab_steps.txt` /
   `colab_short.md`.

## Caveats

- The Drive ids are placeholders until the runs exist (above).
- The per-term reward decomposition (criteria READOUTS) has no wrapper here, as in stance5 (its README said so). It is
  rolled ad hoc from the checkpoint (the env info key reward_stance_cop_band and the other reward terms, stance5
  criteria READOUTS) or reported as not produced.
- Do not edit `judge_stance5r.py`, `sigma_s70_s80.json` or `criteria.txt` after the first training step. Their sha256s
  are pinned in the scorer (and the sidecar's in the judge).
- Nothing here writes under `S5` outside `R`. The stance5 judge, scorer, criteria, tools and data are read-only.
