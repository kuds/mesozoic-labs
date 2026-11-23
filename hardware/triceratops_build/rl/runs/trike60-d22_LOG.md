# Foreleg comparison (D22): run log

What happened to the eight runs on Colab, as read from the owner's Drive (run records, checkpoint listings, notebook
copies). Rules: `analysis/PREREG_FORELEG_D22.md`. Steps: `runs/trike60-d22_STEPS.md`. Times are UTC.

## Round 1, 23 Nov 2026: a mis-set start, stopped and restarted before any result

**What happened.**
- 23:09-23:11: three notebook copies started. `trike60-fore128-s0` was set correctly and is the run that goes on
  (session started 23:11:04, seed 0, fore128).
- The copy saved as "trike60-fore128-s1" had RUN_ID `trike60-fore160-s0` (MODEL fore128, SEED 1). It wrote into
  `trike60-fore160-s0`'s folder, where the real fore160-s0 session (seed 0, fore160) had started at 23:11:48.
  Drive had not yet shown that run's record to the second machine, so train.py's settings check could not refuse it:
  it began a second "new run" in the same folder.
- The owner stopped both: the mis-set session at 1,168 steps (23:12:49), fore160-s0 at 13,936 steps (23:12:55).
  Each left a stop pair in `checkpoints/`. Both wrote `run_record.json` in turn (Drive revisions at 23:11:54,
  23:12:48, 23:12:58 and 23:13:05; the last is fore160-s0's own, intact). Both wrote `monitor/session0/`'s CSV files.
- The owner set the copy's RUN_ID to `trike60-fore128-s1` and started it in a new folder (session 23:14:16, seed 1,
  fore128). They then moved that folder and the mixed fore160-s0 folder to the Drive trash, and started both runs
  again from step 0 in fresh folders: `trike60-fore160-s0` (folder 23:16:57, session 23:18:21, seed 0, fore160) and
  `trike60-fore128-s1` (folder 23:17:02).

**Effect on the study.** None on what is judged:
- No result existed. The furthest any start got was 13,936 of 10,000,000 steps, before the first checkpoint
  (250,000) and the first evaluation.
- The seeds, models and settings are unchanged. The restarted runs start from step 0, as runs without the error would.
- Nothing was chosen by outcome.

**Departure from the stop rule, reported.** The pre-registration says a run that cannot resume is restarted once under
a new RUN_ID, with its first files kept. Here the first folder held another run's files beside fore160-s0's (a stop
pair and the monitor CSVs), so resuming from it was not safe. Instead, the owner restarted under the same RUN_ID and
moved the first files to the trash. Copies of the two discarded run records are kept in `runs/d22_startup/`:
- `fore160-s0_first-folder_run_record.json` (sha256 77d1fb08…): the 23:13:05 version, fore160-s0's session 0.
- `fore128-s1_first-folder_run_record.json` (sha256 804e126c…): the first corrected start, trashed.

The restarted runs are judged as the runs. This start-up error does not count as a run's one allowed restart. The
D22 report repeats this note.

**Machines.** All three runtimes report 2 vCPUs (run 3 had 8). `trike60-fore128-s0` trains at 213 env steps/s
(23:11:40-23:16:09, from `train_health.jsonl`), so a run takes about 13 hours instead of run 3's 3-5. The settings
are the same either way: one torch thread, and each env seeded.

## Checks on the runs as they go (the record-only part of `judge_d22.checks`)

| run | session start | seed | model | commit, clean | settings and arguments as run 3's | vCPUs |
|---|---|---|---|---|---|---|
| trike60-fore128-s0 | 23:11:04 | 0 | fore128 | yes | yes | 2 |
| trike60-fore160-s0 | 23:18:21 (restart) | 0 | fore160 | yes | yes | 2 |
| trike60-fore128-s1 | 23:18:43 (restart) | 1 | fore128 | yes | yes | 2 |
