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

**Machines, and a second restart.** Those three runtimes all had 2 vCPUs (run 3 had 8). `trike60-fore128-s0`
trained at 213 env steps/s (23:11:40-23:16:09, from `train_health.jsonl`), so a run would have taken about 13 hours
instead of run 3's 3-5. The owner therefore stopped all three, moved their folders to the trash, and started all
three again from step 0 on High-RAM CPU runtimes (sessions 23:24:48, 23:25:22 and 23:25:22, 8 vCPUs each). This
restart is part of the same start-up and not a result-driven change: no run had reached 100,000 steps or any
evaluation. The machine does not change the settings: one torch thread, and each env seeded. Copies of the three
stopped records are in `runs/d22_startup/` (`*_2vcpu-*_run_record.json`).

## Checks on the runs as they go (the record-only part of `judge_d22.checks`)

The runs that count, from 23:24 on:

| run | session start | seed | model | commit, clean | settings and arguments as run 3's | vCPUs | first rate |
|---|---|---|---|---|---|---|---|
| trike60-fore128-s0 | 23:24:48 | 0 | fore128 | yes | yes | 8 | 565 env steps/s |
| trike60-fore160-s0 | 23:25:22 | 0 | fore160 | yes | yes | 8 | 878 env steps/s |
| trike60-fore128-s1 | 23:25:22 | 1 | fore128 | yes | yes | 8 | 630 env steps/s |

The rates come from the first 3-5 rollouts and point to about 3-5 hours per run, as in run 3. Stopped starts:
fore128-s0 at 23:11:04 (2 vCPUs); fore160-s0 at 23:11:48 (the mixed folder) and 23:18:21 (2 vCPUs); fore128-s1 at
23:14:16 and 23:18:43 (2 vCPUs).

## Round 1 finished, 24 Nov 2026

Each run had one session, from step 0, completed at 10,002,432 steps, with its final pair finite. Each record passes
the record checks of `judge_d22.checks`. The final pairs were downloaded from Drive for measurement (sha256 below);
the measurements use a `git archive` export of the run commit through `TRIKE_ENV_DIR`.

| run | session (UTC) | hours | env steps/s | final pair, zip sha256 | vecnormalize sha256 |
|---|---|---|---|---|---|
| trike60-fore128-s0 | 23:24:48-04:32:27 | 5.13 | 542 | f7bbeb81… | cc645bd9… |
| trike60-fore160-s0 | 23:25:22-03:06:47 | 3.69 | 753 | 0eb877a9… | 19981491… |
| trike60-fore128-s1 | 23:25:22-03:55:44 | 4.51 | 617 | dc3090ac… | f7636250… |

Box 6's quick test (not a judged measure) at 0.25 m/s with a 10 ms delay: fore160-s0 and fore128-s1 had no falls in
5 episodes, at 0.242 and 0.244 m/s. The first start of the measurements (04:33) stopped at once, because the tools
read `run_record.json` from the checkpoint folder and it was not there. Nothing was measured; it was restarted with
the record in place.

## Round 1 measured, 24 Nov 2026 (interim: figures per run, no choice until all eight)

The measurements ran on 04:33-05:10 UTC with the run commit's tools (an export of f84a6919 through `TRIKE_ENV_DIR`).
Each run's three outputs pass every check in `judge_d22.checks`, using the tree's `judge_d22.py`, which is byte-identical
to the run commit's: the export has no git history to read the run commit's files from. Figures come from
`judge_d22.measure` and `judge_run3.judge_seed`. Outputs: `analytics/results/<run>/`, `robustness/results/<run>/`,
`analysis/results/d22/`.

| run | F1 | F2 (worst drive) | T1 | T2 | T3 | T4 | T5 | L front (hind), N | H_all | H_front | H_elbow | N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fore128-s0 | pass | pass (left hind pitch 0.828) | pass | pass | fail | fail | fail | 122.9 (112.4) | 0.828 | 0.692 | 0.638 | 14 |
| fore160-s0 | pass | **fail (left fore pitch 1.014)** | pass | pass | fail | fail | fail | 134.3 (125.7) | 1.014 | 1.014 | 0.738 | 11 |
| fore128-s1 | pass | pass (right hind flexion 0.888) | pass | pass | fail | fail | fail | 119.5 (106.5) | 0.888 | 0.811 | 0.702 | 24 |
| run 3 s0 (188 mm, reference) | pass | pass (0.836) | pass | pass | pass | pass | fail | 143.5 (135.2) | 0.836 | 0.823 | | 25 |
| run 3 s1 (188 mm, reference) | pass | pass (0.801) | pass | pass | pass | pass | fail | 110.6 (122.9) | 0.801 | 0.722 | | 18 |

- No foot-load episode fell; no run fell on the steps, the ramps or the 0.3 friction floor.
- T3 and T4 fail on sideways drift, not falls. T3's limit is 0.30 m: the runs drifted 0.27-0.43 m (fore128-s0),
  0.46-0.60 m (fore160-s0) and 0.28-0.58 m (fore128-s1), against run 3's 0.08-0.26 m. T4's limit is 0.20 m: they
  drifted 0.43, 0.82 and 0.45 m, against run 3's 0.15 and 0.06 m.
- T5 (pushes): falls at -1.25, -1.0, +1.0 and +1.25 m/s, out of 8 each: 8/0/0/6, 4/0/1/6 and 8/1/7/8.
- **Consequence under the pre-registered order (D24):** 160 mm now cannot pass the walk gate (F1 and F2 on all
  four seeds). The choice can only be 128 mm, if 128 mm passes F1 and F2 on seeds 2 and 3 as well, or none. The
  remaining 160 mm runs still have to be trained and measured: `choose()` makes no choice if any of the eight is
  missing or not judged.

## Round 2 started, 25 Nov 2026

Each record passes the record checks of `judge_d22.checks`: the run commit, a clean tree, its own seed and model,
and run 3's settings and arguments. Each run writes to its own folder, on 8 vCPUs.

| run | session start (UTC) | seed | model | output folder | vCPUs |
|---|---|---|---|---|---|
| trike60-fore160-s1 | 15:13:02 | 1 | fore160 | trike60-fore160-s1 | 8 |
| trike60-fore128-s2 | 15:12:56 | 2 | fore128 | trike60-fore128-s2 | 8 |
| trike60-fore160-s2 | 15:12:47 | 2 | fore160 | trike60-fore160-s2 | 8 |

## Round 2 finished, 25 Nov 2026

Each run had one session, from step 0, completed at 10,002,432 steps, with its final pair finite. Each record passes
the record checks of `judge_d22.checks`. The final pairs were downloaded from Drive (the sizes match Drive's; each
from its own run's `checkpoints/` folder).

| run | session (UTC) | hours | env steps/s | final pair, zip sha256 | vecnormalize sha256 |
|---|---|---|---|---|---|
| trike60-fore160-s1 | 15:13:02-20:05:54 | 4.88 | 569 | 85e6ec3b… | 024f36f6… |
| trike60-fore128-s2 | 15:12:56-20:17:19 | 5.07 | 548 | e41f7bd9… | 029d6dab… |
| trike60-fore160-s2 | 15:12:47-18:46:49 | 3.57 | 779 | 1f53d262… | 040f6355… |

## Round 2 measured, 25 Nov 2026 (interim: figures per run, no choice until all eight)

Measured on 22:26-23:09 UTC, as round 1. Each run's three outputs pass every check in `judge_d22.checks`.

| run | F1 | F2 (worst drive) | T1 | T2 | T3 | T4 | T5 | L front (hind), N | H_all | H_front | H_elbow | N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fore128-s2 | pass | pass (left fore pitch 0.931) | pass | pass | fail | fail | fail | 117.5 (130.4) | 0.931 | 0.931 | 0.757 | 9 |
| fore160-s1 | pass | **fail (right hind flexion 1.013)** | pass | pass | fail | pass | fail | 109.1 (129.9) | 1.013 | 0.965 | 0.768 | 14 |
| fore160-s2 | pass | **fail (left hind flexion 1.094)** | pass | pass | pass | pass | fail | 105.2 (107.7) | 1.094 | 0.902 | 0.834 | 13 |

- No foot-load episode fell; no run fell on the steps, the ramps or the 0.3 friction floor.
- Sideways drift varies by seed more than by length. On T3's points (limit 0.30 m), fore128-s2 drifted 0.21-0.33 m,
  fore160-s1 0.14-0.36 m and fore160-s2 0.09-0.27 m. At friction 0.3 (T4, limit 0.20 m) they drifted 0.65, 0.11
  and 0.20 m.
- T5 (pushes): falls at -1.25, -1.0, +1.0 and +1.25 m/s, out of 8 each: 1/0/0/8, 8/2/0/4 and 7/2/1/3.
- So far, 128 mm passes F1 and F2 on seeds 0, 1 and 2. 160 mm fails F2 on seeds 0, 1 and 2, on a different drive
  each time (left fore pitch, right hind flexion, left hind flexion). Under D24's order the choice is 128 mm if
  fore128-s3 passes F1 and F2, and none if it does not; it is made only once all eight runs are measured and checked.

## Round 3 started, 26 Nov 2026

Each record passes the record checks of `judge_d22.checks`, with each run in its own folder on 8 vCPUs. At 14:26 UTC
both were at 800,000 steps, with saves at 250,000, 500,000 and 750,000.

| run | session start (UTC) | seed | model | output folder | vCPUs |
|---|---|---|---|---|---|
| trike60-fore128-s3 | 14:03:25 | 3 | fore128 | trike60-fore128-s3 | 8 |
| trike60-fore160-s3 | 14:03:43 | 3 | fore160 | trike60-fore160-s3 | 8 |
