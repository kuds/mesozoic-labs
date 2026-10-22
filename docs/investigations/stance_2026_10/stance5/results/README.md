# T. rex stance5 screen results (judged 2026-10-22)

Evidence for §21 of
[TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](../../../TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md):
the outcome of decision D-D40, the stance5 screen of the per-foot
centre-of-pressure band that the record's §20 decides: three 12M T. rex stance
runs of arm LTSB (stance4's arm LTS, the stance3 L+T recipe plus the stance
stagger penalty, plus the band at weight 0.30), from scratch on physics r9,
`stance5_LTSB_s52`, `stance5_LTSB_s60` and `stance5_LTSB_s42`, from the branch
`claude/screen-trex-r9-armlts-copband` at `cf80ecb1` (never merged, no pull
request), judged by the criteria fixed before training
([`../criteria.txt`](../criteria.txt), sha256 `3b28963f…`) against the three
stance4 runs as the control, not retrained. These are copies of the files that
decided or document each verdict, taken from the session's scratch tree on
2026-10-22 before it is lost. A screen certifies nothing: the verdicts are the
criteria's, not a gate's, and nothing here certifies a run, writes a result
bundle or promotes anything; `main` and its digests are unchanged.

The judge, the scorer and the scorer's tests are in [`../judge/`](../judge/)
(the judge's copy carries the prefix and checkout changes and the scorer's the
prefix change, so their sha256 is not the `d9286724…` each judge file records,
nor the scorer's `d5515041…`; the tests are unchanged). The per-episode judge
files (36 for the three runs, about 12 MB together), the diagnostic panels'
files (18, about 5.8 MB), the certificates' per-episode files, the rollouts and
the runs' ledgers and records stay in the session's scratch; the checkpoints are
on Drive under
`MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/stance5_*/` (each
judged checkpoint's sha256 is in `checkpoints.csv`).

**Outcome: Run 1 (s52) holds; Runs 2 and 3 do not; H = 1 and K2p = 3; the
decision table's line is 3, final.** The band is not adopted: "H == 1: not
adopted; not separated from seed luck; replicate on 2 more seeds before any
commit." Fails stay fails, and no run was stopped or changed on its readouts or
on the saturation alarm.

| Run | Run id | K1 gate held | K2 fore-aft held | K3 robustness | K4 roll | Verdict |
|---|---|---|---|---|---|---|
| 1 | `stance5_LTSB_s52` (LTSB, seed 52) | PASS: 40/40 clean at all six J checkpoints | PASS: K2a 0 and K2b 0 at every J; K2c range 17.4 mm | PASS: no noise falls at 6,291,456 or the final | PASS | HOLDS; fore-aft holds |
| 2 | `stance5_LTSB_s60` (LTSB, seed 60) | PASS: 39, 40, 40, 40, 40 and 39 of 40 clean | PASS: K2a 0 and K2b 0 at every J; range 21.6 mm | FAIL: joint-space noise falls 3 of 20 at the final (bar 2); own-sigma 0 | PASS | does not hold (K3 only); fore-aft holds |
| 3 | `stance5_LTSB_s42` (LTSB, seed 42) | FAIL: v2 FAIL at 6,291,456 and the final; 16, 27, 18, 15, 38 and 19 of 40 clean, mostly on the actuator-saturation rail | PASS: K2a and K2b 0 at every J but the final (1 each, a fall); range 18.3 mm | FAIL: joint-space noise falls 13 of 20 at the final; own-sigma 1 of 20 (the limit) | PASS | does not hold (K1, K3) |

The credit rule, against the control's passes recorded in stance4's
`score_final.json` (K1 0/3, K2 0/3, K3 2/3, K4 2/3): the band arm passes K1 2/3,
K2 3/3, K3 1/3 (worse than the control) and K4 3/3, so (a) at least two runs
holding: no, and (b): fails. D = 0 and S = 0 (no band run fails K2, so none is
classed). The control's verdicts reproduce stance4's score of record from its
judge files. The parent comparison (report only, stance3's rule): s52's final
(40/40, no noise falls) ties stance3 Run 2's final (40/40, 0); the rule names
both and does not choose. Walking pilot 2's parent stays stance3 Run 2's final
(D-D36 as amended on 2026-10-20); the maintainer's decisions of 2026-10-22 that
follow from the outcome are in `run_log.txt` from its 15:08 UTC entry on, and
the stance record's §22 records them.

The runtime, measured: the `gpu` block of each run's
`01_stance/stage_config.json` reads an NVIDIA L4 (23.7 GB, CUDA 13.0), with
`git_commit` `cf80ecb1` and each run's seed, as the run log records at 13:46 UTC
on 2026-10-21 and as re-read for this record. Each run's `progress.json` records
one session: `session_started_at` 13:37:07 (s52), 13:36:43 (s60) and 13:38:28
UTC (s42) on 2026-10-21, ended at 12,001,280 steps at 03:32:11, 04:03:00 and
03:14:04 UTC on 2026-10-22, at 239.5, 230.9 and 245.2 steps per second (the
score's throughput lines), with no warnings; so that each ran on the L4
throughout is inferred. The run log gives each run's start from its
`provenance.json` instead (13:33:09, 13:32:45 and 13:34:32 UTC), and an aborted
first s52 start at 13:26 UTC under the wrong run id.

## What changed in the copies

[`../../SOURCES.txt`](../../SOURCES.txt) lists every file here but this README
with the scratch file it came from, that file's sha256, and what the copy
changed; [`../../SHA256SUMS`](../../SHA256SUMS) pins them. The parent folder's
**prefix** (the scratch root becomes `scratchpad`, the session directory above
it `session`) and **whitespace** changes apply as its [README](../../README.md)
defines them (here **whitespace** adds a final newline to
`scores/score_final.json`, the six `saturation/` files and the two certificates'
`run_summary.json`), and **trim**, **redact**, **checkout** and **lines** as
SOURCES.txt defines them, nothing else:

- **trim:** each `notebook_gates/<run id>/gate_verdict.json` keeps the keys
  stance4's do (the gate verdict's identity and verdict fields and, of its
  `stage_result`, the timesteps, the best evaluation's step, the gate's verdict
  and failures, and the clean-seed count, seed count, fraction and LCB); each
  `certificates/<run id>/run_summary.json` keeps the keys stance3's do (the
  panel seeds, the arm's leg and lean keys, the checkpoint path and its and the
  normaliser's sha256, the verdict and its failures, the clean-seed and seed
  counts, their LCB, the task and gate sha256, and per heading its clean count,
  LCB and full-horizon fraction).
- **lines:** `run_log.txt` is lines 208 to 395 of its source, the session's run
  log, which holds the stance4 log (its first 207 lines, copied by #633 as
  `../../stance4/results/run_log.txt`) and then this one; the source had 395
  lines when first copied. The session later corrected in place the times of
  nine entries in those lines (see `run_log.txt` below), and the copy carries
  the corrected times: sha256 of the source's first 395 lines `b6641664…`, of
  lines 208 to 395 `27408eec…` (before the correction, `c954d76c…` and
  `ff83ab9f…`).
- **redact:** in `run_log.txt`, each of the session's tool identifiers is
  replaced by `[id removed]`: 30 workflow run ids (18 distinct; a bare one
  becomes `workflow [id removed]`), 2 scheduled-task ids and the report
  document's artifact link (now `claude.ai artifact [id removed]`); the 9
  shortened Drive folder ids (seven or eight characters and `...`, three per
  line on three lines) are replaced the same way; and the assistant's name in
  the name of a GitHub check by `[name removed]`. Nothing else in the log
  changes.
- **checkout:** the session's checkout directory is replaced by `checkout` once
  in `run_log.txt` (the 07:17 UTC entry of 2026-10-21) and once in
  `../judge/judge_stance5.py` (its `MAIN_CHECKOUT` constant).

`checkpoints.csv`, `checkpoints_csv.py`, `panels_summary.csv`,
`panels_summary.py`, `extras.txt` and `extras.py` (`extras.txt` is its printout)
were written in the scratch for this record on 2026-10-22 and are copied
unchanged. No copy holds a model identifier, an email address or a commit
trailer.

## Contents

**`run_log.txt`: the stance5 log, as written** from 2026-10-20 21:05 UTC (#633
opened) to the entry timed 16:32 UTC on 2026-10-22. The band build's identity
and timing checks; the s42 saturation diagnosis of stance4
(`../../stance4/satdiag/`); the package (criteria, judge, scorer, the screen
branch pushed at `cf80ecb1`); walking pilot 2's rebuild and its push
(`../../pilot2/`); each run's start, read from Drive; per checkpoint the fetch
(each VecNormalize file transcribed twice and compared) and the judges'
readings; the saturation alarm and its naming; the finals, the notebook's gates,
the verdict of record and the report-only extras; the verification of the
session's human-facing report (a document outside the repository, not copied);
and the maintainer's decisions of 2026-10-22. Before the verdict of record its
decision lines are readings as the files then stood. Three readings it gives in
short: the 04:24 UTC "saturation 21" for s42's final is 20 (`scores/`, the K1
failure modes; the log's own 05:15 entry corrects it); the 06:20 UTC "(falls 0,
0, 1)" of the final's diagnostic panels is, per variant (standard, mirrored
legs, prey ahead, both), s52 0/0/0/0, s60 0/0/0/0 and s42 1/0/1/0
(`panels_summary.csv`); and its "no foot left the band" holds per episode, not
per foot-step (the standard panels in `panels_summary.csv`: up to 1,012 of
64,000 window foot-steps, s60 at 8,388,608; the 05:15 entry). Ten entries of
2026-10-22 written after 05:00 first carried guessed times; the session
corrected them in place to the times they were written, from its transcript (its
entry timed 17:47 UTC on 2026-10-22, after the copied lines), and the copy gives
the corrected times (`../../SOURCES.txt` lists the nine it holds).

**`scores/`: the scorer's printout and JSON, the verdict of record**
(`score_stance5.py --runs-ended`, written at 04:57 UTC on 2026-10-22, the files'
time, the run log's entry being timed 04:58; the JSON
is 414,719 bytes as written): K1–K4 per run with every cell, H, K2p, the credit
rule in both forms, the drift-or-split classes, the decision line (`status`
decided, final), the parent comparison, the control's cross-check against
stance4's `score_final.json` (sha256 pinned), and, report only, the
7,340,032-step checkpoint, the S0–S3 selection comparison, K5 (no flag on any
run), the late-fall fraction (above 5% only in the first 262,144-step bin,
46.3–50.2%, and s42's 11,796,480 bin, 6.6%), the band readouts and the
saturation alarm. s60's `best` and `robust_best` (the same checkpoint, at
6,815,744) were judged after it was written (`best` at 05:01, `robust_best` at
05:06 UTC; the judge files' times), so it lists
them as not judged; both are 40/40 clean in `checkpoints.csv`, with 2 of 20
joint-space noise falls. The other report-only 1,048,576-step checkpoints (1.0M,
2.1M, 3.1M, 9.4M, 11.5M) were never fetched, so 15 judge jobs found nothing to
judge, and S3 was not judged.

**`checkpoints.csv`: one row per judge file (36 rows, 12 per run).** The columns
of stance4's. Roles: `J`, the six judged checkpoints; `report (1,048,576-step)`,
7,340,032 only; `selection S0` and `S1`, `robust_best` and `best`
(byte-identical in each run: 917,504 steps in s52 and s42, 6,815,744 in s60);
the fresh panel 7042–7081, the 32-seed forward-start tail panel and the 40-seed
common-noise panel at the final. Only the J rows and the noise counts at
6,291,456 and the final enter a criterion; every other row is report only.
`checkpoints_csv.py` (stance4's, with the stance5 judge identity) wrote it: it
recomputes every number with `panel_counts` from
`../../stance3/judge/score_stance3.py` (the function `score_stance5.py` loads
unchanged) on the per-episode rows, and with `--check` stops on any difference
from the scorer's JSON (25 cells agree: every J, 7,340,032 and selection cell
but s60's `robust_best` and `best`, which the JSON does not have).

**`extras.txt` and `extras.py`: the report-only extras on each final**, measured
for this record on 2026-10-22 from the judge files (the run log gives no numbers
for them). The fresh panel: s52 FAIL 38/40 (1 fall, 2 hop-or-fall episodes
against a bar of 1),
s60 PASS 38/40 (0 falls), s42 FAIL 16/40 (2 falls); the forward-start tail
panel: 3, 0 and 4 falls of 32 (the statue 0); the common-noise panel: 6, 5 and
31 falls of 40. The criteria's references: the stance4 finals 39/40, 38/40 and
14/40 on the fresh panel, 2, 6 and 5 tail falls and 5, 1 and 30 common-noise
falls. None enters a verdict.

**`notebook_gates/`: the notebook's own end-of-stage gate,
`stance_quality_heading/v1`, report only** (trimmed `gate_verdict.json`, as
fetched from each run's Drive folder): each judged the run's `robust_best`
checkpoint at six headings, and all three PASSED, 40/40 seeds clean (LCB 0.928),
at 917,504 (s52, s42) and 6,815,744 (s60). Each `checkpoint_sha256` equals the
`robust_best` row's in `checkpoints.csv` (measured). These are not the screen's
verdicts: the gate judged an early checkpoint, not the finals the criteria
judge.

**`certificates/`: the stage's own six-heading certificate
(`stance_quality_heading/v1`) rolled offline on the s52 and s60 finals, report
only.** The runner is `../../stance3/results/certificates/run_cert_clone.py`
(sha256 `725dc02c…`, unchanged), run in a clone of the screen branch at
`cf80ecb1` with that branch's stance configuration, so each certificate rolls
the band-on task (`8c6998d2…`); its `env_kwargs_arm` lists only the five leg and
lean keys the runner reads, not the stagger and band keys the configuration also
holds. s52's final PASSED, 40/40 seeds clean at all six headings (LCB 0.928);
s60's PASSED, 39/40 (0.887). **Neither is gate evidence**: the band and its
configuration are not on `main`, whose stance task is not the band-on one these
rolled. Per run: `run_summary.json` (trimmed) and `stance_heading_report.txt`
(the certificate as printed); each `model_sha256` equals the final row's in
`checkpoints.csv` (measured). s42's final was not certified.

**`panels_summary.csv` and `panels_summary.py`: the diagnostic panels, report
only.** One row per variant (72 rows): the three band runs at each of the six J
checkpoints, under the judge's panel with no change (`standard`), the legs
mirrored at reset (`mirror_legs`), the prey straight ahead (`prey_ahead`) and
both (`mirror_prey_ahead`): the columns of stance4's, plus the band readouts
over the full-horizon episodes' window foot-steps (the whole-foot CoP outside
[−0.6, +0.4], the pad CoP outside [−0.65, +0.65], the guard at 0 and on its
ramp, the band paid per episode) and the ankle asymmetry (the panel sd of the
right-minus-left ankle servo error and torque), copied from `panels5.py`'s
files. `panels_summary.py` checks each file's model sha256 against the judge
file of the same checkpoint and the standard variant's K2a and K2b against the
judge's (all equal; 18 standard rows). At the finals, clean of 40 per variant:
s52 40/40/40/40, s60 39/37/39/37, s42 19/8/13/7, with falls 0/0/0/0, 0/0/0/0 and
1/0/1/0. The guard was never at 0 or on its ramp in any row (the full-horizon
episodes' window foot-steps). The control's panels are in
`../../stance4/results/panels_summary.csv`.

**`saturation/`: the saturation watch's files for s42, report only.**
`satwatch5_<step>.json` is the watch's reading of the three runs' evaluation
ledgers when each checkpoint was rolled; `stance5_LTSB_s42_<step>.json` is the
20-episode rollout (seeds 3042–3061, window steps 200–999) that names the
actuator. The ledger's first episode on the saturation rail was at 3,145,728 (1
of 10; the run log, 19:40 UTC); the first flagged evaluation (5 or more of 10)
at 3,538,944; the alarm (two flagged in a row) at 3,670,016, against stance4
s42's 2,228,224. In all, 30 of s42's 91 evaluations were flagged: 26 in four
alarm runs, 3,538,944–4,718,592 (10), 6,160,384–6,553,600 (4),
6,815,744–7,208,960 (4) and 7,864,320–8,781,824 (8), and 4 alone (5,111,808,
5,898,240, 10,878,976 and 11,927,552; `scores/`). The rollouts name
`tail_3_pitch_act` at +1 (tail raised) at each checkpoint rolled: at 4,194,304
its command was at 0.9 or more on 77% of window steps (mean +0.92), and 10 of 20
episodes were over the gate's bar; at 5,242,880, 67% (+0.92), 6 of 20; at
6,291,456, 75% (+0.90), 14 of 20. That is the actuator and sign of stance4 s42
and stance3 Run 2. s52 and s60: no evaluation flagged (0 of 91). The watch files
record that no `progress.json` was placed beside the ledgers, so they leave the
ledgers' freshness unchecked; the run log's 19:48 UTC entry says the direct
`progress.json` reads covered it.

## How to re-check

Run on 2026-10-22 (Python 3.13, `-I -B`). The rows that need the judge files run
only where the session's scratch is.

| Check | Command | Result |
|---|---|---|
| The verdicts, H, K2p, the credit rule, the decision line and the parent comparison from the judge files (session scratch only) | from the directory holding the session's scratch as `scratchpad/`: `python3 -I -B scratchpad/foot/stance5/judge/score_stance5.py --runs-ended --json score.json` (its default paths are the scratch's) | identical to the scratch's `scores/score_final.txt` and `.json` (the copies here differ from those only by the prefix change) apart from the path of the JSON it names and s60's two selection cells, judged after the record |
| The committed copy of the scorer (session scratch only) | from the directory holding the session's scratch as `scratchpad/`: `python3 -I -B <this folder>/../judge/score_stance5.py --runs-ended --json score.json` | the same verdicts, H, K2p, credit-rule results and decision line as the row above; its printout gives absolute paths and its JSON `scratchpad/…` ones (as the copies here do), and the parent comparison's reference line reads "(as stance4 recorded it: no)" with a note, because the recomputed reference cell's `path` is `scratchpad/…` where stance4's record holds the absolute scratch path |
| The scorer at every bar (session scratch only) | in `scratchpad/foot/stance5/judge/`: `python3 -m pytest -p no:cacheprovider test_score_stance5.py` | 101 passed |
| `checkpoints.csv` from the judge files (session scratch only) | `python3 -I -B checkpoints_csv.py <judged dir> out.csv --check <score_final.json>` | `25 cells agree with score_final.json`; `out.csv` identical to `checkpoints.csv` |
| `panels_summary.csv` from the panel files (session scratch only) | `python3 -I -B panels_summary.py <panels dir> out.csv --judged <judged dir>` | model sha256 equal to the judge file's in every file, standard K2a and K2b equal in 18; `out.csv` identical |
| `extras.txt` (session scratch only) | from the directory holding the session's scratch as `scratchpad/`: `python3 -I -B <this folder>/extras.py` | identical to `extras.txt` |
| The notebook gates' checkpoints | compare each `notebook_gates/<run id>/gate_verdict.json`'s `checkpoint_sha256` with the `robust_best` row of `checkpoints.csv` | equal for all three |
| The certificates' checkpoints | compare each `certificates/<run id>/run_summary.json`'s `model_sha256` with the final row of `checkpoints.csv` | equal for both (`24a4fada…`, `45245c7c…`) |
| The criteria | `sha256sum ../criteria.txt` | `3b28963f644ffb765e666a8cf6ada5d31ab8a2769dd1a77ff53cbc1e0cf3cf16` |
| The files themselves | `sha256sum -c SHA256SUMS` in `../../` | all files OK (the count is in the folder README) |
