# T. rex stance5 replication results (judged 2026-10-24)

Evidence for §23 of
[TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](../../../../TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md):
the outcome of decision D-D41, the replication of the stance5 per-foot
centre-of-pressure band recipe that the record's §22 decides: two 12M T. rex
stance runs of arm LTSB, unchanged (stance4's arm LTS plus the band at weight
0.30), from scratch on physics r9, `stance5_LTSB_s70` (Run 4) and
`stance5_LTSB_s80` (Run 5), from the branch `claude/screen-trex-r9-armlts-copband`
at `cf80ecb1` (never merged, no pull request), judged by the criteria fixed
before training ([`../criteria.txt`](../criteria.txt), sha256 `a38536de…`)
and read with the three stance5 runs as recorded, under the replication's
reading rule (the band is adopted only if both new runs hold). These are copies
of the files that decided or document each verdict, taken from the session's
scratch tree on 2026-10-24. A screen certifies nothing: the verdicts are the
criteria's, not a gate's, and nothing here certifies a run, writes a result
bundle or promotes anything; `main` and its digests are unchanged.

The criteria, the Colab steps, the joint-noise sigma sidecar and frozen copies
of the judge and scorer are in [`..`](../) (`../judge/` holds
`judge_stance5r.py`, `score_stance5r.py`, the scorer's tests and
`judge_diff.txt`, the judge's whole diff against stance5's judge; the judge's
copy carries the prefix and checkout changes, the scorer's and the sidecar's the
prefix change, so their sha256 is not the `16d39f7d…`, `55c37ba7…` and
`89778b3f…` the judge files, the criteria and the Colab steps record). The
per-episode judge files, the certificates' per-episode files, the runs' ledgers
and records, and the start-up checks' copies stay in the session's scratch; the checkpoints are on Drive under
`MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/stance5_LTSB_s70/`
and `.../stance5_LTSB_s80/`.

**Outcome: Run 4 (s70) holds; Run 5 (s80) does not (K1 only); H = 2 of 5; the
decision table's line is 2, final.** The band is not adopted: "Exactly one of
s70 and s80 HOLDS (H = 2 of 5): not adopted." The maintainer chooses the next
round; stance5 decision line 4's three options are the menu (the tail
saturation habit, the dose, the leg exploration-noise floor). Fails stay fails,
and no run was stopped or changed on its readouts or on the saturation alarm.

| Run | Run id | K1 gate held | K2 fore-aft held | K3 robustness | K4 roll | Verdict |
|---|---|---|---|---|---|---|
| 4 | `stance5_LTSB_s70` (LTSB, seed 70) | PASS: 40, 40, 40, 39, 40 and 40 of 40 clean | PASS: K2a and K2b 0 at every J but 8,388,608 (1 each); K2c range 13.7 mm | PASS: joint-space noise falls 0 at 6,291,456 and 1 at the final; own-sigma 0 and 0 | PASS: 0 and 0 | HOLDS; fore-aft holds |
| 5 | `stance5_LTSB_s80` (LTSB, seed 80) | FAIL: v2 FAIL at 6,291,456; 40, 4, 14, 26, 39 and 38 of 40 clean, the three below 33 all on the actuator-saturation rail (36, 25 and 14 episodes) | PASS: K2a and K2b 0 at every J but the final (1 each); range 10.4 mm | PASS: joint-space noise falls 1 and 2 of 20 (bar 2); own-sigma 0 and 0 | PASS: 1 and 1 | does not hold (K1 only) |

The three stance5 runs count as recorded (s52 holds, s60 and s42 do not): the
scorer reproduces their recorded verdicts, J cells, K1 failure modes, task and
judge from their judge files, against stance5's score of record
(`../../results/scores/`; the scorer pins its scratch original's sha256), and the control's verdicts reproduce stance4's score of record.
Report only (never in the decision): the K3 pass fraction, the five band runs
against the control, is 3 of 5 against 2 of 3, and per test K1 3/5 against 0/3,
K2 5/5 against 0/3, K3 3/5 against 2/3 and K4 5/5 against 2/3. Fore-aft holds on
s52, s60 and s70. The parent comparison (report only, stance3's rule; the new
runs' noise falls use the sidecar's mean sigma): s52's final and stance3 Run
2's final tie (40/40, 0 noise falls); s70's final is next (40/40, 1); the rule
names both tied finals and does not choose. Walking pilot 2's parent stays
stance3 Run 2's final (the maintainer's decision (3) of 2026-10-22, 15:08 UTC).

The runtime, measured: the `gpu` block of each run's
`01_stance/stage_config.json` reads an NVIDIA L4 (23.7 GB, CUDA 13.0), with
`git_commit` `cf80ecb1` and the run's seed. Each run's `progress.json` records
one session: `session_started_at` 19:33:38 (s70) and 19:34:03 UTC (s80) on
2026-10-22, status `ended` at 12,001,280 steps, written at 09:15:41 and
09:20:53 UTC on 2026-10-23, at 243.3 and 241.9 steps per second (the score's
throughput lines), with no warnings; so that each ran on the L4 throughout is
inferred. The run log's start-up check (19:41 UTC on 2026-10-22, two readers
per run, byte-identical) records each run's `stage_config.json` equal to stance5
s52's but for the seed and the label, MuJoCo 3.10.0 as on stance5, and a
zero-action reward of 4367.8, as on stance5.

## What changed in the copies

[`../../../SOURCES.txt`](../../../SOURCES.txt) lists every file here but this
README with the scratch file it came from, that file's sha256, and what the copy
changed; [`../../../SHA256SUMS`](../../../SHA256SUMS) pins them. The folder's
**prefix**, **whitespace**, **gzip**, **trim**, **redact**, **checkout** and
**lines** changes apply as SOURCES.txt defines them, nothing else:

- **gzip:** `scores/score_final.json.gz` is the scorer's JSON with the prefix
  and whitespace changes, compressed (553,362 bytes uncompressed, above the
  repository's 500 KB limit for an added file); `gunzip` restores it.
- **whitespace** adds a final newline to the JSON copies (the score, the two
  certificates' `run_summary.json`, the three `saturation/` rollouts and the
  watch's state file).
- **trim:** each `notebook_gates/<run id>/gate_verdict.json` keeps the keys
  stance4's and stance5's do, and each `certificates/<run id>/run_summary.json`
  the keys stance3's and stance5's do (`../../results/README.md` names them).
- **lines:** `run_log.txt` is lines 396 to 504 of its source, the session's run
  log, whose lines 1 to 207 are `../../../stance4/results/run_log.txt` and lines
  208 to 395 `../../results/run_log.txt`; the source had 504 lines when copied.
- **redact:** in `run_log.txt`, each of the session's tool identifiers is
  replaced by `[id removed]`: 49 workflow run ids (28 distinct; a bare one
  becomes `workflow [id removed]`), 15 background-job ids of the session's
  queue workers (13 distinct), and 6 shortened Drive folder ids (eight
  characters and `...`). Nothing else in the log changes; GitHub's run and job
  ids are kept, as in the earlier copies.
- **checkout:** the session's checkout directory is replaced by `checkout` in
  `../judge/judge_stance5r.py` (its `MAIN_CHECKOUT` constant) and in
  `../judge/judge_diff.txt`.

No copy holds a model identifier, an email address or a commit trailer.

## Contents

**`run_log.txt`: the replication's log, as written** from the entry timed 17:46
UTC on 2026-10-22 (the package made final) to the entry timed 15:27 UTC on
2026-10-24. The package made final and re-pinned after the log's own time
correction (17:47 and 18:00 UTC; criteria `a38536de…`, scorer `55c37ba7…`,
judge `16d39f7d…` and sidecar `89778b3f…`); #634's opening, CI and merge
(18:29, 19:22, 19:36 and 19:42 UTC); the maintainer's report that seeds 70 and
80 had started (19:33) and the start-up checks (19:41); each check-in's ledger
reading and fetch, and the judges' readings; the saturation alarm on s80 and
its naming; the K3 diagnosis and the scratch tail experiment reported (21:04);
the maintainer's OK for the K3 readouts (22:56), the readout tool and its
results; the training-history report's build; three container restarts and the
account's spend limit (about 05:30 UTC on 2026-10-23), after which the session
was idle until 12:06 UTC on 2026-10-24; the final fetch, the judging and the
verdict of record (13:04 UTC on 2026-10-24); the final K3 readouts and the
certificates (13:55); the handoff (13:59); and, in a new session, the restore
(15:14) and the maintainer's decisions of 15:20 UTC on 2026-10-24. Before the
verdict of record its decision lines are readings as the files then stood
("Interim scorer: decision incomplete").

**`scores/`: the scorer's printout and JSON, the verdict of record**
(`score_stance5r.py --runs-ended`; the run log's entry is timed 13:04 UTC on
2026-10-24; sha256 `9be2a231…` and `938da843…` as written): the replication
criteria's sha256 (matched), K1–K4 per run with every cell for the two new runs
and the three stance5 runs as recorded, the stance5 reproduction, the
replication rule and the decision line (`status` decided, line 2, runs ended),
the next-round inputs (s80's failing test, K1 failure modes and saturation
alarm), the report-only fractions, the drift-or-split classes (printed, not
read; all five not classed), the control's cross-check against stance4's
`score_final.json` and the parent comparison; and, report only, the
1,048,576-step checkpoints judged like J (1.0M, 2.1M and 3.1M: s70 39, 39 and 39
of 40 clean, one roll-rail episode each; s80 40, 40 and 40), K5 (no flag on
either run), the late-fall fraction (above 5% only in the first 262,144-step
bin: 55.9% for s70 and 48.5% for s80), the band readouts and the saturation
alarm. Not judged, and listed as missing: the 1,048,576-step checkpoints
7,340,032, 9,437,184 and 11,534,336 and the `best` and `robust_best` pairs (the
selection readouts S0, S1 and S3), which were not fetched, to save usage after
the account hit its spend limit (the run log, 12:06 UTC on 2026-10-24).

**`notebook_gates/`: the notebook's own end-of-stage gate,
`stance_quality_heading/v1`, report only** (trimmed `gate_verdict.json`, as
fetched from each run's Drive folder): each judged the run's `robust_best` at
six headings, and both PASSED: s70's at 3,407,872 steps, 40/40 seeds clean (LCB
0.928; judged at 09:46:21 UTC on 2026-10-23), and s80's at 393,216 steps, 38/40
(0.851; 09:51:20 UTC). These are not the screen's verdicts: the gate judged an
early checkpoint, not the finals the criteria judge, and the `robust_best` pairs
were not fetched, so their `checkpoint_sha256` is not checked against a judge
file here.

**`certificates/`: the stage's own six-heading certificate
(`stance_quality_heading/v1`) rolled offline on the s70 and s80 finals, report
only** (`cert5r_one.sh`, which runs stance3's runner from the scratch path that
`../../../stance3/results/certificates/run_cert_clone.py` was copied from, not
re-hashed here because the handoff did not carry it, in a clone of the screen
branch at `cf80ecb1`, so each certificate rolls the band-on task
`8c6998d2…`; from 13:14:55 to 13:34:47 and 13:34:47 to 13:54:57 UTC on
2026-10-24). s70's final PASSED, 40/40 seeds clean at every heading (LCB
0.928); s80's PASSED, 38/40 (0.851): seeds 3047 and 3081 were not clean at any
of the six headings (in the 0 deg cell, 3047 failed the sole-tilt bar, 2.288 deg
against 2, and 3081 fell at 306 steps). **Neither is
gate evidence**: the band and its configuration are not on `main`. Per run:
`run_summary.json` (trimmed) and `stance_heading_report.txt` (the certificate as
printed); each `model_sha256` equals the `model_sha256` of the run's
`judge_final.json` in the session's scratch (measured).

**`saturation/`: the saturation watch's files for s80, report only.**
`stance5_LTSB_s80_<step>.json` is the 20-episode rollout (seeds 3042–3061,
window steps 200–999) of the checkpoint at 4,718,592, 5,242,880 and 6,291,456
steps that names the actuator (`satroll5r.py`); `satwatch5r_state.json` is the
watch's state file (the alarm's step and the actuator it reported). The alarm
fired at 4,456,448, the first flagged evaluation (5 or more of 10 episodes on
the saturation rail) being 4,325,376; the ledger read 4, 3 and 5 of 10 clean at
4,325,376, 4,456,448 and 4,587,520, flagged on 6, 7 and 5 episodes, with no
falls and `actuator_force_frac_max` 0.29 to 0.31 (the run log, 00:53 UTC on
2026-10-23). In all, 37 of s80's 91 evaluations were flagged: 35 in five alarm
runs, 4,325,376–6,815,744 (20), 7,340,032–7,471,104 (2), 7,733,248–7,864,320
(2), 8,388,608–9,306,112 (8) and 9,699,328–9,961,472 (3), and 2 alone
(7,077,888 and 10,223,616; `scores/`). The rollouts name `tail_3_pitch_act` at
+1 (tail raised) at each checkpoint rolled: its command at 0.9 or more on 74%
of window steps at 4,718,592 (mean +0.91; 14 of 20 episodes over the gate's
bar), 92% at 5,242,880 (+0.97; 19 of 20) and 63% at 6,291,456 (+0.81; 12 of 20),
with no fall. That is the actuator and sign of stance5 s42, stance4 s42 and
stance3 Run 2. s70: no evaluation flagged (0 of 91).

## How to re-check

Run on 2026-10-24 in the session's scratch as restored from the handoff branch
(Python 3.13, `-I`, `PYTHONDONTWRITEBYTECODE=1`). The rows that need the judge
files run only where the session's scratch is.

| Check | Command | Result |
|---|---|---|
| The verdicts, H, the decision line and the parent comparison from the judge files (session scratch only) | `python3 -I scratchpad/foot/stance5/replication/judge/score_stance5r.py --runs-ended --json score.json` (its default paths are the scratch's) | the same verdicts, cells, H and decision line (2) as `scores/score_final.txt`; it differs where the handoff carried no file: the stance5 runs' and the control's late-fall lines ("no train_episodes.csv rows") and the parent comparison's stance3 Run 2 reference cell (not recomputed, so the top is s52 alone, with a note) |
| The scorer's tests (session scratch only) | in `scratchpad/foot/stance5/replication/judge/`: `python3 -m pytest -p no:cacheprovider test_score_stance5r.py` | 42 passed, 2 skipped (the validation judge files are not in the restored scratch), 1 failed (`test_parent_comparison_five_finals`: stance3 Run 2's final is not in the tie, as in the row above); the run log records 66 tests passing at 18:00 UTC on 2026-10-22, before the handoff |
| The certificates' checkpoints (session scratch only) | compare each `certificates/<run id>/run_summary.json`'s `model_sha256` with the scratch's `judged/<run id>/judge_final.json` | equal for both (`9f115034…`, `e7936691…`) |
| The criteria | `sha256sum ../criteria.txt` | `a38536de152e735947fd5322960f9e75b6520b59a5526a414afc0d0c4a0f3ef2` |
| The files themselves | `sha256sum -c SHA256SUMS` in `../../../` | all files OK (the count is in the folder README) |
