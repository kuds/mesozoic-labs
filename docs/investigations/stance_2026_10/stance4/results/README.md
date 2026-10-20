# T. rex stance4 screen results (judged 2026-10-20)

Evidence for §19 of
[TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](../../../TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md):
the outcome of decision D-D39, the stance4 screen that the record's §18
pre-registers: three 12M T. rex stance runs of arm LTS (the stance3 L+T recipe
plus the stance stagger penalty, weight 0.30, tolerance 0.045 m), from scratch
on physics r9, `stance4_LTS_s52`, `stance4_LTS_s60` and `stance4_LTS_s42`,
from the branch `claude/screen-trex-r9-armlt-stagger` at `c1e146a8` (never
merged, no pull request), judged by the criteria fixed before training
([`../criteria.txt`](../criteria.txt), sha256 `978a93da…`) against stance3
Runs 2 and 3 as the control, not retrained. These are copies of the files that
decided or document each verdict, taken from the session's scratch tree on
2026-10-20 before it is lost. A screen certifies nothing: the verdicts are the
criteria's, not a gate's, and nothing here certifies a run, writes a result
bundle or promotes anything; `main` and its digests are unchanged.

The judge and scorer stay in the session's scratch and are not copied
(`scratchpad/foot/stance4/judge/`: `judge_stance4.py`, sha256 `3c8c0759…`, the
value each judge file records; `score_stance4.py`, `0d35e540…`, which loads the
scratch original of `../../stance3/judge/score_stance3.py`, sha256 `2c3879d3…`,
unchanged and extends only its judge-identity constants). So do the per-episode
judge files (51 for the three runs, about 15 MB together), the diagnostic
panels' files (30, about 7.8 MB), the rollouts and the runs' ledgers; the
checkpoints are on Drive under
`MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/stance4_*/` (each
judged checkpoint's sha256 is in `checkpoints.csv`).

**Outcome: all three runs fail their fixed criteria; H = 0 and K2p = 0; the
decision table's line is 5, final.** The stagger penalty is not adopted. Fails
stay fails, and no run was stopped or changed on its readouts.

| Run | Run id | K1 gate held | K2 fore-aft held | K3 robustness | K4 roll | Verdict |
|---|---|---|---|---|---|---|
| 1 | `stance4_LTS_s52` (LTS, seed 52) | FAIL: v2 FAIL at 6,291,456; five J checkpoints below 33/40 clean (28, 28, 21, 23 and 23 from 4.2M to 10.5M) | FAIL: K2a 12, 12, 19, 17 and 17 from 4.2M to 10.5M; K2b 16 to 21 there and 4 at the final | PASS | PASS | does not hold |
| 2 | `stance4_LTS_s60` (LTS, seed 60) | FAIL: v2 FAIL at 6,291,456; five J checkpoints below 33/40 clean (29, 29, 30, 27 and 30) | FAIL: K2a 11, 11, 10, 13 and 10 from 4.2M to 10.5M; K2b 16, 13, 16, 13 and 5 | PASS | PASS | does not hold |
| 3 | `stance4_LTS_s42` (LTS, seed 42) | FAIL: v2 FAIL at 6,291,456 and the final; all six J checkpoints below 33/40 clean (0, 3, 0, 3, 17 and 14) | FAIL: K2a 4 at 6.3M and 5 at the final; K2b 6, 10, 10 and 5 at 5.2M, 6.3M, 10.5M and the final | FAIL: joint-space noise falls 8 of 20 at the final (bar 2) | FAIL: roll rail 4 of 40 at the final (bar 3) | does not hold |

K2c passes on all three (ranges 6.6, 15.0 and 7.4 mm against 30). The credit
rule, against the control's passes recorded in stance3's `score_final.json`:
K1 0/3 against 1/2, K2 0/3 against 0/2, K3 2/3 against 2/2 and K4 2/3 against
2/2, so (a) at least two runs holding: no, and (b): fails. Line 5 of the
decision table reads "H == 0 and K2p <= 1: not adopted; the stagger penalty does
not hold fore-aft from scratch; the next round tests the per-foot CoP band term
(new contact-reward code) or the v6b ankle spring (stance3 line 3b)". The
control's verdicts reproduce from its judge files (stance3 Run 2 fails K1 and
K2, Run 3 fails K2).

The best stagger final by stance3's rule (most clean panel episodes, ties to
fewer joint-space plus own-sigma noise falls) is Run 1's, s52: 39/40 clean, 0
noise falls, one episode short of stance3 Run 2's final (40/40, 0), which stays
the walking pilot 2 parent the criteria offer. The maintainer's decisions of
2026-10-20 that follow from the outcome are logged, in short, in the last lines
of `run_log.txt` (those beginning "MAINTAINER DECISION"); the stance record's
§20 records them and names the source of each detail those lines leave out.

The runtime, measured: the `gpu` block of each run's
`01_stance/stage_config.json` reads an NVIDIA L4 (23.7 GB, CUDA 13.0), as the
run log records at each run's start and as re-read for this record; the
maintainer confirmed the L4 for all three (the run log's last line). That file
records only the session that wrote it. The `progress.json` files of s60 and s42
record one session each, from 00:02 and 00:04 UTC to 12:10 and 12:08 UTC
(`session_started_at` and `wall_clock_s`), so that those two ran on the L4
throughout is inferred; s52's `progress.json` was not fetched (its status, ended
at 12,001,280 steps, was read from Drive's metadata, as the run log says).

## What changed in the copies

[`../../SOURCES.txt`](../../SOURCES.txt) lists every file here but this README
with the scratch file it came from, that file's sha256, and what the copy
changed; [`../../SHA256SUMS`](../../SHA256SUMS) pins them. The parent folder's
**prefix** (the scratch root becomes `scratchpad`, the session directory above
it `session`) and **whitespace** changes apply as its
[README](../../README.md) defines them (here **whitespace** adds a final
newline to `scores/score_final.json` and removes a blank last line from
`split_form.md`), and **trim** and **redact** as
SOURCES.txt defines them, nothing else:

- **trim:** each `notebook_gates/<run id>/gate_verdict.json` keeps the gate
  verdict's identity and verdict fields (the schema, species and stage, the gate
  kind and its sha256, the task sha256, the checkpoint and its and the
  normaliser's sha256, the time it was judged, the verdict and its failures)
  and, of its `stage_result`, the timesteps, the best evaluation's step, the
  gate's verdict and failures, and the clean-seed count, seed count, fraction
  and LCB; it drops the gate's thresholds, the plant identity, the reward,
  length, distance and velocity statistics, the timing and the Drive paths of
  the model and its normaliser.
- **redact:** in `run_log.txt`, each of the session's tool identifiers is
  replaced by `[id removed]`: 6 workflow run ids (3 distinct; a bare one
  becomes `workflow [id removed]`), one scheduled-task id, and the report
  document's artifact link (now `claude.ai artifact [id removed]`); and the
  assistant's name in the name of a GitHub check by `[name removed]`. Nothing
  else in the log changes.

`checkpoints.csv`, `checkpoints_csv.py`, `panels_summary.csv`,
`panels_summary.py` and `extras.txt` (the printout of `extras.py`) were written
in the scratch for this record on 2026-10-20 and are copied unchanged. No copy
holds a model identifier, an email address or a commit trailer.

## Contents

**`run_log.txt`: the stance4 log, as written** from 2026-10-19 about 14:40 UTC
(the maintainer's decision of D-D39) to the maintainer's decisions of 2026-10-20
after 15:30 UTC. The build and the criteria's re-fix for the L4; each run's
start, read from Drive (provenance, the zero-action baseline, the
`stage_config.json`, its `gpu` block); per checkpoint the fetch (each
VecNormalize file transcribed twice and compared) and the judges' readings; the
pre-registered saturation alarm; the replicate study (`../night/`) and the
CoP-band continuation (`../night2/`); the finals, the notebook's gates, the
verdict of record and the report-only extras; the verification of the session's
human-facing report (a document outside the repository, not copied; 25
statements corrected there); and the maintainer's decisions. Before the verdict
of record its decision lines are readings as the files then stood. Its "first
2.10M" for s42's saturation names the first evaluation with 5 or more of 10
saturated episodes; the alarm (two such evaluations in a row) first fires at
2,228,224 (`scores/`). The s42 files record only the aggregate
`max_actuator_saturation_fraction` rail, not which actuator saturated.

**`scores/`: the scorer's printout and JSON, the verdict of record**
(`score_stance4.py --runs-ended`, written at 14:08 UTC): K1–K4 per run with
every cell, H, K2p, the credit rule, the decision line (`status` decided,
final), the parent comparison inputs, the control's cross-check against
stance3's `score_final.json` (sha256 pinned), and, report only, the
1,048,576-step checkpoints, the S0–S3 selection comparison, K5 (no flag on any
run), the late-fall fraction (above 5% only in the first 262,144-step bin,
48.7–50.2%) and the saturation alarm (s42: 70 of 91 evaluations with 5 or more
saturated episodes, the first alarm at 2,228,224; s52 and s60: none). s52's
`robust_best` and `best` were judged after it was written (14:18 and 14:13 UTC),
so it lists them as not judged; both are 40/40 clean in `checkpoints.csv`.

**`checkpoints.csv`: one row per judge file (51 rows, 17 per run).** The columns
of stance3's (its [README](../../stance3/results/README.md) names them), plus
`common_noise_falls`. Roles: `J`, the six judged checkpoints;
`report (1,048,576-step)`, the five other 1,048,576-step checkpoints;
`selection S0` and `S1`, `robust_best` and `best` (917,504 and 524,288 steps in
s52 and s42; byte-identical at 2,752,512 in s60); the fresh panel 7042–7081, the
32-seed forward-start tail panel and the 40-seed common-noise panel at the
final. S3, the late-stable pick, is s42's 8,388,608 J row; s52 and s60 have
none. Only the J rows and the noise counts at 6,291,456 and the final enter a
criterion; every other row is report only. `checkpoints_csv.py` wrote it: it
recomputes every number with `panel_counts` from
`../../stance3/judge/score_stance3.py` (the function `score_stance4.py` loads
unchanged) on the per-episode rows, and with `--check` stops on any difference
from the scorer's JSON (40 cells agree: all 42 periodic, final and selection
cells but s52's `robust_best` and `best`, which the JSON does not have).

**`extras.txt` and `extras.py`: the report-only extras on each final.** The
fresh panel: s52 PASS 39/40 (0 falls), s60 PASS 38/40 (1 fall), s42 FAIL 14/40
(1 fall); the forward-start tail panel: 2, 6 and 5 falls of 32 (the statue 0);
the common-noise panel: 5, 1 and 30 falls of 40, with no control reference
measured. None enters a verdict.

**`notebook_gates/`: the notebook's own end-of-stage gate,
`stance_quality_heading/v1`, report only** (trimmed `gate_verdict.json`, as
fetched from each run's Drive folder): each judged the run's `robust_best`
checkpoint at six headings. s52 PASS, 40/40 seeds clean (LCB 0.928, at
917,504); s60 FAIL, 33/40 (0.696, at 2,752,512); s42 PASS, 38/40 (0.851, at
917,504). Each `checkpoint_sha256` equals the `robust_best` row's in
`checkpoints.csv` (measured). These are not the screen's verdicts: the gate
judged an early checkpoint, not the finals the criteria judge. The runs'
`progress.json` files are not copied.

**`panels_summary.csv` and `panels_summary.py`: the diagnostic panels, report
only.** One row per variant (109 rows): the three stance4 runs and the two
controls at each of the six J checkpoints, under the judge's panel with the legs
mirrored at reset (`mirror_legs`), the prey straight ahead (`prey_ahead`), both
(`mirror_prey_ahead`) and, at every J checkpoint of the stance4 runs and at
8,388,608 of control Run 3, no change (`standard`): the clean count, K2a, K2b,
falls, the window stagger sd (mm), the per-foot CoP split's sd, the common CoP
and the lowest foot CoP (span half-lengths), copied from `panels4.py`'s files.
`panels_summary.py` checks each file's model sha256 against the judge file of
the same checkpoint and the standard variant's K2a and K2b against the judge's
(all equal; 19 standard rows, 18 for the stance4 runs and 1 for the control).
Read across (measured): at 4.2M the mirrored, prey-ahead panel has K2a 0, 1 and
0 (split sd 0.16, 0.07 and 0.03) where the standard panel has 12, 11 and 0; at
the finals it has K2a 0, 3 and 2 (split sd 0.20, 0.27 and 0.32). The standard
window stagger sd of s60 stays at 1.8–5.5 mm to the final; s52's is 4.8 mm at
4.2M and 7.3–12.7 mm from 5.2M to 10.5M (6.4 at the final); s42's is 0.9–4.6 mm
to 10.5M and 19.5 mm at the final.

**`split_form.md`: where the left-right split went, report only** (written at
06:01 UTC on 2026-10-20 from the stored judge files, panels and the controls'
rollouts, when only 4,194,304 had been judged for stance4; its tables mark the
later J checkpoints "not yet judged at writing"). Measured at 4.19M: the stance4
feet were nearly in line (window stagger sd 4.8, 2.3 and 1.2 mm for s52, s60 and
s42, against the zero-action statue's 2.5 and control Run 2's 11.9), while the
per-foot CoP split stayed (sd 0.99 and 1.09 for s52 and s60, against Run 2's
1.05; s42 0.48): per mm of stagger, 3.2, 7.3 and 6.4 times what the statue's
flat-sole geometry gives, a per-foot loading split in s52 and s60. The drivers
are the controls' (the spawn's leg asymmetry and the prey bearing; a two-factor
fit explains R2 0.89–0.91 of the stance4 split). Its implication for the next
term is marked inferred. Its "the pre-registered alarm from 2.10M" names the
first flagged evaluation, as in the run log. The panels above show that the
"nearly in line" reading holds at 4.19M and not for s52 later.

## How to re-check

Run on 2026-10-20 (Python 3.13, `-I -B`). The rows that need the judge files
run only where the session's scratch is.

| Check | Command | Result |
|---|---|---|
| The verdicts, H, K2p, the credit rule, the decision line and the parent inputs from the judge files (session scratch only) | in the session's scratch: `python3 -I -B scratchpad/foot/stance4/judge/score_stance4.py --runs-ended --json score.json` (its default paths are the scratch's) | identical to `scores/score_final.txt` and `.json` apart from the path of the JSON it names and s52's two selection cells, judged after the record |
| `checkpoints.csv` from the judge files (session scratch only) | `python3 -I -B checkpoints_csv.py <judged dir> out.csv --check <score_final.json>` | `40 cells agree with score_final.json`; `out.csv` identical to `checkpoints.csv` |
| `panels_summary.csv` from the panel files (session scratch only) | `python3 -I -B panels_summary.py <panels dir> out.csv --judged <stance4 judged dir> --control-judged <stance3 judged dir>` | model sha256 equal to the judge file's in every file, standard K2a and K2b equal in 19; `out.csv` identical |
| `extras.txt` (session scratch only) | from a directory holding the session's scratch as `scratchpad/`: `python3 -I -B <this folder>/extras.py` | identical to `extras.txt` |
| The notebook gates' checkpoints | compare each `notebook_gates/<run id>/gate_verdict.json`'s `checkpoint_sha256` with the `robust_best` row of `checkpoints.csv` | equal for all three |
| The files themselves | `sha256sum -c SHA256SUMS` in `../../` | all 344 files OK |
