# T. rex stance3 screen results (judged 2026-10-18 to 2026-10-19)

Evidence for §17 of
[TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](../../../TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md):
the judging of the three full T. rex stance screen runs of decision D-D37
(`stance3_armL_s60`, `stance3_armLT_s60` and `stance3_armLT_s52`; the record's
§16) by the criteria fixed before training ([`../criteria.txt`](../criteria.txt),
sha256 `1ce910dc…`), with the frozen judge and scorer in `../judge/`, and the
report-only readouts and the overnight scratch experiment that followed. These
are copies of the files that decided or document each verdict, taken from the
session's scratch tree on 2026-10-19 before it is lost. A screen certifies
nothing: the verdicts are the criteria's, not a gate's, and nothing here
certifies a run, writes a result bundle or promotes anything; `main` and its
digests are unchanged. The per-episode judge files (37 of them, about 0.34 MB
each), the rollouts and the traces stay in the session's scratch, and the
checkpoints on Drive under
`MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/stance3_*/`
(each judged checkpoint's sha256 is in `checkpoints.csv`).

**Outcome: all three runs fail their fixed criteria; the decision table's line
is 3b.** Fails stay fails, and no run was stopped or changed on its readouts.

| Run | Run id | K1 gate held | K2 fore-aft held | K3 robustness | K4 roll | Verdict |
|---|---|---|---|---|---|---|
| 1 | `stance3_armL_s60` (arm L, seed 60) | PASS | FAIL: K2c range 33.9 mm (bar 30) | FAIL: joint-space noise falls 7 and 11 of 20 at 6,291,456 and the final (bar 2) | PASS | does not hold |
| 2 | `stance3_armLT_s60` (L+T, seed 60) | FAIL: v2 FAIL at 6,291,456; three judged checkpoints below 37/40 clean (0, 0 and 36 at 4.2M, 5.2M and 6.3M), two below 33 | FAIL: K2a 10 and 6 at 4.2M and 5.2M; K2b 14, 10 and 6 at 4.2M to 6.3M | PASS | PASS | does not hold |
| 3 | `stance3_armLT_s52` (L+T, seed 52) | PASS | FAIL: K2a 6 at 8.4M, K2b 4 at the final, K2c range 30.6 mm | PASS | PASS | does not hold |

Line 3b of the decision table reads "L+T fails K2 on a seed and Run 1 also
fails K2: a joint-space price does not hold the stance; the next round tests a
fore-aft term that needs code or the v6b ankle spring". What the next round is
stays the maintainer's decision. The best final checkpoint by the fixed rule
(most clean panel episodes, ties to fewer joint-space plus own-sigma noise
falls) is Run 2's final: 40/40 clean, 0 noise falls (runner-up Run 3's final,
39/40, 0). `criteria.txt` offers it as the stance parent of the parked walking
pilot 2; whether pilot 2 runs, and from which parent, is the maintainer's
decision too.

## What changed in the copies

[`../../SOURCES.txt`](../../SOURCES.txt) lists every file here but this README
with the scratch file it came from, that file's sha256, and what the copy
changed; [`../../SHA256SUMS`](../../SHA256SUMS) pins them. The parent folder's
**prefix** (the scratch root becomes `scratchpad`, the session directory above
it `session`) and **whitespace** changes apply as its
[README](../../README.md) defines them, and four more, nothing else:

- **excerpt:** `night_scratch/spec_section7.md` is §7 of the overnight
  experiment's design (`scratchpad/foot/stance3/night/design/spec.md`), from
  its heading to the line before §8, verbatim. The rest of the design (the
  runs, the candidate terms, the continuation mechanics, the launch gates, the
  no-change control and the risks) stays in the scratch.
- **trim:** each `certificates/<run id>/run_summary.json` keeps the
  certificate's verdict fields (the panel seeds, the arm's leg and lean keys,
  the checkpoint path and its and the normaliser's sha256, the verdict and its
  failures, the clean-seed and seed counts, their LCB, the task and gate
  sha256, and per heading its clean count, LCB and full-horizon fraction) and
  drops the per-heading reward means, reward spreads and mean episode
  lengths, the timing, the paths of the clone, its `environments` module and
  the normaliser, and the paths of the files it wrote.
- **extract:** `night_scratch/verify_result.json` is the value of the key
  `result.verify` of the night analysis workflow's output
  (`session/tasks/[id removed].output`; the task id is removed), written
  with an indent of 1: the adversarial verifier's verdict on each claim of
  the interpretation, and what a report may and may not claim.
- **redact:** in `run_log.txt` each of the session's tool identifiers is
  replaced by `[id removed]`: 12 workflow run ids (11 distinct; a bare one
  becomes `workflow [id removed]`) and the report document's id prefix.
  Nothing else in the log changes.

`checkpoints.csv` and `checkpoints_csv.py` were written in the scratch for
this record on 2026-10-19 and are copied unchanged. No copy held a model
identifier, an email address or a commit trailer; `run_log.txt` is kept as
written, apart from the ids removed.

## Contents

**`run_log.txt`: the judging log, as written** from 2026-10-18 20:45 to
2026-10-19 08:20 UTC. Per judged checkpoint: the training status, the fetch
from Drive (each VecNormalize file transcribed twice and compared, each zip
checked with `testzip`; sha256 prefixes), the judge runs (every binding
matched: plant r9, seed, task, symmetry map, the run's `stage_config.json`;
`hard_failures` empty) and the scored cells with the decision line as it
stood. Then the report-only readouts at 4,194,304, the night diagnosis, design
and build, the final verdict (06:00–06:14 UTC), the extras, and the night
analysis with its verifier's corrections. Before the final, its decision
lines are provisional: they say which lines were still open. The session's
workflow run ids and the report document's id are removed (**redact**,
above).

**`scores/`: the frozen scorer's printouts** (`../judge/score_stance3.py`).

- `score_final2.txt` and `score_final2.json`: the verdict of record, written at
  08:17 UTC once the extras were judged: K1–K4 per run with every cell, the
  recipe verdicts, the decision-table line, the best final checkpoint, and,
  report only, K5 (no flag on any run), the late-fall fraction per
  262,144-step bin (above 5% only in the first bin, 49–53%, and Run 1's
  7,340,032 bin, 5.1%) and the S0–S3 selection comparison.
- `score_final.txt`: the same verdict at 06:14 UTC, before the 3,145,728,
  `robust_best`, `best` and S3 checkpoints were judged; its verdict,
  decision-table and best-checkpoint lines equal `score_final2.txt`'s.
- `score_4194304.txt` … `score_10485760.txt`: the printouts at each judged
  checkpoint as it arrived. The scorer fails every criterion whose judge file
  does not exist yet, so their verdict and decision lines are provisional by
  construction; the run log says what each showed.

**`checkpoints.csv`: one row per judge file (37 rows).** Columns: the run,
arm and seed; the slot and its role (`J`, the six judged checkpoints;
`report (judged like J)`, 3,145,728; `selection S0` and `S1`, `robust_best`
and `best`, byte-identical in every run, saved at 2,752,512 in Run 1 and
917,504 in Runs 2 and 3; `selection S3`, Run 2's 11,534,336, the rule's
late-stable pick, which for Runs 1 and 3 is a J checkpoint; `report (night
control at E)`, the Colab checkpoints at 9,961,472 judged as the overnight
experiment's no-change controls; the fresh panel 7042–7081 and the 32-seed
forward-start tail panel at the final); the checkpoint and its step; the
`stance_quality/v2` verdict, clean count, LCB, full-horizon and hop-or-fall
counts; K2a, K2b; the both-feet whole-foot centre of pressure `x_c` (panel
median over full-horizon episodes), the statue's on the same seeds and the
offset in mm; the roll-rail count; the joint-space and own-sigma noise falls
of 20; for a tail panel its falls and the statue's; and the checkpoint's
sha256. Only the J rows and the noise counts at 6,291,456 and the final enter
a criterion; every other row is report only, and K2a and K2b on the fresh
panel are counts, not criteria. `checkpoints_csv.py` wrote it: it recomputes
every number with the frozen scorer's own `panel_counts` from the per-episode
rows, and with `--check` stops on any difference from the scorer's output (28
cells agree).

**`certificates/`: the stage's own six-heading certificate
(`stance_quality_heading/v1`) rolled offline on the three finals, report
only.** `run_cert_clone.py` (adapted from
`../../offline_certificates/scripts/run_cert.py`) calls
`build_stance_heading_report`, the function the notebook's end-of-stage
certificate calls, in each run's own branch clone with that branch's stance
stage configuration, so each certificate rolls the run's own task (`cd963518…`
arm L, `d4a3b8f8…` L+T). All three PASS: Run 1 38/40 seeds clean at all six
headings (LCB 0.851), Run 2 40/40 (0.928, every heading 40/40), Run 3 39/40
(0.887). **None is gate evidence**: the screen configurations are not
committed, so the official judge refuses their task fingerprints. Per run:
`run_summary.json` (trimmed) and `stance_heading_report.txt` (the certificate
as printed). The per-episode `stance_heading_report.json` and panel CSV stay
in the scratch.

**`readouts/`: report-only readouts of the Colab checkpoints** (rollouts
that first reproduce the judge files exactly; never in a verdict).

- `r4194304_saturation_notes.md`: Run 2's saturation at 4,194,304. One
  actuator, `tail_3_pitch_act`, is pinned at +1 in 40/40 episodes while the
  joint sits mid-range at 0.26 of its force limit; the gate's threshold is
  0.99 on clipped actions (the env's 0.9 is the reward ramp). It also
  measures Run 2's left-right fore-aft foot split, left from the spawn
  stagger, and each run's per-actuator `exp(log_std)`.
  `r4194304_k2_recount.txt` is the independent recount of K2a and K2b at
  4,194,304 (Run 2's 10 and 14 confirmed).
- `r4194304_posture_notes.md`: posture at 4,194,304: Run 1's backward drift
  is carried by the ankle (about −10.6 mm of the centre of mass per degree),
  and its two falls cross the heel line before any leg joint leaves its
  envelope by 5°; Run 3's two falls are lateral.
- `split_notes.md`: Run 3's K2 failure at 8,388,608: a fore-aft stagger of
  the feet with the forward foot on its heel, the hip commands pushing into
  the twist the statue resists, a turn toward the prey at that checkpoint
  only, a common backward ankle opening, and counterfactual spawns.
- `drift_notes.md`: the whole-body fore-aft drift in the eval ledgers (arm
  L's three 50–65 mm ankle-carried swings against L+T's bounded drift on the
  same seed) and how each candidate term would price it.

Their paths name the scratch: `SP` is `scratchpad`, `S3` is
`scratchpad/foot/stance3`, and relative paths are below each file's own
scratch folder (`diag/r4194304/<saturation or posture>/`,
`night/diag_split/`, `night/diag_drift/`). Their tables, scripts and rollouts
stay there.

**`night_scratch/`: the overnight scratch experiment of 2026-10-19 (scratch
only, never merged, report only).** Three local CPU continuations from the
Colab 8,388,608 checkpoints, each with one candidate reward term on, judged at
E = 9,961,472 against the Colab runs' own checkpoints at E as the no-change
control: A, Run 3 plus a foot-stagger penalty; B, Run 3 plus the ankle
anchor; C, Run 1 plus the stagger penalty. The code was a scratch clone
(`80328abf`, branch `scratch/stance3-night`, never pushed); its judge files
are relaxed-binding, so the scorer refuses them and they never enter K1–K4.

- `spec_section7.md`: the readouts, the expectations with their supporting
  and refuting lines, and the rules for reading the three together, fixed
  before launch.
- `readouts_night.md` and `readouts_control.md`: the readouts R1–R6 of the
  three arms and of the matched controls at E.
- `interpretation.md`: the reading by those rules, as written. Read it with
  `verify_result.json`, which corrects it: no rule's condition holds cleanly,
  attribution of A's improvement to its term is not established (neither
  shown nor excluded), and "the control did everything A did", "the unchanged
  recipe fixed Run 3" and "the anchor made it worse than no change" may not be
  claimed.

Its paths: `S3` is `scratchpad/foot/stance3`, `C` is `S3/night/cont`, `K` is
`C/control` and `D` is `S3/night/design`.

## How to re-check

Run on 2026-10-19 (Python 3.13; `PYTHONDONTWRITEBYTECODE=1`). The rows that
need the judge files run only where the session's scratch is.

| Check | Command | Result |
|---|---|---|
| The verdicts, the decision line and the best final checkpoint from the judge files (session scratch only) | from a directory holding the session's scratch as `scratchpad/`: `python3 -I <this folder>/../judge/score_stance3.py scratchpad/foot/stance3/judged --runs-root scratchpad/foot/stance3/runs --json score.json` | identical to `scores/score_final2.txt` and `score_final2.json` apart from the paths they name |
| The scorer at every bar (session scratch only) | from the same directory: `python3 -I <this folder>/../judge/test_score_stance3.py` | 80/80 cases as expected |
| `checkpoints.csv` from the judge files (session scratch only) | `python3 -I checkpoints_csv.py <judged dir> out.csv --check <score_final2.json>` | `28 cells agree with score_final2.json`; `out.csv` identical to `checkpoints.csv` |
| The files themselves | `sha256sum -c SHA256SUMS` in `../../` | all 319 files OK |
