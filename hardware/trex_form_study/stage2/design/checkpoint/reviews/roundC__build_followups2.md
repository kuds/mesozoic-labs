<!-- Working paper, 16 Nov 2026: the builder's report on two code follow-ups (the monitor's refusal; the tail's amplitude band). Nothing here is a rule. -->

TWO CODE FOLLOW-UPS: REPORT (16 Nov 2026)

Simulation tooling only. No episode, fit or training was run beyond the packages' own tests and one kinematic playback of the look ladder (no physics). Nothing pushed, no history rewritten, main checkout untouched (`git status --porcelain` empty there).

Record-key hash line (main checkout, and the same in both worktrees):
`45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair).

## A. The trainer (`/home/user/mesozoic-wt/wpc-ct`, branch `wt/wpc-ct`)

Merge: the branch was an ancestor of the working branch, so the merge was a fast-forward to `ffecc2c`. There is no merge commit.

What changed:
- `designlab/stage2/monitor2.py`
  - New `MonitorRefusal` (a `MonitorError` with a `kind`), `EXIT_REFUSAL = 8`, `REFUSALS`, `refusal_path`, `RULES["refusal"]`.
  - `_sources` raises it when the sources differ (kind `sources`). `_checkpoint` raises it when the checkpoint is not intact (kind `checkpoint`). `run` raises it for another run's checkpoint (kind `other_run`).
  - `main` leaves with 8 on a refusal and writes the reason to `monitor/monitor_<n>.refused.json` (baseline: `r0l.refused.json`). A later call of the same check removes that note. Every other fault keeps exit code 1.
- `designlab/stage2/train2.py`
  - `_Run.check`: on the monitor's exit code 8 the attempt is closed with `exit_code` 8, `refused: true` and `refusal: {kind, reason}`. The trainer then raises `Halt` (exit code 5); `run.json` `halt` and `progress.json` `halt` quote the monitor's own reason.
  - The count of closed attempts without a record skips refused attempts.
  - A resume runs the check again. Nothing is read or decided at a refused check.
  - New `_Run._refusal` reads the note. `_Run._baseline`: a refused baseline halted before and still does; its halt now names the refusal.
- `clip2.py`: not changed.

Tests (each fails with its part of the change taken out; I checked three mutations and restored the files):
- `tests/test_stage2_monitor2.py::test_a_monitor_that_refuses_to_run_leaves_with_its_own_exit_code`: the three kinds through the real CLI.
- `tests/test_stage2_train2.py::test_a_monitor_that_refuses_to_run_is_no_attempt`:
  - a trainer process returns 5;
  - refused twice, the check is still not counted without a record;
  - on the next resume one crash is the first of the two attempts, then the record;
  - the run ends bit for bit as the uninterrupted one.
- `tests/test_stage2_train2.py::test_a_refusal_without_its_note_and_a_refused_baseline`.

Exact summary lines (through `bash tools/lowprio.sh`, the standing env line, exit 0 each):
- `tests/test_stage2_monitor2.py`: `23 passed in 20.84s`
- `tests/test_stage2_train2.py`: `48 passed, 3 skipped, 2 warnings in 203.03s (0:03:23)`

The 3 skipped are the slow S1 tests (`STAGE2_SLOW` not set).

`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 .../trex_form_study/designlab/stage2/monitor2.py   | 108 ++++++++++++++++---
 .../trex_form_study/designlab/stage2/train2.py     |  44 ++++++--
 .../trex_form_study/tests/test_stage2_monitor2.py  |  69 ++++++++++++-
 .../trex_form_study/tests/test_stage2_train2.py    | 114 ++++++++++++++++++++-
 4 files changed, 309 insertions(+), 26 deletions(-)
```
Commits, oldest first (no merge): `57b5bf3` (the monitor's exit code), `2fca27b` (the trainer halts and does not count). Head `2fca27b`.

## B. The look reading (`/home/user/mesozoic-wt/wpc-ck`, branch `wt/wpc-ck`)

Merge: `c9b0bed`, a merge commit (parents `9caa811`, `ffecc2c`). No conflict; the merged tree equalled the working branch.

What changed:
- `designlab/stage2/lookread.py`, `compare`: where the tail term is read, it now needs the phase inside its band and the robot's own fitted amplitude not under `TAIL_AMP_MIN_RAD` (0.02 rad).
- Where the reference's amplitude is under 0.02 rad nothing changes: not read, passes, labelled.
- `RULES["bands"]` says so. It quotes the ruling under its own mark (`LEAD_RULING`), between the plan's rule and the module's own reading.
- Both amplitudes and their ratio are still recorded. `compare` gains `tail_amp_rad` (the robot's), `tail_amp_side` and `tail_amp_min_rad`.

Tests in `tests/test_stage2_lookread.py`:
- `test_the_tail_term_fails_under_0_02_rad_of_own_amplitude_while_the_reference_is_read` (both sides of the limit).
- `test_the_tails_amplitude_limit_is_exact_and_bites_only_where_the_tail_is_read`.
- `test_a_tail_held_still_on_a_rungs_own_poses_fails_exactly_the_tail_term`.
- `test_every_rung_of_the_ladder_swings_its_tail_well_over_the_limit`.
- The ladder acceptance and the rule-text test are extended.
- `test_the_tails_amplitude_is_recorded_and_no_band_reads_it` is replaced: it pinned the old reading, with an own amplitude of 0.006 rad passing.

With the limit taken out, 4 tests fail.

Exact summary line (exit 0): `70 passed, 1 skipped in 9.77s`. The skip is the slow live-recorder test.

The ladder, by kinematic playback of each rung's own arrays (the test module's helpers, one low-priority process):
- The module's acceptance, the six V6 rungs against `stage2/look/metrics.json`: all six pass gate Z2 (b)'s limits, and each playback passes all four terms against itself.
- All 73 rungs: the tail term is read and passes against itself on 73 of 73.
- Smallest reference tail amplitude: 0.11998 rad fitted (60-N-T-10, v040, V0); 0.12 rad in `metrics.json`. On the V6 rungs: 0.124951 to 0.125268 rad. The limit is 0.02 rad.
- Seen beside it:
  - The 54 rungs V0 to V4L fail the head term against themselves. That term's code is untouched by this change.
  - The 60 cm body's fr05 V6 rung has a reference toe-off of 0.0187 rad, so its toe-off term is not read. This is for the gates package's list.

`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 .../trex_form_study/designlab/stage2/lookread.py   |  46 ++++--
 .../trex_form_study/tests/test_stage2_lookread.py  | 155 ++++++++++++++++++---
 2 files changed, 174 insertions(+), 27 deletions(-)
```
Commits, oldest first: `c9b0bed` (merge), `5c9fe63` (the tail's amplitude limit). Head `5c9fe63`.

## For the lead

Readings I built; each could be read otherwise.

1. **A, the refusal's note file.** The monitor writes `monitor_<n>.refused.json` (or `r0l.refused.json`) so the halt can quote its reason. The other option is no file: the halt then names only the exit code and the three possible causes (that is also the fallback when no note is found). I recommend keeping the file. It is a new file a job can write.
2. **A, "not intact".** I took it as: no checkpoint record, a record that cannot be read as JSON (before: a traceback and exit 1), or a file that is not the one its record names. A narrower reading would leave the first two at exit 1, where they count as attempts. I recommend the wide one; it is `s2load`'s own definition.
3. **A, no bound on refusals.** Each resume runs the check once and halts again if refused. Option: halt for good after some number. I recommend no bound, since each resume is your act.
4. **A, the attempt's entry.** It stays in `monitor[].attempts`, closed, with `record: false` and `refused: true`. Any reader that counts closed attempts without a record (CN, CG) must skip `refused`.
5. **A, when the sources really moved.** The resume itself is refused by the existing rule until the checkout is put back or `--allow-source-change` is given.
6. **A, owed outside my files.** The one sentence in M1; the design's exit-code text. `run.json` `rules.monitor` and `monitor2.rules()["texts"]` gain the key `refusal`.
7. **B, gate Z2 (c).** A fifth negative control, "tail held still", would hold this rule at the gate. It changes the gate's definition, so I did not add it to `DEGRADED`. I recommend adding it with the next round.
8. **B, how the ruling is read.**
   - "The reference's amplitude is read" means the tail term is read: at or over 0.02 rad in the window.
   - The limit is without slack: exactly 0.02 rad passes.
   - An own amplitude that cannot be computed fails.
   - The amplitude is that of the same fit that gives the phase.

Wording I settled:
- The exit code is 8. Codes 0 to 5 and 130 are taken by the trainer, monitor and pool; the tests' stubs use 6, 7 and 9.
- The names `EXIT_REFUSAL`, `MonitorRefusal`, `TAIL_AMP_MIN_RAD` (its own constant, not `NOT_READ_RAD`) and `LEAD_RULING`.
- The halt and rule texts.

Not done: nothing on the real env for A (stub env, real `monitor2` CLI with stub episodes); the slow tests; the full suite.

Scratch files are outside the repository, under `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/ct_mut` and `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/ck`.
