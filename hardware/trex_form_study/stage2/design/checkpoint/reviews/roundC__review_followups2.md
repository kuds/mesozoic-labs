<!-- Working paper, 16 Nov 2026: the independent review of those two follow-ups. Nothing here is a rule. -->

INDEPENDENT REVIEW OF THE TWO CODE FOLLOW-UPS (16 Nov 2026)

Simulation tooling only. Nothing here is a result on hardware. I changed no tracked file and committed nothing. `git status --porcelain` is empty in the main checkout and in both worktrees; heads are `ffecc2c` (working branch), `2fca27b` (`wt/wpc-ct`), `5c9fe63` (`wt/wpc-ck`).

## Verdicts
- **A, the trainer and monitor (`wt/wpc-ct`): bring in as it stands.** No blocker, no major defect. Three minor points for the next change to these files.
- **B, the look reading (`wt/wpc-ck`): bring in as it stands.** No defect found.

## Findings, most serious first (all minor)

1. **A: the refusal's halt text is wrong in one state.** `/home/user/mesozoic-wt/wpc-ct/hardware/trex_form_study/designlab/stage2/train2.py` lines 1301-1305, and `RULES["refusal"]` in `monitor2.py` lines 177-186, say "a resume runs the check again".
   - Evidence: my scratch test put two never-closed attempts at one check, then a refusal on the `--retry-check` attempt. A plain resume then halted with "never closed"; only a resume with `--retry-check` ran the check.
   - The exit code is 5 both times and the next halt names the parameter, so nothing is lost.
   - Fix: when the attempt is `after_halt`, end the message with "a resume with --retry-check runs the check again".
2. **A: no test in the package kills the trainer around a refusal.** The package's tests refuse twice and then crash once, but never SIGKILL between the monitor's exit and the attempt's closing. My three scratch tests do and pass (below). Fix: take one of them in at the next change.
3. **A: a checkpoint record that parses but has the wrong shape is no refusal.** `monitor2.py` lines 703-704: a `files` value that is a list or a string gives an `AttributeError`, so exit 1 and a counted attempt. Read from the code, not run. A cut write cannot produce this. Fix: test the shapes and raise `MonitorRefusal("checkpoint", ...)`.

## A: what I checked and found right
- **The ruling and no more.**
  - The monitor's CLI leaves with 8 for the three named cases and keeps 1 for every other fault.
  - The trainer closes the attempt with `exit_code` 8, `refused: true` and `refusal: {kind, reason}`, then raises `Halt` (code 5). `run.json` `halt` and `progress.json` `halt` quote the monitor's reason.
  - The count of closed attempts without a record skips `refused` (line 1278). The count of never-closed attempts is untouched.
  - The one addition beyond the ruling is the note file `monitor_<n>.refused.json` / `r0l.refused.json`. The builder lists it for the lead.
  - `monitor2.rules()["texts"]` is not part of the run's settings or their hash.
- **The real seam.** `python -m designlab.stage2.train2 monitor run|baseline ... --sources 000...` on a scratch job directory left with 8 both times. With the right hash and no checkpoint record it left with 8. With no `run.json` it left with 1, and that call had taken the earlier note away. No episode ran.
- **Each part has a test that fails without it.** On a scratch copy (git archive of the branch, outside the repository) I ran nine mutations. Each failed at least one of the three new tests:
  - the monitor leaves with 1;
  - the trainer does not halt;
  - a refused attempt is counted;
  - each of the three kinds raised as a plain `MonitorError`;
  - the note is not taken away;
  - the baseline's halt does not read the refusal;
  - no `refused` mark on the attempt.
- **Kill and resume.** I read `go`, `_begin`, `_rollback`, `_end_children`, `check`, `_learn` and `_finish`, and ran three scratch tests with a SIGKILL of the trainer process on the stub env. All passed.
  - Killed after the monitor left with 8 and before the attempt was closed: the attempt stays never closed, an interruption. The resume ran the check again. The run ended bit for bit as the uninterrupted one, checkpoints 1024 and 1536 and `updates.jsonl`.
  - Killed after the refused attempt was written and before the halt was recorded: `halt` is null and the status is "running". The resume ran the check again, was refused again and halted with 5. The next resume finished, bit for bit as the uninterrupted run.
  - Two such kills at one check: the halt for two attempts never closed, as rule M1 says. A refusal under `--retry-check` halted again without being counted. The next `--retry-check` finished, bit for bit.
- **By reading only.**
  - A refusal at the final check takes the same path.
  - A checkpoint that is not intact is skipped by the resume, and its check runs on the checkpoint saved again.
  - Moved sources make the resume itself refuse with 4 unless `--allow-source-change` is given.
- **Nothing else changed.** `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat` lists `monitor2.py` (108), `train2.py` (44), `test_stage2_monitor2.py` (69), `test_stage2_train2.py` (114): 4 files, 309 insertions, 26 deletions. The test changes are additions only. No other module in the tree globs the monitor's files.

## B: what I checked and found right
- **The ruling and no more.** In `compare`, where the tail term is read, it needs the phase in its band and the robot's own fitted amplitude not under 0.02 rad. Where the reference leaves the tail unread, nothing changes. `compare` gains three keys (`tail_amp_rad`, `tail_amp_side`, `tail_amp_min_rad`). No other function changed.
- **Every ladder rung, my own run.** I played all 73 rungs of `stage2/look/metrics.json` kinematically (no physics) and read them with the branch's module and with the working branch's module side by side.
  - The readings are equal on every rung. `compare` differs only by the three new keys.
  - The tail term is read and passes against itself on 73 of 73.
  - Smallest fitted amplitude: 0.119983 rad (60-N-T-10, v040, V0). Largest: 0.125268 rad.
  - The six V6 rungs pass gate Z2 (b)'s limits and all four terms against themselves. The 60 cm fr05 rung leaves the toe-off unread.
  - Gate Z2 (c)'s controls are ok on all six, with the same failed terms as before the change.
  - 54 rungs (V0 to V4L) fail the head term against themselves, identically in the old module.
- **Each part has a test that fails without it.** Seven mutations on a scratch copy; the unmutated copy gave "70 passed, 1 skipped". Each of these failed at least one test:
  - the limit taken out (4 failures, as the builder says);
  - `<=` in place of `<`;
  - the limit applied where the reference is not read;
  - 0.01 rad and 0.03 rad;
  - an uncomputable amplitude passing;
  - the reference's amplitude compared in place of the robot's.
- **The package's slow test**, run once (`STAGE2_SLOW=1`, one closed-loop episode on the S1 view, no record): "1 passed, 70 deselected in 4.04s". It asserts the tail term passes against the playback.
- **Nothing else changed.** The stat lists `lookread.py` (46) and `test_stage2_lookread.py` (155): 2 files, 174 insertions, 27 deletions. One test is replaced because it pinned the old reading.

## Exact test summary lines (my runs, through `bash tools/lowprio.sh`, the standing env, exit 0 each)
- `tests/test_stage2_monitor2.py` (in `wt/wpc-ct`): `23 passed in 20.77s`
- `tests/test_stage2_train2.py` (in `wt/wpc-ct`): `48 passed, 3 skipped, 2 warnings in 211.84s (0:03:31)`
- `tests/test_stage2_lookread.py` (in `wt/wpc-ck`): `70 passed, 1 skipped in 9.67s`
- On a scratch copy holding both changes (not a git checkout): `23 passed in 23.13s`; `48 passed, 3 skipped, 2 warnings in 210.18s (0:03:30)`; `70 passed, 1 skipped in 10.02s`.

Record-key hashes, in the main checkout and in both worktrees: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.

## Commits not in the working branch

**`wt/wpc-ct` (2 commits, both end with the two lines; no AI model named elsewhere):**
- `57b5bf3` the monitor's exit code
- `2fca27b` the trainer halts and does not count

**`wt/wpc-ck` (15 commits):**
- `cc8eec5` "Merge branch 'wt/wp-r' into wt/wpc-ck": **lacks the two lines**
- `3182a6e` "Merge branch 'wt/wp-b' into wt/wpc-ck": **lacks the two lines**
- These 13 end with the two lines: `6d6d38d`, `11a23b4`, `8fbd47e` (merge), `0ccabf2`, `62751fc`, `01fccec`, `eb477ca` (merge), `8b8cf5a`, `2a64f58`, `5b4e304`, `9caa811`, `c9b0bed` (merge; its tree equals the working branch's), `5c9fe63`.
- I read the full messages of `57b5bf3`, `2fca27b`, `c9b0bed` and `5c9fe63` for an AI model named elsewhere: none.

## What to cherry-pick, in order
1. `57b5bf3`
2. `2fca27b`
3. `5c9fe63`

- Do not merge `wt/wpc-ck`: its history holds the two unattributed merge commits. `c9b0bed` is not needed.
- A fast-forward of the working branch to `2fca27b` would equal picks 1 and 2 while the working branch stays at `ffecc2c`.
- I computed the trees without touching a ref: picks 1 and 2 onto `ffecc2c` give `wt/wpc-ct`'s tree, and pick 3 onto `ffecc2c` gives `5c9fe63`'s tree. The three commits touch disjoint files.

## For the lead (the builder's points, with my recommendation)
1. **The note file.** Keep it. It adds `monitor_<n>.refused.json` and `r0l.refused.json` to the design's list of a job's files.
2. **"Not intact", the wide reading.** Keep it. `s2load.pair_problem` defines intact the same way: a missing or unreadable record, then the files.
3. **No bound on refusals.** Keep it; each resume is your act.
4. **The attempt's entry stays with `refused: true`.** Keep it. The notebook's and the gates' briefs must say that a count of closed attempts skips `refused`.
5. **Owed outside the files.** The M1 sentence; exit code 8 in the design's exit-code text.
6. **B, a fifth negative control "tail held still" in gate Z2 (c).** It changes the gate's definition, so it is yours. I recommend adding it with the next round, before code state B.
7. **B, one more edit is coming (my point).** `LEAD_RULING` and the docstring say the plan "does not hold yet" the sentence. When the draft takes it in, that text must go, and `lookread.py`'s hash changes again. Do it with point 6, before code state B.

Wording I settled: none; I changed no file.

## What I did not check
- A on the real env. Everything ran on the stub env with the real `monitor2` CLI.
- The trainer's three slow S1 tests and the full suite.
- A kill of the monitor child itself, a SIGTERM or Ctrl-C, and a note that cannot be written.
- The tests at `57b5bf3` alone, without `2fca27b`.
- A closed-loop tail amplitude on any set but the one S1 episode of the package's slow test.
- The tests on a real git checkout holding both changes (the combined run was a scratch copy).
- The messages of the 11 older attributed `wt/wpc-ck` commits for an AI model named elsewhere.

My scratch tests ran the stub trainer, as the package's own tests do; no plant, no reference set, no fit.

Everything is in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_followups2/`:
- `train2_ct.log`
- `mutate_ct.py`, `mutate_ct.log`
- `mutate_ck.py`, `mutate_ck.log`
- `ladder_check.py`, `ladder_check.log`
- `both.log`
- `ct_copy/hardware/trex_form_study/tests/test_stage2_train2.py` (the three `test_review_*` tests at its end)
